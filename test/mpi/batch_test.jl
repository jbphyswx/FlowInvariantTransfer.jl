# MPI batch-axis test — run under `mpiexec -n 2` (launched from runtests.jl).
# Distributes independent snapshots across ranks; verifies gather/collate ordering and
# the :sum / :mean reductions against a per-rank serial reference. Exits nonzero on failure.

using MPI: MPI
using FFTW: FFTW
using Random: Random
using NonuniformFFTs: NonuniformFFTs
using CoarseGrainingEnergyFluxes: CoarseGrainingEnergyFluxes
using HelmholtzDecomposition: HelmholtzDecomposition
using FlowInvariantTransfer: FlowInvariantTransfer as FIT
using FlowTransformBindings: FlowTransformBindings as FTB
using ComputationalBackends: ComputationalBackends
using SpectralBackends: SpectralBackends

MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
nproc = MPI.Comm_size(comm)

N = 12; L = 2π
ks = FIT.Utils.wavenumber_grid((N, N), (L, L))
kx = [ks[1][i] for i in 1:N, j in 1:N]
ky = [ks[2][j] for i in 1:N, j in 1:N]
binning = FIT.Types.LinearBinning(2π / L)

# Deterministic incompressible snapshots (identical list on every rank).
function snapshot(seed)
    rng = Random.MersenneTwister(seed)
    ψh = FFTW.fft(randn(rng, N, N)) ./ N^2
    cat(im .* ky .* ψh, -im .* kx .* ψh; dims = 3)
end
snapshots = [snapshot(s) for s in 1:5]

# Per-snapshot diagnostic returned by the batch map.
f(û) = FIT.SpectralFlux.calculate_spectral_flux(û, ks; binning = binning, transform = SpectralBackends.FFTSpectralBackend()).flux

# Serial reference (cheap; every rank can compute it for assertions).
ref = [f(s) for s in snapshots]

failures = 0
check(cond, msg) = (cond || (rank == 0 && println("FAIL: ", msg)); cond || (global failures += 1))

# 1. gather/collate — results in original order on every rank.
gathered = FIT.mpi_batch_map(f, snapshots; comm = comm)
check(length(gathered) == length(snapshots), "gather length")
check(all(isapprox.(gathered, ref; rtol = 1e-10, atol = 1e-12)), "gather values/order")

# 2. mean / sum reductions.
m = FIT.mpi_batch_map(f, snapshots; comm = comm, reduce = :mean)
check(isapprox(m, sum(ref) ./ length(ref); rtol = 1e-10, atol = 1e-12), "mean reduction")
s = FIT.mpi_batch_map(f, snapshots; comm = comm, reduce = :sum)
check(isapprox(s, sum(ref); rtol = 1e-10, atol = 1e-12), "sum reduction")

# 3. The launch has more than one rank.
check(nproc >= 2, "expected ≥2 ranks")

# 4. Every `*_batch` entry on `MPIBackend` returns the serial batch on every rank: each rank runs its
#    round-robin share of the snapshots through the same driver. Five snapshots on two ranks leave the
#    ranks unequal shares. A snapshot from another rank went through that rank's own FFT plans
#    (`FFTW.MEASURE` plans per process), so the comparison is to round-off.
MPIb = ComputationalBackends.MPIBackend(; comm = comm)
Ser = ComputationalBackends.SerialBackend()
FFT = SpectralBackends.FFTSpectralBackend()
same(a, b) = length(a) == length(b) && all(isapprox(x, y; rtol = 1e-12) for (x, y) in zip(a, b))

sfS = FIT.SpectralFlux.calculate_spectral_flux_batch(snapshots, ks; binning = binning, transform = FFT, execution = Ser)
sfM = FIT.SpectralFlux.calculate_spectral_flux_batch(snapshots, ks; binning = binning, transform = FFT, execution = MPIb)
check(maximum(abs, sfS[1].transfer_spectrum) > 1e-12, "spectral-flux reference carries transfer")
check(same((r.transfer_spectrum for r in sfM), (r.transfer_spectrum for r in sfS)), "spectral-flux batch")

ssS = FIT.ShellToShellTransfer.calculate_shell_to_shell_transfer_batch(snapshots, ks; binning = binning, transform = FFT, execution = Ser)
ssM = FIT.ShellToShellTransfer.calculate_shell_to_shell_transfer_batch(snapshots, ks; binning = binning, transform = FFT, execution = MPIb)
check(same((r.transfer_matrix for r in ssM), (r.transfer_matrix for r in ssS)), "shell-to-shell batch")

bands = FIT.Types.SmoothBands([1.0, 2.0, 3.0]; logwidth = 0.5)
bbS = FIT.BandTransfer.calculate_band_to_band_transfer_batch(snapshots, ks; bands = bands, transform = FFT, execution = Ser)
bbM = FIT.BandTransfer.calculate_band_to_band_transfer_batch(snapshots, ks; bands = bands, transform = FFT, execution = MPIb)
check(same((r.transfer_matrix for r in bbM), (r.transfer_matrix for r in bbS)), "band-to-band batch")

pfkw = (; binning = binning, decomposition = FIT.Types.HelmholtzDecomposition(), transform = FFT)
pfS = FIT.calculate_partial_fluxes_batch(snapshots, ks; pfkw..., execution = Ser)
pfM = FIT.calculate_partial_fluxes_batch(snapshots, ks; pfkw..., execution = MPIb)
check(same((r.total.flux for r in pfM), (r.total.flux for r in pfS)), "partial-flux batch")

ρs = [reshape(FFTW.fft(randn(Random.MersenneTwister(100 + s), N, N)) ./ N^2, N, N, 1) for s in 1:5]
cS = FIT.calculate_compressible_flux_batch(snapshots, ρs, ks; binning = binning, execution = Ser)
cM = FIT.calculate_compressible_flux_batch(snapshots, ρs, ks; binning = binning, execution = MPIb)
check(same((r.transfer_spectrum for r in cM), (r.transfer_spectrum for r in cS)), "compressible batch")

xs = collect(range(0, L; length = N + 1)[1:N])
ufields = [(randn(Random.MersenneTwister(200 + s), N, N), randn(Random.MersenneTwister(300 + s), N, N)) for s in 1:5]
filt = FIT.Types.GaussianFilter()
cgS = FIT.CoarseGrainingFlux.calculate_coarse_graining_flux_batch(ufields, (xs, xs), 0.5, filt; execution = Ser)
cgM = FIT.CoarseGrainingFlux.calculate_coarse_graining_flux_batch(ufields, (xs, xs), 0.5, filt; execution = MPIb)
check(same((r.flux_field for r in cgM), (r.flux_field for r in cgS)), "coarse-graining batch")

Mpts = 300
cx = L .* rand(Random.MersenneTwister(400), Mpts); cy = L .* rand(Random.MersenneTwister(401), Mpts)
vf = [(sin.(cx .+ 0.1s) .* cos.(cy), cos.(cx) .* sin.(cy .+ 0.1s)) for s in 1:5]
nkw = (; transform = FTB.NonuniformFFTsBackend(), Ls = (L, L))
ncS = FIT.nufft_coarse_graining_flux_batch(vf, (cx, cy), 0.8, filt, (12, 12); nkw..., execution = Ser)
ncM = FIT.nufft_coarse_graining_flux_batch(vf, (cx, cy), 0.8, filt, (12, 12); nkw..., execution = MPIb)
check(same((r.flux_field for r in ncM), (r.flux_field for r in ncS)), "scattered coarse-graining batch")

MPI.Barrier(comm)
if rank == 0
    println(failures == 0 ? "BATCH_OK" : "BATCH_FAILED ($failures)")
end
MPI.Finalize()
failures == 0 || exit(1)
