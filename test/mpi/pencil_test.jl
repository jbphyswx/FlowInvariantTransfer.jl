# PencilFFTs pencil-axis test — run under `mpiexec -n 2` (launched from runtests.jl).
# Splits one grid across ranks (transpose-based distributed FFT) and verifies the distributed
# KE spectral flux equals the serial SpectralBackends.FFTSpectralBackend result on the same field. Exits nonzero on failure.

using MPI: MPI
using PencilFFTs: PencilFFTs
using PencilArrays: PencilArrays
using FFTW: FFTW
using Random: Random
using KernelAbstractions: KernelAbstractions as KA
using FlowInvariantTransfer: FlowInvariantTransfer as FIT
using ComputationalBackends: ComputationalBackends
using SpectralBackends: SpectralBackends

MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)

N = 16; L = 2π; nd = 2
ks = FIT.Utils.wavenumber_grid((N, N), (L, L))
ksh = FIT.Utils.wavenumber_grid((N, N), (L, L); real = true)   # pencil layout: real input, half spectrum
kx = [ks[1][i] for i in 1:N, j in 1:N]
ky = [ks[2][j] for i in 1:N, j in 1:N]

# Incompressible 2D field (same on every rank); package convention û = fft(u)/Np,
# so the physical velocity is u = bfft(û) = Np·ifft(û).
ψh = FFTW.fft(randn(Random.MersenneTwister(123), N, N)) ./ N^2
ûx =  im .* ky .* ψh
ûy = -im .* kx .* ψh
û  = cat(ûx, ûy; dims = 3)
Uphys = (real.(FFTW.bfft(ûx)), real.(FFTW.bfft(ûy)))

binning = FIT.Types.LinearBinning(2π / L)
ref = FIT.SpectralFlux.calculate_spectral_flux(û, ks; binning = binning,
        dealiasing = FIT.Types.OrszagTwoThirds(), spectral = SpectralBackends.FFTSpectralBackend())

# Distribute the physical field into pencils: each rank fills the portion it owns.
plan = FIT.build_pencil_plan((N, N), comm)
upen = ntuple(nd) do c
    a  = PencilFFTs.allocate_input(plan)
    rl = PencilArrays.range_local(a)                # global physical indices owned by this rank
    for I in CartesianIndices(a)
        gI = CartesianIndex(ntuple(d -> rl[d][I[d]], nd))
        a[I] = Uphys[c][gI]
    end
    a
end
res = FIT.pencil_spectral_flux(upen, plan, ksh; comm = comm, binning = binning,
        dealiasing = FIT.Types.OrszagTwoThirds())

# Composable execution: MPIBackend(GPUBackend(KA.CPU())) unwraps to a per-rank GPUBackend, routing the
# local shell reduction through the atomic device scatter-add instead of the host scalar loop. Verified
# on KA.CPU (plain-Array pencils); the identical code path runs on a CuArray-backed pencil (multi-GPU).
resG = FIT.pencil_spectral_flux(upen, plan, ksh; comm = comm, binning = binning,
        dealiasing = FIT.Types.OrszagTwoThirds(), execution = ComputationalBackends.MPIBackend(ComputationalBackends.GPUBackend(KA.CPU())))

# Enstrophy (2D) — the pencil path now supports every invariant; reuse the same distributed field.
refZ = FIT.SpectralFlux.calculate_spectral_flux(û, ks; binning = binning, dealiasing = FIT.Types.OrszagTwoThirds(),
        spectral = SpectralBackends.FFTSpectralBackend(), invariant = FIT.Types.Enstrophy())
resZ = FIT.pencil_spectral_flux(upen, plan, ksh; comm = comm, binning = binning,
        dealiasing = FIT.Types.OrszagTwoThirds(), invariant = FIT.Types.Enstrophy())

# Helicity (3D) — a divergence-free field u = ∇×A on a 3D grid split across ranks. N=16: at N=8 the
# 2/3-dealiased retained band is too small to form helicity-transferring triads (T_H ≈ 0, degenerate).
N3 = 16; ks3 = FIT.Utils.wavenumber_grid((N3, N3, N3), (L, L, L))
ks3h = FIT.Utils.wavenumber_grid((N3, N3, N3), (L, L, L); real = true)
kx3 = [ks3[1][i] for i in 1:N3, j in 1:N3, k in 1:N3]
ky3 = [ks3[2][j] for i in 1:N3, j in 1:N3, k in 1:N3]
kz3 = [ks3[3][k] for i in 1:N3, j in 1:N3, k in 1:N3]
rng3 = Random.MersenneTwister(1)
Âx = FFTW.fft(randn(rng3, N3, N3, N3)) ./ N3^3
Ây = FFTW.fft(randn(rng3, N3, N3, N3)) ./ N3^3
Âz = FFTW.fft(randn(rng3, N3, N3, N3)) ./ N3^3
û3 = cat(im .* (ky3 .* Âz .- kz3 .* Ây), im .* (kz3 .* Âx .- kx3 .* Âz), im .* (kx3 .* Ây .- ky3 .* Âx); dims = 4)
U3 = ntuple(3) do c; real.(FFTW.bfft(û3[:, :, :, c])); end
ref3 = FIT.SpectralFlux.calculate_spectral_flux(û3, ks3; binning = binning, dealiasing = FIT.Types.OrszagTwoThirds(),
        spectral = SpectralBackends.FFTSpectralBackend(), invariant = FIT.Types.Helicity())
plan3 = FIT.build_pencil_plan((N3, N3, N3), comm)
upen3 = ntuple(3) do c
    a = PencilFFTs.allocate_input(plan3); rl = PencilArrays.range_local(a)
    for I in CartesianIndices(a)
        gI = CartesianIndex(ntuple(d -> rl[d][I[d]], 3))
        a[I] = U3[c][gI]
    end
    a
end
res3 = FIT.pencil_spectral_flux(upen3, plan3, ks3h; comm = comm, binning = binning,
        dealiasing = FIT.Types.OrszagTwoThirds(), invariant = FIT.Types.Helicity())

# Anisotropic geometry (3D): cylindrical k_⊥ = √(kx²+ky²) shells — exercises the geometry generalization.
refP = FIT.SpectralFlux.calculate_spectral_flux(û3, ks3; binning = binning, dealiasing = FIT.Types.OrszagTwoThirds(),
        spectral = SpectralBackends.FFTSpectralBackend(), geometry = FIT.Types.PerpendicularShells())
resP = FIT.pencil_spectral_flux(upen3, plan3, ks3h; comm = comm, binning = binning,
        dealiasing = FIT.Types.OrszagTwoThirds(), geometry = FIT.Types.PerpendicularShells())

# Enstrophy 3D (vector vorticity + vortex stretching) through the pencil path — reuse the ∇×A field.
refZ3 = FIT.SpectralFlux.calculate_spectral_flux(û3, ks3; binning = binning, dealiasing = FIT.Types.OrszagTwoThirds(),
        spectral = SpectralBackends.FFTSpectralBackend(), invariant = FIT.Types.Enstrophy())
resZ3 = FIT.pencil_spectral_flux(upen3, plan3, ks3h; comm = comm, binning = binning,
        dealiasing = FIT.Types.OrszagTwoThirds(), invariant = FIT.Types.Enstrophy())

# 0-alloc reuse: a workspace reused across snapshots of the same distributed grid allocates only the
# small per-shell result vectors (Tglob + flux), not the O(field) Fourier grids/scratch each call.
wsK = FIT.PencilWorkspace(plan, ksh, comm; binning = binning, dealiasing = FIT.Types.OrszagTwoThirds())
resWS = FIT.pencil_spectral_flux!(wsK, upen)              # warm (all ranks; collective Allreduce)
a_reuse = @allocated FIT.pencil_spectral_flux!(wsK, upen)

failures = 0
if rank == 0
    println("pencil ! reuse alloc = ", a_reuse, " bytes")
    (maximum(abs, resWS.transfer_spectrum .- ref.transfer_spectrum) < 1e-9 * (maximum(abs, ref.transfer_spectrum) + eps())) ||
        (println("FAIL: workspace ! transfer mismatch vs serial"); global failures += 1)
    (a_reuse < 8192) || (println("FAIL: pencil ! reuse alloc too high ($a_reuse bytes)"); global failures += 1)
    for (name, r, rf) in (("KE", res, ref), ("KE-gpu(KA.CPU)", resG, ref), ("enstrophy", resZ, refZ), ("enstrophy3D", resZ3, refZ3), ("helicity", res3, ref3), ("KE⊥", resP, refP))
        sT = maximum(abs, rf.transfer_spectrum) + eps()
        eT = maximum(abs, r.transfer_spectrum .- rf.transfer_spectrum) / sT
        eF = maximum(abs, r.flux .- rf.flux) / (maximum(abs, rf.flux) + eps())
        println("pencil ", name, ": relΔT=", eT, " relΔΠ=", eF)
        (sT > 1e-8) || (println("FAIL: $name reference transfer ~0"); global failures += 1)
        (eT < 1e-9)  || (println("FAIL: $name transfer_spectrum mismatch"); global failures += 1)
        (eF < 1e-9)  || (println("FAIL: $name flux mismatch"); global failures += 1)
    end
    println(failures == 0 ? "PENCIL_OK" : "PENCIL_FAILED ($failures)")
end
MPI.Finalize()
failures == 0 || exit(1)
