# PencilFFTs pencil-axis test — run under `mpiexec -n 2` (launched from runtests.jl).
# Splits one grid across ranks (transpose-based distributed FFT) and verifies the distributed
# KE spectral flux equals the serial FFTBackend result on the same field. Exits nonzero on failure.

using MPI: MPI
using PencilFFTs: PencilFFTs, allocate_input
using PencilArrays: PencilArrays, range_local
using FFTW: FFTW
using Random: Random
using FlowInvariantTransfer: FlowInvariantTransfer as FET

MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)

N = 16; L = 2π; nd = 2
ks = FET.wavenumber_grid((N, N), (L, L))
kx = [ks[1][i] for i in 1:N, j in 1:N]
ky = [ks[2][j] for i in 1:N, j in 1:N]

# Incompressible 2D field (same on every rank); package convention û = fft(u)/Np,
# so the physical velocity is u = bfft(û) = Np·ifft(û).
ψh = FFTW.fft(randn(Random.MersenneTwister(123), N, N)) ./ N^2
ûx =  im .* ky .* ψh
ûy = -im .* kx .* ψh
û  = cat(ûx, ûy; dims = 3)
Uphys = (real.(FFTW.bfft(ûx)), real.(FFTW.bfft(ûy)))

binning = FET.LinearBinning(2π / L)
ref = FET.calculate_spectral_flux(û, ks; binning = binning,
        dealiasing = FET.OrszagTwoThirds(), spectral = FET.FFTBackend())

# Distribute the physical field into pencils: each rank fills the portion it owns.
plan = FET.build_pencil_plan((N, N), comm)
upen = ntuple(nd) do c
    a  = allocate_input(plan)
    rl = range_local(a)                # global physical indices owned by this rank
    for I in CartesianIndices(a)
        gI = CartesianIndex(ntuple(d -> rl[d][I[d]], nd))
        a[I] = Uphys[c][gI]
    end
    a
end
res = FET.pencil_spectral_flux(upen, plan, ks; comm = comm, binning = binning,
        dealiasing = FET.OrszagTwoThirds())

failures = 0
if rank == 0
    scaleT = maximum(abs, ref.transfer_spectrum) + eps()
    scaleF = maximum(abs, ref.flux) + eps()
    eT = maximum(abs, res.transfer_spectrum .- ref.transfer_spectrum)
    eF = maximum(abs, res.flux .- ref.flux)
    println("pencil vs serial: relΔT=", eT / scaleT, " relΔΠ=", eF / scaleF)
    (scaleT > 1e-8)            || (println("FAIL: reference transfer is ~0 (non-meaningful test)"); global failures += 1)
    (eT / scaleT < 1e-9)      || (println("FAIL: transfer_spectrum mismatch"); global failures += 1)
    (eF / scaleF < 1e-9)      || (println("FAIL: flux mismatch"); global failures += 1)
    println(failures == 0 ? "PENCIL_OK" : "PENCIL_FAILED ($failures)")
end
MPI.Finalize()
failures == 0 || exit(1)
