# MPI batch-axis test — run under `mpiexec -n 2` (launched from runtests.jl).
# Distributes independent snapshots across ranks; verifies gather/collate ordering and
# the :sum / :mean reductions against a per-rank serial reference. Exits nonzero on failure.

using MPI: MPI
using FFTW: FFTW
using Random: Random
using FlowInvariantTransfer: FlowInvariantTransfer as FET

MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
nproc = MPI.Comm_size(comm)

N = 12; L = 2π
ks = FET.wavenumber_grid((N, N), (L, L))
kx = [ks[1][i] for i in 1:N, j in 1:N]
ky = [ks[2][j] for i in 1:N, j in 1:N]
binning = FET.LinearBinning(2π / L)

# Deterministic incompressible snapshots (identical list on every rank).
function snapshot(seed)
    rng = Random.MersenneTwister(seed)
    ψh = FFTW.fft(randn(rng, N, N)) ./ N^2
    cat(im .* ky .* ψh, -im .* kx .* ψh; dims = 3)
end
snapshots = [snapshot(s) for s in 1:5]

# Per-snapshot diagnostic returned by the batch map.
f(û) = FET.calculate_spectral_flux(û, ks; binning = binning, spectral = FET.FFTBackend()).flux

# Serial reference (cheap; every rank can compute it for assertions).
ref = [f(s) for s in snapshots]

failures = 0
check(cond, msg) = (cond || (rank == 0 && println("FAIL: ", msg)); cond || (global failures += 1))

# 1. gather/collate — results in original order on every rank.
gathered = FET.mpi_batch_map(f, snapshots; comm = comm)
check(length(gathered) == length(snapshots), "gather length")
check(all(isapprox.(gathered, ref; rtol = 1e-10, atol = 1e-12)), "gather values/order")

# 2. mean / sum reductions.
m = FET.mpi_batch_map(f, snapshots; comm = comm, reduce = :mean)
check(isapprox(m, sum(ref) ./ length(ref); rtol = 1e-10, atol = 1e-12), "mean reduction")
s = FET.mpi_batch_map(f, snapshots; comm = comm, reduce = :sum)
check(isapprox(s, sum(ref); rtol = 1e-10, atol = 1e-12), "sum reduction")

# 3. the work really was split (sanity: more ranks than 1 in this launch).
check(nproc >= 2, "expected ≥2 ranks")

MPI.Barrier(comm)
if rank == 0
    println(failures == 0 ? "BATCH_OK" : "BATCH_FAILED ($failures)")
end
MPI.Finalize()
failures == 0 || exit(1)
