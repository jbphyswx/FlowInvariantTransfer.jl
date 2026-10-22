# ---------------------------------------------------------------------------
# Batch entries: a batch reuses one workspace across a worker's snapshots and returns what the
# per-snapshot calls return.
#
# Every comparison asserts the reference carries signal first: 2/3 dealiasing on a small 3-D grid
# retains `|m_d| ≤ ⌊(n−1)/3⌋`, which on 8³ is `|m_d| ≤ 2` and can leave a reference that is only
# round-off.
# ---------------------------------------------------------------------------

using Test: Test
using Random: Random
using FFTW: FFTW
using NonuniformFFTs: NonuniformFFTs
using FlowTransformBindings: FlowTransformBindings as FTB
using OhMyThreads: OhMyThreads
using Distributed: Distributed
using KernelAbstractions: KernelAbstractions as KA
using FlowInvariantTransfer: FlowInvariantTransfer as FIT
using ComputationalBackends: ComputationalBackends

Distributed.nprocs() == 1 && Distributed.addprocs(2)
Distributed.@everywhere using FlowInvariantTransfer: FlowInvariantTransfer as FIT
Distributed.@everywhere using FFTW: FFTW
Distributed.@everywhere using NonuniformFFTs: NonuniformFFTs

const _HOST_EXECS = (("serial", ComputationalBackends.SerialBackend()),
                     ("threaded", ComputationalBackends.ThreadedBackend()),
                     ("distributed", ComputationalBackends.DistributedBackend()))
const _BATCH_EXECS = (_HOST_EXECS..., ("device", ComputationalBackends.GPUBackend(KA.CPU())))

# Serial and threaded batches run the per-snapshot code in this process and agree bit for bit. A worker
# process plans its own FFTs (`FFTW.MEASURE` times candidates per process), and the device path runs
# other kernels, so those agree to round-off.
_same(nm, a, b) = nm in ("serial", "threaded") ? a == b : isapprox(a, b; rtol = 1e-12)

Test.@testset "batch entries reproduce the per-snapshot calls" begin
    b = FIT.Types.LinearBinning(1.0)

    Test.@testset "compressible — $nm" for (nm, exec) in _BATCH_EXECS
        N = 12; nd = 2; nsnap = 5
        ks = FIT.Utils.wavenumber_grid((N, N), (2π, 2π))
        Random.seed!(6)
        us = [randn(ComplexF64, N, N, nd) for _ in 1:nsnap]
        ρs = [reshape(randn(ComplexF64, N, N), N, N, 1) for _ in 1:nsnap]
        ref = [FIT.calculate_compressible_flux(us[i], ρs[i], ks; binning = b) for i in 1:nsnap]
        got = FIT.calculate_compressible_flux_batch(us, ρs, ks; binning = b, execution = exec)
        Test.@test maximum(maximum(abs, ref[i].transfer_spectrum) for i in 1:nsnap) > 1e-8
        Test.@test all(_same(nm, got[i].transfer_spectrum, ref[i].transfer_spectrum) for i in 1:nsnap)
    end

    Test.@testset "partial fluxes — $nm" for (nm, exec) in _BATCH_EXECS
        M3 = 16                       # the 2/3-retained band of 16³ carries transfer
        ks3 = FIT.Utils.wavenumber_grid((M3, M3, M3), (2π, 2π, 2π))
        Random.seed!(17)
        u3 = [randn(ComplexF64, M3, M3, M3, 3) for _ in 1:3]
        ref = [FIT.calculate_partial_fluxes(v, ks3; binning = b) for v in u3]
        got = FIT.calculate_partial_fluxes_batch(u3, ks3; binning = b, execution = exec)
        Test.@test maximum(maximum(abs, ref[i].total.flux) for i in eachindex(u3)) > 1e-8
        Test.@test all(_same(nm, got[i].total.flux, ref[i].total.flux) for i in eachindex(u3))
    end

    Test.@testset "scattered coarse-graining — $nm" for (nm, exec) in _HOST_EXECS
        Ln = 2π; M = 600; ms = (12, 12); ℓ = 0.8
        Random.seed!(12)
        cx = Ln .* rand(M); cy = Ln .* rand(M)
        vf = [(sin.(cx .+ 0.1k) .* cos.(cy), cos.(cx) .* sin.(cy .+ 0.1k)) for k in 1:4]
        sp = FTB.NonuniformFFTsBackend(); filt = FIT.Types.GaussianFilter()
        ref = [FIT.nufft_coarse_graining_flux(v, (cx, cy), ℓ, filt, ms; transform = sp, Ls = (Ln, Ln))
               for v in vf]
        got = FIT.nufft_coarse_graining_flux_batch(vf, (cx, cy), ℓ, filt, ms;
                                                   transform = sp, Ls = (Ln, Ln), execution = exec)
        scale = maximum(maximum(abs, ref[i].flux_field) for i in eachindex(vf))
        Test.@test scale > 1e-8
        Test.@test all(_same(nm, got[i].flux_field, ref[i].flux_field) for i in eachindex(vf))
    end
end

# ---------------------------------------------------------------------------
# `calculate_helical_partial_fluxes!` with a workspace reused across two fields gives what fresh calls give.
# ---------------------------------------------------------------------------
Test.@testset "calculate_helical_partial_fluxes! (in-place)" begin
    M = 16
    ks = FIT.Utils.wavenumber_grid((M, M, M), (2π, 2π, 2π))
    Random.seed!(23)
    û = randn(ComplexF64, M, M, M, 3)
    b = FIT.Types.LinearBinning(1.0)

    ws = FIT.Workspaces.NonlinearTermWorkspace(û, ks)
    FIT.calculate_helical_partial_fluxes!(ws, û, ks; binning = b)

    Random.seed!(24)
    v̂ = randn(ComplexF64, M, M, M, 3)
    ref = FIT.calculate_helical_partial_fluxes(v̂, ks; binning = b)
    Test.@test maximum(abs, ref.total.flux) > 1e-8                  # the reference carries signal
    Test.@test FIT.calculate_helical_partial_fluxes!(ws, v̂, ks; binning = b).total.flux == ref.total.flux
end
