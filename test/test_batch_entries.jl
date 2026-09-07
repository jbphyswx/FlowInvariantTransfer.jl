# ---------------------------------------------------------------------------
# Batch entries: a batch reuses one workspace across a worker's snapshots, so it must return what the
# per-snapshot calls return, and its result vector must be concretely typed — an abstract element type
# puts a dynamic dispatch on every downstream use.
#
# Every comparison asserts the reference carries signal first: 2/3 dealiasing on a small 3-D grid
# retains `|k_d| < n/3`, which on 8³ is `|k_d| < 2` and leaves a reference that is only round-off.
# ---------------------------------------------------------------------------

using Test: Test
using Random: Random
using FFTW: FFTW
using NonuniformFFTs: NonuniformFFTs
using OhMyThreads: OhMyThreads
using FlowInvariantTransfer: FlowInvariantTransfer as FIT
using ComputationalBackends: ComputationalBackends

const _BATCH_EXECS = (("serial", ComputationalBackends.SerialBackend()),
                      ("threaded", ComputationalBackends.ThreadedBackend()))

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
        Test.@test all(got[i].transfer_spectrum == ref[i].transfer_spectrum for i in 1:nsnap)
        Test.@test isconcretetype(eltype(got))
    end

    Test.@testset "partial fluxes — $nm" for (nm, exec) in _BATCH_EXECS
        M3 = 16                       # 8³ would leave a round-off reference under 2/3 dealiasing
        ks3 = FIT.Utils.wavenumber_grid((M3, M3, M3), (2π, 2π, 2π))
        Random.seed!(17)
        u3 = [randn(ComplexF64, M3, M3, M3, 3) for _ in 1:3]
        ref = [FIT.calculate_partial_fluxes(v, ks3; binning = b) for v in u3]
        got = FIT.calculate_partial_fluxes_batch(u3, ks3; binning = b, execution = exec)
        Test.@test maximum(maximum(abs, ref[i].total.flux) for i in eachindex(u3)) > 1e-8
        Test.@test all(got[i].total.flux == ref[i].total.flux for i in eachindex(u3))
    end

    Test.@testset "scattered coarse-graining — $nm" for (nm, exec) in _BATCH_EXECS
        Ln = 2π; M = 600; ms = (12, 12); ℓ = 0.8
        Random.seed!(12)
        cx = Ln .* rand(M); cy = Ln .* rand(M)
        vf = [(sin.(cx .+ 0.1k) .* cos.(cy), cos.(cx) .* sin.(cy .+ 0.1k)) for k in 1:4]
        sp = FIT.Types.NonuniformFFTsBackend(); filt = FIT.Types.GaussianFilter()
        ref = [FIT.nufft_coarse_graining_flux(v, (cx, cy), ℓ, filt, ms; spectral = sp, Ls = (Ln, Ln))
               for v in vf]
        got = FIT.nufft_coarse_graining_flux_batch(vf, (cx, cy), ℓ, filt, ms;
                                                   spectral = sp, Ls = (Ln, Ln), execution = exec)
        scale = maximum(maximum(abs, ref[i].flux_field) for i in eachindex(vf))
        Test.@test scale > 1e-8
        # NonuniformFFTs threads its transforms off `Threads.nthreads()`, and threaded spreading is not
        # bitwise reproducible across plan instances: measured 0.0 at -t1 and ~3e-15 at -t4.
        tol = Threads.nthreads() == 1 ? 0.0 : 1e-12 * scale
        Test.@test maximum(maximum(abs, got[i].flux_field .- ref[i].flux_field) for i in eachindex(vf)) <= tol
        Test.@test isconcretetype(eltype(got))
        Test.@test got[1].flux_field !== got[2].flux_field      # each result owns its field
    end

    Test.@testset "empty batches return typed vectors" begin
        ks = FIT.Utils.wavenumber_grid((8, 8), (2π, 2π))
        e1 = FIT.calculate_compressible_flux_batch(Matrix{ComplexF64}[], Matrix{ComplexF64}[], ks; binning = b)
        Test.@test eltype(e1) <: FIT.Types.CompressibleFluxResult
        Test.@test isempty(e1)
    end
end

# ---------------------------------------------------------------------------
# `calculate_helical_partial_fluxes!` — the in-place form, which the suite otherwise never calls.
#
# Gated on the identity that defines the decomposition: the eight helical channels partition the
# field, so their fluxes sum to the undecomposed flux at every shell. The homochiral/heterochiral
# split is asserted to be non-trivial, since a band with no retained triads makes every channel zero
# and the sum identity hold vacuously.
# ---------------------------------------------------------------------------
Test.@testset "calculate_helical_partial_fluxes! (in-place)" begin
    M = 16
    ks = FIT.Utils.wavenumber_grid((M, M, M), (2π, 2π, 2π))
    Random.seed!(23)
    û = randn(ComplexF64, M, M, M, 3)
    b = FIT.Types.LinearBinning(1.0)

    alloc_ref = FIT.calculate_helical_partial_fluxes(û, ks; binning = b)
    ws = FIT.Workspaces.NonlinearTermWorkspace(û, ks)
    inplace = FIT.calculate_helical_partial_fluxes!(ws, û, ks; binning = b)

    Test.@test maximum(abs, alloc_ref.total.flux) > 1e-8            # the reference carries signal
    Test.@test inplace.total.flux == alloc_ref.total.flux
    Test.@test Set(keys(inplace.channels)) == Set(keys(alloc_ref.channels))

    # The channels partition the field, so they sum to the total shell by shell.
    summed = sum(c.flux for c in values(inplace.channels))
    Test.@test maximum(abs, summed .- inplace.total.flux) <
              1e-10 * maximum(abs, inplace.total.flux)

    # Reusing the workspace for a second field gives what a fresh call gives.
    Random.seed!(24)
    v̂ = randn(ComplexF64, M, M, M, 3)
    Test.@test FIT.calculate_helical_partial_fluxes!(ws, v̂, ks; binning = b).total.flux ==
               FIT.calculate_helical_partial_fluxes(v̂, ks; binning = b).total.flux
end
