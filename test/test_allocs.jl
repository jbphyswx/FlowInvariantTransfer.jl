# ---------------------------------------------------------------------------
# Allocation contract tests.
#
# Every in-place `!()` hot path must allocate EXACTLY 0 bytes after warmup — that is the entire
# point of the preallocated-workspace design (reused across shells/mediators, no per-call garbage).
# Covered here across all backends/dealiasings/invariants, not a hand-picked few. The allocating
# public entry points must allocate only their result + workspace buffers (bounded, no runaway).
#
# Each measurement runs through a small FUNCTION BARRIER that takes the backend/dealiasing/invariant
# as an argument, so those are concretely typed inside the `@allocated` (the test loops iterate
# heterogeneous backend tuples, which would otherwise box the keyword arguments and mis-attribute the
# boxing to the function). This mirrors real usage, where the backend is a fixed concrete type in the
# specialized hot loop.
#
# `@allocated` is defeated by --code-coverage instrumentation (per-line allocation bookkeeping), so
# the strict 0-byte assertions are skipped under coverage. Included into the main suite: loading heavy
# plotting stacks (CairoMakie) does NOT change these numbers (verified) — the workspace design does.
# ---------------------------------------------------------------------------

using Test: Test
using Random: Random
using FFTW: FFTW
using FlowInvariantTransfer: FlowInvariantTransfer as FIT

const _ALLOC_COV = Base.JLOptions().code_coverage != 0

# --- function barriers: warm up, then measure a second call with concrete-typed backend args ---
function _alloc_nlt(ws, û, ks, adv, da, sp)
    FIT.compute_nonlinear_term!(ws, û, ks; dealiasing=da, spectral=sp, advecting_hat=adv)
    return @allocated FIT.compute_nonlinear_term!(ws, û, ks; dealiasing=da, spectral=sp, advecting_hat=adv)
end
function _alloc_td(t, invariant, û, N̂, ks)
    FIT.Invariants.transfer_density!(t, invariant, û, N̂, ks)
    return @allocated FIT.Invariants.transfer_density!(t, invariant, û, N̂, ks)
end
function _alloc_sf!(res, ws, û, ks, sidx, sp)
    FIT.calculate_spectral_flux!(res, ws, û, ks, sidx; spectral=sp)
    return @allocated FIT.calculate_spectral_flux!(res, ws, û, ks, sidx; spectral=sp)
end
function _alloc_s2s!(res, ws, û, ks, sp)
    FIT.calculate_shell_to_shell_transfer!(res, ws, û, ks; spectral=sp)
    return @allocated FIT.calculate_shell_to_shell_transfer!(res, ws, û, ks; spectral=sp)
end
function _alloc_m2m!(res, ws, û_p, û, ks)
    FIT.calculate_mode_to_mode_transfer!(res, ws, û_p, û, ks)
    return @allocated FIT.calculate_mode_to_mode_transfer!(res, ws, û_p, û, ks)
end
function _alloc_ss_scalar!(res, ws, û, θ̂, ks)
    FIT.calculate_scalar_shell_to_shell_transfer!(res, ws, û, θ̂, ks)
    return @allocated FIT.calculate_scalar_shell_to_shell_transfer!(res, ws, û, θ̂, ks)
end
function _alloc_band!(T, net, bws, û, ks)
    FIT.calculate_band_to_band_transfer!(T, net, bws, û, ks)
    return @allocated FIT.calculate_band_to_band_transfer!(T, net, bws, û, ks)
end
function _alloc_comp!(ws, û, ρ̂, ks, b)
    FIT.calculate_compressible_flux!(ws, û, ρ̂, ks; binning=b)
    return @allocated FIT.calculate_compressible_flux!(ws, û, ρ̂, ks; binning=b)
end
function _alloc_scalar_flux!(res, ws, û, θ̂, ks, sidx)
    FIT.calculate_scalar_flux!(res, ws, û, θ̂, ks, sidx)
    return @allocated FIT.calculate_scalar_flux!(res, ws, û, θ̂, ks, sidx)
end
function _alloc_scalar_m2m!(res, ws, ûp, û, θ̂, ks)
    FIT.calculate_scalar_mode_to_mode_transfer!(res, ws, ûp, û, θ̂, ks)
    return @allocated FIT.calculate_scalar_mode_to_mode_transfer!(res, ws, ûp, û, θ̂, ks)
end
function _alloc_partial!(ws, û, ks, b, decomp)
    FIT.calculate_partial_fluxes!(ws, û, ks; binning=b, decomposition=decomp)
    return @allocated FIT.calculate_partial_fluxes!(ws, û, ks; binning=b, decomposition=decomp)
end

Test.@testset "Allocations" begin
    L = 2π
    N = 16
    ks2 = FIT.wavenumber_grid((N, N), (L, L))
    ks3 = FIT.wavenumber_grid((8, 8, 8), (L, L, L))
    Random.seed!(3)
    û2 = randn(ComplexF64, N, N, 2)
    û3 = randn(ComplexF64, 8, 8, 8, 3)
    θ̂  = randn(ComplexF64, N, N, 1)
    b2 = FIT.LinearBinning(2π / L)

    Test.@testset "compute_nonlinear_term! — 0 alloc, every backend/dealiasing" begin
        for (û, ks) in ((û2, ks2), (û3, ks3))
            ws = FIT.NonlinearTermWorkspace(û, ks)
            for sp in (FIT.DirectSumBackend(), FIT.FFTBackend()),
                da in (FIT.OrszagTwoThirds(), FIT.NoDealiasing())
                Test.@test _alloc_nlt(ws, û, ks, û, da, sp) == 0 skip = _ALLOC_COV
            end
            # Exact 3/2 padding: 0 alloc when the workspace is built for it (preplanned padded scratch).
            wsp = FIT.NonlinearTermWorkspace(û, ks; dealiasing=FIT.PaddedThreeHalves())
            Test.@test _alloc_nlt(wsp, û, ks, û, FIT.PaddedThreeHalves(), FIT.FFTBackend()) == 0 skip = _ALLOC_COV
        end
        # Passive scalar (M = 1) advected by the 2D velocity.
        wsθ = FIT.NonlinearTermWorkspace(θ̂, ks2)
        for sp in (FIT.DirectSumBackend(), FIT.FFTBackend()), da in (FIT.OrszagTwoThirds(), FIT.NoDealiasing())
            Test.@test _alloc_nlt(wsθ, θ̂, ks2, û2, da, sp) == 0 skip = _ALLOC_COV
        end
    end

    Test.@testset "transfer_density! — 0 alloc, every invariant" begin
        N̂2 = randn(ComplexF64, N, N, 2); N̂3 = randn(ComplexF64, 8, 8, 8, 3)
        t2 = zeros(Float64, N, N); t3 = zeros(Float64, 8, 8, 8)
        Test.@test _alloc_td(t2, FIT.KineticEnergy(), û2, N̂2, ks2) == 0 skip = _ALLOC_COV
        Test.@test _alloc_td(t3, FIT.KineticEnergy(), û3, N̂3, ks3) == 0 skip = _ALLOC_COV
        Test.@test _alloc_td(t2, FIT.Enstrophy(), û2, N̂2, ks2)     == 0 skip = _ALLOC_COV
        Test.@test _alloc_td(t3, FIT.Enstrophy(), û3, N̂3, ks3)     == 0 skip = _ALLOC_COV
        Test.@test _alloc_td(t3, FIT.Helicity(), û3, N̂3, ks3)      == 0 skip = _ALLOC_COV
        θN = randn(ComplexF64, N, N, 1)
        Test.@test _alloc_td(t2, FIT.PassiveScalar(), θ̂, θN, ks2)  == 0 skip = _ALLOC_COV
    end

    Test.@testset "calculate_spectral_flux! — 0 alloc (Serial, DirectSum & FFT)" begin
        kmag = FIT.shell_coordinate(FIT.IsotropicShells(), ks2)
        edges = FIT.shell_edges(b2, maximum(kmag)); sidx = FIT.assign_shells(kmag, edges)
        centers = collect(FIT.shell_centers(b2, maximum(kmag)))
        ws = FIT.SpectralFluxWorkspace(û2, ks2, b2)
        res = FIT.SpectralFluxResult(centers, similar(centers), similar(centers))
        for sp in (FIT.DirectSumBackend(), FIT.FFTBackend())
            Test.@test _alloc_sf!(res, ws, û2, ks2, sidx, sp) == 0 skip = _ALLOC_COV
        end
    end

    Test.@testset "calculate_shell_to_shell_transfer! — 0 alloc (Serial, DirectSum & FFT)" begin
        ws = FIT.ShellToShellWorkspace(û2, ks2, b2)
        res = FIT.calculate_shell_to_shell_transfer(û2, ks2; binning=b2)  # a result to write into
        for sp in (FIT.DirectSumBackend(), FIT.FFTBackend())
            Test.@test _alloc_s2s!(res, ws, û2, ks2, sp) == 0 skip = _ALLOC_COV
        end
    end

    Test.@testset "allocating entry points — only their buffers (no runaway)" begin
        # Each builds a workspace (O(Nᴰ)) + result. Assert the allocation is bounded by a small
        # multiple of the field size — catches excess/O(N_sh·Nᴰ) regressions without brittle exact bytes.
        fb = sizeof(û2)
        FIT.calculate_spectral_flux(û2, ks2; binning=b2, spectral=FIT.FFTBackend())          # warmup
        Test.@test (@allocated FIT.calculate_spectral_flux(û2, ks2; binning=b2, spectral=FIT.FFTBackend())) <= 60fb skip = _ALLOC_COV
        FIT.calculate_shell_to_shell_transfer(û2, ks2; binning=b2, spectral=FIT.FFTBackend()) # warmup
        Test.@test (@allocated FIT.calculate_shell_to_shell_transfer(û2, ks2; binning=b2, spectral=FIT.FFTBackend())) <= 80fb skip = _ALLOC_COV
    end

    Test.@testset "in-place transfer variants — 0 alloc (reuse caller result + workspace)" begin
        ns = (N, N)
        # mode-to-mode!
        wsm  = FIT.NonlinearTermWorkspace(û2, ks2); ûp = similar(û2)
        Sm   = similar(û2, Float64, ns..., ns...); netm = similar(û2, Float64, ns...)
        resm = FIT.ModeToModeTriadResult(FIT.KineticEnergy(), ks2, netm, Sm)
        Test.@test _alloc_m2m!(resm, wsm, ûp, û2, ks2) == 0 skip = _ALLOC_COV
        # scalar shell-to-shell!
        θ̂c   = randn(ComplexF64, N, N, 1)
        wss  = FIT.ShellToShellWorkspace(θ̂c, ks2, b2)
        ress = FIT.calculate_scalar_shell_to_shell_transfer(û2, θ̂c, ks2; binning=b2)
        Test.@test _alloc_ss_scalar!(ress, wss, û2, θ̂c, ks2) == 0 skip = _ALLOC_COV
        # band-to-band!
        bands = FIT.SmoothBands([2.0, 4.0])
        bws   = FIT.BandTransferWorkspace(û2, ks2, bands)
        Tb    = zeros(2, 2); netb = zeros(2)
        Test.@test _alloc_band!(Tb, netb, bws, û2, ks2) == 0 skip = _ALLOC_COV
        # scalar flux!
        wsf   = FIT.SpectralFluxWorkspace(θ̂c, ks2, b2)
        kmag  = FIT.shell_coordinate(FIT.IsotropicShells(), ks2)
        edges = FIT.shell_edges(b2, maximum(kmag)); sidx = FIT.assign_shells(kmag, edges)
        centers = collect(FIT.shell_centers(b2, maximum(kmag)))
        resf  = FIT.SpectralFluxResult(centers, similar(centers), similar(centers))
        Test.@test _alloc_scalar_flux!(resf, wsf, û2, θ̂c, ks2, sidx) == 0 skip = _ALLOC_COV
        # scalar mode-to-mode!  (delegates to the 0-alloc mode-to-mode! with PassiveScalar)
        wssm  = FIT.NonlinearTermWorkspace(θ̂c, ks2); ûps = similar(θ̂c)
        Ssc   = similar(û2, Float64, ns..., ns...); netsc = similar(û2, Float64, ns...)
        ressm = FIT.ModeToModeTriadResult(FIT.PassiveScalar(), ks2, netsc, Ssc)
        Test.@test _alloc_scalar_m2m!(ressm, wssm, ûps, û2, θ̂c, ks2) == 0 skip = _ALLOC_COV
    end

    Test.@testset "compressible! — reuses workspace (field intermediates not reallocated)" begin
        fb = sizeof(û2)
        ρ̂  = randn(ComplexF64, N, N)
        wsc = FIT.CompressibleWorkspace(û2, ks2)
        # ~34× field (shell-binning setup + per-shell result vectors); NOT the 93× of the allocating form.
        Test.@test _alloc_comp!(wsc, û2, ρ̂, ks2, b2) <= 45fb skip = _ALLOC_COV
    end

    Test.@testset "bounded in-place variants — one shared workspace, output-only allocation" begin
        # partial_fluxes! reuses ONE NonlinearTermWorkspace across every decomposition-channel pair
        # (the allocating form built a fresh workspace per pair); only the channel/total result vectors
        # and the decomposition components remain (output). Helmholtz split needs the loaded ext.
        fb = sizeof(û2)
        wsp = FIT.NonlinearTermWorkspace(û2, ks2)
        Test.@test _alloc_partial!(wsp, û2, ks2, b2, FIT.HelmholtzDecomposition()) <= 15fb skip = _ALLOC_COV

        # TOD serial loop: per-triad SVD scratch reused across triads; residual is the eigen!/qr! LAPACK
        # internals + the per-triad output-mode matrices stored in the result (both inherent).
        X = randn(64, 2, 8) .+ 0.1
        FIT.triadic_orthogonal_decomposition(X; nmode=3, spectral=FIT.FFTBackend())  # warmup
        Test.@test (@allocated FIT.triadic_orthogonal_decomposition(X; nmode=3, spectral=FIT.FFTBackend())) <=
            260 * sizeof(X) skip = _ALLOC_COV
    end
end
