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
using FlowInvariantTransfer: FlowInvariantTransfer as FET

const _ALLOC_COV = Base.JLOptions().code_coverage != 0

# --- function barriers: warm up, then measure a second call with concrete-typed backend args ---
function _alloc_nlt(ws, û, ks, adv, da, sp)
    FET.compute_nonlinear_term!(ws, û, ks; dealiasing=da, spectral=sp, advecting_hat=adv)
    return @allocated FET.compute_nonlinear_term!(ws, û, ks; dealiasing=da, spectral=sp, advecting_hat=adv)
end
function _alloc_td(t, invariant, û, N̂, ks)
    FET.Invariants.transfer_density!(t, invariant, û, N̂, ks)
    return @allocated FET.Invariants.transfer_density!(t, invariant, û, N̂, ks)
end
function _alloc_sf!(res, ws, û, ks, sidx, sp)
    FET.calculate_spectral_flux!(res, ws, û, ks, sidx; spectral=sp)
    return @allocated FET.calculate_spectral_flux!(res, ws, û, ks, sidx; spectral=sp)
end
function _alloc_s2s!(res, ws, û, ks, sp)
    FET.calculate_shell_to_shell_transfer!(res, ws, û, ks; spectral=sp)
    return @allocated FET.calculate_shell_to_shell_transfer!(res, ws, û, ks; spectral=sp)
end

Test.@testset "Allocations" begin
    L = 2π
    N = 16
    ks2 = FET.wavenumber_grid((N, N), (L, L))
    ks3 = FET.wavenumber_grid((8, 8, 8), (L, L, L))
    Random.seed!(3)
    û2 = randn(ComplexF64, N, N, 2)
    û3 = randn(ComplexF64, 8, 8, 8, 3)
    θ̂  = randn(ComplexF64, N, N, 1)
    b2 = FET.LinearBinning(2π / L)

    Test.@testset "compute_nonlinear_term! — 0 alloc, every backend/dealiasing" begin
        for (û, ks) in ((û2, ks2), (û3, ks3))
            ws = FET.NonlinearTermWorkspace(û, ks)
            for sp in (FET.DirectSumBackend(), FET.FFTBackend()),
                da in (FET.OrszagTwoThirds(), FET.NoDealiasing())
                Test.@test _alloc_nlt(ws, û, ks, û, da, sp) == 0 skip = _ALLOC_COV
            end
            # Exact 3/2 padding: 0 alloc when the workspace is built for it (preplanned padded scratch).
            wsp = FET.NonlinearTermWorkspace(û, ks; dealiasing=FET.PaddedThreeHalves())
            Test.@test _alloc_nlt(wsp, û, ks, û, FET.PaddedThreeHalves(), FET.FFTBackend()) == 0 skip = _ALLOC_COV
        end
        # Passive scalar (M = 1) advected by the 2D velocity.
        wsθ = FET.NonlinearTermWorkspace(θ̂, ks2)
        for sp in (FET.DirectSumBackend(), FET.FFTBackend()), da in (FET.OrszagTwoThirds(), FET.NoDealiasing())
            Test.@test _alloc_nlt(wsθ, θ̂, ks2, û2, da, sp) == 0 skip = _ALLOC_COV
        end
    end

    Test.@testset "transfer_density! — 0 alloc, every invariant" begin
        N̂2 = randn(ComplexF64, N, N, 2); N̂3 = randn(ComplexF64, 8, 8, 8, 3)
        t2 = zeros(Float64, N, N); t3 = zeros(Float64, 8, 8, 8)
        Test.@test _alloc_td(t2, FET.KineticEnergy(), û2, N̂2, ks2) == 0 skip = _ALLOC_COV
        Test.@test _alloc_td(t3, FET.KineticEnergy(), û3, N̂3, ks3) == 0 skip = _ALLOC_COV
        Test.@test _alloc_td(t2, FET.Enstrophy(), û2, N̂2, ks2)     == 0 skip = _ALLOC_COV
        Test.@test _alloc_td(t3, FET.Enstrophy(), û3, N̂3, ks3)     == 0 skip = _ALLOC_COV
        Test.@test _alloc_td(t3, FET.Helicity(), û3, N̂3, ks3)      == 0 skip = _ALLOC_COV
        θN = randn(ComplexF64, N, N, 1)
        Test.@test _alloc_td(t2, FET.PassiveScalar(), θ̂, θN, ks2)  == 0 skip = _ALLOC_COV
    end

    Test.@testset "calculate_spectral_flux! — 0 alloc (Serial, DirectSum & FFT)" begin
        kmag = FET.shell_coordinate(FET.IsotropicShells(), ks2)
        edges = FET.shell_edges(b2, maximum(kmag)); sidx = FET.assign_shells(kmag, edges)
        centers = collect(FET.shell_centers(b2, maximum(kmag)))
        ws = FET.SpectralFluxWorkspace(û2, ks2, b2)
        res = FET.SpectralFluxResult(centers, similar(centers), similar(centers))
        for sp in (FET.DirectSumBackend(), FET.FFTBackend())
            Test.@test _alloc_sf!(res, ws, û2, ks2, sidx, sp) == 0 skip = _ALLOC_COV
        end
    end

    Test.@testset "calculate_shell_to_shell_transfer! — 0 alloc (Serial, DirectSum & FFT)" begin
        ws = FET.ShellToShellWorkspace(û2, ks2, b2)
        res = FET.calculate_shell_to_shell_transfer(û2, ks2; binning=b2)  # a result to write into
        for sp in (FET.DirectSumBackend(), FET.FFTBackend())
            Test.@test _alloc_s2s!(res, ws, û2, ks2, sp) == 0 skip = _ALLOC_COV
        end
    end

    Test.@testset "allocating entry points — only their buffers (no runaway)" begin
        # Each builds a workspace (O(Nᴰ)) + result. Assert the allocation is bounded by a small
        # multiple of the field size — catches excess/O(N_sh·Nᴰ) regressions without brittle exact bytes.
        fb = sizeof(û2)
        FET.calculate_spectral_flux(û2, ks2; binning=b2, spectral=FET.FFTBackend())          # warmup
        Test.@test (@allocated FET.calculate_spectral_flux(û2, ks2; binning=b2, spectral=FET.FFTBackend())) <= 60fb skip = _ALLOC_COV
        FET.calculate_shell_to_shell_transfer(û2, ks2; binning=b2, spectral=FET.FFTBackend()) # warmup
        Test.@test (@allocated FET.calculate_shell_to_shell_transfer(û2, ks2; binning=b2, spectral=FET.FFTBackend())) <= 80fb skip = _ALLOC_COV
    end
end
