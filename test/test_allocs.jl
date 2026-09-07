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
# These assertions run in the normal suite, including under --code-coverage: the preallocated-workspace
# hot paths have no elidable temporaries, so coverage instrumentation does not change their allocation
# counts. Loading heavy plotting stacks (CairoMakie) likewise does not change them — the workspace design does.
# ---------------------------------------------------------------------------

using Test: Test
using Random: Random
using FFTW: FFTW
using FlowInvariantTransfer: FlowInvariantTransfer as FIT
using SpectralBackends: SpectralBackends
# Extension triggers for the plan-owning methods whose workspace-reuse allocation contract is asserted
# below (CGEF coarse-graining, FSH/NUFSHT spherical). Loaded here so this file's alloc contract is
# self-contained rather than depending on the includer's load order.
using CoarseGrainingEnergyFluxes: CoarseGrainingEnergyFluxes
using FastSphericalHarmonics: FastSphericalHarmonics
using NUFSHT: NUFSHT

# --- function barriers: warm up, then measure a second call with concrete-typed backend args ---
function _alloc_nlt(ws, û, ks, adv, da, sp)
    FIT.NonlinearTerm.compute_nonlinear_term!(ws, û, ks; dealiasing=da, spectral=sp, advecting_hat=adv)
    return @allocated FIT.NonlinearTerm.compute_nonlinear_term!(ws, û, ks; dealiasing=da, spectral=sp, advecting_hat=adv)
end
function _alloc_td(t, invariant, û, N̂, ks)
    FIT.Invariants.transfer_density!(t, invariant, û, N̂, ks)
    return @allocated FIT.Invariants.transfer_density!(t, invariant, û, N̂, ks)
end
function _alloc_sf!(res, ws, û, ks, sidx, sp)
    FIT.SpectralFlux.calculate_spectral_flux!(res, ws, û, ks, sidx; spectral=sp)
    return @allocated FIT.SpectralFlux.calculate_spectral_flux!(res, ws, û, ks, sidx; spectral=sp)
end
function _alloc_s2s!(res, ws, û, ks, sp)
    FIT.ShellToShellTransfer.calculate_shell_to_shell_transfer!(res, ws, û, ks; spectral=sp)
    return @allocated FIT.ShellToShellTransfer.calculate_shell_to_shell_transfer!(res, ws, û, ks; spectral=sp)
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
    FIT.BandTransfer.calculate_band_to_band_transfer!(T, net, bws, û, ks)
    return @allocated FIT.BandTransfer.calculate_band_to_band_transfer!(T, net, bws, û, ks)
end
function _alloc_comp!(ws, û, ρ̂, ks)
    FIT.Compressible.calculate_compressible_flux!(ws, û, ρ̂, ks)
    return @allocated FIT.Compressible.calculate_compressible_flux!(ws, û, ρ̂, ks)
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
function _alloc_helical!(ws, û, ks, b)
    FIT.calculate_helical_partial_fluxes!(ws, û, ks; binning=b, spectral=SpectralBackends.FFTSpectralBackend())
    return @allocated FIT.calculate_helical_partial_fluxes!(ws, û, ks; binning=b, spectral=SpectralBackends.FFTSpectralBackend())
end

# --- reuse-ratio barriers for plan-owning extension methods --------------------------------------
# These transforms allocate internally (external C plans, or no in-place transform API, or a per-scale
# plan rebuild), so the `!` form cannot be strictly 0-alloc; the contract is that reusing the workspace
# makes a repeat call allocate FAR less than a fresh (workspace-building) call. Ratios, not zeros. Each
# returns (a_reuse, a_fresh) with both paths warmed, args concretely typed inside the `@allocated`.
function _reuse_cgef(ws, u, v, ℓ, filt, xs, ys)
    FIT.CoarseGrainingFlux.calculate_coarse_graining_flux((u, v), (xs, ys), ℓ, filt)
    FIT.CoarseGrainingFlux.calculate_coarse_graining_flux!(ws, (u, v))
    a_reuse = @allocated FIT.CoarseGrainingFlux.calculate_coarse_graining_flux!(ws, (u, v))
    a_fresh = @allocated FIT.CoarseGrainingFlux.calculate_coarse_graining_flux((u, v), (xs, ys), ℓ, filt)
    return (a_reuse, a_fresh)
end
function _reuse_spherical!(ws, ζ, fresh)
    fresh(ζ)
    FIT.Spherical.calculate_spherical_transfer!(ws, ζ)
    a_reuse = @allocated FIT.Spherical.calculate_spherical_transfer!(ws, ζ)
    a_fresh = @allocated fresh(ζ)
    return (a_reuse, a_fresh)
end
function _reuse_divergent!(ws, uθ, uφ, fresh)
    fresh(uθ, uφ)
    FIT.calculate_divergent_spherical_transfer!(ws, uθ, uφ)
    a_reuse = @allocated FIT.calculate_divergent_spherical_transfer!(ws, uθ, uφ)
    a_fresh = @allocated fresh(uθ, uφ)
    return (a_reuse, a_fresh)
end
function _reuse_tod!(ws, X, dt_sig)
    FIT.TriadicOrthogonalDecomposition.triadic_orthogonal_decomposition(X; dt=dt_sig, spectral=SpectralBackends.FFTSpectralBackend())
    FIT.TriadicOrthogonalDecomposition.triadic_orthogonal_decomposition!(ws, X)
    a_reuse = @allocated FIT.TriadicOrthogonalDecomposition.triadic_orthogonal_decomposition!(ws, X)
    a_fresh = @allocated FIT.TriadicOrthogonalDecomposition.triadic_orthogonal_decomposition(X; dt=dt_sig, spectral=SpectralBackends.FFTSpectralBackend())
    return (a_reuse, a_fresh)
end

Test.@testset "Allocations" begin
    L = 2π
    N = 16
    ks2 = FIT.Utils.wavenumber_grid((N, N), (L, L))
    ks3 = FIT.Utils.wavenumber_grid((8, 8, 8), (L, L, L))
    Random.seed!(3)
    û2 = randn(ComplexF64, N, N, 2)
    û3 = randn(ComplexF64, 8, 8, 8, 3)
    θ̂  = randn(ComplexF64, N, N, 1)
    b2 = FIT.Types.LinearBinning(2π / L)

    Test.@testset "compute_nonlinear_term! — 0 alloc, every backend/dealiasing" begin
        for (û, ks) in ((û2, ks2), (û3, ks3))
            ws = FIT.Workspaces.NonlinearTermWorkspace(û, ks)
            for sp in (SpectralBackends.DirectSumSpectralBackend(), SpectralBackends.FFTSpectralBackend()),
                da in (FIT.Types.OrszagTwoThirds(), FIT.Types.NoDealiasing())
                Test.@test _alloc_nlt(ws, û, ks, û, da, sp) == 0
            end
            # Exact 3/2 padding: 0 alloc when the workspace is built for it (preplanned padded scratch).
            wsp = FIT.Workspaces.NonlinearTermWorkspace(û, ks; dealiasing=FIT.Types.PaddedThreeHalves())
            Test.@test _alloc_nlt(wsp, û, ks, û, FIT.Types.PaddedThreeHalves(), SpectralBackends.FFTSpectralBackend()) == 0
        end
        # Passive scalar (M = 1) advected by the 2D velocity.
        wsθ = FIT.Workspaces.NonlinearTermWorkspace(θ̂, ks2)
        for sp in (SpectralBackends.DirectSumSpectralBackend(), SpectralBackends.FFTSpectralBackend()), da in (FIT.Types.OrszagTwoThirds(), FIT.Types.NoDealiasing())
            Test.@test _alloc_nlt(wsθ, θ̂, ks2, û2, da, sp) == 0
        end
    end

    Test.@testset "transfer_density! — 0 alloc, every invariant" begin
        N̂2 = randn(ComplexF64, N, N, 2); N̂3 = randn(ComplexF64, 8, 8, 8, 3)
        t2 = zeros(Float64, N, N); t3 = zeros(Float64, 8, 8, 8)
        Test.@test _alloc_td(t2, FIT.Types.KineticEnergy(), û2, N̂2, ks2) == 0
        Test.@test _alloc_td(t3, FIT.Types.KineticEnergy(), û3, N̂3, ks3) == 0
        Test.@test _alloc_td(t2, FIT.Types.Enstrophy(), û2, N̂2, ks2)     == 0
        Test.@test _alloc_td(t3, FIT.Types.Enstrophy(), û3, N̂3, ks3)     == 0
        Test.@test _alloc_td(t3, FIT.Types.Helicity(), û3, N̂3, ks3)      == 0
        θN = randn(ComplexF64, N, N, 1)
        Test.@test _alloc_td(t2, FIT.Types.PassiveScalar(), θ̂, θN, ks2)  == 0
    end

    Test.@testset "calculate_spectral_flux! — 0 alloc (Serial, DirectSum & FFT)" begin
        kmag = FIT.ShellBinning.shell_coordinate(FIT.Types.IsotropicShells(), ks2)
        edges = FIT.ShellBinning.shell_edges(b2, maximum(kmag)); sidx = FIT.ShellBinning.assign_shells(kmag, edges)
        centers = collect(FIT.ShellBinning.shell_centers(b2, maximum(kmag)))
        ws = FIT.Workspaces.SpectralFluxWorkspace(û2, ks2, b2)
        res = FIT.Types.SpectralFluxResult(centers, similar(centers), similar(centers))
        for sp in (SpectralBackends.DirectSumSpectralBackend(), SpectralBackends.FFTSpectralBackend())
            Test.@test _alloc_sf!(res, ws, û2, ks2, sidx, sp) == 0
        end
    end

    Test.@testset "calculate_shell_to_shell_transfer! — 0 alloc (Serial, DirectSum & FFT)" begin
        ws = FIT.Workspaces.ShellToShellWorkspace(û2, ks2, b2)
        res = FIT.ShellToShellTransfer.calculate_shell_to_shell_transfer(û2, ks2; binning=b2)  # a result to write into
        for sp in (SpectralBackends.DirectSumSpectralBackend(), SpectralBackends.FFTSpectralBackend())
            Test.@test _alloc_s2s!(res, ws, û2, ks2, sp) == 0
        end
    end

    Test.@testset "allocating entry points — only their buffers (no runaway)" begin
        # Each builds a workspace (O(Nᴰ)) + result. Assert the allocation is bounded by a small
        # multiple of the field size — catches excess/O(N_sh·Nᴰ) regressions without brittle exact bytes.
        fb = sizeof(û2)
        FIT.SpectralFlux.calculate_spectral_flux(û2, ks2; binning=b2, spectral=SpectralBackends.FFTSpectralBackend())          # warmup
        Test.@test (@allocated FIT.SpectralFlux.calculate_spectral_flux(û2, ks2; binning=b2, spectral=SpectralBackends.FFTSpectralBackend())) <= 60fb
        FIT.ShellToShellTransfer.calculate_shell_to_shell_transfer(û2, ks2; binning=b2, spectral=SpectralBackends.FFTSpectralBackend()) # warmup
        Test.@test (@allocated FIT.ShellToShellTransfer.calculate_shell_to_shell_transfer(û2, ks2; binning=b2, spectral=SpectralBackends.FFTSpectralBackend())) <= 80fb
    end

    Test.@testset "in-place transfer variants — 0 alloc (reuse caller result + workspace)" begin
        ns = (N, N)
        # mode-to-mode!
        wsm  = FIT.Workspaces.NonlinearTermWorkspace(û2, ks2); ûp = similar(û2)
        Sm   = similar(û2, Float64, ns..., ns...); netm = similar(û2, Float64, ns...)
        resm = FIT.Types.ModeToModeTriadResult(FIT.Types.KineticEnergy(), ks2, netm, Sm)
        Test.@test _alloc_m2m!(resm, wsm, ûp, û2, ks2) == 0
        # scalar shell-to-shell!
        θ̂c   = randn(ComplexF64, N, N, 1)
        wss  = FIT.Workspaces.ShellToShellWorkspace(θ̂c, ks2, b2)
        ress = FIT.calculate_scalar_shell_to_shell_transfer(û2, θ̂c, ks2; binning=b2)
        Test.@test _alloc_ss_scalar!(ress, wss, û2, θ̂c, ks2) == 0
        # band-to-band!
        bands = FIT.Types.SmoothBands([2.0, 4.0])
        bws   = FIT.BandTransfer.BandTransferWorkspace(û2, ks2, bands)
        Tb    = zeros(2, 2); netb = zeros(2)
        Test.@test _alloc_band!(Tb, netb, bws, û2, ks2) == 0
        # scalar flux!
        wsf   = FIT.Workspaces.SpectralFluxWorkspace(θ̂c, ks2, b2)
        kmag  = FIT.ShellBinning.shell_coordinate(FIT.Types.IsotropicShells(), ks2)
        edges = FIT.ShellBinning.shell_edges(b2, maximum(kmag)); sidx = FIT.ShellBinning.assign_shells(kmag, edges)
        centers = collect(FIT.ShellBinning.shell_centers(b2, maximum(kmag)))
        resf  = FIT.Types.SpectralFluxResult(centers, similar(centers), similar(centers))
        Test.@test _alloc_scalar_flux!(resf, wsf, û2, θ̂c, ks2, sidx) == 0
        # scalar mode-to-mode!  (delegates to the 0-alloc mode-to-mode! with PassiveScalar)
        wssm  = FIT.Workspaces.NonlinearTermWorkspace(θ̂c, ks2); ûps = similar(θ̂c)
        Ssc   = similar(û2, Float64, ns..., ns...); netsc = similar(û2, Float64, ns...)
        ressm = FIT.Types.ModeToModeTriadResult(FIT.Types.PassiveScalar(), ks2, netsc, Ssc)
        Test.@test _alloc_scalar_m2m!(ressm, wssm, ûps, û2, θ̂c, ks2) == 0
    end

    Test.@testset "compressible! — reuses workspace (field intermediates not reallocated)" begin
        fb = sizeof(û2)
        ρ̂  = randn(ComplexF64, N, N)
        wsc = FIT.Compressible.CompressibleWorkspace(û2, ks2; binning=b2)
        # Only the small per-shell result vectors (output) remain: the full-grid shell-index/magnitude
        # arrays are hoisted into the workspace, and the nonlinear-assembly / shell-sum inner loops are
        # boxing-free (`::Type{FT}` where-params, not a runtime `FT` value → no `Any`-boxed `+=`).
        Test.@test _alloc_comp!(wsc, û2, ρ̂, ks2) <= fb ÷ 4
    end

    Test.@testset "bounded in-place variants — one shared workspace, output-only allocation" begin
        # partial_fluxes! reuses ONE NonlinearTermWorkspace across every decomposition-channel pair
        # (the allocating form built a fresh workspace per pair); only the channel/total result vectors
        # and the decomposition components remain (output). Helmholtz split needs the loaded ext.
        fb = sizeof(û2)
        wsp = FIT.Workspaces.NonlinearTermWorkspace(û2, ks2)
        Test.@test _alloc_partial!(wsp, û2, ks2, b2, FIT.Types.HelmholtzDecomposition()) <= 15fb

        # calculate_helical_partial_fluxes! (the exported 3D-only ! wrapper) likewise reuses ONE
        # NonlinearTermWorkspace across its 8 channels; only the channel/total result vectors and the ±
        # helical components remain (output-only, measured ≈ 6.7× the field).
        wsh3 = FIT.Workspaces.NonlinearTermWorkspace(û3, ks3)
        Test.@test _alloc_helical!(wsh3, û3, ks3, b2) <= 15 * sizeof(û3)

        # TOD serial loop: per-triad SVD scratch reused across triads; residual is the eigen!/qr! LAPACK
        # internals + the per-triad output-mode matrices stored in the result (both inherent).
        X = randn(64, 2, 8) .+ 0.1
        FIT.TriadicOrthogonalDecomposition.triadic_orthogonal_decomposition(X; nmode=3, spectral=SpectralBackends.FFTSpectralBackend())  # warmup
        Test.@test (@allocated FIT.TriadicOrthogonalDecomposition.triadic_orthogonal_decomposition(X; nmode=3, spectral=SpectralBackends.FFTSpectralBackend())) <=
            260 * sizeof(X)
    end

    # -----------------------------------------------------------------------
    # Workspace-reuse for the plan-owning extension methods (CGEF / FINUFFT / FSH / NUFSHT / TOD): the `!`
    # form reuses its plans + buffers, so a repeat call allocates far less than a fresh call that rebuilds
    # the workspace. With a serial backend CGEF prebuilds every plan and reuse is exactly 0-alloc; the
    # others — and CGEF's own threaded path, whose parallel sweeps carry an OhMyThreads task-spawn floor —
    # hold a ratio (not zero) because each wraps an external transform or task spawn that allocates
    # internally. (Previously scattered inline in the extension correctness testsets in runtests.jl.)
    Test.@testset "reuse ratio — plan-owning extension methods" begin
        Random.seed!(7)

        Test.@testset "coarse_graining (CGEF, uniform grid)" begin
            Nc = 32; Lc = 2π
            xs = range(0, Lc; length = Nc + 1)[1:Nc]; ys = copy(xs)
            u  = [sin(xs[i]) * cos(ys[j]) + 0.2 * randn() for i in 1:Nc, j in 1:Nc]
            v  = [-cos(xs[i]) * sin(ys[j]) + 0.2 * randn() for i in 1:Nc, j in 1:Nc]
            filt = FIT.Types.GaussianFilter(); ℓ = 0.5
            # Serial backend: the workspace prebuilds all three CGEF plans (ΠWorkspace + filter_plan +
            # deriv_plan), so a reused call allocates EXACTLY nothing. This is the plan-reuse contract —
            # deterministic and independent of thread count / platform (unlike a threaded task-spawn floor).
            ws_ser = FIT.CoarseGrainingFlux.CoarseGrainingFluxWorkspace(
                (u, v), (xs, ys), ℓ, filt; backend = FIT.ComputationalBackends.SerialBackend())
            a_reuse_ser, a_fresh = _reuse_cgef(ws_ser, u, v, ℓ, filt, xs, ys)
            Test.@test a_reuse_ser == 0
            Test.@test a_fresh > 0
            # Default (AutoBackend) workspace: under multithreading the parallel filter/derivative sweeps
            # carry an OhMyThreads task-spawn floor, so reuse is not strictly zero there — but complete plan
            # reuse still keeps it far below a fresh (plan-rebuilding) call. Guards a plan-reuse regression.
            ws_auto = FIT.CoarseGrainingFlux.CoarseGrainingFluxWorkspace((u, v), (xs, ys), ℓ, filt)
            a_reuse_auto, _ = _reuse_cgef(ws_auto, u, v, ℓ, filt, xs, ys)
            Test.@test a_reuse_auto < a_fresh ÷ 4
        end

        # (scattered NUFFT coarse-graining workspace-reuse allocation is asserted per provider in the
        # "NUFFT scattered coarse-graining — both providers vs exact NDFT" testset in runtests.jl,
        # alongside the NDFT correctness gate.)

        Test.@testset "spherical (FSH, equiangular grid)" begin
            lmax = 20; Ng = lmax + 1
            ζ = randn(Ng, 2Ng - 1)   # valid FSH grid shape; alloc is field-independent
            ws = FIT.Spherical.SphericalTransferWorkspace(lmax; radius = 1.0, dealias = true)
            fresh(z) = FIT.calculate_energy_transfer(FIT.Types.SphericalTransferMethod(), z)
            a_reuse, a_fresh = _reuse_spherical!(ws, ζ, fresh)
            Test.@test a_reuse < a_fresh
        end

        Test.@testset "spherical (NUFSHT, scattered)" begin
            lmax = 8; M = (2lmax + 1)^2 + 20
            ga = π * (3 - sqrt(5.0))
            θ = [acos(clamp(1 - 2 * (i + 0.5) / M, -1, 1)) for i in 0:M-1]
            φ = [mod(i * ga, 2π) for i in 0:M-1]
            ζ = [cos(θ[i]) * sin(φ[i]) for i in 1:M]
            ws = FIT.Spherical.ScatteredSphericalTransferWorkspace((θ, φ), lmax; radius = 1.0, tol = 1e-12, rtol = 1e-13)
            fresh(z) = FIT.calculate_energy_transfer(
                FIT.Types.SphericalTransferMethod(radius = 1.0), z, (θ, φ); lmax = lmax, tol = 1e-12, rtol = 1e-13)
            a_reuse, a_fresh = _reuse_spherical!(ws, ζ, fresh)
            Test.@test a_reuse < a_fresh ÷ 3
        end

        # Divergent transfer !(): the workspace preserves the FSH work path / the five NUFSHT spin plans,
        # so a repeat call allocates less than a fresh call (only the FSH-internal transforms / NUFSHT CG
        # scratch remain — the irreducible library floor).
        Test.@testset "divergent spherical (FSH, equiangular grid)" begin
            lmax = 20; Ng = lmax + 1
            uθ = randn(Ng, 2Ng - 1); uφ = randn(Ng, 2Ng - 1)
            ws = FIT.Spherical.DivergentSphericalTransferWorkspace(lmax; radius = 1.0, dealias = true)
            fresh(a, c) = FIT.calculate_energy_transfer(FIT.Types.DivergentSphericalTransferMethod(), (a, c))
            a_reuse, a_fresh = _reuse_divergent!(ws, uθ, uφ, fresh)
            Test.@test a_reuse < a_fresh
        end

        Test.@testset "divergent spherical (NUFSHT, scattered)" begin
            lmax = 8; M = (2lmax + 1)^2 + 20
            ga = π * (3 - sqrt(5.0))
            θ = [acos(clamp(1 - 2 * (i + 0.5) / M, -1, 1)) for i in 0:M-1]
            φ = [mod(i * ga, 2π) for i in 0:M-1]
            uθ = [cos(θ[i]) for i in 1:M]; uφ = [sin(φ[i]) for i in 1:M]
            ws = FIT.Spherical.ScatteredDivergentSphericalTransferWorkspace((θ, φ), lmax; radius = 1.0, tol = 1e-12, rtol = 1e-13)
            fresh(a, c) = FIT.calculate_energy_transfer(
                FIT.Types.DivergentSphericalTransferMethod(radius = 1.0), (a, c), (θ, φ); lmax = lmax, tol = 1e-12, rtol = 1e-13)
            a_reuse, a_fresh = _reuse_divergent!(ws, uθ, uφ, fresh)
            Test.@test a_reuse < a_fresh ÷ 2
        end

        Test.@testset "triadic orthogonal decomposition (reuse < fresh)" begin
            X = randn(96, 1, 4); dt_sig = 0.05
            ws = FIT.TriadicOrthogonalDecomposition.TODWorkspace(X; dt = dt_sig, spectral = SpectralBackends.FFTSpectralBackend())
            a_reuse, a_fresh = _reuse_tod!(ws, X, dt_sig)
            Test.@test a_reuse < a_fresh
        end
    end
end
