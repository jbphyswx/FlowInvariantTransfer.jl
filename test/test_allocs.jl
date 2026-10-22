# ---------------------------------------------------------------------------
# Allocation contract: every in-place `!` hot path allocates nothing after warmup, and the bounded
# in-place variants allocate only their output.
#
# Each measurement runs through a function barrier that takes the backend, dealiasing and invariant as
# arguments, so they are concretely typed inside the `@allocated`.
# ---------------------------------------------------------------------------

using Test: Test
using Random: Random
using FFTW: FFTW
using FlowInvariantTransfer: FlowInvariantTransfer as FIT
using SpectralBackends: SpectralBackends
# Extension triggers for the CGEF and NUFSHT workspaces below.
using CoarseGrainingEnergyFluxes: CoarseGrainingEnergyFluxes
using NUFSHT: NUFSHT
using FINUFFT: FINUFFT
using FlowTransformBindings: FlowTransformBindings as FTB

# --- function barriers: warm up, then measure a second call with concrete-typed backend args ---
function _alloc_nlt(ws, û, ks, adv, da, sp)
    FIT.NonlinearTerm.compute_nonlinear_term!(ws, û, ks; dealiasing=da, transform=sp, advecting_hat=adv)
    return @allocated FIT.NonlinearTerm.compute_nonlinear_term!(ws, û, ks; dealiasing=da, transform=sp, advecting_hat=adv)
end
function _alloc_td(t, invariant, û, N̂, ks)
    FIT.Invariants.transfer_density!(t, invariant, û, N̂, ks)
    return @allocated FIT.Invariants.transfer_density!(t, invariant, û, N̂, ks)
end
function _alloc_sf!(res, ws, û, ks, sidx, sp)
    FIT.SpectralFlux.calculate_spectral_flux!(res, ws, û, ks, sidx; transform=sp)
    return @allocated FIT.SpectralFlux.calculate_spectral_flux!(res, ws, û, ks, sidx; transform=sp)
end
function _alloc_s2s!(res, ws, û, ks, sp)
    FIT.ShellToShellTransfer.calculate_shell_to_shell_transfer!(res, ws, û, ks; transform=sp)
    return @allocated FIT.ShellToShellTransfer.calculate_shell_to_shell_transfer!(res, ws, û, ks; transform=sp)
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
    FIT.calculate_helical_partial_fluxes!(ws, û, ks; binning=b, transform=SpectralBackends.FFTSpectralBackend())
    return @allocated FIT.calculate_helical_partial_fluxes!(ws, û, ks; binning=b, transform=SpectralBackends.FFTSpectralBackend())
end

function _reuse_cgef(ws, u, v)
    FIT.CoarseGrainingFlux.calculate_coarse_graining_flux!(ws, (u, v))
    return @allocated FIT.CoarseGrainingFlux.calculate_coarse_graining_flux!(ws, (u, v))
end
function _reuse_spherical!(ws, ζ)
    FIT.Spherical.calculate_spherical_transfer!(ws, ζ)
    return @allocated FIT.Spherical.calculate_spherical_transfer!(ws, ζ)
end
function _reuse_divergent!(ws, uθ, uφ)
    FIT.calculate_divergent_spherical_transfer!(ws, uθ, uφ)
    return @allocated FIT.calculate_divergent_spherical_transfer!(ws, uθ, uφ)
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
        res = FIT.Types.SpectralFluxResult(centers, edges, similar(centers), similar(centers))
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

    Test.@testset "in-place transfer variants — 0 alloc (reuse caller result + workspace)" begin
        ns = (N, N)
        # mode-to-mode!
        wsm  = FIT.Workspaces.NonlinearTermWorkspace(û2, ks2); ûp = similar(û2)
        Sm   = similar(û2, Float64, ns..., ns...); netm = similar(û2, Float64, ns...)
        wm   = FIT.ModeToModeTransfer.mode_weights(û2, Float64, ks2)
        resm = FIT.Types.ModeToModeTriadResult(FIT.Types.KineticEnergy(), ks2, wm, netm, Sm)
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
        resf  = FIT.Types.SpectralFluxResult(centers, edges, similar(centers), similar(centers))
        Test.@test _alloc_scalar_flux!(resf, wsf, û2, θ̂c, ks2, sidx) == 0
        # scalar mode-to-mode!  (delegates to the 0-alloc mode-to-mode! with PassiveScalar)
        wssm  = FIT.Workspaces.NonlinearTermWorkspace(θ̂c, ks2); ûps = similar(θ̂c)
        Ssc   = similar(û2, Float64, ns..., ns...); netsc = similar(û2, Float64, ns...)
        ressm = FIT.Types.ModeToModeTriadResult(FIT.Types.PassiveScalar(), ks2, wm, netsc, Ssc)
        Test.@test _alloc_scalar_m2m!(ressm, wssm, ûps, û2, θ̂c, ks2) == 0
    end

    Test.@testset "compressible! — reuses workspace (field intermediates not reallocated)" begin
        fb = sizeof(û2)
        ρ̂  = randn(ComplexF64, N, N)
        wsc = FIT.Compressible.CompressibleWorkspace(û2, ks2; binning=b2)
        # Only the per-shell result vectors are allocated.
        Test.@test _alloc_comp!(wsc, û2, ρ̂, ks2) <= fb ÷ 4
    end

    Test.@testset "bounded in-place variants — one shared workspace, output-only allocation" begin
        # One NonlinearTermWorkspace serves every decomposition-channel pair; the channel and total result
        # vectors and the decomposition components are the output.
        fb = sizeof(û2)
        wsp = FIT.Workspaces.NonlinearTermWorkspace(û2, ks2)
        Test.@test _alloc_partial!(wsp, û2, ks2, b2, FIT.Types.HelmholtzDecomposition()) <= 15fb

        wsh3 = FIT.Workspaces.NonlinearTermWorkspace(û3, ks3)
        Test.@test _alloc_helical!(wsh3, û3, ks3, b2) <= 15 * sizeof(û3)
    end

    # A workspace holds every plan of the external transforms it runs, so a reused call allocates nothing.
    Test.@testset "workspace reuse — plan-owning extension methods" begin
        Random.seed!(7)

        Test.@testset "coarse_graining (CGEF, uniform grid)" begin
            Nc = 32; Lc = 2π
            xs = range(0, Lc; length = Nc + 1)[1:Nc]; ys = copy(xs)
            u  = [sin(xs[i]) * cos(ys[j]) + 0.2 * randn() for i in 1:Nc, j in 1:Nc]
            v  = [-cos(xs[i]) * sin(ys[j]) + 0.2 * randn() for i in 1:Nc, j in 1:Nc]
            filt = FIT.Types.GaussianFilter(); ℓ = 0.5
            ws_ser = FIT.CoarseGrainingFlux.CoarseGrainingFluxWorkspace(
                (u, v), (xs, ys), ℓ, filt; execution = FIT.ComputationalBackends.SerialBackend())
            Test.@test _reuse_cgef(ws_ser, u, v) == 0
        end

        # FINUFFT allocates nothing per execution, and the workspace holds every fit's LSMR workspace.
        Test.@testset "spherical (NUFSHT, scattered)" begin
            lmax = 8; M = (2lmax + 1)^2 + 20
            ga = π * (3 - sqrt(5.0))
            θ = [acos(clamp(1 - 2 * (i + 0.5) / M, -1, 1)) for i in 0:M-1]
            φ = [mod(i * ga, 2π) for i in 0:M-1]
            ζ = [cos(θ[i]) * sin(φ[i]) for i in 1:M]
            nu = FTB.FINUFFTBackend()
            ws = FIT.Spherical.ScatteredSphericalTransferWorkspace((θ, φ), lmax; radius = 1.0, tol = 1e-12, rtol = 1e-13,
                                                                   nufft = nu)
            Test.@test _reuse_spherical!(ws, ζ) == 0
        end

        # The workspace holds its five plans and four fits' LSMR workspaces.
        Test.@testset "divergent spherical (NUFSHT, scattered)" begin
            lmax = 8; M = (2lmax + 1)^2 + 20
            ga = π * (3 - sqrt(5.0))
            θ = [acos(clamp(1 - 2 * (i + 0.5) / M, -1, 1)) for i in 0:M-1]
            φ = [mod(i * ga, 2π) for i in 0:M-1]
            uθ = [cos(θ[i]) for i in 1:M]; uφ = [sin(φ[i]) for i in 1:M]
            nu = FTB.FINUFFTBackend()
            ws = FIT.Spherical.ScatteredDivergentSphericalTransferWorkspace((θ, φ), lmax; radius = 1.0, tol = 1e-12,
                                                                            rtol = 1e-13, nufft = nu)
            Test.@test _reuse_divergent!(ws, uθ, uφ) == 0
        end
    end
end
