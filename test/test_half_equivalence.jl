# ---------------------------------------------------------------------------
# End-to-end equivalence of the two spectral layouts.
#
# `test_layout.jl` pins the Hermitian weight identity itself. This file pins the consequence: every
# public diagnostic, run on the half spectrum of a real field, must return what it returns on that
# same field's full complex spectrum. The half layout is the default for real input, so a divergence
# here is a wrong answer for the ordinary user.
#
# Every comparison asserts the full-layout reference carries signal before comparing to it: on a small
# grid a 2/3-dealiased band can leave a reference that is only round-off, and any two round-off
# numbers agree to any tolerance.
# ---------------------------------------------------------------------------

using Test: Test
using Random: Random
using FFTW: FFTW
using FlowInvariantTransfer: FlowInvariantTransfer as FIT
using SpectralBackends: SpectralBackends

const FFTB_HE = SpectralBackends.FFTSpectralBackend()

# A real physical field and its two spectral representations of the same data.
function _he_field(ns, D; seed = 31)
    Random.seed!(seed)
    nd = length(ns)
    u = ntuple(_ -> randn(ns...), D)
    ks_full = FIT.Utils.wavenumber_grid(ns, ntuple(_ -> 2π, nd))
    ks_half = FIT.Utils.wavenumber_grid(ns, ntuple(_ -> 2π, nd); real = true)
    Np = prod(ns)
    û_full = cat((FFTW.fft(u[c]) ./ Np for c in 1:D)...; dims = nd + 1)
    û_half = cat((FFTW.rfft(u[c]) ./ Np for c in 1:D)...; dims = nd + 1)
    return u, û_full, ks_full, û_half, ks_half
end

# Compare, having first established the reference is not round-off.
function _he_same(a, b; rtol = 1e-9)
    scale = maximum(abs, b)
    Test.@test scale > 1e-8                       # the reference carries signal
    Test.@test maximum(abs, a .- b) <= rtol * scale
    return nothing
end

Test.@testset "half vs full layout — every Cartesian diagnostic" begin
    binning = FIT.Types.LinearBinning(1.0)

    Test.@testset "2D $(ns)" for ns in ((16, 16), (18, 14))
        _, ûf, ksf, ûh, ksh = _he_field(ns, 2)

        rf = FIT.calculate_spectral_flux(ûf, ksf; binning = binning, spectral = FFTB_HE)
        rh = FIT.calculate_spectral_flux(ûh, ksh; binning = binning, spectral = FFTB_HE)
        _he_same(rh.transfer_spectrum, rf.transfer_spectrum)
        _he_same(rh.flux, rf.flux)

        zf = FIT.calculate_spectral_flux(ûf, ksf; binning = binning, spectral = FFTB_HE,
                                         invariant = FIT.Types.Enstrophy())
        zh = FIT.calculate_spectral_flux(ûh, ksh; binning = binning, spectral = FFTB_HE,
                                         invariant = FIT.Types.Enstrophy())
        _he_same(zh.transfer_spectrum, zf.transfer_spectrum)

        sf = FIT.calculate_shell_to_shell_transfer(ûf, ksf; binning = binning, spectral = FFTB_HE)
        sh = FIT.calculate_shell_to_shell_transfer(ûh, ksh; binning = binning, spectral = FFTB_HE)
        _he_same(sh.transfer_matrix, sf.transfer_matrix)

        bands = FIT.Types.SmoothBands([2.0, 4.0])
        bf = FIT.BandTransfer.calculate_band_to_band_transfer(ûf, ksf; bands = bands, spectral = FFTB_HE)
        bh = FIT.BandTransfer.calculate_band_to_band_transfer(ûh, ksh; bands = bands, spectral = FFTB_HE)
        _he_same(bh.transfer_matrix, bf.transfer_matrix)

        # Compressible: needs a density field on the same layout.
        Random.seed!(77); ρ = randn(ns...) .+ 4.0
        Np = prod(ns)
        ρ̂f = reshape(FFTW.fft(ρ) ./ Np, size(ûf)[1:2]..., 1)
        ρ̂h = reshape(FFTW.rfft(ρ) ./ Np, size(ûh)[1:2]..., 1)
        cf = FIT.calculate_compressible_flux(ûf, ρ̂f, ksf; binning = binning, spectral = FFTB_HE)
        ch = FIT.calculate_compressible_flux(ûh, ρ̂h, ksh; binning = binning, spectral = FFTB_HE)
        _he_same(ch.transfer_spectrum, cf.transfer_spectrum)
        _he_same(ch.channels.rotational, cf.channels.rotational)
    end

    Test.@testset "3D $(ns)" for ns in ((12, 12, 12), (12, 10, 14))
        _, ûf, ksf, ûh, ksh = _he_field(ns, 3)

        rf = FIT.calculate_spectral_flux(ûf, ksf; binning = binning, spectral = FFTB_HE)
        rh = FIT.calculate_spectral_flux(ûh, ksh; binning = binning, spectral = FFTB_HE)
        _he_same(rh.transfer_spectrum, rf.transfer_spectrum)

        hf = FIT.calculate_spectral_flux(ûf, ksf; binning = binning, spectral = FFTB_HE,
                                         invariant = FIT.Types.Helicity())
        hh = FIT.calculate_spectral_flux(ûh, ksh; binning = binning, spectral = FFTB_HE,
                                         invariant = FIT.Types.Helicity())
        _he_same(hh.transfer_spectrum, hf.transfer_spectrum)

        # Helical partial fluxes exercise the decomposition on both layouts.
        pf = FIT.calculate_helical_partial_fluxes(ûf, ksf; binning = binning, spectral = FFTB_HE)
        ph = FIT.calculate_helical_partial_fluxes(ûh, ksh; binning = binning, spectral = FFTB_HE)
        _he_same(ph.total.flux, pf.total.flux)
    end

    # The nonlinear term itself, under every dealiasing, is what all of the above ride on.
    Test.@testset "nonlinear term — $(nameof(typeof(dealias))) $(ns)" for
            dealias in (FIT.Types.OrszagTwoThirds(), FIT.Types.NoDealiasing(), FIT.Types.PaddedThreeHalves()),
            ns in ((16, 16), (18, 14), (12, 12, 12))
        nd = length(ns)
        _, ûf, ksf, ûh, ksh = _he_field(ns, nd)
        wf = FIT.Workspaces.NonlinearTermWorkspace(ûf, ksf; dealiasing = dealias)
        wh = FIT.Workspaces.NonlinearTermWorkspace(ûh, ksh; dealiasing = dealias)
        FIT.NonlinearTerm.compute_nonlinear_term!(wf, ûf, ksf; dealiasing = dealias, spectral = FFTB_HE)
        FIT.NonlinearTerm.compute_nonlinear_term!(wh, ûh, ksh; dealiasing = dealias, spectral = FFTB_HE)
        # The half N̂ is the leading k₁ ≥ 0 block of the full one, mode for mode.
        colons = ntuple(_ -> Colon(), nd - 1)
        nh = size(wh.N̂, 1)
        ref = view(wf.N̂, 1:nh, colons..., :)
        _he_same(wh.N̂, ref; rtol = 1e-10)
    end
end
