# ---------------------------------------------------------------------------
# Spectral-layout gate: the analytic wavenumber axes and the Hermitian half-spectrum weight.
#
# Every diagnostic in the package reduces an even function of k (|û|², Re{conj(û)·N̂}, …) over the
# spectrum. On the half (real-field) layout that reduction is Σ_half w·f with the weight from
# `SpectralLayout.hermitian_weight`. These tests pin that identity to round-off against the full
# layout for every parity of every axis in 1D/2D/3D, unbinned and shell-binned. Nothing downstream
# of the layout is trustworthy unless this file passes.
#
# Half fields here are always built from a REAL field via `rfft`. A random complex array of half
# shape is not the spectrum of any real field — its k₁ = 0 and Nyquist planes carry no Hermitian
# symmetry — and case (E) pins that distinction.
# ---------------------------------------------------------------------------

using Test: Test
using Random: Random
using FFTW: FFTW
using FlowInvariantTransfer: FlowInvariantTransfer as FIT

const SL = FIT.SpectralLayout

# Sizes covering both parities on every axis and all three ranks.
const LAYOUT_CASES = ((8,), (9,), (8, 8), (9, 8), (8, 9), (9, 9), (6, 4, 4), (5, 6, 7))

# Weighted half-grid sum of an even density, scalar weight form.
function _half_sum(td, ks)
    s = zero(eltype(td))
    @inbounds for I in CartesianIndices(td)
        s += SL.hermitian_weight(ks, I) * td[I]
    end
    return s
end

# Shell-binned sums; the weight is 1 on the full layout, so one routine serves both.
function _binned(td, ks, sidx, nsh)
    T = zeros(eltype(td), nsh)
    @inbounds for I in CartesianIndices(td)
        n = sidx[I]
        n == 0 && continue
        T[n] += SL.hermitian_weight(ks, I) * td[I]
    end
    return T
end

Test.@testset "spectral layout" begin
    Test.@testset "analytic axes match FFTW frequencies" begin
        for n in (8, 9, 16, 17), L in (2π, 1.0, 3.5)
            ks_f = FIT.Utils.wavenumber_grid((n,), (L,))
            ks_h = FIT.Utils.wavenumber_grid((n,), (L,); real = true)
            Test.@test ks_f[1] isa SL.FullAxis
            Test.@test ks_h[1] isa SL.HalfAxis
            Test.@test collect(ks_f[1]) ≈ FFTW.fftfreq(n, n * 2π / L)
            Test.@test collect(ks_h[1]) ≈ FFTW.rfftfreq(n, n * 2π / L)
            Test.@test length(ks_h[1]) == n ÷ 2 + 1
            Test.@test SL.full_length(ks_h[1]) == n
            Test.@test SL.max_abs(ks_f[1]) == maximum(abs, collect(ks_f[1]))
            Test.@test SL.max_abs(ks_h[1]) == maximum(abs, collect(ks_h[1]))
            Test.@test SL.max_abs(ks_f[1]) == SL.max_abs(ks_h[1])
        end
        ks = FIT.Utils.wavenumber_grid((8, 6, 4), (2π, 2π, 2π); real = true)
        Test.@test SL.is_half(ks)
        Test.@test SL.full_size(ks) == (8, 6, 4)
        Test.@test SL.spectral_size(ks) == (5, 6, 4)
        Test.@test !SL.is_half(FIT.Utils.wavenumber_grid((8, 6, 4), (2π, 2π, 2π)))
        # A plain vector of wavenumbers is a full axis.
        Test.@test !SL.is_half((collect(FFTW.fftfreq(8)),))
        Test.@test SL.full_size((collect(FFTW.fftfreq(8)),)) == (8,)
    end

    Test.@testset "Hermitian weight identities — every parity, 1D/2D/3D" begin
        Random.seed!(11)
        for ns in LAYOUT_CASES
            nd = length(ns)
            Np = prod(ns)
            Ls = ntuple(d -> 2π * (1 + 0.25d), nd)
            ks_f = FIT.Utils.wavenumber_grid(ns, Ls)
            ks_h = FIT.Utils.wavenumber_grid(ns, Ls; real = true)

            u = randn(ns...)
            N = u .^ 2 .- sum(u .^ 2) / Np          # any real nonlinear function of u
            û_f = FFTW.fft(u) ./ Np;  N̂_f = FFTW.fft(N) ./ Np
            û_h = FFTW.rfft(u) ./ Np; N̂_h = FFTW.rfft(N) ./ Np
            Test.@test size(û_h) == SL.spectral_size(ks_h)

            # (A) energy: Σ_full |û|² == Σ_half w |û|²
            E_f = sum(abs2, û_f)
            E_h = _half_sum(abs2.(û_h), ks_h)
            Test.@test E_h ≈ E_f rtol = 1e-13

            # (B) transfer density: Σ_full Re{conj(û)N̂} == Σ_half w Re{conj(û)N̂}
            td_f = real.(conj.(û_f) .* N̂_f)
            td_h = real.(conj.(û_h) .* N̂_h)
            Test.@test _half_sum(td_h, ks_h) ≈ sum(td_f) rtol = 1e-12 atol = 1e-15 * E_f

            # (C) shell-binned, both layouts give the same edges and the same per-shell sums
            geom = FIT.Types.IsotropicShells()
            kmag_f = FIT.ShellBinning.shell_coordinate(geom, ks_f)
            kmag_h = FIT.ShellBinning.shell_coordinate(geom, ks_h)
            kmax_closed = sqrt(sum(d -> SL.max_abs(ks_f[d])^2, 1:nd))
            Test.@test maximum(kmag_f) ≈ kmax_closed
            Test.@test maximum(kmag_h) ≈ kmax_closed
            for b in (FIT.Types.LinearBinning(2π / Ls[1]), FIT.Types.LogarithmicBinning(2π / Ls[1], 1.5))
                edges = FIT.ShellBinning.shell_edges(b, kmax_closed)
                nsh = length(edges) - 1
                sidx_f = FIT.ShellBinning.assign_shells(kmag_f, edges)
                sidx_h = FIT.ShellBinning.assign_shells(kmag_h, edges)
                T_f = _binned(td_f, ks_f, sidx_f, nsh)
                T_h = _binned(td_h, ks_h, sidx_h, nsh)
                Test.@test T_h ≈ T_f rtol = 1e-12 atol = 1e-15 * E_f
                E_fs = _binned(abs2.(û_f), ks_f, sidx_f, nsh)
                E_hs = _binned(abs2.(û_h), ks_h, sidx_h, nsh)
                Test.@test E_hs ≈ E_fs rtol = 1e-13
            end

            # (D) the broadcast weight equals the scalar weight
            w = SL.hermitian_weights(Float64, ks_h)
            Test.@test size(w, 1) == size(û_h, 1)
            Test.@test sum(w .* abs2.(û_h)) ≈ E_h rtol = 1e-14
            Test.@test all(SL.hermitian_weights(Float64, ks_f) .== 1)

            # (E) `rfft ∘ brfft` is `Np ×` the identity on a genuine real-field half; on an arbitrary
            # complex array of half shape it projects onto the Hermitian-consistent part and moves it.
            Test.@test FFTW.rfft(FFTW.brfft(û_h, ns[1])) ./ Np ≈ û_h rtol = 1e-12
            if ns[1] > 2
                x = randn(ComplexF64, size(û_h)...)
                Test.@test !isapprox(FFTW.rfft(FFTW.brfft(x, ns[1])) ./ Np, x)
            end
        end
    end
end
