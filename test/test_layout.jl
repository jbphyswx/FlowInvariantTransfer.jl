# ---------------------------------------------------------------------------
# The package's wavenumber grids and shell binning on both spectral layouts.
#
# The layout itself (axes, fold weights, Nyquist rules) is FlowTransformBindings' and tested there. Here:
# `wavenumber_grid` builds those axes, and the shell sums of an even density over the half layout equal
# the full-layout ones shell by shell, for every parity of every axis in 1D/2D/3D. Half fields are built
# from a real field via `rfft`.
# ---------------------------------------------------------------------------

using Test: Test
using Random: Random
using FFTW: FFTW
using FlowInvariantTransfer: FlowInvariantTransfer as FIT
using FlowTransformBindings: FlowTransformBindings as FTB

# Sizes covering both parities on every axis and all three ranks.
const LAYOUT_CASES = ((8,), (9,), (8, 8), (9, 8), (8, 9), (9, 9), (6, 4, 4), (5, 6, 7))

function _binned(td, ks, sidx, nsh)
    T = zeros(eltype(td), nsh)
    @inbounds for I in CartesianIndices(td)
        n = sidx[I]
        n == 0 && continue
        T[n] += FTB.mode_fold(ks, I) * td[I]
    end
    return T
end

Test.@testset "spectral layout" begin
    Test.@testset "wavenumber_grid builds the layout's axes" begin
        for n in (8, 9), L in (2π, 3.5)
            ks_f = FIT.Utils.wavenumber_grid((n, 6), (L, 2L))
            ks_h = FIT.Utils.wavenumber_grid((n, 6), (L, 2L); real = true)
            Test.@test ks_f isa Tuple{FTB.FFTAxis{Float64}, FTB.FFTAxis{Float64}}
            Test.@test ks_h isa Tuple{FTB.RFFTAxis{Float64, Nothing}, FTB.FFTAxis{Float64}}
            Test.@test collect(ks_f[1]) ≈ FFTW.fftfreq(n, n * 2π / L)
            Test.@test collect(ks_h[1]) ≈ FFTW.rfftfreq(n, n * 2π / L)
            Test.@test collect(ks_h[2]) ≈ FFTW.fftfreq(6, 6 * 2π / (2L))
        end
        Test.@test eltype(FIT.Utils.wavenumber_grid((4, 4), (1.0f0, 2); real = true)[2]) === Float32
    end

    Test.@testset "shell sums agree on the half and full layouts — every parity, 1D/2D/3D" begin
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
            E_f = sum(abs2, û_f)
            td_f = real.(conj.(û_f) .* N̂_f)
            td_h = real.(conj.(û_h) .* N̂_h)

            geom = FIT.Types.IsotropicShells()
            kmag_f = FIT.ShellBinning.shell_coordinate(geom, ks_f)
            kmag_h = FIT.ShellBinning.shell_coordinate(geom, ks_h)
            kmax_closed = sqrt(sum(d -> FTB.max_abs(ks_f[d])^2, 1:nd))
            Test.@test maximum(kmag_f) ≈ kmax_closed
            Test.@test maximum(kmag_h) ≈ kmax_closed
            Test.@test FIT.ShellBinning.max_shell_coordinate(geom, ks_h) ≈ kmax_closed
            for b in (FIT.Types.LinearBinning(2π / Ls[1]), FIT.Types.LogarithmicBinning(2π / Ls[1], 1.5))
                edges = FIT.ShellBinning.shell_edges(b, kmax_closed)
                nsh = length(edges) - 1
                sidx_f = FIT.ShellBinning.assign_shells(kmag_f, edges)
                sidx_h = FIT.ShellBinning.assign_shells(kmag_h, edges)
                Test.@test _binned(td_h, ks_h, sidx_h, nsh) ≈ _binned(td_f, ks_f, sidx_f, nsh) rtol = 1e-12 atol = 1e-15 * E_f
                Test.@test _binned(abs2.(û_h), ks_h, sidx_h, nsh) ≈ _binned(abs2.(û_f), ks_f, sidx_f, nsh) rtol = 1e-13
            end
        end
    end
end
