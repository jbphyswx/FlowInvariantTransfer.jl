# ---------------------------------------------------------------------------
# Band content of an invariant.
#
# `E(n) = Σ_k w_n(k)·e(k)` is one quantity with the band definition supplying `w_n`, so the gate is
# Parseval: over a partition that covers every mode, the bands sum to the field's own total. That holds
# for the sharp indicator of a shell and for the graded partition of unity alike, and on either spectral
# layout — the half carries the Hermitian weight that stands in for the modes it does not store.
# ---------------------------------------------------------------------------

using Test: Test
using Random: Random
using Statistics: Statistics
using FFTW: FFTW
using FlowGeometries: FlowGeometries as FG
using FlowFieldSpectra: FlowFieldSpectra
using FlowInvariantTransfer: FlowInvariantTransfer as FIT

Test.@testset "band energies — invariant content per band" begin
    N = 32; L = 2π
    x = range(0, L; length = N + 1)[1:N]
    Random.seed!(17)
    u = randn(N, N); v = randn(N, N)
    u .-= Statistics.mean(u); v .-= Statistics.mean(v)      # zero mean: the DC mode carries nothing
    grid = FG.Grids.StructuredGrid(FG.Geometry.CartesianGeometry{Float64}(), x, x; topology = (true, true))
    total = 0.5 * (Statistics.mean(u .^ 2) + Statistics.mean(v .^ 2))
    Test.@test total > 1e-3                                  # the field carries energy

    û_h, ks_h = FIT.to_spectral((u, v), grid)                          # half layout
    û_f, ks_f = FIT.to_spectral((u, v), grid; real_layout = false)     # full layout
    Test.@test FIT.SpectralLayout.is_half(ks_h)
    Test.@test !FIT.SpectralLayout.is_half(ks_f)

    # Sharp bands over a partition whose last edge clears the corner mode, so every mode is assigned
    # and the sum is the whole field. (The package's half-open `[kₙ, kₙ₊₁)` shells otherwise drop the
    # single mode sitting exactly on the outermost edge.)
    kmax = FIT.ShellBinning.max_shell_coordinate(FIT.Types.IsotropicShells(), ks_h)
    covering = FIT.Types.CustomBinning(collect(range(0.0, 1.01 * kmax; length = 12)))
    for (nm, û, ks) in (("half", û_h, ks_h), ("full", û_f, ks_f))
        sharp = FIT.calculate_band_energies(û, ks; bands = covering)
        Test.@test length(sharp.energies) == length(sharp.centers)
        Test.@test all(>=(0), sharp.energies)
        Test.@test isapprox(sum(sharp.energies), total; rtol = 1e-12)
    end

    # The graded partition of unity sums to 1 at every mode, so it too recovers the total exactly, and
    # the two layouts agree.
    bands = FIT.Types.SmoothBands([1.5, 3.0, 6.0, 12.0], 0.7)
    smooth_h = FIT.calculate_band_energies(û_h, ks_h; bands = bands)
    smooth_f = FIT.calculate_band_energies(û_f, ks_f; bands = bands)
    Test.@test all(>=(0), smooth_h.energies)
    Test.@test isapprox(sum(smooth_h.energies), total; rtol = 1e-12)
    Test.@test isapprox(smooth_h.energies, smooth_f.energies; rtol = 1e-12)

    # The same `bands` drives the band-to-band transfer, so the energies describe that partition.
    T = FIT.calculate_band_to_band_transfer(û_h, ks_h; bands = bands)
    Test.@test size(T.transfer_matrix, 1) == length(smooth_h.energies)

    # Enstrophy is ½|ω̂|², with the derivative wavenumber zeroed at an even axis's Nyquist mode — that
    # slot is its own image under k ↦ −k, so it admits no first-order-in-k operator.
    kx = [i <= N ÷ 2 ? i - 1 : i - 1 - N for i in 1:N, _ in 1:N] .* 1.0
    ky = [j <= N ÷ 2 ? j - 1 : j - 1 - N for _ in 1:N, j in 1:N] .* 1.0
    kx[abs.(kx) .== N ÷ 2] .= 0.0
    ky[abs.(ky) .== N ÷ 2] .= 0.0
    ω̂ = im .* (kx .* (FFTW.fft(v) ./ N^2) .- ky .* (FFTW.fft(u) ./ N^2))
    Zref = 0.5 * sum(abs2, ω̂)
    Test.@test Zref > 1.0                                    # non-trivial vorticity
    Z = FIT.calculate_band_energies(û_h, ks_h; bands = covering, invariant = FIT.Types.Enstrophy())
    Test.@test isapprox(sum(Z.energies), Zref; rtol = 1e-12)
end
