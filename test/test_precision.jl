# ---------------------------------------------------------------------------
# Single-precision inputs give the double-precision result to single precision: the uniform-grid front
# door with Float32 coordinates, the triadic decomposition, and the spherical transfer on the regular grid.
# ---------------------------------------------------------------------------

using Test: Test
using Random: Random
using FFTW: FFTW
using FlowFieldSpectra: FlowFieldSpectra
using FastSphericalHarmonics: FastSphericalHarmonics as FSH
using FlowInvariantTransfer: FlowInvariantTransfer as FIT
using SpectralBackends: SpectralBackends

Test.@testset "single precision" begin
    Test.@testset "Float32 coordinates on the uniform-grid front door" begin
        N = 16
        x = Float32.(collect(range(0, 2π; length = N + 1)[1:N]))
        rng = Random.MersenneTwister(2)
        u = randn(rng, Float32, N, N); v = randn(rng, Float32, N, N)
        û, ks = FIT.to_spectral((u, v), (x, x))
        Test.@test maximum(abs, û[:, :, 1] .- FFTW.rfft(u) ./ N^2) < 1f-6 * maximum(abs, û)
        Test.@test isapprox(collect(ks[1]), FFTW.rfftfreq(N, N); rtol = 1f-5)
        # The uniformity check holds a stretched axis to the coordinates' own precision.
        xs = x .+ 0.01f0 .* sin.(x)
        Test.@test_throws ArgumentError FIT.Utils.validate_uniform_grid((xs, x))
        Test.@test FIT.Utils.domain_size_from_coords((x,))[1] ≈ Float32(2π)
    end

    Test.@testset "triadic decomposition: Float32 direct sum" begin
        rng = Random.MersenneTwister(9)
        X = randn(rng, 128, 1, 3)
        r64 = FIT.TriadicOrthogonalDecomposition.triadic_orthogonal_decomposition(X; window = 32,
                  transform = SpectralBackends.FFTSpectralBackend())
        r32 = FIT.TriadicOrthogonalDecomposition.triadic_orthogonal_decomposition(Float32.(X); window = 32,
                  transform = SpectralBackends.DirectSumSpectralBackend())
        a = filter(!isnan, r64.mode_bispectrum); b = filter(!isnan, r32.mode_bispectrum)
        Test.@test maximum(abs, a .- b) < 1e-4 * maximum(a)
    end

    Test.@testset "spherical transfer on the regular grid returns the field's precision" begin
        lmax = 8; N = lmax + 1
        rng = Random.MersenneTwister(6)
        C = zeros(N, 2N - 1)
        for l in 1:lmax, m in -l:l
            C[FSH.sph_mode(l, m)] = randn(rng) / l
        end
        ζ = FSH.sph_evaluate(C)
        r64 = FIT.calculate_energy_transfer(FIT.Types.SphericalTransferMethod(), ζ)
        r32 = FIT.calculate_energy_transfer(FIT.Types.SphericalTransferMethod(), Float32.(ζ))
        s = maximum(abs, r64.energy_transfer)
        Test.@test s > 1e-6
        Test.@test maximum(abs, r32.energy_transfer .- r64.energy_transfer) < 1e-5 * s
    end
end
