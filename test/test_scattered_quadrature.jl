# ---------------------------------------------------------------------------
# Scattered analysis as a quadrature: `û(k) = (1/∏L) Σⱼ wⱼ uⱼ e^{-ik·xⱼ}`.
#
# On clustered points the weights are solved here to integrate every `e^{iq·x}` with `|q_d| ≤ 8`
# exactly, the least-norm solution of `Σⱼ wⱼ e^{iq·xⱼ} = 4π² δ_q`. A field band-limited to `|m_d| ≤ 2`
# then has its coefficients on the `9 × 9` mode grid exactly, and those are the FFT of its values on the
# uniform `9 × 9` grid. On the same points the uniform-weight estimate `type1(u)/N` misses them by more
# than 1e-2.
# ---------------------------------------------------------------------------

using Test: Test
using Random: Random
using LinearAlgebra: LinearAlgebra
using FFTW: FFTW
using FINUFFT: FINUFFT
using NonuniformFFTs: NonuniformFFTs
using FlowTransformBindings: FlowTransformBindings as FTB
using FlowInvariantTransfer: FlowInvariantTransfer as FIT

Test.@testset "scattered analysis with quadrature weights" begin
    L = 2π; ms = (9, 9)
    rng = Random.MersenneTwister(4)
    # Two thirds of the points in the left half of the box; a point at the origin, the phase reference.
    x = vcat(0.0, π .* rand(rng, 500), π .+ π .* rand(rng, 249))
    y = vcat(0.0, L .* rand(rng, 749))
    M = length(x)
    Q = 8
    qs = vec([(a, c) for a in -Q:Q, c in -Q:Q])
    A = [cis(q[1] * x[j] + q[2] * y[j]) for q in qs, j in 1:M]
    rhs = [q == (0, 0) ? complex(L^2) : 0.0im for q in qs]
    w = real.(A' * ((A * A') \ rhs))
    Test.@test maximum(abs, A * w .- rhs) < 1e-9 * L^2               # the rule is exact to |q| ≤ 8

    f(a, c) = sin(a) * cos(c) + 0.3 * cos(2a + c) - 0.2 * sin(2c)
    g(a, c) = cos(a) * sin(c) - 0.4 * sin(a - 2c)
    xg = range(0, L; length = 10)[1:9]
    ref = FFTW.fft([f(a, c) for a in xg, c in xg]) ./ 81
    u = f.(x, y); v = g.(x, y)

    for lib in (FTB.FINUFFTBackend(), FTB.NonuniformFFTsBackend())
        ûw, ks = FIT.to_spectral((u,), (x, y), ms; transform = lib, Ls = (L, L), weights = w, tol = 1e-12)
        Test.@test maximum(abs, ûw[:, :, 1] .- ref) < 1e-9
        ûe, _ = FIT.to_spectral((u,), (x, y), ms; transform = lib, Ls = (L, L), tol = 1e-12)
        Test.@test maximum(abs, ûe[:, :, 1] .- ref) > 1e-2
        Test.@test_throws DimensionMismatch FIT.to_spectral((u,), (x, y), ms; transform = lib, Ls = (L, L),
                                                            weights = w[1:end-1])
    end

    # The coarse-graining flux with the same weights against the weighted direct sums.
    ℓ = 0.9; filt = FIT.Types.GaussianFilter()
    m1 = [a <= 4 ? a : a - 9 for a in 0:8]
    Gh = [FIT.Filters.filter_response(filt, sqrt(m1[a]^2 + m1[b]^2), ℓ) for a in 1:9, b in 1:9]
    t1(h) = [sum(w[j] * h[j] * cis(-(m1[a] * x[j] + m1[b] * y[j])) for j in 1:M) / L^2 for a in 1:9, b in 1:9]
    t2(S) = [real(sum(S[a, b] * cis(m1[a] * x[l] + m1[b] * y[l]) for a in 1:9, b in 1:9)) for l in 1:M]
    ūs = Gh .* t1(u); v̄s = Gh .* t1(v)
    ū = t2(ūs); v̄ = t2(v̄s)
    τxx = t2(Gh .* t1(u .* u)) .- ū .* ū
    τyy = t2(Gh .* t1(v .* v)) .- v̄ .* v̄
    τxy = t2(Gh .* t1(u .* v)) .- ū .* v̄
    Sxx = t2([im * m1[a] * ūs[a, b] for a in 1:9, b in 1:9])
    Syy = t2([im * m1[b] * v̄s[a, b] for a in 1:9, b in 1:9])
    Sxy = 0.5 .* (t2([im * m1[b] * ūs[a, b] for a in 1:9, b in 1:9]) .+ t2([im * m1[a] * v̄s[a, b] for a in 1:9, b in 1:9]))
    Πref = -(τxx .* Sxx .+ τyy .* Syy .+ 2 .* τxy .* Sxy)
    Test.@test maximum(abs, Πref) > 1e-4
    for lib in (FTB.FINUFFTBackend(), FTB.NonuniformFFTsBackend())
        r = FIT.nufft_coarse_graining_flux((u, v), (x, y), ℓ, filt, ms; transform = lib, Ls = (L, L),
                                           weights = w, tol = 1e-12)
        Test.@test maximum(abs, r.flux_field .- Πref) < 1e-9 * maximum(abs, Πref)
        Test.@test r.mean_flux ≈ sum(w .* Πref) / sum(w) rtol = 1e-9
    end
end
