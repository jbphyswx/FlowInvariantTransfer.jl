# ---------------------------------------------------------------------------
# Spherical transfer from a FlowGeometries grid.
#
# The two spherical implementations take different inputs, and the grid's sampling decides which its
# nodes admit: a Clenshaw–Curtis sphere at `nlon = 2·nlat − 1` carries exactly the nodes the
# spherical-harmonic transform is defined on, so it takes the exact path; every other sampling takes
# the scattered least-squares path on the grid's own nodes.
#
# The first testset asserts that node identity. Each routed result is then compared to the shape-based
# entry it must reproduce.
# ---------------------------------------------------------------------------

using Test: Test
using Random: Random
using FlowGeometries: FlowGeometries as FG
using FFTW: FFTW                               # the uniform `to_spectral` transform
using FlowFieldSpectra: FlowFieldSpectra       # `to_spectral` on a grid routes through its plans
using FastSphericalHarmonics: FastSphericalHarmonics as FSH
using NonuniformFFTs: NonuniformFFTs           # the scattered NUFFT provider
using NUFSHT: NUFSHT
using ComputationalBackends: ComputationalBackends
using FlowInvariantTransfer: FlowInvariantTransfer as FIT

const _SS = FG.SphericalSampling

# `@allocated` at a testset's top level measures the boxing of the testset's globals as well as the
# call. Measured through a barrier, its arguments are typed locals and only the call is counted.
_alloc_to_spectral!(ws, fields) = @allocated FIT.to_spectral!(ws, fields)

# A band-limited vorticity field on the FSH grid.
function _sph_field(N)
    Random.seed!(8)
    C = zeros(Float64, N, 2N - 1)
    for l in 1:min(N - 1, 5), m in -l:l
        C[FSH.spinsph_mode(0, l, m)] = randn() / l^2
    end
    return FSH.spinsph_evaluate(C, 0)          # (nlat, nlon)
end

Test.@testset "spherical transfer on a FlowGeometries grid" begin
    N = 12
    ax = _SS.spherical_axes(_SS.ClenshawCurtisSampling(), N)
    grid = FG.Grids.StructuredGrid(FG.Geometry.SphericalGeometry(1.0), ax.λ, ax.φ;
                                   sampling = _SS.ClenshawCurtisSampling())

    Test.@testset "Clenshaw–Curtis nodes are the exact-transform nodes" begin
        θfsh, φfsh = FSH.sph_points(N)
        Test.@test size(grid) == (2N - 1, N)                     # (nlon, nlat)
        lats = [FG.Grids.coords(grid, 1, j)[2] for j in 1:N]
        lons = [FG.Grids.coords(grid, i, 1)[1] for i in 1:(2N - 1)]
        # `π/2 − lat` is a subtraction, so it lands on the FSH colatitude to round-off.
        Test.@test maximum(abs, (π / 2 .- lats) .- collect(θfsh)) < 8 * eps(Float64)
        Test.@test maximum(abs, mod.(lons, 2π) .- collect(φfsh)) == 0.0
    end

    ζ_fsh = _sph_field(N)                                        # (nlat, nlon)
    ζ_grid = permutedims(ζ_fsh, (2, 1))                          # the grid's own (nlon, nlat)

    Test.@testset "exact path reproduces the shape-based entry" begin
        ref = FIT.calculate_energy_transfer(FIT.Types.SphericalTransferMethod(1.0), ζ_fsh)
        got = FIT.calculate_energy_transfer(FIT.Types.SphericalTransferMethod(1.0), ζ_grid, grid)
        Test.@test maximum(abs, ref.energy_transfer) > 1e-10      # the reference carries signal
        Test.@test got.energy_transfer == ref.energy_transfer
        Test.@test got.enstrophy_transfer == ref.enstrophy_transfer
    end

    Test.@testset "a non-quadrature sampling routes to the scattered path" begin
        # A lat–lon sphere has no exact transform, so its nodes go to the least-squares fit. Enough
        # points for the dealiased degree-2·lmax solve: M ≥ (2·lmax+1)².
        lmax = 4
        nlat = 2 * (2 * lmax + 1); nlon = 2 * nlat
        axll = _SS.spherical_axes(_SS.LatLonSampling(), nlat; nlon = nlon)
        gll = FG.Grids.StructuredGrid(FG.Geometry.SphericalGeometry(1.0), axll.λ, axll.φ;
                                      sampling = _SS.LatLonSampling())
        Test.@test !(FG.Grids.sampling(gll) isa _SS.AbstractClenshawCurtisSampling)

        # Band-limited field sampled at those nodes, evaluated from its own coefficients.
        Random.seed!(3)
        Cs = zeros(ComplexF64, lmax + 1, 2lmax + 1)
        for l in 1:lmax, m in -l:l
            Cs[NUFSHT.spin_coeff_index(l, m, lmax)] = (randn() + im * randn()) / l^2
        end
        θ = Float64[]; φ = Float64[]
        for I in CartesianIndices(size(gll))
            λi, φi = FG.Grids.coords(gll, Tuple(I)...)
            push!(θ, π / 2 - φi); push!(φ, mod(λi, 2π))
        end
        plan = NUFSHT.make_spin_plan(ComplexF64, θ, φ, lmax, 0; tol = 1e-10)
        fbuf = zeros(ComplexF64, length(θ))
        NUFSHT.nusht_type2_spin!(fbuf, Cs, plan)
        ζv = real.(fbuf)

        # Passing the grid must agree with handing the same nodes to the scattered entry directly.
        ref = FIT.calculate_energy_transfer(FIT.Types.SphericalTransferMethod(1.0), ζv, (θ, φ);
                                            lmax = lmax)
        got = FIT.calculate_energy_transfer(FIT.Types.SphericalTransferMethod(1.0),
                                            reshape(ζv, size(gll)), gll; lmax = lmax)
        Test.@test maximum(abs, ref.energy_transfer) > 1e-12
        Test.@test got.energy_transfer == ref.energy_transfer

        # Without a bandwidth the scattered route has nothing to solve for, and says so.
        Test.@test_throws ArgumentError FIT.calculate_energy_transfer(
            FIT.Types.SphericalTransferMethod(1.0), reshape(ζv, size(gll)), gll)
    end

    # A sampling whose latitude rule integrates the degree-2·lwork integrand determines the
    # coefficients by projection, so analysis is one weighted adjoint transform. The result must be
    # the transfer the least-squares fit on the same nodes computes.
    Test.@testset "a quadrature sampling analyses by projection" begin
        lmaxq = 6
        nlatq = 16
        sq = _SS.GaussLegendreSampling()
        Test.@test _SS.admits_exact_bandlimited_quadrature(sq)
        Test.@test 2 * lmaxq <= _SS.bandlimit(sq, nlatq)          # the rule is exact at the work degree
        axq = _SS.spherical_axes(sq, nlatq)
        gq = FG.Grids.StructuredGrid(FG.Geometry.SphericalGeometry(1.0), axq.λ, axq.φ; sampling = sq)
        nlonq, nlq = size(gq)
        Mq = nlonq * nlq

        θq = Float64[]; φq = Float64[]
        for I in CartesianIndices(size(gq))
            λi, φi = FG.Grids.coords(gq, Tuple(I)...)
            push!(θq, π / 2 - φi); push!(φq, mod(λi, 2π))
        end

        Random.seed!(23)
        Cq = zeros(ComplexF64, lmaxq + 1, 2lmaxq + 1)
        for l in 1:lmaxq, m in -l:l
            Cq[NUFSHT.spin_coeff_index(l, m, lmaxq)] = (randn() + im * randn()) / (l + 1)^2
        end
        pq = NUFSHT.make_spin_plan(ComplexF64, θq, φq, lmaxq, 0; tol = 1e-12)
        fq = zeros(ComplexF64, Mq)
        NUFSHT.nusht_type2_spin!(fq, Cq, pq)
        ζq = real.(fq)

        mth = FIT.Types.SphericalTransferMethod(1.0)
        quad = FIT.calculate_energy_transfer(mth, reshape(ζq, nlonq, nlq), gq; lmax = lmaxq)
        solv = FIT.calculate_energy_transfer(mth, ζq, (θq, φq); lmax = lmaxq)

        sE = maximum(abs, solv.energy_transfer)
        sZ = maximum(abs, solv.enstrophy_transfer)
        Test.@test sE > 1e-12                                      # the reference carries signal
        Test.@test maximum(abs, quad.energy_transfer .- solv.energy_transfer) < 1e-6 * sE
        Test.@test maximum(abs, quad.enstrophy_transfer .- solv.enstrophy_transfer) < 1e-6 * sZ
        # Σ_l T = 0 is the identity the transfer conserves, and the projection meets it.
        Test.@test abs(sum(quad.energy_transfer)) < 1e-6 * sE
        Test.@test abs(sum(quad.enstrophy_transfer)) < 1e-6 * sZ
    end

    Test.@testset "a Cartesian grid is refused" begin
        cg = FG.Grids.StructuredGrid(FG.Geometry.CartesianGeometry{Float64}(),
                                     range(0, 1; length = 4), range(0, 1; length = 4))
        Test.@test_throws ArgumentError FIT.calculate_energy_transfer(
            FIT.Types.SphericalTransferMethod(1.0), zeros(4, 4), cg)
    end
end

# ---------------------------------------------------------------------------
# Physical → spectral with the domain given as a grid: the transform's preconditions come from the
# grid, and the result is what the coordinate-vector form gives on the same axes.
# ---------------------------------------------------------------------------
Test.@testset "to_spectral on a FlowGeometries grid" begin
    N = 16; L = 2π
    x = range(0, L; length = N + 1)[1:N]; y = copy(x)
    Random.seed!(21)
    u = randn(N, N); v = randn(N, N)
    cart = FG.Geometry.CartesianGeometry{Float64}()
    grid = FG.Grids.StructuredGrid(cart, x, y; topology = (true, true))

    ûg, ksg = FIT.to_spectral((u, v), grid)
    ûc, ksc = FIT.to_spectral((u, v), (x, y))
    # The grid form is gated against `rfft(u)/Nᴰ` itself. It takes the grid's declared period, where the
    # coordinate form reconstructs `N·Δ` from the endpoints and lands an ulp low, so the two forms agree
    # to round-off.
    Test.@test maximum(abs, ûg[:, :, 1] .- FFTW.rfft(u) ./ N^2) == 0.0
    Test.@test maximum(abs, ûg[:, :, 2] .- FFTW.rfft(v) ./ N^2) == 0.0
    Test.@test maximum(abs, ûg .- ûc) < 1e-14 * maximum(abs, ûc)
    Test.@test all(maximum(abs, collect(ksg[d]) .- collect(ksc[d])) < 1e-12 for d in 1:2)
    Test.@test FIT.SpectralLayout.is_half(ksg)
    Test.@test FIT.SpectralLayout.full_size(ksg) == (N, N)

    ws = FIT.ToSpectralWorkspace((u, v), grid)
    Test.@test FIT.to_spectral!(ws, (u, v))[1] == ûg
    _alloc_to_spectral!(ws, (u, v))                       # warm
    Test.@test _alloc_to_spectral!(ws, (u, v)) == 0
    # A threaded plan spawns a task per thread per execution, so that path carries a floor and is
    # asserted as a bound.
    wst = FIT.ToSpectralWorkspace((u, v), grid; execution = ComputationalBackends.ThreadedBackend())
    Test.@test FIT.to_spectral!(wst, (u, v))[1] ≈ ûg
    _alloc_to_spectral!(wst, (u, v))
    Test.@test _alloc_to_spectral!(wst, (u, v)) < 4096

    # A bounded direction carries no Fourier period, and the property is named.
    gb = FG.Grids.StructuredGrid(cart, x, y; topology = (true, false))
    Test.@test_throws ArgumentError FIT.to_spectral((u, v), gb)

    # A stretched axis is a transform this grid still admits — the composite one — so it runs.
    xs = L .* (range(0, 1; length = N + 1)[1:N]) .^ 1.3
    gs = FG.Grids.StructuredGrid(cart, xs, y; topology = (true, true), period = (L, L))
    ûs, _ = FIT.to_spectral((u, v), gs)
    Test.@test size(ûs) == (N ÷ 2 + 1, N, 2)
    Test.@test all(isfinite, ûs)

    # A spherical grid is not this transform's domain.
    ax = _SS.spherical_axes(_SS.ClenshawCurtisSampling(), 8)
    gsph = FG.Grids.StructuredGrid(FG.Geometry.SphericalGeometry(1.0), ax.λ, ax.φ;
                                   sampling = _SS.ClenshawCurtisSampling())
    Test.@test_throws ArgumentError FIT.to_spectral((zeros(15, 8), zeros(15, 8)), gsph)
end

# ---------------------------------------------------------------------------
# Scattered Cartesian samples carried by a node grid: the points and the periodic box both come from
# the grid, so `Ls` is not passed by hand.
# ---------------------------------------------------------------------------
Test.@testset "scattered NUFFT entries on a node grid" begin
    L = 2π; n = 700
    Random.seed!(31)
    xs = L .* rand(n); ys = L .* rand(n)
    u = sin.(xs) .* cos.(ys); v = cos.(xs) .* sin.(ys)
    geom = FG.Geometry.CartesianGeometry{Float64}()
    grid = FG.Grids.UnstructuredGrid(geom, (xs, ys), fill(L^2 / n, n);
                                     periodic = (true, true), period = (L, L))
    ms = (12, 12)

    ûg, ksg = FIT.to_spectral((u, v), grid, ms; spectral = FIT.Types.NonuniformFFTsBackend(), tol = 1e-12)
    ûc, ksc = FIT.to_spectral((u, v), (xs, ys), ms; spectral = FIT.Types.NonuniformFFTsBackend(),
                              Ls = (L, L), tol = 1e-12)
    # NonuniformFFTs threads its transforms off `Threads.nthreads()`, and threaded spreading is not
    # bitwise reproducible across plan instances: measured 0.0 at -t1 and ~4e-16 here.
    _nuf_tol(scale) = Threads.nthreads() == 1 ? 0.0 : 1e-12 * scale
    Test.@test maximum(abs, ûg .- ûc) <= _nuf_tol(maximum(abs, ûc))
    Test.@test all(collect(ksg[d]) == collect(ksc[d]) for d in 1:2)

    ℓ = 0.8
    rg = FIT.nufft_coarse_graining_flux((u, v), grid, ℓ, FIT.Types.GaussianFilter(), ms;
                                        spectral = FIT.Types.NonuniformFFTsBackend())
    rc = FIT.nufft_coarse_graining_flux((u, v), (xs, ys), ℓ, FIT.Types.GaussianFilter(), ms;
                                        spectral = FIT.Types.NonuniformFFTsBackend(), Ls = (L, L))
    Test.@test maximum(abs, rc.flux_field) > 1e-10
    Test.@test maximum(abs, rg.flux_field .- rc.flux_field) <= _nuf_tol(maximum(abs, rc.flux_field))

    # A bounded node grid carries no period for the Fourier box.
    gb = FG.Grids.UnstructuredGrid(geom, (xs, ys), fill(L^2 / n, n))
    Test.@test_throws ArgumentError FIT.to_spectral((u, v), gb, ms;
                                                     spectral = FIT.Types.NonuniformFFTsBackend())
end
