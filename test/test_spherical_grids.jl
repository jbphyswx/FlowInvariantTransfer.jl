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
using NonuniformFFTs: NonuniformFFTs           # the scattered NUFFT library
using FlowTransformBindings: FlowTransformBindings as FTB
using NUFSHT: NUFSHT
using ComputationalBackends: ComputationalBackends
using FlowInvariantTransfer: FlowInvariantTransfer as FIT

const _SS = FG.SphericalSampling

# `@allocated` at a testset's top level measures the boxing of the testset's globals as well as the
# call. Measured through a barrier, its arguments are typed locals and only the call is counted.
_alloc_to_spectral!(ws, fields) = @allocated FIT.to_spectral!(ws, fields)

# A band-limited vorticity field on the FSH grid.
function _sph_field(N, seed = 8)
    Random.seed!(seed)
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
        Test.@test solv.converged
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
# A masked sphere. The inactive cells hold `NaN`; `inactive` decides what stands there. Under
# `ZeroFill` the result is the transfer of the zero-extended field on the unmasked grid, the same
# computation on the same numbers. Under `Exclude` it is the transfer, on the unmasked grid, of the
# least-squares fit to the active nodes, built here with NUFSHT directly.
# ---------------------------------------------------------------------------

# The colatitudes and longitudes of a spherical grid's nodes, in the order its fields and mask flatten.
function _grid_points(g)
    θ = Float64[]; φ = Float64[]
    for I in CartesianIndices(size(g))
        λi, φi = FG.Grids.coords(g, Tuple(I)...)
        push!(θ, π / 2 - φi); push!(φ, mod(λi, 2π))
    end
    return θ, φ
end

# The spin-`s`, degree-`L` least-squares fit to `f` over the nodes `act`, evaluated at every node, and
# whether the fit converged.
function _fit_on_active(f, s, θ, φ, act, L)
    pa = NUFSHT.make_spin_plan(ComplexF64, θ[act], φ[act], L, s; tol = 1e-12)
    pf = NUFSHT.make_spin_plan(ComplexF64, θ, φ, L, s; tol = 1e-12)
    C = zeros(ComplexF64, L + 1, 2L + 1)
    r = NUFSHT.nusht_analyze!(C, ComplexF64.(f[act]), pa; ws = FTB.LSMRWorkspace(pa), rtol = 1e-11,
                              maxiter = 20_000)
    out = zeros(ComplexF64, length(θ))
    NUFSHT.nusht_type2_spin!(out, C, pf)
    NUFSHT.close!(pa); NUFSHT.close!(pf)
    return out, r.converged
end

_transfers(r::FIT.Types.SphericalTransferResult) = (r.energy_transfer, r.enstrophy_transfer)
_transfers(r::FIT.Types.DivergentSphericalTransferResult) =
    (r.energy_transfer, r.rotational_transfer, r.divergent_transfer)

# The barotropic method takes the vorticity, the divergent one the velocity pair.
_input(fields::Tuple{Any}) = fields[1]
_input(fields::Tuple{Any, Any}) = fields

# Both strategies on `gm`, the grid `grid` under the mask `m`, against their references on `grid`. `L`
# is the degree of the fit over the active nodes; `kw` the route's keywords.
function _check_masked(mth, fields, grid, gm, m, L; kw...)
    s = length(fields) == 1 ? 0 : 1
    nan_in = map(f -> ifelse.(m, f, NaN), fields)

    ref0 = FIT.calculate_energy_transfer(mth, _input(map(f -> ifelse.(m, f, 0.0), fields)), grid; kw...)
    got0 = FIT.calculate_energy_transfer(mth, _input(nan_in), gm; kw...)
    for (a, b) in zip(_transfers(got0), _transfers(ref0))
        Test.@test maximum(abs, b) > 1e-10                       # the reference carries signal
        Test.@test a == b
    end

    θ, φ = _grid_points(grid)
    values = s == 0 ? vec(fields[1]) : complex.(vec(fields[1]), vec(fields[2]))
    fit, ok = _fit_on_active(values, s, θ, φ, findall(vec(m)), L)
    Test.@test ok
    sz = size(fields[1])
    fitted = s == 0 ? (reshape(real.(fit), sz),) : (reshape(real.(fit), sz), reshape(imag.(fit), sz))
    refd = FIT.calculate_energy_transfer(mth, _input(fitted), grid; kw...)
    gotd = FIT.calculate_energy_transfer(mth, _input(nan_in), gm; inactive = FIT.Types.Exclude(), kw...)
    Test.@test gotd.converged
    for (a, b) in zip(_transfers(gotd), _transfers(refd))
        sc = maximum(abs, b)
        Test.@test sc > 1e-10
        Test.@test maximum(abs, a .- b) < 1e-6 * sc
    end
    return nothing
end

Test.@testset "masked sphere: inactive cells under ZeroFill and Exclude" begin
    sph = FG.Geometry.SphericalGeometry(1.0)
    baro = FIT.Types.SphericalTransferMethod(1.0)
    divg = FIT.Types.DivergentSphericalTransferMethod(1.0)
    rng = Random.MersenneTwister(57)
    # A smooth field with a component no finite degree represents.
    smooth(θ, φ, k) = sin(k * θ) * cos((k - 1) * φ) + 0.4 * cos(θ) + 0.05 * randn(rng)

    Test.@testset "exact-transform grid" begin
        N = 12
        cc = _SS.ClenshawCurtisSampling()
        ax = _SS.spherical_axes(cc, N)
        grid = FG.Grids.StructuredGrid(sph, ax.λ, ax.φ; sampling = cc)
        m = trues(size(grid)); m[5:8, 4:6] .= false
        gm = FG.Grids.StructuredGrid(sph, ax.λ, ax.φ; sampling = cc, mask = m)
        ζ = permutedims(_sph_field(N), (2, 1)) .+ 0.05 .* randn(rng, size(grid))
        _check_masked(baro, (ζ,), grid, gm, m, N - 1)
        uθ = permutedims(_sph_field(N, 9), (2, 1)) .+ 0.05 .* randn(rng, size(grid))
        uφ = permutedims(_sph_field(N, 10), (2, 1)) .+ 0.05 .* randn(rng, size(grid))
        _check_masked(divg, (uθ, uφ), grid, gm, m, N - 1)
    end

    Test.@testset "least-squares grid (lat–lon)" begin
        lmax = 4
        nlat = 2 * (2 * lmax + 1); nlon = 2 * nlat
        ll = _SS.LatLonSampling()
        ax = _SS.spherical_axes(ll, nlat; nlon = nlon)
        grid = FG.Grids.StructuredGrid(sph, ax.λ, ax.φ; sampling = ll)
        m = trues(size(grid)); m[5:10, 6:9] .= false
        gm = FG.Grids.StructuredGrid(sph, ax.λ, ax.φ; sampling = ll, mask = m)
        θ, φ = _grid_points(grid)
        ζ = reshape([smooth(θ[j], φ[j], 3) for j in eachindex(θ)], size(grid))
        _check_masked(baro, (ζ,), grid, gm, m, lmax; lmax = lmax)
    end

    Test.@testset "quadrature grid (Gauss–Legendre)" begin
        lmax = 6; nlat = 16
        gl = _SS.GaussLegendreSampling()
        ax = _SS.spherical_axes(gl, nlat)
        grid = FG.Grids.StructuredGrid(sph, ax.λ, ax.φ; sampling = gl)
        m = trues(size(grid)); m[7:11, 5:8] .= false
        gm = FG.Grids.StructuredGrid(sph, ax.λ, ax.φ; sampling = gl, mask = m)
        θ, φ = _grid_points(grid)
        ζ = reshape([smooth(θ[j], φ[j], 3) for j in eachindex(θ)], size(grid))
        _check_masked(baro, (ζ,), grid, gm, m, lmax; lmax = lmax)
        uθ = reshape([smooth(θ[j], φ[j], 2) for j in eachindex(θ)], size(grid))
        uφ = reshape([smooth(θ[j], φ[j] + 1.0, 3) for j in eachindex(θ)], size(grid))
        _check_masked(divg, (uθ, uφ), grid, gm, m, lmax; lmax = lmax)
    end
end

# ---------------------------------------------------------------------------
# Physical → spectral with the domain given as a grid: the transform's preconditions come from the
# grid, and the result is `rfft(u)/Nᴰ` on the grid's half-layout wavenumbers.
# ---------------------------------------------------------------------------
Test.@testset "to_spectral on a FlowGeometries grid" begin
    N = 16; L = 2π
    x = range(0, L; length = N + 1)[1:N]; y = copy(x)
    Random.seed!(21)
    u = randn(N, N); v = randn(N, N)
    cart = FG.Geometry.CartesianGeometry{Float64}()
    grid = FG.Grids.StructuredGrid(cart, x, y; topology = (true, true))

    ûg, ksg = FIT.to_spectral((u, v), grid)
    ksc = FIT.Utils.wavenumber_grid((N, N), (L, L); real = true)
    Test.@test maximum(abs, ûg[:, :, 1] .- FFTW.rfft(u) ./ N^2) == 0.0
    Test.@test maximum(abs, ûg[:, :, 2] .- FFTW.rfft(v) ./ N^2) == 0.0
    Test.@test all(maximum(abs, collect(ksg[d]) .- collect(ksc[d])) < 1e-12 for d in 1:2)

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

    # A bounded direction carries no Fourier period.
    gb = FG.Grids.StructuredGrid(cart, x, y; topology = (true, false))
    Test.@test_throws ArgumentError FIT.to_spectral((u, v), gb)

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

    ûg, ksg = FIT.to_spectral((u, v), grid, ms; transform = FTB.NonuniformFFTsBackend(), tol = 1e-12)
    ûc, ksc = FIT.to_spectral((u, v), (xs, ys), ms; transform = FTB.NonuniformFFTsBackend(),
                              Ls = (L, L), tol = 1e-12)
    # Both run their transforms on one thread, so the two plan instances agree bit for bit.
    Test.@test maximum(abs, ûg .- ûc) == 0
    Test.@test all(collect(ksg[d]) == collect(ksc[d]) for d in 1:2)

    ℓ = 0.8
    rg = FIT.nufft_coarse_graining_flux((u, v), grid, ℓ, FIT.Types.GaussianFilter(), ms;
                                        transform = FTB.NonuniformFFTsBackend())
    rc = FIT.nufft_coarse_graining_flux((u, v), (xs, ys), ℓ, FIT.Types.GaussianFilter(), ms;
                                        transform = FTB.NonuniformFFTsBackend(), Ls = (L, L))
    Test.@test maximum(abs, rc.flux_field) > 1e-10
    Test.@test maximum(abs, rg.flux_field .- rc.flux_field) == 0

    # A bounded node grid carries no period for the Fourier box.
    gb = FG.Grids.UnstructuredGrid(geom, (xs, ys), fill(L^2 / n, n))
    Test.@test_throws ArgumentError FIT.to_spectral((u, v), gb, ms;
                                                     transform = FTB.NonuniformFFTsBackend())
end
