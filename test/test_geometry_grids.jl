# ---------------------------------------------------------------------------
# Coarse-graining on a caller-supplied FlowGeometries grid.
#
# The coordinate-vector form builds a 2-D/3-D `StructuredGrid`, reaching one of CGEF's `compute_Π!`
# methods. Passing a grid reaches the rest — curvilinear meshes and node sets, whose strain comes from
# a least-squares tangent-plane gradient.
#
# Each case is gated on a physical identity: a spatially constant velocity has zero strain, so its
# flux is zero. A grid built without adjacency also yields exactly zero — every least-squares gradient
# fits nothing — so non-triviality is asserted alongside, and the no-adjacency grid is checked to
# raise on that zero field.
# ---------------------------------------------------------------------------

using Test: Test
using Random: Random
using CoarseGrainingEnergyFluxes: CoarseGrainingEnergyFluxes   # loads the coarse-graining extension
using FlowGeometries: FlowGeometries as FG
using FlowInvariantTransfer: FlowInvariantTransfer as FIT

const _CGF_T = FIT.CoarseGrainingFlux

# ψ = sin2x sin3y + ½ sin5x cos y  →  (u, v) = (∂ψ/∂y, −∂ψ/∂x): divergence-free and multi-scale.
_gg_u(X, Y) =  3 .* sin.(2 .* X) .* cos.(3 .* Y) .- 0.5 .* sin.(5 .* X) .* sin.(Y)
_gg_v(X, Y) = -2 .* cos.(2 .* X) .* sin.(3 .* Y) .- 2.5 .* cos.(5 .* X) .* cos.(Y)

Test.@testset "coarse-graining on a FlowGeometries grid" begin
    L = 2π
    filt = FIT.Types.GaussianFilter()

    Test.@testset "2D structured: grid form matches the coordinate form" begin
        N = 48
        x = range(0, L; length = N + 1)[1:N]; y = copy(x)
        X = [xi for xi in x, _ in y]; Y = [yj for _ in x, yj in y]
        u = _gg_u(X, Y); v = _gg_v(X, Y)
        ℓ = 20 * (L / N)                       # ~20 cells across the filter window

        grid = FG.Grids.StructuredGrid(FG.Geometry.CartesianGeometry{Float64}(), x, y)
        Test.@test grid.mask isa FG.Grids.AllActive      # omitting the mask stores just a size

        r_coords = FIT.calculate_coarse_graining_flux((u, v), (x, y), ℓ, filt)
        r_grid = FIT.calculate_coarse_graining_flux((u, v), grid, ℓ, filt)
        Test.@test maximum(abs, r_coords.flux_field) > 1e-6
        Test.@test r_grid.flux_field == r_coords.flux_field

        ws = _CGF_T.CoarseGrainingFluxWorkspace((u, v), grid, ℓ, filt)
        r_ws = FIT.calculate_coarse_graining_flux!(ws, (u, v))
        Test.@test r_ws.flux_field == r_coords.flux_field
        FIT.calculate_coarse_graining_flux!(ws, (u, v))
        Test.@test (@allocated FIT.calculate_coarse_graining_flux!(ws, (u, v))) < 4096

        # A grid carries its own mask, so the keyword has no meaning alongside one.
        Test.@test_throws MethodError FIT.calculate_coarse_graining_flux((u, v), grid, ℓ, filt;
                                                                          mask = trues(N, N))
    end

    Test.@testset "3D structured" begin
        N = 24
        x = range(0, L; length = N + 1)[1:N]; y = copy(x); z = copy(x)
        X = [xi for xi in x, _ in y, _ in z]; Y = [yj for _ in x, yj in y, _ in z]
        Z = [zk for _ in x, _ in y, zk in z]
        u = _gg_u(X, Y) .* cos.(Z); v = _gg_v(X, Y) .* cos.(Z); w = sin.(X) .* sin.(Z)
        ℓ = 8 * (L / N)
        grid = FG.Grids.StructuredGrid(FG.Geometry.CartesianGeometry{Float64}(), x, y, z)
        r = FIT.calculate_coarse_graining_flux((u, v, w), grid, ℓ, filt)
        Test.@test maximum(abs, r.flux_field) > 1e-6
        rc = FIT.calculate_coarse_graining_flux(
            (fill(1.7, size(u)), fill(-0.3, size(u)), fill(0.9, size(u))), grid, ℓ, filt)
        Test.@test maximum(abs, rc.flux_field) < 1e-10        # uniform flow ⇒ zero strain ⇒ zero flux
    end

    Test.@testset "curvilinear (least-squares gradient)" begin
        N = 40
        s = range(0, L; length = N + 1)[1:N]
        # A smoothly distorted mesh: no tensor-product axis to difference along.
        X = [si + 0.10 * sin(tj) for si in s, tj in s]
        Y = [tj + 0.10 * sin(si) for si in s, tj in s]
        grid = FG.Grids.CurvilinearGrid(FG.Geometry.CartesianGeometry{Float64}(), X, Y)
        ℓ = 8 * (L / N)
        ws = _CGF_T.CoarseGrainingFluxWorkspace((_gg_u(X, Y), _gg_v(X, Y)), grid, ℓ, filt)
        Test.@test ws.deriv_plan isa FG.Operators.GradientPlan
        r = FIT.calculate_coarse_graining_flux!(ws, (_gg_u(X, Y), _gg_v(X, Y)))
        Test.@test maximum(abs, r.flux_field) > 1e-6
        rc = FIT.calculate_coarse_graining_flux((fill(1.7, size(X)), fill(-0.3, size(X))), grid, ℓ, filt)
        Test.@test maximum(abs, rc.flux_field) < 1e-8
    end

    Test.@testset "unstructured node set" begin
        Random.seed!(4); n = 3000
        x = L .* rand(n); y = L .* rand(n)
        area = fill(L^2 / n, n)
        geom = FG.Geometry.CartesianGeometry{Float64}()
        # k-nearest CSR adjacency; the k-d-tree builder is a NearestNeighbors extension hook.
        k = 12
        nbrs = Int[]; ptr = Vector{Int}(undef, n + 1); ptr[1] = 1
        d2 = Vector{Float64}(undef, n)
        for i in 1:n
            @inbounds for j in 1:n
                d2[j] = (x[j] - x[i])^2 + (y[j] - y[i])^2
            end
            d2[i] = Inf
            append!(nbrs, partialsortperm(d2, 1:k))
            ptr[i + 1] = ptr[i] + k
        end
        grid = FG.Grids.UnstructuredGrid(geom, (x, y), area, FG.Grids.AllActive((n,)), nbrs, ptr)
        ℓ = 6 * (L / sqrt(n))                       # ~6 mean spacings across the window
        r = FIT.calculate_coarse_graining_flux((_gg_u(x, y), _gg_v(x, y)), grid, ℓ, filt)
        Test.@test maximum(abs, r.flux_field) > 1e-6
        rc = FIT.calculate_coarse_graining_flux((fill(1.7, n), fill(-0.3, n)), grid, ℓ, filt)
        Test.@test maximum(abs, rc.flux_field) < 1e-8

        bare = FG.Grids.UnstructuredGrid(geom, (x, y), area)     # no adjacency
        Test.@test_throws ArgumentError FIT.calculate_coarse_graining_flux(
            (_gg_u(x, y), _gg_v(x, y)), bare, ℓ, filt)
    end
end
