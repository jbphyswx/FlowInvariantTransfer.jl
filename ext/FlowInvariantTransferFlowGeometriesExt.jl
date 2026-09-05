module FlowInvariantTransferFlowGeometriesExt

using FlowGeometries: FlowGeometries as FG
using FlowInvariantTransfer: FlowInvariantTransfer as FIT
using ComputationalBackends: ComputationalBackends

# ---------------------------------------------------------------------------
# Spherical transfer from a FlowGeometries grid.
#
# The two spherical implementations take different inputs — FastSphericalHarmonics an exact
# equiangular colatitude–longitude array, NUFSHT arbitrary `(θ, φ)` samples — and the grid's own
# sampling says which one its nodes admit. Routing here reads that sampling and hands the field over
# in the form the chosen path takes.
#
# `FSH.sph_points(N)` is `θᵢ = π(i−½)/N`, `λⱼ = 2π(j−1)/(2N−1)`. `FlowGeometries` builds exactly
# those nodes for `AbstractClenshawCurtisSampling` at `nlat = N`, where `nlon_for_nlat` is `2N−1`
# (`Sampling/TensorProduct.jl:43`, `Sampling/Types.jl:322`). That sampling at that shape is therefore
# the exact-transform case; every other sphere sampling — Gauss–Legendre, Driscoll–Healy, lat–lon,
# reduced Gaussian, HEALPix, cubed-sphere, icosahedral, Yin–Yang, scattered — goes to the scattered
# least-squares path on the grid's own node coordinates.
# ---------------------------------------------------------------------------

# `true` when this grid's nodes are the equiangular array the exact spherical transform is defined on.
_is_fsh_grid(grid) = false
function _is_fsh_grid(grid::FG.Grids.StructuredGrid{T, G, 2}) where {T, G <: FG.Geometry.AbstractSphericalGeometry{T}}
    FG.Grids.sampling(grid) isa FG.Sampling.AbstractClenshawCurtisSampling || return false
    nlon, nlat = size(grid)
    return nlon == 2 * nlat - 1
end

# Geographic node coordinates of any spherical grid as the colatitude/longitude vectors the scattered
# path takes, in the grid's own storage order so a field flattens alongside them.
function _sphere_points(grid)
    n = length(FG.Grids.mask(grid))
    θ = Vector{Float64}(undef, n)
    φ = Vector{Float64}(undef, n)
    k = 0
    for I in CartesianIndices(size(grid))
        λi, φi = FG.Grids.coords(grid, Tuple(I)...)
        k += 1
        θ[k] = π / 2 - φi                      # geographic latitude → colatitude
        φ[k] = mod(λi, 2π)
    end
    return (θ, φ)
end

_sphere_grid_check(grid) =
    FG.Grids.grid_geometry(grid) isa FG.Geometry.AbstractSphericalGeometry || throw(ArgumentError(
        "spherical transfer needs a grid on a spherical geometry; got " *
        "$(nameof(typeof(FG.Grids.grid_geometry(grid))))."))

"""
    calculate_energy_transfer(SphericalTransferMethod(), vorticity, grid; lmax, kwargs...)

Barotropic spherical transfer with the domain given as a `FlowGeometries` grid. A Clenshaw–Curtis
sphere at `nlon = 2·nlat − 1` carries the nodes of the exact spherical-harmonic transform and takes
it; any other sphere sampling takes the scattered path on the grid's own nodes, where `lmax` is
required (the point set does not fix a bandwidth).
"""
function FIT.calculate_energy_transfer(
    method::FIT.Types.SphericalTransferMethod,
    vorticity::AbstractArray,
    grid::FG.Grids.AbstractGrid;
    lmax::Union{Nothing, Integer} = nothing,
    kwargs...,
)
    _sphere_grid_check(grid)
    if _is_fsh_grid(grid)
        return FIT.calculate_energy_transfer(method, reshape(vorticity, size(grid)); kwargs...)
    end
    lmax === nothing && throw(ArgumentError(
        "this grid's sampling ($(nameof(typeof(FG.Grids.sampling(grid))))) has no exact spherical " *
        "transform, so the transfer runs on its nodes as a scattered least-squares fit — pass `lmax` " *
        "for the bandwidth to solve for."))
    return FIT.calculate_energy_transfer(method, vec(vorticity), _sphere_points(grid);
                                         lmax = lmax, kwargs...)
end

"""
    calculate_energy_transfer(DivergentSphericalTransferMethod(), (u_θ, u_φ), grid; lmax, kwargs...)

Divergent (rotational + divergent) spherical KE transfer on a `FlowGeometries` grid, routed the same
way as the barotropic method above.
"""
function FIT.calculate_energy_transfer(
    method::FIT.Types.DivergentSphericalTransferMethod,
    velocity::Tuple{<:AbstractArray, <:AbstractArray},
    grid::FG.Grids.AbstractGrid;
    lmax::Union{Nothing, Integer} = nothing,
    kwargs...,
)
    _sphere_grid_check(grid)
    if _is_fsh_grid(grid)
        return FIT.calculate_energy_transfer(
            method, (reshape(velocity[1], size(grid)), reshape(velocity[2], size(grid))); kwargs...)
    end
    lmax === nothing && throw(ArgumentError(
        "this grid's sampling ($(nameof(typeof(FG.Grids.sampling(grid))))) has no exact spherical " *
        "transform, so the transfer runs on its nodes as a scattered least-squares fit — pass `lmax` " *
        "for the bandwidth to solve for."))
    return FIT.calculate_energy_transfer(method, (vec(velocity[1]), vec(velocity[2])),
                                         _sphere_points(grid); lmax = lmax, kwargs...)
end

end # module
