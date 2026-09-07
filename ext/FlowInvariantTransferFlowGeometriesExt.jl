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
    FG.Grids.sampling(grid) isa FG.SphericalSampling.AbstractClenshawCurtisSampling || return false
    nlon, nlat = size(grid)
    return nlon == 2 * nlat - 1
end

# A spherical grid's directions are `(λ, φ)`, so a field on it is `(nlon, nlat)`; the exact transform
# indexes `(nlat, nlon)`. Verified equal node-for-node at `nlon = 2·nlat − 1`.
_to_fsh_layout(field, grid) = permutedims(reshape(field, size(grid)), (2, 1))

# Geographic node coordinates of any spherical grid as the colatitude/longitude vectors the scattered
# path takes, in the grid's own storage order so a field flattens alongside them, in the field's own
# element type.
function _sphere_points(grid, ::Type{FT}) where {FT}
    n = length(FG.Grids.mask(grid))
    θ = Vector{FT}(undef, n)
    φ = Vector{FT}(undef, n)
    k = 0
    for I in CartesianIndices(size(grid))
        λi, φi = FG.Grids.coords(grid, Tuple(I)...)
        k += 1
        θ[k] = FT(π / 2 - φi)                  # geographic latitude → colatitude
        φ[k] = FT(mod(λi, 2π))
    end
    return (θ, φ)
end

"""
    _sphere_quadrature(grid, lwork, FT) -> Vector or nothing

Per-node quadrature weights of `grid`, `Σw = 4π`, in the node order [`_sphere_points`](@ref) produces.

Analysis at degree `lwork` integrates an integrand of degree `2·lwork`, so the weights are returned
only for a sampling whose latitude rule is exact there —
`SphericalSampling.admits_exact_bandlimited_quadrature` with `lwork` inside the sampling's
`bandlimit`. Gauss–Legendre, Driscoll–Healy and reduced Gaussian qualify at their stated band limit;
Clenshaw–Curtis integrates products only to `lmax ≈ (nlat−1)/2` and qualifies below that.

`nothing` for every other node layout, whose coefficients come from the least-squares fit.
"""
function _sphere_quadrature(grid, lwork::Integer, ::Type{FT}) where {FT}
    s = FG.Grids.sampling(grid)
    s isa FG.SphericalSampling.AbstractTensorProductSphericalSampling || return nothing
    nlon, nlat = size(grid)
    _quadrature_exact(s, nlat, lwork) || return nothing
    q = FG.SphericalSampling.spherical_quadrature(FT, s, nlat; nlon = nlon)
    dλ = FT(2π) / nlon
    w = Vector{FT}(undef, nlon * nlat)
    k = 0
    for I in CartesianIndices((nlon, nlat))
        k += 1
        w[k] = q.w[I[2]] * dλ
    end
    return w
end

# Clenshaw–Curtis's own `bandlimit` is what its grid represents; its weights integrate the degree-2·l
# products analysis forms only for `2·lwork ≤ nlat − 1`.
function _quadrature_exact(s, nlat::Integer, lwork::Integer)
    FG.SphericalSampling.admits_exact_bandlimited_quadrature(s) &&
        return lwork <= FG.SphericalSampling.bandlimit(s, nlat)
    s isa FG.SphericalSampling.AbstractClenshawCurtisSampling && return 2 * lwork <= nlat - 1
    return false
end

_sphere_grid_check(grid) =
    FG.Grids.grid_geometry(grid) isa FG.Geometry.AbstractSphericalGeometry || throw(ArgumentError(
        "spherical transfer needs a grid on a spherical geometry; got " *
        "$(nameof(typeof(FG.Grids.grid_geometry(grid))))."))

"""
    calculate_energy_transfer(SphericalTransferMethod(), vorticity, grid; lmax, kwargs...)

Barotropic spherical transfer with the domain given as a `FlowGeometries` grid.

The grid's sampling picks the analysis it admits, in three cases:

  * a Clenshaw–Curtis sphere at `nlon = 2·nlat − 1` carries the nodes of the exact spherical-harmonic
    transform and takes it;
  * a sampling whose latitude rule integrates the degree-`2·lwork` integrand — Gauss–Legendre,
    Driscoll–Healy, reduced Gaussian, and Clenshaw–Curtis below `2·lwork ≤ nlat − 1` — has its
    coefficients by projection against the grid's own quadrature, one transform per analysis;
  * every other node layout has them from the least-squares fit.

The last two need `lmax`: the nodes alone fix no bandwidth.
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
        return FIT.calculate_energy_transfer(method, _to_fsh_layout(vorticity, grid); kwargs...)
    end
    lmax === nothing && throw(ArgumentError(
        "this grid's sampling ($(nameof(typeof(FG.Grids.sampling(grid))))) has no exact spherical " *
        "transform, so the transfer runs on its nodes — pass `lmax` for the bandwidth to analyse at."))
    FT = float(real(eltype(vorticity)))
    lwork = get(kwargs, :dealias, true) ? 2 * lmax : lmax
    return FIT.calculate_energy_transfer(method, vec(vorticity), _sphere_points(grid, FT);
                                         lmax = lmax,
                                         quadrature_weights = _sphere_quadrature(grid, lwork, FT),
                                         kwargs...)
end

"""
    calculate_energy_transfer(DivergentSphericalTransferMethod(), (u_θ, u_φ), grid; lmax, kwargs...)

Divergent (rotational + divergent) spherical KE transfer on a `FlowGeometries` grid, with the analysis
routed on the grid's sampling exactly as the barotropic method above.
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
            method, (_to_fsh_layout(velocity[1], grid), _to_fsh_layout(velocity[2], grid)); kwargs...)
    end
    lmax === nothing && throw(ArgumentError(
        "this grid's sampling ($(nameof(typeof(FG.Grids.sampling(grid))))) has no exact spherical " *
        "transform, so the transfer runs on its nodes — pass `lmax` for the bandwidth to analyse at."))
    FT = float(real(eltype(velocity[1])))
    lwork = get(kwargs, :dealias, true) ? 2 * lmax : lmax
    return FIT.calculate_energy_transfer(method, (vec(velocity[1]), vec(velocity[2])),
                                         _sphere_points(grid, FT); lmax = lmax,
                                         quadrature_weights = _sphere_quadrature(grid, lwork, FT),
                                         kwargs...)
end

# Physical → spectral on a grid is `FlowFieldSpectra.calculate_spectrum`, which selects the transform
# from the grid; the methods live in `FlowInvariantTransferFlowFieldSpectraExt` so they are defined once.

# ---------------------------------------------------------------------------
# Scattered Cartesian samples carried by a node grid.
#
# The NUFFT entries need the periodic box size `L` per direction, which the samples alone do not fix.
# A node grid declares its own period, so passing one supplies it.
# ---------------------------------------------------------------------------

function _scattered_points_and_period(grid::FG.Grids.AbstractGrid)
    FG.Grids.grid_geometry(grid) isa FG.Geometry.AbstractCartesianGeometry || throw(ArgumentError(
        "the scattered NUFFT entry is Cartesian; this grid is on " *
        "$(nameof(typeof(FG.Grids.grid_geometry(grid))))."))
    # A node set stores its samples as one flat vector, so its storage rank is 1 whatever the space it
    # samples; `ncoordinates` is the number of coordinate directions.
    N = FG.Grids.ncoordinates(grid)
    for d in 1:N
        FG.Grids.isperiodic(grid, d) || throw(ArgumentError(
            "direction $d of this grid is bounded, so it carries no period for the Fourier box. " *
            "Build it with `periodic`/`period` in that direction."))
    end
    coords = ntuple(d -> FG.Grids.coordinates(grid, d), N)
    Ls = ntuple(d -> FG.Grids.period(grid, d), N)
    return coords, Ls
end

"""
    to_spectral(velocity_fields, grid::FlowGeometries.Grids.AbstractGrid, ms::Tuple; spectral, tol)

Scattered-Cartesian NUFFT reconstruction with the sample points and the periodic box taken from a node
grid; see [`to_spectral`](@ref). `Ls` comes from the grid's own period.
"""
function FIT.to_spectral(velocity_fields::Tuple, grid::FG.Grids.AbstractGrid, ms::Tuple; kwargs...)
    coords, Ls = _scattered_points_and_period(grid)
    return FIT.to_spectral(velocity_fields, coords, ms; Ls = Ls, kwargs...)
end

"""
    NUFFTCoarseGrainingWorkspace(grid::FlowGeometries.Grids.AbstractGrid, ms; spectral, tol)

Scattered coarse-graining workspace with the sample points and the periodic box taken from a node
grid; see [`NUFFTCoarseGrainingWorkspace`](@ref).
"""
function FIT.NUFFTCoarseGrainingWorkspace(grid::FG.Grids.AbstractGrid, ms::Tuple; kwargs...)
    coords, Ls = _scattered_points_and_period(grid)
    return FIT.NUFFTCoarseGrainingWorkspace(coords, ms; Ls = Ls, kwargs...)
end

"""
    nufft_coarse_graining_flux(velocity_fields, grid, ℓ, filter, ms; spectral, tol)

Scattered coarse-graining flux `Π_ℓ(x)` on a node grid; see
[`nufft_coarse_graining_flux`](@ref). `Ls` comes from the grid's own period.
"""
function FIT.nufft_coarse_graining_flux(velocity_fields::Tuple, grid::FG.Grids.AbstractGrid,
                                        ℓ::Real, filter::FIT.Types.AbstractFilter, ms::Tuple; kwargs...)
    coords, Ls = _scattered_points_and_period(grid)
    return FIT.nufft_coarse_graining_flux(velocity_fields, coords, ℓ, filter, ms; Ls = Ls, kwargs...)
end

"""
    NUFFTToSpectralWorkspace(grid::FlowGeometries.Grids.AbstractGrid, ms; spectral, tol)

Scattered → uniform reconstruction workspace on a node grid; see
[`NUFFTToSpectralWorkspace`](@ref).
"""
function FIT.NUFFTToSpectralWorkspace(grid::FG.Grids.AbstractGrid, ms::Tuple; kwargs...)
    coords, Ls = _scattered_points_and_period(grid)
    return FIT.NUFFTToSpectralWorkspace(coords, ms; Ls = Ls, kwargs...)
end

end # module
