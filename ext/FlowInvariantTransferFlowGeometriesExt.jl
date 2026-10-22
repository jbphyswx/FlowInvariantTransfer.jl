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
# those nodes for `AbstractClenshawCurtisSampling` at `nlat = N`, where `nlon_for_nlat` is `2N−1`.
# That sampling at that shape is therefore the exact-transform case; every other sphere sampling —
# Gauss–Legendre, Driscoll–Healy, lat–lon,
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
# indexes `(nlat, nlon)`; the two node sets agree node for node at `nlon = 2·nlat − 1`.
_to_fsh_layout(field, grid) = permutedims(reshape(field, size(grid)), (2, 1))

# Geographic node coordinates of any spherical grid as the colatitude/longitude vectors the scattered
# path takes, in the grid's own storage order so a field flattens alongside them, in the field's own
# element type.
function _sphere_points(grid, ::Type{FT}) where {FT}
    λ, φ = FG.Grids.materialize(grid)
    return (FT.(π / 2 .- vec(φ)), FT.(mod.(vec(λ), 2π)))
end

"""
    _sphere_quadrature(grid, lwork, FT) -> Vector or nothing

Per-node quadrature weights of `grid`, `Σw = 4π`, in the node order [`_sphere_points`](@ref) produces:
the grid's measure, which on a tensor-product sampling is its quadrature, scaled to the unit sphere.

Analysis at degree `lwork` integrates an integrand of degree `2·lwork` (`f·conj(Y_lm)`, both of degree
`lwork`), so the weights are returned only for a tensor-product sampling exact there in both
directions: a latitude rule with `SphericalSampling.admits_exact_bandlimited_quadrature` and `lwork`
inside the sampling's `bandlimit` (Gauss–Legendre, Driscoll–Healy; Clenshaw–Curtis integrates products
only to `lmax ≈ (nlat−1)/2` and qualifies below that), and `nlon ≥ 2·lwork + 1` equispaced longitudes,
which integrate `e^{imλ}` exactly for `|m| ≤ 2·lwork`.

`nothing` for every other node layout, whose coefficients come from the least-squares fit.
"""
function _sphere_quadrature(grid, lwork::Integer, ::Type{FT}) where {FT}
    s = FG.Grids.sampling(grid)
    s isa FG.SphericalSampling.AbstractTensorProductSphericalSampling || return nothing
    nlon, nlat = size(grid)
    nlon >= 2 * lwork + 1 || return nothing
    _quadrature_exact(s, nlat, lwork) || return nothing
    m = FG.Grids.measure(grid)
    return FT.(vec(m) .* (4π / sum(m)))
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

_require_lmax(grid, lmax) = lmax === nothing && throw(ArgumentError(
    "this grid's sampling ($(nameof(typeof(FG.Grids.sampling(grid))))) has no exact spherical " *
    "transform, so the transfer runs on its nodes — pass `lmax` for the bandwidth to analyse at."))

# ---------------------------------------------------------------------------
# Inactive nodes. The routes below analyse every node of the grid, so the fields reach them with the
# inactive nodes' values replaced as `inactive` says: zero under `ZeroFill`, the active nodes'
# least-squares fit under `Exclude`.
# ---------------------------------------------------------------------------

# The grid's mask as one `Bool` per node, in the order `_sphere_points` lists the nodes, or `nothing`
# when every node is active.
_node_mask(grid) = _node_mask(FG.Grids.mask(grid))
_node_mask(::FG.Grids.AllActive) = nothing
_node_mask(m::AbstractArray{Bool}) = all(m) ? nothing : vec(m)

# The keywords of the fit `Exclude` runs, which are the scattered transfer's own.
const _FIT_KEYWORDS = (:tol, :rtol, :maxiter, :nufft, :execution)

# `fields` (the vorticity at spin `s = 0`, or the velocity `(u_θ, u_φ)` read as `u_θ + i u_φ` at `s = 1`)
# in their own shapes, the outcome of the fit `Exclude` runs (`nothing` where none runs), and the
# keywords the route takes. The exact transform takes no fit keywords; where a fit runs, they go to it.
function _masked_input(fields::Tuple, grid, s::Integer, fsh::Bool, lmax, inactive::FIT.Types.AbstractInactiveCells,
                       kwargs)
    kw = values(kwargs)
    active = _node_mask(grid)
    fits = inactive isa FIT.Types.Exclude && active !== nothing
    route = (fsh && fits) ? Base.structdiff(kw, NamedTuple{(:tol, :rtol, :maxiter, :nufft)}) : kw
    fitkw = NamedTuple{filter(in(_FIT_KEYWORDS), keys(kw))}(kw)
    filled, fit = _masked_fields(active, inactive, fields, grid, s, fsh ? size(grid, 2) - 1 : lmax, fitkw)
    return filled, fit, route
end

_masked_fields(::Nothing, ::FIT.Types.AbstractInactiveCells, fields, grid, s, lmax, fitkw) = (fields, nothing)

_masked_fields(active::AbstractVector{Bool}, ::FIT.Types.ZeroFill, fields, grid, s, lmax, fitkw) =
    (map(f -> reshape(ifelse.(active, vec(f), zero(eltype(f))), size(f)), fields), nothing)

function _masked_fields(active::AbstractVector{Bool}, ::FIT.Types.Exclude, fields, grid, s, lmax, fitkw)
    FT = float(real(eltype(fields[1])))
    values = s == 0 ? vec(fields[1]) : complex.(vec(fields[1]), vec(fields[2]))
    field, fit = FIT.Spherical.active_fit(values, s, _sphere_points(grid, FT), findall(active), lmax; fitkw...)
    sz = size(fields[1])
    filled = s == 0 ? (reshape(real.(field), sz),) : (reshape(real.(field), sz), reshape(imag.(field), sz))
    return filled, fit
end

"""
    calculate_energy_transfer(SphericalTransferMethod(), vorticity, grid; lmax, inactive = ZeroFill(),
                              kwargs...)

Barotropic spherical transfer with the domain given as a `FlowGeometries` grid.

The grid's sampling picks the analysis it admits, in three cases:

  * a Clenshaw–Curtis sphere at `nlon = 2·nlat − 1` carries the nodes of the exact spherical-harmonic
    transform and takes it;
  * a tensor-product sampling whose rule integrates the degree-`2·lwork` integrand — a latitude rule
    exact there (Gauss–Legendre, Driscoll–Healy, and Clenshaw–Curtis below `2·lwork ≤ nlat − 1`) and
    `nlon ≥ 2·lwork + 1` longitudes — has its coefficients by projection against the grid's own
    quadrature, one transform per analysis;
  * every other node layout has them from the least-squares fit.

The last two need `lmax`: the nodes alone fix no bandwidth.

`inactive` sets what the grid's inactive cells hold, whatever the field stores there (`NaN` included):
[`Types.ZeroFill`](@ref FlowInvariantTransfer.Types.ZeroFill) gives the transfer of the field extended
by zero, and [`Types.Exclude`](@ref FlowInvariantTransfer.Types.Exclude) that of the degree-`lmax`
least-squares fit to the active cells alone (`lmax = nlat − 1` on the exact-transform grid), run on
NUFSHT with the keywords `tol`, `rtol`, `maxiter`, `nufft` and `execution`; the result's `converged`,
`iterations` and `residual` include that fit's.
"""
function FIT.calculate_energy_transfer(
    method::FIT.Types.SphericalTransferMethod,
    vorticity::AbstractArray,
    grid::FG.Grids.AbstractGrid;
    lmax::Union{Nothing, Integer} = nothing,
    inactive::FIT.Types.AbstractInactiveCells = FIT.Types.ZeroFill(),
    kwargs...,
)
    _sphere_grid_check(grid)
    fsh = _is_fsh_grid(grid)
    fsh || _require_lmax(grid, lmax)
    (ζ,), fit, route = _masked_input((vorticity,), grid, 0, fsh, lmax, inactive, kwargs)
    res = if fsh
        FIT.calculate_energy_transfer(method, _to_fsh_layout(ζ, grid); route...)
    else
        FT = float(real(eltype(ζ)))
        lwork = get(route, :dealias, true) ? 2 * lmax : lmax
        FIT.calculate_energy_transfer(method, vec(ζ), _sphere_points(grid, FT); lmax = lmax,
                                      quadrature_weights = _sphere_quadrature(grid, lwork, FT), route...)
    end
    return FIT.Spherical.with_merged_fit(res, fit)
end

"""
    calculate_energy_transfer(DivergentSphericalTransferMethod(), (u_θ, u_φ), grid; lmax,
                              inactive = ZeroFill(), kwargs...)

Divergent (rotational + divergent) spherical KE transfer on a `FlowGeometries` grid, with the analysis
routed on the grid's sampling and the inactive cells read by `inactive` exactly as the barotropic method
above; the fit `Exclude` runs is that of the spin-1 field `u_θ + i u_φ`.
"""
function FIT.calculate_energy_transfer(
    method::FIT.Types.DivergentSphericalTransferMethod,
    velocity::Tuple{<:AbstractArray, <:AbstractArray},
    grid::FG.Grids.AbstractGrid;
    lmax::Union{Nothing, Integer} = nothing,
    inactive::FIT.Types.AbstractInactiveCells = FIT.Types.ZeroFill(),
    kwargs...,
)
    _sphere_grid_check(grid)
    fsh = _is_fsh_grid(grid)
    fsh || _require_lmax(grid, lmax)
    (uθ, uφ), fit, route = _masked_input(velocity, grid, 1, fsh, lmax, inactive, kwargs)
    res = if fsh
        FIT.calculate_energy_transfer(method, (_to_fsh_layout(uθ, grid), _to_fsh_layout(uφ, grid)); route...)
    else
        FT = float(real(eltype(uθ)))
        lwork = get(route, :dealias, true) ? 2 * lmax : lmax
        FIT.calculate_energy_transfer(method, (vec(uθ), vec(uφ)), _sphere_points(grid, FT); lmax = lmax,
                                      quadrature_weights = _sphere_quadrature(grid, lwork, FT), route...)
    end
    return FIT.Spherical.with_merged_fit(res, fit)
end

# Physical → spectral on a grid (`FlowFieldSpectra.calculate_spectrum`) lives in
# `FlowInvariantTransferFlowFieldSpectraExt`.

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
    to_spectral(velocity_fields, grid::FlowGeometries.Grids.AbstractGrid, ms::Tuple; transform, tol)

Scattered-Cartesian NUFFT reconstruction with the sample points and the periodic box taken from a node
grid; see [`to_spectral`](@ref). `Ls` comes from the grid's own period.
"""
function FIT.to_spectral(velocity_fields::Tuple, grid::FG.Grids.AbstractGrid, ms::Tuple; kwargs...)
    coords, Ls = _scattered_points_and_period(grid)
    return FIT.to_spectral(velocity_fields, coords, ms; Ls = Ls, kwargs...)
end

"""
    NUFFTCoarseGrainingWorkspace(grid::FlowGeometries.Grids.AbstractGrid, ms; transform, tol)

Scattered coarse-graining workspace with the sample points and the periodic box taken from a node
grid; see [`NUFFTCoarseGrainingWorkspace`](@ref).
"""
function FIT.NUFFTCoarseGrainingWorkspace(grid::FG.Grids.AbstractGrid, ms::Tuple; kwargs...)
    coords, Ls = _scattered_points_and_period(grid)
    return FIT.NUFFTCoarseGrainingWorkspace(coords, ms; Ls = Ls, kwargs...)
end

"""
    nufft_coarse_graining_flux(velocity_fields, grid, ℓ, filter, ms; transform, tol)

Scattered coarse-graining flux `Π_ℓ(x)` on a node grid; see
[`nufft_coarse_graining_flux`](@ref). `Ls` comes from the grid's own period.
"""
function FIT.nufft_coarse_graining_flux(velocity_fields::Tuple, grid::FG.Grids.AbstractGrid,
                                        ℓ::Real, filter::FIT.Types.AbstractFilter, ms::Tuple; kwargs...)
    coords, Ls = _scattered_points_and_period(grid)
    return FIT.nufft_coarse_graining_flux(velocity_fields, coords, ℓ, filter, ms; Ls = Ls, kwargs...)
end

"""
    NUFFTToSpectralWorkspace(grid::FlowGeometries.Grids.AbstractGrid, ms; transform, tol)

Scattered → uniform reconstruction workspace on a node grid; see
[`NUFFTToSpectralWorkspace`](@ref).
"""
function FIT.NUFFTToSpectralWorkspace(grid::FG.Grids.AbstractGrid, ms::Tuple; kwargs...)
    coords, Ls = _scattered_points_and_period(grid)
    return FIT.NUFFTToSpectralWorkspace(coords, ms; Ls = Ls, kwargs...)
end

end # module
