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
# those nodes for `AbstractFejer1Sampling` at `nlat = N`, where `nlon_for_nlat` is `2N−1`.
# That sampling at that shape is therefore the exact-transform case; every other sphere sampling —
# Gauss–Legendre, Driscoll–Healy, lat–lon,
# reduced Gaussian, HEALPix, cubed-sphere, icosahedral, Yin–Yang, scattered — goes to the scattered
# least-squares path on the grid's own node coordinates.
# ---------------------------------------------------------------------------

# `true` when this grid's nodes are the equiangular array the exact spherical transform is defined on.
_is_fsh_grid(grid) = false
function _is_fsh_grid(grid::FG.Grids.StructuredGrid{T, G, 2}) where {T, G <: FG.Geometry.AbstractSphericalGeometry{T}}
    FG.Grids.sampling(grid) isa FG.SphericalSampling.AbstractFejer1Sampling || return false
    nlon, nlat = size(grid)
    return nlon == 2 * nlat - 1
end

# The two analyses a grid's nodes admit: the exact transform on its equiangular array, or the scattered
# path on its node coordinates.
struct _FSHRoute end
struct _NodeRoute end

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
    _sphere_quadrature(grid, lmax, FT) -> Vector or nothing

Per-node quadrature weights of `grid`, `Σw = 4π`, in the node order [`_sphere_points`](@ref) produces:
the grid's measure, which on a tensor-product sampling is its quadrature, scaled to the unit sphere.

Analysis at degree `lmax` integrates an integrand of degree `2·lmax` (`f·conj(Y_lm)`, both of degree
`lmax`), so the weights are returned only for a tensor-product sampling exact there in both
directions: a latitude rule with `SphericalSampling.admits_exact_bandlimited_quadrature` and `lmax`
inside the sampling's `bandlimit` (Gauss–Legendre, Driscoll–Healy; Fejér's first rule integrates the
products only for `2·lmax ≤ nlat − 1`), and `nlon ≥ 2·lmax + 1` equispaced longitudes, which integrate
`e^{imλ}` exactly for `|m| ≤ 2·lmax`.

`nothing` for every other node layout, whose coefficients come from the least-squares fit weighted by
[`_sphere_measure`](@ref).
"""
function _sphere_quadrature(grid, lmax::Integer, ::Type{FT}) where {FT}
    s = FG.Grids.sampling(grid)
    s isa FG.SphericalSampling.AbstractTensorProductSphericalSampling || return nothing
    nlon, nlat = size(grid)
    nlon >= 2 * lmax + 1 || return nothing
    _quadrature_exact(s, nlat, lmax) || return nothing
    return _sphere_measure(grid, FT)
end

# The grid's measure per node, `Σ = 4π`, in the node order `_sphere_points` produces.
_sphere_measure(grid, ::Type{FT}) where {FT} = (m = FG.Grids.measure(grid); FT.(vec(m) .* (4π / sum(m))))

# The fit's point weights where no quadrature analyses the field: the cells' areas.
_fit_weights(grid, ::Nothing, ::Type{FT}) where {FT} = _sphere_measure(grid, FT)
_fit_weights(_, ::AbstractVector, ::Type) = nothing

# Fejér's first rule's own `bandlimit` is what its grid represents; its weights integrate the degree-2·lmax
# products analysis forms only for `2·lmax ≤ nlat − 1`.
function _quadrature_exact(s, nlat::Integer, lmax::Integer)
    FG.SphericalSampling.admits_exact_bandlimited_quadrature(s) &&
        return lmax <= FG.SphericalSampling.bandlimit(s, nlat)
    s isa FG.SphericalSampling.AbstractFejer1Sampling && return 2 * lmax <= nlat - 1
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
# on a grid whose mask is `AllActive`.
_node_mask(grid) = _node_mask(FG.Grids.mask(grid))
_node_mask(::FG.Grids.AllActive) = nothing
_node_mask(m::AbstractArray{Bool}) = vec(m)

# The degree the fit `Exclude` runs at: the exact transform's grid fixes it at `nlat − 1`.
_fit_degree(::_FSHRoute, grid, _) = size(grid, 2) - 1
_fit_degree(::_NodeRoute, _, lmax) = lmax

_spin(::Tuple{Any}) = 0
_spin(::Tuple{Any,Any}) = 1
_fit_type(::Tuple{Any}, ::Type{T}) where {T} = T
_fit_type(::Tuple{Any,Any}, ::Type{T}) where {T} = Complex{T}

_active_workspace(::FIT.Types.AbstractInactiveCells, ::Nothing, fields, coords, lmax; kwargs...) = nothing
_active_workspace(::FIT.Types.ZeroFill, active::AbstractVector, fields, coords, lmax; kwargs...) = nothing
function _active_workspace(::FIT.Types.Exclude, active::AbstractVector, fields, coords, lmax; kwargs...)
    T = float(real(eltype(first(fields))))
    return FIT.Spherical.ActiveSphericalFit(_fit_type(fields, T), _spin(fields), coords, findall(active), lmax;
                                            kwargs...)
end

_transfer_workspace(::Tuple{Any}, ::_FSHRoute, T, coords, lmax; kwargs...) =
    FIT.Spherical.SphericalTransferWorkspace(T, lmax; kwargs...)
_transfer_workspace(::Tuple{Any,Any}, ::_FSHRoute, T, coords, lmax; kwargs...) =
    FIT.Spherical.DivergentSphericalTransferWorkspace(T, lmax; kwargs...)
_transfer_workspace(::Tuple{Any}, ::_NodeRoute, T, coords, lmax; kwargs...) =
    FIT.Spherical.ScatteredSphericalTransferWorkspace(T, coords, lmax; kwargs...)
_transfer_workspace(::Tuple{Any,Any}, ::_NodeRoute, T, coords, lmax; kwargs...) =
    FIT.Spherical.ScatteredDivergentSphericalTransferWorkspace(T, coords, lmax; kwargs...)

function _build_transfer(fields, route::_FSHRoute, T, coords, grid, lmax; tol, rtol, maxiter, nufft,
                         kwargs...)
    return _transfer_workspace(fields, route, T, coords, lmax; kwargs...)
end
function _build_transfer(fields, route::_NodeRoute, T, coords, grid, lmax;
                          radius, dealias, tol, rtol, maxiter, nufft, execution)
    qw = _sphere_quadrature(grid, lmax, T)
    return _transfer_workspace(fields, route, T, coords, lmax; quadrature_weights = qw,
        weights = _fit_weights(grid, qw, T), radius, dealias, tol, rtol, maxiter, nufft, execution)
end

_work_fields(::_NodeRoute, fields, grid) = map(vec, fields)
_work_fields(::_FSHRoute, fields, grid) = map(f -> similar(f, reverse(size(grid))), fields)

function _grid_workspace(fields, grid, lmax, inactive;
                          radius::Real = 1.0, dealias::Bool = true, tol::Real = 1e-10,
                          rtol::Union{Nothing,Real} = nothing, maxiter::Integer = 4000,
                          nufft::FIT.SpectralBackends.AbstractSpectralBackend = FIT.SpectralBackends.AutoSpectralBackend(),
                          execution::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend(),
                          transform = nothing)
    _sphere_grid_check(grid)
    route = _is_fsh_grid(grid) ? _FSHRoute() : _NodeRoute()
    route isa _NodeRoute && _require_lmax(grid, lmax)
    path = route isa _FSHRoute ? :regular : :scattered
    exec = FIT.Types.resolve_execution(execution, first(fields))
    FIT.Spherical._validate_spherical_backends(transform, exec, path)
    degree = _fit_degree(route, grid, lmax)
    T = float(real(eltype(first(fields))))
    coords = _sphere_points(grid, T)
    active = _node_mask(grid)
    fitkw = (; tol, rtol, maxiter, nufft, execution = exec)
    fit = _active_workspace(inactive, active, fields, coords, degree; fitkw...)
    try
        transfer = _build_transfer(fields, route, T, coords, grid, degree;
                                    radius, dealias, fitkw...)
        try
            staged = map(fields) do field
                length(field) == length(grid) || throw(DimensionMismatch("field and grid sizes differ"))
                proto = FIT.Types._to_execution(exec, similar(field, T, 0))
                similar(proto, T, size(grid))
            end
            mask = active === nothing ? nothing : copyto!(similar(first(staged), Bool, length(active)), active)
            return FIT.Spherical.SphericalGridWorkspace(transfer, staged, _work_fields(route, staged, grid),
                                                         mask, fit, route)
        catch
            _close_transfer(route, transfer)
            rethrow()
        end
    catch
        fit === nothing || FIT.close!(fit)
        rethrow()
    end
end

function FIT.Spherical.SphericalTransferWorkspace(field::AbstractArray, grid::FG.Grids.AbstractGrid;
                                                   lmax = nothing, inactive::FIT.Types.AbstractInactiveCells = FIT.Types.ZeroFill(),
                                                   kwargs...)
    return _grid_workspace((field,), grid, lmax, inactive; kwargs...)
end

function FIT.Spherical.DivergentSphericalTransferWorkspace(fields::Tuple, grid::FG.Grids.AbstractGrid;
                                                            lmax = nothing, inactive::FIT.Types.AbstractInactiveCells = FIT.Types.ZeroFill(),
                                                            kwargs...)
    return _grid_workspace(fields, grid, lmax, inactive; kwargs...)
end

_mask_field!(field, ::Nothing) = field
function _mask_field!(field, mask::AbstractVector)
    values = vec(field)
    @. values = ifelse(mask, values, zero(eltype(values)))
    return field
end

function _fit_input!(out, fields::Tuple{Any})
    copyto!(out, vec(fields[1]))
    return out
end
function _fit_input!(out, fields::Tuple{Any,Any})
    a, b = map(vec, fields)
    @. out = complex(a, b)
    return out
end

function _fit_fields!(fields::Tuple{Any}, out)
    vec(fields[1]) .= real.(out)
end
function _fit_fields!(fields::Tuple{Any,Any}, out)
    vec(fields[1]) .= real.(out)
    vec(fields[2]) .= imag.(out)
end

function _prepare_masked!(fields, mask, ::Nothing)
    foreach(field -> _mask_field!(field, mask), fields)
    return nothing
end
function _prepare_masked!(fields, mask, fit::FIT.Spherical.ActiveSphericalFit)
    _fit_input!(fit.output, fields)
    out, outcome = FIT.Spherical.active_fit!(fit, fit.output)
    _fit_fields!(fields, out)
    return outcome
end

_prepare_work!(::_NodeRoute, work, fields) = nothing
_prepare_work!(::_FSHRoute, work, fields) = foreach((out, field) -> permutedims!(out, field, (2, 1)), work, fields)
_apply_transfer(ws, fields::Tuple{Any}) = FIT.calculate_spherical_transfer!(ws, fields[1])
_apply_transfer(ws, fields::Tuple{Any,Any}) = FIT.calculate_divergent_spherical_transfer!(ws, fields...)

function _grid_transfer!(ws::FIT.Spherical.SphericalGridWorkspace, fields)
    length(fields) == length(ws.fields) || throw(DimensionMismatch("field component counts differ"))
    foreach((out, field) -> copyto!(out, reshape(field, size(out))), ws.fields, fields)
    fit = _prepare_masked!(ws.fields, ws.mask, ws.active_fit)
    _prepare_work!(ws.route, ws.work, ws.fields)
    return FIT.Spherical.with_merged_fit(_apply_transfer(ws.transfer, ws.work), fit)
end

FIT.calculate_spherical_transfer!(ws::FIT.Spherical.SphericalGridWorkspace, field::AbstractArray) =
    _grid_transfer!(ws, (field,))
FIT.calculate_divergent_spherical_transfer!(ws::FIT.Spherical.SphericalGridWorkspace, a::AbstractArray, b::AbstractArray) =
    _grid_transfer!(ws, (a, b))

_close_transfer(::_FSHRoute, transfer) = nothing
_close_transfer(::_NodeRoute, transfer) = FIT.close!(transfer)
function FIT.close!(ws::FIT.Spherical.SphericalGridWorkspace)
    _close_transfer(ws.route, ws.transfer)
    ws.active_fit === nothing || FIT.close!(ws.active_fit)
    return ws
end

function _grid_transfer(method, fields::Tuple, grid, lmax, inactive, kwargs)
    ws = _grid_workspace(fields, grid, lmax, inactive; radius = method.radius, kwargs...)
    try
        return _grid_transfer!(ws, fields)
    finally
        FIT.close!(ws)
    end
end

"""
    calculate_energy_transfer(SphericalTransferMethod(), vorticity, grid; lmax, inactive = ZeroFill(),
                              kwargs...)

Barotropic spherical transfer with the domain given as a `FlowGeometries` grid.

The grid's sampling picks the analysis it admits, in three cases:

  * a Fejér first-rule sphere at `nlon = 2·nlat − 1` carries the nodes of the exact spherical-harmonic
    transform and takes it;
  * a tensor-product sampling whose rule integrates the degree-`2·lmax` integrand — a latitude rule
    exact there (Gauss–Legendre, Driscoll–Healy, and Fejér's first rule below `2·lmax ≤ nlat − 1`) and
    `nlon ≥ 2·lmax + 1` longitudes — has its coefficients by projection against the grid's own
    quadrature, one transform;
  * every other node layout has them from the least-squares fit, each node weighted by its cell's area.

The last two need `lmax`: the nodes alone fix no bandwidth.

`inactive` sets what the grid's inactive cells hold, whatever the field stores there (`NaN` included):
[`Types.ZeroFill`](@ref FlowInvariantTransfer.Types.ZeroFill) gives the transfer of the field extended
by zero, and [`Types.Exclude`](@ref FlowInvariantTransfer.Types.Exclude) that of the degree-`lmax`
least-squares fit to the active cells alone (`lmax = nlat − 1` on the exact-transform grid), run on
NUFSHT with the keywords `tol`, `rtol`, `maxiter`, `nufft` and `execution`; the result's `converged`,
`iterations` and `residual` include that fit's.
"""
FIT.calculate_energy_transfer(
    method::FIT.Types.SphericalTransferMethod,
    vorticity::AbstractArray,
    grid::FG.Grids.AbstractGrid;
    lmax::Union{Nothing, Integer} = nothing,
    inactive::FIT.Types.AbstractInactiveCells = FIT.Types.ZeroFill(),
    kwargs...,
) = _grid_transfer(method, (vorticity,), grid, lmax, inactive, kwargs)

"""
    calculate_energy_transfer(DivergentSphericalTransferMethod(), (u_θ, u_φ), grid; lmax,
                              inactive = ZeroFill(), kwargs...)

Divergent (rotational + divergent) spherical KE transfer on a `FlowGeometries` grid, with the analysis
routed on the grid's sampling and the inactive cells read by `inactive` exactly as the barotropic method
above; the fit `Exclude` runs is that of the spin-1 field `u_θ + i u_φ`.
"""
FIT.calculate_energy_transfer(
    method::FIT.Types.DivergentSphericalTransferMethod,
    velocity::Tuple{<:AbstractArray, <:AbstractArray},
    grid::FG.Grids.AbstractGrid;
    lmax::Union{Nothing, Integer} = nothing,
    inactive::FIT.Types.AbstractInactiveCells = FIT.Types.ZeroFill(),
    kwargs...,
) = _grid_transfer(method, velocity, grid, lmax, inactive, kwargs)

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
    # samples; its coordinates are one vector per coordinate direction.
    coords = FG.Grids.coordinates(grid)
    for d in eachindex(coords)
        FG.Grids.isperiodic(grid, d) || throw(ArgumentError(
            "direction $d of this grid is bounded, so it carries no period for the Fourier box. " *
            "Build it with `topology = FlowGeometries.Grids.Periodic()` and `period` in that direction."))
    end
    Ls = ntuple(d -> FG.Grids.period(grid, d), Val(length(coords)))
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
