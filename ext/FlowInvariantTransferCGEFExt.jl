module FlowInvariantTransferCGEFExt

using CoarseGrainingEnergyFluxes: CoarseGrainingEnergyFluxes as CGEF
using FlowInvariantTransfer: FlowInvariantTransfer as FIT

# ---------------------------------------------------------------------------
# Filter type mapping: FIT.AbstractFilter → CGEF.AbstractFilterKernel
# ---------------------------------------------------------------------------

_to_cgef_kernel(::FIT.Types.GaussianFilter)      = CGEF.Kernels.GaussianKernel()
_to_cgef_kernel(::FIT.Types.TopHatFilter)        = CGEF.Kernels.TopHatKernel()
_to_cgef_kernel(::FIT.Types.SharpSpectralFilter) = CGEF.Kernels.SharpSpectralKernel()

# The vertical velocity of a three-component field, `nothing` for two components.
_vertical(fields::Tuple{Any,Any,Any}) = fields[3]
_vertical(::Tuple{Any,Any}) = nothing

# ---------------------------------------------------------------------------
# Override CoarseGrainingFlux._cg_flux_cgef
# ---------------------------------------------------------------------------

# The weights of CGEF's domain mean: each cell of the grid the filter writes (`output_grid`, the whole
# grid under `ZeroFill`) contributes its area, over the area of the active cells (Storer et al. 2022).
function _mean_weights(grid, filter_plan, ::Type{FT}, ns) where {FT}
    og = CGEF.Diagnostics.output_grid(grid, filter_plan)
    total = CGEF.Diagnostics.active_area(grid)
    w = zeros(FT, ns...)
    for (i, I) in enumerate(CartesianIndices(CGEF.FlowGeometries.Grids.size_tuple(og)))
        t = Tuple(I)
        CGEF.FlowGeometries.Grids.isactive(og, t...) && (w[i] = FT(CGEF.FlowGeometries.Grids.area(og, t...) / total))
    end
    return w
end

# Σ Π·w: an allocation-free loop over host arrays, one reduction in the arrays' own memory otherwise.
function _weighted_sum(Π::Array{FT}, w::Array) where {FT}
    acc = zero(FT)
    @inbounds for i in eachindex(Π, w)
        acc += Π[i] * w[i]
    end
    return acc
end
_weighted_sum(Π::AbstractArray{FT}, w::AbstractArray) where {FT} = mapreduce(*, +, Π, w; init = zero(FT))

"""
    _cg_flux_workspace(velocity_fields, coords_vecs, ℓ, filter; mask=nothing, return_diagnostics=false,
                       mask_strategy=ZeroFill(), vector_filter=nothing, execution=AutoBackend(), radius=nothing)

Build the reusable CGEF `StructuredGrid`, its `ΠWorkspace`, the derivative + filter plans (fixed for
the given `(filter, scale ℓ, mask_strategy, execution)`), the `Π_ℓ(x)` output buffer, and (when
`return_diagnostics`) the `τ̄`/`S̄` diagnostic buffers, wrapped in a `CoarseGrainingFluxWorkspace`.
Supports 2D/3D Cartesian grids (or a lon–lat sphere with `radius`) from tuple inputs. `vector_filter` is
CGEF's filter of vectors and tensors, `nothing` for the one `CGEF.Filtering.resolve_vector_filter`
resolves for the grid and `mask_strategy`.
"""
function FIT.CoarseGrainingFlux._cg_flux_workspace(
    velocity_fields::Tuple,
    coords_vecs::Tuple,
    ℓ::Real,
    filter::FIT.Types.AbstractFilter;
    return_diagnostics::Bool = false,
    mask::Union{Nothing, AbstractArray{Bool}} = nothing,
    radius::Union{Nothing, Real} = nothing,
    execution::CGEF.ComputationalBackends.AbstractExecutionBackend = CGEF.ComputationalBackends.AutoBackend(),
    mask_strategy::CGEF.Filtering.AbstractMaskStrategy = CGEF.Filtering.ZeroFill(),
    vector_filter::Union{Nothing, CGEF.Filtering.AbstractVectorFilter} = nothing,
)
    D  = length(velocity_fields)
    nd = length(coords_vecs)
    # Three components on two coordinates are the thin-layer (2.5-D) flow of one horizontal layer.
    (D == nd || (nd == 2 && D == 3)) || throw(ArgumentError(
        "got $D velocity components on $nd coordinates; pass one per coordinate, or three on a " *
        "two-dimensional grid for the thin-layer flux"))

    FT  = eltype(velocity_fields[1])

    # Geometry: Cartesian (default) or, when `radius` is given, a lon–lat sphere. Omitting the mask gives
    # the grid `Grids.AllActive`, which stores only a size. `_cg_flux_cgef!` (`CGEF.compute_Π!`) acts on
    # whatever `grid` the workspace holds.
    grid = if radius === nothing
        (nd == 2 || nd == 3) || throw(ArgumentError(
            "Cartesian coarse-graining supports 2D or 3D grids (nd=$nd); pass `radius=…` for a spherical " *
            "(lon, lat) surface."))
        geom = CGEF.FlowGeometries.Geometry.CartesianGeometry{FT}()   # spacing lives in the grid axes
        # Axes pass as given, so a uniform range keeps its uniformity and CGEF's separable path.
        CGEF.FlowGeometries.Grids.StructuredGrid(geom, coords_vecs...; mask = mask)
    else
        nd == 2 || throw(ArgumentError(
            "Spherical coarse-graining is on a 2D lon–lat surface: pass coords_vecs = (lon, lat) and two " *
            "horizontal velocity components (got nd=$nd)."))
        geom = CGEF.FlowGeometries.Geometry.SphericalGeometry(FT(radius))
        CGEF.FlowGeometries.Grids.StructuredGrid(geom, coords_vecs[1], coords_vecs[2]; mask = mask)
    end
    return _cg_workspace_on_grid(velocity_fields, grid, ℓ, filter, return_diagnostics, execution, mask_strategy,
                                 vector_filter)
end

"""
    _cg_flux_workspace(velocity_fields, grid::FlowGeometries.Grids.AbstractGrid, ℓ, filter; …)

Build the workspace directly on a caller-supplied `FlowGeometries` grid, which brings its own geometry,
measure, mask, topology and stretched or curvilinear axes. This reaches every `CGEF.compute_Π!` method
the coordinate-vector form reaches and the rest: `StructuredGrid` over any geometry, `CurvilinearGrid`,
and the node-indexed method taking an `AbstractGrid` with vector fields. The velocity has two or three
components (`compute_Π!` takes `u`, `v` and an optional `w`).
"""
function FIT.CoarseGrainingFlux._cg_flux_workspace(
    velocity_fields::Tuple,
    grid::CGEF.FlowGeometries.Grids.AbstractGrid,
    ℓ::Real,
    filter::FIT.Types.AbstractFilter;
    return_diagnostics::Bool = false,
    execution::CGEF.ComputationalBackends.AbstractExecutionBackend = CGEF.ComputationalBackends.AutoBackend(),
    mask_strategy::CGEF.Filtering.AbstractMaskStrategy = CGEF.Filtering.ZeroFill(),
    vector_filter::Union{Nothing, CGEF.Filtering.AbstractVectorFilter} = nothing,
)
    return _cg_workspace_on_grid(velocity_fields, grid, ℓ, filter, return_diagnostics, execution, mask_strategy,
                                 vector_filter)
end

# Shared tail of both workspace builders: everything downstream of having a grid. The stress/strain rank
# is the number of velocity components, which a flat unstructured field does not carry in its own shape.
function _cg_workspace_on_grid(velocity_fields, grid, ℓ, filter, return_diagnostics, backend, mask_strategy,
                               vector_filter)
    FT = eltype(velocity_fields[1])
    D  = length(velocity_fields)
    ns = size(velocity_fields[1])

    # Both CGEF plans are built once here and reused by every `compute_Π!` call: the filter plan for the
    # fixed (kernel, scale, mask, execution), and the derivative plan on the grid its outputs are read on.
    kernel = _to_cgef_kernel(filter)
    filter_plan = CGEF.Filtering.plan_filter(grid, kernel, FT(ℓ); mask_strategy = mask_strategy, execution = backend)
    # The workspace lives in the memory of the plan's backend and holds the vector filter CGEF resolves for
    # the grid and the plan's mask strategy. A third component on a two-dimensional grid is the thin-layer
    # vertical velocity, whose buffers the workspace holds only when given one.
    workspace = CGEF.Diagnostics.ΠWorkspace(grid, (), _vertical(velocity_fields);
                                           execution = CGEF.Filtering.plan_execution(filter_plan),
                                           vector_filter = CGEF.Filtering.resolve_vector_filter(
                                               grid, mask_strategy, vector_filter))
    # `compute_Π!` runs its loops under the workspace's placement, and reads and writes every array below
    # in that memory.
    loops = workspace.at.execution
    zeros_at(dims...) = fill!(CGEF.FlowGeometries.Execution.allocate(loops, FT, dims...), zero(FT))
    Π_out = zeros_at(ns...)
    diagnostics = return_diagnostics ? (zeros_at(ns..., D, D), zeros_at(ns..., D, D)) : nothing
    deriv_plan = CGEF.Derivatives.derivative_plan(CGEF.Diagnostics.output_grid(grid, filter_plan); execution = loops)
    weights = CGEF.FlowGeometries.Execution.on_backend(loops, _mean_weights(grid, filter_plan, FT, ns))
    stage = map(f -> FIT.Types._staging_buffer(loops, f), velocity_fields)

    return FIT.CoarseGrainingFlux.CoarseGrainingFluxWorkspace(
        grid, workspace, Π_out, diagnostics, deriv_plan, filter_plan, kernel, FT(ℓ), weights, stage)
end

# Scatter the CGEF workspace's symmetric stress/strain components into the (ns..., nd, nd) buffers.
function _fill_diagnostics!(τ_arr, S_arr, w, ::Val{2})
    c = ntuple(_ -> Colon(), Val(ndims(τ_arr) - 2))   # the field's own rank, ahead of the (i,j) axes
    for (a, b, τv, Sv) in ((1,1,w.τ_xx,w.S_xx), (2,2,w.τ_yy,w.S_yy), (1,2,w.τ_xy,w.S_xy))
        τ_arr[c..., a, b] .= τv; τ_arr[c..., b, a] .= τv
        S_arr[c..., a, b] .= Sv; S_arr[c..., b, a] .= Sv
    end
    return nothing
end
function _fill_diagnostics!(τ_arr, S_arr, w, ::Val{3})
    c = ntuple(_ -> Colon(), Val(ndims(τ_arr) - 2))
    for (a, b, τv, Sv) in ((1,1,w.τ_xx,w.S_xx), (2,2,w.τ_yy,w.S_yy), (3,3,w.τ_zz,w.S_zz),
                           (1,2,w.τ_xy,w.S_xy), (1,3,w.τ_xz,w.S_xz), (2,3,w.τ_yz,w.S_yz))
        τ_arr[c..., a, b] .= τv; τ_arr[c..., b, a] .= τv
        S_arr[c..., a, b] .= Sv; S_arr[c..., b, a] .= Sv
    end
    return nothing
end

# Core compute: fill ws.Π_out (+ optional diagnostics) via CGEF using the workspace's prebuilt plans
# (workspace + filter_plan + deriv_plan all reused), wrap in a result. Allocation-free apart from the
# small result struct — the kernel/scale/mask/execution are all baked into the prebuilt `filter_plan`.
function FIT.CoarseGrainingFlux._cg_flux_cgef!(
    ws::FIT.CoarseGrainingFlux.CoarseGrainingFluxWorkspace,
    velocity_fields::Tuple,
)
    Π_out = ws.Π_out
    FT = eltype(Π_out)
    size(velocity_fields[1]) == size(Π_out) || throw(DimensionMismatch(
        "velocity field size $(size(velocity_fields[1])) does not match workspace grid $(size(Π_out))"))
    length(velocity_fields) == length(ws.stage) || throw(ArgumentError(
        "this workspace was built for $(length(ws.stage)) velocity components; got $(length(velocity_fields)). " *
        "Build one from fields with $(length(velocity_fields)) components."))

    # The fields in the workspace's memory: host fields on a device workspace are copied into its staging
    # buffers.
    fields = map(FIT.Types._staged!, ws.stage, velocity_fields)
    # Vertical component only for a 3-component field; the 2D flux passes `nothing` (single horizontal
    # layer). The component count is the criterion — an unstructured grid holds a 3-D field as a flat
    # vector, so the array rank does not say how many velocity directions there are. `compute_Π!` takes
    # the kernel and scale positionally and reuses the prebuilt plans.
    D = length(velocity_fields)
    CGEF.Diagnostics.compute_Π!(
        Π_out,
        fields[1], fields[2], _vertical(fields),
        ws.grid,
        ws.kernel,
        ws.scale;
        workspace   = ws.cgef_workspace,
        filter_plan = ws.filter_plan,
        deriv_plan  = ws.deriv_plan,
    )

    mean_Π = _weighted_sum(Π_out, ws.weights)

    if ws.diagnostics === nothing
        return FIT.Types.CoarseGrainingFluxResult(ws.scale, Π_out, FT(mean_Π))
    else
        τ_arr, S_arr = ws.diagnostics
        _fill_diagnostics!(τ_arr, S_arr, ws.cgef_workspace, Val(D))
        return FIT.Types.CoarseGrainingFluxResultWithDiagnostics(ws.scale, Π_out, FT(mean_Π), τ_arr, S_arr)
    end
end

"""
    _cg_flux_cgef(velocity_fields, coords_vecs_or_grid, ℓ, filter; kwargs...)

Allocating coarse-graining flux: build a one-shot [`CoarseGrainingFluxWorkspace`](@ref) and
delegate to [`calculate_coarse_graining_flux!`](@ref) (`CGEF.compute_Π!`). Takes either the
coordinate vectors of a 2D/3D Cartesian grid (or a lon–lat sphere with `radius`) or a
`FlowGeometries` grid, which brings its own geometry, measure and mask.
"""
function FIT.CoarseGrainingFlux._cg_flux_cgef(
    velocity_fields::Tuple,
    geometry,
    ℓ::Real,
    filter::FIT.Types.AbstractFilter;
    kwargs...,
)
    ws = FIT.CoarseGrainingFlux._cg_flux_workspace(velocity_fields, geometry, ℓ, filter; kwargs...)
    return FIT.CoarseGrainingFlux._cg_flux_cgef!(ws, velocity_fields)
end

# ---------------------------------------------------------------------------
# The rest of CGEF's coarse-graining diagnostics, on the same domain argument as the flux.
#
# Each is declared on `FlowGeometries.Grids.AbstractGrid` upstream, so the grid form forwards directly
# and the coordinate-vector form builds the grid those axes describe (`_cg_domain_grid`).
# ---------------------------------------------------------------------------

# The grid a domain argument names: the grid itself, or the one coordinate vectors describe, with the
# point mask (`true` = active) when one is given; a grid carries its own mask.
_cg_domain_grid(grid::CGEF.FlowGeometries.Grids.AbstractGrid, ::Type, ::Any, ::Nothing) = grid
_cg_domain_grid(::CGEF.FlowGeometries.Grids.AbstractGrid, ::Type, ::Any, ::AbstractArray{Bool}) = throw(ArgumentError(
    "a grid carries its own mask; pass `mask` when building the grid."))
function _cg_domain_grid(coords_vecs::Tuple, ::Type{FT}, radius, mask) where {FT}
    nd = length(coords_vecs)
    if radius === nothing
        geom = CGEF.FlowGeometries.Geometry.CartesianGeometry{FT}()
        return CGEF.FlowGeometries.Grids.StructuredGrid(geom, coords_vecs...; mask = mask)
    end
    nd == 2 || throw(ArgumentError(
        "a spherical domain is the 2D (lon, lat) surface; got $(nd) coordinate vectors."))
    geom = CGEF.FlowGeometries.Geometry.SphericalGeometry(FT(radius))
    return CGEF.FlowGeometries.Grids.StructuredGrid(geom, coords_vecs[1], coords_vecs[2]; mask = mask)
end

function FIT.CoarseGrainingFlux.calculate_band_energies(
    velocity_fields::Tuple, geometry, filter::FIT.Types.AbstractFilter, scales::AbstractVector;
    radius = nothing, mask::Union{Nothing, AbstractArray{Bool}} = nothing,
    vector_filter::Union{Nothing, CGEF.Filtering.AbstractVectorFilter} = nothing, kwargs...,
)
    FT = eltype(velocity_fields[1])
    grid = _cg_domain_grid(geometry, FT, radius, mask)
    return CGEF.Diagnostics.band_energies(velocity_fields[1], velocity_fields[2], _vertical(velocity_fields), grid,
                                          _to_cgef_kernel(filter), collect(FT, scales);
                                          vector_filter = vector_filter, kwargs...)
end

function FIT.CoarseGrainingFlux.calculate_enstrophy_flux(
    velocity_fields::Tuple, geometry, ℓ::Real, filter::FIT.Types.AbstractFilter;
    radius = nothing, mask::Union{Nothing, AbstractArray{Bool}} = nothing,
    execution::CGEF.ComputationalBackends.AbstractExecutionBackend = CGEF.ComputationalBackends.AutoBackend(),
    vector_filter::Union{Nothing, CGEF.Filtering.AbstractVectorFilter} = nothing,
    kwargs...,
)
    length(velocity_fields) == 2 || throw(ArgumentError(
        "the enstrophy flux is built from the vertical vorticity of a two-component tangent field; " *
        "got $(length(velocity_fields))."))
    FT = eltype(velocity_fields[1])
    grid = _cg_domain_grid(geometry, FT, radius, mask)
    # The vorticity and the tracer flux read the velocity in the memory their loops run in.
    loops = CGEF.Diagnostics.Placement(grid; execution = execution).execution
    u, v = map(f -> CGEF.FlowGeometries.Execution.on_backend(loops, f), velocity_fields)
    return CGEF.Diagnostics.enstrophy_flux(u, v, grid, _to_cgef_kernel(filter), FT(ℓ);
                                           execution = execution, vector_filter = vector_filter, kwargs...)
end

end # module FlowInvariantTransferCGEFExt
