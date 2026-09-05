module FlowInvariantTransferCGEFExt

using CoarseGrainingEnergyFluxes: CoarseGrainingEnergyFluxes as CGEF
using FlowInvariantTransfer: FlowInvariantTransfer as FIT

# ---------------------------------------------------------------------------
# Filter type mapping: FIT.AbstractFilter → CGEF.AbstractFilterKernel
# ---------------------------------------------------------------------------

_to_cgef_kernel(::FIT.Types.GaussianFilter)      = CGEF.Kernels.GaussianKernel()
_to_cgef_kernel(::FIT.Types.TopHatFilter)        = CGEF.Kernels.TopHatKernel()
_to_cgef_kernel(::FIT.Types.SharpSpectralFilter) = CGEF.Kernels.SharpSpectralKernel()

# ---------------------------------------------------------------------------
# Override CoarseGrainingFlux._cg_flux_cgef
# ---------------------------------------------------------------------------

# Allocation-free masked mean over the included points of the flux field (N-D).
function _masked_mean(Π::AbstractArray{FT}, active::AbstractArray{Bool}) where {FT}
    acc = zero(FT)
    n = 0
    @inbounds for i in eachindex(Π, active)
        if active[i]
            acc += Π[i]
            n += 1
        end
    end
    return acc / max(1, n)
end

# An all-active grid stores its mask as a size, so the mean is over every point.
_masked_mean(Π::AbstractArray{FT}, ::CGEF.FlowGeometries.Grids.AllActive) where {FT} =
    sum(Π) / max(1, length(Π))

"""
    _cg_flux_workspace(velocity_fields, coords_vecs, ℓ, filter; mask=nothing, return_diagnostics=false,
                       mask_strategy=Deformable(), backend=AutoBackend(), radius=nothing)

Build the reusable CGEF `StructuredGrid`, its `ΠWorkspace`, the derivative + filter plans (fixed for
the given `(filter, scale ℓ, mask_strategy, backend)`), the `Π_ℓ(x)` output buffer, and (when
`return_diagnostics`) the `τ̄`/`S̄` diagnostic buffers, wrapped in a `CoarseGrainingFluxWorkspace`.
Supports 2D/3D Cartesian grids (or a lon–lat sphere with `radius`) from tuple inputs.
"""
function FIT.CoarseGrainingFlux._cg_flux_workspace(
    velocity_fields::Tuple,
    coords_vecs::Tuple,
    ℓ::Real,
    filter::FIT.Types.AbstractFilter;
    return_diagnostics::Bool = false,
    mask::Union{Nothing, AbstractArray{Bool}} = nothing,
    radius::Union{Nothing, Real} = nothing,
    backend::CGEF.ComputationalBackends.AbstractExecutionBackend = CGEF.ComputationalBackends.AutoBackend(),
    mask_strategy::CGEF.Filtering.AbstractMaskStrategy = CGEF.Filtering.Deformable(),
)
    D  = length(velocity_fields)
    nd = length(coords_vecs)
    D == nd || throw(ArgumentError(
        "Number of velocity components ($D) must equal number of spatial dimensions ($nd)"))

    FT  = eltype(velocity_fields[1])

    # Geometry: Cartesian (default) or, when `radius` is given, a lon–lat sphere. Omitting the mask gives
    # the grid `Grids.AllActive`, which stores only a size. `_cg_flux_cgef!` (`CGEF.compute_Π!`) is
    # geometry-agnostic — it operates on whatever `grid` the workspace holds — so spherical support is
    # entirely a matter of building a spherical grid here (the sibling `CoarseGrainingEnergyFluxes.jl`
    # implements the spherical filter/gradient stencils).
    grid = if radius === nothing
        (nd == 2 || nd == 3) || throw(ArgumentError(
            "Cartesian coarse-graining supports 2D or 3D grids (nd=$nd); pass `radius=…` for a spherical " *
            "(lon, lat) surface."))
        geom = CGEF.FlowGeometries.Geometry.CartesianGeometry{FT}()   # spacing lives in the grid axes
        # Pass axes as given: `_to_axis` adapts to FT preserving a uniform range's uniformity (CGEF's
        # fast separable path); materializing to a plain vector would look stretched.
        CGEF.FlowGeometries.Grids.StructuredGrid(geom, coords_vecs...; mask = mask)
    else
        nd == 2 || throw(ArgumentError(
            "Spherical coarse-graining is on a 2D lon–lat surface: pass coords_vecs = (lon, lat) and two " *
            "horizontal velocity components (got nd=$nd)."))
        geom = CGEF.FlowGeometries.Geometry.SphericalGeometry(FT(radius))
        CGEF.FlowGeometries.Grids.StructuredGrid(geom, coords_vecs[1], coords_vecs[2]; mask = mask)
    end
    return _cg_workspace_on_grid(velocity_fields, grid, ℓ, filter, return_diagnostics, backend, mask_strategy)
end

"""
    _cg_flux_workspace(velocity_fields, grid::FlowGeometries.Grids.AbstractGrid, ℓ, filter; …)

Build the workspace directly on a caller-supplied `FlowGeometries` grid, which brings its own geometry,
measure, mask, topology and stretched or curvilinear axes. This reaches every `CGEF.compute_Π!` method,
not only the 2-D structured one the coordinate-vector form can build: `StructuredGrid` in 1-D/2-D/3-D
over any geometry, `CurvilinearGrid`, and the node-indexed method taking an `AbstractGrid` with vector
fields.
"""
function FIT.CoarseGrainingFlux._cg_flux_workspace(
    velocity_fields::Tuple,
    grid::CGEF.FlowGeometries.Grids.AbstractGrid,
    ℓ::Real,
    filter::FIT.Types.AbstractFilter;
    return_diagnostics::Bool = false,
    backend::CGEF.ComputationalBackends.AbstractExecutionBackend = CGEF.ComputationalBackends.AutoBackend(),
    mask_strategy::CGEF.Filtering.AbstractMaskStrategy = CGEF.Filtering.Deformable(),
)
    return _cg_workspace_on_grid(velocity_fields, grid, ℓ, filter, return_diagnostics, backend, mask_strategy)
end

# Shared tail of both workspace builders: everything downstream of having a grid. The stress/strain rank
# is the number of velocity components, which a flat unstructured field does not carry in its own shape.
function _cg_workspace_on_grid(velocity_fields, grid, ℓ, filter, return_diagnostics, backend, mask_strategy)
    FT = eltype(velocity_fields[1])
    D  = length(velocity_fields)
    ns = size(velocity_fields[1])

    workspace = CGEF.Diagnostics.ΠWorkspace(grid)                # dimensionality inferred from the grid
    Π_out = zeros(FT, ns...)
    diagnostics = return_diagnostics ? (zeros(FT, ns..., D, D), zeros(FT, ns..., D, D)) : nothing
    # Both CGEF plans are built ONCE here and stored as concrete typed fields (no `Any`): the derivative
    # `StencilPlan` (grid-only) and the filter footprint for the fixed (kernel, scale, mask, backend).
    # `compute_Π!` then reuses both, so a repeat call allocates nothing inside CGEF.
    deriv_plan  = _cg_deriv_plan(grid)
    filter_plan = CGEF.Filtering.plan_filter(grid, _to_cgef_kernel(filter), FT(ℓ);
                                             mask_strategy = mask_strategy, backend = backend)

    return FIT.CoarseGrainingFlux.CoarseGrainingFluxWorkspace(
        grid, workspace, Π_out, diagnostics, deriv_plan, filter_plan)
end

# The derivative plan a grid affords, and the one its `compute_Π!` method accepts
_cg_deriv_plan(grid::CGEF.FlowGeometries.Grids.StructuredGrid) = CGEF.Derivatives.StencilPlan(grid)
_cg_deriv_plan(grid) = CGEF.Derivatives.gradient_plan(grid)

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
# small result struct — the kernel/scale/mask/backend are all baked into the prebuilt `filter_plan`.
function FIT.CoarseGrainingFlux._cg_flux_cgef!(
    ws::FIT.CoarseGrainingFlux.CoarseGrainingFluxWorkspace,
    velocity_fields::Tuple,
)
    Π_out = ws.Π_out
    FT = eltype(Π_out)
    size(velocity_fields[1]) == size(Π_out) || throw(DimensionMismatch(
        "velocity field size $(size(velocity_fields[1])) does not match workspace grid $(size(Π_out))"))

    plan = ws.filter_plan
    # Vertical component only for a 3-component field; the 2D flux passes `nothing` (single horizontal
    # layer). The component count is the criterion — an unstructured grid holds a 3-D field as a flat
    # vector, so the array rank does not say how many velocity directions there are. `plan.kernel` and
    # `plan.scale` are the fixed values the plan was built with — passed positionally (compute_Π!
    # requires them but reuses the prebuilt plan; backend/mask_strategy are unused when all plans are given).
    D = length(velocity_fields)
    w_comp = D == 3 ? velocity_fields[3] : nothing
    CGEF.Diagnostics.compute_Π!(
        Π_out,
        velocity_fields[1], velocity_fields[2], w_comp,
        ws.grid,
        plan.kernel,
        plan.scale;
        workspace   = ws.cgef_workspace,
        filter_plan = plan,
        deriv_plan  = ws.deriv_plan,
    )

    mean_Π = _masked_mean(Π_out, ws.grid.mask)

    if ws.diagnostics === nothing
        return FIT.Types.CoarseGrainingFluxResult(plan.scale, Π_out, FT(mean_Π))
    else
        τ_arr, S_arr = ws.diagnostics
        _fill_diagnostics!(τ_arr, S_arr, ws.cgef_workspace, Val(D))
        return FIT.Types.CoarseGrainingFluxResultWithDiagnostics(plan.scale, Π_out, FT(mean_Π), τ_arr, S_arr)
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

end # module FlowInvariantTransferCGEFExt
