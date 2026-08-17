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
    ns  = size(velocity_fields[1])
    active = mask !== nothing ? mask : trues(ns...)   # boolean point mask (true = included)

    # Geometry: Cartesian (default) or, when `radius` is given, a lon–lat sphere. `_cg_flux_cgef!`
    # (`CGEF.compute_Π!`) is geometry-agnostic — it operates on whatever `grid` the workspace holds —
    # so spherical support is entirely a matter of building a spherical grid here (the sibling
    # `CoarseGrainingEnergyFluxes.jl` implements the spherical filter/gradient stencils).
    grid = if radius === nothing
        (nd == 2 || nd == 3) || throw(ArgumentError(
            "Cartesian coarse-graining supports 2D or 3D grids (nd=$nd); pass `radius=…` for a spherical " *
            "(lon, lat) surface."))
        geom = CGEF.FlowGeometries.Geometry.CartesianGeometry{FT}()   # spacing lives in the grid axes
        # Pass axes as given: `_to_axis` adapts to FT preserving a uniform range's uniformity (CGEF's
        # fast separable path); materializing to a plain vector would look stretched.
        CGEF.FlowGeometries.Grids.StructuredGrid(geom, coords_vecs..., active)
    else
        nd == 2 || throw(ArgumentError(
            "Spherical coarse-graining is on a 2D lon–lat surface: pass coords_vecs = (lon, lat) and two " *
            "horizontal velocity components (got nd=$nd)."))
        geom = CGEF.FlowGeometries.Geometry.SphericalGeometry(FT(radius))
        CGEF.FlowGeometries.Grids.StructuredGrid(geom, coords_vecs[1], coords_vecs[2], active)
    end

    workspace = CGEF.Diagnostics.ΠWorkspace(grid)                # dimensionality inferred from the grid
    Π_out = zeros(FT, ns...)
    diagnostics = return_diagnostics ?
        (zeros(FT, ns..., nd, nd), zeros(FT, ns..., nd, nd)) : nothing
    # Both CGEF plans are built ONCE here and stored as concrete typed fields (no `Any`): the derivative
    # `StencilPlan` (grid-only) and the filter footprint for the fixed (kernel, scale, mask, backend).
    # `compute_Π!` then reuses both, so a repeat call allocates nothing inside CGEF.
    deriv_plan  = CGEF.Derivatives.StencilPlan(grid)
    filter_plan = CGEF.Filtering.plan_filter(grid, _to_cgef_kernel(filter), FT(ℓ);
                                             mask_strategy = mask_strategy, backend = backend)

    return FIT.CoarseGrainingFlux.CoarseGrainingFluxWorkspace(
        grid, workspace, Π_out, diagnostics, deriv_plan, filter_plan)
end

# Scatter the CGEF workspace's symmetric stress/strain components into the (ns..., nd, nd) buffers.
function _fill_diagnostics!(τ_arr, S_arr, w, ::Val{2})
    c = ntuple(_ -> Colon(), 2)
    for (a, b, τv, Sv) in ((1,1,w.τ_xx,w.S_xx), (2,2,w.τ_yy,w.S_yy), (1,2,w.τ_xy,w.S_xy))
        τ_arr[c..., a, b] .= τv; τ_arr[c..., b, a] .= τv
        S_arr[c..., a, b] .= Sv; S_arr[c..., b, a] .= Sv
    end
    return nothing
end
function _fill_diagnostics!(τ_arr, S_arr, w, ::Val{3})
    c = ntuple(_ -> Colon(), 3)
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
    # Vertical component only in 3D; the 2D flux passes `nothing` (single horizontal layer). `plan.kernel`
    # and `plan.scale` are the fixed values the plan was built with — passed positionally (compute_Π!
    # requires them but reuses the prebuilt plan; backend/mask_strategy are unused when all plans are given).
    w_comp = ndims(Π_out) == 3 ? velocity_fields[3] : nothing
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
        _fill_diagnostics!(τ_arr, S_arr, ws.cgef_workspace, Val(ndims(Π_out)))
        return FIT.Types.CoarseGrainingFluxResultWithDiagnostics(plan.scale, Π_out, FT(mean_Π), τ_arr, S_arr)
    end
end

"""
    _cg_flux_cgef(velocity_fields, coords_vecs, ℓ, filter; kwargs...)

Allocating coarse-graining flux: build a one-shot [`CoarseGrainingFluxWorkspace`](@ref) and
delegate to [`calculate_coarse_graining_flux!`](@ref) (`CGEF.compute_Π!`). Supports 2D and 3D
Cartesian grids from tuple inputs. For spherical geometry or more control (masks, backends),
call CGEF directly and wrap the result with `CoarseGrainingFluxResult`.
"""
function FIT.CoarseGrainingFlux._cg_flux_cgef(
    velocity_fields::Tuple,
    coords_vecs::Tuple,
    ℓ::Real,
    filter::FIT.Types.AbstractFilter;
    return_diagnostics::Bool = false,
    mask::Union{Nothing, AbstractArray{Bool}} = nothing,
    radius::Union{Nothing, Real} = nothing,
    kwargs...,
)
    ws = FIT.CoarseGrainingFlux._cg_flux_workspace(
        velocity_fields, coords_vecs, ℓ, filter;
        return_diagnostics = return_diagnostics, mask = mask, radius = radius, kwargs...)
    return FIT.CoarseGrainingFlux._cg_flux_cgef!(ws, velocity_fields)
end

end # module FlowInvariantTransferCGEFExt
