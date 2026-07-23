module FlowInvariantTransferCGEFExt

using CoarseGrainingEnergyFluxes: CoarseGrainingEnergyFluxes as CGEF
using FlowInvariantTransfer: FlowInvariantTransfer as FIT
using FlowInvariantTransfer.Types: AbstractFilter, SharpSpectralFilter, GaussianFilter, TopHatFilter, CoarseGrainingFluxResult, CoarseGrainingFluxResultWithDiagnostics

# ---------------------------------------------------------------------------
# Filter type mapping: FIT.AbstractFilter → CGEF.AbstractFilterKernel
# ---------------------------------------------------------------------------

_to_cgef_kernel(::GaussianFilter)      = CGEF.Kernels.GaussianKernel()
_to_cgef_kernel(::TopHatFilter)        = CGEF.Kernels.TopHatKernel()
_to_cgef_kernel(::SharpSpectralFilter) = CGEF.Kernels.SharpSpectralKernel()

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
    _cg_flux_workspace(velocity_fields, coords_vecs, filter; mask=nothing, return_diagnostics=false)

Build the reusable CGEF `StructuredGrid`, its `ΠWorkspace`, the `Π_ℓ(x)` output buffer, and
(when `return_diagnostics`) the `τ̄`/`S̄` diagnostic buffers, wrapped in a
`CoarseGrainingFluxWorkspace`. Supports 2D and 3D Cartesian grids from tuple inputs.
"""
function FIT.CoarseGrainingFlux._cg_flux_workspace(
    velocity_fields::Tuple,
    coords_vecs::Tuple,
    filter::AbstractFilter;
    return_diagnostics::Bool = false,
    mask::Union{Nothing, AbstractArray{Bool}} = nothing,
    radius::Union{Nothing, Real} = nothing,
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
        # Per-dimension grid spacing from the coordinate vectors.
        dx = ntuple(nd) do i
            v = coords_vecs[i]
            length(v) > 1 ? FT((v[end] - v[begin]) / (length(v) - 1)) : FT(1)
        end
        geom = CGEF.Geometry.CartesianGeometry(dx...)             # (dx,dy) 2D / (dx,dy,dz) 3D
        CGEF.Grids.StructuredGrid(geom, ntuple(i -> FT.(coords_vecs[i]), nd)..., active)
    else
        nd == 2 || throw(ArgumentError(
            "Spherical coarse-graining is on a 2D lon–lat surface: pass coords_vecs = (lon, lat) and two " *
            "horizontal velocity components (got nd=$nd)."))
        geom = CGEF.Geometry.SphericalGeometry(FT(radius))
        CGEF.Grids.StructuredGrid(geom, FT.(coords_vecs[1]), FT.(coords_vecs[2]), active)
    end

    workspace = CGEF.Diagnostics.ΠWorkspace(grid)                # dimensionality inferred from the grid
    Π_out = zeros(FT, ns...)
    diagnostics = return_diagnostics ?
        (zeros(FT, ns..., nd, nd), zeros(FT, ns..., nd, nd)) : nothing

    return FIT.CoarseGrainingFlux.CoarseGrainingFluxWorkspace(
        grid, workspace, Π_out, diagnostics, Base.RefValue{Any}(nothing))
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

# Return the cached filter plan if it was built for this exact (scale, kernel, mask); otherwise
# build it once, cache it, and return it. The footprint is scale-dependent, so a genuinely new
# scale (e.g. a Π(ℓ) sweep) rebuilds — but repeated snapshots at a fixed scale reuse it, which is
# where the ~MB/call footprint allocation was coming from.
function _cached_filter_plan(ws, cgef_kernel, ℓ::Real, mask_strategy, backend)
    cached = ws.filter_plan[]
    if cached !== nothing &&
       cached.scale == ℓ &&
       typeof(cached.kernel) === typeof(cgef_kernel) &&
       typeof(cached.strategy) === typeof(mask_strategy)
        return cached
    end
    plan = CGEF.Filtering.plan_filter(ws.grid, cgef_kernel, ℓ; mask_strategy = mask_strategy, backend = backend)
    ws.filter_plan[] = plan
    return plan
end

# Core compute: fill ws.Π_out (+ optional diagnostics) via CGEF, wrap in a result.
function FIT.CoarseGrainingFlux._cg_flux_cgef!(
    ws::FIT.CoarseGrainingFlux.CoarseGrainingFluxWorkspace,
    velocity_fields::Tuple,
    ℓ::Real,
    filter::AbstractFilter;
    backend::CGEF.Backends.AbstractExecutionBackend = CGEF.Backends.AutoBackend(),
    mask_strategy::CGEF.Filtering.AbstractMaskStrategy = CGEF.Filtering.Deformable(),
    kwargs...,
)
    Π_out = ws.Π_out
    FT = eltype(Π_out)
    size(velocity_fields[1]) == size(Π_out) || throw(DimensionMismatch(
        "velocity field size $(size(velocity_fields[1])) does not match workspace grid $(size(Π_out))"))

    cgef_kernel = _to_cgef_kernel(filter)
    plan = _cached_filter_plan(ws, cgef_kernel, FT(ℓ), mask_strategy, backend)

    # Vertical component only in 3D; the 2D flux passes `nothing` (single horizontal layer).
    w_comp = ndims(Π_out) == 3 ? velocity_fields[3] : nothing
    CGEF.Diagnostics.compute_Π!(
        Π_out,
        velocity_fields[1], velocity_fields[2], w_comp,
        ws.grid,
        cgef_kernel,
        FT(ℓ);
        workspace     = ws.cgef_workspace,
        filter_plan   = plan,
        backend       = backend,
        mask_strategy = mask_strategy,
        kwargs...,
    )

    mean_Π = _masked_mean(Π_out, ws.grid.mask)

    if ws.diagnostics === nothing
        return CoarseGrainingFluxResult(FT(ℓ), Π_out, FT(mean_Π))
    else
        τ_arr, S_arr = ws.diagnostics
        _fill_diagnostics!(τ_arr, S_arr, ws.cgef_workspace, Val(ndims(Π_out)))
        return CoarseGrainingFluxResultWithDiagnostics(FT(ℓ), Π_out, FT(mean_Π), τ_arr, S_arr)
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
    filter::AbstractFilter;
    return_diagnostics::Bool = false,
    mask::Union{Nothing, AbstractArray{Bool}} = nothing,
    radius::Union{Nothing, Real} = nothing,
    kwargs...,
)
    ws = FIT.CoarseGrainingFlux._cg_flux_workspace(
        velocity_fields, coords_vecs, filter; return_diagnostics = return_diagnostics, mask = mask, radius = radius)
    return FIT.CoarseGrainingFlux._cg_flux_cgef!(ws, velocity_fields, ℓ, filter; kwargs...)
end

end # module FlowInvariantTransferCGEFExt
