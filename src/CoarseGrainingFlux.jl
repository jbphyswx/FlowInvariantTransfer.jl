module CoarseGrainingFlux

using ..Types: CoarseGrainingFluxMethod, CoarseGrainingFluxResult, AbstractFilter, AbstractFieldDecomposition, NoDecomposition
using ..Decomposition: decompose_field

export calculate_coarse_graining_flux, calculate_coarse_graining_flux!, CoarseGrainingFluxWorkspace

# ---------------------------------------------------------------------------
# Internal stubs — overridden by FlowInvariantTransferCGEFExt when
# CoarseGrainingEnergyFluxes is loaded.
# ---------------------------------------------------------------------------

"""
    _cg_flux_cgef(velocity_fields, coords_vecs, ℓ, filter; kwargs...)

Coarse-graining flux via CoarseGrainingEnergyFluxes.jl.
Stub overridden by the CoarseGrainingEnergyFluxes extension.
"""
function _cg_flux_cgef(args...; kwargs...)
    throw(ArgumentError(
        "Coarse-graining flux requires CoarseGrainingEnergyFluxes.jl. " *
        "Run `using CoarseGrainingEnergyFluxes` to load the extension."))
end

"""
    _cg_flux_workspace(velocity_fields, coords_vecs, filter; kwargs...)

Build the reusable CGEF grid + `ΠWorkspace` + output buffer.
Stub overridden by the CoarseGrainingEnergyFluxes extension.
"""
function _cg_flux_workspace(args...; kwargs...)
    throw(ArgumentError(
        "CoarseGrainingFluxWorkspace requires CoarseGrainingEnergyFluxes.jl. " *
        "Run `using CoarseGrainingEnergyFluxes` to load the extension."))
end

"""
    _cg_flux_cgef!(ws, velocity_fields, ℓ, filter; kwargs...)

In-place coarse-graining flux reusing `ws`. Stub overridden by the extension.
"""
function _cg_flux_cgef!(args...; kwargs...)
    throw(ArgumentError(
        "Coarse-graining flux requires CoarseGrainingEnergyFluxes.jl. " *
        "Run `using CoarseGrainingEnergyFluxes` to load the extension."))
end

# ---------------------------------------------------------------------------
# Reusable workspace
# ---------------------------------------------------------------------------

"""
    CoarseGrainingFluxWorkspace(velocity_fields, coords_vecs, filter;
                                mask=nothing, return_diagnostics=false)

Preallocated, reusable resources for [`calculate_coarse_graining_flux!`](@ref): the
CGEF `StructuredGrid`, its `ΠWorkspace` (filtered fields + stress/strain scratch), the
`Π_ℓ(x)` output buffer, and — when `return_diagnostics=true` — the `τ̄`/`S̄` diagnostic
buffers. Build once, then reuse across a whole filter-scale sweep `Π(ℓ)` or across
snapshots on the same grid; the grid geometry (`coords_vecs`, `mask`) and array element
type are fixed at construction.

The dominant per-call cost inside CGEF is the scale-dependent *filter footprint/plan*, not the
scratch arrays. `filter_plan` caches the last-built plan so repeated snapshots at a fixed scale
reuse it (the footprint is rebuilt only when the requested scale/kernel/mask changes — e.g. a
`Π(ℓ)` sweep, where a new footprint is genuinely required). It is a deliberately type-erased
cache slot (the plan's concrete type varies with kernel/method/backend and is touched once per
call), so it is not a numeric-payload field; `Π_out`/`diagnostics` carry the real element type.

Requires `CoarseGrainingEnergyFluxes` to be loaded. The grid/workspace fields are typed
via parameters so the struct carries no hardcoded CGEF types into the core package.
"""
struct CoarseGrainingFluxWorkspace{G, W, P, D}
    grid::G                       # CGEF.Grids.StructuredGrid
    cgef_workspace::W             # CGEF.Diagnostics.ΠWorkspace
    Π_out::P                      # reused Π_ℓ(x) output buffer (N-D: Matrix in 2D, Array{,3} in 3D)
    diagnostics::D                # nothing, or (τ_arr, S_arr) reused diagnostic buffers
    filter_plan::Base.RefValue{Any}  # scale-keyed cache of the CGEF filter plan
end

function CoarseGrainingFluxWorkspace(
    velocity_fields::Tuple, coords_vecs::Tuple, filter::AbstractFilter; kwargs...,
)
    return _cg_flux_workspace(velocity_fields, coords_vecs, filter; kwargs...)
end

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

"""
    calculate_coarse_graining_flux(velocity_fields, coords_vecs, ℓ, filter;
                                   decomposition=NoDecomposition(), return_diagnostics=false, kwargs...)
                                   -> CoarseGrainingFluxResult

Compute the pointwise cross-scale kinetic energy flux Π_ℓ(x) = −τ̄ᵢⱼ S̄ᵢⱼ at filter
scale ℓ.  Delegates entirely to `CoarseGrainingEnergyFluxes.jl` (`compute_Π!`,
`filter_field!`).

**Requires** `CoarseGrainingEnergyFluxes` to be loaded:
```julia
using CoarseGrainingEnergyFluxes
result = calculate_coarse_graining_flux((u, v), (x, y), ℓ, GaussianFilter())
```

# Arguments
- `velocity_fields`: Tuple of D real arrays `(u, v[, w])` — velocity components.
- `coords_vecs`: Tuple of D coordinate vectors `(x, y[, z])`.
- `ℓ::Real`: Filter length scale (same units as coordinates).
- `filter::AbstractFilter`: `GaussianFilter()`, `SharpSpectralFilter()`, or `TopHatFilter()`.

# Keyword Arguments
- `decomposition::AbstractFieldDecomposition`: `NoDecomposition()` (default), `HelmholtzDecomposition()`, `RotationalDecomposition()`, or `DivergentDecomposition()`.
- `return_diagnostics::Bool=false`: Also return τ̄ᵢⱼ and S̄ᵢⱼ fields.
- `mask::Union{Nothing,AbstractMatrix{Bool}}=nothing`: boolean point mask (`true` = point included in
  the computation, `false` = excluded). If `nothing`, all points are included.
- Any additional kwargs are forwarded to `CoarseGrainingEnergyFluxes.compute_Π!`.

# Returns
`CoarseGrainingFluxResult` or NamedTuple of `CoarseGrainingFluxResult` depending on decomposition.
"""
function calculate_coarse_graining_flux(
    velocity_fields::Tuple,
    coords_vecs::Tuple,
    ℓ::Real,
    filter::AbstractFilter;
    decomposition::AbstractFieldDecomposition = NoDecomposition(),
    kwargs...,
)
    decomposed = decompose_field(decomposition, velocity_fields, coords_vecs; kwargs...)
    return _calculate_coarse_graining_flux_decomposed(
        decomposed, velocity_fields, coords_vecs, ℓ, filter; kwargs...
    )
end

function _calculate_coarse_graining_flux_decomposed(
    decomp_fields::Tuple,
    velocity_fields::Tuple,
    coords_vecs::Tuple,
    ℓ::Real,
    filter::AbstractFilter;
    kwargs...,
)
    return _cg_flux_cgef(decomp_fields, coords_vecs, ℓ, filter; kwargs...)
end

function _calculate_coarse_graining_flux_decomposed(
    decomposed::NamedTuple,
    velocity_fields::Tuple,
    coords_vecs::Tuple,
    ℓ::Real,
    filter::AbstractFilter;
    kwargs...,
)
    return map(decomposed) do fields
        return _cg_flux_cgef(fields, coords_vecs, ℓ, filter; kwargs...)
    end
end

"""
    calculate_coarse_graining_flux!(ws::CoarseGrainingFluxWorkspace, velocity_fields, ℓ, filter;
                                    kwargs...) -> CoarseGrainingFluxResult

In-place coarse-graining flux reusing the preallocated `ws` (its CGEF grid, `ΠWorkspace`,
and `Π_out` buffer) — no per-call grid/workspace/output allocation. Intended for a
filter-scale sweep `Π(ℓ)` or repeated snapshots on the same grid: build the workspace once
with [`CoarseGrainingFluxWorkspace`](@ref), then call this per `(velocity_fields, ℓ)`.

The returned result wraps the reused `ws.Π_out` buffer (a subsequent call overwrites it);
copy `result.flux` if the field must persist across calls. `return_diagnostics` is fixed at
workspace-construction time. Only `NoDecomposition` is supported here — apply a
decomposition upstream and pass the decomposed component fields if needed.

Requires `CoarseGrainingEnergyFluxes` to be loaded.
"""
function calculate_coarse_graining_flux!(
    ws::CoarseGrainingFluxWorkspace,
    velocity_fields::Tuple,
    ℓ::Real,
    filter::AbstractFilter;
    kwargs...,
)
    return _cg_flux_cgef!(ws, velocity_fields, ℓ, filter; kwargs...)
end

# One-line show (the workspace caches a CGEF filter plan in `filter_plan` → default show can segfault).
Base.show(io::IO, ::CoarseGrainingFluxWorkspace) = print(io, "CoarseGrainingFluxWorkspace(…)")
Base.show(io::IO, ::MIME"text/plain", w::CoarseGrainingFluxWorkspace) = show(io, w)

end # module CoarseGrainingFlux
