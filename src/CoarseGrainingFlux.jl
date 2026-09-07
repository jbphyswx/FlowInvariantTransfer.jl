module CoarseGrainingFlux

using ..Types: Types
using ..Decomposition: Decomposition
using ComputationalBackends: ComputationalBackends

export calculate_coarse_graining_flux, calculate_coarse_graining_flux!, calculate_coarse_graining_flux_batch,
       CoarseGrainingFluxWorkspace,
       calculate_band_energies, calculate_enstrophy_flux

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
    _cg_flux_workspace(velocity_fields, coords_vecs, ℓ, filter; kwargs...)

Build the reusable CGEF grid + `ΠWorkspace` + derivative/filter plans + output buffer for a fixed
`(filter, scale, mask, backend)`. Stub overridden by the CoarseGrainingEnergyFluxes extension.
"""
function _cg_flux_workspace(args...; kwargs...)
    throw(ArgumentError(
        "CoarseGrainingFluxWorkspace requires CoarseGrainingEnergyFluxes.jl. " *
        "Run `using CoarseGrainingEnergyFluxes` to load the extension."))
end

"""
    _cg_flux_cgef!(ws, velocity_fields; kwargs...)

In-place coarse-graining flux reusing `ws` (filter/scale/mask/backend fixed at construction).
Stub overridden by the extension.
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
    CoarseGrainingFluxWorkspace(velocity_fields, coords_vecs, ℓ, filter;
                                mask=nothing, return_diagnostics=false, mask_strategy=…, backend=…)

Preallocated, reusable resources for [`calculate_coarse_graining_flux!`](@ref), built for a **fixed**
`(filter, scale ℓ, mask, backend)` configuration: the CGEF `StructuredGrid`, its `ΠWorkspace`
(filtered fields + stress/strain scratch), the `Π_ℓ(x)` output buffer, the derivative `StencilPlan`,
the filter plan, and — when `return_diagnostics=true` — the `τ̄`/`S̄` diagnostic buffers. Build once,
then apply to snapshots on the same grid with `calculate_coarse_graining_flux!(ws, velocity_fields)`.

All three CGEF plans (`ΠWorkspace`, `filter_plan`, `deriv_plan`) are prebuilt at construction, so a
repeat call allocates **nothing** inside CGEF. Every field is concretely typed — no `Any` — so the
struct carries no hardcoded CGEF types into the core package while staying type-stable in the hot
path. The scale-dependent filter footprint is fixed here; a different scale/filter/mask/backend is a
new workspace (the footprint has to be rebuilt per scale regardless).

Requires `CoarseGrainingEnergyFluxes` to be loaded.
"""
struct CoarseGrainingFluxWorkspace{G, W, P, D, DP, FP}
    grid::G                       # FlowGeometries.Grids.StructuredGrid (via the CGEF extension)
    cgef_workspace::W             # CGEF.Diagnostics.ΠWorkspace
    Π_out::P                      # reused Π_ℓ(x) output buffer (N-D: Matrix in 2D, Array{,3} in 3D)
    diagnostics::D                # nothing, or (τ_arr, S_arr) reused diagnostic buffers
    deriv_plan::DP                # CGEF derivative StencilPlan (grid-only, scale-independent)
    filter_plan::FP               # CGEF filter plan for the fixed (kernel, scale, mask, backend); carries .kernel / .scale
end

function CoarseGrainingFluxWorkspace(
    velocity_fields::Tuple, geometry, ℓ::Real, filter::Types.AbstractFilter; kwargs...,
)
    return _cg_flux_workspace(velocity_fields, geometry, ℓ, filter; kwargs...)
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
    geometry,
    ℓ::Real,
    filter::Types.AbstractFilter;
    decomposition::Types.AbstractFieldDecomposition = Types.NoDecomposition(),
    kwargs...,
)
    decomposition isa Types.NoDecomposition || geometry isa Tuple && throw(ArgumentError(
        "a field decomposition solves a boundary-value problem on the domain, and coordinate vectors " *
        "carry no topology, mask or geometry to solve it on. Pass a `FlowGeometries` grid, or decompose " *
        "upstream and pass the component fields."))
    decomposed = Decomposition.decompose_field(decomposition, velocity_fields, geometry; kwargs...)
    return _calculate_coarse_graining_flux_decomposed(
        decomposed, velocity_fields, geometry, ℓ, filter; kwargs...
    )
end

function _calculate_coarse_graining_flux_decomposed(
    decomp_fields::Tuple,
    velocity_fields::Tuple,
    geometry,
    ℓ::Real,
    filter::Types.AbstractFilter;
    kwargs...,
)
    return _cg_flux_cgef(decomp_fields, geometry, ℓ, filter; kwargs...)
end

function _calculate_coarse_graining_flux_decomposed(
    decomposed::NamedTuple,
    velocity_fields::Tuple,
    geometry,
    ℓ::Real,
    filter::Types.AbstractFilter;
    kwargs...,
)
    return map(decomposed) do fields
        return _cg_flux_cgef(fields, geometry, ℓ, filter; kwargs...)
    end
end

"""
    calculate_coarse_graining_flux!(ws::CoarseGrainingFluxWorkspace, velocity_fields)
        -> CoarseGrainingFluxResult

In-place coarse-graining flux reusing the preallocated `ws` — no allocation (its grid, `ΠWorkspace`,
`Π_out`, and all three CGEF plans are prebuilt). The filter, scale, mask and backend are fixed at
workspace construction, so this takes only the new `velocity_fields`: build the workspace once with
[`CoarseGrainingFluxWorkspace`](@ref) and call this per snapshot on the same grid.

The returned result wraps the reused `ws.Π_out` buffer (a subsequent call overwrites it); copy
`result.flux` if the field must persist across calls. `return_diagnostics` is fixed at
workspace-construction time. Only `NoDecomposition` is supported here — apply a decomposition
upstream and pass the decomposed component fields if needed.

Requires `CoarseGrainingEnergyFluxes` to be loaded.
"""
function calculate_coarse_graining_flux!(
    ws::CoarseGrainingFluxWorkspace,
    velocity_fields::Tuple,
)
    return _cg_flux_cgef!(ws, velocity_fields)
end

"""
    calculate_coarse_graining_flux_batch(velocity_fields_batch, coords_vecs, ℓ, filter; execution, kwargs...)
        -> Vector{CoarseGrainingFluxResult}

Coarse-graining flux Π_ℓ(x) for a batch of snapshots on the same grid (`velocity_fields_batch` an
iterable of velocity-component tuples `(u, v[, w])`). One `CoarseGrainingFluxWorkspace` (grid + all CGEF
plans, fixed for `(ℓ, filter)`) is built per worker and reused across its snapshots — the plans are built
ONCE, not per snapshot. `execution = ThreadedBackend()` (requires `using OhMyThreads`) threads over
snapshots with a serial inner filter (no nested threading). Each result owns an independent copy of its
flux field. Only `NoDecomposition` is supported; `kwargs` (`mask`, `return_diagnostics`, `radius`,
`backend`, …) are forwarded to the workspace. Requires `CoarseGrainingEnergyFluxes` to be loaded.
"""
function calculate_coarse_graining_flux_batch(
    velocity_fields_batch,
    geometry,
    ℓ::Real,
    filter::Types.AbstractFilter;
    execution::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.SerialBackend(),
    kwargs...,
)
    length(velocity_fields_batch) == 0 && return []
    return _cg_flux_batch!(Types.resolve_execution(execution), velocity_fields_batch, geometry, ℓ, filter; kwargs...)
end

# Serial reference: one workspace reused across the batch; each result gets an independent copy (the
# workspace overwrites its output buffer on every snapshot, so the results must not alias it).
function _cg_flux_batch!(::ComputationalBackends.AbstractSerialBackend, velocity_fields_batch, geometry, ℓ, filter; kwargs...)
    ws = CoarseGrainingFluxWorkspace(first(velocity_fields_batch), geometry, ℓ, filter; kwargs...)
    return [deepcopy(calculate_coarse_graining_flux!(ws, vf)) for vf in velocity_fields_batch]
end

# Threaded over the batch — overridden by the OhMyThreads extension (per-chunk workspace, serial inner).
function _cg_flux_batch_threaded!(args...; kwargs...)
    throw(ArgumentError("execution = ThreadedBackend() for the coarse-graining batch requires OhMyThreads. " *
                        "Run `using OhMyThreads` to load the extension."))
end
_cg_flux_batch!(::ComputationalBackends.AbstractThreadedBackend, velocity_fields_batch, geometry, ℓ, filter; kwargs...) =
    _cg_flux_batch_threaded!(velocity_fields_batch, geometry, ℓ, filter; kwargs...)

# Distributed over the batch — overridden by the Distributed extension (snapshots partitioned across workers).
function _cg_flux_batch_distributed!(args...; kwargs...)
    throw(ArgumentError("execution = DistributedBackend() for the coarse-graining batch requires Distributed. " *
                        "Run `using Distributed` to load the extension."))
end
_cg_flux_batch!(::ComputationalBackends.AbstractDistributedBackend, velocity_fields_batch, geometry, ℓ, filter; kwargs...) =
    _cg_flux_batch_distributed!(velocity_fields_batch, geometry, ℓ, filter; kwargs...)

# Any other execution backend has no batch hook — refuse rather than silently run serial.
_cg_flux_batch!(be::ComputationalBackends.AbstractExecutionBackend, velocity_fields_batch, geometry, ℓ, filter; kwargs...) =
    throw(ArgumentError("calculate_coarse_graining_flux_batch supports SerialBackend(), ThreadedBackend(), and DistributedBackend(); " *
                        "got execution = $(typeof(be))."))

# ---------------------------------------------------------------------------
# The other coarse-graining diagnostics CoarseGrainingEnergyFluxes provides, on the same domain
# argument as the flux: coordinate vectors or a `FlowGeometries` grid. Stubs here, methods in the
# extension.
# ---------------------------------------------------------------------------

"""
    calculate_band_energies(velocity_fields, geometry, filter, scales; kwargs...)

Scale-band kinetic energies: the energy in each band between successive entries of `scales`, from the
filtered field at each scale. `geometry` is the coordinate vectors or a `FlowGeometries` grid, as for
[`calculate_coarse_graining_flux`](@ref). Pass `maps = true` for the per-point band fields alongside
the integrals. Requires `CoarseGrainingEnergyFluxes` to be loaded.
"""
function calculate_band_energies(args...; kwargs...)
    throw(ArgumentError(
        "calculate_band_energies requires CoarseGrainingEnergyFluxes.jl. " *
        "Run `using CoarseGrainingEnergyFluxes` to load the extension."))
end

"""
    calculate_enstrophy_flux(velocity_fields, geometry, ℓ, filter; kwargs...)

Pointwise cross-scale **enstrophy** flux at filter scale `ℓ` — the vorticity carried through the
tracer-variance flux, the enstrophy counterpart of [`calculate_coarse_graining_flux`](@ref).
`geometry` is the coordinate vectors or a `FlowGeometries` grid. Requires
`CoarseGrainingEnergyFluxes` to be loaded.
"""
function calculate_enstrophy_flux(args...; kwargs...)
    throw(ArgumentError(
        "calculate_enstrophy_flux requires CoarseGrainingEnergyFluxes.jl. " *
        "Run `using CoarseGrainingEnergyFluxes` to load the extension."))
end

# One-line show (the workspace holds CGEF filter/derivative plans → default field-dump show can segfault).
Base.show(io::IO, ::CoarseGrainingFluxWorkspace) = print(io, "CoarseGrainingFluxWorkspace(…)")
Base.show(io::IO, ::MIME"text/plain", w::CoarseGrainingFluxWorkspace) = show(io, w)

end # module CoarseGrainingFlux
