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
`(filter, scale, mask, execution)`. Stub overridden by the CoarseGrainingEnergyFluxes extension.
"""
function _cg_flux_workspace(args...; kwargs...)
    throw(ArgumentError(
        "CoarseGrainingFluxWorkspace requires CoarseGrainingEnergyFluxes.jl. " *
        "Run `using CoarseGrainingEnergyFluxes` to load the extension."))
end

"""
    _cg_flux_cgef!(ws, velocity_fields; kwargs...)

In-place coarse-graining flux reusing `ws` (filter/scale/mask/execution fixed at construction).
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
                                mask=nothing, return_diagnostics=false, mask_strategy=…, vector_filter=nothing,
                                execution=…)

Preallocated, reusable resources for [`calculate_coarse_graining_flux!`](@ref), built for a **fixed**
`(filter, scale ℓ, mask, execution)` configuration: the grid, its CGEF `ΠWorkspace` (filtered fields and
stress/strain scratch), the `Π_ℓ(x)` output buffer, the derivative plan, the filter plan, and — when
`return_diagnostics=true` — the `τ̄`/`S̄` diagnostic buffers. Build once, then apply to snapshots on the
same grid with `calculate_coarse_graining_flux!(ws, velocity_fields)`. A different scale, filter, mask
or execution backend needs a new workspace.

Every buffer lives in the memory of the filter plan's backend, a device's under `GPUBackend(dev)`. A
workspace on a device built from host fields holds in `stage` the device buffers its host fields are
copied into once per call.

Requires `CoarseGrainingEnergyFluxes` to be loaded.
"""
struct CoarseGrainingFluxWorkspace{G, W, P, D, DP, FP, K, S, A, ST}
    grid::G                       # FlowGeometries.Grids.StructuredGrid (via the CGEF extension)
    cgef_workspace::W             # CGEF.Diagnostics.ΠWorkspace
    Π_out::P                      # reused Π_ℓ(x) output buffer (N-D: Matrix in 2D, Array{,3} in 3D)
    diagnostics::D                # nothing, or (τ_arr, S_arr) reused diagnostic buffers
    deriv_plan::DP                # CGEF.Derivatives.derivative_plan of the filter's output grid
    filter_plan::FP               # CGEF filter plan for the fixed (kernel, scale, mask, execution)
    kernel::K                     # the CGEF kernel the filter plan was built with
    scale::S                      # the filter scale ℓ it was built for
    weights::A                    # per cell: its area over the active area, 0 where not averaged
    stage::ST                     # one staging buffer per velocity component, each `nothing` if unused
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
using FlowInvariantTransfer, CoarseGrainingEnergyFluxes
result = calculate_coarse_graining_flux((u, v), (x, y), ℓ, FlowInvariantTransfer.Types.GaussianFilter())
```

# Arguments
- `velocity_fields`: Tuple of D real arrays `(u, v[, w])` — velocity components.
- `coords_vecs`: Tuple of D coordinate vectors `(x, y[, z])`.
- `ℓ::Real`: Filter length scale (same units as coordinates).
- `filter::AbstractFilter`: `GaussianFilter()`, `SharpSpectralFilter()`, or `TopHatFilter()`.

# Keyword Arguments
- `decomposition::AbstractFieldDecomposition`: `NoDecomposition()` (default), `HelmholtzDecomposition()`, `RotationalDecomposition()`, or `DivergentDecomposition()`.
- `decomposition_kwargs::NamedTuple = (;)`: keywords of the decomposition (for the Helmholtz family
  `solver` and `boundary`; for the helical and toroidal/poloidal splits `transform` and `execution`).
- `return_diagnostics::Bool=false`: Also return τ̄ᵢⱼ and S̄ᵢⱼ fields.
- `mask::Union{Nothing,AbstractMatrix{Bool}}=nothing`: boolean point mask (`true` = point included in
  the computation, `false` = excluded). If `nothing`, all points are included.
- `execution::AbstractExecutionBackend = AutoBackend()`: the backend the filter runs on, passed to
  `CoarseGrainingEnergyFluxes.Filtering.plan_filter`, which resolves `AutoBackend`. Under
  `GPUBackend(dev)` the flux is computed in `dev`'s memory: host fields are copied there once per call,
  `flux_field` and the diagnostics are on `dev`, and `mean_flux` is a host scalar.
- `mask_strategy = ZeroFill()`: CGEF's treatment of the masked points in the filter; `radius`: with
  coordinate vectors `(lon, lat)`, the radius of the sphere they lie on.
- `vector_filter = nothing`: CGEF's filter of vectors and tensors (`CGEF.Filtering.GeneralizedConvolution()`
  or `CartesianComponents()`), `nothing` for the one `CGEF.Filtering.resolve_vector_filter` resolves
  for the grid and `mask_strategy`. On the surface of a sphere under `ZeroFill` that is the generalized
  convolution, whose spin-weighted transforms run through NUFSHT when it is loaded (FastSphericalHarmonics
  on its own grids) and through CGEF's direct sum otherwise.

Coordinate vectors, a longitude among them, build a `FlowGeometries` grid with no periodic direction
(`CoarseGrainingEnergyFluxes` takes the exterior of a bounded grid as zero velocity); pass a grid that
declares its periodic directions (`topology`, or a `PeriodicUniformAxis` longitude) for a periodic domain.

# Returns
`CoarseGrainingFluxResult`; with a decomposition a NamedTuple of them, one per component, alongside the
decomposition's `converged`, `iterations` and `residual` when it reports them.
"""
function calculate_coarse_graining_flux(
    velocity_fields::Tuple,
    geometry,
    ℓ::Real,
    filter::Types.AbstractFilter;
    decomposition::Types.AbstractFieldDecomposition = Types.NoDecomposition(),
    decomposition_kwargs::NamedTuple = (;),
    kwargs...,
)
    decomposition isa Types.NoDecomposition || geometry isa Tuple && throw(ArgumentError(
        "a field decomposition solves a boundary-value problem on the domain, and coordinate vectors " *
        "carry no topology, mask or geometry to solve it on. Pass a `FlowGeometries` grid, or decompose " *
        "upstream and pass the component fields."))
    decomposed = Decomposition.decompose_field(decomposition, velocity_fields, geometry; decomposition_kwargs...)
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

# The entries a physical decomposition reports its solves under, beside its components.
const _SOLVE_KEYS = (:converged, :iterations, :residual)

function _calculate_coarse_graining_flux_decomposed(
    decomposed::NamedTuple,
    velocity_fields::Tuple,
    geometry,
    ℓ::Real,
    filter::Types.AbstractFilter;
    kwargs...,
)
    parts = Base.structdiff(decomposed, NamedTuple{_SOLVE_KEYS})
    fluxes = map(fields -> _cg_flux_cgef(fields, geometry, ℓ, filter; kwargs...), parts)
    return merge(fluxes, Base.structdiff(decomposed, parts))
end

"""
    calculate_coarse_graining_flux!(ws::CoarseGrainingFluxWorkspace, velocity_fields)
        -> CoarseGrainingFluxResult

In-place coarse-graining flux reusing the preallocated `ws`, allocation-free. The filter, scale, mask,
execution backend and `return_diagnostics` are fixed by [`CoarseGrainingFluxWorkspace`](@ref); call
this per snapshot on the same grid.

The returned result wraps the reused `ws.Π_out` buffer (a subsequent call overwrites it); copy
`result.flux_field` if the field must persist across calls. A decomposition is applied upstream, by
passing the component fields.

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
iterable of velocity-component tuples `(u, v[, w])`). One `CoarseGrainingFluxWorkspace` (grid + CGEF
plans, fixed for `(ℓ, filter)`) is built per worker and reused across its snapshots.

`execution` runs the batch axis, and each snapshot's filter runs on the backend it implies: on
`SerialBackend()`, `ThreadedBackend()` (requires `using OhMyThreads`, threads over snapshots),
`DistributedBackend(inner)` and `MPIBackend(inner)` every snapshot is filtered serially (or on the
device of an `inner = GPUBackend(dev)`), and on `GPUBackend(dev)` every snapshot is filtered on `dev`.
Each result owns an independent copy of its flux field. Only `NoDecomposition` is supported; `kwargs`
(`mask`, `return_diagnostics`, `radius`, …) are forwarded to the workspace. Requires
`CoarseGrainingEnergyFluxes` to be loaded.
"""
function calculate_coarse_graining_flux_batch(
    velocity_fields_batch,
    geometry,
    ℓ::Real,
    filter::Types.AbstractFilter;
    execution::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.SerialBackend(),
    kwargs...,
)
    length(velocity_fields_batch) == 0 &&
        return _cg_result_type(eltype(velocity_fields_batch), get(kwargs, :return_diagnostics, false))[]
    return _cg_flux_batch!(Types.resolve_execution(execution), velocity_fields_batch, geometry, ℓ, filter; kwargs...)
end

# The result type of fields of type `F` (a tuple of component arrays): the flux is an `Array` shaped
# like a component, the diagnostics two component axes more.
function _cg_result_type(::Type{F}, diagnostics::Bool) where {F <: Tuple}
    A = fieldtype(F, 1)
    FT = eltype(A); nd = ndims(A)
    return diagnostics ? Types.CoarseGrainingFluxResultWithDiagnostics{FT, Array{FT, nd}, Array{FT, nd + 2}} :
                         Types.CoarseGrainingFluxResult{FT, Array{FT, nd}}
end
_cg_result_type(::Type, diagnostics::Bool) =
    diagnostics ? Types.CoarseGrainingFluxResultWithDiagnostics : Types.CoarseGrainingFluxResult

# One workspace filtering every snapshot on `snapshot_execution`, reused across the batch; each result
# gets an independent copy (the workspace overwrites its output buffer on every snapshot, so the
# results must not alias it).
function _cg_flux_reuse(velocity_fields_batch, geometry, ℓ, filter,
                        snapshot_execution::ComputationalBackends.AbstractExecutionBackend; kwargs...)
    ws = CoarseGrainingFluxWorkspace(first(velocity_fields_batch), geometry, ℓ, filter;
                                     kwargs..., execution = snapshot_execution)
    return [deepcopy(calculate_coarse_graining_flux!(ws, vf)) for vf in velocity_fields_batch]
end

# Serial reference: every snapshot filtered serially.
_cg_flux_batch!(::ComputationalBackends.AbstractSerialBackend, velocity_fields_batch, geometry, ℓ, filter; kwargs...) =
    _cg_flux_reuse(velocity_fields_batch, geometry, ℓ, filter, ComputationalBackends.SerialBackend(); kwargs...)

# Threaded over the batch — overridden by the OhMyThreads extension (per-chunk workspace, serial inner).
function _cg_flux_batch_threaded!(args...; kwargs...)
    throw(ArgumentError("execution = ThreadedBackend() for the coarse-graining batch requires OhMyThreads. " *
                        "Run `using OhMyThreads` to load the extension."))
end
_cg_flux_batch!(::ComputationalBackends.AbstractThreadedBackend, velocity_fields_batch, geometry, ℓ, filter; kwargs...) =
    _cg_flux_batch_threaded!(velocity_fields_batch, geometry, ℓ, filter; kwargs...)

# A device batch: one workspace, every snapshot filtered on the device.
_cg_flux_batch!(gpu::ComputationalBackends.AbstractGPUBackend, velocity_fields_batch, geometry, ℓ, filter; kwargs...) =
    _cg_flux_reuse(velocity_fields_batch, geometry, ℓ, filter, gpu; kwargs...)

# `DistributedBackend` and `MPIBackend` batches are methods of the Distributed and MPI extensions.
_cg_flux_batch!(be::ComputationalBackends.AbstractExecutionBackend, velocity_fields_batch, geometry, ℓ, filter; kwargs...) =
    Types.batch_unavailable("calculate_coarse_graining_flux_batch", be)

# ---------------------------------------------------------------------------
# The other coarse-graining diagnostics CoarseGrainingEnergyFluxes provides, on the same domain
# argument as the flux: coordinate vectors or a `FlowGeometries` grid. Stubs here, methods in the
# extension.
# ---------------------------------------------------------------------------

"""
    calculate_band_energies(velocity_fields, geometry, filter, scales; kwargs...)

Scale-band kinetic energies: the energy in each band between successive entries of `scales`, from the
filtered field at each scale. `geometry` is the coordinate vectors or a `FlowGeometries` grid, as for
[`calculate_coarse_graining_flux`](@ref). Pass `maps = CoarseGrainingEnergyFluxes.Diagnostics.BandMaps()`
for the per-point band fields alongside the integrals. Under `execution = GPUBackend(dev)` host fields are copied to `dev` and the maps are on
`dev`. `vector_filter` is CGEF's filter of the velocity on a sphere, as for
[`calculate_coarse_graining_flux`](@ref). Requires `CoarseGrainingEnergyFluxes` to be loaded.
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
`geometry` is the coordinate vectors or a `FlowGeometries` grid. Under `execution = GPUBackend(dev)`
host fields are copied to `dev` and the flux is on `dev`. `vector_filter` is CGEF's filter of the
vector `u ω` on a sphere, as for [`calculate_coarse_graining_flux`](@ref). Requires
`CoarseGrainingEnergyFluxes` to be loaded.
"""
function calculate_enstrophy_flux(args...; kwargs...)
    throw(ArgumentError(
        "calculate_enstrophy_flux requires CoarseGrainingEnergyFluxes.jl. " *
        "Run `using CoarseGrainingEnergyFluxes` to load the extension."))
end

Base.show(io::IO, ::CoarseGrainingFluxWorkspace) = print(io, "CoarseGrainingFluxWorkspace(…)")
Base.show(io::IO, ::MIME"text/plain", w::CoarseGrainingFluxWorkspace) = show(io, w)

end # module CoarseGrainingFlux
