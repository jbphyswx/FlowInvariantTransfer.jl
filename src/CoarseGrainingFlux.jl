module CoarseGrainingFlux

using ..Types: CoarseGrainingFluxMethod, CoarseGrainingFluxResult, AbstractFilter, AbstractFieldDecomposition, NoDecomposition
using ..Decomposition: decompose_field

export calculate_coarse_graining_flux

# ---------------------------------------------------------------------------
# Internal stub — overridden by FlowInvariantTransferCGEFExt when
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
- `mask::Union{Nothing,AbstractMatrix{Bool}}=nothing`: Wet/dry mask (`true` = wet).
  If `nothing`, all points treated as wet.
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

end # module CoarseGrainingFlux
