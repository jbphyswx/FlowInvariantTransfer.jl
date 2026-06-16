module Decomposition

using ..Types: AbstractFieldDecomposition, NoDecomposition, HelmholtzDecomposition, RotationalDecomposition, DivergentDecomposition

export decompose_field, helmholtz_project_spectral!

# ---------------------------------------------------------------------------
# Default fallback for physical-space decomposition
# ---------------------------------------------------------------------------

"""
    decompose_field(decomp::AbstractFieldDecomposition, fields::Tuple, coords::Tuple; kwargs...)

Decompose a physical-space velocity field `fields` (e.g. `(u, v)`) using the coordinate
vectors `coords` and the specified decomposition strategy.
"""
function decompose_field(::NoDecomposition, fields::Tuple, coords::Tuple; kwargs...)
    return fields
end

function decompose_field(decomp::AbstractFieldDecomposition, fields::Tuple, coords::Tuple; kwargs...)
    return _decompose_field_physical(decomp, fields, coords; kwargs...)
end

# Stub overridden by FlowInvariantTransferHelmholtzDecompositionExt when HelmholtzDecomposition.jl is loaded
function _decompose_field_physical(decomp::AbstractFieldDecomposition, fields::Tuple, coords::Tuple; kwargs...)
    throw(ArgumentError(
        "Physical-space decomposition ($(typeof(decomp))) requires HelmholtzDecomposition.jl. " *
        "Run `using HelmholtzDecomposition` to load the extension."
    ))
end

# ---------------------------------------------------------------------------
# Spectral-space decomposition (Fourier space)
# ---------------------------------------------------------------------------

"""
    decompose_field(decomp::AbstractFieldDecomposition, velocity_hat::AbstractArray{<:Complex}, ks::Tuple)

Decompose a spectral-space velocity field `velocity_hat` along the wavenumbers `ks`.
"""
function decompose_field(::NoDecomposition, velocity_hat::AbstractArray{<:Complex}, ks::Tuple)
    return velocity_hat
end

function decompose_field(decomp::AbstractFieldDecomposition, velocity_hat::AbstractArray{<:Complex}, ks::Tuple)
    return _decompose_field_spectral(decomp, velocity_hat, ks)
end

# Stub overridden by FlowInvariantTransferHelmholtzDecompositionExt when HelmholtzDecomposition.jl is loaded
function _decompose_field_spectral(decomp::AbstractFieldDecomposition, velocity_hat::AbstractArray{<:Complex}, ks::Tuple)
    throw(ArgumentError(
        "Spectral-space decomposition ($(typeof(decomp))) requires HelmholtzDecomposition.jl. " *
        "Run `using HelmholtzDecomposition` to load the extension."
    ))
end

# Stub overridden by FlowInvariantTransferHelmholtzDecompositionExt when HelmholtzDecomposition.jl is loaded
function helmholtz_project_spectral!(args...; kwargs...)
    throw(ArgumentError(
        "helmholtz_project_spectral! requires HelmholtzDecomposition.jl. " *
        "Run `using HelmholtzDecomposition` to load the extension."
    ))
end

end # module Decomposition
