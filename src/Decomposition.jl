module Decomposition

using ..Types: AbstractFieldDecomposition, NoDecomposition, HelmholtzDecomposition, RotationalDecomposition, DivergentDecomposition, HelicalDecomposition, ToroidalPoloidalDecomposition

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
    decompose_field(decomp::AbstractFieldDecomposition, velocity_hat::AbstractArray{<:Complex}, ks)

Decompose a spectral-space velocity field `velocity_hat` along the wavenumbers `ks`.
"""
function decompose_field(::NoDecomposition, velocity_hat::AbstractArray{<:Complex}, ks)
    return velocity_hat
end

function decompose_field(decomp::AbstractFieldDecomposition, velocity_hat::AbstractArray{<:Complex}, ks)
    return _decompose_field_spectral(decomp, velocity_hat, ks)
end

# ---------------------------------------------------------------------------
# Helical (Craya–Herring) decomposition — pure per-mode linear algebra, no external solver
# ---------------------------------------------------------------------------

@inline _cross3(a, b) = (a[2]*b[3] - a[3]*b[2], a[3]*b[1] - a[1]*b[3], a[1]*b[2] - a[2]*b[1])

"""
    decompose_field(::HelicalDecomposition, velocity_hat, ks) -> (positive=u₊, negative=u₋)

Project a 3D spectral velocity onto the positive/negative-helicity vector components (see
[`HelicalDecomposition`](@ref)). Each returned array has the shape of `velocity_hat`; for an
incompressible field `positive .+ negative ≈ velocity_hat`.
"""
function _decompose_field_spectral(::HelicalDecomposition, velocity_hat::AbstractArray{<:Complex}, ks)
    nd = length(ks)
    nd == 3 || throw(ArgumentError("HelicalDecomposition is defined in 3D only (got nd=$nd)."))
    ns = size(velocity_hat)[1:nd]
    D  = size(velocity_hat, nd + 1)
    D >= 3 || throw(ArgumentError("HelicalDecomposition needs ≥3 velocity components (got D=$D)."))
    FT = real(eltype(velocity_hat))
    up = fill!(similar(velocity_hat), zero(eltype(velocity_hat)))
    um = fill!(similar(velocity_hat), zero(eltype(velocity_hat)))
    invsqrt2 = inv(sqrt(FT(2)))
    @inbounds for I in CartesianIndices(ns)
        kx = FT(ks[1][I[1]]); ky = FT(ks[2][I[2]]); kz = FT(ks[3][I[3]])
        kk = sqrt(kx*kx + ky*ky + kz*kz)
        kk == 0 && continue                              # DC mode carries no helicity
        k̂ = (kx/kk, ky/kk, kz/kk)
        # Reference vector not (nearly) parallel to k̂, so k̂×ref is well-conditioned.
        ref = abs(k̂[3]) < FT(0.9) ? (zero(FT), zero(FT), one(FT)) : (one(FT), zero(FT), zero(FT))
        e1 = _cross3(k̂, ref)
        n1 = sqrt(e1[1]^2 + e1[2]^2 + e1[3]^2)
        e1 = (e1[1]/n1, e1[2]/n1, e1[3]/n1)              # unit, ⊥ k̂
        e2 = _cross3(k̂, e1)                              # unit, ⊥ k̂ and e1
        u1 = velocity_hat[I, 1]; u2 = velocity_hat[I, 2]; u3 = velocity_hat[I, 3]
        ue1 = u1*e1[1] + u2*e1[2] + u3*e1[3]
        ue2 = u1*e2[1] + u2*e2[2] + u3*e2[3]
        # h_± = (e1 ± i e2)/√2 ; coefficients u_± = û·h_±*  (h_+* = (e1−ie2)/√2, h_-* = (e1+ie2)/√2)
        upc = (ue1 - im*ue2) * invsqrt2
        umc = (ue1 + im*ue2) * invsqrt2
        for c in 1:3
            hpc = (e1[c] + im*e2[c]) * invsqrt2          # h_+ component c
            hmc = (e1[c] - im*e2[c]) * invsqrt2          # h_- component c
            up[I, c] = upc * hpc
            um[I, c] = umc * hmc
        end
    end
    return (positive = up, negative = um)
end

"""
    decompose_field(::ToroidalPoloidalDecomposition, velocity_hat, ks) -> (toroidal=u_tor, poloidal=u_pol)

Split a 3D solenoidal velocity into toroidal (horizontal/vortical) and poloidal (vertical/wave)
components in the Craya–Herring frame (see [`ToroidalPoloidalDecomposition`](@ref)). Both returned
arrays are divergence-free and sum to the solenoidal part of `velocity_hat`.
"""
function _decompose_field_spectral(::ToroidalPoloidalDecomposition, velocity_hat::AbstractArray{<:Complex}, ks)
    nd = length(ks)
    nd == 3 || throw(ArgumentError("ToroidalPoloidalDecomposition is defined in 3D only (got nd=$nd)."))
    ns = size(velocity_hat)[1:nd]
    D  = size(velocity_hat, nd + 1)
    D >= 3 || throw(ArgumentError("ToroidalPoloidalDecomposition needs ≥3 velocity components (got D=$D)."))
    FT = real(eltype(velocity_hat))
    tor = fill!(similar(velocity_hat), zero(eltype(velocity_hat)))
    pol = fill!(similar(velocity_hat), zero(eltype(velocity_hat)))
    ẑ = (zero(FT), zero(FT), one(FT))
    @inbounds for I in CartesianIndices(ns)
        kx = FT(ks[1][I[1]]); ky = FT(ks[2][I[2]]); kz = FT(ks[3][I[3]])
        kk = sqrt(kx*kx + ky*ky + kz*kz)
        kk == 0 && continue
        k̂ = (kx/kk, ky/kk, kz/kk)
        kperp = sqrt(kx*kx + ky*ky)
        if kperp > 0
            e1 = _cross3(k̂, ẑ)                           # horizontal, ⊥ k (toroidal dir)
            n1 = sqrt(e1[1]^2 + e1[2]^2 + e1[3]^2)
            e1 = (e1[1]/n1, e1[2]/n1, e1[3]/n1)
            e2 = _cross3(k̂, e1)                          # poloidal dir, unit, ⊥ k and e1
        else
            e1 = (one(FT), zero(FT), zero(FT))           # k ∥ ẑ: degenerate, arbitrary horizontal pair
            e2 = (zero(FT), one(FT), zero(FT))
        end
        u1 = velocity_hat[I, 1]; u2 = velocity_hat[I, 2]; u3 = velocity_hat[I, 3]
        c1 = u1*e1[1] + u2*e1[2] + u3*e1[3]              # toroidal coefficient
        c2 = u1*e2[1] + u2*e2[2] + u3*e2[3]              # poloidal coefficient
        for c in 1:3
            tor[I, c] = c1 * e1[c]
            pol[I, c] = c2 * e2[c]
        end
    end
    return (toroidal = tor, poloidal = pol)
end

# Stub overridden by FlowInvariantTransferHelmholtzDecompositionExt when HelmholtzDecomposition.jl is loaded
function _decompose_field_spectral(decomp::AbstractFieldDecomposition, velocity_hat::AbstractArray{<:Complex}, ks)
    throw(ArgumentError(
        "Spectral-space decomposition ($(typeof(decomp))) requires HelmholtzDecomposition.jl. " *
        "Run `using HelmholtzDecomposition` to load the extension."
    ))
end

"""
    helmholtz_project_spectral!(args...; kwargs...)

In-place spectral Helmholtz projection of a velocity field into its rotational/divergent parts.
Provided by the `HelmholtzDecomposition.jl` extension; this core stub errors until it is loaded.
"""
function helmholtz_project_spectral!(args...; kwargs...)
    throw(ArgumentError(
        "helmholtz_project_spectral! requires HelmholtzDecomposition.jl. " *
        "Run `using HelmholtzDecomposition` to load the extension."
    ))
end

end # module Decomposition
