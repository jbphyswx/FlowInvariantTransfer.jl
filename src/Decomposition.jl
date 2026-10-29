module Decomposition

using ..Types: Types
using FlowTransformBindings: FlowTransformBindings as FTB

export decompose_field, decompose_field!, helmholtz_project_spectral!

# ---------------------------------------------------------------------------
# Default fallback for physical-space decomposition
# ---------------------------------------------------------------------------

"""
    decompose_field(decomp::AbstractFieldDecomposition, fields::Tuple, domain; kwargs...)

Decompose a physical-space velocity field `fields` (e.g. `(u, v)`) with the given strategy on the
`FlowGeometries` grid `domain`. The solver is selected from the grid's geometry, sampling, mask and
topology: the Helmholtz family through the HelmholtzDecomposition extension, the helical and
toroidal/poloidal splits through the FlowFieldSpectra extension.

Returns a `NamedTuple` of component tuples keyed by the decomposition's names (`rotational`,
`divergent`; `positive`, `negative`; `toroidal`, `poloidal`). The Helmholtz family solves for
potentials and adds `converged`, `iterations` and `residual`, the least converged of its solves;
`RotationalDecomposition`/`DivergentDecomposition` return that one component.
`NoDecomposition` returns the components untouched and also accepts coordinate vectors.
"""
function decompose_field(::Types.NoDecomposition, fields::Tuple, domain; kwargs...)
    return fields
end
decompose_field(::Types.NoDecomposition, fields::Tuple, coords::Tuple; kwargs...) = fields

function decompose_field(decomp::Types.AbstractFieldDecomposition, fields::Tuple, domain; kwargs...)
    return _decompose_field_physical(decomp, fields, domain; kwargs...)
end

# The package whose extension decomposes a physical field on a FlowGeometries grid.
_physical_package(::Union{Types.HelicalDecomposition, Types.ToroidalPoloidalDecomposition}) = "FlowFieldSpectra"
_physical_package(::Types.AbstractFieldDecomposition) = "HelmholtzDecomposition"

# Reached when no extension method takes `domain`: the decomposition's package is not loaded, or
# `domain` is not a FlowGeometries grid.
function _decompose_field_physical(decomp::Types.AbstractFieldDecomposition, fields::Tuple, domain; kwargs...)
    throw(ArgumentError(
        "physical-space $(nameof(typeof(decomp))) runs on a FlowGeometries grid with " *
        "`using $(_physical_package(decomp))`; got a domain of type $(typeof(domain))."
    ))
end

# ---------------------------------------------------------------------------
# Spectral-space decomposition (Fourier space)
# ---------------------------------------------------------------------------

"""
    decompose_field(decomp::AbstractFieldDecomposition, velocity_hat::AbstractArray{<:Complex}, ks)

Decompose a spectral-space velocity field `velocity_hat` along the wavenumbers `ks`. Each returned
array has the shape of `velocity_hat`.

- [`HelicalDecomposition`](@ref FlowInvariantTransfer.Types.HelicalDecomposition) (3D) returns
  `(positive = u₊, negative = u₋)`, the positive/negative-helicity vector components; for an
  incompressible field `positive .+ negative ≈ velocity_hat`.
- [`ToroidalPoloidalDecomposition`](@ref FlowInvariantTransfer.Types.ToroidalPoloidalDecomposition)
  (3D) returns `(toroidal = u_tor, poloidal = u_pol)` in the
  Craya–Herring frame; both are divergence-free and sum to the solenoidal part of `velocity_hat`.
- `NoDecomposition` returns `velocity_hat`.
"""
function decompose_field(::Types.NoDecomposition, velocity_hat::AbstractArray{<:Complex}, ks)
    return velocity_hat
end

function decompose_field(decomp::Types.AbstractFieldDecomposition, velocity_hat::AbstractArray{<:Complex}, ks)
    return _decompose_field_spectral(decomp, velocity_hat, ks)
end

# ---------------------------------------------------------------------------
# Helical (Craya–Herring) decomposition — pure per-mode linear algebra, no external solver
# ---------------------------------------------------------------------------

# Craya–Herring frame at one mode: the two unit vectors ⊥ k̂. `e1 = k̂ × ẑ = (k̂y,−k̂x,0)` unless k̂ is
# (nearly) ∥ ẑ, where `e1 = k̂ × x̂ = (0,k̂z,−k̂y)`; `e2 = k̂ × e1`. All zero at the DC mode.
@inline function _craya_herring(kx::FT, ky::FT, kz::FT) where {FT}
    kk = sqrt(kx^2 + ky^2 + kz^2)
    invkk = kk > 0 ? inv(kk) : zero(FT)
    k̂x = kx * invkk; k̂y = ky * invkk; k̂z = kz * invkk
    sel = abs(k̂z) < FT(0.9)
    e1x = sel ? k̂y : zero(FT)
    e1y = sel ? -k̂x : k̂z
    e1z = sel ? zero(FT) : -k̂y
    n1sq = e1x^2 + e1y^2 + e1z^2
    invn1 = n1sq > 0 ? inv(sqrt(n1sq)) : zero(FT)
    e1x *= invn1; e1y *= invn1; e1z *= invn1
    return (e1x, e1y, e1z, k̂y * e1z - k̂z * e1y, k̂z * e1x - k̂x * e1z, k̂x * e1y - k̂y * e1x)
end

# Component `c` of the sign-`s` helical part at one mode. `h_± = (e1 ± i e2)/√2`, the coefficient is
# `û·h_±*`, and the part is that coefficient times `h_±`.
@inline function _helical_at(c::Int, s::Int, kx::FT, ky::FT, kz::FT, u1, u2, u3) where {FT}
    e1x, e1y, e1z, e2x, e2y, e2z = _craya_herring(kx, ky, kz)
    is2 = inv(sqrt(FT(2)))
    ue1 = u1 * e1x + u2 * e1y + u3 * e1z
    ue2 = u1 * e2x + u2 * e2y + u3 * e2z
    coef = s > 0 ? (ue1 - im * ue2) * is2 : (ue1 + im * ue2) * is2
    e1c = c == 1 ? e1x : (c == 2 ? e1y : e1z)
    e2c = c == 1 ? e2x : (c == 2 ? e2y : e2z)
    return coef * (e1c + (s > 0 ? im : -im) * e2c) * is2
end

function _decompose_field_spectral(::Types.HelicalDecomposition, velocity_hat::AbstractArray{<:Complex}, ks)
    nd = length(ks)
    nd == 3 || throw(ArgumentError("HelicalDecomposition is defined in 3D only (got nd=$nd)."))
    D  = size(velocity_hat, nd + 1)
    D >= 3 || throw(ArgumentError("HelicalDecomposition needs ≥3 velocity components (got D=$D)."))
    up = similar(velocity_hat); um = similar(velocity_hat)
    return decompose_field!((positive = up, negative = um), Types.HelicalDecomposition(), velocity_hat, ks)
end

"""
    decompose_field!(out::NamedTuple, decomp, velocity_hat, ks) -> out

Spectral decomposition into the caller-provided component buffers of `out` (`(positive, negative)` or
`(toroidal, poloidal)`), allocation-free and device-generic.

Both components are Hermitian for a real field, so either spectral layout represents them: under
`k ↦ −k` the frame obeys `e1 ↦ −e1` and `e2 ↦ e2`, which carries `u_±(−k) = conj(u_±(k))`. The frame
is built from the derivative wavenumbers the package's curl and divergence use, zero along an even
axis at its Nyquist slot, so a mode there has the frame of its own mirror and is split as those
operators see it.
"""
function decompose_field! end

function decompose_field!(out::NamedTuple, ::Types.HelicalDecomposition,
                          velocity_hat::AbstractArray{<:Complex}, ks)
    nd = length(ks)
    nd == 3 || throw(ArgumentError("HelicalDecomposition is defined in 3D only (got nd=$nd)."))
    D  = size(velocity_hat, nd + 1)
    FT = real(eltype(velocity_hat))
    up = out.positive; um = out.negative
    kg = FTB.wavenumber_arrays(velocity_hat, FT, ks; derivative = true)
    kxg, kyg, kzg = kg[1], kg[2], kg[3]
    u1 = selectdim(velocity_hat, nd + 1, 1)
    u2 = selectdim(velocity_hat, nd + 1, 2)
    u3 = selectdim(velocity_hat, nd + 1, 3)
    for c in 1:3
        selectdim(up, nd + 1, c) .= _helical_at.(c,  1, kxg, kyg, kzg, u1, u2, u3)
        selectdim(um, nd + 1, c) .= _helical_at.(c, -1, kxg, kyg, kzg, u1, u2, u3)
    end
    # Components beyond the first 3 (if any) carry no helicity.
    for c in 4:D
        selectdim(up, nd + 1, c) .= zero(eltype(up))
        selectdim(um, nd + 1, c) .= zero(eltype(um))
    end
    return out
end

# Toroidal/poloidal frame at one mode. `e1 = normalise(k̂ × ẑ) = (k̂y,−k̂x,0)/|k̂⊥|`, falling back to x̂
# when k ∥ ẑ; `e2 = k̂ × e1` (with `e1z = 0`). `nz` is 0 at the DC mode, which carries neither part.
@inline function _tor_pol_frame(kx::FT, ky::FT, kz::FT) where {FT}
    kk = sqrt(kx^2 + ky^2 + kz^2)
    invkk = kk > 0 ? inv(kk) : zero(FT)
    k̂x = kx * invkk; k̂y = ky * invkk; k̂z = kz * invkk
    k̂perp = sqrt(k̂x^2 + k̂y^2)
    selp = k̂perp > 0
    invkp = selp ? inv(k̂perp) : zero(FT)
    e1x = selp ? k̂y * invkp : one(FT)
    e1y = selp ? -k̂x * invkp : zero(FT)          # e1z ≡ 0 in both branches
    # The poloidal part (u·e2)e2 is invariant under e2 ↦ −e2, so the sign of the k ∥ ẑ fallback
    # e2 = (0, ±1, 0) does not matter.
    return (e1x, e1y, -k̂z * e1y, k̂z * e1x, k̂x * e1y - k̂y * e1x, kk > 0 ? one(FT) : zero(FT))
end

@inline function _tor_pol_at(c::Int, pol::Bool, kx::FT, ky::FT, kz::FT, u1, u2, u3) where {FT}
    e1x, e1y, e2x, e2y, e2z, nz = _tor_pol_frame(kx, ky, kz)
    if pol
        c2 = (u1 * e2x + u2 * e2y + u3 * e2z) * nz
        return c2 * (c == 1 ? e2x : (c == 2 ? e2y : e2z))
    else
        c1 = (u1 * e1x + u2 * e1y) * nz
        # The toroidal part's third component is c1·e1z = 0.
        return c == 3 ? zero(c1) : c1 * (c == 1 ? e1x : e1y)
    end
end

function _decompose_field_spectral(::Types.ToroidalPoloidalDecomposition, velocity_hat::AbstractArray{<:Complex}, ks)
    nd = length(ks)
    nd == 3 || throw(ArgumentError("ToroidalPoloidalDecomposition is defined in 3D only (got nd=$nd)."))
    D  = size(velocity_hat, nd + 1)
    D >= 3 || throw(ArgumentError("ToroidalPoloidalDecomposition needs ≥3 velocity components (got D=$D)."))
    tor = similar(velocity_hat); pol = similar(velocity_hat)
    return decompose_field!((toroidal = tor, poloidal = pol),
                            Types.ToroidalPoloidalDecomposition(), velocity_hat, ks)
end

function decompose_field!(out::NamedTuple, ::Types.ToroidalPoloidalDecomposition,
                          velocity_hat::AbstractArray{<:Complex}, ks)
    nd = length(ks)
    nd == 3 || throw(ArgumentError("ToroidalPoloidalDecomposition is defined in 3D only (got nd=$nd)."))
    D  = size(velocity_hat, nd + 1)
    FT = real(eltype(velocity_hat))
    tor = out.toroidal; pol = out.poloidal
    kg = FTB.wavenumber_arrays(velocity_hat, FT, ks; derivative = true)
    kxg, kyg, kzg = kg[1], kg[2], kg[3]
    u1 = selectdim(velocity_hat, nd + 1, 1)
    u2 = selectdim(velocity_hat, nd + 1, 2)
    u3 = selectdim(velocity_hat, nd + 1, 3)
    for c in 1:3
        selectdim(tor, nd + 1, c) .= _tor_pol_at.(c, false, kxg, kyg, kzg, u1, u2, u3)
        selectdim(pol, nd + 1, c) .= _tor_pol_at.(c, true,  kxg, kyg, kzg, u1, u2, u3)
    end
    for c in 4:D
        selectdim(tor, nd + 1, c) .= zero(eltype(tor))
        selectdim(pol, nd + 1, c) .= zero(eltype(pol))
    end
    return out
end

# Stub overridden by FlowInvariantTransferHelmholtzDecompositionExt when HelmholtzDecomposition.jl is loaded
function _decompose_field_spectral(decomp::Types.AbstractFieldDecomposition, ::AbstractArray{<:Complex}, ::Any)
    throw(ArgumentError(
        "Spectral-space decomposition ($(typeof(decomp))) requires HelmholtzDecomposition.jl. " *
        "Run `using HelmholtzDecomposition` to load the extension."
    ))
end

"""
    helmholtz_project_spectral!(û_rot, û_div, û_harm, velocity_hat, ks)

In-place spectral Helmholtz–Hodge projection of `velocity_hat` into its rotational, divergent, and
harmonic (k = 0 / constant) parts (`û_rot + û_div + û_harm == velocity_hat`), writing into the three
caller-provided buffers — allocation-free. Provided by the `HelmholtzDecomposition.jl` extension; this
core stub errors until it is loaded.
"""
function helmholtz_project_spectral!(args...; kwargs...)
    throw(ArgumentError(
        "helmholtz_project_spectral! requires HelmholtzDecomposition.jl. " *
        "Run `using HelmholtzDecomposition` to load the extension."
    ))
end

end # module Decomposition
