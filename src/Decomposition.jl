module Decomposition

using ..Types: Types

export decompose_field, helmholtz_project_spectral!

# ---------------------------------------------------------------------------
# Default fallback for physical-space decomposition
# ---------------------------------------------------------------------------

"""
    decompose_field(decomp::AbstractFieldDecomposition, fields::Tuple, coords::Tuple; kwargs...)

Decompose a physical-space velocity field `fields` (e.g. `(u, v)`) using the coordinate
vectors `coords` and the specified decomposition strategy.
"""
function decompose_field(::Types.NoDecomposition, fields::Tuple, coords::Tuple; kwargs...)
    return fields
end

function decompose_field(decomp::Types.AbstractFieldDecomposition, fields::Tuple, coords::Tuple; kwargs...)
    return _decompose_field_physical(decomp, fields, coords; kwargs...)
end

# Stub overridden by FlowInvariantTransferHelmholtzDecompositionExt when HelmholtzDecomposition.jl is loaded
function _decompose_field_physical(decomp::Types.AbstractFieldDecomposition, fields::Tuple, coords::Tuple; kwargs...)
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
function decompose_field(::Types.NoDecomposition, velocity_hat::AbstractArray{<:Complex}, ks)
    return velocity_hat
end

function decompose_field(decomp::Types.AbstractFieldDecomposition, velocity_hat::AbstractArray{<:Complex}, ks)
    return _decompose_field_spectral(decomp, velocity_hat, ks)
end

# ---------------------------------------------------------------------------
# Helical (Craya–Herring) decomposition — pure per-mode linear algebra, no external solver
# ---------------------------------------------------------------------------

"""
    decompose_field(::HelicalDecomposition, velocity_hat, ks) -> (positive=u₊, negative=u₋)

Project a 3D spectral velocity onto the positive/negative-helicity vector components (see
[`HelicalDecomposition`](@ref)). Each returned array has the shape of `velocity_hat`; for an
incompressible field `positive .+ negative ≈ velocity_hat`.
"""
function _decompose_field_spectral(::Types.HelicalDecomposition, velocity_hat::AbstractArray{<:Complex}, ks)
    nd = length(ks)
    nd == 3 || throw(ArgumentError("HelicalDecomposition is defined in 3D only (got nd=$nd)."))
    ns = size(velocity_hat)[1:nd]
    D  = size(velocity_hat, nd + 1)
    D >= 3 || throw(ArgumentError("HelicalDecomposition needs ≥3 velocity components (got D=$D)."))
    FT = real(eltype(velocity_hat))
    invsqrt2 = inv(sqrt(FT(2)))

    # Device-generic (broadcast) helical projection: every step is an elementwise array op over the mode
    # grid — no scalar indexing — so identical code runs on CPU `Array`s and on device arrays (CuArray /
    # JLArray), and needs no KA kernel/synchronize. Per-dimension wavenumber grids are materialised in
    # `velocity_hat`'s own array type so all broadcasts stay on-device.
    kg = ntuple(nd) do d
        v = similar(velocity_hat, FT, ns[d]); copyto!(v, FT.(ks[d]))
        reshape(v, ntuple(i -> i == d ? ns[d] : 1, nd))
    end
    kxg, kyg, kzg = kg[1], kg[2], kg[3]
    kk    = @. sqrt(kxg^2 + kyg^2 + kzg^2)
    invkk = @. ifelse(kk > 0, inv(kk), zero(FT))          # 0 at DC ⇒ k̂ = 0 ⇒ u_± = 0 (no helicity)
    k̂x = @. kxg * invkk; k̂y = @. kyg * invkk; k̂z = @. kzg * invkk
    # Reference not (nearly) ∥ k̂: e1 = k̂ × ẑ = (k̂y,−k̂x,0) when |k̂z|<0.9, else k̂ × x̂ = (0,k̂z,−k̂y).
    sel = @. abs(k̂z) < FT(0.9)
    e1x = @. ifelse(sel, k̂y, zero(FT))
    e1y = @. ifelse(sel, -k̂x, k̂z)
    e1z = @. ifelse(sel, zero(FT), -k̂y)
    invn1 = @. ifelse((e1x^2 + e1y^2 + e1z^2) > 0, inv(sqrt(e1x^2 + e1y^2 + e1z^2)), zero(FT))
    e1x = @. e1x * invn1; e1y = @. e1y * invn1; e1z = @. e1z * invn1   # unit, ⊥ k̂
    e2x = @. k̂y*e1z - k̂z*e1y                              # e2 = k̂ × e1  (unit, ⊥ k̂ and e1)
    e2y = @. k̂z*e1x - k̂x*e1z
    e2z = @. k̂x*e1y - k̂y*e1x

    u1 = selectdim(velocity_hat, nd + 1, 1)
    u2 = selectdim(velocity_hat, nd + 1, 2)
    u3 = selectdim(velocity_hat, nd + 1, 3)
    ue1 = @. u1*e1x + u2*e1y + u3*e1z                     # û·e1, û·e2 (complex)
    ue2 = @. u1*e2x + u2*e2y + u3*e2z
    # h_± = (e1 ± i e2)/√2 ; coefficients u_± = û·h_±* (h_+* = (e1−ie2)/√2, h_-* = (e1+ie2)/√2)
    upc = @. (ue1 - im*ue2) * invsqrt2
    umc = @. (ue1 + im*ue2) * invsqrt2

    up = similar(velocity_hat); um = similar(velocity_hat)
    for (c, e1c, e2c) in ((1, e1x, e2x), (2, e1y, e2y), (3, e1z, e2z))
        selectdim(up, nd + 1, c) .= upc .* (e1c .+ im .* e2c) .* invsqrt2   # u_+ component c
        selectdim(um, nd + 1, c) .= umc .* (e1c .- im .* e2c) .* invsqrt2   # u_- component c
    end
    # Components beyond the first 3 (if any) carry no helicity.
    if D > 3
        for c in 4:D
            selectdim(up, nd + 1, c) .= zero(eltype(up))
            selectdim(um, nd + 1, c) .= zero(eltype(um))
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
function _decompose_field_spectral(::Types.ToroidalPoloidalDecomposition, velocity_hat::AbstractArray{<:Complex}, ks)
    nd = length(ks)
    nd == 3 || throw(ArgumentError("ToroidalPoloidalDecomposition is defined in 3D only (got nd=$nd)."))
    ns = size(velocity_hat)[1:nd]
    D  = size(velocity_hat, nd + 1)
    D >= 3 || throw(ArgumentError("ToroidalPoloidalDecomposition needs ≥3 velocity components (got D=$D)."))
    FT = real(eltype(velocity_hat))

    # Device-generic (broadcast) Craya–Herring projection: every step is an elementwise op over the mode
    # grid (no scalar indexing), so identical code runs on CPU `Array`s and device arrays (CuArray/JLArray)
    # with no KA kernel. Per-dimension wavenumber grids live in `velocity_hat`'s array type (on-device).
    kg = ntuple(nd) do d
        v = similar(velocity_hat, FT, ns[d]); copyto!(v, FT.(ks[d]))
        reshape(v, ntuple(i -> i == d ? ns[d] : 1, nd))
    end
    kxg, kyg, kzg = kg[1], kg[2], kg[3]
    kk    = @. sqrt(kxg^2 + kyg^2 + kzg^2)
    invkk = @. ifelse(kk > 0, inv(kk), zero(FT))
    k̂x = @. kxg * invkk; k̂y = @. kyg * invkk; k̂z = @. kzg * invkk
    nz = @. ifelse(kk > 0, one(FT), zero(FT))                # DC mode carries neither component

    # Toroidal unit vector e1 = normalise(k̂ × ẑ) = (k̂y,−k̂x,0)/|k̂⊥|; when k ∥ ẑ (k̂⊥=0) fall back to x̂.
    k̂perp = @. sqrt(k̂x^2 + k̂y^2)
    selp  = @. k̂perp > 0
    invkp = @. ifelse(selp, inv(k̂perp), zero(FT))
    e1x = @. ifelse(selp, k̂y * invkp, one(FT))
    e1y = @. ifelse(selp, -k̂x * invkp, zero(FT))            # e1z ≡ 0 in both branches
    # Poloidal unit vector e2 = k̂ × e1 (with e1z = 0). Its sign is irrelevant — the projection c·e2 is
    # sign-invariant — so the k ∥ ẑ fallback (0,±1,0) matches the reference (0,1,0).
    e2x = @. -k̂z * e1y
    e2y = @. k̂z * e1x
    e2z = @. k̂x * e1y - k̂y * e1x

    u1 = selectdim(velocity_hat, nd + 1, 1)
    u2 = selectdim(velocity_hat, nd + 1, 2)
    u3 = selectdim(velocity_hat, nd + 1, 3)
    c1 = @. (u1 * e1x + u2 * e1y) * nz                       # toroidal coefficient û·e1 (e1z = 0)
    c2 = @. (u1 * e2x + u2 * e2y + u3 * e2z) * nz            # poloidal coefficient û·e2

    tor = fill!(similar(velocity_hat), zero(eltype(velocity_hat)))
    pol = fill!(similar(velocity_hat), zero(eltype(velocity_hat)))
    selectdim(tor, nd + 1, 1) .= c1 .* e1x
    selectdim(tor, nd + 1, 2) .= c1 .* e1y                   # tor component 3 = c1·e1z = 0 (stays zeroed)
    selectdim(pol, nd + 1, 1) .= c2 .* e2x
    selectdim(pol, nd + 1, 2) .= c2 .* e2y
    selectdim(pol, nd + 1, 3) .= c2 .* e2z
    return (toroidal = tor, poloidal = pol)
end

# Stub overridden by FlowInvariantTransferHelmholtzDecompositionExt when HelmholtzDecomposition.jl is loaded
function _decompose_field_spectral(decomp::Types.AbstractFieldDecomposition, ::AbstractArray{<:Complex}, ::Any)
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
