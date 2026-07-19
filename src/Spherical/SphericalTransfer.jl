module Spherical

using ..Types: SphericalTransferResult

export calculate_spherical_transfer, calculate_spherical_transfer!, SphericalTransferWorkspace

# ---------------------------------------------------------------------------
# Spectral energy/enstrophy transfer for 2D non-divergent (barotropic) flow on the sphere.
#
# Streamfunction ψ, vorticity ζ = ∇²ψ  ⟹  ζ̂_lm = -l(l+1)/a² ψ̂_lm.
# Advection A = J(ψ,ζ) = u·∇ζ,  u = k̂×∇ψ  (nondivergent), computed pseudospectrally from the
# horizontal gradients ∇ψ, ∇ζ (spin-1 "eth" fields) as  A = (1/a²) Im{ conj(∇̂ψ)·∇̂ζ }.
# Degree-spectrum transfers, both conserving (Σ_l T = 0 since ∫ψ J(ψ,ζ) dΩ = 0):
#
#     T_E(l) = -Σ_m Re{ψ̂*_lm Â_lm},   T_Z(l) = Σ_m Re{ζ̂*_lm Â_lm}.
#
# The transform machinery (analysis/synthesis, eth gradients) lives in the extensions:
# FastSphericalHarmonics for regular colatitude–longitude grids; NUFSHT for scattered points.
# Both share the reduction below by flattening their coefficients to per-mode arrays.
# (Augier–Lindborg 2013; Boer 1983; see THEORY.md §"Spherical spectral transfer".)
# ---------------------------------------------------------------------------

"""
    calculate_spherical_transfer(...)

Extension entry point for the spherical spectral transfer. Implemented in the
FastSphericalHarmonics (regular grid) and NUFSHT (scattered) extensions; loading one of those
packages provides the method. Prefer the unified `calculate_energy_transfer(SphericalTransferMethod(), …)`.
"""
function calculate_spherical_transfer end

"""
    calculate_spherical_transfer!(ws::SphericalTransferWorkspace, vorticity; kwargs...) -> SphericalTransferResult

In-place spherical spectral transfer reusing `ws`. Implemented in the FastSphericalHarmonics /
NUFSHT extensions. Reuses the workspace's spectral work arrays, per-mode reduction buffers, and the
result vectors, so a repeat call on the same grid re-allocates only what the underlying transform
library itself allocates (FastSphericalHarmonics has no in-place transform API — that portion is an
irreducible floor). Build the workspace once with [`SphericalTransferWorkspace`](@ref).
"""
function calculate_spherical_transfer! end

"""
    SphericalTransferWorkspace(lmax; ...)

Reusable buffers for [`calculate_spherical_transfer!`](@ref). Constructed by the FastSphericalHarmonics
(regular grid) or NUFSHT (scattered) extension; loading one provides the constructor. The buffer
fields are typed via parameters so the core names no extension type.
"""
struct SphericalTransferWorkspace{CW, JW, GC, IV, RV, RES, R}
    Cζ::CW           # spectral work array — vorticity coefficients (extension layout)
    Cψ::CW           # spectral work array — streamfunction coefficients
    Gψ::GC           # complex spin-1 gradient ðψ on the grid (reused)
    Gζ::GC           # complex spin-1 gradient ðζ on the grid (reused)
    J::JW            # Jacobian A = J(ψ,ζ) on the (dealiased) grid
    degs::IV         # per-mode degree l
    ψv::RV           # per-mode ψ̂ (reduction input)
    ζv::RV           # per-mode ζ̂
    Av::RV           # per-mode Â
    result::RES      # reused SphericalTransferResult (TE/TZ/ΠE/ΠZ)
    radius::R
    lmax::Int
    dealias::Bool
end

function SphericalTransferWorkspace(args...; kwargs...)
    throw(ArgumentError(
        "SphericalTransferWorkspace requires a spherical-transform extension. " *
        "Run `using FastSphericalHarmonics` (regular grid) or `using NUFSHT` (scattered)."))
end

"""
    spherical_transfer_reduce(degree_of_mode, ψ_lm, ζ_lm, A_lm, lmax) -> SphericalTransferResult

Shared degree-spectrum reduction (extension-agnostic). Given, over every `(l,m)` mode `i`,
the mode's degree `degree_of_mode[i]` and the spectral coefficients `ψ_lm`, `ζ_lm`, `A_lm`
(the streamfunction, vorticity, and advection `A = J(ψ,ζ)`), accumulate

    T_E(l) = -Σ_m Re{ψ*_lm A_lm},   T_Z(l) =  Σ_m Re{ζ*_lm A_lm}

and the cumulative fluxes `Π(L) = -Σ_{l≤L} T(l)` (package convention: positive Π ⇒ up-degree
cascade). `conj`/`real` make the reduction correct for both the real-harmonic (regular-grid) and
complex spin-0 (scattered) coefficient conventions.
"""
function spherical_transfer_reduce(
    degree_of_mode::AbstractVector{<:Integer},
    ψ_lm::AbstractVector,
    ζ_lm::AbstractVector,
    A_lm::AbstractVector,
    lmax::Integer,
)
    FT = real(eltype(A_lm))
    result = SphericalTransferResult(
        collect(FT, 0:lmax), zeros(FT, lmax + 1), zeros(FT, lmax + 1),
        zeros(FT, lmax + 1), zeros(FT, lmax + 1))
    return spherical_transfer_reduce!(result, degree_of_mode, ψ_lm, ζ_lm, A_lm)
end

"""
    spherical_transfer_reduce!(result, degree_of_mode, ψ_lm, ζ_lm, A_lm) -> result

In-place degree-spectrum reduction: writes `T_E`/`T_Z` and the cumulative fluxes into the
preallocated `result` vectors (which are zeroed first), reusing them across calls.
"""
function spherical_transfer_reduce!(
    result::SphericalTransferResult,
    degree_of_mode::AbstractVector{<:Integer},
    ψ_lm::AbstractVector,
    ζ_lm::AbstractVector,
    A_lm::AbstractVector,
)
    TE = result.energy_transfer
    TZ = result.enstrophy_transfer
    fill!(TE, zero(eltype(TE)))
    fill!(TZ, zero(eltype(TZ)))
    @inbounds for i in eachindex(degree_of_mode)
        l = degree_of_mode[i]
        a = A_lm[i]
        TE[l + 1] += -real(conj(ψ_lm[i]) * a)
        TZ[l + 1] +=  real(conj(ζ_lm[i]) * a)
    end
    _neg_cumsum!(result.energy_flux, TE)
    _neg_cumsum!(result.enstrophy_flux, TZ)
    return result
end

# Π(L) = -Σ_{l≤L} T(l)  (matches the Cartesian SpectralFlux flux convention), into a preallocated Π.
function _neg_cumsum!(Π::AbstractVector, T::AbstractVector)
    acc = zero(eltype(Π))
    @inbounds for i in eachindex(T)
        acc += T[i]
        Π[i] = -acc
    end
    return Π
end

end # module Spherical
