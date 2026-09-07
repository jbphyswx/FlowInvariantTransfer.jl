module Spherical

using ..Types: Types
using ComputationalBackends: ComputationalBackends
using SpectralBackends: SpectralBackends

export calculate_spherical_transfer, calculate_spherical_transfer!,
       SphericalTransferWorkspace, ScatteredSphericalTransferWorkspace,
       calculate_divergent_spherical_transfer, calculate_divergent_spherical_transfer!,
       DivergentSphericalTransferWorkspace, ScatteredDivergentSphericalTransferWorkspace

# ---------------------------------------------------------------------------
# Backend validation for the spherical transfer (shared by the FSH regular-grid and NUFSHT scattered
# extensions). The transform is selected by the *input sampling*: SpectralBackends.FSHTSpectralBackend ↔ regular colatitude–
# longitude grid (FastSphericalHarmonics), SpectralBackends.NUFSHTSpectralBackend ↔ scattered points (NUFSHT). Passing an
# explicit `spectral` that disagrees, or an execution backend the single-pipeline transform cannot
# honour, raises a clear error rather than silently ignoring it.
# ---------------------------------------------------------------------------

# `path` is `:regular` (FSH) or `:scattered` (NUFSHT).
function _validate_spherical_backends(spectral, execution::ComputationalBackends.AbstractExecutionBackend, path::Symbol)
    expected = path === :regular ? SpectralBackends.FSHTSpectralBackend : SpectralBackends.NUFSHTSpectralBackend
    other    = path === :regular ? "SpectralBackends.NUFSHTSpectralBackend (scattered points)" : "SpectralBackends.FSHTSpectralBackend (regular grid)"
    (spectral === nothing || spectral isa expected) || throw(ArgumentError(
        "spherical transfer on a $(path === :regular ? "regular colatitude–longitude grid" : "scattered point set") " *
        "uses $(nameof(expected)); got spectral = $(typeof(spectral)). Use $other for the other sampling."))
    execution isa Union{ComputationalBackends.SerialBackend, ComputationalBackends.ThreadedBackend} || throw(ArgumentError(
        "spherical transfer runs as a single transform pipeline; execution = $(typeof(execution)) is not a " *
        "distinct code path. Supported: ComputationalBackends.SerialBackend, ComputationalBackends.ThreadedBackend (threads the transform). Distribute " *
        "independent snapshots with `mpi_batch_map`; for a GPU transform use the NUFSHT scattered path with " *
        "device-array coordinates (the device path follows the coordinate array type, not execution=ComputationalBackends.GPUBackend())."))
    return nothing
end

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
# (Augier–Lindborg 2013; Boer 1983.)
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
    result::RES      # reused Types.SphericalTransferResult (TE/TZ/ΠE/ΠZ)
    radius::R
    lmax::Int
    dealias::Bool
end

function SphericalTransferWorkspace(args...; kwargs...)
    throw(ArgumentError(
        "SphericalTransferWorkspace requires `using FastSphericalHarmonics` (regular grid)."))
end

"""
    ScatteredSphericalTransferWorkspace(coords, lmax; ...)

Reusable buffers for the SCATTERED-point spherical transfer `!()` (NUFSHT extension). Holds the three
NUFSHT spin plans (spin-0 at `lmax`, spin-1 at `lmax`, spin-0 at the dealiased `lwork = 2·lmax`) with
the scattered points preset, plus every coefficient/gradient/reduction buffer and the result vectors.
The plans are the dominant cost (FINUFFT planning + the CG least-squares setup); building them once
lets a snapshot time series on the same points reuse them. NUFSHT plans self-finalize their FINUFFT
resources, so this struct needs no finalizer. Fields are typed via parameters so the core names no
NUFSHT type. Requires `using NUFSHT`.
"""
struct ScatteredSphericalTransferWorkspace{P0, P1, P0W, CM, CV, DC, PB, TC, QW, RES, R}
    plan0::P0        # spin-0 analysis plan at lmax
    plan1::P1        # spin-1 synthesis plan at lmax
    plan0w::P0W      # spin-0 analysis plan at lwork (dealiased)
    ζ_lm::CM         # (lmax+1, 2lmax+1) vorticity coefficients
    ψ_lm::CM         # streamfunction coefficients
    ðψ::CM           # spin-1 gradient coefficients of ψ
    ðζ::CM           # spin-1 gradient coefficients of ζ
    A_lw::CM         # (lwork+1, 2lwork+1) advection coefficients
    Gψ::CV           # (M,) ðψ at the scattered points
    Gζ::CV           # (M,) ðζ at the scattered points
    ζdata::CV        # (M,) vorticity as complex (solve input)
    Jc::CV           # (M,) Jacobian A = J(ψ,ζ) as complex (solve input)
    degcol::DC       # (lmax+1, 1) degrees 0:lmax — row-broadcast for the coefficient-space ops
    Pr::PB           # (lmax+1, 2lmax+1) real product scratch for the per-degree row-sum reduce
    Tcol::TC         # (lmax+1, 1) real per-degree column-sum scratch
    # (M,) quadrature weights, `Σw = 4π`, when the nodes carry a rule exact at degree `2·lwork`.
    # Analysis is then the weighted adjoint `Σⱼ wⱼ fⱼ conj(sYlm(xⱼ))`. `nothing` at points carrying no
    # such rule, where analysis is the least-squares fit.
    qw::QW
    result::RES      # reused Types.SphericalTransferResult
    radius::R
    lmax::Int
    lwork::Int
    rtol::R
    maxiter::Int
end

function ScatteredSphericalTransferWorkspace(args...; kwargs...)
    throw(ArgumentError("ScatteredSphericalTransferWorkspace requires `using NUFSHT` (scattered points)."))
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
    result = Types.SphericalTransferResult(
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
    result::Types.SphericalTransferResult,
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

# One-line show (these hold FSH / NUFSHT (FINUFFT-backed) plans → default field-dump show can segfault).
Base.show(io::IO, ::SphericalTransferWorkspace) = print(io, "SphericalTransferWorkspace(…)")
Base.show(io::IO, ::MIME"text/plain", w::SphericalTransferWorkspace) = show(io, w)
Base.show(io::IO, ::ScatteredSphericalTransferWorkspace) = print(io, "ScatteredSphericalTransferWorkspace(…)")
Base.show(io::IO, ::MIME"text/plain", w::ScatteredSphericalTransferWorkspace) = show(io, w)

# ===========================================================================================
# Divergent horizontal kinetic-energy spectral transfer — full (rotational + divergent) flow.
#
# Input: the horizontal velocity u = (u_θ, u_φ). Represent it as one complex spin-1 field
# U₊ = u_θ + i u_φ; its spin-1 harmonic coefficients a₊ carry the full horizontal KE
# (Σ|a₊|² = ∫|u|² dΩ, orthonormal). With the spin-(−1) coefficients a₋ of u_θ − i u_φ the field
# splits into rotational (toroidal) sym = (a₊+a₋)/2 and divergent (spheroidal) anti = (a₊−a₋)/2.
#
# Vorticity ζ = k̂·∇×u and divergence δ = ∇·u come from the eth ladder:
#     ζ_lm = −i √(l(l+1)) sym_lm,   δ_lm = +√(l(l+1)) anti_lm.
# The advection is taken in the energy-conserving skew-symmetric Lamb form
#     A = (u·∇)u + ½ u (∇·u) = ∇(½|u|²) + ζ (k̂×u) + ½ δ u = ∇K + (i ζ + ½ δ) U₊,
# with the scalar gradient ∇f = −ð f (eth) and k̂×u ↔ i U₊. The transfer into degree l is the
# spin-1 vector-harmonic projection, split into the two channels:
#     T_rot(l) = Σ_m Re{ sym*_lm Â_lm},   T_div(l) = Σ_m Re{ anti*_lm Â_lm},   T = T_rot + T_div,
# and Π(L) = −Σ_{l≤L} T(l). The skew-symmetric ½ δ u term makes Σ_l T = 0 for divergent flow too;
# in the non-divergent limit (δ = 0) T reduces to the barotropic `SphericalTransferMethod` energy
# transfer. Products are quadratic, so the advection is analysed at the dealiased degree lwork = 2·lmax
# and truncated back to l ≤ lmax. (Augier–Lindborg 2013; Burgess–Erler–Shepherd 2013.)
#
# The transform machinery lives in the extensions (FastSphericalHarmonics regular grid; NUFSHT
# scattered points); both share the finalisation below.
# ===========================================================================================

"""
    calculate_divergent_spherical_transfer(...)

Extension entry point for the divergent horizontal-KE spherical spectral transfer. Implemented in the
FastSphericalHarmonics (regular grid) and NUFSHT (scattered) extensions; loading one provides the
method. Prefer the unified `calculate_energy_transfer(DivergentSphericalTransferMethod(), …)`.
"""
function calculate_divergent_spherical_transfer end

"""
    calculate_divergent_spherical_transfer!(ws, u_θ, u_φ; kwargs...) -> DivergentSphericalTransferResult

In-place divergent spherical KE transfer reusing `ws`. Implemented in the FastSphericalHarmonics /
NUFSHT extensions. Build the workspace once with [`DivergentSphericalTransferWorkspace`](@ref)
(regular grid) or [`ScatteredDivergentSphericalTransferWorkspace`](@ref) (scattered points).
"""
function calculate_divergent_spherical_transfer! end

"""
    DivergentSphericalTransferWorkspace(lmax; ...)

Reusable buffers for the regular-grid divergent KE transfer `!()` (FastSphericalHarmonics extension;
loading it provides the constructor). FastSphericalHarmonics has no in-place transform API, so the
spin-weighted transforms/eth allocate internally on every call (an irreducible floor); the workspace
therefore just carries the reused [`DivergentSphericalTransferResult`](@ref) and the resolution
parameters. Fields are typed via parameters so the core names no extension type. Requires
`using FastSphericalHarmonics`.
"""
struct DivergentSphericalTransferWorkspace{RES, R, RW, CW, RC}
    uθw::RW          # (lwork) velocity components on the work grid
    uφw::RW
    ζw::RW           # (lwork) vorticity / divergence on the work grid
    δw::RW
    K::RW            # (lwork) kinetic energy ½|u|²
    Adv::CW          # (lwork) complex spin+1 advection A = ∇K + (iζ + ½δ)U₊
    Cw1::CW          # (lwork) spin+1 embed target
    Cw0::RW          # (lwork) spin-0 embed target
    χc::RC           # (lmax) velocity-potential coefficients χ = ∇⁻²δ
    result::RES      # reused Types.DivergentSphericalTransferResult
    radius::R
    lmax::Int
    lwork::Int
    dealias::Bool
end

function DivergentSphericalTransferWorkspace(args...; kwargs...)
    throw(ArgumentError(
        "DivergentSphericalTransferWorkspace requires `using FastSphericalHarmonics` (regular grid)."))
end

"""
    ScatteredDivergentSphericalTransferWorkspace(coords, lmax; ...)

Reusable buffers for the SCATTERED-point divergent KE transfer `!()` (NUFSHT extension; loading it
provides the constructor). Holds the five NUFSHT spin plans (spin ±1 and spin 0 at `lmax`, spin 0 and
spin +1 at the dealiased `lwork = 2·lmax`) with the points preset, plus every coefficient/field/reduction
buffer and the result. The plans are the dominant reusable cost. Fields are typed via parameters so the
core names no NUFSHT type. Requires `using NUFSHT`.
"""
struct ScatteredDivergentSphericalTransferWorkspace{PP, PM, P0, P0W, PPW, CM, CMW, CV, RV, RVW, PB, TC, QW, RES, R}
    planp::PP        # spin+1 analysis plan at lmax
    planm::PM        # spin−1 analysis plan at lmax
    plan0::P0        # spin-0 synthesis plan at lmax (vorticity/divergence)
    plan0w::P0W      # spin-0 analysis plan at lwork (K)
    planpw::PPW      # spin+1 plan at lwork (∇K synthesis, advection analysis)
    ap::CM; am::CM; sym::CM; anti::CM      # (lmax+1, 2lmax+1) spin-1 coefficient work arrays
    ζc::CM; δc::CM                          # vorticity/divergence spin-0 coefficients
    Khat::CMW                               # (lwork+1, 2lwork+1) K spin-0 coefficients
    Adv_lm::CMW                             # advection spin-1 coefficients at lwork
    Up::CV; Um::CV                          # (M,) U₊, U₋ at the points
    ζv::CV; δv::CV; Kv::CV                   # (M,) vorticity, divergence, ½|u|²
    gradK::CV; Advv::CV                      # (M,) ∇K, advection
    ladl::RV                                 # (lmax+1,1) ladder √(l(l+1))
    ladw::RVW                                # (lwork+1,1) ladder at lwork
    Pr::PB                                   # (lmax+1, 2lmax+1) real product scratch (row-sum reduce)
    Tcol::TC                                 # (lmax+1,1) real per-degree column-sum scratch
    # (M,) quadrature weights, `Σw = 4π`, when the nodes carry a rule exact at degree `2·lwork`;
    # `nothing` at points carrying no such rule. See the barotropic workspace.
    qw::QW
    result::RES
    radius::R
    lmax::Int
    lwork::Int
    rtol::R
    maxiter::Int
end

function ScatteredDivergentSphericalTransferWorkspace(args...; kwargs...)
    throw(ArgumentError(
        "ScatteredDivergentSphericalTransferWorkspace requires `using NUFSHT` (scattered points)."))
end

"""
    divergent_transfer_finalize!(result, T_rot, T_div) -> result

Shared finalisation (extension-agnostic): given the per-degree rotational and divergent channel
transfers already written into `result.rotational_transfer` / `result.divergent_transfer`, fill the
total `energy_transfer = T_rot + T_div` and all three cumulative fluxes `Π(L) = −Σ_{l≤L} T(l)`.
"""
function divergent_transfer_finalize!(result::Types.DivergentSphericalTransferResult)
    Trot = result.rotational_transfer
    Tdiv = result.divergent_transfer
    T = result.energy_transfer
    @inbounds for i in eachindex(T)
        T[i] = Trot[i] + Tdiv[i]
    end
    _neg_cumsum!(result.energy_flux, T)
    _neg_cumsum!(result.rotational_flux, Trot)
    _neg_cumsum!(result.divergent_flux, Tdiv)
    return result
end

# One-line show (these hold FSH / NUFSHT (FINUFFT-backed) plans → default field-dump show can segfault).
Base.show(io::IO, ::DivergentSphericalTransferWorkspace) = print(io, "DivergentSphericalTransferWorkspace(…)")
Base.show(io::IO, ::MIME"text/plain", w::DivergentSphericalTransferWorkspace) = show(io, w)
Base.show(io::IO, ::ScatteredDivergentSphericalTransferWorkspace) = print(io, "ScatteredDivergentSphericalTransferWorkspace(…)")
Base.show(io::IO, ::MIME"text/plain", w::ScatteredDivergentSphericalTransferWorkspace) = show(io, w)

end # module Spherical
