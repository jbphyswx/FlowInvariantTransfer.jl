module FlowInvariantTransferNUFSHTExt

using NUFSHT: NUFSHT
using FlowInvariantTransfer: FlowInvariantTransfer as FIT
using FlowInvariantTransfer.Types: SphericalTransferMethod
using FlowInvariantTransfer.Spherical: spherical_transfer_reduce

# ---------------------------------------------------------------------------
# Spherical spectral energy/enstrophy transfer at SCATTERED points on the sphere, via NUFSHT
# (non-uniform spherical-harmonic transforms). Same 2D-barotropic formulation as the FSH regular-grid
# path (THEORY.md §"Spherical spectral transfer"; core reduction: FlowInvariantTransfer.Spherical),
# but analysis/synthesis are FINUFFT-backed scattered transforms.
#
# NUFSHT's spin-weighted harmonics are the standard convention ₛYℓm = √((2ℓ+1)/4π) d^ℓ_{m,-s}(θ) e^{imφ}
# (read from NUFSHT/src/Spin.jl), so the eth ladder is exactly ð(ₛYℓm) = √((ℓ-s)(ℓ+s+1)) ₛ₊₁Yℓm.
# For a spin-0 field the spin-1 synthesis of √(ℓ(ℓ+1))·f̂_lm reproduces ðf = -(∂_θ + i/sinθ ∂_φ)f
# (verified against the analytic gradient to ~1e-12), giving J(ψ,ζ) = (1/a²) Im{conj(ðψ)·ðζ}.
#
# Coefficient recovery from scattered points is a least-squares (CG) solve — well-conditioned only for
# equidistributed points (spherical-Fibonacci reaches machine-precision coefficient accuracy; jittered
# latitude bands do NOT — they recover the field but not the coefficients). The quadratic Jacobian is
# dealiased by solving it at degree 2·lmax, which needs M ≥ (2lmax+1)² points.
# ---------------------------------------------------------------------------

"""
    calculate_energy_transfer(method::SphericalTransferMethod, vorticity::AbstractVector,
                              coords::Tuple{<:AbstractVector,<:AbstractVector};
                              lmax, dealias=true, tol=1e-10, rtol=1e-10, maxiter=4000, kwargs...)

Spherical spectral energy/enstrophy transfer `T_E(l)`, `T_Z(l)` (and fluxes) for 2D non-divergent
flow on the sphere, from the **vorticity field** `ζ` sampled at `M` **scattered** points
`coords = (θ, φ)` (colatitudes `θ ∈ [0,π]`, longitudes `φ ∈ [0,2π)`, each length `M`). Returns a
[`SphericalTransferResult`](@ref) over degrees `l = 0…lmax`.

Coefficients are recovered by NUFSHT's CG least-squares solve, which is well-conditioned only for
**equidistributed** points — use spherical-Fibonacci or a spherical `t`-design, not clustered/jittered
points (those recover the field but not the coefficients the transfer needs). The Jacobian is
dealiased by solving it at degree `2·lmax`, so `M ≥ (2·lmax+1)²` is required (more is better).

Keyword `dealias=false` skips the 2·lmax dealiasing (aliased). `tol` is the FINUFFT tolerance;
`rtol`/`maxiter` control the CG solve. Requires `using NUFSHT`.
"""
function FIT.calculate_energy_transfer(
    method::SphericalTransferMethod,
    vorticity::AbstractVector{<:Real},
    coords::Tuple{<:AbstractVector, <:AbstractVector};
    lmax::Integer,
    dealias::Bool = true,
    tol::Real = 1e-10,
    rtol::Real = 1e-10,
    maxiter::Integer = 4000,
    kwargs...,
)
    θ, φ = coords
    M = length(vorticity)
    (length(θ) == M && length(φ) == M) ||
        throw(ArgumentError("vorticity and both coordinate vectors must have equal length; got $((M, length(θ), length(φ)))."))
    lmax ≥ 1 || throw(ArgumentError("lmax must be ≥ 1; got $lmax."))
    lwork = dealias ? 2 * lmax : lmax
    M ≥ (lwork + 1)^2 || throw(ArgumentError(
        "need M ≥ (2·lmax+1)² = $((lwork+1)^2) scattered points for the dealiased degree-$(lwork) solve; got M=$M. " *
        "Use equidistributed (e.g. spherical-Fibonacci) points for well-conditioned coefficient recovery."))

    # Compute type follows the input (NUFSHT's plans are parametric via `T`); Complex{FT} coefficients.
    FT = float(eltype(vorticity))
    CT = Complex{FT}
    a = FT(method.radius)
    ζdata = CT.(vorticity)

    # Analyse ζ → complex spin-0 coefficients (spin_coeff_index layout).
    plan0 = NUFSHT.make_spin_plan(θ, φ, lmax, 0; tol = tol, T = FT)
    ζ_lm = zeros(CT, lmax + 1, 2lmax + 1)
    NUFSHT.nusht_solve_spin!(ζ_lm, ζdata, plan0; rtol = rtol, maxiter = maxiter)

    # ψ = ∇⁻²ζ (ψ̂_lm = -a²/(l(l+1)) ζ̂_lm) and the eth ladder → spin-1 gradient coefficients.
    ψ_lm = zeros(CT, lmax + 1, 2lmax + 1)
    ðψ = zeros(CT, lmax + 1, 2lmax + 1)
    ðζ = zeros(CT, lmax + 1, 2lmax + 1)
    @inbounds for ℓ in 1:lmax, m in -ℓ:ℓ
        i = NUFSHT.spin_coeff_index(ℓ, m, lmax)
        ψ_lm[i] = -a^2 / (ℓ * (ℓ + 1)) * ζ_lm[i]
        c = sqrt(FT(ℓ * (ℓ + 1)))
        ðψ[i] = c * ψ_lm[i]
        ðζ[i] = c * ζ_lm[i]
    end

    # Synthesise the gradient fields ðψ, ðζ at the scattered points (spin-1).
    plan1 = NUFSHT.make_spin_plan(θ, φ, lmax, 1; tol = tol, T = FT)
    Gψ = zeros(CT, M); NUFSHT.nusht_type2_spin!(Gψ, ðψ, plan1)
    Gζ = zeros(CT, M); NUFSHT.nusht_type2_spin!(Gζ, ðζ, plan1)
    J = @. imag(conj(Gψ) * Gζ) / a^2                             # A = J(ψ,ζ) at the points

    # Analyse A at degree lwork (dealiased), then keep l ≤ lmax.
    plan0w = NUFSHT.make_spin_plan(θ, φ, lwork, 0; tol = tol, T = FT)
    A_lw = zeros(CT, lwork + 1, 2lwork + 1)
    NUFSHT.nusht_solve_spin!(A_lw, CT.(J), plan0w; rtol = rtol, maxiter = maxiter)

    # Flatten to per-mode arrays for the shared degree-spectrum reduction.
    nmode = (lmax + 1)^2
    degs = Vector{Int}(undef, nmode)
    ψv = Vector{CT}(undef, nmode)
    ζv = Vector{CT}(undef, nmode)
    Av = Vector{CT}(undef, nmode)
    k = 0
    @inbounds for ℓ in 0:lmax, m in -ℓ:ℓ
        k += 1
        i = NUFSHT.spin_coeff_index(ℓ, m, lmax)
        iw = NUFSHT.spin_coeff_index(ℓ, m, lwork)
        degs[k] = ℓ
        ψv[k] = ψ_lm[i]
        ζv[k] = ζ_lm[i]
        Av[k] = A_lw[iw]
    end
    return spherical_transfer_reduce(degs, ψv, ζv, Av, lmax)
end

end # module FlowInvariantTransferNUFSHTExt
