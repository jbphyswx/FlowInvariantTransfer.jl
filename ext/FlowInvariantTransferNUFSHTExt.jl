module FlowInvariantTransferNUFSHTExt

using NUFSHT: NUFSHT
using FlowInvariantTransfer: FlowInvariantTransfer as FIT
using FlowInvariantTransfer.Types: SphericalTransferMethod, SphericalTransferResult
using FlowInvariantTransfer.Spherical: spherical_transfer_reduce, spherical_transfer_reduce!

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
function FIT.ScatteredSphericalTransferWorkspace(
    coords::Tuple{<:AbstractVector, <:AbstractVector},
    lmax::Integer;
    radius::Real = 1.0,
    dealias::Bool = true,
    tol::Real = 1e-10,
    rtol::Real = 1e-10,
    maxiter::Integer = 4000,
    T::Type = Float64,
)
    θ, φ = coords
    M = length(θ)
    length(φ) == M || throw(ArgumentError("θ and φ must have equal length; got $((length(θ), length(φ)))."))
    lmax ≥ 1 || throw(ArgumentError("lmax must be ≥ 1; got $lmax."))
    lwork = dealias ? 2 * lmax : lmax
    M ≥ (lwork + 1)^2 || throw(ArgumentError(
        "need M ≥ (2·lmax+1)² = $((lwork+1)^2) scattered points for the dealiased degree-$(lwork) solve; got M=$M. " *
        "Use equidistributed (e.g. spherical-Fibonacci) points for well-conditioned coefficient recovery."))
    FT = float(T)
    CT = Complex{FT}

    # The three NUFSHT spin plans (points preset) — the dominant, reusable cost.
    plan0  = NUFSHT.make_spin_plan(θ, φ, lmax,  0; tol = tol, T = FT)
    plan1  = NUFSHT.make_spin_plan(θ, φ, lmax,  1; tol = tol, T = FT)
    plan0w = NUFSHT.make_spin_plan(θ, φ, lwork, 0; tol = tol, T = FT)

    ζ_lm = zeros(CT, lmax + 1, 2lmax + 1)
    ψ_lm = zeros(CT, lmax + 1, 2lmax + 1)
    ðψ   = zeros(CT, lmax + 1, 2lmax + 1)
    ðζ   = zeros(CT, lmax + 1, 2lmax + 1)
    A_lw = zeros(CT, lwork + 1, 2lwork + 1)
    Gψ = zeros(CT, M); Gζ = zeros(CT, M); ζdata = zeros(CT, M); Jc = zeros(CT, M)
    nmode = (lmax + 1)^2
    degs = Vector{Int}(undef, nmode)
    ψv = Vector{CT}(undef, nmode); ζv = Vector{CT}(undef, nmode); Av = Vector{CT}(undef, nmode)
    result = SphericalTransferResult(
        collect(FT, 0:lmax), zeros(FT, lmax + 1), zeros(FT, lmax + 1),
        zeros(FT, lmax + 1), zeros(FT, lmax + 1))

    return FIT.Spherical.ScatteredSphericalTransferWorkspace(
        plan0, plan1, plan0w, ζ_lm, ψ_lm, ðψ, ðζ, A_lw, Gψ, Gζ, ζdata, Jc,
        degs, ψv, ζv, Av, result, FT(radius), Int(lmax), Int(lwork), FT(rtol), Int(maxiter))
end

function FIT.calculate_spherical_transfer!(
    ws::FIT.Spherical.ScatteredSphericalTransferWorkspace,
    vorticity::AbstractVector{<:Real},
)
    M = length(ws.Gψ)
    length(vorticity) == M || throw(DimensionMismatch(
        "vorticity length $(length(vorticity)) ≠ workspace points $M."))
    lmax = ws.lmax; lwork = ws.lwork
    a = ws.radius
    CT = eltype(ws.ζ_lm); FT = real(CT)

    # Analyse ζ → spin-0 coefficients (CG solve reuses ws.ζ_lm, points preset in plan0).
    ws.ζdata .= vorticity
    fill!(ws.ζ_lm, zero(CT))
    NUFSHT.nusht_solve_spin!(ws.ζ_lm, ws.ζdata, ws.plan0; rtol = ws.rtol, maxiter = ws.maxiter)

    # ψ = ∇⁻²ζ and the eth ladder → spin-1 gradient coefficients (reused buffers).
    fill!(ws.ψ_lm, zero(CT)); fill!(ws.ðψ, zero(CT)); fill!(ws.ðζ, zero(CT))
    @inbounds for ℓ in 1:lmax, m in -ℓ:ℓ
        i = NUFSHT.spin_coeff_index(ℓ, m, lmax)
        ws.ψ_lm[i] = -a^2 / (ℓ * (ℓ + 1)) * ws.ζ_lm[i]
        c = sqrt(FT(ℓ * (ℓ + 1)))
        ws.ðψ[i] = c * ws.ψ_lm[i]
        ws.ðζ[i] = c * ws.ζ_lm[i]
    end

    # Synthesise ðψ, ðζ at the points; A = J(ψ,ζ) into the complex solve buffer.
    NUFSHT.nusht_type2_spin!(ws.Gψ, ws.ðψ, ws.plan1)
    NUFSHT.nusht_type2_spin!(ws.Gζ, ws.ðζ, ws.plan1)
    @. ws.Jc = imag(conj(ws.Gψ) * ws.Gζ) / a^2

    # Analyse A at degree lwork (dealiased); reduce over l ≤ lmax into the reused result.
    fill!(ws.A_lw, zero(CT))
    NUFSHT.nusht_solve_spin!(ws.A_lw, ws.Jc, ws.plan0w; rtol = ws.rtol, maxiter = ws.maxiter)

    k = 0
    @inbounds for ℓ in 0:lmax, m in -ℓ:ℓ
        k += 1
        i = NUFSHT.spin_coeff_index(ℓ, m, lmax)
        iw = NUFSHT.spin_coeff_index(ℓ, m, lwork)
        ws.degs[k] = ℓ
        ws.ψv[k] = ws.ψ_lm[i]
        ws.ζv[k] = ws.ζ_lm[i]
        ws.Av[k] = ws.A_lw[iw]
    end
    return spherical_transfer_reduce!(ws.result, ws.degs, ws.ψv, ws.ζv, ws.Av)
end

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
    M = length(vorticity)
    (length(coords[1]) == M && length(coords[2]) == M) || throw(ArgumentError(
        "vorticity and both coordinate vectors must have equal length; got $((M, length(coords[1]), length(coords[2])))."))
    ws = FIT.ScatteredSphericalTransferWorkspace(
        coords, lmax; radius = float(method.radius), dealias = dealias,
        tol = tol, rtol = rtol, maxiter = maxiter, T = float(eltype(vorticity)))
    return FIT.calculate_spherical_transfer!(ws, vorticity)
end

end # module FlowInvariantTransferNUFSHTExt
