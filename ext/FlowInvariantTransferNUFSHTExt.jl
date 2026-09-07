module FlowInvariantTransferNUFSHTExt

using NUFSHT: NUFSHT
using FlowInvariantTransfer: FlowInvariantTransfer as FIT
using ComputationalBackends: ComputationalBackends

# ---------------------------------------------------------------------------
# Spherical spectral energy/enstrophy transfer at SCATTERED points on the sphere, via NUFSHT
# (non-uniform spherical-harmonic transforms). Same 2D-barotropic formulation as the FSH regular-grid
# path (core reduction: FlowInvariantTransfer.Spherical),
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
function FIT.Spherical.ScatteredSphericalTransferWorkspace(
    coords::Tuple{<:AbstractVector, <:AbstractVector},
    lmax::Integer;
    radius::Real = 1.0,
    dealias::Bool = true,
    tol::Real = 1e-10,
    rtol::Real = 1e-10,
    maxiter::Integer = 4000,
    T::Type = Float64,
    quadrature_weights::Union{Nothing, AbstractVector} = nothing,
    execution::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.SerialBackend(),
)
    θ, φ = coords
    M = length(θ)
    length(φ) == M || throw(ArgumentError("θ and φ must have equal length; got $((length(θ), length(φ)))."))
    lmax ≥ 1 || throw(ArgumentError("lmax must be ≥ 1; got $lmax."))
    lwork = dealias ? 2 * lmax : lmax
    # The point count below is what the least-squares fit needs to determine `(lwork+1)²` coefficients.
    # A quadrature determines them in one adjoint transform, at whatever count the rule is exact for.
    if quadrature_weights === nothing
        M ≥ (lwork + 1)^2 || throw(ArgumentError(
            "need M ≥ (2·lmax+1)² = $((lwork+1)^2) scattered points for the dealiased degree-$(lwork) solve; got M=$M. " *
            "Use equidistributed (e.g. spherical-Fibonacci) points for well-conditioned coefficient recovery."))
    else
        length(quadrature_weights) == M || throw(ArgumentError(
            "quadrature_weights has length $(length(quadrature_weights)) for $M nodes."))
    end
    FT = float(T)
    CT = Complex{FT}

    # The three NUFSHT spin plans (points preset) — the dominant, reusable cost. `nthreads=1` (Serial,
    # default) keeps the FINUFFT-backed transforms single-threaded: FINUFFT shares libfftw3 with FFTW.jl,
    # so a multithreaded plan spawns Julia Tasks per exec (allocating), and single-threaded is 0-alloc +
    # oversubscription-free when the outer batch axis is parallelised. ComputationalBackends.ThreadedBackend threads a lone call.
    nthr = execution isa ComputationalBackends.ThreadedBackend ? Threads.nthreads() : 1
    plan0  = NUFSHT.make_spin_plan(CT, θ, φ,lmax,  0; tol = tol, nthreads = nthr)
    plan1  = NUFSHT.make_spin_plan(CT, θ, φ,lmax,  1; tol = tol, nthreads = nthr)
    plan0w = NUFSHT.make_spin_plan(CT, θ, φ,lwork, 0; tol = tol, nthreads = nthr)

    # Buffers follow the coordinate array type (`similar(θ, …)`): device-array coordinates θ, φ make
    # NUFSHT build device (cuFINUFFT) plans, and these matching device buffers keep the whole transform
    # device-resident. Host coordinates → host buffers → CPU FINUFFT, exactly as before.
    _z(dims...) = fill!(similar(θ, CT, dims...), zero(CT))
    ζ_lm = _z(lmax + 1, 2lmax + 1)
    ψ_lm = _z(lmax + 1, 2lmax + 1)
    ðψ   = _z(lmax + 1, 2lmax + 1)
    ðζ   = _z(lmax + 1, 2lmax + 1)
    A_lw = _z(lwork + 1, 2lwork + 1)
    Gψ = _z(M); Gζ = _z(M); ζdata = _z(M); Jc = _z(M)
    # Degree per matrix row (ℓ at row ℓ+1), on-device, for the row-broadcast coefficient-space ops.
    degcol = reshape(similar(θ, FT, lmax + 1), lmax + 1, 1); copyto!(degcol, FT.(0:lmax))
    Pr   = similar(θ, FT, lmax + 1, 2lmax + 1)   # real product scratch (row-sum reduce)
    Tcol = similar(θ, FT, lmax + 1, 1)           # real per-degree column-sum scratch
    result = FIT.Types.SphericalTransferResult(
        collect(FT, 0:lmax), zeros(FT, lmax + 1), zeros(FT, lmax + 1),
        zeros(FT, lmax + 1), zeros(FT, lmax + 1))

    # Weights follow the coordinate array type, so a device point set keeps the whole analysis on device.
    qw = quadrature_weights === nothing ? nothing : copyto!(similar(θ, FT, M), quadrature_weights)

    return FIT.Spherical.ScatteredSphericalTransferWorkspace(
        plan0, plan1, plan0w, ζ_lm, ψ_lm, ðψ, ðζ, A_lw, Gψ, Gζ, ζdata, Jc,
        degcol, Pr, Tcol, qw, result, FT(radius), Int(lmax), Int(lwork), FT(rtol), Int(maxiter))
end

# `nusht_solve_spin!` returns `(C, iterations, residual, converged)`. The least-squares fit at scattered
# points is the one step whose accuracy is not set by a tolerance the caller can see afterwards: the
# threaded NUFFT spreading underneath it is not bitwise reproducible, and an unconverged solve
# amplifies that by roughly `cond(A)²`, so two identical calls can disagree far above `rtol`. Reporting
# the residual is what separates "this answer is at rtol" from "this answer is noise".
"""
    _analyze!(C, f, plan, qw, what; rtol, maxiter) -> C

Spin-weighted coefficients of `f` at the plan's nodes.

With per-node quadrature weights `qw` summing to `4π`, the coefficients are the projection
`Σⱼ wⱼ fⱼ conj(ₛYℓm(xⱼ))`, which `nusht_type1_spin!` evaluates once the weights are folded into the
field — exact for a rule that integrates the degree-`2·lwork` integrand, and one transform. `f` holds
this call's input only, so the weighting scales it in place.

With `qw === nothing` the nodes carry no such rule and the coefficients come from the least-squares
fit, whose convergence is reported.
"""
_analyze!(C, f, plan, ::Nothing, what::String; rtol, maxiter) =
    _solve_checked!(C, f, plan, what; rtol = rtol, maxiter = maxiter)

function _analyze!(C, f, plan, qw::AbstractVector, what::String; rtol, maxiter)
    f .*= qw
    NUFSHT.nusht_type1_spin!(C, f, plan)
    return C
end

function _solve_checked!(C, f, plan, what::String; rtol, maxiter)
    _, iters, residual, converged = NUFSHT.nusht_solve_spin!(C, f, plan; rtol = rtol, maxiter = maxiter)
    converged || @warn(
        "scattered spherical least-squares solve did not reach `rtol` — the result is accurate only to " *
        "the residual below, and repeat calls need not agree more closely than that. Raise `maxiter` " *
        "or `rtol`, or add points.",
        component = what, iterations = iters, residual = residual, rtol = rtol, maxiter = maxiter,
        maxlog = 1)
    return C
end

function FIT.Spherical.calculate_spherical_transfer!(
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
    _analyze!(ws.ζ_lm, ws.ζdata, ws.plan0, ws.qw, "vorticity"; rtol = ws.rtol, maxiter = ws.maxiter)

    # ψ = ∇⁻²ζ and the eth ladder → spin-1 gradient coefficients, as row-broadcasts over the
    # (degree = row, m = column) coefficient matrices — device-generic, no scalar indexing. ℓ(ℓ+1) is a
    # per-row scalar from `ws.degcol`; the ℓ=0 row and the |m|>ℓ corners are 0 in ζ_lm, so they stay 0.
    dd     = ws.degcol
    ll1    = @. dd * (dd + 1)
    invll1 = @. ifelse(ll1 > 0, inv(ll1), zero(FT))
    cfac   = @. sqrt(ll1)
    @. ws.ψ_lm = -a^2 * invll1 * ws.ζ_lm
    @. ws.ðψ   = cfac * ws.ψ_lm
    @. ws.ðζ   = cfac * ws.ζ_lm

    # Synthesise ðψ, ðζ at the points; A = J(ψ,ζ) into the complex solve buffer.
    NUFSHT.nusht_type2_spin!(ws.Gψ, ws.ðψ, ws.plan1)
    NUFSHT.nusht_type2_spin!(ws.Gζ, ws.ðζ, ws.plan1)
    @. ws.Jc = imag(conj(ws.Gψ) * ws.Gζ) / a^2

    # Analyse A at degree lwork (dealiased).
    fill!(ws.A_lw, zero(CT))
    _analyze!(ws.A_lw, ws.Jc, ws.plan0w, ws.qw, "advection"; rtol = ws.rtol, maxiter = ws.maxiter)

    # Per-degree transfer = sum over m (matrix columns). A_lw (degree lwork) aligns to the lmax layout by a
    # contiguous column slice (m offset lwork−lmax); |m|>ℓ corners are 0 (ζ/ψ/A = 0), so the full row-sum
    # equals the sum over valid m. T_E(ℓ) = −Σ_m Re{ψ* A}, T_Z(ℓ) = +Σ_m Re{ζ* A}; then Π(L) = −Σ_{l≤L}T(l).
    A_al = @view ws.A_lw[1:lmax + 1, (lwork - lmax + 1):(lwork + lmax + 1)]
    @. ws.Pr = real(conj(ws.ψ_lm) * A_al)          # T_E(ℓ) = -Σ_m Re{ψ* A}  (fused, into preallocated Pr)
    sum!(ws.Tcol, ws.Pr)
    copyto!(ws.result.energy_transfer, vec(ws.Tcol));  ws.result.energy_transfer .*= -1
    @. ws.Pr = real(conj(ws.ζ_lm) * A_al)          # T_Z(ℓ) = +Σ_m Re{ζ* A}
    sum!(ws.Tcol, ws.Pr)
    copyto!(ws.result.enstrophy_transfer, vec(ws.Tcol))
    FIT.Spherical._neg_cumsum!(ws.result.energy_flux,    ws.result.energy_transfer)
    FIT.Spherical._neg_cumsum!(ws.result.enstrophy_flux, ws.result.enstrophy_transfer)
    return ws.result
end

function FIT.calculate_energy_transfer(
    method::FIT.Types.SphericalTransferMethod,
    vorticity::AbstractVector{<:Real},
    coords::Tuple{<:AbstractVector, <:AbstractVector};
    lmax::Integer,
    dealias::Bool = true,
    tol::Real = 1e-10,
    rtol::Real = 1e-10,
    maxiter::Integer = 4000,
    spectral = nothing,
    quadrature_weights::Union{Nothing, AbstractVector} = nothing,
    execution::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.SerialBackend(),
    kwargs...,
)
    FIT.Spherical._validate_spherical_backends(spectral, execution, :scattered)
    M = length(vorticity)
    (length(coords[1]) == M && length(coords[2]) == M) || throw(ArgumentError(
        "vorticity and both coordinate vectors must have equal length; got $((M, length(coords[1]), length(coords[2])))."))
    ws = FIT.Spherical.ScatteredSphericalTransferWorkspace(
        coords, lmax; radius = float(method.radius), dealias = dealias,
        tol = tol, rtol = rtol, maxiter = maxiter, T = float(eltype(vorticity)),
        quadrature_weights = quadrature_weights, execution = execution)
    return FIT.Spherical.calculate_spherical_transfer!(ws, vorticity)
end

# ---------------------------------------------------------------------------
# DIVERGENT horizontal-KE spectral transfer at scattered points (full rotational + divergent flow).
# Formulation & verified conventions: FlowInvariantTransfer.Spherical (α=-i vorticity ladder, δ=+ladder
# divergence, ∇=-ð, k̂×u ↔ iU₊, skew-symmetric ½δu conservation term, single 1/a radius factor). Input is
# the horizontal velocity (u_θ, u_φ) at the scattered points; the transfer conserves total KE
# (Σ_l T ≈ 0) and reduces to the barotropic `SphericalTransferMethod` energy transfer when δ = 0.
# ---------------------------------------------------------------------------

"""
    ScatteredDivergentSphericalTransferWorkspace(coords, lmax; radius=1.0, dealias=true,
                                                 tol=1e-10, rtol=1e-10, maxiter=4000, T=Float64,
                                                 execution=ComputationalBackends.SerialBackend())

Reusable buffers + the five NUFSHT spin plans (points preset) for the scattered divergent KE transfer.
`coords = (θ, φ)` are the `M` colatitudes/longitudes. Needs `M ≥ (2·lmax+1)²` **equidistributed**
points (e.g. spherical-Fibonacci) for well-conditioned coefficient recovery. Requires `using NUFSHT`.
"""
function FIT.Spherical.ScatteredDivergentSphericalTransferWorkspace(
    coords::Tuple{<:AbstractVector, <:AbstractVector},
    lmax::Integer;
    radius::Real = 1.0,
    dealias::Bool = true,
    tol::Real = 1e-10,
    rtol::Real = 1e-10,
    maxiter::Integer = 4000,
    T::Type = Float64,
    quadrature_weights::Union{Nothing, AbstractVector} = nothing,
    execution::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.SerialBackend(),
)
    θ, φ = coords
    M = length(θ)
    length(φ) == M || throw(ArgumentError("θ and φ must have equal length; got $((length(θ), length(φ)))."))
    lmax ≥ 1 || throw(ArgumentError("lmax must be ≥ 1; got $lmax."))
    lwork = dealias ? 2 * lmax : lmax
    if quadrature_weights === nothing
        M ≥ (lwork + 1)^2 || throw(ArgumentError(
            "need M ≥ (2·lmax+1)² = $((lwork+1)^2) scattered points for the dealiased degree-$(lwork) solve; got M=$M. " *
            "Use equidistributed (e.g. spherical-Fibonacci) points for well-conditioned coefficient recovery."))
    else
        length(quadrature_weights) == M || throw(ArgumentError(
            "quadrature_weights has length $(length(quadrature_weights)) for $M nodes."))
    end
    FT = float(T)
    CT = Complex{FT}

    # Single-threaded FINUFFT plans by default (see the barotropic workspace above for the rationale);
    # ComputationalBackends.ThreadedBackend threads a lone call. Five plans: spin ±1 & spin-0 at lmax, spin-0 & spin+1 at lwork.
    nthr = execution isa ComputationalBackends.ThreadedBackend ? Threads.nthreads() : 1
    planp  = NUFSHT.make_spin_plan(CT, θ, φ,lmax,   1; tol = tol, nthreads = nthr)
    planm  = NUFSHT.make_spin_plan(CT, θ, φ,lmax,  -1; tol = tol, nthreads = nthr)
    plan0  = NUFSHT.make_spin_plan(CT, θ, φ,lmax,   0; tol = tol, nthreads = nthr)
    plan0w = NUFSHT.make_spin_plan(CT, θ, φ,lwork,  0; tol = tol, nthreads = nthr)
    planpw = NUFSHT.make_spin_plan(CT, θ, φ,lwork,  1; tol = tol, nthreads = nthr)

    # Buffers follow the coordinate array type (device-array coords → device buffers → cuFINUFFT).
    _z(dims...) = fill!(similar(θ, CT, dims...), zero(CT))
    ap = _z(lmax + 1, 2lmax + 1); am = _z(lmax + 1, 2lmax + 1)
    sym = _z(lmax + 1, 2lmax + 1); anti = _z(lmax + 1, 2lmax + 1)
    ζc = _z(lmax + 1, 2lmax + 1); δc = _z(lmax + 1, 2lmax + 1)
    Khat = _z(lwork + 1, 2lwork + 1); Adv_lm = _z(lwork + 1, 2lwork + 1)
    Up = _z(M); Um = _z(M); ζv = _z(M); δv = _z(M); Kv = _z(M); gradK = _z(M); Advv = _z(M)
    # Per-row ladders √(ℓ(ℓ+1)) (on-device, for the row-broadcast coefficient-space ops).
    ladl = reshape(similar(θ, FT, lmax + 1), lmax + 1, 1)
    copyto!(ladl, FT[sqrt(ℓ * (ℓ + 1)) for ℓ in 0:lmax])
    ladw = reshape(similar(θ, FT, lwork + 1), lwork + 1, 1)
    copyto!(ladw, FT[sqrt(ℓ * (ℓ + 1)) for ℓ in 0:lwork])
    Pr = similar(θ, FT, lmax + 1, 2lmax + 1)
    Tcol = similar(θ, FT, lmax + 1, 1)
    result = FIT.Types.DivergentSphericalTransferResult(
        collect(FT, 0:lmax), zeros(FT, lmax + 1), zeros(FT, lmax + 1), zeros(FT, lmax + 1),
        zeros(FT, lmax + 1), zeros(FT, lmax + 1), zeros(FT, lmax + 1))

    qw = quadrature_weights === nothing ? nothing : copyto!(similar(θ, FT, M), quadrature_weights)

    return FIT.Spherical.ScatteredDivergentSphericalTransferWorkspace(
        planp, planm, plan0, plan0w, planpw, ap, am, sym, anti, ζc, δc, Khat, Adv_lm,
        Up, Um, ζv, δv, Kv, gradK, Advv, ladl, ladw, Pr, Tcol, qw, result,
        FT(radius), Int(lmax), Int(lwork), FT(rtol), Int(maxiter))
end

function FIT.calculate_divergent_spherical_transfer!(
    ws::FIT.Spherical.ScatteredDivergentSphericalTransferWorkspace,
    u_θ::AbstractVector{<:Real},
    u_φ::AbstractVector{<:Real},
)
    M = length(ws.Up)
    (length(u_θ) == M && length(u_φ) == M) || throw(DimensionMismatch(
        "velocity component lengths $((length(u_θ), length(u_φ))) ≠ workspace points $M."))
    lmax = ws.lmax; lwork = ws.lwork; a = ws.radius
    CT = eltype(ws.ap)

    # Spin ±1 coefficients of U₊ = u_θ + i u_φ and U₋ = u_θ − i u_φ; rotational/divergent split.
    @. ws.Up = u_θ + im * u_φ
    @. ws.Um = u_θ - im * u_φ
    fill!(ws.ap, zero(CT)); _analyze!(ws.ap, ws.Up, ws.planp, ws.qw, "velocity spin+1"; rtol = ws.rtol, maxiter = ws.maxiter)
    fill!(ws.am, zero(CT)); _analyze!(ws.am, ws.Um, ws.planm, ws.qw, "velocity spin−1"; rtol = ws.rtol, maxiter = ws.maxiter)
    @. ws.sym  = (ws.ap + ws.am) / 2
    @. ws.anti = (ws.ap - ws.am) / 2

    # Unit-sphere vorticity/divergence via the eth ladder (Goldberg convention, ð ₛY=+√((ℓ−s)(ℓ+s+1))ₛ₊₁Y):
    # ζ_lm = +i√(ℓ(ℓ+1)) sym,  δ_lm = -√(ℓ(ℓ+1)) anti; synthesise both at the points (spin-0).
    @. ws.ζc =  im * ws.ladl * ws.sym
    @. ws.δc = -ws.ladl * ws.anti
    NUFSHT.nusht_type2_spin!(ws.ζv, ws.ζc, ws.plan0)
    NUFSHT.nusht_type2_spin!(ws.δv, ws.δc, ws.plan0)

    # K = ½|u|² (real, held complex); analyse at the dealiased degree lwork.
    @. ws.Kv = 0.5 * (u_θ^2 + u_φ^2)
    fill!(ws.Khat, zero(CT)); _analyze!(ws.Khat, ws.Kv, ws.plan0w, ws.qw, "kinetic energy"; rtol = ws.rtol, maxiter = ws.maxiter)

    # ∇K = ð K = +√(ℓ(ℓ+1)) synth_spin+1(K̂)  (reuse Khat for the ladder-scaled coefficients).
    @. ws.Khat = ws.ladw * ws.Khat
    NUFSHT.nusht_type2_spin!(ws.gradK, ws.Khat, ws.planpw)

    # Skew-symmetric energy-conserving advection A = ∇K + (iζ + ½δ) U₊; analyse (spin+1) at lwork.
    # U₊ is re-formed from the components here: `ws.Up` is an analysis input, and analysis against a
    # quadrature scales its input by the weights.
    @. ws.Advv = ws.gradK + (im * ws.ζv + 0.5 * ws.δv) * (u_θ + im * u_φ)
    fill!(ws.Adv_lm, zero(CT)); _analyze!(ws.Adv_lm, ws.Advv, ws.planpw, ws.qw, "advection"; rtol = ws.rtol, maxiter = ws.maxiter)

    # Per-degree channel reduction T_rot = Σ_m Re{sym* Â}, T_div = Σ_m Re{anti* Â} (single 1/a factor).
    # A_lm (degree lwork) aligns to the lmax layout by a centred column slice; |m|>ℓ corners are 0.
    Adv_al = @view ws.Adv_lm[1:lmax + 1, (lwork - lmax + 1):(lwork + lmax + 1)]
    @. ws.Pr = real(conj(ws.sym) * Adv_al)
    sum!(ws.Tcol, ws.Pr)
    copyto!(ws.result.rotational_transfer, vec(ws.Tcol)); ws.result.rotational_transfer ./= a
    @. ws.Pr = real(conj(ws.anti) * Adv_al)
    sum!(ws.Tcol, ws.Pr)
    copyto!(ws.result.divergent_transfer, vec(ws.Tcol)); ws.result.divergent_transfer ./= a
    return FIT.Spherical.divergent_transfer_finalize!(ws.result)
end

function FIT.calculate_energy_transfer(
    method::FIT.Types.DivergentSphericalTransferMethod,
    velocity::Tuple{<:AbstractVector, <:AbstractVector},
    coords::Tuple{<:AbstractVector, <:AbstractVector};
    lmax::Integer,
    dealias::Bool = true,
    tol::Real = 1e-10,
    rtol::Real = 1e-10,
    maxiter::Integer = 4000,
    spectral = nothing,
    quadrature_weights::Union{Nothing, AbstractVector} = nothing,
    execution::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.SerialBackend(),
    kwargs...,
)
    FIT.Spherical._validate_spherical_backends(spectral, execution, :scattered)
    u_θ, u_φ = velocity
    M = length(u_θ)
    (length(u_φ) == M && length(coords[1]) == M && length(coords[2]) == M) || throw(ArgumentError(
        "velocity components and both coordinate vectors must have equal length; got " *
        "$((M, length(u_φ), length(coords[1]), length(coords[2])))."))
    ws = FIT.Spherical.ScatteredDivergentSphericalTransferWorkspace(
        coords, lmax; radius = float(method.radius), dealias = dealias, tol = tol, rtol = rtol,
        maxiter = maxiter, T = float(eltype(u_θ)), quadrature_weights = quadrature_weights,
        execution = execution)
    return FIT.calculate_divergent_spherical_transfer!(ws, u_θ, u_φ)
end

end # module FlowInvariantTransferNUFSHTExt
