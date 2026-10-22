module FlowInvariantTransferNUFSHTExt

using NUFSHT: NUFSHT
using FlowTransformBindings: FlowTransformBindings as FTB
using FlowInvariantTransfer: FlowInvariantTransfer as FIT
using ComputationalBackends: ComputationalBackends
using SpectralBackends: SpectralBackends

# ---------------------------------------------------------------------------
# Spherical spectral energy/enstrophy transfer at scattered points on the sphere, via NUFSHT
# (non-uniform spherical-harmonic transforms): the 2D-barotropic formulation of the FSH regular-grid
# path (core reduction: FlowInvariantTransfer.Spherical) with NUFFT-backed scattered transforms.
#
# NUFSHT's spin-weighted harmonics are ₛYℓm = √((2ℓ+1)/4π) d^ℓ_{m,-s}(θ) e^{imφ}, so the eth ladder is
# ð(ₛYℓm) = √((ℓ-s)(ℓ+s+1)) ₛ₊₁Yℓm. For a spin-0 field the spin-1 synthesis of √(ℓ(ℓ+1))·f̂_lm is
# ðf = -(∂_θ + i/sinθ ∂_φ)f, giving J(ψ,ζ) = (1/a²) Im{conj(ðψ)·ðζ}.
#
# Coefficient recovery from scattered points is a least-squares (LSMR) fit, which determines the
# coefficients on equidistributed points (spherical Fibonacci); clustered or jittered points recover
# the field and leave the coefficients undetermined. The quadratic Jacobian is dealiased by solving it
# at degree 2·lmax, which needs M ≥ (2lmax+1)² points.
# ---------------------------------------------------------------------------

# A fit's relative residual resolves no finer than the NUFFT accuracy `tol`, and NUFSHT floors it at
# `eps(FT)`, so the default tolerance sits above both.
_default_rtol(rtol::Real, ::Real, ::Type{FT}) where {FT} = FT(rtol)
_default_rtol(::Nothing, tol::Real, ::Type{FT}) where {FT} = max(FT(tol), 100 * eps(FT))

function FIT.Spherical.ScatteredSphericalTransferWorkspace(
    coords::Tuple{<:AbstractVector, <:AbstractVector},
    lmax::Integer;
    radius::Real = 1.0,
    dealias::Bool = true,
    tol::Real = 1e-10,
    rtol::Union{Nothing, Real} = nothing,
    maxiter::Integer = 4000,
    T::Type = Float64,
    quadrature_weights::Union{Nothing, AbstractVector} = nothing,
    nufft::SpectralBackends.AbstractSpectralBackend = SpectralBackends.AutoSpectralBackend(),
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

    # The three NUFSHT spin plans (points preset) — the dominant, reusable cost. `execution` sets where
    # they run: one thread under `SerialBackend()` (the default), which leaves the cores to an outer
    # batch axis, every thread under `ThreadedBackend()`, and the device of `GPUBackend(b)`, which holds
    # their nodes. Device-array coordinates give device plans under any of them.
    exec = FIT.Types.resolve_execution(execution)
    plan0  = NUFSHT.make_spin_plan(CT, θ, φ,lmax,  0; tol = tol, execution = exec, nufft = nufft)
    plan1  = NUFSHT.make_spin_plan(CT, θ, φ,lmax,  1; tol = tol, execution = exec, nufft = nufft)
    plan0w = NUFSHT.make_spin_plan(CT, θ, φ,lwork, 0; tol = tol, execution = exec, nufft = nufft)

    # Every buffer lives in the plans' memory, which `allocate_coefficients` allocates in.
    mem = NUFSHT.allocate_coefficients(plan0)
    _z(dims...) = fill!(similar(mem, CT, dims...), zero(CT))
    ζ_lm = _z(lmax + 1, 2lmax + 1)
    ψ_lm = _z(lmax + 1, 2lmax + 1)
    ðψ   = _z(lmax + 1, 2lmax + 1)
    ðζ   = _z(lmax + 1, 2lmax + 1)
    A_lw = _z(lwork + 1, 2lwork + 1)
    Gψ = _z(M); Gζ = _z(M); ζdata = _z(M); Jc = _z(M)
    fr = similar(mem, FT, M)
    # Per-degree factors (ℓ at row ℓ+1) for the row-broadcast coefficient-space ops.
    ladl = reshape(similar(mem, FT, lmax + 1), lmax + 1, 1)
    copyto!(ladl, FT[sqrt(ℓ * (ℓ + 1)) for ℓ in 0:lmax])
    invll1 = reshape(similar(mem, FT, lmax + 1), lmax + 1, 1)
    copyto!(invll1, FT[ℓ == 0 ? 0 : 1 / (ℓ * (ℓ + 1)) for ℓ in 0:lmax])
    Pr   = similar(mem, FT, lmax + 1, 2lmax + 1)   # real product scratch (row-sum reduce)
    Tcol = similar(mem, FT, lmax + 1, 1)           # real per-degree column-sum scratch
    result = FIT.Types.SphericalTransferResult(
        collect(FT, 0:lmax), zeros(FT, lmax + 1), zeros(FT, lmax + 1),
        zeros(FT, lmax + 1), zeros(FT, lmax + 1), true, 0, zero(FT))

    qw = quadrature_weights === nothing ? nothing : copyto!(similar(mem, FT, M), quadrature_weights)
    lsmr0  = _fit_workspace(qw, plan0)
    lsmr0w = _fit_workspace(qw, plan0w)

    return FIT.Spherical.ScatteredSphericalTransferWorkspace(
        plan0, plan1, plan0w, lsmr0, lsmr0w, ζ_lm, ψ_lm, ðψ, ðζ, A_lw, Gψ, Gζ, ζdata, Jc, fr,
        ladl, invll1, Pr, Tcol, qw, result, FT(radius), Int(lmax), Int(lwork),
        _default_rtol(rtol, tol, FT), Int(maxiter))
end

function FIT.close!(ws::FIT.Spherical.ScatteredSphericalTransferWorkspace)
    foreach(NUFSHT.close!, (ws.plan0, ws.plan1, ws.plan0w))
    return ws
end

# The LSMR workspace of a fit; analysis against quadrature weights runs no fit.
_fit_workspace(::Nothing, plan) = FTB.LSMRWorkspace(plan)
_fit_workspace(::AbstractVector, _) = nothing

# One plan over the active nodes fits the coefficients, one over every node evaluates them; the defaults
# are the scattered workspaces'.
function FIT.Spherical.active_fit(
    values::AbstractVector,
    s::Integer,
    coords::Tuple{<:AbstractVector, <:AbstractVector},
    active::AbstractVector{<:Integer},
    lmax::Integer;
    tol::Real = 1e-10,
    rtol::Union{Nothing, Real} = nothing,
    maxiter::Integer = 4000,
    nufft::SpectralBackends.AbstractSpectralBackend = SpectralBackends.AutoSpectralBackend(),
    execution::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.SerialBackend(),
)
    θ, φ = coords
    M = length(θ)
    Ma = length(active)
    Ma ≥ (lmax + 1)^2 || throw(ArgumentError(
        "the fit over the active cells determines (lmax+1)² = $((lmax + 1)^2) coefficients from $Ma active " *
        "nodes, fewer than it needs; analyse at a lower degree, or pass `inactive = Types.ZeroFill()`."))
    FT = float(real(eltype(values)))
    CT = Complex{FT}
    exec = FIT.Types.resolve_execution(execution)
    pa = NUFSHT.make_spin_plan(CT, θ[active], φ[active], lmax, s; tol = tol, execution = exec, nufft = nufft)
    pf = nothing
    try
        pf = NUFSHT.make_spin_plan(CT, θ, φ, lmax, s; tol = tol, execution = exec, nufft = nufft)
        mem = NUFSHT.allocate_coefficients(pa)
        C = fill!(similar(mem, CT, lmax + 1, 2lmax + 1), zero(CT))
        f = copyto!(similar(mem, CT, Ma), CT.(values[active]))
        r = NUFSHT.nusht_analyze!(C, f, pa; ws = FTB.LSMRWorkspace(pa), rtol = _default_rtol(rtol, tol, FT),
                                  maxiter = maxiter)
        out = similar(mem, CT, M)
        NUFSHT.nusht_type2_spin!(out, C, pf)
        return Array(out), (; converged = r.converged, iterations = r.iterations, residual = FT(r.residual))
    finally
        NUFSHT.close!(pa)
        pf === nothing || NUFSHT.close!(pf)
    end
end

# The spin-weighted coefficients of `f` at the plan's nodes, into `C`, by `NUFSHT.nusht_analyze!`: the
# quadrature `Σⱼ wⱼ fⱼ conj(ₛYℓm(xⱼ))` with the weights `qw`, else the least-squares fit in the LSMR
# workspace `lsmr`. Returns the outcome as `Spherical.merge_fits` reads it, the residual in `rtol`'s
# float type.
function _analyze!(C, f, plan, qw, lsmr; rtol, maxiter)
    r = NUFSHT.nusht_analyze!(C, f, plan; weights = qw, ws = lsmr, rtol = rtol, maxiter = maxiter)
    return (; converged = r.converged, iterations = r.iterations, residual = oftype(rtol, r.residual))
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
    CT = eltype(ws.ζ_lm)

    # Analyse ζ → spin-0 coefficients (points preset in plan0).
    copyto!(ws.fr, vorticity)
    ws.ζdata .= ws.fr
    fill!(ws.ζ_lm, zero(CT))
    fit = _analyze!(ws.ζ_lm, ws.ζdata, ws.plan0, ws.qw, ws.lsmr0; rtol = ws.rtol, maxiter = ws.maxiter)

    # ψ = ∇⁻²ζ and the eth ladder → spin-1 gradient coefficients, as row-broadcasts over the
    # (degree = row, m = column) coefficient matrices — device-generic, no scalar indexing. The ℓ=0 row
    # and the |m|>ℓ corners are 0 in ζ_lm, so they stay 0.
    @. ws.ψ_lm = -a^2 * ws.invll1 * ws.ζ_lm
    @. ws.ðψ   = ws.ladl * ws.ψ_lm
    @. ws.ðζ   = ws.ladl * ws.ζ_lm

    # Synthesise ðψ, ðζ at the points; A = J(ψ,ζ) into the complex solve buffer.
    NUFSHT.nusht_type2_spin!(ws.Gψ, ws.ðψ, ws.plan1)
    NUFSHT.nusht_type2_spin!(ws.Gζ, ws.ðζ, ws.plan1)
    @. ws.Jc = imag(conj(ws.Gψ) * ws.Gζ) / a^2

    # Analyse A at degree lwork (dealiased).
    fill!(ws.A_lw, zero(CT))
    fit = FIT.Spherical.merge_fits(fit, _analyze!(ws.A_lw, ws.Jc, ws.plan0w, ws.qw, ws.lsmr0w;
                                                  rtol = ws.rtol, maxiter = ws.maxiter))

    # Per-degree transfer = sum over m (matrix columns). A_lw (degree lwork) aligns to the lmax layout by a
    # contiguous column slice (m offset lwork−lmax); |m|>ℓ corners are 0 (ζ/ψ/A = 0), so the full row-sum
    # equals the sum over valid m. T_E(ℓ) = −Σ_m Re{ψ* A}, T_Z(ℓ) = +Σ_m Re{ζ* A}; then Π(L) = Σ_{l≤L}T(l).
    A_al = @view ws.A_lw[1:lmax + 1, (lwork - lmax + 1):(lwork + lmax + 1)]
    @. ws.Pr = real(conj(ws.ψ_lm) * A_al)          # T_E(ℓ) = -Σ_m Re{ψ* A}  (fused, into preallocated Pr)
    sum!(ws.Tcol, ws.Pr)
    copyto!(ws.result.energy_transfer, ws.Tcol);  ws.result.energy_transfer .*= -1
    @. ws.Pr = real(conj(ws.ζ_lm) * A_al)          # T_Z(ℓ) = +Σ_m Re{ζ* A}
    sum!(ws.Tcol, ws.Pr)
    copyto!(ws.result.enstrophy_transfer, ws.Tcol)
    cumsum!(ws.result.energy_flux,    ws.result.energy_transfer)
    cumsum!(ws.result.enstrophy_flux, ws.result.enstrophy_transfer)
    return FIT.Spherical.with_fit(ws.result, fit)
end

"""
    calculate_energy_transfer(method::SphericalTransferMethod, vorticity::AbstractVector,
                              coords::Tuple{<:AbstractVector,<:AbstractVector};
                              lmax, dealias=true, tol=1e-10, rtol=max(tol, 100eps(T)), maxiter=4000,
                              transform=nothing, quadrature_weights=nothing,
                              nufft=AutoSpectralBackend(), execution=SerialBackend())

Spherical spectral energy/enstrophy transfer `T_E(l)`, `T_Z(l)` (and fluxes) for 2D non-divergent
flow on the sphere, from the **vorticity field** `ζ` sampled at `M` **scattered** points
`coords = (θ, φ)` (colatitudes `θ ∈ [0,π]`, longitudes `φ ∈ [0,2π)`, each length `M`). Returns a
[`SphericalTransferResult`](@ref) over degrees `l = 0…lmax`.

Coefficients are recovered by NUFSHT's least-squares solve, which determines them on
**equidistributed** points (spherical Fibonacci, a spherical `t`-design); clustered or jittered points
reproduce the field while leaving the coefficients the transfer reads undetermined. The Jacobian is
dealiased by solving it at degree `2·lmax`, so `M ≥ (2·lmax+1)²` is required (more is better).
Per-node `quadrature_weights` (`Σw = 4π`), for a rule exact at that degree, replace the solve by a
projection.

Keyword `dealias=false` skips the 2·lmax dealiasing (aliased). `tol` is the NUFFT tolerance and
`nufft` the NUFFT library NUFSHT runs (a FlowTransformBindings tag; `AutoSpectralBackend()` lets NUFSHT
choose); `rtol`/`maxiter` control the fits (`T` the field's element type), and the result's
`converged`, `iterations` and `residual` report how they ended. `transform`, when given, must be
`SpectralBackends.NUFSHTSpectralBackend()`. Requires `using NUFSHT`.
"""
function FIT.calculate_energy_transfer(
    method::FIT.Types.SphericalTransferMethod,
    vorticity::AbstractVector{<:Real},
    coords::Tuple{<:AbstractVector, <:AbstractVector};
    lmax::Integer,
    dealias::Bool = true,
    tol::Real = 1e-10,
    rtol::Union{Nothing, Real} = nothing,
    maxiter::Integer = 4000,
    transform = nothing,
    quadrature_weights::Union{Nothing, AbstractVector} = nothing,
    nufft::SpectralBackends.AbstractSpectralBackend = SpectralBackends.AutoSpectralBackend(),
    execution::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.SerialBackend(),
)
    FIT.Spherical._validate_spherical_backends(transform, execution, :scattered)
    M = length(vorticity)
    (length(coords[1]) == M && length(coords[2]) == M) || throw(ArgumentError(
        "vorticity and both coordinate vectors must have equal length; got $((M, length(coords[1]), length(coords[2])))."))
    ws = FIT.Spherical.ScatteredSphericalTransferWorkspace(
        coords, lmax; radius = float(method.radius), dealias = dealias,
        tol = tol, rtol = rtol, maxiter = maxiter, T = float(eltype(vorticity)),
        quadrature_weights = quadrature_weights, nufft = nufft, execution = execution)
    try
        return FIT.Spherical.calculate_spherical_transfer!(ws, vorticity)
    finally
        FIT.close!(ws)
    end
end

# ---------------------------------------------------------------------------
# Divergent horizontal-KE spectral transfer at scattered points (full rotational + divergent flow), in
# the formulation of FlowInvariantTransfer.Spherical (skew-symmetric ½δu conservation term, single 1/a
# radius factor). Input is the horizontal velocity (u_θ, u_φ) at the scattered points; the transfer
# conserves total KE (Σ_l T ≈ 0) and reduces to the barotropic `SphericalTransferMethod` energy
# transfer when δ = 0.
# ---------------------------------------------------------------------------

"""
    ScatteredDivergentSphericalTransferWorkspace(coords, lmax; radius=1.0, dealias=true,
                                                 tol=1e-10, rtol=max(tol, 100eps(T)), maxiter=4000,
                                                 T=Float64, quadrature_weights=nothing,
                                                 nufft=AutoSpectralBackend(),
                                                 execution=ComputationalBackends.SerialBackend())

Reusable buffers + the five NUFSHT spin plans (points preset) for the scattered divergent KE transfer.
`coords = (θ, φ)` are the `M` colatitudes/longitudes. Without `quadrature_weights` it needs
`M ≥ (2·lmax+1)²` **equidistributed** points (e.g. spherical-Fibonacci) for well-conditioned
coefficient recovery. `nufft` is the NUFFT library NUFSHT runs, as for the barotropic workspace.
Requires `using NUFSHT`.
"""
function FIT.Spherical.ScatteredDivergentSphericalTransferWorkspace(
    coords::Tuple{<:AbstractVector, <:AbstractVector},
    lmax::Integer;
    radius::Real = 1.0,
    dealias::Bool = true,
    tol::Real = 1e-10,
    rtol::Union{Nothing, Real} = nothing,
    maxiter::Integer = 4000,
    T::Type = Float64,
    quadrature_weights::Union{Nothing, AbstractVector} = nothing,
    nufft::SpectralBackends.AbstractSpectralBackend = SpectralBackends.AutoSpectralBackend(),
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

    # `execution` sets where the plans run, as in the barotropic workspace above. Five plans: spin ±1
    # & spin-0 at lmax, spin-0 & spin+1 at lwork.
    exec = FIT.Types.resolve_execution(execution)
    planp  = NUFSHT.make_spin_plan(CT, θ, φ,lmax,   1; tol = tol, execution = exec, nufft = nufft)
    planm  = NUFSHT.make_spin_plan(CT, θ, φ,lmax,  -1; tol = tol, execution = exec, nufft = nufft)
    plan0  = NUFSHT.make_spin_plan(CT, θ, φ,lmax,   0; tol = tol, execution = exec, nufft = nufft)
    plan0w = NUFSHT.make_spin_plan(CT, θ, φ,lwork,  0; tol = tol, execution = exec, nufft = nufft)
    planpw = NUFSHT.make_spin_plan(CT, θ, φ,lwork,  1; tol = tol, execution = exec, nufft = nufft)

    # Every buffer lives in the plans' memory, which `allocate_coefficients` allocates in.
    mem = NUFSHT.allocate_coefficients(plan0)
    _z(dims...) = fill!(similar(mem, CT, dims...), zero(CT))
    ap = _z(lmax + 1, 2lmax + 1); am = _z(lmax + 1, 2lmax + 1)
    sym = _z(lmax + 1, 2lmax + 1); anti = _z(lmax + 1, 2lmax + 1)
    ζc = _z(lmax + 1, 2lmax + 1); δc = _z(lmax + 1, 2lmax + 1)
    Khat = _z(lwork + 1, 2lwork + 1); Adv_lm = _z(lwork + 1, 2lwork + 1)
    Up = _z(M); Um = _z(M); ζv = _z(M); δv = _z(M); Kv = _z(M); gradK = _z(M); Advv = _z(M)
    uθ = similar(mem, FT, M); uφ = similar(mem, FT, M)
    # Per-row ladders √(ℓ(ℓ+1)) for the row-broadcast coefficient-space ops.
    ladl = reshape(similar(mem, FT, lmax + 1), lmax + 1, 1)
    copyto!(ladl, FT[sqrt(ℓ * (ℓ + 1)) for ℓ in 0:lmax])
    ladw = reshape(similar(mem, FT, lwork + 1), lwork + 1, 1)
    copyto!(ladw, FT[sqrt(ℓ * (ℓ + 1)) for ℓ in 0:lwork])
    Pr = similar(mem, FT, lmax + 1, 2lmax + 1)
    Tcol = similar(mem, FT, lmax + 1, 1)
    result = FIT.Types.DivergentSphericalTransferResult(
        collect(FT, 0:lmax), zeros(FT, lmax + 1), zeros(FT, lmax + 1), zeros(FT, lmax + 1),
        zeros(FT, lmax + 1), zeros(FT, lmax + 1), zeros(FT, lmax + 1), true, 0, zero(FT))

    qw = quadrature_weights === nothing ? nothing : copyto!(similar(mem, FT, M), quadrature_weights)

    return FIT.Spherical.ScatteredDivergentSphericalTransferWorkspace(
        planp, planm, plan0, plan0w, planpw,
        _fit_workspace(qw, planp), _fit_workspace(qw, planm), _fit_workspace(qw, plan0w),
        _fit_workspace(qw, planpw),
        ap, am, sym, anti, ζc, δc, Khat, Adv_lm,
        Up, Um, ζv, δv, Kv, gradK, Advv, uθ, uφ, ladl, ladw, Pr, Tcol, qw, result,
        FT(radius), Int(lmax), Int(lwork), _default_rtol(rtol, tol, FT), Int(maxiter))
end

function FIT.close!(ws::FIT.Spherical.ScatteredDivergentSphericalTransferWorkspace)
    foreach(NUFSHT.close!, (ws.planp, ws.planm, ws.plan0, ws.plan0w, ws.planpw))
    return ws
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
    uθ = copyto!(ws.uθ, u_θ); uφ = copyto!(ws.uφ, u_φ)
    @. ws.Up = uθ + im * uφ
    @. ws.Um = uθ - im * uφ
    rt = ws.rtol; mi = ws.maxiter
    fill!(ws.ap, zero(CT)); fit = _analyze!(ws.ap, ws.Up, ws.planp, ws.qw, ws.lsmrp; rtol = rt, maxiter = mi)
    fill!(ws.am, zero(CT))
    fit = FIT.Spherical.merge_fits(fit, _analyze!(ws.am, ws.Um, ws.planm, ws.qw, ws.lsmrm; rtol = rt, maxiter = mi))
    @. ws.sym  = (ws.ap + ws.am) / 2
    @. ws.anti = (ws.ap - ws.am) / 2

    # Unit-sphere vorticity/divergence via the eth ladder (Goldberg convention, ð ₛY=+√((ℓ−s)(ℓ+s+1))ₛ₊₁Y):
    # ζ_lm = +i√(ℓ(ℓ+1)) sym,  δ_lm = -√(ℓ(ℓ+1)) anti; synthesise both at the points (spin-0).
    @. ws.ζc =  im * ws.ladl * ws.sym
    @. ws.δc = -ws.ladl * ws.anti
    NUFSHT.nusht_type2_spin!(ws.ζv, ws.ζc, ws.plan0)
    NUFSHT.nusht_type2_spin!(ws.δv, ws.δc, ws.plan0)

    # K = ½|u|² (real, held complex); analyse at the dealiased degree lwork.
    @. ws.Kv = 0.5 * (uθ^2 + uφ^2)
    fill!(ws.Khat, zero(CT))
    fit = FIT.Spherical.merge_fits(fit, _analyze!(ws.Khat, ws.Kv, ws.plan0w, ws.qw, ws.lsmr0w; rtol = rt, maxiter = mi))

    # ∇K = ð K = +√(ℓ(ℓ+1)) synth_spin+1(K̂)  (reuse Khat for the ladder-scaled coefficients).
    @. ws.Khat = ws.ladw * ws.Khat
    NUFSHT.nusht_type2_spin!(ws.gradK, ws.Khat, ws.planpw)

    # Skew-symmetric energy-conserving advection A = ∇K + (iζ + ½δ) U₊; analyse (spin+1) at lwork.
    @. ws.Advv = ws.gradK + (im * ws.ζv + 0.5 * ws.δv) * ws.Up
    fill!(ws.Adv_lm, zero(CT))
    fit = FIT.Spherical.merge_fits(fit, _analyze!(ws.Adv_lm, ws.Advv, ws.planpw, ws.qw, ws.lsmrpw; rtol = rt, maxiter = mi))

    # Per-degree channel reduction T_rot = Σ_m Re{sym* Â}, T_div = Σ_m Re{anti* Â} (single 1/a factor).
    # A_lm (degree lwork) aligns to the lmax layout by a centred column slice; |m|>ℓ corners are 0.
    Adv_al = @view ws.Adv_lm[1:lmax + 1, (lwork - lmax + 1):(lwork + lmax + 1)]
    @. ws.Pr = real(conj(ws.sym) * Adv_al)
    sum!(ws.Tcol, ws.Pr)
    copyto!(ws.result.rotational_transfer, ws.Tcol); ws.result.rotational_transfer ./= a
    @. ws.Pr = real(conj(ws.anti) * Adv_al)
    sum!(ws.Tcol, ws.Pr)
    copyto!(ws.result.divergent_transfer, ws.Tcol); ws.result.divergent_transfer ./= a
    return FIT.Spherical.with_fit(FIT.Spherical.divergent_transfer_finalize!(ws.result), fit)
end

function FIT.calculate_energy_transfer(
    method::FIT.Types.DivergentSphericalTransferMethod,
    velocity::Tuple{<:AbstractVector, <:AbstractVector},
    coords::Tuple{<:AbstractVector, <:AbstractVector};
    lmax::Integer,
    dealias::Bool = true,
    tol::Real = 1e-10,
    rtol::Union{Nothing, Real} = nothing,
    maxiter::Integer = 4000,
    transform = nothing,
    quadrature_weights::Union{Nothing, AbstractVector} = nothing,
    nufft::SpectralBackends.AbstractSpectralBackend = SpectralBackends.AutoSpectralBackend(),
    execution::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.SerialBackend(),
)
    FIT.Spherical._validate_spherical_backends(transform, execution, :scattered)
    u_θ, u_φ = velocity
    M = length(u_θ)
    (length(u_φ) == M && length(coords[1]) == M && length(coords[2]) == M) || throw(ArgumentError(
        "velocity components and both coordinate vectors must have equal length; got " *
        "$((M, length(u_φ), length(coords[1]), length(coords[2])))."))
    ws = FIT.Spherical.ScatteredDivergentSphericalTransferWorkspace(
        coords, lmax; radius = float(method.radius), dealias = dealias, tol = tol, rtol = rtol,
        maxiter = maxiter, T = float(eltype(u_θ)), quadrature_weights = quadrature_weights,
        nufft = nufft, execution = execution)
    try
        return FIT.calculate_divergent_spherical_transfer!(ws, u_θ, u_φ)
    finally
        FIT.close!(ws)
    end
end

end # module FlowInvariantTransferNUFSHTExt
