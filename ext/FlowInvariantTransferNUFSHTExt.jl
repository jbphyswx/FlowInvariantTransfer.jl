module FlowInvariantTransferNUFSHTExt

using NUFSHT: NUFSHT
using FlowTransformBindings: FlowTransformBindings as FTB
using FlowInvariantTransfer: FlowInvariantTransfer as FIT
using ComputationalBackends: ComputationalBackends
using SpectralBackends: SpectralBackends
using LinearAlgebra: LinearAlgebra

# ---------------------------------------------------------------------------
# Spherical spectral energy/enstrophy transfer at scattered points on the sphere, via NUFSHT
# (non-uniform spherical-harmonic transforms): the 2D-barotropic formulation of the FSH regular-grid
# path (core reduction: FlowInvariantTransfer.Spherical) with NUFFT-backed scattered transforms.
#
# NUFSHT's spin-weighted harmonics are ₛYℓm = √((2ℓ+1)/4π) d^ℓ_{m,-s}(θ) e^{imφ}, so the eth ladder is
# ð(ₛYℓm) = √((ℓ-s)(ℓ+s+1)) ₛ₊₁Yℓm. For a spin-0 field the spin-1 synthesis of √(ℓ(ℓ+1))·f̂_lm is
# ðf = -(∂_θ + i/sinθ ∂_φ)f, giving J(ψ,ζ) = (1/a²) Im{conj(ðψ)·ðζ}.
#
# The degree-lmax coefficients of the sampled field come from a least-squares (LSMR) fit, which
# determines them on equidistributed points (spherical Fibonacci); clustered or jittered points recover
# the field and leave the coefficients undetermined. Every later step reads only those coefficients, so
# the quadratic products, of degree ≤ 2·lmax, are formed on a Gauss–Legendre grid and projected onto
# degree ≤ lmax by its quadrature, exact for the degree-3·lmax integrand.
# ---------------------------------------------------------------------------

# A fit's relative residual resolves no finer than the NUFFT accuracy `tol`, and NUFSHT floors it at
# `eps(FT)`, so the default tolerance sits above both.
_default_rtol(rtol::Real, ::Real, ::Type{FT}) where {FT} = FT(rtol)
_default_rtol(::Nothing, tol::Real, ::Type{FT}) where {FT} = max(FT(tol), 100 * eps(FT))

# Colatitudes, longitudes and weights (`Σw = 4π`) of the product rule exact for every spherical
# polynomial of degree ≤ `L`: the `⌊L/2⌋+1` Gauss–Legendre nodes in cos θ by `L+1` equispaced longitudes.
# Nodes and weights from the eigenvectors of the Legendre recurrence's Jacobi matrix (Golub & Welsch 1969).
function _gauss_grid(L::Integer)
    nlat = L ÷ 2 + 1
    nlon = L + 1
    J = LinearAlgebra.SymTridiagonal(zeros(nlat), [k / sqrt(4k^2 - 1.0) for k in 1:(nlat - 1)])
    E = LinearAlgebra.eigen(J)
    θ = acos.(clamp.(E.values, -1.0, 1.0))
    w = (4π / nlon) .* E.vectors[1, :] .^ 2
    return repeat(θ, nlon), repeat((2π / nlon) .* (0:(nlon - 1)); inner = nlat), repeat(w, nlon)
end

# The degree the grid's rule integrates: projecting a degree-2·lmax product onto degree lmax integrates
# degree 3·lmax. Without dealiasing it integrates degree 2·lmax, exact for products of degree ≤ lmax.
_grid_degree(lmax::Integer, dealias::Bool) = dealias ? 3lmax : 2lmax

# The fit determines `(lmax+1)²` coefficients from one value per point, under at most one weight per
# point; a quadrature takes one weight per point and runs no fit.
function _check_analysis(M::Integer, lmax::Integer, ::Nothing, w)
    M ≥ (lmax + 1)^2 || throw(ArgumentError(
        "the fit determines (lmax+1)² = $((lmax + 1)^2) coefficients from M = $M points, fewer than it " *
        "needs; analyse at a lower degree, or use more points (equidistributed, e.g. spherical Fibonacci)."))
    (w === nothing || length(w) == M) || throw(ArgumentError("weights has length $(length(w)) for $M nodes."))
    return nothing
end
function _check_analysis(M::Integer, ::Integer, qw::AbstractVector, w)
    length(qw) == M || throw(ArgumentError("quadrature_weights has length $(length(qw)) for $M nodes."))
    w === nothing || throw(ArgumentError(
        "quadrature_weights replace the fit by a projection, which takes no fit `weights`."))
    return nothing
end

# The operator a field is fitted on: `plan`, or `plan` under the point weights `w`, whose fit minimizes
# `Σⱼ wⱼ |(A c − f)ⱼ|²`. `w` is copied into the plan's memory.
_fit_operator(plan, ::Nothing, ::Type) = plan
_fit_operator(plan, w::AbstractVector, ::Type{FT}) where {FT} =
    FTB.RangeWeighted(plan, copyto!(similar(NUFSHT.allocate_coefficients(plan), FT, length(w)), FT.(w)))

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
    weights::Union{Nothing, AbstractVector} = nothing,
    nufft::SpectralBackends.AbstractSpectralBackend = SpectralBackends.AutoSpectralBackend(),
    execution::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.SerialBackend(),
)
    θ, φ = coords
    M = length(θ)
    length(φ) == M || throw(ArgumentError("θ and φ must have equal length; got $((length(θ), length(φ)))."))
    lmax ≥ 1 || throw(ArgumentError("lmax must be ≥ 1; got $lmax."))
    _check_analysis(M, lmax, quadrature_weights, weights)
    FT = float(T)
    CT = Complex{FT}

    # `execution` sets where the plans run: one thread under `SerialBackend()` (the default), which
    # leaves the cores to an outer batch axis, every thread under `ThreadedBackend()`, and the device of
    # `GPUBackend(b)`, which holds their nodes. Device-array coordinates give device plans under any of
    # them. The vorticity and the Jacobian are real fields, so their plans are a real field's.
    exec = FIT.Types.resolve_execution(execution)
    θg, φg, wq = _gauss_grid(_grid_degree(lmax, dealias))
    Mg = length(θg)
    kw = (; tol = tol, execution = exec, nufft = nufft)
    plan0 = NUFSHT.make_spin_plan(FT, θ, φ, lmax, 0; kw...)
    grid1 = NUFSHT.make_spin_plan(CT, θg, φg, lmax, 1; ntrans = 2, directions = NUFSHT.SynthesisOnly(), kw...)
    grid0 = NUFSHT.make_spin_plan(FT, θg, φg, lmax, 0; kw...)

    # Every buffer lives in the plans' memory, which `allocate_coefficients` allocates in.
    mem = NUFSHT.allocate_coefficients(plan0)
    _z(S, dims...) = fill!(similar(mem, S, dims...), zero(S))
    ζ_lm = _z(CT, lmax + 1, 2lmax + 1)
    ψ_lm = _z(CT, lmax + 1, 2lmax + 1)
    A_lm = _z(CT, lmax + 1, 2lmax + 1)
    ðc = _z(CT, lmax + 1, 2lmax + 1, 2)
    Gg = _z(CT, Mg, 2)
    fr = similar(mem, FT, M)
    Jg = _z(FT, Mg)
    gw = copyto!(similar(mem, FT, Mg), FT.(wq))
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

    qw = quadrature_weights === nothing ? nothing : copyto!(similar(mem, FT, M), FT.(quadrature_weights))
    fit0 = _fit_operator(plan0, weights, FT)

    return FIT.Spherical.ScatteredSphericalTransferWorkspace(
        plan0, fit0, grid1, grid0, _fit_workspace(qw, fit0), ζ_lm, ψ_lm, A_lm, ðc, Gg, fr, Jg, gw,
        ladl, invll1, Pr, Tcol, qw, result, FT(radius), Int(lmax), _default_rtol(rtol, tol, FT), Int(maxiter))
end

function FIT.close!(ws::FIT.Spherical.ScatteredSphericalTransferWorkspace)
    foreach(NUFSHT.close!, (ws.plan0, ws.grid1, ws.grid0))
    return ws
end

# The LSMR workspace of a fit on `op`; analysis against quadrature weights runs no fit.
_fit_workspace(::Nothing, op) = FTB.LSMRWorkspace(op)
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
    # Real values at spin 0 are a real field, whose plan fits the `(lmax+1)²` real degrees of freedom.
    FE = (s == 0 && eltype(values) <: Real) ? FT : CT
    exec = FIT.Types.resolve_execution(execution)
    pa = NUFSHT.make_spin_plan(FE, θ[active], φ[active], lmax, s; tol = tol, execution = exec, nufft = nufft)
    pf = nothing
    try
        pf = NUFSHT.make_spin_plan(FE, θ, φ, lmax, s; tol = tol, execution = exec, nufft = nufft)
        mem = NUFSHT.allocate_coefficients(pa)
        C = fill!(similar(mem, CT, lmax + 1, 2lmax + 1), zero(CT))
        f = copyto!(similar(mem, FE, Ma), FE.(values[active]))
        r = NUFSHT.nusht_analyze!(C, f, pa; ws = FTB.LSMRWorkspace(pa), rtol = _default_rtol(rtol, tol, FT),
                                  maxiter = maxiter)
        out = similar(mem, FE, M)
        NUFSHT.nusht_type2_spin!(out, C, pf)
        return Array(out), (; converged = r.converged, iterations = r.iterations, residual = FT(r.residual))
    finally
        NUFSHT.close!(pa)
        pf === nothing || NUFSHT.close!(pf)
    end
end

# The spin-weighted coefficients of `f` at the plan's nodes, into `C`, by `NUFSHT.nusht_analyze!`: the
# quadrature `Σⱼ wⱼ fⱼ conj(ₛYℓm(xⱼ))` with the weights `qw`, else the least-squares fit in the LSMR
# workspace `lsmr`; on a `RangeWeighted` operator the weighted fit. Returns the outcome as
# `Spherical.merge_fits` reads it, the residual in `rtol`'s float type.
function _analyze!(C, f, plan, qw, lsmr; rtol, maxiter)
    r = NUFSHT.nusht_analyze!(C, f, plan; weights = qw, ws = lsmr, rtol = rtol, maxiter = maxiter)
    return (; converged = r.converged, iterations = r.iterations, residual = oftype(rtol, r.residual))
end

function _analyze!(C, f, op::FTB.RangeWeighted, ::Nothing, lsmr; rtol, maxiter)
    r = NUFSHT.nusht_solve_spin!(C, f, op; ws = lsmr, rtol = rtol, maxiter = maxiter)
    return (; converged = r.converged, iterations = r.iterations, residual = oftype(rtol, r.residual))
end

function FIT.Spherical.calculate_spherical_transfer!(
    ws::FIT.Spherical.ScatteredSphericalTransferWorkspace,
    vorticity::AbstractVector{<:Real},
)
    M = length(ws.fr)
    length(vorticity) == M || throw(DimensionMismatch(
        "vorticity length $(length(vorticity)) ≠ workspace points $M."))
    a = ws.radius
    CT = eltype(ws.ζ_lm)

    # Fit ζ → spin-0 coefficients (points preset in plan0).
    copyto!(ws.fr, vorticity)
    fill!(ws.ζ_lm, zero(CT))
    fit = _analyze!(ws.ζ_lm, ws.fr, ws.fit0, ws.qw, ws.lsmr0; rtol = ws.rtol, maxiter = ws.maxiter)

    # ψ = ∇⁻²ζ and the eth ladder → spin-1 gradient coefficients, as row-broadcasts over the
    # (degree = row, m = column) coefficient matrices — device-generic, no scalar indexing. The ℓ=0 row
    # and the |m|>ℓ corners are 0 in ζ_lm, so they stay 0.
    @. ws.ψ_lm = -a^2 * ws.invll1 * ws.ζ_lm
    @views @. ws.ðc[:, :, 1] = ws.ladl * ws.ψ_lm
    @views @. ws.ðc[:, :, 2] = ws.ladl * ws.ζ_lm

    # ðψ, ðζ on the grid; A = J(ψ,ζ) there, projected onto degree ≤ lmax by the grid's quadrature.
    NUFSHT.nusht_type2_spin!(ws.Gg, ws.ðc, ws.grid1)
    @views @. ws.Jg = imag(conj(ws.Gg[:, 1]) * ws.Gg[:, 2]) / a^2
    NUFSHT.nusht_analyze!(ws.A_lm, ws.Jg, ws.grid0; weights = ws.gw)

    # Per-degree transfer = sum over m (matrix columns); the |m|>ℓ corners of ψ and ζ are 0, so the full
    # row-sum equals the sum over valid m. T_E(ℓ) = −Σ_m Re{ψ* A}, T_Z(ℓ) = +Σ_m Re{ζ* A}; then
    # Π(L) = Σ_{l≤L}T(l).
    @. ws.Pr = real(conj(ws.ψ_lm) * ws.A_lm)       # T_E(ℓ) = -Σ_m Re{ψ* A}  (fused, into preallocated Pr)
    sum!(ws.Tcol, ws.Pr)
    copyto!(ws.result.energy_transfer, ws.Tcol);  ws.result.energy_transfer .*= -1
    @. ws.Pr = real(conj(ws.ζ_lm) * ws.A_lm)       # T_Z(ℓ) = +Σ_m Re{ζ* A}
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
                              transform=nothing, quadrature_weights=nothing, weights=nothing,
                              nufft=AutoSpectralBackend(), execution=SerialBackend())

Spherical spectral energy/enstrophy transfer `T_E(l)`, `T_Z(l)` (and fluxes) for 2D non-divergent
flow on the sphere, from the **vorticity field** `ζ` sampled at `M` **scattered** points
`coords = (θ, φ)` (colatitudes `θ ∈ [0,π]`, longitudes `φ ∈ [0,2π)`, each length `M`). Returns a
[`SphericalTransferResult`](@ref) over degrees `l = 0…lmax`.

The vorticity's degree-`lmax` coefficients come from NUFSHT's least-squares fit, which needs
`M ≥ (lmax+1)²` points and determines them on **equidistributed** points (spherical Fibonacci, a
spherical `t`-design); clustered or jittered points reproduce the field while leaving the coefficients
the transfer reads undetermined. The fit converges in fewer iterations the more points it has per
coefficient. `weights`, one nonnegative value per point such as its area, make it the weighted fit
`min Σⱼ wⱼ |(A c − ζ)ⱼ|²`. Per-node `quadrature_weights` (`Σw = 4π`), for a rule exact at degree
`2·lmax`, replace the fit by a projection.

The Jacobian of the fitted fields, of degree `2·lmax`, is formed on a Gauss–Legendre grid and projected
onto degree `≤ lmax` by its quadrature, exact for that product (dealiased). `dealias=false` takes the
grid exact only for products of degree `≤ lmax` (aliased). `tol` is the NUFFT tolerance and
`nufft` the NUFFT library NUFSHT runs (a FlowTransformBindings tag; `AutoSpectralBackend()` lets NUFSHT
choose); `rtol`/`maxiter` control the fit (`T` the field's element type), and the result's
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
    weights::Union{Nothing, AbstractVector} = nothing,
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
        quadrature_weights = quadrature_weights, weights = weights, nufft = nufft, execution = execution)
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
                                                 T=Float64, quadrature_weights=nothing, weights=nothing,
                                                 nufft=AutoSpectralBackend(),
                                                 execution=ComputationalBackends.SerialBackend())

Reusable buffers and NUFSHT spin plans (points preset) for the scattered divergent KE transfer.
`coords = (θ, φ)` are the `M` colatitudes/longitudes. Without `quadrature_weights` the fit needs
`M ≥ (lmax+1)²` **equidistributed** points (e.g. spherical-Fibonacci) for well-conditioned coefficient
recovery, and `weights` make it the weighted fit as in the barotropic workspace; the advection of the
fitted velocity is projected on a Gauss–Legendre grid as there. `nufft` is the NUFFT library NUFSHT
runs. Requires `using NUFSHT`.
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
    weights::Union{Nothing, AbstractVector} = nothing,
    nufft::SpectralBackends.AbstractSpectralBackend = SpectralBackends.AutoSpectralBackend(),
    execution::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.SerialBackend(),
)
    θ, φ = coords
    M = length(θ)
    length(φ) == M || throw(ArgumentError("θ and φ must have equal length; got $((length(θ), length(φ)))."))
    lmax ≥ 1 || throw(ArgumentError("lmax must be ≥ 1; got $lmax."))
    _check_analysis(M, lmax, quadrature_weights, weights)
    FT = float(T)
    CT = Complex{FT}

    # `execution` sets where the plans run, as in the barotropic workspace above.
    exec = FIT.Types.resolve_execution(execution)
    θg, φg, wq = _gauss_grid(_grid_degree(lmax, dealias))
    Mg = length(θg)
    kw = (; tol = tol, execution = exec, nufft = nufft)
    planp = NUFSHT.make_spin_plan(CT, θ, φ, lmax, 1; kw...)
    gridp = NUFSHT.make_spin_plan(CT, θg, φg, lmax, 1; kw...)
    grid0 = NUFSHT.make_spin_plan(CT, θg, φg, lmax, 0; kw...)

    # Every buffer lives in the plans' memory, which `allocate_coefficients` allocates in.
    mem = NUFSHT.allocate_coefficients(planp)
    _z(S, dims...) = fill!(similar(mem, S, dims...), zero(S))
    ap = _z(CT, lmax + 1, 2lmax + 1); am = _z(CT, lmax + 1, 2lmax + 1)
    sym = _z(CT, lmax + 1, 2lmax + 1); anti = _z(CT, lmax + 1, 2lmax + 1)
    wc = _z(CT, lmax + 1, 2lmax + 1); Khat = _z(CT, lmax + 1, 2lmax + 1); Ahat = _z(CT, lmax + 1, 2lmax + 1)
    Up = _z(CT, M)
    uθ = similar(mem, FT, M); uφ = similar(mem, FT, M)
    Ug = _z(CT, Mg); wg = _z(CT, Mg); Bg = _z(CT, Mg); Kg = _z(FT, Mg)
    gw = copyto!(similar(mem, FT, Mg), FT.(wq))
    # Per-row ladder √(ℓ(ℓ+1)) and per-column sign (−1)^(m+1) for the coefficient-space ops.
    ladl = reshape(similar(mem, FT, lmax + 1), lmax + 1, 1)
    copyto!(ladl, FT[sqrt(ℓ * (ℓ + 1)) for ℓ in 0:lmax])
    msign = reshape(similar(mem, FT, 2lmax + 1), 1, 2lmax + 1)
    copyto!(msign, FT[isodd(m) ? 1 : -1 for m in -lmax:lmax])
    Pr = similar(mem, FT, lmax + 1, 2lmax + 1)
    Tcol = similar(mem, FT, lmax + 1, 1)
    result = FIT.Types.DivergentSphericalTransferResult(
        collect(FT, 0:lmax), zeros(FT, lmax + 1), zeros(FT, lmax + 1), zeros(FT, lmax + 1),
        zeros(FT, lmax + 1), zeros(FT, lmax + 1), zeros(FT, lmax + 1), true, 0, zero(FT))

    qw = quadrature_weights === nothing ? nothing : copyto!(similar(mem, FT, M), FT.(quadrature_weights))
    fitp = _fit_operator(planp, weights, FT)

    return FIT.Spherical.ScatteredDivergentSphericalTransferWorkspace(
        planp, fitp, gridp, grid0, _fit_workspace(qw, fitp),
        ap, am, sym, anti, wc, Khat, Ahat, Up, uθ, uφ, Ug, wg, Bg, Kg, gw, ladl, msign, Pr, Tcol, qw, result,
        FT(radius), Int(lmax), _default_rtol(rtol, tol, FT), Int(maxiter))
end

function FIT.close!(ws::FIT.Spherical.ScatteredDivergentSphericalTransferWorkspace)
    foreach(NUFSHT.close!, (ws.planp, ws.gridp, ws.grid0))
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
    a = ws.radius
    CT = eltype(ws.ap)

    # Spin+1 coefficients of U₊ = u_θ + i u_φ. U₋ = u_θ − i u_φ = conj(U₊) has the spin−1 coefficients
    # a₋[ℓ,m] = (−1)^(m+1) conj(a₊[ℓ,−m]), from conj(₁Y_ℓm) = (−1)^(m+1) ₋₁Y_ℓ,−m, the least-squares fit
    # included. Rotational/divergent split.
    uθ = copyto!(ws.uθ, u_θ); uφ = copyto!(ws.uφ, u_φ)
    @. ws.Up = uθ + im * uφ
    fill!(ws.ap, zero(CT))
    fit = _analyze!(ws.ap, ws.Up, ws.fitp, ws.qw, ws.lsmrp; rtol = ws.rtol, maxiter = ws.maxiter)
    @views @. ws.am = ws.msign * conj(ws.ap[:, end:-1:1])
    @. ws.sym  = (ws.ap + ws.am) / 2
    @. ws.anti = (ws.ap - ws.am) / 2

    # Unit-sphere vorticity/divergence via the eth ladder (Goldberg convention, ð ₛY=+√((ℓ−s)(ℓ+s+1))ₛ₊₁Y):
    # ζ_lm = +i√(ℓ(ℓ+1)) sym, δ_lm = -√(ℓ(ℓ+1)) anti, so iζ + ½δ has -√(ℓ(ℓ+1)) (sym + anti/2).
    @. ws.wc = -ws.ladl * (ws.sym + ws.anti / 2)

    # Skew-symmetric energy-conserving advection A = ∇K + (iζ + ½δ) U₊, K = ½|u|², with U₊, iζ + ½δ and
    # K on the grid; both terms projected onto degree ≤ lmax by its quadrature, ∇K's spin+1 coefficients
    # being √(ℓ(ℓ+1)) K̂ (∇K = ð K = +√(ℓ(ℓ+1)) synth_spin+1(K̂)).
    NUFSHT.nusht_type2_spin!(ws.Ug, ws.ap, ws.gridp)
    NUFSHT.nusht_type2_spin!(ws.wg, ws.wc, ws.grid0)
    @. ws.Kg = abs2(ws.Ug) / 2
    @. ws.Bg = ws.wg * ws.Ug
    NUFSHT.nusht_analyze!(ws.Khat, ws.Kg, ws.grid0; weights = ws.gw)
    NUFSHT.nusht_analyze!(ws.Ahat, ws.Bg, ws.gridp; weights = ws.gw)
    @. ws.Ahat += ws.ladl * ws.Khat

    # Per-degree channel reduction T_rot = Σ_m Re{sym* Â}, T_div = Σ_m Re{anti* Â} (single 1/a factor);
    # |m|>ℓ corners are 0.
    @. ws.Pr = real(conj(ws.sym) * ws.Ahat)
    sum!(ws.Tcol, ws.Pr)
    copyto!(ws.result.rotational_transfer, ws.Tcol); ws.result.rotational_transfer ./= a
    @. ws.Pr = real(conj(ws.anti) * ws.Ahat)
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
    weights::Union{Nothing, AbstractVector} = nothing,
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
        weights = weights, nufft = nufft, execution = execution)
    try
        return FIT.calculate_divergent_spherical_transfer!(ws, u_θ, u_φ)
    finally
        FIT.close!(ws)
    end
end

end # module FlowInvariantTransferNUFSHTExt
