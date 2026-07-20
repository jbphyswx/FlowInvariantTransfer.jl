module FlowInvariantTransferNUFSHTExt

using NUFSHT: NUFSHT
using FlowInvariantTransfer: FlowInvariantTransfer as FIT
using FlowInvariantTransfer.Types: SphericalTransferMethod, SphericalTransferResult
using FlowInvariantTransfer.Backends: AbstractExecutionBackend, SerialBackend, ThreadedBackend

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
    execution::AbstractExecutionBackend = SerialBackend(),
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

    # The three NUFSHT spin plans (points preset) — the dominant, reusable cost. `nthreads=1` (Serial,
    # default) keeps the FINUFFT-backed transforms single-threaded: FINUFFT shares libfftw3 with FFTW.jl,
    # so a multithreaded plan spawns Julia Tasks per exec (allocating), and single-threaded is 0-alloc +
    # oversubscription-free when the outer batch axis is parallelised. ThreadedBackend threads a lone call.
    nthr = execution isa ThreadedBackend ? Threads.nthreads() : 1
    plan0  = NUFSHT.make_spin_plan(θ, φ, lmax,  0; tol = tol, T = FT, nthreads = nthr)
    plan1  = NUFSHT.make_spin_plan(θ, φ, lmax,  1; tol = tol, T = FT, nthreads = nthr)
    plan0w = NUFSHT.make_spin_plan(θ, φ, lwork, 0; tol = tol, T = FT, nthreads = nthr)

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
    result = SphericalTransferResult(
        collect(FT, 0:lmax), zeros(FT, lmax + 1), zeros(FT, lmax + 1),
        zeros(FT, lmax + 1), zeros(FT, lmax + 1))

    return FIT.Spherical.ScatteredSphericalTransferWorkspace(
        plan0, plan1, plan0w, ζ_lm, ψ_lm, ðψ, ðζ, A_lw, Gψ, Gζ, ζdata, Jc,
        degcol, Pr, Tcol, result, FT(radius), Int(lmax), Int(lwork), FT(rtol), Int(maxiter))
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
    NUFSHT.nusht_solve_spin!(ws.A_lw, ws.Jc, ws.plan0w; rtol = ws.rtol, maxiter = ws.maxiter)

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
    method::SphericalTransferMethod,
    vorticity::AbstractVector{<:Real},
    coords::Tuple{<:AbstractVector, <:AbstractVector};
    lmax::Integer,
    dealias::Bool = true,
    tol::Real = 1e-10,
    rtol::Real = 1e-10,
    maxiter::Integer = 4000,
    execution::AbstractExecutionBackend = SerialBackend(),
    kwargs...,
)
    M = length(vorticity)
    (length(coords[1]) == M && length(coords[2]) == M) || throw(ArgumentError(
        "vorticity and both coordinate vectors must have equal length; got $((M, length(coords[1]), length(coords[2])))."))
    ws = FIT.ScatteredSphericalTransferWorkspace(
        coords, lmax; radius = float(method.radius), dealias = dealias,
        tol = tol, rtol = rtol, maxiter = maxiter, T = float(eltype(vorticity)), execution = execution)
    return FIT.calculate_spherical_transfer!(ws, vorticity)
end

end # module FlowInvariantTransferNUFSHTExt
