module FlowInvariantTransferFSHExt

using FastSphericalHarmonics: FastSphericalHarmonics as FSH
using FlowInvariantTransfer: FlowInvariantTransfer as FIT
using ComputationalBackends: ComputationalBackends

# ---------------------------------------------------------------------------
# Spherical spectral energy/enstrophy transfer on a regular colatitude–longitude grid, via
# FastSphericalHarmonics. Formulation + conventions: THEORY.md §"Spherical spectral transfer";
# core reduction: FlowInvariantTransfer.Spherical.
#
# The horizontal gradient of a real spin-0 field f is the spin-1 "eth" field
#   ðf = -(∂_θ + i/sinθ ∂_φ) f,
# obtained exactly from FSH's real/SVector eth path (spinsph_eth on a real coefficient array
# returns SVector{2} coefficients with FSH's internal ±m storage-sign applied; this is the path
# validated in FSH's own test_spin.jl "grad" testset). The Jacobian is then
#   J(ψ,ζ) = (1/a²) Im{ conj(ðψ)·ðζ },
# which reproduces the analytic Jacobian to machine precision and conserves Σ_l T_E = Σ_l T_Z = 0.
# ---------------------------------------------------------------------------

# ðf = -(∂_θ + i/sinθ ∂_φ)f as a complex field, from real spin-0 coefficients `C0`.
# `spinsph_eth`/`spinsph_evaluate` allocate internally (FSH has no in-place API — floor); the
# assembled complex grid is written into the caller-provided `out` buffer.
function _sph_grad!(out::AbstractMatrix{<:Complex}, C0::AbstractMatrix{<:Real})
    ðC = FSH.spinsph_eth(C0, 0)             # Array{SVector{2,Float64},2}
    G  = FSH.spinsph_evaluate(ðC, 1)        # SVector(re, im) of ðf at each grid point
    @inbounds for j in axes(G, 2), i in axes(G, 1)
        out[i, j] = complex(G[i, j][1], G[i, j][2])
    end
    return out
end

"""
    calculate_energy_transfer(method::SphericalTransferMethod, vorticity::AbstractMatrix;
                              dealias=true, kwargs...)

Spherical spectral energy/enstrophy transfer `T_E(l)`, `T_Z(l)` (and fluxes) for 2D non-divergent
flow on the sphere, from the **vorticity field** `ζ` sampled on the FastSphericalHarmonics
equiangular colatitude–longitude grid — `size(vorticity) == (lmax+1, 2lmax+1)`, i.e. the grid from
`FastSphericalHarmonics.sph_points(lmax+1)`. Returns a [`SphericalTransferResult`](@ref).

The streamfunction is recovered spectrally as `ψ = ∇⁻²ζ` (`ψ̂_lm = -a²/(l(l+1)) ζ̂_lm`), so the flow
is treated as non-divergent; the `l=0` mode carries no transfer.

The advection `A = J(ψ,ζ)` is quadratic, so it has spectral content up to degree `2·lmax`. With
`dealias=true` (default) the products are evaluated on a grid resolving `2·lmax` and truncated back,
so the retained transfers `l ≤ lmax` are alias-free and conserve `Σ_l T_E = Σ_l T_Z = 0` to machine
precision. `dealias=false` computes on the native grid (aliased; conservation only for fields
band-limited well below `lmax`). Requires `using FastSphericalHarmonics`.
"""
# Build the reusable work arrays for a given resolution. FastSphericalHarmonics is Float64-only, so
# every buffer is Float64. `dealias` fixes the work-grid size (2·lmax vs lmax), so it is a
# workspace-level choice. The FSH transforms themselves (spinsph_transform/eth/evaluate) allocate
# internally on every call — no in-place API — so that portion is an irreducible floor; the workspace
# reuses the embed/Jacobian/reduction buffers (~20% of the per-call allocation here).
function FIT.Spherical.SphericalTransferWorkspace(lmax::Integer; radius::Real = 1.0, dealias::Bool = true)
    lwork = dealias ? 2 * lmax : lmax
    Nwork = lwork + 1
    Cζ = zeros(Float64, Nwork, 2Nwork - 1)
    Cψ = zeros(Float64, Nwork, 2Nwork - 1)
    Gψ = zeros(ComplexF64, Nwork, 2Nwork - 1)
    Gζ = zeros(ComplexF64, Nwork, 2Nwork - 1)
    J  = zeros(Float64, Nwork, 2Nwork - 1)
    nmode = (lmax + 1)^2
    degs = Vector{Int}(undef, nmode)
    ψv = Vector{Float64}(undef, nmode)
    ζv = Vector{Float64}(undef, nmode)
    Av = Vector{Float64}(undef, nmode)
    result = FIT.Types.SphericalTransferResult(
        collect(Float64, 0:lmax), zeros(Float64, lmax + 1), zeros(Float64, lmax + 1),
        zeros(Float64, lmax + 1), zeros(Float64, lmax + 1))
    return FIT.Spherical.SphericalTransferWorkspace(
        Cζ, Cψ, Gψ, Gζ, J, degs, ψv, ζv, Av, result, Float64(radius), Int(lmax), dealias)
end

function FIT.Spherical.calculate_spherical_transfer!(
    ws::FIT.Spherical.SphericalTransferWorkspace,
    vorticity::AbstractMatrix{<:Real},
)
    lmax = ws.lmax
    Nθ, Nφ = size(vorticity)
    (Nθ == lmax + 1 && Nφ == 2lmax + 1) || throw(ArgumentError(
        "vorticity size $((Nθ, Nφ)) does not match the workspace lmax=$lmax grid (lmax+1, 2lmax+1)."))
    a = ws.radius

    # ζ̂_lm (real spinsph(0) layout) — FSH-internal allocation (floor).
    Cζ0 = FSH.spinsph_transform(Matrix{Float64}(vorticity), 0)

    # Embed into the (dealiased) work grid, recovering ψ = ∇⁻²ζ mode-by-mode. Reuses ws.Cζ/ws.Cψ.
    fill!(ws.Cζ, 0.0)
    fill!(ws.Cψ, 0.0)
    @inbounds for l in 0:lmax, m in -l:l
        i = FSH.spinsph_mode(0, l, m)
        ws.Cζ[i] = Cζ0[i]
        l ≥ 1 && (ws.Cψ[i] = -a^2 / (l * (l + 1)) * Cζ0[i])
    end

    # A = J(ψ,ζ) = (1/a²) Im{ conj(ðψ)·ðζ }. The eth transforms are FSH-internal (floor); the assembled
    # complex gradients (ws.Gψ/ws.Gζ) and ws.J are reused.
    _sph_grad!(ws.Gψ, ws.Cψ)
    _sph_grad!(ws.Gζ, ws.Cζ)
    @. ws.J = imag(conj(ws.Gψ) * ws.Gζ) / a^2
    CA = FSH.spinsph_transform(ws.J, 0)                            # Â_lm — FSH-internal (floor)

    # Flatten to per-mode arrays (reused) and reduce into the reused result vectors.
    k = 0
    @inbounds for l in 0:lmax, m in -l:l
        k += 1
        i = FSH.spinsph_mode(0, l, m)
        ws.degs[k] = l
        ws.ψv[k] = ws.Cψ[i]
        ws.ζv[k] = ws.Cζ[i]
        ws.Av[k] = CA[i]
    end
    return FIT.Spherical.spherical_transfer_reduce!(ws.result, ws.degs, ws.ψv, ws.ζv, ws.Av)
end

function FIT.calculate_energy_transfer(
    method::FIT.Types.SphericalTransferMethod,
    vorticity::AbstractMatrix{<:Real};
    dealias::Bool = true,
    spectral = nothing,
    execution::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.SerialBackend(),
    kwargs...,
)
    FIT.Spherical._validate_spherical_backends(spectral, execution, :regular)
    Nθ, Nφ = size(vorticity)
    Nφ == 2Nθ - 1 || throw(ArgumentError(
        "vorticity must lie on the FastSphericalHarmonics grid of size (lmax+1, 2lmax+1); got $((Nθ, Nφ))."))
    ws = FIT.Spherical.SphericalTransferWorkspace(Nθ - 1; radius = float(method.radius), dealias = dealias)
    return FIT.Spherical.calculate_spherical_transfer!(ws, vorticity)
end

# ---------------------------------------------------------------------------
# DIVERGENT horizontal-KE spectral transfer on the regular grid (full rotational + divergent flow).
# Verified conventions (FlowInvariantTransfer.Spherical): skew-symmetric Lamb advection
#   A = ∇K + (iζ + ½δ) U₊,  U₊ = u_θ + i u_φ,  T(l) = Σ_m Re{conj(û)·Â}/a,  split into rot/div channels.
#
# FastSphericalHarmonics stores spin-s coefficients at the N-independent index spinsph_mode(s,l,m) =
# (l+1, col(m)), so a lower-degree coefficient array embeds into a higher-degree one by copying the
# same indices — the dealiasing (evaluate the quadratic products on the 2·lmax grid, analyse there,
# truncate to l ≤ lmax) reuses this, exactly like the barotropic path. FSH's spin transforms allocate
# internally (no in-place API — the floor). Three FSH facts underpin the conventions (all machine-precision
# verified): the real→SVector eth gives the physical gradient (ð f = -∇f); the complex ethbar with the
# per-column pre-flip α = sign(m ≥ 0) recovers ð̄U₊ = -(δ + iζ); and the complex transform is orthonormal
# so the coefficient inner product equals the physical one.
# ---------------------------------------------------------------------------

# col(m) ↔ FSH column packing (m=0→1, -1→2, +1→3, -2→4, +2→5, …); inverse used by the eth α-flip.
_fsh_colm(col) = col == 1 ? 0 : (col % 2 == 0 ? -1 : 1) * (col ÷ 2)

# ∇f as a complex spin+1 field, via the validated real→SVector eth path (same as `_sph_grad!`).
function _fsh_nabla(freal::AbstractMatrix{<:Real})
    ðC = FSH.spinsph_eth(FSH.spinsph_transform(Matrix{Float64}(freal), 0), 0)
    G = FSH.spinsph_evaluate(ðC, 1)                        # SVector{2} field = ð f = -∇f
    out = Matrix{ComplexF64}(undef, size(G))
    @inbounds for j in axes(G, 2), i in axes(G, 1)
        out[i, j] = -complex(G[i, j][1], G[i, j][2])       # ∇f = -ð f
    end
    return out
end

# Vorticity ζ = k̂·∇×u and divergence δ = ∇·u (real fields) from the spin+1 velocity, via the complex
# ethbar with the FSH per-column α pre-flip: ð̄U₊ = -(δ + iζ) ⟹ δ = -Re, ζ = -Im.
function _fsh_vort_div(uθ::AbstractMatrix{<:Real}, uφ::AbstractMatrix{<:Real})
    C = FSH.spinsph_transform(ComplexF64.(uθ .+ im .* uφ), 1)
    @inbounds for col in axes(C, 2)
        _fsh_colm(col) ≥ 0 || (@views C[:, col] .*= -1)
    end
    Cf = FSH.spinsph_evaluate(FSH.spinsph_ethbar(C, 1), 0)
    return -imag.(Cf), -real.(Cf)                          # ζ, δ
end

# Embed spin-s coefficients from a degree-lmax array into a fresh degree-lwork array (copy same indices).
function _fsh_embed(Csmall::AbstractMatrix, s::Integer, lmax::Integer, lwork::Integer)
    Cbig = zeros(eltype(Csmall), lwork + 1, 2lwork + 1)
    @inbounds for l in max(abs(s), 1):lmax, m in -l:l
        i = FSH.spinsph_mode(s, l, m); Cbig[i] = Csmall[i]
    end
    return Cbig
end

"""
    DivergentSphericalTransferWorkspace(lmax; radius=1.0, dealias=true)

Reusable result buffer for the regular-grid divergent KE transfer `!()`. FastSphericalHarmonics has no
in-place transform API, so the spin transforms allocate internally (an irreducible floor); this holds
only the reused [`DivergentSphericalTransferResult`](@ref) and resolution parameters. Requires
`using FastSphericalHarmonics`.
"""
function FIT.Spherical.DivergentSphericalTransferWorkspace(lmax::Integer; radius::Real = 1.0, dealias::Bool = true)
    lmax ≥ 1 || throw(ArgumentError("lmax must be ≥ 1; got $lmax."))
    lwork = dealias ? 2 * lmax : lmax
    z() = zeros(Float64, lmax + 1)
    result = FIT.Types.DivergentSphericalTransferResult(collect(Float64, 0:lmax), z(), z(), z(), z(), z(), z())
    return FIT.Spherical.DivergentSphericalTransferWorkspace(result, Float64(radius), Int(lmax), Int(lwork), dealias)
end

function FIT.calculate_divergent_spherical_transfer!(
    ws::FIT.Spherical.DivergentSphericalTransferWorkspace,
    u_θ::AbstractMatrix{<:Real},
    u_φ::AbstractMatrix{<:Real},
)
    lmax = ws.lmax; lwork = ws.lwork; a = ws.radius
    (size(u_θ) == (lmax + 1, 2lmax + 1) && size(u_φ) == (lmax + 1, 2lmax + 1)) || throw(ArgumentError(
        "velocity component sizes $((size(u_θ), size(u_φ))) do not match the workspace lmax=$lmax grid (lmax+1, 2lmax+1)."))

    a₊ = FSH.spinsph_transform(ComplexF64.(u_θ .+ im .* u_φ), 1)   # spin+1 velocity coefficients (lmax)
    ζin, δin = _fsh_vort_div(u_θ, u_φ)                             # vorticity/divergence fields (lmax grid)

    # Evaluate velocity + ζ,δ on the (dealiased) work grid so the quadratic advection is alias-free.
    if ws.dealias
        Uw = FSH.spinsph_evaluate(_fsh_embed(a₊, 1, lmax, lwork), 1)
        uθw = real.(Uw); uφw = imag.(Uw)
        ζw = real.(FSH.spinsph_evaluate(_fsh_embed(FSH.spinsph_transform(Matrix{Float64}(ζin), 0), 0, lmax, lwork), 0))
        δw = real.(FSH.spinsph_evaluate(_fsh_embed(FSH.spinsph_transform(Matrix{Float64}(δin), 0), 0, lmax, lwork), 0))
    else
        uθw = Matrix{Float64}(u_θ); uφw = Matrix{Float64}(u_φ); ζw = ζin; δw = δin
    end

    # Skew-symmetric advection A = ∇K + (iζ + ½δ) U₊ on the work grid; analyse (spin+1).
    K = 0.5 .* (uθw .^ 2 .+ uφw .^ 2)
    Adv = _fsh_nabla(K) .+ (im .* ζw .+ 0.5 .* δw) .* (uθw .+ im .* uφw)
    Â = FSH.spinsph_transform(ComplexF64.(Adv), 1)                 # spin+1 advection coefficients (lwork)

    # Divergent-channel velocity coefficients: anti = spin+1 coeffs of ∇χ, χ = ∇⁻²δ (lmax).
    δc = FSH.spinsph_transform(Matrix{Float64}(δin), 0)
    χc = zeros(Float64, lmax + 1, 2lmax + 1)
    @inbounds for l in 1:lmax, m in -l:l
        i = FSH.spinsph_mode(0, l, m); χc[i] = -δc[i] / (l * (l + 1))
    end
    anti = FSH.spinsph_transform(ComplexF64.(_fsh_nabla(FSH.spinsph_evaluate(χc, 0))), 1)

    # Per-degree channels T_rot = Σ_m Re{sym* Â}, T_div = Σ_m Re{anti* Â}; sym = a₊ − anti. The lwork Â
    # is truncated to l ≤ lmax automatically (same spinsph_mode index in both arrays). Single 1/a factor.
    Trot = ws.result.rotational_transfer; Tdiv = ws.result.divergent_transfer
    fill!(Trot, 0.0); fill!(Tdiv, 0.0)
    @inbounds for l in 1:lmax, m in -l:l
        i = FSH.spinsph_mode(1, l, m)
        tot = real(conj(a₊[i]) * Â[i]); dv = real(conj(anti[i]) * Â[i])
        Trot[l+1] += (tot - dv) / a
        Tdiv[l+1] += dv / a
    end
    return FIT.Spherical.divergent_transfer_finalize!(ws.result)
end

"""
    calculate_energy_transfer(method::DivergentSphericalTransferMethod,
                              velocity::Tuple{<:AbstractMatrix,<:AbstractMatrix}; dealias=true, kwargs...)

Divergent horizontal-KE spectral transfer for the full (rotational + divergent) flow on a regular
colatitude–longitude grid, from the horizontal velocity `(u_θ, u_φ)` on the FastSphericalHarmonics
grid — each `size == (lmax+1, 2lmax+1)`. Returns a [`DivergentSphericalTransferResult`](@ref). The
advection is dealiased at degree `2·lmax` (`dealias=false` skips it). Requires `using FastSphericalHarmonics`.
"""
function FIT.calculate_energy_transfer(
    method::FIT.Types.DivergentSphericalTransferMethod,
    velocity::Tuple{<:AbstractMatrix, <:AbstractMatrix};
    dealias::Bool = true,
    spectral = nothing,
    execution::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.SerialBackend(),
    kwargs...,
)
    FIT.Spherical._validate_spherical_backends(spectral, execution, :regular)
    u_θ, u_φ = velocity
    size(u_θ) == size(u_φ) || throw(ArgumentError(
        "velocity components must have equal size; got $((size(u_θ), size(u_φ)))."))
    Nθ, Nφ = size(u_θ)
    Nφ == 2Nθ - 1 || throw(ArgumentError(
        "velocity must lie on the FastSphericalHarmonics grid of size (lmax+1, 2lmax+1); got $((Nθ, Nφ))."))
    ws = FIT.Spherical.DivergentSphericalTransferWorkspace(Nθ - 1; radius = float(method.radius), dealias = dealias)
    return FIT.calculate_divergent_spherical_transfer!(ws, u_θ, u_φ)
end

end # module FlowInvariantTransferFSHExt
