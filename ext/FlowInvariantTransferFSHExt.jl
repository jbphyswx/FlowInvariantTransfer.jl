module FlowInvariantTransferFSHExt

using FastSphericalHarmonics: FastSphericalHarmonics as FSH
using FlowInvariantTransfer: FlowInvariantTransfer as FIT
using FlowInvariantTransfer.Types: SphericalTransferMethod, SphericalTransferResult
using FlowInvariantTransfer.Spherical: spherical_transfer_reduce, spherical_transfer_reduce!

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
function FIT.SphericalTransferWorkspace(lmax::Integer; radius::Real = 1.0, dealias::Bool = true)
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
    result = SphericalTransferResult(
        collect(Float64, 0:lmax), zeros(Float64, lmax + 1), zeros(Float64, lmax + 1),
        zeros(Float64, lmax + 1), zeros(Float64, lmax + 1))
    return FIT.Spherical.SphericalTransferWorkspace(
        Cζ, Cψ, Gψ, Gζ, J, degs, ψv, ζv, Av, result, Float64(radius), Int(lmax), dealias)
end

function FIT.calculate_spherical_transfer!(
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
    return spherical_transfer_reduce!(ws.result, ws.degs, ws.ψv, ws.ζv, ws.Av)
end

function FIT.calculate_energy_transfer(
    method::SphericalTransferMethod,
    vorticity::AbstractMatrix{<:Real};
    dealias::Bool = true,
    kwargs...,
)
    Nθ, Nφ = size(vorticity)
    Nφ == 2Nθ - 1 || throw(ArgumentError(
        "vorticity must lie on the FastSphericalHarmonics grid of size (lmax+1, 2lmax+1); got $((Nθ, Nφ))."))
    ws = FIT.SphericalTransferWorkspace(Nθ - 1; radius = float(method.radius), dealias = dealias)
    return FIT.calculate_spherical_transfer!(ws, vorticity)
end

end # module FlowInvariantTransferFSHExt
