module FlowInvariantTransferFSHExt

using FastSphericalHarmonics: FastSphericalHarmonics as FSH
using FlowInvariantTransfer: FlowInvariantTransfer as FIT
using FlowInvariantTransfer.Types: SphericalTransferMethod
using FlowInvariantTransfer.Spherical: spherical_transfer_reduce

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
function _sph_grad(C0::AbstractMatrix{<:Real})
    ðC = FSH.spinsph_eth(C0, 0)             # Array{SVector{2,Float64},2}
    G  = FSH.spinsph_evaluate(ðC, 1)        # SVector(re, im) of ðf at each grid point
    return [complex(G[i, j][1], G[i, j][2]) for i in axes(G, 1), j in axes(G, 2)]
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
function FIT.calculate_energy_transfer(
    method::SphericalTransferMethod,
    vorticity::AbstractMatrix{<:Real};
    dealias::Bool = true,
    kwargs...,
)
    Nθ, Nφ = size(vorticity)
    Nφ == 2Nθ - 1 || throw(ArgumentError(
        "vorticity must lie on the FastSphericalHarmonics grid of size (lmax+1, 2lmax+1); got $((Nθ, Nφ))."))
    lmax = Nθ - 1
    a = float(method.radius)

    # FastSphericalHarmonics (a double-precision FastTransforms C library) is Float64-only — its
    # spinsph_transform/eth/evaluate accept only `Array{Float64}`/`Array{Complex{Float64}}` — so this
    # path necessarily computes (and returns) in Float64 regardless of the input element type.
    Cζ0 = FSH.spinsph_transform(Matrix{Float64}(vorticity), 0)     # ζ̂_lm (real, spinsph(0) layout)

    # Dealiasing: evaluate the quadratic Jacobian on a grid resolving 2·lmax (exact for a product of
    # two lmax-band-limited fields), then keep l ≤ lmax. Embed mode-by-mode — spinsph_mode(s,l,m) is
    # grid-size-independent, but the unused entries of a smaller coefficient array must NOT be copied
    # (they map to valid higher-degree modes on the larger grid).
    lwork = dealias ? 2 * lmax : lmax
    Nwork = lwork + 1
    Cζ = zeros(Float64, Nwork, 2Nwork - 1)
    Cψ = zeros(Float64, Nwork, 2Nwork - 1)
    @inbounds for l in 0:lmax, m in -l:l
        i = FSH.spinsph_mode(0, l, m)
        Cζ[i] = Cζ0[i]
        l ≥ 1 && (Cψ[i] = -a^2 / (l * (l + 1)) * Cζ0[i])          # ψ = ∇⁻²ζ  (l=0 → 0)
    end

    # A = J(ψ,ζ) = (1/a²) Im{ conj(ðψ)·ðζ }, evaluated on the (dealiased) work grid.
    Gψ = _sph_grad(Cψ)
    Gζ = _sph_grad(Cζ)
    J = @. imag(conj(Gψ) * Gζ) / a^2
    CA = FSH.spinsph_transform(J, 0)                               # Â_lm on the work grid

    # Reduce over the resolved input degrees l ≤ lmax (higher work-grid modes carry no ψ/ζ content).
    nmode = (lmax + 1)^2
    degs = Vector{Int}(undef, nmode)
    ψv = Vector{Float64}(undef, nmode)
    ζv = Vector{Float64}(undef, nmode)
    Av = Vector{Float64}(undef, nmode)
    k = 0
    @inbounds for l in 0:lmax, m in -l:l
        k += 1
        i = FSH.spinsph_mode(0, l, m)
        degs[k] = l
        ψv[k] = Cψ[i]
        ζv[k] = Cζ[i]
        Av[k] = CA[i]
    end
    return spherical_transfer_reduce(degs, ψv, ζv, Av, lmax)
end

end # module FlowInvariantTransferFSHExt
