module Compressible

using ..Types: CompressibleFluxResult, AbstractShellBinning, LinearBinning, AbstractShellGeometry,
               IsotropicShells, AbstractDealiasing, OrszagTwoThirds, NoDealiasing
using ..ShellBinning: shell_edges, shell_centers, assign_shells, shell_coordinate
using ..Utils: wavenumber_magnitude_grid
using ..NonlinearTerm: _is_dealiased

export calculate_compressible_flux

# ---------------------------------------------------------------------------
# Compressible kinetic-energy spectral transfer (Singh–Tiwari–Sharma–Verma 2025,
# arXiv:2508.04300; spec transcribed in THEORY.md §0.5). Framework A: momentum v = ρu,
# KE E_u(k) = ½Re[v(k)·u*(k)]. The nonlinear transfer is momentum-weighted and conserves
# total KE (Σ_k T_u = 0); the KE↔internal-energy exchange is the *separate* pressure-dilatation
# term Q_{I}, gated on a supplied pressure field.
#
# Net per-mode transfer, reduced from the mode-to-mode form (paper Eq. 20/28) to a
# pseudospectral O(Nᴰ) expression (derivation in THEORY.md; validated by Σ_k T_u = 0 and the
# incompressible limit ρ=const, ∇·u=0 ⇒ T_u = −ρ·Re{û*·(u·∇)u}, i.e. −ρ × the incompressible
# transfer_spectrum):
#
#     T_u(k) = −½ Re{ û*(k)·𝒩̂₁(k) } − ½ Re{ v̂*(k)·𝒩̂₂(k) }
#     𝒩₁ = (u·∇)v + v(∇·u) = ∂_j(v ⊗ u)_j ,   𝒩₂ = (u·∇)u ,   v = ρu.
#
# This reference works entirely by explicit DFT/IDFT (dependency-free, exact), mirroring the
# DirectSumBackend philosophy of the incompressible path; small grids only, correctness-first.
# ---------------------------------------------------------------------------

# Forward/backward direct DFT on the spatial dimensions of an (ns..., C) field. Convention matches
# NonlinearTerm.jl: forward = Σ_x f(x) e^{-i k·x}/N (analysis), backward = Σ_k f̂(k) e^{+i k·x}
# (synthesis), with x_d = (I_d-1)/n_d and integer wavenumber index k_d.
function _idft(field_hat, ns::NTuple{nd,Int}) where {nd}
    FT = real(eltype(field_hat))
    C  = size(field_hat, nd + 1)
    out = Array{complex(FT)}(undef, ns..., C)
    @inbounds for c in 1:C
        for xI in CartesianIndices(ns)
            acc = zero(complex(FT))
            for kI in CartesianIndices(ns)
                phase = zero(FT)
                for d in 1:nd
                    kidx = kI[d] - 1
                    km   = kidx <= ns[d] ÷ 2 ? kidx : kidx - ns[d]
                    phase += FT(2π) * km * FT(xI[d] - 1) / FT(ns[d])
                end
                acc += field_hat[kI, c] * cis(phase)
            end
            out[xI, c] = acc
        end
    end
    return out
end

function _dft(field_phys, ns::NTuple{nd,Int}) where {nd}
    FT = real(eltype(field_phys))
    C  = size(field_phys, nd + 1)
    out = Array{complex(FT)}(undef, ns..., C)
    Np  = prod(ns)
    @inbounds for c in 1:C
        for kI in CartesianIndices(ns)
            acc = zero(complex(FT))
            for xI in CartesianIndices(ns)
                phase = zero(FT)
                for d in 1:nd
                    kidx = kI[d] - 1
                    km   = kidx <= ns[d] ÷ 2 ? kidx : kidx - ns[d]
                    phase += FT(2π) * km * FT(xI[d] - 1) / FT(ns[d])
                end
                acc += field_phys[xI, c] * cis(-phase)
            end
            out[kI, c] = acc / FT(Np)
        end
    end
    return out
end

# Physical-space spatial gradient ∂f_c/∂x_d for every component c and direction d, from the
# spectral field: ∂_d f = IDFT(i k_d f̂). Returns an (ns..., C, nd) real-or-complex array.
function _grad_phys(field_hat, ks, ns::NTuple{nd,Int}) where {nd}
    FT = real(eltype(field_hat))
    C  = size(field_hat, nd + 1)
    g  = Array{complex(FT)}(undef, ns..., C, nd)
    @inbounds for d in 1:nd
        dfield_hat = similar(field_hat)
        for c in 1:C, kI in CartesianIndices(ns)
            dfield_hat[kI, c] = im * FT(ks[d][kI[d]]) * field_hat[kI, c]
        end
        gd = _idft(dfield_hat, ns)
        for c in 1:C, xI in CartesianIndices(ns)
            g[xI, c, d] = gd[xI, c]
        end
    end
    return g
end

# ---------------------------------------------------------------------------
# Helmholtz (rotational/compressive) split in Fourier space:
#   u_C(k) = [k·û(k)/|k|²] k   (compressive, ∥ k) ,   u_R = û − u_C   (rotational, ⊥ k).
# k = 0 has no compressive part. Dimension-generic; no external dependency.
# ---------------------------------------------------------------------------
function _helmholtz_split(field_hat, ks, ns::NTuple{nd,Int}) where {nd}
    FT   = real(eltype(field_hat))
    comp = zero(field_hat)
    rot  = similar(field_hat)
    @inbounds for kI in CartesianIndices(ns)
        k2 = zero(FT)
        for d in 1:nd
            k2 += FT(ks[d][kI[d]])^2
        end
        if k2 == 0
            for c in 1:nd
                comp[kI, c] = zero(eltype(field_hat))
                rot[kI, c]  = field_hat[kI, c]
            end
        else
            kdotu = zero(complex(FT))
            for c in 1:nd
                kdotu += FT(ks[c][kI[c]]) * field_hat[kI, c]
            end
            for c in 1:nd
                cc = (kdotu / k2) * FT(ks[c][kI[c]])
                comp[kI, c] = cc
                rot[kI, c]  = field_hat[kI, c] - cc
            end
        end
    end
    return rot, comp
end

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

"""
    calculate_compressible_flux(velocity_hat, density_hat, ks;
        binning=LinearBinning(...), pressure_hat=nothing,
        decompose=true, geometry=IsotropicShells()) -> CompressibleFluxResult

Compressible kinetic-energy spectral transfer `T_u(k)` and cumulative flux `Π(K)` for a flow with
Fourier-space velocity `velocity_hat` (shape `(ns..., D)`, `D = nd`) and density `density_hat`
(shape `(ns...)` or `(ns..., 1)`), following Singh–Tiwari–Sharma–Verma (2025) — momentum `v = ρu`,
`E_u(k) = ½Re[v(k)·u*(k)]`. The nonlinear transfer conserves total KE (`Σ_k T_u ≈ 0`); the KE↔internal
energy exchange is returned separately as the pressure-dilatation term when `pressure_hat` is given.

# Keyword Arguments
- `binning::AbstractShellBinning`: shell binning (default: linear at the minimum wavenumber spacing).
- `pressure_hat`: optional Fourier-space thermodynamic pressure `σ` (shape `(ns...)`/`(ns...,1)`).
  When supplied, `result.pressure_dilatation` holds `(rotational = Q_{I,R}(k), compressive = Q_{I,C}(k))`
  (paper Eqs. 38–39); otherwise it is `nothing`.
- `decompose::Bool=true`: also compute the Helmholtz rotational/compressive flux channels
  (`result.channels`, paper Eqs. 52–57). `false` skips them (`channels = nothing`).
- `geometry::AbstractShellGeometry`: shell geometry (default isotropic `|k|`).

# Returns
[`CompressibleFluxResult`](@ref). In the incompressible limit (`ρ` constant, `∇·u = 0`) `T_u` reduces
to `−ρ ×` the incompressible transfer spectrum, `channels.compressive`/cross vanish, and
`pressure_dilatation` vanishes — the regression anchors used in the tests.
"""
function calculate_compressible_flux(
    velocity_hat,
    density_hat,
    ks;
    binning::AbstractShellBinning = _default_binning(ks),
    pressure_hat = nothing,
    dealiasing::AbstractDealiasing = OrszagTwoThirds(),
    decompose::Bool = true,
    geometry::AbstractShellGeometry = IsotropicShells(),
)
    nd = length(ks)
    ns = size(velocity_hat)[1:nd]
    FT = real(eltype(velocity_hat))
    size(velocity_hat, nd + 1) == nd ||
        throw(ArgumentError("compressible transfer needs D = nd velocity components (got $(size(velocity_hat, nd+1)) for nd=$nd)."))

    # Orszag 2/3 dealiasing: truncate the quadratic-product inputs (velocity, density, and pressure)
    # to |k| < N/3 so aliased contributions land only in the discarded band |k| ≥ N/3, which is then
    # excluded from the shell sums (see `_bin`). Matches the incompressible NonlinearTerm convention.
    trunc = dealiasing isa OrszagTwoThirds
    velocity_hat = trunc ? _truncate_modes(velocity_hat, ns, nd) : velocity_hat
    ρ̂ = _as_scalar(density_hat, ns)
    ρ̂ = trunc ? _truncate_modes(ρ̂, ns, nd) : ρ̂

    # Physical fields
    u_phys = real.(_idft(velocity_hat, ns))                 # (ns..., nd)
    ρ_phys = real.(_idft(ρ̂, ns))[ntuple(_ -> Colon(), nd)..., 1]  # (ns...)
    v_phys = similar(u_phys)
    @inbounds for c in 1:nd, xI in CartesianIndices(ns)
        v_phys[xI, c] = ρ_phys[xI] * u_phys[xI, c]
    end
    v̂ = _dft(complex.(v_phys), ns)

    # Divergence ∇·u (physical) and gradients of u, v
    gradu = real.(_grad_phys(velocity_hat, ks, ns))         # (ns..., nd, nd) ∂u_c/∂x_d
    gradv = real.(_grad_phys(v̂, ks, ns))
    divu = zeros(FT, ns...)
    @inbounds for xI in CartesianIndices(ns), d in 1:nd
        divu[xI] += gradu[xI, d, d]
    end

    # 𝒩₁ = (u·∇)v + v(∇·u) ;  𝒩₂ = (u·∇)u
    N1_phys = zeros(FT, ns..., nd)
    N2_phys = zeros(FT, ns..., nd)
    @inbounds for c in 1:nd, xI in CartesianIndices(ns)
        adv_v = zero(FT); adv_u = zero(FT)
        for d in 1:nd
            adv_v += u_phys[xI, d] * gradv[xI, c, d]
            adv_u += u_phys[xI, d] * gradu[xI, c, d]
        end
        N1_phys[xI, c] = adv_v + v_phys[xI, c] * divu[xI]
        N2_phys[xI, c] = adv_u
    end
    N̂1 = _dft(complex.(N1_phys), ns)
    N̂2 = _dft(complex.(N2_phys), ns)

    # Per-mode net transfer T_u(k) = −½Re{û*·𝒩̂₁} − ½Re{v̂*·𝒩̂₂}
    td = zeros(FT, ns...)
    @inbounds for kI in CartesianIndices(ns)
        s = zero(FT)
        for c in 1:nd
            s += real(conj(velocity_hat[kI, c]) * N̂1[kI, c]) + real(conj(v̂[kI, c]) * N̂2[kI, c])
        end
        td[kI] = -FT(0.5) * s
    end

    # Shell binning
    k_mag   = shell_coordinate(geometry, ks)
    edges   = shell_edges(binning, maximum(k_mag))
    centers = collect(shell_centers(binning, maximum(k_mag)))
    sidx    = assign_shells(k_mag, edges)
    N_sh    = length(centers)

    T_spec = _bin(td, sidx, N_sh, FT, ns, trunc)
    flux   = _flux_from_transfer(T_spec)

    channels = decompose ? _rc_channels(velocity_hat, v̂, u_phys, v_phys, ρ_phys, divu, ks, ns, sidx, N_sh, FT, trunc) : nothing
    pdil = nothing
    if pressure_hat !== nothing
        σ̂ = _as_scalar(pressure_hat, ns)
        σ̂ = trunc ? _truncate_modes(σ̂, ns, nd) : σ̂
        pdil = _pressure_dilatation(velocity_hat, v̂, σ̂, ρ_phys, ks, ns, sidx, N_sh, FT, trunc)
    end

    return CompressibleFluxResult(centers, T_spec, flux, channels, pdil)
end

# ---------------------------------------------------------------------------
# Rotational/compressive flux channels (paper Eqs. 52–57).
# We form the transfer with the *receiver* field split into R/C (û_R*, v̂_R*, û_C*, v̂_C*) and the
# nonlinear term built from the R/C-filtered *giver* momentum. Each channel is shell-binned and
# accumulated into a flux; the four channels sum to the total flux (validated in tests), and in the
# incompressible limit only the rotational channel survives (paper Eqs. 48–50).
# ---------------------------------------------------------------------------
function _rc_channels(û, v̂, u_phys, v_phys, ρ_phys, divu, ks, ns::NTuple{nd,Int}, sidx, N_sh, FT, trunc) where {nd}
    ûR, ûC = _helmholtz_split(û, ks, ns)
    v̂R, v̂C = _helmholtz_split(v̂, ks, ns)
    uR = real.(_idft(ûR, ns));  uC = real.(_idft(ûC, ns))
    vR = real.(_idft(v̂R, ns));  vC = real.(_idft(v̂C, ns))

    # Transfer density with receiver β-part at k and giver α-part carried through the nonlinear term:
    #   T^{βα}(k) = −½ Re{ β̂*(k)·𝒩̂₁[α] } − ½ Re{ β̂_v*(k)·𝒩̂₂[α] }
    # where 𝒩₁[α] = (u·∇)v_α + v_α(∇·u), 𝒩₂[α] = (u·∇)u_α (full u advects/dilates the α-part).
    density(u_recv, v_recv, u_giv_phys, v_giv_phys) = begin
        gradv = real.(_grad_phys(_dft(complex.(v_giv_phys), ns), ks, ns))
        gradu = real.(_grad_phys(_dft(complex.(u_giv_phys), ns), ks, ns))
        N1 = zeros(FT, ns..., nd); N2 = zeros(FT, ns..., nd)
        @inbounds for c in 1:nd, xI in CartesianIndices(ns)
            av = zero(FT); au = zero(FT)
            for d in 1:nd
                av += u_phys[xI, d] * gradv[xI, c, d]
                au += u_phys[xI, d] * gradu[xI, c, d]
            end
            N1[xI, c] = av + v_giv_phys[xI, c] * divu[xI]
            N2[xI, c] = au
        end
        N̂1 = _dft(complex.(N1), ns); N̂2 = _dft(complex.(N2), ns)
        td = zeros(FT, ns...)
        @inbounds for kI in CartesianIndices(ns)
            s = zero(FT)
            for c in 1:nd
                s += real(conj(u_recv[kI, c]) * N̂1[kI, c]) + real(conj(v_recv[kI, c]) * N̂2[kI, c])
            end
            td[kI] = -FT(0.5) * s
        end
        td
    end

    Π(td) = _flux_from_transfer(_bin(td, sidx, N_sh, FT, ns, trunc))
    rr = Π(density(ûR, v̂R, uR, vR))   # R receiver, R giver
    cc = Π(density(ûC, v̂C, uC, vC))   # C receiver, C giver
    rc = Π(density(ûR, v̂R, uC, vC))   # C→R : R receiver, C giver
    cr = Π(density(ûC, v̂C, uR, vR))   # R→C : C receiver, R giver
    return (rotational = rr, compressive = cc, rot_to_comp = cr, comp_to_rot = rc)
end

# ---------------------------------------------------------------------------
# KE↔IE pressure-dilatation (paper Eqs. 38–39):
#   Q_{I,R}(k) = ½ Re[σ̃(k)·v_R*(k)]
#   Q_{I,C}(k) = ½ Re[σ̃(k)·v_C*(k)] − ½ Im[σ(k){k·u_C*(k)}]
# with σ̃ = ∇σ/ρ (specific pressure gradient). Shell-binned. Vanishes for incompressible div-free flow.
# ---------------------------------------------------------------------------
function _pressure_dilatation(û, v̂, σ̂, ρ_phys, ks, ns::NTuple{nd,Int}, sidx, N_sh, FT, trunc) where {nd}
    _, ûC = _helmholtz_split(û, ks, ns)
    v̂R, v̂C = _helmholtz_split(v̂, ks, ns)
    # σ̃ = ∇σ / ρ : physical gradient of σ divided by ρ, back to spectral.
    gradσ = _grad_phys(σ̂, ks, ns)                          # (ns..., 1, nd) complex
    σ̃_phys = Array{complex(FT)}(undef, ns..., nd)
    @inbounds for d in 1:nd, xI in CartesianIndices(ns)
        σ̃_phys[xI, d] = real(gradσ[xI, 1, d]) / ρ_phys[xI]
    end
    σ̃ = _dft(σ̃_phys, ns)

    QR = zeros(FT, ns...); QC = zeros(FT, ns...)
    @inbounds for kI in CartesianIndices(ns)
        qr = zero(FT); qc = zero(FT); kdotuC = zero(complex(FT))
        for c in 1:nd
            qr += real(σ̃[kI, c] * conj(v̂R[kI, c]))
            qc += real(σ̃[kI, c] * conj(v̂C[kI, c]))
            kdotuC += FT(ks[c][kI[c]]) * conj(ûC[kI, c])
        end
        QR[kI] = FT(0.5) * qr
        QC[kI] = FT(0.5) * qc - FT(0.5) * imag(σ̂[kI, 1] * kdotuC)
    end
    return (rotational = _bin(QR, sidx, N_sh, FT, ns, trunc), compressive = _bin(QC, sidx, N_sh, FT, ns, trunc))
end

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
_as_scalar(field, ns::NTuple{nd,Int}) where {nd} =
    ndims(field) == nd ? reshape(field, ns..., 1) : field

# Zero the 2/3-rule discard modes (|k| ≥ N/3 along any axis) of a spectral field — input truncation.
function _truncate_modes(field, ns::NTuple{nd,Int}, nd_::Int) where {nd}
    out = copy(field)
    C = size(field, nd + 1)
    @inbounds for I in CartesianIndices(ns)
        _is_dealiased(I, ns, nd) || continue
        for c in 1:C
            out[I, c] = zero(eltype(out))
        end
    end
    return out
end

# Shell-sum a per-mode density. With `dealias=true`, the 2/3 discard band (|k| ≥ N/3) is excluded so
# aliased contributions never enter the retained shells (Orszag 2/3 output zeroing).
function _bin(td, sidx, N_sh, FT, ns::NTuple{nd,Int}, dealias::Bool) where {nd}
    T = zeros(FT, N_sh)
    @inbounds for I in CartesianIndices(ns)
        n = sidx[I]; n == 0 && continue
        dealias && _is_dealiased(I, ns, nd) && continue
        T[n] += td[I]
    end
    return T
end

# Cumulative flux Π(K) = Σ_{k>K} T_u(k) (energy passing beyond shell K); with Σ_k T_u = 0 this equals
# −Σ_{k≤K} T_u(k). Returned as a per-shell vector aligned with the shell centers.
function _flux_from_transfer(T_spec)
    n = length(T_spec)
    Π = similar(T_spec)
    acc = zero(eltype(T_spec))
    tot = sum(T_spec)
    @inbounds for i in 1:n
        acc += T_spec[i]
        Π[i] = tot - acc          # Σ_{k>i}
    end
    return Π
end

function _default_binning(ks)
    min_dk = Inf
    for kv in ks, k in kv
        ak = abs(k); ak > 0 && (min_dk = min(min_dk, ak))
    end
    return LinearBinning(isfinite(min_dk) ? min_dk : 1.0)
end

end # module Compressible
