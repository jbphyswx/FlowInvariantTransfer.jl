module Compressible

using ..Types: CompressibleFluxResult, AbstractShellBinning, LinearBinning, AbstractShellGeometry,
               IsotropicShells, AbstractDealiasing, OrszagTwoThirds, NoDealiasing,
               AbstractSpectralBackend, DirectSumBackend, FFTBackend
using ..ShellBinning: shell_edges, shell_centers, assign_shells, shell_coordinate
using ..Utils: wavenumber_magnitude_grid
using ..NonlinearTerm: _is_dealiased

export calculate_compressible_flux, calculate_compressible_flux!, CompressibleWorkspace

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

# In-place explicit synthesis/analysis writing into `out` (no allocation) — for the workspace path.
function _idft!(out, field_hat, ns::NTuple{nd,Int}) where {nd}
    FT = real(eltype(field_hat))
    C  = size(field_hat, nd + 1)
    @inbounds for c in 1:C, xI in CartesianIndices(ns)
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
    return out
end

function _dft!(out, field_phys, ns::NTuple{nd,Int}) where {nd}
    FT = real(eltype(field_phys))
    C  = size(field_phys, nd + 1)
    Np = prod(ns)
    @inbounds for c in 1:C, kI in CartesianIndices(ns)
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
    return out
end

# In-place gradient (no dfield temp — folds i k_d into the synthesis sum) writing into g (ns...,C,nd).
function _grad_phys!(g, field_hat, ks, ns::NTuple{nd,Int}) where {nd}
    FT = real(eltype(field_hat))
    C  = size(field_hat, nd + 1)
    @inbounds for d in 1:nd, c in 1:C, xI in CartesianIndices(ns)
        acc = zero(complex(FT))
        for kI in CartesianIndices(ns)
            phase = zero(FT)
            for dd in 1:nd
                kidx = kI[dd] - 1
                km   = kidx <= ns[dd] ÷ 2 ? kidx : kidx - ns[dd]
                phase += FT(2π) * km * FT(xI[dd] - 1) / FT(ns[dd])
            end
            acc += im * FT(ks[d][kI[d]]) * field_hat[kI, c] * cis(phase)
        end
        g[xI, c, d] = acc
    end
    return g
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
# Transform context — swappable analysis/synthesis/gradient primitives so the physics assembly is
# written once and the transform algorithm is chosen by the spectral backend: the core provides the
# dependency-free explicit-DFT context (`DirectSumBackend`), and the FlowInvariantTransferFFTWExt
# extension provides the O(Nᵈ log Nᵈ) FFT context (`FFTBackend`), reusing preplanned FFTs + scratch.
#
#   tf.idft(field_hat)  : spectral (ns...,C) → physical (ns...,C) complex  (synthesis, u = Σ û e^{ik·x})
#   tf.dft(field_phys)  : physical (ns...,C) → spectral (ns...,C)          (analysis, û = fft/Nᵈ)
#   tf.grad(field_hat)  : spectral (ns...,C) → physical gradients (ns...,C,nd)
# ---------------------------------------------------------------------------
#   In-place siblings write into a caller buffer (for the workspace path): tf.idft!(out, fh),
#   tf.dft!(out, fp), tf.grad!(g, fh).
struct TransformContext{ID, DF, GR, IDB, DFB, GRB}
    idft::ID
    dft::DF
    grad::GR
    idft!::IDB
    dft!::DFB
    grad!::GRB
end

_directsum_tf(ks, ns) = TransformContext(
    fh -> _idft(fh, ns),
    fp -> _dft(fp, ns),
    fh -> _grad_phys(fh, ks, ns),
    (out, fh) -> _idft!(out, fh, ns),
    (out, fp) -> _dft!(out, fp, ns),
    (g, fh) -> _grad_phys!(g, fh, ks, ns),
)

# The FFT context is provided by FlowInvariantTransferFFTWExt; this fallback (less specific than the
# extension's method) gives a clear error when a non-DirectSum backend is requested without FFTW.
_fft_tf(velocity_hat, ks, ns) = throw(ArgumentError(
    "calculate_compressible_flux with an FFT backend requires `using FFTW`; " *
    "or pass `spectral = DirectSumBackend()` for the dependency-free (slow, small-grid) path."))

_resolve_tf(::DirectSumBackend, velocity_hat, ks, ns) = _directsum_tf(ks, ns)
_resolve_tf(::AbstractSpectralBackend, velocity_hat, ks, ns) = _fft_tf(velocity_hat, ks, ns)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

"""
    CompressibleWorkspace(velocity_hat, ks; spectral=FFTBackend())

Reusable field-scratch + transform context for [`calculate_compressible_flux!`](@ref) — holds the
FFT plans and every (ns...)-sized intermediate of the momentum-weighted budget (velocity/density/
momentum physical & spectral fields, gradients, nonlinear terms, and the R/C-channel + pressure-
dilatation scratch), so repeated per-snapshot calls allocate ~0 field memory (only the small shell
vectors of each result). Build once for a given grid/precision and reuse across snapshots.
"""
struct CompressibleWorkspace{TF, CA, CS, RA, RS, CG, RG}
    tf::TF
    vel::CA; ρh::CS; ûc::CA; ρc::CS; v̂::CA; N̂1::CA; N̂2::CA; cscr::CA; sscr::CA
    u_phys::RA; v_phys::RA; N1_phys::RA; N2_phys::RA
    ρ_phys::RS; divu::RS; td::RS
    graduc::CG; gradvc::CG; gradu::RG; gradv::RG
    ûR::CA; ûC::CA; v̂R::CA; v̂C::CA; uR::RA; uC::RA; vR::RA; vC::RA
    N̂1R::CA; N̂2R::CA; N̂1C::CA; N̂2C::CA
    σh::CS; gradσc::CG; σ̃phys::CA; σ̃::CA
end

function CompressibleWorkspace(velocity_hat, ks; spectral::AbstractSpectralBackend = FFTBackend())
    nd = length(ks); ns = size(velocity_hat)[1:nd]; FT = real(eltype(velocity_hat)); CT = complex(FT)
    ca()  = Array{CT}(undef, ns..., nd)
    cs()  = Array{CT}(undef, ns..., 1)
    ra()  = Array{FT}(undef, ns..., nd)
    rs()  = Array{FT}(undef, ns...)
    cg()  = Array{CT}(undef, ns..., nd, nd)
    rg()  = Array{FT}(undef, ns..., nd, nd)
    cg1() = Array{CT}(undef, ns..., 1, nd)
    return CompressibleWorkspace(
        _resolve_tf(spectral, velocity_hat, ks, ns),
        ca(), cs(), ca(), cs(), ca(), ca(), ca(), ca(), ca(),
        ra(), ra(), ra(), ra(),
        rs(), rs(), rs(),
        cg(), cg(), rg(), rg(),
        ca(), ca(), ca(), ca(), ra(), ra(), ra(), ra(),
        ca(), ca(), ca(), ca(),
        cs(), cg1(), ca(), ca())
end

"""
    calculate_compressible_flux(velocity_hat, density_hat, ks; binning, pressure_hat=nothing,
        decompose=true, geometry=IsotropicShells(), spectral=FFTBackend()) -> CompressibleFluxResult

Compressible kinetic-energy spectral transfer `T_u(k)` and cumulative flux `Π(K)` (Singh–Tiwari–
Sharma–Verma 2025): momentum `v = ρu`, `E_u(k) = ½Re[v·u*]`; the nonlinear transfer conserves total KE
(`Σ_k T_u ≈ 0`), and the KE↔internal-energy pressure-dilatation is returned separately when
`pressure_hat` is supplied. `decompose=true` also returns the Helmholtz rotational/compressive flux
channels. In the incompressible limit `T_u` reduces to `−ρ ×` the incompressible transfer spectrum.
This allocates a [`CompressibleWorkspace`](@ref) and delegates to [`calculate_compressible_flux!`](@ref);
build the workspace once and use the in-place form to loop over snapshots allocation-free.
"""
function calculate_compressible_flux(
    velocity_hat,
    density_hat,
    ks;
    spectral::AbstractSpectralBackend = FFTBackend(),
    kwargs...,
)
    nd = length(ks)
    size(velocity_hat, nd + 1) == nd ||
        throw(ArgumentError("compressible transfer needs D = nd velocity components (got $(size(velocity_hat, nd+1)) for nd=$nd)."))
    ws = CompressibleWorkspace(velocity_hat, ks; spectral=spectral)
    return calculate_compressible_flux!(ws, velocity_hat, density_hat, ks; kwargs...)
end

"""
    calculate_compressible_flux!(ws::CompressibleWorkspace, velocity_hat, density_hat, ks; kwargs...)

In-place momentum-weighted compressible transfer reusing `ws` (FFT plans + all field scratch). Returns
a fresh [`CompressibleFluxResult`](@ref) whose arrays are the small per-shell vectors; the O(field)
intermediates live in `ws` and are reused across calls (build the workspace once, loop over snapshots).
"""
function calculate_compressible_flux!(
    ws::CompressibleWorkspace,
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
    tf = ws.tf

    # Orszag 2/3 dealiasing: copy inputs into the workspace, zeroing the |k| ≥ N/3 discard band.
    trunc = dealiasing isa OrszagTwoThirds
    _copy_trunc!(ws.vel, velocity_hat, ns, nd, trunc)
    _copy_trunc!(ws.ρh, _as_scalar(density_hat, ns), ns, nd, trunc)

    # Physical fields  u = idft(û),  ρ = idft(ρ̂),  v = ρu,  v̂ = dft(v)
    tf.idft!(ws.ûc, ws.vel); @. ws.u_phys = real(ws.ûc)
    tf.idft!(ws.ρc, ws.ρh)
    @inbounds for xI in CartesianIndices(ns); ws.ρ_phys[xI] = real(ws.ρc[xI, 1]); end
    @inbounds for c in 1:nd, xI in CartesianIndices(ns); ws.v_phys[xI, c] = ws.ρ_phys[xI] * ws.u_phys[xI, c]; end
    @. ws.cscr = complex(ws.v_phys); tf.dft!(ws.v̂, ws.cscr)

    # Gradients of u, v and the divergence ∇·u
    tf.grad!(ws.graduc, ws.vel); @. ws.gradu = real(ws.graduc)
    tf.grad!(ws.gradvc, ws.v̂);   @. ws.gradv = real(ws.gradvc)
    fill!(ws.divu, zero(FT))
    @inbounds for xI in CartesianIndices(ns), d in 1:nd; ws.divu[xI] += ws.gradu[xI, d, d]; end

    # 𝒩₁ = (u·∇)v + v(∇·u) ;  𝒩₂ = (u·∇)u ;  then N̂₁,N̂₂
    _assemble_N!(ws.N1_phys, ws.N2_phys, ws.u_phys, ws.v_phys, ws.gradu, ws.gradv, ws.divu, ns, nd)
    @. ws.cscr = complex(ws.N1_phys); tf.dft!(ws.N̂1, ws.cscr)
    @. ws.cscr = complex(ws.N2_phys); tf.dft!(ws.N̂2, ws.cscr)

    # Per-mode net transfer T_u(k) = −½Re{û*·𝒩̂₁} − ½Re{v̂*·𝒩̂₂}
    @inbounds for kI in CartesianIndices(ns)
        s = zero(FT)
        for c in 1:nd
            s += real(conj(ws.vel[kI, c]) * ws.N̂1[kI, c]) + real(conj(ws.v̂[kI, c]) * ws.N̂2[kI, c])
        end
        ws.td[kI] = -FT(0.5) * s
    end

    # Shell binning
    k_mag   = shell_coordinate(geometry, ks)
    edges   = shell_edges(binning, maximum(k_mag))
    centers = collect(shell_centers(binning, maximum(k_mag)))
    sidx    = assign_shells(k_mag, edges)
    N_sh    = length(centers)

    T_spec = _bin(ws.td, sidx, N_sh, FT, ns, trunc)
    flux   = _flux_from_transfer(T_spec)

    channels = decompose ? _rc_channels!(ws, ks, ns, sidx, N_sh, FT, trunc) : nothing
    pdil = nothing
    if pressure_hat !== nothing
        _copy_trunc!(ws.σh, _as_scalar(pressure_hat, ns), ns, nd, trunc)
        pdil = _pressure_dilatation!(ws, ks, ns, sidx, N_sh, FT, trunc)
    end

    return CompressibleFluxResult(centers, T_spec, flux, channels, pdil)
end

# Copy `src` (ns...,C) into `dst`, zeroing the Orszag 2/3 discard band (|k| ≥ N/3 on any axis) if trunc.
function _copy_trunc!(dst, src, ns::NTuple{nd,Int}, nd_::Int, trunc::Bool) where {nd}
    C = size(src, nd + 1)
    @inbounds for c in 1:C, I in CartesianIndices(ns)
        dst[I, c] = (trunc && _is_dealiased(I, ns, nd)) ? zero(eltype(dst)) : src[I, c]
    end
    return dst
end

# 𝒩₁ = (u·∇)v + v(∇·u), 𝒩₂ = (u·∇)u  (physical), into preallocated N1,N2.
function _assemble_N!(N1, N2, u_phys, v_phys, gradu, gradv, divu, ns::NTuple{nd,Int}, nd_::Int) where {nd}
    # Accumulate in the concrete output element type. A `FT` passed as a runtime *value* (a `DataType`)
    # would make `zero(FT)` infer to `Any` and box every `+=` in the inner loop — `eltype(N1)` is concrete.
    RT = eltype(N1)
    @inbounds for c in 1:nd_, xI in CartesianIndices(ns)
        adv_v = zero(RT); adv_u = zero(RT)
        for d in 1:nd_
            adv_v += u_phys[xI, d] * gradv[xI, c, d]
            adv_u += u_phys[xI, d] * gradu[xI, c, d]
        end
        N1[xI, c] = adv_v + v_phys[xI, c] * divu[xI]
        N2[xI, c] = adv_u
    end
    return nothing
end

# ---------------------------------------------------------------------------
# Rotational/compressive flux channels (paper Eqs. 52–57).
# We form the transfer with the *receiver* field split into R/C (û_R*, v̂_R*, û_C*, v̂_C*) and the
# nonlinear term built from the R/C-filtered *giver* momentum. Each channel is shell-binned and
# accumulated into a flux; the four channels sum to the total flux (validated in tests), and in the
# incompressible limit only the rotational channel survives (paper Eqs. 48–50).
# ---------------------------------------------------------------------------
function _rc_channels!(ws, ks, ns::NTuple{nd,Int}, sidx, N_sh, FT, trunc) where {nd}
    tf = ws.tf
    _helmholtz_split!(ws.ûR, ws.ûC, ws.vel, ks, ns)
    _helmholtz_split!(ws.v̂R, ws.v̂C, ws.v̂, ks, ns)
    tf.idft!(ws.cscr, ws.ûR); @. ws.uR = real(ws.cscr)
    tf.idft!(ws.cscr, ws.ûC); @. ws.uC = real(ws.cscr)
    tf.idft!(ws.cscr, ws.v̂R); @. ws.vR = real(ws.cscr)
    tf.idft!(ws.cscr, ws.v̂C); @. ws.vC = real(ws.cscr)

    # 𝒩̂₁/𝒩̂₂ depend only on the giver (α) part — two distinct sets (α=R,C), reused across the four
    # channels; each reuses the (now-free) main gradient/nonlinear scratch (ws.gradu/gradv/N1_phys/…).
    giver_N!(N̂1_out, N̂2_out, u_giv, v_giv) = begin
        @. ws.cscr = complex(v_giv); tf.dft!(ws.sscr, ws.cscr); tf.grad!(ws.gradvc, ws.sscr); @. ws.gradv = real(ws.gradvc)
        @. ws.cscr = complex(u_giv); tf.dft!(ws.sscr, ws.cscr); tf.grad!(ws.graduc, ws.sscr); @. ws.gradu = real(ws.graduc)
        _assemble_N!(ws.N1_phys, ws.N2_phys, ws.u_phys, v_giv, ws.gradu, ws.gradv, ws.divu, ns, nd)
        @. ws.cscr = complex(ws.N1_phys); tf.dft!(N̂1_out, ws.cscr)
        @. ws.cscr = complex(ws.N2_phys); tf.dft!(N̂2_out, ws.cscr)
    end
    giver_N!(ws.N̂1R, ws.N̂2R, ws.uR, ws.vR)
    giver_N!(ws.N̂1C, ws.N̂2C, ws.uC, ws.vC)

    # Transfer density: receiver β-part at k, giver α-part carried through the nonlinear term
    #   T^{βα}(k) = −½ Re{ û_β*(k)·𝒩̂₁[α] } − ½ Re{ v̂_β*(k)·𝒩̂₂[α] }  (into the reused ws.td)
    Π(û_recv, v̂_recv, N̂1, N̂2) = begin
        @inbounds for kI in CartesianIndices(ns)
            s = zero(FT)
            for c in 1:nd
                s += real(conj(û_recv[kI, c]) * N̂1[kI, c]) + real(conj(v̂_recv[kI, c]) * N̂2[kI, c])
            end
            ws.td[kI] = -FT(0.5) * s
        end
        _flux_from_transfer(_bin(ws.td, sidx, N_sh, FT, ns, trunc))
    end
    rr = Π(ws.ûR, ws.v̂R, ws.N̂1R, ws.N̂2R)   # R receiver, R giver
    cc = Π(ws.ûC, ws.v̂C, ws.N̂1C, ws.N̂2C)   # C receiver, C giver
    rc = Π(ws.ûR, ws.v̂R, ws.N̂1C, ws.N̂2C)   # C→R : R receiver, C giver
    cr = Π(ws.ûC, ws.v̂C, ws.N̂1R, ws.N̂2R)   # R→C : C receiver, R giver
    return (rotational = rr, compressive = cc, rot_to_comp = cr, comp_to_rot = rc)
end

# In-place Helmholtz split (rot ⊥ k, comp ∥ k) writing into provided buffers.
function _helmholtz_split!(rot, comp, field_hat, ks, ns::NTuple{nd,Int}) where {nd}
    FT = real(eltype(field_hat))
    @inbounds for kI in CartesianIndices(ns)
        k2 = zero(FT)
        for d in 1:nd; k2 += FT(ks[d][kI[d]])^2; end
        if k2 == 0
            for c in 1:nd; comp[kI, c] = zero(eltype(comp)); rot[kI, c] = field_hat[kI, c]; end
        else
            kdotu = zero(complex(FT))
            for c in 1:nd; kdotu += FT(ks[c][kI[c]]) * field_hat[kI, c]; end
            for c in 1:nd
                cc = (kdotu / k2) * FT(ks[c][kI[c]])
                comp[kI, c] = cc
                rot[kI, c]  = field_hat[kI, c] - cc
            end
        end
    end
    return nothing
end

# ---------------------------------------------------------------------------
# KE↔IE pressure-dilatation (paper Eqs. 38–39):
#   Q_{I,R}(k) = ½ Re[σ̃(k)·v_R*(k)]
#   Q_{I,C}(k) = ½ Re[σ̃(k)·v_C*(k)] − ½ Im[σ(k){k·u_C*(k)}]
# with σ̃ = ∇σ/ρ (specific pressure gradient). Shell-binned. Vanishes for incompressible div-free flow.
# ---------------------------------------------------------------------------
function _pressure_dilatation!(ws, ks, ns::NTuple{nd,Int}, sidx, N_sh, FT, trunc) where {nd}
    tf = ws.tf
    _helmholtz_split!(ws.ûR, ws.ûC, ws.vel, ks, ns)       # ûC used below
    _helmholtz_split!(ws.v̂R, ws.v̂C, ws.v̂, ks, ns)         # ws.v̂ preserved (giver_N! uses ws.sscr)
    # σ̃ = ∇σ / ρ : physical gradient of σ divided by ρ, back to spectral.
    tf.grad!(ws.gradσc, ws.σh)                            # (ns..., 1, nd) complex
    @inbounds for d in 1:nd, xI in CartesianIndices(ns)
        ws.σ̃phys[xI, d] = real(ws.gradσc[xI, 1, d]) / ws.ρ_phys[xI]
    end
    tf.dft!(ws.σ̃, ws.σ̃phys)

    QR = ws.divu; QC = ws.td      # reuse main scratch (free after the main transfer + channels)
    @inbounds for kI in CartesianIndices(ns)
        qr = zero(FT); qc = zero(FT); kdotuC = zero(complex(FT))
        for c in 1:nd
            qr += real(ws.σ̃[kI, c] * conj(ws.v̂R[kI, c]))
            qc += real(ws.σ̃[kI, c] * conj(ws.v̂C[kI, c]))
            kdotuC += FT(ks[c][kI[c]]) * conj(ws.ûC[kI, c])
        end
        QR[kI] = FT(0.5) * qr
        QC[kI] = FT(0.5) * qc - FT(0.5) * imag(ws.σh[kI, 1] * kdotuC)
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
function _bin(td, sidx, N_sh, ::Type{FT}, ns::NTuple{nd,Int}, dealias::Bool) where {nd, FT}
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
