module NonlinearTerm

using LinearAlgebra: LinearAlgebra as LA
using ..Types: AbstractSpectralBackend, DirectSumBackend, FFTBackend,
               AbstractDealiasing, NoDealiasing, OrszagTwoThirds, PaddedThreeHalves
using ..Workspaces: NonlinearTermWorkspace

export compute_nonlinear_term, compute_nonlinear_term!
export _nonlinear_term_fft!, _nonlinear_term_padded_fft!   # stubs overridden by FFTW extension

# ---------------------------------------------------------------------------
# Internal FFTW extension stub
# ---------------------------------------------------------------------------

"""
    _nonlinear_term_fft!(ws, velocity_hat, ks; truncate=true, advecting_hat=velocity_hat)

FFT-accelerated, allocation-free computation of N̂(k) = FFT[(u_adv·∇)u] into `ws.N̂`,
using the pre-planned transforms and scratch buffers in `ws.plans`.
This stub is overridden by the FFTW extension when FFTW is loaded.
"""
function _nonlinear_term_fft!(args...; kwargs...)
    throw(ArgumentError(
        "FFT-accelerated nonlinear term requires FFTW. Run `using FFTW` to load the extension."))
end

"""
    _nonlinear_term_padded_fft!(ws, advected_hat, ks; advecting_hat=advected_hat)

Exact 3/2 zero-padded pseudospectral nonlinear term written into `ws.N̂`. Overridden by the FFTW
extension; the stub errors when FFTW is not loaded.
"""
function _nonlinear_term_padded_fft!(args...; kwargs...)
    throw(ArgumentError(
        "PaddedThreeHalves dealiasing requires FFTW. Run `using FFTW` to load the extension."))
end

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

"""
    compute_nonlinear_term(advected_hat, ks; dealiasing=OrszagTwoThirds(),
                           spectral=DirectSumBackend(), advecting_hat=advected_hat)

Compute the pseudospectral nonlinear term `𝒩̂ᵢ(k) = F̂[(uⱼ ∂fᵢ/∂xⱼ)]` — the advection of an
`M`-component field `f` (`advected_hat`) by a velocity `u` (`advecting_hat`). For the momentum
self-advection term pass the velocity as both (the default), giving `N̂ᵢ = F̂[(u·∇)uᵢ]`.

# Arguments
- `advected_hat`: Array of size `(ns..., M)` — Fourier coefficients of the advected field `f`
  (`M = D` for momentum, `M = 1` for a passive scalar / vector potential).
- `ks`: Tuple of 1D wavenumber vectors (length `nd`), one per spatial dimension.

# Keyword Arguments
- `dealiasing::AbstractDealiasing=OrszagTwoThirds()`: dealiasing strategy (NoDealiasing / OrszagTwoThirds / PaddedThreeHalves).
- `spectral::AbstractSpectralBackend`: `DirectSumBackend()` (default, no deps) or `FFTBackend()`
  (requires the FFTW extension) for the O(N log N) path.
- `advecting_hat`: the advecting velocity `u` (shape `(ns..., D)`, `D ≥ nd`); defaults to
  `advected_hat` (self-advection). Only the `nd` spatial components participate in `(u·∇)`.

# Returns
Array of size `(ns..., M)` containing `𝒩̂ᵢ(k)`.
"""
function compute_nonlinear_term(
    velocity_hat,
    ks;
    dealiasing::AbstractDealiasing = OrszagTwoThirds(),
    spectral::AbstractSpectralBackend = DirectSumBackend(),
    advecting_hat = velocity_hat,
)
    ws = NonlinearTermWorkspace(velocity_hat, ks)
    compute_nonlinear_term!(ws, velocity_hat, ks;
        dealiasing=dealiasing, spectral=spectral, advecting_hat=advecting_hat)
    return ws.N̂
end

"""
    compute_nonlinear_term!(ws, advected_hat, ks; dealiasing=OrszagTwoThirds(),
                            spectral=DirectSumBackend(), advecting_hat=advected_hat)

In-place version of `compute_nonlinear_term`. Writes result into `ws.N̂`.
Pass a `NonlinearTermWorkspace` (sized for `advected_hat`) to avoid any allocations in the hot path.
"""
function compute_nonlinear_term!(
    ws::NonlinearTermWorkspace,
    velocity_hat,
    ks;
    dealiasing::AbstractDealiasing = OrszagTwoThirds(),
    spectral::AbstractSpectralBackend = DirectSumBackend(),
    advecting_hat = velocity_hat,
)
    _compute_nonlinear_term!(ws, velocity_hat, ks, spectral, dealiasing; advecting_hat=advecting_hat)
    return ws.N̂
end

# Dispatch on (spectral transform backend, dealiasing strategy). `advecting_hat` is the velocity
# u_j that does the advecting; `velocity_hat` is the advected field whose gradient ∂_j(·)_i is taken:
# N_i = (u_adv)_j ∂_j (u)_i. They coincide for plain self-advection, and differ for shell-to-shell
# mediators ((u_m·∇)u) and scalar/MHD terms. The 2/3 and no-dealias paths share one implementation
# (a `keep`/truncate flag); the exact 3/2-padding path is a separate, FFT-only routine.
_compute_nonlinear_term!(ws, velocity_hat, ks, ::DirectSumBackend, ::OrszagTwoThirds; advecting_hat=velocity_hat) =
    _compute_nonlinear_term_direct!(ws, velocity_hat, ks; truncate=true, advecting_hat=advecting_hat)
_compute_nonlinear_term!(ws, velocity_hat, ks, ::DirectSumBackend, ::NoDealiasing; advecting_hat=velocity_hat) =
    _compute_nonlinear_term_direct!(ws, velocity_hat, ks; truncate=false, advecting_hat=advecting_hat)
_compute_nonlinear_term!(ws, velocity_hat, ks, ::DirectSumBackend, ::PaddedThreeHalves; advecting_hat=velocity_hat) =
    throw(ArgumentError(
        "PaddedThreeHalves dealiasing requires the FFT path — pass spectral=FFTBackend() (and `using FFTW`). " *
        "The dependency-free DirectSumBackend supports only NoDealiasing/OrszagTwoThirds."))

_compute_nonlinear_term!(ws, velocity_hat, ks, ::FFTBackend, ::OrszagTwoThirds; advecting_hat=velocity_hat) =
    _nonlinear_term_fft!(ws, velocity_hat, ks; truncate=true, advecting_hat=advecting_hat)
_compute_nonlinear_term!(ws, velocity_hat, ks, ::FFTBackend, ::NoDealiasing; advecting_hat=velocity_hat) =
    _nonlinear_term_fft!(ws, velocity_hat, ks; truncate=false, advecting_hat=advecting_hat)
_compute_nonlinear_term!(ws, velocity_hat, ks, ::FFTBackend, ::PaddedThreeHalves; advecting_hat=velocity_hat) =
    _nonlinear_term_padded_fft!(ws, velocity_hat, ks; advecting_hat=advecting_hat)

# ---------------------------------------------------------------------------
# 2/3 dealiasing predicate (shared by input-truncation and output-zeroing)
# ---------------------------------------------------------------------------
#
# Orszag's 2/3 rule must truncate the *inputs* of the quadratic product, not only the
# output: a product of two retained modes p,q with p+q wrapping past Nyquist aliases
# back onto a *low* mode (e.g. p=q=N/2−1 → k≈−2), so output-only truncation leaves the
# retained band |k|<N/3 contaminated. We therefore (a) skip dealiased input modes when
# building u and ∇u, and (b) still zero the output above the cutoff for a clean N̂.

"""
    _is_dealiased(I, ns, nd) -> Bool

`true` if Fourier mode `I` lies in the 2/3-rule discard zone (|k_d| ≥ N_d/3 along any
dimension `d`), in FFTW index order.
"""
@inline function _is_dealiased(I::CartesianIndex, ns, nd::Int)
    @inbounds for d in 1:nd
        idx0  = I[d] - 1
        k_abs = idx0 <= ns[d] ÷ 2 ? idx0 : ns[d] - idx0
        k_abs >= ns[d] ÷ 3 && return true
    end
    return false
end

# ---------------------------------------------------------------------------
# Direct-sum reference implementation
# ---------------------------------------------------------------------------

"""
    _compute_nonlinear_term_direct!(ws, velocity_hat, ks; truncate=true)

Reference O(N²) direct-sum computation of the nonlinear advection term.
Uses pre-allocated buffers from `ws::NonlinearTermWorkspace` — no heap allocation.

Algorithm (pseudospectral, direct DFT/IDFT):
  1. uᵢ(x)        = IDFT(ûᵢ) via explicit sum
  2. ∂uᵢ/∂xⱼ(x)  = IDFT(i·kⱼ·ûᵢ) via explicit sum
  3. Nᵢ(x)        = Σⱼ u_j(x) · ∂uᵢ/∂xⱼ(x)
  4. N̂ᵢ(k)        = DFT(Nᵢ(x)) via explicit sum
  5. Apply 2/3 dealiasing.
"""
function _compute_nonlinear_term_direct!(
    ws::NonlinearTermWorkspace,
    velocity_hat,
    ks;
    truncate::Bool = true,   # apply the Orszag 2/3 input/output truncation
    advecting_hat = velocity_hat,
)
    nd  = length(ks)
    ns  = size(velocity_hat)[1:nd]
    M   = size(velocity_hat, nd+1)   # advected-field components (D for momentum, 1 for scalar)
    FT  = real(eltype(velocity_hat))
    Np  = prod(ns)
    phys_idxs = CartesianIndices(ns)

    fill!(ws.N̂, zero(eltype(ws.N̂)))

    # u_phys  shape: (ns..., nd)   — advecting velocity, spatial directions only
    # grad_phys shape: (ns..., M, nd)
    # N_phys  shape: (ns..., M)
    # Index via (phys_I..., comp) or (phys_I..., comp, grad_d)

    # --- (advecting) uⱼ(x_p) = IDFT(û_adv), j = 1:nd (only the advecting directions) ---
    for j in 1:nd
        û_j = selectdim(advecting_hat, nd+1, j)
        for phys_I in phys_idxs
            val = zero(complex(FT))
            for spec_I in CartesianIndices(ns)
                truncate && _is_dealiased(spec_I, ns, nd) && continue  # truncate input
                phase = zero(FT)
                for d in 1:nd
                    xj    = FT(phys_I[d] - 1) / FT(ns[d])
                    kidx  = spec_I[d] - 1
                    km    = kidx <= ns[d] ÷ 2 ? kidx : kidx - ns[d]
                    phase += FT(2π) * km * xj
                end
                val += û_j[spec_I] * exp(im * phase)
            end
            ws.u_phys[phys_I, j] = real(val / FT(Np))
        end
    end

    # --- ∂fᵢ/∂xⱼ(x_p) = IDFT(i·kⱼ·f̂ᵢ),  i = 1:M advected components ---
    for comp in 1:M
        û_c = selectdim(velocity_hat, nd+1, comp)
        for grad_d in 1:nd
            for phys_I in phys_idxs
                val = zero(complex(FT))
                for spec_I in CartesianIndices(ns)
                    truncate && _is_dealiased(spec_I, ns, nd) && continue  # truncate input
                    kphys = ks[grad_d][spec_I[grad_d]]
                    phase = zero(FT)
                    for d in 1:nd
                        xj   = FT(phys_I[d] - 1) / FT(ns[d])
                        kidx = spec_I[d] - 1
                        km   = kidx <= ns[d] ÷ 2 ? kidx : kidx - ns[d]
                        phase += FT(2π) * km * xj
                    end
                    val += (im * kphys) * û_c[spec_I] * exp(im * phase)
                end
                ws.grad_phys[phys_I, comp, grad_d] = real(val / FT(Np))
            end
        end
    end

    # --- 𝒩ᵢ(x_p) = Σⱼ u_j · ∂fᵢ/∂xⱼ ---
    for comp in 1:M
        for phys_I in phys_idxs
            s = zero(FT)
            for j in 1:nd
                s += ws.u_phys[phys_I, j] * ws.grad_phys[phys_I, comp, j]
            end
            ws.N_phys[phys_I, comp] = s
        end
    end

    # --- 𝒩̂ᵢ(k) = DFT(𝒩ᵢ) ---
    for comp in 1:M
        N̂_c = selectdim(ws.N̂, nd+1, comp)
        for spec_I in CartesianIndices(ns)
            val = zero(complex(FT))
            for phys_I in phys_idxs
                phase = zero(FT)
                for d in 1:nd
                    xj   = FT(phys_I[d] - 1) / FT(ns[d])
                    kidx = spec_I[d] - 1
                    km   = kidx <= ns[d] ÷ 2 ? kidx : kidx - ns[d]
                    phase += FT(2π) * km * xj
                end
                val += ws.N_phys[phys_I, comp] * exp(-im * phase)
            end
            N̂_c[spec_I] = val / FT(Np)
        end
    end

    # --- zero output above the 2/3 cutoff (inputs already truncated above) ---
    if truncate
        for I in CartesianIndices(ns)
            _is_dealiased(I, ns, nd) || continue
            for comp in 1:M
                ws.N̂[I, comp] = zero(eltype(ws.N̂))
            end
        end
    end

    return ws.N̂
end

end # module NonlinearTerm
