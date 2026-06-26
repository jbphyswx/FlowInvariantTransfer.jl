module NonlinearTerm

using LinearAlgebra: LinearAlgebra as LA
using ..Types: AbstractSpectralBackend, DirectSumBackend, FFTBackend
using ..Workspaces: NonlinearTermWorkspace

export compute_nonlinear_term, compute_nonlinear_term!
export _nonlinear_term_fft!   # stub overridden by FFTW extension

# ---------------------------------------------------------------------------
# Internal FFTW extension stub
# ---------------------------------------------------------------------------

"""
    _nonlinear_term_fft!(ws, velocity_hat, ks; dealiasing=true, advecting_hat=velocity_hat)

FFT-accelerated, allocation-free computation of N̂(k) = FFT[(u_adv·∇)u] into `ws.N̂`,
using the pre-planned transforms and scratch buffers in `ws.plans`.
This stub is overridden by the FFTW extension when FFTW is loaded.
"""
function _nonlinear_term_fft!(args...; kwargs...)
    throw(ArgumentError(
        "FFT-accelerated nonlinear term requires FFTW. Run `using FFTW` to load the extension."))
end

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

"""
    compute_nonlinear_term(velocity_hat, ks; dealiasing=true, backend=SerialBackend())

Compute N̂ᵢ(k) = F̂[(uⱼ ∂uᵢ/∂xⱼ)] for all components i via the pseudospectral
method.

# Arguments
- `velocity_hat`: Array of size `(ns..., D)` containing the complex Fourier
  coefficients of each velocity component.  `ns` is the D-dimensional grid shape
  and the last dimension indexes the D vector components.
- `ks`: Tuple of 1D wavenumber vectors (length D), one per spatial dimension.

# Keyword Arguments
- `dealiasing::Bool=true`: Apply the 2/3 dealiasing rule after computing products.
- `backend::AbstractExecutionBackend`: `SerialBackend()` (default) or `FFTBackend()` (requires FFTW extension).

# Returns
Array of the same size as `velocity_hat` containing N̂ᵢ(k).

# Notes
Without FFTW, this falls back to a pure Julia O(N²) direct-sum implementation
that is exact but slow.  Load FFTW to activate the O(N log N) path.
"""
function compute_nonlinear_term(
    velocity_hat,
    ks;
    dealiasing::Bool = true,
    spectral::AbstractSpectralBackend = DirectSumBackend(),
)
    ws = NonlinearTermWorkspace(velocity_hat, ks)
    compute_nonlinear_term!(ws, velocity_hat, ks; dealiasing=dealiasing, spectral=spectral)
    return ws.N̂
end

"""
    compute_nonlinear_term!(ws, velocity_hat, ks; dealiasing=true, backend=SerialBackend())

In-place version of `compute_nonlinear_term`. Writes result into `ws.N̂`.
Pass a `NonlinearTermWorkspace` to avoid any allocations in the hot path.
"""
function compute_nonlinear_term!(
    ws::NonlinearTermWorkspace,
    velocity_hat,
    ks;
    dealiasing::Bool = true,
    spectral::AbstractSpectralBackend = DirectSumBackend(),
    advecting_hat = velocity_hat,
)
    _compute_nonlinear_term!(ws, velocity_hat, ks, spectral; dealiasing=dealiasing, advecting_hat=advecting_hat)
    return ws.N̂
end

# Dispatch on the SPECTRAL (transform) backend — direct DFT vs FFT. `advecting_hat` is the velocity
# u_j that does the advecting; `velocity_hat` is the advected field whose gradient ∂_j(·)_i is taken:
# N_i = (u_adv)_j ∂_j (u)_i. They coincide for plain self-advection, and differ for shell-to-shell
# mediators ((u_m·∇)u) and scalar/MHD terms.
_compute_nonlinear_term!(ws, velocity_hat, ks, ::DirectSumBackend; dealiasing, advecting_hat=velocity_hat) =
    _compute_nonlinear_term_direct!(ws, velocity_hat, ks; dealiasing=dealiasing, advecting_hat=advecting_hat)

_compute_nonlinear_term!(ws, velocity_hat, ks, ::FFTBackend; dealiasing, advecting_hat=velocity_hat) =
    _nonlinear_term_fft!(ws, velocity_hat, ks; dealiasing=dealiasing, advecting_hat=advecting_hat)

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
    _compute_nonlinear_term_direct!(ws, velocity_hat, ks; dealiasing=true)

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
    dealiasing::Bool = true,
    advecting_hat = velocity_hat,
)
    nd  = length(ks)
    ns  = size(velocity_hat)[1:nd]
    D   = size(velocity_hat, nd+1)
    FT  = real(eltype(velocity_hat))
    Np  = prod(ns)
    phys_idxs = CartesianIndices(ns)

    fill!(ws.N̂, zero(eltype(ws.N̂)))

    # u_phys  shape: (ns..., D)
    # grad_phys shape: (ns..., D, nd)
    # N_phys  shape: (ns..., D)
    # Index via (phys_I..., comp) or (phys_I..., comp, grad_d)

    # --- (advecting) uⱼ(x_p) = IDFT(û_adv) ---
    for comp in 1:D
        û_c = selectdim(advecting_hat, nd+1, comp)
        for phys_I in phys_idxs
            val = zero(complex(FT))
            for spec_I in CartesianIndices(ns)
                dealiasing && _is_dealiased(spec_I, ns, nd) && continue  # truncate input
                phase = zero(FT)
                for d in 1:nd
                    xj    = FT(phys_I[d] - 1) / FT(ns[d])
                    kidx  = spec_I[d] - 1
                    km    = kidx <= ns[d] ÷ 2 ? kidx : kidx - ns[d]
                    phase += FT(2π) * km * xj
                end
                val += û_c[spec_I] * exp(im * phase)
            end
            ws.u_phys[phys_I, comp] = real(val / FT(Np))
        end
    end

    # --- ∂uᵢ/∂xⱼ(x_p) = IDFT(i·kⱼ·ûᵢ) ---
    for comp in 1:D
        û_c = selectdim(velocity_hat, nd+1, comp)
        for grad_d in 1:nd
            for phys_I in phys_idxs
                val = zero(complex(FT))
                for spec_I in CartesianIndices(ns)
                    dealiasing && _is_dealiased(spec_I, ns, nd) && continue  # truncate input
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

    # --- Nᵢ(x_p) = Σⱼ u_j · ∂uᵢ/∂xⱼ ---
    for comp in 1:D
        for phys_I in phys_idxs
            s = zero(FT)
            for j in 1:nd
                s += ws.u_phys[phys_I, j] * ws.grad_phys[phys_I, comp, j]
            end
            ws.N_phys[phys_I, comp] = s
        end
    end

    # --- N̂ᵢ(k) = DFT(Nᵢ) ---
    for comp in 1:D
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
    if dealiasing
        for I in CartesianIndices(ns)
            _is_dealiased(I, ns, nd) || continue
            for comp in 1:D
                ws.N̂[I, comp] = zero(eltype(ws.N̂))
            end
        end
    end

    return ws.N̂
end

end # module NonlinearTerm
