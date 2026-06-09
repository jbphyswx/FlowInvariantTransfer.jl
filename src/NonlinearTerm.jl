module NonlinearTerm

using LinearAlgebra: LinearAlgebra as LA
using ..Types: AbstractExecutionBackend, SerialBackend, FFTBackend
using ..Workspaces: NonlinearTermWorkspace

export compute_nonlinear_term, compute_nonlinear_term!
export _nonlinear_term_fft   # stub overridden by FFTW extension

# ---------------------------------------------------------------------------
# Internal FFTW extension stub
# ---------------------------------------------------------------------------

"""
    _nonlinear_term_fft(velocity_hat, ks; dealiasing=true)

FFT-accelerated computation of the nonlinear term N̂(k) = FFT[(u·∇)u].
This stub is overridden by the FFTW extension when FFTW is loaded.
"""
function _nonlinear_term_fft(args...; kwargs...)
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
    ks::Tuple;
    dealiasing::Bool = true,
    backend::AbstractExecutionBackend = SerialBackend(),
)
    ws = NonlinearTermWorkspace(velocity_hat, ks)
    compute_nonlinear_term!(ws, velocity_hat, ks; dealiasing=dealiasing, backend=backend)
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
    ks::Tuple;
    dealiasing::Bool = true,
    backend::AbstractExecutionBackend = SerialBackend(),
)
    _compute_nonlinear_term!(ws, velocity_hat, ks, backend; dealiasing=dealiasing)
    return ws.N̂
end

_compute_nonlinear_term!(ws, velocity_hat, ks, ::SerialBackend; dealiasing) =
    _compute_nonlinear_term_direct!(ws, velocity_hat, ks; dealiasing=dealiasing)

_compute_nonlinear_term!(ws, velocity_hat, ks, ::FFTBackend; dealiasing) =
    _nonlinear_term_fft(ws.N̂, velocity_hat, ks; dealiasing=dealiasing)

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
    ks::Tuple;
    dealiasing::Bool = true,
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

    # --- uᵢ(x_p) = IDFT(ûᵢ) ---
    for comp in 1:D
        û_c = selectdim(velocity_hat, nd+1, comp)
        for phys_I in phys_idxs
            val = zero(complex(FT))
            for spec_I in CartesianIndices(ns)
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

    # --- 2/3 dealiasing ---
    if dealiasing
        for I in CartesianIndices(ns)
            kill = false
            for d in 1:nd
                kidx  = I[d] - 1
                k_abs = kidx <= ns[d] ÷ 2 ? kidx : ns[d] - kidx
                k_abs >= ns[d] ÷ 3 && (kill = true; break)
            end
            if kill
                for comp in 1:D
                    ws.N̂[I, comp] = zero(eltype(ws.N̂))
                end
            end
        end
    end

    return ws.N̂
end

end # module NonlinearTerm
