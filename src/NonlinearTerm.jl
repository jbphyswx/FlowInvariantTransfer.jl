module NonlinearTerm

using ..Types: Types
using ..SpectralLayout: SpectralLayout
using ..Workspaces: Workspaces
using ComputationalBackends: ComputationalBackends
using SpectralBackends: SpectralBackends

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
        "Types.PaddedThreeHalves dealiasing requires FFTW. Run `using FFTW` to load the extension."))
end

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

"""
    compute_nonlinear_term(advected_hat, ks; dealiasing=OrszagTwoThirds(),
                           spectral=SpectralBackends.DirectSumSpectralBackend(), advecting_hat=advected_hat)

Compute the pseudospectral nonlinear term `𝒩̂ᵢ(k) = F̂[(uⱼ ∂fᵢ/∂xⱼ)]` — the advection of an
`M`-component field `f` (`advected_hat`) by a velocity `u` (`advecting_hat`). For the momentum
self-advection term pass the velocity as both (the default), giving `N̂ᵢ = F̂[(u·∇)uᵢ]`.

# Arguments
- `advected_hat`: Array of size `(ns..., M)` — Fourier coefficients of the advected field `f`
  (`M = D` for momentum, `M = 1` for a passive scalar / vector potential).
- `ks`: Tuple of 1D wavenumber vectors (length `nd`), one per spatial dimension.

# Keyword Arguments
- `dealiasing::AbstractDealiasing=OrszagTwoThirds()`: dealiasing strategy (NoDealiasing / OrszagTwoThirds / PaddedThreeHalves).
- `spectral::SpectralBackends.AbstractSpectralBackend`: `SpectralBackends.DirectSumSpectralBackend()` (default, no deps) or `SpectralBackends.FFTSpectralBackend()`
  (requires the FFTW extension) for the O(N log N) path.
- `advecting_hat`: the advecting velocity `u` (shape `(ns..., D)`, `D ≥ nd`); defaults to
  `advected_hat` (self-advection). Only the `nd` spatial components participate in `(u·∇)`.

# Returns
Array of size `(ns..., M)` containing `𝒩̂ᵢ(k)`.
"""
function compute_nonlinear_term(
    velocity_hat,
    ks;
    dealiasing::Types.AbstractDealiasing = Types.OrszagTwoThirds(),
    spectral::SpectralBackends.AbstractSpectralBackend = SpectralBackends.AutoSpectralBackend(),
    advecting_hat = velocity_hat,
)
    ws = Workspaces.NonlinearTermWorkspace(velocity_hat, ks; dealiasing=dealiasing)
    compute_nonlinear_term!(ws, velocity_hat, ks;
        dealiasing=dealiasing, spectral=spectral, advecting_hat=advecting_hat)
    return ws.N̂
end

"""
    compute_nonlinear_term!(ws, advected_hat, ks; dealiasing=OrszagTwoThirds(),
                            spectral=SpectralBackends.DirectSumSpectralBackend(), advecting_hat=advected_hat)

In-place version of `compute_nonlinear_term`. Writes result into `ws.N̂`.
Pass a `NonlinearTermWorkspace` (sized for `advected_hat`) to avoid any allocations in the hot path.
"""
function compute_nonlinear_term!(
    ws::Workspaces.NonlinearTermWorkspace,
    velocity_hat,
    ks;
    dealiasing::Types.AbstractDealiasing = Types.OrszagTwoThirds(),
    spectral::SpectralBackends.AbstractSpectralBackend = SpectralBackends.AutoSpectralBackend(),
    advecting_hat = velocity_hat,
)
    # DirectSum builds the nonlinear term with scalar-indexed direct sums (a host O(N²ᴰ) reference); it
    # cannot run on a device array (scalar indexing errors under `allowscalar(false)`). Raise a clear
    # error directing to the device path rather than a cryptic scalar-indexing crash. `ComputationalBackends.is_gpu_array`
    # (the `AbstractGPUArray` trait) — NOT `!(x isa Array)`, which would misflag host non-`Array` types
    # (FixedSizeArray/StaticArray/SubArray/…); `ComputationalBackends.GPUBackend(KA.CPU())`'s host-`Array` proxy stays on the path.
    spectral = Types.resolve_spectral(spectral)
    if spectral isa SpectralBackends.DirectSumSpectralBackend && ComputationalBackends.is_gpu_array(velocity_hat)
        throw(ArgumentError(
            "SpectralBackends.DirectSumSpectralBackend uses scalar-indexed direct sums (a host O(N²ᴰ) reference) and cannot run on " *
            "device arrays; use `spectral = SpectralBackends.FFTSpectralBackend()` (cuFFT via AbstractFFTs) for the device path."))
    end
    _compute_nonlinear_term!(ws, velocity_hat, ks, spectral, dealiasing; advecting_hat=advecting_hat)
    return ws.N̂
end

# Dispatch on (spectral transform backend, dealiasing strategy). `advecting_hat` is the velocity
# u_j that does the advecting; `velocity_hat` is the advected field whose gradient ∂_j(·)_i is taken:
# N_i = (u_adv)_j ∂_j (u)_i. They coincide for plain self-advection, and differ for shell-to-shell
# mediators ((u_m·∇)u) and scalar/MHD terms. The 2/3 and no-dealias paths share one implementation
# (a `keep`/truncate flag); the exact 3/2-padding path is a separate, FFT-only routine.
_compute_nonlinear_term!(ws, velocity_hat, ks, ::SpectralBackends.DirectSumSpectralBackend, ::Types.OrszagTwoThirds; advecting_hat=velocity_hat) =
    _compute_nonlinear_term_direct!(ws, velocity_hat, ks; truncate=true, advecting_hat=advecting_hat)
_compute_nonlinear_term!(ws, velocity_hat, ks, ::SpectralBackends.DirectSumSpectralBackend, ::Types.NoDealiasing; advecting_hat=velocity_hat) =
    _compute_nonlinear_term_direct!(ws, velocity_hat, ks; truncate=false, advecting_hat=advecting_hat)
_compute_nonlinear_term!(ws, velocity_hat, ks, ::SpectralBackends.DirectSumSpectralBackend, ::Types.PaddedThreeHalves; advecting_hat=velocity_hat) =
    throw(ArgumentError(
        "Types.PaddedThreeHalves dealiasing requires the FFT path — pass spectral=SpectralBackends.FFTSpectralBackend() (and `using FFTW`). " *
        "The dependency-free SpectralBackends.DirectSumSpectralBackend supports only Types.NoDealiasing/Types.OrszagTwoThirds."))

_compute_nonlinear_term!(ws, velocity_hat, ks, ::SpectralBackends.FFTSpectralBackend, ::Types.OrszagTwoThirds; advecting_hat=velocity_hat) =
    _nonlinear_term_fft!(ws, velocity_hat, ks; truncate=true, advecting_hat=advecting_hat)
_compute_nonlinear_term!(ws, velocity_hat, ks, ::SpectralBackends.FFTSpectralBackend, ::Types.NoDealiasing; advecting_hat=velocity_hat) =
    _nonlinear_term_fft!(ws, velocity_hat, ks; truncate=false, advecting_hat=advecting_hat)
_compute_nonlinear_term!(ws, velocity_hat, ks, ::SpectralBackends.FFTSpectralBackend, ::Types.PaddedThreeHalves; advecting_hat=velocity_hat) =
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

# ---------------------------------------------------------------------------
# Individual legs of the nonlinear term.
#
# `𝒩 = Σ_j u_j ∂_j f` factors into: synthesise the advecting velocity, accumulate the advection,
# transform forward. A caller whose advected field is ONE Fourier mode (mode-to-mode's giver) knows
# `∂_j f` analytically — it is a plane wave — and needs only the first and third legs. The advecting
# velocity is the same for every giver, so it is synthesised once for the whole loop.
# ---------------------------------------------------------------------------

_nlt_synth_advecting_fft!(args...) = throw(ArgumentError(
    "FFT-accelerated nonlinear term requires FFTW. Run `using FFTW` to load the extension."))
_nlt_forward_fft!(args...) = throw(ArgumentError(
    "FFT-accelerated nonlinear term requires FFTW. Run `using FFTW` to load the extension."))

"""
    nlt_synth_advecting!(ws, advecting_hat, ks, spectral; truncate=true) -> ws.u_phys

Physical advecting velocity `u_j(x) = Σ_k û_j e^{ik·x}` into `ws.u_phys` — the first leg of the
nonlinear term on its own, for drivers that hold the advecting field fixed across an outer loop.
"""
nlt_synth_advecting!(ws, advecting_hat, ks, ::SpectralBackends.FFTSpectralBackend; truncate::Bool = true) =
    _nlt_synth_advecting_fft!(ws, advecting_hat, ks, truncate)

function nlt_synth_advecting!(ws, advecting_hat, ks, ::SpectralBackends.DirectSumSpectralBackend;
                              truncate::Bool = true)
    nd = length(ks); ns = SpectralLayout.full_size(ks)
    for j in 1:nd
        û_j = selectdim(advecting_hat, nd + 1, j)
        for phys_I in CartesianIndices(ns)
            ws.u_phys[phys_I, j] = _direct_synth(û_j, ks, phys_I, truncate, I -> true)
        end
    end
    return ws.u_phys
end

"""
    nlt_forward!(ws, ks, spectral; truncate=true) -> ws.N̂

`𝒩̂ = DFT(ws.N_phys)/Nᵈ`, zeroed above the dealiasing cutoff — the last leg on its own.
"""
nlt_forward!(ws, ks, ::SpectralBackends.FFTSpectralBackend; truncate::Bool = true) =
    _nlt_forward_fft!(ws, ks, truncate)

function nlt_forward!(ws, ks, ::SpectralBackends.DirectSumSpectralBackend; truncate::Bool = true)
    nd = length(ks)
    ns = SpectralLayout.full_size(ks); ms = SpectralLayout.spectral_size(ks)
    M  = size(ws.N̂, nd + 1); FT = real(eltype(ws.N̂))
    Np = prod(ns)
    for comp in 1:M
        N̂_c = selectdim(ws.N̂, nd + 1, comp)
        Nc  = selectdim(ws.N_phys, nd + 1, comp)
        for spec_I in CartesianIndices(ms)
            if truncate && _is_dealiased(ks, spec_I)
                N̂_c[spec_I] = zero(eltype(ws.N̂))
                continue
            end
            val = zero(complex(FT))
            @inbounds for phys_I in CartesianIndices(ns)
                phase = zero(FT)
                for d in 1:nd
                    phase += FT(2π) * SpectralLayout.axis_index_wavenumber(ks[d], spec_I[d]) *
                             FT(phys_I[d] - 1) / FT(ns[d])
                end
                val += Nc[phys_I] * exp(-im * phase)
            end
            N̂_c[spec_I] = val / FT(Np)
        end
    end
    return ws.N̂
end

# Synthesis of one component at one physical point, with `kfac` multiplying each mode's coefficient.
@inline function _direct_synth(û_c, ks, phys_I, truncate::Bool, kfac)
    nd = length(ks); ns = SpectralLayout.full_size(ks); ms = SpectralLayout.spectral_size(ks)
    FT = real(eltype(û_c))
    val = zero(complex(FT))
    @inbounds for spec_I in CartesianIndices(ms)
        truncate && _is_dealiased(ks, spec_I) && continue
        phase = zero(FT)
        for d in 1:nd
            phase += FT(2π) * SpectralLayout.axis_index_wavenumber(ks[d], spec_I[d]) *
                     FT(phys_I[d] - 1) / FT(ns[d])
        end
        val += SpectralLayout.hermitian_weight(ks, spec_I) * kfac(spec_I) * û_c[spec_I] * exp(im * phase)
    end
    return real(val)
end

"""
    _is_dealiased(ks, I) -> Bool

`true` if Fourier mode `I` lies in the 2/3-rule discard zone (|k_d| ≥ N_d/3 along any dimension `d`).
Reads the wavenumber from `ks`, so it holds for both spectral layouts.
"""
@inline _is_dealiased(ks::Tuple, I::CartesianIndex) = SpectralLayout.is_dealiased(ks, I)

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

Normalization: the package convention is `û = fft(u)/Nᵈ` (so `E(k)=½|û|²` is the physical modal
energy). The physical synthesis (IDFT) is therefore `u(x) = Σ_k û(k) e^{ik·x}` — an *unnormalized*
inverse (no `/Nᵈ`); the forward analysis (DFT) is `û(k) = (1/Nᵈ) Σ_x u(x) e^{-ik·x}`. This makes
`T(k)=Re{û*·N̂}` the physically-scaled transfer (verified against an independent FFTW ground truth).
"""
function _compute_nonlinear_term_direct!(
    ws::Workspaces.NonlinearTermWorkspace,
    velocity_hat,
    ks;
    truncate::Bool = true,   # apply the Orszag 2/3 input/output truncation
    advecting_hat = velocity_hat,
)
    nd  = length(ks)
    ns  = SpectralLayout.full_size(ks)        # physical grid
    ms  = SpectralLayout.spectral_size(ks)    # coefficient grid
    M   = size(velocity_hat, nd+1)   # advected-field components (D for momentum, 1 for scalar)
    FT  = real(eltype(velocity_hat))
    Np  = prod(ns)
    phys_idxs = CartesianIndices(ns)
    spec_idxs = CartesianIndices(ms)

    # Synthesis over the stored coefficients. On the full layout every mode is stored and the weight
    # is 1; on the half layout the unstored conjugate mode contributes the conjugate term, which is
    # exactly the Hermitian weight (1 on the self-paired k₁ = 0 / Nyquist planes, 2 elsewhere) — so
    # `real(Σ_stored w·û·e^{ikx})` equals `Σ_full û·e^{ikx}` in both cases.
    @inline function _synth(f, phys_I)
        val = zero(complex(FT))
        @inbounds for spec_I in spec_idxs
            truncate && _is_dealiased(ks, spec_I) && continue      # truncate input
            phase = zero(FT)
            for d in 1:nd
                phase += FT(2π) * SpectralLayout.axis_index_wavenumber(ks[d], spec_I[d]) *
                         FT(phys_I[d] - 1) / FT(ns[d])
            end
            val += SpectralLayout.hermitian_weight(ks, spec_I) * f(spec_I) * exp(im * phase)
        end
        return real(val)
    end

    # --- (advecting) uⱼ(x_p) = IDFT(û_adv), j = 1:nd (only the advecting directions) ---
    # Synthesis is unnormalized (physical u = Σ_k û e^{ik·x}); û already carries the 1/Nᵈ.
    for j in 1:nd
        û_j = selectdim(advecting_hat, nd+1, j)
        for phys_I in phys_idxs
            ws.u_phys[phys_I, j] = _synth(I -> û_j[I], phys_I)
        end
    end

    # --- 𝒩ᵢ = Σⱼ uⱼ ∂fᵢ/∂xⱼ, streaming one gradient at a time through ws.g_phys ---
    for comp in 1:M
        û_c = selectdim(velocity_hat, nd+1, comp)
        Nc  = selectdim(ws.N_phys, nd+1, comp)
        fill!(Nc, zero(FT))
        for grad_d in 1:nd
            # `derivative_wavenumber`, matching the FFT engine: the grid derivative along `grad_d`
            # vanishes at that axis's Nyquist mode.
            for phys_I in phys_idxs
                ws.g_phys[phys_I] = _synth(
                    I -> (im * FT(SpectralLayout.derivative_wavenumber(ks[grad_d], I[grad_d]))) * û_c[I],
                    phys_I)
            end
            uj = selectdim(ws.u_phys, nd+1, grad_d)
            Nc .+= uj .* ws.g_phys
        end
    end

    # --- 𝒩̂ᵢ(k) = DFT(𝒩ᵢ), zeroed above the 2/3 cutoff ---
    for comp in 1:M
        N̂_c = selectdim(ws.N̂, nd+1, comp)
        Nc  = selectdim(ws.N_phys, nd+1, comp)
        for spec_I in spec_idxs
            if truncate && _is_dealiased(ks, spec_I)
                N̂_c[spec_I] = zero(eltype(ws.N̂))
                continue
            end
            val = zero(complex(FT))
            @inbounds for phys_I in phys_idxs
                phase = zero(FT)
                for d in 1:nd
                    phase += FT(2π) * SpectralLayout.axis_index_wavenumber(ks[d], spec_I[d]) *
                             FT(phys_I[d] - 1) / FT(ns[d])
                end
                val += Nc[phys_I] * exp(-im * phase)
            end
            N̂_c[spec_I] = val / FT(Np)
        end
    end

    return ws.N̂
end

end # module NonlinearTerm
