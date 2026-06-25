module SpectralFlux

using ..Types: SpectralFluxMethod, SpectralFluxResult, AbstractShellBinning, LinearBinning, AbstractExecutionBackend, SerialBackend, AbstractInvariant, KineticEnergy, AbstractFieldDecomposition, NoDecomposition, HelmholtzDecomposition, RotationalDecomposition, DivergentDecomposition
using ..Invariants: transfer_density!
using ..Decomposition: decompose_field
using ..ShellBinning: shell_edges, shell_centers, n_shells, assign_shells
using ..Utils: wavenumber_grid, wavenumber_magnitude_grid, domain_size_from_coords
using ..NonlinearTerm: compute_nonlinear_term!
using ..Workspaces: NonlinearTermWorkspace, SpectralFluxWorkspace

export calculate_spectral_flux, calculate_spectral_flux!

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

"""
    calculate_spectral_flux(velocity_hat, ks; binning, dealiasing=true) -> SpectralFluxResult

Compute the spectral energy transfer spectrum T(k) and the cumulative energy
flux Π(K) from Fourier-space velocity data.

# Arguments
- `velocity_hat`: Complex array of size `(ns..., D)` — Fourier coefficients of
  the D velocity components on an N-point periodic grid.
- `ks`: Tuple of 1D physical-wavenumber vectors (matching FFTW fftfreq convention).

# Keyword Arguments
- `binning::AbstractShellBinning`: Shell binning strategy; default `LinearBinning(1.0)`.
- `dealiasing::Bool=true`: Apply 2/3 dealiasing rule when computing (u·∇)u.
- `backend::AbstractExecutionBackend`: `SerialBackend()` (default) or `FFTBackend()` (requires FFTW extension).

# Returns
`SpectralFluxResult` with fields:
- `k_shells`: Representative wavenumber per shell.
- `transfer_spectrum`: T(k_n) — energy input to shell n per unit time.
- `flux`: Π(K_n) — cumulative upscale flux (energy transferred to k > K_n).

# Physics
  T(k_n) = Σ_{|k| ∈ shell_n} Re{ û*(k) · N̂(k) }
  Π(K_n) = −Σ_{m ≤ n} T(k_m)

Positive Π: forward (downscale) cascade; negative Π: inverse (upscale) cascade.

# References
- Verma et al. (2002) [arXiv:nlin/0204027]
- Alexakis, Mininni & Pouquet (2005)
"""
function calculate_spectral_flux(
    velocity_hat,
    ks::Tuple;
    binning::AbstractShellBinning = _default_binning(ks),
    dealiasing::Bool = true,
    invariant::AbstractInvariant = KineticEnergy(),
    decomposition::AbstractFieldDecomposition = NoDecomposition(),
    backend::AbstractExecutionBackend = SerialBackend(),
)
    decomposed = decompose_field(decomposition, velocity_hat, ks)
    return _calculate_spectral_flux_decomposed(
        decomposed, velocity_hat, ks, binning, dealiasing, invariant, backend
    )
end

function _calculate_spectral_flux_decomposed(
    û_decomp::AbstractArray{<:Complex},
    velocity_hat,
    ks::Tuple,
    binning::AbstractShellBinning,
    dealiasing::Bool,
    invariant::AbstractInvariant,
    backend::AbstractExecutionBackend,
)
    ws        = SpectralFluxWorkspace(velocity_hat, ks, binning)
    k_mag     = wavenumber_magnitude_grid(ks)
    edges     = shell_edges(binning, maximum(k_mag))
    centers   = shell_centers(binning, maximum(k_mag))
    shell_idx = assign_shells(k_mag, edges)
    result    = SpectralFluxResult(centers, similar(ws.T_spec), similar(ws.flux))
    
    if û_decomp === velocity_hat
        calculate_spectral_flux!(result, ws, velocity_hat, ks, shell_idx;
                                  dealiasing=dealiasing, invariant=invariant, backend=backend)
    else
        compute_nonlinear_term!(ws.nonlinear, velocity_hat, ks; dealiasing=dealiasing, backend=backend)
        _calculate_spectral_flux_with_N̂!(result, ws, û_decomp, ws.nonlinear.N̂, ks, shell_idx; invariant=invariant)
    end
    return result
end

function _calculate_spectral_flux_decomposed(
    decomposed::NamedTuple,
    velocity_hat,
    ks::Tuple,
    binning::AbstractShellBinning,
    dealiasing::Bool,
    invariant::AbstractInvariant,
    backend::AbstractExecutionBackend,
)
    ws = SpectralFluxWorkspace(velocity_hat, ks, binning)
    compute_nonlinear_term!(ws.nonlinear, velocity_hat, ks; dealiasing=dealiasing, backend=backend)
    N̂ = ws.nonlinear.N̂

    k_mag     = wavenumber_magnitude_grid(ks)
    edges     = shell_edges(binning, maximum(k_mag))
    centers   = shell_centers(binning, maximum(k_mag))
    shell_idx = assign_shells(k_mag, edges)

    return map(decomposed) do û_comp
        res = SpectralFluxResult(centers, similar(ws.T_spec), similar(ws.flux))
        _calculate_spectral_flux_with_N̂!(res, ws, û_comp, N̂, ks, shell_idx; invariant=invariant)
        return res
    end
end

"""
    calculate_spectral_flux!(result, ws, velocity_hat, ks, shell_idx; dealiasing, invariant, backend)

In-place version of `calculate_spectral_flux`. Writes into `result` using
preallocated buffers from `ws` and a precomputed `shell_idx` array (from `assign_shells`).
Zero heap allocations in the hot path.
"""
function calculate_spectral_flux!(
    result::SpectralFluxResult,
    ws::SpectralFluxWorkspace,
    velocity_hat,
    ks::Tuple,
    shell_idx::AbstractArray{Int};
    dealiasing::Bool = true,
    invariant::AbstractInvariant = KineticEnergy(),
    backend::AbstractExecutionBackend = SerialBackend(),
)
    compute_nonlinear_term!(ws.nonlinear, velocity_hat, ks;
                            dealiasing=dealiasing, backend=backend)
    _calculate_spectral_flux_with_N̂!(result, ws, velocity_hat, ws.nonlinear.N̂, ks, shell_idx; invariant=invariant)
    return result
end

function _calculate_spectral_flux_with_N̂!(
    result::SpectralFluxResult,
    ws::SpectralFluxWorkspace,
    velocity_hat,
    N̂,
    ks::Tuple,
    shell_idx::AbstractArray{Int};
    invariant::AbstractInvariant = KineticEnergy(),
)
    nd = length(ks)
    ns = size(velocity_hat)[1:nd]
    FT = real(eltype(velocity_hat))

    # Write per-mode transfer density into ws.transfer_density
    transfer_density!(ws.transfer_density, invariant, velocity_hat, N̂, ks)

    fill!(ws.T_spec, zero(FT))
    for I in CartesianIndices(ns)
        n = shell_idx[I]
        n == 0 && continue
        ws.T_spec[n] += ws.transfer_density[I]
    end

    copyto!(result.transfer_spectrum, ws.T_spec)
    # Flux convention (Alexakis & Biferale 2018, Eqs. 12–17; see THEORY.md §0.5):
    #   T(k) = Re{û*·N̂} is the net energy *loss* from shell k, and
    #   Π(K) = +Σ_{k≤K} T(k)  ⇒  Π>0 forward (down-scale) cascade, Π<0 inverse.
    # (Earlier code negated this, returning −Π — i.e. forward cascades read as negative.)
    cumsum!(ws.flux, ws.T_spec)
    copyto!(result.flux, ws.flux)

    return result
end

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

function _default_binning(ks)
    # Default linear binning with spacing = minimum non-zero wavenumber increment
    min_dk = Inf
    for k_vec in ks
        for k in k_vec
            ak = abs(k)
            ak > 0 && (min_dk = min(min_dk, ak))
        end
    end
    min_dk = isfinite(min_dk) ? min_dk : 1.0
    return LinearBinning(min_dk)
end

end # module SpectralFlux
