module Types

export AbstractEnergyTransferMethod, SpectralFluxMethod, CoarseGrainingFluxMethod, ShellToShellTransferMethod
export AbstractFilter, SharpSpectralFilter, GaussianFilter, TopHatFilter
export AbstractShellBinning, LinearBinning, LogarithmicBinning, DyadicBinning, CustomBinning
export AbstractExecutionBackend, SerialBackend, ThreadedBackend, FFTBackend, NUFFTBackend
export SpectralFluxResult, CoarseGrainingFluxResult, CoarseGrainingFluxResultWithDiagnostics, ShellToShellResult

# ---------------------------------------------------------------------------
# Method hierarchy
# ---------------------------------------------------------------------------

"""
    AbstractEnergyTransferMethod

Abstract supertype for all energy transfer computation methods.
Concrete subtypes dispatch `calculate_energy_transfer` to specific algorithms.
"""
abstract type AbstractEnergyTransferMethod end

# Declare abstract types needed as type parameters before the structs that use them
abstract type AbstractShellBinning end
abstract type AbstractFilter end

"""
    SpectralFluxMethod{B<:AbstractShellBinning} <: AbstractEnergyTransferMethod

Compute the spectral energy flux Π(K) and transfer spectrum T(k) using the
pseudospectral method on a periodic uniform grid.

# Fields
- `binning::B`: Shell binning strategy for grouping wavenumbers.

# Notes
Requires Fourier-space velocity data on a uniform periodic grid.
When FFTW is loaded, all transforms run in O(N log N); without it, falls back
to an O(N²) direct-sum reference implementation.
"""
struct SpectralFluxMethod{B<:AbstractShellBinning} <: AbstractEnergyTransferMethod
    binning::B
end

"""
    CoarseGrainingFluxMethod{F<:AbstractFilter} <: AbstractEnergyTransferMethod

Compute the pointwise cross-scale energy flux Π_ℓ(x) = −τ̄ᵢⱼ S̄ᵢⱼ at filter scale ℓ.

# Fields
- `filter::F`: Filter kernel (Gaussian, sharp-spectral, or top-hat).
- `scale::Float64`: Filter length scale ℓ (same units as the coordinate arrays).

# Notes
Physical-space output; suitable for detecting spatial intermittency in the cascade.
"""
struct CoarseGrainingFluxMethod{F<:AbstractFilter} <: AbstractEnergyTransferMethod
    filter::F
    scale::Float64
end

"""
    ShellToShellTransferMethod{B<:AbstractShellBinning} <: AbstractEnergyTransferMethod

Compute the directed shell-to-shell transfer matrix T(n,m), where T(n,m) is the
rate of energy transfer from shell S_m into shell S_n mediated by the nonlinear
advection term.

# Fields
- `binning::B`: Shell binning strategy.

# Notes
The antisymmetry property T(n,m) = −T(m,n) holds when the mediator velocity is
the full field u (Verma et al. 2002). This is automatically verified by default.
"""
struct ShellToShellTransferMethod{B<:AbstractShellBinning} <: AbstractEnergyTransferMethod
    binning::B
end

# ---------------------------------------------------------------------------
# Filter hierarchy
# ---------------------------------------------------------------------------

"""
    SharpSpectralFilter <: AbstractFilter

Ideal low-pass (brick-wall) filter in spectral space:
  Ĝ(k, ℓ) = 1  if |k| < π/ℓ,  else 0.

Provides exact scale separation but produces Gibbs ringing in physical space.
"""
struct SharpSpectralFilter <: AbstractFilter end

"""
    GaussianFilter <: AbstractFilter

Gaussian filter in spectral space:
  Ĝ(k, ℓ) = exp(−k² ℓ² / 24).

Excellent physical-space locality; widely used in LES and coarse-graining studies.
The normalisation factor 24 follows the convention of Aluie et al. (2018).
"""
struct GaussianFilter <: AbstractFilter end

"""
    TopHatFilter <: AbstractFilter

Top-hat (box) filter in physical space; sinc response in spectral space:
  Ĝ(k, ℓ) = sinc(k ℓ / (2π)).

Compact support in physical space; standard in LES.
"""
struct TopHatFilter <: AbstractFilter end

# ---------------------------------------------------------------------------
# Shell-binning hierarchy
# ---------------------------------------------------------------------------

"""
    LinearBinning(Δk) <: AbstractShellBinning

Uniform shell spacing: k_n = n · Δk.

# Fields
- `Δk::Float64`: Shell width in physical wavenumber units.
"""
struct LinearBinning <: AbstractShellBinning
    Δk::Float64
end

"""
    LogarithmicBinning(k₀, λ) <: AbstractShellBinning

Geometrically-spaced shells: k_n = k₀ · λⁿ.

# Fields
- `k₀::Float64`: First shell lower edge (> 0).
- `λ::Float64`: Ratio between consecutive shell edges (> 1); λ = 2 gives dyadic.
"""
struct LogarithmicBinning <: AbstractShellBinning
    k₀::Float64
    λ::Float64
end

"""
    DyadicBinning(k₀) <: AbstractShellBinning

Dyadic (octave) shells: k_n = k₀ · 2ⁿ.  Equivalent to `LogarithmicBinning(k₀, 2.0)`.

# Fields
- `k₀::Float64`: First shell lower edge (> 0).
"""
struct DyadicBinning <: AbstractShellBinning
    k₀::Float64
end

"""
    CustomBinning(edges) <: AbstractShellBinning

User-specified shell edges.  Shell n covers wavenumbers in [edges[n], edges[n+1]).

# Fields
- `edges::Vector{Float64}`: Monotonically increasing edge values (length = N_shells + 1).
"""
struct CustomBinning <: AbstractShellBinning
    edges::Vector{Float64}
end

# ---------------------------------------------------------------------------
# Execution backends
# ---------------------------------------------------------------------------

"""
    AbstractExecutionBackend

Abstract supertype for computation backends.
"""
abstract type AbstractExecutionBackend end

"""
    SerialBackend <: AbstractExecutionBackend

Single-threaded serial execution; no external dependencies.
"""
struct SerialBackend <: AbstractExecutionBackend end

"""
    ThreadedBackend <: AbstractExecutionBackend

Multi-threaded execution using Julia's base `Threads.@threads`.
For OhMyThreads-based backend, load the `OhMyThreads` extension.
"""
struct ThreadedBackend <: AbstractExecutionBackend end

"""
    FFTBackend <: AbstractExecutionBackend

Fast-path backend using FFTW for all transforms.
Requires `using FFTW` to load the FFTW extension.
"""
struct FFTBackend <: AbstractExecutionBackend end

"""
    NUFFTBackend <: AbstractExecutionBackend

Non-uniform fast Fourier transform backend using FINUFFT.
Requires `using FINUFFT` to load the FINUFFT extension.
"""
struct NUFFTBackend <: AbstractExecutionBackend end

# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

"""
    SpectralFluxResult{V}

Result of a spectral energy flux computation.

# Fields
- `k_shells::V`: Representative wavenumber for each shell (midpoint of bin edges).
- `transfer_spectrum::V`: T(k) — energy transfer rate per shell.
- `flux::V`: Π(K) — cumulative energy flux (negative integral of T(k)).

Parametric on the vector type `V` — works with any `AbstractVector` element type
(Float32, Float64, ForwardDiff.Dual, Unitful quantities, etc.).
"""
struct SpectralFluxResult{V<:AbstractVector}
    k_shells::V
    transfer_spectrum::V
    flux::V
end
SpectralFluxResult(k, T, f) = SpectralFluxResult{typeof(k)}(k, T, f)

"""
    CoarseGrainingFluxResult{S, A}

Result of a coarse-graining energy flux computation (flux field only).

# Fields
- `filter_scale::S`: Filter scale ℓ used.
- `flux_field::A`: Π_ℓ(x) pointwise energy flux field (same shape as input velocity).
- `mean_flux::S`: Area-weighted spatial mean ⟨Π_ℓ⟩.

See also `CoarseGrainingFluxResultWithDiagnostics` for stress/strain output.
"""
struct CoarseGrainingFluxResult{S, A<:AbstractArray}
    filter_scale::S
    flux_field::A
    mean_flux::S
end
CoarseGrainingFluxResult(s, a, m) = CoarseGrainingFluxResult{typeof(s), typeof(a)}(s, a, m)

"""
    CoarseGrainingFluxResultWithDiagnostics{S, A}

Result of a coarse-graining energy flux computation including stress/strain diagnostics.

# Fields
- `filter_scale::S`: Filter scale ℓ used.
- `flux_field::A`: Π_ℓ(x) pointwise energy flux field.
- `mean_flux::S`: Area-weighted spatial mean ⟨Π_ℓ⟩.
- `stress_tensor::A`: τ̄ᵢʲ (same array type as `flux_field`).
- `strain_rate::A`: S̄ᵢʲ (same array type as `flux_field`).

Returned instead of `CoarseGrainingFluxResult` when `return_diagnostics=true`.
All fields are always present — no `Union{Nothing,...}` type instability.
"""
struct CoarseGrainingFluxResultWithDiagnostics{S, A<:AbstractArray}
    filter_scale::S
    flux_field::A
    mean_flux::S
    stress_tensor::A
    strain_rate::A
end
CoarseGrainingFluxResultWithDiagnostics(s, a, m, st, sr) =
    CoarseGrainingFluxResultWithDiagnostics{typeof(s), typeof(a)}(s, a, m, st, sr)

"""
    ShellToShellResult{V, M, E}

Result of a shell-to-shell energy transfer computation.

# Fields
- `shell_centers::V`: Representative wavenumber for each shell.
- `shell_edges::V`: Shell boundary wavenumbers (length = N_shells + 1).
- `transfer_matrix::M`: T(n,m) — N_shells × N_shells matrix; T[n,m] is energy from shell m to shell n.
- `net_transfer::V`: Σ_m T(n,m) for each receiver shell n (net energy gain of shell n).
- `max_antisymmetry_error::E`: max |T(n,m) + T(m,n)| — antisymmetry validation metric.

Parametric on vector type `V`, matrix type `M`, and scalar type `E = eltype(M)`.
"""
struct ShellToShellResult{V<:AbstractVector, M<:AbstractMatrix, E}
    shell_centers::V
    shell_edges::V
    transfer_matrix::M
    net_transfer::V
    max_antisymmetry_error::E
end
ShellToShellResult(c, e, T, n, a) =
    ShellToShellResult{typeof(c), typeof(T), eltype(T)}(c, e, T, n, a)

end # module Types
