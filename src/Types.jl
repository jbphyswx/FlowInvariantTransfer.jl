module Types

export AbstractEnergyTransferMethod, SpectralFluxMethod, CoarseGrainingFluxMethod, ShellToShellTransferMethod, ModeToModeTransferMethod, TriadicOrthogonalDecompositionMethod
export AbstractInvariant, KineticEnergy, Helicity, Enstrophy
export AbstractFieldDecomposition, NoDecomposition, HelmholtzDecomposition, RotationalDecomposition, DivergentDecomposition
export AbstractFilter, SharpSpectralFilter, GaussianFilter, TopHatFilter
export AbstractShellBinning, LinearBinning, LogarithmicBinning, DyadicBinning, CustomBinning
export AbstractExecutionBackend, SerialBackend, ThreadedBackend, DistributedBackend, GPUBackend, AutoBackend, FFTBackend, NUFFTBackend, SHTBackend, NUFSHTBackend
export SpectralFluxResult, CoarseGrainingFluxResult, CoarseGrainingFluxResultWithDiagnostics, ShellToShellResult, ModeToModeTriadResult, TriadicOrthogonalDecompositionResult

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

# ---------------------------------------------------------------------------
# Quadratic-invariant trait
# ---------------------------------------------------------------------------

"""
    AbstractInvariant

Trait supertype selecting *which quadratic inviscid invariant* a transfer
diagnostic accumulates. The same nonlinear-term machinery serves every
invariant; only the per-mode transfer-density weighting changes (see
`Invariants.transfer_density!`).

Concrete subtypes: [`KineticEnergy`](@ref) (default), [`Helicity`](@ref) (3D),
[`Enstrophy`](@ref) (2D).
"""
abstract type AbstractInvariant end

"""
    KineticEnergy <: AbstractInvariant

Kinetic energy `E = ½∫|u|²`. The default invariant; transfer density is
`Re{ û*(k) · N̂(k) }`. Forward cascade in 3D, inverse in 2D.
"""
struct KineticEnergy <: AbstractInvariant end

"""
    Helicity <: AbstractInvariant

Helicity `H = ∫ u·ω`, `ω = ∇×u` (3D only). Transfer density is
`Re{ ω̂*(k) · N̂(k) }` with `ω̂ = i k × û`. Co-directional (forward) with energy.
"""
struct Helicity <: AbstractInvariant end

"""
    Enstrophy <: AbstractInvariant

Enstrophy `Ω = ½∫ ω²` (2D only), `ω = ∂_x v − ∂_y u`. Transfer density is
`Re{ ω̂*(k) · N̂_ω(k) }` with scalar vorticity `ω̂ = i(k_x û_y − k_y û_x)` and
its nonlinear term `N̂_ω = i(k_x N̂_y − k_y N̂_x)`. Counter-directional (forward)
to the inverse energy cascade (Kraichnan–Batchelor).
"""
struct Enstrophy <: AbstractInvariant end

# ---------------------------------------------------------------------------
# Field decomposition / projection traits
# ---------------------------------------------------------------------------

"""
    AbstractFieldDecomposition

Abstract supertype specifying the field decomposition/projection strategy
(e.g., Helmholtz rotational/divergent decomposition).
"""
abstract type AbstractFieldDecomposition end

"""
    NoDecomposition <: AbstractFieldDecomposition

No decomposition or projection is applied; use the full velocity field.
"""
struct NoDecomposition <: AbstractFieldDecomposition end

"""
    HelmholtzDecomposition <: AbstractFieldDecomposition

Decompose the velocity field into rotational (solenoidal) and divergent (dilatational)
components, computing transfer results for both.
"""
struct HelmholtzDecomposition <: AbstractFieldDecomposition end

"""
    RotationalDecomposition <: AbstractFieldDecomposition

Only compute or retain the rotational (solenoidal/divergence-free) component.
"""
struct RotationalDecomposition <: AbstractFieldDecomposition end

"""
    DivergentDecomposition <: AbstractFieldDecomposition

Only compute or retain the divergent (dilatational/curl-free) component.
"""
struct DivergentDecomposition <: AbstractFieldDecomposition end

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
    CoarseGrainingFluxMethod{F<:AbstractFilter, S} <: AbstractEnergyTransferMethod

Compute the pointwise cross-scale energy flux Π_ℓ(x) = −τ̄ᵢⱼ S̄ᵢⱼ at filter scale ℓ.

# Fields
- `filter::F`: Filter kernel (Gaussian, sharp-spectral, or top-hat).
- `scale::S`: Filter length scale ℓ (same units as the coordinate arrays).

# Notes
Physical-space output; suitable for detecting spatial intermittency in the cascade.
"""
struct CoarseGrainingFluxMethod{F<:AbstractFilter, S} <: AbstractEnergyTransferMethod
    filter::F
    scale::S
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

"""
    ModeToModeTransferMethod{B, I<:AbstractInvariant} <: AbstractEnergyTransferMethod

Compute the exact **mode-to-mode triad transfer** `S(k|p|q)` — energy (or other
invariant) given *to* receiver mode `k` *from* giver `p`, mediated by `q`, with
triad closure `k = p + q`:

    S(k|p|q) = −Im{ [k · û(q)] [û*(k) · û(p)] }.

This is the most fundamental (delta-in-`k`) scale-to-scale object; it reduces to
the shell-to-shell matrix and the spectral transfer `T(k)` under summation.

# Fields
- `binning::B`: Optional shell binning for reductions to the magnitude-to-magnitude
  transfer `T(K,Q)`. Use `nothing` to return the raw per-receiver transfer only.
- `invariant::I`: Which quadratic invariant to accumulate (default `KineticEnergy()`).

# Cost
`O(N^D)` per receiver mode; `O(N^{2D})` for the full tensor — exact but slow.
Guard with a mode-count limit unless `force=true`.

# References
- Dar, Verma & Eswaran (2001); Verma (2004 review, 2019 book).
"""
struct ModeToModeTransferMethod{B, I<:AbstractInvariant} <: AbstractEnergyTransferMethod
    binning::B
    invariant::I
end
ModeToModeTransferMethod(; binning=nothing, invariant=KineticEnergy()) =
    ModeToModeTransferMethod(binning, invariant)
ModeToModeTransferMethod(binning::AbstractShellBinning; invariant=KineticEnergy()) =
    ModeToModeTransferMethod(binning, invariant)

"""
    TriadicOrthogonalDecompositionMethod{N, O, M} <: AbstractEnergyTransferMethod

Triadic Orthogonal Decomposition (Yeung, Chu & Schmidt 2026).

Operates on temporal snapshots, decomposing triadic (three-wave) nonlinear
interactions in the temporal-frequency domain via the mode bispectrum.

# Fields
- `nfft`: DFT block length. `nothing` for auto-selection.
- `noverlap`: Block overlap in snapshots. `nothing` for 50% of window.
- `nmode`: Number of modes per triad to retain. `nothing` for nblocks.

# References
- Yeung, Chu & Schmidt (2026), J. Fluid Mech. 1031, A34.
  DOI 10.1017/jfm.2026.11183
"""
struct TriadicOrthogonalDecompositionMethod{N, O, M} <: AbstractEnergyTransferMethod
    nfft::N
    noverlap::O
    nmode::M
end
TriadicOrthogonalDecompositionMethod(; nfft=nothing, noverlap=nothing, nmode=nothing) =
    TriadicOrthogonalDecompositionMethod(nfft, noverlap, nmode)


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
- `Δk`: Shell width in physical wavenumber units.
"""
struct LinearBinning{T} <: AbstractShellBinning
    Δk::T
end

"""
    LogarithmicBinning(k₀, λ) <: AbstractShellBinning

Geometrically-spaced shells: k_n = k₀ · λⁿ.

# Fields
- `k₀`: First shell lower edge (> 0).
- `λ`: Ratio between consecutive shell edges (> 1); λ = 2 gives dyadic.
"""
struct LogarithmicBinning{T} <: AbstractShellBinning
    k₀::T
    λ::T
end
LogarithmicBinning(k₀, λ) = LogarithmicBinning(promote(k₀, λ)...)

"""
    DyadicBinning(k₀) <: AbstractShellBinning

Dyadic (octave) shells: k_n = k₀ · 2ⁿ.  Equivalent to `LogarithmicBinning(k₀, 2.0)`.

# Fields
- `k₀`: First shell lower edge (> 0).
"""
struct DyadicBinning{T} <: AbstractShellBinning
    k₀::T
end

"""
    CustomBinning(edges) <: AbstractShellBinning

User-specified shell edges.  Shell n covers wavenumbers in [edges[n], edges[n+1]).

# Fields
- `edges`: Monotonically increasing edge values (length = N_shells + 1).
"""
struct CustomBinning{V<:AbstractVector} <: AbstractShellBinning
    edges::V
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

"""
    DistributedBackend <: AbstractExecutionBackend

Multi-process execution using the `Distributed` standard library (with
`SharedArrays`). Parallelises the outer loop over receiver shells/modes across
worker processes. Requires `using Distributed, SharedArrays` to load the extension.
"""
struct DistributedBackend <: AbstractExecutionBackend end

"""
    GPUBackend{B} <: AbstractExecutionBackend

GPU execution via `KernelAbstractions`. Holds the concrete KA backend object
`backend::B` (e.g. `GPUBackend(CUDA.CUDABackend())`), so the same kernels run on
any KA-supported device. Requires `using KernelAbstractions` (and a vendor package
such as `CUDA`) to load the extension.
"""
struct GPUBackend{B} <: AbstractExecutionBackend
    backend::B
end

"""
    AutoBackend <: AbstractExecutionBackend

Automatically select the best available execution backend at call time, in the
order distributed → threaded → serial (transform fast-paths such as FFT are
chosen independently when their extensions are loaded).
"""
struct AutoBackend <: AbstractExecutionBackend end

"""
    SHTBackend <: AbstractExecutionBackend

Spherical-harmonic-transform front-end for regular spherical grids, backed by
`FastSphericalHarmonics`. Requires the FastSphericalHarmonics extension.
"""
struct SHTBackend <: AbstractExecutionBackend end

"""
    NUFSHTBackend <: AbstractExecutionBackend

Non-uniform spherical-harmonic-transform front-end for scattered spherical data,
backed by `NUFSHT`. Requires the NUFSHT extension.
"""
struct NUFSHTBackend <: AbstractExecutionBackend end

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

"""
    ModeToModeTriadResult{I, KS, A, NT}

Result of a mode-to-mode triad transfer computation `S(k|p|q)`.

# Fields
- `invariant::I`: The invariant that was accumulated (e.g. `KineticEnergy()`).
- `ks::KS`: The wavenumber vectors `(kx, ky[, kz])` defining the spectral grid.
- `net_transfer::A`: `T(k) = Σ_p S(k|p|q=k−p)` — net per-mode transfer, same
  spatial shape as one velocity component.
- `reductions::NT`: A `NamedTuple` of optional reductions, e.g.
  `(; K, Q, TKQ)` for the magnitude-to-magnitude matrix `T(K,Q)` when a binning
  was supplied, or an empty `NamedTuple` otherwise.

Parametric on all array/field types — GPU-array friendly.
"""
struct ModeToModeTriadResult{I, KS, A, NT}
    invariant::I
    ks::KS
    net_transfer::A
    reductions::NT
end
ModeToModeTriadResult(invariant, ks, net_transfer) =
    ModeToModeTriadResult(invariant, ks, net_transfer, NamedTuple())

"""
    TriadicOrthogonalDecompositionResult{V, A3, PM, EC, XM}

Result container for Triadic Orthogonal Decomposition.

# Fields
- `frequencies::V`: Frequency vector (length nFreq).
- `mode_bispectrum::A3`: Singular values λ(fl, fn, mode) — array of size
  `(nFreq, nFreq, nmode)`.
- `modes::PM`: Dict mapping `(l, n)` index tuples to mode arrays. Each value
  contains convective modes (index 1 along first dim) and recipient modes
  (index 2) with spatial/variable dimensions.
- `modal_energy_budget::A3`: Energy transfer T(fl, fn, mode) per triad per mode.
  Same shape as `mode_bispectrum`.
- `expansion_coefficients::EC`: Expansion coefficients, or `nothing` if not requested.
- `auxiliary_modes::XM`: Dict mapping `(l, n)` to donor/catalyst modes, or `nothing`.

All fields are typed — the optional `EC`/`XM` parameters resolve to `Nothing` or the concrete
container type at construction, so the struct is type-stable (no untyped `Any` fields).
"""
struct TriadicOrthogonalDecompositionResult{V<:AbstractVector, A3<:AbstractArray, PM, EC, XM}
    frequencies::V
    mode_bispectrum::A3
    modes::PM
    modal_energy_budget::A3
    expansion_coefficients::EC
    auxiliary_modes::XM
end

end # module Types
