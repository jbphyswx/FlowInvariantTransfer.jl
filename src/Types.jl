module Types

export AbstractEnergyTransferMethod, SpectralFluxMethod, CoarseGrainingFluxMethod, ShellToShellTransferMethod, ModeToModeTransferMethod, TriadicOrthogonalDecompositionMethod
export AbstractInvariant, KineticEnergy, Helicity, Enstrophy, PassiveScalar
export AbstractFieldDecomposition, NoDecomposition, HelmholtzDecomposition, RotationalDecomposition, DivergentDecomposition, HelicalDecomposition, ToroidalPoloidalDecomposition
export AbstractFilter, SharpSpectralFilter, GaussianFilter, TopHatFilter
export AbstractShellBinning, LinearBinning, LogarithmicBinning, DyadicBinning, CustomBinning
export AbstractShellGeometry, ShellMagnitude, IsotropicShells, PerpendicularShells, ParallelShells
export AbstractExecutionBackend, SerialBackend, ThreadedBackend, DistributedBackend, GPUBackend, AutoBackend
export AbstractSpectralBackend, DirectSumBackend, FFTBackend, NUFFTBackend, SHTBackend, NUFSHTBackend
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
[`Enstrophy`](@ref) (2D), [`PassiveScalar`](@ref) (any D).

# Advected vs. carrier field
Every transfer diagnostic forms `T(k) = Re{ ĉ*(k) · 𝒩̂(k) }`, where `𝒩̂ = FFT[(u·∇)f]` is the
nonlinear term of the *advected* field `f` and `ĉ` is the *carrier*. For the momentum
invariants (KE/helicity/enstrophy) both are the velocity (`f = c = u`, with vorticity weighting
folded into the carrier for helicity/enstrophy). For [`PassiveScalar`](@ref) the advected and
carrier field is the scalar `θ`, advected by the velocity `u` — handled by passing the scalar as
the primary field and the velocity as `advecting_hat`.
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

Enstrophy `Ω = ½∫|ω|²`, transfer density `Re{ ω̂*(k) · N̂_ω(k) }` with `ω̂ = i k×û`
and `N̂_ω = i k×N̂`.

- **2D** (scalar vorticity `ω̂ = i(k_x û_y − k_y û_x)`): enstrophy is an inviscid invariant —
  conserved (`Σ_k T_Ω = 0`), counter-directional forward cascade dual to the inverse energy
  cascade (Kraichnan–Batchelor).
- **3D** (vector vorticity): `N̂_ω = curl[(u·∇)u] = (u·∇)ω − (ω·∇)u` includes vortex
  **stretching**, so enstrophy is **not** conserved (`Σ_k T_Ω ≠ 0`: net production). This is a
  valid transfer/budget diagnostic, not a conservative cascade.

Available in 3D via spectral flux and shell-to-shell; the explicit mode-to-mode triad form
is 2D-only for now.
"""
struct Enstrophy <: AbstractInvariant end

"""
    PassiveScalar <: AbstractInvariant

Passive-scalar variance `E_θ = ½∫θ²` (Obukhov–Corrsin), advected by the velocity:
`∂_tθ + (u·∇)θ = κ∇²θ`. The transfer density is `Re{ θ̂*(k) N̂_θ(k) }` with
`N̂_θ = FFT[(u·∇)θ]`.

The scalar is the *advected and carrier* field; the velocity only advects it. Pass the scalar
(shape `(ns..., 1)`) as the primary field and the velocity as `advecting_hat` (the convenience
entry points `calculate_scalar_*` do this for you).

Scalar variance is an inviscid invariant for incompressible flow (`∫θ(u·∇)θ = −½∫θ²∇·u = 0`),
so it is **conserved** (`Σ_k T_θ ≈ 0`) and cascades **forward** (to small scales) in both 2D and
3D — unlike kinetic energy there is no inverse-cascade dimension.

# A family of canonical invariants
Several other quadratic invariants are advected by the velocity exactly like a passive scalar, so
their cross-scale transfer is computed by this same path (pass the field as the "scalar"):
- **2D-MHD mean-square vector potential** `½⟨a²⟩` (`∂_t a + (u·∇)a = η∇²a`, `b = ∇×(a ẑ)`) —
  inverse-cascades in 2D.
- **Buoyancy / available-potential-energy variance** `½⟨b²⟩` (APE `= ½⟨b²⟩/N²`) in the
  Boussinesq system; the `−N²w` term is a KE↔APE *conversion* (a separate source, not a triad
  transfer), so the variance *cascade* is exactly the scalar transfer of `b`.
- **QG potential enstrophy** `½⟨q²⟩` with PV `q = ∇²ψ + βy` advected by the geostrophic velocity.

# References
- Obukhov (1949); Corrsin (1951); Batchelor (1959); 2D-MHD: Fyfe–Montgomery (1976);
  QG: Charney (1971); stratified APE: Lindborg (2006). See THEORY.md §0.5.
"""
struct PassiveScalar <: AbstractInvariant end

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
    HelicalDecomposition <: AbstractFieldDecomposition

Decompose a 3D velocity field into its **positive- and negative-helicity** components via the
Craya–Herring/helical basis (Waleffe 1992; Alexakis 2017). For each `k ≠ 0` the plane `⊥ k` is
spanned by the orthonormal helical eigenvectors of the curl,
`h_±(k) = (e₁ ± i e₂)/√2` with `i k̂ × h_± = ± h_±` and the Alexakis √2 unit-norm convention
(`h_± · h_±* = 1`, `h_+ · h_-* = 0`). The velocity projects as `û = u_+ h_+ + u_- h_-`
(`u_± = û · h_±*`), so

    E(k) = E⁺(k) + E⁻(k),   E^±(k) = ½|u_±|²,   H(k) = |k|(|u_+|² − |u_-|²) = 2|k|(E⁺ − E⁻),

recovering the realizability bound `|H(k)| ≤ 2|k| E(k)`. Returns the two **vector** components
`(positive = u_+ h_+, negative = u_- h_-)`; for an incompressible field they sum back to `û`.
3D only. Used as the `decomposition` argument to `calculate_spectral_flux` to get
helicity-resolved energy fluxes `Π^±(K)`.
"""
struct HelicalDecomposition <: AbstractFieldDecomposition end

"""
    ToroidalPoloidalDecomposition <: AbstractFieldDecomposition

Split a 3D solenoidal velocity into **toroidal** (horizontal/vortical) and **poloidal**
(vertical/wave) components in the Craya–Herring frame (Craya 1958; Herring 1974; Bartello 1995).
For each `k` with horizontal part `k_⊥ ≠ 0`, the plane `⊥ k` is spanned by
`e⁽¹⁾ = (k × ẑ)/|k × ẑ|` (horizontal, the toroidal direction) and `e⁽²⁾ = (k × e⁽¹⁾)/|k|` (the
poloidal direction); `û = u₁ e⁽¹⁾ + u₂ e⁽²⁾`. The toroidal part carries the vertical vorticity
and has **zero vertical velocity**; the poloidal part carries the vertical velocity (the linear
gravity-wave mode in stratified flow). For purely vertical `k` (`k_⊥ = 0`) the split is
degenerate and `(x̂, ŷ)` are used as an arbitrary horizontal orthonormal pair.

Returns `(toroidal = u₁ e⁽¹⁾, poloidal = u₂ e⁽²⁾)`; both are divergence-free and they sum back
to the solenoidal part of `û`. 3D only.
"""
struct ToroidalPoloidalDecomposition <: AbstractFieldDecomposition end

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
# Shell geometry — WHICH wavenumber coordinate the shells partition
# ---------------------------------------------------------------------------
#
# A binning (LinearBinning, …) sets the shell *spacing*; the geometry sets the *coordinate* that
# spacing is applied to. Isotropic shells partition |k| (spherical in 3D, annular in 2D). For
# rotating/stratified flows the canonical anisotropic fluxes Π(k_⊥), Π(k_∥) (Alexakis & Biferale
# 2018, §IV) partition a subset of components: k_⊥ = √(k_x²+k_y²) (cylindrical) or k_∥ = |k_z|.
# Geometry only changes which scalar each mode is binned by — the transfer physics is identical.

"""
    AbstractShellGeometry

Abstract supertype selecting the wavenumber coordinate the shells partition (isotropic `|k|`,
or an anisotropic projection like `k_⊥`/`k_∥`). Orthogonal to the binning *spacing*
([`AbstractShellBinning`](@ref)).
"""
abstract type AbstractShellGeometry end

"""
    ShellMagnitude(dims) <: AbstractShellGeometry

Bin modes by the Euclidean magnitude of the wavenumber components in `dims`:
`κ(k) = √(Σ_{d∈dims} k_d²)`. `dims === nothing` uses **all** spatial dimensions (isotropic `|k|`).

Use the constructors [`IsotropicShells`](@ref), [`PerpendicularShells`](@ref),
[`ParallelShells`](@ref) for the common cases.
"""
struct ShellMagnitude{D} <: AbstractShellGeometry
    dims::D
end

"""
    IsotropicShells() -> ShellMagnitude

Isotropic shells over `|k|` (all dimensions) — the default geometry.
"""
IsotropicShells() = ShellMagnitude(nothing)

"""
    PerpendicularShells(dims=(1, 2)) -> ShellMagnitude

Cylindrical shells over `k_⊥ = √(Σ_{d∈dims} k_d²)` (the horizontal plane by default), giving the
anisotropic perpendicular flux `Π(k_⊥)` for rotating/stratified flows.
"""
PerpendicularShells(dims=(1, 2)) = ShellMagnitude(dims)

"""
    ParallelShells(dims=(3,)) -> ShellMagnitude

Plane shells over `k_∥ = √(Σ_{d∈dims} k_d²)` (the vertical axis by default), giving the
anisotropic parallel flux `Π(k_∥)`.
"""
ParallelShells(dims=(3,)) = ShellMagnitude(dims)

# ---------------------------------------------------------------------------
# Backends — two orthogonal axes that compose:
#   • AbstractSpectralBackend  : WHICH transform   (direct / FFT / NUFFT / SHT / NUFSHT)
#   • AbstractExecutionBackend : HOW it is run      (serial / threaded / distributed / GPU)
# A computation chooses one of each, e.g. FFT transforms with a threaded mediator loop, or a
# NUFSHT transform with an MPI reduction. Keeping them separate avoids conflating "what" with
# "where" (a single transform's parallelism lives in FFTW threads / the GPU array, not here).
# ---------------------------------------------------------------------------

# --- Transform (spectral) axis ---------------------------------------------

"""
    AbstractSpectralBackend

Abstract supertype for *transform* backends: how physical↔spectral coefficients and the
pseudospectral nonlinear term are computed. Orthogonal to [`AbstractExecutionBackend`](@ref).
"""
abstract type AbstractSpectralBackend end

"""
    DirectSumBackend <: AbstractSpectralBackend

Dependency-free direct DFT/sum reference transform (no external packages); exact but slow.
The default — load `FFTW` and pass [`FFTBackend`](@ref) for the O(N log N) fast path.
"""
struct DirectSumBackend <: AbstractSpectralBackend end

"""
    FFTBackend <: AbstractSpectralBackend

Uniform-grid FFT transform via FFTW (O(N log N)). Requires `using FFTW`.
"""
struct FFTBackend <: AbstractSpectralBackend end

"""
    NUFFTBackend <: AbstractSpectralBackend

Non-uniform FFT for scattered Cartesian points, via FINUFFT. Requires `using FINUFFT`.
"""
struct NUFFTBackend <: AbstractSpectralBackend end

"""
    SHTBackend <: AbstractSpectralBackend

Spherical-harmonic transform for regular spherical grids, via FastSphericalHarmonics.
"""
struct SHTBackend <: AbstractSpectralBackend end

"""
    NUFSHTBackend <: AbstractSpectralBackend

Non-uniform spherical-harmonic transform for scattered spherical data, via NUFSHT.
"""
struct NUFSHTBackend <: AbstractSpectralBackend end

# --- Execution (parallelism) axis ------------------------------------------

"""
    AbstractExecutionBackend

Abstract supertype for *execution* backends: how the outer work (shell/mode loops and
reductions) is parallelised. Orthogonal to [`AbstractSpectralBackend`](@ref).
"""
abstract type AbstractExecutionBackend end

"""
    SerialBackend <: AbstractExecutionBackend

Single-threaded serial execution; no external dependencies.
"""
struct SerialBackend <: AbstractExecutionBackend end

"""
    ThreadedBackend <: AbstractExecutionBackend

Shared-memory multithreading over the outer (shell/mode) loop. Load `OhMyThreads`.
"""
struct ThreadedBackend <: AbstractExecutionBackend end

"""
    DistributedBackend <: AbstractExecutionBackend

Multi-process execution via `Distributed`/`SharedArrays`: parallelises the outer loop over
mediator shells / receiver modes across workers. Requires `using Distributed, SharedArrays`.
"""
struct DistributedBackend <: AbstractExecutionBackend end

"""
    GPUBackend{B} <: AbstractExecutionBackend

GPU execution via `KernelAbstractions`, holding the device backend `backend::B`
(e.g. `GPUBackend(CUDA.CUDABackend())`). Requires `using KernelAbstractions` + a vendor package.
"""
struct GPUBackend{B} <: AbstractExecutionBackend
    backend::B
end

"""
    AutoBackend <: AbstractExecutionBackend

Select the best available execution backend at call time (distributed → threaded → serial).
"""
struct AutoBackend <: AbstractExecutionBackend end

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
    ModeToModeTriadResult{I, KS, A, S}

Result of a fully mode-resolved triad transfer computation.

# Fields
- `invariant::I`: The invariant that was accumulated (e.g. `KineticEnergy()`).
- `ks::KS`: The wavenumber vectors `(kx, ky[, kz])` defining the spectral grid.
- `net_transfer::A`: `T(k) = Σ_p S(k|p)` — net per-mode transfer (shape `ns`); equals the
  spectral transfer from `calculate_spectral_flux`.
- `transfer::S`: the resolved `S(k|p)` — energy delivered to receiver mode `k` from giver mode
  `p` (mediated by `q=k−p`), shape `(ns..., ns...)` (receiver indices then giver indices).
  Antisymmetric (`S(k|p)=−S(p|k)`); summed over `p` gives `net_transfer`; summed over shells
  gives the shell-to-shell matrix.

Parametric on all array/field types — GPU-array friendly.
"""
struct ModeToModeTriadResult{I, KS, A, S}
    invariant::I
    ks::KS
    net_transfer::A
    transfer::S
end

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
