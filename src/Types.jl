module Types

export AbstractEnergyTransferMethod, SpectralFluxMethod, CoarseGrainingFluxMethod, ShellToShellTransferMethod, ModeToModeTransferMethod, TriadicOrthogonalDecompositionMethod, SphericalTransferMethod, DivergentSphericalTransferMethod
export AbstractInvariant, KineticEnergy, Helicity, Enstrophy, PassiveScalar
export AbstractFieldDecomposition, NoDecomposition, HelmholtzDecomposition, RotationalDecomposition, DivergentDecomposition, HelicalDecomposition, ToroidalPoloidalDecomposition
export AbstractFilter, SharpSpectralFilter, GaussianFilter, TopHatFilter
export AbstractShellBinning, LinearBinning, LogarithmicBinning, DyadicBinning, CustomBinning
export AbstractShellGeometry, ShellMagnitude, IsotropicShells, PerpendicularShells, ParallelShells
export SmoothBands
export AbstractDealiasing, NoDealiasing, OrszagTwoThirds, PaddedThreeHalves
export AbstractSpectralBackend, DirectSumBackend, FFTBackend, NUFFTBackend, SHTBackend, NUFSHTBackend
# Execution backends (SerialBackend/ThreadedBackend/GPUBackend/DistributedBackend/MPIBackend/…) live
# in the sibling `Backends` module.
export SpectralFluxResult, CompressibleFluxResult, CoarseGrainingFluxResult, CoarseGrainingFluxResultWithDiagnostics, ShellToShellResult, ModeToModeTriadResult, TriadicOrthogonalDecompositionResult, SphericalTransferResult, DivergentSphericalTransferResult

# ---------------------------------------------------------------------------
# Method hierarchy
# ---------------------------------------------------------------------------

"""
    AbstractEnergyTransferMethod

Abstract supertype for all energy transfer computation methods.
Concrete subtypes dispatch `calculate_energy_transfer` to specific algorithms.
"""
abstract type AbstractEnergyTransferMethod end

# Declare abstract types needed as type parameters before the structs that use them.

"""
    AbstractShellBinning

Supertype for shell *spacing* strategies (how wavenumber space is partitioned into shells):
[`LinearBinning`](@ref), [`LogarithmicBinning`](@ref), [`DyadicBinning`](@ref),
[`CustomBinning`](@ref). Orthogonal to the shell *coordinate* ([`AbstractShellGeometry`](@ref)).
"""
abstract type AbstractShellBinning end

"""
    AbstractFilter

Supertype for spectral filter kernels used by coarse-graining: [`SharpSpectralFilter`](@ref),
[`GaussianFilter`](@ref), [`TopHatFilter`](@ref).
"""
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

Available in 2D and 3D across every diagnostic — spectral flux, shell-to-shell, and the resolved
mode-to-mode triad form (the invariant weighting rides the generic `transfer_density!`).
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
Other quadratic invariants are advected by the velocity exactly like a passive scalar, so their
cross-scale transfer is computed by this same path (pass the field as the "scalar"):
- **Buoyancy / available-potential-energy variance** `½⟨b²⟩` (APE `= ½⟨b²⟩/N²`) in the
  Boussinesq system; the `−N²w` term is a KE↔APE *conversion* (a separate source, not a triad
  transfer), so the variance *cascade* is exactly the scalar transfer of `b`.
- **QG potential enstrophy** `½⟨q²⟩` with PV `q = ∇²ψ + βy` advected by the geostrophic velocity.

# References
- Obukhov (1949); Corrsin (1951); Batchelor (1959); QG: Charney (1971);
  stratified APE: Lindborg (2006). See THEORY.md §0.5.
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

"""
    SphericalTransferMethod{T<:Real} <: AbstractEnergyTransferMethod

Spectral energy/enstrophy transfer for 2D non-divergent (barotropic) flow on the sphere, in the
spherical-harmonic degree spectrum `l`. Given the vorticity field `ζ`, with streamfunction
`ψ = ∇⁻²ζ` (so `ζ̂_lm = -l(l+1)/a² ψ̂_lm`) and advection `A = J(ψ,ζ) = u·∇ζ`, `u = k̂×∇ψ`, the
transfers are

    T_E(l) = -Σ_m Re{ψ̂*_lm Â_lm},   T_Z(l) = Σ_m Re{ζ̂*_lm Â_lm},

both conserving (Σ_l T = 0). See THEORY.md §"Spherical spectral transfer".

Dispatched through [`calculate_energy_transfer`](@ref FlowInvariantTransfer.calculate_energy_transfer): a regular colatitude–longitude grid (an
`AbstractMatrix` vorticity field) routes to the FastSphericalHarmonics extension; scattered points
(a vorticity vector + `(θ, φ)` coordinates) route to the NUFSHT extension.

# Fields
- `radius::T`: sphere radius `a` (default `1.0`).
"""
struct SphericalTransferMethod{T<:Real} <: AbstractEnergyTransferMethod
    radius::T
end
SphericalTransferMethod(; radius=1.0) = SphericalTransferMethod(radius)

"""
    DivergentSphericalTransferMethod{T<:Real} <: AbstractEnergyTransferMethod

Spectral kinetic-energy transfer for the full **horizontal** flow on the sphere — rotational *and*
divergent — in the spherical-harmonic degree spectrum `l`. Generalises
[`SphericalTransferMethod`](@ref) (which assumes non-divergent/barotropic flow, `∇·u = 0`) to a
velocity field carrying divergence, and reduces to it exactly in the non-divergent limit.

The input is the horizontal velocity `u = (u_θ, u_φ)` (colatitude, longitude components), Helmholtz-
decomposed as `u = k̂×∇ψ + ∇χ` (rotational streamfunction `ψ`, divergent velocity potential `χ`).
Writing the advection in Lamb (rotational) form `(u·∇)u = ∇(½|u|²) + ζ (k̂×u)` with vorticity
`ζ = k̂·(∇×u)`, the nonlinear KE transfer into degree `l` is the vector-harmonic projection

    T(l) = Σ_m Re{ û*_lm · Â_lm},   Â = [(u·∇)u]^  (spin-1 vector-harmonic coefficients),

split by the toroidal (rotational) / spheroidal (divergent) parts of `û` into `T = T_rot + T_div`.
Total KE is advectively conserved: `Σ_l T(l) ≈ 0` (the rotational and divergent channels are not
individually conserved — they exchange energy). The Lamb form needs only spin-0/spin-1 transforms
(no spin-2). See THEORY.md §"Divergent spherical spectral transfer" (Augier–Lindborg 2013;
Burgess–Erler–Shepherd 2013).

Dispatched through [`calculate_energy_transfer`](@ref FlowInvariantTransfer.calculate_energy_transfer): a regular colatitude–longitude grid (two
`AbstractMatrix` velocity components) routes to the FastSphericalHarmonics extension; scattered
points (velocity-component vectors + `(θ, φ)` coordinates) route to the NUFSHT extension.

# Fields
- `radius::T`: sphere radius `a` (default `1.0`).
"""
struct DivergentSphericalTransferMethod{T<:Real} <: AbstractEnergyTransferMethod
    radius::T
end
DivergentSphericalTransferMethod(; radius=1.0) = DivergentSphericalTransferMethod(radius)


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

"""
    SmoothBands(centers; logwidth=0.6)

Graded (smooth) spectral bands for band-to-band transfer `T(K,Q)` (Eyink & Aluie 2009), as an
alternative to the sharp shells of [`AbstractShellBinning`](@ref). Each band `n` weights a mode at
coordinate `κ` by a log-Gaussian `exp(−(ln(κ/centers[n]))² / (2·logwidth²))`, renormalized across
bands to a partition of unity (`Σ_n w_n(κ) = 1`) so the smooth bands conserve and reduce to the
band-summed transfer spectrum. Smaller `logwidth` → sharper, more shell-like bands.

# Fields
- `centers`: band-center wavenumbers (monotonically increasing, all > 0).
- `logwidth`: Gaussian width in `ln κ` (dimensionless); default `0.6` (≈ one octave overlap).
"""
struct SmoothBands{V<:AbstractVector, T}
    centers::V
    logwidth::T
end
SmoothBands(centers::AbstractVector; logwidth=0.6) = SmoothBands(centers, float(logwidth))

# ---------------------------------------------------------------------------
# Dealiasing strategy — how the quadratic-product aliasing error is removed
# ---------------------------------------------------------------------------
#
# A pseudospectral product of two N-mode fields generates wavenumbers up to 2× the maximum, which
# alias back onto the resolved band. Two standard cures (Canuto et al. 2006; Orszag 1971):
#   • Orszag 2/3 truncation: zero modes with |k_d| ≥ N_d/3 in the INPUTS and output. Exact on the
#     retained band |k|<N/3, but discards N/3 ≤ |k| < N/2 — the top of the field's spectrum.
#   • 3/2 zero-padding: embed the N-mode field in a (3N/2)-point grid, form the product there
#     (no aliasing), transform back and truncate. Exact for the quadratic term over ALL modes to
#     Nyquist — nothing is discarded.

"""
    AbstractDealiasing

Strategy for removing aliasing from the pseudospectral quadratic product, passed as the
`dealiasing` keyword. Subtypes: [`NoDealiasing`](@ref), [`OrszagTwoThirds`](@ref) (the default),
and [`PaddedThreeHalves`](@ref) (exact 3/2 zero-padding).
"""
abstract type AbstractDealiasing end

"""
    NoDealiasing <: AbstractDealiasing

No dealiasing — the raw pseudospectral product, aliasing included.
"""
struct NoDealiasing <: AbstractDealiasing end

"""
    OrszagTwoThirds <: AbstractDealiasing

Orszag 2/3-rule truncation: zero modes with `|k_d| ≥ N_d/3` in the inputs and output. Exact on the
retained band `|k| < N/3`; the default `dealiasing`.
"""
struct OrszagTwoThirds <: AbstractDealiasing end

"""
    PaddedThreeHalves <: AbstractDealiasing

Exact 3/2 zero-padding: form the quadratic product on a `(3N/2)`-point grid so no aliasing reaches
the resolved band, then truncate back to `N`. Exact for the quadratic nonlinear term over every
mode up to Nyquist (nothing discarded), at ~`(3/2)^D` higher transform cost. Requires FFTW for the
fast path.
"""
struct PaddedThreeHalves <: AbstractDealiasing end

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
# Transform (spectral) backends — WHICH transform (direct / FFT / NUFFT / SHT / NUFSHT). Orthogonal
# to the EXECUTION axis (serial / threaded / distributed / MPI / GPU), which lives in the sibling
# `Backends` module: a computation picks one of each (e.g. FFT transform + threaded mediator loop, or
# a NUFSHT transform + MPI reduction). Keeping "what" and "where" separate avoids conflating them.
# ---------------------------------------------------------------------------

"""
    AbstractSpectralBackend

Abstract supertype for *transform* backends: how physical↔spectral coefficients and the
pseudospectral nonlinear term are computed. Orthogonal to [`AbstractExecutionBackend`](@ref FlowInvariantTransfer.Backends.AbstractExecutionBackend).
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

# ---------------------------------------------------------------------------
# Transform-backend geometry classification + validation
# ---------------------------------------------------------------------------
# Each transform backend targets one (geometry, sampling). The Fourier-coefficient Cartesian
# diagnostics accept only the uniform-Cartesian transforms — the DirectSum reference and FFT — because
# their input already *is* a uniform-Cartesian Fourier field. Passing a scattered (NUFFT) or spherical
# (SHT/NUFSHT) backend to them is a geometry mismatch and must raise a clear error, never a bare
# `MethodError` and never a silent misroute to FFT.

"""
    spectral_geometry(backend) -> Symbol

The (geometry, sampling) a transform backend targets: `:any` (the DirectSum reference works on any
geometry), `:cartesian_uniform`, `:cartesian_scattered`, `:spherical_uniform`, `:spherical_scattered`.
"""
spectral_geometry(::DirectSumBackend) = :any
spectral_geometry(::FFTBackend)       = :cartesian_uniform
spectral_geometry(::NUFFTBackend)     = :cartesian_scattered
spectral_geometry(::SHTBackend)       = :spherical_uniform
spectral_geometry(::NUFSHTBackend)    = :spherical_scattered

"""
    require_coefficient_spectral(spectral) -> spectral

Assert that `spectral` is a valid transform for a Fourier-coefficient Cartesian diagnostic (input is a
uniform-Cartesian Fourier field). Returns `spectral` for [`DirectSumBackend`](@ref)/[`FFTBackend`](@ref);
raises a clear geometry-mismatch error for the scattered/spherical backends directing the caller to the
right entry point.
"""
require_coefficient_spectral(spectral::Union{DirectSumBackend, FFTBackend}) = spectral
require_coefficient_spectral(::NUFFTBackend) = throw(ArgumentError(
    "NUFFTBackend is a scattered-Cartesian transform: it acts on a physical field sampled at scattered " *
    "points, not on Fourier coefficients. Pass the physical field and its scatter coordinates to the " *
    "physical-space entry, e.g. `calculate_spectral_flux(velocity_fields, scatter_coords; spectral=NUFFTBackend())`."))
require_coefficient_spectral(::Union{SHTBackend, NUFSHTBackend}) = throw(ArgumentError(
    "SHTBackend and NUFSHTBackend are spherical transforms. Use `calculate_spherical_transfer` for transfer " *
    "on the sphere (regular grid → SHTBackend, scattered points → NUFSHTBackend). The Cartesian flux " *
    "diagnostics take FFTBackend (uniform grid) or NUFFTBackend (scattered, via the physical-space entry)."))

# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

"""
    SpectralFluxResult{KS, V}

Result of a spectral energy flux computation.

# Fields
- `k_shells::KS`: Representative wavenumber for each shell (midpoint of bin edges).
- `transfer_spectrum::V`: T(k) — energy transfer rate per shell.
- `flux::V`: Π(K) = +cumsum(T(k)) — cumulative energy flux (Π>0 forward/down-scale cascade,
  Π<0 inverse; Alexakis–Biferale 2018, THEORY.md §0.5).

Parametric with no element-type bound (works with Float32/Float64/Dual/Unitful, etc.). `k_shells`
(host-side shell wavenumbers) is parametrised separately from the `transfer_spectrum`/`flux` data so
a device computation can return device data while `k_shells` stays a host vector.
"""
struct SpectralFluxResult{KS<:AbstractVector, V<:AbstractVector}
    k_shells::KS
    transfer_spectrum::V
    flux::V
end
SpectralFluxResult(k, T, f) = SpectralFluxResult{typeof(k), typeof(T)}(k, T, f)

"""
    SphericalTransferResult{V<:AbstractVector}

Result of a spherical spectral energy/enstrophy transfer ([`SphericalTransferMethod`](@ref)),
indexed by spherical-harmonic degree `l = 0…lmax`.

# Fields
- `degrees::V`: the degrees `l`.
- `energy_transfer::V`: `T_E(l)` — nonlinear kinetic-energy transfer into degree `l`; `Σ_l T_E ≈ 0`.
- `enstrophy_transfer::V`: `T_Z(l)` — enstrophy transfer into degree `l`; `Σ_l T_Z ≈ 0`.
- `energy_flux::V`: `Π_E(L) = -Σ_{l≤L} T_E(l)` — cumulative up-degree energy flux.
- `enstrophy_flux::V`: `Π_Z(L) = -Σ_{l≤L} T_Z(l)`.
"""
struct SphericalTransferResult{V<:AbstractVector}
    degrees::V
    energy_transfer::V
    enstrophy_transfer::V
    energy_flux::V
    enstrophy_flux::V
end

"""
    DivergentSphericalTransferResult{V<:AbstractVector}

Result of the divergent horizontal kinetic-energy spectral transfer
([`DivergentSphericalTransferMethod`](@ref)), indexed by spherical-harmonic degree `l = 0…lmax`.

# Fields
- `degrees::V`: the degrees `l`.
- `energy_transfer::V`: total horizontal-KE transfer `T(l) = T_rot(l) + T_div(l)` into degree `l`;
  the skew-symmetric (energy-conserving) advection makes `Σ_l T ≈ 0`.
- `energy_flux::V`: `Π(L) = -Σ_{l≤L} T(l)` — cumulative up-degree KE flux.
- `rotational_transfer::V`: rotational-channel transfer `T_rot(l)` — projection of the advection onto
  the toroidal (streamfunction `ψ`) part of the velocity.
- `divergent_transfer::V`: divergent-channel transfer `T_div(l)` — projection onto the spheroidal
  (velocity-potential `χ`) part.
- `rotational_flux::V`, `divergent_flux::V`: cumulative fluxes `-Σ_{l≤L} T_rot`, `-Σ_{l≤L} T_div`.

Only the total is conserved (`Σ_l T ≈ 0`); the two channels exchange energy, so `Σ_l T_rot` and
`Σ_l T_div` are individually nonzero (equal and opposite up to the total).
"""
struct DivergentSphericalTransferResult{V<:AbstractVector}
    degrees::V
    energy_transfer::V
    energy_flux::V
    rotational_transfer::V
    divergent_transfer::V
    rotational_flux::V
    divergent_flux::V
end

"""
    CompressibleFluxResult{KS, TS, FL, CH, PD}

Result of a compressible kinetic-energy spectral-transfer computation (Singh–Tiwari–Sharma–Verma
2025; see THEORY.md §0.5). The transfer is momentum-weighted (`v = ρu`), so unlike the incompressible
diagnostics it needs the density field and — for the KE↔internal-energy exchange — the pressure.

# Fields
- `k_shells::KS`: representative wavenumber per shell.
- `transfer_spectrum::TS`: `T_u(k)` — net momentum-weighted KE transfer into shell `k` (energy *gain*
  rate; sign is opposite the incompressible loss convention). Conserves total KE: `Σ_k T_u(k) ≈ 0`.
- `flux::FL`: `Π(K)` — cumulative flux, `Π(K) = Σ_{k>K} T_u(k)`.
- `channels::CH`: rotational/compressive (Helmholtz `u = u_R + u_C`) flux channels as a `NamedTuple`
  `(rotational, compressive, rot_to_comp, comp_to_rot)`, or `nothing` if not requested.
- `pressure_dilatation::PD`: KE↔IE conversion `(rotational = Q_{I,R}(k), compressive = Q_{I,C}(k))`
  as a `NamedTuple`, or `nothing` if no pressure field was supplied.

Each array field carries its own type parameter (element-type/container generic — no shared or
`<:AbstractVector`-bounded param); optional `CH`/`PD` resolve to `Nothing` or a concrete `NamedTuple`.
"""
struct CompressibleFluxResult{KS, TS, FL, CH, PD}
    k_shells::KS
    transfer_spectrum::TS
    flux::FL
    channels::CH
    pressure_dilatation::PD
end

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
- `stress_tensor::T`: τ̄ᵢʲ (component-indexed array, e.g. `(Nx,Ny,2,2)`).
- `strain_rate::T`: S̄ᵢʲ (same array type as `stress_tensor`).

Returned instead of `CoarseGrainingFluxResult` when `return_diagnostics=true`.
The tensor diagnostics carry their own type parameter `T` (higher-rank than the scalar
`flux_field::A`); all fields are always present — no `Union{Nothing,...}` type instability.
"""
struct CoarseGrainingFluxResultWithDiagnostics{S, A<:AbstractArray, T<:AbstractArray}
    filter_scale::S
    flux_field::A
    mean_flux::S
    stress_tensor::T
    strain_rate::T
end
CoarseGrainingFluxResultWithDiagnostics(s, a, m, st, sr) =
    CoarseGrainingFluxResultWithDiagnostics{typeof(s), typeof(a), typeof(st)}(s, a, m, st, sr)

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
