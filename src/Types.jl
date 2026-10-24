module Types

using SpectralBackends: SpectralBackends
using ComputationalBackends: ComputationalBackends

export AbstractEnergyTransferMethod, SpectralFluxMethod, CoarseGrainingFluxMethod, ShellToShellTransferMethod, ModeToModeTransferMethod, TriadicOrthogonalDecompositionMethod, SphericalTransferMethod, DivergentSphericalTransferMethod
export AbstractInvariant, KineticEnergy, Helicity, Enstrophy, PassiveScalar
export AbstractFieldDecomposition, NoDecomposition, HelmholtzDecomposition, RotationalDecomposition, DivergentDecomposition, HelicalDecomposition, ToroidalPoloidalDecomposition
export AbstractFilter, SharpSpectralFilter, GaussianFilter, TopHatFilter
export AbstractInactiveCells, ZeroFill, Exclude
export AbstractShellBinning, LinearBinning, LogarithmicBinning, DyadicBinning, CustomBinning
export AbstractShellGeometry, ShellMagnitude, IsotropicShells, PerpendicularShells, ParallelShells
export SmoothBands
export AbstractDealiasing, NoDealiasing, OrszagTwoThirds, PaddedThreeHalves
export AbstractNonlinearForm, Advective, Rotational
export SpectralFluxResult, CompressibleFluxResult, CoarseGrainingFluxResult, CoarseGrainingFluxResultWithDiagnostics, ShellToShellResult, ModeToModeTriadResult, TriadicOrthogonalDecompositionResult, SphericalTransferResult, DivergentSphericalTransferResult

# ---------------------------------------------------------------------------
# Execution-backend resolution
# ---------------------------------------------------------------------------
# Concrete execution backends pass through `ComputationalBackends.resolve_backend`; `AutoBackend`
# resolves to `ThreadedBackend` when the OhMyThreads extension is loaded and Julia has more than one
# thread, else to `SerialBackend`.
resolve_execution(execution::ComputationalBackends.AbstractExecutionBackend) = ComputationalBackends.resolve_backend(execution)
function resolve_execution(::ComputationalBackends.AutoBackend)
    if Threads.nthreads() > 1 &&
       Base.get_extension(parentmodule(@__MODULE__), :FlowInvariantTransferOhMyThreadsExt) !== nothing
        return ComputationalBackends.ThreadedBackend()
    end
    return ComputationalBackends.SerialBackend()
end

# The error a batch `entry` raises when the method for `be` is not loaded: the threaded, Distributed and
# MPI methods live in the extensions of those packages.
function batch_unavailable(entry::AbstractString, be::ComputationalBackends.AbstractExecutionBackend)
    pkg = be isa ComputationalBackends.AbstractThreadedBackend ? "OhMyThreads" :
          be isa ComputationalBackends.AbstractDistributedBackend ? "Distributed" :
          be isa ComputationalBackends.AbstractMPIBackend ? "MPI" : nothing
    pkg === nothing && throw(ArgumentError(
        "$entry runs on SerialBackend, ThreadedBackend, GPUBackend, DistributedBackend and MPIBackend; " *
        "got execution = $(typeof(be))."))
    throw(ArgumentError("$entry on execution = $(typeof(be)) needs `using $pkg`."))
end

# ---------------------------------------------------------------------------
# Transform resolution
# ---------------------------------------------------------------------------
"""
    resolve_transform(transform) -> AbstractSpectralBackend

Concrete transform backends pass through. [`SpectralBackends.AutoSpectralBackend`](@ref), the
default `transform` of every Cartesian coefficient diagnostic, resolves to the FFT backend
(`O(Nᴰ log N)`) when the FFTW extension is loaded, else to the dependency-free direct sum (`O(N²ᴰ)`)
with a one-time warning.
"""
resolve_transform(transform::SpectralBackends.AbstractSpectralBackend) = transform
function resolve_transform(::SpectralBackends.AbstractAutoSpectralBackend)
    if _fft_backend_available()
        return SpectralBackends.FFTSpectralBackend()
    end
    @warn("No FFT backend loaded: falling back to the O(N^2D) direct-sum reference, which is " *
          "practical only on tiny grids. Run `using FFTW` for the O(N^D log N) path.", maxlog = 1)
    return SpectralBackends.DirectSumSpectralBackend()
end

"""
    _is_device(x) -> Bool

`true` for a device array. The GPUArraysCore extension adds the method for
`GPUArraysCore.AbstractGPUArray`; every host array, `Array` or not, is `false`.
"""
_is_device(::Any) = false

"""
    _to_execution(execution, x) -> array

`x` in the memory `execution` computes in: `x` itself on a host backend or where it already lives, a
copy on the backend's device otherwise. The KernelAbstractions extension adds the `GPUBackend` method.
"""
_to_execution(::ComputationalBackends.AbstractExecutionBackend, x) = x

"""
    _fields_to_execution(execution, field, advecting) -> (field′, advecting′)

The advected field and the advecting velocity in `execution`'s memory, one copy when they are the same
array.
"""
function _fields_to_execution(execution::ComputationalBackends.AbstractExecutionBackend, field, advecting)
    f = _to_execution(execution, field)
    return f, advecting === field ? f : _to_execution(execution, advecting)
end

"""
    _staging_buffer(execution, x) -> array or nothing

A copy of `x` in `execution`'s memory, the buffer [`_staged!`](@ref) copies later inputs shaped like `x`
into; `nothing` when `x` already lives there.
"""
function _staging_buffer(execution::ComputationalBackends.AbstractExecutionBackend, x)
    y = _to_execution(execution, x)
    return y === x ? nothing : y
end

"""
    _staged!(buf, x) -> array

`x` in the memory of `buf`, the buffer an input is staged through: `x` itself when it lives there, else
copied into `buf`. A `nothing` buffer stages nothing.
"""
_staged!(::Nothing, x) = x
_staged!(buf, x) = (_is_device(buf) && !_is_device(x)) ? copyto!(buf, x) : x

"""
    _FFT_BACKEND_LOADED

Set to `true` by the FFTW extension's `__init__` and read by [`resolve_transform`](@ref) on every
call. The extension signals through a flag because an extension that overwrites a parent method
fails to precompile.
"""
const _FFT_BACKEND_LOADED = Ref(false)

_fft_backend_available() = _FFT_BACKEND_LOADED[]

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
# Inactive cells of a masked spherical grid
# ---------------------------------------------------------------------------

"""
    AbstractInactiveCells

How the spherical transfer on a masked `FlowGeometries` grid reads the grid's inactive cells, passed as
`inactive`: [`ZeroFill`](@ref) or [`Exclude`](@ref). What an inactive cell holds, `NaN` included, never
enters the transfer.
"""
abstract type AbstractInactiveCells end

"""
    ZeroFill <: AbstractInactiveCells

Inactive cells count as zero: the transfer is that of the field extended by zero over the sphere,
computed on every node of the grid. The default.
"""
struct ZeroFill <: AbstractInactiveCells end

"""
    Exclude <: AbstractInactiveCells

Inactive cells are left out: the field's coefficients are the least-squares fit to the active cells
alone, and the transfer is that of the band-limited field the fit defines, computed on every node of
the grid.
"""
struct Exclude <: AbstractInactiveCells end

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
  **stretching**, so enstrophy is **not** conserved (`Σ_k T_Ω ≠ 0`: net production).

Available in 2D and 3D in the spectral flux, shell-to-shell and mode-to-mode diagnostics.
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
3D.

# A family of canonical invariants
Other quadratic invariants are advected by the velocity like a passive scalar, so their cross-scale
transfer is computed by this same path (pass the field as the "scalar"):
- **Buoyancy / available-potential-energy variance** `½⟨b²⟩` (APE `= ½⟨b²⟩/N²`) in the
  Boussinesq system. The `−N²w` term is a separate KE↔APE *conversion* source, so the variance
  *cascade* is the scalar transfer of `b`.
- **QG potential enstrophy** `½⟨q²⟩` with PV `q = ∇²ψ + βy` advected by the geostrophic velocity.

# References
- Obukhov (1949); Corrsin (1951); Batchelor (1959); QG: Charney (1971);
  stratified APE: Lindborg (2006).
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
Requires Fourier-space velocity data on a uniform periodic grid. The transforms run through FFTW
(`O(Nᴰ log N)`) when it is loaded, else through the `O(N²ᴰ)` direct-sum reference.
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

Compute the directed shell-to-shell transfer matrix T(n,m), the rate at which shell S_n
gives energy to shell S_m through the nonlinear advection term.

# Fields
- `binning::B`: Shell binning strategy.

# Notes
The antisymmetry property T(n,m) = −T(m,n) holds when the mediator velocity is
the full field u (Verma et al. 2002).
"""
struct ShellToShellTransferMethod{B<:AbstractShellBinning} <: AbstractEnergyTransferMethod
    binning::B
end

"""
    ModeToModeTransferMethod{I<:AbstractInvariant} <: AbstractEnergyTransferMethod

Compute the **mode-to-mode transfer** `S(k|p)` between stored coefficients — the rate at which the
modes of slot `k` give energy (or another invariant) to the slot field `u_p`, mediated by `q = k − p`:

    S(k|p) = Re{ û*(k) · N̂_p(k) },   N̂_p = FFT[(u·∇)u_p],   u_p(x) = w(p)·Re{û(p) e^{ip·x}},

with `w` the slot's fold weight (see [`ModeToModeTriadResult`](@ref)). It reduces to the
shell-to-shell matrix and the spectral transfer `T(k)` under summation.

# Fields
- `invariant::I`: Which quadratic invariant to accumulate (default `KineticEnergy()`).

# Cost
One nonlinear-term evaluation per slot `p` and an `O(N^{2D})` result tensor.

# References
- Dar, Verma & Eswaran (2001); Verma (2004 review, 2019 book).
"""
struct ModeToModeTransferMethod{I<:AbstractInvariant} <: AbstractEnergyTransferMethod
    invariant::I
end
ModeToModeTransferMethod(; invariant=KineticEnergy()) = ModeToModeTransferMethod(invariant)

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

both conserving (Σ_l T = 0).

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
The advection is taken in the skew-symmetric form `A = (u·∇)u + ½u(∇·u)` of Augier & Lindborg (2013,
Appendix A; Eq. 35 of arXiv:1211.0607), written in Lamb form as `A = ∇(½|u|²) + ζ(k̂×u) + ½δu` with
vorticity `ζ = k̂·(∇×u)` and divergence `δ = ∇·u`. The nonlinear KE transfer of degree `l`,
`−dE_l/dt` from that advection, is the vector-harmonic projection

    T(l) = Σ_m Re{ û*_lm · Â_lm}   (spin-1 vector-harmonic coefficients),

split by the toroidal (rotational) / spheroidal (divergent) parts of `û` into `T = T_rot + T_div`.
`∫u·A = 0`, so total KE is conserved, `Σ_l T(l) ≈ 0`; the rotational and divergent channels exchange
energy and are not conserved one by one. References: Augier & Lindborg (2013); Burgess, Erler &
Shepherd (2013).

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

Shells of width `Δk`: edges `0, Δk, 2Δk, …` up to the largest wavenumber coordinate, shell `n`
covering `[(n−1)Δk, nΔk)` around its centre `(n−½)Δk`, and the last shell closed; the flux `Π[n]` is
the flux through `nΔk`. With `Δk` the fundamental wavenumber, shell 1 holds the mean mode alone and
`|k| = nΔk` falls in shell `n + 1`.

# Fields
- `Δk`: Shell width in physical wavenumber units.
"""
struct LinearBinning{T} <: AbstractShellBinning
    Δk::T
end

"""
    LogarithmicBinning(k₀, λ) <: AbstractShellBinning

Geometrically-spaced shells: edges `0, k₀, k₀λ, k₀λ², …`, the first shell `[0, k₀)`.

# Fields
- `k₀`: Lower edge of the first geometric shell (> 0).
- `λ`: Ratio between consecutive shell edges (> 1); λ = 2 gives dyadic.
"""
struct LogarithmicBinning{T} <: AbstractShellBinning
    k₀::T
    λ::T
end
LogarithmicBinning(k₀, λ) = LogarithmicBinning(promote(k₀, λ)...)

"""
    DyadicBinning(k₀) <: AbstractShellBinning

Dyadic (octave) shells: edges `0, k₀, 2k₀, 4k₀, …`.  Equivalent to `LogarithmicBinning(k₀, 2.0)`.

# Fields
- `k₀`: Lower edge of the first octave shell (> 0).
"""
struct DyadicBinning{T} <: AbstractShellBinning
    k₀::T
end

"""
    CustomBinning(edges) <: AbstractShellBinning

User-specified shell edges.  Shell n covers wavenumbers in [edges[n], edges[n+1]), and the last
shell includes `edges[end]`; modes outside `[edges[1], edges[end]]` belong to no shell.

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
# Dealiasing strategy
# ---------------------------------------------------------------------------

"""
    AbstractDealiasing

Strategy for removing aliasing from the pseudospectral quadratic product, passed as the
`dealiasing` keyword. A product of two `N`-mode fields generates wavenumbers up to twice the
maximum, which alias back onto the resolved band (Orszag 1971; Canuto et al. 2006). Subtypes:
[`NoDealiasing`](@ref), [`OrszagTwoThirds`](@ref) (the default), and [`PaddedThreeHalves`](@ref)
(exact 3/2 zero-padding).
"""
abstract type AbstractDealiasing end

"""
    NoDealiasing <: AbstractDealiasing

No dealiasing — the raw pseudospectral product, aliasing included.
"""
struct NoDealiasing <: AbstractDealiasing end

"""
    OrszagTwoThirds <: AbstractDealiasing

Orszag 2/3-rule truncation: keep modes with `|m_d| ≤ ⌊(N_d−1)/3⌋` along every axis, in the inputs and
the output (`FlowTransformBindings.dealias_cutoff`). Exact on the retained band, and discards the top
third of the spectrum. The default `dealiasing`.
"""
struct OrszagTwoThirds <: AbstractDealiasing end

"""
    PaddedThreeHalves <: AbstractDealiasing

Exact 3/2 zero-padding: form the quadratic product on the grid of
`FlowTransformBindings.padded_length(N)` points per axis, the smallest past `3N/2` on which no alias
reaches the resolved band, then truncate back to `N`. Exact for the quadratic nonlinear term over every
mode up to Nyquist (nothing discarded), at ~`(3/2)^D` the transform cost of the unpadded term, on the
FFT and the direct-sum transforms alike.
"""
struct PaddedThreeHalves <: AbstractDealiasing end

"""
    AbstractNonlinearForm

How a velocity's self-advection is evaluated for its spectral flux, passed as the `form` keyword:
[`Advective`](@ref) (the default) or [`Rotational`](@ref).
"""
abstract type AbstractNonlinearForm end

"""
    Advective <: AbstractNonlinearForm

`(u·∇)u` itself: `D` velocity components and their `D²` gradients synthesised and the product
transformed back, `D + D² + D` transforms (15 in 3-D, 8 in 2-D). Valid for any field. The default `form`.
"""
struct Advective <: AbstractNonlinearForm end

"""
    Rotational <: AbstractNonlinearForm

`ω × u` in place of `(u·∇)u`: the velocity and its vorticity synthesised and their cross product
transformed back, 9 transforms in 3-D and 5 in 2-D (`ω ẑ × u`). `(u·∇)u = ω × u + ∇(½|u|²)`, and under
the 2/3 rule the two differ on the kept modes by `i k (½|u|²)^`, whose kinetic-energy, enstrophy and
helicity transfers vanish where `k·û = 0`; for a divergence-free velocity this form gives the same
fluxes. For a field with divergence it gives the transfer of `ω × u`, a different quantity.
"""
struct Rotational <: AbstractNonlinearForm end

# ---------------------------------------------------------------------------
# Shell geometry
# ---------------------------------------------------------------------------

"""
    AbstractShellGeometry

Abstract supertype selecting the wavenumber coordinate the shells partition (isotropic `|k|`,
or an anisotropic projection like `k_⊥`/`k_∥`). Orthogonal to the binning *spacing*
([`AbstractShellBinning`](@ref)); the geometry changes only the scalar each mode is binned by. The
anisotropic fluxes `Π(k_⊥)`, `Π(k_∥)` of rotating/stratified flow follow Alexakis & Biferale (2018,
§IV).
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
# Transform-backend geometry classification + validation
# ---------------------------------------------------------------------------

"""
    spectral_geometry(transform) -> Symbol

The (geometry, sampling) a transform targets: `:any` (the DirectSum reference works on any
geometry), `:cartesian_uniform`, `:cartesian_scattered`, `:spherical_uniform`, `:spherical_scattered`.
"""
spectral_geometry(::SpectralBackends.DirectSumSpectralBackend) = :any
spectral_geometry(::SpectralBackends.FFTSpectralBackend)       = :cartesian_uniform
spectral_geometry(::SpectralBackends.AbstractNonUniformFastFourierTransformSpectralBackend) = :cartesian_scattered
spectral_geometry(::SpectralBackends.FSHTSpectralBackend)      = :spherical_uniform
spectral_geometry(::SpectralBackends.NUFSHTSpectralBackend)    = :spherical_scattered

"""
    require_coefficient_transform(transform) -> transform

Assert that `transform` is valid for a Fourier-coefficient Cartesian diagnostic (input is a
uniform-Cartesian Fourier field). Returns `transform` for the Auto, direct-sum and FFT backends; throws
an `ArgumentError` naming the right entry point for the scattered and spherical backends.
"""
require_coefficient_transform(transform::Union{SpectralBackends.AbstractAutoSpectralBackend,
                                               SpectralBackends.DirectSumSpectralBackend,
                                               SpectralBackends.FFTSpectralBackend}) = transform
require_coefficient_transform(::SpectralBackends.AbstractNonUniformFastFourierTransformSpectralBackend) = throw(ArgumentError(
    "A NUFFT backend is a scattered-Cartesian transform: it acts on a physical field sampled at scattered " *
    "points, not on Fourier coefficients. Pass the physical field and its scatter coordinates to the " *
    "physical-space entry, e.g. `to_spectral(velocity_fields, scatter_coords, ms; transform, Ls)` or the " *
    "4-positional `calculate_energy_transfer(method, velocity_fields, scatter_coords, ms; transform, Ls)`."))
require_coefficient_transform(::Union{SpectralBackends.FSHTSpectralBackend, SpectralBackends.NUFSHTSpectralBackend}) = throw(ArgumentError(
    "FSHTSpectralBackend and NUFSHTSpectralBackend are spherical transforms. Use `calculate_spherical_transfer` for transfer " *
    "on the sphere (regular grid → FSHTSpectralBackend, scattered points → NUFSHTSpectralBackend). The Cartesian flux " *
    "diagnostics take FFTSpectralBackend (uniform grid) or a NUFFT backend (scattered, via the physical-space entry)."))

# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

"""
    SpectralFluxResult{KS, E, V}

Result of a spectral energy flux computation over the shells `[shell_edges[n], shell_edges[n+1])`.

# Fields
- `k_shells::KS`: Representative wavenumber for each shell (midpoint of its edges).
- `shell_edges::E`: The shell boundaries (length = number of shells + 1).
- `transfer_spectrum::V`: `T(k_n) = −dE_n/dt` from the nonlinear term, `E_n` the energy of shell `n`:
  the shell's content, a sum over its modes. `T(k_n)/(shell_edges[n+1] − shell_edges[n])` is the
  transfer per unit wavenumber.
- `flux::V`: `Π[n] = Σ_{m≤n} T(k_m)`, the flux through the wavenumber `shell_edges[n+1]` (Π > 0
  forward/down-scale cascade, Π < 0 inverse; Alexakis–Biferale 2018).

`k_shells` and `shell_edges` are host vectors whatever array type the transfer is computed on.
"""
struct SpectralFluxResult{KS<:AbstractVector, E<:AbstractVector, V<:AbstractVector}
    k_shells::KS
    shell_edges::E
    transfer_spectrum::V
    flux::V
end
SpectralFluxResult(k, e, T, f) = SpectralFluxResult{typeof(k), typeof(e), typeof(T)}(k, e, T, f)

"""
    SphericalTransferResult{V<:AbstractVector, T<:Real}

Result of a spherical spectral energy/enstrophy transfer ([`SphericalTransferMethod`](@ref)),
indexed by spherical-harmonic degree `l = 0…lmax`.

# Fields
- `degrees::V`: the degrees `l`.
- `energy_transfer::V`: `T_E(l) = −dE_l/dt` from the advection; `Σ_l T_E ≈ 0`.
- `enstrophy_transfer::V`: `T_Z(l) = −dZ_l/dt` from the advection; `Σ_l T_Z ≈ 0`.
- `energy_flux::V`: `Π_E(L) = Σ_{l≤L} T_E(l)` — cumulative energy flux, positive toward higher degree.
- `enstrophy_flux::V`: `Π_Z(L) = Σ_{l≤L} T_Z(l)`.
- `converged::Bool`, `iterations::Int`, `residual::T`: at scattered points the coefficients come from
  least-squares fits; `converged` is whether every fit reached `rtol`, and `iterations` and `residual`
  (the relative normal-equation residual `‖A†r‖/‖A†f‖`) are the largest over the fits. A transform
  exact on the nodes (a regular grid, or quadrature weights) runs no fit: `true`, `0`, `0`.
"""
struct SphericalTransferResult{V<:AbstractVector, T<:Real}
    degrees::V
    energy_transfer::V
    enstrophy_transfer::V
    energy_flux::V
    enstrophy_flux::V
    converged::Bool
    iterations::Int
    residual::T
end

"""
    DivergentSphericalTransferResult{V<:AbstractVector, T<:Real}

Result of the divergent horizontal kinetic-energy spectral transfer
([`DivergentSphericalTransferMethod`](@ref)), indexed by spherical-harmonic degree `l = 0…lmax`.

# Fields
- `degrees::V`: the degrees `l`.
- `energy_transfer::V`: `T(l) = T_rot(l) + T_div(l) = −dE_l/dt` of the horizontal KE from the
  advection; the skew-symmetric (energy-conserving) advection makes `Σ_l T ≈ 0`.
- `energy_flux::V`: `Π(L) = Σ_{l≤L} T(l)` — cumulative KE flux, positive toward higher degree.
- `rotational_transfer::V`: rotational-channel transfer `T_rot(l)` — projection of the advection onto
  the toroidal (streamfunction `ψ`) part of the velocity.
- `divergent_transfer::V`: divergent-channel transfer `T_div(l)` — projection onto the spheroidal
  (velocity-potential `χ`) part.
- `rotational_flux::V`, `divergent_flux::V`: cumulative fluxes `Σ_{l≤L} T_rot`, `Σ_{l≤L} T_div`.
- `converged::Bool`, `iterations::Int`, `residual::T`: the least-squares fits at scattered points, as
  in [`SphericalTransferResult`](@ref).

Only the total is conserved (`Σ_l T ≈ 0`); the two channels exchange energy, so `Σ_l T_rot` and
`Σ_l T_div` are individually nonzero (equal and opposite up to the total).
"""
struct DivergentSphericalTransferResult{V<:AbstractVector, T<:Real}
    degrees::V
    energy_transfer::V
    energy_flux::V
    rotational_transfer::V
    divergent_transfer::V
    rotational_flux::V
    divergent_flux::V
    converged::Bool
    iterations::Int
    residual::T
end

"""
    CompressibleFluxResult{KS, ES, TS, FL, CH, PD}

Result of a compressible kinetic-energy spectral-transfer computation (Singh–Tiwari–Sharma–Verma
2025). The transfer is momentum-weighted (`v = ρu`), so it needs the density field and, for the
KE↔internal-energy exchange, the pressure.

# Fields
- `k_shells::KS`: representative wavenumber per shell.
- `shell_edges::ES`: the shell boundaries (length = number of shells + 1).
- `transfer_spectrum::TS`: `T_u(k) = −dE_u(k)/dt` of the momentum-weighted KE from the nonlinear
  terms, the sign of the incompressible transfer. Conserves total KE where the products are
  dealiased: `Σ_k T_u(k) ≈ 0`.
- `flux::FL`: `Π[n] = Σ_{m≤n} T_u(k_m)`, the flux through `shell_edges[n+1]`; its last entry is
  `Σ_k T_u`, the residual of conservation.
- `channels::CH`: rotational/compressive (Helmholtz `u = u_R + u_C`) flux channels as a `NamedTuple`
  `(rotational, compressive, rot_to_comp, comp_to_rot)`, or `nothing` if not requested; `rot_to_comp`
  is the flux the rotational part gives the compressive part, `comp_to_rot` the reverse.
- `pressure_dilatation::PD`: KE↔IE conversion `(rotational = Q_{I,R}(k), compressive = Q_{I,C}(k))`
  as a `NamedTuple`, or `nothing` if no pressure field was supplied.
"""
struct CompressibleFluxResult{KS, ES, TS, FL, CH, PD}
    k_shells::KS
    shell_edges::ES
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
    CoarseGrainingFluxResultWithDiagnostics{S, A, T}

Result of a coarse-graining energy flux computation including stress/strain diagnostics, returned
when `return_diagnostics=true`.

# Fields
- `filter_scale::S`: Filter scale ℓ used.
- `flux_field::A`: Π_ℓ(x) pointwise energy flux field.
- `mean_flux::S`: Area-weighted spatial mean ⟨Π_ℓ⟩.
- `stress_tensor::T`: τ̄ᵢʲ (component-indexed array, e.g. `(Nx,Ny,2,2)`).
- `strain_rate::T`: S̄ᵢʲ (same array type as `stress_tensor`).
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
    ShellToShellResult{V, W, M}

Result of a shell-to-shell energy transfer computation.

# Fields
- `shell_centers::V`: Representative wavenumber for each shell.
- `shell_edges::V`: Shell boundary wavenumbers (length = N_shells + 1).
- `transfer_matrix::M`: T(n,m) — N_shells × N_shells matrix; T[n,m] is the rate at which shell n
  gives energy to shell m.
- `net_transfer::W`: Σ_m T(n,m) = −dE_n/dt, equal to the spectral transfer `T(k_n)`.

`V` carries the wavenumber axes' element type; `W` and `M` carry the field's.
"""
struct ShellToShellResult{V<:AbstractVector, W<:AbstractVector, M<:AbstractMatrix}
    shell_centers::V
    shell_edges::V
    transfer_matrix::M
    net_transfer::W
end
ShellToShellResult(c, e, T, n) =
    ShellToShellResult{typeof(c), typeof(n), typeof(T)}(c, e, T, n)

"""
    ModeToModeTriadResult{I, KS, W, A, S}

Result of a mode-resolved triad transfer, indexed by the stored coefficients ("slots") of the spectral
layout.

Slot `k` stands for `w(k)` modes of the full spectrum (`weights`): 1 everywhere on a full layout; on a
half layout 2 on the interior of the halved axis, where the slot also stands for its unstored mirror
`−k`, and 1 on its `k₁ = 0` and Nyquist planes. Its slot field is `u_k(x) = w(k)·Re{û(k) e^{ik·x}}`, a
real field, and the slot fields sum to the velocity.

# Fields
- `invariant::I`: The invariant that was accumulated (e.g. `KineticEnergy()`).
- `ks::KS`: The wavenumber axes of the spectral grid.
- `weights::W`: `w(k)` for every slot (shape `ms`).
- `net_transfer::A`: `T(k) = Σ_p S(k|p)` — net per-slot transfer (shape `ms`); equals the spectral
  transfer from `calculate_spectral_flux`.
- `transfer::S`: `S(k|p) = Re{û*(k)·[(u·∇)u_p]^(k)}`, the rate at which slot `k` gives energy to the
  slot field `u_p`, shape `(ms..., ms...)` (slot `k`'s indices then slot `p`'s).

`w(k)·S(k|p) = ⟨u_k·(u·∇)u_p⟩` is the transfer between the two slot fields. For a divergence-free,
dealiased field it is antisymmetric, `w(k)S(k|p) = −w(p)S(p|k)`, and `Σ_k Σ_p w(k)S(k|p) = 0`; on a
full layout these read `S(k|p) = −S(p|k)`. Summing `w(k)S(k|p)` over the slots of two shells gives the
shell-to-shell matrix.
"""
struct ModeToModeTriadResult{I, KS, W, A, S}
    invariant::I
    ks::KS
    weights::W
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
