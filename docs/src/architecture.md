# Architecture

This document explains how FlowInvariantTransfer.jl is organized internally and how
computations are dispatched.

## Table of Contents
- [Module Organization](#module-organization)
- [Type Hierarchy](#type-hierarchy)
- [Dispatch Pattern](#dispatch-pattern)
- [Extension System](#extension-system)
- [Code Layout](#code-layout)
- [Design Principles](#design-principles)

## Module Organization

### Core Module: FlowInvariantTransfer

The main module (`src/FlowInvariantTransfer.jl`) includes a set of submodules, each
responsible for a distinct concern:

```
FlowInvariantTransfer (top-level)
├── Types              # Method, invariant, decomposition, binning, geometry, dealiasing and result types
├── Utils              # Wavenumber grids, magnitude, input validation
├── Invariants         # Per-mode transfer density for each invariant
├── Decomposition      # Field decompositions (helical, toroidal/poloidal; Helmholtz stubs)
├── ShellBinning       # Shell edge/center/assignment logic
├── Filters            # Spectral filter responses and application
├── Workspaces         # Preallocated buffer structs for zero-alloc paths
├── NonlinearTerm      # Pseudospectral (u·∇)f computation
├── SpectralFlux       # Π(K), T(k) and the partial fluxes
├── CoarseGrainingFlux # Wrapper for CoarseGrainingEnergyFluxes.jl
├── ShellToShellTransfer    # T(n,m) matrix computation
├── BandTransfer       # Smooth band-to-band T(K,Q) and band energies
├── TriadicOrthogonalDecomposition  # TOD (frequency-domain SVD)
├── ModeToModeTransfer    # S(k|p) triad computation
├── Compressible       # Momentum-weighted compressible transfer
└── Spherical          # Spherical (degree-spectrum) transfer
```

### Architecture Pattern: Method × Invariant × Backends

A diagnostic is selected by composing independent choices:

```
Spectral data (û, ks)
    ↓
[Method]            ← which diagnostic to compute
    ↓
[Invariant trait]   ← which quadratic invariant (KE / helicity / enstrophy / scalar variance)
    ↓
[transform]         ← which transform (direct sum / FFT / …)
[execution]         ← how the work runs (serial / threaded / distributed / GPU)
    ↓
Result container    ← transfer spectra, matrices, modes
```

Example, with `const FIT = FlowInvariantTransfer`:
```julia
# Method: shell-to-shell transfer; invariant: enstrophy; execution: threads
method = FIT.Types.ShellToShellTransferMethod(FIT.Types.LinearBinning(1.0))
result = calculate_energy_transfer(method, û, ks;
    invariant = FIT.Types.Enstrophy(),
    execution = FIT.ComputationalBackends.ThreadedBackend())
```

The **method type** determines *what* to calculate, the **invariant trait** *which* physical quantity
to track, and the `transform` and `execution` keywords *how* to execute.

---

## Type Hierarchy

### Method Types

```
AbstractEnergyTransferMethod (abstract)
├── SpectralFluxMethod{B}
├── ShellToShellTransferMethod{B}
├── ModeToModeTransferMethod{I}
├── CoarseGrainingFluxMethod{F, S}
├── TriadicOrthogonalDecompositionMethod{N, O, M}
├── SphericalTransferMethod{T}
└── DivergentSphericalTransferMethod{T}
```

### Invariant Traits

```
AbstractInvariant (abstract)
├── KineticEnergy       # E = ½∫|u|²  (default)
├── Helicity            # H = ∫u·ω    (3D only)
├── Enstrophy           # Ω = ½∫ω²   (2D conserved, 3D with stretching)
└── PassiveScalar       # E_θ = ½∫θ²
```

### Field Decompositions

```
AbstractFieldDecomposition (abstract)
├── NoDecomposition              # Full velocity (default)
├── HelmholtzDecomposition       # Both rot + div
├── RotationalDecomposition      # Rot component only
├── DivergentDecomposition       # Div component only
├── HelicalDecomposition         # ±-helicity components (3D)
└── ToroidalPoloidalDecomposition  # Craya–Herring toroidal/poloidal (3D)
```

### Backends

The transform tags come from `SpectralBackends` (the NUFFT library tags from FlowTransformBindings),
the execution tags from `ComputationalBackends`:

```
SpectralBackends.AbstractSpectralBackend (abstract)
├── AutoSpectralBackend           # FFT when FFTW is loaded, else the direct sum
├── DirectSumSpectralBackend      # Reference O(N²ᴰ) direct sums
├── FFTSpectralBackend            # O(Nᴰ log N) via FFTW
├── FlowTransformBindings NUFFT tags  # Non-uniform FFT (NonuniformFFTs / FINUFFT)
├── FSHTSpectralBackend           # Regular spherical harmonics (FSH)
└── NUFSHTSpectralBackend         # Scattered spherical harmonics (NUFSHT)

ComputationalBackends.AbstractExecutionBackend (abstract)
├── SerialBackend           # Default
├── ThreadedBackend         # OhMyThreads parallelism
├── DistributedBackend      # Distributed.jl worker processes
├── GPUBackend{B}           # KernelAbstractions (parametric on device)
├── MPIBackend              # Per-rank backend of the MPI pencil axis
└── AutoBackend             # Threaded when available, else serial
```

### Shell Binning

```
AbstractShellBinning (abstract)
├── LinearBinning(Δk)           # Uniform width Δk
├── LogarithmicBinning(k₀, λ)   # Geometric ratio λ
├── DyadicBinning(k₀)           # Log with λ=2
└── CustomBinning(edges)         # User-specified edges
```

### Filters

```
AbstractFilter (abstract)
├── SharpSpectralFilter     # Brick-wall |k| < π/ℓ
├── GaussianFilter          # exp(-k²ℓ²/24)
└── TopHatFilter            # sinc(kℓ/2π)
```

### Result Containers

```
SpectralFluxResult{KS, E, V}                    # k_shells, shell_edges, T(k), Π(K)
ShellToShellResult{V, W, M}                     # T(n,m) matrix, net transfer
ModeToModeTriadResult{I, KS, W, A, S}           # weights, net_transfer T(k), resolved S(k|p)
CompressibleFluxResult{KS, ES, TS, FL, CH, PD}  # T_u(k), Π, channels, pressure-dilatation
CoarseGrainingFluxResult{S, A}                  # Π_ℓ(x) field
CoarseGrainingFluxResultWithDiagnostics{S, A, T}  # + stress/strain tensors
TriadicOrthogonalDecompositionResult{V, A3, PM, EC, XM}  # mode bispectrum, modes, energy budget
SphericalTransferResult{V, T}                   # T_E(l), T_Z(l) and fluxes
DivergentSphericalTransferResult{V, T}          # T(l), rotational/divergent channels and fluxes
```

The result types are parametric on their array and scalar types, so they hold `Float32` and device
arrays as they come.

### Workspace Types

```
NonlinearTermWorkspace    # FFT plans, physical-space buffers, N̂, the padded grid under PaddedThreeHalves
SpectralFluxWorkspace     # T_spec, flux accumulators
ShellToShellWorkspace     # Shell-filtered velocity, shell_idx
BandTransferWorkspace     # Band weights, band field, density scratch
CompressibleWorkspace     # Transform context and every field-sized intermediate
TODWorkspace              # Temporal-DFT output, plans, per-triad SVD scratch
```

Workspaces **own all temporaries** for their respective computations. Preallocate once,
reuse across timesteps — zero heap allocation in the hot path.

---

## Dispatch Pattern

### Unified Entry Point

`calculate_energy_transfer(method, data, coords; kwargs...)` dispatches on the
`method` type:

```julia
calculate_energy_transfer(::SpectralFluxMethod, û, ks; ...) →
    calculate_spectral_flux(û, ks; binning=method.binning, ...)

calculate_energy_transfer(::ShellToShellTransferMethod, û, ks; ...) →
    calculate_shell_to_shell_transfer(û, ks; binning=method.binning, ...)

calculate_energy_transfer(::ModeToModeTransferMethod, û, ks; ...) →
    calculate_mode_to_mode_transfer(û, ks; invariant=method.invariant, ...)
```

### Backend Dispatch (Two Levels)

Each diagnostic function dispatches on the `execution` keyword (and on the resolved `transform`) via
an internal `_calculate_*!` method. The core module defines:

1. **`SerialBackend`** — the reference implementation (always available)
2. **Stubs** for other backends that throw informative errors

Extensions then **override the stubs** when their trigger packages are loaded:

```julia
# In core (src/ShellToShell/ShellToShellTransfer.jl):
_calculate_shell_to_shell!(result, ws, û, ks, ::ThreadedBackend, spectral; kwargs...) =
    _shell_to_shell_threaded!(result, ws, û, ks, spectral; kwargs...)  # stub → throws helpful error

# In extension (ext/FlowInvariantTransferOhMyThreadsExt.jl):
function FIT.ShellToShellTransfer._shell_to_shell_threaded!(...)
    # Real OhMyThreads implementation
end
```

This gives:
- static dispatch between backends
- a clear error when a backend's dependency isn't loaded
- a free choice of algorithm per backend

---

## Extension System

### Lazy Loading via Extensions

Optional dependencies are loaded only when needed via Julia's `[weakdeps]` +
`[extensions]` mechanism in `Project.toml`:

```toml
[weakdeps]
FFTW = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
OhMyThreads = "67456a42-1dca-4109-a031-0a68de7e3ad5"
KernelAbstractions = "63c18a36-062a-441e-b654-da1e3ab1ce7c"
# ... (the rest in Project.toml)

[extensions]
FlowInvariantTransferFFTWExt = "FFTW"
FlowInvariantTransferOhMyThreadsExt = "OhMyThreads"
FlowInvariantTransferDistributedExt = "Distributed"
# ... (the rest in Project.toml)
```

**Benefits:**
- Users who don't use GPU pay zero cost (no KernelAbstractions load time)
- Fresh Julia sessions load only `ComputationalBackends`, `FlowTransformBindings`, `LinearAlgebra`,
  `PrecompileTools` and `SpectralBackends` from `[deps]`
- Each extension is a self-contained module overriding specific stubs

---

## Code Layout

### Source Files

| File | Purpose |
|------|---------|
| `src/FlowInvariantTransfer.jl` | Top-level module: includes, exports, unified entry point, `to_spectral`/`from_spectral`, precompilation |
| `src/Types.jl` | Method, invariant, decomposition, binning, geometry, dealiasing and result types; backend resolution |
| `src/Utils.jl` | Wavenumber grids, magnitude, input validation |
| `src/Invariants.jl` | Per-mode transfer density for KE, helicity, enstrophy, passive scalar |
| `src/Decomposition.jl` | Helical and toroidal/poloidal decompositions; stubs for the Helmholtz extension |
| `src/ShellBinning.jl` | Shell edge/center computation, `assign_shells` |
| `src/Filters.jl` | Filter response functions and spectral application |
| `src/Workspaces.jl` | Preallocated workspace structs |
| `src/NonlinearTerm.jl` | Pseudospectral (u·∇)f — direct-sum path and the FFTW-extension stubs |
| `src/SpectralFlux.jl` | Spectral flux Π(K), transfer spectrum T(k), partial fluxes |
| `src/CoarseGrainingFlux.jl` | Wrapper stub for CGEF extension |
| `src/Scattered.jl` | Scattered-Cartesian NUFFT entries (coarse-graining flux, `to_spectral`) over FlowTransformBindings' plans |
| `src/ShellToShell/ShellToShellTransfer.jl` | Shell-to-shell T(n,m) — Serial core |
| `src/BandTransfer.jl` | Smooth band-to-band T(K,Q) and band energies |
| `src/ScaleToScale/ModeToModeTransfer.jl` | Mode-to-mode S(k\|p) — Serial core |
| `src/ScaleToScale/TriadicOrthogonalDecomposition/` | TOD implementation |
| `src/Compressible/CompressibleTransfer.jl` | Compressible momentum-weighted transfer |
| `src/Spherical/SphericalTransfer.jl` | Spherical transfer workspaces and degree-spectrum reductions |

### Extension Files

| File | Trigger | Overrides |
|------|---------|-----------|
| `ext/FlowInvariantTransferFFTWExt.jl` | FFTW | FFT nonlinear term, shell-to-shell, TOD temporal DFT, compressible transform context |
| `ext/FlowInvariantTransferOhMyThreadsExt.jl` | OhMyThreads | Threaded loops of every diagnostic and the batches |
| `ext/FlowInvariantTransferDistributedExt.jl` | Distributed | Distributed loops of every diagnostic and the batches |
| `ext/FlowInvariantTransferKernelAbstractionsExt.jl` | KernelAbstractions | GPU kernels for the transfer densities, shell reductions and mode-to-mode slots |
| `ext/FlowInvariantTransferGPUArraysCoreExt.jl` | GPUArraysCore | Device-array methods of the compressible transfer and TOD |
| `ext/FlowInvariantTransferMPIExt.jl` | MPI | `mpi_batch_map` (batch axis) |
| `ext/FlowInvariantTransferPencilFFTsExt.jl` | MPI + PencilFFTs + PencilArrays | `pencil_spectral_flux` (pencil axis) |
| `ext/FlowInvariantTransferCGEFExt.jl` | CoarseGrainingEnergyFluxes | Coarse-graining flux computation |
| `ext/FlowInvariantTransferHelmholtzDecompositionExt.jl` | HelmholtzDecomposition | Physical and spectral Helmholtz decomposition |
| `ext/FlowInvariantTransferNUFSHTExt.jl` | NUFSHT | Scattered spherical front-end |
| `ext/FlowInvariantTransferFSHExt.jl` | FastSphericalHarmonics + FFTW | Regular spherical front-end |
| `ext/FlowInvariantTransferFlowFieldSpectraExt.jl` | FlowFieldSpectra | Physical ↔ spectral on uniform and `FlowGeometries` grids |
| `ext/FlowInvariantTransferFlowGeometriesExt.jl` | FlowGeometries | Spherical transfer and scattered NUFFT entries on `FlowGeometries` grids |
| `ext/FlowInvariantTransferCairoMakieExt.jl` | CairoMakie | Plotting recipes |

---

## Design Principles

1. **Type Dispatch** — Julia's type system selects the method, backend and invariant.

2. **Static backend dispatch** — the backend argument's type selects the implementation at compile
   time.

3. **`!`-First Design** — every hot function has a mutating variant that writes into
   preallocated workspaces. The allocating convenience API calls the `!` variant internally.

4. **Workspace-Based Allocation** — all temporaries (FFT plans, physical-space buffers,
   shell indices) are owned by workspace structs. Preallocate once, reuse across timesteps.

5. **Parametric Result Types** — result containers are parametric on array/scalar types.

6. **Lean Core** — the hard dependencies are `ComputationalBackends`, `FlowTransformBindings`,
   `LinearAlgebra`, `PrecompileTools` and `SpectralBackends`. Everything else is a weak dependency
   loaded on demand via extensions.

7. **Invariant Traits** — switching between kinetic energy, helicity, enstrophy and scalar variance
   is one keyword argument; the per-mode inner product is the only part that changes.
