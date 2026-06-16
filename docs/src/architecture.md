# Architecture: Design & Implementation

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
├── Types              # All type definitions (methods, invariants, backends, results)
├── Utils              # Wavenumber grids, magnitude, dealiasing masks
├── Invariants         # Per-mode transfer density for each invariant
├── Decomposition      # Field decomposition stubs (Helmholtz etc.)
├── ShellBinning       # Shell edge/center/assignment logic
├── Filters            # Spectral filter responses and application
├── Workspaces         # Preallocated buffer structs for zero-alloc paths
├── NonlinearTerm      # Pseudospectral (u·∇)u computation
├── SpectralFlux       # Π(K) and T(k) accumulation
├── CoarseGrainingFlux # Wrapper for CoarseGrainingEnergyFluxes.jl
├── ShellToShellTransfer    # T(n,m) matrix computation
├── ScaleToScaleTransfer    # S(k|p|q) triad computation
└── TriadicOrthogonalDecomposition  # TOD (frequency-domain SVD)
```

### Architecture Pattern: Method × Backend

FlowInvariantTransfer.jl uses a **method–backend composition** pattern:

```
Spectral data (û, ks)
    ↓
[Method Type] ← Specifies WHICH diagnostic to compute
    ↓
[Invariant Trait] ← Specifies WHICH quadratic invariant (KE/H/Ω)
    ↓
[Execution Backend] ← Specifies HOW to compute (Serial/FFT/GPU/...)
    ↓
Result Container ← Stores transfer spectra, matrices, modes
```

Example:
```julia
# Method: "Compute shell-to-shell transfer"
method = ShellToShellTransferMethod(LinearBinning(1.0))

# Invariant: "Accumulate enstrophy"
# Backend: "Use 4 threads"
result = calculate_energy_transfer(method, û, ks;
    invariant = Enstrophy(),
    backend = ThreadedBackend())
```

The **method type** determines *what* to calculate. The **invariant trait** determines
*which* physical quantity to track. The **backend type** determines *how* to execute.

---

## Type Hierarchy

### Method Types

```
AbstractEnergyTransferMethod (abstract)
├── SpectralFluxMethod{B}
├── ShellToShellTransferMethod{B}
├── ModeToModeTransferMethod{B, I}
├── CoarseGrainingFluxMethod{F}
└── TriadicOrthogonalDecompositionMethod{N, O, M}
```

### Invariant Traits

```
AbstractInvariant (abstract)
├── KineticEnergy       # E = ½∫|u|²  (default)
├── Helicity            # H = ∫u·ω    (3D only)
└── Enstrophy           # Ω = ½∫ω²   (2D only)
```

### Field Decompositions

```
AbstractFieldDecomposition (abstract)
├── NoDecomposition              # Full velocity (default)
├── HelmholtzDecomposition       # Both rot + div
├── RotationalDecomposition      # Rot component only
└── DivergentDecomposition       # Div component only
```

### Execution Backends

```
AbstractExecutionBackend (abstract)
├── SerialBackend           # Reference O(N²) implementation
├── FFTBackend              # O(N log N) via FFTW
├── ThreadedBackend         # OhMyThreads parallelism
├── DistributedBackend      # Distributed.jl + SharedArrays
├── GPUBackend{B}           # KernelAbstractions (parametric on device)
├── AutoBackend             # Auto-detect best available
├── NUFFTBackend            # Non-uniform FFT (FINUFFT)
├── SHTBackend              # Regular spherical harmonics (FSH)
└── NUFSHTBackend           # Scattered spherical harmonics (NUFSHT)
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
SpectralFluxResult{V}                           # k_shells, T(k), Π(K)
ShellToShellResult{V, M, E}                     # T(n,m) matrix, net transfer
ModeToModeTriadResult{I, KS, A, NT}             # net T(k), optional T(K,Q) reductions
CoarseGrainingFluxResult{S, A}                  # Π_ℓ(x) field
CoarseGrainingFluxResultWithDiagnostics{S, A}   # + stress/strain tensors
TriadicOrthogonalDecompositionResult{V, A3, PM} # mode bispectrum, modes, energy budget
```

All result types are **fully parametric** on their array/scalar types — no hardcoded
`Vector{Float64}`. This means they work natively with `Float32`, GPU arrays, AD dual
numbers, and Unitful quantities.

### Workspace Types

```
NonlinearTermWorkspace    # FFT plans, physical-space buffers, N̂
SpectralFluxWorkspace     # T_spec, flux accumulators
ShellToShellWorkspace     # Shell-filtered velocity, shell_idx
ScaleToScaleWorkspace     # net_transfer, T_mat, shell_idx
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
    calculate_mode_to_mode_transfer(û, ks; binning=method.binning, ...)
```

### Backend Dispatch (Two Levels)

Each diagnostic function dispatches on the `backend` keyword via an internal
`_calculate_*!` method. The core module defines:

1. **`SerialBackend`** — the reference implementation (always available)
2. **Stubs** for other backends that throw informative errors

Extensions then **override the stubs** when their trigger packages are loaded:

```julia
# In core (src/ShellToShell/ShellToShellTransfer.jl):
function _calculate_shell_to_shell!(result, ws, û, ks, ::ThreadedBackend; ...)
    _shell_to_shell_threaded!(...)  # stub → throws helpful error
end

# In extension (ext/FlowInvariantTransferOhMyThreadsExt.jl):
function FET.ShellToShellTransfer._shell_to_shell_threaded!(...)
    # Real OhMyThreads implementation
end
```

This ensures:
- ✅ No runtime overhead choosing between backends (static dispatch)
- ✅ Clear error messages when a backend's dependency isn't loaded
- ✅ Each backend can use its optimal algorithm
- ✅ Type-stable throughout

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
# ... 8 more

[extensions]
FlowInvariantTransferFFTWExt = "FFTW"
FlowInvariantTransferOhMyThreadsExt = "OhMyThreads"
FlowInvariantTransferDistributedExt = ["Distributed", "SharedArrays"]
# ... 8 more
```

**Benefits:**
- Users who don't use GPU pay zero cost (no KernelAbstractions load time)
- Fresh Julia sessions start fast (only `LinearAlgebra` + `PrecompileTools` in `[deps]`)
- Each extension is a self-contained module overriding specific stubs

---

## Code Layout

### Source Files

| File | Purpose |
|------|---------|
| `src/FlowInvariantTransfer.jl` | Top-level module: includes, re-exports, unified entry point, precompilation |
| `src/types.jl` | All type definitions (methods, invariants, backends, results) |
| `src/utils.jl` | Wavenumber grids, magnitude, dealiasing masks, input validation |
| `src/Invariants.jl` | Per-mode transfer density for KE, helicity, enstrophy |
| `src/Decomposition.jl` | Field decomposition dispatch (stubs for Helmholtz ext) |
| `src/ShellToShell/ShellBinning.jl` | Shell edge/center computation, `assign_shells` |
| `src/Filters.jl` | Filter response functions and spectral application |
| `src/Workspaces.jl` | Preallocated workspace structs |
| `src/NonlinearTerm.jl` | Pseudospectral (u·∇)u — Serial and FFT paths |
| `src/SpectralFlux.jl` | Spectral flux Π(K) and transfer spectrum T(k) |
| `src/CoarseGrainingFlux.jl` | Wrapper stub for CGEF extension |
| `src/ShellToShell/ShellToShellTransfer.jl` | Shell-to-shell T(n,m) — Serial core |
| `src/ScaleToScale/ScaleToScaleTransfer.jl` | Mode-to-mode S(k\|p\|q) — Serial core |
| `src/ScaleToScale/TriadicOrthogonalDecomposition/` | TOD implementation |

### Extension Files

| File | Trigger | Overrides |
|------|---------|-----------|
| `ext/FlowInvariantTransferFFTWExt.jl` | FFTW | FFT-based nonlinear term, spectral flux, shell-to-shell |
| `ext/FlowInvariantTransferOhMyThreadsExt.jl` | OhMyThreads | Threaded shell-to-shell and mode-to-mode |
| `ext/FlowInvariantTransferDistributedExt.jl` | Distributed+SharedArrays | Distributed shell-to-shell and mode-to-mode |
| `ext/FlowInvariantTransferKernelAbstractionsExt.jl` | KernelAbstractions | GPU kernels for all transfer densities and triads |
| `ext/FlowInvariantTransferCGEFExt.jl` | CoarseGrainingEnergyFluxes | Coarse-graining flux computation |
| `ext/FlowInvariantTransferHelmholtzDecompositionExt.jl` | HelmholtzDecomposition | Physical and spectral Helmholtz decomposition |
| `ext/FlowInvariantTransferFINUFFTExt.jl` | FINUFFT | Non-uniform FFT path |
| `ext/FlowInvariantTransferNUFSHTExt.jl` | NUFSHT | Scattered spherical front-end |
| `ext/FlowInvariantTransferFSHExt.jl` | FastSphericalHarmonics | Regular spherical front-end |
| `ext/FlowInvariantTransferFlowFieldSpectraExt.jl` | FlowFieldSpectra | Spectral analysis integration |
| `ext/FlowInvariantTransferCairoMakieExt.jl` | CairoMakie | Plotting recipes |

---

## Design Principles

1. **Type Dispatch** — use Julia's type system for all method/backend/invariant selection.
   No string dispatch, no runtime branching. The compiler resolves everything statically.

2. **Zero-Cost Abstraction** — backend dispatch adds no runtime overhead. Each backend
   is a singleton type resolved at compile time.

3. **`!`-First Design** — every hot function has a mutating variant that writes into
   preallocated workspaces. The allocating convenience API calls the `!` variant internally.

4. **Workspace-Based Allocation** — all temporaries (FFT plans, physical-space buffers,
   shell indices) are owned by workspace structs. Preallocate once, reuse across timesteps.

5. **Parametric Result Types** — result containers are parametric on array/scalar types.
   `Float32`, GPU arrays, AD dual numbers all work without specialization.

6. **Lean Core** — only `LinearAlgebra` and `PrecompileTools` in hard dependencies.
   Everything else is a weak dependency loaded on demand via extensions.

7. **Invariant Traits** — switching between kinetic energy, helicity, and enstrophy
   requires changing only one keyword argument. The algorithm structure is identical;
   only the per-mode inner product changes.
