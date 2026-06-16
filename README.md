# FlowInvariantTransfer.jl

*Cross-scale transfer of quadratic inviscid invariants — kinetic energy, helicity, and enstrophy — in Julia.*

[![Build Status](https://github.com/jbphyswx/FlowInvariantTransfer.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/jbphyswx/FlowInvariantTransfer.jl/actions/workflows/CI.yml)
[![Dev Docs](https://img.shields.io/badge/docs-dev-blue.svg)](https://jbphyswx.github.io/FlowInvariantTransfer.jl/dev/)

`FlowInvariantTransfer.jl` provides fast, minimally allocating Julia implementations of cross-scale
transfer diagnostics for turbulent flows. It supports **multiple quadratic inviscid invariants**
(kinetic energy, helicity, enstrophy) via a trait-based dispatch, and provides
**partial-flux decompositions** (Helmholtz rotational/divergent) out of the box.

## Diagnostic Methods

| Method | Function | Output |
|--------|----------|--------|
| **Spectral flux** | `calculate_spectral_flux` | Π(K) — cumulative energy crossing wavenumber K |
| **Shell-to-shell transfer** | `calculate_shell_to_shell_transfer` | T(n,m) — directed transfer from shell m to shell n |
| **Mode-to-mode triads** | `calculate_mode_to_mode_transfer` | S(k\|p\|q) — exact triad transfer with periodic wrap |
| **Coarse-graining flux** | `calculate_coarse_graining_flux` | Π_ℓ(x) — pointwise flux at filter scale ℓ |
| **Triadic Orthogonal Decomposition** | `triadic_orthogonal_decomposition` | Mode bispectrum and coherent modes from temporal snapshots |

All methods support an allocating convenience API **and** a zero-allocation `!`-variant
with preallocated workspace structs — suitable for tight loops and time-stepping codes.

---

## Core Features

- **Multi-invariant support** — switch between kinetic energy (default), helicity (3D), and
  enstrophy (2D) with a single `invariant=Enstrophy()` keyword; only the per-mode transfer
  density changes, not the algorithm.
- **Helmholtz partial fluxes** — decompose into rotational/divergent components via
  `decomposition=HelmholtzDecomposition()`, backed by `HelmholtzDecomposition.jl`.
- **Unified entry point** — `calculate_energy_transfer` dispatches on method type.
- **Parametric, allocation-free structs** — result and workspace types are parametric on the
  array type. Works with `Float32`, `Float64`, AD dual numbers, etc.
- **`!`-first design** — every hot function has a `compute_nonlinear_term!`,
  `calculate_spectral_flux!`, `calculate_shell_to_shell_transfer!` variant.
- **Workspace structs** — `NonlinearTermWorkspace`, `SpectralFluxWorkspace`,
  `ShellToShellWorkspace`, `ScaleToScaleWorkspace` own all temporaries.
- **Flexible shell binning** — `LinearBinning`, `LogarithmicBinning`, `DyadicBinning`, `CustomBinning`.
- **Extension architecture** — lean core with 11 optional extensions loaded on demand.

---

## Installation

```julia
using Pkg
Pkg.add("FlowInvariantTransfer")
```

---

## Extension Architecture

The core package ships with a pure-Julia O(N²) direct-sum baseline requiring no compiled
dependencies. Load optional packages to activate fast paths and additional features:

| Extension | Trigger Package(s) | Provides |
|-----------|-------------------|----------|
| `FlowInvariantTransferFFTWExt` | `FFTW` | O(N log N) FFT fast path for all spectral diagnostics |
| `FlowInvariantTransferOhMyThreadsExt` | `OhMyThreads` | Multi-threaded backends for shell/mode/TOD loops |
| `FlowInvariantTransferDistributedExt` | `Distributed` + `SharedArrays` | Multi-process parallelism via `@distributed` |
| `FlowInvariantTransferKernelAbstractionsExt` | `KernelAbstractions` | GPU kernels (CUDA via KernelAbstractions) |
| `FlowInvariantTransferCGEFExt` | `CoarseGrainingEnergyFluxes` | Coarse-graining flux Π_ℓ(x) |
| `FlowInvariantTransferHelmholtzDecompositionExt` | `HelmholtzDecomposition` | Rotational/divergent partial-flux decomposition |
| `FlowInvariantTransferFINUFFTExt` | `FINUFFT` | Non-uniform FFT path for irregular Cartesian grids |
| `FlowInvariantTransferNUFSHTExt` | `NUFSHT` | Scattered spherical grid front-end |
| `FlowInvariantTransferFSHExt` | `FastSphericalHarmonics` | Regular spherical grid front-end |
| `FlowInvariantTransferFlowFieldSpectraExt` | `FlowFieldSpectra` | Spectral analysis integration |
| `FlowInvariantTransferCairoMakieExt` | `CairoMakie` | Plotting recipes |

### Backend Support Matrix

| Diagnostic | Serial | FFT | Threaded | Distributed | GPU |
|-----------|--------|-----|----------|-------------|-----|
| Spectral flux | ✓ | ✓ | ✓ | — | — |
| Shell-to-shell | ✓ | ✓ | ✓ | ✓ | ✓ |
| Mode-to-mode triads | ✓ | — | ✓ | ✓ | ✓ |
| Coarse-graining | — | — | — | — | — |
| TOD | ✓ | ✓ | ✓ | — | — |

Coarse-graining flux is provided entirely by the `CoarseGrainingEnergyFluxes` extension.

---

## Quickstart: Spectral Flux

```julia
using FlowInvariantTransfer
using FFTW   # activates FFTBackend automatically

N = 64; L = 2π
ks = wavenumber_grid((N, N), (L, L))

# Random divergence-free velocity field in spectral space
û = randn(ComplexF64, N, N, 2)

result = calculate_spectral_flux(û, ks;
    binning  = LinearBinning(2π / L),
    dealiasing = true,
    backend  = FFTBackend())

result.k_shells          # shell-centre wavenumbers
result.transfer_spectrum # T(k) — energy transfer rate
result.flux              # Π(K) — cumulative flux
```

## Quickstart: Shell-to-Shell Transfer

```julia
result = calculate_shell_to_shell_transfer(û, ks;
    binning  = LinearBinning(2π / L),
    dealiasing = true,
    backend  = FFTBackend())

result.transfer_matrix         # T(n,m) — N_shells × N_shells
result.max_antisymmetry_error  # should be ≈ 0 for divergence-free fields
```

## Quickstart: Mode-to-Mode Triads

```julia
# Exact triad transfer S(k|p|q) with shell reduction T(K,Q)
result = calculate_mode_to_mode_transfer(û, ks;
    binning   = LinearBinning(2π / L),
    invariant = KineticEnergy(),
    dealiasing = true)

result.net_transfer       # T(k) per mode — same shape as one velocity component
result.reductions.TKQ     # T(K,Q) — shell-reduced magnitude-to-magnitude matrix
result.reductions.K       # shell-centre wavenumbers
```

## Quickstart: Multi-Invariant (Enstrophy)

```julia
# 2D enstrophy transfer — counter-directional to inverse energy cascade
result_E = calculate_spectral_flux(û, ks;
    binning = LinearBinning(2π / L), invariant = KineticEnergy())

result_Ω = calculate_spectral_flux(û, ks;
    binning = LinearBinning(2π / L), invariant = Enstrophy())

# result_E.flux and result_Ω.flux show opposite cascade directions
```

## Quickstart: Helmholtz Partial Fluxes

```julia
using HelmholtzDecomposition  # loads the extension

# Spectral flux of the rotational component only
result_rot = calculate_spectral_flux(û, ks;
    binning = LinearBinning(2π / L),
    decomposition = RotationalDecomposition())

# Full decomposition returns a NamedTuple with :rotational and :divergent
result_helm = calculate_spectral_flux(û, ks;
    binning = LinearBinning(2π / L),
    decomposition = HelmholtzDecomposition())

result_helm.rotational   # SpectralFluxResult for rot component
result_helm.divergent    # SpectralFluxResult for div component
```

## Quickstart: Triadic Orthogonal Decomposition

```julia
using FlowInvariantTransfer
using FFTW

# 3D data array: (nt, nvar, nx)
nt, nvar, nx = 256, 1, 32
X = randn(nt, nvar, nx)

method = TriadicOrthogonalDecompositionMethod(nfft=64, noverlap=32, nmode=2)
result = calculate_energy_transfer(method, X; dt=0.01, backend=FFTBackend())

result.frequencies          # shifted frequency axes
result.mode_bispectrum      # singular values λ(fl, fn, mode)
result.modes                # Dict mapping (l, n) to mode NamedTuples
result.modal_energy_budget  # energy transfer per triad per mode
```

## Zero-Alloc Hot-Loop Usage

```julia
ws  = ShellToShellWorkspace(û, ks, LinearBinning(2π / L))
# ... inside a time loop:
calculate_shell_to_shell_transfer!(result, ws, û, ks; dealiasing=true)
```

---

## Example Figure

![Spectral flux and shell-to-shell transfer for a Taylor-Green vortex](docs/src/assets/energy_transfer.png)

---

## API Summary

### Diagnostic Functions

| Function | Description |
|----------|-------------|
| `calculate_energy_transfer` | Unified entry point dispatching on method type |
| `calculate_spectral_flux` / `!` | Spectral energy flux Π(K) |
| `calculate_shell_to_shell_transfer` / `!` | Shell-to-shell matrix T(n,m) |
| `calculate_mode_to_mode_transfer` / `!` | Mode-to-mode triad transfer S(k\|p\|q) |
| `calculate_coarse_graining_flux` | Pointwise coarse-graining flux Π_ℓ(x) |
| `triadic_orthogonal_decomposition` | Triadic Orthogonal Decomposition (TOD) |
| `compute_nonlinear_term` / `!` | Advection term N̂(k) = FFT[(u·∇)u] |
| `transfer_density` / `!` | Per-mode transfer density for any invariant |
| `decompose_field` | Field decomposition (Helmholtz rot/div) |

### Utility Functions

| Function | Description |
|----------|-------------|
| `wavenumber_grid` | Wavenumber axes for an N-D periodic domain |
| `wavenumber_magnitude_grid` | \|k\| at every spectral grid point |
| `assign_shells` | Integer shell-index array |
| `filter_response` | Filter transfer function Ĝ(k, ℓ) |
| `apply_filter_spectral` / `!` | Apply spectral filter to Fourier coefficients |

### Type Families

| Category | Types |
|----------|-------|
| **Methods** | `SpectralFluxMethod`, `ShellToShellTransferMethod`, `ModeToModeTransferMethod`, `CoarseGrainingFluxMethod`, `TriadicOrthogonalDecompositionMethod` |
| **Invariants** | `KineticEnergy`, `Helicity`, `Enstrophy` |
| **Decompositions** | `NoDecomposition`, `HelmholtzDecomposition`, `RotationalDecomposition`, `DivergentDecomposition` |
| **Backends** | `SerialBackend`, `FFTBackend`, `ThreadedBackend`, `DistributedBackend`, `GPUBackend`, `AutoBackend`, `NUFFTBackend`, `SHTBackend`, `NUFSHTBackend` |
| **Binning** | `LinearBinning`, `LogarithmicBinning`, `DyadicBinning`, `CustomBinning` |
| **Filters** | `SharpSpectralFilter`, `GaussianFilter`, `TopHatFilter` |
| **Results** | `SpectralFluxResult`, `ShellToShellResult`, `ModeToModeTriadResult`, `CoarseGrainingFluxResult`, `TriadicOrthogonalDecompositionResult` |
| **Workspaces** | `NonlinearTermWorkspace`, `SpectralFluxWorkspace`, `ShellToShellWorkspace`, `ScaleToScaleWorkspace` |

---

## References

- Verma et al. (2002) — *Local shell-to-shell energy transfer via nonlocal interactions in fluid turbulence* [arXiv:nlin/0204027](https://arxiv.org/abs/nlin/0204027)
- Alexakis, Mininni & Pouquet (2005) — *Imprint of large-scale flows on turbulence* Phys. Rev. E 72
- Aluie & Eyink (2009) — *Localness of energy cascade in hydrodynamic turbulence* Phys. Rev. Lett. 103
- Dar, Verma & Eswaran (2001) — *Energy transfer in two-dimensional magnetohydrodynamic turbulence* Physica D 157
- Kraichnan (1967) — *Inertial ranges in two-dimensional turbulence* Phys. Fluids 10
- Yeung, Chu & Schmidt (2026) — *Triadic orthogonal decomposition reveals nonlinearity in fluid flows* J. Fluid Mech. 1031, A34