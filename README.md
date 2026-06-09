# FlowEnergyTransfer.jl

*Cross-scale kinetic energy transfer — spectral flux, shell-to-shell, and coarse-graining — in Julia.*

[![Build Status](https://github.com/jbphyswx/FlowEnergyTransfer.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/jbphyswx/FlowEnergyTransfer.jl/actions/workflows/CI.yml)
[![Dev Docs](https://img.shields.io/badge/docs-dev-blue.svg)](https://jbphyswx.github.io/FlowEnergyTransfer.jl/dev/)

`FlowEnergyTransfer.jl` provides fast, minimally allocating Julia implementations of the standard
methods for computing kinetic energy transfer across scales in turbulent flow:

- **Spectral flux** Π(K) — cumulative energy crossing a wavenumber threshold K
- **Shell-to-shell transfer** T(n, m) — directed energy transfer from shell m to shell n
- **Coarse-graining flux** Π_ℓ(x) — pointwise scale-to-scale flux at filter scale ℓ

All methods support both an allocating convenience API and a zero-alloc `!`-variant taking
preallocated workspace structs — suitable for tight loops and time-stepping codes.

---

## Core Features

- **Unified entry point** `calculate_energy_transfer` dispatches on method type.
- **Parametric, allocation-free structs** — result and workspace types are parametric on the
  array type, no `Vector{FT}` hardcoding. Works with `Float32`, `Float64`, AD dual numbers, etc.
- **`!`-first design** — every hot function has a `compute_nonlinear_term!`,
  `calculate_spectral_flux!`, `calculate_shell_to_shell_transfer!` variant.
- **Workspace structs** `NonlinearTermWorkspace`, `SpectralFluxWorkspace`, `ShellToShellWorkspace`
  own all temporaries — zero allocation in the hot path.
- **Flexible shell binning** — `LinearBinning`, `LogarithmicBinning`, `DyadicBinning`, `CustomBinning`.
- **Extension architecture** — slow O(N²) direct-sum baseline out of the box; load `FFTW` to
  activate the O(N log N) FFT path automatically.

---

## Installation

```julia
using Pkg
Pkg.add("FlowEnergyTransfer")
```

---

## Extension Architecture

| Method | Baseline | Fast Path | Required Library |
|--------|----------|-----------|-----------------|
| Nonlinear term / spectral flux | `SerialBackend()` (direct sum) | `FFTBackend()` | `using FFTW` |
| Shell-to-shell transfer | `SerialBackend()` | `FFTBackend()` | `using FFTW` |
| Coarse-graining flux | — | — | `using CoarseGrainingEnergyFluxes` |

---

## Quickstart: Spectral Flux

```julia
using FlowEnergyTransfer
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

| Function | Description |
|----------|-------------|
| `calculate_spectral_flux` / `!` | Spectral energy flux Π(K) |
| `calculate_shell_to_shell_transfer` / `!` | Shell-to-shell matrix T(n,m) |
| `calculate_coarse_graining_flux` | Pointwise coarse-graining flux |
| `compute_nonlinear_term` / `!` | Advection term N̂(k) = FFT[(u·∇)u] |
| `wavenumber_grid` | Wavenumber axes for an N-D periodic domain |
| `wavenumber_magnitude_grid` | |k| at every spectral grid point |
| `assign_shells` | Integer shell-index array (replaces BitArray masks) |
| `NonlinearTermWorkspace` | Preallocated buffers for nonlinear term |
| `SpectralFluxWorkspace` | Preallocated buffers for spectral flux |
| `ShellToShellWorkspace` | Preallocated buffers for shell-to-shell transfer |

---

## References

- Verma et al. (2002) — *Local shell-to-shell energy transfer via nonlocal Interactions in fluid turbulence* [arXiv:nlin/0204027](https://arxiv.org/abs/nlin/0204027)
- Alexakis, Mininni & Pouquet (2005) — *Imprint of large-scale flows on turbulence* Phys. Rev. E 72
- Aluie & Eyink (2009) — *Localness of energy cascade in hydrodynamic turbulence* Phys. Rev. Lett. 103