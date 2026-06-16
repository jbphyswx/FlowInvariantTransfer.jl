# Backends & Extensions

This page documents each execution backend, when to use it, and how extensions are loaded.

## Backend Overview

| Backend | Dependency | Best For |
|---------|-----------|----------|
| [`SerialBackend`](@ref) | None | Reference results, debugging, small grids |
| [`FFTBackend`](@ref) | `FFTW` | Production spectral diagnostics (spectral flux, shell-to-shell, TOD) |
| [`ThreadedBackend`](@ref) | `OhMyThreads` | Multi-core parallelism for shell/mode loops and TOD |
| [`DistributedBackend`](@ref) | `Distributed` + `SharedArrays` | Multi-node or many-process parallelism |
| [`GPUBackend`](@ref) | `KernelAbstractions` + `CUDA` | Large-grid shell-to-shell and mode-to-mode on CUDA GPUs |
| [`AutoBackend`](@ref) | Varies | Automatic best-available selection |
| [`NUFFTBackend`](@ref) | `FINUFFT` | Irregular Cartesian grids |
| [`SHTBackend`](@ref) | `FastSphericalHarmonics` | Regular spherical grids |
| [`NUFSHTBackend`](@ref) | `NUFSHT` | Scattered spherical data |

---

## SerialBackend

The reference implementation. Uses direct O(N²) summation for the nonlinear term and
all accumulation loops. No external dependencies.

```julia
result = calculate_spectral_flux(û, ks;
    binning = LinearBinning(1.0),
    backend = SerialBackend())
```

**When to use:** Debugging, correctness verification, very small grids (N ≤ 16).

---

## FFTBackend

Production fast path using FFTW for all Fourier transforms. Reduces the nonlinear term
computation from O(N²) to O(N log N) via pseudospectral evaluation:

1. IFFT velocity and derivatives to physical space
2. Multiply pointwise (physical-space products)
3. FFT back to spectral space
4. Apply 2/3 dealiasing mask

```julia
using FFTW  # loads the extension automatically

result = calculate_spectral_flux(û, ks;
    binning = LinearBinning(1.0),
    backend = FFTBackend())
```

**When to use:** Standard production runs. Dominant speedup for spectral flux and
shell-to-shell transfer on grids N ≥ 32.

**Supports:** Spectral flux, shell-to-shell transfer, TOD.

---

## ThreadedBackend

Multi-threaded parallelism via OhMyThreads. Parallelises the outer loop over mediator
shells (shell-to-shell), receiver modes (mode-to-mode), and time blocks (TOD).

```julia
using OhMyThreads  # loads the extension

result = calculate_shell_to_shell_transfer(û, ks;
    binning = LinearBinning(1.0),
    backend = ThreadedBackend())
```

**When to use:** Multi-core machines, especially for shell-to-shell with many shells
or mode-to-mode triads with moderate grid sizes.

**Supports:** Shell-to-shell, mode-to-mode, TOD, spectral flux.

---

## DistributedBackend

Multi-process parallelism using Julia's `Distributed` standard library with
`SharedArrays` for conflict-free parallel writing. Uses `@distributed (+)` reduction
over mediator shells (shell-to-shell) or mode chunks (mode-to-mode).

```julia
using Distributed
using SharedArrays
addprocs(4)
@everywhere using FlowInvariantTransfer

# Convert velocity to SharedArray for worker access
s_û = SharedArray(û)

result = calculate_shell_to_shell_transfer(s_û, ks;
    binning = LinearBinning(1.0),
    backend = DistributedBackend())
```

**When to use:** Large grids where the computation exceeds single-machine threading,
or multi-node cluster environments.

**Supports:** Shell-to-shell, mode-to-mode.

---

## GPUBackend

Device-generic GPU execution via KernelAbstractions. Implements custom `@kernel`
functions for transfer density computation, shell accumulation, and triad loops.
Uses `@atomic` for concurrent writes to the shell-to-shell matrix.

```julia
using KernelAbstractions
using CUDA

# Transfer data to GPU
û_gpu = CuArray(û)
ks_gpu = map(CuArray, ks)

result = calculate_shell_to_shell_transfer(û_gpu, ks_gpu;
    binning = LinearBinning(1.0),
    backend = GPUBackend(CUDABackend()))
```

**When to use:** Large 2D/3D grids where the parallelism of shell-to-shell or
mode-to-mode loops maps well to GPU architectures.

**Currently supports:** CUDA only. AMDGPU and Metal are deferred.

**Supports:** Shell-to-shell, mode-to-mode (both with all three invariants).

---

## AutoBackend

Automatically selects the best available backend at call time, checking in order:
distributed → threaded → serial. Transform fast-paths (FFT) are chosen independently
based on whether their extensions are loaded.

```julia
result = calculate_spectral_flux(û, ks;
    binning = LinearBinning(1.0),
    backend = AutoBackend())
```

---

## NUFFTBackend

Non-uniform fast Fourier transform for irregular Cartesian grids, backed by FINUFFT.
Computes regular Fourier coefficients from scattered data, then delegates to the
standard spectral diagnostics.

```julia
using FINUFFT
result = calculate_coarse_graining_flux(velocity_fields, coords, ℓ, filter;
    backend = NUFFTBackend())
```

---

## SHTBackend

Spherical harmonic transform front-end for regular latitude-longitude grids, backed
by FastSphericalHarmonics. Transforms physical-space velocity components to spherical
harmonic coefficients, then delegates to the core spectral calculations.

```julia
using FastSphericalHarmonics

# velocity_fields = (u, v) on a regular spherical grid
# coords = (θ, φ) coordinate vectors
result = calculate_energy_transfer(
    SpectralFluxMethod(LinearBinning(1.0)),
    velocity_fields, coords, (Nθ,);
    backend = SHTBackend())
```

---

## NUFSHTBackend

Non-uniform spherical harmonic transform for scattered spherical data, backed by
NUFSHT.jl. Handles observations at arbitrary (θ, φ) locations.

```julia
using NUFSHT

result = calculate_energy_transfer(
    SpectralFluxMethod(LinearBinning(1.0)),
    velocity_fields, coords, (lmax+1,);
    backend = NUFSHTBackend(), tol=1e-8)
```

---

## Backend Support Matrix

| Diagnostic | Serial | FFT | Threaded | Distributed | GPU |
|-----------|--------|-----|----------|-------------|-----|
| Spectral flux Π(K) | ✓ | ✓ | ✓ | — | — |
| Shell-to-shell T(n,m) | ✓ | ✓ | ✓ | ✓ | ✓ |
| Mode-to-mode S(k\|p\|q) | ✓ | — | ✓ | ✓ | ✓ |
| Coarse-graining Π_ℓ(x) | — | — | — | — | — |
| TOD | ✓ | ✓ | ✓ | — | — |

Coarse-graining flux is provided entirely by the CoarseGrainingEnergyFluxes extension
and has its own parallelism model.

---

## Extension Loading

Extensions are loaded automatically when you `using` their trigger package:

```julia
using FlowInvariantTransfer  # lean core only
using FFTW                   # → FFTWExt loaded, FFTBackend available
using OhMyThreads            # → OhMyThreadsExt loaded, ThreadedBackend available
using HelmholtzDecomposition # → HelmholtzDecompositionExt loaded, decompose_field works
```

If you call a backend whose extension isn't loaded, you get a clear error message:

```
ArgumentError: Threaded mode-to-mode transfer requires OhMyThreads. Run `using OhMyThreads` to load the extension.
```
