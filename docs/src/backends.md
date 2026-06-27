# Backends, Dealiasing & Extensions

Transfer diagnostics are configured along **two orthogonal axes**, plus a dealiasing
strategy. This page documents each, when to use it, and how extensions are loaded.

## Two backend axes

The package never conflates *which transform* it uses with *how that work is run*:

- **Spectral backend** (`spectral::AbstractSpectralBackend`) — **which transform** computes the
  pseudospectral nonlinear term: [`DirectSumBackend`](@ref) (direct DFT, no deps, the correctness
  oracle), [`FFTBackend`](@ref) (FFTW, the `O(Nᴰ log N)` workhorse), [`NUFFTBackend`](@ref)
  (FINUFFT, scattered Cartesian), [`SHTBackend`](@ref) (regular spherical), [`NUFSHTBackend`](@ref)
  (scattered spherical).
- **Execution backend** (`execution::AbstractExecutionBackend`) — **how the outer work** (shell/mode
  loops and reductions) is parallelised: [`SerialBackend`](@ref), [`ThreadedBackend`](@ref)
  (OhMyThreads), [`DistributedBackend`](@ref) (`Distributed`/`SharedArrays`),
  [`GPUBackend`](@ref)`{B}` (KernelAbstractions), [`AutoBackend`](@ref) (best-available).

They compose: e.g. `spectral = FFTBackend(), execution = ThreadedBackend()` runs FFT nonlinear
terms with a threaded mediator loop. A typical call:

```julia
using FFTW, OhMyThreads   # load the two extensions

result = calculate_shell_to_shell_transfer(û, ks;
    binning   = LinearBinning(1.0),
    dealiasing = OrszagTwoThirds(),
    spectral  = FFTBackend(),
    execution = ThreadedBackend())
```

| Spectral backend | Dependency | Best for |
|---|---|---|
| [`DirectSumBackend`](@ref) | None | Reference results, debugging, tiny grids (the oracle) |
| [`FFTBackend`](@ref) | `FFTW` | Production spectral diagnostics on regular periodic grids |
| [`NUFFTBackend`](@ref) | `FINUFFT` | Scattered / non-uniform Cartesian data |
| [`SHTBackend`](@ref) | `FastSphericalHarmonics` | Regular latitude–longitude grids |
| [`NUFSHTBackend`](@ref) | `NUFSHT` | Scattered spherical observations |

| Execution backend | Dependency | Best for |
|---|---|---|
| [`SerialBackend`](@ref) | None | Default; small/medium grids |
| [`ThreadedBackend`](@ref) | `OhMyThreads` | Multi-core mediator/triad loops |
| [`DistributedBackend`](@ref) | `Distributed` + `SharedArrays` | Many-process single-node |
| [`GPUBackend`](@ref) | `KernelAbstractions` + vendor pkg | Large grids on GPU (CUDA validated) |
| [`AutoBackend`](@ref) | Varies | Automatic best-available execution |

---

## Dealiasing

Pseudospectral products alias high wavenumbers back onto resolved modes. Every nonlinear-term
entry point takes a `dealiasing::AbstractDealiasing` strategy (a **type**, so dispatch — not a
boolean — selects the path):

- [`OrszagTwoThirds`](@ref) **(default)** — zero the upper third of every input field *before*
  forming the product (exact on retained modes `|k| < N/3`). Works with any spectral backend.
- [`PaddedThreeHalves`](@ref) — exact `3/2` zero-padding: embed into a `3/2`-sized grid, multiply,
  truncate back. No aliasing at all on any retained mode. **FFT-only** (`DirectSumBackend +
  PaddedThreeHalves` throws).
- [`NoDealiasing`](@ref) — raw product; only for analytic single-triad tests where no aliasing
  can occur.

```julia
calculate_spectral_flux(û, ks; dealiasing = PaddedThreeHalves(), spectral = FFTBackend())
```

Aliasing breaks conservation (`Σ_k T(k) ≈ 0`), so the test suite asserts conservation under the
dealiased paths.

---

## Spectral backends

### DirectSumBackend

The reference implementation: direct `O(N²)` summation for the nonlinear term. No external
dependencies, always available, and the oracle every fast path is tested against.

```julia
calculate_spectral_flux(û, ks; binning = LinearBinning(1.0), spectral = DirectSumBackend())
```

**When to use:** debugging, correctness verification, very small grids (N ≤ 16).

### FFTBackend

Production fast path via FFTW. Reduces the nonlinear term from `O(N²)` to `O(Nᴰ log N)`:
IFFT to physical space → pointwise product → FFT back → apply the dealiasing strategy. Plans are
stored in the workspace and applied with `mul!`/`ldiv!`.

```julia
using FFTW   # loads the extension automatically
calculate_spectral_flux(û, ks; binning = LinearBinning(1.0), spectral = FFTBackend())
```

**When to use:** standard production runs on regular periodic grids (N ≥ 32). Also the only backend
that supports `PaddedThreeHalves` dealiasing.

### NUFFTBackend / SHTBackend / NUFSHTBackend

Front-ends for non-uniform Cartesian (FINUFFT), regular spherical (FastSphericalHarmonics), and
scattered spherical (NUFSHT) data. They transform input data to regular Fourier/spherical-harmonic
coefficients, then delegate to the core spectral diagnostics.

```julia
using FastSphericalHarmonics
result = calculate_energy_transfer(
    SpectralFluxMethod(LinearBinning(1.0)),
    velocity_fields, coords, (Nθ,); spectral = SHTBackend())
```

---

## Execution backends

### SerialBackend

Single-threaded, no dependencies. The default `execution` for every diagnostic.

### ThreadedBackend

Multi-threaded via OhMyThreads with thread-local accumulators (no locks). Parallelises the outer
loop over mediator shells (shell-to-shell), receiver modes (mode-to-mode), and triads (TOD).

```julia
using OhMyThreads
calculate_shell_to_shell_transfer(û, ks; binning = LinearBinning(1.0),
    spectral = FFTBackend(), execution = ThreadedBackend())
```

!!! note "FFTW intra-transform threads ≠ `ThreadedBackend`"
    `ThreadedBackend` parallelises the **outer** shell/mode loop. FFTW also has its own
    **intra-transform** multithreading, set globally with `FFTW.set_num_threads(n)`, which speeds
    up each individual FFT. The two are orthogonal and compose; enabling FFTW threads never changes
    results (asserted by the test suite). Don't oversubscribe — with `execution = ThreadedBackend()`
    each task already runs a transform, so leave FFTW at one thread (or partition cores between the
    two levels).

### DistributedBackend

Multi-process via `Distributed` + `SharedArrays`, using `@distributed (+)` reduction over mediator
shells / mode chunks.

```julia
using Distributed, SharedArrays
addprocs(4); @everywhere using FlowInvariantTransfer
calculate_shell_to_shell_transfer(SharedArray(û), ks;
    binning = LinearBinning(1.0), execution = DistributedBackend())
```

### GPUBackend

Device-generic GPU execution via KernelAbstractions — custom `@kernel` functions for transfer
density, shell accumulation (`Atomix.@atomic` scatter-add), and triad loops; all buffers allocated
with `similar(velocity_hat, …)` so they follow the input array type.

```julia
using KernelAbstractions, CUDA
û_gpu = CuArray(û); ks_gpu = map(CuArray, ks)
calculate_shell_to_shell_transfer(û_gpu, ks_gpu;
    binning = LinearBinning(1.0), execution = GPUBackend(CUDABackend()))
```

**Currently validated on CUDA.** AMDGPU/Metal ride the same KernelAbstractions kernels but are not
yet hardware-validated.

### AutoBackend

Selects the best available execution backend at call time (distributed → threaded → serial). The
spectral fast path is chosen independently from whichever spectral extension is loaded.

---

## Distributed with MPI — two axes

The single-node `DistributedBackend` above shares one array across processes. For genuine
**multi-process / multi-node** work there are two distinct ways to distribute, and the package
provides one entry point for each (loaded by `using MPI` — the pencil axis also needs
`PencilFFTs, PencilArrays`). MPI.jl bundles its own `mpiexec`, so a launcher works out of the box;
both paths are validated single-machine with `mpiexec -n 2`.

### Batch axis — many independent inputs ([`mpi_batch_map`](@ref))

When you have *many* snapshots (a time series) that each fit in one node's memory, distribute the
**set of inputs**: each rank applies `f` to a round-robin subset, then results are **collated** in
original order (default) or reduced. No communication during each item's computation — embarrassingly
parallel. This is the common post-processing mode.

```julia
using FlowInvariantTransfer, FFTW, MPI
MPI.Init()

f(û) = calculate_spectral_flux(û, ks; binning = LinearBinning(dk), spectral = FFTBackend()).flux

series = mpi_batch_map(f, snapshots)                 # Vector of per-snapshot fluxes, in order
mean_Π = mpi_batch_map(f, snapshots; reduce = :mean) # ensemble average instead of collation
```

`reduce` accepts `:gather` (default), `:sum`, `:mean`, or a binary combiner function; the combined
result is returned on every rank.

### Pencil axis — one grid too big for a node ([`pencil_spectral_flux`](@ref))

When a *single* snapshot's grid doesn't fit on one node, split **that grid** into pencils (one slab
per rank). The pseudospectral nonlinear term then needs transpose/all-to-all communication, handled
by a PencilFFTs distributed FFT. The per-shell KE transfer is `MPI.Allreduce`d to a global result
identical on every rank — equal to the serial `calculate_spectral_flux` on the same field (validated
to machine precision).

```julia
using FlowInvariantTransfer, MPI, PencilFFTs, PencilArrays
MPI.Init()

plan = build_pencil_plan((N, N), MPI.COMM_WORLD)     # auto-balanced process grid
u    = ntuple(_ -> allocate_input(plan), 2)          # fill each rank's LOCAL portion of u, v
res  = pencil_spectral_flux(u, plan, ks; binning = LinearBinning(dk))
```

The two axes are complementary and compose (a batch of large grids = batch axis over pencil-axis
groups). The pencil path currently covers kinetic energy on isotropic `|k|` shells.

---

## Backend Support Matrix

`spectral` ∈ {DirectSum, FFT}; `execution` ∈ {Serial, Threaded, Distributed, GPU}.

| Diagnostic | DirectSum | FFT | Serial | Threaded | Distributed | GPU |
|-----------|:---------:|:---:|:------:|:--------:|:-----------:|:---:|
| Spectral flux Π(K) | ✓ | ✓ | ✓ | ✓ | — | — |
| Shell-to-shell T(n,m) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Smooth band-to-band T(K,Q) | ✓ | ✓ | ✓ | — | — | — |
| Mode-to-mode S(k\|p) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Partial / decomposed fluxes | ✓ | ✓ | ✓ | — | — | — |
| TOD | ✓ | ✓ | ✓ | ✓ | — | — |
| Coarse-graining Π_ℓ(x) | — | — | — | — | — | — |

Notes: the net per-mode transfer `T(k)` and the magnitude matrix `T(K,Q)` are routed through the
fast FFT spectral-flux / shell-to-shell paths (exact, `O(Nᴰ log N)`); the fully mode-resolved
`S(k|p)` tensor is the only query that needs the `O(N^{2D})` brute loop (guarded by a mode-count
limit, `force=true` to override). Coarse-graining flux is provided entirely by the
CoarseGrainingEnergyFluxes extension and has its own parallelism model.

---

## Extension Loading

Extensions load automatically when you `using` their trigger package:

```julia
using FlowInvariantTransfer  # lean core only (DirectSumBackend, SerialBackend)
using FFTW                   # → FFTBackend, PaddedThreeHalves
using OhMyThreads            # → ThreadedBackend
using KernelAbstractions     # → GPUBackend (+ a vendor pkg, e.g. CUDA)
using MPI                    # → mpi_batch_map (batch axis)
using PencilFFTs, PencilArrays  # (+ MPI) → pencil_spectral_flux / build_pencil_plan (pencil axis)
using HelmholtzDecomposition # → decompose_field / Helmholtz partial fluxes
using FINUFFT                # → NUFFTBackend
using FastSphericalHarmonics # → SHTBackend
using NUFSHT                 # → NUFSHTBackend
```

Calling a backend whose extension isn't loaded gives a clear error:

```
ArgumentError: Threaded mode-to-mode transfer requires OhMyThreads. Run `using OhMyThreads` to load the extension.
```
