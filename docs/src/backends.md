```@meta
CurrentModule = FlowInvariantTransfer
```

# Backends, Dealiasing & Extensions

Transfer diagnostics are configured along **two orthogonal axes**, plus a dealiasing
strategy. This page documents each, when to use it, and how extensions are loaded. The snippets
assume these names are in scope:

```julia
using FlowInvariantTransfer
using FlowInvariantTransfer.Types: LinearBinning, OrszagTwoThirds, PaddedThreeHalves, SphericalTransferMethod
using FlowInvariantTransfer.SpectralBackends: DirectSumSpectralBackend, FFTSpectralBackend
using FlowInvariantTransfer.ComputationalBackends: ThreadedBackend, DistributedBackend, GPUBackend
```

## Two backend axes

*Which transform* computes a diagnostic and *how that work is run* are separate choices:

- **Transform** (`transform::AbstractSpectralBackend`) — **which transform** computes the
  pseudospectral nonlinear term: [`DirectSumSpectralBackend`](@ref SpectralBackends.DirectSumSpectralBackend) (direct DFT, no deps, the correctness
  oracle), [`FFTSpectralBackend`](@ref SpectralBackends.FastFourierTransformSpectralBackend) (FFTW, the `O(Nᴰ log N)` workhorse), the NUFFT library tags
  `FlowTransformBindings.NonuniformFFTsBackend` / `FINUFFTBackend` (scattered Cartesian), [`FSHTSpectralBackend`](@ref SpectralBackends.FastSphericalHarmonicsTransformSpectralBackend) (regular spherical), [`NUFSHTSpectralBackend`](@ref SpectralBackends.NonUniformFastSphericalHarmonicsTransformSpectralBackend)
  (scattered spherical).
- **Execution backend** (`execution::AbstractExecutionBackend`) — **how the outer work** (shell/mode
  loops and reductions) is parallelised: [`SerialBackend`](@ref ComputationalBackends.SerialBackend), [`ThreadedBackend`](@ref ComputationalBackends.ThreadedBackend)
  (OhMyThreads), [`DistributedBackend`](@ref ComputationalBackends.DistributedBackend) (`Distributed`),
  [`GPUBackend`](@ref ComputationalBackends.GPUBackend)`{B}` (KernelAbstractions), [`AutoBackend`](@ref ComputationalBackends.AutoBackend) (best-available).

They compose: e.g. `transform = FFTSpectralBackend(), execution = ThreadedBackend()` runs FFT nonlinear
terms with a threaded mediator loop. A typical call:

```julia
using FFTW: FFTW
using OhMyThreads: OhMyThreads   # load the two extensions

result = calculate_shell_to_shell_transfer(û, ks;
    binning   = LinearBinning(1.0),
    dealiasing = OrszagTwoThirds(),
    transform = FFTSpectralBackend(),
    execution = ThreadedBackend())
```

| Transform | Dependency | Best for |
|---|---|---|
| [`DirectSumSpectralBackend`](@ref SpectralBackends.DirectSumSpectralBackend) | None | Reference results, debugging, tiny grids (the oracle) |
| [`FFTSpectralBackend`](@ref SpectralBackends.FastFourierTransformSpectralBackend) | `FFTW` | Production spectral diagnostics on regular periodic grids |
| `FlowTransformBindings.NonuniformFFTsBackend` / `FINUFFTBackend` | `NonuniformFFTs` / `FINUFFT` | Scattered / non-uniform Cartesian data |
| [`FSHTSpectralBackend`](@ref SpectralBackends.FastSphericalHarmonicsTransformSpectralBackend) | `FastSphericalHarmonics` | Regular latitude–longitude grids |
| [`NUFSHTSpectralBackend`](@ref SpectralBackends.NonUniformFastSphericalHarmonicsTransformSpectralBackend) | `NUFSHT` | Scattered spherical observations |

| Execution backend | Dependency | Best for |
|---|---|---|
| [`SerialBackend`](@ref ComputationalBackends.SerialBackend) | None | Default; small/medium grids |
| [`ThreadedBackend`](@ref ComputationalBackends.ThreadedBackend) | `OhMyThreads` | Multi-core mediator/triad loops |
| [`DistributedBackend`](@ref ComputationalBackends.DistributedBackend) | `Distributed` | Julia worker processes |
| [`GPUBackend`](@ref ComputationalBackends.GPUBackend) | `KernelAbstractions` + vendor pkg | Large grids on GPU (CUDA validated) |
| [`AutoBackend`](@ref ComputationalBackends.AutoBackend) | Varies | Automatic best-available execution |

---

## Dealiasing

Pseudospectral products alias high wavenumbers back onto resolved modes. Every nonlinear-term
entry point takes a `dealiasing::AbstractDealiasing` strategy, a type on which the path dispatches:

- [`OrszagTwoThirds`](@ref) **(default)** — keep `|m_d| ≤ ⌊(N_d−1)/3⌋` of every input field
  *before* forming the product, the largest band whose products alias outside it (exact on the retained
  modes). Works with any transform.
- [`PaddedThreeHalves`](@ref) — exact `3/2` zero-padding: embed into a `3/2`-sized grid, multiply,
  truncate back. No aliasing at all on any retained mode. Works with any transform; the direct sum
  forms the padded product by direct sums on the padded grid.
- [`NoDealiasing`](@ref) — raw product; only for analytic single-triad tests where no aliasing
  can occur.

```julia
calculate_spectral_flux(û, ks; dealiasing = PaddedThreeHalves(), transform = FFTSpectralBackend())
```

Aliasing breaks conservation (`Σ_k T(k) ≈ 0`); the dealiased paths conserve.

---

## Transforms

### DirectSumSpectralBackend

The reference implementation: direct `O(N²ᴰ)` summation for the nonlinear term. No external
dependencies, always available, and the oracle every fast path is tested against.

```julia
calculate_spectral_flux(û, ks; binning = LinearBinning(1.0), transform = DirectSumSpectralBackend())
```

**When to use:** debugging, correctness verification, very small grids (N ≤ 16).

### FFTSpectralBackend

Production fast path via FFTW. Reduces the nonlinear term from `O(N²ᴰ)` to `O(Nᴰ log N)`:
inverse FFT to physical space → pointwise product → FFT back → apply the dealiasing strategy. Plans
are stored in the workspace and applied with `mul!`.

```julia
using FFTW: FFTW   # loads the extension automatically
calculate_spectral_flux(û, ks; binning = LinearBinning(1.0), transform = FFTSpectralBackend())
```

**When to use:** standard production runs on regular periodic grids (N ≥ 32).

### NUFFT libraries / FSHTSpectralBackend / NUFSHTSpectralBackend

Front-ends for non-uniform Cartesian (FlowTransformBindings' NUFFT plans, on NonuniformFFTs or
FINUFFT), regular spherical (FastSphericalHarmonics), and scattered spherical (NUFSHT) data. The
NUFFT front-ends reconstruct uniform Fourier coefficients from scattered samples and hand them to the
Cartesian diagnostics; the spherical front-ends compute the transfer in the spherical-harmonic degree
spectrum.

```julia
using FFTW: FFTW
using FastSphericalHarmonics: FastSphericalHarmonics
result = calculate_energy_transfer(SphericalTransferMethod(), ζ)   # ζ of size (lmax+1, 2lmax+1)
```

---

## Execution backends

### SerialBackend

Single-threaded, no dependencies. The default `execution` for every diagnostic.

### ThreadedBackend

Multi-threaded via OhMyThreads with thread-local accumulators (no locks). Parallelises the outer
loop over mediator shells (shell-to-shell), slots `p` (mode-to-mode), bands (band-to-band), channel
pairs (partial fluxes), triads (TOD), and the snapshots of every batch entry.

```julia
using OhMyThreads: OhMyThreads
calculate_shell_to_shell_transfer(û, ks; binning = LinearBinning(1.0),
    transform = FFTSpectralBackend(), execution = ThreadedBackend())
```

!!! note "FFTW intra-transform threads"
    `ThreadedBackend` parallelises the **outer** shell/mode loop. FFTW's **intra-transform** thread
    count is fixed per plan when the package builds it: one thread wherever an outer loop or a batch
    is threaded, `Threads.nthreads()` for the single-field spectral flux and compressible transfer
    under `execution = ThreadedBackend()`. The global `FFTW.set_num_threads(n)` does not change
    these plans.

### DistributedBackend

Multi-process via `Distributed`: the outer loop (mediator shells, slots, bands, channel pairs,
triads, or the mode→shell scatter of the spectral flux) is partitioned across worker processes with
`pmap` / `@distributed` and reduced on the caller. The work shipped to the workers captures plain
arrays, and every worker needs `using FlowInvariantTransfer`.

```julia
using Distributed: Distributed
Distributed.addprocs(4); Distributed.@everywhere using FlowInvariantTransfer
calculate_shell_to_shell_transfer(û, ks;
    binning = LinearBinning(1.0), execution = DistributedBackend())
```

### GPUBackend

Device-generic GPU execution via KernelAbstractions — `@kernel` functions for the transfer density,
the shell accumulation (`KA.@atomic` scatter-add) and the slot isolation of mode-to-mode. Every
buffer lives in the backend's memory: a device-resident field is used where it is, and a host field
is copied to the device once per call (once per snapshot in a batch, through one staging buffer). The
per-shell results and domain means come back on the host; the mode-resolved arrays of mode-to-mode
and the pointwise fields of the coarse-graining diagnostics stay on the device. The Triadic Orthogonal Decomposition and compressible paths run as device-generic broadcasts
and linear algebra (GPUArraysCore extension).

```julia
using FFTW: FFTW
using KernelAbstractions: KernelAbstractions
using CUDA: CUDA
û_gpu = CUDA.CuArray(û)            # the wavenumber axes `ks` are passed as they are
calculate_shell_to_shell_transfer(û_gpu, ks;
    binning = LinearBinning(1.0), transform = FFTSpectralBackend(),
    execution = GPUBackend(CUDA.CUDABackend()))
calculate_compressible_flux(û, ρ̂, ks;   # host fields, copied to the device
    binning = LinearBinning(1.0), transform = FFTSpectralBackend(),
    execution = GPUBackend(CUDA.CUDABackend()))
```

The device kernels (transfer density for KE/helicity/enstrophy, shell reduction) and their dispatch
are **validated on the KernelAbstractions CPU backend** (`GPUBackend(KA.CPU())`) and on JLArrays
against the serial reference to machine precision. The on-device FFT needs a real GPU: pass
`transform = FFTSpectralBackend()` with a `CuArray` input so cuFFT rides `AbstractFFTs`. The same
kernels target AMDGPU and Metal; `gpu/` holds the CUDA scripts for hardware runs.

```julia
using KernelAbstractions: KernelAbstractions
result = calculate_shell_to_shell_transfer(û, ks;
    binning = LinearBinning(1.0), execution = GPUBackend(KA.CPU()))  # CPU backend: same kernels, no GPU
```

### AutoBackend

Resolves at call time to `ThreadedBackend()` when the OhMyThreads extension is loaded and Julia has
more than one thread, else to `SerialBackend()` (`FlowInvariantTransfer.Types.resolve_execution`).
The transform resolves independently: `transform = AutoSpectralBackend()` is the FFT when the FFTW
extension is loaded (`FlowInvariantTransfer.Types.resolve_transform`).

---

## Distributed with MPI — two axes

`DistributedBackend` above spreads one call's outer loop over Julia worker processes. For MPI
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
using FlowInvariantTransfer: FlowInvariantTransfer
using FFTW: FFTW
using MPI: MPI
MPI.Init()

f(û) = calculate_spectral_flux(û, ks; binning = LinearBinning(dk), transform = FFTSpectralBackend()).flux

series = mpi_batch_map(f, snapshots)                 # Vector of per-snapshot fluxes, in order
mean_Π = mpi_batch_map(f, snapshots; reduce = :mean) # ensemble average
```

`reduce` accepts `:gather` (default), `:sum`, `:mean`, or a binary combiner function; the combined
result is returned on every rank.

### Pencil axis — one grid too big for a node ([`pencil_spectral_flux`](@ref FlowInvariantTransfer.pencil_spectral_flux))

When a *single* snapshot's grid doesn't fit on one node, split **that grid** into pencils (one slab
per rank). The pseudospectral nonlinear term then needs transpose/all-to-all communication, handled
by a PencilFFTs distributed FFT. The per-shell KE transfer is `MPI.Allreduce`d to a global result
identical on every rank — equal to the serial `calculate_spectral_flux` on the same field (validated
to machine precision).

```julia
using FlowInvariantTransfer: FlowInvariantTransfer
using MPI: MPI
using PencilFFTs: PencilFFTs
using PencilArrays: PencilArrays
MPI.Init()

plan = build_pencil_plan((N, N), MPI.COMM_WORLD)     # auto-balanced process grid
u    = ntuple(_ -> PencilFFTs.allocate_input(plan), 2)   # fill each rank's local portion of u, v
res  = pencil_spectral_flux(u, plan, ks; binning = LinearBinning(dk))
```

The two axes are complementary and compose (a batch of large grids = batch axis over pencil-axis
groups). The pencil path covers every invariant (KE / helicity / enstrophy), every dealiasing
(`PaddedThreeHalves` forms the product on a second, padded pencil plan), and the `ShellMagnitude`
geometries (isotropic `|k|`, perpendicular `k_⊥`, parallel `k_∥`), and `execution=MPIBackend(inner)`
sets the per-rank local backend — so `MPIBackend(GPUBackend(dev))` runs a device-resident pencil
(multi-GPU; the local shell reduction becomes an on-device scatter-add).

---

## Backend Support Matrix

`transform` ∈ {DirectSum, FFT}; `execution` ∈ {Serial, Threaded, Distributed, GPU}.

Transform axis = which transform builds the nonlinear term (every diagnostic supports
DirectSum + FFT). Execution axis = which parallelism runs the outer loop; only the diagnostics with
an outer loop over shells/triads expose it (the rest run serially over the single FFT path).

| Diagnostic | DirectSum | FFT | Serial | Threaded | Distributed | GPU |
|-----------|:---------:|:---:|:------:|:--------:|:-----------:|:---:|
| Spectral flux Π(K) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Shell-to-shell T(n,m) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Mode-to-mode S(k\|p) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Smooth band-to-band T(K,Q) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Partial / decomposed fluxes | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Compressible T_u(k) | ✓ | ✓ | ✓ | ✓* | ✓* | ✓ |
| TOD | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

`✓` verified (serial parity to machine precision). Coarse-graining, spherical, and the MPI
batch/pencil layer are documented separately below (they are not `transform × execution` diagnostics).

Notes:
- **Threaded / GPU** are wired for every Fourier diagnostic. The loop methods (shell-to-shell,
  mode-to-mode, band-to-band, partial) thread the outer shell/slot/band/pair loop and on a GPU run
  their per-mode transfer density through a KernelAbstractions device kernel; spectral flux threads or
  runs on the device the mode→shell reduction, and copies the per-shell spectrum to the host for the
  cumulative sum. `✓*` compressible is a single FFT pipeline: its `Threaded` path threads the FFTs, and
  its `Distributed` path computes the transfer + channel set and the pressure-dilatation on separate
  workers. Its GPU path is a device-generic broadcast pipeline + `GPUArraysCore` reductions. TOD's GPU
  path is the serial triad loop on device arrays (`GPUArraysCore` methods; the small per-triad
  eigenproblem runs on the host). GPU verified on `GPUBackend(KA.CPU())` and on JLArrays device-generic
  building blocks (cuFFT rides `AbstractFFTs` on a `CuArray` by construction).
- **Distributed** partitions the outer loop of every diagnostic across worker processes (the
  mode→shell scatter for spectral flux), with `DistributedBackend(ThreadedBackend())` threading each
  worker. For a grid too large for one node, use the **MPI pencil axis** (below).
- The fully mode-resolved `S(k|p)` tensor is the only query needing the `O(N^{2D})` brute loop;
  `T(k)`/`T(K,Q)`/`Π(K)` use the fast FFT paths.
- `execution = AutoBackend()` resolves to threaded when the OhMyThreads extension is loaded and Julia
  has more than one thread, else serial (`FlowInvariantTransfer.Types.resolve_execution`).
- **MPI** (batch + pencil axes) is a separate distribution layer (above). The pencil path supports every
  invariant (KE/helicity/enstrophy) and every `ShellMagnitude` geometry, with a 0-alloc `PencilWorkspace`
  for snapshot sweeps. **Coarse-graining** flux is provided by the CoarseGrainingEnergyFluxes extension
  (its own parallelism model); scattered coarse-graining uses the NUFFT libraries and scattered
  spherical transfer NUFSHT (FSH for regular spherical grids).

---

## Extension Loading

Extensions load automatically when you `using` their trigger package:

```julia
using FlowInvariantTransfer: FlowInvariantTransfer  # lean core only (DirectSumSpectralBackend, SerialBackend)
using FFTW: FFTW                   # → the FFTSpectralBackend path
using OhMyThreads: OhMyThreads            # → ThreadedBackend
using Distributed: Distributed            # → DistributedBackend
using KernelAbstractions: KernelAbstractions     # → GPUBackend (+ a vendor pkg, e.g. CUDA)
using MPI: MPI                    # → mpi_batch_map (batch axis)
using PencilFFTs: PencilFFTs
using PencilArrays: PencilArrays  # (+ MPI) → pencil_spectral_flux / build_pencil_plan (pencil axis)
using HelmholtzDecomposition: HelmholtzDecomposition # → decompose_field / Helmholtz partial fluxes
using CoarseGrainingEnergyFluxes: CoarseGrainingEnergyFluxes  # → calculate_coarse_graining_flux
using FlowFieldSpectra: FlowFieldSpectra     # → to_spectral / from_spectral on uniform grids
using NonuniformFFTs: NonuniformFFTs         # → FlowTransformBindings.NonuniformFFTsBackend (scattered Cartesian)
using FINUFFT: FINUFFT                # → FlowTransformBindings.FINUFFTBackend (scattered Cartesian)
using FastSphericalHarmonics: FastSphericalHarmonics # (+ FFTW) → regular-grid spherical transfer
using NUFSHT: NUFSHT                 # → scattered-point spherical transfer
```

Calling a backend whose extension isn't loaded gives a clear error:

```
ArgumentError: Threaded mode-to-mode transfer requires OhMyThreads. Run `using OhMyThreads` to load the extension.
```
