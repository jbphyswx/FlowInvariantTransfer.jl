```@meta
CurrentModule = FlowInvariantTransfer
```

# API Reference

## Unified Entry Point

```@docs
calculate_energy_transfer
```

## Physical → Spectral

The diagnostics take Fourier coefficients, so gridded physical data is transformed here once. A real
field takes the half (`rfft`) layout: the non-redundant `k₁ ≥ 0` coefficients, with `ks[1]`
an `rfftfreq` axis. Complex input takes the full spectrum. Build the workspace once to reuse the plans
and buffers across snapshots.

The domain is given either as per-axis ranges describing a uniform periodic box, or as a
`FlowGeometries` grid. Passing a grid (`using FlowFieldSpectra`) selects the transform from the grid
itself through `FlowFieldSpectra.plan_spectrum` — the tensor-product FFT on a uniform Cartesian grid,
the FFT/NUFFT composite where some directions are stretched — and reads the period the grid declares.

```@docs
to_spectral
ToSpectralWorkspace
from_spectral
from_spectral!
```

## Spectral Flux

```@docs
calculate_spectral_flux
calculate_spectral_flux!
calculate_spectral_flux_batch
calculate_scalar_flux
calculate_scalar_flux!
```

## Shell-to-Shell Transfer

```@docs
calculate_shell_to_shell_transfer
calculate_shell_to_shell_transfer!
calculate_shell_to_shell_transfer_batch
calculate_scalar_shell_to_shell_transfer
calculate_scalar_shell_to_shell_transfer!
```

## Mode-to-Mode Triad Transfer

```@docs
calculate_mode_to_mode_transfer
calculate_mode_to_mode_transfer!
calculate_scalar_mode_to_mode_transfer
calculate_scalar_mode_to_mode_transfer!
FlowInvariantTransfer.ModeToModeTransfer.mode_weights
```

## Smooth Band-to-Band Transfer

```@docs
calculate_band_to_band_transfer
calculate_band_to_band_transfer!
calculate_band_to_band_transfer_batch
```

## Partial Fluxes (decomposition-resolved)

```@docs
calculate_partial_fluxes
calculate_partial_fluxes!
calculate_partial_fluxes_batch
calculate_helical_partial_fluxes
calculate_helical_partial_fluxes!
```

## Compressible Energy Transfer

```@docs
calculate_compressible_flux
calculate_compressible_flux!
calculate_compressible_flux_batch
```

## Coarse-Graining Flux

```@docs
calculate_coarse_graining_flux
calculate_coarse_graining_flux_batch
FlowInvariantTransfer.CoarseGrainingFlux.calculate_coarse_graining_flux!
FlowInvariantTransfer.CoarseGrainingFlux.CoarseGrainingFluxWorkspace
calculate_enstrophy_flux
nufft_coarse_graining_flux
nufft_coarse_graining_flux_batch
```

## Scattered-Cartesian NUFFT

The scattered NUFFTs run through FlowTransformBindings' plans, on the library the required `transform`
keyword names: `FlowTransformBindings.NonuniformFFTsBackend()` (`using NonuniformFFTs`) or
`FlowTransformBindings.FINUFFTBackend()` (`using FINUFFT`, and `using CUDA` for cuFINUFFT on device
points). They drive the scattered coarse-graining flux and the scattered→uniform `to_spectral`
reconstruction; the workspace forms hold the plans and buffers for reuse, and `close!` releases the
plans.

For real scattered input, every even requested mode count is widened by one and the result is the
packed half of that odd box. Complex input keeps the requested full box.

```@docs
NUFFTCoarseGrainingWorkspace
nufft_coarse_graining_flux!
NUFFTToSpectralWorkspace
to_spectral!
FlowInvariantTransfer.close!
```

## Spherical Spectral Transfer

```@docs
calculate_spherical_transfer
calculate_spherical_transfer!
calculate_divergent_spherical_transfer
calculate_divergent_spherical_transfer!
```

The spherical transfer on a masked `FlowGeometries` grid takes one of these as `inactive`.

```@docs
FlowInvariantTransfer.Types.AbstractInactiveCells
FlowInvariantTransfer.Types.ZeroFill
FlowInvariantTransfer.Types.Exclude
```

## Distributed (MPI)

```@docs
mpi_batch_map
FlowInvariantTransfer.pencil_spectral_flux
FlowInvariantTransfer.pencil_spectral_flux!
FlowInvariantTransfer.PencilWorkspace
build_pencil_plan
```

## Triadic Orthogonal Decomposition

```@docs
triadic_orthogonal_decomposition
FlowInvariantTransfer.TriadicOrthogonalDecomposition.triadic_orthogonal_decomposition!
FlowInvariantTransfer.TriadicOrthogonalDecomposition.TODWorkspace
FlowInvariantTransfer.TriadicOrthogonalDecomposition.hamming_window
FlowInvariantTransfer.TriadicOrthogonalDecomposition.hann_window
FlowInvariantTransfer.TriadicOrthogonalDecomposition.tukey_window
FlowInvariantTransfer.TriadicOrthogonalDecomposition.ModesOnly
FlowInvariantTransfer.TriadicOrthogonalDecomposition.WithCoefficients
FlowInvariantTransfer.TriadicOrthogonalDecomposition.WithAuxiliaryModes
FlowInvariantTransfer.TriadicOrthogonalDecomposition.ZeroMean
FlowInvariantTransfer.TriadicOrthogonalDecomposition.BlockMean
```

## Nonlinear Term

```@docs
FlowInvariantTransfer.NonlinearTerm.compute_nonlinear_term
FlowInvariantTransfer.NonlinearTerm.compute_nonlinear_term!
FlowInvariantTransfer.NonlinearTerm.synthesize_advecting!
FlowInvariantTransfer.NonlinearTerm.compute_advection!
FlowInvariantTransfer.NonlinearTerm.compute_rotational_term!
```

## Invariant Transfer Density

```@docs
FlowInvariantTransfer.Invariants.transfer_density
FlowInvariantTransfer.Invariants.transfer_density!
FlowInvariantTransfer.Invariants.transfer_density_scatter!
```

## Field Decomposition

`decompose_field` splits a field on either side of the transform. Given coefficients it returns
coefficients; given physical fields and a `FlowGeometries` grid it returns physical fields — the
Helmholtz family by solving on the grid, and the per-mode splits (helical, toroidal/poloidal) by
analysing, projecting and synthesising each component back.

```@docs
FlowInvariantTransfer.Decomposition.decompose_field
FlowInvariantTransfer.Decomposition.decompose_field!
FlowInvariantTransfer.Decomposition.helmholtz_project_spectral!
```

## Band Energies

The content of an invariant in each band, `E(n) = Σ_k w_n(k)·e(k)`. The band definition supplies the
weight `w_n` and nothing else: an `AbstractShellBinning` gives the sharp indicator of a shell, and
`SmoothBands` the graded partition of unity [`calculate_band_to_band_transfer`](@ref) moves energy
between, so passing the same `bands` to both describes one partition. A separate method takes physical
fields, a domain and a filter, where the weight is the filter's own transfer function and the sum runs
in real space (so it also serves masked and bounded domains).

```@docs
calculate_band_energies
```

## Wavenumber Utilities

The wavenumber axes, their fold weights and the dealiasing cutoffs are the packed layout of
[`FlowTransformBindings`](https://github.com/jbphyswx/FlowTransformBindings.jl) (documented there):
`FlowTransformBindings.RFFTAxis`, `FlowTransformBindings.FFTAxis`, `FlowTransformBindings.mode_fold`,
`FlowTransformBindings.dealias_cutoff`.

```@docs
FlowInvariantTransfer.Utils.wavenumber_grid
FlowInvariantTransfer.Utils.wavenumber_magnitude_grid
FlowInvariantTransfer.Utils.as_component_field
FlowInvariantTransfer.Utils.validate_velocity_input
FlowInvariantTransfer.Utils.validate_uniform_grid
FlowInvariantTransfer.Utils.domain_size_from_coords
```

## Shell Binning & Geometry

```@docs
FlowInvariantTransfer.ShellBinning.assign_shells
FlowInvariantTransfer.ShellBinning.shell_edges
FlowInvariantTransfer.ShellBinning.shell_centers
FlowInvariantTransfer.ShellBinning.n_shells
FlowInvariantTransfer.ShellBinning.shell_coordinate
FlowInvariantTransfer.ShellBinning.max_shell_coordinate
FlowInvariantTransfer.ShellBinning.default_binning
```

## Binning Types

```@docs
FlowInvariantTransfer.Types.AbstractShellBinning
FlowInvariantTransfer.Types.LinearBinning
FlowInvariantTransfer.Types.LogarithmicBinning
FlowInvariantTransfer.Types.DyadicBinning
FlowInvariantTransfer.Types.CustomBinning
FlowInvariantTransfer.Types.SmoothBands
```

## Shell Geometry

```@docs
FlowInvariantTransfer.Types.AbstractShellGeometry
FlowInvariantTransfer.Types.ShellMagnitude
FlowInvariantTransfer.Types.IsotropicShells
FlowInvariantTransfer.Types.PerpendicularShells
FlowInvariantTransfer.Types.ParallelShells
```

## Spectral (Transform) Backends

The transform-algorithm tags are provided by the shared
[`SpectralBackends`](https://github.com/jbphyswx/SpectralBackends.jl) package (documented there). FIT
selects the transform by the tag's geometry; the concrete types are `SpectralBackends.DirectSumSpectralBackend`,
`SpectralBackends.FFTSpectralBackend`, `SpectralBackends.FSHTSpectralBackend`, and
`SpectralBackends.NUFSHTSpectralBackend`, all `<: SpectralBackends.AbstractSpectralBackend`. The
scattered-Cartesian NUFFT libraries' tags come from
[`FlowTransformBindings`](https://github.com/jbphyswx/FlowTransformBindings.jl), as subtypes of
`SpectralBackends.AbstractNonUniformFastFourierTransformSpectralBackend`:
`FlowTransformBindings.FINUFFTBackend` (`using FINUFFT`) and
`FlowTransformBindings.NonuniformFFTsBackend` (`using NonuniformFFTs`). The `transform` keyword's
default, `SpectralBackends.AutoSpectralBackend`, resolves through FIT's own
`FlowInvariantTransfer.Types.resolve_transform` (the FFT when the FFTW extension is loaded, else the
direct sum).

## Execution (Parallelism) Backends

The execution tags are provided by the shared
[`ComputationalBackends`](https://github.com/jbphyswx/ComputationalBackends.jl) package (documented
there): `ComputationalBackends.SerialBackend`, `ThreadedBackend`, `DistributedBackend`, `MPIBackend`,
`GPUBackend`, and `AutoBackend`, all `<: ComputationalBackends.AbstractExecutionBackend`. FIT resolves
`ComputationalBackends.AutoBackend` through its own `FlowInvariantTransfer.Types.resolve_execution`
(threaded when the OhMyThreads extension is loaded and `Threads.nthreads() > 1`, else serial).
`AutoBackend()` is the execution default; a device-array input selects its device backend.

## Filters

```@docs
FlowInvariantTransfer.Types.AbstractFilter
FlowInvariantTransfer.Types.SharpSpectralFilter
FlowInvariantTransfer.Types.GaussianFilter
FlowInvariantTransfer.Types.TopHatFilter
FlowInvariantTransfer.Filters.filter_response
FlowInvariantTransfer.Filters.apply_filter_spectral
FlowInvariantTransfer.Filters.apply_filter_spectral!
```

## Plotting

```@docs
plot_energy_transfer
```
