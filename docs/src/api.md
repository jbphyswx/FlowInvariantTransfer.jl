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
field takes the half (`rfft`) layout by default — the non-redundant `k₁ ≥ 0` coefficients, with `ks[1]`
an `rfftfreq` axis — which every diagnostic consumes directly; pass `real_layout = false` for the full
complex spectrum. Build the workspace once to reuse the plans and buffers across snapshots.

The domain is given either as the per-axis coordinate vectors of a uniform periodic box, or as a
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
```

## Shell-to-Shell Transfer

```@docs
calculate_shell_to_shell_transfer
calculate_shell_to_shell_transfer!
calculate_shell_to_shell_transfer_batch
calculate_scalar_shell_to_shell_transfer
```

## Mode-to-Mode Triad Transfer

```@docs
calculate_mode_to_mode_transfer
calculate_scalar_mode_to_mode_transfer
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
calculate_enstrophy_flux
nufft_coarse_graining_flux
nufft_coarse_graining_flux_batch
```

## Scattered-Cartesian NUFFT

Two peer providers back the scattered NUFFT transform — `FlowInvariantTransfer.Types.FINUFFTBackend`
(`using FINUFFT`) and `FlowInvariantTransfer.Types.NonuniformFFTsBackend` (`using NonuniformFFTs`),
selected via the required `spectral` keyword. They drive the scattered coarse-graining flux and the
scattered→uniform `to_spectral` reconstruction; the workspace forms preset the plan + buffers for
allocation-light reuse.

```@docs
NUFFTCoarseGrainingWorkspace
nufft_coarse_graining_flux!
NUFFTToSpectralWorkspace
to_spectral!
FlowInvariantTransfer.Types.FINUFFTBackend
FlowInvariantTransfer.Types.NonuniformFFTsBackend
```

## Spherical Spectral Transfer

```@docs
calculate_spherical_transfer
calculate_spherical_transfer!
calculate_divergent_spherical_transfer
calculate_divergent_spherical_transfer!
```

## Distributed (MPI)

```@docs
mpi_batch_map
FlowInvariantTransfer.pencil_spectral_flux
build_pencil_plan
```

## Triadic Orthogonal Decomposition

```@docs
triadic_orthogonal_decomposition
FlowInvariantTransfer.TriadicOrthogonalDecomposition.hamming_window
FlowInvariantTransfer.TriadicOrthogonalDecomposition.hann_window
FlowInvariantTransfer.TriadicOrthogonalDecomposition.tukey_window
```

## Nonlinear Term

```@docs
FlowInvariantTransfer.NonlinearTerm.compute_nonlinear_term
FlowInvariantTransfer.NonlinearTerm.compute_nonlinear_term!
```

## Invariant Transfer Density

```@docs
FlowInvariantTransfer.Invariants.transfer_density
FlowInvariantTransfer.Invariants.transfer_density!
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

## Method Types

```@docs
FlowInvariantTransfer.Types.AbstractEnergyTransferMethod
FlowInvariantTransfer.Types.SpectralFluxMethod
FlowInvariantTransfer.Types.ShellToShellTransferMethod
FlowInvariantTransfer.Types.ModeToModeTransferMethod
FlowInvariantTransfer.Types.CoarseGrainingFluxMethod
FlowInvariantTransfer.Types.TriadicOrthogonalDecompositionMethod
FlowInvariantTransfer.Types.SphericalTransferMethod
FlowInvariantTransfer.Types.DivergentSphericalTransferMethod
```

## Invariant Types

```@docs
FlowInvariantTransfer.Types.AbstractInvariant
FlowInvariantTransfer.Types.KineticEnergy
FlowInvariantTransfer.Types.Helicity
FlowInvariantTransfer.Types.Enstrophy
FlowInvariantTransfer.Types.PassiveScalar
```

## Decomposition Types

```@docs
FlowInvariantTransfer.Types.AbstractFieldDecomposition
FlowInvariantTransfer.Types.NoDecomposition
FlowInvariantTransfer.Types.HelmholtzDecomposition
FlowInvariantTransfer.Types.RotationalDecomposition
FlowInvariantTransfer.Types.DivergentDecomposition
FlowInvariantTransfer.Types.HelicalDecomposition
FlowInvariantTransfer.Types.ToroidalPoloidalDecomposition
```

## Dealiasing Strategies

```@docs
FlowInvariantTransfer.Types.AbstractDealiasing
FlowInvariantTransfer.Types.NoDealiasing
FlowInvariantTransfer.Types.OrszagTwoThirds
FlowInvariantTransfer.Types.PaddedThreeHalves
```

## Result Types

```@docs
FlowInvariantTransfer.Types.SpectralFluxResult
FlowInvariantTransfer.Types.CompressibleFluxResult
FlowInvariantTransfer.Types.ShellToShellResult
FlowInvariantTransfer.Types.ModeToModeTriadResult
FlowInvariantTransfer.Types.CoarseGrainingFluxResult
FlowInvariantTransfer.Types.CoarseGrainingFluxResultWithDiagnostics
FlowInvariantTransfer.Types.TriadicOrthogonalDecompositionResult
FlowInvariantTransfer.Types.SphericalTransferResult
FlowInvariantTransfer.Types.DivergentSphericalTransferResult
```

## Workspace Types

```@docs
FlowInvariantTransfer.Workspaces.NonlinearTermWorkspace
FlowInvariantTransfer.Workspaces.SpectralFluxWorkspace
FlowInvariantTransfer.Workspaces.ShellToShellWorkspace
FlowInvariantTransfer.Compressible.CompressibleWorkspace
FlowInvariantTransfer.Spherical.SphericalTransferWorkspace
FlowInvariantTransfer.Spherical.ScatteredSphericalTransferWorkspace
FlowInvariantTransfer.Spherical.DivergentSphericalTransferWorkspace
FlowInvariantTransfer.Spherical.ScatteredDivergentSphericalTransferWorkspace
```

## Wavenumber Utilities

```@docs
FlowInvariantTransfer.Utils.wavenumber_grid
FlowInvariantTransfer.Utils.wavenumber_magnitude_grid
FlowInvariantTransfer.Utils.dealiasing_mask
FlowInvariantTransfer.Utils.dealiasing_mask!
```

## Shell Binning & Geometry

```@docs
FlowInvariantTransfer.ShellBinning.assign_shells
FlowInvariantTransfer.ShellBinning.shell_edges
FlowInvariantTransfer.ShellBinning.shell_centers
FlowInvariantTransfer.ShellBinning.n_shells
FlowInvariantTransfer.ShellBinning.shell_coordinate
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
scattered-Cartesian NUFFT transform has two peer providers, defined by FIT as symmetric subtypes of
`SpectralBackends.AbstractNonUniformFastFourierTransformSpectralBackend`: `FlowInvariantTransfer.Types.FINUFFTBackend`
(`using FINUFFT`) and `FlowInvariantTransfer.Types.NonuniformFFTsBackend` (`using NonuniformFFTs`).

## Execution (Parallelism) Backends

The execution tags are provided by the shared
[`ComputationalBackends`](https://github.com/jbphyswx/ComputationalBackends.jl) package (documented
there): `ComputationalBackends.SerialBackend`, `ThreadedBackend`, `DistributedBackend`, `MPIBackend`,
`GPUBackend`, and `AutoBackend`, all `<: ComputationalBackends.AbstractExecutionBackend`. FIT resolves
`ComputationalBackends.AutoBackend` through its own `FlowInvariantTransfer.Types.resolve_execution`
(threaded when the OhMyThreads extension is loaded and `Threads.nthreads() > 1`, else serial).

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
