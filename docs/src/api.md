# API Reference

## Unified Entry Point

```@docs
calculate_energy_transfer
```

## Spectral Flux

```@docs
calculate_spectral_flux
calculate_spectral_flux!
calculate_scalar_flux
```

## Shell-to-Shell Transfer

```@docs
calculate_shell_to_shell_transfer
calculate_shell_to_shell_transfer!
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
```

## Partial Fluxes (decomposition-resolved)

```@docs
calculate_partial_fluxes
calculate_helical_partial_fluxes
```

## Compressible Energy Transfer

```@docs
calculate_compressible_flux
```

## Coarse-Graining Flux

```@docs
calculate_coarse_graining_flux
nufft_coarse_graining_flux
```

## Distributed (MPI)

```@docs
mpi_batch_map
pencil_spectral_flux
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

```@docs
FlowInvariantTransfer.Decomposition.decompose_field
FlowInvariantTransfer.Decomposition.helmholtz_project_spectral!
```

## Method Types

```@docs
AbstractEnergyTransferMethod
SpectralFluxMethod
ShellToShellTransferMethod
ModeToModeTransferMethod
CoarseGrainingFluxMethod
TriadicOrthogonalDecompositionMethod
```

## Invariant Types

```@docs
AbstractInvariant
KineticEnergy
Helicity
Enstrophy
PassiveScalar
```

## Decomposition Types

```@docs
AbstractFieldDecomposition
NoDecomposition
HelmholtzDecomposition
RotationalDecomposition
DivergentDecomposition
HelicalDecomposition
ToroidalPoloidalDecomposition
```

## Dealiasing Strategies

```@docs
AbstractDealiasing
NoDealiasing
OrszagTwoThirds
PaddedThreeHalves
```

## Result Types

```@docs
SpectralFluxResult
CompressibleFluxResult
ShellToShellResult
ModeToModeTriadResult
CoarseGrainingFluxResult
CoarseGrainingFluxResultWithDiagnostics
TriadicOrthogonalDecompositionResult
```

## Workspace Types

```@docs
NonlinearTermWorkspace
SpectralFluxWorkspace
ShellToShellWorkspace
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
AbstractShellBinning
LinearBinning
LogarithmicBinning
DyadicBinning
CustomBinning
SmoothBands
```

## Shell Geometry

```@docs
AbstractShellGeometry
ShellMagnitude
IsotropicShells
PerpendicularShells
ParallelShells
```

## Spectral (Transform) Backends

```@docs
AbstractSpectralBackend
DirectSumBackend
FFTBackend
NUFFTBackend
SHTBackend
NUFSHTBackend
```

## Execution (Parallelism) Backends

```@docs
AbstractExecutionBackend
SerialBackend
ThreadedBackend
DistributedBackend
GPUBackend
AutoBackend
local_backend
resolve_execution
```

## Filters

```@docs
AbstractFilter
SharpSpectralFilter
GaussianFilter
TopHatFilter
FlowInvariantTransfer.Filters.filter_response
FlowInvariantTransfer.Filters.apply_filter_spectral
FlowInvariantTransfer.Filters.apply_filter_spectral!
```
