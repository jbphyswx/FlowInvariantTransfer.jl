# API Reference

## Unified Entry Point

```@docs
calculate_energy_transfer
```

## Spectral Flux

```@docs
calculate_spectral_flux
calculate_spectral_flux!
```

## Shell-to-Shell Transfer

```@docs
calculate_shell_to_shell_transfer
calculate_shell_to_shell_transfer!
```

## Coarse-Graining Flux

```@docs
calculate_coarse_graining_flux
```

## Nonlinear Term

```@docs
compute_nonlinear_term
compute_nonlinear_term!
```

## Workspace Structs

```@docs
NonlinearTermWorkspace
SpectralFluxWorkspace
ShellToShellWorkspace
```

## Result Structs

```@docs
SpectralFluxResult
ShellToShellResult
CoarseGrainingFluxResult
CoarseGrainingFluxResultWithDiagnostics
```

## Wavenumber Utilities

```@docs
wavenumber_grid
wavenumber_magnitude_grid
dealiasing_mask
dealiasing_mask!
```

## Shell Binning

```@docs
assign_shells
shell_edges
shell_centers
shell_mask
n_shells
```

## Binning Types

```@docs
AbstractShellBinning
LinearBinning
LogarithmicBinning
DyadicBinning
CustomBinning
```

## Backend Types

```@docs
AbstractExecutionBackend
SerialBackend
FFTBackend
ThreadedBackend
NUFFTBackend
```

## Filter Types

```@docs
AbstractFilter
SharpSpectralFilter
GaussianFilter
TopHatFilter
filter_response
apply_filter_spectral
apply_filter_spectral!
```
