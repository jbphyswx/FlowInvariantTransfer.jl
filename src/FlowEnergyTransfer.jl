module FlowEnergyTransfer

using PrecompileTools: PrecompileTools

# ---------------------------------------------------------------------------
# Submodule includes
# ---------------------------------------------------------------------------

include("Types.jl")
include("Utils.jl")
include("ShellBinning.jl")
include("Filters.jl")
include("Workspaces.jl")
include("NonlinearTerm.jl")
include("SpectralFlux.jl")
include("CoarseGrainingFlux.jl")
include("ShellToShellTransfer.jl")

# ---------------------------------------------------------------------------
# Re-exports
# ---------------------------------------------------------------------------

using .Types:
    AbstractEnergyTransferMethod,
    SpectralFluxMethod,
    CoarseGrainingFluxMethod,
    ShellToShellTransferMethod,
    AbstractFilter,
    SharpSpectralFilter,
    GaussianFilter,
    TopHatFilter,
    AbstractShellBinning,
    LinearBinning,
    LogarithmicBinning,
    DyadicBinning,
    CustomBinning,
    AbstractExecutionBackend,
    SerialBackend,
    ThreadedBackend,
    FFTBackend,
    NUFFTBackend,
    SpectralFluxResult,
    CoarseGrainingFluxResult,
    CoarseGrainingFluxResultWithDiagnostics,
    ShellToShellResult

export AbstractEnergyTransferMethod, SpectralFluxMethod, CoarseGrainingFluxMethod, ShellToShellTransferMethod
export AbstractFilter, SharpSpectralFilter, GaussianFilter, TopHatFilter
export AbstractShellBinning, LinearBinning, LogarithmicBinning, DyadicBinning, CustomBinning
export AbstractExecutionBackend, SerialBackend, ThreadedBackend, FFTBackend, NUFFTBackend
export SpectralFluxResult, CoarseGrainingFluxResult, CoarseGrainingFluxResultWithDiagnostics, ShellToShellResult

using .Utils:
    wavenumber_grid,
    wavenumber_magnitude_grid,
    dealiasing_mask,
    dealiasing_mask!,
    validate_velocity_input,
    validate_uniform_grid,
    domain_size_from_coords

export wavenumber_grid, wavenumber_magnitude_grid, dealiasing_mask, dealiasing_mask!
export validate_velocity_input, validate_uniform_grid, domain_size_from_coords

using .ShellBinning: shell_edges, shell_centers, n_shells, shell_mask, assign_shells
export shell_edges, shell_centers, n_shells, shell_mask, assign_shells

using .Filters: filter_response, apply_filter_spectral, apply_filter_spectral!
export filter_response, apply_filter_spectral, apply_filter_spectral!

using .Workspaces: NonlinearTermWorkspace, SpectralFluxWorkspace, ShellToShellWorkspace
export NonlinearTermWorkspace, SpectralFluxWorkspace, ShellToShellWorkspace

using .NonlinearTerm: compute_nonlinear_term, compute_nonlinear_term!
export compute_nonlinear_term, compute_nonlinear_term!

using .SpectralFlux: calculate_spectral_flux, calculate_spectral_flux!
using .CoarseGrainingFlux: calculate_coarse_graining_flux
using .ShellToShellTransfer: calculate_shell_to_shell_transfer, calculate_shell_to_shell_transfer!

export calculate_spectral_flux, calculate_spectral_flux!
export calculate_coarse_graining_flux
export calculate_shell_to_shell_transfer, calculate_shell_to_shell_transfer!
export calculate_energy_transfer

# ---------------------------------------------------------------------------
# Extension stubs for CairoMakie
# ---------------------------------------------------------------------------

"""
    plot_energy_transfer(result; kwargs...)

Plot an energy transfer result.
Requires CairoMakie to be loaded.
"""
function plot_energy_transfer(args...; kwargs...)
    throw(ArgumentError("plot_energy_transfer requires CairoMakie. Run `using CairoMakie`."))
end

export plot_energy_transfer

# ---------------------------------------------------------------------------
# Unified entry point
# ---------------------------------------------------------------------------

"""
    calculate_energy_transfer(method, velocity_data, coords_or_ks; kwargs...)

Unified entry point for all energy transfer computations.

# Arguments
- `method::AbstractEnergyTransferMethod`: Which method to use:
  - `SpectralFluxMethod(binning)` — spectral flux Π(K)
  - `CoarseGrainingFluxMethod(filter, ℓ)` — coarse-graining flux Π_ℓ(x)
  - `ShellToShellTransferMethod(binning)` — shell-to-shell T(n,m)
- `velocity_data`: For spectral methods, a complex array of size `(ns..., D)` containing
  Fourier coefficients; for coarse-graining, a tuple of D real physical-space arrays.
- `coords_or_ks`: For spectral methods, a tuple of 1D wavenumber vectors; for
  coarse-graining, a tuple of 1D coordinate vectors.

# Returns
Method-specific result container: `SpectralFluxResult`, `CoarseGrainingFluxResult`,
or `ShellToShellResult`.

# Examples
```julia
using FlowEnergyTransfer, FFTW

# Spectral flux on a 32×32 periodic domain
N = 32; L = 2π
x = range(0.0, L; length=N+1)[1:N]
y = range(0.0, L; length=N+1)[1:N]
u = [cos(x) for x in x, y in y]
v = [sin(y) for x in x, y in y]
û = cat(FFTW.fft(u), FFTW.fft(v); dims=3) ./ N^2  # (N,N,2)
ks = wavenumber_grid((N,N), (L,L))

result = calculate_energy_transfer(SpectralFluxMethod(LinearBinning(2π/L)), û, ks)
```
"""
function calculate_energy_transfer(
    method::SpectralFluxMethod,
    velocity_hat::AbstractArray{<:Complex},
    ks::Tuple;
    kwargs...,
)
    return calculate_spectral_flux(velocity_hat, ks; binning=method.binning, kwargs...)
end

function calculate_energy_transfer(
    method::CoarseGrainingFluxMethod,
    velocity_fields::Tuple,
    coords_vecs::Tuple;
    kwargs...,
)
    return calculate_coarse_graining_flux(
        velocity_fields, coords_vecs, method.scale, method.filter; kwargs...)
end

function calculate_energy_transfer(
    method::ShellToShellTransferMethod,
    velocity_hat::AbstractArray{<:Complex},
    ks::Tuple;
    kwargs...,
)
    return calculate_shell_to_shell_transfer(velocity_hat, ks; binning=method.binning, kwargs...)
end

# ---------------------------------------------------------------------------
# Precompilation workload (small grid to reduce TTFX)
# ---------------------------------------------------------------------------

PrecompileTools.@setup_workload begin
    N = 4
    L = 2π
    ks_1d = [Float64(k <= N÷2 ? k : k-N) * (2π/L) for k in 0:N-1]
    ks = (ks_1d, ks_1d)
    # minimal 4×4×2 spectral data
    û = zeros(ComplexF64, N, N, 2)
    û[2, 1, 1] = 0.5    # single mode u
    û[1, 2, 2] = 0.5    # single mode v

    PrecompileTools.@compile_workload begin
        _ = calculate_spectral_flux(û, ks; binning=LinearBinning(2π/L), dealiasing=false)
        _ = calculate_shell_to_shell_transfer(û, ks;
                binning=LinearBinning(2π/L), dealiasing=false, verify_antisymmetry=false)
        _ = wavenumber_grid((N,N), (L,L))
        _ = dealiasing_mask((N,N))
    end
end

end # module FlowEnergyTransfer