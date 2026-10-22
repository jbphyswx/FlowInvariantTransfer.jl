module FlowInvariantTransfer

# First, so its `__init__` selects the OpenMP runtime's thread-local mode before FastTransforms loads
# that runtime; see `FlowTransformBindings.with_fasttransforms_threads`.
using FlowTransformBindings: FlowTransformBindings
using PrecompileTools: PrecompileTools
using ComputationalBackends: ComputationalBackends
using SpectralBackends: SpectralBackends

public FlowTransformBindings
public ComputationalBackends
public SpectralBackends

# ---------------------------------------------------------------------------
# Submodule includes
# ---------------------------------------------------------------------------

include("Types.jl")
include("Utils.jl")
include("Invariants.jl")
include("Decomposition.jl")
include("ShellBinning.jl")
include("Filters.jl")
include("Workspaces.jl")
include("NonlinearTerm.jl")
include("SpectralFlux.jl")
include("CoarseGrainingFlux.jl")
include("ShellToShell/ShellToShellTransfer.jl")
include("BandTransfer.jl")
include("ScaleToScale/TriadicOrthogonalDecomposition/TriadicOrthogonalDecomposition.jl")
include("ScaleToScale/ModeToModeTransfer.jl")
include("Compressible/CompressibleTransfer.jl")
include("Spherical/SphericalTransfer.jl")

# ---------------------------------------------------------------------------
# Public surface: `export` covers the entry-point functions; the method, binning, geometry,
# dealiasing and result types are reached qualified (`FlowInvariantTransfer.Types.LinearBinning`).
# ---------------------------------------------------------------------------

using .SpectralFlux: SpectralFlux, calculate_spectral_flux, calculate_spectral_flux!, calculate_spectral_flux_batch, calculate_scalar_flux, calculate_scalar_flux!, calculate_partial_fluxes, calculate_partial_fluxes!, calculate_partial_fluxes_batch, calculate_helical_partial_fluxes, calculate_helical_partial_fluxes!
using .Compressible: Compressible, calculate_compressible_flux, calculate_compressible_flux!,
                     calculate_compressible_flux_batch
using .Spherical: Spherical, calculate_spherical_transfer, calculate_spherical_transfer!, calculate_divergent_spherical_transfer, calculate_divergent_spherical_transfer!
using .CoarseGrainingFlux: CoarseGrainingFlux, calculate_coarse_graining_flux, calculate_coarse_graining_flux!, calculate_coarse_graining_flux_batch,
                           calculate_band_energies, calculate_enstrophy_flux
using .ShellToShellTransfer: ShellToShellTransfer, calculate_shell_to_shell_transfer, calculate_shell_to_shell_transfer!, calculate_shell_to_shell_transfer_batch, calculate_scalar_shell_to_shell_transfer, calculate_scalar_shell_to_shell_transfer!
using .BandTransfer: BandTransfer, calculate_band_to_band_transfer, calculate_band_to_band_transfer!, calculate_band_to_band_transfer_batch
using .ModeToModeTransfer: ModeToModeTransfer, calculate_mode_to_mode_transfer, calculate_mode_to_mode_transfer!, calculate_scalar_mode_to_mode_transfer, calculate_scalar_mode_to_mode_transfer!
using .TriadicOrthogonalDecomposition: TriadicOrthogonalDecomposition, triadic_orthogonal_decomposition, triadic_orthogonal_decomposition!

export calculate_spectral_flux, calculate_spectral_flux!, calculate_spectral_flux_batch, calculate_scalar_flux, calculate_scalar_flux!, calculate_partial_fluxes, calculate_partial_fluxes!, calculate_partial_fluxes_batch, calculate_helical_partial_fluxes, calculate_helical_partial_fluxes!
export calculate_compressible_flux, calculate_compressible_flux!, calculate_compressible_flux_batch
export calculate_spherical_transfer, calculate_spherical_transfer!, calculate_divergent_spherical_transfer, calculate_divergent_spherical_transfer!
export calculate_coarse_graining_flux, calculate_coarse_graining_flux!, calculate_coarse_graining_flux_batch
export calculate_band_energies, calculate_enstrophy_flux
export calculate_shell_to_shell_transfer, calculate_shell_to_shell_transfer!, calculate_shell_to_shell_transfer_batch, calculate_scalar_shell_to_shell_transfer, calculate_scalar_shell_to_shell_transfer!
export calculate_band_to_band_transfer, calculate_band_to_band_transfer!, calculate_band_to_band_transfer_batch
export calculate_mode_to_mode_transfer, calculate_mode_to_mode_transfer!, calculate_scalar_mode_to_mode_transfer, calculate_scalar_mode_to_mode_transfer!
export triadic_orthogonal_decomposition, triadic_orthogonal_decomposition!
export calculate_energy_transfer

# ---------------------------------------------------------------------------
# Extension stubs for MPI / PencilFFTs (distributed)
# ---------------------------------------------------------------------------

"""
    mpi_batch_map(f, items; comm=MPI.COMM_WORLD, reduce=:gather, root=0)

Distribute an embarrassingly-parallel **batch** of independent inputs across MPI ranks: each
rank applies `f` to a round-robin subset of `items` (e.g. snapshots of a time series), then the
per-item outputs are combined. With `reduce=:gather` (default) the results are **collated** into
one `Vector` in the original order of `items`, returned on every rank; `reduce=:sum`/`:mean`
returns the element-wise reduction (the outputs of `f` must support `+`, and `/` for `:mean`); a
callable `reduce` is applied as a binary combiner. For one grid split across ranks, see
[`pencil_spectral_flux`](@ref FlowInvariantTransfer.pencil_spectral_flux).

Requires `using MPI` to load the extension.
"""
function mpi_batch_map(args...; kwargs...)
    throw(ArgumentError("mpi_batch_map requires MPI. Run `using MPI` to load the extension."))
end

"""
    pencil_spectral_flux(u_phys, plan, ks; comm=MPI.COMM_WORLD, binning, dealiasing, invariant, geometry, execution)
        -> (centers, shell_edges, transfer_spectrum, flux)

Distributed spectral transfer/flux for a single grid split across MPI ranks along the **pencil
axis**: `u_phys` is the physical-space velocity as an `NTuple{D, PencilArray}` (one component per
entry; each rank owns a pencil of the global grid) and `plan` is a distributed FFT plan from
[`build_pencil_plan`](@ref). The pseudospectral nonlinear term is evaluated with a transpose-based
distributed FFT (PencilFFTs), the transfer density is shell-binned locally, and the per-shell
spectrum is `MPI.Allreduce`d to a global result identical on every rank (matching the serial
[`calculate_spectral_flux`](@ref) on the same field). For many independent snapshots, see
[`mpi_batch_map`](@ref).

Requires `using MPI, PencilFFTs, PencilArrays` to load the extension.
"""
function pencil_spectral_flux(args...; kwargs...)
    throw(ArgumentError("pencil_spectral_flux requires MPI, PencilFFTs and PencilArrays. Run `using MPI, PencilFFTs, PencilArrays`."))
end

"""
    build_pencil_plan(ns, comm=MPI.COMM_WORLD; T=Float64) -> PencilFFTPlan

Convenience constructor for the distributed FFT plan used by
[`pencil_spectral_flux`](@ref FlowInvariantTransfer.pencil_spectral_flux), with an auto-balanced MPI
process grid. The velocity is real, so axis 1 is a real-to-complex transform and each rank's spectral
pencil holds the non-redundant `k₁ ≥ 0` half: pass the matching wavenumber tuple,
`Utils.wavenumber_grid(ns, Ls; real = true)`. Requires `using MPI, PencilFFTs, PencilArrays`.
"""
function build_pencil_plan(args...; kwargs...)
    throw(ArgumentError("build_pencil_plan requires MPI, PencilFFTs and PencilArrays. Run `using MPI, PencilFFTs, PencilArrays`."))
end

"""
    PencilWorkspace(plan, ks, comm=MPI.COMM_WORLD; binning, dealiasing=OrszagTwoThirds(),
                    geometry=IsotropicShells(), invariant=KineticEnergy(), execution=SerialBackend())
        -> PencilWorkspace

Reusable workspace for [`pencil_spectral_flux!`](@ref): the (geometry/dealiasing/binning-fixed)
wavenumber grids + shell structure and every per-snapshot scratch field, so a repeated distributed
flux on the same plan allocates ~0 beyond the small per-shell result vectors. Requires
`using MPI, PencilFFTs, PencilArrays`.

`execution` composes an inner local backend with the MPI (pencil) axis, e.g.
`execution=MPIBackend(GPUBackend(dev))` for a per-rank device pencil (multi-GPU) — the local shell
reduction then runs as an on-device scatter-add. `SerialBackend()` (default) keeps the host scalar path.
"""
function PencilWorkspace(args...; kwargs...)
    throw(ArgumentError("PencilWorkspace requires MPI, PencilFFTs and PencilArrays. Run `using MPI, PencilFFTs, PencilArrays`."))
end

"""
    pencil_spectral_flux!(ws::PencilWorkspace, u_phys; invariant=ws.invariant)
        -> (centers, shell_edges, transfer_spectrum, flux)

In-place distributed pencil spectral flux reusing `ws` (0 alloc beyond the small result vectors) — build
`ws` once and loop over snapshots of the same distributed grid. `invariant` must be the one `ws` was
built for. Requires `using MPI, PencilFFTs, PencilArrays`.
"""
function pencil_spectral_flux!(args...; kwargs...)
    throw(ArgumentError("pencil_spectral_flux! requires MPI, PencilFFTs and PencilArrays. Run `using MPI, PencilFFTs, PencilArrays`."))
end

export mpi_batch_map, pencil_spectral_flux, pencil_spectral_flux!, build_pencil_plan, PencilWorkspace

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

include("Scattered.jl")

# Scattered-Cartesian coarse-graining entry. The 4-positional (…, scatter_coords, ms) form separates it
# from the uniform-grid `calculate_energy_transfer` methods.
function calculate_energy_transfer(method::Types.CoarseGrainingFluxMethod, velocity_fields::Tuple,
                                   scatter_coords::Tuple, ms::Tuple; kwargs...)
    return nufft_coarse_graining_flux(velocity_fields, scatter_coords, method.scale, method.filter, ms; kwargs...)
end

# Physical-space entry for the spectral-flux / shell / mode family on scattered and uniform-grid
# Cartesian data. The 4-positional (…, coords, ms) form separates it from the coefficient methods
# (…, velocity_hat, ks). `_physical_energy_transfer` dispatches on `transform`: a NUFFT backend
# reconstructs from scattered samples, any other goes through the FlowFieldSpectra uniform-grid path.
const _SpectralFamilyMethod = Union{Types.SpectralFluxMethod, Types.ShellToShellTransferMethod, Types.ModeToModeTransferMethod}

function calculate_energy_transfer(method::_SpectralFamilyMethod, velocity_fields::Tuple, domain, ms::Tuple;
                                   transform::SpectralBackends.AbstractSpectralBackend = SpectralBackends.AutoSpectralBackend(),
                                   kwargs...)
    return _physical_energy_transfer(transform, method, velocity_fields, domain, ms; kwargs...)
end

# Scattered NUFFT branch: reconstruct `velocity_hat` on the scattered samples, then run the uniform
# diagnostic. `Ls` (required) is the periodic domain size; `execution` drives both the reconstruction
# and the diagnostic.
function _physical_energy_transfer(spectral::SpectralBackends.AbstractNonUniformFastFourierTransformSpectralBackend,
                                   method::_SpectralFamilyMethod, velocity_fields::Tuple, scatter_coords::Tuple, ms::Tuple;
                                   Ls::Tuple, tol::Union{Nothing, Real} = nothing,
                                   weights::Union{Nothing, AbstractVector} = nothing,
                                   execution::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.SerialBackend(),
                                   kwargs...)
    velocity_hat, ks = to_spectral(velocity_fields, scatter_coords, ms; transform = spectral, Ls = Ls, tol = tol,
                                   weights = weights, execution = execution)
    return calculate_energy_transfer(method, velocity_hat, ks; execution = execution, kwargs...)
end

# Uniform-grid branch: the FlowFieldSpectra extension adds the concrete method. This varargs method
# throws when that extension is not loaded; its signature stays distinct from the extension's, since an
# extension that overwrites a parent method fails to precompile.
_physical_energy_transfer(spectral::SpectralBackends.AbstractSpectralBackend, args...; kwargs...) = throw(ArgumentError(
    "uniform-grid physical → spectral transfer requires `using FlowFieldSpectra`; for scattered (non-uniform) " *
    "Cartesian data pass a NUFFT library (transform = FlowTransformBindings.NonuniformFFTsBackend() / " *
    "FlowTransformBindings.FINUFFTBackend()) with `Ls`."))

export nufft_coarse_graining_flux, nufft_coarse_graining_flux!, nufft_coarse_graining_flux_batch,
       NUFFTCoarseGrainingWorkspace
export ToSpectralWorkspace
export NUFFTToSpectralWorkspace, to_spectral, to_spectral!

# ---------------------------------------------------------------------------
# Unified entry point
# ---------------------------------------------------------------------------

"""
    calculate_energy_transfer(method, velocity_data, coords_or_ks; kwargs...)

Run the diagnostic `method` names and return its result container.

# Arguments
- `method::AbstractEnergyTransferMethod`: Which method to use:
  - `SpectralFluxMethod(binning)` — spectral flux Π(K) → `SpectralFluxResult`
  - `ShellToShellTransferMethod(binning)` — shell-to-shell T(n,m) → `ShellToShellResult`
  - `ModeToModeTransferMethod(; invariant)` — mode-to-mode S(k|p) → `ModeToModeTriadResult`
  - `CoarseGrainingFluxMethod(filter, ℓ)` — coarse-graining flux Π_ℓ(x) → `CoarseGrainingFluxResult`
  - `TriadicOrthogonalDecompositionMethod(; nfft, noverlap, nmode)` — called as
    `calculate_energy_transfer(method, X; kwargs...)` on snapshots `X` →
    `TriadicOrthogonalDecompositionResult`
- `velocity_data`: For spectral methods, a complex array of size `(ms..., D)` containing
  Fourier coefficients; for coarse-graining, a tuple of D real physical-space arrays.
- `coords_or_ks`: For spectral methods, a tuple of 1D wavenumber vectors; for
  coarse-graining, a tuple of 1D coordinate vectors.

Physical-space Cartesian data uses the 4-positional form
`calculate_energy_transfer(method, velocity_fields, coords, ms; transform, …)`, which routes on
`transform`: a NUFFT library (`FlowTransformBindings.NonuniformFFTsBackend()` /
`FlowTransformBindings.FINUFFTBackend()`, with `Ls`) reconstructs from **scattered** samples, while any
other backend transforms **uniform-grid** data through
FlowFieldSpectra (`using FlowFieldSpectra`; `coords` are the per-axis grid vectors). Coarse-graining has
the same 4-positional scattered route. Spherical data uses
`calculate_energy_transfer(SphericalTransferMethod(), ζ, (θ, φ); lmax, …)` (scattered, NUFSHT) or a
regular colatitude–longitude grid (FastSphericalHarmonics).

# Examples
```julia
using FlowInvariantTransfer, FlowFieldSpectra, FFTW
const FIT = FlowInvariantTransfer

# Spectral flux on a 32×32 periodic domain
N = 32; L = 2π
x = range(0.0, L; length=N+1)[1:N]
u = [cos(xi) * sin(yj) for xi in x, yj in x]
v = [-sin(xi) * cos(yj) + 0.3 * sin(2yj) for xi in x, yj in x]
û, ks = to_spectral((u, v), (x, x))                  # real fields: the half layout

result = calculate_energy_transfer(FIT.Types.SpectralFluxMethod(FIT.Types.LinearBinning(2π/L)), û, ks)
```
"""
function calculate_energy_transfer(
    method::Types.SpectralFluxMethod,
    velocity_hat::AbstractArray{<:Complex},
    ks;
    kwargs...,
)
    return calculate_spectral_flux(velocity_hat, ks; binning=method.binning, kwargs...)
end

function calculate_energy_transfer(
    method::Types.CoarseGrainingFluxMethod,
    velocity_fields::Tuple,
    coords_vecs::Tuple;
    kwargs...,
)
    return calculate_coarse_graining_flux(
        velocity_fields, coords_vecs, method.scale, method.filter; kwargs...)
end

function calculate_energy_transfer(
    method::Types.ShellToShellTransferMethod,
    velocity_hat::AbstractArray{<:Complex},
    ks;
    kwargs...,
)
    return calculate_shell_to_shell_transfer(velocity_hat, ks; binning=method.binning, kwargs...)
end

function calculate_energy_transfer(
    method::Types.ModeToModeTransferMethod,
    velocity_hat::AbstractArray{<:Complex},
    ks;
    kwargs...,
)
    return calculate_mode_to_mode_transfer(velocity_hat, ks;
        invariant=method.invariant, kwargs...)
end

function calculate_energy_transfer(
    method::Types.TriadicOrthogonalDecompositionMethod,
    X::AbstractArray;
    kwargs...,
)
    return triadic_orthogonal_decomposition(X;
        window=method.nfft, noverlap=method.noverlap, nmode=method.nmode, kwargs...)
end

# ---------------------------------------------------------------------------
# Physical-space front door for uniform-grid Cartesian data: (u, v[, w]) → (velocity_hat, ks)
# ---------------------------------------------------------------------------

"""
    to_spectral(velocity_fields::Tuple, coords_vecs::Tuple; transform=SpectralBackends.AutoSpectralBackend()) -> (velocity_hat, ks)

Forward-transform physical-space velocity components sampled on a **uniform, periodic, tensor-product
Cartesian grid** into the Fourier-coefficient input `(velocity_hat, ks)` consumed by every Cartesian
flux diagnostic ([`calculate_spectral_flux`](@ref), [`calculate_shell_to_shell_transfer`](@ref),
[`calculate_mode_to_mode_transfer`](@ref), [`calculate_band_to_band_transfer`](@ref),
[`calculate_partial_fluxes`](@ref), [`calculate_compressible_flux`](@ref)). This is the physical-space
entry point for gridded data: the diagnostics operate on coefficients, so a real field is transformed
here once with the package's `û = fft(u)/Nᵈ` normalization (so `E(k) = ½|û|²`) and the FFTW `fftfreq`
wavenumber convention — you do not build `û`/`ks` by hand.

`coords_vecs` are the **1D coordinate vectors** `(x, y[, z])` of the grid, one vector of length `nₐ`
per axis, each `nₐ` periodic samples `x₁, x₁ + Δ, …, x₁ + (nₐ−1)Δ` of the period `L = nₐΔ` (uniform to
`√eps` of their float type, `Float32` or `Float64`). A vector that ends at `x₁ + L` repeats its first
sample; drop that sample, or pass a `FlowGeometries` grid, which carries its period.
Scattered (non-uniform) Cartesian data takes the 3-argument form with a NUFFT library
(`transform = FlowTransformBindings.NonuniformFFTsBackend()` or `FlowTransformBindings.FINUFFTBackend()`),
and spherical data [`calculate_spherical_transfer`](@ref); see [`spectral_geometry`](@ref).

# Keyword Arguments
- `transform::SpectralBackends.AbstractSpectralBackend = SpectralBackends.AutoSpectralBackend()`: the analysis transform. `SpectralBackends.FFTSpectralBackend()` needs
  `using FFTW` (cuFFT is used automatically for device-array inputs); `SpectralBackends.DirectSumSpectralBackend()` is the
  dependency-free `O(N²ᴰ)` reference (tiny grids only).

# Returns
`(velocity_hat, ks)` — `velocity_hat` is `(nₐ..., D)` complex in the input array's backend (a device
field yields a device coefficient array); `ks` a tuple of `D` wavenumber vectors.

# Example
```julia
using FlowInvariantTransfer, FlowFieldSpectra, FFTW
û, ks = to_spectral((u, v), (x, y))
result = calculate_spectral_flux(û, ks)
```

Pass a scalar (density / pressure / passive scalar) as a 1-tuple: `ρ̂, _ = to_spectral((ρ,), coords)`.
"""
function to_spectral(velocity_fields::Tuple, coords_vecs::Tuple;
                     transform::SpectralBackends.AbstractSpectralBackend = SpectralBackends.AutoSpectralBackend(),
                     real_layout::Bool = eltype(velocity_fields[1]) <: Real, kwargs...)
    ws = ToSpectralWorkspace(velocity_fields, coords_vecs; transform = transform,
                             real_layout = real_layout, kwargs...)
    return to_spectral!(ws, velocity_fields)
end

"""
    ToSpectralWorkspace{TF, RA, CA, KS}

Reusable transform + buffers for the uniform-grid [`to_spectral!`](@ref): the analysis plans, the
real `(ns…, D)` staging array the components are gathered into, and the `(ms…, D)` coefficient output
the result aliases. Build once for a grid and reuse across snapshots — the plans and both buffers are
then built once for the whole series.
"""
struct ToSpectralWorkspace{TF, RA, CA, KS}
    tf::TF
    field_phys::RA       # (ns..., D) real staging
    û::CA                # (ms..., D) coefficients (the result aliases this)
    ks::KS
end
Base.show(io::IO, ::ToSpectralWorkspace) = print(io, "ToSpectralWorkspace(…)")
Base.show(io::IO, ::MIME"text/plain", w::ToSpectralWorkspace) = show(io, w)

"""
    ToSpectralWorkspace(velocity_fields, coords_vecs; transform, real_layout=true)

Build the reusable uniform-grid analysis workspace. `real_layout = true` (the default for real input
fields) stores the non-redundant half spectrum; see [`to_spectral`](@ref).
"""
ToSpectralWorkspace(velocity_fields::Tuple, coords_vecs::Tuple; kwargs...) =
    _to_spectral_workspace_on_axes(velocity_fields, coords_vecs; kwargs...)

"""
    _to_spectral_workspace_on_axes(velocity_fields, coords_vecs; kwargs...)

Analysis workspace over per-axis coordinate vectors. Overridden by
`FlowInvariantTransferFlowFieldSpectraExt`, which reads the uniform periodic box `L = N·Δ` those axes
describe and builds the grid form on it.
"""
function _to_spectral_workspace_on_axes(args...; kwargs...)
    throw(ArgumentError(
        "to_spectral is the physical → spectral front-end and transforms through FlowFieldSpectra.jl. " *
        "Run `using FlowFieldSpectra` to load the extension."))
end

"""
    to_spectral!(ws::ToSpectralWorkspace, velocity_fields) -> (velocity_hat, ks)

In-place uniform-grid analysis reusing `ws` — no plan rebuild, no per-call buffers. The returned
`velocity_hat` aliases `ws.û` and a later call overwrites it.
"""
function to_spectral!(ws::ToSpectralWorkspace, velocity_fields::Tuple)
    D = size(ws.û, ndims(ws.û))
    length(velocity_fields) == D || throw(DimensionMismatch(
        "to_spectral! got $(length(velocity_fields)) fields; workspace was built for $D components"))
    nd = ndims(ws.field_phys) - 1
    colons = ntuple(_ -> Colon(), nd)
    for c in 1:D
        view(ws.field_phys, colons..., c) .= velocity_fields[c]
    end
    ws.tf.dft!(ws.û, ws.field_phys)
    return (ws.û, ws.ks)
end

"""
    ToSpectralWorkspace(velocity_hat::AbstractArray, ks::Tuple; transform=AutoSpectralBackend())

Build the workspace from the coefficient side, sizing the physical buffer from `ks` — the form
[`from_spectral`](@ref) needs. The same workspace drives both directions, so a loop that transforms
back and forth builds one.
"""
ToSpectralWorkspace(velocity_hat::AbstractArray, ks::Tuple; kwargs...) =
    _to_spectral_workspace_on_coeffs(velocity_hat, ks; kwargs...)

"""
    _to_spectral_workspace_on_coeffs(velocity_hat, ks; kwargs...)

Workspace sized from the coefficient side. Overridden by
`FlowInvariantTransferFlowFieldSpectraExt`, which rebuilds the periodic box `ks` describes — `ns` from
the axes' own full lengths, `L_d = 2π/Δk_d` — and plans both directions on it.
"""
function _to_spectral_workspace_on_coeffs(args...; kwargs...)
    throw(ArgumentError(
        "from_spectral is the spectral → physical synthesis and transforms through FlowFieldSpectra.jl. " *
        "Run `using FlowFieldSpectra` to load the extension."))
end

"""
    from_spectral!(ws::ToSpectralWorkspace, velocity_hat) -> velocity_phys

In-place synthesis `u = Σ û e^{ik·x}`, the inverse of [`to_spectral!`](@ref), reusing `ws`. Returns
the component-last `(ns…, D)` real array `ws.field_phys`, overwritten by a later call — the same
stacked shape `to_spectral!` returns its coefficients in, and allocation-free on repeat.
"""
function from_spectral!(ws::ToSpectralWorkspace, velocity_hat::AbstractArray)
    nd = ndims(ws.field_phys) - 1
    D  = size(ws.field_phys, nd + 1)
    size(velocity_hat, nd + 1) == D || throw(DimensionMismatch(
        "from_spectral! got $(size(velocity_hat, nd + 1)) components; workspace was built for $D"))
    size(velocity_hat)[1:nd] == size(ws.û)[1:nd] || throw(DimensionMismatch(
        "coefficient size $(size(velocity_hat)[1:nd]) does not match the workspace grid $(size(ws.û)[1:nd])"))
    ws.tf.idft!(ws.field_phys, velocity_hat)
    return ws.field_phys
end

"""
    from_spectral(velocity_hat, ks; transform=AutoSpectralBackend()) -> velocity_fields

Synthesise physical-space components from Fourier coefficients on a uniform periodic Cartesian grid —
the inverse of [`to_spectral`](@ref), and its normalization: `u = Σ û e^{ik·x}`. Accepts either
spectral layout, reading which from `ks`, and returns real fields.

The components come back as a tuple of views into one backing array, mirroring the tuple
[`to_spectral`](@ref) takes. Build a [`ToSpectralWorkspace`](@ref) and call [`from_spectral!`](@ref)
for the stacked array and plan reuse across snapshots.
"""
function from_spectral(velocity_hat::AbstractArray, ks::Tuple;
                       transform::SpectralBackends.AbstractSpectralBackend = SpectralBackends.AutoSpectralBackend())
    ws = ToSpectralWorkspace(velocity_hat, ks; transform = transform)
    phys = from_spectral!(ws, velocity_hat)
    nd = length(ks)
    colons = ntuple(_ -> Colon(), nd)
    return ntuple(c -> view(phys, colons..., c), size(phys, nd + 1))
end

export from_spectral, from_spectral!

# ---------------------------------------------------------------------------
# Precompilation workload
# ---------------------------------------------------------------------------

PrecompileTools.@setup_workload begin
    N = 4
    L = 2π
    ks_1d = Utils.wavenumber_grid((N,), (L,))[1]
    ks = (ks_1d, ks_1d)
    # minimal 4×4×2 spectral data
    û = zeros(ComplexF64, N, N, 2)
    û[2, 1, 1] = 0.5    # single mode u
    û[1, 2, 2] = 0.5    # single mode v

    # The FFTW extension is not loaded during precompilation; naming the direct-sum backend keeps
    # `resolve_transform`'s fallback warning out of the precompile output.
    ds = SpectralBackends.DirectSumSpectralBackend()

    PrecompileTools.@compile_workload begin
        _ = calculate_spectral_flux(û, ks; binning=Types.LinearBinning(2π/L),
                dealiasing=Types.NoDealiasing(), transform=ds)
        _ = calculate_shell_to_shell_transfer(û, ks; binning=Types.LinearBinning(2π/L),
                dealiasing=Types.NoDealiasing(), transform=ds)
        _ = Utils.wavenumber_grid((N,N), (L,L))
    end
end

end # module FlowInvariantTransfer