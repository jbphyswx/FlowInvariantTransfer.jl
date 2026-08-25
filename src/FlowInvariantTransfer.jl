module FlowInvariantTransfer

using PrecompileTools: PrecompileTools
using ComputationalBackends: ComputationalBackends
using SpectralBackends: SpectralBackends

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

# AutoBackend resolution is FIT-owned in `Types.resolve_execution` (not a method on
# `ComputationalBackends.resolve_backend`, which would be type piracy) — see Types.jl.

# ---------------------------------------------------------------------------
# Public surface. Re-import the public names so `FlowInvariantTransfer.foo` resolves, but `export`
# ONLY the entry-point functions — a bare `using FlowInvariantTransfer` brings in the verbs
# (`calculate_*`, `triadic_orthogonal_decomposition`, `to_spectral`), never the method/binning/
# geometry/dealiasing/result TYPES, which stay reachable qualified (`FlowInvariantTransfer.LinearBinning`).
# ---------------------------------------------------------------------------

using .SpectralFlux: SpectralFlux, calculate_spectral_flux, calculate_spectral_flux!, calculate_spectral_flux_batch, calculate_scalar_flux, calculate_scalar_flux!, calculate_partial_fluxes, calculate_partial_fluxes!, calculate_helical_partial_fluxes, calculate_helical_partial_fluxes!
using .Compressible: Compressible, calculate_compressible_flux, calculate_compressible_flux!
using .Spherical: Spherical, calculate_spherical_transfer, calculate_spherical_transfer!, calculate_divergent_spherical_transfer, calculate_divergent_spherical_transfer!
using .CoarseGrainingFlux: CoarseGrainingFlux, calculate_coarse_graining_flux, calculate_coarse_graining_flux!, calculate_coarse_graining_flux_batch
using .ShellToShellTransfer: ShellToShellTransfer, calculate_shell_to_shell_transfer, calculate_shell_to_shell_transfer!, calculate_shell_to_shell_transfer_batch, calculate_scalar_shell_to_shell_transfer, calculate_scalar_shell_to_shell_transfer!
using .BandTransfer: BandTransfer, calculate_band_to_band_transfer, calculate_band_to_band_transfer!, calculate_band_to_band_transfer_batch
using .ModeToModeTransfer: ModeToModeTransfer, calculate_mode_to_mode_transfer, calculate_mode_to_mode_transfer!, calculate_scalar_mode_to_mode_transfer, calculate_scalar_mode_to_mode_transfer!
using .TriadicOrthogonalDecomposition: TriadicOrthogonalDecomposition, triadic_orthogonal_decomposition, triadic_orthogonal_decomposition!

export calculate_spectral_flux, calculate_spectral_flux!, calculate_spectral_flux_batch, calculate_scalar_flux, calculate_scalar_flux!, calculate_partial_fluxes, calculate_partial_fluxes!, calculate_helical_partial_fluxes, calculate_helical_partial_fluxes!
export calculate_compressible_flux, calculate_compressible_flux!
export calculate_spherical_transfer, calculate_spherical_transfer!, calculate_divergent_spherical_transfer, calculate_divergent_spherical_transfer!
export calculate_coarse_graining_flux, calculate_coarse_graining_flux!, calculate_coarse_graining_flux_batch
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
callable `reduce` is applied as a binary combiner. This is the "batch axis" of distribution —
orthogonal to the pencil axis ([`pencil_spectral_flux`](@ref FlowInvariantTransfer.pencil_spectral_flux)), which splits a single grid.

Requires `using MPI` to load the extension.
"""
function mpi_batch_map(args...; kwargs...)
    throw(ArgumentError("mpi_batch_map requires MPI. Run `using MPI` to load the extension."))
end

"""
    pencil_spectral_flux(u_phys, plan, ks; comm=MPI.COMM_WORLD, binning, dealiasing, invariant, geometry, execution) -> (centers, transfer_spectrum, flux)

Distributed spectral transfer/flux for a single grid split across MPI ranks along the **pencil
axis**: `u_phys` is the physical-space velocity as an `NTuple{D, PencilArray}` (one component per
entry; each rank owns a pencil of the global grid) and `plan` is a distributed FFT plan from
[`build_pencil_plan`](@ref). The pseudospectral nonlinear term is evaluated with a transpose-based
distributed FFT (PencilFFTs), the transfer density is shell-binned locally, and the per-shell
spectrum is `MPI.Allreduce`d to a global result identical on every rank (matching the serial
[`calculate_spectral_flux`](@ref) on the same field). Use this when one snapshot's grid is too
large for a single node; for many independent snapshots use [`mpi_batch_map`](@ref) instead.

Requires `using MPI, PencilFFTs, PencilArrays` to load the extension.
"""
function pencil_spectral_flux(args...; kwargs...)
    throw(ArgumentError("pencil_spectral_flux requires MPI, PencilFFTs and PencilArrays. Run `using MPI, PencilFFTs, PencilArrays`."))
end

"""
    build_pencil_plan(ns, comm=MPI.COMM_WORLD; T=Float64) -> PencilFFTPlan

Convenience constructor for the distributed complex-to-complex FFT plan used by
[`pencil_spectral_flux`](@ref FlowInvariantTransfer.pencil_spectral_flux), with an auto-balanced MPI process grid. Requires
`using MPI, PencilFFTs, PencilArrays`.
"""
function build_pencil_plan(args...; kwargs...)
    throw(ArgumentError("build_pencil_plan requires MPI, PencilFFTs and PencilArrays. Run `using MPI, PencilFFTs, PencilArrays`."))
end

"""
    PencilWorkspace(plan, ks, comm=MPI.COMM_WORLD; binning, dealiasing=OrszagTwoThirds(),
                    geometry=IsotropicShells(), execution=SerialBackend()) -> PencilWorkspace

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
    pencil_spectral_flux!(ws::PencilWorkspace, u_phys; invariant=KineticEnergy())
        -> (centers, transfer_spectrum, flux)

In-place distributed pencil spectral flux reusing `ws` (0 alloc beyond the small result vectors) — build
`ws` once and loop over snapshots of the same distributed grid. Requires `using MPI, PencilFFTs, PencilArrays`.
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

"""
    nufft_coarse_graining_flux(velocity_fields, scatter_coords, ℓ, filter, ms; spectral, Ls, kwargs...)

Coarse-graining energy flux `Π_ℓ(x)` at scattered (non-uniform) Cartesian points. `spectral` (required —
the two providers are peers, neither is a default) picks the NUFFT provider: `Types.FINUFFTBackend()`
(`using FINUFFT`) or `Types.NonuniformFFTsBackend()` (pure Julia — `using NonuniformFFTs`). Dispatches on
the backend type, so both coexist in one session.

`Ls` (required) is the periodic domain size per dimension. It sets the wavenumber grid `k = 2πn/L`, and
hence the filter cutoff `Ĝ(|k|, ℓ)` and the strain derivatives `i·kⱼ` — a physical input the samples
cannot supply (points in `[xₘᵢₙ, xₘᵢₙ+Lₐ)` under-span the period), so `L` is never inferred from them.
"""
function nufft_coarse_graining_flux(velocity_fields, scatter_coords, ℓ, filter, ms;
                                    spectral::SpectralBackends.AbstractNonUniformFastFourierTransformSpectralBackend,
                                    Ls::Tuple,
                                    kwargs...)
    return _nufft_coarse_graining_flux(spectral, velocity_fields, scatter_coords, ℓ, filter, ms; Ls = Ls, kwargs...)
end
_nufft_coarse_graining_flux(spectral, args...; kwargs...) = throw(ArgumentError(
    "nufft_coarse_graining_flux with $(nameof(typeof(spectral))) requires its extension: `using FINUFFT` " *
    "(Types.FINUFFTBackend) or `using NonuniformFFTs` (Types.NonuniformFFTsBackend)."))

"""
    NUFFTCoarseGrainingWorkspace(scatter_coords, ms; tol=1e-8)

Reusable resources for [`nufft_coarse_graining_flux!`](@ref): the two FINUFFT guru plans
(type-1 analysis and type-2 synthesis, points preset), the precomputed spectral-side arrays
(rescaled coordinates, `|k|`, per-axis `kⱼ` grids, filter weights `Ĝ`), and **every** working
buffer of the flux computation — the `(ms…, D)` filtered spectral velocities, the `(N, D)`
filtered scattered velocities, the `(N, D, D)` stress/strain tensor arrays, and the spectral /
scattered scratch that `finufft_exec!` writes into. A repeat call therefore allocates nothing on
the Julia side (only the tiny result struct, which wraps the reused `Π` buffer); the FINUFFT plans
avoid re-planning the FFT + re-sorting the points (dominant for small inputs).

Immutable. FINUFFT's C plans (which FINUFFT.jl registers no finalizer for) are freed by a finalizer the
extension attaches to each plan object itself — so the workspace needs no finalizer and no mutability;
NonuniformFFTs plans are pure Julia and need neither. `show` is a one-liner (never introspect a live
plan). Each array/plan field is its own type parameter, so nothing is hardcoded to `Vector`/`Array{T}`
and the core names no provider type.
"""
struct NUFFTCoarseGrainingWorkspace{P1, P2, SC, KM, KC, K1, SA, SH, SD, UM, TA, CI, CV, RV, R<:Real}
    p1::P1              # type-1 (nonuniform → uniform) plan, points set
    p2::P2              # type-2 (uniform → nonuniform) plan, points set
    scaled_coords::SC   # coordinates rescaled to the provider's periodic cell
    k_mag::KM           # |k| grid
    k_comp_grids::KC    # per-dimension kⱼ grids
    ks_1d::K1           # per-axis wavenumber vectors
    Ĝ::KM               # filter weights Ĝ(k) (recomputed per ℓ into this buffer)
    û_filt::SD          # (ms…, D) filtered spectral velocity (page c = component c)
    u_filt::UM          # (N, D) filtered velocity at the scattered points (real)
    τ::TA               # (N, D, D) SFS stress — doubles as the Π-contraction buffer
    S̄::TA               # (N, D, D) strain rate
    Π::RV               # (N,) flux (the result wraps this)
    spec::SA            # (ms…) complex spectral scratch (exec output / filtered product / gradient)
    spec_half::SH       # type-1 output scratch: the analysis plan's mode array (a real-data plan returns
                        # the non-redundant half, expanded into `spec`; a complex plan writes `spec` directly)
    scat_in::CI         # (N,) type-1 input scratch (real for a real-data analysis plan, else complex)
    scat_out::CV        # (N,) complex type-2 output scratch
    prod_r::RV          # (N,) real product / ∂uᵢ∂xⱼ scratch
    grad_j::RV          # (N,) real ∂uⱼ∂xᵢ scratch
    npoints::Int        # number of scattered points (type-1 normalization)
    tol::R
end
Base.show(io::IO, ::NUFFTCoarseGrainingWorkspace) = print(io, "NUFFTCoarseGrainingWorkspace(…)")
Base.show(io::IO, ::MIME"text/plain", w::NUFFTCoarseGrainingWorkspace) = show(io, w)

"""
    NUFFTCoarseGrainingWorkspace(scatter_coords, ms; spectral, Ls, tol=1e-8, execution=…)

Build the reusable coarse-graining workspace for `spectral`'s NUFFT provider (`Types.FINUFFTBackend()`
or `Types.NonuniformFFTsBackend()` — required, the two are peers). Dispatches on the backend type to the
corresponding extension. `Ls` (required) is the periodic domain size per dimension (sets `k = 2πn/L`; not
inferable from the scattered samples).
"""
function NUFFTCoarseGrainingWorkspace(scatter_coords::Tuple, ms::Tuple;
                                      spectral::SpectralBackends.AbstractNonUniformFastFourierTransformSpectralBackend,
                                      Ls::Tuple,
                                      kwargs...)
    return _nufft_cg_workspace(spectral, scatter_coords, ms; Ls = Ls, kwargs...)
end
_nufft_cg_workspace(spectral, args...; kwargs...) = throw(ArgumentError(
    "NUFFTCoarseGrainingWorkspace with $(nameof(typeof(spectral))) requires its extension: `using FINUFFT` " *
    "(Types.FINUFFTBackend) or `using NonuniformFFTs` (Types.NonuniformFFTsBackend)."))

function nufft_coarse_graining_flux!(args...; kwargs...)
    throw(ArgumentError("nufft_coarse_graining_flux! requires a NUFFT extension: `using FINUFFT` or `using NonuniformFFTs`."))
end

# Unified scattered-Cartesian coarse-graining entry (provider-agnostic — routes through the
# backend-dispatched one-shot above). The 4-positional (…, scatter_coords, ms) form disambiguates it
# from the uniform-grid `calculate_energy_transfer` methods.
function calculate_energy_transfer(method::Types.CoarseGrainingFluxMethod, velocity_fields::Tuple,
                                   scatter_coords::Tuple, ms::Tuple; kwargs...)
    return nufft_coarse_graining_flux(velocity_fields, scatter_coords, method.scale, method.filter, ms; kwargs...)
end

# Physical-space one-call entry for the spectral-flux / shell / mode family, covering BOTH scattered
# (non-uniform) and uniform-grid Cartesian data. One method owns the 4-positional (…, coords, ms) form;
# the transform step dispatches on the spectral backend (`_physical_energy_transfer`) so the two data
# layouts route cleanly instead of colliding: a NUFFT provider reconstructs from scattered samples, any
# other backend goes through the FlowFieldSpectra uniform-grid path. The 4-positional form disambiguates
# from the uniform coefficient methods (…, velocity_hat, ks).
const _SpectralFamilyMethod = Union{Types.SpectralFluxMethod, Types.ShellToShellTransferMethod, Types.ModeToModeTransferMethod}

function calculate_energy_transfer(method::_SpectralFamilyMethod, velocity_fields::Tuple, coords::Tuple, ms::Tuple;
                                   spectral::SpectralBackends.AbstractSpectralBackend = SpectralBackends.DirectSumSpectralBackend(),
                                   kwargs...)
    return _physical_energy_transfer(spectral, method, velocity_fields, coords, ms; kwargs...)
end

# Scattered NUFFT branch (core; `to_spectral` dispatches to the loaded provider): reconstruct
# `velocity_hat` on the scattered samples, then run the uniform diagnostic. `Ls` (required) is the
# periodic domain size; `execution` drives both the reconstruction and the diagnostic.
function _physical_energy_transfer(spectral::SpectralBackends.AbstractNonUniformFastFourierTransformSpectralBackend,
                                   method::_SpectralFamilyMethod, velocity_fields::Tuple, scatter_coords::Tuple, ms::Tuple;
                                   Ls::Tuple, tol::Real = 1e-9,
                                   execution::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.SerialBackend(),
                                   kwargs...)
    velocity_hat, ks = to_spectral(velocity_fields, scatter_coords, ms; spectral = spectral, Ls = Ls, tol = tol, execution = execution)
    return calculate_energy_transfer(method, velocity_hat, ks; execution = execution, kwargs...)
end

# Uniform-grid branch: the FlowFieldSpectra extension adds the concrete (more specific) method (physical
# → spectral on a structured grid). This varargs catch-all is the fallback when it is not loaded — a
# non-NUFFT backend then has no physical transform here. (A catch-all, not the FFS signature: an
# extension may not overwrite a same-signature parent method during precompilation.)
_physical_energy_transfer(spectral::SpectralBackends.AbstractSpectralBackend, args...; kwargs...) = throw(ArgumentError(
    "uniform-grid physical → spectral transfer requires `using FlowFieldSpectra`; for scattered (non-uniform) " *
    "Cartesian data pass a NUFFT backend (spectral = Types.FINUFFTBackend() / Types.NonuniformFFTsBackend()) with `Ls`."))

"""
    NUFFTToSpectralWorkspace{P, SC, KS, UH, CV, SP, R}

Reusable resources for the in-place scattered → uniform reconstruction [`to_spectral!`](@ref): the
provider's type-1 plan (points preset), the uniform wavenumber grid `ks`, and every working buffer (the
`(ms…, D)` coefficient array `û` plus complex type-1 scratch). A repeat `to_spectral!` re-plans nothing
and allocates nothing on the Julia side. Immutable — each field its own type parameter, nothing hardcoded
to `Array{T}` and the core names no provider type. FINUFFT's C plan (which FINUFFT.jl registers no
finalizer for) is freed by a finalizer the extension attaches to the plan object itself, so the workspace
needs no finalizer and no mutability; NonuniformFFTs plans are pure Julia and need neither.
"""
struct NUFFTToSpectralWorkspace{P, SC, KS, UH, CV, SP, R<:Real}
    plan::P              # provider type-1 (nonuniform → uniform) plan, points set
    scaled_coords::SC    # coordinates rescaled to the provider's periodic cell
    ks::KS               # per-axis uniform wavenumber vectors (returned with û)
    û::UH                # (ms…, D) coefficient buffer (the result aliases this)
    scat::CV             # (N,) complex type-1 input scratch
    spec::SP             # (ms…) complex type-1 output scratch
    npoints::Int         # number of scattered points (type-1 normalization)
    invN::R
end
Base.show(io::IO, ::NUFFTToSpectralWorkspace) = print(io, "NUFFTToSpectralWorkspace(…)")
Base.show(io::IO, ::MIME"text/plain", w::NUFFTToSpectralWorkspace) = show(io, w)

"""
    NUFFTToSpectralWorkspace(scatter_coords, ms; spectral, Ls, tol=1e-9)

Build the reusable scattered → uniform workspace for `spectral`'s NUFFT provider (peers
`Types.FINUFFTBackend()` / `Types.NonuniformFFTsBackend()`, required). `Ls` (required) is the periodic
domain size per dimension (`k = 2πn/L`; not inferable from the samples). Dispatches on the backend type to
the corresponding extension.
"""
function NUFFTToSpectralWorkspace(scatter_coords::Tuple, ms::Tuple;
                                  spectral::SpectralBackends.AbstractNonUniformFastFourierTransformSpectralBackend,
                                  Ls::Tuple,
                                  kwargs...)
    return _to_spectral_workspace(spectral, scatter_coords, ms; Ls = Ls, kwargs...)
end
_to_spectral_workspace(spectral, args...; kwargs...) = throw(ArgumentError(
    "NUFFTToSpectralWorkspace with $(nameof(typeof(spectral))) requires its extension: `using FINUFFT` " *
    "(Types.FINUFFTBackend) or `using NonuniformFFTs` (Types.NonuniformFFTsBackend)."))

# Host/device allocation strategy for the NonuniformFFTs `to_spectral` workspace, dispatched on the
# execution backend. Host defaults are generic (no NonuniformFFTs / KernelAbstractions dependency); the
# KernelAbstractions extension overrides `_nufft_new` / `_nufft_to_device` for a `GPUBackend` so the plan
# is built on its KA backend and the point/data buffers are device-resident.
_nufft_plan_backend_kw(::ComputationalBackends.AbstractExecutionBackend) = NamedTuple()
_nufft_plan_backend_kw(gpu::ComputationalBackends.AbstractGPUBackend) = (; backend = gpu.backend)
_nufft_new(::ComputationalBackends.AbstractExecutionBackend, ::Type{CT}, dims::Vararg{Int}) where {CT} = Array{CT}(undef, dims...)
_nufft_to_device(::ComputationalBackends.AbstractExecutionBackend, x) = x

# FINUFFT `to_spectral` plan+buffer build, dispatched on the execution backend. The host method (FINUFFT
# extension) and the NVIDIA-GPU cuFINUFFT method (FINUFFT + CUDA extension) both need FINUFFT symbols, so
# only the generic function is owned here; both extensions add their methods to it.
function _finufft_ts_build end

"""
    to_spectral!(ws::NUFFTToSpectralWorkspace, velocity_fields) -> (velocity_hat, ks)

In-place scattered → uniform reconstruction reusing `ws` (plan + buffers): `û = type1(u)/N` in FFTW mode
order, returned with the uniform wavenumber grid `ks`. A repeat call allocates nothing on the Julia side
— the returned `velocity_hat` aliases `ws.û`, overwritten on the next call. Dispatches on the plan type.
"""
function to_spectral!(args...; kwargs...)
    throw(ArgumentError("to_spectral! requires a NUFFT extension: `using FINUFFT` or `using NonuniformFFTs`."))
end

export nufft_coarse_graining_flux, nufft_coarse_graining_flux!, NUFFTCoarseGrainingWorkspace
export NUFFTToSpectralWorkspace, to_spectral!

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

Physical-space Cartesian data uses the 4-positional form
`calculate_energy_transfer(method, velocity_fields, coords, ms; spectral, …)`, which routes on the
spectral backend: a NUFFT provider (`Types.FINUFFTBackend()` / `Types.NonuniformFFTsBackend()`, with `Ls`)
reconstructs from **scattered** samples, while any other backend transforms **uniform-grid** data through
FlowFieldSpectra (`using FlowFieldSpectra`; `coords` are the per-axis grid vectors). Coarse-graining has
the same 4-positional scattered route. Spherical data uses
`calculate_energy_transfer(SphericalTransferMethod(), ζ, (θ, φ); lmax, …)` (scattered, NUFSHT) or a
regular colatitude–longitude grid (FastSphericalHarmonics).

# Returns
Method-specific result container: `SpectralFluxResult`, `CoarseGrainingFluxResult`,
or `ShellToShellResult`.

# Examples
```julia
using FlowInvariantTransfer, FFTW

# Spectral flux on a 32×32 periodic domain
N = 32; L = 2π
x = range(0.0, L; length=N+1)[1:N]
y = range(0.0, L; length=N+1)[1:N]
u = [cos(x) for x in x, y in y]
v = [sin(y) for x in x, y in y]
û = cat(FFTW.fft(u), FFTW.fft(v); dims=3) ./ N^2  # (N,N,2)
ks = FlowInvariantTransfer.Utils.wavenumber_grid((N,N), (L,L))

result = calculate_energy_transfer(SpectralFluxMethod(LinearBinning(2π/L)), û, ks)
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
    to_spectral(velocity_fields::Tuple, coords_vecs::Tuple; spectral=SpectralBackends.FFTSpectralBackend()) -> (velocity_hat, ks)

Forward-transform physical-space velocity components sampled on a **uniform, periodic, tensor-product
Cartesian grid** into the Fourier-coefficient input `(velocity_hat, ks)` consumed by every Cartesian
flux diagnostic ([`calculate_spectral_flux`](@ref), [`calculate_shell_to_shell_transfer`](@ref),
[`calculate_mode_to_mode_transfer`](@ref), [`calculate_band_to_band_transfer`](@ref),
[`calculate_partial_fluxes`](@ref), [`calculate_compressible_flux`](@ref)). This is the physical-space
entry point for gridded data: the diagnostics operate on coefficients, so a real field is transformed
here once with the package's `û = fft(u)/Nᵈ` normalization (so `E(k) = ½|û|²`) and the FFTW `fftfreq`
wavenumber convention — you do not build `û`/`ks` by hand.

`coords_vecs` are the **1D coordinate vectors** `(x, y[, z])` of the grid (one vector per axis, of
length `nₐ`), *not* per-sample coordinates. Scattered (non-uniform) Cartesian data is handled by the
NUFFT path (`spectral = Types.FINUFFTBackend()` or `Types.NonuniformFFTsBackend()`), and spherical data
by [`calculate_spherical_transfer`](@ref) — see [`spectral_geometry`](@ref).

# Keyword Arguments
- `spectral::SpectralBackends.AbstractSpectralBackend = SpectralBackends.FFTSpectralBackend()`: the analysis transform. `SpectralBackends.FFTSpectralBackend()` needs
  `using FFTW` (cuFFT is used automatically for device-array inputs); `SpectralBackends.DirectSumSpectralBackend()` is the
  dependency-free `O(N²ᴰ)` reference (tiny grids only).

# Returns
`(velocity_hat, ks)` — `velocity_hat` is `(nₐ..., D)` complex in the input array's backend (a device
field yields a device coefficient array); `ks` a tuple of `D` wavenumber vectors.

# Example
```julia
using FlowInvariantTransfer, FFTW
û, ks = to_spectral((u, v), (x, y))
Π = calculate_spectral_flux(û, ks; spectral = SpectralBackends.FFTSpectralBackend())
```

Pass a scalar (density / pressure / passive scalar) as a 1-tuple: `ρ̂, _ = to_spectral((ρ,), coords)`.
"""
function to_spectral(velocity_fields::Tuple, coords_vecs::Tuple;
                     spectral::SpectralBackends.AbstractSpectralBackend = SpectralBackends.FFTSpectralBackend())
    D = length(velocity_fields)
    D >= 1 || throw(ArgumentError("to_spectral needs ≥1 physical field component."))
    spectral isa Union{SpectralBackends.DirectSumSpectralBackend, SpectralBackends.FFTSpectralBackend} || throw(ArgumentError(
        "to_spectral transforms uniform-grid Cartesian data (spectral = SpectralBackends.FFTSpectralBackend() or the SpectralBackends.DirectSumSpectralBackend " *
        "reference). For scattered Cartesian data use the NUFFT physical entry (Types.FINUFFTBackend / Types.NonuniformFFTsBackend); for spherical data use " *
        "calculate_spherical_transfer."))
    ns = Utils.validate_velocity_input(velocity_fields, length(coords_vecs))
    Utils.validate_uniform_grid(coords_vecs)
    Ls = Utils.domain_size_from_coords(coords_vecs)
    ks = Utils.wavenumber_grid(ns, Ls)
    CT = complex(float(real(eltype(velocity_fields[1]))))
    field_phys = similar(velocity_fields[1], CT, ns..., D)
    colons = ntuple(_ -> Colon(), length(ns))
    for c in 1:D
        view(field_phys, colons..., c) .= complex.(velocity_fields[c])
    end
    # Reuse the shared analysis transform context (û = fft(u)/Nᵈ): DirectSum in core, FFT in the FFTW
    # ext, cuFFT for device arrays. Errors clearly (no silent downgrade) if SpectralBackends.FFTSpectralBackend is requested
    # without FFTW loaded.
    tf = Compressible._resolve_tf(spectral, field_phys, ks, ns)
    return (tf.dft(field_phys), ks)
end

"""
    to_spectral(velocity_fields::Tuple, scatter_coords::Tuple, ms::Tuple; spectral, Ls, tol=1e-9) -> (velocity_hat, ks)

Scattered-Cartesian physical-space entry: reconstruct the Fourier coefficients `velocity_hat` on a
uniform `ms = (m₁,…,m_D)` grid from velocity components sampled at **scattered** (non-uniform) points,
via a NUFFT type-1 transform, and return them with the matching uniform wavenumber grid `ks`. The
result feeds the ordinary Cartesian flux diagnostics unchanged — the scattered→uniform step lives
entirely here, so the whole flux family works on scattered data through the uniform path.

`scatter_coords = (x, y[, z])` are per-sample coordinate vectors (each length `N`, the number of
samples), *not* grid axes; `ms` is the target uniform mode count per dimension. `spectral` (required —
the two NUFFT providers are peers) picks `Types.FINUFFTBackend()` (`using FINUFFT`) or
`Types.NonuniformFFTsBackend()` (`using NonuniformFFTs`). For uniform-grid data use the 2-argument form
with `SpectralBackends.FFTSpectralBackend()`.

The reconstruction is the density-normalized adjoint (`û = type1(u)/N`, exact for samples on the
uniform grid; the package's established scattered-Cartesian convention). `tol` sets the NUFFT
tolerance. `Ls` (required) is the periodic domain size per dimension — it fixes the wavenumbers
`k = 2πn/L`. The samples live in `[xₘᵢₙ, xₘᵢₙ+Lₐ)` and under-span the period, so `L` cannot be inferred
from them; samples on the uniform `L`-grid give `û = fft(u)/Nᵈ` exactly. `execution` selects host
(default) vs device-resident: a `ComputationalBackends.GPUBackend(dev)` (NonuniformFFTs provider, `dev`
a KernelAbstractions backend) builds the plan and buffers on-device so the scattered → `velocity_hat`
step runs on the device.
"""
function to_spectral(velocity_fields::Tuple, scatter_coords::Tuple, ms::Tuple;
                     spectral::SpectralBackends.AbstractSpectralBackend, tol::Real = 1e-9,
                     Ls::Tuple,
                     execution::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.SerialBackend())
    spectral isa SpectralBackends.AbstractNonUniformFastFourierTransformSpectralBackend || throw(ArgumentError(
        "the 3-argument to_spectral(fields, scatter_coords, ms; …) is the scattered-Cartesian NUFFT " *
        "entry — pass spectral = Types.FINUFFTBackend() or Types.NonuniformFFTsBackend(). For uniform-grid " *
        "data use to_spectral(fields, coords_vecs; spectral = SpectralBackends.FFTSpectralBackend())."))
    ws = NUFFTToSpectralWorkspace(scatter_coords, ms; spectral = spectral,
                                  ncomponents = length(velocity_fields), tol = tol, Ls = Ls, execution = execution)
    return to_spectral!(ws, velocity_fields)
end

# ---------------------------------------------------------------------------
# Precompilation workload (small grid to reduce TTFX)
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

    PrecompileTools.@compile_workload begin
        _ = calculate_spectral_flux(û, ks; binning=Types.LinearBinning(2π/L), dealiasing=Types.NoDealiasing())
        _ = calculate_shell_to_shell_transfer(û, ks;
                binning=Types.LinearBinning(2π/L), dealiasing=Types.NoDealiasing(), verify_antisymmetry=false)
        _ = Utils.wavenumber_grid((N,N), (L,L))
        _ = Utils.dealiasing_mask((N,N))
    end
end

end # module FlowInvariantTransfer