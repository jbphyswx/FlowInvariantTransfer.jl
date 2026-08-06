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
# ---------------------------------------------------------------------------
# Public surface. Re-import the public names so `FlowInvariantTransfer.foo` resolves, but `export`
# ONLY the entry-point functions — a bare `using FlowInvariantTransfer` brings in the verbs
# (`calculate_*`, `triadic_orthogonal_decomposition`, `to_spectral`), never the method/binning/
# geometry/dealiasing/result TYPES, which stay reachable qualified (`FlowInvariantTransfer.LinearBinning`).
# ---------------------------------------------------------------------------

using .SpectralFlux: SpectralFlux, calculate_spectral_flux, calculate_spectral_flux!, calculate_scalar_flux, calculate_scalar_flux!, calculate_partial_fluxes, calculate_partial_fluxes!, calculate_helical_partial_fluxes, calculate_helical_partial_fluxes!
using .Compressible: Compressible, calculate_compressible_flux, calculate_compressible_flux!
using .Spherical: Spherical, calculate_spherical_transfer, calculate_spherical_transfer!, calculate_divergent_spherical_transfer, calculate_divergent_spherical_transfer!
using .CoarseGrainingFlux: CoarseGrainingFlux, calculate_coarse_graining_flux, calculate_coarse_graining_flux!
using .ShellToShellTransfer: ShellToShellTransfer, calculate_shell_to_shell_transfer, calculate_shell_to_shell_transfer!, calculate_scalar_shell_to_shell_transfer, calculate_scalar_shell_to_shell_transfer!
using .BandTransfer: BandTransfer, calculate_band_to_band_transfer, calculate_band_to_band_transfer!
using .ModeToModeTransfer: ModeToModeTransfer, calculate_mode_to_mode_transfer, calculate_mode_to_mode_transfer!, calculate_scalar_mode_to_mode_transfer, calculate_scalar_mode_to_mode_transfer!
using .TriadicOrthogonalDecomposition: TriadicOrthogonalDecomposition, triadic_orthogonal_decomposition, triadic_orthogonal_decomposition!

export calculate_spectral_flux, calculate_spectral_flux!, calculate_scalar_flux, calculate_scalar_flux!, calculate_partial_fluxes, calculate_partial_fluxes!, calculate_helical_partial_fluxes, calculate_helical_partial_fluxes!
export calculate_compressible_flux, calculate_compressible_flux!
export calculate_spherical_transfer, calculate_spherical_transfer!, calculate_divergent_spherical_transfer, calculate_divergent_spherical_transfer!
export calculate_coarse_graining_flux, calculate_coarse_graining_flux!
export calculate_shell_to_shell_transfer, calculate_shell_to_shell_transfer!, calculate_scalar_shell_to_shell_transfer, calculate_scalar_shell_to_shell_transfer!
export calculate_band_to_band_transfer, calculate_band_to_band_transfer!
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
    pencil_spectral_flux(u_phys, ks; comm=MPI.COMM_WORLD, binning, dealiasing, invariant) -> (centers, transfer_spectrum, flux)

Distributed spectral transfer/flux for a single grid split across MPI ranks along the **pencil
axis**: `u_phys` is the physical-space velocity held as a `PencilArray` (each rank owns a pencil
of the global grid). The pseudospectral nonlinear term is evaluated with a transpose-based
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
    nufft_coarse_graining_flux(velocity_fields, scatter_coords, ℓ, filter, ms; kwargs...)

Coarse-graining energy flux `Π_ℓ(x)` at scattered (non-uniform) Cartesian points via FINUFFT.
Requires `using FINUFFT` (the extension supplies the method).
"""
function nufft_coarse_graining_flux(args...; kwargs...)
    throw(ArgumentError("nufft_coarse_graining_flux requires FINUFFT. Run `using FINUFFT`."))
end

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

`mutable` is required ONLY so a `finalizer` can free the two plans (C resources FINUFFT never
finalizes itself — holding them without this would leak the FFTW plan + spreading grid); every
field is `const`. `show` is a one-liner (never introspect a live plan). Requires `using FINUFFT`;
each array/plan field is its own type parameter, so nothing is hardcoded to `Vector`/`Array{T}` and
the core names no FINUFFT type.
"""
mutable struct NUFFTCoarseGrainingWorkspace{P1, P2, SC, KM, KC, K1, SA, SD, UM, TA, CV, RV, R<:Real}
    const p1::P1              # FINUFFT type-1 (nonuniform → uniform) plan, points set
    const p2::P2              # FINUFFT type-2 (uniform → nonuniform) plan, points set
    const scaled_coords::SC   # coordinates rescaled to [-π, π)
    const k_mag::KM           # |k| grid
    const k_comp_grids::KC    # per-dimension kⱼ grids
    const ks_1d::K1           # per-axis wavenumber vectors
    const Ĝ::KM               # filter weights Ĝ(k) (recomputed per ℓ into this buffer)
    const û_filt::SD          # (ms…, D) filtered spectral velocity (page c = component c)
    const u_filt::UM          # (N, D) filtered velocity at the scattered points (real)
    const τ::TA               # (N, D, D) SFS stress — doubles as the Π-contraction buffer
    const S̄::TA               # (N, D, D) strain rate
    const Π::RV               # (N,) flux (the result wraps this)
    const spec::SA            # (ms…) complex spectral scratch (exec output / filtered product / gradient)
    const scat_in::CV         # (N,) complex type-1 input scratch
    const scat_out::CV        # (N,) complex type-2 output scratch
    const prod_r::RV          # (N,) real product / ∂uᵢ∂xⱼ scratch
    const grad_j::RV          # (N,) real ∂uⱼ∂xᵢ scratch
    const npoints::Int        # number of scattered points (type-1 normalization)
    const tol::R
end
Base.show(io::IO, ::NUFFTCoarseGrainingWorkspace) = print(io, "NUFFTCoarseGrainingWorkspace(…)")
Base.show(io::IO, ::MIME"text/plain", w::NUFFTCoarseGrainingWorkspace) = show(io, w)

function NUFFTCoarseGrainingWorkspace(args...; kwargs...)
    throw(ArgumentError("NUFFTCoarseGrainingWorkspace requires FINUFFT. Run `using FINUFFT`."))
end

function nufft_coarse_graining_flux!(args...; kwargs...)
    throw(ArgumentError("nufft_coarse_graining_flux! requires FINUFFT. Run `using FINUFFT`."))
end

export nufft_coarse_graining_flux, nufft_coarse_graining_flux!, NUFFTCoarseGrainingWorkspace

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
NUFFT path (`spectral = SpectralBackends.NUFFTSpectralBackend()`), and spherical data by
[`calculate_spherical_transfer`](@ref) — see [`spectral_geometry`](@ref).

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
        "reference). For scattered Cartesian data use the SpectralBackends.NUFFTSpectralBackend physical entry; for spherical data use " *
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
    to_spectral(velocity_fields::Tuple, scatter_coords::Tuple, ms::Tuple;
                spectral=SpectralBackends.NUFFTSpectralBackend(), tol=1e-9) -> (velocity_hat, ks)

Scattered-Cartesian physical-space entry: reconstruct the Fourier coefficients `velocity_hat` on a
uniform `ms = (m₁,…,m_D)` grid from velocity components sampled at **scattered** (non-uniform) points,
via a FINUFFT type-1 transform, and return them with the matching uniform wavenumber grid `ks`. The
result feeds the ordinary Cartesian flux diagnostics unchanged — the scattered→uniform step lives
entirely here, so the whole flux family works on scattered data through the uniform path.

`scatter_coords = (x, y[, z])` are per-sample coordinate vectors (each length `N`, the number of
samples), *not* grid axes; `ms` is the target uniform mode count per dimension. Requires
`using FINUFFT`. For uniform-grid data use the 2-argument form with `SpectralBackends.FFTSpectralBackend()`.

The reconstruction is the density-normalized adjoint (`û = type1(u)/N`, exact for samples on the
uniform grid; the package's established scattered-Cartesian convention). `tol` sets the FINUFFT
tolerance. `Ls` gives the periodic domain size per dimension; pass it for correct wavenumbers on a
periodic domain (the samples live in `[xₘᵢₙ, xₘᵢₙ+Lₐ)`). If omitted it is inferred from the sample
span `maxₐ − minₐ`, which for scattered samples that do not cover the full period is only approximate.
"""
function to_spectral(velocity_fields::Tuple, scatter_coords::Tuple, ms::Tuple;
                     spectral::SpectralBackends.AbstractSpectralBackend = SpectralBackends.NUFFTSpectralBackend(), tol::Real = 1e-9,
                     Ls::Union{Nothing, Tuple} = nothing)
    spectral isa SpectralBackends.NUFFTSpectralBackend || throw(ArgumentError(
        "the 3-argument to_spectral(fields, scatter_coords, ms; …) is the scattered-Cartesian NUFFT " *
        "entry — use spectral = SpectralBackends.NUFFTSpectralBackend(). For uniform-grid data use to_spectral(fields, coords_vecs; " *
        "spectral = SpectralBackends.FFTSpectralBackend())."))
    return _to_spectral_nufft(velocity_fields, scatter_coords, ms, tol, Ls)
end

# Overridden by the FINUFFT extension (requires `using FINUFFT`).
_to_spectral_nufft(args...; kwargs...) = throw(ArgumentError(
    "Scattered-Cartesian to_spectral (SpectralBackends.NUFFTSpectralBackend) requires FINUFFT. Run `using FINUFFT` to load the extension."))

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
        _ = calculate_spectral_flux(û, ks; binning=Types.LinearBinning(2π/L), dealiasing=Types.NoDealiasing())
        _ = calculate_shell_to_shell_transfer(û, ks;
                binning=Types.LinearBinning(2π/L), dealiasing=Types.NoDealiasing(), verify_antisymmetry=false)
        _ = Utils.wavenumber_grid((N,N), (L,L))
        _ = Utils.dealiasing_mask((N,N))
    end
end

end # module FlowInvariantTransfer