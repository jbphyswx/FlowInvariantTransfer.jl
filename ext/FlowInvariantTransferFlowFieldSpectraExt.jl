module FlowInvariantTransferFlowFieldSpectraExt

using FlowFieldSpectra: FlowFieldSpectra as FFS
using FlowInvariantTransfer: FlowInvariantTransfer as FIT
using FlowTransformBindings: FlowTransformBindings as FTB
using SpectralBackends: SpectralBackends
using ComputationalBackends: ComputationalBackends

# Uniform-grid branch of the physical one-call `calculate_energy_transfer(method, fields, coords, ms;
# transform, …)` for the spectral-flux/shell/mode family (the non-NUFFT `_physical_energy_transfer`
# method; the NUFFT branch lives in core). Velocity component arrays `(u, v[, w])` on per-axis coordinate
# vectors `coords_vecs = (xaxis, yaxis[, zaxis])` are transformed to spectral coefficients via
# `FlowFieldSpectra.calculate_spectrum` and fed to the diagnostic.
#
const _FFSMethod = Union{FIT.Types.SpectralFluxMethod, FIT.Types.ShellToShellTransferMethod,
                         FIT.Types.ModeToModeTransferMethod}

# One `Periodic()` tag per direction of `axes`.
_all_periodic(axes::Tuple) = map(_ -> FFS.FlowGeometries.Grids.Periodic(), axes)

# `FFS.calculate_spectrum` is declared on `FlowGeometries.Grids.AbstractGrid` and selects its transform
# from the grid, so a grid passed here reaches whichever one its geometry and sampling admit. It returns
# `û = fft(u)/∏ms` on FlowTransformBindings' axes, the packed half for a real field, which is the input
# every Fourier-coefficient diagnostic takes.
function FIT._physical_energy_transfer(
    spectral::SpectralBackends.AbstractSpectralBackend,
    method::_FFSMethod,
    velocity_fields::Tuple,
    grid::FFS.FlowGeometries.Grids.AbstractGrid,
    ms::Tuple;
    execution::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend(),
    kwargs...
)
    coeffs, ks = FFS.calculate_spectrum(grid, velocity_fields, ms; transform = spectral, execution = execution)
    return FIT.calculate_energy_transfer(method, coeffs, ks; execution = execution, kwargs...)
end

# Coordinate-vector convenience: an all-periodic Cartesian grid over those axes. Pass each axis as a
# uniform range and its period is inferred (`n·Δ`); a stretched or plain-vector axis needs
# `domain_size`, its per-direction period.
function FIT._physical_energy_transfer(
    spectral::SpectralBackends.AbstractSpectralBackend,
    method::_FFSMethod,
    velocity_fields::Tuple,
    coords_vecs::Tuple,
    ms::Tuple;
    domain_size = nothing,
    kwargs...
)
    FT  = float(eltype(coords_vecs[1]))
    geom = FFS.FlowGeometries.Geometry.CartesianGeometry{FT}()
    grid = FFS.FlowGeometries.Grids.StructuredGrid(geom, coords_vecs...;
                                                   topology = _all_periodic(coords_vecs), period = domain_size)
    return FIT._physical_energy_transfer(spectral, method, velocity_fields, grid, ms; kwargs...)
end

# ---------------------------------------------------------------------------
# Physical → spectral with the domain given as a grid.
#
# `FFS.plan_spectrum` selects the transform from the grid — the tensor-product FFT on a uniform
# Cartesian one, the FFT/NUFFT composite where some directions are stretched, the NUFFT on a node set —
# so a grid reaches whichever its layout admits. The plan is the reusable half and `calculate_spectrum!`
# writes into the workspace's own coefficient array.
# ---------------------------------------------------------------------------

"""
    _ffs_workspace(proto, grid, ns, ET, D, spectral, execution) -> ToSpectralWorkspace

The analysis and synthesis plans for one `(grid, ms, batch, element type)`, with the buffers each side
sizes and types itself. `ET` is the field's element type, and it fixes the coefficient layout on both
plans: the packed half for a real field, the full cube for a complex one. The wavenumbers are the
analysis plan's: along an axis where `±N/2` are distinct modes the box holds both, an odd length.
"""
function _ffs_workspace(proto, grid, ns::Tuple, ::Type{ET}, D::Int, spectral, execution) where {ET}
    execution = FIT.Types.resolve_execution(execution, proto)
    ap = FFS.plan_spectrum(grid, ET, ns; transform = spectral, execution = execution, batch = (D,))
    sp = FFS.plan_synthesis(grid, ET, ns; transform = spectral, execution = execution, batch = (D,))
    ks = FFS.wavenumbers(ap)
    field_phys = similar(proto, FFS.field_type(sp), FFS.field_size(sp)...)
    û = similar(proto, FFS.coefficient_type(ap), FFS.coefficient_size(ap)...)
    tf = (; dft!  = (out, fld) -> FFS.calculate_spectrum!(out, ap, fld),
            idft! = (out, co)  -> FFS.synthesize!(out, sp, co))
    return FIT.ToSpectralWorkspace(tf, field_phys, û, ks)
end

# The domain this entry is defined on: Cartesian, and periodic in every direction.
function _check_fourier_grid(grid)
    # `to_spectral` returns Fourier coefficients on a Cartesian mode grid. A spherical grid has an
    # analysis too, in spherical harmonics, and it is a different object with different wavenumbers.
    FFS.FlowGeometries.Grids.grid_geometry(grid) isa FFS.FlowGeometries.Geometry.AbstractCartesianGeometry ||
        throw(ArgumentError(
            "`to_spectral` is the Cartesian Fourier analysis; this grid is on " *
            "$(nameof(typeof(FFS.FlowGeometries.Grids.grid_geometry(grid)))). Spherical data goes to " *
            "`calculate_energy_transfer(SphericalTransferMethod(), field, grid)`."))
    Ls = ntuple(d -> FFS.FlowGeometries.Grids.period(grid, d), Val(ndims(grid)))
    all(L -> isfinite(L) && L > 0, Ls) || throw(ArgumentError(
        "physical → spectral expands in a periodic basis, and direction " *
        "$(findfirst(L -> !(isfinite(L) && L > 0), Ls)) of this grid carries no period. Build it with " *
        "`topology = FlowGeometries.Grids.Periodic()` and `period` in every direction."))
    return size(grid)
end

"""
    ToSpectralWorkspace(velocity_fields, grid::FlowGeometries.Grids.AbstractGrid; transform, execution)

Analysis workspace with the domain given as a `FlowGeometries` grid. The transform is
`FlowFieldSpectra.plan_spectrum` on that grid, so the grid's own geometry, spacing and topology select
it; a real field takes the half layout.
"""
function FIT.ToSpectralWorkspace(
    velocity_fields::Tuple,
    grid::FFS.FlowGeometries.Grids.AbstractGrid;
    transform::SpectralBackends.AbstractSpectralBackend = SpectralBackends.AutoSpectralBackend(),
    execution::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend(),
)
    D = length(velocity_fields)
    D >= 1 || throw(ArgumentError("to_spectral needs ≥1 physical field component."))
    # This entry analyses gridded data onto a Cartesian mode grid. The spherical transforms and the
    # scattered-point providers answer a different question and are reached by their own entries.
    transform isa Union{SpectralBackends.AbstractNUFFTSpectralBackend,
                        SpectralBackends.FSHTSpectralBackend,
                        SpectralBackends.NUFSHTSpectralBackend} && throw(ArgumentError(
        "`to_spectral` analyses gridded data; got transform = $(nameof(typeof(transform))). Scattered " *
        "Cartesian points take `to_spectral(fields, points, ms; Ls)` with a NUFFT provider, and " *
        "spherical data `calculate_energy_transfer(SphericalTransferMethod(), field, grid)`."))
    ns = _check_fourier_grid(grid)
    ET = float(promote_type(map(eltype, velocity_fields)...))
    return _ffs_workspace(first(velocity_fields), grid, ns, ET, D, transform, execution)
end

"""
    to_spectral(velocity_fields, grid::FlowGeometries.Grids.AbstractGrid; transform, execution)

Physical → spectral on a `FlowGeometries` grid; see [`to_spectral`](@ref).
"""
FIT.to_spectral(velocity_fields::Tuple, grid::FFS.FlowGeometries.Grids.AbstractGrid; kwargs...) =
    FIT.to_spectral!(FIT.ToSpectralWorkspace(velocity_fields, grid; kwargs...), velocity_fields)

# The wavenumber axes fix the periodic box completely: each carries the full grid length `n` it indexes,
# and its spacing is `Δk = 2π/L`. The grid rebuilt from them is the one the forward transform ran on.
function FIT._to_spectral_workspace_on_coeffs(
    velocity_hat::AbstractArray, ks::Tuple;
    transform::SpectralBackends.AbstractSpectralBackend = SpectralBackends.AutoSpectralBackend(),
    execution::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend(),
)
    nd = length(ks)
    ns = FTB.full_size(ks)
    ms = FTB.spectral_size(ks)
    lead = ntuple(d -> size(velocity_hat, d), Val(nd))
    lead == ms || throw(DimensionMismatch("coefficient size $lead does not match the wavenumber grid $ms."))
    D = size(velocity_hat, nd + 1)
    FT = real(eltype(velocity_hat))
    Ls = map(a -> FT(2π) / FT(length(a) > 1 ? abs(a[2] - a[1]) : one(FT)), ks)
    geom = FFS.FlowGeometries.Geometry.CartesianGeometry{FT}()
    axes_ = map((L, n) -> range(zero(FT); step = L / n, length = n), Ls, ns)
    grid = FFS.FlowGeometries.Grids.StructuredGrid(geom, axes_...; topology = _all_periodic(ks), period = Ls)
    # A half layout is the transform of a real field, so the plans are built for one.
    ET = _field_type(first(ks), FT)
    return _ffs_workspace(velocity_hat, grid, ns, ET, D, transform, execution)
end

# Per-axis coordinate vectors describe the uniform periodic box `L = N·Δ` of `domain_size_from_coords`;
# the grid carries that period explicitly.
function FIT._to_spectral_workspace_on_axes(velocity_fields::Tuple, coords_vecs::Tuple; kwargs...)
    FIT.Utils.validate_velocity_input(velocity_fields, length(coords_vecs))
    FIT.Utils.validate_uniform_grid(coords_vecs)
    Ls = FIT.Utils.domain_size_from_coords(coords_vecs)
    FT = float(real(eltype(first(velocity_fields))))
    geom = FFS.FlowGeometries.Geometry.CartesianGeometry{FT}()
    grid = FFS.FlowGeometries.Grids.StructuredGrid(geom, coords_vecs...; topology = _all_periodic(coords_vecs), period = Ls)
    return FIT.ToSpectralWorkspace(velocity_fields, grid; kwargs...)
end

_field_type(::FTB.RFFTAxis, ::Type{T}) where {T} = T
_field_type(::AbstractVector, ::Type{T}) where {T} = Complex{T}


# ---------------------------------------------------------------------------
# Physical → decomposed → physical, in one call.
#
# The helical and toroidal/poloidal splits are defined per mode, so on physical input they are an
# analysis, a per-mode projection, and a synthesis of each component. Every component of a real field is
# itself Hermitian — under `k ↦ −k` the Craya–Herring frame obeys `e1 ↦ −e1`, `e2 ↦ e2`, so
# `u_±(−k) = conj(u_±(k))` — so each synthesises back to a real field on the same layout the analysis
# produced.
#
# The Helmholtz family has its own physical method in the HelmholtzDecomposition extension, which solves
# the Poisson problem the grid poses.
# ---------------------------------------------------------------------------

const _SpectralOnlyDecomp = Union{FIT.Types.HelicalDecomposition, FIT.Types.ToroidalPoloidalDecomposition}

"""
    decompose_field(decomp, velocity_fields::Tuple, grid; transform, execution)

Physical components of a spectral decomposition, on a `FlowGeometries` grid: the fields are analysed,
split per mode, and each component synthesised back. Returns the decomposition's own names
(`(; positive, negative)` for helical, `(; toroidal, poloidal)` for toroidal/poloidal), each a tuple of
physical component arrays shaped like the input.

The components sum to the field over the modes the frame is defined on. The Nyquist mode of an even
axis is its own image under `k ↦ −k`, so it fixes no orientation for the Craya–Herring frame and is
excluded from that identity; a field band-limited below it — which the `2/3` rule enforces wherever
these components feed a transfer — reconstructs to round-off.
"""
function FIT.Decomposition._decompose_field_physical(
    decomp::_SpectralOnlyDecomp,
    velocity_fields::Tuple,
    grid::FFS.FlowGeometries.Grids.AbstractGrid;
    kwargs...,
)
    ws = FIT.ToSpectralWorkspace(velocity_fields, grid; kwargs...)
    û, ks = FIT.to_spectral!(ws, velocity_fields)
    comps = FIT.Decomposition.decompose_field(decomp, û, ks)
    return map(c -> FIT.from_spectral(c, ks), comps)
end

end # module
