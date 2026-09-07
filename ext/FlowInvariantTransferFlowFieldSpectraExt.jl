module FlowInvariantTransferFlowFieldSpectraExt

using FlowFieldSpectra: FlowFieldSpectra as FFS
using FlowInvariantTransfer: FlowInvariantTransfer as FIT
using SpectralBackends: SpectralBackends
using ComputationalBackends: ComputationalBackends

# Uniform-grid branch of the physical one-call `calculate_energy_transfer(method, fields, coords, ms;
# spectral, …)` for the spectral-flux/shell/mode family (the non-NUFFT `_physical_energy_transfer`
# method; the NUFFT branch lives in core). Velocity component arrays `(u, v[, w])` on per-axis coordinate
# vectors `coords_vecs = (xaxis, yaxis[, zaxis])` are transformed to spectral coefficients via
# `FlowFieldSpectra.calculate_spectrum` and fed to the diagnostic.
#
const _FFSMethod = Union{FIT.Types.SpectralFluxMethod, FIT.Types.ShellToShellTransferMethod,
                         FIT.Types.ModeToModeTransferMethod}

# `FlowFieldSpectra.calculate_spectrum` returns `û = fft(u)/∏ms` on the `fftfreq` axes, and the
# rfft-packed half `(m₁÷2+1, m₂…)` for a real field — the core's own convention, verified equal to
# `rfft(u)/n^D` at 0.0. Its axes carry the layout in their own types, so they are restated here as the
# core's analytic axes: `SpectralLayout.is_half` and `full_size` dispatch on those. The spacing is the
# axis's own step, identical on both layouts.
function _core_axes(ks::Tuple, ms::Tuple, ::Type{FT}) where {FT}
    return ntuple(length(ms)) do d
        a = ks[d]
        dk = length(a) > 1 ? a[2] - a[1] : one(eltype(a))
        (d == 1 && FT <: Real) ? FIT.SpectralLayout.HalfAxis(ms[d], dk) :
                                 FIT.SpectralLayout.FullAxis(ms[d], dk)
    end
end

# `FFS.calculate_spectrum` is declared on `FlowGeometries.Grids.AbstractGrid` and selects its transform
# from the grid, so a grid passed here reaches whichever one its geometry and sampling admit.
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
    axes_ = _core_axes(ks, ms, eltype(first(velocity_fields)))
    return FIT.calculate_energy_transfer(method, coeffs, axes_; kwargs...)
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
    nd  = length(coords_vecs)
    FT  = float(eltype(coords_vecs[1]))
    geom = FFS.FlowGeometries.Geometry.CartesianGeometry{FT}()
    grid = FFS.FlowGeometries.Grids.StructuredGrid(geom, coords_vecs...;
                                                   topology = ntuple(_ -> true, nd), period = domain_size)
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
sizes and types itself. `ET` is the FIELD's element type, and it fixes the coefficient layout on both
plans: the packed half for a real field, the full cube for a complex one.

The wavenumbers come from the analysis plan, so a halved axis arrives carrying the Nyquist twin the
transform attached, and are restated as the core's analytic axes for `SpectralLayout` to dispatch on.
"""
function _ffs_workspace(proto, grid, ns::Tuple, ::Type{ET}, D::Int, spectral, execution) where {ET}
    ap = FFS.plan_spectrum(grid, ET, ns; transform = spectral, execution = execution, batch = (D,))
    sp = FFS.plan_synthesis(grid, ET, ns; transform = spectral, execution = execution, batch = (D,))
    ks = _core_axes(FFS.wavenumbers(ap), ns, ET)
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
    nd = ndims(grid)
    Ls = ntuple(d -> FFS.FlowGeometries.Grids.period(grid, d), nd)
    all(L -> isfinite(L) && L > 0, Ls) || throw(ArgumentError(
        "physical → spectral expands in a periodic basis, and direction " *
        "$(findfirst(L -> !(isfinite(L) && L > 0), Ls)) of this grid carries no period. Build it with " *
        "`topology = true` (or `periodic`/`period`) in every direction."))
    return size(grid)
end

"""
    ToSpectralWorkspace(velocity_fields, grid::FlowGeometries.Grids.AbstractGrid; spectral, execution, real_layout)

Analysis workspace with the domain given as a `FlowGeometries` grid. The transform is
`FlowFieldSpectra.plan_spectrum` on that grid, so the grid's own geometry, spacing and topology select
it; a real field takes the half layout.
"""
function FIT.ToSpectralWorkspace(
    velocity_fields::Tuple,
    grid::FFS.FlowGeometries.Grids.AbstractGrid;
    spectral::SpectralBackends.AbstractSpectralBackend = SpectralBackends.AutoSpectralBackend(),
    # The thread count is baked into the transform plan, and a multithreaded plan spawns a task per
    # thread on every execution, so this path allocates per call unless the plan is pinned. The rest of
    # the package pins its transforms the same way (`fft_nthreads = 1`); pass
    # `execution = ThreadedBackend()` to thread a single large transform.
    execution::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.SerialBackend(),
    real_layout::Bool = eltype(first(velocity_fields)) <: Real,
)
    D = length(velocity_fields)
    D >= 1 || throw(ArgumentError("to_spectral needs ≥1 physical field component."))
    # This entry analyses gridded data onto a Cartesian mode grid. The spherical transforms and the
    # scattered-point providers answer a different question and are reached by their own entries.
    spectral isa Union{FIT.Types.FINUFFTBackend, FIT.Types.NonuniformFFTsBackend,
                       SpectralBackends.FSHTSpectralBackend,
                       SpectralBackends.NUFSHTSpectralBackend} && throw(ArgumentError(
        "`to_spectral` analyses gridded data; got spectral = $(nameof(typeof(spectral))). Scattered " *
        "Cartesian points take `to_spectral(fields, points, ms; Ls)` with a NUFFT provider, and " *
        "spherical data `calculate_energy_transfer(SphericalTransferMethod(), field, grid)`."))
    real_layout && !(eltype(first(velocity_fields)) <: Real) && throw(ArgumentError(
        "the half spectral layout stores the non-redundant half of a REAL field's transform, and these " *
        "fields are $(eltype(first(velocity_fields))); pass `real_layout = false` for the full spectrum."))
    ns = _check_fourier_grid(grid)
    FT = float(real(eltype(first(velocity_fields))))
    # The transform's layout follows the element type it is planned for: a real field packs to the
    # non-redundant half, a complex one to the full cube. A real field asked for the full spectrum is
    # therefore staged complex.
    ET = real_layout ? FT : Complex{FT}
    return _ffs_workspace(first(velocity_fields), grid, ns, ET, D, spectral, execution)
end

"""
    to_spectral(velocity_fields, grid::FlowGeometries.Grids.AbstractGrid; spectral, execution, real_layout)

Physical → spectral on a `FlowGeometries` grid; see [`to_spectral`](@ref).
"""
FIT.to_spectral(velocity_fields::Tuple, grid::FFS.FlowGeometries.Grids.AbstractGrid; kwargs...) =
    FIT.to_spectral!(FIT.ToSpectralWorkspace(velocity_fields, grid; kwargs...), velocity_fields)

# Per-axis coordinate vectors describe the uniform periodic box `L = N·Δ` that `domain_size_from_coords`
# reports and every spectral entry transforms on; the grid carries that period explicitly so the
# wavenumbers are the ones the coordinate form has always produced.
# The wavenumber axes fix the periodic box completely: each carries the full grid length `n` it indexes,
# and its spacing is `Δk = 2π/L`. The grid rebuilt from them is the one the forward transform ran on.
function FIT._to_spectral_workspace_on_coeffs(
    velocity_hat::AbstractArray, ks::Tuple;
    spectral::SpectralBackends.AbstractSpectralBackend = SpectralBackends.AutoSpectralBackend(),
    execution::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.SerialBackend(),
)
    nd = length(ks)
    ns = FIT.SpectralLayout.full_size(ks)
    ms = FIT.SpectralLayout.spectral_size(ks)
    size(velocity_hat)[1:nd] == ms || throw(DimensionMismatch(
        "coefficient size $(size(velocity_hat)[1:nd]) does not match the wavenumber grid $ms."))
    D = size(velocity_hat, nd + 1)
    FT = real(eltype(velocity_hat))
    Ls = ntuple(d -> begin
        a = ks[d]
        dk = length(a) > 1 ? abs(a[2] - a[1]) : one(FT)
        FT(2π) / FT(dk)
    end, nd)
    geom = FFS.FlowGeometries.Geometry.CartesianGeometry{FT}()
    axes_ = ntuple(d -> range(zero(FT); step = Ls[d] / ns[d], length = ns[d]), nd)
    grid = FFS.FlowGeometries.Grids.StructuredGrid(
        geom, axes_...; topology = ntuple(_ -> true, nd), period = Ls)
    # A half layout is the transform of a real field, so the plans are built for one.
    ET = FIT.SpectralLayout.is_half(ks) ? FT : Complex{FT}
    return _ffs_workspace(velocity_hat, grid, ns, ET, D, spectral, execution)
end

function FIT._to_spectral_workspace_on_axes(velocity_fields::Tuple, coords_vecs::Tuple; kwargs...)
    FIT.Utils.validate_velocity_input(velocity_fields, length(coords_vecs))
    FIT.Utils.validate_uniform_grid(coords_vecs)
    Ls = FIT.Utils.domain_size_from_coords(coords_vecs)
    FT = float(real(eltype(first(velocity_fields))))
    geom = FFS.FlowGeometries.Geometry.CartesianGeometry{FT}()
    grid = FFS.FlowGeometries.Grids.StructuredGrid(
        geom, coords_vecs...; topology = ntuple(_ -> true, length(coords_vecs)), period = Ls)
    return FIT.ToSpectralWorkspace(velocity_fields, grid; kwargs...)
end


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
    decompose_field(decomp, velocity_fields::Tuple, grid; spectral, execution)

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
