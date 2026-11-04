module FlowInvariantTransferFlowFieldSpectraExt

using FlowFieldSpectra: FlowFieldSpectra as FFS, FlowGeometries as FG
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

# A range's samples closing their period at `n·Δ`, the FFT's own sampling, as a `PeriodicUniformAxis`; any other
# axis as it is.
_closed_axis(x::FG.Axes.PeriodicUniformAxis) = x
_closed_axis(x::AbstractRange) = FG.Axes.PeriodicUniformAxis(first(x), length(x) * step(x), length(x))
_closed_axis(x::AbstractVector) = x

# The all-periodic Cartesian grid over `axes`: with no period given each closes at its own `n·Δ`, and with one
# given the axes keep their samples and wrap at it.
_periodic_grid(geom, axes::Tuple, ::Nothing) =
    FFS.FlowGeometries.Grids.StructuredGrid(geom, map(_closed_axis, axes)...; topology = _all_periodic(axes))
_periodic_grid(geom, axes::Tuple, period) =
    FFS.FlowGeometries.Grids.StructuredGrid(geom, axes...; topology = _all_periodic(axes), period = period)

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

# Coordinate-vector convenience: an all-periodic Cartesian grid over those axes. A uniform range with no
# `domain_size` closes its period at `n·Δ`; a stretched or plain-vector axis needs `domain_size`, its
# per-direction period.
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
    grid = _periodic_grid(geom, coords_vecs, domain_size)
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
    grid = FFS.FlowGeometries.Grids.StructuredGrid(
        geom, map((L, n) -> FG.Axes.PeriodicUniformAxis{FT}(zero(FT), L, n), Ls, ns)...)
    # A half layout is the transform of a real field, so the plans are built for one.
    ET = _field_type(first(ks), FT)
    return _ffs_workspace(velocity_hat, grid, ns, ET, D, transform, execution)
end

# Per-axis coordinate vectors describe the uniform periodic box `L = N·Δ` of `domain_size_from_coords`, each
# axis a `PeriodicUniformAxis` closing there.
function FIT._to_spectral_workspace_on_axes(velocity_fields::Tuple, coords_vecs::Tuple; kwargs...)
    FIT.Utils.validate_velocity_input(velocity_fields, length(coords_vecs))
    FIT.Utils.validate_uniform_grid(coords_vecs)
    FT = float(real(eltype(first(velocity_fields))))
    geom = FFS.FlowGeometries.Geometry.CartesianGeometry{FT}()
    return FIT.ToSpectralWorkspace(velocity_fields, _periodic_grid(geom, coords_vecs, nothing); kwargs...)
end

_field_type(::FTB.RFFTAxis, ::Type{T}) where {T} = T
_field_type(::AbstractVector, ::Type{T}) where {T} = Complex{T}

"""
    ToSpectralWorkspace(fields::NTuple{M,<:FlowGeometries.Decomposition.AbstractSplitField}; transform)

Build reusable analysis and synthesis of component fields held in FlowGeometries blocks.
The returned coefficients and physical staging field retain their process distribution.
"""
function FIT.ToSpectralWorkspace(fields::NTuple{M,<:FG.Decomposition.AbstractSplitField};
        transform::SpectralBackends.AbstractSpectralBackend = SpectralBackends.AutoSpectralBackend()) where {M}
    M > 0 || throw(ArgumentError("to_spectral needs at least one field component"))
    split = FG.Decomposition.split_of(first(fields))
    all(field -> FG.Decomposition.split_of(field) === split, fields) ||
        throw(ArgumentError("field components must live on the same split grid"))
    T = float(promote_type(map(eltype, fields)...))
    analysis = FFS.plan_spectrum(split, T; transform, batch = (M,))
    synthesis = FFS.plan_synthesis(analysis)
    physical = FFS.allocate_field(synthesis)
    coefficients = FFS.allocate_coefficients(analysis)
    transforms = (; dft! = (out, field) -> FFS.calculate_spectrum!(out, analysis, field),
                    idft! = (out, modes) -> FFS.synthesize!(out, synthesis, modes))
    return FIT.ToSpectralWorkspace(transforms, physical, coefficients, FFS.wavenumbers(analysis))
end

function FIT.to_spectral!(workspace::FIT.ToSpectralWorkspace{TF,RA},
        fields::NTuple{M,<:FG.Decomposition.AbstractSplitField}) where
        {TF,RA<:FG.Decomposition.AbstractSplitField,M}
    split = FG.Decomposition.split_of(workspace.field_phys)
    FG.Decomposition.foreach_block(split, workspace.field_phys, fields...) do block, output, components...
        size(output, ndims(output)) == M || throw(DimensionMismatch(
            "workspace holds $(size(output, ndims(output))) components; got $M"))
        foreach(components, ntuple(identity, Val(M))) do component, c
            ndims(component) == length(block.interior) || throw(DimensionMismatch(
                "each component must hold one field per grid cell"))
            view(output, block.interior..., c) .= view(component, block.interior...)
        end
        return nothing
    end
    workspace.tf.dft!(workspace.û, workspace.field_phys)
    return workspace.û, workspace.ks
end

FIT.to_spectral(fields::NTuple{M,<:FG.Decomposition.AbstractSplitField}; kwargs...) where {M} =
    FIT.to_spectral!(FIT.ToSpectralWorkspace(fields; kwargs...), fields)

function FIT.from_spectral!(workspace::FIT.ToSpectralWorkspace{TF,RA},
        coefficients::FG.Decomposition.AbstractSplitField) where
        {TF,RA<:FG.Decomposition.AbstractSplitField}
    workspace.tf.idft!(workspace.field_phys, coefficients)
    return workspace.field_phys
end

# ---------------------------------------------------------------------------
# Spectral flux of a velocity split over Distributed workers, on FlowFieldSpectra's split FFT: the components
# analysed together (`û = fft(u)/N`), the nonlinear term `N̂_i = FFT[Σ_j u_j ∂_j u_i]/N` formed from one synthesis
# of the velocity and one of each `i k_j û`, the products taken block by block, and each coefficient block's modes
# summed into the shells, the block sums added on the caller. The steps and the 2/3 rule are the serial
# `calculate_spectral_flux`'s.
# ---------------------------------------------------------------------------

"""
    calculate_energy_transfer(method::SpectralFluxMethod, fields::NTuple{D,<:FlowGeometries.Decomposition.AbstractWorkerField};
                              dealiasing = OrszagTwoThirds(), invariant = KineticEnergy(), geometry = IsotropicShells(),
                              transform = AutoSpectralBackend()) -> SpectralFluxResult

The spectral flux of a real velocity whose `D` components live on one FlowGeometries worker split of a uniform
periodic Cartesian grid of `D` directions, on the shells of `method.binning`, each worker transforming and multiplying
its own blocks: equal to [`calculate_spectral_flux`](@ref) on the gathered field, to rounding. `dealiasing` is
`OrszagTwoThirds()` or `NoDealiasing()`. Every worker needs FlowInvariantTransfer and FlowFieldSpectra loaded.
"""
function FIT.calculate_energy_transfer(method::FIT.Types.SpectralFluxMethod,
        fields::NTuple{D,<:FG.Decomposition.AbstractWorkerField};
        dealiasing::Union{FIT.Types.OrszagTwoThirds,FIT.Types.NoDealiasing} = FIT.Types.OrszagTwoThirds(),
        invariant::FIT.Types.AbstractInvariant = FIT.Types.KineticEnergy(),
        geometry::FIT.Types.ShellMagnitude = FIT.Types.IsotropicShells(),
        transform::SpectralBackends.AbstractSpectralBackend = SpectralBackends.AutoSpectralBackend()) where {D}
    split = FG.Decomposition.split_of(first(fields))
    all(f -> FG.Decomposition.split_of(f) === split, fields) ||
        throw(ArgumentError("velocity components must live on the same split grid"))
    ndims(FG.Decomposition.global_grid(split)) == D ||
        throw(DimensionMismatch("the spectral flux needs one velocity component per grid direction"))
    T = float(promote_type(map(eltype, fields)...))
    T <: Real || throw(ArgumentError("the split spectral flux takes a real velocity; got $T components"))
    plan = FFS.plan_spectrum(split, T; transform, batch = (D,))
    ks = FFS.wavenumbers(plan)
    return _split_spectral_flux(plan, FFS.plan_synthesis(plan), fields, ks, method.binning,
                                dealiasing isa FIT.Types.OrszagTwoThirds, invariant, geometry)
end

function _split_spectral_flux(plan, synth, fields::NTuple{D}, ks, binning, trunc::Bool, invariant, geometry) where {D}
    FT = real(FFS.coefficient_type(plan))
    # Per axis, each mode's derivative wavenumber, its wavenumber, and whether the 2/3 rule keeps it.
    kd = map(a -> FT[FTB.derivative_wavenumber(a, i) for i in 1:length(a)], ks)
    kr = map(a -> collect(FT, a), ks)
    kept = map(a -> Bool[!trunc || abs(FTB.mode_frequency(a, i)) <= FTB.dealias_cutoff(a, 2) for i in 1:length(a)], ks)
    u = FFS.allocate_field(synth)
    _stage_components!(u, fields)
    û = FFS.allocate_coefficients(plan)
    FFS.calculate_spectrum!(û, plan, u)
    spec = FFS.allocate_coefficients(plan)
    _kept_coefficients!(spec, û, kept, nothing)
    FFS.synthesize!(u, synth, spec)
    N = FFS.allocate_field(synth)
    g = FFS.allocate_field(synth)
    for j in 1:D
        _kept_coefficients!(spec, û, kept, kd[j], j)
        FFS.synthesize!(g, synth, spec)
        _advect!(N, u, g, j)
    end
    FFS.calculate_spectrum!(spec, plan, N)
    kmax = FIT.ShellBinning.max_shell_coordinate(geometry, ks)
    edges = FIT.ShellBinning.shell_edges(binning, kmax)
    centers = collect(FIT.ShellBinning.shell_centers(binning, kmax))
    dims = FIT.ShellBinning._shell_dims(geometry, Val(D))
    fold = FT[FTB.fold_weight(first(ks), i) for i in 1:length(first(ks))]
    sums = FG.Decomposition.map_blocks(FG.Decomposition.split_of(û), û, spec) do b, a, n
        _block_shell_sums(b, a, n, ks, kr, kept, dims, edges, fold, invariant)
    end
    transfer = reduce(+, sums)
    return FIT.Types.SpectralFluxResult(centers, edges, transfer, cumsum(transfer))
end

# The velocity's components into the batch axis of `u`, block by block.
function _stage_components!(u, fields::NTuple{D}) where {D}
    FG.Decomposition.foreach_block(FG.Decomposition.split_of(u), u, fields...) do b, out, components...
        foreach(components, ntuple(identity, Val(D))) do component, c
            view(out, b.interior..., c) .= view(component, b.interior...)
        end
        return nothing
    end
    return u
end

# `spec = û` at the modes the 2/3 rule keeps, zero elsewhere; given `kj`, the derivative wavenumbers of axis `j`,
# `spec = i k_j û` there.
_kept_coefficients!(spec, û, kept, ::Nothing) = _kept_coefficients!(spec, û, kept, nothing, 0)
function _kept_coefficients!(spec, û, kept, kj, j::Int)
    FG.Decomposition.foreach_block(FG.Decomposition.split_of(û), spec, û) do b, s, a
        off = _mode_offset(b)
        @inbounds for c in axes(a, ndims(a)), L in CartesianIndices(map(length, b.ranges))
            G = L + off
            s[L, c] = _kept(kept, G) ? _mode_factor(kj, j, G, eltype(s)) * a[L, c] : zero(eltype(s))
        end
        return nothing
    end
    return spec
end

@inline _mode_factor(::Nothing, _, _, ::Type{C}) where {C} = one(C)
@inline _mode_factor(kj::AbstractVector, j::Int, G::CartesianIndex, ::Type{C}) where {C} = C(im * @inbounds(kj[G[j]]))

@inline _kept(kept::NTuple{N,AbstractVector{Bool}}, G::CartesianIndex{N}) where {N} =
    all(map((k, i) -> @inbounds(k[i]), kept, Tuple(G)))

# A coefficient block's first mode less one, per axis: local index `L` is global mode `L + offset`.
@inline _mode_offset(b) = CartesianIndex(map(r -> first(r) - 1, b.ranges))

# `N[…, i] += u[…, j] · g[…, i]` on each block's own cells.
function _advect!(N, u, g, j::Int)
    FG.Decomposition.foreach_block(FG.Decomposition.split_of(N), N, u, g) do b, n, v, w
        uj = view(v, b.interior..., j)
        for i in axes(n, ndims(n))
            view(n, b.interior..., i) .+= uj .* view(w, b.interior..., i)
        end
        return nothing
    end
    return N
end

# A block's local arrays read at global mode indices, the indices `Invariants._density_at` and the wavenumber axes
# take.
struct _AtModes{A<:AbstractArray,N}
    a::A
    offset::CartesianIndex{N}
end
Base.eltype(::Type{_AtModes{A,N}}) where {A,N} = eltype(A)
@inline Base.getindex(m::_AtModes{A,N}, G::CartesianIndex{N}, c::Int) where {A,N} = @inbounds m.a[G - m.offset, c]

# One coefficient block's shell sums of the transfer density, at the modes the 2/3 rule keeps.
function _block_shell_sums(b, a, n, ks, kr, kept, dims, edges, fold, invariant)
    FT = eltype(fold)
    nshell = length(edges) - 1
    sums = zeros(FT, nshell)
    off = _mode_offset(b)
    û, N̂ = _AtModes(a, off), _AtModes(n, off)
    M = size(a, ndims(a))
    @inbounds for L in CartesianIndices(map(length, b.ranges))
        G = L + off
        _kept(kept, G) || continue
        k = sqrt(sum(d -> kr[d][G[d]]^2, dims; init = zero(FT)))
        s = searchsortedlast(edges, k)
        s = (1 <= s <= nshell) ? s : (s == nshell + 1 && k == edges[end]) ? nshell : 0
        s == 0 && continue
        sums[s] += fold[G[1]] * FIT.Invariants._density_at(invariant, û, N̂, ks, G, M)
    end
    return sums
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
