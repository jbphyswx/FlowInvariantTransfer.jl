# ---------------------------------------------------------------------------
# Scattered Cartesian samples: their Fourier coefficients on a uniform mode grid, and the
# coarse-graining flux at the samples, through FlowTransformBindings' NUFFT plans.
#
# A sample at `x` sits at the phase `2π (x - xₘᵢₙ) / L` of its periodic box, `xₘᵢₙ` the smallest
# coordinate on that axis, so samples on the uniform `L`-grid give `û = fft(u)/Nᵈ`.
#
# The coefficients are the quadrature `û(k) = (1/V) Σⱼ wⱼ uⱼ e^{-ik·xⱼ}` of the box volume `V = ∏L`,
# with `wⱼ` the volume sample `j` stands for. Without `weights` every sample stands for `V/N`, which
# gives `type1(u)/N`: exact on the uniform grid, and an estimate biased toward where the samples
# cluster when their density varies.
# ---------------------------------------------------------------------------

const _NUFFTLibrary = Union{FlowTransformBindings.FINUFFTBackend, FlowTransformBindings.NonuniformFFTsBackend}

_nufft_library(t::_NUFFTLibrary) = t
_nufft_library(t::SpectralBackends.AbstractNonUniformFastFourierTransformSpectralBackend) = throw(ArgumentError(
    "$(nameof(typeof(t))) names no NUFFT library; pass FlowTransformBindings.NonuniformFFTsBackend() " *
    "(`using NonuniformFFTs`) or FlowTransformBindings.FINUFFTBackend() (`using FINUFFT`)."))

_library_threads(::ComputationalBackends.AbstractThreadedBackend) = Threads.nthreads()
_library_threads(::ComputationalBackends.AbstractExecutionBackend) = 1

@inline _page(A::AbstractArray, c::Int) = view(A, ntuple(_ -> Colon(), Val(ndims(A) - 1))..., c)

function _scattered_npoints(scatter_coords::Tuple, ms::Tuple, Ls::Tuple)
    nd = length(scatter_coords)
    nd == length(ms) || throw(ArgumentError("scatter_coords ($(nd)D) and ms ($(length(ms))D) must match"))
    1 <= nd <= 3 || throw(ArgumentError("the scattered NUFFT entries take 1, 2 or 3 dimensions; got $nd"))
    length(Ls) == nd || throw(ArgumentError("Ls ($(length(Ls))D) must match scatter_coords ($(nd)D)"))
    all(>(0), Ls) || throw(ArgumentError("Ls must be positive (periodic domain size per dimension); got $Ls"))
    N = length(first(scatter_coords))
    all(c -> length(c) == N, scatter_coords) ||
        throw(DimensionMismatch("all scatter_coords must have equal length"))
    return N
end

function _plan_keywords(scatter_coords::Tuple, Ls::Tuple, ::Type{FT}, tol, execution) where {FT}
    return (; tol = tol === nothing ? FlowTransformBindings.default_tolerance(FT) : FT(tol),
              period = map(FT, Ls),
              origin = map(x -> FT(minimum(x)), scatter_coords),
              nthreads = _library_threads(execution))
end

# The per-sample quadrature weights and the factor the type-1 sum is scaled by: `1/N` with no weights,
# `1/∏L` with them.
_quadrature(::Nothing, N::Int, Ls::Tuple, ::Type{FT}) where {FT} = (nothing, one(FT) / FT(N))
function _quadrature(w::AbstractVector, N::Int, Ls::Tuple, ::Type{FT}) where {FT}
    length(w) == N || throw(DimensionMismatch("weights has length $(length(w)) for $N samples"))
    all(isfinite, w) || throw(ArgumentError("weights must be finite"))
    return (collect(FT, w), one(FT) / FT(prod(Ls)))
end

# `dest = w ⊙ field`, or `field` with no weights.
_weighted!(dest, ::Nothing, field) = copyto!(dest, field)
_weighted!(dest, w::AbstractVector, field) = (copyto!(dest, field); dest .*= w; dest)

# The type-1 plan of a real field: its real values where the library transforms them natively, the full
# complex spectrum otherwise.
_value_type(lib, ::Type{T}) where {T} = T
_value_type(::FlowTransformBindings.FINUFFTBackend, ::Type{T}) where {T<:Real} = Complex{T}
_analysis_plan(lib, ::Type{T}, nodes, ms, kw) where {T} =
    FlowTransformBindings.plan_nufft(lib, _value_type(lib, T), nodes, ms;
                                     order = FlowTransformBindings.FFTModes(), kw...)

_coefficient_box(::Type{<:Real}, ms) = FlowTransformBindings.box_size(ms, map(_ -> true, ms))
_coefficient_box(::Type{<:Complex}, ms) = ms
_published_coefficients(::Type{<:Real}, plan, spec, box, B) = _half_coefficients(plan, spec, box, B)
_published_coefficients(::Type{<:Complex}, plan, spec, box, B) = spec

_half_coefficients(::FlowTransformBindings.AbstractNUFFTPlan{<:Real}, spec, box, B) = spec
_half_coefficients(::FlowTransformBindings.AbstractNUFFTPlan{<:Complex}, spec, box, B) =
    FlowTransformBindings.packed_half_view(spec, box, FlowTransformBindings.packed_size(Real, box), B)

# Copies `x` onto the device a `GPUBackend` names; the KernelAbstractions extension adds that method.
_nufft_to_device(::ComputationalBackends.AbstractExecutionBackend, x) = x

# ---------------------------------------------------------------------------
# Coarse-graining flux at the samples, through CoarseGrainingEnergyFluxes' nonuniform-FFT filter
# ---------------------------------------------------------------------------

"""
    nufft_coarse_graining_flux(velocity_fields, scatter_coords, ℓ, filter, ms; transform, Ls,
                               tol = nothing, weights = nothing, return_diagnostics = false,
                               execution = SerialBackend())
        -> CoarseGrainingFluxResult

Coarse-graining energy flux `Π_ℓ(x)` at scattered Cartesian points, by CoarseGrainingEnergyFluxes'
nonuniform-FFT filter over the points as a FlowGeometries node set. Requires
`using CoarseGrainingEnergyFluxes`.

`velocity_fields` and `scatter_coords` are `D`-tuples of length-`N` vectors, `D` 2 or 3: the velocity
components and the point coordinates. Each field is analyzed on the `ms` modes of the periodic box,
`û = (1/∏L) Σⱼ wⱼ uⱼ e^{-ik·xⱼ}`, multiplied by the transfer function of `filter`'s
CoarseGrainingEnergyFluxes kernel and synthesized at the points; `S̄ᵢⱼ` is the derivative of that filtered
series and `Π_ℓ = -τ̄ᵢⱼ S̄ᵢⱼ`. `weights` `wⱼ` are the volumes the samples stand for (`Σw = ∏L`); without
them every sample stands for `∏L/N`, which is exact for samples on the uniform grid and an estimate
weighted toward where the samples cluster otherwise. The domain mean is the same quadrature of `Π`. An
even count `m` in `ms` holds the modes `|n| ≤ m/2`, the two at `±m/2` at half weight.

`transform` (required) names the NUFFT library:
`FlowTransformBindings.NonuniformFFTsBackend()` (`using NonuniformFFTs`) or
`FlowTransformBindings.FINUFFTBackend()` (`using FINUFFT`). `Ls` (required) is the periodic domain size
per dimension. It sets the wavenumber grid `k = 2πn/L`, and hence the filter's transfer function and the
strain: points in `[xₘᵢₙ, xₘᵢₙ+Lₐ)` under-span the period, so `L` is never inferred from them.

`tol` is the transforms' relative accuracy (`nothing`: `FlowTransformBindings.default_tolerance` of the
coordinates' float type); `return_diagnostics = true` also returns `τ̄ᵢⱼ` and `S̄ᵢⱼ` at the points, each
`(N, D, D)`; `execution = ThreadedBackend()` threads each transform. For a filter-scale sweep or a time
series on one point set, build an [`NUFFTCoarseGrainingWorkspace`](@ref) and call
[`nufft_coarse_graining_flux!`](@ref).
"""
function nufft_coarse_graining_flux(velocity_fields::Tuple, scatter_coords::Tuple, ℓ::Real,
                                    filter::Types.AbstractFilter, ms::Tuple;
                                    return_diagnostics::Bool = false, kwargs...)
    ws = NUFFTCoarseGrainingWorkspace(scatter_coords, ms; return_diagnostics = return_diagnostics, kwargs...)
    try
        return nufft_coarse_graining_flux!(ws, velocity_fields, ℓ, filter, ms;
                                           return_diagnostics = return_diagnostics)
    finally
        close!(ws)
    end
end

"""
    nufft_coarse_graining_flux_batch(velocity_fields_batch, scatter_coords, ℓ, filter, ms;
                                     transform, Ls, execution, kwargs...) -> Vector

Scattered coarse-graining flux for a batch of snapshots on **one** point set. The NUFFT plans and every
buffer are built once for the whole batch. `execution = ThreadedBackend()` (requires `using OhMyThreads`) threads over snapshots with one
workspace per chunk. Results are in input order and each owns its flux field.
"""
function nufft_coarse_graining_flux_batch(velocity_fields_batch, scatter_coords, ℓ, filter, ms;
                                          transform::SpectralBackends.AbstractNonUniformFastFourierTransformSpectralBackend,
                                          Ls::Tuple,
                                          execution::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.SerialBackend(),
                                          return_diagnostics::Bool = false,
                                          kwargs...)
    n = length(velocity_fields_batch)
    if n == 0
        FT = float(eltype(scatter_coords[1]))
        R = return_diagnostics ?
            Types.CoarseGrainingFluxResultWithDiagnostics{FT, Vector{FT}, Array{FT, 3}} :
            Types.CoarseGrainingFluxResult{FT, Vector{FT}}
        return R[]
    end
    return _nufft_cg_batch!(Types.resolve_execution(execution), velocity_fields_batch, scatter_coords,
                            ℓ, filter, ms; transform = transform, Ls = Ls, kwargs...)
end

# Serial reference: one workspace (plans + buffers) reused across the batch. `nufft_coarse_graining_flux!`
# wraps `ws.Π`, overwritten on the next snapshot, so each result is copied out.
function _nufft_cg_batch!(::ComputationalBackends.AbstractSerialBackend, batch, scatter_coords, ℓ, filter, ms;
                          transform, Ls, return_diagnostics::Bool = false, kwargs...)
    ws = NUFFTCoarseGrainingWorkspace(scatter_coords, ms; transform = transform, Ls = Ls,
                                      return_diagnostics = return_diagnostics, kwargs...)
    try
        return [deepcopy(nufft_coarse_graining_flux!(ws, vf, ℓ, filter, ms;
                                                     return_diagnostics = return_diagnostics))
                for vf in batch]
    finally
        close!(ws)
    end
end

function _nufft_cg_batch_threaded!(args...; kwargs...)
    throw(ArgumentError("execution = ThreadedBackend() for the scattered coarse-graining batch requires " *
                        "OhMyThreads. Run `using OhMyThreads` to load the extension."))
end
_nufft_cg_batch!(::ComputationalBackends.AbstractThreadedBackend, batch, scatter_coords, ℓ, filter, ms; kwargs...) =
    _nufft_cg_batch_threaded!(batch, scatter_coords, ℓ, filter, ms; kwargs...)

_nufft_cg_batch!(be::ComputationalBackends.AbstractExecutionBackend, batch, scatter_coords, ℓ, filter, ms; kwargs...) =
    Types.batch_unavailable("nufft_coarse_graining_flux_batch", be)

_nufft_cg_batch!(::ComputationalBackends.AbstractGPUBackend, batch, scatter_coords, ℓ, filter, ms; kwargs...) =
    throw(ArgumentError("nufft_coarse_graining_flux_batch keeps its workspace in host memory. On a GPUBackend, " *
                        "filter on the points as an FG UnstructuredGrid with CoarseGrainingEnergyFluxes' NUFFT engine."))

"""
    NUFFTCoarseGrainingWorkspace

Reusable resources for [`nufft_coarse_graining_flux!`](@ref): the points as a FlowGeometries node set,
CoarseGrainingEnergyFluxes' nonuniform-FFT filter plan over them (its transforms serve every scale and
filter, its transfer function refilled per call), the flux workspace, the spectra of the velocity and its
products, the `Π` output and, when built for them, the `(N, D, D)` stress and strain. A repeat call
re-plans nothing; its result wraps the reused `Π`.

The plans hold library resources that [`close!`](@ref FlowInvariantTransfer.close!) releases; the
allocating entries close theirs before returning. One task at a time may use a workspace. Requires
`using CoarseGrainingEnergyFluxes`.
"""
struct NUFFTCoarseGrainingWorkspace{G, FP, W, AN, P, D, A, ST, M}
    grid::G               # the points as a FlowGeometries `UnstructuredGrid`
    filter_plan::FP       # CGEF's nonuniform-FFT filter plan over the points
    cgef_workspace::W     # CGEF.Diagnostics.ΠWorkspace
    analysis::AN          # spectra of the velocity and its products, refilled per call
    Π_out::P              # (N,) flux (the result wraps this)
    diagnostics::D        # nothing, or the (N, D, D) stress and strain
    weights::A            # per point: its volume over the total
    stage::ST             # one staging buffer per velocity component, each `nothing` if unused
    modes::M              # the mode counts `ms`
end
Base.show(io::IO, ::NUFFTCoarseGrainingWorkspace) = print(io, "NUFFTCoarseGrainingWorkspace(…)")
Base.show(io::IO, ::MIME"text/plain", w::NUFFTCoarseGrainingWorkspace) = show(io, w)

"""
    NUFFTCoarseGrainingWorkspace(scatter_coords, ms; transform, Ls, tol = nothing, weights = nothing,
                                 return_diagnostics = false, execution = SerialBackend())

Build the reusable coarse-graining workspace over `scatter_coords` for the NUFFT library `transform`
names, with the periodic domain size `Ls` and the samples' quadrature `weights`; see
[`nufft_coarse_graining_flux`](@ref). `execution = ThreadedBackend()` threads each transform. Requires
`using CoarseGrainingEnergyFluxes`.
"""
NUFFTCoarseGrainingWorkspace(scatter_coords::Tuple, ms::Tuple; kwargs...) =
    _nufft_cg_workspace(scatter_coords, ms; kwargs...)

_nufft_cg_workspace(args...; kwargs...) = throw(ArgumentError(
    "the scattered coarse-graining flux requires CoarseGrainingEnergyFluxes.jl. Run " *
    "`using CoarseGrainingEnergyFluxes` to load the extension."))

# The flux of one call on a workspace: a method of the CoarseGrainingEnergyFluxes extension.
function _nufft_cg_flux! end

"""
    nufft_coarse_graining_flux!(ws::NUFFTCoarseGrainingWorkspace, velocity_fields, ℓ, filter, ms;
                                return_diagnostics = false) -> CoarseGrainingFluxResult

In-place scattered coarse-graining flux reusing `ws`: every transform runs through its plan and every
intermediate is a workspace buffer. The result wraps `ws.Π_out`, and a later call overwrites it.
"""
nufft_coarse_graining_flux!(ws::NUFFTCoarseGrainingWorkspace, velocity_fields::Tuple, ℓ::Real,
                            filter::Types.AbstractFilter, ms::Tuple; return_diagnostics::Bool = false) =
    _nufft_cg_flux!(ws, velocity_fields, ℓ, filter, ms, return_diagnostics)

# ---------------------------------------------------------------------------
# Scattered → uniform Fourier coefficients
# ---------------------------------------------------------------------------

"""
    NUFFTToSpectralWorkspace

Reusable resources for the in-place scattered → uniform reconstruction [`to_spectral!`](@ref): the
type-1 plan over the points, the uniform wavenumber grid `ks`, and every working buffer (the `(ms…, D)`
coefficient array `û` plus the transform's input and output), so a repeat `to_spectral!` re-plans
nothing. The buffers live where `execution` puts them, on the device for a `GPUBackend`.

The plan holds library resources that [`close!`](@ref FlowInvariantTransfer.close!) releases; the
allocating [`to_spectral`](@ref) closes its own before returning. One task at a time may use a
workspace.
"""
struct NUFFTToSpectralWorkspace{P, KS, UH, CV, SP, QW, R<:Real}
    plan::P              # type-1 plan over the points
    ks::KS               # per-axis uniform wavenumber vectors (returned with û)
    û::UH                # (ms…, D) coefficient buffer (the result aliases this)
    scat::CV             # (N, D) type-1 input
    spec::SP             # type-1 output
    npoints::Int         # number of scattered points
    qw::QW               # (N,) quadrature weights on the points' device, or `nothing` for `∏L/N`
    scale::R             # factor of the type-1 sum: 1/N, or 1/∏L with `qw`
end
Base.show(io::IO, ::NUFFTToSpectralWorkspace) = print(io, "NUFFTToSpectralWorkspace(…)")
Base.show(io::IO, ::MIME"text/plain", w::NUFFTToSpectralWorkspace) = show(io, w)

"""
    NUFFTToSpectralWorkspace(T, scatter_coords, ms; transform, Ls, tol = nothing, weights = nothing,
                             ncomponents = D, execution = AutoBackend())

Build the reusable scattered → uniform workspace over `scatter_coords` for the NUFFT library
`transform` names (`FlowTransformBindings.NonuniformFFTsBackend()` or
`FlowTransformBindings.FINUFFTBackend()`), with the periodic domain size `Ls` (`k = 2πn/L`; the
samples do not determine it) and the samples' quadrature `weights` (see [`to_spectral`](@ref)).
`execution = GPUBackend(dev)` builds the plan and buffers on `dev`; `ThreadedBackend()` threads each
transform. `T` is the field element type; its default is the coordinates' floating-point type.
Real fields use FlowTransformBindings' packed half of an odd coefficient box, with every even requested mode count
widened by one. Complex fields use the requested full box.
"""
function NUFFTToSpectralWorkspace(::Type{FE}, scatter_coords::Tuple, ms::Tuple;
                                  transform::SpectralBackends.AbstractNonUniformFastFourierTransformSpectralBackend,
                                  Ls::Tuple, ncomponents::Int = length(scatter_coords),
                                  tol::Union{Nothing, Real} = nothing,
                                  weights::Union{Nothing, AbstractVector} = nothing,
                                  execution::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend()) where {FE<:Number}
    lib = _nufft_library(transform)
    N = _scattered_npoints(scatter_coords, ms, Ls)
    ncomponents >= 1 || throw(ArgumentError("ncomponents must be ≥ 1"))
    FT = real(FE)
    box = _coefficient_box(FE, ms)
    ks = Utils.wavenumber_grid(FE, box, map(FT, Ls))
    exec = Types.resolve_execution(execution, first(scatter_coords))
    nodes = map(x -> _nufft_to_device(exec, x), scatter_coords)
    qw, scale = _quadrature(weights, N, Ls, FT)
    qwd = qw === nothing ? nothing : _nufft_to_device(exec, qw)
    kw = merge(_plan_keywords(scatter_coords, Ls, FT, tol, exec), (; ntrans = ncomponents))
    plan = _analysis_plan(lib, FE, nodes, box, kw)
    return _spectral_workspace(plan, FE, ks, box, ncomponents, N, qwd, scale)
end

function _spectral_workspace(plan, ::Type{FE}, ks, box, ncomponents, N, qw, scale) where {FE}
    spec = FlowTransformBindings.allocate_modes(plan)
    û = _published_coefficients(FE, plan, spec, box, ncomponents)
    return NUFFTToSpectralWorkspace(plan, ks, û, FlowTransformBindings.allocate_values(plan),
                                    spec, N, qw, scale)
end

NUFFTToSpectralWorkspace(scatter_coords::Tuple, ms::Tuple; kwargs...) =
    NUFFTToSpectralWorkspace(float(eltype(first(scatter_coords))), scatter_coords, ms; kwargs...)

"""
    close!(ws::NUFFTToSpectralWorkspace) -> ws

Release the workspace's NUFFT resources.
"""
close!(ws::NUFFTToSpectralWorkspace) = (FlowTransformBindings.close!(ws.plan); ws)

"""
    to_spectral!(ws::NUFFTToSpectralWorkspace, velocity_fields) -> (velocity_hat, ks)

In-place scattered → uniform reconstruction reusing `ws`: `û = (1/∏L) Σⱼ wⱼ uⱼ e^{-ik·xⱼ}` (`type1(u)/N`
without weights) in FFTW mode order, returned with the uniform wavenumber grid `ks`. The returned
`velocity_hat` aliases `ws.û`, and a later call overwrites it.
"""
function to_spectral!(ws::NUFFTToSpectralWorkspace, velocity_fields::Tuple)
    D = size(ws.û, ndims(ws.û))
    length(velocity_fields) == D || throw(DimensionMismatch(
        "to_spectral! got $(length(velocity_fields)) fields; workspace was built for $D components"))
    for c in 1:D
        length(velocity_fields[c]) == ws.npoints ||
            throw(DimensionMismatch("field length ≠ workspace points $(ws.npoints)"))
        _weighted!(view(ws.scat, :, c), ws.qw, velocity_fields[c])
    end
    FlowTransformBindings.nufft_type1!(ws.spec, ws.plan, ws.scat)
    ws.û .*= ws.scale
    return (ws.û, ws.ks)
end

"""
    to_spectral(velocity_fields::Tuple, scatter_coords::Tuple, ms::Tuple; transform, Ls, tol = nothing,
                weights = nothing, execution = AutoBackend()) -> (velocity_hat, ks)

Scattered-Cartesian physical-space entry: reconstruct the Fourier coefficients `velocity_hat` on a
uniform `ms = (m₁,…,m_D)` grid from velocity components sampled at **scattered** (non-uniform) points,
through a NUFFT type-1 transform, and return them with the matching uniform wavenumber grid `ks`, the
input of every Cartesian flux diagnostic. Real fields use the packed half of FlowTransformBindings' odd coefficient
box: an even requested count is widened by one on that axis. Complex fields use the full `ms` box.

`scatter_coords = (x, y[, z])` hold one coordinate per sample on each axis (each vector of length `N`,
the number of samples); `ms` is the target uniform mode count per dimension. `transform` (required)
names the NUFFT library: `FlowTransformBindings.NonuniformFFTsBackend()`
(`using NonuniformFFTs`) or `FlowTransformBindings.FINUFFTBackend()` (`using FINUFFT`). For
uniform-grid data use the 2-argument form with `SpectralBackends.FFTSpectralBackend()`.

The coefficients are the quadrature `û(k) = (1/∏L) Σⱼ wⱼ uⱼ e^{-ik·xⱼ}`, with `weights` `wⱼ` the
volume sample `j` stands for (`Σw = ∏L`). It is exact for a field band-limited to `ms` when the weights
integrate every `e^{iq·x}` with `|q_d| < m_d` exactly. Without `weights` every sample stands for `∏L/N`
and `û = type1(u)/N`: exact for samples on the uniform `L`-grid (`û = fft(u)/Nᵈ`), and otherwise an
estimate that weights each sample equally, so a region sampled densely counts for more than its
volume.

`tol` sets the NUFFT accuracy (`nothing`: `FlowTransformBindings.default_tolerance` of the coordinates'
float type). `Ls` (required) is the periodic domain size per dimension and fixes the wavenumbers
`k = 2πn/L`: the samples live in `[xₘᵢₙ, xₘᵢₙ+Lₐ)` and under-span the period, so `L` cannot be
inferred from them. `execution = GPUBackend(dev)` (`dev` a KernelAbstractions backend) builds the plan
and buffers on the device, so the scattered → `velocity_hat` step runs there.
"""
function to_spectral(velocity_fields::Tuple, scatter_coords::Tuple, ms::Tuple;
                     transform::SpectralBackends.AbstractSpectralBackend,
                     tol::Union{Nothing, Real} = nothing,
                     weights::Union{Nothing, AbstractVector} = nothing,
                     Ls::Tuple,
                     execution::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend())
    transform isa SpectralBackends.AbstractNonUniformFastFourierTransformSpectralBackend || throw(ArgumentError(
        "the 3-argument to_spectral(fields, scatter_coords, ms; …) is the scattered-Cartesian NUFFT " *
        "entry: pass transform = FlowTransformBindings.NonuniformFFTsBackend() or " *
        "FlowTransformBindings.FINUFFTBackend(). For uniform-grid data use " *
        "to_spectral(fields, coords_vecs; transform = SpectralBackends.FFTSpectralBackend())."))
    FE = float(promote_type(map(eltype, velocity_fields)...))
    execution = Types.resolve_execution(execution, first(velocity_fields))
    ws = NUFFTToSpectralWorkspace(FE, scatter_coords, ms; transform = transform, weights = weights,
                                  ncomponents = length(velocity_fields), tol = tol, Ls = Ls, execution = execution)
    try
        return to_spectral!(ws, velocity_fields)
    finally
        close!(ws)
    end
end
