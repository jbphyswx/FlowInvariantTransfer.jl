# ---------------------------------------------------------------------------
# Scattered Cartesian samples: their Fourier coefficients on a uniform mode grid, and the
# coarse-graining flux at the samples, through FlowTransformBindings' NUFFT plans.
#
# A sample at `x` sits at the phase `2π (x - xₘᵢₙ) / L` of its periodic box, `xₘᵢₙ` the smallest
# coordinate on that axis, so samples on the uniform `L`-grid give `û = fft(u)/Nᵈ`.
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

function _plan_keywords(scatter_coords::Tuple, Ls::Tuple, ::Type{FT}, tol::Real, execution) where {FT}
    nd = length(scatter_coords)
    return (; tol = FT(tol), period = ntuple(d -> FT(Ls[d]), nd),
              origin = ntuple(d -> FT(minimum(scatter_coords[d])), nd),
              nthreads = _library_threads(execution))
end

# The type-1 plan of a real field: its real values where the library transforms them natively, the full
# complex spectrum otherwise.
_analysis_plan(lib, ::Type{FT}, nodes, ms, kw) where {FT} =
    FlowTransformBindings.plan_nufft(lib, FlowTransformBindings.has_real_transform(lib) ? FT : Complex{FT},
                                     nodes, ms; order = FlowTransformBindings.FFTModes(), kw...)

# One mode of the Hermitian expansion of a real-data NUFFT type-1 half spectrum onto the full fftfreq
# grid. `fk_half` holds axis-1 modes `0…ms₁÷2` (rfftfreq) with fftfreq on the rest; `us` is the plan's
# oversampled half spectrum (sizes `novs`) and `gk[d]` its kernel Fourier coefficients.
#
#   k₁ ≥ 0                → straight from `fk_half`.
#   k₁ < 0                → Hermitian mirror `conj(fk_half[-k])`.
#   k₁ < 0 and some even
#   axis d ≥ 2 at −N_d/2  → the mirror is `C[-k₁, +N_d/2]`. At scattered points `+N_d/2 ≠ −N_d/2`, and
#                           the half holds only `−N_d/2`; `us` holds `+N_d/2`, deconvolved here by
#                           `normfactor / ∏ ĝ`.
#
# Exact for even and odd sizes. Written per-mode so the host loop and the device kernel share it.
@inline function _r2c_value(I, fk_half, us, gk, ms::NTuple{D, Int}, novs::NTuple{D, Int}, normfactor, invN) where {D}
    # fftfreq integer at each output index, then the mode's own index in the mirrored grid.
    kk = ntuple(d -> (Int(I[d]) - 1 <= (ms[d] - 1) ÷ 2) ? (Int(I[d]) - 1) : (Int(I[d]) - 1 - ms[d]), D)
    k1 = kk[1]
    if k1 >= 0
        return fk_half[CartesianIndex(ntuple(d -> d == 1 ? k1 + 1 : Int(I[d]), D))] * invN
    end
    even_nyquist = false
    for d in 2:D
        (iseven(ms[d]) && kk[d] == -(ms[d] ÷ 2)) && (even_nyquist = true)
    end
    if !even_nyquist
        # output index of −freq(I[d])
        return conj(fk_half[CartesianIndex(ntuple(d -> d == 1 ? -k1 + 1 :
                                                  (Int(I[d]) == 1 ? 1 : ms[d] - Int(I[d]) + 2), D))]) * invN
    end
    negk = ntuple(d -> -kk[d], D)
    ovsI = CartesianIndex(ntuple(d -> negk[d] >= 0 ? negk[d] + 1 : novs[d] + negk[d] + 1, D))
    β = normfactor / gk[1][negk[1] + 1]
    for d in 2:D
        β /= gk[d][negk[d] >= 0 ? negk[d] + 1 : ms[d] + negk[d] + 1] # ĝ is even ⇒ +N/2 lands in the −N/2 slot
    end
    return conj(β * us[ovsI]) * invN
end

# Hermitian expansion over the whole output grid, dispatched on the execution backend: this host method
# is a scalar loop, and the KernelAbstractions extension adds the device method. Both take arrays and
# plain numbers only.
function _r2c_expand!(::ComputationalBackends.AbstractExecutionBackend, full, fk_half, us, gk,
                      ms::NTuple{D, Int}, novs::NTuple{D, Int}, normfactor, invN) where {D}
    @inbounds for I in CartesianIndices(ms)
        full[I] = _r2c_value(I, fk_half, us, gk, ms, novs, normfactor, invN)
    end
    return full
end

# `full = type1(u)/N` over the `FFTModes` spectrum, from `spec`, the analysis plan's output of its last
# execution.
function _to_full!(execution, full, spec, plan::FlowTransformBindings.AbstractNUFFTPlan{<:Real},
                   ms::NTuple{D,Int}, invN) where {D}
    spectra, normfactor, gk = FlowTransformBindings.oversampled_spectra(plan)
    us = first(spectra)
    novs = ntuple(d -> d == 1 ? 2 * (size(us, 1) - 1) : size(us, d), Val(D))
    return _r2c_expand!(execution, full, spec, us, gk, ms, novs, normfactor, invN)
end
_to_full!(execution, full, spec, ::FlowTransformBindings.AbstractNUFFTPlan{<:Complex}, ms, invN) =
    (full .= spec .* invN; full)

# Copies `x` onto the device a `GPUBackend` names; the KernelAbstractions extension adds that method.
_nufft_to_device(::ComputationalBackends.AbstractExecutionBackend, x) = x

# ---------------------------------------------------------------------------
# Coarse-graining flux at the samples
# ---------------------------------------------------------------------------

"""
    nufft_coarse_graining_flux(velocity_fields, scatter_coords, ℓ, filter, ms; spectral, Ls,
                               tol = 1e-8, return_diagnostics = false, execution = SerialBackend())
        -> CoarseGrainingFluxResult

Coarse-graining energy flux `Π_ℓ(x)` at scattered (non-uniform) Cartesian points.

`velocity_fields` and `scatter_coords` are `D`-tuples of length-`N` vectors (`D ≤ 3`): the velocity
components and the point coordinates. The fields are filtered on the uniform `ms` mode grid,
`ū = type2(Ĝ · type1(u)/N)`, and `Π_ℓ = -τ̄ᵢⱼ S̄ᵢⱼ` is formed at the points.

`spectral` (required; the two libraries are peers) names the NUFFT library:
`FlowTransformBindings.NonuniformFFTsBackend()` (`using NonuniformFFTs`) or
`FlowTransformBindings.FINUFFTBackend()` (`using FINUFFT`). `Ls` (required) is the periodic domain size
per dimension. It sets the wavenumber grid `k = 2πn/L`, and hence the filter cutoff `Ĝ(|k|, ℓ)` and the
strain derivatives `i·kⱼ`: points in `[xₘᵢₙ, xₘᵢₙ+Lₐ)` under-span the period, so `L` is never inferred
from them.

`tol` is the transforms' relative accuracy; `return_diagnostics = true` also returns `τ̄ᵢⱼ` and `S̄ᵢⱼ`
at the points; `execution = ThreadedBackend()` threads each transform, for a lone call. For a
filter-scale sweep or a time series on one point set, build an [`NUFFTCoarseGrainingWorkspace`](@ref)
and call [`nufft_coarse_graining_flux!`](@ref).
"""
function nufft_coarse_graining_flux(velocity_fields, scatter_coords, ℓ, filter, ms;
                                    spectral::SpectralBackends.AbstractNonUniformFastFourierTransformSpectralBackend,
                                    Ls::Tuple, return_diagnostics::Bool = false, kwargs...)
    ws = NUFFTCoarseGrainingWorkspace(scatter_coords, ms; spectral = spectral, Ls = Ls,
                                      return_diagnostics = return_diagnostics, kwargs...)
    try
        return nufft_coarse_graining_flux!(ws, velocity_fields, ℓ, filter, ms;
                                           return_diagnostics = return_diagnostics)
    finally
        close!(ws)
    end
end

"""
    nufft_coarse_graining_flux_batch(velocity_fields_batch, scatter_coords, ℓ, filter, ms;
                                     spectral, Ls, execution, kwargs...) -> Vector

Scattered coarse-graining flux for a batch of snapshots on **one** point set. The NUFFT plans and every
buffer are built once for the whole batch, since for these libraries the plan is the dominant cost.
`execution = ThreadedBackend()` (requires `using OhMyThreads`) threads over snapshots with one
workspace per chunk. Results are in input order and each owns its flux field.
"""
function nufft_coarse_graining_flux_batch(velocity_fields_batch, scatter_coords, ℓ, filter, ms;
                                          spectral::SpectralBackends.AbstractNonUniformFastFourierTransformSpectralBackend,
                                          Ls::Tuple,
                                          execution::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.SerialBackend(),
                                          kwargs...)
    n = length(velocity_fields_batch)
    n == 0 && return Types.CoarseGrainingFluxResult[]
    return _nufft_cg_batch!(Types.resolve_execution(execution), velocity_fields_batch, scatter_coords,
                            ℓ, filter, ms; spectral = spectral, Ls = Ls, kwargs...)
end

# Serial reference: one workspace (plans + buffers) reused across the batch. `nufft_coarse_graining_flux!`
# wraps `ws.Π`, overwritten on the next snapshot, so each result is copied out.
function _nufft_cg_batch!(::ComputationalBackends.AbstractSerialBackend, batch, scatter_coords, ℓ, filter, ms;
                          spectral, Ls, return_diagnostics::Bool = false, kwargs...)
    ws = NUFFTCoarseGrainingWorkspace(scatter_coords, ms; spectral = spectral, Ls = Ls,
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
    throw(ArgumentError("nufft_coarse_graining_flux_batch supports SerialBackend() and ThreadedBackend(); " *
                        "got execution = $(typeof(be))."))

"""
    NUFFTCoarseGrainingWorkspace

Reusable resources for [`nufft_coarse_graining_flux!`](@ref): the NUFFT plans over the points (type-1
analysis, type-2 synthesis of the full spectrum), the spectral-side arrays (`|k|`, per-axis `kⱼ` grids,
filter weights `Ĝ`), and every working buffer of the flux, so a repeat call re-plans nothing and
allocates only the small result struct, which wraps the reused `Π`.

The plans hold library resources that [`close!`](@ref FlowInvariantTransfer.close!) releases; the
allocating entries close theirs before returning. One task at a time may use a workspace.
"""
struct NUFFTCoarseGrainingWorkspace{P1, P2, KM, KC, K1, SD, UM, TA, SA, SH, CI, CV, RV}
    p1::P1              # type-1 plan; real values where the library transforms them natively
    p2::P2              # type-2 plan of the full `FFTModes` spectrum (`p1` itself for a complex `p1`)
    k_mag::KM           # |k| over the coefficient grid
    k_comp_grids::KC    # per-axis kⱼ arrays, reshaped to broadcast along their own axis
    ks_1d::K1           # per-axis wavenumber vectors
    Ĝ::KM               # filter weights Ĝ(k) (recomputed per ℓ into this buffer)
    û_filt::SD          # (ms…, D) filtered spectral velocity (page c = component c)
    u_filt::UM          # (N, D) filtered velocity at the scattered points (real)
    τ::TA               # (N, D, D) SFS stress, or `nothing` when the workspace excludes diagnostics
    S̄::TA               # (N, D, D) strain rate, or `nothing`
    Π::RV               # (N,) flux (the result wraps this)
    spec::SA            # (ms…) full spectrum: analysis result, filtered product, gradient
    spec_half::SH       # `p1`'s output; `spec` itself for a complex `p1`
    scat_in::CI         # (N,) `p1`'s input
    scat_out::CV        # (N,) `p2`'s output
    prod_r::RV          # (N,) real product / ∂uᵢ∂xⱼ scratch
    grad_j::RV          # (N,) real ∂uⱼ∂xᵢ scratch
    tau_ij::RV          # (N,) stress component being contracted
    s_ij::RV            # (N,) strain component being contracted
    npoints::Int        # number of scattered points (type-1 normalization)
end
Base.show(io::IO, ::NUFFTCoarseGrainingWorkspace) = print(io, "NUFFTCoarseGrainingWorkspace(…)")
Base.show(io::IO, ::MIME"text/plain", w::NUFFTCoarseGrainingWorkspace) = show(io, w)

"""
    NUFFTCoarseGrainingWorkspace(scatter_coords, ms; spectral, Ls, tol = 1e-8,
                                 return_diagnostics = false, execution = SerialBackend())

Build the reusable coarse-graining workspace over `scatter_coords` for the NUFFT library `spectral`
names, with the periodic domain size `Ls`; see [`nufft_coarse_graining_flux`](@ref). `execution =
ThreadedBackend()` threads each transform.
"""
function NUFFTCoarseGrainingWorkspace(scatter_coords::Tuple, ms::Tuple;
                                      spectral::SpectralBackends.AbstractNonUniformFastFourierTransformSpectralBackend,
                                      Ls::Tuple, tol::Real = 1e-8, return_diagnostics::Bool = false,
                                      execution::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.SerialBackend())
    lib = _nufft_library(spectral)
    N  = _scattered_npoints(scatter_coords, ms, Ls)
    D  = length(scatter_coords)
    FT = float(eltype(scatter_coords[1]))
    CT = Complex{FT}

    ks_1d = Utils.wavenumber_grid(ms, ntuple(d -> FT(Ls[d]), D))   # FFTModes order
    k_mag = ShellBinning.shell_coordinate(Types.IsotropicShells(), ks_1d)
    k_comp_grids = SpectralLayout.wavenumber_arrays(zeros(FT, 0), FT, ks_1d)

    kw = _plan_keywords(scatter_coords, Ls, FT, tol, execution)
    p1 = _analysis_plan(lib, FT, scatter_coords, ms, kw)
    real_analysis = FlowTransformBindings.has_real_transform(lib)
    p2 = if real_analysis
        try
            FlowTransformBindings.plan_nufft(lib, CT, scatter_coords, ms;
                                             order = FlowTransformBindings.FFTModes(), kw...)
        catch
            FlowTransformBindings.close!(p1)
            rethrow()
        end
    else
        p1
    end

    spec = zeros(CT, ms...)
    return NUFFTCoarseGrainingWorkspace(
        p1, p2, k_mag, k_comp_grids, ks_1d, zeros(FT, ms...), zeros(CT, ms..., D), zeros(FT, N, D),
        return_diagnostics ? zeros(FT, N, D, D) : nothing, return_diagnostics ? zeros(FT, N, D, D) : nothing,
        zeros(FT, N), spec, real_analysis ? FlowTransformBindings.allocate_modes(p1) : spec,
        FlowTransformBindings.allocate_values(p1), zeros(CT, N),
        zeros(FT, N), zeros(FT, N), zeros(FT, N), zeros(FT, N), N)
end

"""
    close!(ws) -> ws

Release the NUFFT plans an [`NUFFTCoarseGrainingWorkspace`](@ref) or [`NUFFTToSpectralWorkspace`](@ref)
holds. Idempotent; a closed workspace refuses to execute.
"""
close!(ws::NUFFTCoarseGrainingWorkspace) =
    (FlowTransformBindings.close!(ws.p1); FlowTransformBindings.close!(ws.p2); ws)

# `ws.spec = type1(field)/N` over the full spectrum.
function _analyze!(ws::NUFFTCoarseGrainingWorkspace, field, ms, invN)
    ws.scat_in .= field
    FlowTransformBindings.nufft_type1!(ws.spec_half, ws.p1, ws.scat_in)
    return _to_full!(ComputationalBackends.SerialBackend(), ws.spec, ws.spec_half, ws.p1, ms, invN)
end

"""
    nufft_coarse_graining_flux!(ws::NUFFTCoarseGrainingWorkspace, velocity_fields, ℓ, filter, ms;
                                return_diagnostics = false) -> CoarseGrainingFluxResult

In-place scattered coarse-graining flux reusing `ws`: every transform runs through its plans and every
intermediate is a workspace buffer, so a repeat call allocates only the small result struct. That
result wraps `ws.Π`, and a later call overwrites it.
"""
function nufft_coarse_graining_flux!(ws::NUFFTCoarseGrainingWorkspace, velocity_fields::Tuple, ℓ::Real,
                                     filter::Types.AbstractFilter, ms::Tuple; return_diagnostics::Bool = false)
    D  = length(velocity_fields)
    nd = length(ws.ks_1d)
    D == nd || throw(ArgumentError("velocity components ($D) ≠ spatial dimensions ($nd)"))
    size(ws.spec) == ms || throw(ArgumentError("workspace spectral grid $(size(ws.spec)) ≠ ms $ms"))
    N  = ws.npoints
    FT = eltype(ws.Π)
    length(velocity_fields[1]) == N || throw(DimensionMismatch(
        "velocity field length $(length(velocity_fields[1])) ≠ workspace points $N"))
    return_diagnostics && ws.τ === nothing && throw(ArgumentError(
        "this NUFFTCoarseGrainingWorkspace was built with return_diagnostics = false; rebuild it " *
        "with return_diagnostics = true to get τ̄ᵢⱼ and S̄ᵢⱼ."))
    invN = one(FT) / FT(N)

    Ĝ = ws.Ĝ
    @inbounds for I in CartesianIndices(ws.k_mag)
        Ĝ[I] = FT(Filters.filter_response(filter, ws.k_mag[I], FT(ℓ)))
    end

    # Per component: û_filt = Ĝ·type1(u)/N, then the filtered velocity at the points.
    for c in 1:D
        _analyze!(ws, velocity_fields[c], ms, invN)
        ûfc = _page(ws.û_filt, c)
        @. ûfc = Ĝ * ws.spec
        FlowTransformBindings.nufft_type2!(ws.scat_out, ws.p2, ûfc)
        @views @. ws.u_filt[:, c] = real(ws.scat_out)
    end

    # Stress τ̄ᵢⱼ, strain S̄ᵢⱼ, and the flux Π = -Σ factor·τ·S̄, streamed pair by pair.
    fill!(ws.Π, 0)
    τij = ws.tau_ij; S̄ij = ws.s_ij
    @inbounds for i in 1:D, j in i:D
        @. ws.prod_r = velocity_fields[i] * velocity_fields[j]
        _analyze!(ws, ws.prod_r, ms, invN)
        @. ws.spec = Ĝ * ws.spec
        FlowTransformBindings.nufft_type2!(ws.scat_out, ws.p2, ws.spec)
        @views @. τij = real(ws.scat_out) - ws.u_filt[:, i] * ws.u_filt[:, j]

        # ∂ūᵢ/∂xⱼ = type2(i·kⱼ·û_filt_i). The page views are hoisted out of `@.`, which broadcasts
        # every call inside it.
        ûfi = _page(ws.û_filt, i)
        kj = ws.k_comp_grids[j]
        @. ws.spec = im * kj * ûfi
        FlowTransformBindings.nufft_type2!(ws.scat_out, ws.p2, ws.spec)
        @. ws.prod_r = real(ws.scat_out)
        if i == j
            S̄ij .= ws.prod_r
        else
            ûfj = _page(ws.û_filt, j)
            ki = ws.k_comp_grids[i]
            @. ws.spec = im * ki * ûfj
            FlowTransformBindings.nufft_type2!(ws.scat_out, ws.p2, ws.spec)
            @. ws.grad_j = real(ws.scat_out)
            @. S̄ij = FT(0.5) * (ws.prod_r + ws.grad_j)
        end

        factor = i == j ? FT(1) : FT(2)
        @. ws.Π -= factor * τij * S̄ij
        if ws.τ !== nothing
            @views ws.τ[:, i, j] .= τij
            @views ws.S̄[:, i, j] .= S̄ij
            if i != j
                @views ws.τ[:, j, i] .= τij
                @views ws.S̄[:, j, i] .= S̄ij
            end
        end
    end
    mean_Π = FT(sum(ws.Π) / N)

    if return_diagnostics
        return Types.CoarseGrainingFluxResultWithDiagnostics(FT(ℓ), ws.Π, mean_Π, ws.τ, ws.S̄)
    else
        return Types.CoarseGrainingFluxResult(FT(ℓ), ws.Π, mean_Π)
    end
end

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
struct NUFFTToSpectralWorkspace{P, KS, UH, CV, SP, R<:Real, EX}
    plan::P              # type-1 plan over the points
    ks::KS               # per-axis uniform wavenumber vectors (returned with û)
    û::UH                # (ms…, D) coefficient buffer (the result aliases this)
    scat::CV             # (N,) type-1 input
    spec::SP             # type-1 output
    npoints::Int         # number of scattered points (type-1 normalization)
    invN::R
    execution::EX        # backend the buffers live on; selects the host or device Hermitian expansion
end
Base.show(io::IO, ::NUFFTToSpectralWorkspace) = print(io, "NUFFTToSpectralWorkspace(…)")
Base.show(io::IO, ::MIME"text/plain", w::NUFFTToSpectralWorkspace) = show(io, w)

"""
    NUFFTToSpectralWorkspace(scatter_coords, ms; spectral, Ls, tol = 1e-9, ncomponents = D,
                             execution = SerialBackend())

Build the reusable scattered → uniform workspace over `scatter_coords` for the NUFFT library
`spectral` names (`FlowTransformBindings.NonuniformFFTsBackend()` or
`FlowTransformBindings.FINUFFTBackend()`), with the periodic domain size `Ls` (`k = 2πn/L`; the
samples do not determine it). `execution = GPUBackend(dev)` builds the plan and buffers on `dev`;
`ThreadedBackend()` threads each transform.
"""
function NUFFTToSpectralWorkspace(scatter_coords::Tuple, ms::Tuple;
                                  spectral::SpectralBackends.AbstractNonUniformFastFourierTransformSpectralBackend,
                                  Ls::Tuple, ncomponents::Int = length(scatter_coords), tol::Real = 1e-9,
                                  execution::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.SerialBackend())
    lib = _nufft_library(spectral)
    N = _scattered_npoints(scatter_coords, ms, Ls)
    ncomponents >= 1 || throw(ArgumentError("ncomponents must be ≥ 1"))
    FT = float(real(eltype(scatter_coords[1])))
    ks = Utils.wavenumber_grid(ms, ntuple(d -> FT(Ls[d]), length(ms)))   # FFTModes order
    nodes = map(x -> _nufft_to_device(execution, x), scatter_coords)
    plan = _analysis_plan(lib, FT, nodes, ms, _plan_keywords(scatter_coords, Ls, FT, tol, execution))
    û = similar(first(nodes), Complex{FT}, (ms..., ncomponents))
    return NUFFTToSpectralWorkspace(plan, ks, û, FlowTransformBindings.allocate_values(plan),
                                    FlowTransformBindings.allocate_modes(plan), N, one(FT) / FT(N), execution)
end

close!(ws::NUFFTToSpectralWorkspace) = (FlowTransformBindings.close!(ws.plan); ws)

"""
    to_spectral!(ws::NUFFTToSpectralWorkspace, velocity_fields) -> (velocity_hat, ks)

In-place scattered → uniform reconstruction reusing `ws`: `û = type1(u)/N` in FFTW mode order, returned
with the uniform wavenumber grid `ks`. The returned `velocity_hat` aliases `ws.û`, and a later call
overwrites it.
"""
function to_spectral!(ws::NUFFTToSpectralWorkspace, velocity_fields::Tuple)
    D = size(ws.û, ndims(ws.û))
    length(velocity_fields) == D || throw(DimensionMismatch(
        "to_spectral! got $(length(velocity_fields)) fields; workspace was built for $D components"))
    ms = Base.front(size(ws.û))
    for c in 1:D
        length(velocity_fields[c]) == ws.npoints ||
            throw(DimensionMismatch("field length ≠ workspace points $(ws.npoints)"))
        ws.scat .= velocity_fields[c]
        FlowTransformBindings.nufft_type1!(ws.spec, ws.plan, ws.scat)
        _to_full!(ws.execution, _page(ws.û, c), ws.spec, ws.plan, ms, ws.invN)
    end
    return (ws.û, ws.ks)
end

"""
    to_spectral(velocity_fields::Tuple, scatter_coords::Tuple, ms::Tuple; spectral, Ls, tol = 1e-9,
                execution = SerialBackend()) -> (velocity_hat, ks)

Scattered-Cartesian physical-space entry: reconstruct the Fourier coefficients `velocity_hat` on a
uniform `ms = (m₁,…,m_D)` grid from velocity components sampled at **scattered** (non-uniform) points,
through a NUFFT type-1 transform, and return them with the matching uniform wavenumber grid `ks`. The
result feeds the ordinary Cartesian flux diagnostics unchanged, so the whole flux family works on
scattered data through the uniform path.

`scatter_coords = (x, y[, z])` hold one coordinate per sample on each axis (each vector of length `N`,
the number of samples); `ms` is the target uniform mode count per dimension. `spectral` (required;
the two libraries are peers) names the NUFFT library: `FlowTransformBindings.NonuniformFFTsBackend()`
(`using NonuniformFFTs`) or `FlowTransformBindings.FINUFFTBackend()` (`using FINUFFT`). For
uniform-grid data use the 2-argument form with `SpectralBackends.FFTSpectralBackend()`.

The reconstruction is the density-normalized adjoint `û = type1(u)/N`, exact for samples on the uniform
grid. `tol` sets the NUFFT accuracy. `Ls` (required) is the periodic domain size per dimension and
fixes the wavenumbers `k = 2πn/L`: the samples live in `[xₘᵢₙ, xₘᵢₙ+Lₐ)` and under-span the period,
so `L` cannot be inferred from them. Samples on the uniform `L`-grid give `û = fft(u)/Nᵈ` exactly.
`execution = GPUBackend(dev)` (`dev` a KernelAbstractions backend) builds the plan and buffers on the
device, so the scattered → `velocity_hat` step runs there.
"""
function to_spectral(velocity_fields::Tuple, scatter_coords::Tuple, ms::Tuple;
                     spectral::SpectralBackends.AbstractSpectralBackend, tol::Real = 1e-9,
                     Ls::Tuple,
                     execution::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.SerialBackend())
    spectral isa SpectralBackends.AbstractNonUniformFastFourierTransformSpectralBackend || throw(ArgumentError(
        "the 3-argument to_spectral(fields, scatter_coords, ms; …) is the scattered-Cartesian NUFFT " *
        "entry: pass spectral = FlowTransformBindings.NonuniformFFTsBackend() or " *
        "FlowTransformBindings.FINUFFTBackend(). For uniform-grid data use " *
        "to_spectral(fields, coords_vecs; spectral = SpectralBackends.FFTSpectralBackend())."))
    ws = NUFFTToSpectralWorkspace(scatter_coords, ms; spectral = spectral,
                                  ncomponents = length(velocity_fields), tol = tol, Ls = Ls, execution = execution)
    try
        return to_spectral!(ws, velocity_fields)
    finally
        close!(ws)
    end
end
