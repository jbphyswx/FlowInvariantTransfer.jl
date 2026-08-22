module FlowInvariantTransferFINUFFTExt

using FINUFFT: FINUFFT
using FlowInvariantTransfer: FlowInvariantTransfer as FIT
using ComputationalBackends: ComputationalBackends

# ---------------------------------------------------------------------------
# Non-uniform coarse-graining flux via FINUFFT type-1/type-2 round-trips.
#
# A single Π_ℓ evaluation issues ~4·D²+2·D type-1/type-2 transforms — all on the SAME scattered
# points, spectral-grid size, and tolerance. `NUFFTCoarseGrainingWorkspace` builds ONE type-1 and
# ONE type-2 guru plan (points set once) plus every working buffer, so a repeat call re-plans
# nothing and allocates nothing on the Julia side (`finufft_exec!` writes into the reused buffers).
# FINUFFT.jl registers no finalizer on its plans, so each C plan gets one attached to the (mutable) plan
# object itself when built — letting the workspaces stay immutable.
# ---------------------------------------------------------------------------

# View of the c-th component page of an (spatial…, D) array — contiguous (last-dim slice), so
# FINUFFT's `InputArray = Union{Array,SubArray}` + contiguity check accepts it.
@inline _page(A::AbstractArray, c::Int) = view(A, ntuple(_ -> Colon(), ndims(A) - 1)..., c)

# Provider builder dispatched from `NUFFTCoarseGrainingWorkspace(scatter_coords, ms;
# spectral = Types.FINUFFTBackend(), …)` — builds the reusable FINUFFT plans (type-1 analysis, type-2
# synthesis; points set) and every working buffer for `nufft_coarse_graining_flux!`. 1D/2D/3D scattered
# Cartesian points. `execution` selects the FINUFFT plan thread count: `SerialBackend()` (default) →
# single-threaded, 0 allocation per `finufft_exec!` (FINUFFT shares libfftw3 with FFTW.jl, so a
# multithreaded plan routes its FFT through FFTW's Julia-thread callback and allocates Task scratch per
# exec) — the right choice when the outer batch axis is parallelised one-worker-per-item.
# `ThreadedBackend()` threads a single (or under-saturated) transform across `Threads.nthreads()` cores.
function FIT._nufft_cg_workspace(::FIT.Types.FINUFFTBackend, scatter_coords::Tuple, ms::Tuple; Ls::Tuple, tol::Real = 1e-8,
                                 execution::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.SerialBackend())
    fft_nthreads = execution isa ComputationalBackends.ThreadedBackend ? Threads.nthreads() : 1
    nd = length(scatter_coords)
    nd == length(ms) || throw(ArgumentError("scatter_coords ($(nd)D) and ms ($(length(ms))D) must match"))
    1 <= nd <= 3 || throw(ArgumentError("FINUFFT supports 1D, 2D, 3D only; got nd=$nd."))
    length(Ls) == nd || throw(ArgumentError("Ls ($(length(Ls))D) must match scatter_coords ($(nd)D)"))
    all(>(0), Ls) || throw(ArgumentError("Ls must be positive (periodic domain size per dimension); got $Ls"))
    N  = length(scatter_coords[1])
    FT = float(eltype(scatter_coords[1]))
    CT = Complex{FT}
    D  = nd    # velocity has one component per spatial dimension

    Lf = ntuple(d -> FT(Ls[d]), nd)              # periodic domain per dimension (k = 2πn/L)
    ks_1d = FIT.Utils.wavenumber_grid(ms, Lf)    # fftfreq/FFTW ordering, matching the modeord=1 plans below
    k_mag = FIT.Utils.wavenumber_magnitude_grid(ks_1d)
    k_comp_grids = ntuple(d -> _build_k_component_nufft(ks_1d, d, ms), nd)

    # Coordinates mapped to [-π, π) at the *physical* scale x ↦ 2π(x−xₘᵢₙ)/L − π. `xₘᵢₙ` is a pure phase
    # reference (the type-1→type-2 round-trip cancels it and Πℓ is shift-invariant); `L` sets the scale,
    # so samples on the uniform L-grid land on the DFT nodes and |k|/kⱼ carry physical units.
    scaled_coords = ntuple(nd) do d
        cmin = FT(minimum(scatter_coords[d]))
        (FT.(scatter_coords[d]) .- cmin) ./ Lf[d] .* (2 * FT(π)) .- FT(π)
    end

    # Working buffers (all reused across calls).
    Ĝ      = zeros(FT, ms...)
    û_filt = zeros(CT, ms..., D)
    u_filt = zeros(FT, N, D)
    τ      = zeros(FT, N, D, D)
    S̄      = zeros(FT, N, D, D)
    Π      = zeros(FT, N)
    spec   = zeros(CT, ms...)
    scat_in  = zeros(CT, N)
    scat_out = zeros(CT, N)
    prod_r = zeros(FT, N)
    grad_j = zeros(FT, N)

    nmodes = Int64[ms...]
    # type-1 is the ANALYSIS transform, so it must use e^{-ikx} (iflag = -1) to match the fft/DFT
    # convention — the ū = type2(Ĝ·type1(v)/N) round-trip is then a convolution filter, not a correlation.
    # modeord = 1 → FFTW mode ordering (0,1,…,N/2-1,-N/2,…,-1), matching `ks_1d`/`Ĝ`; the default (CMCL,
    # -N/2…N/2-1) would scramble the filter/derivative against the k-grid.
    p1 = FINUFFT.finufft_makeplan(1, nmodes, -1, 1, FT(tol); dtype = FT, nthreads = fft_nthreads, modeord = 1)   # nonuniform → uniform, e^{-ikx}
    p2 = FINUFFT.finufft_makeplan(2, nmodes,  1, 1, FT(tol); dtype = FT, nthreads = fft_nthreads, modeord = 1)   # uniform → nonuniform, e^{+ikx}
    FINUFFT.finufft_setpts!(p1, scaled_coords...)
    FINUFFT.finufft_setpts!(p2, scaled_coords...)
    # FINUFFT.jl registers no finalizer on its plans, so free each C plan when it is GC'd. Attaching the
    # finalizer to the (mutable) plan object — not the workspace — lets the workspace stay immutable.
    finalizer(FINUFFT.finufft_destroy!, p1)
    finalizer(FINUFFT.finufft_destroy!, p2)

    return FIT.NUFFTCoarseGrainingWorkspace(
        p1, p2, scaled_coords, k_mag, k_comp_grids, ks_1d, Ĝ, û_filt, u_filt,
        τ, S̄, Π, spec, scat_in, scat_out, prod_r, grad_j, N, FT(tol))
end

"""
    nufft_coarse_graining_flux!(ws::NUFFTCoarseGrainingWorkspace, velocity_fields, ℓ, filter, ms;
                                return_diagnostics=false) -> CoarseGrainingFluxResult

In-place scattered coarse-graining flux reusing `ws` — every transform runs through the preset
FINUFFT plans via `finufft_exec!`, and every intermediate is a reused workspace buffer, so a repeat
call allocates only the tiny result struct (which wraps `ws.Π`; a later call overwrites it). Intended
for a filter-scale sweep `Π(ℓ)` or a snapshot time series on the same points.
"""
function FIT.nufft_coarse_graining_flux!(
    ws::FIT.NUFFTCoarseGrainingWorkspace,
    velocity_fields::Tuple,
    ℓ::Real,
    filter::FIT.Types.AbstractFilter,
    ms::Tuple;
    return_diagnostics::Bool = false,
)
    D  = length(velocity_fields)
    nd = length(ws.scaled_coords)
    D == nd || throw(ArgumentError("velocity components ($D) ≠ spatial dimensions ($nd)"))
    size(ws.u_filt, 2) == D || throw(ArgumentError(
        "workspace was built for $(size(ws.u_filt, 2)) components, got $D"))
    size(ws.spec) == ms || throw(ArgumentError(
        "workspace spectral grid $(size(ws.spec)) ≠ ms $ms"))
    N  = ws.npoints
    FT = eltype(ws.Π)
    length(velocity_fields[1]) == N || throw(DimensionMismatch(
        "velocity field length $(length(velocity_fields[1])) ≠ workspace points $N"))
    invN = one(FT) / FT(N)

    # Filter weights Ĝ(k) for this scale.
    Ĝ = ws.Ĝ
    @inbounds for I in CartesianIndices(ws.k_mag)
        Ĝ[I] = FT(FIT.Filters.filter_response(filter, ws.k_mag[I], FT(ℓ)))
    end

    # Per component: û_filt = Ĝ·(type-1 u)/N (spectral), then filtered velocity at the points (type-2).
    for c in 1:D
        ws.scat_in .= velocity_fields[c]                         # real → complex, no temp
        FINUFFT.finufft_exec!(ws.p1, ws.scat_in, ws.spec)        # type-1: nonuniform → uniform
        ûfc = _page(ws.û_filt, c)
        @. ûfc = Ĝ * ws.spec * invN
        FINUFFT.finufft_exec!(ws.p2, ûfc, ws.scat_out)           # type-2: uniform → nonuniform
        @views @. ws.u_filt[:, c] = real(ws.scat_out)
    end

    # Stress τ̄ᵢⱼ, strain S̄ᵢⱼ, and the flux contraction Π = −Σ factor·τ·S̄, streamed pair-by-pair.
    fill!(ws.Π, 0)
    @inbounds for i in 1:D, j in i:D
        # Filtered product [uᵢuⱼ]̄ at the points.
        @. ws.prod_r = velocity_fields[i] * velocity_fields[j]
        ws.scat_in .= ws.prod_r
        FINUFFT.finufft_exec!(ws.p1, ws.scat_in, ws.spec)
        @. ws.spec = Ĝ * ws.spec * invN
        FINUFFT.finufft_exec!(ws.p2, ws.spec, ws.scat_out)
        τij = view(ws.τ, :, i, j)
        @views @. τij = real(ws.scat_out) - ws.u_filt[:, i] * ws.u_filt[:, j]

        # Strain: ∂ūᵢ/∂xⱼ = type-2(i·kⱼ·û_filt_i).  (page views hoisted out of `@.`, which would
        # otherwise broadcast the `_page` call itself over the array elements.)
        ûfi = _page(ws.û_filt, i)
        @. ws.spec = im * ws.k_comp_grids[j] * ûfi
        FINUFFT.finufft_exec!(ws.p2, ws.spec, ws.scat_out)
        @. ws.prod_r = real(ws.scat_out)                          # ∂ūᵢ/∂xⱼ (reuse prod_r)
        S̄ij = view(ws.S̄, :, i, j)
        if i == j
            S̄ij .= ws.prod_r
        else
            ûfj = _page(ws.û_filt, j)
            @. ws.spec = im * ws.k_comp_grids[i] * ûfj
            FINUFFT.finufft_exec!(ws.p2, ws.spec, ws.scat_out)
            @. ws.grad_j = real(ws.scat_out)                      # ∂ūⱼ/∂xᵢ
            @. S̄ij = FT(0.5) * (ws.prod_r + ws.grad_j)
        end

        factor = i == j ? FT(1) : FT(2)
        @. ws.Π -= factor * τij * S̄ij
        if i != j                                                  # mirror the symmetric entries
            @views ws.τ[:, j, i] .= τij
            @views ws.S̄[:, j, i] .= S̄ij
        end
    end
    mean_Π = FT(sum(ws.Π) / N)

    if return_diagnostics
        return FIT.Types.CoarseGrainingFluxResultWithDiagnostics(FT(ℓ), ws.Π, mean_Π, ws.τ, ws.S̄)
    else
        return FIT.Types.CoarseGrainingFluxResult(FT(ℓ), ws.Π, mean_Π)
    end
end

"""
    nufft_coarse_graining_flux(velocity_fields, scatter_coords, ℓ, filter, ms;
                               return_diagnostics=false, tol=1e-8) -> CoarseGrainingFluxResult

Coarse-graining energy flux `Π_ℓ(x)` at scattered (non-uniform) Cartesian points via FINUFFT.

Builds a one-shot [`NUFFTCoarseGrainingWorkspace`](@ref) and delegates to
[`nufft_coarse_graining_flux!`](@ref). For repeated evaluations on the same points (a filter-scale
sweep, or a snapshot time series), build the workspace once and call the in-place form so the plans
and all intermediate buffers are reused.

# Arguments
- `velocity_fields`: Tuple of D real vectors of length N — velocity at the scattered points.
- `scatter_coords`: Tuple of D real vectors of length N — the point coordinates.
- `ℓ::Real`: Filter scale.  `filter::AbstractFilter`: filter kernel.
- `ms::NTuple{D,Int}`: intermediate uniform spectral-grid size.

# Keyword Arguments
- `return_diagnostics::Bool=false`: also return τ̄ᵢⱼ and S̄ᵢⱼ at the points.
- `tol=1e-8`: FINUFFT accuracy tolerance.
- `execution=ComputationalBackends.SerialBackend()`: `ComputationalBackends.ThreadedBackend()` threads the FINUFFT transforms across cores (for a
  lone/under-saturated call); `ComputationalBackends.SerialBackend()` keeps them single-threaded (batch the outer axis instead).
"""
function FIT._nufft_coarse_graining_flux(
    ::FIT.Types.FINUFFTBackend,
    velocity_fields::Tuple,
    scatter_coords::Tuple,
    ℓ::Real,
    filter::FIT.Types.AbstractFilter,
    ms::Tuple;
    Ls::Tuple,
    return_diagnostics::Bool = false,
    tol::Real = 1e-8,
    execution::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.SerialBackend(),
)
    ws = FIT._nufft_cg_workspace(FIT.Types.FINUFFTBackend(), scatter_coords, ms; Ls = Ls, tol = tol, execution = execution)
    return FIT.nufft_coarse_graining_flux!(
        ws, velocity_fields, ℓ, filter, ms; return_diagnostics = return_diagnostics)
end

# ---------------------------------------------------------------------------
# NUFFT helpers
# ---------------------------------------------------------------------------

function _build_k_component_nufft(ks_1d, d::Int, ms::Tuple)
    kc = zeros(eltype(ks_1d[d]), ms...)
    for I in CartesianIndices(ms)
        kc[I] = ks_1d[d][I[d]]
    end
    return kc
end

# Plan + buffers for `to_spectral`, dispatched on the execution backend so the provider stays the same
# whether the transform runs on the host (this method) or an NVIDIA GPU (the cuFINUFFT extension). Host
# build: a type-1 guru plan (points preset) + reused Julia buffers. iflag=−1 → e^{-ik·x} analysis (Julia
# `fft` convention); modeord=1 → FFTW mode order, matching Utils.wavenumber_grid. SerialBackend()
# (fft_nthreads = 1) keeps the exec off FFTW's Julia-thread callback → 0-alloc reuse (batch the outer axis
# to parallelise); ThreadedBackend() threads one transform across `Threads.nthreads()` at the cost of
# per-exec Task scratch. Returns (plan, coords, û, scat, spec).
function FIT._finufft_ts_build(execution::ComputationalBackends.AbstractExecutionBackend, ms::Tuple, tol::Real,
                               scaled::Tuple, ::Type{CT}, ncomponents::Int, N::Int) where {CT}
    FT = real(CT)
    fft_nthreads = execution isa ComputationalBackends.ThreadedBackend ? Threads.nthreads() : 1
    p1 = FINUFFT.finufft_makeplan(1, Int64[ms...], -1, 1, FT(tol); dtype = FT, modeord = 1, nthreads = fft_nthreads)
    FINUFFT.finufft_setpts!(p1, scaled...)
    finalizer(FINUFFT.finufft_destroy!, p1)   # FINUFFT.jl registers none; free the C plan when GC'd
    û    = Array{CT}(undef, ms..., ncomponents)
    scat = Vector{CT}(undef, N)
    spec = Array{CT}(undef, ms...)
    return (p1, scaled, û, scat, spec)
end

# A GPUBackend routes to the cuFINUFFT device build (FINUFFT + CUDA extension); without `using CUDA` that
# override is absent, so fail loudly rather than silently building a host plan for a device request.
FIT._finufft_ts_build(::ComputationalBackends.AbstractGPUBackend, ms::Tuple, tol::Real, scaled::Tuple,
                      ::Type{CT}, ncomponents::Int, N::Int) where {CT} = throw(ArgumentError(
    "device FINUFFT to_spectral needs the cuFINUFFT extension — load it with `using CUDA` (NVIDIA GPU only)"))

# ---------------------------------------------------------------------------
# Scattered-Cartesian physical → uniform Fourier coefficients (the Types.FINUFFTBackend `to_spectral`
# path). The workspace holds a preset type-1 plan + reusable buffers; `to_spectral!` reuses them so a
# repeat call allocates nothing. Reconstruction is the density-normalized adjoint û = type1(u)/N (exact
# for samples on the uniform grid); ks is the fftfreq/FFTW wavenumber grid, so the result feeds the
# uniform flux diagnostics unchanged.
# ---------------------------------------------------------------------------
function FIT._to_spectral_workspace(::FIT.Types.FINUFFTBackend, scatter_coords::Tuple, ms::Tuple;
                                    ncomponents::Int = length(scatter_coords), tol::Real = 1e-9,
                                    Ls::Tuple,
                                    execution::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.SerialBackend())
    nd = length(scatter_coords)
    nd == length(ms) || throw(ArgumentError("scatter_coords ($(nd)D) and ms ($(length(ms))D) must match"))
    1 <= nd <= 3 || throw(ArgumentError("FINUFFT supports 1D, 2D, 3D only; got nd=$nd."))
    length(Ls) == nd || throw(ArgumentError("Ls ($(length(Ls))D) must match scatter_coords ($(nd)D)"))
    all(>(0), Ls) || throw(ArgumentError("Ls must be positive (periodic domain size per dimension); got $Ls"))
    ncomponents >= 1 || throw(ArgumentError("ncomponents must be ≥ 1"))
    N = length(scatter_coords[1])
    for d in 2:nd
        length(scatter_coords[d]) == N || throw(DimensionMismatch("all scatter_coords must have equal length"))
    end
    FT = float(real(eltype(scatter_coords[1])))
    CT = Complex{FT}

    # Periodic domain size per dimension (k = 2πn/L). Required — the samples under-span the period, so L
    # is a caller input, not inferred.
    Lused = ntuple(d -> FT(Ls[d]), nd)
    ks = FIT.Utils.wavenumber_grid(ms, Lused)   # fftfreq/FFTW ordering, matching the modeord=1 plan below
    # Samples mapped to [0, 2π) about each dimension's minimum: for samples on a uniform grid this lands
    # exactly on the DFT node locations, so `û = type1(u)/N` (iflag=−1) equals `fft(u)/Nᵈ`.
    scaled = ntuple(nd) do d
        cmin = FT(minimum(scatter_coords[d]))
        (FT.(scatter_coords[d]) .- cmin) ./ Lused[d] .* (2 * FT(π))
    end
    (plan, coords, û, scat, spec) = FIT._finufft_ts_build(execution, ms, tol, scaled, CT, ncomponents, N)
    return FIT.NUFFTToSpectralWorkspace(plan, coords, ks, û, scat, spec, N, one(FT) / FT(N))
end

function FIT.to_spectral!(ws::FIT.NUFFTToSpectralWorkspace{<:FINUFFT.finufft_plan}, velocity_fields::Tuple)
    D = size(ws.û, ndims(ws.û))
    length(velocity_fields) == D || throw(DimensionMismatch(
        "to_spectral! got $(length(velocity_fields)) fields; workspace was built for $D components"))
    colons = ntuple(_ -> Colon(), ndims(ws.û) - 1)
    @inbounds for c in 1:D
        length(velocity_fields[c]) == ws.npoints || throw(DimensionMismatch("field length ≠ workspace points $(ws.npoints)"))
        ws.scat .= velocity_fields[c]
        FINUFFT.finufft_exec!(ws.plan, ws.scat, ws.spec)   # nonuniform → uniform, into reused buffers
        ûc = view(ws.û, colons..., c)
        @. ûc = ws.spec * ws.invN
    end
    return (ws.û, ws.ks)
end

end # module FlowInvariantTransferFINUFFTExt
