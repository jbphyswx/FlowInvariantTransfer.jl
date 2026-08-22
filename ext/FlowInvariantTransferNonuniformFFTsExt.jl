module FlowInvariantTransferNonuniformFFTsExt

using NonuniformFFTs: NonuniformFFTs
using FlowInvariantTransfer: FlowInvariantTransfer as FIT
using ComputationalBackends: ComputationalBackends

# ---------------------------------------------------------------------------
# Scattered-Cartesian coarse-graining flux + scattered→uniform `to_spectral` via NonuniformFFTs.jl — the
# pure-Julia peer of the FINUFFT provider (`Types.NonuniformFFTsBackend` vs `Types.FINUFFTBackend`). The
# API (workspace builders, one-shots, `to_spectral!`, `calculate_energy_transfer`) dispatches on the
# backend type, so both providers coexist in one session; the in-place methods additionally dispatch on
# the plan type (`PlanNUFFT`). NonuniformFFTs uses ONE `PlanNUFFT` for both directions (`exec_type1!`
# nonuniform→uniform, `exec_type2!` uniform→nonuniform), stored in both `p1`/`p2` slots of the shared
# `NUFFTCoarseGrainingWorkspace`. The plan is a plain Julia object (GC'd), so no finalizer is needed.
# ---------------------------------------------------------------------------

@inline _page(A::AbstractArray, c::Int) = view(A, ntuple(_ -> Colon(), ndims(A) - 1)..., c)

_build_k_component(ks_1d, d::Int, ms::Tuple) = begin
    kc = zeros(eltype(ks_1d[d]), ms...)
    for I in CartesianIndices(ms)
        kc[I] = ks_1d[d][I[d]]
    end
    kc
end

# KaiserBessel half-support honoring the requested `tol` (σ = 2 default ⇒ ≈1e-7 at m = 4; each extra
# unit adds ~1.5 digits), capped so the oversampled grid fits the kernel — NonuniformFFTs requires
# σ·N ≥ 2m, i.e. m ≤ minimum(ms) at σ = 2.
function _kb_halfsupport(tol, ms::Tuple, ::Type{FT}) where {FT<:AbstractFloat}
    # A tol below the element eps requests an unachievable half-support → NaN spreading (e.g. tol=1e-8 on
    # Float32). Clamp to eps(FT); eps(Float64) is tiny, so this only bites the low-precision path.
    teff = max(float(tol), eps(FT))
    hi = min(16, minimum(ms))
    return clamp(ceil(Int, -log10(teff)) + 1, min(4, hi), hi)
end

# Provider builder dispatched from `NUFFTCoarseGrainingWorkspace(scatter_coords, ms;
# spectral = Types.NonuniformFFTsBackend(), …)` — builds the reusable `NUFFTCoarseGrainingWorkspace`
# backed by a NonuniformFFTs `PlanNUFFT` (points preset) plus every working buffer for
# `nufft_coarse_graining_flux!`. 1D/2D/3D scattered Cartesian points. NonuniformFFTs threads off
# `Threads.nthreads()` internally, so `execution` is accepted for API symmetry but does not gate here.
function FIT._nufft_cg_workspace(
    ::FIT.Types.NonuniformFFTsBackend, scatter_coords::Tuple, ms::Tuple; Ls::Tuple, tol::Real = 1e-8,
    execution::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.SerialBackend(),
)
    nd = length(scatter_coords)
    nd == length(ms) || throw(ArgumentError("scatter_coords ($(nd)D) and ms ($(length(ms))D) must match"))
    1 <= nd <= 3 || throw(ArgumentError("NonuniformFFTs coarse-graining supports 1D/2D/3D; got nd=$nd."))
    length(Ls) == nd || throw(ArgumentError("Ls ($(length(Ls))D) must match scatter_coords ($(nd)D)"))
    all(>(0), Ls) || throw(ArgumentError("Ls must be positive (periodic domain size per dimension); got $Ls"))
    N  = length(scatter_coords[1])
    FT = float(eltype(scatter_coords[1]))
    CT = Complex{FT}

    Lf = ntuple(d -> FT(Ls[d]), nd)              # periodic domain per dimension (k = 2πn/L)
    ks_1d = FIT.Utils.wavenumber_grid(ms, Lf)    # fftfreq/FFTW ordering, matching PlanNUFFT(fftshift=false)
    k_mag = FIT.Utils.wavenumber_magnitude_grid(ks_1d)
    k_comp_grids = ntuple(d -> _build_k_component(ks_1d, d, ms), nd)

    # Coordinates mapped to NonuniformFFTs' periodic cell [0, 2π) at the *physical* scale x ↦ 2π(x−xₘᵢₙ)/L.
    # `xₘᵢₙ` is a pure phase reference (the type-1→type-2 round-trip cancels it, Πℓ is shift-invariant); `L`
    # sets the scale, so samples on the uniform L-grid land on the DFT nodes and |k|/kⱼ carry physical units.
    scaled_coords = ntuple(nd) do d
        cmin = FT(minimum(scatter_coords[d]))
        (FT.(scatter_coords[d]) .- cmin) ./ Lf[d] .* (2 * FT(π))
    end

    D  = nd
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

    # One complex-data plan (fftshift=false → FFTW mode ordering, matching Utils.wavenumber_grid + Ĝ).
    # NonuniformFFTs threads its transforms off `Threads.nthreads()` internally (there is no per-plan
    # thread count), so `execution` is accepted for API symmetry with the FINUFFT backend but does not
    # gate threading here — batch the outer axis (a scale sweep / snapshot series) to parallelise.
    plan = NonuniformFFTs.PlanNUFFT(CT, ms; fftshift = false, m = NonuniformFFTs.HalfSupport(_kb_halfsupport(tol, ms, FT)))
    NonuniformFFTs.set_points!(plan, scaled_coords)

    return FIT.NUFFTCoarseGrainingWorkspace(
        plan, plan, scaled_coords, k_mag, k_comp_grids, ks_1d, Ĝ, û_filt, u_filt,
        τ, S̄, Π, spec, scat_in, scat_out, prod_r, grad_j, N, FT(tol))
end

"""
    nufft_coarse_graining_flux!(ws::NUFFTCoarseGrainingWorkspace{<:PlanNUFFT}, velocity_fields, ℓ, filter, ms;
                                return_diagnostics=false) -> CoarseGrainingFluxResult

In-place scattered coarse-graining flux reusing the NonuniformFFTs-backed `ws`. Every transform runs
through the preset `PlanNUFFT` and every intermediate is a reused buffer, so a repeat call allocates only
the small result struct (wrapping `ws.Π`, overwritten on the next call). Dispatches on the plan type, so
it never shadows the FINUFFT method.
"""
function FIT.nufft_coarse_graining_flux!(
    ws::FIT.NUFFTCoarseGrainingWorkspace{<:NonuniformFFTs.PlanNUFFT},
    velocity_fields::Tuple,
    ℓ::Real,
    filter::FIT.Types.AbstractFilter,
    ms::Tuple;
    return_diagnostics::Bool = false,
)
    D  = length(velocity_fields)
    nd = length(ws.scaled_coords)
    D == nd || throw(ArgumentError("velocity components ($D) ≠ spatial dimensions ($nd)"))
    size(ws.spec) == ms || throw(ArgumentError("workspace spectral grid $(size(ws.spec)) ≠ ms $ms"))
    N  = ws.npoints
    FT = eltype(ws.Π)
    length(velocity_fields[1]) == N || throw(DimensionMismatch(
        "velocity field length $(length(velocity_fields[1])) ≠ workspace points $N"))
    invN = one(FT) / FT(N)
    p    = ws.p1                      # the single NonuniformFFTs plan (p1 === p2)

    Ĝ = ws.Ĝ
    @inbounds for I in CartesianIndices(ws.k_mag)
        Ĝ[I] = FT(FIT.Filters.filter_response(filter, ws.k_mag[I], FT(ℓ)))
    end

    # Filtered velocity at the points: û_filt = Ĝ·(type-1 u)/N, then type-2 back.
    for c in 1:D
        ws.scat_in .= velocity_fields[c]
        NonuniformFFTs.exec_type1!(ws.spec, p, ws.scat_in)       # nonuniform → uniform
        ûfc = _page(ws.û_filt, c)
        @. ûfc = Ĝ * ws.spec * invN
        NonuniformFFTs.exec_type2!(ws.scat_out, p, ûfc)          # uniform → nonuniform
        @views @. ws.u_filt[:, c] = real(ws.scat_out)
    end

    # Stress τ̄ᵢⱼ, strain S̄ᵢⱼ, flux Π = −Σ factor·τ·S̄, streamed pair-by-pair.
    fill!(ws.Π, 0)
    @inbounds for i in 1:D, j in i:D
        @. ws.prod_r = velocity_fields[i] * velocity_fields[j]
        ws.scat_in .= ws.prod_r
        NonuniformFFTs.exec_type1!(ws.spec, p, ws.scat_in)
        @. ws.spec = Ĝ * ws.spec * invN
        NonuniformFFTs.exec_type2!(ws.scat_out, p, ws.spec)
        τij = view(ws.τ, :, i, j)
        @views @. τij = real(ws.scat_out) - ws.u_filt[:, i] * ws.u_filt[:, j]

        ûfi = _page(ws.û_filt, i)
        @. ws.spec = im * ws.k_comp_grids[j] * ûfi
        NonuniformFFTs.exec_type2!(ws.scat_out, p, ws.spec)
        @. ws.prod_r = real(ws.scat_out)                          # ∂ūᵢ/∂xⱼ
        S̄ij = view(ws.S̄, :, i, j)
        if i == j
            S̄ij .= ws.prod_r
        else
            ûfj = _page(ws.û_filt, j)
            @. ws.spec = im * ws.k_comp_grids[i] * ûfj
            NonuniformFFTs.exec_type2!(ws.scat_out, p, ws.spec)
            @. ws.grad_j = real(ws.scat_out)                      # ∂ūⱼ/∂xᵢ
            @. S̄ij = FT(0.5) * (ws.prod_r + ws.grad_j)
        end

        factor = i == j ? FT(1) : FT(2)
        @. ws.Π -= factor * τij * S̄ij
        if i != j
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
                               return_diagnostics=false, tol=1e-8, execution=SerialBackend())

Scattered coarse-graining flux `Π_ℓ(x)` via NonuniformFFTs.jl — builds a one-shot workspace and
delegates to the in-place form. `using NonuniformFFTs` selects this pure-Julia backend (in place of
`using FINUFFT`); results match FINUFFT to the NUFFT tolerance. For repeated evaluations on the same
points build the workspace once and call the in-place form.
"""
function FIT._nufft_coarse_graining_flux(
    ::FIT.Types.NonuniformFFTsBackend,
    velocity_fields::Tuple, scatter_coords::Tuple, ℓ::Real, filter::FIT.Types.AbstractFilter, ms::Tuple;
    Ls::Tuple, return_diagnostics::Bool = false, tol::Real = 1e-8,
    execution::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.SerialBackend(),
)
    ws = FIT._nufft_cg_workspace(FIT.Types.NonuniformFFTsBackend(), scatter_coords, ms; Ls = Ls, tol = tol, execution = execution)
    return FIT.nufft_coarse_graining_flux!(ws, velocity_fields, ℓ, filter, ms; return_diagnostics = return_diagnostics)
end

# Scattered-Cartesian physical → uniform Fourier coefficients (Types.NonuniformFFTsBackend `to_spectral`
# path) — the pure-Julia peer of the FINUFFT one. The workspace holds a preset PlanNUFFT + reusable
# buffers; `to_spectral!` reuses them (0 alloc on repeat). `û = type1(u)/N` in FFTW mode order
# (`fftshift = false`), `ks` from `Utils.wavenumber_grid`; a drop-in for the uniform Cartesian diagnostics.
function FIT._to_spectral_workspace(::FIT.Types.NonuniformFFTsBackend, scatter_coords::Tuple, ms::Tuple;
                                    ncomponents::Int = length(scatter_coords), tol::Real = 1e-9,
                                    Ls::Tuple,
                                    execution::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.SerialBackend())
    # `execution` selects host (default) vs device-resident: a GPUBackend builds the plan on its KA
    # backend with device points/buffers (the whole scattered → velocity_hat step runs on-device); any
    # other backend stays host. NonuniformFFTs threads its own transforms off `Threads.nthreads()`.
    nd = length(scatter_coords)
    nd == length(ms) || throw(ArgumentError("scatter_coords ($(nd)D) and ms ($(length(ms))D) must match"))
    1 <= nd <= 3 || throw(ArgumentError("NonuniformFFTs to_spectral supports 1D/2D/3D; got nd=$nd."))
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
    ks = FIT.Utils.wavenumber_grid(ms, Lused)   # fftfreq/FFTW ordering, matching fftshift=false below
    scaled = ntuple(nd) do d
        cmin = FT(minimum(scatter_coords[d]))
        (FT.(scatter_coords[d]) .- cmin) ./ Lused[d] .* (2 * FT(π))   # → [0, 2π)
    end
    # Device-generic: for a GPUBackend the points and the plan's data buffers are moved/allocated onto the
    # KA backend (helpers dispatch to the KernelAbstractions ext); for any other backend they stay host.
    scaled_dev = map(x -> FIT._nufft_to_device(execution, x), scaled)
    plan = NonuniformFFTs.PlanNUFFT(CT, ms; FIT._nufft_plan_backend_kw(execution)...,
                                    fftshift = false, m = NonuniformFFTs.HalfSupport(_kb_halfsupport(tol, ms, FT)))
    NonuniformFFTs.set_points!(plan, scaled_dev)   # pure-Julia plan — GC'd, no finalizer needed

    û    = FIT._nufft_new(execution, CT, ms..., ncomponents)
    scat = FIT._nufft_new(execution, CT, N)
    spec = FIT._nufft_new(execution, CT, ms...)
    return FIT.NUFFTToSpectralWorkspace(plan, scaled_dev, ks, û, scat, spec, N, one(FT) / FT(N))
end

function FIT.to_spectral!(ws::FIT.NUFFTToSpectralWorkspace{<:NonuniformFFTs.PlanNUFFT}, velocity_fields::Tuple)
    D = size(ws.û, ndims(ws.û))
    length(velocity_fields) == D || throw(DimensionMismatch(
        "to_spectral! got $(length(velocity_fields)) fields; workspace was built for $D components"))
    @inbounds for c in 1:D
        length(velocity_fields[c]) == ws.npoints || throw(DimensionMismatch("field length ≠ workspace points $(ws.npoints)"))
        ws.scat .= velocity_fields[c]
        NonuniformFFTs.exec_type1!(ws.spec, ws.plan, ws.scat)     # nonuniform → uniform, e^{-ikx}, reused buffers
        ûc = _page(ws.û, c)
        @. ûc = ws.spec * ws.invN                                 # density-normalized adjoint (fft convention)
    end
    return (ws.û, ws.ks)
end

end # module FlowInvariantTransferNonuniformFFTsExt
