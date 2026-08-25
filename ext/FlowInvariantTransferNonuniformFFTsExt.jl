module FlowInvariantTransferNonuniformFFTsExt

using NonuniformFFTs: NonuniformFFTs
using FlowInvariantTransfer: FlowInvariantTransfer as FIT
using ComputationalBackends: ComputationalBackends

# ---------------------------------------------------------------------------
# Scattered-Cartesian coarse-graining flux + scattered→uniform `to_spectral` via NonuniformFFTs.jl — the
# pure-Julia peer of the FINUFFT provider (`Types.NonuniformFFTsBackend` vs `Types.FINUFFTBackend`). The
# API (workspace builders, one-shots, `to_spectral!`, `calculate_energy_transfer`) dispatches on the
# backend type, so both providers coexist in one session; the in-place methods additionally dispatch on
# the plan type (`PlanNUFFT`). The velocity is real, so the analysis (`exec_type1!`, nonuniform→uniform)
# runs through a real-input (r2c) plan and `_r2c_full!` expands its non-redundant half to the full
# spectrum; the synthesis (`exec_type2!`, uniform→nonuniform) consumes a full complex spectrum and so uses
# a complex plan (the `p1`/`p2` slots of `NUFFTCoarseGrainingWorkspace`; `to_spectral` is analysis-only and
# needs just the real plan). Plans are plain Julia objects (GC'd), so no finalizer is needed.
# ---------------------------------------------------------------------------

@inline _page(A::AbstractArray, c::Int) = view(A, ntuple(_ -> Colon(), ndims(A) - 1)..., c)

# Build the full fftfreq-ordered `(ms…)` complex mode array from the non-redundant half a real (r2c)
# `PlanNUFFT` returns (`fk_half`: axis-1 modes `0…ms₁÷2` (rfftfreq), fftfreq on the rest), applying the
# ×invN normalization. A real field's transform is exactly Hermitian, `C[k]=conj(C[-k])`:
#   • axis-1 `k₁≥0`  → taken directly from `fk_half`.
#   • `k₁<0`, no even axis d≥2 at its Nyquist → Hermitian mirror `conj(fk_half[-k])`.
#   • `k₁<0` AND some even axis d≥2 at `k_d=-N_d/2` → the fold needs `C[-k₁,+N_d/2]`, and `+N_d/2` is not
#     an output mode. For scattered points `+N_d/2 ≠ -N_d/2`, so no output row supplies it — but it IS an
#     interior mode of the OVERSAMPLED spectrum the transform already computed (`plan.data.ûs`, size `Ñ`).
#     Read it there and apply the same deconvolution (`normfactor / ∏ ĝ_k`) the non-oversampled copy uses.
# Exact for even and odd sizes; scalar loop over the output grid (host / KA.CPU arrays).
function _r2c_full!(full::AbstractArray, fk_half::AbstractArray, plan, ms::NTuple{D, Int}, invN) where {D}
    us = plan.data.ûs[1]                                              # oversampled rfft-half spectrum
    RT = real(eltype(full))
    novs = ntuple(d -> d == 1 ? 2 * (size(us, 1) - 1) : size(us, d), D)   # oversampled full sizes Ñ
    gk = ntuple(d -> NonuniformFFTs.fourier_coefficients(plan.kernels[d]), D)
    normfactor = prod(ntuple(d -> 2 * RT(π) / novs[d], D))
    hi = ntuple(d -> (ms[d] - 1) ÷ 2, D)
    half = ntuple(d -> ms[d] ÷ 2, D)
    kof(i, d) = (i - 1 <= hi[d]) ? (i - 1) : (i - 1 - ms[d])          # fftfreq integer at output index i
    mir(i, d) = i == 1 ? 1 : ms[d] - i + 2                            # output index of -freq(i)
    ovs(f, Ñ) = f >= 0 ? f + 1 : Ñ + f + 1                            # oversampled fftfreq index
    gki(f, N) = f >= 0 ? f + 1 : N + f + 1                            # ĝ index, fftfreq (ĝ even ⇒ +N/2 ↦ −N/2 slot)
    @inbounds for I in CartesianIndices(ms)
        k1 = kof(I[1], 1)
        if k1 >= 0
            full[I] = fk_half[CartesianIndex(ntuple(d -> d == 1 ? k1 + 1 : Int(I[d]), D))] * invN
            continue
        end
        evenNyq = false
        for d in 2:D
            (iseven(ms[d]) && kof(I[d], d) == -half[d]) && (evenNyq = true; break)
        end
        if !evenNyq
            full[I] = conj(fk_half[CartesianIndex(ntuple(d -> d == 1 ? -k1 + 1 : mir(I[d], d), D))]) * invN
        else
            negk = ntuple(d -> -kof(I[d], d), D)
            ovsI = CartesianIndex(ntuple(d -> ovs(negk[d], novs[d]), D))
            β = normfactor / gk[1][negk[1] + 1]
            for d in 2:D
                β /= gk[d][gki(negk[d], ms[d])]
            end
            full[I] = conj(β * us[ovsI]) * invN
        end
    end
    return full
end

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

    # Two plans on the same points: the velocity is real, so the ANALYSIS (type-1) runs through a real
    # (r2c) plan — a real-to-complex FFT, ~2× faster, no real→complex widen — and `_r2c_full!` expands its
    # non-redundant half into the exact full fftfreq spectrum. The SYNTHESIS (type-2) consumes that full
    # complex spectrum (filtered field, gradients), so it uses a complex plan. Filter/gradient math stays
    # on the full grid. NonuniformFFTs threads off `Threads.nthreads()` internally, so `execution` is
    # accepted for API symmetry with FINUFFT but does not gate here — batch the outer axis to parallelise.
    hs = NonuniformFFTs.HalfSupport(_kb_halfsupport(tol, ms, FT))
    p1 = NonuniformFFTs.PlanNUFFT(FT, ms; fftshift = false, m = hs)   # real analysis (half-spectrum out)
    p2 = NonuniformFFTs.PlanNUFFT(CT, ms; fftshift = false, m = hs)   # complex synthesis (full spectrum in)
    NonuniformFFTs.set_points!(p1, scaled_coords)
    NonuniformFFTs.set_points!(p2, scaled_coords)

    D  = nd
    Ĝ      = zeros(FT, ms...)
    û_filt = zeros(CT, ms..., D)
    u_filt = zeros(FT, N, D)
    τ      = zeros(FT, N, D, D)
    S̄      = zeros(FT, N, D, D)
    Π      = zeros(FT, N)
    spec   = zeros(CT, ms...)
    spec_half = zeros(CT, size(p1)...)           # r2c analysis output (expanded into `spec`)
    scat_in  = zeros(FT, N)                      # real type-1 input (r2c analysis)
    scat_out = zeros(CT, N)                      # complex type-2 output
    prod_r = zeros(FT, N)
    grad_j = zeros(FT, N)

    return FIT.NUFFTCoarseGrainingWorkspace(
        p1, p2, scaled_coords, k_mag, k_comp_grids, ks_1d, Ĝ, û_filt, u_filt,
        τ, S̄, Π, spec, spec_half, scat_in, scat_out, prod_r, grad_j, N, FT(tol))
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
    p1 = ws.p1                        # real (r2c) analysis plan
    p2 = ws.p2                        # complex synthesis plan

    Ĝ = ws.Ĝ
    @inbounds for I in CartesianIndices(ws.k_mag)
        Ĝ[I] = FT(FIT.Filters.filter_response(filter, ws.k_mag[I], FT(ℓ)))
    end

    # Filtered velocity at the points: û_filt = Ĝ·(type-1 u)/N, then type-2 back. The type-1 runs on the
    # real plan (r2c) and `_r2c_full!` expands its half-spectrum to the exact full fftfreq spectrum.
    for c in 1:D
        ws.scat_in .= velocity_fields[c]
        NonuniformFFTs.exec_type1!(ws.spec_half, p1, ws.scat_in)         # nonuniform → uniform (real, half-spectrum)
        _r2c_full!(ws.spec, ws.spec_half, p1, ms, invN)                  # half → full complex modes (×1/N)
        ûfc = _page(ws.û_filt, c)
        @. ûfc = Ĝ * ws.spec
        NonuniformFFTs.exec_type2!(ws.scat_out, p2, ûfc)         # uniform → nonuniform
        @views @. ws.u_filt[:, c] = real(ws.scat_out)
    end

    # Stress τ̄ᵢⱼ, strain S̄ᵢⱼ, flux Π = −Σ factor·τ·S̄, streamed pair-by-pair.
    fill!(ws.Π, 0)
    @inbounds for i in 1:D, j in i:D
        @. ws.prod_r = velocity_fields[i] * velocity_fields[j]
        ws.scat_in .= ws.prod_r
        NonuniformFFTs.exec_type1!(ws.spec_half, p1, ws.scat_in)
        _r2c_full!(ws.spec, ws.spec_half, p1, ms, invN)
        @. ws.spec = Ĝ * ws.spec
        NonuniformFFTs.exec_type2!(ws.scat_out, p2, ws.spec)
        τij = view(ws.τ, :, i, j)
        @views @. τij = real(ws.scat_out) - ws.u_filt[:, i] * ws.u_filt[:, j]

        ûfi = _page(ws.û_filt, i)
        @. ws.spec = im * ws.k_comp_grids[j] * ûfi
        NonuniformFFTs.exec_type2!(ws.scat_out, p2, ws.spec)
        @. ws.prod_r = real(ws.scat_out)                          # ∂ūᵢ/∂xⱼ
        S̄ij = view(ws.S̄, :, i, j)
        if i == j
            S̄ij .= ws.prod_r
        else
            ûfj = _page(ws.û_filt, j)
            @. ws.spec = im * ws.k_comp_grids[i] * ûfj
            NonuniformFFTs.exec_type2!(ws.scat_out, p2, ws.spec)
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
    # Real (r2c) plan for the real velocity field: the FFTW real-to-complex transform is ~2× faster and
    # halves memory vs a complex plan, returning the non-redundant half spectrum (first axis ms₁÷2+1).
    plan = NonuniformFFTs.PlanNUFFT(FT, ms; FIT._nufft_plan_backend_kw(execution)...,
                                    fftshift = false, m = NonuniformFFTs.HalfSupport(_kb_halfsupport(tol, ms, FT)))
    NonuniformFFTs.set_points!(plan, scaled_dev)   # pure-Julia plan — GC'd, no finalizer needed

    mh   = size(plan)                                          # half-spectrum dims (first axis halved)
    û    = FIT._nufft_new(execution, CT, ms..., ncomponents)   # full complex modes (Hermitian-mirrored output)
    scat = FIT._nufft_new(execution, FT, N)                    # real type-1 input
    spec = FIT._nufft_new(execution, CT, mh...)                # half complex modes (r2c type-1 output)
    return FIT.NUFFTToSpectralWorkspace(plan, scaled_dev, ks, û, scat, spec, N, one(FT) / FT(N))
end

function FIT.to_spectral!(ws::FIT.NUFFTToSpectralWorkspace{<:NonuniformFFTs.PlanNUFFT}, velocity_fields::Tuple)
    D = size(ws.û, ndims(ws.û))
    length(velocity_fields) == D || throw(DimensionMismatch(
        "to_spectral! got $(length(velocity_fields)) fields; workspace was built for $D components"))
    ms = size(ws.û)[1:end-1]                                       # full mode dims (drop the component axis)
    @inbounds for c in 1:D
        length(velocity_fields[c]) == ws.npoints || throw(DimensionMismatch("field length ≠ workspace points $(ws.npoints)"))
        ws.scat .= velocity_fields[c]                             # real input
        NonuniformFFTs.exec_type1!(ws.spec, ws.plan, ws.scat)     # nonuniform → uniform half-spectrum, e^{-ikx}
        _r2c_full!(_page(ws.û, c), ws.spec, ws.plan, ms, ws.invN) # half → full complex modes (×1/N, Hermitian)
    end
    return (ws.û, ws.ks)
end

end # module FlowInvariantTransferNonuniformFFTsExt
