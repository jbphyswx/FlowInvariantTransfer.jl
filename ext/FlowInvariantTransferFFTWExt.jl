module FlowInvariantTransferFFTWExt

using FFTW: FFTW
using FlowInvariantTransfer: FlowInvariantTransfer as FIT
using SpectralBackends: SpectralBackends
using LinearAlgebra: LinearAlgebra

# With this extension loaded, `AutoSpectralBackend` resolves to the FFT path.
function __init__()
    FIT.Types._FFT_BACKEND_LOADED[] = true
    return nothing
end

# ---------------------------------------------------------------------------
# FFT plan/scratch bundle + allocation-free nonlinear term
# ---------------------------------------------------------------------------

# FFTW plan creation is NOT thread-safe and the threaded backend builds a workspace per task,
# so serialize planning behind a lock.
const _PLAN_LOCK = ReentrantLock()

# Build forward/backward FFT plans for `x` with an EXPLICIT FFTW thread count (FFTW bakes the count into
# the plan at creation, so execution uses exactly this many threads regardless of the process-global
# `FFTW.set_num_threads(...)` a sibling like FlowFieldSpectra/NUFSHT may have set). Performance model:
# we own threading and saturate the coarsest parallel axis, never nesting. So `nthreads = 1` is the
# default — correct and 0-alloc for the loop-heavy methods (shell/mode/band/channel loop threaded, one
# single-threaded FFT per task) and for the serial path; the single-FFT-dominant methods (spectral flux,
# compressible) pass `nthreads > 1` when their outer loop is narrower than the core count, so the FFT
# itself soaks up the remaining cores (its per-transform scratch is negligible/amortized there). The
# global count is saved/restored under the lock so we don't perturb other packages' transforms.
function _plan_fft_bfft(x; nthreads::Int = 1)
    lock(_PLAN_LOCK) do
        old_nt = FFTW.get_num_threads()
        FFTW.set_num_threads(nthreads)
        try
            return (FFTW.plan_fft(x), FFTW.plan_bfft(x))
        finally
            FFTW.set_num_threads(old_nt)
        end
    end
end

# Reusable scratch + preplanned transforms on the padded (≈3N/2) grid, built at workspace
# construction ONLY when PaddedThreeHalves is requested (so the common 2/3 path never allocates the
# larger buffers). Fully parametric — buffers follow the same `(Ms..., C)` layout as
# `NonlinearTermWorkspace`, and the forward normalization is computed inline in the field eltype.
# One type per spectral layout, matching the two plan sets below.
struct ComplexPaddedScratch{PF, PB, CA, RA, GA, EM}
    p_fft::PF      # unnormalized forward c2c plan on the padded grid
    p_bfft::PB     # unnormalized backward c2c plan on the padded grid
    spec::CA       # complex (Msp...) scratch: embed target / transform input
    out::CA        # complex (Msp...) scratch: transform output
    u_phys::RA     # real (Ms..., nd) advecting-velocity physical buffer
    g_phys::GA     # real (Ms...) gradient physical scratch (reused per component/direction)
    n_phys::RA     # real (Ms..., M) nonlinear-term physical buffer
    emap::EM       # length-prod(ms) Vector{CartesianIndex}: ms→Msp index permutation, built once so
                   # embed/truncate are O(1)-lookup zero-pads (no fftshift allocations)
end

struct RealPaddedScratch{PF, PB, CA, RA, GA, EM}
    p_rfft::PF     # real (Ms...) → complex (Msp...)
    p_brfft::PB    # complex (Msp...) → real (Ms...), unnormalized
    spec::CA       # complex (Msp...) scratch
    u_phys::RA
    g_phys::GA
    n_phys::RA
    emap::EM
end

const AnyPaddedScratch = Union{ComplexPaddedScratch, RealPaddedScratch}

# What `NonlinearTermWorkspace.plans` holds: the pre-planned FFTW transforms, the scratch buffers
# they read and write, and the wavenumber / dealias factors the hot path multiplies by — so
# `_nonlinear_term_fft!` allocates nothing. Which of the two applies is set by the field's spectral
# layout, and the hot path dispatches on the type.
#
# `ComplexFFTPlans` — a field stored on the full fftfreq grid. Both legs are c2c on `(ms...)`; the
# backward transform writes to its own output buffer and the physical value is the real part.
#
# `RealFFTPlans` — a real field stored on the non-redundant half. The backward leg is a c2r
# (`brfft`) writing a real array directly and the forward leg an r2c (`rfft`): half the spectral
# memory, no real↔complex widening, no conjugate half to carry. FFTW's multi-dimensional c2r
# consumes its input, so `spec` holds no live value across a backward transform.
#
# Both keep the wavenumber components and the dealias keep-mask as `nd` per-axis arrays reshaped to
# broadcast along their own axis (`SpectralLayout.wavenumber_arrays` / `dealias_factors`): the hot
# broadcasts fuse against `Σ_d m_d` stored numbers.
struct ComplexFFTPlans{PF, PB, CA, KC, KM, PS}
    p_fft::PF      # unnormalized forward c2c plan on a single (ms...) component
    p_bfft::PB     # unnormalized backward c2c plan on a single (ms...) component
    spec::CA       # complex (ms...) scratch
    spec2::CA      # complex (ms...) scratch
    kg::KC         # nd wavenumber arrays, reshaped to broadcast along their own axis
    keep::KM       # nd dealias keep factors, likewise reshaped
    pad::PS        # PaddedScratch when built for FIT.Types.PaddedThreeHalves, else `nothing` — concrete either
                   # way, so `plans.pad` access is type-stable (no dynamic dispatch in the hot path).
end

struct RealFFTPlans{PF, PB, CA, KC, KM, PS}
    p_rfft::PF     # real (ns...) → complex (ms...)
    p_brfft::PB    # complex (ms...) → real (ns...), unnormalized; consumes its input
    spec::CA       # complex (ms...) scratch
    kg::KC
    keep::KM
    pad::PS
end

const AnyFFTPlans = Union{ComplexFFTPlans, RealFFTPlans}

# Custom one-line `show`: NEVER recurse into a live FFTW/cuFFT plan — the default field-by-field show
# calls `show(::IO, ::FFTW.Plan)` → the C `fftw_sprint_plan`, which can SEGFAULT on a fragile/closed
# plan (see the plan-show hazard). These make displaying any workspace that holds the bundle safe.
Base.show(io::IO, p::AnyPaddedScratch) = print(io, nameof(typeof(p)), "(…)")
Base.show(io::IO, ::MIME"text/plain", p::AnyPaddedScratch) = show(io, p)
Base.show(io::IO, b::AnyFFTPlans) =
    print(io, nameof(typeof(b)), "(", b.pad === nothing ? "2/3 dealias" : "3/2 padded", ")")
Base.show(io::IO, ::MIME"text/plain", b::AnyFFTPlans) = show(io, b)

# More specific than the core fallback `_make_fft_plans(::Any, ::Any, ::Any) = nothing`, so this ADDS
# a method (no overwriting) — dispatched only for complex spectral fields when FFTW is loaded. The
# `dealiasing` is known at workspace construction, so the (larger) padded scratch is built here ONLY
# for PaddedThreeHalves and its concrete type flows into the plan set's `pad` — keeping every
# `plans.pad` access type-stable with no allocation on the common 2/3 path.
function FIT.Workspaces._make_fft_plans(velocity_hat::AbstractArray{<:Complex}, ks, dealiasing, fft_nthreads::Int)
    nd = length(ks)
    ns = FIT.SpectralLayout.full_size(ks)        # physical grid
    ms = FIT.SpectralLayout.spectral_size(ks)    # coefficient grid
    FT = real(eltype(velocity_hat))
    twothirds = dealiasing isa FIT.Types.OrszagTwoThirds
    # `derivative = true`: axis d's array carries 0 at ITS OWN Nyquist mode, where the grid
    # derivative along d vanishes. A mode at Nyquist on some OTHER axis keeps a well-defined (and
    # Hermitian) derivative along d, so the zero belongs in the wavenumber, not in a product mask.
    kg   = FIT.SpectralLayout.wavenumber_arrays(velocity_hat, FT, ks; derivative = true)
    keep = FIT.SpectralLayout.dealias_factors(velocity_hat, FT, ks, twothirds)
    pad  = dealiasing isa FIT.Types.PaddedThreeHalves ? _make_padded_scratch(velocity_hat, ks, fft_nthreads) : nothing

    if FIT.SpectralLayout.is_half(ks)
        rp = try
            _plan_r2c_c2r(velocity_hat, FT, ns, ms; nthreads = fft_nthreads)
        catch err
            err isa MethodError || rethrow()
            nothing
        end
        rp === nothing && return nothing
        p_rfft, p_brfft = rp
        return RealFFTPlans(p_rfft, p_brfft, similar(velocity_hat, ms...), kg, keep, pad)
    end

    ct  = similar(velocity_hat, ms...)   # complex (ms...) — device-generic
    ct2 = similar(velocity_hat, ms...)
    # `AbstractFFTs.plan_fft` dispatches by array type (FFTW for host `Array`s, cuFFT for `CuArray`s,
    # …), so this one engine runs on CPU and device. ESTIMATE planning does not overwrite the array;
    # serialize because FFTW planning is not thread-safe (the threaded backend builds a ws per task).
    # `_plan_fft_bfft` dispatches BY ARRAY TYPE (FFTW imports+extends `AbstractFFTs.plan_fft`): a host
    # `Array` → FFTW plan, a `CuArray` (CUDA loaded) → cuFFT plan — one device-generic engine, built
    # single-threaded. If the array type has no registered FFT provider (e.g. a JLArray test surrogate),
    # planning throws a `MethodError`: catch it and return `nothing` so workspace construction still
    # succeeds; the FFT compute path then errors clearly (never silently) if `SpectralBackends.FFTSpectralBackend` is requested.
    plans = try
        _plan_fft_bfft(ct; nthreads = fft_nthreads)
    catch err
        err isa MethodError || rethrow()
        nothing
    end
    plans === nothing && return nothing
    p_fft, p_bfft = plans
    return ComplexFFTPlans(p_fft, p_bfft, ct, ct2, kg, keep, pad)
end

# Real-to-complex pair for the half layout: `rfft` on a real `(ns...)` component and the
# unnormalized inverse `brfft` back. Planned under the same lock and explicit thread count as the
# c2c pair; `plan_brfft` is planned on a scratch copy because FFTW's multi-dimensional c2r planning
# overwrites the array it is given.
function _plan_r2c_c2r(proto, ::Type{FT}, ns::NTuple{nd,Int}, ms::NTuple{nd,Int}; nthreads::Int = 1) where {FT, nd}
    # `proto` supplies the array kind only; each buffer names its own element type, since callers pass
    # either a complex coefficient array or a real physical one.
    rbuf = similar(proto, FT, ns...)
    cbuf = similar(proto, complex(FT), ms...)
    # UNALIGNED: the physical side of these transforms is a component slice of an `(ns…, C)` buffer,
    # whose base pointer moves by `prod(ns)*sizeof(FT)` per component. A plan built on a freshly
    # allocated array records that array's SIMD alignment and rejects any argument that does not match
    # it, which those slices do not. Writing through the slice keeps the transform's result where it
    # is used; the alternative costs a full-grid copy per component.
    flags = FFTW.ESTIMATE | FFTW.UNALIGNED
    lock(_PLAN_LOCK) do
        old_nt = FFTW.get_num_threads()
        FFTW.set_num_threads(nthreads)
        try
            return (FFTW.plan_rfft(rbuf; flags = flags), FFTW.plan_brfft(cbuf, ns[1]; flags = flags))
        finally
            FFTW.set_num_threads(old_nt)
        end
    end
end

# Build the padded (≈3N/2) scratch once, for the exact-3/2 nonlinear term. `similar(velocity_hat,…)`
# propagates the array kind + eltype (fully generic — no hardcoded Float64).
function _make_padded_scratch(velocity_hat::AbstractArray{<:Complex}, ks, fft_nthreads::Int)
    nd   = length(ks)
    ns   = FIT.SpectralLayout.full_size(ks)
    ms   = FIT.SpectralLayout.spectral_size(ks)
    half = FIT.SpectralLayout.is_half(ks)
    M    = size(velocity_hat, nd + 1)
    FT   = real(eltype(velocity_hat))
    Ms   = ntuple(d -> _padded_len(ns[d]), nd)                      # padded physical grid
    Msp  = ntuple(d -> (half && d == 1) ? Ms[1] ÷ 2 + 1 : Ms[d], nd)  # padded coefficient grid
    spec   = similar(velocity_hat, Msp...)
    u_phys = similar(velocity_hat, FT, Ms..., nd)
    g_phys = similar(velocity_hat, FT, Ms...)
    n_phys = similar(velocity_hat, FT, Ms..., M)
    # ms→Msp permutation: mode index i on axis d carries the signed wavenumber `km`, landing at
    # padded index km≥0 ? km+1 : Msp+km+1 (positive frequencies at the front, negative at the back).
    # A half layout's first axis holds only km ≥ 0, so that axis takes the first branch throughout.
    emap = _padded_map(ks, ms, Msp, FT)
    if half
        p_rfft, p_brfft = _plan_r2c_c2r(velocity_hat, FT, Ms, Msp; nthreads = fft_nthreads)
        return RealPaddedScratch(p_rfft, p_brfft, spec, u_phys, g_phys, n_phys, emap)
    end
    p_fft, p_bfft = _plan_fft_bfft(spec; nthreads = fft_nthreads)
    return ComplexPaddedScratch(p_fft, p_bfft, spec, similar(velocity_hat, Msp...),
                                u_phys, g_phys, n_phys, emap)
end

"""
    _nonlinear_term_fft!(ws, velocity_hat, ks; truncate=true, advecting_hat=velocity_hat)

Allocation-free pseudospectral nonlinear term N̂ = FFT[(u_adv·∇)u] written into `ws.N̂`, using
the pre-planned transforms / scratch in `ws.plans`. The 2/3 input truncation is folded into the
spectral copies (no temporary dealiased array) and the output is re-zeroed above the cutoff.
Normalisation: `ifft = bfft/Np`, and the forward result is divided by `Np` (package coefficient
convention). `N_i = (u_adv)_j ∂_j u_i`: `u_phys` is the advecting velocity and `g_phys` carries one
advected gradient at a time.
"""
function FIT.NonlinearTerm._nonlinear_term_fft!(
    ws,
    velocity_hat,
    ks;
    truncate::Bool = true,
    advecting_hat = velocity_hat,
)
    return _nlt!(ws.plans, ws, velocity_hat, ks, truncate, advecting_hat)
end

# Product of the per-axis dealias keep factors at every mode. A lazy broadcast: fusing it into the
# surrounding expression multiplies by `Σ_d m_d` stored numbers and materialises nothing.
@inline _keepmask(keep, ::Val{1}) = keep[1]
@inline _keepmask(keep, ::Val{2}) = Broadcast.broadcasted(*, keep[1], keep[2])
@inline _keepmask(keep, ::Val{3}) = Broadcast.broadcasted(*, keep[1], keep[2], keep[3])

# The term splits into three legs, each usable on its own: synthesise the advecting velocity,
# accumulate `Σ_j u_j ∂_j f` into `N_phys`, transform that forward. The mode-to-mode driver reuses the
# first and third and supplies its own middle leg, whose giver is a single mode.
function _nlt!(pb::AnyFFTPlans, ws, velocity_hat, ks, truncate::Bool, advecting_hat)
    _synth_advecting!(pb, ws, advecting_hat, ks, truncate)
    _accumulate_advection!(pb, ws, velocity_hat, ks, truncate)
    _forward_nhat!(pb, ws, ks, truncate)
    return ws.N̂
end

# --- leg 1: advecting velocity into ws.u_phys ---
# `u_phys[...,j] = Σ_k û e^{ik·x}` — an UNNORMALIZED inverse, since `û` already carries the `1/Nᵈ` of
# the package convention. `j = 1:nd` (spatial directions only).
function _synth_advecting!(pb::ComplexFFTPlans, ws, advecting_hat, ks, truncate::Bool)
    nd = length(ks)
    keep = _keepmask(pb.keep, Val(nd))
    for j in 1:nd
        a_j = selectdim(advecting_hat, nd + 1, j)
        truncate ? (pb.spec .= keep .* a_j) : (pb.spec .= a_j)
        LinearAlgebra.mul!(pb.spec2, pb.p_bfft, pb.spec)
        selectdim(ws.u_phys, nd + 1, j) .= real.(pb.spec2)
    end
    return ws.u_phys
end

# `brfft` reconstructs the sum over the full Hermitian spectrum from the stored half, so the physical
# field comes out real and unnormalized — the value the c2c engine gets from `real(bfft(û))`.
function _synth_advecting!(pb::RealFFTPlans, ws, advecting_hat, ks, truncate::Bool)
    nd = length(ks)
    keep = _keepmask(pb.keep, Val(nd))
    for j in 1:nd
        a_j = selectdim(advecting_hat, nd + 1, j)
        truncate ? (pb.spec .= keep .* a_j) : (pb.spec .= a_j)
        LinearAlgebra.mul!(selectdim(ws.u_phys, nd + 1, j), pb.p_brfft, pb.spec)
    end
    return ws.u_phys
end

# --- leg 2: 𝒩_i = Σ_j (u_adv)_j ∂_j f_i, one gradient at a time through ws.g_phys ---
function _accumulate_advection!(pb::ComplexFFTPlans, ws, velocity_hat, ks, truncate::Bool)
    nd = length(ks); M = size(velocity_hat, nd + 1); FT = real(eltype(velocity_hat))
    keep = _keepmask(pb.keep, Val(nd))
    for c in 1:M
        v_c = selectdim(velocity_hat, nd + 1, c)
        Nc  = selectdim(ws.N_phys, nd + 1, c)
        fill!(Nc, zero(FT))
        for j in 1:nd
            kj = pb.kg[j]
            pb.spec .= (im .* kj) .* keep .* v_c
            LinearAlgebra.mul!(pb.spec2, pb.p_bfft, pb.spec)
            ws.g_phys .= real.(pb.spec2)
            Nc .+= selectdim(ws.u_phys, nd + 1, j) .* ws.g_phys
        end
    end
    return ws.N_phys
end

function _accumulate_advection!(pb::RealFFTPlans, ws, velocity_hat, ks, truncate::Bool)
    nd = length(ks); M = size(velocity_hat, nd + 1); FT = real(eltype(velocity_hat))
    keep = _keepmask(pb.keep, Val(nd))
    for c in 1:M
        v_c = selectdim(velocity_hat, nd + 1, c)
        Nc  = selectdim(ws.N_phys, nd + 1, c)
        fill!(Nc, zero(FT))
        for j in 1:nd
            kj = pb.kg[j]
            pb.spec .= (im .* kj) .* keep .* v_c
            LinearAlgebra.mul!(ws.g_phys, pb.p_brfft, pb.spec)
            Nc .+= selectdim(ws.u_phys, nd + 1, j) .* ws.g_phys
        end
    end
    return ws.N_phys
end

# --- leg 3: 𝒩̂_i = fft(𝒩_i)/Np, zeroed above the 2/3 cutoff ---
function _forward_nhat!(pb::ComplexFFTPlans, ws, ks, truncate::Bool)
    nd = length(ks); M = size(ws.N̂, nd + 1)
    Np = real(eltype(ws.N̂))(prod(FIT.SpectralLayout.full_size(ks)))
    keep = _keepmask(pb.keep, Val(nd))
    for c in 1:M
        pb.spec .= selectdim(ws.N_phys, nd + 1, c)
        LinearAlgebra.mul!(pb.spec2, pb.p_fft, pb.spec)
        Nhat_c = selectdim(ws.N̂, nd + 1, c)
        truncate ? (Nhat_c .= (pb.spec2 .* keep) ./ Np) : (Nhat_c .= pb.spec2 ./ Np)
    end
    return ws.N̂
end

function _forward_nhat!(pb::RealFFTPlans, ws, ks, truncate::Bool)
    nd = length(ks); M = size(ws.N̂, nd + 1)
    Np = real(eltype(ws.N̂))(prod(FIT.SpectralLayout.full_size(ks)))
    keep = _keepmask(pb.keep, Val(nd))
    for c in 1:M
        LinearAlgebra.mul!(pb.spec, pb.p_rfft, selectdim(ws.N_phys, nd + 1, c))
        Nhat_c = selectdim(ws.N̂, nd + 1, c)
        truncate ? (Nhat_c .= (pb.spec .* keep) ./ Np) : (Nhat_c .= pb.spec ./ Np)
    end
    return ws.N̂
end

# Entry points the core mode-to-mode driver calls (stubs live in `NonlinearTerm`).
FIT.NonlinearTerm._nlt_synth_advecting_fft!(ws, advecting_hat, ks, truncate::Bool) =
    _synth_advecting!(ws.plans, ws, advecting_hat, ks, truncate)
FIT.NonlinearTerm._nlt_forward_fft!(ws, ks, truncate::Bool) =
    _forward_nhat!(ws.plans, ws, ks, truncate)

# ---------------------------------------------------------------------------
# Exact 3/2 zero-padded nonlinear term (PaddedThreeHalves)
# ---------------------------------------------------------------------------

# Smallest padded length ≥ 3N/2 with (M − N) even (so the centred block embeds symmetrically).
function _padded_len(n::Int)
    m = cld(3n, 2)
    return iseven(m - n) ? m : m + 1
end

# Coarse ↔ padded mode correspondence, precomputed once.
#
# Away from Nyquist each coarse mode is one padded mode and the map is a permutation. The Nyquist
# mode of an even coarse axis is not: that one slot carries `+n/2` and `−n/2` together, and on the
# finer grid those are two distinct modes. The unique real band-limited function it represents is
# `û_Nyq·cos(n·x/2)`, so it embeds as `û_Nyq/2` at EACH of `±n/2` — halved, and duplicated on every
# axis whose coarse mode sits at Nyquist. A half-layout first axis stores only `+n/2`, its conjugate
# implied by the transform, so it takes the halving with a single target.
#
# `wt` inverts that on the way back, so `truncate ∘ embed` is the identity on every mode.
struct PaddedMap{NDIM, FT}
    src::Vector{Int}                      # linear index into the coarse (ms) array
    dst::Vector{CartesianIndex{NDIM}}     # padded index
    we::Vector{FT}                        # embed weight
    wt::Vector{FT}                        # truncate weight
    nyq_row::Int                          # coarse axis-1 Nyquist row to recombine, 0 when there is none
end

function _padded_map(ks, ms::NTuple{nd,Int}, Msp::NTuple{nd,Int}, ::Type{FT}) where {nd, FT}
    src = Int[]; dst = CartesianIndex{nd}[]; we = FT[]; wt = FT[]
    # On a half first axis the coarse Nyquist plane is stored once and is its own image under k ↦ −k,
    # so truncation has to recombine it (see `_sym_nyquist_row!`). Any other layout needs nothing.
    nyq_row = (ks[1] isa FIT.SpectralLayout.HalfAxis &&
               iseven(FIT.SpectralLayout.full_length(ks[1]))) ? ms[1] : 0
    lin = LinearIndices(ms)
    for I in CartesianIndices(ms)
        # Per axis: the padded index (or the two of them, at a full-axis Nyquist) and the split count.
        opts = ntuple(nd) do d
            km = FIT.SpectralLayout.axis_index_wavenumber(ks[d], I[d])
            at_nyq = FIT.SpectralLayout.is_nyquist(ks[d], I[d])
            if at_nyq && !(d == 1 && ks[d] isa FIT.SpectralLayout.HalfAxis)
                a = abs(km)
                (a + 1, Msp[d] - a + 1)            # +n/2 and −n/2 on the finer grid
            elseif at_nyq
                (abs(km) + 1,)                     # half layout: the conjugate half is implied
            else
                (km >= 0 ? km + 1 : Msp[d] + km + 1,)
            end
        end
        nyq_axes = count(d -> FIT.SpectralLayout.is_nyquist(ks[d], I[d]), 1:nd)
        halving = FT(1) / FT(2)^nyq_axes           # amplitude split across the Nyquist images
        ntar = prod(length, opts)
        for choice in Iterators.product(opts...)
            push!(src, lin[I])
            push!(dst, CartesianIndex(choice))
            push!(we, halving)
            push!(wt, FT(1) / (FT(ntar) * halving))
        end
    end
    return PaddedMap{nd, FT}(src, dst, we, wt, nyq_row)
end

# Recombine the coarse Nyquist plane of a half first axis.
#
# That plane is stored once and is its own image under `k ↦ −k`, so its value is the sum of the two
# padded modes `±n₁/2` carry. `PaddedMap` names one unmirrored target per coarse mode, so the
# accumulation leaves each entry holding twice its own padded image; the `(k₂…, −k₂…)` pair combines
# here into `(a + conj(b))/2`, which is self-conjugate — the condition for the plane to belong to a
# real field's spectrum. Each unordered pair is visited once, and a self-mirrored entry becomes real.
function _sym_nyquist_row!(dst::AbstractArray{CT}, r::Int) where {CT}
    nd = ndims(dst)
    tail = ntuple(d -> size(dst, d + 1), nd - 1)
    lin = LinearIndices(tail)
    @inbounds for J in CartesianIndices(tail)
        t = Tuple(J)
        Jm = CartesianIndex(ntuple(d -> t[d] == 1 ? 1 : tail[d] - t[d] + 2, nd - 1))
        lin[J] <= lin[Jm] || continue
        tm = Tuple(Jm)
        v = (dst[r, t...] + conj(dst[r, tm...])) / 2
        dst[r, t...] = v
        dst[r, tm...] = conj(v)
    end
    return dst
end

# In-place zero-pad through the coarse→padded map, rest zeroed. No fftshift, no allocation.
function _embed!(dest, src, pm::PaddedMap)
    fill!(dest, zero(eltype(dest)))
    @inbounds for i in eachindex(pm.src)
        dest[pm.dst[i]] += pm.we[i] * src[pm.src[i]]
    end
    return dest
end

# In-place embed of the spectral derivative i·k_j·src through the same map. `kgj`/`dkj` are the
# axis-`j` wavenumber and derivative-keep arrays reshaped to broadcast along axis `j`; each holds one
# value per index on that axis, so the linear index `I[j]` selects the entry for mode `I`. `dkj`
# zeroes the Nyquist derivative, so the split entries there contribute nothing.
function _embed_deriv!(dest, src, kgj, j::Int, pm::PaddedMap, ms)
    fill!(dest, zero(eltype(dest)))
    cis = CartesianIndices(ms)
    @inbounds for i in eachindex(pm.src)
        I = cis[pm.src[i]]
        dest[pm.dst[i]] += pm.we[i] * (im * kgj[I[j]]) * src[I]
    end
    return dest
end

# In-place truncate + normalization, accumulating the padded images of each coarse mode.
function _truncate_scaled!(dst, src, pm::PaddedMap, scale)
    fill!(dst, zero(eltype(dst)))
    @inbounds for i in eachindex(pm.src)
        dst[pm.src[i]] += pm.wt[i] * src[pm.dst[i]] * scale
    end
    pm.nyq_row == 0 || _sym_nyquist_row!(dst, pm.nyq_row)
    return dst
end

"""
    _nonlinear_term_padded_fft!(ws, velocity_hat, ks; advecting_hat=velocity_hat)

Exact 3/2 zero-padded pseudospectral nonlinear term `𝒩 = (u_adv·∇)f` written into `ws.N̂`. Each
field is embedded in a `(3N/2)`-point grid, the product is formed there (so the quadratic generates
no aliasing into the resolved band), then transformed back and truncated to the original `N` modes.
Allocation-free: uses the preplanned transforms + reusable `(Ms…)` buffers in `ws.plans.pad`
(built for `PaddedThreeHalves` at workspace construction) and in-place `mul!` / fused broadcasts.
Physical normalization: `N̂[k] = DFT_Ms(𝒩)[k] / Mtot` (unnormalized synthesis; see NonlinearTerm docstring).
"""
function FIT.NonlinearTerm._nonlinear_term_padded_fft!(ws, velocity_hat, ks; advecting_hat=velocity_hat)
    pad = ws.plans.pad
    pad === nothing && throw(ArgumentError(
        "PaddedThreeHalves needs a workspace built for it — construct with dealiasing=PaddedThreeHalves()."))
    return _nlt_padded!(pad, ws, velocity_hat, ks, advecting_hat)
end

# Number of physical points of the padded grid, from the padded buffer that is sized by it.
_padded_points(pad) = length(pad.g_phys)

function _nlt_padded!(pad::ComplexPaddedScratch, ws, velocity_hat, ks, advecting_hat)
    nd   = length(ks)
    M    = size(velocity_hat, nd + 1)
    FT   = real(eltype(velocity_hat))
    emap = pad.emap
    kg   = ws.plans.kg
    ms   = FIT.SpectralLayout.spectral_size(ks)
    invM = FT(1) / FT(_padded_points(pad))     # 1/Mtot forward normalization

    for j in 1:nd
        _embed!(pad.spec, selectdim(advecting_hat, nd + 1, j), emap)
        LinearAlgebra.mul!(pad.out, pad.p_bfft, pad.spec)
        selectdim(pad.u_phys, nd + 1, j) .= real.(pad.out)
    end

    for c in 1:M
        nc  = selectdim(pad.n_phys, nd + 1, c)
        v_c = selectdim(velocity_hat, nd + 1, c)
        fill!(nc, zero(FT))
        for j in 1:nd
            _embed_deriv!(pad.spec, v_c, kg[j], j, emap, ms)
            LinearAlgebra.mul!(pad.out, pad.p_bfft, pad.spec)
            pad.g_phys .= real.(pad.out)
            nc .+= selectdim(pad.u_phys, nd + 1, j) .* pad.g_phys
        end
    end

    for c in 1:M
        pad.spec .= selectdim(pad.n_phys, nd + 1, c)          # real → complex (fused, no allocation)
        LinearAlgebra.mul!(pad.out, pad.p_fft, pad.spec)      # DFT_Ms(𝒩_c)
        _truncate_scaled!(selectdim(ws.N̂, nd + 1, c), pad.out, emap, invM)
    end
    return ws.N̂
end

function _nlt_padded!(pad::RealPaddedScratch, ws, velocity_hat, ks, advecting_hat)
    nd   = length(ks)
    M    = size(velocity_hat, nd + 1)
    FT   = real(eltype(velocity_hat))
    emap = pad.emap
    kg   = ws.plans.kg
    ms   = FIT.SpectralLayout.spectral_size(ks)
    invM = FT(1) / FT(_padded_points(pad))

    for j in 1:nd
        _embed!(pad.spec, selectdim(advecting_hat, nd + 1, j), emap)
        LinearAlgebra.mul!(selectdim(pad.u_phys, nd + 1, j), pad.p_brfft, pad.spec)
    end

    for c in 1:M
        nc  = selectdim(pad.n_phys, nd + 1, c)
        v_c = selectdim(velocity_hat, nd + 1, c)
        fill!(nc, zero(FT))
        for j in 1:nd
            _embed_deriv!(pad.spec, v_c, kg[j], j, emap, ms)
            LinearAlgebra.mul!(pad.g_phys, pad.p_brfft, pad.spec)
            nc .+= selectdim(pad.u_phys, nd + 1, j) .* pad.g_phys
        end
    end

    for c in 1:M
        LinearAlgebra.mul!(pad.spec, pad.p_rfft, selectdim(pad.n_phys, nd + 1, c))
        _truncate_scaled!(selectdim(ws.N̂, nd + 1, c), pad.spec, emap, invM)
    end
    return ws.N̂
end

# ---------------------------------------------------------------------------
# Override ShellToShellTransfer._shell_to_shell_fft
# ---------------------------------------------------------------------------

"""
    _shell_to_shell_fft!(result, ws, velocity_hat, ks; dealiasing, verify_antisymmetry)

FFT-accelerated shell-to-shell energy transfer T(n,m) using Alexakis et al. (2005)
antisymmetric definition. Writes into `result` using workspace `ws`.
Reuses ws.û_m and ws.nonlinear.N̂ buffers per mediator shell — no N_sh-fold allocations.
"""
function FIT.ShellToShellTransfer._shell_to_shell_fft!(
    result::FIT.Types.ShellToShellResult,
    ws::FIT.Workspaces.ShellToShellWorkspace,
    velocity_hat,
    ks;
    dealiasing::FIT.Types.AbstractDealiasing = FIT.Types.OrszagTwoThirds(),
    verify_antisymmetry::Bool = true,
    invariant::FIT.Types.AbstractInvariant = FIT.Types.KineticEnergy(),
    advecting_hat = velocity_hat,
)
    nd    = length(ks)
    ns    = size(velocity_hat)[1:nd]
    M     = size(velocity_hat, nd+1)   # components of the binned/carried primary field
    FT    = real(eltype(velocity_hat))
    N_sh  = size(result.transfer_matrix, 1)

    # T(n,m) = A[n,m] = Σ_{I∈S_n} Re{c*·N̂_m}, N̂_m = (u·∇)f_m (full velocity advects band-m; AMP 2005).
    # One mediator `m` at a time, reusing ws.û_m / ws.nonlinear.N̂ / ws.transfer_density. The per-mediator
    # nonlinear term is delegated to the allocation-free `compute_nonlinear_term!` FFT path (preplanned
    # transforms in ws.nonlinear.plans, in-place `mul!`) — it handles input truncation/output zeroing for
    # OrszagTwoThirds and the exact PaddedThreeHalves engine, so no dealias copies or per-mediator FFT
    # temporaries are allocated here. For energy A is antisymmetric and reduces as Σ_m A[n,m] = T(n).
    fill!(result.transfer_matrix, zero(FT))
    col = ws.net_transfer            # length-N_sh scratch, rewritten per mediator then copied out
    for m in 1:N_sh
        fill!(ws.û_m, zero(eltype(ws.û_m)))
        @inbounds for I in CartesianIndices(ns)
            ws.shell_idx[I] == m || continue
            for c in 1:M; ws.û_m[I, c] = velocity_hat[I, c]; end
        end
        FIT.NonlinearTerm.compute_nonlinear_term!(ws.nonlinear, ws.û_m, ks;
            dealiasing=dealiasing, spectral=SpectralBackends.FFTSpectralBackend(), advecting_hat=advecting_hat)
        # Density and receiver-shell scatter in one pass over the modes.
        FIT.Invariants.transfer_density_scatter!(col, invariant, velocity_hat, ws.nonlinear.N̂, ks, ws.shell_idx)
        @inbounds for n in 1:N_sh
            result.transfer_matrix[n, m] = col[n]
        end
    end

    for n in 1:N_sh
        s = zero(FT)
        for m in 1:N_sh; s += result.transfer_matrix[n, m]; end
        result.net_transfer[n] = s
    end

    max_asym = if verify_antisymmetry
        v = zero(FT)
        for n in 1:N_sh, m in 1:N_sh
            a = abs(result.transfer_matrix[n, m] + result.transfer_matrix[m, n])
            a > v && (v = a)
        end
        v
    else
        FT(NaN)
    end

    return max_asym
end

# ---------------------------------------------------------------------------
# Override TriadicOrthogonalDecomposition._temporal_block_dft_fft!
# ---------------------------------------------------------------------------

# A reusable FFTW plan + scratch for a whole `(nDFT, nx)` block, built once per `TODWorkspace` and
# batched over the time axis: one transform per (block, variable) in place of `nx` length-`nDFT` ones.
# A real segment takes the r2c plan, whose `nDFT÷2+1` output is the non-redundant half of a spectrum
# that is Hermitian by construction; `_temporal_block_dft_fft!` mirrors it back to the full bin range
# the triad indexing needs. `plan_fft`/`plan_rfft` are the AbstractFFTs generics, so a device-array
# prototype plans on its own library (cuFFT) and the block transform runs device-resident.
function FIT.TriadicOrthogonalDecomposition._temporal_dft_plan(
    nDFT, nx, isreal_data::Bool, ::SpectralBackends.FFTSpectralBackend, ::Type{RT}, proto,
) where {RT}
    CT = Complex{RT}
    if isreal_data
        windowed = similar(proto, RT, nDFT, nx)
        spec = similar(proto, CT, nDFT ÷ 2 + 1, nx)
        return (plan = FFTW.plan_rfft(windowed, 1), windowed = windowed, spec = spec, half = true)
    end
    windowed = similar(proto, CT, nDFT, nx)
    spec = similar(proto, CT, nDFT, nx)
    return (plan = FFTW.plan_fft(windowed, 1), windowed = windowed, spec = spec, half = false)
end

"""
    _temporal_block_dft_fft!(Q_blk, segment, seg_before_mean, window, win_weight, nDFT, shift, blk_mean, ctx)

Windowed temporal DFT of a whole `(nDFT, nx)` block through the reused plan/scratch `ctx`, written into
`Q_blk` fftshifted along the frequency axis. With `blk_mean` the DC bin is replaced by the un-centered
block's windowed sum, matching the direct-sum path bin for bin.
"""
function FIT.TriadicOrthogonalDecomposition._temporal_block_dft_fft!(
    Q_blk, segment, seg_before_mean, window, win_weight, nDFT, shift, blk_mean, ctx,
)
    windowed = ctx.windowed; spec = ctx.spec
    @. windowed = segment * window
    LinearAlgebra.mul!(spec, ctx.plan, windowed)
    spec .*= (win_weight / nDFT)
    if blk_mean
        # DC from the un-centered segment: Σ_t seg_before_mean[t,ix]·window[t]·(w/N).
        @views spec[1, :] .= vec(sum(seg_before_mean .* window; dims = 1)) .* (win_weight / nDFT)
    end
    # The fftshift is folded into the write, as the two contiguous runs it is: destination `shift+1 …
    # nDFT` takes sources `1 … nDFT−shift`, and destination `1 … shift` takes sources `nDFT−shift+1 …
    # nDFT`. Frequency is the leading axis of both arrays, so each run walks a column of each.
    #
    # `shift` is `nDFT÷2`, so the first run's sources all lie in the stored `1 … nDFT÷2+1` half; the
    # second run splits at `nh`, above which a source is the conjugate of its mirror, `û(−k) = conj(û(k))`.
    # Both bounds are loop-invariant, so neither run carries a division or a branch per element.
    nh = size(spec, 1)
    n2 = nDFT - shift                 # length of the run that maps to the tail of the destination
    s0 = n2 + 1                       # first source of the run that maps to the head
    sdirect = min(nh, nDFT)           # sources up to here are stored; beyond it they are mirrored
    @inbounds for ix in axes(spec, 2)
        for s in 1:n2
            Q_blk[shift + s, ix] = spec[s, ix]
        end
        for s in s0:sdirect
            Q_blk[s - n2, ix] = spec[s, ix]
        end
        for s in max(s0, nh + 1):nDFT
            Q_blk[s - n2, ix] = conj(spec[nDFT - s + 2, ix])
        end
    end
    return Q_blk
end

# ---------------------------------------------------------------------------
# FFT transform context for the compressible spectral transfer (SpectralBackends.FFTSpectralBackend). Provides the
# O(Nᵈ log Nᵈ) analysis/synthesis/gradient primitives the compressible core assembly calls through
# `FIT.Compressible.TransformContext`, replacing its dependency-free explicit-DFT (SpectralBackends.DirectSumSpectralBackend).
# One forward + one backward plan and two (ns...) scratch buffers are shared across every component
# and every transform in a call (created once under the plan lock). Convention matches the core:
# synthesis u = Σ û e^{ik·x} = bfft(û); analysis û = fft(u)/Nᵈ; gradient ∂_d f = bfft(i k_d f̂).
# ---------------------------------------------------------------------------
function FIT.Compressible._fft_tf(velocity_hat, ks, ns::NTuple{nd, Int}; fft_nthreads::Int = 1) where {nd}
    FT = real(eltype(velocity_hat))
    CT = complex(FT)
    Np = FT(prod(ns))
    ms = FIT.SpectralLayout.spectral_size(ks)
    half = FIT.SpectralLayout.is_half(ks)
    # Device-generic: buffers + wavenumber arrays are built in `velocity_hat`'s own array type and every
    # per-component transfer writes a contiguous last-dim slice (no scalar indexing), so the compressible
    # transform runs on CPU `Array`s (FFTW) and on device arrays (cuFFT via AbstractFFTs) unchanged.
    # The physical side is REAL on both layouts: `brfft` on the half, `real(bfft(·))` on the full.
    cbuf = similar(velocity_hat, CT, ms)
    kg = FIT.SpectralLayout.wavenumber_arrays(velocity_hat, FT, ks; derivative = true)

    if half
        p_rfft, p_brfft = _plan_r2c_c2r(velocity_hat, FT, ns, ms; nthreads = fft_nthreads)
        # `brfft` consumes its input, so each leg copies the coefficients into `cbuf` first.
        idft! = function (out, fh)
            @inbounds for c in 1:size(fh, nd + 1)
                copyto!(cbuf, selectdim(fh, nd + 1, c))
                LinearAlgebra.mul!(selectdim(out, nd + 1, c), p_brfft, cbuf)
            end
            return out
        end
        dft! = function (out, fp)
            @inbounds for c in 1:size(fp, nd + 1)
                LinearAlgebra.mul!(cbuf, p_rfft, selectdim(fp, nd + 1, c))
                selectdim(out, nd + 1, c) .= cbuf ./ Np
            end
            return out
        end
        grad! = function (g, fh)
            @inbounds for d in 1:nd, c in 1:size(fh, nd + 1)
                kd = kg[d]
                fc = selectdim(fh, nd + 1, c)
                cbuf .= (im .* kd) .* fc
                LinearAlgebra.mul!(selectdim(selectdim(g, nd + 2, d), nd + 1, c), p_brfft, cbuf)
            end
            return g
        end
    else
        obuf = similar(velocity_hat, CT, ms)
        p_fft, p_bfft = _plan_fft_bfft(cbuf; nthreads = fft_nthreads)
        idft! = function (out, fh)
            @inbounds for c in 1:size(fh, nd + 1)
                copyto!(cbuf, selectdim(fh, nd + 1, c))
                LinearAlgebra.mul!(obuf, p_bfft, cbuf)
                selectdim(out, nd + 1, c) .= real.(obuf)
            end
            return out
        end
        dft! = function (out, fp)
            @inbounds for c in 1:size(fp, nd + 1)
                cbuf .= selectdim(fp, nd + 1, c)
                LinearAlgebra.mul!(obuf, p_fft, cbuf)
                selectdim(out, nd + 1, c) .= obuf ./ Np
            end
            return out
        end
        grad! = function (g, fh)
            @inbounds for d in 1:nd, c in 1:size(fh, nd + 1)
                kd = kg[d]
                fc = selectdim(fh, nd + 1, c)
                cbuf .= (im .* kd) .* fc
                LinearAlgebra.mul!(obuf, p_bfft, cbuf)
                selectdim(selectdim(g, nd + 2, d), nd + 1, c) .= real.(obuf)
            end
            return g
        end
    end
    # Out-of-place siblings allocate a device-kind output then delegate to the in-place kernels.
    idft = fh -> idft!(similar(fh, FT, ns..., size(fh, nd + 1)), fh)
    dft  = fp -> dft!(similar(fp, CT, ms..., size(fp, nd + 1)), fp)
    grad = fh -> grad!(similar(fh, FT, ns..., size(fh, nd + 1), nd), fh)
    return FIT.Compressible.TransformContext(idft, dft, grad, idft!, dft!, grad!)
end

end # module FlowInvariantTransferFFTWExt
