module FlowInvariantTransferFFTWExt

using FFTW: FFTW
using FlowInvariantTransfer: FlowInvariantTransfer as FIT
using FlowTransformBindings: FlowTransformBindings as FTB
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

# Forward/backward FFT plans for `x` at `nthreads` FFTW threads, built under FFTW.jl's planner lock
# with the planner count set for the build and restored after (`FFTW.set_num_threads(f, n)`); a plan
# executes at the count it was built with. The array type picks the library (FFTW for a host `Array`,
# cuFFT for a `CuArray`), and a device library ignores the count.
_plan_fft_bfft(x; nthreads::Int = 1) =
    FFTW.set_num_threads(() -> (FFTW.plan_fft(x), FFTW.plan_bfft(x)), nthreads)

# Preplanned transforms on the padded grid of the workspace's `Workspaces.PaddedGrid`, built only for
# PaddedThreeHalves; every buffer but the complex layout's transform output is the padded grid's own.
# Coefficients move between the caller's grid and the padded one through FlowTransformBindings'
# `PaddedMaps` (gathers built in the field's array type) and the padded derivative wavenumbers, so the
# padded term runs on the field's own device. `spec1`/`out1` are `(Msp..., 1)` reshapes of `spec`/`out`,
# sharing their memory, in the shape the maps take. One type per spectral layout, matching the two
# plan sets below.
struct ComplexPaddedScratch{PF, PB, CA, C1, RA, GA, PM, KD, R}
    p_fft::PF      # unnormalized forward c2c plan on the padded grid
    p_bfft::PB     # unnormalized backward c2c plan on the padded grid
    spec::CA       # complex (Msp...) embed target / forward input
    spec1::C1
    out::CA        # complex (Msp...) transform output
    out1::C1
    u_phys::RA     # real (Ms..., nd) advecting velocity
    g_phys::GA     # real (Ms...) one gradient at a time
    n_phys::RA     # real (Ms..., M) nonlinear term
    maps::PM       # FlowTransformBindings.PaddedMaps
    kd::KD         # padded derivative wavenumbers, one array per axis
    invM::R        # 1 / prod(Ms), the forward normalization
end

struct RealPaddedScratch{PF, PB, CA, C1, RA, GA, PM, KD, R}
    p_rfft::PF     # real (Ms...) → complex (Msp...)
    p_brfft::PB    # complex (Msp...) → real (Ms...), unnormalized; consumes its input
    spec::CA
    spec1::C1
    u_phys::RA
    g_phys::GA
    n_phys::RA
    maps::PM
    kd::KD
    invM::R
end

const AnyPaddedScratch = Union{ComplexPaddedScratch, RealPaddedScratch}

# `NonlinearTermWorkspace.plans`: the pre-planned FFTW transforms, the scratch buffers they read and
# write, and the wavenumber / dealias factors of the hot path. The field's spectral layout selects the
# type.
#
# `ComplexFFTPlans` — a field stored on the full fftfreq grid. Both legs are c2c on `(ms...)`; the
# backward transform writes to its own output buffer and the physical value is the real part.
#
# `RealFFTPlans` — a real field stored on the non-redundant half. The backward leg is a c2r
# (`brfft`) writing a real array directly and the forward leg an r2c (`rfft`). FFTW's
# multi-dimensional c2r consumes its input, so `spec` holds no live value across a backward transform.
#
# Both keep the wavenumber components and the dealias keep-mask as `nd` per-axis arrays reshaped to
# broadcast along their own axis (`FlowTransformBindings.wavenumber_arrays` / `dealias_factors`).
struct ComplexFFTPlans{PF, PB, CA, KC, KM, PS}
    p_fft::PF      # unnormalized forward c2c plan on a single (ms...) component
    p_bfft::PB     # unnormalized backward c2c plan on a single (ms...) component
    spec::CA       # complex (ms...) scratch
    spec2::CA      # complex (ms...) scratch
    kg::KC         # nd wavenumber arrays, reshaped to broadcast along their own axis
    keep::KM       # nd dealias keep factors, likewise reshaped
    pad::PS        # padded scratch when built for FIT.Types.PaddedThreeHalves, else `nothing`
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

Base.show(io::IO, p::AnyPaddedScratch) = print(io, nameof(typeof(p)), "(…)")
Base.show(io::IO, ::MIME"text/plain", p::AnyPaddedScratch) = show(io, p)
Base.show(io::IO, b::AnyFFTPlans) =
    print(io, nameof(typeof(b)), "(", b.pad === nothing ? "2/3 dealias" : "3/2 padded", ")")
Base.show(io::IO, ::MIME"text/plain", b::AnyFFTPlans) = show(io, b)

# More specific than the core `_make_fft_plans`, so the extension adds a method without overwriting
# one. The padded plans are built only when the workspace has a padded grid. `AbstractFFTs.plan_fft`
# dispatches on the array type: FFTW for an `Array`, cuFFT for a `CuArray`.
function FIT.Workspaces._make_fft_plans(velocity_hat::AbstractArray{<:Complex}, ks, pad, fft_nthreads::Int)
    ns = FTB.full_size(ks)        # physical grid
    ms = FTB.spectral_size(ks)    # coefficient grid
    FT = real(eltype(velocity_hat))
    # `derivative = true`: axis d's array is 0 at axis d's own Nyquist mode, where the grid derivative
    # along d vanishes. A mode at Nyquist on another axis keeps a well-defined, Hermitian derivative
    # along d, so the zero goes in the per-axis wavenumber array.
    kg   = FTB.wavenumber_arrays(velocity_hat, FT, ks; derivative = true)
    # The 2/3 keep factors; the legs apply them only when called to truncate.
    keep = FTB.dealias_factors(velocity_hat, FT, ks, true)

    if FTB.is_halved(ks)
        p_rfft, p_brfft = _plan_r2c_c2r(velocity_hat, FT, ns, ms; nthreads = fft_nthreads)
        return RealFFTPlans(p_rfft, p_brfft, similar(velocity_hat, ms...), kg, keep,
                            _padded_scratch(velocity_hat, ks, pad, fft_nthreads))
    end

    ct  = similar(velocity_hat, ms...)   # complex (ms...) — device-generic
    ct2 = similar(velocity_hat, ms...)
    p_fft, p_bfft = _plan_fft_bfft(ct; nthreads = fft_nthreads)
    return ComplexFFTPlans(p_fft, p_bfft, ct, ct2, kg, keep,
                           _padded_scratch(velocity_hat, ks, pad, fft_nthreads))
end

_padded_scratch(velocity_hat, ks, pad::FIT.Workspaces.PaddedGrid, fft_nthreads) =
    _make_padded_scratch(velocity_hat, ks, pad, fft_nthreads)
_padded_scratch(velocity_hat, ks, ::Nothing, fft_nthreads) = nothing

# The workspace's plan set; a workspace built before this extension was loaded holds `nothing`.
_require_plans(ws) = _require_plans(ws.plans, ws)
_require_plans(p::AnyFFTPlans, ws) = p
_require_plans(::Nothing, ws) = throw(ArgumentError(
    "this workspace holds no FFT plans: it was built before FFTW was loaded. Build it again, or pass " *
    "transform = SpectralBackends.DirectSumSpectralBackend()."))

_require_pad(p::AnyFFTPlans) = _require_pad(p.pad)
_require_pad(pad::AnyPaddedScratch) = pad
_require_pad(::Nothing) = throw(ArgumentError(
    "PaddedThreeHalves needs a workspace built for it — construct with dealiasing=PaddedThreeHalves()."))

# Real-to-complex pair for the half layout: `rfft` on a real `(ns...)` component and the
# unnormalized inverse `brfft` back. Planned as the c2c pair is; `plan_brfft` is planned on a scratch
# copy because FFTW's multi-dimensional c2r planning overwrites the array it is given.
function _plan_r2c_c2r(proto, ::Type{FT}, ns::NTuple{nd,Int}, ms::NTuple{nd,Int}; nthreads::Int = 1) where {FT, nd}
    # `proto` supplies the array kind only; each buffer names its own element type, since callers pass
    # either a complex coefficient array or a real physical one.
    rbuf = similar(proto, FT, ns...)
    cbuf = similar(proto, complex(FT), ms...)
    # The physical side of these transforms is a component slice of an `(ns…, C)` buffer, whose base
    # pointer moves by `prod(ns)*sizeof(FT)` per component. A plan records its array's SIMD alignment
    # (`FFTW.alignment_of`) and rejects an argument with another, so it is `UNALIGNED` where that step
    # is not a multiple of FFTW's alignment unit.
    flags = (prod(ns) * sizeof(FT)) % 16 == 0 ? FFTW.ESTIMATE : FFTW.ESTIMATE | FFTW.UNALIGNED
    return FFTW.set_num_threads(nthreads) do
        (FFTW.plan_rfft(rbuf; flags = flags), FFTW.plan_brfft(cbuf, ns[1]; flags = flags))
    end
end

# The padded plans over the buffers and gather tables of `pad`, in `velocity_hat`'s array type.
function _make_padded_scratch(velocity_hat::AbstractArray{<:Complex}, ks, pad::FIT.Workspaces.PaddedGrid,
                              fft_nthreads::Int)
    FT  = real(eltype(velocity_hat))
    Ms  = FTB.full_size(pad.ks)          # padded physical grid
    Msp = FTB.spectral_size(pad.ks)      # padded coefficient grid
    if FTB.is_halved(ks)
        p_rfft, p_brfft = _plan_r2c_c2r(velocity_hat, FT, Ms, Msp; nthreads = fft_nthreads)
        return RealPaddedScratch(p_rfft, p_brfft, pad.spec, pad.spec1,
                                 pad.u_phys, pad.g_phys, pad.n_phys, pad.maps, pad.kd, pad.invM)
    end
    out = similar(velocity_hat, Msp...)
    p_fft, p_bfft = _plan_fft_bfft(pad.spec; nthreads = fft_nthreads)
    return ComplexPaddedScratch(p_fft, p_bfft, pad.spec, pad.spec1, out, reshape(out, Msp..., 1),
                                pad.u_phys, pad.g_phys, pad.n_phys, pad.maps, pad.kd, pad.invM)
end

# Product of the per-axis dealias keep factors at every mode. A lazy broadcast: fusing it into the
# surrounding expression multiplies by `Σ_d m_d` stored numbers and materialises nothing.
@inline _keepmask(keep, ::Val{1}) = keep[1]
@inline _keepmask(keep, ::Val{2}) = Broadcast.broadcasted(*, keep[1], keep[2])
@inline _keepmask(keep, ::Val{3}) = Broadcast.broadcasted(*, keep[1], keep[2], keep[3])

# The nonlinear term `N̂ = FFT[(u_adv·∇)f]/Np` on the preplanned transforms in `ws.plans`, in three legs:
# synthesise the advecting velocity into `u_phys`, accumulate `Σ_j u_j ∂_j f` into `N_phys` with one
# gradient at a time in `g_phys`, transform that forward. The 2/3 input truncation is folded into the
# spectral copies and the output is zeroed above the cutoff. Synthesis is the unnormalized `bfft`.

# --- leg 1: advecting velocity into ws.u_phys ---
# `u_phys[...,j] = Σ_k û e^{ik·x}` — an unnormalized inverse, since `û` already carries the `1/Nᵈ` of
# the package convention. `j = 1:nd` (spatial directions only).
_synth_advecting!(pb::AnyFFTPlans, ws, advecting_hat, ks, truncate::Bool) =
    _synth_components!(pb, ws.u_phys, advecting_hat, ks, length(ks), truncate)

# `dest[…, j] = Σ_k src[k, j] e^{ik·x}` for `j = 1:ncomp`, the inputs truncated to the kept band when
# `truncate`.
function _synth_components!(pb::ComplexFFTPlans, dest, src_hat, ks, ncomp::Int, truncate::Bool)
    nd = length(ks)
    keep = _keepmask(pb.keep, Val(nd))
    for j in 1:ncomp
        a_j = selectdim(src_hat, nd + 1, j)
        truncate ? (pb.spec .= keep .* a_j) : (pb.spec .= a_j)
        LinearAlgebra.mul!(pb.spec2, pb.p_bfft, pb.spec)
        selectdim(dest, nd + 1, j) .= real.(pb.spec2)
    end
    return dest
end

# `brfft` reconstructs the sum over the full Hermitian spectrum from the stored half, so the physical
# field comes out real and unnormalized — the value the c2c engine gets from `real(bfft(û))`.
function _synth_components!(pb::RealFFTPlans, dest, src_hat, ks, ncomp::Int, truncate::Bool)
    nd = length(ks)
    keep = _keepmask(pb.keep, Val(nd))
    for j in 1:ncomp
        a_j = selectdim(src_hat, nd + 1, j)
        truncate ? (pb.spec .= keep .* a_j) : (pb.spec .= a_j)
        LinearAlgebra.mul!(selectdim(dest, nd + 1, j), pb.p_brfft, pb.spec)
    end
    return dest
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
            truncate ? (pb.spec .= (im .* kj) .* keep .* v_c) : (pb.spec .= (im .* kj) .* v_c)
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
            truncate ? (pb.spec .= (im .* kj) .* keep .* v_c) : (pb.spec .= (im .* kj) .* v_c)
            LinearAlgebra.mul!(ws.g_phys, pb.p_brfft, pb.spec)
            Nc .+= selectdim(ws.u_phys, nd + 1, j) .* ws.g_phys
        end
    end
    return ws.N_phys
end

# --- leg 3: 𝒩̂_i = fft(𝒩_i)/Np, zeroed above the 2/3 cutoff ---
function _forward_nhat!(pb::ComplexFFTPlans, ws, ks, truncate::Bool)
    nd = length(ks); M = size(ws.N̂, nd + 1)
    Np = real(eltype(ws.N̂))(prod(FTB.full_size(ks)))
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
    Np = real(eltype(ws.N̂))(prod(FTB.full_size(ks)))
    keep = _keepmask(pb.keep, Val(nd))
    for c in 1:M
        LinearAlgebra.mul!(pb.spec, pb.p_rfft, selectdim(ws.N_phys, nd + 1, c))
        Nhat_c = selectdim(ws.N̂, nd + 1, c)
        truncate ? (Nhat_c .= (pb.spec .* keep) ./ Np) : (Nhat_c .= pb.spec ./ Np)
    end
    return ws.N̂
end

# The legs `NonlinearTerm` calls (its stubs).
FIT.NonlinearTerm._nlt_synth_advecting_fft!(ws, advecting_hat, ks, truncate::Bool) =
    _synth_advecting!(_require_plans(ws), ws, advecting_hat, ks, truncate)
function FIT.NonlinearTerm._nlt_advect_fft!(ws, advected_hat, ks, truncate::Bool)
    pb = _require_plans(ws)
    _accumulate_advection!(pb, ws, advected_hat, ks, truncate)
    return _forward_nhat!(pb, ws, ks, truncate)
end
FIT.NonlinearTerm._nlt_forward_fft!(ws, ks, truncate::Bool) =
    _forward_nhat!(_require_plans(ws), ws, ks, truncate)

# ---------------------------------------------------------------------------
# Rotational form 𝒩̂ = FFT[ω × u] (stub and contract in `NonlinearTerm.compute_rotational_term!`)
# ---------------------------------------------------------------------------

# `ω̂ = i k × v̂` into `out`, with the per-axis derivative wavenumbers `kg` (zero at an even axis's
# Nyquist slot). Broadcasts over the coefficient grid, so it runs on device arrays.
function _curl_hat!(out, v, kg)
    d = ndims(out)
    v1 = selectdim(v, d, 1); v2 = selectdim(v, d, 2); v3 = selectdim(v, d, 3)
    o1 = selectdim(out, d, 1); o2 = selectdim(out, d, 2); o3 = selectdim(out, d, 3)
    k1 = kg[1]; k2 = kg[2]; k3 = kg[3]
    @. o1 = im * (k2 * v3 - k3 * v2)
    @. o2 = im * (k3 * v1 - k1 * v3)
    @. o3 = im * (k1 * v2 - k2 * v1)
    return out
end

# ω̂ is formed in `ws.N̂`, synthesised into `ws.N_phys` beside u in `ws.u_phys`, crossed in place, and
# `ws.N̂` then takes the forward transform of `ω × u`.
function FIT.NonlinearTerm._rotational_term_fft!(ws, velocity_hat, vortex_hat, ks::NTuple{3,Any}, truncate::Bool)
    pb = _require_plans(ws)
    _curl_hat!(ws.N̂, vortex_hat, pb.kg)
    _synth_components!(pb, ws.N_phys, ws.N̂, ks, 3, truncate)
    _synth_components!(pb, ws.u_phys, velocity_hat, ks, 3, truncate)
    FIT.NonlinearTerm._cross_in_place!(ws.N_phys, ws.u_phys, ws.g_phys)
    return _forward_nhat!(pb, ws, ks, truncate)
end

# 2-D: `ω̂` into component 1 of `ws.N̂`, `ω` and `u` synthesised, `ω ẑ × u` formed in `ws.N_phys`.
function FIT.NonlinearTerm._rotational_term_fft!(ws, velocity_hat, vortex_hat, ks::NTuple{2,Any}, truncate::Bool)
    pb = _require_plans(ws)
    _curl2_hat!(ws.N̂, vortex_hat, pb.kg)
    _synth_components!(pb, ws.N_phys, ws.N̂, ks, 1, truncate)
    _synth_components!(pb, ws.u_phys, velocity_hat, ks, 2, truncate)
    FIT.NonlinearTerm._cross2_in_place!(ws.N_phys, ws.u_phys)
    return _forward_nhat!(pb, ws, ks, truncate)
end

# `ω̂ = i(k₁ v̂₂ − k₂ v̂₁)` into component 1 of `out`, with the derivative wavenumbers `kg`.
function _curl2_hat!(out, v, kg)
    d = ndims(out)
    v1 = selectdim(v, d, 1); v2 = selectdim(v, d, 2); o1 = selectdim(out, d, 1)
    k1 = kg[1]; k2 = kg[2]
    @. o1 = im * (k1 * v2 - k2 * v1)
    return out
end

# The same on the padded grid: ω̂ on the caller's grid with its derivative wavenumbers, both fields
# embedded and synthesised on the padded grid, the product formed there and truncated back.
function FIT.NonlinearTerm._rotational_term_padded_fft!(ws, velocity_hat, vortex_hat, ks::NTuple{3,Any})
    pb = _require_plans(ws)
    pad = _require_pad(pb)
    _curl_hat!(ws.N̂, vortex_hat, pb.kg)
    _padded_synth!(pad, pad.n_phys, ws.N̂, 3)
    _padded_synth!(pad, pad.u_phys, velocity_hat, 3)
    FIT.NonlinearTerm._cross_in_place!(pad.n_phys, pad.u_phys, pad.g_phys)
    return _padded_forward!(pad, ws.N̂)
end

function FIT.NonlinearTerm._rotational_term_padded_fft!(ws, velocity_hat, vortex_hat, ks::NTuple{2,Any})
    pb = _require_plans(ws)
    pad = _require_pad(pb)
    _curl2_hat!(ws.N̂, vortex_hat, pb.kg)
    _padded_synth!(pad, pad.n_phys, ws.N̂, 1)
    _padded_synth!(pad, pad.u_phys, velocity_hat, 2)
    FIT.NonlinearTerm._cross2_in_place!(pad.n_phys, pad.u_phys)
    return _padded_forward!(pad, ws.N̂)
end

# ---------------------------------------------------------------------------
# Exact 3/2 zero-padded nonlinear term (PaddedThreeHalves)
# ---------------------------------------------------------------------------

# `dest[…, j]` for `j = 1:ncomp`: the coefficients of `src_hat` embedded in the padded grid
# (`FlowTransformBindings.padded_embed!`, which splits an even axis's Nyquist slot across `±n/2`) and
# synthesised there.
function _padded_synth!(pad::ComplexPaddedScratch, dest, src_hat, ncomp::Int)
    nd = ndims(pad.g_phys)
    for j in 1:ncomp
        FTB.padded_embed!(pad.spec1, FIT.NonlinearTerm._component(src_hat, j), pad.maps)
        LinearAlgebra.mul!(pad.out, pad.p_bfft, pad.spec)
        selectdim(dest, nd + 1, j) .= real.(pad.out)
    end
    return dest
end

function _padded_synth!(pad::RealPaddedScratch, dest, src_hat, ncomp::Int)
    nd = ndims(pad.g_phys)
    for j in 1:ncomp
        FTB.padded_embed!(pad.spec1, FIT.NonlinearTerm._component(src_hat, j), pad.maps)
        LinearAlgebra.mul!(selectdim(dest, nd + 1, j), pad.p_brfft, pad.spec)
    end
    return dest
end

# `∂_j` of the coefficients `src_c` (a `(ms..., 1)` view) synthesised on the padded grid into
# `pad.g_phys`. The padded derivative wavenumbers are zero on the images of the caller's Nyquist slot,
# where the caller's grid derivative vanishes.
function _padded_grad!(pad::ComplexPaddedScratch, src_c, j::Int)
    FTB.padded_embed!(pad.spec1, src_c, pad.maps)
    pad.spec .*= im .* pad.kd[j]
    LinearAlgebra.mul!(pad.out, pad.p_bfft, pad.spec)
    pad.g_phys .= real.(pad.out)
    return pad.g_phys
end

function _padded_grad!(pad::RealPaddedScratch, src_c, j::Int)
    FTB.padded_embed!(pad.spec1, src_c, pad.maps)
    pad.spec .*= im .* pad.kd[j]
    LinearAlgebra.mul!(pad.g_phys, pad.p_brfft, pad.spec)
    return pad.g_phys
end

# Every component of `pad.n_phys` transformed forward, divided by the padded point count and returned
# to the caller's grid in `N̂` (`FlowTransformBindings.padded_truncate!`).
function _padded_forward!(pad::ComplexPaddedScratch, N̂)
    nd = ndims(pad.g_phys)
    for c in 1:size(N̂, nd + 1)
        pad.spec .= selectdim(pad.n_phys, nd + 1, c)
        LinearAlgebra.mul!(pad.out, pad.p_fft, pad.spec)
        pad.out .*= pad.invM
        FTB.padded_truncate!(FIT.NonlinearTerm._component(N̂, c), pad.out1, pad.maps)
    end
    return N̂
end

function _padded_forward!(pad::RealPaddedScratch, N̂)
    nd = ndims(pad.g_phys)
    for c in 1:size(N̂, nd + 1)
        LinearAlgebra.mul!(pad.spec, pad.p_rfft, selectdim(pad.n_phys, nd + 1, c))
        pad.spec .*= pad.invM
        FTB.padded_truncate!(FIT.NonlinearTerm._component(N̂, c), pad.spec1, pad.maps)
    end
    return N̂
end

# The exact 3/2 zero-padded term `𝒩 = (u_adv·∇)f`: both fields are embedded in the padded grid, the
# product is formed there, so the quadratic aliases nothing into the resolved band, and the forward
# transform is truncated back, `N̂[k] = DFT_Ms(𝒩)[k] / Mtot`. The advecting velocity is synthesised into
# `pad.u_phys` once; each advected field then takes the gradients and the forward leg.
function FIT.NonlinearTerm._nlt_padded_synth_fft!(ws, advecting_hat, ks)
    pad = _require_pad(_require_plans(ws))
    return _padded_synth!(pad, pad.u_phys, advecting_hat, length(ks))
end

function FIT.NonlinearTerm._nlt_padded_advect_fft!(ws, advected_hat, ks)
    pad = _require_pad(_require_plans(ws))
    nd = length(ks)
    FT = real(eltype(advected_hat))
    for c in 1:size(advected_hat, nd + 1)
        nc  = selectdim(pad.n_phys, nd + 1, c)
        v_c = FIT.NonlinearTerm._component(advected_hat, c)
        fill!(nc, zero(FT))
        for j in 1:nd
            _padded_grad!(pad, v_c, j)
            nc .+= selectdim(pad.u_phys, nd + 1, j) .* pad.g_phys
        end
    end
    return _padded_forward!(pad, ws.N̂)
end

# ---------------------------------------------------------------------------
# Override ShellToShellTransfer._shell_to_shell_fft
# ---------------------------------------------------------------------------

"""
    _shell_to_shell_fft!(result, ws, velocity_hat, ks; dealiasing, invariant, advecting_hat) -> result

FFT-accelerated shell-to-shell transfer T(n,m) in the antisymmetric definition of Alexakis, Mininni &
Pouquet (2005), written into `result.transfer_matrix` and `result.net_transfer` with the buffers of
`ws`.
"""
function FIT.ShellToShellTransfer._shell_to_shell_fft!(
    result::FIT.Types.ShellToShellResult,
    ws::FIT.Workspaces.ShellToShellWorkspace,
    velocity_hat,
    ks;
    dealiasing::FIT.Types.AbstractDealiasing = FIT.Types.OrszagTwoThirds(),
    invariant::FIT.Types.AbstractInvariant = FIT.Types.KineticEnergy(),
    advecting_hat = velocity_hat,
)
    nd    = length(ks)
    ns    = size(velocity_hat)[1:nd]
    M     = size(velocity_hat, nd+1)   # components of the binned/carried primary field
    FT    = real(eltype(velocity_hat))
    N_sh  = size(result.transfer_matrix, 1)

    # T(n,m) = A[n,m] = Σ_{I∈S_n} Re{c*·N̂_m}, N̂_m = (u·∇)f_m: the full velocity advects the band-m
    # field (Alexakis–Mininni–Pouquet 2005). The advecting velocity is synthesised once; each mediator
    # `m` then takes the advection of its band field. For energy A is antisymmetric and
    # Σ_m A[n,m] = T(n).
    fill!(result.transfer_matrix, zero(FT))
    col = ws.net_transfer            # length-N_sh scratch, rewritten per mediator then copied out
    fft = SpectralBackends.FFTSpectralBackend()
    FIT.NonlinearTerm.synthesize_advecting!(ws.nonlinear, advecting_hat, ks; dealiasing = dealiasing, transform = fft)
    for m in 1:N_sh
        fill!(ws.û_m, zero(eltype(ws.û_m)))
        @inbounds for I in CartesianIndices(ns)
            ws.shell_idx[I] == m || continue
            for c in 1:M; ws.û_m[I, c] = velocity_hat[I, c]; end
        end
        FIT.NonlinearTerm.compute_advection!(ws.nonlinear, ws.û_m, ks; dealiasing = dealiasing, transform = fft)
        # Density and shell scatter in one pass over the modes.
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

    return result
end

# ---------------------------------------------------------------------------
# Override TriadicOrthogonalDecomposition._temporal_block_dft_fft!
# ---------------------------------------------------------------------------

# A reusable FFTW plan + scratch for a whole `(nDFT, nx)` block, built once per `TODWorkspace` and
# batched over the time axis. A real segment takes the r2c plan, whose `nDFT÷2+1` output is the
# non-redundant half of a Hermitian spectrum; `_temporal_block_dft_fft!` mirrors it back to the full
# bin range the triad indexing needs. A device-array prototype plans on its own library (cuFFT).
function FIT.TriadicOrthogonalDecomposition._temporal_dft_plan(
    nDFT, nx, isreal_data::Bool, ::SpectralBackends.FFTSpectralBackend, ::Type{RT}, proto,
) where {RT}
    CT = Complex{RT}
    # One FFTW thread: the block transforms run inside the per-block loop.
    if isreal_data
        windowed = similar(proto, RT, nDFT, nx)
        spec = similar(proto, CT, nDFT ÷ 2 + 1, nx)
        plan = FFTW.set_num_threads(() -> FFTW.plan_rfft(windowed, 1), 1)
        return (plan = plan, windowed = windowed, spec = spec, half = true)
    end
    windowed = similar(proto, CT, nDFT, nx)
    spec = similar(proto, CT, nDFT, nx)
    plan = FFTW.set_num_threads(() -> FFTW.plan_fft(windowed, 1), 1)
    return (plan = plan, windowed = windowed, spec = spec, half = false)
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
# FFT transform context for the compressible spectral transfer (`SpectralBackends.FFTSpectralBackend`),
# the `FIT.Compressible.TransformContext` of `O(Nᵈ log Nᵈ)` primitives. One forward and one backward
# plan and the coefficient scratch serve every component and transform of a call. Synthesis
# u = Σ û e^{ik·x} = bfft(û); analysis û = fft(u)/Nᵈ; gradient ∂_d f = bfft(i k_d f̂).
# ---------------------------------------------------------------------------
function FIT.Compressible._fft_tf(velocity_hat, ks, ns::NTuple{nd, Int}, kg; fft_nthreads::Int = 1) where {nd}
    FT = real(eltype(velocity_hat))
    CT = complex(FT)
    Np = FT(prod(ns))
    ms = FTB.spectral_size(ks)
    half = FTB.is_halved(ks)
    # The buffers and `kg` (the per-axis derivative wavenumbers) are in `velocity_hat`'s array type, and
    # every per-component transfer writes a contiguous last-dim slice, so host and device arrays share
    # this code. The physical side is real on both layouts: `brfft` on the half, `real(bfft(·))` on the
    # full.
    cbuf = similar(velocity_hat, CT, ms)

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
    return FIT.Compressible.TransformContext(idft!, dft!, grad!)
end

end # module FlowInvariantTransferFFTWExt
