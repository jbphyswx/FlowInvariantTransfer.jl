module FlowInvariantTransferFFTWExt

using FFTW: FFTW
using FlowInvariantTransfer: FlowInvariantTransfer as FIT
using SpectralBackends: SpectralBackends
using LinearAlgebra: LinearAlgebra

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

"""
    FFTPlanBundle

Pre-planned transforms + scratch buffers stored in `NonlinearTermWorkspace.plans` so the
FFT-accelerated nonlinear term allocates nothing in the hot path.
"""
# Reusable scratch + preplanned transforms on the padded (≈3N/2) grid, built at workspace
# construction ONLY when PaddedThreeHalves is requested (so the common 2/3 path never allocates the
# larger buffers). Fully parametric — buffers follow the same `(Ms..., C)` layout as
# `NonlinearTermWorkspace`, and the forward normalization is computed inline in the field eltype.
struct PaddedScratch{PF, PB, CA, RA, EM}
    p_fft::PF      # unnormalized forward plan on the padded grid
    p_bfft::PB     # unnormalized backward plan on the padded grid
    spec::CA       # complex (Ms...) scratch: embed target / transform input
    out::CA        # complex (Ms...) scratch: transform output
    u_phys::RA     # real (Ms..., nd) advecting-velocity physical buffer
    g_phys::RA     # real (Ms..., 1) gradient physical scratch (reused per component/direction)
    n_phys::RA     # real (Ms..., M) nonlinear-term physical buffer
    emap::EM       # length-prod(ns) Vector{CartesianIndex}: ns→Ms fftfreq index permutation, built
                   # once so embed/truncate are O(1)-lookup zero-pads (no fftshift allocations)
end

struct FFTPlanBundle{PF, PB, CA, KC, MA, PS}
    p_fft::PF      # unnormalized forward plan on a single (ns...) component
    p_bfft::PB     # unnormalized backward plan on a single (ns...) component
    ctmp::CA       # complex (ns...) scratch
    ctmp2::CA      # complex (ns...) scratch
    k_comp::KC     # nd real (ns...) wavenumber-component arrays (on the input's device)
    keepmask::MA   # Bool (ns...): true where the mode is KEPT (not 2/3-dealiased) (on the device)
    pad::PS        # PaddedScratch when built for FIT.Types.PaddedThreeHalves, else `nothing` — concrete either
                   # way, so `bundle.pad` access is type-stable (no dynamic dispatch in the hot path).
end

# Custom one-line `show`: NEVER recurse into a live FFTW/cuFFT plan — the default field-by-field show
# calls `show(::IO, ::FFTW.Plan)` → the C `fftw_sprint_plan`, which can SEGFAULT on a fragile/closed
# plan (see the plan-show hazard). These make displaying any workspace that holds the bundle safe.
Base.show(io::IO, ::PaddedScratch) = print(io, "PaddedScratch(…)")
Base.show(io::IO, ::MIME"text/plain", p::PaddedScratch) = show(io, p)
Base.show(io::IO, b::FFTPlanBundle) = print(io, "FFTPlanBundle(", b.pad === nothing ? "2/3 dealias" : "3/2 padded", ")")
Base.show(io::IO, ::MIME"text/plain", b::FFTPlanBundle) = show(io, b)

# More specific than the core fallback `_make_fft_plans(::Any, ::Any, ::Any) = nothing`, so this ADDS
# a method (no overwriting) — dispatched only for complex spectral fields when FFTW is loaded. The
# `dealiasing` is known at workspace construction, so the (larger) padded scratch is built here ONLY
# for PaddedThreeHalves and its concrete type flows into `FFTPlanBundle`'s `pad` — keeping every
# `bundle.pad` access type-stable with no allocation on the common 2/3 path.
function FIT.Workspaces._make_fft_plans(velocity_hat::AbstractArray{<:Complex}, ks, dealiasing, fft_nthreads::Int)
    nd = length(ks)
    ns = size(velocity_hat)[1:nd]
    FT = real(eltype(velocity_hat))
    ct  = similar(velocity_hat, ns...)   # complex (ns...) — device-generic
    ct2 = similar(velocity_hat, ns...)
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
    # Per-axis wavenumber grids + the 2/3 keep-mask: built on the host once, then copied to the
    # input's device, so the hot-path broadcasts (`ct .= im .* k_comp[j] .* keep .* v_c`) stay on-device.
    # A `Vector` (not a tuple): the hot loop does `pb.k_comp[j]` with a runtime `j`, and indexing a
    # tuple with a runtime index is type-unstable (inferred as the element union) — it would box in the
    # nonlinear-term broadcast. Vector element-type is concrete, so the indexing stays inferred.
    k_comp = [begin
        kc = similar(velocity_hat, FT, ns...)
        copyto!(kc, FT[FT(ks[d][I[d]]) for I in CartesianIndices(ns)])
        kc
    end for d in 1:nd]
    keep = similar(velocity_hat, Bool, ns...)
    copyto!(keep, Bool[!FIT.NonlinearTerm._is_dealiased(I, ns, nd) for I in CartesianIndices(ns)])
    pad = dealiasing isa FIT.Types.PaddedThreeHalves ? _make_padded_scratch(velocity_hat, ks, fft_nthreads) : nothing
    return FFTPlanBundle(p_fft, p_bfft, ct, ct2, k_comp, keep, pad)
end

# Build the padded (≈3N/2) scratch once, for the exact-3/2 nonlinear term. `similar(velocity_hat,…)`
# propagates the array kind + eltype (fully generic — no hardcoded Float64).
function _make_padded_scratch(velocity_hat::AbstractArray{<:Complex}, ks, fft_nthreads::Int)
    nd  = length(ks)
    ns  = size(velocity_hat)[1:nd]
    M   = size(velocity_hat, nd + 1)
    FT  = real(eltype(velocity_hat))
    Ms  = ntuple(d -> _padded_len(ns[d]), nd)
    spec   = similar(velocity_hat, Ms...)
    out    = similar(velocity_hat, Ms...)
    u_phys = similar(velocity_hat, FT, Ms..., nd)
    g_phys = similar(velocity_hat, FT, Ms..., 1)
    n_phys = similar(velocity_hat, FT, Ms..., M)
    p_fft, p_bfft = _plan_fft_bfft(spec; nthreads = fft_nthreads)
    # ns→Ms fftfreq permutation: mode index i (dim d) has km = (i-1)≤ns÷2 ? i-1 : i-1-ns, landing at
    # Ms index km≥0 ? km+1 : Ms+km+1 (positive freqs at the front, negative at the back).
    emap = [CartesianIndex(ntuple(d -> (let i = I[d], n = ns[d], m = Ms[d]
                km = (i - 1) <= n ÷ 2 ? i - 1 : i - 1 - n
                km >= 0 ? km + 1 : m + km + 1
            end), nd)) for I in CartesianIndices(ns)]
    return PaddedScratch(p_fft, p_bfft, spec, out, u_phys, g_phys, n_phys, emap)
end

"""
    _nonlinear_term_fft!(ws, velocity_hat, ks; truncate=true, advecting_hat=velocity_hat)

Allocation-free pseudospectral nonlinear term N̂ = FFT[(u_adv·∇)u] written into `ws.N̂`, using
the pre-planned transforms / scratch in `ws.plans`. The 2/3 input truncation is folded into the
spectral copies (no temporary dealiased array) and the output is re-zeroed above the cutoff.
Normalisation: `ifft = bfft/Np`, and the forward result is divided by `Np` (package coefficient
convention). `N_i = (u_adv)_j ∂_j u_i`: `u_phys` is the advecting velocity, `grad_phys` the
advected gradient.
"""
function FIT.NonlinearTerm._nonlinear_term_fft!(
    ws,
    velocity_hat,
    ks;
    truncate::Bool = true,
    advecting_hat = velocity_hat,
)
    pb   = ws.plans
    nd   = length(ks)
    ns   = size(velocity_hat)[1:nd]
    M    = size(velocity_hat, nd+1)   # advected-field components (D for momentum, 1 for scalar)
    FT   = real(eltype(velocity_hat))
    Np   = FT(prod(ns))
    ct   = pb.ctmp
    ct2  = pb.ctmp2
    keep = pb.keepmask

    # Advecting velocity: physical u_phys[...,j] = Σ_k û e^{ik·x} = bfft(û) (UNNORMALIZED inverse —
    # û already carries the 1/Nᵈ of the package convention, so no extra /Np; that /Np was a latent
    # bug making every transfer diagnostic Nᵈ² too small). j = 1:nd (spatial directions of u only).
    for j in 1:nd
        a_j = selectdim(advecting_hat, nd+1, j)
        truncate ? (ct .= keep .* a_j) : (ct .= a_j)
        LinearAlgebra.mul!(ct2, pb.p_bfft, ct)
        uj = selectdim(ws.u_phys, nd+1, j)
        uj .= real.(ct2)
    end

    # Advected gradient: ∂_j f_i = Σ_k i k_j f̂_i e^{ik·x} = bfft(i k_j f̂_i) (unnormalized inverse).
    for c in 1:M
        v_c = selectdim(velocity_hat, nd+1, c)
        for j in 1:nd
            truncate ? (ct .= im .* pb.k_comp[j] .* keep .* v_c) :
                         (ct .= im .* pb.k_comp[j] .* v_c)
            LinearAlgebra.mul!(ct2, pb.p_bfft, ct)
            gcj = selectdim(selectdim(ws.grad_phys, nd+2, j), nd+1, c)
            gcj .= real.(ct2)
        end
    end

    # 𝒩_i = Σ_j (u_adv)_j ∂_j f_i
    for c in 1:M
        Nc = selectdim(ws.N_phys, nd+1, c)
        fill!(Nc, zero(FT))
        for j in 1:nd
            uj  = selectdim(ws.u_phys, nd+1, j)
            gcj = selectdim(selectdim(ws.grad_phys, nd+2, j), nd+1, c)
            Nc .+= uj .* gcj
        end
    end

    # 𝒩̂_i = fft(𝒩_i)/Np, zeroed above the 2/3 cutoff
    for c in 1:M
        Nc = selectdim(ws.N_phys, nd+1, c)
        ct .= Nc
        LinearAlgebra.mul!(ct2, pb.p_fft, ct)
        Nhat_c = selectdim(ws.N̂, nd+1, c)
        truncate ? (Nhat_c .= (ct2 .* keep) ./ Np) : (Nhat_c .= ct2 ./ Np)
    end

    return ws.N̂
end

# ---------------------------------------------------------------------------
# Exact 3/2 zero-padded nonlinear term (PaddedThreeHalves)
# ---------------------------------------------------------------------------

# Smallest padded length ≥ 3N/2 with (M − N) even (so the centred block embeds symmetrically).
function _padded_len(n::Int)
    m = cld(3n, 2)
    return iseven(m - n) ? m : m + 1
end

# In-place zero-pad: dest[emap[i]] = src[i] over the ns→Ms fftfreq permutation (`emap`), rest zeroed.
# No fftshift, no allocation.
function _embed!(dest, src, emap)
    fill!(dest, zero(eltype(dest)))
    @inbounds for (lin, I) in enumerate(CartesianIndices(axes(src)))
        dest[emap[lin]] = src[I]
    end
    return dest
end

# In-place embed of the spectral derivative i·k_j·src (same permutation).
function _embed_deriv!(dest, src, kcj, emap)
    fill!(dest, zero(eltype(dest)))
    @inbounds for (lin, I) in enumerate(CartesianIndices(axes(src)))
        dest[emap[lin]] = im * kcj[I] * src[I]
    end
    return dest
end

# In-place truncate (inverse permutation) + normalization: dst[i] = src[emap[i]] * scale.
function _truncate_scaled!(dst, src, emap, scale)
    @inbounds for (lin, I) in enumerate(CartesianIndices(axes(dst)))
        dst[I] = src[emap[lin]] * scale
    end
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
    nd   = length(ks)
    M    = size(velocity_hat, nd+1)
    FT   = real(eltype(velocity_hat))
    emap = pad.emap
    kc   = ws.plans.k_comp                      # ns-grid wavenumber components (reused from the bundle)
    invM = FT(1) / FT(prod(size(pad.spec)))     # 1/Mtot forward normalization

    # 1. Advecting velocity on the padded physical grid (spatial dirs only): u = bfft(embed(û)).
    for j in 1:nd
        _embed!(pad.spec, selectdim(advecting_hat, nd+1, j), emap)
        LinearAlgebra.mul!(pad.out, pad.p_bfft, pad.spec)
        selectdim(pad.u_phys, nd+1, j) .= real.(pad.out)
    end

    # 2. Nonlinear term 𝒩_c = Σ_j u_j ∂_j f_c on the padded grid, accumulated into pad.n_phys.
    g = selectdim(pad.g_phys, nd+1, 1)
    for c in 1:M
        nc  = selectdim(pad.n_phys, nd+1, c)
        v_c = selectdim(velocity_hat, nd+1, c)
        fill!(nc, zero(FT))
        for j in 1:nd
            _embed_deriv!(pad.spec, v_c, kc[j], emap)
            LinearAlgebra.mul!(pad.out, pad.p_bfft, pad.spec)
            g .= real.(pad.out)
            uj = selectdim(pad.u_phys, nd+1, j)
            @. nc += uj * g
        end
    end

    # 3. Forward transform + truncate to ns + normalize → ws.N̂.
    for c in 1:M
        nc = selectdim(pad.n_phys, nd+1, c)
        pad.spec .= nc                          # real → complex (fused, no allocation)
        LinearAlgebra.mul!(pad.out, pad.p_fft, pad.spec)      # DFT_Ms(𝒩_c)
        _truncate_scaled!(selectdim(ws.N̂, nd+1, c), pad.out, emap, invM)
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
    for m in 1:N_sh
        fill!(ws.û_m, zero(eltype(ws.û_m)))
        @inbounds for I in CartesianIndices(ns)
            ws.shell_idx[I] == m || continue
            for c in 1:M; ws.û_m[I, c] = velocity_hat[I, c]; end
        end
        FIT.NonlinearTerm.compute_nonlinear_term!(ws.nonlinear, ws.û_m, ks;
            dealiasing=dealiasing, spectral=SpectralBackends.FFTSpectralBackend(), advecting_hat=advecting_hat)
        FIT.Invariants.transfer_density!(ws.transfer_density, invariant, velocity_hat, ws.nonlinear.N̂, ks)
        @inbounds for I in CartesianIndices(ns)
            n = ws.shell_idx[I]
            n == 0 && continue
            result.transfer_matrix[n, m] += ws.transfer_density[I]
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

# A reusable length-`nDFT` FFTW plan + windowed scratch, built once per `triadic_orthogonal_decomposition`
# call and shared across every column — replacing the per-column `fft()` (which re-plans + allocates
# on each of ~nBlks·nVar·nx calls).
function FIT.TriadicOrthogonalDecomposition._temporal_dft_plan(nDFT, ::SpectralBackends.FFTSpectralBackend, ::Type{CT}) where {CT}
    windowed = Vector{CT}(undef, nDFT)
    return (plan = FFTW.plan_fft(windowed), windowed = windowed)
end

"""
    _temporal_block_dft_fft!(dft_col, segment_col, window, win_weight, nDFT, ctx)

FFTW-accelerated temporal block DFT for a single spatial point, through the reused plan/scratch `ctx`.
Applies the window, transforms via FFTW, and normalizes — returning the DFT in **natural (unshifted)
bin order** `0,1,…,nDFT-1`, exactly like the direct-sum path. The caller
(`triadic_orthogonal_decomposition`) applies the single `fftshift`; this routine must NOT shift as
well, or the result is shifted twice and `Q_hat` ends up misaligned with the frequency axis.
"""
function FIT.TriadicOrthogonalDecomposition._temporal_block_dft_fft!(
    dft_col,
    segment_col,
    window,
    win_weight,
    nDFT,
    ctx,
)
    windowed = ctx.windowed
    @. windowed = segment_col * window
    LinearAlgebra.mul!(dft_col, ctx.plan, windowed)
    dft_col .*= (win_weight / nDFT)
    return dft_col
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
    # Device-generic: buffers + wavenumber grids are built in `velocity_hat`'s own array type, and every
    # per-component transfer is a `copyto!`/broadcast over a contiguous last-dim slice (no scalar indexing),
    # so the compressible transform runs on CPU `Array`s (FFTW) and on device arrays (cuFFT via AbstractFFTs
    # — `FFTW.plan_fft` IS `AbstractFFTs.plan_fft`) unchanged. `_plan_fft_bfft` pins the FFTW thread count.
    inbuf  = similar(velocity_hat, CT, ns)
    outbuf = similar(velocity_hat, CT, ns)
    p_fft, p_bfft = _plan_fft_bfft(inbuf; nthreads = fft_nthreads)
    kg = ntuple(nd) do d
        v = similar(velocity_hat, FT, ns[d]); copyto!(v, collect(FT, ks[d]))
        reshape(v, ntuple(i -> i == d ? ns[d] : 1, nd))
    end

    idft! = function (out, fh)                             # spectral (ns...,C) → physical (bfft), in place
        @inbounds for c in 1:size(fh, nd + 1)
            copyto!(inbuf, selectdim(fh, nd + 1, c))
            LinearAlgebra.mul!(outbuf, p_bfft, inbuf)
            copyto!(selectdim(out, nd + 1, c), outbuf)
        end
        return out
    end
    dft! = function (out, fp)                              # physical (ns...,C) → spectral (fft/Nᵈ), in place
        @inbounds for c in 1:size(fp, nd + 1)
            copyto!(inbuf, selectdim(fp, nd + 1, c))
            LinearAlgebra.mul!(outbuf, p_fft, inbuf)
            selectdim(out, nd + 1, c) .= outbuf ./ Np
        end
        return out
    end
    grad! = function (g, fh)                               # ∂_d f_c = bfft(i k_d f̂_c) → (ns...,C,nd), in place
        @inbounds for d in 1:nd, c in 1:size(fh, nd + 1)
            inbuf .= (im .* kg[d]) .* selectdim(fh, nd + 1, c)
            LinearAlgebra.mul!(outbuf, p_bfft, inbuf)
            copyto!(selectdim(selectdim(g, nd + 2, d), nd + 1, c), outbuf)
        end
        return g
    end
    # Out-of-place siblings allocate a device-kind output then delegate to the in-place kernels.
    idft = fh -> idft!(similar(fh, CT, ns..., size(fh, nd + 1)), fh)
    dft  = fp -> dft!(similar(fp, CT, ns..., size(fp, nd + 1)), fp)
    grad = fh -> grad!(similar(fh, CT, ns..., size(fh, nd + 1), nd), fh)
    return FIT.Compressible.TransformContext(idft, dft, grad, idft!, dft!, grad!)
end

end # module FlowInvariantTransferFFTWExt
