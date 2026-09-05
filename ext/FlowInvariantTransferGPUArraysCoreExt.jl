module FlowInvariantTransferGPUArraysCoreExt

using GPUArraysCore: GPUArraysCore
using FlowInvariantTransfer: FlowInvariantTransfer as FIT
using ComputationalBackends: ComputationalBackends
using SpectralBackends: SpectralBackends
using LinearAlgebra: LinearAlgebra as LA

# Device-array detection trait: any `GPUArraysCore.AbstractGPUArray` (CuArray / JLArray / ROCArray / …) is a device
# array; host arrays keep the core default `false`. Used to route device inputs correctly (reject the
# host-only DirectSum reference and the host-array-under-GPUBackend case).
ComputationalBackends.is_gpu_array(::GPUArraysCore.AbstractGPUArray) = true

# ---------------------------------------------------------------------------
# Device-generic versions of the three compressible-pipeline reductions/masks that the scalar `src`
# methods can't express without scalar indexing (a dealias-truncated copy, the Helmholtz split, and the
# shell-binning reduction). Selected for device arrays (CuArray / JLArray) via `GPUArraysCore.AbstractGPUArray`
# dispatch; host `Array`s keep the scalar 0-alloc `src` methods. Everything else in the compressible
# pipeline is already device-generic broadcasts + the device-generic FFT transform context, so these
# complete the on-device path (FFT → cuFFT by construction). Pure broadcasts/reductions — no KA kernel,
# no `synchronize` — so they run under `allowscalar(false)` and are verifiable on JLArrays.
# ---------------------------------------------------------------------------

# Dealias keep-mask (mode kept iff |k_d| < n_d÷3 for all d), device-resident. Built from per-dimension
# keep vectors (host Int test) moved to the device and broadcast-AND'd into the (ns) grid.
function _dealias_keep(proto, ns::NTuple{nd,Int}) where {nd}
    vs = ntuple(nd) do d
        n = ns[d]
        kv = Bool[(i0 = i - 1; kabs = i0 <= n ÷ 2 ? i0 : n - i0; kabs < n ÷ 3) for i in 1:n]
        vd = similar(proto, Bool, n); copyto!(vd, kv)
        reshape(vd, ntuple(j -> j == d ? n : 1, nd))
    end
    return (&).(vs...)   # one fused pass over the grid, one `(ns)` Bool result
end

# dst = src, zeroing the Orszag 2/3 discard band (|k| ≥ n_d÷3) when `trunc`.
function FIT.Compressible._copy_trunc!(dst::GPUArraysCore.AbstractGPUArray, src, ks, ns::NTuple{nd,Int}, trunc::Bool) where {nd}
    if trunc
        keep = _dealias_keep(dst, ns)
        dst .= reshape(keep, ns..., 1) .* src
    else
        copyto!(dst, src)
    end
    return dst
end

# Helmholtz split (rot ⊥ k, comp ∥ k) via broadcasts with a guarded 1/k² (0 at the DC mode → comp=0).
function FIT.Compressible._helmholtz_split!(rot::GPUArraysCore.AbstractGPUArray, comp, field_hat, ks, ns::NTuple{nd,Int}) where {nd}
    FT = real(eltype(field_hat)); colons = ntuple(_ -> Colon(), nd)
    # `derivative_wavenumber` per axis, matching the host method: the Nyquist component of an even
    # axis contributes nothing to the grid divergence the split is built from.
    kg = ntuple(nd) do d
        h = FT[FIT.SpectralLayout.derivative_wavenumber(ks[d], i) for i in 1:ns[d]]
        v = similar(rot, FT, ns[d]); copyto!(v, h)
        reshape(v, ntuple(i -> i == d ? ns[d] : 1, nd))
    end
    k2 = similar(rot, FT, ns); fill!(k2, zero(FT))
    for d in 1:nd; k2 .+= kg[d] .^ 2; end
    kdotu = similar(rot, ns); fill!(kdotu, zero(eltype(kdotu)))   # Σ_c k_c·field_c (complex, ns)
    for c in 1:nd
        kdotu .+= kg[c] .* view(field_hat, colons..., c)
    end
    # Each component's compressive part is written straight into `comp` and read back for `rot`, so the
    # guarded 1/k² (0 at the DC mode → comp = 0) fuses in and no `(ns)` temporary is formed.
    for c in 1:nd
        cc = view(comp, colons..., c)
        cc .= kdotu .* ifelse.(k2 .> 0, inv.(k2), zero(FT)) .* kg[c]
        view(rot, colons..., c) .= view(field_hat, colons..., c) .- cc
    end
    return nothing
end

# Shell-bin a per-mode density into shell sums (device reductions → host vector); the keep-mask excludes
# the 2/3 band when `dealias`. Mode → shell 0 (unassigned) is excluded since `n` runs 1:N_sh.
function FIT.Compressible._bin(td::GPUArraysCore.AbstractGPUArray, sidx, N_sh, ::Type{FT}, ks, ns::NTuple{nd,Int}, dealias::Bool) where {nd, FT}
    sidx_d = sidx isa GPUArraysCore.AbstractGPUArray ? sidx : copyto!(similar(td, Int, ns), sidx)
    T = Vector{FT}(undef, N_sh)
    # Each shell sum is one fused device reduction over the mode grid: the membership test and the
    # dealias predicate are evaluated per mode inside the reduction, so no `(ns)` mask or product is
    # materialized for any of the `N_sh` shells.
    if dealias
        keep = _dealias_keep(td, ns)
        for n in 1:N_sh
            T[n] = mapreduce((s, t, k) -> (s == n) & k ? t : zero(FT), +, sidx_d, td, keep; init = zero(FT))
        end
    else
        for n in 1:N_sh
            T[n] = mapreduce((s, t) -> s == n ? t : zero(FT), +, sidx_d, td; init = zero(FT))
        end
    end
    return T
end

# ---------------------------------------------------------------------------
# Device-generic Triadic Orthogonal Decomposition kernels (dispatched on GPUArraysCore.AbstractGPUArray). The host
# `Array` methods (0-alloc, scalar-optimized) are untouched; these run the same math with broadcasts /
# array-dispatched LinearAlgebra so a device-array input `X` executes device-resident. The big
# `O(nDFT²·nx)` / `O(nStateNx·nBlks)` products go through cuBLAS/cuSOLVER (`*`, `mul!`, `qr!`); the tiny
# `r×r` Hermitian eig hops to the host (small dense eig is CPU-optimal + universally supported — JLArrays
# lacks a native one). Verified device-generic on JLArrays under allowscalar(false).
# ---------------------------------------------------------------------------

# Whole-block temporal DFT for a device-array segment on the direct-sum reference path: one matmul with
# the DFT matrix + fftshift, where the host method is a per-column scalar loop. With a plan present the
# transform is the batched one from the FFTW extension, whose `plan_rfft`/`plan_fft` are the AbstractFFTs
# generics and so plan on the array's own library (cuFFT) — one code path for host and device. `_` args
# match the host signature (per-column scratch, unused here). Runs on any
# GPUArraysCore.AbstractGPUArray and on JLArrays under allowscalar(false).
function FIT.TriadicOrthogonalDecomposition._tod_dft_block!(
    Q_blk, segment::GPUArraysCore.AbstractGPUArray, seg_before_mean, window, win_weight, nDFT, shift, blk_mean,
    backend, plan, _dft_col, _windowed, _shifted)
    plan === nothing || return FIT.TriadicOrthogonalDecomposition._temporal_block_dft_fft!(
        Q_blk, segment, seg_before_mean, window, win_weight, nDFT, shift, blk_mean, plan)
    CT = eltype(Q_blk); RT = real(CT)
    win = window isa GPUArraysCore.AbstractGPUArray ? window : copyto!(similar(segment, RT, length(window)), window)
    wnd = segment .* win .* (win_weight / nDFT)                            # (nDFT × nx)
    idx = copyto!(similar(segment, RT, nDFT), RT.(0:nDFT-1))
    W = exp.((-2 * RT(π) * im / RT(nDFT)) .* (idx * transpose(idx)))       # DFT matrix W[k,t]=e^{-2πi(k-1)(t-1)/N}
    Qd = W * wnd                                                           # (nDFT × nx)
    if blk_mean
        # DC (k=1, pre-shift) from the un-centered segment: Σ_t seg_before_mean[t,ix]·window[t]·(w/N).
        view(Qd, 1:1, :) .= reshape(sum(seg_before_mean .* win; dims = 1) .* (win_weight / nDFT), 1, :)
    end
    Q_blk .= circshift(Qd, (shift, 0))                                     # fftshift along dim 1
    return Q_blk
end

# Put `A` on `proto`'s device (type-stable dispatch; host method returns `A` unchanged).
FIT.TriadicOrthogonalDecomposition._dev_like(proto::GPUArraysCore.AbstractGPUArray, A) =
    copyto!(similar(proto, eltype(A), size(A)), A)

# Modal energy budget on device: T_j = s_j · Re Σ_k conj(v[k,j])·W[k]·u[k,j] via a broadcast column
# reduction (u/v are device views, wdev a genuine device vector). Result brought to host for the scalar
# T_budget writes (T_budget is the host result array).
function FIT.TriadicOrthogonalDecomposition._tod_modal_budget!(T_budget, fi_l, fi_n, u, v, s, nm, wdev::GPUArraysCore.AbstractGPUArray)
    tb = Array(real.(vec(sum(conj.(v) .* wdev .* u; dims = 1))))
    @inbounds for j in 1:nm
        T_budget[fi_l, fi_n, j] = s[j] * tb[j]
    end
    return T_budget
end

# Default quadratic nonlinearity out[ix,iv,b] = q1[iv,ix,b]·q2[iv,ix,b] — the fused product+permute as a
# device broadcast + permutedims! (the host method is a scalar loop). `out` is a genuine device buffer.
function FIT.TriadicOrthogonalDecomposition._apply_nonlinear!(
    out::GPUArraysCore.AbstractGPUArray, ::typeof(FIT.TriadicOrthogonalDecomposition._default_nonlinear), q1, q2)
    permutedims!(out, q1 .* q2, (2, 1, 3))
    return out
end

# Per-triad method-of-snapshots SVD — device-generic broadcasts + array-dispatched qr!/mul!, tiny eig
# on the host. Same math + same return contract (views into sc.Ubuf/sc.Vbuf) as the host method.
# Dispatched on the SCRATCH element type: `sc`'s buffers are genuine device arrays (`similar(X,…)`),
# whereas `Q_hat_n`/`Q_hat_kl` arrive as `reshape`d views (wrappers, not `<:GPUArraysCore.AbstractGPUArray`), so
# dispatching on them would miss. Their broadcasts still run device-resident (reshape is a lazy device view).
function FIT.TriadicOrthogonalDecomposition._triadic_svd_serial!(
    sc::FIT.TriadicOrthogonalDecomposition._TriadSVDScratch{<:GPUArraysCore.AbstractGPUArray}, sqrt_w, inv_sqrt_w,
    Q_hat_n, Q_hat_kl, nBlks)
    nStateNx = size(Q_hat_n, 1); RT = real(eltype(Q_hat_n)); CT = eltype(Q_hat_n)
    # √w / 1/√w are host metadata vectors (shared struct type param) — move to the device once for the
    # broadcasts (constant across triads; small).
    sw  = sqrt_w     isa GPUArraysCore.AbstractGPUArray ? sqrt_w     : copyto!(similar(Q_hat_n, RT, length(sqrt_w)), sqrt_w)
    isw = inv_sqrt_w isa GPUArraysCore.AbstractGPUArray ? inv_sqrt_w : copyto!(similar(Q_hat_n, RT, length(inv_sqrt_w)), inv_sqrt_w)
    Xw, Q3 = sc.Xw, sc.Q3
    Q3 .= Q_hat_kl .* sw                                         # [i,b] = Q̂_kl[i,b]·√w[i]
    Xw .= LA.adjoint(Q_hat_n) .* transpose(sw) ./ nBlks          # [b,i] = conj(Q̂_n[i,b])·√w[i]/nBlks
    r = min(nStateNx, nBlks)
    F = LA.qr!(Q3)
    Qmat = sc.Qmat
    Ithin = copyto!(fill!(similar(Q_hat_n, CT, nStateNx, r), zero(CT)), Matrix{CT}(LA.I, nStateNx, r))
    Qmat .= F.Q * Ithin                                         # thin Q (nStateNx × r), device
    Y = sc.Y; LA.mul!(Y, F.R, Xw)
    M = sc.Mmat; LA.mul!(M, Y, Y')
    eigh = LA.eigen!(LA.Hermitian(Matrix(M)))                   # tiny r×r Hermitian eig on the host
    λ = sc.λ; sqrt_λ = sc.sqrt_λ; permh = sc.perm
    @inbounds for i in eachindex(λ)
        λ[i] = real(eigh.values[i])
    end
    sortperm!(permh, λ; rev = true)
    @inbounds for j in eachindex(sqrt_λ)
        sqrt_λ[j] = sqrt(max(λ[permh[j]], zero(RT)))
    end
    nz = count(>(eps(RT) * 100), sqrt_λ)
    if nz == 0
        return view(sc.Ubuf, :, 1:0), view(sqrt_λ, 1:0), view(sc.Vbuf, :, 1:0)
    end
    copyto!(sc.Uperm, eigh.vectors[:, permh])                   # descending-λ eigenvectors → device
    U_trunc = view(sc.Uperm, :, 1:nz)
    s = view(sqrt_λ, 1:nz)
    Vv = view(sc.Vbuf, :, 1:nz); LA.mul!(Vv, Y', U_trunc)       # V = Yᴴ·U
    Uv = view(sc.Ubuf, :, 1:nz); LA.mul!(Uv, Qmat, U_trunc)     # U = Q·U
    invs = copyto!(similar(Q_hat_n, CT, 1, nz), reshape(CT.(inv.(s)), 1, nz))
    Vv .*= invs .* isw                                         # [i,j] *= (1/√λ_j)·(1/√w_i)
    Uv .*= isw
    return Uv, s, Vv
end

end # module
