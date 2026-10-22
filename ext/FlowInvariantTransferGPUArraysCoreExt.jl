module FlowInvariantTransferGPUArraysCoreExt

using GPUArraysCore: GPUArraysCore
using FlowInvariantTransfer: FlowInvariantTransfer as FIT
using FlowTransformBindings: FlowTransformBindings as FTB
using LinearAlgebra: LinearAlgebra as LA

FIT.Types._is_device(::GPUArraysCore.AbstractGPUArray) = true

# ---------------------------------------------------------------------------
# Device methods (`GPUArraysCore.AbstractGPUArray`) of the three compressible-pipeline steps the host
# methods write with scalar indexing: the dealias-truncated copy, the Helmholtz split and the
# shell-binning reduction. Broadcasts and reductions only, so they hold under `allowscalar(false)`.
# ---------------------------------------------------------------------------

# The keep-mask of the band the compressible transfer's cubic products keep alias-free, device-resident:
# the product of the per-axis keep factors of `FlowTransformBindings.dealias_factors`, which read each
# axis's wavenumbers from `ks` and so hold for either spectral layout.
_dealias_keep(proto, ::Type{FT}, ks) where {FT} =
    reduce((a, b) -> a .* b, FTB.dealias_factors(proto, FT, ks, true; order = 3))

# dst = src, zeroing the modes outside that band when `trunc`.
function FIT.Compressible._copy_trunc!(dst::GPUArraysCore.AbstractGPUArray, src, ks, ns::NTuple{nd,Int}, trunc::Bool) where {nd}
    if trunc
        keep = _dealias_keep(dst, real(eltype(dst)), ks)
        dst .= reshape(keep, ns..., 1) .* src
    else
        copyto!(dst, src)
    end
    return dst
end

# Helmholtz split (rot ⊥ k, comp ∥ k) via broadcasts with a guarded 1/k² (0 at the DC mode → comp=0).
# `kg` holds the per-axis derivative wavenumbers, reshaped to broadcast along their own axis.
function FIT.Compressible._helmholtz_split!(rot::GPUArraysCore.AbstractGPUArray, comp, field_hat, kg, ns::NTuple{nd,Int}) where {nd}
    FT = real(eltype(field_hat)); colons = ntuple(_ -> Colon(), nd)
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

# Shell-bin a per-mode density into shell sums (device reductions → host vector), as the host `_bin`:
# each mode weighted by its Hermitian weight. Mode → shell 0 (unassigned) is excluded since `n` runs
# 1:N_sh.
function FIT.Compressible._bin(td::GPUArraysCore.AbstractGPUArray, sidx, N_sh, ::Type{FT}, ks, ns::NTuple{nd,Int}) where {nd, FT}
    sidx_d = sidx isa GPUArraysCore.AbstractGPUArray ? sidx : copyto!(similar(td, Int, ns), sidx)
    hw = FTB.fold_weights(FT, ks)
    # The weight on the whole mode grid, once; each shell sum is then one fused reduction.
    wk = copyto!(similar(td, FT, size(hw)), hw) .* fill!(similar(td, FT, ns), one(FT))
    T = Vector{FT}(undef, N_sh)
    for n in 1:N_sh
        T[n] = mapreduce((s, t, c) -> s == n ? c * t : zero(FT), +, sidx_d, td, wk; init = zero(FT))
    end
    return T
end

# ---------------------------------------------------------------------------
# Device methods (`GPUArraysCore.AbstractGPUArray`) of the Triadic Orthogonal Decomposition steps: the
# host math in broadcasts and array-dispatched LinearAlgebra, so the `O(nDFT²·nx)` / `O(nStateNx·nBlks)`
# products (`*`, `mul!`, `qr!`) run on the device libraries. The `r×r` Hermitian eig runs on the host
# (JLArrays has no device eig).
# ---------------------------------------------------------------------------

# Whole-block temporal DFT of a device segment on the direct-sum path: one matmul with the DFT matrix,
# then the fftshift. With a plan, the FFTW extension's batched transform runs, planned on the array's
# own library (cuFFT). The `_` arguments are the host method's per-column scratch.
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

# Per-triad method-of-snapshots SVD: device broadcasts and array-dispatched qr!/mul!, the small eig on
# the host, with the host method's math and return contract (views into sc.Ubuf/sc.Vbuf). Dispatched on
# the scratch type: `sc`'s buffers are device arrays, while `Q_hat_n`/`Q_hat_kl` arrive as `reshape`
# wrappers outside `GPUArraysCore.AbstractGPUArray`, whose broadcasts still run on the device.
function FIT.TriadicOrthogonalDecomposition._triadic_svd_serial!(
    sc::FIT.TriadicOrthogonalDecomposition._TriadSVDScratch{<:GPUArraysCore.AbstractGPUArray}, sqrt_w, inv_sqrt_w,
    Q_hat_n, Q_hat_kl, nBlks)
    nStateNx = size(Q_hat_n, 1); RT = real(eltype(Q_hat_n)); CT = eltype(Q_hat_n)
    # √w and 1/√w are host vectors, copied to the device for the broadcasts.
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
