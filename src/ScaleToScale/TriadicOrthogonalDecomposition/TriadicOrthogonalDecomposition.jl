module TriadicOrthogonalDecomposition

using LinearAlgebra: LinearAlgebra
using ..Types: Types
using ComputationalBackends: ComputationalBackends
using SpectralBackends: SpectralBackends

export triadic_orthogonal_decomposition, triadic_orthogonal_decomposition!, TODWorkspace,
       hamming_window, hann_window, tukey_window

# ---------------------------------------------------------------------------
# Extension stubs — overridden by FFTW / OhMyThreads / Distributed / GPU exts
# ---------------------------------------------------------------------------

"""
    _temporal_block_dft_fft!(Q_hat_blk, segment, window, win_weight, nDFT)

FFT-accelerated temporal block DFT. Stub overridden by the FFTW extension.
"""
function _temporal_block_dft_fft!(args...; kwargs...)
    throw(ArgumentError(
        "FFT-accelerated temporal DFT requires FFTW. Run `using FFTW` to load the extension."))
end

# Reusable temporal-DFT plan (+ scratch), built once per call and shared across all the per-column
# DFTs — otherwise `fft()` re-plans and allocates on every column. `nothing` for the direct-sum path
# (which needs no plan) and the FFT path when FFTW is not loaded; the FFTW extension overrides it.
_temporal_dft_plan(nDFT, backend, ::Type) = nothing

"""
    _triadic_loop_threaded!(args...; kwargs...)

Thread-parallel triad loop using OhMyThreads.
Stub overridden by the OhMyThreads extension.
"""
function _triadic_loop_threaded!(args...; kwargs...)
    throw(ArgumentError(
        "Threaded triadic decomposition requires OhMyThreads. Run `using OhMyThreads` to load the extension."))
end

"""
    _triadic_loop_distributed!(args...; kwargs...)

Distributed triad loop. Stub for future Distributed extension.
"""
function _triadic_loop_distributed!(args...; kwargs...)
    throw(ArgumentError(
        "Distributed triadic decomposition is not yet implemented."))
end

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

"""
    hamming_window(N, ::Type{T} = Float64) -> Vector{T}

Standard Hamming window of length N: w[n] = 0.54 − 0.46·cos(2πn/(N−1)), in element type `T`.
"""
function hamming_window(N, ::Type{T} = Float64) where {T}
    return T[0.54 - 0.46 * cospi(2 * (n - 1) / (N - 1)) for n in 1:N]
end

"""
    hann_window(N, ::Type{T} = Float64) -> Vector{T}

Hann (raised-cosine) window of length N: w[n] = ½(1 − cos(2πn/(N−1))), in element type `T`. Tapers to
zero at both ends — lower spectral leakage than Hamming. Pass to `triadic_orthogonal_decomposition`
via `window`.
"""
function hann_window(N, ::Type{T} = Float64) where {T}
    return T[0.5 * (1 - cospi(2 * (n - 1) / (N - 1))) for n in 1:N]
end

"""
    tukey_window(N, ::Type{T} = Float64; α=0.5) -> Vector{T}

Tukey (tapered-cosine) window of length N in element type `T`: a flat middle with cosine tapers over
a fraction `α` of the length at each end. `α = 0` is rectangular (no taper), `α = 1` is the Hann
window; intermediate `α` trades main-lobe width against leakage.
"""
function tukey_window(N, ::Type{T} = Float64; α=0.5) where {T}
    0 <= α <= 1 || throw(ArgumentError("tukey_window: α must be in [0,1] (got $α)."))
    α == 0 && return ones(T, N)
    w = ones(T, N)
    edge = α * (N - 1) / 2
    @inbounds for n in 1:N
        x = n - 1
        if x < edge
            w[n] = 0.5 * (1 + cospi(x / edge - 1))
        elseif x > (N - 1) - edge
            w[n] = 0.5 * (1 + cospi((x - (N - 1) + edge) / edge))
        end
    end
    return w
end

"""
    parse_parameters(nt, nx; window, weight, noverlap, dt)
        -> (window_vec, weight_vec, noverlap, dt, nDFT, nBlks)

Parse and validate spectral estimation parameters with sensible defaults.
Follows the MATLAB `parser()` logic from the reference implementation.
"""
function parse_parameters(nt, nx, ::Type{RT} = Float64; window=nothing, weight=nothing, noverlap=nothing, dt=nothing) where {RT}
    # Window (built directly in the compute element type RT)
    if window === nothing
        nDFT = 2^floor(Int, log2(nt / 5))
        nDFT > 256 && (nDFT = 256)
        window_vec = hamming_window(nDFT, RT)
    elseif window isa Integer
        nDFT = Int(window)
        window_vec = hamming_window(nDFT, RT)
    elseif window isa AbstractVector
        nDFT = length(window)
        window_vec = convert(Vector{RT}, window)
    else
        throw(ArgumentError("window must be nothing, an integer, or a vector"))
    end

    # Block overlap
    if noverlap === nothing
        noverlap_val = nDFT ÷ 2
    else
        noverlap_val = Int(noverlap)
        noverlap_val >= nDFT && throw(ArgumentError("Overlap ($noverlap_val) must be < nDFT ($nDFT)"))
    end

    # Time step
    dt_val = dt === nothing ? one(RT) / nDFT : RT(dt)

    # Spatial weight (in the compute element type RT)
    if weight === nothing
        weight_vec = ones(RT, nx)
    else
        weight_vec = convert(Vector{RT}, vec(weight))
        length(weight_vec) == nx || throw(ArgumentError(
            "weight must have $(nx) elements (matching spatial dimensions), got $(length(weight_vec))"))
    end

    # Number of blocks
    nBlks = (nt - noverlap_val) ÷ (nDFT - noverlap_val)

    # Feasibility check
    nDFT < 4 && throw(ArgumentError("nDFT ($nDFT) must be ≥ 4"))
    nBlks < 2 && throw(ArgumentError(
        "Not enough data for ≥ 2 blocks (nt=$nt, nDFT=$nDFT, noverlap=$noverlap_val → nBlks=$nBlks)"))

    return (window_vec, weight_vec, noverlap_val, dt_val, nDFT, nBlks)
end

"""
    frequency_axes(nDFT, dt; isreal_data, nfreq)
        -> (f, nFreq, include_triad, f_idx, fk_idx, fl_idx, fn_idx)

Compute frequency vector, triad index arrays, and the `include_triad` mask.
Ported from the MATLAB `faxes()` function.

Returns:
- `f`: Physical frequency vector (length nFreq, fftshifted).
- `nFreq`: Number of frequency bins (= nDFT).
- `include_triad`: Boolean mask on (fl, fn) grid marking valid triads.
- `f_idx`: Frequency index array (centered, e.g. -N/2 .. N/2-1).
- `fk_idx`, `fl_idx`, `fn_idx`: Linear index arrays into valid triads.
"""
function frequency_axes(nDFT, dt; isreal_data=true, nfreq=nothing)
    # Build fftshifted frequency index array
    f_idx = collect(0:nDFT-1)
    if iseven(nDFT)
        f_idx[nDFT÷2+1:end] .= f_idx[nDFT÷2+1:end] .- nDFT
    else
        f_idx[(nDFT+1)÷2+1:end] .= f_idx[(nDFT+1)÷2+1:end] .- nDFT
    end
    # fftshift: move negative frequencies to the front
    shift = iseven(nDFT) ? nDFT ÷ 2 : (nDFT - 1) ÷ 2
    f_idx = circshift(f_idx, shift)

    f = f_idx ./ (dt * nDFT)
    nFreq = length(f_idx)

    f_idx_min = f_idx[1]
    f_idx_max = f_idx[end]
    if nfreq !== nothing && nfreq < f_idx_max
        f_idx_max = nfreq
        f_idx_min = -nfreq
    end

    # Build (fl, fn) grid and compute fk = fn - fl via Toeplitz structure
    # fl_grid[i,j] = i, fn_grid[i,j] = j (1-indexed into f_idx)
    f0_idx = (nFreq ÷ 2) + 1  # index of f=0 in the shifted array

    # Build Toeplitz matrix for fk indices
    # fk_grid[i,j] maps (fl=i, fn=j) → index of fk=fn-fl in the shifted array
    fk_grid = zeros(Int, nFreq, nFreq)
    for j in 1:nFreq, i in 1:nFreq
        fk_grid[i, j] = f0_idx + (j - i)  # offset from f=0
    end

    # Build the include_triad mask
    include_triad = trues(nFreq, nFreq)

    for j in 1:nFreq, i in 1:nFreq
        fl = f_idx[i]
        fn = f_idx[j]
        fk = fn - fl

        # For real data, restrict to upper half-plane (fn ≥ 0)
        isreal_data && fn < 0 && (include_triad[i, j] = false; continue)

        # Truncate when fk, fl, or fn exceeds bounds
        if fk < f_idx_min || fk > f_idx_max || fl < f_idx_min || fl > f_idx_max || fn < f_idx_min || fn > f_idx_max
            include_triad[i, j] = false
        end

        # Ensure fk_grid index is in bounds
        if include_triad[i, j] && (fk_grid[i, j] < 1 || fk_grid[i, j] > nFreq)
            include_triad[i, j] = false
        end
    end

    # Extract linear indices of included triads
    triad_indices = findall(include_triad)
    fl_idx_out = [idx[1] for idx in triad_indices]
    fn_idx_out = [idx[2] for idx in triad_indices]
    fk_idx_out = [fk_grid[idx] for idx in triad_indices]

    return (f, nFreq, include_triad, f_idx, fk_idx_out, fl_idx_out, fn_idx_out)
end

# ---------------------------------------------------------------------------
# SVD helpers (ported from MATLAB todAlgorithm, lowrankSVD, sirovichSVD)
# ---------------------------------------------------------------------------

"""
    sirovich_svd(X) -> (U, S_diag, V)

Method-of-snapshots SVD: compute `eig(X·Xᴴ)`, sort by descending eigenvalue,
recover right singular vectors as V = Xᴴ·U·diag(1/√λ).

Returns left singular vectors U, singular values S_diag (vector), and right singular vectors V.
"""
function sirovich_svd(X)
    # X is (n, m) where n is the smaller dimension (snapshots)
    M = X * X'                                    # fresh temp — safe to consume in-place
    # In-place Hermitian eigendecomposition (destroys M, which is ours).
    eigen_result = LinearAlgebra.eigen!(LinearAlgebra.Hermitian(M))
    λ = real.(eigen_result.values)
    U = eigen_result.vectors

    # Sort by descending eigenvalue
    perm = sortperm(λ; rev=true)
    λ = λ[perm]
    U = U[:, perm]

    # Clamp small negative eigenvalues to zero (numerical noise)
    λ = max.(λ, 0)
    sqrt_λ = sqrt.(λ)

    # Only keep non-zero singular values
    nz = count(>(eps(real(eltype(X))) * 100), sqrt_λ)
    if nz == 0
        return similar(X, size(X, 1), 0), similar(sqrt_λ, 0), similar(X, size(X, 2), 0)
    end

    U_trunc = U[:, 1:nz]
    s_trunc = sqrt_λ[1:nz]
    V_trunc = X' * U_trunc * LinearAlgebra.Diagonal(1 ./ s_trunc)

    return U_trunc, s_trunc, V_trunc
end

"""
    lowrank_svd(X, Q3) -> (U, S_diag, V)

Low-rank SVD via QR factorization of Q3, then Sirovich SVD of the reduced product R·X.
Returns left singular vectors (in the original space of Q3), singular values, and right singular vectors.
"""
function lowrank_svd(X, Q3)
    Q, R = LinearAlgebra.qr!(Q3)      # Q3 is a caller-owned temp — consume it in-place
    Q_mat = Matrix(Q)  # materialize for multiplication
    U_r, S_diag, V = sirovich_svd(R * X)
    U = Q_mat * U_r
    return U, S_diag, V
end

"""
    triadic_svd(Q_hat_n, Q_hat_kl, weights, nBlks) -> (U, s, V)

Core per-triad SVD computation. Applies spatial weighting, computes low-rank SVD,
and un-weights the resulting modes.

- Q_hat_n: Recipient data matrix, size (nState*nx, nBlks)
- Q_hat_kl: Nonlinear/convective data matrix, size (nState*nx, nBlks)
- weights: Spatial weight vector (length nState*nx)
- nBlks: Number of blocks

Returns:
- U: Convective modes (un-weighted)
- s: Singular values (vector)
- V: Recipient modes (un-weighted)
"""
function triadic_svd(Q_hat_n, Q_hat_kl, weights, nBlks)
    sqrt_w = sqrt.(weights)

    # Weighted matrices: X = (Q̂_n · √w)ᴴ / nBlks,  Q3 = Q̂_kl · √w
    X = (Q_hat_n .* sqrt_w)' ./ nBlks
    Q3 = Q_hat_kl .* sqrt_w

    U, s, V = lowrank_svd(X, Q3)

    # Un-weight modes in place (U, V are fresh outputs of lowrank_svd).
    inv_sqrt_w = 1 ./ sqrt_w
    U .*= inv_sqrt_w
    V .*= inv_sqrt_w

    return U, s, V
end

# ---------------------------------------------------------------------------
# Direct-sum temporal DFT (fallback without FFTW)
# ---------------------------------------------------------------------------

"""
    _temporal_block_dft_direct!(Q_hat_blk, segment, window, win_weight, nDFT)

Direct-sum O(N²) computation of the windowed temporal DFT for a single block.
This is the fallback when FFTW is not loaded.

`segment` is a matrix of size `(nDFT, nVar*nx)`.
Result is written into `Q_hat_blk` of size `(nDFT, nVar*nx)`, fftshifted.
"""
function _temporal_block_dft_direct!(Q_hat_blk, segment, window, win_weight, nDFT)
    nCols = size(segment, 2)

    # Apply window and normalize
    windowed = segment .* window .* (win_weight / nDFT)

    # Direct DFT
    for freq_idx in 1:nDFT
        for col in 1:nCols
            val = zero(eltype(Q_hat_blk))
            for t in 1:nDFT
                # DFT: X[k] = Σ_n x[n] * exp(-2πi*(k-1)*(n-1)/N)
                phase = -2π * (freq_idx - 1) * (t - 1) / nDFT
                val += windowed[t, col] * exp(im * phase)
            end
            Q_hat_blk[freq_idx, col] = val
        end
    end

    # fftshift along first dimension
    shift = iseven(nDFT) ? nDFT ÷ 2 : (nDFT - 1) ÷ 2
    # In-place circular shift along dim 1
    buf = similar(Q_hat_blk, shift, nCols)
    buf .= Q_hat_blk[1:shift, :]
    Q_hat_blk[1:nDFT-shift, :] .= Q_hat_blk[shift+1:nDFT, :]
    Q_hat_blk[nDFT-shift+1:nDFT, :] .= buf

    return Q_hat_blk
end

# ---------------------------------------------------------------------------
# Backend dispatch for temporal DFT
# ---------------------------------------------------------------------------

# Dispatch the temporal DFT on the SPECTRAL (transform) backend.
function _compute_temporal_dft!(Q_hat_blk, segment, window, win_weight, nDFT, ::SpectralBackends.DirectSumSpectralBackend)
    _temporal_block_dft_direct!(Q_hat_blk, segment, window, win_weight, nDFT)
end

function _compute_temporal_dft!(Q_hat_blk, segment, window, win_weight, nDFT, ::SpectralBackends.FFTSpectralBackend)
    _temporal_block_dft_fft!(Q_hat_blk, segment, window, win_weight, nDFT)
end

# Whole-block temporal DFT of `segment` (nDFT × nx) into `Q_blk` (nDFT × nx), windowed + fftshifted,
# with the `blk_mean` DC (k=0) correction taken from the un-centered `seg_before_mean`. This is the
# per-block transform the analysis loop calls once per (block, variable). The host method here is the
# exact original per-column computation (FFT via the plan, or the dependency-free direct sum); the
# GPUArraysCore extension adds an `AbstractGPUArray` method that runs the whole block as one matmul so a
# device-array input executes device-resident. Dispatch selects — host path is byte-identical.
function _tod_dft_block!(Q_blk, segment, seg_before_mean, window, win_weight, nDFT, shift, blk_mean,
                         backend, plan, dft_col, windowed, shifted)
    CT = eltype(Q_blk); nx = size(segment, 2)
    for ix in 1:nx
        if backend isa SpectralBackends.FFTSpectralBackend
            _temporal_block_dft_fft!(dft_col, view(segment, :, ix), window, win_weight, nDFT, plan)
        else
            @inbounds for t in 1:nDFT
                windowed[t] = segment[t, ix] * window[t]
            end
            @inbounds for freq_k in 1:nDFT
                val = zero(CT)
                for t in 1:nDFT
                    phase = -2π * (freq_k - 1) * (t - 1) / nDFT
                    val += windowed[t] * exp(im * phase)
                end
                dft_col[freq_k] = val * (win_weight / nDFT)
            end
        end
        if blk_mean
            dc = zero(CT)
            @inbounds for t in 1:nDFT
                dc += seg_before_mean[t, ix] * window[t]
            end
            dft_col[1] = dc * (win_weight / nDFT)
        end
        circshift!(shifted, dft_col, shift)
        @inbounds @views Q_blk[:, ix] .= shifted
    end
    return Q_blk
end

# ---------------------------------------------------------------------------
# Default quadratic nonlinearity
# ---------------------------------------------------------------------------

"""
    _default_nonlinear(q1, q2)

Default quadratic nonlinearity Q(q1, q2) = q1 .* q2 with permutation
of the first two dimensions. Matches the MATLAB default:
`@(q1,q2) permute(q1.*q2, [2 1 3])`.

q1, q2 are arrays of size (nVar, nx, nBlks) or similar.
"""
function _default_nonlinear(q1, q2)
    # Element-wise product with permutation of first two dims
    # MATLAB: permute(q1.*q2, [2 1 3])
    product = q1 .* q2
    return permutedims(product, (2, 1, ntuple(i -> i + 2, ndims(product) - 2)...))
end

# In-place quadratic nonlinearity into the reused giver buffer `out` (nx, nState, nBlks), avoiding the
# default's per-triad product+permute allocation. The default Q(q1,q2)=permute(q1.*q2,(2,1,3)) is fused
# directly into `out`; a user-supplied `Q` is evaluated (its own allocation is the user's) and copied in.
function _apply_nonlinear!(out, ::typeof(_default_nonlinear), q1, q2)
    nVar, nx, nBlks = size(q1)
    @inbounds for b in 1:nBlks, iv in 1:nVar, ix in 1:nx
        out[ix, iv, b] = q1[iv, ix, b] * q2[iv, ix, b]
    end
    return out
end
_apply_nonlinear!(out, Q, q1, q2) = copyto!(out, Q(q1, q2))

# ---------------------------------------------------------------------------
# Serial triad loop
# ---------------------------------------------------------------------------

# Reusable scratch for the per-triad SVD, reused across every triad in the serial loop so the
# hot loop's matrix products (Xw, RᵀX, YYᴴ, YᴴU, QU) write into fixed buffers via `mul!` instead
# of reallocating each triad. All buffers are sized by `r = min(nStateNx, nBlks)` (the QR/SVD rank
# bound), which is correct whether the state-space is taller or wider than the block count. The
# scalar `weights` and its √/1-over-√ (constant across triads) are hoisted out of the loop too.
# `qr!`, `eigen!`, and the thin `Matrix(Q)`/eigenvector arrays are the residual per-triad allocs
# (all O(nBlks²), small); the large O(nStateNx·nBlks) products are the ones eliminated here.
struct _TriadSVDScratch{M<:AbstractMatrix}
    Xw::M     # nBlks × nStateNx      — (Q̂_n·√w)ᴴ / nBlks
    Q3::M     # nStateNx × nBlks      — Q̂_kl·√w   (consumed by qr!)
    Qmat::M   # nStateNx × r          — thin Q (materialized via lmul! into this buffer)
    Y::M      # r × nStateNx          — R · Xw
    Mmat::M   # r × r                 — Y · Yᴴ
    Uperm::M  # r × r                 — eigenvectors, columns reordered by descending eigenvalue
    Ubuf::M   # nStateNx × r          — convective modes (view [:,1:nz] returned)
    Vbuf::M   # nStateNx × r          — recipient modes  (view [:,1:nz] returned)
end
# `proto` is a prototype array (the temporal-DFT `Q_hat`) whose type the scratch buffers follow, so a
# device-array input yields device-resident buffers and the per-triad `eigen!`/`qr!`/`mul!` dispatch
# to the device solver (CUSOLVER/cuBLAS). For a host `Array` this is identical to the old `zeros`
# buffers (the `mul!`/factorizations overwrite them, so the uninitialized `similar` is fine).
function _TriadSVDScratch(proto::AbstractArray, ::Type{CT}, nStateNx::Int, nBlks::Int) where {CT}
    r = min(nStateNx, nBlks)
    z(dims...) = fill!(similar(proto, CT, dims...), zero(CT))
    return _TriadSVDScratch(
        z(nBlks, nStateNx), z(nStateNx, nBlks), z(nStateNx, r),
        z(r, nStateNx), z(r, r), z(r, r),
        z(nStateNx, r), z(nStateNx, r),
    )
end

# Fused, allocation-reusing per-triad SVD for the serial loop: identical math to
# `triadic_svd` → `lowrank_svd` → `sirovich_svd`, but every large product goes through `mul!` into
# `sc`. Returns views into `sc.Ubuf`/`sc.Vbuf` (overwritten by the next triad — the caller copies
# out the truncated `[:,1:nm]` modes it stores).
function _triadic_svd_serial!(sc::_TriadSVDScratch, sqrt_w, inv_sqrt_w, Q_hat_n, Q_hat_kl, nBlks)
    nStateNx = size(Q_hat_n, 1)
    RT = real(eltype(Q_hat_n))
    # Xw = (Q̂_n .* √w)ᴴ ./ nBlks ; Q3 = Q̂_kl .* √w
    Xw, Q3 = sc.Xw, sc.Q3
    @inbounds for i in 1:nStateNx
        wi = sqrt_w[i]
        for b in 1:nBlks
            Xw[b, i] = conj(Q_hat_n[i, b] * wi) / nBlks
            Q3[i, b] = Q_hat_kl[i, b] * wi
        end
    end
    # Q3 = QR ; thin Q into sc.Qmat via lmul! (bit-identical to Matrix(F.Q), reuses the buffer) ; Y = R · Xw
    r = min(nStateNx, nBlks)
    F = LinearAlgebra.qr!(Q3)
    Qmat = sc.Qmat
    fill!(Qmat, zero(eltype(Qmat)))
    @inbounds for i in 1:r
        Qmat[i, i] = one(eltype(Qmat))
    end
    LinearAlgebra.lmul!(F.Q, Qmat)
    Y = sc.Y
    LinearAlgebra.mul!(Y, F.R, Xw)
    # Method-of-snapshots SVD of Y: eig(Y·Yᴴ), sort ↓, recover V = Yᴴ·U·diag(1/√λ)
    M = sc.Mmat
    LinearAlgebra.mul!(M, Y, Y')
    eig = LinearAlgebra.eigen!(LinearAlgebra.Hermitian(M))
    λ = real.(eig.values)
    Uev = eig.vectors
    perm = sortperm(λ; rev = true)
    # Reorder eigenvector columns by descending eigenvalue into sc.Uperm (reused) — avoids the
    # Uev[:,perm] and U_trunc[:,1:nz] copies (each O(nBlks²)).
    Uperm = sc.Uperm
    @inbounds for j in 1:r, i in 1:r
        Uperm[i, j] = Uev[i, perm[j]]
    end
    λ = λ[perm]
    λ = max.(λ, 0)
    sqrt_λ = sqrt.(λ)
    nz = count(>(eps(RT) * 100), sqrt_λ)
    if nz == 0
        return view(sc.Ubuf, :, 1:0), similar(sqrt_λ, 0), view(sc.Vbuf, :, 1:0)
    end
    U_trunc = view(Uperm, :, 1:nz)
    s = sqrt_λ[1:nz]
    Vv = view(sc.Vbuf, :, 1:nz)
    LinearAlgebra.mul!(Vv, Y', U_trunc)      # V = Yᴴ · U_trunc
    Uv = view(sc.Ubuf, :, 1:nz)
    LinearAlgebra.mul!(Uv, Qmat, U_trunc)    # U = Q · U_trunc
    @inbounds for j in 1:nz
        isj = inv(s[j])
        for i in 1:nStateNx
            Vv[i, j] *= isj                  # · diag(1/√λ)
            Vv[i, j] *= inv_sqrt_w[i]        # un-weight recipient modes
            Uv[i, j] *= inv_sqrt_w[i]        # un-weight convective modes
        end
    end
    return Uv, s, Vv
end

# Return `A` on the same device as `proto` (host no-op). Dispatched (type-stable per `proto`): the
# GPUArraysCore extension adds the `proto::AbstractGPUArray` method that copies `A` to the device. Used to
# put the weights / 1-over-√s on the modes' device for the device-generic reductions in the triad loop.
_dev_like(proto, A) = A

# Modal energy budget T_j = s_j · Re⟨v_j, W u_j⟩. Host method: the original 0-alloc scalar accumulator.
# The GPUArraysCore extension adds a `wdev::AbstractGPUArray` method (device-generic column reduction), so
# the hot host path stays allocation-free while a device path exists — selected by dispatch on `wdev`.
function _tod_modal_budget!(T_budget, fi_l, fi_n, u, v, s, nm, wdev)
    @inbounds for j in 1:nm
        acc = zero(eltype(u))
        for k in axes(u, 1)
            acc += conj(v[k, j]) * wdev[k] * u[k, j]
        end
        T_budget[fi_l, fi_n, j] = s[j] * real(acc)
    end
    return T_budget
end

"""
    _triadic_loop_serial!(L, P, T_budget, A_out, Xi_out,
                          Q_hat, f_idx, fk_idx, fl_idx, fn_idx,
                          weights, nBlks, nFreq, nState, nx, nmode,
                          Q_nonlinear, LHS,
                          return_coefficients, return_auxiliary_modes)

Serial loop over all included triads. Computes mode bispectrum, modes,
and optionally energy budget, expansion coefficients, and auxiliary modes.
"""
function _triadic_loop_serial!(
    L, P, T_budget, A_out, Xi_out,
    Q_hat, f_idx, fk_idx, fl_idx, fn_idx,
    weights, nBlks, nFreq, nState, nx, nmode,
    Q_nonlinear, LHS,
    return_coefficients, return_auxiliary_modes,
    sc, sqrt_w, inv_sqrt_w, permbuf, permbuf_kl,   # reusable scratch supplied by the caller (TODWorkspace)
)
    nTriads = length(fk_idx)
    nStateNx = nState * nx
    wdev = _dev_like(Q_hat, weights)   # weights on the modes' device (host no-op); selected by dispatch

    for i in 1:nTriads
        fi_k = fk_idx[i]
        fi_l = fl_idx[i]
        fi_n = fn_idx[i]

        # Extract Fourier realizations for this triad (views — no per-triad copy).
        # Q_hat is (nFreq, nVar, nx, nBlks); LHS transforms (nVar, nx, nBlks) -> (nState, nx, nBlks)
        Q_n_raw = view(Q_hat, fi_n, :, :, :)    # (nVar, nx, nBlks)
        Q_k_raw = view(Q_hat, fi_k, :, :, :)
        Q_l_raw = view(Q_hat, fi_l, :, :, :)

        permutedims!(permbuf, LHS(Q_n_raw), (2, 1, 3))   # recipient reshape into the reused buffer
        Q_hat_n = reshape(permbuf, nStateNx, nBlks)
        _apply_nonlinear!(permbuf_kl, Q_nonlinear, Q_k_raw, Q_l_raw)   # giver reshape into the reused buffer
        Q_hat_kl = reshape(permbuf_kl, nStateNx, nBlks)

        # Core SVD (reuses `sc` across triads)
        U, s, V = _triadic_svd_serial!(sc, sqrt_w, inv_sqrt_w, Q_hat_n, Q_hat_kl, nBlks)

        # Store results (truncated to nmode)
        nm = min(nmode, length(s))
        u = U[:, 1:nm]
        v = V[:, 1:nm]

        # Mode bispectrum (singular values)
        for j in 1:nm
            L[fi_l, fi_n, j] = s[j]
        end

        # Modes: convective (u) and recipient (v)
        P[(fi_l, fi_n)] = (convective=u, recipient=v)

        # Modal energy budget T_j = s_j · Re⟨v_j, W u_j⟩ — dispatched on `wdev`: host 0-alloc scalar
        # accumulator, or the device-generic column reduction from the GPUArraysCore extension.
        _tod_modal_budget!(T_budget, fi_l, fi_n, u, v, s, nm, wdev)

        # Expansion coefficients
        if return_coefficients
            A_conv = u' * (Q_hat_kl .* wdev)          # A_conv  = Uᴴ · (Q̂_kl .* W)
            A_recip = v' * (Q_hat_n .* wdev)          # A_recip = Vᴴ · (Q̂_n  .* W)
            A_out[(fi_l, fi_n)] = (convective=A_conv, recipient=A_recip)

            # Donor and catalyst modes
            if return_auxiliary_modes
                Q_hat_l = reshape(permutedims(LHS(Q_l_raw), (2, 1, 3)), nStateNx, nBlks)
                Q_hat_k = reshape(permutedims(LHS(Q_k_raw), (2, 1, 3)), nStateNx, nBlks)

                # donor/catalyst = Q̂_{l,k} · Aᴴ_recip · diag(1/s) / nBlks. The diag(1/s) column-scale is a
                # broadcast (device-generic; `_dev_like` puts 1/s on the modes' device — avoids the
                # `Diagonal(host) × device` mix). Identical to the Diagonal form on the host.
                inv_s_row = _dev_like(Q_hat, reshape(1 ./ s[1:nm], 1, nm))
                donor_mode = (Q_hat_l * A_recip') .* inv_s_row ./ nBlks
                catalyst_mode = (Q_hat_k * A_recip') .* inv_s_row ./ nBlks

                Xi_out[(fi_l, fi_n)] = (donor=donor_mode[:, 1:nm], catalyst=catalyst_mode[:, 1:nm])
            end
        end
    end
end

# Compute one triad `i`'s outputs standalone (own scratch) — the unit of work for the distributed loop
# (each worker process runs this for its triads and ships the result back for master-side assembly). Uses
# the SAME `_triadic_svd_serial!` / `_apply_nonlinear!` as the serial loop, so results are bit-identical.
function _triad_result(i, Q_hat, fk_idx, fl_idx, fn_idx, weights, sqrt_w, inv_sqrt_w,
                       nBlks, nState, nx, nmode, Q_nonlinear, LHS,
                       return_coefficients, return_auxiliary_modes)
    CT = eltype(Q_hat)
    nStateNx = nState * nx
    fi_k = fk_idx[i]; fi_l = fl_idx[i]; fi_n = fn_idx[i]
    Q_n_raw = view(Q_hat, fi_n, :, :, :)
    Q_k_raw = view(Q_hat, fi_k, :, :, :)
    Q_l_raw = view(Q_hat, fi_l, :, :, :)
    permbuf = Array{CT}(undef, nx, nState, nBlks)
    permbuf_kl = Array{CT}(undef, nx, nState, nBlks)
    permutedims!(permbuf, LHS(Q_n_raw), (2, 1, 3))
    Q_hat_n = reshape(permbuf, nStateNx, nBlks)
    _apply_nonlinear!(permbuf_kl, Q_nonlinear, Q_k_raw, Q_l_raw)
    Q_hat_kl = reshape(permbuf_kl, nStateNx, nBlks)
    sc = _TriadSVDScratch(Q_hat, CT, nStateNx, nBlks)   # follow Q_hat's array type (device-generic)
    U, s, V = _triadic_svd_serial!(sc, sqrt_w, inv_sqrt_w, Q_hat_n, Q_hat_kl, nBlks)
    nm = min(nmode, length(s))
    u = U[:, 1:nm]; v = V[:, 1:nm]
    Tb = Vector{real(CT)}(undef, nm)
    @inbounds for j in 1:nm
        acc = zero(CT)
        for k in axes(u, 1)
            acc += conj(v[k, j]) * weights[k] * u[k, j]
        end
        Tb[j] = s[j] * real(acc)
    end
    sv = collect(s[1:nm])
    A_conv = A_recip = donor = catalyst = nothing
    if return_coefficients
        A_conv = u' * (Q_hat_kl .* weights)
        A_recip = v' * (Q_hat_n .* weights)
        if return_auxiliary_modes
            Q_hat_l = reshape(permutedims(LHS(Q_l_raw), (2, 1, 3)), nStateNx, nBlks)
            Q_hat_k = reshape(permutedims(LHS(Q_k_raw), (2, 1, 3)), nStateNx, nBlks)
            inv_s = 1 ./ sv
            donor = (Q_hat_l * A_recip' * LinearAlgebra.Diagonal(inv_s) ./ nBlks)[:, 1:nm]
            catalyst = (Q_hat_k * A_recip' * LinearAlgebra.Diagonal(inv_s) ./ nBlks)[:, 1:nm]
        end
    end
    return (fi_l = fi_l, fi_n = fi_n, nm = nm, s = sv, Tb = Tb, u = u, v = v,
            A_conv = A_conv, A_recip = A_recip, donor = donor, catalyst = catalyst)
end

# Dispatch the triad loop on the EXECUTION (parallelism) backend.
function _dispatch_triadic_loop!(args_tuple...; execution::ComputationalBackends.AbstractExecutionBackend=ComputationalBackends.SerialBackend(), kwargs...)
    _dispatch_triadic_loop_impl!(Types.resolve_execution(execution), args_tuple...; kwargs...)
end

_dispatch_triadic_loop_impl!(::ComputationalBackends.SerialBackend, args...; kwargs...) =
    _triadic_loop_serial!(args...; kwargs...)

_dispatch_triadic_loop_impl!(::ComputationalBackends.ThreadedBackend, args...; kwargs...) =
    _triadic_loop_threaded!(args...; kwargs...)

# The ComputationalBackends.DistributedBackend passes itself through so the ext can read its inner (per-worker) backend.
_dispatch_triadic_loop_impl!(exec::ComputationalBackends.DistributedBackend, args...; kwargs...) =
    _triadic_loop_distributed!(args..., exec; kwargs...)

# GPU path: the triad loop is device-generic — it just runs the loop, and every per-triad primitive
# (temporal DFT, weighting, `qr!`/`mul!`, the Hermitian eig, the reshape scratch) dispatches on the
# array type. Host `Array`s use the 0-alloc scalar-optimized methods here; device arrays
# (`AbstractGPUArray`) use the broadcast/matmul methods added by the GPUArraysCore extension, so the
# heavy `O(nStateNx·nBlks)` products run device-resident (cuBLAS/cuSOLVER) and the tiny `r×r` Hermitian
# eig hops to the host (small dense eig is CPU-optimal + universally supported). No separate code path,
# no host↔device movement forced by this knob: a device-array `X` flows to a device-resident `Q_hat`
# and device factorizations purely by dispatch. Verified device-generic on JLArrays.
_dispatch_triadic_loop_impl!(::ComputationalBackends.GPUBackend, args...; kwargs...) =
    _triadic_loop_serial!(args...; kwargs...)

# ---------------------------------------------------------------------------
# Reusable workspace
# ---------------------------------------------------------------------------

"""
    TODWorkspace(X; window, weight, noverlap, dt, Q, LHS, nmode, nfreq, isreal_data, mean_type, spectral)

Preallocated, reusable state for [`triadic_orthogonal_decomposition!`](@ref): the temporal-DFT output
`Q_hat`, the DFT plan + per-column scratch, the spatial weights and their √/1-over-√, the per-triad SVD
scratch and permute buffer, and the `L`/`T_budget` output arrays. Sized once from `X`'s dimensions and
the analysis parameters (all of which are fixed at construction); reuse it across snapshots of the same
shape so a repeat decomposition allocates only the per-triad output modes (the genuine result). Every
array field is its own type parameter (nothing hardcoded to `Array`).
"""
struct TODWorkspace{QH, SG, VC, PB, RV, L3, XM, IV, DP, SC, WW, LH, QF, SP}
    Q_hat::QH
    segment::SG
    seg_before_mean::SG
    windowed::VC
    dft_col::VC
    shifted::VC
    dft_plan::DP
    permbuf::PB
    permbuf_kl::PB
    window_vec::RV
    weight_vec::RV
    weights::RV
    sqrt_w::RV
    inv_sqrt_w::RV
    f::RV
    L::L3
    T_budget::L3
    X_mean::XM
    sc::SC
    f_idx::IV
    fk_idx::IV
    fl_idx::IV
    fn_idx::IV
    win_weight::WW
    LHS::LH
    Q::QF
    spectral::SP
    nt::Int
    nVar::Int
    nx::Int
    nDFT::Int
    nBlks::Int
    nFreq::Int
    nState::Int
    nmode_val::Int
    noverlap_val::Int
    shift::Int
    blk_mean::Bool
    isreal_data::Bool
end

function TODWorkspace(
    X::AbstractArray;
    window = nothing, weight = nothing, noverlap = nothing, dt = nothing,
    Q = _default_nonlinear, LHS = identity, nmode = nothing, nfreq = nothing,
    isreal_data = nothing, mean_type = :zero, spectral::SpectralBackends.AbstractSpectralBackend = SpectralBackends.DirectSumSpectralBackend(),
)
    # TOD's `spectral` selects the *temporal* DFT (over the leading time axis); only the uniform-1D
    # transforms apply. The scattered/spherical backends are a category error here.
    spectral isa Union{SpectralBackends.DirectSumSpectralBackend, SpectralBackends.FFTSpectralBackend} || throw(ArgumentError(
        "triadic_orthogonal_decomposition uses a temporal DFT; `spectral` must be SpectralBackends.DirectSumSpectralBackend() " *
        "or SpectralBackends.FFTSpectralBackend() (got $(typeof(spectral)))."))
    dims = size(X)
    ndims(X) >= 2 || throw(ArgumentError("X must have at least 2 dimensions (time × variables)"))
    nt = dims[1]
    nVar = dims[2]
    nx = prod(dims[3:end]; init = 1)
    isr = isreal_data === nothing ? eltype(X) <: Real : isreal_data

    RT = real(float(eltype(X)))
    CT = Complex{RT}
    (window_vec, weight_vec, noverlap_val, dt_val, nDFT, nBlks) =
        parse_parameters(nt, nx, RT; window = window, weight = weight, noverlap = noverlap, dt = dt)
    nmode_val = nmode === nothing ? nBlks : Int(nmode)
    win_weight = RT(1) / (sum(window_vec) / length(window_vec))

    X_mean = if mean_type === :zero || mean_type === :blockwise
        fill!(similar(X, CT, nVar, nx), zero(CT))                       # device-resident for device X
    elseif mean_type isa AbstractArray
        copyto!(similar(X, CT, nVar, nx), reshape(mean_type[1:nVar, :], nVar, nx))
    else
        throw(ArgumentError("mean_type must be :zero, :blockwise, or an array"))
    end
    blk_mean = mean_type === :blockwise

    (f, nFreq, _include_triad, f_idx, fk_idx, fl_idx, fn_idx) =
        frequency_axes(nDFT, dt_val; isreal_data = isr, nfreq = nfreq)   # f::Vector{RT} (dt_val::RT)

    nState = size(LHS(zeros(CT, nVar, nx, 1)), 1)

    # All compute buffers follow the input array type (`similar(X, …)`) so a device-array `X` gives a
    # device-resident workspace and the whole pipeline (temporal DFT, weighting, per-triad SVD) runs
    # on-device by dispatch. For a host `Array` this is byte-identical to the old `Array{CT}(undef)`/
    # `zeros` buffers (the fills/products overwrite them). Result arrays (`L`/`T_budget`) stay host —
    # they are the small degree×degree×mode outputs handed back to the caller.
    Q_hat = fill!(similar(X, CT, nFreq, nVar, nx, nBlks), zero(CT))
    shift = iseven(nDFT) ? nDFT ÷ 2 : (nDFT - 1) ÷ 2
    dft_plan = _temporal_dft_plan(nDFT, spectral, CT)
    segment  = similar(X, CT, nDFT, nx)
    windowed = similar(X, CT, nDFT)
    dft_col  = similar(X, CT, nDFT)
    shifted  = similar(X, CT, nDFT)
    seg_before_mean = blk_mean ? similar(X, CT, nDFT, nx) : segment

    weights = repeat(weight_vec, nState)
    sqrt_w = sqrt.(weights)          # host (shares the metadata type param); device SVD kernel moves it on-device
    inv_sqrt_w = inv.(sqrt_w)
    sc = _TriadSVDScratch(X, CT, nState * nx, nBlks)   # scratch follows the input array type
    permbuf = similar(X, CT, nx, nState, nBlks)        # recipient (n) reshape scratch
    permbuf_kl = similar(X, CT, nx, nState, nBlks)     # giver (k,l) nonlinear reshape scratch
    L = fill(RT(NaN), nFreq, nFreq, nmode_val)
    T_budget = fill(RT(NaN), nFreq, nFreq, nmode_val)

    return TODWorkspace(
        Q_hat, segment, seg_before_mean, windowed, dft_col, shifted, dft_plan, permbuf, permbuf_kl,
        window_vec, weight_vec, weights, sqrt_w, inv_sqrt_w, f, L, T_budget, X_mean, sc,
        f_idx, fk_idx, fl_idx, fn_idx, win_weight, LHS, Q, spectral,
        nt, nVar, nx, nDFT, nBlks, nFreq, nState, nmode_val, noverlap_val, shift, blk_mean, isr,
    )
end

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

"""
    triadic_orthogonal_decomposition!(ws::TODWorkspace, X; return_coefficients=false,
                                      return_auxiliary_modes=false, execution=ComputationalBackends.SerialBackend())
        -> TriadicOrthogonalDecompositionResult

In-place Triadic Orthogonal Decomposition reusing the preallocated `ws` (its `Q_hat`, DFT plan/scratch,
weights, SVD scratch, and `L`/`T_budget` buffers). `X` must match the workspace dimensions. Repeated
calls on same-shaped snapshots re-plan nothing and allocate only the per-triad output modes; the
returned result wraps the reused `L`/`T_budget` (a later call overwrites them — copy if persisting).
The analysis parameters (`window`/`weight`/`LHS`/`Q`/…) are fixed at workspace construction.
"""
function triadic_orthogonal_decomposition!(
    ws::TODWorkspace,
    X::AbstractArray;
    return_coefficients::Bool = false,
    return_auxiliary_modes::Bool = false,
    execution::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.SerialBackend(),
)
    dims = size(X)
    (dims[1] == ws.nt && dims[2] == ws.nVar && prod(dims[3:end]; init = 1) == ws.nx) ||
        throw(DimensionMismatch("X size $dims does not match workspace (nt=$(ws.nt), nVar=$(ws.nVar), nx=$(ws.nx))"))
    CT = eltype(ws.Q_hat)
    X_flat = reshape(X, ws.nt, ws.nVar, ws.nx)

    nDFT = ws.nDFT; nx = ws.nx; nVar = ws.nVar; nBlks = ws.nBlks
    blk_mean = ws.blk_mean; win_weight = ws.win_weight
    segment = ws.segment; windowed = ws.windowed; dft_col = ws.dft_col; shifted = ws.shifted
    seg_before_mean = ws.seg_before_mean; Q_hat = ws.Q_hat
    window_vec = ws.window_vec; X_mean = ws.X_mean; dft_backend = ws.spectral

    for iBlk in 1:nBlks
        offset = min((iBlk - 1) * (nDFT - ws.noverlap_val) + nDFT, ws.nt) - nDFT
        for iVar in 1:nVar
            # Extract this block's segment (device-generic broadcast; lazy views → ~0 alloc on host):
            #   segment[t,ix] = X_flat[offset+t, iVar, ix] − X_mean[iVar, ix].
            segment .= view(X_flat, offset+1:offset+nDFT, iVar, :) .- transpose(view(X_mean, iVar, :))
            if blk_mean
                copyto!(seg_before_mean, segment)
                segment .-= sum(segment; dims = 1) ./ nDFT          # subtract each column's block mean
            end
            # Windowed temporal DFT + fftshift for the whole block (per-column on host via the plan/direct
            # sum; one matmul on device — see the AbstractGPUArray method in the GPUArraysCore extension).
            _tod_dft_block!(view(Q_hat, :, iVar, :, iBlk), segment, seg_before_mean, window_vec,
                            win_weight, nDFT, ws.shift, blk_mean, dft_backend, ws.dft_plan,
                            dft_col, windowed, shifted)
        end
    end

    fill!(ws.L, real(eltype(ws.L))(NaN))
    fill!(ws.T_budget, real(eltype(ws.T_budget))(NaN))
    P = Dict{Tuple{Int,Int}, NamedTuple}()
    A_out = return_coefficients ? Dict{Tuple{Int,Int}, NamedTuple}() : nothing
    Xi_out = return_auxiliary_modes ? Dict{Tuple{Int,Int}, NamedTuple}() : nothing

    _dispatch_triadic_loop!(
        ws.L, P, ws.T_budget, A_out, Xi_out,
        ws.Q_hat, ws.f_idx, ws.fk_idx, ws.fl_idx, ws.fn_idx,
        ws.weights, ws.nBlks, ws.nFreq, ws.nState, ws.nx, ws.nmode_val,
        ws.Q, ws.LHS, return_coefficients, return_auxiliary_modes,
        ws.sc, ws.sqrt_w, ws.inv_sqrt_w, ws.permbuf, ws.permbuf_kl;
        execution = execution,
    )
    return Types.TriadicOrthogonalDecompositionResult(ws.f, ws.L, P, ws.T_budget, A_out, Xi_out)
end

"""
    triadic_orthogonal_decomposition(X; kwargs...)
        -> TriadicOrthogonalDecompositionResult

Compute the Triadic Orthogonal Decomposition of data array X.

TOD decomposes triadic (three-wave) nonlinear interactions in time-series data,
identifying coherent flow structures that optimally capture spectral momentum
transfer. It produces a mode bispectrum (singular values quantifying coupling
strength per frequency triad), convective/recipient modes, and a modal energy budget.

# Arguments
- `X::AbstractArray`: Data of size `(nt, nvar, spatial_dims...)`.
  First dimension is time, second is variable indices, remaining are spatial.

# Keyword Arguments
- `window`: Temporal window. Vector → used directly. Integer → Hamming of that length.
  `nothing` → auto Hamming (length = 2^floor(log2(nt/5)), capped at 256).
- `weight`: Spatial inner-product weight (same spatial dims as X). `nothing` → uniform.
- `noverlap`: Block overlap in snapshots. `nothing` → 50% of window length.
- `dt`: Time step between snapshots. `nothing` → 1/nDFT (frequency index output).
- `Q`: Quadratic nonlinearity function `(q1, q2) -> product`. Default: element-wise
  product with permutation matching the MATLAB reference.
- `LHS`: Left-hand side operator `q -> Lq`. Default: `identity`.
- `nmode`: Modes per triad to store. `nothing` → nBlks.
- `nfreq`: Restrict to `|l|, |k|, |n| ≤ nfreq`. `nothing` → all.
- `isreal_data`: Whether data is real (restricts bispectrum to fn ≥ 0). `nothing` → auto.
- `mean_type`: `:zero` (default), `:blockwise`, or an array (long-time mean to subtract).
- `return_coefficients::Bool=false`: Also compute expansion coefficients.
- `return_auxiliary_modes::Bool=false`: Also compute donor/catalyst modes.
- `spectral::SpectralBackends.AbstractSpectralBackend=SpectralBackends.DirectSumSpectralBackend()`: temporal-DFT transform.
  `SpectralBackends.FFTSpectralBackend()` uses FFTW (much faster; requires `using FFTW`).
- `execution::ComputationalBackends.AbstractExecutionBackend=ComputationalBackends.SerialBackend()`: triad-loop parallelism.
  `ComputationalBackends.ThreadedBackend()` parallelises the triad loop (requires OhMyThreads).

# Returns
`TriadicOrthogonalDecompositionResult` containing:
- `frequencies`: Frequency vector.
- `mode_bispectrum`: Singular values per triad per mode.
- `modes`: Dict of convective/recipient mode pairs.
- `modal_energy_budget`: Energy transfer per triad per mode.
- `expansion_coefficients`: Expansion coefficients (or `nothing`).
- `auxiliary_modes`: Donor/catalyst modes (or `nothing`).

# References
- Yeung, Chu & Schmidt (2026), J. Fluid Mech. 1031, A34.
  DOI 10.1017/jfm.2026.11183
"""
function triadic_orthogonal_decomposition(
    X::AbstractArray;
    window=nothing,
    weight=nothing,
    noverlap=nothing,
    dt=nothing,
    Q=_default_nonlinear,
    LHS=identity,
    nmode=nothing,
    nfreq=nothing,
    isreal_data=nothing,
    mean_type=:zero,
    return_coefficients=false,
    return_auxiliary_modes=false,
    spectral::SpectralBackends.AbstractSpectralBackend=SpectralBackends.DirectSumSpectralBackend(),
    execution::ComputationalBackends.AbstractExecutionBackend=ComputationalBackends.SerialBackend(),
)
    ndims(X) >= 2 || throw(ArgumentError("X must have at least 2 dimensions (time × variables)"))
    ws = TODWorkspace(X; window = window, weight = weight, noverlap = noverlap, dt = dt, Q = Q,
                      LHS = LHS, nmode = nmode, nfreq = nfreq, isreal_data = isreal_data,
                      mean_type = mean_type, spectral = spectral)
    return triadic_orthogonal_decomposition!(ws, X; return_coefficients = return_coefficients,
                                             return_auxiliary_modes = return_auxiliary_modes,
                                             execution = execution)
end

# One-line show (the workspace holds a temporal-DFT FFTW plan → default show can segfault).
Base.show(io::IO, ::TODWorkspace) = print(io, "TODWorkspace(…)")
Base.show(io::IO, ::MIME"text/plain", w::TODWorkspace) = show(io, w)

end # module TriadicOrthogonalDecomposition
