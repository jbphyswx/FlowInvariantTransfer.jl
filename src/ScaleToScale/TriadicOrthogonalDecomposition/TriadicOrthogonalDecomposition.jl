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

# Temporal-DFT plan (+ scratch) for a whole `(nDFT, nx)` block, built once per workspace: `nothing` in
# the core; the FFTW extension returns one plan batched over the time axis, real-input (r2c) for real
# data.
_temporal_dft_plan(nDFT, nx, isreal_data, backend, ::Type, proto) = nothing

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

Distributed triad loop. Stub overridden by the Distributed extension.
"""
function _triadic_loop_distributed!(args...; kwargs...)
    throw(ArgumentError(
        "Distributed triadic decomposition requires Distributed. Run `using Distributed` to load the extension."))
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

Parse and validate the spectral estimation parameters, filling the defaults listed in
[`triadic_orthogonal_decomposition`](@ref).
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

# Whole-block temporal DFT of `segment` (nDFT × nx) into `Q_blk` (nDFT × nx), windowed + fftshifted,
# with the `blk_mean` DC (k=0) correction taken from the un-centered `seg_before_mean`. Host arrays go
# through the plan or a per-column direct sum; the GPUArraysCore extension adds an `AbstractGPUArray`
# method that runs the block as one matmul.
function _tod_dft_block!(Q_blk, segment, seg_before_mean, window, win_weight, nDFT, shift, blk_mean,
                         backend, plan, dft_col, windowed, shifted)
    if plan !== nothing
        # One batched transform for the whole block through the preset plan (FFTW extension).
        return _temporal_block_dft_fft!(Q_blk, segment, seg_before_mean, window, win_weight, nDFT,
                                        shift, blk_mean, plan)
    end
    CT = eltype(Q_blk); RT = real(CT); nx = size(segment, 2)
    for ix in 1:nx
        @inbounds for t in 1:nDFT
            windowed[t] = segment[t, ix] * window[t]
        end
        @inbounds for freq_k in 1:nDFT
            val = zero(CT)
            for t in 1:nDFT
                # e^{-2πi(k−1)(t−1)/N}, the exponent reduced mod N in integers first.
                val += windowed[t] * cispi(-2 * RT(mod((freq_k - 1) * (t - 1), nDFT)) / RT(nDFT))
            end
            dft_col[freq_k] = val * (win_weight / nDFT)
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

Default quadratic nonlinearity: the elementwise product `q1 .* q2` of two `(nVar, nx, nBlks)` arrays
with its first two dimensions swapped, `(nx, nVar, nBlks)`.
"""
function _default_nonlinear(q1, q2)
    product = q1 .* q2
    return permutedims(product, (2, 1, ntuple(i -> i + 2, ndims(product) - 2)...))
end

# Quadratic nonlinearity into the reused `(k, l)` buffer `out` (nx, nState, nBlks): the default
# `permute(q1 .* q2, (2,1,3))` is fused into `out`; a user-supplied `Q` is evaluated and copied in.
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

# Per-triad SVD scratch, reused across triads: the products (Xw, RᵀX, YYᴴ, YᴴU, QU) write into these
# buffers through `mul!`. Sizes follow `r = min(nStateNx, nBlks)`, the QR/SVD rank bound. The `r × r`
# Hermitian eig runs on the host for either array kind, so `λ`, `sqrt_λ` and `perm` are host vectors.
struct _TriadSVDScratch{M<:AbstractMatrix, RV<:AbstractVector, PV<:AbstractVector{Int}}
    Xw::M       # nBlks × nStateNx      — (Q̂_n·√w)ᴴ / nBlks
    Q3::M       # nStateNx × nBlks      — Q̂_kl·√w   (consumed by qr!)
    Qmat::M     # nStateNx × r          — thin Q (materialized via lmul! into this buffer)
    Y::M        # r × nStateNx          — R · Xw
    Mmat::M     # r × r                 — Y · Yᴴ
    Uperm::M    # r × r                 — eigenvectors, columns reordered by descending eigenvalue
    Ubuf::M     # nStateNx × r          — convective modes (view [:,1:nz] returned)
    Vbuf::M     # nStateNx × r          — recipient modes  (view [:,1:nz] returned)
    λ::RV       # r                     — eigenvalues
    sqrt_λ::RV  # r                     — singular values, descending (view [1:nz] returned)
    perm::PV    # r                     — ordering by descending eigenvalue
end
# The matrix buffers follow `proto`'s array type (the temporal-DFT `Q_hat`), so a device input gets
# device buffers and the per-triad `qr!`/`mul!` dispatch to the device libraries.
function _TriadSVDScratch(proto::AbstractArray, ::Type{CT}, nStateNx::Int, nBlks::Int) where {CT}
    r = min(nStateNx, nBlks)
    RT = real(CT)
    z(dims...) = fill!(similar(proto, CT, dims...), zero(CT))
    return _TriadSVDScratch(
        z(nBlks, nStateNx), z(nStateNx, nBlks), z(nStateNx, r),
        z(r, nStateNx), z(r, r), z(r, r),
        z(nStateNx, r), z(nStateNx, r),
        zeros(RT, r), zeros(RT, r), zeros(Int, r),
    )
end

# Per-triad SVD of the weighted cross-correlation `(Q̂_kl√w)(Q̂_n√w)ᴴ/nBlks`: a thin QR of `Q̂_kl√w`, then
# the method of snapshots on `R·Xw` (eig of its Gram matrix), every large product through `mul!` into
# `sc`. Returns views into `sc.Ubuf`/`sc.Vbuf`, overwritten by the next triad; the caller copies out
# the `[:,1:nm]` modes it stores.
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
    # Q3 = QR; thin Q into sc.Qmat via lmul! (equal to Matrix(F.Q)); Y = R · Xw
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
    Uev = eig.vectors
    λ = sc.λ; sqrt_λ = sc.sqrt_λ; perm = sc.perm
    @inbounds for i in 1:r
        λ[i] = real(eig.values[i])
    end
    sortperm!(perm, λ; rev = true)
    # Eigenvector columns by descending eigenvalue into sc.Uperm. A small negative eigenvalue is
    # numerical noise and clamps to zero before the square root.
    Uperm = sc.Uperm
    @inbounds for j in 1:r
        pj = perm[j]
        sqrt_λ[j] = sqrt(max(λ[pj], zero(RT)))
        for i in 1:r
            Uperm[i, j] = Uev[i, pj]
        end
    end
    nz = count(>(eps(RT) * 100), sqrt_λ)
    if nz == 0
        return view(sc.Ubuf, :, 1:0), view(sqrt_λ, 1:0), view(sc.Vbuf, :, 1:0)
    end
    U_trunc = view(Uperm, :, 1:nz)
    s = view(sqrt_λ, 1:nz)
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

# `A` on the device of `proto` (host no-op); the GPUArraysCore extension adds the `AbstractGPUArray`
# method. Puts the weights and `1/s` on the modes' device for the triad loop's reductions.
_dev_like(proto, A) = A

# Modal energy budget T_j = s_j · Re⟨v_j, W u_j⟩, a host scalar accumulator; the GPUArraysCore extension
# adds the `wdev::AbstractGPUArray` method.
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

# Concrete per-triad result types. `MT` is the matrix type the modes come back as — taken from the
# temporal-DFT output's array kind, so a device-array input yields device-resident mode types.
_modes_nt(::Type{MT}) where {MT} = NamedTuple{(:convective, :recipient), Tuple{MT, MT}}
_coeffs_nt(::Type{MT}) where {MT} = NamedTuple{(:convective, :recipient), Tuple{MT, MT}}
_aux_nt(::Type{MT}) where {MT} = NamedTuple{(:donor, :catalyst), Tuple{MT, MT}}

# `(fl, fn)` → per-triad entry, the caller-facing form, built once from the slot vector.
function _triad_dict(slots::Vector{NT}, fl_idx, fn_idx) where {NT}
    d = Dict{Tuple{Int,Int}, NT}()
    sizehint!(d, length(slots))
    @inbounds for i in eachindex(slots)
        d[(fl_idx[i], fn_idx[i])] = slots[i]
    end
    return d
end
_triad_dict(::Nothing, fl_idx, fn_idx) = nothing

"""
    _triadic_loop_serial!(L, P, T_budget, A_out, Xi_out,
                          Q_hat, f_idx, fk_idx, fl_idx, fn_idx,
                          weights, nBlks, nFreq, nState, nx, nmode,
                          Q_nonlinear, LHS,
                          return_coefficients, return_auxiliary_modes,
                          sc, sqrt_w, inv_sqrt_w, permbuf, permbuf_kl)

Serial loop over all included triads. Computes mode bispectrum, modes, energy budget, and optionally
expansion coefficients and auxiliary modes, reusing the `TODWorkspace` scratch passed in.
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

        # This triad's Fourier realizations, contiguous slices of Q_hat (nVar, nx, nBlks, nFreq); `LHS`
        # maps (nVar, nx, nBlks) -> (nState, nx, nBlks).
        Q_n_raw = view(Q_hat, :, :, :, fi_n)
        Q_k_raw = view(Q_hat, :, :, :, fi_k)
        Q_l_raw = view(Q_hat, :, :, :, fi_l)

        permutedims!(permbuf, LHS(Q_n_raw), (2, 1, 3))   # recipient reshape into the reused buffer
        Q_hat_n = reshape(permbuf, nStateNx, nBlks)
        _apply_nonlinear!(permbuf_kl, Q_nonlinear, Q_k_raw, Q_l_raw)   # (k, l) product into the reused buffer
        Q_hat_kl = reshape(permbuf_kl, nStateNx, nBlks)

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
        P[i] = (convective = u, recipient = v)

        # Modal energy budget T_j = s_j · Re⟨v_j, W u_j⟩
        _tod_modal_budget!(T_budget, fi_l, fi_n, u, v, s, nm, wdev)

        # Expansion coefficients
        if return_coefficients
            A_conv = u' * (Q_hat_kl .* wdev)          # A_conv  = Uᴴ · (Q̂_kl .* W)
            A_recip = v' * (Q_hat_n .* wdev)          # A_recip = Vᴴ · (Q̂_n  .* W)
            A_out[i] = (convective = A_conv, recipient = A_recip)

            # Donor and catalyst modes
            if return_auxiliary_modes
                Q_hat_l = reshape(permutedims(LHS(Q_l_raw), (2, 1, 3)), nStateNx, nBlks)
                Q_hat_k = reshape(permutedims(LHS(Q_k_raw), (2, 1, 3)), nStateNx, nBlks)

                # donor/catalyst = Q̂_{l,k} · Aᴴ_recip · diag(1/s) / nBlks, with diag(1/s) a broadcast of a
                # row of 1/s on the modes' device.
                inv_s_row = _dev_like(Q_hat, reshape(1 ./ s[1:nm], 1, nm))
                donor_mode = (Q_hat_l * A_recip') .* inv_s_row ./ nBlks
                catalyst_mode = (Q_hat_k * A_recip') .* inv_s_row ./ nBlks

                Xi_out[i] = (donor = donor_mode[:, 1:nm], catalyst = catalyst_mode[:, 1:nm])
            end
        end
    end
end

# One triad `i`'s outputs with its own scratch, the unit of work of the distributed loop. It runs
# `_triadic_svd_serial!` and `_apply_nonlinear!` as the serial loop does, so results are bit-identical.
function _triad_result(i, Q_hat, fk_idx, fl_idx, fn_idx, weights, sqrt_w, inv_sqrt_w,
                       nBlks, nState, nx, nmode, Q_nonlinear, LHS,
                       return_coefficients, return_auxiliary_modes)
    CT = eltype(Q_hat)
    nStateNx = nState * nx
    fi_k = fk_idx[i]; fi_l = fl_idx[i]; fi_n = fn_idx[i]
    Q_n_raw = view(Q_hat, :, :, :, fi_n)
    Q_k_raw = view(Q_hat, :, :, :, fi_k)
    Q_l_raw = view(Q_hat, :, :, :, fi_l)
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
    sv = s[1:nm]
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
    return (i = i, fi_l = fi_l, fi_n = fi_n, nm = nm, s = sv, Tb = Tb, u = u, v = v,
            A_conv = A_conv, A_recip = A_recip, donor = donor, catalyst = catalyst)
end

# Dispatch the triad loop on the execution backend.
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

# GPU: the serial loop is device-generic. Each per-triad primitive (temporal DFT, weighting,
# `qr!`/`mul!`, the Hermitian eig, the reshape scratch) dispatches on the array type, with the
# `AbstractGPUArray` methods in the GPUArraysCore extension; the `r × r` eig runs on the host.
_dispatch_triadic_loop_impl!(::ComputationalBackends.GPUBackend, args...; kwargs...) =
    _triadic_loop_serial!(args...; kwargs...)

# ---------------------------------------------------------------------------
# Reusable workspace
# ---------------------------------------------------------------------------

"""
    TODWorkspace(X; window, weight, noverlap, dt, Q, LHS, nmode, nfreq, isreal_data, mean_type, transform)

Preallocated, reusable state for [`triadic_orthogonal_decomposition!`](@ref): the temporal-DFT output
`Q_hat`, `(nVar, nx, nBlks, nFreq)` so that a frequency's realizations are contiguous, one block's
transform `Q_blk` `(nDFT, nx)` before it is stored there, the DFT plan + per-column scratch, the spatial
weights and their √/1-over-√, the per-triad SVD scratch and permute buffer, and the `L`/`T_budget` output
arrays. Sized from `X`'s dimensions and the analysis parameters, all fixed at construction; reused
across snapshots of the same shape, a repeat decomposition allocates only the per-triad output modes.
"""
struct TODWorkspace{QH, QB, SG, VC, PB, RV, L3, XM, IV, DP, SC, WW, LH, QF, SP}
    Q_hat::QH
    Q_blk::QB
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
    transform::SP
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
    zero_mean::Bool
    isreal_data::Bool
end

function TODWorkspace(
    X::AbstractArray;
    window = nothing, weight = nothing, noverlap = nothing, dt = nothing,
    Q = _default_nonlinear, LHS = identity, nmode = nothing, nfreq = nothing,
    isreal_data = nothing, mean_type = :zero, transform::SpectralBackends.AbstractSpectralBackend = SpectralBackends.AutoSpectralBackend(),
)
    # `transform` selects the temporal DFT over the leading time axis: the direct sum or the FFT.
    transform = Types.resolve_transform(transform)
    transform isa Union{SpectralBackends.DirectSumSpectralBackend, SpectralBackends.FFTSpectralBackend} || throw(ArgumentError(
        "triadic_orthogonal_decomposition uses a temporal DFT; `transform` must be SpectralBackends.DirectSumSpectralBackend() " *
        "or SpectralBackends.FFTSpectralBackend() (got $(typeof(transform)))."))
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

    # The block segment is `X` minus this mean, real when both are, and that selects the real-input
    # (r2c) temporal transform. `isreal_data` sets only the triad mask and may be set for complex data.
    real_x = eltype(X) <: Real
    X_mean = if mean_type === :zero || mean_type === :blockwise
        MET = real_x ? RT : CT
        fill!(similar(X, MET, nVar, nx), zero(MET))                     # device-resident for device X
    elseif mean_type isa AbstractArray
        MET = (real_x && eltype(mean_type) <: Real) ? RT : CT
        copyto!(similar(X, MET, nVar, nx), reshape(mean_type[1:nVar, :], nVar, nx))
    else
        throw(ArgumentError("mean_type must be :zero, :blockwise, or an array"))
    end
    blk_mean = mean_type === :blockwise

    (f, nFreq, _include_triad, f_idx, fk_idx, fl_idx, fn_idx) =
        frequency_axes(nDFT, dt_val; isreal_data = isr, nfreq = nfreq)   # f::Vector{RT} (dt_val::RT)

    nState = size(LHS(zeros(CT, nVar, nx, 1)), 1)

    # Compute buffers follow `X`'s array type, so a device `X` runs the temporal DFT, weighting and
    # per-triad SVD on-device. The `L`/`T_budget` outputs (nFreq × nFreq × nmode) stay on the host.
    Q_hat = fill!(similar(X, CT, nVar, nx, nBlks, nFreq), zero(CT))
    Q_blk = similar(X, CT, nDFT, nx)
    shift = iseven(nDFT) ? nDFT ÷ 2 : (nDFT - 1) ÷ 2
    ET = (real_x && eltype(X_mean) <: Real) ? RT : CT     # block-segment element type
    segment  = similar(X, ET, nDFT, nx)
    seg_before_mean = blk_mean ? similar(X, ET, nDFT, nx) : segment
    dft_plan = _temporal_dft_plan(nDFT, nx, ET <: Real, transform, RT, X)
    # The transform is selected by the presence of a plan, so a resolved FFT backend without one means
    # the FFTW extension is absent.
    (transform isa SpectralBackends.FFTSpectralBackend && dft_plan === nothing) && throw(ArgumentError(
        "transform = SpectralBackends.FFTSpectralBackend() needs FFTW for the temporal transform. " *
        "Run `using FFTW`, or pass SpectralBackends.DirectSumSpectralBackend() for the O(nDFT²) reference."))
    windowed = similar(X, CT, nDFT)                       # direct-sum per-column scratch
    dft_col  = similar(X, CT, nDFT)
    shifted  = similar(X, CT, nDFT)

    weights = repeat(weight_vec, nState)
    sqrt_w = sqrt.(weights)          # host; the device SVD moves it on-device
    inv_sqrt_w = inv.(sqrt_w)
    sc = _TriadSVDScratch(X, CT, nState * nx, nBlks)   # scratch follows the input array type
    permbuf = similar(X, CT, nx, nState, nBlks)        # recipient (n) reshape scratch
    permbuf_kl = similar(X, CT, nx, nState, nBlks)     # (k, l) nonlinear reshape scratch
    L = fill(RT(NaN), nFreq, nFreq, nmode_val)
    T_budget = fill(RT(NaN), nFreq, nFreq, nmode_val)

    return TODWorkspace(
        Q_hat, Q_blk, segment, seg_before_mean, windowed, dft_col, shifted, dft_plan, permbuf, permbuf_kl,
        window_vec, weight_vec, weights, sqrt_w, inv_sqrt_w, f, L, T_budget, X_mean, sc,
        f_idx, fk_idx, fl_idx, fn_idx, win_weight, LHS, Q, transform,
        nt, nVar, nx, nDFT, nBlks, nFreq, nState, nmode_val, noverlap_val, shift, blk_mean,
        mean_type === :zero, isr,
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
    windowed = ws.windowed; dft_col = ws.dft_col; shifted = ws.shifted
    seg_before_mean = ws.seg_before_mean; Q_hat = ws.Q_hat
    window_vec = ws.window_vec; X_mean = ws.X_mean; dft_backend = ws.transform

    for iBlk in 1:nBlks
        offset = min((iBlk - 1) * (nDFT - ws.noverlap_val) + nDFT, ws.nt) - nDFT
        for iVar in 1:nVar
            # segment[t,ix] = X_flat[offset+t, iVar, ix] − X_mean[iVar, ix]; with a zero mean the
            # transform reads the slice of `X` directly.
            blk = view(X_flat, offset+1:offset+nDFT, iVar, :)
            segment = ws.zero_mean ? blk : ws.segment
            if !ws.zero_mean
                segment .= blk .- transpose(view(X_mean, iVar, :))
            end
            if blk_mean
                copyto!(seg_before_mean, segment)
                segment .-= sum(segment; dims = 1) ./ nDFT          # subtract each column's block mean
            end
            # Windowed temporal DFT + fftshift of the block, stored frequency-last.
            _tod_dft_block!(ws.Q_blk, segment, seg_before_mean, window_vec,
                            win_weight, nDFT, ws.shift, blk_mean, dft_backend, ws.dft_plan,
                            dft_col, windowed, shifted)
            view(Q_hat, iVar, :, iBlk, :) .= transpose(ws.Q_blk)
        end
    end

    fill!(ws.L, real(eltype(ws.L))(NaN))
    fill!(ws.T_budget, real(eltype(ws.T_budget))(NaN))
    # Per-triad slots indexed by triad number, each written once, so a parallel loop needs no lock; the
    # `(fl, fn)`-keyed `Dict`s are built from them after the loop.
    nTriads = length(ws.fk_idx)
    MT = typeof(similar(ws.Q_hat, CT, 0, 0))   # the array kind the per-triad modes come back as
    P = Vector{_modes_nt(MT)}(undef, nTriads)
    A_out = return_coefficients ? Vector{_coeffs_nt(MT)}(undef, nTriads) : nothing
    Xi_out = return_auxiliary_modes ? Vector{_aux_nt(MT)}(undef, nTriads) : nothing

    _dispatch_triadic_loop!(
        ws.L, P, ws.T_budget, A_out, Xi_out,
        ws.Q_hat, ws.f_idx, ws.fk_idx, ws.fl_idx, ws.fn_idx,
        ws.weights, ws.nBlks, ws.nFreq, ws.nState, ws.nx, ws.nmode_val,
        ws.Q, ws.LHS, return_coefficients, return_auxiliary_modes,
        ws.sc, ws.sqrt_w, ws.inv_sqrt_w, ws.permbuf, ws.permbuf_kl;
        execution = execution,
    )
    return Types.TriadicOrthogonalDecompositionResult(
        ws.f, ws.L, _triad_dict(P, ws.fl_idx, ws.fn_idx), ws.T_budget,
        _triad_dict(A_out, ws.fl_idx, ws.fn_idx), _triad_dict(Xi_out, ws.fl_idx, ws.fn_idx))
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
- `Q`: Quadratic nonlinearity `(q1, q2) -> product` of two `(nVar, nx, nBlks)` realizations, `q1` at
  `f_k = f_n − f_l` and `q2` at `f_l`, returning `(nx, nState, nBlks)`. Default: the elementwise product,
  `Q(q1, q2)[ix, iv, b] = q1[iv, ix, b]·q2[iv, ix, b]`, which takes no derivative.
- `LHS`: Left-hand side operator `q -> Lq`. Default: `identity`.
- `nmode`: Modes per triad to store. `nothing` → nBlks.
- `nfreq`: Restrict to `|l|, |k|, |n| ≤ nfreq`. `nothing` → all.
- `isreal_data`: Whether data is real (restricts bispectrum to fn ≥ 0). `nothing` → auto.
- `mean_type`: `:zero` (default), `:blockwise`, or an array (long-time mean to subtract).
- `return_coefficients::Bool=false`: Also compute expansion coefficients.
- `return_auxiliary_modes::Bool=false`: Also compute donor/catalyst modes.
- `transform::SpectralBackends.AbstractSpectralBackend=SpectralBackends.AutoSpectralBackend()`: temporal-DFT
  transform, resolving to `FFTSpectralBackend()` when FFTW is loaded and to the `O(nDFT²)`
  `DirectSumSpectralBackend()` reference otherwise.
- `execution::ComputationalBackends.AbstractExecutionBackend=ComputationalBackends.SerialBackend()`: triad-loop parallelism.
  `ComputationalBackends.ThreadedBackend()` parallelises the triad loop (requires OhMyThreads).

# Returns
`TriadicOrthogonalDecompositionResult` containing:
- `frequencies`: Frequency vector.
- `mode_bispectrum`: Singular values per triad per mode.
- `modes`: Dict of convective/recipient mode pairs.
- `modal_energy_budget`: `λ_m Re⟨φ_m, W ψ_m⟩` per triad per mode, `φ_m` the recipient and `ψ_m` the
  convective mode. It is the modal energy transfer of Yeung, Chu & Schmidt when `Q` is the quadratic
  term of the flow's evolution equation (for velocity data, its advective term with the sign it
  carries there); with the default `Q` it is that same projection of the elementwise product. Summed
  over every mode it is `Re⟨q̂_n, W·Q(q̂_k, q̂_l)⟩` averaged over the blocks, and with
  `Q(q₁, q₂) = −(q₁·∇)q₂` on a divergence-free periodic field the sums at `(f_l, f_n)` and
  `(f_n, f_l)` cancel (their Eq. 38).
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
    transform::SpectralBackends.AbstractSpectralBackend=SpectralBackends.AutoSpectralBackend(),
    execution::ComputationalBackends.AbstractExecutionBackend=ComputationalBackends.SerialBackend(),
)
    ndims(X) >= 2 || throw(ArgumentError("X must have at least 2 dimensions (time × variables)"))
    ws = TODWorkspace(X; window = window, weight = weight, noverlap = noverlap, dt = dt, Q = Q,
                      LHS = LHS, nmode = nmode, nfreq = nfreq, isreal_data = isreal_data,
                      mean_type = mean_type, transform = transform)
    return triadic_orthogonal_decomposition!(ws, X; return_coefficients = return_coefficients,
                                             return_auxiliary_modes = return_auxiliary_modes,
                                             execution = execution)
end

Base.show(io::IO, ::TODWorkspace) = print(io, "TODWorkspace(…)")
Base.show(io::IO, ::MIME"text/plain", w::TODWorkspace) = show(io, w)

end # module TriadicOrthogonalDecomposition
