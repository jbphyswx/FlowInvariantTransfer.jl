#=

    Triadic Orthogonal Decomposition

    Yeung, B., Chu, T., and Schmidt, O. T.,
    Triadic orthogonal decomposition reveals nonlinearity in fluid flows,
    J. Fluid Mech. 1031, A34, 2026.
    https://doi.org/10.1017/jfm.2026.11183

    Reference implementations:
        - MATLAB:
            * https://www.mathworks.com/matlabcentral/fileexchange/183683-triadic-orthogonal-decomposition
        - GitHub:
            * https://github.com/FlowPhysicsGroup/Triadic-Orthogonal-Decomposition

=#


module TriadicOrthogonalDecomposition

using LinearAlgebra: LinearAlgebra
using ..Types: TriadicOrthogonalDecompositionMethod,
               TriadicOrthogonalDecompositionResult,
               AbstractExecutionBackend, SerialBackend, ThreadedBackend,
               AbstractSpectralBackend, DirectSumBackend, FFTBackend

export triadic_orthogonal_decomposition, hamming_window, hann_window, tukey_window

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
    hamming_window(N) -> Vector{Float64}

Standard Hamming window of length N: w[n] = 0.54 − 0.46·cos(2πn/(N−1)).
"""
function hamming_window(N)
    return [0.54 - 0.46 * cospi(2 * (n - 1) / (N - 1)) for n in 1:N]
end

"""
    hann_window(N) -> Vector{Float64}

Hann (raised-cosine) window of length N: w[n] = ½(1 − cos(2πn/(N−1))). Tapers to zero at both
ends — lower spectral leakage than Hamming. Pass to `triadic_orthogonal_decomposition` via `window`.
"""
function hann_window(N)
    return [0.5 * (1 - cospi(2 * (n - 1) / (N - 1))) for n in 1:N]
end

"""
    tukey_window(N; α=0.5) -> Vector{Float64}

Tukey (tapered-cosine) window of length N: a flat middle with cosine tapers over a fraction `α` of
the length at each end. `α = 0` is rectangular (no taper), `α = 1` is the Hann window; intermediate
`α` trades main-lobe width against leakage.
"""
function tukey_window(N; α=0.5)
    0 <= α <= 1 || throw(ArgumentError("tukey_window: α must be in [0,1] (got $α)."))
    α == 0 && return ones(Float64, N)
    w = ones(Float64, N)
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
function parse_parameters(nt, nx; window=nothing, weight=nothing, noverlap=nothing, dt=nothing)
    # Window
    if window === nothing
        nDFT = 2^floor(Int, log2(nt / 5))
        nDFT > 256 && (nDFT = 256)
        window_vec = hamming_window(nDFT)
    elseif window isa Integer
        nDFT = Int(window)
        window_vec = hamming_window(nDFT)
    elseif window isa AbstractVector
        nDFT = length(window)
        window_vec = Vector{Float64}(window)
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
    dt_val = dt === nothing ? 1.0 / nDFT : Float64(dt)

    # Spatial weight
    if weight === nothing
        weight_vec = ones(Float64, nx)
    else
        weight_vec = vec(Float64.(weight))
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
    M = X * X'
    # Make Hermitian for eigen
    M = LinearAlgebra.Hermitian(M)
    eigen_result = LinearAlgebra.eigen(M)
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
    Q, R = LinearAlgebra.qr(Q3)
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

    # Un-weight modes
    inv_sqrt_w = 1 ./ sqrt_w
    U = U .* inv_sqrt_w
    V = V .* inv_sqrt_w

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
    FT = Float64
    nCols = size(segment, 2)

    # Apply window and normalize
    windowed = segment .* window .* (win_weight / nDFT)

    # Direct DFT
    for freq_idx in 1:nDFT
        for col in 1:nCols
            val = zero(ComplexF64)
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
function _compute_temporal_dft!(Q_hat_blk, segment, window, win_weight, nDFT, ::DirectSumBackend)
    _temporal_block_dft_direct!(Q_hat_blk, segment, window, win_weight, nDFT)
end

function _compute_temporal_dft!(Q_hat_blk, segment, window, win_weight, nDFT, ::FFTBackend)
    _temporal_block_dft_fft!(Q_hat_blk, segment, window, win_weight, nDFT)
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

# ---------------------------------------------------------------------------
# Serial triad loop
# ---------------------------------------------------------------------------

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
)
    nTriads = length(fk_idx)
    nStateNx = nState * nx

    for i in 1:nTriads
        fi_k = fk_idx[i]
        fi_l = fl_idx[i]
        fi_n = fn_idx[i]

        # Extract Fourier realizations for this triad
        # Q_hat is (nFreq, nVar, nx, nBlks)
        # LHS transforms (nVar, nx, nBlks) -> (nState, nx, nBlks)
        Q_n_raw = Q_hat[fi_n, :, :, :]    # (nVar, nx, nBlks)
        Q_k_raw = Q_hat[fi_k, :, :, :]
        Q_l_raw = Q_hat[fi_l, :, :, :]

        Q_hat_n = reshape(permutedims(LHS(Q_n_raw), (2, 1, 3)), nStateNx, nBlks)
        Q_hat_kl = reshape(Q_nonlinear(Q_k_raw, Q_l_raw), nStateNx, nBlks)

        # Core SVD
        U, s, V = triadic_svd(Q_hat_n, Q_hat_kl, weights, nBlks)

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

        # Modal energy budget: T = s .* dot(V, W .* U) for each mode
        for j in 1:nm
            # Weighted inner product of V[:,j] and U[:,j]
            T_budget[fi_l, fi_n, j] = s[j] * real(LinearAlgebra.dot(v[:, j], weights .* u[:, j]))
        end

        # Expansion coefficients
        if return_coefficients
            # A_conv = Uᴴ · (Q_hat_kl .* weights)
            # A_recip = Vᴴ · (Q_hat_n .* weights)
            A_conv = u' * (Q_hat_kl .* weights)
            A_recip = v' * (Q_hat_n .* weights)
            A_out[(fi_l, fi_n)] = (convective=A_conv, recipient=A_recip)

            # Donor and catalyst modes
            if return_auxiliary_modes
                Q_hat_l = reshape(permutedims(LHS(Q_l_raw), (2, 1, 3)), nStateNx, nBlks)
                Q_hat_k = reshape(permutedims(LHS(Q_k_raw), (2, 1, 3)), nStateNx, nBlks)

                # donor = Q̂_l · Aᴴ_recip · diag(1/s) / nBlks
                # catalyst = Q̂_k · Aᴴ_recip · diag(1/s) / nBlks
                inv_s = 1 ./ s[1:nm]
                donor_mode = Q_hat_l * A_recip' * LinearAlgebra.Diagonal(inv_s) ./ nBlks
                catalyst_mode = Q_hat_k * A_recip' * LinearAlgebra.Diagonal(inv_s) ./ nBlks

                Xi_out[(fi_l, fi_n)] = (donor=donor_mode[:, 1:nm], catalyst=catalyst_mode[:, 1:nm])
            end
        end
    end
end

# Dispatch the triad loop on the EXECUTION (parallelism) backend.
function _dispatch_triadic_loop!(args_tuple...; execution::AbstractExecutionBackend=SerialBackend(), kwargs...)
    _dispatch_triadic_loop_impl!(execution, args_tuple...; kwargs...)
end

_dispatch_triadic_loop_impl!(::SerialBackend, args...; kwargs...) =
    _triadic_loop_serial!(args...; kwargs...)

_dispatch_triadic_loop_impl!(::ThreadedBackend, args...; kwargs...) =
    _triadic_loop_threaded!(args...; kwargs...)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

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
- `spectral::AbstractSpectralBackend=DirectSumBackend()`: temporal-DFT transform.
  `FFTBackend()` uses FFTW (much faster; requires `using FFTW`).
- `execution::AbstractExecutionBackend=SerialBackend()`: triad-loop parallelism.
  `ThreadedBackend()` parallelises the triad loop (requires OhMyThreads).

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
    spectral::AbstractSpectralBackend=DirectSumBackend(),
    execution::AbstractExecutionBackend=SerialBackend(),
)
    # --- Problem dimensions ---
    dims = size(X)
    ndims(X) >= 2 || throw(ArgumentError("X must have at least 2 dimensions (time × variables)"))
    nt = dims[1]
    nVar = dims[2]
    spatial_dims = dims[3:end]
    nx = prod(spatial_dims; init=1)

    # --- Auto-detect reality ---
    if isreal_data === nothing
        isreal_data = eltype(X) <: Real
    end

    # --- Parse parameters ---
    (window_vec, weight_vec, noverlap_val, dt_val, nDFT, nBlks) =
        parse_parameters(nt, nx; window=window, weight=weight, noverlap=noverlap, dt=dt)

    # Determine number of modes to store
    nmode_val = nmode === nothing ? nBlks : Int(nmode)

    # Window correction factor
    win_weight = 1.0 / (sum(window_vec) / length(window_vec))

    # --- Handle mean subtraction ---
    X_mean = if mean_type === :zero || mean_type === :blockwise
        zeros(eltype(X), nVar, nx)
    elseif mean_type isa AbstractArray
        # Reshape provided mean to (nVar, nx)
        reshape(mean_type[1:nVar, :], nVar, nx)
    else
        throw(ArgumentError("mean_type must be :zero, :blockwise, or an array"))
    end
    blk_mean = mean_type === :blockwise

    # --- Compute temporal DFT for all blocks ---
    (f, nFreq, include_triad, f_idx, fk_idx, fl_idx, fn_idx) =
        frequency_axes(nDFT, dt_val; isreal_data=isreal_data, nfreq=nfreq)

    # Determine nState from LHS
    # Apply LHS to a dummy to determine output size
    dummy_input = zeros(ComplexF64, nVar, nx, 1)
    nState = size(LHS(dummy_input), 1)

    # Preallocate Q_hat: (nFreq, nVar, nx, nBlks)
    Q_hat = zeros(ComplexF64, nFreq, nVar, nx, nBlks)

    # Reshape X for processing: (nt, nVar, nx)
    X_flat = reshape(X, nt, nVar, nx)

    # Temporary for block DFT
    Q_hat_blk = zeros(ComplexF64, nDFT, nVar * nx)

    # Temporal-DFT transform is chosen by the spectral backend.
    dft_backend = spectral

    for iBlk in 1:nBlks
        # Time indices for this block
        offset = min((iBlk - 1) * (nDFT - noverlap_val) + nDFT, nt) - nDFT
        time_idx = (1:nDFT) .+ offset

        for iVar in 1:nVar
            # Extract segment: (nDFT, nx)
            segment = ComplexF64.(X_flat[time_idx, iVar, :]) .- transpose(X_mean[iVar, :])

            # Blockwise mean subtraction
            if blk_mean
                seg_before_mean = copy(segment)
                blk_avg = sum(segment; dims=1) ./ nDFT
                segment .-= blk_avg
            end

            # Apply window, FFT, normalize, fftshift → (nDFT,) per spatial point
            for ix in 1:nx
                seg_col = segment[:, ix]
                windowed = seg_col .* window_vec
                # DFT
                dft_col = zeros(ComplexF64, nDFT)
                if dft_backend isa FFTBackend
                    # Will be overridden by extension
                    _temporal_block_dft_fft!(dft_col, seg_col, window_vec, win_weight, nDFT)
                else
                    # Direct sum
                    for freq_k in 1:nDFT
                        val = zero(ComplexF64)
                        for t in 1:nDFT
                            phase = -2π * (freq_k - 1) * (t - 1) / nDFT
                            val += windowed[t] * exp(im * phase)
                        end
                        dft_col[freq_k] = val * (win_weight / nDFT)
                    end
                end

                # Handle blockwise mean: preserve the DC component from pre-mean data
                if blk_mean
                    windowed_pre = seg_before_mean[:, ix] .* window_vec
                    dc_val = sum(windowed_pre) * (win_weight / nDFT)
                    dft_col[1] = dc_val
                end

                # fftshift
                shift = iseven(nDFT) ? nDFT ÷ 2 : (nDFT - 1) ÷ 2
                dft_col = circshift(dft_col, shift)

                Q_hat[:, iVar, ix, iBlk] = dft_col
            end
        end
    end

    # --- Build spatial weights for weighted inner products ---
    weights = repeat(weight_vec, nState)

    # --- Preallocate output arrays ---
    L = fill(NaN, nFreq, nFreq, nmode_val)
    T_budget = fill(NaN, nFreq, nFreq, nmode_val)
    P = Dict{Tuple{Int,Int}, NamedTuple}()
    A_out = return_coefficients ? Dict{Tuple{Int,Int}, NamedTuple}() : nothing
    Xi_out = return_auxiliary_modes ? Dict{Tuple{Int,Int}, NamedTuple}() : nothing

    # --- Main triad loop ---
    _dispatch_triadic_loop!(
        L, P, T_budget, A_out, Xi_out,
        Q_hat, f_idx, fk_idx, fl_idx, fn_idx,
        weights, nBlks, nFreq, nState, nx, nmode_val,
        Q, LHS,
        return_coefficients, return_auxiliary_modes;
        execution=execution
    )

    return TriadicOrthogonalDecompositionResult(
        f,
        L,
        P,
        T_budget,
        A_out,
        Xi_out,
    )
end

end # module TriadicOrthogonalDecomposition
