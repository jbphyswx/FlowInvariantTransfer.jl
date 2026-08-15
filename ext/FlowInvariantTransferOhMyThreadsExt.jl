module FlowInvariantTransferOhMyThreadsExt

using OhMyThreads: OhMyThreads
using LinearAlgebra: LinearAlgebra
using FlowInvariantTransfer: FlowInvariantTransfer as FIT
using SpectralBackends: SpectralBackends

# ---------------------------------------------------------------------------
# Thread-parallel shell-to-shell transfer (OhMyThreads scheduler)
# ---------------------------------------------------------------------------

"""
    _shell_to_shell_threaded!(result, ws, velocity_hat, ks; dealiasing, verify_antisymmetry)

Thread-parallel shell-to-shell transfer using OhMyThreads. Overrides the core stub
dispatched by `ThreadedBackend`. The outer loop over mediator shells `m` is
parallelised; each task writes a disjoint column of `result.transfer_matrix`, so
there is no data race. Writes into `result` and `ws`, and returns `max_asym`
(matching the serial `_calculate_shell_to_shell_direct!` contract).

The mediator shells are chunked so each task builds ONE `NonlinearTermWorkspace` (+ band-field and
density scratch) and reuses it across its shells — the shared workspace cannot be reused concurrently
across threads, and a per-shell build would put full-grid allocation (GC contention, a stop-the-world
sync under threads) inside the parallel region. Mirrors the band/mode/partial threaded methods below.
"""
function FIT.ShellToShellTransfer._shell_to_shell_threaded!(
    result,
    ws,
    velocity_hat,
    ks,
    spectral;            # transform backend, passed to each per-mediator nonlinear term
    dealiasing::FIT.Types.AbstractDealiasing = FIT.Types.OrszagTwoThirds(),
    verify_antisymmetry::Bool = true,
    invariant::FIT.Types.AbstractInvariant = FIT.Types.KineticEnergy(),
    advecting_hat = velocity_hat,
)
    nd        = length(ks)
    ns        = size(velocity_hat)[1:nd]
    M         = size(velocity_hat, nd+1)   # components of the binned/carried primary field
    FT        = real(eltype(velocity_hat))
    N_sh      = size(result.transfer_matrix, 1)
    shell_idx = ws.shell_idx

    fill!(result.transfer_matrix, zero(FT))

    # Chunk the mediator shells so each task builds its workspace + scratch ONCE (per chunk) and reuses
    # them across its shells — NOT once per shell (a full-grid workspace+plan build inside the parallel
    # region puts GC contention in the hot loop). Each shell m writes only column m → race-free. The pool
    # is built serially, OUTSIDE the parallel region, exactly as the band/mode/partial methods below.
    nchunks = max(1, min(Threads.nthreads(), N_sh))
    rngs = collect(OhMyThreads.index_chunks(1:N_sh; n = nchunks))
    pool = [(nlw = FIT.Workspaces.NonlinearTermWorkspace(velocity_hat, ks; dealiasing = dealiasing),
             ûm = similar(velocity_hat),
             d  = similar(velocity_hat, FT, ns...)) for _ in 1:length(rngs)]

    OhMyThreads.tforeach(eachindex(rngs)) do ci
        p = pool[ci]
        for m in rngs[ci]
            fill!(p.ûm, zero(eltype(p.ûm)))
            for I in CartesianIndices(ns)
                shell_idx[I] == m || continue
                for c in 1:M
                    p.ûm[I, c] = velocity_hat[I, c]
                end
            end

            # 𝒩̂_m = (u·∇)f_m: full velocity (advecting_hat) advects the band-m primary field
            # (AMP 2005) — for energy gives an antisymmetric A[n,m] reducing to transfer_spectrum[n].
            FIT.NonlinearTerm.compute_nonlinear_term!(p.nlw, p.ûm, ks;
                dealiasing=dealiasing, spectral=spectral, advecting_hat=advecting_hat)

            FIT.Invariants.transfer_density!(p.d, invariant, velocity_hat, p.nlw.N̂, ks)

            for n in 1:N_sh
                s = zero(FT)
                for I in CartesianIndices(ns)
                    shell_idx[I] == n || continue
                    s += p.d[I]
                end
                result.transfer_matrix[n, m] = s
            end
        end
    end

    # Net energy gain of each shell: Σ_m T(n,m)
    for n in 1:N_sh
        s = zero(FT)
        for m in 1:N_sh
            s += result.transfer_matrix[n, m]
        end
        result.net_transfer[n] = s
    end

    # Antisymmetry check: max |T(n,m) + T(m,n)|
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
# Thread-parallel spectral flux Π(K) reduction
# ---------------------------------------------------------------------------

"""
    _spectral_flux_threaded!(result, ws, velocity_hat, N̂, ks, shell_idx; invariant)

Thread-parallel mode→shell reduction for spectral flux, overriding the core stub dispatched by
`ThreadedBackend`. The transfer density is written once, then the scatter into shells is split into
`nthreads` index chunks, each accumulating a private length-`N_sh` partial sum (race-free); the
partials are `+`-reduced. `O(Nᴰ)` total work in parallel — no `O(N_sh·Nᴰ)` per-shell rescan. (The
nonlinear-term FFT is threaded separately via the `spectral` backend / FFTW threads.)
"""
function FIT.SpectralFlux._spectral_flux_threaded!(
    result,
    ws,
    velocity_hat,
    N̂,
    ks,
    shell_idx;
    invariant::FIT.Types.AbstractInvariant = FIT.Types.KineticEnergy(),
)
    nd   = length(ks)
    ns   = size(velocity_hat)[1:nd]
    FT   = real(eltype(velocity_hat))
    N_sh = length(ws.T_spec)
    Np   = prod(ns)

    FIT.Invariants.transfer_density!(ws.transfer_density, invariant, velocity_hat, N̂, ks)
    td = ws.transfer_density

    nchunks = max(1, Threads.nthreads())
    T = OhMyThreads.tmapreduce(+, OhMyThreads.index_chunks(1:Np; n = nchunks)) do rng
        t = zeros(FT, N_sh)
        @inbounds for i in rng
            n = shell_idx[i]
            n == 0 && continue
            t[n] += td[i]
        end
        t
    end
    copyto!(ws.T_spec, T)
    return FIT.SpectralFlux._finalize_spectral_flux!(result, ws)
end

# ---------------------------------------------------------------------------
# Override TriadicOrthogonalDecomposition._triadic_loop_threaded!
# ---------------------------------------------------------------------------

"""
    _triadic_loop_threaded!(...)

Thread-parallel triad loop using OhMyThreads. Each triad is independent
(read-only Q_hat, writes to separate Dict slots), so this is embarrassingly parallel.
"""
function FIT.TriadicOrthogonalDecomposition._triadic_loop_threaded!(
    L, P, T_budget, A_out, Xi_out,
    Q_hat, f_idx, fk_idx, fl_idx, fn_idx,
    weights, nBlks, nFreq, nState, nx, nmode,
    Q_nonlinear, LHS,
    return_coefficients, return_auxiliary_modes,
    _sc, sqrt_w, inv_sqrt_w, _permbuf, _permbuf_kl,   # sqrt_w/inv_sqrt_w read-only (shared); sc/permbuf are per-chunk
)
    nTriads = length(fk_idx)
    nStateNx = nState * nx
    CT = eltype(Q_hat)
    lk = ReentrantLock()

    # Chunk the triads so each task reuses ONE set of scratch (SVD buffers + reshape buffers) across its
    # triads — exactly like the serial loop — instead of allocating per triad. The pool is built ONCE,
    # outside the parallel region (per-triad allocation inside would put GC contention in the hot loop).
    # `sqrt_w`/`inv_sqrt_w` are constant across triads → shared read-only; sc/permbuf are written → per-chunk.
    nchunks = max(1, min(Threads.nthreads(), nTriads))
    rngs = collect(OhMyThreads.index_chunks(1:nTriads; n = nchunks))
    pool = [(sc = FIT.TriadicOrthogonalDecomposition._TriadSVDScratch(Q_hat, CT, nStateNx, nBlks),
             permbuf = similar(_permbuf), permbuf_kl = similar(_permbuf_kl)) for _ in eachindex(rngs)]

    OhMyThreads.tforeach(eachindex(rngs)) do ci
        p = pool[ci]
        for i in rngs[ci]
            fi_k = fk_idx[i]; fi_l = fl_idx[i]; fi_n = fn_idx[i]

            # Views (no per-triad copy); recipient + giver reshaped into this chunk's reused buffers.
            Q_n_raw = view(Q_hat, fi_n, :, :, :)
            Q_k_raw = view(Q_hat, fi_k, :, :, :)
            Q_l_raw = view(Q_hat, fi_l, :, :, :)
            permutedims!(p.permbuf, LHS(Q_n_raw), (2, 1, 3))
            Q_hat_n = reshape(p.permbuf, nStateNx, nBlks)
            FIT.TriadicOrthogonalDecomposition._apply_nonlinear!(p.permbuf_kl, Q_nonlinear, Q_k_raw, Q_l_raw)
            Q_hat_kl = reshape(p.permbuf_kl, nStateNx, nBlks)

            # Same allocation-reusing SVD as the serial loop → threaded result is bit-identical to serial.
            U, s, V = FIT.TriadicOrthogonalDecomposition._triadic_svd_serial!(
                p.sc, sqrt_w, inv_sqrt_w, Q_hat_n, Q_hat_kl, nBlks)

            nm = min(nmode, length(s))
            u = U[:, 1:nm]     # copy out — p.sc buffers are overwritten by this chunk's next triad
            v = V[:, 1:nm]

            # L / T_budget: each triad writes a disjoint (fi_l, fi_n) slice → race-free without a lock.
            for j in 1:nm
                L[fi_l, fi_n, j] = s[j]
            end
            @inbounds for j in 1:nm
                acc = zero(eltype(u))
                for k in axes(u, 1)
                    acc += conj(v[k, j]) * weights[k] * u[k, j]
                end
                T_budget[fi_l, fi_n, j] = s[j] * real(acc)
            end
            lock(lk) do
                P[(fi_l, fi_n)] = (convective = u, recipient = v)
            end

            if return_coefficients
                A_conv = u' * (Q_hat_kl .* weights)
                A_recip = v' * (Q_hat_n .* weights)
                lock(lk) do
                    A_out[(fi_l, fi_n)] = (convective = A_conv, recipient = A_recip)
                end
                if return_auxiliary_modes
                    Q_hat_l = reshape(permutedims(LHS(Q_l_raw), (2, 1, 3)), nStateNx, nBlks)
                    Q_hat_k = reshape(permutedims(LHS(Q_k_raw), (2, 1, 3)), nStateNx, nBlks)
                    inv_s = 1 ./ s[1:nm]
                    donor_mode = Q_hat_l * A_recip' * LinearAlgebra.Diagonal(inv_s) ./ nBlks
                    catalyst_mode = Q_hat_k * A_recip' * LinearAlgebra.Diagonal(inv_s) ./ nBlks
                    lock(lk) do
                        Xi_out[(fi_l, fi_n)] = (donor = donor_mode[:, 1:nm], catalyst = catalyst_mode[:, 1:nm])
                    end
                end
            end
        end
    end
    return nothing
end

# ---------------------------------------------------------------------------
# Thread-parallel mode-to-mode transfer S(k|p)
# ---------------------------------------------------------------------------
# Overrides the core `_mode_to_mode_threaded!` stub. Threads the outer giver loop — each giver `p`
# writes a DISJOINT column `S[·, p]`, so it is embarrassingly parallel (near-linear: N_modes ≫ nthreads).
# Each chunk keeps a private partial `net` (Σ_p S[·,p]) reduced with `+`; each task builds its OWN
# single-threaded workspace + giver scratch (the shared one can't be reused concurrently, and the inner
# FFTs stay single-threaded so the giver-loop threading is never oversubscribed).
function FIT.ModeToModeTransfer._mode_to_mode_threaded!(
    result, ws, û_p, velocity_hat, ks;
    invariant::FIT.Types.AbstractInvariant = FIT.Types.KineticEnergy(),
    dealiasing::FIT.Types.AbstractDealiasing = FIT.Types.OrszagTwoThirds(),
    spectral::SpectralBackends.AbstractSpectralBackend = SpectralBackends.DirectSumSpectralBackend(),
    advecting_hat = velocity_hat,
)
    nd = length(ks)
    ns = size(velocity_hat)[1:nd]
    M  = size(velocity_hat, nd + 1)
    FT = real(eltype(velocity_hat))
    S       = result.transfer
    Nscales  = prod(ns)
    cis     = CartesianIndices(ns)
    colons  = ntuple(_ -> Colon(), nd)
    nchunks = max(1, Threads.nthreads())
    net_partial = OhMyThreads.tmapreduce(+, OhMyThreads.index_chunks(1:Nscales; n = nchunks)) do rng
        local_ws  = FIT.Workspaces.NonlinearTermWorkspace(velocity_hat, ks; dealiasing = dealiasing)
        local_ûp  = similar(velocity_hat)
        local_net = zeros(FT, ns...)
        @inbounds for i in rng
            p = cis[i]
            fill!(local_ûp, zero(eltype(local_ûp)))
            for c in 1:M
                local_ûp[p, c] = velocity_hat[p, c]
            end
            FIT.NonlinearTerm.compute_nonlinear_term!(local_ws, local_ûp, ks;
                dealiasing = dealiasing, spectral = spectral, advecting_hat = advecting_hat)
            Sp = view(S, colons..., p)
            FIT.Invariants.transfer_density!(Sp, invariant, velocity_hat, local_ws.N̂, ks)
            local_net .+= Sp
        end
        local_net
    end
    copyto!(result.net_transfer, net_partial)
    return result
end

# ---------------------------------------------------------------------------
# Thread-parallel smooth band-to-band transfer T(n,m)
# ---------------------------------------------------------------------------
# Overrides the core `_band_to_band_threaded!` stub. Threads the outer band loop — each band `m`
# writes a DISJOINT column `T[·, m]`, so it is embarrassingly parallel; each task builds its OWN
# single-threaded workspace + band-field/density scratch (inner FFTs single-threaded → no
# oversubscription). The band-weight masks `W` are shared read-only.
function FIT.BandTransfer._band_to_band_threaded!(
    T, bws, velocity_hat, ks;
    dealiasing::FIT.Types.AbstractDealiasing = FIT.Types.OrszagTwoThirds(),
    invariant::FIT.Types.AbstractInvariant = FIT.Types.KineticEnergy(),
    spectral::SpectralBackends.AbstractSpectralBackend = SpectralBackends.DirectSumSpectralBackend(),
    advecting_hat = velocity_hat,
)
    nd = length(ks)
    ns = size(velocity_hat)[1:nd]
    D  = size(velocity_hat, nd + 1)
    FT = real(eltype(velocity_hat))
    nb = length(bws.centers)
    W  = bws.W
    # Chunk the bands so each task builds its workspace + scratch ONCE (per chunk) and reuses it across
    # its bands — NOT once per band. Per-band allocation would put full-grid workspace+plan builds inside
    # the parallel region (GC contention, a stop-the-world sync under threads → threading loses at large N).
    nchunks = max(1, min(Threads.nthreads(), nb))
    rngs = collect(OhMyThreads.index_chunks(1:nb; n = nchunks))
    # Build the per-chunk pool (workspace + scratch) ONCE, serially, OUTSIDE the parallel region: doing
    # it inside would allocate full-grid workspaces + plans concurrently, and the resulting GC (a
    # stop-the-world sync under threads) contends with the compute — which made threading LOSE at large N.
    pool = [(ws = FIT.Workspaces.NonlinearTermWorkspace(velocity_hat, ks; dealiasing = dealiasing),
             fm = similar(velocity_hat), d = similar(velocity_hat, FT, ns...)) for _ in 1:length(rngs)]
    OhMyThreads.tforeach(eachindex(rngs)) do ci
        p = pool[ci]
        for m in rngs[ci]
            for c in 1:D, I in CartesianIndices(ns)
                p.fm[I, c] = W[m][I] * velocity_hat[I, c]
            end
            FIT.NonlinearTerm.compute_nonlinear_term!(p.ws, p.fm, ks;
                dealiasing = dealiasing, spectral = spectral, advecting_hat = advecting_hat)
            FIT.Invariants.transfer_density!(p.d, invariant, velocity_hat, p.ws.N̂, ks)
            for n in 1:nb
                s = zero(FT)
                for I in CartesianIndices(ns)
                    s += W[n][I] * p.d[I]
                end
                T[n, m] = s
            end
        end
    end
    return T
end

# ---------------------------------------------------------------------------
# Thread-parallel partial (decomposition-channel) fluxes
# ---------------------------------------------------------------------------
# Overrides the core `_partial_fluxes_threaded!` stub. Threads the n²-pair loop — each (sp,sq) pair
# is an independent nonlinear-term build writing DISJOINT channel keys (sk,sp,sq); each task chunk
# builds its OWN single-threaded workspace + density scratch + partial channel Dict ONCE (outside the
# parallel region, so no full-grid/plan allocation contends with compute), then the partials are merged.
function FIT.SpectralFlux._partial_fluxes_threaded!(
    channels, ws, comps, names, velocity_hat, ks, sidx, centers, Nsh;
    dealiasing::FIT.Types.AbstractDealiasing = FIT.Types.OrszagTwoThirds(),
    spectral::SpectralBackends.AbstractSpectralBackend = SpectralBackends.DirectSumSpectralBackend(),
)
    FT = real(eltype(velocity_hat)); nd = length(ks); ns = size(velocity_hat)[1:nd]
    prs = [(sp, sq) for sp in names for sq in names]
    np  = length(prs)
    nchunks = max(1, min(Threads.nthreads(), np))
    rngs = collect(OhMyThreads.index_chunks(1:np; n = nchunks))
    pool = [(ws = FIT.Workspaces.NonlinearTermWorkspace(velocity_hat, ks; dealiasing = dealiasing),
             d  = similar(velocity_hat, FT, ns...),
             ch = Dict{NTuple{3,Symbol}, FIT.Types.SpectralFluxResult}()) for _ in 1:length(rngs)]
    OhMyThreads.tforeach(eachindex(rngs)) do ci
        p = pool[ci]
        for pi in rngs[ci]
            sp, sq = prs[pi]
            FIT.NonlinearTerm.compute_nonlinear_term!(p.ws, comps[sq], ks;
                advecting_hat = comps[sp], dealiasing = dealiasing, spectral = spectral)
            for sk in names
                FIT.Invariants.transfer_density!(p.d, FIT.Types.KineticEnergy(), comps[sk], p.ws.N̂, ks)
                p.ch[(sk, sp, sq)] = FIT.SpectralFlux._partial_binflux(p.d, sidx, centers, Nsh)
            end
        end
    end
    for p in pool
        merge!(channels, p.ch)
    end
    return channels
end

end # module FlowInvariantTransferOhMyThreadsExt
