module FlowInvariantTransferOhMyThreadsExt

using OhMyThreads: OhMyThreads
using LinearAlgebra: LinearAlgebra
using FlowInvariantTransfer: FlowInvariantTransfer as FIT
using FlowTransformBindings: FlowTransformBindings as FTB
using SpectralBackends: SpectralBackends

# Each threaded method below chunks its loop and gives every chunk its own single-threaded workspace,
# built serially before the parallel region unless noted; the tasks write disjoint parts of the output.

# ---------------------------------------------------------------------------
# Thread-parallel shell-to-shell transfer (OhMyThreads scheduler)
# ---------------------------------------------------------------------------

"""
    _shell_to_shell_threaded!(result, ws, velocity_hat, ks, spectral; dealiasing, invariant,
                              advecting_hat) -> result

Thread-parallel shell-to-shell transfer, the `ThreadedBackend` method of the core stub. The mediator
shells `m` are split into chunks, each with its own `NonlinearTermWorkspace` and band-field scratch,
and shell `m` writes only column `m` of `result.transfer_matrix`. Writes `result.transfer_matrix` and
`result.net_transfer`, as `_calculate_shell_to_shell_direct!` does.
"""
function FIT.ShellToShellTransfer._shell_to_shell_threaded!(
    result,
    ws,
    velocity_hat,
    ks,
    spectral;            # transform backend, passed to each per-mediator nonlinear term
    dealiasing::FIT.Types.AbstractDealiasing = FIT.Types.OrszagTwoThirds(),
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

    nchunks = max(1, min(Threads.nthreads(), N_sh))
    rngs = collect(OhMyThreads.index_chunks(1:N_sh; n = nchunks))
    pool = [(nlw = FIT.Workspaces.NonlinearTermWorkspace(velocity_hat, ks; dealiasing = dealiasing),
             ûm = similar(velocity_hat),
             col = zeros(FT, N_sh)) for _ in 1:length(rngs)]

    OhMyThreads.tforeach(eachindex(rngs)) do ci
        p = pool[ci]
        FIT.NonlinearTerm.synthesize_advecting!(p.nlw, advecting_hat, ks; dealiasing = dealiasing, transform = spectral)
        for m in rngs[ci]
            fill!(p.ûm, zero(eltype(p.ûm)))
            for I in CartesianIndices(ns)
                shell_idx[I] == m || continue
                for c in 1:M
                    p.ûm[I, c] = velocity_hat[I, c]
                end
            end

            # 𝒩̂_m = (u·∇)f_m: the full velocity (advecting_hat) advects the band-m primary field
            # (Alexakis–Mininni–Pouquet 2005); for energy A[n,m] is antisymmetric and reduces to
            # transfer_spectrum[n].
            FIT.NonlinearTerm.compute_advection!(p.nlw, p.ûm, ks; dealiasing = dealiasing, transform = spectral)

            # Column m in one pass over the modes, each mode adding to its own shell.
            FIT.Invariants.transfer_density_scatter!(p.col, invariant, velocity_hat, p.nlw.N̂, ks, shell_idx)
            @inbounds for n in 1:N_sh
                result.transfer_matrix[n, m] = p.col[n]
            end
        end
    end

    # net_transfer[n] = Σ_m T(n,m) = −dE_n/dt
    for n in 1:N_sh
        s = zero(FT)
        for m in 1:N_sh
            s += result.transfer_matrix[n, m]
        end
        result.net_transfer[n] = s
    end

    return result
end

# ---------------------------------------------------------------------------
# Thread-parallel spectral flux Π(K) reduction
# ---------------------------------------------------------------------------

"""
    _spectral_flux_threaded!(result, ws, velocity_hat, N̂, ks, shell_idx; invariant)

Thread-parallel mode→shell reduction for spectral flux, overriding the core stub dispatched by
`ThreadedBackend`. The modes are split into `nthreads` index chunks, each forming its modes' transfer
density and accumulating a private length-`N_sh` partial sum (race-free); the partials are
`+`-reduced, `O(Nᴰ)` work in total. The nonlinear-term FFTs are threaded by their plans.
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
    M    = size(velocity_hat, nd + 1)
    FT   = real(eltype(velocity_hat))
    N_sh = length(ws.T_spec)
    Np   = prod(ns)

    nchunks = max(1, Threads.nthreads())
    T = OhMyThreads.tmapreduce(+, OhMyThreads.index_chunks(1:Np; n = nchunks)) do rng
        t = zeros(FT, N_sh)
        cis = CartesianIndices(ns)
        @inbounds for i in rng
            n = shell_idx[i]
            n == 0 && continue
            I = cis[i]
            t[n] += FTB.mode_fold(ks, I) * FIT.Invariants._density_at(invariant, velocity_hat, N̂, ks, I, M)
        end
        t
    end
    copyto!(ws.T_spec, T)
    return FIT.SpectralFlux._finalize_spectral_flux!(result, ws)
end

# ---------------------------------------------------------------------------
# Thread-parallel spectral-flux batch (snapshots are the outer parallel axis)
# ---------------------------------------------------------------------------
# Overrides the core `_spectral_flux_batch_threaded!` stub. Each chunk reuses one `SpectralFluxWorkspace`
# across its snapshots with single-threaded transforms, bit-identical to the serial batch.
function FIT.SpectralFlux._spectral_flux_batch_threaded!(
    results, velocity_hats, ks, shell_idx;
    binning, dealiasing, invariant, transform, geometry,
)
    n  = length(velocity_hats)
    û1 = first(velocity_hats)
    nchunks = max(1, min(Threads.nthreads(), n))
    rngs = collect(OhMyThreads.index_chunks(1:n; n = nchunks))
    pool = [FIT.Workspaces.SpectralFluxWorkspace(û1, ks, binning; geometry = geometry, dealiasing = dealiasing)
            for _ in 1:length(rngs)]
    OhMyThreads.tforeach(eachindex(rngs)) do ci
        ws = pool[ci]
        for i in rngs[ci]
            FIT.SpectralFlux.calculate_spectral_flux!(results[i], ws, velocity_hats[i], ks, shell_idx;
                dealiasing = dealiasing, invariant = invariant, transform = transform,
                execution = FIT.ComputationalBackends.SerialBackend())
        end
    end
    return results
end

# ---------------------------------------------------------------------------
# Thread-parallel shell-to-shell batch (snapshots are the outer parallel axis)
# ---------------------------------------------------------------------------
# Overrides `_shell_to_shell_batch_threaded!`. Each chunk reuses one ShellToShellWorkspace across its
# snapshots with a serial transform, bit-identical to the serial batch.
function FIT.ShellToShellTransfer._shell_to_shell_batch_threaded!(
    results, velocity_hats, ks, centers, edges, N_sh;
    binning, dealiasing, invariant, transform, geometry,
)
    n  = length(velocity_hats)
    û1 = first(velocity_hats)
    FT = real(eltype(û1))
    nchunks = max(1, min(Threads.nthreads(), n))
    rngs = collect(OhMyThreads.index_chunks(1:n; n = nchunks))
    pool = [FIT.Workspaces.ShellToShellWorkspace(û1, ks, binning; geometry = geometry, dealiasing = dealiasing)
            for _ in 1:length(rngs)]
    OhMyThreads.tforeach(eachindex(rngs)) do ci
        ws = pool[ci]
        for i in rngs[ci]
            vh = velocity_hats[i]
            res = FIT.Types.ShellToShellResult(centers, edges, Matrix{FT}(undef, N_sh, N_sh), Vector{FT}(undef, N_sh))
            FIT.ShellToShellTransfer._calculate_shell_to_shell!(res, ws, vh, ks,
                FIT.ComputationalBackends.SerialBackend(), transform;
                dealiasing = dealiasing, invariant = invariant, advecting_hat = vh)
            results[i] = res
        end
    end
    return results
end

# ---------------------------------------------------------------------------
# Thread-parallel partial-flux batch (snapshots are the outer parallel axis)
# ---------------------------------------------------------------------------
# Overrides `_partial_fluxes_batch_threaded!`. One nonlinear-term workspace per chunk, built before the
# parallel region and reused across that chunk's snapshots with a serial inner transform.
function FIT.SpectralFlux._partial_fluxes_batch_threaded!(velocity_hats, ks; dealiasing, kwargs...)
    n = length(velocity_hats)
    û1 = first(velocity_hats)
    nchunks = max(1, min(Threads.nthreads(), n))
    rngs = collect(OhMyThreads.index_chunks(1:n; n = nchunks))
    pool = [FIT.Workspaces.NonlinearTermWorkspace(û1, ks; dealiasing = dealiasing) for _ in eachindex(rngs)]
    chunk_results = OhMyThreads.tmap(eachindex(rngs)) do ci
        [FIT.SpectralFlux.calculate_partial_fluxes!(pool[ci], velocity_hats[i], ks; dealiasing = dealiasing, kwargs...)
         for i in rngs[ci]]
    end
    return reduce(vcat, chunk_results)
end

# Overrides `_nufft_cg_batch_threaded!`. One NUFFT workspace (plans + buffers) per chunk, built inside
# its task, reused across that chunk's snapshots and closed when the chunk ends. Each result is copied out because
# `nufft_coarse_graining_flux!` wraps the workspace's own `Π`, which the chunk's next snapshot overwrites.
function FIT._nufft_cg_batch_threaded!(batch, scatter_coords, ℓ, filter, ms;
                                       transform, Ls, return_diagnostics::Bool = false, kwargs...)
    n = length(batch)
    nchunks = max(1, min(Threads.nthreads(), n))
    rngs = collect(OhMyThreads.index_chunks(1:n; n = nchunks))
    chunk_results = OhMyThreads.tmap(eachindex(rngs)) do ci
        ws = FIT.NUFFTCoarseGrainingWorkspace(scatter_coords, ms; transform = transform, Ls = Ls,
                                              return_diagnostics = return_diagnostics, kwargs...)
        try
            [deepcopy(FIT.nufft_coarse_graining_flux!(ws, batch[i], ℓ, filter, ms;
                                                      return_diagnostics = return_diagnostics))
             for i in rngs[ci]]
        finally
            FIT.close!(ws)
        end
    end
    return reduce(vcat, chunk_results)
end

# Overrides `_compressible_batch_threaded!`. Each chunk reuses one CompressibleWorkspace across its
# snapshots with a serial pipeline, bit-identical to the serial batch. The chunks are contiguous
# ascending ranges, so `reduce(vcat, …)` of their vectors is in input order.
function FIT.Compressible._compressible_batch_threaded!(
    velocity_hats, density_hats, pressure_hats, ks;
    transform, binning, geometry, dealiasing, decompose,
)
    n = length(velocity_hats)
    û1 = first(velocity_hats)
    nchunks = max(1, min(Threads.nthreads(), n))
    rngs = collect(OhMyThreads.index_chunks(1:n; n = nchunks))
    pool = [FIT.Compressible.CompressibleWorkspace(û1, ks; transform = transform, binning = binning,
                                                   geometry = geometry, dealiasing = dealiasing,
                                                   decompose = decompose,
                                                   with_pressure = pressure_hats !== nothing)
            for _ in eachindex(rngs)]
    chunk_results = OhMyThreads.tmap(eachindex(rngs)) do ci
        ws = pool[ci]
        [FIT.Compressible.calculate_compressible_flux!(
             ws, velocity_hats[i], density_hats[i], ks;
             pressure_hat = pressure_hats === nothing ? nothing : pressure_hats[i],
             dealiasing = dealiasing, decompose = decompose) for i in rngs[ci]]
    end
    return reduce(vcat, chunk_results)
end

# Overrides `_band_to_band_batch_threaded!`. Each chunk reuses one BandTransferWorkspace across its
# snapshots with a serial transform, bit-identical to the serial batch.
function FIT.BandTransfer._band_to_band_batch_threaded!(
    results, velocity_hats, ks, bands, nb;
    dealiasing, invariant, transform, geometry,
)
    n  = length(velocity_hats)
    û1 = first(velocity_hats)
    FT = real(eltype(û1))
    nchunks = max(1, min(Threads.nthreads(), n))
    rngs = collect(OhMyThreads.index_chunks(1:n; n = nchunks))
    pool = [FIT.BandTransfer.BandTransferWorkspace(û1, ks, bands; geometry = geometry, dealiasing = dealiasing)
            for _ in 1:length(rngs)]
    OhMyThreads.tforeach(eachindex(rngs)) do ci
        bws = pool[ci]
        for i in rngs[ci]
            vh = velocity_hats[i]
            T = zeros(FT, nb, nb); net = zeros(FT, nb)
            results[i] = FIT.BandTransfer.calculate_band_to_band_transfer!(T, net, bws, vh, ks;
                dealiasing = dealiasing, invariant = invariant, transform = transform,
                execution = FIT.ComputationalBackends.SerialBackend(), advecting_hat = vh)
        end
    end
    return results
end

# ---------------------------------------------------------------------------
# Thread-parallel coarse-graining batch (snapshots are the outer parallel axis)
# ---------------------------------------------------------------------------
# Overrides `_cg_flux_batch_threaded!`. Each chunk reuses one config-fixed workspace (grid + CGEF plans)
# across its snapshots, filtering each one on `SerialBackend()`. Each result is deep-copied so it does
# not alias the reused output buffer, and the chunks concatenate in input order.
function FIT.CoarseGrainingFlux._cg_flux_batch_threaded!(velocity_fields_batch, coords_vecs, ℓ, filter; kwargs...)
    n = length(velocity_fields_batch)
    nchunks = max(1, min(Threads.nthreads(), n))
    rngs = collect(OhMyThreads.index_chunks(1:n; n = nchunks))
    pool = [FIT.CoarseGrainingFlux.CoarseGrainingFluxWorkspace(velocity_fields_batch[first(rng)], coords_vecs, ℓ, filter;
                                                               kwargs..., execution = FIT.ComputationalBackends.SerialBackend())
            for rng in rngs]
    chunk_results = OhMyThreads.tmap(eachindex(rngs)) do ci
        ws = pool[ci]
        [deepcopy(FIT.CoarseGrainingFlux.calculate_coarse_graining_flux!(ws, velocity_fields_batch[i])) for i in rngs[ci]]
    end
    return reduce(vcat, chunk_results)
end

# ---------------------------------------------------------------------------
# Override TriadicOrthogonalDecomposition._triadic_loop_threaded!
# ---------------------------------------------------------------------------

"""
    _triadic_loop_threaded!(...)

Thread-parallel triad loop using OhMyThreads. Triads are independent: `Q_hat` is read-only and each
triad writes its own slots of `P`, `A_out` and `Xi_out` and its own `(fl, fn)` slice of `L` and
`T_budget`.
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

    nchunks = max(1, min(Threads.nthreads(), nTriads))
    rngs = collect(OhMyThreads.index_chunks(1:nTriads; n = nchunks))
    pool = [(sc = FIT.TriadicOrthogonalDecomposition._TriadSVDScratch(Q_hat, CT, nStateNx, nBlks),
             permbuf = similar(_permbuf), permbuf_kl = similar(_permbuf_kl)) for _ in eachindex(rngs)]

    OhMyThreads.tforeach(eachindex(rngs)) do ci
        p = pool[ci]
        for i in rngs[ci]
            fi_k = fk_idx[i]; fi_l = fl_idx[i]; fi_n = fn_idx[i]

            # The recipient and the (k, l) product into this chunk's reused buffers.
            Q_n_raw = view(Q_hat, :, :, :, fi_n)
            Q_k_raw = view(Q_hat, :, :, :, fi_k)
            Q_l_raw = view(Q_hat, :, :, :, fi_l)
            permutedims!(p.permbuf, LHS(Q_n_raw), (2, 1, 3))
            Q_hat_n = reshape(p.permbuf, nStateNx, nBlks)
            FIT.TriadicOrthogonalDecomposition._apply_nonlinear!(p.permbuf_kl, Q_nonlinear, Q_k_raw, Q_l_raw)
            Q_hat_kl = reshape(p.permbuf_kl, nStateNx, nBlks)

            # The serial loop's SVD, so the threaded result is bit-identical to the serial one.
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
            P[i] = (convective = u, recipient = v)

            if return_coefficients
                A_conv = u' * (Q_hat_kl .* weights)
                A_recip = v' * (Q_hat_n .* weights)
                A_out[i] = (convective = A_conv, recipient = A_recip)
                if return_auxiliary_modes
                    Q_hat_l = reshape(permutedims(LHS(Q_l_raw), (2, 1, 3)), nStateNx, nBlks)
                    Q_hat_k = reshape(permutedims(LHS(Q_k_raw), (2, 1, 3)), nStateNx, nBlks)
                    inv_s = 1 ./ s[1:nm]
                    donor_mode = Q_hat_l * A_recip' * LinearAlgebra.Diagonal(inv_s) ./ nBlks
                    catalyst_mode = Q_hat_k * A_recip' * LinearAlgebra.Diagonal(inv_s) ./ nBlks
                    Xi_out[i] = (donor = donor_mode[:, 1:nm], catalyst = catalyst_mode[:, 1:nm])
                end
            end
        end
    end
    return nothing
end

# ---------------------------------------------------------------------------
# Thread-parallel mode-to-mode transfer S(k|p)
# ---------------------------------------------------------------------------
# Overrides the core `_mode_to_mode_threaded!` stub. Threads the loop over slots `p`: each `p` writes
# the column `S[·, p]`, and each chunk keeps its own partial `net` (Σ_p S[·,p]), summed in chunk order.
# Each chunk synthesises the advecting velocity once and takes the serial loop's single-mode path (a
# zero column for a slot the dealiasing rule discards), or under `PaddedThreeHalves` the advection of
# the slot's field on the padded grid.
function FIT.ModeToModeTransfer._mode_to_mode_threaded!(
    result, ws, û_p, velocity_hat, ks;
    invariant::FIT.Types.AbstractInvariant = FIT.Types.KineticEnergy(),
    dealiasing::FIT.Types.AbstractDealiasing = FIT.Types.OrszagTwoThirds(),
    transform::SpectralBackends.AbstractSpectralBackend = SpectralBackends.DirectSumSpectralBackend(),
    advecting_hat = velocity_hat,
)
    nd = length(ks)
    ms = size(velocity_hat)[1:nd]
    M  = size(velocity_hat, nd + 1)
    FT = real(eltype(velocity_hat))
    S      = result.transfer
    cis    = CartesianIndices(ms)
    colons = ntuple(_ -> Colon(), nd)
    padded = dealiasing isa FIT.Types.PaddedThreeHalves
    truncate = !(dealiasing isa FIT.Types.NoDealiasing)
    nchunks = max(1, min(Threads.nthreads(), length(cis)))
    rngs = collect(OhMyThreads.index_chunks(1:length(cis); n = nchunks))
    pool = [(ws = FIT.Workspaces.NonlinearTermWorkspace(velocity_hat, ks; dealiasing = dealiasing),
             ûp = fill!(similar(velocity_hat), zero(eltype(velocity_hat))),
             net = zeros(FT, ms...)) for _ in eachindex(rngs)]
    OhMyThreads.tforeach(eachindex(rngs)) do ci
        p = pool[ci]
        FIT.NonlinearTerm.synthesize_advecting!(p.ws, advecting_hat, ks; dealiasing = dealiasing, transform = transform)
        @inbounds for i in rngs[ci]
            q = cis[i]
            Sq = view(S, colons..., q)
            if padded
                for c in 1:M
                    p.ûp[q, c] = velocity_hat[q, c]
                end
                FIT.NonlinearTerm.compute_advection!(p.ws, p.ûp, ks; dealiasing = dealiasing, transform = transform)
                for c in 1:M
                    p.ûp[q, c] = zero(eltype(p.ûp))
                end
            elseif truncate && FIT.NonlinearTerm._is_dealiased(ks, q)
                fill!(Sq, zero(eltype(Sq)))
                continue
            else
                FIT.ModeToModeTransfer._mode_to_mode_giver!(p.ws, velocity_hat, ks, q, p.ws.phase, truncate, transform)
            end
            FIT.Invariants.transfer_density!(Sq, invariant, velocity_hat, p.ws.N̂, ks)
            p.net .+= Sq
        end
    end
    net = result.net_transfer
    fill!(net, zero(eltype(net)))
    for p in pool
        net .+= p.net
    end
    return result
end

# ---------------------------------------------------------------------------
# Thread-parallel smooth band-to-band transfer T(n,m)
# ---------------------------------------------------------------------------
# Overrides the core `_band_to_band_threaded!` stub. Threads the outer band loop: each band `m` writes
# the column `T[·, m]`. The band-weight masks `W` are shared read-only.
function FIT.BandTransfer._band_to_band_threaded!(
    T, bws, velocity_hat, ks;
    dealiasing::FIT.Types.AbstractDealiasing = FIT.Types.OrszagTwoThirds(),
    invariant::FIT.Types.AbstractInvariant = FIT.Types.KineticEnergy(),
    transform::SpectralBackends.AbstractSpectralBackend = SpectralBackends.DirectSumSpectralBackend(),
    advecting_hat = velocity_hat,
)
    nd = length(ks)
    ns = size(velocity_hat)[1:nd]
    D  = size(velocity_hat, nd + 1)
    FT = real(eltype(velocity_hat))
    nb = length(bws.centers)
    W  = bws.W
    nchunks = max(1, min(Threads.nthreads(), nb))
    rngs = collect(OhMyThreads.index_chunks(1:nb; n = nchunks))
    pool = [(ws = FIT.Workspaces.NonlinearTermWorkspace(velocity_hat, ks; dealiasing = dealiasing),
             fm = similar(velocity_hat), d = similar(velocity_hat, FT, ns...)) for _ in 1:length(rngs)]
    OhMyThreads.tforeach(eachindex(rngs)) do ci
        p = pool[ci]
        FIT.NonlinearTerm.synthesize_advecting!(p.ws, advecting_hat, ks; dealiasing = dealiasing, transform = transform)
        for m in rngs[ci]
            for c in 1:D, I in CartesianIndices(ns)
                p.fm[I, c] = W[m][I] * velocity_hat[I, c]
            end
            FIT.NonlinearTerm.compute_advection!(p.ws, p.fm, ks; dealiasing = dealiasing, transform = transform)
            FIT.Invariants.transfer_density!(p.d, invariant, velocity_hat, p.ws.N̂, ks)
            for n in 1:nb
                s = zero(FT)
                for I in CartesianIndices(ns)
                    s += FTB.mode_fold(ks, I) * W[n][I] * p.d[I]
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
# Overrides the core `_partial_fluxes_threaded!` stub. Threads the n² (sp, sq) pairs, each an
# independent nonlinear-term build writing its own channel keys (sk, sp, sq) into a per-chunk Dict; the
# Dicts are merged after the loop.
function FIT.SpectralFlux._partial_fluxes_threaded!(
    channels, ws, comps, names, velocity_hat, ks, shells;
    dealiasing::FIT.Types.AbstractDealiasing,
    transform::SpectralBackends.AbstractSpectralBackend,
    decomposition::FIT.Types.AbstractFieldDecomposition,
)
    FT = real(eltype(velocity_hat)); nd = length(ks); ns = size(velocity_hat)[1:nd]
    prs = [(sp, sq) for sp in names for sq in names]
    np  = length(prs)
    nchunks = max(1, min(Threads.nthreads(), np))
    rngs = collect(OhMyThreads.index_chunks(1:np; n = nchunks))
    pool = [(ws = FIT.Workspaces.NonlinearTermWorkspace(velocity_hat, ks; dealiasing = dealiasing),
             d  = similar(velocity_hat, FT, ns...),
             ch = empty(channels)) for _ in 1:length(rngs)]
    OhMyThreads.tforeach(eachindex(rngs)) do ci
        p = pool[ci]
        for pi in rngs[ci]
            sp, sq = prs[pi]
            FIT.SpectralFlux._channel_term!(p.ws, decomposition, comps[sp], comps[sq], ks;
                                           dealiasing = dealiasing, transform = transform)
            for sk in names
                FIT.Invariants.transfer_density!(p.d, FIT.Types.KineticEnergy(), comps[sk], p.ws.N̂, ks)
                p.ch[(sk, sp, sq)] = FIT.SpectralFlux._partial_binflux(p.d, ks, shells)
            end
        end
    end
    for p in pool
        merge!(channels, p.ch)
    end
    return channels
end

end # module FlowInvariantTransferOhMyThreadsExt
