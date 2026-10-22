module FlowInvariantTransferDistributedExt

using Distributed: Distributed
using FlowInvariantTransfer: FlowInvariantTransfer as FIT
using FlowTransformBindings: FlowTransformBindings as FTB
using ComputationalBackends: ComputationalBackends
using SpectralBackends: SpectralBackends

# The closures shipped to workers capture plain arrays, never a workspace: a workspace's FFTW plan
# bundle does not deserialize on a worker without the FFTW extension. Worker processes must have
# `using FlowInvariantTransfer` loaded.

# Distributed Shell-to-Shell Transfer Implementation (overrides the core `_shell_to_shell_distributed!` stub)
function FIT.ShellToShellTransfer._shell_to_shell_distributed!(
    result::FIT.Types.ShellToShellResult,
    ws::FIT.Workspaces.ShellToShellWorkspace,
    velocity_hat,
    ks,
    execution::ComputationalBackends.DistributedBackend,
    spectral;            # transform backend, passed to each per-mediator nonlinear term
    dealiasing::FIT.Types.AbstractDealiasing,
    invariant::FIT.Types.AbstractInvariant = FIT.Types.KineticEnergy(),
    advecting_hat = velocity_hat,
)
    N_sh = size(result.transfer_matrix, 1)
    FT = real(eltype(velocity_hat))
    inner = ComputationalBackends.local_backend(execution)   # per-worker backend (Serial default, or Threaded for hybrid)
    shell_idx = ws.shell_idx

    # Mediators are partitioned into one contiguous chunk per worker; each worker builds one
    # `NonlinearTermWorkspace` for its chunk and returns `(m, column)` pairs.
    nw = max(1, Distributed.nworkers())
    chunks = collect(Iterators.partition(1:N_sh, max(1, cld(N_sh, nw))))
    chunk_cols = Distributed.pmap(chunks) do ms
        _shell_to_shell_worker_columns(ms, velocity_hat, ks, shell_idx, N_sh, invariant,
                                       dealiasing, FT, spectral, advecting_hat, inner)
    end
    fill!(result.transfer_matrix, zero(FT))
    for cols in chunk_cols, (m, col) in cols
        @inbounds for n in 1:N_sh
            result.transfer_matrix[n, m] = col[n]
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
# Distributed spectral flux Π(K) reduction
# ---------------------------------------------------------------------------
# Overrides the core `_spectral_flux_distributed!` stub dispatched by `ComputationalBackends.DistributedBackend`. The
# transfer density is computed on the caller, then the mode→shell scatter is partitioned across
# workers (strided over the linear mode index) and `+`-reduced into the shell spectrum. The
# nonlinear-term FFTs stay on the caller and the density array is shipped to the workers; a grid too
# large for one node takes `pencil_spectral_flux` (PencilFFTs), which splits the FFT itself.
function FIT.SpectralFlux._spectral_flux_distributed!(
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

    td = FIT.Invariants.transfer_density(invariant, velocity_hat, N̂, ks)
    sidx = shell_idx
    nw = max(1, Distributed.nworkers())

    T = Distributed.@distributed (+) for w in 1:nw
        t = zeros(FT, N_sh)
        cis = CartesianIndices(ns)
        @inbounds for i in w:nw:Np
            n = sidx[i]
            n == 0 && continue
            t[n] += FTB.mode_fold(ks, cis[i]) * td[i]
        end
        t
    end
    copyto!(ws.T_spec, T)
    return FIT.SpectralFlux._finalize_spectral_flux!(result, ws)
end

# One worker's chunk of mediator shells → `[(m, column), …]`, with one workspace and band-field scratch
# per run of mediators. 𝒩̂_m = (u·∇)f_m: the full velocity (advecting_hat) advects the band-m primary
# field (Alexakis–Mininni–Pouquet 2005), so for energy A[n,m] is antisymmetric and reduces to
# transfer_spectrum[n]. Each column is one pass over the modes.
#
# `inner` is the worker-local backend: `ThreadedBackend()` splits the worker's mediators across its
# threads, one run and one workspace per thread; each mediator writes its own slot of `out`.
function _shell_to_shell_worker_columns(ms, velocity_hat, ks, shell_idx, N_sh, invariant, dealiasing,
                                        FT, spectral, advecting_hat, inner)
    nd = length(ks)
    ns = size(velocity_hat)[1:nd]
    M  = size(velocity_hat, nd + 1)   # components of the binned/carried primary field
    out = Vector{Tuple{Int, Vector{FT}}}(undef, length(ms))
    fill_run!(run) = begin
        nl_ws = FIT.Workspaces.NonlinearTermWorkspace(velocity_hat, ks; dealiasing = dealiasing)
        û_m = similar(velocity_hat)
        FIT.NonlinearTerm.synthesize_advecting!(nl_ws, advecting_hat, ks; dealiasing = dealiasing, transform = spectral)
        for i in run
            m = ms[i]
            fill!(û_m, zero(eltype(û_m)))
            @inbounds for I in CartesianIndices(ns)
                shell_idx[I] == m || continue
                for comp in 1:M
                    û_m[I, comp] = velocity_hat[I, comp]
                end
            end
            FIT.NonlinearTerm.compute_advection!(nl_ws, û_m, ks; dealiasing = dealiasing, transform = spectral)
            col = zeros(FT, N_sh)
            FIT.Invariants.transfer_density_scatter!(col, invariant, velocity_hat, nl_ws.N̂, ks, shell_idx)
            out[i] = (m, col)
        end
    end
    if inner isa ComputationalBackends.ThreadedBackend
        nchunks = max(1, min(Threads.nthreads(), length(ms)))
        runs = [c:nchunks:length(ms) for c in 1:nchunks]
        Threads.@threads for ci in eachindex(runs)
            fill_run!(runs[ci])
        end
    else
        fill_run!(eachindex(ms))
    end
    return out
end

# ---------------------------------------------------------------------------
# Distributed Triadic Orthogonal Decomposition (overrides the core `_triadic_loop_distributed!` stub).
# Triads are independent and are distributed over worker processes with `@distributed (vcat)`: each
# worker runs `_triad_result` (the serial loop's SVD, so the results are bit-identical) for its triads,
# and the master assembles L / T_budget / P and the optional A_out / Xi_out.
function FIT.TriadicOrthogonalDecomposition._triadic_loop_distributed!(
    L, P, T_budget, A_out, Xi_out,
    Q_hat, f_idx, fk_idx, fl_idx, fn_idx,
    weights, nBlks, nFreq, nState, nx, nmode,
    Q_nonlinear, LHS,
    return_coefficients, return_auxiliary_modes,
    _sc, sqrt_w, inv_sqrt_w, _permbuf, _permbuf_kl,
    execution::ComputationalBackends.DistributedBackend,
)
    nTriads = length(fk_idx)
    results = Distributed.@distributed (vcat) for i in 1:nTriads
        [FIT.TriadicOrthogonalDecomposition._triad_result(
            i, Q_hat, fk_idx, fl_idx, fn_idx, weights, sqrt_w, inv_sqrt_w,
            nBlks, nState, nx, nmode, Q_nonlinear, LHS,
            return_coefficients, return_auxiliary_modes)]
    end
    for r in results
        @inbounds for j in 1:r.nm
            L[r.fi_l, r.fi_n, j] = r.s[j]
            T_budget[r.fi_l, r.fi_n, j] = r.Tb[j]
        end
        P[r.i] = (convective = r.u, recipient = r.v)
        if return_coefficients
            A_out[r.i] = (convective = r.A_conv, recipient = r.A_recip)
            return_auxiliary_modes && (Xi_out[r.i] = (donor = r.donor, catalyst = r.catalyst))
        end
    end
    return nothing
end

# ---------------------------------------------------------------------------
# Distributed mode-to-mode S(k|p)  (overrides the core `_mode_to_mode_distributed!` stub)
# ---------------------------------------------------------------------------
# Each slot `p` writes the column S[·,p]. Slots are partitioned across workers (strided); each worker
# returns the columns of its slots, which the master writes into the tensor, and `net[k] = Σ_p S[k,p]`
# is summed on the master. A worker synthesises the advecting velocity once and takes the serial loop's
# single-mode path (or, under `PaddedThreeHalves`, the advection of the slot's field on the padded
# grid). The inner backend (`ComputationalBackends.local_backend`, for the hybrid
# `DistributedBackend(ThreadedBackend())`) threads each worker's slots, one workspace per thread.
function FIT.ModeToModeTransfer._mode_to_mode_distributed!(
    result, ws, û_p, velocity_hat, ks, execution::ComputationalBackends.DistributedBackend;
    invariant = FIT.Types.KineticEnergy(),
    dealiasing = FIT.Types.OrszagTwoThirds(),
    transform = SpectralBackends.DirectSumSpectralBackend(),
    advecting_hat = velocity_hat,
)
    nd = length(ks); ns = size(velocity_hat)[1:nd]; M = size(velocity_hat, nd + 1)
    FT = real(eltype(velocity_hat))
    cis = CartesianIndices(ns); colons = ntuple(_ -> Colon(), nd)
    inner = ComputationalBackends.local_backend(execution)
    nw = max(1, Distributed.nworkers())
    idx_sets = [collect(w:nw:length(cis)) for w in 1:nw]
    parts = Distributed.pmap(idx_sets) do idxs
        _mode_to_mode_worker_columns(idxs, velocity_hat, ks, ns, M, FT,
            invariant, dealiasing, transform, advecting_hat, cis, colons, inner)
    end
    S = result.transfer
    for (idxs, cols) in zip(idx_sets, parts), (j, i) in enumerate(idxs)
        view(S, colons..., cis[i]) .= view(cols, colons..., j)
    end

    net = result.net_transfer
    fill!(net, zero(eltype(net)))
    @inbounds for p in cis
        net .+= view(S, colons..., p)
    end
    return result
end

# The columns `S[·, p]` of the slots `cis[idxs]`, in that order.
function _mode_to_mode_worker_columns(idxs, velocity_hat, ks, ns, M, FT, invariant, dealiasing, spectral,
                                      advecting_hat, cis, colons, inner)
    cols = zeros(FT, ns..., length(idxs))
    padded = dealiasing isa FIT.Types.PaddedThreeHalves
    truncate = !(dealiasing isa FIT.Types.NoDealiasing)
    fill_range!(rng) = begin
        lws = FIT.Workspaces.NonlinearTermWorkspace(velocity_hat, ks; dealiasing = dealiasing)
        lup = fill!(similar(velocity_hat), zero(eltype(velocity_hat)))
        FIT.NonlinearTerm.synthesize_advecting!(lws, advecting_hat, ks; dealiasing = dealiasing, transform = spectral)
        @inbounds for j in rng
            p = cis[idxs[j]]
            if padded
                for c in 1:M; lup[p, c] = velocity_hat[p, c]; end
                FIT.NonlinearTerm.compute_advection!(lws, lup, ks; dealiasing = dealiasing, transform = spectral)
                for c in 1:M; lup[p, c] = zero(eltype(lup)); end
            elseif truncate && FIT.NonlinearTerm._is_dealiased(ks, p)
                continue                                         # a discarded slot's column is zero
            else
                FIT.ModeToModeTransfer._mode_to_mode_giver!(lws, velocity_hat, ks, p, lws.phase, truncate, spectral)
            end
            FIT.Invariants.transfer_density!(view(cols, colons..., j), invariant, velocity_hat, lws.N̂, ks)
        end
    end
    if inner isa ComputationalBackends.ThreadedBackend
        nchunks = max(1, min(Threads.nthreads(), length(idxs)))
        chunks = [c:nchunks:length(idxs) for c in 1:nchunks]
        Threads.@threads for ci in eachindex(chunks)
            fill_range!(chunks[ci])
        end
    else
        fill_range!(eachindex(idxs))
    end
    return cols
end

# ---------------------------------------------------------------------------
# Distributed smooth band-to-band T(n,m)  (overrides `_band_to_band_distributed!`)
# ---------------------------------------------------------------------------
# Each band `m` writes the column T[·,m]; bands are partitioned across workers (strided), each worker
# fills a local T (zero outside its columns), and the locals are `+`-reduced into the full matrix. The
# inner backend threads the worker's band loop over those disjoint columns.
function FIT.BandTransfer._band_to_band_distributed!(
    T, bws, velocity_hat, ks, execution::ComputationalBackends.DistributedBackend;
    dealiasing = FIT.Types.OrszagTwoThirds(),
    invariant = FIT.Types.KineticEnergy(),
    transform = SpectralBackends.DirectSumSpectralBackend(),
    advecting_hat = velocity_hat,
)
    nd = length(ks); ns = size(velocity_hat)[1:nd]; D = size(velocity_hat, nd + 1)
    FT = real(eltype(velocity_hat)); nb = length(bws.centers); W = bws.W
    inner = ComputationalBackends.local_backend(execution)
    nw = max(1, Distributed.nworkers())

    Tred = Distributed.@distributed (+) for w in 1:nw
        _band_to_band_worker_T(collect(w:nw:nb), W, velocity_hat, ks, ns, D, FT, nb,
            invariant, dealiasing, transform, advecting_hat, inner)
    end
    copyto!(T, Tred)
    return T
end

function _band_to_band_worker_T(ms, W, velocity_hat, ks, ns, D, FT, nb, invariant, dealiasing, spectral,
                                advecting_hat, inner)
    local_T = zeros(FT, nb, nb)
    fill_bands!(mrng) = begin
        lws = FIT.Workspaces.NonlinearTermWorkspace(velocity_hat, ks; dealiasing = dealiasing)
        fm = similar(velocity_hat); d = similar(velocity_hat, FT, ns...)
        FIT.NonlinearTerm.synthesize_advecting!(lws, advecting_hat, ks; dealiasing = dealiasing, transform = spectral)
        @inbounds for m in mrng
            for c in 1:D, I in CartesianIndices(ns)
                fm[I, c] = W[m][I] * velocity_hat[I, c]
            end
            FIT.NonlinearTerm.compute_advection!(lws, fm, ks; dealiasing = dealiasing, transform = spectral)
            FIT.Invariants.transfer_density!(d, invariant, velocity_hat, lws.N̂, ks)
            for n in 1:nb
                s = zero(FT)
                for I in CartesianIndices(ns)
                    s += FTB.mode_fold(ks, I) * W[n][I] * d[I]
                end
                local_T[n, m] = s                                # disjoint per m → thread-safe
            end
        end
    end
    if inner isa ComputationalBackends.ThreadedBackend
        nchunks = max(1, min(Threads.nthreads(), length(ms)))
        chunks = [ms[c:nchunks:end] for c in 1:nchunks]
        Threads.@threads for ci in eachindex(chunks)
            fill_bands!(chunks[ci])
        end
    else
        fill_bands!(ms)
    end
    return local_T
end

# ---------------------------------------------------------------------------
# Distributed partial (decomposition-channel) fluxes  (overrides `_partial_fluxes_distributed!`)
# ---------------------------------------------------------------------------
# Each (sp, sq) pair is an independent nonlinear-term build writing its own channel keys; pairs are
# partitioned across workers (strided), each worker builds a partial channel Dict, and the Dicts are
# `merge`-reduced. Under a threaded inner backend each task keeps its own workspace and Dict, merged
# within the worker.
function FIT.SpectralFlux._partial_fluxes_distributed!(
    channels, ws, comps, names, velocity_hat, ks, shells, execution::ComputationalBackends.DistributedBackend;
    dealiasing::FIT.Types.AbstractDealiasing,
    transform::SpectralBackends.AbstractSpectralBackend,
    decomposition::FIT.Types.AbstractFieldDecomposition,
)
    FT = real(eltype(velocity_hat)); nd = length(ks); ns = size(velocity_hat)[1:nd]
    prs = [(sp, sq) for sp in names for sq in names]; np = length(prs)
    inner = ComputationalBackends.local_backend(execution)
    nw = max(1, Distributed.nworkers())
    proto = empty(channels)

    merged = Distributed.@distributed (merge) for w in 1:nw
        _partial_fluxes_worker(collect(w:nw:np), prs, comps, names, velocity_hat, ks, ns, FT,
            shells, proto, dealiasing, transform, decomposition, inner)
    end
    merge!(channels, merged)
    return channels
end

function _partial_fluxes_worker(pis, prs, comps, names, velocity_hat, ks, ns, FT, shells, proto,
                                dealiasing, spectral, decomposition, inner)
    build(pirng) = begin
        lws = FIT.Workspaces.NonlinearTermWorkspace(velocity_hat, ks; dealiasing = dealiasing)
        d = similar(velocity_hat, FT, ns...)
        ch = empty(proto)
        @inbounds for pi in pirng
            sp, sq = prs[pi]
            FIT.SpectralFlux._channel_term!(lws, decomposition, comps[sp], comps[sq], ks;
                                           dealiasing = dealiasing, transform = spectral)
            for sk in names
                FIT.Invariants.transfer_density!(d, FIT.Types.KineticEnergy(), comps[sk], lws.N̂, ks)
                ch[(sk, sp, sq)] = FIT.SpectralFlux._partial_binflux(d, ks, shells)
            end
        end
        return ch
    end
    if inner isa ComputationalBackends.ThreadedBackend
        nchunks = max(1, min(Threads.nthreads(), length(pis)))
        chunks = [pis[c:nchunks:end] for c in 1:nchunks]
        parts = Vector{typeof(proto)}(undef, length(chunks))
        Threads.@threads for ci in eachindex(chunks)
            parts[ci] = build(chunks[ci])
        end
        out = empty(proto)
        for p in parts; merge!(out, p); end
        return out
    else
        return build(pis)
    end
end

# ---------------------------------------------------------------------------
# Distributed compressible transfer  (overrides the core `_compressible_distributed` stub)
# ---------------------------------------------------------------------------
# The independent work units of the compressible pipeline are the momentum-weighted transfer +
# Helmholtz channel set (group A) and the KE↔IE pressure-dilatation (group B). Each group runs on its
# own worker, building its own `CompressibleWorkspace` from the raw inputs, and the master assembles the
# result. Each worker runs the inner backend (`ComputationalBackends.local_backend`), so
# `DistributedBackend(ThreadedBackend())` threads each worker's FFTs. Without a pressure field the one
# work unit runs on one worker. Bit-identical to the serial pipeline.
function FIT.Compressible._compressible_distributed(
    velocity_hat, density_hat, ks, execution::ComputationalBackends.DistributedBackend;
    transform, binning, geometry,
    pressure_hat = nothing,
    dealiasing = FIT.Types.OrszagTwoThirds(),
    decompose = true,
)
    inner = ComputationalBackends.local_backend(execution)
    # Each group's workspace carries only the scratch its own call uses: group A the channel set,
    # group B the pressure term. The flags are fixed at construction, so they are set to match.
    runA = () -> begin
        ws = FIT.Compressible.CompressibleWorkspace(velocity_hat, ks;
            transform = transform, binning = binning, geometry = geometry, execution = inner,
            dealiasing = dealiasing, decompose = decompose, with_pressure = false)
        FIT.Compressible.calculate_compressible_flux!(ws, velocity_hat, density_hat, ks;
            dealiasing = dealiasing, decompose = decompose, pressure_hat = nothing)
    end
    if pressure_hat === nothing
        return fetch(Distributed.@spawnat :any runA())
    end
    runB = () -> begin
        ws = FIT.Compressible.CompressibleWorkspace(velocity_hat, ks;
            transform = transform, binning = binning, geometry = geometry, execution = inner,
            dealiasing = dealiasing, decompose = false, with_pressure = true)
        FIT.Compressible.calculate_compressible_flux!(ws, velocity_hat, density_hat, ks;
            dealiasing = dealiasing, decompose = false, pressure_hat = pressure_hat)
    end
    fA = Distributed.@spawnat :any runA()
    fB = Distributed.@spawnat :any runB()
    rA = fetch(fA); rB = fetch(fB)
    return FIT.Types.CompressibleFluxResult(rA.k_shells, rA.shell_edges, rA.transfer_spectrum, rA.flux,
                                            rA.channels, rB.pressure_dilatation)
end

# ---------------------------------------------------------------------------
# `*_batch` entries under `execution = DistributedBackend(inner)`
# ---------------------------------------------------------------------------
# One contiguous chunk of snapshots per worker. A worker is sent its own chunk of each per-snapshot input
# and runs the batch's driver on it with `inner`; the chunks come back in input order, so the results
# are the serial batch's.

_chunks(n::Integer) = collect(Iterators.partition(1:n, max(1, cld(n, max(1, Distributed.nworkers())))))

# `run(inner, parts...)` on the workers, `parts` the chunk's slice of each input (`nothing` passes through).
function _distributed_batch(run, b, n::Integer, inputs::Tuple)
    inner = ComputationalBackends.local_backend(b)
    parts = [map(x -> x === nothing ? nothing : x[c], inputs) for c in _chunks(n)]
    return reduce(vcat, Distributed.pmap(p -> run(inner, p...), parts))
end

# The drivers that fill `results` are sent their chunk of it: the spectral-flux driver writes into the
# result objects it is given.
function FIT.SpectralFlux._spectral_flux_batch!(b::ComputationalBackends.AbstractDistributedBackend, results,
                                                velocity_hats, ks, shell_idx; kwargs...)
    flat = _distributed_batch(b, length(results), (results, collect(velocity_hats))) do inner, rs, vs
        FIT.SpectralFlux._spectral_flux_batch!(inner, rs, vs, ks, shell_idx; kwargs...)
    end
    return copyto!(results, flat)
end

function FIT.ShellToShellTransfer._shell_to_shell_batch!(b::ComputationalBackends.AbstractDistributedBackend,
                                                         results, velocity_hats, ks, centers, edges, N_sh; kwargs...)
    flat = _distributed_batch(b, length(results), (results, collect(velocity_hats))) do inner, rs, vs
        FIT.ShellToShellTransfer._shell_to_shell_batch!(inner, rs, vs, ks, centers, edges, N_sh; kwargs...)
    end
    return copyto!(results, flat)
end

function FIT.BandTransfer._band_to_band_batch!(b::ComputationalBackends.AbstractDistributedBackend, results,
                                               velocity_hats, ks, bands, nb; kwargs...)
    flat = _distributed_batch(b, length(results), (results, collect(velocity_hats))) do inner, rs, vs
        FIT.BandTransfer._band_to_band_batch!(inner, rs, vs, ks, bands, nb; kwargs...)
    end
    return copyto!(results, flat)
end

function FIT.SpectralFlux._partial_fluxes_batch!(b::ComputationalBackends.AbstractDistributedBackend,
                                                 velocity_hats, ks; kwargs...)
    vs = collect(velocity_hats)
    return _distributed_batch(b, length(vs), (vs,)) do inner, part
        FIT.SpectralFlux._partial_fluxes_batch!(inner, part, ks; kwargs...)
    end
end

function FIT.CoarseGrainingFlux._cg_flux_batch!(b::ComputationalBackends.AbstractDistributedBackend,
                                                velocity_fields_batch, geometry, ℓ, filter; kwargs...)
    vs = collect(velocity_fields_batch)
    return _distributed_batch(b, length(vs), (vs,)) do inner, part
        FIT.CoarseGrainingFlux._cg_flux_batch!(inner, part, geometry, ℓ, filter; kwargs...)
    end
end

function FIT._nufft_cg_batch!(b::ComputationalBackends.AbstractDistributedBackend, batch, scatter_coords, ℓ,
                              filter, ms; kwargs...)
    vs = collect(batch)
    return _distributed_batch(b, length(vs), (vs,)) do inner, part
        FIT._nufft_cg_batch!(inner, part, scatter_coords, ℓ, filter, ms; kwargs...)
    end
end

function FIT.Compressible._compressible_batch!(b::ComputationalBackends.AbstractDistributedBackend,
                                               velocity_hats, density_hats, pressure_hats, ks; kwargs...)
    vs = collect(velocity_hats)
    inputs = (vs, collect(density_hats), pressure_hats === nothing ? nothing : collect(pressure_hats))
    return _distributed_batch(b, length(vs), inputs) do inner, v, d, p
        FIT.Compressible._compressible_batch!(inner, v, d, p, ks; kwargs...)
    end
end

end # module
