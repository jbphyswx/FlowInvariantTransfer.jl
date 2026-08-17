module FlowInvariantTransferDistributedExt

using Distributed: Distributed
using FlowInvariantTransfer: FlowInvariantTransfer as FIT
using ComputationalBackends: ComputationalBackends
using SpectralBackends: SpectralBackends

# Distributed Shell-to-Shell Transfer Implementation (overrides the core `_shell_to_shell_distributed!` stub)
function FIT.ShellToShellTransfer._shell_to_shell_distributed!(
    result::FIT.Types.ShellToShellResult,
    ws::FIT.Workspaces.ShellToShellWorkspace,
    velocity_hat,
    ks,
    execution::ComputationalBackends.DistributedBackend,
    spectral;            # transform backend, passed to each per-mediator nonlinear term
    dealiasing::FIT.Types.AbstractDealiasing,
    verify_antisymmetry::Bool,
    invariant::FIT.Types.AbstractInvariant = FIT.Types.KineticEnergy(),
    advecting_hat = velocity_hat,
)
    N_sh = size(result.transfer_matrix, 1)
    FT = real(eltype(velocity_hat))
    inner = ComputationalBackends.local_backend(execution)   # per-worker backend (Serial default, or Threaded for hybrid)

    # Hoist shell_idx to a local so the @distributed closure captures only this plain
    # Int array — NOT the whole `ws`, whose nonlinear workspace may hold an FFTW-ext plan
    # bundle that workers can't deserialize (they need not have FFTWExt loaded).
    shell_idx = ws.shell_idx

    # We distribute the computation over the mediator shells `m`.
    # Using `Distributed.@distributed (+)` reduces the resulting N_sh x N_sh matrices.
    T_mat_reduced = Distributed.@distributed (+) for m in 1:N_sh
        # Compute column m on the worker process, using the worker-local backend.
        col = compute_mediator_transfer_column(m, velocity_hat, ks, shell_idx, N_sh, invariant, dealiasing, FT, spectral, advecting_hat, inner)
        
        # Construct an array where only column m is filled
        local_T = zeros(FT, N_sh, N_sh)
        local_T[:, m] = col
        local_T
    end
    
    # Copy reduced results into our in-place result structure
    copyto!(result.transfer_matrix, T_mat_reduced)
    
    # Net energy gain of each shell: Σ_m T(n,m)
    for n in 1:N_sh
        s = zero(FT)
        for m in 1:N_sh
            s += result.transfer_matrix[n, m]
        end
        result.net_transfer[n] = s
    end

    # Antisymmetry check
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
# Distributed spectral flux Π(K) reduction
# ---------------------------------------------------------------------------
# Overrides the core `_spectral_flux_distributed!` stub dispatched by `ComputationalBackends.DistributedBackend`. The
# transfer density is computed on the caller, then the mode→shell scatter is partitioned across
# workers (strided over the linear mode index) and `+`-reduced into the shell spectrum.
#
# NOTE: this distributes only the reduction, not the (dominant) nonlinear-term FFT, and it ships the
# density array to the workers — so for a single in-memory field it is rarely faster than Serial /
# Threaded. To distribute a grid too large for one node (splitting the FFT itself), use
# `pencil_spectral_flux` (PencilFFTs). Provided here for API parity with the other diagnostics and
# for pipelines whose field is already worker-resident.
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

    FIT.Invariants.transfer_density!(ws.transfer_density, invariant, velocity_hat, N̂, ks)
    # Hoist plain arrays so the @distributed closure captures only these (not the whole ws, whose
    # nonlinear workspace may hold an FFTW-ext plan bundle workers can't deserialize).
    td = ws.transfer_density
    sidx = shell_idx
    nw = max(1, Distributed.nworkers())

    T = Distributed.@distributed (+) for w in 1:nw
        t = zeros(FT, N_sh)
        @inbounds for i in w:nw:Np
            n = sidx[i]
            n == 0 && continue
            t[n] += td[i]
        end
        t
    end
    copyto!(ws.T_spec, T)
    return FIT.SpectralFlux._finalize_spectral_flux!(result, ws)
end

# Helper function executed on worker processes for Shell-to-Shell
function compute_mediator_transfer_column(m, velocity_hat, ks, shell_idx, N_sh, invariant, dealiasing, FT, spectral, advecting_hat=velocity_hat, inner=ComputationalBackends.SerialBackend())
    nd = length(ks)
    ns = size(velocity_hat)[1:nd]
    M  = size(velocity_hat, nd+1)   # components of the binned/carried primary field

    # Restrict the primary field to shell m
    û_m = zeros(eltype(velocity_hat), size(velocity_hat)...)
    for I in CartesianIndices(ns)
        shell_idx[I] == m || continue
        for comp in 1:M
            û_m[I, comp] = velocity_hat[I, comp]
        end
    end

    # Allocate a local NonlinearTermWorkspace.
    # 𝒩̂_m = (u·∇)f_m: full velocity (advecting_hat) advects the band-m primary field (AMP 2005) —
    # for energy gives antisymmetric A[n,m] reducing to transfer_spectrum[n] (matches serial/FFT).
    nl_ws = FIT.Workspaces.NonlinearTermWorkspace(velocity_hat, ks)
    FIT.NonlinearTerm.compute_nonlinear_term!(nl_ws, û_m, ks; dealiasing=dealiasing,
        spectral=spectral, advecting_hat=advecting_hat)

    # Write per-mode transfer density
    transfer_density = similar(velocity_hat, FT, ns...)
    FIT.Invariants.transfer_density!(transfer_density, invariant, velocity_hat, nl_ws.N̂, ks)

    # Accumulate into the column vector. `inner` is the worker-local execution backend from
    # `ComputationalBackends.local_backend`: ComputationalBackends.SerialBackend (default) sums receiver shells serially; ComputationalBackends.ThreadedBackend threads
    # that reduction with the worker's own threads (Base `@threads`, no cross-extension dependency) —
    # this realises ComputationalBackends.DistributedBackend(ComputationalBackends.ThreadedBackend()) when workers are started with `-t N`. Each
    # receiver shell n writes a disjoint col[n], so the threaded loop is race-free.
    col = zeros(FT, N_sh)
    if inner isa ComputationalBackends.ThreadedBackend
        Threads.@threads for n in 1:N_sh
            s = zero(FT)
            @inbounds for I in CartesianIndices(ns)
                shell_idx[I] == n || continue
                s += transfer_density[I]
            end
            col[n] = s
        end
    else
        for n in 1:N_sh
            s = zero(FT)
            @inbounds for I in CartesianIndices(ns)
                shell_idx[I] == n || continue
                s += transfer_density[I]
            end
            col[n] = s
        end
    end
    return col
end

# ---------------------------------------------------------------------------
# Distributed Triadic Orthogonal Decomposition (overrides the core `_triadic_loop_distributed!` stub).
# Triads are independent, so they are distributed over worker processes with `@distributed (vcat)`: each
# worker runs `_triad_result` (the SAME allocation-reusing SVD as the serial loop → bit-identical) for
# its triads and ships the per-triad outputs back; the master assembles L / T_budget / P (and the
# optional A_out / Xi_out). Only plain arrays are captured by the closure (Q_hat, weights, indices),
# never the workspace (whose FFT-plan bundle workers may not be able to deserialize). Worker processes
# must have `using FlowInvariantTransfer` loaded.
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
        P[(r.fi_l, r.fi_n)] = (convective = r.u, recipient = r.v)
        if return_coefficients
            A_out[(r.fi_l, r.fi_n)] = (convective = r.A_conv, recipient = r.A_recip)
            return_auxiliary_modes && (Xi_out[(r.fi_l, r.fi_n)] = (donor = r.donor, catalyst = r.catalyst))
        end
    end
    return nothing
end

# ---------------------------------------------------------------------------
# Distributed mode-to-mode S(k|p)  (overrides the core `_mode_to_mode_distributed!` stub)
# ---------------------------------------------------------------------------
# Each giver mode `p` writes a DISJOINT column S[·,p], so the giver loop is embarrassingly parallel.
# Givers are partitioned across workers (strided); each worker fills a local S (zero except its own
# columns) and the columns are `+`-reduced into the full tensor. `net[k]=Σ_p S[k,p]` is derived on the
# master. The inner backend (`ComputationalBackends.local_backend`, for the hybrid `ComputationalBackends.DistributedBackend(ComputationalBackends.ThreadedBackend())`)
# threads each worker's giver loop with its own threads + per-thread workspace; the columns written are
# disjoint, so writing into the shared local S is race-free. Only plain arrays are captured by the
# closure (never `ws`, whose FFT-plan bundle workers may not be able to deserialize).
function FIT.ModeToModeTransfer._mode_to_mode_distributed!(
    result, ws, û_p, velocity_hat, ks, execution::ComputationalBackends.DistributedBackend;
    invariant = FIT.Types.KineticEnergy(),
    dealiasing = FIT.Types.OrszagTwoThirds(),
    spectral = SpectralBackends.DirectSumSpectralBackend(),
    advecting_hat = velocity_hat,
)
    nd = length(ks); ns = size(velocity_hat)[1:nd]; M = size(velocity_hat, nd + 1)
    FT = real(eltype(velocity_hat)); Nscales = prod(ns)
    cis = CartesianIndices(ns); colons = ntuple(_ -> Colon(), nd)
    inner = ComputationalBackends.local_backend(execution)
    nw = max(1, Distributed.nworkers())

    S = Distributed.@distributed (+) for w in 1:nw
        _mode_to_mode_worker_S(collect(w:nw:Nscales), velocity_hat, ks, ns, M, FT,
            invariant, dealiasing, spectral, advecting_hat, cis, colons, inner)
    end
    copyto!(result.transfer, S)

    net = result.net_transfer
    fill!(net, zero(eltype(net)))
    @inbounds for p in cis
        net .+= view(result.transfer, colons..., p)
    end
    return result
end

function _mode_to_mode_worker_S(idxs, velocity_hat, ks, ns, M, FT, invariant, dealiasing, spectral,
                                advecting_hat, cis, colons, inner)
    local_S = zeros(FT, ns..., ns...)
    fill_range!(rng) = begin
        lws = FIT.Workspaces.NonlinearTermWorkspace(velocity_hat, ks; dealiasing = dealiasing)
        lup = similar(velocity_hat)
        @inbounds for i in rng
            p = cis[i]
            fill!(lup, zero(eltype(lup)))
            for c in 1:M; lup[p, c] = velocity_hat[p, c]; end
            FIT.NonlinearTerm.compute_nonlinear_term!(lws, lup, ks;
                dealiasing = dealiasing, spectral = spectral, advecting_hat = advecting_hat)
            Sp = view(local_S, colons..., p)                    # disjoint per p → thread-safe
            FIT.Invariants.transfer_density!(Sp, invariant, velocity_hat, lws.N̂, ks)
        end
    end
    if inner isa ComputationalBackends.ThreadedBackend
        nchunks = max(1, min(Threads.nthreads(), length(idxs)))
        chunks = [idxs[c:nchunks:end] for c in 1:nchunks]
        Threads.@threads for ci in eachindex(chunks)
            fill_range!(chunks[ci])
        end
    else
        fill_range!(idxs)
    end
    return local_S
end

# ---------------------------------------------------------------------------
# Distributed smooth band-to-band T(n,m)  (overrides `_band_to_band_distributed!`)
# ---------------------------------------------------------------------------
# Each band `m` writes a DISJOINT column T[·,m]; bands are partitioned across workers (strided), each
# worker fills a local T (zero except its columns), `+`-reduced into the full matrix. Inner backend
# threads the worker's band loop (disjoint columns → shared write is race-free). Only the band masks
# `bws.W` (plain arrays) are captured, not the plan-holding workspace.
function FIT.BandTransfer._band_to_band_distributed!(
    T, bws, velocity_hat, ks, execution::ComputationalBackends.DistributedBackend;
    dealiasing = FIT.Types.OrszagTwoThirds(),
    invariant = FIT.Types.KineticEnergy(),
    spectral = SpectralBackends.DirectSumSpectralBackend(),
    advecting_hat = velocity_hat,
)
    nd = length(ks); ns = size(velocity_hat)[1:nd]; D = size(velocity_hat, nd + 1)
    FT = real(eltype(velocity_hat)); nb = length(bws.centers); W = bws.W
    inner = ComputationalBackends.local_backend(execution)
    nw = max(1, Distributed.nworkers())

    Tred = Distributed.@distributed (+) for w in 1:nw
        _band_to_band_worker_T(collect(w:nw:nb), W, velocity_hat, ks, ns, D, FT, nb,
            invariant, dealiasing, spectral, advecting_hat, inner)
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
        @inbounds for m in mrng
            for c in 1:D, I in CartesianIndices(ns)
                fm[I, c] = W[m][I] * velocity_hat[I, c]
            end
            FIT.NonlinearTerm.compute_nonlinear_term!(lws, fm, ks;
                dealiasing = dealiasing, spectral = spectral, advecting_hat = advecting_hat)
            FIT.Invariants.transfer_density!(d, invariant, velocity_hat, lws.N̂, ks)
            for n in 1:nb
                s = zero(FT)
                for I in CartesianIndices(ns); s += W[n][I] * d[I]; end
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
# Each (sp,sq) pair is an independent nonlinear-term build writing DISJOINT channel keys; pairs are
# partitioned across workers (strided), each worker builds a partial channel Dict, `merge`-reduced.
# Inner backend threads the worker's pair loop; each task keeps its own workspace + a private Dict,
# merged within the worker (no shared-Dict race). Captures only `comps`/`sidx`/`centers` (plain arrays).
function FIT.SpectralFlux._partial_fluxes_distributed!(
    channels, ws, comps, names, velocity_hat, ks, sidx, centers, Nsh, execution::ComputationalBackends.DistributedBackend;
    dealiasing = FIT.Types.OrszagTwoThirds(),
    spectral = SpectralBackends.DirectSumSpectralBackend(),
)
    FT = real(eltype(velocity_hat)); nd = length(ks); ns = size(velocity_hat)[1:nd]
    prs = [(sp, sq) for sp in names for sq in names]; np = length(prs)
    inner = ComputationalBackends.local_backend(execution)
    nw = max(1, Distributed.nworkers())

    merged = Distributed.@distributed (merge) for w in 1:nw
        _partial_fluxes_worker(collect(w:nw:np), prs, comps, names, velocity_hat, ks, ns, FT,
            sidx, centers, Nsh, dealiasing, spectral, inner)
    end
    merge!(channels, merged)
    return channels
end

function _partial_fluxes_worker(pis, prs, comps, names, velocity_hat, ks, ns, FT, sidx, centers, Nsh,
                                dealiasing, spectral, inner)
    build(pirng) = begin
        lws = FIT.Workspaces.NonlinearTermWorkspace(velocity_hat, ks; dealiasing = dealiasing)
        d = similar(velocity_hat, FT, ns...)
        ch = Dict{NTuple{3,Symbol}, FIT.Types.SpectralFluxResult}()
        @inbounds for pi in pirng
            sp, sq = prs[pi]
            FIT.NonlinearTerm.compute_nonlinear_term!(lws, comps[sq], ks;
                advecting_hat = comps[sp], dealiasing = dealiasing, spectral = spectral)
            for sk in names
                FIT.Invariants.transfer_density!(d, FIT.Types.KineticEnergy(), comps[sk], lws.N̂, ks)
                ch[(sk, sp, sq)] = FIT.SpectralFlux._partial_binflux(d, sidx, centers, Nsh)
            end
        end
        return ch
    end
    if inner isa ComputationalBackends.ThreadedBackend
        nchunks = max(1, min(Threads.nthreads(), length(pis)))
        chunks = [pis[c:nchunks:end] for c in 1:nchunks]
        parts = Vector{Dict{NTuple{3,Symbol}, FIT.Types.SpectralFluxResult}}(undef, length(chunks))
        Threads.@threads for ci in eachindex(chunks)
            parts[ci] = build(chunks[ci])
        end
        out = Dict{NTuple{3,Symbol}, FIT.Types.SpectralFluxResult}()
        for p in parts; merge!(out, p); end
        return out
    else
        return build(pis)
    end
end

# ---------------------------------------------------------------------------
# Distributed compressible transfer  (overrides the core `_compressible_distributed` stub)
# ---------------------------------------------------------------------------
# Compressible is a single FFT pipeline with no outer loop, so its independent coarse-grained work
# units are the momentum-weighted transfer + Helmholtz channel set (group A) and the KE↔IE
# pressure-dilatation (group B). Each group runs on its own worker, rebuilding its own
# `CompressibleWorkspace` from the raw (plain-array) inputs — no shared mutable state to serialize —
# and the master assembles the full result. The inner backend (`ComputationalBackends.local_backend`) is used per worker, so
# `ComputationalBackends.DistributedBackend(ComputationalBackends.ThreadedBackend())` threads each worker's FFTs. When no pressure field is given
# there is a single work unit, offloaded to one worker. Bit-identical to the serial pipeline.
function FIT.Compressible._compressible_distributed(
    velocity_hat, density_hat, ks, execution::ComputationalBackends.DistributedBackend;
    spectral, binning, geometry,
    pressure_hat = nothing,
    dealiasing = FIT.Types.OrszagTwoThirds(),
    decompose = true,
)
    inner = ComputationalBackends.local_backend(execution)
    runA = () -> begin
        ws = FIT.Compressible.CompressibleWorkspace(velocity_hat, ks;
            spectral = spectral, binning = binning, geometry = geometry, execution = inner)
        FIT.Compressible.calculate_compressible_flux!(ws, velocity_hat, density_hat, ks;
            dealiasing = dealiasing, decompose = decompose, pressure_hat = nothing)
    end
    if pressure_hat === nothing
        return fetch(Distributed.@spawnat :any runA())
    end
    runB = () -> begin
        ws = FIT.Compressible.CompressibleWorkspace(velocity_hat, ks;
            spectral = spectral, binning = binning, geometry = geometry, execution = inner)
        FIT.Compressible.calculate_compressible_flux!(ws, velocity_hat, density_hat, ks;
            dealiasing = dealiasing, decompose = false, pressure_hat = pressure_hat)
    end
    fA = Distributed.@spawnat :any runA()
    fB = Distributed.@spawnat :any runB()
    rA = fetch(fA); rB = fetch(fB)
    return FIT.Types.CompressibleFluxResult(rA.k_shells, rA.transfer_spectrum, rA.flux,
                                            rA.channels, rB.pressure_dilatation)
end

# ---------------------------------------------------------------------------
# Distributed BATCH methods (snapshots are the outer parallel axis)
# ---------------------------------------------------------------------------
# Each overrides a core `_*_batch_distributed!` stub. Snapshots are partitioned into contiguous chunks
# (one per worker), `pmap`'d in order; each worker builds its OWN workspace (the FFTW-plan workspace is
# not serializable) and reuses it across its chunk with a serial inner transform. Results are reassembled
# in input order and are bit-identical to the serial batch. (`velocity_hats` is captured whole by the
# closure; a subset-only send is a later bandwidth optimization.)

function FIT.SpectralFlux._spectral_flux_batch_distributed!(results, velocity_hats, ks, shell_idx;
                                                            binning, dealiasing, invariant, spectral, geometry)
    n = length(velocity_hats); û1 = first(velocity_hats); RT = real(eltype(û1))
    centers = FIT.ShellBinning.shell_centers(binning, maximum(FIT.ShellBinning.shell_coordinate(geometry, ks)))
    nw = max(1, Distributed.nworkers())
    chunks = collect(Iterators.partition(1:n, max(1, cld(n, nw))))
    chunk_results = Distributed.pmap(chunks) do chunk
        ws = FIT.Workspaces.SpectralFluxWorkspace(û1, ks, binning; geometry = geometry, dealiasing = dealiasing)
        map(chunk) do i
            res = FIT.Types.SpectralFluxResult(centers, similar(centers, RT, length(centers)), similar(centers, RT, length(centers)))
            FIT.SpectralFlux.calculate_spectral_flux!(res, ws, velocity_hats[i], ks, shell_idx;
                dealiasing = dealiasing, invariant = invariant, spectral = spectral, execution = ComputationalBackends.SerialBackend())
            res
        end
    end
    flat = reduce(vcat, chunk_results)
    for i in 1:n; results[i] = flat[i]; end
    return results
end

function FIT.ShellToShellTransfer._shell_to_shell_batch_distributed!(results, velocity_hats, ks, centers, edges, N_sh;
                                                                     binning, dealiasing, verify_antisymmetry, invariant, spectral, geometry)
    n = length(velocity_hats); û1 = first(velocity_hats); FT = real(eltype(û1))
    nw = max(1, Distributed.nworkers())
    chunks = collect(Iterators.partition(1:n, max(1, cld(n, nw))))
    chunk_results = Distributed.pmap(chunks) do chunk
        ws = FIT.Workspaces.ShellToShellWorkspace(û1, ks, binning; geometry = geometry, dealiasing = dealiasing)
        map(chunk) do i
            vh = velocity_hats[i]
            T_mat = Matrix{FT}(undef, N_sh, N_sh); net = Vector{FT}(undef, N_sh)
            res = FIT.Types.ShellToShellResult(centers, edges, T_mat, net, FT(NaN))
            ma = FIT.ShellToShellTransfer._calculate_shell_to_shell!(res, ws, vh, ks, ComputationalBackends.SerialBackend(), spectral;
                dealiasing = dealiasing, verify_antisymmetry = verify_antisymmetry, invariant = invariant, advecting_hat = vh)
            FIT.Types.ShellToShellResult(centers, edges, T_mat, net, ma)
        end
    end
    flat = reduce(vcat, chunk_results)
    for i in 1:n; results[i] = flat[i]; end
    return results
end

function FIT.BandTransfer._band_to_band_batch_distributed!(results, velocity_hats, ks, bands, nb;
                                                           dealiasing, invariant, spectral, geometry)
    n = length(velocity_hats); û1 = first(velocity_hats); FT = real(eltype(û1))
    nw = max(1, Distributed.nworkers())
    chunks = collect(Iterators.partition(1:n, max(1, cld(n, nw))))
    chunk_results = Distributed.pmap(chunks) do chunk
        bws = FIT.BandTransfer.BandTransferWorkspace(û1, ks, bands; geometry = geometry)
        map(chunk) do i
            vh = velocity_hats[i]
            T = zeros(FT, nb, nb); net = zeros(FT, nb)
            FIT.BandTransfer.calculate_band_to_band_transfer!(T, net, bws, vh, ks;
                dealiasing = dealiasing, invariant = invariant, spectral = spectral,
                execution = ComputationalBackends.SerialBackend(), advecting_hat = vh)
        end
    end
    flat = reduce(vcat, chunk_results)
    for i in 1:n; results[i] = flat[i]; end
    return results
end

function FIT.CoarseGrainingFlux._cg_flux_batch_distributed!(velocity_fields_batch, coords_vecs, ℓ, filter; kwargs...)
    n = length(velocity_fields_batch)
    nw = max(1, Distributed.nworkers())
    chunks = collect(Iterators.partition(1:n, max(1, cld(n, nw))))
    chunk_results = Distributed.pmap(chunks) do chunk
        ws = FIT.CoarseGrainingFlux.CoarseGrainingFluxWorkspace(velocity_fields_batch[first(chunk)], coords_vecs, ℓ, filter; kwargs...)
        [deepcopy(FIT.CoarseGrainingFlux.calculate_coarse_graining_flux!(ws, velocity_fields_batch[i])) for i in chunk]
    end
    return reduce(vcat, chunk_results)
end

end # module
