module FlowInvariantTransferDistributedExt

using Distributed: Distributed
using SharedArrays: SharedArrays
using FlowInvariantTransfer: FlowInvariantTransfer as FET
using FlowInvariantTransfer.Types: DistributedBackend, ThreadedBackend, ShellToShellResult, AbstractInvariant, KineticEnergy, local_backend
using FlowInvariantTransfer.ShellBinning: assign_shells

# Distributed Shell-to-Shell Transfer Implementation
function FET.ShellToShellTransfer._calculate_shell_to_shell!(
    result::ShellToShellResult,
    ws::FET.Workspaces.ShellToShellWorkspace,
    velocity_hat,
    ks,
    execution::DistributedBackend,
    spectral;            # transform backend, passed to each per-mediator nonlinear term
    dealiasing::FET.Types.AbstractDealiasing,
    verify_antisymmetry::Bool,
    invariant::AbstractInvariant = KineticEnergy(),
    advecting_hat = velocity_hat,
)
    N_sh = size(result.transfer_matrix, 1)
    FT = real(eltype(velocity_hat))
    inner = local_backend(execution)   # per-worker backend (Serial default, or Threaded for hybrid)

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
# Overrides the core `_spectral_flux_distributed!` stub dispatched by `DistributedBackend`. The
# transfer density is computed on the caller, then the mode→shell scatter is partitioned across
# workers (strided over the linear mode index) and `+`-reduced into the shell spectrum.
#
# NOTE: this distributes only the reduction, not the (dominant) nonlinear-term FFT, and it ships the
# density array to the workers — so for a single in-memory field it is rarely faster than Serial /
# Threaded. To distribute a grid too large for one node (splitting the FFT itself), use
# `pencil_spectral_flux` (PencilFFTs). Provided here for API parity with the other diagnostics and
# for pipelines whose field is already worker-resident.
function FET.SpectralFlux._spectral_flux_distributed!(
    result,
    ws,
    velocity_hat,
    N̂,
    ks,
    shell_idx;
    invariant::AbstractInvariant = KineticEnergy(),
)
    nd   = length(ks)
    ns   = size(velocity_hat)[1:nd]
    FT   = real(eltype(velocity_hat))
    N_sh = length(ws.T_spec)
    Np   = prod(ns)

    FET.Invariants.transfer_density!(ws.transfer_density, invariant, velocity_hat, N̂, ks)
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
    return FET.SpectralFlux._finalize_spectral_flux!(result, ws)
end

# Helper function executed on worker processes for Shell-to-Shell
function compute_mediator_transfer_column(m, velocity_hat, ks, shell_idx, N_sh, invariant, dealiasing, FT, spectral, advecting_hat=velocity_hat, inner=FET.Types.SerialBackend())
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
    nl_ws = FET.Workspaces.NonlinearTermWorkspace(velocity_hat, ks)
    FET.NonlinearTerm.compute_nonlinear_term!(nl_ws, û_m, ks; dealiasing=dealiasing,
        spectral=spectral, advecting_hat=advecting_hat)

    # Write per-mode transfer density
    transfer_density = similar(velocity_hat, FT, ns...)
    FET.Invariants.transfer_density!(transfer_density, invariant, velocity_hat, nl_ws.N̂, ks)

    # Accumulate into the column vector. `inner` is the worker-local execution backend from
    # `local_backend`: SerialBackend (default) sums receiver shells serially; ThreadedBackend threads
    # that reduction with the worker's own threads (Base `@threads`, no cross-extension dependency) —
    # this realises DistributedBackend(ThreadedBackend()) when workers are started with `-t N`. Each
    # receiver shell n writes a disjoint col[n], so the threaded loop is race-free.
    col = zeros(FT, N_sh)
    if inner isa ThreadedBackend
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

end # module
