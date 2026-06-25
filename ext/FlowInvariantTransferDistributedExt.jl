module FlowInvariantTransferDistributedExt

using Distributed: Distributed
using SharedArrays: SharedArrays
using FlowInvariantTransfer: FlowInvariantTransfer as FET
using FlowInvariantTransfer.Types: DistributedBackend, ShellToShellResult, AbstractShellBinning, AbstractInvariant, KineticEnergy
using FlowInvariantTransfer.Workspaces: ScaleToScaleWorkspace
using FlowInvariantTransfer.ShellBinning: assign_shells
using FlowInvariantTransfer.Utils: dealiasing_mask

# Define a custom reduced structure for mode-to-mode reduction to optimize communication
struct ScaleToScaleReduced{A, M}
    net_transfer::A
    T_mat::M
end

function Base.:+(a::ScaleToScaleReduced, b::ScaleToScaleReduced)
    return ScaleToScaleReduced(a.net_transfer .+ b.net_transfer, a.T_mat .+ b.T_mat)
end

# 1. Distributed Shell-to-Shell Transfer Implementation
function FET.ShellToShellTransfer._calculate_shell_to_shell!(
    result::ShellToShellResult,
    ws::FET.Workspaces.ShellToShellWorkspace,
    velocity_hat,
    ks,
    ::DistributedBackend;
    dealiasing::Bool,
    verify_antisymmetry::Bool,
    invariant::AbstractInvariant = KineticEnergy(),
)
    N_sh = size(result.transfer_matrix, 1)
    FT = real(eltype(velocity_hat))

    # Hoist shell_idx to a local so the @distributed closure captures only this plain
    # Int array — NOT the whole `ws`, whose nonlinear workspace may hold an FFTW-ext plan
    # bundle that workers can't deserialize (they need not have FFTWExt loaded).
    shell_idx = ws.shell_idx

    # We distribute the computation over the mediator shells `m`.
    # Using `Distributed.@distributed (+)` reduces the resulting N_sh x N_sh matrices.
    T_mat_reduced = Distributed.@distributed (+) for m in 1:N_sh
        # Compute column m on the worker process
        col = compute_mediator_transfer_column(m, velocity_hat, ks, shell_idx, N_sh, invariant, dealiasing, FT)
        
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

# Helper function executed on worker processes for Shell-to-Shell
function compute_mediator_transfer_column(m, velocity_hat, ks, shell_idx, N_sh, invariant, dealiasing, FT)
    nd = length(ks)
    ns = size(velocity_hat)[1:nd]
    D  = size(velocity_hat, nd+1)
    
    # Restrict velocity field to shell m
    û_m = zeros(eltype(velocity_hat), size(velocity_hat)...)
    for I in CartesianIndices(ns)
        shell_idx[I] == m || continue
        for comp in 1:D
            û_m[I, comp] = velocity_hat[I, comp]
        end
    end
    
    # Allocate a local NonlinearTermWorkspace.
    # N̂_m = (u·∇)u_m: full velocity advects the band-m field (AMP 2005) — antisymmetric A[n,m]
    # that reduces to transfer_spectrum[n] (matches serial/FFT/threaded).
    nl_ws = FET.Workspaces.NonlinearTermWorkspace(velocity_hat, ks)
    FET.NonlinearTerm.compute_nonlinear_term!(nl_ws, û_m, ks; dealiasing=dealiasing,
        backend=FET.SerialBackend(), advecting_hat=velocity_hat)

    # Write per-mode transfer density
    transfer_density = similar(velocity_hat, FT, ns...)
    FET.Invariants.transfer_density!(transfer_density, invariant, velocity_hat, nl_ws.N̂, ks)
    
    # Accumulate into column vector
    col = zeros(FT, N_sh)
    for n in 1:N_sh
        s = zero(FT)
        for I in CartesianIndices(ns)
            shell_idx[I] == n || continue
            s += transfer_density[I]
        end
        col[n] = s
    end
    return col
end

# 2. Distributed Scale-to-Scale (Mode-to-Mode) Transfer Implementation
function FET.ScaleToScaleTransfer._calculate_mode_to_mode!(
    ws::ScaleToScaleWorkspace,
    velocity_hat,
    ks,
    ::DistributedBackend;
    binning::Union{Nothing, AbstractShellBinning},
    invariant::AbstractInvariant,
    dealiasing::Bool,
)
    nd = length(ks)
    ns = size(velocity_hat)[1:nd]
    D  = size(velocity_hat, nd+1)
    FT = real(eltype(velocity_hat))
    
    N_sh = binning !== nothing ? size(ws.T_mat, 1) : 0
    mask = dealiasing ? dealiasing_mask(ns) : trues(ns...)
    
    # Gather indices to distribute
    indices_list = collect(CartesianIndices(ns))
    n_total = length(indices_list)
    n_workers = Distributed.nworkers()
    
    # Divide total modes into chunks to minimize worker serialization/reduction overhead
    chunk_size = div(n_total, n_workers)
    rem_size = rem(n_total, n_workers)
    
    reduced = Distributed.@distributed (+) for c in 1:n_workers
        start_idx = (c - 1) * chunk_size + min(c - 1, rem_size) + 1
        end_idx = c * chunk_size + min(c, rem_size)
        
        local_net = zeros(FT, ns...)
        local_T = binning !== nothing ? zeros(FT, N_sh, N_sh) : zeros(FT, 0, 0)
        
        for idx in start_idx:end_idx
            k_idx = indices_list[idx]
            if dealiasing && !mask[k_idx]
                continue
            end

            net_k = zero(FT)
            for p_idx in CartesianIndices(ns)
                if dealiasing && !mask[p_idx]
                    continue
                end

                # q = k - p
                q_idx = CartesianIndex(Tuple(mod(k_idx[d] - p_idx[d], ns[d]) + 1 for d in 1:nd))
                if dealiasing && !mask[q_idx]
                    continue
                end

                S_val = FET.ScaleToScaleTransfer.compute_triad_S(invariant, velocity_hat, k_idx, p_idx, q_idx, ks, D)
                net_k += S_val

                if binning !== nothing
                    K_sh = ws.shell_idx[k_idx]
                    Q_sh = ws.shell_idx[p_idx]
                    if K_sh > 0 && Q_sh > 0
                        local_T[K_sh, Q_sh] += S_val
                    end
                end
            end
            local_net[k_idx] = net_k
        end
        ScaleToScaleReduced(local_net, local_T)
    end
    
    # Copy reduced results into our in-place workspace
    copyto!(ws.net_transfer, reduced.net_transfer)
    if binning !== nothing
        copyto!(ws.T_mat, reduced.T_mat)
    end
end

end # module
