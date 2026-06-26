module FlowInvariantTransferDistributedExt

using Distributed: Distributed
using SharedArrays: SharedArrays
using FlowInvariantTransfer: FlowInvariantTransfer as FET
using FlowInvariantTransfer.Types: DistributedBackend, ShellToShellResult, AbstractInvariant, KineticEnergy
using FlowInvariantTransfer.ShellBinning: assign_shells

# Distributed Shell-to-Shell Transfer Implementation
function FET.ShellToShellTransfer._calculate_shell_to_shell!(
    result::ShellToShellResult,
    ws::FET.Workspaces.ShellToShellWorkspace,
    velocity_hat,
    ks,
    ::DistributedBackend,
    spectral;            # transform backend, passed to each per-mediator nonlinear term
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
        col = compute_mediator_transfer_column(m, velocity_hat, ks, shell_idx, N_sh, invariant, dealiasing, FT, spectral)
        
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
function compute_mediator_transfer_column(m, velocity_hat, ks, shell_idx, N_sh, invariant, dealiasing, FT, spectral)
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
        spectral=spectral, advecting_hat=velocity_hat)

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

end # module
