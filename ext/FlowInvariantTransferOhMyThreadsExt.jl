module FlowInvariantTransferOhMyThreadsExt

using OhMyThreads: OhMyThreads
using LinearAlgebra: LinearAlgebra
using FlowInvariantTransfer: FlowInvariantTransfer as FET
using FlowInvariantTransfer.Types: AbstractInvariant, KineticEnergy

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

Each task allocates its own `NonlinearTermWorkspace` because the shared workspace
cannot be reused concurrently across threads.
"""
function FET.ShellToShellTransfer._shell_to_shell_threaded!(
    result,
    ws,
    velocity_hat,
    ks::Tuple;
    dealiasing::Bool = true,
    verify_antisymmetry::Bool = true,
    invariant::AbstractInvariant = KineticEnergy(),
)
    nd        = length(ks)
    ns        = size(velocity_hat)[1:nd]
    D         = size(velocity_hat, nd+1)
    FT        = real(eltype(velocity_hat))
    N_sh      = size(result.transfer_matrix, 1)
    shell_idx = ws.shell_idx

    fill!(result.transfer_matrix, zero(FT))

    # Thread-parallel over mediator shells. Each task writes only column m.
    OhMyThreads.@tasks for m in 1:N_sh
        local_ws = FET.Workspaces.NonlinearTermWorkspace(velocity_hat, ks)
        û_m      = similar(velocity_hat)
        fill!(û_m, zero(eltype(û_m)))
        for I in CartesianIndices(ns)
            shell_idx[I] == m || continue
            for c in 1:D
                û_m[I, c] = velocity_hat[I, c]
            end
        end

        FET.NonlinearTerm.compute_nonlinear_term!(local_ws, û_m, ks;
            dealiasing=dealiasing, backend=FET.SerialBackend())
        N̂_m = local_ws.N̂

        local_density = Array{FT}(undef, ns...)
        FET.Invariants.transfer_density!(local_density, invariant, velocity_hat, N̂_m, ks)

        for n in 1:N_sh
            s = zero(FT)
            for I in CartesianIndices(ns)
                shell_idx[I] == n || continue
                s += local_density[I]
            end
            result.transfer_matrix[n, m] = s
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
# Override TriadicOrthogonalDecomposition._triadic_loop_threaded!
# ---------------------------------------------------------------------------

"""
    _triadic_loop_threaded!(...)

Thread-parallel triad loop using OhMyThreads. Each triad is independent
(read-only Q_hat, writes to separate Dict slots), so this is embarrassingly parallel.
"""
function FET.TriadicOrthogonalDecomposition._triadic_loop_threaded!(
    L, P, T_budget, A_out, Xi_out,
    Q_hat, f_idx, fk_idx, fl_idx, fn_idx,
    weights, nBlks, nFreq, nState, nx, nmode,
    Q_nonlinear, LHS,
    return_coefficients, return_auxiliary_modes,
)
    nTriads = length(fk_idx)
    nStateNx = nState * nx

    lk = ReentrantLock()

    OhMyThreads.@tasks for i in 1:nTriads
        fi_k = fk_idx[i]
        fi_l = fl_idx[i]
        fi_n = fn_idx[i]

        Q_n_raw = Q_hat[fi_n, :, :, :]
        Q_k_raw = Q_hat[fi_k, :, :, :]
        Q_l_raw = Q_hat[fi_l, :, :, :]

        Q_hat_n = reshape(permutedims(LHS(Q_n_raw), (2, 1, 3)), nStateNx, nBlks)
        Q_hat_kl = reshape(Q_nonlinear(Q_k_raw, Q_l_raw), nStateNx, nBlks)

        U, s, V = FET.TriadicOrthogonalDecomposition.triadic_svd(Q_hat_n, Q_hat_kl, weights, nBlks)

        nm = min(nmode, length(s))
        u = U[:, 1:nm]
        v = V[:, 1:nm]

        # L and T_budget are preallocated and each triad writes to disjoint (fi_l, fi_n) slices, so this is thread-safe
        for j in 1:nm
            L[fi_l, fi_n, j] = s[j]
            T_budget[fi_l, fi_n, j] = s[j] * real(LinearAlgebra.dot(v[:, j], weights .* u[:, j]))
        end

        # Dict updates are protected by a lock to prevent concurrency corruption
        lock(lk) do
            P[(fi_l, fi_n)] = (convective=u, recipient=v)
        end

        if return_coefficients
            A_conv = u' * (Q_hat_kl .* weights)
            A_recip = v' * (Q_hat_n .* weights)
            lock(lk) do
                A_out[(fi_l, fi_n)] = (convective=A_conv, recipient=A_recip)
            end

            if return_auxiliary_modes
                Q_hat_l = reshape(permutedims(LHS(Q_l_raw), (2, 1, 3)), nStateNx, nBlks)
                Q_hat_k = reshape(permutedims(LHS(Q_k_raw), (2, 1, 3)), nStateNx, nBlks)
                inv_s = 1 ./ s[1:nm]
                donor_mode = Q_hat_l * A_recip' * LinearAlgebra.Diagonal(inv_s) ./ nBlks
                catalyst_mode = Q_hat_k * A_recip' * LinearAlgebra.Diagonal(inv_s) ./ nBlks
                lock(lk) do
                    Xi_out[(fi_l, fi_n)] = (donor=donor_mode[:, 1:nm], catalyst=catalyst_mode[:, 1:nm])
                end
            end
        end
    end
end

# ---------------------------------------------------------------------------
# Thread-parallel mode-to-mode transfer
# ---------------------------------------------------------------------------

function FET.ScaleToScaleTransfer._mode_to_mode_threaded!(
    ws,
    velocity_hat,
    ks::Tuple;
    binning,
    invariant,
    dealiasing,
)
    nd = length(ks)
    ns = size(velocity_hat)[1:nd]
    D  = size(velocity_hat, nd+1)
    FT = real(eltype(velocity_hat))

    fill!(ws.net_transfer, zero(FT))
    if binning !== nothing
        fill!(ws.T_mat, zero(FT))
    end

    mask = dealiasing ? FET.Utils.dealiasing_mask(ns) : trues(ns...)
    lk = ReentrantLock()

    OhMyThreads.@tasks for k_idx in CartesianIndices(ns)
        if !dealiasing || mask[k_idx]
            net_k = zero(FT)
            for p_idx in CartesianIndices(ns)
                if !dealiasing || mask[p_idx]
                    # q = k - p
                    q_idx = CartesianIndex(Tuple(mod(k_idx[d] - p_idx[d], ns[d]) + 1 for d in 1:nd))
                    if !dealiasing || mask[q_idx]
                        S_val = FET.ScaleToScaleTransfer.compute_triad_S(invariant, velocity_hat, k_idx, p_idx, q_idx, ks, D)
                        net_k += S_val

                        if binning !== nothing
                            K_sh = ws.shell_idx[k_idx]
                            Q_sh = ws.shell_idx[p_idx]
                            if K_sh > 0 && Q_sh > 0
                                lock(lk) do
                                    ws.T_mat[K_sh, Q_sh] += S_val
                                end
                            end
                        end
                    end
                end
            end
            ws.net_transfer[k_idx] = net_k
        else
            ws.net_transfer[k_idx] = zero(FT)
        end
    end
end

end # module FlowInvariantTransferOhMyThreadsExt
