module FlowEnergyTransferFFTWExt

using FFTW: FFTW
using FlowEnergyTransfer: FlowEnergyTransfer as FET
using FlowEnergyTransfer.Types: AbstractShellBinning, LinearBinning, ShellToShellResult
using FlowEnergyTransfer.ShellBinning: shell_edges, shell_centers, n_shells, assign_shells
using FlowEnergyTransfer.Utils: wavenumber_magnitude_grid
using FlowEnergyTransfer.Workspaces: ShellToShellWorkspace

# ---------------------------------------------------------------------------
# Override NonlinearTerm._nonlinear_term_fft
# ---------------------------------------------------------------------------

"""
    _nonlinear_term_fft(N̂, velocity_hat, ks; dealiasing=true)

FFT-accelerated computation of the nonlinear advection term N̂(k) = FFT[(u·∇)u].

Algorithm (pseudospectral):
1. u_i(x)   = IFFT(û_i(k))
2. ∂u_i/∂x_j(x) = IFFT(i·k_j · û_i(k))
3. N_i(x)   = Σ_j u_j(x) · ∂u_i/∂x_j(x)
4. N̂_i(k)  = FFT(N_i(x))
5. Apply 2/3 dealiasing mask.
"""
function FET.NonlinearTerm._nonlinear_term_fft(
    N̂,
    velocity_hat,
    ks::Tuple;
    dealiasing::Bool = true,
)
    nd  = length(ks)
    ns  = size(velocity_hat)[1:nd]
    D   = size(velocity_hat, nd+1)
    FT  = real(eltype(velocity_hat))
    Np  = prod(ns)

    k_comp = [_build_k_component_fft(ks, d, ns) for d in 1:nd]

    u_phys = [real.(FFTW.ifft(selectdim(velocity_hat, nd+1, c))) for c in 1:D]
    grad   = [[real.(FFTW.ifft(im .* k_comp[j] .* selectdim(velocity_hat, nd+1, c)))
               for j in 1:nd] for c in 1:D]
    N_phys = [sum(u_phys[j] .* grad[c][j] for j in 1:nd) for c in 1:D]

    for c in 1:D
        selectdim(N̂, nd+1, c) .= FFTW.fft(N_phys[c]) ./ FT(Np)
    end

    if dealiasing
        for I in CartesianIndices(ns)
            kill = false
            for d in 1:nd
                k_idx = I[d] - 1
                k_abs = k_idx <= ns[d] ÷ 2 ? k_idx : ns[d] - k_idx
                k_abs >= ns[d] ÷ 3 && (kill = true; break)
            end
            if kill
                for c in 1:D; N̂[I, c] = zero(eltype(N̂)); end
            end
        end
    end

    return N̂
end

# ---------------------------------------------------------------------------
# Override ShellToShellTransfer._shell_to_shell_fft
# ---------------------------------------------------------------------------

"""
    _shell_to_shell_fft!(result, ws, velocity_hat, ks; dealiasing, verify_antisymmetry)

FFT-accelerated shell-to-shell energy transfer T(n,m) using Alexakis et al. (2005)
antisymmetric definition. Writes into `result` using workspace `ws`.
Reuses ws.û_m and ws.nonlinear.N̂ buffers per mediator shell — no N_sh-fold allocations.
"""
function FET.ShellToShellTransfer._shell_to_shell_fft!(
    result::ShellToShellResult,
    ws::ShellToShellWorkspace,
    velocity_hat,
    ks::Tuple;
    dealiasing::Bool = true,
    verify_antisymmetry::Bool = true,
)
    nd    = length(ks)
    ns    = size(velocity_hat)[1:nd]
    D     = size(velocity_hat, nd+1)
    FT    = real(eltype(velocity_hat))
    N_sh  = size(result.transfer_matrix, 1)
    Np    = FT(prod(ns))

    k_comp    = [_build_k_component_fft(ks, d, ns) for d in 1:nd]
    grad_full = [[real.(FFTW.ifft(im .* k_comp[j] .* selectdim(velocity_hat, nd+1, c)))
                  for j in 1:nd] for c in 1:D]

    # Precompute N̂_m for each shell into a single reused buffer
    # Store each result in a preallocated N_sh-length vector of views
    # We need to hold all N̂_m simultaneously for the antisymmetric formula,
    # so allocate one array per shell (N_sh allocations, unavoidable for Alexakis form)
    N̂_all = [similar(velocity_hat) for _ in 1:N_sh]

    for m in 1:N_sh
        fill!(ws.û_m, zero(eltype(ws.û_m)))
        for I in CartesianIndices(ns)
            ws.shell_idx[I] == m || continue
            for c in 1:D; ws.û_m[I, c] = velocity_hat[I, c]; end
        end
        u_m_phys = [real.(FFTW.ifft(selectdim(ws.û_m, nd+1, c))) for c in 1:D]
        N_phys_m = [sum(u_m_phys[j] .* grad_full[c][j] for j in 1:nd) for c in 1:D]
        for c in 1:D
            selectdim(N̂_all[m], nd+1, c) .= FFTW.fft(N_phys_m[c]) ./ Np
        end
        if dealiasing
            for I in CartesianIndices(ns)
                kill = false
                for d in 1:nd
                    k_idx = I[d] - 1
                    k_abs = k_idx <= ns[d] ÷ 2 ? k_idx : ns[d] - k_idx
                    k_abs >= ns[d] ÷ 3 && (kill = true; break)
                end
                kill && (for c in 1:D; N̂_all[m][I, c] = zero(eltype(N̂_all[m])); end)
            end
        end
    end

    # Antisymmetric T(n,m) = ½[Σ_{S_n} Re(û* N̂_m) - Σ_{S_m} Re(û* N̂_n)]
    fill!(result.transfer_matrix, zero(FT))
    for n in 1:N_sh
        for m in 1:N_sh
            n == m && continue
            s_nm = zero(FT)
            s_mn = zero(FT)
            for I in CartesianIndices(ns)
                c_n = ws.shell_idx[I] == n
                c_m = ws.shell_idx[I] == m
                (c_n || c_m) || continue
                for c in 1:D
                    uc = conj(velocity_hat[I, c])
                    c_n && (s_nm += real(uc * N̂_all[m][I, c]))
                    c_m && (s_mn += real(uc * N̂_all[n][I, c]))
                end
            end
            result.transfer_matrix[n, m] = FT(0.5) * (s_nm - s_mn)
        end
    end

    for n in 1:N_sh
        s = zero(FT)
        for m in 1:N_sh; s += result.transfer_matrix[n, m]; end
        result.net_transfer[n] = s
    end

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
# Helpers
# ---------------------------------------------------------------------------

function _build_k_component_fft(ks::Tuple, d::Int, ns::Tuple)
    nd = length(ns)
    FT = eltype(ks[1])
    kc = zeros(FT, ns...)
    for I in CartesianIndices(ns)
        kc[I] = ks[d][I[d]]
    end
    return kc
end

end # module FlowEnergyTransferFFTWExt
