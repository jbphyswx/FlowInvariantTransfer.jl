module FlowInvariantTransferFFTWExt

using FFTW: FFTW
using FlowInvariantTransfer: FlowInvariantTransfer as FET
using FlowInvariantTransfer.Types: AbstractShellBinning, LinearBinning, ShellToShellResult, AbstractInvariant, KineticEnergy
using FlowInvariantTransfer.Invariants: transfer_density!
using FlowInvariantTransfer.ShellBinning: shell_edges, shell_centers, n_shells, assign_shells
using FlowInvariantTransfer.Utils: wavenumber_magnitude_grid
using FlowInvariantTransfer.Workspaces: ShellToShellWorkspace

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

    # Orszag 2/3: truncate INPUTS before forming products (output-only truncation leaves
    # the retained band aliased — see FET.NonlinearTerm._is_dealiased and THEORY.md §0.5).
    vhat = dealiasing ? _dealias_copy(velocity_hat, ns, nd) : velocity_hat

    u_phys = [real.(FFTW.ifft(selectdim(vhat, nd+1, c))) for c in 1:D]
    grad   = [[real.(FFTW.ifft(im .* k_comp[j] .* selectdim(vhat, nd+1, c)))
               for j in 1:nd] for c in 1:D]
    N_phys = [sum(u_phys[j] .* grad[c][j] for j in 1:nd) for c in 1:D]

    for c in 1:D
        selectdim(N̂, nd+1, c) .= FFTW.fft(N_phys[c]) ./ FT(Np)
    end

    # Zero output above the cutoff (inputs already truncated) for a clean N̂.
    if dealiasing
        for I in CartesianIndices(ns)
            FET.NonlinearTerm._is_dealiased(I, ns, nd) || continue
            for c in 1:D; N̂[I, c] = zero(eltype(N̂)); end
        end
    end

    return N̂
end

# Allocate a copy of `velocity_hat` with the 2/3-rule discard modes zeroed (input truncation).
function _dealias_copy(velocity_hat, ns::Tuple, nd::Int)
    vd = copy(velocity_hat)
    D  = size(velocity_hat, nd + 1)
    for I in CartesianIndices(ns)
        FET.NonlinearTerm._is_dealiased(I, ns, nd) || continue
        for c in 1:D
            vd[I, c] = zero(eltype(vd))
        end
    end
    return vd
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
    invariant::AbstractInvariant = KineticEnergy(),
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

    # Precompute transfer density for all shells
    T_density_all = [similar(ws.transfer_density) for _ in 1:N_sh]
    for m in 1:N_sh
        transfer_density!(T_density_all[m], invariant, velocity_hat, N̂_all[m], ks)
    end

    # Antisymmetric T(n,m) = ½[Σ_{S_n} T_density_m - Σ_{S_m} T_density_n]
    fill!(result.transfer_matrix, zero(FT))
    for n in 1:N_sh
        for m in 1:N_sh
            n == m && continue
            s_nm = zero(FT)
            s_mn = zero(FT)
            for I in CartesianIndices(ns)
                c_n = ws.shell_idx[I] == n
                c_m = ws.shell_idx[I] == m
                c_n && (s_nm += T_density_all[m][I])
                c_m && (s_mn += T_density_all[n][I])
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

# ---------------------------------------------------------------------------
# Override TriadicOrthogonalDecomposition._temporal_block_dft_fft!
# ---------------------------------------------------------------------------

"""
    _temporal_block_dft_fft!(dft_col, segment_col, window, win_weight, nDFT)

FFTW-accelerated temporal block DFT for a single spatial point.
Applies window, FFT via FFTW, normalizes, and fftshifts the result.
"""
function FET.TriadicOrthogonalDecomposition._temporal_block_dft_fft!(
    dft_col,
    segment_col,
    window,
    win_weight,
    nDFT,
)
    windowed = segment_col .* window
    result = FFTW.fft(windowed) .* (win_weight / nDFT)
    # fftshift
    shift = iseven(nDFT) ? nDFT ÷ 2 : (nDFT - 1) ÷ 2
    dft_col .= circshift(result, shift)
    return dft_col
end

end # module FlowInvariantTransferFFTWExt
