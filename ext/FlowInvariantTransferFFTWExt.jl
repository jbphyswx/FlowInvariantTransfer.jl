module FlowInvariantTransferFFTWExt

using FFTW: FFTW
using FlowInvariantTransfer: FlowInvariantTransfer as FET
using FlowInvariantTransfer.Types: AbstractShellBinning, LinearBinning, ShellToShellResult, AbstractInvariant, KineticEnergy
using FlowInvariantTransfer.Invariants: transfer_density!
using FlowInvariantTransfer.ShellBinning: shell_edges, shell_centers, n_shells, assign_shells
using FlowInvariantTransfer.Utils: wavenumber_magnitude_grid
using FlowInvariantTransfer.Workspaces: ShellToShellWorkspace
using LinearAlgebra: mul!

# ---------------------------------------------------------------------------
# FFT plan/scratch bundle + allocation-free nonlinear term
# ---------------------------------------------------------------------------

# FFTW plan creation is NOT thread-safe and the threaded backend builds a workspace per task,
# so serialize planning behind a lock.
const _PLAN_LOCK = ReentrantLock()

"""
    FFTPlanBundle

Pre-planned transforms + scratch buffers stored in `NonlinearTermWorkspace.plans` so the
FFT-accelerated nonlinear term allocates nothing in the hot path.
"""
struct FFTPlanBundle{PF, PB, CA, KC, MA}
    p_fft::PF      # unnormalized forward plan on a single (ns...) component
    p_bfft::PB     # unnormalized backward plan on a single (ns...) component
    ctmp::CA       # complex (ns...) scratch
    ctmp2::CA      # complex (ns...) scratch
    k_comp::KC     # nd real (ns...) wavenumber-component arrays
    keepmask::MA   # Bool (ns...): true where the mode is KEPT (not 2/3-dealiased)
end

# More specific than the core fallback `_make_fft_plans(::Any, ::Any) = nothing`, so this ADDS
# a method (no overwriting) — dispatched only for complex spectral fields when FFTW is loaded.
function FET.Workspaces._make_fft_plans(velocity_hat::AbstractArray{<:Complex}, ks)
    nd = length(ks)
    ns = size(velocity_hat)[1:nd]
    ct  = similar(velocity_hat, ns...)   # complex (ns...)
    ct2 = similar(velocity_hat, ns...)
    # ESTIMATE (default) does not overwrite the array during planning; serialize for thread-safety.
    p_fft, p_bfft = lock(_PLAN_LOCK) do
        (FFTW.plan_fft(ct), FFTW.plan_bfft(ct))
    end
    k_comp   = [_build_k_component_fft(ks, d, ns) for d in 1:nd]
    keepmask = [!FET.NonlinearTerm._is_dealiased(I, ns, nd) for I in CartesianIndices(ns)]
    return FFTPlanBundle(p_fft, p_bfft, ct, ct2, k_comp, keepmask)
end

"""
    _nonlinear_term_fft!(ws, velocity_hat, ks; dealiasing=true, advecting_hat=velocity_hat)

Allocation-free pseudospectral nonlinear term N̂ = FFT[(u_adv·∇)u] written into `ws.N̂`, using
the pre-planned transforms / scratch in `ws.plans`. The 2/3 input truncation is folded into the
spectral copies (no temporary dealiased array) and the output is re-zeroed above the cutoff.
Normalisation: `ifft = bfft/Np`, and the forward result is divided by `Np` (package coefficient
convention). `N_i = (u_adv)_j ∂_j u_i`: `u_phys` is the advecting velocity, `grad_phys` the
advected gradient.
"""
function FET.NonlinearTerm._nonlinear_term_fft!(
    ws,
    velocity_hat,
    ks;
    dealiasing::Bool = true,
    advecting_hat = velocity_hat,
)
    pb   = ws.plans
    nd   = length(ks)
    ns   = size(velocity_hat)[1:nd]
    M    = size(velocity_hat, nd+1)   # advected-field components (D for momentum, 1 for scalar)
    FT   = real(eltype(velocity_hat))
    Np   = FT(prod(ns))
    ct   = pb.ctmp
    ct2  = pb.ctmp2
    keep = pb.keepmask

    # Advecting velocity: u_phys[...,j] = real(ifft(advecting_hat[...,j])) = real(bfft)/Np,
    # j = 1:nd (only the spatial directions of the velocity participate in (u·∇)).
    for j in 1:nd
        a_j = selectdim(advecting_hat, nd+1, j)
        dealiasing ? (ct .= keep .* a_j) : (ct .= a_j)
        mul!(ct2, pb.p_bfft, ct)
        uj = selectdim(ws.u_phys, nd+1, j)
        uj .= real.(ct2) ./ Np
    end

    # Advected gradient: ∂_j f_i = real(ifft(i k_j f̂_i)), i = 1:M
    for c in 1:M
        v_c = selectdim(velocity_hat, nd+1, c)
        for j in 1:nd
            dealiasing ? (ct .= im .* pb.k_comp[j] .* keep .* v_c) :
                         (ct .= im .* pb.k_comp[j] .* v_c)
            mul!(ct2, pb.p_bfft, ct)
            gcj = selectdim(selectdim(ws.grad_phys, nd+2, j), nd+1, c)
            gcj .= real.(ct2) ./ Np
        end
    end

    # 𝒩_i = Σ_j (u_adv)_j ∂_j f_i
    for c in 1:M
        Nc = selectdim(ws.N_phys, nd+1, c)
        fill!(Nc, zero(FT))
        for j in 1:nd
            uj  = selectdim(ws.u_phys, nd+1, j)
            gcj = selectdim(selectdim(ws.grad_phys, nd+2, j), nd+1, c)
            Nc .+= uj .* gcj
        end
    end

    # 𝒩̂_i = fft(𝒩_i)/Np, zeroed above the 2/3 cutoff
    for c in 1:M
        Nc = selectdim(ws.N_phys, nd+1, c)
        ct .= Nc
        mul!(ct2, pb.p_fft, ct)
        Nhat_c = selectdim(ws.N̂, nd+1, c)
        dealiasing ? (Nhat_c .= (ct2 .* keep) ./ Np) : (Nhat_c .= ct2 ./ Np)
    end

    return ws.N̂
end

# Allocate a copy of `velocity_hat` with the 2/3-rule discard modes zeroed (input truncation).
function _dealias_copy(velocity_hat, ns, nd::Int)
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
    ks;
    dealiasing::Bool = true,
    verify_antisymmetry::Bool = true,
    invariant::AbstractInvariant = KineticEnergy(),
    advecting_hat = velocity_hat,
)
    nd    = length(ks)
    ns    = size(velocity_hat)[1:nd]
    M     = size(velocity_hat, nd+1)   # components of the binned/carried primary field
    FT    = real(eltype(velocity_hat))
    N_sh  = size(result.transfer_matrix, 1)
    Np    = FT(prod(ns))

    # Orszag 2/3: truncate inputs before forming products (see _dealias_copy / THEORY.md §0.5).
    vhat = dealiasing ? _dealias_copy(velocity_hat, ns, nd) : velocity_hat
    ahat = dealiasing ? _dealias_copy(advecting_hat, ns, nd) : advecting_hat

    k_comp = [_build_k_component_fft(ks, d, ns) for d in 1:nd]
    # Advecting field is the FULL velocity (computed once, spatial dirs only); the band-m primary
    # field is what gets advected. For energy primary==velocity; for a scalar primary==θ̂.
    u_full_phys = [real.(FFTW.ifft(selectdim(ahat, nd+1, j))) for j in 1:nd]

    # T(n,m) = A[n,m] = Σ_{I∈S_n} Re{c*·N̂_m}, N̂_m = (u·∇)f_m (full velocity advects band-m;
    # AMP 2005). Accumulated one mediator `m` at a time directly into result.transfer_matrix,
    # reusing ws.nonlinear.N̂ / ws.transfer_density — peak memory O(N^D), not O(N_sh·N^D) (the old
    # N_sh-fold allocation was a ~100 GB trap at 256³). For energy A is antisymmetric
    # (A[n,m]+A[m,n]=0) and reduces as Σ_m A[n,m] = transfer_spectrum[n], so NO ½(A−Aᵀ).
    fill!(result.transfer_matrix, zero(FT))
    for m in 1:N_sh
        fill!(ws.û_m, zero(eltype(ws.û_m)))
        for I in CartesianIndices(ns)
            ws.shell_idx[I] == m || continue
            for c in 1:M; ws.û_m[I, c] = vhat[I, c]; end
        end
        grad_m   = [[real.(FFTW.ifft(im .* k_comp[j] .* selectdim(ws.û_m, nd+1, c)))
                     for j in 1:nd] for c in 1:M]
        N_phys_m = [sum(u_full_phys[j] .* grad_m[c][j] for j in 1:nd) for c in 1:M]
        for c in 1:M
            selectdim(ws.nonlinear.N̂, nd+1, c) .= FFTW.fft(N_phys_m[c]) ./ Np
        end
        if dealiasing
            for I in CartesianIndices(ns)
                FET.NonlinearTerm._is_dealiased(I, ns, nd) || continue
                for c in 1:M; ws.nonlinear.N̂[I, c] = zero(eltype(ws.nonlinear.N̂)); end
            end
        end
        transfer_density!(ws.transfer_density, invariant, velocity_hat, ws.nonlinear.N̂, ks)
        @inbounds for I in CartesianIndices(ns)
            n = ws.shell_idx[I]
            n == 0 && continue
            result.transfer_matrix[n, m] += ws.transfer_density[I]
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

function _build_k_component_fft(ks, d::Int, ns)
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

FFTW-accelerated temporal block DFT for a single spatial point. Applies the window, transforms
via FFTW, and normalizes — returning the DFT in **natural (unshifted) bin order** `0,1,…,nDFT-1`,
exactly like the direct-sum path. The caller (`triadic_orthogonal_decomposition`) applies the
single `fftshift` to centre the spectrum; this routine must NOT shift as well, or the result is
shifted twice (a full wrap for even `nDFT`) and `Q_hat` ends up misaligned with the frequency axis.
"""
function FET.TriadicOrthogonalDecomposition._temporal_block_dft_fft!(
    dft_col,
    segment_col,
    window,
    win_weight,
    nDFT,
)
    windowed = segment_col .* window
    dft_col .= FFTW.fft(windowed) .* (win_weight / nDFT)
    return dft_col
end

end # module FlowInvariantTransferFFTWExt
