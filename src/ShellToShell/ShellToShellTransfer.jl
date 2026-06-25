module ShellToShellTransfer

using ..Types: ShellToShellTransferMethod, ShellToShellResult, AbstractShellBinning, LinearBinning, AbstractExecutionBackend, SerialBackend, FFTBackend, ThreadedBackend, AbstractInvariant, KineticEnergy
using ..Invariants: transfer_density!
using ..ShellBinning: shell_edges, shell_centers, n_shells, assign_shells
using ..Utils: wavenumber_magnitude_grid
using ..NonlinearTerm: compute_nonlinear_term!
using ..Workspaces: NonlinearTermWorkspace, ShellToShellWorkspace

export calculate_shell_to_shell_transfer, calculate_shell_to_shell_transfer!

# ---------------------------------------------------------------------------
# Internal FFTW-path stub (overridden by FlowInvariantTransferFFTWExt)
# ---------------------------------------------------------------------------

"""
    _shell_to_shell_fft!(result, ws, velocity_hat, ks; kwargs...)

FFT-accelerated shell-to-shell transfer.  Stub overridden by the FFTW extension.
"""
function _shell_to_shell_fft!(args...; kwargs...)
    throw(ArgumentError(
        "FFT-accelerated shell-to-shell transfer requires FFTW. Run `using FFTW` to load the extension."))
end

function _shell_to_shell_threaded!(args...; kwargs...)
    throw(ArgumentError(
        "Threaded shell-to-shell transfer requires OhMyThreads. Run `using OhMyThreads` to load the extension."))
end

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

"""
    calculate_shell_to_shell_transfer(velocity_hat, ks;
        binning, dealiasing=true, verify_antisymmetry=true, backend=nothing)
        -> ShellToShellResult

Compute the directed shell-to-shell kinetic energy transfer matrix T(n,m).

# Arguments
- `velocity_hat`: Complex array of size `(ns..., D)` — Fourier coefficients of
  the D velocity components on a periodic uniform grid.
- `ks`: Tuple of D 1D physical-wavenumber vectors.

# Keyword Arguments
- `binning::AbstractShellBinning`: Shell binning; default `LinearBinning(1.0)`.
- `dealiasing::Bool=true`: Apply 2/3 rule dealiasing.
- `verify_antisymmetry::Bool=true`: Compute `max|T(n,m)+T(m,n)|` and store in result.
- `backend::AbstractExecutionBackend`: `SerialBackend()` (default), `FFTBackend()` (requires FFTW), or `ThreadedBackend()` (requires OhMyThreads).

# Returns
`ShellToShellResult` with:
- `transfer_matrix[n,m]`: Energy transferred from shell m to shell n.
- `net_transfer[n]` = Σ_m T(n,m): net energy gain of shell n.
- `max_antisymmetry_error`: validation diagnostic.

# Algorithm (Verma 2002 formulation)
For each pair of receiver shell n and mediator shell m:
  T(n,m) = Σ_{k∈S_n} Re{ û_n*(k) · N̂_m(k) }
where N̂_m(k) = FFT[(u_m · ∇)u], u_m = IFFT(û · χ_m).

This formulation uses the mediator velocity restricted to shell m, so:
  T(n,m) + T(m,n) = 0  exactly (antisymmetry).

# Cost
O(N_shells² · N^D log N^D) with FFTW; O(N_shells² · N^{2D}) direct-sum.

# References
- Verma et al. (2002), arXiv:nlin/0204027
- Alexakis, Mininni & Pouquet (2005)
"""
function calculate_shell_to_shell_transfer(
    velocity_hat,
    ks::Tuple;
    binning::AbstractShellBinning = _default_binning(ks),
    dealiasing::Bool = true,
    verify_antisymmetry::Bool = true,
    invariant::AbstractInvariant = KineticEnergy(),
    backend::AbstractExecutionBackend = SerialBackend(),
)
    ws      = ShellToShellWorkspace(velocity_hat, ks, binning)
    k_mag   = wavenumber_magnitude_grid(ks)
    edges   = shell_edges(binning, maximum(k_mag))
    centers = shell_centers(binning, maximum(k_mag))
    N_sh    = length(centers)
    FT      = real(eltype(velocity_hat))
    T_mat   = Matrix{FT}(undef, N_sh, N_sh)
    net     = Vector{FT}(undef, N_sh)
    # Use a mutable wrapper so ! variants can write max_asym back
    result_mut = ShellToShellResult(centers, edges, T_mat, net, FT(NaN))
    max_asym = _calculate_shell_to_shell!(result_mut, ws, velocity_hat, ks, backend;
        dealiasing=dealiasing, verify_antisymmetry=verify_antisymmetry, invariant=invariant)
    return ShellToShellResult(centers, edges, T_mat, net, max_asym)
end

"""
    calculate_shell_to_shell_transfer!(result, ws, velocity_hat, ks; kwargs...)

In-place version. Writes into `result` using preallocated buffers from `ws`.
Zero heap allocations in the hot path.
"""
function calculate_shell_to_shell_transfer!(
    result::ShellToShellResult,
    ws::ShellToShellWorkspace,
    velocity_hat,
    ks::Tuple;
    dealiasing::Bool = true,
    verify_antisymmetry::Bool = true,
    invariant::AbstractInvariant = KineticEnergy(),
    backend::AbstractExecutionBackend = SerialBackend(),
)
    _calculate_shell_to_shell!(result, ws, velocity_hat, ks, backend;
        dealiasing=dealiasing, verify_antisymmetry=verify_antisymmetry, invariant=invariant)
    return result
end

_calculate_shell_to_shell!(result, ws, velocity_hat, ks, ::SerialBackend; kwargs...) =
    _calculate_shell_to_shell_direct!(result, ws, velocity_hat, ks; kwargs...)

_calculate_shell_to_shell!(result, ws, velocity_hat, ks, ::FFTBackend; kwargs...) =
    _shell_to_shell_fft!(result, ws, velocity_hat, ks; kwargs...)

_calculate_shell_to_shell!(result, ws, velocity_hat, ks, ::ThreadedBackend; kwargs...) =
    _shell_to_shell_threaded!(result, ws, velocity_hat, ks; kwargs...)

# ---------------------------------------------------------------------------
# Direct reference implementation
# ---------------------------------------------------------------------------

"""
    _calculate_shell_to_shell_direct!(result, ws, velocity_hat, ks; dealiasing, verify_antisymmetry, invariant)

Direct-sum (SerialBackend) shell-to-shell transfer. Writes into `result` using
workspace buffers from `ws` — no heap allocation in the hot path.

For each mediator shell m:
  1. Build û_m = û restricted to shell m (using ws.shell_idx)
  2. Compute N̂_m = FFT[(u_m·∇)u] using ws.nonlinear buffers
  3. Accumulate T(n,m) for all receiver shells n
"""
function _calculate_shell_to_shell_direct!(
    result::ShellToShellResult,
    ws::ShellToShellWorkspace,
    velocity_hat,
    ks::Tuple;
    dealiasing::Bool,
    verify_antisymmetry::Bool,
    invariant::AbstractInvariant = KineticEnergy(),
)
    nd    = length(ks)
    ns    = size(velocity_hat)[1:nd]
    D     = size(velocity_hat, nd+1)
    FT    = real(eltype(velocity_hat))
    N_sh  = size(result.transfer_matrix, 1)

    fill!(result.transfer_matrix, zero(FT))

    for m in 1:N_sh
        # Build û_m: velocity restricted to shell m — reuse ws.û_m
        fill!(ws.û_m, zero(eltype(ws.û_m)))
        for I in CartesianIndices(ns)
            ws.shell_idx[I] == m || continue
            for comp in 1:D
                ws.û_m[I, comp] = velocity_hat[I, comp]
            end
        end

        # N̂_m = (u·∇)u_m: the FULL velocity advects the band-m field (Alexakis–Mininni–Pouquet
        # 2005). This makes A[n,m] = Σ_{k∈S_n} Re{û*·N̂_m} both antisymmetric (A[n,m]+A[m,n]=0)
        # and correctly reducing (Σ_m A[n,m] = transfer_spectrum[n]) — no ½(A−Aᵀ) needed.
        compute_nonlinear_term!(ws.nonlinear, ws.û_m, ks;
                                dealiasing=dealiasing, backend=SerialBackend(),
                                advecting_hat=velocity_hat)
        N̂_m = ws.nonlinear.N̂

        # Write per-mode transfer density into ws.transfer_density
        transfer_density!(ws.transfer_density, invariant, velocity_hat, N̂_m, ks)

        # Accumulate A(n,m) = Σ_{k∈S_n} Re{û*·N̂_m} for all receiver shells n
        for n in 1:N_sh
            s = zero(FT)
            for I in CartesianIndices(ns)
                ws.shell_idx[I] == n || continue
                s += ws.transfer_density[I]
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

    # Antisymmetry check: max |T(n,m) + T(m,n)| — in-place, no temp matrix
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

function _default_binning(ks)
    min_dk = Inf
    for k_vec in ks
        for k in k_vec
            ak = abs(k)
            ak > 0 && (min_dk = min(min_dk, ak))
        end
    end
    min_dk = isfinite(min_dk) ? min_dk : 1.0
    return LinearBinning(min_dk)
end

end # module ShellToShellTransfer
