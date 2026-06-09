module FlowEnergyTransferOhMyThreadsExt

using OhMyThreads: OhMyThreads
using FlowEnergyTransfer: FlowEnergyTransfer as FET
using FlowEnergyTransfer.Types: AbstractShellBinning, LinearBinning, ShellToShellResult
using FlowEnergyTransfer.ShellBinning: shell_edges, shell_centers, shell_mask
using FlowEnergyTransfer.Utils: wavenumber_magnitude_grid

# ---------------------------------------------------------------------------
# Thread-parallel shell-to-shell transfer (OhMyThreads scheduler)
# ---------------------------------------------------------------------------

"""
    threaded_shell_to_shell_transfer(velocity_hat, ks;
        binning, dealiasing=true, verify_antisymmetry=true)
        -> ShellToShellResult

Thread-parallel version of shell-to-shell transfer using OhMyThreads.
The outer loop over mediator shells (m index) is parallelised.

Requires FFTW to also be loaded for the FFT transforms.
"""
function FET.ShellToShellTransfer._shell_to_shell_threaded(
    velocity_hat::AbstractArray{<:Complex},
    ks::Tuple;
    binning::AbstractShellBinning = LinearBinning(1.0),
    dealiasing::Bool = true,
    verify_antisymmetry::Bool = true,
)
    # Delegate to the FFTW path if available, but parallelise the shell loop.
    # If FFTW ext is not loaded this will fall back to the direct path.
    nd   = length(ks)
    ns   = size(velocity_hat)[1:nd]
    D    = size(velocity_hat, nd+1)
    FT   = eltype(real(velocity_hat[1]))

    k_mag   = wavenumber_magnitude_grid(ks)
    k_max   = maximum(k_mag)
    edges   = shell_edges(binning, k_max)
    N_sh    = length(edges) - 1
    centers = shell_centers(binning, k_max)
    masks   = [shell_mask(k_mag, edges, n) for n in 1:N_sh]

    T_mat_rows = Vector{Vector{FT}}(undef, N_sh)

    # Thread-parallel loop over mediator shells
    OhMyThreads.@tasks for m in 1:N_sh
        T_mat_rows[m] = _compute_row_for_mediator(
            m, velocity_hat, masks, ks, N_sh, D, nd, FT; dealiasing=dealiasing)
    end

    T_mat = hcat(T_mat_rows...)'  # (N_sh, N_sh): row = receiver, col = mediator

    net_transfer = vec(sum(T_mat; dims=2))
    max_asym = verify_antisymmetry ? maximum(abs, T_mat .+ T_mat') : FT(NaN)

    return ShellToShellResult{FT}(
        convert(Vector{FT}, centers),
        convert(Vector{FT}, edges),
        T_mat,
        convert(Vector{FT}, net_transfer),
        max_asym,
    )
end

# Compute T[:,m] — one column of the transfer matrix for mediator shell m.
# Each task is independent so this is trivially parallel.
function _compute_row_for_mediator(
    m::Int,
    velocity_hat::AbstractArray{<:Complex},
    masks::Vector,
    ks::Tuple,
    N_sh::Int,
    D::Int,
    nd::Int,
    FT::Type;
    dealiasing::Bool,
)
    ns = size(velocity_hat)[1:nd]

    # Call the direct nonlinear term for mediator m — thread-safe (no shared state)
    û_m = zeros(Complex{FT}, size(velocity_hat)...)
    for I in CartesianIndices(ns)
        masks[m][I] || continue
        for c in 1:D
            û_m[I, c] = velocity_hat[I, c]
        end
    end

    N̂_m = FET.NonlinearTerm.compute_nonlinear_term(û_m, ks;
            dealiasing=dealiasing, backend=FET.SerialBackend())

    col = zeros(FT, N_sh)
    for n in 1:N_sh
        s = zero(FT)
        for I in CartesianIndices(ns)
            masks[n][I] || continue
            for c in 1:D
                s += real(conj(velocity_hat[I, c]) * N̂_m[I, c])
            end
        end
        col[n] = s
    end
    return col
end

# Register stub override so ThreadedBackend dispatches here
function FET.ShellToShellTransfer._shell_to_shell_threaded(args...; kwargs...)
    throw(ArgumentError("ThreadedBackend shell-to-shell requires OhMyThreads. Run `using OhMyThreads`."))
end

end # module FlowEnergyTransferOhMyThreadsExt
