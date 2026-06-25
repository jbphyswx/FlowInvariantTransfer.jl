module ScaleToScaleTransfer

using LinearAlgebra: LinearAlgebra
using ..Types: ModeToModeTransferMethod, ModeToModeTriadResult, AbstractShellBinning, AbstractInvariant, KineticEnergy, Helicity, Enstrophy, AbstractExecutionBackend, SerialBackend, ThreadedBackend, DistributedBackend
using ..ShellBinning: shell_edges, shell_centers, n_shells, assign_shells
using ..Utils: wavenumber_magnitude_grid, dealiasing_mask
using ..Workspaces: ScaleToScaleWorkspace

export calculate_mode_to_mode_transfer, calculate_mode_to_mode_transfer!

"""
    calculate_mode_to_mode_transfer(velocity_hat, ks; binning=nothing, invariant=KineticEnergy(), dealiasing=true, backend=SerialBackend()) -> ModeToModeTriadResult

Compute the exact mode-to-mode triad transfer `S(k|p|q)` for the specified invariant.
"""
function calculate_mode_to_mode_transfer(
    velocity_hat,
    ks::Tuple;
    binning::Union{Nothing, AbstractShellBinning} = nothing,
    invariant::AbstractInvariant = KineticEnergy(),
    dealiasing::Bool = true,
    backend::AbstractExecutionBackend = SerialBackend(),
)
    ws = ScaleToScaleWorkspace(velocity_hat, ks, binning)
    calculate_mode_to_mode_transfer!(ws, velocity_hat, ks;
        binning=binning, invariant=invariant, dealiasing=dealiasing, backend=backend)
    
    # Prepare reductions
    reductions = if binning !== nothing
        k_mag = wavenumber_magnitude_grid(ks)
        edges = shell_edges(binning, maximum(k_mag))
        centers = shell_centers(binning, maximum(k_mag))
        (; K=centers, Q=centers, TKQ=copy(ws.T_mat))
    else
        NamedTuple()
    end

    return ModeToModeTriadResult(invariant, ks, copy(ws.net_transfer), reductions)
end

"""
    calculate_mode_to_mode_transfer!(ws, velocity_hat, ks; binning=nothing, invariant=KineticEnergy(), dealiasing=true, backend=SerialBackend()) -> ScaleToScaleWorkspace

In-place version of `calculate_mode_to_mode_transfer`.
"""
function calculate_mode_to_mode_transfer!(
    ws::ScaleToScaleWorkspace,
    velocity_hat,
    ks::Tuple;
    binning::Union{Nothing, AbstractShellBinning} = nothing,
    invariant::AbstractInvariant = KineticEnergy(),
    dealiasing::Bool = true,
    backend::AbstractExecutionBackend = SerialBackend(),
)
    nd = length(ks)
    D  = size(velocity_hat, nd + 1)
    _validate_invariant_dims(invariant, nd, D)
    _calculate_mode_to_mode!(ws, velocity_hat, ks, backend;
        binning=binning, invariant=invariant, dealiasing=dealiasing)
    return ws
end

# Guard invariant/dimension compatibility (mirrors Invariants.transfer_density!): the
# mode-to-mode kernels index components directly, so a 2D field + Helicity() would otherwise
# read out of bounds or silently misbehave (this path previously had no check).
_validate_invariant_dims(::KineticEnergy, nd, D) = nothing
function _validate_invariant_dims(::Helicity, nd, D)
    nd == 3 || throw(ArgumentError("Helicity transfer is defined in 3D only (got nd=$nd)."))
    D  == 3 || throw(ArgumentError("Helicity transfer requires 3 velocity components (got D=$D)."))
    return nothing
end
function _validate_invariant_dims(::Enstrophy, nd, D)
    nd == 2 || throw(ArgumentError("Enstrophy transfer is defined in 2D only (got nd=$nd)."))
    return nothing
end

# Direct serial implementation
function _calculate_mode_to_mode!(
    ws::ScaleToScaleWorkspace,
    velocity_hat,
    ks::Tuple,
    ::SerialBackend;
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

    mask = dealiasing ? dealiasing_mask(ns) : trues(ns...)

    @inbounds for k_idx in CartesianIndices(ns)
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

            S_val = compute_triad_S(invariant, velocity_hat, k_idx, p_idx, q_idx, ks, D)
            net_k += S_val

            if binning !== nothing
                K_sh = ws.shell_idx[k_idx]
                Q_sh = ws.shell_idx[p_idx]
                if K_sh > 0 && Q_sh > 0
                    ws.T_mat[K_sh, Q_sh] += S_val
                end
            end
        end
        ws.net_transfer[k_idx] = net_k
    end
end

# ThreadedBackend stub
function _calculate_mode_to_mode!(
    ws::ScaleToScaleWorkspace,
    velocity_hat,
    ks::Tuple,
    ::ThreadedBackend;
    kwargs...
)
    _mode_to_mode_threaded!(ws, velocity_hat, ks; kwargs...)
end

function _mode_to_mode_threaded!(args...; kwargs...)
    throw(ArgumentError(
        "Threaded mode-to-mode transfer requires OhMyThreads. Run `using OhMyThreads` to load the extension."))
end

# Incompressible triad transfer helpers
@inline function compute_triad_S(::KineticEnergy, velocity_hat, k_idx, p_idx, q_idx, ks, D)
    k_dot_uq = zero(eltype(velocity_hat))
    for c in 1:length(ks)
        k_dot_uq += ks[c][k_idx[c]] * velocity_hat[q_idx, c]
    end
    uk_dot_up = zero(eltype(velocity_hat))
    for c in 1:D
        uk_dot_up += conj(velocity_hat[k_idx, c]) * velocity_hat[p_idx, c]
    end
    return -imag(k_dot_uq * uk_dot_up)
end

@inline function compute_triad_S(::Helicity, velocity_hat, k_idx, p_idx, q_idx, ks, D)
    k_dot_uq = zero(eltype(velocity_hat))
    for c in 1:3
        k_dot_uq += ks[c][k_idx[c]] * velocity_hat[q_idx, c]
    end
    
    kx, ky, kz = ks[1][k_idx[1]], ks[2][k_idx[2]], ks[3][k_idx[3]]
    ux, uy, uz = velocity_hat[k_idx, 1], velocity_hat[k_idx, 2], velocity_hat[k_idx, 3]
    ωx = im * (ky * uz - kz * uy)
    ωy = im * (kz * ux - kx * uz)
    ωz = im * (kx * uy - ky * ux)

    ωk_dot_up = conj(ωx) * velocity_hat[p_idx, 1] + conj(ωy) * velocity_hat[p_idx, 2] + conj(ωz) * velocity_hat[p_idx, 3]

    return -imag(k_dot_uq * ωk_dot_up)
end

@inline function compute_triad_S(::Enstrophy, velocity_hat, k_idx, p_idx, q_idx, ks, D)
    k_dot_uq = ks[1][k_idx[1]] * velocity_hat[q_idx, 1] + ks[2][k_idx[2]] * velocity_hat[q_idx, 2]
    
    kx, ky = ks[1][k_idx[1]], ks[2][k_idx[2]]
    ω_k = im * (kx * velocity_hat[k_idx, 2] - ky * velocity_hat[k_idx, 1])

    px, py = ks[1][p_idx[1]], ks[2][p_idx[2]]
    ω_p = im * (px * velocity_hat[p_idx, 2] - py * velocity_hat[p_idx, 1])

    ωk_dot_ωp = conj(ω_k) * ω_p

    return -imag(k_dot_uq * ωk_dot_ωp)
end

end # module ScaleToScaleTransfer