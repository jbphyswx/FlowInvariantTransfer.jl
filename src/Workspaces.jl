module Workspaces

using ..Types: AbstractShellBinning, LinearBinning, AbstractExecutionBackend, SerialBackend
using ..ShellBinning: shell_edges, assign_shells
using ..Utils: wavenumber_magnitude_grid

export NonlinearTermWorkspace, SpectralFluxWorkspace, ShellToShellWorkspace, ScaleToScaleWorkspace

# ---------------------------------------------------------------------------
# NonlinearTermWorkspace
# ---------------------------------------------------------------------------

"""
    NonlinearTermWorkspace{CA, RA}

Preallocated buffers for computing the nonlinear advection term N̂(k) = FFT[(u·∇)u].

# Fields
- `u_phys::RA`:    `(ns..., D)` real physical-space velocities.
- `grad_phys::RA`: `(ns..., D, nd)` real physical-space velocity gradients ∂u_i/∂x_j.
- `N_phys::RA`:    `(ns..., D)` real physical-space nonlinear term.
- `N̂::CA`:         `(ns..., D)` complex spectral output buffer.

Parametric on array types `CA` (complex) and `RA` (real) — no element-type bounds.
"""
struct NonlinearTermWorkspace{CA<:AbstractArray, RA<:AbstractArray}
    u_phys::RA
    grad_phys::RA
    N_phys::RA
    N̂::CA
end

"""
    NonlinearTermWorkspace(velocity_hat, ks)

Construct a `NonlinearTermWorkspace` sized for `velocity_hat` and wavenumber tuple `ks`.
"""
function NonlinearTermWorkspace(velocity_hat, ks)
    FT  = real(eltype(velocity_hat))
    CT  = eltype(velocity_hat)   # complex type
    ns  = size(velocity_hat)[1:length(ks)]
    D   = size(velocity_hat, ndims(velocity_hat))
    nd  = length(ks)
    RA  = Array{FT}
    CA  = Array{CT}
    return NonlinearTermWorkspace{CA, RA}(
        RA(undef, ns..., D),       # u_phys
        RA(undef, ns..., D, nd),   # grad_phys
        RA(undef, ns..., D),       # N_phys
        CA(undef, ns..., D),       # N̂
    )
end

# ---------------------------------------------------------------------------
# SpectralFluxWorkspace
# ---------------------------------------------------------------------------

"""
    SpectralFluxWorkspace{NW, V, A}

Preallocated buffers for `calculate_spectral_flux!`.

# Fields
- `nonlinear::NW`:        `NonlinearTermWorkspace` for computing N̂(k).
- `T_spec::V`:            Shell transfer spectrum buffer (length N_sh).
- `flux::V`:              Cumulative flux buffer (length N_sh).
- `transfer_density::A`:  Per-mode transfer density buffer.
"""
struct SpectralFluxWorkspace{NW<:NonlinearTermWorkspace, V<:AbstractVector, A<:AbstractArray}
    nonlinear::NW
    T_spec::V
    flux::V
    transfer_density::A
end

"""
    SpectralFluxWorkspace(velocity_hat, ks, binning)

Construct a `SpectralFluxWorkspace` for the given input and binning.
"""
function SpectralFluxWorkspace(velocity_hat, ks, binning::AbstractShellBinning)
    k_mag  = wavenumber_magnitude_grid(ks)
    edges  = shell_edges(binning, maximum(k_mag))
    N_sh   = length(edges) - 1
    FT     = real(eltype(velocity_hat))
    ns     = size(velocity_hat)[1:length(ks)]
    return SpectralFluxWorkspace(
        NonlinearTermWorkspace(velocity_hat, ks),
        Vector{FT}(undef, N_sh),
        Vector{FT}(undef, N_sh),
        Array{FT}(undef, ns...),
    )
end

# ---------------------------------------------------------------------------
# ShellToShellWorkspace
# ---------------------------------------------------------------------------

"""
    ShellToShellWorkspace{NW, CA, M, V, IA}

Preallocated buffers for `calculate_shell_to_shell_transfer!`.

# Fields
- `nonlinear::NW`:   `NonlinearTermWorkspace` (owns N̂_m and all physical-space temps).
- `û_m::CA`:         Band-filtered mediator velocity buffer (reused each shell `m`).
- `T_mat::M`:        Output transfer matrix (N_sh × N_sh), written in-place.
- `net_transfer::V`: Net per-shell transfer buffer (length N_sh).
- `shell_idx::IA`:   Integer shell-index array (same shape as k_mag).
"""
struct ShellToShellWorkspace{NW<:NonlinearTermWorkspace,
                              CA<:AbstractArray,
                              M<:AbstractMatrix,
                              V<:AbstractVector,
                              IA<:AbstractArray{Int},
                              A<:AbstractArray}
    nonlinear::NW
    û_m::CA
    T_mat::M
    net_transfer::V
    shell_idx::IA
    transfer_density::A
end

"""
    ShellToShellWorkspace(velocity_hat, ks, binning)

Construct a `ShellToShellWorkspace` for the given input and binning.
"""
function ShellToShellWorkspace(velocity_hat, ks, binning::AbstractShellBinning)
    FT        = real(eltype(velocity_hat))
    k_mag     = wavenumber_magnitude_grid(ks)
    edges     = shell_edges(binning, maximum(k_mag))
    N_sh      = length(edges) - 1
    shell_idx = assign_shells(k_mag, edges)
    ns        = size(velocity_hat)[1:length(ks)]
    return ShellToShellWorkspace(
        NonlinearTermWorkspace(velocity_hat, ks),
        similar(velocity_hat),                   # û_m
        Matrix{FT}(undef, N_sh, N_sh),           # T_mat
        Vector{FT}(undef, N_sh),                 # net_transfer
        shell_idx,
        Array{FT}(undef, ns...),
    )
end

# ---------------------------------------------------------------------------
# ScaleToScaleWorkspace
# ---------------------------------------------------------------------------

"""
    ScaleToScaleWorkspace{M, V, IA}

Preallocated buffers for `calculate_mode_to_mode_transfer!`.

# Fields
- `T_mat::M`:        Output transfer matrix (N_sh × N_sh), written in-place.
- `net_transfer::V`: Net per-mode transfer buffer.
- `shell_idx::IA`:   Integer shell-index array.
"""
struct ScaleToScaleWorkspace{M<:AbstractMatrix, V<:AbstractArray, IA<:AbstractArray{Int}}
    T_mat::M
    net_transfer::V
    shell_idx::IA
end

"""
    ScaleToScaleWorkspace(velocity_hat, ks, binning)

Construct a `ScaleToScaleWorkspace` for the given input and binning.
"""
function ScaleToScaleWorkspace(velocity_hat, ks, binning)
    FT        = real(eltype(velocity_hat))
    nd        = length(ks)
    ns        = size(velocity_hat)[1:nd]
    k_mag     = wavenumber_magnitude_grid(ks)

    if !isnothing(binning)
        edges     = shell_edges(binning, maximum(k_mag))
        N_sh      = length(edges) - 1
        shell_idx = assign_shells(k_mag, edges)
        T_mat     = Matrix{FT}(undef, N_sh, N_sh)
    else
        shell_idx = zeros(Int, ns...)
        T_mat     = Matrix{FT}(undef, 0, 0)
    end

    net_transfer = Array{FT}(undef, ns...)
    return ScaleToScaleWorkspace(T_mat, net_transfer, shell_idx)
end

end # module Workspaces
