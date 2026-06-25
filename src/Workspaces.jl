module Workspaces

using ..Types: AbstractShellBinning, LinearBinning, AbstractExecutionBackend, SerialBackend
using ..ShellBinning: shell_edges, assign_shells
using ..Utils: wavenumber_magnitude_grid

export NonlinearTermWorkspace, SpectralFluxWorkspace, ShellToShellWorkspace, ScaleToScaleWorkspace

# ---------------------------------------------------------------------------
# NonlinearTermWorkspace
# ---------------------------------------------------------------------------

"""
    _make_fft_plans(velocity_hat, ks)

Hook returning an FFT plan/scratch bundle stored in `NonlinearTermWorkspace.plans`, or
`nothing`. The core returns `nothing` (the direct-DFT path needs no plans); the FFTW
extension overrides this to build pre-planned transforms + scratch buffers so the
FFT-accelerated hot path allocates nothing.
"""
_make_fft_plans(velocity_hat, ks) = nothing

"""
    NonlinearTermWorkspace{CA, RA, GA, P}

Preallocated buffers for computing the nonlinear advection term N̂(k) = FFT[(u·∇)u].

# Fields
- `u_phys::RA`:    `(ns..., D)` real physical-space velocities (rank `nd+1`).
- `grad_phys::GA`: `(ns..., D, nd)` real physical-space velocity gradients ∂u_i/∂x_j (rank `nd+2`).
- `N_phys::RA`:    `(ns..., D)` real physical-space nonlinear term (rank `nd+1`).
- `N̂::CA`:         `(ns..., D)` complex spectral output buffer (rank `nd+1`).
- `plans::P`:      FFT plan/scratch bundle (set by the FFTW extension) or `nothing`.

Parametric on the concrete array types `CA` (complex), `RA` (real, rank `nd+1`), `GA`
(real gradient buffer, rank `nd+2`), and the plan-bundle type `P` — no element-type bounds,
and each field is concretely typed (`grad_phys` has a separate parameter because its rank
differs from the others).
"""
struct NonlinearTermWorkspace{CA<:AbstractArray, RA<:AbstractArray, GA<:AbstractArray, P}
    u_phys::RA
    grad_phys::GA
    N_phys::RA
    N̂::CA
    plans::P
end

"""
    NonlinearTermWorkspace(velocity_hat, ks)

Construct a `NonlinearTermWorkspace` sized for `velocity_hat` and wavenumber tuple `ks`.
When FFTW is loaded, `plans` is populated with pre-planned transforms (see `_make_fft_plans`).
"""
function NonlinearTermWorkspace(velocity_hat, ks)
    FT  = real(eltype(velocity_hat))
    ns  = size(velocity_hat)[1:length(ks)]
    D   = size(velocity_hat, ndims(velocity_hat))
    nd  = length(ks)
    # `similar` propagates the array kind (CPU Array, CuArray, …) — GPU-generic.
    u_phys    = similar(velocity_hat, FT, ns..., D)
    grad_phys = similar(velocity_hat, FT, ns..., D, nd)
    N_phys    = similar(velocity_hat, FT, ns..., D)
    N̂         = similar(velocity_hat, ns..., D)   # keeps complex eltype
    plans     = _make_fft_plans(velocity_hat, ks)
    return NonlinearTermWorkspace(u_phys, grad_phys, N_phys, N̂, plans)
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
        similar(velocity_hat, FT, N_sh),     # T_spec
        similar(velocity_hat, FT, N_sh),     # flux
        similar(velocity_hat, FT, ns...),    # transfer_density
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
        similar(velocity_hat, FT, N_sh, N_sh),   # T_mat
        similar(velocity_hat, FT, N_sh),         # net_transfer
        shell_idx,
        similar(velocity_hat, FT, ns...),        # transfer_density
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
        T_mat     = similar(velocity_hat, FT, N_sh, N_sh)
    else
        shell_idx = fill!(similar(velocity_hat, Int, ns...), 0)
        T_mat     = similar(velocity_hat, FT, 0, 0)
    end

    net_transfer = similar(velocity_hat, FT, ns...)
    return ScaleToScaleWorkspace(T_mat, net_transfer, shell_idx)
end

end # module Workspaces
