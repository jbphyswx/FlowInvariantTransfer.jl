module Workspaces

using ..Types: AbstractShellBinning, LinearBinning, AbstractExecutionBackend, SerialBackend
using ..ShellBinning: shell_edges, assign_shells
using ..Utils: wavenumber_magnitude_grid

export NonlinearTermWorkspace, SpectralFluxWorkspace, ShellToShellWorkspace

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

Preallocated buffers for the generalized pseudospectral nonlinear term
`𝒩(k) = FFT[(u·∇)f]`, where the advecting velocity `u` has `nd` advecting (spatial)
components and the advected field `f` has `M` components. For the momentum term `f = u`
(`M = D`); for passive-scalar / vector-potential advection `f = θ`/`a` (`M = 1`).

# Fields
- `u_phys::RA`:    `(ns..., nd)` real physical-space advecting velocity (rank `nd+1`); only the
  `nd` spatial directions of the velocity participate in `(u·∇)`, so this never depends on `D`.
- `grad_phys::GA`: `(ns..., M, nd)` real physical-space gradients ∂f_i/∂x_j (rank `nd+2`).
- `N_phys::RA`:    `(ns..., M)` real physical-space nonlinear term (rank `nd+1`).
- `N̂::CA`:         `(ns..., M)` complex spectral output buffer (rank `nd+1`).
- `plans::P`:      FFT plan/scratch bundle (set by the FFTW extension) or `nothing`.

Parametric on the concrete array types `CA` (complex), `RA` (real, rank `nd+1`), `GA`
(real gradient buffer, rank `nd+2`), and the plan-bundle type `P` — no element-type bounds,
and each field is concretely typed (`grad_phys` has a separate parameter because its rank
differs from the others; `u_phys` and `N_phys` share `RA` — same rank/eltype, possibly
different trailing extent).
"""
struct NonlinearTermWorkspace{CA<:AbstractArray, RA<:AbstractArray, GA<:AbstractArray, P}
    u_phys::RA
    grad_phys::GA
    N_phys::RA
    N̂::CA
    plans::P
end

"""
    NonlinearTermWorkspace(advected_hat, ks)

Construct a `NonlinearTermWorkspace` sized for advecting an `M`-component field `advected_hat`
(shape `(ns..., M)`) by a velocity, on wavenumber tuple `ks` (length `nd`). The advecting
velocity needs only its `nd` spatial components, so `u_phys` is `(ns..., nd)` regardless of how
many components the velocity carries. For the momentum self-advection term pass the velocity
itself (`M = D`). When FFTW is loaded, `plans` is populated with pre-planned transforms.
"""
function NonlinearTermWorkspace(advected_hat, ks)
    FT  = real(eltype(advected_hat))
    nd  = length(ks)
    ns  = size(advected_hat)[1:nd]
    M   = size(advected_hat, nd + 1)              # advected-field component count
    # `similar` propagates the array kind (CPU Array, CuArray, …) — GPU-generic.
    u_phys    = similar(advected_hat, FT, ns..., nd)    # advecting velocity, spatial dirs only
    grad_phys = similar(advected_hat, FT, ns..., M, nd)
    N_phys    = similar(advected_hat, FT, ns..., M)
    N̂         = similar(advected_hat, ns..., M)         # keeps complex eltype
    plans     = _make_fft_plans(advected_hat, ks)
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

end # module Workspaces
