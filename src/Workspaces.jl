module Workspaces

using ..Types: Types
using ..ShellBinning: ShellBinning
using FlowTransformBindings: FlowTransformBindings as FTB
export NonlinearTermWorkspace, SpectralFluxWorkspace, ShellToShellWorkspace, PaddedGrid

# ---------------------------------------------------------------------------
# The grid padded for exact quadratic products
# ---------------------------------------------------------------------------

"""
    PaddedGrid(advected_hat, ks)

The grid of `FlowTransformBindings.padded_length` points per axis on which `PaddedThreeHalves` forms the
quadratic product, in `advected_hat`'s array type: its axes, the coefficient maps between it and the
grid of `ks`, its derivative wavenumbers and the buffers the padded nonlinear term writes. Every
transform reads it; the FFTW extension adds its plans.

# Fields
- `ks`: the padded grid's axes, in the layout of `ks`.
- `maps`: `FlowTransformBindings.PaddedMaps` between the two grids.
- `kd`: per-axis derivative wavenumbers on the padded grid, zero on the images of an even axis's
  Nyquist slot.
- `u_phys`, `g_phys`, `n_phys`: real `(Ms..., nd)` advecting velocity, `(Ms...)` gradient and
  `(Ms..., M)` nonlinear term.
- `spec`, `spec1`: one component's `(Msp...)` coefficients, and the same memory shaped `(Msp..., 1)`.
- `invM`: `1/prod(Ms)`, the forward normalization.
"""
struct PaddedGrid{KS, PM, KD, RA, GA, CA, C1, R}
    ks::KS
    maps::PM
    kd::KD
    u_phys::RA
    g_phys::GA
    n_phys::RA
    spec::CA
    spec1::C1
    invM::R
end

function PaddedGrid(advected_hat, ks)
    nd  = length(ks)
    M   = size(advected_hat, nd + 1)
    FT  = real(eltype(advected_hat))
    Ms  = map(n -> FTB.padded_length(n), FTB.full_size(ks))
    ksp = FTB.padded_axes(ks, Ms)
    Msp = FTB.spectral_size(ksp)
    spec = similar(advected_hat, Msp...)                 # keeps the complex eltype
    return PaddedGrid(ksp, FTB.PaddedMaps(advected_hat, FT, ks, Ms),
                      FTB.padded_derivative_wavenumbers(advected_hat, FT, ks, Ms),
                      similar(advected_hat, FT, Ms..., nd), similar(advected_hat, FT, Ms...),
                      similar(advected_hat, FT, Ms..., M), spec, reshape(spec, Msp..., 1),
                      one(FT) / FT(prod(Ms)))
end

# ---------------------------------------------------------------------------
# NonlinearTermWorkspace
# ---------------------------------------------------------------------------

"""
    _make_fft_plans(advected_hat, ks, pad, fft_nthreads)

Hook returning the FFT plan/scratch bundle stored in `NonlinearTermWorkspace.plans`: `nothing` in the
core (the direct sum needs no plans), pre-planned transforms and scratch buffers from the FFTW
extension, which plans the padded transforms on `pad`, the workspace's [`PaddedGrid`](@ref) or
`nothing`. `fft_nthreads` is FFTW's per-transform thread count.
"""
_make_fft_plans(advected_hat, ks, pad, fft_nthreads) = nothing

"""
    NonlinearTermWorkspace{CA, RA, GA, PH, PD, P}

Preallocated buffers for the generalized pseudospectral nonlinear term
`𝒩(k) = FFT[(u·∇)f]`, where the advecting velocity `u` has `nd` advecting (spatial)
components and the advected field `f` has `M` components. For the momentum term `f = u`
(`M = D`); for passive-scalar / vector-potential advection `f = θ`/`a` (`M = 1`).

Physical-space buffers are sized `ns`, the grid the field lives on; the spectral buffer is sized
`ms`, the shape of the coefficient array. The two differ on the half (real-field) layout, where
`ms₁ = ns₁÷2+1` — see `FlowTransformBindings.RFFTAxis`.

# Fields
- `u_phys::RA`:  `(ns..., nd)` real physical-space advecting velocity (rank `nd+1`); only the
  `nd` spatial directions of the velocity participate in `(u·∇)`, so this never depends on `D`.
- `g_phys::GA`:  `(ns...)` real physical-space gradient `∂f_i/∂x_j` for the direction pair being
  accumulated (rank `nd`); each `(i,j)` is transformed, multiplied into `N_phys` and discarded, so
  one buffer serves all `M·nd` pairs.
- `N_phys::RA`:  `(ns..., M)` real physical-space nonlinear term (rank `nd+1`).
- `N̂::CA`:       `(ms..., M)` complex spectral output buffer (rank `nd+1`).
- `phase::PH`:   `nd` complex vectors, `phase[d]` of length `ns[d]`, holding the per-axis factors of a
  single mode's plane wave `e^{ip·x}`. Written by `ModeToModeTransfer._mode_to_mode_giver!`, which
  evaluates that mode's gradient in closed form.
- `pad::PD`:     the [`PaddedGrid`](@ref) under `PaddedThreeHalves`, else `nothing`.
- `plans::P`:    FFT plan/scratch bundle (set by the FFTW extension) or `nothing`.
"""
struct NonlinearTermWorkspace{CA<:AbstractArray, RA<:AbstractArray, GA<:AbstractArray,
                              PH<:AbstractVector, PD, P}
    u_phys::RA
    g_phys::GA
    N_phys::RA
    N̂::CA
    phase::PH
    pad::PD
    plans::P
end

"""
    NonlinearTermWorkspace(advected_hat, ks; dealiasing=OrszagTwoThirds(), fft_nthreads=1)

Construct a `NonlinearTermWorkspace` sized for advecting an `M`-component field `advected_hat`
(shape `(ms..., M)`) by a velocity, on wavenumber tuple `ks` (length `nd`). The advecting
velocity needs only its `nd` spatial components, so `u_phys` is `(ns..., nd)` whatever the
velocity's component count. For the momentum self-advection term pass the velocity itself
(`M = D`). With FFTW loaded, `plans` holds pre-planned transforms. Pass the `dealiasing` the
workspace will be used with: only `PaddedThreeHalves` gets the padded grid.
"""
function NonlinearTermWorkspace(advected_hat, ks; dealiasing::Types.AbstractDealiasing = Types.OrszagTwoThirds(),
                                fft_nthreads::Int = 1)
    FT  = real(eltype(advected_hat))
    nd  = length(ks)
    ns  = FTB.full_size(ks)            # physical grid
    ms  = FTB.spectral_size(ks)        # coefficient grid
    M   = size(advected_hat, nd + 1)              # advected-field component count
    size(advected_hat)[1:nd] == ms || throw(DimensionMismatch(
        "field spatial size $(size(advected_hat)[1:nd]) does not match the wavenumber grid $ms."))
    u_phys = similar(advected_hat, FT, ns..., nd)      # advecting velocity, spatial dirs only
    g_phys = similar(advected_hat, FT, ns...)          # one streamed gradient ∂f_i/∂x_j
    N_phys = similar(advected_hat, FT, ns..., M)
    N̂      = similar(advected_hat, ms..., M)           # keeps complex eltype
    # Per-axis plane-wave factors for `_mode_to_mode_giver!`. A `Vector` of vectors, so `phase[d]` with
    # a runtime `d` stays inferred; every element has the same concrete type.
    phase  = [similar(advected_hat, ns[d]) for d in 1:nd]
    pad    = dealiasing isa Types.PaddedThreeHalves ? PaddedGrid(advected_hat, ks) : nothing
    # `fft_nthreads` is FFTW's per-transform thread count, fixed at planning: 1 where an outer loop
    # carries the parallelism, `Threads.nthreads()` for a single-field method whose only parallel axis
    # is the transform.
    plans  = _make_fft_plans(advected_hat, ks, pad, fft_nthreads)
    return NonlinearTermWorkspace(u_phys, g_phys, N_phys, N̂, phase, pad, plans)
end

# ---------------------------------------------------------------------------
# SpectralFluxWorkspace
# ---------------------------------------------------------------------------

"""
    SpectralFluxWorkspace{NW, V}

Preallocated buffers for `calculate_spectral_flux!`.

# Fields
- `nonlinear::NW`:        `NonlinearTermWorkspace` for computing N̂(k).
- `T_spec::V`:            Shell transfer spectrum buffer (length N_sh), in the field's array type.
- `flux::V`:              Cumulative flux buffer (length N_sh).
"""
struct SpectralFluxWorkspace{NW<:NonlinearTermWorkspace, V<:AbstractVector}
    nonlinear::NW
    T_spec::V
    flux::V
end

"""
    SpectralFluxWorkspace(velocity_hat, ks, binning; geometry=IsotropicShells(),
                          dealiasing=OrszagTwoThirds(), fft_nthreads=1)

Construct a `SpectralFluxWorkspace` for the given input and binning.
"""
function SpectralFluxWorkspace(velocity_hat, ks, binning::Types.AbstractShellBinning;
                               geometry::Types.AbstractShellGeometry = Types.IsotropicShells(),
                               dealiasing::Types.AbstractDealiasing = Types.OrszagTwoThirds(),
                               fft_nthreads::Int = 1)
    edges  = ShellBinning.shell_edges(binning, ShellBinning.max_shell_coordinate(geometry, ks))
    N_sh   = length(edges) - 1
    FT     = real(eltype(velocity_hat))
    return SpectralFluxWorkspace(
        NonlinearTermWorkspace(velocity_hat, ks; dealiasing=dealiasing, fft_nthreads=fft_nthreads),
        similar(velocity_hat, FT, N_sh),     # T_spec
        similar(velocity_hat, FT, N_sh),     # flux
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
                              IA<:AbstractArray{Int}}
    nonlinear::NW
    û_m::CA
    T_mat::M
    net_transfer::V
    shell_idx::IA
end

"""
    ShellToShellWorkspace(velocity_hat, ks, binning; geometry=IsotropicShells(),
                          dealiasing=OrszagTwoThirds())

Construct a `ShellToShellWorkspace` for the given input and binning.
"""
function ShellToShellWorkspace(velocity_hat, ks, binning::Types.AbstractShellBinning;
                               geometry::Types.AbstractShellGeometry = Types.IsotropicShells(),
                               dealiasing::Types.AbstractDealiasing = Types.OrszagTwoThirds())
    FT        = real(eltype(velocity_hat))
    k_mag     = ShellBinning.shell_coordinate(geometry, ks)
    edges     = ShellBinning.shell_edges(binning, ShellBinning.max_shell_coordinate(geometry, ks))
    N_sh      = length(edges) - 1
    ns        = FTB.spectral_size(ks)
    # The field's array kind, so a device field carries a device-resident index grid for the reduction
    # kernels.
    shell_idx = similar(velocity_hat, Int, ns...)
    copyto!(shell_idx, ShellBinning.assign_shells(k_mag, edges))
    return ShellToShellWorkspace(
        NonlinearTermWorkspace(velocity_hat, ks; dealiasing=dealiasing),
        similar(velocity_hat),                   # û_m
        similar(velocity_hat, FT, N_sh, N_sh),   # T_mat
        similar(velocity_hat, FT, N_sh),         # net_transfer
        shell_idx,
    )
end

# One-line `show` for plan-owning structs: the default field-by-field show of a struct holding an FFTW
# plan can crash in `fftw_sprint_plan` (after `close!`, or across a Distributed transfer).
Base.show(io::IO, ::PaddedGrid) = print(io, "PaddedGrid(…)")
Base.show(io::IO, ::MIME"text/plain", p::PaddedGrid) = show(io, p)
Base.show(io::IO, ::NonlinearTermWorkspace) = print(io, "NonlinearTermWorkspace(…)")
Base.show(io::IO, ::MIME"text/plain", w::NonlinearTermWorkspace) = show(io, w)
Base.show(io::IO, ::SpectralFluxWorkspace) = print(io, "SpectralFluxWorkspace(…)")
Base.show(io::IO, ::MIME"text/plain", w::SpectralFluxWorkspace) = show(io, w)
Base.show(io::IO, ::ShellToShellWorkspace) = print(io, "ShellToShellWorkspace(…)")
Base.show(io::IO, ::MIME"text/plain", w::ShellToShellWorkspace) = show(io, w)

end # module Workspaces
