module SpectralFlux

using ..Types: Types
using ..SpectralLayout: SpectralLayout
using ComputationalBackends: ComputationalBackends
using SpectralBackends: SpectralBackends
using ..Invariants: Invariants
using ..Decomposition: Decomposition
using ..ShellBinning: ShellBinning
using ..Utils: Utils
using ..NonlinearTerm: NonlinearTerm
using ..Workspaces: Workspaces

export calculate_spectral_flux, calculate_spectral_flux!, calculate_spectral_flux_batch, calculate_scalar_flux, calculate_scalar_flux!, calculate_partial_fluxes, calculate_partial_fluxes!, calculate_partial_fluxes_batch, calculate_helical_partial_fluxes, calculate_helical_partial_fluxes!

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

"""
    calculate_spectral_flux(velocity_hat, ks; binning, dealiasing=OrszagTwoThirds()) -> SpectralFluxResult

Compute the spectral energy transfer spectrum T(k) and the cumulative energy
flux Π(K) from Fourier-space velocity data.

# Arguments
- `velocity_hat`: Complex array of size `(ns..., D)` — Fourier coefficients of
  the D velocity components on an N-point periodic grid.
- `ks`: Tuple of 1D physical-wavenumber vectors (matching FFTW fftfreq convention).

# Keyword Arguments
- `binning::AbstractShellBinning`: Shell binning strategy; default `LinearBinning(1.0)`.
- `dealiasing::AbstractDealiasing=OrszagTwoThirds()`: Apply 2/3 dealiasing rule when computing (u·∇)u.
- `spectral::SpectralBackends.AbstractSpectralBackend`: transform backend — `SpectralBackends.DirectSumSpectralBackend()` (default, no deps) or `SpectralBackends.FFTSpectralBackend()` (requires FFTW extension).
- `execution::ComputationalBackends.AbstractExecutionBackend=ComputationalBackends.SerialBackend()`: how the transfer-density write and the
  mode→shell reduction are parallelised (orthogonal to `spectral`, which threads the FFT itself):
    - `ComputationalBackends.SerialBackend()` (default) — host scalar reduction.
    - `ComputationalBackends.ThreadedBackend()` (requires `using OhMyThreads`) — the mode→shell scatter is split into chunks
      with per-chunk partial shell sums (race-free), an `O(Nᴰ)` parallel pass.
    - `ComputationalBackends.DistributedBackend()` (requires `using Distributed`) — the scatter is partitioned across worker
      processes and `+`-reduced. Note: this distributes only the reduction, not the (dominant) nonlinear-term
      FFT; to distribute a grid too large for one node, use [`pencil_spectral_flux`](@ref FlowInvariantTransfer.pencil_spectral_flux) (PencilFFTs).
    - `ComputationalBackends.GPUBackend(dev)` (requires `using KernelAbstractions`) — the transfer density is written by a device
      kernel and reduced per shell on-device (no host round-trips), so a device-resident field stays on the GPU.

# Returns
`SpectralFluxResult` with fields:
- `k_shells`: Representative wavenumber per shell.
- `transfer_spectrum`: T(k_n) — energy input to shell n per unit time.
- `flux`: Π(K_n) — cumulative upscale flux (energy transferred to k > K_n).

# Physics
  T(k_n) = Σ_{|k| ∈ shell_n} Re{ û*(k) · N̂(k) }
  Π(K_n) = −Σ_{m ≤ n} T(k_m)

Positive Π: forward (downscale) cascade; negative Π: inverse (upscale) cascade.

# References
- Verma et al. (2002) [arXiv:nlin/0204027]
- Alexakis, Mininni & Pouquet (2005)
"""
function calculate_spectral_flux(
    velocity_hat,
    ks;
    binning::Types.AbstractShellBinning = _default_binning(ks),
    dealiasing::Types.AbstractDealiasing = Types.OrszagTwoThirds(),
    invariant::Types.AbstractInvariant = Types.KineticEnergy(),
    decomposition::Types.AbstractFieldDecomposition = Types.NoDecomposition(),
    spectral::SpectralBackends.AbstractSpectralBackend = SpectralBackends.AutoSpectralBackend(),
    execution::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.SerialBackend(),
    advecting_hat = velocity_hat,
    geometry::Types.AbstractShellGeometry = Types.IsotropicShells(),
)
    spectral = Types.resolve_spectral(Types.require_coefficient_spectral(spectral))
    # DirectSum uses scalar-indexed direct sums (host-only); reject a device array up front, before the
    # workspace/FFT-plan build would fail with an opaque "JLArray to Ptr" (FFTW plan on a device array).
    (spectral isa SpectralBackends.DirectSumSpectralBackend && ComputationalBackends.is_gpu_array(velocity_hat)) &&
        throw(ArgumentError(
            "SpectralBackends.DirectSumSpectralBackend uses scalar-indexed direct sums (a host O(N²ᴰ) reference) and cannot " *
            "run on device arrays; use spectral = SpectralBackends.FFTSpectralBackend() (cuFFT via AbstractFFTs) for the device path."))
    decomposed = Decomposition.decompose_field(decomposition, velocity_hat, ks)
    return _calculate_spectral_flux_decomposed(
        decomposed, velocity_hat, ks, binning, dealiasing, invariant, spectral, Types.resolve_execution(execution), advecting_hat, geometry
    )
end

function _calculate_spectral_flux_decomposed(
    û_decomp::AbstractArray{<:Complex},
    velocity_hat,
    ks,
    binning::Types.AbstractShellBinning,
    dealiasing::Types.AbstractDealiasing,
    invariant::Types.AbstractInvariant,
    spectral::SpectralBackends.AbstractSpectralBackend,
    execution::ComputationalBackends.AbstractExecutionBackend,
    advecting_hat,
    geometry::Types.AbstractShellGeometry,
)
    # Single-field method: its only parallel axis is the transform, so under ComputationalBackends.ThreadedBackend the FFTs
    # run multithreaded (FFTW), else single-threaded (0-alloc). No outer loop to thread → no nesting.
    ws        = Workspaces.SpectralFluxWorkspace(velocity_hat, ks, binning; geometry=geometry, dealiasing=dealiasing,
                                       fft_nthreads = execution isa ComputationalBackends.ThreadedBackend ? Threads.nthreads() : 1)
    k_mag     = ShellBinning.shell_coordinate(geometry, ks)
    edges     = ShellBinning.shell_edges(binning, maximum(k_mag))
    centers   = ShellBinning.shell_centers(binning, maximum(k_mag))
    shell_idx = ShellBinning.assign_shells(k_mag, edges)
    RT        = real(eltype(velocity_hat))
    result    = Types.SpectralFluxResult(centers, similar(centers, RT, length(centers)), similar(centers, RT, length(centers)))

    if û_decomp === velocity_hat
        calculate_spectral_flux!(result, ws, velocity_hat, ks, shell_idx;
                                  dealiasing=dealiasing, invariant=invariant, spectral=spectral,
                                  execution=execution, advecting_hat=advecting_hat)
    else
        NonlinearTerm.compute_nonlinear_term!(ws.nonlinear, velocity_hat, ks; dealiasing=dealiasing,
                                spectral=spectral, advecting_hat=advecting_hat)
        _calculate_spectral_flux_with_N̂!(result, ws, û_decomp, ws.nonlinear.N̂, ks, shell_idx, execution; invariant=invariant)
    end
    return result
end

function _calculate_spectral_flux_decomposed(
    decomposed::NamedTuple,
    velocity_hat,
    ks,
    binning::Types.AbstractShellBinning,
    dealiasing::Types.AbstractDealiasing,
    invariant::Types.AbstractInvariant,
    spectral::SpectralBackends.AbstractSpectralBackend,
    execution::ComputationalBackends.AbstractExecutionBackend,
    advecting_hat,
    geometry::Types.AbstractShellGeometry,
)
    ws = Workspaces.SpectralFluxWorkspace(velocity_hat, ks, binning; geometry=geometry, dealiasing=dealiasing)
    NonlinearTerm.compute_nonlinear_term!(ws.nonlinear, velocity_hat, ks; dealiasing=dealiasing,
                            spectral=spectral, advecting_hat=advecting_hat)
    N̂ = ws.nonlinear.N̂

    k_mag     = ShellBinning.shell_coordinate(geometry, ks)
    edges     = ShellBinning.shell_edges(binning, maximum(k_mag))
    centers   = ShellBinning.shell_centers(binning, maximum(k_mag))
    shell_idx = ShellBinning.assign_shells(k_mag, edges)
    RT        = real(eltype(velocity_hat))

    return map(decomposed) do û_comp
        res = Types.SpectralFluxResult(centers, similar(centers, RT, length(centers)), similar(centers, RT, length(centers)))
        _calculate_spectral_flux_with_N̂!(res, ws, û_comp, N̂, ks, shell_idx, execution; invariant=invariant)
        return res
    end
end

"""
    calculate_spectral_flux!(result, ws, velocity_hat, ks, shell_idx;
        dealiasing, invariant, spectral, execution, advecting_hat)

In-place version of `calculate_spectral_flux`. Writes into `result` using
preallocated buffers from `ws` and a precomputed `shell_idx` array (from `assign_shells`).
Zero heap allocations in the serial hot path. `execution` selects the mode→shell reduction
backend (see [`calculate_spectral_flux`](@ref)).
"""
function calculate_spectral_flux!(
    result::Types.SpectralFluxResult,
    ws::Workspaces.SpectralFluxWorkspace,
    velocity_hat,
    ks,
    shell_idx::AbstractArray{Int};
    dealiasing::Types.AbstractDealiasing = Types.OrszagTwoThirds(),
    invariant::Types.AbstractInvariant = Types.KineticEnergy(),
    spectral::SpectralBackends.AbstractSpectralBackend = SpectralBackends.AutoSpectralBackend(),
    execution::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.SerialBackend(),
    advecting_hat = velocity_hat,
)
    NonlinearTerm.compute_nonlinear_term!(ws.nonlinear, velocity_hat, ks;
                            dealiasing=dealiasing, spectral=spectral, advecting_hat=advecting_hat)
    _calculate_spectral_flux_with_N̂!(result, ws, velocity_hat, ws.nonlinear.N̂, ks, shell_idx, Types.resolve_execution(execution); invariant=invariant)
    return result
end

"""
    calculate_spectral_flux_batch(velocity_hats, ks; binning, dealiasing, invariant, spectral, geometry, execution)
        -> Vector{SpectralFluxResult}

Spectral flux for a batch of snapshots that share one grid (`velocity_hats` an iterable of `(ns..., D)`
coefficient arrays). The snapshot-independent shell structure is built ONCE, and each worker reuses a
single `SpectralFluxWorkspace` (FFT plans + scratch) across its snapshots — so the plan build is
amortised over the whole batch, not paid per snapshot. `execution = ThreadedBackend()` (requires
`using OhMyThreads`) splits the snapshots across threads with a **serial** inner transform (the batch is
the outer parallel axis; the FFTs stay single-threaded, so there is no nested threading). Results are
returned in input order and are bit-identical to calling [`calculate_spectral_flux`](@ref) per snapshot.
"""
function calculate_spectral_flux_batch(
    velocity_hats,
    ks;
    binning::Types.AbstractShellBinning = _default_binning(ks),
    dealiasing::Types.AbstractDealiasing = Types.OrszagTwoThirds(),
    invariant::Types.AbstractInvariant = Types.KineticEnergy(),
    spectral::SpectralBackends.AbstractSpectralBackend = SpectralBackends.AutoSpectralBackend(),
    geometry::Types.AbstractShellGeometry = Types.IsotropicShells(),
    execution::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.SerialBackend(),
)
    spectral = Types.resolve_spectral(Types.require_coefficient_spectral(spectral))
    n = length(velocity_hats)
    n == 0 && return Types.SpectralFluxResult[]
    û1        = first(velocity_hats)
    k_mag     = ShellBinning.shell_coordinate(geometry, ks)
    edges     = ShellBinning.shell_edges(binning, maximum(k_mag))
    centers   = ShellBinning.shell_centers(binning, maximum(k_mag))
    shell_idx = ShellBinning.assign_shells(k_mag, edges)
    RT        = real(eltype(û1))
    results   = [Types.SpectralFluxResult(centers, similar(centers, RT, length(centers)), similar(centers, RT, length(centers)))
                 for _ in 1:n]
    _spectral_flux_batch!(Types.resolve_execution(execution), results, velocity_hats, ks, shell_idx;
                          binning=binning, dealiasing=dealiasing, invariant=invariant, spectral=spectral, geometry=geometry)
    return results
end

# Serial reference: one workspace reused across the whole batch.
function _spectral_flux_batch!(::ComputationalBackends.AbstractSerialBackend, results, velocity_hats, ks, shell_idx;
                               binning, dealiasing, invariant, spectral, geometry)
    ws = Workspaces.SpectralFluxWorkspace(first(velocity_hats), ks, binning; geometry=geometry, dealiasing=dealiasing)
    for i in eachindex(velocity_hats)
        calculate_spectral_flux!(results[i], ws, velocity_hats[i], ks, shell_idx;
                                 dealiasing=dealiasing, invariant=invariant, spectral=spectral,
                                 execution=ComputationalBackends.SerialBackend())
    end
    return results
end

# Threaded over the batch — overridden by the OhMyThreads extension (per-chunk workspace pool, serial inner).
function _spectral_flux_batch_threaded!(args...; kwargs...)
    throw(ArgumentError("execution = ThreadedBackend() for the spectral-flux batch requires OhMyThreads. " *
                        "Run `using OhMyThreads` to load the extension."))
end
_spectral_flux_batch!(::ComputationalBackends.AbstractThreadedBackend, results, velocity_hats, ks, shell_idx; kwargs...) =
    _spectral_flux_batch_threaded!(results, velocity_hats, ks, shell_idx; kwargs...)

# Distributed over the batch — overridden by the Distributed extension (snapshots partitioned across workers).
function _spectral_flux_batch_distributed!(args...; kwargs...)
    throw(ArgumentError("execution = DistributedBackend() for the spectral-flux batch requires Distributed. " *
                        "Run `using Distributed` to load the extension."))
end
_spectral_flux_batch!(::ComputationalBackends.AbstractDistributedBackend, results, velocity_hats, ks, shell_idx; kwargs...) =
    _spectral_flux_batch_distributed!(results, velocity_hats, ks, shell_idx; kwargs...)

# GPU over the batch — loop the single-shot device kernel, reusing one device workspace (built from the
# first snapshot via `similar`, so device inputs give device buffers). The batch is the outer axis; each
# flux runs on-device. Requires the FFT backend (DirectSum is a host-only O(N²ᴰ) reference).
function _spectral_flux_batch!(gpu::ComputationalBackends.AbstractGPUBackend, results, velocity_hats, ks, shell_idx;
                               binning, dealiasing, invariant, spectral, geometry)
    spectral isa SpectralBackends.DirectSumSpectralBackend && throw(ArgumentError(
        "calculate_spectral_flux_batch on a GPUBackend requires spectral = SpectralBackends.FFTSpectralBackend() " *
        "(cuFFT via AbstractFFTs); SpectralBackends.DirectSumSpectralBackend is a host-only reference."))
    ws = Workspaces.SpectralFluxWorkspace(first(velocity_hats), ks, binning; geometry=geometry, dealiasing=dealiasing)
    for i in eachindex(velocity_hats)
        calculate_spectral_flux!(results[i], ws, velocity_hats[i], ks, shell_idx;
                                 dealiasing=dealiasing, invariant=invariant, spectral=spectral, execution=gpu)
    end
    return results
end

# Any other execution backend has no batch hook — refuse rather than silently run serial.
_spectral_flux_batch!(be::ComputationalBackends.AbstractExecutionBackend, results, velocity_hats, ks, shell_idx; kwargs...) =
    throw(ArgumentError("calculate_spectral_flux_batch supports SerialBackend(), ThreadedBackend(), and DistributedBackend(); " *
                        "got execution = $(typeof(be))."))

# ---------------------------------------------------------------------------
# Transfer-density write + mode→shell reduction, dispatched on the execution backend.
# The ComputationalBackends.SerialBackend method (below) is the reference host reduction. The Threaded / Distributed /
# GPU methods live in extensions (OhMyThreads / Distributed / KernelAbstractions) and override the
# named stubs `_spectral_flux_{threaded,distributed,gpu}!` — same core-stub-overridden-by-extension
# pattern as `ShellToShellTransfer._shell_to_shell_threaded!`. This is the transfer-density + scatter
# step only; `compute_nonlinear_term!` (the N̂/FFT) runs upstream on the array's own kind/threads.
# ---------------------------------------------------------------------------

function _calculate_spectral_flux_with_N̂!(
    result::Types.SpectralFluxResult,
    ws::Workspaces.SpectralFluxWorkspace,
    velocity_hat,
    N̂,
    ks,
    shell_idx::AbstractArray{Int},
    ::ComputationalBackends.SerialBackend;
    invariant::Types.AbstractInvariant = Types.KineticEnergy(),
)
    # Density and shell scatter in one pass: the density grid is written, read and discarded within
    # this call, so it never needs to exist.
    Invariants.transfer_density_scatter!(ws.T_spec, invariant, velocity_hat, N̂, ks, shell_idx)
    return _finalize_spectral_flux!(result, ws)
end

# Shared tail: copy T(k) out and accumulate the cumulative flux Π(K). Called by every backend once
# `ws.T_spec` holds the shell transfer spectrum, so the flux convention lives in exactly one place.
# Flux convention (Alexakis & Biferale 2018, Eqs. 12–17):
#   T(k) = Re{û*·N̂} is the net energy *loss* from shell k, and
#   Π(K) = +Σ_{k≤K} T(k)  ⇒  Π>0 forward (down-scale) cascade, Π<0 inverse.
# (Earlier code negated this, returning −Π — i.e. forward cascades read as negative.)
function _finalize_spectral_flux!(result::Types.SpectralFluxResult, ws::Workspaces.SpectralFluxWorkspace)
    # Bring the (small, N_sh) shell spectrum to the host result and form Π = cumsum(T) there. Doing the
    # cumsum on the result vectors keeps this device-agnostic: a device `ws.T_spec` is copied to the host
    # result once and `cumsum!` never needs a device scan (Base's `cumsum!` scalar-indexes non-CUDA device
    # arrays). The heavy per-mode work already ran on-device upstream; the summary is cheap on the host.
    copyto!(result.transfer_spectrum, ws.T_spec)
    cumsum!(result.flux, result.transfer_spectrum)
    return result
end

# Dispatch the parallel backends to named stubs, overridden by the matching extension. Until the
# extension is loaded, each stub throws an informative "run `using X`" error (never a bare MethodError).
_calculate_spectral_flux_with_N̂!(result::Types.SpectralFluxResult, ws::Workspaces.SpectralFluxWorkspace, velocity_hat, N̂, ks, shell_idx::AbstractArray{Int}, ::ComputationalBackends.ThreadedBackend; kwargs...) =
    _spectral_flux_threaded!(result, ws, velocity_hat, N̂, ks, shell_idx; kwargs...)
_calculate_spectral_flux_with_N̂!(result::Types.SpectralFluxResult, ws::Workspaces.SpectralFluxWorkspace, velocity_hat, N̂, ks, shell_idx::AbstractArray{Int}, ::ComputationalBackends.DistributedBackend; kwargs...) =
    _spectral_flux_distributed!(result, ws, velocity_hat, N̂, ks, shell_idx; kwargs...)
_calculate_spectral_flux_with_N̂!(result::Types.SpectralFluxResult, ws::Workspaces.SpectralFluxWorkspace, velocity_hat, N̂, ks, shell_idx::AbstractArray{Int}, gpu::ComputationalBackends.GPUBackend; kwargs...) =
    _spectral_flux_gpu!(result, ws, velocity_hat, N̂, ks, shell_idx, gpu; kwargs...)

_spectral_flux_threaded!(args...; kwargs...) = throw(ArgumentError(
    "Threaded spectral flux requires OhMyThreads. Run `using OhMyThreads` to load the extension."))
_spectral_flux_distributed!(args...; kwargs...) = throw(ArgumentError(
    "Distributed spectral flux requires Distributed. Run `using Distributed` to load the extension."))
_spectral_flux_gpu!(args...; kwargs...) = throw(ArgumentError(
    "GPU spectral flux requires KernelAbstractions. Run `using KernelAbstractions` to load the extension."))

# ---------------------------------------------------------------------------
# Passive-scalar variance flux (convenience over the generalized advecting_hat path)
# ---------------------------------------------------------------------------

"""
    calculate_scalar_flux(velocity_hat, scalar_hat, ks; binning, dealiasing=OrszagTwoThirds(), spectral) -> SpectralFluxResult

Compute the passive-scalar **variance** transfer spectrum `T_θ(k)` and flux `Π_θ(K)`, for a
scalar `θ` advected by the velocity `u` (`∂_tθ + (u·∇)θ = κ∇²θ`):

    T_θ(k_n) = Σ_{|k|∈shell_n} Re{ θ̂*(k) N̂_θ(k) },   N̂_θ = FFT[(u·∇)θ],   Π_θ(K) = Σ_{k≤K} T_θ(k).

Scalar variance is conserved for incompressible `u` (`Σ_k T_θ ≈ 0`) and cascades forward in any
dimension (Obukhov–Corrsin). Thin wrapper over [`calculate_spectral_flux`](@ref) with
`invariant = PassiveScalar()` and `advecting_hat = velocity_hat`.

# Arguments
- `velocity_hat`: complex `(ns..., D)` velocity Fourier coefficients (the advecting field).
- `scalar_hat`: complex scalar field, either `(ns...)` or `(ns..., 1)`.
- `ks`: tuple of `nd` 1D wavenumber vectors.
"""
function calculate_scalar_flux(
    velocity_hat,
    scalar_hat,
    ks;
    binning::Types.AbstractShellBinning = _default_binning(ks),
    dealiasing::Types.AbstractDealiasing = Types.OrszagTwoThirds(),
    spectral::SpectralBackends.AbstractSpectralBackend = SpectralBackends.AutoSpectralBackend(),
    execution::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.SerialBackend(),
    geometry::Types.AbstractShellGeometry = Types.IsotropicShells(),
)
    θ̂ = Utils.as_component_field(scalar_hat, length(ks))
    return calculate_spectral_flux(θ̂, ks; binning=binning, dealiasing=dealiasing,
        invariant=Types.PassiveScalar(), advecting_hat=velocity_hat, spectral=spectral,
        execution=execution, geometry=geometry)
end

"""
    calculate_scalar_flux!(result, ws, velocity_hat, scalar_hat, ks, shell_idx; kwargs...)

In-place passive-scalar variance flux — thin wrapper over [`calculate_spectral_flux!`](@ref)
(`invariant = PassiveScalar()`, scalar advected by `velocity_hat`), writing into the
caller-provided `result`/`ws` (0 alloc beyond them; `ws` sized for the scalar field).
"""
function calculate_scalar_flux!(result, ws, velocity_hat, scalar_hat, ks, shell_idx; kwargs...)
    θ̂ = Utils.as_component_field(scalar_hat, length(ks))
    return calculate_spectral_flux!(result, ws, θ̂, ks, shell_idx;
        invariant=Types.PassiveScalar(), advecting_hat=velocity_hat, kwargs...)
end

# ---------------------------------------------------------------------------
# Helical partial energy fluxes Π^{s_k s_p s_q} (Waleffe 1992; Biferale–Musacchio–Toschi 2012)
# ---------------------------------------------------------------------------

"""
    calculate_partial_fluxes(velocity_hat, ks; decomposition=HelicalDecomposition(), binning,
        dealiasing=OrszagTwoThirds(), spectral=SpectralBackends.DirectSumSpectralBackend(), geometry=IsotropicShells())
        -> (channels::Dict{NTuple{3,Symbol},SpectralFluxResult}, total::SpectralFluxResult, k_shells)

Decompose the kinetic-energy flux into **per-component partial fluxes** `Π^{s_k s_p s_q}(K)`,
where each of the three fields in a triad interaction is one component of a velocity decomposition
`u = Σ_s u_s` (e.g. `±`-helical via [`HelicalDecomposition`](@ref), or rotational/divergent via
[`HelmholtzDecomposition`](@ref)):

    T^{s_k s_p s_q}(k) = Re{ û_{s_k}*(k) · [ (u_{s_p}·∇) u_{s_q} ](k) },   Π = Σ_{k≤K} T.

With an `n`-component decomposition this gives `n³` channels that sum to the full energy flux. For
helical components the **homochiral** channels (`s_k=s_p=s_q`) drive the inverse cascade and the
heterochiral ones the forward cascade (Biferale–Musacchio–Toschi 2012); for the Helmholtz split
the off-diagonal channels are the **rotational↔divergent cross-flux** (zero for incompressible
flow, since `u_div = 0`). `channels` is keyed by the component-name triple `(s_k, s_p, s_q)`.
Built from the decomposition + generalized nonlinear term, so it inherits all backends/dealiasing.
"""
function calculate_partial_fluxes(velocity_hat, ks; kwargs...)
    ws = Workspaces.NonlinearTermWorkspace(velocity_hat, ks)
    return calculate_partial_fluxes!(ws, velocity_hat, ks; kwargs...)
end

"""
    calculate_partial_fluxes_batch(velocity_hats, ks; execution, kwargs...) -> Vector

Decomposition-channel fluxes for a batch of snapshots sharing one grid. One nonlinear-term workspace is
reused across a worker's snapshots (the same workspace the single-snapshot form builds once and threads
through all `n³` channel pairs). `execution = ThreadedBackend()` (requires `using OhMyThreads`) threads
over snapshots with a serial inner transform. Results are in input order.
"""
function calculate_partial_fluxes_batch(
    velocity_hats, ks;
    execution::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.SerialBackend(),
    kwargs...,
)
    length(velocity_hats) == 0 && return NamedTuple[]
    return _partial_fluxes_batch!(Types.resolve_execution(execution), velocity_hats, ks; kwargs...)
end

function _partial_fluxes_batch!(::ComputationalBackends.AbstractSerialBackend, velocity_hats, ks; kwargs...)
    ws = Workspaces.NonlinearTermWorkspace(first(velocity_hats), ks)
    return [calculate_partial_fluxes!(ws, vh, ks; kwargs...) for vh in velocity_hats]
end

function _partial_fluxes_batch_threaded!(args...; kwargs...)
    throw(ArgumentError("execution = ThreadedBackend() for the partial-flux batch requires OhMyThreads. " *
                        "Run `using OhMyThreads` to load the extension."))
end
_partial_fluxes_batch!(::ComputationalBackends.AbstractThreadedBackend, velocity_hats, ks; kwargs...) =
    _partial_fluxes_batch_threaded!(velocity_hats, ks; kwargs...)

_partial_fluxes_batch!(be::ComputationalBackends.AbstractExecutionBackend, velocity_hats, ks; kwargs...) =
    throw(ArgumentError("calculate_partial_fluxes_batch supports SerialBackend() and ThreadedBackend(); " *
                        "got execution = $(typeof(be))."))

"""
    calculate_partial_fluxes!(ws::NonlinearTermWorkspace, velocity_hat, ks; kwargs...)

In-place partial-flux computation reusing a caller-provided `ws` across **all** decomposition-channel
pairs. The channel `SpectralFluxResult`s and their total are the (inherent) output.
"""
function calculate_partial_fluxes!(
    ws::Workspaces.NonlinearTermWorkspace,
    velocity_hat,
    ks;
    decomposition::Types.AbstractFieldDecomposition = Types.HelicalDecomposition(),
    binning::Types.AbstractShellBinning = _default_binning(ks),
    dealiasing::Types.AbstractDealiasing = Types.OrszagTwoThirds(),
    spectral::SpectralBackends.AbstractSpectralBackend = SpectralBackends.AutoSpectralBackend(),
    execution::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.SerialBackend(),
    geometry::Types.AbstractShellGeometry = Types.IsotropicShells(),
)
    spectral = Types.resolve_spectral(Types.require_coefficient_spectral(spectral))
    comps = Decomposition.decompose_field(decomposition, velocity_hat, ks)
    comps isa NamedTuple || throw(ArgumentError(
        "calculate_partial_fluxes needs a decomposition that splits u into ≥2 named components " *
        "(e.g. Types.HelicalDecomposition or Types.HelmholtzDecomposition); got $(typeof(decomposition))."))
    names = keys(comps)

    k_coord = ShellBinning.shell_coordinate(geometry, ks)
    edges   = ShellBinning.shell_edges(binning, maximum(k_coord))
    centers = collect(ShellBinning.shell_centers(binning, maximum(k_coord)))
    sidx    = ShellBinning.assign_shells(k_coord, edges)
    Nsh     = length(centers)

    channels = Dict{NTuple{3,Symbol}, Types.SpectralFluxResult}()
    # Fill the n²-pair × n-channel decomposition, dispatched on execution: each (sp,sq) pair is an
    # independent nonlinear-term build → embarrassingly parallel over the n² pairs (single-threaded
    # inner FFTs per worker → no oversubscription). The channel `SpectralFluxResult`s are the output.
    _partial_fluxes_fill!(Types.resolve_execution(execution), channels, ws, comps, names,
        velocity_hat, ks, sidx, centers, Nsh; dealiasing=dealiasing, spectral=spectral)
    total = Types.SpectralFluxResult(centers,
        sum(c.transfer_spectrum for c in values(channels)),
        sum(c.flux for c in values(channels)))
    return (channels = channels, total = total, k_shells = centers)
end

# Bin a transfer density into shells → a channel SpectralFluxResult (part of the inherent output).
function _partial_binflux(td, sidx, ks, centers, Nsh)
    FT = eltype(td)
    T = zeros(FT, Nsh)
    @inbounds for I in CartesianIndices(td)
        n = sidx[I]; n == 0 && continue
        T[n] += SpectralLayout.hermitian_weight(ks, I) * td[I]
    end
    return Types.SpectralFluxResult(centers, T, cumsum(T))
end

# Serial channel fill: reuse the caller ws + one td scratch across every (sp,sq) pair.
function _partial_fluxes_fill!(::ComputationalBackends.SerialBackend, channels, ws, comps, names, velocity_hat, ks, sidx, centers, Nsh;
                               dealiasing, spectral)
    FT = real(eltype(velocity_hat)); nd = length(ks); ns = size(velocity_hat)[1:nd]
    td = similar(velocity_hat, FT, ns...)
    for sp in names, sq in names
        # (u_{sp}·∇)u_{sq} into the shared workspace's N̂ (reused across every pair).
        NonlinearTerm.compute_nonlinear_term!(ws, comps[sq], ks; advecting_hat=comps[sp], dealiasing=dealiasing, spectral=spectral)
        for sk in names
            Invariants.transfer_density!(td, Types.KineticEnergy(), comps[sk], ws.N̂, ks)
            channels[(sk, sp, sq)] = _partial_binflux(td, sidx, ks, centers, Nsh)
        end
    end
    return channels
end
_partial_fluxes_fill!(::ComputationalBackends.ThreadedBackend, channels, ws, comps, names, velocity_hat, ks, sidx, centers, Nsh; kwargs...) =
    _partial_fluxes_threaded!(channels, ws, comps, names, velocity_hat, ks, sidx, centers, Nsh; kwargs...)
_partial_fluxes_fill!(exec::ComputationalBackends.DistributedBackend, channels, ws, comps, names, velocity_hat, ks, sidx, centers, Nsh; kwargs...) =
    _partial_fluxes_distributed!(channels, ws, comps, names, velocity_hat, ks, sidx, centers, Nsh, exec; kwargs...)
_partial_fluxes_fill!(gpu::ComputationalBackends.GPUBackend, channels, ws, comps, names, velocity_hat, ks, sidx, centers, Nsh; kwargs...) =
    _partial_fluxes_gpu!(channels, ws, comps, names, velocity_hat, ks, sidx, centers, Nsh, gpu; kwargs...)

_partial_fluxes_threaded!(args...; kwargs...) = throw(ArgumentError(
    "Threaded partial fluxes require OhMyThreads. Run `using OhMyThreads` to load the extension."))
_partial_fluxes_distributed!(args...; kwargs...) = throw(ArgumentError(
    "Distributed partial fluxes require Distributed. Run `using Distributed` to load the extension."))
_partial_fluxes_gpu!(args...; kwargs...) = throw(ArgumentError(
    "GPU partial fluxes require KernelAbstractions. Run `using KernelAbstractions` to load the extension."))

"""
    calculate_helical_partial_fluxes(velocity_hat, ks; kwargs...)

The eight **helical** partial energy fluxes `Π^{s_k s_p s_q}(K)`, `s ∈ {positive, negative}` —
[`calculate_partial_fluxes`](@ref) with `decomposition = HelicalDecomposition()` (3D only).
Homochiral channels drive the inverse cascade, heterochiral the forward (Waleffe 1992;
Biferale–Musacchio–Toschi 2012; Alexakis 2017).
"""
calculate_helical_partial_fluxes(velocity_hat, ks; kwargs...) =
    calculate_partial_fluxes(velocity_hat, ks; decomposition=Types.HelicalDecomposition(), kwargs...)

"""
    calculate_helical_partial_fluxes!(ws::NonlinearTermWorkspace, velocity_hat, ks; kwargs...)

In-place helical partial fluxes — [`calculate_partial_fluxes!`](@ref) with
`decomposition = HelicalDecomposition()`, reusing the caller's `ws`.
"""
calculate_helical_partial_fluxes!(ws, velocity_hat, ks; kwargs...) =
    calculate_partial_fluxes!(ws, velocity_hat, ks; decomposition=Types.HelicalDecomposition(), kwargs...)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

function _default_binning(ks)
    # Default linear binning with spacing = minimum non-zero wavenumber increment
    min_dk = Inf
    for k_vec in ks
        for k in k_vec
            ak = abs(k)
            ak > 0 && (min_dk = min(min_dk, ak))
        end
    end
    min_dk = isfinite(min_dk) ? min_dk : 1.0
    return Types.LinearBinning(min_dk)
end

end # module SpectralFlux
