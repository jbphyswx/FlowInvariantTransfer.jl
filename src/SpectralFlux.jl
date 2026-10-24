module SpectralFlux

using ..Types: Types
using FlowTransformBindings: FlowTransformBindings as FTB
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
    calculate_spectral_flux(velocity_hat, ks; binning, dealiasing=OrszagTwoThirds(), invariant=KineticEnergy(),
        decomposition=NoDecomposition(), transform=AutoSpectralBackend(), execution=SerialBackend(),
        advecting_hat=velocity_hat, geometry=IsotropicShells()) -> SpectralFluxResult

Compute the spectral energy transfer spectrum T(k) and the cumulative energy
flux Π(K) from Fourier-space velocity data.

# Arguments
- `velocity_hat`: Complex array of size `(ms..., D)` — Fourier coefficients of the D velocity
  components on the spectral layout `ks` describes (`ms = ns`, or the half layout of a real field).
- `ks`: Tuple of 1D physical-wavenumber vectors (matching FFTW fftfreq convention).

# Keyword Arguments
- `binning::AbstractShellBinning`: Shell binning strategy; default `LinearBinning(Δk)` with `Δk` the
  smallest nonzero `|k|` on any axis.
- `dealiasing::AbstractDealiasing=OrszagTwoThirds()`: Apply 2/3 dealiasing rule when computing (u·∇)u.
- `invariant::AbstractInvariant=KineticEnergy()`: the quadratic invariant whose transfer is binned.
- `decomposition::AbstractFieldDecomposition=NoDecomposition()`: with a splitting decomposition the
  result is a `NamedTuple` of `SpectralFluxResult`s, one per component, each against the full `N̂`.
- `transform::SpectralBackends.AbstractSpectralBackend`: transform backend — `SpectralBackends.AutoSpectralBackend()`
  (default; `FFTSpectralBackend()` when FFTW is loaded), `SpectralBackends.FFTSpectralBackend()`, or
  `SpectralBackends.DirectSumSpectralBackend()` (no dependencies).
- `execution::ComputationalBackends.AbstractExecutionBackend=ComputationalBackends.SerialBackend()`: how the
  transfer-density write and the mode→shell reduction are parallelised:
    - `ComputationalBackends.SerialBackend()` (default) — host scalar reduction.
    - `ComputationalBackends.ThreadedBackend()` (requires `using OhMyThreads`) — the mode→shell scatter is split into chunks
      with per-chunk partial shell sums (race-free), an `O(Nᴰ)` parallel pass.
    - `ComputationalBackends.DistributedBackend()` (requires `using Distributed`) — the scatter is partitioned across worker
      processes and `+`-reduced; the nonlinear-term FFTs stay on the calling process. A grid too large
      for one node takes [`pencil_spectral_flux`](@ref FlowInvariantTransfer.pencil_spectral_flux) (PencilFFTs).
    - `ComputationalBackends.GPUBackend(dev)` (requires `using KernelAbstractions`) — the transfer density is written by a device
      kernel and reduced per shell on-device, so a device-resident field stays on the GPU; host fields
      are copied to `dev` once per call.
- `advecting_hat=velocity_hat`: the advecting velocity of `(u·∇)f`.
- `geometry::AbstractShellGeometry=IsotropicShells()`: the wavenumber coordinate the shells partition.
- `form::AbstractNonlinearForm=Advective()`: [`Rotational()`](@ref Types.Rotational) evaluates `ω × u`
  with fewer transforms, the same fluxes for a divergence-free velocity advecting itself under the 2/3
  rule; it takes kinetic energy, enstrophy or helicity and no `advecting_hat`.

# Returns
`SpectralFluxResult` with fields:
- `k_shells`: Representative wavenumber per shell; `shell_edges`: the shell boundaries.
- `transfer_spectrum`: T(k_n) = −dE_n/dt from the nonlinear term, the energy shell n gives to the
  other shells per unit time — a sum over the shell's modes.
- `flux`: Π[n] — the energy shells 1…n pass to larger wavenumbers per unit time, the flux through
  `shell_edges[n+1]`.

# Physics
  T(k_n) = Σ_{|k| ∈ shell_n} Re{ û*(k) · N̂(k) }
  Π[n]   = Σ_{m ≤ n} T(k_m)

Positive Π: forward (downscale) cascade; negative Π: inverse (upscale) cascade.

# References
- Verma et al. (2002) [arXiv:nlin/0204027]
- Alexakis, Mininni & Pouquet (2005)
"""
function calculate_spectral_flux(
    velocity_hat,
    ks;
    binning::Types.AbstractShellBinning = ShellBinning.default_binning(ks),
    dealiasing::Types.AbstractDealiasing = Types.OrszagTwoThirds(),
    invariant::Types.AbstractInvariant = Types.KineticEnergy(),
    decomposition::Types.AbstractFieldDecomposition = Types.NoDecomposition(),
    transform::SpectralBackends.AbstractSpectralBackend = SpectralBackends.AutoSpectralBackend(),
    execution::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.SerialBackend(),
    advecting_hat = velocity_hat,
    geometry::Types.AbstractShellGeometry = Types.IsotropicShells(),
    form::Types.AbstractNonlinearForm = Types.Advective(),
)
    transform = Types.resolve_transform(Types.require_coefficient_transform(transform))
    _check_form(form, velocity_hat, ks, invariant, advecting_hat)
    exec = Types.resolve_execution(execution)
    velocity_hat, advecting_hat = Types._fields_to_execution(exec, velocity_hat, advecting_hat)
    # The direct sum scalar-indexes, so a device array is rejected before the workspace build.
    (transform isa SpectralBackends.DirectSumSpectralBackend && Types._is_device(velocity_hat)) &&
        throw(ArgumentError(
            "SpectralBackends.DirectSumSpectralBackend uses scalar-indexed direct sums (a host O(N²ᴰ) reference) and cannot " *
            "run on device arrays; use transform = SpectralBackends.FFTSpectralBackend() (cuFFT via AbstractFFTs) for the device path."))
    decomposed = Decomposition.decompose_field(decomposition, velocity_hat, ks)
    return _calculate_spectral_flux_decomposed(
        decomposed, velocity_hat, ks, binning, dealiasing, invariant, transform, exec, advecting_hat, geometry,
        form
    )
end

# `Rotational()` stands for `(u·∇)u` only in the transfer of a velocity advecting itself, of an invariant
# whose transfer projects out a gradient.
_check_form(::Types.Advective, velocity_hat, ks, invariant, advecting_hat) = nothing
function _check_form(::Types.Rotational, velocity_hat, ks, invariant, advecting_hat)
    nd = length(ks)
    (nd in (2, 3) && size(velocity_hat, nd + 1) == nd) || throw(ArgumentError(
        "form = Rotational() takes a 2-D or 3-D velocity with one component per dimension; got " *
        "$nd dimensions and $(size(velocity_hat, nd + 1)) components."))
    invariant isa Union{Types.KineticEnergy, Types.Enstrophy, Types.Helicity} || throw(ArgumentError(
        "form = Rotational() takes KineticEnergy(), Enstrophy() or Helicity(); got $(nameof(typeof(invariant)))()."))
    advecting_hat === velocity_hat || throw(ArgumentError(
        "form = Rotational() evaluates a velocity advecting itself; pass no `advecting_hat`."))
    return nothing
end

# The nonlinear term the flux reads, in `form`.
_flux_term!(ws, velocity_hat, ks, ::Types.Advective; dealiasing, transform, advecting_hat) =
    NonlinearTerm.compute_nonlinear_term!(ws, velocity_hat, ks; dealiasing = dealiasing, transform = transform,
                                          advecting_hat = advecting_hat)
_flux_term!(ws, velocity_hat, ks, ::Types.Rotational; dealiasing, transform, advecting_hat) =
    NonlinearTerm.compute_rotational_term!(ws, velocity_hat, velocity_hat, ks; dealiasing = dealiasing,
                                           transform = transform)

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
    form::Types.AbstractNonlinearForm,
)
    # Single-field method with no threaded outer loop, so `ThreadedBackend` threads the FFTs.
    ws        = Workspaces.SpectralFluxWorkspace(velocity_hat, ks, binning; geometry=geometry, dealiasing=dealiasing,
                                       fft_nthreads = execution isa ComputationalBackends.ThreadedBackend ? Threads.nthreads() : 1)
    k_mag     = ShellBinning.shell_coordinate(geometry, ks)
    edges     = ShellBinning.shell_edges(binning, maximum(k_mag))
    centers   = ShellBinning.shell_centers(binning, maximum(k_mag))
    shell_idx = ShellBinning.assign_shells(k_mag, edges)
    RT        = real(eltype(velocity_hat))
    result    = Types.SpectralFluxResult(centers, edges, similar(centers, RT, length(centers)),
                                         similar(centers, RT, length(centers)))

    if û_decomp === velocity_hat
        calculate_spectral_flux!(result, ws, velocity_hat, ks, shell_idx;
                                  dealiasing=dealiasing, invariant=invariant, transform=spectral,
                                  execution=execution, advecting_hat=advecting_hat, form=form)
    else
        _flux_term!(ws.nonlinear, velocity_hat, ks, form; dealiasing=dealiasing, transform=spectral,
                    advecting_hat=advecting_hat)
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
    form::Types.AbstractNonlinearForm,
)
    ws = Workspaces.SpectralFluxWorkspace(velocity_hat, ks, binning; geometry=geometry, dealiasing=dealiasing)
    _flux_term!(ws.nonlinear, velocity_hat, ks, form; dealiasing=dealiasing, transform=spectral,
                advecting_hat=advecting_hat)
    N̂ = ws.nonlinear.N̂

    k_mag     = ShellBinning.shell_coordinate(geometry, ks)
    edges     = ShellBinning.shell_edges(binning, maximum(k_mag))
    centers   = ShellBinning.shell_centers(binning, maximum(k_mag))
    shell_idx = ShellBinning.assign_shells(k_mag, edges)
    RT        = real(eltype(velocity_hat))

    return map(decomposed) do û_comp
        res = Types.SpectralFluxResult(centers, edges, similar(centers, RT, length(centers)),
                                       similar(centers, RT, length(centers)))
        _calculate_spectral_flux_with_N̂!(res, ws, û_comp, N̂, ks, shell_idx, execution; invariant=invariant)
        return res
    end
end

"""
    calculate_spectral_flux!(result, ws, velocity_hat, ks, shell_idx;
        dealiasing, invariant, transform, execution, advecting_hat, form)

In-place version of `calculate_spectral_flux`. Writes into `result` using
preallocated buffers from `ws` and a precomputed `shell_idx` array (from `assign_shells`).
Zero heap allocations in the serial hot path. `execution` selects the mode→shell reduction
backend and `form` the nonlinear term (see [`calculate_spectral_flux`](@ref)).
"""
function calculate_spectral_flux!(
    result::Types.SpectralFluxResult,
    ws::Workspaces.SpectralFluxWorkspace,
    velocity_hat,
    ks,
    shell_idx::AbstractArray{Int};
    dealiasing::Types.AbstractDealiasing = Types.OrszagTwoThirds(),
    invariant::Types.AbstractInvariant = Types.KineticEnergy(),
    transform::SpectralBackends.AbstractSpectralBackend = SpectralBackends.AutoSpectralBackend(),
    execution::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.SerialBackend(),
    advecting_hat = velocity_hat,
    form::Types.AbstractNonlinearForm = Types.Advective(),
)
    _check_form(form, velocity_hat, ks, invariant, advecting_hat)
    _flux_term!(ws.nonlinear, velocity_hat, ks, form; dealiasing=dealiasing, transform=transform,
                advecting_hat=advecting_hat)
    _calculate_spectral_flux_with_N̂!(result, ws, velocity_hat, ws.nonlinear.N̂, ks, shell_idx, Types.resolve_execution(execution); invariant=invariant)
    return result
end

"""
    calculate_spectral_flux_batch(velocity_hats, ks; binning, dealiasing, invariant, transform, geometry, execution,
        form) -> Vector{SpectralFluxResult}

Spectral flux for a batch of snapshots that share one grid (`velocity_hats` an iterable of `(ms..., D)`
coefficient arrays). The shell structure is built once, and each worker reuses one
`SpectralFluxWorkspace` (FFT plans + scratch) across its snapshots. `execution = ThreadedBackend()`
(requires `using OhMyThreads`) splits the snapshots across threads, each running single-threaded
transforms. Results are returned in input order and are bit-identical to calling
[`calculate_spectral_flux`](@ref) per snapshot.
"""
function calculate_spectral_flux_batch(
    velocity_hats,
    ks;
    binning::Types.AbstractShellBinning = ShellBinning.default_binning(ks),
    dealiasing::Types.AbstractDealiasing = Types.OrszagTwoThirds(),
    invariant::Types.AbstractInvariant = Types.KineticEnergy(),
    transform::SpectralBackends.AbstractSpectralBackend = SpectralBackends.AutoSpectralBackend(),
    geometry::Types.AbstractShellGeometry = Types.IsotropicShells(),
    execution::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.SerialBackend(),
    form::Types.AbstractNonlinearForm = Types.Advective(),
)
    transform = Types.resolve_transform(Types.require_coefficient_transform(transform))
    n = length(velocity_hats)
    k_mag     = ShellBinning.shell_coordinate(geometry, ks)
    edges     = ShellBinning.shell_edges(binning, maximum(k_mag))
    centers   = ShellBinning.shell_centers(binning, maximum(k_mag))
    RT        = real(eltype(eltype(velocity_hats)))
    n == 0 && return Types.SpectralFluxResult{typeof(centers), typeof(edges), typeof(similar(centers, RT, 0))}[]
    shell_idx = ShellBinning.assign_shells(k_mag, edges)
    results   = [Types.SpectralFluxResult(centers, edges, similar(centers, RT, length(centers)),
                                          similar(centers, RT, length(centers)))
                 for _ in 1:n]
    _spectral_flux_batch!(Types.resolve_execution(execution), results, velocity_hats, ks, shell_idx;
                          binning=binning, dealiasing=dealiasing, invariant=invariant, transform=transform, geometry=geometry,
                          form=form)
    return results
end

# Serial reference: one workspace reused across the whole batch.
function _spectral_flux_batch!(::ComputationalBackends.AbstractSerialBackend, results, velocity_hats, ks, shell_idx;
                               binning, dealiasing, invariant, transform, geometry, form)
    ws = Workspaces.SpectralFluxWorkspace(first(velocity_hats), ks, binning; geometry=geometry, dealiasing=dealiasing)
    for i in eachindex(velocity_hats)
        calculate_spectral_flux!(results[i], ws, velocity_hats[i], ks, shell_idx;
                                 dealiasing=dealiasing, invariant=invariant, transform=transform, form=form,
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

# GPU over the batch: one workspace and one copy of the shell index in the device's memory, reused for
# every snapshot; host snapshots are copied into the staging buffer.
function _spectral_flux_batch!(gpu::ComputationalBackends.AbstractGPUBackend, results, velocity_hats, ks, shell_idx;
                               binning, dealiasing, invariant, transform, geometry, form)
    buf = Types._staging_buffer(gpu, first(velocity_hats))
    proto = buf === nothing ? first(velocity_hats) : buf
    ws = Workspaces.SpectralFluxWorkspace(proto, ks, binning; geometry=geometry, dealiasing=dealiasing)
    sidx = copyto!(similar(proto, Int, size(shell_idx)), shell_idx)
    for i in eachindex(velocity_hats)
        calculate_spectral_flux!(results[i], ws, Types._staged!(buf, velocity_hats[i]), ks, sidx;
                                 dealiasing=dealiasing, invariant=invariant, transform=transform, form=form,
                                 execution=gpu)
    end
    return results
end

# `DistributedBackend` and `MPIBackend` batches are methods of the Distributed and MPI extensions.
_spectral_flux_batch!(be::ComputationalBackends.AbstractExecutionBackend, results, velocity_hats, ks, shell_idx; kwargs...) =
    Types.batch_unavailable("calculate_spectral_flux_batch", be)

# ---------------------------------------------------------------------------
# Transfer-density write + mode→shell reduction, dispatched on the execution backend.
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
    Invariants.transfer_density_scatter!(ws.T_spec, invariant, velocity_hat, N̂, ks, shell_idx)
    return _finalize_spectral_flux!(result, ws)
end

# Copy T(k) out of `ws.T_spec` and accumulate Π(K); every backend ends here. Convention (Alexakis &
# Biferale 2018, Eqs. 12–17):
#   T(k) = Re{û*·N̂} = −dE(k)/dt from the nonlinear term, and
#   Π(K) = Σ_{k≤K} T(k)  ⇒  Π>0 forward (down-scale) cascade, Π<0 inverse.
function _finalize_spectral_flux!(result::Types.SpectralFluxResult, ws::Workspaces.SpectralFluxWorkspace)
    # The cumsum runs on the host result vectors: Base's `cumsum!` scalar-indexes non-CUDA device arrays.
    copyto!(result.transfer_spectrum, ws.T_spec)
    cumsum!(result.flux, result.transfer_spectrum)
    return result
end

# The parallel backends dispatch to named stubs that throw until the matching extension
# (OhMyThreads, Distributed, KernelAbstractions) defines them.
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
    calculate_scalar_flux(velocity_hat, scalar_hat, ks; binning, dealiasing=OrszagTwoThirds(), transform,
                          execution=SerialBackend(), geometry=IsotropicShells()) -> SpectralFluxResult

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
    binning::Types.AbstractShellBinning = ShellBinning.default_binning(ks),
    dealiasing::Types.AbstractDealiasing = Types.OrszagTwoThirds(),
    transform::SpectralBackends.AbstractSpectralBackend = SpectralBackends.AutoSpectralBackend(),
    execution::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.SerialBackend(),
    geometry::Types.AbstractShellGeometry = Types.IsotropicShells(),
)
    θ̂ = Utils.as_component_field(scalar_hat, length(ks))
    return calculate_spectral_flux(θ̂, ks; binning=binning, dealiasing=dealiasing,
        invariant=Types.PassiveScalar(), advecting_hat=velocity_hat, transform=transform,
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
        dealiasing=OrszagTwoThirds(), transform=AutoSpectralBackend(), execution=SerialBackend(),
        geometry=IsotropicShells())
        -> (channels::Dict{NTuple{3,Symbol},SpectralFluxResult}, total::SpectralFluxResult, k_shells)

Decompose the kinetic-energy flux into **per-component partial fluxes** `Π^{s_k s_p s_q}(K)`,
where each of the three fields in a triad interaction is one component of a velocity decomposition
`u = Σ_s u_s`; `channels` is keyed by the component-name triple `(s_k, s_p, s_q)` and `total` is their
sum. The channel term depends on the decomposition:

  * [`HelicalDecomposition`](@ref) (3-D) — the rotational form of Alexakis (2017, Eq. 17),
    `Π^{s₁s₂s₃}(K) = −⟨u^{s₁}_{K<}·(u^{s₂} × ω^{s₃})⟩`:

        T^{s_k s_p s_q}(k) = Re{ û_{s_k}*(k) · [ω_{s_q} × u_{s_p}]^(k) },   ω_s = ∇ × u_s.

    `s_k` is the component whose energy the term changes, `s_p` the velocity of the cross product
    and `s_q` the field whose vorticity it carries. The helical components are solenoidal, so the
    channels sum to the flux of the solenoidal part of `u`.
  * every other decomposition — the advective form

        T^{s_k s_p s_q}(k) = Re{ û_{s_k}*(k) · [(u_{s_p}·∇) u_{s_q}](k) },

    whose channels sum to the full flux. For [`HelmholtzDecomposition`](@ref) the off-diagonal
    channels are the rotational↔divergent cross-flux, zero for incompressible flow.

`Π = Σ_{k≤K} T` in each channel. The decomposition and the channel terms run on every
`transform`, `execution` and `dealiasing` the nonlinear term does; under `execution = GPUBackend(dev)`
a host field is copied to `dev` once per call.
"""
function calculate_partial_fluxes(velocity_hat, ks;
                                  dealiasing::Types.AbstractDealiasing = Types.OrszagTwoThirds(),
                                  execution::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.SerialBackend(),
                                  kwargs...)
    exec = Types.resolve_execution(execution)
    velocity_hat = Types._to_execution(exec, velocity_hat)
    ws = Workspaces.NonlinearTermWorkspace(velocity_hat, ks; dealiasing = dealiasing)
    return calculate_partial_fluxes!(ws, velocity_hat, ks; dealiasing = dealiasing, execution = exec, kwargs...)
end

"""
    calculate_partial_fluxes_batch(velocity_hats, ks; execution, kwargs...) -> Vector

Decomposition-channel fluxes ([`calculate_partial_fluxes`](@ref)) for a batch of snapshots sharing
one grid. One nonlinear-term workspace is reused across each worker's snapshots.
`execution = ThreadedBackend()` (requires `using OhMyThreads`) threads over snapshots with a serial
inner transform. Results are in input order.
"""
function calculate_partial_fluxes_batch(
    velocity_hats, ks;
    execution::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.SerialBackend(),
    dealiasing::Types.AbstractDealiasing = Types.OrszagTwoThirds(),
    binning::Types.AbstractShellBinning = ShellBinning.default_binning(ks),
    geometry::Types.AbstractShellGeometry = Types.IsotropicShells(),
    kwargs...,
)
    if length(velocity_hats) == 0
        kmax = ShellBinning.max_shell_coordinate(geometry, ks)
        centers = collect(ShellBinning.shell_centers(binning, kmax))
        FT = real(eltype(eltype(velocity_hats)))
        return Vector{_partial_result_type(FT, centers, ShellBinning.shell_edges(binning, kmax))}()
    end
    return _partial_fluxes_batch!(Types.resolve_execution(execution), velocity_hats, ks;
                                  dealiasing = dealiasing, binning = binning, geometry = geometry, kwargs...)
end

function _partial_fluxes_batch!(::ComputationalBackends.AbstractSerialBackend, velocity_hats, ks;
                                dealiasing, kwargs...)
    ws = Workspaces.NonlinearTermWorkspace(first(velocity_hats), ks; dealiasing = dealiasing)
    return [calculate_partial_fluxes!(ws, vh, ks; dealiasing = dealiasing, kwargs...) for vh in velocity_hats]
end

# A device batch: one workspace in the device's memory, each snapshot's channels on the device; host
# snapshots are copied into the staging buffer.
function _partial_fluxes_batch!(gpu::ComputationalBackends.AbstractGPUBackend, velocity_hats, ks;
                                dealiasing, kwargs...)
    buf = Types._staging_buffer(gpu, first(velocity_hats))
    ws = Workspaces.NonlinearTermWorkspace(buf === nothing ? first(velocity_hats) : buf, ks; dealiasing = dealiasing)
    return [calculate_partial_fluxes!(ws, Types._staged!(buf, vh), ks; dealiasing = dealiasing, execution = gpu,
                                      kwargs...)
            for vh in velocity_hats]
end

# The result types of the partial fluxes on shells with these centres and edges.
_channel_result_type(::Type{FT}, centers, edges) where {FT} =
    Types.SpectralFluxResult{typeof(centers), typeof(edges), Vector{FT}}
_partial_result_type(::Type{FT}, centers, edges) where {FT} =
    NamedTuple{(:channels, :total, :k_shells),
               Tuple{Dict{NTuple{3,Symbol}, _channel_result_type(FT, centers, edges)},
                     _channel_result_type(FT, centers, edges), typeof(centers)}}

# The nonlinear term of the `(s_p, s_q)` channels, into `ws.N̂`.
_channel_term!(ws, ::Types.HelicalDecomposition, u_p, u_q, ks; dealiasing, transform) =
    NonlinearTerm.compute_rotational_term!(ws, u_p, u_q, ks; dealiasing = dealiasing, transform = transform)
_channel_term!(ws, ::Types.AbstractFieldDecomposition, u_p, u_q, ks; dealiasing, transform) =
    NonlinearTerm.compute_nonlinear_term!(ws, u_q, ks; advecting_hat = u_p, dealiasing = dealiasing,
                                          transform = transform)

function _partial_fluxes_batch_threaded!(args...; kwargs...)
    throw(ArgumentError("execution = ThreadedBackend() for the partial-flux batch requires OhMyThreads. " *
                        "Run `using OhMyThreads` to load the extension."))
end
_partial_fluxes_batch!(::ComputationalBackends.AbstractThreadedBackend, velocity_hats, ks; kwargs...) =
    _partial_fluxes_batch_threaded!(velocity_hats, ks; kwargs...)

_partial_fluxes_batch!(be::ComputationalBackends.AbstractExecutionBackend, velocity_hats, ks; kwargs...) =
    Types.batch_unavailable("calculate_partial_fluxes_batch", be)

"""
    calculate_partial_fluxes!(ws::NonlinearTermWorkspace, velocity_hat, ks; kwargs...)

The partial fluxes of [`calculate_partial_fluxes`](@ref), reusing the caller's `ws` for every
decomposition-channel pair; the channel `SpectralFluxResult`s and their total are allocated per call.
"""
function calculate_partial_fluxes!(
    ws::Workspaces.NonlinearTermWorkspace,
    velocity_hat,
    ks;
    decomposition::Types.AbstractFieldDecomposition = Types.HelicalDecomposition(),
    binning::Types.AbstractShellBinning = ShellBinning.default_binning(ks),
    dealiasing::Types.AbstractDealiasing = Types.OrszagTwoThirds(),
    transform::SpectralBackends.AbstractSpectralBackend = SpectralBackends.AutoSpectralBackend(),
    execution::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.SerialBackend(),
    geometry::Types.AbstractShellGeometry = Types.IsotropicShells(),
)
    transform = Types.resolve_transform(Types.require_coefficient_transform(transform))
    comps = Decomposition.decompose_field(decomposition, velocity_hat, ks)
    comps isa NamedTuple || throw(ArgumentError(
        "calculate_partial_fluxes needs a decomposition that splits u into ≥2 named components " *
        "(e.g. Types.HelicalDecomposition or Types.HelmholtzDecomposition); got $(typeof(decomposition))."))
    names = keys(comps)

    k_coord = ShellBinning.shell_coordinate(geometry, ks)
    edges   = ShellBinning.shell_edges(binning, maximum(k_coord))
    centers = collect(ShellBinning.shell_centers(binning, maximum(k_coord)))
    shells  = (sidx = ShellBinning.assign_shells(k_coord, edges), centers = centers, edges = edges,
               Nsh = length(centers))

    FT = real(eltype(velocity_hat))
    channels = Dict{NTuple{3,Symbol}, _channel_result_type(FT, centers, edges)}()
    # Each (sp, sq) pair is an independent nonlinear-term build; the parallel backends split the n²
    # pairs, each worker with single-threaded FFTs.
    _partial_fluxes_fill!(Types.resolve_execution(execution), channels, ws, comps, names,
        velocity_hat, ks, shells; dealiasing = dealiasing, transform = transform, decomposition = decomposition)
    total = Types.SpectralFluxResult(centers, edges,
        sum(c.transfer_spectrum for c in values(channels)),
        sum(c.flux for c in values(channels)))
    return (channels = channels, total = total, k_shells = centers)
end

# Bin a transfer density into the shells `(; sidx, centers, edges, Nsh)` → a channel result.
function _partial_binflux(td, ks, shells)
    FT = eltype(td)
    T = zeros(FT, shells.Nsh)
    @inbounds for I in CartesianIndices(td)
        n = shells.sidx[I]; n == 0 && continue
        T[n] += FTB.mode_fold(ks, I) * td[I]
    end
    return Types.SpectralFluxResult(shells.centers, shells.edges, T, cumsum(T))
end

# Serial channel fill: reuse the caller ws + one td scratch across every (sp,sq) pair.
function _partial_fluxes_fill!(::ComputationalBackends.SerialBackend, channels, ws, comps, names, velocity_hat, ks, shells;
                               dealiasing, transform, decomposition)
    FT = real(eltype(velocity_hat)); nd = length(ks); ns = size(velocity_hat)[1:nd]
    td = similar(velocity_hat, FT, ns...)
    for sp in names, sq in names
        _channel_term!(ws, decomposition, comps[sp], comps[sq], ks; dealiasing = dealiasing, transform = transform)
        for sk in names
            Invariants.transfer_density!(td, Types.KineticEnergy(), comps[sk], ws.N̂, ks)
            channels[(sk, sp, sq)] = _partial_binflux(td, ks, shells)
        end
    end
    return channels
end
_partial_fluxes_fill!(::ComputationalBackends.ThreadedBackend, channels, ws, comps, names, velocity_hat, ks, shells; kwargs...) =
    _partial_fluxes_threaded!(channels, ws, comps, names, velocity_hat, ks, shells; kwargs...)
_partial_fluxes_fill!(exec::ComputationalBackends.DistributedBackend, channels, ws, comps, names, velocity_hat, ks, shells; kwargs...) =
    _partial_fluxes_distributed!(channels, ws, comps, names, velocity_hat, ks, shells, exec; kwargs...)
_partial_fluxes_fill!(gpu::ComputationalBackends.GPUBackend, channels, ws, comps, names, velocity_hat, ks, shells; kwargs...) =
    _partial_fluxes_gpu!(channels, ws, comps, names, velocity_hat, ks, shells, gpu; kwargs...)

_partial_fluxes_threaded!(args...; kwargs...) = throw(ArgumentError(
    "Threaded partial fluxes require OhMyThreads. Run `using OhMyThreads` to load the extension."))
_partial_fluxes_distributed!(args...; kwargs...) = throw(ArgumentError(
    "Distributed partial fluxes require Distributed. Run `using Distributed` to load the extension."))
_partial_fluxes_gpu!(args...; kwargs...) = throw(ArgumentError(
    "GPU partial fluxes require KernelAbstractions. Run `using KernelAbstractions` to load the extension."))

"""
    calculate_helical_partial_fluxes(velocity_hat, ks; kwargs...)

The eight **helical** partial energy fluxes `Π^{s_k s_p s_q}(K)`, `s ∈ {positive, negative}`, in the
rotational form of Alexakis (2017, Eq. 17) — [`calculate_partial_fluxes`](@ref) with
`decomposition = HelicalDecomposition()` (3D only). Keys are `(s_k, s_p, s_q)` with `s` the symbols
`:positive`/`:negative`: `s_k` is the component whose energy the term changes, `s_p` the velocity and
`s_q` the vorticity of `u × ω`.
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

end # module SpectralFlux
