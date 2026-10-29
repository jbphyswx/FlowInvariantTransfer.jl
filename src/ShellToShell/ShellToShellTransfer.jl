module ShellToShellTransfer

using ..Types: Types
using ComputationalBackends: ComputationalBackends
using SpectralBackends: SpectralBackends
using ..Invariants: Invariants
using ..ShellBinning: ShellBinning
using ..Utils: Utils
using ..NonlinearTerm: NonlinearTerm
using ..Workspaces: Workspaces

export calculate_shell_to_shell_transfer, calculate_shell_to_shell_transfer!, calculate_shell_to_shell_transfer_batch,
       calculate_scalar_shell_to_shell_transfer, calculate_scalar_shell_to_shell_transfer!

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
        binning, dealiasing=OrszagTwoThirds(), invariant=KineticEnergy(),
        transform=SpectralBackends.AutoSpectralBackend(), execution=ComputationalBackends.SerialBackend(),
        advecting_hat=velocity_hat, geometry=IsotropicShells())
        -> ShellToShellResult

Compute the directed shell-to-shell kinetic energy transfer matrix T(n,m).

# Arguments
- `velocity_hat`: Complex array of size `(ms..., D)` — Fourier coefficients of
  the D velocity components on a periodic uniform grid.
- `ks`: Tuple of `nd` 1D physical-wavenumber vectors.

# Keyword Arguments
- `binning::AbstractShellBinning`: Shell binning; default `LinearBinning(Δk)` with `Δk` the smallest
  nonzero `|k|` on any axis.
- `dealiasing::AbstractDealiasing=OrszagTwoThirds()`: Apply 2/3 rule dealiasing.
- `transform::SpectralBackends.AbstractSpectralBackend`: `SpectralBackends.AutoSpectralBackend()`
  (default; `FFTSpectralBackend()` when FFTW is loaded), `SpectralBackends.FFTSpectralBackend()`, or
  `SpectralBackends.DirectSumSpectralBackend()`.
- `execution::ComputationalBackends.AbstractExecutionBackend`: outer (mediator-loop) parallelism — `ComputationalBackends.AutoBackend()` (default),
  `ComputationalBackends.ThreadedBackend()` (OhMyThreads), `ComputationalBackends.DistributedBackend()`, or `ComputationalBackends.GPUBackend(dev)`,
  which copies host fields to `dev` once per call.
- `invariant`, `advecting_hat`, `geometry`: as for
  [`calculate_spectral_flux`](@ref FlowInvariantTransfer.SpectralFlux.calculate_spectral_flux).

# Returns
`ShellToShellResult` with:
- `transfer_matrix[n,m]`: the rate at which shell n gives energy to shell m.
- `net_transfer[n]` = Σ_m T(n,m) = −dE_n/dt, the spectral transfer of shell n.

# Algorithm (Alexakis–Mininni–Pouquet 2005 formulation)
For each pair of shells n and m:
  T(n,m) = Σ_{k∈S_n} Re{ û*(k) · N̂_m(k) }
where N̂_m(k) = FFT[(u · ∇)u_m], u_m = IFFT(û · χ_m): the full velocity advects the shell-m field.
For a divergence-free u, ∫u_n·(u·∇)u_m = −∫u_m·(u·∇)u_n, so
  T(n,m) + T(m,n) = 0  exactly (antisymmetry).

# Cost
One advection per mediator shell: O(N_shells · Nᴰ log N) with FFTW, O(N_shells · N²ᴰ) by direct sums.

# References
- Verma et al. (2002), arXiv:nlin/0204027
- Alexakis, Mininni & Pouquet (2005)
"""
function calculate_shell_to_shell_transfer(
    velocity_hat,
    ks;
    binning::Types.AbstractShellBinning = ShellBinning.default_binning(ks),
    dealiasing::Types.AbstractDealiasing = Types.OrszagTwoThirds(),
    invariant::Types.AbstractInvariant = Types.KineticEnergy(),
    transform::SpectralBackends.AbstractSpectralBackend = SpectralBackends.AutoSpectralBackend(),
    execution::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend(),
    advecting_hat = velocity_hat,
    geometry::Types.AbstractShellGeometry = Types.IsotropicShells(),
)
    transform = Types.resolve_transform(Types.require_coefficient_transform(transform))
    exec    = Types.resolve_execution(execution, velocity_hat)
    velocity_hat, advecting_hat = Types._fields_to_execution(exec, velocity_hat, advecting_hat)
    ws      = Workspaces.ShellToShellWorkspace(velocity_hat, ks, binning; geometry=geometry, dealiasing=dealiasing)
    k_mag   = ShellBinning.shell_coordinate(geometry, ks)
    edges   = ShellBinning.shell_edges(binning, maximum(k_mag))
    centers = ShellBinning.shell_centers(binning, maximum(k_mag))
    N_sh    = length(centers)
    FT      = real(eltype(velocity_hat))
    result  = Types.ShellToShellResult(centers, edges, Matrix{FT}(undef, N_sh, N_sh), Vector{FT}(undef, N_sh))
    _calculate_shell_to_shell!(result, ws, velocity_hat, ks, exec, transform; dealiasing=dealiasing,
        invariant=invariant, advecting_hat=advecting_hat)
    return result
end

"""
    calculate_shell_to_shell_transfer!(result, ws, velocity_hat, ks; kwargs...)

In-place [`calculate_shell_to_shell_transfer`](@ref), writing `result.transfer_matrix` and
`result.net_transfer` with the buffers of `ws`; zero heap allocations in the hot path.
"""
function calculate_shell_to_shell_transfer!(
    result::Types.ShellToShellResult,
    ws::Workspaces.ShellToShellWorkspace,
    velocity_hat,
    ks;
    dealiasing::Types.AbstractDealiasing = Types.OrszagTwoThirds(),
    invariant::Types.AbstractInvariant = Types.KineticEnergy(),
    transform::SpectralBackends.AbstractSpectralBackend = SpectralBackends.AutoSpectralBackend(),
    execution::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend(),
    advecting_hat = velocity_hat,
)
    transform = Types.resolve_transform(Types.require_coefficient_transform(transform))
    _calculate_shell_to_shell!(result, ws, velocity_hat, ks,
        Types.resolve_execution(execution, velocity_hat), transform; dealiasing=dealiasing,
        invariant=invariant, advecting_hat=advecting_hat)
    return result
end

"""
    calculate_shell_to_shell_transfer_batch(velocity_hats, ks; binning, dealiasing,
        invariant, transform, geometry, execution) -> Vector{ShellToShellResult}

Shell-to-shell transfer for a batch of snapshots that share one grid (`velocity_hats` an iterable of
`(ms..., D)` coefficient arrays). The shell structure is built once and each worker reuses one
`ShellToShellWorkspace` across its snapshots; `execution = ThreadedBackend()` (requires
`using OhMyThreads`) threads over snapshots, each with a serial inner transform. Results are in input
order, bit-identical to calling [`calculate_shell_to_shell_transfer`](@ref) per snapshot (each
snapshot self-advects).
"""
function calculate_shell_to_shell_transfer_batch(
    velocity_hats,
    ks;
    binning::Types.AbstractShellBinning = ShellBinning.default_binning(ks),
    dealiasing::Types.AbstractDealiasing = Types.OrszagTwoThirds(),
    invariant::Types.AbstractInvariant = Types.KineticEnergy(),
    transform::SpectralBackends.AbstractSpectralBackend = SpectralBackends.AutoSpectralBackend(),
    geometry::Types.AbstractShellGeometry = Types.IsotropicShells(),
    execution::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend(),
)
    transform = Types.resolve_transform(Types.require_coefficient_transform(transform))
    n = length(velocity_hats)
    k_mag   = ShellBinning.shell_coordinate(geometry, ks)
    edges   = ShellBinning.shell_edges(binning, maximum(k_mag))
    centers = ShellBinning.shell_centers(binning, maximum(k_mag))
    N_sh    = length(centers)
    FT      = real(eltype(eltype(velocity_hats)))
    results = Vector{Types.ShellToShellResult{typeof(centers), Vector{FT}, Matrix{FT}}}(undef, n)
    n == 0 && return results
    _shell_to_shell_batch!(Types.resolve_execution(execution, first(velocity_hats)), results, velocity_hats, ks, centers, edges, N_sh;
        binning=binning, dealiasing=dealiasing, invariant=invariant, transform=transform, geometry=geometry)
    return results
end

# Serial reference: one workspace reused across the whole batch; each snapshot self-advects.
function _shell_to_shell_batch!(::ComputationalBackends.AbstractSerialBackend, results, velocity_hats, ks, centers, edges, N_sh;
                                binning, dealiasing, invariant, transform, geometry)
    FT = real(eltype(first(velocity_hats)))
    ws = Workspaces.ShellToShellWorkspace(first(velocity_hats), ks, binning; geometry=geometry, dealiasing=dealiasing)
    for i in eachindex(velocity_hats)
        vh = velocity_hats[i]
        res = Types.ShellToShellResult(centers, edges, Matrix{FT}(undef, N_sh, N_sh), Vector{FT}(undef, N_sh))
        _calculate_shell_to_shell!(res, ws, vh, ks, ComputationalBackends.SerialBackend(), transform;
            dealiasing=dealiasing, invariant=invariant, advecting_hat=vh)
        results[i] = res
    end
    return results
end

# Threaded over the batch — overridden by the OhMyThreads extension (per-chunk workspace pool, serial inner).
function _shell_to_shell_batch_threaded!(args...; kwargs...)
    throw(ArgumentError("execution = ThreadedBackend() for the shell-to-shell batch requires OhMyThreads. " *
                        "Run `using OhMyThreads` to load the extension."))
end
_shell_to_shell_batch!(::ComputationalBackends.AbstractThreadedBackend, results, velocity_hats, ks, centers, edges, N_sh; kwargs...) =
    _shell_to_shell_batch_threaded!(results, velocity_hats, ks, centers, edges, N_sh; kwargs...)

# GPU over the batch: one workspace in the device's memory reused for every snapshot; host snapshots are
# copied into the staging buffer.
function _shell_to_shell_batch!(gpu::ComputationalBackends.AbstractGPUBackend, results, velocity_hats, ks, centers, edges, N_sh;
                                binning, dealiasing, invariant, transform, geometry)
    FT = real(eltype(first(velocity_hats)))
    buf = Types._staging_buffer(gpu, first(velocity_hats))
    ws = Workspaces.ShellToShellWorkspace(buf === nothing ? first(velocity_hats) : buf, ks, binning;
                                          geometry=geometry, dealiasing=dealiasing)
    for i in eachindex(velocity_hats)
        vh = Types._staged!(buf, velocity_hats[i])
        res = Types.ShellToShellResult(centers, edges, Matrix{FT}(undef, N_sh, N_sh), Vector{FT}(undef, N_sh))
        _calculate_shell_to_shell!(res, ws, vh, ks, gpu, transform;
            dealiasing=dealiasing, invariant=invariant, advecting_hat=vh)
        results[i] = res
    end
    return results
end

# `DistributedBackend` and `MPIBackend` batches are methods of the Distributed and MPI extensions.
_shell_to_shell_batch!(be::ComputationalBackends.AbstractExecutionBackend, results, velocity_hats, ks, centers, edges, N_sh; kwargs...) =
    Types.batch_unavailable("calculate_shell_to_shell_transfer_batch", be)

"""
    calculate_scalar_shell_to_shell_transfer(velocity_hat, scalar_hat, ks; kwargs...) -> ShellToShellResult

Shell-to-shell transfer of passive-scalar **variance** `T_θ(n,m)`: the rate at which scalar-shell
`n` gives variance to scalar-shell `m`, mediated by the velocity:

    T_θ(n,m) = Σ_{k∈S_n} Re{ θ̂*(k) · 𝒩̂_m(k) },   𝒩̂_m = FFT[(u·∇)θ_m],   θ_m = θ̂·χ_m.

The scalar field is band-filtered and carried; the velocity advects it. Thin wrapper over
[`calculate_shell_to_shell_transfer`](@ref) with `invariant = PassiveScalar()` and
`advecting_hat = velocity_hat`. `scalar_hat` may be `(ns...)` or `(ns..., 1)`. As with energy,
`T_θ(n,m)` is antisymmetric for incompressible `u` and reduces to `T_θ(k)` over mediators.
"""
function calculate_scalar_shell_to_shell_transfer(velocity_hat, scalar_hat, ks; kwargs...)
    θ̂ = Utils.as_component_field(scalar_hat, length(ks))
    return calculate_shell_to_shell_transfer(θ̂, ks;
        invariant=Types.PassiveScalar(), advecting_hat=velocity_hat, kwargs...)
end

"""
    calculate_scalar_shell_to_shell_transfer!(result, ws, velocity_hat, scalar_hat, ks; kwargs...)

In-place passive-scalar shell-to-shell variance transfer — thin wrapper over
[`calculate_shell_to_shell_transfer!`](@ref) (`invariant = PassiveScalar()`), writing into the
caller-provided `result`/`ws` (0 alloc beyond them; `ws` sized for the scalar field).
"""
function calculate_scalar_shell_to_shell_transfer!(result, ws, velocity_hat, scalar_hat, ks; kwargs...)
    θ̂ = Utils.as_component_field(scalar_hat, length(ks))
    return calculate_shell_to_shell_transfer!(result, ws, θ̂, ks;
        invariant=Types.PassiveScalar(), advecting_hat=velocity_hat, kwargs...)
end

# Dispatch on (execution, transform). The threaded, distributed and GPU methods (extensions)
# parallelise the mediator loop and pass the transform to each per-mediator nonlinear term.
_calculate_shell_to_shell!(result, ws, velocity_hat, ks, ::ComputationalBackends.SerialBackend, spectral::SpectralBackends.DirectSumSpectralBackend; kwargs...) =
    _calculate_shell_to_shell_direct!(result, ws, velocity_hat, ks; kwargs...)

_calculate_shell_to_shell!(result, ws, velocity_hat, ks, ::ComputationalBackends.SerialBackend, spectral::SpectralBackends.FFTSpectralBackend; kwargs...) =
    _shell_to_shell_fft!(result, ws, velocity_hat, ks; kwargs...)

_calculate_shell_to_shell!(result, ws, velocity_hat, ks, ::ComputationalBackends.ThreadedBackend, spectral::SpectralBackends.AbstractSpectralBackend; kwargs...) =
    _shell_to_shell_threaded!(result, ws, velocity_hat, ks, spectral; kwargs...)

# Distributed / GPU: named stubs that throw until the Distributed / KernelAbstractions extension
# defines them.
_calculate_shell_to_shell!(result, ws, velocity_hat, ks, execution::ComputationalBackends.DistributedBackend, spectral::SpectralBackends.AbstractSpectralBackend; kwargs...) =
    _shell_to_shell_distributed!(result, ws, velocity_hat, ks, execution, spectral; kwargs...)
_calculate_shell_to_shell!(result, ws, velocity_hat, ks, gpu::ComputationalBackends.GPUBackend, spectral::SpectralBackends.AbstractSpectralBackend; kwargs...) =
    _shell_to_shell_gpu!(result, ws, velocity_hat, ks, gpu, spectral; kwargs...)

_shell_to_shell_distributed!(args...; kwargs...) = throw(ArgumentError(
    "Distributed shell-to-shell transfer requires Distributed. Run `using Distributed` to load the extension."))
_shell_to_shell_gpu!(args...; kwargs...) = throw(ArgumentError(
    "GPU shell-to-shell transfer requires KernelAbstractions. Run `using KernelAbstractions` to load the extension."))

# ---------------------------------------------------------------------------
# Direct reference implementation
# ---------------------------------------------------------------------------

"""
    _calculate_shell_to_shell_direct!(result, ws, velocity_hat, ks; dealiasing, invariant, advecting_hat) -> result

Direct-sum (ComputationalBackends.SerialBackend) shell-to-shell transfer. Writes into `result` using
workspace buffers from `ws` — no heap allocation in the hot path.

For each mediator shell m:
  1. Build û_m = û restricted to shell m (using ws.shell_idx)
  2. Compute N̂_m = FFT[(u·∇)u_m] using ws.nonlinear buffers
  3. Accumulate T(n,m) for every shell n
"""
function _calculate_shell_to_shell_direct!(
    result::Types.ShellToShellResult,
    ws::Workspaces.ShellToShellWorkspace,
    velocity_hat,
    ks;
    dealiasing::Types.AbstractDealiasing,
    invariant::Types.AbstractInvariant = Types.KineticEnergy(),
    advecting_hat = velocity_hat,
)
    nd    = length(ks)
    ns    = Base.front(size(velocity_hat))
    M     = size(velocity_hat, nd+1)   # components of the binned/carried primary field
    FT    = real(eltype(velocity_hat))
    N_sh  = size(result.transfer_matrix, 1)

    fill!(result.transfer_matrix, zero(FT))
    col = ws.net_transfer            # length-N_sh scratch, rewritten per mediator then copied out
    direct = SpectralBackends.DirectSumSpectralBackend()
    NonlinearTerm.synthesize_advecting!(ws.nonlinear, advecting_hat, ks; dealiasing = dealiasing, transform = direct)

    for m in 1:N_sh
        # Build û_m: velocity restricted to shell m — reuse ws.û_m
        fill!(ws.û_m, zero(eltype(ws.û_m)))
        for I in CartesianIndices(ns)
            ws.shell_idx[I] == m || continue
            for comp in 1:M
                ws.û_m[I, comp] = velocity_hat[I, comp]
            end
        end

        # N̂_m = (u·∇)u_m: the full velocity advects the band-m field (Alexakis–Mininni–Pouquet
        # 2005), so A[n,m] = Σ_{k∈S_n} Re{û*·N̂_m} is antisymmetric (A[n,m] + A[m,n] = 0) and
        # Σ_m A[n,m] = transfer_spectrum[n].
        NonlinearTerm.compute_advection!(ws.nonlinear, ws.û_m, ks; dealiasing = dealiasing, transform = direct)
        N̂_m = ws.nonlinear.N̂

        # A(n,m) for every shell n in one pass over the modes, each mode adding to its own shell.
        Invariants.transfer_density_scatter!(col, invariant, velocity_hat, N̂_m, ks, ws.shell_idx)
        @inbounds for n in 1:N_sh
            result.transfer_matrix[n, m] = col[n]
        end
    end

    # net_transfer[n] = Σ_m T(n,m) = −dE_n/dt
    for n in 1:N_sh
        s = zero(FT)
        for m in 1:N_sh
            s += result.transfer_matrix[n, m]
        end
        result.net_transfer[n] = s
    end

    return result
end

end # module ShellToShellTransfer
