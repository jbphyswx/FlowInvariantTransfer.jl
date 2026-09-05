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
        binning, dealiasing=OrszagTwoThirds(), verify_antisymmetry=true,
        spectral=SpectralBackends.DirectSumSpectralBackend(), execution=ComputationalBackends.SerialBackend())
        -> ShellToShellResult

Compute the directed shell-to-shell kinetic energy transfer matrix T(n,m).

# Arguments
- `velocity_hat`: Complex array of size `(ns..., D)` — Fourier coefficients of
  the D velocity components on a periodic uniform grid.
- `ks`: Tuple of D 1D physical-wavenumber vectors.

# Keyword Arguments
- `binning::AbstractShellBinning`: Shell binning; default `LinearBinning(1.0)`.
- `dealiasing::AbstractDealiasing=OrszagTwoThirds()`: Apply 2/3 rule dealiasing.
- `verify_antisymmetry::Bool=true`: Compute `max|T(n,m)+T(m,n)|` and store in result.
- `spectral::SpectralBackends.AbstractSpectralBackend`: transform — `SpectralBackends.DirectSumSpectralBackend()` (default) or `SpectralBackends.FFTSpectralBackend()` (FFTW).
- `execution::ComputationalBackends.AbstractExecutionBackend`: outer (mediator-loop) parallelism — `ComputationalBackends.SerialBackend()` (default),
  `ComputationalBackends.ThreadedBackend()` (OhMyThreads), `ComputationalBackends.DistributedBackend()`, or `ComputationalBackends.GPUBackend(...)`.

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
    ks;
    binning::Types.AbstractShellBinning = _default_binning(ks),
    dealiasing::Types.AbstractDealiasing = Types.OrszagTwoThirds(),
    verify_antisymmetry::Bool = true,
    invariant::Types.AbstractInvariant = Types.KineticEnergy(),
    spectral::SpectralBackends.AbstractSpectralBackend = SpectralBackends.AutoSpectralBackend(),
    execution::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.SerialBackend(),
    advecting_hat = velocity_hat,
    geometry::Types.AbstractShellGeometry = Types.IsotropicShells(),
)
    spectral = Types.resolve_spectral(Types.require_coefficient_spectral(spectral))
    ws      = Workspaces.ShellToShellWorkspace(velocity_hat, ks, binning; geometry=geometry, dealiasing=dealiasing)
    k_mag   = ShellBinning.shell_coordinate(geometry, ks)
    edges   = ShellBinning.shell_edges(binning, maximum(k_mag))
    centers = ShellBinning.shell_centers(binning, maximum(k_mag))
    N_sh    = length(centers)
    FT      = real(eltype(velocity_hat))
    T_mat   = Matrix{FT}(undef, N_sh, N_sh)
    net     = Vector{FT}(undef, N_sh)
    # Use a mutable wrapper so ! variants can write max_asym back
    result_mut = Types.ShellToShellResult(centers, edges, T_mat, net, FT(NaN))
    max_asym = _calculate_shell_to_shell!(result_mut, ws, velocity_hat, ks, Types.resolve_execution(execution), spectral;
        dealiasing=dealiasing, verify_antisymmetry=verify_antisymmetry, invariant=invariant,
        advecting_hat=advecting_hat)
    return Types.ShellToShellResult(centers, edges, T_mat, net, max_asym)
end

"""
    calculate_shell_to_shell_transfer!(result, ws, velocity_hat, ks; kwargs...)

In-place version. Writes into `result` using preallocated buffers from `ws`.
Zero heap allocations in the hot path.
"""
function calculate_shell_to_shell_transfer!(
    result::Types.ShellToShellResult,
    ws::Workspaces.ShellToShellWorkspace,
    velocity_hat,
    ks;
    dealiasing::Types.AbstractDealiasing = Types.OrszagTwoThirds(),
    verify_antisymmetry::Bool = true,
    invariant::Types.AbstractInvariant = Types.KineticEnergy(),
    spectral::SpectralBackends.AbstractSpectralBackend = SpectralBackends.AutoSpectralBackend(),
    execution::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.SerialBackend(),
    advecting_hat = velocity_hat,
)
    spectral = Types.resolve_spectral(Types.require_coefficient_spectral(spectral))
    _calculate_shell_to_shell!(result, ws, velocity_hat, ks, Types.resolve_execution(execution), spectral;
        dealiasing=dealiasing, verify_antisymmetry=verify_antisymmetry, invariant=invariant,
        advecting_hat=advecting_hat)
    return result
end

"""
    calculate_shell_to_shell_transfer_batch(velocity_hats, ks; binning, dealiasing, verify_antisymmetry,
        invariant, spectral, geometry, execution) -> Vector{ShellToShellResult}

Shell-to-shell transfer for a batch of snapshots that share one grid (`velocity_hats` an iterable of
`(ns..., D)` coefficient arrays). The shell structure is built ONCE and each worker reuses a single
`ShellToShellWorkspace` across its snapshots; `execution = ThreadedBackend()` (requires `using
OhMyThreads`) threads over snapshots with a **serial** inner transform (the batch is the outer parallel
axis, so no nested threading). Results are in input order, bit-identical to calling
[`calculate_shell_to_shell_transfer`](@ref) per snapshot (each snapshot self-advects).
"""
function calculate_shell_to_shell_transfer_batch(
    velocity_hats,
    ks;
    binning::Types.AbstractShellBinning = _default_binning(ks),
    dealiasing::Types.AbstractDealiasing = Types.OrszagTwoThirds(),
    verify_antisymmetry::Bool = true,
    invariant::Types.AbstractInvariant = Types.KineticEnergy(),
    spectral::SpectralBackends.AbstractSpectralBackend = SpectralBackends.AutoSpectralBackend(),
    geometry::Types.AbstractShellGeometry = Types.IsotropicShells(),
    execution::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.SerialBackend(),
)
    spectral = Types.resolve_spectral(Types.require_coefficient_spectral(spectral))
    n = length(velocity_hats)
    n == 0 && return Types.ShellToShellResult[]
    k_mag   = ShellBinning.shell_coordinate(geometry, ks)
    edges   = ShellBinning.shell_edges(binning, maximum(k_mag))
    centers = ShellBinning.shell_centers(binning, maximum(k_mag))
    N_sh    = length(centers)
    results = Vector{Types.ShellToShellResult}(undef, n)
    _shell_to_shell_batch!(Types.resolve_execution(execution), results, velocity_hats, ks, centers, edges, N_sh;
        binning=binning, dealiasing=dealiasing, verify_antisymmetry=verify_antisymmetry, invariant=invariant,
        spectral=spectral, geometry=geometry)
    return results
end

# Serial reference: one workspace reused across the whole batch; each snapshot self-advects.
function _shell_to_shell_batch!(::ComputationalBackends.AbstractSerialBackend, results, velocity_hats, ks, centers, edges, N_sh;
                                binning, dealiasing, verify_antisymmetry, invariant, spectral, geometry)
    FT = real(eltype(first(velocity_hats)))
    ws = Workspaces.ShellToShellWorkspace(first(velocity_hats), ks, binning; geometry=geometry, dealiasing=dealiasing)
    for i in eachindex(velocity_hats)
        vh = velocity_hats[i]
        T_mat = Matrix{FT}(undef, N_sh, N_sh); net = Vector{FT}(undef, N_sh)
        res  = Types.ShellToShellResult(centers, edges, T_mat, net, FT(NaN))
        max_asym = _calculate_shell_to_shell!(res, ws, vh, ks, ComputationalBackends.SerialBackend(), spectral;
            dealiasing=dealiasing, verify_antisymmetry=verify_antisymmetry, invariant=invariant, advecting_hat=vh)
        results[i] = Types.ShellToShellResult(centers, edges, T_mat, net, max_asym)
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

# Distributed over the batch — overridden by the Distributed extension (snapshots partitioned across workers).
function _shell_to_shell_batch_distributed!(args...; kwargs...)
    throw(ArgumentError("execution = DistributedBackend() for the shell-to-shell batch requires Distributed. " *
                        "Run `using Distributed` to load the extension."))
end
_shell_to_shell_batch!(::ComputationalBackends.AbstractDistributedBackend, results, velocity_hats, ks, centers, edges, N_sh; kwargs...) =
    _shell_to_shell_batch_distributed!(results, velocity_hats, ks, centers, edges, N_sh; kwargs...)

# GPU over the batch — loop the single-shot device kernel, reusing one device workspace (device inputs →
# device buffers via `similar`). Requires the FFT backend (the DirectSum shell path is host scalar-indexed).
function _shell_to_shell_batch!(gpu::ComputationalBackends.AbstractGPUBackend, results, velocity_hats, ks, centers, edges, N_sh;
                                binning, dealiasing, verify_antisymmetry, invariant, spectral, geometry)
    spectral isa SpectralBackends.DirectSumSpectralBackend && throw(ArgumentError(
        "calculate_shell_to_shell_transfer_batch on a GPUBackend requires spectral = SpectralBackends.FFTSpectralBackend() " *
        "(cuFFT via AbstractFFTs); SpectralBackends.DirectSumSpectralBackend is a host-only reference."))
    FT = real(eltype(first(velocity_hats)))
    ws = Workspaces.ShellToShellWorkspace(first(velocity_hats), ks, binning; geometry=geometry, dealiasing=dealiasing)
    for i in eachindex(velocity_hats)
        vh = velocity_hats[i]
        T_mat = Matrix{FT}(undef, N_sh, N_sh); net = Vector{FT}(undef, N_sh)
        res  = Types.ShellToShellResult(centers, edges, T_mat, net, FT(NaN))
        max_asym = _calculate_shell_to_shell!(res, ws, vh, ks, gpu, spectral;
            dealiasing=dealiasing, verify_antisymmetry=verify_antisymmetry, invariant=invariant, advecting_hat=vh)
        results[i] = Types.ShellToShellResult(centers, edges, T_mat, net, max_asym)
    end
    return results
end

# Any other execution backend has no batch hook — refuse rather than silently run serial.
_shell_to_shell_batch!(be::ComputationalBackends.AbstractExecutionBackend, results, velocity_hats, ks, centers, edges, N_sh; kwargs...) =
    throw(ArgumentError("calculate_shell_to_shell_transfer_batch supports SerialBackend(), ThreadedBackend(), and DistributedBackend(); " *
                        "got execution = $(typeof(be))."))

"""
    calculate_scalar_shell_to_shell_transfer(velocity_hat, scalar_hat, ks; kwargs...) -> ShellToShellResult

Shell-to-shell transfer of passive-scalar **variance** `T_θ(n,m)`: the rate at which scalar
variance is transferred from scalar-shell `m` to scalar-shell `n`, mediated by the velocity:

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

# Dispatch on (execution, spectral). Serial loop: direct vs the optimized FFT path; the
# threaded/distributed/GPU execution backends (extensions) parallelise the mediator loop and
# pass the spectral backend down to each per-mediator nonlinear term.
_calculate_shell_to_shell!(result, ws, velocity_hat, ks, ::ComputationalBackends.SerialBackend, spectral::SpectralBackends.DirectSumSpectralBackend; kwargs...) =
    _calculate_shell_to_shell_direct!(result, ws, velocity_hat, ks; kwargs...)

_calculate_shell_to_shell!(result, ws, velocity_hat, ks, ::ComputationalBackends.SerialBackend, spectral::SpectralBackends.FFTSpectralBackend; kwargs...) =
    _shell_to_shell_fft!(result, ws, velocity_hat, ks; kwargs...)

_calculate_shell_to_shell!(result, ws, velocity_hat, ks, ::ComputationalBackends.ThreadedBackend, spectral::SpectralBackends.AbstractSpectralBackend; kwargs...) =
    _shell_to_shell_threaded!(result, ws, velocity_hat, ks, spectral; kwargs...)

# Distributed / GPU dispatch → named stubs overridden by the Distributed / KernelAbstractions
# extensions (same pattern as the threaded path). Without the extension loaded these raise an
# informative `using X` error rather than a bare `MethodError`.
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
    _calculate_shell_to_shell_direct!(result, ws, velocity_hat, ks; dealiasing, verify_antisymmetry, invariant)

Direct-sum (ComputationalBackends.SerialBackend) shell-to-shell transfer. Writes into `result` using
workspace buffers from `ws` — no heap allocation in the hot path.

For each mediator shell m:
  1. Build û_m = û restricted to shell m (using ws.shell_idx)
  2. Compute N̂_m = FFT[(u_m·∇)u] using ws.nonlinear buffers
  3. Accumulate T(n,m) for all receiver shells n
"""
function _calculate_shell_to_shell_direct!(
    result::Types.ShellToShellResult,
    ws::Workspaces.ShellToShellWorkspace,
    velocity_hat,
    ks;
    dealiasing::Types.AbstractDealiasing,
    verify_antisymmetry::Bool,
    invariant::Types.AbstractInvariant = Types.KineticEnergy(),
    advecting_hat = velocity_hat,
)
    nd    = length(ks)
    ns    = size(velocity_hat)[1:nd]
    M     = size(velocity_hat, nd+1)   # components of the binned/carried primary field
    FT    = real(eltype(velocity_hat))
    N_sh  = size(result.transfer_matrix, 1)

    fill!(result.transfer_matrix, zero(FT))
    col = ws.net_transfer            # length-N_sh scratch, rewritten per mediator then copied out

    for m in 1:N_sh
        # Build û_m: velocity restricted to shell m — reuse ws.û_m
        fill!(ws.û_m, zero(eltype(ws.û_m)))
        for I in CartesianIndices(ns)
            ws.shell_idx[I] == m || continue
            for comp in 1:M
                ws.û_m[I, comp] = velocity_hat[I, comp]
            end
        end

        # N̂_m = (u·∇)u_m: the FULL velocity advects the band-m field (Alexakis–Mininni–Pouquet
        # 2005). This makes A[n,m] = Σ_{k∈S_n} Re{û*·N̂_m} both antisymmetric (A[n,m]+A[m,n]=0)
        # and correctly reducing (Σ_m A[n,m] = transfer_spectrum[n]) — no ½(A−Aᵀ) needed.
        NonlinearTerm.compute_nonlinear_term!(ws.nonlinear, ws.û_m, ks;
                                dealiasing=dealiasing, spectral=SpectralBackends.DirectSumSpectralBackend(),
                                advecting_hat=advecting_hat)
        N̂_m = ws.nonlinear.N̂

        # A(n,m) = Σ_{k∈S_n} Re{û*·N̂_m} for every receiver shell n, in ONE pass over the modes:
        # each mode contributes to exactly the shell it belongs to. Scanning the grid once per
        # receiver shell instead costs N_sh passes for the same result.
        Invariants.transfer_density_scatter!(col, invariant, velocity_hat, N̂_m, ks, ws.shell_idx)
        @inbounds for n in 1:N_sh
            result.transfer_matrix[n, m] = col[n]
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
    return Types.LinearBinning(min_dk)
end

end # module ShellToShellTransfer
