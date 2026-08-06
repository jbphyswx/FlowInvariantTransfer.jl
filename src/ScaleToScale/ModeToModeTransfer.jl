module ModeToModeTransfer

using ..Types: Types
using ComputationalBackends: ComputationalBackends
using SpectralBackends: SpectralBackends
using ..Invariants: Invariants
using ..NonlinearTerm: NonlinearTerm
using ..Workspaces: Workspaces
using ..Utils: Utils

export calculate_mode_to_mode_transfer, calculate_mode_to_mode_transfer!
export calculate_scalar_mode_to_mode_transfer, calculate_scalar_mode_to_mode_transfer!

"""
    calculate_mode_to_mode_transfer(velocity_hat, ks; invariant=KineticEnergy(), dealiasing=OrszagTwoThirds(),
                                    spectral=SpectralBackends.DirectSumSpectralBackend(), max_scales=1024, force=false)
        -> ModeToModeTriadResult

Fully **mode-resolved** triad transfer `S(k|p)` — the rate at which the chosen quadratic
invariant is delivered to receiver scale `k` from giver scale `p` (mediated by `q = k−p`), the
finest object in the reduction hierarchy `S(k|p)` → `T(K,Q)` (shell-to-shell) → `T(k)`, `Π(K)`
(spectral flux).

It is built from the validated pseudospectral nonlinear term — for each giver scale `p`,
`N̂_p = (u·∇)u_p` (the full velocity advecting the single-scale field `u_p`), and

    S(k|p) = Re{ û*(k) · N̂_p(k) }   (generalised per invariant via `transfer_density!`).

This construction is exact and inherits the right structural properties (verified by tests):
- **reduces**: `Σ_p S(k|p) = T(k)` = the spectral transfer (`calculate_spectral_flux`),
- **antisymmetric**: `S(k|p) + S(p|k) = 0` (incompressible, since `∫(u·∇)(u_p·u_k)=0`),
- **conserves**: `Σ_k Σ_p S(k|p) = 0`.

# Cost & memory
Resolving every receiver/giver pair is `O(N_scales)` nonlinear-term evaluations →
`O(N_scales · Nᴰ log N)` time with `SpectralBackends.FFTSpectralBackend` (strongly recommended) and an `O(N_scales²)`
result tensor. For the *aggregates* prefer the cheaper, coarser diagnostics:
`calculate_spectral_flux` (`T(k)`, `Π`) or `calculate_shell_to_shell_transfer` (`T(n,m)`).
A guard errors when `N_scales = prod(size grid) > max_scales`; pass `force=true` to override.

# Keyword arguments
- `invariant::AbstractInvariant`: which quadratic invariant (default `KineticEnergy()`).
- `dealiasing::AbstractDealiasing=OrszagTwoThirds()`: 2/3-rule dealiasing of the nonlinear term.
- `spectral::SpectralBackends.AbstractSpectralBackend`: transform (`SpectralBackends.DirectSumSpectralBackend()` default, `SpectralBackends.FFTSpectralBackend()` fast).
- `max_scales::Int=1024`, `force::Bool=false`: resolved-tensor size guard.

# Returns
`ModeToModeTriadResult` with `net_transfer` (`T(k)`, shape `ns`) and `transfer` (the resolved
`S(k|p)`, shape `(ns..., ns...)`).
"""
function calculate_mode_to_mode_transfer(
    velocity_hat,
    ks;
    invariant::Types.AbstractInvariant = Types.KineticEnergy(),
    dealiasing::Types.AbstractDealiasing = Types.OrszagTwoThirds(),
    spectral::SpectralBackends.AbstractSpectralBackend = SpectralBackends.DirectSumSpectralBackend(),
    execution::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.SerialBackend(),
    max_scales::Int = 1024,
    force::Bool = false,
    advecting_hat = velocity_hat,
)
    Types.require_coefficient_spectral(spectral)
    nd = length(ks)
    ns = size(velocity_hat)[1:nd]
    FT = real(eltype(velocity_hat))
    Nscales = prod(ns)
    (force || Nscales <= max_scales) || throw(ArgumentError(
        "calculate_mode_to_mode_transfer resolves S(k|p) over all $Nscales×$Nscales scale pairs " *
        "(O(N^{2D}) time/memory); N_scales=$Nscales exceeds max_scales=$max_scales. Use " *
        "calculate_shell_to_shell_transfer / calculate_spectral_flux for the aggregates, or pass force=true."))

    ws     = Workspaces.NonlinearTermWorkspace(velocity_hat, ks)
    û_p    = similar(velocity_hat)
    S      = similar(velocity_hat, FT, ns..., ns...)     # S[k..., p...]  (inherent O(N^{2D}) output)
    net    = similar(velocity_hat, FT, ns...)
    result = Types.ModeToModeTriadResult(invariant, ks, net, S)
    return calculate_mode_to_mode_transfer!(result, ws, û_p, velocity_hat, ks;
        invariant=invariant, dealiasing=dealiasing, spectral=spectral, execution=execution,
        advecting_hat=advecting_hat)
end

"""
    calculate_mode_to_mode_transfer!(result, ws, û_p, velocity_hat, ks; kwargs...) -> result

In-place mode-to-mode transfer: writes `S(k|p)` and the net into the caller-provided
`result::ModeToModeTriadResult`, reusing the `ws::NonlinearTermWorkspace` and the giver-scale scratch
`û_p` (same shape as `velocity_hat`). Allocates **zero** bytes beyond those buffers — for reuse across
snapshots/parameters. The per-`p` loop drives the 0-alloc [`compute_nonlinear_term!`] / `transfer_density!`
hot paths and writes each giver column directly into a view of `result.transfer`.
"""
function calculate_mode_to_mode_transfer!(
    result::Types.ModeToModeTriadResult,
    ws::Workspaces.NonlinearTermWorkspace,
    û_p,
    velocity_hat,
    ks;
    invariant::Types.AbstractInvariant = Types.KineticEnergy(),
    dealiasing::Types.AbstractDealiasing = Types.OrszagTwoThirds(),
    spectral::SpectralBackends.AbstractSpectralBackend = SpectralBackends.DirectSumSpectralBackend(),
    execution::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.SerialBackend(),
    advecting_hat = velocity_hat,
)
    Types.require_coefficient_spectral(spectral)
    _mode_to_mode_loop!(Types.resolve_execution(execution), result, ws, û_p, velocity_hat, ks;
        invariant=invariant, dealiasing=dealiasing, spectral=spectral, advecting_hat=advecting_hat)
    return result
end

# Per-giver loop, dispatched on the execution backend. Each giver `p` writes a DISJOINT column
# `S[·, p]`, so the loop is embarrassingly parallel (near-linear scaling): the threaded/distributed/GPU
# backends override the named stubs (extensions), each thread/rank using its OWN single-threaded
# workspace (no nested FFT threads). `net[k] = Σ_p S[k,p]` is a `+`-reduction of per-worker partials.
function _mode_to_mode_loop!(::ComputationalBackends.SerialBackend, result, ws, û_p, velocity_hat, ks;
                             invariant, dealiasing, spectral, advecting_hat)
    nd = length(ks)
    ns = size(velocity_hat)[1:nd]
    M  = size(velocity_hat, nd + 1)   # components of the giver/carried primary field
    S   = result.transfer
    net = result.net_transfer
    fill!(net, zero(eltype(net)))
    colons = ntuple(_ -> Colon(), nd)
    @inbounds for p in CartesianIndices(ns)
        # Isolate giver scale p of the primary field, advected by the full velocity:
        # 𝒩̂_p = (u·∇)f_p (f = u for energy; f = θ for a passive scalar).
        fill!(û_p, zero(eltype(û_p)))
        for c in 1:M
            û_p[p, c] = velocity_hat[p, c]
        end
        NonlinearTerm.compute_nonlinear_term!(ws, û_p, ks; dealiasing=dealiasing, spectral=spectral, advecting_hat=advecting_hat)
        Sp = view(S, colons..., p)                                   # write S(·|p) in place
        Invariants.transfer_density!(Sp, invariant, velocity_hat, ws.N̂, ks)     # validates invariant/dimension
        net .+= Sp
    end
    return result
end
_mode_to_mode_loop!(::ComputationalBackends.ThreadedBackend, result, ws, û_p, velocity_hat, ks; kwargs...) =
    _mode_to_mode_threaded!(result, ws, û_p, velocity_hat, ks; kwargs...)
_mode_to_mode_loop!(exec::ComputationalBackends.DistributedBackend, result, ws, û_p, velocity_hat, ks; kwargs...) =
    _mode_to_mode_distributed!(result, ws, û_p, velocity_hat, ks, exec; kwargs...)
_mode_to_mode_loop!(gpu::ComputationalBackends.GPUBackend, result, ws, û_p, velocity_hat, ks; kwargs...) =
    _mode_to_mode_gpu!(result, ws, û_p, velocity_hat, ks, gpu; kwargs...)

_mode_to_mode_threaded!(args...; kwargs...) = throw(ArgumentError(
    "Threaded mode-to-mode transfer requires OhMyThreads. Run `using OhMyThreads` to load the extension."))
_mode_to_mode_distributed!(args...; kwargs...) = throw(ArgumentError(
    "Distributed mode-to-mode transfer requires Distributed. Run `using Distributed` to load the extension."))
_mode_to_mode_gpu!(args...; kwargs...) = throw(ArgumentError(
    "GPU mode-to-mode transfer requires KernelAbstractions. Run `using KernelAbstractions` to load the extension."))

"""
    calculate_scalar_mode_to_mode_transfer(velocity_hat, scalar_hat, ks; kwargs...) -> ModeToModeTriadResult

Fully mode-resolved passive-scalar **variance** transfer `S_θ(k|p)` — variance delivered to
scalar scale `k` from scalar scale `p` (mediated by the velocity, `q = k−p`). Thin wrapper over
[`calculate_mode_to_mode_transfer`](@ref) with `invariant = PassiveScalar()` and
`advecting_hat = velocity_hat`; the scalar may be `(ns...)` or `(ns..., 1)`.
"""
function calculate_scalar_mode_to_mode_transfer(velocity_hat, scalar_hat, ks; kwargs...)
    θ̂ = Utils.as_component_field(scalar_hat, length(ks))
    return calculate_mode_to_mode_transfer(θ̂, ks;
        invariant=Types.PassiveScalar(), advecting_hat=velocity_hat, kwargs...)
end

"""
    calculate_scalar_mode_to_mode_transfer!(result, ws, û_p, velocity_hat, scalar_hat, ks; kwargs...)

In-place passive-scalar mode-to-mode variance transfer — thin wrapper over
[`calculate_mode_to_mode_transfer!`](@ref) (`invariant = PassiveScalar()`), writing into the
caller-provided `result`/`ws`/`û_p` (0 alloc beyond those). `ws`/`û_p` are sized for the scalar field.
"""
function calculate_scalar_mode_to_mode_transfer!(result, ws, û_p, velocity_hat, scalar_hat, ks; kwargs...)
    θ̂ = Utils.as_component_field(scalar_hat, length(ks))
    return calculate_mode_to_mode_transfer!(result, ws, û_p, θ̂, ks;
        invariant=Types.PassiveScalar(), advecting_hat=velocity_hat, kwargs...)
end

end # module ModeToModeTransfer
