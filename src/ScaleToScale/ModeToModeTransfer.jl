module ModeToModeTransfer

using ..Types: Types
using FlowTransformBindings: FlowTransformBindings as FTB
using ComputationalBackends: ComputationalBackends
using SpectralBackends: SpectralBackends
using ..Invariants: Invariants
using ..NonlinearTerm: NonlinearTerm
using ..Workspaces: Workspaces
using ..Utils: Utils

export calculate_mode_to_mode_transfer, calculate_mode_to_mode_transfer!, mode_weights
export calculate_scalar_mode_to_mode_transfer, calculate_scalar_mode_to_mode_transfer!

"""
    calculate_mode_to_mode_transfer(velocity_hat, ks; invariant=KineticEnergy(), dealiasing=OrszagTwoThirds(),
                                    transform=AutoSpectralBackend(), execution=SerialBackend(),
                                    advecting_hat=velocity_hat)
        -> ModeToModeTriadResult

Mode-resolved triad transfer `S(k|p)` between the stored coefficients ("slots") of the spectral
layout — the rate at which slot `k` gives the chosen quadratic invariant to the slot field
`u_p(x) = w(p)·Re{û(p) e^{ip·x}}` (mediated by `q = k−p`), the part of `−dE(k)/dt` that advection of
`u_p` accounts for:

    S(k|p) = Re{ û*(k) · [(u·∇)u_p]^(k) }   (generalised per invariant via `transfer_density!`).

`w(p)` is the number of full-spectrum modes slot `p` stands for (1 on a full layout; on a half layout 2
on the interior of the halved axis, where the slot also stands for its mirror `−p`), returned as
`result.weights`; the slot fields sum to `u`. It is the finest object in the reduction hierarchy
`S(k|p)` → `T(K,Q)` (shell-to-shell) → `T(k)`, `Π(K)` (spectral flux):

- **reduces**: `Σ_p S(k|p) = T(k)`, the spectral transfer (`calculate_spectral_flux`);
- **antisymmetric** (divergence-free, dealiased): `w(k)S(k|p) = −w(p)S(p|k)`, since
  `w(k)S(k|p) = ⟨u_k·(u·∇)u_p⟩` and `∫(u·∇)(u_k·u_p) = 0` — `S(k|p) = −S(p|k)` on a full layout;
- **conserves**: `Σ_k Σ_p w(k)S(k|p) = 0`.

# Cost & memory
One nonlinear-term evaluation per slot `p`: `O(N_scales)` evaluations, `O(N_scales · Nᴰ log N)` time
with `SpectralBackends.FFTSpectralBackend`, and an `O(N_scales²)` result tensor, where
`N_scales = prod(ms)` is the number of stored slots. The aggregates come cheaper from
`calculate_spectral_flux` (`T(k)`, `Π`) or `calculate_shell_to_shell_transfer` (`T(n,m)`).

# Keyword arguments
- `invariant::AbstractInvariant`: which quadratic invariant (default `KineticEnergy()`).
- `dealiasing::AbstractDealiasing=OrszagTwoThirds()`: dealiasing of the nonlinear term; under
  `PaddedThreeHalves()` each slot's term is formed on the padded grid.
- `transform::SpectralBackends.AbstractSpectralBackend`: `AutoSpectralBackend()` (default, the FFT when FFTW is loaded).
- `execution::ComputationalBackends.AbstractExecutionBackend`: parallelism over the slots `p`
  (`SerialBackend()` default; threaded, distributed and GPU through their extensions). Under
  `GPUBackend(dev)` host fields are copied to `dev` once per call, and the result arrays are on `dev`.
- `advecting_hat=velocity_hat`: the advecting velocity of `(u·∇)u_p`.

# Returns
`ModeToModeTriadResult` with `weights` (`w`, shape `ms`), `net_transfer` (`T(k)`, shape `ms`) and
`transfer` (the resolved `S(k|p)`, shape `(ms..., ms...)`).
"""
function calculate_mode_to_mode_transfer(
    velocity_hat,
    ks;
    invariant::Types.AbstractInvariant = Types.KineticEnergy(),
    dealiasing::Types.AbstractDealiasing = Types.OrszagTwoThirds(),
    transform::SpectralBackends.AbstractSpectralBackend = SpectralBackends.AutoSpectralBackend(),
    execution::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.SerialBackend(),
    advecting_hat = velocity_hat,
)
    transform = Types.resolve_transform(Types.require_coefficient_transform(transform))
    exec = Types.resolve_execution(execution)
    velocity_hat, advecting_hat = Types._fields_to_execution(exec, velocity_hat, advecting_hat)
    nd = length(ks)
    ns = size(velocity_hat)[1:nd]
    FT = real(eltype(velocity_hat))

    ws     = Workspaces.NonlinearTermWorkspace(velocity_hat, ks; dealiasing = dealiasing)
    û_p    = similar(velocity_hat)
    S      = similar(velocity_hat, FT, ns..., ns...)     # S[k..., p...]  (inherent O(N^{2D}) output)
    net    = similar(velocity_hat, FT, ns...)
    result = Types.ModeToModeTriadResult(invariant, ks, mode_weights(velocity_hat, FT, ks), net, S)
    return calculate_mode_to_mode_transfer!(result, ws, û_p, velocity_hat, ks;
        invariant=invariant, dealiasing=dealiasing, transform=transform, execution=exec,
        advecting_hat=advecting_hat)
end

"""
    mode_weights(proto, FT, ks) -> AbstractArray{FT}

The fold weight `w(k)` of every stored coefficient of the layout `ks`
(`FlowTransformBindings.mode_fold`), shape `spectral_size(ks)`, in `proto`'s array type: the
`weights` of a [`ModeToModeTriadResult`](@ref).
"""
function mode_weights(proto::AbstractArray, ::Type{FT}, ks::Tuple) where {FT}
    ms = FTB.spectral_size(ks)
    w = FT[FTB.mode_fold(ks, I) for I in CartesianIndices(ms)]
    return copyto!(similar(proto, FT, ms...), w)
end

"""
    calculate_mode_to_mode_transfer!(result, ws, û_p, velocity_hat, ks; kwargs...) -> result

In-place mode-to-mode transfer: writes `S(k|p)` and the net into the caller-provided
`result::ModeToModeTriadResult` (its `weights` from [`mode_weights`](@ref)), reusing the
`ws::NonlinearTermWorkspace` (built for the same `dealiasing`) and the slot scratch `û_p` (same shape
as `velocity_hat`). Allocation-free beyond those buffers.
"""
function calculate_mode_to_mode_transfer!(
    result::Types.ModeToModeTriadResult,
    ws::Workspaces.NonlinearTermWorkspace,
    û_p,
    velocity_hat,
    ks;
    invariant::Types.AbstractInvariant = Types.KineticEnergy(),
    dealiasing::Types.AbstractDealiasing = Types.OrszagTwoThirds(),
    transform::SpectralBackends.AbstractSpectralBackend = SpectralBackends.AutoSpectralBackend(),
    execution::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.SerialBackend(),
    advecting_hat = velocity_hat,
)
    transform = Types.resolve_transform(Types.require_coefficient_transform(transform))
    _mode_to_mode_loop!(Types.resolve_execution(execution), result, ws, û_p, velocity_hat, ks;
        invariant=invariant, dealiasing=dealiasing, transform=transform, advecting_hat=advecting_hat)
    return result
end

"""
    _mode_to_mode_giver!(ws, velocity_hat, ks, p, ph, truncate, spectral) -> ws.N̂

`𝒩̂_p = (u·∇)f_p` for the single-mode field `f_p` of slot `p`, written into `ws.N̂`.

The slot holds one Fourier mode, so its gradient is known in closed form: with `A_c = w_p f̂_c(p)`
(`w_p` the slot's fold weight) and `g(x) = Σ_j k_j(p) u_j(x)` (from `ws.u_phys`),

    ∂_j f_c(x) = Re[i k_j(p) A_c e^{ip·x}]   ⇒   𝒩_c(x) = Re[i A_c e^{ip·x}] · g(x),

since every `k_j(p)` is a real scalar. `ph` holds the per-axis factors of `e^{ip·x}`. Each component
costs one forward transform; the caller synthesizes `ws.u_phys` once for all slots.
"""
function _mode_to_mode_giver!(ws, velocity_hat, ks, p, ph, truncate::Bool, spectral)
    nd = length(ks)
    ns = FTB.full_size(ks)
    M  = size(velocity_hat, nd + 1)
    FT = real(eltype(velocity_hat))
    # Per-axis phase factors of e^{ip·x}. `wgt` is the slot's Hermitian weight: on a half layout the
    # stored mode carries its own amplitude and its mirror's.
    for d in 1:nd
        pd = FTB.mode_frequency(ks[d], p[d])
        @inbounds for i in 1:ns[d]
            ph[d][i] = cis(2 * FT(π) * pd * FT(i - 1) / FT(ns[d]))
        end
    end
    wgt = FT(FTB.mode_fold(ks, p))
    # g(x) = Σ_j k_j(p) u_j(x)
    fill!(ws.g_phys, zero(FT))
    for j in 1:nd
        kj = FT(FTB.derivative_wavenumber(ks[j], p[j]))
        kj == 0 && continue
        ws.g_phys .+= kj .* selectdim(ws.u_phys, nd + 1, j)
    end
    for c in 1:M
        A = velocity_hat[p, c] * wgt
        Nc = selectdim(ws.N_phys, nd + 1, c)
        _fill_plane_wave!(Nc, A, ph, ws.g_phys, ns)
    end
    return NonlinearTerm.nlt_forward!(ws, ks, spectral; truncate = truncate)
end

# Nc[I] = Re[i·A·∏_d ph[d][I_d]] · w[I]
function _fill_plane_wave!(Nc, A, ph, w, ns::NTuple{nd,Int}) where {nd}
    iA = im * A
    @inbounds for I in CartesianIndices(ns)
        e = ph[1][I[1]]
        for d in 2:nd
            e *= ph[d][I[d]]
        end
        Nc[I] = real(iA * e) * w[I]
    end
    return Nc
end

# Loop over the slots `p`, dispatched on execution. Each `p` writes the disjoint column `S[·, p]`; the
# parallel stubs below run one single-threaded workspace per worker and form `net[k] = Σ_p S[k,p]` from
# per-worker partials.
function _mode_to_mode_loop!(::ComputationalBackends.SerialBackend, result, ws, û_p, velocity_hat, ks;
                             invariant, dealiasing, transform, advecting_hat)
    dealiasing isa Types.PaddedThreeHalves &&
        return _mode_to_mode_one_hot!(result, ws, û_p, velocity_hat, ks; invariant = invariant,
                                      dealiasing = dealiasing, transform = transform, advecting_hat = advecting_hat)
    nd = length(ks)
    ms = FTB.spectral_size(ks)
    S   = result.transfer
    net = result.net_transfer
    fill!(net, zero(eltype(net)))
    colons = ntuple(_ -> Colon(), nd)
    truncate = !(dealiasing isa Types.NoDealiasing)
    NonlinearTerm.synthesize_advecting!(ws, advecting_hat, ks; dealiasing = dealiasing, transform = transform)
    ph = ws.phase
    @inbounds for p in CartesianIndices(ms)
        Sp = view(S, colons..., p)
        # A slot the dealiasing rule discards has a zero field, so its column is zero.
        if truncate && NonlinearTerm._is_dealiased(ks, p)
            fill!(Sp, zero(eltype(Sp)))
            continue
        end
        _mode_to_mode_giver!(ws, velocity_hat, ks, p, ph, truncate, transform)
        Invariants.transfer_density!(Sp, invariant, velocity_hat, ws.N̂, ks)     # validates invariant/dimension
        net .+= Sp
    end
    return result
end

# Each slot `p` through the full nonlinear term on `û_p`, which holds slot `p` alone; under
# `PaddedThreeHalves` each product is formed on the padded grid.
function _mode_to_mode_one_hot!(result, ws, û_p, velocity_hat, ks; invariant, dealiasing, transform, advecting_hat)
    nd = length(ks)
    ms = FTB.spectral_size(ks)
    M  = size(velocity_hat, nd + 1)
    S   = result.transfer
    net = result.net_transfer
    fill!(net, zero(eltype(net)))
    fill!(û_p, zero(eltype(û_p)))
    colons = ntuple(_ -> Colon(), nd)
    NonlinearTerm.synthesize_advecting!(ws, advecting_hat, ks; dealiasing = dealiasing, transform = transform)
    @inbounds for p in CartesianIndices(ms)
        for c in 1:M
            û_p[p, c] = velocity_hat[p, c]
        end
        NonlinearTerm.compute_advection!(ws, û_p, ks; dealiasing = dealiasing, transform = transform)
        Sp = view(S, colons..., p)
        Invariants.transfer_density!(Sp, invariant, velocity_hat, ws.N̂, ks)
        net .+= Sp
        for c in 1:M
            û_p[p, c] = zero(eltype(û_p))
        end
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

Fully mode-resolved passive-scalar **variance** transfer `S_θ(k|p)` — the rate at which scalar
mode `k` gives variance to scalar mode `p` (mediated by the velocity, `q = k−p`). Thin wrapper over
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
