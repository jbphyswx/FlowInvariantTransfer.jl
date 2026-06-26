module ScaleToScaleTransfer

using ..Types: ModeToModeTransferMethod, ModeToModeTriadResult, AbstractInvariant, KineticEnergy, AbstractDealiasing, OrszagTwoThirds,
               PassiveScalar, AbstractSpectralBackend, DirectSumBackend, FFTBackend
using ..Invariants: transfer_density!
using ..NonlinearTerm: compute_nonlinear_term!
using ..Workspaces: NonlinearTermWorkspace
using ..Utils: as_component_field

export calculate_mode_to_mode_transfer, calculate_scalar_mode_to_mode_transfer

"""
    calculate_mode_to_mode_transfer(velocity_hat, ks; invariant=KineticEnergy(), dealiasing=true,
                                    spectral=DirectSumBackend(), max_modes=1024, force=false)
        -> ModeToModeTriadResult

Fully **mode-resolved** triad transfer `S(k|p)` — the rate at which the chosen quadratic
invariant is delivered to receiver mode `k` from giver mode `p` (mediated by `q = k−p`), the
finest object in the reduction hierarchy `S(k|p)` → `T(K,Q)` (shell-to-shell) → `T(k)`, `Π(K)`
(spectral flux).

It is built from the validated pseudospectral nonlinear term — for each giver mode `p`,
`N̂_p = (u·∇)u_p` (the full velocity advecting the single-mode field `u_p`), and

    S(k|p) = Re{ û*(k) · N̂_p(k) }   (generalised per invariant via `transfer_density!`).

This construction is exact and inherits the right structural properties (verified by tests):
- **reduces**: `Σ_p S(k|p) = T(k)` = the spectral transfer (`calculate_spectral_flux`),
- **antisymmetric**: `S(k|p) + S(p|k) = 0` (incompressible, since `∫(u·∇)(u_p·u_k)=0`),
- **conserves**: `Σ_k Σ_p S(k|p) = 0`.

# Cost & memory
Resolving every receiver/giver pair is `O(N_modes)` nonlinear-term evaluations →
`O(N_modes · Nᴰ log N)` time with `FFTBackend` (strongly recommended) and an `O(N_modes²)`
result tensor. For the *aggregates* prefer the cheaper, coarser diagnostics:
`calculate_spectral_flux` (`T(k)`, `Π`) or `calculate_shell_to_shell_transfer` (`T(n,m)`).
A guard errors when `N_modes = prod(size grid) > max_modes`; pass `force=true` to override.

# Keyword arguments
- `invariant::AbstractInvariant`: which quadratic invariant (default `KineticEnergy()`).
- `dealiasing::AbstractDealiasing=OrszagTwoThirds()`: 2/3-rule dealiasing of the nonlinear term.
- `spectral::AbstractSpectralBackend`: transform (`DirectSumBackend()` default, `FFTBackend()` fast).
- `max_modes::Int=1024`, `force::Bool=false`: resolved-tensor size guard.

# Returns
`ModeToModeTriadResult` with `net_transfer` (`T(k)`, shape `ns`) and `transfer` (the resolved
`S(k|p)`, shape `(ns..., ns...)`).
"""
function calculate_mode_to_mode_transfer(
    velocity_hat,
    ks;
    invariant::AbstractInvariant = KineticEnergy(),
    dealiasing::AbstractDealiasing = OrszagTwoThirds(),
    spectral::AbstractSpectralBackend = DirectSumBackend(),
    max_modes::Int = 1024,
    force::Bool = false,
    advecting_hat = velocity_hat,
)
    nd = length(ks)
    ns = size(velocity_hat)[1:nd]
    M  = size(velocity_hat, nd + 1)   # components of the giver/carried primary field
    FT = real(eltype(velocity_hat))
    Nmodes = prod(ns)
    (force || Nmodes <= max_modes) || throw(ArgumentError(
        "calculate_mode_to_mode_transfer resolves S(k|p) over all $Nmodes×$Nmodes mode pairs " *
        "(O(N^{2D}) time/memory); N_modes=$Nmodes exceeds max_modes=$max_modes. Use " *
        "calculate_shell_to_shell_transfer / calculate_spectral_flux for the aggregates, or pass force=true."))

    ws  = NonlinearTermWorkspace(velocity_hat, ks)
    û_p = similar(velocity_hat)
    td  = similar(velocity_hat, FT, ns...)
    S   = similar(velocity_hat, FT, ns..., ns...)     # S[k..., p...]
    net = fill!(similar(velocity_hat, FT, ns...), zero(FT))

    @inbounds for p in CartesianIndices(ns)
        # Isolate giver mode p of the primary field, advected by the full velocity:
        # 𝒩̂_p = (u·∇)f_p (f = u for energy; f = θ for a passive scalar).
        fill!(û_p, zero(eltype(û_p)))
        for c in 1:M
            û_p[p, c] = velocity_hat[p, c]
        end
        compute_nonlinear_term!(ws, û_p, ks; dealiasing=dealiasing, spectral=spectral, advecting_hat=advecting_hat)
        transfer_density!(td, invariant, velocity_hat, ws.N̂, ks)   # validates invariant/dimension
        for k in CartesianIndices(ns)
            S[k, p] = td[k]
            net[k] += td[k]
        end
    end

    return ModeToModeTriadResult(invariant, ks, net, S)
end

"""
    calculate_scalar_mode_to_mode_transfer(velocity_hat, scalar_hat, ks; kwargs...) -> ModeToModeTriadResult

Fully mode-resolved passive-scalar **variance** transfer `S_θ(k|p)` — variance delivered to
scalar mode `k` from scalar mode `p` (mediated by the velocity, `q = k−p`). Thin wrapper over
[`calculate_mode_to_mode_transfer`](@ref) with `invariant = PassiveScalar()` and
`advecting_hat = velocity_hat`; the scalar may be `(ns...)` or `(ns..., 1)`.
"""
function calculate_scalar_mode_to_mode_transfer(velocity_hat, scalar_hat, ks; kwargs...)
    θ̂ = as_component_field(scalar_hat, length(ks))
    return calculate_mode_to_mode_transfer(θ̂, ks;
        invariant=PassiveScalar(), advecting_hat=velocity_hat, kwargs...)
end

end # module ScaleToScaleTransfer
