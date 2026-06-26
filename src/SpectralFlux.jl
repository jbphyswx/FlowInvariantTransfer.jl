module SpectralFlux

using ..Types: SpectralFluxMethod, SpectralFluxResult, AbstractShellBinning, LinearBinning, AbstractSpectralBackend, DirectSumBackend, AbstractInvariant, KineticEnergy, PassiveScalar, AbstractFieldDecomposition, NoDecomposition, HelmholtzDecomposition, RotationalDecomposition, DivergentDecomposition, HelicalDecomposition, AbstractShellGeometry, IsotropicShells, AbstractDealiasing, OrszagTwoThirds
using ..Invariants: transfer_density!
using ..Decomposition: decompose_field
using ..ShellBinning: shell_edges, shell_centers, n_shells, assign_shells, shell_coordinate
using ..Utils: wavenumber_grid, wavenumber_magnitude_grid, domain_size_from_coords, as_component_field
using ..NonlinearTerm: compute_nonlinear_term, compute_nonlinear_term!
using ..Workspaces: NonlinearTermWorkspace, SpectralFluxWorkspace

export calculate_spectral_flux, calculate_spectral_flux!, calculate_scalar_flux, calculate_helical_partial_fluxes

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

"""
    calculate_spectral_flux(velocity_hat, ks; binning, dealiasing=true) -> SpectralFluxResult

Compute the spectral energy transfer spectrum T(k) and the cumulative energy
flux Π(K) from Fourier-space velocity data.

# Arguments
- `velocity_hat`: Complex array of size `(ns..., D)` — Fourier coefficients of
  the D velocity components on an N-point periodic grid.
- `ks`: Tuple of 1D physical-wavenumber vectors (matching FFTW fftfreq convention).

# Keyword Arguments
- `binning::AbstractShellBinning`: Shell binning strategy; default `LinearBinning(1.0)`.
- `dealiasing::AbstractDealiasing=OrszagTwoThirds()`: Apply 2/3 dealiasing rule when computing (u·∇)u.
- `spectral::AbstractSpectralBackend`: transform backend — `DirectSumBackend()` (default, no deps) or `FFTBackend()` (requires FFTW extension).

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
    binning::AbstractShellBinning = _default_binning(ks),
    dealiasing::AbstractDealiasing = OrszagTwoThirds(),
    invariant::AbstractInvariant = KineticEnergy(),
    decomposition::AbstractFieldDecomposition = NoDecomposition(),
    spectral::AbstractSpectralBackend = DirectSumBackend(),
    advecting_hat = velocity_hat,
    geometry::AbstractShellGeometry = IsotropicShells(),
)
    decomposed = decompose_field(decomposition, velocity_hat, ks)
    return _calculate_spectral_flux_decomposed(
        decomposed, velocity_hat, ks, binning, dealiasing, invariant, spectral, advecting_hat, geometry
    )
end

function _calculate_spectral_flux_decomposed(
    û_decomp::AbstractArray{<:Complex},
    velocity_hat,
    ks,
    binning::AbstractShellBinning,
    dealiasing::AbstractDealiasing,
    invariant::AbstractInvariant,
    spectral::AbstractSpectralBackend,
    advecting_hat,
    geometry::AbstractShellGeometry,
)
    ws        = SpectralFluxWorkspace(velocity_hat, ks, binning; geometry=geometry)
    k_mag     = shell_coordinate(geometry, ks)
    edges     = shell_edges(binning, maximum(k_mag))
    centers   = shell_centers(binning, maximum(k_mag))
    shell_idx = assign_shells(k_mag, edges)
    result    = SpectralFluxResult(centers, similar(ws.T_spec), similar(ws.flux))

    if û_decomp === velocity_hat
        calculate_spectral_flux!(result, ws, velocity_hat, ks, shell_idx;
                                  dealiasing=dealiasing, invariant=invariant, spectral=spectral,
                                  advecting_hat=advecting_hat)
    else
        compute_nonlinear_term!(ws.nonlinear, velocity_hat, ks; dealiasing=dealiasing,
                                spectral=spectral, advecting_hat=advecting_hat)
        _calculate_spectral_flux_with_N̂!(result, ws, û_decomp, ws.nonlinear.N̂, ks, shell_idx; invariant=invariant)
    end
    return result
end

function _calculate_spectral_flux_decomposed(
    decomposed::NamedTuple,
    velocity_hat,
    ks,
    binning::AbstractShellBinning,
    dealiasing::AbstractDealiasing,
    invariant::AbstractInvariant,
    spectral::AbstractSpectralBackend,
    advecting_hat,
    geometry::AbstractShellGeometry,
)
    ws = SpectralFluxWorkspace(velocity_hat, ks, binning; geometry=geometry)
    compute_nonlinear_term!(ws.nonlinear, velocity_hat, ks; dealiasing=dealiasing,
                            spectral=spectral, advecting_hat=advecting_hat)
    N̂ = ws.nonlinear.N̂

    k_mag     = shell_coordinate(geometry, ks)
    edges     = shell_edges(binning, maximum(k_mag))
    centers   = shell_centers(binning, maximum(k_mag))
    shell_idx = assign_shells(k_mag, edges)

    return map(decomposed) do û_comp
        res = SpectralFluxResult(centers, similar(ws.T_spec), similar(ws.flux))
        _calculate_spectral_flux_with_N̂!(res, ws, û_comp, N̂, ks, shell_idx; invariant=invariant)
        return res
    end
end

"""
    calculate_spectral_flux!(result, ws, velocity_hat, ks, shell_idx; dealiasing, invariant, backend)

In-place version of `calculate_spectral_flux`. Writes into `result` using
preallocated buffers from `ws` and a precomputed `shell_idx` array (from `assign_shells`).
Zero heap allocations in the hot path.
"""
function calculate_spectral_flux!(
    result::SpectralFluxResult,
    ws::SpectralFluxWorkspace,
    velocity_hat,
    ks,
    shell_idx::AbstractArray{Int};
    dealiasing::AbstractDealiasing = OrszagTwoThirds(),
    invariant::AbstractInvariant = KineticEnergy(),
    spectral::AbstractSpectralBackend = DirectSumBackend(),
    advecting_hat = velocity_hat,
)
    compute_nonlinear_term!(ws.nonlinear, velocity_hat, ks;
                            dealiasing=dealiasing, spectral=spectral, advecting_hat=advecting_hat)
    _calculate_spectral_flux_with_N̂!(result, ws, velocity_hat, ws.nonlinear.N̂, ks, shell_idx; invariant=invariant)
    return result
end

function _calculate_spectral_flux_with_N̂!(
    result::SpectralFluxResult,
    ws::SpectralFluxWorkspace,
    velocity_hat,
    N̂,
    ks,
    shell_idx::AbstractArray{Int};
    invariant::AbstractInvariant = KineticEnergy(),
)
    nd = length(ks)
    ns = size(velocity_hat)[1:nd]
    FT = real(eltype(velocity_hat))

    # Write per-mode transfer density into ws.transfer_density
    transfer_density!(ws.transfer_density, invariant, velocity_hat, N̂, ks)

    fill!(ws.T_spec, zero(FT))
    for I in CartesianIndices(ns)
        n = shell_idx[I]
        n == 0 && continue
        ws.T_spec[n] += ws.transfer_density[I]
    end

    copyto!(result.transfer_spectrum, ws.T_spec)
    # Flux convention (Alexakis & Biferale 2018, Eqs. 12–17; see THEORY.md §0.5):
    #   T(k) = Re{û*·N̂} is the net energy *loss* from shell k, and
    #   Π(K) = +Σ_{k≤K} T(k)  ⇒  Π>0 forward (down-scale) cascade, Π<0 inverse.
    # (Earlier code negated this, returning −Π — i.e. forward cascades read as negative.)
    cumsum!(ws.flux, ws.T_spec)
    copyto!(result.flux, ws.flux)

    return result
end

# ---------------------------------------------------------------------------
# Passive-scalar variance flux (convenience over the generalized advecting_hat path)
# ---------------------------------------------------------------------------

"""
    calculate_scalar_flux(velocity_hat, scalar_hat, ks; binning, dealiasing=true, spectral) -> SpectralFluxResult

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
    binning::AbstractShellBinning = _default_binning(ks),
    dealiasing::AbstractDealiasing = OrszagTwoThirds(),
    spectral::AbstractSpectralBackend = DirectSumBackend(),
    geometry::AbstractShellGeometry = IsotropicShells(),
)
    θ̂ = as_component_field(scalar_hat, length(ks))
    return calculate_spectral_flux(θ̂, ks; binning=binning, dealiasing=dealiasing,
        invariant=PassiveScalar(), advecting_hat=velocity_hat, spectral=spectral, geometry=geometry)
end

# ---------------------------------------------------------------------------
# Helical partial energy fluxes Π^{s_k s_p s_q} (Waleffe 1992; Biferale–Musacchio–Toschi 2012)
# ---------------------------------------------------------------------------

"""
    calculate_helical_partial_fluxes(velocity_hat, ks; binning, dealiasing=OrszagTwoThirds(),
        spectral=DirectSumBackend(), geometry=IsotropicShells())
        -> (channels::Dict{NTuple{3,Int},SpectralFluxResult}, total::SpectralFluxResult, k_shells)

Decompose the 3D kinetic-energy flux into the **eight helical partial fluxes** `Π^{s_k s_p s_q}(K)`,
where each of the three modes in a triad interaction carries a definite helicity sign
`s ∈ {+1,-1}` (Waleffe 1992; Biferale, Musacchio & Toschi 2012; Alexakis 2017). Writing the
velocity as `u = u₊ + u₋` (see [`HelicalDecomposition`](@ref)),

    T^{s_k s_p s_q}(k) = Re{ û_{s_k}*(k) · [ (u_{s_p}·∇) u_{s_q} ](k) },

and `Π^{s_k s_p s_q}(K) = Σ_{k≤K} T^{s_k s_p s_q}`. The eight channels sum to the full energy flux.
The **homochiral** channels (`s_k=s_p=s_q`, i.e. `(+++)`,`(---)`) tend to drive an inverse cascade,
the **heterochiral** ones a forward cascade — the central result of helical turbulence.

`channels` is keyed by `(s_k, s_p, s_q)` with entries `±1`. Built entirely from the package's
helical decomposition + generalized nonlinear term; 3D only.
"""
function calculate_helical_partial_fluxes(
    velocity_hat,
    ks;
    binning::AbstractShellBinning = _default_binning(ks),
    dealiasing::AbstractDealiasing = OrszagTwoThirds(),
    spectral::AbstractSpectralBackend = DirectSumBackend(),
    geometry::AbstractShellGeometry = IsotropicShells(),
)
    nd = length(ks)
    nd == 3 || throw(ArgumentError("helical partial fluxes are defined in 3D only (got nd=$nd)."))
    FT = real(eltype(velocity_hat))
    ns = size(velocity_hat)[1:nd]

    dec = decompose_field(HelicalDecomposition(), velocity_hat, ks)
    u = Dict(1 => dec.positive, -1 => dec.negative)

    k_coord = shell_coordinate(geometry, ks)
    edges   = shell_edges(binning, maximum(k_coord))
    centers = collect(shell_centers(binning, maximum(k_coord)))
    sidx    = assign_shells(k_coord, edges)
    Nsh     = length(centers)

    binflux(t) = begin
        T = zeros(FT, Nsh)
        @inbounds for I in CartesianIndices(ns)
            n = sidx[I]; n == 0 && continue; T[n] += t[I]
        end
        SpectralFluxResult(centers, T, cumsum(T))
    end

    channels = Dict{NTuple{3,Int}, SpectralFluxResult}()
    td = similar(velocity_hat, FT, ns...)
    for sp in (1, -1), sq in (1, -1)
        N̂ = compute_nonlinear_term(u[sq], ks; advecting_hat=u[sp], dealiasing=dealiasing, spectral=spectral)  # (u_{sp}·∇)u_{sq}
        for sk in (1, -1)
            transfer_density!(td, KineticEnergy(), u[sk], N̂, ks)
            channels[(sk, sp, sq)] = binflux(td)
        end
    end
    total = SpectralFluxResult(centers,
        sum(c.transfer_spectrum for c in values(channels)),
        sum(c.flux for c in values(channels)))
    return (channels = channels, total = total, k_shells = centers)
end

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
    return LinearBinning(min_dk)
end

end # module SpectralFlux
