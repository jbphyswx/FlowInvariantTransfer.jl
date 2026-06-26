module MHDTransfer

using ..Types: SpectralFluxResult, AbstractShellBinning, LinearBinning, AbstractSpectralBackend,
               DirectSumBackend, AbstractShellGeometry, IsotropicShells, KineticEnergy
using ..Invariants: transfer_density!
using ..NonlinearTerm: compute_nonlinear_term
using ..ShellBinning: shell_edges, shell_centers, assign_shells, shell_coordinate

export calculate_mhd_energy_transfer, calculate_mhd_cross_helicity_transfer

# ---------------------------------------------------------------------------
# Incompressible MHD cross-scale transfer (any dimension D ≥ nd)
# ---------------------------------------------------------------------------
#
# Incompressible MHD (∇·u = ∇·b = 0), in the Elsässer-free primitive form:
#
#   ∂_t u = −(u·∇)u + (b·∇)b − ∇P        (Lorentz force; ∇P absorbs magnetic pressure)
#   ∂_t b = −(u·∇)b + (b·∇)u
#
# Two quadratic ideal invariants are dimension-agnostic and handled here:
#   • total energy        E   = ½⟨|u|² + |b|²⟩
#   • cross-helicity      H_c = ⟨u·b⟩
# (Mean-square vector potential ½⟨a²⟩ in 2D is a passive-scalar-type invariant — use
#  calculate_scalar_flux with the flux function a; magnetic helicity ⟨a·b⟩ in 3D is future work.)
#
# All four nonlinear terms are formed with the SAME validated pseudospectral engine via its
# `advecting_hat` argument — `compute_nonlinear_term(f; advecting_hat=g) = (g·∇)f`:
#   N_uu = (u·∇)u,  N_bb = (b·∇)b,  N_ub = (u·∇)b,  N_bu = (b·∇)u.
# Per-mode transfer densities (flux convention Π = +cumsum(T), Π>0 forward; reduces to the pure
# hydrodynamic T(k) when b = 0):
#   t_KE = Re{û*·N_uu} − Re{û*·N_bb}              (kinetic; advection minus Lorentz work)
#   t_ME = Re{b̂*·N_ub} − Re{b̂*·N_bu}              (magnetic; induction)
#   t_E  = t_KE + t_ME            (Σ_k t_E = 0, total energy conserved)
#   t_Hc = Re{b̂*·N_uu} − Re{b̂*·N_bb} + Re{û*·N_ub} − Re{û*·N_bu}   (Σ_k t_Hc = 0)
#
# References: Pouquet, Frisch & Léorat (1976); Fyfe & Montgomery (1976); Alexakis & Biferale (2018).

# Bin a real per-mode density into shells: returns (centers, T(k), Π(K)=cumsum T).
function _bin_density(density, ks, binning::AbstractShellBinning, geometry::AbstractShellGeometry)
    FT        = eltype(density)
    k_coord   = shell_coordinate(geometry, ks)
    edges     = shell_edges(binning, maximum(k_coord))
    centers   = shell_centers(binning, maximum(k_coord))
    shell_idx = assign_shells(k_coord, edges)
    N_sh      = length(centers)
    T = fill!(similar(density, FT, N_sh), zero(FT))
    @inbounds for I in CartesianIndices(size(density))
        n = shell_idx[I]
        n == 0 && continue
        T[n] += density[I]
    end
    flux = cumsum(T)
    return SpectralFluxResult(collect(centers), T, flux)
end

# Compute the four MHD nonlinear terms (u-valued: N_uu, N_bu; b-valued: N_ub, N_bb).
function _mhd_nonlinear_terms(velocity_hat, magnetic_hat, ks, dealiasing, spectral)
    N_uu = compute_nonlinear_term(velocity_hat, ks; dealiasing=dealiasing, spectral=spectral, advecting_hat=velocity_hat) # (u·∇)u
    N_bb = compute_nonlinear_term(magnetic_hat, ks; dealiasing=dealiasing, spectral=spectral, advecting_hat=magnetic_hat) # (b·∇)b
    N_ub = compute_nonlinear_term(magnetic_hat, ks; dealiasing=dealiasing, spectral=spectral, advecting_hat=velocity_hat) # (u·∇)b
    N_bu = compute_nonlinear_term(velocity_hat, ks; dealiasing=dealiasing, spectral=spectral, advecting_hat=magnetic_hat) # (b·∇)u
    return (N_uu=N_uu, N_bb=N_bb, N_ub=N_ub, N_bu=N_bu)
end

# Re{ carrier*(k) · N(k) } summed over components, as a real per-mode density.
_dot_density(carrier, N, ks) = transfer_density!(similar(carrier, real(eltype(carrier)), size(carrier)[1:length(ks)]...),
                                                 KineticEnergy(), carrier, N, ks)

"""
    calculate_mhd_energy_transfer(velocity_hat, magnetic_hat, ks; binning, dealiasing=true,
                                  spectral=DirectSumBackend(), geometry=IsotropicShells())
        -> (total=SpectralFluxResult, kinetic=SpectralFluxResult, magnetic=SpectralFluxResult)

Cross-scale transfer of **total MHD energy** `E = ½⟨|u|² + |b|²⟩` for incompressible MHD, split
into kinetic and magnetic parts. `velocity_hat` and `magnetic_hat` are complex `(ns..., D)`
Fourier coefficients (the magnetic field `b` in Alfvén/velocity units). Total energy is conserved
by the nonlinear terms (`Σ_k T_E ≈ 0`); when `b = 0` the kinetic result reduces to the pure
hydrodynamic spectral flux. See the module header for the exact densities and references.
"""
function calculate_mhd_energy_transfer(
    velocity_hat,
    magnetic_hat,
    ks;
    binning::AbstractShellBinning,
    dealiasing::Bool = true,
    spectral::AbstractSpectralBackend = DirectSumBackend(),
    geometry::AbstractShellGeometry = IsotropicShells(),
)
    N = _mhd_nonlinear_terms(velocity_hat, magnetic_hat, ks, dealiasing, spectral)
    t_KE = _dot_density(velocity_hat, N.N_uu, ks) .- _dot_density(velocity_hat, N.N_bb, ks)
    t_ME = _dot_density(magnetic_hat, N.N_ub, ks) .- _dot_density(magnetic_hat, N.N_bu, ks)
    t_E  = t_KE .+ t_ME
    return (
        total    = _bin_density(t_E,  ks, binning, geometry),
        kinetic  = _bin_density(t_KE, ks, binning, geometry),
        magnetic = _bin_density(t_ME, ks, binning, geometry),
    )
end

"""
    calculate_mhd_cross_helicity_transfer(velocity_hat, magnetic_hat, ks; binning, dealiasing=true,
                                          spectral=DirectSumBackend(), geometry=IsotropicShells())
        -> SpectralFluxResult

Cross-scale transfer of **cross-helicity** `H_c = ⟨u·b⟩` for incompressible MHD. Conserved by the
nonlinear terms (`Σ_k T_{H_c} ≈ 0`). Returns the transfer spectrum `T_{H_c}(k)` and flux
`Π_{H_c}(K)`.
"""
function calculate_mhd_cross_helicity_transfer(
    velocity_hat,
    magnetic_hat,
    ks;
    binning::AbstractShellBinning,
    dealiasing::Bool = true,
    spectral::AbstractSpectralBackend = DirectSumBackend(),
    geometry::AbstractShellGeometry = IsotropicShells(),
)
    N = _mhd_nonlinear_terms(velocity_hat, magnetic_hat, ks, dealiasing, spectral)
    t_Hc = _dot_density(magnetic_hat, N.N_uu, ks) .- _dot_density(magnetic_hat, N.N_bb, ks) .+
           _dot_density(velocity_hat, N.N_ub, ks) .- _dot_density(velocity_hat, N.N_bu, ks)
    return _bin_density(t_Hc, ks, binning, geometry)
end

end # module MHDTransfer
