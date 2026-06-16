module FlowInvariantTransferFSHExt

using FastSphericalHarmonics: FastSphericalHarmonics as FSH
using FlowInvariantTransfer: FlowInvariantTransfer as FET
using FlowInvariantTransfer.Types: SHTBackend, SpectralFluxMethod, ShellToShellTransferMethod, ModeToModeTransferMethod

# SHTBackend front-end helpers for regular spherical grids
function FET.calculate_energy_transfer(
    method::Union{SpectralFluxMethod, ShellToShellTransferMethod, ModeToModeTransferMethod},
    velocity_fields::Tuple,
    coords_vecs::Tuple,
    ms::Tuple;
    backend::SHTBackend = SHTBackend(),
    kwargs...
)
    # Perform regular Spherical Harmonic Transform
    Nθ = ms[1]
    lmax = Nθ - 1
    
    # FastSphericalHarmonics expects velocities on the standard colatitude-longitude grid.
    # We transform each velocity field to spherical harmonic coefficients.
    # Note: For spherical vector fields, using Vector Spherical Harmonics (VSH) is the canonical formulation,
    # but regular scalar transforms of components can also be performed.
    # We allocate the coefficients matrix of shape (lmax+1, 2lmax+1, length(velocity_fields))
    FT = eltype(velocity_fields[1])
    Nphi = 2 * Nθ # standard grid dimensions
    
    coeffs = zeros(Complex{FT}, lmax+1, 2*lmax+1, length(velocity_fields))
    for c in 1:length(velocity_fields)
        grid_data = copy(velocity_fields[c])
        FSH.sph_transform!(grid_data)
        for l in 0:lmax
            for m in -l:l
                idx = FSH.sph_mode(l, m)
                coeffs[l+1, m+l+1, c] = grid_data[idx]
            end
        end
    end
    
    # Generate wavenumbers corresponding to spherical degrees l
    ks_l = collect(FT, 0:lmax)
    ks = (ks_l, ks_l) # degree components
    
    # Delegate to the core spectral calculation
    return FET.calculate_energy_transfer(method, coeffs, ks; kwargs...)
end

end # module
