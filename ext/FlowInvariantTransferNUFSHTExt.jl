module FlowInvariantTransferNUFSHTExt

using NUFSHT: NUFSHT
using FlowInvariantTransfer: FlowInvariantTransfer as FET
using FlowInvariantTransfer.Types: NUFSHTBackend, SpectralFluxMethod, ShellToShellTransferMethod, ModeToModeTransferMethod

# NUFSHTBackend front-end helpers for scattered/non-uniform spherical grids
function FET.calculate_energy_transfer(
    method::Union{SpectralFluxMethod, ShellToShellTransferMethod, ModeToModeTransferMethod},
    velocity_fields::Tuple,
    coords_vecs::Tuple,
    ms::Tuple;
    backend::NUFSHTBackend = NUFSHTBackend(),
    tol::Real = 1e-8,
    kwargs...
)
    FT = eltype(velocity_fields[1])
    Nθ = ms[1]
    lmax = Nθ - 1
    
    # Coordinates: theta_nodes, phi_nodes
    θ = coords_vecs[1]
    φ = coords_vecs[2]
    
    # Setup NUFSHT plan
    plan = NUFSHT.nufsht_plan(lmax, θ, φ; tol=tol)
    
    # Allocate coefficients
    coeffs = zeros(Complex{FT}, lmax+1, 2*lmax+1, length(velocity_fields))
    for c in 1:length(velocity_fields)
        # Non-uniform transform of the component
        f_lm = NUFSHT.nufsht_adjoint(plan, velocity_fields[c])
        # Map output to coefficients matrix
        for l in 0:lmax
            for m in -l:l
                coeffs[l+1, m+l+1, c] = f_lm[l+1, m+l+1]
            end
        end
    end
    
    # Generate wavenumbers corresponding to spherical degrees l
    ks_l = collect(FT, 0:lmax)
    ks = (ks_l, ks_l)
    
    # Delegate to the core spectral calculation
    return FET.calculate_energy_transfer(method, coeffs, ks; kwargs...)
end

end # module
