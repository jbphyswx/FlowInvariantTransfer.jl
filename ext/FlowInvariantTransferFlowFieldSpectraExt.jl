module FlowInvariantTransferFlowFieldSpectraExt

using FlowFieldSpectra: FlowFieldSpectra as FFS, calculate_spectrum
using FlowInvariantTransfer: FlowInvariantTransfer as FET
using FlowInvariantTransfer.Types: SpectralFluxMethod, ShellToShellTransferMethod, ModeToModeTransferMethod

"""
    calculate_energy_transfer(method, velocity_fields::Tuple, coords_vecs::Tuple, ms::Tuple; backend, kwargs...)

Overload that allows starting with physical-space velocity fields (e.g. `(u, v)`) and coordinates,
automatically transforming them to spectral coefficients using `FlowFieldSpectra.calculate_spectrum`
before running the energy transfer diagnostics.
"""
function FET.calculate_energy_transfer(
    method::Union{SpectralFluxMethod, ShellToShellTransferMethod, ModeToModeTransferMethod},
    velocity_fields::Tuple,
    coords_vecs::Tuple,
    ms::Tuple;
    backend::FFS.AbstractSpectralBackend = FFS.DirectSumBackend(),
    domain_size = nothing,
    kwargs...
)
    # 1. Transform physical-space fields to spectral coefficients
    coeffs, ks = calculate_spectrum(
        backend, coords_vecs, velocity_fields, ms; 
        domain_size = domain_size, kwargs...
    )
    
    # 2. Delegate to the core spectral calculation
    return FET.calculate_energy_transfer(method, coeffs, ks; kwargs...)
end

end # module
