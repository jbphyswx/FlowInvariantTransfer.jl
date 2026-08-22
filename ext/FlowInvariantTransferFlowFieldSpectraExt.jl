module FlowInvariantTransferFlowFieldSpectraExt

using FlowFieldSpectra: FlowFieldSpectra as FFS
using FlowInvariantTransfer: FlowInvariantTransfer as FIT
using SpectralBackends: SpectralBackends
using ComputationalBackends: ComputationalBackends

# Uniform-grid branch of the physical one-call `calculate_energy_transfer(method, fields, coords, ms;
# spectral, …)` for the spectral-flux/shell/mode family (the non-NUFFT `_physical_energy_transfer`
# method; the NUFFT branch lives in core). Velocity component arrays `(u, v[, w])` on per-axis coordinate
# vectors `coords_vecs = (xaxis, yaxis[, zaxis])` are transformed to spectral coefficients via
# `FlowFieldSpectra.calculate_spectrum` and fed to the diagnostic.
#
# Convention bridge: FlowFieldSpectra returns fftSHIFTED (centred, `k = −N/2 … N/2−1`) coefficients
# `(ms…, ncomp)`, whereas the core uses FFTW fftfreq order (`0,1,…,N/2−1,−N/2,…,−1`); this `ifftshift`s
# them on the way out. The axes build an all-periodic Cartesian `FlowGeometries.Grids.StructuredGrid`:
# pass each axis as a uniform range so its period is inferred (`n·Δ`); a stretched or plain-vector axis
# needs `domain_size` (its per-direction period). The direct-sum transform is exact on uniform axes.
function FIT._physical_energy_transfer(
    spectral::SpectralBackends.AbstractSpectralBackend,
    method::Union{FIT.Types.SpectralFluxMethod, FIT.Types.ShellToShellTransferMethod, FIT.Types.ModeToModeTransferMethod},
    velocity_fields::Tuple,
    coords_vecs::Tuple,
    ms::Tuple;
    execution::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend(),
    domain_size = nothing,
    kwargs...
)
    nd  = length(coords_vecs)
    FT  = float(eltype(coords_vecs[1]))
    geom = FFS.FlowGeometries.Geometry.CartesianGeometry{FT}()
    grid = FFS.FlowGeometries.Grids.StructuredGrid(geom, coords_vecs...;
                                                   topology = ntuple(_ -> true, nd), period = domain_size)
    coeffs, ks = FFS.calculate_spectrum(grid, velocity_fields, ms; transform = spectral, execution = execution)
    shifts = ntuple(d -> d <= nd ? -(size(coeffs, d) ÷ 2) : 0, ndims(coeffs))
    coeffs_ff = circshift(coeffs, shifts)
    ks_ff = ntuple(d -> circshift(collect(ks[d]), -(length(ks[d]) ÷ 2)), nd)
    return FIT.calculate_energy_transfer(method, coeffs_ff, ks_ff; kwargs...)
end

end # module
