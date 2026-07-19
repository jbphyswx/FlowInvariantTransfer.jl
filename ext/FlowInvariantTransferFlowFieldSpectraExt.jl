module FlowInvariantTransferFlowFieldSpectraExt

using FlowFieldSpectra: FlowFieldSpectra as FFS, calculate_spectrum
using FlowInvariantTransfer: FlowInvariantTransfer as FIT
using FlowInvariantTransfer.Types: SpectralFluxMethod, ShellToShellTransferMethod, ModeToModeTransferMethod

"""
    calculate_energy_transfer(method, velocity_fields::Tuple, coords_vecs::Tuple, ms::Tuple; kwargs...)

Physical-space front-end for the Cartesian spectral diagnostics: pass the velocity component
arrays `(u, v[, w])` (each shaped like the grid, e.g. `(Nx, Ny)`) and the per-axis coordinate vectors
`coords_vecs = (xaxis, yaxis[, zaxis])`; the fields are transformed to spectral coefficients via
`FlowFieldSpectra.calculate_spectrum` and fed to the requested transfer diagnostic.

Convention bridge: FlowFieldSpectra takes a tensor-product grid (per-axis coordinate vectors) and a
grid-shaped field tensor `(spatial…, batch…)`, returning fftSHIFTED (centred, `k = −N/2 … N/2−1`)
coefficients `(ms…, ncomp)`, whereas FlowInvariantTransfer's core uses grid-shaped arrays in FFTW
fftfreq order (`0,1,…,N/2−1,−N/2,…,−1`). This method `ifftshift`s the coefficients on the way out.
The coordinate axes are treated as a `NonuniformCartesianGrid` (correct for arbitrary axis vectors;
the direct-sum transform is exact on uniform axes too).
"""
function FIT.calculate_energy_transfer(
    method::Union{SpectralFluxMethod, ShellToShellTransferMethod, ModeToModeTransferMethod},
    velocity_fields::Tuple,
    coords_vecs::Tuple,
    ms::Tuple;
    backend::FFS.AbstractSpectralBackend = FFS.DirectSumBackend(),
    execution::FFS.AbstractExecutionBackend = FFS.AutoBackend(),
    domain_size = nothing,
    kwargs...
)
    nd  = length(coords_vecs)
    ds  = domain_size === nothing ?
          ntuple(d -> (cv = coords_vecs[d]; length(cv) > 1 ? length(cv) * (cv[2] - cv[1]) : one(eltype(cv))), nd) :
          domain_size

    # FlowFieldSpectra reads per-axis coordinate vectors + grid-shaped field components directly (it
    # stacks the tuple onto a trailing batch axis), so no flattening is needed.
    grid = FFS.NonuniformCartesianGrid(coords_vecs; domain_size = ds)

    # Transform physical → spectral, then reorder centred → fftfreq (ifftshift = circshift by −N/2 per axis)
    # so the coefficient at grid index i carries the fftfreq wavenumber FlowInvariantTransfer's core expects.
    # `calculate_spectrum` uses two-axis (transform × execution) dispatch; the grid-first keyword entry
    # resolves the execution backend (Serial/Threaded/GPU) internally.
    coeffs, ks = FFS.calculate_spectrum(grid, velocity_fields, ms; transform = backend, execution = execution)
    shifts = ntuple(d -> d <= nd ? -(size(coeffs, d) ÷ 2) : 0, ndims(coeffs))
    coeffs_ff = circshift(coeffs, shifts)
    ks_ff = ntuple(d -> circshift(collect(ks[d]), -(length(ks[d]) ÷ 2)), nd)

    return FIT.calculate_energy_transfer(method, coeffs_ff, ks_ff; kwargs...)
end

end # module
