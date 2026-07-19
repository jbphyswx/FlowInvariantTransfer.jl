module FlowInvariantTransferFlowFieldSpectraExt

using FlowFieldSpectra: FlowFieldSpectra as FFS, calculate_spectrum
using FlowInvariantTransfer: FlowInvariantTransfer as FIT
using FlowInvariantTransfer.Types: SpectralFluxMethod, ShellToShellTransferMethod, ModeToModeTransferMethod

"""
    calculate_energy_transfer(method, velocity_fields::Tuple, coords_vecs::Tuple, ms::Tuple; kwargs...)

Physical-space front-end for the uniform-Cartesian spectral diagnostics: pass the velocity component
arrays `(u, v[, w])` (each shaped like the grid, e.g. `(Nx, Ny)`) and the per-axis coordinate vectors
`coords_vecs = (xaxis, yaxis[, zaxis])`; the fields are transformed to spectral coefficients via
`FlowFieldSpectra.calculate_spectrum` and fed to the requested transfer diagnostic.

Convention bridge: FlowFieldSpectra takes coordinates and fields as *flattened* point vectors (its
grid abstraction covers scattered points too) and returns fftSHIFTED (centred, `k = −N/2 … N/2−1`)
coefficients, whereas FlowInvariantTransfer's core uses grid-shaped arrays in FFTW fftfreq order
(`0,1,…,N/2−1,−N/2,…,−1`). This method flattens on the way in and `ifftshift`s on the way out.
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
    gsz = ntuple(d -> length(coords_vecs[d]), nd)        # grid shape from the axes
    CI  = CartesianIndices(gsz)
    ds  = domain_size === nothing ?
          ntuple(d -> (cv = coords_vecs[d]; length(cv) > 1 ? length(cv) * (cv[2] - cv[1]) : one(eltype(cv))), nd) :
          domain_size

    # Flatten to FlowFieldSpectra's point-vector convention. `vec` is column-major (first axis fastest),
    # and `[coords_vecs[d][I[d]] for I in CI]` walks CI in the same order, so fields and coords align.
    flat_coords = ntuple(d -> [coords_vecs[d][I[d]] for I in CI], nd)
    flat_fields = ntuple(c -> vec(velocity_fields[c]), length(velocity_fields))
    grid = FFS.UniformCartesianGrid(flat_coords; domain_size = ds)

    # Transform physical → spectral, then reorder centred → fftfreq (ifftshift = circshift by −N/2 per axis)
    # so the coefficient at grid index i carries the fftfreq wavenumber FlowInvariantTransfer's core expects.
    # FlowFieldSpectra's `calculate_spectrum` uses two-axis (transform × execution) dispatch; the
    # grid-first keyword entry resolves the execution backend (Serial/Threaded/GPU) internally.
    coeffs, ks = FFS.calculate_spectrum(grid, flat_fields, ms; transform = backend, execution = execution)
    shifts = ntuple(d -> d <= nd ? -(size(coeffs, d) ÷ 2) : 0, ndims(coeffs))
    coeffs_ff = circshift(coeffs, shifts)
    ks_ff = ntuple(d -> circshift(collect(ks[d]), -(length(ks[d]) ÷ 2)), nd)

    return FIT.calculate_energy_transfer(method, coeffs_ff, ks_ff; kwargs...)
end

end # module
