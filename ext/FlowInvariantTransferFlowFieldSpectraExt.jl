module FlowInvariantTransferFlowFieldSpectraExt

using FlowFieldSpectra: FlowFieldSpectra as FFS
using FlowInvariantTransfer: FlowInvariantTransfer as FIT
using SpectralBackends: SpectralBackends
using ComputationalBackends: ComputationalBackends

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
The axes build an all-periodic Cartesian `FlowGeometries.Grids.StructuredGrid`. Pass each axis as a
uniform range so its period is inferred (`n·Δ`); a stretched or plain-vector axis needs `domain_size`
(its per-direction period). The direct-sum transform is exact on uniform axes.
"""
function FIT.calculate_energy_transfer(
    method::Union{FIT.Types.SpectralFluxMethod, FIT.Types.ShellToShellTransferMethod, FIT.Types.ModeToModeTransferMethod},
    velocity_fields::Tuple,
    coords_vecs::Tuple,
    ms::Tuple;
    backend::SpectralBackends.AbstractSpectralBackend = SpectralBackends.DirectSumSpectralBackend(),
    execution::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend(),
    domain_size = nothing,
    kwargs...
)
    nd  = length(coords_vecs)
    FT  = float(eltype(coords_vecs[1]))

    # Build an all-periodic Cartesian StructuredGrid straight from the coordinate axes. FlowGeometries'
    # `_to_axis` adapts each axis to `FT` while keeping its type, so a uniform *range* axis carries its
    # spacing and its period is inferred (n·Δ, the spectral domain) — no hand-computed Δ. `domain_size`
    # supplies the period only for axes handed in as plain vectors, whose type can't carry the spacing.
    geom = FFS.FlowGeometries.Geometry.CartesianGeometry{FT}()
    grid = FFS.FlowGeometries.Grids.StructuredGrid(geom, coords_vecs...;
                                                   topology = ntuple(_ -> true, nd), period = domain_size)

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
