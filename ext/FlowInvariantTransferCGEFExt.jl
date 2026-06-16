module FlowInvariantTransferCGEFExt

using CoarseGrainingEnergyFluxes: CoarseGrainingEnergyFluxes as CGEF
using FlowInvariantTransfer: FlowInvariantTransfer as FET
using FlowInvariantTransfer.Types: AbstractFilter, SharpSpectralFilter, GaussianFilter, TopHatFilter, CoarseGrainingFluxResult, CoarseGrainingFluxResultWithDiagnostics

# ---------------------------------------------------------------------------
# Filter type mapping: FET.AbstractFilter → CGEF.AbstractFilterKernel
# ---------------------------------------------------------------------------

_to_cgef_kernel(::GaussianFilter)      = CGEF.Kernels.GaussianKernel()
_to_cgef_kernel(::TopHatFilter)        = CGEF.Kernels.TopHatKernel()
_to_cgef_kernel(::SharpSpectralFilter) = CGEF.Kernels.SharpSpectralKernel()

# ---------------------------------------------------------------------------
# Override CoarseGrainingFlux._cg_flux_cgef
# ---------------------------------------------------------------------------

"""
    _cg_flux_cgef(velocity_fields, coords_vecs, ℓ, filter; kwargs...)

Delegate to `CoarseGrainingEnergyFluxes.compute_Π!` for the actual computation.

Supports 2D Cartesian grids from tuple inputs.  For spherical geometry or
more control (masks, backends), call CGEF directly and wrap the result with
`CoarseGrainingFluxResult`.
"""
function FET.CoarseGrainingFlux._cg_flux_cgef(
    velocity_fields::Tuple,
    coords_vecs::Tuple,
    ℓ::Real,
    filter::AbstractFilter;
    return_diagnostics::Bool = false,
    mask::Union{Nothing, AbstractMatrix{Bool}} = nothing,
    backend::CGEF.Filtering.AbstractExecutionBackend = CGEF.Filtering.AutoBackend(),
    kwargs...,
)
    D  = length(velocity_fields)
    nd = length(coords_vecs)
    D == nd || throw(ArgumentError(
        "Number of velocity components ($D) must equal number of spatial dimensions ($nd)"))
    nd == 2 || throw(ArgumentError(
        "FlowInvariantTransferCGEFExt currently supports 2D Cartesian grids only (nd=$nd). " *
        "For 3D or spherical, call CoarseGrainingEnergyFluxes directly."))

    FT  = eltype(velocity_fields[1])
    ns  = size(velocity_fields[1])
    Nlon, Nlat = ns

    # Build CGEF grid objects
    x_vec = coords_vecs[1]
    y_vec = coords_vecs[2]
    dx = length(x_vec) > 1 ? FT((x_vec[end] - x_vec[begin]) / (length(x_vec) - 1)) : FT(1)
    dy = length(y_vec) > 1 ? FT((y_vec[end] - y_vec[begin]) / (length(y_vec) - 1)) : FT(1)

    geom = CGEF.Geometry.CartesianGeometry(dx, dy)

    # All-wet mask if not provided
    wet = mask !== nothing ? mask : trues(Nlon, Nlat)
    grid = CGEF.Grids.StructuredGrid(geom, FT.(x_vec), FT.(y_vec), wet)

    cgef_kernel = _to_cgef_kernel(filter)

    # Allocate workspace and output
    workspace = CGEF.Diagnostics.ΠWorkspace(grid)
    Π_out = zeros(FT, Nlon, Nlat)

    u = velocity_fields[1]
    v = velocity_fields[2]

    CGEF.Diagnostics.compute_Π!(
        Π_out,
        u, v, nothing,
        grid,
        cgef_kernel,
        FT(ℓ);
        workspace = workspace,
        backend   = backend,
        kwargs...,
    )

    mean_Π = sum(Π_out[wet]) / max(1, count(wet))

    if return_diagnostics
        # Extract τ and S̄ from workspace after compute_Π! has populated them
        τ_arr = zeros(FT, Nlon, Nlat, 2, 2)
        S_arr = zeros(FT, Nlon, Nlat, 2, 2)
        τ_arr[:, :, 1, 1] .= workspace.τ_xx
        τ_arr[:, :, 1, 2] .= workspace.τ_xy
        τ_arr[:, :, 2, 1] .= workspace.τ_xy
        τ_arr[:, :, 2, 2] .= workspace.τ_yy
        S_arr[:, :, 1, 1] .= workspace.S_xx
        S_arr[:, :, 1, 2] .= workspace.S_xy
        S_arr[:, :, 2, 1] .= workspace.S_xy
        S_arr[:, :, 2, 2] .= workspace.S_yy
        return CoarseGrainingFluxResultWithDiagnostics(FT(ℓ), Π_out, FT(mean_Π), τ_arr, S_arr)
    else
        return CoarseGrainingFluxResult(FT(ℓ), Π_out, FT(mean_Π))
    end
end

end # module FlowInvariantTransferCGEFExt
