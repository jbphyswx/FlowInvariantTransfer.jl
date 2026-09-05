module FlowInvariantTransferHelmholtzDecompositionExt

using HelmholtzDecomposition: HelmholtzDecomposition as HDjl
using FlowInvariantTransfer: FlowInvariantTransfer as FIT

const _HDDecomp = Union{FIT.Types.HelmholtzDecomposition, FIT.Types.RotationalDecomposition,
                        FIT.Types.DivergentDecomposition}

# 1. Physical-space decomposition on a FlowGeometries grid.
#
# The Poisson solve is left to `HDjl.AutoSolver`, which resolves it from the grid: it walks the
# geometry's spectral algorithms in preference order (Cartesian: FFT then NUFFT; spherical: FSHT then
# NUFSHT), each candidate checking the mask, axis uniformity, sampling, boundary and periodicity it
# needs, and reaches the iterative `CGSolver` when none applies. A stretched Cartesian grid therefore
# still reaches a NUFFT solve, and a Clenshaw–Curtis sphere a spherical-harmonic one. Pass `solver=`
# to name one.
#
# `helmholtz_decompose` covers `StructuredGrid` (spectral or iterative); every other grid goes to
# `helmholtz_decompose_spectral`, the entry declared on `AbstractGrid`. Both return the 3-way Hodge
# split `u_rot + u_div + u_harm` as component-last `(ns…, D)`; the harmonic (constant/mean,
# divergence-free) part folds into the rotational component so rotational + divergent == the field,
# matching the spectral path.
function FIT.Decomposition._decompose_field_physical(
    decomp::_HDDecomp,
    velocity_fields::Tuple,
    grid::HDjl.FlowGeometries.Grids.AbstractGrid;
    solver::HDjl.AbstractPoissonSolver = HDjl.AutoSolver(),
    boundary::HDjl.AbstractBoundaryCondition = HDjl.Neumann(),
    kwargs...,          # the caller's own keywords (a filter scale, diagnostics flags) pass by
)
    D = length(velocity_fields)
    nd = ndims(velocity_fields[1])
    ustacked = stack(velocity_fields; dims = nd + 1)
    res = grid isa HDjl.FlowGeometries.Grids.StructuredGrid ?
          HDjl.helmholtz_decompose(ustacked, grid; solver = solver, boundary = boundary) :
          HDjl.helmholtz_decompose_spectral(ustacked, grid; solver = solver)
    colons = ntuple(_ -> Colon(), nd)
    rot = ntuple(c -> res.u_rot[colons..., c] .+ res.u_harm[colons..., c], D)
    div = ntuple(c -> res.u_div[colons..., c], D)
    if decomp isa FIT.Types.HelmholtzDecomposition
        return (; rotational = rot, divergent = div)
    elseif decomp isa FIT.Types.RotationalDecomposition
        return rot
    else
        return div
    end
end

# Coordinate-vector convenience: build the Cartesian grid those axes describe and take the path above.
# `_to_axis` adapts each axis to `FT` preserving a uniform range's uniformity, so the solver selection
# reads the same grid either way.
function FIT.Decomposition._decompose_field_physical(
    decomp::_HDDecomp,
    velocity_fields::Tuple,
    coords_vecs::Tuple;
    mask = nothing,
    kwargs...
)
    D  = length(velocity_fields)
    nd = length(coords_vecs)
    D == nd || throw(ArgumentError(
        "number of velocity components ($D) must equal spatial dimensions ($nd)"))
    FT = float(real(eltype(velocity_fields[1])))
    geom = HDjl.FlowGeometries.Geometry.CartesianGeometry{FT}()
    grid = HDjl.FlowGeometries.Grids.StructuredGrid(geom, coords_vecs...; mask = mask)
    return FIT.Decomposition._decompose_field_physical(decomp, velocity_fields, grid; kwargs...)
end

# 2. Spectral-space decomposition
function FIT.Decomposition._decompose_field_spectral(
    decomp::Union{FIT.Types.HelmholtzDecomposition, FIT.Types.RotationalDecomposition, FIT.Types.DivergentDecomposition},
    velocity_hat::AbstractArray{<:Complex},
    ks
)
    # N-D Leray projection straight from the wavenumber vectors (no grid reconstruction): the sibling's
    # `helmholtz_project_spectral(velocity_hat, ks)` is a device-generic pure-broadcast projection, so
    # this works in 2D and 3D and on device arrays. Component-last (ns..., D) convention both ways.
    res = HDjl.helmholtz_project_spectral(velocity_hat, ks)

    # SpectralCartesianResult is a 3-way Hodge split (u_rot + u_div + u_harm) with the k = 0
    # (constant/mean) mode isolated as the harmonic part — a constant field is both curl-free and
    # divergence-free. To keep FIT's 2-way (rotational, divergent) API and rotational + divergent ==
    # velocity_hat, fold the harmonic into the (divergence-free) rotational component. All are
    # component-last stacked (ns..., D), matching the FIT spectral convention.
    if decomp isa FIT.Types.HelmholtzDecomposition
        return (; rotational = res.u_rot .+ res.u_harm, divergent = res.u_div)
    elseif decomp isa FIT.Types.RotationalDecomposition
        return res.u_rot .+ res.u_harm
    elseif decomp isa FIT.Types.DivergentDecomposition
        return res.u_div
    else
        throw(ArgumentError("Unknown decomposition type: $decomp"))
    end
end

# 3. Direct spectral project method override — the in-place primitive. HD's mutating form separates the
# harmonic (k = 0) part, so the caller supplies its buffer `û_harm`; nothing is allocated here (the
# 3-way û_rot + û_div + û_harm == velocity_hat, exact and device-generic).
function FIT.Decomposition.helmholtz_project_spectral!(û_rot, û_div, û_harm, velocity_hat, ks)
    return HDjl.helmholtz_project_spectral!(û_rot, û_div, û_harm, velocity_hat, ks)
end

end # module
