module FlowInvariantTransferHelmholtzDecompositionExt

using HelmholtzDecomposition: HelmholtzDecomposition as HDjl
using FlowInvariantTransfer: FlowInvariantTransfer as FIT

const _HDDecomp = Union{FIT.Types.HelmholtzDecomposition, FIT.Types.RotationalDecomposition,
                        FIT.Types.DivergentDecomposition}

# 1. Physical-space decomposition on a FlowGeometries grid.
#
# HD selects the solver from the grid, boundary and loaded libraries. Its Cartesian FFT solver uses
# the spectral decomposition of the cell values. Other structured-grid solvers use HD's face operators;
# their returned cell values carry the cell-to-face-to-cell averaging.
#
# HD's three component-last parts are `u_rot + u_div + u_harm`. FIT folds the harmonic part into the
# rotational component and returns component views. The solve outcome reports the least converged
# potential solve.
function FIT.Decomposition._decompose_field_physical(
    decomp::_HDDecomp,
    velocity_fields::Tuple,
    grid::HDjl.FlowGeometries.Grids.AbstractGrid;
    solver::HDjl.AbstractPoissonSolver = HDjl.AutoSolver(),
    boundary::HDjl.FlowGeometries.Operators.AbstractBoundaryCondition = HDjl.FlowGeometries.Operators.Neumann(),
)
    D = length(velocity_fields)
    nd = ndims(velocity_fields[1])
    ustacked = stack(velocity_fields; dims = nd + 1)
    res = if grid isa HDjl.FlowGeometries.Grids.StructuredGrid
        concrete = HDjl.select_solver(solver, grid, boundary)
        concrete isa HDjl.CartesianSpectralSolver ?
            HDjl.helmholtz_decompose_spectral(ustacked, grid; solver = concrete) :
            HDjl.helmholtz_decompose(ustacked, grid; solver = concrete, boundary = boundary)
    else
        HDjl.helmholtz_decompose_spectral(ustacked, grid; solver = solver)
    end
    colons = ntuple(_ -> Colon(), Val(nd))
    res.u_rot .+= res.u_harm
    rot = ntuple(c -> view(res.u_rot, colons..., c), Val(D))
    div = ntuple(c -> view(res.u_div, colons..., c), Val(D))
    parts = decomp isa FIT.Types.HelmholtzDecomposition ? (; rotational = rot, divergent = div) :
            decomp isa FIT.Types.RotationalDecomposition ? (; rotational = rot) : (; divergent = div)
    return merge(parts, _solve_outcome(res))
end

# The least converged of the potential solves of `res`.
function _solve_outcome(res::HDjl.HelmholtzResult)
    solves = (res.χ_solve, res.rot_solve...)
    return (; converged = all(s -> s.converged, solves),
              iterations = maximum(s -> s.iterations, solves),
              residual = maximum(s -> s.residual, solves))
end

# Coordinate vectors state no topology, mask or geometry, and the solver is selected from all three, so
# the physical decomposition takes a grid. The spectral form below needs none of it.
function FIT.Decomposition._decompose_field_physical(
    decomp::_HDDecomp,
    velocity_fields::Tuple,
    coords_vecs::Tuple;
    kwargs...
)
    throw(ArgumentError(
        "physical-space $(nameof(typeof(decomp))) needs a `FlowGeometries` grid: the Poisson solve is " *
        "selected from the domain's topology, mask, axis spacing and geometry, and coordinate vectors " *
        "carry none of them. Build the grid those axes describe — e.g. " *
        "`FlowGeometries.Grids.StructuredGrid(FlowGeometries.Geometry.CartesianGeometry{Float64}(), x, y; " *
        "topology = (FlowGeometries.Grids.Periodic(), FlowGeometries.Grids.Periodic()))` for a periodic box " *
        "— and pass that."))
end

# 2. Spectral-space decomposition
function FIT.Decomposition._decompose_field_spectral(
    decomp::Union{FIT.Types.HelmholtzDecomposition, FIT.Types.RotationalDecomposition, FIT.Types.DivergentDecomposition},
    velocity_hat::AbstractArray{<:Complex},
    ks
)
    # HD's Leray projection on FIT's wavenumber axes, which it reads as derivative wavenumbers, zero at
    # an even axis's Nyquist slot as in FIT's own derivatives. A broadcast over component-last
    # `(ns..., D)` arrays, host or device.
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
