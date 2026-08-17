module FlowInvariantTransferHelmholtzDecompositionExt

using HelmholtzDecomposition: HelmholtzDecomposition as HDjl
using FlowInvariantTransfer: FlowInvariantTransfer as FIT

# 1. Physical-space decomposition
function FIT.Decomposition._decompose_field_physical(
    decomp::Union{FIT.Types.HelmholtzDecomposition, FIT.Types.RotationalDecomposition, FIT.Types.DivergentDecomposition},
    velocity_fields::Tuple,
    coords_vecs::Tuple;
    kwargs...
)
    D  = length(velocity_fields)
    nd = length(coords_vecs)
    D == nd || throw(ArgumentError(
        "number of velocity components ($D) must equal spatial dimensions ($nd)"))
    (nd == 2 || nd == 3) || throw(ArgumentError(
        "physical-space Helmholtz decomposition supports 2D and 3D Cartesian grids (nd=$nd)."))
    FT = float(real(eltype(velocity_fields[1])))

    # FlowGeometries Cartesian grid from the coordinate axes (default topology — works for any grid). Pass
    # the axes as given: `_to_axis` adapts them to `FT` while preserving a uniform range's uniformity.
    # `mask` selects active cells.
    geom = HDjl.FlowGeometries.Geometry.CartesianGeometry{FT}()
    mask = get(kwargs, :mask, nothing)
    grid = HDjl.FlowGeometries.Grids.StructuredGrid(geom, coords_vecs...; mask = mask)

    # `helmholtz_decompose` takes ONE component-last `(ns…, D)` array and returns the 3-way Hodge split
    # (`u_rot + u_div + u_harm`, all `(ns…, D)`). A uniform grid gets HD's fast spectral (FFT) Poisson
    # solver via the default AutoSolver; a stretched (nonuniform) grid has no spectral solver, so use HD's
    # grid-agnostic CG solver explicitly. Fold the harmonic (constant/mean, divergence-free) into the
    # rotational component so rotational + divergent == the field — matching the spectral path.
    ustacked = stack(velocity_fields; dims = nd + 1)
    res = HDjl.FlowGeometries.Grids.isuniform(grid) ?
          HDjl.helmholtz_decompose(ustacked, grid) :
          HDjl.helmholtz_decompose(ustacked, grid; solver = HDjl.CGSolver())
    colons = ntuple(_ -> Colon(), nd)
    rot = ntuple(c -> res.u_rot[colons..., c] .+ res.u_harm[colons..., c], nd)
    div = ntuple(c -> res.u_div[colons..., c], nd)
    if decomp isa FIT.Types.HelmholtzDecomposition
        return (; rotational = rot, divergent = div)
    elseif decomp isa FIT.Types.RotationalDecomposition
        return rot
    elseif decomp isa FIT.Types.DivergentDecomposition
        return div
    else
        throw(ArgumentError("Unknown decomposition type: $decomp"))
    end
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
