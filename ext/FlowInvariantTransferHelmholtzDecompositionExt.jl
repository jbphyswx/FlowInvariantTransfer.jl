module FlowInvariantTransferHelmholtzDecompositionExt

using HelmholtzDecomposition: HelmholtzDecomposition
using FlowInvariantTransfer: FlowInvariantTransfer as FIT
using FlowInvariantTransfer.Types: AbstractFieldDecomposition, NoDecomposition, HelmholtzDecomposition as HelmholtzDecompType, RotationalDecomposition, DivergentDecomposition

# 1. Physical-space decomposition
function FIT.Decomposition._decompose_field_physical(
    decomp::Union{HelmholtzDecompType, RotationalDecomposition, DivergentDecomposition},
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
    FT = eltype(velocity_fields[1])

    # Per-dimension grid spacing → HelmholtzDecomposition structured grid (2D or 3D).
    dx = ntuple(nd) do i
        c = coords_vecs[i]
        length(c) > 1 ? FT((c[end] - c[begin]) / (length(c) - 1)) : FT(1)
    end
    geom   = HelmholtzDecomposition.CartesianGeometry(dx...)
    coords = ntuple(i -> FT.(coords_vecs[i]), nd)
    mask   = get(kwargs, :mask, nothing)
    grid = mask !== nothing ? HelmholtzDecomposition.StructuredGrid(geom, coords..., mask) :
                              HelmholtzDecomposition.StructuredGrid(geom, coords...)

    # `HelmholtzResult` stores component-last stacked fields (ns..., D); split into a component tuple.
    # NB: bounded physical-space Helmholtz–Hodge also has a harmonic part (`res.u_harm`); the rot/div
    # tuple returned here omits it (harmonic ≈ 0 on a periodic domain), matching the spectral convention.
    res    = HelmholtzDecomposition.helmholtz_decompose(velocity_fields..., grid)
    colons = ntuple(_ -> Colon(), nd)
    rot = ntuple(c -> res.u_rot[colons..., c], nd)
    div = ntuple(c -> res.u_div[colons..., c], nd)
    if decomp isa HelmholtzDecompType
        return (; rotational = rot, divergent = div)
    elseif decomp isa RotationalDecomposition
        return rot
    elseif decomp isa DivergentDecomposition
        return div
    else
        throw(ArgumentError("Unknown decomposition type: $decomp"))
    end
end

# 2. Spectral-space decomposition
function FIT.Decomposition._decompose_field_spectral(
    decomp::Union{HelmholtzDecompType, RotationalDecomposition, DivergentDecomposition},
    velocity_hat::AbstractArray{<:Complex},
    ks
)
    # N-D Leray projection straight from the wavenumber vectors (no grid reconstruction): the sibling's
    # `helmholtz_project_spectral(velocity_hat, ks)` is a device-generic pure-broadcast projection, so
    # this works in 2D and 3D and on device arrays. Component-last (ns..., D) convention both ways.
    res = HelmholtzDecomposition.helmholtz_project_spectral(velocity_hat, ks)

    # SpectralCartesianResult stores component-last stacked fields (ns..., D) — exactly the FIT
    # spectral-decomposition convention, so use them directly.
    if decomp isa HelmholtzDecompType
        return (; rotational = res.u_rot, divergent = res.u_div)
    elseif decomp isa RotationalDecomposition
        return res.u_rot
    elseif decomp isa DivergentDecomposition
        return res.u_div
    else
        throw(ArgumentError("Unknown decomposition type: $decomp"))
    end
end

# 3. Direct spectral project method override
function FIT.Decomposition.helmholtz_project_spectral!(û_rot, û_div, velocity_hat, ks)
    return HelmholtzDecomposition.helmholtz_project_spectral!(û_rot, û_div, velocity_hat, ks)
end

end # module
