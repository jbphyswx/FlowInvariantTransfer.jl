module FlowInvariantTransferHelmholtzDecompositionExt

using HelmholtzDecomposition: HelmholtzDecomposition
using FlowInvariantTransfer: FlowInvariantTransfer as FET
using FlowInvariantTransfer.Types: AbstractFieldDecomposition, NoDecomposition, HelmholtzDecomposition as HelmholtzDecompType, RotationalDecomposition, DivergentDecomposition

# 1. Physical-space decomposition
function FET.Decomposition._decompose_field_physical(
    decomp::Union{HelmholtzDecompType, RotationalDecomposition, DivergentDecomposition},
    velocity_fields::Tuple,
    coords_vecs::Tuple;
    kwargs...
)
    u = velocity_fields[1]
    v = velocity_fields[2]
    FT = eltype(u)
    
    x_vec = coords_vecs[1]
    y_vec = coords_vecs[2]
    
    # Grid spacing
    dx = length(x_vec) > 1 ? FT((x_vec[end] - x_vec[begin]) / (length(x_vec) - 1)) : FT(1)
    dy = length(y_vec) > 1 ? FT((y_vec[end] - y_vec[begin]) / (length(y_vec) - 1)) : FT(1)
    
    # Build HelmholtzDecomposition structured grid
    geom = HelmholtzDecomposition.CartesianGeometry(dx, dy)
    mask = get(kwargs, :mask, nothing)
    grid = mask !== nothing ? HelmholtzDecomposition.StructuredGrid(geom, FT.(x_vec), FT.(y_vec), mask) :
                              HelmholtzDecomposition.StructuredGrid(geom, FT.(x_vec), FT.(y_vec))
                              
    # Decompose
    res = HelmholtzDecomposition.helmholtz_decompose(u, v, grid)

    # HelmholtzResult stores component-last stacked velocities (ns..., 2); this physical path is
    # contracted to return an (x, y) component tuple, so split them back out.
    rot = (res.u_rot[:, :, 1], res.u_rot[:, :, 2])
    div = (res.u_div[:, :, 1], res.u_div[:, :, 2])
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
function FET.Decomposition._decompose_field_spectral(
    decomp::Union{HelmholtzDecompType, RotationalDecomposition, DivergentDecomposition},
    velocity_hat::AbstractArray{<:Complex},
    ks
)
    D = size(velocity_hat, 3)
    D == 2 || throw(ArgumentError("Helmholtz decomposition currently supports 2D fields only."))
    
    FT = real(eltype(velocity_hat))
    
    # Find minimum non-zero absolute value in ks[1] to get dk_x
    min_kx = FT(Inf)
    for k in ks[1]
        ak = abs(k)
        if ak > 0
            min_kx = min(min_kx, ak)
        end
    end
    L_x = isfinite(min_kx) ? FT(2π / min_kx) : FT(1)
    dx = L_x / length(ks[1])
    
    min_ky = FT(Inf)
    for k in ks[2]
        ak = abs(k)
        if ak > 0
            min_ky = min(min_ky, ak)
        end
    end
    L_y = isfinite(min_ky) ? FT(2π / min_ky) : FT(1)
    dy = L_y / length(ks[2])
    
    # Construct structured grid
    geom = HelmholtzDecomposition.CartesianGeometry(dx, dy)
    grid = HelmholtzDecomposition.StructuredGrid(
        geom,
        collect(range(zero(FT), L_x; length=length(ks[1])+1)[1:end-1]),
        collect(range(zero(FT), L_y; length=length(ks[2])+1)[1:end-1])
    )
    
    u_hat = velocity_hat[:, :, 1]
    v_hat = velocity_hat[:, :, 2]
    res = HelmholtzDecomposition.helmholtz_project_spectral(u_hat, v_hat, grid)

    # SpectralCartesianResult already stores component-last stacked fields (ns..., 2) — exactly the
    # (ns..., D) convention FIT's spectral decompositions return, so use them directly.
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
function FET.Decomposition.helmholtz_project_spectral!(û_rot, û_div, velocity_hat, ks)
    return HelmholtzDecomposition.helmholtz_project_spectral!(û_rot, û_div, velocity_hat, ks)
end

end # module
