module FlowInvariantTransferFINUFFTExt

using FINUFFT: FINUFFT
using FlowInvariantTransfer: FlowInvariantTransfer as FET
using FlowInvariantTransfer.Types: AbstractFilter, CoarseGrainingFluxResult, CoarseGrainingFluxResultWithDiagnostics
using FlowInvariantTransfer.Filters: filter_response
using FlowInvariantTransfer.Utils: wavenumber_magnitude_grid

export nufft_coarse_graining_flux

# ---------------------------------------------------------------------------
# Non-uniform coarse-graining flux via FINUFFT type-1/type-2 round-trips
# ---------------------------------------------------------------------------

"""
    nufft_coarse_graining_flux(velocity_fields, scatter_coords, ℓ, filter, ms;
                                return_diagnostics=false, tol=1e-8)
                                -> CoarseGrainingFluxResult

Compute the coarse-graining energy flux Π_ℓ(x) at scattered (non-uniform) Cartesian
points using FINUFFT for the spectral transforms.

# Arguments
- `velocity_fields`: Tuple of D real-valued vectors of length N — velocity at scattered points.
- `scatter_coords`: Tuple of D real-valued vectors of length N — spatial coordinates.
- `ℓ::Real`: Filter scale.
- `filter::AbstractFilter`: Filter kernel.
- `ms::NTuple{D,Int}`: Target spectral grid size for intermediate transforms.

# Keyword Arguments
- `return_diagnostics::Bool=false`: If `true`, also return τ̄ᵢⱼ and S̄ᵢⱼ at the output points.
- `tol::Float64=1e-8`: FINUFFT accuracy tolerance.

# Returns
`CoarseGrainingFluxResult` with `flux_field` sampled at the input scatter coordinates.

# Notes
This implements the approach:
  1. Type-1 NUFFT: scattered → uniform spectral grid
  2. Spectral filter application in Fourier space
  3. Type-2 NUFFT: uniform spectral grid → scattered points
  4. Compute SFS stress and strain rate, contract to Π_ℓ

For 2D inputs, uses `nufft2d1`/`nufft2d2`; for 3D, `nufft3d1`/`nufft3d2`.
"""
function nufft_coarse_graining_flux(
    velocity_fields::Tuple,
    scatter_coords::Tuple,
    ℓ::Real,
    filter::AbstractFilter,
    ms::Tuple;
    return_diagnostics::Bool = false,
    tol::Float64 = 1e-8,
)
    D  = length(velocity_fields)
    nd = length(scatter_coords)
    D == nd || throw(ArgumentError("velocity components ($D) ≠ spatial dimensions ($nd)"))
    N  = length(velocity_fields[1])
    FT = Float64

    # Build uniform wavenumber grid for the spectral representation
    # Infer domain size from coordinate ranges
    Ls = ntuple(Val(nd)) do d
        cv = scatter_coords[d]
        range = maximum(cv) - minimum(cv)
        range > 0 ? range : 1.0
    end
    ks_1d = ntuple(Val(nd)) do d
        N_d = ms[d]
        dk  = 2π / Ls[d]
        [Float64(k <= N_d÷2 ? k : k - N_d) * dk for k in 0:N_d-1]
    end
    k_mag = wavenumber_magnitude_grid(ks_1d)

    # Rescale coordinates to [-π, π) for FINUFFT
    scaled_coords = ntuple(Val(nd)) do d
        cmin = minimum(scatter_coords[d])
        cmax = maximum(scatter_coords[d])
        range = cmax - cmin
        range > 0 ? (scatter_coords[d] .- cmin) ./ range .* 2π .- π :
                    zeros(eltype(scatter_coords[d]), N)
    end

    # Type-1 NUFFT: scattered → spectral (analyse each velocity component)
    û = _nufft_type1(scaled_coords, velocity_fields, ms, tol)

    # Filter weights Ĝ(k)
    Ĝ = [FT(filter_response(filter, k_mag[I], Float64(ℓ))) for I in CartesianIndices(size(k_mag))]

    # Filtered velocity at scattered points via Type-2 NUFFT
    û_filt = [Ĝ .* û[c] for c in 1:D]
    u_filt_scattered = _nufft_type2(scaled_coords, û_filt, ms, tol)

    # Filtered cross-products [uᵢuⱼ]̄ at scattered points
    ij_pairs = [(i, j) for i in 1:D for j in i:D]
    uu_filt_scattered = Dict{Tuple{Int,Int}, Vector{FT}}()
    for (i, j) in ij_pairs
        prod_ij = velocity_fields[i] .* velocity_fields[j]
        û_prod = _nufft_type1(scaled_coords, (prod_ij,), ms, tol)
        û_prod_filt = [Ĝ .* û_prod[1]]
        uu_filt_scattered[(i,j)] = real.(_nufft_type2(scaled_coords, û_prod_filt, ms, tol)[1])
    end

    # SFS stress τ̄ᵢⱼ = [uᵢuⱼ]̄ − ūᵢ ūⱼ  (at scattered points)
    τ_scattered = Dict{Tuple{Int,Int}, Vector{FT}}()
    for (i, j) in ij_pairs
        τ_scattered[(i,j)] = uu_filt_scattered[(i,j)] .-
                              real.(u_filt_scattered[i]) .* real.(u_filt_scattered[j])
    end

    # Strain rate S̄ᵢⱼ via spectral derivative and Type-2 NUFFT
    # ∂ūᵢ/∂xⱼ(x_p) = NUFFT_type2(i·kⱼ·û_filt_i)
    k_comp_grids = [_build_k_component_nufft(ks_1d, d, ms) for d in 1:nd]

    S̄_scattered = Dict{Tuple{Int,Int}, Vector{FT}}()
    for (i, j) in ij_pairs
        grad_ij_spec = [im .* k_comp_grids[j] .* û_filt[i]]
        dui_dxj_scattered = real.(_nufft_type2(scaled_coords, grad_ij_spec, ms, tol)[1])
        if i == j
            S̄_scattered[(i,j)] = dui_dxj_scattered
        else
            grad_ji_spec = [im .* k_comp_grids[i] .* û_filt[j]]
            duj_dxi_scattered = real.(_nufft_type2(scaled_coords, grad_ji_spec, ms, tol)[1])
            S̄_scattered[(i,j)] = FT(0.5) .* (dui_dxj_scattered .+ duj_dxi_scattered)
        end
    end

    # Flux Π_ℓ = −Σᵢⱼ factor·τᵢⱼ·S̄ᵢⱼ  (at scattered points)
    Π = zeros(FT, N)
    for (i, j) in ij_pairs
        factor = i == j ? FT(1) : FT(2)
        @. Π -= factor * τ_scattered[(i,j)] * S̄_scattered[(i,j)]
    end

    mean_Π = FT(sum(Π) / N)

    if return_diagnostics
        τ_arr = zeros(FT, N, D, D)
        S_arr = zeros(FT, N, D, D)
        for i in 1:D, j in 1:D
            key = i <= j ? (i, j) : (j, i)
            τ_arr[:, i, j] .= τ_scattered[key]
            S_arr[:, i, j] .= S̄_scattered[key]
        end
        return CoarseGrainingFluxResultWithDiagnostics(FT(ℓ), Π, mean_Π, τ_arr, S_arr)
    else
        return CoarseGrainingFluxResult(FT(ℓ), Π, mean_Π)
    end
end

# ---------------------------------------------------------------------------
# NUFFT helpers
# ---------------------------------------------------------------------------

function _nufft_type1(scaled_coords::Tuple, fields::Tuple, ms::Tuple, tol::Float64)
    nd = length(scaled_coords)
    D  = length(fields)
    N  = length(fields[1])
    Np = prod(ms)
    result = Vector{Array{ComplexF64}}(undef, D)

    for c in 1:D
        f_c = ComplexF64.(fields[c])
        if nd == 1
            raw = FINUFFT.nufft1d1(
                scaled_coords[1], f_c, 1, tol, ms[1])
            result[c] = raw ./ N
        elseif nd == 2
            raw = FINUFFT.nufft2d1(
                scaled_coords[1], scaled_coords[2], f_c, 1, tol, ms[1], ms[2])
            result[c] = raw ./ N
        elseif nd == 3
            raw = FINUFFT.nufft3d1(
                scaled_coords[1], scaled_coords[2], scaled_coords[3],
                f_c, 1, tol, ms[1], ms[2], ms[3])
            result[c] = raw ./ N
        else
            throw(ArgumentError("FINUFFT supports 1D, 2D, 3D only; got nd=$nd."))
        end
    end
    return result
end

function _nufft_type2(scaled_coords::Tuple, û_list::Vector, ms::Tuple, tol::Float64)
    nd = length(scaled_coords)
    D  = length(û_list)
    result = Vector{Vector{ComplexF64}}(undef, D)

    for c in 1:D
        coeff_c = ComplexF64.(û_list[c])
        if nd == 1
            result[c] = FINUFFT.nufft1d2(scaled_coords[1], 1, tol, coeff_c)
        elseif nd == 2
            result[c] = FINUFFT.nufft2d2(
                scaled_coords[1], scaled_coords[2], 1, tol, coeff_c)
        elseif nd == 3
            result[c] = FINUFFT.nufft3d2(
                scaled_coords[1], scaled_coords[2], scaled_coords[3],
                1, tol, coeff_c)
        else
            throw(ArgumentError("FINUFFT supports 1D, 2D, 3D only; got nd=$nd."))
        end
    end
    return result
end

function _build_k_component_nufft(ks_1d::Tuple, d::Int, ms::Tuple)
    nd = length(ms)
    kc = zeros(Float64, ms...)
    for I in CartesianIndices(ms)
        kc[I] = ks_1d[d][I[d]]
    end
    return kc
end

end # module FlowInvariantTransferFINUFFTExt
