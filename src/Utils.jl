module Utils

using LinearAlgebra: LinearAlgebra as LA

export wavenumber_grid, wavenumber_magnitude_grid, dealiasing_mask, dealiasing_mask!
export validate_velocity_input, validate_uniform_grid, domain_size_from_coords
export as_component_field

# ---------------------------------------------------------------------------
# Component-axis normalization
# ---------------------------------------------------------------------------

"""
    as_component_field(f, nd) -> array of rank nd+1

Normalize a field to the package's `(ns..., M)` component-axis convention: a rank-`nd` array
`(ns...)` (e.g. a scalar) is reshaped to `(ns..., 1)`; a rank-`nd+1` array is returned unchanged.
"""
function as_component_field(f, nd::Int)
    r = ndims(f)
    r == nd     && return reshape(f, size(f)..., 1)
    r == nd + 1 && return f
    throw(ArgumentError(
        "field has $r dims; expected nd=$nd (shape (ns...)) or nd+1=$(nd+1) (shape (ns...,1))."))
end

# ---------------------------------------------------------------------------
# Wavenumber grid construction (matches FFTW fftfreq convention)
# ---------------------------------------------------------------------------

"""
    wavenumber_grid(ns, Ls) -> NTuple{D, Vector{Float64}}

Return a tuple of 1D physical-wavenumber vectors matching the FFTW `fftfreq`
convention (centered at zero after fftshift).

# Arguments
- `ns::NTuple{D,Int}`: Number of grid points along each dimension.
- `Ls::NTuple{D,Float64}`: Physical domain size along each dimension.

# Returns
Tuple of length D; element d is the range
  k_d ∈ [−⌊N_d/2⌋, ⌊(N_d−1)/2⌋] × (2π / L_d).

# Example
```julia
ks = wavenumber_grid((16, 16), (2π, 2π))
# ks[1] and ks[2] are both [-8, -7, ..., 7] * (2π/2π)
```
"""
function wavenumber_grid(ns::NTuple{D,Int}, Ls::NTuple{D}) where {D}
    FT = promote_type(map(x -> typeof(float(x)), Ls)...)
    return ntuple(Val(D)) do d
        N  = ns[d]
        dk = FT(2π) / FT(Ls[d])
        ks = zeros(FT, N)
        for i in 1:N
            k_idx = i - 1
            ks[i] = k_idx <= N ÷ 2 ? k_idx * dk : (k_idx - N) * dk
        end
        return ks
    end
end

"""
    wavenumber_magnitude_grid(ks) -> Array{Float64, D}

Compute the isotropic wavenumber magnitude |k| at every grid point.

# Arguments
- `ks::NTuple{D, AbstractVector}`: Tuple of 1D wavenumber vectors (e.g., from `wavenumber_grid`).

# Returns
`D`-dimensional array of the same size as the full grid, with entry `[i₁,…,i_D] = sqrt(ks[1][i₁]² + … + ks[D][i_D]²)`.
"""
function wavenumber_magnitude_grid(ks::NTuple{D, <:AbstractVector}) where {D}
    FT    = promote_type(map(eltype, ks)...)
    sizes = ntuple(d -> length(ks[d]), Val(D))
    k_mag = Array{FT, D}(undef, sizes...)
    for I in CartesianIndices(sizes)
        s = zero(FT)
        for d in 1:D
            s += ks[d][I[d]]^2
        end
        k_mag[I] = sqrt(s)
    end
    return k_mag
end

# ---------------------------------------------------------------------------
# Dealiasing
# ---------------------------------------------------------------------------

"""
    dealiasing_mask(ns; rule=:twothirds) -> BitArray{D}

Build a spectral dealiasing mask.

# Arguments
- `ns::NTuple{D,Int}`: Grid sizes.
- `rule::Symbol`: `:twothirds` (2/3 rule) or `:half`.

# Returns
`BitArray` of the same shape as the spectral grid; `true` where the mode is
*kept* (i.e., |k_d| < N_d/2 * threshold for all d).

# Notes
For the 2/3 rule, modes with |k_d| ≥ N_d/3 along any dimension are zeroed.
"""
function dealiasing_mask(ns::NTuple{D,Int}; rule::Symbol = :twothirds) where {D}
    mask = trues(ns...)
    dealiasing_mask!(mask, ns; rule=rule)
    return mask
end

function dealiasing_mask!(mask::AbstractArray{Bool}, ns::NTuple{D,Int}; rule::Symbol = :twothirds) where {D}
    threshold = rule === :twothirds ? 1.0/3.0 : 0.5
    for I in CartesianIndices(ns)
        keep = true
        for d in 1:D
            # FFTW-order index (0-based)
            idx0 = I[d] - 1
            k_abs = idx0 <= ns[d] ÷ 2 ? idx0 : ns[d] - idx0
            if k_abs >= ns[d] * threshold
                keep = false
                break
            end
        end
        mask[I] = keep
    end
    return mask
end

# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

"""
    validate_velocity_input(velocity_components, dims)

Check that all velocity components have the same size and that it is consistent
with `dims` dimensions.

Throws `DimensionMismatch` with a descriptive message on failure.
"""
function validate_velocity_input(velocity_components::Tuple, dims::Int)
    D = length(velocity_components)
    D >= 1 || throw(ArgumentError("velocity_components must contain at least 1 array."))
    ref_size = size(velocity_components[1])
    ndims(velocity_components[1]) == dims || throw(DimensionMismatch(
        "Velocity component has $(ndims(velocity_components[1])) dimensions; expected $dims."))
    for c in 2:D
        size(velocity_components[c]) == ref_size || throw(DimensionMismatch(
            "Velocity component $c has size $(size(velocity_components[c])); expected $ref_size."))
    end
    return ref_size
end

"""
    validate_uniform_grid(coords_vecs; tol=1e-10)

Check that each coordinate vector represents a uniform (equi-spaced) grid,
returning the spacing tuple.

Throws `ArgumentError` if any dimension is non-uniform beyond `tol`.
"""
function validate_uniform_grid(coords_vecs::NTuple{D, <:AbstractVector}; tol::Float64 = 1e-10) where {D}
    spacings = ntuple(Val(D)) do d
        cv = coords_vecs[d]
        length(cv) < 2 && return 0.0
        Δ = (cv[end] - cv[begin]) / (length(cv) - 1)
        for i in 2:length(cv)
            abs((cv[i] - cv[i-1]) - Δ) > tol * abs(Δ) + tol &&
                throw(ArgumentError("Coordinate dimension $d is not uniform (spacing varies beyond tol=$tol)."))
        end
        return Δ
    end
    return spacings
end

"""
    domain_size_from_coords(coords_vecs) -> NTuple{D, Float64}

Infer the physical domain size from coordinate vectors, assuming uniform periodic grids
where spacing Δ = (max − min) / N (so domain = N * Δ).
"""
function domain_size_from_coords(coords_vecs::NTuple{D, <:AbstractVector}) where {D}
    return ntuple(Val(D)) do d
        cv = coords_vecs[d]
        N = length(cv)
        N < 2 && return 1.0
        Δ = (cv[end] - cv[begin]) / (N - 1)
        return N * Δ
    end
end

end # module Utils
