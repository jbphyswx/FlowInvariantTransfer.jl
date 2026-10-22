module Utils

using FlowTransformBindings: FlowTransformBindings as FTB

export wavenumber_grid, wavenumber_magnitude_grid
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
    wavenumber_grid(ns, Ls; real=false) -> NTuple{D, AbstractWavenumberAxis}

The physical wavenumber axes of an `ns`-point periodic grid of side lengths `Ls`, `k = 2πm/L`, in the
FFTW `fftfreq` order (`m = 0, 1, …, ⌊(N−1)/2⌋, −⌊N/2⌋, …, −1`). With `real = true` the first axis is
the `rfftfreq` half `m = 0…N₁÷2` — the layout of a real field's `rfft` — and the remaining axes stay
`fftfreq`. The axes are `FlowTransformBindings.RFFTAxis` / `FlowTransformBindings.FFTAxis`: no
wavenumber array is stored, and they carry the field's spectral layout for every diagnostic.
"""
wavenumber_grid(ns::NTuple{D,Int}, Ls::NTuple{D,Real}; real::Bool = false) where {D} =
    real ? FTB.wavenumber_axes(Ls, ns, Val(true)) : FTB.wavenumber_axes(Ls, ns, Val(false))

"""
    wavenumber_magnitude_grid(ks) -> Array{T, D}

The isotropic wavenumber magnitude `|k|` at every stored mode of the layout `ks`.

# Arguments
- `ks::NTuple{D, AbstractVector}`: Tuple of 1D wavenumber vectors (e.g., from `wavenumber_grid`).

# Returns
`D`-dimensional array of the coefficient shape of `ks` (the halved first axis included), with entry
`[i₁,…,i_D] = sqrt(ks[1][i₁]² + … + ks[D][i_D]²)` and element type `T`, the promotion of the axes'
element types.
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
    validate_uniform_grid(coords_vecs; tol = nothing) -> NTuple{D}

The spacing `Δ = (x_N − x_1)/(N − 1)` of each coordinate vector, in its float type, after checking
that every step equals `Δ` to `tol` relative to the coordinates' magnitude `max(|x_1|, |x_N|, |Δ|)`
(`tol = nothing`: `√eps` of the coordinates' float type). A vector of fewer than two points has spacing
zero.

Throws `ArgumentError` if a step differs from `Δ` by more than that.
"""
function validate_uniform_grid(coords_vecs::NTuple{D, <:AbstractVector}; tol::Union{Nothing, Real} = nothing) where {D}
    return ntuple(Val(D)) do d
        cv = coords_vecs[d]
        FT = float(eltype(cv))
        length(cv) < 2 && return zero(FT)
        Δ = FT(cv[end] - cv[begin]) / (length(cv) - 1)
        rtol = tol === nothing ? sqrt(eps(FT)) : FT(tol)
        bound = rtol * max(abs(FT(cv[begin])), abs(FT(cv[end])), abs(Δ))
        for i in 2:length(cv)
            abs(FT(cv[i] - cv[i-1]) - Δ) > bound && throw(ArgumentError(
                "coordinate direction $d is not uniform: step $(cv[i] - cv[i-1]) at index $i against the " *
                "mean spacing $Δ (tolerance $bound)."))
        end
        return Δ
    end
end

"""
    domain_size_from_coords(coords_vecs) -> NTuple{D}

The period `L = NΔ` of each coordinate vector read as `N` periodic samples `x_1, x_1 + Δ, …,
x_1 + (N−1)Δ`, the last one step short of `x_1 + L`, in the coordinates' float type. A vector that
ends at `x_1 + L` repeats the first sample and does not meet this contract. A vector of fewer than two
points has period 1.
"""
function domain_size_from_coords(coords_vecs::NTuple{D, <:AbstractVector}) where {D}
    return ntuple(Val(D)) do d
        cv = coords_vecs[d]
        FT = float(eltype(cv))
        N = length(cv)
        N < 2 && return one(FT)
        return N * (FT(cv[end] - cv[begin]) / (N - 1))
    end
end

end # module Utils
