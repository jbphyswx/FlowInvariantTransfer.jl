module Filters

using ..Types: AbstractFilter, SharpSpectralFilter, GaussianFilter, TopHatFilter

export filter_response, apply_filter_spectral, apply_filter_spectral!

# ---------------------------------------------------------------------------
# Spectral response functions
# ---------------------------------------------------------------------------

"""
    filter_response(filter, k, ℓ) -> Real

Evaluate the spectral transfer function Ĝ(k, ℓ) of `filter` at wavenumber
magnitude `k` and filter scale `ℓ`.

The filter scale ℓ is defined so that the filter retains scales larger than ℓ.
"""
filter_response(::SharpSpectralFilter, k::Real, ℓ::Real) =
    abs(k) < π / ℓ ? one(typeof(k)) : zero(typeof(k))

filter_response(::GaussianFilter, k::Real, ℓ::Real) =
    exp(-k^2 * ℓ^2 / 24)

filter_response(::TopHatFilter, k::Real, ℓ::Real) =
    sinc(k * ℓ / (2π))   # Julia sinc is normalised: sinc(x) = sin(πx)/(πx)

# ---------------------------------------------------------------------------
# Spectral-domain application
# ---------------------------------------------------------------------------

"""
    apply_filter_spectral!(û_out, û_in, k_mag, filter, ℓ)

Apply `filter` at scale `ℓ` in spectral space by pointwise multiplication:
  û_out[I] = Ĝ(|k[I]|, ℓ) * û_in[I].

# Arguments
- `û_out`: Output spectral array (same shape as `û_in`).
- `û_in`: Input spectral array (complex or real).
- `k_mag`: Array of wavenumber magnitudes (same shape as `û_in`).
- `filter::AbstractFilter`: Filter kernel.
- `ℓ::Real`: Filter scale.

Modifies `û_out` in-place and returns it.
"""
function apply_filter_spectral!(
    û_out::AbstractArray,
    û_in::AbstractArray,
    k_mag::AbstractArray,
    filter::AbstractFilter,
    ℓ::Real,
)
    size(û_out) == size(û_in) == size(k_mag) ||
        throw(DimensionMismatch("û_out, û_in, and k_mag must have identical sizes."))
    @inbounds for I in eachindex(û_in)
        û_out[I] = filter_response(filter, k_mag[I], ℓ) * û_in[I]
    end
    return û_out
end

"""
    apply_filter_spectral(û_in, k_mag, filter, ℓ) -> Array

Non-mutating version of `apply_filter_spectral!`.
"""
function apply_filter_spectral(
    û_in::AbstractArray,
    k_mag::AbstractArray,
    filter::AbstractFilter,
    ℓ::Real,
)
    û_out = similar(û_in)
    return apply_filter_spectral!(û_out, û_in, k_mag, filter, ℓ)
end

end # module Filters
