module SpectralLayout

export AbstractWavenumberAxis, FullAxis, HalfAxis
export full_length, is_half, full_size, spectral_size, max_abs
export hermitian_weight, hermitian_weights
export axis_index_wavenumber, is_dealiased, is_nyquist, wavenumber_arrays, derivative_wavenumber
export dealias_factors

"""
    AbstractWavenumberAxis{T} <: AbstractVector{T}

An analytic wavenumber axis: `k_i` is computed from the index, nothing is stored beyond the full
grid length `n` and the spacing `dk = 2π/L`. Carries the spectral layout of the field it indexes —
see [`FullAxis`](@ref) and [`HalfAxis`](@ref).
"""
abstract type AbstractWavenumberAxis{T<:Real} <: AbstractVector{T} end

"""
    FullAxis(n, dk)

The `fftfreq` axis of an `n`-point grid: `k_i = dk·m`, `m = i−1` for `2(i−1) < n`, else `i−1−n`.
Length `n`. A field indexed by `FullAxis` on every dimension holds the full complex spectrum.
"""
struct FullAxis{T} <: AbstractWavenumberAxis{T}
    n::Int
    dk::T
end

"""
    HalfAxis(n, dk)

The `rfftfreq` axis of an `n`-point grid: `k_i = dk·(i−1)`, `i = 1…n÷2+1`. Length `n÷2+1`, and `n`
itself is kept because it is not recoverable from the length (`n = 2m−2` and `n = 2m−1` give the
same half). A field whose first axis is a `HalfAxis` is the real-to-complex transform of a real
field: it stores the non-redundant half `k₁ ≥ 0` and its conjugate half is implied by
`û(−k) = conj(û(k))`. Only the first axis is ever a `HalfAxis`.
"""
struct HalfAxis{T} <: AbstractWavenumberAxis{T}
    n::Int
    dk::T
end

Base.size(a::FullAxis) = (a.n,)
Base.size(a::HalfAxis) = (a.n ÷ 2 + 1,)
Base.IndexStyle(::Type{<:AbstractWavenumberAxis}) = IndexLinear()

Base.@propagate_inbounds function Base.getindex(a::FullAxis, i::Int)
    @boundscheck checkbounds(a, i)
    m = i - 1
    return (2m < a.n ? m : m - a.n) * a.dk
end

Base.@propagate_inbounds function Base.getindex(a::HalfAxis, i::Int)
    @boundscheck checkbounds(a, i)
    return (i - 1) * a.dk
end

Base.show(io::IO, a::FullAxis) = print(io, "FullAxis(n=", a.n, ", dk=", a.dk, ")")
Base.show(io::IO, a::HalfAxis) = print(io, "HalfAxis(n=", a.n, ", dk=", a.dk, ")")
Base.show(io::IO, ::MIME"text/plain", a::AbstractWavenumberAxis) = show(io, a)

"""
    full_length(axis) -> Int

Number of points of the physical grid along this axis: `n` for a wavenumber axis, `length` for any
other vector (a plain vector of wavenumbers is always a full `fftfreq` axis).
"""
full_length(a::AbstractWavenumberAxis) = a.n
full_length(a::AbstractVector) = length(a)

"""
    is_half(ks) -> Bool

Whether the wavenumber tuple `ks` indexes a half-spectrum (real-field) layout.
"""
is_half(ks::Tuple) = first(ks) isa HalfAxis

"""
    full_size(ks) -> NTuple{nd,Int}

The physical grid size `ns` the field lives on, whatever its spectral layout.
"""
full_size(ks::Tuple) = map(full_length, ks)

"""
    spectral_size(ks) -> NTuple{nd,Int}

The shape of the coefficient array's spatial axes: `full_size` for a full layout, `(ns₁÷2+1, ns₂, …)`
for a half layout.
"""
spectral_size(ks::Tuple) = map(length, ks)

"""
    max_abs(axis) -> Real

`maximum(abs, axis)` in closed form. Equal for the full and half axis of the same grid, so shell
edges built from it are layout-independent.
"""
max_abs(a::AbstractWavenumberAxis) = (a.n ÷ 2) * a.dk
max_abs(a::AbstractVector) = maximum(abs, a)

"""
    hermitian_weight(ks, I) -> Int

Weight of coefficient `I` when a sum over the full spectrum of a function even under `k ↦ −k` is
taken over the half layout instead: `Σ_full f = Σ_half w·f`. Every transfer density in this package
is even (`û(−k) = conj(û(k))` and `N̂(−k) = conj(N̂(k))` for real fields), as is `|û|²`.

The `k₁ = 0` plane is stored whole and is mapped onto itself by `k ↦ −k`, so every mode on it already
has its mirror in the half array: weight 1. For even `n₁` the same holds for the Nyquist plane
`k₁ = n₁/2 ≡ −n₁/2`: weight 1. Every other stored mode stands in for itself and its unstored mirror:
weight 2. On a full layout the weight is 1 everywhere.
"""
@inline function hermitian_weight(ks::Tuple, I::CartesianIndex)
    a = first(ks)
    a isa HalfAxis || return 1
    i1 = I[1]
    return (i1 == 1 || (iseven(a.n) && i1 == a.n ÷ 2 + 1)) ? 1 : 2
end

"""
    hermitian_weights(FT, ks) -> AbstractArray{FT}

[`hermitian_weight`](@ref) as a length-`length(ks[1])` vector of `FT`, reshaped to broadcast along
axis 1 of the coefficient array (a `1`-filled `(1,)` for a full layout).
"""
function hermitian_weights(::Type{FT}, ks::Tuple) where {FT}
    a = first(ks)
    nd = length(ks)
    if a isa HalfAxis
        w = FT[hermitian_weight(ks, CartesianIndex(ntuple(d -> d == 1 ? i : 1, nd))) for i in 1:length(a)]
    else
        w = FT[1]
    end
    return reshape(w, ntuple(d -> d == 1 ? length(w) : 1, nd))
end

# ---------------------------------------------------------------------------
# Separable per-axis quantities.
#
# The wavenumber components and the Orszag 2/3 keep-mask are both separable — `k_j` depends on the
# index along axis `j` alone, and the mask is a product of per-axis predicates. Kept as `nd` length-`m_d`
# arrays reshaped to broadcast along their own axis, so what a dense `(ns…)` grid per component would
# hold is `Σ_d m_d` numbers. A broadcast over them fuses into the surrounding expression with no
# intermediate.
# ---------------------------------------------------------------------------

"""
    axis_index_wavenumber(axis, i) -> Int

The signed integer wavenumber (in units of `dk`) at index `i`. `i−1` on a half axis, the folded
fftfreq integer on a full one.
"""
@inline axis_index_wavenumber(a::HalfAxis, i::Integer) = i - 1
@inline axis_index_wavenumber(a::FullAxis, i::Integer) = (2(i - 1) < a.n ? i - 1 : i - 1 - a.n)
@inline function axis_index_wavenumber(a::AbstractVector, i::Integer)
    n = length(a)
    return 2(i - 1) < n ? i - 1 : i - 1 - n
end

"""
    is_dealiased(ks, I) -> Bool

`true` if coefficient `I` lies in the Orszag 2/3 discard band (`|k_d| ≥ n_d/3` along any axis `d`),
for either spectral layout.
"""
@inline function is_dealiased(ks::Tuple, I::CartesianIndex{nd}) where {nd}
    @inbounds for d in 1:nd
        a = ks[d]
        abs(axis_index_wavenumber(a, I[d])) >= full_length(a) ÷ 3 && return true
    end
    return false
end

"""
    wavenumber_arrays(proto, FT, ks) -> Vector

`nd` arrays of `k_d` values, each of length `length(ks[d])` and reshaped to broadcast along axis `d`,
built in `proto`'s array type (device-resident for a device field). A `Vector` (not a tuple) so the
hot loop's `kg[j]` with a runtime `j` stays inferred; every element has the same concrete type.
"""
function wavenumber_arrays(proto::AbstractArray, ::Type{FT}, ks::Tuple; derivative::Bool = false) where {FT}
    nd = length(ks)
    return [begin
        m = length(ks[d])
        h = derivative ? FT[derivative_wavenumber(ks[d], i) for i in 1:m] : collect(FT, ks[d])
        v = similar(proto, FT, m)
        copyto!(v, h)
        reshape(v, ntuple(i -> i == d ? m : 1, nd))
    end for d in 1:nd]
end

"""
    is_nyquist(axis, i) -> Bool

Whether index `i` addresses the Nyquist mode of an even-length axis. That slot holds `+n/2` and
`−n/2` at once — it is its own image under `k ↦ −k`. Every operation that distinguishes a mode from
its mirror treats it separately: see [`derivative_wavenumber`](@ref).
"""
@inline function is_nyquist(a, i::Integer)
    n = full_length(a)
    return iseven(n) && abs(axis_index_wavenumber(a, i)) == n ÷ 2
end

"""
    derivative_wavenumber(axis, i) -> Real

The wavenumber to use when `i` indexes a *derivative* of a real field: `axis[i]`, and zero at the
Nyquist mode of an even axis: a self-mirrored slot
admits no non-zero derivative, and the grid derivative of `cos(n·x/2)` is identically zero.

Applies to every first-order spectral operator on a real field: `∂_d`, and the vorticity `i k × û`
that helicity and enstrophy are built from. An operator carrying two factors of `k` (enstrophy's
`conj(ω̂)·N̂_ω`) is insensitive to the sign convention at that slot; one carrying a single factor
(helicity's `conj(ω̂)·N̂`) changes sign with it.
"""
@inline derivative_wavenumber(a, i::Integer) = is_nyquist(a, i) ? zero(eltype(a)) : a[i]

"""
    dealias_factors(proto, FT, ks, twothirds) -> Vector

`nd` arrays of `1`/`0` marking the modes the dealiasing rule keeps along each axis, reshaped to
broadcast along their own axis; their product is the full keep-mask. All-ones for a rule that
discards nothing. In `FT` (not `Bool`) so a keep factor multiplies into a numeric broadcast without
promoting the expression. Applies to a field and to a derivative alike — the Nyquist rule for a
derivative lives in [`derivative_wavenumber`](@ref), per axis, not in this product mask.
"""
function dealias_factors(proto::AbstractArray, ::Type{FT}, ks::Tuple, twothirds::Bool) where {FT}
    nd = length(ks)
    return [begin
        a = ks[d]
        m = length(a)
        cut = full_length(a) ÷ 3
        h = FT[(!twothirds || abs(axis_index_wavenumber(a, i)) < cut) ? one(FT) : zero(FT) for i in 1:m]
        v = similar(proto, FT, m)
        copyto!(v, h)
        reshape(v, ntuple(i -> i == d ? m : 1, nd))
    end for d in 1:nd]
end

end # module SpectralLayout
