module SpectralLayout

export AbstractWavenumberAxis, FullAxis, HalfAxis
export full_length, is_half, full_size, spectral_size, max_abs
export hermitian_weight, hermitian_weights
export axis_index_wavenumber, is_dealiased, is_nyquist, wavenumber_arrays, derivative_wavenumber
export dealias_cutoff, dealias_factors
export padded_length, padded_images, padded_preimage

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
    dealias_cutoff(n, order = 2) -> Int
    dealias_cutoff(axis, order = 2) -> Int

The largest `|m|` truncation dealiasing keeps on an `n`-point axis for products of `order` factors,
`⌊(n−1)/(order+1)⌋`: the largest `K` with `(order+1)K < n`, so a product of kept modes reaches
`|m| ≤ order·K` and its alias `m ∓ n` falls outside `[−K, K]`. `order = 2` is the Orszag 2/3 rule
(Orszag 1971).
"""
@inline dealias_cutoff(n::Integer, order::Integer = 2) = (Int(n) - 1) ÷ (order + 1)
@inline dealias_cutoff(a::AbstractVector, order::Integer = 2) = dealias_cutoff(full_length(a), order)

"""
    is_dealiased(ks, I, order = 2) -> Bool

`true` if coefficient `I` lies in the truncation discard band (`|m_d| > dealias_cutoff(n_d, order)`
along any axis `d`), for either spectral layout.
"""
@inline function is_dealiased(ks::Tuple, I::CartesianIndex{nd}, order::Integer = 2) where {nd}
    @inbounds for d in 1:nd
        a = ks[d]
        abs(axis_index_wavenumber(a, I[d])) > dealias_cutoff(a, order) && return true
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

# ---------------------------------------------------------------------------
# Exact 3/2 padding: the coarse ↔ padded mode correspondence, one axis at a time.
#
# Away from Nyquist each coarse mode is one padded mode. The Nyquist mode of an even full axis holds
# `+n/2` and `−n/2` together, and on the finer grid those are two modes: the real band-limited
# function it represents is `û_Nyq·cos(n·x/2)`, so it embeds as `û_Nyq/2` at each. A half axis stores
# only `+n/2`, its conjugate implied by the transform, so its Nyquist takes the halving with one image.
# ---------------------------------------------------------------------------

"""
    padded_length(n, order = 2) -> Int

The padded length of an `n`-point axis for exact dealiasing of products of `order` factors: the
smallest `M > (order+1)·(n÷2)` with `M − n` even. The coarse modes reach `|m| = n÷2`, their products
`order·(n÷2)`, and an alias `m − M` of such a product falls outside `[−n÷2, n÷2]` exactly when
`M > (order+1)·(n÷2)`, the Nyquist included. `order = 2` is the 3/2 rule.
"""
@inline function padded_length(n::Integer, order::Integer = 2)
    m = (order + 1) * (Int(n) ÷ 2) + 1
    return iseven(m - n) ? m : m + 1
end

"""
    padded_images(axis, i, M) -> NTuple

The indices of the `M`-point padded axis that coefficient `i` embeds into: two for the Nyquist of an
even full axis (`+n/2` and `−n/2`), one otherwise. On a half axis the index is on the padded half,
`M ÷ 2 + 1` long.
"""
@inline function padded_images(a, i::Integer, M::Integer)
    km = axis_index_wavenumber(a, i)
    if is_nyquist(a, i) && !(a isa HalfAxis)
        h = abs(km)
        return (h + 1, M - h + 1)
    end
    return (km >= 0 ? km + 1 : M + km + 1,)
end

"""
    padded_preimage(axis, p, M) -> Int

The coefficient of `axis` that index `p` of the `M`-point padded axis is an image of, or `0` for a
padded mode no coefficient embeds into. The inverse of [`padded_images`](@ref).
"""
@inline function padded_preimage(a, p::Integer, M::Integer)
    n = full_length(a)
    q = p - 1
    a isa HalfAxis && return q <= n ÷ 2 ? q + 1 : 0
    q <= n ÷ 2 && return q + 1                         # q = n/2 is the Nyquist slot of an even axis
    km = q - M
    km >= -((n - 1) ÷ 2) && return n + km + 1
    return (iseven(n) && km == -(n ÷ 2)) ? n ÷ 2 + 1 : 0
end

"""
    padded_axes(ks, Ms) -> Tuple

The wavenumber axes of the padded grid `Ms`: the spacing of `ks`, with the first axis a half axis
when `ks` is a half layout.
"""
function padded_axes(ks::Tuple, Ms::Tuple)
    dk(a) = a isa AbstractWavenumberAxis ? a.dk : a[2] - a[1]
    return ntuple(d -> (d == 1 && ks[1] isa HalfAxis) ? HalfAxis(Ms[1], dk(ks[1])) : FullAxis(Ms[d], dk(ks[d])),
                  length(ks))
end

"""
    padded_derivative_wavenumbers(proto, FT, ks, Ms) -> Vector

[`wavenumber_arrays`](@ref) with `derivative = true` for the padded grid `Ms`, except that the images
of an even coarse axis's Nyquist mode carry zero, the derivative the coarse grid gives that mode.
"""
function padded_derivative_wavenumbers(proto::AbstractArray, ::Type{FT}, ks::Tuple, Ms::Tuple) where {FT}
    nd = length(ks)
    kp = padded_axes(ks, Ms)
    function axis_values(d)
        m = length(kp[d])
        h = FT[_padded_derivative(ks[d], kp[d], p, Ms[d], FT) for p in 1:m]
        return reshape(copyto!(similar(proto, FT, m), h), ntuple(i -> i == d ? m : 1, nd))
    end
    return [axis_values(d) for d in 1:nd]
end

@inline function _padded_derivative(a, ap, p::Integer, M::Integer, ::Type{FT}) where {FT}
    c = padded_preimage(a, p, M)
    return (c != 0 && is_nyquist(a, c)) ? zero(FT) : FT(derivative_wavenumber(ap, p))
end

"""
    PaddedMaps(proto, FT, ks, Ms)

Gathers between the coefficients of `ks` and those of the padded grid `Ms`, in `proto`'s array type:
`padded_embed!` writes each coefficient at its images ([`padded_images`](@ref)), halved once per
Nyquist axis; `padded_truncate!` returns every coefficient the sum of its images with the inverse
weight, and on a half layout's even Nyquist plane averages it with the conjugate of its mirror, the
plane being its own image under `k ↦ −k`. `truncate ∘ embed` is the identity.
"""
struct PaddedMaps{VI, VF, VVI <: AbstractVector{VI}, VVF <: AbstractVector{VF}}
    e_src::VI; e_w::VF                          # per padded coefficient: source and weight (0: no source)
    t_src::VVI; t_w::VVF          # per coarse coefficient, one slot per image
    c_src::VVI; c_w::VVF         # conjugated slots, the half layout's Nyquist plane
end

function PaddedMaps(proto::AbstractArray, ::Type{FT}, ks::Tuple, Ms::Tuple) where {FT}
    nd = length(ks)
    ms = spectral_size(ks)
    kp = padded_axes(ks, Ms)
    Msp = spectral_size(kp)
    linC = LinearIndices(ms); linP = LinearIndices(Msp)
    nyqaxes(C) = count(d -> is_nyquist(ks[d], C[d]), 1:nd)
    images(C) = vec(collect(Iterators.product(ntuple(d -> padded_images(ks[d], C[d], Ms[d]), nd)...)))
    e_src = ones(Int, prod(Msp)); e_w = zeros(FT, prod(Msp))
    for P in CartesianIndices(Msp)
        C = ntuple(d -> padded_preimage(ks[d], P[d], Ms[d]), nd)
        any(iszero, C) && continue
        e_src[linP[P]] = linC[C...]; e_w[linP[P]] = FT(1) / FT(2)^nyqaxes(C)
    end
    nyq_row = (ks[1] isa HalfAxis && iseven(full_length(ks[1]))) ? ms[1] : 0
    mirror(C) = (C[1], ntuple(d -> C[d + 1] == 1 ? 1 : ms[d + 1] - C[d + 1] + 2, nd - 1)...)
    plain = [Tuple{Int, FT}[] for _ in 1:prod(ms)]; conj_ = [Tuple{Int, FT}[] for _ in 1:prod(ms)]
    for I in CartesianIndices(ms)
        C = Tuple(I)
        imgs = images(C)
        wt = FT(2)^nyqaxes(C) / FT(length(imgs))
        if C[1] == nyq_row
            append!(plain[linC[I]], (linP[P...], wt / 2) for P in imgs)
            append!(conj_[linC[I]], (linP[P...], wt / 2) for P in images(mirror(C)))
        else
            append!(plain[linC[I]], (linP[P...], wt) for P in imgs)
        end
    end
    dev(v) = copyto!(similar(proto, eltype(v), length(v)), v)
    es = dev(e_src); ew = dev(e_w)
    VI = typeof(es); VF = typeof(ew)
    nslots(lists) = maximum(length, lists; init = 0)
    src_slot(lists, s) = dev([s <= length(l) ? l[s][1] : 1 for l in lists])
    w_slot(lists, s) = dev([s <= length(l) ? l[s][2] : zero(FT) for l in lists])
    return PaddedMaps(es, ew,
                      VI[src_slot(plain, s) for s in 1:nslots(plain)], VF[w_slot(plain, s) for s in 1:nslots(plain)],
                      VI[src_slot(conj_, s) for s in 1:nslots(conj_)], VF[w_slot(conj_, s) for s in 1:nslots(conj_)])
end

"""
    padded_embed!(dst, src, maps::PaddedMaps) -> dst

The coefficients `src` (`(ms..., C)`) at their padded images in `dst` (`(Msp..., C)`), zero elsewhere.
"""
function padded_embed!(dst, src, pm::PaddedMaps)
    nd = ndims(dst) - 1
    for c in 1:size(src, nd + 1)
        s = vec(selectdim(src, nd + 1, c))
        vec(selectdim(dst, nd + 1, c)) .= pm.e_w .* view(s, pm.e_src)
    end
    return dst
end

"""
    padded_truncate!(dst, src, maps::PaddedMaps) -> dst

The padded coefficients `src` (`(Msp..., C)`) returned to the coarse coefficients `dst` (`(ms..., C)`).
"""
function padded_truncate!(dst, src, pm::PaddedMaps)
    nd = ndims(dst) - 1
    for c in 1:size(src, nd + 1)
        s = vec(selectdim(src, nd + 1, c)); d = vec(selectdim(dst, nd + 1, c))
        fill!(d, zero(eltype(d)))
        for (i, w) in zip(pm.t_src, pm.t_w)
            d .+= w .* view(s, i)
        end
        for (i, w) in zip(pm.c_src, pm.c_w)
            d .+= w .* conj.(view(s, i))
        end
    end
    return dst
end

"""
    dealias_factors(proto, FT, ks, twothirds; order = 2) -> Vector

`nd` arrays of `1`/`0` marking the modes truncation keeps along each axis for products of `order`
factors ([`dealias_cutoff`](@ref)), reshaped to
broadcast along their own axis; their product is the full keep-mask. All-ones for a rule that
discards nothing. In `FT` (not `Bool`) so a keep factor multiplies into a numeric broadcast without
promoting the expression. Applies to a field and to a derivative alike — the Nyquist rule for a
derivative lives in [`derivative_wavenumber`](@ref), per axis, not in this product mask.
"""
function dealias_factors(proto::AbstractArray, ::Type{FT}, ks::Tuple, twothirds::Bool;
                         order::Integer = 2) where {FT}
    nd = length(ks)
    return [begin
        a = ks[d]
        m = length(a)
        cut = dealias_cutoff(a, order)
        h = FT[(!twothirds || abs(axis_index_wavenumber(a, i)) <= cut) ? one(FT) : zero(FT) for i in 1:m]
        v = similar(proto, FT, m)
        copyto!(v, h)
        reshape(v, ntuple(i -> i == d ? m : 1, nd))
    end for d in 1:nd]
end

end # module SpectralLayout
