module ShellBinning

using ..Types: AbstractShellBinning, LinearBinning, LogarithmicBinning, DyadicBinning, CustomBinning
using ..Types: AbstractShellGeometry, ShellMagnitude

export shell_edges, shell_centers, n_shells, assign_shells, shell_coordinate

# ---------------------------------------------------------------------------
# Shell coordinate — the per-mode scalar the shells partition (set by geometry)
# ---------------------------------------------------------------------------

"""
    shell_coordinate(geometry, ks) -> Array

Return an array (shape `ns`) giving the wavenumber coordinate each mode is binned by under
`geometry`. For [`ShellMagnitude`](@ref) this is `√(Σ_{d∈dims} k_d²)` — `|k|` when `dims` covers
all dimensions (isotropic), `k_⊥`/`k_∥` for an anisotropic projection.
"""
function shell_coordinate(g::ShellMagnitude, ks)
    nd   = length(ks)
    ns   = ntuple(d -> length(ks[d]), nd)
    dims = g.dims === nothing ? ntuple(identity, nd) : g.dims
    all(d -> 1 <= d <= nd, dims) || throw(ArgumentError(
        "ShellMagnitude dims=$(g.dims) out of range for nd=$nd spatial dimensions."))
    FT  = float(eltype(ks[1]))
    out = Array{FT}(undef, ns...)
    @inbounds for I in CartesianIndices(ns)
        s = zero(FT)
        for d in dims
            kd = FT(ks[d][I[d]])
            s += kd * kd
        end
        out[I] = sqrt(s)
    end
    return out
end

# ---------------------------------------------------------------------------
# Shell edge generation
# ---------------------------------------------------------------------------

"""
    shell_edges(binning, k_max) -> Vector{Float64}

Return the monotonically increasing shell boundary vector for `binning` up to `k_max`.

The resulting vector has length `n_shells(binning, k_max) + 1`; shell n covers
wavenumbers in `[edges[n], edges[n+1])`.
"""
function shell_edges(b::LinearBinning, k_max::Real)
    b.Δk > 0 || throw(ArgumentError("LinearBinning: Δk must be positive."))
    k_max > 0 || throw(ArgumentError("k_max must be positive."))
    FT    = typeof(float(k_max))
    edges = collect(zero(FT) : FT(b.Δk) : FT(k_max))
    edges[end] < FT(k_max) && push!(edges, FT(k_max))
    return edges
end

function shell_edges(b::LogarithmicBinning, k_max::Real)
    b.k₀ > 0 || throw(ArgumentError("LogarithmicBinning: k₀ must be positive."))
    b.λ > 1  || throw(ArgumentError("LogarithmicBinning: λ must be > 1."))
    k_max >= b.k₀ || throw(ArgumentError("k_max must be >= k₀."))
    FT    = typeof(float(k_max))
    n_max = floor(Int, log(FT(k_max) / FT(b.k₀)) / log(FT(b.λ)))
    edges = [FT(b.k₀) * FT(b.λ)^n for n in 0:n_max+1]
    while length(edges) > 2 && edges[end-1] > FT(k_max)
        pop!(edges)
    end
    return edges
end

function shell_edges(b::DyadicBinning, k_max::Real)
    return shell_edges(LogarithmicBinning(b.k₀, oftype(b.k₀, 2)), k_max)
end

function shell_edges(b::CustomBinning, k_max::Real)
    issorted(b.edges) || throw(ArgumentError("CustomBinning: edges must be monotonically increasing."))
    return b.edges
end

# ---------------------------------------------------------------------------
# Derived helpers
# ---------------------------------------------------------------------------

"""
    shell_centers(binning, k_max) -> Vector{Float64}

Return the geometric midpoint of each shell.  For logarithmic binnings, this is
the geometric mean of the edge pair; for linear binnings, the arithmetic mean.
"""
function shell_centers(b::AbstractShellBinning, k_max::Real)
    edges = shell_edges(b, k_max)
    N = length(edges) - 1
    N > 0 || throw(ArgumentError("No shells within k_max=$k_max for this binning."))
    centers = similar(edges, N)
    if b isa LogarithmicBinning || b isa DyadicBinning
        for n in 1:N
            centers[n] = sqrt(edges[n] * edges[n+1])
        end
    else
        for n in 1:N
            centers[n] = (edges[n] + edges[n+1]) / 2
        end
    end
    return centers
end

"""
    n_shells(binning, k_max) -> Int

Return the number of shells for `binning` up to `k_max`.
"""
function n_shells(b::AbstractShellBinning, k_max::Real)
    return length(shell_edges(b, k_max)) - 1
end

"""
    assign_shells(k_mag, edges) -> Array{Int}

Return an integer array (same shape as `k_mag`) where `[I] = n` if
`edges[n] <= k_mag[I] < edges[n+1]`, and `0` if the mode falls outside all shells.

One integer per mode (single allocation, cache-friendly): the canonical shell-membership
representation used by every transfer accumulation kernel.
"""
function assign_shells(k_mag::AbstractArray, edges::AbstractVector)
    idx  = similar(k_mag, Int)
    fill!(idx, 0)
    N_sh = length(edges) - 1
    for I in CartesianIndices(k_mag)
        k = k_mag[I]
        for n in 1:N_sh
            if edges[n] <= k < edges[n+1]
                idx[I] = n
                break
            end
        end
    end
    return idx
end

end # module ShellBinning
