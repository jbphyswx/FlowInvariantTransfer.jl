module ShellBinning

using ..Types: Types
using ..SpectralLayout: SpectralLayout
using ComputationalBackends: ComputationalBackends

export shell_edges, shell_centers, n_shells, assign_shells, shell_coordinate, max_shell_coordinate

# ---------------------------------------------------------------------------
# Shell coordinate — the per-mode scalar the shells partition (set by geometry)
# ---------------------------------------------------------------------------

"""
    shell_coordinate(geometry, ks) -> Array

Return an array (shape `ns`) giving the wavenumber coordinate each mode is binned by under
`geometry`. For [`ShellMagnitude`](@ref) this is `√(Σ_{d∈dims} k_d²)` — `|k|` when `dims` covers
all dimensions (isotropic), `k_⊥`/`k_∥` for an anisotropic projection.
"""
function shell_coordinate(g::Types.ShellMagnitude, ks)
    nd   = length(ks)
    ns   = ntuple(d -> length(ks[d]), nd)
    dims = _shell_dims(g, nd)
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

function _shell_dims(g::Types.ShellMagnitude, nd::Int)
    dims = g.dims === nothing ? ntuple(identity, nd) : g.dims
    all(d -> 1 <= d <= nd, dims) || throw(ArgumentError(
        "ShellMagnitude dims=$(g.dims) out of range for nd=$nd spatial dimensions."))
    return dims
end

"""
    max_shell_coordinate(geometry, ks) -> Real

`maximum(shell_coordinate(geometry, ks))` without building the grid. The axes are independent, so
the maximum of `√(Σ_d k_d²)` over a tensor grid is `√(Σ_d max_d k_d²)` — `O(nd)` where materialising
the magnitude grid to take its maximum is `O(Nᴰ)`. Equal on the full and half layouts, so shell edges
built from it do not depend on how the field is stored.
"""
function max_shell_coordinate(g::Types.ShellMagnitude, ks)
    nd = length(ks)
    FT = float(eltype(ks[1]))
    s = zero(FT)
    for d in _shell_dims(g, nd)
        kd = FT(SpectralLayout.max_abs(ks[d]))
        s += kd * kd
    end
    return sqrt(s)
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
function shell_edges(b::Types.LinearBinning, k_max::Real)
    b.Δk > 0 || throw(ArgumentError("LinearBinning: Δk must be positive."))
    k_max > 0 || throw(ArgumentError("k_max must be positive."))
    FT    = typeof(float(k_max))
    edges = collect(zero(FT) : FT(b.Δk) : FT(k_max))
    edges[end] < FT(k_max) && push!(edges, FT(k_max))
    return edges
end

function shell_edges(b::Types.LogarithmicBinning, k_max::Real)
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

function shell_edges(b::Types.DyadicBinning, k_max::Real)
    return shell_edges(Types.LogarithmicBinning(b.k₀, oftype(b.k₀, 2)), k_max)
end

function shell_edges(b::Types.CustomBinning, k_max::Real)
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
function shell_centers(b::Types.AbstractShellBinning, k_max::Real)
    edges = shell_edges(b, k_max)
    N = length(edges) - 1
    N > 0 || throw(ArgumentError("No shells within k_max=$k_max for this binning."))
    centers = similar(edges, N)
    if b isa Types.LogarithmicBinning || b isa Types.DyadicBinning
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
function n_shells(b::Types.AbstractShellBinning, k_max::Real)
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
    N_sh = length(edges) - 1
    # `searchsortedlast` on sorted edges returns the last `n` with `edges[n] ≤ k`, which IS the
    # half-open bin `[edges[n], edges[n+1])`: `k ≥ edges[end]` gives `N_sh+1` and `k < edges[1]` gives
    # `0`, both of which fall outside `1:N_sh` and mean "no shell". Base specialises this to a closed
    # form when `edges` is a range, so a uniform binning costs O(1) per mode and an arbitrary one
    # O(log N_sh) — where a scan over the edges costs O(N_sh).
    @inbounds for I in CartesianIndices(k_mag)
        n = searchsortedlast(edges, k_mag[I])
        idx[I] = (1 <= n <= N_sh) ? n : 0
    end
    return idx
end

# Shell reduction `T_spec[shell_idx[I]] += density[I]` over all modes (shell index 0 = dropped),
# zeroing `T_spec` first. The host method is a 0-alloc scalar loop; the KernelAbstractions extension
# adds a `ComputationalBackends.GPUBackend` method (atomic device scatter-add) so device-backed arrays reduce with no scalar
# indexing. Backend-dispatched so the serial reductions and the distributed pencil flux share one op.
function shell_scatter_add!(T_spec, density, shell_idx, ::ComputationalBackends.AbstractExecutionBackend)
    fill!(T_spec, zero(eltype(T_spec)))
    @inbounds for I in CartesianIndices(density)
        s = shell_idx[I]
        s == 0 && continue
        T_spec[s] += density[I]
    end
    return T_spec
end

end # module ShellBinning
