module FlowInvariantTransferGPUArraysCoreExt

using GPUArraysCore: AbstractGPUArray
using FlowInvariantTransfer: FlowInvariantTransfer as FIT

# ---------------------------------------------------------------------------
# Device-generic versions of the three compressible-pipeline reductions/masks that the scalar `src`
# methods can't express without scalar indexing (a dealias-truncated copy, the Helmholtz split, and the
# shell-binning reduction). Selected for device arrays (CuArray / JLArray) via `AbstractGPUArray`
# dispatch; host `Array`s keep the scalar 0-alloc `src` methods. Everything else in the compressible
# pipeline is already device-generic broadcasts + the device-generic FFT transform context, so these
# complete the on-device path (FFT → cuFFT by construction). Pure broadcasts/reductions — no KA kernel,
# no `synchronize` — so they run under `allowscalar(false)` and are verifiable on JLArrays.
# ---------------------------------------------------------------------------

# Dealias keep-mask (mode kept iff |k_d| < n_d÷3 for all d), device-resident. Built from per-dimension
# keep vectors (host Int test) moved to the device and broadcast-AND'd into the (ns) grid.
function _dealias_keep(proto, ns::NTuple{nd,Int}) where {nd}
    keep = nothing
    for d in 1:nd
        n = ns[d]
        kv = Bool[(i0 = i - 1; kabs = i0 <= n ÷ 2 ? i0 : n - i0; kabs < n ÷ 3) for i in 1:n]
        vd = similar(proto, Bool, n); copyto!(vd, kv)
        vr = reshape(vd, ntuple(j -> j == d ? n : 1, nd))
        keep = d == 1 ? (vr .& true) : (keep .& vr)
    end
    return keep
end

# dst = src, zeroing the Orszag 2/3 discard band (|k| ≥ n_d÷3) when `trunc`.
function FIT.Compressible._copy_trunc!(dst::AbstractGPUArray, src, ns::NTuple{nd,Int}, nd_::Int, trunc::Bool) where {nd}
    if trunc
        keep = _dealias_keep(dst, ns)
        dst .= reshape(keep, ns..., 1) .* src
    else
        copyto!(dst, src)
    end
    return dst
end

# Helmholtz split (rot ⊥ k, comp ∥ k) via broadcasts with a guarded 1/k² (0 at the DC mode → comp=0).
function FIT.Compressible._helmholtz_split!(rot::AbstractGPUArray, comp, field_hat, ks, ns::NTuple{nd,Int}) where {nd}
    FT = real(eltype(field_hat)); colons = ntuple(_ -> Colon(), nd)
    kg = ntuple(nd) do d
        v = similar(rot, FT, ns[d]); copyto!(v, collect(FT, ks[d]))
        reshape(v, ntuple(i -> i == d ? ns[d] : 1, nd))
    end
    k2 = similar(rot, FT, ns); fill!(k2, zero(FT))
    for d in 1:nd; k2 .+= kg[d] .^ 2; end
    invk2 = ifelse.(k2 .> 0, inv.(k2), zero(FT))
    kdotu = similar(rot, ns); fill!(kdotu, zero(eltype(kdotu)))   # Σ_c k_c·field_c (complex, ns)
    for c in 1:nd
        kdotu .+= kg[c] .* view(field_hat, colons..., c)
    end
    for c in 1:nd
        cc = (kdotu .* invk2) .* kg[c]                           # compressive component c (∥ k)
        view(comp, colons..., c) .= cc
        view(rot,  colons..., c) .= view(field_hat, colons..., c) .- cc
    end
    return nothing
end

# Shell-bin a per-mode density into shell sums (device reductions → host vector); the keep-mask excludes
# the 2/3 band when `dealias`. Mode → shell 0 (unassigned) is excluded since `n` runs 1:N_sh.
function FIT.Compressible._bin(td::AbstractGPUArray, sidx, N_sh, ::Type{FT}, ns::NTuple{nd,Int}, dealias::Bool) where {nd, FT}
    sidx_d = similar(td, Int, ns); copyto!(sidx_d, sidx)
    keep = dealias ? _dealias_keep(td, ns) : nothing
    T = Vector{FT}(undef, N_sh)
    for n in 1:N_sh
        m = keep === nothing ? (sidx_d .== n) : ((sidx_d .== n) .& keep)
        T[n] = sum(td .* m)
    end
    return T
end

end # module
