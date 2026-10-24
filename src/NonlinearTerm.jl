module NonlinearTerm

using ..Types: Types
using FlowTransformBindings: FlowTransformBindings as FTB
using ..Workspaces: Workspaces
using SpectralBackends: SpectralBackends

export compute_nonlinear_term, compute_nonlinear_term!, compute_rotational_term!
export synthesize_advecting!, compute_advection!

# ---------------------------------------------------------------------------
# FFTW extension stubs: the legs of the FFT nonlinear term
# ---------------------------------------------------------------------------

_nlt_synth_advecting_fft!(args...) = throw(ArgumentError(
    "FFT-accelerated nonlinear term requires FFTW. Run `using FFTW` to load the extension."))
_nlt_advect_fft!(args...) = throw(ArgumentError(
    "FFT-accelerated nonlinear term requires FFTW. Run `using FFTW` to load the extension."))
_nlt_forward_fft!(args...) = throw(ArgumentError(
    "FFT-accelerated nonlinear term requires FFTW. Run `using FFTW` to load the extension."))
_nlt_padded_synth_fft!(args...) = throw(ArgumentError(
    "Types.PaddedThreeHalves dealiasing requires FFTW. Run `using FFTW` to load the extension."))
_nlt_padded_advect_fft!(args...) = throw(ArgumentError(
    "Types.PaddedThreeHalves dealiasing requires FFTW. Run `using FFTW` to load the extension."))

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

"""
    compute_nonlinear_term(advected_hat, ks; dealiasing=OrszagTwoThirds(),
                           transform=SpectralBackends.AutoSpectralBackend(), advecting_hat=advected_hat)

Compute the pseudospectral nonlinear term `𝒩̂ᵢ(k) = F̂[(uⱼ ∂fᵢ/∂xⱼ)]` — the advection of an
`M`-component field `f` (`advected_hat`) by a velocity `u` (`advecting_hat`). For the momentum
self-advection term pass the velocity as both (the default), giving `N̂ᵢ = F̂[(u·∇)uᵢ]`.

# Arguments
- `advected_hat`: Array of size `(ms..., M)` — Fourier coefficients of the advected field `f`
  (`M = D` for momentum, `M = 1` for a passive scalar / vector potential).
- `ks`: Tuple of 1D wavenumber vectors (length `nd`), one per spatial dimension.

# Keyword Arguments
- `dealiasing::AbstractDealiasing=OrszagTwoThirds()`: dealiasing strategy (NoDealiasing / OrszagTwoThirds / PaddedThreeHalves).
- `transform::SpectralBackends.AbstractSpectralBackend`: `SpectralBackends.AutoSpectralBackend()`
  (default: FFT when FFTW is loaded, else the direct sum), `SpectralBackends.DirectSumSpectralBackend()`
  (no dependencies) or `SpectralBackends.FFTSpectralBackend()` (requires the FFTW extension).
- `advecting_hat`: the advecting velocity `u` (shape `(ms..., D)`, `D ≥ nd`); defaults to
  `advected_hat` (self-advection). Only the `nd` spatial components participate in `(u·∇)`.

# Returns
Array of size `(ms..., M)` containing `𝒩̂ᵢ(k)`.
"""
function compute_nonlinear_term(
    velocity_hat,
    ks;
    dealiasing::Types.AbstractDealiasing = Types.OrszagTwoThirds(),
    transform::SpectralBackends.AbstractSpectralBackend = SpectralBackends.AutoSpectralBackend(),
    advecting_hat = velocity_hat,
)
    ws = Workspaces.NonlinearTermWorkspace(velocity_hat, ks; dealiasing=dealiasing)
    compute_nonlinear_term!(ws, velocity_hat, ks;
        dealiasing=dealiasing, transform=transform, advecting_hat=advecting_hat)
    return ws.N̂
end

"""
    compute_nonlinear_term!(ws, advected_hat, ks; dealiasing=OrszagTwoThirds(),
                            transform=SpectralBackends.AutoSpectralBackend(), advecting_hat=advected_hat) -> ws.N̂

In-place version of [`compute_nonlinear_term`](@ref), writing into `ws.N̂`. `ws` is a
`NonlinearTermWorkspace` sized for `advected_hat`.
"""
function compute_nonlinear_term!(
    ws::Workspaces.NonlinearTermWorkspace,
    velocity_hat,
    ks;
    dealiasing::Types.AbstractDealiasing = Types.OrszagTwoThirds(),
    transform::SpectralBackends.AbstractSpectralBackend = SpectralBackends.AutoSpectralBackend(),
    advecting_hat = velocity_hat,
)
    transform = _checked_transform(transform, velocity_hat)
    _synthesize_advecting!(ws, advecting_hat, ks, transform, dealiasing)
    _compute_advection!(ws, velocity_hat, ks, transform, dealiasing)
    return ws.N̂
end

"""
    synthesize_advecting!(ws, advecting_hat, ks; dealiasing=OrszagTwoThirds(),
                          transform=AutoSpectralBackend()) -> ws

The advecting velocity of [`compute_nonlinear_term!`](@ref), synthesized into `ws` (on the padded
grid under `PaddedThreeHalves`). A driver that advects several fields by one velocity calls this once
and [`compute_advection!`](@ref) per field.
"""
function synthesize_advecting!(
    ws::Workspaces.NonlinearTermWorkspace,
    advecting_hat,
    ks;
    dealiasing::Types.AbstractDealiasing = Types.OrszagTwoThirds(),
    transform::SpectralBackends.AbstractSpectralBackend = SpectralBackends.AutoSpectralBackend(),
)
    _synthesize_advecting!(ws, advecting_hat, ks, _checked_transform(transform, advecting_hat), dealiasing)
    return ws
end

"""
    compute_advection!(ws, advected_hat, ks; dealiasing=OrszagTwoThirds(),
                       transform=AutoSpectralBackend()) -> ws.N̂

[`compute_nonlinear_term!`](@ref) of `advected_hat` by the velocity [`synthesize_advecting!`](@ref)
left in `ws`, with the same `dealiasing` and `transform`.
"""
function compute_advection!(
    ws::Workspaces.NonlinearTermWorkspace,
    advected_hat,
    ks;
    dealiasing::Types.AbstractDealiasing = Types.OrszagTwoThirds(),
    transform::SpectralBackends.AbstractSpectralBackend = SpectralBackends.AutoSpectralBackend(),
)
    _compute_advection!(ws, advected_hat, ks, _checked_transform(transform, advected_hat), dealiasing)
    return ws.N̂
end

# The resolved transform; the direct sum scalar-indexes, so it refuses a device array before any work.
function _checked_transform(transform, field)
    transform = Types.resolve_transform(transform)
    if transform isa SpectralBackends.DirectSumSpectralBackend && Types._is_device(field)
        throw(ArgumentError(
            "SpectralBackends.DirectSumSpectralBackend uses scalar-indexed direct sums (a host O(N²ᴰ) reference) and cannot run on " *
            "device arrays; use `transform = SpectralBackends.FFTSpectralBackend()` (cuFFT via AbstractFFTs) for the device path."))
    end
    return transform
end

# Dispatch on (transform, dealiasing). The advecting velocity `u` is synthesized into
# `ws.u_phys`, or the padded grid's `u_phys` under `PaddedThreeHalves`; the advected field `f` then
# gives `𝒩_i = u_j ∂_j f_i`.
_synthesize_advecting!(ws, a, ks, ::SpectralBackends.DirectSumSpectralBackend, ::Types.OrszagTwoThirds) =
    _direct_synth_into!(ws.u_phys, a, ks, length(ks), true)
_synthesize_advecting!(ws, a, ks, ::SpectralBackends.DirectSumSpectralBackend, ::Types.NoDealiasing) =
    _direct_synth_into!(ws.u_phys, a, ks, length(ks), false)
_synthesize_advecting!(ws, a, ks, ::SpectralBackends.DirectSumSpectralBackend, ::Types.PaddedThreeHalves) =
    _padded_synth_advecting_direct!(_require_pad(ws.pad), a, length(ks))
_synthesize_advecting!(ws, a, ks, ::SpectralBackends.FFTSpectralBackend, ::Types.OrszagTwoThirds) =
    _nlt_synth_advecting_fft!(ws, a, ks, true)
_synthesize_advecting!(ws, a, ks, ::SpectralBackends.FFTSpectralBackend, ::Types.NoDealiasing) =
    _nlt_synth_advecting_fft!(ws, a, ks, false)
_synthesize_advecting!(ws, a, ks, ::SpectralBackends.FFTSpectralBackend, ::Types.PaddedThreeHalves) =
    _nlt_padded_synth_fft!(ws, a, ks)

_compute_advection!(ws, f, ks, ::SpectralBackends.DirectSumSpectralBackend, ::Types.OrszagTwoThirds) =
    _advection_direct!(ws, f, ks, true)
_compute_advection!(ws, f, ks, ::SpectralBackends.DirectSumSpectralBackend, ::Types.NoDealiasing) =
    _advection_direct!(ws, f, ks, false)
_compute_advection!(ws, f, ks, ::SpectralBackends.DirectSumSpectralBackend, ::Types.PaddedThreeHalves) =
    _padded_advection_direct!(_require_pad(ws.pad), ws.N̂, f, length(ks))
_compute_advection!(ws, f, ks, ::SpectralBackends.FFTSpectralBackend, ::Types.OrszagTwoThirds) =
    _nlt_advect_fft!(ws, f, ks, true)
_compute_advection!(ws, f, ks, ::SpectralBackends.FFTSpectralBackend, ::Types.NoDealiasing) =
    _nlt_advect_fft!(ws, f, ks, false)
_compute_advection!(ws, f, ks, ::SpectralBackends.FFTSpectralBackend, ::Types.PaddedThreeHalves) =
    _nlt_padded_advect_fft!(ws, f, ks)

# The workspace's padded grid; a workspace built for another dealiasing has none.
_require_pad(pad::Workspaces.PaddedGrid) = pad
_require_pad(::Nothing) = throw(ArgumentError(
    "PaddedThreeHalves needs a workspace built for it — construct with dealiasing = Types.PaddedThreeHalves()."))

# Component `c` of a component-last array as a `(…, 1)` view, the shape
# `FlowTransformBindings.padded_embed!`/`padded_truncate!` read and write.
@inline _component(A::AbstractArray{T, N}, c::Int) where {T, N} =
    view(A, ntuple(_ -> Colon(), Val(N - 1))..., c:c)

# ---------------------------------------------------------------------------
# 2/3 dealiasing
# ---------------------------------------------------------------------------
#
# The 2/3 rule truncates both the inputs and the output of the quadratic product: an untruncated
# pair p, q whose sum wraps past Nyquist aliases onto a low mode (p = q = N/2−1 → k ≈ −2). Dealiased
# input modes are skipped when building u and ∇u, and the output is zeroed above the cutoff.

# ---------------------------------------------------------------------------
# Legs of the nonlinear term
#
# `𝒩 = Σ_j u_j ∂_j f` factors into: synthesise the advecting velocity, accumulate the advection,
# transform forward. A caller whose advected field is one Fourier mode (mode-to-mode's slot field) knows
# `∂_j f` analytically, a plane wave, and needs only the first and third legs.
# ---------------------------------------------------------------------------

# `dest[…, c] = Σ_k src[k, c] e^{ik·x}` for the first `ncomp` components, by direct sums.
function _direct_synth_into!(dest, src_hat, ks, ncomp::Int, truncate::Bool)
    nd = length(ks); ns = FTB.full_size(ks)
    for c in 1:ncomp
        û_c = selectdim(src_hat, nd + 1, c)
        for phys_I in CartesianIndices(ns)
            dest[phys_I, c] = _direct_synth(û_c, ks, phys_I, truncate, I -> true)
        end
    end
    return dest
end

"""
    nlt_forward!(ws, ks, spectral; truncate=true) -> ws.N̂

`𝒩̂ = DFT(ws.N_phys)/Nᵈ`, zeroed above the dealiasing cutoff — the last leg on its own.
"""
nlt_forward!(ws, ks, ::SpectralBackends.FFTSpectralBackend; truncate::Bool = true) =
    _nlt_forward_fft!(ws, ks, truncate)

function nlt_forward!(ws, ks, ::SpectralBackends.DirectSumSpectralBackend; truncate::Bool = true)
    nd = length(ks)
    M  = size(ws.N̂, nd + 1); FT = real(eltype(ws.N̂))
    scale = one(FT) / FT(prod(FTB.full_size(ks)))
    for comp in 1:M
        _direct_analysis!(selectdim(ws.N̂, nd + 1, comp), selectdim(ws.N_phys, nd + 1, comp), ks, scale,
                          I -> truncate && _is_dealiased(ks, I))
    end
    return ws.N̂
end

# `dest[k] = scale · Σ_x src[x] e^{-ik·x}` over the coefficient grid of `ks`, and zero at the modes
# `skip` names.
function _direct_analysis!(dest, src, ks, scale, skip)
    nd = length(ks); ns = FTB.full_size(ks); ms = FTB.spectral_size(ks)
    FT = real(eltype(dest))
    @inbounds for spec_I in CartesianIndices(ms)
        if skip(spec_I)
            dest[spec_I] = zero(eltype(dest))
            continue
        end
        val = zero(complex(FT))
        for phys_I in CartesianIndices(ns)
            phase = zero(FT)
            for d in 1:nd
                phase += FT(2π) * FTB.mode_frequency(ks[d], spec_I[d]) *
                         FT(phys_I[d] - 1) / FT(ns[d])
            end
            val += src[phys_I] * exp(-im * phase)
        end
        dest[spec_I] = val * scale
    end
    return dest
end

# ---------------------------------------------------------------------------
# Direct sums on the padded grid (`PaddedThreeHalves`): each field is embedded in the grid of the
# workspace's `PaddedGrid`, the product formed there, and its forward transform returned to the
# caller's grid, as on the FFT path.
# ---------------------------------------------------------------------------

# `dest[…, j]` for `j = 1:ncomp`: component `j` of `src_hat` embedded in the padded grid and
# synthesised there.
function _padded_direct_synth!(pg, dest, src_hat, ncomp::Int)
    nd = length(pg.ks); Ms = FTB.full_size(pg.ks)
    for j in 1:ncomp
        FTB.padded_embed!(pg.spec1, _component(src_hat, j), pg.maps)
        dj = selectdim(dest, nd + 1, j)
        for x in CartesianIndices(Ms)
            dj[x] = _direct_synth(pg.spec, pg.ks, x, false, I -> true)
        end
    end
    return dest
end

_padded_synth_advecting_direct!(pg, advecting_hat, nd::Int) =
    _padded_direct_synth!(pg, pg.u_phys, advecting_hat, nd)

# `N̂ = (u·∇)f` with `u` in `pg.u_phys`: each gradient of `f` synthesised on the padded grid with the
# padded derivative wavenumbers, the product accumulated there and transformed back.
function _padded_advection_direct!(pg, N̂, advected_hat, nd::Int)
    FT = real(eltype(advected_hat)); Ms = FTB.full_size(pg.ks)
    for c in 1:size(advected_hat, nd + 1)
        nc = selectdim(pg.n_phys, nd + 1, c)
        fill!(nc, zero(FT))
        FTB.padded_embed!(pg.spec1, _component(advected_hat, c), pg.maps)
        for j in 1:nd
            kdj = pg.kd[j]
            for x in CartesianIndices(Ms)
                pg.g_phys[x] = _direct_synth(pg.spec, pg.ks, x, false, I -> im * kdj[I[j]])
            end
            nc .+= selectdim(pg.u_phys, nd + 1, j) .* pg.g_phys
        end
    end
    return _padded_direct_forward!(pg, N̂, nd)
end

# Every component of `pg.n_phys` analysed on the padded grid, divided by its point count and returned to
# the caller's grid in `N̂` (`FlowTransformBindings.padded_truncate!`).
function _padded_direct_forward!(pg, N̂, nd::Int)
    for c in 1:size(N̂, nd + 1)
        _direct_analysis!(pg.spec, selectdim(pg.n_phys, nd + 1, c), pg.ks, pg.invM, I -> false)
        FTB.padded_truncate!(_component(N̂, c), pg.spec1, pg.maps)
    end
    return N̂
end

# Synthesis of one component at one physical point, with `kfac` multiplying each mode's coefficient. On a
# half layout a stored mode stands for its unstored conjugate too, which `mode_fold` counts, so the real
# part of the stored sum is the sum over the full spectrum.
@inline function _direct_synth(û_c, ks, phys_I, truncate::Bool, kfac)
    nd = length(ks); ns = FTB.full_size(ks); ms = FTB.spectral_size(ks)
    FT = real(eltype(û_c))
    val = zero(complex(FT))
    @inbounds for spec_I in CartesianIndices(ms)
        truncate && _is_dealiased(ks, spec_I) && continue
        phase = zero(FT)
        for d in 1:nd
            phase += FT(2π) * FTB.mode_frequency(ks[d], spec_I[d]) *
                     FT(phys_I[d] - 1) / FT(ns[d])
        end
        val += FTB.mode_fold(ks, spec_I) * kfac(spec_I) * û_c[spec_I] * exp(im * phase)
    end
    return real(val)
end

"""
    _is_dealiased(ks, I) -> Bool

`true` if Fourier mode `I` lies in the 2/3-rule discard zone (`FlowTransformBindings.is_dealiased`). Reads the
wavenumber from `ks`, so it holds for both spectral layouts.
"""
@inline _is_dealiased(ks::Tuple, I::CartesianIndex) = FTB.is_dealiased(ks, I)

# ---------------------------------------------------------------------------
# Rotational form ω × u
# ---------------------------------------------------------------------------

_rotational_term_fft!(args...) = throw(ArgumentError(
    "FFT-accelerated nonlinear term requires FFTW. Run `using FFTW` to load the extension."))
_rotational_term_padded_fft!(args...) = throw(ArgumentError(
    "Types.PaddedThreeHalves dealiasing requires FFTW. Run `using FFTW` to load the extension."))

"""
    compute_rotational_term!(ws, velocity_hat, vortex_hat, ks; dealiasing=OrszagTwoThirds(),
                             transform=AutoSpectralBackend()) -> ws.N̂

`𝒩̂ = FFT[ω × u]`, written into `ws.N̂`: `u` is synthesised from `velocity_hat` and `ω = ∇ × v` from
`vortex_hat`, both `(ms..., D)` coefficient arrays in `D = 2` or `3` dimensions (in 2-D `ω` is the scalar
`∂₁v₂ − ∂₂v₁` and the product `ω ẑ × u`), with the normalization, input truncation and output zeroing of
[`compute_nonlinear_term!`](@ref).

`(u·∇)u = ω × u + ∇(½|u|²)`, and the gradient has no projection on a divergence-free `û`, so
`Re{û*·𝒩̂}` is the kinetic-energy transfer density of `(u·∇)u` when `v = u` is solenoidal.
"""
function compute_rotational_term!(
    ws::Workspaces.NonlinearTermWorkspace,
    velocity_hat,
    vortex_hat,
    ks;
    dealiasing::Types.AbstractDealiasing = Types.OrszagTwoThirds(),
    transform::SpectralBackends.AbstractSpectralBackend = SpectralBackends.AutoSpectralBackend(),
)
    nd = length(ks)
    nd in (2, 3) || throw(ArgumentError("the rotational form ω × u is two- or three-dimensional; got nd=$nd."))
    (size(velocity_hat, nd + 1) == nd && size(vortex_hat, nd + 1) == nd && size(ws.N̂, nd + 1) == nd) ||
        throw(DimensionMismatch("the rotational form takes $nd-component fields and a $nd-component workspace."))
    _compute_rotational_term!(ws, velocity_hat, vortex_hat, ks, _checked_transform(transform, velocity_hat), dealiasing)
    return ws.N̂
end

_compute_rotational_term!(ws, u, v, ks, ::SpectralBackends.DirectSumSpectralBackend, ::Types.OrszagTwoThirds) =
    _rotational_term_direct!(ws, u, v, ks, true)
_compute_rotational_term!(ws, u, v, ks, ::SpectralBackends.DirectSumSpectralBackend, ::Types.NoDealiasing) =
    _rotational_term_direct!(ws, u, v, ks, false)
_compute_rotational_term!(ws, u, v, ks, ::SpectralBackends.DirectSumSpectralBackend, ::Types.PaddedThreeHalves) =
    _rotational_term_padded_direct!(ws, u, v, ks)
_compute_rotational_term!(ws, u, v, ks, ::SpectralBackends.FFTSpectralBackend, ::Types.OrszagTwoThirds) =
    _rotational_term_fft!(ws, u, v, ks, true)
_compute_rotational_term!(ws, u, v, ks, ::SpectralBackends.FFTSpectralBackend, ::Types.NoDealiasing) =
    _rotational_term_fft!(ws, u, v, ks, false)
_compute_rotational_term!(ws, u, v, ks, ::SpectralBackends.FFTSpectralBackend, ::Types.PaddedThreeHalves) =
    _rotational_term_padded_fft!(ws, u, v, ks)

# `ω̂ = i k × v̂` into `out` by a host loop over the modes, with each axis's derivative wavenumber.
function _curl_hat_direct!(out, v, ks)
    ms = FTB.spectral_size(ks)
    @inbounds for I in CartesianIndices(ms)
        kx = FTB.derivative_wavenumber(ks[1], I[1])
        ky = FTB.derivative_wavenumber(ks[2], I[2])
        kz = FTB.derivative_wavenumber(ks[3], I[3])
        out[I, 1] = im * (ky * v[I, 3] - kz * v[I, 2])
        out[I, 2] = im * (kz * v[I, 1] - kx * v[I, 3])
        out[I, 3] = im * (kx * v[I, 2] - ky * v[I, 1])
    end
    return out
end

# The 2-D vorticity `ω̂ = i(k₁ v̂₂ − k₂ v̂₁)` into component 1 of `out` by a host loop over the modes.
function _curl2_hat_direct!(out, v, ks)
    ms = FTB.spectral_size(ks)
    @inbounds for I in CartesianIndices(ms)
        k1 = FTB.derivative_wavenumber(ks[1], I[1])
        k2 = FTB.derivative_wavenumber(ks[2], I[2])
        out[I, 1] = im * (k1 * v[I, 2] - k2 * v[I, 1])
    end
    return out
end

"""
    _cross2_in_place!(W, U) -> W

`W ← ω ẑ × U = (−ω U₂, ω U₁)` with `ω` held in component 1 of `W`. Elementwise broadcasts, so it runs on
device arrays.
"""
function _cross2_in_place!(W, U)
    d = ndims(W)
    ω = selectdim(W, d, 1); w2 = selectdim(W, d, 2)
    u1 = selectdim(U, d, 1); u2 = selectdim(U, d, 2)
    @. w2 = ω * u1
    @. ω = -ω * u2
    return W
end

"""
    _cross_in_place!(W, U, g) -> W

`W ← W × U` over the trailing component axis of three-component fields, with `g` (one component's
shape) as the only scratch. Elementwise broadcasts, so it runs on device arrays.
"""
function _cross_in_place!(W, U, g)
    d = ndims(W)
    w1 = selectdim(W, d, 1); w2 = selectdim(W, d, 2); w3 = selectdim(W, d, 3)
    u1 = selectdim(U, d, 1); u2 = selectdim(U, d, 2); u3 = selectdim(U, d, 3)
    @. g  = w2 * u3 - w3 * u2      # x component
    @. w2 = w1 * u2 - w2 * u1      # z component, held in slot 2
    @. w1 = w3 * u1 - w1 * u3      # y component, held in slot 1
    w3 .= w2
    w2 .= w1
    w1 .= g
    return W
end

# ω̂ into `ws.N̂`, then ω into `ws.N_phys` and u into `ws.u_phys`; `ws.N̂` is overwritten by the forward
# transform of `ω × u` at the end.
function _rotational_term_direct!(ws, velocity_hat, vortex_hat, ks::NTuple{3,Any}, truncate::Bool)
    _curl_hat_direct!(ws.N̂, vortex_hat, ks)
    _direct_synth_into!(ws.N_phys, ws.N̂, ks, 3, truncate)
    _direct_synth_into!(ws.u_phys, velocity_hat, ks, 3, truncate)
    _cross_in_place!(ws.N_phys, ws.u_phys, ws.g_phys)
    return nlt_forward!(ws, ks, SpectralBackends.DirectSumSpectralBackend(); truncate = truncate)
end

function _rotational_term_direct!(ws, velocity_hat, vortex_hat, ks::NTuple{2,Any}, truncate::Bool)
    _curl2_hat_direct!(ws.N̂, vortex_hat, ks)
    _direct_synth_into!(ws.N_phys, ws.N̂, ks, 1, truncate)
    _direct_synth_into!(ws.u_phys, velocity_hat, ks, 2, truncate)
    _cross2_in_place!(ws.N_phys, ws.u_phys)
    return nlt_forward!(ws, ks, SpectralBackends.DirectSumSpectralBackend(); truncate = truncate)
end

# The same on the padded grid: ω̂ on the caller's grid, both fields embedded and synthesised on the
# padded grid, the product formed there and returned to the caller's grid.
function _rotational_term_padded_direct!(ws, velocity_hat, vortex_hat, ks::NTuple{3,Any})
    pg = _require_pad(ws.pad)
    _curl_hat_direct!(ws.N̂, vortex_hat, ks)
    _padded_direct_synth!(pg, pg.n_phys, ws.N̂, 3)
    _padded_direct_synth!(pg, pg.u_phys, velocity_hat, 3)
    _cross_in_place!(pg.n_phys, pg.u_phys, pg.g_phys)
    return _padded_direct_forward!(pg, ws.N̂, 3)
end

function _rotational_term_padded_direct!(ws, velocity_hat, vortex_hat, ks::NTuple{2,Any})
    pg = _require_pad(ws.pad)
    _curl2_hat_direct!(ws.N̂, vortex_hat, ks)
    _padded_direct_synth!(pg, pg.n_phys, ws.N̂, 1)
    _padded_direct_synth!(pg, pg.u_phys, velocity_hat, 2)
    _cross2_in_place!(pg.n_phys, pg.u_phys)
    return _padded_direct_forward!(pg, ws.N̂, 2)
end

# ---------------------------------------------------------------------------
# Direct-sum reference implementation
# ---------------------------------------------------------------------------

"""
    _advection_direct!(ws, advected_hat, ks, truncate) -> ws.N̂

Reference `O(N²ᴰ)` direct-sum advection of `advected_hat` by the velocity in `ws.u_phys`:
`∂_j f_i(x) = Σ_k i k_j f̂_i e^{ik·x}`, `𝒩_i = Σ_j u_j ∂_j f_i`, `𝒩̂ = DFT(𝒩)/Nᵈ`, with the inputs and
the output truncated to the 2/3 band when `truncate`.

The package convention is `û = fft(u)/Nᵈ`, so synthesis `u(x) = Σ_k û(k) e^{ik·x}` is the unnormalized
inverse and analysis `û(k) = (1/Nᵈ) Σ_x u(x) e^{-ik·x}`; `T(k) = Re{û*·N̂}` is then the physical transfer.
"""
function _advection_direct!(ws::Workspaces.NonlinearTermWorkspace, advected_hat, ks, truncate::Bool)
    nd = length(ks)
    ns = FTB.full_size(ks)
    M  = size(advected_hat, nd + 1)
    FT = real(eltype(advected_hat))
    for comp in 1:M
        f̂_c = selectdim(advected_hat, nd + 1, comp)
        Nc  = selectdim(ws.N_phys, nd + 1, comp)
        fill!(Nc, zero(FT))
        for grad_d in 1:nd
            # `derivative_wavenumber`, matching the FFT engine: the grid derivative along `grad_d`
            # vanishes at that axis's Nyquist mode.
            kfac = I -> im * FT(FTB.derivative_wavenumber(ks[grad_d], I[grad_d]))
            for phys_I in CartesianIndices(ns)
                ws.g_phys[phys_I] = _direct_synth(f̂_c, ks, phys_I, truncate, kfac)
            end
            Nc .+= selectdim(ws.u_phys, nd + 1, grad_d) .* ws.g_phys
        end
    end
    return nlt_forward!(ws, ks, SpectralBackends.DirectSumSpectralBackend(); truncate = truncate)
end

end # module NonlinearTerm
