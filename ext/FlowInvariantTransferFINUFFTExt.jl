module FlowInvariantTransferFINUFFTExt

using FINUFFT: FINUFFT
using FlowInvariantTransfer: FlowInvariantTransfer as FIT
using FlowInvariantTransfer.Types: AbstractFilter, CoarseGrainingFluxMethod, CoarseGrainingFluxResult, CoarseGrainingFluxResultWithDiagnostics
using FlowInvariantTransfer.Filters: filter_response
using FlowInvariantTransfer.Utils: wavenumber_magnitude_grid

# ---------------------------------------------------------------------------
# Non-uniform coarse-graining flux via FINUFFT type-1/type-2 round-trips.
#
# A single Π_ℓ evaluation issues ~4·D²+2·D type-1/type-2 transforms — all on the SAME scattered
# points, spectral-grid size, and tolerance. `NUFFTCoarseGrainingWorkspace` builds ONE type-1 and
# ONE type-2 guru plan (points set once) plus every working buffer, so a repeat call re-plans
# nothing and allocates nothing on the Julia side (`finufft_exec!` writes into the reused buffers).
# Plans are C resources with no finalizer of their own — the workspace registers one.
# ---------------------------------------------------------------------------

# View of the c-th component page of an (spatial…, D) array — contiguous (last-dim slice), so
# FINUFFT's `InputArray = Union{Array,SubArray}` + contiguity check accepts it.
@inline _page(A::AbstractArray, c::Int) = view(A, ntuple(_ -> Colon(), ndims(A) - 1)..., c)

"""
    NUFFTCoarseGrainingWorkspace(scatter_coords, ms; tol=1e-8)

Build the reusable FINUFFT plans (type-1 analysis, type-2 synthesis; points set) and every working
buffer for [`nufft_coarse_graining_flux!`](@ref). 1D/2D/3D scattered Cartesian points.
"""
function FIT.NUFFTCoarseGrainingWorkspace(scatter_coords::Tuple, ms::Tuple; tol::Real = 1e-8)
    nd = length(scatter_coords)
    nd == length(ms) || throw(ArgumentError("scatter_coords ($(nd)D) and ms ($(length(ms))D) must match"))
    1 <= nd <= 3 || throw(ArgumentError("FINUFFT supports 1D, 2D, 3D only; got nd=$nd."))
    N  = length(scatter_coords[1])
    FT = float(eltype(scatter_coords[1]))
    CT = Complex{FT}
    D  = nd    # velocity has one component per spatial dimension

    # Uniform wavenumber grid inferred from the coordinate spans.
    Ls = ntuple(nd) do d
        cv = scatter_coords[d]; rng = maximum(cv) - minimum(cv)
        rng > 0 ? FT(rng) : one(FT)
    end
    ks_1d = ntuple(nd) do d
        N_d = ms[d]; dk = 2 * FT(π) / Ls[d]
        [FT(k <= N_d ÷ 2 ? k : k - N_d) * dk for k in 0:N_d-1]
    end
    k_mag = wavenumber_magnitude_grid(ks_1d)
    k_comp_grids = ntuple(d -> _build_k_component_nufft(ks_1d, d, ms), nd)

    # Coordinates rescaled to [-π, π) for FINUFFT.
    scaled_coords = ntuple(nd) do d
        cmin = FT(minimum(scatter_coords[d])); rng = FT(maximum(scatter_coords[d])) - cmin
        rng > 0 ? (FT.(scatter_coords[d]) .- cmin) ./ rng .* (2 * FT(π)) .- FT(π) : zeros(FT, N)
    end

    # Working buffers (all reused across calls).
    Ĝ      = zeros(FT, ms...)
    û_filt = zeros(CT, ms..., D)
    u_filt = zeros(FT, N, D)
    τ      = zeros(FT, N, D, D)
    S̄      = zeros(FT, N, D, D)
    Π      = zeros(FT, N)
    spec   = zeros(CT, ms...)
    scat_in  = zeros(CT, N)
    scat_out = zeros(CT, N)
    prod_r = zeros(FT, N)
    grad_j = zeros(FT, N)

    nmodes = Int64[ms...]
    p1 = FINUFFT.finufft_makeplan(1, nmodes, 1, 1, FT(tol); dtype = FT)   # nonuniform → uniform
    p2 = FINUFFT.finufft_makeplan(2, nmodes, 1, 1, FT(tol); dtype = FT)   # uniform → nonuniform
    FINUFFT.finufft_setpts!(p1, scaled_coords...)
    FINUFFT.finufft_setpts!(p2, scaled_coords...)

    ws = FIT.NUFFTCoarseGrainingWorkspace(
        p1, p2, scaled_coords, k_mag, k_comp_grids, ks_1d, Ĝ, û_filt, u_filt,
        τ, S̄, Π, spec, scat_in, scat_out, prod_r, grad_j, N, FT(tol))
    finalizer(ws) do w
        FINUFFT.finufft_destroy!(w.p1)
        FINUFFT.finufft_destroy!(w.p2)
    end
    return ws
end

"""
    nufft_coarse_graining_flux!(ws::NUFFTCoarseGrainingWorkspace, velocity_fields, ℓ, filter, ms;
                                return_diagnostics=false) -> CoarseGrainingFluxResult

In-place scattered coarse-graining flux reusing `ws` — every transform runs through the preset
FINUFFT plans via `finufft_exec!`, and every intermediate is a reused workspace buffer, so a repeat
call allocates only the tiny result struct (which wraps `ws.Π`; a later call overwrites it). Intended
for a filter-scale sweep `Π(ℓ)` or a snapshot time series on the same points.
"""
function FIT.nufft_coarse_graining_flux!(
    ws::FIT.NUFFTCoarseGrainingWorkspace,
    velocity_fields::Tuple,
    ℓ::Real,
    filter::AbstractFilter,
    ms::Tuple;
    return_diagnostics::Bool = false,
)
    D  = length(velocity_fields)
    nd = length(ws.scaled_coords)
    D == nd || throw(ArgumentError("velocity components ($D) ≠ spatial dimensions ($nd)"))
    size(ws.u_filt, 2) == D || throw(ArgumentError(
        "workspace was built for $(size(ws.u_filt, 2)) components, got $D"))
    size(ws.spec) == ms || throw(ArgumentError(
        "workspace spectral grid $(size(ws.spec)) ≠ ms $ms"))
    N  = ws.npoints
    FT = eltype(ws.Π)
    length(velocity_fields[1]) == N || throw(DimensionMismatch(
        "velocity field length $(length(velocity_fields[1])) ≠ workspace points $N"))
    invN = one(FT) / FT(N)

    # Filter weights Ĝ(k) for this scale.
    Ĝ = ws.Ĝ
    @inbounds for I in CartesianIndices(ws.k_mag)
        Ĝ[I] = FT(filter_response(filter, ws.k_mag[I], FT(ℓ)))
    end

    # Per component: û_filt = Ĝ·(type-1 u)/N (spectral), then filtered velocity at the points (type-2).
    for c in 1:D
        ws.scat_in .= velocity_fields[c]                         # real → complex, no temp
        FINUFFT.finufft_exec!(ws.p1, ws.scat_in, ws.spec)        # type-1: nonuniform → uniform
        ûfc = _page(ws.û_filt, c)
        @. ûfc = Ĝ * ws.spec * invN
        FINUFFT.finufft_exec!(ws.p2, ûfc, ws.scat_out)           # type-2: uniform → nonuniform
        @views @. ws.u_filt[:, c] = real(ws.scat_out)
    end

    # Stress τ̄ᵢⱼ, strain S̄ᵢⱼ, and the flux contraction Π = −Σ factor·τ·S̄, streamed pair-by-pair.
    fill!(ws.Π, 0)
    @inbounds for i in 1:D, j in i:D
        # Filtered product [uᵢuⱼ]̄ at the points.
        @. ws.prod_r = velocity_fields[i] * velocity_fields[j]
        ws.scat_in .= ws.prod_r
        FINUFFT.finufft_exec!(ws.p1, ws.scat_in, ws.spec)
        @. ws.spec = Ĝ * ws.spec * invN
        FINUFFT.finufft_exec!(ws.p2, ws.spec, ws.scat_out)
        τij = view(ws.τ, :, i, j)
        @views @. τij = real(ws.scat_out) - ws.u_filt[:, i] * ws.u_filt[:, j]

        # Strain: ∂ūᵢ/∂xⱼ = type-2(i·kⱼ·û_filt_i).  (page views hoisted out of `@.`, which would
        # otherwise broadcast the `_page` call itself over the array elements.)
        ûfi = _page(ws.û_filt, i)
        @. ws.spec = im * ws.k_comp_grids[j] * ûfi
        FINUFFT.finufft_exec!(ws.p2, ws.spec, ws.scat_out)
        @. ws.prod_r = real(ws.scat_out)                          # ∂ūᵢ/∂xⱼ (reuse prod_r)
        S̄ij = view(ws.S̄, :, i, j)
        if i == j
            S̄ij .= ws.prod_r
        else
            ûfj = _page(ws.û_filt, j)
            @. ws.spec = im * ws.k_comp_grids[i] * ûfj
            FINUFFT.finufft_exec!(ws.p2, ws.spec, ws.scat_out)
            @. ws.grad_j = real(ws.scat_out)                      # ∂ūⱼ/∂xᵢ
            @. S̄ij = FT(0.5) * (ws.prod_r + ws.grad_j)
        end

        factor = i == j ? FT(1) : FT(2)
        @. ws.Π -= factor * τij * S̄ij
        if i != j                                                  # mirror the symmetric entries
            @views ws.τ[:, j, i] .= τij
            @views ws.S̄[:, j, i] .= S̄ij
        end
    end
    mean_Π = FT(sum(ws.Π) / N)

    if return_diagnostics
        return CoarseGrainingFluxResultWithDiagnostics(FT(ℓ), ws.Π, mean_Π, ws.τ, ws.S̄)
    else
        return CoarseGrainingFluxResult(FT(ℓ), ws.Π, mean_Π)
    end
end

"""
    nufft_coarse_graining_flux(velocity_fields, scatter_coords, ℓ, filter, ms;
                               return_diagnostics=false, tol=1e-8) -> CoarseGrainingFluxResult

Coarse-graining energy flux `Π_ℓ(x)` at scattered (non-uniform) Cartesian points via FINUFFT.

Builds a one-shot [`NUFFTCoarseGrainingWorkspace`](@ref) and delegates to
[`nufft_coarse_graining_flux!`](@ref). For repeated evaluations on the same points (a filter-scale
sweep, or a snapshot time series), build the workspace once and call the in-place form so the plans
and all intermediate buffers are reused.

# Arguments
- `velocity_fields`: Tuple of D real vectors of length N — velocity at the scattered points.
- `scatter_coords`: Tuple of D real vectors of length N — the point coordinates.
- `ℓ::Real`: Filter scale.  `filter::AbstractFilter`: filter kernel.
- `ms::NTuple{D,Int}`: intermediate uniform spectral-grid size.

# Keyword Arguments
- `return_diagnostics::Bool=false`: also return τ̄ᵢⱼ and S̄ᵢⱼ at the points.
- `tol=1e-8`: FINUFFT accuracy tolerance.
"""
function FIT.nufft_coarse_graining_flux(
    velocity_fields::Tuple,
    scatter_coords::Tuple,
    ℓ::Real,
    filter::AbstractFilter,
    ms::Tuple;
    return_diagnostics::Bool = false,
    tol::Real = 1e-8,
)
    ws = FIT.NUFFTCoarseGrainingWorkspace(scatter_coords, ms; tol = tol)
    return FIT.nufft_coarse_graining_flux!(
        ws, velocity_fields, ℓ, filter, ms; return_diagnostics = return_diagnostics)
end

# ---------------------------------------------------------------------------
# Unified entry: scattered-Cartesian coarse-graining flux through calculate_energy_transfer
# ---------------------------------------------------------------------------

"""
    calculate_energy_transfer(method::CoarseGrainingFluxMethod, velocity_fields, scatter_coords, ms; kwargs...)

Scattered / non-uniform-Cartesian coarse-graining flux `Π_ℓ(x)` through the unified
[`calculate_energy_transfer`](@ref) front-end. The extra `ms::Tuple` positional (the intermediate
uniform spectral-grid size) distinguishes this from the uniform-grid 3-argument `CoarseGrainingFluxMethod`
method in the core; it routes to [`nufft_coarse_graining_flux`](@ref) using the method's `scale`/`filter`.
Requires `using FINUFFT`.
"""
function FIT.calculate_energy_transfer(
    method::CoarseGrainingFluxMethod,
    velocity_fields::Tuple,
    scatter_coords::Tuple,
    ms::Tuple;
    kwargs...,
)
    return FIT.nufft_coarse_graining_flux(
        velocity_fields, scatter_coords, method.scale, method.filter, ms; kwargs...)
end

# ---------------------------------------------------------------------------
# NUFFT helpers
# ---------------------------------------------------------------------------

function _build_k_component_nufft(ks_1d, d::Int, ms::Tuple)
    kc = zeros(eltype(ks_1d[d]), ms...)
    for I in CartesianIndices(ms)
        kc[I] = ks_1d[d][I[d]]
    end
    return kc
end

end # module FlowInvariantTransferFINUFFTExt
