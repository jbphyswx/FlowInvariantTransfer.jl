module FlowInvariantTransferFSHExt

using FastSphericalHarmonics: FastSphericalHarmonics as FSH
using FFTW: FFTW
using FlowInvariantTransfer: FlowInvariantTransfer as FIT
using ComputationalBackends: ComputationalBackends
using FlowTransformBindings: FlowTransformBindings as FTB

# The spin transforms run FastTransforms, so each goes through `FTB.with_fasttransforms_threads`, which
# sets its OpenMP count on the calling OS thread to `n` and restores it after, and reads its plans from
# the workspace's `plans`. `spinsph_eth`/`spinsph_ethbar` act on coefficients in Julia and need no guard.
_transform(F, s, plans, n::Int) =
    FTB.with_fasttransforms_threads(() -> FSH.spinsph_transform(F, s; cache = plans), n)
_evaluate(C, s, plans, n::Int) =
    FTB.with_fasttransforms_threads(() -> FSH.spinsph_evaluate(C, s; cache = plans), n)

# FastTransforms' thread count for `execution`: one on `SerialBackend()`, `Threads.nthreads()` on
# `ThreadedBackend()`. The regular-grid transfer runs on the host and refuses every other backend.
function _fsh_threads(execution::ComputationalBackends.AbstractExecutionBackend)
    FIT.Spherical._validate_spherical_backends(nothing, execution, :regular)
    return FIT._library_threads(FIT.Types.resolve_execution(execution))
end

# The spin-0 and spin-1 transforms' plans on the grids of `N` colatitudes for each `N` in `Ns`, built
# by one transform and one evaluation of a zero field each; every spin transform here reads them.
# FastTransforms builds them on the libfftw3 FFTW.jl loads, whose planner is process-global, so they
# are built under FFTW.jl's planner lock, the planner at one thread.
function _warmed_plans(Ns, n::Int)
    plans = FSH.SpinSphPlanCache{ComplexF64}()
    FFTW.set_num_threads(1) do
        FTB.with_fasttransforms_threads(n) do
            for N in unique(Ns), s in (0, 1)
                F = zeros(ComplexF64, N, 2N - 1)
                FSH.spinsph_transform!(F, s; cache = plans)
                FSH.spinsph_evaluate!(F, s; cache = plans)
            end
        end
    end
    return plans
end

# ---------------------------------------------------------------------------
# Spherical spectral energy/enstrophy transfer on a regular colatitude–longitude grid, via
# FastSphericalHarmonics. Core reduction: FlowInvariantTransfer.Spherical.
#
# The horizontal gradient of a real spin-0 field f is the spin-1 "eth" field
#   ðf = -(∂_θ + i/sinθ ∂_φ) f,
# from FSH's real/SVector eth path (`spinsph_eth` on a real coefficient array returns SVector{2}
# coefficients with FSH's ±m storage sign applied). The Jacobian is then
#   J(ψ,ζ) = (1/a²) Im{ conj(ðψ)·ðζ },
# which conserves Σ_l T_E = Σ_l T_Z = 0.
# ---------------------------------------------------------------------------

# ðf = -(∂_θ + i/sinθ ∂_φ)f as a complex field, from real spin-0 coefficients `C0`.
# `spinsph_eth`/`spinsph_evaluate` allocate internally (FSH has no in-place API); the assembled complex
# grid is written into the caller-provided `out` buffer.
function _sph_grad!(out::AbstractMatrix{<:Complex}, C0::AbstractMatrix{<:Real}, plans, n::Int)
    ðC = FSH.spinsph_eth(C0, 0)             # Array{SVector{2,Float64},2}
    G  = _evaluate(ðC, 1, plans, n)         # SVector(re, im) of ðf at each grid point
    @inbounds for j in axes(G, 2), i in axes(G, 1)
        out[i, j] = complex(G[i, j][1], G[i, j][2])
    end
    return out
end

# Build the reusable work arrays for a given resolution. FastSphericalHarmonics is Float64-only, so
# every work buffer is Float64; the result vectors are in `T`, the field's float type. `dealias` fixes
# the work-grid size (2·lmax vs lmax), so it is a workspace-level choice, and `execution` the
# transforms' thread count. The FSH transforms (spinsph_transform/eth/evaluate) allocate internally on
# every call, having no in-place API; the workspace reuses the embed, Jacobian and reduction buffers.
function FIT.Spherical.SphericalTransferWorkspace(lmax::Integer; radius::Real = 1.0, dealias::Bool = true,
                                                  T::Type{<:AbstractFloat} = Float64,
                                                  execution::ComputationalBackends.AbstractExecutionBackend =
                                                      ComputationalBackends.SerialBackend())
    nthreads = _fsh_threads(execution)
    lwork = dealias ? 2 * lmax : lmax
    Nwork = lwork + 1
    Cζ = zeros(Float64, Nwork, 2Nwork - 1)
    Cψ = zeros(Float64, Nwork, 2Nwork - 1)
    Gψ = zeros(ComplexF64, Nwork, 2Nwork - 1)
    Gζ = zeros(ComplexF64, Nwork, 2Nwork - 1)
    J  = zeros(Float64, Nwork, 2Nwork - 1)
    nmode = (lmax + 1)^2
    degs = Vector{Int}(undef, nmode)
    ψv = Vector{Float64}(undef, nmode)
    ζv = Vector{Float64}(undef, nmode)
    Av = Vector{Float64}(undef, nmode)
    result = FIT.Types.SphericalTransferResult(
        collect(T, 0:lmax), zeros(T, lmax + 1), zeros(T, lmax + 1),
        zeros(T, lmax + 1), zeros(T, lmax + 1), true, 0, zero(T))
    return FIT.Spherical.SphericalTransferWorkspace(
        Cζ, Cψ, Gψ, Gζ, J, degs, ψv, ζv, Av, _warmed_plans((lmax + 1, Nwork), nthreads), result,
        Float64(radius), Int(lmax), dealias, nthreads)
end

function FIT.Spherical.calculate_spherical_transfer!(
    ws::FIT.Spherical.SphericalTransferWorkspace,
    vorticity::AbstractMatrix{<:Real},
)
    lmax = ws.lmax
    Nθ, Nφ = size(vorticity)
    (Nθ == lmax + 1 && Nφ == 2lmax + 1) || throw(ArgumentError(
        "vorticity size $((Nθ, Nφ)) does not match the workspace lmax=$lmax grid (lmax+1, 2lmax+1)."))
    a = ws.radius
    n = ws.nthreads

    # ζ̂_lm (real spinsph(0) layout). FSH's real spin transform takes any `AbstractMatrix{Float64}` and
    # widens internally, so a Float64 input goes straight in.
    Cζ0 = _transform(vorticity isa AbstractMatrix{Float64} ? vorticity : Matrix{Float64}(vorticity), 0,
                     ws.plans, n)

    # Embed into the (dealiased) work grid, recovering ψ = ∇⁻²ζ mode-by-mode. Reuses ws.Cζ/ws.Cψ.
    fill!(ws.Cζ, 0.0)
    fill!(ws.Cψ, 0.0)
    @inbounds for l in 0:lmax, m in -l:l
        i = FSH.spinsph_mode(0, l, m)
        ws.Cζ[i] = Cζ0[i]
        l ≥ 1 && (ws.Cψ[i] = -a^2 / (l * (l + 1)) * Cζ0[i])
    end

    # A = J(ψ,ζ) = (1/a²) Im{ conj(ðψ)·ðζ }; the assembled complex gradients (ws.Gψ/ws.Gζ) and ws.J are
    # reused.
    _sph_grad!(ws.Gψ, ws.Cψ, ws.plans, n)
    _sph_grad!(ws.Gζ, ws.Cζ, ws.plans, n)
    @. ws.J = imag(conj(ws.Gψ) * ws.Gζ) / a^2
    CA = _transform(ws.J, 0, ws.plans, n)                          # Â_lm

    # Flatten to per-mode arrays (reused) and reduce into the reused result vectors.
    k = 0
    @inbounds for l in 0:lmax, m in -l:l
        k += 1
        i = FSH.spinsph_mode(0, l, m)
        ws.degs[k] = l
        ws.ψv[k] = ws.Cψ[i]
        ws.ζv[k] = ws.Cζ[i]
        ws.Av[k] = CA[i]
    end
    return FIT.Spherical.spherical_transfer_reduce!(ws.result, ws.degs, ws.ψv, ws.ζv, ws.Av)
end

"""
    calculate_energy_transfer(method::SphericalTransferMethod, vorticity::AbstractMatrix;
                              dealias=true, transform=nothing, execution=SerialBackend())

Spherical spectral energy/enstrophy transfer `T_E(l)`, `T_Z(l)` (and fluxes) for 2D non-divergent
flow on the sphere, from the **vorticity field** `ζ` sampled on the FastSphericalHarmonics
equiangular colatitude–longitude grid — `size(vorticity) == (lmax+1, 2lmax+1)`, i.e. the grid from
`FastSphericalHarmonics.sph_points(lmax+1)`. Returns a [`SphericalTransferResult`](@ref) in the field's
float type; the transforms run in `Float64`, the only precision FastSphericalHarmonics provides.

The streamfunction is recovered spectrally as `ψ = ∇⁻²ζ` (`ψ̂_lm = -a²/(l(l+1)) ζ̂_lm`), so the flow
is treated as non-divergent; the `l=0` mode carries no transfer.

The advection `A = J(ψ,ζ)` is quadratic, so it has spectral content up to degree `2·lmax`. With
`dealias=true` (default) the products are evaluated on a grid resolving `2·lmax` and truncated back,
so the retained transfers `l ≤ lmax` are alias-free and conserve `Σ_l T_E = Σ_l T_Z = 0` to machine
precision. `dealias=false` computes on the native grid (aliased; conservation only for fields
band-limited well below `lmax`). `transform`, when given, must be
`SpectralBackends.FSHTSpectralBackend()`. `execution = ThreadedBackend()` runs the spherical-harmonic
transforms on `Threads.nthreads()` threads, `SerialBackend()` on one. Requires
`using FastSphericalHarmonics`.
"""
function FIT.calculate_energy_transfer(
    method::FIT.Types.SphericalTransferMethod,
    vorticity::AbstractMatrix{<:Real};
    dealias::Bool = true,
    transform = nothing,
    execution::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.SerialBackend(),
)
    FIT.Spherical._validate_spherical_backends(transform, execution, :regular)
    Nθ, Nφ = size(vorticity)
    Nφ == 2Nθ - 1 || throw(ArgumentError(
        "vorticity must lie on the FastSphericalHarmonics grid of size (lmax+1, 2lmax+1); got $((Nθ, Nφ))."))
    ws = FIT.Spherical.SphericalTransferWorkspace(Nθ - 1; radius = float(method.radius), dealias = dealias,
                                                  T = float(eltype(vorticity)), execution = execution)
    return FIT.Spherical.calculate_spherical_transfer!(ws, vorticity)
end

# ---------------------------------------------------------------------------
# Divergent horizontal-KE spectral transfer on the regular grid (full rotational + divergent flow),
# in the conventions of FlowInvariantTransfer.Spherical: skew-symmetric Lamb advection
#   A = ∇K + (iζ + ½δ) U₊,  U₊ = u_θ + i u_φ,  T(l) = Σ_m Re{conj(û)·Â}/a,  split into rot/div channels.
#
# FastSphericalHarmonics stores spin-s coefficients at the N-independent index spinsph_mode(s,l,m) =
# (l+1, col(m)), so a lower-degree coefficient array embeds into a higher-degree one by copying the
# same indices; the dealiasing (evaluate the quadratic products on the 2·lmax grid, analyse there,
# truncate to l ≤ lmax) uses this, as the barotropic path does. FSH's spin transforms allocate
# internally (no in-place API). The conventions rest on three FSH facts: the real→SVector eth gives the
# physical gradient (ð f = -∇f); the complex ethbar with the per-column pre-flip α = sign(m ≥ 0)
# recovers ð̄U₊ = -(δ + iζ); and the complex transform is orthonormal, so the coefficient inner product
# equals the physical one.
# ---------------------------------------------------------------------------

# col(m) ↔ FSH column packing (m=0→1, -1→2, +1→3, -2→4, +2→5, …); inverse used by the eth α-flip.
_fsh_colm(col) = col == 1 ? 0 : (col % 2 == 0 ? -1 : 1) * (col ÷ 2)

# ∇f as a complex spin+1 field, via the real→SVector eth path of `_sph_grad!`.
function _fsh_nabla(freal::AbstractMatrix{<:Real}, plans, n::Int)
    ðC = FSH.spinsph_eth(_transform(Matrix{Float64}(freal), 0, plans, n), 0)
    G = _evaluate(ðC, 1, plans, n)                         # SVector{2} field = ð f = -∇f
    out = Matrix{ComplexF64}(undef, size(G))
    @inbounds for j in axes(G, 2), i in axes(G, 1)
        out[i, j] = -complex(G[i, j][1], G[i, j][2])       # ∇f = -ð f
    end
    return out
end

# Vorticity ζ = k̂·∇×u and divergence δ = ∇·u (real fields) from the spin+1 velocity, via the complex
# ethbar with the FSH per-column α pre-flip: ð̄U₊ = -(δ + iζ) ⟹ δ = -Re, ζ = -Im.
function _fsh_vort_div(uθ::AbstractMatrix{<:Real}, uφ::AbstractMatrix{<:Real}, plans, n::Int)
    C = _transform(ComplexF64.(uθ .+ im .* uφ), 1, plans, n)
    @inbounds for col in axes(C, 2)
        _fsh_colm(col) ≥ 0 || (@views C[:, col] .*= -1)
    end
    Cf = _evaluate(FSH.spinsph_ethbar(C, 1), 0, plans, n)
    return -imag.(Cf), -real.(Cf)                          # ζ, δ
end

# Embed spin-s coefficients from a degree-lmax array into the degree-lwork buffer `Cbig` (same indices).
function _fsh_embed!(Cbig::AbstractMatrix, Csmall::AbstractMatrix, s::Integer, lmax::Integer)
    fill!(Cbig, zero(eltype(Cbig)))
    @inbounds for l in max(abs(s), 1):lmax, m in -l:l
        i = FSH.spinsph_mode(s, l, m); Cbig[i] = Csmall[i]
    end
    return Cbig
end

"""
    DivergentSphericalTransferWorkspace(lmax; radius=1.0, dealias=true, T=Float64, execution=SerialBackend())

Reusable buffers for the regular-grid `calculate_divergent_spherical_transfer!`: the work-grid
velocity, vorticity, divergence, kinetic energy and advection fields, the two coefficient embed
targets, the velocity-potential coefficients, and the reused `DivergentSphericalTransferResult`.
FastSphericalHarmonics has no in-place transform API, so each `spinsph_transform`/`evaluate`/`eth`
allocates its own output. Every work buffer is `Float64`: FSH's spin transforms are defined only on
`Float64`/`ComplexF64`/`SVector{2,Float64}`, so the transforms run in double precision whatever the
input element type; the result vectors are in `T`. `execution = ThreadedBackend()` runs the transforms
on `Threads.nthreads()` threads. Requires `using FastSphericalHarmonics`.
"""
function FIT.Spherical.DivergentSphericalTransferWorkspace(lmax::Integer; radius::Real = 1.0, dealias::Bool = true,
                                                           T::Type{<:AbstractFloat} = Float64,
                                                           execution::ComputationalBackends.AbstractExecutionBackend =
                                                               ComputationalBackends.SerialBackend())
    lmax ≥ 1 || throw(ArgumentError("lmax must be ≥ 1; got $lmax."))
    nthreads = _fsh_threads(execution)
    lwork = dealias ? 2 * lmax : lmax
    Nw = lwork + 1
    z() = zeros(T, lmax + 1)
    rw() = zeros(Float64, Nw, 2Nw - 1)
    result = FIT.Types.DivergentSphericalTransferResult(collect(T, 0:lmax), z(), z(), z(), z(), z(), z(),
                                                        true, 0, zero(T))
    return FIT.Spherical.DivergentSphericalTransferWorkspace(
        rw(), rw(), rw(), rw(), rw(),                       # uθw, uφw, ζw, δw, K
        zeros(ComplexF64, Nw, 2Nw - 1), zeros(ComplexF64, Nw, 2Nw - 1), rw(),   # Adv, Cw1, Cw0
        zeros(Float64, lmax + 1, 2lmax + 1),                # χc
        _warmed_plans((lmax + 1, Nw), nthreads),
        result, Float64(radius), Int(lmax), Int(lwork), dealias, nthreads)
end

function FIT.calculate_divergent_spherical_transfer!(
    ws::FIT.Spherical.DivergentSphericalTransferWorkspace,
    u_θ::AbstractMatrix{<:Real},
    u_φ::AbstractMatrix{<:Real},
)
    lmax = ws.lmax; a = ws.radius
    (size(u_θ) == (lmax + 1, 2lmax + 1) && size(u_φ) == (lmax + 1, 2lmax + 1)) || throw(ArgumentError(
        "velocity component sizes $((size(u_θ), size(u_φ))) do not match the workspace lmax=$lmax grid (lmax+1, 2lmax+1)."))

    P = ws.plans; n = ws.nthreads
    a₊ = _transform(ComplexF64.(u_θ .+ im .* u_φ), 1, P, n)        # spin+1 velocity coefficients (lmax)
    ζin, δin = _fsh_vort_div(u_θ, u_φ, P, n)                       # vorticity/divergence fields (lmax grid)
    δc = _transform(δin, 0, P, n)                                  # reused for the work grid and for χ

    # Evaluate velocity + ζ,δ on the (dealiased) work grid so the quadratic advection is alias-free.
    uθw = ws.uθw; uφw = ws.uφw; ζw = ws.ζw; δw = ws.δw
    if ws.dealias
        Uw = _evaluate(_fsh_embed!(ws.Cw1, a₊, 1, lmax), 1, P, n)
        @. uθw = real(Uw); @. uφw = imag(Uw)
        ζw .= _evaluate(_fsh_embed!(ws.Cw0, _transform(ζin, 0, P, n), 0, lmax), 0, P, n)
        δw .= _evaluate(_fsh_embed!(ws.Cw0, δc, 0, lmax), 0, P, n)
    else
        copyto!(uθw, u_θ); copyto!(uφw, u_φ); copyto!(ζw, ζin); copyto!(δw, δin)
    end

    # Skew-symmetric advection A = ∇K + (iζ + ½δ) U₊ on the work grid; analyse (spin+1).
    @. ws.K = 0.5 * (uθw^2 + uφw^2)
    ∇K = _fsh_nabla(ws.K, P, n)
    @. ws.Adv = ∇K + (im * ζw + 0.5 * δw) * (uθw + im * uφw)
    Â = _transform(ws.Adv, 1, P, n)                                # spin+1 advection coefficients (lwork)

    # Divergent-channel velocity coefficients: anti = spin+1 coeffs of ∇χ, χ = ∇⁻²δ (lmax).
    χc = ws.χc
    fill!(χc, 0.0)
    @inbounds for l in 1:lmax, m in -l:l
        i = FSH.spinsph_mode(0, l, m); χc[i] = -δc[i] / (l * (l + 1))
    end
    anti = _transform(ComplexF64.(_fsh_nabla(_evaluate(χc, 0, P, n), P, n)), 1, P, n)

    # Per-degree channels T_rot = Σ_m Re{sym* Â}, T_div = Σ_m Re{anti* Â}; sym = a₊ − anti. The lwork Â
    # is truncated to l ≤ lmax automatically (same spinsph_mode index in both arrays). Single 1/a factor.
    Trot = ws.result.rotational_transfer; Tdiv = ws.result.divergent_transfer
    fill!(Trot, 0.0); fill!(Tdiv, 0.0)
    @inbounds for l in 1:lmax, m in -l:l
        i = FSH.spinsph_mode(1, l, m)
        tot = real(conj(a₊[i]) * Â[i]); dv = real(conj(anti[i]) * Â[i])
        Trot[l+1] += (tot - dv) / a
        Tdiv[l+1] += dv / a
    end
    return FIT.Spherical.divergent_transfer_finalize!(ws.result)
end

"""
    calculate_energy_transfer(method::DivergentSphericalTransferMethod,
                              velocity::Tuple{<:AbstractMatrix,<:AbstractMatrix};
                              dealias=true, transform=nothing, execution=SerialBackend())

Divergent horizontal-KE spectral transfer for the full (rotational + divergent) flow on a regular
colatitude–longitude grid, from the horizontal velocity `(u_θ, u_φ)` on the FastSphericalHarmonics
grid — each `size == (lmax+1, 2lmax+1)`. Returns a [`DivergentSphericalTransferResult`](@ref) in the
field's float type, the transforms running in `Float64`. The advection is dealiased at degree `2·lmax`
(`dealias=false` skips it). `execution = ThreadedBackend()` runs the transforms on
`Threads.nthreads()` threads. Requires `using FastSphericalHarmonics`.
"""
function FIT.calculate_energy_transfer(
    method::FIT.Types.DivergentSphericalTransferMethod,
    velocity::Tuple{<:AbstractMatrix, <:AbstractMatrix};
    dealias::Bool = true,
    transform = nothing,
    execution::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.SerialBackend(),
)
    FIT.Spherical._validate_spherical_backends(transform, execution, :regular)
    u_θ, u_φ = velocity
    size(u_θ) == size(u_φ) || throw(ArgumentError(
        "velocity components must have equal size; got $((size(u_θ), size(u_φ)))."))
    Nθ, Nφ = size(u_θ)
    Nφ == 2Nθ - 1 || throw(ArgumentError(
        "velocity must lie on the FastSphericalHarmonics grid of size (lmax+1, 2lmax+1); got $((Nθ, Nφ))."))
    ws = FIT.Spherical.DivergentSphericalTransferWorkspace(Nθ - 1; radius = float(method.radius), dealias = dealias,
                                                           T = float(promote_type(eltype(u_θ), eltype(u_φ))),
                                                           execution = execution)
    return FIT.calculate_divergent_spherical_transfer!(ws, u_θ, u_φ)
end

end # module FlowInvariantTransferFSHExt
