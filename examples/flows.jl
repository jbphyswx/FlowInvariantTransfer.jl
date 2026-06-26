"""
Canonical evolved flows for the FlowInvariantTransfer.jl examples.

A static random field has *no developed cascade* — its transfer spectrum is meaningless noise.
To make the examples actually *show* the physics, we evolve canonical initial conditions with a
small pseudospectral solver until a cascade is established, then diagnose that snapshot:

  - `evolve_taylor_green`  : 3D Taylor–Green vortex (forward energy cascade), optional passive scalar.
  - `evolve_orszag_tang`   : 2D Orszag–Tang vortex (canonical incompressible-MHD test).

Velocities are returned as raw `fft` Fourier coefficients packed `(ns..., D)` — exactly the
convention `FlowInvariantTransfer` expects (it applies the inverse transform internally).
"""

using FFTW: FFTW

# 2/3-rule dealias mask for an N-point axis (FFTW order).
_dealias_mask_1d(N) = [abs(k) < N ÷ 3 for k in FFTW.fftfreq(N, N)]

# ─────────────────────────────────────────────────────────────────────────────
# 3D incompressible Navier–Stokes solver (shared), optional passive scalar
# ─────────────────────────────────────────────────────────────────────────────
# Pseudospectral RK2 + viscous integrating factor + 2/3 dealiasing + pressure projection.
# `u0fun(X,Y,Z)` returns the three physical velocity components as a tuple; if `θ0fun` is given a
# passive scalar is advected alongside. Returns `(û, ks, L)` or `(û, θ̂, ks, L)`.
function _evolve_ns3d(u0fun; N, ν, dt, steps, θ0fun=nothing, κ=0.005)
    L = 2π
    k1d = collect(FFTW.fftfreq(N, N))
    KX = reshape(k1d, N, 1, 1) .* ones(1, N, N)
    KY = reshape(k1d, 1, N, 1) .* ones(N, 1, N)
    KZ = reshape(k1d, 1, 1, N) .* ones(N, N, 1)
    K2 = KX.^2 .+ KY.^2 .+ KZ.^2
    K2[1, 1, 1] = 1.0
    dmask = _dealias_mask_1d(N)
    dmask3 = reshape(dmask, N,1,1) .& reshape(dmask, 1,N,1) .& reshape(dmask, 1,1,N)
    visc  = exp.(-ν .* K2 .* dt)
    cdiff = exp.(-κ .* K2 .* dt)
    K2[1, 1, 1] = 0.0

    x = range(0, L; length=N+1)[1:N]
    X = reshape(x, N, 1, 1); Y = reshape(x, 1, N, 1); Z = reshape(x, 1, 1, N)
    u0, v0, w0 = u0fun(X, Y, Z)
    uh = FFTW.fft(u0); vh = FFTW.fft(v0); wh = FFTW.fft(w0)
    with_scalar = θ0fun !== nothing
    θh = with_scalar ? FFTW.fft(θ0fun(X, Y, Z)) : nothing

    dealias!(f) = (f .*= dmask3; f)
    proj(Nu, Nv, Nw) = begin
        d = KX.*Nu .+ KY.*Nv .+ KZ.*Nw; d[1,1,1] = 0
        P = d ./ (K2 .+ 1e-30)
        (-Nu .+ P.*KX, -Nv .+ P.*KY, -Nw .+ P.*KZ)
    end
    function rhs(uh, vh, wh)
        u = real.(FFTW.ifft(uh)); v = real.(FFTW.ifft(vh)); w = real.(FFTW.ifft(wh))
        ux = real.(FFTW.ifft(im.*KX.*uh)); uy = real.(FFTW.ifft(im.*KY.*uh)); uz = real.(FFTW.ifft(im.*KZ.*uh))
        vx = real.(FFTW.ifft(im.*KX.*vh)); vy = real.(FFTW.ifft(im.*KY.*vh)); vz = real.(FFTW.ifft(im.*KZ.*vh))
        wx = real.(FFTW.ifft(im.*KX.*wh)); wy = real.(FFTW.ifft(im.*KY.*wh)); wz = real.(FFTW.ifft(im.*KZ.*wh))
        Nu = dealias!(FFTW.fft(u.*ux .+ v.*uy .+ w.*uz))
        Nv = dealias!(FFTW.fft(u.*vx .+ v.*vy .+ w.*vz))
        Nw = dealias!(FFTW.fft(u.*wx .+ v.*wy .+ w.*wz))
        proj(Nu, Nv, Nw)
    end
    function rhs_scalar(uh, vh, wh, θh)
        u = real.(FFTW.ifft(uh)); v = real.(FFTW.ifft(vh)); w = real.(FFTW.ifft(wh))
        θx = real.(FFTW.ifft(im.*KX.*θh)); θy = real.(FFTW.ifft(im.*KY.*θh)); θz = real.(FFTW.ifft(im.*KZ.*θh))
        -dealias!(FFTW.fft(u.*θx .+ v.*θy .+ w.*θz))
    end

    for _ in 1:steps
        ru1, rv1, rw1 = rhs(uh, vh, wh)
        rθ1 = with_scalar ? rhs_scalar(uh, vh, wh, θh) : nothing
        ut = (uh .+ dt.*ru1).*visc; vt = (vh .+ dt.*rv1).*visc; wt = (wh .+ dt.*rw1).*visc
        θt = with_scalar ? (θh .+ dt.*rθ1).*cdiff : nothing
        ru2, rv2, rw2 = rhs(ut, vt, wt)
        uh = (uh .+ 0.5dt.*(ru1.+ru2)).*visc
        vh = (vh .+ 0.5dt.*(rv1.+rv2)).*visc
        wh = (wh .+ 0.5dt.*(rw1.+rw2)).*visc
        if with_scalar
            rθ2 = rhs_scalar(ut, vt, wt, θt)
            θh = (θh .+ 0.5dt.*(rθ1.+rθ2)).*cdiff
        end
    end

    ks = ((2π/L).*k1d, (2π/L).*k1d, (2π/L).*k1d)
    û  = cat(uh, vh, wh; dims=4)
    return with_scalar ? (û, θh, ks, L) : (û, ks, L)
end

"""
    evolve_taylor_green(; N=32, ν=0.005, dt=0.02, steps=250, with_scalar=false, κ=0.005)

Evolve the 3D Taylor–Green vortex `u=sin x cos y cos z, v=−cos x sin y cos z, w=0` (energy starts
on the `k=√3` shell and cascades forward). Returns `(û, ks, L)` — or `(û, θ̂, ks, L)` when
`with_scalar=true`, where the scalar `θ₀ = sin x` is stirred by the flow.
"""
function evolve_taylor_green(; N=32, ν=0.005, dt=0.02, steps=250, with_scalar=false, κ=0.005)
    u0(X, Y, Z) = (@.(sin(X)*cos(Y)*cos(Z) + 0*Y*Z), @.(-cos(X)*sin(Y)*cos(Z) + 0*X*Z), zeros(size(X,1), size(Y,2), size(Z,3)))
    θ0 = with_scalar ? ((X, Y, Z) -> @.(sin(X) + 0*Y*Z + 0*X)) : nothing
    return _evolve_ns3d(u0; N=N, ν=ν, dt=dt, steps=steps, θ0fun=θ0, κ=κ)
end

"""
    evolve_abc_flow(; N=32, ν=0.01, dt=0.02, steps=250, A=1.0, B=0.7, C=0.43)

Evolve the maximally-helical ABC (Arnold–Beltrami–Childress) flow
`u = A sin z + C cos y,  v = B sin x + A cos z,  w = C sin y + B cos x`. With unequal `A,B,C`
the flow is chaotic and cascades while retaining strong net helicity — so the ±-helical fluxes
`Π⁺(K)`, `Π⁻(K)` differ markedly. Returns `(û, ks, L)`.
"""
function evolve_abc_flow(; N=32, ν=0.01, dt=0.02, steps=250, A=1.0, B=0.7, C=0.43)
    u0(X, Y, Z) = (@.(A*sin(Z) + C*cos(Y) + 0*X), @.(B*sin(X) + A*cos(Z) + 0*Y), @.(C*sin(Y) + B*cos(X) + 0*Z))
    return _evolve_ns3d(u0; N=N, ν=ν, dt=dt, steps=steps)
end

# ─────────────────────────────────────────────────────────────────────────────
# 2D incompressible Navier–Stokes turbulence (decaying, from a band-limited IC)
# ─────────────────────────────────────────────────────────────────────────────
"""
    evolve_2d_turbulence(; N=24, ν=2e-3, dt=0.01, steps=300, seed=7)

Evolve 2D decaying turbulence from a divergence-free, band-limited random initial condition
(energy seeded on shells ~3–5). Develops a 2D cascade with well-populated intermediate shells —
a good small-`N` flow for the (cost-limited) mode-to-mode diagnostic. Returns `(û, ks, L)` with
the velocity packed `(N, N, 2)`. Phases come from a fixed linear-congruential sequence keyed by
`seed`, so no RNG dependency.
"""
function evolve_2d_turbulence(; N=24, ν=2e-3, dt=0.01, steps=300, seed=7)
    L = 2π
    k1d = collect(FFTW.fftfreq(N, N))
    KX = reshape(k1d, N, 1) .* ones(1, N)
    KY = reshape(k1d, 1, N) .* ones(N, 1)
    K2 = KX.^2 .+ KY.^2; K2[1,1] = 1.0
    dmask = _dealias_mask_1d(N); dmask2 = reshape(dmask, N, 1) .& reshape(dmask, 1, N)
    visc = exp.(-ν .* K2 .* dt)

    # Deterministic streamfunction with energy on shells 3..5 (LCG phases — no RNG dependency).
    s = UInt64(seed); nextf() = (s = (6364136223846793005*s + 1442695040888963407) % 0xFFFFFFFFFFFFFFFF; (s >> 11) / 2.0^53)
    ψh = zeros(ComplexF64, N, N)
    for i in 1:N, j in 1:N
        kk = sqrt(KX[i,j]^2 + KY[i,j]^2)
        if 2.5 <= kk <= 5.5
            ψh[i,j] = (nextf() - 0.5 + im*(nextf() - 0.5)) / kk^2
        end
    end
    # Hermitian-symmetrize so the physical field is real.
    for i in 1:N, j in 1:N
        ci = i == 1 ? 1 : N - i + 2; cj = j == 1 ? 1 : N - j + 2
        if (ci, cj) > (i, j); ψh[ci, cj] = conj(ψh[i, j]); end
    end
    ψh[1,1] = 0
    K2[1,1] = 0.0
    uh =  im .* KY .* ψh    # u =  ∂ψ/∂y
    vh = -im .* KX .* ψh    # v = -∂ψ/∂x

    dealias!(f) = (f .*= dmask2; f)
    proj2(Fx, Fy) = begin
        d = KX.*Fx .+ KY.*Fy; d[1,1] = 0; P = d ./ (K2 .+ 1e-30)
        (Fx .- P.*KX, Fy .- P.*KY)
    end
    function rhs(uh, vh)
        u = real.(FFTW.ifft(uh)); v = real.(FFTW.ifft(vh))
        ux = real.(FFTW.ifft(im.*KX.*uh)); uy = real.(FFTW.ifft(im.*KY.*uh))
        vx = real.(FFTW.ifft(im.*KX.*vh)); vy = real.(FFTW.ifft(im.*KY.*vh))
        Nu = dealias!(FFTW.fft(u.*ux .+ v.*uy)); Nv = dealias!(FFTW.fft(u.*vx .+ v.*vy))
        proj2(-Nu, -Nv)
    end
    for _ in 1:steps
        ru1, rv1 = rhs(uh, vh)
        ut = (uh .+ dt.*ru1).*visc; vt = (vh .+ dt.*rv1).*visc
        ru2, rv2 = rhs(ut, vt)
        uh = (uh .+ 0.5dt.*(ru1.+ru2)).*visc
        vh = (vh .+ 0.5dt.*(rv1.+rv2)).*visc
    end
    ks = ((2π/L).*k1d, (2π/L).*k1d)
    return cat(uh, vh; dims=3), ks, L
end

# ─────────────────────────────────────────────────────────────────────────────
# 2D Orszag–Tang vortex (incompressible MHD)
# ─────────────────────────────────────────────────────────────────────────────
"""
    evolve_orszag_tang(; N=128, ν=2e-3, η=2e-3, dt=0.01, steps=150)

Evolve the canonical 2D Orszag–Tang vortex — `u = (−sin y, sin x)`, `b = (−sin y, sin 2x)` — with
a pseudospectral RK2 + integrating-factor MHD solver (∂ₜu = −(u·∇)u + (b·∇)b − ∇P,
∂ₜb = −(u·∇)b + (b·∇)u). It develops current sheets and a forward energy cascade. Returns
`(û, b̂, ks, L)` with velocity/field packed `(N, N, 2)`.
"""
function evolve_orszag_tang(; N=128, ν=2e-3, η=2e-3, dt=0.01, steps=150)
    L = 2π
    k1d = collect(FFTW.fftfreq(N, N))
    KX = reshape(k1d, N, 1) .* ones(1, N)
    KY = reshape(k1d, 1, N) .* ones(N, 1)
    K2 = KX.^2 .+ KY.^2; K2[1,1] = 1.0
    dmask = _dealias_mask_1d(N)
    dmask2 = reshape(dmask, N, 1) .& reshape(dmask, 1, N)
    viscU = exp.(-ν .* K2 .* dt); viscB = exp.(-η .* K2 .* dt)
    K2[1,1] = 0.0

    x = range(0, L; length=N+1)[1:N]
    X = reshape(x, N, 1); Y = reshape(x, 1, N)
    uh = FFTW.fft(@. -sin(Y) + 0*X); vh = FFTW.fft(@. sin(X) + 0*Y)
    bxh = FFTW.fft(@. -sin(Y) + 0*X); byh = FFTW.fft(@. sin(2X) + 0*Y)

    dealias!(f) = (f .*= dmask2; f)
    proj2(Fx, Fy) = begin
        d = KX.*Fx .+ KY.*Fy; d[1,1] = 0
        P = d ./ (K2 .+ 1e-30)
        (Fx .- P.*KX, Fy .- P.*KY)
    end
    adv(ax, ay, fxh, fyh) = begin   # (a·∇)f for vector f, returns spectral components
        fx_x = real.(FFTW.ifft(im.*KX.*fxh)); fx_y = real.(FFTW.ifft(im.*KY.*fxh))
        fy_x = real.(FFTW.ifft(im.*KX.*fyh)); fy_y = real.(FFTW.ifft(im.*KY.*fyh))
        (dealias!(FFTW.fft(ax.*fx_x .+ ay.*fx_y)), dealias!(FFTW.fft(ax.*fy_x .+ ay.*fy_y)))
    end
    function rhs(uh, vh, bxh, byh)
        u = real.(FFTW.ifft(uh));  v = real.(FFTW.ifft(vh))
        bx = real.(FFTW.ifft(bxh)); by = real.(FFTW.ifft(byh))
        uadvu_x, uadvu_y = adv(u, v, uh, vh)        # (u·∇)u
        badvb_x, badvb_y = adv(bx, by, bxh, byh)    # (b·∇)b
        uadvb_x, uadvb_y = adv(u, v, bxh, byh)      # (u·∇)b
        badvu_x, badvu_y = adv(bx, by, uh, vh)      # (b·∇)u
        ru_x, ru_y = proj2(-uadvu_x .+ badvb_x, -uadvu_y .+ badvb_y)
        rb_x, rb_y = proj2(-uadvb_x .+ badvu_x, -uadvb_y .+ badvu_y)
        (ru_x, ru_y, rb_x, rb_y)
    end

    for _ in 1:steps
        a1 = rhs(uh, vh, bxh, byh)
        ut = (uh .+ dt.*a1[1]).*viscU; vt = (vh .+ dt.*a1[2]).*viscU
        bxt = (bxh .+ dt.*a1[3]).*viscB; byt = (byh .+ dt.*a1[4]).*viscB
        a2 = rhs(ut, vt, bxt, byt)
        uh  = (uh  .+ 0.5dt.*(a1[1].+a2[1])).*viscU
        vh  = (vh  .+ 0.5dt.*(a1[2].+a2[2])).*viscU
        bxh = (bxh .+ 0.5dt.*(a1[3].+a2[3])).*viscB
        byh = (byh .+ 0.5dt.*(a1[4].+a2[4])).*viscB
    end

    ks = ((2π/L).*k1d, (2π/L).*k1d)
    return cat(uh, vh; dims=3), cat(bxh, byh; dims=3), ks, L
end
