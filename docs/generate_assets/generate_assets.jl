"""
Generate static figure assets for FlowInvariantTransfer.jl docs and README.md.

Run from the repo root:
    julia --project=docs/generate_assets docs/generate_assets/generate_assets.jl

Generates:
  - energy_transfer.png  : final-state static figure (4 panels)
  - energy_transfer.gif  : animation of cascade development from t=0 to t=10
"""

using FlowInvariantTransfer: FlowInvariantTransfer as FET
using FFTW: FFTW
using CairoMakie: CairoMakie

const ASSETS_DIR = joinpath(@__DIR__, "..", "src", "assets")
CairoMakie.mkpath(ASSETS_DIR)

# ─── 3D Taylor-Green Vortex pseudospectral NS solver ─────────────────────────
# Initial condition: u=sin(x)cos(y)cos(z), v=-cos(x)sin(y)cos(z), w=0
# RK2 with viscous integrating factor (same algorithm as reference implementation).
# Energy concentrates at k=√2 shell at t=0 and cascades forward.

function run_tgv(; N=32, ν=0.005, dt=0.02, steps=150, frame_every=5)
    L = 2π

    # 3D wavenumber arrays (FFTW order: 0,1,...,N/2,-N/2+1,...,-1)
    k1d = collect(FFTW.fftfreq(N, N))  # integers 0..N/2, -N/2+1..-1
    KX  = reshape(k1d, N, 1, 1) .* ones(1, N, N)
    KY  = reshape(k1d, 1, N, 1) .* ones(N, 1, N)
    KZ  = reshape(k1d, 1, 1, N) .* ones(N, N, 1)
    K2  = KX.^2 .+ KY.^2 .+ KZ.^2
    K2[1,1,1] = 1.0   # avoid /0; reset after

    # 2/3 dealiasing mask
    kmax = N ÷ 3
    dmask = (abs.(KX) .< kmax) .& (abs.(KY) .< kmax) .& (abs.(KZ) .< kmax)

    # Viscous integrating factor exp(-ν k² dt) applied each half-step
    visc = exp.(-ν .* K2 .* dt)
    K2[1,1,1] = 0.0

    # Initial condition in physical space
    x = range(0, L; length=N+1)[1:N]
    X = reshape(x, N, 1, 1); Y = reshape(x, 1, N, 1); Z = reshape(x, 1, 1, N)
    u0 =  sin.(X) .* cos.(Y) .* cos.(Z)
    v0 = -cos.(X) .* sin.(Y) .* cos.(Z)
    w0 =  zeros(N, N, N)

    uh = FFTW.fft(u0)
    vh = FFTW.fft(v0)
    wh = FFTW.fft(w0)

    function fftn(a)  FFTW.fft(a)  end
    function ifftn(a) FFTW.ifft(a) end

    function dealias!(f)
        f .*= dmask
        return f
    end

    function rhs(uh, vh, wh)
        u = real.(ifftn(uh)); v = real.(ifftn(vh)); w = real.(ifftn(wh))
        ux = real.(ifftn(im .* KX .* uh)); uy = real.(ifftn(im .* KY .* uh)); uz = real.(ifftn(im .* KZ .* uh))
        vx = real.(ifftn(im .* KX .* vh)); vy = real.(ifftn(im .* KY .* vh)); vz = real.(ifftn(im .* KZ .* vh))
        wx = real.(ifftn(im .* KX .* wh)); wy = real.(ifftn(im .* KY .* wh)); wz = real.(ifftn(im .* KZ .* wh))
        Nu = dealias!(fftn(u.*ux .+ v.*uy .+ w.*uz))
        Nv = dealias!(fftn(u.*vx .+ v.*vy .+ w.*vz))
        Nw = dealias!(fftn(u.*wx .+ v.*wy .+ w.*wz))
        # pressure projection
        divN = KX.*Nu .+ KY.*Nv .+ KZ.*Nw
        divN[1,1,1] = 0
        P = divN ./ (K2 .+ 1e-10)
        return -Nu .+ P.*KX, -Nv .+ P.*KY, -Nw .+ P.*KZ
    end

    # Storage for animation frames
    frames_t    = Float64[]
    frames_Eslice = Matrix{Float64}[]   # KE slice at z=N÷2
    frames_T    = Matrix{Float64}[]
    frames_Tnet = Vector{Float64}[]

    # FET binning (unit shells in 3D; k=√(kx²+ky²+kz²))
    # We compute T(n,m) using FET on the 3D→packed representation.
    # FET expects (N,N,N,3) velocity array or we use the 2D interface on slices.
    # For simplicity: use the built-in FET 3D wavenumber grid + shell-to-shell.
    ks3 = FET.wavenumber_grid((N, N, N), (L, L, L))
    b   = FET.LinearBinning(2π/L)   # unit shells

    function compute_diagnostics(uh, vh, wh)
        # Pack into (N,N,N,3) array expected by FET
        û3 = zeros(ComplexF64, N, N, N, 3)
        û3[:,:,:,1] .= uh; û3[:,:,:,2] .= vh; û3[:,:,:,3] .= wh
        s2s = FET.calculate_shell_to_shell_transfer(û3, ks3;
            binning=b, dealiasing=true, verify_antisymmetry=false,
            backend=FET.FFTBackend())
        return s2s.transfer_matrix, s2s.net_transfer
    end

    println("  Simulating TGV (N=$N, steps=$steps)...")
    for i in 0:steps
        if i % frame_every == 0
            t = i * dt
            print("  t=$(round(t; digits=2))  ")
            # KE slice at z=N÷2 for flow visualisation
            u_p = real.(ifftn(uh))
            v_p = real.(ifftn(vh))
            w_p = real.(ifftn(wh))
            E_slice = 0.5 .* (u_p[:, :, N÷2].^2 .+ v_p[:, :, N÷2].^2 .+ w_p[:, :, N÷2].^2)
            T_mat, T_net = compute_diagnostics(uh, vh, wh)
            push!(frames_t,      t)
            push!(frames_Eslice, E_slice)
            push!(frames_T,      T_mat)
            push!(frames_Tnet,   T_net)
        end
        i == steps && break

        # RK2 with viscous integrating factor
        ru1, rv1, rw1 = rhs(uh, vh, wh)
        utmp = (uh .+ dt .* ru1) .* visc
        vtmp = (vh .+ dt .* rv1) .* visc
        wtmp = (wh .+ dt .* rw1) .* visc
        ru2, rv2, rw2 = rhs(utmp, vtmp, wtmp)
        uh = (uh .+ 0.5 .* dt .* (ru1 .+ ru2)) .* visc
        vh = (vh .+ 0.5 .* dt .* (rv1 .+ rv2)) .* visc
        wh = (wh .+ 0.5 .* dt .* (rw1 .+ rw2)) .* visc
    end
    println()
    return frames_t, frames_Eslice, frames_T, frames_Tnet, uh, vh, wh, ks3, b
end

# ─── Figure helpers ──────────────────────────────────────────────────────────

function make_frame_figure(t, E_slice, T_mat, T_net; K_max=nothing)
    N_sh = length(T_net)
    K_max = something(K_max, N_sh)

    # Trim to active shells
    T_sub = T_mat[1:K_max, 1:K_max]
    T_net_sub = T_net[1:K_max]
    shells = 1:K_max
    T_lim = max(maximum(abs, T_sub), 1e-20)

    fig = CairoMakie.Figure(size=(1200, 500), fontsize=13)
    CairoMakie.Label(fig[0, 1:3],
        "3D Taylor-Green Vortex Energy Cascade  (t = $(round(t; digits=2)))",
        fontsize=16, font=:bold, tellwidth=false)

    # Panel 1: KE slice
    ax1 = CairoMakie.Axis(fig[1, 1], title="Kinetic Energy  (z=π slice)",
        xlabel="x", ylabel="y", aspect=CairoMakie.DataAspect())
    x_phys = range(0, 2π; length=size(E_slice,1)+1)[1:end-1]
    hm1 = CairoMakie.heatmap!(ax1, x_phys, x_phys, E_slice,
        colormap=:inferno, colorrange=(0, maximum(E_slice) + 1e-20))
    CairoMakie.Colorbar(fig[1, 2], hm1, label="KE", width=12)

    # Panel 2: T(n,m) matrix
    ax2 = CairoMakie.Axis(fig[1, 3], title="Shell-to-Shell Transfer T(n,m)",
        xlabel="Source shell m", ylabel="Receiver shell n",
        aspect=CairoMakie.DataAspect())
    hm2 = CairoMakie.heatmap!(ax2, collect(shells), collect(shells), T_sub,
        colormap=:RdBu_9, colorrange=(-T_lim, T_lim))
    CairoMakie.lines!(ax2, [0.5, K_max+0.5], [0.5, K_max+0.5],
        color=:black, linewidth=1.5, linestyle=:dash)
    CairoMakie.Colorbar(fig[1, 4], hm2, label="T(n,m)", width=12)

    # Panel 3: Net transfer
    ax3 = CairoMakie.Axis(fig[1, 5], title="Net Transfer  Σₘ T(n,m)",
        xlabel="Shell n", ylabel="Net energy gain")
    bar_colors = [v >= 0 ? CairoMakie.RGBf(0.27,0.51,0.71) :
                           CairoMakie.RGBf(0.84,0.15,0.16) for v in T_net_sub]
    CairoMakie.barplot!(ax3, collect(shells), T_net_sub, color=bar_colors, gap=0.15)
    CairoMakie.hlines!(ax3, [0]; color=:black, linewidth=1.0)

    CairoMakie.colgap!(fig.layout, 1, 4); CairoMakie.colgap!(fig.layout, 2, 20)
    CairoMakie.colgap!(fig.layout, 3, 4); CairoMakie.colgap!(fig.layout, 4, 20)
    return fig
end

# ─── Main ─────────────────────────────────────────────────────────────────────

println("Running 3D TGV simulation...")
frames_t, frames_E, frames_T, frames_Tnet, uh, vh, wh, ks3, b =
    run_tgv(; N=32, ν=0.005, dt=0.02, steps=500, frame_every=10)

# Determine display shell range from final frame
T_final = frames_T[end]
T_net_final = frames_Tnet[end]
T_lim_global = maximum(abs, T_final)
# Active shells: those with any entry > 1% of max
K_max_disp = findlast(1:size(T_final,1)) do n
    any(abs.(T_final[n,:]) .> 0.01*T_lim_global) ||
    any(abs.(T_final[:,n]) .> 0.01*T_lim_global)
end
K_max_disp = something(K_max_disp, min(10, size(T_final,1)))

println("Active shells: 1:$K_max_disp")

# ── Static PNG: final state ───────────────────────────────────────────────────
println("Saving static figure...")
# Snapshot at t≈5 (frame index where t is closest to 5)
snap_idx = argmin(abs.(frames_t .- 5.0))
fig_final = make_frame_figure(frames_t[snap_idx], frames_E[snap_idx], frames_T[snap_idx], frames_Tnet[snap_idx];
    K_max=K_max_disp)
CairoMakie.save(joinpath(ASSETS_DIR, "energy_transfer.png"), fig_final; px_per_unit=2)
println("Saved: energy_transfer.png")

# ── GIF animation ─────────────────────────────────────────────────────────────
println("Rendering animation ($(length(frames_t)) frames)...")
fig_anim = CairoMakie.Figure(size=(1200, 500), fontsize=13)

ax_E  = CairoMakie.Axis(fig_anim[1,1], title="Kinetic Energy (z=π)", xlabel="x", ylabel="y", aspect=CairoMakie.DataAspect())
ax_T  = CairoMakie.Axis(fig_anim[1,3], title="Shell-to-Shell T(n,m)", xlabel="Source m", ylabel="Receiver n", aspect=CairoMakie.DataAspect())
ax_N  = CairoMakie.Axis(fig_anim[1,5], title="Net Transfer Σₘ T(n,m)", xlabel="Shell n", ylabel="Net gain")

x_phys = range(0, 2π; length=size(frames_E[1],1)+1)[1:end-1]
shells  = 1:K_max_disp

# Global color limits across all frames for consistent colorscale
E_max_global = maximum(maximum, frames_E)
T_lim_anim   = maximum(maximum.(abs, frames_T))
N_net_lim    = maximum(maximum.(abs, frames_Tnet))

hm_E = CairoMakie.heatmap!(ax_E, x_phys, x_phys, frames_E[1],
    colormap=:inferno, colorrange=(0, E_max_global))
CairoMakie.Colorbar(fig_anim[1,2], hm_E, label="KE", width=12)

hm_T = CairoMakie.heatmap!(ax_T, collect(shells), collect(shells),
    frames_T[1][1:K_max_disp, 1:K_max_disp],
    colormap=:RdBu_9, colorrange=(-T_lim_anim, T_lim_anim))
CairoMakie.lines!(ax_T, [0.5, K_max_disp+0.5], [0.5, K_max_disp+0.5],
    color=:black, linewidth=1.5, linestyle=:dash)
CairoMakie.Colorbar(fig_anim[1,4], hm_T, label="T(n,m)", width=12)

net_obs   = CairoMakie.Observable(frames_Tnet[1][1:K_max_disp])
title_obs = CairoMakie.Observable("3D TGV Energy Cascade  (t = 0.00)")

bar_color_fn(v) = v >= 0 ? CairoMakie.RGBf(0.27,0.51,0.71) : CairoMakie.RGBf(0.84,0.15,0.16)
CairoMakie.barplot!(ax_N, collect(shells), net_obs,
    color = CairoMakie.@lift([bar_color_fn(v) for v in $net_obs]),
    gap=0.15)
CairoMakie.hlines!(ax_N, [0]; color=:black, linewidth=1.0)
CairoMakie.ylims!(ax_N, -N_net_lim*1.1, N_net_lim*1.1)

CairoMakie.Label(fig_anim[0, 1:5], title_obs, fontsize=16, font=:bold, tellwidth=false)

CairoMakie.colgap!(fig_anim.layout, 1, 4); CairoMakie.colgap!(fig_anim.layout, 2, 20)
CairoMakie.colgap!(fig_anim.layout, 3, 4); CairoMakie.colgap!(fig_anim.layout, 4, 20)

CairoMakie.record(fig_anim, joinpath(ASSETS_DIR, "energy_transfer.gif"),
        1:length(frames_t); framerate=8) do i
    title_obs[] = "3D TGV Energy Cascade  (t = $(round(frames_t[i]; digits=2)))"
    hm_E[3]     = frames_E[i]
    hm_T[3]     = frames_T[i][1:K_max_disp, 1:K_max_disp]
    net_obs[]   = frames_Tnet[i][1:K_max_disp]
end

println("Saved: energy_transfer.gif")
println("Done.")
