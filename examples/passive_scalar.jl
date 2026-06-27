"""
Passive-Scalar Variance Transfer Example — FlowInvariantTransfer.jl

A passive scalar θ stirred by a developed 3D Taylor–Green vortex develops a forward
variance cascade. `calculate_scalar_flux` gives the variance transfer T_θ(k) and flux Π_θ(K)
(positive ⇒ forward), and `calculate_scalar_shell_to_shell_transfer` the antisymmetric T_θ(n,m).

The same machinery computes 2D-MHD mean-square vector potential, buoyancy/APE variance, and QG
potential enstrophy — just pass that field as the "scalar".

Run from the repo root:
    julia --project=examples examples/passive_scalar.jl
"""

using FlowInvariantTransfer: FlowInvariantTransfer as FET
using CairoMakie: CairoMakie
include(joinpath(@__DIR__, "flows.jl"))

function run_passive_scalar_example(; N=32)
    println("--- Passive-Scalar Variance Transfer Example (3D Taylor–Green vortex) ---")
    û, θ̂, ks, L = evolve_taylor_green(; N=N, with_scalar=true)

    b = FET.LinearBinning(2π / L)
    flux = FET.calculate_scalar_flux(û, θ̂, ks; binning=b, dealiasing = FET.OrszagTwoThirds(), spectral=FET.FFTBackend())
    s2s  = FET.calculate_scalar_shell_to_shell_transfer(û, θ̂, ks; binning=b, dealiasing = FET.OrszagTwoThirds(),
        verify_antisymmetry=true, spectral=FET.FFTBackend())

    println("Peak variance flux Π_θ = ", round(maximum(flux.flux); sigdigits=4))
    println("antisymmetry of T_θ(n,m) = ",
            round(s2s.max_antisymmetry_error / sqrt(sum(abs2, s2s.transfer_matrix)); sigdigits=3))

    T = s2s.transfer_matrix
    Tlim = maximum(abs, T)
    kmax = findlast(n -> any(>(0.01Tlim), abs.(T[n, :])) || any(>(0.01Tlim), abs.(T[:, n])), 1:size(T,1))
    kmax = something(kmax, size(T, 1)); sh = 1:kmax

    fig = CairoMakie.Figure(size=(1300, 430), fontsize=14)
    CairoMakie.Label(fig[0, 1:4], "Passive-Scalar Variance Transfer — scalar stirred by a 3D Taylor–Green vortex",
        fontsize=16, font=:bold, tellwidth=false)

    ax1 = CairoMakie.Axis(fig[1, 1], title="Variance transfer T_θ(k)", xlabel="k", ylabel="T_θ(k)")
    CairoMakie.lines!(ax1, flux.k_shells, flux.transfer_spectrum, color=:seagreen, linewidth=2.5)
    CairoMakie.hlines!(ax1, [0]; color=:black, linewidth=0.8, linestyle=:dot)

    ax2 = CairoMakie.Axis(fig[1, 2], title="Variance flux Π_θ(K)  (>0 ⇒ forward)", xlabel="K", ylabel="Π_θ(K)")
    CairoMakie.band!(ax2, flux.k_shells, zero(flux.flux), flux.flux; color=(:darkorange, 0.18))
    CairoMakie.lines!(ax2, flux.k_shells, flux.flux, color=:darkorange, linewidth=2.5)
    CairoMakie.hlines!(ax2, [0]; color=:black, linewidth=0.8, linestyle=:dot)

    ax3 = CairoMakie.Axis(fig[1, 3], title="Shell-to-shell T_θ(n,m)",
        xlabel="source m", ylabel="receiver n", aspect=CairoMakie.DataAspect())
    hm = CairoMakie.heatmap!(ax3, collect(sh), collect(sh), T[sh, sh], colormap=:RdBu_9, colorrange=(-Tlim, Tlim))
    CairoMakie.lines!(ax3, [0.5, kmax+0.5], [0.5, kmax+0.5]; color=:black, linewidth=1.2, linestyle=:dash)
    CairoMakie.Colorbar(fig[1, 4], hm, width=12)

    outpath = joinpath(@__DIR__, "passive_scalar.png")
    CairoMakie.save(outpath, fig)
    println("Saved figure: $outpath")
    return (flux=flux, shell_to_shell=s2s)
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_passive_scalar_example()
end
