"""
Spectral Flux Example — FlowInvariantTransfer.jl

Computes the spectral energy transfer T(k) and cumulative flux Π(K) for a 3D Taylor–Green vortex
evolved to a developed state (forward energy cascade). A positive Π(K) plateau across the
inertial range is the signature of a forward cascade — energy leaving the large scales and
flowing toward small scales.

Run from the repo root:
    julia --project=examples examples/spectral_flux.jl
"""

using FlowInvariantTransfer: FlowInvariantTransfer as FET
using CairoMakie: CairoMakie
include(joinpath(@__DIR__, "flows.jl"))

function run_spectral_flux_example(; N=32)
    println("--- Spectral Flux Example (3D Taylor–Green vortex) ---")
    û, ks, L = evolve_taylor_green(; N=N)

    b = FET.LinearBinning(2π / L)
    result = FET.calculate_spectral_flux(û, ks; binning=b, dealiasing = FET.OrszagTwoThirds(), spectral=FET.FFTBackend())

    imax = argmax(result.flux)
    println("Shells: ", length(result.k_shells))
    println("Peak forward flux Π = ", round(maximum(result.flux); sigdigits=4),
            " at k ≈ ", round(result.k_shells[imax]; digits=2))

    fig = CairoMakie.Figure(size=(960, 430), fontsize=14)
    CairoMakie.Label(fig[0, 1:2], "Spectral Energy Transfer — 3D Taylor–Green Vortex",
        fontsize=17, font=:bold)

    ax1 = CairoMakie.Axis(fig[1, 1], title="Transfer spectrum T(k)",
        xlabel="wavenumber k", ylabel="T(k)")
    CairoMakie.lines!(ax1, result.k_shells, result.transfer_spectrum, color=:steelblue, linewidth=2.5)
    CairoMakie.hlines!(ax1, [0]; color=:black, linewidth=0.8, linestyle=:dot)

    ax2 = CairoMakie.Axis(fig[1, 2], title="Cumulative flux Π(K)  (>0 ⇒ forward cascade)",
        xlabel="wavenumber K", ylabel="Π(K)")
    CairoMakie.band!(ax2, result.k_shells, zero(result.flux), result.flux;
        color=(:crimson, 0.18))
    CairoMakie.lines!(ax2, result.k_shells, result.flux, color=:crimson, linewidth=2.5)
    CairoMakie.hlines!(ax2, [0]; color=:black, linewidth=0.8, linestyle=:dot)

    outpath = joinpath(@__DIR__, "spectral_flux.png")
    CairoMakie.save(outpath, fig)
    println("Saved figure: $outpath")
    return result
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_spectral_flux_example()
end
