"""
Anisotropic Directional Flux Example — FlowInvariantTransfer.jl

The `geometry` argument of `calculate_spectral_flux` bins by a chosen wavenumber coordinate:
isotropic Π(|k|), perpendicular Π(k⊥)=Π(√(kx²+ky²)), or parallel Π(k∥)=Π(|kz|). For a flow with
a preferred axis these differ — the natural diagnostics for rotating/stratified turbulence
(Alexakis & Biferale 2018). We use a Taylor–Green vortex (initialised with w=0, hence anisotropic).

Run from the repo root:
    julia --project=examples examples/anisotropic_flux.jl
"""

using FlowInvariantTransfer: FlowInvariantTransfer as FET
using CairoMakie: CairoMakie
include(joinpath(@__DIR__, "flows.jl"))

function run_anisotropic_flux_example(; N=32)
    println("--- Anisotropic Directional Flux Example (3D Taylor–Green vortex) ---")
    û, ks, L = evolve_taylor_green(; N=N)

    b = FET.LinearBinning(2π / L)
    iso  = FET.calculate_spectral_flux(û, ks; binning=b, dealiasing=true, spectral=FET.FFTBackend(), geometry=FET.IsotropicShells())
    perp = FET.calculate_spectral_flux(û, ks; binning=b, dealiasing=true, spectral=FET.FFTBackend(), geometry=FET.PerpendicularShells())
    par  = FET.calculate_spectral_flux(û, ks; binning=b, dealiasing=true, spectral=FET.FFTBackend(), geometry=FET.ParallelShells())

    println("peak Π(|k|)=", round(maximum(iso.flux); sigdigits=3),
            "  Π(k⊥)=", round(maximum(perp.flux); sigdigits=3),
            "  Π(k∥)=", round(maximum(par.flux); sigdigits=3))

    fig = CairoMakie.Figure(size=(840, 480), fontsize=14)
    ax = CairoMakie.Axis(fig[1, 1], title="Isotropic vs Anisotropic Energy Flux — 3D Taylor–Green Vortex",
        xlabel="shell wavenumber", ylabel="Π")
    CairoMakie.lines!(ax, iso.k_shells,  iso.flux,  label="Π(|k|) isotropic",   linewidth=2.5, color=:black)
    CairoMakie.lines!(ax, perp.k_shells, perp.flux, label="Π(k⊥) perpendicular", linewidth=2.5, color=:seagreen)
    CairoMakie.lines!(ax, par.k_shells,  par.flux,  label="Π(k∥) parallel",      linewidth=2.5, color=:purple)
    CairoMakie.hlines!(ax, [0]; color=:black, linewidth=0.8, linestyle=:dot)
    CairoMakie.axislegend(ax; position=:rt)

    outpath = joinpath(@__DIR__, "anisotropic_flux.png")
    CairoMakie.save(outpath, fig)
    println("Saved figure: $outpath")
    return (isotropic=iso, perpendicular=perp, parallel=par)
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_anisotropic_flux_example()
end
