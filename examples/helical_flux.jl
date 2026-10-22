"""
Helicity-Resolved Energy Flux Example — FlowInvariantTransfer.jl

`HelicalDecomposition` splits a 3D velocity into its ±-helicity components; passing it as the
`decomposition` argument of `calculate_spectral_flux` gives the helicity-resolved energy fluxes
Π⁺(K), Π⁻(K). Their sum reproduces the total kinetic-energy flux exactly — the dashed curve
(Π⁺+Π⁻) lies on top of the total. For the (mirror-symmetric, non-helical) Taylor–Green vortex the
two helical channels carry comparable flux, as expected.

Run from the repo root:
    julia --project=examples examples/helical_flux.jl
"""

using FlowInvariantTransfer: FlowInvariantTransfer as FIT
using CairoMakie: CairoMakie
include(joinpath(@__DIR__, "flows.jl"))

function run_helical_flux_example(; N=32)
    println("--- Helicity-Resolved Energy Flux Example (3D Taylor–Green vortex) ---")
    û, ks, L = evolve_taylor_green(; N=N)

    b = FIT.Types.LinearBinning(2π / L)
    total = FIT.SpectralFlux.calculate_spectral_flux(û, ks; binning=b, dealiasing = FIT.Types.OrszagTwoThirds(), transform=FIT.SpectralBackends.FFTSpectralBackend())
    hel   = FIT.SpectralFlux.calculate_spectral_flux(û, ks; binning=b, dealiasing = FIT.Types.OrszagTwoThirds(), transform=FIT.SpectralBackends.FFTSpectralBackend(),
        decomposition=FIT.Types.HelicalDecomposition())

    summed = hel.positive.flux .+ hel.negative.flux
    println("max|Π⁺+Π⁻ − Π| / max|Π| = ",
            round(maximum(abs, summed .- total.flux) / maximum(abs, total.flux); sigdigits=3))

    fig = CairoMakie.Figure(size=(820, 480), fontsize=14)
    ax = CairoMakie.Axis(fig[1, 1], title="Helicity-Resolved Energy Flux — 3D Taylor–Green Vortex",
        xlabel="wavenumber K", ylabel="Π(K)")
    K = total.shell_edges[2:end]           # Π[n] is the flux through the upper edge of shell n
    CairoMakie.lines!(ax, K, total.flux,        label="Π (total)",  linewidth=3,   color=:black)
    CairoMakie.lines!(ax, K, hel.negative.flux, label="Π⁻",         linewidth=2.5, color=:steelblue)
    CairoMakie.scatterlines!(ax, K, hel.positive.flux, label="Π⁺ (≈Π⁻: non-helical flow)",
        linewidth=0, color=:crimson, marker=:circle, markersize=7)
    CairoMakie.lines!(ax, K, summed,            label="Π⁺+Π⁻ = Π",  linewidth=2,   color=:gold, linestyle=:dash)
    CairoMakie.hlines!(ax, [0]; color=:black, linewidth=0.8, linestyle=:dot)
    CairoMakie.axislegend(ax; position=:rt)

    outpath = joinpath(@__DIR__, "helical_flux.png")
    CairoMakie.save(outpath, fig)
    println("Saved figure: $outpath")
    return (total=total, helical=hel)
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_helical_flux_example()
end
