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

using FlowInvariantTransfer: FlowInvariantTransfer as FET
using CairoMakie: CairoMakie
include(joinpath(@__DIR__, "flows.jl"))

function run_helical_flux_example(; N=32)
    println("--- Helicity-Resolved Energy Flux Example (3D Taylor–Green vortex) ---")
    û, ks, L = evolve_taylor_green(; N=N)

    b = FET.LinearBinning(2π / L)
    total = FET.calculate_spectral_flux(û, ks; binning=b, dealiasing=true, spectral=FET.FFTBackend())
    hel   = FET.calculate_spectral_flux(û, ks; binning=b, dealiasing=true, spectral=FET.FFTBackend(),
        decomposition=FET.HelicalDecomposition())

    summed = hel.positive.flux .+ hel.negative.flux
    println("max|Π⁺+Π⁻ − Π| / max|Π| = ",
            round(maximum(abs, summed .- total.flux) / maximum(abs, total.flux); sigdigits=3))

    fig = CairoMakie.Figure(size=(820, 480), fontsize=14)
    ax = CairoMakie.Axis(fig[1, 1], title="Helicity-Resolved Energy Flux — 3D Taylor–Green Vortex",
        xlabel="wavenumber K", ylabel="Π(K)")
    CairoMakie.lines!(ax, total.k_shells,        total.flux,        label="Π (total)",  linewidth=3,   color=:black)
    CairoMakie.lines!(ax, hel.negative.k_shells, hel.negative.flux, label="Π⁻",         linewidth=2.5, color=:steelblue)
    CairoMakie.scatterlines!(ax, hel.positive.k_shells, hel.positive.flux, label="Π⁺ (≈Π⁻: non-helical flow)",
        linewidth=0, color=:crimson, marker=:circle, markersize=7)
    CairoMakie.lines!(ax, total.k_shells,        summed,            label="Π⁺+Π⁻ = Π",  linewidth=2,   color=:gold, linestyle=:dash)
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
