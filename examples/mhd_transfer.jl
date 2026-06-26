"""
Incompressible MHD Transfer Example — FlowInvariantTransfer.jl

`calculate_mhd_energy_transfer` (total / kinetic / magnetic) and
`calculate_mhd_cross_helicity_transfer` for the canonical 2D Orszag–Tang vortex evolved into a
developed state (current sheets, forward cascade). The total-energy flux Π_E(K) is positive
(forward), split into the kinetic and magnetic channels that exchange energy across scales.

Run from the repo root:
    julia --project=examples examples/mhd_transfer.jl
"""

using FlowInvariantTransfer: FlowInvariantTransfer as FET
using FFTW: FFTW
using CairoMakie: CairoMakie
include(joinpath(@__DIR__, "flows.jl"))

function run_mhd_example(; N=128)
    println("--- Incompressible MHD Transfer Example (2D Orszag–Tang vortex) ---")
    û, b̂, ks, L = evolve_orszag_tang(; N=N)

    bin = FET.LinearBinning(2π / L)
    e  = FET.calculate_mhd_energy_transfer(û, b̂, ks; binning=bin, dealiasing=true, spectral=FET.FFTBackend())
    hc = FET.calculate_mhd_cross_helicity_transfer(û, b̂, ks; binning=bin, dealiasing=true, spectral=FET.FFTBackend())
    println("peak total-energy flux Π_E = ", round(maximum(e.total.flux); sigdigits=4))

    # Electric current density j = ∂x b_y − ∂y b_x (physical space) — shows the Orszag–Tang sheets.
    kx = [ks[1][i] for i in 1:N, j in 1:N]; ky = [ks[2][j] for i in 1:N, j in 1:N]
    j_phys = real.(FFTW.ifft(im .* kx .* b̂[:, :, 2] .- im .* ky .* b̂[:, :, 1]))

    fig = CairoMakie.Figure(size=(1320, 450), fontsize=14)
    CairoMakie.Label(fig[0, 1:4], "Incompressible MHD Cross-Scale Transfer — 2D Orszag–Tang Vortex",
        fontsize=17, font=:bold, tellwidth=false)

    ax0 = CairoMakie.Axis(fig[1, 1], title="current density j (developed state)",
        xlabel="x", ylabel="y", aspect=CairoMakie.DataAspect())
    xp = range(0, 2π; length=N+1)[1:N]
    cl = maximum(abs, j_phys)
    hm = CairoMakie.heatmap!(ax0, xp, xp, j_phys, colormap=:balance, colorrange=(-cl, cl))
    CairoMakie.Colorbar(fig[1, 2], hm, width=12)

    # Total energy is the conserved invariant (Π_E = genuine forward cascade); kinetic and
    # magnetic are not separately conserved (they exchange), and total = kinetic + magnetic.
    ax1 = CairoMakie.Axis(fig[1, 3], title="Energy flux Π(K)  (>0 ⇒ forward; total = kinetic + magnetic)",
        xlabel="K", ylabel="Π(K)")
    CairoMakie.lines!(ax1, e.total.k_shells,    e.total.flux,    label="total",    linewidth=3,   color=:black)
    CairoMakie.lines!(ax1, e.kinetic.k_shells,  e.kinetic.flux,  label="kinetic",  linewidth=2.5, color=:steelblue)
    CairoMakie.lines!(ax1, e.magnetic.k_shells, e.magnetic.flux, label="magnetic", linewidth=2.5, color=:crimson)
    CairoMakie.hlines!(ax1, [0]; color=:black, linewidth=0.8, linestyle=:dot)
    CairoMakie.axislegend(ax1; position=:rc)

    ax2 = CairoMakie.Axis(fig[1, 4], title="Cross-helicity flux Π_Hc(K)", xlabel="K", ylabel="Π_Hc(K)")
    CairoMakie.lines!(ax2, hc.k_shells, hc.flux, color=:purple, linewidth=2.5)
    CairoMakie.hlines!(ax2, [0]; color=:black, linewidth=0.8, linestyle=:dot)

    outpath = joinpath(@__DIR__, "mhd_transfer.png")
    CairoMakie.save(outpath, fig)
    println("Saved figure: $outpath")
    return (energy=e, cross_helicity=hc)
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_mhd_example()
end
