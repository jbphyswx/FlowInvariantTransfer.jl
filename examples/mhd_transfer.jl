"""
Incompressible MHD Transfer Example — FlowInvariantTransfer.jl

Demonstrates `calculate_mhd_energy_transfer` (total/kinetic/magnetic) and
`calculate_mhd_cross_helicity_transfer` for a 2D incompressible MHD field built from a velocity
streamfunction ψ and a magnetic flux function a (b = ∇×(a ẑ)). Total energy E = ½⟨|u|²+|b|²⟩ and
cross-helicity H_c = ⟨u·b⟩ are conserved by the nonlinear terms (Σ_k T ≈ 0).

Run from the repo root:
    julia --project=examples examples/mhd_transfer.jl
"""

using FlowInvariantTransfer: FlowInvariantTransfer as FET
using FFTW: FFTW
using CairoMakie: CairoMakie
using Random: Random

function run_mhd_example(; N=64, seed=42)
    println("--- Incompressible MHD Transfer Example ---")
    Random.seed!(seed)

    L  = 2π
    ks = FET.wavenumber_grid((N, N), (L, L))
    kx = [ks[1][i] for i in 1:N, j in 1:N]
    ky = [ks[2][j] for i in 1:N, j in 1:N]

    ψh = FFTW.fft(randn(N, N)) ./ N^2       # velocity streamfunction
    ah = FFTW.fft(randn(N, N)) ./ N^2       # magnetic flux function
    û  = cat(im .* ky .* ψh, -im .* kx .* ψh; dims = 3)
    b̂  = cat(im .* ky .* ah, -im .* kx .* ah; dims = 3)

    bin = FET.LinearBinning(2π / L)
    e  = FET.calculate_mhd_energy_transfer(û, b̂, ks; binning = bin, dealiasing = true,
        spectral = FET.FFTBackend())
    hc = FET.calculate_mhd_cross_helicity_transfer(û, b̂, ks; binning = bin, dealiasing = true,
        spectral = FET.FFTBackend())

    println("Σ_k T_E(k)   = ", sum(e.total.transfer_spectrum), " (total energy conserved, ≈ 0)")
    println("Σ_k T_Hc(k)  = ", sum(hc.transfer_spectrum), " (cross-helicity conserved, ≈ 0)")

    # --- Plot ---
    fig = CairoMakie.Figure(size=(1100, 450), fontsize=14)
    CairoMakie.Label(fig[0, 1:2], "Incompressible MHD Cross-Scale Transfer — 2D",
        fontsize=16, font=:bold)

    ax1 = CairoMakie.Axis(fig[1, 1], title="Energy flux Π(K)", xlabel="K", ylabel="Π(K)")
    CairoMakie.lines!(ax1, e.total.k_shells,    e.total.flux,    label="total",    linewidth=2)
    CairoMakie.lines!(ax1, e.kinetic.k_shells,  e.kinetic.flux,  label="kinetic",  linewidth=2)
    CairoMakie.lines!(ax1, e.magnetic.k_shells, e.magnetic.flux, label="magnetic", linewidth=2)
    CairoMakie.hlines!(ax1, [0]; color=:black, linewidth=0.8, linestyle=:dot)
    CairoMakie.axislegend(ax1; position=:rb)

    ax2 = CairoMakie.Axis(fig[1, 2], title="Cross-helicity flux Π_Hc(K)", xlabel="K", ylabel="Π_Hc(K)")
    CairoMakie.lines!(ax2, hc.k_shells, hc.flux, color=:purple, linewidth=2)
    CairoMakie.hlines!(ax2, [0]; color=:black, linewidth=0.8, linestyle=:dot)

    outpath = joinpath(@__DIR__, "mhd_transfer.png")
    CairoMakie.save(outpath, fig)
    println("Saved figure: $outpath")
    println("Done.")
    return (energy=e, cross_helicity=hc)
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_mhd_example()
end
