"""
Helicity-Resolved Energy Flux Example — FlowInvariantTransfer.jl

Demonstrates `HelicalDecomposition`: splitting a 3D velocity into ±-helicity components and
computing the helicity-resolved energy fluxes Π⁺(K), Π⁻(K) via the `decomposition` argument of
`calculate_spectral_flux`. Their sum equals the total kinetic-energy flux.

Run from the repo root:
    julia --project=examples examples/helical_flux.jl
"""

using FlowInvariantTransfer: FlowInvariantTransfer as FET
using FFTW: FFTW
using CairoMakie: CairoMakie
using Random: Random

function run_helical_flux_example(; N=32, seed=42)
    println("--- Helicity-Resolved Energy Flux Example ---")
    Random.seed!(seed)

    L  = 2π
    ks = FET.wavenumber_grid((N, N, N), (L, L, L))
    kx = [ks[1][i] for i in 1:N, j in 1:N, l in 1:N]
    ky = [ks[2][j] for i in 1:N, j in 1:N, l in 1:N]
    kz = [ks[3][l] for i in 1:N, j in 1:N, l in 1:N]

    # Divergence-free velocity û = i k × Â from a random vector potential Â.
    Â = randn(ComplexF64, N, N, N, 3)
    ûx = im .* (ky .* Â[:, :, :, 3] .- kz .* Â[:, :, :, 2])
    ûy = im .* (kz .* Â[:, :, :, 1] .- kx .* Â[:, :, :, 3])
    ûz = im .* (kx .* Â[:, :, :, 2] .- ky .* Â[:, :, :, 1])
    û  = cat(ûx, ûy, ûz; dims = 4)

    b = FET.LinearBinning(2π / L)
    total = FET.calculate_spectral_flux(û, ks; binning = b, dealiasing = true,
        spectral = FET.FFTBackend())
    hel = FET.calculate_spectral_flux(û, ks; binning = b, dealiasing = true,
        spectral = FET.FFTBackend(), decomposition = FET.HelicalDecomposition())

    resid = maximum(abs, hel.positive.flux .+ hel.negative.flux .- total.flux)
    println("max|Π⁺+Π⁻ − Π| = ", resid, " (should be ≈ 0)")

    # --- Plot ---
    fig = CairoMakie.Figure(size=(800, 480), fontsize=14)
    ax = CairoMakie.Axis(fig[1, 1],
        title="Helicity-Resolved Energy Flux — 3D",
        xlabel="K", ylabel="Π(K)")
    CairoMakie.lines!(ax, total.k_shells,          total.flux,          label="Π total", linewidth=2, color=:black)
    CairoMakie.lines!(ax, hel.positive.k_shells,   hel.positive.flux,   label="Π⁺",      linewidth=2, color=:crimson)
    CairoMakie.lines!(ax, hel.negative.k_shells,   hel.negative.flux,   label="Π⁻",      linewidth=2, color=:steelblue)
    CairoMakie.hlines!(ax, [0]; color=:black, linewidth=0.8, linestyle=:dot)
    CairoMakie.axislegend(ax; position=:rb)

    outpath = joinpath(@__DIR__, "helical_flux.png")
    CairoMakie.save(outpath, fig)
    println("Saved figure: $outpath")
    println("Done.")
    return (total=total, helical=hel)
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_helical_flux_example()
end
