"""
Anisotropic Directional Flux Example — FlowInvariantTransfer.jl

Demonstrates the `geometry` argument of `calculate_spectral_flux`: isotropic Π(|k|) vs the
anisotropic perpendicular Π(k⊥) and parallel Π(k∥) fluxes (Alexakis & Biferale 2018) for a 3D
field — the natural diagnostics for rotating/stratified turbulence.

Run from the repo root:
    julia --project=examples examples/anisotropic_flux.jl
"""

using FlowInvariantTransfer: FlowInvariantTransfer as FET
using FFTW: FFTW
using CairoMakie: CairoMakie
using Random: Random

function run_anisotropic_flux_example(; N=32, seed=42)
    println("--- Anisotropic Directional Flux Example ---")
    Random.seed!(seed)

    L  = 2π
    ks = FET.wavenumber_grid((N, N, N), (L, L, L))
    kx = [ks[1][i] for i in 1:N, j in 1:N, l in 1:N]
    ky = [ks[2][j] for i in 1:N, j in 1:N, l in 1:N]
    kz = [ks[3][l] for i in 1:N, j in 1:N, l in 1:N]

    Â = randn(ComplexF64, N, N, N, 3)
    ûx = im .* (ky .* Â[:, :, :, 3] .- kz .* Â[:, :, :, 2])
    ûy = im .* (kz .* Â[:, :, :, 1] .- kx .* Â[:, :, :, 3])
    ûz = im .* (kx .* Â[:, :, :, 2] .- ky .* Â[:, :, :, 1])
    û  = cat(ûx, ûy, ûz; dims = 4)

    b = FET.LinearBinning(2π / L)
    iso  = FET.calculate_spectral_flux(û, ks; binning=b, dealiasing=true,
        spectral=FET.FFTBackend(), geometry=FET.IsotropicShells())
    perp = FET.calculate_spectral_flux(û, ks; binning=b, dealiasing=true,
        spectral=FET.FFTBackend(), geometry=FET.PerpendicularShells())
    par  = FET.calculate_spectral_flux(û, ks; binning=b, dealiasing=true,
        spectral=FET.FFTBackend(), geometry=FET.ParallelShells())

    println("isotropic shells: ", length(iso.k_shells),
            "  perp: ", length(perp.k_shells), "  par: ", length(par.k_shells))

    fig = CairoMakie.Figure(size=(800, 480), fontsize=14)
    ax = CairoMakie.Axis(fig[1, 1], title="Isotropic vs anisotropic energy flux — 3D",
        xlabel="shell wavenumber", ylabel="Π")
    CairoMakie.lines!(ax, iso.k_shells,  iso.flux,  label="Π(|k|)", linewidth=2, color=:black)
    CairoMakie.lines!(ax, perp.k_shells, perp.flux, label="Π(k⊥)",  linewidth=2, color=:seagreen)
    CairoMakie.lines!(ax, par.k_shells,  par.flux,  label="Π(k∥)",  linewidth=2, color=:purple)
    CairoMakie.hlines!(ax, [0]; color=:black, linewidth=0.8, linestyle=:dot)
    CairoMakie.axislegend(ax; position=:rb)

    outpath = joinpath(@__DIR__, "anisotropic_flux.png")
    CairoMakie.save(outpath, fig)
    println("Saved figure: $outpath")
    println("Done.")
    return (isotropic=iso, perpendicular=perp, parallel=par)
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_anisotropic_flux_example()
end
