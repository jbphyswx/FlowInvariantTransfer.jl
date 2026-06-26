"""
Passive-Scalar Variance Transfer Example — FlowInvariantTransfer.jl

Demonstrates `calculate_scalar_flux` and `calculate_scalar_shell_to_shell_transfer`: the
cross-scale transfer of passive-scalar variance ½⟨θ²⟩ for a scalar θ advected by an
incompressible 2D velocity. Scalar variance is conserved (Σ_k T_θ ≈ 0) and cascades forward.

The same machinery computes 2D-MHD mean-square vector potential, buoyancy/APE variance, and QG
potential enstrophy — just pass that field as the "scalar".

Run from the repo root:
    julia --project=examples examples/passive_scalar.jl
"""

using FlowInvariantTransfer: FlowInvariantTransfer as FET
using FFTW: FFTW
using CairoMakie: CairoMakie
using Random: Random

function run_passive_scalar_example(; N=64, seed=42)
    println("--- Passive-Scalar Variance Transfer Example ---")
    Random.seed!(seed)

    L  = 2π
    ks = FET.wavenumber_grid((N, N), (L, L))
    kx = [ks[1][i] for i in 1:N, j in 1:N]
    ky = [ks[2][j] for i in 1:N, j in 1:N]

    # Divergence-free velocity from a streamfunction, and a random scalar field.
    ψh = FFTW.fft(randn(N, N)) ./ N^2
    û  = cat(im .* ky .* ψh, -im .* kx .* ψh; dims = 3)
    θ  = FFTW.fft(randn(N, N)) ./ N^2

    b = FET.LinearBinning(2π / L)

    flux = FET.calculate_scalar_flux(û, θ, ks; binning = b, dealiasing = true,
        spectral = FET.FFTBackend())
    s2s  = FET.calculate_scalar_shell_to_shell_transfer(û, θ, ks; binning = b, dealiasing = true,
        verify_antisymmetry = true, spectral = FET.FFTBackend())

    println("Σ_k T_θ(k) = ", sum(flux.transfer_spectrum), " (conserved, ≈ 0)")
    println("max|T_θ(n,m)+T_θ(m,n)| = ", s2s.max_antisymmetry_error, " (antisymmetric, ≈ 0)")

    # --- Plot ---
    fig = CairoMakie.Figure(size=(1100, 450), fontsize=14)
    CairoMakie.Label(fig[0, 1:3], "Passive-Scalar Variance Transfer — 2D",
        fontsize=16, font=:bold)

    ax1 = CairoMakie.Axis(fig[1, 1], title="Variance transfer T_θ(k)",
        xlabel="k", ylabel="T_θ(k)")
    CairoMakie.lines!(ax1, flux.k_shells, flux.transfer_spectrum, color=:seagreen, linewidth=2)
    CairoMakie.hlines!(ax1, [0]; color=:black, linewidth=0.8, linestyle=:dot)

    ax2 = CairoMakie.Axis(fig[1, 2], title="Variance flux Π_θ(K)",
        xlabel="K", ylabel="Π_θ(K)")
    CairoMakie.lines!(ax2, flux.k_shells, flux.flux, color=:darkorange, linewidth=2)
    CairoMakie.hlines!(ax2, [0]; color=:black, linewidth=0.8, linestyle=:dot)

    Tmax = maximum(abs, s2s.transfer_matrix); Tmax = Tmax == 0 ? 1.0 : Tmax
    Nsh = length(s2s.shell_centers)
    ax3 = CairoMakie.Axis(fig[1, 3], title="Shell-to-shell T_θ(n,m)",
        xlabel="giver m", ylabel="receiver n", aspect=CairoMakie.DataAspect())
    hm = CairoMakie.heatmap!(ax3, 1:Nsh, 1:Nsh, s2s.transfer_matrix,
        colormap=:RdBu, colorrange=(-Tmax, Tmax))
    CairoMakie.Colorbar(fig[1, 4], hm)

    outpath = joinpath(@__DIR__, "passive_scalar.png")
    CairoMakie.save(outpath, fig)
    println("Saved figure: $outpath")
    println("Done.")
    return (flux=flux, shell_to_shell=s2s)
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_passive_scalar_example()
end
