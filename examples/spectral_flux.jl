"""
Spectral Flux Example — FlowEnergyTransfer.jl

Demonstrates `calculate_spectral_flux` on a 2D incompressible flow field
constructed from a random streamfunction.

Run from the repo root:
    julia --project=examples examples/spectral_flux.jl
"""

using FlowEnergyTransfer: FlowEnergyTransfer as FET
using FFTW: FFTW
using CairoMakie: CairoMakie
using Random: Random

function run_spectral_flux_example(; N=64, seed=42)
    println("--- Spectral Flux Example ---")
    Random.seed!(seed)

    L  = 2π
    ks = FET.wavenumber_grid((N, N), (L, L))

    # Build a divergence-free (incompressible) velocity field from a streamfunction ψ̂.
    # Populate many modes with Hermitian symmetry so IFFT is real.
    # u =  ∂ψ/∂y  →  û_x =  i k_y ψ̂
    # v = -∂ψ/∂x  →  û_y = -i k_x ψ̂
    ψ̂ = zeros(ComplexF64, N, N)
    for kx in 1:12, ky in 1:12
        amp = 1.0 / (kx^2 + ky^2)
        ψ̂[kx+1, ky+1] = amp * exp(im * 2π * Random.rand())
        # Hermitian: ψ̂(-k) = conj(ψ̂(k))
        ix_m = kx == 0 ? 1 : N - kx + 2
        iy_m = ky == 0 ? 1 : N - ky + 2
        if ix_m <= N && iy_m <= N
            ψ̂[ix_m, iy_m] = conj(ψ̂[kx+1, ky+1])
        end
    end

    û = zeros(ComplexF64, N, N, 2)
    for ix in 1:N, iy in 1:N
        û[ix, iy, 1] =  im * ks[2][iy] * ψ̂[ix, iy]
        û[ix, iy, 2] = -im * ks[1][ix] * ψ̂[ix, iy]
    end

    # --- Compute spectral flux via FFTBackend ---
    b = FET.LinearBinning(2π / L)

    result = FET.calculate_spectral_flux(û, ks;
        binning    = b,
        dealiasing = true,
        backend    = FET.FFTBackend())

    println("Shells: ", length(result.k_shells))
    println("Max |T(k)|: ", maximum(abs, result.transfer_spectrum))
    println("Max |Π(K)|: ", maximum(abs, result.flux))

    # --- Plot ---
    fig = CairoMakie.Figure(size=(900, 500), fontsize=14)
    CairoMakie.Label(fig[0, 1:2],
        "Spectral Flux Π(K) and Transfer T(k) — 2D Streamfunction Field",
        fontsize=16, font=:bold)

    ax1 = CairoMakie.Axis(fig[1, 1],
        title="Transfer Spectrum T(k)",
        xlabel="Wavenumber k", ylabel="T(k)")
    CairoMakie.lines!(ax1, result.k_shells, result.transfer_spectrum,
        color=:steelblue, linewidth=2)
    CairoMakie.hlines!(ax1, [0]; color=:black, linewidth=0.8, linestyle=:dot)

    ax2 = CairoMakie.Axis(fig[1, 2],
        title="Cumulative Flux Π(K)",
        xlabel="Wavenumber K", ylabel="Π(K)")
    CairoMakie.lines!(ax2, result.k_shells, result.flux,
        color=:crimson, linewidth=2)
    CairoMakie.hlines!(ax2, [0]; color=:black, linewidth=0.8, linestyle=:dot)

    outpath = joinpath(@__DIR__, "spectral_flux.png")
    CairoMakie.save(outpath, fig)
    println("Saved figure: $outpath")
    println("Done.")
    return result
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_spectral_flux_example()
end
