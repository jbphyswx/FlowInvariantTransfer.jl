"""
Generate static figure assets for FlowEnergyTransfer.jl docs and README.md.

Run from the repo root:
    julia --project=docs/generate_assets docs/generate_assets/generate_assets.jl
"""

using FlowEnergyTransfer: FlowEnergyTransfer as FET
using FFTW: FFTW
using CairoMakie: CairoMakie
using Statistics: Statistics
using Random: Random

const ASSETS_DIR = joinpath(@__DIR__, "..", "src", "assets")
CairoMakie.mkpath(ASSETS_DIR)

# ─── Figure 1: Spectral flux + shell-to-shell for Taylor-Green vortex ────────

function generate_energy_transfer_figure()
    Random.seed!(42)
    N  = 64; L = 2π
    ks = FET.wavenumber_grid((N, N), (L, L))

    # Multi-mode streamfunction → divergence-free (u, v) with real nonlinear interactions
    ψ̂ = zeros(ComplexF64, N, N)
    for kx in [2, 3, 4, 5, 6], ky in [1, 2, 3, 4]
        amp = 1.0 / (kx^2 + ky^2)
        ψ̂[kx+1, ky+1] = amp * exp(im * 2π * Random.rand())
        # Hermitian symmetry
        ψ̂[N-kx+1, N-ky+1] = conj(ψ̂[kx+1, ky+1])
    end
    û = zeros(ComplexF64, N, N, 2)
    for ix in 1:N, iy in 1:N
        û[ix, iy, 1] =  im * ks[2][iy] * ψ̂[ix, iy]
        û[ix, iy, 2] = -im * ks[1][ix] * ψ̂[ix, iy]
    end

    b = FET.LinearBinning(2 * 2π / L)   # 2-wavenumber shells for readability

    flux_result = FET.calculate_spectral_flux(û, ks;
        binning=b, dealiasing=true, backend=FET.FFTBackend())

    s2s_result = FET.calculate_shell_to_shell_transfer(û, ks;
        binning=b, dealiasing=true, verify_antisymmetry=false,
        backend=FET.FFTBackend())

    # ── Figure layout ────────────────────────────────────────────────────────
    fig = CairoMakie.Figure(size=(1400, 900), fontsize=14)
    CairoMakie.Label(fig[0, 1:3],
        "Energy Transfer — Taylor-Green Vortex (N=64, 2D)",
        fontsize=20, font=:bold)

    # Panel A: 2D spectral energy
    ax_A = CairoMakie.Axis(fig[1, 1],
        title="A. log₁₀ Energy |û|² (2D)",
        xlabel="k_x", ylabel="k_y", aspect=CairoMakie.DataAspect())
    E2d = log10.(0.5 .* (abs2.(û[:,:,1]) .+ abs2.(û[:,:,2])) .+ 1e-20)
    E2d_min, E2d_max = extrema(E2d)
    hm = CairoMakie.heatmap!(ax_A, ks[1], ks[2], E2d, colormap=:viridis,
        colorrange=(E2d_min, E2d_min == E2d_max ? E2d_max + 1 : E2d_max))
    CairoMakie.Colorbar(fig[1, 2], hm, label="log₁₀ E")

    # Panel B: Spectral transfer T(k) and flux Π(K)
    ax_B = CairoMakie.Axis(fig[1, 3],
        title="B. Spectral Flux Π(K) and Transfer T(k)",
        xlabel="Wavenumber K", ylabel="Energy transfer rate")
    CairoMakie.lines!(ax_B, flux_result.k_shells, flux_result.transfer_spectrum,
        label="T(k)", color=:steelblue, linewidth=2)
    CairoMakie.lines!(ax_B, flux_result.k_shells, flux_result.flux,
        label="Π(K)", color=:crimson, linewidth=2, linestyle=:dash)
    CairoMakie.hlines!(ax_B, [0]; color=:black, linewidth=0.8, linestyle=:dot)
    CairoMakie.axislegend(ax_B, position=:rt)

    # Panel C: Shell-to-shell matrix T(n,m)
    N_sh = size(s2s_result.transfer_matrix, 1)
    ax_C = CairoMakie.Axis(fig[2, 1:2],
        title="C. Shell-to-Shell Transfer Matrix T(n,m)",
        xlabel="Donor shell m", ylabel="Receiver shell n")
    T_plot = s2s_result.transfer_matrix
    T_max  = maximum(abs, T_plot)
    T_max  = T_max == 0 ? 1.0 : T_max
    hm2 = CairoMakie.heatmap!(ax_C,
        1:N_sh, 1:N_sh, T_plot,
        colormap=:RdBu, colorrange=(-T_max, T_max))
    CairoMakie.Colorbar(fig[2, 3], hm2, label="T(n,m)")

    # Panel D: Net transfer per shell
    ax_D = CairoMakie.Axis(fig[3, 1:3],
        title="D. Net Energy Gain per Shell  Σ_m T(n,m)",
        xlabel="Shell n", ylabel="Net transfer")
    CairoMakie.barplot!(ax_D, 1:N_sh, s2s_result.net_transfer,
        color=ifelse.(s2s_result.net_transfer .>= 0, :steelblue, :crimson))
    CairoMakie.hlines!(ax_D, [0]; color=:black, linewidth=0.8)

    outpath = joinpath(ASSETS_DIR, "energy_transfer.png")
    CairoMakie.save(outpath, fig)
    println("Saved: $outpath")
    return fig
end

println("Generating figure assets...")
generate_energy_transfer_figure()
println("Done.")
