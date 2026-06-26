"""
Shell-to-Shell Transfer Example — FlowInvariantTransfer.jl

Demonstrates `calculate_shell_to_shell_transfer` on a 2D incompressible flow,
verifying antisymmetry of T(n,m) and showing the transfer matrix heat map.
Also demonstrates the zero-alloc `!`-variant for use in a hot loop.

Run from the repo root:
    julia --project=examples examples/shell_to_shell.jl
"""

using FlowInvariantTransfer: FlowInvariantTransfer as FET
using FFTW: FFTW
using CairoMakie: CairoMakie
using Random: Random

function run_shell_to_shell_example(; N=32, seed=7)
    println("--- Shell-to-Shell Transfer Example ---")
    Random.seed!(seed)

    L  = 2π
    ks = FET.wavenumber_grid((N, N), (L, L))

    # Build a divergence-free velocity field from a random streamfunction
    ψ̂ = zeros(ComplexF64, N, N)
    for ix in 1:N, iy in 1:N
        ψ̂[ix, iy] = randn() + im * randn()
    end
    # Enforce Hermitian symmetry so IFFT is real
    for ix in 1:N, iy in 1:N
        cix = ix == 1 ? 1 : N - ix + 2
        ciy = iy == 1 ? 1 : N - iy + 2
        if (cix, ciy) > (ix, iy)
            ψ̂[cix, ciy] = conj(ψ̂[ix, iy])
        end
    end

    û = zeros(ComplexF64, N, N, 2)
    for ix in 1:N, iy in 1:N
        û[ix, iy, 1] =  im * ks[2][iy] * ψ̂[ix, iy]
        û[ix, iy, 2] = -im * ks[1][ix] * ψ̂[ix, iy]
    end

    b = FET.LinearBinning(2π / L)

    # --- Allocating convenience API ---
    result = FET.calculate_shell_to_shell_transfer(û, ks;
        binning             = b,
        dealiasing          = true,
        verify_antisymmetry = true,
        spectral            = FET.FFTBackend())

    N_sh = length(result.shell_centers)
    T_norm = sqrt(sum(abs2, result.transfer_matrix))
    println("Shells: $N_sh")
    println("‖T‖₂ = $T_norm")
    println("max|T(n,m)+T(m,n)| / ‖T‖₂ = ",
        result.max_antisymmetry_error / (T_norm + eps()))

    # --- Zero-alloc !-variant (for time-stepping use) ---
    ws = FET.ShellToShellWorkspace(û, ks, b)
    FET.calculate_shell_to_shell_transfer!(result, ws, û, ks;
        dealiasing=true, verify_antisymmetry=false)
    println("Re-used workspace successfully.")

    # --- Plot ---
    fig = CairoMakie.Figure(size=(1100, 800), fontsize=14)
    CairoMakie.Label(fig[0, 1:3],
        "Shell-to-Shell Transfer T(n,m) — 2D Random Streamfunction (N=$N)",
        fontsize=16, font=:bold)

    # T(n,m) heatmap
    T_max = maximum(abs, result.transfer_matrix)
    ax1 = CairoMakie.Axis(fig[1, 1:2],
        title="T(n,m) — donor shell m → receiver shell n",
        xlabel="Donor shell m", ylabel="Receiver shell n",
        aspect=CairoMakie.DataAspect())
    hm = CairoMakie.heatmap!(ax1,
        1:N_sh, 1:N_sh, result.transfer_matrix,
        colormap=:RdBu, colorrange=(-T_max, T_max))
    CairoMakie.Colorbar(fig[1, 3], hm, label="T(n,m)")

    # Net transfer per shell
    ax2 = CairoMakie.Axis(fig[2, 1:3],
        title="Net energy gain per shell  Σ_m T(n,m)",
        xlabel="Shell n", ylabel="Net transfer rate")
    colors = ifelse.(result.net_transfer .>= 0, :steelblue, :crimson)
    CairoMakie.barplot!(ax2, 1:N_sh, result.net_transfer, color=colors)
    CairoMakie.hlines!(ax2, [0]; color=:black, linewidth=0.8)

    outpath = joinpath(@__DIR__, "shell_to_shell.png")
    CairoMakie.save(outpath, fig)
    println("Saved figure: $outpath")
    println("Done.")
    return result
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_shell_to_shell_example()
end
