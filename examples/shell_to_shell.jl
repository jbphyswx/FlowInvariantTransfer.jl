"""
Shell-to-Shell Transfer Example — FlowInvariantTransfer.jl

Computes the directed shell-to-shell energy transfer T(n,m) for a developed 3D Taylor–Green
vortex. The matrix is antisymmetric (T(n,m) = −T(m,n)) and dominated by near-diagonal
local transfer; the net transfer Σₘ T(n,m) shows the large scales losing energy and the
small scales gaining it — a forward cascade.

Run from the repo root:
    julia --project=examples examples/shell_to_shell.jl
"""

using FlowInvariantTransfer: FlowInvariantTransfer as FET
using CairoMakie: CairoMakie
include(joinpath(@__DIR__, "flows.jl"))

function run_shell_to_shell_example(; N=32)
    println("--- Shell-to-Shell Transfer Example (3D Taylor–Green vortex) ---")
    û, ks, L = evolve_taylor_green(; N=N)

    b = FET.LinearBinning(2π / L)
    result = FET.calculate_shell_to_shell_transfer(û, ks; binning=b, dealiasing = FET.OrszagTwoThirds(),
        verify_antisymmetry=true, spectral=FET.FFTBackend())

    Tn = sqrt(sum(abs2, result.transfer_matrix))
    println("Shells: ", length(result.shell_centers))
    println("antisymmetry max|T(n,m)+T(m,n)| / ‖T‖ = ",
            round(result.max_antisymmetry_error / Tn; sigdigits=3))

    # Trim to the active shells (those carrying ≳1% of the peak transfer) for a readable plot.
    T = result.transfer_matrix
    Tlim = maximum(abs, T)
    kmax = findlast(n -> any(>(0.01Tlim), abs.(T[n, :])) || any(>(0.01Tlim), abs.(T[:, n])),
                    1:size(T, 1))
    kmax = something(kmax, size(T, 1))
    sh = 1:kmax

    fig = CairoMakie.Figure(size=(1050, 460), fontsize=14)
    CairoMakie.Label(fig[0, 1:3], "Shell-to-Shell Energy Transfer — 3D Taylor–Green Vortex",
        fontsize=17, font=:bold, tellwidth=false)

    ax1 = CairoMakie.Axis(fig[1, 1], title="T(n,m)  (blue: gain, red: loss)",
        xlabel="source shell m", ylabel="receiver shell n", aspect=CairoMakie.DataAspect())
    hm = CairoMakie.heatmap!(ax1, collect(sh), collect(sh), T[sh, sh],
        colormap=:RdBu_9, colorrange=(-Tlim, Tlim))
    CairoMakie.lines!(ax1, [0.5, kmax+0.5], [0.5, kmax+0.5]; color=:black, linewidth=1.5, linestyle=:dash)
    CairoMakie.Colorbar(fig[1, 2], hm, label="T(n,m)", width=12)

    netsub = result.net_transfer[sh]
    ax2 = CairoMakie.Axis(fig[1, 3], title="Net transfer Σₘ T(n,m)",
        xlabel="shell n", ylabel="net energy gain")
    cols = [v >= 0 ? CairoMakie.RGBf(0.27,0.51,0.71) : CairoMakie.RGBf(0.84,0.15,0.16) for v in netsub]
    CairoMakie.barplot!(ax2, collect(sh), netsub, color=cols)
    CairoMakie.hlines!(ax2, [0]; color=:black, linewidth=1.0)

    outpath = joinpath(@__DIR__, "shell_to_shell.png")
    CairoMakie.save(outpath, fig)
    println("Saved figure: $outpath")
    return result
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_shell_to_shell_example()
end
