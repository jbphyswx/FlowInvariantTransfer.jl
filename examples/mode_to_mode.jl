"""
Mode-to-Mode Triad Transfer Example — FlowInvariantTransfer.jl

`calculate_mode_to_mode_transfer` returns the fully mode-resolved triad transfer S(k|p) — the
finest object in the hierarchy. Summing it over wavenumber shells must reproduce the
shell-to-shell matrix T(K,Q) computed directly. This example shows the two side by side
(they are identical) for a developed 2D turbulence field. The resolved tensor is O(N^{2D}), so we
use a small grid.

Run from the repo root:
    julia --project=examples examples/mode_to_mode.jl
"""

using FlowInvariantTransfer: FlowInvariantTransfer as FET
using CairoMakie: CairoMakie
include(joinpath(@__DIR__, "flows.jl"))

function run_mode_to_mode_example(; N=24)
    println("--- Mode-to-Mode Triad Transfer Example (2D turbulence) ---")
    û, ks, L = evolve_2d_turbulence(; N=N)
    b = FET.LinearBinning(2π / L)

    # Fully-resolved S(k|p): O(N^{2D}); FFTBackend keeps each per-giver term O(N^D log N).
    m2m = FET.calculate_mode_to_mode_transfer(û, ks; dealiasing=true, spectral=FET.FFTBackend())
    S = m2m.transfer

    # Reduce S(k|p) over shells → T(K,Q).
    kmag = FET.wavenumber_magnitude_grid(ks)
    edges = FET.shell_edges(b, maximum(kmag)); sidx = FET.assign_shells(kmag, edges)
    K = FET.shell_centers(b, maximum(kmag)); Nsh = length(K)
    TKQ = zeros(Nsh, Nsh)
    for kI in CartesianIndices((N, N)), pI in CartesianIndices((N, N))
        n = sidx[kI]; m = sidx[pI]
        (n == 0 || m == 0) && continue
        TKQ[n, m] += S[kI, pI]
    end

    # Direct shell-to-shell T(n,m) for comparison.
    s2s = FET.calculate_shell_to_shell_transfer(û, ks; binning=b, dealiasing=true,
        verify_antisymmetry=false, spectral=FET.FFTBackend())
    Tdir = s2s.transfer_matrix
    println("max|reduce(S) − T_shell| / ‖T‖ = ",
            round(maximum(abs, TKQ .- Tdir) / sqrt(sum(abs2, Tdir)); sigdigits=3))

    # Trim to active shells.
    Tlim = maximum(abs, TKQ)
    kmax = findlast(n -> any(>(0.01Tlim), abs.(TKQ[n, :])) || any(>(0.01Tlim), abs.(TKQ[:, n])), 1:Nsh)
    kmax = something(kmax, Nsh); sh = 1:kmax

    fig = CairoMakie.Figure(size=(1050, 470), fontsize=14)
    CairoMakie.Label(fig[0, 1:2], "Mode-to-Mode S(k|p) reduces to Shell-to-Shell T(K,Q) — 2D turbulence",
        fontsize=16, font=:bold, tellwidth=false)
    for (col, (data, ttl)) in enumerate(((TKQ, "Σ over shells of resolved S(k|p)"),
                                         (Tdir, "direct shell-to-shell T(n,m)")))
        ax = CairoMakie.Axis(fig[1, col], title=ttl, xlabel="source shell m", ylabel="receiver shell n",
            aspect=CairoMakie.DataAspect())
        CairoMakie.heatmap!(ax, collect(sh), collect(sh), data[sh, sh], colormap=:RdBu_9, colorrange=(-Tlim, Tlim))
        CairoMakie.lines!(ax, [0.5, kmax+0.5], [0.5, kmax+0.5]; color=:black, linewidth=1.2, linestyle=:dash)
    end

    outpath = joinpath(@__DIR__, "mode_to_mode.png")
    CairoMakie.save(outpath, fig)
    println("Saved figure: $outpath")
    return m2m
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_mode_to_mode_example()
end
