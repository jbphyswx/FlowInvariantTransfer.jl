"""
Mode-to-Mode Triad Transfer Example — FlowInvariantTransfer.jl

Demonstrates `calculate_mode_to_mode_transfer` on a 2D incompressible flow,
computing the exact triad transfer S(k|p|q) and the shell-reduced T(K,Q) matrix.

Run from the repo root:
    julia --project=examples examples/mode_to_mode.jl
"""

using FlowInvariantTransfer: FlowInvariantTransfer as FET
using FFTW: FFTW
using CairoMakie: CairoMakie
using Random: Random

function run_mode_to_mode_example(; N=16, seed=42)
    println("--- Mode-to-Mode Triad Transfer Example ---")
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

    # --- Compute the fully mode-resolved triad transfer S(k|p) ---
    # Note: O(N^{2D}) cost — keep N small for this example. `spectral=FFTBackend()` makes each
    # per-giver-mode nonlinear term O(N^D log N). The result carries `net_transfer` (= T(k),
    # shape ns) and `transfer` (the resolved S[k...,p...], shape (ns..., ns...)).
    result = FET.calculate_mode_to_mode_transfer(û, ks;
        invariant  = FET.KineticEnergy(),
        dealiasing = true,
        spectral   = FET.FFTBackend())

    # Verify conservation: Σ_k T(k) ≈ 0
    net_sum = sum(result.net_transfer)
    println("Σ_k T(k) = ", net_sum, " (should be ≈ 0)")

    # Shell-reduce the resolved tensor S(k|p) to the magnitude-to-magnitude matrix T(K,Q).
    S         = result.transfer                              # (N, N, N, N): S[k..., p...]
    k_mag     = FET.wavenumber_magnitude_grid(ks)
    edges     = FET.shell_edges(b, maximum(k_mag))
    shell_idx = FET.assign_shells(k_mag, edges)
    K         = FET.shell_centers(b, maximum(k_mag))
    N_sh      = length(K)
    TKQ       = zeros(N_sh, N_sh)
    for kI in CartesianIndices((N, N)), pI in CartesianIndices((N, N))
        n = shell_idx[kI]; m = shell_idx[pI]
        (n == 0 || m == 0) && continue
        TKQ[n, m] += S[kI, pI]
    end
    println("Shell-reduced T(K,Q): $(N_sh) × $(N_sh) matrix")
    println("Max |T(K,Q)|: ", maximum(abs, TKQ))

    # --- Plot ---
    fig = CairoMakie.Figure(size=(1100, 500), fontsize=14)
    CairoMakie.Label(fig[0, 1:3],
        "Mode-to-Mode Triad Transfer — 2D Random Streamfunction (N=$N)",
        fontsize=16, font=:bold)

    # Panel 1: T(K,Q) heatmap
    T_max = maximum(abs, TKQ)
    T_max = T_max == 0 ? 1.0 : T_max
    ax1 = CairoMakie.Axis(fig[1, 1:2],
        title="Shell-Reduced T(K,Q)",
        xlabel="Giver shell Q", ylabel="Receiver shell K",
        aspect=CairoMakie.DataAspect())
    hm = CairoMakie.heatmap!(ax1,
        1:N_sh, 1:N_sh, TKQ,
        colormap=:RdBu, colorrange=(-T_max, T_max))
    CairoMakie.Colorbar(fig[1, 3], hm, label="T(K,Q)")

    # Panel 2: Net transfer per shell
    net_per_shell = vec(sum(TKQ; dims=2))
    ax2 = CairoMakie.Axis(fig[2, 1:3],
        title="Net Transfer per Shell  Σ_Q T(K,Q)",
        xlabel="Shell K", ylabel="Net transfer")
    colors = ifelse.(net_per_shell .>= 0, :steelblue, :crimson)
    CairoMakie.barplot!(ax2, 1:N_sh, net_per_shell, color=colors)
    CairoMakie.hlines!(ax2, [0]; color=:black, linewidth=0.8)

    outpath = joinpath(@__DIR__, "mode_to_mode.png")
    CairoMakie.save(outpath, fig)
    println("Saved figure: $outpath")
    println("Done.")
    return result
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_mode_to_mode_example()
end
