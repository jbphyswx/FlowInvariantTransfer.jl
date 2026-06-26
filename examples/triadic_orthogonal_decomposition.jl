"""
Triadic Orthogonal Decomposition Example — FlowInvariantTransfer.jl

Demonstrates `triadic_orthogonal_decomposition` on a simulated multi-modal signal
with a known triadic interaction (resonance condition f_k + f_l = f_n).

Run from the repo root:
    julia --project=examples examples/triadic_orthogonal_decomposition.jl
"""

using FlowInvariantTransfer: FlowInvariantTransfer as FET
using FFTW: FFTW
using CairoMakie: CairoMakie
using Random: Random
using Statistics: Statistics

function run_tod_example(; nt=512, nx=64, dt=0.02, seed=42)
    println("--- Triadic Orthogonal Decomposition Example ---")
    Random.seed!(seed)

    t = collect(0:nt-1) .* dt

    # Define three resonant frequencies satisfying: f_k + f_l = f_n
    # Let f_k = 2.0 Hz, f_l = 3.0 Hz, then f_n = 5.0 Hz
    fk_res, fl_res, fn_res = 2.0, 3.0, 5.0

    # Build simulated spatial patterns for the convective/recipient modes
    x = range(0, 2π; length=nx)
    mode_k = sin.(2x)
    mode_l = cos.(3x)
    mode_n = sin.(5x)

    # Construct the data array of size (nt, nvar, nx)
    # Here nvar = 1
    X = zeros(nt, 1, nx)
    for it in 1:nt
        # A simple triadic coupling model:
        # Each frequency has a corresponding spatial mode modulated in time
        X[it, 1, :] .= (
            1.2 * sin(2π * fk_res * t[it]) .* mode_k .+
            0.8 * cos(2π * fl_res * t[it]) .* mode_l .+
            1.5 * sin(2π * fn_res * t[it]) .* mode_n
        )
    end

    # --- Compute TOD ---
    # We specify:
    # - nfft = 128 (window block length)
    # - noverlap = 64 (50% overlap)
    # - nmode = 2 (retain top 2 modes per triad)
    # - spectral = FFTBackend() (use FFTW for temporal DFTs)
    method = FET.TriadicOrthogonalDecompositionMethod(nfft=128, noverlap=64, nmode=2)

    result = FET.calculate_energy_transfer(method, X;
        dt=dt,
        isreal_data=true,
        spectral=FET.FFTBackend())

    # Find the indices corresponding to our resonant frequencies
    freqs = result.frequencies
    fl_idx = argmin(abs.(freqs .- fl_res))
    fn_idx = argmin(abs.(freqs .- fn_res))

    println("Target f_l: ", freqs[fl_idx], " Hz (idx: ", fl_idx, ")")
    println("Target f_n: ", freqs[fn_idx], " Hz (idx: ", fn_idx, ")")

    # Singular values for the strongest mode at the target triad
    coupling_strength = result.mode_bispectrum[fl_idx, fn_idx, 1]
    println("Mode 1 coupling strength at target triad: ", coupling_strength)

    # Extract the convective and recipient modes for this triad
    modes_pair = result.modes[(fl_idx, fn_idx)]
    # convective mode: modes_pair.convective[:, 1]
    # recipient mode: modes_pair.recipient[:, 1]
    phi = real.(modes_pair.convective[:, 1])
    psi = real.(modes_pair.recipient[:, 1])

    # --- Plot the results ---
    fig = CairoMakie.Figure(size=(1000, 750), fontsize=14)
    CairoMakie.Label(fig[0, 1:2],
        "Triadic Orthogonal Decomposition — Simulated Resonant Triad",
        fontsize=18, font=:bold)

    # 1. Mode Bispectrum Heatmap (mode 1)
    ax_bispec = CairoMakie.Axis(fig[1, 1],
        title="Mode Bispectrum λ(f_l, f_n) — Mode 1",
        xlabel="f_l (Hz)", ylabel="f_n (Hz)")
    
    # We only plot positive frequencies
    pos_idxs = findall(>=(0), freqs)
    f_pos = freqs[pos_idxs]
    bispec_pos = result.mode_bispectrum[pos_idxs, pos_idxs, 1]
    
    # Replace NaNs with 0 for plotting
    bispec_plot = copy(bispec_pos)
    bispec_plot[isnan.(bispec_plot)] .= 0.0

    hm = CairoMakie.heatmap!(ax_bispec, f_pos, f_pos, bispec_plot, colormap=:viridis)
    CairoMakie.Colorbar(fig[1, 2], hm, label="Singular Value (coupling)")

    # 2. Spatial Modes at the resonant triad
    ax_modes = CairoMakie.Axis(fig[2, 1:2],
        title="TOD Spatial Modes at Triad ($fl_res, $fn_res) Hz",
        xlabel="x", ylabel="Amplitude")

    CairoMakie.lines!(ax_modes, x, phi ./ maximum(abs, phi),
        label="Convective Mode (normalized)", color=:steelblue, linewidth=2.5)
    CairoMakie.lines!(ax_modes, x, psi ./ maximum(abs, psi),
        label="Recipient Mode (normalized)", color=:crimson, linewidth=2.5)
    CairoMakie.axislegend(ax_modes)

    outpath = joinpath(@__DIR__, "triadic_orthogonal_decomposition.png")
    CairoMakie.save(outpath, fig)
    println("Saved figure: $outpath")
    println("Done.")
    return result
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_tod_example()
end
