"""
Triadic Orthogonal Decomposition Example — FlowInvariantTransfer.jl

Builds a signal with ONE genuine quadratic triad and shows TOD finds it.

Temporal modes live at f_k=1, f_l=2, f_n=3 Hz (so f_k+f_l=f_n), plus an UNCOUPLED control wave at
f_u=4 Hz. The f_n mode is a phase-locked sum-frequency wave (phase φ_k+φ_l — a real quadratic
interaction), and every block draws fresh random parent phases, so the daughter's biphase stays
consistent while the control and all accidental pairs have random biphase and average away. The
mode bispectrum λ(f_l, f_n) therefore lights up on the triad family {1,2,3} (marked) and stays
dark at the uncoupled control; the recovered recipient mode matches the imposed shape m_n=m_k·m_l.

Run from the repo root:
    julia --project=examples examples/triadic_orthogonal_decomposition.jl
"""

using FlowInvariantTransfer: FlowInvariantTransfer as FET
using FFTW: FFTW          # loads the FFTW extension so spectral=FFTBackend() works
using CairoMakie: CairoMakie
using Random: Random

function run_tod_example(; nfft=100, nblocks=64, nx=64, dt=0.1, seed=42)
    println("--- Triadic Orthogonal Decomposition Example ---")
    Random.seed!(seed)

    fk, fl, fn, fu = 1.0, 2.0, 3.0, 4.0          # f_k + f_l = f_n; f_u is an uncoupled control
    x  = range(0, 2π; length=nx)
    mk = cos.(x); ml = cos.(2x); mu = sin.(x)
    mn = mk .* ml                                # daughter spatial shape = product of parents

    nt = nfft * nblocks
    X  = zeros(nt, 1, nx)
    for blk in 0:nblocks-1
        φk, φl, φu = 2π .* rand(3)               # fresh random phases each block
        for τ in 0:nfft-1
            tt = τ * dt
            an = cos(2π*fn*tt + φk + φl)         # daughter: sum-frequency wave, phase LOCKED to parents
            X[blk*nfft + τ + 1, 1, :] .= cos(2π*fk*tt + φk).*mk .+ cos(2π*fl*tt + φl).*ml .+
                                          an.*mn .+ cos(2π*fu*tt + φu).*mu
        end
    end

    method = FET.TriadicOrthogonalDecompositionMethod(nfft=nfft, noverlap=0, nmode=1)
    result = FET.calculate_energy_transfer(method, X; dt=dt, isreal_data=true, spectral=FET.FFTBackend())

    freqs = result.frequencies
    λ = copy(result.mode_bispectrum[:, :, 1]); λ[isnan.(λ)] .= 0.0
    pk = argmax(λ)
    println("bispectrum peak at (f_l, f_n) = (", round(freqs[pk[1]]; digits=2), ", ",
            round(freqs[pk[2]]; digits=2), ") Hz, f_k = ", round(freqs[pk[2]] - freqs[pk[1]]; digits=2),
            "  (a member of the {1,2,3} triad family)")

    li = argmin(abs.(freqs .- fl)); ni = argmin(abs.(freqs .- fn))
    psi = real.(result.modes[(li, ni)].recipient[:, 1])

    # ── plot ──────────────────────────────────────────────────────────────────
    posn = findall(>=(0), freqs)                 # f_n ≥ 0 for real data
    band = findall(ff -> -4.5 <= ff <= 4.5, freqs)
    fl_ax = freqs[band]; fn_ax = freqs[posn]
    λsub = λ[band, posn]
    fig = CairoMakie.Figure(size=(1180, 520), fontsize=14)
    CairoMakie.Label(fig[0, 1:3], "Triadic Orthogonal Decomposition — detecting a known quadratic triad",
        fontsize=17, font=:bold, tellwidth=false)

    ax1 = CairoMakie.Axis(fig[1, 1], title="Mode bispectrum λ(f_l, f_n)   (bright ⇒ triadic coupling)",
        xlabel="f_l  (Hz)", ylabel="f_n  (Hz)")
    hm = CairoMakie.heatmap!(ax1, fl_ax, fn_ax, λsub,
        colormap=CairoMakie.cgrad([:white, :gold, :orangered, :darkred]), colorrange=(0, maximum(λsub)))
    CairoMakie.Colorbar(fig[1, 2], hm, label="coupling strength")
    # the triad family f_k+f_l=f_n with {|f_k|,|f_l|,|f_n|}={1,2,3}
    triad_cells = [(2.0,3.0), (1.0,3.0), (-1.0,2.0), (3.0,2.0), (-2.0,1.0), (3.0,1.0)]
    CairoMakie.scatter!(ax1, first.(triad_cells), last.(triad_cells); marker=:rect, markersize=15,
        color=:transparent, strokecolor=:dodgerblue, strokewidth=2.5)
    CairoMakie.scatter!(ax1, [4.0], [4.0]; marker=:xcross, markersize=14, color=:gray,
        strokecolor=:gray, strokewidth=2)   # control (should stay dark)
    CairoMakie.text!(ax1, -4.3, 4.0; text="□ triad members   ✕ uncoupled control",
        color=:black, fontsize=11, align=(:left, :center))

    ax2 = CairoMakie.Axis(fig[1, 3], title="Recipient mode at (f_l=2, f_n=3): recovered vs imposed",
        xlabel="x", ylabel="amplitude")
    nrm(v) = v ./ maximum(abs, v)
    s = sign(sum(nrm(psi) .* nrm(mn)))           # fix the arbitrary sign of the SVD mode
    CairoMakie.lines!(ax2, x, s .* nrm(psi), label="recovered (TOD)", color=:crimson, linewidth=3)
    CairoMakie.lines!(ax2, x, nrm(mn), label="imposed m_n = m_k·m_l", color=:black, linewidth=2, linestyle=:dash)
    CairoMakie.axislegend(ax2; position=:rb)

    outpath = joinpath(@__DIR__, "triadic_orthogonal_decomposition.png")
    CairoMakie.save(outpath, fig)
    println("Saved figure: $outpath")
    return result
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_tod_example()
end
