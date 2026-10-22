module FlowInvariantTransferCairoMakieExt

using CairoMakie: CairoMakie
using FlowInvariantTransfer: FlowInvariantTransfer as FIT

# ---------------------------------------------------------------------------
# Override stub
# ---------------------------------------------------------------------------

"""
    plot_energy_transfer(result; kwargs...) -> Figure

Dispatch to the appropriate plot function based on result type.

- `SpectralFluxResult`: plots T(k) and Π(K) on semi-log axes.
- `CoarseGrainingFluxResult`: heatmap of Π_ℓ(x); a 3-D field shows the plane `slice` of its third
  axis (keyword, default 1).
- `ShellToShellResult`: heatmap of T(n,m) with diverging colormap.
- `TriadicOrthogonalDecompositionResult`: the mode bispectrum λ(f_l, f_n) and modal energy budget
  T(f_l, f_n) as heatmaps (Yeung–Chu–Schmidt Fig. 4 style).
"""
function FIT.plot_energy_transfer(result::FIT.Types.SpectralFluxResult; kwargs...)
    return _plot_spectral_flux(result; kwargs...)
end

function FIT.plot_energy_transfer(result::FIT.Types.CoarseGrainingFluxResult; kwargs...)
    return _plot_cg_flux(result; kwargs...)
end

function FIT.plot_energy_transfer(result::FIT.Types.ShellToShellResult; kwargs...)
    return _plot_shell_transfer_matrix(result; kwargs...)
end

function FIT.plot_energy_transfer(result::FIT.Types.TriadicOrthogonalDecompositionResult; kwargs...)
    return _plot_tod_bispectrum(result; kwargs...)
end

# ---------------------------------------------------------------------------
# SpectralFluxResult plot
# ---------------------------------------------------------------------------

function _plot_spectral_flux(r::FIT.Types.SpectralFluxResult;
    title::String = "Spectral Energy Flux",
    xscale = CairoMakie.log10,
)
    k = r.k_shells
    positive_k = k .> 0
    k_pos = k[positive_k]
    # Π[n] is the flux through the shell's upper edge.
    K = r.shell_edges[2:end]

    fig = CairoMakie.Figure(size = (900, 400))
    CairoMakie.Label(fig[0, :], title; fontsize = 18, font = :bold)

    # T(k)
    ax1 = CairoMakie.Axis(fig[1, 1];
        title  = "Transfer Spectrum T(k)",
        xlabel = "k",
        ylabel = "T(k)",
        xscale = xscale,
    )
    CairoMakie.lines!(ax1, k_pos, r.transfer_spectrum[positive_k]; color = :steelblue, linewidth = 2)
    CairoMakie.hlines!(ax1, [0.0]; color = :black, linewidth = 0.7, linestyle = :dash)

    # Π(K)
    ax2 = CairoMakie.Axis(fig[1, 2];
        title  = "Energy Flux Π(K)",
        xlabel = "K",
        ylabel = "Π(K)",
        xscale = xscale,
    )
    CairoMakie.lines!(ax2, K, r.flux; color = :crimson, linewidth = 2)
    CairoMakie.hlines!(ax2, [0.0]; color = :black, linewidth = 0.7, linestyle = :dash)

    return fig
end

# ---------------------------------------------------------------------------
# CoarseGrainingFluxResult plot
# ---------------------------------------------------------------------------

function _plot_cg_flux(r::FIT.Types.CoarseGrainingFluxResult;
    title::String = "Coarse-Graining Energy Flux Π_ℓ(x)",
    slice::Integer = 1,
)
    N = ndims(r.flux_field)
    N in (2, 3) || throw(ArgumentError(
        "plot_energy_transfer draws a 2-D field or a plane of a 3-D one; this flux field has $N dimensions"))
    Π = N == 2 ? r.flux_field : view(r.flux_field, :, :, slice)
    where_ = N == 2 ? "" : ", z index $slice"

    fig = CairoMakie.Figure()
    ax  = CairoMakie.Axis(fig[1, 1];
        title  = "$title  [ℓ = $(round(r.filter_scale; sigdigits=4))$where_]",
        xlabel = "x",
        ylabel = "y",
        aspect = CairoMakie.DataAspect(),
    )
    nx, ny = size(Π)
    vmax = maximum(abs, Π)
    hm = CairoMakie.heatmap!(ax, 1:nx, 1:ny, Π;
        colormap = :RdBu_9,
        colorrange = (-vmax, vmax),
    )
    CairoMakie.Colorbar(fig[1, 2], hm; label = "Π_ℓ")
    return fig
end

# ---------------------------------------------------------------------------
# ShellToShellResult plot
# ---------------------------------------------------------------------------

function _plot_shell_transfer_matrix(r::FIT.Types.ShellToShellResult;
    title::String = "Shell-to-Shell Transfer T(n,m)",
)
    T    = r.transfer_matrix
    N_sh = size(T, 1)
    k    = r.shell_centers

    fig  = CairoMakie.Figure(size = (800, 650))
    CairoMakie.Label(fig[0, :], title; fontsize = 18, font = :bold)

    # T[n, m] at (n, m): the shell that gives energy on x, the shell that receives it on y.
    ax1  = CairoMakie.Axis(fig[1, 1];
        title  = "T(n,m)",
        xlabel = "Giving shell n",
        ylabel = "Receiving shell m",
    )
    vmax = maximum(abs, T)
    hm   = CairoMakie.heatmap!(ax1, 1:N_sh, 1:N_sh, T;
        colormap   = :RdBu_9,
        colorrange = (-vmax, vmax),
    )
    CairoMakie.Colorbar(fig[1, 2], hm; label = "T(n,m)")

    # Net transfer bar chart
    ax2 = CairoMakie.Axis(fig[2, 1:2];
        title  = "Net Transfer Σ_m T(n,m) = −dE_n/dt",
        xlabel = "Shell center k",
        ylabel = "Net T(n)",
    )
    net = r.net_transfer
    colors = [v >= 0 ? :steelblue : :crimson for v in net]
    CairoMakie.barplot!(ax2, k, net; color = colors)
    CairoMakie.hlines!(ax2, [0.0]; color = :black, linewidth = 0.7)

    return fig
end

# ---------------------------------------------------------------------------
# TriadicOrthogonalDecompositionResult plot (Yeung–Chu–Schmidt Fig. 4 style)
# ---------------------------------------------------------------------------

"""
    _plot_tod_bispectrum(r; mode=1, fmax=nothing, title="Triadic Orthogonal Decomposition")

Bispectrum figure for a TOD result: the mode bispectrum `λ(f_l, f_n)` (triadic coupling strength,
sequential colormap) and the modal energy budget `T(f_l, f_n)` (signed energy flow, diverging
colormap) for the selected `mode` (default the leading mode), as in Yeung, Chu & Schmidt (2026,
Fig. 4). `f_n` is restricted to the non-negative half (real data); pass `fmax` to clip the symmetric
`f_l` band.
"""
function _plot_tod_bispectrum(r::FIT.Types.TriadicOrthogonalDecompositionResult;
    mode::Int = 1,
    fmax = nothing,
    title::String = "Triadic Orthogonal Decomposition",
)
    freqs  = r.frequencies
    nmode  = size(r.mode_bispectrum, 3)
    1 <= mode <= nmode || throw(ArgumentError("mode=$mode out of range 1:$nmode"))

    posn = findall(>=(zero(eltype(freqs))), freqs)                      # f_n ≥ 0 for real data
    band = fmax === nothing ? eachindex(freqs) :
           findall(ff -> -fmax <= ff <= fmax, freqs)                    # symmetric f_l window
    fl_ax = freqs[band]
    fn_ax = freqs[posn]
    λ = r.mode_bispectrum[band, posn, mode]
    T = r.modal_energy_budget[band, posn, mode]

    fig = CairoMakie.Figure(size = (1050, 460))
    CairoMakie.Label(fig[0, :], "$title  (mode $mode)"; fontsize = 18, font = :bold)

    ax1 = CairoMakie.Axis(fig[1, 1];
        title = "Mode bispectrum λ(f_l, f_n)", xlabel = "f_l", ylabel = "f_n")
    λmax = maximum(λ; init = zero(eltype(λ)))
    hm1 = CairoMakie.heatmap!(ax1, fl_ax, fn_ax, λ;
        colormap = CairoMakie.cgrad([:white, :gold, :orangered, :darkred]),
        colorrange = (0, λmax > 0 ? λmax : one(λmax)))
    CairoMakie.Colorbar(fig[1, 2], hm1; label = "coupling strength")

    ax2 = CairoMakie.Axis(fig[1, 3];
        title = "Modal energy budget T(f_l, f_n)", xlabel = "f_l", ylabel = "f_n")
    vmax = maximum(abs, T; init = zero(eltype(T)))
    vmax = vmax > 0 ? vmax : one(vmax)
    hm2 = CairoMakie.heatmap!(ax2, fl_ax, fn_ax, T;
        colormap = :RdBu_9, colorrange = (-vmax, vmax))
    CairoMakie.Colorbar(fig[1, 4], hm2; label = "T (energy flow)")

    return fig
end

end # module FlowInvariantTransferCairoMakieExt
