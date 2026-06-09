module FlowEnergyTransferCairoMakieExt

using CairoMakie: CairoMakie
using FlowEnergyTransfer: FlowEnergyTransfer as FET
using FlowEnergyTransfer.Types: SpectralFluxResult, CoarseGrainingFluxResult, ShellToShellResult

# ---------------------------------------------------------------------------
# Override stub
# ---------------------------------------------------------------------------

"""
    plot_energy_transfer(result; kwargs...) -> Figure

Dispatch to the appropriate plot function based on result type.

- `SpectralFluxResult`: plots T(k) and Π(K) on semi-log axes.
- `CoarseGrainingFluxResult`: heatmap of Π_ℓ(x) (2D only).
- `ShellToShellResult`: heatmap of T(n,m) with diverging colormap.
"""
function FET.plot_energy_transfer(result::SpectralFluxResult; kwargs...)
    return _plot_spectral_flux(result; kwargs...)
end

function FET.plot_energy_transfer(result::CoarseGrainingFluxResult; kwargs...)
    return _plot_cg_flux(result; kwargs...)
end

function FET.plot_energy_transfer(result::ShellToShellResult; kwargs...)
    return _plot_shell_transfer_matrix(result; kwargs...)
end

# ---------------------------------------------------------------------------
# SpectralFluxResult plot
# ---------------------------------------------------------------------------

function _plot_spectral_flux(r::SpectralFluxResult{FT};
    title::String = "Spectral Energy Flux",
    xscale = CairoMakie.log10,
) where {FT}
    k = r.k_shells
    positive_k = k .> 0
    k_pos = k[positive_k]

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
    CairoMakie.lines!(ax2, k_pos, r.flux[positive_k]; color = :crimson, linewidth = 2)
    CairoMakie.hlines!(ax2, [0.0]; color = :black, linewidth = 0.7, linestyle = :dash)

    return fig
end

# ---------------------------------------------------------------------------
# CoarseGrainingFluxResult plot
# ---------------------------------------------------------------------------

function _plot_cg_flux(r::CoarseGrainingFluxResult{FT, N};
    title::String = "Coarse-Graining Energy Flux Π_ℓ(x)",
) where {FT, N}
    N == 2 || @warn "plot_energy_transfer: CoarseGrainingFluxResult has $N spatial dimensions; only 2D heatmaps are supported."

    fig = CairoMakie.Figure()
    ax  = CairoMakie.Axis(fig[1, 1];
        title  = "$title  [ℓ = $(round(r.filter_scale; sigdigits=4))]",
        xlabel = "x",
        ylabel = "y",
        aspect = CairoMakie.DataAspect(),
    )

    if N == 2
        Π  = r.flux_field
        nx, ny = size(Π)
        # Symmetric colormap around zero
        vmax = maximum(abs, Π)
        hm = CairoMakie.heatmap!(ax, 1:nx, 1:ny, Π;
            colormap = :RdBu_9,
            colorrange = (-vmax, vmax),
        )
        CairoMakie.Colorbar(fig[1, 2], hm; label = "Π_ℓ")
    else
        # Fallback: plot first slice
        CairoMakie.text!(ax, 0.5, 0.5; text = "3D result — plotting first slice",
            align = (:center, :center))
    end

    return fig
end

# ---------------------------------------------------------------------------
# ShellToShellResult plot
# ---------------------------------------------------------------------------

function _plot_shell_transfer_matrix(r::ShellToShellResult{FT};
    title::String = "Shell-to-Shell Transfer T(n,m)",
) where {FT}
    T    = r.transfer_matrix
    N_sh = size(T, 1)
    k    = r.shell_centers

    fig  = CairoMakie.Figure(size = (800, 650))
    CairoMakie.Label(fig[0, :], title; fontsize = 18, font = :bold)

    # Transfer matrix heatmap (diverging)
    ax1  = CairoMakie.Axis(fig[1, 1];
        title  = "T(n,m)",
        xlabel = "Donor shell m",
        ylabel = "Receiver shell n",
    )
    vmax = maximum(abs, T)
    hm   = CairoMakie.heatmap!(ax1, 1:N_sh, 1:N_sh, T;
        colormap   = :RdBu_9,
        colorrange = (-vmax, vmax),
    )
    CairoMakie.Colorbar(fig[1, 2], hm; label = "T(n,m)")

    # Net transfer bar chart
    ax2 = CairoMakie.Axis(fig[2, 1:2];
        title  = "Net Transfer Σ_m T(n,m) per Receiver Shell",
        xlabel = "Shell center k",
        ylabel = "Net T(n)",
    )
    net = r.net_transfer
    colors = [v >= 0 ? :steelblue : :crimson for v in net]
    CairoMakie.barplot!(ax2, k, net; color = colors)
    CairoMakie.hlines!(ax2, [0.0]; color = :black, linewidth = 0.7)

    return fig
end

end # module FlowEnergyTransferCairoMakieExt
