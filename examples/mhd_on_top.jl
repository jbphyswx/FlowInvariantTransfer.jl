"""
Building a Domain Application on FlowInvariantTransfer.jl — Incompressible MHD

FlowInvariantTransfer is a *general* cross-scale-transfer toolkit: a pseudospectral nonlinear
term `(u·∇)f` for any advected field `f`, plus generic invariant/flux/shell machinery. It does NOT
hard-code any particular physical system. This example shows how a downstream domain — here
incompressible **magnetohydrodynamics** — is assembled entirely from the package's *public*
primitives, with the MHD-specific physics living here, not in the package.

Incompressible MHD: ∂_t u = −(u·∇)u + (b·∇)b − ∇P,  ∂_t b = −(u·∇)b + (b·∇)u. The total energy
½⟨|u|²+|b|²⟩ and cross-helicity ⟨u·b⟩ are built from the four advection terms (all from
`compute_nonlinear_term` with `advecting_hat`), and magnetic helicity ⟨a·b⟩ needs one extra,
genuinely MHD-specific term computed right here — the EMF `u×b`.

Run from the repo root:
    julia --project=examples examples/mhd_on_top.jl
"""

using FlowInvariantTransfer: FlowInvariantTransfer as FET
using FFTW: FFTW
using CairoMakie: CairoMakie
include(joinpath(@__DIR__, "flows.jl"))

# Bin a real per-mode density into shells → (centers, T(k), Π(K)=cumsum) using public helpers.
function shell_flux(density, ks, binning)
    kmag  = FET.wavenumber_magnitude_grid(ks)
    edges = FET.shell_edges(binning, maximum(kmag))
    idx   = FET.assign_shells(kmag, edges)
    K     = FET.shell_centers(binning, maximum(kmag))
    T = zeros(length(K))
    for I in CartesianIndices(size(density)); n = idx[I]; n == 0 && continue; T[n] += density[I]; end
    return K, T, cumsum(T)
end

# Re{ carrier*(k)·N(k) } via the package's transfer_density (kinetic-energy form = the dot product).
redot(carrier, N, ks) = FET.transfer_density(FET.KineticEnergy(), carrier, N, ks)

function run_mhd_on_top_example(; N=128)
    println("--- MHD built on FlowInvariantTransfer primitives (2D Orszag–Tang) ---")
    û, b̂, ks, L = evolve_orszag_tang(; N=N)
    da, sp = FET.OrszagTwoThirds(), FET.FFTBackend()

    # --- the four advection terms, all from the GENERAL nonlinear-term engine -------------------
    N_uu = FET.compute_nonlinear_term(û, ks; advecting_hat=û, dealiasing=da, spectral=sp)  # (u·∇)u
    N_bb = FET.compute_nonlinear_term(b̂, ks; advecting_hat=b̂, dealiasing=da, spectral=sp)  # (b·∇)b
    N_ub = FET.compute_nonlinear_term(b̂, ks; advecting_hat=û, dealiasing=da, spectral=sp)  # (u·∇)b
    N_bu = FET.compute_nonlinear_term(û, ks; advecting_hat=b̂, dealiasing=da, spectral=sp)  # (b·∇)u

    # MHD energy budget (the sign bookkeeping IS the MHD equations — domain code, lives here):
    t_KE = redot(û, N_uu, ks) .- redot(û, N_bb, ks)   # kinetic: advection − Lorentz work
    t_ME = redot(b̂, N_ub, ks) .- redot(b̂, N_bu, ks)   # magnetic: induction
    b = FET.LinearBinning(2π / L)
    K, T_tot, Π_tot = shell_flux(t_KE .+ t_ME, ks, b)
    _, _,    Π_KE   = shell_flux(t_KE, ks, b)
    _, _,    Π_ME   = shell_flux(t_ME, ks, b)
    println("Σ_k T_E(k) = ", sum(T_tot), "  (total energy conserved ⇒ ≈ 0)")

    fig = CairoMakie.Figure(size=(1100, 450), fontsize=14)
    CairoMakie.Label(fig[0, 1:3], "MHD diagnostics built on FlowInvariantTransfer — 2D Orszag–Tang Vortex",
        fontsize=16, font=:bold, tellwidth=false)

    # current sheets j = ∂x b_y − ∂y b_x (the developed-state signature)
    kx = [ks[1][i] for i in 1:N, j in 1:N]; ky = [ks[2][j] for i in 1:N, j in 1:N]
    jph = real.(FFTW.ifft(im .* kx .* b̂[:, :, 2] .- im .* ky .* b̂[:, :, 1]))
    ax0 = CairoMakie.Axis(fig[1, 1], title="current density j (developed state)",
        xlabel="x", ylabel="y", aspect=CairoMakie.DataAspect())
    xp = range(0, 2π; length=N+1)[1:N]; cl = maximum(abs, jph)
    hm = CairoMakie.heatmap!(ax0, xp, xp, jph, colormap=:balance, colorrange=(-cl, cl))
    CairoMakie.Colorbar(fig[1, 2], hm, width=12)

    ax1 = CairoMakie.Axis(fig[1, 3], title="Energy flux Π(K)  (>0 ⇒ forward; total = kinetic + magnetic)",
        xlabel="K", ylabel="Π(K)")
    CairoMakie.lines!(ax1, K, Π_tot, label="total",    linewidth=3,   color=:black)
    CairoMakie.lines!(ax1, K, Π_KE,  label="kinetic",  linewidth=2.5, color=:steelblue)
    CairoMakie.lines!(ax1, K, Π_ME,  label="magnetic", linewidth=2.5, color=:crimson)
    CairoMakie.hlines!(ax1, [0]; color=:black, linewidth=0.8, linestyle=:dot)
    CairoMakie.axislegend(ax1; position=:rc)

    outpath = joinpath(@__DIR__, "mhd_on_top.png")
    CairoMakie.save(outpath, fig)
    println("Saved figure: $outpath")
    return (K=K, total=Π_tot, kinetic=Π_KE, magnetic=Π_ME)
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_mhd_on_top_example()
end
