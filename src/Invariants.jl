module Invariants

using ..Types: AbstractInvariant, KineticEnergy, Helicity, Enstrophy

export transfer_density, transfer_density!

# ---------------------------------------------------------------------------
# Per-mode transfer density
# ---------------------------------------------------------------------------
#
# Every Fourier-space transfer diagnostic (spectral flux, shell-to-shell) is a
# *shell sum* of a real per-mode transfer density `t[I]`. The only thing that
# changes between invariants is how `t[I]` is formed from the velocity `û` and
# the nonlinear term `N̂`. Centralising that here lets a single accumulation
# kernel serve kinetic energy, helicity (3D), and enstrophy (2D).
#
#   KineticEnergy : t[I] = Σ_c Re{ conj(û_c)  N̂_c }
#   Helicity (3D) : t[I] = Σ_c Re{ conj(ω̂_c)  N̂_c },   ω̂ = i k × û
#   Enstrophy (2D): t[I] =       Re{ conj(ω̂)   N̂_ω },   ω̂  = i(k_x û_y − k_y û_x)
#                                                        N̂_ω = i(k_x N̂_y − k_y N̂_x)
# ---------------------------------------------------------------------------

"""
    transfer_density!(t, invariant, velocity_hat, N̂, ks) -> t

Write the real per-mode transfer density for `invariant` into `t` (shape `ns`),
given Fourier-space velocity `velocity_hat` and nonlinear term `N̂` (both shape
`(ns..., D)`) and wavenumber vectors `ks` (length `D`). No allocations.
"""
function transfer_density! end

function transfer_density!(t, ::KineticEnergy, velocity_hat, N̂, ks::Tuple)
    nd = length(ks)
    ns = size(velocity_hat)[1:nd]
    D  = size(velocity_hat, nd + 1)
    FT = real(eltype(velocity_hat))
    @inbounds for I in CartesianIndices(ns)
        s = zero(FT)
        for c in 1:D
            s += real(conj(velocity_hat[I, c]) * N̂[I, c])
        end
        t[I] = s
    end
    return t
end

function transfer_density!(t, ::Helicity, velocity_hat, N̂, ks::Tuple)
    nd = length(ks)
    nd == 3 || throw(ArgumentError("Helicity transfer is defined in 3D only (got nd=$nd)."))
    ns = size(velocity_hat)[1:nd]
    @inbounds for I in CartesianIndices(ns)
        kx = ks[1][I[1]]; ky = ks[2][I[2]]; kz = ks[3][I[3]]
        ux = velocity_hat[I, 1]; uy = velocity_hat[I, 2]; uz = velocity_hat[I, 3]
        ωx = im * (ky * uz - kz * uy)
        ωy = im * (kz * ux - kx * uz)
        ωz = im * (kx * uy - ky * ux)
        t[I] = real(conj(ωx) * N̂[I, 1] + conj(ωy) * N̂[I, 2] + conj(ωz) * N̂[I, 3])
    end
    return t
end

function transfer_density!(t, ::Enstrophy, velocity_hat, N̂, ks::Tuple)
    nd = length(ks)
    nd == 2 || throw(ArgumentError("Enstrophy transfer is defined in 2D only (got nd=$nd)."))
    ns = size(velocity_hat)[1:nd]
    @inbounds for I in CartesianIndices(ns)
        kx = ks[1][I[1]]; ky = ks[2][I[2]]
        ω̂   = im * (kx * velocity_hat[I, 2] - ky * velocity_hat[I, 1])
        N̂_ω = im * (kx * N̂[I, 2] - ky * N̂[I, 1])
        t[I] = real(conj(ω̂) * N̂_ω)
    end
    return t
end

"""
    transfer_density(invariant, velocity_hat, N̂, ks) -> Array

Allocating version of [`transfer_density!`](@ref): returns a real array of shape
`ns` with the per-mode transfer density.
"""
function transfer_density(invariant::AbstractInvariant, velocity_hat, N̂, ks::Tuple)
    nd = length(ks)
    ns = size(velocity_hat)[1:nd]
    FT = real(eltype(velocity_hat))
    t  = Array{FT}(undef, ns...)
    return transfer_density!(t, invariant, velocity_hat, N̂, ks)
end

end # module Invariants
