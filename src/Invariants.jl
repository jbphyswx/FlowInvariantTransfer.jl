module Invariants

using ..Types: Types
using FlowTransformBindings: FlowTransformBindings as FTB

export transfer_density, transfer_density!, transfer_density_scatter!

# ---------------------------------------------------------------------------
# Per-mode transfer density
# ---------------------------------------------------------------------------
#
# Every Fourier-space transfer diagnostic is a shell sum of a real per-mode transfer density `t[I]`
# formed from the velocity `û` and the nonlinear term `N̂`:
#
#   KineticEnergy : t[I] = Σ_c Re{ conj(û_c)  N̂_c }
#   PassiveScalar : t[I] = Re{ conj(θ̂)  N̂_θ },   N̂_θ = FFT[(u·∇)θ]   (same dot form, M=1 field)
#   Helicity (3D) : t[I] = Σ_c Re{ conj(ω̂_c)  N̂_c },   ω̂ = i k × û
#   Enstrophy     : t[I] = Re{ conj(ω̂) · N̂_ω },   ω̂ = i k × û,  N̂_ω = i k × N̂.
#                   2D: scalar vorticity, conserved (Σ_k t = 0).
#                   3D: vector vorticity; N̂_ω = curl[(u·∇)u] = (u·∇)ω − (ω·∇)u includes vortex
#                   stretching, so Σ_k t ≠ 0 (net production).
# ---------------------------------------------------------------------------

"""
    transfer_density!(t, invariant, velocity_hat, N̂, ks) -> t

Write the real per-mode transfer density for `invariant` into `t` (shape `ns`),
given Fourier-space velocity `velocity_hat` and nonlinear term `N̂` (both shape
`(ns..., D)`) and wavenumber vectors `ks` (length `D`). No allocations.
"""
function transfer_density! end

# t[I] = Σ_c Re{ conj(carrier_c) · N̂_c }, for kinetic energy (carrier û) and passive-scalar variance
# (carrier θ̂). Broadcasts only, so host and device arrays share it.
function _transfer_density_dot!(t, carrier_hat, N̂, ks)
    d = length(ks) + 1
    M = size(carrier_hat, d)
    # `selectdim` takes `d` directly so `d` const-folds from `length(ks)`; a closure capturing `d`
    # makes the component views box.
    if M == 1
        a1 = selectdim(carrier_hat, d, 1); b1 = selectdim(N̂, d, 1)
        @. t = real(conj(a1) * b1)
    elseif M == 2
        a1 = selectdim(carrier_hat, d, 1); b1 = selectdim(N̂, d, 1)
        a2 = selectdim(carrier_hat, d, 2); b2 = selectdim(N̂, d, 2)
        @. t = real(conj(a1) * b1) + real(conj(a2) * b2)
    elseif M == 3
        a1 = selectdim(carrier_hat, d, 1); b1 = selectdim(N̂, d, 1)
        a2 = selectdim(carrier_hat, d, 2); b2 = selectdim(N̂, d, 2)
        a3 = selectdim(carrier_hat, d, 3); b3 = selectdim(N̂, d, 3)
        @. t = real(conj(a1) * b1) + real(conj(a2) * b2) + real(conj(a3) * b3)
    else
        fill!(t, zero(eltype(t)))
        for c in 1:M
            ac = selectdim(carrier_hat, d, c); bc = selectdim(N̂, d, c)
            @. t += real(conj(ac) * bc)
        end
    end
    return t
end

transfer_density!(t, ::Types.KineticEnergy, velocity_hat, N̂, ks) =
    _transfer_density_dot!(t, velocity_hat, N̂, ks)

transfer_density!(t, ::Types.PassiveScalar, scalar_hat, N̂, ks) =
    _transfer_density_dot!(t, scalar_hat, N̂, ks)

# ---------------------------------------------------------------------------
# Vorticity (k×û) invariants: host loops over the per-mode wavenumbers. Device arrays go through the
# KernelAbstractions extension's kernels (via `GPUBackend`).
# ---------------------------------------------------------------------------

# 3D helicity: ω̂ = i k×û; t[I] = Σ_c Re{ conj(ω̂_c) N̂_c }.
function transfer_density!(t, ::Types.Helicity, velocity_hat, N̂, ks)
    nd = length(ks)
    nd == 3 || throw(ArgumentError("Helicity transfer is defined in 3D only (got nd=$nd)."))
    ns = size(velocity_hat)[1:nd]
    @inbounds for I in CartesianIndices(ns)
        kx = FTB.derivative_wavenumber(ks[1], I[1])
        ky = FTB.derivative_wavenumber(ks[2], I[2])
        kz = FTB.derivative_wavenumber(ks[3], I[3])
        ux = velocity_hat[I, 1]; uy = velocity_hat[I, 2]; uz = velocity_hat[I, 3]
        ωx = im * (ky * uz - kz * uy)
        ωy = im * (kz * ux - kx * uz)
        ωz = im * (kx * uy - ky * ux)
        t[I] = real(conj(ωx) * N̂[I, 1] + conj(ωy) * N̂[I, 2] + conj(ωz) * N̂[I, 3])
    end
    return t
end

# Enstrophy: ω̂ = i(k_x û_y − k_y û_x) in 2D, ω̂ = i k×û in 3D; N̂_ω = i k×N̂.
function transfer_density!(t, ::Types.Enstrophy, velocity_hat, N̂, ks)
    nd = length(ks)
    ns = size(velocity_hat)[1:nd]
    if nd == 2
        @inbounds for I in CartesianIndices(ns)
            kx = FTB.derivative_wavenumber(ks[1], I[1])
            ky = FTB.derivative_wavenumber(ks[2], I[2])
            ω̂   = im * (kx * velocity_hat[I, 2] - ky * velocity_hat[I, 1])
            N̂_ω = im * (kx * N̂[I, 2] - ky * N̂[I, 1])
            t[I] = real(conj(ω̂) * N̂_ω)
        end
    elseif nd == 3
        @inbounds for I in CartesianIndices(ns)
            kx = FTB.derivative_wavenumber(ks[1], I[1])
            ky = FTB.derivative_wavenumber(ks[2], I[2])
            kz = FTB.derivative_wavenumber(ks[3], I[3])
            ux = velocity_hat[I, 1]; uy = velocity_hat[I, 2]; uz = velocity_hat[I, 3]
            Nx = N̂[I, 1];           Ny = N̂[I, 2];           Nz = N̂[I, 3]
            ωx  = im * (ky * uz - kz * uy); ωy  = im * (kz * ux - kx * uz); ωz  = im * (kx * uy - ky * ux)
            Nωx = im * (ky * Nz - kz * Ny); Nωy = im * (kz * Nx - kx * Nz); Nωz = im * (kx * Ny - ky * Nx)
            t[I] = real(conj(ωx) * Nωx + conj(ωy) * Nωy + conj(ωz) * Nωz)
        end
    else
        throw(ArgumentError(
            "Enstrophy transfer is defined in 2D (scalar vorticity, conserved) or 3D " *
            "(vector vorticity, non-conservative via stretching); got nd=$nd."))
    end
    return t
end

# ---------------------------------------------------------------------------
# Fused density + shell reduction
# ---------------------------------------------------------------------------

"""
    transfer_density_scatter!(T_spec, invariant, carrier_hat, N̂, ks, shell_idx) -> T_spec

Shell sums of the transfer density, without materialising the density. `T_spec[n]` accumulates
`w(I)·t[I]` over the modes `I` with `shell_idx[I] == n`, for the same per-mode density `t` as
[`transfer_density!`](@ref) and the Hermitian weight `w` of the spectral layout
(`FlowTransformBindings.mode_fold`) — 1 throughout on a full spectrum, and on a half spectrum the
factor by which each stored mode accounts for the ones it stands in for. Callers that need the
per-mode density itself use [`transfer_density!`](@ref).
"""
function transfer_density_scatter!(T_spec, invariant::Types.AbstractInvariant, carrier_hat, N̂, ks,
                                   shell_idx::AbstractArray{Int})
    nd = length(ks)
    ms = size(carrier_hat)[1:nd]
    M  = size(carrier_hat, nd + 1)
    fill!(T_spec, zero(eltype(T_spec)))
    @inbounds for I in CartesianIndices(ms)
        n = shell_idx[I]
        n == 0 && continue
        T_spec[n] += FTB.mode_fold(ks, I) * _density_at(invariant, carrier_hat, N̂, ks, I, M)
    end
    return T_spec
end

# Per-mode density at a single mode, matching `transfer_density!` term for term.
@inline function _density_at(::Union{Types.KineticEnergy, Types.PassiveScalar}, û, N̂, ks, I, M)
    s = zero(real(eltype(û)))
    @inbounds for c in 1:M
        s += real(conj(û[I, c]) * N̂[I, c])
    end
    return s
end

@inline function _density_at(::Types.Helicity, û, N̂, ks, I, M)
    length(ks) == 3 || throw(ArgumentError("Helicity transfer is defined in 3D only (got nd=$(length(ks)))."))
    @inbounds begin
        kx = FTB.derivative_wavenumber(ks[1], I[1])
        ky = FTB.derivative_wavenumber(ks[2], I[2])
        kz = FTB.derivative_wavenumber(ks[3], I[3])
        ux = û[I, 1]; uy = û[I, 2]; uz = û[I, 3]
        ωx = im * (ky * uz - kz * uy)
        ωy = im * (kz * ux - kx * uz)
        ωz = im * (kx * uy - ky * ux)
        return real(conj(ωx) * N̂[I, 1] + conj(ωy) * N̂[I, 2] + conj(ωz) * N̂[I, 3])
    end
end

@inline function _density_at(::Types.Enstrophy, û, N̂, ks, I, M)
    nd = length(ks)
    @inbounds if nd == 2
        kx = FTB.derivative_wavenumber(ks[1], I[1])
        ky = FTB.derivative_wavenumber(ks[2], I[2])
        ω̂   = im * (kx * û[I, 2] - ky * û[I, 1])
        N̂_ω = im * (kx * N̂[I, 2] - ky * N̂[I, 1])
        return real(conj(ω̂) * N̂_ω)
    elseif nd == 3
        kx = FTB.derivative_wavenumber(ks[1], I[1])
        ky = FTB.derivative_wavenumber(ks[2], I[2])
        kz = FTB.derivative_wavenumber(ks[3], I[3])
        ux = û[I, 1]; uy = û[I, 2]; uz = û[I, 3]
        Nx = N̂[I, 1];  Ny = N̂[I, 2];  Nz = N̂[I, 3]
        ωx  = im * (ky * uz - kz * uy); ωy  = im * (kz * ux - kx * uz); ωz  = im * (kx * uy - ky * ux)
        Nωx = im * (ky * Nz - kz * Ny); Nωy = im * (kz * Nx - kx * Nz); Nωz = im * (kx * Ny - ky * Nx)
        return real(conj(ωx) * Nωx + conj(ωy) * Nωy + conj(ωz) * Nωz)
    else
        throw(ArgumentError(
            "Enstrophy transfer is defined in 2D (scalar vorticity, conserved) or 3D " *
            "(vector vorticity, non-conservative via stretching); got nd=$nd."))
    end
end

"""
    transfer_density(invariant, velocity_hat, N̂, ks) -> Array

Allocating version of [`transfer_density!`](@ref): returns a real array of shape
`ns` with the per-mode transfer density.
"""
function transfer_density(invariant::Types.AbstractInvariant, velocity_hat, N̂, ks)
    nd = length(ks)
    ns = size(velocity_hat)[1:nd]
    FT = real(eltype(velocity_hat))
    t  = similar(velocity_hat, FT, ns...)
    return transfer_density!(t, invariant, velocity_hat, N̂, ks)
end

end # module Invariants
