module FlowInvariantTransferKernelAbstractionsExt

using KernelAbstractions: KernelAbstractions as KA, @kernel, @index
using FlowInvariantTransfer: FlowInvariantTransfer as FET
using FlowInvariantTransfer.Types: GPUBackend, ShellToShellResult, AbstractInvariant, KineticEnergy, Helicity, Enstrophy

# ---------------------------------------------------------------------------
# Device kernels (KernelAbstractions ≥ 0.9 API: launch then KA.synchronize).
#
# These run on any KA backend — including KA.CPU(), which is how the logic is
# validated in CI without GPU hardware. On a vendor backend (CUDA/ROC/Metal) the
# same kernels execute on-device; the transfer-density write is per-mode and the
# shell reduction uses fused broadcast+reduce (no scalar indexing), so the path is
# safe under `allowscalar(false)`.
# ---------------------------------------------------------------------------

# Kinetic energy: t[I] = Σ_c Re{ conj(û_c) N̂_c }
@kernel function transfer_density_ke_kernel!(t, @Const(velocity_hat), @Const(N̂), D)
    I = @index(Global, Cartesian)
    FT = eltype(t)
    s = zero(FT)
    for c in 1:D
        s += real(conj(velocity_hat[I, c]) * N̂[I, c])
    end
    t[I] = s
end

# Helicity (3D): t[I] = Re{ conj(ω̂)·N̂ }, ω̂ = i k × û
@kernel function transfer_density_helicity_kernel!(t, @Const(velocity_hat), @Const(N̂), ks1, ks2, ks3)
    I = @index(Global, Cartesian)
    kx = ks1[I[1]]; ky = ks2[I[2]]; kz = ks3[I[3]]
    ux = velocity_hat[I, 1]; uy = velocity_hat[I, 2]; uz = velocity_hat[I, 3]
    ωx = im * (ky * uz - kz * uy)
    ωy = im * (kz * ux - kx * uz)
    ωz = im * (kx * uy - ky * ux)
    t[I] = real(conj(ωx) * N̂[I, 1] + conj(ωy) * N̂[I, 2] + conj(ωz) * N̂[I, 3])
end

# Enstrophy (2D): scalar vorticity ω̂ = i(k_x û_y − k_y û_x)
@kernel function transfer_density_enstrophy_kernel!(t, @Const(velocity_hat), @Const(N̂), ks1, ks2)
    I = @index(Global, Cartesian)
    kx = ks1[I[1]]; ky = ks2[I[2]]
    ω̂   = im * (kx * velocity_hat[I, 2] - ky * velocity_hat[I, 1])
    N̂_ω = im * (kx * N̂[I, 2] - ky * N̂[I, 1])
    t[I] = real(conj(ω̂) * N̂_ω)
end

# Run the per-mode transfer-density kernel for the requested invariant.
function _launch_transfer_density!(dev, td, velocity_hat, N̂, ks, invariant, D, ns, ks_dev)
    if invariant isa KineticEnergy
        transfer_density_ke_kernel!(dev)(td, velocity_hat, N̂, D; ndrange = ns)
    elseif invariant isa Helicity
        length(ks) == 3 || throw(ArgumentError("Helicity transfer is 3D only (got nd=$(length(ks)))."))
        transfer_density_helicity_kernel!(dev)(td, velocity_hat, N̂, ks_dev[1], ks_dev[2], ks_dev[3]; ndrange = ns)
    elseif invariant isa Enstrophy
        length(ks) == 2 || throw(ArgumentError("GPU Enstrophy kernel is 2D only (got nd=$(length(ks)))."))
        transfer_density_enstrophy_kernel!(dev)(td, velocity_hat, N̂, ks_dev[1], ks_dev[2]; ndrange = ns)
    else
        throw(ArgumentError("GPU transfer-density kernel not implemented for $(typeof(invariant))."))
    end
    KA.synchronize(dev)
    return td
end

# ---------------------------------------------------------------------------
# Shell-to-shell transfer on a KA backend
# ---------------------------------------------------------------------------
function FET.ShellToShellTransfer._calculate_shell_to_shell!(
    result::ShellToShellResult,
    ws::FET.Workspaces.ShellToShellWorkspace,
    velocity_hat,
    ks,
    gpu_backend::GPUBackend,
    spectral;            # transform backend for each per-mediator nonlinear term
    dealiasing::FET.Types.AbstractDealiasing,
    verify_antisymmetry::Bool,
    invariant::AbstractInvariant = KineticEnergy(),
    advecting_hat = velocity_hat,
)
    dev  = gpu_backend.backend
    N_sh = size(result.transfer_matrix, 1)
    FT   = real(eltype(velocity_hat))
    nd   = length(ks)
    ns   = size(velocity_hat)[1:nd]
    D    = size(velocity_hat, nd + 1)      # components of the binned/carried primary field

    fill!(result.transfer_matrix, zero(FT))
    fill!(result.net_transfer, zero(FT))

    # Wavenumber components on the device (only needed by helicity/enstrophy kernels).
    ks_dev = ntuple(nd) do d
        a = KA.allocate(dev, FT, length(ks[d]))
        copyto!(a, collect(FT, ks[d]))
        a
    end

    for m in 1:N_sh
        # 1. Band-m field: û_m = velocity_hat ⊙ 1[shell == m]  (device broadcast, no scalar indexing)
        ws.û_m .= velocity_hat .* reshape(ws.shell_idx .== m, ns..., 1)

        # 2. Nonlinear term 𝒩̂_m = (u·∇)f_m (uses the spectral backend; on a GPU array this needs a
        #    GPU-FFT-capable spectral path, e.g. FFTBackend with cuFFT riding AbstractFFTs).
        FET.NonlinearTerm.compute_nonlinear_term!(ws.nonlinear, ws.û_m, ks;
            dealiasing = dealiasing, spectral = spectral, advecting_hat = advecting_hat)

        # 3. Per-mode transfer density via the device kernel.
        _launch_transfer_density!(dev, ws.transfer_density, velocity_hat, ws.nonlinear.N̂, ks, invariant, D, ns, ks_dev)

        # 4. Column m: A[n,m] = Σ_{I ∈ shell n} density[I]. Fused broadcast+reduce per receiver shell —
        #    device reduction returning a host scalar (no scalar indexing). An Atomix scatter-add over
        #    modes would drop the O(N_sh) factor; kept simple here since the GPU path is correctness-first.
        for n in 1:N_sh
            result.transfer_matrix[n, m] = sum(ws.transfer_density .* (ws.shell_idx .== n))
        end
    end

    # Net transfer Σ_m T(n,m) and antisymmetry check on the host matrix.
    for n in 1:N_sh
        s = zero(FT)
        for m in 1:N_sh
            s += result.transfer_matrix[n, m]
        end
        result.net_transfer[n] = s
    end

    max_asym = if verify_antisymmetry
        v = zero(FT)
        for n in 1:N_sh, m in 1:N_sh
            a = abs(result.transfer_matrix[n, m] + result.transfer_matrix[m, n])
            a > v && (v = a)
        end
        v
    else
        FT(NaN)
    end

    return max_asym
end

end # module
