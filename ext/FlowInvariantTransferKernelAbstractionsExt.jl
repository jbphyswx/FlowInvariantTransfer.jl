module FlowInvariantTransferKernelAbstractionsExt

using KernelAbstractions: KernelAbstractions as KA, @kernel, @index
using FlowInvariantTransfer: FlowInvariantTransfer as FET
using FlowInvariantTransfer.Types: GPUBackend, ShellToShellResult, AbstractInvariant, KineticEnergy, Helicity, Enstrophy

# ---------------------------------------------------------------------------
# Device kernels
# ---------------------------------------------------------------------------

# Kinetic Energy Density Kernel
@kernel function transfer_density_ke_kernel!(t, @Const(velocity_hat), @Const(N̂), D)
    I = @index(Global, Cartesian)
    FT = eltype(t)
    s = zero(FT)
    for c in 1:D
        s += real(conj(velocity_hat[I, c]) * N̂[I, c])
    end
    t[I] = s
end

# Helicity Density Kernel
@kernel function transfer_density_helicity_kernel!(t, @Const(velocity_hat), @Const(N̂), ks1, ks2, ks3)
    I = @index(Global, Cartesian)
    kx = ks1[I[1]]
    ky = ks2[I[2]]
    kz = ks3[I[3]]
    ux = velocity_hat[I, 1]
    uy = velocity_hat[I, 2]
    uz = velocity_hat[I, 3]
    ωx = im * (ky * uz - kz * uy)
    ωy = im * (kz * ux - kx * uz)
    ωz = im * (kx * uy - ky * ux)
    t[I] = real(conj(ωx) * N̂[I, 1] + conj(ωy) * N̂[I, 2] + conj(ωz) * N̂[I, 3])
end

# Enstrophy Density Kernel
@kernel function transfer_density_enstrophy_kernel!(t, @Const(velocity_hat), @Const(N̂), ks1, ks2)
    I = @index(Global, Cartesian)
    kx = ks1[I[1]]
    ky = ks2[I[2]]
    ω̂   = im * (kx * velocity_hat[I, 2] - ky * velocity_hat[I, 1])
    N̂_ω = im * (kx * N̂[I, 2] - ky * N̂[I, 1])
    t[I] = real(conj(ω̂) * N̂_ω)
end

# ---------------------------------------------------------------------------
# ShellToShell Transfer GPU Dispatch
# ---------------------------------------------------------------------------
function FET.ShellToShellTransfer._calculate_shell_to_shell!(
    result::ShellToShellResult,
    ws::FET.Workspaces.ShellToShellWorkspace,
    velocity_hat,
    ks,
    gpu_backend::GPUBackend,
    spectral;            # transform backend for each per-mediator nonlinear term
    dealiasing::Bool,
    verify_antisymmetry::Bool,
    invariant::AbstractInvariant = KineticEnergy(),
)
    dev = gpu_backend.backend
    N_sh = size(result.transfer_matrix, 1)
    FT = real(eltype(velocity_hat))
    nd = length(ks)
    ns = size(velocity_hat)[1:nd]
    D = size(velocity_hat, nd+1)
    
    # Pre-allocate output matrix on device/host matching velocity_hat type
    fill!(result.transfer_matrix, zero(FT))
    fill!(result.net_transfer, zero(FT))
    
    # Ensure ks components are available as KA-accessible arrays
    ks1 = KA.allocate(dev, FT, length(ks[1]))
    KA.copyto!(dev, ks1, collect(ks[1]))
    ks2 = KA.allocate(dev, FT, length(ks[2]))
    KA.copyto!(dev, ks2, collect(ks[2]))
    ks3 = nd == 3 ? KA.allocate(dev, FT, length(ks[3])) : nothing
    if nd == 3
        KA.copyto!(dev, ks3, collect(ks[3]))
    end
    
    # Loop over mediator shells m
    for m in 1:N_sh
        # 1. Filter velocity field to shell m
        # We can run a small device assignment or reuse ws.û_m if it is a GPU array
        fill!(ws.û_m, zero(eltype(ws.û_m)))
        # Filter (standard broadcasting or device assignment)
        # Note: Since ws is allocated matching velocity_hat, broadcasting works device-agnostically:
        # ws.û_m .= velocity_hat .* (ws.shell_idx .== m)
        ws.û_m .= velocity_hat .* reshape(ws.shell_idx .== m, ns..., 1)
        
        # 2. Compute nonlinear term on GPU
        # In a real GPU run, we expect FFTW to be replaced by a GPU FFT backend.
        # But we fall back to SerialBackend computation or a KA-compatible FFT.
        # For this KA implementation, we dispatch using compute_nonlinear_term! with Serial/FFTBackend depending on array type.
        FET.NonlinearTerm.compute_nonlinear_term!(ws.nonlinear, ws.û_m, ks; dealiasing=dealiasing, spectral=spectral, advecting_hat=velocity_hat)
        
        # 3. Compute per-mode transfer density on GPU using KA kernel
        # Run our custom KA transfer density kernel
        if invariant isa KineticEnergy
            kernel = transfer_density_ke_kernel!(dev)
            event = kernel(ws.transfer_density, velocity_hat, ws.nonlinear.N̂, D, ndrange=ns)
            KA.wait(event)
        elseif invariant isa Helicity
            kernel = transfer_density_helicity_kernel!(dev)
            event = kernel(ws.transfer_density, velocity_hat, ws.nonlinear.N̂, ks1, ks2, ks3, ndrange=ns)
            KA.wait(event)
        elseif invariant isa Enstrophy
            kernel = transfer_density_enstrophy_kernel!(dev)
            event = kernel(ws.transfer_density, velocity_hat, ws.nonlinear.N̂, ks1, ks2, ndrange=ns)
            KA.wait(event)
        end
        
        # 4. Accumulate receiver shells n
        for n in 1:N_sh
            # Use reduction or simple sum
            # For simplicity, do a host-side sum or element-wise sum of selected indices
            mask_n = ws.shell_idx .== n
            result.transfer_matrix[n, m] = sum(ws.transfer_density .* mask_n)
        end
    end
    
    # Compute net transfer
    for n in 1:N_sh
        s = zero(FT)
        for m in 1:N_sh
            s += result.transfer_matrix[n, m]
        end
        result.net_transfer[n] = s
    end
    
    # Antisymmetry error
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
