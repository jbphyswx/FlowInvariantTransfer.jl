module FlowInvariantTransferKernelAbstractionsExt

using KernelAbstractions: KernelAbstractions as KA, @kernel, @index
using FlowInvariantTransfer: FlowInvariantTransfer as FET
using FlowInvariantTransfer.Types: GPUBackend, ShellToShellResult, AbstractShellBinning, AbstractInvariant, KineticEnergy, Helicity, Enstrophy
using FlowInvariantTransfer.Workspaces: ScaleToScaleWorkspace
using FlowInvariantTransfer.ShellBinning: assign_shells
using FlowInvariantTransfer.Utils: dealiasing_mask

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

# Scale-to-Scale Triad Kernel
@kernel function scale_to_scale_triad_kernel!(
    net_transfer,
    T_mat,
    @Const(velocity_hat),
    @Const(ks1),
    @Const(ks2),
    @Const(ks3),
    @Const(shell_idx),
    @Const(mask),
    N_sh,
    nd,
    D,
    FT,
    invariant_type, # 1: KineticEnergy, 2: Helicity, 3: Enstrophy
    binning_present
)
    k_idx = @index(Global, Cartesian)
    ns = size(net_transfer)
    
    if !mask[k_idx]
        net_transfer[k_idx] = zero(FT)
        # return # avoid return inside KA kernels
    else
        net_k = zero(FT)
        
        # We perform the loop over p_idx
        # Note: In a production GPU solver, this loop is tiled/parallelised across threads,
        # but for compatibility with generic KA backends, a 1D parallel loop over k and sequential inner p loop is standard.
        for p_idx in CartesianIndices(ns)
            if !mask[p_idx]
                continue
            end
            
            # q = k - p (with periodic wrap-around)
            q1 = mod(k_idx[1] - p_idx[1], ns[1]) + 1
            q2 = mod(k_idx[2] - p_idx[2], ns[2]) + 1
            q3 = nd == 3 ? mod(k_idx[3] - p_idx[3], ns[3]) + 1 : 1
            q_idx = nd == 3 ? CartesianIndex(q1, q2, q3) : CartesianIndex(q1, q2)
            
            if !mask[q_idx]
                continue
            end
            
            # S_val computation based on invariant
            S_val = zero(FT)
            if invariant_type == 1 # KineticEnergy
                k_dot_uq = zero(eltype(velocity_hat))
                k_dot_uq += ks1[k_idx[1]] * velocity_hat[q_idx, 1]
                k_dot_uq += ks2[k_idx[2]] * velocity_hat[q_idx, 2]
                if nd == 3
                    k_dot_uq += ks3[k_idx[3]] * velocity_hat[q_idx, 3]
                end
                
                uk_dot_up = zero(eltype(velocity_hat))
                for c in 1:D
                    uk_dot_up += conj(velocity_hat[k_idx, c]) * velocity_hat[p_idx, c]
                end
                S_val = -imag(k_dot_uq * uk_dot_up)
                
            elseif invariant_type == 2 # Helicity
                k_dot_uq = ks1[k_idx[1]] * velocity_hat[q_idx, 1] + ks2[k_idx[2]] * velocity_hat[q_idx, 2] + ks3[k_idx[3]] * velocity_hat[q_idx, 3]
                
                kx, ky, kz = ks1[k_idx[1]], ks2[k_idx[2]], ks3[k_idx[3]]
                ux, uy, uz = velocity_hat[k_idx, 1], velocity_hat[k_idx, 2], velocity_hat[k_idx, 3]
                ωx = im * (ky * uz - kz * uy)
                ωy = im * (kz * ux - kx * uz)
                ωz = im * (kx * uy - ky * ux)
                
                ωk_dot_up = conj(ωx) * velocity_hat[p_idx, 1] + conj(ωy) * velocity_hat[p_idx, 2] + conj(ωz) * velocity_hat[p_idx, 3]
                S_val = -imag(k_dot_uq * ωk_dot_up)
                
            elseif invariant_type == 3 # Enstrophy
                k_dot_uq = ks1[k_idx[1]] * velocity_hat[q_idx, 1] + ks2[k_idx[2]] * velocity_hat[q_idx, 2]
                
                kx, ky = ks1[k_idx[1]], ks2[k_idx[2]]
                ω_k = im * (kx * velocity_hat[k_idx, 2] - ky * velocity_hat[k_idx, 1])
                
                px, py = ks1[p_idx[1]], ks2[p_idx[2]]
                ω_p = im * (px * velocity_hat[p_idx, 2] - py * velocity_hat[p_idx, 1])
                
                ωk_dot_ωp = conj(ω_k) * ω_p
                S_val = -imag(k_dot_uq * ωk_dot_ωp)
            end
            
            net_k += S_val
            
            if binning_present
                K_sh = shell_idx[k_idx]
                Q_sh = shell_idx[p_idx]
                if K_sh > 0 && Q_sh > 0
                    # Note: Concurrent atomic addition is required since multiple threads write to the same T_mat bins.
                    # In KernelAbstractions, @atomic is device-agnostic.
                    KA.@atomic T_mat[K_sh, Q_sh] += S_val
                end
            end
        end
        net_transfer[k_idx] = net_k
    end
end

# ---------------------------------------------------------------------------
# ShellToShell Transfer GPU Dispatch
# ---------------------------------------------------------------------------
function FET.ShellToShellTransfer._calculate_shell_to_shell!(
    result::ShellToShellResult,
    ws::FET.Workspaces.ShellToShellWorkspace,
    velocity_hat,
    ks,
    gpu_backend::GPUBackend;
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
        FET.NonlinearTerm.compute_nonlinear_term!(ws.nonlinear, ws.û_m, ks; dealiasing=dealiasing, backend=FET.SerialBackend())
        
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

# ---------------------------------------------------------------------------
# ScaleToScale (Mode-to-Mode) Transfer GPU Dispatch
# ---------------------------------------------------------------------------
function FET.ScaleToScaleTransfer._calculate_mode_to_mode!(
    ws::ScaleToScaleWorkspace,
    velocity_hat,
    ks,
    gpu_backend::GPUBackend;
    binning::Union{Nothing, AbstractShellBinning},
    invariant::AbstractInvariant,
    dealiasing::Bool,
)
    dev = gpu_backend.backend
    nd = length(ks)
    ns = size(velocity_hat)[1:nd]
    D  = size(velocity_hat, nd+1)
    FT = real(eltype(velocity_hat))
    
    N_sh = binning !== nothing ? size(ws.T_mat, 1) : 0
    mask = dealiasing ? dealiasing_mask(ns) : trues(ns...)
    
    # Pre-allocate / fill outputs
    fill!(ws.net_transfer, zero(FT))
    if binning !== nothing
        fill!(ws.T_mat, zero(FT))
    end
    
    # Upload parameters to device
    ks1 = KA.allocate(dev, FT, length(ks[1]))
    KA.copyto!(dev, ks1, collect(ks[1]))
    ks2 = KA.allocate(dev, FT, length(ks[2]))
    KA.copyto!(dev, ks2, collect(ks[2]))
    ks3 = nd == 3 ? KA.allocate(dev, FT, length(ks[3])) : nothing
    if nd == 3
        KA.copyto!(dev, ks3, collect(ks[3]))
    end
    
    dev_shell_idx = KA.allocate(dev, Int, size(ws.shell_idx)...)
    KA.copyto!(dev, dev_shell_idx, ws.shell_idx)
    
    dev_mask = KA.allocate(dev, Bool, size(mask)...)
    KA.copyto!(dev, dev_mask, mask)
    
    invariant_type = invariant isa KineticEnergy ? 1 : (invariant isa Helicity ? 2 : 3)
    
    # Launch Scale-to-Scale Triad computation on device
    kernel = scale_to_scale_triad_kernel!(dev)
    event = kernel(
        ws.net_transfer,
        ws.T_mat,
        velocity_hat,
        ks1,
        ks2,
        ks3,
        dev_shell_idx,
        dev_mask,
        N_sh,
        nd,
        D,
        FT,
        invariant_type,
        binning !== nothing,
        ndrange=ns
    )
    KA.wait(event)
end

end # module
