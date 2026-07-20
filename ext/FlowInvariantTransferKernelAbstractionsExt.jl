module FlowInvariantTransferKernelAbstractionsExt

using KernelAbstractions: KernelAbstractions as KA, @kernel, @index
using FlowInvariantTransfer: FlowInvariantTransfer as FIT
using FlowInvariantTransfer.Backends: GPUBackend
using FlowInvariantTransfer.Types: ShellToShellResult, SpectralFluxResult, AbstractInvariant, KineticEnergy, Helicity, Enstrophy

# Copy a host array to a fresh device array on `dev` (a plain Array on GPUBackend(KA.CPU())). Used to
# move the integer shell-index grid onto the device: `assign_shells` always builds a host Array{Int},
# so broadcasting a shell mask (`shell_idx .== n`) against a device field would otherwise mix host and
# device arrays — a latent bug invisible only because CI runs on GPUBackend(KA.CPU()). Copying once keeps
# the band-filter and the shell reduction fully on-device under `allowscalar(false)`.
function _to_device(dev, A::AbstractArray)
    d = KA.allocate(dev, eltype(A), size(A)...)
    copyto!(d, A)
    return d
end

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

# Enstrophy (3D): vector vorticity ω̂ = i k×û, N̂_ω = i k×N̂ (includes vortex stretching); t = Σ_c Re{conj(ω̂_c) N̂_ω_c}
@kernel function transfer_density_enstrophy3d_kernel!(t, @Const(velocity_hat), @Const(N̂), ks1, ks2, ks3)
    I = @index(Global, Cartesian)
    kx = ks1[I[1]]; ky = ks2[I[2]]; kz = ks3[I[3]]
    ux = velocity_hat[I, 1]; uy = velocity_hat[I, 2]; uz = velocity_hat[I, 3]
    Nx = N̂[I, 1];           Ny = N̂[I, 2];           Nz = N̂[I, 3]
    ωx  = im * (ky * uz - kz * uy); ωy  = im * (kz * ux - kx * uz); ωz  = im * (kx * uy - ky * ux)
    Nωx = im * (ky * Nz - kz * Ny); Nωy = im * (kz * Nx - kx * Nz); Nωz = im * (kx * Ny - ky * Nx)
    t[I] = real(conj(ωx) * Nωx + conj(ωy) * Nωy + conj(ωz) * Nωz)
end

# Mode→shell scatter-add: T_spec[shell_idx[I]] += density[I], summed over all modes I in one pass.
# Atomic because many modes map to the same shell. O(Nᴰ) (vs the O(N_sh·Nᴰ) per-shell broadcast+sum),
# writes straight into the device T_spec vector — no host temporary, no scalar indexing.
@kernel function shell_scatter_add_kernel!(T_spec, @Const(density), @Const(shell_idx))
    I = @index(Global, Cartesian)
    n = shell_idx[I]
    if n != 0
        KA.@atomic T_spec[n] += density[I]
    end
end

# GPUBackend method of the shared `ShellBinning.shell_scatter_add!` (host scalar method lives in core):
# atomic device scatter-add over the local mode grid (`parent` unwraps a PencilArray to its dense local
# array; identity for a plain device array), result copied back into the host `T_spec` so it stays
# MPI-reducible. This is the device shell reduction for the distributed pencil flux; runs on any KA
# backend (KA.CPU for verification, CuArray/etc. on hardware) under `allowscalar(false)`.
function FIT.ShellBinning.shell_scatter_add!(T_spec, density, shell_idx, gpu_backend::GPUBackend)
    dev = gpu_backend.backend
    d   = parent(density)
    T_dev = KA.allocate(dev, eltype(T_spec), length(T_spec))
    fill!(T_dev, zero(eltype(T_spec)))
    shell_idx_dev = _to_device(dev, parent(shell_idx))
    shell_scatter_add_kernel!(dev)(T_dev, d, shell_idx_dev; ndrange = size(d))
    KA.synchronize(dev)
    copyto!(T_spec, T_dev)
    return T_spec
end

# Run the per-mode transfer-density kernel for the requested invariant.
function _launch_transfer_density!(dev, td, velocity_hat, N̂, ks, invariant, D, ns, ks_dev)
    if invariant isa KineticEnergy
        transfer_density_ke_kernel!(dev)(td, velocity_hat, N̂, D; ndrange = ns)
    elseif invariant isa Helicity
        length(ks) == 3 || throw(ArgumentError("Helicity transfer is 3D only (got nd=$(length(ks)))."))
        transfer_density_helicity_kernel!(dev)(td, velocity_hat, N̂, ks_dev[1], ks_dev[2], ks_dev[3]; ndrange = ns)
    elseif invariant isa Enstrophy
        if length(ks) == 2
            transfer_density_enstrophy_kernel!(dev)(td, velocity_hat, N̂, ks_dev[1], ks_dev[2]; ndrange = ns)
        elseif length(ks) == 3
            transfer_density_enstrophy3d_kernel!(dev)(td, velocity_hat, N̂, ks_dev[1], ks_dev[2], ks_dev[3]; ndrange = ns)
        else
            throw(ArgumentError("Enstrophy transfer is defined in 2D or 3D (got nd=$(length(ks)))."))
        end
    else
        throw(ArgumentError("GPU transfer-density kernel not implemented for $(typeof(invariant))."))
    end
    KA.synchronize(dev)
    return td
end

# ---------------------------------------------------------------------------
# Shell-to-shell transfer on a KA backend
# ---------------------------------------------------------------------------
function FIT.ShellToShellTransfer._shell_to_shell_gpu!(
    result::ShellToShellResult,
    ws::FIT.Workspaces.ShellToShellWorkspace,
    velocity_hat,
    ks,
    gpu_backend::GPUBackend,
    spectral;            # transform backend for each per-mediator nonlinear term
    dealiasing::FIT.Types.AbstractDealiasing,
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

    # Device-resident shell index (see `_to_device`) so shell masks broadcast against device fields.
    shell_idx_dev = _to_device(dev, ws.shell_idx)
    col_dev = KA.allocate(dev, FT, N_sh)   # per-mediator shell sums (device); scatter-add target

    for m in 1:N_sh
        # 1. Band-m field: û_m = velocity_hat ⊙ 1[shell == m]  (device broadcast, no scalar indexing)
        ws.û_m .= velocity_hat .* reshape(shell_idx_dev .== m, ns..., 1)

        # 2. Nonlinear term 𝒩̂_m = (u·∇)f_m (uses the spectral backend; on a GPU array this needs a
        #    GPU-FFT-capable spectral path, e.g. FFTBackend with cuFFT riding AbstractFFTs).
        FIT.NonlinearTerm.compute_nonlinear_term!(ws.nonlinear, ws.û_m, ks;
            dealiasing = dealiasing, spectral = spectral, advecting_hat = advecting_hat)

        # 3. Per-mode transfer density via the device kernel.
        _launch_transfer_density!(dev, ws.transfer_density, velocity_hat, ws.nonlinear.N̂, ks, invariant, D, ns, ks_dev)

        # 4. Column m: A[n,m] = Σ_{I ∈ shell n} density[I], via one O(Nᴰ) atomic scatter-add over modes
        #    (replaces the O(N_sh·Nᴰ) per-receiver-shell broadcast+sum), then copy the device column into
        #    the host result matrix. No scalar indexing on the device.
        fill!(col_dev, zero(FT))
        shell_scatter_add_kernel!(dev)(col_dev, ws.transfer_density, shell_idx_dev; ndrange = ns)
        KA.synchronize(dev)
        copyto!(view(result.transfer_matrix, :, m), col_dev)
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

# ---------------------------------------------------------------------------
# Spectral flux Π(K) on a KA backend
# ---------------------------------------------------------------------------
# Overrides the core `_spectral_flux_gpu!` stub. N̂ is computed upstream on the array's own kind
# (cuFFT via the FFT spectral backend on a device array); here the per-mode transfer density is
# written by a device kernel and scatter-added into the device T_spec in a single pass, then the
# cumulative flux is formed on-device with `cumsum!`. No host temporaries, no scalar indexing —
# a device-resident field stays on the GPU end to end (the point of issue #12 for the SMODE path).
function FIT.SpectralFlux._spectral_flux_gpu!(
    result::SpectralFluxResult,
    ws,
    velocity_hat,
    N̂,
    ks,
    shell_idx,
    gpu_backend::GPUBackend;
    invariant::AbstractInvariant = KineticEnergy(),
)
    dev = gpu_backend.backend
    nd  = length(ks)
    ns  = size(velocity_hat)[1:nd]
    FT  = real(eltype(velocity_hat))
    D   = size(velocity_hat, nd + 1)

    ks_dev        = ntuple(nd) do d
        a = KA.allocate(dev, FT, length(ks[d]))
        copyto!(a, collect(FT, ks[d]))
        a
    end
    shell_idx_dev = _to_device(dev, shell_idx)

    # 1. Per-mode transfer density (KE/helicity/enstrophy) via the device kernel.
    _launch_transfer_density!(dev, ws.transfer_density, velocity_hat, N̂, ks, invariant, D, ns, ks_dev)

    # 2. Scatter-add modes into shells → ws.T_spec (device), single pass.
    fill!(ws.T_spec, zero(FT))
    shell_scatter_add_kernel!(dev)(ws.T_spec, ws.transfer_density, shell_idx_dev; ndrange = ns)
    KA.synchronize(dev)

    # 3. Cumulative flux Π = +cumsum(T) via the shared host-summary finalizer (see THEORY.md §0.5).
    return FIT.SpectralFlux._finalize_spectral_flux!(result, ws)
end

# ---------------------------------------------------------------------------
# Smooth band-to-band transfer T(n,m) on a KA backend
# ---------------------------------------------------------------------------
# Overrides the core `_band_to_band_gpu!` stub. Bands are independent columns: for each band m the band
# field Wₘ⊙û is built by a device broadcast, its nonlinear term rides the (GPU-)FFT spectral backend, the
# per-mode transfer density is written by the device kernel, and T(n,m)=Σ_I Wₙ[I]·d[I] is a device
# reduction. The band-weight masks are host-built (shell_coordinate), so copied onto the device once.
function FIT.BandTransfer._band_to_band_gpu!(
    T, bws, velocity_hat, ks, gpu_backend::GPUBackend;
    dealiasing = FIT.Types.OrszagTwoThirds(),
    invariant::AbstractInvariant = KineticEnergy(),
    spectral = FIT.Types.DirectSumBackend(),
    advecting_hat = velocity_hat,
)
    dev = gpu_backend.backend
    nd = length(ks); ns = size(velocity_hat)[1:nd]; D = size(velocity_hat, nd + 1)
    FT = real(eltype(velocity_hat)); nb = length(bws.centers)
    ks_dev = ntuple(nd) do d
        a = KA.allocate(dev, FT, length(ks[d])); copyto!(a, collect(FT, ks[d])); a
    end
    W_dev = [_to_device(dev, bws.W[n]) for n in 1:nb]   # host masks → device (see _to_device)
    fill!(T, zero(FT))
    for m in 1:nb
        bws.f_m .= reshape(W_dev[m], ns..., 1) .* velocity_hat
        FIT.NonlinearTerm.compute_nonlinear_term!(bws.nlt, bws.f_m, ks;
            dealiasing = dealiasing, spectral = spectral, advecting_hat = advecting_hat)
        _launch_transfer_density!(dev, bws.d, velocity_hat, bws.nlt.N̂, ks, invariant, D, ns, ks_dev)
        KA.synchronize(dev)
        for n in 1:nb
            T[n, m] = sum(W_dev[n] .* bws.d)   # device reduction of the fused broadcast (no host temp)
        end
    end
    return T
end

# ---------------------------------------------------------------------------
# Mode-to-mode transfer S(k|p) on a KA backend
# ---------------------------------------------------------------------------
# Overrides the core `_mode_to_mode_gpu!` stub. Each giver mode p is isolated with a one-hot mask built
# by a device broadcast (a device grid of linear indices compared to p's linear index — NOT scalar
# `û_p[p,c] = …`), its nonlinear term rides the (GPU-)FFT spectral backend, and S(·|p) is written by the
# device transfer-density kernel into the p-th column of the result tensor. `net` accumulates on-device.
function FIT.ScaleToScaleTransfer._mode_to_mode_gpu!(
    result, ws, û_p, velocity_hat, ks, gpu_backend::GPUBackend;
    invariant::AbstractInvariant = KineticEnergy(),
    dealiasing = FIT.Types.OrszagTwoThirds(),
    spectral = FIT.Types.DirectSumBackend(),
    advecting_hat = velocity_hat,
)
    dev = gpu_backend.backend
    nd = length(ks); ns = size(velocity_hat)[1:nd]; M = size(velocity_hat, nd + 1); FT = real(eltype(velocity_hat))
    S = result.transfer; net = result.net_transfer
    fill!(net, zero(FT))
    ks_dev = ntuple(nd) do d
        a = KA.allocate(dev, FT, length(ks[d])); copyto!(a, collect(FT, ks[d])); a
    end
    colons = ntuple(_ -> Colon(), nd)
    lin = _to_device(dev, reshape(collect(1:prod(ns)), ns))   # linear-index grid for the one-hot mask
    linidx = LinearIndices(ns)
    for p in CartesianIndices(ns)
        plin = linidx[p]
        û_p .= velocity_hat .* reshape(lin .== plin, ns..., 1)   # isolate giver mode p (device broadcast)
        FIT.NonlinearTerm.compute_nonlinear_term!(ws, û_p, ks;
            dealiasing = dealiasing, spectral = spectral, advecting_hat = advecting_hat)
        Sp = view(S, colons..., p)
        _launch_transfer_density!(dev, Sp, velocity_hat, ws.N̂, ks, invariant, M, ns, ks_dev)
        KA.synchronize(dev)
        net .+= Sp
    end
    return result
end

# ---------------------------------------------------------------------------
# Partial (decomposition-channel) fluxes on a KA backend
# ---------------------------------------------------------------------------
# Overrides the core `_partial_fluxes_gpu!` stub. The decomposition `comps` are device-generic (broadcast
# helical/Helmholtz projection), so the whole path is on-device: for each (sp,sq) pair the nonlinear term
# rides the (GPU-)FFT backend, each receiver channel sk gets a device transfer-density kernel, and the
# per-mode density is shell-binned by the atomic scatter-add kernel (replacing the scalar `_partial_binflux`
# host loop). The n³ channel `SpectralFluxResult`s (small per-shell vectors) are the inherent host output.
function FIT.SpectralFlux._partial_fluxes_gpu!(
    channels, ws, comps, names, velocity_hat, ks, sidx, centers, Nsh, gpu_backend::GPUBackend;
    dealiasing = FIT.Types.OrszagTwoThirds(),
    spectral = FIT.Types.DirectSumBackend(),
)
    dev = gpu_backend.backend
    nd = length(ks); ns = size(velocity_hat)[1:nd]; FT = real(eltype(velocity_hat))
    D = size(velocity_hat, nd + 1)
    ks_dev = ntuple(nd) do d
        a = KA.allocate(dev, FT, length(ks[d])); copyto!(a, collect(FT, ks[d])); a
    end
    sidx_dev = _to_device(dev, sidx)
    d = similar(velocity_hat, FT, ns...)         # per-mode transfer density (device)
    Tspec = KA.allocate(dev, FT, Nsh)            # shell sums (device), reused per channel
    for sp in names, sq in names
        FIT.NonlinearTerm.compute_nonlinear_term!(ws, comps[sq], ks;
            advecting_hat = comps[sp], dealiasing = dealiasing, spectral = spectral)
        for sk in names
            _launch_transfer_density!(dev, d, comps[sk], ws.N̂, ks, KineticEnergy(), D, ns, ks_dev)
            fill!(Tspec, zero(FT))
            shell_scatter_add_kernel!(dev)(Tspec, d, sidx_dev; ndrange = ns)
            KA.synchronize(dev)
            T = Array(Tspec)                     # small per-shell vector to host (inherent output)
            channels[(sk, sp, sq)] = SpectralFluxResult(centers, T, cumsum(T))
        end
    end
    return channels
end

end # module
