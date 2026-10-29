module FlowInvariantTransferKernelAbstractionsExt

using KernelAbstractions: KernelAbstractions as KA, @kernel, @index, @Const
using FlowInvariantTransfer: FlowInvariantTransfer as FIT
using FlowTransformBindings: FlowTransformBindings as FTB
using ComputationalBackends: ComputationalBackends
using SpectralBackends: SpectralBackends
using LinearAlgebra: LinearAlgebra

# Copy a host array to a fresh device array on `dev` (a plain Array on `GPUBackend(KA.CPU())`).
# `assign_shells` builds a host `Array{Int}`; a shell mask (`shell_idx .== n`) broadcast against a
# device field needs the index grid on the device, under `allowscalar(false)` as well.
function _to_device(dev, A::AbstractArray)
    d = KA.allocate(dev, eltype(A), size(A)...)
    copyto!(d, A)
    return d
end

# `A` itself when it already lives on `dev`, a fresh device copy otherwise. A workspace built on the
# field's array kind passes its shell-index grid and band masks straight through.
_on_device(dev, A::AbstractArray) = KA.get_backend(A) === dev ? A : _to_device(dev, A)

# The scattered `to_spectral` workspace's points, copied onto the GPUBackend's device; the NUFFT plan and
# every buffer built from them then live there too.
FIT._nufft_to_device(gpu::ComputationalBackends.AbstractGPUBackend, x::AbstractArray) = _to_device(gpu.backend, x)

# An input on the GPUBackend's device: itself when it lives there, a device copy otherwise.
FIT.Types._to_execution(gpu::ComputationalBackends.GPUBackend, x::AbstractArray) = _on_device(gpu.backend, x)
FIT.Types._device_execution(x::AbstractArray) = ComputationalBackends.GPUBackend(KA.get_backend(x))

# ---------------------------------------------------------------------------
# Device kernels (KernelAbstractions ≥ 0.9 API: launch then KA.synchronize).
#
# They run on any KA backend, KA.CPU() included. The transfer-density write is per mode and the
# reductions index no scalar, so the path holds under `allowscalar(false)`.
# ---------------------------------------------------------------------------

# The per-mode transfer density of `invariant` at mode `I`, term for term the host
# `Invariants.transfer_density!`, with `kd` each axis's derivative wavenumbers on the device.
#   kinetic energy, passive scalar: Σ_c Re{ conj(û_c) N̂_c }
#   helicity (3D):                  Σ_c Re{ conj(ω̂_c) N̂_c },  ω̂ = i k × û
#   enstrophy:                      Re{ conj(ω̂)·N̂_ω },  ω̂ = i k × û,  N̂_ω = i k × N̂ (scalar in 2D)
@inline function _dev_density(::Union{FIT.Types.KineticEnergy, FIT.Types.PassiveScalar}, û, N̂, kd, I, D)
    s = real(conj(û[I, 1]) * N̂[I, 1])
    for c in 2:D
        s += real(conj(û[I, c]) * N̂[I, c])
    end
    return s
end

@inline function _dev_density(::FIT.Types.Helicity, û, N̂, kd::NTuple{3}, I, D)
    kx = kd[1][I[1]]; ky = kd[2][I[2]]; kz = kd[3][I[3]]
    ux = û[I, 1]; uy = û[I, 2]; uz = û[I, 3]
    ωx = im * (ky * uz - kz * uy)
    ωy = im * (kz * ux - kx * uz)
    ωz = im * (kx * uy - ky * ux)
    return real(conj(ωx) * N̂[I, 1] + conj(ωy) * N̂[I, 2] + conj(ωz) * N̂[I, 3])
end

@inline function _dev_density(::FIT.Types.Enstrophy, û, N̂, kd::NTuple{2}, I, D)
    kx = kd[1][I[1]]; ky = kd[2][I[2]]
    ω̂   = im * (kx * û[I, 2] - ky * û[I, 1])
    N̂_ω = im * (kx * N̂[I, 2] - ky * N̂[I, 1])
    return real(conj(ω̂) * N̂_ω)
end

@inline function _dev_density(::FIT.Types.Enstrophy, û, N̂, kd::NTuple{3}, I, D)
    kx = kd[1][I[1]]; ky = kd[2][I[2]]; kz = kd[3][I[3]]
    ux = û[I, 1]; uy = û[I, 2]; uz = û[I, 3]
    Nx = N̂[I, 1]; Ny = N̂[I, 2]; Nz = N̂[I, 3]
    ωx  = im * (ky * uz - kz * uy); ωy  = im * (kz * ux - kx * uz); ωz  = im * (kx * uy - ky * ux)
    Nωx = im * (ky * Nz - kz * Ny); Nωy = im * (kz * Nx - kx * Nz); Nωz = im * (kx * Ny - ky * Nx)
    return real(conj(ωx) * Nωx + conj(ωy) * Nωy + conj(ωz) * Nωz)
end

# The invariants and dimensions `_dev_density` has a method for.
function _check_density(invariant::FIT.Types.AbstractInvariant, nd::Int)
    if invariant isa Union{FIT.Types.KineticEnergy, FIT.Types.PassiveScalar}
        return nothing
    elseif invariant isa FIT.Types.Helicity
        nd == 3 || throw(ArgumentError("Helicity transfer is 3D only (got nd=$nd)."))
    elseif invariant isa FIT.Types.Enstrophy
        nd in (2, 3) || throw(ArgumentError("Enstrophy transfer is defined in 2D or 3D (got nd=$nd)."))
    else
        throw(ArgumentError("GPU transfer-density kernel not implemented for $(typeof(invariant))."))
    end
    return nothing
end

# t[I] = the per-mode density.
@kernel function transfer_density_kernel!(t, @Const(û), @Const(N̂), kd, invariant, D)
    I = @index(Global, Cartesian)
    t[I] = _dev_density(invariant, û, N̂, kd, I, D)
end

# T_spec[shell_idx[I]] += w1[I[1]]·t(I) over every mode in one pass, `t` the per-mode density and `w1`
# the Hermitian weight along axis 1 (`FlowTransformBindings.fold_weight`); no density array is formed.
# Atomic because many modes map to the same shell.
@kernel function shell_transfer_kernel!(T_spec, @Const(û), @Const(N̂), @Const(shell_idx), @Const(w1), kd,
                                        invariant, D)
    I = @index(Global, Cartesian)
    n = shell_idx[I]
    if n != 0
        KA.@atomic T_spec[n] += w1[I[1]] * _dev_density(invariant, û, N̂, kd, I, D)
    end
end

# Slot isolation: û_p holds `velocity_hat` at the single mode `p` and zero elsewhere, in one pass over
# the mode grid.
@kernel function one_hot_kernel!(out, @Const(field), p, C)
    I = @index(Global, Cartesian)
    keep = I == p
    for c in 1:C
        out[I, c] = keep ? field[I, c] : zero(eltype(out))
    end
end

# Mode→shell scatter-add of a density that already carries the Hermitian weight:
# T_spec[shell_idx[I]] += density[I] over every mode in one pass.
@kernel function shell_scatter_add_kernel!(T_spec, @Const(density), @Const(shell_idx))
    I = @index(Global, Cartesian)
    n = shell_idx[I]
    if n != 0
        KA.@atomic T_spec[n] += density[I]
    end
end

@kernel function compressible_shell_kernel!(out, @Const(density), @Const(shell_idx), @Const(fold))
    I = @index(Global, Cartesian)
    n = shell_idx[I]
    if n != 0
        KA.@atomic out[n] += fold[I[1]] * density[I]
    end
end

function FIT.Compressible._bin_device!(out::AbstractVector, density, shell_idx, fold)
    dev = KA.get_backend(density)
    fill!(out, zero(eltype(out)))
    compressible_shell_kernel!(dev)(out, density, shell_idx, fold; ndrange = size(density))
    KA.synchronize(dev)
    return Array(out)
end

# `GPUBackend` method of `ShellBinning.shell_scatter_add!`, the device shell reduction of the
# distributed pencil flux: an atomic scatter-add over the local mode grid (`parent` unwraps a PencilArray
# to its dense local array), copied back into the host `T_spec` so it stays MPI-reducible.
function FIT.ShellBinning.shell_scatter_add!(T_spec, density, shell_idx,
                                            gpu_backend::ComputationalBackends.GPUBackend, T_dev)
    dev = gpu_backend.backend
    d   = parent(density)
    fill!(T_dev, zero(eltype(T_spec)))
    shell_idx_dev = _on_device(dev, parent(shell_idx))
    shell_scatter_add_kernel!(dev)(T_dev, d, shell_idx_dev; ndrange = size(d))
    KA.synchronize(dev)
    copyto!(T_spec, T_dev)
    return T_spec
end

# The per-mode transfer density of `invariant` into `td`. A passive scalar is the one-component case of
# the kinetic-energy contraction, the scalar passed as `velocity_hat`.
function _launch_transfer_density!(dev, td, velocity_hat, N̂, ks, invariant, D, ns, ks_dev)
    _check_density(invariant, length(ks))
    transfer_density_kernel!(dev)(td, velocity_hat, N̂, ks_dev, invariant, D; ndrange = ns)
    return td
end

# The weighted shell sums of the per-mode transfer density into the device vector `T_spec`, zeroed
# first.
function _launch_shell_transfer!(dev, T_spec, velocity_hat, N̂, shell_idx_dev, w1, ks, invariant, D, ns, ks_dev)
    _check_density(invariant, length(ks))
    fill!(T_spec, zero(eltype(T_spec)))
    shell_transfer_kernel!(dev)(T_spec, velocity_hat, N̂, shell_idx_dev, w1, ks_dev, invariant, D; ndrange = ns)
    return T_spec
end

# ---------------------------------------------------------------------------
# Shell-to-shell transfer on a KA backend
# ---------------------------------------------------------------------------
function FIT.ShellToShellTransfer._shell_to_shell_gpu!(
    result::FIT.Types.ShellToShellResult,
    ws::FIT.Workspaces.ShellToShellWorkspace,
    velocity_hat,
    ks,
    gpu_backend::ComputationalBackends.GPUBackend,
    spectral;            # transform backend for each per-mediator nonlinear term
    dealiasing::FIT.Types.AbstractDealiasing,
    invariant::FIT.Types.AbstractInvariant = FIT.Types.KineticEnergy(),
    advecting_hat = velocity_hat,
)
    dev  = gpu_backend.backend
    N_sh = size(result.transfer_matrix, 1)
    FT   = real(eltype(velocity_hat))
    nd   = length(ks)
    ns   = Base.front(size(velocity_hat))
    D    = size(velocity_hat, nd + 1)      # components of the binned/carried primary field

    fill!(result.transfer_matrix, zero(FT))
    fill!(result.net_transfer, zero(FT))

    ks_dev = ws.nonlinear.operators.kd
    w1 = ws.nonlinear.operators.fold

    # Device-resident shell index (see `_to_device`) so shell masks broadcast against device fields.
    shell_idx_dev = _on_device(dev, ws.shell_idx)
    col_dev = ws.net_transfer
    FIT.NonlinearTerm.synthesize_advecting!(ws.nonlinear, advecting_hat, ks; dealiasing = dealiasing, transform = spectral)

    sidx_c = reshape(shell_idx_dev, ns..., 1)   # broadcasts against the component axis
    for m in 1:N_sh
        # 1. Band-m field: û_m = velocity_hat ⊙ 1[shell == m]  (device broadcast, no scalar indexing)
        ws.û_m .= velocity_hat .* (sidx_c .== m)

        # 2. 𝒩̂_m = (u·∇)f_m by the advecting velocity synthesised above (on a GPU array the spectral
        #    backend is FFTSpectralBackend, cuFFT through AbstractFFTs).
        FIT.NonlinearTerm.compute_advection!(ws.nonlinear, ws.û_m, ks; dealiasing = dealiasing, transform = spectral)

        # 3. Column m: A[n,m] = Σ_{I ∈ shell n} w(I)·density(I), one O(Nᴰ) pass over the modes, then the
        #    device column into the host result matrix.
        _launch_shell_transfer!(dev, col_dev, velocity_hat, ws.nonlinear.N̂, shell_idx_dev, w1, ks, invariant,
                                D, ns, ks_dev)
        KA.synchronize(dev)
        copyto!(view(result.transfer_matrix, :, m), col_dev)
    end

    # Net transfer Σ_m T(n,m) on the host matrix.
    for n in 1:N_sh
        s = zero(FT)
        for m in 1:N_sh
            s += result.transfer_matrix[n, m]
        end
        result.net_transfer[n] = s
    end

    return result
end

# ---------------------------------------------------------------------------
# Spectral flux Π(K) on a KA backend
# ---------------------------------------------------------------------------
# Overrides the core `_spectral_flux_gpu!` stub. N̂ is computed upstream on the array's own kind
# (cuFFT via the FFT spectral backend on a device array); here one device kernel forms each mode's
# transfer density and adds it to its shell in the device T_spec. The `N_sh`-long shell spectrum is
# then copied to the host result, where `_finalize_spectral_flux!` forms the cumulative flux. No scalar
# indexing on the device.
function FIT.SpectralFlux._spectral_flux_gpu!(
    result::FIT.Types.SpectralFluxResult,
    ws,
    velocity_hat,
    N̂,
    ks,
    shell_idx,
    gpu_backend::ComputationalBackends.GPUBackend;
    invariant::FIT.Types.AbstractInvariant = FIT.Types.KineticEnergy(),
)
    dev = gpu_backend.backend
    nd  = length(ks)
    ns  = Base.front(size(velocity_hat))
    FT  = real(eltype(velocity_hat))
    D   = size(velocity_hat, nd + 1)
    _launch_shell_transfer!(dev, ws.T_spec, velocity_hat, N̂, _on_device(dev, shell_idx),
                            ws.nonlinear.operators.fold, ks, invariant, D, ns,
                            ws.nonlinear.operators.kd)
    KA.synchronize(dev)
    return FIT.SpectralFlux._finalize_spectral_flux!(result, ws)
end

# ---------------------------------------------------------------------------
# Smooth band-to-band transfer T(n,m) on a KA backend
# ---------------------------------------------------------------------------
# Overrides the core `_band_to_band_gpu!` stub. Bands are independent columns: for each band m the band
# field wₘ⊙û is built on the device from the weight table and each mode's shell index, its nonlinear term
# rides the (GPU-)FFT spectral backend, the per-mode transfer density is written by the device kernel and
# summed per shell coordinate with its Hermitian weight, and the table applied to those sums on the
# host gives T(·,m).
function FIT.BandTransfer._band_to_band_gpu!(
    T, bws, velocity_hat, ks, gpu_backend::ComputationalBackends.GPUBackend;
    dealiasing = FIT.Types.OrszagTwoThirds(),
    invariant::FIT.Types.AbstractInvariant = FIT.Types.KineticEnergy(),
    transform = SpectralBackends.DirectSumSpectralBackend(),
    advecting_hat = velocity_hat,
)
    dev = gpu_backend.backend
    nd = length(ks); ns = Base.front(size(velocity_hat)); D = size(velocity_hat, nd + 1)
    FT = real(eltype(velocity_hat)); nb = length(bws.centers)
    ks_dev = bws.nlt.operators.kd
    W = bws.W
    idx_dev = _on_device(dev, W.idx)
    table_dev = _on_device(dev, W.table)
    w1_dev = bws.nlt.operators.fold
    Dq_dev = bws.Dq
    N = prod(ns)
    FIT.NonlinearTerm.synthesize_advecting!(bws.nlt, advecting_hat, ks; dealiasing = dealiasing, transform = transform)
    for m in 1:nb
        _band_field_kernel!(dev)(bws.f_m, table_dev, idx_dev, velocity_hat, m, N, D; ndrange = N)
        FIT.NonlinearTerm.compute_advection!(bws.nlt, bws.f_m, ks; dealiasing = dealiasing, transform = transform)
        _launch_transfer_density!(dev, bws.d, velocity_hat, bws.nlt.N̂, ks, invariant, D, ns, ks_dev)
        fill!(Dq_dev, zero(FT))
        _shell_bin_kernel!(dev)(Dq_dev, idx_dev, w1_dev, bws.d, CartesianIndices(ns); ndrange = N)
        LinearAlgebra.mul!(bws.column, table_dev, Dq_dev)
        KA.synchronize(dev)
        copyto!(view(T, :, m), bws.column)
    end
    return T
end

# `f[·, c] = table[m, idx] ⊙ v[·, c]` over the `D` components.
@kernel function _band_field_kernel!(f, @Const(table), @Const(idx), @Const(v), m::Int, N::Int, D::Int)
    t = @index(Global)
    @inbounds w = table[m, idx[t]]
    @inbounds for c in 1:D
        f[t + (c - 1) * N] = w * v[t + (c - 1) * N]
    end
end

# `Dq[idx[I]] += fold(I) d[I]`, the fold weight read along axis 1.
@kernel function _shell_bin_kernel!(Dq, @Const(idx), @Const(w1), @Const(d), cart)
    t = @index(Global)
    @inbounds begin
        I = cart[t]
        KA.@atomic Dq[idx[t]] += w1[ifelse(length(w1) == 1, 1, I[1])] * d[t]
    end
end

# ---------------------------------------------------------------------------
# Mode-to-mode transfer S(k|p) on a KA backend
# ---------------------------------------------------------------------------
# Overrides the core `_mode_to_mode_gpu!` stub. The advecting velocity is synthesised once; each slot p
# is isolated by `one_hot_kernel!` (a device pass comparing the mode index against p), advected through
# the (GPU-)FFT spectral backend, and S(·|p) is written by the device transfer-density kernel into the
# p-th column of the result tensor. A slot the dealiasing rule discards has a zero column. `net`
# accumulates on-device.
function FIT.ModeToModeTransfer._mode_to_mode_gpu!(
    result, ws, û_p, velocity_hat, ks, gpu_backend::ComputationalBackends.GPUBackend;
    invariant::FIT.Types.AbstractInvariant = FIT.Types.KineticEnergy(),
    dealiasing = FIT.Types.OrszagTwoThirds(),
    transform = SpectralBackends.DirectSumSpectralBackend(),
    advecting_hat = velocity_hat,
)
    dev = gpu_backend.backend
    nd = length(ks); ns = Base.front(size(velocity_hat)); M = size(velocity_hat, nd + 1); FT = real(eltype(velocity_hat))
    S = result.transfer; net = result.net_transfer
    fill!(net, zero(FT))
    ks_dev = ws.operators.kd
    colons = ntuple(_ -> Colon(), Val(length(ks)))
    discard = dealiasing isa FIT.Types.OrszagTwoThirds
    FIT.NonlinearTerm.synthesize_advecting!(ws, advecting_hat, ks; dealiasing = dealiasing, transform = transform)
    for p in CartesianIndices(ns)
        Sp = view(S, colons..., p)
        if discard && FIT.NonlinearTerm._is_dealiased(ks, p)
            fill!(Sp, zero(FT))
            continue
        end
        one_hot_kernel!(dev)(û_p, velocity_hat, p, M; ndrange = ns)   # isolate slot p
        FIT.NonlinearTerm.compute_advection!(ws, û_p, ks; dealiasing = dealiasing, transform = transform)
        _launch_transfer_density!(dev, Sp, velocity_hat, ws.N̂, ks, invariant, M, ns, ks_dev)
        net .+= Sp
    end
    return result
end

# ---------------------------------------------------------------------------
# Partial (decomposition-channel) fluxes on a KA backend
# ---------------------------------------------------------------------------
# Overrides the core `_partial_fluxes_gpu!` stub. The decomposition `comps` are device-generic (broadcast
# helical/Helmholtz projection), so the whole path is on-device: for each (sp,sq) pair the nonlinear term
# rides the (GPU-)FFT backend, and each channel sk is one device pass forming the per-mode density and
# adding it to its shell. The n³ channel `SpectralFluxResult`s are small host vectors.
function FIT.SpectralFlux._partial_fluxes_gpu!(
    channels, ws, comps, names, velocity_hat, ks, shells, gpu_backend::ComputationalBackends.GPUBackend;
    dealiasing::FIT.Types.AbstractDealiasing,
    transform::SpectralBackends.AbstractSpectralBackend,
    decomposition::FIT.Types.AbstractFieldDecomposition,
)
    dev = gpu_backend.backend
    nd = length(ks); ns = Base.front(size(velocity_hat))
    D = size(velocity_hat, nd + 1)
    ks_dev = ws.operators.kd
    w1 = ws.operators.fold
    sidx_dev = shells.sidx
    Tspec = shells.sums
    for sp in names, sq in names
        FIT.SpectralFlux._channel_term!(ws, decomposition, comps[sp], comps[sq], ks;
                                       dealiasing = dealiasing, transform = transform)
        for sk in names
            _launch_shell_transfer!(dev, Tspec, comps[sk], ws.N̂, sidx_dev, w1, ks, FIT.Types.KineticEnergy(),
                                    D, ns, ks_dev)
            KA.synchronize(dev)
            T = Array(Tspec)                     # small per-shell vector to the host
            channels[(sk, sp, sq)] = FIT.Types.SpectralFluxResult(shells.centers, shells.edges, T, cumsum(T))
        end
    end
    return channels
end

end # module
