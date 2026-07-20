module FlowInvariantTransferPencilFFTsExt

using MPI: MPI
using PencilFFTs: PencilFFTs, PencilFFTPlan, Transforms, allocate_input, allocate_output
using PencilArrays: PencilArrays, localgrid
using LinearAlgebra: mul!, ldiv!
using FlowInvariantTransfer: FlowInvariantTransfer as FIT
using FlowInvariantTransfer.Types: AbstractShellBinning, AbstractInvariant, KineticEnergy, Helicity, Enstrophy,
                                   AbstractDealiasing, OrszagTwoThirds, NoDealiasing,
                                   AbstractShellGeometry, ShellMagnitude, IsotropicShells
using FlowInvariantTransfer.ShellBinning: shell_edges, shell_centers, assign_shells
using FlowInvariantTransfer.Backends: AbstractExecutionBackend, SerialBackend, GPUBackend, local_backend

# ---------------------------------------------------------------------------
# Pencil axis: split ONE grid across ranks; transpose-based distributed FFT.
#
# The pseudospectral nonlinear term N̂ = FFT[(u·∇)u]/Np is built with PencilFFTs
# (mul! = unnormalised forward fft, ldiv! = normalised inverse ifft, matching the FFTW
# extension's fft/ifft pair). The package coefficient convention is û = fft(u)/Np, so
# u_phys = real(ifft(û)). Each rank owns a pencil of both the physical (input) and the
# spectral (output) grid; products are pointwise-local, gradients/dealiasing use the
# rank's local Fourier wavenumbers (via localgrid, permutation-aware), and the per-shell
# KE transfer spectrum is MPI.Allreduce'd to a global result identical on every rank and
# equal to the serial calculate_spectral_flux on the same field.
#
#   plan = build_pencil_plan(ns, comm)               # convenience below
#   u    = ntuple(_ -> allocate_input(plan), D)      # fill each rank's LOCAL portion
#   res  = pencil_spectral_flux(u, plan, ks; binning = LinearBinning(dk))
# ---------------------------------------------------------------------------

# Implements the FIT.build_pencil_plan stub (docstring lives on the core stub).
function FIT.build_pencil_plan(ns::NTuple{nd,Int}, comm = MPI.COMM_WORLD; T = Float64) where {nd}
    proc_dims  = Tuple(Int.(MPI.Dims_create(MPI.Comm_size(comm), ntuple(_ -> 0, nd - 1))))
    transforms = ntuple(_ -> Transforms.FFT(), nd)
    return PencilFFTPlan(ns, transforms, proc_dims, comm, T)
end

# Per-mode transfer density for the pencil (distributed) layout, IN PLACE into `d` (real spectral),
# using `ω` (nd complex-spectral scratch buffers) for the helicity/enstrophy vorticity. Same KE /
# helicity / enstrophy formulas as the serial `transfer_density!`, as broadcasts over each rank's LOCAL
# spectral pencil arrays (û, N̂, local wavenumber components KC). 0-alloc (writes into caller buffers).
function _pencil_transfer_density!(d, ::KineticEnergy, û, Nhat, KC, nd, ω)
    fill!(d, zero(eltype(d)))
    for i in 1:nd
        d .+= real.(conj.(û[i]) .* Nhat[i])
    end
    return d
end
function _pencil_transfer_density!(d, ::Helicity, û, Nhat, KC, nd, ω)
    nd == 3 || throw(ArgumentError("Helicity transfer is defined in 3D only (got nd=$nd)."))
    @. ω[1] = im * (KC[2] * û[3] - KC[3] * û[2])   # ω̂ = i k × û
    @. ω[2] = im * (KC[3] * û[1] - KC[1] * û[3])
    @. ω[3] = im * (KC[1] * û[2] - KC[2] * û[1])
    @. d = real(conj(ω[1]) * Nhat[1] + conj(ω[2]) * Nhat[2] + conj(ω[3]) * Nhat[3])
    return d
end
function _pencil_transfer_density!(d, ::Enstrophy, û, Nhat, KC, nd, ω)
    if nd == 2
        @. ω[1] = im * (KC[1] * û[2] - KC[2] * û[1])          # scalar vorticity ω̂
        @. ω[2] = im * (KC[1] * Nhat[2] - KC[2] * Nhat[1])   # N̂_ω
        @. d = real(conj(ω[1]) * ω[2])
    elseif nd == 3
        # Vector vorticity ω̂ = i k×û (into the 3 spectral scratch buffers); N̂_ω = i k×N̂ inline in the
        # contraction t = Σ_c Re{conj(ω̂_c) N̂_ω_c} (non-conservative 3D enstrophy w/ vortex stretching).
        @. ω[1] = im * (KC[2] * û[3] - KC[3] * û[2])
        @. ω[2] = im * (KC[3] * û[1] - KC[1] * û[3])
        @. ω[3] = im * (KC[1] * û[2] - KC[2] * û[1])
        @. d = real(conj(ω[1]) * (im * (KC[2] * Nhat[3] - KC[3] * Nhat[2]))) +
               real(conj(ω[2]) * (im * (KC[3] * Nhat[1] - KC[1] * Nhat[3]))) +
               real(conj(ω[3]) * (im * (KC[1] * Nhat[2] - KC[2] * Nhat[1])))
    else
        throw(ArgumentError("Enstrophy transfer is defined in 2D or 3D (got nd=$nd)."))
    end
    return d
end

# Reusable pencil workspace: the (geometry/dealiasing/binning-fixed) wavenumber grids + shell structure
# and every per-snapshot scratch field, so `pencil_spectral_flux!` reuses them (0 alloc beyond the small
# per-shell result vectors). Spectral-layout buffers via allocate_output; physical-layout via allocate_input.
struct PencilWorkspace{PL, CM, KCT, KM, KP, SI, CE, UST, SP, UPT, IB, WT, R, EX}
    plan::PL; comm::CM
    KC::KCT; KMAG::KM; KEEP::KP; shell_idx::SI; centers::CE
    û::UST; Nhat::UST; ω::UST; spec::SP; dloc::KM
    uphys::UPT; N_i::IB; g::IB; ph::IB
    Tloc::WT; Np::R; do_trunc::Bool; nd::Int; inner::EX
end

function FIT.PencilWorkspace(plan, ks, comm = MPI.COMM_WORLD;
        binning::AbstractShellBinning,
        dealiasing::AbstractDealiasing = OrszagTwoThirds(),
        geometry::AbstractShellGeometry = IsotropicShells(),
        execution::AbstractExecutionBackend = SerialBackend())
    geometry isa ShellMagnitude || throw(ArgumentError(
        "pencil workspace supports ShellMagnitude shell geometries (IsotropicShells / " *
        "PerpendicularShells / ParallelShells); got $(typeof(geometry))."))
    nd = length(ks)
    ns = ntuple(d -> length(ks[d]), nd)
    out_proto = allocate_output(plan)
    FT = real(eltype(out_proto))
    Np = FT(prod(ns))
    do_trunc = !(dealiasing isa NoDealiasing)

    # Local Fourier wavenumbers in the spectral (output) layout — permutation-aware.
    gf = localgrid(out_proto, ks)
    KC = ntuple(nd) do d
        a = similar(out_proto, FT); a .= gf[d]; a               # k_d value at every local spectral point
    end
    # Shell coordinate = √(Σ_{d∈dims} k_d²) — |k| (isotropic) or k_⊥/k_∥ for an anisotropic projection.
    gdims = geometry.dims === nothing ? ntuple(identity, nd) : geometry.dims
    KMAG = similar(out_proto, FT); fill!(KMAG, zero(FT))
    for d in gdims
        KMAG .+= KC[d] .^ 2
    end
    KMAG .= sqrt.(KMAG)
    # Orszag 2/3 keep-mask: discard if the folded integer index |k_d| ≥ N_d ÷ 3 (matches serial exactly).
    dk = ntuple(d -> abs(ks[d][2] - ks[d][1]), nd)
    KEEP = similar(out_proto, Bool); fill!(KEEP, true)
    if do_trunc
        for d in 1:nd
            cutoff = ns[d] ÷ 3
            KEEP .&= (round.(Int, abs.(KC[d]) ./ FT(dk[d])) .< cutoff)
        end
    end
    # Local execution backend (Serial default; GPUBackend for a per-rank device pencil / multi-GPU,
    # possibly wrapped in MPIBackend, unwrapped here). Drives the per-rank shell reduction below.
    inner = local_backend(execution)
    # Shell edges/centers from the GLOBAL max |k| (Allreduce → identical on all ranks); local shell index.
    kmax = MPI.Allreduce(maximum(KMAG), max, comm)
    edges     = shell_edges(binning, kmax)
    centers   = collect(shell_centers(binning, kmax))
    # `assign_shells` is a host scalar-indexed builder; on a device backend build it once from a host copy
    # of the local |k| grid (the device reduction moves this index grid back on-device per snapshot).
    shell_idx = inner isa GPUBackend ? assign_shells(Array(parent(KMAG)), edges) : assign_shells(KMAG, edges)

    û     = ntuple(_ -> allocate_output(plan), nd)   # spectral velocity coeffs
    Nhat  = ntuple(_ -> allocate_output(plan), nd)   # nonlinear-term spectral
    ω     = ntuple(_ -> allocate_output(plan), nd)   # helicity/enstrophy vorticity scratch
    spec  = allocate_output(plan)                    # spectral scratch (KEEP·û, i k·û)
    dloc  = similar(out_proto, FT)                   # transfer density (real spectral)
    uphys = ntuple(_ -> similar(allocate_input(plan), FT), nd)   # physical velocity (real)
    N_i   = allocate_input(plan)                     # nonlinear accumulator (physical)
    g     = allocate_input(plan)                     # gradient (physical)
    ph    = allocate_input(plan)                     # ifft output (physical)
    Tloc  = zeros(FT, length(centers))
    return PencilWorkspace(plan, comm, KC, KMAG, KEEP, shell_idx, centers,
                           û, Nhat, ω, spec, dloc, uphys, N_i, g, ph, Tloc, Np, do_trunc, nd, inner)
end

# In-place distributed pencil spectral flux — reuses `ws`; 0 alloc beyond the small per-shell vectors.
function FIT.pencil_spectral_flux!(ws::PencilWorkspace, u_phys::NTuple{D};
                                   invariant::AbstractInvariant = KineticEnergy()) where {D}
    D == ws.nd || throw(ArgumentError("got $D velocity components for an $(ws.nd)-D pencil workspace."))
    plan = ws.plan; nd = ws.nd; Np = ws.Np; do_trunc = ws.do_trunc
    û = ws.û; Nhat = ws.Nhat; KC = ws.KC; KEEP = ws.KEEP; spec = ws.spec; uphys = ws.uphys

    for c in 1:nd
        mul!(û[c], plan, u_phys[c]); û[c] ./= Np                 # û_c = fft(u_c)/Np
    end
    # Physical advecting velocity u_j = Np·ifft(û_j) (dealiased). ×Np: û carries the 1/Nᵈ; ldiv! is ifft.
    for j in 1:nd
        do_trunc ? (spec .= KEEP .* û[j]) : copyto!(spec, û[j])
        ldiv!(ws.ph, plan, spec)
        uphys[j] .= real.(ws.ph) .* Np
    end
    # N̂_i = fft( Σ_j u_j ∂_j u_i )/Np, dealiased
    for i in 1:nd
        fill!(ws.N_i, zero(eltype(ws.N_i)))
        for j in 1:nd
            do_trunc ? (spec .= (im .* KC[j]) .* KEEP .* û[i]) : (spec .= (im .* KC[j]) .* û[i])
            ldiv!(ws.g, plan, spec)
            ws.N_i .+= uphys[j] .* (real.(ws.g) .* Np)           # physical u_j ∂_j u_i
        end
        mul!(Nhat[i], plan, ws.N_i); Nhat[i] ./= Np
        do_trunc && (Nhat[i] .*= KEEP)
    end
    # Per-mode transfer density (KE/helicity/enstrophy) → local shell sums → global Allreduce.
    _pencil_transfer_density!(ws.dloc, invariant, û, Nhat, KC, nd, ws.ω)
    # Local shell reduction dispatched on the per-rank backend: scalar host loop (Serial/Threaded) or an
    # atomic device scatter-add (GPUBackend, device-resident) — writes into the host `Tloc`, then reduce.
    FIT.ShellBinning.shell_scatter_add!(ws.Tloc, ws.dloc, ws.shell_idx, ws.inner)
    Tglob = MPI.Allreduce(ws.Tloc, +, ws.comm)     # global per-shell transfer
    flux  = cumsum(Tglob)                           # Π(K) = Σ_{k≤K} T(k)
    return (centers = ws.centers, transfer_spectrum = Tglob, flux = flux)
end

# Allocating convenience: build a one-shot workspace and delegate to the in-place form.
function FIT.pencil_spectral_flux(
    u_phys::NTuple{D, <:PencilArrays.PencilArray},
    plan,
    ks;
    comm = MPI.COMM_WORLD,
    binning::AbstractShellBinning,
    dealiasing::AbstractDealiasing = OrszagTwoThirds(),
    invariant::AbstractInvariant = KineticEnergy(),
    geometry::AbstractShellGeometry = IsotropicShells(),
    execution::AbstractExecutionBackend = SerialBackend(),
) where {D}
    ws = FIT.PencilWorkspace(plan, ks, comm; binning = binning, dealiasing = dealiasing,
                             geometry = geometry, execution = execution)
    return FIT.pencil_spectral_flux!(ws, u_phys; invariant = invariant)
end

end # module FlowInvariantTransferPencilFFTsExt
