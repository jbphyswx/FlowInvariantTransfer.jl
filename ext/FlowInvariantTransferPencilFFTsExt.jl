module FlowInvariantTransferPencilFFTsExt

using MPI: MPI
using PencilFFTs: PencilFFTs
using PencilArrays: PencilArrays
using LinearAlgebra: LinearAlgebra
using FlowInvariantTransfer: FlowInvariantTransfer as FIT
using ComputationalBackends: ComputationalBackends

# ---------------------------------------------------------------------------
# Pencil axis: split ONE grid across ranks; transpose-based distributed FFT.
#
# The pseudospectral nonlinear term N̂ = FFT[(u·∇)u]/Np is built with PencilFFTs
# (LinearAlgebra.mul! = unnormalised forward fft, LinearAlgebra.ldiv! = normalised inverse ifft, matching the FFTW
# extension's fft/ifft pair). The package coefficient convention is û = fft(u)/Np, so
# u_phys = real(ifft(û)). Each rank owns a pencil of both the physical (input) and the
# spectral (output) grid; products are pointwise-local, gradients/dealiasing use the
# rank's local Fourier wavenumbers (via PencilArrays.localgrid, permutation-aware), and the per-shell
# KE transfer spectrum is MPI.Allreduce'd to a global result identical on every rank and
# equal to the serial calculate_spectral_flux on the same field.
#
#   plan = build_pencil_plan(ns, comm)               # convenience below
#   u    = ntuple(_ -> PencilFFTs.allocate_input(plan), D)      # fill each rank's LOCAL portion
#   res  = pencil_spectral_flux(u, plan, ks; binning = LinearBinning(dk))
# ---------------------------------------------------------------------------

# Implements the FIT.build_pencil_plan stub (docstring lives on the core stub).
#
# The velocity is real, so axis 1 is a real-to-complex transform and the distributed spectral grid holds
# the non-redundant half `k₁ ≥ 0`; `PencilWorkspace` takes the matching half wavenumber tuple
# (`Utils.wavenumber_grid(ns, Ls; real = true)`) and carries the Hermitian weight into the shell sums.
function FIT.build_pencil_plan(ns::NTuple{nd,Int}, comm = MPI.COMM_WORLD; T = Float64) where {nd}
    proc_dims  = Tuple(Int.(MPI.Dims_create(MPI.Comm_size(comm), ntuple(_ -> 0, nd - 1))))
    transforms = (PencilFFTs.Transforms.RFFT(), ntuple(_ -> PencilFFTs.Transforms.FFT(), nd - 1)...)
    return PencilFFTs.PencilFFTPlan(ns, transforms, proc_dims, comm, T)
end

# Per-mode transfer density for the pencil (distributed) layout, IN PLACE into `d` (real spectral),
# using `ω` (nd complex-spectral scratch buffers) for the helicity/enstrophy vorticity. Same KE /
# helicity / enstrophy formulas as the serial `transfer_density!`, as broadcasts over each rank's LOCAL
# spectral pencil arrays (û, N̂, local wavenumber components KC). 0-alloc (writes into caller buffers).
function _pencil_transfer_density!(d, ::FIT.Types.KineticEnergy, û, Nhat, KC, nd, ω)
    fill!(d, zero(eltype(d)))
    for i in 1:nd
        d .+= real.(conj.(û[i]) .* Nhat[i])
    end
    return d
end
function _pencil_transfer_density!(d, ::FIT.Types.Helicity, û, Nhat, KC, nd, ω)
    nd == 3 || throw(ArgumentError("FIT.Types.Helicity transfer is defined in 3D only (got nd=$nd)."))
    @. ω[1] = im * (KC[2] * û[3] - KC[3] * û[2])   # ω̂ = i k × û
    @. ω[2] = im * (KC[3] * û[1] - KC[1] * û[3])
    @. ω[3] = im * (KC[1] * û[2] - KC[2] * û[1])
    @. d = real(conj(ω[1]) * Nhat[1] + conj(ω[2]) * Nhat[2] + conj(ω[3]) * Nhat[3])
    return d
end
function _pencil_transfer_density!(d, ::FIT.Types.Enstrophy, û, Nhat, KC, nd, ω)
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
        throw(ArgumentError("FIT.Types.Enstrophy transfer is defined in 2D or 3D (got nd=$nd)."))
    end
    return d
end

# Reusable pencil workspace: the (geometry/dealiasing/binning-fixed) wavenumber grids + shell structure
# and every per-snapshot scratch field, so `pencil_spectral_flux!` reuses them (0 alloc beyond the small
# per-shell result vectors). Spectral-layout buffers via PencilFFTs.allocate_output; physical-layout via PencilFFTs.allocate_input.
struct PencilWorkspace{PL, CM, KCT, KM, KP, SI, CE, UST, OT, SP, UPT, IB, WT, R, EX, IV}
    plan::PL; comm::CM
    KC::KCT; KD::KCT; KMAG::KM; KEEP::KP; W::KM; shell_idx::SI; centers::CE
    û::UST; Nhat::UST; ω::OT; spec::SP; dloc::KM
    uphys::UPT; N_i::IB; g::IB; ph::IB
    Tloc::WT; Np::R; do_trunc::Bool; nd::Int; inner::EX; invariant::IV
end

function FIT.PencilWorkspace(plan, ks, comm = MPI.COMM_WORLD;
        binning::FIT.Types.AbstractShellBinning,
        dealiasing::FIT.Types.AbstractDealiasing = FIT.Types.OrszagTwoThirds(),
        geometry::FIT.Types.AbstractShellGeometry = FIT.Types.IsotropicShells(),
        invariant::FIT.Types.AbstractInvariant = FIT.Types.KineticEnergy(),
        execution::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.SerialBackend())
    geometry isa FIT.Types.ShellMagnitude || throw(ArgumentError(
        "pencil workspace supports FIT.Types.ShellMagnitude shell geometries (FIT.Types.IsotropicShells / " *
        "FIT.Types.PerpendicularShells / FIT.Types.ParallelShells); got $(typeof(geometry))."))
    nd = length(ks)
    ns = FIT.SpectralLayout.full_size(ks)          # physical grid
    out_proto = PencilFFTs.allocate_output(plan)
    ph = PencilFFTs.allocate_input(plan)           # ifft output (physical)
    FT = real(eltype(out_proto))
    (eltype(ph) <: Real) == FIT.SpectralLayout.is_half(ks) || throw(ArgumentError(
        "a real-input pencil plan needs the half wavenumber layout and a complex-input plan the full " *
        "one; build `ks` with `Utils.wavenumber_grid(ns, Ls; real = $(eltype(ph) <: Real))`."))
    Np = FT(prod(ns))
    do_trunc = !(dealiasing isa FIT.Types.NoDealiasing)

    # Local Fourier wavenumbers in the spectral (output) layout — permutation-aware.
    gf = PencilArrays.localgrid(out_proto, ks)
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
    # Hermitian weight: on the half layout a stored mode also stands for its unstored mirror `−k`, except
    # on the `k₁ = 0` plane and (even `n₁`) the Nyquist plane, which are stored whole and are their own
    # mirrors. `Σ_full f = Σ_half W·f` for the even densities below. All ones on a full layout.
    W = similar(out_proto, FT); fill!(W, one(FT))
    if FIT.SpectralLayout.is_half(ks)
        i1  = round.(Int, KC[1] ./ FT(dk[1]))
        nyq = iseven(ns[1]) ? ns[1] ÷ 2 : -1
        W .= ifelse.((i1 .== 0) .| (i1 .== nyq), one(FT), FT(2))
    end
    # Wavenumber for the operators that are first order in k (the gradient, and the vorticity behind the
    # helicity/enstrophy densities). The Nyquist slot of an even axis holds `+n/2` and `−n/2` at once, so
    # `k` is not single-valued there and it carries zero — matching the serial `derivative_wavenumber`.
    KD = ntuple(nd) do d
        a = similar(out_proto, FT); a .= KC[d]
        if iseven(ns[d])
            a .= ifelse.(abs.(round.(Int, KC[d] ./ FT(dk[d]))) .== ns[d] ÷ 2, zero(FT), a)
        end
        a
    end
    # Local execution backend (Serial default; ComputationalBackends.GPUBackend for a per-rank device pencil / multi-GPU,
    # possibly wrapped in ComputationalBackends.MPIBackend, unwrapped here). Drives the per-rank shell reduction below.
    inner = ComputationalBackends.local_backend(execution)
    # Shell edges/centers from the GLOBAL max |k| (MPI.Allreduce → identical on all ranks); local shell index.
    kmax = MPI.Allreduce(maximum(KMAG), max, comm)
    edges     = FIT.ShellBinning.shell_edges(binning, kmax)
    centers   = collect(FIT.ShellBinning.shell_centers(binning, kmax))
    # `FIT.ShellBinning.assign_shells` is a host scalar-indexed builder; on a device backend build it once from a host copy
    # of the local |k| grid (the device reduction moves this index grid back on-device per snapshot).
    shell_idx = inner isa ComputationalBackends.GPUBackend ? FIT.ShellBinning.assign_shells(Array(parent(KMAG)), edges) : FIT.ShellBinning.assign_shells(KMAG, edges)

    û     = ntuple(_ -> PencilFFTs.allocate_output(plan), nd)   # spectral velocity coeffs
    Nhat  = ntuple(_ -> PencilFFTs.allocate_output(plan), nd)   # nonlinear-term spectral
    # Vorticity scratch: only the helicity and enstrophy densities form ω̂ = i k × û.
    ω     = invariant isa FIT.Types.KineticEnergy ? nothing :
            ntuple(_ -> PencilFFTs.allocate_output(plan), nd)
    spec  = PencilFFTs.allocate_output(plan)                    # spectral scratch (KEEP·û, i k·û)
    dloc  = similar(out_proto, FT)                   # transfer density (real spectral)
    uphys = ntuple(_ -> PencilFFTs.allocate_input(plan), nd)    # physical velocity
    N_i   = PencilFFTs.allocate_input(plan)                     # nonlinear accumulator (physical)
    g     = PencilFFTs.allocate_input(plan)                     # gradient (physical)
    Tloc  = zeros(FT, length(centers))
    return PencilWorkspace(plan, comm, KC, KD, KMAG, KEEP, W, shell_idx, centers,
                           û, Nhat, ω, spec, dloc, uphys, N_i, g, ph, Tloc, Np, do_trunc, nd, inner, invariant)
end

# In-place distributed pencil spectral flux — reuses `ws`; 0 alloc beyond the small per-shell vectors.
function FIT.pencil_spectral_flux!(ws::PencilWorkspace, u_phys::NTuple{D};
                                   invariant::FIT.Types.AbstractInvariant = ws.invariant) where {D}
    D == ws.nd || throw(ArgumentError("got $D velocity components for an $(ws.nd)-D pencil workspace."))
    (invariant isa FIT.Types.KineticEnergy || ws.ω !== nothing) && typeof(invariant) === typeof(ws.invariant) ||
        throw(ArgumentError("this PencilWorkspace was built for $(nameof(typeof(ws.invariant))); rebuild it " *
                            "with `invariant = $(nameof(typeof(invariant)))()` to use that one."))
    plan = ws.plan; nd = ws.nd; Np = ws.Np; do_trunc = ws.do_trunc
    û = ws.û; Nhat = ws.Nhat; KD = ws.KD; KEEP = ws.KEEP; spec = ws.spec; uphys = ws.uphys

    for c in 1:nd
        LinearAlgebra.mul!(û[c], plan, u_phys[c]); û[c] ./= Np                 # û_c = fft(u_c)/Np
    end
    # Physical advecting velocity u_j = Np·ifft(û_j) (dealiased). ×Np: û carries the 1/Nᵈ; LinearAlgebra.ldiv! is ifft.
    for j in 1:nd
        do_trunc ? (spec .= KEEP .* û[j]) : copyto!(spec, û[j])
        LinearAlgebra.ldiv!(uphys[j], plan, spec)
        uphys[j] .*= Np
    end
    # N̂_i = fft( Σ_j u_j ∂_j u_i )/Np, dealiased
    for i in 1:nd
        fill!(ws.N_i, zero(eltype(ws.N_i)))
        for j in 1:nd
            do_trunc ? (spec .= (im .* KD[j]) .* KEEP .* û[i]) : (spec .= (im .* KD[j]) .* û[i])
            LinearAlgebra.ldiv!(ws.g, plan, spec)
            ws.N_i .+= uphys[j] .* (ws.g .* Np)                  # physical u_j ∂_j u_i
        end
        LinearAlgebra.mul!(Nhat[i], plan, ws.N_i); Nhat[i] ./= Np
        do_trunc && (Nhat[i] .*= KEEP)
    end
    # Per-mode transfer density (KE/helicity/enstrophy), weighted so the half-layout shell sums equal the
    # full-spectrum ones → local shell sums → global MPI.Allreduce.
    _pencil_transfer_density!(ws.dloc, invariant, û, Nhat, KD, nd, ws.ω)
    ws.dloc .*= ws.W
    # Local shell reduction dispatched on the per-rank backend: scalar host loop (Serial/Threaded) or an
    # atomic device scatter-add (ComputationalBackends.GPUBackend, device-resident) — writes into the host `Tloc`, then reduce.
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
    binning::FIT.Types.AbstractShellBinning,
    dealiasing::FIT.Types.AbstractDealiasing = FIT.Types.OrszagTwoThirds(),
    invariant::FIT.Types.AbstractInvariant = FIT.Types.KineticEnergy(),
    geometry::FIT.Types.AbstractShellGeometry = FIT.Types.IsotropicShells(),
    execution::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.SerialBackend(),
) where {D}
    ws = FIT.PencilWorkspace(plan, ks, comm; binning = binning, dealiasing = dealiasing,
                             geometry = geometry, invariant = invariant, execution = execution)
    return FIT.pencil_spectral_flux!(ws, u_phys; invariant = invariant)
end

end # module FlowInvariantTransferPencilFFTsExt
