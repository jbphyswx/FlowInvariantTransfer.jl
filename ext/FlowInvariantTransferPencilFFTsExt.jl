module FlowInvariantTransferPencilFFTsExt

using MPI: MPI
using PencilFFTs: PencilFFTs, PencilFFTPlan, Transforms, allocate_input, allocate_output
using PencilArrays: PencilArrays, localgrid
using LinearAlgebra: mul!, ldiv!
using FlowInvariantTransfer: FlowInvariantTransfer as FET
using FlowInvariantTransfer.Types: AbstractShellBinning, AbstractInvariant, KineticEnergy,
                                   AbstractDealiasing, OrszagTwoThirds, NoDealiasing,
                                   AbstractShellGeometry, ShellMagnitude, IsotropicShells
using FlowInvariantTransfer.ShellBinning: shell_edges, shell_centers, assign_shells

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

# Implements the FET.build_pencil_plan stub (docstring lives on the core stub).
function FET.build_pencil_plan(ns::NTuple{nd,Int}, comm = MPI.COMM_WORLD; T = Float64) where {nd}
    proc_dims  = Tuple(Int.(MPI.Dims_create(MPI.Comm_size(comm), ntuple(_ -> 0, nd - 1))))
    transforms = ntuple(_ -> Transforms.FFT(), nd)
    return PencilFFTPlan(ns, transforms, proc_dims, comm, T)
end

function FET.pencil_spectral_flux(
    u_phys::NTuple{D, <:PencilArrays.PencilArray},
    plan,
    ks;
    comm = MPI.COMM_WORLD,
    binning::AbstractShellBinning,
    dealiasing::AbstractDealiasing = OrszagTwoThirds(),
    invariant::AbstractInvariant = KineticEnergy(),
    geometry::AbstractShellGeometry = IsotropicShells(),
) where {D}
    invariant isa KineticEnergy ||
        throw(ArgumentError("pencil_spectral_flux currently supports KineticEnergy only (got $(typeof(invariant)))."))
    geometry isa ShellMagnitude{Nothing} ||
        throw(ArgumentError("pencil_spectral_flux currently supports isotropic |k| shells (IsotropicShells()) only."))
    nd = length(ks)
    nd == D || throw(ArgumentError("got $D velocity components for an $nd-D grid; pencil flux needs D == nd."))
    ns = ntuple(d -> length(ks[d]), nd)
    FT = real(eltype(u_phys[1]))
    Np = FT(prod(ns))
    do_trunc = !(dealiasing isa NoDealiasing)

    # Local Fourier wavenumbers in the spectral (output) layout — permutation-aware.
    out_proto = allocate_output(plan)
    gf = localgrid(out_proto, ks)
    KC = ntuple(nd) do d
        a = similar(out_proto, FT); a .= gf[d]; a               # k_d value at every local spectral point
    end
    KMAG = similar(out_proto, FT)
    fill!(KMAG, zero(FT))
    for d in 1:nd
        KMAG .= sqrt.(KMAG .^ 2 .+ KC[d] .^ 2)
    end

    # Orszag 2/3 keep-mask: discard if the folded integer index |k_d| ≥ N_d ÷ 3 (integer cutoff,
    # matching the serial _is_dealiased exactly). dk_d converts k-value → integer index.
    dk = ntuple(d -> abs(ks[d][2] - ks[d][1]), nd)
    KEEP = similar(out_proto, Bool)
    fill!(KEEP, true)
    if do_trunc
        for d in 1:nd
            cutoff = ns[d] ÷ 3
            KEEP .&= (round.(Int, abs.(KC[d]) ./ FT(dk[d])) .< cutoff)
        end
    end

    # û_c = fft(u_c)/Np  (package coefficient convention)
    û = ntuple(nd) do c
        out = allocate_output(plan)
        mul!(out, plan, u_phys[c])
        out ./= Np
        out
    end

    # Physical advecting velocity u_phys_j = real(ifft(keep ⊙ û_j))
    uphys = ntuple(nd) do j
        spec = do_trunc ? (KEEP .* û[j]) : copy(û[j])
        ph   = allocate_input(plan)
        ldiv!(ph, plan, spec)
        real.(ph)
    end

    # N̂_i = fft( Σ_j u_j ∂_j u_i )/Np, dealiased
    Nhat = ntuple(nd) do i
        N_i = allocate_input(plan)
        fill!(N_i, zero(eltype(N_i)))
        for j in 1:nd
            spec = (im .* KC[j]) .* (do_trunc ? (KEEP .* û[i]) : û[i])   # i k_j û_i (dealiased)
            g = allocate_input(plan)
            ldiv!(g, plan, spec)
            N_i .+= uphys[j] .* real.(g)
        end
        out = allocate_output(plan)
        mul!(out, plan, N_i)
        out ./= Np
        do_trunc && (out .*= KEEP)
        out
    end

    # Shell edges/centers from the GLOBAL max |k| (Allreduce of the local max → identical on all ranks).
    kmax = MPI.Allreduce(maximum(KMAG), max, comm)
    edges   = shell_edges(binning, kmax)
    centers = shell_centers(binning, kmax)
    shell_idx = assign_shells(KMAG, edges)        # local per-mode shell index
    nsh = length(centers)

    # KE transfer density d(k) = Σ_i Re{conj(û_i) N̂_i}; bin into local shells.
    Tloc = zeros(FT, nsh)
    @inbounds for I in CartesianIndices(û[1])
        s = shell_idx[I]
        s == 0 && continue
        dval = zero(FT)
        for i in 1:nd
            dval += real(conj(û[i][I]) * Nhat[i][I])
        end
        Tloc[s] += dval
    end

    Tglob = MPI.Allreduce(Tloc, +, comm)          # global per-shell transfer
    flux  = cumsum(Tglob)                          # Π(K) = Σ_{k≤K} T(k)
    return (centers = centers, transfer_spectrum = Tglob, flux = flux)
end

end # module FlowInvariantTransferPencilFFTsExt
