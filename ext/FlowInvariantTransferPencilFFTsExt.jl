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

# ---------------------------------------------------------------------------
# Exact 3/2 padding on the pencil axis. The product is formed on a second pencil plan of the padded size
# (`SpectralLayout.padded_length` per axis, the serial FFTW path's grid), and each coefficient moves to
# the rank owning its padded image and back with one `Alltoallv` each way, following the serial map
# (`SpectralLayout.padded_images`).
# ---------------------------------------------------------------------------

# Owner lookup for a pencil: per decomposed dimension, the topology coordinate holding each global index,
# and the rank of every topology coordinate in the pencil's communicator.
struct _Owners{M, P, VI <: AbstractVector{<:Integer}, A <: AbstractArray{Int, M}}
    pen::P
    coord_of::NTuple{M, VI}
    ranks::A
end

function _Owners(pen)
    decomp = PencilArrays.decomposition(pen)
    M = length(decomp)
    pdims = size(PencilArrays.topology(pen))
    ng = PencilArrays.size_global(pen)
    coord_of = ntuple(M) do i
        v = zeros(Int, ng[decomp[i]])
        for c in 1:pdims[i]
            for g in PencilArrays.range_remote(pen, ntuple(j -> j == i ? c : 1, M))[decomp[i]]
                v[g] = c
            end
        end
        v
    end
    comm = PencilArrays.get_comm(pen)
    ranks = [MPI.Cart_rank(comm, collect(Tuple(c) .- 1)) for c in CartesianIndices(pdims)]
    return _Owners(pen, coord_of, ranks)
end

# The rank owning global (logical) index `G`, and the linear index into that rank's `parent` array.
function _owner(o::_Owners, G::NTuple{N, Int}) where {N}
    decomp = PencilArrays.decomposition(o.pen)
    coords = ntuple(i -> o.coord_of[i][G[decomp[i]]], length(decomp))
    rg = PencilArrays.range_remote(o.pen, coords)
    L = ntuple(d -> G[d] - first(rg[d]) + 1, N)
    szm = map(length, PencilArrays.range_remote(o.pen, coords, PencilArrays.MemoryOrder()))
    Lm = PencilArrays.permutation(o.pen) * L
    return o.ranks[coords...], LinearIndices(szm)[Lm...]
end

# A fixed redistribution `dst[j] = Σ w·(conj?)(src[i])` between two pencils, with its send/receive
# layout and buffers built once. Receive entries are grouped into slots in which every destination
# index occurs at most once, so each slot accumulates with one broadcast.
struct _Exchange{VI, VW, VC, SB, RB}
    sendidx::VI; sendw::VW; conjpos::VI
    sendbuf::VC; recvbuf::VC
    sendv::SB; recvv::RB
    slots::Vector{Tuple{VI, VI}}
end

# `sends`: (rank, key, source parent index, weight, conjugate); `recvs`: (rank, key, destination parent
# index). Both sides order by (rank, key), so a sender's entries for a rank line up with that rank's
# receive entries from it.
function _Exchange(proto, ::Type{CT}, sends, recvs, nranks) where {CT}
    sort!(sends; by = s -> (s[1], s[2])); sort!(recvs; by = r -> (r[1], r[2]))
    dev(v) = copyto!(similar(proto, eltype(v), length(v)), v)
    counts(v) = [count(e -> e[1] == r, v) for r in 0:nranks-1]
    sendbuf = similar(proto, CT, length(sends)); recvbuf = similar(proto, CT, length(recvs))
    sendc = counts(sends); recvc = counts(recvs)
    occ = Dict{Int, Int}(); slotpos = Vector{Int}[]; slotidx = Vector{Int}[]
    for (p, r) in enumerate(recvs)
        s = occ[r[3]] = get(occ, r[3], 0) + 1
        s > length(slotpos) && (push!(slotpos, Int[]); push!(slotidx, Int[]))
        push!(slotpos[s], p); push!(slotidx[s], r[3])
    end
    return _Exchange(dev([s[3] for s in sends]), dev([s[4] for s in sends]),
                     dev([i for (i, s) in enumerate(sends) if s[5]]),
                     sendbuf, recvbuf,
                     MPI.VBuffer(sendbuf, sendc, [0; cumsum(sendc)[1:end-1]]),
                     MPI.VBuffer(recvbuf, recvc, [0; cumsum(recvc)[1:end-1]]),
                     [(dev(slotpos[s]), dev(slotidx[s])) for s in eachindex(slotpos)])
end

function _exchange!(dst, src, ex::_Exchange, comm)
    s = parent(src); d = parent(dst)
    ex.sendbuf .= ex.sendw .* view(s, ex.sendidx)
    isempty(ex.conjpos) || (view(ex.sendbuf, ex.conjpos) .= conj.(view(ex.sendbuf, ex.conjpos)))
    MPI.Alltoallv!(ex.sendv, ex.recvv, comm)
    fill!(d, zero(eltype(d)))
    for (pos, idx) in ex.slots
        view(d, idx) .+= view(ex.recvbuf, pos)
    end
    return dst
end

# Padded plan, buffers and the two exchanges: embed (coarse → padded) and truncate (padded → coarse,
# scaled by the forward normalization `1/M`).
struct _PencilPadding{PL, UPT, IB, SP, R, EX, CM}
    plan::PL; uphys::UPT; N_i::IB; g::IB; spec::SP; Mp::R
    embed::EX; trunc::EX; comm::CM
end

function _PencilPadding(plan, ks, comm, ::Type{FT}) where {FT}
    nd = length(ks)
    ns = FIT.SpectralLayout.full_size(ks); ms = FIT.SpectralLayout.spectral_size(ks)
    half = FIT.SpectralLayout.is_half(ks)
    Ms  = ntuple(d -> FIT.SpectralLayout.padded_length(ns[d]), nd)
    Msp = ntuple(d -> (half && d == 1) ? Ms[1] ÷ 2 + 1 : Ms[d], nd)
    pin = PencilArrays.Pencil(PencilArrays.pencil(PencilFFTs.allocate_input(plan)); size_global = Ms)
    transforms = half ? (PencilFFTs.Transforms.RFFT(), ntuple(_ -> PencilFFTs.Transforms.FFT(), nd - 1)...) :
                        ntuple(_ -> PencilFFTs.Transforms.FFT(), nd)
    pplan = PencilFFTs.PencilFFTPlan(pin, transforms, FT)
    spec = PencilFFTs.allocate_output(pplan)
    CT = eltype(spec)
    coarse = PencilArrays.pencil(PencilFFTs.allocate_output(plan)); padded = PencilArrays.pencil(spec)
    oc = _Owners(coarse); op = _Owners(padded)
    linP = LinearIndices(Msp)
    nyq_row = (half && iseven(ns[1])) ? ms[1] : 0
    mirror(C) = (C[1], ntuple(d -> C[d + 1] == 1 ? 1 : ms[d + 1] - C[d + 1] + 2, nd - 1)...)
    nyqaxes(C) = count(d -> FIT.SpectralLayout.is_nyquist(ks[d], C[d]), 1:nd)
    images(C) = Iterators.product(ntuple(d -> FIT.SpectralLayout.padded_images(ks[d], C[d], Ms[d]), nd)...)
    preimage(P) = ntuple(d -> FIT.SpectralLayout.padded_preimage(ks[d], P[d], Ms[d]), nd)
    local_globals(pen) = (Tuple(I) for I in CartesianIndices(PencilArrays.range_local(pen)))
    # The owner ranks are ranks of the pencils' Cartesian communicator, which the exchanges run on.
    cart = PencilArrays.get_comm(coarse)
    nranks = MPI.Comm_size(cart)
    # Embed: every coarse mode sends to each of its images, halved once per Nyquist axis.
    esend = Tuple{Int, Int, Int, FT, Bool}[]; erecv = Tuple{Int, Int, Int}[]
    for C in local_globals(coarse)
        _, ci = _owner(oc, C)
        we = FT(1) / FT(2)^nyqaxes(C)
        for P in images(C)
            r, _ = _owner(op, P)
            push!(esend, (r, linP[P...], ci, we, false))
        end
    end
    for P in local_globals(padded)
        C = preimage(P); any(iszero, C) && continue
        r, _ = _owner(oc, C); _, pi = _owner(op, P)
        push!(erecv, (r, linP[P...], pi))
    end
    # Truncate: each image returns to its mode, weighted to undo the embed; on a half layout's Nyquist
    # plane each mode takes half of its own images and half the conjugate of its mirror's.
    tsend = Tuple{Int, Int, Int, FT, Bool}[]; trecv = Tuple{Int, Int, Int}[]
    invM = FT(1) / FT(prod(Ms))
    for P in local_globals(padded)
        C = preimage(P); any(iszero, C) && continue
        _, pi = _owner(op, P)
        wt = invM * FT(2)^nyqaxes(C) / FT(length(images(C)))
        if C[1] == nyq_row
            push!(tsend, (first(_owner(oc, C)), 2linP[P...], pi, wt / 2, false))
            push!(tsend, (first(_owner(oc, mirror(C))), 2linP[P...] + 1, pi, wt / 2, true))
        else
            push!(tsend, (first(_owner(oc, C)), 2linP[P...], pi, wt, false))
        end
    end
    for C in local_globals(coarse)
        _, ci = _owner(oc, C)
        for P in images(C)
            push!(trecv, (first(_owner(op, P)), 2linP[P...], ci))
        end
        if C[1] == nyq_row
            for P in images(mirror(C))
                push!(trecv, (first(_owner(op, P)), 2linP[P...] + 1, ci))
            end
        end
    end
    proto = parent(spec)
    embed = _Exchange(proto, CT, esend, erecv, nranks)
    trunc = _Exchange(proto, CT, tsend, trecv, nranks)
    uphys = ntuple(_ -> PencilFFTs.allocate_input(pplan), nd)
    return _PencilPadding(pplan, uphys, PencilFFTs.allocate_input(pplan), PencilFFTs.allocate_input(pplan),
                          spec, FT(prod(Ms)), embed, trunc, cart)
end

# N̂_i = truncate(fft_padded(Σ_j u_j ∂_j u_i)) with u and ∂u synthesised on the padded grid.
function _pencil_padded_nonlinear!(ws, pad::_PencilPadding)
    nd = ws.nd; û = ws.û
    for j in 1:nd
        _exchange!(pad.spec, û[j], pad.embed, pad.comm)
        LinearAlgebra.ldiv!(pad.uphys[j], pad.plan, pad.spec)
        pad.uphys[j] .*= pad.Mp
    end
    for i in 1:nd
        fill!(pad.N_i, zero(eltype(pad.N_i)))
        for j in 1:nd
            ws.spec .= (im .* ws.KD[j]) .* û[i]
            _exchange!(pad.spec, ws.spec, pad.embed, pad.comm)
            LinearAlgebra.ldiv!(pad.g, pad.plan, pad.spec)
            pad.N_i .+= pad.uphys[j] .* (pad.g .* pad.Mp)
        end
        LinearAlgebra.mul!(pad.spec, pad.plan, pad.N_i)
        _exchange!(ws.Nhat[i], pad.spec, pad.trunc, pad.comm)
    end
    return ws.Nhat
end

# Reusable pencil workspace: the (geometry/dealiasing/binning-fixed) wavenumber grids + shell structure
# and every per-snapshot scratch field, so `pencil_spectral_flux!` reuses them (0 alloc beyond the small
# per-shell result vectors). Spectral-layout buffers via PencilFFTs.allocate_output; physical-layout via PencilFFTs.allocate_input.
struct PencilWorkspace{PL, CM, KCT, KM, KP, SI, CE, UST, OT, SP, UPT, IB, WT, R, EX, IV, PD}
    plan::PL; comm::CM
    KC::KCT; KD::KCT; KMAG::KM; KEEP::KP; W::KM; shell_idx::SI; centers::CE
    û::UST; Nhat::UST; ω::OT; spec::SP; dloc::KM
    uphys::UPT; N_i::IB; g::IB; ph::IB
    Tloc::WT; Np::R; do_trunc::Bool; nd::Int; inner::EX; invariant::IV
    pad::PD                                  # `_PencilPadding` for PaddedThreeHalves, else `nothing`
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
    do_trunc = dealiasing isa FIT.Types.OrszagTwoThirds
    pad = dealiasing isa FIT.Types.PaddedThreeHalves ? _PencilPadding(plan, ks, comm, FT) : nothing

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
    # Orszag 2/3 keep-mask: keep the folded integer index |m_d| ≤ `SpectralLayout.dealias_cutoff(N_d)`.
    dk = ntuple(d -> abs(ks[d][2] - ks[d][1]), nd)
    KEEP = similar(out_proto, Bool); fill!(KEEP, true)
    if do_trunc
        for d in 1:nd
            cutoff = FIT.SpectralLayout.dealias_cutoff(ns[d])
            KEEP .&= (round.(Int, abs.(KC[d]) ./ FT(dk[d])) .<= cutoff)
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
                           û, Nhat, ω, spec, dloc, uphys, N_i, g, ph, Tloc, Np, do_trunc, nd, inner, invariant,
                           pad)
end

# N̂_i = fft(Σ_j u_j ∂_j u_i)/Np on the pencil grid, with the 2/3 keep-mask on inputs and output when
# `ws.do_trunc`. ×Np on the synthesis: û carries the 1/Nᵈ, and `ldiv!` is the normalised inverse.
function _pencil_nonlinear!(ws)
    plan = ws.plan; nd = ws.nd; Np = ws.Np; do_trunc = ws.do_trunc
    û = ws.û; Nhat = ws.Nhat; KD = ws.KD; KEEP = ws.KEEP; spec = ws.spec; uphys = ws.uphys
    for j in 1:nd
        do_trunc ? (spec .= KEEP .* û[j]) : copyto!(spec, û[j])
        LinearAlgebra.ldiv!(uphys[j], plan, spec)
        uphys[j] .*= Np
    end
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
    return Nhat
end

# In-place distributed pencil spectral flux — reuses `ws`; 0 alloc beyond the small per-shell vectors.
function FIT.pencil_spectral_flux!(ws::PencilWorkspace, u_phys::NTuple{D};
                                   invariant::FIT.Types.AbstractInvariant = ws.invariant) where {D}
    D == ws.nd || throw(ArgumentError("got $D velocity components for an $(ws.nd)-D pencil workspace."))
    (invariant isa FIT.Types.KineticEnergy || ws.ω !== nothing) && typeof(invariant) === typeof(ws.invariant) ||
        throw(ArgumentError("this PencilWorkspace was built for $(nameof(typeof(ws.invariant))); rebuild it " *
                            "with `invariant = $(nameof(typeof(invariant)))()` to use that one."))
    plan = ws.plan; nd = ws.nd; Np = ws.Np
    û = ws.û; Nhat = ws.Nhat; KD = ws.KD

    for c in 1:nd
        LinearAlgebra.mul!(û[c], plan, u_phys[c]); û[c] ./= Np                 # û_c = fft(u_c)/Np
    end
    if ws.pad === nothing
        _pencil_nonlinear!(ws)
    else
        _pencil_padded_nonlinear!(ws, ws.pad)
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
