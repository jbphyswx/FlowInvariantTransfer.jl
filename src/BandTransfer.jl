module BandTransfer

using ..Types: Types
using FlowTransformBindings: FlowTransformBindings as FTB
using ComputationalBackends: ComputationalBackends
using SpectralBackends: SpectralBackends
using ..Invariants: Invariants
using ..NonlinearTerm: NonlinearTerm
using ..Workspaces: Workspaces
using ..ShellBinning: ShellBinning
using ..CoarseGrainingFlux: CoarseGrainingFlux

export calculate_band_to_band_transfer, calculate_band_to_band_transfer!, calculate_band_to_band_transfer_batch, BandTransferWorkspace

# ---------------------------------------------------------------------------
# Smooth band-to-band transfer T(K,Q) (Eyink & Aluie 2009)
# ---------------------------------------------------------------------------
#
# Replace the sharp shell indicator χ_m with a graded partition of unity w_m(κ) (see SmoothBands),
# giving a smooth band-to-band transfer. With B(g,h) = Σ_k Re{ĝ*·(u·∇)ĥ}, the band transfer is
#   T(n,m) = B(w_n ⊙ û, w_m ⊙ û),
# which is antisymmetric (B(g,h)+B(h,g) = ∫(u·∇)(g·h) = 0 for incompressible u) and, since
# Σ_n w_n = 1, conserves (Σ_{n,m} T = 0) and reduces to the band-summed transfer spectrum
# (Σ_m T(n,m) = the w_n-weighted transfer density). One nonlinear evaluation per band.

"""
    BandWeights{FT}

Renormalized log-Gaussian band weights `w_n(κ)` (a partition of unity over the bands) of a mode set,
held over the distinct shell coordinates `κ` the modes take: `idx[I]` is the position of mode `I`'s
coordinate among them and `table[n, q]` the weight of band `n` there. A band's weight at a mode is
`table[n, idx[I]]`.
"""
struct BandWeights{FT, IA<:AbstractArray{Int32}, M<:AbstractMatrix{FT}}
    idx::IA
    table::M
end

Base.show(io::IO, w::BandWeights) = print(io, "BandWeights(", size(w.table, 1), " bands, ",
                                          size(w.table, 2), " shell coordinates)")

function _band_weights(centers, logwidth, k_coord)
    nb = length(centers)
    FT = eltype(k_coord)
    σ  = FT(logwidth)
    κs = sort!(unique(vec(k_coord)))
    idx = Int32.(searchsortedfirst.(Ref(κs), k_coord))
    table = zeros(FT, nb, length(κs))
    @inbounds for (q, κ) in enumerate(κs)
        κ <= 0 && continue                       # DC / zero-coordinate modes carry no band weight
        s = zero(FT)
        for n in 1:nb
            r = log(κ / FT(centers[n]))
            wn = exp(-(r*r) / (2*σ*σ))
            table[n, q] = wn
            s += wn
        end
        if s > 0
            for n in 1:nb
                table[n, q] /= s
            end
        end
    end
    return BandWeights(idx, table)
end

# `f_m = w_m ⊙ û` over the `D` components.
function _band_field!(f_m, bw::BandWeights, m::Int, velocity_hat, ns, D::Int)
    idx = bw.idx; table = bw.table
    @inbounds for c in 1:D, I in CartesianIndices(ns)
        f_m[I, c] = table[m, idx[I]] * velocity_hat[I, c]
    end
    return f_m
end

# `T[n, m] = Σ_I fold(I) w_n(I) d(I)` for every band `n`: `d` summed into `Dq` by shell coordinate, then
# the table applied to `Dq`.
function _band_column!(T, m::Int, bw::BandWeights, d, ks, Dq)
    idx = bw.idx; table = bw.table
    fill!(Dq, zero(eltype(Dq)))
    @inbounds for I in CartesianIndices(size(idx))
        Dq[idx[I]] += FTB.mode_fold(ks, I) * d[I]
    end
    @inbounds for n in axes(table, 1)
        s = zero(eltype(Dq))
        for q in eachindex(Dq)
            s += table[n, q] * Dq[q]
        end
        T[n, m] = s
    end
    return T
end

"""
    BandTransferWorkspace(velocity_hat, ks, bands::SmoothBands; geometry=IsotropicShells(),
                          dealiasing=OrszagTwoThirds())

Reusable workspace for [`calculate_band_to_band_transfer!`](@ref): the nonlinear-term workspace,
band-field/transfer-density scratch, the smooth band weights ([`BandWeights`](@ref)) and the per-shell
sums `Dq`. `dealiasing` is the one the transfer will be called with; `PaddedThreeHalves()` adds the
padded-grid scratch.
"""
struct BandTransferWorkspace{WS, FA, DA, WV, CV, QV}
    nlt::WS
    f_m::FA
    d::DA
    W::WV
    centers::CV
    Dq::QV
end

function BandTransferWorkspace(velocity_hat, ks, bands::Types.SmoothBands;
                               geometry::Types.AbstractShellGeometry = Types.IsotropicShells(),
                               dealiasing::Types.AbstractDealiasing = Types.OrszagTwoThirds())
    nd = length(ks)
    ns = size(velocity_hat)[1:nd]
    FT = real(eltype(velocity_hat))
    k_coord = ShellBinning.shell_coordinate(geometry, ks)
    W = _band_weights(bands.centers, bands.logwidth, k_coord)
    return BandTransferWorkspace(Workspaces.NonlinearTermWorkspace(velocity_hat, ks; dealiasing = dealiasing),
                                 similar(velocity_hat), similar(velocity_hat, FT, ns...),
                                 W, collect(bands.centers), zeros(FT, size(W.table, 2)))
end

"""
    calculate_band_to_band_transfer!(T, net, bws::BandTransferWorkspace, velocity_hat, ks; kwargs...)

In-place smooth band-to-band transfer: writes the `nb×nb` matrix `T` and the `nb`-vector `net`,
reusing `bws` (0 alloc beyond those). Returns the same `(centers, transfer_matrix, net_transfer)`
named tuple as the allocating version.
"""
function calculate_band_to_band_transfer!(
    T, net, bws::BandTransferWorkspace, velocity_hat, ks;
    dealiasing::Types.AbstractDealiasing = Types.OrszagTwoThirds(),
    invariant::Types.AbstractInvariant = Types.KineticEnergy(),
    transform::SpectralBackends.AbstractSpectralBackend = SpectralBackends.AutoSpectralBackend(),
    execution::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.SerialBackend(),
    advecting_hat = velocity_hat,
)
    transform = Types.resolve_transform(Types.require_coefficient_transform(transform))
    FT = real(eltype(velocity_hat))
    nb = length(bws.centers)
    _band_to_band_fill!(Types.resolve_execution(execution), T, bws, velocity_hat, ks;
        dealiasing=dealiasing, invariant=invariant, transform=transform, advecting_hat=advecting_hat)
    @inbounds for n in 1:nb
        acc = zero(FT)
        for m in 1:nb; acc += T[n, m]; end
        net[n] = acc
    end
    return (centers = bws.centers, transfer_matrix = T, net_transfer = net)
end

# Fill T[·,m], one nonlinear term per band m, dispatched on execution. Bands write disjoint columns;
# the threaded, distributed and GPU stubs below are defined by the extensions, each worker with its
# own single-threaded workspace.
function _band_to_band_fill!(::ComputationalBackends.SerialBackend, T, bws::BandTransferWorkspace, velocity_hat, ks;
                             dealiasing, invariant, transform, advecting_hat)
    nd = length(ks); ns = size(velocity_hat)[1:nd]; D = size(velocity_hat, nd + 1)
    nb = length(bws.centers)
    ws = bws.nlt
    NonlinearTerm.synthesize_advecting!(ws, advecting_hat, ks; dealiasing = dealiasing, transform = transform)
    for m in 1:nb
        _band_field!(bws.f_m, bws.W, m, velocity_hat, ns, D)
        NonlinearTerm.compute_advection!(ws, bws.f_m, ks; dealiasing = dealiasing, transform = transform)
        Invariants.transfer_density!(bws.d, invariant, velocity_hat, ws.N̂, ks)
        _band_column!(T, m, bws.W, bws.d, ks, bws.Dq)
    end
    return T
end
_band_to_band_fill!(::ComputationalBackends.ThreadedBackend, T, bws, velocity_hat, ks; kwargs...) =
    _band_to_band_threaded!(T, bws, velocity_hat, ks; kwargs...)
_band_to_band_fill!(exec::ComputationalBackends.DistributedBackend, T, bws, velocity_hat, ks; kwargs...) =
    _band_to_band_distributed!(T, bws, velocity_hat, ks, exec; kwargs...)
_band_to_band_fill!(gpu::ComputationalBackends.GPUBackend, T, bws, velocity_hat, ks; kwargs...) =
    _band_to_band_gpu!(T, bws, velocity_hat, ks, gpu; kwargs...)

_band_to_band_threaded!(args...; kwargs...) = throw(ArgumentError(
    "Threaded band-to-band transfer requires OhMyThreads. Run `using OhMyThreads` to load the extension."))
_band_to_band_distributed!(args...; kwargs...) = throw(ArgumentError(
    "Distributed band-to-band transfer requires Distributed. Run `using Distributed` to load the extension."))
_band_to_band_gpu!(args...; kwargs...) = throw(ArgumentError(
    "GPU band-to-band transfer requires KernelAbstractions. Run `using KernelAbstractions` to load the extension."))

"""
    calculate_band_to_band_transfer(velocity_hat, ks; bands::SmoothBands, dealiasing=OrszagTwoThirds(),
        invariant=KineticEnergy(), transform=SpectralBackends.AutoSpectralBackend(), execution=SerialBackend(),
        advecting_hat=velocity_hat, geometry=IsotropicShells())
        -> (centers, transfer_matrix, net_transfer)

Smooth band-to-band transfer `T(n,m)` of a quadratic invariant between the graded spectral
`bands` (Eyink & Aluie 2009) — the smooth-filter analogue of `calculate_shell_to_shell_transfer`.
For incompressible flow `T` is antisymmetric (`T(n,m) = −T(m,n)`), conserves (`Σ T = 0`), and
`Σ_m T(n,m)` is the band-summed transfer spectrum. Accepts an `advecting_hat` (primary field can be
a passive scalar advected by the velocity) and an anisotropic `geometry`. Under
`execution = GPUBackend(dev)` host fields are copied to `dev` once per call. See
[`calculate_band_to_band_transfer!`](@ref) for the in-place, workspace-reusing variant.
"""
function calculate_band_to_band_transfer(
    velocity_hat,
    ks;
    bands::Types.SmoothBands,
    dealiasing::Types.AbstractDealiasing = Types.OrszagTwoThirds(),
    invariant::Types.AbstractInvariant = Types.KineticEnergy(),
    transform::SpectralBackends.AbstractSpectralBackend = SpectralBackends.AutoSpectralBackend(),
    execution::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.SerialBackend(),
    advecting_hat = velocity_hat,
    geometry::Types.AbstractShellGeometry = Types.IsotropicShells(),
)
    transform = Types.resolve_transform(Types.require_coefficient_transform(transform))
    exec = Types.resolve_execution(execution)
    velocity_hat, advecting_hat = Types._fields_to_execution(exec, velocity_hat, advecting_hat)
    FT = real(eltype(velocity_hat))
    nb = length(bands.centers)
    bws = BandTransferWorkspace(velocity_hat, ks, bands; geometry=geometry, dealiasing=dealiasing)
    T   = zeros(FT, nb, nb)
    net = zeros(FT, nb)
    return calculate_band_to_band_transfer!(T, net, bws, velocity_hat, ks;
        dealiasing=dealiasing, invariant=invariant, transform=transform, execution=exec,
        advecting_hat=advecting_hat)
end

"""
    calculate_band_to_band_transfer_batch(velocity_hats, ks; bands, dealiasing, invariant, transform,
        geometry, execution) -> Vector

Band-to-band transfer for a batch of snapshots sharing one grid (`velocity_hats` an iterable of
`(ms..., D)` coefficient arrays). The band weights are built once and each worker reuses one
`BandTransferWorkspace` across its snapshots; `execution = ThreadedBackend()` (requires
`using OhMyThreads`) threads over snapshots, each with a serial inner transform. Results (the same
NamedTuple as the single-snapshot form) are in input order, bit-identical to calling
[`calculate_band_to_band_transfer`](@ref) per snapshot.
"""
function calculate_band_to_band_transfer_batch(
    velocity_hats,
    ks;
    bands::Types.SmoothBands,
    dealiasing::Types.AbstractDealiasing = Types.OrszagTwoThirds(),
    invariant::Types.AbstractInvariant = Types.KineticEnergy(),
    transform::SpectralBackends.AbstractSpectralBackend = SpectralBackends.AutoSpectralBackend(),
    geometry::Types.AbstractShellGeometry = Types.IsotropicShells(),
    execution::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.SerialBackend(),
)
    transform = Types.resolve_transform(Types.require_coefficient_transform(transform))
    n       = length(velocity_hats)
    FT      = real(eltype(first(velocity_hats)))
    nb      = length(bands.centers)
    centers = collect(bands.centers)
    NT = @NamedTuple{centers::typeof(centers), transfer_matrix::Matrix{FT}, net_transfer::Vector{FT}}
    results = Vector{NT}(undef, n)
    n == 0 && return results
    _band_to_band_batch!(Types.resolve_execution(execution), results, velocity_hats, ks, bands, nb;
        dealiasing=dealiasing, invariant=invariant, transform=transform, geometry=geometry)
    return results
end

# Serial reference: one workspace reused across the whole batch; each snapshot self-advects.
function _band_to_band_batch!(::ComputationalBackends.AbstractSerialBackend, results, velocity_hats, ks, bands, nb;
                              dealiasing, invariant, transform, geometry)
    FT  = real(eltype(first(velocity_hats)))
    bws = BandTransferWorkspace(first(velocity_hats), ks, bands; geometry=geometry, dealiasing=dealiasing)
    for i in eachindex(velocity_hats)
        vh = velocity_hats[i]
        T = zeros(FT, nb, nb); net = zeros(FT, nb)
        results[i] = calculate_band_to_band_transfer!(T, net, bws, vh, ks;
            dealiasing=dealiasing, invariant=invariant, transform=transform,
            execution=ComputationalBackends.SerialBackend(), advecting_hat=vh)
    end
    return results
end

# Threaded over the batch — overridden by the OhMyThreads extension (per-chunk workspace pool, serial inner).
function _band_to_band_batch_threaded!(args...; kwargs...)
    throw(ArgumentError("execution = ThreadedBackend() for the band-to-band batch requires OhMyThreads. " *
                        "Run `using OhMyThreads` to load the extension."))
end
_band_to_band_batch!(::ComputationalBackends.AbstractThreadedBackend, results, velocity_hats, ks, bands, nb; kwargs...) =
    _band_to_band_batch_threaded!(results, velocity_hats, ks, bands, nb; kwargs...)

# GPU over the batch: one workspace in the device's memory reused for every snapshot; host snapshots are
# copied into the staging buffer.
function _band_to_band_batch!(gpu::ComputationalBackends.AbstractGPUBackend, results, velocity_hats, ks, bands, nb;
                              dealiasing, invariant, transform, geometry)
    FT  = real(eltype(first(velocity_hats)))
    buf = Types._staging_buffer(gpu, first(velocity_hats))
    bws = BandTransferWorkspace(buf === nothing ? first(velocity_hats) : buf, ks, bands;
                                geometry=geometry, dealiasing=dealiasing)
    for i in eachindex(velocity_hats)
        vh = Types._staged!(buf, velocity_hats[i])
        T = zeros(FT, nb, nb); net = zeros(FT, nb)
        results[i] = calculate_band_to_band_transfer!(T, net, bws, vh, ks;
            dealiasing=dealiasing, invariant=invariant, transform=transform, execution=gpu, advecting_hat=vh)
    end
    return results
end

# `DistributedBackend` and `MPIBackend` batches are methods of the Distributed and MPI extensions.
_band_to_band_batch!(be::ComputationalBackends.AbstractExecutionBackend, results, velocity_hats, ks, bands, nb; kwargs...) =
    Types.batch_unavailable("calculate_band_to_band_transfer_batch", be)

Base.show(io::IO, ::BandTransferWorkspace) = print(io, "BandTransferWorkspace(…)")
Base.show(io::IO, ::MIME"text/plain", w::BandTransferWorkspace) = show(io, w)

# ---------------------------------------------------------------------------
# Band content of an invariant, the quantity the band-to-band transfer moves between bands.
# ---------------------------------------------------------------------------

# `E = ½|û|²` and `Z = ½|ω̂|²` carry the half their definitions do; helicity's `u·ω` does not.
_density_scale(::Union{Types.KineticEnergy, Types.PassiveScalar, Types.Enstrophy}, ::Type{FT}) where {FT} = FT(0.5)
_density_scale(::Types.Helicity, ::Type{FT}) where {FT} = one(FT)

"""
    calculate_band_energies(velocity_hat, ks; bands, invariant, geometry) -> (; centers, energies)

Content of `invariant` in each band of `bands`, `E(n) = Σ_k w_n(k)·e(k)`.

`bands` fixes the weight `w_n`: an [`AbstractShellBinning`](@ref) gives the indicator of shell `n`,
[`SmoothBands`](@ref) the graded partition of unity of [`calculate_band_to_band_transfer`](@ref), so
the same `bands` passed to both describe one partition. `e` is the invariant's quadratic spectral
density (`½Σ_c|û_c|²` for kinetic energy, `½|ω̂|²` for enstrophy, `Re{conj(û)·ω̂}` for helicity),
with the Hermitian weight on a half spectrum so the total matches the full-spectrum sum.
"""
function CoarseGrainingFlux.calculate_band_energies(
    velocity_hat::AbstractArray{<:Complex}, ks::Tuple;
    bands::Union{Types.AbstractShellBinning, Types.SmoothBands},
    invariant::Types.AbstractInvariant = Types.KineticEnergy(),
    geometry::Types.AbstractShellGeometry = Types.IsotropicShells(),
)
    return _band_energies(velocity_hat, ks, bands, invariant, geometry)
end

function _band_energies(velocity_hat, ks, binning::Types.AbstractShellBinning, invariant, geometry)
    FT = real(eltype(velocity_hat))
    kmax = ShellBinning.max_shell_coordinate(geometry, ks)
    centers = collect(ShellBinning.shell_centers(binning, kmax))
    sidx = ShellBinning.assign_shells(ShellBinning.shell_coordinate(geometry, ks),
                                      ShellBinning.shell_edges(binning, kmax))
    E = zeros(FT, length(centers))
    # The sharp band sum is the transfer drivers' shell scatter with the field in both slots.
    Invariants.transfer_density_scatter!(E, invariant, velocity_hat, velocity_hat, ks, sidx)
    E .*= _density_scale(invariant, FT)
    return (; centers = centers, energies = E)
end

function _band_energies(velocity_hat, ks, bands::Types.SmoothBands, invariant, geometry)
    nd = length(ks)
    ms = size(velocity_hat)[1:nd]
    M  = size(velocity_hat, nd + 1)
    FT = real(eltype(velocity_hat))
    W = _band_weights(bands.centers, bands.logwidth, ShellBinning.shell_coordinate(geometry, ks))
    Eq = zeros(FT, size(W.table, 2))
    @inbounds for I in CartesianIndices(ms)
        Eq[W.idx[I]] += FTB.mode_fold(ks, I) *
                        Invariants._density_at(invariant, velocity_hat, velocity_hat, ks, I, M)
    end
    E = W.table * Eq
    E .*= _density_scale(invariant, FT)
    return (; centers = collect(bands.centers), energies = E)
end

end # module BandTransfer
