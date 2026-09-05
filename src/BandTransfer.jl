module BandTransfer

using ..Types: Types
using ..SpectralLayout: SpectralLayout
using ComputationalBackends: ComputationalBackends
using SpectralBackends: SpectralBackends
using ..Invariants: Invariants
using ..NonlinearTerm: NonlinearTerm
using ..Workspaces: Workspaces
using ..ShellBinning: ShellBinning

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

# Renormalized log-Gaussian band weights w_n(κ) at each mode (partition of unity over bands).
function _band_weights(centers, logwidth, k_coord)
    nb = length(centers)
    FT = eltype(k_coord)
    ns = size(k_coord)
    σ  = FT(logwidth)
    W  = [fill!(similar(k_coord), zero(FT)) for _ in 1:nb]
    @inbounds for I in CartesianIndices(ns)
        κ = k_coord[I]
        κ <= 0 && continue                       # DC / zero-coordinate modes carry no band weight
        s = zero(FT)
        for n in 1:nb
            r = log(κ / FT(centers[n]))
            wn = exp(-(r*r) / (2*σ*σ))
            W[n][I] = wn
            s += wn
        end
        if s > 0
            for n in 1:nb
                W[n][I] /= s
            end
        end
    end
    return W
end

"""
    BandTransferWorkspace(velocity_hat, ks, bands::SmoothBands; geometry=IsotropicShells())

Reusable workspace for [`calculate_band_to_band_transfer!`](@ref): the nonlinear-term workspace,
band-field/transfer-density scratch, and the precomputed smooth band-weight masks.
"""
struct BandTransferWorkspace{WS, FA, DA, WV, CV}
    nlt::WS
    f_m::FA
    d::DA
    W::WV
    centers::CV
end

function BandTransferWorkspace(velocity_hat, ks, bands::Types.SmoothBands;
                               geometry::Types.AbstractShellGeometry = Types.IsotropicShells())
    nd = length(ks)
    ns = size(velocity_hat)[1:nd]
    FT = real(eltype(velocity_hat))
    k_coord = ShellBinning.shell_coordinate(geometry, ks)
    W = _band_weights(bands.centers, bands.logwidth, k_coord)
    return BandTransferWorkspace(Workspaces.NonlinearTermWorkspace(velocity_hat, ks),
                                 similar(velocity_hat), similar(velocity_hat, FT, ns...),
                                 W, collect(bands.centers))
end

"""
    calculate_band_to_band_transfer!(T, net, bws::BandTransferWorkspace, velocity_hat, ks; kwargs...)

In-place smooth band-to-band transfer: writes the `nb×nb` matrix `T` and the `nb`-vector `net`,
reusing `bws` (0 alloc beyond those). Returns the same `(centers, transfer_matrix, net_transfer,
max_antisymmetry_error)` named tuple as the allocating version.
"""
function calculate_band_to_band_transfer!(
    T, net, bws::BandTransferWorkspace, velocity_hat, ks;
    dealiasing::Types.AbstractDealiasing = Types.OrszagTwoThirds(),
    invariant::Types.AbstractInvariant = Types.KineticEnergy(),
    spectral::SpectralBackends.AbstractSpectralBackend = SpectralBackends.AutoSpectralBackend(),
    execution::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.SerialBackend(),
    advecting_hat = velocity_hat,
)
    spectral = Types.resolve_spectral(Types.require_coefficient_spectral(spectral))
    FT = real(eltype(velocity_hat))
    nb = length(bws.centers)
    # Fill the transfer matrix (one nonlinear term per band m → column T[·,m]), dispatched on execution.
    _band_to_band_fill!(Types.resolve_execution(execution), T, bws, velocity_hat, ks;
        dealiasing=dealiasing, invariant=invariant, spectral=spectral, advecting_hat=advecting_hat)
    @inbounds for n in 1:nb
        acc = zero(FT)
        for m in 1:nb; acc += T[n, m]; end
        net[n] = acc
    end
    asym = zero(FT)
    for n in 1:nb, m in 1:nb
        a = abs(T[n, m] + T[m, n])
        a > asym && (asym = a)
    end
    return (centers = bws.centers, transfer_matrix = T,
            net_transfer = net, max_antisymmetry_error = asym)
end

# Fill T[·,m] over bands, dispatched on execution. Bands are independent (disjoint columns) →
# embarrassingly parallel; threaded/distributed/GPU override the named stubs, each worker using its
# own single-threaded workspace (inner FFTs stay single-threaded → no oversubscription).
function _band_to_band_fill!(::ComputationalBackends.SerialBackend, T, bws::BandTransferWorkspace, velocity_hat, ks;
                             dealiasing, invariant, spectral, advecting_hat)
    nd = length(ks); ns = size(velocity_hat)[1:nd]; D = size(velocity_hat, nd + 1)
    FT = real(eltype(velocity_hat)); nb = length(bws.centers)
    ws = bws.nlt; f_m = bws.f_m; d = bws.d; W = bws.W
    @inbounds for m in 1:nb
        for c in 1:D, I in CartesianIndices(ns)
            f_m[I, c] = W[m][I] * velocity_hat[I, c]
        end
        NonlinearTerm.compute_nonlinear_term!(ws, f_m, ks; dealiasing=dealiasing, spectral=spectral, advecting_hat=advecting_hat)
        Invariants.transfer_density!(d, invariant, velocity_hat, ws.N̂, ks)
        for n in 1:nb
            s = zero(FT)
            for I in CartesianIndices(ns)
                s += SpectralLayout.hermitian_weight(ks, I) * W[n][I] * d[I]
            end
            T[n, m] = s
        end
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
        invariant=KineticEnergy(), spectral=SpectralBackends.DirectSumSpectralBackend(), advecting_hat=velocity_hat,
        geometry=IsotropicShells())
        -> (centers, transfer_matrix, net_transfer, max_antisymmetry_error)

Smooth band-to-band transfer `T(n,m)` of a quadratic invariant between the graded spectral
`bands` (Eyink & Aluie 2009) — the smooth-filter analogue of `calculate_shell_to_shell_transfer`.
For incompressible flow `T` is antisymmetric (`T(n,m) = −T(m,n)`), conserves (`Σ T = 0`), and
`Σ_m T(n,m)` is the band-summed transfer spectrum. Accepts an `advecting_hat` (primary field can be
a passive scalar advected by the velocity) and an anisotropic `geometry`. See
[`calculate_band_to_band_transfer!`](@ref) for the in-place, workspace-reusing variant.
"""
function calculate_band_to_band_transfer(
    velocity_hat,
    ks;
    bands::Types.SmoothBands,
    dealiasing::Types.AbstractDealiasing = Types.OrszagTwoThirds(),
    invariant::Types.AbstractInvariant = Types.KineticEnergy(),
    spectral::SpectralBackends.AbstractSpectralBackend = SpectralBackends.AutoSpectralBackend(),
    execution::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.SerialBackend(),
    advecting_hat = velocity_hat,
    geometry::Types.AbstractShellGeometry = Types.IsotropicShells(),
)
    spectral = Types.resolve_spectral(Types.require_coefficient_spectral(spectral))
    FT = real(eltype(velocity_hat))
    nb = length(bands.centers)
    bws = BandTransferWorkspace(velocity_hat, ks, bands; geometry=geometry)
    T   = zeros(FT, nb, nb)
    net = zeros(FT, nb)
    return calculate_band_to_band_transfer!(T, net, bws, velocity_hat, ks;
        dealiasing=dealiasing, invariant=invariant, spectral=spectral, execution=execution,
        advecting_hat=advecting_hat)
end

"""
    calculate_band_to_band_transfer_batch(velocity_hats, ks; bands, dealiasing, invariant, spectral,
        geometry, execution) -> Vector

Band-to-band transfer for a batch of snapshots sharing one grid (`velocity_hats` an iterable of
`(ns..., D)` coefficient arrays). The band weights are built ONCE and each worker reuses one
`BandTransferWorkspace` across its snapshots; `execution = ThreadedBackend()` (requires `using
OhMyThreads`) threads over snapshots with a **serial** inner transform (the batch is the outer parallel
axis, no nested threading). Results (the same NamedTuple as the single-snapshot form) are in input
order, bit-identical to calling [`calculate_band_to_band_transfer`](@ref) per snapshot.
"""
function calculate_band_to_band_transfer_batch(
    velocity_hats,
    ks;
    bands::Types.SmoothBands,
    dealiasing::Types.AbstractDealiasing = Types.OrszagTwoThirds(),
    invariant::Types.AbstractInvariant = Types.KineticEnergy(),
    spectral::SpectralBackends.AbstractSpectralBackend = SpectralBackends.AutoSpectralBackend(),
    geometry::Types.AbstractShellGeometry = Types.IsotropicShells(),
    execution::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.SerialBackend(),
)
    spectral = Types.resolve_spectral(Types.require_coefficient_spectral(spectral))
    n       = length(velocity_hats)
    FT      = real(eltype(first(velocity_hats)))
    nb      = length(bands.centers)
    centers = collect(bands.centers)
    NT = @NamedTuple{centers::typeof(centers), transfer_matrix::Matrix{FT}, net_transfer::Vector{FT}, max_antisymmetry_error::FT}
    results = Vector{NT}(undef, n)
    n == 0 && return results
    _band_to_band_batch!(Types.resolve_execution(execution), results, velocity_hats, ks, bands, nb;
        dealiasing=dealiasing, invariant=invariant, spectral=spectral, geometry=geometry)
    return results
end

# Serial reference: one workspace reused across the whole batch; each snapshot self-advects.
function _band_to_band_batch!(::ComputationalBackends.AbstractSerialBackend, results, velocity_hats, ks, bands, nb;
                              dealiasing, invariant, spectral, geometry)
    FT  = real(eltype(first(velocity_hats)))
    bws = BandTransferWorkspace(first(velocity_hats), ks, bands; geometry=geometry)
    for i in eachindex(velocity_hats)
        vh = velocity_hats[i]
        T = zeros(FT, nb, nb); net = zeros(FT, nb)
        results[i] = calculate_band_to_band_transfer!(T, net, bws, vh, ks;
            dealiasing=dealiasing, invariant=invariant, spectral=spectral,
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

# Distributed over the batch — overridden by the Distributed extension (snapshots partitioned across workers).
function _band_to_band_batch_distributed!(args...; kwargs...)
    throw(ArgumentError("execution = DistributedBackend() for the band-to-band batch requires Distributed. " *
                        "Run `using Distributed` to load the extension."))
end
_band_to_band_batch!(::ComputationalBackends.AbstractDistributedBackend, results, velocity_hats, ks, bands, nb; kwargs...) =
    _band_to_band_batch_distributed!(results, velocity_hats, ks, bands, nb; kwargs...)

# GPU over the batch — loop the single-shot device kernel, reusing one device workspace (device inputs →
# device buffers via `similar`). Requires the FFT backend (DirectSum is a host-only reference).
function _band_to_band_batch!(gpu::ComputationalBackends.AbstractGPUBackend, results, velocity_hats, ks, bands, nb;
                              dealiasing, invariant, spectral, geometry)
    spectral isa SpectralBackends.DirectSumSpectralBackend && throw(ArgumentError(
        "calculate_band_to_band_transfer_batch on a GPUBackend requires spectral = SpectralBackends.FFTSpectralBackend() " *
        "(cuFFT via AbstractFFTs); SpectralBackends.DirectSumSpectralBackend is a host-only reference."))
    FT  = real(eltype(first(velocity_hats)))
    bws = BandTransferWorkspace(first(velocity_hats), ks, bands; geometry=geometry)
    for i in eachindex(velocity_hats)
        vh = velocity_hats[i]
        T = zeros(FT, nb, nb); net = zeros(FT, nb)
        results[i] = calculate_band_to_band_transfer!(T, net, bws, vh, ks;
            dealiasing=dealiasing, invariant=invariant, spectral=spectral, execution=gpu, advecting_hat=vh)
    end
    return results
end

# Any other execution backend has no batch hook — refuse rather than silently run serial.
_band_to_band_batch!(be::ComputationalBackends.AbstractExecutionBackend, results, velocity_hats, ks, bands, nb; kwargs...) =
    throw(ArgumentError("calculate_band_to_band_transfer_batch supports SerialBackend(), ThreadedBackend(), and DistributedBackend(); " *
                        "got execution = $(typeof(be))."))

# One-line show (the workspace holds a Workspaces.NonlinearTermWorkspace whose FFTW plan bundle can segfault
# under the default field-dump show).
Base.show(io::IO, ::BandTransferWorkspace) = print(io, "BandTransferWorkspace(…)")
Base.show(io::IO, ::MIME"text/plain", w::BandTransferWorkspace) = show(io, w)

end # module BandTransfer
