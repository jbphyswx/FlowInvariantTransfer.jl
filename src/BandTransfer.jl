module BandTransfer

using ..Types: SmoothBands, AbstractInvariant, KineticEnergy, AbstractDealiasing, OrszagTwoThirds, AbstractSpectralBackend,
               DirectSumBackend, AbstractShellGeometry, IsotropicShells
using ..Invariants: transfer_density!
using ..NonlinearTerm: compute_nonlinear_term!
using ..Workspaces: NonlinearTermWorkspace
using ..ShellBinning: shell_coordinate

export calculate_band_to_band_transfer, calculate_band_to_band_transfer!, BandTransferWorkspace

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

function BandTransferWorkspace(velocity_hat, ks, bands::SmoothBands;
                               geometry::AbstractShellGeometry = IsotropicShells())
    nd = length(ks)
    ns = size(velocity_hat)[1:nd]
    FT = real(eltype(velocity_hat))
    k_coord = shell_coordinate(geometry, ks)
    W = _band_weights(bands.centers, bands.logwidth, k_coord)
    return BandTransferWorkspace(NonlinearTermWorkspace(velocity_hat, ks),
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
    dealiasing::AbstractDealiasing = OrszagTwoThirds(),
    invariant::AbstractInvariant = KineticEnergy(),
    spectral::AbstractSpectralBackend = DirectSumBackend(),
    advecting_hat = velocity_hat,
)
    nd = length(ks)
    ns = size(velocity_hat)[1:nd]
    D  = size(velocity_hat, nd + 1)
    FT = real(eltype(velocity_hat))
    nb = length(bws.centers)
    ws = bws.nlt; f_m = bws.f_m; d = bws.d; W = bws.W

    @inbounds for m in 1:nb
        # band-m field f_m = w_m ⊙ (primary field), advected by the full velocity
        for c in 1:D, I in CartesianIndices(ns)
            f_m[I, c] = W[m][I] * velocity_hat[I, c]
        end
        compute_nonlinear_term!(ws, f_m, ks; dealiasing=dealiasing, spectral=spectral, advecting_hat=advecting_hat)
        transfer_density!(d, invariant, velocity_hat, ws.N̂, ks)
        for n in 1:nb
            s = zero(FT)
            for I in CartesianIndices(ns)
                s += W[n][I] * d[I]
            end
            T[n, m] = s
        end
    end

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

"""
    calculate_band_to_band_transfer(velocity_hat, ks; bands::SmoothBands, dealiasing=OrszagTwoThirds(),
        invariant=KineticEnergy(), spectral=DirectSumBackend(), advecting_hat=velocity_hat,
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
    bands::SmoothBands,
    dealiasing::AbstractDealiasing = OrszagTwoThirds(),
    invariant::AbstractInvariant = KineticEnergy(),
    spectral::AbstractSpectralBackend = DirectSumBackend(),
    advecting_hat = velocity_hat,
    geometry::AbstractShellGeometry = IsotropicShells(),
)
    FT = real(eltype(velocity_hat))
    nb = length(bands.centers)
    bws = BandTransferWorkspace(velocity_hat, ks, bands; geometry=geometry)
    T   = zeros(FT, nb, nb)
    net = zeros(FT, nb)
    return calculate_band_to_band_transfer!(T, net, bws, velocity_hat, ks;
        dealiasing=dealiasing, invariant=invariant, spectral=spectral, advecting_hat=advecting_hat)
end

end # module BandTransfer
