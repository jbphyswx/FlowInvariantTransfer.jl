module Compressible

using ..Types: Types
using ..SpectralLayout: SpectralLayout
using ComputationalBackends: ComputationalBackends
using SpectralBackends: SpectralBackends
using ..ShellBinning: ShellBinning

export calculate_compressible_flux, calculate_compressible_flux!, calculate_compressible_flux_batch,
       CompressibleWorkspace

# ---------------------------------------------------------------------------
# Compressible kinetic-energy spectral transfer (Singh–Tiwari–Sharma–Verma 2025,
# arXiv:2508.04300). Framework A: momentum v = ρu,
# KE E_u(k) = ½Re[v(k)·u*(k)]. The nonlinear transfer is momentum-weighted and conserves
# total KE (Σ_k T_u = 0); the KE↔internal-energy exchange is the *separate* pressure-dilatation
# term Q_{I}, gated on a supplied pressure field.
#
# Per-mode transfer, reduced from the scale-to-scale form (paper Eq. 20/28) to a pseudospectral
# O(Nᴰ) expression. The nonlinear terms give ∂_t v̂ = −𝒩̂₁ and ∂_t û = −𝒩̂₂, so T_u(k) = −dE_u(k)/dt
# (positive when mode k gives energy to other modes); for ρ = const and ∇·u = 0,
# T_u = ρ·Re{û*·(u·∇)u}, ρ × the incompressible transfer_spectrum:
#
#     T_u(k) = ½ Re{ û*(k)·𝒩̂₁(k) } + ½ Re{ v̂*(k)·𝒩̂₂(k) }
#     𝒩₁ = (u·∇)v + v(∇·u) = ∂_j(v ⊗ u)_j ,   𝒩₂ = (u·∇)u ,   v = ρu.
#
# This reference works entirely by explicit DFT/IDFT (dependency-free, exact), mirroring the
# SpectralBackends.DirectSumSpectralBackend philosophy of the incompressible path; small grids only, correctness-first.
# ---------------------------------------------------------------------------

# Direct transforms on the spatial dimensions of an (ms..., C) coefficient / (ns..., C) physical field.
# Convention matches NonlinearTerm.jl: analysis Σ_x f(x) e^{-i k·x}/N, synthesis Σ_k f̂(k) e^{+i k·x},
# with x_d = (I_d-1)/n_d and integer wavenumber index k_d.
#
# The physical side of every transform below is REAL: these fields (velocity, density, momentum,
# their gradients, the nonlinear terms) are real by construction, and the synthesis of a Hermitian
# spectrum is real. Synthesis sums the stored coefficients with the Hermitian weight, which equals
# the full-spectrum sum on either layout (`SpectralLayout.hermitian_weight`).
@inline function _synth_at(field_hat, c, ks, xI, ns::NTuple{nd,Int}, ms::NTuple{nd,Int}, kfac) where {nd}
    FT = real(eltype(field_hat))
    acc = zero(complex(FT))
    @inbounds for kI in CartesianIndices(ms)
        phase = zero(FT)
        for d in 1:nd
            phase += FT(2π) * SpectralLayout.axis_index_wavenumber(ks[d], kI[d]) *
                     FT(xI[d] - 1) / FT(ns[d])
        end
        acc += SpectralLayout.hermitian_weight(ks, kI) * kfac(kI) * field_hat[kI, c] * cis(phase)
    end
    return real(acc)
end

# In-place explicit synthesis/analysis writing into `out` (no allocation) — for the workspace path.
function _idft!(out, field_hat, ks, ns::NTuple{nd,Int}, ms::NTuple{nd,Int}) where {nd}
    C = size(field_hat, nd + 1)
    one_ = one(real(eltype(field_hat)))
    @inbounds for c in 1:C, xI in CartesianIndices(ns)
        out[xI, c] = _synth_at(field_hat, c, ks, xI, ns, ms, _ -> one_)
    end
    return out
end

function _dft!(out, field_phys, ks, ns::NTuple{nd,Int}, ms::NTuple{nd,Int}) where {nd}
    FT = real(eltype(field_phys))
    C  = size(field_phys, nd + 1)
    Np = prod(ns)
    @inbounds for c in 1:C, kI in CartesianIndices(ms)
        acc = zero(complex(FT))
        for xI in CartesianIndices(ns)
            phase = zero(FT)
            for d in 1:nd
                phase += FT(2π) * SpectralLayout.axis_index_wavenumber(ks[d], kI[d]) *
                         FT(xI[d] - 1) / FT(ns[d])
            end
            acc += field_phys[xI, c] * cis(-phase)
        end
        out[kI, c] = acc / FT(Np)
    end
    return out
end

# In-place gradient (folds i k_d into the synthesis sum) writing into g (ns...,C,nd). `kd` holds each
# axis's derivative wavenumbers (`SpectralLayout.wavenumber_arrays(...; derivative = true)`, or the
# padded-grid table), zero where the grid derivative of a mode vanishes.
function _grad_phys!(g, field_hat, ks, ns::NTuple{nd,Int}, ms::NTuple{nd,Int}, kd) where {nd}
    C = size(field_hat, nd + 1)
    @inbounds for d in 1:nd, c in 1:C, xI in CartesianIndices(ns)
        g[xI, c, d] = _synth_at(field_hat, c, ks, xI, ns, ms, kI -> im * kd[d][kI[d]])
    end
    return g
end

# ---------------------------------------------------------------------------
# Transform context — swappable analysis/synthesis/gradient primitives so the physics assembly is
# written once and the transform algorithm is chosen by the spectral backend: the core provides the
# dependency-free explicit-DFT context (`SpectralBackends.DirectSumSpectralBackend`), and the FlowInvariantTransferFFTWExt
# extension provides the O(Nᵈ log Nᵈ) FFT context (`SpectralBackends.FFTSpectralBackend`), reusing preplanned FFTs + scratch.
#
#   tf.idft(field_hat)  : spectral (ms...,C) → physical (ns...,C) REAL   (synthesis, u = Σ û e^{ik·x})
#   tf.dft(field_phys)  : physical (ns...,C) REAL → spectral (ms...,C)   (analysis, û = fft/Nᵈ)
#   tf.grad(field_hat)  : spectral (ms...,C) → physical gradients (ns...,C,nd) REAL
#
# The physical side is real throughout: these fields are real by construction and the synthesis of a
# Hermitian spectrum is real, so nothing is carried as a complex "physical" array and no `real(...)`
# pass follows a transform. `ms` is the coefficient shape and `ns` the grid; they differ on the half
# layout.
# ---------------------------------------------------------------------------
#   In-place siblings write into a caller buffer (for the workspace path): tf.idft!(out, fh),
#   tf.dft!(out, fp), tf.grad!(g, fh).
struct TransformContext{ID, DF, GR, IDB, DFB, GRB}
    idft::ID
    dft::DF
    grad::GR
    idft!::IDB
    dft!::DFB
    grad!::GRB
end

function _directsum_tf(ks, ns::NTuple{nd,Int}, kd) where {nd}
    ms = SpectralLayout.spectral_size(ks)
    FT = float(eltype(ks[1]))
    CT = complex(FT)
    return TransformContext(
        fh -> _idft!(Array{FT}(undef, ns..., size(fh, nd + 1)), fh, ks, ns, ms),
        fp -> _dft!(Array{CT}(undef, ms..., size(fp, nd + 1)), fp, ks, ns, ms),
        fh -> _grad_phys!(Array{FT}(undef, ns..., size(fh, nd + 1), nd), fh, ks, ns, ms, kd),
        (out, fh) -> _idft!(out, fh, ks, ns, ms),
        (out, fp) -> _dft!(out, fp, ks, ns, ms),
        (g, fh) -> _grad_phys!(g, fh, ks, ns, ms, kd),
    )
end

# The FFT context is provided by FlowInvariantTransferFFTWExt; this fallback (less specific than the
# extension's method) gives a clear error when a non-DirectSum backend is requested without FFTW.
# `fft_nthreads` pins the FFTW plan thread count (baked into the plan → no per-call scratch alloc); the
# threaded execution path passes `> 1` so the single-pipeline FFTs run multithreaded (no outer loop).
_fft_tf(velocity_hat, ks, ns, kd; fft_nthreads::Int = 1) = throw(ArgumentError(
    "calculate_compressible_flux with an FFT backend requires `using FFTW`; " *
    "or pass `spectral = SpectralBackends.DirectSumSpectralBackend()` for the dependency-free (slow, small-grid) path."))

# Explicit per-backend transform contexts — NO `::SpectralBackends.AbstractSpectralBackend` catch-all: a catch-all
# silently routed the scattered/spherical backends through the FFT context (wrong answers, no error).
# The public entry validates the backend (`require_coefficient_spectral`), so only DirectSum/FFT reach here.
_resolve_tf(::SpectralBackends.DirectSumSpectralBackend, proto, ks, ns, kd; fft_nthreads::Int = 1) =
    _directsum_tf(ks, ns, kd)
_resolve_tf(::SpectralBackends.FFTSpectralBackend, proto, ks, ns, kd; fft_nthreads::Int = 1) =
    _fft_tf(proto, ks, ns, kd; fft_nthreads = fft_nthreads)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

"""
    CompressibleWorkspace

Reusable field-scratch + transform context for [`calculate_compressible_flux!`](@ref) — holds the
FFT plans and every grid-sized intermediate of the momentum-weighted budget (velocity/density/
momentum physical & spectral fields, gradients, nonlinear terms, and the R/C-channel + pressure-
dilatation scratch), so repeated per-snapshot calls allocate ~0 field memory (only the small shell
vectors of each result). Build once for a given grid/precision and reuse across snapshots.

The products are formed on the working grid: the caller's grid, or under `PaddedThreeHalves` the grid
padded for the transfer's cubic products. The densities are taken on the caller's grid, from the
working grid's spectral outputs (`coarse`, the same arrays unless padded).
"""
struct CompressibleWorkspace{ND, DA, KS, KD, TF, CA, CS, RA, RS, RG, CH, PD, PM, CO, RM, SI, CE}
    dealiasing::DA
    ks::KS; kd::KD; kdc::KD        # working-grid axes and derivative wavenumbers; the caller grid's
    tf::TF
    vel::CA; ρh::CS; v̂::CA; N̂1::CA; N̂2::CA
    u_phys::RA; v_phys::RA; N1_phys::RA; N2_phys::RA
    ρ_phys::RS; divu::RS
    gradu::RG; gradv::RG
    channels::CH                  # R/C-channel scratch, or `nothing` when the workspace excludes them
    pressure::PD                  # pressure-dilatation scratch, or `nothing`
    maps::PM                      # `SpectralLayout.PaddedMaps` under padding, else `nothing`
    coarse::CO                    # (v̂, N̂1, N̂2) on the caller's grid
    td::RM                        # per-mode density on the caller's grid
    sidx::SI; centers::CE         # shell structure of the caller's grid
    ns::NTuple{ND, Int}           # working physical grid
    ms::NTuple{ND, Int}           # working coefficient grid (differs from `ns` on the half layout)
end

# Buffers used only by the Helmholtz R/C channel decomposition: the working-grid splits and giver
# fields, and `rc`, the receivers and the givers' nonlinear terms on the caller's grid.
struct CompressibleChannels{CA, RA, RC}
    sscr::CA
    ûR::CA; ûC::CA; v̂R::CA; v̂C::CA
    uR::RA; uC::RA; vR::RA; vC::RA
    N̂1R::CA; N̂2R::CA; N̂1C::CA; N̂2C::CA
    rc::RC
end

# Buffers used only by the KE↔IE pressure-dilatation term; `σ̃c`, `σhc`, `kdotuC` and `qscr` are on the
# caller's grid.
struct CompressiblePressure{CS, RG1, RA, CA, CCA, CCS, C1, RM}
    σh::CS; gradσ::RG1; σ̃phys::RA; σ̃::CA
    σ̃c::CCA; σhc::CCS
    kdotuC::C1
    qscr::RM
end

"""
    CompressibleWorkspace(velocity_hat, ks; spectral, binning, geometry, execution,
                          dealiasing = OrszagTwoThirds(), decompose = true, with_pressure = false)

`dealiasing` fixes the working grid. The transfer's products are cubic (`ρ·u·∇u`), so
`OrszagTwoThirds()` keeps `|m_d| ≤ ⌊(N_d−1)/4⌋` (`SpectralLayout.dealias_cutoff(N, 3)`), where every
term is exact and `Σ_k T_u = 0`; `PaddedThreeHalves()` forms the products on the grid padded for them
(`SpectralLayout.padded_length(N, 3)`) and returns every spectral output to the caller's grid;
`NoDealiasing()` takes the raw products. `decompose` and `with_pressure` fix which optional terms the
workspace can compute, and their scratch is allocated only when selected.
"""
function CompressibleWorkspace(velocity_hat, ks;
                               spectral::SpectralBackends.AbstractSpectralBackend = SpectralBackends.AutoSpectralBackend(),
                               binning::Types.AbstractShellBinning = _default_binning(ks),
                               geometry::Types.AbstractShellGeometry = Types.IsotropicShells(),
                               execution::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.SerialBackend(),
                               dealiasing::Types.AbstractDealiasing = Types.OrszagTwoThirds(),
                               decompose::Bool = true,
                               with_pressure::Bool = false)
    spectral = Types.resolve_spectral(Types.require_coefficient_spectral(spectral))
    nd = length(ks)
    mc = SpectralLayout.spectral_size(ks)      # the caller's coefficient grid
    FT = real(eltype(velocity_hat)); CT = complex(FT)
    size(velocity_hat)[1:nd] == mc || throw(DimensionMismatch(
        "field spatial size $(size(velocity_hat)[1:nd]) does not match the wavenumber grid $mc."))
    # Compressible is a single ~O(nd) FFT pipeline (not an outer loop), so ComputationalBackends.ThreadedBackend threads the
    # FFTs themselves — plans baked at nthreads (no per-call scratch alloc, no oversubscription).
    fft_nthreads = execution isa ComputationalBackends.ThreadedBackend ? Threads.nthreads() : 1
    padded = dealiasing isa Types.PaddedThreeHalves
    Ms  = padded ? map(n -> SpectralLayout.padded_length(n, 3), SpectralLayout.full_size(ks)) :
                   SpectralLayout.full_size(ks)
    ksw = padded ? SpectralLayout.padded_axes(ks, Ms) : ks
    ns  = SpectralLayout.full_size(ksw); ms = SpectralLayout.spectral_size(ksw)
    kdc = SpectralLayout.wavenumber_arrays(velocity_hat, FT, ks; derivative = true)
    kd  = padded ? SpectralLayout.padded_derivative_wavenumbers(velocity_hat, FT, ks, Ms) : kdc
    maps = padded ? SpectralLayout.PaddedMaps(velocity_hat, FT, ks, Ms) : nothing
    # Buffers built in `velocity_hat`'s own array type → device-resident for device input (the whole
    # pipeline is device-generic broadcasts), plain `Array`s for host input. Coefficient buffers are
    # sized by `ms`, physical ones by `ns`, and every physical buffer is REAL.
    ca()  = similar(velocity_hat, CT, ms..., nd)
    cs()  = similar(velocity_hat, CT, ms..., 1)
    ra()  = similar(velocity_hat, FT, ns..., nd)
    rs()  = similar(velocity_hat, FT, ns...)
    rg()  = similar(velocity_hat, FT, ns..., nd, nd)
    rg1() = similar(velocity_hat, FT, ns..., 1, nd)
    # On the caller's grid, or the working array itself when the two grids are one.
    onc(w, k...) = padded ? similar(velocity_hat, CT, mc..., k...) : w
    # Shell structure of the caller's grid, hoisted so the per-snapshot `!` reallocates nothing.
    k_mag   = ShellBinning.shell_coordinate(geometry, ks)
    kmax    = ShellBinning.max_shell_coordinate(geometry, ks)
    edges   = ShellBinning.shell_edges(binning, kmax)
    centers = collect(ShellBinning.shell_centers(binning, kmax))
    sidx    = ShellBinning.assign_shells(k_mag, edges)
    vel = ca(); v̂ = ca(); N̂1 = ca(); N̂2 = ca()
    coarse = (v̂ = onc(v̂, nd), N̂1 = onc(N̂1, nd), N̂2 = onc(N̂2, nd))
    # The pressure-dilatation term projects the velocity and momentum onto the rotational/compressive
    # split and reads it out of this scratch, so it is allocated for either term.
    channels = if decompose || with_pressure
        w = (ûR = ca(), ûC = ca(), v̂R = ca(), v̂C = ca(), N̂1R = ca(), N̂2R = ca(), N̂1C = ca(), N̂2C = ca())
        CompressibleChannels(ca(), w.ûR, w.ûC, w.v̂R, w.v̂C, ra(), ra(), ra(), ra(),
                             w.N̂1R, w.N̂2R, w.N̂1C, w.N̂2C, map(a -> onc(a, nd), w))
    else
        nothing
    end
    pressure = if with_pressure
        σh = cs(); σ̃ = ca()
        CompressiblePressure(σh, rg1(), ra(), σ̃, onc(σ̃, nd), onc(σh, 1),
                             similar(velocity_hat, CT, mc...), similar(velocity_hat, FT, mc...))
    else
        nothing
    end
    return CompressibleWorkspace(
        dealiasing, ksw, kd, kdc,
        _resolve_tf(spectral, velocity_hat, ksw, ns, kd; fft_nthreads = fft_nthreads),
        vel, cs(), v̂, N̂1, N̂2,
        ra(), ra(), ra(), ra(),
        rs(), rs(),
        rg(), rg(),
        channels, pressure, maps, coarse,
        similar(velocity_hat, FT, mc...),
        sidx, centers, ns, ms)
end

"""
    calculate_compressible_flux(velocity_hat, density_hat, ks; binning, pressure_hat=nothing,
        decompose=true, geometry=IsotropicShells(), spectral=SpectralBackends.FFTSpectralBackend()) -> CompressibleFluxResult

Compressible kinetic-energy spectral transfer `T_u(k)` and cumulative flux `Π(K)` (Singh–Tiwari–
Sharma–Verma 2025): momentum `v = ρu`, `E_u(k) = ½Re[v·u*]`; the nonlinear transfer conserves total KE
(`Σ_k T_u ≈ 0`), and the KE↔internal-energy pressure-dilatation is returned separately when
`pressure_hat` is supplied. `decompose=true` also returns the Helmholtz rotational/compressive flux
channels. `T_u(k) = −dE_u(k)/dt` from the nonlinear terms (positive when shell `k` gives energy to
other shells) and `Π(K) = Σ_{k≤K} T_u(k)`; in the incompressible limit `T_u` reduces to `ρ ×` the
incompressible transfer spectrum.
This allocates a [`CompressibleWorkspace`](@ref) and delegates to [`calculate_compressible_flux!`](@ref);
build the workspace once and use the in-place form to loop over snapshots allocation-free.
"""
function calculate_compressible_flux(
    velocity_hat,
    density_hat,
    ks;
    spectral::SpectralBackends.AbstractSpectralBackend = SpectralBackends.AutoSpectralBackend(),
    binning::Types.AbstractShellBinning = _default_binning(ks),
    geometry::Types.AbstractShellGeometry = Types.IsotropicShells(),
    execution::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.SerialBackend(),
    kwargs...,
)
    nd = length(ks)
    size(velocity_hat, nd + 1) == nd ||
        throw(ArgumentError("compressible transfer needs D = nd velocity components (got $(size(velocity_hat, nd+1)) for nd=$nd)."))
    exec = Types.resolve_execution(execution)
    # Which optional terms a workspace can compute is fixed when it is built, so the one-shot sizes it
    # from what this call asks for: a supplied `pressure_hat` needs the pressure scratch, `decompose`
    # the channel set. The batch entry derives them the same way.
    with_pressure = get(kwargs, :pressure_hat, nothing) !== nothing
    decompose = get(kwargs, :decompose, true)
    dealiasing = get(kwargs, :dealiasing, Types.OrszagTwoThirds())
    if exec isa ComputationalBackends.DistributedBackend
        # Compressible is a single FFT pipeline (no outer loop): its independent work units are the
        # decomposition-channel set and the pressure-dilatation, computed on separate workers (each
        # rebuilding its own workspace from the raw inputs) and assembled on the master. Overridden by
        # the Distributed extension; the core stub errors clearly if that ext isn't loaded.
        return _compressible_distributed(velocity_hat, density_hat, ks, exec;
            spectral=spectral, binning=binning, geometry=geometry, kwargs...)
    elseif exec isa ComputationalBackends.GPUBackend
        # The compressible pipeline is device-generic (broadcasts + cuFFT via AbstractFFTs) with no KA
        # kernel, so the device path is selected by the INPUT ARRAY TYPE, not this knob. A host `Array`
        # under ComputationalBackends.GPUBackend cannot be honoured (no data movement, no separate device kernel) → clear error
        # rather than a silent serial run; a device-array input runs on device by construction.
        Types._is_device(velocity_hat) || throw(ArgumentError(
            "compressible transfer runs on-device automatically for device-array inputs (the pipeline is " *
            "device-generic broadcasts + cuFFT via AbstractFFTs); execution=ComputationalBackends.GPUBackend() does not move a host " *
            "array to the device. Pass device-array inputs (e.g. CuArray), or use ComputationalBackends.SerialBackend()/ComputationalBackends.ThreadedBackend()."))
        ws = CompressibleWorkspace(velocity_hat, ks; spectral=spectral, binning=binning, geometry=geometry,
                                   execution=ComputationalBackends.SerialBackend(), dealiasing=dealiasing,
                                   decompose=decompose, with_pressure=with_pressure)
        return calculate_compressible_flux!(ws, velocity_hat, density_hat, ks; kwargs...)
    end
    ws = CompressibleWorkspace(velocity_hat, ks; spectral=spectral, binning=binning, geometry=geometry,
                               execution=exec, dealiasing=dealiasing, decompose=decompose,
                               with_pressure=with_pressure)
    return calculate_compressible_flux!(ws, velocity_hat, density_hat, ks; kwargs...)
end

# Overridden by the Distributed extension (requires `using Distributed`).
_compressible_distributed(args...; kwargs...) = throw(ArgumentError(
    "Distributed compressible transfer requires Distributed. " *
    "Run `using Distributed` to load the extension."))

"""
    calculate_compressible_flux_batch(velocity_hats, density_hats, ks; pressure_hats=nothing,
        spectral, binning, geometry, dealiasing, decompose, execution) -> Vector

Compressible transfer for a batch of snapshots sharing one grid (`velocity_hats`/`density_hats`
iterables of coefficient arrays). One [`CompressibleWorkspace`](@ref) is built per worker and reused
across its snapshots — this is the workspace the package allocates most for, so a time series pays for
it once. `execution = ThreadedBackend()` (requires `using OhMyThreads`) threads over snapshots with a
serial inner pipeline. Results are in input order and equal the per-snapshot
[`calculate_compressible_flux`](@ref).
"""
function calculate_compressible_flux_batch(
    velocity_hats,
    density_hats,
    ks;
    pressure_hats = nothing,
    spectral::SpectralBackends.AbstractSpectralBackend = SpectralBackends.AutoSpectralBackend(),
    binning::Types.AbstractShellBinning = _default_binning(ks),
    geometry::Types.AbstractShellGeometry = Types.IsotropicShells(),
    dealiasing::Types.AbstractDealiasing = Types.OrszagTwoThirds(),
    decompose::Bool = true,
    execution::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.SerialBackend(),
)
    spectral = Types.resolve_spectral(Types.require_coefficient_spectral(spectral))
    n = length(velocity_hats)
    length(density_hats) == n || throw(DimensionMismatch(
        "got $n velocity snapshots and $(length(density_hats)) density snapshots"))
    pressure_hats === nothing || length(pressure_hats) == n || throw(DimensionMismatch(
        "got $n velocity snapshots and $(length(pressure_hats)) pressure snapshots"))
    n == 0 && return Types.CompressibleFluxResult[]
    return _compressible_batch!(Types.resolve_execution(execution), velocity_hats, density_hats,
                                pressure_hats, ks; spectral = spectral, binning = binning,
                                geometry = geometry, dealiasing = dealiasing, decompose = decompose)
end

# Serial reference: one workspace reused across the whole batch.
function _compressible_batch!(::ComputationalBackends.AbstractSerialBackend, velocity_hats, density_hats,
                              pressure_hats, ks; spectral, binning, geometry, dealiasing, decompose)
    ws = CompressibleWorkspace(first(velocity_hats), ks; spectral = spectral, binning = binning,
                               geometry = geometry, dealiasing = dealiasing, decompose = decompose,
                               with_pressure = pressure_hats !== nothing)
    return [calculate_compressible_flux!(ws, velocity_hats[i], density_hats[i], ks;
                pressure_hat = pressure_hats === nothing ? nothing : pressure_hats[i],
                dealiasing = dealiasing, decompose = decompose)
            for i in eachindex(velocity_hats)]
end

# Threaded over the batch — overridden by the OhMyThreads extension (per-chunk workspace, serial inner).
function _compressible_batch_threaded!(args...; kwargs...)
    throw(ArgumentError("execution = ThreadedBackend() for the compressible batch requires OhMyThreads. " *
                        "Run `using OhMyThreads` to load the extension."))
end
_compressible_batch!(::ComputationalBackends.AbstractThreadedBackend, velocity_hats, density_hats,
                     pressure_hats, ks; kwargs...) =
    _compressible_batch_threaded!(velocity_hats, density_hats, pressure_hats, ks; kwargs...)

# A device batch loops the device pipeline on one device-resident workspace.
function _compressible_batch!(gpu::ComputationalBackends.AbstractGPUBackend, velocity_hats, density_hats,
                              pressure_hats, ks; spectral, binning, geometry, dealiasing, decompose)
    spectral isa SpectralBackends.DirectSumSpectralBackend && throw(ArgumentError(
        "calculate_compressible_flux_batch on a GPUBackend requires spectral = SpectralBackends.FFTSpectralBackend() " *
        "(cuFFT via AbstractFFTs); SpectralBackends.DirectSumSpectralBackend is a host-only reference."))
    return _compressible_batch!(ComputationalBackends.SerialBackend(), velocity_hats, density_hats,
                                pressure_hats, ks; spectral = spectral, binning = binning,
                                geometry = geometry, dealiasing = dealiasing, decompose = decompose)
end

_compressible_batch!(be::ComputationalBackends.AbstractExecutionBackend, velocity_hats, density_hats,
                     pressure_hats, ks; kwargs...) =
    throw(ArgumentError("calculate_compressible_flux_batch supports SerialBackend(), ThreadedBackend() " *
                        "and GPUBackend(); got execution = $(typeof(be))."))

"""
    calculate_compressible_flux!(ws::CompressibleWorkspace, velocity_hat, density_hat, ks; kwargs...)

In-place momentum-weighted compressible transfer reusing `ws` (FFT plans + all field scratch). Returns
a fresh [`CompressibleFluxResult`](@ref) whose arrays are the small per-shell vectors; the O(field)
intermediates live in `ws` and are reused across calls (build the workspace once, loop over snapshots).
"""
function calculate_compressible_flux!(
    ws::CompressibleWorkspace,
    velocity_hat,
    density_hat,
    ks;
    pressure_hat = nothing,
    dealiasing::Types.AbstractDealiasing = ws.dealiasing,
    decompose::Bool = true,
)
    nd = length(ks)
    ns = ws.ns          # working physical grid
    mc = SpectralLayout.spectral_size(ks)   # the caller's coefficient grid
    FT = real(eltype(velocity_hat))
    tf = ws.tf
    colons = ntuple(_ -> Colon(), nd)   # component-slice views for device-generic broadcasts
    typeof(dealiasing) === typeof(ws.dealiasing) || throw(ArgumentError(
        "this CompressibleWorkspace was built for $(nameof(typeof(ws.dealiasing))); rebuild it with " *
        "dealiasing = $(nameof(typeof(dealiasing)))()."))
    decompose && ws.channels === nothing && throw(ArgumentError(
        "this CompressibleWorkspace was built with decompose = false; rebuild it with decompose = true " *
        "to compute the Helmholtz R/C channels."))
    pressure_hat !== nothing && ws.pressure === nothing && throw(ArgumentError(
        "this CompressibleWorkspace was built with with_pressure = false; rebuild it with " *
        "with_pressure = true to compute the pressure-dilatation term."))

    # Inputs on the working grid: truncated to the kept band, embedded in the padded grid, or copied.
    _load!(ws.vel, velocity_hat, ws, ks)
    _load!(ws.ρh, _as_scalar(density_hat, mc), ws, ks)

    # Physical fields  u = idft(û),  ρ = idft(ρ̂),  v = ρu,  v̂ = dft(v). Every synthesis lands in a
    # real buffer, so there is no complex "physical" array and no `real(...)` pass after a transform.
    tf.idft!(ws.u_phys, ws.vel)
    tf.idft!(reshape(ws.ρ_phys, ns..., 1), ws.ρh)
    ws.v_phys .= reshape(ws.ρ_phys, ns..., 1) .* ws.u_phys
    tf.dft!(ws.v̂, ws.v_phys)

    # Gradients of u, v and the divergence ∇·u
    tf.grad!(ws.gradu, ws.vel)
    tf.grad!(ws.gradv, ws.v̂)
    fill!(ws.divu, zero(FT))
    for d in 1:nd
        ws.divu .+= view(ws.gradu, colons..., d, d)
    end

    # 𝒩₁ = (u·∇)v + v(∇·u) ;  𝒩₂ = (u·∇)u ;  then N̂₁,N̂₂
    _assemble_N!(ws.N1_phys, ws.N2_phys, ws.u_phys, ws.v_phys, ws.gradu, ws.gradv, ws.divu, ns, nd)
    tf.dft!(ws.N̂1, ws.N1_phys)
    tf.dft!(ws.N̂2, ws.N2_phys)

    # The spectral outputs on the caller's grid, where the densities are taken.
    co = ws.coarse
    _unload!(co.v̂, ws.v̂, ws); _unload!(co.N̂1, ws.N̂1, ws); _unload!(co.N̂2, ws.N̂2, ws)
    ûc = ws.maps === nothing ? ws.vel : velocity_hat

    # Per-mode transfer T_u(k) = −dE_u(k)/dt = ½Re{û*·𝒩̂₁} + ½Re{v̂*·𝒩̂₂}
    _density!(ws.td, ûc, co.v̂, co.N̂1, co.N̂2)

    # Shell binning — precomputed in the workspace (0-alloc across snapshots)
    sidx    = ws.sidx
    centers = ws.centers
    N_sh    = length(centers)

    T_spec = _bin(ws.td, sidx, N_sh, FT, ks, mc)
    flux   = _flux_from_transfer(T_spec)

    channels = decompose ? _rc_channels!(ws, ûc, ks, mc, sidx, N_sh, FT) : nothing
    pdil = nothing
    if pressure_hat !== nothing
        pr = ws.pressure
        _load!(pr.σh, _as_scalar(pressure_hat, mc), ws, ks)
        σhc = ws.maps === nothing ? pr.σh : _as_scalar(pressure_hat, mc)
        pdil = _pressure_dilatation!(ws, ûc, σhc, ks, mc, sidx, N_sh, FT)
    end

    return Types.CompressibleFluxResult(centers, T_spec, flux, channels, pdil)
end

# `td = ½Re{û*·N̂₁} + ½Re{v̂*·N̂₂}` summed over components, on the caller's grid. `td`'s rank fixes the
# component-slice views at compile time.
function _density!(td::AbstractArray{<:Any, nd}, û, v̂, N̂1, N̂2) where {nd}
    colons = ntuple(_ -> Colon(), Val(nd))
    fill!(td, zero(eltype(td)))
    for c in 1:size(û, nd + 1)
        td .+= real.(conj.(view(û, colons..., c)) .* view(N̂1, colons..., c) .+
                     conj.(view(v̂, colons..., c)) .* view(N̂2, colons..., c))
    end
    td .*= eltype(td)(0.5)
    return td
end

# Coefficients of the caller's grid onto the working grid: embedded in the padded grid, truncated to
# the band the cubic products keep alias-free, or copied.
function _load!(dst, src, ws, ks)
    ws.maps === nothing || return SpectralLayout.padded_embed!(dst, src, ws.maps)
    return _copy_trunc!(dst, src, ks, ws.ms, ws.dealiasing isa Types.OrszagTwoThirds)
end

# The working grid's coefficients returned to the caller's grid; the same array when they are one.
_unload!(dst, src, ws) = ws.maps === nothing ? dst : SpectralLayout.padded_truncate!(dst, src, ws.maps)

# Copy `src` (ms...,C) into `dst`, zeroing the modes outside the band the transfer's cubic products keep
# alias-free (`SpectralLayout.dealias_cutoff(n, 3)` on every axis) if `trunc`.
function _copy_trunc!(dst, src, ks, ms::NTuple{nd,Int}, trunc::Bool) where {nd}
    C = size(src, nd + 1)
    @inbounds for c in 1:C, I in CartesianIndices(ms)
        dst[I, c] = (trunc && SpectralLayout.is_dealiased(ks, I, 3)) ? zero(eltype(dst)) : src[I, c]
    end
    return dst
end

# 𝒩₁ = (u·∇)v + v(∇·u), 𝒩₂ = (u·∇)u  (physical), into preallocated N1,N2. Device-generic: component
# slices are broadcast/accumulated (no scalar indexing), so this runs on CPU `Array`s and device arrays.
function _assemble_N!(N1, N2, u_phys, v_phys, gradu, gradv, divu, ns::NTuple{nd,Int}, nd_::Int) where {nd}
    colons = ntuple(_ -> Colon(), nd)
    @inbounds for c in 1:nd_
        N1c = view(N1, colons..., c); N2c = view(N2, colons..., c)
        fill!(N1c, zero(eltype(N1))); fill!(N2c, zero(eltype(N2)))
        for d in 1:nd_
            N1c .+= view(u_phys, colons..., d) .* view(gradv, colons..., c, d)
            N2c .+= view(u_phys, colons..., d) .* view(gradu, colons..., c, d)
        end
        N1c .+= view(v_phys, colons..., c) .* divu
    end
    return nothing
end

# ---------------------------------------------------------------------------
# Rotational/compressive flux channels (paper Eqs. 52–57).
# We form the transfer with the *receiver* field split into R/C (û_R*, v̂_R*, û_C*, v̂_C*) and the
# nonlinear term built from the R/C-filtered *giver* momentum. Each channel is shell-binned and
# accumulated into a flux; the four channels sum to the total flux (validated in tests), and in the
# incompressible limit only the rotational channel survives (paper Eqs. 48–50).
# ---------------------------------------------------------------------------
function _rc_channels!(ws, ûc, ks, mc::NTuple{nd,Int}, sidx, N_sh, ::Type{FT}) where {nd, FT}
    tf = ws.tf
    ch = ws.channels
    rc = ch.rc
    ns = ws.ns
    # Givers on the working grid, split with the working grid's derivative wavenumbers.
    _helmholtz_split!(ch.ûR, ch.ûC, ws.vel, ws.kd, ws.ms)
    _helmholtz_split!(ch.v̂R, ch.v̂C, ws.v̂, ws.kd, ws.ms)
    tf.idft!(ch.uR, ch.ûR)
    tf.idft!(ch.uC, ch.ûC)
    tf.idft!(ch.vR, ch.v̂R)
    tf.idft!(ch.vC, ch.v̂C)

    # 𝒩̂₁/𝒩̂₂ depend only on the giver (α) part — two distinct sets (α=R,C), reused across the four
    # channels; each reuses the (now-free) main gradient/nonlinear scratch (ws.gradu/gradv/N1_phys/…).
    giver_N!(N̂1_out, N̂2_out, u_giv, v_giv) = begin
        tf.dft!(ch.sscr, v_giv); tf.grad!(ws.gradv, ch.sscr)
        tf.dft!(ch.sscr, u_giv); tf.grad!(ws.gradu, ch.sscr)
        _assemble_N!(ws.N1_phys, ws.N2_phys, ws.u_phys, v_giv, ws.gradu, ws.gradv, ws.divu, ns, nd)
        tf.dft!(N̂1_out, ws.N1_phys)
        tf.dft!(N̂2_out, ws.N2_phys)
    end
    giver_N!(ch.N̂1R, ch.N̂2R, ch.uR, ch.vR)
    giver_N!(ch.N̂1C, ch.N̂2C, ch.uC, ch.vC)

    # Receivers and the givers' nonlinear terms on the caller's grid.
    _receiver_splits!(ws, ûc, mc)
    for (c, w) in ((rc.N̂1R, ch.N̂1R), (rc.N̂2R, ch.N̂2R), (rc.N̂1C, ch.N̂1C), (rc.N̂2C, ch.N̂2C))
        _unload!(c, w, ws)
    end

    # Transfer density: receiver β-part at k, giver α-part carried through the nonlinear term
    #   T^{βα}(k) = ½ Re{ û_β*(k)·𝒩̂₁[α] } + ½ Re{ v̂_β*(k)·𝒩̂₂[α] }  (into the reused ws.td)
    Π(û_recv, v̂_recv, N̂1, N̂2) =
        _flux_from_transfer(_bin(_density!(ws.td, û_recv, v̂_recv, N̂1, N̂2), sidx, N_sh, FT, ks, mc))
    rr = Π(rc.ûR, rc.v̂R, rc.N̂1R, rc.N̂2R)   # R receiver, R giver
    cc = Π(rc.ûC, rc.v̂C, rc.N̂1C, rc.N̂2C)   # C receiver, C giver
    rcf = Π(rc.ûR, rc.v̂R, rc.N̂1C, rc.N̂2C)  # C→R : R receiver, C giver
    cr = Π(rc.ûC, rc.v̂C, rc.N̂1R, rc.N̂2R)   # R→C : C receiver, R giver
    return (rotational = rr, compressive = cc, rot_to_comp = cr, comp_to_rot = rcf)
end

# The rotational/compressive split of `û` and `v̂` on the caller's grid, into `channels.rc`. Under
# padding the working-grid splits hold the givers, so the receivers are split here from the coarse
# fields; otherwise the two are the same arrays.
function _receiver_splits!(ws, ûc, mc)
    rc = ws.channels.rc
    _helmholtz_split!(rc.ûR, rc.ûC, ûc, ws.kdc, mc)
    _helmholtz_split!(rc.v̂R, rc.v̂C, ws.coarse.v̂, ws.kdc, mc)
    return rc
end

# In-place Helmholtz split (rot ⊥ k, comp ∥ k) writing into provided buffers. `kd` holds each axis's
# derivative wavenumbers, so the split is built from the grid divergence `i k·û`: the Nyquist component
# of an even axis contributes nothing to `∂_d` and so nothing to `∇·u`, and a mode at Nyquist on every
# axis is divergence-free on the grid and comes out purely rotational.
function _helmholtz_split!(rot, comp, field_hat, kd, ms::NTuple{nd,Int}) where {nd}
    FT = real(eltype(field_hat))
    @inbounds for kI in CartesianIndices(ms)
        k = ntuple(d -> FT(kd[d][kI[d]]), nd)
        k2 = zero(FT)
        for d in 1:nd; k2 += k[d]^2; end
        if k2 == 0
            for c in 1:nd; comp[kI, c] = zero(eltype(comp)); rot[kI, c] = field_hat[kI, c]; end
        else
            kdotu = zero(complex(FT))
            for c in 1:nd; kdotu += k[c] * field_hat[kI, c]; end
            for c in 1:nd
                cc = (kdotu / k2) * k[c]
                comp[kI, c] = cc
                rot[kI, c]  = field_hat[kI, c] - cc
            end
        end
    end
    return nothing
end

# ---------------------------------------------------------------------------
# KE↔IE pressure-dilatation (paper Eqs. 38–39):
#   Q_{I,R}(k) = ½ Re[σ̃(k)·v_R*(k)]
#   Q_{I,C}(k) = ½ Re[σ̃(k)·v_C*(k)] − ½ Im[σ(k){k·u_C*(k)}]
# with σ̃ = ∇σ/ρ (specific pressure gradient). Shell-binned. Vanishes for incompressible div-free flow.
# σ̃ divides by ρ, so its spectrum is unbounded and neither dealiasing strategy makes it exact.
# ---------------------------------------------------------------------------
function _pressure_dilatation!(ws, ûc, σhc, ks, mc::NTuple{nd,Int}, sidx, N_sh, ::Type{FT}) where {nd, FT}
    tf = ws.tf
    pr = ws.pressure
    colons = ntuple(_ -> Colon(), nd)
    rc = _receiver_splits!(ws, ûc, mc)
    # σ̃ = ∇σ / ρ : physical gradient of σ divided by ρ, back to spectral, then to the caller's grid.
    tf.grad!(pr.gradσ, pr.σh)                             # (ns..., 1, nd) real
    for d in 1:nd
        view(pr.σ̃phys, colons..., d) .= view(pr.gradσ, colons..., 1, d) ./ ws.ρ_phys
    end
    tf.dft!(pr.σ̃, pr.σ̃phys)
    _unload!(pr.σ̃c, pr.σ̃, ws)

    # `QR`/`QC` reuse the main scratch (free once the transfer and channels are done); the wavenumber
    # arrays and `kdotuC` live in the workspace, so this stays on the 0-alloc reuse contract.
    QR = pr.qscr; QC = ws.td
    kdotuC = pr.kdotuC            # Σ_c k_c·conj(û_C,c), with the grid's derivative wavenumbers
    fill!(QR, zero(FT)); fill!(QC, zero(FT)); fill!(kdotuC, zero(eltype(kdotuC)))
    for c in 1:nd
        QR .+= FT(0.5) .* real.(view(pr.σ̃c, colons..., c) .* conj.(view(rc.v̂R, colons..., c)))
        QC .+= FT(0.5) .* real.(view(pr.σ̃c, colons..., c) .* conj.(view(rc.v̂C, colons..., c)))
        kdotuC .+= ws.kdc[c] .* conj.(view(rc.ûC, colons..., c))
    end
    QC .-= FT(0.5) .* imag.(reshape(σhc, mc) .* kdotuC)
    return (rotational = _bin(QR, sidx, N_sh, FT, ks, mc), compressive = _bin(QC, sidx, N_sh, FT, ks, mc))
end

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
_as_scalar(field, ns::NTuple{nd,Int}) where {nd} =
    ndims(field) == nd ? reshape(field, ns..., 1) : field

# Shell-sum a per-mode density with the Hermitian weight. Every mode enters: under truncation the
# momentum term `v̂*·𝒩̂₂` is exact out to twice the kept band, and conservation needs all of it.
function _bin(td, sidx, N_sh, ::Type{FT}, ks, ms::NTuple{nd,Int}) where {nd, FT}
    T = zeros(FT, N_sh)
    @inbounds for I in CartesianIndices(ms)
        n = sidx[I]; n == 0 && continue
        T[n] += SpectralLayout.hermitian_weight(ks, I) * td[I]
    end
    return T
end

# Π(K) = Σ_{k≤K} T_u(k); the last entry is Σ_k T_u, the conservation residual.
_flux_from_transfer(T_spec) = cumsum(T_spec)

function _default_binning(ks)
    min_dk = Inf
    for kv in ks, k in kv
        ak = abs(k); ak > 0 && (min_dk = min(min_dk, ak))
    end
    return Types.LinearBinning(isfinite(min_dk) ? min_dk : 1.0)
end

# One-line show (the workspace's transform context holds FFTW plans → default show can segfault).
Base.show(io::IO, ::CompressibleWorkspace) = print(io, "CompressibleWorkspace(…)")
Base.show(io::IO, ::MIME"text/plain", w::CompressibleWorkspace) = show(io, w)

end # module Compressible


