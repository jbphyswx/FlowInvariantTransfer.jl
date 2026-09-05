module Compressible

using ..Types: Types
using ..SpectralLayout: SpectralLayout
using ComputationalBackends: ComputationalBackends
using SpectralBackends: SpectralBackends
using ..ShellBinning: ShellBinning
using ..NonlinearTerm: NonlinearTerm

export calculate_compressible_flux, calculate_compressible_flux!, CompressibleWorkspace

# ---------------------------------------------------------------------------
# Compressible kinetic-energy spectral transfer (Singh–Tiwari–Sharma–Verma 2025,
# arXiv:2508.04300). Framework A: momentum v = ρu,
# KE E_u(k) = ½Re[v(k)·u*(k)]. The nonlinear transfer is momentum-weighted and conserves
# total KE (Σ_k T_u = 0); the KE↔internal-energy exchange is the *separate* pressure-dilatation
# term Q_{I}, gated on a supplied pressure field.
#
# Net per-mode transfer, reduced from the scale-to-scale form (paper Eq. 20/28) to a
# pseudospectral O(Nᴰ) expression (validated by Σ_k T_u = 0 and the
# incompressible limit ρ=const, ∇·u=0 ⇒ T_u = −ρ·Re{û*·(u·∇)u}, i.e. −ρ × the incompressible
# transfer_spectrum):
#
#     T_u(k) = −½ Re{ û*(k)·𝒩̂₁(k) } − ½ Re{ v̂*(k)·𝒩̂₂(k) }
#     𝒩₁ = (u·∇)v + v(∇·u) = ∂_j(v ⊗ u)_j ,   𝒩₂ = (u·∇)u ,   v = ρu.
#
# This reference works entirely by explicit DFT/IDFT (dependency-free, exact), mirroring the
# SpectralBackends.DirectSumSpectralBackend philosophy of the incompressible path; small grids only, correctness-first.
# ---------------------------------------------------------------------------

# Forward/backward direct DFT on the spatial dimensions of an (ns..., C) field. Convention matches
# NonlinearTerm.jl: forward = Σ_x f(x) e^{-i k·x}/N (analysis), backward = Σ_k f̂(k) e^{+i k·x}
# (synthesis), with x_d = (I_d-1)/n_d and integer wavenumber index k_d.
function _idft(field_hat, ns::NTuple{nd,Int}) where {nd}
    FT = real(eltype(field_hat))
    C  = size(field_hat, nd + 1)
    out = Array{complex(FT)}(undef, ns..., C)
    @inbounds for c in 1:C
        for xI in CartesianIndices(ns)
            acc = zero(complex(FT))
            for kI in CartesianIndices(ns)
                phase = zero(FT)
                for d in 1:nd
                    kidx = kI[d] - 1
                    km   = 2 * kidx < ns[d] ? kidx : kidx - ns[d]
                    phase += FT(2π) * km * FT(xI[d] - 1) / FT(ns[d])
                end
                acc += field_hat[kI, c] * cis(phase)
            end
            out[xI, c] = acc
        end
    end
    return out
end

function _dft(field_phys, ns::NTuple{nd,Int}) where {nd}
    FT = real(eltype(field_phys))
    C  = size(field_phys, nd + 1)
    out = Array{complex(FT)}(undef, ns..., C)
    Np  = prod(ns)
    @inbounds for c in 1:C
        for kI in CartesianIndices(ns)
            acc = zero(complex(FT))
            for xI in CartesianIndices(ns)
                phase = zero(FT)
                for d in 1:nd
                    kidx = kI[d] - 1
                    km   = 2 * kidx < ns[d] ? kidx : kidx - ns[d]
                    phase += FT(2π) * km * FT(xI[d] - 1) / FT(ns[d])
                end
                acc += field_phys[xI, c] * cis(-phase)
            end
            out[kI, c] = acc / FT(Np)
        end
    end
    return out
end

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

# In-place gradient (no dfield temp — folds i k_d into the synthesis sum) writing into g (ns...,C,nd).
# `derivative_wavenumber` drops the Nyquist mode of an even axis, whose grid derivative is zero.
function _grad_phys!(g, field_hat, ks, ns::NTuple{nd,Int}, ms::NTuple{nd,Int}) where {nd}
    C = size(field_hat, nd + 1)
    @inbounds for d in 1:nd, c in 1:C, xI in CartesianIndices(ns)
        g[xI, c, d] = _synth_at(field_hat, c, ks, xI, ns, ms,
                                kI -> im * SpectralLayout.derivative_wavenumber(ks[d], kI[d]))
    end
    return g
end

# Physical-space spatial gradient ∂f_c/∂x_d for every component c and direction d, from the
# spectral field: ∂_d f = IDFT(i k_d f̂). Returns an (ns..., C, nd) real-or-complex array.
function _grad_phys(field_hat, ks, ns::NTuple{nd,Int}) where {nd}
    FT = real(eltype(field_hat))
    C  = size(field_hat, nd + 1)
    g  = Array{complex(FT)}(undef, ns..., C, nd)
    @inbounds for d in 1:nd
        dfield_hat = similar(field_hat)
        for c in 1:C, kI in CartesianIndices(ns)
            dfield_hat[kI, c] = im * FT(ks[d][kI[d]]) * field_hat[kI, c]
        end
        gd = _idft(dfield_hat, ns)
        for c in 1:C, xI in CartesianIndices(ns)
            g[xI, c, d] = gd[xI, c]
        end
    end
    return g
end

# ---------------------------------------------------------------------------
# Helmholtz (rotational/compressive) split in Fourier space:
#   u_C(k) = [k·û(k)/|k|²] k   (compressive, ∥ k) ,   u_R = û − u_C   (rotational, ⊥ k).
# k = 0 has no compressive part. Dimension-generic; no external dependency.
# ---------------------------------------------------------------------------
function _helmholtz_split(field_hat, ks, ns::NTuple{nd,Int}) where {nd}
    FT   = real(eltype(field_hat))
    comp = zero(field_hat)
    rot  = similar(field_hat)
    @inbounds for kI in CartesianIndices(ns)
        k2 = zero(FT)
        for d in 1:nd
            k2 += FT(ks[d][kI[d]])^2
        end
        if k2 == 0
            for c in 1:nd
                comp[kI, c] = zero(eltype(field_hat))
                rot[kI, c]  = field_hat[kI, c]
            end
        else
            kdotu = zero(complex(FT))
            for c in 1:nd
                kdotu += FT(ks[c][kI[c]]) * field_hat[kI, c]
            end
            for c in 1:nd
                cc = (kdotu / k2) * FT(ks[c][kI[c]])
                comp[kI, c] = cc
                rot[kI, c]  = field_hat[kI, c] - cc
            end
        end
    end
    return rot, comp
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

function _directsum_tf(ks, ns::NTuple{nd,Int}) where {nd}
    ms = SpectralLayout.spectral_size(ks)
    FT = float(eltype(ks[1]))
    CT = complex(FT)
    return TransformContext(
        fh -> _idft!(Array{FT}(undef, ns..., size(fh, nd + 1)), fh, ks, ns, ms),
        fp -> _dft!(Array{CT}(undef, ms..., size(fp, nd + 1)), fp, ks, ns, ms),
        fh -> _grad_phys!(Array{FT}(undef, ns..., size(fh, nd + 1), nd), fh, ks, ns, ms),
        (out, fh) -> _idft!(out, fh, ks, ns, ms),
        (out, fp) -> _dft!(out, fp, ks, ns, ms),
        (g, fh) -> _grad_phys!(g, fh, ks, ns, ms),
    )
end

# The FFT context is provided by FlowInvariantTransferFFTWExt; this fallback (less specific than the
# extension's method) gives a clear error when a non-DirectSum backend is requested without FFTW.
# `fft_nthreads` pins the FFTW plan thread count (baked into the plan → no per-call scratch alloc); the
# threaded execution path passes `> 1` so the single-pipeline FFTs run multithreaded (no outer loop).
_fft_tf(velocity_hat, ks, ns; fft_nthreads::Int = 1) = throw(ArgumentError(
    "calculate_compressible_flux with an FFT backend requires `using FFTW`; " *
    "or pass `spectral = SpectralBackends.DirectSumSpectralBackend()` for the dependency-free (slow, small-grid) path."))

# Explicit per-backend transform contexts — NO `::SpectralBackends.AbstractSpectralBackend` catch-all: a catch-all
# silently routed the scattered/spherical backends through the FFT context (wrong answers, no error).
# The public entry validates the backend (`require_coefficient_spectral`), so only DirectSum/FFT reach here.
_resolve_tf(::SpectralBackends.DirectSumSpectralBackend, velocity_hat, ks, ns; fft_nthreads::Int = 1) = _directsum_tf(ks, ns)
_resolve_tf(::SpectralBackends.FFTSpectralBackend, velocity_hat, ks, ns; fft_nthreads::Int = 1) =
    _fft_tf(velocity_hat, ks, ns; fft_nthreads = fft_nthreads)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

"""
    CompressibleWorkspace(velocity_hat, ks; spectral=SpectralBackends.FFTSpectralBackend())

Reusable field-scratch + transform context for [`calculate_compressible_flux!`](@ref) — holds the
FFT plans and every (ns...)-sized intermediate of the momentum-weighted budget (velocity/density/
momentum physical & spectral fields, gradients, nonlinear terms, and the R/C-channel + pressure-
dilatation scratch), so repeated per-snapshot calls allocate ~0 field memory (only the small shell
vectors of each result). Build once for a given grid/precision and reuse across snapshots.
"""
struct CompressibleWorkspace{ND, TF, CA, CS, RA, RS, RM, RG, CH, PD, SI, CE}
    tf::TF
    vel::CA; ρh::CS; v̂::CA; N̂1::CA; N̂2::CA
    u_phys::RA; v_phys::RA; N1_phys::RA; N2_phys::RA
    ρ_phys::RS; divu::RS         # physical, sized `ns`
    td::RM                       # per-MODE transfer density, sized `ms`
    gradu::RG; gradv::RG
    channels::CH                  # R/C-channel scratch, or `nothing` when the workspace excludes them
    pressure::PD                  # pressure-dilatation scratch, or `nothing`
    sidx::SI; centers::CE         # precomputed shell structure (geometry+binning fixed for the workspace)
    ns::NTuple{ND, Int}           # physical grid
    ms::NTuple{ND, Int}           # coefficient grid (differs from `ns` on the half layout)
end

# Buffers used only by the Helmholtz R/C channel decomposition.
struct CompressibleChannels{CA, RA}
    sscr::CA
    ûR::CA; ûC::CA; v̂R::CA; v̂C::CA
    uR::RA; uC::RA; vR::RA; vC::RA
    N̂1R::CA; N̂2R::CA; N̂1C::CA; N̂2C::CA
end

# Buffers used only by the KE↔IE pressure-dilatation term.
struct CompressiblePressure{CS, RG1, RA, CA, C1, RM, KG}
    σh::CS; gradσ::RG1; σ̃phys::RA; σ̃::CA
    kdotuC::C1
    qscr::RM                     # per-MODE rotational accumulator, sized `ms`
    kg::KG
end

"""
    CompressibleWorkspace(velocity_hat, ks; spectral, binning, geometry, execution,
                          decompose=true, with_pressure=false)

`decompose` and `with_pressure` fix which optional terms the workspace can compute. Their scratch is
allocated only when selected: the R/C channels are eight coefficient-sized and four grid-sized
arrays, and the pressure-dilatation term four more. A flux-only workspace holds neither.
"""
function CompressibleWorkspace(velocity_hat, ks;
                               spectral::SpectralBackends.AbstractSpectralBackend = SpectralBackends.AutoSpectralBackend(),
                               binning::Types.AbstractShellBinning = _default_binning(ks),
                               geometry::Types.AbstractShellGeometry = Types.IsotropicShells(),
                               execution::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.SerialBackend(),
                               decompose::Bool = true,
                               with_pressure::Bool = false)
    spectral = Types.resolve_spectral(Types.require_coefficient_spectral(spectral))
    nd = length(ks)
    ns = SpectralLayout.full_size(ks)          # physical grid
    ms = SpectralLayout.spectral_size(ks)      # coefficient grid
    FT = real(eltype(velocity_hat)); CT = complex(FT)
    size(velocity_hat)[1:nd] == ms || throw(DimensionMismatch(
        "field spatial size $(size(velocity_hat)[1:nd]) does not match the wavenumber grid $ms."))
    # Compressible is a single ~O(nd) FFT pipeline (not an outer loop), so ComputationalBackends.ThreadedBackend threads the
    # FFTs themselves — plans baked at nthreads (no per-call scratch alloc, no oversubscription).
    fft_nthreads = execution isa ComputationalBackends.ThreadedBackend ? Threads.nthreads() : 1
    # Buffers built in `velocity_hat`'s own array type → device-resident for device input (the whole
    # pipeline is device-generic broadcasts), plain `Array`s for host input. Coefficient buffers are
    # sized by `ms`, physical ones by `ns`, and every physical buffer is REAL.
    ca()  = similar(velocity_hat, CT, ms..., nd)
    cs()  = similar(velocity_hat, CT, ms..., 1)
    c1()  = similar(velocity_hat, CT, ms...)
    ra()  = similar(velocity_hat, FT, ns..., nd)
    rs()  = similar(velocity_hat, FT, ns...)
    rm()  = similar(velocity_hat, FT, ms...)
    rg()  = similar(velocity_hat, FT, ns..., nd, nd)
    rg1() = similar(velocity_hat, FT, ns..., 1, nd)
    # Shell structure depends only on (ks, geometry, binning) — hoisted here so the per-snapshot `!`
    # never reallocates the full-grid magnitude/shell-index arrays.
    k_mag   = ShellBinning.shell_coordinate(geometry, ks)
    kmax    = ShellBinning.max_shell_coordinate(geometry, ks)
    edges   = ShellBinning.shell_edges(binning, kmax)
    centers = collect(ShellBinning.shell_centers(binning, kmax))
    sidx    = ShellBinning.assign_shells(k_mag, edges)
    channels = decompose ?
        CompressibleChannels(ca(), ca(), ca(), ca(), ca(), ra(), ra(), ra(), ra(), ca(), ca(), ca(), ca()) :
        nothing
    pressure = with_pressure ?
        CompressiblePressure(cs(), rg1(), ra(), ca(), c1(), rm(),
                             SpectralLayout.wavenumber_arrays(velocity_hat, FT, ks)) :
        nothing
    return CompressibleWorkspace(
        _resolve_tf(spectral, velocity_hat, ks, ns; fft_nthreads = fft_nthreads),
        ca(), cs(), ca(), ca(), ca(),
        ra(), ra(), ra(), ra(),
        rs(), rs(), rm(),
        rg(), rg(),
        channels, pressure,
        sidx, centers, ns, ms)
end

"""
    calculate_compressible_flux(velocity_hat, density_hat, ks; binning, pressure_hat=nothing,
        decompose=true, geometry=IsotropicShells(), spectral=SpectralBackends.FFTSpectralBackend()) -> CompressibleFluxResult

Compressible kinetic-energy spectral transfer `T_u(k)` and cumulative flux `Π(K)` (Singh–Tiwari–
Sharma–Verma 2025): momentum `v = ρu`, `E_u(k) = ½Re[v·u*]`; the nonlinear transfer conserves total KE
(`Σ_k T_u ≈ 0`), and the KE↔internal-energy pressure-dilatation is returned separately when
`pressure_hat` is supplied. `decompose=true` also returns the Helmholtz rotational/compressive flux
channels. In the incompressible limit `T_u` reduces to `−ρ ×` the incompressible transfer spectrum.
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
        ComputationalBackends.is_gpu_array(velocity_hat) || throw(ArgumentError(
            "compressible transfer runs on-device automatically for device-array inputs (the pipeline is " *
            "device-generic broadcasts + cuFFT via AbstractFFTs); execution=ComputationalBackends.GPUBackend() does not move a host " *
            "array to the device. Pass device-array inputs (e.g. CuArray), or use ComputationalBackends.SerialBackend()/ComputationalBackends.ThreadedBackend()."))
        ws = CompressibleWorkspace(velocity_hat, ks; spectral=spectral, binning=binning, geometry=geometry, execution=ComputationalBackends.SerialBackend())
        return calculate_compressible_flux!(ws, velocity_hat, density_hat, ks; kwargs...)
    end
    ws = CompressibleWorkspace(velocity_hat, ks; spectral=spectral, binning=binning, geometry=geometry, execution=exec)
    return calculate_compressible_flux!(ws, velocity_hat, density_hat, ks; kwargs...)
end

# Overridden by the Distributed extension (requires `using Distributed`).
_compressible_distributed(args...; kwargs...) = throw(ArgumentError(
    "Distributed compressible transfer requires Distributed. " *
    "Run `using Distributed` to load the extension."))

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
    dealiasing::Types.AbstractDealiasing = Types.OrszagTwoThirds(),
    decompose::Bool = true,
)
    nd = length(ks)
    ns = ws.ns          # physical grid
    ms = ws.ms          # coefficient grid
    FT = real(eltype(velocity_hat))
    tf = ws.tf
    colons = ntuple(_ -> Colon(), nd)   # component-slice views for device-generic broadcasts
    decompose && ws.channels === nothing && throw(ArgumentError(
        "this CompressibleWorkspace was built with decompose = false; rebuild it with decompose = true " *
        "to compute the Helmholtz R/C channels."))
    pressure_hat !== nothing && ws.pressure === nothing && throw(ArgumentError(
        "this CompressibleWorkspace was built with with_pressure = false; rebuild it with " *
        "with_pressure = true to compute the pressure-dilatation term."))

    # Orszag 2/3 dealiasing: copy inputs into the workspace, zeroing the |k| ≥ N/3 discard band.
    trunc = dealiasing isa Types.OrszagTwoThirds
    _copy_trunc!(ws.vel, velocity_hat, ks, ms, trunc)
    _copy_trunc!(ws.ρh, _as_scalar(density_hat, ms), ks, ms, trunc)

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

    # Per-mode net transfer T_u(k) = −½Re{û*·𝒩̂₁} − ½Re{v̂*·𝒩̂₂}
    fill!(ws.td, zero(FT))
    for c in 1:nd
        ws.td .+= real.(conj.(view(ws.vel, colons..., c)) .* view(ws.N̂1, colons..., c) .+
                        conj.(view(ws.v̂, colons..., c)) .* view(ws.N̂2, colons..., c))
    end
    ws.td .*= -FT(0.5)

    # Shell binning — precomputed in the workspace (0-alloc across snapshots)
    sidx    = ws.sidx
    centers = ws.centers
    N_sh    = length(centers)

    T_spec = _bin(ws.td, sidx, N_sh, FT, ks, ms, trunc)
    flux   = _flux_from_transfer(T_spec)

    channels = decompose ? _rc_channels!(ws, ks, ns, ms, sidx, N_sh, FT, trunc) : nothing
    pdil = nothing
    if pressure_hat !== nothing
        _copy_trunc!(ws.pressure.σh, _as_scalar(pressure_hat, ms), ks, ms, trunc)
        pdil = _pressure_dilatation!(ws, ks, ns, ms, sidx, N_sh, FT, trunc)
    end

    return Types.CompressibleFluxResult(centers, T_spec, flux, channels, pdil)
end

# Copy `src` (ns...,C) into `dst`, zeroing the Orszag 2/3 discard band (|k| ≥ N/3 on any axis) if trunc.
function _copy_trunc!(dst, src, ks, ns::NTuple{nd,Int}, trunc::Bool) where {nd}
    C = size(src, nd + 1)
    @inbounds for c in 1:C, I in CartesianIndices(ns)
        dst[I, c] = (trunc && NonlinearTerm._is_dealiased(ks, I)) ? zero(eltype(dst)) : src[I, c]
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
function _rc_channels!(ws, ks, ns::NTuple{nd,Int}, ms::NTuple{nd,Int}, sidx, N_sh, ::Type{FT}, trunc) where {nd, FT}
    tf = ws.tf
    ch = ws.channels
    colons = ntuple(_ -> Colon(), nd)
    _helmholtz_split!(ch.ûR, ch.ûC, ws.vel, ks, ms)
    _helmholtz_split!(ch.v̂R, ch.v̂C, ws.v̂, ks, ms)
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

    # Transfer density: receiver β-part at k, giver α-part carried through the nonlinear term
    #   T^{βα}(k) = −½ Re{ û_β*(k)·𝒩̂₁[α] } − ½ Re{ v̂_β*(k)·𝒩̂₂[α] }  (into the reused ws.td)
    Π(û_recv, v̂_recv, N̂1, N̂2) = begin
        fill!(ws.td, zero(FT))
        for c in 1:nd
            ws.td .+= real.(conj.(view(û_recv, colons..., c)) .* view(N̂1, colons..., c) .+
                            conj.(view(v̂_recv, colons..., c)) .* view(N̂2, colons..., c))
        end
        ws.td .*= -FT(0.5)
        _flux_from_transfer(_bin(ws.td, sidx, N_sh, FT, ks, ms, trunc))
    end
    rr = Π(ch.ûR, ch.v̂R, ch.N̂1R, ch.N̂2R)   # R receiver, R giver
    cc = Π(ch.ûC, ch.v̂C, ch.N̂1C, ch.N̂2C)   # C receiver, C giver
    rc = Π(ch.ûR, ch.v̂R, ch.N̂1C, ch.N̂2C)   # C→R : R receiver, C giver
    cr = Π(ch.ûC, ch.v̂C, ch.N̂1R, ch.N̂2R)   # R→C : C receiver, R giver
    return (rotational = rr, compressive = cc, rot_to_comp = cr, comp_to_rot = rc)
end

# In-place Helmholtz split (rot ⊥ k, comp ∥ k) writing into provided buffers.
# The split is built from the GRID divergence `i k·û`, so it uses `derivative_wavenumber`: the
# Nyquist component of an even axis contributes nothing to `∂_d` and so nothing to `∇·u`. A mode
# sitting at Nyquist on every axis is divergence-free on the grid and comes out purely rotational.
# The raw `k` at that slot carries a sign the mode itself does not fix — it is its own mirror, and
# the two spectral layouts label it `+n/2` and `−n/2` respectively.
function _helmholtz_split!(rot, comp, field_hat, ks, ms::NTuple{nd,Int}) where {nd}
    FT = real(eltype(field_hat))
    @inbounds for kI in CartesianIndices(ms)
        kd = ntuple(d -> FT(SpectralLayout.derivative_wavenumber(ks[d], kI[d])), nd)
        k2 = zero(FT)
        for d in 1:nd; k2 += kd[d]^2; end
        if k2 == 0
            for c in 1:nd; comp[kI, c] = zero(eltype(comp)); rot[kI, c] = field_hat[kI, c]; end
        else
            kdotu = zero(complex(FT))
            for c in 1:nd; kdotu += kd[c] * field_hat[kI, c]; end
            for c in 1:nd
                cc = (kdotu / k2) * kd[c]
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
# ---------------------------------------------------------------------------
function _pressure_dilatation!(ws, ks, ns::NTuple{nd,Int}, ms::NTuple{nd,Int}, sidx, N_sh, ::Type{FT}, trunc) where {nd, FT}
    tf = ws.tf
    pr = ws.pressure
    ch = ws.channels
    colons = ntuple(_ -> Colon(), nd)
    _helmholtz_split!(ch.ûR, ch.ûC, ws.vel, ks, ms)       # ûC used below
    _helmholtz_split!(ch.v̂R, ch.v̂C, ws.v̂, ks, ms)         # ws.v̂ preserved (giver_N! uses ch.sscr)
    # σ̃ = ∇σ / ρ : physical gradient of σ divided by ρ, back to spectral.
    tf.grad!(pr.gradσ, pr.σh)                             # (ns..., 1, nd) real
    for d in 1:nd
        view(pr.σ̃phys, colons..., d) .= view(pr.gradσ, colons..., 1, d) ./ ws.ρ_phys
    end
    tf.dft!(pr.σ̃, pr.σ̃phys)

    # `QR`/`QC` reuse the main scratch (free once the transfer and channels are done); the wavenumber
    # arrays and `kdotuC` live in the workspace, so this stays on the 0-alloc reuse contract.
    QR = pr.qscr; QC = ws.td
    kdotuC = pr.kdotuC            # Σ_c k_c·conj(û_C,c)
    fill!(QR, zero(FT)); fill!(QC, zero(FT)); fill!(kdotuC, zero(eltype(kdotuC)))
    for c in 1:nd
        QR .+= FT(0.5) .* real.(view(pr.σ̃, colons..., c) .* conj.(view(ch.v̂R, colons..., c)))
        QC .+= FT(0.5) .* real.(view(pr.σ̃, colons..., c) .* conj.(view(ch.v̂C, colons..., c)))
        kdotuC .+= pr.kg[c] .* conj.(view(ch.ûC, colons..., c))
    end
    QC .-= FT(0.5) .* imag.(reshape(pr.σh, ms) .* kdotuC)
    return (rotational = _bin(QR, sidx, N_sh, FT, ks, ms, trunc), compressive = _bin(QC, sidx, N_sh, FT, ks, ms, trunc))
end

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
_as_scalar(field, ns::NTuple{nd,Int}) where {nd} =
    ndims(field) == nd ? reshape(field, ns..., 1) : field

# Shell-sum a per-mode density. With `dealias=true`, the 2/3 discard band (|k| ≥ N/3) is excluded so
# aliased contributions never enter the retained shells (Orszag 2/3 output zeroing).
function _bin(td, sidx, N_sh, ::Type{FT}, ks, ns::NTuple{nd,Int}, dealias::Bool) where {nd, FT}
    T = zeros(FT, N_sh)
    @inbounds for I in CartesianIndices(ns)
        n = sidx[I]; n == 0 && continue
        dealias && NonlinearTerm._is_dealiased(ks, I) && continue
        T[n] += SpectralLayout.hermitian_weight(ks, I) * td[I]
    end
    return T
end

# Cumulative flux Π(K) = Σ_{k>K} T_u(k) (energy passing beyond shell K); with Σ_k T_u = 0 this equals
# −Σ_{k≤K} T_u(k). Returned as a per-shell vector aligned with the shell centers.
function _flux_from_transfer(T_spec)
    n = length(T_spec)
    Π = similar(T_spec)
    acc = zero(eltype(T_spec))
    tot = sum(T_spec)
    @inbounds for i in 1:n
        acc += T_spec[i]
        Π[i] = tot - acc          # Σ_{k>i}
    end
    return Π
end

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


