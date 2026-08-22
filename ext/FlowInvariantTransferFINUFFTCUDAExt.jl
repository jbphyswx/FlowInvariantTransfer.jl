module FlowInvariantTransferFINUFFTCUDAExt

using FINUFFT: FINUFFT
using CUDA: CUDA
using FlowInvariantTransfer: FlowInvariantTransfer as FIT
using ComputationalBackends: ComputationalBackends

# ---------------------------------------------------------------------------
# cuFINUFFT device path for the Types.FINUFFTBackend `to_spectral` — the FINUFFT peer of the KA-generic
# NonuniformFFTs device path, so a GPUBackend keeps the scattered → velocity_hat step on an NVIDIA GPU.
# FINUFFT.jl loads its cuFINUFFT interface lazily under `using CUDA`, so `cufinufft_plan` and the
# `cufinufft_*` functions resolve only at runtime: every call to them lives in a function body, and the
# runtime-defined plan is held in an owned handle so no lazily-defined symbol appears in a dispatch
# signature (which would fail to precompile). Requires NVIDIA hardware to run.
# ---------------------------------------------------------------------------

# Owned wrapper around FINUFFT's runtime-defined cuFINUFFT plan; `to_spectral!` dispatches on this type
# rather than on `FINUFFT.cufinufft_plan`.
struct CuFINUFFTPlan{P}
    plan::P
end

# Device build (peer of the host `_finufft_ts_build`): a type-1 cuFINUFFT plan with the points preset
# on-device + device buffers. iflag=−1 → e^{-ik·x} analysis (Julia `fft` convention); modeord=1 → FFTW
# mode order, matching Utils.wavenumber_grid. Returns (plan, coords, û, scat, spec).
function FIT._finufft_ts_build(::ComputationalBackends.GPUBackend{<:CUDA.CUDABackend}, ms::Tuple, tol::Real,
                               scaled::Tuple, ::Type{CT}, ncomponents::Int, N::Int) where {CT}
    FT = real(CT)
    plan   = FINUFFT.cufinufft_makeplan(1, Int64[ms...], -1, 1, FT(tol); dtype = FT, modeord = 1)
    coords = map(x -> CUDA.CuArray{FT}(x), scaled)          # points uploaded once to the device
    FINUFFT.cufinufft_setpts!(plan, coords...)
    finalizer(FINUFFT.cufinufft_destroy!, plan)            # free the device plan when GC'd
    û    = CUDA.CuArray{CT}(undef, ms..., ncomponents)
    scat = CUDA.CuArray{CT}(undef, N)
    spec = CUDA.CuArray{CT}(undef, ms...)
    return (CuFINUFFTPlan(plan), coords, û, scat, spec)
end

# Device analysis û = type1(u)/N reusing the preset plan + device buffers (nothing allocated on repeat).
# Dispatches on the owned handle; the velocity fields are expected on the workspace's device.
function FIT.to_spectral!(ws::FIT.NUFFTToSpectralWorkspace{<:CuFINUFFTPlan}, velocity_fields::Tuple)
    D = size(ws.û, ndims(ws.û))
    length(velocity_fields) == D || throw(DimensionMismatch(
        "to_spectral! got $(length(velocity_fields)) fields; workspace was built for $D components"))
    colons = ntuple(_ -> Colon(), ndims(ws.û) - 1)
    @inbounds for c in 1:D
        length(velocity_fields[c]) == ws.npoints || throw(DimensionMismatch("field length ≠ workspace points $(ws.npoints)"))
        ws.scat .= velocity_fields[c]
        FINUFFT.cufinufft_exec!(ws.plan.plan, ws.scat, ws.spec)   # nonuniform → uniform, device buffers
        ûc = view(ws.û, colons..., c)
        @. ûc = ws.spec * ws.invN
    end
    return (ws.û, ws.ks)
end

end # module
