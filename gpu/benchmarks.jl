# GPU benchmark — device path via GPUBackend, against the serial CPU path on the same field.
#
#   julia --project=gpu gpu/benchmarks.jl
#   FIT_GPU_BENCH_SIZES=32,64 FIT_BENCH_SECONDS=0.5 julia --project=gpu gpu/benchmarks.jl   # fast smoke
#
# Benchmarks the GPU-capable diagnostics (spectral flux, shell-to-shell, band-to-band, compressible,
# and — smallest grid only — the O(N^{2D}) mode-to-mode tensor) on the device (GPUBackend) against the
# serial CPU path on the same field. The device kernels' *correctness* is validated on the KA CPU
# backend in the test suite ("GPU kernels via KA CPU backend"); this script measures on-device
# performance, which needs real hardware. With no functional GPU it falls back to GPUBackend(KA.CPU())
# so the script still runs the identical device code path (on plain Arrays). `FIT_GPU_BENCH_SIZES` and
# `FIT_BENCH_SECONDS` override the grid sweep and the per-measurement budget.

using CUDA: CUDA
using KernelAbstractions: KernelAbstractions as KA
using BenchmarkTools: @belapsed
using FFTW: FFTW
using Random: Random
using FlowInvariantTransfer: FlowInvariantTransfer as FIT

function field2d(N; L = 2π, seed = 1)
    ks = FIT.Utils.wavenumber_grid((N, N), (L, L))
    kx = [ks[1][i] for i in 1:N, j in 1:N]
    ky = [ks[2][j] for i in 1:N, j in 1:N]
    ψh = FFTW.fft(randn(Random.MersenneTwister(seed), N, N)) ./ N^2
    cat(im .* ky .* ψh, -im .* kx .* ψh; dims = 3), ks
end
density2d(N) = FFTW.fft([1.0 + 0.1 * cospi(2i / N) * sinpi(2j / N) for i in 1:N, j in 1:N]) ./ N^2

const HAS_GPU = CUDA.functional()
const DEVICE  = HAS_GPU ? FIT.GPUBackend(CUDA.CUDABackend()) : FIT.GPUBackend(KA.CPU())
const SECONDS = parse(Float64, get(ENV, "FIT_BENCH_SECONDS", "5"))
const SIZES   = let s = get(ENV, "FIT_GPU_BENCH_SIZES", "")
    isempty(s) ? (64, 128, 256) : Tuple(parse(Int, strip(x)) for x in split(s, ','))
end
const B     = FIT.LinearBinning(1.0)
const BANDS = FIT.SmoothBands([2.0, 4.0, 8.0])
to_device(x) = HAS_GPU ? CUDA.CuArray(x) : x
to_ks(ks)    = HAS_GPU ? map(CUDA.CuArray, ks) : ks
println("CUDA functional: ", HAS_GPU, "   (device = ", HAS_GPU ? "CUDA" : "KA.CPU", ")")

report(name, N, tc, td) = println(
    rpad(name, 16), " N=", rpad(N, 4),
    " serial-CPU=", lpad(round(tc * 1e3; digits = 2), 9), "ms  ",
    HAS_GPU ? "GPU=" : "KA-CPU=", lpad(round(td * 1e3; digits = 2), 9), "ms  ",
    "speedup=", round(tc / td; digits = 2), "×")

for N in SIZES
    û, ks   = field2d(N)
    ûd, ksd = to_device(û), to_ks(ks)
    ρ̂, ρ̂d   = density2d(N), to_device(density2d(N))

    report("spectral_flux", N,
        (@belapsed FIT.calculate_spectral_flux($û, $ks; binning = $B, spectral = FIT.FFTBackend(), execution = FIT.SerialBackend()) seconds = SECONDS),
        (@belapsed FIT.calculate_spectral_flux($ûd, $ksd; binning = $B, spectral = FIT.FFTBackend(), execution = $DEVICE) seconds = SECONDS))

    report("shell_to_shell", N,
        (@belapsed FIT.calculate_shell_to_shell_transfer($û, $ks; binning = $B, spectral = FIT.FFTBackend(), execution = FIT.SerialBackend(), verify_antisymmetry = false) seconds = SECONDS),
        (@belapsed FIT.calculate_shell_to_shell_transfer($ûd, $ksd; binning = $B, spectral = FIT.FFTBackend(), execution = $DEVICE, verify_antisymmetry = false) seconds = SECONDS))

    report("band_to_band", N,
        (@belapsed FIT.calculate_band_to_band_transfer($û, $ks; bands = $BANDS, spectral = FIT.FFTBackend(), execution = FIT.SerialBackend()) seconds = SECONDS),
        (@belapsed FIT.calculate_band_to_band_transfer($ûd, $ksd; bands = $BANDS, spectral = FIT.FFTBackend(), execution = $DEVICE) seconds = SECONDS))

    report("compressible", N,
        (@belapsed FIT.calculate_compressible_flux($û, $ρ̂, $ks; binning = $B, spectral = FIT.FFTBackend(), execution = FIT.SerialBackend()) seconds = SECONDS),
        (@belapsed FIT.calculate_compressible_flux($ûd, $ρ̂d, $ksd; binning = $B, spectral = FIT.FFTBackend(), execution = $DEVICE) seconds = SECONDS))
end

# mode-to-mode S(k|p) is O(N^{2D}); smallest grid only, force past the size guard.
let N = min(32, first(SIZES))
    û, ks   = field2d(N)
    ûd, ksd = to_device(û), to_ks(ks)
    report("mode_to_mode", N,
        (@belapsed FIT.calculate_mode_to_mode_transfer($û, $ks; spectral = FIT.FFTBackend(), execution = FIT.SerialBackend(), force = true) seconds = SECONDS),
        (@belapsed FIT.calculate_mode_to_mode_transfer($ûd, $ksd; spectral = FIT.FFTBackend(), execution = $DEVICE, force = true) seconds = SECONDS))
end
