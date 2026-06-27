# GPU benchmark — requires a functional CUDA device.
#
#   julia --project=gpu gpu/benchmarks.jl
#
# Benchmarks shell-to-shell transfer on the GPU (GPUBackend) against the serial CPU path on the
# same field. The device kernels are validated for correctness on the KA CPU backend in the test
# suite ("GPU kernels via KA CPU backend"); this script measures on-device performance, which needs
# real hardware. With no functional GPU it falls back to the KA CPU backend so the script still runs.

using CUDA: CUDA
using KernelAbstractions: KernelAbstractions as KA
using BenchmarkTools: @belapsed
using FFTW: FFTW
using Random: Random
using FlowInvariantTransfer: FlowInvariantTransfer as FET

function field2d(N; L = 2π, seed = 1)
    ks = FET.wavenumber_grid((N, N), (L, L))
    kx = [ks[1][i] for i in 1:N, j in 1:N]
    ky = [ks[2][j] for i in 1:N, j in 1:N]
    ψh = FFTW.fft(randn(Random.MersenneTwister(seed), N, N)) ./ N^2
    cat(im .* ky .* ψh, -im .* kx .* ψh; dims = 3), ks
end

const HAS_GPU = CUDA.functional()
println("CUDA functional: ", HAS_GPU)
device_backend = HAS_GPU ? FET.GPUBackend(CUDA.CUDABackend()) : FET.GPUBackend(KA.CPU())
to_device(x) = HAS_GPU ? CUDA.CuArray(x) : x

for N in (64, 128, 256)
    û, ks = field2d(N)
    b = FET.LinearBinning(2π / (2π))

    t_cpu = @belapsed FET.calculate_shell_to_shell_transfer($û, $ks; binning = $b,
        spectral = FET.FFTBackend(), execution = FET.SerialBackend(), verify_antisymmetry = false)

    ûd  = to_device(û)
    ksd = HAS_GPU ? map(CUDA.CuArray, ks) : ks
    t_dev = @belapsed FET.calculate_shell_to_shell_transfer($ûd, $ksd; binning = $b,
        spectral = FET.FFTBackend(), execution = $device_backend, verify_antisymmetry = false)

    println("N=$N  serial-CPU=$(round(t_cpu*1e3; digits=2))ms  ",
            HAS_GPU ? "GPU" : "KA-CPU", "=$(round(t_dev*1e3; digits=2))ms  ",
            "speedup=$(round(t_cpu/t_dev; digits=2))×")
end
