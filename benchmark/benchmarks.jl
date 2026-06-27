# FlowInvariantTransfer.jl — CPU benchmark suite.
#
#   julia --project=benchmark benchmark/benchmarks.jl            # run all
#   julia --threads=4 --project=benchmark benchmark/benchmarks.jl  # include threaded scaling
#
# Tracks the hot paths the overhaul targets: the pseudospectral nonlinear term (DirectSum O(N²ᴰ)
# vs FFT O(Nᴰ log N)), spectral flux, shell-to-shell (serial vs threaded), and the resolved
# mode-to-mode tensor. GPU/MPI benchmarks live in gpu/ and need hardware. Uses BenchmarkTools.

using BenchmarkTools: BenchmarkTools, @benchmarkable, BenchmarkGroup, run, median
using FFTW: FFTW
using OhMyThreads: OhMyThreads
using Random: Random
using FlowInvariantTransfer: FlowInvariantTransfer as FET

# Incompressible 2D velocity coefficients (package convention û = fft(u)/Np) from a streamfunction.
function field2d(N; L = 2π, seed = 1)
    ks = FET.wavenumber_grid((N, N), (L, L))
    kx = [ks[1][i] for i in 1:N, j in 1:N]
    ky = [ks[2][j] for i in 1:N, j in 1:N]
    ψh = FFTW.fft(randn(Random.MersenneTwister(seed), N, N)) ./ N^2
    û  = cat(im .* ky .* ψh, -im .* kx .* ψh; dims = 3)
    return û, ks
end

const SUITE = BenchmarkGroup()
const SIZES = (32, 64, 128)
const THREADED = Threads.nthreads() > 1

for axis in ("nonlinear_term", "spectral_flux", "shell_to_shell", "mode_to_mode")
    SUITE[axis] = BenchmarkGroup()
end

for N in SIZES
    û, ks = field2d(N)
    b  = FET.LinearBinning(2π / (2π))
    ws = FET.NonlinearTermWorkspace(û, ks)

    # Nonlinear term: direct sum (O(N^{2D}), small grids only) vs FFT (O(Nᴰ log N)).
    if N <= 32
        SUITE["nonlinear_term"]["N$N/directsum"] =
            @benchmarkable FET.compute_nonlinear_term!($ws, $û, $ks; spectral = FET.DirectSumBackend()) evals=1
    end
    SUITE["nonlinear_term"]["N$N/fft"] =
        @benchmarkable FET.compute_nonlinear_term!($ws, $û, $ks; spectral = FET.FFTBackend()) evals=1

    # Spectral flux Π(K) (FFT path).
    SUITE["spectral_flux"]["N$N/fft"] =
        @benchmarkable FET.calculate_spectral_flux($û, $ks; binning = $b, spectral = FET.FFTBackend()) evals=1

    # Shell-to-shell T(n,m): serial vs threaded (FFT spectral).
    SUITE["shell_to_shell"]["N$N/serial"] =
        @benchmarkable FET.calculate_shell_to_shell_transfer($û, $ks; binning = $b,
            spectral = FET.FFTBackend(), execution = FET.SerialBackend(), verify_antisymmetry = false) evals=1
    if THREADED
        SUITE["shell_to_shell"]["N$N/threaded"] =
            @benchmarkable FET.calculate_shell_to_shell_transfer($û, $ks; binning = $b,
                spectral = FET.FFTBackend(), execution = FET.ThreadedBackend(), verify_antisymmetry = false) evals=1
    end

    # Resolved mode-to-mode S(k|p) — O(N^{2D}); only the smallest grid (force past the guard).
    if N == first(SIZES)
        SUITE["mode_to_mode"]["N$N/fft"] =
            @benchmarkable FET.calculate_mode_to_mode_transfer($û, $ks; spectral = FET.FFTBackend(), force = true) evals=1
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    println("Threads: ", Threads.nthreads(), "  FFTW threads: ", FFTW.get_num_threads())
    results = run(SUITE; verbose = true, seconds = 3)
    println("\n== median times ==")
    for axis in sort(collect(keys(results)))
        println("[$axis]")
        for key in sort(collect(keys(results[axis])))
            t = median(results[axis][key]).time / 1e6   # ms
            println("  ", rpad(key, 22), round(t; digits = 3), " ms")
        end
    end
end
