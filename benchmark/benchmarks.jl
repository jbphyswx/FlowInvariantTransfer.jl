# FlowInvariantTransfer.jl — CPU benchmark suite.
#
#   julia --project=benchmark benchmark/benchmarks.jl            # run all
#   julia --threads=4 --project=benchmark benchmark/benchmarks.jl  # include threaded scaling
#
# Tracks the hot paths across their parallel axes: the pseudospectral nonlinear term (DirectSum
# O(N²ᴰ) vs FFT O(Nᴰ log N)), and spectral flux / shell-to-shell / band-to-band / compressible /
# mode-to-mode each serial vs threaded (FFT spectral), over a grid-size sweep. Reports time AND
# allocations. Threaded rows appear only under `--threads>1`. GPU/MPI benchmarks live in gpu/ and
# need hardware. Uses BenchmarkTools.

using BenchmarkTools: BenchmarkTools, @benchmarkable, BenchmarkGroup, run, median
using FFTW: FFTW
using OhMyThreads: OhMyThreads
using Random: Random
using FlowInvariantTransfer: FlowInvariantTransfer as FIT

# Incompressible 2D velocity coefficients (package convention û = fft(u)/Np) from a streamfunction.
function field2d(N; L = 2π, seed = 1)
    ks = FIT.Utils.wavenumber_grid((N, N), (L, L))
    kx = [ks[1][i] for i in 1:N, j in 1:N]
    ky = [ks[2][j] for i in 1:N, j in 1:N]
    ψh = FFTW.fft(randn(Random.MersenneTwister(seed), N, N)) ./ N^2
    û  = cat(im .* ky .* ψh, -im .* kx .* ψh; dims = 3)
    return û, ks
end

# Real positive density field's coefficients (same û = fft(ρ)/Np convention) for compressible transfer.
function density2d(N)
    ρ = [1.0 + 0.1 * cospi(2i / N) * sinpi(2j / N) for i in 1:N, j in 1:N]
    return FFTW.fft(ρ) ./ N^2
end

const SUITE = BenchmarkGroup()
# Grid sizes swept. Override for a fast CI smoke, e.g. `FIT_BENCH_SIZES=8,16`.
const SIZES = let s = get(ENV, "FIT_BENCH_SIZES", "")
    isempty(s) ? (32, 64, 128) : Tuple(parse(Int, strip(x)) for x in split(s, ','))
end
const THREADED = Threads.nthreads() > 1

for axis in ("nonlinear_term", "spectral_flux", "shell_to_shell", "mode_to_mode",
             "band_to_band", "compressible")
    SUITE[axis] = BenchmarkGroup()
end

for N in SIZES
    û, ks = field2d(N)
    b  = FIT.LinearBinning(2π / (2π))
    ws = FIT.NonlinearTermWorkspace(û, ks)

    # Nonlinear term: direct sum (O(N^{2D}), small grids only) vs FFT (O(Nᴰ log N)).
    if N <= 32
        SUITE["nonlinear_term"]["N$N/directsum"] =
            @benchmarkable FIT.NonlinearTerm.compute_nonlinear_term!($ws, $û, $ks; spectral = FIT.DirectSumBackend()) evals=1
    end
    SUITE["nonlinear_term"]["N$N/fft"] =
        @benchmarkable FIT.NonlinearTerm.compute_nonlinear_term!($ws, $û, $ks; spectral = FIT.FFTBackend()) evals=1

    # Spectral flux Π(K): serial vs threaded (FFT spectral).
    SUITE["spectral_flux"]["N$N/serial"] =
        @benchmarkable FIT.calculate_spectral_flux($û, $ks; binning = $b,
            spectral = FIT.FFTBackend(), execution = FIT.SerialBackend()) evals=1
    if THREADED
        SUITE["spectral_flux"]["N$N/threaded"] =
            @benchmarkable FIT.calculate_spectral_flux($û, $ks; binning = $b,
                spectral = FIT.FFTBackend(), execution = FIT.ThreadedBackend()) evals=1
    end

    # Shell-to-shell T(n,m): serial vs threaded (FFT spectral).
    SUITE["shell_to_shell"]["N$N/serial"] =
        @benchmarkable FIT.calculate_shell_to_shell_transfer($û, $ks; binning = $b,
            spectral = FIT.FFTBackend(), execution = FIT.SerialBackend(), verify_antisymmetry = false) evals=1
    if THREADED
        SUITE["shell_to_shell"]["N$N/threaded"] =
            @benchmarkable FIT.calculate_shell_to_shell_transfer($û, $ks; binning = $b,
                spectral = FIT.FFTBackend(), execution = FIT.ThreadedBackend(), verify_antisymmetry = false) evals=1
    end

    # Smooth band-to-band T(K,Q): serial vs threaded (one nonlinear term per band, FFT spectral).
    bands = FIT.SmoothBands([2.0, 4.0, 8.0])
    SUITE["band_to_band"]["N$N/serial"] =
        @benchmarkable FIT.calculate_band_to_band_transfer($û, $ks; bands = $bands,
            spectral = FIT.FFTBackend(), execution = FIT.SerialBackend()) evals=1
    if THREADED
        SUITE["band_to_band"]["N$N/threaded"] =
            @benchmarkable FIT.calculate_band_to_band_transfer($û, $ks; bands = $bands,
                spectral = FIT.FFTBackend(), execution = FIT.ThreadedBackend()) evals=1
    end

    # Compressible momentum-weighted flux: serial vs threaded (Helmholtz channels, FFT spectral).
    ρ̂ = density2d(N)
    SUITE["compressible"]["N$N/serial"] =
        @benchmarkable FIT.calculate_compressible_flux($û, $ρ̂, $ks; binning = $b,
            spectral = FIT.FFTBackend(), execution = FIT.SerialBackend()) evals=1
    if THREADED
        SUITE["compressible"]["N$N/threaded"] =
            @benchmarkable FIT.calculate_compressible_flux($û, $ρ̂, $ks; binning = $b,
                spectral = FIT.FFTBackend(), execution = FIT.ThreadedBackend()) evals=1
    end

    # Resolved mode-to-mode S(k|p) — O(N^{2D}); only the smallest grid (force past the guard).
    if N == first(SIZES)
        SUITE["mode_to_mode"]["N$N/serial"] =
            @benchmarkable FIT.calculate_mode_to_mode_transfer($û, $ks;
                spectral = FIT.FFTBackend(), execution = FIT.SerialBackend(), force = true) evals=1
        if THREADED
            SUITE["mode_to_mode"]["N$N/threaded"] =
                @benchmarkable FIT.calculate_mode_to_mode_transfer($û, $ks;
                    spectral = FIT.FFTBackend(), execution = FIT.ThreadedBackend(), force = true) evals=1
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    println("Threads: ", Threads.nthreads(), "  FFTW threads: ", FFTW.get_num_threads())
    seconds = parse(Float64, get(ENV, "FIT_BENCH_SECONDS", "3"))   # shorten for a CI smoke
    results = run(SUITE; verbose = true, seconds = seconds)
    println("\n== median time / allocations ==")
    for axis in sort(collect(keys(results)))
        println("[$axis]")
        for key in sort(collect(keys(results[axis])))
            m = median(results[axis][key])
            t   = m.time / 1e6      # ms
            mem = m.memory / 1024   # KiB
            println("  ", rpad(key, 22), lpad(round(t; digits = 3), 10), " ms  ",
                    lpad(round(mem; digits = 1), 10), " KiB  (", m.allocs, " allocs)")
        end
    end
end
