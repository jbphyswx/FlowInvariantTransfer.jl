# FlowInvariantTransfer.jl — CPU benchmark suite.
#
#   julia --project=benchmark benchmark/benchmarks.jl
#   julia --threads=4 --project=benchmark benchmark/benchmarks.jl     # adds the threaded rows
#   FIT_BENCH_2D=64,128 FIT_BENCH_3D=16,32 julia --project=benchmark benchmark/benchmarks.jl
#   FIT_BENCH_BASELINE=benchmark/baseline_pre_rewrite.json julia --project=benchmark benchmark/benchmarks.jl
#
# Every axis appears twice: a `hot` row that reuses a prebuilt workspace (what a production loop
# costs per snapshot) and a `convenience` row that calls the allocating wrapper (which rebuilds the
# workspace and re-plans every transform). Alongside the timings, a resident-scratch table reports
# the bytes each workspace owns and the bytes per grid point — the size-independent figure the buffer
# inventory is written against, which `BenchmarkTools`'s per-call `memory` cannot show.
#
# GPU and MPI benchmarks live in gpu/ and test/mpi/ and need hardware.

using BenchmarkTools: BenchmarkTools
using FFTW: FFTW
using LinearAlgebra: LinearAlgebra
using OhMyThreads: OhMyThreads
using Random: Random
using JSON3: JSON3
using FlowInvariantTransfer: FlowInvariantTransfer as FIT

# Extension triggers: coarse-graining (CGEF) and both scattered-NUFFT providers are benchmarked axes.
using CoarseGrainingEnergyFluxes: CoarseGrainingEnergyFluxes
using NonuniformFFTs: NonuniformFFTs
using FINUFFT: FINUFFT

include("harness.jl")

const SIZES_2D = let s = get(ENV, "FIT_BENCH_2D", "")
    isempty(s) ? (128, 256) : Tuple(parse(Int, strip(x)) for x in split(s, ','))
end
const SIZES_3D = let s = get(ENV, "FIT_BENCH_3D", "")
    isempty(s) ? (32, 64) : Tuple(parse(Int, strip(x)) for x in split(s, ','))
end
const NPTS = parse(Int, get(ENV, "FIT_BENCH_NPTS", "200000"))   # scattered-point count
const THREADED = Threads.nthreads() > 1
const FFT = FIT.SpectralBackends.FFTSpectralBackend()
const SER = FIT.ComputationalBackends.SerialBackend()
const THR = FIT.ComputationalBackends.ThreadedBackend()

const SUITE = BenchmarkTools.BenchmarkGroup()
const MEMROWS = Tuple{String, Any, Tuple}[]

function group(axis)
    haskey(SUITE, axis) || (SUITE[axis] = BenchmarkTools.BenchmarkGroup())
    return SUITE[axis]
end

# Shell structure for a grid, matching what every workspace builds internally.
function shells(ks, binning)
    kmag = FIT.ShellBinning.shell_coordinate(FIT.Types.IsotropicShells(), ks)
    kmax = maximum(kmag)
    edges = FIT.ShellBinning.shell_edges(binning, kmax)
    centers = collect(FIT.ShellBinning.shell_centers(binning, kmax))
    return (edges = edges, centers = centers, idx = FIT.ShellBinning.assign_shells(kmag, edges))
end

# ---------------------------------------------------------------------------
# Uniform-grid Cartesian axes, 2D and 3D
# ---------------------------------------------------------------------------

function add_uniform!(ns::NTuple{D,Int}) where {D}
    tag = "$(D)D/N$(ns[1])"
    û, ks = field_nd(ns)
    b = FIT.Types.LinearBinning(1.0)
    sh = shells(ks, b)
    RT = Float64

    # --- nonlinear term (the engine every Cartesian diagnostic sits on) ---
    nlw = FIT.Workspaces.NonlinearTermWorkspace(û, ks)
    group("nonlinear_term")[HOT * "/" * tag] =
        BenchmarkTools.@benchmarkable FIT.NonlinearTerm.compute_nonlinear_term!($nlw, $û, $ks; spectral = $FFT) evals = 1
    push!(MEMROWS, ("NonlinearTermWorkspace $tag", nlw, ns))

    # --- spectral flux ---
    sfw = FIT.Workspaces.SpectralFluxWorkspace(û, ks, b)
    sfres = FIT.Types.SpectralFluxResult(sh.centers, similar(sh.centers, RT), similar(sh.centers, RT))
    group("spectral_flux")[HOT * "/" * tag] =
        BenchmarkTools.@benchmarkable FIT.SpectralFlux.calculate_spectral_flux!(
            $sfres, $sfw, $û, $ks, $(sh.idx); spectral = $FFT, execution = $SER) evals = 1
    group("spectral_flux")[CONV * "/" * tag] =
        BenchmarkTools.@benchmarkable FIT.SpectralFlux.calculate_spectral_flux(
            $û, $ks; binning = $b, spectral = $FFT, execution = $SER) evals = 1
    push!(MEMROWS, ("SpectralFluxWorkspace $tag", sfw, ns))

    # --- shell-to-shell ---
    s2sw = FIT.Workspaces.ShellToShellWorkspace(û, ks, b)
    nsh = length(sh.centers)
    s2sres = FIT.Types.ShellToShellResult(sh.centers, sh.edges,
                                          Matrix{RT}(undef, nsh, nsh), Vector{RT}(undef, nsh), RT(NaN))
    group("shell_to_shell")[HOT * "/" * tag] =
        BenchmarkTools.@benchmarkable FIT.ShellToShellTransfer.calculate_shell_to_shell_transfer!(
            $s2sres, $s2sw, $û, $ks; spectral = $FFT, execution = $SER, verify_antisymmetry = false) evals = 1
    if THREADED
        group("shell_to_shell")[HOT * "/threaded/" * tag] =
            BenchmarkTools.@benchmarkable FIT.ShellToShellTransfer.calculate_shell_to_shell_transfer!(
                $s2sres, $s2sw, $û, $ks; spectral = $FFT, execution = $THR, verify_antisymmetry = false) evals = 1
    end
    push!(MEMROWS, ("ShellToShellWorkspace $tag  (N_sh=$nsh)", s2sw, ns))

    # --- smooth band-to-band ---
    bands = FIT.Types.SmoothBands([2.0, 4.0, 8.0, 16.0])
    nb = length(bands.centers)
    bws = FIT.BandTransfer.BandTransferWorkspace(û, ks, bands)
    Tb = zeros(RT, nb, nb); netb = zeros(RT, nb)
    group("band_to_band")[HOT * "/" * tag] =
        BenchmarkTools.@benchmarkable FIT.BandTransfer.calculate_band_to_band_transfer!(
            $Tb, $netb, $bws, $û, $ks; spectral = $FFT, execution = $SER) evals = 1
    push!(MEMROWS, ("BandTransferWorkspace $tag  (nb=$nb)", bws, ns))

    # --- compressible momentum-weighted flux ---
    ρ̂ = density_nd(ns)
    cws = FIT.Compressible.CompressibleWorkspace(û, ks; spectral = FFT, binning = b)
    group("compressible")[HOT * "/" * tag] =
        BenchmarkTools.@benchmarkable FIT.Compressible.calculate_compressible_flux!($cws, $û, $ρ̂, $ks) evals = 1
    group("compressible")[CONV * "/" * tag] =
        BenchmarkTools.@benchmarkable FIT.Compressible.calculate_compressible_flux(
            $û, $ρ̂, $ks; binning = $b, spectral = $FFT) evals = 1
    push!(MEMROWS, ("CompressibleWorkspace $tag", cws, ns))

    # --- decomposition + partial fluxes (3D only: helical needs 3 components) ---
    if D == 3
        group("decomposition")[HOT * "/helical/" * tag] =
            BenchmarkTools.@benchmarkable FIT.Decomposition.decompose_field(
                FIT.Types.HelicalDecomposition(), $û, $ks) evals = 1
        pfw = FIT.Workspaces.NonlinearTermWorkspace(û, ks)
        group("partial_fluxes")[HOT * "/helical/" * tag] =
            BenchmarkTools.@benchmarkable FIT.calculate_helical_partial_fluxes!(
                $pfw, $û, $ks; binning = $b, spectral = $FFT) evals = 1
    end

    # --- uniform-grid physical → spectral (no workspace exists: convenience only) ---
    coords = ntuple(d -> collect(range(0, 2π; length = ns[d] + 1)[1:ns[d]]), D)
    uphys = ntuple(c -> real.(FFTW.bfft(view(û, ntuple(_ -> Colon(), D)..., c))), D)
    group("to_spectral_uniform")[CONV * "/" * tag] =
        BenchmarkTools.@benchmarkable FIT.to_spectral($uphys, $coords; spectral = $FFT) evals = 1

    return nothing
end

# ---------------------------------------------------------------------------
# Mode-to-mode S(k|p) — O(N^{2D}) output, smallest grid only
# ---------------------------------------------------------------------------

function add_mode_to_mode!()
    ns = (16, 16)
    û, ks = field_nd(ns)
    ws = FIT.Workspaces.NonlinearTermWorkspace(û, ks)
    û_p = similar(û)
    S = similar(û, Float64, ns..., ns...)
    net = similar(û, Float64, ns...)
    res = FIT.Types.ModeToModeTriadResult(FIT.Types.KineticEnergy(), ks, net, S)
    group("mode_to_mode")[HOT * "/2D/N16"] =
        BenchmarkTools.@benchmarkable FIT.calculate_mode_to_mode_transfer!(
            $res, $ws, $û_p, $û, $ks; spectral = $FFT, execution = $SER) evals = 1
    return nothing
end

# ---------------------------------------------------------------------------
# Scattered (NUFFT) axes — both providers, and the CGEF uniform coarse-graining
# ---------------------------------------------------------------------------

function add_scattered!(D::Int, ms::NTuple{N,Int}) where {N}
    fields, coords = scattered_nd(NPTS, D)
    Ls = ntuple(_ -> 2π, D)
    tag = "$(D)D/np$(NPTS)/ms$(ms[1])"
    ℓ = 0.5
    filt = FIT.Types.GaussianFilter()
    for (name, be) in (("nonuniformffts", FIT.Types.NonuniformFFTsBackend()),
                       ("finufft", FIT.Types.FINUFFTBackend()))
        tsw = FIT.NUFFTToSpectralWorkspace(coords, ms; spectral = be, Ls = Ls, ncomponents = D, tol = 1e-9)
        group("to_spectral_scattered")[HOT * "/$name/" * tag] =
            BenchmarkTools.@benchmarkable FIT.to_spectral!($tsw, $fields) evals = 1
        push!(MEMROWS, ("NUFFTToSpectralWorkspace $name $tag", tsw, ms))

        cgw = FIT.NUFFTCoarseGrainingWorkspace(coords, ms; spectral = be, Ls = Ls, tol = 1e-8)
        group("nufft_coarse_graining")[HOT * "/$name/" * tag] =
            BenchmarkTools.@benchmarkable FIT.nufft_coarse_graining_flux!(
                $cgw, $fields, $ℓ, $filt, $ms) evals = 1
        group("nufft_coarse_graining")[CONV * "/$name/" * tag] =
            BenchmarkTools.@benchmarkable FIT.nufft_coarse_graining_flux(
                $fields, $coords, $ℓ, $filt, $ms; spectral = $be, Ls = $Ls) evals = 1
        push!(MEMROWS, ("NUFFTCoarseGrainingWorkspace $name $tag", cgw, ms))
    end
    return nothing
end

function add_cgef!(n::Int)
    xs = collect(range(0, 2π; length = n + 1)[1:n]); ys = copy(xs)
    u = [sin(x) * cos(y) for x in xs, y in ys]
    v = [-cos(x) * sin(y) for x in xs, y in ys]
    ℓ = 2π / 16   # ~16 cells per filter window, sized off the grid rather than a fixed fraction
    filt = FIT.Types.GaussianFilter()
    ws = FIT.CoarseGrainingFlux.CoarseGrainingFluxWorkspace((u, v), (xs, ys), ℓ, filt; backend = SER)
    tag = "2D/N$n"
    group("cgef_coarse_graining")[HOT * "/" * tag] =
        BenchmarkTools.@benchmarkable FIT.CoarseGrainingFlux.calculate_coarse_graining_flux!($ws, ($u, $v)) evals = 1
    group("cgef_coarse_graining")[CONV * "/" * tag] =
        BenchmarkTools.@benchmarkable FIT.CoarseGrainingFlux.calculate_coarse_graining_flux(
            ($u, $v), ($xs, $ys), $ℓ, $filt) evals = 1
    push!(MEMROWS, ("CoarseGrainingFluxWorkspace $tag", ws, (n, n)))
    return nothing
end

for n in SIZES_2D; add_uniform!((n, n)); end
for n in SIZES_3D; add_uniform!((n, n, n)); end
add_mode_to_mode!()
add_scattered!(2, (256, 256))
add_scattered!(3, (64, 64, 64))
add_cgef!(last(SIZES_2D))

# ---------------------------------------------------------------------------

if abspath(PROGRAM_FILE) == @__FILE__
    # Configuration first, never a timing: a flat result is a configuration smell before it is a finding.
    println("Julia threads : ", Threads.nthreads())
    println("FFTW threads  : ", FFTW.get_num_threads())
    println("BLAS threads  : ", LinearAlgebra.BLAS.get_num_threads())
    println("2D sizes      : ", SIZES_2D, "   3D sizes: ", SIZES_3D, "   scattered points: ", NPTS)

    seconds = parse(Float64, get(ENV, "FIT_BENCH_SECONDS", "3"))
    results = BenchmarkTools.run(SUITE; verbose = true, seconds = seconds)
    report_times(results)
    report_memory(MEMROWS)

    out = get(ENV, "FIT_BENCH_SAVE", "")
    if !isempty(out)
        payload = Dict(
            "threads" => Threads.nthreads(),
            "times_ns" => Dict(axis => Dict(k => BenchmarkTools.median(results[axis][k]).time
                                            for k in keys(results[axis])) for axis in keys(results)),
            "allocs" => Dict(axis => Dict(k => BenchmarkTools.median(results[axis][k]).memory
                                          for k in keys(results[axis])) for axis in keys(results)),
            "workspace_bytes" => Dict(label => workspace_bytes(ws) for (label, ws, _) in MEMROWS),
        )
        open(out, "w") do io
            JSON3.pretty(io, payload)
        end
        println("\nsaved → ", out)
    end
end
