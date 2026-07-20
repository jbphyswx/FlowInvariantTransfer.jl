# Benchmarks

CPU benchmark suite covering the hot paths across their parallel axes.

```bash
julia --project=benchmark benchmark/benchmarks.jl                  # full sweep, serial paths
julia --threads=4 --project=benchmark benchmark/benchmarks.jl      # adds the threaded rows
FIT_BENCH_SIZES=8,16 FIT_BENCH_SECONDS=0.5 \
  julia --project=benchmark benchmark/benchmarks.jl                # fast smoke (tiny grids)
```

`FIT_BENCH_SIZES` (comma-separated grid sizes) and `FIT_BENCH_SECONDS` (per-benchmark budget) override
the defaults `32,64,128` / `3`.

Covered, each reporting median time **and** allocations:

- pseudospectral nonlinear term — `DirectSumBackend` `O(N^{2D})` vs `FFTBackend` `O(Nᴰ log N)`
- spectral flux `Π(K)` — serial vs threaded
- shell-to-shell `T(n,m)` — serial vs threaded
- smooth band-to-band `T(K,Q)` — serial vs threaded
- compressible momentum-weighted flux — serial vs threaded
- resolved mode-to-mode `S(k|p)` — serial vs threaded (smallest grid only; `O(N^{2D})`)

Built on `BenchmarkTools`. Absolute numbers are machine- and size-dependent, so none are quoted here —
run the suite for your hardware. The one hardware-independent result is asymptotic: the `DirectSum → FFT`
nonlinear term replaces `O(N^{2D})` with `O(Nᴰ log N)`, a gap that widens with `N`.

## GPU

`../gpu/benchmarks.jl` measures shell-to-shell on a CUDA device (falls back to the
KernelAbstractions CPU backend if no GPU is functional). The device kernels' *correctness* is
validated on the KA CPU backend in the test suite; this script measures on-device performance and
needs real hardware.

```bash
julia --project=gpu gpu/benchmarks.jl
```
