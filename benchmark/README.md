# Benchmarks

CPU benchmark suite for the hot paths the overhaul targets.

```bash
julia --project=benchmark benchmark/benchmarks.jl              # serial + FFT paths
julia --threads=4 --project=benchmark benchmark/benchmarks.jl  # adds threaded shell-to-shell scaling
```

Covered: pseudospectral nonlinear term (`DirectSumBackend` `O(N^{2D})` vs `FFTBackend`
`O(Nᴰ log N)`), spectral flux `Π(K)`, shell-to-shell `T(n,m)` (serial vs threaded), and the
resolved mode-to-mode `S(k|p)` tensor. Built on `BenchmarkTools`.

Representative medians on a 2D field (Apple-silicon laptop, 4 threads — your numbers will differ):

| path | N=32 | N=64 | N=128 |
|------|-----:|-----:|------:|
| nonlinear term — DirectSum | 125 ms | — | — |
| nonlinear term — FFT | 0.05 ms | 0.27 ms | 1.4 ms |
| spectral flux — FFT | 0.14 ms | 0.85 ms | 4.6 ms |
| shell-to-shell — serial | 4.9 ms | 41 ms | 381 ms |
| shell-to-shell — threaded (×4) | 1.6 ms | 16 ms | 186 ms |
| mode-to-mode `S(k\|p)` — FFT | 60 ms | — | — |

The DirectSum→FFT nonlinear term is ~2500× at N=32 (the `O(N^{2D})→O(Nᴰ log N)` win); threading
gives ~2–2.5× on 4 cores for shell-to-shell.

## GPU

`../gpu/benchmarks.jl` measures shell-to-shell on a CUDA device (falls back to the
KernelAbstractions CPU backend if no GPU is functional). The device kernels' *correctness* is
validated on the KA CPU backend in the test suite; this script measures on-device performance and
needs real hardware.

```bash
julia --project=gpu gpu/benchmarks.jl
```
