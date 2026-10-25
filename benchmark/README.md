# Benchmarks

CPU benchmark suite covering the hot paths across their parallel axes.

```bash
julia --project=benchmark benchmark/benchmarks.jl                  # full sweep, serial paths
julia --threads=4 --project=benchmark benchmark/benchmarks.jl      # adds the threaded rows
FIT_BENCH_2D=64 FIT_BENCH_3D=16 FIT_BENCH_SECONDS=0.5 \
  julia --project=benchmark benchmark/benchmarks.jl                # fast smoke (small grids)
```

| variable | sets | default |
|---|---|---|
| `FIT_BENCH_2D` | 2-D grid sizes, comma-separated | `128,256` |
| `FIT_BENCH_3D` | 3-D grid sizes, comma-separated | `32,64` |
| `FIT_BENCH_NPTS` | scattered Cartesian point count | `200000` |
| `FIT_BENCH_LMAX` | scattered-sphere degree (four points per coefficient) | `64` |
| `FIT_BENCH_SECONDS` | time budget per benchmark | `3` |
| `FIT_BENCH_SAVE` | JSON file the medians, allocations and workspace sizes are written to | none |

Covered, each reporting median time **and** allocations:

- pseudospectral nonlinear term
- spectral flux `Π(K)`
- shell-to-shell `T(n,m)` — serial vs threaded
- smooth band-to-band `T(K,Q)`
- compressible momentum-weighted flux
- helical decomposition and partial fluxes (3-D)
- resolved scale-to-scale `S(k|p)` (smallest grid only; `O(N^{2D})`)
- physical → spectral, uniform and scattered (both NUFFT libraries); NUFFT and CGEF coarse-graining flux
- spherical transfer from vorticity at scattered points

A second table divides each median by the operation the method is built from, timed alone: one real
FFT of the same grid on one thread for the Cartesian axes, and one iteration of the vorticity fit for
the scattered sphere. A row then reads as the number of those operations the call costs; the spectral
flux needs 15 real FFTs in 3-D and 8 in 2-D. Absolute numbers are machine- and size-dependent, so none
are quoted here — run the suite for your hardware.

## GPU

`../gpu/benchmarks.jl` measures shell-to-shell on a CUDA device (falls back to the
KernelAbstractions CPU backend if no GPU is functional). The device kernels' *correctness* is
validated on the KA CPU backend in the test suite; this script measures on-device performance and
needs real hardware.

```bash
julia --project=gpu gpu/benchmarks.jl
```
