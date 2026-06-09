# FlowEnergyTransfer.jl

`FlowEnergyTransfer.jl` provides fast, minimally-allocating Julia implementations of the
standard methods for computing kinetic energy transfer across scales in turbulent flow:

- **Spectral flux** Π(K) — cumulative energy crossing wavenumber threshold K
- **Shell-to-shell transfer** T(n, m) — directed energy from mediator shell m to receiver shell n
- **Coarse-graining flux** Π_ℓ(x) — pointwise scale-to-scale flux at filter scale ℓ

---

## Installation

```julia
using Pkg
Pkg.add("FlowEnergyTransfer")
```

---

## Extension Architecture

The core package ships with a pure-Julia O(N²) direct-sum baseline requiring no compiled
dependencies. Load `FFTW` to activate the O(N log N) FFT fast path automatically:

| Method | Baseline | Fast Path | Required |
|--------|----------|-----------|----------|
| Spectral flux | `SerialBackend()` | `FFTBackend()` | `using FFTW` |
| Shell-to-shell | `SerialBackend()` | `FFTBackend()` | `using FFTW` |
| Coarse-graining | — | — | `using CoarseGrainingEnergyFluxes` |

---

## Quickstart: Spectral Flux Π(K)

```julia
using FlowEnergyTransfer
using FFTW   # activates FFTBackend

N = 64; L = 2π
ks = wavenumber_grid((N, N), (L, L))

# Build a divergence-free velocity field from a streamfunction
ψ̂ = randn(ComplexF64, N, N)
û = zeros(ComplexF64, N, N, 2)
for ix in 1:N, iy in 1:N
    û[ix, iy, 1] =  im * ks[2][iy] * ψ̂[ix, iy]   # u =  ∂ψ/∂y
    û[ix, iy, 2] = -im * ks[1][ix] * ψ̂[ix, iy]   # v = -∂ψ/∂x
end

result = calculate_spectral_flux(û, ks;
    binning    = LinearBinning(2π / L),
    dealiasing = true,
    backend    = FFTBackend())

result.k_shells          # shell-centre wavenumbers
result.transfer_spectrum # T(k) — energy transfer rate per shell
result.flux              # Π(K) — cumulative energy flux
```

## Quickstart: Shell-to-Shell Transfer T(n, m)

```julia
result = calculate_shell_to_shell_transfer(û, ks;
    binning            = LinearBinning(2π / L),
    dealiasing         = true,
    verify_antisymmetry = true,
    backend            = FFTBackend())

result.transfer_matrix        # T(n,m) — N_sh × N_sh
result.net_transfer           # Σ_m T(n,m) per receiver shell
result.max_antisymmetry_error # max|T(n,m)+T(m,n)| — should be ≈ 0
```

## Zero-Alloc Hot-Loop Usage

Preallocate once, reuse every timestep:

```julia
ws     = ShellToShellWorkspace(û, ks, LinearBinning(2π / L))
result = calculate_shell_to_shell_transfer(û, ks; ...)  # first call

# inside time loop — zero heap allocation:
calculate_shell_to_shell_transfer!(result, ws, û_new, ks)
```

---

## Example Figure

![Spectral flux and shell-to-shell transfer for a Taylor-Green vortex](assets/energy_transfer.png)

---

## See Also

- [Methods & Theory](@ref) — mathematical background for each method
- [API Reference](@ref) — full docstring index
