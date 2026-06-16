# FlowInvariantTransfer.jl — Theory & Knowledge Base

> Living theoretical record for the package: definitions, relationships, what we implement, and how it maps to related packages.

## 0. Scope

`FlowInvariantTransfer.jl` computes **cross-scale transfer of quadratic inviscid invariants** (primarily kinetic energy) in turbulent flows, in Fourier space and via coarse-graining. Energy *spectra* `E(k)` live in the sibling package `FlowFieldSpectra.jl`; pointwise coarse-graining flux machinery lives in `CoarseGrainingEnergyFluxes.jl` (CGEF) and is wrapped here via an extension.

---

## 1. Governing equations & the nonlinear term

Incompressible Navier–Stokes on a periodic domain:
```
∂_t u + (u·∇)u = −∇p + ν∇²u + f,    ∇·u = 0.
```
All inter-scale transfer originates in the quadratic nonlinearity `(u·∇)u`. In Fourier space, with the Leray projector `P(k) = I − kk/|k|²` enforcing incompressibility:
```
N̂(k) = −P(k) · FFT[(u·∇)u]
```
`N̂(k)` is the engine of every transfer diagnostic below. (Implemented in `NonlinearTerm.jl`.)

---

## 2. Inviscid quadratic invariants

The same machinery applies to any quadratic invariant; only the inner product changes.

| Invariant | Definition | Cascade direction | Where it matters |
|---|---|---|---|
| **Kinetic energy** `E` | `½∫|u|²` | forward (3D), inverse (2D) | universal |
| **Helicity** `H` (3D) | `∫ u·ω`, `ω=∇×u` | forward; co-directional with `E` | 3D, rotating |
| **Enstrophy** `Ω` (2D) | `½∫ ω²` | forward; counter-directional to `E` | 2D, geophysical |

2D turbulence: **counter-directional dual** energy–enstrophy cascade (Kraichnan–Batchelor). 3D: **co-directional** energy–helicity dual cascade. (Alexakis & Biferale 2018, §3.5–3.6.)

---

## 3. Fourier-space diagnostics

### 3.1 Transfer spectrum `T(k)` and flux `Π(K)`
```
T(k_n) = Σ_{|k|∈shell n} Re{ û*(k) · N̂(k) }
Π(K)   = −Σ_{k_n ≤ K} T(k_n)   (= energy crossing K to smaller scales)
```
`Π>0`: forward (downscale) cascade; `Π<0`: inverse. (Implemented in `SpectralFlux.jl`.)

### 3.2 Shell-to-shell transfer `T(n,m)`
Directed energy transfer rate **into** shell `n` **from** shell `m`, mediated by the nonlinear term:
```
T(n,m) = Σ_{k∈S_n} Re{ û*(k) · N̂_m(k) },   N̂_m = FFT[(u_m·∇)u],  u_m = IFFT(û·χ_m)
```
Antisymmetric: `T(n,m) = −T(m,n)`. Reduces to `T(k)` when summed over `m`. (Implemented in `ShellToShell/`.)
References: Domaradzki & Rogallo (1990); Verma (2002); Alexakis, Mininni & Pouquet (2005).

### 3.3 Mode-to-mode triad transfer `S(k|p|q)` — THE fundamental object
Energy given **to** receiver mode `k` **from** giver `p`, mediated by `q`, with triad closure `k = p + q`:
```
S(k|p|q) = −Im{ [k · û(q)] [û*(k) · û(p)] }
```
Properties (and tests):
- Giver/receiver antisymmetry: `S(k|p|q) = −S(p|k|q)`.
- Net transfer: `T(k) = Σ_p S(k|p|q=k−p)`.
- Conservation: `Σ_k T(k) = 0`.
- Hermitian symmetry: `û(−k) = û*(k)`.
- p/q split within a triad is ambiguous; only `S(k|p|q)+S(k|q|p)` is unambiguous ("combined" transfer, Kraichnan).
Cost: `O(N^D)` per receiver, `O(N^{2D})` for the full tensor — slow, exact.
References: Dar, Verma & Eswaran (2001); Verma (2004 review, 2019 book). Compressible extension: arXiv:2508.04300 (2025).

### 3.4 The reduction hierarchy (delta vs band)
A "scale" is a single wavenumber; only a **delta** kernel isolates one. Finite-width kernels bin a **band** → that is shell-to-shell, not scale-to-scale.
```
S(k|p|q)   delta in vector k   most fundamental, directional, O(N^{2D})
   │  sum over directions on |k|=K, |p|=Q
   ▼
T(K,Q)     delta in |k|        exact magnitude-to-magnitude (Δk→0 shell limit)
   │  finite shell width
   ▼
T(n,m)     sharp Fourier shells  implemented
   │  sum over givers / smooth bands
   ▼
T(k), Π(K)            implemented
```

---

## 4. Coarse-graining / filtering framework (physical space)

Filter: `ū_ℓ(x) = (G_ℓ * u)(x)`, spectral multiplier `Ĝ_ℓ(k)`. SGS stress and flux:
```
τ_ℓ,ij = (u_i u_j)‾_ℓ − ū_i ū_j
Π_ℓ(x) = −τ_ℓ,ij S̄_ij,    S̄_ij = ½(∂_i ū_j + ∂_j ū_i)
```
For a **sharp spectral** projector, `⟨Π_ℓ⟩ = Π(k=1/ℓ)` — the space-average equals the Fourier flux (Alexakis & Biferale §2.5). Smooth (Gaussian/top-hat) kernels give better physical-space locality (Eyink & Aluie 2009 I/II).

**Filtering spectrum:** `E(ℓ) = ½⟨|ū_ℓ|²⟩` (a scale-space energy spectrum).

**Status:** `Π_ℓ(x)` and `E(ℓ)` are provided by **CGEF** (`compute_Π!`, `compute_filtering_spectrum`, `coarse_grain`) and wrapped here via `CoarseGrainingFlux.jl` + the CGEF extension. CGEF supports Gaussian/top-hat/sharp kernels, spherical & Cartesian grids, land masks. **CGEF does NOT produce a band-to-band matrix** — a matrix is not part of the coarse-graining framework; its canonical outputs are `Π_ℓ(x)` (function of one scale `ℓ`) and `E(ℓ)`.

### 4.1 Smooth band-to-band `T(K,Q)` (Eyink–Aluie band-pass)
Eyink & Aluie (2009, arXiv:0909.2386) decompose KE into **band-pass** contributions using smooth graded filters and define **inter-band** transfer. This is the *smooth-kernel generalization of shell-to-shell* (bands = `ū_{ℓ_n} − ū_{ℓ_{n+1}}`). Canonical, but a band (finite kernel) is still a shell — so this belongs in `ShellToShell` as a smooth-band binning option, NOT in a "scale-to-scale" (delta) module.

Note: DOI 10.1029/2020MS002090 ("On Energy Cascades in General Flows: A Lagrangian Application") is the **same `Π_ℓ` coarse-graining family** (Aluie group), not a new object.

---

## 5. Partial-flux decompositions (Alexakis & Biferale §3.6.2)

The total flux can be split into physically meaningful **partial fluxes** `Π = Σ_i Π_i`. Three DISTINCT decompositions (often confused):

| Decomposition | Splits | Basis | Available where |
|---|---|---|---|
| **Helmholtz** | rotational (non-divergent) `u_rot=∇×ψ` + divergent (irrotational) `u_div=∇φ` | streamfunction/velocity-potential (Poisson solve) | **`HelmholtzDecomposition.jl`** (dedicated pkg: Cartesian+spherical, regular & scattered, SOR + FFTW/FINUFFT/FSH/NUFSHT solvers, `AutoSolver`). CGEF also has an internal 2D helper. |
| **Helical `±`** | curl eigenmodes `û(k)=u_+ ĥ_+ + u_- ĥ_-`, `i k×ĥ_± = ±|k|ĥ_±` | diagonalizes helicity; `E=|u_+|²+|u_-|²`, `H=|k|(|u_+|²−|u_-|²)` | not yet (this package) |
| **Toroidal/poloidal** | splits the solenoidal (`∇·u=0`) part: `u = ∇×(ψẑ) + ∇×∇×(φẑ)` | 3D stratified/rotating | not yet |

**Key clarification:** Helmholtz (rot/div) is **NOT** the same as helical or toroidal/poloidal. Helmholtz separates divergent from rotational flow; helical splits the rotational part into two curl-eigenmode chiralities; toroidal/poloidal splits the solenoidal part by geometry. The dedicated **`HelmholtzDecomposition.jl`** package provides the rot/div split (geometry-aware, multi-solver) and explicitly targets *separating energy flux Π into toroidal/potential contributions* (Buzzicotti et al. 2023) — so we leverage it for partial fluxes. (NB: this *spatial* Poisson-based Helmholtz differs from the Lindborg-2015 structure-function integral relations of the same name in `StructureFunctions.jl`.) Helical `±` and toroidal/poloidal are separate, additional work.
References: Aluie (2019); Buzzicotti, Storer, Khatri, Griffies & Aluie (2023); Waleffe (1992); Constantin & Majda (1988); Biferale, Musacchio & Toschi (2012).

---

## 6. Anisotropic / directional shells

For rotating/stratified/geophysical flows, isotropic `|k|` shells are inadequate; one uses **2D shells** in `(k_⊥, k_∥)` (perpendicular/parallel to rotation axis or gravity). Fluxes become directional (`Π_⊥`, `Π_∥`). Larger, geometry-specific effort. (Alexakis & Biferale §4.3–4.5.)

---

## 7. Triadic Orthogonal Decomposition (TOD)

Frequency-domain modal decomposition of triadic nonlinear interactions via the mode bispectrum / SVD, on temporal snapshots. Complements the spatial Fourier diagnostics. (Yeung, Chu & Schmidt 2026.) Implemented in `ScaleToScale/TriadicOrthogonalDecomposition/`.

---

## 8. Canonical coverage map (verified against Alexakis & Biferale 2018)

| Diagnostic | Status |
|---|---|
| Energy spectrum `E(k)` | `FlowFieldSpectra.jl` (sibling) |
| Spectral flux `Π(k)` / transfer `T(k)` | done (`SpectralFlux`) |
| Shell-to-shell `T(n,m)` (sharp Fourier) | done (`ShellToShell`) |
| Coarse-graining flux `Π_ℓ(x)`, `E(ℓ)` (smooth/sharp kernels) | done (CGEF + ext) |
| Triadic Orthogonal Decomposition | done (`TOD`) |
| **Mode-to-mode triads `S(k\|p\|q)`** | planned (this work) |
| **Helicity flux (3D), enstrophy flux (2D)** | gap |
| **Partial fluxes: Helmholtz rot/div (leverage `HelmholtzDecomposition.jl`), helical ±, tor/pol** | gap |
| **Smooth band-to-band `T(K,Q)`** | gap (belongs in `ShellToShell`) |
| **Anisotropic shells `(k_⊥,k_∥)`** | gap (future) |

---

## 9. References

- Kolmogorov (1941); Kraichnan (1967, 1971); Batchelor (1969) — cascade phenomenology.
- Domaradzki & Rogallo (1990) — shell-to-shell transfer.
- Dar, Verma & Eswaran (2001) — mode-to-mode formalism.
- Verma (2004, Phys. Rep.; 2019 book) — energy transfer review.
- Alexakis, Mininni & Pouquet (2005) — shell-to-shell, large-scale imprint.
- Eyink & Aluie (2009, arXiv:0909.2386) I; Aluie & Eyink (2009) II — smooth vs sharp coarse-graining, band-pass.
- Aluie, Hecht & Vallis (2018, JPO, arXiv:1710.07963) — ocean coarse-graining.
- Alexakis & Biferale (2018, Phys. Rep. 767–769, arXiv:1808.06186) — comprehensive cascade taxonomy.
- Waleffe (1992); Constantin & Majda (1988); Biferale, Musacchio & Toschi (2012) — helical decomposition.
- Yeung, Chu & Schmidt (2026, JFM) — TOD.
- Aluie (2019, doi:10.1007/s13137-019-0123-9) — convolutions on the sphere (Helmholtz filtering commutes with ∇).
- Buzzicotti, Storer, Khatri, Griffies & Aluie (2023, doi:10.1126/sciadv.adi7420) — global KE cascade via Helmholtz filtering.

### Sibling packages leveraged
- `FlowFieldSpectra.jl` — spectral coefficients / energy spectra (Cartesian/spherical, structured/scattered); physical→`û` front-end.
- `CoarseGrainingEnergyFluxes.jl` — coarse-graining flux `Π_ℓ(x)` and filtering spectrum `E(ℓ)`.
- `HelmholtzDecomposition.jl` — rotational/divergent (ψ/χ) decomposition; Cartesian+spherical; multi-solver (SOR/FFTW/FINUFFT/FSH/NUFSHT).
- `NUFSHT.jl` — non-uniform spherical harmonic transforms + filtering/masking (scattered spherical).
- `StructureFunctions.jl` — real-space structure functions (and the ecosystem's backend-architecture template).
