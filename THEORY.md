# FlowInvariantTransfer.jl — Theory & Knowledge Base

> Living theoretical record for the package: definitions, relationships, what we implement, and how it maps to related packages.

## 0. Scope

`FlowInvariantTransfer.jl` computes **cross-scale transfer of quadratic inviscid invariants** (primarily kinetic energy) in turbulent flows, in Fourier space and via coarse-graining. Energy *spectra* `E(k)` live in the sibling package `FlowFieldSpectra.jl`; pointwise coarse-graining flux machinery lives in `CoarseGrainingEnergyFluxes.jl` (CGEF) and is wrapped here via an extension.

---

## 0.5 Conventions (verified against primary sources — WV)

> Ground-truth spec for the overhaul. Each row is transcribed from the *original* paper (equation
> numbers cited) and is what unit tests assert against. Where two conventions coexist in the
> literature, we record both and pick one (marked **[CHOSEN]**).

### Energy transfer & flux sign — [CHOSEN: Alexakis–Biferale 2018]
Verified from arXiv:1808.06186 (HTML):
- Spectral balance **Eq. (12):** `∂_t E(k) = −T(k) − 2νk²E(k) − 2αE(k) + F(k)` — `T(k)` is the net
  **loss** from shell k (positive `T` = energy removed).
- Transfer **Eq. (13):** `T(k) = ℑ Σ_{shell} Σ_{p+q=k} û*_i(k) P_ij(k) q_l û_l(p) û_j(q)`,
  Leray projector `P_ij(k)=δ_ij − k_i k_j/k²`.
- Spectrum **Eq. (11):** `E(k) = (1/2Δk) Σ_{k≤|k'|<k+Δk} |û(k')|²` (explicit ½).
- Flux **Eq. (17):** `Π(K) = +Σ_{k<K} T(k) = ⟨u^{<K}·(u·∇)u⟩`. Inviscid: `Π(∞)=0`.
- **Sign anchor:** `Π>0` forward/direct (= `ε` in the forward inertial range); `Π<0` inverse (= `−ε_α`).

Equivalence used by the FFT path (derived + verified): the package's `transfer_density = Re{û*·N̂}`
with `N̂=FFT[(u·∇)u]` equals AB's `T(k)` (for incompressible `û`, `û*·P·N̂ = û*·N̂`). Therefore the
correct flux is **`Π = +cumsum(T)`**.
> ⚠ **BUG to fix in W6:** [SpectralFlux.jl:162-163](src/SpectralFlux.jl) computes `flux = −cumsum(T)`,
> returning `−Π_AB` — inverted vs the `Π>0 ⇒ forward` anchor. Drop the negation. The
> `transfer_spectrum` sign (= AB `T`, loss) is already correct.

### Mode-to-mode triad transfer — [CHOSEN: Dar–Verma–Eswaran 2001]
Verified from arXiv:nlin/0109004 (HTML):
- **Eq. (11):** `S^{uu}(k|p|q) = −ℑ([k·u(q)][u(k)·u(p)])`; arguments are **k = receiver, p = giver,
  q = mediator**.
- Triad convention in DVE is **`k+p+q=0`**. The package uses the equivalent **`k=p+q`** (so `q=k−p`);
  both are valid (real fields: `û(−q)=û*(q)`). *Document both; the package's `q=k−p` form is correct
  and sums to the FFT net transfer — verified analytically: `Σ_p S(k|p|k−p) = Re{û*·N̂} = T(k)`.*
- Antisymmetry **Eq. (A4):** `R^{uu}(k|p|q) + R^{uu}(p|k|q) = 0`.
- Gauge: full transfer `R = S + X_Δ`; `X_Δ` is the **circulating** term (q→k→p→q, changes no mode's
  energy). Only `S`-sums (combined, shell-to-shell, flux) are unique (cf. Plunian–Stepanov–Verma 2020).
- Flux **Eq. (17):** `Π(K) = Σ_{|k|>K} Σ_{|p|<K} S(k|p|q)`.

### Helical ± decomposition — [CHOSEN: Alexakis 2017 √2-unit-norm]
Verified from arXiv:1606.02540 (HTML):
- Basis **Eq. (10):** `h^s_k = (e_z×k)/(√2|e_z×k|) + i s (k×(e_z×k))/(√2|k×(e_z×k)|)`, with
  `h^s·h^s = 0`, `h^s·h^{-s} = 1`, and `i k×h^s_k = s|k| h^s_k`.
- Decomposition **Eq. (9):** `û_k = u^+_k h^+_k + u^-_k h^-_k`.
- **Eq. (13):** `E^± = ½Σ|u^±|²`, `H^± = ±½Σ|k||u^±|²` (so `E=E^++E^-`, `H=H^++H^-`).
- Partial flux **Eq. (17):** `Π_E^{s1,s2,s3}(k) = −⟨u^{s1<}_k·(u^{s2}×w^{s3})⟩_T`, `w=∇×u`.
- Homochiral `Π^{+++}+Π^{---}` is constant & **negative** (hidden inverse transfer inside the 3D
  forward cascade); heterochiral channels are net **forward**.

### Anisotropic / directional shells — [CHOSEN: Alexakis–Marino–Mininni 2025]
Verified from arXiv:2508.00340 (HTML):
- Shell sets **Eq. (4):** spherical `k≤|q|<k+k_L`; cylindrical/axisymmetric `k_⊥≤|q_⊥|<k_⊥+k_L`;
  plane-averaged `k_∥≤|q_∥|<k_∥+k_L`; 2D = both bounds.
- Spectrum normalization `1/(2k_L)`, `k_L=1/L` the bin width (one-mode-one-bin).
- Directional flux **Eq. (9):** `Π_K(k)=⟨u_{S_k}·(u·∇)u⟩`, `Π_P(k)=⟨φ_{S_k}(u·∇)φ⟩`, `Π_T=Π_K+Π_P`.
- Sign: `Π<0` inverse, `Π>0` forward (strong-rotation inverse ⊥-cascade saturates near `−1`).

### Coarse-graining flux & smooth band-to-band — [CHOSEN: Eyink–Aluie 2009]
Verified from arXiv:0909.2386 (HTML):
- Filter **Eq. (1):** `ū_ℓ(x)=∫dr G_ℓ(r) u(x+r)`, `G_ℓ(r)=ℓ^{−d}G(r/ℓ)` (`∫G=1`).
- SGS stress **Eq. (3):** `τ̄_ℓ(u,u)=(uu)‾_ℓ − ū_ℓ ū_ℓ` (full Germano stress — Galilean-invariant).
- Space-local flux **Eq. (5):** `Π̄_ℓ = −(∂_j ū_i) τ̄_ij = −S̄_ij τ̄_ij` (τ symmetric),
  `S̄_ij=½(∂_iū_j+∂_jū_i)`. `Π̄_ℓ>0` forward (sink in the resolved-KE budget Eq. 4).
- Band energy **Eq. (8):** `½τ̃(ū;ū)`; inter-band transfer `T_n = Π_{n−1} − Π_n` (net flux
  difference at band boundaries) — the smooth-kernel generalization of shell-to-shell.

### MHD invariants & gauge — [CHOSEN: Plunian–Stepanov–Verma 2020]
Verified from arXiv:2004.10107 (HTML):
- KE mode-to-mode is **gauge-dependent** — **Eq. (53):**
  `ΔE^u(k|p|q)=α^u_E Re{(u_k,u_p,ω_q)+(u_k,ω_p,u_q)+(ω_k,u_p,u_q)}`, `α` arbitrary (DVE = 0).
  → only combined/shell/flux sums are physical (matches the helicity-gauge note above).
- **Eq. (16):** `E^u_k=½u_k·u*_k`, `E^b_k=½b_k·b*_k`; **Eq. (28):** magnetic helicity `H^b_k=½b_k·a*_k`.
- Magnetic-field energy transfers become **uniquely defined** when split into magnetic *advection*
  vs *stretching* (unlike KE); magnetic-helicity transfer (Eq. 34) is unique with no free coefficient.
- *Cascade directions* + cross-helicity `H_c=∫u·b` + 2D mean-square vector potential `∫½a²` not
  defined in this paper — still to confirm against Pouquet–Frisch–Léorat 1976 / Fyfe–Montgomery 1976
  (working values from survey: 3D total energy & cross-helicity forward; magnetic helicity inverse;
  2D MHD energy forward; 2D `∫½a²` inverse).

### Triadic Orthogonal Decomposition — [CHOSEN: Yeung–Chu–Schmidt]
Verified from arXiv:2411.12057 (HTML; re-confirm vs final JFM 1031:A34):
- Convective term **Eq. (6):** `ĉ_{l→n} = −(û_{n−l}·∇)û_l` (explicit minus; donor `û_l`, catalyst
  `û_{n−l}`, recipient `û_n`).
- Covariance kernel **Eq. (9):** `S(x,x';l,n)=E{ĉ_{l→n}(x) û_n^H(x')}`; SVE **Eq. (10):**
  `S=Σ_j σ_j ψ̂_{l→n,j}(x) φ̂_{n,j}^H(x')`.
- Modal energy budget **Eq. (33):** `T̂_{l→n}=−∫_Ω û_n^H (û_{n−l}·∇)û_l dx`; Re part = energy flow;
  pairwise conservation `T̂^R_{l→n}+T̂^R_{n→l}=0`.
- Welch blocks (Eqs. 14–17), `S_{l,n}=(1/N_blk)Ĉ_{l→n}Û_n^H` (Eq. 17), W-weighted SVD with
  `Φ̂_n^H W Φ̂_n = I` (Eq. 18). Triad: `f_{n−l}+f_l=f_n`.

### Compressible mode-to-mode — [CHOSEN: Singh–Tiwari–Sharma–Verma 2025]
Verified from arXiv:2508.04300 (HTML):
- KE (Framework A, **Eqs. 14,15,19**): `v=ρu`, `E_u(k)=½Re[v(k)·u*(k)]`. Framework B
  (**App. B, Eqs. 91,98**): `w=√ρ u`, `E_u=½|w|²`. **[CHOSEN: A for transfer; B available for 4/5-law.]**
- Transfer **Eq. (28):** `S^{uu}(a|b|c) = −½ Im[{a·u(c)}{v(b)·u(a)} − {b·u(c)}{u(b)·v(a)}]`,
  triad `a+b+c=0`; antisymmetric **Eq. (30):** `S^{uu}(a|b|c)=−S^{uu}(b|a|c)`.
- Helmholtz **Eq. (32):** `u=u_R+u_C` (`u_C∥k` compressive, `u_R⊥k` rotational); flux channels
  `Π_R` (Eq. 52), `Π_C` (Eq. 53), cross `Π^{R<}_{C>}` (Eq. 56), `Π^{C<}_{R>}` (Eq. 57).
- KE↔IE pressure-dilatation **Eqs. (38–39):** `Q_{I,R}=½Re[σ̃·v_R*]`,
  `Q_{I,C}=½Re[σ̃·v_C*] − ½Im[σ{k·u_C*}]`, `σ̃=∇σ/ρ`.

### Standard / non-contested conventions (canonical refs; no ambiguity to resolve)
- **Kinetic energy cascade:** `E(k)=C_K ε^{2/3}k^{−5/3}`, `Π=ε` forward (Kolmogorov 1941).
- **Passive scalar variance** `∫E_θ dk=½⟨θ²⟩`, `T_θ` from `−Re{θ̂*·FFT[(u·∇)θ]}`, forward in 2D & 3D;
  `E_θ=C_θ ε_θ ε^{−1/3}k^{−5/3}` (Obukhov 1949; Corrsin 1951); `k^{−1}` viscous-convective for Sc≫1
  (Batchelor 1959).
- **2D dual cascade:** `Z=½⟨ω²⟩`, `Z(k)=k²E(k)`; inverse energy (`Π_E<0`, `k^{−5/3}`) + forward
  enstrophy (`Π_Z>0`, `k^{−3}`) (Kraichnan 1967; Batchelor 1969; Leith 1968).
- **Buoyancy/APE:** `APE=½⟨b²⟩/N²`, `b=gαθ'`; reversible KE↔APE conversion `B=⟨wb⟩`; total energy
  forward in strong stratification (Lindborg 2006, JFM 550).
- **Compressible coarse-graining (Favre):** `ũ=(ρu)‾/ρ̄`; flux = deformation-work + pressure-dilatation
  + baropycnal (Aluie 2011 PRL 106; 2013 Physica D 247).
- **QG:** potential enstrophy `½q²` forward / energy inverse (Charney 1971).

### WV status
All contested/high-risk conventions locked against primary sources (HTML). Remaining open items are
narrow sign confirmations (MHD cross-helicity / 2D vector-potential cascade direction vs
Pouquet–Frisch–Léorat 1976 & Fyfe–Montgomery 1976) — to confirm when MHD invariants are implemented
in W4. Every locked formula gets a numerical unit test in W14.

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
