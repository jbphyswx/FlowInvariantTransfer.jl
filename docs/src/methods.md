# Methods & Theory

## Spectral Flux Π(K)

The spectral energy flux quantifies how much kinetic energy per unit time crosses a
wavenumber threshold K from large to small scales.

Starting from the spectral energy equation for incompressible flow,

```math
\frac{\partial E(\mathbf{k})}{\partial t} = T(\mathbf{k}) + \text{dissipation} + \text{forcing}
```

the *transfer spectrum* accumulated per isotropic shell ``S_n = \{|\mathbf{k}| \in [\kappa_n, \kappa_{n+1})\}``
is

```math
T(k_n) = \sum_{|\mathbf{k}| \in S_n} \sum_i \text{Re}\bigl[\hat{u}_i^*(\mathbf{k})\,\hat{N}_i(\mathbf{k})\bigr]
```

where ``\hat{N}_i(\mathbf{k}) = \widehat{(u \cdot \nabla) u_i}`` is the nonlinear advection term
and ``\hat{u}_i^*`` its complex conjugate.  The cumulative flux is

```math
\Pi(K_n) = -\sum_{m \le n} T(k_m).
```

Positive Π indicates a **forward (downscale) energy cascade**; negative Π an inverse
(upscale) cascade.

**References:** Verma et al. (2002) [arXiv:nlin/0204027](https://arxiv.org/abs/nlin/0204027)

---

## Shell-to-Shell Transfer T(n, m)

The shell-to-shell transfer matrix resolves *which* donor shell m sends energy to *which*
receiver shell n.  Using the Alexakis et al. (2005) antisymmetrized definition:

```math
T(n,m) = \frac{1}{2}\left[
    \sum_{\mathbf{k} \in S_n} \text{Re}\bigl[\hat{u}^*(\mathbf{k}) \cdot \hat{N}_m(\mathbf{k})\bigr]
  - \sum_{\mathbf{k} \in S_m} \text{Re}\bigl[\hat{u}^*(\mathbf{k}) \cdot \hat{N}_n(\mathbf{k})\bigr]
\right]
```

where ``\hat{N}_m`` is the nonlinear term computed with the velocity **band-filtered to shell m**
as the advecting field.  This construction guarantees exact antisymmetry

```math
T(n,m) + T(m,n) = 0
```

by construction for divergence-free (incompressible) velocity fields.

The net energy gain of shell n is ``\sum_m T(n,m)``.

**References:** Alexakis, Mininni & Pouquet (2005) Phys. Rev. E 72, 046301.

---

## Mode-to-Mode Triad Transfer S(k|p|q)

The mode-to-mode triad transfer is the **most fundamental** scale-to-scale diagnostic.
It gives the energy transferred *to* receiver mode ``\mathbf{k}`` *from* giver ``\mathbf{p}``,
mediated by ``\mathbf{q}``, with the triad closure constraint ``\mathbf{k} = \mathbf{p} + \mathbf{q}``:

```math
S(\mathbf{k}|\mathbf{p}|\mathbf{q}) = -\text{Im}\bigl\{
    [\mathbf{k} \cdot \hat{\mathbf{u}}(\mathbf{q})] \,
    [\hat{\mathbf{u}}^*(\mathbf{k}) \cdot \hat{\mathbf{u}}(\mathbf{p})]
\bigr\}
```

### Properties
- **Giver/receiver antisymmetry:** ``S(\mathbf{k}|\mathbf{p}|\mathbf{q}) = -S(\mathbf{p}|\mathbf{k}|\mathbf{q})``
- **Net transfer:** ``T(\mathbf{k}) = \sum_{\mathbf{p}} S(\mathbf{k}|\mathbf{p}|\mathbf{q}=\mathbf{k}-\mathbf{p})``
- **Conservation:** ``\sum_{\mathbf{k}} T(\mathbf{k}) = 0``

### Reduction Hierarchy

All other diagnostics are reductions of the mode-to-mode tensor:

```
S(k|p|q)   delta in vector k   most fundamental, O(N^{2D})
   │  sum over directions on |k|=K, |p|=Q
   ▼
T(K,Q)     delta in |k|        magnitude-to-magnitude
   │  finite shell width
   ▼
T(n,m)     sharp Fourier shells (shell-to-shell)
   │  sum over givers
   ▼
T(k), Π(K) spectral flux
```

### Computational Cost

The full tensor requires ``O(N^{2D})`` operations — exact but expensive. With a binning
strategy, the code also computes the shell-reduced matrix ``T(K,Q)`` simultaneously.

**References:** Dar, Verma & Eswaran (2001); Verma (2004 review, 2019 book).

---

## Multi-Invariant Generalization

The same nonlinear-term machinery applies to any quadratic inviscid invariant. Only the
**per-mode transfer density** — the inner product weighting — changes. This is controlled
by the `invariant` keyword argument.

| Invariant | Definition | Transfer density ``t(\mathbf{k})`` | Cascade |
|-----------|------------|----------------------------------|---------|
| `KineticEnergy()` | ``E = \tfrac{1}{2}\int |\mathbf{u}|^2`` | ``\sum_c \text{Re}[\hat{u}_c^* \hat{N}_c]`` | Forward (3D), inverse (2D) |
| `Helicity()` | ``H = \int \mathbf{u}\cdot\boldsymbol{\omega}`` (3D) | ``\sum_c \text{Re}[\hat{\omega}_c^* \hat{N}_c]``, ``\hat{\boldsymbol{\omega}} = i\mathbf{k}\times\hat{\mathbf{u}}`` | Forward, co-directional with ``E`` |
| `Enstrophy()` | ``\Omega = \tfrac{1}{2}\int \omega^2`` (2D) | ``\text{Re}[\hat{\omega}^* \hat{N}_\omega]``, scalar vorticity | Forward, counter-directional to ``E`` |

In 2D turbulence, kinetic energy cascades *inversely* (upscale) while enstrophy cascades
*forward* (downscale) — the Kraichnan–Batchelor dual cascade. In 3D, energy and helicity
both cascade forward (co-directional).

**References:** Kraichnan (1967); Alexakis & Biferale (2018), §3.5–3.6.

---

## Partial-Flux Decompositions (Helmholtz)

The total energy flux can be decomposed into contributions from physically distinct
velocity components. The **Helmholtz decomposition** splits the velocity field into
rotational (solenoidal, ``\nabla\times\boldsymbol{\psi}``) and divergent (irrotational,
``\nabla\phi``) parts:

```math
\mathbf{u} = \mathbf{u}_{\text{rot}} + \mathbf{u}_{\text{div}}, \qquad
\nabla\cdot\mathbf{u}_{\text{rot}} = 0, \quad
\nabla\times\mathbf{u}_{\text{div}} = \mathbf{0}
```

The flux then decomposes as ``\Pi = \Pi_{\text{rot}} + \Pi_{\text{div}} + \text{cross terms}``,
enabling separation of rotational (vortical) and divergent (wave/compressible) contributions
to the energy cascade.

This is implemented via the `decomposition` keyword:

- `NoDecomposition()` — full velocity (default)
- `RotationalDecomposition()` — rotational component only
- `DivergentDecomposition()` — divergent component only
- `HelmholtzDecomposition()` — both, returned as a `NamedTuple`

The decomposition is backed by `HelmholtzDecomposition.jl` and supports both physical-space
(Poisson solve) and spectral-space (projection) paths.

!!! note "Helmholtz ≠ helical ≠ toroidal/poloidal"
    Helmholtz separates divergent from rotational flow. The **helical ±** decomposition
    splits the rotational part into curl-eigenmode chiralities. **Toroidal/poloidal**
    splits the solenoidal part by geometry. These are distinct decompositions.

**References:** Aluie (2019); Buzzicotti, Storer, Khatri, Griffies & Aluie (2023).

---

## Nonlinear Term N̂(k)

Both methods above require the nonlinear advection term

```math
\hat{N}_i(\mathbf{k}) = \widehat{(u \cdot \nabla) u_i}(\mathbf{k})
         = \sum_j \widehat{u_j \partial_j u_i}(\mathbf{k})
```

computed pseudospectrally.

**SerialBackend (reference):** Direct O(N²) DFT/IDFT summation.  Exact but slow.

**FFTBackend (production):** O(N log N) via FFTW:
1. ``u_i(x) = \text{IFFT}(\hat{u}_i)``
2. ``\partial_j u_i(x) = \text{IFFT}(i k_j \hat{u}_i)``
3. ``N_i(x) = \sum_j u_j(x)\,\partial_j u_i(x)``
4. ``\hat{N}_i(\mathbf{k}) = \text{FFT}(N_i) \,/\, N_p`` (where ``N_p = \prod_d N_d``)
5. 2/3 dealiasing: zero all modes with ``|k_d| \ge N_d/3`` along any dimension.

---

## Shell Binning

Isotropic shells are defined by monotonically increasing edge vectors ``\kappa_0 < \kappa_1 < \cdots < \kappa_{N_{sh}}``.
Mode ``\mathbf{k}`` belongs to shell n iff ``\kappa_n \le |\mathbf{k}| < \kappa_{n+1}``.

Available binning schemes:

| Type | Description |
|------|-------------|
| `LinearBinning(Δk)` | Uniform shells of width Δk |
| `LogarithmicBinning(k₀, λ)` | Shells with ``\kappa_n = k_0 \lambda^n`` |
| `DyadicBinning(k₀)` | Logarithmic with ``\lambda = 2`` |
| `CustomBinning(edges)` | User-supplied edge vector |

Shell membership is computed once via `assign_shells(k_mag, edges)` — a single integer array
allocation replacing the old `Vector{BitArray}` per-shell masks.

---

## Coarse-Graining Flux Π_ℓ(x)

The Leonard-type coarse-graining flux at filter scale ℓ is

```math
\Pi_\ell(\mathbf{x}) = -\bar{\tau}_{ij}(\mathbf{x})\,\bar{S}_{ij}(\mathbf{x})
```

where ``\bar{\tau}_{ij} = \overline{u_i u_j} - \bar{u}_i \bar{u}_j`` is the subgrid stress
and ``\bar{S}_{ij} = \tfrac{1}{2}(\partial_i \bar{u}_j + \partial_j \bar{u}_i)`` the
filtered strain-rate tensor. The overbar denotes convolution with a kernel at scale ℓ.

This method is implemented via the `CoarseGrainingEnergyFluxes` extension.

**References:** Aluie & Eyink (2009) Phys. Rev. Lett. 103, 174505.

---

## Triadic Orthogonal Decomposition (TOD)

Triadic Orthogonal Decomposition (Yeung, Chu & Schmidt 2026) decomposes three-wave (triadic) nonlinear interactions in the temporal-frequency domain. It uses the method of snapshots to extract coherent modes that optimally capture spectral momentum transfer for a given frequency triad $(f_k, f_l, f_n)$ satisfying the resonance condition:

```math
f_k + f_l = f_n
```

### SVD Formulation

For each valid frequency triad, we construct:
1. The **recipient** (LHS) data matrix ``\hat{Q}_n`` at frequency ``f_n`` across all time blocks (size ``n_{State} \times n_{Blocks}``).
2. The **convective** (RHS nonlinear advection) matrix ``\hat{Q}_{kl} = Q(\hat{q}_k, \hat{q}_l)`` (size ``n_{State} \times n_{Blocks}``).

We perform a spatially-weighted low-rank SVD of the cross-correlation matrix between the convective and recipient fields. The resulting singular values and singular vectors define:
- **Mode Bispectrum** ``\lambda(f_l, f_n, m)``: The singular values quantifying the coupling strength of mode $m$.
- **Convective Modes** ``\psi_m``: Coherent spatial structures of the advecting field.
- **Recipient Modes** ``\phi_m``: Coherent spatial structures of the recipient field.
- **Modal Energy Budget** ``T(f_l, f_n, m)``: The net energy transfer associated with mode $m$, computed as:

```math
T(f_l, f_n, m) = \lambda_m \text{Re}\langle \phi_m, W \psi_m \rangle
```

where ``W`` is the spatial inner-product weighting matrix.

**References:** Yeung, Chu & Schmidt (2026) — *Triadic orthogonal decomposition reveals nonlinearity in fluid flows* J. Fluid Mech. 1031, A34. [DOI: 10.1017/jfm.2026.11183](https://doi.org/10.1017/jfm.2026.11183)
