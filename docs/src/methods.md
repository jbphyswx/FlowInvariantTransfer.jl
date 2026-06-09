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
