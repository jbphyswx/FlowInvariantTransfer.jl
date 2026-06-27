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

## Mode-to-Mode Triad Transfer S(k|p)

The mode-to-mode transfer is the **finest** scale-to-scale object: the rate at which the invariant
is delivered *to* receiver mode ``\mathbf{k}`` *from* giver mode ``\mathbf{p}`` (mediated by
``\mathbf{q}=\mathbf{k}-\mathbf{p}``). It is built directly from the validated pseudospectral
nonlinear term — for each giver mode ``\mathbf{p}``, ``\hat{N}_{\mathbf p}=\widehat{(u\cdot\nabla)u_{\mathbf p}}``
(the full velocity advecting the single-mode field ``u_{\mathbf p}``), and

```math
S(\mathbf{k}|\mathbf{p}) = \text{Re}\bigl\{ \hat{\mathbf{u}}^*(\mathbf{k}) \cdot \hat{N}_{\mathbf p}(\mathbf{k}) \bigr\}
```

(generalized per invariant via `transfer_density`). This construction is exact and inherits the
correct structure (all verified by tests).

### Properties
- **Reduces:** ``\sum_{\mathbf p} S(\mathbf{k}|\mathbf{p}) = T(\mathbf{k})`` — the spectral transfer
- **Antisymmetric:** ``S(\mathbf{k}|\mathbf{p}) = -S(\mathbf{p}|\mathbf{k})`` (incompressible)
- **Conserves:** ``\sum_{\mathbf{k}}\sum_{\mathbf p} S(\mathbf{k}|\mathbf{p}) = 0``

The result carries `net_transfer` (``T(k)``) and `transfer` (the resolved ``S(k|p)``, shape
`(ns..., ns...)`). Summed over shells it gives the shell-to-shell matrix ``T(n,m)``.

### Reduction Hierarchy

```
S(k|p)     resolved triads      finest, O(N^{2D})
   │  sum over shells (k,p)
   ▼
T(n,m)     sharp Fourier shells (shell-to-shell)
   │  sum over givers
   ▼
T(k), Π(K) spectral flux
```

### Computational Cost

Resolving every receiver/giver pair is ``O(N_\text{modes})`` nonlinear-term evaluations —
``O(N_\text{modes}\,N^D\log N)`` with `FFTBackend` and an ``O(N_\text{modes}^2)`` result tensor; a
mode-count guard errors above `max_modes` (pass `force=true` to override). For the aggregates prefer
the cheaper `calculate_spectral_flux` / `calculate_shell_to_shell_transfer`.

**References:** Dar, Verma & Eswaran (2001); Verma (2004 review, 2019 book); Alexakis & Biferale (2018).

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
| `PassiveScalar()` | ``E_\theta = \tfrac{1}{2}\int \theta^2`` | ``\text{Re}[\hat{\theta}^* \hat{N}_\theta]``, ``\hat{N}_\theta=\widehat{(u\cdot\nabla)\theta}`` | Forward (any D); conserved |

In 2D turbulence, kinetic energy cascades *inversely* (upscale) while enstrophy cascades
*forward* (downscale) — the Kraichnan–Batchelor dual cascade. In 3D, energy and helicity
both cascade forward (co-directional).

A **passive scalar** ``\theta`` advected by the velocity is handled by the same engine (it is the
*advected* field, the velocity merely advects it); the convenience wrappers `calculate_scalar_flux`,
`calculate_scalar_shell_to_shell_transfer`, and `calculate_scalar_mode_to_mode_transfer` set
`invariant=PassiveScalar()` and `advecting_hat=velocity`. The identical path computes the buoyancy /
available-potential-energy variance and the QG potential enstrophy — pass that field as the scalar.

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
    splits the solenoidal part by geometry. These are distinct decompositions, all available as
    `decomposition=` (`HelmholtzDecomposition`, `HelicalDecomposition`, `ToroidalPoloidalDecomposition`).

### Per-component partial fluxes

`calculate_partial_fluxes(...; decomposition=...)` resolves the flux by the component of *each* of
the three fields in a triad: with ``u=\sum_s u_s``,
``T^{s_k s_p s_q}(k)=\text{Re}\{\hat{u}_{s_k}^*\cdot\widehat{(u_{s_p}\cdot\nabla)u_{s_q}}\}``, giving
``n^3`` channels that sum to the full flux. With `HelicalDecomposition` these are the **eight
helical channels** — the homochiral ones (``s_k=s_p=s_q``) drive the inverse cascade, the
heterochiral the forward (Waleffe 1992; Biferale, Musacchio & Toschi 2012; Alexakis 2017);
`calculate_helical_partial_fluxes` is the shortcut. With `HelmholtzDecomposition` the off-diagonal
channels are the **rotational↔divergent cross-flux** (zero for incompressible flow).

**References:** Aluie (2019); Buzzicotti, Storer, Khatri, Griffies & Aluie (2023).

---

## Anisotropic geometry, smooth bands, and dealiasing

**Shell geometry** (`geometry=`) sets which wavenumber coordinate the shells partition: isotropic
``|k|`` ([`IsotropicShells`](@ref)), or the anisotropic ``k_\perp`` / ``k_\parallel``
([`PerpendicularShells`](@ref) / [`ParallelShells`](@ref)) that give the directional fluxes
``\Pi(k_\perp)``, ``\Pi(k_\parallel)`` for rotating/stratified flows (Alexakis & Biferale 2018, §IV).

**Smooth band-to-band** ``T(K,Q)`` ([`calculate_band_to_band_transfer`](@ref), [`SmoothBands`](@ref))
replaces the sharp shell indicator with a graded log-Gaussian partition of unity (Eyink & Aluie
2009); it is antisymmetric, conserves, and reduces to the band-summed transfer spectrum.

**Dealiasing** (`dealiasing=`) of the quadratic product: [`OrszagTwoThirds`](@ref) (default; exact
on ``|k|<N/3``), [`NoDealiasing`](@ref), or [`PaddedThreeHalves`](@ref) — exact 3/2 zero-padding
that is alias-free over every retained mode to Nyquist (FFT path).

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
