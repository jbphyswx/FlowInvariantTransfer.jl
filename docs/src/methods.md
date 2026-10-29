```@meta
CurrentModule = FlowInvariantTransfer
```

# Methods & Theory

## Spectral Flux Π(K)

The spectral energy flux quantifies how much kinetic energy per unit time crosses a
wavenumber threshold K from large to small scales.

Starting from the spectral energy equation for incompressible flow,

```math
\frac{\partial E(\mathbf{k})}{\partial t} = -T(\mathbf{k}) + \text{dissipation} + \text{forcing}
```

the *transfer spectrum* of each isotropic shell ``S_n = \{|\mathbf{k}| \in [\kappa_n, \kappa_{n+1})\}``,
``T(k_n) = -dE(k_n)/dt`` from the nonlinear term (positive when shell ``n`` gives energy to the
others), is

```math
T(k_n) = \sum_{|\mathbf{k}| \in S_n} \sum_i \text{Re}\bigl[\hat{u}_i^*(\mathbf{k})\,\hat{N}_i(\mathbf{k})\bigr]
```

where ``\hat{N}_i(\mathbf{k}) = \widehat{(u \cdot \nabla) u_i}`` is the nonlinear advection term
and ``\hat{u}_i^*`` its complex conjugate (Alexakis & Biferale 2018, Eq. 12). The flux through
``K_{n+1}``, the energy the shells up to ``n`` pass to larger wavenumbers per unit time, is

```math
\Pi(K_{n+1}) = \sum_{m \le n} T(k_m).
```

Positive Π indicates a **forward (downscale) energy cascade**; negative Π an inverse
(upscale) cascade. Every transfer in the package is ``-dE/dt`` of the shell, degree or mode it is
indexed by, and every flux is its running sum, on Cartesian, spherical and compressible fields alike.

**References:** Verma et al. (2002) [arXiv:nlin/0204027](https://arxiv.org/abs/nlin/0204027);
Alexakis & Biferale (2018), Phys. Rep. 767–769, 1–101.

---

## Shell-to-Shell Transfer T(n, m)

The shell-to-shell transfer matrix resolves the rate at which each shell n gives energy to each
shell m:

```math
T(n,m) = \sum_{\mathbf{k} \in S_n} \text{Re}\bigl[\hat{\mathbf{u}}^*(\mathbf{k}) \cdot \hat{N}_m(\mathbf{k})\bigr],
\qquad \hat{N}_m = \widehat{(\mathbf{u} \cdot \nabla)\,\mathbf{u}_m},
```

where ``\mathbf{u}_m`` is the velocity **band-filtered to shell m** and the full velocity advects it
(Verma et al. 2002; Alexakis, Mininni & Pouquet 2005). For a divergence-free velocity
``\int \mathbf{u}_n \cdot (\mathbf{u}\cdot\nabla)\mathbf{u}_m = -\int \mathbf{u}_m \cdot (\mathbf{u}\cdot\nabla)\mathbf{u}_n``,
so

```math
T(n,m) + T(m,n) = 0.
```

A positive ``T(n,m)`` is energy leaving shell n for shell m, and ``\sum_m T(n,m) = T(k_n)``.

**References:** Alexakis, Mininni & Pouquet (2005) Phys. Rev. E 72, 046301.

---

## Mode-to-Mode Triad Transfer S(k|p)

The mode-to-mode transfer is the **finest** scale-to-scale object, indexed by the stored coefficients
("slots") of the spectral layout. Slot ``\mathbf{p}`` stands for ``w(\mathbf{p})`` modes of the full
spectrum (the Hermitian weight: 2 inside the halved axis of a real field's half layout, where the slot
also stands for its mirror ``-\mathbf{p}``, and 1 elsewhere), and its slot field
``u_{\mathbf p}(x) = w(\mathbf{p})\,\text{Re}\{\hat{\mathbf u}(\mathbf p)\,e^{i\mathbf p\cdot x}\}`` is real;
the slot fields sum to ``u``. With ``\hat{N}_{\mathbf p}=\widehat{(u\cdot\nabla)u_{\mathbf p}}`` (the full
velocity advecting the slot field),

```math
S(\mathbf{k}|\mathbf{p}) = \text{Re}\bigl\{ \hat{\mathbf{u}}^*(\mathbf{k}) \cdot \hat{N}_{\mathbf p}(\mathbf{k}) \bigr\}
```

(generalized per invariant via `transfer_density`), and ``w(\mathbf{k})S(\mathbf{k}|\mathbf{p})`` is the
rate at which slot field ``u_{\mathbf k}`` gives energy to slot field ``u_{\mathbf p}``, mediated by
``\mathbf{q}=\mathbf{k}-\mathbf{p}``.

### Properties
- **Reduces:** ``\sum_{\mathbf p} S(\mathbf{k}|\mathbf{p}) = T(\mathbf{k})`` — the spectral transfer
- **Antisymmetric** (divergence-free, dealiased):
  ``w(\mathbf{k})S(\mathbf{k}|\mathbf{p}) = -w(\mathbf{p})S(\mathbf{p}|\mathbf{k})``, which reads
  ``S(\mathbf{k}|\mathbf{p}) = -S(\mathbf{p}|\mathbf{k})`` on a full layout, where ``w \equiv 1``
- **Conserves:** ``\sum_{\mathbf{k}}\sum_{\mathbf p} w(\mathbf{k})S(\mathbf{k}|\mathbf{p}) = 0``

The result carries `weights` (``w``), `net_transfer` (``T(k)``) and `transfer` (the resolved
``S(k|p)``, shape `(ms..., ms...)`). Summed over the slots of two shells, ``w(k)S(k|p)`` gives the
shell-to-shell matrix ``T(n,m)``.

### Reduction Hierarchy

```
S(k|p)     resolved triads      finest, O(N^{2D})
   │  sum over shells (k,p)
   ▼
T(n,m)     sharp Fourier shells (shell-to-shell)
   │  sum over m
   ▼
T(k), Π(K) spectral flux
```

### Computational Cost

One nonlinear-term evaluation per slot ``\mathbf{p}``: ``O(N_\text{modes})`` evaluations,
``O(N_\text{modes}\,N^D\log N)`` with `FFTSpectralBackend`, and an ``O(N_\text{modes}^2)`` result
tensor. The aggregates come cheaper from `calculate_spectral_flux` /
`calculate_shell_to_shell_transfer`.

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
``n^3`` channels that sum to the full flux. With `HelicalDecomposition` the channel term takes the
rotational form of Alexakis (2017, Eq. 17),
``\Pi^{s_1s_2s_3}(K) = -\langle u^{s_1}_{K<}\cdot(u^{s_2}\times\omega^{s_3})\rangle``, i.e.
``T^{s_k s_p s_q}(k)=\text{Re}\{\hat{u}_{s_k}^*\cdot\widehat{\omega_{s_q}\times u_{s_p}}\}``, and the
**eight helical channels** sum to the flux of the solenoidal part of ``u``. Pair sums over the last two
labels equal those of the advective form; single heterochiral channels of the two forms differ. In
Alexakis's simulations the homochiral sum ``\Pi^{+++}+\Pi^{---}`` is negative (an inverse flux) and the
mixed-sign channels carry the forward flux (Waleffe 1992; Biferale, Musacchio & Toschi 2012; Alexakis
2017); `calculate_helical_partial_fluxes` is the shortcut. With `HelmholtzDecomposition` the
off-diagonal channels are the **rotational↔divergent cross-flux** (zero for incompressible flow).

**References:** Aluie (2019); Buzzicotti, Storer, Khatri, Griffies & Aluie (2023).

---

## Anisotropic geometry, smooth bands, and dealiasing

**Shell geometry** (`geometry=`) sets which wavenumber coordinate the shells partition: isotropic
``|k|`` ([`IsotropicShells`](@ref FlowInvariantTransfer.Types.IsotropicShells)), or the anisotropic ``k_\perp`` / ``k_\parallel``
([`PerpendicularShells`](@ref FlowInvariantTransfer.Types.PerpendicularShells) / [`ParallelShells`](@ref FlowInvariantTransfer.Types.ParallelShells)) that give the directional fluxes
``\Pi(k_\perp)``, ``\Pi(k_\parallel)`` for rotating/stratified flows (Alexakis & Biferale 2018, §IV).

**Smooth band-to-band** ``T(K,Q)`` ([`calculate_band_to_band_transfer`](@ref), [`SmoothBands`](@ref FlowInvariantTransfer.Types.SmoothBands))
replaces the sharp shell indicator with a graded log-Gaussian partition of unity (Eyink & Aluie
2009); it is antisymmetric, conserves, and reduces to the band-summed transfer spectrum.

**Dealiasing** (`dealiasing=`) of the quadratic product: [`OrszagTwoThirds`](@ref FlowInvariantTransfer.Types.OrszagTwoThirds) (default; keeps
``|m_d| \le \lfloor (N_d-1)/3 \rfloor``, exact there), [`NoDealiasing`](@ref FlowInvariantTransfer.Types.NoDealiasing), or [`PaddedThreeHalves`](@ref FlowInvariantTransfer.Types.PaddedThreeHalves) — exact 3/2 zero-padding
that is alias-free over every retained mode to Nyquist, on the FFT and the direct-sum transforms.

---

## Nonlinear Term N̂(k)

Both methods above require the nonlinear advection term

```math
\hat{N}_i(\mathbf{k}) = \widehat{(u \cdot \nabla) u_i}(\mathbf{k})
         = \sum_j \widehat{u_j \partial_j u_i}(\mathbf{k})
```

computed pseudospectrally.

**DirectSumSpectralBackend (reference):** direct `O(N²ᴰ)` DFT summation. Exact but slow.

**FFTSpectralBackend (production):** `O(Nᴰ log N)` via FFTW, with the unnormalized inverse
(``\hat{u}`` carries the ``1/N_p``):
1. ``u_i(x) = \sum_{\mathbf{k}} \hat{u}_i(\mathbf{k})\, e^{i\mathbf{k}\cdot x}``
2. ``\partial_j u_i(x) = \sum_{\mathbf{k}} i k_j \hat{u}_i(\mathbf{k})\, e^{i\mathbf{k}\cdot x}``
3. ``N_i(x) = \sum_j u_j(x)\,\partial_j u_i(x)``
4. ``\hat{N}_i(\mathbf{k}) = \text{FFT}(N_i) \,/\, N_p`` (where ``N_p = \prod_d N_d``)
5. 2/3 dealiasing: zero all modes with ``|m_d| > \lfloor (N_d-1)/3 \rfloor`` along any dimension, in
   the inputs and the output — the largest band whose quadratic products alias outside it.

---

## Shell Binning

Isotropic shells are defined by monotonically increasing edge vectors ``\kappa_0 < \kappa_1 < \cdots < \kappa_{N_{sh}}``.
Mode ``\mathbf{k}`` belongs to shell n iff ``\kappa_n \le |\mathbf{k}| < \kappa_{n+1}``.

Available binning schemes:

| Type | Description |
|------|-------------|
| `LinearBinning(Δk)` | Uniform shells of width Δk |
| `LogarithmicBinning(k₀, λ)` | Edges ``0, k_0, k_0\lambda, k_0\lambda^2, \ldots`` (first shell ``[0, k_0)``) |
| `DyadicBinning(k₀)` | Logarithmic with ``\lambda = 2`` |
| `CustomBinning(edges)` | User-supplied edge vector |

Shell membership is computed once via `assign_shells(k_mag, edges)`, one integer shell index per mode
(the last shell includes its upper edge).

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

## Spectral layout: the half spectrum of a real field

A real field's transform is Hermitian, ``\hat{u}(-k) = \overline{\hat{u}(k)}``, so half its coefficients
are redundant. `to_spectral` therefore returns the **half layout** for real input: axis 1 carries
``k_1 \ge 0`` on an `rfftfreq` axis of length ``n_1/2 + 1``, the remaining axes carry the full
`fftfreq` range, and the conjugate half is implied. Complex input gives the
full `fftfreq` grid on every axis. Which layout an array is in is carried by `ks`, so every diagnostic
reads it from there.

### The Hermitian weight

Every transfer density in this package is even under ``k \mapsto -k``: ``|\hat{u}|^2`` and
``\mathrm{Re}\{\overline{\hat{u}} \hat{N}\}`` both are, because ``\hat{u}`` and ``\hat{N}`` are
transforms of real fields. A full-spectrum sum of such an ``f`` equals a weighted half-spectrum sum,

```math
\sum_{k \in \text{full}} f(k) \;=\; \sum_{k \in \text{half}} w(k)\, f(k),
```

with a weight ``w(k) \in \{1, 2\}``. The ``k_1 = 0`` hyperplane is stored whole and is mapped
onto itself by ``k \mapsto -k``, so each of its modes already has its mirror in the array: ``w = 1``.
For even ``n_1`` the Nyquist plane ``k_1 = n_1/2 \equiv -n_1/2`` is the same situation: ``w = 1``.
Every other stored mode stands for itself and its unstored mirror: ``w = 2``. On the full layout
``w \equiv 1``.

### The Nyquist mode of an even axis

That self-mirrored Nyquist slot holds ``+n/2`` and ``-n/2`` at once, so any operator **first order in
``k``** is not single-valued there for a real field, and it carries zero: the gradient ``\partial_d``,
the vorticity ``i k \times \hat{u}`` behind the helicity and enstrophy densities, and the Helmholtz
projector. The ``3/2``-padded dealiasing instead
**splits** that coefficient across ``\pm n/2`` on the finer grid — the unique real band-limited
function it represents is ``\hat{u}_{\text{Nyq}} \cos(n x / 2)`` — and recombines the pair on the way
back. The ``2/3`` rule discards the mode entirely.

### Which decompositions stay real

A half layout is only available to a field that is genuinely real. Under ``k \mapsto -k`` the helical
frame obeys ``e_1 \mapsto -e_1`` and ``e_2 \mapsto e_2``, so ``u_\pm(-k) = \overline{u_\pm(k)}`` and the
helical components of a real field are individually Hermitian; the same holds for the toroidal/poloidal
and Helmholtz (rotational/divergent) components, whose projectors are real and commute with
conjugation. Every decomposition this package provides therefore runs on the half layout.
