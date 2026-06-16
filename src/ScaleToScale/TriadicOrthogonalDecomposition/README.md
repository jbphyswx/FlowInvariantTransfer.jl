# Triadic Orthogonal Decomposition (TOD)

This directory contains the implementation of the **Triadic Orthogonal Decomposition (TOD)** algorithm as described by Yeung, Chu & Schmidt (JFM 2026). 

TOD is a scale-to-scale energy transfer analysis method that operates in the temporal-frequency domain. It decomposes three-wave (triadic) nonlinear interactions in time-series data and extracts coherent flow structures (convective and recipient modes) that optimally capture spectral momentum transfer.

---

## Mathematical Formulation

### 1. Resonance Condition
We look at triads of frequencies $(f_k, f_l, f_n)$ that satisfy the resonance condition:
$$f_k + f_l = f_n$$

In this interaction, $f_k$ and $f_l$ are the advecting (convective) frequencies, and $f_n$ is the recipient frequency.

### 2. Spectral Estimation (Welch's Method)
The input data $X(t, \text{var}, x)$ is divided into $n_{\text{Blocks}}$ overlapping blocks of length $n_{\text{DFT}}$. For each block $b$ and variable $v$:
1. The mean is subtracted.
2. A window (e.g. Hamming) is applied.
3. A temporal DFT is computed, resulting in $\hat{Q}(f, v, x, b)$.

### 3. Triadic Cross-Correlation & SVD
For each triad $(f_k, f_l, f_n)$:
1. The **recipient** (LHS) data matrix $\hat{Q}_n$ is constructed at frequency $f_n$ across all blocks (size $n_{\text{State}} \times n_{\text{Blocks}}$).
2. The **convective** (RHS nonlinear advection) matrix $\hat{Q}_{kl} = Q(\hat{q}_k, \hat{q}_l)$ is constructed (size $n_{\text{State}} \times n_{\text{Blocks}}$).
3. We define the spatial correlation matrix using a spatial weighting matrix $W$:
   $$R = \hat{Q}_{kl} W \hat{Q}_n^\dagger$$
4. We perform a spatially-weighted low-rank SVD. Mathematically, this corresponds to:
   $$\text{SVD}\left( W^{1/2} \hat{Q}_{kl}, W^{1/2} \hat{Q}_n \right)$$
   which is solved efficiently using a thin QR decomposition followed by the Sirovich (method of snapshots) SVD to handle large spatial dimensions.

The decomposition yields:
- **Mode Bispectrum** $\lambda_m(f_l, f_n)$: Singular values representing the interaction coupling strength for mode $m$.
- **Convective Modes** $\psi_m(x)$: Coherent spatial patterns of the advecting field.
- **Recipient Modes** $\phi_m(x)$: Coherent spatial patterns of the recipient field.
- **Modal Energy Budget** $T_m(f_l, f_n)$: The energy transfer associated with mode $m$:
  $$T_m(f_l, f_n) = \lambda_m \text{Re} \langle \phi_m, W \psi_m \rangle$$

---

## Implementation Details

The implementation is split into a core module and package extensions for optimal performance and modularity:

1. **Core Module (`TriadicOrthogonalDecomposition.jl`):**
   - Handles parameter parsing and frequency triad index building (`frequency_axes`).
   - Implements O(N²) direct-sum fallback for the temporal DFT (runs when `FFTW` is not loaded).
   - Implements the serial SVD loop over all triads.
   - Implements rank-truncated `sirovich_svd` and `lowrank_svd` to ensure correct dimensions when the system rank is less than the number of blocks.

2. **FFTW Extension (`FlowInvariantTransferFFTWExt.jl`):**
   - Overrides `_temporal_block_dft_fft!` using `FFTW.fft` for O(N log N) temporal transforms. This is highly recommended for any production run.

3. **OhMyThreads Extension (`FlowInvariantTransferOhMyThreadsExt.jl`):**
   - Overrides the triad loop using `OhMyThreads.@tasks` to parallelize SVD computations across all CPU cores.

---

## Usage Example

```julia
using FlowInvariantTransfer
using FFTW # activates FFTW extension for fast DFTs

# Data: 512 time snapshots, 1 variable, 64 spatial points
nt, nvar, nx = 512, 1, 64
X = randn(nt, nvar, nx)

# Configure the method
method = TriadicOrthogonalDecompositionMethod(
    nfft = 128,      # block size
    noverlap = 64,   # 50% overlap
    nmode = 2        # keep top 2 modes per triad
)

# Run TOD via the unified interface
result = calculate_energy_transfer(method, X; dt=0.01, backend=FFTBackend())

# Inspect results
result.frequencies         # shifted frequency axes
result.mode_bispectrum     # singular values array (nFreq, nFreq, nmode)
result.modes               # Dict of (l, n) => (convective, recipient) mode arrays
result.modal_energy_budget # energy transfer array (nFreq, nFreq, nmode)
```

---

## References
- Yeung, B., Chu, T., and Schmidt, O. T., *Triadic orthogonal decomposition reveals nonlinearity in fluid flows*, J. Fluid Mech. 1031, A34, 2026. [DOI:10.1017/jfm.2026.11183](https://doi.org/10.1017/jfm.2026.11183)
- Reference Matlab Implementation: [FlowPhysicsGroup/Triadic-Orthogonal-Decomposition](https://github.com/FlowPhysicsGroup/Triadic-Orthogonal-Decomposition)
