using Test: Test
using Random: Random
using Statistics: Statistics
using Aqua: Aqua
using FFTW: FFTW
using LinearAlgebra: LinearAlgebra
using Distributed: Distributed
using SharedArrays: SharedArrays
using OhMyThreads: OhMyThreads
using KernelAbstractions: KernelAbstractions as KA
using MPI: MPI
using CoarseGrainingEnergyFluxes: CoarseGrainingEnergyFluxes
using HelmholtzDecomposition: HelmholtzDecomposition
using CairoMakie: CairoMakie
using FINUFFT: FINUFFT
using NonuniformFFTs: NonuniformFFTs
using FlowFieldSpectra: FlowFieldSpectra
using FastSphericalHarmonics: FastSphericalHarmonics as FSH
using NUFSHT: NUFSHT

using FlowInvariantTransfer: FlowInvariantTransfer as FIT
using ComputationalBackends: ComputationalBackends
using SpectralBackends: SpectralBackends
using JLArrays: JLArrays   # reference GPU-array backend: allowscalar(false) → catches non-device-generic code
using CUDA: CUDA           # triggers FlowInvariantTransferFINUFFTCUDAExt (precompile/load guard; device transform runs only on NVIDIA hardware)

# -----------------------------------------------------------------------
Test.@testset "Utils — wavenumber_grid" begin
    L = 2π
    dk = 2π / L
    # Gate the WHOLE grid against FFTW's fftfreq (an independent reference) for BOTH parities. The
    # even-N Nyquist mode is -N/2, not +N/2; the old test only checked indices 1, 2, N and silently
    # skipped the Nyquist, letting a sign bug there break every signed-k derivative undetected.
    for N in (8, 9)
        ks = FIT.Utils.wavenumber_grid((N, N), (L, L))
        Test.@test length(ks) == 2
        ref = collect(FFTW.fftfreq(N, N)) .* dk
        Test.@test isapprox(collect(ks[1]), ref; atol = 1e-14)
        Test.@test isapprox(collect(ks[2]), ref; atol = 1e-14)
    end
    Test.@test isapprox(FIT.Utils.wavenumber_grid((8,), (L,))[1][5], -4 * dk; atol = 1e-14)  # Nyquist = -N/2

    k_mag = FIT.Utils.wavenumber_magnitude_grid(FIT.Utils.wavenumber_grid((8, 8), (L, L)))
    Test.@test size(k_mag) == (8, 8)
    Test.@test all(k_mag .>= 0)
    Test.@test isapprox(k_mag[1, 1], 0.0)
end

# -----------------------------------------------------------------------
Test.@testset "Utils — dealiasing_mask" begin
    N = 12
    mask = FIT.Utils.dealiasing_mask((N, N))
    Test.@test size(mask) == (N, N)
    # All modes with |k_d| >= N/3 = 4 along any dim should be zeroed
    # k_idx=0:3 kept, 4:8 removed (FFTW order: 0..N/2 then -(N/2-1)..-1)
    Test.@test mask[1, 1]   # (k=0,k=0) kept
    Test.@test mask[2, 1]   # (k=1,k=0) kept
    Test.@test !mask[5, 1]  # (k=4,k=0) removed by 2/3 rule (4 >= 12/3=4)
end

# -----------------------------------------------------------------------
Test.@testset "ShellBinning — LinearBinning" begin
    b = FIT.Types.LinearBinning(1.0)
    edges = FIT.ShellBinning.shell_edges(b, 5.0)
    Test.@test edges[1] == 0.0
    Test.@test edges[end] >= 5.0
    centers = FIT.ShellBinning.shell_centers(b, 5.0)
    Test.@test length(centers) == length(edges) - 1
    Test.@test all(diff(centers) .> 0)
end

Test.@testset "ShellBinning — LogarithmicBinning" begin
    b = FIT.Types.LogarithmicBinning(1.0, 2.0)
    edges = FIT.ShellBinning.shell_edges(b, 16.0)
    Test.@test edges[1] == 1.0
    Test.@test issorted(edges)
    Test.@test all(edges[2:end] ./ edges[1:end-1] .≈ 2.0)
end

Test.@testset "ShellBinning — DyadicBinning vs LogarithmicBinning(2)" begin
    k_max = 16.0
    b_d = FIT.Types.DyadicBinning(1.0)
    b_l = FIT.Types.LogarithmicBinning(1.0, 2.0)
    Test.@test FIT.ShellBinning.shell_edges(b_d, k_max) == FIT.ShellBinning.shell_edges(b_l, k_max)
end

Test.@testset "ShellBinning — CustomBinning" begin
    edges = [0.0, 1.0, 3.0, 6.0, 10.0]
    b = FIT.Types.CustomBinning(edges)
    Test.@test FIT.ShellBinning.shell_edges(b, 10.0) == edges
    Test.@test FIT.ShellBinning.n_shells(b, 10.0) == 4
end

Test.@testset "ShellBinning — assign_shells" begin
    ks = FIT.Utils.wavenumber_grid((8,), (2π,))
    k_mag_1d = FIT.Utils.wavenumber_magnitude_grid(ks)
    b = FIT.Types.LinearBinning(2π / 8)
    edges = FIT.ShellBinning.shell_edges(b, maximum(k_mag_1d))
    idx = FIT.ShellBinning.assign_shells(k_mag_1d, edges)
    Test.@test size(idx) == size(k_mag_1d)
    Test.@test eltype(idx) === Int
    Test.@test all(0 .<= idx .<= length(edges) - 1)  # 0 = outside all shells
    Test.@test any(idx .== 1)                         # shell 1 is populated
end

# -----------------------------------------------------------------------
Test.@testset "Filters — spectral responses" begin
    k = 2.0; ℓ = 1.0
    # SharpSpectralFilter: passes k < π/ℓ ≈ 3.14
    Test.@test FIT.Filters.filter_response(FIT.Types.SharpSpectralFilter(), k, ℓ) == 1.0
    Test.@test FIT.Filters.filter_response(FIT.Types.SharpSpectralFilter(), 4.0, ℓ) == 0.0
    # GaussianFilter: always in (0,1], decays with k
    g1 = FIT.Filters.filter_response(FIT.Types.GaussianFilter(), k, ℓ)
    g2 = FIT.Filters.filter_response(FIT.Types.GaussianFilter(), 4.0, ℓ)
    Test.@test 0.0 < g2 < g1 <= 1.0
    # TopHatFilter: sinc, = 1 at k=0
    Test.@test FIT.Filters.filter_response(FIT.Types.TopHatFilter(), 0.0, ℓ) ≈ 1.0
end

Test.@testset "Filters — apply_filter_spectral!" begin
    k_mag = Float64[0, 1, 2, 3, 4]
    û_in  = ComplexF64[1.0, 1.0, 1.0, 1.0, 1.0]
    û_out = similar(û_in)
    FIT.Filters.apply_filter_spectral!(û_out, û_in, k_mag, FIT.Types.SharpSpectralFilter(), 1.0)
    # SharpSpectralFilter passes k < π/ℓ = π ≈ 3.14
    # k=0,1,2,3 < π → pass; k=4 > π → zeroed
    Test.@test û_out[1] ≈ 1.0  # k=0 passes
    Test.@test û_out[2] ≈ 1.0  # k=1 passes
    Test.@test û_out[3] ≈ 1.0  # k=2 passes
    Test.@test û_out[4] ≈ 1.0  # k=3 < π passes
    Test.@test û_out[5] ≈ 0.0  # k=4 > π zeroed
end

# -----------------------------------------------------------------------
Test.@testset "SpectralFlux — single-mode zero transfer" begin
    # Single Fourier mode u(x) = A*cos(k₀x): no nonlinear interaction → T(k)=0
    N = 8; L = 2π
    # Build û: only mode at index 2 (k = 2π/L = 1 rad/m)
    û = zeros(ComplexF64, N, 1)
    û[2, 1] = 0.5 * N   # corresponds to A*cos(k₀x) after IFFT normalisation
    û[N, 1] = 0.5 * N   # conjugate symmetric part
    # Divide by N to get FFTW-normalised coefficients
    û ./= N
    ks = FIT.Utils.wavenumber_grid((N,), (L,))

    result = FIT.SpectralFlux.calculate_spectral_flux(û, ks;
        binning = FIT.Types.LinearBinning(2π/L), dealiasing = FIT.Types.NoDealiasing())

    Test.@test result isa FIT.Types.SpectralFluxResult
    Test.@test length(result.k_shells) == length(result.transfer_spectrum) == length(result.flux)
    # For a single cosine mode, nonlinear term should be zero → T(k)≈0 everywhere
    Test.@test all(abs.(result.transfer_spectrum) .< 1e-10)
    Test.@test all(abs.(result.flux) .< 1e-10)
end

# -----------------------------------------------------------------------
Test.@testset "SpectralFlux — FFTW vs direct consistency (1D)" begin
    Random.seed!(42)
    N = 8; L = 2π
    # Two-mode field (will have a nonlinear interaction)
    k1, k2 = 1, 2
    x = range(0, 2π; length=N+1)[1:N]
    u = cos.(k1 .* x) .+ 0.3 .* sin.(k2 .* x)
    û_phys = ComplexF64.(reshape(FFTW.fft(u) ./ N, N, 1))
    ks = FIT.Utils.wavenumber_grid((N,), (L,))

    # Direct path
    result_direct = FIT.SpectralFlux.calculate_spectral_flux(û_phys, ks;
        binning=FIT.Types.LinearBinning(2π/L), dealiasing = FIT.Types.NoDealiasing(), spectral=SpectralBackends.DirectSumSpectralBackend())

    # FFTW path (extension)
    result_fft = FIT.SpectralFlux.calculate_spectral_flux(û_phys, ks;
        binning=FIT.Types.LinearBinning(2π/L), dealiasing = FIT.Types.NoDealiasing(), spectral=SpectralBackends.FFTSpectralBackend())

    Test.@test isapprox(result_direct.transfer_spectrum,
                            result_fft.transfer_spectrum; atol=1e-10)
    Test.@test isapprox(result_direct.flux, result_fft.flux; atol=1e-10)
end

# -----------------------------------------------------------------------
Test.@testset "SpectralFlux — energy conservation Σ T(k) ≈ 0 (dealiased, div-free)" begin
    # A divergence-free 2D field built from a streamfunction ψ:
    #   û_x = i k_y ψ̂,  û_y = −i k_x ψ̂  ⇒  k·û = 0, u real (ψ real).
    # With the corrected 2/3 dealiasing (INPUTS truncated, so no Nyquist mode and no
    # aliasing), the pseudospectral nonlinear term conserves energy exactly:
    # Σ_k Re{û*·N̂} = 0 by discrete skew-symmetry. This also distinguishes the fix from
    # the old output-only truncation (which leaves the retained band non-conserving).
    # (Without dealiasing the retained Nyquist mode breaks conservation at ~1e-8 — a
    #  standard pseudospectral artefact, not a bug.)
    N = 16; L = 2π
    Random.seed!(7)
    ψ  = randn(N, N)
    ψh = FFTW.fft(ψ) ./ N^2
    ks = FIT.Utils.wavenumber_grid((N, N), (L, L))
    kx = [ks[1][i] for i in 1:N, j in 1:N]
    ky = [ks[2][j] for i in 1:N, j in 1:N]
    û  = cat(im .* ky .* ψh, -im .* kx .* ψh; dims = 3)
    Test.@test maximum(abs.(kx .* û[:, :, 1] .+ ky .* û[:, :, 2])) < 1e-12  # div-free

    for spectral in (SpectralBackends.DirectSumSpectralBackend(), SpectralBackends.FFTSpectralBackend())
        N̂ = FIT.NonlinearTerm.compute_nonlinear_term(û, ks; dealiasing = FIT.Types.OrszagTwoThirds(), spectral = spectral)
        t = FIT.Invariants.transfer_density(FIT.Types.KineticEnergy(), û, N̂, ks)
        scale = sum(abs, t)
        Test.@test abs(sum(t)) < 1e-10 * scale       # energy-conserving, alias-free
    end
end

# -----------------------------------------------------------------------
Test.@testset "SpectralFlux — flux-sign convention Π = +cumsum(T)" begin
    # Pins the Alexakis–Biferale convention (Π>0 forward): flux is the *positive*
    # cumulative sum of the transfer spectrum (not negated).
    Random.seed!(11)
    N = 8; L = 2π
    x = range(0, L; length = N + 1)[1:N]
    u = cos.(x) .+ 0.3 .* sin.(2 .* x) .+ 0.1 .* cos.(3 .* x)
    û = ComplexF64.(reshape(FFTW.fft(u) ./ N, N, 1))
    ks = FIT.Utils.wavenumber_grid((N,), (L,))
    r = FIT.SpectralFlux.calculate_spectral_flux(û, ks; binning = FIT.Types.LinearBinning(2π/L), dealiasing = FIT.Types.NoDealiasing())
    Test.@test isapprox(r.flux, cumsum(r.transfer_spectrum); atol = 1e-12)
end

# -----------------------------------------------------------------------
# Spectral-layout gate: analytic wavenumber axes, and the Hermitian weight that makes a half-spectrum
# reduction equal the full-spectrum one. Every real-field diagnostic rests on this identity.
include("test_layout.jl")

# -----------------------------------------------------------------------
# The consequence of that identity: every public diagnostic returns the same answer on a real field's
# half spectrum as on its full one.
include("test_half_equivalence.jl")

# -----------------------------------------------------------------------
# Device paths on a real device-array type (JLArrays under allowscalar(false)), which the
# GPUBackend(KA.CPU()) proxy cannot exercise.
include("test_device_paths.jl")

# -----------------------------------------------------------------------
# Coarse-graining on a caller-supplied FlowGeometries grid: structured 2D/3D, curvilinear, node set.
include("test_geometry_grids.jl")

# -----------------------------------------------------------------------
# Comprehensive allocation contract (every !() path = 0 alloc; allocating entry points bounded).
include("test_allocs.jl")

# -----------------------------------------------------------------------
Test.@testset "Enstrophy — 2D conserved, 3D works (vortex stretching)" begin
    L = 2π
    # 2D: enstrophy is an inviscid invariant ⇒ Σ_k T_Ω ≈ 0 (div-free, dealiased).
    N = 16
    Random.seed!(5)
    ψ  = randn(N, N); ψh = FFTW.fft(ψ) ./ N^2
    ks2 = FIT.Utils.wavenumber_grid((N, N), (L, L))
    kx = [ks2[1][i] for i in 1:N, j in 1:N]; ky = [ks2[2][j] for i in 1:N, j in 1:N]
    û2 = cat(im .* ky .* ψh, -im .* kx .* ψh; dims = 3)
    N̂2 = FIT.NonlinearTerm.compute_nonlinear_term(û2, ks2; dealiasing = FIT.Types.OrszagTwoThirds(), spectral=SpectralBackends.FFTSpectralBackend())
    tΩ2 = FIT.Invariants.transfer_density(FIT.Types.Enstrophy(), û2, N̂2, ks2)
    Test.@test abs(sum(tΩ2)) < 1e-9 * sum(abs, tΩ2)        # 2D enstrophy conserved

    # 3D: vector-vorticity enstrophy transfer runs (non-conservative; sanity only).
    M = 8
    ks3 = FIT.Utils.wavenumber_grid((M, M, M), (L, L, L))
    Random.seed!(6)
    Â = randn(ComplexF64, M, M, M, 3)   # u = ∇×A ⇒ û = i k × Â is divergence-free
    kx3 = [ks3[1][i] for i in 1:M, j in 1:M, l in 1:M]
    ky3 = [ks3[2][j] for i in 1:M, j in 1:M, l in 1:M]
    kz3 = [ks3[3][l] for i in 1:M, j in 1:M, l in 1:M]
    ûx = im .* (ky3 .* Â[:, :, :, 3] .- kz3 .* Â[:, :, :, 2])
    ûy = im .* (kz3 .* Â[:, :, :, 1] .- kx3 .* Â[:, :, :, 3])
    ûz = im .* (kx3 .* Â[:, :, :, 2] .- ky3 .* Â[:, :, :, 1])
    û3 = cat(ûx, ûy, ûz; dims = 4)
    Test.@test maximum(abs.(kx3 .* ûx .+ ky3 .* ûy .+ kz3 .* ûz)) < 1e-10  # div-free
    res3 = FIT.SpectralFlux.calculate_spectral_flux(û3, ks3; binning = FIT.Types.LinearBinning(1.0),
        invariant = FIT.Types.Enstrophy(), dealiasing = FIT.Types.OrszagTwoThirds(), spectral=SpectralBackends.FFTSpectralBackend())
    Test.@test res3 isa FIT.Types.SpectralFluxResult
    Test.@test all(isfinite, res3.transfer_spectrum)
    # mode-to-mode aggregates now route through the FFT paths, so 3D enstrophy net works
    m2m3 = FIT.calculate_mode_to_mode_transfer(û3, ks3; invariant = FIT.Types.Enstrophy(), spectral=SpectralBackends.FFTSpectralBackend())
    Test.@test m2m3 isa FIT.Types.ModeToModeTriadResult
    Test.@test all(isfinite, m2m3.net_transfer)
end

# -----------------------------------------------------------------------
Test.@testset "NonlinearTerm — generalized advection of an M=1 scalar (u·∇)θ" begin
    # The generalized engine advects an arbitrary M-component field by the velocity. Here a
    # passive scalar (M=1) advected by a divergence-free 2D velocity: the direct and FFT
    # backends must agree, and scalar variance must be conserved (Σ_k Re{θ̂* N̂_θ} ≈ 0,
    # since ∫θ(u·∇)θ = −½∫θ²(∇·u) = 0 for incompressible u).
    N = 12; L = 2π
    Random.seed!(31)
    ks = FIT.Utils.wavenumber_grid((N, N), (L, L))
    kx = [ks[1][i] for i in 1:N, j in 1:N]; ky = [ks[2][j] for i in 1:N, j in 1:N]
    ψh = FFTW.fft(randn(N, N)) ./ N^2
    û  = cat(im .* ky .* ψh, -im .* kx .* ψh; dims = 3)        # div-free velocity (D=2)
    θ̂  = reshape(FFTW.fft(randn(N, N)) ./ N^2, N, N, 1)        # passive scalar (M=1)

    # N̂_θ = (u·∇)θ via direct DFT and FFT backends — must match.
    N̂_dir = FIT.NonlinearTerm.compute_nonlinear_term(θ̂, ks; dealiasing = FIT.Types.OrszagTwoThirds(),
        spectral = SpectralBackends.DirectSumSpectralBackend(), advecting_hat = û)
    N̂_fft = FIT.NonlinearTerm.compute_nonlinear_term(θ̂, ks; dealiasing = FIT.Types.OrszagTwoThirds(),
        spectral = SpectralBackends.FFTSpectralBackend(), advecting_hat = û)
    Test.@test size(N̂_dir) == (N, N, 1)
    Test.@test isapprox(N̂_dir, N̂_fft; atol = 1e-10 * maximum(abs, N̂_fft))

    # Scalar variance conservation: Σ_k Re{θ̂*(k) N̂_θ(k)} ≈ 0.
    tθ = FIT.Invariants.transfer_density(FIT.Types.PassiveScalar(), θ̂, N̂_fft, ks)
    Test.@test abs(sum(tθ)) < 1e-9 * sum(abs, tθ)
end

# -----------------------------------------------------------------------
Test.@testset "Dealiasing — exact 3/2 padding (PaddedThreeHalves)" begin
    # On a field band-limited to |k|<N/3, the 3/2-padded nonlinear term must MATCH the 2/3 rule
    # on the shared retained band (validates normalization) and ALSO carry the N/3≤|k|<N/2
    # content the 2/3 rule discards (the whole point of padding).
    N = 24; L = 2π
    ks = FIT.Utils.wavenumber_grid((N, N), (L, L))
    kx = [ks[1][i] for i in 1:N, j in 1:N]; ky = [ks[2][j] for i in 1:N, j in 1:N]
    kmag = sqrt.(kx.^2 .+ ky.^2)
    Random.seed!(81)
    ψ = zeros(ComplexF64, N, N)
    for i in 1:N, j in 1:N
        0 < kmag[i, j] < 8 && (ψ[i, j] = (randn() + im*randn()) / kmag[i, j]^2)
    end
    for i in 1:N, j in 1:N            # Hermitian symmetry ⇒ real physical field
        ci = i == 1 ? 1 : N - i + 2; cj = j == 1 ? 1 : N - j + 2
        (ci, cj) > (i, j) && (ψ[ci, cj] = conj(ψ[i, j]))
    end
    û = cat(im .* ky .* ψ, -im .* kx .* ψ; dims = 3)        # div-free, band-limited |k|<N/3

    N23  = FIT.NonlinearTerm.compute_nonlinear_term(û, ks; dealiasing = FIT.Types.OrszagTwoThirds(),   spectral = SpectralBackends.FFTSpectralBackend())
    Npad = FIT.NonlinearTerm.compute_nonlinear_term(û, ks; dealiasing = FIT.Types.PaddedThreeHalves(), spectral = SpectralBackends.FFTSpectralBackend())
    low = repeat(kmag .< 8, 1, 1, 2)
    mid = repeat((kmag .>= 8) .& (kmag .< 12), 1, 1, 2)
    Test.@test maximum(abs.(N23 .- Npad)[low]) < 1e-10 * maximum(abs.(Npad)[low])   # agree on |k|<N/3
    Test.@test sum(abs2, Npad[mid]) > 3 * sum(abs2, N23[mid])                       # padded keeps more

    # Padded spectral flux still conserves for an incompressible field.
    b  = FIT.Types.LinearBinning(2π/L)
    sf = FIT.SpectralFlux.calculate_spectral_flux(û, ks; binning = b, dealiasing = FIT.Types.PaddedThreeHalves(), spectral = SpectralBackends.FFTSpectralBackend())
    Test.@test abs(sum(sf.transfer_spectrum)) < 1e-9 * sum(abs, sf.transfer_spectrum)

    # Padding requires the FFT path; the dependency-free SpectralBackends.DirectSumSpectralBackend errors clearly.
    Test.@test_throws ArgumentError FIT.NonlinearTerm.compute_nonlinear_term(û, ks;
        dealiasing = FIT.Types.PaddedThreeHalves(), spectral = SpectralBackends.DirectSumSpectralBackend())
end

# -----------------------------------------------------------------------
Test.@testset "SpectralFlux — passive-scalar variance flux (conserved, cumulative→0)" begin
    N = 16; L = 2π
    Random.seed!(32)
    ks = FIT.Utils.wavenumber_grid((N, N), (L, L))
    kx = [ks[1][i] for i in 1:N, j in 1:N]; ky = [ks[2][j] for i in 1:N, j in 1:N]
    ψh = FFTW.fft(randn(N, N)) ./ N^2
    û  = cat(im .* ky .* ψh, -im .* kx .* ψh; dims = 3)        # div-free velocity
    θ  = FFTW.fft(randn(N, N)) ./ N^2                          # scalar as (N,N)
    b  = FIT.Types.LinearBinning(2π/L)

    res = FIT.calculate_scalar_flux(û, θ, ks; binning = b, dealiasing = FIT.Types.OrszagTwoThirds(),
        spectral = SpectralBackends.FFTSpectralBackend())
    Test.@test res isa FIT.Types.SpectralFluxResult
    sT = sqrt(sum(abs2, res.transfer_spectrum)); Test.@test sT > 0
    Test.@test abs(sum(res.transfer_spectrum)) < 1e-9 * sT       # variance conserved
    Test.@test abs(res.flux[end]) < 1e-9 * sT                    # cumulative flux returns to 0

    # Passing the scalar already shaped (N,N,1) gives the identical result.
    res1 = FIT.calculate_scalar_flux(û, reshape(θ, N, N, 1), ks; binning = b,
        dealiasing = FIT.Types.OrszagTwoThirds(), spectral = SpectralBackends.FFTSpectralBackend())
    Test.@test res1.transfer_spectrum ≈ res.transfer_spectrum
end

# -----------------------------------------------------------------------
Test.@testset "ShellToShellTransfer — antisymmetry (divergence-free field)" begin
    # T(n,m) = -T(m,n) holds exactly for divergence-free (incompressible) fields.
    # Build u = ∂ψ/∂y, v = -∂ψ/∂x from a random streamfunction ψ.
    Random.seed!(7)
    N = 16; L = 2π   # large enough that the 2/3-retained band has real inter-shell coupling
    Np = N * N
    ks = FIT.Utils.wavenumber_grid((N, N), (L, L))
    # Random streamfunction in spectral space (Hermitian so IFFT is real)
    ψ̂ = zeros(ComplexF64, N, N)
    Random.seed!(7)
    for ix in 1:N, iy in 1:N
        ψ̂[ix, iy] = randn() + im * randn()
    end
    # Enforce Hermitian symmetry: ψ̂[-k] = conj(ψ̂[k])
    for ix in 1:N, iy in 1:N
        cix = ix == 1 ? 1 : N - ix + 2
        ciy = iy == 1 ? 1 : N - iy + 2
        if (cix, ciy) > (ix, iy)
            ψ̂[cix, ciy] = conj(ψ̂[ix, iy])
        end
    end
    # No normalisation: use raw FFT convention û=fft(u) so T values are O(1)
    # û = iky·ψ̂,  v̂ = -ikx·ψ̂  → divergence-free by construction
    kx_vec = ks[1]; ky_vec = ks[2]
    û = zeros(ComplexF64, N, N, 2)
    for ix in 1:N, iy in 1:N
        û[ix, iy, 1] =  im * ky_vec[iy] * ψ̂[ix, iy]
        û[ix, iy, 2] = -im * kx_vec[ix] * ψ̂[ix, iy]
    end

    result = FIT.ShellToShellTransfer.calculate_shell_to_shell_transfer(û, ks;
        binning=FIT.Types.LinearBinning(2π/L), dealiasing = FIT.Types.OrszagTwoThirds(),
        verify_antisymmetry=true, spectral=SpectralBackends.FFTSpectralBackend())

    Test.@test result isa FIT.Types.ShellToShellResult
    T_norm = sqrt(sum(abs2, result.transfer_matrix))
    # T(n,m) = -T(m,n) exactly by construction; verify to machine precision
    Test.@test result.max_antisymmetry_error < 1e-12 * T_norm
end

# -----------------------------------------------------------------------
Test.@testset "ShellToShell — backend consistency + reduction to T(k)" begin
    # Divergence-free field from a random streamfunction (non-degenerate after dealiasing).
    N = 16; L = 2π
    Random.seed!(13)
    ψ  = randn(N, N)
    ψh = FFTW.fft(ψ) ./ N^2
    ks = FIT.Utils.wavenumber_grid((N, N), (L, L))
    kx = [ks[1][i] for i in 1:N, j in 1:N]
    ky = [ks[2][j] for i in 1:N, j in 1:N]
    û  = cat(im .* ky .* ψh, -im .* kx .* ψh; dims = 3)
    b  = FIT.Types.LinearBinning(2π/L)

    r_direct = FIT.ShellToShellTransfer.calculate_shell_to_shell_transfer(û, ks; binning=b, dealiasing = FIT.Types.OrszagTwoThirds(),
        verify_antisymmetry=true, spectral=SpectralBackends.DirectSumSpectralBackend())
    r_fft = FIT.ShellToShellTransfer.calculate_shell_to_shell_transfer(û, ks; binning=b, dealiasing = FIT.Types.OrszagTwoThirds(),
        verify_antisymmetry=true, spectral=SpectralBackends.FFTSpectralBackend())

    T_norm = sqrt(sum(abs2, r_direct.transfer_matrix))
    Test.@test T_norm > 0                                            # non-degenerate
    # serial and FFT implement the SAME (u·∇)u_m form → agree to roundoff
    Test.@test isapprox(r_direct.transfer_matrix, r_fft.transfer_matrix; atol = 1e-9 * T_norm)
    Test.@test r_direct.max_antisymmetry_error < 1e-10 * T_norm      # A is antisymmetric

    # Reduction: Σ_m T(n,m) must equal the spectral transfer T(k) (same field/binning).
    sf = FIT.SpectralFlux.calculate_spectral_flux(û, ks; binning = b, dealiasing = FIT.Types.OrszagTwoThirds())
    Test.@test isapprox(r_direct.net_transfer, sf.transfer_spectrum; atol = 1e-9 * T_norm)
end

# -----------------------------------------------------------------------
Test.@testset "ShellToShell — passive scalar T_θ(n,m): antisym, direct==FFT, reduces" begin
    N = 16; L = 2π
    Random.seed!(14)
    ks = FIT.Utils.wavenumber_grid((N, N), (L, L))
    kx = [ks[1][i] for i in 1:N, j in 1:N]; ky = [ks[2][j] for i in 1:N, j in 1:N]
    ψh = FFTW.fft(randn(N, N)) ./ N^2
    û  = cat(im .* ky .* ψh, -im .* kx .* ψh; dims = 3)        # div-free velocity
    θ  = FFTW.fft(randn(N, N)) ./ N^2                          # scalar
    b  = FIT.Types.LinearBinning(2π/L)

    r_dir = FIT.calculate_scalar_shell_to_shell_transfer(û, θ, ks; binning = b,
        dealiasing = FIT.Types.OrszagTwoThirds(), verify_antisymmetry = true, spectral = SpectralBackends.DirectSumSpectralBackend())
    r_fft = FIT.calculate_scalar_shell_to_shell_transfer(û, θ, ks; binning = b,
        dealiasing = FIT.Types.OrszagTwoThirds(), verify_antisymmetry = true, spectral = SpectralBackends.FFTSpectralBackend())

    T_norm = sqrt(sum(abs2, r_dir.transfer_matrix))
    Test.@test T_norm > 0                                                  # non-degenerate
    Test.@test isapprox(r_dir.transfer_matrix, r_fft.transfer_matrix; atol = 1e-9 * T_norm)
    Test.@test r_dir.max_antisymmetry_error < 1e-10 * T_norm               # T_θ antisymmetric
    # Reduction: Σ_m T_θ(n,m) == scalar transfer spectrum T_θ(k).
    sfθ = FIT.calculate_scalar_flux(û, θ, ks; binning = b, dealiasing = FIT.Types.OrszagTwoThirds(),
        spectral = SpectralBackends.FFTSpectralBackend())
    Test.@test isapprox(r_fft.net_transfer, sfθ.transfer_spectrum; atol = 1e-9 * T_norm)
end

# -----------------------------------------------------------------------
Test.@testset "ModeToMode — passive scalar S_θ(k|p): antisym, conserves, reduces" begin
    N = 12; L = 2π
    Random.seed!(15)
    ks = FIT.Utils.wavenumber_grid((N, N), (L, L))
    kx = [ks[1][i] for i in 1:N, j in 1:N]; ky = [ks[2][j] for i in 1:N, j in 1:N]
    ψh = FFTW.fft(randn(N, N)) ./ N^2
    û  = cat(im .* ky .* ψh, -im .* kx .* ψh; dims = 3)        # div-free velocity
    θ  = FFTW.fft(randn(N, N)) ./ N^2                          # scalar

    m2m = FIT.calculate_scalar_mode_to_mode_transfer(û, θ, ks; dealiasing = FIT.Types.OrszagTwoThirds(),
        spectral = SpectralBackends.FFTSpectralBackend())
    S   = m2m.transfer
    nrm = sqrt(sum(abs2, S)); Test.@test nrm > 0
    asym = 0.0
    for k in CartesianIndices((N, N)), p in CartesianIndices((N, N))
        asym = max(asym, abs(S[k, p] + S[p, k]))
    end
    Test.@test asym < 1e-10 * nrm                              # S_θ(k|p) = −S_θ(p|k)
    Test.@test abs(sum(S)) < 1e-10 * nrm                       # conserves
    # net (= Σ_p S_θ) shell-summed == scalar transfer spectrum
    b = FIT.Types.LinearBinning(2π/L)
    sfθ = FIT.calculate_scalar_flux(û, θ, ks; binning = b, dealiasing = FIT.Types.OrszagTwoThirds(),
        spectral = SpectralBackends.FFTSpectralBackend())
    kmag = FIT.Utils.wavenumber_magnitude_grid(ks)
    edges = FIT.ShellBinning.shell_edges(b, maximum(kmag)); sidx = FIT.ShellBinning.assign_shells(kmag, edges)
    netshell = zeros(length(edges) - 1)
    for I in CartesianIndices((N, N)); n = sidx[I]; n == 0 && continue; netshell[n] += m2m.net_transfer[I]; end
    Test.@test isapprox(netshell, sfθ.transfer_spectrum; atol = 1e-9 * sqrt(sum(abs2, sfθ.transfer_spectrum)))
end

# -----------------------------------------------------------------------
Test.@testset "Shell geometry — isotropic / perpendicular / parallel fluxes" begin
    M = 16; L = 2π
    ks = FIT.Utils.wavenumber_grid((M, M, M), (L, L, L))
    kx = [ks[1][i] for i in 1:M, j in 1:M, l in 1:M]
    ky = [ks[2][j] for i in 1:M, j in 1:M, l in 1:M]
    kz = [ks[3][l] for i in 1:M, j in 1:M, l in 1:M]
    # shell coordinates: isotropic == |k|; perpendicular uses kx,ky; parallel uses kz
    Test.@test FIT.ShellBinning.shell_coordinate(FIT.Types.IsotropicShells(), ks) ≈ FIT.Utils.wavenumber_magnitude_grid(ks)
    Test.@test FIT.ShellBinning.shell_coordinate(FIT.Types.PerpendicularShells(), ks) ≈ sqrt.(kx.^2 .+ ky.^2)
    Test.@test FIT.ShellBinning.shell_coordinate(FIT.Types.ParallelShells(), ks)      ≈ abs.(kz)

    # Divergence-free 3D velocity û = i k × Â (non-degenerate after dealiasing at M=16)
    Random.seed!(41)
    Â = randn(ComplexF64, M, M, M, 3)
    ûx = im .* (ky .* Â[:, :, :, 3] .- kz .* Â[:, :, :, 2])
    ûy = im .* (kz .* Â[:, :, :, 1] .- kx .* Â[:, :, :, 3])
    ûz = im .* (kx .* Â[:, :, :, 2] .- ky .* Â[:, :, :, 1])
    û3 = cat(ûx, ûy, ûz; dims = 4)
    b  = FIT.Types.LinearBinning(2π/L)

    r_def  = FIT.SpectralFlux.calculate_spectral_flux(û3, ks; binning=b, dealiasing = FIT.Types.OrszagTwoThirds(), spectral=SpectralBackends.FFTSpectralBackend())
    r_iso  = FIT.SpectralFlux.calculate_spectral_flux(û3, ks; binning=b, dealiasing = FIT.Types.OrszagTwoThirds(),
        spectral=SpectralBackends.FFTSpectralBackend(), geometry=FIT.Types.IsotropicShells())
    r_perp = FIT.SpectralFlux.calculate_spectral_flux(û3, ks; binning=b, dealiasing = FIT.Types.OrszagTwoThirds(),
        spectral=SpectralBackends.FFTSpectralBackend(), geometry=FIT.Types.PerpendicularShells())
    r_par  = FIT.SpectralFlux.calculate_spectral_flux(û3, ks; binning=b, dealiasing = FIT.Types.OrszagTwoThirds(),
        spectral=SpectralBackends.FFTSpectralBackend(), geometry=FIT.Types.ParallelShells())

    # Default geometry IS isotropic (backward compatible).
    Test.@test r_iso.transfer_spectrum == r_def.transfer_spectrum
    Test.@test all(isfinite, r_perp.transfer_spectrum)
    Test.@test all(isfinite, r_par.transfer_spectrum)
    sT = sqrt(sum(abs2, r_iso.transfer_spectrum)); Test.@test sT > 0   # non-degenerate
    # Isotropic shells cover every non-DC mode (DC density = 0 here), so the isotropic flux
    # conserves: Σ_k T(|k|) ≈ 0. (Anisotropic geometries drop a zero-coordinate plane each,
    # so their sums need NOT vanish — geometry repartitions, it does not conserve per-axis.)
    Test.@test abs(sum(r_iso.transfer_spectrum)) < 1e-9 * sum(abs, r_iso.transfer_spectrum)
    # Geometry genuinely changes the partition: fewer shells along a single axis than |k|.
    Test.@test length(r_par.transfer_spectrum) < length(r_iso.transfer_spectrum)
end

# -----------------------------------------------------------------------
Test.@testset "HelicalDecomposition — reconstruct, orthonormal, helicity split, flux" begin
    M = 16; L = 2π
    ks = FIT.Utils.wavenumber_grid((M, M, M), (L, L, L))
    kx = [ks[1][i] for i in 1:M, j in 1:M, l in 1:M]
    ky = [ks[2][j] for i in 1:M, j in 1:M, l in 1:M]
    kz = [ks[3][l] for i in 1:M, j in 1:M, l in 1:M]
    Random.seed!(51)
    Â = randn(ComplexF64, M, M, M, 3)
    ûx = im .* (ky .* Â[:, :, :, 3] .- kz .* Â[:, :, :, 2])
    ûy = im .* (kz .* Â[:, :, :, 1] .- kx .* Â[:, :, :, 3])
    ûz = im .* (kx .* Â[:, :, :, 2] .- ky .* Â[:, :, :, 1])
    û  = cat(ûx, ûy, ûz; dims = 4)                     # divergence-free by construction

    dec = FIT.Decomposition.decompose_field(FIT.Types.HelicalDecomposition(), û, ks)
    up = dec.positive; um = dec.negative
    # 1. reconstruction u₊ + u₋ ≈ û (incompressible)
    Test.@test isapprox(up .+ um, û; atol = 1e-12 * maximum(abs, û))
    # 2. orthonormal split: Σ(|u₊|²+|u₋|²) ≈ Σ|û|²
    Test.@test isapprox(sum(abs2, up) + sum(abs2, um), sum(abs2, û); rtol = 1e-12)
    # 3. each helical component is divergence-free (k·u± = 0)
    divp = kx .* up[:,:,:,1] .+ ky .* up[:,:,:,2] .+ kz .* up[:,:,:,3]
    divm = kx .* um[:,:,:,1] .+ ky .* um[:,:,:,2] .+ kz .* um[:,:,:,3]
    Test.@test maximum(abs, divp) < 1e-10 * maximum(abs, û)
    Test.@test maximum(abs, divm) < 1e-10 * maximum(abs, û)
    # 4. helicity split: Σ Re{û*·(ik×û)} ≈ Σ |k|(|u₊|² − |u₋|²)
    ωx = im .* (ky .* ûz .- kz .* ûy); ωy = im .* (kz .* ûx .- kx .* ûz); ωz = im .* (kx .* ûy .- ky .* ûx)
    H_direct = sum(real.(conj.(ûx).*ωx .+ conj.(ûy).*ωy .+ conj.(ûz).*ωz))
    kmag = sqrt.(kx.^2 .+ ky.^2 .+ kz.^2)
    H_heli = sum(kmag .* (sum(abs2, up; dims=4)[:,:,:,1] .- sum(abs2, um; dims=4)[:,:,:,1]))
    Test.@test isapprox(H_direct, H_heli; rtol = 1e-10)
    # 5. helicity-resolved energy flux: Π⁺ + Π⁻ == total KE flux (same N̂, u₊+u₋=û)
    b = FIT.Types.LinearBinning(2π/L)
    rtot = FIT.SpectralFlux.calculate_spectral_flux(û, ks; binning=b, dealiasing = FIT.Types.OrszagTwoThirds(), spectral=SpectralBackends.FFTSpectralBackend())
    rhel = FIT.SpectralFlux.calculate_spectral_flux(û, ks; binning=b, dealiasing = FIT.Types.OrszagTwoThirds(), spectral=SpectralBackends.FFTSpectralBackend(),
        decomposition=FIT.Types.HelicalDecomposition())
    Test.@test rhel isa NamedTuple
    Test.@test isapprox(rhel.positive.transfer_spectrum .+ rhel.negative.transfer_spectrum,
        rtot.transfer_spectrum; atol = 1e-9 * (sqrt(sum(abs2, rtot.transfer_spectrum)) + eps()))
end

# -----------------------------------------------------------------------
Test.@testset "Helical partial fluxes — 8 channels sum to the total energy flux" begin
    M = 16; L = 2π
    ks = FIT.Utils.wavenumber_grid((M, M, M), (L, L, L))
    kx = [ks[1][i] for i in 1:M, j in 1:M, l in 1:M]
    ky = [ks[2][j] for i in 1:M, j in 1:M, l in 1:M]
    kz = [ks[3][l] for i in 1:M, j in 1:M, l in 1:M]
    Random.seed!(91)
    Â = randn(ComplexF64, M, M, M, 3)
    ûx = im .* (ky .* Â[:, :, :, 3] .- kz .* Â[:, :, :, 2])
    ûy = im .* (kz .* Â[:, :, :, 1] .- kx .* Â[:, :, :, 3])
    ûz = im .* (kx .* Â[:, :, :, 2] .- ky .* Â[:, :, :, 1])
    û  = cat(ûx, ûy, ûz; dims = 4)
    b  = FIT.Types.LinearBinning(2π/L)

    hp = FIT.calculate_helical_partial_fluxes(û, ks; binning=b, dealiasing=FIT.Types.OrszagTwoThirds(),
        spectral=SpectralBackends.FFTSpectralBackend())
    Test.@test length(hp.channels) == 8                              # all (s_k,s_p,s_q) present
    sf = FIT.SpectralFlux.calculate_spectral_flux(û, ks; binning=b, dealiasing=FIT.Types.OrszagTwoThirds(), spectral=SpectralBackends.FFTSpectralBackend())
    # The 8 channels reconstruct the full KE flux. The individual channels are large and partly
    # cancel into a smaller total, so compare at a tolerance set by the CHANNEL scale.
    chan_scale = maximum(sqrt(sum(abs2, c.transfer_spectrum)) for c in values(hp.channels))
    Test.@test chan_scale > 0
    Test.@test isapprox(hp.total.transfer_spectrum, sf.transfer_spectrum; atol = 1e-10 * chan_scale)

    # Exported in-place wrapper reuses a NonlinearTermWorkspace and matches the allocating form.
    wsh = FIT.Workspaces.NonlinearTermWorkspace(û, ks; dealiasing = FIT.Types.OrszagTwoThirds())
    hp_ip = FIT.calculate_helical_partial_fluxes!(wsh, û, ks; binning = b, dealiasing = FIT.Types.OrszagTwoThirds(),
        spectral = SpectralBackends.FFTSpectralBackend())
    Test.@test length(hp_ip.channels) == 8
    Test.@test isapprox(hp_ip.total.transfer_spectrum, hp.total.transfer_spectrum; atol = 1e-10 * chan_scale)
end

# -----------------------------------------------------------------------
Test.@testset "Helmholtz partial fluxes — incompressible ⇒ rotational channel only" begin
    # Same generic machinery, Helmholtz (rot/div) decomposition. For a divergence-free field the
    # divergent component vanishes, so only the (rotational,rotational,rotational) channel is
    # non-zero and it equals the full flux; the rot↔div cross channels are ≈ 0.
    N = 16; L = 2π
    Random.seed!(92)
    ks = FIT.Utils.wavenumber_grid((N, N), (L, L))
    kx = [ks[1][i] for i in 1:N, j in 1:N]; ky = [ks[2][j] for i in 1:N, j in 1:N]
    ψh = FFTW.fft(randn(N, N)) ./ N^2
    û  = cat(im .* ky .* ψh, -im .* kx .* ψh; dims = 3)        # div-free (rotational only)
    b  = FIT.Types.LinearBinning(2π/L)

    hp = FIT.calculate_partial_fluxes(û, ks; decomposition=FIT.Types.HelmholtzDecomposition(),
        binning=b, dealiasing=FIT.Types.OrszagTwoThirds(), spectral=SpectralBackends.FFTSpectralBackend())
    Test.@test length(hp.channels) == 8
    sf = FIT.SpectralFlux.calculate_spectral_flux(û, ks; binning=b, dealiasing=FIT.Types.OrszagTwoThirds(), spectral=SpectralBackends.FFTSpectralBackend())
    sT = sqrt(sum(abs2, sf.transfer_spectrum)); Test.@test sT > 0
    rrr = hp.channels[(:rotational, :rotational, :rotational)]
    Test.@test isapprox(rrr.transfer_spectrum, sf.transfer_spectrum; atol = 1e-9 * sT)
    # every channel touching the (empty) divergent component is ≈ 0
    for key in keys(hp.channels)
        any(==(:divergent), key) || continue
        Test.@test maximum(abs, hp.channels[key].transfer_spectrum) < 1e-9 * sT
    end
end

# -----------------------------------------------------------------------
Test.@testset "ToroidalPoloidalDecomposition — reconstruct, div-free, toroidal has w=0" begin
    M = 8; L = 2π
    ks = FIT.Utils.wavenumber_grid((M, M, M), (L, L, L))
    kx = [ks[1][i] for i in 1:M, j in 1:M, l in 1:M]
    ky = [ks[2][j] for i in 1:M, j in 1:M, l in 1:M]
    kz = [ks[3][l] for i in 1:M, j in 1:M, l in 1:M]
    Random.seed!(52)
    Â = randn(ComplexF64, M, M, M, 3)
    ûx = im .* (ky .* Â[:, :, :, 3] .- kz .* Â[:, :, :, 2])
    ûy = im .* (kz .* Â[:, :, :, 1] .- kx .* Â[:, :, :, 3])
    ûz = im .* (kx .* Â[:, :, :, 2] .- ky .* Â[:, :, :, 1])
    û  = cat(ûx, ûy, ûz; dims = 4)                     # solenoidal

    dec = FIT.Decomposition.decompose_field(FIT.Types.ToroidalPoloidalDecomposition(), û, ks)
    tor = dec.toroidal; pol = dec.poloidal
    Test.@test isapprox(tor .+ pol, û; atol = 1e-12 * maximum(abs, û))            # reconstruction
    Test.@test isapprox(sum(abs2, tor) + sum(abs2, pol), sum(abs2, û); rtol = 1e-12)  # orthogonal
    # toroidal mode is horizontal: zero vertical velocity
    Test.@test maximum(abs, tor[:, :, :, 3]) < 1e-12 * maximum(abs, û)
    # both divergence-free
    for f in (tor, pol)
        divf = kx .* f[:,:,:,1] .+ ky .* f[:,:,:,2] .+ kz .* f[:,:,:,3]
        Test.@test maximum(abs, divf) < 1e-10 * maximum(abs, û)
    end
end

# -----------------------------------------------------------------------
Test.@testset "HelmholtzDecomposition — 3D Leray: reconstruct, rot div-free, div curl-free" begin
    M = 8; L = 2π
    ks = FIT.Utils.wavenumber_grid((M, M, M), (L, L, L))
    kx = [ks[1][i] for i in 1:M, j in 1:M, l in 1:M]
    ky = [ks[2][j] for i in 1:M, j in 1:M, l in 1:M]
    kz = [ks[3][l] for i in 1:M, j in 1:M, l in 1:M]
    Random.seed!(53)
    û = randn(ComplexF64, M, M, M, 3)                     # generic: both rotational + divergent parts
    dec = FIT.Decomposition.decompose_field(FIT.Types.HelmholtzDecomposition(), û, ks)
    rot = dec.rotational; div = dec.divergent
    Test.@test isapprox(rot .+ div, û; atol = 1e-12 * maximum(abs, û))                # reconstruction
    drot = kx .* rot[:,:,:,1] .+ ky .* rot[:,:,:,2] .+ kz .* rot[:,:,:,3]             # rot: k·û_rot ≈ 0
    Test.@test maximum(abs, drot) < 1e-10 * maximum(abs, û)
    cx = ky .* div[:,:,:,3] .- kz .* div[:,:,:,2]                                     # div: k×û_div ≈ 0
    cy = kz .* div[:,:,:,1] .- kx .* div[:,:,:,3]
    cz = kx .* div[:,:,:,2] .- ky .* div[:,:,:,1]
    Test.@test max(maximum(abs, cx), maximum(abs, cy), maximum(abs, cz)) < 1e-10 * maximum(abs, û)
    # RotationalDecomposition / DivergentDecomposition return the single corresponding component.
    Test.@test FIT.Decomposition.decompose_field(FIT.Types.RotationalDecomposition(), û, ks) ≈ rot
    Test.@test FIT.Decomposition.decompose_field(FIT.Types.DivergentDecomposition(), û, ks) ≈ div
end

# -----------------------------------------------------------------------
Test.@testset "BandToBand — smooth T(K,Q): antisymmetric, conserves, reduces" begin
    N = 16; L = 2π
    Random.seed!(71)
    ks = FIT.Utils.wavenumber_grid((N, N), (L, L))
    kx = [ks[1][i] for i in 1:N, j in 1:N]; ky = [ks[2][j] for i in 1:N, j in 1:N]
    ψh = FFTW.fft(randn(N, N)) ./ N^2
    û  = cat(im .* ky .* ψh, -im .* kx .* ψh; dims = 3)        # div-free
    centers = [1.0, 2.0, 3.0, 4.0]
    bands = FIT.Types.SmoothBands(centers; logwidth = 0.5)

    r = FIT.BandTransfer.calculate_band_to_band_transfer(û, ks; bands = bands, dealiasing = FIT.Types.OrszagTwoThirds(),
        spectral = SpectralBackends.FFTSpectralBackend())
    T = r.transfer_matrix
    Tn = sqrt(sum(abs2, T)); Test.@test Tn > 0
    Test.@test r.max_antisymmetry_error < 1e-10 * Tn                     # antisymmetric
    Test.@test abs(sum(T)) < 1e-10 * Tn                                 # conserves
    Test.@test isapprox(r.net_transfer, vec(sum(T, dims = 2)); atol = 1e-12 * Tn)  # net = Σ_m T(n,m)
    # net summed over bands ≈ 0 (banded transfer spectrum integrates to zero)
    Test.@test abs(sum(r.net_transfer)) < 1e-10 * Tn
end

# -----------------------------------------------------------------------
Test.@testset "CoarseGrainingFlux — CGEF loaded" begin
    # CoarseGrainingEnergyFluxes is loaded at the top of this file, so the call should succeed
    N = 4; L = 2π
    x = [L * (i-1) / N for i in 1:N]
    y = [L * (j-1) / N for j in 1:N]
    u = zeros(N, N); v = zeros(N, N)
    result = FIT.CoarseGrainingFlux.calculate_coarse_graining_flux(
        (u, v), (x, y), π/2, FIT.Types.GaussianFilter())
    Test.@test result isa FIT.Types.CoarseGrainingFluxResult
end

# -----------------------------------------------------------------------
Test.@testset "CoarseGrainingFlux — 3D Cartesian Π_ℓ + diagnostics == CGEF directly" begin
    FT = Float64; N = 12; L = 2π
    xs = collect(range(0, L; length = N + 1)[1:N]); ys = copy(xs); zs = copy(xs)
    Random.seed!(29)
    u = randn(N, N, N); v = randn(N, N, N); w = randn(N, N, N); ℓ = L / 4
    r = FIT.CoarseGrainingFlux.calculate_coarse_graining_flux((u, v, w), (xs, ys, zs), ℓ, FIT.Types.GaussianFilter())
    Test.@test r isa FIT.Types.CoarseGrainingFluxResult
    Test.@test size(r.flux_field) == (N, N, N)
    Test.@test all(isfinite, r.flux_field)
    # Exact match to the sibling's true-3D compute_Π! on the same field/grid (wiring correctness).
    geom = CoarseGrainingEnergyFluxes.FlowGeometries.Geometry.CartesianGeometry{FT}()
    grid = CoarseGrainingEnergyFluxes.FlowGeometries.Grids.StructuredGrid(geom, FT.(xs), FT.(ys), FT.(zs), trues(N, N, N))
    Πref = zeros(FT, N, N, N)
    CoarseGrainingEnergyFluxes.Diagnostics.compute_Π!(Πref, u, v, w, grid,
        CoarseGrainingEnergyFluxes.Kernels.GaussianKernel(), FT(ℓ))
    Test.@test maximum(abs, r.flux_field .- Πref) == 0.0
    # 3D diagnostics: (ns, 3, 3) symmetric stress/strain tensors.
    rd = FIT.CoarseGrainingFlux.calculate_coarse_graining_flux((u, v, w), (xs, ys, zs), ℓ, FIT.Types.GaussianFilter(); return_diagnostics = true)
    Test.@test size(rd.stress_tensor) == (N, N, N, 3, 3)
    Test.@test maximum(abs, rd.stress_tensor .- permutedims(rd.stress_tensor, (1,2,3,5,4))) == 0.0
    Test.@test maximum(abs, rd.strain_rate  .- permutedims(rd.strain_rate,  (1,2,3,5,4))) == 0.0
end

# -----------------------------------------------------------------------
Test.@testset "CoarseGrainingFlux — spherical (lon–lat) Π_ℓ == CGEF directly" begin
    FT = Float64; R = 6.371e6
    Nlon, Nlat = 24, 20
    lon = collect(range(0, 2π; length = Nlon + 1)[1:Nlon])   # radians, periodic
    lat = collect(range(-1.2, 1.2; length = Nlat))           # radians, away from the poles
    Random.seed!(37)
    u = randn(Nlon, Nlat); v = randn(Nlon, Nlat); ℓ = 5.0e5
    # radius=… selects the spherical grid; the sibling implements the spherical filter stencils.
    r = FIT.CoarseGrainingFlux.calculate_coarse_graining_flux((u, v), (lon, lat), ℓ, FIT.Types.GaussianFilter(); radius = R)
    Test.@test r isa FIT.Types.CoarseGrainingFluxResult
    Test.@test all(isfinite, r.flux_field)
    # Exact match to the sibling's spherical compute_Π! on the same grid (wiring correctness).
    geom = CoarseGrainingEnergyFluxes.FlowGeometries.Geometry.SphericalGeometry(R)
    grid = CoarseGrainingEnergyFluxes.FlowGeometries.Grids.StructuredGrid(geom, lon, lat, trues(Nlon, Nlat))
    Πref = zeros(FT, Nlon, Nlat)
    CoarseGrainingEnergyFluxes.Diagnostics.compute_Π!(Πref, u, v, nothing, grid,
        CoarseGrainingEnergyFluxes.Kernels.GaussianKernel(), FT(ℓ))
    Test.@test maximum(abs, r.flux_field .- Πref) == 0.0
    # Spherical is a 2D lon–lat surface: a 3D request errors clearly.
    Test.@test_throws ArgumentError FIT.CoarseGrainingFlux.calculate_coarse_graining_flux(
        (u, v, u), (lon, lat, lat), ℓ, FIT.Types.GaussianFilter(); radius = R)
end

# -----------------------------------------------------------------------
Test.@testset "CoarseGrainingFlux — in-place !() + diagnostics + workspace reuse" begin
    Random.seed!(7)
    N = 32; L = 2π
    xs = collect(range(0, L; length = N + 1)[1:N]); ys = copy(xs)
    u  = [sin(xs[i]) * cos(ys[j]) + 0.2 * randn() for i in 1:N, j in 1:N]
    v  = [-cos(xs[i]) * sin(ys[j]) + 0.2 * randn() for i in 1:N, j in 1:N]
    filt = FIT.Types.GaussianFilter(); ℓ = 0.5

    # in-place matches the allocating path to machine precision (same computation)
    r0 = FIT.CoarseGrainingFlux.calculate_coarse_graining_flux((u, v), (xs, ys), ℓ, filt)
    ws = FIT.CoarseGrainingFlux.CoarseGrainingFluxWorkspace((u, v), (xs, ys), ℓ, filt)
    r1 = FIT.CoarseGrainingFlux.calculate_coarse_graining_flux!(ws, (u, v))
    Test.@test r1 isa FIT.Types.CoarseGrainingFluxResult
    Test.@test maximum(abs.(r0.flux_field .- r1.flux_field)) == 0.0
    Test.@test r0.mean_flux == r1.mean_flux

    # return_diagnostics=true works end-to-end (regression: the diagnostics result type used to
    # over-constrain the 4-D τ̄/S̄ to the 2-D flux array type, so this path errored for everyone).
    r0d = FIT.CoarseGrainingFlux.calculate_coarse_graining_flux((u, v), (xs, ys), ℓ, filt; return_diagnostics = true)
    Test.@test r0d isa FIT.Types.CoarseGrainingFluxResultWithDiagnostics
    Test.@test size(r0d.stress_tensor) == (N, N, 2, 2)
    wsd = FIT.CoarseGrainingFlux.CoarseGrainingFluxWorkspace((u, v), (xs, ys), ℓ, filt; return_diagnostics = true)
    r1d = FIT.CoarseGrainingFlux.calculate_coarse_graining_flux!(wsd, (u, v))
    Test.@test maximum(abs.(r0d.stress_tensor .- r1d.stress_tensor)) == 0.0
    Test.@test maximum(abs.(r0d.strain_rate  .- r1d.strain_rate))  == 0.0

    # Filter-scale sweep: the workspace is scale-specific (footprint fixed at construction), so build one
    # per scale; the reused `!` matches the allocating path.
    for l in (0.3, 0.5, 0.8, 1.2)
        ra = FIT.CoarseGrainingFlux.calculate_coarse_graining_flux((u, v), (xs, ys), l, filt)
        wl = FIT.CoarseGrainingFlux.CoarseGrainingFluxWorkspace((u, v), (xs, ys), l, filt)
        rb = FIT.CoarseGrainingFlux.calculate_coarse_graining_flux!(wl, (u, v))
        Test.@test maximum(abs.(ra.flux_field .- rb.flux_field)) == 0.0
    end

    # (workspace-reuse allocation ratio asserted in test_allocs.jl)
end

# -----------------------------------------------------------------------
Test.@testset "calculate_energy_transfer — unified dispatch" begin
    N = 8; L = 2π
    ks = FIT.Utils.wavenumber_grid((N,), (L,))
    û = zeros(ComplexF64, N, 1)

    r1 = FIT.calculate_energy_transfer(
        FIT.Types.SpectralFluxMethod(FIT.Types.LinearBinning(2π/L)), û, ks)
    Test.@test r1 isa FIT.Types.SpectralFluxResult

    r2 = FIT.calculate_energy_transfer(
        FIT.Types.ShellToShellTransferMethod(FIT.Types.LinearBinning(2π/L)), û, ks)
    Test.@test r2 isa FIT.Types.ShellToShellResult

    x = [L * (i-1) / N for i in 1:N]
    y = [L * (j-1) / N for j in 1:N]
    u = zeros(N, N); v = zeros(N, N)
    r3 = FIT.calculate_energy_transfer(
        FIT.Types.CoarseGrainingFluxMethod(FIT.Types.GaussianFilter(), Float64(π/2)),
        (u, v), (x, y))
    Test.@test r3 isa FIT.Types.CoarseGrainingFluxResult
end

# -----------------------------------------------------------------------
Test.@testset "to_spectral — physical-space entry (uniform Cartesian grid)" begin
    N = 16; L = 2π
    x = range(0.0, L; length = N + 1)[1:N]
    y = range(0.0, L; length = N + 1)[1:N]
    Random.seed!(11)
    u = randn(N, N); v = randn(N, N)

    # Real input takes the half (r2c) layout by default: the non-redundant `k₁ ≥ 0` coefficients and
    # an `rfftfreq` first axis, reproducing the canonical û = rfft(u)/Nᵈ exactly.
    û_ts, ks_ts = FIT.to_spectral((u, v), (x, y); spectral = SpectralBackends.FFTSpectralBackend())
    û_half = cat(FFTW.rfft(u), FFTW.rfft(v); dims = 3) ./ N^2
    ks_half = FIT.Utils.wavenumber_grid((N, N), (L, L); real = true)
    Test.@test size(û_ts) == (N ÷ 2 + 1, N, 2)
    Test.@test isapprox(û_ts, û_half; atol = 1e-12)
    Test.@test all(isapprox.(ks_ts, ks_half; atol = 1e-12))

    # `real_layout = false` keeps the full complex spectrum, û = fft(u)/Nᵈ on the fftfreq grid.
    û_full, ks_full = FIT.to_spectral((u, v), (x, y); spectral = SpectralBackends.FFTSpectralBackend(),
                                      real_layout = false)
    û_man = cat(FFTW.fft(u), FFTW.fft(v); dims = 3) ./ N^2
    ks_man = FIT.Utils.wavenumber_grid((N, N), (L, L))
    Test.@test isapprox(û_full, û_man; atol = 1e-12)
    Test.@test all(isapprox.(ks_full, ks_man; atol = 1e-12))

    # DirectSum reference agrees with FFT, on both layouts.
    û_ds, _ = FIT.to_spectral((u, v), (x, y); spectral = SpectralBackends.DirectSumSpectralBackend(),
                              real_layout = false)
    Test.@test isapprox(û_ds, û_man; atol = 1e-10)
    û_dsh, _ = FIT.to_spectral((u, v), (x, y); spectral = SpectralBackends.DirectSumSpectralBackend())
    Test.@test isapprox(û_dsh, û_half; atol = 1e-10)

    # The two layouts are the same field, so every diagnostic built on them agrees: the half spectrum
    # carries the Hermitian weight that makes its shell sums equal the full-spectrum ones.
    Π_ts  = FIT.SpectralFlux.calculate_spectral_flux(û_ts, ks_ts; spectral = SpectralBackends.FFTSpectralBackend())
    Π_man = FIT.SpectralFlux.calculate_spectral_flux(û_man, ks_man; spectral = SpectralBackends.FFTSpectralBackend())
    Test.@test maximum(abs, Π_man.flux) > 1e-8
    Test.@test isapprox(Π_ts.flux, Π_man.flux; rtol = 1e-10)

    # Scalar 1-tuple (density/pressure/passive-scalar style).
    ρ = randn(N, N)
    ρ̂, _ = FIT.to_spectral((ρ,), (x, y); spectral = SpectralBackends.FFTSpectralBackend())
    Test.@test size(ρ̂) == (N ÷ 2 + 1, N, 1)
    Test.@test isapprox(ρ̂[:, :, 1], FFTW.rfft(ρ) ./ N^2; atol = 1e-12)

    # Complex input has no Hermitian symmetry to exploit and stays on the full grid.
    ûc, ksc = FIT.to_spectral((ComplexF64.(u), ComplexF64.(v)), (x, y); spectral = SpectralBackends.FFTSpectralBackend())
    Test.@test size(ûc) == (N, N, 2)
    Test.@test isapprox(ûc, û_man; atol = 1e-12)

    # Scattered / spherical transforms are a geometry mismatch here → clear error, not a silent misroute.
    Test.@test_throws ArgumentError FIT.to_spectral((u, v), (x, y); spectral = FIT.Types.FINUFFTBackend())
    Test.@test_throws ArgumentError FIT.to_spectral((u, v), (x, y); spectral = SpectralBackends.FSHTSpectralBackend())
    Test.@test_throws ArgumentError FIT.to_spectral((u, v), (x, y); spectral = SpectralBackends.NUFSHTSpectralBackend())
end

# -----------------------------------------------------------------------
Test.@testset "assign_shells" begin
    ks  = FIT.Utils.wavenumber_grid((8,), (2π,))
    k_mag = FIT.Utils.wavenumber_magnitude_grid(ks)
    b     = FIT.Types.LinearBinning(2π/8)
    edges = FIT.ShellBinning.shell_edges(b, maximum(k_mag))
    idx   = FIT.ShellBinning.assign_shells(k_mag, edges)
    Test.@test size(idx) == size(k_mag)
    Test.@test eltype(idx) == Int
    N_sh  = length(edges) - 1
    Test.@test all(0 .<= idx .<= N_sh)
    for n in 1:N_sh
        for I in CartesianIndices(k_mag)
            if idx[I] == n
                Test.@test edges[n] <= k_mag[I] < edges[n+1]
            end
        end
    end
end

# -----------------------------------------------------------------------
Test.@testset "SpectralFlux !-variant" begin
    N = 8; L = 2π
    ks  = FIT.Utils.wavenumber_grid((N,), (L,))
    û  = zeros(ComplexF64, N, 1)
    b   = FIT.Types.LinearBinning(2π/L)
    ws  = FIT.Workspaces.SpectralFluxWorkspace(û, ks, b)
    k_mag     = FIT.Utils.wavenumber_magnitude_grid(ks)
    edges     = FIT.ShellBinning.shell_edges(b, maximum(k_mag))
    centers   = FIT.ShellBinning.shell_centers(b, maximum(k_mag))
    shell_idx = FIT.ShellBinning.assign_shells(k_mag, edges)
    result    = FIT.Types.SpectralFluxResult(centers, similar(ws.T_spec), similar(ws.flux))
    FIT.SpectralFlux.calculate_spectral_flux!(result, ws, û, ks, shell_idx; dealiasing = FIT.Types.NoDealiasing())
    Test.@test result isa FIT.Types.SpectralFluxResult
    Test.@test all(abs.(result.transfer_spectrum) .< 1e-14)
end

# -----------------------------------------------------------------------
Test.@testset "ShellToShellTransfer !-variant" begin
    N = 6; L = 2π
    ks = FIT.Utils.wavenumber_grid((N, N), (L, L))
    û  = zeros(ComplexF64, N, N, 2)
    b   = FIT.Types.LinearBinning(2π/L)
    ws  = FIT.Workspaces.ShellToShellWorkspace(û, ks, b)
    k_mag   = FIT.Utils.wavenumber_magnitude_grid(ks)
    edges   = FIT.ShellBinning.shell_edges(b, maximum(k_mag))
    centers = FIT.ShellBinning.shell_centers(b, maximum(k_mag))
    N_sh    = length(centers)
    FT      = Float64
    result  = FIT.Types.ShellToShellResult(
        centers, edges,
        Matrix{FT}(undef, N_sh, N_sh),
        Vector{FT}(undef, N_sh),
        FT(NaN),
    )
    FIT.ShellToShellTransfer.calculate_shell_to_shell_transfer!(result, ws, û, ks;
        dealiasing = FIT.Types.NoDealiasing(), verify_antisymmetry=false)
    Test.@test result isa FIT.Types.ShellToShellResult
    Test.@test all(abs.(result.transfer_matrix) .< 1e-14)
end

# -----------------------------------------------------------------------
Test.@testset "Float32 propagation" begin
    N = 8
    Ls = (Float32(2π), Float32(2π))
    ks = FIT.Utils.wavenumber_grid((N, N), Ls)
    Test.@test eltype(ks[1]) == Float32
    Test.@test eltype(ks[2]) == Float32
    k_mag = FIT.Utils.wavenumber_magnitude_grid(ks)
    Test.@test eltype(k_mag) == Float32
    b     = FIT.Types.LinearBinning(Float32(2π) / N)
    edges = FIT.ShellBinning.shell_edges(b, maximum(k_mag))
    Test.@test eltype(edges) == Float32
    centers = FIT.ShellBinning.shell_centers(b, maximum(k_mag))
    Test.@test eltype(centers) == Float32

    û = zeros(ComplexF32, N, N, 2)
    result = FIT.SpectralFlux.calculate_spectral_flux(û, ks;
        binning=b, dealiasing = FIT.Types.NoDealiasing(), spectral=SpectralBackends.DirectSumSpectralBackend())
    Test.@test result isa FIT.Types.SpectralFluxResult
    Test.@test eltype(result.k_shells) == Float32
    Test.@test eltype(result.transfer_spectrum) == Float32

    # New diagnostics also preserve Float32 throughout.
    θ̂ = zeros(ComplexF32, N, N)
    rθ = FIT.calculate_scalar_flux(û, θ̂, ks; binning=b, dealiasing = FIT.Types.NoDealiasing(), spectral=SpectralBackends.DirectSumSpectralBackend())
    Test.@test eltype(rθ.transfer_spectrum) == Float32
end

# -----------------------------------------------------------------------
Test.@testset "TriadicOrthogonalDecomposition" begin
    # Set seed
    Random.seed!(123)

    # 1. Parameter parsing and validation
    # Parse parameters with default values
    nt = 100
    nx = 5
    win_vec, weight_vec, noverlap, dt, nDFT, nBlks =
        FIT.TriadicOrthogonalDecomposition.parse_parameters(nt, nx)
    Test.@test length(win_vec) == nDFT
    Test.@test length(weight_vec) == nx
    Test.@test nBlks >= 2

    # Error cases for parameters
    Test.@test_throws ArgumentError FIT.TriadicOrthogonalDecomposition.parse_parameters(nt, nx; window=3)  # nDFT < 4
    Test.@test_throws ArgumentError FIT.TriadicOrthogonalDecomposition.parse_parameters(nt, nx; window=zeros(3)) # nDFT < 4
    Test.@test_throws ArgumentError FIT.TriadicOrthogonalDecomposition.parse_parameters(nt, nx; noverlap=256) # noverlap >= nDFT

    # 2. SVD helper functions
    # Sirovich SVD
    M = randn(4, 10)
    U, s, V = FIT.TriadicOrthogonalDecomposition.sirovich_svd(M)
    Test.@test length(s) == 4
    Test.@test size(U) == (4, 4)
    Test.@test size(V) == (10, 4)
    Test.@test all(s .>= 0)
    # Verify reconstruction: M' * U ≈ V * diag(s)
    # In sirovich_svd: V = M' * U * diag(1/s) -> M' * U = V * diag(s)
    Test.@test isapprox(M' * U, V * LinearAlgebra.Diagonal(s); atol=1e-12)

    # Low-rank SVD
    X_lr = randn(4, 10)
    Q3 = randn(5, 4)
    U_lr, s_lr, V_lr = FIT.TriadicOrthogonalDecomposition.lowrank_svd(X_lr, Q3)
    Test.@test size(U_lr) == (5, 4)
    Test.@test size(V_lr) == (10, 4)

    # 3. Known-triad interaction / basic run
    # We generate a signal with 2 frequencies f1 and f2.
    dt_sig = 0.1
    t = collect(0:255) .* dt_sig
    nt_sig = length(t)
    nx_sig = 2
    f1, f2 = 2.0, 3.0
    X = zeros(nt_sig, 1, nx_sig)
    for ix in 1:nx_sig
        X[:, 1, ix] = sin.(2π * f1 .* t) .+ cos.(2π * f2 .* t)
    end

    # Run with default settings
    res = FIT.TriadicOrthogonalDecomposition.triadic_orthogonal_decomposition(X; dt=dt_sig, isreal_data=true)
    Test.@test res isa FIT.Types.TriadicOrthogonalDecompositionResult
    Test.@test res.frequencies isa AbstractVector
    Test.@test all(res.mode_bispectrum .>= 0.0 .|| isnan.(res.mode_bispectrum))
    Test.@test all(res.modal_energy_budget .>= 0.0 .|| res.modal_energy_budget .<= 0.0 .|| isnan.(res.modal_energy_budget))

    # Check default dispatch via calculate_energy_transfer
    method = FIT.Types.TriadicOrthogonalDecompositionMethod(nfft=64, noverlap=32, nmode=2)
    res_dispatch = FIT.calculate_energy_transfer(method, X; dt=dt_sig)
    Test.@test res_dispatch isa FIT.Types.TriadicOrthogonalDecompositionResult
    Test.@test size(res_dispatch.mode_bispectrum, 3) == 2

    # 4. SpectralBackends.FFTSpectralBackend consistency
    res_serial = FIT.TriadicOrthogonalDecomposition.triadic_orthogonal_decomposition(X; dt=dt_sig, spectral=SpectralBackends.DirectSumSpectralBackend())
    res_fft = FIT.TriadicOrthogonalDecomposition.triadic_orthogonal_decomposition(X; dt=dt_sig, spectral=SpectralBackends.FFTSpectralBackend())
    Test.@test isapprox(res_serial.frequencies, res_fft.frequencies)
    Test.@test isapprox(filter(!isnan, res_serial.mode_bispectrum), filter(!isnan, res_fft.mode_bispectrum); atol=1e-12)

    # 5. ThreadedBackend — OhMyThreads is loaded so it should work
    res_threaded = FIT.TriadicOrthogonalDecomposition.triadic_orthogonal_decomposition(X; dt=dt_sig, execution=ComputationalBackends.ThreadedBackend())
    Test.@test res_threaded isa FIT.Types.TriadicOrthogonalDecompositionResult
    Test.@test isapprox(res_serial.frequencies, res_threaded.frequencies)

    # 6. Coefficients and auxiliary modes
    res_aux = FIT.TriadicOrthogonalDecomposition.triadic_orthogonal_decomposition(X; dt=dt_sig, return_coefficients=true, return_auxiliary_modes=true)
    Test.@test res_aux.expansion_coefficients isa Dict
    Test.@test res_aux.auxiliary_modes isa Dict

    # 7. Precision genericity: Float32 snapshots stay in Float32 (no ComplexF64 upcast).
    res32 = FIT.TriadicOrthogonalDecomposition.triadic_orthogonal_decomposition(Float32.(X); dt=Float32(dt_sig))
    Test.@test res32 isa FIT.Types.TriadicOrthogonalDecompositionResult
    Test.@test eltype(res32.mode_bispectrum) === Float32
    Test.@test isapprox(Float64.(res32.frequencies), res_serial.frequencies; atol=1e-4)

    # 8. In-place workspace form: bit-identical to the allocating path (same buffers/math), and a
    # repeat call on a same-shaped snapshot reuses Q_hat / DFT plan / SVD scratch / L / T_budget.
    ws = FIT.TriadicOrthogonalDecomposition.TODWorkspace(X; dt=dt_sig, spectral=SpectralBackends.FFTSpectralBackend())
    ip = FIT.TriadicOrthogonalDecomposition.triadic_orthogonal_decomposition!(ws, X)
    Test.@test ip isa FIT.Types.TriadicOrthogonalDecompositionResult
    Test.@test isequal(ip.mode_bispectrum, res_fft.mode_bispectrum)   # NaN-aware; exact
    for k in keys(res_fft.modes)
        Test.@test ip.modes[k].convective == res_fft.modes[k].convective
    end
    # (workspace-reuse allocation ratio asserted in test_allocs.jl)
    Test.@test_throws DimensionMismatch FIT.TriadicOrthogonalDecomposition.triadic_orthogonal_decomposition!(ws, X[:, :, 1:end-1])
end

# -----------------------------------------------------------------------
Test.@testset "TOD windows — Hann / Tukey generators" begin
    n = 64
    h = FIT.TriadicOrthogonalDecomposition.hann_window(n)
    Test.@test length(h) == 64
    Test.@test h[1] < 1e-12 && h[end] < 1e-12        # tapers to zero at both ends
    Test.@test 0.99 < maximum(h) <= 1.0              # ≈unity at centre (exact 1 falls between samples for even N)
    # Tukey limits: α=0 is rectangular, α=1 is Hann
    Test.@test FIT.TriadicOrthogonalDecomposition.tukey_window(n; α=0.0) ≈ ones(n)
    Test.@test isapprox(FIT.TriadicOrthogonalDecomposition.tukey_window(n; α=1.0), h; atol=1e-12)
    Test.@test_throws ArgumentError FIT.TriadicOrthogonalDecomposition.tukey_window(n; α=1.5)
    # a custom window flows through TOD
    X = zeros(128, 1, 4)
    for ix in 1:4; X[:, 1, ix] = sin.(2π * 2.0 .* (0:127) .* 0.05); end
    r = FIT.TriadicOrthogonalDecomposition.triadic_orthogonal_decomposition(X; dt=0.05, window=FIT.TriadicOrthogonalDecomposition.hann_window(64), noverlap=32)
    Test.@test r isa FIT.Types.TriadicOrthogonalDecompositionResult
end

# -----------------------------------------------------------------------
Test.@testset "TOD — detects a known quadratic triad (FFT == direct)" begin
    # Regression for a double-fftshift bug in the FFTW temporal DFT: the FFT backend was
    # shifted twice, misaligning Q_hat with the frequency axis, so genuine triads were not
    # detected. Build a real phase-locked triad f_k+f_l=f_n (daughter = sum-frequency wave
    # with phase φ_k+φ_l), random parent phases per block, and require BOTH backends to (a)
    # agree, (b) make the triad a dominant peak, and (c) score it far above an unlocked control.
    nfft = 100; nblocks = 40; nx = 12; dt = 0.1   # Δf=0.1 ⇒ f=1,2,3 land on integer bins
    fk, fl, fn = 1.0, 2.0, 3.0
    xs = range(0, 2π; length = nx)
    mk = cos.(xs); ml = cos.(2 .* xs); mn = mk .* ml
    function build_signal(locked)
        Random.seed!(20)
        nt = nfft * nblocks; X = zeros(nt, 1, nx)
        for blk in 0:nblocks-1
            φk, φl = 2π .* rand(2); φn = locked ? (φk + φl) : 2π * rand()
            for τ in 0:nfft-1
                tt = τ * dt
                X[blk*nfft + τ + 1, 1, :] .= cos(2π*fk*tt + φk).*mk .+ cos(2π*fl*tt + φl).*ml .+
                                                cos(2π*fn*tt + φn).*mn
            end
        end
        return X
    end
    Xl = build_signal(true)
    rd = FIT.TriadicOrthogonalDecomposition.triadic_orthogonal_decomposition(Xl; window=nfft, noverlap=0, nmode=1, dt=dt,
        isreal_data=true, spectral=SpectralBackends.DirectSumSpectralBackend())
    rf = FIT.TriadicOrthogonalDecomposition.triadic_orthogonal_decomposition(Xl; window=nfft, noverlap=0, nmode=1, dt=dt,
        isreal_data=true, spectral=SpectralBackends.FFTSpectralBackend())
    f = rd.frequencies
    Ld = copy(rd.mode_bispectrum[:, :, 1]); Ld[isnan.(Ld)] .= 0
    Lf = copy(rf.mode_bispectrum[:, :, 1]); Lf[isnan.(Lf)] .= 0
    li = argmin(abs.(f .- fl)); ni = argmin(abs.(f .- fn))
    # (a) the FFT bug is fixed: backends agree (this previously diverged badly)
    Test.@test isapprox(filter(!isnan, rd.mode_bispectrum), filter(!isnan, rf.mode_bispectrum); atol = 1e-10)
    # (b) the genuine triad (f_l=2, f_n=3) is a dominant peak (not buried)
    Test.@test Lf[li, ni] >= 0.5 * maximum(Lf)
    # (c) phase-locking matters: unlocked daughter scores far lower at the same cell
    Xu = build_signal(false)
    ru = FIT.TriadicOrthogonalDecomposition.triadic_orthogonal_decomposition(Xu; window=nfft, noverlap=0, nmode=1, dt=dt,
        isreal_data=true, spectral=SpectralBackends.FFTSpectralBackend())
    Lu = copy(ru.mode_bispectrum[:, :, 1]); Lu[isnan.(Lu)] .= 0
    Test.@test Lu[li, ni] < 0.3 * Lf[li, ni]
end

# -----------------------------------------------------------------------
Test.@testset "Field Decomposition (Helmholtz / Partial Flux)" begin
    # 1. Spectral flux decomposition test
    N = 8; L = 2π
    ks = FIT.Utils.wavenumber_grid((N, N), (L, L))
    û = zeros(ComplexF64, N, N, 2)
    û[2, 1, 1] = 0.5; û[N, 1, 1] = 0.5   # k=(1,0) in u
    û[1, 2, 2] = 0.5; û[1, N, 2] = 0.5   # k=(0,1) in v

    res_none = FIT.SpectralFlux.calculate_spectral_flux(û, ks; decomposition=FIT.Types.NoDecomposition(), dealiasing = FIT.Types.NoDealiasing())
    res_helm = FIT.SpectralFlux.calculate_spectral_flux(û, ks; decomposition=FIT.Types.HelmholtzDecomposition(), dealiasing = FIT.Types.NoDealiasing())
    res_rot  = FIT.SpectralFlux.calculate_spectral_flux(û, ks; decomposition=FIT.Types.RotationalDecomposition(), dealiasing = FIT.Types.NoDealiasing())
    res_div  = FIT.SpectralFlux.calculate_spectral_flux(û, ks; decomposition=FIT.Types.DivergentDecomposition(), dealiasing = FIT.Types.NoDealiasing())

    Test.@test res_none isa FIT.Types.SpectralFluxResult
    Test.@test res_helm isa NamedTuple
    Test.@test haskey(res_helm, :rotational) && haskey(res_helm, :divergent)
    Test.@test res_rot isa FIT.Types.SpectralFluxResult
    Test.@test res_div isa FIT.Types.SpectralFluxResult

    # For these divergence-free/rotational modes, verify consistency:
    # T_none ≈ T_rot + T_div
    Test.@test isapprox(res_none.transfer_spectrum, res_rot.transfer_spectrum + res_div.transfer_spectrum; atol=1e-12)

    # 2. Coarse-graining flux decomposition test
    x = range(0, L; length=N+1)[1:N]
    y = range(0, L; length=N+1)[1:N]
    u = [cos(x) for x in x, y in y]
    v = [sin(y) for x in x, y in y]

    cg_none = FIT.CoarseGrainingFlux.calculate_coarse_graining_flux((u, v), (x, y), 1.0, FIT.Types.GaussianFilter(); decomposition=FIT.Types.NoDecomposition())
    cg_helm = FIT.CoarseGrainingFlux.calculate_coarse_graining_flux((u, v), (x, y), 1.0, FIT.Types.GaussianFilter(); decomposition=FIT.Types.HelmholtzDecomposition())
    cg_rot  = FIT.CoarseGrainingFlux.calculate_coarse_graining_flux((u, v), (x, y), 1.0, FIT.Types.GaussianFilter(); decomposition=FIT.Types.RotationalDecomposition())
    cg_div  = FIT.CoarseGrainingFlux.calculate_coarse_graining_flux((u, v), (x, y), 1.0, FIT.Types.GaussianFilter(); decomposition=FIT.Types.DivergentDecomposition())

    Test.@test cg_none isa FIT.Types.CoarseGrainingFluxResult
    Test.@test cg_helm isa NamedTuple
    Test.@test haskey(cg_helm, :rotational) && haskey(cg_helm, :divergent)
    Test.@test cg_rot isa FIT.Types.CoarseGrainingFluxResult
    Test.@test cg_div isa FIT.Types.CoarseGrainingFluxResult

    # 3. 3D coarse-graining with Helmholtz decomposition (physical-space decompose + 3D CGEF flux).
    M = 8; x3 = collect(range(0, L; length=M+1)[1:M]); y3 = copy(x3); z3 = copy(x3)
    u3 = [cos(xi) * sin(zi) for xi in x3, yi in y3, zi in z3]
    v3 = [sin(yi)           for xi in x3, yi in y3, zi in z3]
    w3 = [cos(zi)           for xi in x3, yi in y3, zi in z3]
    cg3_none = FIT.CoarseGrainingFlux.calculate_coarse_graining_flux((u3, v3, w3), (x3, y3, z3), 1.0, FIT.Types.GaussianFilter(); decomposition=FIT.Types.NoDecomposition())
    cg3_helm = FIT.CoarseGrainingFlux.calculate_coarse_graining_flux((u3, v3, w3), (x3, y3, z3), 1.0, FIT.Types.GaussianFilter(); decomposition=FIT.Types.HelmholtzDecomposition())
    Test.@test cg3_none isa FIT.Types.CoarseGrainingFluxResult && size(cg3_none.flux_field) == (M, M, M)
    Test.@test cg3_helm isa NamedTuple && haskey(cg3_helm, :rotational) && haskey(cg3_helm, :divergent)
    Test.@test size(cg3_helm.rotational.flux_field) == (M, M, M)
    Test.@test all(isfinite, cg3_helm.divergent.flux_field)
end

# -----------------------------------------------------------------------
Test.@testset "Parallel Backends Parity (Threaded / Distributed)" begin
    # Add workers if not present
    if Distributed.nprocs() == 1
        Distributed.addprocs(2)
    end
    # Load the package and extensions on all workers
    Distributed.@everywhere using FlowInvariantTransfer: FlowInvariantTransfer as FIT
    Distributed.@everywhere using SharedArrays

    # Create sample data
    Random.seed!(42)
    N = 8; L = 2π
    ks = FIT.Utils.wavenumber_grid((N, N), (L, L))
    û = zeros(ComplexF64, N, N, 2)
    û[2, 1, 1] = 0.5; û[N, 1, 1] = 0.5
    û[1, 2, 2] = 0.5; û[1, N, 2] = 0.5

    # 1. Shell-to-Shell Transfer Parity
    b = FIT.Types.LinearBinning(2π / L)
    res_serial = FIT.ShellToShellTransfer.calculate_shell_to_shell_transfer(û, ks; binning=b, dealiasing = FIT.Types.OrszagTwoThirds(), verify_antisymmetry=true, execution=ComputationalBackends.SerialBackend())
    res_thread = FIT.ShellToShellTransfer.calculate_shell_to_shell_transfer(û, ks; binning=b, dealiasing = FIT.Types.OrszagTwoThirds(), verify_antisymmetry=true, execution=ComputationalBackends.ThreadedBackend())
    
    # For DistributedBackend, we convert velocity_hat to a SharedArray so workers can read it efficiently
    s_û = SharedArrays.SharedArray(û)
    res_dist = FIT.ShellToShellTransfer.calculate_shell_to_shell_transfer(s_û, ks; binning=b, dealiasing = FIT.Types.OrszagTwoThirds(), verify_antisymmetry=true, execution=ComputationalBackends.DistributedBackend())

    Test.@test isapprox(res_serial.transfer_matrix, res_thread.transfer_matrix; atol=1e-12)
    Test.@test isapprox(res_serial.transfer_matrix, res_dist.transfer_matrix; atol=1e-12)
    Test.@test isapprox(res_serial.net_transfer, res_thread.net_transfer; atol=1e-12)
    Test.@test isapprox(res_serial.net_transfer, res_dist.net_transfer; atol=1e-12)

    # #4 — parametric DistributedBackend{Inner} + local_backend accessor.
    Test.@test ComputationalBackends.DistributedBackend() === ComputationalBackends.DistributedBackend(ComputationalBackends.SerialBackend())
    Test.@test ComputationalBackends.local_backend(ComputationalBackends.DistributedBackend(ComputationalBackends.ThreadedBackend())) === ComputationalBackends.ThreadedBackend()
    Test.@test ComputationalBackends.local_backend(ComputationalBackends.SerialBackend()) === ComputationalBackends.SerialBackend()   # identity for non-distributed
    # Hybrid distributed+threaded per-worker path must match serial (workers here are single-threaded,
    # so this exercises the ThreadedBackend inner branch of compute_mediator_transfer_column).
    res_hybrid = FIT.ShellToShellTransfer.calculate_shell_to_shell_transfer(s_û, ks; binning=b, dealiasing = FIT.Types.OrszagTwoThirds(), verify_antisymmetry=true, execution=ComputationalBackends.DistributedBackend(ComputationalBackends.ThreadedBackend()))
    Test.@test isapprox(res_serial.transfer_matrix, res_hybrid.transfer_matrix; atol=1e-12)
    Test.@test isapprox(res_serial.net_transfer, res_hybrid.net_transfer; atol=1e-12)

    # 2. Triadic Orthogonal Decomposition parity — triads distributed over workers, each running the
    # SAME allocation-reusing SVD as the serial loop → bit-identical L / T_budget / modes / coeffs.
    # The temporal DFT (Q_hat) is master-side, so workers need only FlowInvariantTransfer (no FFTW).
    Random.seed!(123)
    Xtod = randn(64, 1, 4)
    tod_serial = FIT.TriadicOrthogonalDecomposition.triadic_orthogonal_decomposition(Xtod; dt = 0.05, spectral = SpectralBackends.DirectSumSpectralBackend(), execution = ComputationalBackends.SerialBackend(), return_coefficients = true, return_auxiliary_modes = true)
    tod_dist   = FIT.TriadicOrthogonalDecomposition.triadic_orthogonal_decomposition(Xtod; dt = 0.05, spectral = SpectralBackends.DirectSumSpectralBackend(), execution = ComputationalBackends.DistributedBackend(), return_coefficients = true, return_auxiliary_modes = true)
    Test.@test isequal(tod_serial.mode_bispectrum, tod_dist.mode_bispectrum)
    Test.@test isequal(tod_serial.modal_energy_budget, tod_dist.modal_energy_budget)
    Test.@test length(tod_serial.modes) == length(tod_dist.modes)
    let k = first(keys(tod_serial.modes))
        Test.@test tod_serial.modes[k].convective == tod_dist.modes[k].convective
        Test.@test tod_serial.expansion_coefficients[k].recipient == tod_dist.expansion_coefficients[k].recipient
        Test.@test tod_serial.auxiliary_modes[k].donor == tod_dist.auxiliary_modes[k].donor
    end

    # 3. Distributed parity for the loop diagnostics + compressible (all vs serial). DirectSum so
    # workers need only FlowInvariantTransfer (decomposition / density / pressure are master-side or
    # shipped as plain arrays). Pins that DistributedBackend genuinely matches serial for
    # mode-to-mode / band-to-band / partial fluxes / compressible, including the hybrid
    # DistributedBackend(ThreadedBackend()).
    spD = SpectralBackends.DirectSumSpectralBackend()
    m_ser = FIT.calculate_mode_to_mode_transfer(û, ks; spectral=spD, execution=ComputationalBackends.SerialBackend(), force=true)
    m_dst = FIT.calculate_mode_to_mode_transfer(û, ks; spectral=spD, execution=ComputationalBackends.DistributedBackend(), force=true)
    m_hyb = FIT.calculate_mode_to_mode_transfer(û, ks; spectral=spD, execution=ComputationalBackends.DistributedBackend(ComputationalBackends.ThreadedBackend()), force=true)
    Test.@test isapprox(m_ser.transfer, m_dst.transfer; atol=1e-12)
    Test.@test isapprox(m_ser.net_transfer, m_dst.net_transfer; atol=1e-12)
    Test.@test isapprox(m_ser.transfer, m_hyb.transfer; atol=1e-12)

    bands = FIT.Types.SmoothBands([2.0, 4.0, 6.0])
    bb_ser = FIT.BandTransfer.calculate_band_to_band_transfer(û, ks; bands=bands, spectral=spD, execution=ComputationalBackends.SerialBackend())
    bb_dst = FIT.BandTransfer.calculate_band_to_band_transfer(û, ks; bands=bands, spectral=spD, execution=ComputationalBackends.DistributedBackend())
    bb_hyb = FIT.BandTransfer.calculate_band_to_band_transfer(û, ks; bands=bands, spectral=spD, execution=ComputationalBackends.DistributedBackend(ComputationalBackends.ThreadedBackend()))
    Test.@test isapprox(bb_ser.transfer_matrix, bb_dst.transfer_matrix; atol=1e-12)
    Test.@test isapprox(bb_ser.transfer_matrix, bb_hyb.transfer_matrix; atol=1e-12)

    pf_ser = FIT.calculate_partial_fluxes(û, ks; decomposition=FIT.Types.HelmholtzDecomposition(), spectral=spD, execution=ComputationalBackends.SerialBackend())
    pf_dst = FIT.calculate_partial_fluxes(û, ks; decomposition=FIT.Types.HelmholtzDecomposition(), spectral=spD, execution=ComputationalBackends.DistributedBackend())
    Test.@test Set(keys(pf_ser.channels)) == Set(keys(pf_dst.channels))
    Test.@test isapprox(pf_ser.total.transfer_spectrum, pf_dst.total.transfer_spectrum; atol=1e-12)

    ρ̂c = randn(Random.MersenneTwister(7), ComplexF64, N, N)
    p̂c = randn(Random.MersenneTwister(8), ComplexF64, N, N)
    c_ser = FIT.Compressible.calculate_compressible_flux(û, ρ̂c, ks; spectral=spD, execution=ComputationalBackends.SerialBackend(), pressure_hat=p̂c, decompose=true)
    c_dst = FIT.Compressible.calculate_compressible_flux(û, ρ̂c, ks; spectral=spD, execution=ComputationalBackends.DistributedBackend(), pressure_hat=p̂c, decompose=true)
    c_hyb = FIT.Compressible.calculate_compressible_flux(û, ρ̂c, ks; spectral=spD, execution=ComputationalBackends.DistributedBackend(ComputationalBackends.ThreadedBackend()), pressure_hat=p̂c, decompose=true)
    Test.@test isapprox(c_ser.transfer_spectrum, c_dst.transfer_spectrum; atol=1e-12)
    Test.@test isapprox(c_ser.flux, c_dst.flux; atol=1e-12)
    Test.@test isapprox(c_ser.channels.rotational, c_dst.channels.rotational; atol=1e-12)
    Test.@test isapprox(c_ser.channels.compressive, c_dst.channels.compressive; atol=1e-12)
    Test.@test isapprox(c_ser.pressure_dilatation.compressive, c_dst.pressure_dilatation.compressive; atol=1e-12)
    Test.@test isapprox(c_ser.transfer_spectrum, c_hyb.transfer_spectrum; atol=1e-12)
end

# -----------------------------------------------------------------------
# Batch axis: the *_batch entries process many snapshots that share one grid, building the
# snapshot-independent structure (shells / bands / FFT plans) ONCE and reusing one workspace per worker.
# Gate the contract directly: batch(Serial) is bit-identical to the per-snapshot single call — correct
# VALUES and correct input ORDER; Threaded / Distributed match the serial batch; an unsupported execution
# backend refuses (never silently runs serial); an empty batch returns empty. The 2 workers added by the
# parity testset above persist, but carry only `using FIT` — the distributed path needs FFTW/CGEF too.
Test.@testset "Batch axis — spectral / shell / band / coarse-graining over snapshots" begin
    Distributed.nprocs() == 1 && Distributed.addprocs(2)
    Distributed.@everywhere using FlowInvariantTransfer: FlowInvariantTransfer as FIT
    Distributed.@everywhere using FFTW
    Distributed.@everywhere using CoarseGrainingEnergyFluxes

    Random.seed!(2024)
    N = 12; L = 2π; nsnap = 4
    ks = FIT.Utils.wavenumber_grid((N, N), (L, L))
    kx = [ks[1][i] for i in 1:N, j in 1:N]; ky = [ks[2][j] for i in 1:N, j in 1:N]
    b     = FIT.Types.LinearBinning(2π / L)
    bands = FIT.Types.SmoothBands([1.0, 2.0, 3.0]; logwidth = 0.5)
    FTB = SpectralBackends.FFTSpectralBackend()
    Ser = ComputationalBackends.SerialBackend()
    Thr = ComputationalBackends.ThreadedBackend()
    Dst = ComputationalBackends.DistributedBackend()
    GPU = ComputationalBackends.GPUBackend(KA.CPU())               # KA.CPU() → device path on host: CPU-parity check (CG has no device path → refuses)
    vhats = map(1:nsnap) do _
        ψh = FFTW.fft(randn(N, N)) ./ N^2
        cat(im .* ky .* ψh, -im .* kx .* ψh; dims = 3)            # divergence-free snapshot
    end
    xs = collect(range(0, L; length = N + 1)[1:N]); ys = copy(xs)
    ufields = map(_ -> (randn(N, N), randn(N, N)), 1:nsnap)        # physical fields for coarse-graining
    filt = FIT.Types.GaussianFilter(); ℓ = 0.5

    # --- spectral flux ---
    sf1 = [FIT.SpectralFlux.calculate_spectral_flux(vh, ks; binning = b, spectral = FTB) for vh in vhats]
    sfS = FIT.SpectralFlux.calculate_spectral_flux_batch(vhats, ks; binning = b, spectral = FTB, execution = Ser)
    sfT = FIT.SpectralFlux.calculate_spectral_flux_batch(vhats, ks; binning = b, spectral = FTB, execution = Thr)
    sfD = FIT.SpectralFlux.calculate_spectral_flux_batch(vhats, ks; binning = b, spectral = FTB, execution = Dst)
    sfG = FIT.SpectralFlux.calculate_spectral_flux_batch(vhats, ks; binning = b, spectral = FTB, execution = GPU)
    Test.@test length(sfS) == nsnap
    Test.@test maximum(abs, sf1[1].transfer_spectrum) > 0                       # genuine nonzero transfer
    for i in 1:nsnap
        Test.@test sfS[i].transfer_spectrum == sf1[i].transfer_spectrum         # serial batch == single, in order
        Test.@test sfS[i].flux == sf1[i].flux
        Test.@test isapprox(sfT[i].transfer_spectrum, sfS[i].transfer_spectrum; atol = 1e-12)
        Test.@test isapprox(sfD[i].transfer_spectrum, sfS[i].transfer_spectrum; atol = 1e-12)
        Test.@test isapprox(sfG[i].transfer_spectrum, sfS[i].transfer_spectrum; atol = 1e-12)   # GPU device path (KA.CPU)
    end
    # GPU batch requires the FFT backend (DirectSum is a host-only reference).
    Test.@test_throws ArgumentError FIT.SpectralFlux.calculate_spectral_flux_batch(vhats, ks; binning = b, spectral = SpectralBackends.DirectSumSpectralBackend(), execution = GPU)
    Test.@test isempty(FIT.SpectralFlux.calculate_spectral_flux_batch(typeof(vhats[1])[], ks; binning = b, spectral = FTB))

    # --- shell to shell ---
    ss1 = [FIT.ShellToShellTransfer.calculate_shell_to_shell_transfer(vh, ks; binning = b, spectral = FTB) for vh in vhats]
    ssS = FIT.ShellToShellTransfer.calculate_shell_to_shell_transfer_batch(vhats, ks; binning = b, spectral = FTB, execution = Ser)
    ssT = FIT.ShellToShellTransfer.calculate_shell_to_shell_transfer_batch(vhats, ks; binning = b, spectral = FTB, execution = Thr)
    ssD = FIT.ShellToShellTransfer.calculate_shell_to_shell_transfer_batch(vhats, ks; binning = b, spectral = FTB, execution = Dst)
    ssG = FIT.ShellToShellTransfer.calculate_shell_to_shell_transfer_batch(vhats, ks; binning = b, spectral = FTB, execution = GPU)
    for i in 1:nsnap
        Test.@test ssS[i].transfer_matrix == ss1[i].transfer_matrix
        Test.@test ssS[i].net_transfer == ss1[i].net_transfer
        Test.@test isapprox(ssT[i].transfer_matrix, ssS[i].transfer_matrix; atol = 1e-12)
        Test.@test isapprox(ssD[i].transfer_matrix, ssS[i].transfer_matrix; atol = 1e-12)
        Test.@test isapprox(ssG[i].transfer_matrix, ssS[i].transfer_matrix; atol = 1e-12)   # GPU device path (KA.CPU)
    end

    # --- band to band ---
    bb1 = [FIT.BandTransfer.calculate_band_to_band_transfer(vh, ks; bands = bands, spectral = FTB) for vh in vhats]
    bbS = FIT.BandTransfer.calculate_band_to_band_transfer_batch(vhats, ks; bands = bands, spectral = FTB, execution = Ser)
    bbT = FIT.BandTransfer.calculate_band_to_band_transfer_batch(vhats, ks; bands = bands, spectral = FTB, execution = Thr)
    bbD = FIT.BandTransfer.calculate_band_to_band_transfer_batch(vhats, ks; bands = bands, spectral = FTB, execution = Dst)
    bbG = FIT.BandTransfer.calculate_band_to_band_transfer_batch(vhats, ks; bands = bands, spectral = FTB, execution = GPU)
    for i in 1:nsnap
        Test.@test bbS[i].transfer_matrix == bb1[i].transfer_matrix
        Test.@test bbS[i].net_transfer == bb1[i].net_transfer
        Test.@test isapprox(bbT[i].transfer_matrix, bbS[i].transfer_matrix; atol = 1e-12)
        Test.@test isapprox(bbD[i].transfer_matrix, bbS[i].transfer_matrix; atol = 1e-12)
        Test.@test isapprox(bbG[i].transfer_matrix, bbS[i].transfer_matrix; atol = 1e-12)   # GPU device path (KA.CPU)
    end

    # --- coarse-graining (physical fields via CGEF) ---
    cg1 = [FIT.CoarseGrainingFlux.calculate_coarse_graining_flux(uv, (xs, ys), ℓ, filt) for uv in ufields]
    cgS = FIT.CoarseGrainingFlux.calculate_coarse_graining_flux_batch(ufields, (xs, ys), ℓ, filt; execution = Ser)
    cgT = FIT.CoarseGrainingFlux.calculate_coarse_graining_flux_batch(ufields, (xs, ys), ℓ, filt; execution = Thr)
    cgD = FIT.CoarseGrainingFlux.calculate_coarse_graining_flux_batch(ufields, (xs, ys), ℓ, filt; execution = Dst)
    for i in 1:nsnap
        Test.@test cgS[i].flux_field == cg1[i].flux_field
        Test.@test cgS[i].mean_flux == cg1[i].mean_flux
        Test.@test isapprox(cgT[i].flux_field, cgS[i].flux_field; atol = 1e-12)
        Test.@test isapprox(cgD[i].flux_field, cgS[i].flux_field; atol = 1e-12)
    end
    Test.@test_throws ArgumentError FIT.CoarseGrainingFlux.calculate_coarse_graining_flux_batch(ufields, (xs, ys), ℓ, filt; execution = GPU)
end

# -----------------------------------------------------------------------
# W10 — FFTW *intra-transform* threads (orthogonal to ThreadedBackend, which
# parallelises the outer shell loop). FFTW's own multi-threading is a global
# setting; turning it on must not change any result. Validated single-machine.
Test.@testset "FFTW intra-transform threads — correctness invariance" begin
    Random.seed!(7)
    N = 24; L = 2π
    ks = FIT.Utils.wavenumber_grid((N, N), (L, L))
    ψ  = randn(N, N); ψh = FFTW.fft(ψ) ./ N^2
    kx = [ks[1][i] for i in 1:N, j in 1:N]; ky = [ks[2][j] for i in 1:N, j in 1:N]
    û  = cat(im .* ky .* ψh, -im .* kx .* ψh; dims = 3)   # incompressible 2D field
    b  = FIT.Types.LinearBinning(2π / L)

    nthr0 = FFTW.get_num_threads()
    try
        FFTW.set_num_threads(1)
        flux1  = FIT.SpectralFlux.calculate_spectral_flux(û, ks; binning=b, spectral=SpectralBackends.FFTSpectralBackend())
        s2s1   = FIT.ShellToShellTransfer.calculate_shell_to_shell_transfer(û, ks; binning=b, spectral=SpectralBackends.FFTSpectralBackend())
        FFTW.set_num_threads(4)
        flux4  = FIT.SpectralFlux.calculate_spectral_flux(û, ks; binning=b, spectral=SpectralBackends.FFTSpectralBackend())
        s2s4   = FIT.ShellToShellTransfer.calculate_shell_to_shell_transfer(û, ks; binning=b, spectral=SpectralBackends.FFTSpectralBackend())

        Test.@test isapprox(flux1.flux, flux4.flux; atol=1e-12, rtol=1e-10)
        Test.@test isapprox(flux1.transfer_spectrum, flux4.transfer_spectrum; atol=1e-12, rtol=1e-10)
        Test.@test isapprox(s2s1.transfer_matrix, s2s4.transfer_matrix; atol=1e-12, rtol=1e-10)
    finally
        FFTW.set_num_threads(nthr0)   # restore global FFTW thread count
    end
end

# -----------------------------------------------------------------------
Test.@testset "ModeToMode — invariant/dimension guards" begin
    L = 2π
    # 2D field + Helicity() must error (Helicity is 3D-only); routed via transfer_density.
    ks2 = FIT.Utils.wavenumber_grid((4, 4), (L, L))
    û2  = zeros(ComplexF64, 4, 4, 2); û2[2, 1, 1] = 0.5; û2[1, 2, 2] = 0.5
    Test.@test_throws ArgumentError FIT.calculate_mode_to_mode_transfer(û2, ks2; invariant=FIT.Types.Helicity())
    # 3D field + Enstrophy() now works (vector-vorticity transfer, routed).
    ks3 = FIT.Utils.wavenumber_grid((4, 4, 4), (L, L, L))
    û3  = zeros(ComplexF64, 4, 4, 4, 3); û3[2, 1, 1, 1] = 0.5
    Test.@test FIT.calculate_mode_to_mode_transfer(û3, ks3; invariant=FIT.Types.Enstrophy()) isa FIT.Types.ModeToModeTriadResult
    # KineticEnergy works in both dimensionalities.
    Test.@test FIT.calculate_mode_to_mode_transfer(û2, ks2; invariant=FIT.Types.KineticEnergy()) isa FIT.Types.ModeToModeTriadResult
    Test.@test FIT.calculate_mode_to_mode_transfer(û3, ks3; invariant=FIT.Types.KineticEnergy()) isa FIT.Types.ModeToModeTriadResult
end

# -----------------------------------------------------------------------
Test.@testset "ModeToMode — resolved S(k|p): antisym, conserves, reduces to spectral/shell-to-shell" begin
    # mode-to-mode now owns the fully-resolved S(k|p) (built from the validated nonlinear
    # term), which must be antisymmetric, conserve, and reduce to the coarser diagnostics.
    N = 12; L = 2π
    Random.seed!(21)
    ψ  = randn(N, N); ψh = FFTW.fft(ψ) ./ N^2
    ks = FIT.Utils.wavenumber_grid((N, N), (L, L))
    kx = [ks[1][i] for i in 1:N, j in 1:N]; ky = [ks[2][j] for i in 1:N, j in 1:N]
    û  = cat(im .* ky .* ψh, -im .* kx .* ψh; dims = 3)
    m2m = FIT.calculate_mode_to_mode_transfer(û, ks; dealiasing = FIT.Types.OrszagTwoThirds(), spectral = SpectralBackends.FFTSpectralBackend())
    S   = m2m.transfer                              # shape (N,N,N,N): S[k..., p...]
    nrm = sqrt(sum(abs2, S)); Test.@test nrm > 0    # non-degenerate
    asym = 0.0
    for k in CartesianIndices((N, N)), p in CartesianIndices((N, N))
        asym = max(asym, abs(S[k, p] + S[p, k]))
    end
    Test.@test asym < 1e-10 * nrm                   # antisymmetric S(k|p) = −S(p|k)
    Test.@test abs(sum(S)) < 1e-10 * nrm            # conserves Σ_kΣ_p S = 0

    b = FIT.Types.LinearBinning(2π/L)
    sf = FIT.SpectralFlux.calculate_spectral_flux(û, ks; binning = b, dealiasing = FIT.Types.OrszagTwoThirds(), spectral = SpectralBackends.FFTSpectralBackend())
    ss = FIT.ShellToShellTransfer.calculate_shell_to_shell_transfer(û, ks; binning = b, dealiasing = FIT.Types.OrszagTwoThirds(),
        verify_antisymmetry = false, spectral = SpectralBackends.FFTSpectralBackend())
    kmag  = FIT.Utils.wavenumber_magnitude_grid(ks)
    edges = FIT.ShellBinning.shell_edges(b, maximum(kmag))
    sidx  = FIT.ShellBinning.assign_shells(kmag, edges)
    # net (= Σ_p S) shell-summed == spectral transfer T(k)
    netshell = zeros(length(edges) - 1)
    for I in CartesianIndices((N, N)); n = sidx[I]; n == 0 && continue; netshell[n] += m2m.net_transfer[I]; end
    Test.@test isapprox(netshell, sf.transfer_spectrum; atol = 1e-9 * sqrt(sum(abs2, sf.transfer_spectrum)))
    # shell-reduction of S(k|p) == shell-to-shell matrix T(n,m)
    N_sh = size(ss.transfer_matrix, 1)
    TKQ  = zeros(N_sh, N_sh)
    for k in CartesianIndices((N, N)), p in CartesianIndices((N, N))
        n = sidx[k]; m = sidx[p]
        (n == 0 || m == 0) && continue
        TKQ[n, m] += S[k, p]
    end
    Test.@test isapprox(TKQ, ss.transfer_matrix; atol = 1e-9 * sqrt(sum(abs2, ss.transfer_matrix)))
end

# -----------------------------------------------------------------------
# GPU kernels validated on the KernelAbstractions CPU backend (GPUBackend(KA.CPU())).
# This exercises the exact device kernels + dispatch used on CUDA/ROC/Metal — the only
# part that needs real hardware is the on-device FFT, not the transfer-density/reduction
# logic asserted here. Results must match the serial reference to machine precision.
Test.@testset "GPU kernels via KA CPU backend" begin
    L = 2π
    # KE (2D)
    N = 16
    ks = FIT.Utils.wavenumber_grid((N, N), (L, L))
    kx = [ks[1][i] for i in 1:N, j in 1:N]; ky = [ks[2][j] for i in 1:N, j in 1:N]
    ψh = FFTW.fft(randn(Random.MersenneTwister(5), N, N)) ./ N^2
    û  = cat(im .* ky .* ψh, -im .* kx .* ψh; dims = 3)
    b  = FIT.Types.LinearBinning(2π / L)
    ref = FIT.ShellToShellTransfer.calculate_shell_to_shell_transfer(û, ks; binning=b, spectral=SpectralBackends.FFTSpectralBackend(), execution=ComputationalBackends.SerialBackend())
    ka  = FIT.ShellToShellTransfer.calculate_shell_to_shell_transfer(û, ks; binning=b, spectral=SpectralBackends.FFTSpectralBackend(), execution=ComputationalBackends.GPUBackend(KA.CPU()))
    Test.@test isapprox(ka.transfer_matrix, ref.transfer_matrix; atol=1e-12 * (maximum(abs, ref.transfer_matrix)+eps()))
    Test.@test isapprox(ka.net_transfer, ref.net_transfer; atol=1e-12 * (maximum(abs, ref.net_transfer)+eps()))

    # Helicity (3D) and Enstrophy (2D + 3D) device kernels. Use a REAL divergence-free field (curl of a
    # random vector potential) so the helicity transfer is genuine and well-conditioned — a random
    # complex field gives near-zero, heavily-cancelling transfer that no tolerance can match.
    ks3 = FIT.Utils.wavenumber_grid((8, 8, 8), (L, L, L))
    kx3 = [ks3[1][i] for i in 1:8, j in 1:8, k in 1:8]
    ky3 = [ks3[2][j] for i in 1:8, j in 1:8, k in 1:8]
    kz3 = [ks3[3][k] for i in 1:8, j in 1:8, k in 1:8]
    Âx = FFTW.fft(randn(Random.MersenneTwister(7), 8, 8, 8)) ./ 8^3
    Ây = FFTW.fft(randn(Random.MersenneTwister(8), 8, 8, 8)) ./ 8^3
    Âz = FFTW.fft(randn(Random.MersenneTwister(9), 8, 8, 8)) ./ 8^3
    û3 = cat(im .* (ky3 .* Âz .- kz3 .* Ây),
                im .* (kz3 .* Âx .- kx3 .* Âz),
                im .* (kx3 .* Ây .- ky3 .* Âx); dims = 4)   # u = ∇×A: real, divergence-free, helical
    rh = FIT.ShellToShellTransfer.calculate_shell_to_shell_transfer(û3, ks3; binning=FIT.Types.LinearBinning(2π/L), spectral=SpectralBackends.FFTSpectralBackend(), execution=ComputationalBackends.SerialBackend(), invariant=FIT.Types.Helicity())
    gh = FIT.ShellToShellTransfer.calculate_shell_to_shell_transfer(û3, ks3; binning=FIT.Types.LinearBinning(2π/L), spectral=SpectralBackends.FFTSpectralBackend(), execution=ComputationalBackends.GPUBackend(KA.CPU()), invariant=FIT.Types.Helicity())
    Test.@test isapprox(gh.transfer_matrix, rh.transfer_matrix; atol=1e-12 * (maximum(abs, rh.transfer_matrix)+eps()))

    re = FIT.ShellToShellTransfer.calculate_shell_to_shell_transfer(û, ks; binning=b, spectral=SpectralBackends.FFTSpectralBackend(), execution=ComputationalBackends.SerialBackend(), invariant=FIT.Types.Enstrophy())
    ge = FIT.ShellToShellTransfer.calculate_shell_to_shell_transfer(û, ks; binning=b, spectral=SpectralBackends.FFTSpectralBackend(), execution=ComputationalBackends.GPUBackend(KA.CPU()), invariant=FIT.Types.Enstrophy())
    Test.@test isapprox(ge.transfer_matrix, re.transfer_matrix; atol=1e-12 * (maximum(abs, re.transfer_matrix)+eps()))

    # Enstrophy 3D (vector vorticity + vortex stretching) device kernel — reuse the ∇×A field above.
    re3 = FIT.ShellToShellTransfer.calculate_shell_to_shell_transfer(û3, ks3; binning=FIT.Types.LinearBinning(2π/L), spectral=SpectralBackends.FFTSpectralBackend(), execution=ComputationalBackends.SerialBackend(), invariant=FIT.Types.Enstrophy())
    ge3 = FIT.ShellToShellTransfer.calculate_shell_to_shell_transfer(û3, ks3; binning=FIT.Types.LinearBinning(2π/L), spectral=SpectralBackends.FFTSpectralBackend(), execution=ComputationalBackends.GPUBackend(KA.CPU()), invariant=FIT.Types.Enstrophy())
    Test.@test isapprox(ge3.transfer_matrix, re3.transfer_matrix; atol=1e-12 * (maximum(abs, re3.transfer_matrix)+eps()))
end

# -----------------------------------------------------------------------
# Spectral flux Π(K): every execution backend (Threaded/Distributed/GPU) must reproduce the
# serial reduction to machine precision, for each invariant with a device kernel. GPU is the
# KernelAbstractions CPU backend (same device path used on CUDA/ROC/Metal); Distributed runs
# over however many workers are present (≥1). Guards against a regression (spectral
# flux was serial-only) and against the execution axis diverging from serial.
Test.@testset "Spectral flux execution backends" begin
    L = 2π
    N = 16
    ks = FIT.Utils.wavenumber_grid((N, N), (L, L))
    kx = [ks[1][i] for i in 1:N, j in 1:N]; ky = [ks[2][j] for i in 1:N, j in 1:N]
    ψh = FFTW.fft(randn(Random.MersenneTwister(11), N, N)) ./ N^2
    û  = cat(im .* ky .* ψh, -im .* kx .* ψh; dims = 3)
    b  = FIT.Types.LinearBinning(2π / L)

    ref = FIT.SpectralFlux.calculate_spectral_flux(û, ks; binning=b, spectral=SpectralBackends.FFTSpectralBackend(), execution=ComputationalBackends.SerialBackend())
    atolT = 1e-12 * (maximum(abs, ref.transfer_spectrum) + eps())
    atolΠ = 1e-12 * (maximum(abs, ref.flux) + eps())
    for exec in (ComputationalBackends.ThreadedBackend(), ComputationalBackends.DistributedBackend(), ComputationalBackends.GPUBackend(KA.CPU()))
        res = FIT.SpectralFlux.calculate_spectral_flux(û, ks; binning=b, spectral=SpectralBackends.FFTSpectralBackend(), execution=exec)
        Test.@test isapprox(res.transfer_spectrum, ref.transfer_spectrum; atol=atolT)
        Test.@test isapprox(res.flux, ref.flux; atol=atolΠ)
    end

    # 3D helicity + 2D enstrophy device kernels through the spectral-flux GPU path.
    ks3 = FIT.Utils.wavenumber_grid((8, 8, 8), (L, L, L))
    û3  = randn(Random.MersenneTwister(13), ComplexF64, 8, 8, 8, 3) .* 0.1
    rh = FIT.SpectralFlux.calculate_spectral_flux(û3, ks3; binning=FIT.Types.LinearBinning(2π/L), spectral=SpectralBackends.FFTSpectralBackend(), invariant=FIT.Types.Helicity())
    gh = FIT.SpectralFlux.calculate_spectral_flux(û3, ks3; binning=FIT.Types.LinearBinning(2π/L), spectral=SpectralBackends.FFTSpectralBackend(), invariant=FIT.Types.Helicity(), execution=ComputationalBackends.GPUBackend(KA.CPU()))
    Test.@test isapprox(gh.transfer_spectrum, rh.transfer_spectrum; atol=1e-12 * (maximum(abs, rh.transfer_spectrum)+eps()))
    Test.@test isapprox(gh.flux, rh.flux; atol=1e-12 * (maximum(abs, rh.flux)+eps()))

    re = FIT.SpectralFlux.calculate_spectral_flux(û, ks; binning=b, spectral=SpectralBackends.FFTSpectralBackend(), invariant=FIT.Types.Enstrophy())
    ge = FIT.SpectralFlux.calculate_spectral_flux(û, ks; binning=b, spectral=SpectralBackends.FFTSpectralBackend(), invariant=FIT.Types.Enstrophy(), execution=ComputationalBackends.GPUBackend(KA.CPU()))
    Test.@test isapprox(ge.transfer_spectrum, re.transfer_spectrum; atol=1e-12 * (maximum(abs, re.transfer_spectrum)+eps()))

    # AutoBackend resolves to the best available backend (threaded here, since OhMyThreads is
    # loaded and the suite runs multithreaded in CI) and must match the serial reference.
    auto = FIT.SpectralFlux.calculate_spectral_flux(û, ks; binning=b, spectral=SpectralBackends.FFTSpectralBackend(), execution=ComputationalBackends.AutoBackend())
    Test.@test isapprox(auto.transfer_spectrum, ref.transfer_spectrum; atol=atolT)
    Test.@test isapprox(auto.flux, ref.flux; atol=atolΠ)
end

# -----------------------------------------------------------------------
# Mode-to-mode S(k|p) and band-to-band T(n,m) on the KernelAbstractions GPU path (KA.CPU backend =
# the same device code exercised on CUDA/ROC/Metal): the per-giver / per-band nonlinear term + the
# device transfer-density kernel must reproduce the serial reduction to machine precision.
Test.@testset "mode-to-mode & band-to-band GPU (KA.CPU) parity" begin
    L = 2π; N = 16
    ks = FIT.Utils.wavenumber_grid((N, N), (L, L))
    kx = [ks[1][i] for i in 1:N, j in 1:N]; ky = [ks[2][j] for i in 1:N, j in 1:N]
    ψh = FFTW.fft(randn(Random.MersenneTwister(17), N, N)) ./ N^2
    û  = cat(im .* ky .* ψh, -im .* kx .* ψh; dims = 3)
    b  = FIT.Types.LinearBinning(2π / L)
    for sp in (SpectralBackends.DirectSumSpectralBackend(), SpectralBackends.FFTSpectralBackend())
        ms = FIT.calculate_mode_to_mode_transfer(û, ks; spectral=sp, execution=ComputationalBackends.SerialBackend())
        mg = FIT.calculate_mode_to_mode_transfer(û, ks; spectral=sp, execution=ComputationalBackends.GPUBackend(KA.CPU()))
        Test.@test isapprox(mg.transfer, ms.transfer; atol=1e-12 * (maximum(abs, ms.transfer)+eps()))
        Test.@test isapprox(mg.net_transfer, ms.net_transfer; atol=1e-12 * (maximum(abs, ms.net_transfer)+eps()))

        bnds = FIT.Types.SmoothBands(collect(1.0:6.0))
        bs = FIT.BandTransfer.calculate_band_to_band_transfer(û, ks; bands=bnds, spectral=sp, execution=ComputationalBackends.SerialBackend())
        bg = FIT.BandTransfer.calculate_band_to_band_transfer(û, ks; bands=bnds, spectral=sp, execution=ComputationalBackends.GPUBackend(KA.CPU()))
        Test.@test isapprox(bg.transfer_matrix, bs.transfer_matrix; atol=1e-12 * (maximum(abs, bs.transfer_matrix)+eps()))
    end

    # Partial (helical decomposition-channel) fluxes on the KA GPU path (3D): the device-generic
    # broadcast decomposition + per-pair nonlinear term + device transfer-density + scatter-add bin
    # must reproduce the serial channel fluxes to machine precision.
    Nc = 12; xs3 = collect(range(0, L; length=Nc+1)[1:Nc])
    u3 = [cos(x)*sin(y)*cos(z) for x in xs3, y in xs3, z in xs3]
    v3 = [-sin(x)*cos(y)*cos(z) for x in xs3, y in xs3, z in xs3]
    w3 = [sin(x)*sin(y)*sin(z) for x in xs3, y in xs3, z in xs3]
    û3 = cat(FFTW.fft(u3), FFTW.fft(v3), FFTW.fft(w3); dims=4) ./ Nc^3
    ks3p = FIT.Utils.wavenumber_grid((Nc, Nc, Nc), (L, L, L))
    for sp in (SpectralBackends.DirectSumSpectralBackend(), SpectralBackends.FFTSpectralBackend())
        ps = FIT.calculate_partial_fluxes(û3, ks3p; spectral=sp, execution=ComputationalBackends.SerialBackend())
        pg = FIT.calculate_partial_fluxes(û3, ks3p; spectral=sp, execution=ComputationalBackends.GPUBackend(KA.CPU()))
        Test.@test length(pg.channels) == length(ps.channels)
        for k in keys(ps.channels)
            Test.@test isapprox(pg.channels[k].flux, ps.channels[k].flux;
                                atol=1e-12 * (maximum(abs, ps.channels[k].flux)+eps()))
        end
    end
end

Test.@testset "passive-scalar GPU (KA.CPU) parity — scalar flux / shell-to-shell / mode-to-mode" begin
    # PassiveScalar shares the KE device transfer-density kernel (M=1 dot contraction). Pins that the
    # scalar GPU variants (scalar flux / shell-to-shell / mode-to-mode) run and match serial.
    L = 2π; N = 16
    ks = FIT.Utils.wavenumber_grid((N, N), (L, L))
    rng = Random.MersenneTwister(23)
    û = randn(rng, ComplexF64, N, N, 2)
    θ̂ = randn(rng, ComplexF64, N, N)
    b = FIT.Types.LinearBinning(2π / L)
    for sp in (SpectralBackends.DirectSumSpectralBackend(), SpectralBackends.FFTSpectralBackend())
        fs = FIT.calculate_scalar_flux(û, θ̂, ks; binning=b, spectral=sp, execution=ComputationalBackends.SerialBackend())
        fg = FIT.calculate_scalar_flux(û, θ̂, ks; binning=b, spectral=sp, execution=ComputationalBackends.GPUBackend(KA.CPU()))
        Test.@test isapprox(fg.flux, fs.flux; atol=1e-12 * (maximum(abs, fs.flux)+eps()))

        ss = FIT.calculate_scalar_shell_to_shell_transfer(û, θ̂, ks; binning=b, spectral=sp, execution=ComputationalBackends.SerialBackend(), verify_antisymmetry=false)
        sg = FIT.calculate_scalar_shell_to_shell_transfer(û, θ̂, ks; binning=b, spectral=sp, execution=ComputationalBackends.GPUBackend(KA.CPU()), verify_antisymmetry=false)
        Test.@test isapprox(sg.transfer_matrix, ss.transfer_matrix; atol=1e-12 * (maximum(abs, ss.transfer_matrix)+eps()))

        ms = FIT.calculate_scalar_mode_to_mode_transfer(û, θ̂, ks; spectral=sp, execution=ComputationalBackends.SerialBackend(), force=true)
        mg = FIT.calculate_scalar_mode_to_mode_transfer(û, θ̂, ks; spectral=sp, execution=ComputationalBackends.GPUBackend(KA.CPU()), force=true)
        Test.@test isapprox(mg.net_transfer, ms.net_transfer; atol=1e-12 * (maximum(abs, ms.net_transfer)+eps()))
    end
end

# -----------------------------------------------------------------------
Test.@testset "GPU backend routing — clear errors, device detection, dispatch" begin
    N = 8; L = 2π
    ks = FIT.Utils.wavenumber_grid((N, N), (L, L))
    rng = Random.MersenneTwister(41)
    û = randn(rng, ComplexF64, N, N, 2); ρ̂ = randn(rng, ComplexF64, N, N)
    gpu = ComputationalBackends.GPUBackend(KA.CPU())

    # is_gpu_array trait: host arrays (incl. non-`Array` host types) are NOT device; JLArray is.
    Test.@test ComputationalBackends.is_gpu_array(û) == false
    Test.@test ComputationalBackends.is_gpu_array(view(û, :, :, 1)) == false   # host SubArray ≠ device
    Test.@test ComputationalBackends.is_gpu_array(JLArrays.JLArray(û)) == true

    # DirectSum can't run on a device array (scalar reference) → clear error (via the trait), not a crash.
    Test.@test_throws ArgumentError FIT.SpectralFlux.calculate_spectral_flux(JLArrays.JLArray(û), ks; spectral = SpectralBackends.DirectSumSpectralBackend())
    # …but a host Array with DirectSum is fine.
    Test.@test FIT.SpectralFlux.calculate_spectral_flux(û, ks; spectral = SpectralBackends.DirectSumSpectralBackend()) isa FIT.Types.SpectralFluxResult

    # compressible: device path is array-driven; GPUBackend on a host array can't be honoured → clear error.
    Test.@test_throws ArgumentError FIT.Compressible.calculate_compressible_flux(û, ρ̂, ks; spectral = SpectralBackends.FFTSpectralBackend(), execution = gpu)

    # TOD: GPUBackend dispatches through the (device-generic) loop; on a host input it runs the host
    # kernels == serial (no MethodError, no dismissive error).
    X = randn(Random.MersenneTwister(42), 64, 4)
    tser = FIT.TriadicOrthogonalDecomposition.triadic_orthogonal_decomposition(X; dt = 0.05)
    tgpu = FIT.TriadicOrthogonalDecomposition.triadic_orthogonal_decomposition(X; dt = 0.05, execution = gpu)
    Test.@test isequal(tser.mode_bispectrum, tgpu.mode_bispectrum)
end

# -----------------------------------------------------------------------
# Device-genericity on JLArrays (allowscalar(false) — real device semantics, unlike GPUBackend(KA.CPU)
# which runs on plain Arrays). Broadcast/kernel building blocks must run device-resident with no scalar
# indexing and reproduce the CPU result. (FFT-based full pipelines can't run here: JLArrays has no
# AbstractFFTs provider — those are device-generic by construction, verified on KA.CPU above.)
Test.@testset "device-generic on JLArrays (no scalar indexing)" begin
    N = 8; L = 2π
    ks3 = FIT.Utils.wavenumber_grid((N, N, N), (L, L, L))
    û3 = randn(Random.MersenneTwister(5), ComplexF64, N, N, N, 3)

    # Helical decomposition — device-resident + bit-identical to CPU (pure broadcast projection).
    up, um = FIT.Decomposition.decompose_field(FIT.Types.HelicalDecomposition(), û3, ks3)
    upj, umj = FIT.Decomposition.decompose_field(FIT.Types.HelicalDecomposition(), JLArrays.JLArray(û3), ks3)
    Test.@test upj isa JLArrays.JLArray
    Test.@test maximum(abs, Array(upj) .- up) == 0
    Test.@test maximum(abs, Array(umj) .- um) == 0

    # Toroidal–poloidal decomposition — device-resident + bit-identical (broadcast Craya–Herring frame).
    tp  = FIT.Decomposition.decompose_field(FIT.Types.ToroidalPoloidalDecomposition(), û3, ks3)
    tpj = FIT.Decomposition.decompose_field(FIT.Types.ToroidalPoloidalDecomposition(), JLArrays.JLArray(û3), ks3)
    Test.@test tpj.toroidal isa JLArrays.JLArray
    Test.@test maximum(abs, Array(tpj.toroidal) .- tp.toroidal) == 0
    Test.@test maximum(abs, Array(tpj.poloidal) .- tp.poloidal) == 0

    # Helmholtz (Leray) decomposition — N-D device-generic projection, device-resident + bit-identical.
    hm  = FIT.Decomposition.decompose_field(FIT.Types.HelmholtzDecomposition(), û3, ks3)
    hmj = FIT.Decomposition.decompose_field(FIT.Types.HelmholtzDecomposition(), JLArrays.JLArray(û3), ks3)
    Test.@test hmj.rotational isa JLArrays.JLArray
    Test.@test maximum(abs, Array(hmj.rotational) .- hm.rotational) == 0
    Test.@test maximum(abs, Array(hmj.divergent) .- hm.divergent) == 0

    # transfer_density! (KE) — device kernel/broadcast matches the CPU reduction.
    ks2 = FIT.Utils.wavenumber_grid((N, N), (L, L))
    û2 = randn(Random.MersenneTwister(6), ComplexF64, N, N, 2)
    N̂2 = randn(Random.MersenneTwister(7), ComplexF64, N, N, 2)
    td_h = similar(û2, Float64, N, N); FIT.Invariants.transfer_density!(td_h, FIT.Types.KineticEnergy(), û2, N̂2, ks2)
    td_d = JLArrays.JLArray(similar(û2, Float64, N, N))
    FIT.Invariants.transfer_density!(td_d, FIT.Types.KineticEnergy(), JLArrays.JLArray(û2), JLArrays.JLArray(N̂2), ks2)
    Test.@test maximum(abs, Array(td_d) .- td_h) == 0
    # Passive scalar (M=1) — same device-generic dot broadcast as KE. (Helicity/Enstrophy use the
    # KA-extension vorticity kernels via GPUBackend, verified on KA.CPU above, not this broadcast path.)
    θ2 = randn(Random.MersenneTwister(8), ComplexF64, N, N, 1)
    θN = randn(Random.MersenneTwister(9), ComplexF64, N, N, 1)
    ts_h = similar(θ2, Float64, N, N); FIT.Invariants.transfer_density!(ts_h, FIT.Types.PassiveScalar(), θ2, θN, ks2)
    ts_d = JLArrays.JLArray(similar(θ2, Float64, N, N))
    FIT.Invariants.transfer_density!(ts_d, FIT.Types.PassiveScalar(), JLArrays.JLArray(θ2), JLArrays.JLArray(θN), ks2)
    Test.@test maximum(abs, Array(ts_d) .- ts_h) == 0

    # Compressible device helpers (copy-trunc / Helmholtz split / shell bin) match the scalar CPU path.
    dch = similar(û2); FIT.Compressible._copy_trunc!(dch, û2, ks2, (N, N), true)
    dcd = JLArrays.JLArray(similar(û2)); FIT.Compressible._copy_trunc!(dcd, JLArrays.JLArray(û2), ks2, (N, N), true)
    Test.@test maximum(abs, Array(dcd) .- dch) == 0
    rh = similar(û2); ch = similar(û2); FIT.Compressible._helmholtz_split!(rh, ch, û2, ks2, (N, N))
    rd = JLArrays.JLArray(similar(û2)); cd = JLArrays.JLArray(similar(û2))
    FIT.Compressible._helmholtz_split!(rd, cd, JLArrays.JLArray(û2), ks2, (N, N))
    Test.@test maximum(abs, Array(rd) .- rh) < 1e-14 * (maximum(abs, rh) + eps())
    Test.@test maximum(abs, Array(cd) .- ch) < 1e-14 * (maximum(abs, ch) + eps())
    kmag = FIT.ShellBinning.shell_coordinate(FIT.Types.IsotropicShells(), ks2); bb = FIT.Types.LinearBinning(2π / L)
    edges = FIT.ShellBinning.shell_edges(bb, maximum(kmag)); sidx = FIT.ShellBinning.assign_shells(kmag, edges)
    Nsh = length(collect(FIT.ShellBinning.shell_centers(bb, maximum(kmag))))
    Th = FIT.Compressible._bin(td_h, sidx, Nsh, Float64, ks2, (N, N), true)
    Td = FIT.Compressible._bin(JLArrays.JLArray(td_h), sidx, Nsh, Float64, ks2, (N, N), true)
    Test.@test maximum(abs, Td .- Th) == 0
end

# -----------------------------------------------------------------------
# Compressible KE spectral transfer (Singh–Tiwari–Sharma–Verma 2025). Validated by the
# analytic identities that make it trustworthy: (a) the momentum-weighted nonlinear transfer
# conserves total KE, Σ_k T_u = 0; (b) the incompressible limit ρ≡1, ∇·u=0 reduces T_u to
# −(incompressible transfer_spectrum) (paper Eqs. 48–50); (c) the R/C flux channels reconstruct
# the total flux and the compressive/cross channels vanish for incompressible flow; (d) uniform
# pressure ⇒ zero pressure-dilatation.
Test.@testset "Compressible energy transfer" begin
    L = 2π; N = 16
    ks = FIT.Utils.wavenumber_grid((N, N), (L, L))
    kx = [ks[1][i] for i in 1:N, j in 1:N]; ky = [ks[2][j] for i in 1:N, j in 1:N]
    # Broadband real divergence-free velocity from a random real streamfunction (nonzero net
    # inter-shell transfer, so the incompressible reference and the tolerances are non-degenerate).
    ψh = FFTW.fft(randn(Random.MersenneTwister(101), N, N)) ./ N^2
    û  = cat(im .* ky .* ψh, -im .* kx .* ψh; dims=3)
    b = FIT.Types.LinearBinning(2π / L)
    ρ̂ = zeros(ComplexF64, N, N); ρ̂[1, 1] = 1.0    # ρ(x) ≡ 1  (k=0 mode)

    res = FIT.Compressible.calculate_compressible_flux(û, ρ̂, ks; binning=b)
    ref = FIT.SpectralFlux.calculate_spectral_flux(û, ks; binning=b, spectral=SpectralBackends.FFTSpectralBackend())
    scaleT = maximum(abs, ref.transfer_spectrum) + eps()
    # (a) conservation
    Test.@test abs(sum(res.transfer_spectrum)) < 1e-10 * scaleT
    # (b) incompressible limit: T_u = −ρ·(incompressible T), here ρ=1
    Test.@test isapprox(res.transfer_spectrum, -ref.transfer_spectrum; atol=1e-10 * scaleT)
    # (c) channels: compressive & cross vanish; the four reconstruct the total flux
    Test.@test maximum(abs, res.channels.compressive) < 1e-10 * scaleT
    Test.@test maximum(abs, res.channels.comp_to_rot) < 1e-10 * scaleT
    Test.@test maximum(abs, res.channels.rot_to_comp) < 1e-10 * scaleT
    Test.@test isapprox(res.channels.rotational .+ res.channels.compressive .+
                        res.channels.rot_to_comp .+ res.channels.comp_to_rot, res.flux; atol=1e-10 * scaleT)
    # (d) uniform pressure ⇒ ∇σ = 0 ⇒ zero pressure-dilatation
    σ̂ = zeros(ComplexF64, N, N); σ̂[1, 1] = 1.0
    resp = FIT.Compressible.calculate_compressible_flux(û, ρ̂, ks; binning=b, pressure_hat=σ̂)
    Test.@test resp.pressure_dilatation !== nothing
    Test.@test maximum(abs, resp.pressure_dilatation.rotational) < 1e-10 * scaleT
    Test.@test maximum(abs, resp.pressure_dilatation.compressive) < 1e-10 * scaleT
    # no pressure ⇒ nothing
    Test.@test res.pressure_dilatation === nothing
end

# -----------------------------------------------------------------------
# Workspace-reuse allocation must be measured inside a fixed-arity function (no captured Module locals);
# an inline @allocated in a testset measures the closure/dynamic dispatch, not the call — it even reports
# reuse > fresh (nonsense). These return (reuse, fresh) so the tests gate a real ratio.
function _nufft_cg_reuse_fresh(ws, U, V, ℓ, filt, ms, X, Y, spectral, Ls)
    FIT.nufft_coarse_graining_flux!(ws, (U, V), ℓ, filt, ms)
    a_reuse = @allocated FIT.nufft_coarse_graining_flux!(ws, (U, V), ℓ, filt, ms)
    FIT.nufft_coarse_graining_flux((U, V), (X, Y), ℓ, filt, ms; spectral = spectral, Ls = Ls)
    a_fresh = @allocated FIT.nufft_coarse_graining_flux((U, V), (X, Y), ℓ, filt, ms; spectral = spectral, Ls = Ls)
    return (a_reuse, a_fresh)
end
function _nufft_ts_reuse_fresh(ws, fields, coords, ms, Ls, spectral)
    FIT.to_spectral!(ws, fields)
    a_reuse = @allocated FIT.to_spectral!(ws, fields)
    FIT.to_spectral(fields, coords, ms; spectral = spectral, Ls = Ls)
    a_fresh = @allocated FIT.to_spectral(fields, coords, ms; spectral = spectral, Ls = Ls)
    return (a_reuse, a_fresh)
end

# Extension smoke tests: exercise the previously-untested extensions with meaningful
# numerical assertions, not @test true. CairoMakie (plot dispatch incl. the new TOD figure),
# FINUFFT (scattered-Cartesian coarse-graining + the calculate_energy_transfer wiring),
# and FlowFieldSpectra (physical→spectral front-end). FSH/NUFSHT spherical transfer is tested
# in its own testset once the genuine spherical implementation lands.
Test.@testset "Extension smoke tests (CairoMakie / FINUFFT / FlowFieldSpectra)" begin
    L = 2π; N = 16
    ks = FIT.Utils.wavenumber_grid((N, N), (L, L))
    kx = [ks[1][i] for i in 1:N, j in 1:N]; ky = [ks[2][j] for i in 1:N, j in 1:N]
    ψh = FFTW.fft(randn(Random.MersenneTwister(21), N, N)) ./ N^2
    û  = cat(im .* ky .* ψh, -im .* kx .* ψh; dims = 3)
    b  = FIT.Types.LinearBinning(2π / L)

    Test.@testset "CairoMakie plot dispatch" begin
        sf = FIT.SpectralFlux.calculate_spectral_flux(û, ks; binning=b, spectral=SpectralBackends.FFTSpectralBackend())
        Test.@test FIT.plot_energy_transfer(sf) isa CairoMakie.Figure
        ss = FIT.ShellToShellTransfer.calculate_shell_to_shell_transfer(û, ks; binning=b, spectral=SpectralBackends.FFTSpectralBackend())
        Test.@test FIT.plot_energy_transfer(ss) isa CairoMakie.Figure
        # TOD bispectrum plot: build a minimal result and render it.
        nF = 6; nm = 2
        freqs = collect(range(-2.0, 2.0; length=nF))
        λ = abs.(randn(Random.MersenneTwister(3), nF, nF, nm))
        Tb = randn(Random.MersenneTwister(4), nF, nF, nm)
        tod = FIT.Types.TriadicOrthogonalDecompositionResult(freqs, λ, Dict{Tuple{Int,Int},Any}(), Tb, nothing, nothing)
        Test.@test FIT.plot_energy_transfer(tod) isa CairoMakie.Figure
        Test.@test FIT.plot_energy_transfer(tod; mode=2, fmax=1.5) isa CairoMakie.Figure
        Test.@test_throws ArgumentError FIT.plot_energy_transfer(tod; mode=99)
    end

    Test.@testset "NUFFT scattered coarse-graining — both providers vs exact NDFT" begin
        # Correctness gate against a brute-force EXACT NDFT (flux + stress τ̄ + strain S̄) — the
        # independent ground truth for scattered spectral coarse-graining. Both providers (peers) run
        # in-process via backend-type dispatch. The S̄ gate is the Nyquist-sign regression test: τ̄ is
        # |k|-based (immune), so only a strain/derivative check catches a wrong Nyquist mode.
        Random.seed!(3)
        Np = 2000; ms = (20, 20)
        X = 2π .* rand(Np); Y = 2π .* rand(Np)
        U = @. sin(X) * cos(Y); V = @. -cos(X) * sin(Y)
        ℓ = 0.8; filt = FIT.Types.GaussianFilter()

        Lx = 2π; Ly = 2π                       # true periodic domain (X, Y ∈ [0, 2π)); not the sample span
        xt = (X .- minimum(X)) ./ Lx .* 2π; yt = (Y .- minimum(Y)) ./ Ly .* 2π
        modeof(a, n) = 2a < n ? a : a - n
        m1 = [modeof(a, ms[1]) for a in 0:ms[1]-1]; m2 = [modeof(b, ms[2]) for b in 0:ms[2]-1]
        kpx = m1 .* (2π / Lx); kpy = m2 .* (2π / Ly)
        A = eachindex(m1); Bx = eachindex(m2)
        Gh = [FIT.Filters.filter_response(filt, sqrt(kpx[a]^2 + kpy[b]^2), ℓ) for a in A, b in Bx]
        t1(w) = [sum(w[j] * cis(-(m1[a] * xt[j] + m2[b] * yt[j])) for j in 1:Np) for a in A, b in Bx]
        t2(S) = [real(sum(S[a, b] * cis(+(m1[a] * xt[l] + m2[b] * yt[l])) for a in A, b in Bx)) for l in 1:Np]
        invN = 1 / Np
        ūs = Gh .* t1(U) .* invN; v̄s = Gh .* t1(V) .* invN
        ū = t2(ūs); v̄ = t2(v̄s)
        τxx = t2(Gh .* t1(U .* U) .* invN) .- ū .* ū
        τyy = t2(Gh .* t1(V .* V) .* invN) .- v̄ .* v̄
        Sxx = t2([im * kpx[a] * ūs[a, b] for a in A, b in Bx])
        Syy = t2([im * kpy[b] * v̄s[a, b] for a in A, b in Bx])
        Sxy = 0.5 .* (t2([im * kpy[b] * ūs[a, b] for a in A, b in Bx]) .+ t2([im * kpx[a] * v̄s[a, b] for a in A, b in Bx]))
        τxy = t2(Gh .* t1(U .* V) .* invN) .- ū .* v̄
        Πref = -(τxx .* Sxx .+ τyy .* Syy .+ 2 .* τxy .* Sxy)
        relq(a, b) = sqrt(sum(abs2, a .- b) / sum(abs2, a))
        method = FIT.Types.CoarseGrainingFluxMethod(filt, ℓ)

        for spectral in (FIT.Types.FINUFFTBackend(), FIT.Types.NonuniformFFTsBackend())
            r = FIT.nufft_coarse_graining_flux((U, V), (X, Y), ℓ, filt, ms; spectral = spectral, Ls = (Lx, Ly), return_diagnostics = true)
            Test.@test relq(τxx, r.stress_tensor[:, 1, 1]) < 1e-6
            Test.@test relq(τyy, r.stress_tensor[:, 2, 2]) < 1e-6
            Test.@test relq(Sxx, r.strain_rate[:, 1, 1]) < 1e-6   # Nyquist-sign regression gate
            Test.@test relq(Sxy, r.strain_rate[:, 1, 2]) < 1e-6
            Test.@test relq(Πref, r.flux_field) < 1e-6
            wired = FIT.calculate_energy_transfer(method, (U, V), (X, Y), ms; spectral = spectral, Ls = (Lx, Ly))
            Test.@test isapprox(wired.flux_field, r.flux_field; rtol = 1e-8)
            ws = FIT.NUFFTCoarseGrainingWorkspace((X, Y), ms; spectral = spectral, Ls = (Lx, Ly))
            a_reuse, a_fresh = _nufft_cg_reuse_fresh(ws, U, V, ℓ, filt, ms, X, Y, spectral, (Lx, Ly))
            # Reuse skips the plan rebuild (the dominant cost), so it stays below a fresh build. A tighter
            # ratio isn't portable: NonuniformFFTs threads its transforms off Threads.nthreads() with no
            # per-plan control, so at -t>1 each exec carries a scratch floor that FINUFFT (pinnable to 1
            # thread) doesn't — the exact allocation is thread-count dependent for one provider but not the
            # other, so gate the claim that holds for both.
            Test.@test a_reuse < a_fresh
        end
        # Odd mode counts exercise the other branch of the real-input (r2c) analysis: an odd axis has no
        # Nyquist mode, an even one needs the oversampled-spectrum read. Both must hit the same reference.
        let mso = (21, 21)
            m1o = [modeof(a, mso[1]) for a in 0:mso[1]-1]; m2o = [modeof(b, mso[2]) for b in 0:mso[2]-1]
            kxo = m1o .* (2π / Lx); kyo = m2o .* (2π / Ly)
            Ao = eachindex(m1o); Bo = eachindex(m2o)
            Go = [FIT.Filters.filter_response(filt, sqrt(kxo[a]^2 + kyo[b]^2), ℓ) for a in Ao, b in Bo]
            t1o(w) = [sum(w[j] * cis(-(m1o[a] * xt[j] + m2o[b] * yt[j])) for j in 1:Np) for a in Ao, b in Bo]
            t2o(S) = [real(sum(S[a, b] * cis(+(m1o[a] * xt[l] + m2o[b] * yt[l])) for a in Ao, b in Bo)) for l in 1:Np]
            ūso = Go .* t1o(U) .* invN; v̄so = Go .* t1o(V) .* invN
            ūo = t2o(ūso); v̄o = t2o(v̄so)
            τxxo = t2o(Go .* t1o(U .* U) .* invN) .- ūo .* ūo
            τyyo = t2o(Go .* t1o(V .* V) .* invN) .- v̄o .* v̄o
            τxyo = t2o(Go .* t1o(U .* V) .* invN) .- ūo .* v̄o
            Sxxo = t2o([im * kxo[a] * ūso[a, b] for a in Ao, b in Bo])
            Syyo = t2o([im * kyo[b] * v̄so[a, b] for a in Ao, b in Bo])
            Sxyo = 0.5 .* (t2o([im * kyo[b] * ūso[a, b] for a in Ao, b in Bo]) .+ t2o([im * kxo[a] * v̄so[a, b] for a in Ao, b in Bo]))
            Πo = -(τxxo .* Sxxo .+ τyyo .* Syyo .+ 2 .* τxyo .* Sxyo)
            for spectral in (FIT.Types.FINUFFTBackend(), FIT.Types.NonuniformFFTsBackend())
                ro = FIT.nufft_coarse_graining_flux((U, V), (X, Y), ℓ, filt, mso; spectral = spectral, Ls = (Lx, Ly), return_diagnostics = true)
                Test.@test relq(Sxxo, ro.strain_rate[:, 1, 1]) < 1e-6
                Test.@test relq(Πo, ro.flux_field) < 1e-6
            end
        end
        # Ls is a required physical input (the domain the samples under-span), never guessed from the span.
        Test.@test_throws UndefKeywordError FIT.nufft_coarse_graining_flux((U, V), (X, Y), ℓ, filt, ms; spectral = FIT.Types.FINUFFTBackend())
    end

    Test.@testset "NUFFT scattered → to_spectral (uniform reconstruction feeds the flux family)" begin
        Nn = 8; Ln = 2π
        xs1 = [(i - 1) * Ln / Nn for i in 1:Nn]
        Xg = [xs1[i] for i in 1:Nn, j in 1:Nn]; Yg = [xs1[j] for i in 1:Nn, j in 1:Nn]
        Random.seed!(31)
        ug = randn(Nn, Nn); vg = randn(Nn, Nn)

        # Samples on a uniform grid → û == fft(u)/Nᵈ exactly (drop-in for the uniform diagnostics).
        û_sc, ks_sc = FIT.to_spectral((vec(ug), vec(vg)), (vec(Xg), vec(Yg)), (Nn, Nn);
                                        spectral = FIT.Types.FINUFFTBackend(), Ls = (Ln, Ln))
        û_man  = cat(FFTW.fft(ug), FFTW.fft(vg); dims = 3) ./ Nn^2
        ks_man = FIT.Utils.wavenumber_grid((Nn, Nn), (Ln, Ln))
        Test.@test isapprox(û_sc, û_man; atol = 1e-8)
        Test.@test all(isapprox.(ks_sc, ks_man; atol = 1e-10))

        # The reconstructed coefficients feed the ordinary uniform spectral flux, matching the manual path.
        Π_sc  = FIT.SpectralFlux.calculate_spectral_flux(û_sc, ks_sc; spectral = SpectralBackends.FFTSpectralBackend())
        Π_man = FIT.SpectralFlux.calculate_spectral_flux(û_man, ks_man; spectral = SpectralBackends.FFTSpectralBackend())
        Test.@test isapprox(Π_sc.flux, Π_man.flux; atol = 1e-10)

        # Genuinely scattered points: a single-mode field recovers its exact wavenumber.
        Np = 4000
        rng2 = Random.MersenneTwister(99)
        xr = rand(rng2, Np) .* Ln; yr = rand(rng2, Np) .* Ln
        kx0 = 2π / Ln * 2; ky0 = 2π / Ln * 3
        us = cos.(kx0 .* xr .+ ky0 .* yr)
        ûs, kss = FIT.to_spectral((us,), (xr, yr), (Nn, Nn); spectral = FIT.Types.FINUFFTBackend(), Ls = (Ln, Ln))
        peak = argmax(abs.(ûs[:, :, 1]))
        Test.@test abs(abs(kss[1][peak[1]]) - kx0) < 1e-8
        Test.@test abs(abs(kss[2][peak[2]]) - ky0) < 1e-8

        # In-place workspace form reuses the plan + buffers across calls (both providers reconstruct
        # û == fft(u)/Nᵈ on the uniform grid). Serial FINUFFT (C) is 0-alloc on repeat; NonuniformFFTs
        # carries an inherent per-exec KA/library alloc floor, so its reuse is gated well below the fresh
        # (plan-building) call rather than == 0.
        for spectral in (FIT.Types.FINUFFTBackend(), FIT.Types.NonuniformFFTsBackend())
            wsts = FIT.NUFFTToSpectralWorkspace((vec(Xg), vec(Yg)), (Nn, Nn); spectral = spectral, ncomponents = 2, Ls = (Ln, Ln))
            û_ip, _ = FIT.to_spectral!(wsts, (vec(ug), vec(vg)))
            Test.@test isapprox(û_ip, û_man; atol = 1e-8)
            a_reuse, a_fresh = _nufft_ts_reuse_fresh(wsts, (vec(ug), vec(vg)), (vec(Xg), vec(Yg)), (Nn, Nn), (Ln, Ln), spectral)
            if spectral isa FIT.Types.FINUFFTBackend
                Test.@test a_reuse == 0                                     # C plan pinned single-threaded → genuinely 0-alloc reuse
            else
                Test.@test a_reuse < a_fresh                                # NonuniformFFTs threads off nthreads() (unpinnable per-exec floor); reuse still skips the plan rebuild
            end
        end

        # Device-resident path (NonuniformFFTs provider): a GPUBackend builds the PlanNUFFT on its KA
        # backend with device points/buffers, so scattered → velocity_hat runs on-device. GPUBackend(KA.CPU())
        # exercises that plumbing on the host (device-array kind = Array), CPU-parity-testable with no GPU.
        û_host, _ = FIT.to_spectral((vec(ug), vec(vg)), (vec(Xg), vec(Yg)), (Nn, Nn);
                                    spectral = FIT.Types.NonuniformFFTsBackend(), Ls = (Ln, Ln))
        û_dev, _  = FIT.to_spectral((vec(ug), vec(vg)), (vec(Xg), vec(Yg)), (Nn, Nn);
                                    spectral = FIT.Types.NonuniformFFTsBackend(), Ls = (Ln, Ln),
                                    execution = ComputationalBackends.GPUBackend(KA.CPU()))
        Test.@test isapprox(Array(û_dev), û_host; atol = 1e-10)   # device path == host path
        Test.@test isapprox(Array(û_dev), û_man;  atol = 1e-8)    # …and == fft(u)/Nᵈ on the uniform grid

        # Float32 with a below-eps(Float32) tol: half-support clamps to eps(FT), so spreading stays finite.
        û_f32, _ = FIT.to_spectral((Float32.(vec(ug)), Float32.(vec(vg))),
                                   (Float32.(vec(Xg)), Float32.(vec(Yg))), (Nn, Nn);
                                   spectral = FIT.Types.NonuniformFFTsBackend(), Ls = (Ln, Ln), tol = 1e-9)
        Test.@test eltype(û_f32) == ComplexF32
        Test.@test all(isfinite, û_f32)
        Test.@test maximum(abs, û_f32 .- ComplexF32.(û_man)) < 1e-2

        # 3-arg scattered form requires a NUFFT backend (Types.FINUFFTBackend) — a clear error, not a silent wrong path.
        Test.@test_throws ArgumentError FIT.to_spectral((us,), (xr, yr), (Nn, Nn); spectral = SpectralBackends.FFTSpectralBackend(), Ls = (Ln, Ln))

        # The NonuniformFFTs provider transforms the real velocity through a real-input (r2c) plan and
        # expands the non-redundant half to the full spectrum. On an even axis d ≥ 2 the Hermitian fold
        # needs mode +N_d/2, which is not an output mode (for scattered points +N_d/2 ≠ −N_d/2), so it is
        # read from the oversampled spectrum the transform already computed. Gate that against the complex
        # (FINUFFT) path over even / odd / mixed mode counts — an even axis is the case that breaks first.
        rngr = Random.MersenneTwister(5)
        Nr = 3000
        xr2 = Ln .* rand(rngr, Nr); yr2 = Ln .* rand(rngr, Nr)
        ur2 = @. sin(3 * xr2) * cos(2 * yr2) + 0.25 * cos(7 * xr2)
        for msr in ((20, 20), (21, 21), (20, 21), (21, 20))
            û_r2c, _ = FIT.to_spectral((ur2,), (xr2, yr2), msr; spectral = FIT.Types.NonuniformFFTsBackend(), Ls = (Ln, Ln))
            û_cpx, _ = FIT.to_spectral((ur2,), (xr2, yr2), msr; spectral = FIT.Types.FINUFFTBackend(), Ls = (Ln, Ln))
            Test.@test sqrt(sum(abs2, û_r2c .- û_cpx) / sum(abs2, û_cpx)) < 1e-7
        end
    end

    Test.@testset "cuFINUFFT device to_spectral (FINUFFT provider)" begin
        # Ext load + device-method registration: checkable wherever CUDA is present, no GPU needed. A
        # cuFINUFFT symbol named in a dispatch signature (rather than wrapped in the owned plan handle)
        # would fail to precompile the ext and trip these.
        ext = Base.get_extension(FIT, :FlowInvariantTransferFINUFFTCUDAExt)
        Test.@test ext !== nothing
        Test.@test any(m -> occursin("CUDABackend", string(m.sig)), Base.methods(FIT._finufft_ts_build))
        Test.@test any(m -> occursin("CuFINUFFTPlan", string(m.sig)), Base.methods(FIT.to_spectral!))

        # The device transform needs an NVIDIA GPU; where one exists it reconstructs the same coefficients
        # as the host FINUFFT path. No functional GPU here → the transform is not exercised (logged).
        if CUDA.functional()
            Nn = 8; Ln = 2π
            xs1 = [(i - 1) * Ln / Nn for i in 1:Nn]
            Xg = [xs1[i] for i in 1:Nn, j in 1:Nn]; Yg = [xs1[j] for i in 1:Nn, j in 1:Nn]
            Random.seed!(7); ug = randn(Nn, Nn); vg = randn(Nn, Nn)
            û_host, _ = FIT.to_spectral((vec(ug), vec(vg)), (vec(Xg), vec(Yg)), (Nn, Nn);
                                        spectral = FIT.Types.FINUFFTBackend(), Ls = (Ln, Ln))
            wsd = FIT.NUFFTToSpectralWorkspace((vec(Xg), vec(Yg)), (Nn, Nn); spectral = FIT.Types.FINUFFTBackend(),
                                               ncomponents = 2, Ls = (Ln, Ln),
                                               execution = ComputationalBackends.GPUBackend(CUDA.CUDABackend()))
            û_dev, _ = FIT.to_spectral!(wsd, (CUDA.CuArray(vec(ug)), CUDA.CuArray(vec(vg))))
            Test.@test isapprox(Array(û_dev), û_host; atol = 1e-6)
        else
            @info "cuFINUFFT device transform not exercised: no functional CUDA GPU on this host"
        end
    end

    Test.@testset "scattered one-call calculate_energy_transfer (Cartesian spectral family)" begin
        # A NUFFT spectral backend routes the 4-positional (method, fields, coords, ms) entry to the
        # scattered reconstruction + diagnostic; it must equal the explicit two-step (to_spectral then the
        # coefficient diagnostic). A NUFFT backend and the uniform (FlowFieldSpectra) backend share this
        # signature and dispatch cleanly on the backend type.
        Nn = 8; Ln = 2π
        xs1 = [(i - 1) * Ln / Nn for i in 1:Nn]
        Xg = [xs1[i] for i in 1:Nn, j in 1:Nn]; Yg = [xs1[j] for i in 1:Nn, j in 1:Nn]
        Random.seed!(17); ug = randn(Nn, Nn); vg = randn(Nn, Nn)
        fields = (vec(ug), vec(vg)); coords = (vec(Xg), vec(Yg))
        b = FIT.Types.LinearBinning(2π / Ln)
        for spectral in (FIT.Types.FINUFFTBackend(), FIT.Types.NonuniformFFTsBackend())
            û_oc, ks_oc = FIT.to_spectral(fields, coords, (Nn, Nn); spectral = spectral, Ls = (Ln, Ln))
            sf1 = FIT.calculate_energy_transfer(FIT.Types.SpectralFluxMethod(b), fields, coords, (Nn, Nn); spectral = spectral, Ls = (Ln, Ln))
            sf2 = FIT.calculate_energy_transfer(FIT.Types.SpectralFluxMethod(b), û_oc, ks_oc)
            Test.@test isapprox(sf1.flux, sf2.flux; rtol = 1e-10, atol = 1e-12)
            ss1 = FIT.calculate_energy_transfer(FIT.Types.ShellToShellTransferMethod(b), fields, coords, (Nn, Nn); spectral = spectral, Ls = (Ln, Ln))
            ss2 = FIT.calculate_energy_transfer(FIT.Types.ShellToShellTransferMethod(b), û_oc, ks_oc)
            Test.@test isapprox(ss1.transfer_matrix, ss2.transfer_matrix; rtol = 1e-10, atol = 1e-12)
            mm1 = FIT.calculate_energy_transfer(FIT.Types.ModeToModeTransferMethod(b), fields, coords, (Nn, Nn); spectral = spectral, Ls = (Ln, Ln))
            mm2 = FIT.calculate_energy_transfer(FIT.Types.ModeToModeTransferMethod(b), û_oc, ks_oc)
            Test.@test isapprox(mm1.transfer, mm2.transfer; rtol = 1e-10, atol = 1e-12)
        end
        # NUFFT backend needs Ls (periodic domain) — a clear error, not a silent guess.
        Test.@test_throws UndefKeywordError FIT.calculate_energy_transfer(
            FIT.Types.SpectralFluxMethod(b), fields, coords, (Nn, Nn); spectral = FIT.Types.FINUFFTBackend())
    end

    Test.@testset "FlowFieldSpectra front-end" begin
        # Physical-space uniform periodic field → spectral coeffs via FlowFieldSpectra, then the
        # spectral-flux transfer. Band-limit (|k| ≤ 3) so the 2/3 dealiasing is a no-op and the
        # incompressible conservation Σ_k T = 0 holds exactly (not just up to the dealiasing residual).
        ψbl = copy(ψh)
        for i in 1:N, j in 1:N
            (abs(kx[i, j]) <= 3 && abs(ky[i, j]) <= 3) || (ψbl[i, j] = 0)
        end
        ûbl = cat(im .* ky .* ψbl, -im .* kx .* ψbl; dims = 3)
        xs = range(0, 2π; length=N+1)[1:N]   # uniform axis as a range → FlowGeometries infers its period
        # Physical field from the fft/N² coefficients is bfft(û) = Σ_k û e^{ik·x} (unnormalized
        # inverse); ifft(û) would give u/Nᵈ (a scaled field) and mis-scale the FFS comparison.
        uxp = real.(FFTW.bfft(ûbl[:, :, 1]))
        uyp = real.(FFTW.bfft(ûbl[:, :, 2]))
        ffs = FIT.calculate_energy_transfer(FIT.Types.SpectralFluxMethod(b), (uxp, uyp), (xs, xs), (N, N))
        Test.@test ffs isa FIT.Types.SpectralFluxResult
        Test.@test abs(sum(ffs.transfer_spectrum)) < 1e-8 * (maximum(abs, ffs.transfer_spectrum) + eps())
        # Correctness: the physical→spectral front-end must reproduce the transfer computed directly
        # from the field's own FFT coefficients (same 2/3 dealiasing on both sides).
        ref = FIT.SpectralFlux.calculate_spectral_flux(ûbl, ks; binning=b, spectral=SpectralBackends.FFTSpectralBackend())
        Test.@test isapprox(ffs.transfer_spectrum, ref.transfer_spectrum;
                            atol = 1e-10 * (maximum(abs, ref.transfer_spectrum) + eps()))
    end
end

# -----------------------------------------------------------------------
# Spherical spectral transfer (FSH extension, 2D-barotropic).
# The rigorous anchor is exact conservation Σ_l T = 0 (= ∫ψ J(ψ,ζ) dΩ = 0 by antisymmetry),
# which holds to machine precision iff the quadratic Jacobian is dealiased (evaluated on the
# 2·lmax grid). Convention verified directly against FSH's eth definition.
Test.@testset "Spherical spectral transfer (FastSphericalHarmonics, 2D barotropic)" begin
    lmax = 20; N = lmax + 1
    rng = Random.MersenneTwister(2024)
    # Band-limited real vorticity field on the FSH equiangular grid.
    ζ = FSH.spinsph_evaluate(FSH.spinsph_transform(randn(rng, N, 2N - 1), 0), 0)
    ζ = real.(ζ)

    res = FIT.calculate_energy_transfer(FIT.Types.SphericalTransferMethod(), ζ)
    Test.@test res isa FIT.Types.SphericalTransferResult
    Test.@test res.degrees == collect(0.0:lmax)

    scaleE = maximum(abs, res.energy_transfer) + eps()
    scaleZ = maximum(abs, res.enstrophy_transfer) + eps()
    # Genuine, nonzero transfer (not a degenerate zero field).
    Test.@test scaleE > 1e-6
    Test.@test scaleZ > 1e-6
    # Exact conservation (dealiased) — the physical anchor.
    Test.@test abs(sum(res.energy_transfer))    < 1e-12 * scaleE
    Test.@test abs(sum(res.enstrophy_transfer)) < 1e-12 * scaleZ
    # Flux convention Π(L) = -Σ_{l≤L} T; total flux out of the whole spectrum ≈ 0.
    Test.@test res.energy_flux ≈ -cumsum(res.energy_transfer)
    Test.@test abs(res.energy_flux[end]) < 1e-12 * scaleE
    # l = 0 (mean) carries no transfer (energy exactly, enstrophy to machine precision since
    # A₀₀ = mean(J) = ∫∇·(ζu) dΩ ≈ 0).
    Test.@test res.energy_transfer[1] == 0
    Test.@test abs(res.enstrophy_transfer[1]) < 1e-12 * scaleZ

    # Radius independence of conservation.
    res2 = FIT.calculate_energy_transfer(FIT.Types.SphericalTransferMethod(radius = 2.0), ζ)
    Test.@test abs(sum(res2.enstrophy_transfer)) < 1e-12 * (maximum(abs, res2.enstrophy_transfer) + eps())

    # Dealiasing does real work: on the same broadband field, the aliased path does NOT conserve.
    aliased = FIT.calculate_energy_transfer(FIT.Types.SphericalTransferMethod(), ζ; dealias = false)
    Test.@test abs(sum(aliased.enstrophy_transfer)) > 1e-6 * (maximum(abs, aliased.enstrophy_transfer) + eps())

    # Grid-shape guard.
    Test.@test_throws ArgumentError FIT.calculate_energy_transfer(FIT.Types.SphericalTransferMethod(), randn(N, N))

    # Backend selectors + execution axis: SpectralBackends.FSHTSpectralBackend is the regular-grid selector (== default); a
    # scattered/spherical-mismatched backend and a non-serial execution both raise a clear error.
    res_sel = FIT.calculate_energy_transfer(FIT.Types.SphericalTransferMethod(), ζ; spectral = SpectralBackends.FSHTSpectralBackend())
    Test.@test res_sel.energy_transfer == res.energy_transfer
    Test.@test_throws ArgumentError FIT.calculate_energy_transfer(FIT.Types.SphericalTransferMethod(), ζ; spectral = SpectralBackends.NUFSHTSpectralBackend())
    Test.@test_throws ArgumentError FIT.calculate_energy_transfer(FIT.Types.SphericalTransferMethod(), ζ; execution = ComputationalBackends.DistributedBackend())

    # In-place workspace form: matches the allocating path exactly (same math), and a repeat call
    # reuses the embed/gradient/Jacobian/reduction buffers — the FastSphericalHarmonics transforms
    # (no in-place API) are the irreducible floor, so this saves the FIT-side portion, not all of it.
    ws = FIT.Spherical.SphericalTransferWorkspace(lmax; radius = 1.0, dealias = true)
    ip = FIT.Spherical.calculate_spherical_transfer!(ws, ζ)
    Test.@test ip isa FIT.Types.SphericalTransferResult
    Test.@test ip.energy_transfer == res.energy_transfer
    Test.@test ip.enstrophy_transfer == res.enstrophy_transfer
    Test.@test ip.energy_flux == res.energy_flux
    Test.@test abs(sum(ip.enstrophy_transfer)) < 1e-12 * scaleZ
    # (workspace-reuse allocation ratio asserted in test_allocs.jl)
    # Workspace grid must match the field.
    Test.@test_throws ArgumentError FIT.Spherical.calculate_spherical_transfer!(ws, randn(N, N))
end

# -----------------------------------------------------------------------
# Scattered spherical transfer (NUFSHT extension). Same
# 2D-barotropic transfer at scattered points via NUFSHT's FINUFFT-backed spin transforms.
# Anchor: cross-check against the validated FSH regular-grid path on the SAME field, sampled
# identically (random real-SH coeffs C → FSH grid via sph_evaluate, scattered via nusht_type2!).
# Coefficient recovery is well-conditioned only for equidistributed (spherical-Fibonacci) points.
Test.@testset "Scattered spherical transfer (NUFSHT, 2D barotropic)" begin
    a = 1.0
    lmax = 8; N = lmax + 1
    rng = Random.MersenneTwister(99)
    C = zeros(N, 2N - 1)
    for ℓ in 0:lmax, m in -ℓ:ℓ; C[FSH.sph_mode(ℓ, m)] = randn(rng); end   # broadband real-SH field
    ζgrid = FSH.sph_evaluate(C)
    resF = FIT.calculate_energy_transfer(FIT.Types.SphericalTransferMethod(radius = a), ζgrid)

    # Spherical-Fibonacci scattered points (equidistributed), M ≥ (2lmax+1)² for the dealiased solve.
    Msolve = 8 * (2lmax + 1)^2
    ga = π * (3 - sqrt(5))
    zf = [1 - 2 * (k + 0.5) / Msolve for k in 0:Msolve-1]
    θs = acos.(clamp.(zf, -1.0, 1.0)); φs = mod.(ga .* (0:Msolve-1), 2π)
    ζscat = zeros(Msolve)
    NUFSHT.nusht_type2!(ζscat, C, NUFSHT.make_plan(θs, φs, lmax; tol = 1e-12))

    res = FIT.calculate_energy_transfer(FIT.Types.SphericalTransferMethod(radius = a), ζscat, (θs, φs);
                                        lmax = lmax, tol = 1e-12, rtol = 1e-13)
    Test.@test res isa FIT.Types.SphericalTransferResult
    Test.@test res.degrees == collect(0.0:lmax)

    scaleE = maximum(abs, resF.energy_transfer) + eps()
    scaleZ = maximum(abs, resF.enstrophy_transfer) + eps()
    Test.@test scaleE > 1e-3   # genuine nonzero transfer in the reference
    # Conservation (CG-tolerance-limited on scattered points).
    Test.@test abs(sum(res.energy_transfer))    < 1e-8 * scaleE
    Test.@test abs(sum(res.enstrophy_transfer)) < 1e-8 * scaleZ
    # Cross-check: scattered transfer matches the validated FSH grid transfer for the same field.
    Test.@test maximum(abs.(res.energy_transfer    .- resF.energy_transfer))    < 0.02 * scaleE
    Test.@test maximum(abs.(res.enstrophy_transfer .- resF.enstrophy_transfer)) < 0.02 * scaleZ

    # Guard: too few points for the dealiased solve.
    Test.@test_throws ArgumentError FIT.calculate_energy_transfer(
        FIT.Types.SphericalTransferMethod(), ζscat[1:10], (θs[1:10], φs[1:10]); lmax = lmax)

    # Backend selectors + execution axis: SpectralBackends.NUFSHTSpectralBackend is the scattered selector (== default); the
    # regular-grid mismatch and non-serial execution error clearly (device path is coord-driven).
    res_sel = FIT.calculate_energy_transfer(FIT.Types.SphericalTransferMethod(radius = a), ζscat, (θs, φs); lmax = lmax, tol = 1e-12, rtol = 1e-13, spectral = SpectralBackends.NUFSHTSpectralBackend())
    Test.@test res_sel.energy_transfer == res.energy_transfer
    Test.@test_throws ArgumentError FIT.calculate_energy_transfer(FIT.Types.SphericalTransferMethod(radius = a), ζscat, (θs, φs); lmax = lmax, spectral = SpectralBackends.FSHTSpectralBackend())
    Test.@test_throws ArgumentError FIT.calculate_energy_transfer(FIT.Types.SphericalTransferMethod(radius = a), ζscat, (θs, φs); lmax = lmax, execution = ComputationalBackends.DistributedBackend())

    # In-place workspace form: reuses the three NUFSHT plans (the dominant cost) + all buffers, so
    # a snapshot sweep on the same points re-plans nothing. Matches the allocating path to the CG
    # tolerance; a repeat call allocates far less (only the NUFSHT-internal CG scratch remains).
    ws = FIT.Spherical.ScatteredSphericalTransferWorkspace((θs, φs), lmax; radius = a, tol = 1e-12, rtol = 1e-13)
    ip = FIT.Spherical.calculate_spherical_transfer!(ws, ζscat)
    Test.@test ip isa FIT.Types.SphericalTransferResult
    Test.@test maximum(abs.(ip.energy_transfer .- res.energy_transfer)) < 1e-10 * scaleE
    Test.@test maximum(abs.(ip.enstrophy_transfer .- res.enstrophy_transfer)) < 1e-10 * scaleZ
    Test.@test abs(sum(ip.energy_transfer)) < 1e-8 * scaleE
    # (workspace-reuse allocation ratio asserted in test_allocs.jl)
    Test.@test_throws DimensionMismatch FIT.Spherical.calculate_spherical_transfer!(ws, ζscat[1:end-1])
end

# -----------------------------------------------------------------------
# Divergent horizontal-KE spectral transfer for the full (rotational + divergent) flow, on both the
# regular grid (FastSphericalHarmonics) and scattered points (NUFSHT). Anchored by: total-KE
# conservation Σ_l T ≈ 0 (skew-symmetric advection); exact reduction to the barotropic
# SphericalTransferMethod energy transfer for non-divergent flow (radius 1 and 2); the channel split
# T = T_rot + T_div (with T_div ≈ 0 for a rotational field); and FSH↔NUFSHT parity on one field.
Test.@testset "Divergent spherical transfer (rotational + divergent)" begin
    lmax = 10; N = lmax + 1
    rng = Random.MersenneTwister(11)
    _nabla(freal) = (ð = FSH.spinsph_eth(FSH.spinsph_transform(Matrix{Float64}(freal), 0), 0);
        G = FSH.spinsph_evaluate(ð, 1); -[complex(G[i, j][1], G[i, j][2]) for i in axes(G, 1), j in axes(G, 2)])
    _randc() = (C = zeros(Float64, N, 2N - 1); for l in 1:4, m in -l:l
            C[FSH.spinsph_mode(0, l, m)] = randn(rng) / (l + 1); end; C)
    _lapf(f) = FSH.spinsph_evaluate((C = FSH.spinsph_transform(Matrix{Float64}(f), 0); D = copy(C);
        for l in 0:lmax, m in -l:l; i = FSH.spinsph_mode(0, l, m); D[i] = -l * (l + 1) * C[i]; end; D), 0)
    ψf = FSH.spinsph_evaluate(_randc(), 0); χf = FSH.spinsph_evaluate(_randc(), 0)
    U = im .* _nabla(ψf) .+ _nabla(χf); uθ = real.(U); uφ = imag.(U)
    Ur = im .* _nabla(ψf); uθr = real.(Ur); uφr = imag.(Ur)
    ζr = _lapf(ψf)

    Test.@testset "FSH regular-grid" begin
        r = FIT.calculate_energy_transfer(FIT.Types.DivergentSphericalTransferMethod(), (uθ, uφ))
        Test.@test r isa FIT.Types.DivergentSphericalTransferResult
        Test.@test maximum(abs, r.energy_transfer .- (r.rotational_transfer .+ r.divergent_transfer)) < 1e-10
        Test.@test abs(sum(r.energy_transfer)) / maximum(abs, r.energy_transfer) < 1e-8
        Test.@test maximum(abs, r.divergent_transfer) > 1e-3
        rr = FIT.calculate_energy_transfer(FIT.Types.DivergentSphericalTransferMethod(), (uθr, uφr))
        TE = FIT.calculate_energy_transfer(FIT.Types.SphericalTransferMethod(), ζr).energy_transfer
        Test.@test maximum(abs, rr.energy_transfer .- TE) / maximum(abs, TE) < 1e-8
        Test.@test maximum(abs, rr.divergent_transfer) < 1e-9
        a = 2.0
        ra = FIT.calculate_energy_transfer(FIT.Types.DivergentSphericalTransferMethod(; radius = a), (uθr ./ a, uφr ./ a))
        TEa = FIT.calculate_energy_transfer(FIT.Types.SphericalTransferMethod(; radius = a), ζr ./ a^2).energy_transfer
        Test.@test maximum(abs, ra.energy_transfer .- TEa) / maximum(abs, TEa) < 1e-8
        ws = FIT.Spherical.DivergentSphericalTransferWorkspace(lmax)
        Test.@test sprint(show, ws) == "DivergentSphericalTransferWorkspace(…)"
        Test.@test_throws ArgumentError FIT.calculate_energy_transfer(FIT.Types.DivergentSphericalTransferMethod(), (uθ, uφ); spectral = SpectralBackends.NUFSHTSpectralBackend())
    end

    Test.@testset "NUFSHT scattered + FSH↔NUFSHT parity" begin
        Θ, Φ = FSH.sph_points(N); M = N * (2N - 1)
        θv = Vector{Float64}(undef, M); φv = Vector{Float64}(undef, M)
        for j in 1:2N-1, i in 1:N
            k = i + (j - 1) * N; θv[k] = Θ[i]; φv[k] = Φ[j]
        end
        rn = FIT.calculate_energy_transfer(FIT.Types.DivergentSphericalTransferMethod(), (vec(uθ), vec(uφ)),
                                            (θv, φv); lmax = lmax, dealias = false)
        Test.@test rn isa FIT.Types.DivergentSphericalTransferResult
        Test.@test abs(sum(rn.energy_transfer)) / maximum(abs, rn.energy_transfer) < 1e-6
        rf = FIT.calculate_energy_transfer(FIT.Types.DivergentSphericalTransferMethod(), (uθ, uφ); dealias = false)
        Test.@test maximum(abs, rf.energy_transfer .- rn.energy_transfer) / maximum(abs, rf.energy_transfer) < 1e-6
        wss = FIT.Spherical.ScatteredDivergentSphericalTransferWorkspace((θv, φv), lmax; dealias = false)
        Test.@test sprint(show, wss) == "ScatteredDivergentSphericalTransferWorkspace(…)"
        Test.@test_throws ArgumentError FIT.calculate_energy_transfer(FIT.Types.DivergentSphericalTransferMethod(),
            (vec(uθ), vec(uφ)), (θv, φv); lmax = lmax, spectral = SpectralBackends.FSHTSpectralBackend())
    end
end

# -----------------------------------------------------------------------
# Executable transform-axis backend matrix (anti-facade capstone). One data structure lists, per
# diagnostic, the SUPPORTED spectral backends (must return a valid result) and the REJECTED ones
# (must raise a clear `ArgumentError` — never a `MethodError`, never a silent result or misroute).
# A facade backend (dispatches nowhere / silently reroutes) fails both ways, so it cannot ship green.
Test.@testset "Backend matrix — transform axis (facade-proof)" begin
    Random.seed!(0)
    N = 8; L = 2π
    ks = FIT.Utils.wavenumber_grid((N, N), (L, L))
    kx = [ks[1][i] for i in 1:N, j in 1:N]; ky = [ks[2][j] for i in 1:N, j in 1:N]
    ψh = FFTW.fft(randn(N, N)) ./ N^2
    û = cat(im .* ky .* ψh, -im .* kx .* ψh; dims = 3)          # divergence-free 2D velocity coeffs
    ρ̂ = FFTW.fft(1.0 .+ 0.1 .* randn(N, N)) ./ N^2
    b = FIT.Types.LinearBinning(2π / L); bands = FIT.Types.SmoothBands([2.0, 4.0])
    DS = SpectralBackends.DirectSumSpectralBackend(); FTB = SpectralBackends.FFTSpectralBackend()
    NU = FIT.Types.FINUFFTBackend(); SH = SpectralBackends.FSHTSpectralBackend(); NS = SpectralBackends.NUFSHTSpectralBackend()

    # Cartesian Fourier-coefficient diagnostics: SUPPORTED = {DirectSum, FFT}; REJECTED = scattered/spherical.
    cart = (
        ("spectral_flux",  sp -> FIT.SpectralFlux.calculate_spectral_flux(û, ks; binning = b, spectral = sp)),
        ("shell_to_shell", sp -> FIT.ShellToShellTransfer.calculate_shell_to_shell_transfer(û, ks; binning = b, spectral = sp)),
        ("mode_to_mode",   sp -> FIT.calculate_mode_to_mode_transfer(û, ks; spectral = sp, force = true)),
        ("band_to_band",   sp -> FIT.BandTransfer.calculate_band_to_band_transfer(û, ks; bands = bands, spectral = sp)),
        ("compressible",   sp -> FIT.Compressible.calculate_compressible_flux(û, ρ̂, ks; spectral = sp)),
        ("partial_fluxes", sp -> FIT.calculate_partial_fluxes(û, ks; decomposition = FIT.Types.HelmholtzDecomposition(), spectral = sp)),
    )
    Test.@testset "Cartesian coefficient diagnostics" begin
        for (name, invoke) in cart
            for sp in (DS, FTB)
                Test.@test invoke(sp) !== nothing
            end
            for sp in (NU, SH, NS)
                Test.@test_throws ArgumentError invoke(sp)
            end
        end
    end

    # Spherical diagnostics: SUPPORTED = {nothing, the geometry's own backend}; REJECTED = the others.
    lm = 4; Ng = lm + 1
    ζmat = randn(Ng, 2Ng - 1); uθm = randn(Ng, 2Ng - 1); uφm = randn(Ng, 2Ng - 1)
    Mp = 120; θs = Vector{Float64}(undef, Mp); φs = Vector{Float64}(undef, Mp); ga = π * (3 - sqrt(5))
    for i in 0:Mp-1
        θs[i+1] = acos(clamp(1 - 2 * (i + 0.5) / Mp, -1, 1)); φs[i+1] = mod(ga * i, 2π)
    end
    ζv = randn(Mp); uθv = randn(Mp); uφv = randn(Mp)
    sph = (
        ("spherical(regular)",   :SH, sp -> FIT.calculate_energy_transfer(FIT.Types.SphericalTransferMethod(), ζmat; spectral = sp)),
        ("divergent(regular)",   :SH, sp -> FIT.calculate_energy_transfer(FIT.Types.DivergentSphericalTransferMethod(), (uθm, uφm); spectral = sp)),
        ("spherical(scattered)", :NS, sp -> FIT.calculate_energy_transfer(FIT.Types.SphericalTransferMethod(), ζv, (θs, φs); lmax = lm, dealias = false, spectral = sp)),
        ("divergent(scattered)", :NS, sp -> FIT.calculate_energy_transfer(FIT.Types.DivergentSphericalTransferMethod(), (uθv, uφv), (θs, φs); lmax = lm, dealias = false, spectral = sp)),
    )
    Test.@testset "Spherical diagnostics" begin
        for (name, geom, invoke) in sph
            ok_backends  = geom === :SH ? (nothing, SH) : (nothing, NS)
            bad_backends = geom === :SH ? (FTB, NU, NS) : (FTB, NU, SH)
            for sp in ok_backends
                Test.@test invoke(sp) !== nothing
            end
            for sp in bad_backends
                Test.@test_throws ArgumentError invoke(sp)
            end
        end
    end
end

# -----------------------------------------------------------------------
# Element-type genericity: every public entry point is `eltype`-derived, so a Float32
# field must flow through end-to-end as Float32 (no silent Float64 promotion). Pins the
# genericity contract across the whole transfer surface, not just the incompressible core.
Test.@testset "Float32 genericity — outputs preserve input eltype" begin
    Random.seed!(1)
    N = 16; L = 2π
    for T in (Float64, Float32)
        ks0 = FIT.Utils.wavenumber_grid((N, N), (L, L))
        ks  = ntuple(d -> T.(ks0[d]), 2)
        kx  = [ks[1][i] for i in 1:N, j in 1:N]
        ky  = [ks[2][j] for i in 1:N, j in 1:N]
        ψh  = FFTW.fft(randn(N, N)) ./ N^2
        û   = Complex{T}.(cat(im .* ky .* ψh, -im .* kx .* ψh; dims = 3))
        ρ̂   = Complex{T}.(FFTW.fft(1.0 .+ 0.1 .* randn(N, N)) ./ N^2)
        b   = FIT.Types.LinearBinning(T(2π / L))

        r1 = FIT.SpectralFlux.calculate_spectral_flux(û, ks; binning = b, spectral = SpectralBackends.FFTSpectralBackend())
        r2 = FIT.ShellToShellTransfer.calculate_shell_to_shell_transfer(û, ks; binning = b, spectral = SpectralBackends.FFTSpectralBackend())
        r3 = FIT.calculate_mode_to_mode_transfer(û, ks)
        r4 = FIT.BandTransfer.calculate_band_to_band_transfer(û, ks; bands = FIT.Types.SmoothBands(T[2, 4, 6]))
        r5 = FIT.calculate_partial_fluxes(û, ks; binning = b, decomposition = FIT.Types.HelmholtzDecomposition())
        r6 = FIT.Compressible.calculate_compressible_flux(û, ρ̂, ks; binning = b)

        Test.@test eltype(r1.transfer_spectrum)     === T
        Test.@test eltype(r2.transfer_matrix)        === T
        Test.@test eltype(r3.transfer)               === T
        Test.@test eltype(r4.transfer_matrix)        === T
        Test.@test eltype(r5.total.transfer_spectrum) === T
        Test.@test eltype(r6.transfer_spectrum)      === T
    end
end

# -----------------------------------------------------------------------
# Distributed (MPI) extensions — launched single-machine via MPI's bundled mpiexec.
# Batch axis: independent snapshots gathered/reduced across ranks.
# Pencil axis: one grid split across ranks (PencilFFTs transpose-based FFT) vs serial.
Test.@testset "MPI distributed (mpiexec -n 2)" begin
    mpiexec_path = MPI.mpiexec()
    proj = Base.active_project()
    for script in ("batch_test.jl", "pencil_test.jl")
        path = joinpath(@__DIR__, "mpi", script)
        cmd  = `$(mpiexec_path) -n 2 $(Base.julia_cmd()) --project=$(proj) $(path)`
        p = run(pipeline(ignorestatus(cmd); stdout = stdout, stderr = stderr))
        Test.@test success(p)
    end
end

# -----------------------------------------------------------------------
Test.@testset "Aqua Code Quality" begin
    Aqua.test_all(FIT; ambiguities = true, stale_deps = (ignore=[:Documenter],))
end
