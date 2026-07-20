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
using FlowFieldSpectra: FlowFieldSpectra
using FastSphericalHarmonics: FastSphericalHarmonics as FSH
using NUFSHT: NUFSHT

using FlowInvariantTransfer: FlowInvariantTransfer as FIT
using JLArrays: JLArrays   # reference GPU-array backend: allowscalar(false) → catches non-device-generic code

Test.@testset "FlowInvariantTransfer.jl Test Suite" begin

    # -----------------------------------------------------------------------
    Test.@testset "Aqua Code Quality" begin
        Aqua.test_all(FIT; ambiguities = true, stale_deps = (ignore=[:Documenter],))
    end

    # -----------------------------------------------------------------------
    Test.@testset "Utils — wavenumber_grid" begin
        N = 8
        L = 2π
        ks = FIT.Utils.wavenumber_grid((N, N), (L, L))
        Test.@test length(ks) == 2
        Test.@test length(ks[1]) == N
        # FFTW order: ks[1] should contain 0, 1, 2, 3, -4, -3, -2, -1 * (2π/L)
        dk = 2π / L
        Test.@test isapprox(ks[1][1], 0.0)
        Test.@test isapprox(ks[1][2], dk, atol=1e-14)
        Test.@test isapprox(ks[1][N], -dk, atol=1e-14)

        k_mag = FIT.Utils.wavenumber_magnitude_grid(ks)
        Test.@test size(k_mag) == (N, N)
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
        b = FIT.LinearBinning(1.0)
        edges = FIT.ShellBinning.shell_edges(b, 5.0)
        Test.@test edges[1] == 0.0
        Test.@test edges[end] >= 5.0
        centers = FIT.ShellBinning.shell_centers(b, 5.0)
        Test.@test length(centers) == length(edges) - 1
        Test.@test all(diff(centers) .> 0)
    end

    Test.@testset "ShellBinning — LogarithmicBinning" begin
        b = FIT.LogarithmicBinning(1.0, 2.0)
        edges = FIT.ShellBinning.shell_edges(b, 16.0)
        Test.@test edges[1] == 1.0
        Test.@test issorted(edges)
        Test.@test all(edges[2:end] ./ edges[1:end-1] .≈ 2.0)
    end

    Test.@testset "ShellBinning — DyadicBinning vs LogarithmicBinning(2)" begin
        k_max = 16.0
        b_d = FIT.DyadicBinning(1.0)
        b_l = FIT.LogarithmicBinning(1.0, 2.0)
        Test.@test FIT.ShellBinning.shell_edges(b_d, k_max) == FIT.ShellBinning.shell_edges(b_l, k_max)
    end

    Test.@testset "ShellBinning — CustomBinning" begin
        edges = [0.0, 1.0, 3.0, 6.0, 10.0]
        b = FIT.CustomBinning(edges)
        Test.@test FIT.ShellBinning.shell_edges(b, 10.0) == edges
        Test.@test FIT.ShellBinning.n_shells(b, 10.0) == 4
    end

    Test.@testset "ShellBinning — assign_shells" begin
        ks = FIT.Utils.wavenumber_grid((8,), (2π,))
        k_mag_1d = FIT.Utils.wavenumber_magnitude_grid(ks)
        b = FIT.LinearBinning(2π / 8)
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
        Test.@test FIT.Filters.filter_response(FIT.SharpSpectralFilter(), k, ℓ) == 1.0
        Test.@test FIT.Filters.filter_response(FIT.SharpSpectralFilter(), 4.0, ℓ) == 0.0
        # GaussianFilter: always in (0,1], decays with k
        g1 = FIT.Filters.filter_response(FIT.GaussianFilter(), k, ℓ)
        g2 = FIT.Filters.filter_response(FIT.GaussianFilter(), 4.0, ℓ)
        Test.@test 0.0 < g2 < g1 <= 1.0
        # TopHatFilter: sinc, = 1 at k=0
        Test.@test FIT.Filters.filter_response(FIT.TopHatFilter(), 0.0, ℓ) ≈ 1.0
    end

    Test.@testset "Filters — apply_filter_spectral!" begin
        k_mag = Float64[0, 1, 2, 3, 4]
        û_in  = ComplexF64[1.0, 1.0, 1.0, 1.0, 1.0]
        û_out = similar(û_in)
        FIT.Filters.apply_filter_spectral!(û_out, û_in, k_mag, FIT.SharpSpectralFilter(), 1.0)
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

        result = FIT.calculate_spectral_flux(û, ks;
            binning = FIT.LinearBinning(2π/L), dealiasing = FIT.NoDealiasing())

        Test.@test result isa FIT.SpectralFluxResult
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
        result_direct = FIT.calculate_spectral_flux(û_phys, ks;
            binning=FIT.LinearBinning(2π/L), dealiasing = FIT.NoDealiasing(), spectral=FIT.DirectSumBackend())

        # FFTW path (extension)
        result_fft = FIT.calculate_spectral_flux(û_phys, ks;
            binning=FIT.LinearBinning(2π/L), dealiasing = FIT.NoDealiasing(), spectral=FIT.FFTBackend())

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

        for spectral in (FIT.DirectSumBackend(), FIT.FFTBackend())
            N̂ = FIT.NonlinearTerm.compute_nonlinear_term(û, ks; dealiasing = FIT.OrszagTwoThirds(), spectral = spectral)
            t = FIT.Invariants.transfer_density(FIT.KineticEnergy(), û, N̂, ks)
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
        r = FIT.calculate_spectral_flux(û, ks; binning = FIT.LinearBinning(2π/L), dealiasing = FIT.NoDealiasing())
        Test.@test isapprox(r.flux, cumsum(r.transfer_spectrum); atol = 1e-12)
    end

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
        N̂2 = FIT.NonlinearTerm.compute_nonlinear_term(û2, ks2; dealiasing = FIT.OrszagTwoThirds(), spectral=FIT.FFTBackend())
        tΩ2 = FIT.Invariants.transfer_density(FIT.Enstrophy(), û2, N̂2, ks2)
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
        res3 = FIT.calculate_spectral_flux(û3, ks3; binning = FIT.LinearBinning(1.0),
            invariant = FIT.Enstrophy(), dealiasing = FIT.OrszagTwoThirds(), spectral=FIT.FFTBackend())
        Test.@test res3 isa FIT.SpectralFluxResult
        Test.@test all(isfinite, res3.transfer_spectrum)
        # mode-to-mode aggregates now route through the FFT paths, so 3D enstrophy net works
        m2m3 = FIT.calculate_mode_to_mode_transfer(û3, ks3; invariant = FIT.Enstrophy(), spectral=FIT.FFTBackend())
        Test.@test m2m3 isa FIT.ModeToModeTriadResult
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
        N̂_dir = FIT.NonlinearTerm.compute_nonlinear_term(θ̂, ks; dealiasing = FIT.OrszagTwoThirds(),
            spectral = FIT.DirectSumBackend(), advecting_hat = û)
        N̂_fft = FIT.NonlinearTerm.compute_nonlinear_term(θ̂, ks; dealiasing = FIT.OrszagTwoThirds(),
            spectral = FIT.FFTBackend(), advecting_hat = û)
        Test.@test size(N̂_dir) == (N, N, 1)
        Test.@test isapprox(N̂_dir, N̂_fft; atol = 1e-10 * maximum(abs, N̂_fft))

        # Scalar variance conservation: Σ_k Re{θ̂*(k) N̂_θ(k)} ≈ 0.
        tθ = FIT.Invariants.transfer_density(FIT.PassiveScalar(), θ̂, N̂_fft, ks)
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

        N23  = FIT.NonlinearTerm.compute_nonlinear_term(û, ks; dealiasing = FIT.OrszagTwoThirds(),   spectral = FIT.FFTBackend())
        Npad = FIT.NonlinearTerm.compute_nonlinear_term(û, ks; dealiasing = FIT.PaddedThreeHalves(), spectral = FIT.FFTBackend())
        low = repeat(kmag .< 8, 1, 1, 2)
        mid = repeat((kmag .>= 8) .& (kmag .< 12), 1, 1, 2)
        Test.@test maximum(abs.(N23 .- Npad)[low]) < 1e-10 * maximum(abs.(Npad)[low])   # agree on |k|<N/3
        Test.@test sum(abs2, Npad[mid]) > 3 * sum(abs2, N23[mid])                       # padded keeps more

        # Padded spectral flux still conserves for an incompressible field.
        b  = FIT.LinearBinning(2π/L)
        sf = FIT.calculate_spectral_flux(û, ks; binning = b, dealiasing = FIT.PaddedThreeHalves(), spectral = FIT.FFTBackend())
        Test.@test abs(sum(sf.transfer_spectrum)) < 1e-9 * sum(abs, sf.transfer_spectrum)

        # Padding requires the FFT path; the dependency-free DirectSumBackend errors clearly.
        Test.@test_throws ArgumentError FIT.NonlinearTerm.compute_nonlinear_term(û, ks;
            dealiasing = FIT.PaddedThreeHalves(), spectral = FIT.DirectSumBackend())
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
        b  = FIT.LinearBinning(2π/L)

        res = FIT.calculate_scalar_flux(û, θ, ks; binning = b, dealiasing = FIT.OrszagTwoThirds(),
            spectral = FIT.FFTBackend())
        Test.@test res isa FIT.SpectralFluxResult
        sT = sqrt(sum(abs2, res.transfer_spectrum)); Test.@test sT > 0
        Test.@test abs(sum(res.transfer_spectrum)) < 1e-9 * sT       # variance conserved
        Test.@test abs(res.flux[end]) < 1e-9 * sT                    # cumulative flux returns to 0

        # Passing the scalar already shaped (N,N,1) gives the identical result.
        res1 = FIT.calculate_scalar_flux(û, reshape(θ, N, N, 1), ks; binning = b,
            dealiasing = FIT.OrszagTwoThirds(), spectral = FIT.FFTBackend())
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

        result = FIT.calculate_shell_to_shell_transfer(û, ks;
            binning=FIT.LinearBinning(2π/L), dealiasing = FIT.OrszagTwoThirds(),
            verify_antisymmetry=true, spectral=FIT.FFTBackend())

        Test.@test result isa FIT.ShellToShellResult
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
        b  = FIT.LinearBinning(2π/L)

        r_direct = FIT.calculate_shell_to_shell_transfer(û, ks; binning=b, dealiasing = FIT.OrszagTwoThirds(),
            verify_antisymmetry=true, spectral=FIT.DirectSumBackend())
        r_fft = FIT.calculate_shell_to_shell_transfer(û, ks; binning=b, dealiasing = FIT.OrszagTwoThirds(),
            verify_antisymmetry=true, spectral=FIT.FFTBackend())

        T_norm = sqrt(sum(abs2, r_direct.transfer_matrix))
        Test.@test T_norm > 0                                            # non-degenerate
        # serial and FFT implement the SAME (u·∇)u_m form → agree to roundoff
        Test.@test isapprox(r_direct.transfer_matrix, r_fft.transfer_matrix; atol = 1e-9 * T_norm)
        Test.@test r_direct.max_antisymmetry_error < 1e-10 * T_norm      # A is antisymmetric

        # Reduction: Σ_m T(n,m) must equal the spectral transfer T(k) (same field/binning).
        sf = FIT.calculate_spectral_flux(û, ks; binning = b, dealiasing = FIT.OrszagTwoThirds())
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
        b  = FIT.LinearBinning(2π/L)

        r_dir = FIT.calculate_scalar_shell_to_shell_transfer(û, θ, ks; binning = b,
            dealiasing = FIT.OrszagTwoThirds(), verify_antisymmetry = true, spectral = FIT.DirectSumBackend())
        r_fft = FIT.calculate_scalar_shell_to_shell_transfer(û, θ, ks; binning = b,
            dealiasing = FIT.OrszagTwoThirds(), verify_antisymmetry = true, spectral = FIT.FFTBackend())

        T_norm = sqrt(sum(abs2, r_dir.transfer_matrix))
        Test.@test T_norm > 0                                                  # non-degenerate
        Test.@test isapprox(r_dir.transfer_matrix, r_fft.transfer_matrix; atol = 1e-9 * T_norm)
        Test.@test r_dir.max_antisymmetry_error < 1e-10 * T_norm               # T_θ antisymmetric
        # Reduction: Σ_m T_θ(n,m) == scalar transfer spectrum T_θ(k).
        sfθ = FIT.calculate_scalar_flux(û, θ, ks; binning = b, dealiasing = FIT.OrszagTwoThirds(),
            spectral = FIT.FFTBackend())
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

        m2m = FIT.calculate_scalar_mode_to_mode_transfer(û, θ, ks; dealiasing = FIT.OrszagTwoThirds(),
            spectral = FIT.FFTBackend())
        S   = m2m.transfer
        nrm = sqrt(sum(abs2, S)); Test.@test nrm > 0
        asym = 0.0
        for k in CartesianIndices((N, N)), p in CartesianIndices((N, N))
            asym = max(asym, abs(S[k, p] + S[p, k]))
        end
        Test.@test asym < 1e-10 * nrm                              # S_θ(k|p) = −S_θ(p|k)
        Test.@test abs(sum(S)) < 1e-10 * nrm                       # conserves
        # net (= Σ_p S_θ) shell-summed == scalar transfer spectrum
        b = FIT.LinearBinning(2π/L)
        sfθ = FIT.calculate_scalar_flux(û, θ, ks; binning = b, dealiasing = FIT.OrszagTwoThirds(),
            spectral = FIT.FFTBackend())
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
        Test.@test FIT.ShellBinning.shell_coordinate(FIT.IsotropicShells(), ks) ≈ FIT.Utils.wavenumber_magnitude_grid(ks)
        Test.@test FIT.ShellBinning.shell_coordinate(FIT.PerpendicularShells(), ks) ≈ sqrt.(kx.^2 .+ ky.^2)
        Test.@test FIT.ShellBinning.shell_coordinate(FIT.ParallelShells(), ks)      ≈ abs.(kz)

        # Divergence-free 3D velocity û = i k × Â (non-degenerate after dealiasing at M=16)
        Random.seed!(41)
        Â = randn(ComplexF64, M, M, M, 3)
        ûx = im .* (ky .* Â[:, :, :, 3] .- kz .* Â[:, :, :, 2])
        ûy = im .* (kz .* Â[:, :, :, 1] .- kx .* Â[:, :, :, 3])
        ûz = im .* (kx .* Â[:, :, :, 2] .- ky .* Â[:, :, :, 1])
        û3 = cat(ûx, ûy, ûz; dims = 4)
        b  = FIT.LinearBinning(2π/L)

        r_def  = FIT.calculate_spectral_flux(û3, ks; binning=b, dealiasing = FIT.OrszagTwoThirds(), spectral=FIT.FFTBackend())
        r_iso  = FIT.calculate_spectral_flux(û3, ks; binning=b, dealiasing = FIT.OrszagTwoThirds(),
            spectral=FIT.FFTBackend(), geometry=FIT.IsotropicShells())
        r_perp = FIT.calculate_spectral_flux(û3, ks; binning=b, dealiasing = FIT.OrszagTwoThirds(),
            spectral=FIT.FFTBackend(), geometry=FIT.PerpendicularShells())
        r_par  = FIT.calculate_spectral_flux(û3, ks; binning=b, dealiasing = FIT.OrszagTwoThirds(),
            spectral=FIT.FFTBackend(), geometry=FIT.ParallelShells())

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

        dec = FIT.Decomposition.decompose_field(FIT.HelicalDecomposition(), û, ks)
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
        b = FIT.LinearBinning(2π/L)
        rtot = FIT.calculate_spectral_flux(û, ks; binning=b, dealiasing = FIT.OrszagTwoThirds(), spectral=FIT.FFTBackend())
        rhel = FIT.calculate_spectral_flux(û, ks; binning=b, dealiasing = FIT.OrszagTwoThirds(), spectral=FIT.FFTBackend(),
            decomposition=FIT.HelicalDecomposition())
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
        b  = FIT.LinearBinning(2π/L)

        hp = FIT.calculate_helical_partial_fluxes(û, ks; binning=b, dealiasing=FIT.OrszagTwoThirds(),
            spectral=FIT.FFTBackend())
        Test.@test length(hp.channels) == 8                              # all (s_k,s_p,s_q) present
        sf = FIT.calculate_spectral_flux(û, ks; binning=b, dealiasing=FIT.OrszagTwoThirds(), spectral=FIT.FFTBackend())
        # The 8 channels reconstruct the full KE flux. The individual channels are large and partly
        # cancel into a smaller total, so compare at a tolerance set by the CHANNEL scale.
        chan_scale = maximum(sqrt(sum(abs2, c.transfer_spectrum)) for c in values(hp.channels))
        Test.@test chan_scale > 0
        Test.@test isapprox(hp.total.transfer_spectrum, sf.transfer_spectrum; atol = 1e-10 * chan_scale)
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
        b  = FIT.LinearBinning(2π/L)

        hp = FIT.calculate_partial_fluxes(û, ks; decomposition=FIT.HelmholtzDecomposition(),
            binning=b, dealiasing=FIT.OrszagTwoThirds(), spectral=FIT.FFTBackend())
        Test.@test length(hp.channels) == 8
        sf = FIT.calculate_spectral_flux(û, ks; binning=b, dealiasing=FIT.OrszagTwoThirds(), spectral=FIT.FFTBackend())
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

        dec = FIT.Decomposition.decompose_field(FIT.ToroidalPoloidalDecomposition(), û, ks)
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
        dec = FIT.Decomposition.decompose_field(FIT.HelmholtzDecomposition(), û, ks)
        rot = dec.rotational; div = dec.divergent
        Test.@test isapprox(rot .+ div, û; atol = 1e-12 * maximum(abs, û))                # reconstruction
        drot = kx .* rot[:,:,:,1] .+ ky .* rot[:,:,:,2] .+ kz .* rot[:,:,:,3]             # rot: k·û_rot ≈ 0
        Test.@test maximum(abs, drot) < 1e-10 * maximum(abs, û)
        cx = ky .* div[:,:,:,3] .- kz .* div[:,:,:,2]                                     # div: k×û_div ≈ 0
        cy = kz .* div[:,:,:,1] .- kx .* div[:,:,:,3]
        cz = kx .* div[:,:,:,2] .- ky .* div[:,:,:,1]
        Test.@test max(maximum(abs, cx), maximum(abs, cy), maximum(abs, cz)) < 1e-10 * maximum(abs, û)
        # RotationalDecomposition / DivergentDecomposition return the single corresponding component.
        Test.@test FIT.Decomposition.decompose_field(FIT.RotationalDecomposition(), û, ks) ≈ rot
        Test.@test FIT.Decomposition.decompose_field(FIT.DivergentDecomposition(), û, ks) ≈ div
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
        bands = FIT.SmoothBands(centers; logwidth = 0.5)

        r = FIT.calculate_band_to_band_transfer(û, ks; bands = bands, dealiasing = FIT.OrszagTwoThirds(),
            spectral = FIT.FFTBackend())
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
        result = FIT.calculate_coarse_graining_flux(
            (u, v), (x, y), π/2, FIT.GaussianFilter())
        Test.@test result isa FIT.CoarseGrainingFluxResult
    end

    # -----------------------------------------------------------------------
    Test.@testset "CoarseGrainingFlux — 3D Cartesian Π_ℓ + diagnostics == CGEF directly" begin
        FT = Float64; N = 12; L = 2π
        xs = collect(range(0, L; length = N + 1)[1:N]); ys = copy(xs); zs = copy(xs)
        Random.seed!(29)
        u = randn(N, N, N); v = randn(N, N, N); w = randn(N, N, N); ℓ = L / 4
        r = FIT.calculate_coarse_graining_flux((u, v, w), (xs, ys, zs), ℓ, FIT.GaussianFilter())
        Test.@test r isa FIT.CoarseGrainingFluxResult
        Test.@test size(r.flux_field) == (N, N, N)
        Test.@test all(isfinite, r.flux_field)
        # Exact match to the sibling's true-3D compute_Π! on the same field/grid (wiring correctness).
        dx = FT((xs[end] - xs[begin]) / (N - 1))
        geom = CoarseGrainingEnergyFluxes.Geometry.CartesianGeometry(dx, dx, dx)
        grid = CoarseGrainingEnergyFluxes.Grids.StructuredGrid(geom, FT.(xs), FT.(ys), FT.(zs), trues(N, N, N))
        Πref = zeros(FT, N, N, N)
        CoarseGrainingEnergyFluxes.Diagnostics.compute_Π!(Πref, u, v, w, grid,
            CoarseGrainingEnergyFluxes.Kernels.GaussianKernel(), FT(ℓ))
        Test.@test maximum(abs, r.flux_field .- Πref) == 0.0
        # 3D diagnostics: (ns, 3, 3) symmetric stress/strain tensors.
        rd = FIT.calculate_coarse_graining_flux((u, v, w), (xs, ys, zs), ℓ, FIT.GaussianFilter(); return_diagnostics = true)
        Test.@test size(rd.stress_tensor) == (N, N, N, 3, 3)
        Test.@test maximum(abs, rd.stress_tensor .- permutedims(rd.stress_tensor, (1,2,3,5,4))) == 0.0
        Test.@test maximum(abs, rd.strain_rate  .- permutedims(rd.strain_rate,  (1,2,3,5,4))) == 0.0
    end

    # -----------------------------------------------------------------------
    Test.@testset "CoarseGrainingFlux — in-place !() + diagnostics + plan cache" begin
        Random.seed!(7)
        N = 32; L = 2π
        xs = collect(range(0, L; length = N + 1)[1:N]); ys = copy(xs)
        u  = [sin(xs[i]) * cos(ys[j]) + 0.2 * randn() for i in 1:N, j in 1:N]
        v  = [-cos(xs[i]) * sin(ys[j]) + 0.2 * randn() for i in 1:N, j in 1:N]
        filt = FIT.GaussianFilter(); ℓ = 0.5

        # in-place matches the allocating path to machine precision (same computation)
        r0 = FIT.calculate_coarse_graining_flux((u, v), (xs, ys), ℓ, filt)
        ws = FIT.CoarseGrainingFluxWorkspace((u, v), (xs, ys), filt)
        r1 = FIT.calculate_coarse_graining_flux!(ws, (u, v), ℓ, filt)
        Test.@test r1 isa FIT.CoarseGrainingFluxResult
        Test.@test maximum(abs.(r0.flux_field .- r1.flux_field)) == 0.0
        Test.@test r0.mean_flux == r1.mean_flux

        # return_diagnostics=true works end-to-end (regression: the diagnostics result type used to
        # over-constrain the 4-D τ̄/S̄ to the 2-D flux array type, so this path errored for everyone).
        r0d = FIT.calculate_coarse_graining_flux((u, v), (xs, ys), ℓ, filt; return_diagnostics = true)
        Test.@test r0d isa FIT.CoarseGrainingFluxResultWithDiagnostics
        Test.@test size(r0d.stress_tensor) == (N, N, 2, 2)
        wsd = FIT.CoarseGrainingFluxWorkspace((u, v), (xs, ys), filt; return_diagnostics = true)
        r1d = FIT.calculate_coarse_graining_flux!(wsd, (u, v), ℓ, filt)
        Test.@test maximum(abs.(r0d.stress_tensor .- r1d.stress_tensor)) == 0.0
        Test.@test maximum(abs.(r0d.strain_rate  .- r1d.strain_rate))  == 0.0

        # Filter-scale sweep reusing one workspace matches per-call allocation (plan rebuilt per ℓ).
        for l in (0.3, 0.5, 0.8, 1.2)
            ra = FIT.calculate_coarse_graining_flux((u, v), (xs, ys), l, filt)
            rb = FIT.calculate_coarse_graining_flux!(ws, (u, v), l, filt)
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
            FIT.SpectralFluxMethod(FIT.LinearBinning(2π/L)), û, ks)
        Test.@test r1 isa FIT.SpectralFluxResult

        r2 = FIT.calculate_energy_transfer(
            FIT.ShellToShellTransferMethod(FIT.LinearBinning(2π/L)), û, ks)
        Test.@test r2 isa FIT.ShellToShellResult

        x = [L * (i-1) / N for i in 1:N]
        y = [L * (j-1) / N for j in 1:N]
        u = zeros(N, N); v = zeros(N, N)
        r3 = FIT.calculate_energy_transfer(
            FIT.CoarseGrainingFluxMethod(FIT.GaussianFilter(), Float64(π/2)),
            (u, v), (x, y))
        Test.@test r3 isa FIT.CoarseGrainingFluxResult
    end

    # -----------------------------------------------------------------------
    Test.@testset "assign_shells" begin
        ks  = FIT.Utils.wavenumber_grid((8,), (2π,))
        k_mag = FIT.Utils.wavenumber_magnitude_grid(ks)
        b     = FIT.LinearBinning(2π/8)
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
        b   = FIT.LinearBinning(2π/L)
        ws  = FIT.SpectralFluxWorkspace(û, ks, b)
        k_mag     = FIT.Utils.wavenumber_magnitude_grid(ks)
        edges     = FIT.ShellBinning.shell_edges(b, maximum(k_mag))
        centers   = FIT.ShellBinning.shell_centers(b, maximum(k_mag))
        shell_idx = FIT.ShellBinning.assign_shells(k_mag, edges)
        result    = FIT.SpectralFluxResult(centers, similar(ws.T_spec), similar(ws.flux))
        FIT.calculate_spectral_flux!(result, ws, û, ks, shell_idx; dealiasing = FIT.NoDealiasing())
        Test.@test result isa FIT.SpectralFluxResult
        Test.@test all(abs.(result.transfer_spectrum) .< 1e-14)
    end

    # -----------------------------------------------------------------------
    Test.@testset "ShellToShellTransfer !-variant" begin
        N = 6; L = 2π
        ks = FIT.Utils.wavenumber_grid((N, N), (L, L))
        û  = zeros(ComplexF64, N, N, 2)
        b   = FIT.LinearBinning(2π/L)
        ws  = FIT.ShellToShellWorkspace(û, ks, b)
        k_mag   = FIT.Utils.wavenumber_magnitude_grid(ks)
        edges   = FIT.ShellBinning.shell_edges(b, maximum(k_mag))
        centers = FIT.ShellBinning.shell_centers(b, maximum(k_mag))
        N_sh    = length(centers)
        FT      = Float64
        result  = FIT.ShellToShellResult(
            centers, edges,
            Matrix{FT}(undef, N_sh, N_sh),
            Vector{FT}(undef, N_sh),
            FT(NaN),
        )
        FIT.calculate_shell_to_shell_transfer!(result, ws, û, ks;
            dealiasing = FIT.NoDealiasing(), verify_antisymmetry=false)
        Test.@test result isa FIT.ShellToShellResult
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
        b     = FIT.LinearBinning(Float32(2π) / N)
        edges = FIT.ShellBinning.shell_edges(b, maximum(k_mag))
        Test.@test eltype(edges) == Float32
        centers = FIT.ShellBinning.shell_centers(b, maximum(k_mag))
        Test.@test eltype(centers) == Float32

        û = zeros(ComplexF32, N, N, 2)
        result = FIT.calculate_spectral_flux(û, ks;
            binning=b, dealiasing = FIT.NoDealiasing(), spectral=FIT.DirectSumBackend())
        Test.@test result isa FIT.SpectralFluxResult
        Test.@test eltype(result.k_shells) == Float32
        Test.@test eltype(result.transfer_spectrum) == Float32

        # New diagnostics also preserve Float32 throughout.
        θ̂ = zeros(ComplexF32, N, N)
        rθ = FIT.calculate_scalar_flux(û, θ̂, ks; binning=b, dealiasing = FIT.NoDealiasing(), spectral=FIT.DirectSumBackend())
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
        res = FIT.triadic_orthogonal_decomposition(X; dt=dt_sig, isreal_data=true)
        Test.@test res isa FIT.TriadicOrthogonalDecompositionResult
        Test.@test res.frequencies isa AbstractVector
        Test.@test all(res.mode_bispectrum .>= 0.0 .|| isnan.(res.mode_bispectrum))
        Test.@test all(res.modal_energy_budget .>= 0.0 .|| res.modal_energy_budget .<= 0.0 .|| isnan.(res.modal_energy_budget))

        # Check default dispatch via calculate_energy_transfer
        method = FIT.TriadicOrthogonalDecompositionMethod(nfft=64, noverlap=32, nmode=2)
        res_dispatch = FIT.calculate_energy_transfer(method, X; dt=dt_sig)
        Test.@test res_dispatch isa FIT.TriadicOrthogonalDecompositionResult
        Test.@test size(res_dispatch.mode_bispectrum, 3) == 2

        # 4. FFTBackend consistency
        res_serial = FIT.triadic_orthogonal_decomposition(X; dt=dt_sig, spectral=FIT.DirectSumBackend())
        res_fft = FIT.triadic_orthogonal_decomposition(X; dt=dt_sig, spectral=FIT.FFTBackend())
        Test.@test isapprox(res_serial.frequencies, res_fft.frequencies)
        Test.@test isapprox(filter(!isnan, res_serial.mode_bispectrum), filter(!isnan, res_fft.mode_bispectrum); atol=1e-12)

        # 5. ThreadedBackend — OhMyThreads is loaded so it should work
        res_threaded = FIT.triadic_orthogonal_decomposition(X; dt=dt_sig, execution=FIT.ThreadedBackend())
        Test.@test res_threaded isa FIT.TriadicOrthogonalDecompositionResult
        Test.@test isapprox(res_serial.frequencies, res_threaded.frequencies)

        # 6. Coefficients and auxiliary modes
        res_aux = FIT.triadic_orthogonal_decomposition(X; dt=dt_sig, return_coefficients=true, return_auxiliary_modes=true)
        Test.@test res_aux.expansion_coefficients isa Dict
        Test.@test res_aux.auxiliary_modes isa Dict

        # 7. Precision genericity: Float32 snapshots stay in Float32 (no ComplexF64 upcast).
        res32 = FIT.triadic_orthogonal_decomposition(Float32.(X); dt=Float32(dt_sig))
        Test.@test res32 isa FIT.TriadicOrthogonalDecompositionResult
        Test.@test eltype(res32.mode_bispectrum) === Float32
        Test.@test isapprox(Float64.(res32.frequencies), res_serial.frequencies; atol=1e-4)

        # 8. In-place workspace form: bit-identical to the allocating path (same buffers/math), and a
        # repeat call on a same-shaped snapshot reuses Q_hat / DFT plan / SVD scratch / L / T_budget.
        ws = FIT.TODWorkspace(X; dt=dt_sig, spectral=FIT.FFTBackend())
        ip = FIT.triadic_orthogonal_decomposition!(ws, X)
        Test.@test ip isa FIT.TriadicOrthogonalDecompositionResult
        Test.@test isequal(ip.mode_bispectrum, res_fft.mode_bispectrum)   # NaN-aware; exact
        for k in keys(res_fft.modes)
            Test.@test ip.modes[k].convective == res_fft.modes[k].convective
        end
        # (workspace-reuse allocation ratio asserted in test_allocs.jl)
        Test.@test_throws DimensionMismatch FIT.triadic_orthogonal_decomposition!(ws, X[:, :, 1:end-1])
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
        r = FIT.triadic_orthogonal_decomposition(X; dt=0.05, window=FIT.TriadicOrthogonalDecomposition.hann_window(64), noverlap=32)
        Test.@test r isa FIT.TriadicOrthogonalDecompositionResult
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
        rd = FIT.triadic_orthogonal_decomposition(Xl; window=nfft, noverlap=0, nmode=1, dt=dt,
            isreal_data=true, spectral=FIT.DirectSumBackend())
        rf = FIT.triadic_orthogonal_decomposition(Xl; window=nfft, noverlap=0, nmode=1, dt=dt,
            isreal_data=true, spectral=FIT.FFTBackend())
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
        ru = FIT.triadic_orthogonal_decomposition(Xu; window=nfft, noverlap=0, nmode=1, dt=dt,
            isreal_data=true, spectral=FIT.FFTBackend())
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

        res_none = FIT.calculate_spectral_flux(û, ks; decomposition=FIT.NoDecomposition(), dealiasing = FIT.NoDealiasing())
        res_helm = FIT.calculate_spectral_flux(û, ks; decomposition=FIT.HelmholtzDecomposition(), dealiasing = FIT.NoDealiasing())
        res_rot  = FIT.calculate_spectral_flux(û, ks; decomposition=FIT.RotationalDecomposition(), dealiasing = FIT.NoDealiasing())
        res_div  = FIT.calculate_spectral_flux(û, ks; decomposition=FIT.DivergentDecomposition(), dealiasing = FIT.NoDealiasing())

        Test.@test res_none isa FIT.SpectralFluxResult
        Test.@test res_helm isa NamedTuple
        Test.@test haskey(res_helm, :rotational) && haskey(res_helm, :divergent)
        Test.@test res_rot isa FIT.SpectralFluxResult
        Test.@test res_div isa FIT.SpectralFluxResult

        # For these divergence-free/rotational modes, verify consistency:
        # T_none ≈ T_rot + T_div
        Test.@test isapprox(res_none.transfer_spectrum, res_rot.transfer_spectrum + res_div.transfer_spectrum; atol=1e-12)

        # 2. Coarse-graining flux decomposition test
        x = range(0, L; length=N+1)[1:N]
        y = range(0, L; length=N+1)[1:N]
        u = [cos(x) for x in x, y in y]
        v = [sin(y) for x in x, y in y]

        cg_none = FIT.calculate_coarse_graining_flux((u, v), (x, y), 1.0, FIT.GaussianFilter(); decomposition=FIT.NoDecomposition())
        cg_helm = FIT.calculate_coarse_graining_flux((u, v), (x, y), 1.0, FIT.GaussianFilter(); decomposition=FIT.HelmholtzDecomposition())
        cg_rot  = FIT.calculate_coarse_graining_flux((u, v), (x, y), 1.0, FIT.GaussianFilter(); decomposition=FIT.RotationalDecomposition())
        cg_div  = FIT.calculate_coarse_graining_flux((u, v), (x, y), 1.0, FIT.GaussianFilter(); decomposition=FIT.DivergentDecomposition())

        Test.@test cg_none isa FIT.CoarseGrainingFluxResult
        Test.@test cg_helm isa NamedTuple
        Test.@test haskey(cg_helm, :rotational) && haskey(cg_helm, :divergent)
        Test.@test cg_rot isa FIT.CoarseGrainingFluxResult
        Test.@test cg_div isa FIT.CoarseGrainingFluxResult

        # 3. 3D coarse-graining with Helmholtz decomposition (physical-space decompose + 3D CGEF flux).
        M = 8; x3 = collect(range(0, L; length=M+1)[1:M]); y3 = copy(x3); z3 = copy(x3)
        u3 = [cos(xi) * sin(zi) for xi in x3, yi in y3, zi in z3]
        v3 = [sin(yi)           for xi in x3, yi in y3, zi in z3]
        w3 = [cos(zi)           for xi in x3, yi in y3, zi in z3]
        cg3_none = FIT.calculate_coarse_graining_flux((u3, v3, w3), (x3, y3, z3), 1.0, FIT.GaussianFilter(); decomposition=FIT.NoDecomposition())
        cg3_helm = FIT.calculate_coarse_graining_flux((u3, v3, w3), (x3, y3, z3), 1.0, FIT.GaussianFilter(); decomposition=FIT.HelmholtzDecomposition())
        Test.@test cg3_none isa FIT.CoarseGrainingFluxResult && size(cg3_none.flux_field) == (M, M, M)
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
        b = FIT.LinearBinning(2π / L)
        res_serial = FIT.calculate_shell_to_shell_transfer(û, ks; binning=b, dealiasing = FIT.OrszagTwoThirds(), verify_antisymmetry=true, execution=FIT.SerialBackend())
        res_thread = FIT.calculate_shell_to_shell_transfer(û, ks; binning=b, dealiasing = FIT.OrszagTwoThirds(), verify_antisymmetry=true, execution=FIT.ThreadedBackend())
        
        # For DistributedBackend, we convert velocity_hat to a SharedArray so workers can read it efficiently
        s_û = SharedArrays.SharedArray(û)
        res_dist = FIT.calculate_shell_to_shell_transfer(s_û, ks; binning=b, dealiasing = FIT.OrszagTwoThirds(), verify_antisymmetry=true, execution=FIT.DistributedBackend())

        Test.@test isapprox(res_serial.transfer_matrix, res_thread.transfer_matrix; atol=1e-12)
        Test.@test isapprox(res_serial.transfer_matrix, res_dist.transfer_matrix; atol=1e-12)
        Test.@test isapprox(res_serial.net_transfer, res_thread.net_transfer; atol=1e-12)
        Test.@test isapprox(res_serial.net_transfer, res_dist.net_transfer; atol=1e-12)

        # #4 — parametric DistributedBackend{Inner} + local_backend accessor.
        Test.@test FIT.DistributedBackend() === FIT.DistributedBackend(FIT.SerialBackend())
        Test.@test FIT.local_backend(FIT.DistributedBackend(FIT.ThreadedBackend())) === FIT.ThreadedBackend()
        Test.@test FIT.local_backend(FIT.SerialBackend()) === FIT.SerialBackend()   # identity for non-distributed
        # Hybrid distributed+threaded per-worker path must match serial (workers here are single-threaded,
        # so this exercises the ThreadedBackend inner branch of compute_mediator_transfer_column).
        res_hybrid = FIT.calculate_shell_to_shell_transfer(s_û, ks; binning=b, dealiasing = FIT.OrszagTwoThirds(), verify_antisymmetry=true, execution=FIT.DistributedBackend(FIT.ThreadedBackend()))
        Test.@test isapprox(res_serial.transfer_matrix, res_hybrid.transfer_matrix; atol=1e-12)
        Test.@test isapprox(res_serial.net_transfer, res_hybrid.net_transfer; atol=1e-12)

        # 2. Triadic Orthogonal Decomposition parity — triads distributed over workers, each running the
        # SAME allocation-reusing SVD as the serial loop → bit-identical L / T_budget / modes / coeffs.
        # The temporal DFT (Q_hat) is master-side, so workers need only FlowInvariantTransfer (no FFTW).
        Random.seed!(123)
        Xtod = randn(64, 1, 4)
        tod_serial = FIT.triadic_orthogonal_decomposition(Xtod; dt = 0.05, spectral = FIT.DirectSumBackend(), execution = FIT.SerialBackend(), return_coefficients = true, return_auxiliary_modes = true)
        tod_dist   = FIT.triadic_orthogonal_decomposition(Xtod; dt = 0.05, spectral = FIT.DirectSumBackend(), execution = FIT.DistributedBackend(), return_coefficients = true, return_auxiliary_modes = true)
        Test.@test isequal(tod_serial.mode_bispectrum, tod_dist.mode_bispectrum)
        Test.@test isequal(tod_serial.modal_energy_budget, tod_dist.modal_energy_budget)
        Test.@test length(tod_serial.modes) == length(tod_dist.modes)
        let k = first(keys(tod_serial.modes))
            Test.@test tod_serial.modes[k].convective == tod_dist.modes[k].convective
            Test.@test tod_serial.expansion_coefficients[k].recipient == tod_dist.expansion_coefficients[k].recipient
            Test.@test tod_serial.auxiliary_modes[k].donor == tod_dist.auxiliary_modes[k].donor
        end
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
        b  = FIT.LinearBinning(2π / L)

        nthr0 = FFTW.get_num_threads()
        try
            FFTW.set_num_threads(1)
            flux1  = FIT.calculate_spectral_flux(û, ks; binning=b, spectral=FIT.FFTBackend())
            s2s1   = FIT.calculate_shell_to_shell_transfer(û, ks; binning=b, spectral=FIT.FFTBackend())
            FFTW.set_num_threads(4)
            flux4  = FIT.calculate_spectral_flux(û, ks; binning=b, spectral=FIT.FFTBackend())
            s2s4   = FIT.calculate_shell_to_shell_transfer(û, ks; binning=b, spectral=FIT.FFTBackend())

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
        Test.@test_throws ArgumentError FIT.calculate_mode_to_mode_transfer(û2, ks2; invariant=FIT.Helicity())
        # 3D field + Enstrophy() now works (vector-vorticity transfer, routed).
        ks3 = FIT.Utils.wavenumber_grid((4, 4, 4), (L, L, L))
        û3  = zeros(ComplexF64, 4, 4, 4, 3); û3[2, 1, 1, 1] = 0.5
        Test.@test FIT.calculate_mode_to_mode_transfer(û3, ks3; invariant=FIT.Enstrophy()) isa FIT.ModeToModeTriadResult
        # KineticEnergy works in both dimensionalities.
        Test.@test FIT.calculate_mode_to_mode_transfer(û2, ks2; invariant=FIT.KineticEnergy()) isa FIT.ModeToModeTriadResult
        Test.@test FIT.calculate_mode_to_mode_transfer(û3, ks3; invariant=FIT.KineticEnergy()) isa FIT.ModeToModeTriadResult
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
        m2m = FIT.calculate_mode_to_mode_transfer(û, ks; dealiasing = FIT.OrszagTwoThirds(), spectral = FIT.FFTBackend())
        S   = m2m.transfer                              # shape (N,N,N,N): S[k..., p...]
        nrm = sqrt(sum(abs2, S)); Test.@test nrm > 0    # non-degenerate
        asym = 0.0
        for k in CartesianIndices((N, N)), p in CartesianIndices((N, N))
            asym = max(asym, abs(S[k, p] + S[p, k]))
        end
        Test.@test asym < 1e-10 * nrm                   # antisymmetric S(k|p) = −S(p|k)
        Test.@test abs(sum(S)) < 1e-10 * nrm            # conserves Σ_kΣ_p S = 0

        b = FIT.LinearBinning(2π/L)
        sf = FIT.calculate_spectral_flux(û, ks; binning = b, dealiasing = FIT.OrszagTwoThirds(), spectral = FIT.FFTBackend())
        ss = FIT.calculate_shell_to_shell_transfer(û, ks; binning = b, dealiasing = FIT.OrszagTwoThirds(),
            verify_antisymmetry = false, spectral = FIT.FFTBackend())
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
        b  = FIT.LinearBinning(2π / L)
        ref = FIT.calculate_shell_to_shell_transfer(û, ks; binning=b, spectral=FIT.FFTBackend(), execution=FIT.SerialBackend())
        ka  = FIT.calculate_shell_to_shell_transfer(û, ks; binning=b, spectral=FIT.FFTBackend(), execution=FIT.GPUBackend(KA.CPU()))
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
        rh = FIT.calculate_shell_to_shell_transfer(û3, ks3; binning=FIT.LinearBinning(2π/L), spectral=FIT.FFTBackend(), execution=FIT.SerialBackend(), invariant=FIT.Helicity())
        gh = FIT.calculate_shell_to_shell_transfer(û3, ks3; binning=FIT.LinearBinning(2π/L), spectral=FIT.FFTBackend(), execution=FIT.GPUBackend(KA.CPU()), invariant=FIT.Helicity())
        Test.@test isapprox(gh.transfer_matrix, rh.transfer_matrix; atol=1e-12 * (maximum(abs, rh.transfer_matrix)+eps()))

        re = FIT.calculate_shell_to_shell_transfer(û, ks; binning=b, spectral=FIT.FFTBackend(), execution=FIT.SerialBackend(), invariant=FIT.Enstrophy())
        ge = FIT.calculate_shell_to_shell_transfer(û, ks; binning=b, spectral=FIT.FFTBackend(), execution=FIT.GPUBackend(KA.CPU()), invariant=FIT.Enstrophy())
        Test.@test isapprox(ge.transfer_matrix, re.transfer_matrix; atol=1e-12 * (maximum(abs, re.transfer_matrix)+eps()))

        # Enstrophy 3D (vector vorticity + vortex stretching) device kernel — reuse the ∇×A field above.
        re3 = FIT.calculate_shell_to_shell_transfer(û3, ks3; binning=FIT.LinearBinning(2π/L), spectral=FIT.FFTBackend(), execution=FIT.SerialBackend(), invariant=FIT.Enstrophy())
        ge3 = FIT.calculate_shell_to_shell_transfer(û3, ks3; binning=FIT.LinearBinning(2π/L), spectral=FIT.FFTBackend(), execution=FIT.GPUBackend(KA.CPU()), invariant=FIT.Enstrophy())
        Test.@test isapprox(ge3.transfer_matrix, re3.transfer_matrix; atol=1e-12 * (maximum(abs, re3.transfer_matrix)+eps()))
    end

    # -----------------------------------------------------------------------
    # Spectral flux Π(K): every execution backend (Threaded/Distributed/GPU) must reproduce the
    # serial reduction to machine precision, for each invariant with a device kernel. GPU is the
    # KernelAbstractions CPU backend (same device path used on CUDA/ROC/Metal); Distributed runs
    # over however many workers are present (≥1). Guards against the issue #12 regression (spectral
    # flux was serial-only) and against the execution axis diverging from serial.
    Test.@testset "Spectral flux execution backends" begin
        L = 2π
        N = 16
        ks = FIT.Utils.wavenumber_grid((N, N), (L, L))
        kx = [ks[1][i] for i in 1:N, j in 1:N]; ky = [ks[2][j] for i in 1:N, j in 1:N]
        ψh = FFTW.fft(randn(Random.MersenneTwister(11), N, N)) ./ N^2
        û  = cat(im .* ky .* ψh, -im .* kx .* ψh; dims = 3)
        b  = FIT.LinearBinning(2π / L)

        ref = FIT.calculate_spectral_flux(û, ks; binning=b, spectral=FIT.FFTBackend(), execution=FIT.SerialBackend())
        atolT = 1e-12 * (maximum(abs, ref.transfer_spectrum) + eps())
        atolΠ = 1e-12 * (maximum(abs, ref.flux) + eps())
        for exec in (FIT.ThreadedBackend(), FIT.DistributedBackend(), FIT.GPUBackend(KA.CPU()))
            res = FIT.calculate_spectral_flux(û, ks; binning=b, spectral=FIT.FFTBackend(), execution=exec)
            Test.@test isapprox(res.transfer_spectrum, ref.transfer_spectrum; atol=atolT)
            Test.@test isapprox(res.flux, ref.flux; atol=atolΠ)
        end

        # 3D helicity + 2D enstrophy device kernels through the spectral-flux GPU path.
        ks3 = FIT.Utils.wavenumber_grid((8, 8, 8), (L, L, L))
        û3  = randn(Random.MersenneTwister(13), ComplexF64, 8, 8, 8, 3) .* 0.1
        rh = FIT.calculate_spectral_flux(û3, ks3; binning=FIT.LinearBinning(2π/L), spectral=FIT.FFTBackend(), invariant=FIT.Helicity())
        gh = FIT.calculate_spectral_flux(û3, ks3; binning=FIT.LinearBinning(2π/L), spectral=FIT.FFTBackend(), invariant=FIT.Helicity(), execution=FIT.GPUBackend(KA.CPU()))
        Test.@test isapprox(gh.transfer_spectrum, rh.transfer_spectrum; atol=1e-12 * (maximum(abs, rh.transfer_spectrum)+eps()))
        Test.@test isapprox(gh.flux, rh.flux; atol=1e-12 * (maximum(abs, rh.flux)+eps()))

        re = FIT.calculate_spectral_flux(û, ks; binning=b, spectral=FIT.FFTBackend(), invariant=FIT.Enstrophy())
        ge = FIT.calculate_spectral_flux(û, ks; binning=b, spectral=FIT.FFTBackend(), invariant=FIT.Enstrophy(), execution=FIT.GPUBackend(KA.CPU()))
        Test.@test isapprox(ge.transfer_spectrum, re.transfer_spectrum; atol=1e-12 * (maximum(abs, re.transfer_spectrum)+eps()))

        # AutoBackend resolves to the best available backend (threaded here, since OhMyThreads is
        # loaded and the suite runs multithreaded in CI) and must match the serial reference.
        auto = FIT.calculate_spectral_flux(û, ks; binning=b, spectral=FIT.FFTBackend(), execution=FIT.AutoBackend())
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
        b  = FIT.LinearBinning(2π / L)
        for sp in (FIT.DirectSumBackend(), FIT.FFTBackend())
            ms = FIT.calculate_mode_to_mode_transfer(û, ks; spectral=sp, execution=FIT.SerialBackend())
            mg = FIT.calculate_mode_to_mode_transfer(û, ks; spectral=sp, execution=FIT.GPUBackend(KA.CPU()))
            Test.@test isapprox(mg.transfer, ms.transfer; atol=1e-12 * (maximum(abs, ms.transfer)+eps()))
            Test.@test isapprox(mg.net_transfer, ms.net_transfer; atol=1e-12 * (maximum(abs, ms.net_transfer)+eps()))

            bnds = FIT.SmoothBands(collect(1.0:6.0))
            bs = FIT.calculate_band_to_band_transfer(û, ks; bands=bnds, spectral=sp, execution=FIT.SerialBackend())
            bg = FIT.calculate_band_to_band_transfer(û, ks; bands=bnds, spectral=sp, execution=FIT.GPUBackend(KA.CPU()))
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
        for sp in (FIT.DirectSumBackend(), FIT.FFTBackend())
            ps = FIT.calculate_partial_fluxes(û3, ks3p; spectral=sp, execution=FIT.SerialBackend())
            pg = FIT.calculate_partial_fluxes(û3, ks3p; spectral=sp, execution=FIT.GPUBackend(KA.CPU()))
            Test.@test length(pg.channels) == length(ps.channels)
            for k in keys(ps.channels)
                Test.@test isapprox(pg.channels[k].flux, ps.channels[k].flux;
                                    atol=1e-12 * (maximum(abs, ps.channels[k].flux)+eps()))
            end
        end
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
        up, um = FIT.Decomposition.decompose_field(FIT.HelicalDecomposition(), û3, ks3)
        upj, umj = FIT.Decomposition.decompose_field(FIT.HelicalDecomposition(), JLArrays.JLArray(û3), ks3)
        Test.@test upj isa JLArrays.JLArray
        Test.@test maximum(abs, Array(upj) .- up) == 0
        Test.@test maximum(abs, Array(umj) .- um) == 0

        # Toroidal–poloidal decomposition — device-resident + bit-identical (broadcast Craya–Herring frame).
        tp  = FIT.Decomposition.decompose_field(FIT.ToroidalPoloidalDecomposition(), û3, ks3)
        tpj = FIT.Decomposition.decompose_field(FIT.ToroidalPoloidalDecomposition(), JLArrays.JLArray(û3), ks3)
        Test.@test tpj.toroidal isa JLArrays.JLArray
        Test.@test maximum(abs, Array(tpj.toroidal) .- tp.toroidal) == 0
        Test.@test maximum(abs, Array(tpj.poloidal) .- tp.poloidal) == 0

        # Helmholtz (Leray) decomposition — N-D device-generic projection, device-resident + bit-identical.
        hm  = FIT.Decomposition.decompose_field(FIT.HelmholtzDecomposition(), û3, ks3)
        hmj = FIT.Decomposition.decompose_field(FIT.HelmholtzDecomposition(), JLArrays.JLArray(û3), ks3)
        Test.@test hmj.rotational isa JLArrays.JLArray
        Test.@test maximum(abs, Array(hmj.rotational) .- hm.rotational) == 0
        Test.@test maximum(abs, Array(hmj.divergent) .- hm.divergent) == 0

        # transfer_density! (KE) — device kernel/broadcast matches the CPU reduction.
        ks2 = FIT.Utils.wavenumber_grid((N, N), (L, L))
        û2 = randn(Random.MersenneTwister(6), ComplexF64, N, N, 2)
        N̂2 = randn(Random.MersenneTwister(7), ComplexF64, N, N, 2)
        td_h = similar(û2, Float64, N, N); FIT.Invariants.transfer_density!(td_h, FIT.KineticEnergy(), û2, N̂2, ks2)
        td_d = JLArrays.JLArray(similar(û2, Float64, N, N))
        FIT.Invariants.transfer_density!(td_d, FIT.KineticEnergy(), JLArrays.JLArray(û2), JLArrays.JLArray(N̂2), ks2)
        Test.@test maximum(abs, Array(td_d) .- td_h) == 0
        # Passive scalar (M=1) — same device-generic dot broadcast as KE. (Helicity/Enstrophy use the
        # KA-extension vorticity kernels via GPUBackend, verified on KA.CPU above, not this broadcast path.)
        θ2 = randn(Random.MersenneTwister(8), ComplexF64, N, N, 1)
        θN = randn(Random.MersenneTwister(9), ComplexF64, N, N, 1)
        ts_h = similar(θ2, Float64, N, N); FIT.Invariants.transfer_density!(ts_h, FIT.PassiveScalar(), θ2, θN, ks2)
        ts_d = JLArrays.JLArray(similar(θ2, Float64, N, N))
        FIT.Invariants.transfer_density!(ts_d, FIT.PassiveScalar(), JLArrays.JLArray(θ2), JLArrays.JLArray(θN), ks2)
        Test.@test maximum(abs, Array(ts_d) .- ts_h) == 0

        # Compressible device helpers (copy-trunc / Helmholtz split / shell bin) match the scalar CPU path.
        dch = similar(û2); FIT.Compressible._copy_trunc!(dch, û2, (N, N), 2, true)
        dcd = JLArrays.JLArray(similar(û2)); FIT.Compressible._copy_trunc!(dcd, JLArrays.JLArray(û2), (N, N), 2, true)
        Test.@test maximum(abs, Array(dcd) .- dch) == 0
        rh = similar(û2); ch = similar(û2); FIT.Compressible._helmholtz_split!(rh, ch, û2, ks2, (N, N))
        rd = JLArrays.JLArray(similar(û2)); cd = JLArrays.JLArray(similar(û2))
        FIT.Compressible._helmholtz_split!(rd, cd, JLArrays.JLArray(û2), ks2, (N, N))
        Test.@test maximum(abs, Array(rd) .- rh) < 1e-14 * (maximum(abs, rh) + eps())
        Test.@test maximum(abs, Array(cd) .- ch) < 1e-14 * (maximum(abs, ch) + eps())
        kmag = FIT.ShellBinning.shell_coordinate(FIT.IsotropicShells(), ks2); bb = FIT.LinearBinning(2π / L)
        edges = FIT.ShellBinning.shell_edges(bb, maximum(kmag)); sidx = FIT.ShellBinning.assign_shells(kmag, edges)
        Nsh = length(collect(FIT.ShellBinning.shell_centers(bb, maximum(kmag))))
        Th = FIT.Compressible._bin(td_h, sidx, Nsh, Float64, (N, N), true)
        Td = FIT.Compressible._bin(JLArrays.JLArray(td_h), sidx, Nsh, Float64, (N, N), true)
        Test.@test maximum(abs, Td .- Th) == 0
    end

    # -----------------------------------------------------------------------
    # Compressible KE spectral transfer (#1 / Singh–Tiwari–Sharma–Verma 2025). Validated by the
    # analytic identities that make it trustworthy: (a) the momentum-weighted nonlinear transfer
    # conserves total KE, Σ_k T_u = 0; (b) the incompressible limit ρ≡1, ∇·u=0 reduces T_u to
    # −(incompressible transfer_spectrum) (paper Eqs. 48–50); (c) the R/C flux channels reconstruct
    # the total flux and the compressive/cross channels vanish for incompressible flow; (d) uniform
    # pressure ⇒ zero pressure-dilatation.
    Test.@testset "Compressible energy transfer (#1)" begin
        L = 2π; N = 16
        ks = FIT.Utils.wavenumber_grid((N, N), (L, L))
        kx = [ks[1][i] for i in 1:N, j in 1:N]; ky = [ks[2][j] for i in 1:N, j in 1:N]
        # Broadband real divergence-free velocity from a random real streamfunction (nonzero net
        # inter-shell transfer, so the incompressible reference and the tolerances are non-degenerate).
        ψh = FFTW.fft(randn(Random.MersenneTwister(101), N, N)) ./ N^2
        û  = cat(im .* ky .* ψh, -im .* kx .* ψh; dims=3)
        b = FIT.LinearBinning(2π / L)
        ρ̂ = zeros(ComplexF64, N, N); ρ̂[1, 1] = 1.0    # ρ(x) ≡ 1  (k=0 mode)

        res = FIT.calculate_compressible_flux(û, ρ̂, ks; binning=b)
        ref = FIT.calculate_spectral_flux(û, ks; binning=b, spectral=FIT.FFTBackend())
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
        resp = FIT.calculate_compressible_flux(û, ρ̂, ks; binning=b, pressure_hat=σ̂)
        Test.@test resp.pressure_dilatation !== nothing
        Test.@test maximum(abs, resp.pressure_dilatation.rotational) < 1e-10 * scaleT
        Test.@test maximum(abs, resp.pressure_dilatation.compressive) < 1e-10 * scaleT
        # no pressure ⇒ nothing
        Test.@test res.pressure_dilatation === nothing
    end

    # -----------------------------------------------------------------------
    # Extension smoke tests (#10): exercise the previously-untested extensions with meaningful
    # numerical assertions, not @test true. CairoMakie (plot dispatch incl. the new TOD figure),
    # FINUFFT (scattered-Cartesian coarse-graining + the calculate_energy_transfer wiring, #11),
    # and FlowFieldSpectra (physical→spectral front-end). FSH/NUFSHT spherical transfer is tested
    # in its own testset once the genuine spherical implementation lands.
    Test.@testset "Extension smoke tests (CairoMakie / FINUFFT / FlowFieldSpectra)" begin
        L = 2π; N = 16
        ks = FIT.Utils.wavenumber_grid((N, N), (L, L))
        kx = [ks[1][i] for i in 1:N, j in 1:N]; ky = [ks[2][j] for i in 1:N, j in 1:N]
        ψh = FFTW.fft(randn(Random.MersenneTwister(21), N, N)) ./ N^2
        û  = cat(im .* ky .* ψh, -im .* kx .* ψh; dims = 3)
        b  = FIT.LinearBinning(2π / L)

        Test.@testset "CairoMakie plot dispatch" begin
            sf = FIT.calculate_spectral_flux(û, ks; binning=b, spectral=FIT.FFTBackend())
            Test.@test FIT.plot_energy_transfer(sf) isa CairoMakie.Figure
            ss = FIT.calculate_shell_to_shell_transfer(û, ks; binning=b, spectral=FIT.FFTBackend())
            Test.@test FIT.plot_energy_transfer(ss) isa CairoMakie.Figure
            # New TOD Fig-4 bispectrum plot (#3): build a minimal result and render it.
            nF = 6; nm = 2
            freqs = collect(range(-2.0, 2.0; length=nF))
            λ = abs.(randn(Random.MersenneTwister(3), nF, nF, nm))
            Tb = randn(Random.MersenneTwister(4), nF, nF, nm)
            tod = FIT.TriadicOrthogonalDecompositionResult(freqs, λ, Dict{Tuple{Int,Int},Any}(), Tb, nothing, nothing)
            Test.@test FIT.plot_energy_transfer(tod) isa CairoMakie.Figure
            Test.@test FIT.plot_energy_transfer(tod; mode=2, fmax=1.5) isa CairoMakie.Figure
            Test.@test_throws ArgumentError FIT.plot_energy_transfer(tod; mode=99)
        end

        Test.@testset "FINUFFT scattered coarse-graining (#11)" begin
            # Sample a smooth periodic velocity field at scattered points; the scattered coarse-graining
            # flux must be finite and its calculate_energy_transfer wiring must match the direct call.
            rng = Random.MersenneTwister(31)
            Np = 400
            xs = 2π .* rand(rng, Np); ys = 2π .* rand(rng, Np)
            u  = @. sin(xs) * cos(ys); v = @. -cos(xs) * sin(ys)
            ms = (16, 16); ℓ = 0.5
            filt = FIT.GaussianFilter()
            direct = FIT.nufft_coarse_graining_flux((u, v), (xs, ys), ℓ, filt, ms)
            Test.@test direct isa FIT.CoarseGrainingFluxResult
            Test.@test all(isfinite, direct.flux_field)
            Test.@test length(direct.flux_field) == Np
            # Part D: unified calculate_energy_transfer entry for scattered CoarseGrainingFlux.
            method = FIT.CoarseGrainingFluxMethod(filt, ℓ)
            wired = FIT.calculate_energy_transfer(method, (u, v), (xs, ys), ms)
            Test.@test isapprox(wired.flux_field, direct.flux_field; rtol=1e-10)

            # In-place workspace form: matches the allocating call, diagnostics work, and a repeat
            # call reuses the plans + every buffer — so it allocates only the tiny result struct
            # (a genuine reduction, not the plan-reuse-only 1.06× it would be without buffer reuse).
            ws = FIT.NUFFTCoarseGrainingWorkspace((xs, ys), ms)
            ip = FIT.nufft_coarse_graining_flux!(ws, (u, v), ℓ, filt, ms)
            Test.@test isapprox(ip.flux_field, direct.flux_field; rtol=1e-10)
            ipd = FIT.nufft_coarse_graining_flux!(ws, (u, v), ℓ, filt, ms; return_diagnostics=true)
            Test.@test ipd isa FIT.CoarseGrainingFluxResultWithDiagnostics
            Test.@test size(ipd.stress_tensor) == (Np, 2, 2)
            # scale sweep reuses one workspace
            for l in (0.3, 0.7)
                sweep_alloc = FIT.nufft_coarse_graining_flux((u, v), (xs, ys), l, filt, ms)
                sweep_ip = FIT.nufft_coarse_graining_flux!(ws, (u, v), l, filt, ms)
                Test.@test isapprox(sweep_alloc.flux_field, sweep_ip.flux_field; rtol=1e-10)
            end
            # (workspace-reuse allocation ratio asserted in test_allocs.jl)
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
            xs = collect(range(0, 2π; length=N+1)[1:N])
            # Physical field from the fft/N² coefficients is bfft(û) = Σ_k û e^{ik·x} (unnormalized
            # inverse); ifft(û) would give u/Nᵈ (a scaled field) and mis-scale the FFS comparison.
            uxp = real.(FFTW.bfft(ûbl[:, :, 1]))
            uyp = real.(FFTW.bfft(ûbl[:, :, 2]))
            ffs = FIT.calculate_energy_transfer(FIT.SpectralFluxMethod(b), (uxp, uyp), (xs, xs), (N, N))
            Test.@test ffs isa FIT.SpectralFluxResult
            Test.@test abs(sum(ffs.transfer_spectrum)) < 1e-8 * (maximum(abs, ffs.transfer_spectrum) + eps())
            # Correctness: the physical→spectral front-end must reproduce the transfer computed directly
            # from the field's own FFT coefficients (same 2/3 dealiasing on both sides).
            ref = FIT.calculate_spectral_flux(ûbl, ks; binning=b, spectral=FIT.FFTBackend())
            Test.@test isapprox(ffs.transfer_spectrum, ref.transfer_spectrum;
                                atol = 1e-10 * (maximum(abs, ref.transfer_spectrum) + eps()))
        end
    end

    # -----------------------------------------------------------------------
    # Spherical spectral transfer (#10 FSH extension; genuine 2D-barotropic implementation).
    # The rigorous anchor is exact conservation Σ_l T = 0 (= ∫ψ J(ψ,ζ) dΩ = 0 by antisymmetry),
    # which holds to machine precision iff the quadratic Jacobian is dealiased (evaluated on the
    # 2·lmax grid). Convention verified directly against FSH's eth definition (see THEORY.md §6.1).
    Test.@testset "Spherical spectral transfer (FastSphericalHarmonics, 2D barotropic)" begin
        lmax = 20; N = lmax + 1
        rng = Random.MersenneTwister(2024)
        # Band-limited real vorticity field on the FSH equiangular grid.
        ζ = FSH.spinsph_evaluate(FSH.spinsph_transform(randn(rng, N, 2N - 1), 0), 0)
        ζ = real.(ζ)

        res = FIT.calculate_energy_transfer(FIT.SphericalTransferMethod(), ζ)
        Test.@test res isa FIT.SphericalTransferResult
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
        res2 = FIT.calculate_energy_transfer(FIT.SphericalTransferMethod(radius = 2.0), ζ)
        Test.@test abs(sum(res2.enstrophy_transfer)) < 1e-12 * (maximum(abs, res2.enstrophy_transfer) + eps())

        # Dealiasing does real work: on the same broadband field, the aliased path does NOT conserve.
        aliased = FIT.calculate_energy_transfer(FIT.SphericalTransferMethod(), ζ; dealias = false)
        Test.@test abs(sum(aliased.enstrophy_transfer)) > 1e-6 * (maximum(abs, aliased.enstrophy_transfer) + eps())

        # Grid-shape guard.
        Test.@test_throws ArgumentError FIT.calculate_energy_transfer(FIT.SphericalTransferMethod(), randn(N, N))

        # In-place workspace form: matches the allocating path exactly (same math), and a repeat call
        # reuses the embed/gradient/Jacobian/reduction buffers — the FastSphericalHarmonics transforms
        # (no in-place API) are the irreducible floor, so this saves the FIT-side portion, not all of it.
        ws = FIT.SphericalTransferWorkspace(lmax; radius = 1.0, dealias = true)
        ip = FIT.calculate_spherical_transfer!(ws, ζ)
        Test.@test ip isa FIT.SphericalTransferResult
        Test.@test ip.energy_transfer == res.energy_transfer
        Test.@test ip.enstrophy_transfer == res.enstrophy_transfer
        Test.@test ip.energy_flux == res.energy_flux
        Test.@test abs(sum(ip.enstrophy_transfer)) < 1e-12 * scaleZ
        # (workspace-reuse allocation ratio asserted in test_allocs.jl)
        # Workspace grid must match the field.
        Test.@test_throws ArgumentError FIT.calculate_spherical_transfer!(ws, randn(N, N))
    end

    # -----------------------------------------------------------------------
    # Scattered spherical transfer (#10 NUFSHT extension; rewrite off a fabricated API). Same
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
        resF = FIT.calculate_energy_transfer(FIT.SphericalTransferMethod(radius = a), ζgrid)

        # Spherical-Fibonacci scattered points (equidistributed), M ≥ (2lmax+1)² for the dealiased solve.
        Msolve = 8 * (2lmax + 1)^2
        ga = π * (3 - sqrt(5))
        zf = [1 - 2 * (k + 0.5) / Msolve for k in 0:Msolve-1]
        θs = acos.(clamp.(zf, -1.0, 1.0)); φs = mod.(ga .* (0:Msolve-1), 2π)
        ζscat = zeros(Msolve)
        NUFSHT.nusht_type2!(ζscat, C, NUFSHT.make_plan(θs, φs, lmax; tol = 1e-12))

        res = FIT.calculate_energy_transfer(FIT.SphericalTransferMethod(radius = a), ζscat, (θs, φs);
                                            lmax = lmax, tol = 1e-12, rtol = 1e-13)
        Test.@test res isa FIT.SphericalTransferResult
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
            FIT.SphericalTransferMethod(), ζscat[1:10], (θs[1:10], φs[1:10]); lmax = lmax)

        # In-place workspace form: reuses the three NUFSHT plans (the dominant cost) + all buffers, so
        # a snapshot sweep on the same points re-plans nothing. Matches the allocating path to the CG
        # tolerance; a repeat call allocates far less (only the NUFSHT-internal CG scratch remains).
        ws = FIT.ScatteredSphericalTransferWorkspace((θs, φs), lmax; radius = a, tol = 1e-12, rtol = 1e-13)
        ip = FIT.calculate_spherical_transfer!(ws, ζscat)
        Test.@test ip isa FIT.SphericalTransferResult
        Test.@test maximum(abs.(ip.energy_transfer .- res.energy_transfer)) < 1e-10 * scaleE
        Test.@test maximum(abs.(ip.enstrophy_transfer .- res.enstrophy_transfer)) < 1e-10 * scaleZ
        Test.@test abs(sum(ip.energy_transfer)) < 1e-8 * scaleE
        # (workspace-reuse allocation ratio asserted in test_allocs.jl)
        Test.@test_throws DimensionMismatch FIT.calculate_spherical_transfer!(ws, ζscat[1:end-1])
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
            b   = FIT.LinearBinning(T(2π / L))

            r1 = FIT.calculate_spectral_flux(û, ks; binning = b, spectral = FIT.FFTBackend())
            r2 = FIT.calculate_shell_to_shell_transfer(û, ks; binning = b, spectral = FIT.FFTBackend())
            r3 = FIT.calculate_mode_to_mode_transfer(û, ks)
            r4 = FIT.calculate_band_to_band_transfer(û, ks; bands = FIT.SmoothBands(T[2, 4, 6]))
            r5 = FIT.calculate_partial_fluxes(û, ks; binning = b, decomposition = FIT.HelmholtzDecomposition())
            r6 = FIT.calculate_compressible_flux(û, ρ̂, ks; binning = b)

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

end # top-level testset
