using Test: Test
using Random: Random
using Statistics: Statistics
using Aqua: Aqua
using FFTW: FFTW
using LinearAlgebra: LinearAlgebra
using Distributed: Distributed
using SharedArrays: SharedArrays
using OhMyThreads: OhMyThreads
using CoarseGrainingEnergyFluxes: CoarseGrainingEnergyFluxes
using HelmholtzDecomposition: HelmholtzDecomposition

using FlowInvariantTransfer: FlowInvariantTransfer as FET

Test.@testset "FlowInvariantTransfer.jl Test Suite" begin

    # -----------------------------------------------------------------------
    Test.@testset "Aqua Code Quality" begin
        Aqua.test_all(FET; ambiguities = false, stale_deps = (ignore=[:Documenter],))
    end

    # -----------------------------------------------------------------------
    Test.@testset "Utils — wavenumber_grid" begin
        N = 8
        L = 2π
        ks = FET.wavenumber_grid((N, N), (L, L))
        Test.@test length(ks) == 2
        Test.@test length(ks[1]) == N
        # FFTW order: ks[1] should contain 0, 1, 2, 3, -4, -3, -2, -1 * (2π/L)
        dk = 2π / L
        Test.@test isapprox(ks[1][1], 0.0)
        Test.@test isapprox(ks[1][2], dk, atol=1e-14)
        Test.@test isapprox(ks[1][N], -dk, atol=1e-14)

        k_mag = FET.wavenumber_magnitude_grid(ks)
        Test.@test size(k_mag) == (N, N)
        Test.@test all(k_mag .>= 0)
        Test.@test isapprox(k_mag[1, 1], 0.0)
    end

    # -----------------------------------------------------------------------
    Test.@testset "Utils — dealiasing_mask" begin
        N = 12
        mask = FET.dealiasing_mask((N, N))
        Test.@test size(mask) == (N, N)
        # All modes with |k_d| >= N/3 = 4 along any dim should be zeroed
        # k_idx=0:3 kept, 4:8 removed (FFTW order: 0..N/2 then -(N/2-1)..-1)
        Test.@test mask[1, 1]   # (k=0,k=0) kept
        Test.@test mask[2, 1]   # (k=1,k=0) kept
        Test.@test !mask[5, 1]  # (k=4,k=0) removed by 2/3 rule (4 >= 12/3=4)
    end

    # -----------------------------------------------------------------------
    Test.@testset "ShellBinning — LinearBinning" begin
        b = FET.LinearBinning(1.0)
        edges = FET.shell_edges(b, 5.0)
        Test.@test edges[1] == 0.0
        Test.@test edges[end] >= 5.0
        centers = FET.shell_centers(b, 5.0)
        Test.@test length(centers) == length(edges) - 1
        Test.@test all(diff(centers) .> 0)
    end

    Test.@testset "ShellBinning — LogarithmicBinning" begin
        b = FET.LogarithmicBinning(1.0, 2.0)
        edges = FET.shell_edges(b, 16.0)
        Test.@test edges[1] == 1.0
        Test.@test issorted(edges)
        Test.@test all(edges[2:end] ./ edges[1:end-1] .≈ 2.0)
    end

    Test.@testset "ShellBinning — DyadicBinning vs LogarithmicBinning(2)" begin
        k_max = 16.0
        b_d = FET.DyadicBinning(1.0)
        b_l = FET.LogarithmicBinning(1.0, 2.0)
        Test.@test FET.shell_edges(b_d, k_max) == FET.shell_edges(b_l, k_max)
    end

    Test.@testset "ShellBinning — CustomBinning" begin
        edges = [0.0, 1.0, 3.0, 6.0, 10.0]
        b = FET.CustomBinning(edges)
        Test.@test FET.shell_edges(b, 10.0) == edges
        Test.@test FET.n_shells(b, 10.0) == 4
    end

    Test.@testset "ShellBinning — assign_shells" begin
        ks = FET.wavenumber_grid((8,), (2π,))
        k_mag_1d = FET.wavenumber_magnitude_grid(ks)
        b = FET.LinearBinning(2π / 8)
        edges = FET.shell_edges(b, maximum(k_mag_1d))
        idx = FET.assign_shells(k_mag_1d, edges)
        Test.@test size(idx) == size(k_mag_1d)
        Test.@test eltype(idx) === Int
        Test.@test all(0 .<= idx .<= length(edges) - 1)  # 0 = outside all shells
        Test.@test any(idx .== 1)                         # shell 1 is populated
    end

    # -----------------------------------------------------------------------
    Test.@testset "Filters — spectral responses" begin
        k = 2.0; ℓ = 1.0
        # SharpSpectralFilter: passes k < π/ℓ ≈ 3.14
        Test.@test FET.filter_response(FET.SharpSpectralFilter(), k, ℓ) == 1.0
        Test.@test FET.filter_response(FET.SharpSpectralFilter(), 4.0, ℓ) == 0.0
        # GaussianFilter: always in (0,1], decays with k
        g1 = FET.filter_response(FET.GaussianFilter(), k, ℓ)
        g2 = FET.filter_response(FET.GaussianFilter(), 4.0, ℓ)
        Test.@test 0.0 < g2 < g1 <= 1.0
        # TopHatFilter: sinc, = 1 at k=0
        Test.@test FET.filter_response(FET.TopHatFilter(), 0.0, ℓ) ≈ 1.0
    end

    Test.@testset "Filters — apply_filter_spectral!" begin
        k_mag = Float64[0, 1, 2, 3, 4]
        û_in  = ComplexF64[1.0, 1.0, 1.0, 1.0, 1.0]
        û_out = similar(û_in)
        FET.apply_filter_spectral!(û_out, û_in, k_mag, FET.SharpSpectralFilter(), 1.0)
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
        ks = FET.wavenumber_grid((N,), (L,))

        result = FET.calculate_spectral_flux(û, ks;
            binning = FET.LinearBinning(2π/L), dealiasing = false)

        Test.@test result isa FET.SpectralFluxResult
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
        ks = FET.wavenumber_grid((N,), (L,))

        # Direct path
        result_direct = FET.calculate_spectral_flux(û_phys, ks;
            binning=FET.LinearBinning(2π/L), dealiasing=false, spectral=FET.DirectSumBackend())

        # FFTW path (extension)
        result_fft = FET.calculate_spectral_flux(û_phys, ks;
            binning=FET.LinearBinning(2π/L), dealiasing=false, spectral=FET.FFTBackend())

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
        ks = FET.wavenumber_grid((N, N), (L, L))
        kx = [ks[1][i] for i in 1:N, j in 1:N]
        ky = [ks[2][j] for i in 1:N, j in 1:N]
        û  = cat(im .* ky .* ψh, -im .* kx .* ψh; dims = 3)
        Test.@test maximum(abs.(kx .* û[:, :, 1] .+ ky .* û[:, :, 2])) < 1e-12  # div-free

        for spectral in (FET.DirectSumBackend(), FET.FFTBackend())
            N̂ = FET.compute_nonlinear_term(û, ks; dealiasing = true, spectral = spectral)
            t = FET.transfer_density(FET.KineticEnergy(), û, N̂, ks)
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
        ks = FET.wavenumber_grid((N,), (L,))
        r = FET.calculate_spectral_flux(û, ks; binning = FET.LinearBinning(2π/L), dealiasing = false)
        Test.@test isapprox(r.flux, cumsum(r.transfer_spectrum); atol = 1e-12)
    end

    # -----------------------------------------------------------------------
    Test.@testset "NonlinearTerm — allocation-free in-place hot path" begin
        # The !-variant must allocate nothing per call: FFT path uses pre-planned transforms
        # (ws.plans) + mul! into preallocated buffers; direct path is pure loops.
        N = 16; L = 2π
        ks = FET.wavenumber_grid((N, N), (L, L))
        Random.seed!(3)
        û  = randn(ComplexF64, N, N, 2)
        ws = FET.NonlinearTermWorkspace(û, ks)
        for spectral in (FET.DirectSumBackend(), FET.FFTBackend())
            FET.compute_nonlinear_term!(ws, û, ks; dealiasing = true, spectral = spectral)  # warmup
            a = @allocated FET.compute_nonlinear_term!(ws, û, ks; dealiasing = true, spectral = spectral)
            Test.@test a == 0
        end
    end

    # -----------------------------------------------------------------------
    Test.@testset "Enstrophy — 2D conserved, 3D works (vortex stretching)" begin
        L = 2π
        # 2D: enstrophy is an inviscid invariant ⇒ Σ_k T_Ω ≈ 0 (div-free, dealiased).
        N = 16
        Random.seed!(5)
        ψ  = randn(N, N); ψh = FFTW.fft(ψ) ./ N^2
        ks2 = FET.wavenumber_grid((N, N), (L, L))
        kx = [ks2[1][i] for i in 1:N, j in 1:N]; ky = [ks2[2][j] for i in 1:N, j in 1:N]
        û2 = cat(im .* ky .* ψh, -im .* kx .* ψh; dims = 3)
        N̂2 = FET.compute_nonlinear_term(û2, ks2; dealiasing = true, spectral=FET.FFTBackend())
        tΩ2 = FET.transfer_density(FET.Enstrophy(), û2, N̂2, ks2)
        Test.@test abs(sum(tΩ2)) < 1e-9 * sum(abs, tΩ2)        # 2D enstrophy conserved

        # 3D: vector-vorticity enstrophy transfer runs (non-conservative; sanity only).
        M = 8
        ks3 = FET.wavenumber_grid((M, M, M), (L, L, L))
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
        res3 = FET.calculate_spectral_flux(û3, ks3; binning = FET.LinearBinning(1.0),
            invariant = FET.Enstrophy(), dealiasing = true, spectral=FET.FFTBackend())
        Test.@test res3 isa FET.SpectralFluxResult
        Test.@test all(isfinite, res3.transfer_spectrum)
        # mode-to-mode aggregates now route through the FFT paths, so 3D enstrophy net works
        m2m3 = FET.calculate_mode_to_mode_transfer(û3, ks3; invariant = FET.Enstrophy(), spectral=FET.FFTBackend())
        Test.@test m2m3 isa FET.ModeToModeTriadResult
        Test.@test all(isfinite, m2m3.net_transfer)
    end

    # -----------------------------------------------------------------------
    Test.@testset "ShellToShellTransfer — antisymmetry (divergence-free field)" begin
        # T(n,m) = -T(m,n) holds exactly for divergence-free (incompressible) fields.
        # Build u = ∂ψ/∂y, v = -∂ψ/∂x from a random streamfunction ψ.
        Random.seed!(7)
        N = 16; L = 2π   # large enough that the 2/3-retained band has real inter-shell coupling
        Np = N * N
        ks = FET.wavenumber_grid((N, N), (L, L))
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

        result = FET.calculate_shell_to_shell_transfer(û, ks;
            binning=FET.LinearBinning(2π/L), dealiasing=true,
            verify_antisymmetry=true, spectral=FET.FFTBackend())

        Test.@test result isa FET.ShellToShellResult
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
        ks = FET.wavenumber_grid((N, N), (L, L))
        kx = [ks[1][i] for i in 1:N, j in 1:N]
        ky = [ks[2][j] for i in 1:N, j in 1:N]
        û  = cat(im .* ky .* ψh, -im .* kx .* ψh; dims = 3)
        b  = FET.LinearBinning(2π/L)

        r_direct = FET.calculate_shell_to_shell_transfer(û, ks; binning=b, dealiasing=true,
            verify_antisymmetry=true, spectral=FET.DirectSumBackend())
        r_fft = FET.calculate_shell_to_shell_transfer(û, ks; binning=b, dealiasing=true,
            verify_antisymmetry=true, spectral=FET.FFTBackend())

        T_norm = sqrt(sum(abs2, r_direct.transfer_matrix))
        Test.@test T_norm > 0                                            # non-degenerate
        # serial and FFT implement the SAME (u·∇)u_m form → agree to roundoff
        Test.@test isapprox(r_direct.transfer_matrix, r_fft.transfer_matrix; atol = 1e-9 * T_norm)
        Test.@test r_direct.max_antisymmetry_error < 1e-10 * T_norm      # A is antisymmetric

        # Reduction: Σ_m T(n,m) must equal the spectral transfer T(k) (same field/binning).
        sf = FET.calculate_spectral_flux(û, ks; binning = b, dealiasing = true)
        Test.@test isapprox(r_direct.net_transfer, sf.transfer_spectrum; atol = 1e-9 * T_norm)
    end

    # -----------------------------------------------------------------------
    Test.@testset "CoarseGrainingFlux — CGEF loaded" begin
        # CoarseGrainingEnergyFluxes is loaded at the top of this file, so the call should succeed
        N = 4; L = 2π
        x = [L * (i-1) / N for i in 1:N]
        y = [L * (j-1) / N for j in 1:N]
        u = zeros(N, N); v = zeros(N, N)
        result = FET.calculate_coarse_graining_flux(
            (u, v), (x, y), π/2, FET.GaussianFilter())
        Test.@test result isa FET.CoarseGrainingFluxResult
    end

    # -----------------------------------------------------------------------
    Test.@testset "calculate_energy_transfer — unified dispatch" begin
        N = 8; L = 2π
        ks = FET.wavenumber_grid((N,), (L,))
        û = zeros(ComplexF64, N, 1)

        r1 = FET.calculate_energy_transfer(
            FET.SpectralFluxMethod(FET.LinearBinning(2π/L)), û, ks)
        Test.@test r1 isa FET.SpectralFluxResult

        r2 = FET.calculate_energy_transfer(
            FET.ShellToShellTransferMethod(FET.LinearBinning(2π/L)), û, ks)
        Test.@test r2 isa FET.ShellToShellResult

        x = [L * (i-1) / N for i in 1:N]
        y = [L * (j-1) / N for j in 1:N]
        u = zeros(N, N); v = zeros(N, N)
        r3 = FET.calculate_energy_transfer(
            FET.CoarseGrainingFluxMethod(FET.GaussianFilter(), Float64(π/2)),
            (u, v), (x, y))
        Test.@test r3 isa FET.CoarseGrainingFluxResult
    end

    # -----------------------------------------------------------------------
    Test.@testset "assign_shells" begin
        ks  = FET.wavenumber_grid((8,), (2π,))
        k_mag = FET.wavenumber_magnitude_grid(ks)
        b     = FET.LinearBinning(2π/8)
        edges = FET.shell_edges(b, maximum(k_mag))
        idx   = FET.assign_shells(k_mag, edges)
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
        ks  = FET.wavenumber_grid((N,), (L,))
        û  = zeros(ComplexF64, N, 1)
        b   = FET.LinearBinning(2π/L)
        ws  = FET.SpectralFluxWorkspace(û, ks, b)
        k_mag     = FET.wavenumber_magnitude_grid(ks)
        edges     = FET.shell_edges(b, maximum(k_mag))
        centers   = FET.shell_centers(b, maximum(k_mag))
        shell_idx = FET.assign_shells(k_mag, edges)
        result    = FET.SpectralFluxResult(centers, similar(ws.T_spec), similar(ws.flux))
        FET.calculate_spectral_flux!(result, ws, û, ks, shell_idx; dealiasing=false)
        Test.@test result isa FET.SpectralFluxResult
        Test.@test all(abs.(result.transfer_spectrum) .< 1e-14)
    end

    # -----------------------------------------------------------------------
    Test.@testset "ShellToShellTransfer !-variant" begin
        N = 6; L = 2π
        ks = FET.wavenumber_grid((N, N), (L, L))
        û  = zeros(ComplexF64, N, N, 2)
        b   = FET.LinearBinning(2π/L)
        ws  = FET.ShellToShellWorkspace(û, ks, b)
        k_mag   = FET.wavenumber_magnitude_grid(ks)
        edges   = FET.shell_edges(b, maximum(k_mag))
        centers = FET.shell_centers(b, maximum(k_mag))
        N_sh    = length(centers)
        FT      = Float64
        result  = FET.ShellToShellResult(
            centers, edges,
            Matrix{FT}(undef, N_sh, N_sh),
            Vector{FT}(undef, N_sh),
            FT(NaN),
        )
        FET.calculate_shell_to_shell_transfer!(result, ws, û, ks;
            dealiasing=false, verify_antisymmetry=false)
        Test.@test result isa FET.ShellToShellResult
        Test.@test all(abs.(result.transfer_matrix) .< 1e-14)
    end

    # -----------------------------------------------------------------------
    Test.@testset "Float32 propagation" begin
        N = 8
        Ls = (Float32(2π), Float32(2π))
        ks = FET.wavenumber_grid((N, N), Ls)
        Test.@test eltype(ks[1]) == Float32
        Test.@test eltype(ks[2]) == Float32
        k_mag = FET.wavenumber_magnitude_grid(ks)
        Test.@test eltype(k_mag) == Float32
        b     = FET.LinearBinning(Float32(2π) / N)
        edges = FET.shell_edges(b, maximum(k_mag))
        Test.@test eltype(edges) == Float32
        centers = FET.shell_centers(b, maximum(k_mag))
        Test.@test eltype(centers) == Float32

        û = zeros(ComplexF32, N, N, 2)
        result = FET.calculate_spectral_flux(û, ks;
            binning=b, dealiasing=false, spectral=FET.DirectSumBackend())
        Test.@test result isa FET.SpectralFluxResult
        Test.@test eltype(result.k_shells) == Float32
        Test.@test eltype(result.transfer_spectrum) == Float32
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
            FET.TriadicOrthogonalDecomposition.parse_parameters(nt, nx)
        Test.@test length(win_vec) == nDFT
        Test.@test length(weight_vec) == nx
        Test.@test nBlks >= 2

        # Error cases for parameters
        Test.@test_throws ArgumentError FET.TriadicOrthogonalDecomposition.parse_parameters(nt, nx; window=3)  # nDFT < 4
        Test.@test_throws ArgumentError FET.TriadicOrthogonalDecomposition.parse_parameters(nt, nx; window=zeros(3)) # nDFT < 4
        Test.@test_throws ArgumentError FET.TriadicOrthogonalDecomposition.parse_parameters(nt, nx; noverlap=256) # noverlap >= nDFT

        # 2. SVD helper functions
        # Sirovich SVD
        M = randn(4, 10)
        U, s, V = FET.TriadicOrthogonalDecomposition.sirovich_svd(M)
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
        U_lr, s_lr, V_lr = FET.TriadicOrthogonalDecomposition.lowrank_svd(X_lr, Q3)
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
        res = FET.triadic_orthogonal_decomposition(X; dt=dt_sig, isreal_data=true)
        Test.@test res isa FET.TriadicOrthogonalDecompositionResult
        Test.@test res.frequencies isa AbstractVector
        Test.@test all(res.mode_bispectrum .>= 0.0 .|| isnan.(res.mode_bispectrum))
        Test.@test all(res.modal_energy_budget .>= 0.0 .|| res.modal_energy_budget .<= 0.0 .|| isnan.(res.modal_energy_budget))

        # Check default dispatch via calculate_energy_transfer
        method = FET.TriadicOrthogonalDecompositionMethod(nfft=64, noverlap=32, nmode=2)
        res_dispatch = FET.calculate_energy_transfer(method, X; dt=dt_sig)
        Test.@test res_dispatch isa FET.TriadicOrthogonalDecompositionResult
        Test.@test size(res_dispatch.mode_bispectrum, 3) == 2

        # 4. FFTBackend consistency
        res_serial = FET.triadic_orthogonal_decomposition(X; dt=dt_sig, spectral=FET.DirectSumBackend())
        res_fft = FET.triadic_orthogonal_decomposition(X; dt=dt_sig, spectral=FET.FFTBackend())
        Test.@test isapprox(res_serial.frequencies, res_fft.frequencies)
        Test.@test isapprox(filter(!isnan, res_serial.mode_bispectrum), filter(!isnan, res_fft.mode_bispectrum); atol=1e-12)

        # 5. ThreadedBackend — OhMyThreads is loaded so it should work
        res_threaded = FET.triadic_orthogonal_decomposition(X; dt=dt_sig, execution=FET.ThreadedBackend())
        Test.@test res_threaded isa FET.TriadicOrthogonalDecompositionResult
        Test.@test isapprox(res_serial.frequencies, res_threaded.frequencies)

        # 6. Coefficients and auxiliary modes
        res_aux = FET.triadic_orthogonal_decomposition(X; dt=dt_sig, return_coefficients=true, return_auxiliary_modes=true)
        Test.@test res_aux.expansion_coefficients isa Dict
        Test.@test res_aux.auxiliary_modes isa Dict
    end

    # -----------------------------------------------------------------------
    Test.@testset "Field Decomposition (Helmholtz / Partial Flux)" begin
        # 1. Spectral flux decomposition test
        N = 8; L = 2π
        ks = FET.wavenumber_grid((N, N), (L, L))
        û = zeros(ComplexF64, N, N, 2)
        û[2, 1, 1] = 0.5; û[N, 1, 1] = 0.5   # k=(1,0) in u
        û[1, 2, 2] = 0.5; û[1, N, 2] = 0.5   # k=(0,1) in v

        res_none = FET.calculate_spectral_flux(û, ks; decomposition=FET.NoDecomposition(), dealiasing=false)
        res_helm = FET.calculate_spectral_flux(û, ks; decomposition=FET.HelmholtzDecomposition(), dealiasing=false)
        res_rot  = FET.calculate_spectral_flux(û, ks; decomposition=FET.RotationalDecomposition(), dealiasing=false)
        res_div  = FET.calculate_spectral_flux(û, ks; decomposition=FET.DivergentDecomposition(), dealiasing=false)

        Test.@test res_none isa FET.SpectralFluxResult
        Test.@test res_helm isa NamedTuple
        Test.@test haskey(res_helm, :rotational) && haskey(res_helm, :divergent)
        Test.@test res_rot isa FET.SpectralFluxResult
        Test.@test res_div isa FET.SpectralFluxResult

        # For these divergence-free/rotational modes, verify consistency:
        # T_none ≈ T_rot + T_div
        Test.@test isapprox(res_none.transfer_spectrum, res_rot.transfer_spectrum + res_div.transfer_spectrum; atol=1e-12)

        # 2. Coarse-graining flux decomposition test
        x = range(0, L; length=N+1)[1:N]
        y = range(0, L; length=N+1)[1:N]
        u = [cos(x) for x in x, y in y]
        v = [sin(y) for x in x, y in y]

        cg_none = FET.calculate_coarse_graining_flux((u, v), (x, y), 1.0, FET.GaussianFilter(); decomposition=FET.NoDecomposition())
        cg_helm = FET.calculate_coarse_graining_flux((u, v), (x, y), 1.0, FET.GaussianFilter(); decomposition=FET.HelmholtzDecomposition())
        cg_rot  = FET.calculate_coarse_graining_flux((u, v), (x, y), 1.0, FET.GaussianFilter(); decomposition=FET.RotationalDecomposition())
        cg_div  = FET.calculate_coarse_graining_flux((u, v), (x, y), 1.0, FET.GaussianFilter(); decomposition=FET.DivergentDecomposition())

        Test.@test cg_none isa FET.CoarseGrainingFluxResult
        Test.@test cg_helm isa NamedTuple
        Test.@test haskey(cg_helm, :rotational) && haskey(cg_helm, :divergent)
        Test.@test cg_rot isa FET.CoarseGrainingFluxResult
        Test.@test cg_div isa FET.CoarseGrainingFluxResult
    end

    # -----------------------------------------------------------------------
    Test.@testset "Parallel Backends Parity (Threaded / Distributed)" begin
        # Add workers if not present
        if Distributed.nprocs() == 1
            Distributed.addprocs(2)
        end
        # Load the package and extensions on all workers
        Distributed.@everywhere using FlowInvariantTransfer: FlowInvariantTransfer as FET
        Distributed.@everywhere using SharedArrays

        # Create sample data
        Random.seed!(42)
        N = 8; L = 2π
        ks = FET.wavenumber_grid((N, N), (L, L))
        û = zeros(ComplexF64, N, N, 2)
        û[2, 1, 1] = 0.5; û[N, 1, 1] = 0.5
        û[1, 2, 2] = 0.5; û[1, N, 2] = 0.5

        # 1. Shell-to-Shell Transfer Parity
        b = FET.LinearBinning(2π / L)
        res_serial = FET.calculate_shell_to_shell_transfer(û, ks; binning=b, dealiasing=true, verify_antisymmetry=true, execution=FET.SerialBackend())
        res_thread = FET.calculate_shell_to_shell_transfer(û, ks; binning=b, dealiasing=true, verify_antisymmetry=true, execution=FET.ThreadedBackend())
        
        # For DistributedBackend, we convert velocity_hat to a SharedArray so workers can read it efficiently
        s_û = SharedArrays.SharedArray(û)
        res_dist = FET.calculate_shell_to_shell_transfer(s_û, ks; binning=b, dealiasing=true, verify_antisymmetry=true, execution=FET.DistributedBackend())

        Test.@test isapprox(res_serial.transfer_matrix, res_thread.transfer_matrix; atol=1e-12)
        Test.@test isapprox(res_serial.transfer_matrix, res_dist.transfer_matrix; atol=1e-12)
        Test.@test isapprox(res_serial.net_transfer, res_thread.net_transfer; atol=1e-12)
        Test.@test isapprox(res_serial.net_transfer, res_dist.net_transfer; atol=1e-12)
    end

    # -----------------------------------------------------------------------
    Test.@testset "ModeToMode — invariant/dimension guards" begin
        L = 2π
        # 2D field + Helicity() must error (Helicity is 3D-only); routed via transfer_density.
        ks2 = FET.wavenumber_grid((4, 4), (L, L))
        û2  = zeros(ComplexF64, 4, 4, 2); û2[2, 1, 1] = 0.5; û2[1, 2, 2] = 0.5
        Test.@test_throws ArgumentError FET.calculate_mode_to_mode_transfer(û2, ks2; invariant=FET.Helicity())
        # 3D field + Enstrophy() now works (vector-vorticity transfer, routed).
        ks3 = FET.wavenumber_grid((4, 4, 4), (L, L, L))
        û3  = zeros(ComplexF64, 4, 4, 4, 3); û3[2, 1, 1, 1] = 0.5
        Test.@test FET.calculate_mode_to_mode_transfer(û3, ks3; invariant=FET.Enstrophy()) isa FET.ModeToModeTriadResult
        # KineticEnergy works in both dimensionalities.
        Test.@test FET.calculate_mode_to_mode_transfer(û2, ks2; invariant=FET.KineticEnergy()) isa FET.ModeToModeTriadResult
        Test.@test FET.calculate_mode_to_mode_transfer(û3, ks3; invariant=FET.KineticEnergy()) isa FET.ModeToModeTriadResult
    end

    # -----------------------------------------------------------------------
    Test.@testset "ModeToMode — resolved S(k|p): antisym, conserves, reduces to spectral/shell-to-shell" begin
        # mode-to-mode now owns the fully-resolved S(k|p) (built from the validated nonlinear
        # term), which must be antisymmetric, conserve, and reduce to the coarser diagnostics.
        N = 12; L = 2π
        Random.seed!(21)
        ψ  = randn(N, N); ψh = FFTW.fft(ψ) ./ N^2
        ks = FET.wavenumber_grid((N, N), (L, L))
        kx = [ks[1][i] for i in 1:N, j in 1:N]; ky = [ks[2][j] for i in 1:N, j in 1:N]
        û  = cat(im .* ky .* ψh, -im .* kx .* ψh; dims = 3)
        m2m = FET.calculate_mode_to_mode_transfer(û, ks; dealiasing = true, spectral = FET.FFTBackend())
        S   = m2m.transfer                              # shape (N,N,N,N): S[k..., p...]
        nrm = sqrt(sum(abs2, S)); Test.@test nrm > 0    # non-degenerate
        asym = 0.0
        for k in CartesianIndices((N, N)), p in CartesianIndices((N, N))
            asym = max(asym, abs(S[k, p] + S[p, k]))
        end
        Test.@test asym < 1e-10 * nrm                   # antisymmetric S(k|p) = −S(p|k)
        Test.@test abs(sum(S)) < 1e-10 * nrm            # conserves Σ_kΣ_p S = 0

        b = FET.LinearBinning(2π/L)
        sf = FET.calculate_spectral_flux(û, ks; binning = b, dealiasing = true, spectral = FET.FFTBackend())
        ss = FET.calculate_shell_to_shell_transfer(û, ks; binning = b, dealiasing = true,
            verify_antisymmetry = false, spectral = FET.FFTBackend())
        kmag  = FET.wavenumber_magnitude_grid(ks)
        edges = FET.shell_edges(b, maximum(kmag))
        sidx  = FET.assign_shells(kmag, edges)
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

end # top-level testset
