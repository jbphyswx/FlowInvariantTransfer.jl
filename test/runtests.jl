using Test: Test
using Random: Random
using Statistics: Statistics
using Aqua: Aqua
using FFTW: FFTW

using FlowEnergyTransfer: FlowEnergyTransfer as FET

Test.@testset "FlowEnergyTransfer.jl Test Suite" begin

    # -----------------------------------------------------------------------
    Test.@testset "Aqua Code Quality" begin
        Aqua.test_all(FET; ambiguities = false)
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

    Test.@testset "ShellBinning — shell_mask" begin
        ks = FET.wavenumber_grid((8,), (2π,))
        k_mag_1d = FET.wavenumber_magnitude_grid(ks)
        b = FET.LinearBinning(2π / 8)
        edges = FET.shell_edges(b, maximum(k_mag_1d))
        mask1 = FET.shell_mask(k_mag_1d, edges, 1)
        Test.@test any(mask1)
        Test.@test !any(mask1 .& FET.shell_mask(k_mag_1d, edges, 2))  # no overlap
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
            binning=FET.LinearBinning(2π/L), dealiasing=false, backend=FET.SerialBackend())

        # FFTW path (extension)
        result_fft = FET.calculate_spectral_flux(û_phys, ks;
            binning=FET.LinearBinning(2π/L), dealiasing=false, backend=FET.FFTBackend())

        Test.@test isapprox(result_direct.transfer_spectrum,
                             result_fft.transfer_spectrum; atol=1e-10)
        Test.@test isapprox(result_direct.flux, result_fft.flux; atol=1e-10)
    end

    # -----------------------------------------------------------------------
    Test.@testset "ShellToShellTransfer — antisymmetry (divergence-free field)" begin
        # T(n,m) = -T(m,n) holds exactly for divergence-free (incompressible) fields.
        # Build u = ∂ψ/∂y, v = -∂ψ/∂x from a random streamfunction ψ.
        Random.seed!(7)
        N = 8; L = 2π
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
            verify_antisymmetry=true, backend=FET.FFTBackend())

        Test.@test result isa FET.ShellToShellResult
        T_norm = sqrt(sum(abs2, result.transfer_matrix))
        # T(n,m) = -T(m,n) exactly by construction; verify to machine precision
        Test.@test result.max_antisymmetry_error < 1e-12 * T_norm
    end

    # -----------------------------------------------------------------------
    Test.@testset "ShellToShellTransfer — FFTW vs direct (small 2D)" begin
        Random.seed!(13)
        N = 6; L = 2π
        ks = FET.wavenumber_grid((N, N), (L, L))
        û = zeros(ComplexF64, N, N, 2)
        û[2, 1, 1] = 0.4; û[N, 1, 1] = 0.4   # k=(1,0) in u
        û[1, 2, 2] = 0.3; û[1, N, 2] = 0.3   # k=(0,1) in v

        b = FET.LinearBinning(2π/L)

        result_direct = FET.calculate_shell_to_shell_transfer(û, ks;
            binning=b, dealiasing=false, verify_antisymmetry=false, backend=FET.SerialBackend())
        result_fft = FET.calculate_shell_to_shell_transfer(û, ks;
            binning=b, dealiasing=false, verify_antisymmetry=false, backend=FET.FFTBackend())

        Test.@test isapprox(result_direct.transfer_matrix,
                             result_fft.transfer_matrix; atol=1e-10)
    end

    # -----------------------------------------------------------------------
    Test.@testset "CoarseGrainingFlux — requires CGEF (helpful error)" begin
        # Without CoarseGrainingEnergyFluxes loaded the stub should throw a clear error
        N = 4; L = 2π
        x = [L * (i-1) / N for i in 1:N]
        y = [L * (j-1) / N for j in 1:N]
        u = zeros(N, N); v = zeros(N, N)
        Test.@test_throws ArgumentError FET.calculate_coarse_graining_flux(
            (u, v), (x, y), π/2, FET.GaussianFilter())
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
        Test.@test_throws ArgumentError FET.calculate_energy_transfer(
            FET.CoarseGrainingFluxMethod(FET.GaussianFilter(), Float64(π/2)),
            (u, v), (x, y))
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
            binning=b, dealiasing=false, backend=FET.SerialBackend())
        Test.@test result isa FET.SpectralFluxResult
        Test.@test eltype(result.k_shells) == Float32
        Test.@test eltype(result.transfer_spectrum) == Float32
    end

end # top-level testset
