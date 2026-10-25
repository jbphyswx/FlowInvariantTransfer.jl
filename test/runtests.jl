using Test: Test
using Random: Random
using Aqua: Aqua
using FFTW: FFTW
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
using FlowGeometries: FlowGeometries as FG

using FlowInvariantTransfer: FlowInvariantTransfer as FIT
using ComputationalBackends: ComputationalBackends
using SpectralBackends: SpectralBackends
using FlowTransformBindings: FlowTransformBindings as FTB
using JLArrays: JLArrays   # reference GPU-array backend: allowscalar(false) → catches non-device-generic code

# -----------------------------------------------------------------------
Test.@testset "ShellBinning — LinearBinning" begin
    b = FIT.Types.LinearBinning(1.0)
    Test.@test FIT.ShellBinning.shell_edges(b, 5.0) == 0.0:5.0
    Test.@test FIT.ShellBinning.shell_edges(b, 5.5) == [0.0:5.0; 5.5]
    Test.@test FIT.ShellBinning.shell_centers(b, 5.0) == 0.5:4.5
end

Test.@testset "ShellBinning — LogarithmicBinning" begin
    b = FIT.Types.LogarithmicBinning(1.0, 2.0)
    Test.@test FIT.ShellBinning.shell_edges(b, 20.0) == [0.0, 1, 2, 4, 8, 16, 32]
    Test.@test FIT.ShellBinning.shell_edges(b, 12.0) == [0.0, 1, 2, 4, 8, 16]
    Test.@test FIT.ShellBinning.shell_centers(b, 12.0) ≈ [0.5, √2, √8, √32, √128]
end

Test.@testset "ShellBinning — every mode lies in the shell its coordinate falls in" begin
    binnings = (FIT.Types.LinearBinning(1.0), FIT.Types.LinearBinning(0.7),
                FIT.Types.LogarithmicBinning(1.5, 1.7), FIT.Types.DyadicBinning(2.0))
    geometries = (FIT.Types.IsotropicShells(), FIT.Types.PerpendicularShells(), FIT.Types.ParallelShells())
    for ns in ((8, 8, 8), (9, 7, 6)), g in geometries, b in binnings
        ks = FIT.Utils.wavenumber_grid(ns, (2π, 2π, 2π))
        k = FIT.ShellBinning.shell_coordinate(g, ks)
        edges = FIT.ShellBinning.shell_edges(b, maximum(k))
        idx = FIT.ShellBinning.assign_shells(k, edges)
        Nsh = length(edges) - 1
        Test.@test all(I -> 1 <= idx[I] <= Nsh &&
                            edges[idx[I]] <= k[I] && (k[I] < edges[idx[I] + 1] || idx[I] == Nsh),
                       CartesianIndices(k))
    end
    # So the flux through the last shell is the transfer density summed over every mode.
    for ns in ((8, 8, 8), (9, 7, 6)), g in geometries, b in binnings
        ks = FIT.Utils.wavenumber_grid(ns, (2π, 2π, 2π))
        û = randn(Random.MersenneTwister(3), ComplexF64, ns..., 3)
        N̂ = FIT.NonlinearTerm.compute_nonlinear_term(û, ks; dealiasing = FIT.Types.NoDealiasing(),
                                                     transform = SpectralBackends.FFTSpectralBackend())
        t = FIT.Invariants.transfer_density(FIT.Types.KineticEnergy(), û, N̂, ks)
        r = FIT.calculate_spectral_flux(û, ks; binning = b, geometry = g, dealiasing = FIT.Types.NoDealiasing(),
                                        transform = SpectralBackends.FFTSpectralBackend())
        Test.@test r.flux[end] ≈ sum(t) atol = 1e-12 * sum(abs, t)
    end
end

# -----------------------------------------------------------------------
Test.@testset "Filters — spectral responses" begin
    k = 2.0; ℓ = 1.0
    # Pope (2000), Table 13.2: Ĝ = exp(−k²ℓ²/24) for the Gaussian, sin(kℓ/2)/(kℓ/2) for the top hat.
    Test.@test FIT.Filters.filter_response(FIT.Types.GaussianFilter(), k, ℓ) ≈ exp(-(k * ℓ)^2 / 24)
    Test.@test FIT.Filters.filter_response(FIT.Types.TopHatFilter(), k, ℓ) ≈ sin(k * ℓ / 2) / (k * ℓ / 2)
end

Test.@testset "Filters — apply_filter_spectral!" begin
    # The sharp filter passes k < π/ℓ.
    k_mag = Float64[0, 1, 2, 3, 4]
    û_out = similar(k_mag, ComplexF64)
    FIT.Filters.apply_filter_spectral!(û_out, ones(ComplexF64, 5), k_mag, FIT.Types.SharpSpectralFilter(), 1.0)
    Test.@test û_out == [1, 1, 1, 1, 0]
end

# -----------------------------------------------------------------------
Test.@testset "SpectralFlux — energy conservation Σ T(k) ≈ 0 (dealiased, div-free)" begin
    # A divergence-free 2D field from a streamfunction ψ: û_x = i k_y ψ̂, û_y = −i k_x ψ̂. The 2/3 rule
    # truncates the inputs, so the pseudospectral nonlinear term conserves energy exactly,
    # Σ_k Re{û*·N̂} = 0, by discrete skew-symmetry.
    N = 16; L = 2π
    Random.seed!(7)
    ψ  = randn(N, N)
    ψh = FFTW.fft(ψ) ./ N^2
    ks = FIT.Utils.wavenumber_grid((N, N), (L, L))
    kx = [ks[1][i] for i in 1:N, j in 1:N]
    ky = [ks[2][j] for i in 1:N, j in 1:N]
    û  = cat(im .* ky .* ψh, -im .* kx .* ψh; dims = 3)
    Test.@test maximum(abs.(kx .* û[:, :, 1] .+ ky .* û[:, :, 2])) < 1e-12  # div-free

    for transform in (SpectralBackends.DirectSumSpectralBackend(), SpectralBackends.FFTSpectralBackend())
        N̂ = FIT.NonlinearTerm.compute_nonlinear_term(û, ks; dealiasing = FIT.Types.OrszagTwoThirds(), transform)
        t = FIT.Invariants.transfer_density(FIT.Types.KineticEnergy(), û, N̂, ks)
        scale = sum(abs, t)
        Test.@test abs(sum(t)) < 1e-10 * scale       # energy-conserving, alias-free
    end
end

# -----------------------------------------------------------------------
Test.@testset "An analytic triad fixes the transfer and the flux sign" begin
    # ψ = Σ_j (A_j e^{ik_j·x} + c.c.) with k₁ + k₂ = k₃ and u = ẑ×∇ψ = (−∂_yψ, ∂_xψ). Mode j holds
    # E_j = |k_j|²|A_j|², and the vorticity equation ζ_t + u·∇ζ = 0 (ζ = ∇²ψ) gives, with S = k₁×k₂ and
    # R = Re{A₁A₂Ā₃},
    #   dE₁/dt = 2S(|k₃|² − |k₂|²)R,  dE₂/dt = 2S(|k₁|² − |k₃|²)R,  dE₃/dt = 2S(|k₂|² − |k₁|²)R.
    # Every transfer is T = −dE/dt and the flux through K is Π(K) = Σ_{|k|<K} T.
    N = 16; L = 2π
    kk = ((1, 1), (1, -3), (2, -2))                 # |k|² = 2, 10, 8: one mode per unit shell
    A = (0.7, 0.5, 0.4)
    S = kk[1][1] * kk[2][2] - kk[1][2] * kk[2][1]
    R = prod(A)
    n2(k) = k[1]^2 + k[2]^2
    dE = (2S * (n2(kk[3]) - n2(kk[2])) * R, 2S * (n2(kk[1]) - n2(kk[3])) * R, 2S * (n2(kk[2]) - n2(kk[1])) * R)
    x = range(0, L; length = N + 1)[1:N]
    ψ = [2 * sum(A[j] * cos(kk[j][1] * xi + kk[j][2] * yj) for j in 1:3) for xi in x, yj in x]
    ψh = FFTW.fft(ψ) ./ N^2
    ks = FIT.Utils.wavenumber_grid((N, N), (L, L))
    kx = [ks[1][i] for i in 1:N, j in 1:N]
    ky = [ks[2][j] for i in 1:N, j in 1:N]
    û = cat(-im .* ky .* ψh, im .* kx .* ψh; dims = 3)
    idx(k) = CartesianIndex(mod(k[1], N) + 1, mod(k[2], N) + 1)
    for transform in (SpectralBackends.DirectSumSpectralBackend(), SpectralBackends.FFTSpectralBackend())
        N̂ = FIT.NonlinearTerm.compute_nonlinear_term(û, ks; dealiasing = FIT.Types.NoDealiasing(), transform)
        t = FIT.Invariants.transfer_density(FIT.Types.KineticEnergy(), û, N̂, ks)
        for j in 1:3
            Test.@test t[idx(kk[j])] + t[idx(.-kk[j])] ≈ -dE[j] rtol = 1e-12
        end
    end
    # The middle mode gives 16R down to |k| = √2 and 48R up to |k| = √10, so the flux is −16R through
    # K = 2 and +48R, a forward cascade, through K = 3. The compressible transfer at ρ ≡ 1 is the same.
    b = FIT.Types.LinearBinning(1.0)
    r = FIT.SpectralFlux.calculate_spectral_flux(û, ks; binning = b, dealiasing = FIT.Types.NoDealiasing())
    ρ̂ = zeros(ComplexF64, N, N); ρ̂[1, 1] = 1
    rc = FIT.Compressible.calculate_compressible_flux(û, ρ̂, ks; binning = b, decompose = false,
        dealiasing = FIT.Types.NoDealiasing())
    for res in (r, rc)
        # `flux[n]` is the flux through `shell_edges[n+1]`.
        through(K) = res.flux[findfirst(==(K), res.shell_edges) - 1]
        Test.@test through(2) ≈ -16R rtol = 1e-12
        Test.@test through(3) ≈ 48R rtol = 1e-12
    end
end

# -----------------------------------------------------------------------
# The wavenumber grids, and shell sums that agree on a real field's half and full spectra.
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
# Spherical transfer routed from a grid's own sampling: the exact transform on the Clenshaw–Curtis
# nodes it is defined on, the scattered least-squares fit on every other sphere sampling.
include("test_spherical_grids.jl")

# -----------------------------------------------------------------------
# Batch entries: one workspace reused across a worker's snapshots, returning what the per-snapshot
# calls return.
include("test_batch_entries.jl")
include("test_band_energies.jl")
include("test_decompose_physical.jl")

# -----------------------------------------------------------------------
# Mode-to-mode transfer between stored coefficients, decomposition-channel fluxes, scattered analysis
# as a quadrature, and single-precision inputs.
include("test_mode_to_mode.jl")
include("test_partial_fluxes.jl")
include("test_scattered_quadrature.jl")
include("test_precision.jl")

# -----------------------------------------------------------------------
# Allocation contract of the in-place paths.
include("test_allocs.jl")

# -----------------------------------------------------------------------
Test.@testset "Enstrophy — 2D conserved" begin
    L = 2π
    # Enstrophy is a 2D inviscid invariant: Σ_k T_Ω ≈ 0 for a dealiased divergence-free field.
    N = 16
    Random.seed!(5)
    ψ  = randn(N, N); ψh = FFTW.fft(ψ) ./ N^2
    ks2 = FIT.Utils.wavenumber_grid((N, N), (L, L))
    kx = [ks2[1][i] for i in 1:N, j in 1:N]; ky = [ks2[2][j] for i in 1:N, j in 1:N]
    û2 = cat(im .* ky .* ψh, -im .* kx .* ψh; dims = 3)
    N̂2 = FIT.NonlinearTerm.compute_nonlinear_term(û2, ks2; dealiasing = FIT.Types.OrszagTwoThirds(), transform=SpectralBackends.FFTSpectralBackend())
    tΩ2 = FIT.Invariants.transfer_density(FIT.Types.Enstrophy(), û2, N̂2, ks2)
    Test.@test abs(sum(tΩ2)) < 1e-9 * sum(abs, tΩ2)
end

# -----------------------------------------------------------------------
Test.@testset "NonlinearTerm — generalized advection of an M=1 scalar (u·∇)θ" begin
    # A passive scalar (M = 1) advected by a divergence-free 2D velocity: DirectSum and FFT agree, and
    # the variance is conserved, Σ_k Re{θ̂* N̂_θ} = 0, since ∫θ(u·∇)θ = −½∫θ²(∇·u) = 0.
    N = 12; L = 2π
    Random.seed!(31)
    ks = FIT.Utils.wavenumber_grid((N, N), (L, L))
    kx = [ks[1][i] for i in 1:N, j in 1:N]; ky = [ks[2][j] for i in 1:N, j in 1:N]
    ψh = FFTW.fft(randn(N, N)) ./ N^2
    û  = cat(im .* ky .* ψh, -im .* kx .* ψh; dims = 3)        # div-free velocity (D=2)
    θ̂  = reshape(FFTW.fft(randn(N, N)) ./ N^2, N, N, 1)        # passive scalar (M=1)

    N̂_dir = FIT.NonlinearTerm.compute_nonlinear_term(θ̂, ks; dealiasing = FIT.Types.OrszagTwoThirds(),
        transform = SpectralBackends.DirectSumSpectralBackend(), advecting_hat = û)
    N̂_fft = FIT.NonlinearTerm.compute_nonlinear_term(θ̂, ks; dealiasing = FIT.Types.OrszagTwoThirds(),
        transform = SpectralBackends.FFTSpectralBackend(), advecting_hat = û)
    Test.@test isapprox(N̂_dir, N̂_fft; atol = 1e-10 * maximum(abs, N̂_fft))

    tθ = FIT.Invariants.transfer_density(FIT.Types.PassiveScalar(), θ̂, N̂_fft, ks)
    Test.@test abs(sum(tθ)) < 1e-9 * sum(abs, tθ)
end

# -----------------------------------------------------------------------
Test.@testset "Dealiasing — exact 3/2 padding (PaddedThreeHalves)" begin
    # On a field band-limited to |k| < N/3 the 3/2-padded nonlinear term equals the 2/3 one on the
    # shared retained band, and carries the N/3 ≤ |k| < N/2 content the 2/3 rule discards.
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

    N23  = FIT.NonlinearTerm.compute_nonlinear_term(û, ks; dealiasing = FIT.Types.OrszagTwoThirds(),   transform = SpectralBackends.FFTSpectralBackend())
    Npad = FIT.NonlinearTerm.compute_nonlinear_term(û, ks; dealiasing = FIT.Types.PaddedThreeHalves(), transform = SpectralBackends.FFTSpectralBackend())
    low = repeat(kmag .< 8, 1, 1, 2)
    mid = repeat((kmag .>= 8) .& (kmag .< 12), 1, 1, 2)
    Test.@test maximum(abs.(N23 .- Npad)[low]) < 1e-10 * maximum(abs.(Npad)[low])   # agree on |k|<N/3
    Test.@test sum(abs2, Npad[mid]) > 3 * sum(abs2, N23[mid])                       # padded keeps more

    # Padded spectral flux still conserves for an incompressible field.
    b  = FIT.Types.LinearBinning(2π/L)
    sf = FIT.SpectralFlux.calculate_spectral_flux(û, ks; binning = b, dealiasing = FIT.Types.PaddedThreeHalves(), transform = SpectralBackends.FFTSpectralBackend())
    Test.@test abs(sum(sf.transfer_spectrum)) < 1e-9 * sum(abs, sf.transfer_spectrum)

    # Exact to Nyquist: u = (0, b cos(n x/2) cos y) gives u_y ∂_y u_y = −(b²/4)(1 + cos n x) sin 2y, whose
    # truncation to the grid is −(b²/4) sin 2y: N̂_y = ±i b²/8 at k = (0, ±2) and zero elsewhere, the
    # x-Nyquist entries included (cos n x is the product of the two Nyquist halves). The direct sums
    # form the same padded product.
    for n in (16, 12), real_layout in (false, true),
        transform in (SpectralBackends.FFTSpectralBackend(), SpectralBackends.DirectSumSpectralBackend())
        bq = 0.7
        xs = range(0, 2π; length = n + 1)[1:n]
        uy = [bq * cos(n * x / 2) * cos(y) for x in xs, y in xs]
        ksq = FIT.Utils.wavenumber_grid((n, n), (2π, 2π); real = real_layout)
        tr(f) = real_layout ? FFTW.rfft(f) ./ n^2 : FFTW.fft(f) ./ n^2
        ûq = cat(tr(zeros(n, n)), tr(uy); dims = 3)
        N̂q = FIT.NonlinearTerm.compute_nonlinear_term(ûq, ksq; dealiasing = FIT.Types.PaddedThreeHalves(),
                                                      transform = transform)
        ref = zeros(ComplexF64, size(ûq))
        ref[1, 3, 2] = im * bq^2 / 8
        ref[1, n - 1, 2] = -im * bq^2 / 8
        Test.@test maximum(abs, N̂q .- ref) < 1e-13 * bq^2
    end

    # A broadband field, Nyquist modes included: the direct-sum padded term equals the FFT one.
    for n in (8, 7), real_layout in (false, true)
        rngd = Random.MersenneTwister(n)
        ksd = FIT.Utils.wavenumber_grid((n, n), (2π, 2π); real = real_layout)
        trd(f) = real_layout ? FFTW.rfft(f) ./ n^2 : FFTW.fft(f) ./ n^2
        ûd = cat(trd(randn(rngd, n, n)), trd(randn(rngd, n, n)); dims = 3)
        pad = FIT.Types.PaddedThreeHalves()
        N̂f = FIT.NonlinearTerm.compute_nonlinear_term(ûd, ksd; dealiasing = pad,
                                                      transform = SpectralBackends.FFTSpectralBackend())
        N̂d = FIT.NonlinearTerm.compute_nonlinear_term(ûd, ksd; dealiasing = pad,
                                                      transform = SpectralBackends.DirectSumSpectralBackend())
        Test.@test maximum(abs, N̂f) > 1e-6
        Test.@test maximum(abs, N̂d .- N̂f) < 1e-12 * maximum(abs, N̂f)
    end
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
        transform=SpectralBackends.DirectSumSpectralBackend())
    r_fft = FIT.ShellToShellTransfer.calculate_shell_to_shell_transfer(û, ks; binning=b, dealiasing = FIT.Types.OrszagTwoThirds(),
        transform=SpectralBackends.FFTSpectralBackend())

    T_norm = sqrt(sum(abs2, r_direct.transfer_matrix))
    Test.@test T_norm > 0                                            # non-degenerate
    Test.@test isapprox(r_direct.transfer_matrix, r_fft.transfer_matrix; atol = 1e-9 * T_norm)
    Test.@test maximum(abs, r_direct.transfer_matrix .+ r_direct.transfer_matrix') < 1e-10 * T_norm   # T(n,m) = −T(m,n)

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
        dealiasing = FIT.Types.OrszagTwoThirds(), transform = SpectralBackends.DirectSumSpectralBackend())
    r_fft = FIT.calculate_scalar_shell_to_shell_transfer(û, θ, ks; binning = b,
        dealiasing = FIT.Types.OrszagTwoThirds(), transform = SpectralBackends.FFTSpectralBackend())

    T_norm = sqrt(sum(abs2, r_dir.transfer_matrix))
    Test.@test T_norm > 0                                                  # non-degenerate
    Test.@test isapprox(r_dir.transfer_matrix, r_fft.transfer_matrix; atol = 1e-9 * T_norm)
    Test.@test maximum(abs, r_dir.transfer_matrix .+ r_dir.transfer_matrix') < 1e-10 * T_norm   # T_θ antisymmetric
    # Reduction: Σ_m T_θ(n,m) == scalar transfer spectrum T_θ(k).
    sfθ = FIT.calculate_scalar_flux(û, θ, ks; binning = b, dealiasing = FIT.Types.OrszagTwoThirds(),
        transform = SpectralBackends.FFTSpectralBackend())
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
        transform = SpectralBackends.FFTSpectralBackend())
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
        transform = SpectralBackends.FFTSpectralBackend())
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
    Test.@test FIT.ShellBinning.shell_coordinate(FIT.Types.IsotropicShells(), ks) ≈ FIT.Utils.wavenumber_magnitude_grid(ks)
    Test.@test FIT.ShellBinning.shell_coordinate(FIT.Types.PerpendicularShells(), ks) ≈ sqrt.(kx.^2 .+ ky.^2)
    Test.@test FIT.ShellBinning.shell_coordinate(FIT.Types.ParallelShells(), ks)      ≈ abs.(kz)

    # Divergence-free 3D velocity û = i k × Â
    Random.seed!(41)
    Â = randn(ComplexF64, M, M, M, 3)
    ûx = im .* (ky .* Â[:, :, :, 3] .- kz .* Â[:, :, :, 2])
    ûy = im .* (kz .* Â[:, :, :, 1] .- kx .* Â[:, :, :, 3])
    ûz = im .* (kx .* Â[:, :, :, 2] .- ky .* Â[:, :, :, 1])
    û3 = cat(ûx, ûy, ûz; dims = 4)
    b  = FIT.Types.LinearBinning(2π/L)

    r_iso  = FIT.SpectralFlux.calculate_spectral_flux(û3, ks; binning=b, dealiasing = FIT.Types.OrszagTwoThirds(),
        transform=SpectralBackends.FFTSpectralBackend(), geometry=FIT.Types.IsotropicShells())
    sT = sqrt(sum(abs2, r_iso.transfer_spectrum)); Test.@test sT > 0   # non-degenerate
    # Isotropic shells cover every non-DC mode, so the isotropic flux conserves: Σ_k T(|k|) ≈ 0.
    Test.@test abs(sum(r_iso.transfer_spectrum)) < 1e-9 * sum(abs, r_iso.transfer_spectrum)
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
    rtot = FIT.SpectralFlux.calculate_spectral_flux(û, ks; binning=b, dealiasing = FIT.Types.OrszagTwoThirds(), transform=SpectralBackends.FFTSpectralBackend())
    rhel = FIT.SpectralFlux.calculate_spectral_flux(û, ks; binning=b, dealiasing = FIT.Types.OrszagTwoThirds(), transform=SpectralBackends.FFTSpectralBackend(),
        decomposition=FIT.Types.HelicalDecomposition())
    Test.@test isapprox(rhel.positive.transfer_spectrum .+ rhel.negative.transfer_spectrum,
        rtot.transfer_spectrum; atol = 1e-9 * (sqrt(sum(abs2, rtot.transfer_spectrum)) + eps()))
end

# -----------------------------------------------------------------------
Test.@testset "Helmholtz partial fluxes — incompressible ⇒ rotational channel only" begin
    # For a divergence-free field the divergent component vanishes, so the (rotational, rotational,
    # rotational) channel is the full flux and every channel touching the divergent part is ≈ 0.
    N = 16; L = 2π
    Random.seed!(92)
    ks = FIT.Utils.wavenumber_grid((N, N), (L, L))
    kx = [ks[1][i] for i in 1:N, j in 1:N]; ky = [ks[2][j] for i in 1:N, j in 1:N]
    ψh = FFTW.fft(randn(N, N)) ./ N^2
    û  = cat(im .* ky .* ψh, -im .* kx .* ψh; dims = 3)        # div-free (rotational only)
    b  = FIT.Types.LinearBinning(2π/L)

    hp = FIT.calculate_partial_fluxes(û, ks; decomposition=FIT.Types.HelmholtzDecomposition(),
        binning=b, dealiasing=FIT.Types.OrszagTwoThirds(), transform=SpectralBackends.FFTSpectralBackend())
    sf = FIT.SpectralFlux.calculate_spectral_flux(û, ks; binning=b, dealiasing=FIT.Types.OrszagTwoThirds(), transform=SpectralBackends.FFTSpectralBackend())
    sT = sqrt(sum(abs2, sf.transfer_spectrum)); Test.@test sT > 0
    rrr = hp.channels[(:rotational, :rotational, :rotational)]
    Test.@test isapprox(rrr.transfer_spectrum, sf.transfer_spectrum; atol = 1e-9 * sT)
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
    # The grid derivative, zero at the Nyquist slot of an even axis.
    kx = [FTB.derivative_wavenumber(ks[1], i) for i in 1:M, j in 1:M, l in 1:M]
    ky = [FTB.derivative_wavenumber(ks[2], j) for i in 1:M, j in 1:M, l in 1:M]
    kz = [FTB.derivative_wavenumber(ks[3], l) for i in 1:M, j in 1:M, l in 1:M]
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
Test.@testset "BandToBand — smooth T(K,Q): antisymmetric, conserves" begin
    N = 16; L = 2π
    Random.seed!(71)
    ks = FIT.Utils.wavenumber_grid((N, N), (L, L))
    kx = [ks[1][i] for i in 1:N, j in 1:N]; ky = [ks[2][j] for i in 1:N, j in 1:N]
    ψh = FFTW.fft(randn(N, N)) ./ N^2
    û  = cat(im .* ky .* ψh, -im .* kx .* ψh; dims = 3)        # div-free
    centers = [1.0, 2.0, 3.0, 4.0]
    bands = FIT.Types.SmoothBands(centers; logwidth = 0.5)

    r = FIT.BandTransfer.calculate_band_to_band_transfer(û, ks; bands = bands, dealiasing = FIT.Types.OrszagTwoThirds(),
        transform = SpectralBackends.FFTSpectralBackend())
    T = r.transfer_matrix
    Tn = sqrt(sum(abs2, T)); Test.@test Tn > 0
    Test.@test maximum(abs, T .+ T') < 1e-10 * Tn                       # antisymmetric
    Test.@test abs(sum(T)) < 1e-10 * Tn                                 # conserves
end

# -----------------------------------------------------------------------
# Every grid the coarse-graining entry builds from its arguments gives CGEF's own `compute_Π!` on that grid.
Test.@testset "CoarseGrainingFlux — Π_ℓ == CGEF compute_Π! directly" begin
    CG = CoarseGrainingEnergyFluxes
    filt = FIT.Types.GaussianFilter()

    Test.@testset "3D Cartesian" begin
        FT = Float64; N = 12; L = 2π
        xs = range(0, L; length = N + 1)[1:N]; ys = copy(xs); zs = copy(xs)
        Random.seed!(29)
        u = randn(N, N, N); v = randn(N, N, N); w = randn(N, N, N); ℓ = L / 4
        r = FIT.CoarseGrainingFlux.calculate_coarse_graining_flux((u, v, w), (xs, ys, zs), ℓ, filt)
        geom = CG.FlowGeometries.Geometry.CartesianGeometry{FT}()
        # The axes go in as the ranges the entry received: broadcasting them to `FT` materializes a dense
        # vector, and a grid whose spacing is not in its axis type reaches a different CGEF engine.
        grid = CG.FlowGeometries.Grids.StructuredGrid(geom, xs, ys, zs, trues(N, N, N))
        Πref = zeros(FT, N, N, N)
        CG.Diagnostics.compute_Π!(Πref, u, v, w, grid, CG.Kernels.GaussianKernel(), FT(ℓ))
        Test.@test maximum(abs, r.flux_field .- Πref) == 0.0
    end

    Test.@testset "spherical lon–lat" begin
        FT = Float64; R = 6.371e6
        Nlon, Nlat = 24, 20
        lon = range(0, 2π; length = Nlon + 1)[1:Nlon]   # radians, the whole circle
        lat = range(-1.2, 1.2; length = Nlat)           # radians, away from the poles
        Random.seed!(37)
        u = randn(Nlon, Nlat); v = randn(Nlon, Nlat); ℓ = 5.0e5
        geom = CG.FlowGeometries.Geometry.SphericalGeometry(R)
        grid = CG.FlowGeometries.Grids.StructuredGrid(geom, lon, lat, trues(Nlon, Nlat))
        G = CG.Kernels.GaussianKernel()
        # The grid the entry builds from the coordinates is CGEF's, compared under the real-space filter of
        # the Cartesian components.
        cc = CG.Filtering.CartesianComponents()
        r = FIT.CoarseGrainingFlux.calculate_coarse_graining_flux((u, v), (lon, lat), ℓ, filt; radius = R,
                                                                  vector_filter = cc)
        Πref = zeros(FT, Nlon, Nlat)
        CG.Diagnostics.compute_Π!(Πref, u, v, nothing, grid, G, FT(ℓ); vector_filter = cc)
        Test.@test maximum(abs, Πref) > 0
        Test.@test maximum(abs, r.flux_field .- Πref) == 0.0
        # By default CGEF filters the velocity on a sphere by the generalized convolution, here through
        # NUFSHT's spin transforms.
        rg = FIT.CoarseGrainingFlux.calculate_coarse_graining_flux((u, v), (lon, lat), ℓ, filt; radius = R)
        Πg = zeros(FT, Nlon, Nlat)
        CG.Diagnostics.compute_Π!(Πg, u, v, nothing, grid, G, FT(ℓ))
        Test.@test maximum(abs, Πg) > 0
        Test.@test isapprox(rg.flux_field, Πg; atol = 1e-12 * maximum(abs, Πg))
        # Under `Deformable` CGEF resolves the Cartesian components, the filter that plan takes.
        df = CG.Filtering.Deformable()
        rd = FIT.CoarseGrainingFlux.calculate_coarse_graining_flux((u, v), (lon, lat), ℓ, filt; radius = R,
                                                                  mask_strategy = df)
        Πd = zeros(FT, Nlon, Nlat)
        CG.Diagnostics.compute_Π!(Πd, u, v, nothing, grid, G, FT(ℓ); mask_strategy = df)
        Test.@test maximum(abs, Πd) > 0
        Test.@test maximum(abs, rd.flux_field .- Πd) == 0.0
        # The mean flux is area-weighted: a cell spans R²Δλ(sin φ₊ − sin φ₋) = 2R²Δλ cos φ sin(Δφ/2).
        Δλ = step(lon); Δφ = step(lat)
        A = [2R^2 * Δλ * cos(φ) * sin(Δφ / 2) for λ in lon, φ in lat]
        Test.@test r.mean_flux ≈ sum(Πref .* A) / sum(A) rtol = 1e-12
        # A lon–lat sphere is a 2D surface.
        Test.@test_throws ArgumentError FIT.CoarseGrainingFlux.calculate_coarse_graining_flux(
            (u, v, u), (lon, lat, lat), ℓ, filt; radius = R)
    end

    # A vertical component on the two-dimensional grid is CGEF's thin-layer flux, and on a masked grid the
    # flux is CGEF's under its own default mask strategy.
    Test.@testset "thin-layer vertical component and mask" begin
        Random.seed!(7)
        N = 32; L = 2π
        xs = range(0, L; length = N + 1)[1:N]; ys = copy(xs)
        u  = [sin(xs[i]) * cos(ys[j]) + 0.2 * randn() for i in 1:N, j in 1:N]
        v  = [-cos(xs[i]) * sin(ys[j]) + 0.2 * randn() for i in 1:N, j in 1:N]
        ℓ = 0.5
        cart = CG.FlowGeometries.Geometry.CartesianGeometry{Float64}()
        w = [0.3 * cos(xs[i] + ys[j]) for i in 1:N, j in 1:N]
        ws3 = FIT.CoarseGrainingFlux.CoarseGrainingFluxWorkspace((u, v, w), (xs, ys), ℓ, filt)
        r3 = FIT.CoarseGrainingFlux.calculate_coarse_graining_flux!(ws3, (u, v, w))
        Π3 = zeros(N, N)
        CG.Diagnostics.compute_Π!(Π3, u, v, w, CG.FlowGeometries.Grids.StructuredGrid(cart, xs, ys),
            CG.Kernels.GaussianKernel(), ℓ)
        Test.@test maximum(abs, r3.flux_field .- Π3) == 0.0
        r2 = FIT.CoarseGrainingFlux.calculate_coarse_graining_flux((u, v), (xs, ys), ℓ, filt)
        Test.@test maximum(abs, r3.flux_field .- r2.flux_field) > 0
        mask = trues(N, N); mask[5:9, 12:20] .= false
        rm = FIT.CoarseGrainingFlux.calculate_coarse_graining_flux((u, v), (xs, ys), ℓ, filt; mask)
        Πm = zeros(N, N)
        CG.Diagnostics.compute_Π!(Πm, u, v, nothing, CG.FlowGeometries.Grids.StructuredGrid(cart, xs, ys; mask),
            CG.Kernels.GaussianKernel(), ℓ)
        Test.@test isequal(rm.flux_field, Πm)
    end
end

# -----------------------------------------------------------------------
# Function barrier: `@allocated` at a testset's top level measures its own scope, so the call being
# measured is made through a function with concretely-typed arguments.
_alloc_from_spectral(ws, û) = (FIT.from_spectral!(ws, û); @allocated FIT.from_spectral!(ws, û))

Test.@testset "to_spectral — physical-space entry (uniform Cartesian grid)" begin
    N = 16; L = 2π
    x = range(0.0, L; length = N + 1)[1:N]
    y = range(0.0, L; length = N + 1)[1:N]
    Random.seed!(11)
    u = randn(N, N); v = randn(N, N)

    # Real input takes the half (r2c) layout by default: the non-redundant `k₁ ≥ 0` coefficients and
    # an `rfftfreq` first axis, reproducing the canonical û = rfft(u)/Nᵈ exactly.
    û_ts, ks_ts = FIT.to_spectral((u, v), (x, y); transform = SpectralBackends.FFTSpectralBackend())
    û_half = cat(FFTW.rfft(u), FFTW.rfft(v); dims = 3) ./ N^2
    ks_half = FIT.Utils.wavenumber_grid((N, N), (L, L); real = true)
    Test.@test isapprox(û_ts, û_half; atol = 1e-12)
    Test.@test all(isapprox.(ks_ts, ks_half; atol = 1e-12))

    # `real_layout = false` keeps the full complex spectrum, û = fft(u)/Nᵈ on the fftfreq grid.
    û_full, ks_full = FIT.to_spectral((u, v), (x, y); transform = SpectralBackends.FFTSpectralBackend(),
                                      real_layout = false)
    û_man = cat(FFTW.fft(u), FFTW.fft(v); dims = 3) ./ N^2
    ks_man = FIT.Utils.wavenumber_grid((N, N), (L, L))
    Test.@test isapprox(û_full, û_man; atol = 1e-12)
    Test.@test all(isapprox.(ks_full, ks_man; atol = 1e-12))

    # DirectSum reference agrees with FFT, on both layouts.
    û_ds, _ = FIT.to_spectral((u, v), (x, y); transform = SpectralBackends.DirectSumSpectralBackend(),
                              real_layout = false)
    Test.@test isapprox(û_ds, û_man; atol = 1e-10)
    û_dsh, _ = FIT.to_spectral((u, v), (x, y); transform = SpectralBackends.DirectSumSpectralBackend())
    Test.@test isapprox(û_dsh, û_half; atol = 1e-10)

    # A single-component tuple (a scalar field).
    ρ = randn(N, N)
    ρ̂, _ = FIT.to_spectral((ρ,), (x, y); transform = SpectralBackends.FFTSpectralBackend())
    Test.@test isapprox(ρ̂[:, :, 1], FFTW.rfft(ρ) ./ N^2; atol = 1e-12)

    # Complex input has no Hermitian symmetry and keeps the full grid.
    ûc, ksc = FIT.to_spectral((ComplexF64.(u), ComplexF64.(v)), (x, y); transform = SpectralBackends.FFTSpectralBackend())
    Test.@test isapprox(ûc, û_man; atol = 1e-12)

    # `from_spectral` inverts it on either layout.
    for rl in (true, false)
        ûr, ksr = FIT.to_spectral((u, v), (x, y); real_layout = rl)
        back = FIT.from_spectral(ûr, ksr)
        Test.@test maximum(abs, back[1] .- u) < 1e-12 * maximum(abs, u)
        Test.@test maximum(abs, back[2] .- v) < 1e-12 * maximum(abs, v)
    end
    # The `!` form returns the stacked `(ns…, D)` array.
    û_rt, ks_rt = FIT.to_spectral((u, v), (x, y))
    ws_rt = FIT.ToSpectralWorkspace(û_rt, ks_rt)
    phys_rt = FIT.from_spectral!(ws_rt, û_rt)
    Test.@test maximum(abs, phys_rt[:, :, 1] .- u) < 1e-12 * maximum(abs, u)
    Test.@test _alloc_from_spectral(ws_rt, û_rt) == 0

    # Scattered and spherical transforms do not take uniform-grid coordinates.
    Test.@test_throws ArgumentError FIT.to_spectral((u, v), (x, y); transform = FTB.FINUFFTBackend())
    Test.@test_throws ArgumentError FIT.to_spectral((u, v), (x, y); transform = SpectralBackends.FSHTSpectralBackend())
end

# -----------------------------------------------------------------------
Test.@testset "TriadicOrthogonalDecomposition" begin
    Test.@test_throws ArgumentError FIT.TriadicOrthogonalDecomposition.parse_parameters(100, 5; window=3)      # nDFT < 4
    Test.@test_throws ArgumentError FIT.TriadicOrthogonalDecomposition.parse_parameters(100, 5; noverlap=256)  # noverlap >= nDFT

    # A workspace takes snapshots of the shape it was built for.
    t = collect(0:255) .* 0.1
    X = repeat(sin.(2π * 2.0 .* t) .+ cos.(2π * 3.0 .* t), 1, 1, 2)
    ws = FIT.TriadicOrthogonalDecomposition.TODWorkspace(X; dt=0.1, transform=SpectralBackends.FFTSpectralBackend())
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
end

# -----------------------------------------------------------------------
Test.@testset "TOD — detects a known quadratic triad (FFT == direct)" begin
    # A phase-locked triad f_k + f_l = f_n (daughter phase φ_k + φ_l, random parent phases per block):
    # the two backends agree, the triad is a dominant peak, and it scores far above the same triad with
    # an unlocked daughter.
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
        isreal_data=true, transform=SpectralBackends.DirectSumSpectralBackend())
    rf = FIT.TriadicOrthogonalDecomposition.triadic_orthogonal_decomposition(Xl; window=nfft, noverlap=0, nmode=1, dt=dt,
        isreal_data=true, transform=SpectralBackends.FFTSpectralBackend())
    f = rd.frequencies
    Ld = copy(rd.mode_bispectrum[:, :, 1]); Ld[isnan.(Ld)] .= 0
    Lf = copy(rf.mode_bispectrum[:, :, 1]); Lf[isnan.(Lf)] .= 0
    li = argmin(abs.(f .- fl)); ni = argmin(abs.(f .- fn))
    Test.@test isapprox(filter(!isnan, rd.mode_bispectrum), filter(!isnan, rf.mode_bispectrum); atol = 1e-10)
    # The triad (f_l = 2, f_n = 3) is a dominant peak.
    Test.@test Lf[li, ni] >= 0.5 * maximum(Lf)
    # An unlocked daughter scores far lower at the same cell.
    Xu = build_signal(false)
    ru = FIT.TriadicOrthogonalDecomposition.triadic_orthogonal_decomposition(Xu; window=nfft, noverlap=0, nmode=1, dt=dt,
        isreal_data=true, transform=SpectralBackends.FFTSpectralBackend())
    Lu = copy(ru.mode_bispectrum[:, :, 1]); Lu[isnan.(Lu)] .= 0
    Test.@test Lu[li, ni] < 0.3 * Lf[li, ni]
end

# -----------------------------------------------------------------------
Test.@testset "TOD — the mode-summed budget of the advective term is pairwise conservative" begin
    # With Q(q₁, q₂) = −(q₁·∇)q₂ the budget summed over every mode is the block average of
    # 𝒯_{l→n} = −Re⟨q̂_n, (q̂_{n−l}·∇)q̂_l⟩, and 𝒯_{l→n} + 𝒯_{n→l} = −Re⟨(q̂_{n−l}·∇)(q̄̂_n·q̂_l)⟩
    # vanishes for a divergence-free periodic field (Yeung, Chu & Schmidt 2026, Eq. 38). The field is
    # band-limited to |m_d| ≤ 1 on an 8-point grid, so every product is resolved and the identity holds
    # on the grid sum.
    Nx = 8; nt = 128; nfft = 16
    xs = [2π * (i - 1) / Nx for i in 1:Nx]
    kf = [i - 1 <= (Nx - 1) ÷ 2 ? i - 1 : i - 1 - Nx for i in 1:Nx]
    kx = [kf[i] for i in 1:Nx, j in 1:Nx]; ky = [kf[j] for i in 1:Nx, j in 1:Nx]
    rng = Random.MersenneTwister(8)
    X = zeros(nt, 2, Nx, Nx)
    for t in 1:nt, (m1, m2) in ((1, 0), (0, 1), (1, 1), (1, -1))
        A = randn(rng) + im * randn(rng)
        e = [cis(m1 * x + m2 * y) for x in xs, y in xs]
        X[t, 1, :, :] .+= 2 .* real.(im * m2 * A .* e)      # u = ∂ψ/∂y
        X[t, 2, :, :] .-= 2 .* real.(im * m1 * A .* e)      # v = −∂ψ/∂x
    end
    function advective(q1, q2)
        out = zeros(ComplexF64, Nx * Nx, 2, size(q1, 3))
        for b in axes(q1, 3)
            a1 = reshape(q1[1, :, b], Nx, Nx); a2 = reshape(q1[2, :, b], Nx, Nx)
            for c in 1:2
                f̂ = FFTW.fft(reshape(q2[c, :, b], Nx, Nx))
                fx = FFTW.ifft(im .* kx .* f̂); fy = FFTW.ifft(im .* ky .* f̂)
                out[:, c, b] .= vec(.-(a1 .* fx .+ a2 .* fy))
            end
        end
        return out
    end
    r = FIT.TriadicOrthogonalDecomposition.triadic_orthogonal_decomposition(X; window = nfft, noverlap = 0,
            Q = advective, transform = SpectralBackends.FFTSpectralBackend())
    f = r.frequencies                                        # integers: dt defaults to 1/nfft
    at(fr) = findfirst(==(fr), f)
    summed(l, n) = sum(filter(!isnan, r.modal_energy_budget[at(l), at(n), :]))
    pairs = [(summed(l, n), summed(n, l)) for l in 0:3 for n in 0:3 if l < n]
    scale = maximum(p -> max(abs(p[1]), abs(p[2])), pairs)
    Test.@test scale > 1e-8
    Test.@test maximum(p -> abs(p[1] + p[2]), pairs) < 1e-10 * scale
end

# -----------------------------------------------------------------------
Test.@testset "Field Decomposition (Helmholtz / Partial Flux)" begin
    # 1. Spectral flux decomposition test: a random real field with both a rotational and a divergent
    # part, so every channel carries transfer.
    N = 8; L = 2π
    ks = FIT.Utils.wavenumber_grid((N, N), (L, L))
    rngd = Random.MersenneTwister(21)
    û = cat((FFTW.fft(randn(rngd, N, N)) ./ N^2 for _ in 1:2)...; dims = 3)

    res_none = FIT.SpectralFlux.calculate_spectral_flux(û, ks; decomposition=FIT.Types.NoDecomposition(), dealiasing = FIT.Types.NoDealiasing())
    res_helm = FIT.SpectralFlux.calculate_spectral_flux(û, ks; decomposition=FIT.Types.HelmholtzDecomposition(), dealiasing = FIT.Types.NoDealiasing())
    res_rot  = FIT.SpectralFlux.calculate_spectral_flux(û, ks; decomposition=FIT.Types.RotationalDecomposition(), dealiasing = FIT.Types.NoDealiasing())
    res_div  = FIT.SpectralFlux.calculate_spectral_flux(û, ks; decomposition=FIT.Types.DivergentDecomposition(), dealiasing = FIT.Types.NoDealiasing())

    Test.@test minimum(maximum(abs, r.transfer_spectrum) for r in (res_rot, res_div)) > 1e-8

    # The receiver split is linear, so the channels sum to the undecomposed transfer.
    Test.@test isapprox(res_none.transfer_spectrum, res_rot.transfer_spectrum + res_div.transfer_spectrum;
                        atol = 1e-12 * maximum(abs, res_none.transfer_spectrum))
    Test.@test res_helm.rotational.transfer_spectrum ≈ res_rot.transfer_spectrum

    # 2. A decomposition solves on the domain, so the coarse-graining flux takes it on a grid; coordinate
    # vectors carry no topology to solve on.
    x = range(0, L; length=N+1)[1:N]
    y = range(0, L; length=N+1)[1:N]
    u = [cos(x) for x in x, y in y]
    v = [sin(y) for x in x, y in y]
    Test.@test_throws ArgumentError FIT.CoarseGrainingFlux.calculate_coarse_graining_flux(
        (u, v), (x, y), 1.0, FIT.Types.GaussianFilter(); decomposition=FIT.Types.HelmholtzDecomposition())

    # 3. The physical-space Helmholtz split on a periodic 3D grid reconstructs the field.
    cart = FG.Geometry.CartesianGeometry{Float64}()
    M = 8; x3 = range(0, L; length=M+1)[1:M]; y3 = copy(x3); z3 = copy(x3)
    u3 = [cos(xi) * sin(zi) for xi in x3, yi in y3, zi in z3]
    v3 = [sin(yi)           for xi in x3, yi in y3, zi in z3]
    w3 = [cos(zi)           for xi in x3, yi in y3, zi in z3]
    grid3 = FG.Grids.StructuredGrid(cart, x3, y3, z3; topology = (true, true, true))
    hd3 = FIT.Decomposition.decompose_field(FIT.Types.HelmholtzDecomposition(), (u3, v3, w3), grid3)
    rot3, div3 = hd3.rotational, hd3.divergent
    for (c, f) in enumerate((u3, v3, w3))
        Test.@test maximum(abs, rot3[c] .+ div3[c] .- f) < 1e-10 * maximum(abs, f)
    end
    # The divergent part of this field is nonzero, so the split is doing work.
    Test.@test maximum(abs, div3[1]) > 1e-8
    Test.@test hd3.converged
    # A conjugate-gradient solve stopped after one iteration reports it: the divergence of this field
    # spans two eigenvalues of the grid Laplacian, which one iteration cannot resolve.
    short = FIT.Decomposition.decompose_field(FIT.Types.HelmholtzDecomposition(), (u3, v3, w3), grid3;
                solver = HelmholtzDecomposition.CGSolver(; max_iter = 1, multigrid = false))
    Test.@test !short.converged
    Test.@test short.iterations <= 1
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
    # The transform runs where the mediator columns are computed, so the workers need it too: the
    # entries resolve `AutoSpectralBackend` on the master and hand the workers a concrete backend.
    Distributed.@everywhere using FFTW

    # A random real field: every diagnostic below carries transfer, so parity compares signal.
    N = 8; L = 2π
    ks = FIT.Utils.wavenumber_grid((N, N), (L, L))
    rngp = Random.MersenneTwister(42)
    û = cat((FFTW.fft(randn(rngp, N, N)) ./ N^2 for _ in 1:2)...; dims = 3)

    # 1. Shell-to-Shell Transfer Parity
    b = FIT.Types.LinearBinning(2π / L)
    res_serial = FIT.ShellToShellTransfer.calculate_shell_to_shell_transfer(û, ks; binning=b, dealiasing = FIT.Types.OrszagTwoThirds(), execution=ComputationalBackends.SerialBackend())
    Test.@test maximum(abs, res_serial.transfer_matrix) > 1e-8
    res_thread = FIT.ShellToShellTransfer.calculate_shell_to_shell_transfer(û, ks; binning=b, dealiasing = FIT.Types.OrszagTwoThirds(), execution=ComputationalBackends.ThreadedBackend())
    
    # For DistributedBackend, we convert velocity_hat to a SharedArray so workers can read it efficiently
    s_û = SharedArrays.SharedArray(û)
    res_dist = FIT.ShellToShellTransfer.calculate_shell_to_shell_transfer(s_û, ks; binning=b, dealiasing = FIT.Types.OrszagTwoThirds(), execution=ComputationalBackends.DistributedBackend())

    Test.@test isapprox(res_serial.transfer_matrix, res_thread.transfer_matrix; atol=1e-12)
    Test.@test isapprox(res_serial.transfer_matrix, res_dist.transfer_matrix; atol=1e-12)
    Test.@test isapprox(res_serial.net_transfer, res_thread.net_transfer; atol=1e-12)
    Test.@test isapprox(res_serial.net_transfer, res_dist.net_transfer; atol=1e-12)

    # The hybrid DistributedBackend(ThreadedBackend()) runs each worker's mediators on that worker's
    # threads. The workers here have one thread each, so the worker function is also run on this
    # process, whose threads split the mediators: every column is the spectral-flux reduction's.
    res_hybrid = FIT.ShellToShellTransfer.calculate_shell_to_shell_transfer(s_û, ks; binning=b, dealiasing = FIT.Types.OrszagTwoThirds(), execution=ComputationalBackends.DistributedBackend(ComputationalBackends.ThreadedBackend()))
    Test.@test isapprox(res_serial.transfer_matrix, res_hybrid.transfer_matrix; atol=1e-12)
    Test.@test isapprox(res_serial.net_transfer, res_hybrid.net_transfer; atol=1e-12)
    let ext = Base.get_extension(FIT, :FlowInvariantTransferDistributedExt)
        ws_s = FIT.Workspaces.ShellToShellWorkspace(û, ks, b)
        N_sh = size(res_serial.transfer_matrix, 1)
        for inner in (ComputationalBackends.SerialBackend(), ComputationalBackends.ThreadedBackend())
            cols = ext._shell_to_shell_worker_columns(1:N_sh, û, ks, ws_s.shell_idx, N_sh, FIT.Types.KineticEnergy(),
                FIT.Types.OrszagTwoThirds(), Float64, SpectralBackends.FFTSpectralBackend(), û, inner)
            Tm = reduce(hcat, (c[2] for c in cols))
            sf = FIT.SpectralFlux.calculate_spectral_flux(û, ks; binning = b, transform = SpectralBackends.FFTSpectralBackend())
            Test.@test maximum(abs, vec(sum(Tm; dims = 2)) .- sf.transfer_spectrum) < 1e-12 * maximum(abs, Tm)
            Test.@test maximum(abs, Tm .- res_serial.transfer_matrix) < 1e-12 * maximum(abs, Tm)
        end
    end

    # 2. TOD with the triads distributed over workers is bit-identical to serial: each worker runs the
    # serial loop's SVD, and the temporal DFT stays on the master.
    Random.seed!(123)
    Xtod = randn(64, 1, 4)
    tod_serial = FIT.TriadicOrthogonalDecomposition.triadic_orthogonal_decomposition(Xtod; dt = 0.05, transform = SpectralBackends.DirectSumSpectralBackend(), execution = ComputationalBackends.SerialBackend(), return_coefficients = true, return_auxiliary_modes = true)
    tod_dist   = FIT.TriadicOrthogonalDecomposition.triadic_orthogonal_decomposition(Xtod; dt = 0.05, transform = SpectralBackends.DirectSumSpectralBackend(), execution = ComputationalBackends.DistributedBackend(), return_coefficients = true, return_auxiliary_modes = true)
    Test.@test isequal(tod_serial.mode_bispectrum, tod_dist.mode_bispectrum)
    Test.@test isequal(tod_serial.modal_energy_budget, tod_dist.modal_energy_budget)
    let k = first(keys(tod_serial.modes))
        Test.@test tod_serial.modes[k].convective == tod_dist.modes[k].convective
        Test.@test tod_serial.expansion_coefficients[k].recipient == tod_dist.expansion_coefficients[k].recipient
        Test.@test tod_serial.auxiliary_modes[k].donor == tod_dist.auxiliary_modes[k].donor
    end

    # 3. Mode-to-mode, band-to-band, partial fluxes and compressible: distributed and hybrid match serial.
    spD = SpectralBackends.DirectSumSpectralBackend()
    m_ser = FIT.calculate_mode_to_mode_transfer(û, ks; transform=spD, execution=ComputationalBackends.SerialBackend())
    m_dst = FIT.calculate_mode_to_mode_transfer(û, ks; transform=spD, execution=ComputationalBackends.DistributedBackend())
    m_hyb = FIT.calculate_mode_to_mode_transfer(û, ks; transform=spD, execution=ComputationalBackends.DistributedBackend(ComputationalBackends.ThreadedBackend()))
    Test.@test maximum(abs, m_ser.transfer) > 1e-8
    Test.@test isapprox(m_ser.transfer, m_dst.transfer; atol=1e-12)
    Test.@test isapprox(m_ser.net_transfer, m_dst.net_transfer; atol=1e-12)
    Test.@test isapprox(m_ser.transfer, m_hyb.transfer; atol=1e-12)

    bands = FIT.Types.SmoothBands([2.0, 4.0, 6.0])
    bb_ser = FIT.BandTransfer.calculate_band_to_band_transfer(û, ks; bands=bands, transform=spD, execution=ComputationalBackends.SerialBackend())
    bb_dst = FIT.BandTransfer.calculate_band_to_band_transfer(û, ks; bands=bands, transform=spD, execution=ComputationalBackends.DistributedBackend())
    bb_hyb = FIT.BandTransfer.calculate_band_to_band_transfer(û, ks; bands=bands, transform=spD, execution=ComputationalBackends.DistributedBackend(ComputationalBackends.ThreadedBackend()))
    Test.@test maximum(abs, bb_ser.transfer_matrix) > 1e-8
    Test.@test isapprox(bb_ser.transfer_matrix, bb_dst.transfer_matrix; atol=1e-12)
    Test.@test isapprox(bb_ser.transfer_matrix, bb_hyb.transfer_matrix; atol=1e-12)

    pf_ser = FIT.calculate_partial_fluxes(û, ks; decomposition=FIT.Types.HelmholtzDecomposition(), transform=spD, execution=ComputationalBackends.SerialBackend())
    pf_dst = FIT.calculate_partial_fluxes(û, ks; decomposition=FIT.Types.HelmholtzDecomposition(), transform=spD, execution=ComputationalBackends.DistributedBackend())
    Test.@test maximum(abs, pf_ser.total.transfer_spectrum) > 1e-8
    Test.@test isapprox(pf_ser.total.transfer_spectrum, pf_dst.total.transfer_spectrum; atol=1e-12)
    for k in keys(pf_ser.channels)
        Test.@test isapprox(pf_ser.channels[k].transfer_spectrum, pf_dst.channels[k].transfer_spectrum; atol = 1e-12)
    end

    ρ̂c = randn(Random.MersenneTwister(7), ComplexF64, N, N)
    p̂c = randn(Random.MersenneTwister(8), ComplexF64, N, N)
    c_ser = FIT.Compressible.calculate_compressible_flux(û, ρ̂c, ks; transform=spD, execution=ComputationalBackends.SerialBackend(), pressure_hat=p̂c, decompose=true)
    c_dst = FIT.Compressible.calculate_compressible_flux(û, ρ̂c, ks; transform=spD, execution=ComputationalBackends.DistributedBackend(), pressure_hat=p̂c, decompose=true)
    c_hyb = FIT.Compressible.calculate_compressible_flux(û, ρ̂c, ks; transform=spD, execution=ComputationalBackends.DistributedBackend(ComputationalBackends.ThreadedBackend()), pressure_hat=p̂c, decompose=true)
    Test.@test maximum(abs, c_ser.transfer_spectrum) > 1e-8
    Test.@test isapprox(c_ser.transfer_spectrum, c_dst.transfer_spectrum; atol=1e-12)
    Test.@test isapprox(c_ser.flux, c_dst.flux; atol=1e-12)
    Test.@test isapprox(c_ser.channels.rotational, c_dst.channels.rotational; atol=1e-12)
    Test.@test isapprox(c_ser.channels.compressive, c_dst.channels.compressive; atol=1e-12)
    Test.@test isapprox(c_ser.pressure_dilatation.compressive, c_dst.pressure_dilatation.compressive; atol=1e-12)
    Test.@test isapprox(c_ser.transfer_spectrum, c_hyb.transfer_spectrum; atol=1e-12)
end

# -----------------------------------------------------------------------
# The *_batch entries reuse one workspace per worker across snapshots that share a grid. The serial batch
# is bit-identical to the per-snapshot call, in input order; Threaded / Distributed / GPU match the serial
# batch; an empty batch returns empty.
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
    FFT = SpectralBackends.FFTSpectralBackend()
    Ser = ComputationalBackends.SerialBackend()
    Thr = ComputationalBackends.ThreadedBackend()
    Dst = ComputationalBackends.DistributedBackend()
    GPU = ComputationalBackends.GPUBackend(KA.CPU())               # the device path on host arrays
    vhats = map(1:nsnap) do _
        ψh = FFTW.fft(randn(N, N)) ./ N^2
        cat(im .* ky .* ψh, -im .* kx .* ψh; dims = 3)            # divergence-free snapshot
    end
    xs = range(0, L; length = N + 1)[1:N]; ys = copy(xs)
    ufields = map(_ -> (randn(N, N), randn(N, N)), 1:nsnap)        # physical fields for coarse-graining
    filt = FIT.Types.GaussianFilter(); ℓ = 0.5

    # --- spectral flux ---
    sf1 = [FIT.SpectralFlux.calculate_spectral_flux(vh, ks; binning = b, transform = FFT) for vh in vhats]
    sfS = FIT.SpectralFlux.calculate_spectral_flux_batch(vhats, ks; binning = b, transform = FFT, execution = Ser)
    sfT = FIT.SpectralFlux.calculate_spectral_flux_batch(vhats, ks; binning = b, transform = FFT, execution = Thr)
    sfD = FIT.SpectralFlux.calculate_spectral_flux_batch(vhats, ks; binning = b, transform = FFT, execution = Dst)
    sfG = FIT.SpectralFlux.calculate_spectral_flux_batch(vhats, ks; binning = b, transform = FFT, execution = GPU)
    Test.@test length(sfS) == nsnap
    Test.@test maximum(abs, sf1[1].transfer_spectrum) > 0                       # genuine nonzero transfer
    for i in 1:nsnap
        Test.@test sfS[i].transfer_spectrum == sf1[i].transfer_spectrum         # serial batch == single, in order
        Test.@test sfS[i].flux == sf1[i].flux
        Test.@test isapprox(sfT[i].transfer_spectrum, sfS[i].transfer_spectrum; atol = 1e-12)
        Test.@test isapprox(sfD[i].transfer_spectrum, sfS[i].transfer_spectrum; atol = 1e-12)
        Test.@test isapprox(sfG[i].transfer_spectrum, sfS[i].transfer_spectrum; atol = 1e-12)   # GPU device path (KA.CPU)
    end
    Test.@test isempty(FIT.SpectralFlux.calculate_spectral_flux_batch(typeof(vhats[1])[], ks; binning = b, transform = FFT))

    # --- shell to shell ---
    ss1 = [FIT.ShellToShellTransfer.calculate_shell_to_shell_transfer(vh, ks; binning = b, transform = FFT) for vh in vhats]
    ssS = FIT.ShellToShellTransfer.calculate_shell_to_shell_transfer_batch(vhats, ks; binning = b, transform = FFT, execution = Ser)
    ssT = FIT.ShellToShellTransfer.calculate_shell_to_shell_transfer_batch(vhats, ks; binning = b, transform = FFT, execution = Thr)
    ssD = FIT.ShellToShellTransfer.calculate_shell_to_shell_transfer_batch(vhats, ks; binning = b, transform = FFT, execution = Dst)
    ssG = FIT.ShellToShellTransfer.calculate_shell_to_shell_transfer_batch(vhats, ks; binning = b, transform = FFT, execution = GPU)
    for i in 1:nsnap
        Test.@test ssS[i].transfer_matrix == ss1[i].transfer_matrix
        Test.@test ssS[i].net_transfer == ss1[i].net_transfer
        Test.@test isapprox(ssT[i].transfer_matrix, ssS[i].transfer_matrix; atol = 1e-12)
        Test.@test isapprox(ssD[i].transfer_matrix, ssS[i].transfer_matrix; atol = 1e-12)
        Test.@test isapprox(ssG[i].transfer_matrix, ssS[i].transfer_matrix; atol = 1e-12)   # GPU device path (KA.CPU)
    end

    # --- band to band ---
    bb1 = [FIT.BandTransfer.calculate_band_to_band_transfer(vh, ks; bands = bands, transform = FFT) for vh in vhats]
    bbS = FIT.BandTransfer.calculate_band_to_band_transfer_batch(vhats, ks; bands = bands, transform = FFT, execution = Ser)
    bbT = FIT.BandTransfer.calculate_band_to_band_transfer_batch(vhats, ks; bands = bands, transform = FFT, execution = Thr)
    bbD = FIT.BandTransfer.calculate_band_to_band_transfer_batch(vhats, ks; bands = bands, transform = FFT, execution = Dst)
    bbG = FIT.BandTransfer.calculate_band_to_band_transfer_batch(vhats, ks; bands = bands, transform = FFT, execution = GPU)
    for i in 1:nsnap
        Test.@test bbS[i].transfer_matrix == bb1[i].transfer_matrix
        Test.@test bbS[i].net_transfer == bb1[i].net_transfer
        Test.@test isapprox(bbT[i].transfer_matrix, bbS[i].transfer_matrix; atol = 1e-12)
        Test.@test isapprox(bbD[i].transfer_matrix, bbS[i].transfer_matrix; atol = 1e-12)
        Test.@test isapprox(bbG[i].transfer_matrix, bbS[i].transfer_matrix; atol = 1e-12)   # GPU device path (KA.CPU)
    end

    # --- coarse-graining (physical fields via CGEF) ---
    cg1 = [FIT.CoarseGrainingFlux.calculate_coarse_graining_flux(uv, (xs, ys), ℓ, filt; execution = Ser) for uv in ufields]
    cgS = FIT.CoarseGrainingFlux.calculate_coarse_graining_flux_batch(ufields, (xs, ys), ℓ, filt; execution = Ser)
    cgT = FIT.CoarseGrainingFlux.calculate_coarse_graining_flux_batch(ufields, (xs, ys), ℓ, filt; execution = Thr)
    cgD = FIT.CoarseGrainingFlux.calculate_coarse_graining_flux_batch(ufields, (xs, ys), ℓ, filt; execution = Dst)
    cgG = FIT.CoarseGrainingFlux.calculate_coarse_graining_flux_batch(ufields, (xs, ys), ℓ, filt; execution = GPU)
    for i in 1:nsnap
        Test.@test cgS[i].flux_field == cg1[i].flux_field
        Test.@test cgS[i].mean_flux == cg1[i].mean_flux
        Test.@test isapprox(cgT[i].flux_field, cgS[i].flux_field; atol = 1e-12)
        Test.@test isapprox(cgD[i].flux_field, cgS[i].flux_field; atol = 1e-12)
        Test.@test isapprox(cgG[i].flux_field, cgS[i].flux_field; atol = 1e-12)
    end

    # Host fields under the device execution: the flux with its diagnostics, the enstrophy flux and the
    # band energies, each against serial. The scales span a few grid cells.
    uv = ufields[1]; ℓg = 1.5
    rS = FIT.CoarseGrainingFlux.calculate_coarse_graining_flux(uv, (xs, ys), ℓg, filt; execution = Ser,
                                                               return_diagnostics = true)
    rG = FIT.CoarseGrainingFlux.calculate_coarse_graining_flux(uv, (xs, ys), ℓg, filt; execution = GPU,
                                                               return_diagnostics = true)
    sΠ = maximum(abs, rS.flux_field)
    Test.@test sΠ > 1e-8
    Test.@test isapprox(rG.flux_field, rS.flux_field; atol = 1e-12 * sΠ)
    Test.@test isapprox(rG.mean_flux, rS.mean_flux; atol = 1e-12 * sΠ)
    Test.@test isapprox(rG.stress_tensor, rS.stress_tensor; atol = 1e-12 * maximum(abs, rS.stress_tensor))
    Test.@test isapprox(rG.strain_rate, rS.strain_rate; atol = 1e-12 * maximum(abs, rS.strain_rate))

    zS = FIT.CoarseGrainingFlux.calculate_enstrophy_flux(uv, (xs, ys), ℓg, filt; execution = Ser)
    zG = FIT.CoarseGrainingFlux.calculate_enstrophy_flux(uv, (xs, ys), ℓg, filt; execution = GPU)
    Test.@test maximum(abs, zS) > 1e-8
    Test.@test isapprox(zG, zS; atol = 1e-12 * maximum(abs, zS))

    scales = [1.0, 2.0]
    bS = FIT.CoarseGrainingFlux.calculate_band_energies(uv, (xs, ys), filt, scales; execution = Ser, maps = true)
    bG = FIT.CoarseGrainingFlux.calculate_band_energies(uv, (xs, ys), filt, scales; execution = GPU, maps = true)
    Test.@test minimum(bS.bands) > 1e-8
    Test.@test isapprox(bG.bands, bS.bands; rtol = 1e-12)
    Test.@test all(isapprox(bG.band_maps[n], bS.band_maps[n]; atol = 1e-12 * maximum(abs, bS.band_maps[n]))
                   for n in eachindex(scales))
end

# -----------------------------------------------------------------------
Test.@testset "ModeToMode — invariant/dimension guards" begin
    L = 2π
    # Helicity is a 3D invariant.
    ks2 = FIT.Utils.wavenumber_grid((4, 4), (L, L))
    û2  = zeros(ComplexF64, 4, 4, 2); û2[2, 1, 1] = 0.5; û2[1, 2, 2] = 0.5
    Test.@test_throws ArgumentError FIT.calculate_mode_to_mode_transfer(û2, ks2; invariant=FIT.Types.Helicity())
end

# -----------------------------------------------------------------------
Test.@testset "ModeToMode — resolved S(k|p): antisym, conserves, reduces to spectral/shell-to-shell" begin
    # The resolved S(k|p) is antisymmetric, conserves, and reduces to the shell-binned diagnostics.
    N = 12; L = 2π
    Random.seed!(21)
    ψ  = randn(N, N); ψh = FFTW.fft(ψ) ./ N^2
    ks = FIT.Utils.wavenumber_grid((N, N), (L, L))
    kx = [ks[1][i] for i in 1:N, j in 1:N]; ky = [ks[2][j] for i in 1:N, j in 1:N]
    û  = cat(im .* ky .* ψh, -im .* kx .* ψh; dims = 3)
    m2m = FIT.calculate_mode_to_mode_transfer(û, ks; dealiasing = FIT.Types.OrszagTwoThirds(), transform = SpectralBackends.FFTSpectralBackend())
    S   = m2m.transfer                              # shape (N,N,N,N): S[k..., p...]
    nrm = sqrt(sum(abs2, S)); Test.@test nrm > 0    # non-degenerate
    asym = 0.0
    for k in CartesianIndices((N, N)), p in CartesianIndices((N, N))
        asym = max(asym, abs(S[k, p] + S[p, k]))
    end
    Test.@test asym < 1e-10 * nrm                   # antisymmetric S(k|p) = −S(p|k)
    Test.@test abs(sum(S)) < 1e-10 * nrm            # conserves Σ_kΣ_p S = 0

    b = FIT.Types.LinearBinning(2π/L)
    sf = FIT.SpectralFlux.calculate_spectral_flux(û, ks; binning = b, dealiasing = FIT.Types.OrszagTwoThirds(), transform = SpectralBackends.FFTSpectralBackend())
    ss = FIT.ShellToShellTransfer.calculate_shell_to_shell_transfer(û, ks; binning = b, dealiasing = FIT.Types.OrszagTwoThirds(),
        transform = SpectralBackends.FFTSpectralBackend())
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
# The spectral flux on every execution backend reproduces the serial reduction. GPU is the device path
# on host arrays (GPUBackend(KA.CPU())); Distributed runs over the workers present.
Test.@testset "Spectral flux execution backends" begin
    L = 2π
    N = 16
    ks = FIT.Utils.wavenumber_grid((N, N), (L, L))
    kx = [ks[1][i] for i in 1:N, j in 1:N]; ky = [ks[2][j] for i in 1:N, j in 1:N]
    ψh = FFTW.fft(randn(Random.MersenneTwister(11), N, N)) ./ N^2
    û  = cat(im .* ky .* ψh, -im .* kx .* ψh; dims = 3)
    b  = FIT.Types.LinearBinning(2π / L)

    ref = FIT.SpectralFlux.calculate_spectral_flux(û, ks; binning=b, transform=SpectralBackends.FFTSpectralBackend(), execution=ComputationalBackends.SerialBackend())
    atolT = 1e-12 * (maximum(abs, ref.transfer_spectrum) + eps())
    atolΠ = 1e-12 * (maximum(abs, ref.flux) + eps())
    for exec in (ComputationalBackends.ThreadedBackend(), ComputationalBackends.DistributedBackend(), ComputationalBackends.GPUBackend(KA.CPU()))
        res = FIT.SpectralFlux.calculate_spectral_flux(û, ks; binning=b, transform=SpectralBackends.FFTSpectralBackend(), execution=exec)
        Test.@test isapprox(res.transfer_spectrum, ref.transfer_spectrum; atol=atolT)
        Test.@test isapprox(res.flux, ref.flux; atol=atolΠ)
    end
end

Test.@testset "passive-scalar GPU (KA.CPU) parity — scalar flux / shell-to-shell / mode-to-mode" begin
    L = 2π; N = 16
    ks = FIT.Utils.wavenumber_grid((N, N), (L, L))
    rng = Random.MersenneTwister(23)
    û = randn(rng, ComplexF64, N, N, 2)
    θ̂ = randn(rng, ComplexF64, N, N)
    b = FIT.Types.LinearBinning(2π / L)
    for sp in (SpectralBackends.DirectSumSpectralBackend(), SpectralBackends.FFTSpectralBackend())
        fs = FIT.calculate_scalar_flux(û, θ̂, ks; binning=b, transform=sp, execution=ComputationalBackends.SerialBackend())
        fg = FIT.calculate_scalar_flux(û, θ̂, ks; binning=b, transform=sp, execution=ComputationalBackends.GPUBackend(KA.CPU()))
        Test.@test isapprox(fg.flux, fs.flux; atol=1e-12 * (maximum(abs, fs.flux)+eps()))

        ss = FIT.calculate_scalar_shell_to_shell_transfer(û, θ̂, ks; binning=b, transform=sp, execution=ComputationalBackends.SerialBackend())
        sg = FIT.calculate_scalar_shell_to_shell_transfer(û, θ̂, ks; binning=b, transform=sp, execution=ComputationalBackends.GPUBackend(KA.CPU()))
        Test.@test isapprox(sg.transfer_matrix, ss.transfer_matrix; atol=1e-12 * (maximum(abs, ss.transfer_matrix)+eps()))

        ms = FIT.calculate_scalar_mode_to_mode_transfer(û, θ̂, ks; transform=sp, execution=ComputationalBackends.SerialBackend())
        mg = FIT.calculate_scalar_mode_to_mode_transfer(û, θ̂, ks; transform=sp, execution=ComputationalBackends.GPUBackend(KA.CPU()))
        Test.@test isapprox(mg.net_transfer, ms.net_transfer; atol=1e-12 * (maximum(abs, ms.net_transfer)+eps()))
    end
end

# -----------------------------------------------------------------------
Test.@testset "GPU backend routing — clear errors, device detection, dispatch" begin
    N = 8; L = 2π
    ks = FIT.Utils.wavenumber_grid((N, N), (L, L))
    rng = Random.MersenneTwister(41)
    û = randn(rng, ComplexF64, N, N, 2)
    gpu = ComputationalBackends.GPUBackend(KA.CPU())

    # DirectSum is a scalar-indexed host reference and refuses a device array.
    Test.@test_throws ArgumentError FIT.SpectralFlux.calculate_spectral_flux(JLArrays.JLArray(û), ks; transform = SpectralBackends.DirectSumSpectralBackend())

    # TOD on GPUBackend with a host input runs the device-generic loop on host kernels and equals serial.
    X = randn(Random.MersenneTwister(42), 64, 4)
    tser = FIT.TriadicOrthogonalDecomposition.triadic_orthogonal_decomposition(X; dt = 0.05)
    tgpu = FIT.TriadicOrthogonalDecomposition.triadic_orthogonal_decomposition(X; dt = 0.05, execution = gpu)
    Test.@test isequal(tser.mode_bispectrum, tgpu.mode_bispectrum)
end

# -----------------------------------------------------------------------
# The broadcast and kernel building blocks on JLArrays under allowscalar(false) reproduce the host
# result. JLArrays has no AbstractFFTs provider, so the FFT pipelines are not run here.
Test.@testset "device-generic on JLArrays (no scalar indexing)" begin
    N = 8; L = 2π
    ks3 = FIT.Utils.wavenumber_grid((N, N, N), (L, L, L))
    û3 = randn(Random.MersenneTwister(5), ComplexF64, N, N, N, 3)

    up, um = FIT.Decomposition.decompose_field(FIT.Types.HelicalDecomposition(), û3, ks3)
    upj, umj = FIT.Decomposition.decompose_field(FIT.Types.HelicalDecomposition(), JLArrays.JLArray(û3), ks3)
    Test.@test maximum(abs, Array(upj) .- up) == 0
    Test.@test maximum(abs, Array(umj) .- um) == 0

    tp  = FIT.Decomposition.decompose_field(FIT.Types.ToroidalPoloidalDecomposition(), û3, ks3)
    tpj = FIT.Decomposition.decompose_field(FIT.Types.ToroidalPoloidalDecomposition(), JLArrays.JLArray(û3), ks3)
    Test.@test maximum(abs, Array(tpj.toroidal) .- tp.toroidal) == 0
    Test.@test maximum(abs, Array(tpj.poloidal) .- tp.poloidal) == 0

    hm  = FIT.Decomposition.decompose_field(FIT.Types.HelmholtzDecomposition(), û3, ks3)
    hmj = FIT.Decomposition.decompose_field(FIT.Types.HelmholtzDecomposition(), JLArrays.JLArray(û3), ks3)
    Test.@test maximum(abs, Array(hmj.rotational) .- hm.rotational) == 0
    Test.@test maximum(abs, Array(hmj.divergent) .- hm.divergent) == 0

    ks2 = FIT.Utils.wavenumber_grid((N, N), (L, L))
    û2 = randn(Random.MersenneTwister(6), ComplexF64, N, N, 2)
    N̂2 = randn(Random.MersenneTwister(7), ComplexF64, N, N, 2)
    td_h = similar(û2, Float64, N, N); FIT.Invariants.transfer_density!(td_h, FIT.Types.KineticEnergy(), û2, N̂2, ks2)
    td_d = JLArrays.JLArray(similar(û2, Float64, N, N))
    FIT.Invariants.transfer_density!(td_d, FIT.Types.KineticEnergy(), JLArrays.JLArray(û2), JLArrays.JLArray(N̂2), ks2)
    Test.@test maximum(abs, Array(td_d) .- td_h) == 0
    θ2 = randn(Random.MersenneTwister(8), ComplexF64, N, N, 1)
    θN = randn(Random.MersenneTwister(9), ComplexF64, N, N, 1)
    ts_h = similar(θ2, Float64, N, N); FIT.Invariants.transfer_density!(ts_h, FIT.Types.PassiveScalar(), θ2, θN, ks2)
    ts_d = JLArrays.JLArray(similar(θ2, Float64, N, N))
    FIT.Invariants.transfer_density!(ts_d, FIT.Types.PassiveScalar(), JLArrays.JLArray(θ2), JLArrays.JLArray(θN), ks2)
    Test.@test maximum(abs, Array(ts_d) .- ts_h) == 0

    # The compressible truncation, Helmholtz split and shell binning.
    dch = similar(û2); FIT.Compressible._copy_trunc!(dch, û2, ks2, (N, N), true)
    dcd = JLArrays.JLArray(similar(û2)); FIT.Compressible._copy_trunc!(dcd, JLArrays.JLArray(û2), ks2, (N, N), true)
    Test.@test maximum(abs, Array(dcd) .- dch) == 0
    kd2 = FTB.wavenumber_arrays(û2, Float64, ks2; derivative = true)
    rh = similar(û2); ch = similar(û2); FIT.Compressible._helmholtz_split!(rh, ch, û2, kd2, (N, N))
    rd = JLArrays.JLArray(similar(û2)); cd = JLArrays.JLArray(similar(û2))
    FIT.Compressible._helmholtz_split!(rd, cd, JLArrays.JLArray(û2),
        FTB.wavenumber_arrays(JLArrays.JLArray(û2), Float64, ks2; derivative = true), (N, N))
    Test.@test maximum(abs, Array(rd) .- rh) < 1e-14 * (maximum(abs, rh) + eps())
    Test.@test maximum(abs, Array(cd) .- ch) < 1e-14 * (maximum(abs, ch) + eps())
    kmag = FIT.ShellBinning.shell_coordinate(FIT.Types.IsotropicShells(), ks2); bb = FIT.Types.LinearBinning(2π / L)
    edges = FIT.ShellBinning.shell_edges(bb, maximum(kmag)); sidx = FIT.ShellBinning.assign_shells(kmag, edges)
    Nsh = length(collect(FIT.ShellBinning.shell_centers(bb, maximum(kmag))))
    Th = FIT.Compressible._bin(td_h, sidx, Nsh, Float64, ks2, (N, N))
    Td = FIT.Compressible._bin(JLArrays.JLArray(td_h), sidx, Nsh, Float64, ks2, (N, N))
    # The device sum is a tree reduction, the host one sequential: equal to round-off in the summands.
    Test.@test maximum(abs, Td .- Th) <= 8eps() * sum(abs, td_h)
end

# -----------------------------------------------------------------------
# Compressible KE spectral transfer (Singh–Tiwari–Sharma–Verma 2025): (a) the momentum-weighted
# nonlinear transfer conserves total KE, Σ_k T_u = 0; (b) the incompressible limit ρ≡1, ∇·u=0 reduces T_u and Π to
# the incompressible transfer_spectrum and flux (paper Eqs. 48–50); (c) the R/C flux channels reconstruct
# the total flux and the compressive/cross channels vanish for incompressible flow; (d) uniform
# pressure ⇒ zero pressure-dilatation. The products are cubic (ρ·u·∇u), so a field band-limited to
# |m| ≤ ⌊(N−1)/4⌋ is exact under every dealiasing; the truncation rule keeps exactly that band.
Test.@testset "Compressible energy transfer" begin
    L = 2π
    b = FIT.Types.LinearBinning(2π / L)
    FFTB = SpectralBackends.FFTSpectralBackend()
    grid(N) = (ks = FIT.Utils.wavenumber_grid((N, N), (L, L));
               (ks, [ks[1][i] for i in 1:N, j in 1:N], [ks[2][j] for i in 1:N, j in 1:N]))
    band(ks, K) = [abs(FTB.mode_frequency(ks[1], i)) <= K && abs(FTB.mode_frequency(ks[2], j)) <= K
                   for i in eachindex(ks[1]), j in eachindex(ks[2])]
    real_hat(N, rng, keep) = (f = FFTW.fft(randn(rng, N, N)) ./ N^2 .* keep;
                              FFTW.fft(real.(FFTW.bfft(f))) ./ N^2)
    # A real divergence-free velocity from a random streamfunction, band-limited to `keep`.
    solenoidal(N, rng, keep) = ((ks, kx, ky) = grid(N); ψh = real_hat(N, rng, keep);
                                cat(im .* ky .* ψh, -im .* kx .* ψh; dims = 3))
    one_hat(N) = (ρ̂ = zeros(ComplexF64, N, N); ρ̂[1, 1] = 1.0; ρ̂)

    for N in (16, 15)
        ks, _, _ = grid(N)
        K = FTB.dealias_cutoff(N, 3)
        keep = band(ks, K)
        û = solenoidal(N, Random.MersenneTwister(101), keep)
        ρ̂ = one_hat(N)
        res = FIT.Compressible.calculate_compressible_flux(û, ρ̂, ks; binning = b)
        ref = FIT.SpectralFlux.calculate_spectral_flux(û, ks; binning = b, transform = FFTB)
        scaleT = maximum(abs, ref.transfer_spectrum)
        Test.@test scaleT > 1e-8
        # (a) conservation
        Test.@test abs(sum(res.transfer_spectrum)) < 1e-12 * scaleT
        # (b) incompressible limit: T_u = ρ·(incompressible T) and Π likewise, here ρ=1
        Test.@test isapprox(res.transfer_spectrum, ref.transfer_spectrum; atol = 1e-12 * scaleT)
        Test.@test isapprox(res.flux, ref.flux; atol = 1e-12 * scaleT)
        # (c) channels: compressive & cross vanish; the four reconstruct the total flux
        Test.@test maximum(abs, res.channels.compressive) < 1e-12 * scaleT
        Test.@test maximum(abs, res.channels.comp_to_rot) < 1e-12 * scaleT
        Test.@test maximum(abs, res.channels.rot_to_comp) < 1e-12 * scaleT
        Test.@test isapprox(res.channels.rotational .+ res.channels.compressive .+
                            res.channels.rot_to_comp .+ res.channels.comp_to_rot, res.flux; atol = 1e-12 * scaleT)
        # (d) uniform pressure ⇒ ∇σ = 0 ⇒ zero pressure-dilatation
        resp = FIT.Compressible.calculate_compressible_flux(û, ρ̂, ks; binning = b, pressure_hat = one_hat(N))
        Test.@test maximum(abs, resp.pressure_dilatation.rotational) < 1e-12 * scaleT
        Test.@test maximum(abs, resp.pressure_dilatation.compressive) < 1e-12 * scaleT

        # A band-limited compressive field with variable density: every dealiasing is exact, so they
        # agree, and the transfer conserves.
        rng = Random.MersenneTwister(7)
        ûc = cat(real_hat(N, rng, keep), real_hat(N, rng, keep); dims = 3)
        ρ̂c = real_hat(N, rng, keep); ρ̂c[1, 1] += 4
        rs = map((FIT.Types.OrszagTwoThirds(), FIT.Types.NoDealiasing(), FIT.Types.PaddedThreeHalves())) do da
            FIT.Compressible.calculate_compressible_flux(ûc, ρ̂c, ks; binning = b, dealiasing = da)
        end
        sc = maximum(abs, rs[1].transfer_spectrum)
        Test.@test sc > 1e-8
        Test.@test abs(sum(rs[1].transfer_spectrum)) < 1e-12 * sc
        for r in rs[2:3]
            Test.@test isapprox(r.transfer_spectrum, rs[1].transfer_spectrum; atol = 1e-12 * sc)
            Test.@test isapprox(r.channels.rot_to_comp, rs[1].channels.rot_to_comp; atol = 1e-12 * sc)
        end

        # A broadband field: truncation keeps the band its cubic products leave alias-free, so the
        # transfer still conserves.
        ûb = cat(real_hat(N, rng, trues(N, N)), real_hat(N, rng, trues(N, N)); dims = 3)
        ρ̂b = real_hat(N, rng, trues(N, N)); ρ̂b[1, 1] += 4
        rb = FIT.Compressible.calculate_compressible_flux(ûb, ρ̂b, ks; binning = b)
        Test.@test abs(sum(rb.transfer_spectrum)) < 1e-12 * maximum(abs, rb.transfer_spectrum)

        # Padded at constant density: the cubic grid gives the quadratic term exactly, so the transfer
        # equals the padded incompressible one on a broadband field, Nyquist modes included.
        ûs = solenoidal(N, Random.MersenneTwister(11), trues(N, N))
        pad = FIT.Types.PaddedThreeHalves()
        rp = FIT.Compressible.calculate_compressible_flux(ûs, ρ̂, ks; binning = b, dealiasing = pad)
        rpi = FIT.SpectralFlux.calculate_spectral_flux(ûs, ks; binning = b, dealiasing = pad, transform = FFTB)
        sp = maximum(abs, rpi.transfer_spectrum)
        Test.@test sp > 1e-8
        Test.@test isapprox(rp.transfer_spectrum, rpi.transfer_spectrum; atol = 1e-12 * sp)
        Test.@test maximum(abs, rp.channels.compressive) < 1e-12 * sp
    end

    # The direct-sum transforms on the padded grid equal the FFT ones.
    N = 8
    ks, _, _ = grid(N)
    rng = Random.MersenneTwister(13)
    ûd = cat(real_hat(N, rng, trues(N, N)), real_hat(N, rng, trues(N, N)); dims = 3)
    ρ̂d = real_hat(N, rng, trues(N, N)); ρ̂d[1, 1] += 4
    kw = (; binning = b, dealiasing = FIT.Types.PaddedThreeHalves(), pressure_hat = ρ̂d)
    rf = FIT.Compressible.calculate_compressible_flux(ûd, ρ̂d, ks; kw..., transform = FFTB)
    rd = FIT.Compressible.calculate_compressible_flux(ûd, ρ̂d, ks; kw...,
             transform = SpectralBackends.DirectSumSpectralBackend())
    sd = maximum(abs, rf.transfer_spectrum)
    Test.@test isapprox(rd.transfer_spectrum, rf.transfer_spectrum; atol = 1e-12 * sd)
    Test.@test isapprox(rd.pressure_dilatation.compressive, rf.pressure_dilatation.compressive;
                        atol = 1e-12 * maximum(abs, rf.pressure_dilatation.compressive))

    # GPUBackend with host inputs: on the KernelAbstractions CPU backend the device memory is host
    # memory, so the call runs the serial pipeline and returns its result.
    gpu = ComputationalBackends.GPUBackend(KA.CPU())
    for da in (FIT.Types.OrszagTwoThirds(), FIT.Types.PaddedThreeHalves())
        kg = (; binning = b, dealiasing = da, pressure_hat = ρ̂d, transform = FFTB)
        rser = FIT.Compressible.calculate_compressible_flux(ûd, ρ̂d, ks; kg...)
        rgpu = FIT.Compressible.calculate_compressible_flux(ûd, ρ̂d, ks; kg..., execution = gpu)
        Test.@test maximum(abs, rser.transfer_spectrum) > 1e-8
        Test.@test maximum(abs, rser.channels.rot_to_comp) > 1e-8
        Test.@test maximum(abs, rser.pressure_dilatation.compressive) > 1e-8
        Test.@test rgpu.transfer_spectrum == rser.transfer_spectrum
        Test.@test rgpu.flux == rser.flux
        Test.@test rgpu.channels == rser.channels
        Test.@test rgpu.pressure_dilatation == rser.pressure_dilatation
    end
end

# -----------------------------------------------------------------------
_nufft_ts_reuse(ws, fields) = (FIT.to_spectral!(ws, fields); @allocated FIT.to_spectral!(ws, fields))

# Optional-package entries: the CairoMakie plot guard, the scattered-Cartesian NUFFT entries through
# FlowTransformBindings, and the FlowFieldSpectra physical → spectral front end.
Test.@testset "Optional-package entries (CairoMakie / NUFFT / FlowFieldSpectra)" begin
    L = 2π; N = 16
    ks = FIT.Utils.wavenumber_grid((N, N), (L, L))
    kx = [ks[1][i] for i in 1:N, j in 1:N]; ky = [ks[2][j] for i in 1:N, j in 1:N]
    ψh = FFTW.fft(randn(Random.MersenneTwister(21), N, N)) ./ N^2
    b  = FIT.Types.LinearBinning(2π / L)

    Test.@testset "CairoMakie plot dispatch" begin
        nF = 6; nm = 2
        freqs = collect(range(-2.0, 2.0; length=nF))
        λ = abs.(randn(Random.MersenneTwister(3), nF, nF, nm))
        Tb = randn(Random.MersenneTwister(4), nF, nF, nm)
        tod = FIT.Types.TriadicOrthogonalDecompositionResult(freqs, λ, Dict{Tuple{Int,Int},Any}(), Tb, nothing, nothing)
        Test.@test_throws ArgumentError FIT.plot_energy_transfer(tod; mode=99)
    end

    Test.@testset "NUFFT scattered coarse-graining — both providers vs exact NDFT" begin
        # Flux, stress τ̄ and strain S̄ against a brute-force NDFT for both NUFFT providers. τ̄ depends on
        # |k| alone, so the S̄ comparison is the one that fixes the sign of the Nyquist mode.
        Random.seed!(3)
        Np = 2000; ms = (20, 20)
        X = 2π .* rand(Np); Y = 2π .* rand(Np)
        U = @. sin(X) * cos(Y); V = @. -cos(X) * sin(Y)
        ℓ = 0.8; filt = FIT.Types.GaussianFilter()

        Lx = 2π; Ly = 2π                       # the periodic domain, X, Y ∈ [0, 2π)
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

        for transform in (FTB.FINUFFTBackend(), FTB.NonuniformFFTsBackend())
            r = FIT.nufft_coarse_graining_flux((U, V), (X, Y), ℓ, filt, ms; transform, Ls = (Lx, Ly), return_diagnostics = true)
            Test.@test relq(τxx, r.stress_tensor[:, 1, 1]) < 1e-6
            Test.@test relq(τyy, r.stress_tensor[:, 2, 2]) < 1e-6
            Test.@test relq(Sxx, r.strain_rate[:, 1, 1]) < 1e-6
            Test.@test relq(Sxy, r.strain_rate[:, 1, 2]) < 1e-6
            Test.@test relq(Πref, r.flux_field) < 1e-6
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
            for transform in (FTB.FINUFFTBackend(), FTB.NonuniformFFTsBackend())
                ro = FIT.nufft_coarse_graining_flux((U, V), (X, Y), ℓ, filt, mso; transform, Ls = (Lx, Ly), return_diagnostics = true)
                Test.@test relq(Sxxo, ro.strain_rate[:, 1, 1]) < 1e-6
                Test.@test relq(Πo, ro.flux_field) < 1e-6
            end
        end
    end

    Test.@testset "NUFFT scattered → to_spectral (uniform reconstruction feeds the flux family)" begin
        Nn = 8; Ln = 2π
        xs1 = [(i - 1) * Ln / Nn for i in 1:Nn]
        Xg = [xs1[i] for i in 1:Nn, j in 1:Nn]; Yg = [xs1[j] for i in 1:Nn, j in 1:Nn]
        Random.seed!(31)
        ug = randn(Nn, Nn); vg = randn(Nn, Nn)

        # Samples on a uniform grid give û = fft(u)/Nᵈ to the NUFFT tolerance.
        û_sc, ks_sc = FIT.to_spectral((vec(ug), vec(vg)), (vec(Xg), vec(Yg)), (Nn, Nn);
                                        transform = FTB.FINUFFTBackend(), Ls = (Ln, Ln), tol = 1e-13)
        û_man  = cat(FFTW.fft(ug), FFTW.fft(vg); dims = 3) ./ Nn^2
        ks_man = FIT.Utils.wavenumber_grid((Nn, Nn), (Ln, Ln))
        Test.@test isapprox(û_sc, û_man; atol = 1e-12)
        Test.@test all(isapprox.(ks_sc, ks_man; atol = 1e-10))

        # On scattered points a single-mode field peaks at its wavenumber.
        Np = 4000
        rng2 = Random.MersenneTwister(99)
        xr = rand(rng2, Np) .* Ln; yr = rand(rng2, Np) .* Ln
        kx0 = 2π / Ln * 2; ky0 = 2π / Ln * 3
        us = cos.(kx0 .* xr .+ ky0 .* yr)
        ûs, kss = FIT.to_spectral((us,), (xr, yr), (Nn, Nn); transform = FTB.FINUFFTBackend(), Ls = (Ln, Ln))
        peak = argmax(abs.(ûs[:, :, 1]))
        Test.@test abs(abs(kss[1][peak[1]]) - kx0) < 1e-8
        Test.@test abs(abs(kss[2][peak[2]]) - ky0) < 1e-8

        # The workspace form reconstructs û = fft(u)/Nᵈ on both providers. The FINUFFT plan runs on one
        # thread and a reused workspace allocates nothing.
        for transform in (FTB.FINUFFTBackend(), FTB.NonuniformFFTsBackend())
            wsts = FIT.NUFFTToSpectralWorkspace((vec(Xg), vec(Yg)), (Nn, Nn); transform, ncomponents = 2, Ls = (Ln, Ln))
            û_ip, _ = FIT.to_spectral!(wsts, (vec(ug), vec(vg)))
            Test.@test isapprox(û_ip, û_man; atol = 1e-8)
            if transform isa FTB.FINUFFTBackend
                Test.@test _nufft_ts_reuse(wsts, (vec(ug), vec(vg))) == 0
            end
        end

        # The NonuniformFFTs plan built on GPUBackend(KA.CPU()) runs the device path on host arrays.
        û_host, _ = FIT.to_spectral((vec(ug), vec(vg)), (vec(Xg), vec(Yg)), (Nn, Nn);
                                    transform = FTB.NonuniformFFTsBackend(), Ls = (Ln, Ln))
        û_dev, _  = FIT.to_spectral((vec(ug), vec(vg)), (vec(Xg), vec(Yg)), (Nn, Nn);
                                    transform = FTB.NonuniformFFTsBackend(), Ls = (Ln, Ln),
                                    execution = ComputationalBackends.GPUBackend(KA.CPU()))
        Test.@test isapprox(Array(û_dev), û_host; atol = 1e-10)
        Test.@test isapprox(Array(û_dev), û_man;  atol = 1e-8)

        # Float32 with a tol below eps(Float32): the spreading half-support clamps to eps(FT).
        û_f32, _ = FIT.to_spectral((Float32.(vec(ug)), Float32.(vec(vg))),
                                   (Float32.(vec(Xg)), Float32.(vec(Yg))), (Nn, Nn);
                                   transform = FTB.NonuniformFFTsBackend(), Ls = (Ln, Ln), tol = 1e-9)
        Test.@test maximum(abs, û_f32 .- ComplexF32.(û_man)) < 1e-2

        # The scattered form takes a NUFFT backend.
        Test.@test_throws ArgumentError FIT.to_spectral((us,), (xr, yr), (Nn, Nn); transform = SpectralBackends.FFTSpectralBackend(), Ls = (Ln, Ln))

        # The NonuniformFFTs provider runs a real-input (r2c) plan and expands the half spectrum to the full
        # one. On an even axis d ≥ 2 the Hermitian fold needs mode +N_d/2, which is not an output mode
        # (for scattered points +N_d/2 ≠ −N_d/2), so it is read from the oversampled spectrum. Gated against
        # the complex FINUFFT path over even, odd and mixed mode counts.
        rngr = Random.MersenneTwister(5)
        Nr = 3000
        xr2 = Ln .* rand(rngr, Nr); yr2 = Ln .* rand(rngr, Nr)
        ur2 = @. sin(3 * xr2) * cos(2 * yr2) + 0.25 * cos(7 * xr2)
        for msr in ((20, 20), (21, 21), (20, 21), (21, 20))
            û_r2c, _ = FIT.to_spectral((ur2,), (xr2, yr2), msr; transform = FTB.NonuniformFFTsBackend(), Ls = (Ln, Ln))
            û_cpx, _ = FIT.to_spectral((ur2,), (xr2, yr2), msr; transform = FTB.FINUFFTBackend(), Ls = (Ln, Ln))
            Test.@test sqrt(sum(abs2, û_r2c .- û_cpx) / sum(abs2, û_cpx)) < 1e-7
        end
    end

    Test.@testset "scattered one-call calculate_energy_transfer (Cartesian spectral family)" begin
        # A NUFFT backend routes the (method, fields, coords, ms) entry to the scattered reconstruction and
        # then the coefficient diagnostic. One method serves the spectral flux, shell-to-shell and
        # mode-to-mode methods.
        Nn = 8; Ln = 2π
        xs1 = [(i - 1) * Ln / Nn for i in 1:Nn]
        Xg = [xs1[i] for i in 1:Nn, j in 1:Nn]; Yg = [xs1[j] for i in 1:Nn, j in 1:Nn]
        Random.seed!(17); ug = randn(Nn, Nn); vg = randn(Nn, Nn)
        fields = (vec(ug), vec(vg)); coords = (vec(Xg), vec(Yg))
        b = FIT.Types.LinearBinning(2π / Ln)
        for transform in (FTB.FINUFFTBackend(), FTB.NonuniformFFTsBackend())
            û_oc, ks_oc = FIT.to_spectral(fields, coords, (Nn, Nn); transform, Ls = (Ln, Ln))
            sf1 = FIT.calculate_energy_transfer(FIT.Types.SpectralFluxMethod(b), fields, coords, (Nn, Nn); transform, Ls = (Ln, Ln))
            sf2 = FIT.calculate_energy_transfer(FIT.Types.SpectralFluxMethod(b), û_oc, ks_oc)
            Test.@test isapprox(sf1.flux, sf2.flux; rtol = 1e-10, atol = 1e-12)
        end
    end

    Test.@testset "FlowFieldSpectra front-end" begin
        # A uniform periodic field through FlowFieldSpectra, then the spectral flux. Band-limited to
        # |k| ≤ 3, so the 2/3 dealiasing is a no-op and Σ_k T = 0 holds exactly.
        ψbl = copy(ψh)
        for i in 1:N, j in 1:N
            (abs(kx[i, j]) <= 3 && abs(ky[i, j]) <= 3) || (ψbl[i, j] = 0)
        end
        ûbl = cat(im .* ky .* ψbl, -im .* kx .* ψbl; dims = 3)
        xs = range(0, 2π; length=N+1)[1:N]   # uniform axis as a range → FlowGeometries infers its period
        # The physical field of the fft/N² coefficients is bfft(û) = Σ_k û e^{ik·x}.
        uxp = real.(FFTW.bfft(ûbl[:, :, 1]))
        uyp = real.(FFTW.bfft(ûbl[:, :, 2]))
        ffs = FIT.calculate_energy_transfer(FIT.Types.SpectralFluxMethod(b), (uxp, uyp), (xs, xs), (N, N))
        Test.@test abs(sum(ffs.transfer_spectrum)) < 1e-8 * (maximum(abs, ffs.transfer_spectrum) + eps())
        # The front end reproduces the transfer of the field's own FFT coefficients.
        ref = FIT.SpectralFlux.calculate_spectral_flux(ûbl, ks; binning=b, transform=SpectralBackends.FFTSpectralBackend())
        Test.@test isapprox(ffs.transfer_spectrum, ref.transfer_spectrum;
                            atol = 1e-10 * (maximum(abs, ref.transfer_spectrum) + eps()))
    end
end

# -----------------------------------------------------------------------
# Spherical spectral transfer (FSH extension, 2D-barotropic). Σ_l T = 0, from ∫ψ J(ψ,ζ) dΩ = 0, holds to
# round-off when the quadratic Jacobian is formed on the 2·lmax grid.
Test.@testset "Spherical spectral transfer (FastSphericalHarmonics, 2D barotropic)" begin
    lmax = 20; N = lmax + 1
    rng = Random.MersenneTwister(2024)
    # Band-limited real vorticity field on the FSH equiangular grid.
    ζ = FSH.spinsph_evaluate(FSH.spinsph_transform(randn(rng, N, 2N - 1), 0), 0)
    ζ = real.(ζ)

    res = FIT.calculate_energy_transfer(FIT.Types.SphericalTransferMethod(), ζ)
    scaleE = maximum(abs, res.energy_transfer) + eps()
    scaleZ = maximum(abs, res.enstrophy_transfer) + eps()
    Test.@test scaleE > 1e-6
    Test.@test scaleZ > 1e-6
    Test.@test abs(sum(res.energy_transfer))    < 1e-12 * scaleE
    Test.@test abs(sum(res.enstrophy_transfer)) < 1e-12 * scaleZ
    # A₀₀ = mean(J) = ∫∇·(ζu) dΩ = 0, so l = 0 carries no enstrophy transfer.
    Test.@test abs(res.enstrophy_transfer[1]) < 1e-12 * scaleZ

    # On the same broadband field the aliased path does not conserve.
    aliased = FIT.calculate_energy_transfer(FIT.Types.SphericalTransferMethod(), ζ; dealias = false)
    Test.@test abs(sum(aliased.enstrophy_transfer)) > 1e-6 * (maximum(abs, aliased.enstrophy_transfer) + eps())

    Test.@test_throws ArgumentError FIT.calculate_energy_transfer(FIT.Types.SphericalTransferMethod(), randn(N, N))
    # The regular grid takes the FSHT transform, and the transfer runs as one transform pipeline.
    Test.@test_throws ArgumentError FIT.calculate_energy_transfer(FIT.Types.SphericalTransferMethod(), ζ; transform = SpectralBackends.NUFSHTSpectralBackend())
    Test.@test_throws ArgumentError FIT.calculate_energy_transfer(FIT.Types.SphericalTransferMethod(), ζ; execution = ComputationalBackends.DistributedBackend())
    ws = FIT.Spherical.SphericalTransferWorkspace(lmax; radius = 1.0, dealias = true)
    Test.@test_throws ArgumentError FIT.Spherical.calculate_spherical_transfer!(ws, randn(N, N))
end

Test.@testset "Spherical transfer equals −dE_l/dt from the closed-form Jacobian" begin
    # ζ = Z₂ + Z₃ + Z₄, each Z_l a harmonic homogeneous polynomial of degree l (a pure degree-l field on
    # the unit sphere), so ψ = −Σ_l Z_l/(l(l+1)). With u = r̂×∇ψ the advection is
    # J = u·∇ζ = r̂·(∇ψ × ∇ζ), and ζ_t = −J gives −dE_l/dt = −∫ψ_l J dΩ and −dZ_l/dt = ∫ζ_l J dΩ.
    Z = (x -> x[1] * x[2] + 0.4x[2] * x[3] - 0.7(x[1]^2 - x[2]^2),
         x -> x[1]^3 - 3x[1] * x[2]^2 + 0.6x[1] * x[2] * x[3],
         x -> x[1]^4 - 6x[1]^2 * x[2]^2 + x[2]^4 + 0.5(x[1]^3 * x[3] - 3x[1] * x[2]^2 * x[3]))
    ∇Z = (x -> (x[2] - 1.4x[1], x[1] + 0.4x[3] + 1.4x[2], 0.4x[2]),
          x -> (3x[1]^2 - 3x[2]^2 + 0.6x[2] * x[3], -6x[1] * x[2] + 0.6x[1] * x[3], 0.6x[1] * x[2]),
          x -> (4x[1]^3 - 12x[1] * x[2]^2 + 1.5(x[1]^2 - x[2]^2) * x[3],
                -12x[1]^2 * x[2] + 4x[2]^3 - 3x[1] * x[2] * x[3], 0.5(x[1]^3 - 3x[1] * x[2]^2)))
    λ = (6, 12, 20)
    cross(a, b) = (a[2]b[3] - a[3]b[2], a[3]b[1] - a[1]b[3], a[1]b[2] - a[2]b[1])
    function J(x)
        gζ = reduce(.+, (g(x) for g in ∇Z))
        gψ = reduce(.+, (.-∇Z[i](x) ./ λ[i] for i in 1:3))
        return sum(x .* cross(gψ, gζ))
    end
    lmax = 12; N = lmax + 1; M = 2N - 1
    θs, φs = FSH.sph_points(N)
    # Fejér's first rule on θ_j = π(j − ½)/N, exact for polynomials in cos θ of degree < N.
    w = [(2 / N) * (1 - 2sum(cos(2k * θ) / (4k^2 - 1) for k in 1:(N ÷ 2))) for θ in θs]
    Test.@test sum(w) ≈ 2
    pts = [(sin(θ) * cos(φ), sin(θ) * sin(φ), cos(θ)) for θ in θs, φ in φs]
    sphere_integral(f) = sum(w[j] * (2π / M) * f(pts[j, k]) for j in 1:N, k in 1:M)
    TE = zeros(lmax + 1); TZ = zeros(lmax + 1)
    for (i, l) in enumerate(2:4)
        TE[l + 1] = sphere_integral(x -> Z[i](x) / λ[i] * J(x))
        TZ[l + 1] = sphere_integral(x -> Z[i](x) * J(x))
    end
    Test.@test maximum(abs, TE) > 1e-2
    # FastTransforms on one thread and on every thread.
    for execution in (ComputationalBackends.SerialBackend(), ComputationalBackends.ThreadedBackend())
        res = FIT.calculate_energy_transfer(FIT.Types.SphericalTransferMethod(), [sum(z(p) for z in Z) for p in pts];
                                            execution = execution)
        Test.@test isapprox(res.energy_transfer, TE; atol = 1e-12 * maximum(abs, TE))
        Test.@test isapprox(res.enstrophy_transfer, TZ; atol = 1e-12 * maximum(abs, TZ))
    end
end

# -----------------------------------------------------------------------
# Scattered spherical transfer (NUFSHT extension), against the FSH regular-grid transfer of the same
# band-limited field: real-SH coefficients C evaluated on the FSH grid with sph_evaluate and at scattered
# points with nusht_type2!. Coefficient recovery is well-conditioned for equidistributed
# (spherical-Fibonacci) points.
Test.@testset "Scattered spherical transfer (NUFSHT, 2D barotropic)" begin
    a = 1.0
    lmax = 8; N = lmax + 1
    rng = Random.MersenneTwister(99)
    C = zeros(N, 2N - 1)
    for ℓ in 0:lmax, m in -ℓ:ℓ; C[FSH.sph_mode(ℓ, m)] = randn(rng); end   # broadband real-SH field
    ζgrid = FSH.sph_evaluate(C)
    resF = FIT.calculate_energy_transfer(FIT.Types.SphericalTransferMethod(radius = a), ζgrid)

    # Spherical-Fibonacci scattered points (equidistributed), M ≥ (lmax+1)² for the fit.
    Msolve = 8 * (2lmax + 1)^2
    ga = π * (3 - sqrt(5))
    zf = [1 - 2 * (k + 0.5) / Msolve for k in 0:Msolve-1]
    θs = acos.(clamp.(zf, -1.0, 1.0)); φs = mod.(ga .* (0:Msolve-1), 2π)
    ζscat = zeros(Msolve)
    NUFSHT.nusht_type2!(ζscat, C, NUFSHT.make_plan(θs, φs, lmax; tol = 1e-12))

    res = FIT.calculate_energy_transfer(FIT.Types.SphericalTransferMethod(radius = a), ζscat, (θs, φs);
                                        lmax = lmax, tol = 1e-12, rtol = 1e-13)
    Test.@test res.converged

    # A fit stopped at `maxiter` is marked unconverged.
    short = FIT.calculate_energy_transfer(FIT.Types.SphericalTransferMethod(radius = a), ζscat, (θs, φs);
                                          lmax = lmax, tol = 1e-12, rtol = 1e-13, maxiter = 2)
    Test.@test !short.converged
    Test.@test short.iterations == 2

    # The default rtol is reachable in Float32.
    r32 = FIT.calculate_energy_transfer(FIT.Types.SphericalTransferMethod(radius = a), Float32.(ζscat),
                                        (Float32.(θs), Float32.(φs)); lmax = lmax)
    Test.@test r32.converged
    Test.@test maximum(abs, r32.energy_transfer .- res.energy_transfer) < 1e-3 * maximum(abs, res.energy_transfer)

    scaleE = maximum(abs, resF.energy_transfer) + eps()
    scaleZ = maximum(abs, resF.enstrophy_transfer) + eps()
    Test.@test scaleE > 1e-3
    Test.@test abs(sum(res.energy_transfer))    < 1e-8 * scaleE
    Test.@test abs(sum(res.enstrophy_transfer)) < 1e-8 * scaleZ
    # Both paths are exact for a field band-limited at lmax, to the NUFFT and fit tolerances.
    Test.@test maximum(abs.(res.energy_transfer    .- resF.energy_transfer))    < 1e-8 * scaleE
    Test.@test maximum(abs.(res.enstrophy_transfer .- resF.enstrophy_transfer)) < 1e-8 * scaleZ

    # Too few points for the fit.
    Test.@test_throws ArgumentError FIT.calculate_energy_transfer(
        FIT.Types.SphericalTransferMethod(), ζscat[1:10], (θs[1:10], φs[1:10]); lmax = lmax)
    # Scattered points take the NUFSHT transform.
    Test.@test_throws ArgumentError FIT.calculate_energy_transfer(FIT.Types.SphericalTransferMethod(radius = a), ζscat, (θs, φs); lmax = lmax, transform = SpectralBackends.FSHTSpectralBackend())
    ws = FIT.Spherical.ScatteredSphericalTransferWorkspace((θs, φs), lmax; radius = a, tol = 1e-12, rtol = 1e-13)
    Test.@test_throws DimensionMismatch FIT.Spherical.calculate_spherical_transfer!(ws, ζscat[1:end-1])

    # The spin plans on either NUFFT library give the same transfer.
    for lib in (FTB.FINUFFTBackend(), FTB.NonuniformFFTsBackend())
        wl = FIT.Spherical.ScatteredSphericalTransferWorkspace((θs, φs), lmax; radius = a, tol = 1e-12, rtol = 1e-13,
                                                               nufft = lib)
        rl = FIT.Spherical.calculate_spherical_transfer!(wl, ζscat)
        Test.@test maximum(abs.(rl.energy_transfer .- res.energy_transfer)) < 1e-10 * scaleE
        FIT.close!(wl)
    end

    # A device backend, given host coordinates and field.
    rg = FIT.calculate_energy_transfer(FIT.Types.SphericalTransferMethod(radius = a), ζscat, (θs, φs);
                                       lmax = lmax, tol = 1e-12, rtol = 1e-13,
                                       execution = ComputationalBackends.GPUBackend(KA.CPU()))
    Test.@test rg.converged
    Test.@test maximum(abs.(rg.energy_transfer .- res.energy_transfer)) < 1e-10 * scaleE
    Test.@test maximum(abs.(rg.enstrophy_transfer .- res.enstrophy_transfer)) < 1e-10 * scaleZ
end

# -----------------------------------------------------------------------
# Divergent horizontal-KE spectral transfer for the full (rotational + divergent) flow, on the regular
# grid (FastSphericalHarmonics) and scattered points (NUFSHT): total-KE conservation Σ_l T ≈ 0 from the
# skew-symmetric advection, the reduction to the barotropic energy transfer for non-divergent flow
# (radius 1 and 2) with T_div ≈ 0 there, and FSH↔NUFSHT parity on one field.
Test.@testset "Divergent spherical transfer (rotational + divergent)" begin
    lmax = 10; N = lmax + 1
    rng = Random.MersenneTwister(11)
    _nabla(freal) = (ð = FSH.spinsph_eth(FSH.spinsph_transform(Matrix{Float64}(freal), 0), 0);
        G = FSH.spinsph_evaluate(ð, 1); -[complex(G[i, j][1], G[i, j][2]) for i in axes(G, 1), j in axes(G, 2)])
    _randc() = (C = zeros(Float64, N, 2N - 1); for l in 1:lmax, m in -l:l
            C[FSH.spinsph_mode(0, l, m)] = randn(rng) / (l + 1); end; C)
    _lapf(f) = FSH.spinsph_evaluate((C = FSH.spinsph_transform(Matrix{Float64}(f), 0); D = copy(C);
        for l in 0:lmax, m in -l:l; i = FSH.spinsph_mode(0, l, m); D[i] = -l * (l + 1) * C[i]; end; D), 0)
    ψf = FSH.spinsph_evaluate(_randc(), 0); χf = FSH.spinsph_evaluate(_randc(), 0)
    U = im .* _nabla(ψf) .+ _nabla(χf); uθ = real.(U); uφ = imag.(U)
    Ur = im .* _nabla(ψf); uθr = real.(Ur); uφr = imag.(Ur)
    ζr = _lapf(ψf)

    Test.@testset "FSH regular-grid" begin
        r = FIT.calculate_energy_transfer(FIT.Types.DivergentSphericalTransferMethod(), (uθ, uφ))
        Test.@test abs(sum(r.energy_transfer)) / maximum(abs, r.energy_transfer) < 1e-8
        Test.@test maximum(abs, r.divergent_transfer) > 1e-3
        TE = FIT.calculate_energy_transfer(FIT.Types.SphericalTransferMethod(), ζr).energy_transfer
        for execution in (ComputationalBackends.SerialBackend(), ComputationalBackends.ThreadedBackend())
            rr = FIT.calculate_energy_transfer(FIT.Types.DivergentSphericalTransferMethod(), (uθr, uφr);
                                               execution = execution)
            Test.@test maximum(abs, rr.energy_transfer .- TE) / maximum(abs, TE) < 1e-8
            Test.@test maximum(abs, rr.divergent_transfer) < 1e-9
        end
        a = 2.0
        ra = FIT.calculate_energy_transfer(FIT.Types.DivergentSphericalTransferMethod(; radius = a), (uθr ./ a, uφr ./ a))
        TEa = FIT.calculate_energy_transfer(FIT.Types.SphericalTransferMethod(; radius = a), ζr ./ a^2).energy_transfer
        Test.@test maximum(abs, ra.energy_transfer .- TEa) / maximum(abs, TEa) < 1e-8
    end

    Test.@testset "NUFSHT scattered + FSH↔NUFSHT parity" begin
        Θ, Φ = FSH.sph_points(N); M = N * (2N - 1)
        θv = Vector{Float64}(undef, M); φv = Vector{Float64}(undef, M)
        for j in 1:2N-1, i in 1:N
            k = i + (j - 1) * N; θv[k] = Θ[i]; φv[k] = Φ[j]
        end
        rn = FIT.calculate_energy_transfer(FIT.Types.DivergentSphericalTransferMethod(), (vec(uθ), vec(uφ)),
                                            (θv, φv); lmax = lmax, tol = 1e-12, rtol = 1e-13)
        Test.@test abs(sum(rn.energy_transfer)) / maximum(abs, rn.energy_transfer) < 1e-8
        rf = FIT.calculate_energy_transfer(FIT.Types.DivergentSphericalTransferMethod(), (uθ, uφ))
        Test.@test maximum(abs, rf.energy_transfer .- rn.energy_transfer) / maximum(abs, rf.energy_transfer) < 1e-8
        rg = FIT.calculate_energy_transfer(FIT.Types.DivergentSphericalTransferMethod(), (vec(uθ), vec(uφ)),
                                           (θv, φv); lmax = lmax, tol = 1e-12, rtol = 1e-13,
                                           execution = ComputationalBackends.GPUBackend(KA.CPU()))
        Test.@test maximum(abs, rg.energy_transfer .- rn.energy_transfer) / maximum(abs, rn.energy_transfer) < 1e-8
    end
end

# -----------------------------------------------------------------------
# Every Cartesian coefficient diagnostic refuses a scattered (NUFFT) and a spherical transform with an
# `ArgumentError`.
Test.@testset "Backend matrix — transform axis (facade-proof)" begin
    Random.seed!(0)
    N = 8; L = 2π
    ks = FIT.Utils.wavenumber_grid((N, N), (L, L))
    kx = [ks[1][i] for i in 1:N, j in 1:N]; ky = [ks[2][j] for i in 1:N, j in 1:N]
    ψh = FFTW.fft(randn(N, N)) ./ N^2
    û = cat(im .* ky .* ψh, -im .* kx .* ψh; dims = 3)          # divergence-free 2D velocity coeffs
    ρ̂ = FFTW.fft(1.0 .+ 0.1 .* randn(N, N)) ./ N^2
    b = FIT.Types.LinearBinning(2π / L); bands = FIT.Types.SmoothBands([2.0, 4.0])
    cart = (
        ("spectral_flux",  sp -> FIT.SpectralFlux.calculate_spectral_flux(û, ks; binning = b, transform = sp)),
        ("shell_to_shell", sp -> FIT.ShellToShellTransfer.calculate_shell_to_shell_transfer(û, ks; binning = b, transform = sp)),
        ("mode_to_mode",   sp -> FIT.calculate_mode_to_mode_transfer(û, ks; transform = sp)),
        ("band_to_band",   sp -> FIT.BandTransfer.calculate_band_to_band_transfer(û, ks; bands = bands, transform = sp)),
        ("compressible",   sp -> FIT.Compressible.calculate_compressible_flux(û, ρ̂, ks; transform = sp)),
        ("partial_fluxes", sp -> FIT.calculate_partial_fluxes(û, ks; decomposition = FIT.Types.HelmholtzDecomposition(), transform = sp)),
    )
    for (name, invoke) in cart, sp in (FTB.FINUFFTBackend(), SpectralBackends.FSHTSpectralBackend())
        Test.@test_throws ArgumentError invoke(sp)
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
