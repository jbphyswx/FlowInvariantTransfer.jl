# ---------------------------------------------------------------------------
# Decomposition-channel fluxes.
#
# The helical channels are the rotational form of Alexakis (2017, Eq. 17),
# `T^{s₁s₂s₃}(k) = Re{û_{s₁}*·[ω_{s₃} × u_{s₂}]^(k)}`, gated against that product formed here with
# FFTW from the helical components. `(u·∇)u = ω × u + ∇(½|u|²)` makes each pair sum over the last two
# labels equal the advective pair sum, while single heterochiral channels of the two forms differ.
# ---------------------------------------------------------------------------

using Test: Test
using Random: Random
using FFTW: FFTW
using OhMyThreads: OhMyThreads
using FlowInvariantTransfer: FlowInvariantTransfer as FIT
using ComputationalBackends: ComputationalBackends
using SpectralBackends: SpectralBackends

# Integer frequencies of an N-point axis in FFT order, and a divergence-free real 3-D field on the 2π box
# band-limited to `|m_d| ≤ K`, as full-layout coefficients `fft(u)/N³`.
_pf_freq(N) = [i <= N ÷ 2 ? i - 1 : i - 1 - N for i in 1:N]
function _pf_solenoidal(N, K; seed = 9)
    rng = Random.MersenneTwister(seed)
    m = _pf_freq(N)
    keep = [abs(m[i]) <= K && abs(m[j]) <= K && abs(m[l]) <= K for i in 1:N, j in 1:N, l in 1:N]
    Â = [FFTW.fft(real.(FFTW.ifft(keep .* FFTW.fft(randn(rng, N, N, N))))) ./ N^3 for _ in 1:3]
    kx = [m[i] for i in 1:N, j in 1:N, l in 1:N]
    ky = [m[j] for i in 1:N, j in 1:N, l in 1:N]
    kz = [m[l] for i in 1:N, j in 1:N, l in 1:N]
    û = cat(im .* (ky .* Â[3] .- kz .* Â[2]), im .* (kz .* Â[1] .- kx .* Â[3]),
            im .* (kx .* Â[2] .- ky .* Â[1]); dims = 4)
    return û, (kx, ky, kz)
end

# Shell sums of a per-mode density on the full layout.
function _pf_binned(t, sidx, nsh)
    T = zeros(nsh)
    for I in CartesianIndices(t)
        n = sidx[I]; n == 0 || (T[n] += t[I])
    end
    return T
end

Test.@testset "decomposition-channel fluxes" begin
    FFTB = SpectralBackends.FFTSpectralBackend()
    b = FIT.Types.LinearBinning(1.0)

    Test.@testset "helical channels are −⟨u^{s₁}·(u^{s₂} × ω^{s₃})⟩" begin
        N = 8
        û, (kx, ky, kz) = _pf_solenoidal(N, 2)
        ks = FIT.Utils.wavenumber_grid((N, N, N), (2π, 2π, 2π))
        hp = FIT.calculate_helical_partial_fluxes(û, ks; binning = b, transform = FFTB)
        comps = FIT.Decomposition.decompose_field(FIT.Types.HelicalDecomposition(), û, ks)
        kmag = FIT.ShellBinning.shell_coordinate(FIT.Types.IsotropicShells(), ks)
        edges = FIT.ShellBinning.shell_edges(b, maximum(kmag))
        sidx = FIT.ShellBinning.assign_shells(kmag, edges); nsh = length(edges) - 1
        synth(F) = real.(FFTW.bfft(F))
        page(A, c) = A[:, :, :, c]
        names = (:positive, :negative)
        rot = Dict{NTuple{3,Symbol}, Vector{Float64}}()
        for s2 in names, s3 in names
            u2 = ntuple(c -> synth(page(comps[s2], c)), 3)
            a = comps[s3]
            ω = (synth(im .* (ky .* page(a, 3) .- kz .* page(a, 2))),
                 synth(im .* (kz .* page(a, 1) .- kx .* page(a, 3))),
                 synth(im .* (kx .* page(a, 2) .- ky .* page(a, 1))))
            C = (ω[2] .* u2[3] .- ω[3] .* u2[2], ω[3] .* u2[1] .- ω[1] .* u2[3], ω[1] .* u2[2] .- ω[2] .* u2[1])
            N̂ = map(c -> FFTW.fft(c) ./ N^3, C)
            for s1 in names
                t = sum(real.(conj.(page(comps[s1], c)) .* N̂[c]) for c in 1:3)
                rot[(s1, s2, s3)] = _pf_binned(t, sidx, nsh)
            end
        end
        scale = maximum(maximum(abs, v) for v in values(rot))
        Test.@test scale > 1e-6                                       # the reference carries signal
        for key in keys(rot)
            Test.@test maximum(abs, hp.channels[key].transfer_spectrum .- rot[key]) < 1e-12 * scale
        end

        # The pair sums over the last two labels equal the advective ones; the heterochiral channels
        # taken one by one do not.
        adv = Dict{NTuple{3,Symbol}, Vector{Float64}}()
        for s2 in names, s3 in names
            N̂a = FIT.NonlinearTerm.compute_nonlinear_term(comps[s3], ks; advecting_hat = comps[s2], transform = FFTB)
            for s1 in names
                adv[(s1, s2, s3)] = _pf_binned(FIT.Invariants.transfer_density(FIT.Types.KineticEnergy(), comps[s1], N̂a, ks),
                                               sidx, nsh)
            end
        end
        for s1 in names, s2 in names, s3 in names
            Test.@test maximum(abs, rot[(s1, s2, s3)] .+ rot[(s1, s3, s2)] .- adv[(s1, s2, s3)] .- adv[(s1, s3, s2)]) <
                       1e-12 * scale
        end
        Test.@test maximum(abs, rot[(:positive, :positive, :negative)] .- adv[(:positive, :positive, :negative)]) > 1e-3 * scale

        # The channels sum to the advective flux of the field.
        sf = FIT.calculate_spectral_flux(û, ks; binning = b, transform = FFTB)
        Test.@test maximum(abs, hp.total.transfer_spectrum .- sf.transfer_spectrum) < 1e-12 * scale
    end

    Test.@testset "rotational term: direct sum == FFT" begin
        N = 6
        û, _ = _pf_solenoidal(N, 2; seed = 4)
        v̂, _ = _pf_solenoidal(N, 2; seed = 5)
        ks = FIT.Utils.wavenumber_grid((N, N, N), (2π, 2π, 2π))
        for da in (FIT.Types.OrszagTwoThirds(), FIT.Types.NoDealiasing(), FIT.Types.PaddedThreeHalves())
            wd = FIT.Workspaces.NonlinearTermWorkspace(û, ks; dealiasing = da)
            wf = FIT.Workspaces.NonlinearTermWorkspace(û, ks; dealiasing = da)
            FIT.NonlinearTerm.compute_rotational_term!(wd, û, v̂, ks; dealiasing = da,
                                                       transform = SpectralBackends.DirectSumSpectralBackend())
            FIT.NonlinearTerm.compute_rotational_term!(wf, û, v̂, ks; dealiasing = da, transform = FFTB)
            Test.@test maximum(abs, wf.N̂) > 1e-6
            Test.@test maximum(abs, wd.N̂ .- wf.N̂) < 1e-12 * maximum(abs, wf.N̂)
        end
    end

    # Under PaddedThreeHalves the products are exact to the band edge. A field below the Nyquist mode
    # and above the 2/3 band makes the padded and 2/3 results differ.
    Test.@testset "PaddedThreeHalves channels sum to the padded flux" begin
        pad = FIT.Types.PaddedThreeHalves()
        N = 12
        û, _ = _pf_solenoidal(N, N ÷ 2 - 1; seed = 2)
        ks = FIT.Utils.wavenumber_grid((N, N, N), (2π, 2π, 2π))
        sf = FIT.calculate_spectral_flux(û, ks; binning = b, transform = FFTB, dealiasing = pad)
        hp = FIT.calculate_helical_partial_fluxes(û, ks; binning = b, transform = FFTB, dealiasing = pad)
        sT = maximum(abs, sf.transfer_spectrum)
        Test.@test sT > 1e-6
        Test.@test maximum(abs, hp.total.transfer_spectrum .- sf.transfer_spectrum) < 1e-11 * sT
        sf23 = FIT.calculate_spectral_flux(û, ks; binning = b, transform = FFTB)
        Test.@test maximum(abs, sf23.transfer_spectrum .- sf.transfer_spectrum) > 1e-3 * sT

        # The 2-D Helmholtz channels on every batch backend, and the band-to-band transfer.
        N2 = 16
        rng = Random.MersenneTwister(12)
        û2 = cat((FFTW.fft(randn(rng, N2, N2)) ./ N2^2 for _ in 1:2)...; dims = 3)
        ks2 = FIT.Utils.wavenumber_grid((N2, N2), (2π, 2π))
        sf2 = FIT.calculate_spectral_flux(û2, ks2; binning = b, transform = FFTB, dealiasing = pad)
        s2 = maximum(abs, sf2.transfer_spectrum)
        for exec in (ComputationalBackends.SerialBackend(), ComputationalBackends.ThreadedBackend())
            pf = FIT.calculate_partial_fluxes(û2, ks2; decomposition = FIT.Types.HelmholtzDecomposition(),
                                              binning = b, transform = FFTB, dealiasing = pad, execution = exec)
            Test.@test maximum(abs, pf.total.transfer_spectrum .- sf2.transfer_spectrum) < 1e-11 * s2
            pb = FIT.calculate_partial_fluxes_batch([û2, û2], ks2; decomposition = FIT.Types.HelmholtzDecomposition(),
                                                    binning = b, transform = FFTB, dealiasing = pad, execution = exec)
            Test.@test all(maximum(abs, r.total.transfer_spectrum .- sf2.transfer_spectrum) < 1e-11 * s2 for r in pb)
        end
        # Band n's net transfer is the padded per-mode transfer weighted by the band.
        bands = FIT.Types.SmoothBands([2.0, 4.0, 6.0])
        bt = FIT.calculate_band_to_band_transfer(û2, ks2; bands = bands, transform = FFTB, dealiasing = pad)
        N̂ = FIT.NonlinearTerm.compute_nonlinear_term(û2, ks2; dealiasing = pad, transform = FFTB)
        t = FIT.Invariants.transfer_density(FIT.Types.KineticEnergy(), û2, N̂, ks2)
        W = FIT.BandTransfer._band_weights(bands.centers, bands.logwidth,
                                           FIT.ShellBinning.shell_coordinate(FIT.Types.IsotropicShells(), ks2))
        ref = [sum(W[n] .* t) for n in eachindex(W)]
        Test.@test maximum(abs, ref) > 1e-6
        Test.@test maximum(abs, bt.net_transfer .- ref) < 1e-11 * maximum(abs, ref)
    end

    # A Float32 field on Float64 wavenumber axes: the net transfer is the spectral transfer of the same
    # field, and the batch returns the same net transfer.
    Test.@testset "shell-to-shell net transfer with field and axes in different precisions" begin
        N = 16
        rng = Random.MersenneTwister(8)
        ks = FIT.Utils.wavenumber_grid((N, N), (2π, 2π))
        m = _pf_freq(N)
        ψh = FFTW.fft(randn(rng, N, N)) ./ N^2
        kx = [m[i] for i in 1:N, j in 1:N]; ky = [m[j] for i in 1:N, j in 1:N]
        û = ComplexF32.(cat(im .* ky .* ψh, -im .* kx .* ψh; dims = 3))
        r = FIT.calculate_shell_to_shell_transfer(û, ks; binning = b, transform = FFTB)
        sf = FIT.calculate_spectral_flux(û, ks; binning = b, transform = FFTB)
        Test.@test maximum(abs, sf.transfer_spectrum) > 1e-6
        Test.@test maximum(abs, r.net_transfer .- sf.transfer_spectrum) < 1f-4 * maximum(abs, sf.transfer_spectrum)
        rb = FIT.calculate_shell_to_shell_transfer_batch([û], ks; binning = b, transform = FFTB)
        Test.@test rb[1].net_transfer == r.net_transfer
    end
end
