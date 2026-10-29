# ---------------------------------------------------------------------------
# Mode-to-mode transfer between stored coefficients ("slots").
#
# Slot `k` stands for `w(k)` modes of the full spectrum, and its slot field is
# `u_k(x) = w(k)·Re{û(k) e^{ik·x}}`. The returned `S(k|p)` satisfies `w(k)S(k|p) = ⟨u_k·(u·∇)u_p⟩`, the
# transfer between two real slot fields, which is antisymmetric for a divergence-free field. The gate
# evaluates `⟨u_k·(u·∇)u_p⟩` on the grid from the slot fields' closed forms, slot pair by slot pair, on
# the half and the full layout.
# ---------------------------------------------------------------------------

using Test: Test
using Random: Random
using FFTW: FFTW
using OhMyThreads: OhMyThreads
using FlowInvariantTransfer: FlowInvariantTransfer as FIT
using ComputationalBackends: ComputationalBackends
using SpectralBackends: SpectralBackends

# The integer frequency of index `i` on an `n`-point axis in FFT order.
_m2m_freq(i, n) = i - 1 <= (n - 1) ÷ 2 ? i - 1 : i - 1 - n

# A real divergence-free 2-D field on the `N × N` grid of the 2π box, band-limited to `|m_d| ≤ K`.
function _m2m_field(N; K = (N - 1) ÷ 3, seed = 3)
    rng = Random.MersenneTwister(seed)
    m = [_m2m_freq(i, N) for i in 1:N]
    ψ̂ = [abs(m[i]) <= K && abs(m[j]) <= K ? randn(rng, ComplexF64) : 0.0im for i in 1:N, j in 1:N]
    ψh = FFTW.fft(real.(FFTW.bfft(ψ̂))) ./ N^2
    kx = [m[i] for i in 1:N, j in 1:N]; ky = [m[j] for i in 1:N, j in 1:N]
    return real.(FFTW.bfft(im .* ky .* ψh)), real.(FFTW.bfft(-im .* kx .* ψh))
end

# `P[a, b] = ⟨u_k·(u·∇)u_p⟩` over the stored slots `k = slots[a]`, `p = slots[b]`, the slot fields and their
# gradients evaluated on the grid in closed form. `f1` maps an index of axis 1 to its integer frequency.
function _m2m_pair_transfer(u, v, û, w, f1)
    N = size(u, 1)
    x = [2π * (i - 1) / N for i in 1:N]
    kd(mm) = (iseven(N) && abs(mm) == N ÷ 2) ? 0 : mm          # the grid derivative's wavenumber
    slots = vec(CartesianIndices(size(w)))
    fields = map(slots) do I
        m1 = f1(I[1]); m2 = _m2m_freq(I[2], N)
        e = [cis(m1 * xi + m2 * yj) for xi in x, yj in x]
        A = (w[I] * û[I, 1], w[I] * û[I, 2])
        (map(a -> real.(a .* e), A),
         map(a -> real.(im * kd(m1) * a .* e), A),
         map(a -> real.(im * kd(m2) * a .* e), A))
    end
    P = zeros(length(slots), length(slots))
    for a in eachindex(slots), b in eachindex(slots)
        uk = fields[a][1]; dxp = fields[b][2]; dyp = fields[b][3]
        P[a, b] = sum(uk[1] .* (u .* dxp[1] .+ v .* dyp[1]) .+ uk[2] .* (u .* dxp[2] .+ v .* dyp[2])) / N^2
    end
    return P
end

Test.@testset "mode-to-mode slot transfer" begin
    FFTB = SpectralBackends.FFTSpectralBackend()

    Test.@testset "w(k)S(k|p) = ⟨u_k·(u·∇)u_p⟩, $(half ? "half" : "full") layout, N = $N" for
            N in (8, 9), half in (true, false)
        u, v = _m2m_field(N)
        ks = FIT.Utils.wavenumber_grid(half ? Float64 : ComplexF64, (N, N), (2π, 2π))
        tr = half ? FFTW.rfft : FFTW.fft
        û = cat(tr(u), tr(v); dims = 3) ./ N^2
        r = FIT.calculate_mode_to_mode_transfer(û, ks; transform = FFTB)
        w = r.weights
        ms = size(r.net_transfer)
        # The fold weight: on the half layout 1 on the k₁ = 0 plane and, for even N, the Nyquist plane,
        # and 2 elsewhere; 1 everywhere on the full layout.
        Test.@test all(w[i, j] == ((!half || i == 1 || (iseven(N) && i == N ÷ 2 + 1)) ? 1 : 2)
                       for i in 1:ms[1], j in 1:ms[2])

        P = _m2m_pair_transfer(u, v, û, w, half ? (i -> i - 1) : (i -> _m2m_freq(i, N)))
        slots = vec(CartesianIndices(ms))
        WS = [w[k] * r.transfer[k, p] for k in slots, p in slots]
        scale = maximum(abs, P)
        Test.@test scale > 1e-6                                       # the reference carries signal
        Test.@test maximum(abs, WS .- P) < 1e-12 * scale

        # The slot-to-slot transfer is antisymmetric and conserves.
        Test.@test maximum(abs, WS .+ transpose(WS)) < 1e-12 * scale
        Test.@test abs(sum(WS)) < 1e-12 * sum(abs, WS)
    end

    # Under PaddedThreeHalves every giver's term is formed on the padded grid; summed over the givers it
    # is the per-mode transfer of the padded nonlinear term, computed here in one evaluation.
    Test.@testset "PaddedThreeHalves reduces to the padded transfer — $nm" for
            (nm, exec) in (("serial", ComputationalBackends.SerialBackend()),
                           ("threaded", ComputationalBackends.ThreadedBackend()))
        N = 8
        rng = Random.MersenneTwister(5)
        u = randn(rng, N, N); v = randn(rng, N, N)              # broadband, Nyquist included
        ksh = FIT.Utils.wavenumber_grid(Float64, (N, N), (2π, 2π))
        ûh = cat(FFTW.rfft(u), FFTW.rfft(v); dims = 3) ./ N^2
        pad = FIT.Types.PaddedThreeHalves()
        r = FIT.calculate_mode_to_mode_transfer(ûh, ksh; transform = FFTB, dealiasing = pad, execution = exec)
        N̂ = FIT.NonlinearTerm.compute_nonlinear_term(ûh, ksh; dealiasing = pad, transform = FFTB)
        t = FIT.Invariants.transfer_density(FIT.Types.KineticEnergy(), ûh, N̂, ksh)
        Test.@test maximum(abs, t) > 1e-6
        Test.@test maximum(abs, r.net_transfer .- t) < 1e-12 * maximum(abs, t)
        # 2/3 discards this field's outer modes, so the padded result is a different one.
        r23 = FIT.calculate_mode_to_mode_transfer(ûh, ksh; transform = FFTB, execution = exec)
        Test.@test maximum(abs, r23.net_transfer .- t) > 1e-3 * maximum(abs, t)
    end
end
