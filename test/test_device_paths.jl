# ---------------------------------------------------------------------------
# Device paths on a genuine device-array type.
#
# `GPUBackend(KA.CPU())` is a host `Array` behind a backend tag: it exercises the dispatch while
# scalar indexing stays legal, so a path that scalar-indexes still passes there. These tests run on
# JLArrays under `allowscalar(false)`, where such a path raises.
#
# JLArrays runs kernels to completion at launch and defines no `KA.synchronize`; the method below
# supplies the one the device methods call.
# ---------------------------------------------------------------------------

using Test: Test
using Random: Random
using LinearAlgebra: LinearAlgebra
using FFTW: FFTW                     # the KA drivers run their nonlinear term through the FFT backend
using JLArrays: JLArrays
using KernelAbstractions: KernelAbstractions as KA
using FlowInvariantTransfer: FlowInvariantTransfer as FIT
using ComputationalBackends: ComputationalBackends
using SpectralBackends: SpectralBackends

KA.synchronize(::JLArrays.JLBackend) = nothing

const _DEV_JL = ComputationalBackends.GPUBackend(JLArrays.JLBackend())
const _DEV_CPU = ComputationalBackends.GPUBackend(KA.CPU())
const _SER_D = ComputationalBackends.SerialBackend()
const _FFTB_D = SpectralBackends.FFTSpectralBackend()

Test.@testset "device paths (JLArrays, allowscalar(false))" begin
    JLArrays.allowscalar(false)

    Test.@testset "real-NUFFT Hermitian expansion — $(ms)" for ms in ((8, 6), (9, 6), (8, 7), (6, 4, 4), (5, 6, 7))
        # The expansion is a pure function of arrays and numbers, so the host loop and the device
        # kernel can be compared directly, with no NUFFT plan in the way.
        D = length(ms); CT = ComplexF64
        Random.seed!(7 + D)
        novs = ntuple(d -> 2 * ms[d], D)
        fk = randn(CT, ms[1] ÷ 2 + 1, ms[2:end]...)
        us = randn(CT, novs[1] ÷ 2 + 1, novs[2:end]...)
        gk = ntuple(d -> rand(novs[d]) .+ 0.5, D)
        nf = prod(ntuple(d -> 2π / novs[d], D)); invN = 0.125

        host = zeros(CT, ms...)
        FIT._r2c_expand!(_SER_D, host, fk, us, gk, ms, novs, nf, invN)
        cpu = zeros(CT, ms...)
        FIT._r2c_expand!(_DEV_CPU, cpu, fk, us, gk, ms, novs, nf, invN)
        Test.@test cpu == host
        dev = JLArrays.JLArray(zeros(CT, ms...))
        FIT._r2c_expand!(_DEV_JL, dev, JLArrays.JLArray(fk), JLArrays.JLArray(us),
                         map(JLArrays.JLArray, gk), ms, novs, nf, invN)
        Test.@test Array(dev) == host
    end

    Test.@testset "KA drivers reproduce the serial result" begin
        N = 8; ns = (N, N)
        Random.seed!(9)
        ks = FIT.Utils.wavenumber_grid(ns, (2π, 2π))
        û = randn(ComplexF64, ns..., 2)
        binning = FIT.Types.LinearBinning(1.0)

        fh = FIT.calculate_spectral_flux(û, ks; binning = binning, spectral = _FFTB_D)
        fd = FIT.calculate_spectral_flux(û, ks; binning = binning, spectral = _FFTB_D, execution = _DEV_CPU)
        Test.@test maximum(abs, fh.transfer_spectrum) > 1e-8
        Test.@test fd.transfer_spectrum ≈ fh.transfer_spectrum rtol = 1e-12

        sh = FIT.calculate_shell_to_shell_transfer(û, ks; binning = binning, spectral = _FFTB_D)
        sd = FIT.calculate_shell_to_shell_transfer(û, ks; binning = binning, spectral = _FFTB_D,
                                                   execution = _DEV_CPU)
        Test.@test maximum(abs, sh.transfer_matrix) > 1e-8
        Test.@test sd.transfer_matrix ≈ sh.transfer_matrix rtol = 1e-12

        bands = FIT.Types.SmoothBands([2.0, 4.0])
        bh = FIT.BandTransfer.calculate_band_to_band_transfer(û, ks; bands = bands, spectral = _FFTB_D)
        bd = FIT.BandTransfer.calculate_band_to_band_transfer(û, ks; bands = bands, spectral = _FFTB_D,
                                                              execution = _DEV_CPU)
        Test.@test maximum(abs, bh.transfer_matrix) > 1e-8
        Test.@test bd.transfer_matrix ≈ bh.transfer_matrix rtol = 1e-12

        # The one-hot giver kernel replaced a per-giver mask grid; S(k|p) must be unchanged.
        mh = FIT.calculate_mode_to_mode_transfer(û, ks; spectral = _FFTB_D)
        md = FIT.calculate_mode_to_mode_transfer(û, ks; spectral = _FFTB_D, execution = _DEV_CPU)
        Test.@test maximum(abs, mh.transfer) > 1e-8
        Test.@test md.transfer ≈ mh.transfer rtol = 1e-12
    end

    Test.@testset "workspace shell index follows the field's array kind" begin
        ns = (8, 8)
        ks = FIT.Utils.wavenumber_grid(ns, (2π, 2π))
        Random.seed!(11)
        û = randn(ComplexF64, ns..., 2)
        ws_host = FIT.Workspaces.ShellToShellWorkspace(û, ks, FIT.Types.LinearBinning(1.0))
        Test.@test ws_host.shell_idx isa Array{Int}
        ws_dev = FIT.Workspaces.ShellToShellWorkspace(JLArrays.JLArray(û), ks, FIT.Types.LinearBinning(1.0))
        Test.@test ws_dev.shell_idx isa JLArrays.JLArray
        Test.@test Array(ws_dev.shell_idx) == ws_host.shell_idx
    end
end
