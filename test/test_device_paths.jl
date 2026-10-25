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
using FFTW: FFTW
using JLArrays: JLArrays
using KernelAbstractions: KernelAbstractions as KA
using FlowInvariantTransfer: FlowInvariantTransfer as FIT
using ComputationalBackends: ComputationalBackends

KA.synchronize(::JLArrays.JLBackend) = nothing

# An FFT for JLArrays, which has none: a plan holds FFTW's plan of a host copy, and an execution moves
# its arrays across in bulk.
struct _HostBounceFFT{P}
    inner::P
end
FFTW.AbstractFFTs.plan_fft(x::JLArrays.JLArray, region; kwargs...) = _HostBounceFFT(FFTW.plan_fft(Array(x), region))
FFTW.AbstractFFTs.plan_bfft(x::JLArrays.JLArray, region; kwargs...) = _HostBounceFFT(FFTW.plan_bfft(Array(x), region))
Base.:*(p::_HostBounceFFT, x::JLArrays.JLArray) = JLArrays.JLArray(p.inner * Array(x))

const _DEV_JL = ComputationalBackends.GPUBackend(JLArrays.JLBackend())
const _DEV_CPU = ComputationalBackends.GPUBackend(KA.CPU())
const _SER_D = ComputationalBackends.SerialBackend()

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

    Test.@testset "host inputs staged into device memory" begin
        Random.seed!(13)
        x = randn(ComplexF64, 6, 5, 2); y = randn(ComplexF64, 6, 5, 2)
        buf = FIT.Types._staging_buffer(_DEV_JL, x)
        Test.@test buf isa JLArrays.JLArray
        Test.@test Array(buf) == x
        Test.@test Array(FIT.Types._staged!(buf, y)) == y
        Test.@test Array(buf) == y
        d = JLArrays.JLArray(x)
        Test.@test FIT.Types._staged!(buf, d) === d
    end

    Test.@testset "workspace shell index follows the field's array kind" begin
        ns = (8, 8)
        ks = FIT.Utils.wavenumber_grid(ns, (2π, 2π))
        Random.seed!(11)
        û = randn(ComplexF64, ns..., 2)
        ws_host = FIT.Workspaces.ShellToShellWorkspace(û, ks, FIT.Types.LinearBinning(1.0))
        ws_dev = FIT.Workspaces.ShellToShellWorkspace(JLArrays.JLArray(û), ks, FIT.Types.LinearBinning(1.0))
        Test.@test Array(ws_dev.shell_idx) == ws_host.shell_idx
    end
end
