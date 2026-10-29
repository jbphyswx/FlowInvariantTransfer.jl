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
