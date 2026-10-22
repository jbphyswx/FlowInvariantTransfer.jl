# cuFINUFFT device `to_spectral!` against the host FINUFFT path, on an NVIDIA GPU.
#
#   julia --project=gpu gpu/cufinufft.jl

using Test: Test
using CUDA: CUDA
using FINUFFT: FINUFFT
using Random: Random
using ComputationalBackends: ComputationalBackends as CB
using FlowTransformBindings: FlowTransformBindings as FTB
using FlowInvariantTransfer: FlowInvariantTransfer as FIT

CUDA.functional() || error("cufinufft.jl needs a functional CUDA GPU")

Test.@testset "cuFINUFFT device to_spectral equals the host FINUFFT path" begin
    Nn = 8; Ln = 2π
    xs1 = [(i - 1) * Ln / Nn for i in 1:Nn]
    Xg = [xs1[i] for i in 1:Nn, j in 1:Nn]; Yg = [xs1[j] for i in 1:Nn, j in 1:Nn]
    Random.seed!(7); ug = randn(Nn, Nn); vg = randn(Nn, Nn)
    û_host, _ = FIT.to_spectral((vec(ug), vec(vg)), (vec(Xg), vec(Yg)), (Nn, Nn);
                                transform = FTB.FINUFFTBackend(), Ls = (Ln, Ln))
    wsd = FIT.NUFFTToSpectralWorkspace((vec(Xg), vec(Yg)), (Nn, Nn); transform = FTB.FINUFFTBackend(),
                                       ncomponents = 2, Ls = (Ln, Ln),
                                       execution = CB.GPUBackend(CUDA.CUDABackend()))
    û_dev, _ = FIT.to_spectral!(wsd, (CUDA.CuArray(vec(ug)), CUDA.CuArray(vec(vg))))
    Test.@test isapprox(Array(û_dev), û_host; atol = 1e-6)
end
