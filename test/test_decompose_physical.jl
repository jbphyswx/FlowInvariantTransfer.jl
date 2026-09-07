# ---------------------------------------------------------------------------
# Physical → decomposed → physical in one call, for the decompositions defined per mode.
#
# The gate is the identity that defines a two-way split: the components sum to the field. It holds to
# round-off for a field band-limited below the Nyquist mode of each even axis — that mode is its own
# image under `k ↦ −k` and fixes no orientation for the Craya–Herring frame. The reconstruction is
# asserted to break there too, which keeps the band limit a stated precondition of the test.
# ---------------------------------------------------------------------------

using Test: Test
using Random: Random
using FFTW: FFTW
using FlowGeometries: FlowGeometries as FG
using FlowFieldSpectra: FlowFieldSpectra
using FlowInvariantTransfer: FlowInvariantTransfer as FIT

# Divergence-free 3-D field from the curl of a random vector potential, band-limited at `kcut`.
function _solenoidal(N, kcut)
    Random.seed!(5)
    kx = [i <= N ÷ 2 ? i - 1 : i - 1 - N for i in 1:N, _ in 1:N, _ in 1:N] .* 1.0
    ky = [j <= N ÷ 2 ? j - 1 : j - 1 - N for _ in 1:N, j in 1:N, _ in 1:N] .* 1.0
    kz = [k <= N ÷ 2 ? k - 1 : k - 1 - N for _ in 1:N, _ in 1:N, k in 1:N] .* 1.0
    keep = (abs.(kx) .<= kcut) .& (abs.(ky) .<= kcut) .& (abs.(kz) .<= kcut)
    A = [randn(ComplexF64, N, N, N) .* keep for _ in 1:3]
    return (real.(FFTW.bfft(im .* (ky .* A[3] .- kz .* A[2]))),
            real.(FFTW.bfft(im .* (kz .* A[1] .- kx .* A[3]))),
            real.(FFTW.bfft(im .* (kx .* A[2] .- ky .* A[1]))))
end

Test.@testset "physical decomposition — one-shot resynthesis" begin
    N = 16; L = 2π
    x = range(0, L; length = N + 1)[1:N]
    grid = FG.Grids.StructuredGrid(FG.Geometry.CartesianGeometry{Float64}(), x, x, x;
                                   topology = (true, true, true))
    cases = (("helical", FIT.Types.HelicalDecomposition(), (:positive, :negative)),
             ("toroidal/poloidal", FIT.Types.ToroidalPoloidalDecomposition(), (:toroidal, :poloidal)))

    fields = _solenoidal(N, N ÷ 2 - 1)              # band-limited below the Nyquist mode
    scale = maximum(abs, fields[1])
    Test.@test scale > 1e-6                          # the field carries signal

    for (nm, decomp, names) in cases
        r = FIT.Decomposition.decompose_field(decomp, fields, grid)
        Test.@test keys(r) == names
        a = r[names[1]]; b = r[names[2]]
        Test.@test length(a) == 3 && length(b) == 3
        Test.@test all(eltype(c) <: Real for c in a)             # real components of a real field
        Test.@test all(size(c) == (N, N, N) for c in a)
        # Both parts carry signal, so the split is doing work.
        Test.@test maximum(abs, a[1]) > 1e-8
        Test.@test maximum(abs, b[1]) > 1e-8
        for (c, f) in enumerate(fields)
            Test.@test maximum(abs, a[c] .+ b[c] .- f) < 1e-10 * scale
        end
    end

    # The precondition is real: with content at the Nyquist mode the frame has no orientation there and
    # the sum of the two parts departs from the field.
    nyq = _solenoidal(N, N ÷ 2)
    rn = FIT.Decomposition.decompose_field(FIT.Types.HelicalDecomposition(), nyq, grid)
    Test.@test maximum(abs, rn.positive[1] .+ rn.negative[1] .- nyq[1]) > 1e-3 * maximum(abs, nyq[1])
end
