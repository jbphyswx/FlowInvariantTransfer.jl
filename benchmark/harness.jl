# Shared measurement helpers for the FlowInvariantTransfer benchmark suite.
#
# Two numbers matter for this package and BenchmarkTools reports neither directly:
#
#   * `workspace_bytes` — the RESIDENT scratch a workspace owns. `BenchmarkTools`'s `memory` field is
#     total bytes allocated during a call, which is ~0 for a correctly reused workspace and therefore
#     says nothing about the working set. The working set is what decides whether a 256³ run fits in
#     memory, so it is measured directly by walking the workspace's fields.
#   * `bytes_per_point` — that figure divided by the grid point count, which is the size-independent
#     quantity the buffer inventory is written against.
#
# Both are deterministic: no GC, no resident-set sampling, no run-to-run noise.

using AbstractFFTs: AbstractFFTs
using BenchmarkTools: BenchmarkTools
using Printf: Printf

# Sum the bytes of every array reachable from `x` through struct fields and array elements.
# `AbstractFFTs.Plan`s are skipped: an FFTW/cuFFT plan's memory lives in the C library, is not part of
# the Julia working set this package controls, and its fields are not meaningfully walkable. Plans
# that ARE plain Julia objects (NonuniformFFTs' `PlanNUFFT`, whose oversampled grids are genuine
# resident scratch) are walked like any other struct. `seen` makes shared buffers count once, so a
# workspace that aliases one array into two fields is not double-charged.
function workspace_bytes(x, seen::Base.IdSet{Any} = Base.IdSet{Any}())
    x === nothing && return 0
    x isa AbstractFFTs.Plan && return 0
    (x isa Function || x isa Type || x isa Module) && return 0
    isbits(x) && return 0
    x in seen && return 0
    push!(seen, x)
    if x isa AbstractArray
        isbitstype(eltype(x)) && return sizeof(eltype(x)) * length(x)
        return sum(e -> workspace_bytes(e, seen), x; init = 0)
    end
    T = typeof(x)
    isstructtype(T) || return 0
    total = 0
    for i in 1:fieldcount(T)
        isdefined(x, i) || continue
        total += workspace_bytes(getfield(x, i), seen)
    end
    return total
end

bytes_per_point(x, ns) = workspace_bytes(x) / prod(ns)

# Two benchmark groups per axis:
#   HOT  — a prebuilt workspace and the `!` call; the cost a production loop pays per snapshot.
#   CONV — the allocating wrapper, which rebuilds the workspace and re-plans every transform. Work
#          moved out of the hot path and into workspace construction stays visible in this group.
const HOT = "hot"
const CONV = "convenience"

# Incompressible velocity coefficients (package convention û = fft(u)/Np).
function field_nd(ns::NTuple{D,Int}; L = 2π, seed = 1) where {D}
    ks = FIT.Utils.wavenumber_grid(ns, ntuple(_ -> L, D))
    kg = ntuple(d -> reshape(ks[d], ntuple(i -> i == d ? ns[d] : 1, D)), D)
    if D == 2
        ψh = FFTW.fft(randn(Random.MersenneTwister(seed), ns...)) ./ prod(ns)
        û = cat(im .* kg[2] .* ψh, -im .* kg[1] .* ψh; dims = 3)
    else
        # Solenoidal projection of a random vector field: û_c = a_c − k_c (k·a)/|k|².
        a = ntuple(c -> FFTW.fft(randn(Random.MersenneTwister(seed + c), ns...)) ./ prod(ns), D)
        k2 = reduce((x, y) -> x .+ y, ntuple(d -> kg[d] .^ 2, D))
        invk2 = ifelse.(k2 .> 0, inv.(k2), 0.0)
        kdota = reduce((x, y) -> x .+ y, ntuple(d -> kg[d] .* a[d], D))
        û = cat(ntuple(c -> a[c] .- kg[c] .* (kdota .* invk2), D)...; dims = D + 1)
    end
    return û, ks
end

# Real positive density field's coefficients (same û = fft(ρ)/Np convention) for compressible transfer.
function density_nd(ns::NTuple{D,Int}) where {D}
    ρ = fill(1.0, ns...)
    for I in CartesianIndices(ns)
        ρ[I] += 0.1 * cospi(2 * I[1] / ns[1]) * sinpi(2 * I[2] / ns[2])
    end
    return FFTW.fft(ρ) ./ prod(ns)
end

# Scattered points and a real velocity sampled on them, for the NUFFT axes.
function scattered_nd(np::Int, D::Int; L = 2π, seed = 5)
    rng = Random.MersenneTwister(seed)
    coords = ntuple(_ -> L .* rand(rng, np), D)
    fields = ntuple(D) do c
        x = coords[c]
        y = coords[mod1(c + 1, D)]
        @. sin(x) * cos(y)
    end
    return fields, coords
end

# Print the resident-scratch table. `rows` is a vector of (label, workspace, ns).
function report_memory(rows)
    println("\n== workspace resident scratch (deterministic; excludes C-side FFT plans) ==")
    println("  ", rpad("workspace", 42), lpad("bytes", 14), lpad("B/point", 12))
    for (label, ws, ns) in rows
        b = workspace_bytes(ws)
        println("  ", rpad(label, 42), lpad(b, 14), lpad((Printf.@sprintf "%.1f" b / prod(ns)), 12))
    end
end

function report_times(results)
    println("\n== median time / allocations ==")
    for axis in sort(collect(keys(results)))
        println("[$axis]")
        for key in sort(collect(keys(results[axis])))
            m = BenchmarkTools.median(results[axis][key])
            println("  ", rpad(key, 34),
                    lpad((Printf.@sprintf "%.3f" m.time / 1e6), 11), " ms  ",
                    lpad((Printf.@sprintf "%.1f" m.memory / 1024), 11), " KiB  (", m.allocs, " allocs)")
        end
    end
end