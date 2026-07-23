module Backends

# ---------------------------------------------------------------------------
# Execution-backend taxonomy — two orthogonal concerns that COMPOSE:
#
#   LOCAL compute backend  — what one process/rank computes on:
#         SerialBackend, ThreadedBackend (OhMyThreads ext), GPUBackend{B} (KernelAbstractions ext).
#   DISTRIBUTION wrapper    — how work is split across processes, PARAMETRIC over the inner local
#         backend:  DistributedBackend{Inner} (Distributed ext), MPIBackend{Inner,C} (MPI ext).
#
# Composition expresses real HPC layouts:
#         MPIBackend(ThreadedBackend())     — hybrid MPI + threads,
#         MPIBackend(GPUBackend(dev))       — multi-GPU cluster,
#         DistributedBackend(ThreadedBackend()) — multithreaded workers.
#
# `AutoBackend` resolves to the best available LOCAL backend. Heavy implementations live in
# extensions; this module defines only the dispatch types + helpers, so the core carries no
# parallel/GPU dependency. Backend TYPES use the `…Backend` suffix so they never collide with the
# packages the extensions load (the stdlib `Distributed`, `MPI.jl`, …). Orthogonal to the transform
# axis (`Types.AbstractSpectralBackend`): a computation picks one of each.
# ---------------------------------------------------------------------------

export AbstractExecutionBackend, SerialBackend, ThreadedBackend, GPUBackend, AutoBackend
export DistributedBackend, MPIBackend
export local_backend, is_distributed, resolve_execution, is_gpu_array

"""
    AbstractExecutionBackend

Supertype for all execution backends — local compute backends ([`SerialBackend`](@ref),
[`ThreadedBackend`](@ref), [`GPUBackend`](@ref)) and distribution wrappers
([`DistributedBackend`](@ref), [`MPIBackend`](@ref)). Orthogonal to the transform axis
(`AbstractSpectralBackend`): how the outer work (shell/mode loops and reductions) is parallelised,
independent of which physical↔spectral transform is used.
"""
abstract type AbstractExecutionBackend end

"""
    SerialBackend <: AbstractExecutionBackend

Single-threaded serial execution; no external dependencies. The reference every other backend is
validated against.
"""
struct SerialBackend <: AbstractExecutionBackend end

"""
    ThreadedBackend <: AbstractExecutionBackend

Shared-memory multithreading over the dominant (shell/mode/band/triad) loop. Requires
`using OhMyThreads`.
"""
struct ThreadedBackend <: AbstractExecutionBackend end

"""
    GPUBackend{B} <: AbstractExecutionBackend

GPU execution on the KernelAbstractions backend object `B` (e.g. `GPUBackend(CUDA.CUDABackend())`,
or `GPUBackend(KernelAbstractions.CPU())` to validate device-genericity without hardware). Requires
`using KernelAbstractions` + a vendor package (for a real device) or a device-array provider.
"""
struct GPUBackend{B} <: AbstractExecutionBackend
    backend::B
end

"""
    DistributedBackend{Inner} <: AbstractExecutionBackend
    DistributedBackend(inner = SerialBackend())

Single-node multi-process execution via `Distributed`/`SharedArrays`, each worker running `inner`
locally. Parametric over the inner local backend, so hybrids compose:
`DistributedBackend(ThreadedBackend())` runs multithreaded workers
(`addprocs(n; exeflags="-t k")` + `@everywhere using OhMyThreads`). Retrieve the per-worker backend
with [`local_backend`](@ref). Requires `using Distributed, SharedArrays`.
"""
struct DistributedBackend{Inner<:AbstractExecutionBackend} <: AbstractExecutionBackend
    inner::Inner
end
DistributedBackend() = DistributedBackend(SerialBackend())

"""
    MPIBackend{Inner,C} <: AbstractExecutionBackend
    MPIBackend(inner = SerialBackend(); comm = nothing)

Multi-node (distributed-memory) execution via `MPI`, each rank running `inner` locally. Parametric
over the inner local backend (like [`DistributedBackend`](@ref)), so `MPIBackend(ThreadedBackend())`
is hybrid MPI+threads and `MPIBackend(GPUBackend(dev))` targets a multi-GPU cluster. Used for the
single-grid rank/pencil split and the batch axis; requires `using MPI` and running under `mpiexec`.

`comm = nothing` lets the MPI extension substitute `MPI.COMM_WORLD` (the core cannot name `MPI`).
"""
struct MPIBackend{Inner<:AbstractExecutionBackend, C} <: AbstractExecutionBackend
    inner::Inner
    comm::C
end
MPIBackend(inner::AbstractExecutionBackend = SerialBackend(); comm = nothing) = MPIBackend(inner, comm)

"""
    AutoBackend <: AbstractExecutionBackend

Select the best available LOCAL backend at call time (see [`resolve_execution`](@ref)): resolves to
[`ThreadedBackend`](@ref) when the OhMyThreads extension is loaded and Julia was started with more
than one thread, otherwise [`SerialBackend`](@ref). [`GPUBackend`](@ref) (needs a device) and the
distribution wrappers (need a worker pool / MPI runtime) are never chosen automatically — pass them
explicitly.
"""
struct AutoBackend <: AbstractExecutionBackend end

"""
    local_backend(execution::AbstractExecutionBackend) -> AbstractExecutionBackend

The per-process compute backend: the wrapped `inner` for a distribution wrapper
([`DistributedBackend`](@ref)/[`MPIBackend`](@ref)), else the backend itself (identity).
"""
local_backend(execution::AbstractExecutionBackend) = execution
local_backend(execution::DistributedBackend) = execution.inner
local_backend(execution::MPIBackend) = execution.inner

"""
    is_distributed(execution::AbstractExecutionBackend) -> Bool

`true` if `execution` distributes work across processes ([`DistributedBackend`](@ref)/[`MPIBackend`](@ref)).
"""
is_distributed(::AbstractExecutionBackend) = false
is_distributed(::DistributedBackend) = true
is_distributed(::MPIBackend) = true

"""
    is_gpu_array(x) -> Bool

`true` if `x` is a GPU/device array (a `GPUArraysCore.AbstractGPUArray` — `CuArray`, `JLArray`,
`ROCArray`, …). Defaults to `false` for every host array (`Array`, and host wrappers like
`SubArray`/`StaticArray`/`FixedSizeArray`/`OffsetArray` that are NOT `Base.Array`); the
GPUArraysCore extension adds the `AbstractGPUArray` method. Used to route device inputs correctly
(e.g. reject the host-only DirectSum reference / the host-array-under-GPUBackend case) without the
fragile `!(x isa Array)` test, which would misclassify host non-`Array` types as device arrays.
"""
is_gpu_array(::Any) = false

"""
    resolve_execution(execution::AbstractExecutionBackend) -> AbstractExecutionBackend

Map an execution backend to the concrete backend to actually run on. Concrete backends pass through
unchanged; [`AutoBackend`](@ref) resolves to [`ThreadedBackend`](@ref) when threading is available
(the OhMyThreads extension is loaded and `Threads.nthreads() > 1`), else [`SerialBackend`](@ref).
Called at every diagnostic entry point that accepts `execution`, so `AutoBackend` works uniformly.
"""
resolve_execution(execution::AbstractExecutionBackend) = execution
function resolve_execution(::AutoBackend)
    if Threads.nthreads() > 1 &&
       Base.get_extension(parentmodule(@__MODULE__), :FlowInvariantTransferOhMyThreadsExt) !== nothing
        return ThreadedBackend()
    end
    return SerialBackend()
end

end # module Backends
