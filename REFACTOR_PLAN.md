# FlowInvariantTransfer.jl — Performance Refactor Plan

## Guiding principles

1. **Parametric on array type, not element type.** Every struct that holds arrays carries
   a type parameter for the array itself (e.g. `A<:AbstractArray`). The element type `FT`
   is recovered as `eltype(A)` when needed internally. Never write `Vector{FT}` or
   `Matrix{FT}` in a struct field.

2. **No `AbstractFloat` or other element-type bounds on struct type parameters.**
   `{FT<:AbstractFloat, V<:AbstractVector{FT}}` breaks AD (ForwardDiff.Dual), Unitful,
   Measurements.jl, etc. Use `{V<:AbstractVector}` — no element bound. The compiler
   specialises on the concrete type; the bound adds nothing but restriction.

3. **No concrete type annotations on public function arguments.** Annotating
   `velocity_hat::AbstractArray{<:Complex}` is already too restrictive — it excludes
   custom array types whose eltype isn't a plain Complex. Unannotated args let the
   compiler fully specialise. Annotate only when needed for dispatch disambiguation.

4. **`!`-first design.** Every public function that writes output has a mutating `!`-variant
   that takes all output buffers as the first argument(s) and allocates nothing. The
   allocating convenience wrapper constructs those buffers and calls the `!`-variant.

5. **Workspace structs carry all temporaries.** Any array created inside a hot loop
   belongs in a workspace struct, not allocated on every call. Workspace constructors are
   provided for convenience. Workspace structs are parametric on their array types, with
   no element-type bounds.

6. **Single shell-index array, not Vector{BitArray}.** Replace per-shell `BitArray` masks
   with one integer array `shell_idx` (same shape as `k_mag`) where `shell_idx[I] = n`
   if mode `I` belongs to shell `n`, and `0` otherwise. One allocation, cache-friendly.

7. **No `Float64` hardcoding.** `shell_edges`, `shell_centers`, `wavenumber_grid`,
   `wavenumber_magnitude_grid` must propagate the numeric type from their inputs.
   Use `similar`, `eltype`, `typeof` — never spell out `Float64` internally.

---

## Step 1 — Parametric result structs  (`src/Types.jl`)

### Current (wrong)
```julia
struct SpectralFluxResult{FT<:AbstractFloat}
    k_shells::Vector{FT}
    transfer_spectrum::Vector{FT}
    flux::Vector{FT}
end

struct ShellToShellResult{FT<:AbstractFloat}
    shell_centers::Vector{FT}
    shell_edges::Vector{FT}
    transfer_matrix::Matrix{FT}
    net_transfer::Vector{FT}
    max_antisymmetry_error::FT
end

struct CoarseGrainingFluxResult{FT<:AbstractFloat, N}
    filter_scale::FT
    flux_field::Array{FT, N}
    mean_flux::FT
    stress_tensor::Union{Nothing, Array{FT}}
    strain_rate::Union{Nothing, Array{FT}}
end
```

### Target (correct)
```julia
# No element-type bound: works with Float32, Float64, Dual, Measurement, etc.
struct SpectralFluxResult{V<:AbstractVector, M<:AbstractVector}  # M unused, shown for pattern
    k_shells::V
    transfer_spectrum::V
    flux::V
end
SpectralFluxResult(k, T, f) = SpectralFluxResult{typeof(k)}(k, T, f)

struct ShellToShellResult{V<:AbstractVector, M<:AbstractMatrix}
    shell_centers::V
    shell_edges::V
    transfer_matrix::M
    net_transfer::V
    max_antisymmetry_error::eltype(V)  # scalar: use eltype, not a separate FT param
end
# convenience constructor — types inferred from arguments
ShellToShellResult(c, e, T, n, a) =
    ShellToShellResult{typeof(c), typeof(T)}(c, e, T, n, a)

# No diagnostics — returned by default
struct CoarseGrainingFluxResult{S, A<:AbstractArray}
    filter_scale::S   # scalar, unconstrained — could be Unitful, Dual, etc.
    flux_field::A
    mean_flux::S
end

# With diagnostics — separate type, no Union{Nothing,...}
struct CoarseGrainingFluxResultWithDiagnostics{S, A<:AbstractArray}
    filter_scale::S
    flux_field::A
    mean_flux::S
    stress_tensor::A
    strain_rate::A
end
```

**Note on `max_antisymmetry_error`:** it is a scalar derived from the matrix elements.
Do not give it a separate type parameter — use `eltype(transfer_matrix)` in code that
reads it. In the struct definition it must still be a concrete field type; the simplest
correct approach is to give it its own unconstrained parameter `E` or fold it into `V`
via `eltype`. Final call during implementation: likely just `{V, M}` with the scalar
field typed as `eltype(M)` enforced only via the constructor, not the struct bound.

---

## Step 2 — Parametric workspace structs  (new file `src/Workspaces.jl`)

All array fields parametric on the array type, not the element type.

```julia
# Nonlinear term workspace — holds physical-space temporaries
struct NonlinearTermWorkspace{CA<:AbstractArray{<:Complex},
                               RA<:AbstractArray{<:Real}}
    u_phys::RA        # size (ns..., D) — physical velocity components
    grad_phys::RA     # size (ns..., D, D) — ∂u_i/∂x_j
    N_phys::RA        # size (ns..., D) — nonlinear term in physical space
    N̂::CA             # size (ns..., D) — spectral output buffer
end

function NonlinearTermWorkspace(velocity_hat::AbstractArray{Complex{FT}},
                                ks::Tuple) where {FT}
    ns = size(velocity_hat)[1:length(ks)]
    D  = size(velocity_hat, ndims(velocity_hat))
    u_phys    = Array{FT}(undef, ns..., D)
    grad_phys = Array{FT}(undef, ns..., D, length(ks))
    N_phys    = Array{FT}(undef, ns..., D)
    N̂         = similar(velocity_hat)
    return NonlinearTermWorkspace(u_phys, grad_phys, N_phys, N̂)
end

# SpectralFlux workspace
struct SpectralFluxWorkspace{NW<:NonlinearTermWorkspace,
                              V<:AbstractVector}
    nonlinear::NW
    T_spec::V
    flux::V
end

function SpectralFluxWorkspace(velocity_hat, ks, binning)
    edges  = shell_edges(binning, maximum(wavenumber_magnitude_grid(ks)))
    N_sh   = length(edges) - 1
    FT     = real(eltype(velocity_hat))
    T_spec = Vector{FT}(undef, N_sh)       # ← OK here: workspace alloc, not result
    flux   = Vector{FT}(undef, N_sh)
    return SpectralFluxWorkspace(NonlinearTermWorkspace(velocity_hat, ks), T_spec, flux)
end

# ShellToShell workspace
struct ShellToShellWorkspace{NW<:NonlinearTermWorkspace,
                              CA<:AbstractArray{<:Complex},
                              M<:AbstractMatrix,
                              V<:AbstractVector}
    nonlinear::NW     # for computing N̂_m
    û_m::CA           # band-filtered mediator buffer (reused each m)
    T_mat::M
    net_transfer::V
end
```

Note: `Vector{FT}` inside workspace constructors is *acceptable* — the workspace is the
one-time allocation. The point is that workspace fields use abstract types so GPU arrays
etc. can be passed in.

---

## Step 3 — Single shell-index array  (`src/ShellBinning.jl`)

Replace `shell_mask(k_mag, edges, n)` + vector-of-masks pattern with:

```julia
"""
    assign_shells(k_mag, edges) -> AbstractArray{Int}

Return an integer array (same shape as `k_mag`) where entry `[I] = n` if
`edges[n] <= k_mag[I] < edges[n+1]`, and `0` if the mode falls outside all shells.
Single allocation; replaces `[shell_mask(k_mag, edges, n) for n in 1:N_sh]`.
"""
function assign_shells(k_mag::AbstractArray, edges::AbstractVector)
    idx = similar(k_mag, Int)
    fill!(idx, 0)
    N_sh = length(edges) - 1
    for I in CartesianIndices(k_mag)
        k = k_mag[I]
        for n in 1:N_sh
            if edges[n] <= k < edges[n+1]
                idx[I] = n
                break
            end
        end
    end
    return idx
end
```

All inner loops then become `shell_idx[I] == n` rather than `masks[n][I]`.

---

## Step 4 — Remove `Float64` hardcoding  (`src/ShellBinning.jl`, `src/Utils.jl`)

### `ShellBinning`
- `shell_edges(b, k_max::Real)` — infer FT from `typeof(k_max)` or add explicit `FT` argument
- `shell_centers` — same
- `assign_shells` — FT propagated from `k_mag` element type

### `Utils`
- `wavenumber_grid(ns, Ls)` — infer FT from `eltype(Ls)`, currently always `Float64`
- `wavenumber_magnitude_grid(ks)` — propagate FT from `eltype(ks[1])`

Concrete change in `wavenumber_grid`:
```julia
# current
ks = zeros(Float64, N)
# target
FT = eltype(Ls[1])   # or promote_type(eltype.(Ls)...)
ks = zeros(FT, N)
```

---

## Step 5 — In-place `!`-variants for public API  (`src/SpectralFlux.jl`, `src/ShellToShellTransfer.jl`)

```julia
# SpectralFlux
function calculate_spectral_flux!(result::SpectralFluxResult,
                                  ws::SpectralFluxWorkspace,
                                  velocity_hat, ks; kwargs...)
    # zero allocations — writes into result.T_spec, result.flux, ws.nonlinear.N̂ etc.
end

function calculate_spectral_flux(velocity_hat, ks; binning=..., kwargs...)
    ws     = SpectralFluxWorkspace(velocity_hat, ks, binning)
    edges  = shell_edges(binning, ...)
    result = SpectralFluxResult(shell_centers(...), similar(ws.T_spec), similar(ws.flux))
    calculate_spectral_flux!(result, ws, velocity_hat, ks; kwargs...)
    return result
end

# ShellToShell — same pattern
function calculate_shell_to_shell_transfer!(result, ws, velocity_hat, ks; kwargs...)
    ...
end
function calculate_shell_to_shell_transfer(velocity_hat, ks; kwargs...)
    ...
end
```

---

## Step 6 — Fix internal temporaries in direct-sum paths

`_compute_nonlinear_term_direct!` currently allocates `u_phys` and `grad_phys` every call.
Move these into `NonlinearTermWorkspace` and thread it through:

```julia
function _compute_nonlinear_term_direct!(N̂, velocity_hat, ks, ws::NonlinearTermWorkspace; dealiasing)
    # uses ws.u_phys, ws.grad_phys, ws.N_phys — no heap allocation in hot path
end
```

The FFTW ext overrides similarly receive a workspace and use pre-planned transforms.

---

## File change summary

| File | Changes |
|------|---------|
| `src/Types.jl` | Parametric result structs (Step 1) |
| `src/Workspaces.jl` | New file: all workspace structs + constructors (Step 2) |
| `src/ShellBinning.jl` | `assign_shells`, remove Float64 hardcoding (Steps 3, 4) |
| `src/Utils.jl` | Parametric FT in `wavenumber_grid`, `wavenumber_magnitude_grid` (Step 4) |
| `src/NonlinearTerm.jl` | Thread workspace through, remove internal allocs (Step 6) |
| `src/SpectralFlux.jl` | `!`-variant, use workspace + `assign_shells` (Step 5) |
| `src/ShellToShellTransfer.jl` | `!`-variant, use workspace + `assign_shells` (Step 5) |
| `src/FlowInvariantTransfer.jl` | Export new workspace types and `!`-variants |
| `ext/FlowInvariantTransferFFTWExt.jl` | Update FFTW path to accept workspace |
| `test/runtests.jl` | Update tests for new API (both `!` and allocating variants) |

---

## Execution order

1. Step 1 (Types) — no downstream breakage, just wider types
2. Step 4 (Utils/ShellBinning FT) + Step 3 (assign_shells) — independent of Step 1
3. Step 2 (Workspaces) — depends on Steps 1+3+4
4. Steps 5+6 (in-place API) — depends on all above
5. Update ext and tests last
