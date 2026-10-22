module FlowInvariantTransferMPIExt

using MPI: MPI
using FlowInvariantTransfer: FlowInvariantTransfer as FIT
using ComputationalBackends: ComputationalBackends

# ---------------------------------------------------------------------------
# Batch axis: distribute independent inputs across ranks (round-robin), then
# collate (gather) or reduce. No communication during each item's computation —
# every rank's item fits in its own memory. This is the embarrassingly-parallel
# "many snapshots" mode, orthogonal to the pencil decomposition.
# ---------------------------------------------------------------------------

# Combine the gathered per-rank index/result pairs into a single ordered Vector.
function _collate_in_order(gathered, N::Int)
    out = Vector{Any}(undef, N)
    for rank_chunk in gathered          # one chunk per rank: Vector of (idx, result)
        for (idx, res) in rank_chunk
            out[idx] = res
        end
    end
    # Narrow the element type now that every slot is filled.
    return identity.(out)
end

function _apply_reduction(flat::AbstractVector, reduction)
    if reduction === :gather
        return flat
    elseif reduction === :sum
        return reduce(+, flat)
    elseif reduction === :mean
        return reduce(+, flat) ./ length(flat)
    elseif reduction isa Function
        return reduce(reduction, flat)
    else
        throw(ArgumentError("reduce must be :gather, :sum, :mean, or a binary function (got $(reduction))."))
    end
end

function FIT.mpi_batch_map(
    f,
    items;
    comm = MPI.COMM_WORLD,
    reduce = :gather,
    root::Int = 0,
)
    MPI.Initialized() || throw(ArgumentError("MPI is not initialized; call MPI.Init() first."))
    rank  = MPI.Comm_rank(comm)
    nproc = MPI.Comm_size(comm)
    N     = length(items)

    # Round-robin assignment keeps load balanced when item costs are similar and
    # is deterministic, so the collated order is reproducible.
    my_indices = (rank + 1):nproc:N
    local_pairs = [(i, f(items[i])) for i in my_indices]

    # Arbitrary (non-bits) results: MPI.gather serializes; returns Vector-of-chunks on root.
    gathered = MPI.gather(local_pairs, comm; root = root)

    result = if rank == root
        flat = _collate_in_order(gathered, N)
        _apply_reduction(flat, reduce)
    else
        nothing
    end

    # Make the combined result available on every rank.
    return MPI.bcast(result, comm; root = root)
end

# ---------------------------------------------------------------------------
# `*_batch` entries under `execution = MPIBackend(inner; comm)`: rank `r` of `p` runs snapshots
# `r+1, r+1+p, …` through the batch's own driver on `inner`, and every rank returns every result, in
# input order.
# ---------------------------------------------------------------------------

_comm(b::ComputationalBackends.AbstractMPIBackend) = something(b.comm, MPI.COMM_WORLD)
_mine(comm, n::Integer) = (MPI.Comm_rank(comm) + 1):MPI.Comm_size(comm):Int(n)
_vec(x) = x isa AbstractVector ? x : collect(x)

# `(index, result)` pairs from every rank, on every rank.
function _allgather(pairs, comm)
    gathered = MPI.gather(pairs, comm; root = 0)
    return MPI.bcast(MPI.Comm_rank(comm) == 0 ? reduce(vcat, gathered) : nothing, comm; root = 0)
end

# A driver that fills `results`: this rank's share through `run!(inner, results_view, items_view)`.
function _mpi_fill!(run!, b, results, items)
    comm = _comm(b)
    mine = _mine(comm, length(items))
    isempty(mine) || run!(ComputationalBackends.local_backend(b), view(results, mine), view(items, mine))
    for (i, r) in _allgather([(i, results[i]) for i in mine], comm)
        results[i] = r
    end
    return results
end

# A driver that returns its results: this rank's share through `run(inner, indices)`.
function _mpi_collect(run, b, n::Integer)
    comm = _comm(b)
    mine = _mine(comm, n)
    own = isempty(mine) ? Any[] : run(ComputationalBackends.local_backend(b), mine)
    pairs = sort!(_allgather(collect(zip(mine, own)), comm); by = first)
    return [r for (_, r) in pairs]
end

FIT.SpectralFlux._spectral_flux_batch!(b::ComputationalBackends.AbstractMPIBackend, results, velocity_hats,
                                       ks, shell_idx; kwargs...) =
    _mpi_fill!(b, results, _vec(velocity_hats)) do inner, rs, vs
        FIT.SpectralFlux._spectral_flux_batch!(inner, rs, vs, ks, shell_idx; kwargs...)
    end

FIT.ShellToShellTransfer._shell_to_shell_batch!(b::ComputationalBackends.AbstractMPIBackend, results,
                                                velocity_hats, ks, centers, edges, N_sh; kwargs...) =
    _mpi_fill!(b, results, _vec(velocity_hats)) do inner, rs, vs
        FIT.ShellToShellTransfer._shell_to_shell_batch!(inner, rs, vs, ks, centers, edges, N_sh; kwargs...)
    end

FIT.BandTransfer._band_to_band_batch!(b::ComputationalBackends.AbstractMPIBackend, results, velocity_hats,
                                      ks, bands, nb; kwargs...) =
    _mpi_fill!(b, results, _vec(velocity_hats)) do inner, rs, vs
        FIT.BandTransfer._band_to_band_batch!(inner, rs, vs, ks, bands, nb; kwargs...)
    end

function FIT.SpectralFlux._partial_fluxes_batch!(b::ComputationalBackends.AbstractMPIBackend, velocity_hats,
                                                 ks; kwargs...)
    vs = _vec(velocity_hats)
    return _mpi_collect(b, length(vs)) do inner, mine
        FIT.SpectralFlux._partial_fluxes_batch!(inner, view(vs, mine), ks; kwargs...)
    end
end

function FIT.CoarseGrainingFlux._cg_flux_batch!(b::ComputationalBackends.AbstractMPIBackend,
                                                velocity_fields_batch, geometry, ℓ, filter; kwargs...)
    vs = _vec(velocity_fields_batch)
    return _mpi_collect(b, length(vs)) do inner, mine
        FIT.CoarseGrainingFlux._cg_flux_batch!(inner, view(vs, mine), geometry, ℓ, filter; kwargs...)
    end
end

function FIT._nufft_cg_batch!(b::ComputationalBackends.AbstractMPIBackend, batch, scatter_coords, ℓ, filter,
                              ms; kwargs...)
    vs = _vec(batch)
    return _mpi_collect(b, length(vs)) do inner, mine
        FIT._nufft_cg_batch!(inner, view(vs, mine), scatter_coords, ℓ, filter, ms; kwargs...)
    end
end

function FIT.Compressible._compressible_batch!(b::ComputationalBackends.AbstractMPIBackend, velocity_hats,
                                               density_hats, pressure_hats, ks; kwargs...)
    vs = _vec(velocity_hats); ds = _vec(density_hats)
    ps = pressure_hats === nothing ? nothing : _vec(pressure_hats)
    return _mpi_collect(b, length(vs)) do inner, mine
        FIT.Compressible._compressible_batch!(inner, view(vs, mine), view(ds, mine),
                                              ps === nothing ? nothing : view(ps, mine), ks; kwargs...)
    end
end

end # module FlowInvariantTransferMPIExt
