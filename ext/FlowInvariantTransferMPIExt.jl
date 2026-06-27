module FlowInvariantTransferMPIExt

using MPI: MPI
using FlowInvariantTransfer: FlowInvariantTransfer as FET

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

function FET.mpi_batch_map(
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

end # module FlowInvariantTransferMPIExt
