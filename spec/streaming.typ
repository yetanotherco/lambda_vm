#import "/book.typ": book-page, rj, aside, xref
#import "/src.typ": load_config, load_chip
#import "/chip.typ": (
  render_chip_assumptions,
  render_chip_variable_table,
  render_chip_padding_table,
  render_constraint_table,
  total_nr_instantiated_columns,
  total_nr_variables,
)

#show: book-page("streaming.typ")

We present two potential approaches to reducing the required amount of prover working memory
when proving larger programs.
We avoid resorting to a full-blown sharding+recursion approach, and as such avoid the complications
complications for the verifier to deal with cross-shard consistency constraints.

The overarching idea in both is to let the prover build up tables, and once memory-pressure grows
too much, perform some prover work to allow evicting tables from memory.
the difference lies in exactly what and how much proving happens, and how much extra computation is needed
later on.

= Approach 1: Prove-and-retire

In the first potential approach, we proceed in multiple passes:
- *Commit*: The prover proceeds through execution, witness generation and interpolation as usual
  once the memory pressure becomes too large, the prover batch commits to all full (or large enough) tables
  in memory; then these tables are dropped from memory ("retired").
  All FRI polynomials generated during this phase can already be accumulated in a single batch polynomial,
  by sampling the batching coefficients after commiting to the polynomials they randomize.
- *Challenge*: At the end of the execution, the remaining tables are padded and commited to.
  LogUp challenges are sampled.
- *LogUp re-execution*: The prover restarts execution from zero, performing the reduction to FRI polynomials
  for the LogUp interactions, and batching the polynomials into the batch FRI polynomial from before.
  Again, this can be optimized for memory usage by only keeping 
- *FRI*: At this point, all FRI instances are available in a single batch polynomial,
  and the FRI folding can be performed, resulting in a set of query opening points.
- *Open*: One final re-execution can now be performed to generate the requested Merkle openings.

This approach comes at a clear cost of extra interpolations and Merkle tree evaluations,
but manages to keep a clean evaluation model that almost directly (apart from the early commitments and batching)
corresponds to the naive, memory-intensive proving model.
An optimization is possible to reduce the cost of the *Open* phase, by keeping the internal nodes of the merkle tree
in memory, obviating the need to recompute it; while still dropping the biggest memory cost (the leaves).

To distribute this workload across worker nodes or multiple GPUs, it is possible to "send off"
each batch of tables being retired to the next available worker.
With some care, it is likely possible to decouple the Fiat-Shamir challenges of multiple retirement batches,
so that this doesn't introduce a dependency on a previous batch completing before the next one can happen.

= Approach 2: Prove-epoch

In the second potential approach, we observe that the majority of constraints on a contiguous section of cycles
is independent of the other cycles: only the consistency of the memory accesses has to cross all boundaries.
As such, we proceed in "epochs" ---the size of which is again informed by the memory pressure--- where
all table-to-table interactions can be proven within a single epoch.

To deal with cross-epoch memory, we introduce a "local-to-global" table per epoch that, in essence,
is an epoch-local memory initialiation and finalization mechanism.
It initializes any memory cell that is accessed in the epoch, by claiming its value, originating epoch, and timestamp.
Similarly, each accessed memory cell is finalized in the epoch-local LogUp, and claims the current value, epoch and last accessed timestamp.
This both allows frequent access to a small number of addresses within an epoch to have a small cross-epoch footprint,
and addresses that are not accessed for multiple consecutive epochs to not incur a cost when not accessed.

As such, each epoch proceeds by commiting to all its tables,#footnote[
  The local-to-global will require a separate Merkle tree to allow the separation of epoch-local and cross-epoch proving.
] obtaining epoch-local LogUp challenges,
proving the epoch-local (that is, without cross-epoch memory interaction) LogUp sum is zero, and performing a full batch FRI, including the queries.
All tables other than the local-to-global ones can be permanently evicted from memory at this point, and no re-execution (or interpolation and commitment) for these tables is required.
It is plausible that the total size of all local-to-global tables remains small enough across a 1B-cycle execution,
that no re-execution at all is required.

Finally, one more LogUp is proven on the aggregation of all local-to-global tables, with the global memory initialization and finalization, to prove the global memory consistency.

Similarly to the previous approach, this should distribute cleanly across multiple worker nodes or GPUs.

While this approach cleanly sidesteps the need for re-execution ---or if re-execution is needed still for
the local-to-global tables, avoids most of the computational work in interpolation and hashing---
it too comes with some tradeoffs:
- An extra table has to be introduced and proven. This is likely to be very minor compared to the overall work.
- By having a single cycle at which all tables need to be proven and retired,
  rather than only retiring completed tables, a larger cost in padding is expected.
  Having many tables that could consist of only a few rows is likely to make this even more pronounced.
  Lowering the table size and proving more tables per epoch may alleviate this somewhat.
- The overall proof conceptually consists of several sub-proofs that should however _not_ be considered
  in isolation, as there must be shared commitments between the epoch-local proofs and the global memory proof.
