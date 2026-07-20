#import "/book.typ": book-page
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

#let config = load_config()
#let l2gchip = load_chip("src/l2g.toml", config)
#let l2g = raw(l2gchip.name)
#let pagechip = load_chip("src/page.toml", config)
#let page = raw(pagechip.name)

In this chapter, we present our approach, which we name "epoch proving",#footnote[
  We additionally considered a "streaming" prover approach,
  proceeding in multiple phases (commit, logup, FRI, open),
  where only the tables that are actively being filled are required in memory.
  It comes at the cost of additional engineering complexity and re-executions.
]
to reducing the required amount of prover working memory
when proving larger programs.
We avoid resorting to a full-blown sharding+recursion approach, and as such avoid the
complications for the verifier to deal with cross-shard consistency constraints.

The overarching idea is to let the prover build up tables, and once memory-pressure grows
too much, perform some prover work to allow evicting tables from memory.

We observe that the majority of constraints on a contiguous section of cycles
is independent of the other cycles: only the consistency of the memory accesses has to cross all boundaries (see also @memory).
As such, we proceed in "epochs" ---the size of which is again informed by the memory pressure--- where
all table-to-table interactions can be proven within a single epoch.
One additional constraint on the size of an epoch comes from the padding scheme of the CPU (@cpu):
every epoch but the last should have a power-of-two number of rows for the CPU table(s) ---that is, it needs no padding---
since padding the CPU table requires the program to have already halted.

To deal with cross-epoch memory, we introduce a "local-to-global" table (`L2G`) per epoch that, in essence,
is an epoch-local memory initialization and finalization mechanism.
It initializes any memory cell that is accessed in the epoch, by claiming its value and originating epoch.
Similarly, it finalizes each accessed memory cell in the epoch-local LogUp, and claims the current value and last accessed timestamp.
This both allows frequent access to a small number of addresses within an epoch to have a small cross-epoch footprint,
and addresses that are not accessed for multiple consecutive epochs to not incur a cost when not accessed.
In this system, timestamp values only carry significance within the epoch they occur in,
and the ordered pair $(serif("epoch"), serif("timestamp"))$ can be considered as the "global timestamp",
although no part of the VM requires a global timestamp to be materialized.
Since we choose to represent the epoch-local timestamps as 32-bit `Word` values,
an epoch _should not_ consist of more than $2^30$ cycles
(refer to the scaling factor in @memory:aside:granularity).
The epochs should be 1-indexed, such that @l2g:c:lt_epoch can be properly satisfied in combination with the
initialization performed by the #page chip.

As such, each epoch proceeds by committing to all its tables,#footnote[
  The local-to-global will require a separate Merkle tree to allow the separation of epoch-local and cross-epoch proving.
] obtaining epoch-local LogUp challenges,
proving the epoch-local (that is, without cross-epoch memory interaction) LogUp sum is zero, and performing a full batch FRI, including the queries.
All tables other than the local-to-global ones can be permanently evicted from memory at this point, and no further work for these tables is required.
It is plausible that the total size of all local-to-global tables remains small enough across a 1B-cycle execution,
that their LDE and Merkle tree can be kept in memory throughout.

Finally, one more LogUp is proven on the aggregation of all local-to-global tables, with the global memory initialization and finalization, to prove the global memory consistency.
Note that the Fiat-Shamir transcripts of both the epoch proofs and the global proof should be bound to
the same local-to-global commitments.

This system should allow simple scaling across multiple GPU or worker nodes, as each epoch can be proven
in isolation by a worker, while the execution and proving continues with the next epoch in tandem.

While this approach cleanly sidesteps the need for re-execution ---or if re-execution is needed still for
the local-to-global tables, avoids most of the computational work in interpolation and hashing---
it comes with some tradeoffs:
- An extra table has to be introduced and proven. This is likely to be very minor compared to the overall work.
- By having a single cycle at which all tables need to be proven and retired,
  rather than only retiring completed tables, a larger cost in padding is expected.
  Having many tables that could consist of only a few rows is likely to make this even more pronounced.
  Lowering the table size and proving more tables per epoch may alleviate this somewhat.
- The overall proof conceptually consists of several sub-proofs that should however _not_ be considered
  in isolation, as there must be shared commitments between the epoch-local proofs and the global memory proof.

= #l2g chip

== Variables

#render_chip_variable_table(l2gchip, config)

== Constraints

First, we have the consistency constraints that need to be enforced for the chip's correctness.
These can be enforced in the epoch-local proof, and will still hold during the global proof by
making sure the commitment matches between the local and global proof.
We currently assume that no more than $2^20$ epochs can occur in one execution.
This can be extended at the cost of extra interactions and potentially extra columns.

#render_constraint_table(l2gchip, config, groups: "consistency")

Then we have local constraints that should be enforced inside of the epoch.

#render_constraint_table(l2gchip, config, groups: "local")

And finally, the interactions that are part of the global memory consistency proof.

#render_constraint_table(l2gchip, config, groups: "global")

== Padding

Observe that all interactions in this table are unconditional.
As a result, padding rows need to be chosen such that they have no unwanted effects.
We achieve this by "bringing forward" unused values from an older epoch.
Even though they are not touched by the current epoch, the #l2g chips claims they are
and simply has finalization clean up the spurious initialization, by setting
`fini_value = init_value` and `fini_timestamp = 0`.

= #page chip

We resume here the description of memory initialization and finalization from @memory,
as it applies after integration of the epoch system described above.
Concretely, each page gets an associated `PAGE` table, consisting of #total_nr_variables(pagechip) variables
over #total_nr_instantiated_columns(pagechip, config) columns.
For each such table, the `page` variable is instantiated as the constant base address of the page.
The `offset` column is preprocessed, which helps the verifier ensure that each page has a single fixed size,
but the verifier should still check that no pages overlap and all `page` values are page-aligned.

Observe that this table deals with boundaries on the RAM memory.
Registers can still be initialized and finalized in the global memory directly by the verifier,
and then transparently bridged between epochs in the same #l2g tables that are already present for RAM values too.

== Constraints

We present here a set of constraints on the #page table that

+ enforces the initial and final values of each address are bytes
+ adds the initial and final interaction to the global memory LogUp argument

For zero-initialized pages, `init` can be a constant `0`,
and hence doesn't need a column, nor a range check.

#render_chip_variable_table(pagechip, config)
#render_constraint_table(pagechip, config)
