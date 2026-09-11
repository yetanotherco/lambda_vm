#import "/src.typ": load_config, load_chip
#import "/chip.typ": (
  render_chip_variable_table,
  total_nr_variables,
  total_nr_instantiated_columns,
  compute_nr_interactions,
  render_constraint_table,
  render_chip_assumptions,
  render_chip_padding_table,
)

#let config = load_config()
#let chip = load_chip("src/memmove.toml", config)
#let memmove = raw(chip.name)

The #memmove chip moves a range of bytes from one location to another, eight bytes per row.
It is the only copying primitive of this VM: one chip serves `memcpy`, `memmove`, `memset` and the byte loop of a commitment.
#footnote([Linux man-pages on `memmove`, `memcpy` and `memset`; man7.org. #link("https://man7.org/linux/man-pages/man3/memmove.3.html")[[src]]])
The three functionalities differ only in where the bytes go and in which of the two accesses happens first:

#figure(
  table(
    columns: 5,
    align: left,
    table.header[*functionality*][*entered by*][*destination*][*read at*][*write at*],
    [`memcpy`/`memmove`], [`ECALL` $-30$], [RAM], [$#`timestamp` + 1$], [$#`timestamp` + 2$],
    [`memset`], [`ECALL` $-32$], [RAM], [$#`timestamp` + 2$], [$#`timestamp` + 1$],
    [`commit`], [`COMMIT` (@commit)], [commitment domain], [$#`timestamp` + 1$], [---],
  ),
  caption: [The three functionalities of #memmove.],
) <memmove:functionalities>

Neither the destination domain nor the order of the two accesses is chosen by the caller; both follow from `is_set` and `is_commit`, which are decoded from the way the sequence was entered.

= Variables
#let nr_variables = total_nr_variables(chip)
#let nr_columns = total_nr_instantiated_columns(chip, config)
#let nr_interactions = compute_nr_interactions(chip)

The #memmove chip is comprised of #nr_variables variables that are expressed using #nr_columns columns and leverages #nr_interactions interaction(s):
#render_chip_variable_table(chip, config)

= Assumptions
#render_chip_assumptions(chip, config)

These concern the _first_ row of a sequence, where the values come from the register file or from `COMMIT`; every later row receives them over `MEMMOVE_NEXT`, where @memmove:c:range_src_incr, @memmove:c:range_dst_incr and @memmove:c:range_count_decr range-check three of the four on the sending side.
`timestamp` is range-checked by neither side and holds only because it travels unchanged from the `ECALL` at the root.
@memmove:a:dst is not discharged at all on a commitment sequence, where `dst` is an index that `COMMIT` carries as a `BaseField`; see the notes below.

= Constraints
In this VM, we assign system call number $-30$ to the copy functionality and $-32$ to `memset`.
Since the number of bytes is not known in advance, this chip is recursive: each row moves one chunk and "calls" itself to move the remainder, so only the `first` row of a sequence accepts an entry.
There are two entries --- an `ECALL` from the `CPU`, or the byte loop `COMMIT` defers (@commit) --- and `first_ecall` and `first_defer` split `first` between them.
#render_constraint_table(chip, config, groups: "incoming")

== Selecting the functionality
@memmove:c:receive_ecall receives the system call number as $2^32 - 30 - 2 dot #`is_set`$, so `is_set` is decoded from the `ECALL` the guest executed rather than chosen.
Note that the low limb of that tuple is a line in `is_set` and so reaches every system call number in the negative range: @memmove:c:range_is_set is what excludes them, and it therefore carries the whole decoding argument.
`is_commit` is decoded instead from _which_ bus the first row accepted from, `COMMIT_DEFER` having exactly one sender.
Both selectors ride inside the `MEMMOVE_NEXT` tuple in either direction, so a sequence cannot change functionality half way through it.
#render_constraint_table(chip, config, groups: "functionality")

The last three define nothing new --- they are a selector combined with `first`, with $#`μ` - #`end`$ and with `tail` --- and exist as columns only because a multiplicity must be linear in the columns of the chip (@logup).

== Reading the operands
The guest-side `memcpy` this chip accelerates has the following signature:

```c
void *memcpy(size_t count; void dest[restrict count], const void src[restrict count], size_t count);
```

That is to say, `A0` contains the address of the first byte to write, `A1` the address of the first byte to read, and `A2` the number of bytes to move; `memset` uses the same three registers for the same three roles.
Each read writes back the value it read, so the operation leaves the registers untouched and the guest produces the return value.
These are conditioned on `first_ecall`, since a deferred commitment sequence takes its operands from `COMMIT`.
#render_constraint_table(chip, config, groups: "read_input")

== Chunk width
A row moves eight bytes, or a single byte when `tail` is set.
@memmove:c:short pins $#`short` = (#`count` < 8)$ to `LT` (@lt) and @memmove:c:wide_needs_eight forbids a wide row when fewer than eight bytes remain; the other direction is deliberately free, so the prover may cut any row to a single byte.
That freedom decides which memory chip a row reaches: `MEMW_A` (@memw) admits an access that does not cross a $2^16$ limb boundary and whose bytes share one old timestamp, and a buffer last written in eight-byte groups has one timestamp per group, so a schedule can spend narrow rows to land its wide rows on those groups.
@memmove:c:bound proves $#`count` < 257$ on the first row of every `ECALL`-entered sequence, capping it at $257$ rows; the guest stubs chunk larger operations.
#render_constraint_table(chip, config, groups: "width")

Note that @memmove:c:bound carries `first_ecall`, so a commitment sequence proves no byte bound, and `COMMIT` range-checks none either.
No prover gain follows --- the commitment bus must balance against the committed output, which the verifier knows in full --- but the verifier contributes a token pair per committed byte, so an absurd `count` is unverifiable as well as unprovable.

== Performing the move
The bytes are read at $#`timestamp` + 1 + #`is_set`$ and written at $#`timestamp` + 2 - #`is_set`$, one timestamp apart, with `is_set` deciding which comes first.
The `CPU`'s preprocessed timestamp column holds $4 dot (i + 1)$ at row $i$ (@vars), so neither expression leaves the `Word` range.
Both interactions are expressed over the _same_ `value` variable, which is what makes the moved bytes equal, and the read carries `value` as input and output, pinning it to whatever the memory argument (@memory) holds at `src`.
@memmove:c:tail_lanes canonicalises a narrow row, whose seven unused lanes `MEMW` gates out of the memory argument but not out of its own tuple.
#render_constraint_table(chip, config, groups: "copy")

Every row of a sequence carries the same `timestamp`, so an entire sequence reads at one instant and writes at another.
Under the normal order every read therefore observes memory as it was before the sequence started, which is `memmove`'s guarantee for an overlapping range; under the inverted order every read observes memory after all of the sequence's writes, and the sequence propagates rather than copies.
That is `memset`: the guest seeds eight bytes with ordinary stores and calls with `src` the start of the seed and `dst` its end, and each row forces $"mem"[#`src` + k + 8] = "mem"[#`src` + k]$, replicating the seed across the range.

@memmove:c:set_gap_lo and @memmove:c:set_gap_hi pin that gap, and they are load-bearing: at $#`dst` = #`src`$ the read and the write address one cell at adjacent timestamps, the memory argument closes on $#`value` = #`value`$, and all eight lanes --- pinned by the read alone --- become free field elements.
Only $#`dst` = #`src`$ frees them; the gap is $8$ because that is the widest row, so a wide row's read and write ranges stay disjoint.
The gate is `is_set` alone rather than a product, which would cost a degree.
Being limb-wise, the pair admits no carry out of the low limb, so a `memset` whose range crosses the $2^32$ boundary has no satisfying assignment at all --- a precondition on the caller, which the executor enforces.

== Writing to the commitment domain
When `is_commit` is set, `dst` is the index of the byte in the committed output rather than an address, and the write goes to a domain-separated part of memory with separator $2$, which the verifier initializes and finalizes itself (@memory, @streaming).#footnote[
  In order to make sure the verifier can properly finalize the committed values, the last epoch can "bring forward" all commitments from earlier epochs, similar to padded values, in the `L2G` table.
  Then the contribution of the commitments only consists of the tuples `(2, address, last_epoch_index, value)`, which is entirely known to the verifier.
]
@memmove:c:write_value is gated on `ram_write` and does not fire; these take its place, and need no `MEMW`, since a commitment cell is written once and read by nobody.
#render_constraint_table(chip, config, groups: "commit")

A committing row emits one interaction per _byte_, at index $#`dst` + i$, because the verifier rebuilds this side of the bus from the committed output alone and cannot reproduce the prover's row schedule, which restarts at every system call.
Addressing every byte by its own index removes the grouping; `commit_lane` keeps a narrow row from committing the seven bytes it never read.

== Advancing to the next chunk
In parallel, we compute $#`src_incr` = #`src` + #`step`$ and $#`dst_incr` = #`dst` + #`step`$ as the positions at which the next chunk starts, and $#`count_decr` = #`count` - #`step`$ as the number of bytes still to move.
The first two of @memmove:c:range_src_incr, @memmove:c:range_dst_incr and @memmove:c:range_count_decr are included to satisfy @addnw:a:sum, and the last to satisfy @sub:a:diff.
#render_constraint_table(chip, config, groups: "incr_decr")

The positions use `ADDNW` (@add), which forbids wraparound modulo $2^64$: without it a sequence could walk `src` past the end of the address space, or close into a ring that balances every bus while moving nothing that was asked for.
The count uses plain `SUB`, which permits it, because the terminal row holds $#`count` = 0$ and hence $#`count_decr` = 2^64 - 1$.
That is safe because $#`step` <= #`count`$ on every row with $#`count` >= 1$: a wide row needs $#`short` = 0$ and hence $#`count` >= 8 = #`step`$, and a narrow row has $#`step` = 1$.

== Terminating the sequence
When `count` hits $0$ we stop recursing, which the `end` bit indicates.
#render_constraint_table(chip, config, groups: "end")

*Note*:
+ We set $#`end` = 1$ when $#`count_decr` = -1$ rather than when $#`count` = 0$, which allows `count` to be stored in a `DWordWL` rather than a `DWordHL`.
+ $forall i in [0, 3]: 65535 - #`count_decr`_i >= 0$ as a result of @memmove:c:range_count_decr, hence $sum_(i=0)^3 65535 - #`count_decr`_i = 0 arrow.l.r.double.long forall i: #`count_decr`_i = 65535$.
  Without those range checks one limb could compensate another and `end` would be claimable at a nonzero count --- a silently truncated operation with every bus balanced, since every memory interaction vanishes with `end`.
+ $#`end` = 1$ still forces $#`count` = 0$ even though the prover picks the width: the other candidate, $#`count` = 7$ with a wide row, gives $#`short` = 1$ and is rejected by @memmove:c:wide_needs_eight.
+ An operation on zero bytes is a single row with $#`first` = #`end` = 1$.

== Chaining the rows
When this was not the last chunk, we recursively move the next one over `MEMMOVE_NEXT`, carrying the timestamp, the three updated values and the functionality.
Both tuples carry the `timestamp`, which is what separates one sequence from another; since the CPU's timestamps strictly increase per instruction, no two sequences share one.
#render_constraint_table(chip, config, groups: "lookups")

This chip has no constraint demanding that a sequence terminates, and the reason it needs none is worth stating, because the counting argument alone is not sufficient.
Fix a timestamp: balancing `MEMMOVE_NEXT` forces the number of rows claiming `end` to equal the number claiming `first`, and $#`first` = #`first_ecall` + #`first_defer`$ caps that at one, since the `CPU` sends one `ECALL` per timestamp and that `ECALL` cannot be both a copy and a `write`.
That rules out an _open_ sequence and nothing more: a ring of rows with neither `first` nor `end` set sends and receives one tuple each, so it balances while consuming no entry.
What forbids the ring is @memmove:c:src_incr, since `ADDNW` forces $#`src_incr` = #`src` + #`step`$ over the integers with $#`step` >= 1$, so `src` strictly increases and can never return to a value it held.

== Bits
Lastly, the seven independent bits must be bits, and $#`first` = 1$ or $#`end` = 1$ must imply $#`μ` = 1$, to keep the multiplicities $-(#`μ` - #`first`)$ and $#`μ` - #`end`$ binary.
`first_ecall`, `commit_write` and `commit_write_wide` need no range check, being already equated to products of bits.
#render_constraint_table(chip, config, groups: "bits")

= Padding
To pad this chip, use the below data.
#render_chip_padding_table(chip, config)

This padding row is not all-zero: @memmove:c:count_decr is unconditional, so $#`tail` = 1$ makes $#`step` = 1$, which $#`count` = 1$ and $#`count_decr` = 0$ satisfy, and $#`tail` = 1$ also satisfies @memmove:c:wide_needs_eight.
The low-limb carry of the two position updates is constrained on every row (@addnw:c:carry), which $#`src_incr` = #`dst_incr` = 1$ satisfies with a zero carry.

= Notes/potential optimizations
- On a commitment sequence @memmove:a:dst is undischarged, and @memmove:c:dst_incr leans on it through @addnw:a:lhs, so a denormalized index weakens that `ADDNW` to a field statement; the per-lane commitment addresses are likewise not carry-normalized as `MEMW`'s are. Neither is a prover gain, since such a token has no receiver, but range-checking `index` where it enters `COMMIT` would settle both --- and would also stop @commit:c:read_index writing past the `Word` range into `x254`.
- `count` need not be a full `DWordWL` on the `ECALL` path, where @memmove:c:bound already proves $#`count` < 257$; the commitment path has no such bound, so this costs a range check where the value enters from `COMMIT`.
- @memmove:c:range_src_incr and @memmove:c:range_dst_incr carry multiplicity $#`μ`$ while `src_incr` and `dst_incr` are consumed only at $#`μ` - #`end`$. Lowering both would drop eight `IS_HALF` lookups per terminal row, though not shrink the proof, that table being preprocessed at a fixed height. @memmove:c:range_count_decr genuinely needs $#`μ`$.
- Selecting between exactly two widths keeps `tail` a single bit, so $#`step` = 8 - 7 dot #`tail`$ stays linear and every constraint stays of degree 2. Four-, two- and one-byte chunks would save at most eight rows per sequence, at the cost of a two-bit selector decoded into `MEMW`'s width flags.
- A row could move sixteen or thirty-two bytes, at the cost of a wider `MEMW` signature, but `MEMW_A` needs every byte of an access to share one old timestamp, which a wider row only manages where the buffer was written in groups at least that wide.
- `COMMIT` could send its deferral on `MEMMOVE_NEXT` directly, retiring the `COMMIT_DEFER` bus and the `first_ecall` column, at the cost of `first` no longer meaning "head of the sequence" and of an added $#`first` dot #`is_commit` = 0$.
- The `memmove` property belongs to one `ECALL`: past 256 bytes the stub splits into several at distinct timestamps, and chunk $k+1$ reads what chunk $k$ wrote. That is in-contract for `memcpy`, whose buffers may not overlap.

= The Accelerated Memory Operations standard
The Ethereum Foundation's Accelerated Memory Operations standard fixes what an accelerated `memcpy`, `memmove` and `memset` must provide.
#footnote([Accelerated Memory Operations; eth-act/zkevm-standards. #link("https://github.com/eth-act/zkevm-standards/tree/main/standards/accelerated-memory-operations")[[src]]])
Of the chip itself it asks that operands of arbitrary alignment be accepted, which they are: no constraint here refers to the alignment of `src`, `dst` or `count`, and a row's width is tied to none of them.
The one restriction this chip does impose is not an alignment --- a `memset` may not straddle the $2^32$ limb boundary --- and the standard's fourth operation, `memcmp`, is not covered, as it does not copy.
Its two remaining requirements fall outside this chapter: that the accelerated symbol behave identically to the C library function, which the guest stub is responsible for, and that it be a strong definition in an unconditionally linked object, which is a matter of linking.
