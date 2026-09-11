#import "/meta.typ": aside
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
It is the single copying primitive of this VM: one chip serves `memcpy`, `memmove`, `memset` and the byte loop of a commitment.
#footnote([Linux man-pages on `memmove`, `memcpy` and `memset`; man7.org. #link("https://man7.org/linux/man-pages/man3/memmove.3.html")[[src]]])
Without it the guest would run these loops in RISC-V --- per doubleword a load, a store, two pointer increments and a branch, five instructions the proof pays for.

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

Neither the destination domain nor the order of the two accesses is chosen by the caller.
Both are derived from `is_set` and `is_commit`, and those two bits are in turn decoded from the way the sequence was entered.

= Variables
#let nr_variables = total_nr_variables(chip)
#let nr_columns = total_nr_instantiated_columns(chip, config)
#let nr_interactions = compute_nr_interactions(chip)

The #memmove chip is comprised of #nr_variables variables that are expressed using #nr_columns columns and leverages #nr_interactions interaction(s):
#render_chip_variable_table(chip, config)

= Assumptions
#render_chip_assumptions(chip, config)

These assumptions concern the _first_ row of a sequence.
There, `src`, `dst` and `count` come either from the register file or from `COMMIT`, and `timestamp` from the `CPU`.
Every later row receives all four over the `MEMMOVE_NEXT` bus, where @memmove:c:range_src_incr, @memmove:c:range_dst_incr and @memmove:c:range_count_decr range-check three of them on the sending side.
The fourth, `timestamp`, is range-checked by neither side; it holds because the value travels unchanged from the `ECALL` at the root of the sequence, which is where @memmove:a:timestamp is discharged.

@memmove:a:dst is not discharged on a commitment sequence; that case is taken up at the end of this chapter.

= Constraints
This VM assigns system call number $-30$ to the copy functionality and $-32$ to `memset`.
Since the number of bytes is not known in advance, this chip is recursive: each row moves one chunk and "calls" itself to move the remainder.
A sequence is therefore entered exactly once, and only that first row accepts an entry.
There are two ways in: from the `CPU`, by accepting an `ECALL`, or from `COMMIT`, which keeps the `write` system call for itself and defers only the byte loop.
#render_constraint_table(chip, config, groups: "incoming")

== Selecting the functionality
@memmove:c:receive_ecall receives the system call number as $2^32 - 30 - 2 dot #`is_set`$, a linear function of the selector, so `is_set` is decoded from the `ECALL` the guest actually executed rather than chosen by the prover.
`is_commit` is decoded from _which_ bus the first row accepted from: `COMMIT_DEFER` has exactly one sender, and that is `COMMIT`.

#aside("The syscall number alone does not pin the selector")[
  The low limb of @memmove:c:receive_ecall is a line in `is_set` and so runs over the whole field, while the high limb is the constant $2^32 - 1$.
  Every system call number in the negative range is therefore reproduced by some field element, and @memmove:c:range_is_set is what rules those out --- it carries the whole decoding argument.
  This matters when `ECALL` numbers are allocated (@ecall): a negative number is safe from this chip because `is_set` is a bit, never because it is far away from $-30$ and $-32$.
]

@memmove:c:one_hot makes the two selectors mutually exclusive and @memmove:c:functionality_implies_mu keeps both clear on a padding row.
Both selectors also ride _inside_ the `MEMMOVE_NEXT` tuple in either direction (@memmove:c:send_next_chunk, @memmove:c:receive_next_chunk), so a sequence cannot change functionality half way through it.
#render_constraint_table(chip, config, groups: "functionality")

The last three of these define nothing new; they combine a selector with `first`, with $#`μ` - #`end`$, and with `tail` respectively.
They exist as columns because a multiplicity has to be linear in the columns of the chip (@logup), and a product of two of them is not.
The remaining combinations, `first_defer` and `ram_write`, are differences of columns and so need no column of their own.

== Reading the operands
The guest-side `memcpy` this chip accelerates has the following signature:

```c
void *memcpy(size_t count; void dest[restrict count], const void src[restrict count], size_t count);
```

That is to say,
- `A0` contains the address of the first byte to write,
- `A1` contains the address of the first byte to read, and
- `A2` contains `count`; the number of bytes to move.

`memset` uses the same three registers for the same three roles --- `A1` is a source address there too, not a fill byte --- and what the guest puts in them is discussed at the bottom of this chapter.
@memmove:c:read_dst, @memmove:c:read_src and @memmove:c:read_count read the three registers.
Each of them writes back the value that was read, so the operation leaves the registers untouched;
the guest is responsible for producing the return value.
These reads are conditioned on `first_ecall`: a deferred commitment sequence takes its operands from `COMMIT` instead, which has already read the registers of the `write` system call.
#render_constraint_table(chip, config, groups: "read_input")

== Chunk width
A row moves eight bytes, or a single byte when `tail` is set.
@memmove:c:short pins $#`short` = (#`count` < 8)$ to `LT` (@lt) and @memmove:c:wide_needs_eight forbids a wide row when fewer than eight bytes remain.
The other direction is deliberately left free: the prover may cut any row down to a single byte.
That freedom is what lets a schedule choose where its wide rows fall, which is what decides whether they are admitted by `MEMW_A` rather than by the wider `MEMW` (@memw).
Which schedule is used is a prover-side choice; the AIR grants the freedom and charges for the rows.

Because a row is the unit in which this chip charges, an unbounded `count` would let a single guest instruction append an unbounded number of rows to the trace.
@memmove:c:bound therefore proves $#`count` < 257$ on the first row of every `ECALL`-entered sequence, which caps it at $257$ rows --- the bound is the one-byte-per-row schedule, not the honest one.
The guest-side stubs chunk larger operations into multiple `ECALL`s; the executor rejects any chunk exceeding 256 bytes.
#render_constraint_table(chip, config, groups: "width")

Note that @memmove:c:bound carries `first_ecall`, so a commitment sequence proves no byte bound at all --- and `COMMIT` range-checks no `count` either.
No prover gain follows: the commitment bus still has to balance against the committed output, which the verifier knows in full, so a sequence can only be as long as the output it produces.
What it does cost is work, and not only the prover's --- the verifier contributes a token pair per committed byte --- so a `write` with an absurd `count` is unprovable and unverifiable rather than merely expensive.

#aside("Why a one-byte tail")[
  Selecting between exactly two widths lets `tail` be a single bit, so $#`step` = 8 - 7 dot #`tail`$ stays linear and every constraint in this chip stays of degree 2.
  Splitting the remainder into four-, two- and one-byte chunks instead would shave off at most eight rows per sequence, at the cost of a two-bit width selector and a decoding of that selector into `MEMW`'s `write2`/`write4`/`write8` flags.
]

== Performing the move
The bytes are read at $#`timestamp` + 1 + #`is_set`$ and written at $#`timestamp` + 2 - #`is_set`$: the two accesses are one timestamp apart, and `is_set` decides which of them comes first.
The `CPU`'s preprocessed timestamp column holds $4 dot (i + 1)$ at row $i$ (@vars), so neither expression can leave the `Word` range --- `IS_WORD` on `timestamp` alone would not rule that out.
Both interactions are expressed over the _same_ `value` variable, which is what makes the moved bytes equal:
there is nothing to constrain, since there is only one set of columns.
The read carries `value` as both its input and its output, so `value` is pinned to whatever the memory argument (@memory) says resides at `src`.
#render_constraint_table(chip, config, groups: "copy")

Every row of a sequence carries the same `timestamp`, so an entire sequence reads at one instant and writes at another.
With the normal order that makes every read observe memory as it was before the sequence started, which is precisely `memmove`'s guarantee for an overlapping range.
With the order inverted every read instead observes memory after all of the sequence's writes, and the sequence propagates rather than copies; that is `memset`, and @memmove:c:set_gap_lo and @memmove:c:set_gap_hi are what make it well-defined.
Both orders keep the read and the write at distinct timestamps and both strictly after `timestamp`, so the memory argument is undisturbed either way.

@memmove:c:tail_lanes canonicalises a one-byte row, which addresses `MEMW` with $#`write2` = #`write4` = #`write8` = 0$.
`MEMW` gates lane $i >= 1$ on those same flags (@memw), so the seven unused lanes never reach the memory argument --- but they do reach the `MEMW` tuple, and pinning them to zero is what keeps it the canonical encoding of a single-byte access rather than one carrying seven free field elements.

Which memory chip the two accesses reach is decided by their addresses, and this chip constrains neither.
`MEMW_A` is the fast path (@memw): it stores a single old timestamp, so it needs one `LT` row (@lt) to order the access, where the general `MEMW` stores one per byte and needs eight, on a row that is the wider of the two to begin with.
Two conditions admit an access there, and neither is the eight-byte alignment of its address: the access must not cross a $2^16$ limb boundary, and all the bytes it touches must carry the same old timestamp.
Alignment matters only through the second: a buffer last written in eight-byte aligned groups has one timestamp per group, so an eight-byte access that sits on a group reads one timestamp and an access that straddles two reads two.
That is what a schedule is buying when it spends one-byte rows to move a wide row onto a group boundary, and it is a property of how the buffer was written rather than of this chip.
The read and the write are routed independently, so a sequence can take the fast path at one end and not at the other, and the row counts this chapter reasons about are therefore not on their own the cost of an operation.

== Writing to the commitment domain
When `is_commit` is set the destination is not RAM.
`dst` is then the index of the byte in the committed output rather than an address#footnote[
  For very large commitments (with index $>= 2^32$), the commitment-domain address can become denormalized, but since no other chip interacts with this memory domain, there is no issue.
  The usual consistency guarantee from the LogUp argument and correct initialization as for general addresses applies.
], and the write goes to a domain-separated part of memory with domain separator $2$, which the verifier initializes and finalizes itself (@memory, @streaming).#footnote[
  In order to make sure the verifier can properly finalize the committed values, the last epoch can "bring forward"
  all commitments from earlier epochs, similar to padded values, in the `L2G` table.
  Then the contribution of the commitments only consists of the tuples `(2, address, last_epoch_index, value)`, which is entirely known to the verifier.
]
@memmove:c:write_value is therefore gated on `ram_write` and does not fire, and @memmove:c:commit_value_out and @memmove:c:commit_value_in take its place.
These need no `MEMW`: a commitment cell is written once and read by nobody, so there is no old value to produce and no timestamp to order.
#render_constraint_table(chip, config, groups: "commit")

A committing row emits one interaction _per byte_, at index $#`dst` + i$, rather than one for the row, and the grouping is what forces that.
The verifier reconstructs this side of the bus from the committed output alone, where it sees the concatenation of every commitment the program made --- not where one `write` ended and the next began --- so it cannot reproduce the prover's row schedule, which restarts at every system call.
A guest committing four bytes and then four more sends eight one-byte rows where a verifier chunking the eight bytes it sees expects a single wide row, and an honest proof would be rejected.
Addressing every byte by its own index removes the grouping; `commit_lane` keeps a one-byte row from committing the seven bytes it never read.

== Advancing to the next chunk
In parallel, we compute $#`src_incr` = #`src` + #`step`$ and $#`dst_incr` = #`dst` + #`step`$ as the positions at which the next chunk starts, and $#`count_decr` = #`count` - #`step`$ as the number of bytes that still have to be moved afterwards.
The first two of @memmove:c:range_src_incr, @memmove:c:range_dst_incr and @memmove:c:range_count_decr are included to satisfy @addnw:a:sum, and the last to satisfy @sub:a:diff.
#render_constraint_table(chip, config, groups: "incr_decr")

Note the asymmetry between the two position updates and the count update:

+ The positions use `ADDNW` (@add), which forbids wraparound modulo $2^64$.
  Without it a sequence could walk `src` past the end of the address space and continue at low addresses, touching memory unrelated to the requested range --- and, less obviously, a sequence could close into a ring that balances every bus while moving nothing that was asked for; see the discussion of termination below.
  The condition is $#`μ` - #`end`$: on the terminal row and on padding rows the computed successor is consumed by nobody, since @memmove:c:send_next_chunk carries that same multiplicity.
+ The count uses plain `SUB` (@add), which permits wraparound, because the terminal row holds $#`count` = 0$ and hence $#`count_decr` = 0 - 1 = 2^64 - 1$.
  That permission is safe because $#`step` <= #`count`$ on every row with $#`count` >= 1$: a wide row needs $#`short` = 0$ by @memmove:c:wide_needs_eight and hence $#`count` >= 8 = #`step`$, and a narrow row has $#`step` = 1$.
  The subtraction can therefore only wrap on the terminal row.

== Terminating the sequence
When `count` hits $0$, we should stop performing further recursive calls.
We use the `end` bit to indicate these circumstances.
#render_constraint_table(chip, config, groups: "end")

*Note*:
+ We set $#`end` = 1$ when $#`count_decr` = -1$ rather than when $#`count` = 0$, which allows `count` to be stored in a `DWordWL` rather than a `DWordHL`.
+ $forall i in [0, 3]: 65535 - #`count_decr`_i >= 0$ as a result of @memmove:c:range_count_decr.
 Hence,
  $
  sum_(i=0)^3 65535 - #`count_decr`_i = 0 arrow.l.r.double.long forall i in [0, 3]: #`count_decr`_i = 65535
  $
  Without those range checks the sum could _vanish_ for a `count_decr` other than $2^64 - 1$ --- one limb above $65535$ compensating another below it, as in $(65534, 65536, 65535, 65535)$ --- and `end` would be claimable at a nonzero count.
  That matters more here than the shape of the constraint suggests: since the read and both writes carry a multiplicity that vanishes with `end`, a row that wrongly claims `end` emits no memory operations at all, which is a silently truncated operation with every bus balanced.
+ $#`end` = 1$ still forces $#`count` = 0$ even though the prover picks the width.
  The other candidate is $#`count` = 7$ with a wide row, which wraps `count_decr` to $2^64 - 1$; but $#`count` = 7$ gives $#`short` = 1$, and @memmove:c:wide_needs_eight then rejects the wide row.
+ An operation on zero bytes is a single row with $#`first` = #`end` = 1$: it accepts the entry and reads the three registers, but emits no memory operations and starts no recursion.

== Chaining the rows
When this was not the last chunk of this sequence, we recursively move the next chunk over the `MEMMOVE_NEXT` bus, specifying the timestamp, the positions to continue reading and writing at, the number of bytes that still have to be moved, and the functionality (@memmove:c:send_next_chunk).
Since that certainly won't be the `first` row of the sequence, we read `src_incr`, `dst_incr` and `count_decr` from the previous recursion level into `src`, `dst` and `count`, and continue.
#render_constraint_table(chip, config, groups: "lookups")

Both tuples carry the `timestamp`, and that is what separates one sequence from another:
without it, rows belonging to two different sequences could be spliced into each other's while the bus still balances.
Since the CPU's timestamps strictly increase per instruction, no two #memmove sequences share one.

This chip has no constraint demanding that a sequence terminates, and the reason it needs none is worth stating carefully, because the obvious counting argument is not sufficient on its own.
Fix a timestamp.
Balancing `MEMMOVE_NEXT` forces the number of rows claiming `end` to equal the number claiming `first`, and $#`first` = #`first_ecall` + #`first_defer`$ caps the number of rows claiming `first` at one.
The `CPU` sends a single `ECALL` per timestamp, which is what caps @memmove:c:receive_ecall; and `COMMIT` puts at most one row on `COMMIT_DEFER` per timestamp, for the same reason, which is what caps @memmove:c:receive_commit_defer.
The two are moreover exclusive, since that single `ECALL` cannot be both a copy and a `write`.
So a sequence that simply runs on without ever setting `end` sends one tuple more than it receives, and the bus does not balance.

That rules out an _open_ sequence and nothing more: a ring of rows carrying $#`μ` = 1$ with neither `first` nor `end` set sends and receives one tuple each, so it balances, consumes no entry, and would still emit a read and a write per row.
What forbids the ring is @memmove:c:src_incr: `ADDNW` forces $#`src_incr` = #`src` + #`step`$ _over the integers_ with $#`step` >= 1$, so `src` strictly increases along the sequence and can never return to a value it already held.
This is the second reason the positions use `ADDNW` rather than `ADD`, and the more important of the two.

== Bits
Lastly, we must make sure the seven independent bits are bits, and that either $#`first` = 1$ or $#`end` = 1$ implies $#`μ` = 1$ (@memmove:c:first_or_end_implies_mu).
The latter is required to ensure the multiplicities $-(#`μ` - #`first`)$ and $#`μ` - #`end`$ are binary.
`first_ecall`, `commit_write` and `commit_write_wide` need no range check of their own: the constraints of @memmove:c:first_ecall, @memmove:c:commit_write and @memmove:c:commit_write_wide already equate each of them to a product of bits.
#render_constraint_table(chip, config, groups: "bits")

= Padding
To pad this chip, use the below data.
#render_chip_padding_table(chip, config)

Note that this padding row is not all-zero.
@memmove:c:count_decr is unconditional, so a padding row has to satisfy it too: $#`tail` = 1$ makes $#`step` = 1$, which $#`count` = 1$ and $#`count_decr` = 0$ then satisfy.
$#`tail` = 1$ in turn satisfies @memmove:c:wide_needs_eight for either value of `short`.
The two position updates are conditioned on $#`μ` - #`end`$ and so do not forbid a wraparound here, but their low-limb carry is constrained on every row (@addnw:c:carry), so a padding row must satisfy that relation too; $#`src_incr` = #`dst_incr` = 1$ is the assignment that does so with a zero carry.

= `memset` as a propagating `memmove`
`memset` needs no machinery of its own beyond the inverted order, but it does need the guest to hand it the right operands.

The stub seeds the first eight bytes of the range with ordinary stores and then calls the accelerator with `src` the start of that seed, `dst` its end and `count` the remaining length; fills shorter than sixteen bytes take a plain store loop instead, since they cannot amortise the seed.
The row at offset $k$ then reads at $#`src` + k$ and writes at $#`src` + k + 8$.
Because every row of the sequence writes at $#`timestamp` + 1$ and reads at $#`timestamp` + 2$, each read observes the whole sequence's writes, so the row forces $"mem"[#`src` + k + 8] = "mem"[#`src` + k]$ for every $k$, and the eight seeded bytes are replicated across the range.
The recursion bottoms out in the seed, which the sequence never wrote.
None of this depends on how the prover schedules the widths.

The call carries one precondition beyond the gap itself: neither `src` nor `src` $+$ `count` may cross the $2^32$ limb boundary, for the reason given in the third bullet below.
This is a real restriction on the caller rather than a property the chip enforces --- the AIR simply has no satisfying assignment for such a call --- so the executor rejects it up front and the guest stub keeps its chunks clear of the boundary.

@memmove:c:set_gap_lo and @memmove:c:set_gap_hi pin the gap, and the pinning is load-bearing rather than a convention.
The degenerate case is $#`dst` = #`src`$: the read and the write then address the same cell at two adjacent timestamps, the memory argument is satisfied by $#`value` = #`value`$, and all eight lanes become free field elements.
Nothing else in this chip touches them --- they are typed as bytes but range-checked nowhere, because on a moving row @memmove:c:read_value pins them --- so a prover could put anything it liked into RAM, and hence into the committed output.

Three details of the pinning are worth spelling out:
- Only $#`dst` = #`src`$ frees the lanes; at a distance $d$ with $1 <= d <= 8$ a row still pins $#`value`_i = #`value`_(i-d)$ for $i >= d$, and pins the first $d$ lanes against memory it did not write.
  The gap is $8$ because that is the width of the widest row, so a wide row's read range and its write range stay disjoint, and because it is the length of the seed the guest lays down.
- The gate is `is_set` alone, and not `is_set` together with a multiplicity.
  Gating on a product would cost a degree, and this chip stays at degree 2; a padding row leaves $#`is_set` = 0$, and the terminal row satisfies the relation for the same reason every other row does.
- The two constraints are _limb-wise_, and that is stronger than a gap of $8$ on the 64-bit value.
  They admit no carry out of the low limb: $#`src` = (2^32 - 16, 0)$ with $#`dst` = (2^32 - 8, 0)$ satisfies them, but one eight-byte step sends `dst` to $(0, 1)$ while `src` stays in limb $0$, and the successor row satisfies neither.
  An honest `memset` whose range crosses the $2^32$ limb boundary therefore has no satisfying assignment at all.

= The commitment index
On a commitment sequence `dst` is a byte index rather than an address, and `COMMIT` carries it as a `BaseField` (@commit), so @memmove:a:dst is not discharged there.
Two things in this chapter do lean on it, and both degrade rather than break.

@memmove:c:dst_incr invokes `ADDNW`, whose @addnw:a:lhs _is_ that assumption: without it the template no longer forces $#`dst_incr` = #`dst` + #`step`$ over the integers, only the field statement, so a denormalized index admits a spurious carry into the high limb.
And the per-lane addresses of @memmove:c:commit_value_out are built as $#`dst`_0 + i$ without the carry normalisation `MEMW` applies to its own lanes (@memw), so a row with $#`dst`_0 > 2^32 - 8$ addresses its upper lanes outside the low limb.

Neither is a gain for a prover, because a token of that shape has no receiver: the commitment domain is reached by no chip other than this one, and the verifier supplies exactly one $(2, a, 0, dot)$ token per index.
The argument that rules out a ring is unaffected, since it is a statement about `src`, which `COMMIT` does read from `x11`.
Both would be settled at the source by range-checking `index` where it enters `COMMIT`; note that with `count` unbounded on this path, @commit:c:read_index can also write a value past the `Word` range into `x254`.

= The Accelerated Memory Operations standard
The Ethereum Foundation's Accelerated Memory Operations standard fixes what an accelerated `memcpy`, `memmove` and `memset` must provide.
#footnote([Accelerated Memory Operations; eth-act/zkevm-standards. #link("https://github.com/eth-act/zkevm-standards/tree/main/standards/accelerated-memory-operations")[[src]]])

Two of its requirements fall outside this chapter.
The first is behavioural: the accelerated symbol must behave identically to the C library function, which the guest-side stub is responsible for.
The second concerns linking: the symbol must be a strong definition in an unconditionally linked object, or be linked with `--whole-archive`, so that a weak definition elsewhere cannot silently displace it.

What the standard asks of the chip itself is that it accept operands of arbitrary alignment, which it does: no constraint here refers to the alignment of `src`, `dst` or `count`, and a row's width is not tied to any of them.
The one operand restriction this chip does impose is not an alignment: a `memset` may not straddle the $2^32$ limb boundary, as described above.
The standard's fourth operation, `memcmp`, is not covered: it does not copy, so it does not fit this chip, and accelerating it would need a table of its own.

= Notes/optimizations
- `COMMIT` could send its deferral on `MEMMOVE_NEXT` directly, with $#`is_commit` = 1$ in the tuple, which would retire the `COMMIT_DEFER` bus and the `first_ecall` column.
  A commitment sequence would then have no `first` row at all, so `first` would come to mean "entered by `ECALL`" rather than "head of the sequence", and $#`first` dot #`is_commit` = 0$ would have to be added to stop an `ECALL`-entered sequence from writing to the commitment domain.
- `count` need not be a full `DWordWL` on the `ECALL` path: @memmove:c:bound already proves $#`count` < 257$ there, and every later `count` is smaller still.
  The commitment path has no such bound, so this would have to be paid for with a range check where the value enters from `COMMIT`.
- The `value` variable is typed as bytes, but this chip range-checks none of its lanes.
  On a row that moves, they are pinned by @memmove:c:read_value instead: a lane holds whatever the memory argument says resides at that address.
  Only the seven lanes that a one-byte row leaves unused need @memmove:c:tail_lanes, since those never reach the read.
  On a row that moves nothing --- the terminal row, and padding rows --- there is no read, so $#`value`_0$ is an arbitrary field element there.
  That is harmless, because every write carries a multiplicity that vanishes with `end` and so does not fire either.
- @memmove:c:range_src_incr and @memmove:c:range_dst_incr carry multiplicity $#`μ`$, but `src_incr` and `dst_incr` are _consumed_ only at $#`μ` - #`end`$ --- on a terminal row they are pinned by nothing beyond @addnw:c:carry, which holds unconditionally.
  Lowering both to $#`μ` - #`end`$ would drop eight `IS_HALF` lookups on every terminal row at no cost, though not a smaller proof: that table is preprocessed at a fixed height, so the committed cell count is unchanged.
  @memmove:c:range_count_decr genuinely needs $#`μ`$, since @memmove:c:end consumes `count_decr` at that multiplicity.
- A row could move sixteen or thirty-two bytes rather than eight, at the cost of a wider `MEMW` signature.
  It would also narrow the fast path: `MEMW_A` needs every byte of an access to share one old timestamp, which a sixteen-byte row can only manage where the buffer was last written in groups at least that wide, and the widths `MEMW`'s signature can express today are one, two, four and eight.
  Rows saved and cells saved therefore move in opposite directions here, and only the second is what the proof pays for.
- The `memmove` property belongs to one `ECALL`, not to an arbitrarily large guest-level copy: past 256 bytes the stub splits into several `ECALL`s at distinct timestamps, and chunk $k+1$ reads what chunk $k$ wrote.
  That is in-contract for `memcpy`, whose buffers may not overlap; a guest-side `memmove` must keep each copy within one `ECALL` or chunk in the direction that preserves the semantics.
