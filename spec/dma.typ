#import "/book.typ": book-page, aside
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

#show: book-page("dma.typ")

#let config = load_config()
#let chip = load_chip("src/dma.toml", config)
#let dma = raw(chip.name)

The #dma chip copies a range of bytes from one location in memory to another, that is, it performs a `memcpy`.
#footnote([Linux man-page on `memcpy`; man7.org. #link("https://man7.org/linux/man-pages/man3/memcpy.3.html")[[src]]])
The guest performs such a copy with a RISC-V loop --- per copied doubleword a load, a store, two pointer increments and a branch, each of them a `CPU` row (@cpu) together with its memory operations.
This accelerator replaces all of that with a single row per eight copied bytes.
An accelerated `memcpy` is expected to conform to the Ethereum Foundation's Accelerated Memory Operations standard, which fixes the semantics the symbol must keep and how it must win linking.
#footnote([Accelerated Memory Operations; eth-act/zkevm-standards. #link("https://github.com/eth-act/zkevm-standards/tree/main/standards/accelerated-memory-operations")[[src]]])
Both obligations fall on the guest-side stub and on the link, outside this chapter; what the standard asks of the chip itself is that it assume no particular alignment of `dst`, `src` or `count`, which @dma:c:tail guarantees by choosing the width from `count` alone.

= Variables
#let nr_variables = total_nr_variables(chip)
#let nr_columns = total_nr_instantiated_columns(chip, config)
#let nr_interactions = compute_nr_interactions(chip)

The #dma chip is comprised of #nr_variables variables that are expressed using #nr_columns columns and leverages #nr_interactions interactions:
#render_chip_variable_table(chip, config)

= Assumptions
#render_chip_assumptions(chip, config)

The obligations on `src`, `dst` and `count` concern the _first_ row of a copy sequence only:
every subsequent row receives all three over the `DMA_NEXT` bus, whose sender range-checks them (@dma:c:range_src_incr, @dma:c:range_dst_incr and @dma:c:range_count_decr).
On the first row they are discharged by the register file, outside this chapter.

`timestamp` is different: it is forwarded verbatim by @dma:c:send_copy_next_chunk and never range-checked here.
The first row's `timestamp` is pinned by @dma:c:receive_ecall to the `CPU`'s preprocessed timestamp column (@vars), and every later row inherits it through the bus.
That column also carries $#`timestamp` = 4 dot (i + 1)$, which is what keeps $#`timestamp` + 2$ from leaving the `Word` range in @dma:c:write_value --- `IS_WORD` alone would not.

= Constraints
In this VM, we assign system call number $-3$ to the #dma accelerator.
Since we do not know how many bytes are to be copied, this chip employs the same recursive design as `COMMIT` (@commit):
each iteration copies one chunk of bytes, and recursively "calls" itself to copy the remainder.
As such, only the call from the CPU to this chip (i.e., the `first` in the recursion tree) should accept the `ECALL`; later recursive calls should not.
#render_constraint_table(chip, config, groups: "incoming")

The `memcpy` operation has the following signature:

```c
void *memcpy(void dest[restrict count], const void src[restrict count], size_t count);
```

That is to say,
- `A0` contains the address of the first byte to write,
- `A1` contains the address of the first byte to read, and
- `A2` contains `count`.

@dma:c:read_dst, @dma:c:read_src and @dma:c:read_count read these three registers.
Each of them writes back the value that was read, so the copy leaves the registers untouched;
the guest is responsible for producing `memcpy`'s return value.
Again, these memory interactions only take place when this is the `first` call in the recursion tree.
#render_constraint_table(chip, config, groups: "read_input")

== Chunk width
A row copies eight bytes whenever at least eight bytes remain, and a single byte otherwise.
@dma:c:tail pins that choice to `LT` (@lt): the prover cannot select a convenient partition of the requested range.
A copy of $n$ bytes is therefore laid out as $floor(n \/ 8)$ eight-byte rows, followed by $n mod 8$ one-byte rows, followed by one terminal row.

Because a row is the unit in which this chip charges for a copy, an unbounded `count` would let a single guest instruction append an unbounded number of rows to the trace.
@dma:c:bound therefore proves $#`count` < 257$ on the first row of every sequence, which caps a sequence at $39$ rows --- attained at $n = 255$, not at $n = 256$, since a byte short of the bound trades one eight-byte row for seven one-byte rows.
The guest-side `memcpy` chunks larger copies into multiple `ECALL`s; the executor rejects any chunk exceeding 256 bytes.
#render_constraint_table(chip, config, groups: "width")

#aside("Why a one-byte tail")[
  Selecting between exactly two widths lets `tail` be a single bit, so $#`step` = 8 - 7 dot #`tail`$ stays linear and every constraint in this chip stays of degree 2.
  Splitting the remainder into four-, two- and one-byte chunks instead would shave off at most four rows per sequence, at the cost of a two-bit width selector and a decoding of that selector into `MEMW`'s `write2`/`write4`/`write8` flags.
]

== Performing the copy
The bytes are read from `src` at $#`timestamp` + 1$ and written to `dst` at $#`timestamp` + 2$.
Both interactions are expressed over the _same_ `value` variable, which is what makes the copied bytes equal:
there is nothing to constrain, since there is only one set of columns.
The read carries `value` as both its input and its output, so `value` is pinned to whatever the memory argument (@memory) says resides at `src`.

Splitting the two accesses over two consecutive timestamps is what gives an overlapping copy well-defined semantics:
the memory argument orders accesses to an address by timestamp, so _every_ read of this `ECALL` observes memory as it was before the copy started.
A single `ECALL` hence behaves like `memmove` rather than like a byte-by-byte forward copy.

#render_constraint_table(chip, config, groups: "copy")

@dma:c:tail_lanes canonicalises a one-byte row.
Such a row addresses `MEMW` with $#`write2` = #`write4` = #`write8` = 0$, i.e. it presents a single-byte access.
`MEMW` gates every memory interaction for lane $i >= 1$ on those same width flags (@memw), so the seven unused lanes never reach the memory argument at all;
what they do reach is the `MEMW` tuple itself, and pinning them to zero is what keeps that tuple the canonical encoding of a single-byte access rather than one carrying seven free field elements.

== Advancing to the next chunk
In parallel, we compute $#`src_incr` = #`src` + #`step`$ and $#`dst_incr` = #`dst` + #`step`$ as the addresses at which the next chunk starts, and $#`count_decr` = #`count` - #`step`$ as the number of bytes that still have to be copied afterwards.
The first two of @dma:c:range_src_incr, @dma:c:range_dst_incr and @dma:c:range_count_decr are included to satisfy @addnw:a:sum, and the last to satisfy @sub:a:diff.
#render_constraint_table(chip, config, groups: "incr_decr")

Note the asymmetry between the two address updates and the count update:

+ The addresses use `ADDNW` (@add), which forbids wraparound modulo $2^64$.
  Without it a sequence could walk `src` past the end of the address space and continue at low addresses, touching memory unrelated to the requested range --- and, less obviously, a sequence could close into a ring that balances every bus while copying nothing that was asked for; see the discussion of termination below.
  The condition is $#`μ` - #`end`$: on the terminal row and on padding rows the computed successor is consumed by nobody, since @dma:c:send_copy_next_chunk carries that same multiplicity.
+ The count uses plain `SUB` (@add), which permits wraparound, because the terminal row holds $#`count` = 0$ and hence $#`count_decr` = 0 - 1 = 2^64 - 1$.
  That permission is safe precisely because @dma:c:tail pins $#`tail` = (#`count` < 8)$, so $#`step` <= #`count`$ on every row with $#`count` >= 1$ and the subtraction can only wrap on the terminal row.
  Dropping the pin would make $#`count` = 7$ with $#`tail` = 0$ acceptable, which wraps `count_decr` and thus claims `end` while seven requested bytes were never copied.

== Terminating the sequence
When `count` hits $0$, we should stop performing further recursive calls.
We use the `end` bit to indicate these circumstances.
#render_constraint_table(chip, config, groups: "end")

*Note*:
+ As in `COMMIT` (@commit), we set $#`end` = 1$ when $#`count_decr` = -1$ rather than when $#`count` = 0$, which allows `count` to be stored in a `DWordWL` rather than a `DWordHL`.
+ $forall i in [0, 3]: 65535 - #`count_decr`_i >= 0$ as a result of @dma:c:range_count_decr.
 Hence,
  $
  sum_(i=0)^3 65535 - #`count_decr`_i = 0 arrow.l.r.double.long forall i in [0, 3]: #`count_decr`_i = 65535
  $
  Without those range checks the sum could _vanish_ for a `count_decr` other than $2^64 - 1$ --- one limb above $65535$ compensating another below it, as in $(65534, 65536, 65535, 65535)$ --- and `end` would be claimable at a nonzero count.
  That matters more here than the shape of the constraint suggests: since both memory interactions carry multiplicity $#`μ` - #`end`$, a row that wrongly claims `end` emits no memory operations at all, which is a silently truncated copy with every bus balanced.
+ A copy of zero bytes is a single row with $#`first` = #`end` = 1$: it accepts the `ECALL` and reads the three registers, but emits no memory operations and starts no recursion.

== Chaining the rows
When this was not the last chunk of this sequence, we recursively copy the next chunk over the `DMA_NEXT` bus, specifying the timestamp, the addresses to continue reading and writing at, and the number of bytes that still have to be copied (@dma:c:send_copy_next_chunk).
Since that certainly won't be the `first` call in the sequence, we read `src_incr`, `dst_incr` and `count_decr` from the previous recursion level into `src`, `dst` and `count`, and continue copying.
#render_constraint_table(chip, config, groups: "lookups")

Both tuples carry the `timestamp`, and that is what separates one copy from another:
without it, rows belonging to two different copies could be spliced into each other's sequences while the bus still balances.
Since the CPU's timestamps strictly increase per instruction, no two #dma `ECALL`s share one.

Observe also that this chip has no constraint demanding that a sequence terminates.
It does not need one, but the reason is worth stating carefully, because the obvious counting argument is not sufficient on its own.

Fix a timestamp. Balancing `DMA_NEXT` forces the number of rows claiming `end` to equal the number claiming `first`, and @dma:c:receive_ecall caps the latter at one, since the `CPU` sends a single `ECALL` per timestamp.
So a sequence that simply runs on without ever setting `end` sends one tuple more than it receives, and the bus does not balance.

That argument rules out an _open_ sequence, and nothing more.
It does not by itself rule out a _closed_ one: a ring of rows carrying $#`μ` = 1$ with neither `first` nor `end` set sends and receives one tuple each, so it balances, consumes no `ECALL` at all, and would still emit a read and a write per row.
What forbids the ring is @dma:c:src_incr: `ADDNW` forces $#`src_incr` = #`src` + #`step`$ _over the integers_ with $#`step` >= 1$, so `src` strictly increases along the chain and can never return to a value it already held.
This is the second reason the addresses use `ADDNW` rather than `ADD`, and the more important of the two.

== Bits
Lastly, we must make sure `first`, `end`, `tail` and `μ` are bits, and that either $#`first` = 1$ or $#`end` = 1$ implies $#`μ` = 1$ (@dma:c:first_or_end_implies_mu).
The latter is required to ensure the multiplicities $-(#`μ` - #`first`)$ and $#`μ` - #`end`$ are binary.
#render_constraint_table(chip, config, groups: "bits")

= Padding
To pad this chip, use the below data.
#render_chip_padding_table(chip, config)

Note that this padding row is not all-zero.
@dma:c:count_decr is unconditional, so a padding row has to satisfy it too: $#`tail` = 1$ makes $#`step` = 1$, which $#`count` = 1$ and $#`count_decr` = 0$ then satisfy.
The two address updates are conditioned on $#`μ` - #`end`$ and so do not forbid a wraparound here, but their low-limb carry is constrained on every row (@addnw:c:carry), so a padding row must satisfy that relation too; $#`src_incr` = #`dst_incr` = 1$ is the assignment that does so with a zero carry.
It is not the only one --- @dma:c:range_src_incr and @dma:c:range_dst_incr, which would pin the limbs, carry multiplicity $#`μ`$ and are inert here --- but a padding row feeds no interaction either way.

= Notes/optimizations
- The copy is a `memmove` per `ECALL`, but _not_ per guest-level `memcpy`: a copy larger than 256 bytes is chunked into several `ECALL`s at distinct timestamps, and chunk $k+1$ reads what chunk $k$ has already written.
  This is in-contract for `memcpy`, whose buffers may not overlap, but the `memmove` property must not be claimed at the guest level.
- The `value` variable is typed as bytes, but this chip range-checks none of its lanes.
  On a row that copies, they are pinned by @dma:c:read_value instead: a lane holds whatever the memory argument says resides at that address.
  Only the seven lanes that a one-byte row leaves unused need @dma:c:tail_lanes, since those never reach the read.
  On a row that copies nothing --- the terminal row, and padding rows --- there is no read, so $#`value`_0$ is an arbitrary field element there.
  That is harmless, because @dma:c:write_value carries the same multiplicity and so does not fire either.
- @dma:c:range_src_incr and @dma:c:range_dst_incr carry multiplicity $#`μ`$, but `src_incr` and `dst_incr` are constrained and consumed only at $#`μ` - #`end`$.
  Lowering both to $#`μ` - #`end`$ would drop eight `IS_HALF` lookups on every terminal row at no cost.
  @dma:c:range_count_decr genuinely needs $#`μ`$, since @dma:c:end consumes `count_decr` at that multiplicity.
- `count` need not be a full `DWordWL`: @dma:c:bound already proves $#`count` < 257$ on the first row of a sequence, and every later `count` is smaller still.
  Representing it as a single `Word`, or even as a `Half`, would save a column and shrink both `LT` interactions --- at the cost of an extra range check where the value enters from the register.
- A row could copy sixteen or thirty-two bytes rather than eight, at the cost of a wider `MEMW` signature.
  For a copy of exactly $256$ bytes the row count would drop from $33$ to $17$ and $9$.
  The _worst case_ over all admissible lengths behaves quite differently, however, since it is attained at $n = 255$ rather than at $n = 256$: it is $31 + 7 + 1 = 39$ rows today, $15 + 15 + 1 = 31$ with sixteen-byte rows, and $7 + 31 + 1 = 39$ again with thirty-two-byte rows.
  Widening the chunk moves work out of the wide rows and into the one-byte tail, so past sixteen bytes it buys nothing at all where it matters.
