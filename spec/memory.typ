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

#let config = load_config()
#let chip = load_chip("src/page.toml", config)

#show: book-page("memory.typ")

As part of fully proving the correct execution of a RISC-V program,
the VM must ensure that memory reads and writes are consistent.
That is, every byte read from some address corresponds to the byte that was last written to that address
--- or the initial value if nothing has been written yet.
We consider "memory" in a broad sense here:
both RAM and the general purpose registers can be seen as instantiations of memory
and are therefore handled simultaneously.
#footnote[
  While RAM is byte addressed, we do choose to store registers as a `DWordWL` over two word addresses.
]
In particular, our memory addressing scheme will consist of two parts: a domain separator, and an address within the domain.
For specific domains and domain separators, we use the following assignment.

/ RAM memory: $0$
/ Registers: $1$
/ Committed values: $2$

On a high level, we ensure memory consistency by an interacting system of
reads and writes to a lookup argument, combined with an initialization and finalization scheme.
The initialization and finalization schemes together ensure both that (1) the necessary preconditions
for the lookup system are satisfied, and (2) the program is executed with the correct
initial memory and register contents as specified by the ELF binary and the ISA.

= Memory types

A commonly made distinction of memory types is that of _read-only_ and _read-write_ memory,
with the more restrictive read-only variant often allowing for more efficient solutions
(be that regarding prover time, verifier time or proof size) via table lookup proofs.
Naturally, the VM’s main memory and registers should be handled by a read-write system
as the guest program/environment can issue instructions that write to memory.
While there are some subsystems that can be modelled as read-only memory
---e.g., the program memory and instruction decoding---
we opt to integrate these into the proof system via chip interactions (relying on techniques derived from table lookup arguments).
As such, we only concern ourselves with read-write memory, moving forward.

= Memory operations

Every memory operation has some conceptual attributes that are relevant to mention or discuss:

- The type of operation (read or write)
- The memory address --- this is an address in the broad sense:
  main memory and registers have their own dedicated part of the unified address space.
- The value being read from or written to the memory address
- When the value was read or written, see the below paragraph

Since we will have to ensure that memory accesses are temporally consistent within the execution of the VM,
we additionally consider a _timestamp_ for  every memory access, that should be strictly increasing.
As such, it should never be possible for the system to generate accesses to the same address at identical timestamps.
Multiple memory accesses can (and indeed will, consider e.g. register reads) occur in a single execution cycle of the VM,
so we cannot use the cycle counter directly as timestamp for register accesses.
We can, however, statically bound the maximal number of memory accesses made during a single execution by a granularity constant $k$
and derive timestamps from the cycle counter.
The $i$th possible memory access in cycle $c$ will obtain as timestamp the value $k dot c + i$.
For simplicity, we will always reserve a timestamp for every possible memory access, and leave the timestamp unused if an instruction does not use it.


#aside(ref: <memory:aside:granularity>)[Note on "simultaneous" memory accesses][
  For reasons of completeness (since temporal integrity as discussed below is a security necessity),
  we cannot deal with multiple accesses to the same address at identical timestamps.
  However, if multiple accesses are guaranteed to be independent (that is, to different addresses), they can still share a timestamp
  --- consider, e.g., the case of reading a word as 4 bytes with the `LW` load instruction.
  This property is already taken into account where possible in the design of the system.
  For instance, in the CPU chip, we can ensure that there are at most 3 memory accesses not guaranteed
  to be independent, so a timestamp granularity of 4 timestamps per cycle is enough.
]


= Permutation argument

We can conceptually organise the state of the memory as a collection of "tokens" that represent tuples
$(serif("timestamp"), serif("address"), serif("value"))$,
meaning the current value written to $serif("address")$ is $serif("value")$,
last written to memory at $serif("timestamp")$.
Having exactly one value associated with any address will be ensured (see further down in this document)
by the interaction of memory initialization, memory finalization, and the effects of memory operations.

Each memory operation will then do two things:

- Consume the current token in the memory
- Emit a new token to replace it

Naturally, for a read operation, the _values_ embedded in the consumed and emitted tokens must be identical.
From the need to consume a token even on the first memory access,
we can see the necessity for a memory initialization procedure
---in addition to having to make sure the initial memory content lines up with what the binary dictates.

So long as we can properly constrain temporal integrity (that is, no memory operation can consume future tokens),
this "balancing" act of tokens can be integrated (with sufficient domain separation) into the existing LogUp argument (@logup):
consuming a token corresponds to a "receive" and emitting a new token is a "send".

= Temporal integrity

To ensure temporal integrity, every memory operation needs to be constrained for the newly emitted token
to have a strictly greater timestamp than the consumed token.
This raises the question of how to represent timestamps and cleanly perform this check,
as over a finite field the “less than” relation is ill-defined
(though it is common and natural to consider it as the less than relation over the natural lift of the field into the integers).
We choose to represent timestamps as 32-bit words, using the existing `LT` chip (@lt) functionality for comparisons.
The full implementation of the timestamp system can be seen in the `timestamp` column of the `CPU` (@cpu) and `MEMW` chips (@memw).
The `CPU` merely passes in the current timestamp, while `MEMW` can recall the previously written timestamp and constrain the correct sequencing.

#aside[Note on options and trade-offs for timestamp representation][
 #grid(columns: (1fr, 1fr), gutter: 1em)[#align(center, emph[Machine word])][#align(center, emph[Field element])][
    - Clean definition of “less-than”, using the already existing `LT` functionality in the ALU
    - Harder to perform increments, needing extra constraints beyond field arithmetic
      - But this can be alleviated by providing a precomputed column that has a fixed increment per CPU row
  ][
    - Comparison is more annoying, but can work by:
      - Decomposition into a machine word and chip interaction with the LT chip
      - Bit decomposition and comparison constraints
      - Range-checking the difference to be sufficiently small w.r.t. the field characteristic.
    - Increments and basic arithmetic operations are cheap
  ]
]

= Initialization and Finalization

Because the LogUp argument handling token consumption and emission needs to be fully balanced
--- every token emitted should be consumed, and vice versa ---
we need to have a system to emit the initial tokens and consume the final tokens.
This needs to ensure that every address has at most a single initializing emission, and at most one finalizing consumption.
Having at most one initialization will, through the correctness of the lookup argument,
immediately lead to having at most one correct finalization, and vice versa.

The initialization will need to correspond to a fixed initial register state for the VM,
as well as the memory loaded from the program binary, zero-initialization of memory elsewhere, and private input provided by the prover.
The contribution of initialization with static data from the ELF executable and the initial register state to the sum
can be handled directly by the verifier, ensuring correctness corresponding to the ELF binary being proven.
To enable the loading of the PC in @cpu:memory, register initialization happens at timestamp 1.
Register finalization is made possible for the verifier by having a known state from the HALT chip (@halt).
This leaves only zero-initialization and prover input as prover-side concerns for initialization,
alongside the finalization of the entire used memory.

For our chosen scheme (which we refer to as "paged initialization/finalization"),
the available memory range is split into equally (power-of-two) sized "pages".
Each address can then be represented as `address = page_base_address + page_offset`,
with `page_base_address` being "page-aligned", and `page_offset` belonging to a limited range (the page size).
As such, initialization or finalization of a page is represented by a table with columns `page`, `offset`, `value`, and ---for finalization--- `timestamp`.
The `page` column is a preprocessed, constant value (which can be entirely virtualized/inlined into the constraints for this table),
and the `offset` column is a preprocessed column containing its row index.
Depending on the type of initialization, `value` can be a prover-committed column (input data), or a precomputed, constant column containing `0` (free memory space).
This table then feeds into the LogUp system in the normal way,
emitting the initial tokens for all addresses in a page, without consuming any tokens.
Since the `offset` column is always the same, it can be reused across all paged initialization and finalization tables.

Due to its interaction with the epoch-based proving system from @streaming,
we defer a concrete instantiation and description of this scheme as an AIR table until then.


#aside[Note on alternatives and trade-offs][
  We identify a few alternatives that would achieve the desired initialization/finalization functionalities, and consider their respective trade-offs.

  _"Free-zero" initialization_

  Zero-initialization could be achieved by allowing the `MEMW` chip to output a zero
  without consuming a token from the lookup argument.
  This would in turn be made secure by finalization consuming at most one token per address:
  if an address is initialized more than once, the proof cannot be finalized.
  - This requires fewer pages (and hence tables) for zero-initialization.
  - But it comes at a cost of added complexity in the `MEMW `chip, and likely some extra columns to handle this.
    Keeping track of initialized addresses, and potentially having to initialize only some of the bytes in a word-read
    may make bookkeeping challenging.
  - This is an alternative form of sparse initialization (see below), so it is incompatible with paged finalization.
    Paged finalization can be made into a compatible sparse form by adding a bit-checked multiplicity column.

  _Sparse initialization/finalization_

  One or more STARK tables (depending on the amount of memory used) consisting of `(address, value)` columns are introduced,
  where for zero-initialization, `value` can be constant zero.
  Transition constraints ensure that `address` is strictly increasing, enforcing the "at most once" property;
  `value` is range-checked to consist of bytes.
  Similar to paged finalization, an additional `timestamp` column is added, containing the final timestamp each address was accessed.
  This table is then further used to contribute to the LogUp sum as with any other interactions.
  - The transition constraints can be chosen to only apply on finalization, as at-most-once finalization is enough to ensure consistency.
  - Sparse initialization is incompatible with paged finalization, see also the remark under free-zero initialization above.
  - This would require transition constraints, which currently are not needed elsewhere in the VM design
    - Additionally, for memory use exceeding the capacity of a single initialization/finalization table, some form of transition constraint between tables is needed
    - Alternatively, transition constraints could potentially be avoided by more integration into the LogUp system, but this could turn out more costly in practice
  - This is compatible with the above "free zero" initialization
  - Since a prover-committed address column is needed (rather than a precomputed one), the number of required columns increases.
    - As an optimization, the address column could potentially be used simultaneously for initialization and finalization
  - Sparse initialization/finalization reduces the cost for sparse memory access patterns,
    where only a few addresses would be accessed per page.
      Most programs and compilers should however favor a memory locality that makes paged initialization/finalization comparable.
]

== Register initialization/finalization

The initial and final state of registers can be entirely known by
the verifier, since the relevant initialization values are either zero,
or embedded in the ELF, and the final values can be set to a known value
by the `HALT` ecall (@ecall).
As additionally, the number of registers is small, the verifier can directly
add the required balancing terms to the LogUp sum.

= Notes and considerations

- Register reads and writes may interact within a single cycle, so a correct and fixed ordering needs to be ensured
- Correctness of initialization and completeness of finalization need to be ensured

= Future topics of interest

- Optimize memory systems after determining factual bottlenecks (e.g. taking inspiration from Twist and Shout, or other recent research)
