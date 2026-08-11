#import "/book.typ": book-page, aside
#import "/src.typ": load_config, load_chip
#import "/chip.typ": (
  compute_nr_interactions,
  render_chip_assumptions,
  render_chip_variable_table,
  total_nr_variables,
  total_nr_instantiated_columns,
  render_constraint_table,
  render_chip_padding_table,
)

#let config = load_config()
#let chip = load_chip("src/blake3.toml", config)

#show: book-page(chip.name)
#let blake3 = raw(chip.name)

The #blake3 chip applies the *6-round internal variant* of the BLAKE3
compression function to a 176-byte memory region. It is an internal
Merkle / Fiat–Shamir hash accelerator: one compression digests a 64-byte
message block — exactly one 2-to-1 merge of two 32-byte chaining values —
and produces the full 16-word output (the truncated chaining value is
`out[0..8]`).

⚠ *This is not standard BLAKE3.* The standard function applies 7 rounds;
this chip applies 6 (see @blake3-a6r). No external system will ever agree
on these digests. Standard library implementations (e.g. the official
`blake3` Rust crate) hardwire 7 rounds and cannot compute this function;
the host-side reference implementation lives in
`executor/src/vm/instruction/execution.rs` (`blake3_compress_6round`),
differentially pinned to the validated oracle in
`thoughts/blake3/blake3-oracle/`.

= ECALL interface

ECALL number `-3` (`0xFFFF_FFFF_FFFF_FFFD`). `A0` holds an 8-byte-aligned
pointer to a 176-byte state region of 22 consecutive little-endian dwords:

#table(
  columns: (auto, auto, auto),
  [*dwords*], [*contents*], [*direction*],
  [0..=3], [`h[0..8]` chaining value (2 u32 words per dword)], [read],
  [4..=11], [`m[0..16]` message block], [read],
  [12], [`t` counter (`t_lo` = low u32 → `v[12]`, `t_hi` = high u32 → `v[13]`)], [read],
  [13], [`block_len` (low u32) | `flags` (high u32)], [read],
  [14..=21], [`out[0..16]`], [written],
)

Unaligned or overflowing state addresses are rejected by the executor.
The counter split order (`t_lo → v[12]`, `t_hi → v[13]`) is load-bearing
and was behaviourally verified against the official BLAKE3 crate through
two independent counter paths (oracle audit, 44/44).

= Chip structure

== Columns and interactions

#let nr_variables = total_nr_variables(chip)
#let nr_columns = total_nr_instantiated_columns(chip, config)
#let nr_interactions = compute_nr_interactions(chip)

The formalized I/O-and-range surface below comprises #nr_variables
variables over #nr_columns columns and #nr_interactions interactions; the
full chip has 3,219 main columns and 1,397 interactions (the difference is
the mixing core's `BYTE_ALU[XOR]` lookups, whose SSA operand wiring is
normatively specified by the single-source Rust dataflow and the z3 gate —
see @blake3-scope).

#render_chip_variable_table(chip, config)

== Structure <blake3-scope>

One row per compression call, fully unrolled: 6 rounds × 8 G-functions in
SSA form. The message schedule is a compile-time index permutation (the
`sched` array in `run_flow`, composed from `BLAKE3_MSG_PERMUTATION`), so
every round references the 16 original committed message words — there is
no state or message handoff between rows. I/O follows the KECCAK core idiom: an `ECALL` receiver binds
(timestamp, syscall number), a `MEMW` register read binds the `x10`
pointer, and 22 per-dword `MEMW` operations carry the reads and writes.

Key constraint-design decisions (full rationale:
`thoughts/blake3/blake3-chip/DESIGN.md`, deltas in `IMPLEMENTATION.md`):

- every eval constraint is gated by the multiplicity column $mu$, and the
  maximum constraint degree *including* the $times mu$ factor is 3;
- 3-operand adds commit *two summed carry bits* with an explicit sum
  identity (a ternary carry would be degree 4 after gating);
- 2-operand adds use an expression carry (no committed cell) with a
  $mu$-gated booleanity;
- `rotr16`/`rotr8` are free byte relabels; `rotr12`/`rotr7` are inline
  $mu$-gated Euclidean shift identities whose soundness rests on the
  tight $[0, 2^16)$ bound of the `SLL` halfwords ($2^16$ is invertible
  mod $p$);
- every add/shift output feeds a downstream `BYTE_ALU[XOR]` lookup, which
  is its only byte range check; the message words, the previous
  out-region content and the address bytes are never XOR-consumed and
  carry explicit `ARE_BYTES` checks instead.

The chip's wiring is single-sourced: the compression dataflow is written
once in `prover/src/tables/blake3.rs` (`run_flow`) and interpreted both as
column wiring (constraints + bus senders) and as the u32 witness (trace
fill + lookup multiplicities), so the two cannot diverge structurally.

== Formalized constraints

#render_constraint_table(chip, config, groups: "io")
#render_constraint_table(chip, config, groups: "addr")
#render_constraint_table(chip, config, groups: "range")
#render_constraint_table(chip, config, groups: "mu")

= Verification evidence

The design was taken to a z3-gated model *before* the Rust implementation
(`thoughts/blake3/blake3-chip/z3_blake_verify.py`): the G quarter-round
and the init/feed-forward layout are UNSAT under free inputs, five
negative controls and two field-level bound-necessity controls are SAT,
and the concrete 6- and 7-round pipelines reproduce the oracle's pinned
vectors. Two independent transcription audits
(`thoughts/blake3/TRANSCRIPTION-AUDIT.md`,
`GATE-TRANSCRIPTION-AUDIT.md`) checked the gate against the oracle. The
Rust chip is additionally pinned by the 10 canonical 6-round vectors at
the syscall level and by an end-to-end prove+verify of chained
compressions.

= The 6-round assumption <blake3-a6r>

*A6R.* The BLAKE3 compression function restricted to 6 rounds is
collision-resistant and suitable as a 2-to-1 compression for Merkle
hashing and as a PRF for Fiat–Shamir, in the same sense the full 7-round
function is believed to be. (Precedent: KangarooTwelve's reduced-round
Keccak. Best public cryptanalysis of BLAKE3 reaches far fewer rounds; the
margin removed here is one round of seven.)

*External review (2026-08).* The round-count choice was reviewed with
external symmetric-cryptography experts consulted by the project: removing
*one* round (7 → 6) was judged comfortable; removing *two* (7 → 5) was
explicitly not. Accordingly, 6 rounds is the endorsed floor. Variants
below 6 rounds are not formally ruled out, but they are not available on
the project's own authority: adopting one would require the external
experts to study the reduced-round margin specifically — a dedicated
cryptanalytic review, not an engineering or configuration decision.

Any use of #blake3 as a Merkle or transcript hash *invokes this
assumption*. The z3 gate proves the chip computes 6-round BLAKE3
correctly; it neither proves nor addresses whether 6 rounds are secure.

*The assumption-free alternative.* The chip design is round-parameterised;
a 7-round instantiation (standard BLAKE3 compression, bit-compatible with
official parent-node merges) costs roughly 10–12% more per merge
end-to-end and requires no assumption beyond standard BLAKE3.

*Ordering, reversed 2026-08-10.* The 7-round variant is the primary
target; the 6-round variant is the measured performance variant, kept
behind the round parameter and adopted only if that 10–12% is judged worth
signing A6R for. This reverses the ordering this section recorded before,
and the argument is the reference chain rather than cryptanalysis: at 7
rounds the official crate is a direct external test vector for both the
primitive and its framing, and there is no assumption left to ratify or
defend at audit. It retracts nothing from the external review above — 6
rounds remains the endorsed floor — and it is not a signature on A6R,
which falls due only if the default moves back to 6
(`thoughts/shared/lfm-real-hash/A6R-signoff.md`). The chip specified on
this page is still the 6-round instantiation, so the above is recorded
intent and not a change that has landed here. If both are instantiated
they are distinct chips with distinct ECALL numbers.

= Cost

Measured on the CPU bench box (32 cores, blowup 2, single-epoch
continuations): ≈5,473 compressions/s at ≥#raw("2^17") table rows, ≈7,194
committed cell-equivalents per compression end-to-end (≈5,316 table-only)
— ≈12× the keccak-f permutation per 2-to-1 merge at equal wall time and
memory. Details and methodology: PR \#903.
