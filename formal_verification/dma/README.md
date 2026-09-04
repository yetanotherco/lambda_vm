# DMA memcpy — oracle + z3 gate

Machine-checks that the DMA memcpy chip (PR #874) copies the bytes it claims to.
Same home and shape as `formal_verification/keccak/` (PR #923): one flat
directory per verified chip, one README carrying the method, a committed run log.

**Verification code only — no constraint, trace or performance change.** The one
exception is `prover/src/tests/dma_tests.rs`, which consumes this directory's
emitted fixture, plus a `#[cfg(test)]` accessor in `prover/src/tables/trace_builder.rs`
and a `verify-dma` target in the `Makefile`.

```sh
make verify-dma          # all three, from the repo root
pip install z3-solver    # the only dependency; validated on 5.0.0
```

**The degraded-run contract**, because a CI job or a human greps for the word
`VALIDATED` and a partial run must not print it bare:

| exit | meaning |
|---|---|
| 0 | full board green, or `--quick` green (status token says `VALIDATED (--quick, reduced sweeps)`) |
| 1 | a real failure — a mutant survived, an anchor disagreed, or a solver returned `unknown` |
| 2 | **ran but degraded**: an external anchor was unavailable. Status token says `PARTIALLY VALIDATED (n anchor(s) skipped)` |

A missing dependency SKIPs only its own anchor and never cascades, the banner
names the anchors it is *not* anchored on, and **a mutant whose target anchor
skipped is reported `NOT RUN`, never `caught`** — scoring a skip as a catch
credits a mutant to a check that never ran, which is exactly what happened until
it was fixed: an unloadable libc printed `PASS all 8 mutants caught` while
running six. `make verify-dma` tolerates exit 2 and aborts on exit 1.

The gate scores `unknown` as **failure** everywhere. On z3 5.0.0 the full board
is ~92 s; on 4.12.2 it takes ~1210 s and two queries blow their budgets, so a
solver timeout is reported as `TIMEOUT` and distinguished from a rejection — an
earlier version would have printed "the AIR REJECTS an honest row" on a slow box.

| file | what it is |
|---|---|
| `dma_ref.py` | the oracle: four levels of reference model, no repo code |
| `test_ref.py` | anchors the oracle against libc and CPython; emits the fixtures |
| `tamper_test.py` | are those anchors sensitive? eight deliberate defects |
| `z3_verify.py` | the gate: field-exact model of the AIR |
| `audit_transcription.py` | 104 claims tying the gate and the Rust together |
| `verify.log` | the gate's own output, committed |
| `canonical_dma_rows.txt` | pinned vectors, geometry only; `include_str!`-ed by the Rust row-decomposition test |
| `canonical_dma_vectors.json` | the same vectors with all 32 committed columns per row; `include_str!`-ed by the Rust column test |

## Results

```
oracle:  [1] libc memmove                  PASS  3855 cases x overlap/alignment
         [2] CPython slice assignment       PASS  3855 cases
         [3] row/bus level <-> byte level   PASS  257 lengths x 15 overlaps
         [4] guest stub chunking            PASS  1100 lengths
         [5] tamper tests                   PASS  8/8 mutants caught
         VALIDATION STATUS: VALIDATED

gate:    layer 1 (row semantics)            PASS  6/6 UNSAT
         layer 2 (chain structure)          PASS  4 integer + 2 field-exact UNSAT
         layer 2 controls                   PASS  4 positive + 3 negative
         negative controls                  PASS  10/10 SAT
         width audit (bound necessity)      PASS  6/6
         completeness sweep                 PASS  5153 honest + 257 padding rows
         OVERALL: PASS                            (~92 s on z3 5.0.0)

audit:   104 claims, 0 findings; mutation-tested against 17 source mutants
rust:    cargo test -p lambda-vm-prover --lib tests::dma_tests   11 passed
```

Full gate transcript in `verify.log`, which is the gate's own stdout, **pasted not
retyped** — including the `solver:` line, which an earlier version silently
dropped while the surrounding sentence claimed the block was verbatim. The
`## Results` block above is hand-copied from it and nothing enforces that; if you
regenerate, replace both.

**`--lib dma` is 19 tests, but 7 need guest ELFs** — run
`make compile-programs-rust` first (RISC-V target) or use the narrower filter
above. `make verify-dma` runs no cargo.

## The method

Four levels, each checkable against the next, so no level is trusted on its own.

1. **Byte semantics** (`memcpy_ref`) — the C `memmove` contract. Anchored against
   the platform libc and CPython slice assignment, 3855 cases each over every
   length `0..256` × 15 overlap configurations. Genuinely non-circular: neither
   shares code with this model or with the other.
2. **Row decomposition** (`row_decomposition`) — the row sequence the AIR is
   obliged to contain. Written as the greedy loop the AIR actually performs
   (`tail = count < 8`), *not* the closed form; that they agree is checked, not
   assumed.
3. **MEMW multiset** (`memw_ops`) — three register reads at `T`, every source read
   at `T+1`, every destination write at `T+2`. `replay_memw` runs this back down
   to level 1 and raises if a read's recorded value disagrees with memory, so a
   mis-ordered op list fails loudly instead of quietly producing the right answer.
4. **Guest chunking** (`chunk_ecalls`) — the `memcpy` stub's loop, `min(remaining, 256)`.

The gate then models the AIR itself: **every committed column a free Goldilocks
element, every constraint an equation mod p, every lookup modelled as the
constraints of the table that receives it** — not as its advertised contract.
Assert `output != oracle(input)` and ask z3: UNSAT means the row does what the
oracle says; SAT is a counterexample. A bit-vector model cannot do this job — the
question is whether a range check is *missing*, and BV silently bounds an
unconstrained column, so the bug disappears.

Two layers: field-exact single/paired rows prove the row abstraction, then a
multi-row layer takes that abstraction and models `DmaNext` as a **free bijection**
between senders and receivers rather than an assumed chain — which is where
"a source row skipped forward", "the copy ended early" and "a disjoint cycle also
balances the bus" get answered.

## The oracle's independence, and its limits

Anchors 1 and 2 are genuinely non-circular: the platform C library and CPython's
`bytearray` slice assignment are two `memmove` implementations sharing no code with
`dma_ref.py` or with each other. libc in particular is the definition the guest's
`compiler_builtins` `memcpy` was replacing, which makes it the right anchor rather
than a convenient one. Anchor 3 is the one the chip depends on and has no external
counterpart — it is the only check that the row sequence the AIR proves is the byte
copy the guest asked for. Anchor 5 (the tamper sweep) is what makes 1–4 worth running.

**The one thing the reference is not independent of: the row decomposition itself.**
The greedy `8-while-≥8-then-1` rule is a design decision, and the oracle transcribes
it from the same place the AIR gets it. What the oracle proves is that this
decomposition *implements the byte copy*; it cannot tell you the decomposition is
the right one, and it would not catch a design where both the AIR and the oracle
chunked differently but consistently. That is why the Rust test
`dma_trace_matches_oracle_row_decomposition` exists — it pins the trace builder's
decomposition against the emitted vectors rather than against a recomputation.

Known limitations, carried forward verbatim in intent:

- **O1 — the model does not model the memory table.** `replay_memw` enforces read
  faithfulness at its own timestamp; it does not model per-address ordering of
  *multi-byte* accesses, unaligned 8-byte operations, or the `Memw` width decode.
- **O2 — the per-ecall snapshot is not a `memcpy`-level `memmove`.** Chunk *k+1*
  reads what chunk *k* wrote. Agrees with `memmove` for `dst < src` and for
  disjoint ranges; disagrees for `dst > src` with an overlap wider than 256 bytes.
  In contract for `memcpy` — anchor 4 deliberately excludes overlap for this
  reason — but the claim "the DMA ecall has memmove semantics" must not be repeated
  at the C level.
- **O3 — `value` bytes are modelled as integers, not range-checked bytes.** The
  oracle emits `0..255` because it reads them out of a byte memory; the AIR gets its
  byte range from the `Memw` receiver, outside both oracle and gate.
- **O4 — the anchors test the semantics, not the executor.** `dma_ref` is a model of
  `execution.rs` checked against libc; that `execution.rs` matches it is covered by
  the PR's own 256-case proptest and `executor/src/tests/dma_tests.rs`, not by
  anything here.
- **O5 — register reads are modelled as three ops at `T` and nothing more.** The
  old-value/old-timestamp fields, and the fact that the DMA table *writes back* the
  same value it read, are not modelled. Not part of the copy semantics, but part of
  the trace — the audit script is the only thing looking at them.

### The canonical vectors

Ten cases, chosen so every structural case appears exactly once. Regenerated by the
harness into `canonical_dma_vectors.json` and `canonical_dma_rows.txt`.

Only the second one is consumed: `dma_tests.rs` embeds it with `include_str!`, and the
transcription audit checks that it does. The JSON is a human reference — the full
per-row column expansion, for reading when a column's expected value is in question.
**It is not the gate's input**: `z3_verify.py` performs no file I/O at all, and builds
every column as a free field element. Nothing reads the JSON, so nothing would notice
if it went stale; the one guard is that emission is byte-stable, so regenerating it
leaves the tree dirty when it has drifted.

| name | dst | src | n | rows | MEMW ops |
|---|---|---|---|---|---|
| empty | 0x1000 | 0x2000 | 0 | 1 | 3 |
| single byte | 0x1000 | 0x2000 | 1 | 2 | 5 |
| one wide row | 0x1000 | 0x2000 | 8 | 2 | 5 |
| wide plus tail | 0x1000 | 0x2000 | 9 | 3 | 7 |
| widest tail | 0x1000 | 0x2000 | 7 | 8 | 17 |
| unaligned body and tail | 0x2005 | 0x1003 | 27 | 7 | 15 |
| forward overlap | 0x3004 | 0x3000 | 24 | 4 | 9 |
| backward overlap | 0x3000 | 0x3004 | 24 | 4 | 9 |
| page crossing | 0x0FFC | 0x1FFC | 16 | 3 | 7 |
| maximum chunk | 0x1000 | 0x2000 | 256 | 33 | 67 |

"widest tail" is the expensive shape: eight rows to move seven bytes. "maximum
chunk" is the only case with **no tail row at all** — 256 is 8-aligned — which is
why it is pinned. The two files exist separately because the prover crate has no
JSON parser, and a hand-rolled scanner over nested JSON is the fragile coupling that
goes stale silently: the first attempt broke on `rows[i].columns` repeating the
`src`/`dst`/`count` keys. An earlier version hand-transcribed 7 of the 10 vectors
into Rust literals with nothing enforcing the transcription.

## The chip, as recovered from the implementation

`spec/dma.typ` (PR #931) is the normative chapter; this section is what the gate
checks and agrees with it except where noted.

**Row layout.** A row copies eight bytes while `count ≥ 8`, otherwise one byte;
the design is cloned from `commit.rs` — one row per chunk, rows chained by a bus
rather than by a transition constraint. Not one row per byte: 257 rows per
maximal ecall instead of 33. Not one row per copy: the row would need `n` value
columns for unbounded `n`, and 8 bytes is the widest operation `Memw` serves. The
1-byte tail rather than 4/2/1 halving because a `tail` **bit** selects between
exactly two widths, so `step = 8 − 7·tail` stays linear and every constraint stays
degree 2; halving would need a two-bit selector and a width-to-`w2/w4/w8` decode.

The cost, stated precisely (an earlier draft had this wrong in both directions):
`n` bytes takes `n/8` wide rows, `n % 8` tail rows and one terminal row. So tail
rows are 0% for the 8-aligned lengths that dominate, `7/39 = 17.9%` at `n = 255`
(the longest length that *has* a tail — the maximal `n = 256` has none), and
**`7/8 = 87.5%` at the genuine worst case `n = 7`**, where every data row is a
tail row. Against 4/2/1 halving the delta is at most 4 rows, not the 7 an earlier
draft claimed by comparing against a zero-tail design instead of against the
alternative it was arguing with.

**32 columns.** `timestamp` (DWordWL), `src`/`dst` (DWordWL), `src_incr`/`dst_incr`/
`count_decr` (DWordHL — the halfword split exists *because* they need
`IsHalfword`), `count` (DWordWL), `first`/`end`/`tail`/`mu` (Bit), `value[8]`.
(The spec says 31, typing `timestamp` as a `Word`; the Rust carries two limbs. A
spec-wide convention, not a DMA discrepancy.)

**18 constraints, all degree 2.** Machine-measured, not declared:
`constraint_set_tests_b.rs` runs `check_table` which tree-measures every emitted
expression against `max_degree()`.

| idx | constraint |
|---|---|
| 0-3 | `first`, `end`, `tail`, `mu` are bits |
| 4 | `(first + end)·(1 − mu) = 0` |
| 5-6, 7-8 | `src`/`dst` `+ step`, no `2^64` wrap on active non-terminal rows |
| 9-10 | `count_decr + step = count` — **the plain pair; wrap permitted** |
| 11-17 | `tail · value[i] = 0` |

**23 bus interactions.** 1 `Ecall` receive (`first`), 2 `DmaNext` (`mu−end` send /
`mu−first` receive), 12 `IsHalfword` (`mu`), 1 `Zero` (`mu`), 3 `Memw` register
reads (`first`), 2 `Alu` LT, 2 `Memw` data ops (`mu−end`).

### Soundness ledger — ten spots a change must not touch

1. **`DmaNext` carries the timestamp in both tuples.** Removing it splices two
   calls' rows into each other's chains with the multiset still balancing — the
   failure the BLAKE3 design review found the hard way.
2. **`IsHalfword` on all twelve halfwords.** Each one is either an end-detection
   forgery or a wrapped address.
3. **`emit_add_pair` (plain) on `count`, `emit_add_pair_no_overflow` on
   `src`/`dst`.** Swapping either direction breaks the terminal row or admits an
   address wrap. Detail below.
4. **`mu − end` on both data `Memw` sends.** Detail below.
5. **The `Alu` width pin.** Gone, the prover partitions at will and
   `count = 7, tail = 0` truncates seven bytes.
6. **`tail · value[i] = 0`.** Gone, a one-byte row carries seven unconstrained
   field elements into the `Memw` bus.
7. **The bound constant is taken from the executor.** `dma.rs` imports
   `DMA_MEMCPY_MAX_BYTES as EXECUTOR_DMA_MEMCPY_MAX_BYTES` rather than restating
   `257`, so the bound the AIR proves cannot drift from the bound execution
   enforces. Restating it invites exactly that drift; the audit checks it stays
   imported.
8. **One `first` per timestamp**, supplied by the `Ecall` receiver against the
   CPU's single send — two heads at one timestamp would unbalance `Ecall`. This is
   what makes Layer 2's "exactly one head row" a real scope restriction rather
   than an arbitrary one: the bus is what enforces it, outside the gate's model.
9. **The single-`end` obligation is implicit**, not a constraint: a chain with no
   terminal row has one more `DmaNext` send than receive, so the bus does not
   balance. Worth knowing that this is where termination comes from — and that
   bus counting alone rules out only an *open* chain, never a closed cycle. That
   is item 3's second job.
10. **The `DmaNext` tuples' element alignment.** Both are 8 bus elements and align
    pairwise; that is what makes the link a per-limb binding and hence what
    supplies the range provenance for every non-head row. A packing change on
    either side silently changes what the bus binds. Audit §G, and see
    "The retracted finding".

Three of these are asymmetries that will look like untidiness to the next reader:

- **Constraint 9-10 is the plain add pair on purpose.** The terminal row holds
  `count = 0`, `count_decr = 0 − 1 = 2^64−1`; a no-overflow form would reject it.
  MAIN 2 is exactly the statement that this permission is safe — the subtraction
  can only wrap where `end = 1`, because `tail` is pinned to `count < 8`.
- **Constraints 5-8 are the no-overflow form on purpose**, and for two reasons.
  The obvious one: a chain could otherwise walk `src` past `2^64` and continue at
  low addresses, which the executor rejects and the AIR would not. The more
  important one: `src_incr = src + step` over the **integers** with `step ≥ 1`
  makes `src` strictly increase, so no bijective matching can close a **ring** —
  a cycle of `mu=1, first=end=0` rows would otherwise send and receive one tuple
  each, balance every bus, consume no `Ecall`, and still emit a read and a write
  per row. The counting argument alone ("no terminal row ⇒ one more send than
  receive") rules out only an *open* chain.
- **`mu − end` on both data `Memw` sends.** This is what makes a wrongly claimed
  `end` a *silent* truncation rather than an unbalanced bus, and it is why end
  detection carries the weight it does.

**End detection.** `end = 1` iff `4·65535 − Σ count_decr_i = 0`, i.e. iff all four
halfwords are `0xFFFF`. One `Zero` lookup instead of four. It rests on the
`IsHalfword` bounds — drop them and `(0xFFFF+d, 0xFFFF−d, 0xFFFF, 0xFFFF)` reaches
the same sum with a different `count_decr`, so `end` becomes claimable at a nonzero
count — and on the receiving table's **domain**: `bitwise.rs` serves `Zero[v]` only
for `v < 2^20`, and the send lands in `[0, 262140]`.

**Value binding is structural, not proved.** The read and write tuples are built
from the same `value_columns()`; nothing needs to prove `read == write` because
there is one set of columns. On a one-byte row lanes 1-7 must be zero
(constraints 11-17) so the `Memw` tuple is the canonical single-byte encoding —
note this is *canonicalisation*, not a forgery closed: `memw.toml` gates the
per-lane memory tokens on `w2`/`w4`/`write8`, so on a tail row those lanes never
reach the memory argument at all.

**Why the value lanes carry no range check of their own**, since this is the kind
of gap that is usually a bug. The reason is **not** that the receiving table
checks them. `spec/memw.typ:42-45` declines to: *"these properties are necessary
for the consistency of the system as a whole"* — i.e. somebody must, and it says
who only in the negative. That obligation is **A5** below.

The actual argument is narrower and specific to this chip: the `T+1` read tuple
carries `old == value` **against real memory**, so each lane is pinned to whatever
byte the memory argument says is at that address. The lanes are not free field
elements; they are whatever memory already held. This chip *propagates* byte-ness
rather than establishing it.

> **Do not cite `keccak.rs` as authority for relying on the receiver here.** An
> earlier draft did, and it is backwards: `keccak.rs:355-378` emits four
> `AreBytes` senders for its address bytes **precisely because** the receiver does
> not pin them, and its comment spells out the forgery — keeping a linear
> combination's field value correct while encoding non-byte values in the
> individual cells. That file is evidence for the opposite conclusion.

**Overlap.** All reads at `T+1`, all writes at `T+2`, both AIR constants, giving
snapshot (`memmove`) semantics per ecall. **Not** per guest `memcpy`: chunk *k+1*
reads what chunk *k* wrote, so a copy over 256 bytes is a forward copy. In
contract for `memcpy`, but the `memmove` property must not be claimed at the C level.

## Assumptions

Obligations on the **caller**, not checks this chip performs.

| id | assumption | discharged by | status |
|---|---|---|---|
| **A1** | `src`/`dst` limbs are 32-bit wherever a `Memw` data op fires | `spec/src/memw.toml`: `[[assumptions]] IS_WORD[base_address[i]]`. `memw.rs` justifies its own bound via *the CPU table*; DMA is a non-CPU sender | not discharged locally, and **not exhibited**: `memw_addr32` is the only one of the eleven premises with no negative control, and dropping it leaves `check_row`, `check_end_detection`, `check_wrap_only_terminal`, `check_tail_lanes` and `check_row_budget` all `unsat`. So within what the gate models, A1 buys nothing. The claim that it is load-bearing on the head row, and derived elsewhere by the `DmaNext` link, is an argument this gate does not check — unlike A2, which `drop_reg32` exhibits |
| **A2** | `count` limbs are 32-bit on the head row | `spec/src/memw_register.toml`: `[[assumptions]] IS_WORD[val[i]]` | not discharged locally; `drop_reg32` shows what it buys — without it the `count < 257` lookup caps only a residue class |
| **A3** | `ts₀ + 2` stays in `Word` range | the CPU's timestamp stride is 4 and `T = 4i+4`, so it cannot carry | not discharged locally; benign at the current stride |
| **A4** | two DMA ecalls never share a timestamp | CPU timestamps strictly increase per instruction | holds by construction; it is what the `DmaNext` timestamp binding relies on |
| **A5** | domain-0 cells hold bytes | nothing here, and nothing in `memw.rs` — `spec/memw.typ:42-45` assigns it to "the consistency of the system as a whole" | **not discharged, and not DMA's to discharge.** This chip *propagates* byte-ness rather than establishing it |

`spec/` has a broader problem: `IS_WORD` appears across 12 chapters **exclusively**
inside `[[assumptions]]`, never as an interaction or template, and no 2³² table
exists. So the spec asserts a range obligation for nearly every address, register
value and timestamp in the VM without naming a discharger. An obligation owned by
everyone is owned by nobody.

## Padding

`generate_dma_trace` pads to the next power of two, minimum 4, and the row is
**not** all-zero: constraints 9-10 are unconditional, so `count = 1`, `tail = 1`
(hence `step = 1`), `count_decr = 0`, and `src_incr = dst_incr = 1` because
`ADDNW`'s low-limb carry is constrained on every row. `mu = 0` kills all 23
interactions; constraint 4 then forces `first = end = 0`. The gate's completeness
sweep pins exactly this row.

## What the gate proves, and what it does not

**Proves**, given the modelled contracts and given that bus balance means multiset
equality: every satisfying assignment of one row does what the oracle says; among
groups with exactly one head row, the only bus-balanced multi-row structure at
depth ≤ 5 is a single chain tiling `[src, src+n)` exactly once with the greedy
widths; ten of the eleven modelled premises are individually necessary — each has a
negative control (`drop_*`) that returns SAT, i.e. exhibits a concrete forgery, when
that premise alone is removed, and three of the ten are spelled out in full below;
Layer 2's premise set is satisfiable and sensitive to each field of the bus tuple;
and the AIR accepts every honest trace for every length `0..256`.

The eleventh, `memw_addr32` (assumption A1), has **no control on purpose**: dropping
it leaves every check on the board unchanged, because the limb-wise `DmaNext` link
derives well-formedness from the sender's `IsHalfword` checks instead. A control that
cannot fail is worse than no control. The gate's docstring previously claimed "every
negative control shows what breaks without them", which was false for exactly this
premise.

**On the depth bound.** The chain checks run at depth ≤ 5 (integer) and ≤ 3
(field-exact). The general depth case is not machine-checked; it rests on MAIN 2's
wrap lemma plus the strict decrease of `count`, which together bound the chain
length. Treat depth ≤ 5 as the checked case and that argument as the reason to
believe it generalises — not as a proof that it does.

**Does not prove**: assumptions A1–A5. The memory-consistency argument, hence
overlap ordering for unaligned 8-byte accesses — the largest remaining gap around
this feature, and not DMA's to close. LogUp soundness. The multi-call case
(Layer 2 models one head row; two ecalls are separated by the `ts` both `DmaNext`
tuples carry, which the integer abstraction does not model). And that the *Rust*
implements this — that is `audit_transcription.py`'s 104 textual claims plus the
end-to-end prove/verify and forgery tests.

### The named forgeries — what each bound actually buys

Each is run twice, with the bound and without, so "the bound is necessary" is a
measured claim rather than an assertion.

| result | with the bound | without |
|---|---|---|
| `Σ count_decr = 4·65535 ⟺ count_decr = 2^64−1` | unsat (the identity holds) | **sat** — `(0xFFFF+d, 0xFFFF−d, 0xFFFF, 0xFFFF)` reaches the same sum with a different `count_decr`, so `end` is claimable at a nonzero count. An `end` row's two `Memw` sends have multiplicity `mu − end = 0`, so **it emits no memory operations at all**: a silently truncated copy with every bus balanced |
| `carry_1 = 0 ⟹ src + width < 2^64` | unsat (pinned) | **sat** — at `src1 = 2^32−1` the high half can be exactly `2^32`, which the `IsHalfword` pair forbids and an unbounded pair does not. The row hands on a *wrapped* address that the executor's `checked_add` rejects |
| the `LT` width pin blocks `count = 7, end = 1` | unsat | **sat** — a free `tail` takes `tail = 0`, so `step = 8`, so `count_decr = 7 − 8 = 0xFFFF…`, so `end = 1`. **Seven requested bytes silently not copied** |

The last one is worth reading twice. `end` requires `count = step − 1`, so a free
`tail` buys exactly `count = 7` and no other value — the two constraints compose to
leave precisely one hole. That is also why an earlier draft wrote this forgery at
`count = 3` and was wrong: it is not reachable there.

### Which mechanism rejects each shipped forgery

The four forgery tests in `prover/src/tests/prove_elfs_tests.rs` only observe that
verification fails. The gate says *which mechanism* blocks each — the more useful
fact, and the one that tells you what a future refactor would break.

| shipped Rust forgery test | what it perturbs | the mechanism that rejects it | gate check |
|---|---|---|---|
| `forged_early_end_rejected` | `END := 1` on a data row | the `Zero` lookup (the sum no longer reads zero) **and** the three sends gated on `mu − end`, which vanish — so `DmaNext` and both `Memw` buses unbalance too. Not the `Zero` bus alone, as an earlier version of this table said | MAIN 1, and its `drop_zero_end` / `drop_halfword_count_decr` controls |
| `forged_wide_tail_rejected` | `TAIL := 1` on a wide row | **overdetermined — at least five independent mechanisms reject it.** Row-locally: `step = 8 − 7·tail` breaks the *ungated* idx-9 `emit_add_pair` on `count`; idx 5-6 and 7-8 fail identically; idx 11-17 (`tail·value[i] = 0`) fail whenever the eight copied bytes are not all zero. On the buses: the `Alu` width pin (bus 20, multiplicity `mu`) sends `[count, 8, 0, LT, TAIL, 0]`, so with `TAIL = 1` on a `count ≥ 8` row it asks `lt.rs` for output 1 where that table holds 0 — no matching row, `Alu` unbalances; and `w8 = 1 − tail` changes the `Memw` width | MAIN 0 |
| `forged_intermediate_source_rejected` | `SRC_0` **and** `SRC_INCR_0` shifted together | **nothing row-local** — the row's own ADD stays satisfied. The predecessor's `DmaNext` tuple no longer matches, and the source read no longer matches memory | CHAIN / CHAIN-F, exactly the check that treats `DmaNext` as a free bijection rather than an assumed chain |
| `forged_value_rejected` | `VALUE[0]` | **not the copy relation** — read and write still agree with each other, because they are one set of columns. What rejects it is the `Memw` read no longer matching memory | **none.** Audit §D pins the one-set-of-columns wiring; no solver query establishes this one |

Two rows repay attention. `forged_intermediate_source_rejected` is the case where
per-row soundness is genuinely insufficient and the chain argument does the work —
which is why the gate builds the bijection model instead of assuming rows are
chained. And `forged_value_rejected` passes for a reason **no solver query
establishes**: the only thing behind it is a textual fact about how two bus tuples
are constructed. That asymmetry is why the gate and the audit are separate artifacts.

> **How the `forged_wide_tail` cell got written, kept because it is instructive.**
> An earlier version credited the `Alu` width pin alone. A review called that
> incomplete, and the replacement over-corrected into *"**Not** the `Alu` width
> pin"* — which is false; the pin does reject it, by the argument in the cell. The
> chain was: a finder wrote "the Alu lookup is not what blocks it", that was
> accepted without checking, and it was then sharpened into an explicit negation.
> **An overstatement became a falsehood by being propagated.** When a mechanism is
> overdetermined, "X rejects it" and "Y rejects it" are both true, and the tempting
> edit — replacing one with the other — is the one that introduces the error.
> Prefer "at least these", never "not that".

## The transcription audit

The gate proves things about a **model**. Everything it proves is worthless if the
model and `prover/src/tables/dma.rs` have drifted, and the dangerous direction is a
model **stronger** than the object it models: it yields UNSAT where the real table
is forgeable, and no positive anchor can catch it, because honest inputs satisfy a
correct model and an over-strong one equally well.

`audit_transcription.py` therefore reads the Rust and asserts, textually:

```
A. constants    10   every number the oracle and gate hard-code
B. columns      28   the full dma::cols layout, NUM_COLUMNS, density
C. constraints  10   each index, template, operands, the degree bound, and that
                     no index exists the gate does not model
D. buses        23   23 interactions, bus mix, every multiplicity, and the wiring
                     facts the gate cannot see
E. executor      5   the ecall validates what the oracle validates, in that order
F. generator     7   the padding row is the row the oracle describes
G. bus packing  17   element counts per Packing, and DmaNext tuple ALIGNMENT
H. fixture       4   the Rust test consumes the oracle's current output
```

Counts are **printed by the script**, not documented by hand — an earlier version
stated them in prose and got five of six wrong, apportioned to sum to the real
total instead of measured, which is the "declared, not derived" defect this file
exists to catch.

It is deliberately textual rather than a Rust test: the point is to catch a change
in `dma.rs` that nobody reflected here, and a Rust test would be edited in the same
commit as the code it guards. Source is whitespace-normalised before literal
matching, so a `rustfmt` reflow does not produce a spurious red — which matters,
because a spurious red is how a check gets deleted rather than fixed.

**Mutation-tested, and it needed it.** An audit that cannot fail is not an audit.
Seventeen semantic mutants plus two must-not-fire controls, applied to copies of the
**Rust** source with the script re-run. (Distinct from `tamper_test.py`'s eight
mutants, which perturb the **Python oracle** to test the anchors — two separate
regression sets.)

| mutant | findings | notes |
|---|---|---|
| `timestamp_with_offset(2)` → `(1)` on the write tuple | 1 | |
| the write tuple's `value_columns()` → eight zero constants | 1 | |
| one `halfword(cols::COUNT_DECR_0)` send deleted | 3 | |
| `DMA_MEMCPY_MAX_BYTES + 1` → `+ 2` in the bound lookup | 1 | |
| the executor's `if n > DMA_MEMCPY_MAX_BYTES` guard → `if false` | 1 | **initially missed** — needed a strengthened check |
| `num_bus_elements(DWordHL)` `2 → 1` | 2 | **initially missed entirely** — this is the gap R1 came through |
| `num_bus_elements(DWordHHW)` `2 → 1` | 2 | added later: the first §G guard was **dark for this arm** (see below) |
| `DmaNext` receiver tuple reordered (`SRC_0`↔`DST_0`) | 1 | **initially missed** — §D pinned membership, not order |
| `DmaNext` sender tuple reordered (`SRC_INCR_0`↔`DST_INCR_0`) | 1 | **initially missed**, same cause |
| the read tuple's `is_register` `constant(0)` → `(1)` | 1 | **initially missed** — the claim tested for the *comment* |
| the read tuple's `w2` `constant(0)` → `(1)` | 1 | **initially missed** — three zero constants, the count needed only two |
| `w8` `1 - tail` → `tail` | 1 | **initially missed** — `cols::TAIL` appears either way |
| the `_INCR_0` sums swapped between constraint 5 and 7 | 2 | **initially missed** — §C pinned membership, not order |
| `emit_is_bit(b, 2, cols::TAIL, None)` → `Some(cols::MU)` | 1 | **initially missed** — the pattern stopped before `cond_col` |
| `value_columns()`'s `Packing::Direct` → `Word2L` | 1 | **initially missed** — `.*?` crossed the function boundary |
| the `ZERO` receiver's `65536` coefficient → `65537` | 2 | **initially missed** — the domain claim compared two constants of its own |
| the `ZERO` table's `for z in 0u32..16` → `..8` | 2 | same claim, other direction |
| a `rustfmt`-style reflow of `if tail { 1 } else { 8 }` | **0** | must NOT fire |
| `// 22. MEMW read` renamed to `// 22. Memw read` | **0** | must NOT fire — used to raise `IndexError` and print no report at all |

**Eleven of the seventeen were initially missed, in a file whose entire job is catching
exactly this.** The causes are worth naming because they are all the same species —
a check that cannot fail:

- The executor mutant: the check asserted the `DmaMemcpyChunkTooLarge` variant
  appeared *before* the `checked_add` calls, which a guard rewritten to `if false`
  satisfies. It now requires the literal predicate.
- The packing mutants: §D checked only that the strings `DWordWL`/`DWordHL`
  *appeared* in the two tuples, never that their element counts aligned. §G exists
  now, and **it took three tries to make live** — which is the most on-thesis fact in
  this file. The first searched ASCII `2x` where the source writes `2×` (U+00D7), so
  it could never match. The second searched `Packing::\w+ => 1, // 2×`, matching only
  arms whose comment *begins* `2×` — dark for `DWordHHW` ("Direct + Word2L") and
  `DWordWHH`, both equally 64-bit, which is why the `DWordHHW` mutant is in the table
  above. The third keys off the source's own `// Compounds` section marker, so all
  seven compound arms are covered and a newly added variant is covered by default.
  A guard written to close a gap was itself dark, twice, in a row.
- The two `DmaNext` ordering mutants: bus tuples were pinned by membership and not
  by ordinal position, so a swap silently re-paired every field the gate models.
- **And then the same defect again, in §C, which the `ordinal()` fix was never
  propagated to.** The add operands were checked for membership anywhere in the
  eval body, so swapping the two `_INCR_0` sums between constraint 5 and 7 — the
  AIR proving `src + step = dst_incr` and `dst + step = src_incr` while the gate
  models the opposite — passed. The lesson a gate writes down is not the lesson it
  has applied everywhere; the operands are now pinned to their index, the way the
  idx 9 claim always did it.
- The three data-tuple mutants: a bus tuple is matched element-for-element against
  its receiver, but the claims read it as a bag of substrings — and one of them
  read the `// is_register` *comment* rather than the value, so a tuple that made
  DMA read the register file at address `src` passed while deleting the comment
  failed. Both tuples are now pinned as an ordered shape, which also cannot be
  broken by a cosmetic comment edit.
- The two `ZERO` domain mutants: the claim that the send fits the receiver's domain
  compared `GATE_ZERO_SUM` against `GATE_ZERO_DOMAIN`, both defined in the audit
  itself — a tautology reading no source. The domain is now derived from
  `bitwise.rs`: the receiver's coefficients, the preprocessed loop bounds, and the
  fact that each coefficient is the stride of the digits below it.
- The `value_columns` mutant: `read()` strips every newline, so a `.*?` between two
  anchors crosses function boundaries freely — the match simply ran on to a later
  `Packing::Direct`. Character classes that exclude `}` cannot leave the body.
- The `emit_is_bit` mutant, the only one of the eleven that drifts the other way:
  the pattern stopped before the template's 4th argument, so `None` →
  `Some(cols::MU)` was invisible. That leaves `tail` unconstrained wherever
  `mu = 0` while the gate asserts `is_bit` unconditionally — a model STRONGER
  than the AIR, which fails safe for the prover and unsafe for the campaign,
  since every UNSAT the gate reports would then be proving something the AIR
  does not enforce.

**A section that dies has not passed.** The second must-not-fire control is a
comment rename, which used to raise `IndexError` from `split(marker)[1]` and kill
the run before anything was printed — so a cosmetic edit was a hard red and, to
anything scoring by exit code, a crash was indistinguishable from a catch. The
dispatch loop now reports it as a finding and the remaining sections still run;
the two MEMW tuples and the `ZERO` receiver are located structurally instead.

**The reflow mutant must produce zero findings, and used to produce two.** The
literal checks match fragments like `if tail { 1 } else { 8 }`, and `rustfmt` breaks
those across lines the moment one grows past `max_width`. The original guard,
`src.replace("\n", " ")`, collapsed the newline but left the indentation, so it
could never match a reflowed form. `read()` now whitespace-normalises. This matters
because the script is meant to run unattended: **a spurious red is how a check gets
deleted rather than fixed.**

## Lessons worth carrying to the next gate

**Model the receiving table's constraints, not its advertised contract.** The gate
models `Alu[a,b,LT] → o` as `lt.rs`'s own columns and carries rather than as
`o = (a < b)`. Apply the same rule to the **bus itself**: "how many field elements
does this value cross the bus as?" is a premise like any other and must be read
from `num_bus_elements()`, never assumed. That one omission produced a phantom
finding published as this campaign's headline result — see below.

**Classify the direction of every modelling gap.** Weaker than the AIR ⇒ false
alarms, never false proofs. Stronger ⇒ false proofs no positive anchor can catch.
Say which, in the verdict.

**Pair each negative control with the check that premise is load-bearing for.**
Dropping a premise and re-running a check whose reference never mentioned it yields
UNSAT — a control that cannot fail. Three of the original eight had this bug. And a
multi-row check needs its *own* positive control: `Not(property)` returning UNSAT is
worthless if the premise set is unsatisfiable.

**Never negate a modular equality carrying a witness quotient.** `Not(a − b == k·m)`
is satisfiable by picking a nonzero `k`. Spell such claims out witness-free.

**Field-exact, over integers, linear.** Columns are `Int` in `[0, p)`, modular
equalities carry explicit quotients, and `x·(1−x) = 0` becomes `x ∈ {0,1}` (exact
for `x < p` prime). The naive `%p` encoding is nonlinear and timed out on the main
check; this rewrite is what made a 5410-row completeness sweep affordable.

**When a mechanism is overdetermined, prefer "at least these", never "not that".**
A forgery rejected by five independent mechanisms invites the edit that credits one
and denies another — and that edit is how an incomplete claim becomes a false one.

## The retracted finding

An earlier version reported a residual — that `count`'s limb split was
unconstrained on non-head rows — and made it the headline across five documents.
**It was wrong**, and the story is the most transferable thing here.

`DmaNext` does not compare packed 64-bit values. `Packing::num_bus_elements()`
returns **2** for both `DWordWL` ("2× Direct") and `DWordHL` ("2× Word2L"), each
element gets its own alpha power, and **no `Packing` variant contains a 2³² shift**
— so a 64-bit value is never one bus element anywhere in this codebase. Both tuples
are `1+1+2+2+2 = 8` elements and align pairwise, so balance imposes two equations
per value:

```
receiver.COUNT_0 == sender.cd₀ + 2¹⁶·cd₁     (low word)
receiver.COUNT_1 == sender.cd₂ + 2¹⁶·cd₃     (high word)
```

With the sender's halfwords `IsHalfword`-checked, the receiver's limbs are 32-bit
**for free**. Modelling the hop as one equation is strictly weaker than the AIR: it
lets the receiver re-split its limbs, manufacturing an alias the real bus rejects.

Three things to keep:

- Every UNSAT survived the correction, having been proven under weaker hypotheses
  than reality supplies. The error direction was the safe one.
- **A phantom finding causes real damage.** Working around it led to asserting
  `count ≤ 256` on *every* row of the field-exact chain check when the AIR bounds
  only the head — a genuinely over-strong assertion, in the dangerous direction,
  added to accommodate something that did not exist.
- **A proposed fix that is a no-op means the gap is not there.** The recommended
  fix was "receive `count` as `DWordHL`", which changes nothing under the real
  semantics. That should have stopped the write-up.

§G now asserts the element counts and the tuple alignment directly. Its absence is
what let the phantom through, and `spec/dma.typ`, written independently, reaches the
same conclusion by a different route.

## Where to send the next reviewer

1. **The `Memw` ordering argument for unaligned 8-byte accesses.** A misaligned copy
   generates one on nearly every row, and the whole snapshot story rests on `T+1`
   reads preceding `T+2` writes per address. Nobody has checked it.
2. **A1–A5 centrally**, rather than per chip. `IS_WORD` has no discharger anywhere.
3. **For PR #874, not here:** `end·(1 − tail) = 0` would close the
   `count = 7, tail = 0` truncation inside the AIR instead of leaving it to the
   `Alu` bus (defence in depth — the pin is sound today), and DMA is the only
   high-volume table with no `max_rows`/chunking.

### Still open, report-only

Recorded rather than closed, so the next reviewer does not have to rediscover them.

1. **`replay_dma_memcpy_for_sizing`** (`trace_builder.rs:1090`) is a
   `#[cfg(feature = "disk-spill")]` duplicate of the payload logic in
   `collect_dma_memcpy_ops` — lines 1117, 1140 and 1160-1168 mirror 1006, 1033 and
   1060. `dma_ops_for_test` calls the primary directly, so **none of this
   directory's mutation coverage reaches the mirror**; the five payload mutations
   the Rust tests catch would all survive there. `count_table_lengths_drift_tests`
   is the only thing touching it and it compares row counts, not payload fields.
   Deduplicating is a change to shipped code and therefore out of scope for a
   verification-only branch.
2. **`count_table_lengths`** — the disk-spill sizing pass. Covered by the PR's own
   `count_table_lengths_drift_tests.rs`; not re-derived here.
3. **The `n = 0` ecall.** One row, both `first` and `end`, no `DmaNext` traffic, no
   memory operations. Pinned by the completeness sweep and by
   `empty_dma_call_is_a_single_first_and_terminal_row`, but it is the row shape most
   likely to be broken by a future multiplicity change, because **every
   multiplicity on it is zero** — nothing about it is load-bearing until it is.
4. **Two ecalls at one timestamp.** Ruled out by CPU timestamps strictly increasing
   per instruction (A4). Asserted, not verified here, and it is what the `DmaNext`
   timestamp binding rests on.
5. **The `## Results` block is still hand-copied.** `verify.log` is no longer on
   this list — `make verify-dma` now diffs the gate's live output against the
   committed transcript and fails if they disagree, so that file cannot go stale
   silently. The `## Results` block above aggregates four sources by hand and can.

No CI workflow runs any of this yet; `make verify-dma` is the entry point.
Two cross-references point at siblings that are **not merged**:
`formal_verification/keccak/` (PR #923) and `spec/dma.typ` (PR #931).
