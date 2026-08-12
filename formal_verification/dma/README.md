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

| file | what it is |
|---|---|
| `dma_ref.py` | the oracle: four levels of reference model, no repo code |
| `test_ref.py` | anchors the oracle against libc and CPython; emits the fixtures |
| `tamper_test.py` | are those anchors sensitive? eight deliberate defects |
| `z3_verify.py` | the gate: field-exact model of the AIR |
| `audit_transcription.py` | 104 claims tying gate, spec and Rust together |
| `verify.log` | the gate's own output, committed |
| `canonical_dma_rows.txt` | pinned vectors, `include_str!`-ed by the Rust test |
| `canonical_dma_vectors.json` | the same vectors with full per-row column expansions |

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

audit:   104 claims, 0 findings; mutation-tested against 8 source mutants
rust:    cargo test -p lambda-vm-prover --lib tests::dma_tests   10 passed
```

Full gate transcript in `verify.log`. **`--lib dma` is 18 tests, but 7 need guest
ELFs** — run `make compile-programs-rust` first (RISC-V target) or use the
narrower filter above. `make verify-dma` runs no cargo.

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

## The chip, as recovered from the implementation

`spec/dma.typ` (PR #931) is the normative chapter; this section is what the gate
checks and agrees with it except where noted.

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

Three asymmetries are load-bearing and must not be "tidied":

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

**Overlap.** All reads at `T+1`, all writes at `T+2`, both AIR constants, giving
snapshot (`memmove`) semantics per ecall. **Not** per guest `memcpy`: chunk *k+1*
reads what chunk *k* wrote, so a copy over 256 bytes is a forward copy. In
contract for `memcpy`, but the `memmove` property must not be claimed at the C level.

## Assumptions

Obligations on the **caller**, not checks this chip performs.

| id | assumption | discharged by | status |
|---|---|---|---|
| **A1** | `src`/`dst` limbs are 32-bit wherever a `Memw` data op fires | `spec/src/memw.toml`: `[[assumptions]] IS_WORD[base_address[i]]`. `memw.rs` justifies its own bound via *the CPU table*; DMA is a non-CPU sender | not discharged locally. Load-bearing only on the head row — elsewhere the `DmaNext` link derives it, which is why the gate's `memw_addr32` toggle is **inert and carries no control on purpose** |
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
widths; ten of the eleven modelled premises are individually necessary, each with a
named forgery; Layer 2's premise set is satisfiable and sensitive to each field of
the bus tuple; and the AIR accepts every honest trace for every length `0..256`.

**Does not prove**: assumptions A1–A5. The memory-consistency argument, hence
overlap ordering for unaligned 8-byte accesses — the largest remaining gap around
this feature, and not DMA's to close. LogUp soundness. The multi-call case
(Layer 2 models one head row; two ecalls are separated by the `ts` both `DmaNext`
tuples carry, which the integer abstraction does not model). And that the *Rust*
implements this — that is `audit_transcription.py`'s 104 textual claims plus the
end-to-end prove/verify and forgery tests.

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

**Mutation-tested, and it needed it.** Eight source mutants; **four were initially
missed**: an `if false` guard passed because a check asserted an error variant was
*mentioned* rather than that the guard existed; §G's "no variant folds 64 bits into
one element" searched for ASCII `2x` where the source writes `2×`, so the regex
could never match; and the `DmaNext` tuple *order* was unpinned in two directions.
An audit that cannot fail is not an audit.

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

No CI workflow runs any of this yet; `make verify-dma` is the entry point.
