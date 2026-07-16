# Continuations (Approach 2) — design

This is the single design document for the "continuations" prover (Approach 2,
"prove-epoch" from the streaming spec). It covers the things a continuation must
carry across epoch boundaries — **memory** (the bulk of the doc: §1–§5, including
the cross-epoch local-to-global table and the Design X vs Design Y decision),
**registers** including the commit index x254 (§6), and the **Fiat-Shamir statement
binding** (§7) that ties each epoch proof to its program and position — plus the
soundness mechanisms that make each safe. §8 describes the **standalone (split)
prover/verifier** that checks a proof bundle with only the ELF.

It is written to be read by a human picking this up cold.

---

## 1. Why continuations

A monolithic proof builds the trace for the **whole** execution in memory at
once; for large programs that exhausts RAM. Continuations split the execution
into fixed-size **epochs** and prove each independently, so peak memory stays
flat as program size grows.

Almost every constraint in a proof is local to its slice of cycles — *except
memory*. A load in a late epoch may read what an early epoch wrote. So the only
thing that must be stitched across epoch boundaries is **memory consistency**.

```
   one execution (e.g. 4,000,000 cycles)
   ┌───────────────────────────────────────────────┐
   │  split into epochs of N cycles                 │
   └───────────────────────────────────────────────┘
        │            │            │            │
        ▼            ▼            ▼            ▼
   ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐
   │ Epoch 0 │  │ Epoch 1 │  │ Epoch 2 │  │ Epoch 3 │  each proven on its own
   │ CPU MEMW│  │ CPU MEMW│  │ CPU MEMW│  │ CPU MEMW│  (tables dropped from RAM
   │ ... L2G │  │ ... L2G │  │ ... L2G │  │ ... L2G │   after each epoch)
   └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘
        └────────────┴─────┬──────┴────────────┘
                           ▼
                ┌────────────────────────┐
                │   ONE global proof     │  links the epochs together
                │   (cross-epoch memory) │
                └────────────────────────┘
```

---

## 2. The pieces

A **bus** is a LogUp channel: tables *send* and *receive* tokens, and the proof
checks that everything sent is received (the bus "balances"). An unmatched token
makes the proof fail.

- **MEMW** — the actual loads/stores, driven by the CPU executing the program.
- **L2G** (local-to-global) — one row per memory cell an epoch *touches*. Two roles:
  - inside an epoch, on the **Memory bus**, it is the *bookend* — it supplies a
    cell's starting value (seed at timestamp 0) and collects its ending value.
    It **replaces the PAGE table**, which is switched off inside continuation
    epochs.
  - across epochs, on the **GlobalMemory bus**, it carries each cell's
    "where did this value come from / where is it going" claims.
- **global_memory** — the *anchors* on the GlobalMemory bus:
  - **genesis**: a cell's starting value. For ELF/runtime pages it is **preprocessed**
    (read from the ELF, so the verifier recomputes it — the prover cannot choose initial
    memory). For **private-input pages** it is a **committed** (non-preprocessed) column
    the verifier never recomputes from the ELF — the raw private input is neither bundled
    nor reconstructed by the verifier, and the value is pinned by the bus instead (see
    §3.6); this mirrors the monolithic PAGE table. (Not a ZK/hiding guarantee — see §3.6.)
  - **finalization**: a cell's final value after the last epoch that touched it.

### A single L2G row

```
   ┌──────────┬───────────────────────────┬───────────────────────────┐
   │ address  │ init: value, epoch        │ fini: value, time         │
   └──────────┴───────────────────────────┴───────────────────────────┘
     which       what it was when this        what it is at this
     cell         epoch first saw it, and       epoch's end (its last
                  which epoch wrote it          access timestamp)
```

Column layout (8 columns): `address_lo/hi` (32-bit), `init_value` (byte),
`init_epoch` (two 16-bit halfwords), `fini_value` (byte),
`fini_timestamp` (32-bit `Word`), `MU` (selector).

Note: **`fini_epoch` is NOT a column** — it is supplied as a per-table constant
(see §4.2).

Note: there is **no `init_timestamp`**. Timestamps are epoch-local (each epoch's
clock restarts; the Memory-bus seed is `ts = 0`) and order accesses only *within*
an epoch. The cross-epoch chain is ordered by the **epoch number** (§3.3), so the
GlobalMemory bus carries no timestamp at all (see §2 telescoping). `fini_timestamp`
stays only because the epoch-local **Memory bus** needs it (matched against MEMW).

Note: timestamps are single 32-bit `Word`s, so an epoch's largest real timestamp
(`halt_timestamp + 4·num_padding_rows + 1`) must stay strictly below the `2^32-1`
finalization sentinel — capping an epoch at roughly `2^30` cycles. This is enforced
by the `epoch_size_log2 <= 28` bound in `prove_continuation` and, as a backstop for
every proving path, by a trace-build guard that returns `Error::TimestampOverflow`.

### Cross-epoch telescoping

For a cell touched in epochs 1, 2, 3, the GlobalMemory bus checks:

```
 global_memory      L2G(ep1)        L2G(ep2)        L2G(ep3)      global_memory
   GENESIS  ───────►  init
   (value v0,         fini  ───────►  init
    from ELF)                         fini  ───────►  init
                                                      fini ───────► FINAL
                                                                    (last value)

   each  "fini ───► init"  is one matched token:
   epoch i's fini == epoch (i+1)'s init  (same address, value, epoch — no timestamp)
```

The bus balances **iff** every `fini` is consumed by the next-touching epoch's
`init`, anchored by GENESIS (the one source) and FINAL (the one sink). That
chain *is* "memory stayed consistent across epochs." Inside each epoch, ordinary
memory checking (MEMW + timestamp ordering) handles consistency; L2G only
provides the seam at the edges.

---

## 3. Soundness, by component

The skeleton above is correct but not *sound* on its own — a cheating prover
could make the buses balance while lying. Four mechanisms close the gaps.

### 3.1 Range checks on the L2G columns

Raw field columns must be forced into their intended ranges, or a prover can
stuff out-of-range junk into them.

Principle: **only check what nothing else already checks.**

- `address`, `fini_timestamp`, the value bytes — these travel on the Memory bus
  and are matched against **MEMW**, which already range-checks them (exactly how
  PAGE relied on MEMW). No extra check.
- The **cross-epoch-only** field `init_epoch` has no MEMW partner, so L2G checks it
  itself: store as 16-bit halfwords, check each with the `IsHalfword` lookup, and
  rebuild the value as `lo + 2^16·hi`. Because only the range-checked halfwords feed
  the reconstruction, no extra AIR constraint is needed. (There is no
  `init_timestamp` to check — the GlobalMemory bus carries no timestamp; see §2.)

The value bytes get PAGE's batched `AreBytes` check (the `init` value is a
trusted source and must be checked).

### 3.2 `fini_epoch` as a per-table constant

Inside epoch *i*'s table, **every** row's `fini_epoch` is just *i*. So it does
not need to be a per-row committed column — it is supplied to the AIR as a
constant `epoch_label`, computed by the verifier from the epoch's position.

This is *strictly more sound* than a column: the prover cannot choose it. The
genesis sentinel is `0` and real epochs are labelled `1, 2, 3, …`
(`epoch_label(i) = i + 1`), so genesis is below every real epoch.

### 3.3 Cross-epoch ordering (the subtle one)

The GlobalMemory bus only proves the tokens **match as a set** — not that they
are chained in increasing-epoch order. Without that, a cheater can make a row's
`init` and `fini` cancel each other (point `init` at its own epoch), so the row
**vanishes** from the chain — letting an epoch read a *forged* value for a cell
while a later epoch absorbs that cell's real genesis. The bus balances; the
program ran on a lie.

Fix: force every row to reference a strictly earlier source —
`init_epoch < fini_epoch`. With genesis `= 0` and 1-based epochs, genesis (`0`)
satisfies it with no special case.

How `a < b` is checked without a dedicated comparison table (the same trick
MEMW uses for timestamps): in the field, `a < b` ⟺ `b − 1 − a` is a small,
in-range number. If `a ≥ b`, that subtraction wraps to a huge field element that
fails the range check. So we range-check `fini_epoch − 1 − init_epoch` with the
`IsB20` (20-bit) lookup — reusing the bit-table already present, near-zero cost.

```
   honest:  init=2, fini=5  →  5-1-2 = 2     small  ✓ passes
   cheat:   init=5, fini=5  →  5-1-5 = -1    wraps  ✗ fails  (self-reference)
   cheat:   init=9, fini=5  →  5-1-9 = -5    wraps  ✗ fails  (future reference)
```

Strict `<` (not `≤`) is required: `≤` would permit `init_epoch == fini_epoch`,
which is exactly the self-cancel that enables the forgery. Strict `<` guarantees
a real row's `init` and `fini` epochs always differ, so a real row can never
self-cancel.

Cost: this bounds the **number** of epochs to `< 2^20` (~1M) — *not* their size.
Unreachable in practice (optimal epochs are millions of cycles → thousands of
epochs even for a billion-cycle run) and fails closed. If ever needed, widen the
gap check to 32-bit or switch to the LT table.

### 3.4 The `MU` selector

Traces are padded with blank rows to a power of two (an FFT requirement). Those
padding rows must not disturb any bus.

Originally padding was harmless because a blank row's `init` and `fini` tokens
were identical and self-cancelled. **Two** of the changes above broke that, each
on its own:

- §3.2 (constant `fini_epoch`): a padding row's `fini` now carries
  `epoch = the constant` while its `init` carries `epoch = 0`, so the tokens
  differ and no longer cancel.
- §3.3 (the ordering check): a padding row has `init_epoch == fini_epoch` (both
  `0`), which fails the strict `<` check.

So `MU` is required by *either* change.

Fix: a selector column `MU` (1 on real rows, 0 on padding). Interactions gated by
`Multiplicity::Column(MU)` contribute nothing on padding rows.

`MU` is itself constrained boolean (`MU·(1−MU)=0`), and pinned to the right
rows by bus balance (a real row with `MU=0` drops its telescoping link →
imbalance).

### 3.5 CPU padding and the power-of-two epoch size

The CPU table is padded to a power of two (the same FFT requirement). After the
inline-PC rework, padding rows are **not** inert: each carries `pc = 1` and
reads/writes it on the inline-PC `memory` chain, and that chain is anchored only by
the HALT chip's `consume_pc`/`emit_pc` — which converts the last real `next_pc`
into the `pc = 1` sentinel the padding rows expect.

An **intermediate** continuation epoch excludes HALT (only the *final* epoch
halts). So if an intermediate epoch had padding rows, their `pc = 1` tokens would
dangle — no HALT to anchor them, and the REGISTER FINI carries the real next PC,
not `1` — and the Memory bus would not balance. The honest prover could not produce
a verifying proof.

Fix: **epoch size is expressed as `epoch_size_log2`**, so the driver slices at
exactly `2^epoch_size_log2` cycles. An intermediate epoch runs that exact
power-of-two number of cycles, so its CPU table already has a power-of-two row
count and therefore **zero padding rows** — nothing to dangle. The final epoch
keeps its remainder *and* its HALT, so its padding chain is anchored as usual. A
program shorter than one epoch runs as a single final (monolithic-style) epoch.

This is a **completeness** fix: it changes no constraint and nothing the verifier
accepts — only how the driver slices cycles. A debug-assert enforces the
"intermediate epoch ⟹ power-of-two cycle count" invariant.

### 3.6 Private-input genesis (committed, not ELF-bound)

Genesis for ELF/runtime pages is preprocessed, so the verifier recomputes it from the
ELF — that is what stops a prover from choosing initial memory (§2). But **private
input** is, by definition, *not* in the ELF, so it must not be verifier-recomputed and
must not be shipped in the proof bundle. So a private-input page's genesis cannot be
ELF-recomputed.

Fix (mirrors the monolithic PAGE table exactly): build the `global_memory` AIR for a
private-input page **non-preprocessed**, so its `INIT` (genesis) is a **committed
main-trace column** the verifier never recomputes from the ELF. Correctness is enforced by
the same bus chain as everything else: the genesis token telescopes into the first
touching epoch's L2G `init`, which is pinned on the epoch-local Memory bus to MEMW's
true first-read value. A forged genesis would leave an unmatched Memory-bus term. This
is the same "output pinned by a complete chain" argument as the finalization (§4): the
private genesis is prover-supplied *by design* (it is the private input), so the proof
attests "**there exists** a private input producing this output" — the intended
semantics, identical to the monolithic prover.

**Scope of the guarantee (not zero-knowledge).** What this buys is that the raw private
input is **neither bundled in the proof nor recomputed by the verifier** — not that it is
cryptographically hidden. This proving stack is a non-ZK STARK: the committed private
`INIT` column, like every committed column, is opened at FRI query positions, so a
verifier does learn some trace evaluations. Cryptographic hiding of the private input
would require a ZK/blinded proof system (a separate, larger change). Phrase any external
claim as "raw private input is not bundled or recomputed by the verifier," not "the
verifier never sees it."

**One prerequisite — the region must hold only private input.** Skipping the ELF
recomputation is safe *only* if no ELF-declared data lives in the private-input region;
otherwise a prover could classify that page private and forge the ELF byte's genesis
(the value would be committed but never checked against the ELF). This reservation is
**enforced by the loader**: `Elf::load` rejects any `PT_LOAD` segment reaching at or above
`PRIVATE_INPUT_START_INDEX` (`ElfError::SegmentInPrivateInputRegion`) — covering every page
the verifier can classify private, which slightly exceeds `[base, base+MAX_PRIVATE_INPUT_SIZE)`
because the length prefix pushes an honest max-size input onto one more page (the count
bound is that tight span, with no extra slack).
Turning the reservation from convention into an enforced invariant closes this gap for
**both** the continuation and monolithic paths (they share the loader and the same
non-preprocessed-private-page design).

**Which pages are private** is decided by **count**, not by the raw byte range: the
first `num_private_input_pages` pages from `PRIVATE_INPUT_START_INDEX` (the page-aligned
span the input occupies), exactly matching the monolithic verifier's
`page_configs_from_elf_and_runtime`. The count is a public value in the bundle:
bound-checked against the max, absorbed into the global Fiat-Shamir statement (§7), and
additionally pinned by the committed AIR shape — a wrong count flips a *touched* page's
preprocessed mode, so the rebuilt AIR no longer matches the committed trace and the
proof fails. The verifier is given **only the count**, never the private bytes
(`verify_continuation` takes `elf + bundle` alone).

Before this, the continuation bundle shipped the raw `private_inputs` and the verifier
recomputed the private genesis from them — which both **leaked** the input and
contradicted the memory spec (`memory.md`: prover/private input is a *committed* column,
not verifier-recomputed). §3.6 removes both problems.

---

## 4. Design X vs Design Y — *where* `MU` is applied

`MU` is needed to neutralize padding, but **which** interactions should it gate?

```
                       GlobalMemory   Memory     range +
                       (telescoping)  (bookend)  ordering
   Design X (SOUND):       MU            MU          MU      ← MU gates everything
   Design Y (UNSOUND):     MU           One         One      ← MU only on GlobalMemory
```

**Conclusion up front: Design X is sound; Design Y is *not*.** We initially
believed Y was a cleaner equivalent (and two adversarial reviews agreed). They
were wrong — Y opens a chain-truncation attack. Below is X, then Y, then the
attack and why X blocks it.

### Design X

`MU` gates **every** L2G interaction (matches the standard table pattern —
LT/MUL/MEMW each gate all their interactions with one multiplicity column).

The crucial consequence — which we first mistook for redundancy — is that gating
the **Memory bus bookend** with `MU` forces `MU = 1` on every *touched* cell:
a touched cell's MEMW accesses need the L2G seed/fini on the Memory bus (PAGE is
off), so `MU = 0` would dangle them → the epoch proof fails. This is **Statement
S** below. Forcing `MU = 1` on every touched cell forces every touching epoch
**into the global chain** — so the chain is **complete**, and cannot be truncated.

### Design Y (rejected — unsound)

`MU` gates **only the GlobalMemory bus**; the Memory bus and range/ordering checks
use `Multiplicity::One`. The intended win was that the ordering check then fires
unconditionally so `MU` can't skip it. But decoupling the Memory bookend from `MU`
**broke Statement S**: a touched cell's bookend now fires regardless of `MU`
(`Multiplicity::One`), so the epoch proof passes even with `MU = 0`. Nothing then
forces `MU = 1` on a *non-first-touch* row — and that is exploitable.

### The attack Design Y allows: orphan a touched epoch

Cell A, touched by epochs e1 then e2. Honest: genesis `v0` → e1 writes `f1` →
e2 writes `f2` → final `f2`. A cheating prover sets **`MU = 0` on e2's L2G row**
and sets `global_memory`'s finalization for A to `f1`:

```
   genesis(v0) ──► e1.init        ✓ (genesis must be consumed — forces e1 only)
   e1.fini(f1) ──► FINAL(f1)      ✓ (prover-chosen finalization absorbs it)
   e2.init / e2.fini              ✗ MU=0 — orphaned, don't fire
```

- The GlobalMemory bus **balances** (every fired token matched).
- e2's **epoch proof still passes** — in Design Y its Memory bookend is
  `Multiplicity::One`, so it fires regardless of `MU`; e2 ran internally-consistently.
- **Nothing forces `MU_e2 = 1`:** e2 isn't first-touch (genesis went to e1), and
  the finalization is a *prover column*, so it just absorbs whatever the last fired
  fini was.

Result: e2's write to A is silently dropped — A's final value is claimed `f1`
when it's really `f2`. A false statement, proven. (For a *middle* epoch, reroute
the later init to consume the earlier fini, skipping the middle one.)

The root cause is the **input/output asymmetry** of the anchors: genesis is the
*input* — a single per-cell **source** that must be consumed — while the finalization
is the *output*, a prover column that must be *forced* to consume the chain's tail. The
finalization is only trustworthy if the chain is **complete** so that the last fini is
forced into it. A complete chain pins the finalization; a truncatable chain leaves it
free. Design X forces completeness (via `MU=1` on every touched cell); Design Y does
not. (Genesis's *value* is ELF-recomputed for ELF/runtime pages and prover-committed for
private-input pages (§3.6), but either way it is the one source token the first-touch
epoch must consume, so this completeness argument is unchanged.)

### Statement S (why Design X is sound, and what Y broke)

> In a continuation epoch, the only table that provides a RAM cell's seed (its
> value at timestamp 0) on the Memory bus is L2G (PAGE is off). If a cell is
> accessed by MEMW during the epoch, the memory argument requires that seed; with
> `MU = 0` the seed is absent and the Memory bus cannot balance. Therefore any
> accessed cell is forced to `MU = 1`.

S rests on three checkable facts: (1) PAGE is off in continuation epochs;
(2) MEMW enforces timestamp ordering, so a cell's access chain must bottom out at
the seed; (3) no other table provides a RAM seed (REGISTER is registers only, a
disjoint token subspace).

**S requires the Memory bookend to be `MU`-gated** — that is exactly what Design X
has and Design Y removed. So the "redundant" `MU` on the Memory bus in Design X is
in fact load-bearing: it's what forces every touched epoch into the chain, making
the chain complete and the finalization trustworthy.

### The anchoring chain (why a real access cannot be dropped at all)

`MU = 1` being forced bottoms out at the program itself:

```
   ELF ─DECODE(preprocessed)─► each row's instruction (LOAD/STORE flags) is fixed
   PC-continuity ───────────► every executed instruction is present, in order
        │
   ▼ a real load/store row has its flag = 1 (DECODE match + IsBit) ⟹ CPU sends Memw req
   ▼ MEMW must receive it (MU_READ/MU_WRITE) — dropping it ⟹ Memw-bus imbalance
   ▼ MEMW's bookend pairing needs the L2G seed/fini — in Design X (MU-gated) ⟹ MU=1
   ▼ MU=1 ⟹ the cell is in the global chain ⟹ chain complete ⟹ finalization pinned
```

This is the VM's core execution soundness (DECODE + PC-continuity + IsBit flags,
verified in `cpu.rs` / `constraints/cpu.rs`), extended one link at a time up to
cross-epoch memory. Design X keeps every link; Design Y cut the MEMW→L2G link.

### How `global_memory`'s finalization is constrained — and the parallel with `main`

The finalization is **not** checked against an external value (it's the computed
output, not a known input). It is pinned **internally** by the bus: it must consume
the last fini of each cell's chain, which (with a complete chain) is the cell's
real last-written value. This is exactly how **PAGE** works in the monolithic
prover — PAGE's `fini` is pinned by the (single, complete) Memory bus to the last
MEMW write. Design X is the faithful cross-epoch extension; Design Y silently
dropped the "chain is complete" property both rely on.

---

## 5. Adversarial review summary

1. **`MU` safety (Design X).** Could `MU=0` on a real row, or a non-boolean `MU`,
   skip the ordering or forge a balance? No — caught by the Memory bus (Statement
   S) and the boolean constraint. **Holds.**
2. **Design Y.** Two adversarial reviews concluded Y was sound (padding harmless,
   ordering unconditional, "ghost row" attack defeated). **They were wrong.** Both
   only tested *first-touch* `MU=0` (genesis dangles → caught) and added/forged
   rows; neither tested **truncating the chain at a non-first-touch row** while
   pointing the prover-controlled finalization at the truncation. That attack (§4)
   makes Y unsound. Lesson: a review that misses an attack class proves nothing
   about it — the truncation/orphan class was the gap.
3. **`fini_epoch` as a constant.** Sound — strictly more so than a column. Labels
   are verifier-computed from epoch position (unforgeable); prove/verify use
   identical labels (no off-by-one); the free `init_epoch` column and
   `global_memory`'s `FINI_EPOCH` column are pinned by bus balance **when the chain
   is complete** (Design X). Independent of the X/Y choice.

---

## 6. Registers (cross-epoch)

Registers must also carry across epochs: epoch *i+1* must start from epoch *i*'s
final register file. Unlike memory, the register file is **small and fixed** (34
registers / 67 word-addresses, all present every epoch), so it needs no L2G /
global telescoping — we bind the whole snapshot directly.

**Mechanism (no new bus).** The REGISTER table is the register analog of PAGE — it
already puts each register's init/fini tokens on the epoch-local Memory bus
(REG-C1 init, REG-C2 fini, matched against MEMW). For continuation epochs we
**also preprocess the FINI column** = the epoch's final register file `R_{i+1}`
(on top of the already-preprocessed INIT = `R_i`). "Preprocessed" means
*verifier-known*: the verifier recomputes the column's commitment, so the prover
cannot choose it. The verifier reuses the **same** `R_{i+1}` as epoch *i*'s FINI
and epoch *i+1*'s INIT, so `init(i+1) == fini(i)` **by construction** — no equality
check and no bus. Genesis is epoch 0's INIT = the ELF entry-point registers
(verifier-derived).

```
   epoch i REGISTER              epoch i+1 REGISTER
     INIT = R_i      (pre)         INIT = R_{i+1}   (pre)   ← same R_{i+1}
     FINI = R_{i+1}  (pre) ────────┘                          reused both sides
```

### Register soundness (two locks)

For `R_{i+1}` to be the *real* final registers (not a free prover claim), two
locks compose:

1. **Preprocessing** pins the trace's FINI column = the public `R_{i+1}` (the
   verifier recomputes the commitment; the proof's FINI openings must authenticate
   against it, so the prover can't deviate).
2. **REG-C2 on the Memory bus** pins that FINI column = MEMW's true last write to
   each register (or the Memory bus doesn't balance).

Compose them: public `R_{i+1}` = trace FINI = real last write. So the value handed
to the next epoch is pinned to real execution.

The **monolithic prover is unchanged**: it keeps FINI as a main-trace column (it
has no verifier-known final state) and preprocesses 2 columns, not 3.

### Commit index (x254)

The COMMIT chip's running output index lives in a synthetic single-word register
**x254** (word-address 508), so it rides the **same** register binding above —
epoch *i*'s `FINI[x254]` becomes epoch *i+1*'s `INIT[x254]`, pinned by the two
locks like any register. Each epoch therefore indexes its committed bytes from the
*carried* value, not from `0`:

- the COMMIT trace seeds `current_commit_index` from x254
  (`register_state.read_index()` in `trace_builder.rs`), with a debug-assert
  pinning the two in sync every step;
- the verifier's commit-bus offset (`compute_commit_bus_offset`'s `start_index`)
  starts at the same carried x254.

The driver concatenates each epoch's committed slice into the run-wide output.
Because every slice is commit-bus-bound *and* the x254 indices are forced
contiguous (`init(i+1) == fini(i)`), the concatenation equals the true output
stream — no separate global "commit output" bus is needed.

---

## 7. Fiat-Shamir statement binding

Each epoch proof and the global proof seed their Fiat-Shamir transcript with a
**statement** before the challenges are drawn (they previously started empty). The
seeding only *adds* input to the transcript, so it can strengthen binding but never
weaken soundness — and it pins every proof to its program and position, so a proof
can't be replayed elsewhere:

- Each **epoch** absorbs: a domain tag, the ELF digest, the public output, the
  table layout, and the **epoch label** (its position).
- The **global** proof absorbs: a (distinct) domain tag, the ELF digest, the
  **epoch count**, the **private-input page count** (§3.6), and the **touched page-base
  set** — so the whole genesis AIR layout (which GLOBAL_MEMORY tables exist and which are
  non-preprocessed) is pinned in the statement, matching the monolithic path's
  `absorb_statement`.

The monolithic encoding is unchanged (same function, monolithic tag, no label).
The genesis / register / memory anchor values are *additionally* bound via the
preprocessed commitments absorbed during proving.

The standalone *split* verifier (§8) carries these statement fields in the proof
bundle and takes the epoch label / count from its own trusted enumeration, so the
binding holds there too — not just on the integrated path.

---

## 8. Standalone (split) prover/verifier

The continuation can be proved and verified by separate parties. `prove_continuation`
emits a self-contained `ContinuationProof` bundle; `verify_continuation(elf, &bundle)`
checks it using **only the bundle and the ELF** — nothing from the prover's memory.
The integrated `prove_and_verify_continuation` is now a thin wrapper
(`prove_continuation` then `verify_continuation`), and `prove_verify_epoch` is
likewise split into `prove_epoch` + `verify_epoch`.

The bundle is prover-supplied and therefore **untrusted**. Per epoch it carries the
`MultiProof`, the `public_output` slice, `table_counts`, `runtime_page_ranges`, the bound
`reg_fini` (`R_{i+1}`), and the epoch `l2g_root`; plus the global `MultiProof`, a top-level
`num_private_input_pages` **count** (§3.6), and the top-level **`touched_page_bases`** — the
sorted, deduped set of page bases the run touched. It carries **no cell values**: not the
raw private input, and — since the per-epoch `CellBoundary` list is *not* serialized — not
the touched-cell values either (a `CellBoundary.init.value` is a private-input byte for a
private read, so shipping it would leak the input in plaintext even though the raw blob is
gone). The verifier only ever needed the epoch count and the touched page-base set from
those boundaries; `touched_page_bases` supplies exactly that, value-free and at page
granularity. The full boundaries stay prover-local (they build the L2G traces and
final-state inside `prove_global`). Everything the integrated path reused from prover memory
becomes an **explicit verifier action**:

- **Enumerate, don't trust.** The verifier assigns each epoch's `label` and the
  `is_final` flag **by position** (`0..N-1`; the last is final), so the prover can't
  relabel, reorder, truncate, or append epochs — a wrong label diverges that epoch's
  Fiat-Shamir challenges, and a wrong `is_final` builds the HALT table in/out and
  mismatches the committed proof.
- **Derive the register / x254 chain.** Epoch 0's register INIT is derived from the
  ELF entry point; epoch *i+1*'s INIT is derived from epoch *i*'s bundle `reg_fini`
  (incl. x254 @ 508). So `init(i+1) == fini(i)` is now *enforced by the verifier
  rebuilding the AIR from the previous FINI* (via the shared `build_epoch_airs`),
  not merely true-by-construction. The commit-bus `start_index` is taken from the
  carried `register_init[508]`, not a free scalar.
- **Genesis from the ELF (private input excepted).** `verify_global` rebuilds the
  ELF/runtime genesis from the ELF alone (no private bytes) and closes the GlobalMemory
  bus; private-input pages are built non-preprocessed (§3.6), so their genesis is a
  committed, bus-pinned column the verifier neither recomputes nor sees.
  `verify_l2g_commitment_binding` ties each epoch's `l2g_root` to the corresponding
  global-proof sub-table root. The prover-supplied `touched_page_bases` is canonicalized
  (sorted/deduped) on ingest and pinned the same way the old `boundary` addresses were: a
  wrong set imbalances the GlobalMemory bus / mismatches the AIR count, and it is bound
  into the global Fiat-Shamir statement — so a reordered-but-same-set list still verifies
  while any different set is rejected.
- **Reconstruct the output** by concatenating the per-epoch commit slices (each
  commit-bus-bound, contiguous via the x254 chain).
- The verifier also `validate()`s `table_counts` and never trusts a prover-supplied
  page config (continuation epochs have none — PAGE is skipped under the L2G
  bookend, so `page_configs` is always empty).

A single `build_epoch_airs` helper builds the AIR set identically on both sides, so
prove and verify cannot diverge.

**Reviewed.** An adversarial "construct-a-break" audit (Phase-3 dismissal audit with
fresh agents) of the register/x254 chain, the L2G root binding, and
completeness-by-enumeration found no false-accept: each forgery is caught by a
Merkle/hash collision, a bus imbalance, or a Fiat-Shamir divergence.

The bundle derives rkyv and round-trips through `rkyv` (exactly like a
monolithic `VmProof`); the CLI drives it via `prove --continuations` (writes the
bundle) and `verify --continuations` (checks bundle + ELF only). `prove` picks the
epoch size from `--epoch-size-log2 N` (`N=20` means 1,048,576 cycles), defaulting
to `20`. A local ethrex 10-transfer distinct-account
sweep measured peak heap at roughly 6.9 GB (`19`), 9.5 GB (`20`), 15.8 GB (`21`),
and 26.8 GB (`22`); pick the highest value the workload and machine can run
without swapping.

**Limitation — not succinct.** The bundle carries, and the verifier checks, all *N*
epoch proofs plus the global proof. Continuations keep peak *prover* memory flat;
they do **not** shrink proof size or verify time. A single succinct proof needs a
recursion/aggregation layer (deferred).

---

## 9. Status and open items

- Implemented and tested: range checks (§3.1), `fini_epoch` constant (§3.2),
  ordering check (§3.3), the `MU` selector (§3.4), the **power-of-two epoch size**
  (§3.5), **private-input genesis not bundled/recomputed** (§3.6), **cross-epoch registers**
  (§6), the **commit index x254** across epochs (§6), the **Fiat-Shamir statement
  binding** (§7), and the **standalone split prover/verifier** (§8) — bundle serialized
  with `rkyv` and driven from the CLI (`prove`/`verify --continuations`).
- **The committed code implements Design X** (`MU` gates every L2G interaction),
  which is the sound design. Design Y was implemented briefly, then found unsound
  (§4, the chain-truncation attack) and **reverted**. Do not re-introduce the
  Design Y wiring: gating only the GlobalMemory bus reopens the orphan attack.
- Deferred:
  - **Succinctness.** The split verifier is non-succinct (N+1 proofs, §8). A single
    small proof needs a recursion/aggregation layer — a separate, larger effort.
  - **Private-input *content* binding.** The bundle no longer carries the private input
    in the clear (§3.6 — it carries only the page count; the raw input is neither bundled
    nor recomputed by the verifier). What remains deferred is pinning *which specific input*
    produced the output: the proof attests only that *some* private input does. A guest that
    needs "this exact input" must commit a hash of it to the public output — the framework
    provides no such binding on either the continuation or monolithic path.
  - **Zero-knowledge / hiding.** As noted in §3.6, this is a non-ZK STARK: committed private
    columns are opened at query positions, so the private input is not cryptographically
    hidden. Cryptographic hiding would need a ZK/blinded proof system.

---

## 10. Where the code lives

- `prover/src/tables/local_to_global.rs` — L2G columns, trace generation, the
  Memory/GlobalMemory bus interactions, range checks, the ordering lookup, and
  the per-row selector.
- `prover/src/tables/global_memory.rs` — the genesis (ELF-bound for ELF/runtime pages,
  committed/private for private-input pages, §3.6) and finalization anchors.
- `prover/src/tables/register.rs` — the REGISTER table: REG-C1/REG-C2 Memory-bus
  tokens, the preprocessed FINI commitment (`compute_precomputed_commitment_with_fini`,
  `NUM_PREPROCESSED_COLS_WITH_FINI`), and `fini_from_trace`.
- `prover/src/statement.rs` — the Fiat-Shamir statement absorbers
  (`absorb_statement` with `StatementKind`, `absorb_continuation_global_statement`).
- `prover/src/continuation.rs` — the split prover/verifier: `prove_continuation` /
  `verify_continuation` and the `ContinuationProof` bundle; the per-epoch
  `prove_epoch` / `verify_epoch` with the shared `build_epoch_airs` helper; the
  global proof (`prove_global` / `verify_global`); the per-epoch AIRs
  (`l2g_memory_air` / `l2g_global_air`); the power-of-two epoch sizing from
  `epoch_size_log2`; the register-FINI preprocessing; the transcript seeding; and
  `prove_and_verify_continuation` (the thin integrated wrapper).
- `prover/src/lib.rs` — `verify_l2g_commitment_binding` (epoch L2G root ↔ global
  sub-table root) and the commit-bus offset/balance helpers
  (`compute_commit_bus_offset`, `compute_expected_commit_bus_balance`) that take the
  carried x254 as `start_index`.
- `prover/src/tables/trace_builder.rs` — seeds `current_commit_index` from x254
  (`read_index`) so committed-byte indexing carries across epochs.
