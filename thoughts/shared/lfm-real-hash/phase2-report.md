# Phase 2 — BLAKE3 as a first-class `LFM_HASH` hasher

**Status:** GREEN. `lfm_prove_with_hasher(…, HasherKind::Blake3)` proves and the production
verifier accepts, at **both** 6 and 7 rounds, with the digest matching the socket KATs at
both. `make lint` and `make fmt` clean. No new test failures.

**Worktree:** `/Users/maurofab/workspace/lambda_vm-blake3-impl`, branch `blake3-real-hash`,
**all changes uncommitted** as instructed.
**Route:** A (host behind the frozen socket). **Mapping:** Option A + domain separation.
Both as locked by the plan.
**Date:** 2026-08-10.

Claims are marked ✓ EXECUTED (a test ran and passed, named), ✓ VERIFIED (read the code) or
? INFERRED (arithmetic shown).

---

## 1. Headline numbers

| | 6 rounds (A6R) | 7 rounds (standard, **default**) |
|---|---|---|
| main (value) columns / compression | **2,956** | **3,436** |
| bus interactions / compression | **1,190** | **1,382** |
| aux columns (`⌈interactions/2⌉`) | **595** | **691** |
| **base-field-equivalent cells** (`main + 3·aux`) | **4,741** | **5,509** |
| constraints | **794** | **922** |
| max degree | 3 | 3 |

✓ EXECUTED — `blake3_socket_tests::the_socket_budget_is_the_predicted_one_at_both_round_counts`
pins all ten numbers as literals against a closed form; `the_built_layout_matches_the_prediction`
and `the_census_prices_the_blake3_arm` confirm the *built* layout equals the prediction, and
the whole suite was run once per round count (`--features blake3-6round`), so both columns
are measured rather than one measured and one projected.

> ### ⚠ CORRECTION BOX — this table is a 2026-08-10 snapshot, overtaken TWICE since
>
> **The figures above are what the socket cost on the date of this report.** They are left
> as measured rather than edited, because a dated report that quietly acquires today's
> numbers stops being evidence of anything. What has moved, in order:
>
> | main columns | 6r | 7r | when |
> |---|---:|---:|---|
> | as reported here | 2,956 | 3,436 | 2026-08-10 |
> | + the LEAF mode's canonicity block (`Z`/`GINV` per felt, present on every row) | 2,964 | 3,444 | option C |
> | + the leaf RATE's four extra lanes (COMMIT.md §1.2) | **2,980** | **3,460** | 2026-08-13 |
>
> | cells (`main + 3·aux`) | 6r | 7r | when |
> |---|---:|---:|---|
> | as reported here | 4,741 | 5,509 | 2026-08-10 |
> | + canonicity block | 4,749 | 5,517 | option C |
> | + leaf RATE (which also adds 8 bus interactions) | **4,777** | **5,545** | 2026-08-13 |
>
> ⚠ **So the +16.19% A6R price quoted just below reads +16.07% today**, and the −4.1% / −3.6%
> hosting saving further down reads −3.4% / −3.0%. Neither conclusion moves: A6R is still the
> cheaper round count and hosting is still cheaper than the standalone chip at both.
>
> The current figures are deliberately not restated here as literals to be copied — copying
> is what let the first correction sit unnoticed for two months. They are derived by
> `blake3_socket_tests::predicted_cells`, and `blake3_probe.rs`'s socket-vs-standalone
> comparison now calls that function instead of quoting its output.

**The A6R price on this socket is +16.19% per compression** (4,741 → 5,509). PLAN §7's paper
estimate for the syscall-shaped chip was +15.5%; the socket pays slightly more because its
constant framing shrinks the round-*independent* part, so the rounds are a larger share.

**Hosting is cheaper than the standalone chip at both round counts** — and the standalone
chip is now compiled and measured at both, not projected at one:

| | standalone `LFM_BLAKE3` | `LFM_HASH` BLAKE3 arm | saving |
|---|---|---|---|
| 6 rounds | 4,946 ✓ MEASURED | 4,741 ✓ MEASURED | −4.1% |
| 7 rounds | 5,714 ✓ MEASURED | 5,509 ✓ MEASURED | −3.6% |

The saving comes from three things the socket framing makes constant: `h = IV` (the entire
initial state is constant, so zero input-state columns), `m[8..16]` (the domain tag and the
zero padding), and the truncation window — twelve of the sixteen output words are never
built.

### `BLAKE3_ROUNDS` is now 7 by default, one knob for both chips

Per the A6R sign-off. `BLAKE3_ROUNDS` was 6 and baked into the standalone chip; it is now
`7` unless `--features blake3-6round`, and `blake3_socket::SOCKET_ROUNDS` is an **alias** for
it rather than a second knob — two knobs would let a sweep leave the machine's hash and the
chip it is priced against describing different functions.

**The four figures you asked me to verify rather than trust are all confirmed by execution**
(`blake3_probe::the_hosted_chip_cell_budget_at_both_round_counts`,
`the_chip_emits_its_constraints_at_degree_3`):

| standalone chip @ 7 rounds | A6R sheet §4 projected | measured |
|---|---|---|
| G-block region (`cols::OUT − cols::G`) | 3,360 | **3,360** ✓ |
| bus interactions | 1,451 | **1,451** ✓ |
| cell-equivalent | 5,714 | **5,714** ✓ |
| constraints | 897 | **897** ✓ |

Its full 7-round budget: main 3,536, interactions 1,451, aux 726, cell-equiv 5,714,
constraints 897, BITWISE feed 1,440 per compression (was 1,248 at 6 rounds). Both round
counts' literals are pinned side by side in the probe, so the A6R price stays visible
whichever way the build is compiled.

**The 6-round vector pin survives the flip.** `CANONICAL_VECTORS` are 6-round data, and
`blake3_probe` asserted the chip's `OUT` columns against them — that assertion would have
become vacuous or wrong at 7. So `blake3.rs` gained `CANONICAL_OUT_7ROUND`: the same ten
inputs at 7 rounds, emitted by the gate-oracle's Python reference and cross-checked
word-for-word against the second in-repo reference (`blake3_ref.py`) — **two implementations
agreeing on all ten** ✓ EXECUTED, with the same run re-deriving the 6-round table and
reproducing it 10/10. Both references' 7-round paths are themselves pinned by the official
BLAKE3 vectors, so this table has an *external* anchor where the 6-round one has an anchor a
step removed. `canonical_expected_out(i)` selects by the knob, and a negative control asserts
the two tables differ on every vector.

---

## 2. What changed — file:line map

### New files

| file | lines | what |
|---|---|---|
| `prover/src/lfm/blake3_socket.rs` | 932 | the whole arm: framing constants, host hasher, column layout, wire interpretation, senders, BITWISE mirror, trace filler, constraints |
| `prover/src/lfm/blake3_socket_tests.rs` | 1,198 | 25 tests: KATs, 14 framing controls, layout, degree, O1/O2/O3, prove+verify, tamper, binding |
| `prover/src/lfm/blake3_socket_kats.rs` | 132 | GENERATED — 15 socket vectors × 2 round counts |

`blake3_socket_kats.rs` is the **union of the two independently produced vector tables**:
`thoughts/blake3/socket-kats/socket_kats.json` (Phase 1) and the gate-oracle's
`socket_kats.json`. They share 5 of the 15 input pairs and **agree on every one of them at
both round counts** — ✓ EXECUTED (checked before generating). The other 10 differ only in
which inputs were sampled, so the union is two sources, not one transcribed twice. The
SOCKET.md §5 worked example (`nibble_ramp`) reproduces exactly.

### Changed files

| file:line | change |
|---|---|
| `blake3.rs:63-85` | `BLAKE3_STANDARD_ROUNDS = 7`, `BLAKE3_SIX_ROUNDS = 6`, and **`BLAKE3_ROUNDS` flipped to 7** behind `blake3-6round` |
| `blake3.rs:407` | `CANONICAL_OUT_7ROUND` — the ten canonical inputs at 7 rounds, from two agreeing references |
| `blake3.rs:472` | `canonical_expected_out(i)` — selects the table matching the knob |
| `blake3_chip.rs:720` | `output_words()` follows `BLAKE3_ROUNDS` (was hardwired 6-round) |
| `blake3_probe.rs` | cell budget, constraint count and BITWISE feed all parameterised, both round counts' literals pinned; the `#[ignore]`d census rows de-staled so they cannot mislead if un-ignored |
| `blake3.rs:107` | `blake3_compress_rounds(…, rounds)`; `blake3_compress_6round` now delegates. One loop bound, no second copy |
| `blake3.rs:561` | ★ `seven_rounds_is_the_blake3_crate` — the deferred Phase-1 crate cross-check, over 65 message lengths |
| `blake3.rs:604` | `six_rounds_is_not_the_blake3_crate` — its negative control |
| `blake3_chip.rs:238` | new `FlowConfig { rounds, out_window, full_output }` |
| `blake3_chip.rs:301` | `run_flow(f, cfg)` — the framing decisions that change *which calls happen* moved into the single dataflow |
| `blake3_chip.rs:292,295` | `feed_forward` split into `_low` / `_high` |
| `blake3_chip.rs:394` | `Add3Wire.m` widened `[usize;4]` → `WordRef` (constant message words) |
| `blake3_chip.rs:581` | `ValueFlow::compute_with(…, cfg)` |
| `hash.rs:49` | new `LfmHasher::compress_out` (default = permute-and-truncate); `compress` now derives from it |
| `hash.rs:72` | new `LfmHasher::admits(mode, state)` — the domain-restriction declaration |
| `hash.rs:156` | `HasherKind::Blake3 = 2` |
| `hash.rs:189-215` | explicit delegation of `compress` / `compress_out` / `admits` / `permute` / `compress_iv` |
| `chips.rs:587,683,707` | `num_columns` / `num_constraints` / `eval` arms |
| `chips.rs:603` | **`bus_interactions(kind)` — the signature change.** BLAKE3 appends its BITWISE lookups to the frozen six `LfmMem` tuples (`lfm_mem_interactions`, `chips.rs:612`) |
| `airs.rs:189,429` | the two `bus_interactions` call sites threaded (the census reads `airs.rs:189`) |
| `trace.rs:244` | witness-filling arm |
| `trace.rs:185-196` | BITWISE multiplicities — the one place the shared table's histogram depends on the hash choice |
| `executor.rs:69` | `LfmExecError::HasherRejected(&'static str)` |
| `executor.rs:395-410` | `admits` guard, and Compress now goes through `compress_out` **not** `permute` |
| `blake3_probe.rs:357-375` | the 7-round standalone figures, pinned |
| `blake3_socket.rs:153` | **D9** — `const _: () = assert!(SOCKET_ROUNDS == BLAKE3_ROUNDS)`, the single-knob tripwire |
| `blake3_socket_tests.rs:940` | **D2** — `the_lane_range_check_is_load_bearing_on_its_own` |
| `poseidon_chip_tests.rs:219` | renamed//widened: the *`LfmMem` tuple* contract is hasher-independent; the interaction list is not |
| `prover/Cargo.toml` | `blake3-6round` feature; `blake3 1.8.5` dev-dependency (was already in the local registry cache; resolved `--offline`) |

**The executor change is load-bearing and easy to miss.** It previously computed *every*
hash row as `hasher.permute(state)`, inlining the trait's default `compress`. An overriding
`compress` was therefore never honoured on the prove path. BLAKE3 must override it
(obligation O3: the IV enters through `h`, not the capacity lanes), so `compress_out` was
added and the executor routed through it. Test/Poseidon behaviour is unchanged by
construction — the default `compress_out` *is* the old expression.

---

## 3. Conformance against `chip_model.py`

Every `CHIP CONSTRAINT` / `CHIP SENDS` comment in the model, mapped to the Rust that
realizes it. Verdict: **conformant, with one deliberate deviation (row 6) that is provably
equivalent and strictly cheaper, plus four constraints the model does not cover because it
does not model the host socket.**

| # | `chip_model.py` | obligation | Rust | ✓ |
|---|---|---|---|---|
| 1 | `emit_lane_bytes` — `MU·(LANE_j − Σ MB[j][k]·2^{8k}) = 0`, per lane | eval, mu-gated, deg 2 | `blake3_socket.rs:820` (idx 6–13) | ✓ exact |
| 2 | `emit_lane_bytes` — `AreBytes(MB[j][0],MB[j][1])`, `AreBytes(MB[j][2],MB[j][3])` | 2 sends/lane = 16 | `blake3_socket.rs:609-620` | ✓ exact |
| 3 | `message_words` — m[8..16] carry NO columns and NO range checks | structural | `message_word_ref`, `blake3_socket.rs:402` → `WordRef::Const` for `i ≥ 8` | ✓ exact |
| 4 | `init_state` — all sixteen initial words compile-time constants | structural | `SocketWire::{input_h, iv_const, input_v12}` → `WordRef::Const` | ✓ exact |
| 5 | `emit_xor` — 4 × `ByteAlu[XOR]`, no eval constraint | 4 sends/word | `blake3_socket.rs:584-597` | ✓ exact |
| 6 | `emit_add2` — s bytes, **NO carry column**; `MU·carry·(1−carry)` with `carry := (A+B−s)·2^{−32}` | 4 cells, 1 constraint | `blake3_socket.rs:880-888` | ✓ exact — **the model was revised to match; see §3.1** |
| 7 | `emit_add3` — s + **2 carry columns**; sum identity + two booleanities; NOT a ternary carry | 6 cells, 3 constraints | `blake3_socket.rs:856-878` | ✓ exact |
| 8 | `emit_rotr` — SLL_lo/SLLC_lo/SLL_hi/SLLC_hi (2B each) + Y(4B); 4 mu-gated linear identities | 12 cells, 4 constraints | `blake3_socket.rs:894-930` | ✓ exact |
| 9 | `emit_rotr` — `AreBytes` over the 8 shift bytes = 4 sends | 4 sends/rotation | `blake3_socket.rs:599-607` | ✓ exact |
| 10 | `rotr16`/`rotr8` — FREE byte relabel, no columns | structural | `SocketWire::rotr16/rotr8` permute the `WordRef` byte indices | ✓ exact |
| 11 | `emit_feedforward` — `out[i] = v[i] XOR v[i+8]`, window only | 4 words, via `emit_xor` | `SocketWire::feed_forward_low`; `feed_forward_high` is `unreachable!()` under `FLOW.full_output = false` | ✓ exact |
| 12 | `digest_lane_values` — `MU·(OUT_C[i] − Σ OUTW[i][k]·2^{8k}) = 0`; no range check needed | eval, mu-gated, deg 2 | `blake3_socket.rs:841` (idx 22–25) | ✓ exact |
| 13 | BLOCK 0 — reuse the host's EXISTING cell columns, do not commit a second copy | structural | `cols::{IN0, OUT0, S8}` re-exported from `chips::hash::cols`; no duplicate columns | ✓ exact |
| 14 | MU-GATING — every eval constraint × MU, every send `Multiplicity::Column(MU)`, padding all-zero | structural | `MU = MODE_C` (preprocessed, so prover-unchosen); every BLOCK 1–5 constraint gated; every BITWISE send `Column(MU)` | ✓ exact |
| 15 | MU booleanity + all-zero padding — "NOT BV theorems, checked structurally" | — | emitted as a real constraint (idx 4) and ✓ EXECUTED by `padding_is_satisfied_and_a_real_marked_empty_row_is_not` | ✓ stronger |
| 16 | `tail_truncate` — permitted, off by default | default `False` | not implemented | ✓ conformant |
| 17 | round-0 constant folding — "permitted but must be re-gated" | — | **not done** | ✓ conformant |

**Sends match the model exactly.** ✓ EXECUTED — running `SocketChip(...).build()` at both
round counts gives `census.sends` = **1,190** (6r) and **1,382** (7r), and
`census.aux_cells()/3` = **595** and **691**: identical to the built chip's, to the unit. The
deviation below costs no sends, only columns.

*Version note:* the conformance table is against `chip_model.py` as of its 18:12 revision.
Its `CHIP CONSTRAINT` / `CHIP SENDS` / `CHIP COLUMNS` anchor set is unchanged from the
17:46 version I started against; what changed is `ColumnCensus.sends`, which became a
property summing the contract counter and the I/O tuples instead of a hand-incremented field
that omitted the `ByteAlu[XOR]` sends. I had derived the old accounting as an understatement
and was about to report it — executing the current file refuted that, because it had already
been fixed. Recording it only because it is the reason the two send counts now agree.

### 3.1 The `emit_add2` deviation is RESOLVED — and it went the other way

I reported this as the one deviation: `chip_model.py` witnessed the add2 carry as a column
and constrained it twice, while the implementation derives it as the expression
`carry := (A + B − s)·2^{−32}` and emits one degree-3 constraint. I recommended re-expressing
the model before Phase 4.

**That is done — by the oracle side, not by me.** `chip_model.py` (mtime 20:40) now reads
*"CHIP COLUMNS: s[0..4] bytes. **NO carry column.**"* and *"the model follows the chip"*,
citing the implementation's line range. ✓ VERIFIED by reading it.

The consequence is worth stating precisely, because it closes the gap I flagged:

| ✓ EXECUTED, current `chip_model.py` | 6 rounds | 7 rounds |
|---|---|---|
| model main columns | 2,956 | 3,436 |
| **implementation main columns** | **2,956** | **3,436** |
| model cell-equivalent | 4,741 | 5,509 |
| **implementation cell-equivalent** | **4,741** | **5,509** |
| model sends | 1,190 | 1,382 |
| **implementation sends** | **1,190** | **1,382** |

**Zero delta, on every figure, at both round counts.** The earlier −81/−97 column difference
was entirely the carry column plus the frozen-socket prefix accounting, and both are gone:
the model now counts the socket's 28-column shared prefix the way the chip carries it.

The equivalence argument itself was independently confirmed by the verifier, and by
computation rather than by argument: with `A`, `B`, `s` byte-bound below `2^32` the reachable
integer range of `A + B − s` is `[−4294967295, 8589934590]`, and within that range the field
values `0` and `2^32` have **exactly one** integer preimage each — so a negative difference
cannot alias `2^32 mod p` and the existential really is eliminated by a determined witness.
`INV_SHIFT_32` was confirmed to be `2^{−32} mod p`. The same audit covers add3 and both
rotation identities: only `0` is a multiple of `p` in range, so every field identity in the
arm is an exact integer identity.

### ⚠ 3.1a The recorded gate verdict is STALE — re-run before task #4

`run-gate.log` is **20:09**. `chip_model.py` is **20:40** and `gate.py` is **20:41**.
✓ VERIFIED by `stat`. So the recorded GATE VERDICT: PASS predates the model it is supposed
to certify by half an hour, and it certified the *carry-column* model, not the one now on
disk. **The gate must be re-run before task #4 claims anything about the real chip.** This is
not a defect in either the chip or the model — it is a sequencing artifact of the two sides
converging — but a green log that predates its own inputs is exactly the kind of evidence
that should not be cited.

`ORACLE.md` (20:16) is stale for the same reason: its §3.2 census table still reports the
carry-column figures (main 3,037/3,533, cell-equiv 4,822/5,606), and its §3.2 reconciliation
against the standalone chip is computed from them. The current model gives 2,956/3,436 and
4,741/5,509. The lead's "expected census targets from the gated model" came from that table
and are superseded — the model and the chip now agree exactly, which is a better outcome than
the "small explainable deltas" that was being aimed at.

One small thing for whoever owns the oracle: `chip_model.py`'s `emit_add2` docstring cites
`blake3_socket.rs:826-834`, which was correct when written and is now `880-888` — the O5 doc
block, the D9 tripwire and the D10 rewrite moved it.

### 3.2 Four constraints the model does not cover

`chip_model.py`'s BLOCK 0 says the socket I/O felts "are not modelled in the BV domain". The
implementation adds four framing constraints there, all additions rather than omissions:

| idx | constraint | why |
|---|---|---|
| 0–3 | `S_k − (MODE_P·IN_{8+k} + MODE_C·IV_k)` | keeps the shared capacity prefix meaning the same thing under every hasher |
| 4 | `mode_sum·(1 − mode_sum)` | MU booleanity (item 15 above) |
| 5 | **`MODE_P = 0`** | ✗ no permute socket — see §5 |
| 14–21 | `OUT_{4+j} = 0`, j ∈ 0..8 | the digest is one cell; the upper eight lanes carry nothing |

Total framing constraints 26, hence `NUM_CONSTRAINTS = 26 + 16·NUM_G`.

---

### 3.3 ORACLE.md §7 obligations, and §3.1's degree ledger

I read ORACLE.md §3, §3.1 and §7 after the 18:12 revision landed. Conformance:

| | obligation | status |
|---|---|---|
| **O1** | input lanes range-checked to 32 bits; host must **reject**, not reduce | ✓ **DONE** — mu-gated linear identity per lane (idx 6–13) *plus* the 16 `AreBytes` sends; `lanes_of` returns `None` and `admits` turns it into `LfmExecError::HasherRejected`. Both halves tested, with honest controls. ⚠ **the recorded REASON was wrong — see §3.4** |
| **O2** | the socket is closed on its own output | ✓ **DONE** + tested (`the_socket_output_is_always_a_valid_input`), and exercised for real by the 3-compress program feeding `d0`/`d1` back in |
| **O3** | `compress_iv()` does not participate; the override honoured through `HasherKind::compress`'s explicit delegation | ✓ **DONE** — and this is what forced the executor change; `compress_out` is delegated explicitly alongside `compress` |
| **O4** | byte order is the `keccak_host` convention (one felt = one u32 = four LE bytes), **not** `word::pack_digest` | ✓ **DONE** — `lanes_of`/`word_of` are LE u32; `pack_digest` is never called here. The `lanes_big_endian` control fires |
| **O5** | leaf/parent domain separation | ✗ **OPEN — needs a decision, and it is not mine to make.** See below |
| **R1/R2/R3** | reuse the host's cell columns; no columns for `m[8..16]`; build only the four in-window output words | ✓ **DONE** — all three, rows 13/3/11 of the table above |
| §3.1 | degree ledger | ✓ **CONFORMANT** — every constraint lands inside it, worst = 3, and the rejected ternary carry is not used (two summed carry bits instead). The four host-socket constraints §3.1 does not list are degree 2, 2, 1, 1 |
| §3.3 | tail truncation — OPTIONAL, NOT recommended | ✓ **NOT IMPLEMENTED**, as instructed |

**O5, stated so it does not get lost.** This socket has one tag, so it separates LFM
compressions from other BLAKE3 uses but **not leaves from parents within a tree**. If leaves
ever enter a tree as raw cells rather than through a distinct domain, a variable-depth tree
admits the classic Merkle second-preimage confusion — an internal node replayed as a leaf.
Either fix the tree depth or give leaves the reserved `"LFML"` tag; BLAKE3's own `PARENT`
flag cannot be reused without leaving the standard-hash framing that makes `blake3::hash` a
direct KAT. I have recorded it in `blake3_socket.rs`'s module docs rather than picking an
answer, because it is a protocol decision and nothing in the implementation depends on which
way it goes. It does **not** block anything Phase 2 delivers: the current consumer,
`merkle_walk`, has no leaf-hashing path at all.

Also on the record, from the same section: the digest is 128 bits, so the socket offers
**64-bit collision resistance** by the birthday bound. That follows from
`HASH_DIGEST_FELTS = 4` and the machine's declared 128-bit target, not from BLAKE3 or from
the truncation.

---

### 3.4 ⚠ D10 — the chip was right, the stated REASON for O1 was wrong

The verifier found this by pulling on my own case-(c) surprise, and it is the most
interesting thing to come out of the review. **No constraint changes; three doc sites do.**

**What I had written, in `blake3_socket.rs`, and what ORACLE.md §7 O1 and `chip_model.py`'s
`emit_lane_bytes` docstring also say:** that without the lane check, `v` and `v + 2^32` hash
alike — a free, prover-chosen collision, hence a forged Merkle path.

**That attack is unconstructible against this chip**, and my own failed assertion is the
proof. The mixing core reads the *same linear form* for `m[lane]` that the identity ties
`IN_lane` to (`message_word_ref` → `word_expr`), so `IN_lane` and `m[lane]` are the same
field element by construction. Move the lane and you move the message word. That is exactly
what case (c) of the D2 test demonstrates: absorbing the carry into `MB[3]` satisfies the
lane identity and then breaks `add3` instead.

**What the `AreBytes` sends actually buy** ✓ VERIFIED by reading `run_flow`: the message
words reach `add3` at `blake3_chip.rs:327,333` and **nothing else** — never an XOR — so
unlike almost every other word in this design they get no free byte bound from a consuming
lookup. These 16 sends are `m[0..8]`'s only range check. And `add3`'s exactness needs
`m < 2^32`: in round 0 the `a` and `b` operands are compile-time constants (`input_h`,
`input_v12`, `iv_const` all return `WordRef::Const` ✓ VERIFIED) and the output `s` is
byte-bounded by the XOR that consumes it, so with `m` unbounded a prover solves
`m ≡ s + 2^32·k − a − b (mod p)` for any chosen `s`, puts the whole value in `MB[0]` with the
other three bytes zero — satisfying the identity, since nothing bounds them — and hints the
sibling cell to match. The first `add3`'s output, hence the entire compression, is
prover-chosen.

So the sends are *more* load-bearing than the collision story suggested, not less. Why it
mattered enough to fix rather than wave through: the next auditor reads the O1 bullet, tries
to build the collision, fails exactly as I did, and may reasonably conclude the range check
is redundant.

Fixed in `blake3_socket.rs`'s module docs, the `idx 6–13` eval comment, and two test doc
comments.

**Scope correction — ONE out-of-tree site, not two.** Both the verifier and I initially said
`ORACLE.md` §7 O1 carried the same wrong reason. ✓ VERIFIED by reading: it does not, and
neither does anything else in that file — `grep` for the collision story across `ORACLE.md`
returns nothing. Its BLOCK 1 already gives the *correct* argument, and independently of mine:
*"Without the `AreBytes`, the byte columns are full field elements, one linear equation in
four unknowns leaves three of them free, and the prover chooses the message that gets
hashed."* It even records the supporting fact — *"The message enters `f` only through `add3`
— it is never XORed"* — and notes O1's marginal cost over a chip that merely range-checked
its message is 8 linear constraints. §7 O1 is a milder and also-correct statement about host
and chip disagreeing.

So the only site still carrying the unconstructible attack is **`chip_model.py`'s
`emit_lane_bytes` docstring, lines 153 and 156**. And per the verifier, that file still
*enforces* `are_bytes` (two sends per lane) — the constraint is gated correctly and only the
prose explaining it is wrong, so it is a comment fix, not a re-derivation, and must not be
allowed to become a reason to defer the gate re-run.

Note the standalone chip already had it right too — `blake3_chip.rs:52` says "all 64 `m`
bytes keep their explicit `AreBytes` (they are never XORed)". So D10 was a localized prose
regression in the socket module plus one stale docstring, not a gap in the design or in the
gate's reasoning.

---

## 4. KAT results

| check | result |
|---|---|
| 15 socket vectors at **6 rounds** vs `socket_digest_rounds(a,b,6)` | ✓ **15/15** |
| 15 socket vectors at **7 rounds** vs `socket_digest_rounds(a,b,7)` | ✓ **15/15** |
| 7-round socket == `blake3::hash(a ‖ b ‖ "LFMC")[0..16]`, message rebuilt from the byte-level spec | ✓ **15/15** |
| the KAT table itself agrees with the crate | ✓ **15/15** |
| primitive at 7 rounds == `blake3::hash`, message lengths 0..=64 | ✓ **65/65** |
| primitive at 6 rounds ≠ `blake3::hash` (round-count discriminator) | ✓ |
| the chip's `OUT` **and** `OUTW` byte columns == the vectors | ✓ 15/15 (`an_honest_row_satisfies_every_constraint`) |
| public output of a proved 3-compress Merkle program == the reference | ✓ (`the_blake3_socket_proves_and_verifies`) |

**SOCKET.md §6's one ✗ DEFERRED row is now discharged**: *"the same equality against the
Rust `blake3` crate — DEFERRED to a build phase, needs cargo."* It is
`blake3_socket_tests::seven_rounds_is_blake3_of_the_domain_separated_message`, and it
re-derives the 36-byte message from §2.1's byte-level form rather than calling
`socket_message`, so the word-level and byte-level routes remain two statements that can
disagree. So is the last row (*"the chip's `OUT` columns match these vectors"*).

### Framing negative controls

All 14 fire: `swap_a_b`, `tag_changed`, `tag_omitted`, `tag_slot_moved`,
`truncate_high_half`, `flags_parent`, `flags_no_root`, `block_len_64`, `block_len_32`,
`counter_one`, `cv_zero`, `lanes_big_endian`, `msg_perm_swapped`, `other_round_count`.

Each must change the digest on **every** vector whose effective trace differs from the
honest one, and applicability is *derived* (initial state + the message schedule at every
round + the output window) rather than hand-listed — a hand-list goes stale as controls are
added, and a stale entry is a control that looks covered and is not. Inapplicable cases are
asserted to produce the *same* digest, which checks the applicability derivation itself.

> Worth recording: writing applicability over the *permutation* instead of the *schedules*
> produced a false failure on the `a_one` vector, whose message has `m[2] = m[6] = 0` — so
> transposing the first two permutation entries yields an identical schedule and the control
> genuinely cannot fire. `socket_ref.py` gets this right for the same reason; I got it wrong
> first and the test caught it.

---

## 5. Deviations from the brief, and why

### 5.1 ✗ No `permute` socket — the BLAKE3 arm implements `compress` only

`LFM_HASH` has two modes. SOCKET.md §7 states plainly that the `permute` socket (12 felts
in, 12 out) is **not specified**: no mapping decision, no vectors, and a security argument
that is not the same argument as `compress`'s. Its §7 sketch is labelled "a sketch, not a
decision — unreviewed". Building an unreviewed, un-KAT'd crypto framing is exactly what rule
9 forbids, and it would also roughly double the arm (12 feed-forward words and 12 lane
decompositions instead of 4 and 8).

So: **the AIR pins `MODE_P = 0`** (idx 5), making a program containing a `permute`
*unprovable* under BLAKE3, and `LfmHasher::admits` refuses it at execution with a message
naming SOCKET.md §7. Defence in depth, both directions ✓ EXECUTED
(`a_permute_row_is_refused_under_blake3`, `a_permute_marked_row_violates_the_air`).

**Practical consequence, and this is the one thing to carry forward:** `edsl::merkle_walk`
(which compresses) works under BLAKE3; `edsl::SpongeVar` (which permutes) does not. So
**`TrivialV0` and `FriToyV0` cannot be proved under BLAKE3** — both contain a `permute` —
and **the F3.4 disclosure is only half retired**. This matches existing task #8. The
prove/verify acceptance criterion is therefore met with a purpose-built compress-only
program (`compress_program_source`, two leaf merges and a parent merge — the Merkle-parent
shape the socket exists for, which also exercises O2 by feeding socket outputs back in as
inputs) rather than with `trivial_program`.

### 5.2 `LfmHasher` gained two methods

`compress_out` and `admits`. Both are defaulted, so no existing implementor changes
behaviour. `compress_out` was unavoidable: without it the executor's inlined
permute-and-truncate silently bypasses any overriding `compress`, and BLAKE3 must override
(O3). `admits` is how "reject, not silently reduce" (PLAN §3.2, SOCKET.md O1) becomes a
returned error rather than a panic.

`Blake3Permutation::permute` **panics**. It is unreachable — `admits` rejects first and the
AIR pins `MODE_P = 0` — and every value it could return would be a hash the chip does not
prove. Documented as such at `blake3_socket.rs:267`.

### 5.3 `blake3-6round` is a cargo feature, not a runtime parameter

The *host* reference is runtime-parameterised (`socket_digest_rounds(a, b, rounds)`), so the
KATs pin both variants in one run. The *chips* cannot be: their layouts are `8·rounds`
G-blocks wide and their width functions are `const fn`. Default is **7 rounds** (standard,
externally anchored, A6R-free), matching the signed decision; `--features blake3-6round`
selects 6, and it drives both chips through the single `BLAKE3_ROUNDS`.

`blake3_compress_6round` keeps its name and its meaning — it now delegates with
`BLAKE3_SIX_ROUNDS`, not with the knob, so the 6-round vectors it is tested against stay
pinned in every build. The same reasoning applies to the test module's `CANONICAL`
conventions and to `six_rounds_is_not_the_blake3_crate`: both read `BLAKE3_SIX_ROUNDS`
explicitly. Reading the knob in either place would have silently turned a discriminating
control into a tautology at the default — which is exactly what happened on the first run,
and is why those two now name the constant.

Note `make lint` does not cover the feature — I ran `cargo clippy --features blake3-6round`
separately (clean). Worth adding to CI if the 6-round variant is meant to stay supported.

### 5.4 Not done, deliberately

- **Not added to the registry's 6 kinds** — per the brief, a later separate decision.
- **No round-0 constant folding.** The entire initial state is constant, so it is available
  and would be a real saving, but `chip_model.py` says a folded round 0 "no longer matches
  this model" and must be re-gated. Left on the table.
- **`spec/blake3.typ` not updated** — that is task #7.

---

## 6. Test and lint status

| | result |
|---|---|
| `lfm::blake3*` at 7 rounds (default) | ✓ **44 passed, 0 failed**, 2 ignored |
| `lfm::blake3*` at 6 rounds (`--features blake3-6round`) | ✓ **44 passed, 0 failed**, 2 ignored |
| full `lfm::` suite | 263 passed, **19 failed — all pre-existing** |
| `make fmt` | ✓ clean |
| `make lint` (fmt check + 4 clippy passes) | ✓ **clean** |
| `cargo clippy --features blake3-6round` | ✓ clean |

**The 19 failures are the known fixture issue and none is mine.** ✓ VERIFIED by reading each
panic: every one traces to `proof_fixture.rs:73`, `failed to read
executor/program_artifacts/recursion/fibonacci.elf — run make compile-recursion-elfs`, or to
its downstream `ArenaLenMismatch`/epoch-count consequences in `machine_tests`. The set is
`epoch_tests` ×7, `epoch_verify_tests` ×6, `logup_tests` ×1, `machine_tests` ×5. None touches
`LFM_HASH`, and `poseidon_chip_tests` (the closest neighbour, and the one existing test I
edited) passes in full.

### Honest-path controls

Per the standing rule that a rejection test passes equally well when the fix rejects
everything, every rejection test here is paired:

| rejection test | its honest control |
|---|---|
| `tampering_with_the_witness_is_not_accepted` (4 mutations) | `the_blake3_socket_proves_and_verifies` |
| `an_out_of_range_lane_is_rejected_rather_than_reduced` | in-test: the in-range pair is still admitted |
| `a_non_u32_arena_word_fails_execution_under_blake3` | in-test: `execute(&program, &arenas(), …).is_ok()` |
| `a_permute_row_is_refused_under_blake3` | in-test: the same program still executes under `Test` |
| `the_lane_decomposition_binds_the_felt_to_its_bytes` | in-test: `violations(&base)` is empty |
| `the_lane_range_check_is_load_bearing_on_its_own` (D2) | in-test: part (a) asserts the eval set is SILENT, which is what stops the proof-level half from degenerating into a duplicate |
| `padding_is_satisfied_…_empty_row_is_not` | the padding half is itself the control |

---

## 7. What I would look at next

1. **Re-run the gate.** `run-gate.log` (20:09) predates `chip_model.py` and `gate.py` (20:40,
   20:41), so the recorded PASS certified the superseded carry-column model. §3.1a. Nothing
   else stands between the gate and the chip — the `emit_add2` divergence I flagged has been
   closed from the oracle side and the two now agree on every census figure.
1b. **Refresh `ORACLE.md` §3.2** — still the superseded carry-column census as of its 21:27
   edit ✓ VERIFIED. Its BLOCK 1 and §7 are correct and need nothing.
1c. **`chip_model.py`**: fix the `emit_lane_bytes` docstring per §3.4 (lines 153, 156 — prose
   only, the sends are enforced correctly) and update the `blake3_socket.rs:826-834`
   citation, now `880-888`.
1d. *Nit, offered by the verifier and not filed as a finding, recorded so it survives.*
   `ORACLE.md` BLOCK 1's intermediate step — "one linear equation in four unknowns leaves
   three of them free" — is exactly right for the standalone chip, where `m` has its own
   columns. In the **socket** the lane identity pins `m[lane]` to `IN_lane`, so those three
   spare byte degrees of freedom buy the prover nothing: the message word is that same linear
   form either way. The operative freedom is `IN_lane` itself being an unbounded
   prover-hinted felt. ✓ VERIFIED, and the conclusion ("the prover chooses the message that
   gets hashed") is correct on both routes — only the route differs. Worth one sentence if
   someone is editing BLOCK 1 anyway; not worth a change on its own, and **not** a gate
   re-derivation.
2. **The `permute` socket** (task #8) is what stands between this and a fully retired F3.4,
   and between BLAKE3 and the two registered toy programs.
3. **Round-0 constant folding** is a real, unclaimed saving — the whole initial state is
   constant — but it needs a re-gate first.
4. `make lint` does not build the `blake3-6round` feature; if the 6-round variant is meant
   to stay supported, add a lint/test pass for it in CI.
