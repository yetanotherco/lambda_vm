# B1 adversarial verification — the compress-chain transcript

**Ground:** worktree `lambda_vm-blake3-impl`, branch `blake3-real-hash`, parent
`2957c3f9`, uncommitted (19 modified `.rs` under `prover/src/lfm/` + `SOCKET.md`
+ 2 new files). **Date:** 2026-08-11. **Method:** read the code and the diff,
then execute. Nothing here is inherited from the implementer's report; where the
report is quoted it is because I checked the claim.

Claims are ✓ EXECUTED (ran it, output quoted), ✓ VERIFIED (read the code, cited)
or ✗ UNVERIFIABLE.

**Bottom line: no soundness defect found in the change.** Seven defects, all
outside the soundness core: one MEDIUM (a live cost-model constant whose
derivation B1 deleted, missed by the report), one inherited HIGH-if-unrebased
framework gap that B1's central argument leans on, five LOW.

---

## 0. Verdict table

| # | target | verdict |
|---|---|---|
| 1 | selector/layout change (PREP_WIDTH 11→12, MODE_T@8, MULT 9..11) | **CONFIRMED-SOUND** |
| 2 | M5/M6 closure on the verify path | **CONFIRMED-SOUND** — with inherited caveat **D5** |
| 3 | trait additions (`transcript`/`transcript_out`) | **CONFIRMED-SOUND** |
| 4 | trace/AIR/bus coherence of the domain tag | **CONFIRMED-SOUND** — design note **D4** |
| 5 | SpongeVar/HostSponge lockstep | **CONFIRMED-SOUND** (by test, not by construction; the test exists) — latent **D6** |
| 6 | `TrivialV0` public-output shape | **CONFIRMED-SOUND** (verified by search, not by suite-greenness) |
| 7 | O1 tripwire | **CONFIRMED** — all four legs present, passes at both round counts |
| 8 | claim verification by execution | **CONFIRMED** — every number reproduces exactly, zero deviation |
| 9 | hygiene | **CONFIRMED** for the panics, the re-bless and debug leftovers; **D2/D3/D7** on docs |

---

## 1. The selector/layout change — CONFIRMED-SOUND

### (a) The #915 admission bound follows the shift

`validator.rs:400-405` — the multiplicity columns are named symbolically, so
`3638b825`'s bound moved with the layout automatically:

```rust
("LFM_HASH", &g.hash, vec![hash::MULT0, hash::MULT1, hash::MULT2]),
```

`layout::hash::MULT0/1/2` are now 9/10/11 (`layout.rs:113-115`). ✓ VERIFIED: the
negative-multiplicity `Compress` forgery stays closed; **MULT2 is bounded**, and
`MODE_T` is *not* in `mult_columns` (it is a selector, bounded by `one_hot`
instead). There is no hard-coded `8`, `9` or `10` anywhere in the hash paths — I
grepped every `layout::hash::` consumer across `prover/src`.

The one place that *did* hard-code positions was `compiler.rs`'s
`hash_rows.push(vec![…])`, an 11-element positional literal. It was rewritten to
write by name (`compiler.rs:332-350`):

```rust
let mut row = vec![FE::zero(); layout::hash::PREP_WIDTH];
row[layout::hash::IN_ADDR0] = fe(ins[0].0);
…
row[match mode { Compress => MODE_C, Transcript => MODE_T, Permute => MODE_P }] = FE::one();
row[layout::hash::MULT0] = fe(mults[0]);
```

Left positional, the shift would have written `mults[0]` into `MODE_T` and
dropped `MULT2` entirely. **This rewrite is what makes the layout move safe**,
and it is the highest-value line in the diff.

### (b) The one-hot span

`validator.rs:304-308` now reads `layout::hash::NUM_SELECTORS = 3` from
`layout::hash::MODE_C = 6`, i.e. columns 6, 7, 8 = MODE_C, MODE_P, MODE_T —
contiguous and complete. ✓ VERIFIED. Report §7.2's account of why `MODE_T` sits
at 8 rather than 11 is accurate: at 11 it would have been outside this span.

### (c) Does the one-hot check run for every row of every admitted program?

Yes, and — importantly — `validate` is on the real admission path, not only in
tests. `prover/src/bin/compute_lfm_registry.rs:51` calls it for each of the six
programs before emitting the entry:

```rust
validate(program).unwrap_or_else(|v| panic!("{kind} is not admissible: {v:?}"));
```

Coverage of the hash group is total: `one_hot` walks `0..group.real_rows`
(`validator.rs:468`), and `DirtyPadding` walks `real_rows..padded_rows`
(`validator.rs:328-334`) requiring every column zero. No row escapes.

---

## 2. The M5/M6 closure on the VERIFY path — CONFIRMED-SOUND, one inherited caveat

I traced the verify path, not the prove path.

1. `proof.rs:150-166` `lfm_verify` resolves the registry entry and passes
   `entry.roots` — a hard-coded constant table — to `verify_against`. There is no
   path that reads roots off the proof.
2. `airs.rs:429-436` builds the `LFM_HASH` AIR as
   `build_air(…, roots[5], layout::hash::PREP_WIDTH)` →
   `.with_preprocessed(root, 12)` (`airs.rs:340-349`).
3. `crypto/stark/src/verifier.rs:1183-1202`: if `air.is_preprocessed()`, the
   proof's precomputed Merkle root must equal `air.precomputed_commitment()` or
   verification returns `false`; a missing root also returns `false`. The
   transcript then absorbs the **expected** (hard-coded) root, not the proof's.
4. `verifier.rs:548-556`: each query's precomputed opening is Merkle-authenticated
   against that same hard-coded root, leaf-hashed over the whole opening.

So `MODE_C`, `MODE_P`, `MODE_T` are values the prover supplies but cannot
*choose*: any deviation breaks step 3 or 4. The fractional split M5/M6 exhibits
is unreachable through the registry verify path. ✓ VERIFIED.

Mechanism 2 (the registrar) is likewise real, per §1(c) above.

### D5 — INHERITED, HIGH if this branch merges unrebased

The verifier **never consults `air.num_precomputed_columns()`**. The
precomputed/main split is taken from the proof's own opening lengths:

```rust
// crypto/stark/src/verifier.rs:949
let num_precomputed = lde_trace_precomputed_evaluations.len();
let num_base = num_precomputed + lde_trace_main_evaluations.len();
```

The fix — `6949ceb9` *"fix(verifier): pin each trace-opening column width to the
AIR, not just their sum (#909)"*, with the precomputed-split PoC at `03870867` —
exists on other branches but is **not in this branch's ancestry**
(`git merge-base --is-ancestor 6949ceb9 HEAD` → *NOT in ancestry*, and
`verifier.rs` here contains no `num_precomputed_columns` call).

**Not introduced by B1.** But B1's report states mechanism (1) — "the mode
columns are preprocessed, so a prover supplies none of them" — as unconditional,
and on *this* branch its enforcement rests on the precomputed leaf hash alone,
with the width unpinned. The class is the one already recorded in
`opening-width-unpinned-splits`. **Action: rebase onto a main containing #909
before merging, and re-run the M5/M6 control.** Severity is about what happens if
that step is skipped, not about the diff.

---

## 3. The trait additions — CONFIRMED-SOUND

**(a) Defaults preserve Test/Poseidon exactly.** `hash.rs:85-93`:
`transcript_out` defaults to `compress_out`, `transcript` truncates it. Neither
`TestPermutation` nor `PoseidonGoldilocks` overrides them; `HasherKind`'s
dispatch (`hash.rs:236-256`) forwards to the concrete type, which then takes the
default. ✓ EXECUTED: `transcript_tests::the_transcript_proves_and_verifies_under_every_hasher`
proves and verifies the preamble under Test, Poseidon and BLAKE3 — passes.

**(b) No other call site moves Test/Poseidon semantics.** I enumerated every
`HashMode::` branch in non-test code (7 sites: `blake3_socket.rs:295-297,387`,
`compiler.rs:343-345`, `executor.rs:402-411`, `builder.rs:286,299,323`,
`instr.rs:75`); all are exhaustive. On a `Compress` or `Permute` row every
`MODE_T` term added to the Test and Poseidon arms is multiplied by zero:

- capacity `S_i = MODE_P·IN_i + (MODE_C + MODE_T)·IV_i` (`chips.rs:730-737`)
- round-constant scale `m = MODE_C + MODE_T + MODE_P` (`chips.rs:758-759`, `:799`)
- mode-sum booleanity (`chips.rs:771`, `:818`)
- the `LfmMem` receive gate `selector_sum(MODE_C, 3)` (`chips.rs:620`)

The only behavioural movement for those hashers is `PREP_WIDTH` 11→12 shifting
their value columns by one, which is exactly what the re-bless captures. ✓ VERIFIED.

**(c) Executor routing, no cross-wiring.** `executor.rs:395-412`:

```rust
HashMode::Compress | HashMode::Transcript => {
    …
    if *mode == HashMode::Compress { hasher.compress_out(&a, &b) }
    else { hasher.transcript_out(&a, &b) }
}
HashMode::Permute => hasher.permute(state),
```

Compress→`compress_out`, Transcript→`transcript_out`, Permute→`permute`. ✓ VERIFIED.

---

## 4. Trace/AIR/bus coherence of the domain tag — CONFIRMED-SOUND

**The tag cannot disagree with the row's mode, by two independent arguments.**

*Custody, prover side.* `program.instrs` is the single source. The compiler sets
the mode column from it (`compiler.rs:340-346`); the trace filler re-derives
`hash_modes` by filtering the **same** `program.instrs` in the same order
(`trace.rs:132-139`); the executor pushes `records.hash` once per `Instr::Hash`
in the same order. Row *i* of the hash group ↔ `hash_modes[i]` ↔
`records.hash[i]` by construction. ✓ VERIFIED.

*Enforcement, verifier side.* The AIR reads the tag from the **preprocessed**
columns, never from the witness. `TAG_SELECTOR`
(`blake3_socket.rs:518-522`) is `[(MODE_C, TAG_LFMC), (MODE_T, TAG_LFMT)]`;
`message_word_ref(8)` returns `WordRef::ModeSelected(TAG_SELECTOR)`
(`blake3_socket.rs:536`); `word_expr`'s new arm (`blake3_chip.rs:1041-1047`)
emits `Σ main(col)·tag`. That expression enters the mu-gated `add3` sum identity
(`blake3_socket.rs:1000-1013`), so a witness computed under the wrong tag
violates a constraint. ✓ EXECUTED — M1 and M2 do exactly this in both directions
and both are rejected, each with an honest control that the same row in its own
domain passes.

`m[8]` reaches only `add3` — never `byte()` and never `rotr_bytes()` — which is
what makes the `unreachable!()`s in §7.4 unreachable *structurally*, not by luck:
BLAKE3's G uses message words solely in `a = a + b + m`. Empirically confirmed
too, since `socket_wires()` runs at every AIR construction and 52 socket/transcript
tests build it without panicking.

**BITWISE accounting is identical for the two domains.** `bitwise_ops_for`
(`blake3_socket.rs:757-765`) takes `(a, b, tag)` per hash record and runs the same
`ValueFlow` for both modes — same number of XOR and `AreBytes` ops, different
values. The senders are gated by `Multiplicity::Sum(MODE_C, MODE_T)`
(`blake3_socket.rs:711`), which is 1 on every real row of either mode and 0 on
padding. ✓ VERIFIED.

### D4 — LOW: the trace filler takes the tag as an argument, not from the row

`trace.rs:251-255` calls `fill_socket_witness(out, tag_for_mode(hash_modes[row]))`.
But `chip_trace` copies the group into the leading columns of **every** row
(`trace.rs:67-70`) *before* calling `fill` (`trace.rs:71-73`), so `out[MODE_C]`
(=6) and `out[MODE_T]` (=8) are already populated when the filler runs. Two
functions below, the Poseidon filler makes the opposite choice deliberately, and
says why (`trace.rs:80-86`):

> The permutation input is read back out of the row's own `IN`/`S` columns — the
> exact cells round 0's constraints read — rather than from the executor record,
> so the witness cannot describe a different input than the one the AIR constrains.

Same file, opposite discipline. Not exploitable (the AIR constrains the tag; a
mismatch is a failed proof, not a forged one), but it is an invariant held by
caller convention where it could be held by construction — the same shape as the
positional-`vec!` hazard the implementer correctly removed from `compiler.rs`.
The `bitwise_ops_for` feed at `trace.rs:186-197` is the same pattern.

---

## 5. SpongeVar / HostSponge lockstep — CONFIRMED-SOUND

Agreement is **by test, not by construction** — they are separate code — and the
tests that would catch a divergence exist and pass.

Checked for an input shape that diverges, found none:

- **operand order**: machine `transcript_step(state.as_digest(), c.as_digest())`
  (`edsl.rs:96-99`); host `hasher.transcript(&self.state, c)` (`fixture.rs:105`);
  reference `transcript_digest_rounds(state, operand, …)`. All `(state, operand)`. ✓
- **absorb2 ordering**: both are literally `absorb(c0); absorb(c1)`
  (`edsl.rs:104-107`, `fixture.rs:108-111`). ✓
- **squeeze counter timing**: both output the pre-advance state, then advance with
  `SQ(i)`, then increment (`edsl.rs:113-131`, `fixture.rs:115-122`). ✓
- **zero-init**: machine `b.felt_const(FE::zero()).as_cell()` → the word
  `[0,0,0,0]`; host `[FE::zero(); 4]`. ✓

**Interning.** `builder.rs:123-136` keys the constant pool on the canonical
four-lane value, so one `LFM_CONST` row per distinct `SQ(i)` within a program and
distinct `i` are distinct rows. A user constant that happened to equal
`[SQZ0, i, 0, 0]` would *share* the row — harmless, because the separation
`SQUEEZE_MARK` provides is explicitly defence-in-depth: the load-bearing argument
is that the operation sequence is a compile-time constant bound by `program_id`,
which `edsl.rs:16-20` states correctly. ✓ VERIFIED.

Caught by `the_machine_and_the_host_chain_agree_under_every_hasher` and
`the_machine_reproduces_the_end_to_end_vector` if either side moves.

### D6 — LOW, latent, not introduced here

`HostSponge` became hasher-parameterised; `HostTree::build` (`fixture.rs:141-148`)
still hard-codes `TestPermutation.compress`, and `fixture_prove_columns`
(`fixture.rs:216`) calls `HostSponge::new()` (default = Test). Harmless today
because `FriToyV0` cannot run under BLAKE3 at all. When O1 closes, the fixture's
Merkle tree and the machine's `edsl::merkle_walk` will hash with different
functions and the authentication paths will not verify — a completeness trap
waiting at exactly the milestone this work is aimed at.

---

## 6. `TrivialV0`'s public output — CONFIRMED-SOUND, by search

The shape moved from `[d1, permuted_cell, m]` to `[d1, d2, m]`
(`programs.rs:57-68`). I searched rather than relying on the suite:

- `.rs` across `prover/`, `executor/`, `crypto/`: every `TrivialV0` /
  `trivial_program` use is shape-agnostic. `machine_tests.rs:36-142` proves,
  verifies, tampers `claimed[0]`, cross-claims, and pins the registry entry —
  none reads slot 1 or asserts a count. `blake3_socket_tests.rs:772-786` asserts
  only "no permute". `transcript_tests.rs:648-653` asserts row counts `(3, 0)`.
- `.md` / `.typ` / `.py` across `thoughts/` and `spec/`: no hit on the old shape
  (`permuted_cell`, `st[0]`, "one permuted cell").
- Registry metadata carries roots/heights/id, not output shape.

✓ VERIFIED — the report's inference is correct, and now it is a search result
rather than an inference.

---

## 7. The O1 tripwire — CONFIRMED

`blake3_socket_tests.rs:1558-1597`. All four legs the report claims are present
and each is a real assertion:

| leg | line | form |
|---|---|---|
| no permute remains | 1563-1568 | `!instrs.any(mode == Permute)` |
| fixture not u32-laned | 1571-1580 | counts values failing `lanes_of`, asserts `> 0`, with a doc note that a zero count means the test must be replaced |
| refusal is *specifically* O1 | 1584-1591 | `Err(HasherRejected(msg)) if msg.contains("O1")` |
| honest control under `Test` | 1596 | same program, same arenas, `.expect(…)` |

✓ EXECUTED, passes at both round counts.

---

## 8. Claim verification by execution — every number reproduces, zero deviation

| claim | executed result |
|---|---|
| full `lfm::` suite @7r | **290 passed; 19 failed; 7 ignored** (202.0s) ✓ exact |
| full `lfm::` suite @6r (`--features blake3-6round`) | **290 passed; 19 failed; 7 ignored** (180.7s) ✓ exact |
| the 19 are the pre-existing `fibonacci.elf` set, byte-identical at both round counts | ✓ — the two failure lists are identical, name for name |
| `transcript_tests` + `blake3_socket_tests` @7r | **52 passed; 0 failed** = 17 + 35 ✓ exact |
| the three-hasher transcript test | `the_transcript_proves_and_verifies_under_every_hasher` … ok |
| the two cost-exactness tests | `the_programs_cost_what_option_b_priced_them_at` … ok; `the_preamble_costs_eleven_transcript_steps` … ok |
| `make lint` (fmt + 4 feature combos) | **clean** |
| `cargo clippy --features blake3-6round -D warnings` | **clean** |

**No deviation from the report's numbers.** The failure set at 6 rounds:
7 × `epoch_tests`, 6 × `epoch_verify_tests`, 1 × `logup_tests`, 5 ×
`machine_tests` — identical to 7 rounds.

---

## 9. Hygiene — confirmed, with three documentation defects

- **`WordRef::ModeSelected` panics** (`blake3_chip.rs:378-407`): unreachable from
  library callers, structurally (see §4) and empirically (every AIR construction
  walks `socket_wires()`).
- **Registry re-bless completeness**: six drift tests, one per entry
  (`machine_tests.rs:118, 230, 511, 754, 1417, 2281`), each pinning **roots,
  log_heights, keccak_rnd_chunks, hasher and program_id**. `FriToyV0`'s
  `LFM_CONST` group move 4→5 is inside `log_heights` and therefore pinned. All six
  pass. ✓
- **No debug leftovers**: the only `println!`s under `prover/src/lfm/` are in
  `blake3_probe.rs`, untouched by this diff. No `dbg!`, `TODO`, `FIXME`.
- **Diff scope exactly as claimed**: 19 modified `.rs` all under
  `prover/src/lfm/`, plus `thoughts/blake3/socket-kats/SOCKET.md`, plus 2 new
  files (`transcript_kats.rs`, `transcript_tests.rs`). Nothing in `crypto/`,
  `executor/`, `prover/src/tables/`. ✓

---

## Defects

### D1 — MEDIUM. `LFM_HASH_RATE_FELTS` is derived from the construction B1 deleted. **Missed by the report.**

`prover/src/lfm/epoch_verify.rs:428-436`:

```rust
/// Felts an `LFM_HASH` permutation absorbs — the sponge's rate is 2 of its 3
/// state cells (`edsl::SpongeVar`: "state = 3 cells (rate 2, capacity 1)") and a
/// cell is [`super::hash::HASH_DIGEST_FELTS`] felts.
///
/// **This is 2.125× WORSE than keccak's 17** …
pub const LFM_HASH_RATE_FELTS: usize = 8;
```

The value `8` is `2 cells × 4 felts`, taken directly from the three-cell duplex
that `edsl.rs` no longer contains. **Under B1 the chain absorbs one cell per
step, so the rate is 4, not 8.**

This is a live constant, not a comment. It drives the epoch verifier's
permutation-axis projection:

- `epoch_verify.rs:465-471` `leaf_permutations_at_rate`
- `epoch_verify.rs:485-492` `query_permutations_at_rate`
- `epoch_verify_tests.rs:641-706` — the "HASH MATRIX — the PERMUTATION axis"
  block, whose printed ratio and whose `assert_eq!(cand_p, leaf_c + path_and_fri)`
  are the numbers the hash decision cites.

**Concrete consequence.** At the true rate the leaf term roughly doubles, so the
projected candidate/keccak permutation ratio is currently understated. Worse, the
model's *rate-invariance* premise breaks: `epoch_verify_tests.rs:650-655` asserts

```rust
s.fri.num_committed() == 0 || 6 <= LFM_HASH_RATE_FELTS,
"a FRI layer leaf must fit one block at the candidate's rate"
```

which holds at 8 and **fails at 4** — a 6-felt FRI-layer leaf no longer fits one
block, so `epoch_verify.rs:483-484`'s "a FRI layer leaf … fits any rate ≥ 6" and
the "only the leaf term may move with the rate" decomposition both stop being
true. The decision paper rests on the same number
(`others/lfm-hash-matrix-scope.md:128, 228`).

The enclosing test is `the_assembled_epoch_verifier_runs`, one of the 19
currently blocked on `fibonacci.elf` — so this is *unexercised in this
environment* and will surface in CI where the ELF exists.

This is precisely the "fixing the mechanism ≠ restoring the invariant" class. The
report's §6 statement that "the entire diff is 19 files under `prover/src/lfm/`"
is true of the *diff* but was never checked against **semantic dependents of the
deleted sponge**, and this is one.

### D2 — LOW. Two stale `PREP_WIDTH = 11` claims survive. **Missed by the report.**

- `prover/src/lfm/statement.rs:43` — *"`LFM_HASH`'s preprocessed width is 11
  under every candidate"*. `statement.rs` is not in the diff at all.
- `prover/src/lfm/poseidon_chip_tests.rs:546` — *"`PREP_WIDTH` is 11 in both
  layouts"*. The same file's assertion at line 163-165 **was** updated to 12; the
  prose 380 lines later was not.

Doc-only; the conclusions still hold. But `statement.rs`'s comment is the
justification for why `lfm_program_id` folds the hasher tag in, so it is load-
bearing prose in the one file that explains program identity.

### D3 — LOW. Report §6's SOCKET.md claim is now false (a race, not an error).

The report says its `SOCKET.md` §2.2 edit was "backed out … the oracle's pass is
byte-for-byte intact", and separately lists §2.2's `m[8]` row as
"⚠ REPORTED, not edited — one genuine staleness".

`git diff thoughts/blake3/socket-kats/SOCKET.md` **does** now contain the §2.2
row rewritten to `MODE_C·TAG_LFMC + MODE_T·TAG_LFMT` plus a "⚠ UPDATED FOR B1"
note block and a §2.3 rewrite. Timestamps say this is a race, not a
misstatement: `SOCKET.md` mtime `01:00:45`, report mtime `00:55:40`. The oracle
did its re-transcription pass five minutes after the report was written
(`ORACLE.md` `01:00`, `chip_model.py` `01:01`, `gate.py` `01:03`,
`artifact_pin.*` `01:04`, `CHIP-GATE.md` `01:13`).

Net: the report's §8 open items *"the two stale `m[8]` framing rows"* are closed;
the report should be amended rather than the files. I verified the new §2.2 text
matches `TAG_SELECTOR` exactly.

### D4 — LOW. Trace filler takes the tag by argument. See §4.

### D5 — INHERITED, HIGH if merged unrebased. #909 width pin absent from ancestry. See §2.

### D6 — LOW, latent. `HostTree` vs `HostSponge` hasher asymmetry. See §5.

### D7 — LOW. Residual contradiction inside the oracle's updated `ORACLE.md` §2.2.

The word-level table row now reads *"on the built chip, a mode-selected linear
form"*, but the sentence immediately below the table still reads:

> Everything except `a` and `b` is a compile-time constant.

`SOCKET.md` got the corresponding sentence fixed (*"Everything in that table
except `a` and `b` **and `m[8]`**…"*); `ORACLE.md` did not. Since §2.2 is the
table the gate transcribes into `chip_model.py`, the contradiction sits in the
one place a transcriber reads. Reported, not edited — `gate-oracle/` is the
oracle's instrument.

---

## What I did not verify

- **`FriToyV0` under BLAKE3.** Blocked by O1, correctly and with a tripwire; not
  re-derived here.
- **The gate extension** (`chip_model.py` `MODE_T` role, `gate.py` B0a/B0b
  widening). The oracle's files moved during this review (mtimes `01:01`–`01:13`);
  I did not read or run them, per instruction. The chip exposes everything the
  report says it does — `cols::MODE_T`, `cols::MU_COLUMNS`, `TAG_SELECTOR`,
  `tag_for_mode`, constraint indices 0–5 unchanged — ✓ VERIFIED against the source.
- **Squeeze-run entropy analysis.** A spec claim, not a code claim.
