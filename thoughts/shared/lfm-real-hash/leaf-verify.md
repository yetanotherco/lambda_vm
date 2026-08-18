# `MODE_L` / `LFML` leaf mode — adversarial verification

**Verdict: ONE SOUNDNESS DEFECT (D1), high severity, in the field-native arms —
the BLAKE3 arm itself is sound.** Plus two claim/coverage defects and three
documentation defects. The leaf mode's *own* machinery — the canonicity block,
the lane-identity gating change, the tag separation, the transcript layering —
is **CONFIRMED SOUND** under adversarial reading and execution.

**Ground:** worktree `lambda_vm-blake3-impl`, branch `blake3-real-hash`, on
committed B1 (`9bcc9ee2`), uncommitted. Nothing fixed, nothing committed. All
probe edits reverted and md5-verified back to their pre-probe bytes.

Claims are ✓ EXECUTED / ✓ VERIFIED (read + traced) / ✗ UNVERIFIABLE.

---

## 0. Verdict table

| # | target | verdict |
|---|---|---|
| 1a | leaf half-lanes bound by byte decomposition + `AreBytes` | **CONFIRMED SOUND** ✓ VERIFIED |
| 1b | same-linear-form: lane == message word in leaf mode | **CONFIRMED SOUND** ✓ VERIFIED |
| 1c | a mode where neither lane identity nor leaf block applies but the core runs | **CONFIRMED SOUND** (impossible) ✓ VERIFIED |
| 2 | canonicity algebra (idx 30–45), `Z`/`GINV` abuse, the alias | **CONFIRMED SOUND** ✓ VERIFIED + ✓ EXECUTED |
| 3 | `absorb_felts` + transcript layering, call sites, sweep | **SOUND under BLAKE3**; **broken under Test/Poseidon by D1** |
| 4 | selector shift, by-name discipline, one-hot span, #915 mults | **CONFIRMED SOUND** ✓ VERIFIED |
| 5 | `TrivialV0` left on `LFMC` | **CONFIRMED SOUND** ✓ VERIFIED |
| 6 | `HostTree`/`HostSponge` hasher parameterisation (D6) | **CONFIRMED SOUND** ✓ VERIFIED |
| 7 | claim verification by execution | **1 DEVIATION** (D4) — everything else reproduces |
| 8 | hygiene, registry re-bless, O5 doc accuracy | **3 DOC DEFECTS** (D5, D6, D7) |
| — | **the leaf mode's own AIR under the OTHER two hashers** | ★ **DEFECT D1** |

---

## ★ D1 — `MODE_L` rows leave four value columns free under `Test` and `Poseidon`

**Severity: HIGH.** Soundness. Not disclosed anywhere in the report or the spec.
✓ EXECUTED, two independent ways, with a BLAKE3 control that fires.

### The defect

`MODE_L` reads ONE input cell. Three places were updated to say so, and one was
not:

| what | where | leaf-aware? |
|---|---|---|
| the bus receive for the 2nd cell | `chips.rs:635` `reads_two() = Sum3(MODE_C, MODE_T, MODE_P)` | ✓ excludes `MODE_L` |
| the validator's address-slot pin | `validator.rs:232` `ins[mode.num_input_cells()..] == Addr(0)` | ✓ |
| the BLAKE3 AIR's value-column pin | `blake3_socket.rs:1324-1327` (idx 26–29) `mode_l · IN_{4+j} = 0` | ✓ |
| **the `Test` AIR** | `chips.rs:755-762` — round 0 reads `A_i = IN_i` for **i < 8** | ✗ **no pin** |
| **the `Poseidon` AIR** | `chips.rs:838-845` — round 0 reads `A_i = IN_i` for **i < 8** | ✗ **no pin** |

So on a `MODE_L` row under `Test` or `Poseidon`, columns `IN4..IN8`

- receive nothing from `LfmMem` (multiplicity is literally zero there), and
- are pinned by no constraint, and
- **are read by the permutation the AIR proves.**

Four Goldilocks felts of free prover choice per leaf row. `leaf(c)` stops being
a function of `c`.

**Why it was missed** is worth recording: the codebase already tolerates
unconstrained `IN` lanes — `executor.rs:112-114` says "lanes 8–11 are
unconstrained on those rows" — and that was safe *because nothing reads them*.
`MODE_L` is the first mode whose unread cell is nevertheless read by an AIR.

### Executed leg 1 — the AIR admits it (probe, since reverted)

A `MODE_L` row over fixed data `IN0..4`, built twice: `IN4..8 = 0` and
`IN4..8 = [1,2,3,4]`, each with the honest permutation output. Every constraint
of the set evaluated with the production `ProverEvalFolder`:

```
TEST arm  : violations []  for both rows; digests differ
            honest [8100894340827603473, 14773174770469813971, …]
            forged [1250454659320479132,  7922735088962689630, …]
POSEIDON  : violations []  for both rows; digests differ
            honest [18148906729156086505, 16418236894463812223, …]
            forged [507135091515794632,   11308586702453336741, …]
BLAKE3 control: violations [26, 27, 28, 29]      ← the pin fires
```

### Executed leg 2 — end-to-end, bus included

Executor patched to write attacker junk into `state[4..8]` **and** `in_cols[4..8]`
on every `Leaf` row, with the host `LfmHasher::leaf_out` default given the same
junk (an attacker controls both the trace and the arena hints, so this is
exactly their position). Then
`leaf_tests::fri_toy_proves_and_verifies_under_every_hasher`, `LFM_LEAF_JUNK` set:

```
Test      → proved and VERIFIED
Poseidon  → proved and VERIFIED
Blake3    → proved, then verification FAILED   ← the control
```

The LogUp bus balances, the proof verifies against the same `program_id`. This
is not an AIR-only artefact.

### Consequence (reasoning, not executed)

- **Merkle-root binding survives.** `FriToyV0`'s walks end at a root pinned to a
  public input, and hitting a fixed root still needs a preimage; the four free
  felts do not help (Poseidon's capacity is separately pinned to zero on a leaf
  row by idx 0–3).
- **Fiat–Shamir does NOT survive.** `programs.rs:618-619` derives every challenge
  after `absorb_felts(t0w); absorb_felts(t1w)`, and `absorb_felts` is
  `leaf` → `absorb` (`edsl.rs:106-109`). The leaf digest there is compared with
  nothing, so a prover re-randomises the junk, recomputes forward — no inversion,
  no search — and **chooses `alpha`, `zeta0`, `zeta1` and all four query
  indices**, with the public statement (both roots) unchanged. That is a
  complete FS break for any program that absorbs data.

### Why this matters today, and where it does not

All six `LFM_REGISTRY` entries are `hasher: HasherKind::Test`
(`registry.rs:271,355,439,523,607,691`) and `HasherKind::default() == Test`
(`hash.rs:198`). `TestPermutation` is already non-cryptographic, so D1 adds no
*new* exploit to the blessed configuration. **Poseidon is where it bites**: it is
a named production candidate, it is exercised by
`fri_toy_proves_and_verifies_under_every_hasher`, and MODE_L breaks its FS
soundness where B1's `absorb2` did not. BLAKE3 — the intended production hasher —
is unaffected.

### Fix shape (not applied)

Mirror idx 26–29 in `eval_test` and `eval_poseidon`: `mode_l · IN_{4+j} = 0` for
`j ∈ 0..4`, degree 2, four constraints each. The honest-path control matters here
— `leaf_out`'s default already writes zeros there, so honest rows keep proving.

---

## 1. The lane-identity gating change — CONFIRMED SOUND

The highest-risk item in the brief, and it holds.

**(a) All eight half-lanes ARE byte-bound on a leaf row.** ✓ VERIFIED.
`bitwise_interactions` (`blake3_socket.rs:946-957`) sends `AreBytes` over
`(lane_byte(l,0), lane_byte(l,1))` and `(lane_byte(l,2), lane_byte(l,3))` for
all 8 lanes, with multiplicity `mu() = Sum3(MODE_C, MODE_T, MODE_L)` —
`MODE_L` **is** in the sum (`blake3_socket.rs:913-919`), so all 32 byte columns
are range-checked on a leaf row. The canonicity block's `u32` premise is
therefore established, not assumed. The trap the spec's §2.2 warning names is
not present.

**(b) The same-linear-form property is preserved.** ✓ VERIFIED. `lo` and `hi` in
the leaf binding are `word_expr` over `cols::lane_byte(2i, ·)` / `lane_byte(2i+1, ·)`
(`blake3_socket.rs:1345-1352`), and `message_word_ref(i)` for `i < 8` is
`WordRef::Cols(word_cols(cols::lane_byte(i, 0)))` (`:726`) — **the same columns**.
So `IN_i = m[2i] + 2^32·m[2i+1]` is an identity over the very words the mixing
core consumes. A leaf row cannot bind one value and hash another.

**(c) No mode admits the core with neither gate.** ✓ VERIFIED.
`mu = digest_mu + mode_l` by construction (`MU_COLUMNS` = C,T,L;
`DIGEST_MODE_COLUMNS` = C,T), so `mu ≠ 0 ⇒ digest_mu ≠ 0 or mode_l ≠ 0`, and each
of those makes its block bite (a nonzero field scalar does not weaken
`s·(x) = 0`). Fractional-selector rows such as `MODE_C = x, MODE_T = −x,
MODE_L = 1` do exist in the AIR's solution set — they satisfy idx 4 and 5 and
blend `m[8]` to an arbitrary field element — but that is the **pre-existing**
M5/M6 class, answered by the registrar's one-hot check, which now covers
`MODE_L` (§4). No new hole.

---

## 2. The canonicity block (idx 30–45) — CONFIRMED SOUND

With `lo, hi < 2^32` established by (1a) and `mode_l = 1`:

| constraint | with `hi = 2^32−1` (`G = 0`) | with `hi ≠ 2^32−1` (`G ≠ 0`) |
|---|---|---|
| canon-b `1 − Z − G·GINV` | forces `Z = 1` **whatever `GINV` is** | with canon-a's `Z = 0`, forces `GINV = G^{-1}` |
| canon-a `Z·G` | vacuous | forces `Z = 0` |
| canon-c `Z·lo` | forces `lo = 0` | vacuous |

`Z` is **fully determined** by `hi` in both branches, so **no prover-chosen
`GINV` can make the check vacuous** — the one abuse the brief asked about.
No wraparound is possible: `hi ≤ 2^32−1 ≪ p`, so `G = (2^32−1) − hi` is zero
exactly when `hi` is maximal.

The accepted set is exactly `{(lo,hi) : ¬(hi = 2^32−1 ∧ lo ≥ 1)}`, whose
complement is exactly the `2^32 − 1` pairs encoding `v ∈ [p, 2^64)` — i.e.
**exactly `v < p`**, tight at `p − 1 = (lo 0, hi 2^32−1)`.

**Indices, recomputed independently:** `LEAF_IDX = 26`, `base = 30 + 4i`, so
felt `i`'s (binding, canon-a, canon-b, canon-c) sit at `30+4i … 33+4i`:
felt 0 → 30,31,32,**33**; felt 1 → 34–**37**; felt 2 → 38–**41**;
felt 3 → 42–**45**. The report's "canon-c for felt 0 is idx 33" ✓, and the other
seven of the eight canon-c/others land where claimed. Framing total
`4+1+1+8+8+4+4+16 = 46 = CORE_IDX` ✓.

**The alias** ✓ EXECUTED (`m10_a_leaf_row_cannot_skip_canonicity`, and
independently re-derived here): `1 + 2^32·(2^32−1) = p ≡ 0`, so felt `0` has a
second half-pair; the binding constraint is satisfied by it and **only canon-c
catches it**. The test asserts the index rather than "something fired", which is
the right shape.

**Degree** stays 3: canon-a/b/c are `mode_l · (deg-2)`, binding is `mode_l ·
(deg-1)`. Pinned by `the_arm_emits_its_constraints_at_degree_3` ✓ EXECUTED.

---

## 3. `absorb_felts` and the transcript layering

**(a) The two-step is mirrored exactly.** ✓ VERIFIED.
Machine `edsl.rs:106-109`: `let d = b.leaf(c); self.absorb(b, d.as_cell())`.
Host `fixture.rs:113-116`: `let d = self.hasher.leaf(c); self.absorb(&d)`.
Same order, same count (1 `LFML` + 1 `LFMT` per call). Under BLAKE3 the binding
is to the data up to leaf-hash collision resistance, i.e. the socket's already
declared 64-bit birthday bound — **not a new weakening**, since a direct absorb
would have had the same bound through the compress. Under Test/Poseidon this
argument is void — see D1.

**(b) Every call site is the right one.** ✓ VERIFIED, one by one:

| site | absorbs | call | correct? |
|---|---|---|---|
| `programs.rs:608` | `main_root` | `absorb` | ✓ digest |
| `programs.rs:612` | `l1_root` | `absorb` | ✓ digest |
| `programs.rs:618-619` | `t0w`, `t1w` | `absorb_felts` | ✓ data |
| `fixture.rs:262` | `main_tree.root()` | `absorb` | ✓ |
| `fixture.rs:295` | `l1_tree.root()` | `absorb` | ✓ |
| `fixture.rs:322-323` | `t0`, `t1` | `absorb_felts` | ✓ |

**(c) Sweep — no other raw-felt absorb, and no other `LFMC` leaf.** ✓ VERIFIED,
performed independently of the report's §4 and reaching the same four sites.
`SpongeVar` appears in exactly one program (`programs.rs:606`); the only
`compress` call sites in non-test code are `edsl.rs:170` (parent over two leaf
digests), `edsl.rs:187` (`merkle_walk` parent), `fixture.rs:146,170` (the host
mirrors) and `programs.rs:69-71` (`TrivialV0`, §5). Nothing else.

---

## 4. The selector shift — CONFIRMED SOUND

- **By-name discipline held.** ✓ VERIFIED. No literal `9`/`10`/`11` reaches a
  hash path: `compiler.rs:343-350` uses `layout::hash::MODE_*`/`MULT*`,
  `validator.rs:316,415` likewise, `blake3_socket.rs:593` re-exports by name, and
  `airs.rs:187-189,428-432` reads `hash::num_columns(hasher)` and
  `layout::hash::PREP_WIDTH` — which is why `airs.rs` needed no edit at all.
- **The one-hot span covers all four.** ✓ VERIFIED. `MODE_C=6, MODE_P=7,
  MODE_T=8, MODE_L=9` and `one_hot(&g.hash, "LFM_HASH", MODE_C, NUM_SELECTORS=4)`
  (`validator.rs:313-318`) walks columns 6..10 — `MODE_L` inside, `MULT0..2`
  (10,11,12) outside. The §7.2 mistake is not repeated.
- **#915 multiplicity bounding follows the move.** ✓ VERIFIED.
  `validator.rs:415` lists `vec![hash::MULT0, hash::MULT1, hash::MULT2]` by name,
  so `check_mult_ranges` reads 10/11/12.
- **`is_real` widened correctly**: `selector_sum(MODE_C, 4)` (`chips.rs:624`).
- Note (pre-existing, restated because `MODE_L` widens it): `validate` is called
  only from tests — it is a *registration-time* gate, and the runtime binding is
  `program_id`. Correct as designed; the one-hot claim rests on every registered
  program having a `validate` test (`machine_tests.rs:32,152,315,607,1324,…`).

---

## 5. `TrivialV0` left on `LFMC` — CONFIRMED SOUND

✓ VERIFIED. `programs.rs:69-71` is `compress(h0,h1) → compress(d0,l2) →
compress(d1,h3)` — a **chain**: every compress after the first consumes the
previous one's digest as its left operand and there is no level structure, so
there is no leaf/parent pair for `MODE_L` to separate. The recorded argument is
at `programs.rs:60-68` and says exactly this, including the consequence.
Its BLAKE3 arenas are `word_of(&[u32; 4])` (`blake3_socket_tests.rs:1344-1350`),
so obligation O1 is satisfied by construction. ✓

---

## 6. Fixture hasher parameterisation (D6) — CONFIRMED SOUND

✓ VERIFIED. `TestPermutation` no longer appears anywhere in `fixture.rs` (import
removed; grep confirms zero occurrences). Every hash goes through the
`HasherKind`: `HostTree::build(hasher, …)`, `HostSponge::with_hasher`,
`host_leaf_hash_pair(hasher, …)`, both tree constructions and both transcript
absorbs. `host_leaf_hash_pair` is
`hasher.compress(&hasher.leaf(c0), &hasher.leaf(c1))` (`fixture.rs:146`) —
2 `LFML` + 1 `LFMC`, the same association and order as `edsl::leaf_hash_pair`
(`edsl.rs:168-170`). ✓

---

## 7. Claim verification by execution

| claim | executed | verdict |
|---|---|---|
| full `lfm::` @7r = 304 pass / 19 fail | **304 / 19 / 7 ignored** (255.8s) | ✓ exact |
| full `lfm::` @6r | **304 / 19 / 7 ignored** (266.5s) | ✓ exact |
| the 19 are the pre-existing set | **name-for-name identical at both round counts**; module split 7 `epoch_tests` / 6 `epoch_verify_tests` / 1 `logup_tests` / 5 `machine_tests` — identical to the B1 record (`b1-verify.md` §8) | ✓ |
| `leaf_tests` 14/14 both counts | **14 @7r**; 6r totals identical ⇒ 14 | ✓ |
| `transcript_tests` 17/17 | **17 @7r** | ✓ |
| **`blake3_socket_tests` 35 @7r / 34 @6r, "one 7r-only"** | **34 @7r** | ✗ **D4** |
| the milestone + negative leg + three-hasher control | all in the 304 | ✓ |
| `TrivialV0` 16,551 and `FriToyV0` 93 rows / 513,081 | `the_programs_cost_what_the_leaf_spec_priced_them_at` asserts `(56,11,26)`, `93`, `16_551`, `513_081` — passes | ✓ |
| `make lint` (fmt + 4 combos) | see §10 | — |
| leaf KATs really rendered from the spec JSON | all **12** digests (5 rows × 2 round counts + `fri_leaf` × 2) present verbatim in `leaf_kats.rs` | ✓ |

### D4 — `blake3_socket_tests` is 34, not "35 @7r / 34 @6r"

**Severity: LOW (claims accuracy).** ✓ EXECUTED: 34 tests pass at 7 rounds. The
file contains exactly 34 `#[test]` and **no** `cfg`-gated ones (grep for
`cfg(…blake3-6round…)` / `cfg(not` returns nothing), so there is no
"7-round-only by construction" test. HEAD had 35; this diff deletes the O1
tripwire and adds none. Both halves of the report's §7 cell are wrong.

Corollary the report also gets wrong: it says "+13 passes are the new leaf
tests", but B1's record is 290 passed and this is 304 — **+14**, while the net
test-count change is +13 (14 new leaf tests − 1 deleted tripwire; 318 → 331
`#[test]` under `prover/src/lfm/`). The +1 does not close from either side's
records. Not a defect in the change, but the "290 → 304" arithmetic in the
report is not the one the files support; the per-module numbers executed above
are the authoritative ones.

---

## 8. D2 — the transcript KAT no longer models `FriToyV0`, and still claims to

**Severity: MEDIUM (stale verified-claim + real coverage loss).** Not disclosed.

`transcript_kats.rs:76-79` says the end-to-end vector's op sequence is

> `absorb, squeeze, squeeze, absorb, squeeze, **absorb2**, 4× squeeze_bits`,
> ✓ VERIFIED against `programs::fri_toy_program_source`

and `transcript_tests.rs:203-205` repeats it. But `fri_toy_program_source` no
longer contains `absorb2(t0w, t1w)` — `programs.rs:618-619` is now
`absorb_felts(t0w); absorb_felts(t1w)`, i.e. leaf-then-absorb. The model that
replays the preamble was not updated: `transcript_tests.rs:416` still calls
`sponge.absorb2(&mut b, h[2], h[3])`, and the host replay at `:304` and `:574`
likewise.

- The **step count** claim survives (2 absorbs either way, so `FRI_TOY_COMPRESSIONS
  = 11` still holds and the cost test still passes) — which is precisely why
  nothing went red.
- The **state** claim does not. The end-to-end vector, which is the one anchor
  rendered from an independent Python reference, now pins a transcript
  `FriToyV0` does not run. `FriToyV0`'s actual challenge derivation is left
  checked only by machine-vs-host agreement — and host and machine were changed
  together, so a shared error in the leaf convention would not be caught.

The `✓ VERIFIED` marker on a claim that the same change set falsified is the
part worth flagging: it is exactly the marker a future reader will trust.

---

## 9. Documentation defects

**D3 — spec criterion 4 was replaced, not met, and the board says otherwise.**
Severity: LOW-MEDIUM (claims accuracy). `LEAF.md` §5 criterion 4 requires "a
deliberately non-canonical arena value must make the proof fail, and fail *for
canonicity*". The delivered negative leg is
`fri_toy_rejects_a_fixture_built_under_another_hasher` — a **hasher-mismatch**
test whose failure mode is a Merkle-walk root mismatch, not canonicity; the
mismatched fixture's arena values are ordinary canonical `FE`s. The report's §5
lists it under criterion 4 without saying it is a substitution, and its board row
reads "negative canonicity leg **in the assembled proof** — ✓ EXECUTED". There is
no such leg: canonicity is exercised only at the chip level (`M10`).
In fairness the criterion as written is **unsatisfiable** — every `FE` is
canonical by construction, so a non-canonical arena value cannot be built
(`admits`' leaf arm at `blake3_socket.rs:528-536` is dead code by the same
argument, as its own comment concedes). The right disposition is to record the
criterion as retired-because-impossible, not to mark it met.

**D5 — the O5 retirement is asserted hasher-independently where it is
BLAKE3-only.** Severity: LOW. `instr.rs:75-79` (`HashMode::Leaf`, a
hasher-independent ISA doc) states flatly: "Leaves and parents now occupy
different hash domains by construction (`"LFML"` vs `"LFMC"`), so an internal
node cannot be replayed as a leaf whatever the tree's shape." Under `Test` and
`Poseidon` that is **false** — `leaf`, `transcript` and `compress` are the same
function. `layout.rs:82-89`'s selector/domain table has the same problem. The
caveat *is* recorded correctly at `hash.rs:104-108` (`leaf_out`) and the report's
§9 last row names it, so this is a placement defect rather than an omission: the
weakened statement lives in the hasher-specific file and the unqualified one in
the hasher-independent files, which is backwards.

**D6 — `blake3_socket.rs:1319-1323` calls the `IN4..8` pin "hygiene rather than
soundness".** Severity: LOW, but it is the comment that would talk a reader out
of the fix D1 needs. On the BLAKE3 arm the claim is true (its message lanes come
from the byte columns, not from `IN`), but the comment reads as a general
statement about the mode, and the identical pin is load-bearing on the two arms
that do read `IN4..8`.

**D7 — diff scope has grown by one file since the report.** Severity: NONE,
noted for the record. `thoughts/blake3/socket-kats/SOCKET.md` is now modified
(the `"LFML"` row flipped from "reserved" to "LIVE"), which closes the report's
§9 open item. It appeared at 12:54, after this review started, and is another
workstream's edit — not part of the 19+2. Code scope is otherwise exactly as
claimed: 19 modified `.rs` all under `prover/src/lfm/`, 2 new
(`leaf_kats.rs`, `leaf_tests.rs`), nothing in `crypto/`, `executor/` or
`prover/src/tables/`. No `println!`/`dbg!`/`TODO`/`FIXME` anywhere in the diff.

---

## 10. Registry re-bless and lint

Registry: all six entries re-blessed in one pass, and each is pinned by a drift
test that also covers roots/log_heights/keccak_rnd_chunks/hasher — all six pass
inside the 304. The report's `program_id` prefixes reproduce
(`TrivialV0` → `7087e2838dae1171` at `registry.rs:273-277` ✓, spot-checked).

`make lint` — ✓ EXECUTED, **exit 0, clean**, all combos including the
`lambda-vm-prover/cuda` pass. Reproduces the report.

**Probe hygiene:** the three files touched by the executed probes
(`hash.rs`, `executor.rs`, `mod.rs`) were snapshotted before editing and
restored after; md5s match byte-for-byte and `prover/src/lfm/zz_probe.rs` is
deleted. The working tree is exactly as handed over.

---

## 11. What a follow-up should do

1. **Fix D1** — four constraints in each of `eval_test` and `eval_poseidon`,
   with an honest-path control (`leaf_out`'s zeros must still prove). Consider
   deriving the pin from `HashMode::num_input_cells()` in one place so the next
   mode cannot repeat it.
2. **Re-point the transcript end-to-end KAT at the real `FriToyV0` preamble**
   (D2), or delete the `✓ VERIFIED against fri_toy_program_source` claim and say
   plainly that the vector models an `absorb2` transcript.
3. Correct §7's `blake3_socket_tests` cell (D4) and the criterion-4 disposition
   (D3).
4. Move the leaf/parent-separation caveat into `instr.rs` and `layout.rs` (D5).
5. The z3 oracle's WA8 "canonicity dropped ⇒ SAT" leg remains the only thing
   that can show the block is *necessary* rather than merely satisfied; nothing
   here substitutes for it.
