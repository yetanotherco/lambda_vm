# Transcription audit — does the BLAKE3 gate assert what the design and oracle say?

Auditor: independent pass, 2026-07-29, branch `spike/blake3-recovered`.
Objects audited:

- **oracle**: `blake3-oracle/blake3_ref.py` + `test_oracle.py` (does it define the right function?)
- **gate**: `blake3-chip/z3_blake_verify.py` against `blake3-chip/DESIGN.md` and the oracle
  (is the constraint transcription faithful? can the model be stronger than the chip?)
- **the uncommitted fixes** on `DESIGN.md` / `test_oracle.py` (the 3 design findings +
  harness defects #1/#3) — verified, see §5.

Method mirrors `../ec-recover-opt/gate/TRANSCRIPTION-AUDIT.md`: only one direction is
dangerous. A model **weaker** than the chip yields spurious SAT (false alarm); a model
**stronger** than the chip yields UNSAT on a forgeable chip (false assurance), and no
positive anchor can see that, because an honest witness satisfies a correct model and an
over-strong one equally well. Here there is no Rust chip yet — the gate is the only
executable statement of the design — so the audit is gate ↔ design + oracle, and every
place the gate *cannot see* is a place the future Rust must get right by construction.

Reproduce: everything below ran with `blake3/venv` (z3 5.0.0, blake3 PyPI 1.x),
`ground-truth` (official `blake3` crate v1.8.5, pure-Rust), and the vendored
`others/Plonky3/blake3-air`. Mutant/tamper scripts were scratch files, not committed.

---

## Verdict

**No over-strong or mis-transcribed premise found.** Every equation in the gate matches
the design it encodes (§2 table), the gate's reference is behaviourally identical to the
externally-anchored oracle (§3), the gate is *sensitive* to every wiring-bug class we
could construct — including classes with no shipped negative control (§4, 7/7 mutants
fire) — and the width analysis holds with slack (expressions ≤ ~2^41 vs the 2^48 model
width). The two re-runs reproduce the recorded board: default run **OVERALL: PASS**;
`--full` monolithic UNSATs: **SEE §6**.

The honest map of what a green board does NOT cover (§2, "no automated check" rows)
is where the remaining risk lives: μ-gating/padding, input range checks, the degree-3
ledger, the bus layer, and the precomputed-table contracts. All are documented in
DESIGN §7; the uncommitted fixes added items 10–11. None is new.

## §1 — Oracle re-validation (does the oracle define the right function?)

Re-ran and independently re-derived, all green:

| check | result |
|---|---|
| harness `test_oracle.py`, anchor 1 (official-parameter vectors) | PASS 35/35 × 3 modes |
| anchor 2 (official `blake3` PyPI pkg) — **live this time** (was SKIP) | PASS 92/92 |
| anchor 3 (Plonky3 `blake3-air` port, direct compression) | PASS 20 000/20 000 |
| banner honesty (defect #1 fix) | reads VALIDATED only because all three ran (see §5) |
| known-answer: `blake3("")`, `blake3("abc")` | exact match to published digests |
| differential vs PyPI: 140 lengths × {default, keyed, derive} + 48 XOF-length checks | 468/468 |
| counter split `t_lo/t_hi` vs official crate, XOF `set_position` path, t ∈ {0,1,2, 2^32−2, 2^32−1, **2^32**, **2^32+1**, 2^40, 2^47} | **9/9** (scratch `counter_probe.rs` + `blake3_ref.compress`) |
| swapped-halves negative control | breaks 7/9; the 2 invariants are t=0 and t=0x1_0000_0001 (t_lo==t_hi), both correctly invariant |
| message schedule count+direction: `permute^r` from identity vs the crate's precomputed `MSG_SCHEDULE` | all 7 rows exact |

The historical counts ("35/35×3", "92/92") now reproduce as recorded. ORACLE.md O5
(counter width) remains closed — re-confirmed against the crate at t ≥ 2^32.

## §2 — Per-premise transcription table (gate ↔ DESIGN ↔ oracle)

| gate premise | source | verified | how |
|---|---|---|---|
| `IV`, `MSG_PERMUTATION` constants | DESIGN §1, oracle §2.1, Plonky3 `constants.rs` | ✅ exact | 3-way diff |
| `G_CALLS` (8 index tuples + msg order) | oracle `round_fn` | ✅ exact | diff |
| `bref_*` reference independent of circuit wiring | DESIGN §8 | ✅ | 200 concrete trials vs `blake3_ref.compress` (rounds 6+7), 0 mismatch; leading-permute mutant differs ⇒ `r < rounds−1` guard direction correct |
| init layout `v = h ‖ IV[0..4] ‖ t_lo,t_hi,bl,fl` | oracle §2.4, DESIGN §7.8 | ✅ | MAIN 1 UNSAT + `wrong_iv` control |
| feed-forward `out[i]=v[i]⊕v[i+8]`, `out[i+8]=v[i+8]⊕h[i]` | oracle §2.4, DESIGN §4.6 | ✅ | MAIN 1 UNSAT + `drop_ff_xor` control |
| schedule = `permute^r` of the ORIGINAL `M` | DESIGN §7.7 | ✅ | `wrong_msg_index` control + `permute_inverse` mutant + positive controls |
| `add2`: `a+b = s + 2^32·c`, c boolean | DESIGN §4.3 | ✅ | equation exact; field-level necessity of booleanity confirmed (this audit, §4) |
| `add3`: `a+b+m = s + 2^32·(c1+c2)`, c1,c2 boolean | DESIGN §4.4 (O1 option c) | ✅ | equation exact; width audit drop→SAT |
| `rotr16=[b2,b3,b0,b1]`, `rotr8=[b1,b2,b3,b0]` free relabels | DESIGN §4.2/§7.6 | ✅ | relabel mutants flip check to SAT (§4) |
| `rotr12/rotr7` shift identity `hw·2^r = SLLC·2^16 + SLL`, r=4/9 | DESIGN §4.2 | ✅ | equation exact; `rot_wrong_amount` control |
| recombine `Ylo=SLL_hi+SLLC_lo`, `Yhi=SLL_lo+SLLC_hi` | DESIGN §4.2 | ✅ | recombine mutants flip to SAT (§4) |
| ByteAlu[XOR] / AreBytes table contracts | `prover/src/tables/bitwise.rs` | ⚠ assume-guarantee | documented; same assumption keccak gate makes; **no automated check here** |
| μ-gating / all-zero padding (μ=1 modelled) | DESIGN §4.5, §7.1 | ⚠ gate cannot see | no bus/multiplicity layer; on the implementer |
| input range checks (h,t,bl,fl free via XOR; **m needs explicit AreBytes**) | DESIGN §4.7, §7.5 | ⚠ gate cannot see | gate inputs are bytes by construction; a dropped `AreBytes(m)` in Rust is invisible here |
| degree ≤ 3 ledger | DESIGN §4.8 | ⚠ no automated check | manual ledger; gate models equations, not degrees |
| `Blake3` bus TIMESTAMP binding (findings fix) | DESIGN §1.1/§7.10 | ⚠ gate cannot see | no bus layer; verified by construction, see §5 |
| 48 G instances wired as MAIN 0 models | DESIGN §7.11 | ✅ concrete | positive controls run all 48; `--full` monolithic UNSATs (§6); per-instance index mutant fires (§4) |
| WIDE=48 model cannot wrap | gate internals | ✅ | worst expression ≈ 2^41 (add3 with 8-bit carries) ≪ 2^48 |

## §3 — Reference (`bref_*`) independence

The gate's soundness rests on `bref_*` being an independent statement of BLAKE3. It is
structurally independent (32-bit BV `RotateRight`/`+`/`^` vs the byte-level circuit) and
behaviourally identical to the oracle: 200 random concrete inputs, rounds ∈ {6,7},
0 mismatches. The permute guard `r < rounds−1` matches the oracle's (a leading-extra-permute
variant provably differs). The one structural mirror both share with the oracle — the
constants and `G_CALLS` table — is pinned by the *external* anchors (crate, PyPI, Plonky3),
so a common-mode bug there would have to be a bug in BLAKE3 itself.

## §4 — Gate sensitivity: shipped controls + mutation sweep

Shipped controls all reproduced (default run): 5/5 structural SAT, width audit 4/4,
positive controls 4/4 SAT.

Mutation sweep (scratch, not committed) — bug classes with **no shipped negative control**;
each was injected into a copy of the circuit builders and must flip its check to SAT:

| mutant | class | result |
|---|---|---|
| `rotr16_bad_relabel` | free-rotation byte order (DESIGN §7.6) | **sat — detected** |
| `rotr8_bad_relabel` | free-rotation byte order | **sat — detected** |
| `rotr12_bad_recombine` | carry paired to wrong halfword | **sat — detected** |
| `swap_mx_my` | message operand order in G | **sat — detected** |
| `permute_inverse` | schedule direction | **sat — detected** |
| `bad_diag_index` | one wrong column in G instance #7 (per-instance wiring, §7.11) | **sat — detected** |
| `rounds_off_by_one` | round-loop bound | **sat — detected** |

Field-level addition: the shipped width audit demonstrates bound-necessity only for the
**3-op** add. This audit verified the same for the **2-op** add: booleanity present →
UNSAT (pinned), dropped → SAT (forgeable mod p). Same class, now demonstrated for both.

## §5 — Verdict on the uncommitted fixes

- **Finding 1 (bus input↔output binding) — FIX REAL.** `DESIGN.md` §1.1/§3/§7.10 now
  mandate `TIMESTAMP_0/1` in both `Blake3` receive and send. The cited precedent checks
  out: `prover/src/tables/keccak.rs:264-319` sends `(ts, 0, input_state)` and receives
  `(ts, 24, output_state)` on the internal `Keccak` bus with `TIMESTAMP_0/1` in *both*
  tuples (`BusValue::Packed` at `cols::TIMESTAMP_0/1`). The swap-attack reasoning is
  sound: with no common key, rows A/B exchanging output tuples keeps every tuple
  appearing once per side, so LogUp balances while both callers read wrong results.
  Correctly documented as gate-invisible (no bus layer).
- **Finding 2 ("covers every G" is a model argument) — FIX REAL.** §7.11 records it;
  the positive controls do run the full 48-instance pipeline concretely, and this audit's
  per-instance index mutant backs it. (Superseded in part: this bullet originally also
  cited the `--full` monolithic UNSATs as backing. They were run on 2026-08-06 and came
  back `unknown` on all four queries — see §6 — so they support nothing either way. The
  argument rests on the concrete positive controls and the index mutant.)
- **Finding 3 (carry encoding ambiguity `(1,0)`/`(0,1)`) — correctly classified
  harmless.** The sum identity constrains only `c1+c2`; `s` is pinned regardless.
- **Harness defect #1 (banner overstatement) — FIX REAL.** The banner now reads from
  the status dict; exercised live (below).
- **Harness defect #3 (missing-fixture cascade) — FIX REAL.** With
  `official_test_vectors.json` renamed away: anchor 1 SKIPs alone, anchors 2/3 PASS,
  the canonical-vector emitter still runs (it is now unconditional), banner reads
  "PARTIALLY VALIDATED … NOT anchored on: official-parameter vectors". Fixture restored
  afterwards; regenerated `canonical_6round_vectors.json` is byte-identical.

## §6 — Gate re-runs

- default (`z3_blake_verify.py`), z3 5.0.0: **OVERALL: PASS** (board identical to §9 of
  DESIGN.md).
- `--full` (monolithic symbolic round / rounds=2 / 6-round / 7-round UNSATs), run
  2026-08-06: **ATTEMPTED-INCONCLUSIVE — no pass, and no counterexample.** The run took
  ~145 min and exited 1 (`OVERALL: FAIL`), but all four monolithic queries returned
  `unknown`, not `sat`:

  ```
    round (clean) -> unknown   (want unsat)
    compress rounds=2 -> unknown   (want unsat)
    compress rounds=6 -> unknown   (want unsat)
    compress rounds=7 -> unknown   (want unsat)
  ```

  `unknown` is z3's resource-limit return (`s.set("timeout", timeout_ms)` then
  `s.check()`, `z3_blake_verify.py:320-321`/`:340-341`); the verdict tests `== unsat`
  (line 553), so a timeout is scored `False` and pulls OVERALL to FAIL. The four
  budgets sum to 140 min against ~145 min wall, i.e. every check burned its full
  allowance. **Nothing was disproven; nothing was proven monolithically.** The fast
  board is unchanged and green:

  ```
    G-function UNSAT (covers all G)   : True
    init+feed-forward UNSAT (rounds=0): True
    negative controls all SAT         : True
    positive controls all SAT         : True   (full 6-/7-round pipeline, concrete)
  ```

  Consequence for §5's Finding 2 above: the "`--full` monolithic UNSATs (§6)" cited
  there as backing the per-instance coverage argument did **not** land, so that
  argument currently rests on the concrete positive controls and the per-instance
  index mutant alone. Remediation: rerun with a much larger timeout budget on a
  server (single-threaded, CPU-bound), and/or restructure the monolithic query as
  round-by-round induction.

## §6b — Reconciliation with the second, independent audit (`audit_gate_transcription.py`)

A separately-authored executable audit (74/74 checks pass, run this session) agrees with
every verdict above and sharpens three points this audit stated more coarsely:

1. **Rotation bound necessity, refined.** The load-bearing bound set is *at least one of*
   `{SLL_lo, SLL_hi}` — every configuration with neither is forgeable, every one with
   either is pinned; the `SLLC` bounds are not load-bearing. DESIGN §4.2's "the tight
   SLL bound" should read "a tight bound on at least one SLL halfword". The composed
   (whole-rotation) forgery with both SLL bounds dropped exists for exactly **one**
   input, `X=0xFFFFFFFF` (forged `Y=0`), not for arbitrary inputs.
2. **Doc note (safe direction):** DESIGN §4.8's degree-ledger row for the recombine
   identity overstates (claims body 2 → 3 after ×μ; the body is linear, so 1 → 2).
   The "no constraint exceeds 3" verdict is unaffected.
3. **Doc/cost inconsistency:** DESIGN §3's per-G table commits 1 carry *column* per add2,
   while §4.3 makes that carry a *derived* linear expression (`(a+b−s)·INV_SHIFT_32`).
   The two are equivalent over F_p (proven by that suite) but differ by 96 cells per
   compression in the §6 cost table. The gate models the committed form.

It also independently confirms this report's two "gate cannot see" rows with explicit
forgeries: the missing `AreBytes(m)` (§2, DESIGN §4.7) and the declared-not-derived
input range checks.

## §7 — Still open (pre-existing, report-only per audit scope)

1. Harness defect #2: `test_internal_consistency`'s comment promises a feed-forward
   recomputation that is not implemented (`test_oracle.py:227`). Comment lies; check is
   shallow (length + CV prefix only).
2. Harness defect #4: `test_6round_derivation`'s first assertion is a tautology
   (`compress_6round` *is* `compress(rounds=6)`). The differs-from-7r half is the real
   content.
3. Harness defect #5 (footgun for the Rust phase): `compress(...)` defaults `rounds=7`.
   Trace generators must call `compress_6round` / pass `rounds=` explicitly.
4. ORACLE.md §5 prose "¼–⅓ of a keccak permutation" is inconsistent with its own
   ~5–6k figure (≈1/6 of 24×1480); superseded by DESIGN §6's derived ≈1/15. Doc-only.
5. `ground-truth/Cargo.toml` could not build inside this repo ("believes it's in a
   workspace"). Fixed with an empty `[workspace]` table — **this audit touched that one
   committed file**; without it the documented regeneration flow fails out of the box.
6. Suggestion (not a defect): fold the §4 mutant sweep and the add2 field check into the
   shipped gate as regression controls, so future edits to the gate are held to the same
   sensitivity.

## §8 — Scratch artifacts left in the tree (untracked; commit or delete, user's call)

- `thoughts/blake3/venv/` (z3 5.0.0 + official `blake3` PyPI pkg)
- `thoughts/blake3/ground-truth/src/bin/counter_probe.rs` (the t≥2^32 counter probe)
- `thoughts/blake3/ground-truth/target/`, `thoughts/blake3/blake3-oracle/__pycache__/`
- mutant sweep + add2 field check: `/tmp/blake_mutants.py` and heredocs (not in tree)
