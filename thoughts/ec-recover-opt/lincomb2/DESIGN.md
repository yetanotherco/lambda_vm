# lincomb2 precompile — design study (candidate D)

Decision doc for replacing the guest's 4× x-only `ecsm_mul` dance with one
`ecsm_lincomb2` syscall computing `Q = u1·P1 + u2·P2`, both coordinates
returned. Companion to `../chips-map.md` (census) and `../gate/RESULTS.md`
(soundness baseline L1–L8). Status: PAPER STUDY — no code changed.

**TL;DR verdict: BUILD.** −70.4% EC committed cells standalone, −74.3%
combined with the AreBytes pairing already in flight; clears the 2× bar ~4×
over. One genuine design finding: the joint chain's incomplete-addition edge
is **attacker-reachable** (constructive attack below, §4) — unlike the
single-scalar chain the gate proved safe (L5b) — so the chip needs NUMS
accumulator blinding, which converts that one lemma from unconditional to a
**named computational assumption** (dlog-class, same family ecrecover already
rests on). Needs user sign-off, like blake3's 6-round assumption.

---

## 1. Why 4 calls exist today, and what one call removes

Current guest flow (`crypto/ethrex-crypto/src/lib.rs`):
`pk = u1·G + u2·R` is evaluated via four x-only queries — `x(u1·G)`,
`x((u1+1)·G)`, `x(u2·R)`, `x((u2+1)·R)` (`lincomb2_with_oracle`,
lib.rs:195-250). The `+1` queries exist ONLY to recover y from an x-only
oracle (`solve_y` λ-linear trick, lib.rs:252-272). Each query is a full
~256-bit scalar mul: 1 ECSM row + ~382 ECDAS rows.

A lincomb2 precompile returning (xQ, yQ) directly:
- 4 chains → 1 joint Shamir/Straus chain (~448 rows),
- deletes `solve_y`, the shared field inversion, and 3 syscall round-trips
  from the guest (~100–150k guest cycles/ecrecover, secondary win),
- the k+1 queries and the x-only sign disambiguation disappear structurally.

## 2. Cost model

Baseline census (gate-confirmed): ECDAS row = 521 logic + 388 interactions
× 1.5 = 582 aux → **1,103 cells**; with AreBytes pairing → **956**. ECSM row
= 1,537. Today/ecrecover = 4×382×1,103 + 4×1,537 ≈ **1.69M cells**.

### Proposed row shape (Design B — addend bus; chosen over Design A, §2.1)

ECDAS' (joint-step row): same 3-relation core (λ/xR/yR conv blocks untouched
— L1/L2/L3/L4 port as-is). Changes: XG/YG (the fixed addend) → XB/YB
(received per-add from a new **Addend bus**); op/next_op → digit selectors
s1,s2,s3 (addend ∈ {P1,P2,P12}) + next-digit bits nb1,nb2 + one `nb_or`
helper column (round bookkeeping, deg-2 def). Logic ≈ **525** (vs 521).

Interactions: Ecdas 2 + Addend receive 1 (mult `Sum3(s1,s2,s3)` —
supported, lookup.rs:1336-1349) + TWO Bit sends (one per scalar stream,
mults nb1/nb2) + AreBytes 196 + IsHalfword 189 = **390** → aux 585 →
**1,110 cells/row** (≈ today's 1,103; paired: 292 → **963**). XB/YB need no
new byte checks — they inherit byte-ness from the publisher's checked
columns via tuple equality (same C4-inheritance the gate used for YR).

Addend binding soundness: the pending digit bits ride the accumulator tuple
(as `op` does today); the Addend receive is keyed `[ts, sel]` with
`sel = s1+2·s2+3·s3` linear in tuple-bound bits; ECSM publishes
`{1:P1, 2:P2, 3:P12}` with witnessed count multiplicities (balance forces
correctness; counts need no range check under C5).

### Rows per ecrecover (random 256-bit u1, u2)

Joint MSB-first, acc seeded = T₀ (NUMS blinding point, §4): doubles = len ≈
255, adds = nonzero joint digits ≈ 0.75×255 ≈ 191, P12 = P1+P2 precompute
1 row (reuses the add machinery via a special round), blinding correction
(subtract 2^len·T₀ from a 256-row preprocessed constant table, keccak_rc
precedent) 1 row → **≈ 448 rows**.

Blinding simplification bonus: `len` no longer needs the exact-MSB pinning
lemma (gate L6.5) — any len ∈ [max_msb+1, 256] yields correct Q (extra
leading doubles just double T₀; the keyed correction absorbs them). Bit
balance still forces every set bit consumed below len.

ECSM' (1 row/ecrecover): today's 667 + 256 more scalar bits + 2nd K_SUB_N +
yP2 (32) + P2 membership relations (2nd x2/q0/q1/c0/c1 ≈ 225) + P12 columns
(64) + xP2/yP2/xQ/yQ canonicalization halfword blocks (§5 — load-bearing,
N6 pattern) + counts (3) ≈ 1,310 logic; interactions ≈ 1,150 → **≈ 3.0k
cells**, ×1 (vs 4×1,537 = 6.1k today).

### Verdict table (EC committed base cells / ecrecover)

| Variant | cells | vs today | confidence |
|---|---|---|---|
| Today (gate-verified) | 1.69M | — | measured census |
| + AreBytes pairing (in flight) | 1.47M | −13.3% | high |
| lincomb2 alone | 0.50M | **−70.4%** | high (±5% on row shape) |
| **lincomb2 + pairing** | **0.43M** | **−74.3%** | high — recommended target |
| lincomb2 + GLV (4-way, phase 2) | 0.29M (0.25M paired) | −83% (−85%) | LOW — signed-digit machinery unmodeled |

At ethrex scale the multiplier is the EC share of total prover cells
(measure first: epoch reports give per-table rows; at 2857 tx ECDAS alone is
~4.4M rows). If EC = 50% of cells, lincomb2+pairing ≈ −37% whole-prover.

### 2.1 Design A (selector columns) — rejected

Carrying P2, P12 in every row (+128 cells) + a witnessed selected-addend
pair (+64) ≈ +190 cells/row ≈ +17%, vs Design B's +1 interaction (+1.5
cells). Degree works in both (selection must be materialized into columns to
keep λ·xB at deg ≤ 3), but B is strictly cheaper and keeps the tuple narrow.

## 3. y-parity binding (the directive's subtlety) — resolved, with one new required check

Who binds recid parity today: the **guest**, entirely. It decompresses R
from (r, v) in software (lib.rs:98-104, proven CPU execution); the chip is
x-only and never sees a y; `solve_y` recovers y(A) relative to the KNOWN
base y (sign flows guest-side; the +1-query consistency check is a
completeness guard, not the parity authority). "Guest-supplied" ≠ trusted:
guest code is proven execution.

Under lincomb2 the same split holds: the guest decompresses (r, v) →
(xP2, yP2) and passes both coordinates; parity remains proven guest logic.
The CHIP's obligations are exactly:
1. **Membership**: yP2² ≡ xP2³ + 7 (off-curve inputs break the step lemma —
   gate L3b/L4 assume on-curve). Second membership block in ECSM'.
2. **Canonicalization — NEW and load-bearing**: yP2 < p. Without it, a
   malicious prover submits yP2' = yP2 + p (fits in 32 bytes when
   yP2 < 2^256 − p): same point mod p, **opposite parity as bytes** — and
   under lincomb2 the sign of P2 changes Q. Same class as the gate's N6
   finding (XR_SUB_P load-bearing). Likewise xQ < p and yQ < p on the
   output (the guest keccaks pk bytes; a +p-shifted coordinate hashes to a
   different address). Three extra halfword blocks, costed in §2.
   (Today's chip deliberately leaves yG sign/canonicity free — sound only
   because x(k·P) = x(k·(−P)); that symmetry is exactly what lincomb2 loses.)

## 4. The incomplete-addition edge is attacker-REACHABLE (key finding) → NUMS blinding

Gate L5b proved A = ±addend unreachable for the single-scalar chain,
unconditionally, from k < N + prefix structure. **That argument does not
survive the joint chain, and no analog exists.** Constructive attack sketch
(all quantities prover-chosen — ecrecover inputs (z, v, r, s) are free bytes;
z is NOT forced through any hash):

1. Pick ρ, set r = x(ρ·G) — the prover now KNOWS dlog_G(R) = ρ.
2. Pick u2; at a chosen step j its consumed prefix is c2.
3. The collision "accumulator = ±addend" at step j reads
   c1 + c2·ρ ≡ t (mod N) for a small known t. Solve for the required c1
   residue; with probability ≈ 2^−j it is a valid j-step prefix value; set
   u1's top bits to it (j small ⇒ cheap retries over u2/ρ).
4. Back out z = −u1·r, s = u2·r mod N. Valid-looking signature whose joint
   chain hits a degenerate add ⇒ λ unconstrained ⇒ forged Q ⇒ forged
   ecrecover ⇒ arbitrary "valid" tx sender in a proven block.

Mitigations considered:
- **Complete addition formulas** (Renes–Costello, projective): sound,
  unconditional, ~3× relations/row — erases most of the win. Rejected.
- **Detect-and-branch rows** (equal-x → tangent variant; A = −B → infinity
  flag + gated bypass): keeps unconditional soundness, but infinity
  representation + branch selection ≈ +2 relation variants, degree pressure,
  and a much fatter L6. Fallback option if the assumption below is refused.
- **NUMS accumulator blinding — CHOSEN**: seed acc = T₀, a fixed
  nothing-up-my-sleeve curve point (hash-to-curve from a spec'd tag, e.g.
  try-and-increment on SHA-256("lambdavm/ecsm/lincomb2/T0/v1")); drain
  subtracts 2^len·T₀ via the preprocessed table. Every intermediate
  accumulator is 2^j·T₀ + (c1·P1 + c2·P2); a collision now implies a known
  linear relation on dlog_G(T₀) — i.e. the attacker computes a discrete log
  nobody knows. Cost: ~2 rows + one 256×~70-cell preprocessed table.
  **Consequence**: chip soundness for this lemma becomes computational —
  "no efficient prover can produce a linear relation on dlog(T₀)" — a
  dlog-class assumption, strictly within what ecrecover/ECDSA already
  assumes for the chain being proven. This must be a NAMED assumption in
  the spec (blake3-6-round precedent) and **requires user sign-off**.
  The final correction add A − 2^len·T₀ is itself degenerate only when
  Q = ∞, handled by the status contract (§6).

## 5. Interface & implementation surface

Syscall (new number, ECSM pattern): a0 = addr to write Q (64 bytes:
xQ‖yQ LE), a1 = addr of (xP1‖yP1) (64B), a2 = addr of (xP2‖yP2) (64B),
a3 = addr of (u1‖u2) (64B) — or split registers per current
`ecsm_addr_ok`/overlap-guard style (execution.rs:97, :432-446); executor arm
mirrors execution.rs:424-456: load, guard, `ecsm::lincomb2_witness(...)`,
store, log. **Status contract**: executor detects degenerate inputs
(u∈{0}, P2 = ±P1 ⇒ P12 degenerate, Q = ∞) cheaply in software and returns
status ≠ 0 (register or sentinel); the guest then takes the existing
pure-Rust `ProjectivePoint::lincomb` fallback (lib.rs:116-117 already has
exactly this unwrap_or_else structure). Status=1 is always SOUND — the
fallback is proven guest code; a lying status only wastes cycles. No trap ⇒
no crafted-tx censorship of a whole block (today's ECSM traps,
execution.rs:450; acceptable there because inputs reaching it are
guest-guarded, lib.rs:210-212 — keep that pattern).

Touch list: `syscalls/src/syscalls.rs` (+1 fn), executor `execution.rs`
(+~80 lines), `crypto/ecsm/src/witness.rs` (joint-schedule replay +
blinding + P12; compute_witness stays for the old syscall — keep both
syscalls alive through the transition), `prover/src/tables/{ecsm,ecdas}.rs`
(major rework — largest chip job since the tables were written),
`trace_builder.rs` collectors, `crypto/ethrex-crypto/src/lib.rs` (net
DELETION: 4-query + solve_y path → 1 call + fallback), a preprocessed
T₀-table chip (keccak_rc.rs pattern), and a NEW spec chapter (none exists
for EC today — stale `spec/src/ecsm.toml` references confirmed dead).

## 6. Verification plan (gate reuse)

- **Port unchanged**: L1 (same conv/carry shape), L2 mechanics (recompute
  the same offsets), L3a/L3b, L4 chord/tangent, L5a (no 2-torsion).
- **Replaced**: L5b → the NUMS reduction argument (named assumption,
  documented + user-signed; the reduction itself is a short pen-and-paper
  proof with its arithmetic steps z3-checked).
- **Redone**: L6 — joint schedule (two Bit streams, Addend-bus binding via
  tuple-carried digit bits, per-scalar counting; skeleton identical, and the
  exact-MSB sub-lemma DROPS, §2); L7 — small-joint-scalar unrollings vs an
  extended oracle (`ec_ref.py` + ~10-line lincomb2 reference).
- **New negative controls**: drop yP2_SUB_P (parity-flip forgery must SAT —
  the §3 attack), drop xQ/yQ canonicalization, addend-selector tamper,
  status-contract mismatch, correction-table wrong-len.
- **Oracle anchors**: Wycheproof ECDSA-verify secp256k1 vectors (exercise
  the sR − zG lincomb shape), the existing 40-sig 3-way ecrecover
  differential (oracle bonus_ecrecover.py), ≥500 random lincomb differential
  vs k256 `ProjectivePoint::lincomb`.

## 7. Alternatives considered

- **(a) GLV on top** (4 half-scalars, 15-entry addend table): →~260 rows,
  extra −41% relative. Real but needs signed-scalar handling (conditional
  point negation, mod-N decomposition relation, |ki| < 2^129 bounds) — new
  proof surface comparable to lincomb2 itself for ~40% more. Phase 2 only;
  the Addend-bus design leaves room (table just grows).
- **(b) x-only lincomb**: 2 calls + solve_y still needed → only −41%, keeps
  the fragile reconstruction. Rejected.
- **(c) w=2 windowed single-scalar** (no interface change): −8%. Rejected.
- **(d) pairing only**: −13.3%, already in flight; composes with everything.

## 8. Recommendation

Ship pairing now (in flight). Then build lincomb2 + pairing (−74.3% EC
cells, ~4× the 2× bar) in this order: (1) user sign-off on the NUMS
assumption + status-contract ABI; (2) oracle extension + T₀ derivation
spec'd; (3) witness + executor behind the new syscall (old one untouched);
(4) chips + gate port; (5) guest switch; (6) bench (extend
gen_ec_bench.sh with an ecrecover-loop guest) + EC-share measurement on a
real block for the whole-prover claim. GLV deferred until lincomb2 benches.

Honest uncertainties: row-shape estimate ±5% (nb/selector column count,
correction-row plumbing); the L6 redo is the riskiest proof work (two
interleaved counting arguments); preprocessed-table plumbing for T₀ assumed
cheap by keccak_rc precedent.
