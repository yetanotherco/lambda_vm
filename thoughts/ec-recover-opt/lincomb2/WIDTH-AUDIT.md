# Width audit — the varying addend (IMPL-PLAN §11 risk 8)

**Verdict: the existing widths still bound it, with 2^39 of headroom, and no new
constraint is required.** `P12` does **not** need canonicalization for this
argument. The proposed `D_INV` relation also fits inside the existing window
scheme.

But the *reason* is not the one the census implies, and the difference matters
for anyone touching the Addend bus later:

> The carry-width argument never depended on the addend being **canonical**. It
> depends on the addend's **limbs being bytes**. Those are different properties,
> and only the second one is load-bearing — but it is *absolutely* load-bearing:
> a single limb of ~2^29 breaks the integer-lifting argument outright.

Byte-ness *is* proven for all four addends, through a chain traced in §3. It
survives only because the Addend bus carries **one field element per byte**. If
that tuple were ever repacked (e.g. `Word4L`, four bytes per element) the
inheritance would silently become invalid — §3.1.

Reproduce:
```sh
cd thoughts/ec-recover-opt/oracle
<venv>/bin/python width_audit.py        # intervals + corners + real-witness carries
<venv>/bin/python width_audit_z3.py 63  # z3 confirmation + negative control (~25 min)
```
Logs: `width_audit.log`, `width_audit_z3.log`.

---

## 1. What was actually asked, split into two questions

Today's ECDAS proves `A + G` where `G` is loop-invariant, external and
canonical. Under lincomb2 the addend varies per row over `{P1, P2, P12,
−2^len·T₀}` and `P12 = P1 + P2` is an interior chip output: byte-bounded, never
proven `< p`.

**A correction to the premise this audit was commissioned under.** The task
described `chips-map.md:93-100` as having "derived the width argument on the old
assumption" that the addend is canonical. Read directly, it did not: item 4 is
titled *"Non-canonical reps mid-chain"* and already states that `xR`/`yR`/`λ`
are only byte-checked, that "relations are mod-p (quotient absorbs), so values ≡
correct mod p", and that curve membership propagates by induction from the
ECSM-checked seed. That reasoning is correct and generalizes to the varying
addend unchanged. What the note does not do — reasonably, since it is a census
of the *existing* single-scalar chips — is say anything about an addend that
varies per row or is produced inside the chip. This audit supplies that, and
confirms rather than repairs the existing reasoning.

Two questions hide under "does the width argument survive", and they have
different answers, so they are kept apart throughout:

| | question | depends on |
|---|---|---|
| **Soundness** | does the field equation `256·c_i − c_{i−1} − S_i = 0` still imply the **integer** equation? | limbs being bytes |
| **Completeness** | can the honest prover always find carries inside the windows and a quotient `≥ 0` fitting 33 bytes? | composed values being `< p` |

A soundness failure is a forgery. A completeness failure is an honest prover
that cannot build a proof. Only the first is a security bug.

## 2. Soundness — unaffected, 2^39 of headroom

The lifting argument (gate L1) needs `Σ 256^i·S_i = 0` over ℤ, which follows
from the carry recurrence only if nothing wraps modulo Goldilocks, i.e.

```
|256·c_i − c_{i−1} − S_i|  <  p_g = 2^64 − 2^32 + 1
```

Read the relations, not the census (`prover/src/tables/ecdas.rs:348-397`, and
its twin `ecdas2.rs`, whose header states the core is byte-for-byte identical
and which imports the very same `CARRY_OFFSET_*` constants at `ecdas2.rs:71`):
**every operand enters `S_i` as an individual byte limb** — `lam(j)`, `xa(j)`,
`ya(j)`, `xg(j)`/`xb(j)`, `yg(j)`/`yb(j)`, `xr(j)`, `yr(j)`, `q[j]`. No composed
256-bit value appears anywhere in a constraint. So the magnitude of `S_i` is a
function of the limb bounds alone, and is completely blind to whether the
composed value is below `p`.

Maximised over byte limbs, bit `op`/`mu`, and carries inside their `IsHalfword`
windows (`c_i ∈ [−offset, 2^16−1−offset]`, with `c_63` pinned to 0 by
`ColIsZero` and `c_{−1}` structurally 0):

| relation | offset | carry window | max \|256c − c⁻ − S\| | vs `p_g` |
|---|---|---|---|---|
| λ | 32636 | `[-32636, 32899]` | 22,771,245 | 8.1×10^11 |
| xR | 8161 | `[-8161, 57374]` | 16,723,665 | 1.1×10^12 |
| yR | 16320 | `[-16320, 49215]` | 18,739,185 | 9.8×10^11 |
| `D_INV` (proposed) | — | — | 14,578,095 | 1.3×10^12 |

Worst case **2^24.4 against a 2^64 modulus — 2^39 of headroom.** These are
over-approximations (repeated variables such as `lam·lam` and `xa·xa` are
treated as independent, the safe direction) computed with the addend limbs
**free bytes, nothing assumed canonical**. Constraining any operand to `[0, p)`
can only shrink them.

**The varying addend changes nothing here** because `P1`, `P2`, `P12` and the
correction constant all occupy the same `xB`/`yB` limb slots the loop-invariant
`xG`/`yG` occupied, with the same byte bounds.

### Machine-checked, not just argued

`width_audit_z3.py` states both steps directly to z3 with the addend limbs free
bytes, at limb indices `i ∈ {0,1,2,3,7,15,31,47,63}`:

- **W1** (the interval arithmetic above is correct): **UNSAT ×36** — 4 relations × 9 indices.
- **W2** (`|256c_i − c_{i−1} − S_i| < p_g` directly, bypassing the interval): **UNSAT ×36**.
- 36,000 random/corner samples in `width_audit.py`: 0 interval violations.

### The negative control, and why it fires unevenly

**N-WIDTH** drops the byte constraint on the addend's limbs only, leaving them
free field elements, and asks the same W2 question:

```
lambda   i=31:sat   i=63:unsat
xr       i=31:sat   i=63:unsat
yr       i=31:unsat i=63:unsat
dinv     i=31:sat   i=63:unsat
```

3 SAT is the *correct* result, not a partial failure, and the two UNSAT patterns
are both explained by the relations themselves — which is a useful check that
the transcription is faithful:

- **`yr` never reads the addend.** Its term is `Σ lam_j·(xa − xr)_{i−j} − ya_i −
  yr_i`; `xb`/`yb` do not appear. Tampering with them cannot change it.
- **At `i = 63` nothing reads the addend either.** Values are 32 bytes, so
  `xb[k] = 0` for `k ≥ 32`; the convolution `Σ_j lam_j·xb_{63−j}` needs
  `j ≤ 31` *and* `63 − j ≤ 31`, i.e. `j ≥ 32` — an empty index set.

So the control fires in exactly the places where the addend is actually read.
Where it fires, wraparound becomes reachable: **byte-ness of the addend limbs is
load-bearing**, it is what this audit rests on, and §3 is therefore the most
important section of this document.

### How much slack is there before byte-ness matters?

A single non-byte limb of about **2^29** is enough to push `3·xa_j·xa_{i−j}` (λ,
`op=0`) or `lam_j·lam_{i−j}` (xR) past `p_g` and destroy the integer identity.
That is 2^29, not 2^63 — the margin against a *malformed limb* is small even
though the margin against *byte* limbs is astronomical.

## 3. Why byte-ness holds for all four addends

`ECDAS2` deliberately does **not** byte-check `XB`/`YB` (`ecdas2.rs:409-410`:
"XB/YB are deliberately absent: they inherit byte-ness from the publisher's
already-checked columns through Addend tuple equality"). Traced to its root, per
selector, by reading `ecsm2.rs:765-800`:

| `sel` | published as | byte-ness root | canonical too? |
|---|---|---|---|
| 1 `P1` | `const_coord(GENERATOR_LE)` | **compile-time constants** — not columns at all | yes |
| 2 `P2` | `coord(X_P2)`, `coord(Y_P2)` | byte-checked at MEMW store time (contract C4, the authority `xG`/`k` already rely on) | yes, for honest input |
| 3 `P12` | `coord(X_P12)`, `coord(Y_P12)` | inherited from the phase-0 `Ecdas` drain (`ecsm2.rs:832-844`), i.e. **ECDAS2's own precompute-row `XR`/`YR`, which carry real `AreBytes` sends** (`ecdas2.rs:418-427`) | **not proven** — see §4 |
| 4 `−2^len·T₀` | `coord(X_T0N)`, `coord(Y_T0N)` | the `EC_T0` preprocessed table — generated constants under a static commitment | yes, by construction |

The chain bottoms out at real range checks and constants; there is no cycle
(`P12`'s byte-ness comes from ECDAS2's `XR`, which is checked directly, not
inherited).

### 3.1 The inheritance is valid only because the bus is unpacked

`addend_tuple` (`ecdas2.rs:295-311`) lays out the coordinates with `coord()` →
`point_coord_busvalues` (`ecsm.rs:274-276`), which is

```rust
(0..32).map(|b| packed(col + b)).collect()   // 32 × Packing::Direct
```

— **one bus element per byte**. Tuple equality is therefore per-limb equality,
and byte-ness transfers limb by limb.

**This is a load-bearing layout choice and should be commented as such.** Had
the tuple packed bytes (`Word4L`, `b₀ + 2⁸b₁ + 2¹⁶b₂ + 2²⁴b₃`, four columns per
element), a receiver could satisfy the same packed value with a different limb
decomposition — `(b₀ + 2⁸k, b₁ − k)` for any `k` — because its own limbs carry
no range check. Reachable limb magnitudes then run to ~2^63, far past the ~2^29
that breaks the integer identity. A future "let's shrink the Addend bus by
packing" optimization would convert this audit's PASS into a forgery, silently.

## 4. Completeness — measured, and structurally unchanged

Honest carries must land in `[−offset, 2^16−1−offset]` and the honest quotient
must be `≥ 0` and fit 33 bytes. This is the half that *does* depend on composed
values.

Measured over **6,346 real rows** from 15 lincomb2 evaluations covering every row
type, including the two that break telescoping:

| relation | honest carry range | window | fits | min offset needed |
|---|---|---|---|---|
| λ | `[-4303, 6728]` | `[-32636, 32899]` | yes | 4303 |
| xR | `[-112, 8308]` | `[-8161, 57374]` | yes | 112 |
| yR | `[-465, 5914]` | `[-16320, 49215]` | yes | 465 |
| `D_INV` | `[-581, 6041]` | (any of the above works) | yes | 581 |

Quotients: λ `[253, 259]` bits, xR/yR/`D_INV` 258 bits — all `≥ 0`, all far
below `2^264`.

Per row type, showing the new shapes are not outliers:

```
Precompute   lambda=[-95, 4619]    xr=[-24, 6773]   yr=[-170, 4729]
Double       lambda=[-4303, 6728]  xr=[-112, 8308]  yr=[-368, 5914]
AddP1        lambda=[-304, 5624]   xr=[-94, 7722]   yr=[-300, 5686]
AddP2        lambda=[-290, 6170]   xr=[-109, 8074]  yr=[-328, 5509]
AddP12       lambda=[-412, 5439]   xr=[-100, 7897]  yr=[-465, 5700]
Correction   lambda=[-146, 5019]   xr=[-13, 7493]   yr=[-251, 5156]
```

The widest carries come from **doublings**, which do not read the addend at all.

**These are the prover's own numbers, not a reimplementation of them.** The
script derives carries and quotients independently from the group law, then
diffs them against `ecsm::lincomb2_witness` through the oracle harness:
**38,076 (quotient, carry-array) comparisons, 0 mismatches**.

### Why the honest bound is unchanged by construction

Beyond the measurement: the honest carry magnitude is a function of the number
of convolution products in a relation and the size of its operands. The varying
addend changes neither — `P1`/`P2`/`P12`/`−2^len·T₀` sit in the same slots the
loop-invariant `xG`/`yG` sat in, with the same term counts, and all four are
canonical for an honest prover (constants, guest-canonical input, a reduced
`ec_add` output, and generated table rows respectively). So gate lemma L2b's
minimal offsets for the single-scalar chain carry over verbatim.

**Honest limitation:** §4's table is a *measurement over 6,346 rows*, not a proof
over all valid inputs. The structural argument above is what makes it
believable; a closed-form worst-case honest bound per relation is the remaining
gap, and it is completeness-only — a miss costs an unprovable honest witness,
never a forgery. The loose analytic bound (`max|S_i|/255 ≈ 87,000`) exceeds the
windows and is therefore too weak to serve as that proof; it must go through the
operands being `< p`.

## 5. Does `P12` need canonicalization?

**No — not for this argument, and not for soundness generally.**

- The width argument needs byte-ness, which `P12` has (§3).
- `P12`'s *value mod p* is pinned by the precompute row's own three relations,
  so a prover cannot publish a `P12` that is not `P1 + P2` mod `p`.
- The only freedom a missing `< p` check leaves is the non-canonical *encoding*
  `v + p`, which requires `v < 2^32 + 977` and denotes the same field element.
  Every downstream relation is mod `p`, so the chain computes the same thing.
  A prover choosing that encoding would more likely just push its own carries
  out of the windows and fail to build a proof — rejection-only.

Contrast `xQ`/`yQ`, which *are* load-bearing: those bytes leave the chip and get
hashed, so a `+p` shift changes the recovered address. `P12` never leaves.

**Price, if wanted as defence in depth** (the lead asked): the same
`XG_SUB_P`-style block already used three times — 16 halfword columns + 16
`IsHalfword` sends + the overflow/carry-bit constraints, per coordinate. Two
blocks ≈ 32 columns + 32 interactions ≈ **80 committed cells on ECSM2's single
row per ecrecover**, i.e. ~0.02% of the ~430k EC cells per ecrecover. Cheap, and
unnecessary. Recommendation: **skip it**, and instead put a comment on
`point_coord_busvalues` recording that the unpacked layout is load-bearing
(§3.1) — that is where the real fragility is.

## 6. The proposed `D_INV` relation

Auditing the relation that closes the NUMS finding
(`FINDING-nums-blinding.log`), in the chip's own idiom:

```
S_i = Σ_j d_inv(j)·(xB − xA)(i−j) − [i = 0] + μ·(R·P)_i − (q3·P)_i
```

- **Soundness width**: worst case 14,578,095, the *smallest* of the four
  relations (it has one convolution product where λ's `op=0` branch has two).
  1.3×10^12 margin against `p_g`. W1/W2 UNSAT at all nine limb indices.
- **Degree**: `op·(d_inv·(xB − xA))` is degree 3 if the relation is `op`-gated —
  the same degree as the existing λ relation, so the ≤ 3 budget is unaffected.
- **Honest carries**: `[-581, 6041]`, comfortably inside any of the three
  existing windows. **No new `CARRY_OFFSET_*` constant is needed** — reuse
  `CARRY_OFFSET_XR = 8161` (window `[-8161, 57374]`) or
  `CARRY_OFFSET_LAMBDA`; both fit with an order of magnitude of slack.
- **Honest quotient**: 258 bits, `≥ 0`, fits 33 bytes with `r = 3p` unchanged.
- **Gating is a cost choice, not a correctness one**: `x = 0` is not on
  secp256k1 (7 is a quadratic non-residue mod `p`, checked), so on a doubling
  row `xB − xA = −xA ≠ 0` is still invertible. An ungated relation is therefore
  satisfiable everywhere; gating by `op` just lets doubling rows carry
  `d_inv = 0`, `q3 = 3p` and zero carries.

## 7. Findings

1. **[no action] The widths hold.** 2^39 of headroom, machine-checked, with the
   addend free-byte and non-canonical.
2. **[no action — premise corrected] `chips-map.md:93-100` is not wrong.** It
   was cited to me as having assumed a canonical addend; it did not (§1). Its
   mod-`p` argument generalizes verbatim. The gap was coverage of a varying,
   chip-produced addend, which this document adds. Recorded so the correction
   does not get lost and nobody "fixes" a note that is already right.
3. **[documentation, please do] Comment `point_coord_busvalues` as load-bearing.**
   The 32×`Direct` layout is what makes byte-ness inheritance valid (§3.1).
   Nothing currently records that packing it would be a soundness change; the
   name reads like a formatting helper.
4. **[no action] `P12` canonicalization not required.** Priced at ~80 cells if
   wanted anyway; recommendation is to skip.
5. **[input for phase D] `D_INV` needs no new offset constant** and no degree
   budget beyond what λ already uses.
