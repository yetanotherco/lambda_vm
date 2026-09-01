# Constraint-degree cost model (d = 3 / 5 / 7)

Branch `degree-cost-model` off `main` @ 528a8411. Worktree `~/workspace/lambda_vm-degree`.

Every claim is marked ✓ VERIFIED (read the code and/or measured it), ? INFERRED, or
⚖ ASSESSMENT.

**Cell convention, used on every number in this document:** base-field-equivalent
`main + 3·aux`, an extension element being three base felts. Composition-polynomial
parts are extension-valued, so each part counts as **3 per LDE point**. The census
convention (`main + aux`) is *not* used anywhere here; mixing them inflates a slope
by ~1.29×.

---

## 0. ★ THE QUESTION: does raising degree change prover cost at fixed cells?

> *"My purpose is to model the cost of a degree-5 constraint. Before, cost was mostly
> trace cells and constraints didn't matter. Is this still the case?"* — Mauro

**Provisional answer: still mostly yes.** At identical committed cells and identical
blowup, moving from degree 3 to degree 5 costs about **+10% prove time** — real, but an
order of magnitude short of the cell term. Cells still set prover cost; degree is a
correction on top, not a new dominant axis.

**And most of that +10% is not the arithmetic.** Raising a degree does two separable
things — more multiplications per row, and more composition parts. `DegreeAir` decouples
them via `LVM_DEGREE_DECLARED`, so each can be priced alone. All three arms use the same
width and rows with zero interactions, so committed cells are identical **by
construction** — the test asserts equality rather than assuming it.

⚠ **PROVISIONAL — laptop, one small shape (2^18 rows × 32 cols, blowup 4, 110 q).**
Server confirmation over a row-size ladder is pending; treat the split as more reliable
than the absolute percentage.

| arm | true degree | parts | mean prove s | n | cells (`main + 3·aux`) |
|---|---|---|---|---|---|
| A | 3 | 2 | 2.2485 | 3 | 8,388,608 |
| C | 3 | **4** | 2.4115 | 3 | 8,388,608 |
| B | **5** | 4 | 2.4734 | 4 | 8,388,608 |

* **C − A = +7.25%** — the extra composition parts
* **B − C = +2.57%** — the degree-5 arithmetic
* **B − A = +10.00%** — total, degree 3 → 5 at fixed cells

**≈72% of the cost of degree is the part count; ≈28% is the raw arithmetic.** The
mechanism that bites is commitment and opening work, not multiplication count. Arm C
exists precisely so this could have been refuted — had parts been free it would have
landed near zero — and it was not refuted.

The 72% half is also the half the recursive verifier pays, per query. It is priced there
in §3 and is nearly free (+0.45% permutations at blowup 4). No contradiction: the prover
commits every part across the whole LDE domain, while the verifier only opens them at
110 points.

**Caveats carried with the number.** `DegreeAir` has `aux = 0`. That is what makes the
isolation exact, but it under-represents a VM where the bus dominates the cells — there
the aux columns enlarge the denominator, making the +10% *smaller*, not larger. So
**+10% is an upper bound on the fraction**, biased in favour of degree 5. Parts in this
AIR are base-field rather than ext3, so its *absolute* parts cost is not the VM's; the
B − C split is unaffected (both arms have 4 parts, so it cancels exactly), and the parts
cost proper is measured separately on the real VM in §3 and §4.

---

## 1. The degree ↔ blowup relationship

### 1.1 Degree sets the quotient part count, linearly

✓ VERIFIED. `LookupAir::composition_poly_degree_bound` (`crypto/stark/src/lookup.rs:1042`)
returns `trace_length * (max_degree − 1)`, and the prover recovers the part count by
dividing it back out (`crypto/stark/src/prover.rs:1259`):

```
parts = max(constraint max_degree, logup_max_degree) − 1
```

| d | 3 | 5 | 7 |
|---|---|---|---|
| composition parts | 2 | 4 | 6 |

### 1.2 The LogUp framework is pinned at degree 3 and cannot exceed it

✓ VERIFIED. `logup_max_degree` = max(3 if any committed pairs, `1 + absorbed`), and
`split_interactions` (`lookup.rs:117`) caps `absorbed` at 2. So the framework's own
constraints are never worse than degree 3, and the effective degree is
`max(base_degree, 3)`.

**Consequence for a VM designer: degree 3 is free.** The bus machinery already forces
it, so an AIR that stays at 3 pays nothing for degree. d = 5 is the first increment
that costs anything.

### 1.3 `max_degree ≤ blowup + 1`, and nothing enforces it

✓ VERIFIED by construction *and* by measurement. The composition polynomial
H = ΣβᵢCᵢ/Zᵢ has degree `(d−1)·N`, but the prover recovers it by interpolating the
`blowup·N` constraint evaluations (`prover.rs`, `interpolate_offset_fft`). It is
therefore representable iff `d − 1 ≤ blowup`.

Measured with a purpose-built degree-parameterised AIR
(`crypto/stark/src/examples/degree_air.rs`: `W` columns of `x_{i+1} = x_i^D`), test
`true_degree_vs_blowup_bound`. **9 of 9 arms matched the prediction:**

| D | blowup | parts | outcome | predicted |
|---|---|---|---|---|
| 3 | 2 | 2 | ok | ok |
| 3 | 4 | 2 | ok | ok |
| 5 | 2 | 4 | **VERIFY_REJECT** | reject |
| 5 | 4 | 4 | ok | ok |
| 5 | 8 | 4 | ok | ok |
| 7 | 4 | 6 | **VERIFY_REJECT** | reject |
| 7 | 8 | 6 | ok | ok |
| 9 | 8 | 8 | ok | ok |
| 9 | 4 | 8 | **VERIFY_REJECT** | reject |

So **blowup 4 permits d ≤ 5; d = 7 requires blowup 8** (parts 6 rounds up to the next
power of two).

**⚠ There is no guard anywhere.** The failure mode is `VERIFY_REJECT`, not a prover
error: the prover cheerfully produces an invalid proof and nothing notices until
verification. Worth a cheap up-front check in `ProofOptions`/AIR construction; filed
as bycatch, not fixed here.

### 1.4 Production runs at degree 3 exclusively

✓ VERIFIED. All ten VM constraint sites declare `max_degree() == 3`, so `parts == 2`
always. That matters more than it sounds: the prover **special-cases 2 parts** with a
hand-written algebraic split (`decompose_and_extend_d2`), and everything else falls
into a generic branch commented *"Fallback for any future AIR with d > 2"* — a
full-size `interpolate_offset_fft` plus a forward LDE per part, **never exercised in
production and never tuned**. Section 4.2 measures that cliff so it can be subtracted
rather than misattributed to degree.

---

## 2. Experiment design

### 2.1 The lever

The framework asserts `measured ≤ max_degree`, never equality, so **over-declaring the
degree is legal**. One constant `VM_MAX_DEGREE` (`prover/src/lib.rs`) now sits behind
all ten `max_degree()` sites, and **both the host prover and the in-guest recursive
verifier read it**, so one value moves both sides of the experiment together.
✓ VERIFIED end-to-end: the real VM proves and verifies at declared 5 and 7.

Because the VM's *true* degree stays 3, H never exceeds the LDE domain, so these arms
are **not** bound by §1.3 — confirmed by proving with parts = 8 at blowup 4. **Degree
and blowup are therefore fully de-confounded**: the part count can be swept at fixed
blowup, with constraint-evaluation work held exactly constant.

What the lever does *not* cover: the extra arithmetic of genuinely higher-degree
expressions. That is §5.

### 2.2 Arms

| arm | parts | blowup | isolates | confounds |
|---|---|---|---|---|
| B0 | 2 | 4 | production baseline | — |
| B1 | 4 | 4 | +2 parts, blowup and constraint-eval fixed | includes the path flip → subtract C0 |
| B2 | 6 | 4 | +4 parts, same | same |
| B3 | 2 | 8 | blowup axis alone | — |
| B4 | 6 | 8 | the deployable d = 7 point | parts + blowup; decomposed via B2, B3 |
| C0 | 2 | 4, forced generic | **the implementation cliff alone** | — |
| S | D−1 | ≥ D−1 | true degree-D constraint evaluation | synthetic AIR, not VM-representative |

`C0 − B0` = implementation cliff. `B1 − C0` = pure marginal cost of parts.
`B3 − B0` = pure blowup. `B4` vs `B2 + B3 − B0` = additivity check.

### 2.3 Instrument

The in-circuit verifier cost is measured **host-side and exactly**, with no guest
builds and no timing noise. Every Merkle hash funnels through two chokepoints —
`hash_streamed` (leaves, variable width) and `hash_new_parent_bytes` (parents, fixed
64 bytes). Behind a `hash-count` feature these now tally leaf hashes, absorbed leaf
bytes, parent compressions, and **exact keccak-f permutations** (`bytes/136 + 1`,
accounting for pad10*1). Counters are reset after proving, so a reading is
verifier-only.

Counting is deterministic ⇒ **one run is an exact integer**; no replicates, no spread
to report. This is why the verifier half of this document needed zero box time.

⚠ **The counters are process-global and the prover hashes on rayon workers**, so a
second arm running concurrently corrupts the reading — silently, and by a lot
(551,944 permutations against a true 158,730). An `ArmGuard` now asserts exclusivity
at **both ends of the measured window**; an entry-only check was written first and
mutation-testing showed it did not fire, because the pollution lands *between* the
reset and the read. The two-point check fired on 3 of 3 polluted runs and passes when
an arm runs alone. Every number in §3.2 was produced by a single-arm run.

---

## 3. The verifier cost of degree — measured

### 3.1 Mechanism

✓ VERIFIED by reading `verifier.rs`. The verifier **never evaluates constraints on the
LDE domain** — only once, at the OOD point z. Degree reaches it through exactly one
channel: the part count. Per query it does

* **one** leaf hash over `2 · parts` extension elements (`verify_composition_poly_opening`), and
* **one** Merkle path whose length depends on the domain size, **not** on parts.

So **degree widens the composition leaf but does not add hashes and does not lengthen
the Merkle walk.** Parents outnumber leaf hashes ~5.6 : 1 (bench_32k: 135,960 vs
14,410), and parts touch none of them.

Confirmed exactly: raising declared degree left `leaf_hashes` and `parent_hashes`
**bit-identical** and moved only leaf bytes — by 105,600 B at blowup 4, against a
prediction of Δparts(20) × queries(110) × 2 × 3 felts × 8 B = 105,600 B. Exact to the
byte.

### 3.2 Verifier keccak-f permutations

| workload | blowup | d = 3 | d = 5 | d = 7 | d5 vs d3 | d7 vs d3 |
|---|---|---|---|---|---|---|
| sub | 2 | 296,526 | 298,716 | 300,906 | +0.74% | +1.48% |
| sub | 4 | 158,730 | 159,830 | 160,930 | +0.69% | +1.39% |
| sub | 8 | 111,836 | 112,566 | 113,296 | +0.65% | +1.31% |
| bench_32k | 2 | 457,929 | — | — | | |
| bench_32k | 4 | 244,420 | 245,520 | 246,620 | +0.45% | +0.90% |
| bench_32k | 8 | 171,769 | 172,499 | 173,229 | +0.42% | +0.85% |

**Marginal cost: 0.5 keccak-f permutations per query per part** (48 B added to a
136 B-rate sponge, quantised up).

Blowup axis at fixed degree: **−29.5%** (sub) / **−29.7%** (bench_32k) for 4 → 8.

### 3.3 ★ The headline

**d = 7 at blowup 8 costs the in-circuit verifier ~29% LESS than d = 3 at blowup 4.**
−28.6% (sub), −29.1% (bench_32k) — stable across a 2.5× difference in Merkle tree
depth, so not a small-workload artifact.

The blowup that d = 7 *forces* cuts queries 110 → 73, and that refunds far more than
its four extra parts cost. **For a recursion VM the degree axis is nearly free; the
query axis is everything.**

### 3.4 Pre-registered prediction for the guest-cycle check

Written **before** the run landed, so the check can refute rather than rationalise.
Arms `a3cd3c0c` (d=7) vs `bb3da304` (d=3), `blowup4` preset, `empty` inner program,
big box, both PRE-949.

The only channel by which degree can cost the guest *more* than the host permutation
counts predict is the DEEP composition loop, which iterates over parts per query. From
the measured totals that is `(88 − 48) parts × 110 queries = 4,400` extra iterations.
At 10–60 guest instructions per extension-field step, that is 44k–264k cycles, i.e.
**0.02%–0.53%** of a 50–200M-cycle proof.

So:

| quantity | predicted |
|---|---|
| keccak calls (d=7 vs d=3) | **+1.3% to +1.5%** (host permutations gave +1.39% on `sub`, the closest workload) |
| guest cycles | **+1% to +2%** — tracking keccak calls closely |
| non-hash tail (cycles − keccak share) | **< 0.5%** |

**Refutation condition: if guest cycles rise by more than ~3%, or by more than about
double the keccak-call delta, the tail is material and §3.3's −29% headline must be
re-derived in cycles rather than permutations.** Anything inside the table above leaves
the headline standing as measured.

### 3.5 Honest scaling

⚖ ASSESSMENT. The lever moves the ten constraint-carrying tables; the other fifteen
(EmptyConstraints / preprocessed) stay at parts = 2, so `total_parts` went
48 → 68 → 88 rather than 48 → 100 → 150. For a VM where **every** table sits at degree
d, scale the penalty by ~2.5×: d = 7 becomes ≈ **+2.25%**, not +0.90%. The conclusion
is unchanged — +2.25% against −29.7% is not a close call.

---

## 4. The prover cost of degree

### 4.1 Committed volume — exact, deterministic

Per trace row, at LDE scale:

```
V(d, blowup) = blowup × ( main + 3·aux + 3·(d − 1) )
```

The `3·(d−1)` term is the composition parts. This yields the equivalence that makes
the whole trade computable:

> ★ **One composition part = one aux column = three main trace columns**, in committed
> base-felt volume at the same blowup.

Measured composition volume (bench_32k, blowup 4), ✓ VERIFIED, exact:

| d | parts | composition cells (LDE) | vs d = 3 |
|---|---|---|---|
| 3 | 2 | 12,196,632 | — |
| 5 | 4 | 14,065,080 | +15.3% |
| 7 | 6 | 15,933,528 | +30.6% |

Perfectly linear (+1,868,448 per degree step). Trace cells (`main + 3·aux`) are
11,961,518 and unchanged across arms, so composition is **25.5% of trace LDE volume
already at d = 3** — the composition commitment is not a rounding error on the prover
side, unlike on the verifier side.

### 4.2 ★ The implementation cliff, isolated

Forcing the 2-part case down the generic path (`LVM_FORCE_GENERIC_PARTS=1`, part count
and everything else held identical) on the isolated `instruments` decomposition timer:

| path | parts | decomposition stage |
|---|---|---|
| fast (`decompose_and_extend_d2`) | 2 | 0.21 s, 0.22 s |
| generic fallback | 2 | 0.44 s, 0.47 s |

**The generic path costs 2.12× the fast path at identical part count** — roughly +6% of
whole-prove time on this workload, entirely implementation, not degree.

The toggle was mutation-tested: an assertion in the generic branch stayed silent with
the flag off and fired with it on, so the arm demonstrably switches paths.

⚖ ASSESSMENT: **a naive d = 5 prover measurement would bundle this 2.12× into "the
cost of degree" and overstate it.** A mature d = 5 implementation would get its own
specialised decomposition, exactly as d = 3 has. Quote the cliff separately.

Holding the path constant (both arms generic), 2 parts vs 4 parts moved the
decomposition timer by nothing measurable (0.455 s vs 0.445 s) — the dominant term is
the single full-size inverse FFT, whose cost is independent of the part count.

### 4.3 What still needs the box

⧗ **NOT MEASURED: peak RSS and wall clock at real scale.** Local workloads run in
~1–4 s with ~15% warmup drift, which swamps the effect; §4.1's volumes are exact but
they are a *proxy* for memory and time, not a measurement of them. The established
affine law (`1.242 GiB + 33.94 B/cell`, R² 0.989) predicts RSS from §4.1's volumes,
but that law was fitted over **trace** cells and I have not verified the slope
transfers to the composition-part population — arms B0–B2 exist to check exactly that,
and they need the box.

---

## 5. What I could not measure, and why

**True degree-5/7 constraint evaluation on the real VM.** The lever inflates the
*declared* degree; it cannot make the VM's constraints genuinely degree 5. Doing that
means re-arithmetising the VM, which is a project rather than an experiment.

This matters less than it appears, for a reason worth stating plainly: **the cost of
evaluating a degree-d constraint is not a function of d.** It is a function of how many
multiplications the expression is written with. `x⁷`, `a·b·c·d·e·f·g`, and a sum of
twenty degree-7 monomials are all "degree 7" and cost wildly different amounts. So a
single "constraint evaluation cost at d = 7" number would be a fiction. The composable
quantity is **cost per constraint-multiplication per row**, which the partner team can
multiply by their own constraint structure; `DegreeAir<F, D, W>` exists to measure it
(sweep D at fixed W, and W at fixed D, reading the `instruments` `R2 evaluate` timer).
That sweep is cheap and not yet run.

Also unmeasured: the **non-hash tail of in-circuit verifier cost**. Per query the DEEP
composition loops over parts in the extension field (~11k extra ext3 operations for
parts 2 → 6 across 25 tables at 110 queries). With keccak accelerated to one ecall, that
tail could be a non-trivial share of guest cycles while being invisible to a hash
counter. One `bench_recursion_cycles.sh` run at blowup4, d = 3 vs d = 7, would bound it.

---

## 6. The usable model

### 6.1 Formulae

```
parts(d)        = d − 1
min blowup(d)   = smallest power of two ≥ d − 1        (d=3→2, d=5→4, d=7→8)
queries(blowup) = ⌈(128 − 20) / −log₂(√(1/blowup) + 1/300)⌉    (Johnson bound)
                = 219 / 110 / 73 for blowup 2 / 4 / 8

prover committed volume per row  V = blowup × ( main + 3·aux + 3·(d−1) )
verifier permutations            ≈ queries × (path + leaf work)
                                   + 0.5 per query per part
```

### 6.2 Applied to the partner team's register budget

Using their numbers — ~40 other columns; at d = 3 registers cost `4·(N+2)` columns in
**both** the VM and its decoding table; at d = 5, N = 3 is free and N = 7 costs 4 extra
columns; at d = 7, N = 5 is free and N = 11 costs 4 extra — plus N for the registers
themselves. `aux = 0` here because their interaction count is unspecified; add
`3·aux` to the bracket when known.

**At fixed blowup 4** (this campaign's operating point; d = 7 is not legal there):

| option | registers | columns | parts | volume/row | vs baseline |
|---|---|---|---|---|---|
| d = 3 | 3 | 83 | 2 | 356 | 1.000× |
| d = 3 | 5 | 101 | 2 | 428 | 1.202× |
| d = 3 | 7 | 119 | 2 | 500 | 1.404× |
| **d = 5** | **3** | **43** | **4** | **220** | **0.618×** |
| **d = 5** | **7** | **51** | **4** | **252** | **0.708×** |

**At each degree's minimum blowup**, with both costs:

| option | registers | prover volume | verifier perms |
|---|---|---|---|
| d = 3 @ blowup 2 | 3 | 1.000× | 1.874× |
| d = 3 @ blowup 2 | 5 | 1.202× | 1.874× |
| d = 5 @ blowup 4 | 3 | 1.236× | 1.005× |
| d = 5 @ blowup 4 | 7 | 1.416× | 1.005× |
| d = 7 @ blowup 8 | 5 | 2.831× | 0.709× |
| d = 7 @ blowup 8 | 11 | 3.281× | 0.709× |

(Verifier column normalised to d = 3 @ blowup 4 = 1.000, from bench_32k.)

### 6.3 Recommendations

⚖ ASSESSMENT, resting on the measurements above.

1. **d = 5 at blowup 4 is the recommendation.** At equal register count it is *strictly
   better than d = 3 on both axes*: −38% prover committed volume (the 40 columns saved
   dwarf the 6 column-equivalents the two extra parts cost) for +0.5% verifier work.
   With **seven** registers it still costs 29% less prover volume than d = 3 with
   three. There is no axis on which d = 3 wins at the same blowup.

2. **d = 7 is a genuine trade, not a free win.** Its cost is almost entirely the forced
   blowup doubling, not its parts: ~2.8× prover volume for 0.71× verifier work. Take it
   only if verifier cost dominates — deep recursion where the proof is re-verified in
   circuit many times. At one or two layers, d = 5 wins.

3. **Do not read d = 3 as the safe default.** It is the *only* degree the framework
   gives away free (§1.2), but its register pricing (`4·(N+2)` in two places) is by far
   the most expensive per register of the three.

4. **Budget the extra parts as trace columns.** One part = three main columns. That
   single conversion answers most "can we afford it" questions without measurement.

5. If d ≥ 5 is chosen, **write a specialised decomposition** for the resulting part
   count. The generic fallback costs 2.12× the hand-written 2-part path (§4.2) and that
   penalty is implementation, not mathematics.

---

## 7. Reproduction

```bash
# ★ the fixed-cells constraint-degree arm (§0) — interleaved ABBA
for R in 18 20 22; do LVM_DEGREE_ROWS_LOG2=$R LVM_DEGREE_BLOWUP=4 LVM_DEGREE_REPS=3 \
  cargo test -p stark --release degree_fixed_cells_sweep -- \
  --ignored --nocapture --test-threads=1 | grep '^DEGREEFIXED'; done
# one arm per process, for peak RSS: LVM_DEGREE_ARM=A|B|C

# degree ↔ blowup bound (§1.3)
cargo test -p stark --release true_degree_vs_blowup_bound -- --nocapture --test-threads=1

# verifier cost (§3) — deterministic, seconds, no box needed
#   set VM_MAX_DEGREE in prover/src/lib.rs to 3 / 5 / 7 between arms
LVM_DEGREE_ELF=bench_32k LVM_DEGREE_BLOWUP=4 \
  cargo test -p lambda-vm-prover --release --features hash-count \
  degree_cost_verifier_hashes -- --ignored --nocapture

# committed volume (§4.1)
LVM_DEGREE_ELF=bench_32k LVM_DEGREE_BLOWUP=4 \
  cargo test -p lambda-vm-prover --release degree_cost_prove -- --ignored --nocapture

# implementation cliff (§4.2) — isolated stage timer, ABBA over the flag
LVM_FORCE_GENERIC_PARTS=1 LVM_DEGREE_ELF=sub LVM_DEGREE_BLOWUP=4 \
  cargo test -p lambda-vm-prover --release --features instruments \
  degree_cost_prove_instrumented -- --ignored --nocapture
```

Peak RSS: run the built test binary directly under `/usr/bin/time -v`, **one arm per
process** (RSS is a high-water mark, so two configurations in one process measure only
the larger). `peak_rss_gib()` does not exist in this tree, so there is no macOS `/proc`
trap to avoid — external measurement sidesteps it.

## 8. Experiment code (not for merge)

| file | purpose |
|---|---|
| `crypto/stark/src/examples/degree_air.rs` | `DegreeAir<F, D, W>` — true degree-D AIR; `LVM_DEGREE_DECLARED` decouples declared degree (parts) from true degree (arithmetic) |
| `prover/src/lib.rs` `VM_MAX_DEGREE` | one knob behind all ten tables; read by prover *and* guest verifier |
| `crypto/crypto/src/hash_count.rs` | leaf/parent/permutation counters, `hash-count` feature |
| `crypto/stark/src/prover.rs` `LVM_FORCE_GENERIC_PARTS` | routes 2 parts through the generic path (mutation-tested) |
| `prover/src/tests/degree_tests.rs` | the measurement arms + the exclusivity guard |
| `crypto/stark/src/tests/air_tests.rs` | `degree_fixed_cells_sweep` (§0), `true_degree_vs_blowup_bound`, `degree_probe_parts_vs_blowup` |
