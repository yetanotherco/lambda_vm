# HASH-SPLIT-PLAN — the fleet endgame: split proving + a specialized blake3 circuit

**Mandate (Mauro, 2026-08-13):** *"450 [GPUs] is a lot, the current sota is 4-8 gpus. We may
need the split proving with some specialized blake circuit."*

**Status: SCOPING / Round-0 projection. Read-only; no code touched, nothing built.**
This joins SOLUTION-ARRAY.md as the fleet-endgame track. Its Round 0 is arithmetic and is
complete in this document; every number is reproducible from
`~/workspace/lambda_vm_bench_cache/lfm_census_2026-08-12/` plus the two scripts named in §7.

---

## 0. Verdict — read this before scheduling anything

**1. The 450 figure reconciles, and the model reproduces it without tuning.** Today's
configuration (2^21 epochs, blowup2/219q, leaf rate 4, no batching, 2-ary tower, hosted
socket) projects **1,267 GPUs**; the partially-optimised region Mauro is quoting from —
rate-8 absorption or partial batching — lands at 400-750. DERIVED, §1.4. So the fleet
question is real and the arithmetic behind it is sound.

**2. ★ Neither the split nor a specialized circuit gets to 4-8 GPUs.** At the best
configuration the model can reach (2^23 epochs, blowup4/110q, rate 8, 4-ary tower, batched
FRI+MMCS, plus every hash-circuit lever in this document), **a hash chip that cost literally
zero still leaves 23 GPUs of work**. The residue — the LFM machine's own arithmetic
marshalling felts into and out of the hash chip — is 18.3 B cells per block and no hash
lever touches it. §2.3. The required cells-per-compression at 8 GPUs is *negative*: the
budget is exhausted before the first compression is priced. **This survives a 10× swing in
the residue coefficient** (§1.4): even at the optimistic end, 8 GPUs needs 891
cells/compression against a ✓ VERIFIED chip of 4,946 and a plausible floor of ~4,060.
**32 GPUs is the reachable endgame; 16 needs everything to go right; 4-8 is not in this
design space.**

**3. And there is no floor worth chasing under the current chip. ✓ VERIFIED from source.**
`blake3_chip.rs` is **one row per compression** (`:781`), **3,056 main** = 112 input bytes +
48 G-blocks × 60 cells + 64 output bytes (`:162, :224`), **1,259 bus interactions → 630 ext
aux** (`:911-913`), of which **1,248 are BITWISE byte-lookups** (`:990-991`). So
**4,946 cells/compression is fully verified**, including the aux width the campaign has been
carrying as inferred. The design is already tight: only the two non-byte-aligned rotations
cost anything (`ROT_SHIFT_R`, `:126-128`), and a tighter encoding of the 14 words per
G-block plausibly reaches **≈4,060** — **a 1.1-1.2× chip, not a 10× one.** §1.3.
**The brief's premise that a specialized blake3 circuit is the lever should be retired.**

**4. ★ The finding that pays for this document is about SHAPE, not about splitting.**
Verification cost is proportional to a table's **WIDTH** and only logarithmic in its rows
(leaf absorption = `Σ_groups 2·cols·kind / RATE`; Merkle depth = `log2_lde − 1`). Proving
cost is `rows × width` — invariant under reshaping. So laying the blake3 AIR out
**narrow-and-tall (one G-call per row, ~100 columns × 48× the rows) costs the same to
prove and 1.5× less to verify**:
- the D1 tower node drops **157 → 103 GiB** at RATE 4 and **104 → 69 GiB** at RATE 8 (§3.2)
  — the best tower-node figure the campaign has, within 8% of the 64 GiB production target,
  and *independent of the split*;
- it is also what makes a split viable at all: at a 2^18-compression shard the
  1-compression-per-row layout needs ρ > 2.50 to break even (we have ρ ≈ 1.7, so it
  **loses**), while the G-per-row layout breaks even at ρ = 1.17 and **wins**. §2.4.

**This reprices D9.** RATE and row-shape are the same lever seen twice, and row-shape is
the cheaper half: it needs no bus-arity change, no `num_input_cells` change, and none of
the lane-map hazards the RATE=5 refute pass found.

**5. What gets from 1,267 to 78 GPUs is scheduling, not silicon; the hash circuit takes it
from 78 to 51.** Ladder (§1.4): batching 1.7×, RATE 4→8 1.5×, tower arity 2→4 1.3×,
**epoch 2^21→2^23 3.7×**, inner blowup4/110q 1.4× — 16.2× cumulative, and the biggest rung
is a *scheduling* decision already unlocked by MMCS-PLAN §1.2 ("epoch size becomes nearly
free after batching") that costs no build. Then the hash-circuit rungs (packed AIR + the two
shape changes) add 1.54×. **Do the scheduling first: it is free, and it is 10× the size of
the thing the brief asked about.**

**6. Cadence and latency are different problems and the brief conflates them.** "4-8 GPUs
keeping up with 12 s blocks" is a *throughput* requirement: fleet size = total work ÷
(12 s × per-GPU throughput), and pipelining across blocks satisfies it. *Latency* — a
block proof available within one slot — is a separate requirement that the current shape
misses by 14-45×: one base wrap is 28-65 s on one GPU and the tower adds 2-3 layers of
69-162 s each. **Intra-proof distribution (§2.1 route d) is the only lever that attacks
latency, and it is also the only route with zero soundness surface.** §3.

**7. One number decides the split's value and it is free to measure.** ρ — the ratio of
total wrap cells to hash-chip cells — is **MEASURED at 1.081 under keccak**
(`census_logs/ethrex_e21_b2_q4.log`: hash chips 92.5% of cells) and **DERIVED at ~1.7-1.9
under blake3**, because blake3 shrinks the hash term ~4× while the marshalling residue
tracks *felts absorbed*, which is hash-independent. The whole split case turns on whether
the blake3-native emitter (`edsl::leaf_hash_pair` over one-cell digests) is as expensive
per felt as the keccak emitter's byte-level sponge packing. **Census one wrap under the
blake3 emitter and read off the hash-chip share.** That is the minimal experiment (§5.1),
it needs no proving, and it is decisive in both directions.

**8. The obstacle to a *true* split is a real open problem, not an engineering gap.** Our
LogUp challenge is sampled only after every main root is in the transcript
(`crypto/stark/src/prover.rs:2447`), so a bus cannot cross proof boundaries. SP1's answer is
a **challenge-free** binding — each interaction is hashed to a septic-curve point, sends
minus receives must sum to the identity, asserted in the recursion layer
(`global_interaction.rs:33-45`, `complete.rs:147`) — ✓ VERIFIED. **But it costs a Poseidon2
permutation per interaction.** That is fine for offloading a keccak-f from a RISC-V shard
and self-defeating for offloading the compression function the glue is made of. Routes
(a)-(c) all need a cheap challenge-free binding at hash rate, and none is known. §2.1.

**Open decisions for Mauro: D10, D11, D12 — §6.**

---

## 1. THE BACKWARDS TARGET

### 1.1 Anchors

| quantity | value | provenance |
|---|---|---|
| GPU throughput | **67.13 M base-field-equivalent cells/s** | 481,327,124 cells (`EXPLORATION.md:186`) ÷ 7.17 s ABBA mean (`BOX-RESULTS.md:53,57`), one RTX 5090, `LAMBDA_VM_GPU_LDE_THRESHOLD=262144`, verify green ✓ MEASURED |
| slot | 12 s | Ethereum |
| bytes per cell (host RSS) | 33.7 | `wrap_tests.rs` `MEASURED_BYTES_PER_CELL` ✓ MEASURED |
| hosted socket | **4,946** cells/compression | 3,056 main + 3×630 aux — ✓ **VERIFIED both**, `blake3_chip.rs:162,224` (main) and `:911-913` (1,259 interactions → 630 ext aux). MMCS-PLAN §5 and `tower.py:15` carried the aux as `? INFERRED`; it is now confirmed. |
| #903 standalone chip | **5,316** cells/compression | 3,219 main / 1,397 sends → 3,219 + 3×699; PA-PLAN.md:541 quoting commit `35038501` ✓ VERIFIED from the commit message, ✗ UNVERIFIED against the branch source |
| epochs per block | 2^20→72, 2^21→36, 2^22→18, 2^23→9 | block 25368371, 74.8M cycles, PLAN.md:8 ✓ MEASURED |

**Budget** = G × 12 s × 67.13 M cells:

| GPUs | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|
| cells/block | **3.22 B** | **6.44 B** | **12.89 B** | **25.78 B** | **51.56 B** |

⚠ **The throughput assumption and what breaks it.** 67.13 M cells/s is measured on a
481 M-cell wrap whose dominant chip (KECCAK_RND, 88.1% of main cells) is device-resident
at 12.7 GiB peak VRAM on a 32 GiB card. Extrapolating it to a 2-4 B-cell wrap assumes:
1. **VRAM.** Linear scaling puts a 1.9 B-cell wrap at ~50 GiB VRAM — over a 5090. The
   measured configuration therefore *cannot* hold; S3 host-recompute or device-recompute
   (SOLUTION-ARRAY B/C+) must run, and the seam audit prices that at **+40-60% wall**.
   Applying that penalty multiplies every fleet number below by ~1.5.
2. **Host RAM.** 33.7 B/cell puts one config-F base wrap at **59 GiB** — inside the 64 GiB
   target, which is the first time in this campaign a wrap has fitted. §3.1.
3. **Fixed costs amortise the other way.** 7.17 s on 481 M cells implies ~26.8 KB of memory
   traffic per cell against the 5090's 1.79 TB/s — i.e. the measured point is *not*
   bandwidth-bound, so a larger proof may run faster per cell. Direction unknown; this is
   the main reason to treat the anchor as ±40% rather than ±10%.
4. **The blake3 chip is lookup-heavy** (~57% of its cells are LogUp aux), and LogUp aux
   generation is batch-inversion-bound, not NTT-bound. A blake3-dominated wrap may have a
   materially different cells/s than a KECCAK_RND-dominated one. **Unmeasured.**

### 1.2 Compressions per block

Per-query leg cost from the calibrated model (`mmcs_project.py`, validated to the unit
against the measured census at four points — MMCS-PLAN §1.0), RATE-parameterised, × queries
× N epochs. `rate` = felts of message the socket absorbs per compression invocation; the
byte-optimal value is 8 (two 4-felt cells = one 64-byte BLAKE3 block).

**Base layer (blake3 inner, after P-a), compressions per BLOCK:**

| epoch | N | batching | RATE 4 | RATE 5 | RATE 8 |
|---|---|---|---|---|---|
| 2^20 | 72 | off | 59,891,040 | 54,085,680 | 43,797,600 |
| 2^20 | 72 | **ON** | 33,580,800 | 27,632,880 | **18,501,120** |
| 2^21 | 36 | off | 68,559,264 | 62,654,148 | 51,687,504 |
| 2^21 | 36 | **ON** | 34,374,240 | 28,295,676 | **18,961,020** |
| 2^23 | 9 | off | 34,352,559 | 32,612,166 | 28,636,659 |
| 2^23 | 9 | **ON** | 10,245,258 | 8,410,257 | **5,593,698** |

DERIVED from the calibrated model. Two readings: **batching is worth 1.8× (2^20) to 5.1×
(2^23)** on compression count, and **RATE 4→8 is worth a flat 1.8×** everywhere.

**Tower**, N leaves aggregated k-ary: total proof-verifications = N + ⌈N/k⌉ + ⌈N/k²⌉ + …
Node cost = one LFM-proof verify at 110q, native LFML/LFMC (`tower.py` construction).

| epoch | arity | nodes | depth | verifications | tower comps (rate 8, batched) | as % of base |
|---|---|---|---|---|---|---|
| 2^21 | 2 | 38 | 6 | 73 | 28,281,660 | 149% |
| 2^21 | 4 | 13 | 3 | 48 | 18,596,160 | 98% |
| 2^21 | 8 | 6 | 2 | 41 | 15,884,220 | 84% |
| 2^23 | 2 | 11 | 4 | 19 | 7,360,980 | 132% |
| 2^23 | 4 | 4 | 2 | 12 | 4,649,040 | 83% |
| 2^23 | 8 | 3 | 2 | 11 | 4,261,620 | 76% |

**★ The tower is not a rounding error — it is 76-149% of the base layer.** Every campaign
number quoted so far has been per-wrap; the block costs roughly *twice* the base layer.
Arity 2→4 removes a third of it and halves the depth; 4→8 adds little (§3.3).

⚠ This holds node cost constant across layers, which is only legitimate under D1's
**static-shape premise** ("14 fixed tables, known log-heights, wrap options fixed → the
program shape is static per (K, options)", PLAN.md:184-190). If a node's own proof is
larger than what it verifies, upper layers cost more and the tower diverges. **Gate D1 must
demonstrate the fixed point, not just the single node.** ✗ UNVERIFIED.

### 1.3 What a compression costs, and the floor

**★ The existing chip is already within ~10% of its floor. ✓ VERIFIED from source**
(`lambda_vm-blake3-impl@blake3-real-hash`, `prover/src/lfm/blake3_chip.rs`):

| fact | value | citation |
|---|---|---|
| **one row per compression** | — | `:781` *"One row per compression; padding rows are ALL ZERO."* |
| G-instances | `NUM_G = BLAKE3_ROUNDS * 8` = **48** at 6r | `:101-102` |
| per-G columns | `G_SIZE = 60` — *"56 bytes + 4 carry bits"* | `cols::G_SIZE`, `:158` |
| input bytes | `4 × IN_U32` = 112 (`h[32] | m[64] | t_lo|t_hi|len|flags[16]`) | `:104, :156-157` |
| output bytes | `4 × OUT_U32` = 64 | `:106, :161` |
| **main columns** | 112 + 48×60 + 64 = **3,056** = `NUM_COLUMNS(3072) − PREP_WIDTH(16)` | `:162, :224` |
| **bus interactions** | **1,259** → aux = ⌈1259/2⌉ = **630 ext** | `:911-913` `Vec::with_capacity(1_259)` |
| of which BITWISE lookups | **1,248 per compression** | `:990-991` `ops.len() * 1_248` |
| per-G operations | 2 add3 + 2 add2 (= **6 additions**), **4 XOR**, **2 rotations** | `:485-488` |
| rotations | only 12 and 7 cost anything: *"rotr12 = rotl20 = rotl16∘rotl4; rotr7 = rotl25 = rotl16∘rotl9"*; 16 and 8 are byte permutations, free | `ROT_SHIFT_R`, `:126-128` |

> **4,946 = 3,056 main + 3×630 aux, ✓ FULLY VERIFIED** — including the aux width, which
> `tower.py:15` and MMCS-PLAN §5 both carried as `? INFERRED`.

**What a floor would have to beat.** 2,880 of the 3,056 main columns are the 48 G-blocks;
the other 176 are I/O. Within a G-block, 56 bytes = 14 byte-decomposed 32-bit words (4 add
results + 4 XOR results + rotation split parts) plus 4 carry bits. A tighter encoding might
carry 10-12 words instead of 14 → `G_SIZE` ≈ 44-52 → main ≈ 2,300-2,700, with the lookup
count falling proportionally. **Realistic floor ≈ 4,200-4,700 cells/compression — a 5-15%
improvement, not a factor.**

| candidate | main | aux (ext) | cells/compression |
|---|---|---|---|
| hosted socket today | 3,056 | 630 | **4,946** ✓ VERIFIED |
| #903 standalone chip | 3,219 | 699 | **5,316** (✓ from commit `35038501`'s message) |
| plausible floor, same byte-decomposed family | ~2,500 | ~520 | **≈4,060** DERIVED |

**The chip is not where the win is.** 38% of a compression is LogUp aux paying for 1,248
BITWISE byte-lookups; going materially below means trading those lookups for algebraic
constraints, which costs main columns roughly 1:1. **A specialized blake3 circuit is worth
~1.1×, not 10× — and the brief's premise that it is the lever should be retired.**

### 1.4 The target table, and the ladder

**Cost model.** The flat-ρ framing in the brief is not adequate, because the residue does
not scale with compressions — it scales with **felts absorbed**, which is a property of the
proof being verified and is invariant under every hash-chip lever. Two-term model:

```
cells_per_wrap = compressions × c_chip  +  felts_absorbed × 439  +  26.5 M (fixed-height tables)
```

✓ MEASURED calibration, `census_logs/ethrex_e21_b2_q4.log` chip census (2^21/blowup2/q=4):
total 1,965,702,420 cells; hash chips (KECCAK_RND + LFM_KECCAK) 1,818,755,072 = **92.5%**;
fixed-height tables (BITWISE 2^20 + LFM_RANGE + KECCAK_RC) 26,476,672; q-scaling arithmetic
120,470,676. Against 4 × 50,870 = 203,480 felts absorbed → 592 cells/felt, corrected to
**439** for the spine's 25.9% instruction share (which does not scale with q).
**ρ = 1.081 at this point** — ✓ MEASURED, and the single most load-bearing input here.

**★ The ladder** (each rung cumulative; GPUs = cells ÷ (67.13 M × 12 s)):

| step | comps/block | B cells | **GPUs** | cum | hash share |
|---|---|---|---|---|---|
| **A** today: 2^21 b2/219q, rate 4, no batching, arity 2, hosted socket | 138,942,214 | 1,020.6 | **1,267** | 1.0× | 67% |
| **B** + batched FRI + MMCS | 87,934,340 | 599.7 | **744** | 1.7× | 73% |
| **C** + leaf RATE 4→8 | 47,242,680 | 398.4 | **495** | 2.6× | 59% |
| **D** + tower arity 2→4 | 37,557,180 | 316.0 | **392** | 3.2× | 59% |
| **E** + epoch 2^21→2^23 (N 36→9) | 10,242,738 | 86.0 | **107** | 11.9× | 59% |
| **F** + inner blowup4/110q | 7,476,480 | 63.2 | **78** | 16.2× | 59% |
| **G** + packed AIR at 4,400 cells/compression (12% headroom, §1.3) | 7,476,480 | 59.1 | **73** | 17.4× | 56% |
| **I** + G-per-row shape on the tower's LFM_HASH (§3.2) | 5,926,800 | 46.8 | **58** | 21.8× | 56% |
| **J** + G-per-row shape on the inner blake3 chip too (§3.3) | 5,174,400 | 41.0 | **51** | 24.8× | 55% |
| **H** + a *free* hash chip — the residue-only floor | 5,174,400 | 18.3 | **23** | 55.1× | 0% |

Rungs A-F are configuration and scheduling; **G-J are the hash-circuit work this document
was commissioned about, and together they are worth 1.54×** (78 → 51) — **of which the chip
itself is 1.07× and the two shape changes are 1.44×.** The shape rungs reduce *both* terms
— fewer compressions to host *and* fewer felts absorbed — which is why they dominate the
per-compression rung that costs far more to build.

**Required cells/compression at config J** (5,174,400 compressions/block):

| GPUs | budget | required, **pessimistic residue** (18.3 B) | **optimistic residue** (1.83 B) |
|---|---|---|---|
| 4 | 3.22 B | impossible | 269 |
| 8 | 6.44 B | impossible | 891 |
| 16 | 12.89 B | impossible | 2,138 |
| 32 | 25.78 B | **1,446** | **4,629** |
| 64 | 51.56 B | 6,428 | 9,225 |

Against the §1.3 floor (**4,946 today, ≈4,060 plausible best**): **32 GPUs is reachable if
the residue is small — the required 4,629 is 6% under today's chip and comfortably inside
the plausible floor. 16 needs 2,138, which is 2× below anything this chip family can
reach. 8 needs 891 and 4 needs 269 — 4.6× and 15× below the floor. Those are not in this
design space, and no amount of blake3-circuit work puts them there.**

The brief's requested {batching} × {RATE} × {GPU count} grid is in §7's script output; it is
not reproduced in full here because **every cell at ≤32 GPUs reads "impossible" once the
residue is priced**, and the grid computed against hash cells alone (which is what a flat-ρ
model does) is misleading in exactly the direction that would authorise the wrong build.

**★ Robustness.** 439 cells/felt is the pessimistic end — it is calibrated on the *keccak*
emitter's byte-level sponge packing, and the blake3-native emitter over one-cell digests
should be much leaner (§5.1). The "optimistic residue" column above is that coefficient
divided by 10. **The verdict at 4-8 GPUs survives the full 10× swing**; what the swing
changes is *which lever matters next* (residue vs chip), and whether 32 needs a new chip at
all. That is exactly why §5.1 is the first thing to run.

---

## 2. THE SPLIT DESIGN SPACE

### 2.1 Four routes, and what each actually is

**(d) — DISTRIBUTED PROVING of a table that is already separate.** *Named first because it
is the cheapest and is not what the brief assumed.* Our proofs are already multi-table: an
epoch proof has 28-64 sub-proofs, an LFM proof 14, each with its own commitment, all bound
by a **shared LogUp challenge sampled after every main root is in the transcript** —
✓ VERIFIED **`crypto/stark/src/prover.rs:2447-2448`**, "Round 1, Phase A: Commit all main
traces … All main trace commitments must be in the transcript before sampling LogUp
challenges." (MMCS-PLAN §1.0 cites this as `prover.rs:3213-3238`; line numbers have drifted
since, the constraint now sits at 2447.) So the hash chip is *already* a separate table on a
shared, already-sound bus. "Splitting hash out" can mean nothing more than **proving that
table on a different GPU**.
- *Soundness:* **unchanged — it is the same proof.** No new protocol surface at all.
- *What is new:* a distributed prover with a small number of synchronisation barriers
  (commit mains → gather roots → advance transcript → broadcast challenge → commit aux →
  …). The seams are already named: residency-seam-audit.md S1-S7 (`multi_prove` takes a
  per-index producer; `LfmTraces` goes lazy).
- *What it buys:* **latency only.** Total work is identical, so the fleet number in §1.4
  does not move. It is the only route that attacks §3's 14-45× latency miss.
- *Effort:* **M** (orchestration + the S1-S7 refactor, which S3 has already started).

**(a) — SP1-style deferred/precompile shards with a global EC-digest accumulator.**
✓ VERIFIED in `others/sp1`:
- Each global interaction's 8-word payload is **hashed to a point on a septic-extension
  curve** — `SepticCurve::<F>::lift_x(new_values)` — with the interaction `kind` folded into
  the top byte of word 0 (`crates/core/machine/src/operations/global_interaction.rs:33-45`).
  A **send is the point; a receive is its negation** (`:41-44`).
- The per-shard accumulation is the **elliptic-curve sum** of those points
  (`operations/global_accumulation.rs`, `global/mod.rs:208` `global_cumulative_sum`).
- The reconciliation is **not in the per-shard verifier — it is in the recursion layer**:
  `crates/recursion/circuit/src/machine/complete.rs:147`
  `builder.assert_digest_zero_v2(is_complete, *global_cumulative_sum)`, seeded from the vk's
  `initial_global_cumulative_sum` (`machine/core.rs:136-140`) and observed into the
  challenger (`:149-150`).

> **★ The architectural answer to the problem route (b) runs into: an EC digest needs NO
> shared challenge.** Soundness rests on the hardness of finding a non-trivial zero-sum
> combination of hash-to-curve points, not on a Fiat-Shamir challenge sampled after
> commitments. *That* is why it composes across independently-proved shards, and it is
> exactly what our LogUp bus cannot do.

- ⚠ *And the cost is structurally wrong for our use case.* Each global interaction carries
  `x_coordinate: SepticBlock` (7) + `y_coordinate` (7) + **a full `Poseidon2Operation`
  permutation** + `offset` + `y6_byte_decomp[4]`
  (`global_interaction.rs:24-30`). **The glue spends a hash permutation per interaction.**
  At one interaction per blake3 compression, offloading hashing would cost a hash per hash —
  the mechanism is priced for offloading *expensive* precompiles (a keccak-f, a 256-bit EC
  op) from a cheap RISC-V shard, not for offloading the compression function that the glue
  itself is built from.
- *Effort:* **L**, and the cost analysis above argues it is the wrong tool. Adversarial
  debate mandatory if it is ever scheduled.
- ⚠ A fuller survey (per-shard row limits / `SplitOpts`, the deferred-proof digest path,
  exact `Poseidon2Cols` width) was commissioned and had not returned; the figures above are
  the load-bearing ones and are ✓ VERIFIED, the rest of route (a) remains ? INFERRED.

**(b) — CROSS-PROOF LogUp with a joint transcript.** The wrap emits a fingerprint send per
compression; the hash shard emits the matching receive; both partial sums are public and
must cancel.
- *The obstacle is the challenge, not the bus.* The shard's aux trace needs the LogUp
  challenge γ, and γ must be bound to the wrap's commitments too, or a malicious shard
  prover chooses its list after seeing γ. That forces **both provers to interleave**:
  commit mains → joint γ → commit auxes. This is a single Fiat-Shamir transcript spanning
  two proofs — i.e. **route (d) with the two halves relabelled as separate proofs**, and it
  keeps (d)'s synchronisation barrier while adding a new wire format.
- *The escape route is (a)'s trick, not more transcript engineering.* A **challenge-free**
  accumulator — SP1's hash-to-curve digest — removes the interleaving requirement entirely,
  which is precisely why SP1 chose it. But it re-imports a hash permutation per interaction
  (§2.1a), so for *hash* offload it is self-defeating. **There is no known cheap,
  challenge-free binding for a bus whose payload rate equals the hash rate.** That is the
  real obstacle to routes (a)-(c), and it is a genuine open problem, not an engineering gap.
- *Effort:* **M-L**, and it is strictly worse than (d) unless the shard genuinely needs to
  be a standalone verifiable object.

**(c) — RECURSION-WITHIN-RECURSION.** The specialized blake3 STARK proves a batch of
compressions and publishes a commitment to its (input, output) list; the WRAP verifies that
proof instead of hosting the rows.
- *The binding still needs (b).* The wrap must check that the compressions it consumed are
  the ones in the shard's list. Merkle-opening each one costs more than hashing it; the only
  cheap check is a fingerprint under a shared challenge — which is (b). **So (c) = (b) + an
  imported verify cost.** Its only advantage is that the shard proof is a standalone object
  (schedulable, cacheable, re-usable across blocks for repeated inputs).
- *Effort:* **L.**

### 2.2 When (c) wins over hosting — the break-even, honestly

Hosting H compressions costs `H · c_host · ρ`. Splitting costs `H · c_ded · ρ_shard +
V · c_host · ρ`, where V is the compressions the wrap spends *verifying* the shard proof.
At `ρ_shard = 1` and `c_ded = c_host`:

> **break-even ρ = 1 / (1 − V/H)**

V is a native (LFM-hash) leg walk over the shard's single table at 110 q, batched. DERIVED:

| shard holds | AIR layout | main | rows | V (comps) | V/H | break-even ρ |
|---|---|---|---|---|---|---|
| 2^18 compressions | 1 compression/row | 3,056 | 2^18 | 157,080 | 0.599 | **2.50** |
| 2^18 | 1 round/row (6r) | 509 | 2^20 | 48,730 | 0.186 | **1.23** |
| 2^18 | 1 G-call/row (48) | 63 | 2^23 | 37,070 | 0.141 | **1.17** |
| 2^20 compressions | 1 compression/row | 3,056 | 2^20 | 162,030 | 0.155 | **1.18** |
| 2^20 | 1 G-call/row (48) | 63 | 2^25 | 43,120 | 0.041 | **1.04** |

(rate 8 throughout; the rate-4 rows are in §7's output and are ~1.2× worse.)

**Read it against ρ ≈ 1.7-1.9 (blake3, §1.4) and against the memory ceiling.** A 2^20-
compression shard at 1 compression/row is 5.19 B cells = **163 GiB host** — unprovable. The
shard sizes that fit 64 GiB are ~2^18, and *at 2^18 the wide layout loses* (needs ρ > 2.50,
we have ~1.8). **So the split is only viable in the narrow-and-tall layout.** That is not a
tuning preference; it is the condition of the design.

### 2.3 What the split actually buys, bounded

The split's work win is `ρ / ρ_split × c_host / c_ded`, and both factors are smaller than
they look:
- **ρ ≈ 1.7-1.9** (DERIVED) — so removing *all* residue is ≤1.9×. But the split does not
  remove all of it: the guest still computes the leaf bytes it is absorbing and still has to
  present them to a bus. Only the socket's memory plumbing (LFM_LANES / LFM_HINT / address
  arithmetic) goes away. **ρ_split ∈ (1.0, 1.9), unmeasured, plausibly 1.2-1.4.**
- **c_host/c_ded ≤ 1.22×** (§1.3, now ✓ VERIFIED rather than derived).

> **Split ceiling ≈ 1.2-1.6× on total work**, and it is the *last* rung of a 25× ladder.
> Rungs G-H in §1.4 bracket the whole hash-circuit family: 78 → 51 GPUs with every lever
> landed, 78 → 23 with a chip that costs nothing at all.

⚠ And note what the split does **not** do: it does not reduce the compression *count*, only
the cost of each one, so it cannot substitute for any rung A-F. It also re-imports a verify
cost (§2.2) and, unlike route (d), it buys no latency — the shards are parallel, but so are
the tables in route (d), for free.

### 2.4 The shape lever, restated as the actual recommendation

Everything above points at one cheap change that is **not** a split:

> **Lay the blake3 AIR out narrow-and-tall.** Cells are conserved (`rows × width`), so
> proving cost is unchanged. Verification cost falls ~1.5× at the tower and makes the split
> break even where it currently does not.

Costs to weigh (✗ UNVERIFIED, needs the chip read): a G-per-row layout must carry the
16-word state and the message schedule in every row, so the width floor is ~100 columns,
not 3,056/48 = 64; and it adds row-transition constraints plus a round/G selector. If the
realistic width is 100-120 rather than 64, §3.2's tower win drops from 1.54× to ~1.45×
— still the largest single lever on the tower.

---

## 3. PARALLELISM STRUCTURE

### 3.1 Cadence ≠ latency

- **Cadence** (one block proof per 12 s): fleet = total work ÷ per-GPU throughput.
  Pipelining across blocks satisfies it; no intra-proof parallelism required. **This is
  what §1 answers, and it is what the "4-8 GPUs" target means.**
- **Latency** (a block proof within one slot): needs intra-proof parallelism *and* a shallow
  tower. Config F, hosted socket:

| stage | cells | 1-GPU latency | host RSS |
|---|---|---|---|
| one base wrap (2^23, blowup4/110q, rate 8, batched) | 1.86-4.37 B | **28-65 s** | 59-137 GiB |
| one 4-ary tower node | 4.65-10.88 B | **69-162 s** | 146-342 GiB |
| tower depth (2^23, arity 4) | — | 2 layers | — |
| **critical path, perfect fan-out** | — | **166-389 s = 14-32 slots** | — |

**The 12 s cadence binds on total work; the 12 s *latency* binds on per-wrap and per-node
latency, not on tower depth.** Depth contributes 2 of the ~5 stage-times at arity 4. Even
an infinitely wide fleet cannot produce a block proof in 12 s without splitting a *single*
wrap across GPUs — which is exactly route (d).

**Shard latency under intra-wrap distribution** (config F, floor chip):

| shards | shard latency | shard host RSS |
|---|---|---|
| 1 | 27.8 s | 59 GiB |
| 4 | 6.9 s | 15 GiB |
| 16 | 1.7 s | 4 GiB |
| 64 | 0.4 s | 1 GiB |

Route (d) at 4-16 shards puts a wrap inside a slot and each shard inside a 5090's VRAM.
**This is the parallelism structure the fleet endgame needs, and it is the route with no
soundness surface.**

### 3.2 The arity trade, with the D1 node model

N = 9 (2^23 epochs), two-term model, batched. A node verifying k proofs costs k × one
proof-verify; **one** proof-verify is 733,700 comps / 157 GiB as D0+D9 are specified today,
258,280 comps / 69 GiB with RATE 8 + G-per-row.

| arity | nodes | depth | verifications | node host RSS (spec'd / recommended) | tower comps (recommended) |
|---|---|---|---|---|---|
| 2 | 11 | 4 | 19 | 315 / **138 GiB** | 4,907,320 |
| **4** | **4** | **2** | **12** | 629 / **277 GiB** | **3,099,360** |
| 8 | 3 | 2 | 11 | 1,258 / 554 GiB | 2,841,080 |
| 16 | 1 | 1 | 9 | 2,517 / 1,108 GiB | 2,324,520 |

**Arity 4 is the knee on work: 2→4 buys 1.58× and halves depth; 4→8 buys 8% more and
*doubles* the node.**

⚠ **But read the node column: at every arity the aggregating node is far over 64 GiB, and
the shape lever does not fix that.** Gate D1's ~81 GiB (PLAN.md:176) and this document's
69 GiB are both **one-proof-verify** figures; the smallest node that actually *aggregates*
is arity 2 at **138 GiB**, 2.2× the target. **The tower node — not the base wrap — is the
campaign's binding memory constraint, and the only lever that touches it is residency
(S3/S6), not any hash lever.** That is a finding for the tower track, and it argues for
running S3's seams on tower nodes from the start rather than treating them as a base-layer
concern.

**Row shape at the D1 node** — width moved, **rows scaled to compensate so cells are
conserved**, priced under §1.4's two-term model (batched, 110 q):

| LFM_HASH layout | main | aux | rows | node @ RATE 4 | node @ RATE 8 |
|---|---|---|---|---|---|
| 1 compression / row (**D0 as specified**) | 2,964 | 630 | 4 | **157 GiB** | **104 GiB** |
| 1 round / row (6r) | 494 | 105 | 32 | 110 GiB (1.43×) | 74 GiB (1.40×) |
| **1 G-call / row (48)** | 100 | 20 | 256 | **103 GiB (1.53×)** | **69 GiB (1.50×)** |
| (keccak-era LFM_HASH, the unreachable bound) | 28 | 3 | 4 | 102 GiB (1.55×) | 68 GiB (1.51×) |

The row-shape change recovers **96% of the gap back to the keccak-era node**, and it holds
at 1.50× even after scaling rows (a G-per-row LFM_HASH is 256 rows, so Merkle depth grows
from 2 to 8 — six extra parent compressions per group per query, against ~1,200 leaf
compressions saved).

⚠ **This does not reproduce MMCS-PLAN §1.4's 122 GiB at RATE 4 / D0 width — it gives 157
GiB, 29% worse.** The difference is the residue model, and it is the same disagreement as
§1.4: `tower.py` and MMCS-PLAN price non-hash cells at a flat **6.5%** of the node, while the
measured chip census says the residue tracks *felts absorbed* and therefore does **not**
shrink when the hash term does. At the D0 width the two agree to within a few percent; at
RATE 8 with a narrow chip the flat model says 43 GiB and the two-term model says 69 GiB.
**§5.1's census resolves which is right, and the answer moves Gate D1 by 1.6×.** Until then
the tower numbers circulating in the campaign should be read as the optimistic end.

RATE 8 + G-per-row lands at **69 GiB — within 8% of the 64 GiB production target**, against
**157 GiB** as D0 and D9 are specified today. Nothing else in the campaign gets a tower node
that close.

### 3.3 Where the inner chip's shape matters (less)

The same lever applied to the *inner* proof's blake3 chip is weaker, because a 2^23 epoch
has 64 sub-proofs and the hash chip is one of them:

| inner chip layout | felts/query | comps/query | block GPUs |
|---|---|---|---|
| keccak KECCAK_RND (1,480/516) — what the leg model prices today | 21,614 | 2,856 | 78 |
| blake3, 1 compression/row (3,056/630) | 25,450 | 3,365 | **84** |
| blake3, 1 round/row | 17,230 | 2,273 | 73 |
| blake3, 1 G-call/row | 15,878 | 2,092 | **71** |

⚠ **A correction the campaign should absorb: P-a makes the wrap's leaf absorption *worse*,
by 8%.** The leg sets in `mmcs_project.py` carry the keccak-era `(1480, 516)` inner hash
chip; after P-a that leg is the blake3 chip at `(3056, 630)`, and leaf absorption is
proportional to width. Every post-P-a number in MMCS-PLAN and in §1 of this document is
optimistic by ~8% for this reason. It does not change any verdict; it should be fixed in the
leg data before the next projection round.

---

## 4. SEQUENCING, AND WHAT THIS OBSOLETES

### 4.1 Against the live tracks

| track | verdict under the split analysis |
|---|---|
| **P-a** (inner → blake3-6r) | **Unaffected, still first.** It is the ÷4 on the hash term at every layer. But it *widens* the inner hash chip 1,480→3,056, costing 8% back on the wrap's leaf absorption (§3.3) — worth knowing, not worth re-sequencing. |
| **D0 step 3-4** (LFM proof commits blake3) | **Unaffected, still required.** The tower legs recompute the LFM proof's own trees. |
| **Batching** (FRI + MMCS) | **Confirmed, and it is a prerequisite for everything here.** Every number in §1.2-1.4 assumes it. It is rung B (1.7×) and it is what makes rung E (epoch size, 3.7×) *possible* — MMCS-PLAN §1.2's "epoch size becomes nearly free". **Not obsoleted; promoted.** |
| **S3 / S6** (residency) | **Unaffected as a fit lever, and route (d) subsumes its seams.** S1-S7 are the same seams a distributed prover needs. Building (d) on top of S3 is nearly free; building (d) without S3 is not possible. |
| **D9 / RATE** | **★ Repriced, and the question changes.** RATE 4→8 is rung C, worth 1.5× — larger than the split. But **row-shape is the cheaper half of the same lever** (1.5× at the node, no bus-arity change, none of the RATE=5 lane-map hazards). D9 should be re-framed as "RATE *and* row shape", and row shape should go first. |
| **Tower arity** | New: **take 4**, not 2 (rung D, 1.3×, halves depth, §3.2). |
| **Epoch size** | New and largest: **take 2^23, not 2^21** (rung E, 3.7×). Pure scheduling. Census 2^24/2^25 before assuming it continues. |

### 4.2 Does a specialized hash circuit reduce pressure on D9 and batching, or multiply it?

**It multiplies both, and neither substitutes for it.**
- **On batching:** batching moves the wrap into the leaf-absorption-dominated regime (74-77%
  of the bill after batching, MMCS-PLAN §1.1). Leaf absorption is `Σ cols / RATE` —
  precisely what RATE and row-shape attack. So after batching, hash-chip levers are worth
  *more*, not less.
- **On D9:** identical logic at the tower (94% leaf after batching). MMCS-PLAN §1.4 already
  says "batching magnifies D9 rather than substituting for it"; the split does the same.
- **But the pressure that matters most has moved off all three.** At rung F the residue is
  41% of the block and rising as the hash levers land (rung H: a free chip still needs 33
  GPUs). **The next campaign question after this one is the emitter's cells-per-absorbed-felt,
  not the hash chip.**

### 4.3 Build order

| # | item | effort | worth | gate |
|---|---|---|---|---|
| 0 | **Measure ρ under the blake3 emitter** (§5.1) | **S**, zero proving | — | decides the order of everything below |
| 1 | Census 2^24 / 2^25 (§5.2) | **S**, zero proving | — | finds where the epoch lever turns over |
| 2 | Take epoch 2^23 + tower arity 4 + inner blowup4/110q | **S** (config) | **7.4×** | census confirms |
| 3 | Batched FRI + MMCS (already scoped, MMCS-PLAN §2) | M | 1.7× | ≥2× $/wrap at LARGE |
| 4 | RATE 4→8 (D9) | M | 1.5× | the existing D9 gate |
| 5 | **Row-shape the blake3 AIR narrow-and-tall** (tower, then inner) | **M** | **1.44×** | D1 one-proof node ≤ 70 GiB at RATE 8 |
| 6 | **Route (d): distributed proving of one wrap across GPUs** | **M** | latency only | a wrap inside a slot; no soundness surface |
| 7 | Packed blake3 AIR at the floor (~3,000 cells/compression) | L | 1.3× | measured on the shape A/B harness |
| 8 | Route (a)/(b)/(c) true split | **L** | ≤1.4× | only if step 0 says ρ_split ≥ 1.5, step 5 landed, **and** the challenge-free-binding problem in §2.1(b) has an answer cheaper than a hash per interaction |

**Steps 0-6 are worth ~19× and carry no new soundness surface. Steps 7-8 are worth ~1.4×
each and step 8 carries all the soundness risk in this document.** That ordering — and in
particular putting the row-shape change (step 5, M) ahead of both the packed AIR (L) and the
split (L) — is the document's main recommendation.

---

## 5. THE MINIMAL EXPERIMENTS

### 5.1 ★ Round 0 (decisive, free, no proving): measure ρ under the blake3 emitter

The census harness already dumps the per-chip cell table
(`census_logs/ethrex_e21_b2_q4.log` shows the exact format). Run the same census point with
the blake3-native emitter and read off two numbers: **hash-chip share** and **q-scaling
arithmetic cells ÷ felts absorbed**.

- **If cells/felt ≈ 439** (the keccak-emitter value): ρ_blake3 ≈ 1.9, the residue is 41% of
  the block, and **the split is worth at most 1.9× while step 0-4 are worth 16×** — do the
  split last or not at all.
- **If cells/felt ≈ 50-100** (a felt-native emitter over one-cell digests should be far
  leaner than byte-level sponge packing): ρ_blake3 ≈ 1.1-1.2, the residue nearly vanishes,
  **and the hash chip becomes ~90% of the block again** — at which point the split and the
  packed AIR become the dominant levers and should be promoted above everything except
  batching.

**This single number flips the plan's order.** It is a census, not a prove.

### 5.2 Round 0b (free): census 2^24 and 2^25

Rung E (2^21→2^23) is the largest in the ladder and the model says the direction continues.
But sub-proof count grows ~1.5× per doubling (28/32/43/64 measured at 2^20…2^23) and batched
leaf absorption is proportional to *total columns across all sub-proofs*, so returns damp.
The census is closed-form and free. **Find where the epoch-size lever turns over before
building anything.**

### 5.3 Round 1 (M, after step 1): the shape A/B

Prove one wrap with the blake3 AIR at 1 compression/row and at 1 G-call/row. Predictions to
falsify: **proving cells within 3%** (cells are conserved), **D1 node census 1.5× cheaper**,
**verify time unchanged**. If proving cells move more than 10%, the row-transition
constraints cost more than this model allows and §2.4/§3.2 must be re-derived.

---

## 6. OPEN DECISIONS — need Mauro

- **D10 — cadence or latency?** "4-8 GPUs keeping up with 12 s blocks" is a throughput
  target (§3.1) and is satisfied by pipelining. If a block proof is also required *within* a
  slot, route (d) becomes mandatory and moves to the front of §4.3. **These are different
  builds; the plan cannot pick.**
- **D11 — is 4-8 GPUs a requirement or an aspiration?** The honest projection is **78 GPUs**
  at the best schedulable configuration, **51 with every hash-circuit lever in this document
  landed**, and **23 even with a free hash chip**. Reaching 8 needs ~5× more that no
  identified lever supplies. If it is a requirement, the answer is not in
  this design space — it is in reducing *felts absorbed* (fewer/narrower inner tables,
  higher blowup / fewer queries, or **single-row leaves**: leaf absorption is
  `ROWS_PER_LEAF · cols · kind` and ✓ VERIFIED `crypto/stark/src/commitment.rs:42`
  `pub const ROWS_PER_LEAF: usize = 2` — dropping to 1 halves the dominant term at the cost
  of one Merkle level, and is a wire-format change nobody has priced). That is a different
  document.
- **D12 — D9 re-framing.** Row shape is a second, cheaper half of the RATE lever with none of
  the RATE=5 lane-map hazards (§4.1). Should D9 be re-opened as "RATE and row shape", with
  row shape scheduled first? **This changes what task #35 builds.**

---

## 7. REPRODUCTION

Scripts checkpointed at **`~/workspace/lambda_vm_bench_cache/hash_split_2026-08-13/`**
(out of tree, per the lean-PR rule). Run them from that directory with the calibrated model
directory on `sys.path` (each script inserts it):
`hashsplit.py` (anchors, compression counts, target grid, fleet inverse),
`hashsplit2.py` (blowup-4 derivation, break-even sweep, latency), `residue.py`
(ρ calibration from the measured chip census), `final2.py` (ladder A-H),
`ladder2.py` (ladder F-J, the shape rungs), `shape2.py` (tower node vs row shape, rows
scaled), `arity.py` (the arity table), `innerwidth.py`. All import the calibrated model at
`~/workspace/lambda_vm_bench_cache/lfm_census_2026-08-12/` (`mmcs_project.py`, `project.py`,
`tower.py`) and reproduce MMCS-PLAN §1.1's blake3 column exactly at RATE 8, which is the
regression check that the RATE parameterisation did not perturb the calibration.

## 8. Confidence ledger

| claim | mark |
|---|---|
| 67.13 M cells/s; 481,327,124 cells; 7.17 s | ✓ MEASURED (`EXPLORATION.md:186`, `BOX-RESULTS.md:53,57`) |
| ρ = 1.081 under keccak; hash chips 92.5% of cells | ✓ MEASURED (`census_logs/ethrex_e21_b2_q4.log`) |
| 439 residue cells per absorbed felt | DERIVED from that measurement (spine-corrected) |
| ρ ≈ 1.7-1.9 under blake3 | DERIVED — **the number §5.1 exists to replace** |
| compression counts per block | DERIVED from the calibrated model (unit-exact at 4 measured points) |
| hosted socket 4,946 cells/compression | ✓ **VERIFIED both terms** — `blake3_chip.rs:162,224` (3,056 main) and `:911-913` (1,259 interactions → 630 ext aux). Upgrades `tower.py`'s own `? INFERRED` caveat. |
| plausible floor ≈ 4,060 | DERIVED from the verified G-block encoding |
| break-even ρ table; row-shape 1.5× at the node | DERIVED from the calibrated model |
| tower is 76-149% of the base layer | DERIVED; rests on D1's static-shape premise ✗ UNVERIFIED |
| ladder rungs A-J | DERIVED |
| SP1 binds shards with a **challenge-free septic-curve digest**, reconciled in the recursion layer, at the cost of a Poseidon2 permutation per interaction | ✓ VERIFIED (`global_interaction.rs:24-45`, `complete.rs:147`, `machine/core.rs:136-150`) |
| SP1 `SplitOpts` / deferred-proof-digest detail | ? INFERRED — survey commissioned, not returned |
| blake3 chip: one row/compression, 48 G-blocks × 60 cells, 1,248 BITWISE lookups | ✓ VERIFIED `blake3_chip.rs:101,158,781,990-991` |
| #903 chip 5,316 | ✓ VERIFIED from commit `35038501`'s message; ✗ UNVERIFIED against source |
