# Accelerator archaeology — chip inventories for recursion-verifier accelerator prioritization

Distilled from agent archaeology, 2026-07-20. Sources: `others/sp1` (worktree @ v1.1.0, tag `923d77d7d`,
pre-jagged) vs modern main; `others/openvm` (worktree @ v1.0.0). Feeds the prioritization matrix in
memory `accelerator-prioritization-plan`.

## SP1 v1.1.0 recursion VM (BabyBear native-field VM, word = field element)

Three machines: **compress** (DEGREE=3, the workhorse), **shrink** (9), **wrap** (17) — shrink/wrap pack
FriFold+Poseidon2 into one **Multi** chip (318 cols). Constants: D=4 ext, WIDTH=16, RATE=8, DIGEST=8.
Query policy: inner `log_blowup=1, 100 queries, PoW 16`; compressed `log_blowup=3, 33 q, PoW 16`.

### Per-chip inventory (struct-verified)

| Chip | Cols | Rows / invocation | Max deg | Shape |
|---|---|---|---|---|
| Program | 36 preproc + 1 mult | 1/program-instr (ROM, mult = exec count) | ~1 | bookkeeping |
| Cpu | 76 | 1 per executed instruction | ~3 | mixed (ALU + dispatch) |
| MemoryGlobal | 14 | 1 init + 1 finalize per touched address | ~2-3 | bookkeeping |
| **Poseidon2Wide** | **449** (deg3, w/ sbox) / **300** (deg9/17) | **compress = 2 rows**; absorb ⌈len/8⌉; finalize 1 | =DEGREE | **algebra** |
| **FriFold** | **79** | **m+1 rows** per instr (1 row per opened poly at a query pt) | =DEGREE | **protocol** |
| ExpReverseBitsLen | 23 | `len` rows (1 per exponent bit) | =DEGREE | protocol |
| RangeCheck | 2 preproc + 2 mult | fixed 2^16-row lookup | 1-2 | algebra |
| Multi (shrink/wrap) | 318 | packs FriFold+Poseidon2 rows | =DEGREE | wrapper |

### FriFold semantics (their fused reduced-opening — our candidate #2)
Dedicated instruction, NOT composed from ALU. Per row, two ext-field constraints:
`alpha_pow' = alpha_pow · α` and `(ro' − ro)·(x − z) = (p_at_x − p_at_z)·alpha_pow`
⇔ `ro += αⁱ·(p(x)−p(z))/(x−z)`. ABI: one `input_ptr` → struct {z, α (ext), x, log_height,
mat_opening_ptr, ps_at_z_ptr, alpha_pow_ptr, ro_ptr}; one instruction loops m+1 columns of one opened
row (1 chip row each), accumulators indexed by log_height in memory. Cross-round α-combination is done
with plain CPU ext-ALU ops, not FriFold. One instr per (query × round × matrix × point).
Commit-phase folding (`folded = e0 + (β−x0)(e1−e0)/(x1−x0)`) is plain EADD/ESUB/EMUL/EDIV — no chip.

### Poseidon2 modes & transcript
One Poseidon2Wide chip, three modes: 2-to-1 **Compress** (Merkle levels; 2 rows), **Absorb/Finalize**
(rate-8 sponge, persistent width-16 state). Transcript is a duplex sponge over the same permutation
(challenger `observe`/`sample` → duplexing = 1 permutation per 8 felts absorbed / per squeeze;
`sample_ext` = 4 samples). NOTE minor internal discrepancy between the two agent reads: the compiler
trace shows `DuplexChallengerVariable.duplexing` lowering to the Compress opcode while runtime docs
put transcript on Absorb/Finalize — either way it's the same chip and ~1 permutation per duplexing;
cost conclusion unaffected. Batch-opening leaf hashes use Absorb/Finalize (`reduce_fast`).

### Merkle path cost (height h)
h compressions = **2h Poseidon2Wide rows + ~h Cpu rows** (+ leaf hash absorb ⌈w/8⌉). Merkle verify is
essentially pure Poseidon2 work.

### Frequency shape per verified proof (estimate; Q=queries, L=log_max_height, 4 PCS rounds)
- Poseidon2Compress: transcript (#felts/8 + #samples) + batch Merkle ≈ Q·4·L + commit-phase Merkle ≈ Q·L²/2
- FriFold: instrs ≈ Q · Σ_mats(points); chip rows ≈ Q · total opened columns (×2 for trace mats at ζ, ζ·g)
- ExpReverseBitsLen: ≈ Q·(#matrices+1) instrs, `len`≈L rows each — turns bit-reversed query index into
  the two-adic generator power (the x in the DEEP denominator and fold points)
- Constraint evaluation: pure ext-ALU, Q-independent.
Row share: **Poseidon2 #1, FriFold #2**, everything else minor. Wide traces push FriFold up; tall/narrow
push Poseidon2 up.

### Survival analysis: v1 → modern main (jagged/Hypercube) — migration-robustness evidence
Modern `RecursionAir` = 11 variants: MemoryConst, MemoryVar, BaseAlu, ExtAlu, Poseidon2Wide,
Poseidon2LinearLayer, Poseidon2SBox, ExtFeltConvert, Select, PrefixSumChecks, PublicValues.

| v1 chip | Fate |
|---|---|
| Poseidon2Wide | **KEPT** (compress machine; wrap uses decomposed LinearLayer+SBox helpers) |
| FriFold | **DELETED** (FRI→jagged) |
| ExpReverseBitsLen | **DELETED** |
| Cpu | **DELETED** — no fetch-decode row; chip-per-opcode + addressed memory, program in preprocessed cols |
| Program | **DELETED** (folded into per-chip preprocessed cols) |
| RangeCheck, Multi | **DELETED** |
| MemoryGlobal | REPLACED by MemoryConst + MemoryVar |

New chips: BaseAlu, ExtAlu, Select, ExtFeltConvert (algebra); PrefixSumChecks (the NEW protocol chip —
Lagrange-eval/bit2felt for jagged/basefold). Pattern: **the algebra layer survives verbatim; each PCS
generation swaps in its own protocol chip** (FriFold → PrefixSumChecks).

## OpenVM v1.0.0 native extension (BabyBear native-field VM, quartic ext β=11)

Worktree @ v1.0.0 (`f41640c37`), `extensions/native/{circuit,compiler,recursion}` +
`crates/circuits/poseidon2-air` + `crates/sdk`. Cols read from actual structs/const_asserts.

### Per-chip inventory

| Chip | Opcode(s) | Cols | Rows / invocation | Shape |
|---|---|---|---|---|
| **FriReducedOpening** | FRI_REDUCED_OPENING | **27** | **ℓ+2** (1 Workload row per opened column + 2 instr rows) | algebra kernel (FRI-shaped role) |
| **NativePoseidon2** | VERIFY_BATCH, PERM_POS2, COMP_POS2 | **397** (298 p3-Poseidon2 + 99 native) | every row = 1 Poseidon2-16 perm; see path cost below | Poseidon2 core algebra; VERIFY_BATCH orchestration protocol |
| FieldExtension | FE4ADD/SUB, BBE4MUL/DIV | core 20 (+vec adapter) | 1 | algebra |
| FieldArithmetic | ADD/SUB/MUL/DIV | core 8 (+adapter=29) | 1 | algebra |
| JalRangeCheck | JAL, RANGE_CHECK | 12 | 1 | bookkeeping |
| NativeLoadStore<1>/<4> | LOADW/STOREW/HINT, block | core 5 / 8 (+adapter) | 1 | bookkeeping |
| NativeBranchEq | BEQ/BNE | reuses rv32im core | 1 | bookkeeping |
| CastF | CASTF | core 6 (+adapter) | 1 | marshaling |

### FRI_REDUCED_OPENING — the stated bottleneck
Pure Horner α-fold over the quartic ext: `acc ← acc·α + (bᵢ − aᵢ)`, i=ℓ-1…0 → `Σ αˡ⁻¹⁻ⁱ(bᵢ−aᵢ)`;
aᵢ = base-field opened cell, bᵢ = ext claimed eval. NOTE: unlike sp1's FriFold it does NOT include the
DEEP quotient /(x−z) or log-height accumulator indexing — the quotient is applied outside. ABI: 7
native-AS operands {a_ptr_ptr, b_ptr_ptr, length_ptr, alpha_ptr, result_ptr, hint_id_ptr, is_init_ptr};
is_init=0 first-touch writes a-values from the hint stream. Max deg ≈3. Fires once per
(matrix × point × query).
**Bottleneck quote CONFIRMED** — `crates/sdk/src/keygen/mod.rs:228-229`: "This computes the number of
rows in the `FRI_REDUCED_OPENING` chip, which is the expected bottleneck of the recursive verifier."
With explicit height model (:230-234): `height = num_queries · Σ_rounds total_pts·(total_width + 2·num_airs)`.

### VERIFY_BATCH — fused mixed-matrix Merkle path
ONE instruction verifies one full mixed-matrix auth path for one query: walks levels h_max…0, rolling-
hashes each level's concatenated opened rows into a leaf (InsideRow, rate 8), compresses with node +
supplied sibling (hint-fed, per index bit). One path, single matrix width w height h ≈
**⌈w/8⌉ + 1 + log₂(h) rows** at 397 cols; every row does one permutation. Fires once per
(commitment × query) — one path decommits ALL matrices sharing the tree (they enter at their heights).

### Transcript
No chip — eDSL duplex sponge (WIDTH 16, rate 8) over PERM_POS2 → 1 SimplePoseidon row per duplexing
(every 8 observes / squeeze-refill). sample_ext = 4 samples. Same as sp1: transcript = the hash chip.

### PCS batching (fire-rate structure)
Mixed-matrix commitment rounds = {per-AIR preproc} + ONE round for ALL common-main + {cached mains} +
{after-challenge} + {quotient}. So VERIFY_BATCH ∝ commitments × queries (few), FRI_REDUCED_OPENING ∝
total columns × points × queries (many) → reduced-opening dominates. Row-share estimate: (1)
FRI_REDUCED_OPENING, (2) NativePoseidon2, (3) everything else minor.

### Migration robustness (modern main, v2.x)
`extensions/native/` NO LONGER EXISTS. Recursion = dedicated `crates/recursion/` verifying via **WHIR**
(whir/folding.cu, batch_constraint/gkr/stacking AIRs). FRI_REDUCED_OPENING, VERIFY_BATCH/MMCS, the
native DSL VM, and the duplex challenger are ALL retired; the Poseidon2 permutation and
BabyBear+quartic algebra survive as primitives. Same lesson as sp1, stronger: protocol-shaped dies
(even the fused-Merkle chip), algebra survives.

## Complexity gauge (LOC of cols+AIR+trace-gen, measured 2026-07-20; calibrated vs our shipped chips)

Reference chips: openvm FriReducedOpening **897** (one file, 27 cols, ~3 row kinds);
sp1 FriFold **404** (79 cols); sp1 ExpReverseBits 552; openvm NativePoseidon2+VERIFY_BATCH **1,926**
(+ shared p3 poseidon2-air subair, not counted); sp1 Poseidon2Wide **3,079** (absorb state machine
IN-chip is the bloat); sp1 Multi 403; openvm ext-field ALU ~330, base ALU 224.
Ours (prover/src/tables): lt.rs 449, mul.rs 843, ecsm.rs 901, shift.rs 1014,
keccak family 1,685 (keccak 510 + keccak_rnd 925 + keccak_rc 250).
→ **Reduced-opening chip = lt.rs/mul.rs class** (smallest real chip in both stacks; rebuild ≈ small).
**Fused Merkle path = keccak-family/ecsm class** (biggest, stateful, hint-fed — worst complexity AND
worst staying power). **Hash-interface**: perm core ALREADY SHIPPED (keccak tables); new work = limb
ABI/marshaling binding + executor ecall. KEY STRUCTURAL LESSON: openvm kept sponge STATE in guest
code (chip = stateless perm + path) → 1.9k LOC; sp1 put absorb state in-chip → 3.1k LOC. Keep sponge
state in guest; chip stays stateless → mul.rs-class delta.
Rubric for any future candidate: LOC(air+trace) · #row-types (cross-row state = the multiplier) ·
#cols · #interactions · audit surface; rebuild-cost = same measure → robustness column should be
read as coupling × rebuild-cost (small protocol-coupled chips are acceptable; LARGE ones aren't).

## Cross-system synthesis (what this means for our candidates)

1. **Both systems built exactly two heavy chips**: a fused reduced-opening α-ladder and a Poseidon2
   permutation serving Merkle+transcript+leaf via modes. Everything else is 1-row ALU/bookkeeping.
   This independently validates our candidate list (#1 hash ABI, #2 fused reduced-opening) and the
   "not worth building" list (base ALU, exp-reverse-bits — sp1 built ExpReverseBitsLen, then deleted it).
2. **Candidate #2 design guidance**: copy openvm's shape (pure Horner `acc·α + (b−a)` at the
   field-array boundary, 27 cols, ℓ+2 rows, quotient outside), NOT sp1's (embeds DEEP /(x−z) +
   log-height accumulator indexing → more protocol-coupled; died with FRI at sp1 while openvm's purer
   kernel died only because the whole native VM was retired). openvm's own keygen names it THE
   recursive-verifier bottleneck.
3. **Candidate #1 validation**: in both systems the transcript is a duplex sponge riding the SAME
   permutation chip (1 perm per 8 absorbed felts / per squeeze) — one field-native hash primitive
   serves Merkle nodes, leaf hashes, and transcript. Our hash ABI should expose exactly those modes
   (compress 2-to-1, absorb-rate-N, squeeze).
4. **Candidate #3 (fused Merkle path) confirmed lowest-robustness**: VERIFY_BATCH is the most
   protocol-coupled chip in either system (mixed-matrix path shape, hint-fed siblings) and it died in
   BOTH migrations. Its economics also depend on commitment batching (#768): it fires per-commitment ×
   query — worth little while we're per-table, more after #768. Design last.
5. **Hash CHOICE is out of scope** (user directive 2026-07-20): assume our keccak is perfect/free —
   the permutation is already a 1-cycle ecall and large guest cost remains regardless. The hash swap
   is the user's own decision (endgame: Blake). What transfers from the archaeology is the
   hash-AGNOSTIC interface structure: one primitive with compress/absorb/squeeze modes serving
   Merkle + leaf + transcript, field-native operands (no byte marshaling in-guest). The residual
   sponge-bucket cycles are plumbing around the (free) permutation — that is what candidate #1
   attacks, independent of which permutation is behind the ecall.
6. **Cost model to reuse for our matrix**: openvm's chip-height formula
   `num_queries · Σ_rounds total_pts·(total_width + 2·num_airs)` adapts directly to estimating our
   fused-reduced-opening chip rows from per-table widths (see recursion-optimization-state table-width
   census).

