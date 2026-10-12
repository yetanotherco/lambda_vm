# Poseidon2 accelerator — cost study vs BLAKE3-6r and keccak (2026-08-05)

Produced by a multi-agent study (three mining agents over the vendored
references in `others/` — Plonky3, zisk, stwo, openvm, SP1 old+new, risc0,
airbender, pil2-proofman — plus a synthesis agent applying this repo's cost
model). Model calibration: reproduces Plonky3's Goldilocks w8/SR=1 column
count exactly (180) and zisk's measured 490 cells/perm to 1.0%.

Companion measured numbers (this branch, 32-core box, blowup 2):
keccak-f 72,672 table / 73,020 end-to-end; BLAKE3-6r 5,316 table / 7,337
end-to-end per 2-to-1 merge; blake3 throughput 5,217 compressions/s at 2^17
rows vs keccak 433 perms/s at 2^20 rows.

---

Both calibrations land: my model reproduces Plonky3's Goldilocks w8/SR=1 figure **exactly** (180), and at zisk's degree budget it gives 495 against zisk's measured 490 — **1.0%**. That's a two-point validation of the whole cost model before applying it to our constraints.

---

# Poseidon2 accelerator chip — cost study (final)

**Headline: ≈ 651 cell-equiv table-only per 2-to-1 merge** (recommended in-place ABI; 753 under the brief's separate-output ABI). Against BLAKE3-6r's 5,316 that is **8.2× cheaper**; against keccak-f's 72,672, **112×**. End-to-end the advantage over BLAKE3 holds at roughly 5–9×, but the absolute win is small change next to what BLAKE3 already banked.

## Calibration first — the model reproduces two independent mined numbers

Before trusting it on our constraints, I ran the same model at other systems' degree budgets:

| target | their budget | my model | mined | agreement |
|---|---|---:|---:|---:|
| Plonky3 Goldilocks w8, `SBOX_REGISTERS=1`, lookup-free | deg 3, ungated | `8 + core(8,REG=1)` = **180** | 180 | **exact** |
| zisk Goldilocks perm, no sbox registers, incl. memory plumbing | deg 7, ungated | `187 + 86 + 24 + 198` = **495** | 490 | **1.0%** |

The zisk check is the valuable one: it exercises the core formula, the byte-level I/O apparatus *and* the LogUp aux rate simultaneously, and lands within 1%. It also isolates the one thing that makes our number bigger than everyone else's — the degree budget, nothing else.

## (b) Our number, line by line

**Width 8, truncated permutation — justified.** A digest is 4 Goldilocks elements (32 B), so a 2-to-1 merge absorbs 8 elements. Two shapes do that in *one* permutation: width 8 as a truncated permutation (`P(left‖right)[0..4]` — Plonky3's `TruncatedPermutation<P,2,4,8>`, `others/Plonky3/symmetric/src/compression.rs:17`), or width 12 as a rate-8/capacity-4 sponge (Plonky2 style). I priced both on identical I/O: **width 8 = 651, width 12 = 779**. Width 8 wins by 16% and is what Plonky3/SP1 ship for merges. Parameters are forced: `RF = 8 (4+4)`, `RP = 22`, S-box `x⁷` — `others/Plonky3/goldilocks/src/poseidon2.rs:22,32,70-73`; x³ and x⁵ are not permutations since `p−1 = 2^32·3·5·17·257·65537` (`goldilocks/src/poseidon1.rs:41-44`).

**4 committed cells per S-box — forced by μ-gating, and minimal.** Max degree 3 *including* ×μ means bodies are capped at degree 2. Chain: `a=x²`, `b=a·x=x³`, `c=b·b=x⁶`, then `post = M·(c·x)` absorbs the last multiply into the linear layer. All four constraints are degree 2 → 3 after ×μ. Four is provably minimal: from `{1}`, three degree-≤2 steps reach at most exponent 6. This is Plonky3's `SBOX_REGISTERS=3` — their width formula (`poseidon2-air/src/columns.rs:12-69`) is generic in REGISTERS, but `eval_sbox` (`air.rs:288-323`) only ships `(7,1)→deg 3`, so we are one rung past anything in the wild.

Committing the S-box *output* (Plonky3's `post_sbox`, `air.rs:274-277`) rather than the post-linear element (SP1's `s0`) keeps the whole state at expression-degree 1 through all 22 internal rounds, so the 7 non-S-boxed elements ride free and **no boundary re-commit is needed** — SP1's choice would cost +8 cells here.

```
CORE (one row per permutation, fully unrolled)
  full rounds    2 × 4 rounds × 8 elems × (3 registers + 1 post)   = 256
  partial rounds       22 rounds × 1 elem × (3 registers + 1)      =  88
                                                            core   = 344 cells
  sends in the core                                                =   0
      — field-native: no ByteAlu, no AreBytes, no lookups whatsoever
  cross-check: 8 inputs + 344 = 352 = Plonky3 num_cols<8,7,3,4,22>   ✓

CANONICITY (byte→field must be injective or the tree isn't binding:
  x and x+p are distinct byte strings with the same field element)
  per element: commit is_max, dinv; constrain
      μ·(is_max + (H−(2³²−1))·dinv − 1)      deg 3
      μ·(is_max·(H−(2³²−1)))                 deg 3
      μ·(is_max·L)                           deg 3      booleanity implied
  2 cells, 0 sends × 12 elements                                   =  24 cells

I/O APPARATUS (idiom copied from the shipped chip, prover/src/tables/blake3.rs:97-123
  columns and :747-1030 interactions; 2 bytes per AreBytes send, 4 IsHalfword per
  dword pointer, pointer-arith carries are expression-form with no cells)

              A: 12 dwords (brief)      A′: 8 dwords, in-place (SP1 ABI)
  TIMESTAMP_0/1          2                        2
  ADDR bytes             8                        8
  PTR halfwords         48                       32
  IN bytes              64                       64
  OUT bytes             32                       32
  OLD_OUT bytes         32                        0   ← old = the input bytes
  MU                     1                        1
  I/O columns          187                      139

  Ecall receive          1                        1
  Memw register read     1                        1
  Memw per dword        12                        8
  IsHalfword            48                       32
  AreBytes addr          4                        4
  ByteAlu AND (align)    1                        1
  AreBytes IN/OUT/OLD   64                       48
  sends N              131                       95

TOTAL
  A :  main 187+344+24 = 555 ; aux = 1.5×131 = 198 ;  TOTAL  753
  A′:  main 139+344+24 = 507 ; aux = 1.5× 95 = 144 ;  TOTAL  651   ← recommended
```

Arithmetic machine-checked: `/private/tmp/claude-501/-Users-maurofab-workspace-lambda-vm/931cf0e4-cfb3-4d8a-b940-5360f4374a8b/scratchpad/pos2_cost.py`.

**End-to-end plumbing — the weakest number here, and I won't pretend otherwise.** The two known marginals disagree about what a memory op costs: BLAKE3 is `7,337 − 5,316 = 2,021` over 23 chip Memw ops (**88/op**); keccak is `73,020 − 72,672 = 348` over 26 (25 lanes + register read, `prover/src/tables/keccak.rs:3-5`) — **13/op**. A 6.5× spread means "per Memw op" is the wrong model. The likely driver is guest-side marshalling: BLAKE3's ABI makes the guest lay out a fresh 176-byte region every call, while keccak operates in place on a resident 200-byte state (hypothesis, unverified). Poseidon2-A′ is in-place over a 96-byte region with 9 ops, i.e. structurally keccak-shaped, so I expect the low end — but I quote the full band:

```
A′ end-to-end = 651 + 9 ops × [13 … 88] = [768 … 1,443]   central estimate ≈ 900
A  end-to-end = 753 + 13 ops × [13 … 88] = [922 … 1,897]
```

## (c) Comparison, per 2-to-1 merge (64 B in, 32 B out)

| | table-only | end-to-end | vs keccak (e2e) | vs BLAKE3-6r (e2e) |
|---|---:|---:|---:|---:|
| keccak-f (measured) | 72,672 | 73,020 | 1× | 0.10× |
| BLAKE3-6r (measured) | 5,316 | 7,337 | 10.0× | 1× |
| **Poseidon2 A** (derived) | **753** | ~922–1,897 (est) | 39–79× | 3.9–8.0× |
| **Poseidon2 A′** (derived, recommended) | **651** | ~768–1,443 (est) | 51–95× | 5.1–9.6× |
| Poseidon2 B (deg-4 bodies) | 479 | ~596–1,271 (est) | 57–123× | 5.8–12.3× |
| Poseidon2 C (internal bus, no memory) | 358 | 358 | 204× | 20.5× |

Note the shape change between the two columns: table-only, Poseidon2 looks 112× better than keccak; end-to-end that collapses toward ~50–95×, because keccak's plumbing is rounding error against its enormous table while Poseidon2's plumbing is comparable to its entire chip.

## (d) Caveats

1. **The chip is I/O-bound, not hash-bound.** Core 344 cells; syscall apparatus 283 cell-equiv (139 cols + 144 aux) even in the in-place variant. Every lever that removes memory crossing beats every lever inside the permutation: in-place ABI −14%, internal Merkle-parent bus −45% (358).
2. **Canonical `< p` input range checks are required and cheap.** 2 cells + 3 constraints per element, 0 sends, 24 cells for all 12. SP1 Hypercube ships exactly this check on both inputs *and* outputs (`others/hypercube-verifier/crates/core/machine/src/operations/sp1_field_word.rs:44-88`; `input_range_checkers[16]` + `hash_result_range_checkers[16]` at `syscall/precompiles/poseidon2/air.rs:66-70`) — so it isn't optional in practice. Separately: absorbing *arbitrary* byte strings rather than chip-produced digests needs 7-byte-per-element packing to stay injective, cutting sponge rate 32 B → 28 B.
3. **The degree budget is the one thing making us expensive, and it's ours alone.** Every mined design runs ungated bodies. Our ×μ factor doubles the core (344 vs 172 at deg-3 bodies) and quadruples it against zisk's deg-7 budget (344 vs 86). Good news: `logup_max_degree` already floors any table with committed pairs at 3 (`crypto/stark/src/lookup.rs:2287-2298`), so degree 3 is free. Going to 4 costs one composition part for that table alone — `composition_poly_degree_bound = trace_length·(max_degree−1)` (`lookup.rs:1078`), i.e. 3 parts instead of 2 — in exchange for −172 cells/row. That trade is plausibly a win and should be measured, not assumed.
4. **The verifier hash must switch, and that is the real bill.** All three Merkle backends are keccak (`crypto/stark/src/config.rs:10,19,23`). A Poseidon2 chip pays for nothing unless FRI/Merkle/FS move to Poseidon2 — which means a new GPU Merkle kernel (the keccak one is at `crypto/stark/src/gpu_lde.rs:861`) and a native-prover slowdown of roughly 5–10× per byte versus keccak (**order-of-magnitude, unmeasured**). BLAKE3 is the opposite trade: faster than keccak natively, so switching costs the prover nothing. This asymmetry appears nowhere in the cell count and is the single biggest difference between the two candidates.
5. **Keccak is not displaceable either way.** EVM/ethrex needs keccak256. Poseidon2 and BLAKE3-6r compete for the same internal-hash slot.
6. **Constraint-eval cost ≠ cell cost.** The 22 internal rounds carry non-S-boxed state as symbolic linear combinations — degree stays 1 (that's the point) but fan-out reaches ~30 terms by the last round, ~700 extra field mults per row. Fine, provided the IR stays a DAG.
7. **Always-on AIR tax.** `FIXED_TABLE_COUNT` +1. Per the EC regression (PR #871: +3 near-empty AIRs → +25% prove time), a real-block ABBA is mandatory regardless of how good the cell count looks.
8. **Uncertainty.** Core 344 is exact given the design and validated to 1% against zisk. I/O is exact given the shipped idiom. Table-only band: **620–700 for A′**. End-to-end is the soft number, band **768–1,443**, and it is directly measurable rather than arguable.

## (e) Verdict

Poseidon2 beats BLAKE3-6r here, and by a solid margin: **8.2× table-only (651 vs 5,316), 5–9× end-to-end.** The derivation is well-anchored — the same model reproduces Plonky3's Goldilocks figure exactly and zisk's to 1% — so I'd defend the number itself. What I would not defend is the conclusion that this justifies building it. Measured against keccak end-to-end, BLAKE3-6r already captures **91%** of the total addressable saving per merge (65,683 of 72,120 cell-equiv); Poseidon2 adds the remaining 9%. And Poseidon2 cannot go much lower as a syscall — roughly half its cost is the ecall/MEMW apparatus it shares with every other chip, so even a perfect permutation would only reach ~400. Meanwhile it uniquely imposes a native-prover hashing slowdown and a new GPU Merkle kernel that BLAKE3 does not, and the in-VM digests it produces are field elements crossing a byte-addressed memory, which is what the canonicity gadget and the 64 AreBytes sends are paying for. The decision should turn on one measurement nobody has taken: after the BLAKE3 switch, what share of a real recursion-verifier trace is still hashing? That is precisely the question the EC campaign skipped — a −61.9% win on 0.61% of the trace — and it is cheap to answer before committing to a chip. If Poseidon2 is pursued anyway, the leverage order is unambiguous and none of it lives in the permutation: in-place ABI (−14%), internal Merkle-parent bus (−45%), then relaxing the μ-gated degree cap (−23%).