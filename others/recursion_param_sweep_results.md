# Recursion-Verifier Parameter Sweep — Results

Metric (minimize): `recursion_cost = guest_cycles + 200 * keccak_calls`
Inner workload: **ethrex_bench_16.bin** (16 txs, 11.62M cycles / 356 keccak / 64 ecsm — the heaviest
fixture available; shipped by the bench branch, heavier than ethrex_10_transfers=6.81M and the
documented 4-tx fallback). Inner ELF: `executor/program_artifacts/rust/ethrex.elf`.
Server: 32 cores, 124 GB RAM. Peak-RSS budget: ~105 GB.
Branch: `sweep/throwaway` = main(@3ea4f916, incl #844/#845/#847) + #853 (per-table instruments)
+ bench harness (`bench/recursion-full-queries`), composed on server. Not pushed.
Cont guest runs the **production archived (zero-copy) verify path** (`verify_continuation_and_attest`).

## Progress log
- DONE: Phase 0 — branch composed + cargo-check clean; guests {ethrex, recursion-cont-blowup2/4/8} + cli built;
  inner workload chosen = ethrex_bench_16.bin (11.62M cycles, 356 keccak, 64 ecsm — heaviest available);
  epoch grid sized (2^20→12 ep, 2^21→6 ep, 2^22→3 ep, 2^23→2 ep); baseline blowup2@2^21 measured + #853 report confirmed.
- DONE: Phase 1 — blowup {2,4,8} @ 2^21. Winner blowup8 (cost 3.68B, -59.2% vs b2), RSS 40.6GB.
  Cost ~ proportional to queries (b2=219,b4=110,b8=73); marginal cost/GB b2->b4=505M/GB, b4->b8=73M/GB.
- DONE: Phase 2 — epoch axis at blowup8: e20(12ep)=6.09B/27GB, e21(6ep)=3.68B/41GB, e22(3ep)=2.48B/70.5GB,
  e23(2ep)=OOM (123.6GB, SIGKILL). Epoch ceiling at b8 = 2^22. Cost ~halves per epoch-count halving
  (12->6: 0.60x, 6->3: 0.67x; sublinear because the GLOBAL/pages proof is ~epoch-invariant).
- DONE: Phase 3-4 (table caps) — patched continuation.rs to read MaxRowsConfig from env at runtime.
  Cap knee at capall2^epoch-log2 (unsplit cycle-bound tables); −27% at b8_e22 for +7GB, cheapest cost/GB.
- DONE: iso-memory + caps-fair + never-split (caps=epoch bound) sweep. VERDICT: blowup4 dominates the whole
  frontier from ~73 GB up; blowup8 pushed off the top. Memory WALL mapped (b4_e24 caps2^24 OOMs @ 123.6 GB).
- FINAL BEST: b4_e24+caps2^23 = 1.15B @ 113.8 GB (full box) | b4_e23+caps2^23 = 1.74B @ 76.6 GB (within 105 GB).
  Both −87% / −81% vs the 9.01B b2_e21 baseline. Prove-times recorded (col); b4 frontier ~2× faster than b8.
- COMPLETE — 21 configs measured (+2 OOM walls documented); see frontier + model + policy below.

### Table-cap tunability (Phase 0 answer)
Per-table caps live in `MaxRowsConfig` (prover/src/tables/mod.rs:105; Default from `max_rows::*`).
The continuation prove path hardcodes `&MaxRowsConfig::default()` at continuation.rs:1100 — NOT env-exposed.
Cheaply exposable via a one-time small patch that reads env at runtime (no per-config rebuild).
Default caps: CPU/MEMW/MEMW_A/DVRM/CPU32 = 2^19; MUL/LT/SHIFT/LOAD/BRANCH/MEMW_R/EQ/BYTEWISE/STORE = 2^20.
Eff. widths (for Phase 4 memw analysis): CPU 194, DVRM 136, MEMW 127, MEMW_A 89, MUL 74, SHIFT 72, LT 42,
LOAD 33, BRANCH 32, MEMW_R 31. → narrow high-row tables (MEMW_R w31, LT w42) consolidate cheaply.

## Measurement protocol (per config)
(A) Prove+dump with instruments+peak memory:
```
/usr/bin/time -v env RECURSION_DUMP_PRESET=<blowupN> \
  RECURSION_DUMP_INNER_ELF=<abs ethrex.elf> \
  RECURSION_DUMP_INNER_INPUT=<abs ethrex_10_transfers.bin> \
  RECURSION_DUMP_EPOCH_LOG2=<E> \
  cargo test --release -p lambda-vm-prover --features instruments --lib \
    test_dump_recursion_input -- --ignored --nocapture 2>&1 | tee run_<cfg>.log
```
→ peak RSS (Maximum resident set size), per-epoch per-table report (#853), epoch count, blob size.
Blob at /tmp/recursion_input.bin → copy to blob_<cfg>.bin.

(B) Measure recursion cost (deterministic; each config its own blob):
```
target/release/cli execute executor/program_artifacts/recursion/recursion-cont-<blowupN>.elf \
  --private-input blob_<cfg>.bin --cycles
```
→ guest_cycles, keccak_calls. recursion_cost = cycles + 200*keccak. Re-run once to confirm integer.

## Fitted model (from Phases 0-2, 5 points; max |err| 0.92%)
2D: `recursion_cost = 4.556M·(queries·epochs) + 9.20M·queries + 69.23M·epochs + 585.7M`
- Matches the plan's `queries × (a·N_tables + b) + c`: the 4.556M·queries·epochs term = queries × per-epoch-tables body
  (N_tables scales with epochs); 9.20M·queries = queries × epoch-invariant GLOBAL/pages openings;
  69.23M·epochs + 585.7M = query-independent per-epoch work + fixed overhead.
- queries(blowup): b2=219, b4=110, b8=73 (preset-fixed; higher blowup legitimately needs fewer queries).
Clean 1D at blowup8: `cost = 401M·epochs + 1274M` (per-epoch slope 401M; epoch-invariant floor 1274M = global proof + fixed).
Extrapolation: for a block of C total inner cycles at epoch 2^E, epochs = ceil(C / 2^E); plug into the model.
Bigger box (more RAM) → larger epoch (fewer epochs) until the per-epoch RSS (≈ doubles per +1 in E) hits the budget.

## Results table
| cfg | blowup | epoch_log2 | epochs | queries | guest_cycles | keccak | recursion_cost | peak_RSS_GB | inner_prove_s | OOM? | notes |
|-----|--------|-----------|--------|---------|--------------|--------|----------------|-------------|---------------|------|-------|
| b2_e21 | 2 | 21 | 6 | 219 | 7,326,585,399 | 8,435,774 | 9,013,740,199 | 14.75 | 102 | no | Phase0 baseline; blob 391MB; execute reproduced exactly |
| b4_e21 | 4 | 21 | 6 | 110 | 4,039,402,550 | 4,674,102 | 4,974,222,950 | 22.80 | 140 | no | Phase1; -44.8% vs b2; blob 207MB |
| b8_e21 | 8 | 21 | 6 | 73 | 3,000,472,862 | 3,396,812 | 3,679,835,262 | 40.63 | 253 | no | Phase1 BEST; -59.2% vs b2; blob 144MB |
| b8_e20 | 8 | 20 | 12 | 73 | 5,039,358,228 | 5,235,126 | 6,086,383,428 | 26.95 | 328 | no | Phase2 anchor; blob 233MB |
| b8_e22 | 8 | 22 | 3 | 73 | 1,983,546,515 | 2,470,330 | 2,477,612,515 | 70.54 | 223 | no | Phase2 BEST feasible; -32.7% vs b8_e21; blob 99MB |
| b8_e23 | 8 | 23 | 2 | 73 | — | — | — | 123.62 | — | **OOM** | Phase2; SIGKILL, peak 123.6GB > 124GB phys & 105GB budget; epoch ceiling is 2^22 at b8 |
| b8_e22_capall20 | 8 | 22 | 3 | 73 | 1,825,260,465 | 2,241,680 | 2,273,596,465 | 70.78 | 224 | no | Phase3; all caps->2^20 (CPU 8->4 chunks); -8.2% vs b8_e22 at ~same RSS |
| b8_e22_narrow23 | 8 | 22 | 3 | 73 | 1,669,887,511 | 1,957,253 | 2,061,338,111 | 76.12 | 266 | no | Phase4; MEMW_R->2^23, med tables->2^22, CPU default; -16.8% vs b8_e22 for +5.6GB |
| b8_e22_capall21 | 8 | 22 | 3 | 73 | 1,586,597,174 | 1,863,577 | 1,959,312,574 | 73.44 | 269 | no | Phase3; all caps->2^21 (CPU->2 chunks); -21% vs b8_e22 for +3GB (superseded by capall22/23) |
| b4_e22 | 4 | 22 | 3 | 110 | 2,661,590,116 | 3,423,431 | 3,346,276,316 | 39.71 | 177 | no | iso-mem; at ~40GB BEATS b8_e21 (3.68B) |
| b4_e23 | 4 | 23 | 2 | 110 | 2,186,236,700 | 2,978,259 | 2,781,888,500 | 71.45 | 117 | no | iso-mem; b4 at ~72GB > b8_e22 (2.48B) |
| b2_e24 | 2 | 24 | 1 | 219 | 3,137,525,226 | 4,710,419 | 4,079,609,026 | 56.85 | 127 | no | iso-mem; 1 epoch = monolithic regime (workload < 2^24) |
| b2_e23 | 2 | 23 | 2 | 219 | 3,952,475,757 | 5,439,565 | 5,040,388,757 | 42.33 | 69 | no | iso-mem curve |
| b2_e22 | 2 | 22 | 3 | 219 | 4,817,848,015 | 6,228,402 | 6,063,528,415 | 23.85 | 76 | no | iso-mem curve |
| b8_e22_capall22 | 8 | 22 | 3 | 73 | 1,470,813,010 | 1,676,531 | 1,806,119,210 | 77.42 | 270 | no | cap-push; -27% vs b8_e22; CPU unsplit at 2^22 |
| b8_e22_capall23 | 8 | 22 | 3 | 73 | 1,433,635,739 | 1,611,791 | 1,755,993,939 | 77.87 | 282 | no | cap-push; only -3% below capall22 (knee); NO prove-time spike |
| b8_e22_cap22_memwr23 | 8 | 22 | 3 | 73 | 1,433,480,740 | 1,611,791 | 1,755,838,940 | 79.17 | 279 | no | hybrid; == capall23 (MEMW_R already the binding table) |
| b4_e23_capall22 | 4 | 23 | 2 | 110 | 1,471,218,104 | 1,801,178 | 1,831,453,704 | 73.22 | 133 | no | CAPS-FAIR b4; ~tied w/ b8_e22_capall22 (1.81B) at LESS RAM |
| b4_e24_capall22 | 4 | 24 | 1 | 110 | 977,079,977 | 1,317,909 | 1,240,661,777 | 108.40 | 119 | over105 | b4 reaches 1-epoch monolithic (b8 can't); UNDER-capped (CPU/MEMW_R still split) |
| b4_e23_capall23 | 4 | 23 | 2 | 110 | 1,401,845,025 | 1,677,680 | 1,737,381,025 | 76.60 | 142 | no | never-split at e23; -5% vs b4_e23_capall22; edges b8_capall23 (1.76B@77.9) |
| b4_e24_capall23 | 4 | 24 | 1 | 110 | 907,533,312 | 1,194,411 | 1,146,415,512 | 113.80 | 123 | over105 | **BEST overall** 1.15B; MEMW_R unsplit, CPU 2 chunks; fits 124GB box, no OOM |
| b4_e24_capall24 | 4 | 24 | 1 | 110 | — | — | — | 123.62 | — | **OOM** | true CPU never-split (caps 2^24) SIGKILLs at 123.6GB > 124 phys — the memory WALL |

## DELIVERABLE — cost-vs-RSS Pareto frontier (FINAL, caps-fair)
Frontier point = nothing else has both lower recursion_cost AND lower peak RSS. Every b8/b4 point here is
with caps maxed for its epoch (unsplit), so the blowup comparison is fair.

| RAM | frontier config | recursion_cost | RSS_GB | prove_s | note |
|-----|-----------------|----------------|--------|---------|------|
| ~15 | b2_e21 (bl2, 6ep) | 9.01B | 14.7 | 101 | |
| ~23 | b4_e21 (bl4, 6ep) | 4.97B | 22.8 | 140 | |
| ~40 | b4_e22 (bl4, 3ep) | 3.35B | 39.7 | 177 | beats b8_e21 (3.68B@40.6) — b4 wins at mid-RAM |
| ~70 | b8_e22 (bl8, 3ep) | 2.48B | 70.5 | 223 | last point b8 holds |
| ~71 | b8_e22 + caps2^20 | 2.27B | 70.8 | 224 | |
| ~73 | b4_e23 + caps2^22 | 1.83B | 73.2 | 133 | b4 takes over from here up |
| ~77 | b4_e23 + caps2^23 | 1.74B | 76.6 | 142 | never-split at e23; edges b8+caps2^23 (1.76B@77.9) |
| ~108 | b4_e24 + caps2^22 | 1.24B | 108.4 | 119 | 1-epoch monolithic |
| ~114 | **b4_e24 + caps2^23** | **1.15B** | 113.8 | 123 | **BEST**; MEMW_R unsplit; fits 124GB box, no OOM |
| WALL | b4_e24 + caps2^24 | OOM | 123.6 | — | true CPU never-split SIGKILLs > 124GB phys |

BEST within the strict ~105 GB budget: b4_e23 + caps2^23 = 1.74B @ 76.6 GB (bl4 dominates b8's best here,
b8_e22+caps2^23 = 1.76B @ 77.9 GB, at less RAM and half the prove-time). BEST if you allow ~114 GB (the full
box, 8% headroom, no OOM): b4_e24 + caps2^23 = 1.15B (−87% vs the 9.01B baseline).

HEADLINE (caps-fair, epoch-pushed, caps=epoch-bound — the corrected verdict):
- blowup8 is PUSHED OFF the top of the frontier. Once every point is compared with caps ≥ its epoch bound,
  blowup4 dominates the ENTIRE ≥73 GB range; blowup8 holds only ~70 GB. b4's lower LDE lets it reach the
  1-epoch (monolithic) regime, which b8 cannot (b8_e23 already needed 123 GB for 2 epochs). "Lower blowup
  buys more epoch per GB" — strongly CONFIRMED. b4's frontier points are also 2× faster to prove than b8's.
- The controlling variable is EPOCHS (fewer = lower cost, ~linear); blowup matters mainly through how large
  an epoch it lets you fit. At the memory ceiling, LOWER blowup wins because it reaches fewer epochs.
- The memory WALL is documented: b4_e24 fully-unsplit (caps2^24) OOMs at 123.6 GB; the feasible ceiling is
  caps2^23 (CPU stays 2 chunks, everything else unsplit) = 1.15B @ 113.8 GB.

## Which knob per GB + prove-time (narrative)
1. epoch/epochs — the master lever: recursion_cost is essentially linear in epoch COUNT (b8: cost=401M·ep+1274M).
   Fewer epochs (bigger epoch) is the biggest reducer; it is memory-bounded, so lower blowup (cheaper per-epoch
   RAM) reaches fewer epochs → the b4_e24 win.
2. TABLE CAPS — cheapest cost/GB: unsplitting cycle-bound tables at the epoch bound (capall2^E) cuts −27% at
   b8_e22 (2.48B→1.81B) for +7 GB. KNEE at caps=2^epoch-log2: capall23/hybrid add only −3% and NO prove-time
   spike (270→282 s), because the tallest table (CPU, 4.2M rows/epoch) is already ~unsplit at 2^22. So the rule
   is simply: caps ≥ epoch cycle bound, no higher.
3. blowup — real but secondary once caps+epoch are fair (see headline).
PROVE-TIME: 69–282 s across all configs (indicative, machine-bound). It rises with blowup (b8 caps ~270-282 s)
and epoch count, but the winner b4_e24 is the FASTEST at 119 s. End-to-end = inner_prove + recursion_prove, and
recursion_prove ∝ recursion_cost (billions of cycles) dominates the ~100-280 s inner prove by orders of
magnitude — so minimizing recursion_cost is the right objective; NO config inverts (recursion saving always ≫
inner-prove overhead). If anything, prove-time REINFORCES the verdict: b4_e24 is best on cost AND prove-time.

## Derived mixed-blowup policy (analytic, from per-table data)
A table's recursion-cost contribution ∝ queries(blowup) × chunks; its memory ∝ rows × width × blowup.
Cost-saving per extra GB of raising a table b2→b8 works out to ∝ 1/(cap × width) (the rows cancel), so the
priority to keep a table at HIGH blowup (and the LAST to demote under memory pressure) ranks by cap×width:
  MEMW_R(32.5M) < BRANCH(33.6M) < LOAD(34.6M) < LT(44M) < MEMW_A(46.7M) < MEMW(66.6M) < DVRM(71.3M)
  < SHIFT(75.5M) < MUL(77.6M) < CPU(101.7M).
So the WIDE tables (CPU, MUL, SHIFT) are the first to demote to a lower blowup when RAM is tight; the narrow
high-row tables (MEMW_R, BRANCH, LOAD, LT) stay at blowup8.
BUT for THIS workload/box, uniform blowup8 already FITS (b8_e22 = 70.5 GB < 105 GB), so no demotion is
needed — uniform blowup8 is cost-optimal here. Mixed-blowup only pays off for a larger block where uniform
blowup8 would exceed RAM: then demote CPU (and MUL/SHIFT) to blowup4/2 first, keep the rest at blowup8.
Table-CAP policy (empirical, stronger for this workload): set every cap ≥ the epoch cycle bound so
cycle-bound tables are UNSPLIT at the chosen epoch (capall2^epoch-log2). This is the cheapest cost/GB knob
and should be the default at every frontier point; epoch then spends whatever RAM remains.

## Fitted-model extrapolation (recipe)
For a block of C inner cycles, blowup b (queries q_b: b2=219,b4=110,b8=73), epoch 2^E:
  epochs = ceil(C / 2^E);  recursion_cost ≈ 4.556M·(q_b·epochs) + 9.20M·q_b + 69.23M·epochs + 585.7M.
Pick the largest E whose per-epoch peak RSS (≈ doubles per +1 E, measured ~ blowup-scaled) fits the budget,
caps maxed. On a bigger box, raise E (fewer epochs) until RSS = budget; that is the min-cost config.
