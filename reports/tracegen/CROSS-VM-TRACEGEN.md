# FINAL BENCHMARK — main vs PR (perf/tracegen-cpu-optimizations @ af9751c9), ethrex 5tx, ABBA

Metrics: total prove time, trace generation time, trace-gen % of total. 4 runs/variant, medians.
PR = full trace-gen work (direct-to-column + histogram) + #794 (parallel histogram) + `zeroed_fe_vec`.

| machine | metric | main | PR | Δ |
|---|---|--:|--:|--:|
| **Apple M3 Max** (arm64) | trace generation | 2.22s | **1.49s** | **−33%** |
| | total prove | 25.5s | 25.2s | −1.2% |
| | trace-gen % of total | 8.6% | 5.9% | −2.7pp |
| **Server .226** (16-core x86, CPU) | trace generation | 4.18s | **3.17s** | **−24%** |
| | total prove | 56.2s | 55.0s | −2.2% |
| | trace-gen % of total | 7.4% | 5.8% | −1.6pp |
| **RTX 5090 (GPU)** | trace generation | 9.3s | **4.6s** | **−51%** |
| | **total prove** | **37.8s** | **29.5s** | **−22%** |
| | trace-gen % of total | 25.6% | 15.6% | −10pp |

**GPU is the payoff:** on GPU this PR cuts TOTAL prove **~22%** (vs ~1–2% on CPU), because the GPU offloads
LDE/commit/FRI so trace-gen is ~26% of total (vs ~7–9% on CPU). GPU-fused path confirmed
(`Aux commit (Merkle, CPU only) = 0.00s`). PR trace-gen rock-stable (4.48–4.66s); main noisier on the
shared 256-core box (8.8–14.3s), but every PR total (28–31s) beat every main total (35–39s).

On CPU: trace-gen cut −24% to −33% (machine-dependent, bandwidth-bound); total prove only −1% to −2%
(LDE/commit/FRI-dominated). Every PR run beat every main run on trace-gen, all three machines.

---

# Cross-zkVM trace-generation comparison — lambda_vm vs OpenVM vs SP1 (ZisK pending)

## Three-way summary (same 16-core box, same ethrex 10-tx block)

| VM | instructions | trace-gen | core prove | trace-gen % |
|---|--:|--:|--:|--:|
| **lambda_vm** (branch) | 6.8M (RV64) | **4.3s** | **80.7s** | **5.3%** |
| **SP1** (v5.0.8) | 2.67M (RV32) | 5.3s | 355s | **1.5%** |
| **OpenVM** (v1.4.1) | 124M (RV32) | 22.5s | 1641s | **1.4%** |

core prove = STARK proof excluding recursion/compression (SP1 total incl. `compress` = 1048s; OpenVM
incl. aggregation = 58 min). **Consistent across all three:** the low trace-gen % in SP1 and OpenVM is a
slow-proving/large-denominator artifact — our trace-gen is competitive-to-faster in absolute wall-clock,
and our core prove is 4–20× faster. No trace-gen efficiency problem on our side. SP1 even runs *fewer*
instructions than us (precompiles collapse the transfer crypto) yet its trace-gen is slower and its prove
4× slower (it pads each shard to ~2²¹ rows). SP1's trace-gen span = `generate main traces` (2 shards:
1.38s + 3.94s); it's pipelined/overlapping so treat as work, not serial wall-clock.

---



**Question:** what fraction of total proving time is spent on *trace generation* (materializing the
main-trace columns, before LDE/commit), for the same ethrex block, across zkVMs — and is our ~16–20%
a problem?

**Workload:** an ethrex 10-transfer L1 block, CPU proving only.

**TL;DR:** The scary-looking "OpenVM spends <2% on trace-gen, we spend ~16–20%" is a **ratio artifact**,
and it inverts under scrutiny. On identical hardware our trace-gen is **~5× faster in wall-clock** and
our whole prove **~20× faster**. Our high *fraction* is a sign our proving is efficient (small
denominator), not that our trace-gen is bloated. There is **no trace-gen inefficiency to fix** — but
OpenVM's code does contain two genuinely transferable techniques (below).

---

## 1. The measurements

### 1a. Our number is hardware-dependent (this is the crux)

Same lambda_vm code, same ethrex 10-tx block, two machines:

| machine | trace-gen time | total prove | trace-gen % |
|---|--:|--:|--:|
| 2× EPYC 9J14, 384 threads (PR benchmark box) | 5.5s | 33.5s | **16.5%** |
| 1× 16-core Debian (the OpenVM box) | 4.3s | 80.7s | **5.3%** |

Trace-gen time barely moves (it's **memory-bandwidth-bound** — more threads don't help), but the total
more than doubles on fewer cores (the LDE/commit/FRI is **compute-bound** and scales with cores). So the
*fraction* swings with core count. The 16.5% in the PR table is correct **for the EPYC**; 5.3% is the
same prover on the 16-core box.

### 1b. Honest same-server comparison (both on the 16-core Debian box)

| metric | **lambda_vm** (branch) | **OpenVM** (v1.4.1) |
|---|--:|--:|
| instructions executed | 6.8M (RV64) | 124M (RV32) |
| main-trace cells | 638M | 24.6B |
| **trace-gen time** | **4.3s** | **22.5s** |
| rest of prove (commit/FRI/quotient) | ~76s | ~1611s |
| **total core prove** | **80.7s** | **1641s** |
| **trace-gen %** | **5.3%** | **1.4%** |

- We are **~5× faster at trace-gen** (4.3s vs 22.5s) and **~20× faster overall** (80.7s vs 1641s).
- OpenVM's 1.4% is small only because its proving denominator is ~20× larger and it generates ~39× more
  trace — **not** because its trace generation is quick (its trace-gen is actually *slower* than ours).

## 2. Why OpenVM does ~18× more instructions / ~39× more cells

The guests are the **same program** (ethrex/LEVM block execution) but **not** a byte-identical
binary+input:

| | lambda_vm | OpenVM |
|---|---|---|
| ISA | riscv64 (RV64) | riscv32 (RV32) |
| ethrex version | pinned rev of `feat/lambdavm-prover-backend` | ethrex `main` (666582c) |
| crypto/precompiles | `lambda-vm-ethrex-crypto`, VM-accelerated (off the instruction stream) | openvm precompile crates |
| input block | committed `ethrex_10_transfers.bin` | ethrex-replay synthetic 10-transfer block |

Dominant factor: **RV32 vs RV64.** ethrex is saturated with 64-bit and 256-bit (U256) arithmetic —
balances, gas, hashing, secp256k1 field math. On RV32 each 64-bit op splits into 2+ instructions and each
256-bit op into many more; on RV64 they're native. Secondary: our accelerated crypto keeps signature
recovery/keccak off the instruction stream. This ~18× instruction bloat → ~39× more trace → the ~20×
slower prove. It is a **real architectural advantage of our VM**, not a measurement artifact — and it's
why per-instruction normalization across the two isn't clean.

## 3. Conclusion

- The "2% vs 20%" is a **denominator/work-volume artifact**, not evidence of a trace-gen problem.
- On the same box we win trace-gen **5×** and total **20×**.
- Our trace-gen fraction is *healthy*: it's a meaningful slice of a small total, which is exactly why the
  PR's ~2× trace-gen reduction was worthwhile (for OpenVM the same work would be lost in the noise).

---

## 4. What OpenVM does in trace generation, and what's worth stealing

OpenVM's CPU trace fill is **scalar** (no SIMD/PackedField in column fill — that's only in Poseidon2 and
the FFT). Its speed comes from **memory layout**, which is our (bandwidth-bound) lever. Two techniques
transfer:

### Steal #1 — Record-in-place + recompute derived columns
Instead of materializing an op struct then scattering it to columns, OpenVM writes a tiny packed
`#[repr(C)]` **byte** record into the *front of the destination trace row* (aliasing the `&mut [F]`
buffer), then during fill expands the row **in place, in reverse field order**, and **recomputes derived
columns instead of storing them** (e.g. the ALU result and timestamp range-check decompositions are
regenerated at fill, never stored). Record ≈ 9 bytes vs a 48-byte+ row — an explicit "compute is cheaper
than bandwidth" bet.
- Code: `extensions/rv32im/circuit/src/base_alu/core.rs:168-288`; framework in
  `crates/vm/src/arch/integration_api.rs` and `record_arena.rs`.
- **Relevance:** a more aggressive version of our `RegRow` direct-to-column path — we removed the struct
  only for register accesses; they remove it everywhere *and* drop stored results. Field-agnostic; the win
  is *larger* on Goldilocks (8-byte cells) than BabyBear (4-byte). Cost: `unsafe` aliasing + reverse-write
  order + a `record_size ≤ row_bytes` assert (same discipline as our `[u32;8]` shrink).

### Steal #2 — Atomic dense histogram with implicit preimage
We already have a dense multiplicity histogram. OpenVM's refinements: counters are `Vec<AtomicU32>`
incremented **concurrently from inside the parallel row fill** (`fetch_add(1, Relaxed)`) — no separate
counting pass, no per-thread merge — and the **preimage columns are never materialized** (row index *is*
`(x,y,z)`, enforced by the AIR); `generate_trace` writes only the multiplicity column over the full `2^k`
table.
- Code: `crates/circuits/primitives/src/bitwise_op_lookup/mod.rs:116-178`.
- **Relevance:** if our bitwise/range/PAGE tables still materialize preimage cells, dropping them cuts
  bandwidth on our largest (`1×2^21`) matrices; atomic counting folded into the fill removes our separate
  collect + tree-reduce.

### Architectural confirmations
- Within a chip: rayon `par_chunks_exact_mut` over rows. **Across chips: deliberately serial** to bound
  memory (their own `Perf` comment) — confirms our finding that the walk is bandwidth-bound and
  parallelism isn't the lever.
- Preallocate the whole matrix zeroed once, fill in place, reuse arenas across segments; padding rows are
  free (no-op dummy fill over already-zero buffer).

### Not worth chasing
- PackedField/AVX for fill (not used by OpenVM; Plonky3/BabyBear-specific, only in Poseidon2/FFT).
- The `u8×4` limb specifics (RV32-shaped), `DenseRecordArena` (GPU path), and the Plonky3 trait
  scaffolding — port the *pattern*, not the code.

## 4b. What SP1 does in trace generation (and what's worth stealing)

SP1's trace-gen is the same shape as ours/OpenVM (preallocate zeroed matrix, rayon fill in place,
`AlignedBorrow` direct-to-column, **no SIMD** in the fill — its perf reputation is a scalar C++/CUDA FFI,
i.e. codegen/GPU offload, not vectorization). Several SP1 choices are *worse* than what we already do:
per-thread **HashMap** multiplicity counting + serial merge (we have a dense/atomic histogram); a
**two-phase `generate_dependencies`** that re-runs `event_to_row` twice; and a separate **event array →
gather** into the trace (OpenVM's in-place byte-record and our direct-to-column both avoid this gather).

**The one genuinely transferable SP1 technique: `zeroed_f_vec` (calloc-backed zero init).** SP1 allocates
`vec![0u32; len]` then `transmute`s to `Vec<F>` instead of `vec![F::zero(); len]`. The former hits the
allocator's calloc / demand-zeroed-page path (OS zero pages touched lazily on first write); the latter is
an element-wise clone loop that often isn't lowered to `memset` — a full eager zeroing sweep of the whole
trace before fill.
- Code: `crates/core/machine/src/utils/mod.rs:159`.
- **Directly applicable to us — and we're on the slow path.** Every table uses
  `vec![FE::zero(); num_rows * NUM_COLUMNS]` (memw, cpu, page, branch, eq, commit, …). `FieldElement<F>`
  is `#[repr(transparent)]` over `BaseType` and our Goldilocks has **no Montgomery form**
  (`p = 2^64-2^32+1`), so canonical zero is all-zero bits → the transmute is sound. For a bandwidth-bound
  fill this removes a whole zeroing sweep of the ~5 GB main trace from `trace_build`.

**Verdict:** OpenVM's design is the one to learn from for our bandwidth-bound RV64/Goldilocks prover;
SP1's concrete contribution is just `zeroed_f_vec` (the easiest, safest item on the list).

### Results (measured on the 16-core box, ethrex 10-tx, ABBA)
- **Opt 1 (`zeroed_fe_vec`) — DONE, kept.** Baseline trace_build 4.32s → 4.22s (**−2.3%**), TOTAL unchanged
  (~80.7s). Correct (prove+verify green, `zeroed_fe_vec_matches_fe_zero` test). Small but free/safe.
  Implemented on branch `perf/tracegen-experiments` (helper in `tables/types.rs`, 37 sites converted,
  unused `FE` imports cleaned). Modest because the eager zeroing was only ~2% of our trace_build.
- **Opt 2b (drop preimage) — already done.** Our BITWISE/KECCAK_RC lookup tables are `is_preprocessed()`;
  preimage is committed statically once, only multiplicity columns filled per proof. At/ahead of OpenVM.
- **Opt 2a (eliminate bitwise-op Vec) — TESTED, NOT WORTH IT (negative result).** `p4_bitwise_collect` is
  21% of trace_build (~930ms). Converting the lt/memw_aligned/branch collectors to bump the histogram
  directly (no `Vec<BitwiseOperation>`) left p4 **unchanged** (~932ms → ~949ms, i.e. noise). Reason: p4's
  cost is the **histogram counting itself** — ~140M lookups each doing `row_index` + increment into an
  80 MiB dense array is *random-access* memory traffic (cache misses); the Vec write/read is *sequential*
  (cheap). Removing the Vec removes the cheap part, not the bottleneck. (Note: mul/dvrm not converted, but
  the random-access hypothesis predicts they wouldn't help either.) Also note 940ff62f already migrated the
  collector list to a `Fn(&mut BitwiseHistogram)` sink interface; the collectors just still `add_ops(&vec)`.
- **Radix-partition counting — TESTED, NEGATIVE (slower).** Idea: buffer the 140M histogram bumps by
  top index bits, flush per cache-resident 512 KiB slice, to cut the random-access counting cost. A
  single-threaded micro-benchmark showed **~1.9× faster** counting (925ms→483ms). But integrated into the
  real (parallel) prover it was **~15% slower** on p4 (967ms→1104ms; trace_build 4.32s→4.46s). Correct
  (prove+verify green), just slower. Lesson: the micro-bench misled — single-thread + uniform-random ≠ the
  real parallel run with real (higher-locality) data, where direct counting is already cache-served and
  radix's buffering overhead exceeds the savings.
- Opt 3 (recompute-don't-store) — not tested; predicted small (PR already pulled the direct-to-column lever).

## FINAL VERDICT (all experiments)
The **only** worthwhile trace-gen optimization found is **Opt 1 (`zeroed_fe_vec`, ~2%)**. Everything more
ambitious is null, negative, or already done:
| optimization | result |
|---|---|
| `zeroed_fe_vec` (SP1) | **+2% trace_build — KEEP** |
| preprocess preimage (OpenVM 2b) | already how we do it |
| eliminate bitwise-op Vec (OpenVM 2a) | null (cost is counting, not the Vec) |
| radix-partition counting | negative (slower in the real parallel prover) |
| record-in-place / recompute (OpenVM 1, 3) | already covered by direct-to-column; not worth pursuing |

Conclusion: our trace generation is already well-optimized (the PR's direct-to-column + histogram captured
the real levers). trace_build is only ~5% of total CPU prove; the remaining phase costs (the memory walk,
the multiplicity counting) are inherent. **Ship `zeroed_fe_vec`; the highest-leverage future work is in the
prover's LDE/commit/FRI, not trace generation.**

## Coworker PRs (#794, #795) — evaluated
- **#794 "Parallelize the two dominant bitwise-histogram sources internally" — REAL WIN, MERGE.** Different
  lever than our failed p4 experiments: not per-op cost, but **load-balancing** — the two heavy sources
  (in-walk lookups + MEMW_R) each pinned a single core (`par_chunks` can't work-steal into a busy chunk).
  #794 splits them into row-range slices round-robined into `cap` buckets. **Measured (ABBA, `.226`, same
  baseline):** p4_bitwise_collect **936→866ms (−7.5%)**, trace_build **4.29→4.21s (−1.9%)**, TOTAL
  unchanged. Byte-identical (commutative monoid), memory-neutral. It found the p4 inefficiency our
  radix/eliminate-Vec attempts missed.
- **#795 "Trace-gen cleanup" — cleanup, not perf.** MU_COLUMNS single source + pairwise-distinct assert,
  `push_reg_access([u32;2],[u32;2])`, `pub(crate)` visibility, stale docs. Byte-identical. Merge separately.
- **Orthogonality:** #794 (−1.9%) and our `zeroed_fe_vec` (−2%) hit different phases (p4 counting vs trace
  zeroing) → they **stack** to ~4% trace_build. Recommended: merge #794 + #795, then add `zeroed_fe_vec`.

### Suggested next experiments (easiest-first, independently shippable)
1. **`zeroed_f_vec`** — replace `vec![FE::zero(); N]` trace allocations with a calloc-backed
   `vec![0u64; N]` transmuted to `Vec<FE>` (Goldilocks zero = 0 bits, `repr(transparent)` — both verified).
   Lowest risk, from SP1. Measure trace_build delta.
2. Atomic histogram + drop preimage materialization on our bitwise/range/PAGE tables (from OpenVM). Measure.
3. Recompute-don't-store for one hot table (extend direct-to-column to omit a stored derived column and
   recompute at fill) — validates the "compute < bandwidth" bet on Goldilocks (from OpenVM).

---

## 5. Methodology / reproducibility

- Box: `app@195.154.100.227`, Debian 12, 16 cores, 62 GB RAM.
- lambda_vm: branch `perf/tracegen-cpu-optimizations`, `cargo build --release -p cli --features instruments`
  (CPU, no cuda); `cli prove ethrex.elf --private-input ethrex_10_transfers.bin --time` (trace_build +
  TOTAL), `--elements` (main-trace cells), `--cycles` (instructions). Warm-up + 2 runs (4.29s/4.27s
  trace-gen — stable).
- OpenVM: v1.4.1 via ethrex-replay (`custom block --tx eth-transfer --n-txs 10 --zkvm open-vm --action
  prove`), `FmtSpan::CLOSE` patch on ethrex-replay's subscriber to surface `trace_gen` /
  `stark_prove_excluding_trace` span durations; app-proof segments only (aggregation/recursion excluded to
  match our single-shot prove). Single full run (58 min wall incl aggregation).
- Build gotchas on OpenVM (ethrex@main OpenVM backend is ⚠️ flaky): needed `CC_riscv32im_risc0_zkvm_elf=clang`
  for the RISC-V C cross-compile; a stale `openvm-stark-sdk v1.2.1` pin conflicting with untagged
  `openvm-sdk` (pinned the openvm.git source to v1.4.1); and an off-by-one `include_bytes!` path for the
  guest ELF (placed the ELF where the path expected it). Worth reporting upstream.

## 6. Caveats
- Not a byte-identical guest/input (see §2), so per-instruction/per-cell normalization is not clean; the
  defensible claims are the same-box wall-clock ones.
- OpenVM number is a single run (its 58-min prove makes repeats expensive); the gap is an order of
  magnitude so error bars don't change the conclusion.
- SP1 and ZisK still pending (to be run on the same box, same guest).
