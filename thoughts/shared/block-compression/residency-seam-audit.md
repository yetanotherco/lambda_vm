# Bounded-Residency Proving — Code Audit (the "seam audit")

Delegated audit, 2026-08-12. Worktree `/Users/maurofab/workspace/lambda_vm-blake3-impl`
(branch `blake3-real-hash`). Read-only; nothing built, nothing edited. This is the
document PLAN.md's P-b entries cite; CENSUS.md Part 2 §1 carries the census agent's
independent read of the same question and the reconciliation boxes.

**Headline: the linearity assumption is TRUE, but the coefficient is wrong (too high by
~2.1× for a `KECCAK_RND` chunk). Peak really is a SUM over all sub-proofs for the main
trace + main LDE + main tree + aux trace; the aux LDE and all round-2-to-4 buffers are
`k`-bounded, not summed. Nothing in the prover bounds residency to a few tables at a
time.**

## 1. Does the prover hold ALL table traces resident simultaneously?

✓ VERIFIED — yes, unconditionally, structurally forced by the API.

- `crypto/stark/src/prover.rs:50-54`: `AirTracePair<'a,…> = (&'a dyn AIR, &'a mut
  TraceTable, &'a PI)`; `multi_prove` takes `Vec<AirTracePair>` (`prover.rs:3032-3036`) —
  every trace must exist and be borrowed for the entire call. No iterator/factory/
  TraceSource exists; the only producers are `VmAirs::air_trace_pairs`
  (`prover/src/lib.rs:542`) and `LfmAirs::air_trace_pairs`
  (`prover/src/lfm/airs.rs:551-583`), both building a complete Vec.
- LFM wrap: `prover/src/lfm/trace.rs:43` `keccak_rnd: Vec<TraceTable>`; all N chunk
  traces built eagerly at `trace.rs:162-167`; `lfm/proof.rs:140-145` passes them whole.
- The one streaming precedent — `trace_builder.rs:2922-2949` `chunk_and_generate` under
  `StorageMode::Disk` — spills each chunk's trace to mmap after build (VM only, trace
  only, never the LDE) and still returns a Vec of all chunks.

## 2. Where peak accretes (verified round structure, `prover.rs:3032-3689`)

| Stage | Residency class | Bytes |
|---|---|---|
| Trace build (caller) | global — alive past return | rows·cols·8 |
| R1 main LDE (`commit_main_trace` → drained to `main_ldes` at :3145, :3201) | **retained for all N** | rows·blowup·cols·8 |
| R1 main Merkle (`TableCommit.tree` :114; cells :3306-3309) | retained for all N | ≈32 B/LDE row |
| FS boundary (:3190-3204 roots absorbed → :3219-3225 shared LogUp challenges) | — | — |
| Aux trace (`lookup.rs:1209-1211` writes into caller-owned TraceTable) | **global — never freed inside multi_prove** | rows·aux_cols·24 |
| Aux-build transients (`lookup.rs:1272` full `columns_main()` copy; :1287-1337 committed_columns) | per-table, ≤ k | rows·(main·8+aux·24) |
| Aux LDE + aux tree (:3377-3516, inside `aux_stage`) | ≤ k | rows·blowup·aux_cols·24 |
| Composition/DEEP/FRI (R2-4) | ≤ k | ~160 B/LDE row |

The retention is documented in the code's own words — `prover.rs:263-274` (`Lde` struct
doc): main LDEs "all N tables' … live at once (O(N × main_cols × lde_size))"; aux "at
most `table_parallelism()` of them coexist". The barrier is real: `run_admitted`
(:689-722) joins a thread scope before roots are absorbed and challenges sampled, before
the second `run_admitted` for the fused phase at :3628. The independent in-repo model
agrees: `prover/src/auto_storage.rs:243-266` / :53-95.

**⚠ Fiat-Shamir forces the ROOTS, not the LDEs** (`prover.rs:3196-3225`, verifier mirror
`verifier.rs:1295-1317`). After the shared challenge, each table gets a private
transcript fork (:3263-3271 / :1349-1356) and every later round is per-table
independent; each `StarkProof` is self-contained (:3856-3887). Retaining the LDE is a
performance choice, not a protocol constraint — this is the seam that makes bounded
residency possible.

cuda note: `device_only` relocates the main LDE to VRAM without reducing N-way retention
(? INFERRED that this makes cuda strictly worse for this workload).

## 3. What disk-spill / StorageMode bound TODAY

Spilled to mmap under `StorageMode::Disk`: main trace Table (:3113-3122, trace_builder
:2938-2949, :3658-3680); aux trace Table (:3366-3371); main/precomputed/mult Merkle trees
(:1172, :1186, :1239, :1273, :1293 via `spill_tree` :1316-1331); aux Merkle CPU path only
(:3501; the GPU aux arms return early at :3419-3423/:3457-3461 without spilling).

NEVER spilled (✓ VERIFIED — spill_tree has exactly 6 call sites; `LDETraceTable` has no
mmap field, `trace.rs:316-343`): main LDE, aux LDE, composition evals, composition tree,
FRI layers, the `columns_main()` copy.

Wiring: the only live selector is `auto_storage::decide` on the monolithic VM path
(`lib.rs:1221-1226`); `continuation.rs` (:791-796, :991-992, :1282-1283) hardcodes Ram;
**the wrap (`lfm/proof.rs:140-145`) passes `Default::default()` = Ram**; feature
`disk-spill` is off by default (`prover/Cargo.toml:8`, opt-in :17) — under a normal build
the parameter does not exist (cfg at `prover.rs:3035`).

**Spilling moves allocation (trace + trees only); the main LDE — the largest N-way
retained buffer — stays on the heap in every configuration, and Disk is unreachable from
the LFM path anyway.**

## 4. Verdict

(a) available behind flags? **NO.** `TABLE_PARALLELISM=1` bounds only aux/R2-4
transients; `FORCE_DISK_SPILL` unreachable from LFM and wouldn't touch the LDE.

(b)/(c): **a real refactor with named seams** — deeper than "moderate", shallower than
"fundamental". Not forced by FS ordering, not by the proof struct, not by chunk pairing
(`chunking.rs:12-21`: zero cross-chunk logic — a chunk trace is a pure function of its
`round_ops` slice, so **regeneration is trivially available**).

What forces retention: (1) the `Vec<(&dyn AIR, &mut TraceTable, &PI)>` signature; (2) the
eagerly-built `LfmTraces.keccak_rnd`; (3) the deliberate R1 LDE cache (:3145, :3201,
:3311-3316); (4) `allocate_aux_table` writing into the caller-owned trace.

### The seams

| # | File:line | Change |
|---|---|---|
| S1 | `prover.rs:3032-3036` | `multi_prove` takes a per-index producer (`Fn(usize) -> (AIR, TraceTable, PI)` + `num_airs`); `run_admitted` already dispatches by index |
| S2 | `prover.rs:50-54` | `AirTracePair` stops carrying `&mut TraceTable` — the task owns its trace |
| S3 | `prover.rs:3144-3145, 3190-3204, 3311-3316` | delete `main_ldes`/`main_lde_cells`; drop LDE at end of R1 task, recompute at top of aux_stage (twiddles process-cached :517-574) |
| S4 | `prover.rs:3306-3309` | trees: keep (32 MiB/chunk), spill via existing `spill_tree`, or recompute → root-only peak |
| S5 | `lookup.rs:1209-1211` | aux trace freed with the owned TraceTable (automatic once S2 lands) |
| S6 | `lfm/airs.rs:551-583` + `lfm/trace.rs:162-167` | `air_trace_pairs` → lazy per-index generator over `chunking.split(&round_ops)`; keep `round_ops` (~92 MB) instead of N traces |
| S7 | `lfm/proof.rs:140-145` | call-site update |

Wire compatibility preserved: verifier ordering depends only on root-absorption order and
the index-domain-separated fork, both unchanged.

### The floor

Per KECCAK_RND chunk (2^19 rows × 1480 main + 516 aux ext, blowup 2): main trace 5.78 /
main LDE 11.56 / aux trace 6.05 / aux LDE 12.09 / columns_main copy 5.78 /
committed_columns 6.05 / trees+R2-4 ≈0.2 GiB. Peak ≈ **17.37·N + 30.2·k GiB**:

| N | today (k=1) | S3 only (recompute LDE, keep traces) | S3+S6 (regenerate traces; roots-only) |
|---|---|---|---|
| 23 | ~430 GiB | ~309 GiB | **~49 GiB** |
| 133 | ~2,341 GiB | ~819 GiB | **~56 GiB** |

**Bounding only the LDE does NOT reach ~50 GiB** — the main+aux traces are still summed;
the flat floor requires the trace streamed too (regenerate each chunk twice: R1 commit +
aux build). Cost of the floor: 2× chunk trace generation + 2× main coset LDE per chunk,
serialization to k=1 — ? INFERRED roughly +40-60% wall on the KECCAK_RND family.

### Coefficient correction to the census projection

`MEASURED_BYTES_PER_CELL = 33.7` (`wrap_tests.rs:629-644`) prices a KECCAK_RND chunk at
49.8 GiB; the true persistent cost is 23.4 GiB — **~2.1× high** for this shape (aux LDE
priced persistent when k-bounded; slice-0 calibration mix). Linearity in N correct, slope
not. Gate-A band reads roughly ~560-3,200 GiB. **No verdict moves.**

## Confidence ledger

✓ VERIFIED by reading: round structure and R1 barrier; AirTracePair signature + both
producers; the Lde doc; auto_storage's split; all spill call sites; no mmap on
LDETraceTable; every StorageMode argument at every multi_prove call site;
allocate_aux_table; Table::columns() full copy; verifier transcript order; KECCAK_RND
1480 cols (`tables/keccak_rnd.rs:95`), 1031 interactions → 516 aux (`lookup.rs:117-125`).
? INFERRED: all GiB arithmetic; +40-60% wall; cuda VRAM note. ✗ NOT CHECKED: other 13
chips' rows (totals are the chunk family's contribution); VramGate interaction.
