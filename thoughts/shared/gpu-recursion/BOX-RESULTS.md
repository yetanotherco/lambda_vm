# GPU box measurements — EXPLORATION.md Stages 0/1 (2026-08-12)

Box: a rented vast.ai instance (endpoint in the session notes, not committed here).
RTX 5090 32 GiB (GPU-8798ec09, driver 595.71.05, compute cap 12.0), CUDA toolkit **13.1**,
cgroup quota **30.7 cores** (nproc 32 — whole machine, not a fractional slice), 93 GiB RAM,
64 GiB disk. Code: `blake3-real-hash` @ `77cc7df1` (PR #930 tip + handoff docs commit),
shipped as a git bundle (no credentials on the box), toolchain 1.94.0 +
nightly-2026-02-01 for guest ELFs.

## Health gates (Stage 0 step 0) — GREEN

- `make test-math-cuda`: 87 tests pass on device. EXIT=0.
- `make test-cuda-integration`: 7/7 pass in 13.3 s — the R1-R4 dispatch counters all fire
  on a real RV64 prove and the proof verifies. The stack works on CUDA 13.1 + sm_120.

## Stage 0 — baseline wrap counters (default threshold 2^19)

New harness: `lfm::wrap_tests::the_wrap_reports_gpu_counters` (`#[cfg(feature = "cuda")]`,
`#[ignore]`d) — resets the process-global counters AFTER the inner epoch is built (the inner
RV64 continuation prove has its own GPU traffic) and prints all 15 counters after `lfm_prove`.

### ★ Found the root cause of the "19 pre-existing fibonacci.elf failures"

A fresh `make compile-recursion-elfs` builds a `recursion/fibonacci.elf` of **1,344 bytes**
that **finishes in ≤16 cycles** → `real_epoch_with` panics `"wanted an INTERMEDIATE epoch"`
(`epoch_tests.rs:659`). The fixture premise (`proof_fixture.rs:41-49`: guest runs 17–64
cycles, splits only at a 16-cycle epoch) was measured against an older build — the ELF in
`lambda_vm_2`/`lambda_vm_3` worktrees (**1,368 bytes**, Jul 21, sha `4346975f…`) still works.
Codegen drift shaved ~6 instructions and broke the split. This is why the whole
fixture-dependent lfm suite reports 19 failures on any machine that rebuilds the ELF.
Workaround on the box: shipped the Jul-21 ELF. Real fix is Mauro's call (re-measure
`FIXTURE_EPOCH_LOG2`, or pin the fixture guest against drift).

RESULT (chip log-heights `[11, 21, 17, 11, 15, 2, 12, 16, 15, 8, 16, 0, 5, 20]` — BALU 2^21,
BITWISE 2^20, KECCAK_RND 2^17, matches the census):

```
lfm_prove 16.8 s, peak VRAM 5,789 MiB
lde 470 / leaf_hash 13 / merkle_tree 13 / extend_halves 2 / logup 10
composition 2 / comp_poly_tree 2 / parts_lde 0 / bary 20 / deep 2
batch_invert 8 / fri 2 / opening_gather 12 / device_only 0
```

**All four Stage 0 falsifiable predictions CONFIRMED**: composition ≥ 2 (=2, BITWISE+BALU),
merkle_tree ≥ 4 cold (=13), device_only == 0 (threshold-caused), lde far below what
KECCAK_RND's 1,480 columns would add. §1's map survives falsification.

## Stage 1(a) — LAMBDA_VM_GPU_LDE_THRESHOLD sweep

| threshold | device_only | composition | lde | prove s (runs) | peak VRAM | verify |
|---|---|---|---|---|---|---|
| default 2^19 | 0 | 2 | 470 | **16.8, 16.7, 16.7** | 5.8 GiB | ✓ |
| 262144 (2^18) | **1** | 4 | 3533 | **7.2, 7.1, 7.2** | 12.7 GiB | ✓ |
| 131072 (2^17) | 1 | 6 | 3557 | 7.2 | 12.9 GiB | ✓ |
| 4096 (2^12) | 1 | 11 | 4601 | 7.1 | 11.3 GiB | ✓ |

★ **LEVER 1 CONFIRMED, ABBA-tight (A-B-B-A run order): −57% prove time (16.73 → 7.17 s
mean, sd ≈ 0.05 s), zero code changed.** `LAMBDA_VM_GPU_LDE_THRESHOLD=262144` flips
`device_only` 0→1 — KECCAK_RND (88.1% of main cells, the one non-preprocessed chip) sails
through `device_only_gate` once past the size gate; its whole R1→FRI pipeline moves
on-device (`fri` 2→4, `merkle_tree` 13→17, `deep` 2→4) and the proof verifies.

**The knee is exactly 2^18**: thresholds 2^17 and 2^12 stay at ~7.2 s — KECCAK_RND is the
entire win, and admitting every remaining chip (composition 2→11 across the sweep) neither
helps nor hurts at this scale. `device_only` is pinned at 1 at EVERY threshold: the other
13 chips are preprocessed and excluded by `&& !is_preprocessed` — that ceiling is lever 2.

Attribution note (open): `composition` went +2 at 2^18 though only one chip (KECCAK_RND) newly
clears the ROW gate — the R2 fused path's own gate admits by a different rule than R1's
(sweep: 2→4→6→11). Doesn't affect the conclusion; worth settling when writing the permanent gate.

## Stage 1(b) — GPU composition A/B at threshold 2^18

`LAMBDA_VM_DISABLE_GPU_COMPOSITION=1`: **13.8 s vs 7.2 s — disabling it nearly doubles prove
time.** EXPLORATION's prediction ("small, per the VM's −2.7%") is **falsified**: on the LFM
machine the fused composition path is a co-headline win, which is what you'd expect from
16k-IR-node × 1,480-column constraint programs. Note `composition 0 / device_only 0` in that
run — `device_only_gate` requires `!gpu_composition_disabled()`, so the kill switch also
demotes KECCAK_RND to host copies, yet 13.8 s still beats the 16.8 s baseline (the GPU
LDE + trees keep helping).

## Stage 1(c) — TABLE_PARALLELISM at threshold 2^18

tp=1: 8.4 s / tp=4: 7.1 s / tp=14: 7.2 s / default (cores·2/3): 7.2 s.
Saturates at ≥4; even fully serial costs only +1.2 s. No lever here; default is fine.

## Stage 3 correction — the cell-aware gate is BIGGER than EXPLORATION §4 scoped

EXPLORATION said "3 sites + the `device_only_gate` mirror". The audit says otherwise:
`gpu_lde_threshold()` has **18 consumer sites** across R1/R2/R3/R4/FRI, and several
re-derive admission downstream even when they already hold the device handle
(e.g. `gpu_lde.rs:1170` checks `handle.lde_size < gpu_lde_threshold()`). A cell-admitted
table (KECCAK_RND: LDE 2^18 < row default 2^19) would pass R1 and then be REFUSED by
downstream row-checks — for a device-only table that is the documented LOCKSTEP hazard
(`gpu_lde.rs:185-188`): hard abort, or a silent fallback that forfeits the win. FRI folds
are width-1, so a naive `lde_size × m` rule degenerates to the row rule exactly where the
device-only pipeline must keep firing.

The right permanent shape is **admission decided once at R1, device-handle presence as the
admission token downstream** — handle-bearing sites stop re-checking the size threshold.
That is a proper reviewed PR (~6-10 sites + the LOCKSTEP audit), with this box's counter
test as the oracle (device_only=1, ~7.2 s, verify green at DEFAULT env). NOT rushed here.

**Operational recommendation until then:** run wrap proves with
`LAMBDA_VM_GPU_LDE_THRESHOLD=262144`. That exact configuration is what was proved end to
end here, ABBA-tight, verify green, 12.7 GiB peak VRAM on a 32 GiB card. (The env var is
process-global: in the wrap test process it also lowers the gate for the inner RV64 epoch
prove, which the t4096 run shows is harmless at this scale.)

## Evidence

Raw logs + 1 Hz VRAM traces: `~/workspace/lambda_vm_bench_cache/gpu_lfm_wrap_2026-08-12/`
(26 files). Counter harness committed on `blake3-real-hash` as `c495e9fc` (signed, unpushed).
