---
name: optimize-prover
description: Iterative performance optimization loop for the Lambda VM STARK prover. Profile, identify bottlenecks, implement ONE fix at a time, measure, keep or revert.
user_invocable: true
keywords: [optimize, performance, profile, instruments, bottleneck, perf, benchmark, prover, stark]
---

# Optimize Prover — Iterative Performance Loop

Fully autonomous performance optimization agent for the Lambda VM STARK prover. Runs a disciplined loop: **measure → identify → implement ONE change → measure → verify → keep or abandon**.

Zero external input expected. The user may interrupt at any time.

## Known Bottlenecks (input for Step 1)

We already have a snapshot of where Lambda loses to Plonky3 upstream from
the `bench_vs_plonky3` benchmark (see
[`lambda_vs_p3_port.md`](lambda_vs_p3_port.md) and the raw TSVs in
[`raw_data/`](raw_data/)). The loop **starts from this prioritized list
instead of searching blindly through instruments output**.

Measurement: 4 breakdowns at log=21 with num-sequences ∈ {16, 32, 64}
(32/64/128 cols) + log=19 at 32 cols. EPYC 48-core server, `--scalar`,
10 runs per point, CVs <2% at log≥20.

### Lambda bottleneck table (log=21)

| Lambda phase                 | % prove_total @ 32c | Scaling 32→128c   | L/P3 @ 32c | L/P3 @ 128c | Verdict |
|------------------------------|--------------------:|:-----------------:|----------:|------------:|---------|
| `r2_constraints`             | 25%                 | **×3.42**         | 2.69×     | **8.61×**   | 🔴 **Bottleneck #1** — breaks worst with cols |
| `r1_main_lde`                | 19%                 | **×4.00**         | (part of LDE+Merkle 1.87× → 2.32×)         |             | 🔴 **Bottleneck #2** — worst absolute scaling |
| `r1_main_merkle`             |  9%                 | ×3.29             | (paired with #2)         |             | 🟠 paired with #2 |
| `r3_ood + r4_deep_*`         | 21%                 | ×1.23             | ~2.38×    | ~2.5×       | 🟡 scales OK, medium gap |
| `r4_fri_commit`              | 17%                 | ×0.91 (decreases) | 1.99×     | 1.88×       | 🟢 invariant w/ cols; `PLAN-fri-optimizations.md` targets this |
| `r2_comp_commit` (quotient)  |  3%                 | ×1.00             | **0.41×** | **0.50×**   | 🟢 **Lambda wins — DO NOT touch** |
| `prepass`                    |  3%                 | ×1.00             | ∞ (P3 has no equivalent) |             | 🟢 negligible |

### Prioritized candidate list

Recommended order of attack:

1. **Constraint eval** (`r2_constraints`) — highest payoff as cols grow.
   Lambda evaluates on the full LDE (`2N`), P3 on the quotient domain
   (`N` when `d_max=1`). Approaches:
   - Quotient-domain eval for `d_max=1` AIRs (bench-only win, does not
     translate to production with Keccak `d_max=3` — see "Previously Tried")
   - Some kind of column-subset / batched evaluation
2. **Trace LDE + Merkle** — worst absolute scaling with cols (×4 going to
   128 cols). P3 uses batched `coset_lde_batch`; Lambda does per-column
   iFFT+FFT. Attack with batched FFT.
3. **FRI commit** — invariant with cols but constant 1.99× gap. The plan
   `PLAN-fri-optimizations.md` (repo root) proposes early-stop +
   folding=4: ~165 ms savings at log=21, universal (does not depend on
   `d_max`).
4. **DEEP + OOD** — smaller payoff (×1.23 with cols, gap ~2.4×). Leave
   for later.
5. **Do not touch**: `r2_comp_commit` (Lambda wins 0.41×), `prepass`,
   `r4_queries`.

### When to fall back to blind mode

If after several attempts you exhaust the prioritized list (all abandoned
or already shipped), fall back to the original Step 1 mode: measure with
`bench_prove.sh --instruments` and pick the phase with the largest
absolute time. But **always start from this list** to avoid re-discovering
what bench_vs_p3 already tells us.

## Invocation

```plain
/optimize-prover [description]
```

- `description`: optional focus area (e.g., "FFT performance", "Merkle hashing")

## Benchmark Programs

Available ELFs in `executor/program_artifacts/asm/`:

| Program                  | Use                                         |
|--------------------------|---------------------------------------------|
| `fib_iterative_500k.elf` | Quick intuition runs (~fast)                |
| `fib_iterative_8M.elf`   | **Real benchmark** (use this for decisions) |

## Measurement Strategy

**Always benchmark with `TABLE_PARALLELISM=1`.** This gives sequential table processing (lower memory pressure), and optimizations to per-table parallelism (rayon work within each table) are more impactful than increasing table-level parallelism. All measurements must use the same setting for comparability.

### Primary tool: `scripts/bench_prove.sh`

This script handles building, running, and collecting wall-clock time + peak heap (jemalloc). Pass `--instruments` to also capture detailed phase breakdown.

```bash
# Quick intuition (1 run, no comparison with main):
TABLE_PARALLELISM=1 scripts/bench_prove.sh executor/program_artifacts/asm/fib_iterative_500k.elf 1 --no-compare

# Real benchmark (3+ runs, no comparison with main):
TABLE_PARALLELISM=1 scripts/bench_prove.sh executor/program_artifacts/asm/fib_iterative_8M.elf 3 --no-compare

# With instruments breakdown (adds a single instrumented run after the bench):
TABLE_PARALLELISM=1 scripts/bench_prove.sh executor/program_artifacts/asm/fib_iterative_8M.elf 3 --no-compare --instruments
```

When `--instruments` is passed, the script builds with `--features instruments`, runs one instrumented prove, and saves the report to `/tmp/bench_prove/instruments.txt`. Copy this file with a descriptive name for comparison:

```bash
cp /tmp/bench_prove/instruments.txt /tmp/instruments_<descriptive-name>.txt
```

Use names that describe the optimization: `instruments_baseline.txt`, `instruments_before_fft_batching.txt`, `instruments_after_fft_batching.txt`, `instruments_before_merkle_parallel.txt`.

**Rules:**

- Always use `--no-compare` (we manage our own baseline, don't checkout main)
- Minimum 3 samples for any keep/abandon decision
- If variance between samples is >5%, increase to 5 samples
- Use 500k for quick smoke tests during development
- Use 8M with 3+ samples for the actual before/after comparison

Parse `=== PROVER TIMING ===` for phase timings, sub-ops, per-table breakdown, FFT/Merkle totals.

### Secondary bench: `bench_vs_plonky3` (Lambda vs Plonky3 upstream)

In addition to the VM-level bench above, we have a STARK-level bench that
isolates the prover from VM concerns (no aux trace, no logup, no
multi-table) and compares Lambda directly against Plonky3 upstream on the
same Fibonacci AIR. Use it for fine-grained validation that a fix moves
the right phase.

```bash
# On the bench server (vm-benchmarks-1):
cd ~/juan/lambda_vm
./bench_vs_plonky3/run.sh \
    --scalar --breakdown \
    --log-rows 21 --num-sequences 16 --runs 10 \
    --report-dir /home/app/juan/lambda_vm/bench_vs_p3_<descriptive-name>_$(date +%Y%m%d_%H%M)
```

Output goes to `<report-dir>/breakdown.tsv` (per-phase + P3 span timings)
and `<report-dir>/results.tsv` (totals).

**When to use which bench:**

| Change touches… | Use `bench_prove.sh` (VM, fib_iterative_8M) | Use `bench_vs_plonky3/run.sh` (STARK fib_pair) |
|---|:-:|:-:|
| Trace building, multi-table, logup, aux trace | ✅ required | ❌ not measured here |
| FFT, Merkle, constraint eval, FRI (core STARK) | ✅ end-to-end signal | ✅ fine-grained per-phase signal |
| New AIR / new chip | ✅ required | ❌ unrelated |

For FFT / Merkle / constraint eval / FRI optimizations, **use both**:
`bench_vs_plonky3` confirms the targeted phase moved; `bench_prove.sh`
confirms the gain translates to the real VM. The runtime is ~10 min for
bench_vs_p3 (server) vs ~minutes for bench_prove.sh (local).

Reference snapshot of where the gap is per-phase:
[`lambda_vs_p3_port.md`](lambda_vs_p3_port.md). Raw per-run timings:
[`raw_data/`](raw_data/).

## Scope Rule

**Optimizations must be transparent to the ORIGINAL verifier.**

### ALLOWED (prover infrastructure)

- `crypto/stark/src/prover.rs` — scheduling, batching, parallelism, memory reuse
- `crypto/math/src/fft/` — faster algorithms, cache usage, same mathematical result
- `crypto/crypto/src/merkle_tree/` — construction speed, hashing, parallelism
- `crypto/stark/src/fri/` — commitment phase, query phase, evaluation order
- `crypto/math/` — field arithmetic, polynomial operations
- `prover/src/` — trace building, table construction
- Memory layout, allocation patterns, pool reuse
- Parallelism: rayon strategies, chunk sizes, work distribution

### FORBIDDEN (protocol and verification)

- `crypto/stark/src/verifier.rs` — never touch
- AIR/Constraint definitions
- `crypto/stark/src/proof/` — proof structure
- Protocol: rounds, commitments, challenge sampling, Fiat-Shamir transcript
- Table AIR trait implementations

## Branch Strategy

Each optimization lives on its own branch. **Never merge into base.** Branches are historical records of what was tried and what worked.

1. Start from the **base branch** (the branch active when `/optimize-prover` is invoked).
2. For each optimization: `git checkout -b opt/<N>-<descriptive-name>` from base.
3. Implement and commit on that branch.
4. Record the result (kept/abandoned) in the report.
5. `git checkout <base>` — always return to base regardless of outcome.
6. Next optimization: create a new branch from the (unchanged) base.

## Report File

Write a live status report to `optimizer/optimize_report.md`. This file:

- Is NOT committed anywhere
- Is updated after each iteration
- Contains the full history of attempts + current status
- The user can read it at any time to check progress or decide to interrupt

Format:

```markdown
# Optimization Report
Base branch: <name>
Benchmark: fib_iterative_8M (3 samples)
Started: <timestamp>

## Baseline
bench_prove.sh median: X.Xs, mean: X.Xs, heap: X MB
Instruments breakdown:
<paste key instruments lines>

## Attempt 1: <description>
Branch: opt/1-<name>
Status: KEPT / ABANDONED
bench_prove.sh (8M, 3 samples): X.Xs → Y.Ys (±Z%)
Heap: X MB → Y MB (±Z%)
Verification: PASS / FAIL
Instruments delta: <key changes>
Notes: <observations>

## Attempt 2: ...

## Current State
Total improvement so far: X.Xs → Y.Ys (±Z%)
Next target: <what to optimize next>
```

## Previously Tried (Do Not Retry)

History of attempts on this prover. Skip these during Step 1.5 and Step 2 —
don't re-research, re-benchmark, or re-invent.

### Targeted attempts

| Attempt | When | Outcome | Notes |
|---|---|---|---|
| Quotient-domain constraint eval (`feat/eval-form-quotient`) | 2026-05-04 | **PAUSED** | Bench-only win (~50% saving on `r2_constraints` for `d_max=1` AIRs like fib_pair). Does NOT translate to production with Keccak (`d_max=3`): Lambda's LDE 2N is insufficient for the composition poly of degree ~2N. Would require blowup ≥ 4 across the board, raising other phases. 5-commit chain with batched-Merkle deps. |
| Parallel FRI fold (PR #448) | 2026-04-08 | ABANDONED | No improvement. |
| Sequential FFT + Truncated FFT | 2026-03-31 | DISCARDED | No improvement. |
| SIMD / PackedField FFT | 2026-03-26 | FAILED | Implementation never beat scalar. |
| Execution sharding (parallel chunk proving) | 2026-04-08 | ABANDONED | Architectural mismatch — transcript fork is the bottleneck. |
| Plonky3 upstream migration (commit `fb20f767`) | 2026-05-13 | COMPLETED | Bench moved from `yetanotherco/Plonky3#feat/goldilocks_deg3` to upstream. P3 paid +5–10% at small logs (trinomial mul); Lambda unchanged. |

### Pending / not yet attempted

- **FRI optimizations** (`PLAN-fri-optimizations.md` in repo root): early-stop
  + folding=4. Expected savings ~165 ms on `r4_fri_commit` at log=21,
  universal (any AIR, any blowup). Localized to `crypto/stark/src/fri/`.
- **Batched FFT for trace LDE** (would mirror P3's `coset_lde_batch`):
  Lambda currently does per-column iFFT+FFT; P3 batches across columns.
  Target the worst-scaling phase with cols (`r1_main_lde` ×4.00 going to
  128 cols).

### Notes for the next run

- Apple Silicon was compute-bound on FFT/Merkle — memory-layout tricks didn't help. x86 CI may behave differently.
- Nested rayon inside FFT columns is load-bearing; don't remove it.
- Merkle hashing is Keccak-bound, not allocation-bound (#503 changed the allocation pattern, not the allocation count).
- Avoid architecture-dependent optimizations that may work on some machines but not on others. For example, compiling the VM with a specific PGO.
- NEON base-field mul on aarch64 is SLOWER than scalar (~0.92×) for
  Goldilocks. Only Fp3 NEON helps (~1.40×). Profiling 2026-04-22.

## The Loop

### Step 0: Setup (once)

1. Record the current branch as the **base branch**.

2. **Detect the environment** and pick the right benchmark size. Run this
   once and export the variables for the rest of the session:

    ```bash
    TOTAL_RAM_GB=$(free -g | awk '/^Mem:/{print $2}')
    FREE_RAM_GB=$(free -g | awk '/^Mem:/{print $7}')
    N_CORES=$(nproc)
    HOSTNAME=$(hostname)

    echo "Env: $HOSTNAME — $N_CORES cores, $TOTAL_RAM_GB GiB RAM ($FREE_RAM_GB free)"

    if [ "$FREE_RAM_GB" -lt 32 ]; then
        # Small server (Ralph-class: Ryzen 5 PRO 3600, 31 GiB)
        export PRIMARY_ELF=executor/program_artifacts/asm/fib_iterative_500k.elf
        export SMOKE_ELF=executor/program_artifacts/asm/fib_iterative_500k.elf
        export BENCH_VS_P3_MAX_LOG=21
        export BENCH_VS_P3_MAX_N=64
        export ENV_PROFILE=small
        echo "→ small env: 500k.elf for real bench, bench_vs_p3 capped at log=21, n≤64"
    else
        # Big server (EPYC 48-core, 125 GiB)
        export PRIMARY_ELF=executor/program_artifacts/asm/fib_iterative_8M.elf
        export SMOKE_ELF=executor/program_artifacts/asm/fib_iterative_500k.elf
        export BENCH_VS_P3_MAX_LOG=23
        export BENCH_VS_P3_MAX_N=64
        export ENV_PROFILE=big
        echo "→ big env: 8M.elf for real bench, full bench_vs_p3 sweep available"
    fi
    ```

    Record `HOSTNAME`, `ENV_PROFILE`, `PRIMARY_ELF`, and limits at the top
    of the report file so future readers know which benchmarks were used.

    **Hard guards** (the loop must NOT cross these on small envs):
    - On `ENV_PROFILE=small`: do not invoke `fib_iterative_8M.elf` under
      any circumstance — it OOMs / swaps to HDD which invalidates timing.
    - On `ENV_PROFILE=small`: cap bench_vs_p3 at `--log-rows
      $BENCH_VS_P3_MAX_LOG --num-sequences $BENCH_VS_P3_MAX_N`.
    - After every prove run, check `dmesg --since "1 minute ago" 2>/dev/null
      | grep -i oom`. If present, the run is invalid — discard and rerun
      with a smaller workload.

3. Build the baseline verifier binary (original protocol, before any changes):

    ```bash
    cargo build --release -p cli
    cp target/release/cli /tmp/cli_baseline_verifier
    ```

4. **Run baseline benchmarks on the current branch** using `$PRIMARY_ELF`
   (which was set by Step 0.2 to either `fib_iterative_8M.elf` on big
   servers or `fib_iterative_500k.elf` on small servers):

    ```bash
    TABLE_PARALLELISM=1 scripts/bench_prove.sh "$PRIMARY_ELF" 10 --no-compare --instruments
    cp /tmp/bench_prove/instruments.txt /tmp/instruments_baseline.txt
    ```

    10 samples gives a tight baseline CV. Record the baseline median,
    mean, CV, and heap in the report. All future comparisons are against
    this baseline. Note: on small servers the 500k workload has noisier CV
    than the 8M would (the smaller the workload, the higher relative jitter).

5. Initialize the report file at `optimizer/optimize_report.md` with
   baseline data and the environment profile from Step 0.2.

### Step 1: Identify ONE Bottleneck

**Start from the prioritized list in "Known Bottlenecks (input for Step 1)"
above.** Pick the next unattempted candidate in order: constraint eval →
trace LDE+Merkle → FRI commit → DEEP+OOD. This list is grounded in the
`bench_vs_plonky3` measurements (4 breakdowns × {32, 64, 128} cols at
log=21), so we know exactly which phases matter most and how they scale.

For the picked bottleneck, drill in with `bench_prove.sh --instruments` to
identify sub-phases worth attacking. Map the high-level phase to the
sub-ops:

- Constraint eval → `r2_constraints` plus per-table eval cost
- Trace LDE+Merkle → `r1_main_lde` (iFFT + coset eval) + `r1_main_merkle`
- FRI commit → `r4_fri_commit` (folding + per-layer Merkle commits)
- DEEP+OOD → `r3_ood` + `r4_deep_comp` + `r4_deep_extend`

**Fallback (blind mode)**: only after the prioritized list is exhausted
(everything abandoned or already shipped), revert to the original
heuristic: pick the single largest opportunity from instruments by
absolute time. Common patterns in that case:

- FFT dominates → `expand_pool_to_lde`, cache layout, batched FFT
- Merkle dominates → hash function, tree parallelism
- One table dominates R2-4 → constraint evaluation
- Aux build slow → batch_inverse, fingerprint
- Pre-pass slow → twiddle computation

### Step 1.5: Check Existing PRs (dedup)

First consult the "Previously Tried (Do Not Retry)" section above. Then run this to check for overlap with other in-flight PRs:

```bash
gh pr list --state all --limit 50 --json number,title,headRefName,state | jq -r '.[] | select(.headRefName | test("opt|perf|fft|merkle|bench|fast"; "i")) | "[\(.state)] #\(.number) \(.title)"'
```

Check `/bench` comments on relevant PRs for benchmark results. Rules:

- If a **focused PR** already benchmarks this exact optimization → skip it, note in report, move to next bottleneck
- If a PR is a **bundle of multiple changes** but one overlaps → you can still try the standalone optimization (isolating just that one change is valuable)
- Record any relevant PR numbers and their results in the report for reference

### Step 2: Research (parallel agents)

Spawn two teammates in parallel to explore optimization approaches:

**Agent 1 — "researcher"** (scout):

- Search `others/plonky3/` for how it handles the identified bottleneck
- Look at the equivalent code paths (FFT, Merkle, parallelism, memory layout, etc.)
- Report back: the approach used, key code snippets, relevant file paths

**Agent 2 — "creative"** (scout):

- Given only the bottleneck description and the lambda_vm code, independently brainstorm an optimization
- Do NOT look at reference codebases — work from first principles and general systems/perf knowledge
- Report back: proposed approach, which files to change, expected impact

Wait for both to complete. Collect all ideas into a ranked list. Each idea becomes its own attempt — tried sequentially, one at a time, always from the unchanged base branch.

**CRITICAL: ONE idea = ONE branch. Never combine multiple ideas into a single branch.** This keeps changes minimal and isolated. If idea A is good but idea B is bad, you don't want B polluting A's branch.

### Steps 3–7: Try Each Idea (inner loop)

For each idea from Step 2, repeat this sequence:

#### Step 3: Branch and Implement

```bash
git checkout <base>
git checkout -b opt/<N>-<descriptive-name>
```

Implement **only this one idea**. Keep it minimal — smallest possible change to test the hypothesis.

Build and sanity-check:

```bash
cargo build --release -p cli
cargo test --release -p stark
cargo test --release -p lambda-vm-prover
```

Commit the change on the branch.

#### Step 4: Measure

Quick smoke test first (500k, 1 run — always the 500k regardless of env):

```bash
TABLE_PARALLELISM=1 scripts/bench_prove.sh "$SMOKE_ELF" 1 --no-compare
```

If it looks promising, run the real benchmark with instruments using
`$PRIMARY_ELF` (set by Step 0.2):

```bash
TABLE_PARALLELISM=1 scripts/bench_prove.sh "$PRIMARY_ELF" 3 --no-compare --instruments
cp /tmp/bench_prove/instruments.txt /tmp/instruments_after_<descriptive-name>.txt
```

**OOM check after every run** (especially on `ENV_PROFILE=small`):

```bash
dmesg --since "1 minute ago" 2>/dev/null | grep -i "out of memory\|oom-kill" \
    && echo "WARNING: OOM detected — measurement invalid, rerun with smaller workload"
```

If OOM was triggered, the timing is invalid (the kernel paged out the
process); discard the run, drop to a smaller ELF or smaller bench_vs_p3
config, and try again.

#### Step 5: Verify

`scripts/bench_prove.sh` deletes its proofs. Produce a fresh one with the
current branch's cli (using the same `$PRIMARY_ELF` from Step 0.2), then
verify with the baseline binary:

```bash
target/release/cli prove "$PRIMARY_ELF" -o /tmp/proof_after.bin
/tmp/cli_baseline_verifier verify /tmp/proof_after.bin "$PRIMARY_ELF"
```

#### Step 6: Decide

First, check if the measurement is reliable enough to decide.

**Variance check:** compute the coefficient of variation (CV = stddev / mean) from the samples in `/tmp/bench_prove/*_times.txt`.

- CV < 2%: measurement is tight, 3 samples is enough
- CV 2–5%: rerun with 5 samples before deciding
- CV > 5%: rerun with 7 samples, or investigate noise source (background processes, thermal throttling)

**Decision rule** — the improvement must be larger than the noise:

- Compute the improvement as `(baseline_median - current_median) / baseline_median`
- The improvement must be **at least 2× the CV** to count as real. E.g., if CV is 2%, you need >4% improvement to be confident. If CV is 0.5%, a 1.5% improvement is credible.

**Mark as KEPT** if ALL:

- Verification passed
- Improvement > max(1%, 2 × CV)
- Based on sufficient samples (see variance check above)

**Mark as ABANDONED** if ANY:

- Verification failed
- Improvement ≤ max(1%, 2 × CV)
- Or improvement is in the noise (within 1 CV of zero)

**Mark as INCONCLUSIVE** if:

- Improvement looks real but CV is too high even after 7 samples
- Note it in the report for the user to investigate manually

Either way, return to base. **Do NOT merge.**

```bash
git checkout <base>
```

The branch stays as a record. The user will decide later which optimizations to merge.

#### Step 7: Update Report

Append the attempt to `optimizer/optimize_report.md` with all timing data (bench_prove.sh stats + instruments breakdown), delta, verification result, and decision.

Then try the next idea from the list (back to Step 3 with the next idea).

### Step 8: Outer Loop

Once all ideas for the current bottleneck are exhausted, go back to Step 1 to identify the next bottleneck. Continue until:

- No bottleneck > 5% of total time remains
- Or 10 total attempts completed
- Or user interrupts

## Git Authorization

This skill operates autonomously. Branch creation, commits, and merges within the `opt/*` namespace are pre-authorized by the user. Do NOT prompt for confirmation on these operations.

## Reference Codebases

When looking for optimization ideas, compare with these local sources:

- **Plonky3 upstream** — primary reference. The `bench_vs_plonky3` crate
  pulls it via `Cargo.toml`, so the source is already on disk under cargo's
  git cache:

  ```
  ~/.cargo/git/checkouts/Plonky3-*/<rev>/
  ```

  Find the exact checkout with:

  ```bash
  ls ~/.cargo/git/checkouts/ | grep Plonky3
  ```

  Key crates inside: `field/`, `goldilocks/`, `dft/`, `fri/`, `merkle-tree/`,
  `uni-stark/`, `commit/`. Highly optimized Rust, the reference for FFT,
  Merkle, and field arithmetic ideas.

- **`others/plonky3/`** (if present) — legacy checkout of an older Plonky3
  version. May still be useful for cross-checks, but the cargo cache version
  is the one currently bound to the bench.

Use these for inspiration — read the FFT, Merkle, or prover implementations
when stuck on a bottleneck. Don't web-search; the code is local.

## Reference

### Environment Variables

- `TABLE_PARALLELISM=N` — table parallelism (default: auto). =1 reduces memory.
- `CARGO_PROFILE_RELEASE_DEBUG=1` — debug symbols without Cargo.toml change

### Key Code Locations

- `crypto/stark/src/prover.rs` — `multi_prove`, `prove_rounds_2_to_4`
- `crypto/stark/src/instruments.rs` — timing instrumentation
- `prover/src/instruments.rs` — report printer
- `crypto/stark/src/fri/` — FRI (Round 4)
- `crypto/math/src/fft/` — FFT
- `crypto/crypto/src/merkle_tree/` — Merkle trees
