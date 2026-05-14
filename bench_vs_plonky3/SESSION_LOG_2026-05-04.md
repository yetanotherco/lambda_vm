# Session Log — 2026-05-04

> Branch: `bench_vs_p3_improvements` (off `bench_vs_p3` after PR-style merge with main)
> Server: `vm-benchmarks-1` (AMD EPYC 9454P, 48 cores)
> Goal: improve Lambda STARK prover to close gap vs Plonky3.

---

## TL;DR (what happened today)

1. **Fixed post-merge build break**: trait migration `TransitionConstraint` →
   `TransitionConstraintEvaluator` in `bench_vs_plonky3/src/lambda_fibonacci_pair.rs`
   after main introduced a new evaluator API.
2. **Hardened the bench harness**: added proof_size, verify_s, peak_rss,
   throughput metrics. Default runs 3 → 10 with CV reporting. Added
   `--breakdown` mode for per-phase timing. New `breakdown.tsv` artifact.
3. **Wrote `METHODOLOGY.md`**: full doc explaining what/how the bench
   measures + comparison vs P3/SP1/RISC0/OpenVM + 20 improvement options.
4. **Ran baselines on the server**: 3 sizes × 10 runs with full metrics.
5. **Ran phase breakdowns**: log=19 and log=21 with `--breakdown`.
6. **Ran column sweep**: confirmed L/P3 ratio degrades linearly with #cols
   (1.46× → 2.64× from 8 → 128 columns).
7. **Designed plan for quotient-domain constraint eval**, then **stepped
   back after theoretical analysis** showed the win is bench-specific and
   does not translate to Lambda VM real (degree-3 AIRs).
8. **Recommendation locked in**: FRI early-stop + folding=4 is the better
   target. Plan already written in `PLAN-fri-optimizations.md`.

---

## Commits made today

- `4b19d250` — Migrate FibonacciPair AIR constraints (trait migration after
  main merge).
- `7ee4cbf0` — Add full metrics and phase breakdown to bench_vs_plonky3.

Both committed to `bench_vs_p3_improvements`.

## Untracked stuff in workdir (NOT committed, on purpose or pending)

- `bench_vs_plonky3/METHODOLOGY.md` — methodology doc (new today).
- `bench_vs_plonky3/ANALYSIS_LOG.md` — preexisting analysis log.
- `bench_vs_plonky3/inform.md` — preexisting parallelism scaling report.
- `bench_vs_plonky3/SESSION_LOG_2026-05-04.md` — this file.
- `Cargo.lock`, `bench_vs_plonky3/README.md` — modified, NOT staged
  (user explicitly opted out: ".lock no va, los .md tampoco").
- Top-level `.md` files (`AUDIT-memw-register.md`, `CLAUDE.md`,
  `KECCAK_PRECOMPILE_AUDIT.md`, `PLAN-fri-optimizations.md`,
  `optimizations_3mi.md`, `blog-radix8-butterfly-fusion.md`) — separate
  workstreams, not touched.
- `.claude/` — local config.

---

## What we changed in code

### `bench_vs_plonky3/src/lambda_fibonacci_pair.rs`

- `impl TransitionConstraint` → `impl TransitionConstraintEvaluator` for
  both `FibPairShiftConstraint` and `FibPairSumConstraint`.
- `fn evaluate(eval_ctx, out)` → `fn evaluate_verifier(eval_ctx, out)`
  (same body, just renamed).
- Storage type: `Vec<Box<dyn TransitionConstraint<F,E>>>` →
  `Vec<Box<dyn TransitionConstraintEvaluator<F,E>>>`.
- Added `fn name(&self) -> &str { "fib_pair" }` so breakdown.tsv shows
  `table=fib_pair` instead of `table=unknown`.
- (Linter also added `#[serde(bound = "")]` to `FibonacciPairPublicInputs`
  and serde derives — not by us, by a tool.)

### `bench_vs_plonky3/src/bin/prove_bench.rs` (the other AI's work + my fix)

- New METRICS line emitted per run with: `prove_s, verify_s,
  proof_size_bytes, peak_rss_kb, rows_per_sec, cells_per_sec`.
- `getrusage(RUSAGE_SELF)` for peak RSS.
- `serde_cbor` for proof size measurement.
- `--blowup`, `--queries`, `--grinding` now respected on P3 side too.
- `--breakdown` flag dumps Lambda phases + P3 tracing spans.
- My fix: type alias `P3SpanResults = Arc<Mutex<Vec<(String, f64)>>>` to
  silence `clippy::type_complexity` (was breaking `make lint`).

### `bench_vs_plonky3/run.sh`

- Default `RUNS=10` (was 3).
- `--breakdown` flag rebuilds with `--features instruments`.
- New artifacts: `results.tsv`, `raw_metrics.tsv`, `breakdown.tsv`,
  `metrics.txt`, `raw/`.
- CV computation alongside median.

### `bench_vs_plonky3/src/plonky3_config.rs`

- Now accepts blowup, queries, grinding from caller (was hardcoded).

### `.github/workflows/bench-vs-p3-nightly.yml`

- Nightly switched from `--runs 3` to `--runs 10`.

### Validation done

- `cargo check -p bench-vs-plonky3 --bin prove_bench` ✅
- `cargo check -p bench-vs-plonky3 --features instruments --bin prove_bench` ✅
- `cargo test -p bench-vs-plonky3 --lib` ✅ (3 passed, 1 ignored)
- `make lint` ✅ (after my type-alias fix)
- Smoke run of `run.sh --breakdown` on log=12 ✅
- Smoke run of `run.sh` baseline on log=12 ✅

---

## Bench results (server, EPYC 48-core, --scalar)

### Baseline — 3 sizes × 10 runs (`bench_vs_p3_baseline_<date>/`)

| log-rows | rows  | L prove | L CV  | P3 prove | P3 CV | L/P3 | L verify | P3 verify | L proof | P3 proof | L RSS  | P3 RSS |
|----------|-------|---------|-------|----------|-------|------|----------|-----------|---------|----------|--------|--------|
| 17       | 131k  | 0.254s  | 7.96% | 0.137s   | 3.00% | 1.85× | 19.6 ms | 17.1 ms   | 3.50 MB | 1.66 MB  | 224 MB | 168 MB |
| 19       | 524k  | 0.574s  | 3.07% | 0.324s   | 3.20% | 1.77× | 23.5 ms | 20.5 ms   | 4.12 MB | 1.99 MB  | 805 MB | 627 MB |
| 21       | 2.1M  | 1.890s  | 0.46% | 0.982s   | 1.73% | 1.92× | 27.0 ms | 24.3 ms   | 4.80 MB | 2.35 MB  | 3.22 GB | 2.46 GB |

**Key reads**:
- Lambda **proof is 2.0-2.1× larger** than P3 across all sizes — strongest
  justification for `PLAN-fri-optimizations.md`.
- Lambda **uses 28-31% more peak RSS** — confirms partial memory-bandwidth
  hypothesis.
- Verify is 12-15% slower in Lambda — competitive, not a target.
- CVs are healthy (0.46-7.96%); only log=17 would benefit from 20 runs.

### Breakdown — log=19 (`bench_vs_p3_breakdown_<date>/breakdown.tsv`)

Lambda total **545 ms**, P3 total **313 ms**, gap **232 ms**.

| Phase                  | Lambda | P3   | L/P3   | Gap (ms) | % of gap |
|------------------------|-------:|-----:|-------:|---------:|---------:|
| Constraint eval        | 168    | 46   | 3.64×  | +122     | **52.6%**|
| FRI commit             | 95     | 60   | 1.58×  | +35      | 15.0%    |
| Deep comp + OOD + ext  | 91     | 49   | 1.86×  | +42      | 18.1%    |
| Trace LDE+Merkle       | 130    | 103  | 1.27×  | +27      | 11.8%    |
| Quotient commit        | 20     | 44   | 0.46×  | -24      | -10.3% (L wins) |
| Queries                | 2      | 3    | 0.59×  | -1       | -0.6% (L wins)  |

### Breakdown — log=21 (`bench_vs_p3_breakdown_<date>/breakdown.tsv`)

Lambda total **1914 ms**, P3 total **965 ms**, gap **949 ms**. **Same dir
name as log=19 — overwrote it on the server**, only log=21 breakdown is
preserved.

| Phase                  | Lambda | P3   | L/P3   | Gap (ms) | % of gap | Trend (vs log=19) |
|------------------------|-------:|-----:|-------:|---------:|---------:|-------------------|
| Constraint eval        | 482    | 171  | 2.82×  | +311     | 32.8%    | improves          |
| Trace LDE+Merkle       | 540    | 298  | 1.81×  | +242     | 25.5%    | **worsens**       |
| Deep comp + OOD + ext  | 395    | 167  | 2.37×  | +228     | 24.0%    | **worsens**       |
| FRI commit             | 326    | 161  | 2.03×  | +165     | 17.4%    | worsens           |

**Reads**:
- Trace LDE+Merkle scales **super-linearly in Lambda** (4.15× when N×4)
  vs P3 (2.89×) — confirms per-column FFT is the structural problem at
  large N.
- At log=21 the gap distributes more evenly across phases — single-target
  optimisation will not be enough; multiple phases need attention.

### Column sweep — log=19, n ∈ {4, 8, 32, 64} (`bench_vs_p3_cols_n*/`)

| cols | n  | L prove | P3 prove | L/P3 | L RSS  | P3 RSS  |
|------|----|---------|----------|------|--------|---------|
| 8    | 4  | 0.445s  | 0.305s   | 1.46×| 484 MB | 446 MB  |
| 16   | 8  | 0.509s  | 0.317s   | 1.61×| 580 MB | 512 MB  |
| 32   | 16 | 0.574s  | 0.324s   | 1.77× (baseline) | 805 MB | 627 MB |
| 64   | 32 | 0.788s  | 0.373s   | 2.11×| 1.48 GB| 905 MB  |
| 128  | 64 | 1.274s  | 0.483s   | **2.64×** | 2.80 GB| 1.43 GB |

**Reads**:
- 16× more columns: Lambda **2.86×** wall-clock, P3 **1.59×**.
- Ratio L/P3 **goes from 1.46× to 2.64×** — degrades linearly with cols.
- **Production is worse than n=16 suggests** — Lambda VM has tables with
  70-200 columns, so production gap is closer to 2-3× than 1.77×.

---

## Plan we approved (then paused)

`/Users/juanpablo/.claude/plans/hice-git-checkout-b-calm-lake.md` — full
plan for "constraint eval over quotient domain (squared coset)".

### Why we paused

Reading deeper into prover/verifier code revealed:

1. **The prototype branch `origin/feat/eval-form-quotient` cherry-pick has
   serious conflicts** vs current main (#522, #566, #573 all touched
   prover.rs). 20 conflict markers in `5e9f05b8` alone.

2. **The "first commit" of the prototype (`5e9f05b8`) is just a refactor
   with no perf win** — its own commit message says "mathematically
   identical to the old path".

3. **The real win commit (`96e893ca`) depends on 3 other commits**
   (`80f1c299` BatchedLayout, `321f11a4` BatchedProof, `ab4da494` batched
   main trace) that introduce a multi-week refactor of the Merkle
   commitment infrastructure.

4. **The optimisation does not translate to production**:
   - Bench AIR Fibonacci-pair has `d_max = 1` ⇒ quotient_blowup=1 ⇒ N
     points (vs LDE 2N). **Half the work.** Bench gets a big win.
   - Lambda VM real (Keccak) has `d_max = 3` constraints ⇒ quotient_blowup=4
     ⇒ 4N points. **Twice the work** vs current LDE 2N. Production gets
     **negative impact** unless blowup is also raised to 4 (which makes
     trace/FRI slower).

5. **There is a deeper theoretical issue**: with `blowup=2` (Lambda
   production) and `d_max=3` (Keccak), the LDE 2N is **mathematically
   insufficient** to faithfully represent the composition polynomial of
   degree ~2N. Lambda's `interpolate + break_in_parts` fallback path
   commits to a **projection P** of degree < 2N, not the real H. This is
   probably soundness-preserving but is a different object than what P3's
   quotient domain commits to. Implementing the P3-style approach to
   Lambda VM real would require changing blowup to 4 (cost trade-off
   uncertain) or a different formulation.

### Final recommendation (open)

**Pivot to FRI early-stop + folding=4** (`PLAN-fri-optimizations.md`):
- Plan already written, step-by-step.
- Universal benefit: helps any AIR (degree 1 or 3), any blowup.
- Localised to `crypto/stark/src/fri/` — does not touch the constraint
  eval / quotient pipeline.
- ~1 week of work.
- Closes proof-size gap (4.1 MB → ~2.4 MB at log=19) directly.
- Saves ~165 ms of FRI commit at log=21 (17% of total gap).

**User did not commit to a direction by end of session — left to think.**

---

## Server-side bench output paths (still on `vm-benchmarks-1`)

```
~/juan/lambda_vm/bench_vs_p3_baseline_<YYYYMMDD>/         # 3-size baseline
~/juan/lambda_vm/bench_vs_p3_breakdown_<YYYYMMDD>/        # log=21 breakdown (log=19 was overwritten)
~/juan/lambda_vm/bench_vs_p3_cols_n4/                     # column sweep
~/juan/lambda_vm/bench_vs_p3_cols_n8/
~/juan/lambda_vm/bench_vs_p3_cols_n32/
~/juan/lambda_vm/bench_vs_p3_cols_n64/
```

To pull locally:
```bash
scp -r app@vm-benchmarks-1:~/juan/lambda_vm/bench_vs_p3_baseline_* ./
scp -r app@vm-benchmarks-1:~/juan/lambda_vm/bench_vs_p3_breakdown_* ./
scp -r app@vm-benchmarks-1:~/juan/lambda_vm/bench_vs_p3_cols_n* ./
```

---

## Open questions for next session

1. **Do we go with FRI plan or keep pushing on quotient-domain manually?**
   Recommendation: FRI.

2. **Should the bench-only gain from quotient-domain optimisation be
   pursued anyway** for paper/comparison purposes, even if it does not
   help production? (Bench at n=16 would go ratio 1.74× → ~1.42× at
   log=19, ~1.71× at log=21.)

3. **Where does Lambda actually evaluate Keccak's 200 degree-3
   constraints in practice?** The fallback path `interpolate +
   break_in_parts + FFT extension` was identified but not traced
   end-to-end. Worth confirming on a real-VM run that `number_of_parts ==
   3` is what actually fires for production AIRs.

4. **Does the `decompose_and_extend_d2` fast path (degree=2) also fire
   for Lambda VM real or only for d=2 AIRs?** If most production AIRs are
   d=2, the fast path covers them and only Keccak needs the fallback.

5. **What's the actual `composition_poly_degree_bound` for production
   AIRs?** All point to the default `lookup.rs:994` formula
   `trace_length * max_degree`. Worth grepping `compute_transition_prover`
   call sites to see what degree each AIR actually reports.

6. **Sweep blowup factor (2 vs 4 vs 8) on Lambda before committing to
   any approach** — would directly answer whether raising blowup is a
   viable path.

---

## How to resume

```bash
cd ~/Documents/lambda_vm5
git checkout bench_vs_p3_improvements
# everything we did is in the last 2 commits + a few uncommitted .md files
git log --oneline -3   # 7ee4cbf0, 4b19d250, 4bcd8771

# the plan file lives at:
cat ~/.claude/plans/hice-git-checkout-b-calm-lake.md

# the methodology doc is uncommitted:
cat bench_vs_plonky3/METHODOLOGY.md

# this log:
cat bench_vs_plonky3/SESSION_LOG_2026-05-04.md
```

If you decide to switch to FRI:
```bash
git checkout main && git pull
git checkout -b bench/fri-folding-and-early-stop
# follow PLAN-fri-optimizations.md step by step
```

If you decide to keep going with quotient-domain:
- Read `bench_vs_plonky3/METHODOLOGY.md` § "What could we do differently".
- Re-read `~/.claude/plans/hice-git-checkout-b-calm-lake.md`.
- Investigate the open questions 3-5 above before coding to confirm the
  optimisation translates to production.

---

# Continuation — 2026-05-05

> Branch: `bench_vs_p3_improvements` (same as 2026-05-04, pushed to origin
> mid-session, server checked out clean from origin).
> Server: `vm-benchmarks-1` (AMD EPYC 9454P, 48 cores).
> Goal: regenerate the data lost between sessions and run the **blowup sweep**
> that was missing — the bench that decides FRI vs quotient-domain.

## TL;DR

1. Pushed `bench_vs_p3_improvements` to origin and pulled it on the server
   (server was on stale `bench_vs_p3` with uncommitted local changes; stash
   applied to preserve them).
2. **Added `--blowup` to `bench_vs_plonky3/run.sh`** — the wrapper had blowup
   hardcoded to 2 (`prove_bench` already accepted the flag, but `run.sh` did
   not parse it). 5-line addition: a new `--blowup)` case in the arg parser.
3. Re-ran B0 baseline 3 sizes, B1+B2 breakdowns at log=19/21, B3 blowup sweep
   (3 blowups × 2 sizes), B4 column sweep at log=21. **16 dirs total**, all
   `scp`-ed to `~/Documents/lambda_vm5/bench_vs_plonky3/reports/server/`.
4. **Decision unblocked: pivot to FRI.** Quotient-domain raise-blowup is dead
   in Lambda — penalty of going from blowup=2→4 (+1240 ms at log=21) exceeds
   the constraint-eval ceiling (+482 ms saving) by ~1.6×. Net loss of ~758
   ms.

## What changed in code (server, branch `bench_vs_p3_improvements`)

`bench_vs_plonky3/run.sh` — added `--blowup N` flag. Inserted between
`--num-sequences)` and `--runs)` in the arg parser (line ~58):

```bash
        --blowup)
            if [[ $# -lt 2 ]]; then echo "--blowup requires an argument"; exit 1; fi
            BLOWUP=$2
            shift 2
            ;;
```

`BLOWUP=2` default and `--blowup "$BLOWUP"` passthrough to `prove_bench` were
already in place. **Not committed yet** — local change on the server. Should
be cherry-picked into the laptop branch and committed before merging.

## Bench results — 2026-05-05

### B0. Baseline reproducibility (3 sizes × 10 runs, default blowup=2)

| log | rows | L prove | L CV | P3 prove | P3 CV | L/P3 | vs prior session |
|-----|------|---------|------|----------|-------|------|------------------|
| 17  | 131k | 0.249 s | 7.79% | 0.134 s | 3.30% | **1.86×** | prior 1.85× — Δ +1% |
| 19  | 524k | 0.566 s | 9.89% | 0.322 s | 4.47% | **1.76×** | prior 1.77× — Δ -1% |
| 21  | 2.1M | 1.880 s | 1.03% | 0.976 s | 1.53% | **1.93×** | prior 1.92× — Δ +1% |

Reproducible within ±1% on the median. Log=19 Lambda CV touched the 10%
threshold; the breakdown re-run (B1) tightened it to 4.5%.

### B1+B2. Phase breakdown (log=19, log=21, default blowup=2)

Lambda phases come from `instruments` feature in the Lambda prover. P3 phases
come from `tracing-subscriber` spans. The same wall-clock total is reproduced
±1% in both runs. **All numbers are medians over 10 runs.**

#### log=19 (524k rows, total gap +242.3 ms)

| Phase | Lambda (ms) | P3 (ms) | L/P3 | Gap (ms) | % of gap |
|---|---:|---:|---:|---:|---:|
| **Constraint eval** | 171.5 (`r2_constraints`) | 48.6 (`quotient_values`) | **3.53×** | **+122.9** | **50.7%** |
| Trace LDE+Merkle (main) | 153.0 (`main_commits`) | 102.9 (`commit to trace data`) | 1.49× | +50.1 | 20.7% |
| FRI commit | 100.6 (`r4_fri_commit`) | 66.4 (`FRI prover`) | 1.51× | +34.2 | 14.1% |
| Prepass | 16.2 (`prepass`) | — | ∞ | +16.2 | 6.7% |
| Open / Deep+OOD | 92.4 (`r3_ood`+`r4_deep_*`) | 112.4 (`open`) | **0.82×** | **−20.0** | −8.3% (L wins) |
| Quotient commit | 20.8 (`r2_comp_commit`) | 46.8 (`commit to quotient poly chunks`) | **0.44×** | **−26.0** | −10.7% (L wins) |
| Queries | 1.9 (`r4_queries`) | ~3.4 (`query phase`) | 0.56× | −1.5 | −0.6% (L wins) |

Sum of attributed gap: 175.9 ms = 72.6% of 242.3 ms total. Remaining ~27%
unaccounted (Lambda `rounds_2_4` overhead vs sum of inner phases, P3 outer
span overhead).

#### log=21 (2.1M rows, total gap +951.2 ms)

| Phase | Lambda (ms) | P3 (ms) | L/P3 | Gap (ms) | % of gap |
|---|---:|---:|---:|---:|---:|
| **Constraint eval** | 478.2 | 175.7 | **2.72×** | **+302.5** | **31.8%** |
| Trace LDE+Merkle (main) | 540.2 | 289.2 | **1.87×** | **+251.0** | **26.4%** |
| FRI commit | 312.7 | 162.1 | 1.93× | +150.6 | 15.8% |
| Open / Deep+OOD | 394.3 | 327.1 | 1.21× | +67.2 | 7.1% |
| Prepass | 52.0 | — | ∞ | +52.0 | 5.5% |
| Quotient commit | 57.7 | 142.3 | **0.41×** | **−84.6** | −8.9% (L wins) |
| Queries | 2.5 | ~3.5 | 0.71× | −1.0 | −0.1% (L wins) |

Sum attributed: 737.7 ms = 77.6% of 951.2 ms total.

#### Trend log=19 → log=21

| Phase | log=19 ratio | log=21 ratio | Direction |
|---|---|---|---|
| Constraint eval | 3.53× | 2.72× | **improves** |
| Trace LDE+Merkle | 1.49× | 1.87× | **worsens** |
| FRI commit | 1.51× | 1.93× | worsens |
| Open / Deep+OOD | 0.82× (L wins) | 1.21× | **worsens** (L stops winning) |
| Quotient commit | 0.44× (L wins) | 0.41× (L wins) | flat |

The shape of the problem changes with N. At small N constraint eval
dominates; at large N it redistributes — trace LDE and FRI become more
significant. **At log=21 the gap is broadly distributed**, no single phase
fix kills it.

### B3. Blowup sweep — the decisive bench

**Question**: does raising blowup hurt Lambda more than P3? (Answers whether
quotient-domain optimisations that need higher blowup are viable.)

| log | blowup | L prove | L CV | P3 prove | P3 CV | L/P3 | L proof | L peak RSS |
|-----|-------:|--------:|-----:|---------:|------:|-----:|--------:|-----------:|
| 19  | 2 | 0.570 s | 7.04% | 0.318 s | 4.35% | **1.79×** | 4.12 MB | 0.82 GB |
| 19  | 4 | 0.921 s | 1.14% | 0.471 s | 1.46% | **1.95×** | 4.42 MB | 1.50 GB |
| 19  | 8 | 1.613 s | 1.09% | 0.738 s | 1.86% | **2.19×** | 4.73 MB | 2.84 GB |
| 21  | 2 | 1.897 s | 0.71% | 0.972 s | 1.52% | **1.95×** | 4.80 MB | 3.30 GB |
| 21  | 4 | 3.137 s | 0.46% | 1.443 s | 0.67% | **2.17×** | 5.13 MB | 5.99 GB |
| 21  | 8 | 5.760 s | 0.89% | 2.334 s | 0.66% | **2.47×** | 5.46 MB | 11.4 GB |

**Reads**:
- **Lambda scales worse than P3 with blowup.** Doubling blowup multiplies
  prove time by 1.65×→1.83× on Lambda, 1.49×→1.62× on P3. The L/P3 gap
  widens by ~+13% per blowup doubling.
- At log=21, going from blowup=2 to blowup=4 costs Lambda **+1240 ms**
  (1.897 → 3.137).
- Constraint-eval ceiling (the saving quotient-domain could yield): **482
  ms** at log=21 per the breakdown above.
- **Net of blowup-raising quotient-domain at log=21: +1240 − 482 = +758 ms
  worse.** The optimisation costs more than it saves.

**→ Quotient-domain optimisation that requires raising blowup is
DESCARTADA for Lambda VM real (Keccak deg-3 AIRs).**

### B4. Column sweep at log=21

| cols | num-sequences (n) | L prove | P3 prove | L/P3 |
|-----:|------------------:|--------:|---------:|-----:|
| 8    | 4   | 1.297 s | 0.829 s | **1.57×** |
| 16   | 8   | 1.459 s | 0.881 s | 1.66× |
| 32   | 16  | 1.886 s | 0.975 s | 1.93× (default) |
| 64   | 32  | 2.742 s | 1.172 s | 2.34× |
| 128  | 64  | 4.723 s | 1.643 s | **2.88×** |

Linear degradation with #cols, same trend as the (lost) log=19 sweep. **Going
from 8 to 128 cols moves L/P3 from 1.57× to 2.88×** — a 1.83× wider gap.
Lambda VM real has tables with 70-200 columns; **production-shape gap is
2.5-3.5×, not 1.93×**.

## Decision (unblocked)

**Pivot to FRI plan** (`PLAN-fri-optimizations.md`: folding=4 + early-stop):

- Blowup-independent — does not enter the regime where Lambda scales worse.
- Attacks FRI commit, which is **15.8% of the gap at log=21** (16% at
  log=19), the third-largest after constraint eval and trace LDE.
- Reduces proof size from 2.0× larger than P3 to ~1.2× — significant
  bandwidth/storage win independent of latency.
- Universal (helps any AIR, any blowup).
- ~1 week of work, plan already detailed.

**Quotient-domain (raise-blowup variant) is descartada.** Constraint eval
remains the largest single gap component at log=19 (50.7%), but the only
practical way to attack it (subir blowup) costs more than it saves.

A **third path** — quotient-domain optimisation **without raising blowup**,
working only on the existing 2N LDE — could yield smaller savings on
constraint eval without the blowup penalty. This was not explored. Open
question for a future session.

## Critical raw data on disk (laptop)

```
~/Documents/lambda_vm5/bench_vs_plonky3/reports/server/
├── bench_vs_p3_baseline_log{17,19,21}_20260505/   # 3 dirs
├── bench_vs_p3_breakdown_log{19,21}_20260505/     # 2 dirs (with breakdown.tsv)
├── bench_vs_p3_blowup{2,4,8}_log{19,21}_20260505/ # 6 dirs
├── bench_vs_p3_cols_log21_n{4,8,16,32,64}_20260505/ # 5 dirs
```

Each dir has `results.tsv`, `raw_metrics.tsv`, `metrics.txt`, `raw/*.stdout`,
plus `breakdown.tsv` for the breakdown runs. Total: 16 directories.
