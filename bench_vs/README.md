# Lambda VM vs SP1 v6 Benchmark

Compares proving time for an identical u64 wrapping Fibonacci computation.

## Prerequisites

1. **Lambda VM CLI** (built from this repo):
   ```bash
   cargo build --release -p cli
   ```

2. **SP1 toolchain** (Succinct's prover):
   ```bash
   curl -L https://sp1up.succinct.xyz | bash
   sp1up
   ```

3. **Rust nightly** (for cross-compiling Lambda VM guest):
   ```bash
   rustup toolchain install nightly
   ```

## Usage

```bash
# Default series: 1k, 10k, 100k, 300k iterations
./bench_vs/run.sh

# Custom series
./bench_vs/run.sh -n 1000 50000

# Approximate workload steps (converted with 5 steps/iteration)
./bench_vs/run.sh --steps 1000000 2000000 4000000 8000000

# Project to a target cycle count
./bench_vs/run.sh --target-cycles 500000000

# Run only one prover
./bench_vs/run.sh --lambda-only
./bench_vs/run.sh --sp1-only
```

## What is measured

Both provers execute the same program: iterative Fibonacci with `u64::wrapping_add`.

The timing window on both sides is **end-to-end single-shot proving, with no
verification and no recursion/compression**. Concretely:

| Phase                                      | Lambda VM timer | SP1 v6 timer |
|--------------------------------------------|:---------------:|:------------:|
| Read ELF + input from disk                 |        ❌       |       ❌     |
| Pre-pass execution to count cycles         |        ❌       |       ❌     |
| `setup` / verifying-key derivation         |  N/A (none)     |       ✅     |
| ELF parse + guest execution (inside prove) |        ✅       |       ✅     |
| Trace build                                |        ✅       |       ✅     |
| AIR construction                           |        ✅       |       ✅     |
| STARK prove (`core` mode)                  |        ✅       |       ✅     |
| Proof serialization / write                |        ❌       |       ❌     |
| Verify                                     |        ❌       |       ❌     |

Both sides run one extra execution pass **outside** the timer to report dynamic
instruction counts (SP1's `execute(...)` / Lambda's executor pre-pass). This
costs wall-clock time in the CI job but does not inflate the measured proving
time, and the cost is symmetric between the two provers.

Lambda VM uses the default proof options from `prover::prove_with_inputs`
(`GoldilocksCubicProofOptions::with_blowup(2)`, 50 FRI queries). SP1 v6 uses
the `core` proof mode exposed by `sp1-sdk::ProverClient::from_env()`.

## Projection axis

The linear projection uses **measured cycles** per prover — Lambda's executor
log count and SP1's `report.total_instruction_count()`. For Fibonacci the two
values agree to within ~1% (both compile to the same inner loop shape on
RISC-V). When cycle data is missing, the script falls back to the approximate
`target_workload_steps ~= 5 * n` label that was passed on the command line.

## Output

```
=== Summary ===
Program: Fibonacci (u64 wrapping)

  Target steps  Iterations      Lambda (s)   Lambda cycles         SP1 (s)      SP1 cycles     Ratio
  ------------  ----------      ----------   -------------         -------      ----------     -----
  1000000       200000              ...s         1004794             ...s         1004794       ...
  2000000       400000              ...s         2004794             ...s         2004794       ...

Timing window covers single-shot end-to-end proving; SP1 includes setup; both exclude verification.
Green ratio = Lambda VM faster, Red = SP1 faster
```

With `--report-dir DIR` the script writes:
- `results.tsv` — raw per-run data (`target_steps`, `iterations`, `lambda_time_s`, `lambda_axis_value`, `lambda_cycles`, `sp1_time_s`, `sp1_axis_value`, `sp1_cycles`, `ratio`).
- `metrics.txt` — key=value pairs including `timing_window=setup_plus_end_to_end_prove_no_verify`.
- `summary.md` — the same table plus linear projection to `TARGET_CYCLES` cycles.
- `raw/` — stdout/stderr of every individual run.
