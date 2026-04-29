# Benchmarking

## Quick benchmark (default: fib_iterative_372k, 1 run)

```bash
scripts/bench_prove.sh
```

## More samples

```bash
scripts/bench_prove.sh executor/program_artifacts/asm/fib_iterative_372k.elf 3
```

## Branch comparison

Run from a feature branch to automatically compare against main:

```bash
git checkout feat/my-feature
scripts/bench_prove.sh executor/program_artifacts/asm/fib_iterative_372k.elf 3
```

## Single branch (no comparison)

```bash
scripts/bench_prove.sh executor/program_artifacts/asm/fib_iterative_372k.elf 3 --no-compare
```

Output:

```
=== Results ===
Program: fib_iterative_372k.elf
Runs: 3

  main        time(mean):  121.0s  time(median):  121.0s  heap(median): 20163 MB
  feature     time(mean):  121.4s  time(median):  121.4s  heap(median): 17629 MB

  Comparison:
    Time: +0.3%
    Heap: -2534 MB (-12.6%) <- DETERMINISTIC
```

## What it measures

- **Peak heap** (deterministic): jemalloc `stats.allocated` high-water mark, polled every 10ms from a background thread. Same value every run for the same code.
- **Time**: wall-clock proving time from the CLI `--time` flag.

## How it works

The CLI always uses jemalloc as the global allocator (faster than the system allocator). Building with `--features jemalloc-stats` additionally enables a background thread that polls `stats.allocated` every 10ms and reports the peak heap after proving completes.

In short:
- `cargo build -p cli` → jemalloc (no tracking overhead)
- `cargo build -p cli --features jemalloc-stats` → jemalloc + peak heap reporting

---

# Multi-workload suite (`bench_prove_suite.sh`)

`bench_prove.sh` measures one program at a time. By default everyone benches `fib_iterative_*`, but `fib` is a loop of 5 instructions: it activates only the `cpu`, `branch`, `register`, and `decode` chips. Chips like `shift`, `mul`, `dvrm`, `bitwise`, `load`, `memw`, `page`, and `commit` end up at 0% real usage on a fib run. A PR that improves the ALU+branch path by 3% but silently regresses `shift` (5 lookups per row, the costliest per-instruction chip) by 8% can ship green if `fib` is the only signal.

`bench_prove_suite.sh` runs `bench_prove.sh` across one or more programs that, together, exercise every chip. Default keeps fib alone (canary, ~10 min on the bench server vs ~30–60 min for `--all`); opt in to `--all` for full coverage.

## Quick start

```bash
scripts/bench_prove_suite.sh                          # fib only, 1 run, vs main (default)
scripts/bench_prove_suite.sh 3 main                   # fib only, 3 runs, vs main
scripts/bench_prove_suite.sh 3 main --all             # full 5-program suite
scripts/bench_prove_suite.sh 3 main --only keccak,quicksort
scripts/bench_prove_suite.sh 3 main --skip hashmap    # full suite minus hashmap
scripts/bench_prove_suite.sh 3 --no-compare --all     # all 5 on current branch only
```

The script wraps `bench_prove.sh`, so the `<runs>` and `<base_branch>` arguments behave identically. `--instruments` is forwarded too.

## Programs in the suite

| Name | ELF | Dominant stress | Steps |
|---|---|---|---|
| `fib` | `executor/program_artifacts/asm/fib_iterative_8M.elf` | cpu + branch + register + decode (canary) | ~8M |
| `keccak` | `executor/program_artifacts/bench/keccak.elf` | shift (5×HWSL/row) + bitwise AND/OR/XOR + memw | ~3.6M |
| `quicksort` | `executor/program_artifacts/bench/quicksort.elf` | load + memw_aligned + lt (BLT) + branch (JAL/JALR) | ~3M |
| `modular_exp` | `executor/program_artifacts/bench/modular_exp.elf` | mul (MULHU 64-bit) + dvrm (REM) + shift | ~1.3M |
| `hashmap` | `executor/program_artifacts/bench/hashmap.elf` | page (heap) + bitwise+mul (SipHash) + memw irregular + commit | ~5M |

### Why these five

- **`fib`** stays the canary. Cheapest real signal for the ALU + branch path.
- **`keccak`** is the only program whose hot path lives in `shift` and `bitwise` AND/OR/XOR. Without it, optimizations that reorder bitwise byte range-checks have nothing watching them.
- **`quicksort`** is recursive with random memory access. Covers `load`, `memw_aligned`, and the `lt` chip via BLT.
- **`modular_exp`** is the only program in the set that pulls heavily on `mul` and `dvrm` together.
- **`hashmap`** is real Rust: heap allocation hits the `page` chip, SipHash mixes `mul` + `bitwise`, and the print path covers `commit`.

Programs deliberately **not** in the default suite:

- `rust/ethrex.elf` — workload too noisy block-to-block (>20% variance) to use as per-PR signal. Reserve for integration runs.
- `rust/ckzg.elf` — clang-compiled C; flag drift between PRs muddies the signal vs Rust-only `modular_exp`.
- `bench/sum_array.elf`, `bench/binary_search.elf` — strict subsets of `quicksort` coverage.
- `bench/sieve.elf`, `bench/bitwise_ops.elf` — overlap with `keccak` + `hashmap`.
- `bench/syscall_commit.elf` — `commit` chip already covered by `hashmap`'s print path.
- `bench/matrix_multiply.elf` — overlap with `modular_exp` (both MUL-dominated).
- `bench/fibonacci_26.elf` — too small (~1.2M steps).

## Verdict

When run with a comparison branch, the script applies a per-program threshold to the time delta:

| Δ time vs base | Status |
|---|---|
| ≤ +3% | OK |
| > +3%, ≤ +5% | WARN |
| > +5% | FAIL |

Overall verdict is the worst per-program status, with priority `FAIL > INCONCLUSIVE > WARN > PASS`. A program that crashed or produced no comparison data is reported as `INCONCLUSIVE` rather than `FAIL`, so an environment problem is not confused with a real regression. Heap delta is reported but does not affect the verdict.

Exit codes:
- `0` — `PASS` or `WARN`
- `1` — `FAIL` (real regression detected)
- `2` — `INCONCLUSIVE` (one or more programs produced no comparison data), missing ELF artifact, or invalid arguments

These thresholds are deliberately loose for a 1–3 run sample. For a contested PR, re-run with `--only <program> 5 main` (or higher) to tighten confidence.

## Setup

The bench programs are not built by default. One-time setup:

```bash
make prepare-sysroot   # downloads RV64IM sysroot if missing
make compile-bench     # builds bench/*.elf (first run takes several minutes)
```

`fib_iterative_8M.elf` lives under `asm/` and is built by `make compile-programs-asm`.

If an ELF is missing, `bench_prove_suite.sh` exits with a clear error pointing to the right Make target — it never silently skips programs.

## Output

Per-program logs are saved under `/tmp/bench_prove_suite/<name>.log`. When `--instruments` is passed, each program's instrumented run is also preserved as `/tmp/bench_prove_suite/<name>_instruments.txt` (the underlying `bench_prove.sh` only keeps the most recent one in `/tmp/bench_prove/instruments.txt`, which would otherwise be overwritten between programs).

The aggregate verdict is printed last:

```
  program       time delta        heap delta        status
  ----------------------------------------------------------------
  fib           time:    +0.3%   heap:   -12.6%   [OK]
  keccak        time:    +6.7%   heap:    +0.1%   [FAIL]
  quicksort     time:    -1.1%   heap:    -0.4%   [OK]
  modular_exp   time:    +3.2%   heap:    +0.0%   [WARN]
  hashmap       time:    +0.8%   heap:    -2.3%   [OK]

  Thresholds: WARN > +3%, FAIL > +5% (regression vs main)

Overall: FAIL
```
