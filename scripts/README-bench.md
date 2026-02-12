# Benchmarking

## Quick benchmark (default: 1 run)

```bash
scripts/bench_prove.sh executor/program_artifacts/asm/fib_iterative_372k.elf
```

## More samples if needed

```bash
scripts/bench_prove.sh executor/program_artifacts/asm/fib_iterative_372k.elf 3
```

## Branch comparison

Run from a feature branch to automatically build both branches and compare:

```bash
git checkout feat/my-feature
scripts/bench_prove.sh executor/program_artifacts/asm/fib_iterative_372k.elf 3
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
- Peak RSS is collected in raw data (`/tmp/bench_prove/`) but not shown in the summary — it's noisy and varies across runs.

## How it works

Builds the CLI with `--features jemalloc-stats`, which:

1. Uses jemalloc as the global allocator (`tikv-jemallocator`)
2. Spawns a background thread that polls `stats.allocated` every 10ms
3. Reports the peak after proving completes

Normal builds (without the feature) use the system allocator and don't include any tracking overhead.

## Other scripts

- `bench_compare.sh` — hyperfine-based time comparison across branches (no memory tracking). Useful for quick A/B timing with statistical rigor.
