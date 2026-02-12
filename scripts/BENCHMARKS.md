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
