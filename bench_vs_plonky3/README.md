# Lambda STARK vs Plonky3 Fibonacci

This benchmark proves the same Fibonacci-pair statement in both systems:

- `num_sequences` independent sequences.
- Two columns per sequence: `(left, right)`.
- Transition: `next.left = left + right`, `next.right = right + next.left`.
- Public inputs pin the first `(left, right)` pair for each sequence.
- Trace generation, AIR construction, proof serialization, and verification are reported but excluded from the prove-time ratio.

The runner prints an `AUDIT` line before timing so the protocol workload is visible:

- Lambda: transition constraints, base-transition constraints, boundary constraints, composition chunks, columns.
- Plonky3: AIR constraints, first-row constraints, transition constraints, quotient chunks, columns, and packing width.

Run:

```sh
./bench_vs_plonky3/run.sh --log-rows 19 --num-sequences 16 --runs 10
```

By default the script requests no explicit Plonky3 SIMD packing on x86_64 with:

```sh
RUSTFLAGS="-C target-feature=-avx2,-avx512f"
```

Use `--native-simd` to benchmark compiler defaults. On aarch64, Plonky3 Goldilocks selects NEON from `target_arch = "aarch64"` and needs a Plonky3 source patch for a true scalar packing flag.
