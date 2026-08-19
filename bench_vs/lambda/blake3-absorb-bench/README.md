# blake3-absorb-bench — the in-guest BLAKE3 micro cycle harness

In the tree on purpose. Stage 4's equivalent lived in a scratch directory and is
gone, so its numbers (556 / 4276 / 781 / 2254) can no longer be reproduced or
re-derived — only quoted. This one stays.

## What it measures

Per-hash guest cycles for one message length under one hashing arm. Each
`(arm, length, N)` triple is its own `[[bin]]`, so a build never has two shapes
in one binary and parallel builds cannot clobber each other's output.

Six arms, and the reason there are six rather than two:

| arm | message | absorb compiled in | what it isolates |
|---|---|---|---|
| `none` | — | — | the loop, subtracted from the rest |
| `keccak` | aligned | — | `PlatformKeccak256`, the comparison |
| `b3old` | **aligned** | **no** | ★ the honest BLAKE3 control |
| `b3oldun` | unaligned | no | the misalignment cost, alone |
| `b3single` | unaligned | yes | absorb compiled in but declined |
| `b3absorb` | **aligned** | **yes** | the absorb path |

★ **Use `b3old` as the control, never `b3single`.** Comparing aligned-absorb
against unaligned-non-absorb looks like a clean A/B — same binary, one condition
— but it is confounded and it flatters the absorb path: an unaligned message
also makes the hasher's `copy_from_slice` into its pending block cost more,
measured at **+80 cycles per 64-byte block**, about a third of the old path's
248. The 2×2 above separates alignment from the feature so neither is charged
the other's cost.

## Running it

Build one bin with the canned guest recipe (`Makefile`, `define build_guest_elf`),
then execute it:

```sh
cd bench_vs/lambda/blake3-absorb-bench
CARGO_TARGET_DIR=../../../executor/shared_target \
rustup run nightly-2026-02-01 cargo build --release \
  --target ../../../executor/programs/riscv64im-lambda-vm-elf.json \
  -Z build-std=core,alloc,std,compiler_builtins,panic_abort \
  -Z build-std-features=compiler-builtins-mem -Z json-target-spec \
  --bin b3bench-b3absorb-1024-n1000 --features "arm_b3absorb,len1024,n1000"

cargo run -p cli --release -- execute \
  executor/shared_target/riscv64im-lambda-vm-elf/release/b3bench-b3absorb-1024-n1000 --cycles
```

`--cycles` also prints per-accelerator call counts, and those are the check that
an arm is what it claims. At 1024 bytes `b3absorb` must report **2000** BLAKE3
calls per 1000 hashes (one absorb plus one final compression) where `b3old`
reports **16000**. If that number is wrong the cycle figure is meaningless — the
feature did not take, or the buffer was not aligned.

## Two ways to get the per-hash number, and what each includes

- **Two-point**, `(C(N=2000) − C(N=1000)) / 1000`: no baseline arm needed and the
  harness's fixed setup cancels exactly. **Includes** the per-iteration loop
  body. This convention reproduces Stage 4's published figures to +0.1%.
- **Baseline-subtracted**, `(C_arm(N) − C_none(N)) / N`: isolates the hash alone.

They differ by the loop body (~114–128 cycles here) and both are correct — but
they are not interchangeable, so say which one a number is. ⚠ Stage 4's figures
are the loop-INCLUSIVE kind: its `none` arm optimized away to a bare counting
loop (4,781 cycles for 1,000 iterations cannot contain a real 32-byte XOR body),
so quoting its "308 fixed" against a loop-exclusive "184 fixed" compares two
different quantities.

## The numbers this produced (2026-08-18)

Baseline-subtracted, per hash:

| bytes | keccak | b3old | **b3absorb** |
|---:|---:|---:|---:|
| 64 | 669 | 432 | **454** |
| 256 | 948 | 1 176 | **552** |
| 1024 | 2 142 | 4 152 | **552** |
| 4096 | 6 885 | 16 062 | **558** |

Marginal cost per absorbed block: **248.1 → ~0.1 cycles**. 256 B and 1 KiB are
bit-identical at 552.0 despite absorbing 3 versus 15 blocks, which is the
property stated in its strongest form.

⚠ At 64 bytes `b3absorb` is **worse** than `b3old` by 22 cycles: nothing can be
absorbed at one block, so that is what compiling the arm in costs the shape FRI
query paths are made of. `Blake3Chain::update` guards the call with
`input.len() > BLOCK_LEN`, which recovers 14 of it.
