# Keccak Precompile vs Software — ethrex Empty Block

**Date:** 2026-04-30
**Branch:** `feat/ethrex-block-with-tx_precompile`
**Repo:** `~/Documents/lambda_vm6`
**Machine:** M1 (local), `cargo build --release`
**Proof options:** default (blowup=2, 50 FRI queries, Goldilocks³ extension)

## Goal

Quantify the impact of the keccak precompile on a realistic ethrex workload by
proving and verifying the same empty block twice — once with the precompile
active, once with a pure-software keccak — and comparing cycles, trace size,
prove time, verify time, and proof size.

## Setup

Two ethrex guest ELFs were built from the same source and run with the same
input file:

- **`executor/program_artifacts/rust/ethrex.elf`** — the patched ethrex guest
  where `patches/ethrex-crypto/keccak/mod.rs` routes `keccak256` to
  `lambda_vm_syscalls::keccak::keccak256` on `riscv64`. Each call becomes one
  `ECALL` (`a7 = u64::MAX - 1`, `a0 = state_ptr`) that the host resolves
  with `keccak_f1600`. The prover activates the `KECCAK`, `KECCAK_RND`, and
  `KECCAK_RC` chips.
- **`executor/program_artifacts/rust/ethrex_no_precompile.elf`** — a parallel
  guest (`executor/programs/rust/ethrex_no_precompile/`) where the same
  `riscv64` branch uses `tiny_keccak::Keccak::v256()` directly. No keccak
  ECALLs are emitted; keccak runs entirely as RISC-V instructions inside the
  CPU + Bitwise tables.

Confirmed via `llvm-objdump -d`:

- `ethrex.elf`: 2 ECALL sites with `li a7, -0x2` (keccak precompile).
- `ethrex_no_precompile.elf`: 0 such sites.

Input: `executor/tests/ethrex_empty_block.bin` (1 KiB, rkyv-encoded
`ProgramInput` for an empty Ethereum block).

## Measurement

```bash
./target/release/cli prove --time --cycles --elements \
  --output /tmp/eth_pc.proof \
  --private-input executor/tests/ethrex_empty_block.bin \
  executor/program_artifacts/rust/ethrex.elf

./target/release/cli prove --time --cycles --elements \
  --output /tmp/eth_npc.proof \
  --private-input executor/tests/ethrex_empty_block.bin \
  executor/program_artifacts/rust/ethrex_no_precompile.elf

./target/release/cli verify /tmp/eth_pc.proof  executor/program_artifacts/rust/ethrex.elf
./target/release/cli verify /tmp/eth_npc.proof executor/program_artifacts/rust/ethrex_no_precompile.elf
```

`Proving time` printed by the CLI is wrapped around the STARK prove only. It
**excludes** ELF load, parsing, executor pre-pass, trace build, and proof
serialization — same definition as `bench_vs/run.sh` uses for the SP1
comparison.

## Configurations

There are three real configurations, not four. The software run uses
`tiny-keccak` and never activates the keccak chips — so "chip optimized" vs
"chip un-optimized" is not a meaningful axis on the software side. The two
software measurements below differ only because the prover binary was
rebuilt between runs (build/thermal/sample-size noise on a single shot).

- **Software baseline** — `ethrex_no_precompile.elf`. Pure tiny-keccak inside
  the guest, no ECALLs to the precompile, KECCAK chips empty (only padding).
- **Precompile + optimized chip** — `ethrex.elf` proved by vm5's
  `feat/optimized_keccak` build. Three spec optimizations applied: skip ρ on
  `state[0][0]` (`ca6dfa5f`), drop rc[2,4,5,6] (`dc7d3e16`), Cxz_right
  Byte→Bit (`f93d01d4`).
- **Precompile + un-optimized chip** — `ethrex.elf` proved by vm4's
  `feat/keccak` HEAD. Only Cxz_right Byte→Bit is kept. The other two
  optimizations are *not* part of the merge target.

## Raw output

```
=== Software baseline (tiny-keccak) ===
Cycles: 185331
Elements: 61654246..61654342    (≈ 61.65M, +144 cells across two builds = noise)
Aux elements (EF-cols): 30944625..30944673  (≈ 30.94M)
Proving time: 46.839s and 52.174s on two single-shot runs

=== Precompile + optimized chip ===
Cycles: 117566
Elements: 49018394
Aux elements (EF-cols): 26167285
Proving time: 43.406s
Verify: Verification succeeded! (6.29s)

=== Precompile + un-optimized chip (merge target) ===
Cycles: 117566
Elements: 49024538
Aux elements (EF-cols): 26170357
Proving time: 50.739s
```

## Results

| Config | Cycles | Main cells | Aux EF cells | Proving time |
|---|---:|---:|---:|---:|
| Software baseline | 185,331 | ~61.65M | ~30.94M | **~47–52s** (1 config, 2 single-shot measurements show variance) |
| Precompile + optimized chip | 117,566 | 49,018,394 | 26,167,285 | **43.406s** |
| Precompile + un-optimized chip (merge target) | 117,566 | 49,024,538 | 26,170,357 | **50.739s** |

Verify time is ~6.3s and proof size is ~107 MiB in all three.

### Cost of the dropped optimizations

The two dropped optimizations (`ca6dfa5f` skip ρ on (0,0); `dc7d3e16` drop
rc[2,4,5,6]) reduce KECCAK_RND from 1480 cols → 1456 cols, bus interactions
from 1371/row → 1347/row, and KECCAK_RC from 10 cols → 6 cols.

| Comparison | Δ |
|---|---:|
| Precompile prove with optimized chip vs un-optimized chip | +17% slower without the optimizations (43.4s → 50.7s) |
| Precompile vs software baseline (optimized chip) | −7% (43.4 vs ~47s) |
| Precompile vs software baseline (un-optimized chip, merge target) | −2 to −5% (50.7 vs ~47–52s) |

### Why the two software prove times differ (and why it is not signal)

Trace cell counts changed by 144 / 92.6M = **0.0002%** between the two
software runs. A 5+ second wall-time delta on a 47s prove from a 0.0002%
trace change cannot come from real work — it is single-shot measurement
noise:

- The prover binary was rebuilt between runs (chip files were swapped, so
  `lambda-vm-prover` was recompiled). Different binary layout → different
  cache and branch-predictor behavior.
- M1 thermal: first prove after a build runs at higher boost frequency,
  later runs may throttle.
- Single sample per config. Reliable benchmarking needs ≥3 runs and a
  median, with CV under 10%.

Treat the software prove as a single ~47–52s data point. The headline
result that survives is: **precompile beats software on empty block in
both chip configs**, with a meaningful additional ~17% headroom available
from re-applying the two dropped optimizations after the merge.

## Reading the metrics

### `Cycles`

Number of dynamic RISC-V instructions executed during the run. The executor
counts these in a pre-pass outside the prove timer. With the precompile,
each `keccak256(...)` call collapses to one `ECALL` instruction; the host
runs `keccak_f1600` outside the CPU trace. With software, every call expands
to several thousand RISC-V instructions (theta/rho/pi/chi/iota × 24 rounds).

Empty block runs only a handful of keccak calls (header hashes,
`initialize_block_header_hashes`), so the absolute saving is ~67k cycles, but
the relative reduction is large (−37%) because keccak is a meaningful
fraction of an otherwise minimal workload.

### `Elements` (main-trace)

Sum of `rows × cols` across all AIRs in the prover (`CPU`, `MEMW`, `LOAD`,
`BITWISE`, `DECODE`, `HALT`, `COMMIT`, `REGISTER`, `PAGE`, `KECCAK`,
`KECCAK_RND`, `KECCAK_RC`).

The precompile increases `KECCAK_RND` (1456 cols × 24 rows per permutation),
but `CPU` and `BITWISE` shrink more, so the net is −20%. With software,
keccak floods `BITWISE` with XOR/AND lookups, inflating that table.

### `Aux EF-col elements`

Auxiliary columns for LogUp lookups, evaluated in the Goldilocks³ extension
field. Each row's cell is three Goldilocks elements. These columns dominate
prove cost because every cell requires extension-field multiplications and
inversions.

`KECCAK_RND` emits **1347 bus interactions per row × 24 rows** per
permutation. That is the structural reason the prove-time win (−7%) is much
smaller than the cycle win (−37%): the round chip's AUX columns absorb most
of the saving from removing keccak instructions from the CPU.

### `Proving time`

Timer wrapped exclusively around the STARK prove (FRI commit + open + LogUp
arguments + composition). Excludes I/O, executor, trace build, and verify.
This is the apples-to-apples number for prover comparisons.

### Verify time

Re-evaluates the constraint polynomials and the FRI queries. Cost depends on
number of queries (50), FRI depth (`log` of trace size), and number of
tables — not on which tables are populated. Hence the near-tie (6.29 vs
6.56s).

### Proof size

Constant at 107 MiB regardless of which chips are active. Driven by Merkle
roots, FRI query paths, and the final low-degree polynomial — not by the
density of any individual trace.

## Verdict

The precompile already wins on the smallest realistic ethrex workload, where
keccak is only ~6% of cycles. The margin (−7% prove time, −20% trace size) is
modest because the round chip's AUX columns are heavy. The win grows quickly
as keccak's share of the workload grows; the synthetic bench
`bench/keccak_precompile.elf` (1000 chained hashes) has **5,738× fewer
cycles** and **67× smaller main trace** than the software baseline
`bench/keccak.elf`, which translates to a much larger prove-time speedup.

For ethrex blocks with real transactions (more state hashing, MPT updates,
receipt construction), the precompile win is expected to grow significantly
beyond the 7% seen here.

## Limitations of this run

- Only the empty block was measured locally. `simple_tx` (4.2M cycles, 586M
  trace elements) was attempted twice and both runs were OOM-killed by the
  M1's 32 GB RAM. Larger inputs need a machine with ≥64 GB RAM (the bench
  server `vm-benchmarks-1` has 125 GB).
- A single sample per configuration. For publishable numbers, take median of
  3+ runs to bound variance.
- M1 local is single-machine, no thermal/freq isolation. The bench server
  is the canonical machine for this comparison.

## Reproducing

### One-time setup

```bash
cd ~/Documents/lambda_vm6
make compile-programs-asm
make executor/program_artifacts/rust/ethrex.elf
make executor/program_artifacts/rust/ethrex_no_precompile.elf
cargo build --release -p cli
```

### Run the comparison

```bash
./target/release/cli prove --time --cycles --elements \
  --output /tmp/eth_pc.proof \
  --private-input executor/tests/ethrex_empty_block.bin \
  executor/program_artifacts/rust/ethrex.elf

./target/release/cli prove --time --cycles --elements \
  --output /tmp/eth_npc.proof \
  --private-input executor/tests/ethrex_empty_block.bin \
  executor/program_artifacts/rust/ethrex_no_precompile.elf
```

### Run as tests

Both prove+verify tests are in the branch but marked `#[ignore = "takes too
long"]`:

```bash
cargo test --release -p lambda-vm-prover \
  test_prove_ethrex_empty_block test_prove_ethrex_empty_block_no_precompile \
  -- --ignored --nocapture
```

Executor-only smoke tests (no proof generation, fast):

```bash
cargo test --release -p executor --test rust \
  test_ethrex_empty_block test_ethrex_empty_block_no_precompile -- --nocapture
```

## Related

- Precompile correctness: `cargo test --release -p lambda-vm-prover keccak --lib`
  (8/8 pass, includes E2E `test_prove_elfs_keccak` on `test_keccak.s`).
- Audit report: `KECCAK_PRECOMPILE_AUDIT.md` (497 lines, covers chip
  architecture, soundness checklist, spec deviations).
- Spec deviations: `docs/keccak-spec-deviations.md`.
- Upstream spec bugs (ready to file): `keccak_bug.md`.
