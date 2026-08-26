# Bumping the ethrex pin to `797df554`

Moves the guest from `4f658c2b` (2026-07-31, on the unmerged
`feat/lambdavm-prover-backend` branch) to `797df554` (2026-08-07, on ethrex
`main`).

## Results

Real mainnet block 25368371 — the fixture `bench_abba.sh`, `bench_verify.sh`,
`perf_diff.sh` and `benchmark-pr.yml` all resolve through
`make print-real-block-fixture`.

| | before (`4f658c2b`) | after (`797df554`) | delta |
|---|---|---|---|
| **Proving time** (median of 3) | **120.715 s** | **110.259 s** | **−8.66%** |
| run spread | 0.681 s (0.56%) | 0.821 s (0.74%) | |
| Executed cycles | 34,241,608 | 30,498,818 | −10.93% |
| **Epochs** (2^22) | **9** | **8** | −1 |
| Peak heap | 47.3 GB | 46.2 GB | −2.4% |

10.46 s off every proof, against a run spread below 0.8% on both sides — the
delta is roughly twelve times the noise.

Against `main`'s pin (`156cb8d6`, 2026-05-22) the guest goes from **39,563,400**
to **30,498,818** cycles, **−22.9%**.

### Time does not track cycles

−10.93% of cycles buys −8.66% of proving time, and that gap is structural rather
than measurement error: proving cost also carries per-epoch fixed work and table
rows that do not shrink with the instruction count. Quoting a cycle delta as if it
were a time delta will overpromise.

The epoch count is the clearest instance. At 2^22 cycles per epoch, 34.2M needed 9
and 30.5M fits in 8, and that single step accounts for much of the 10 s. It also
means the next increment of cycle savings returns less until it drops the ninth
epoch — the benefit arrives in steps, not smoothly.

## What changed

**The pin**, in all 11 places (`scripts/set_ethrex_rev.sh` moves them together, so
the guest and the fixture tooling cannot drift apart — a mismatch there does not
fail to build, it produces a fixture the guest silently misreads).

**The `lambdavm` feature is gone**, and not replaced. `ethrex-guest-program`
carries one feature per zkVM and none of them selects a backend: each is only a
list of optional crypto dependencies, and none gates any method of the `Crypto`
trait (its gates are `secp256k1`, `c-kzg`, `blst`, `std`). Measured against the
sibling feature that activates the widest dependency set, ethrex's own precompile
stress fixtures come out identical to the cycle — `stress_modexp_150M`
6,373,285,966 and `stress_alt_bn128_150M` 22,986,061,145 both ways — with the real
block at 30,498,818 vs 30,501,620 and a 2,928-byte smaller ELF.

Requiring `lambdavm` is what tied the guest to the backend branch, since that is
the only place it exists. What makes this guest LambdaVM never travelled through
it: `lambda-vm-syscalls`, `lambda-vm-ethrex-crypto` and the
`riscv64im-lambda-vm-elf` target are direct, and every run above reports 116 ECSM
and 10,659 keccak precompile calls — the injected crypto is live on both sides.

**Fixtures regenerated.** The rkyv `ProgramInput` layout moved with the rev, so
every committed `.bin` from the old pin fails to deserialize. All four synthetic
fixtures plus the real block were rebuilt; the converter's reproducibility digest
was updated, which is what that test asks for on a legitimate rev bump.

| fixture | cycles | ECSM |
|---|---|---|
| `ethrex_empty_block` | 431,162 | 0 |
| `ethrex_simple_tx` | 653,263 | 4 |
| `ethrex_bench_4` | 1,047,001 | 16 |
| `ethrex_10_transfers` | 1,286,761 | 40 |
| real block 25368371 | 30,498,818 | 116 |

## Why `797df554` and not something newer

`b5271885` (2026-08-14) rewrites the guest entry point from `ProgramInput` /
`execution_program` to `run_stateless_guest` taking schema-prefixed SSZ
`statelessInputBytes`. Past that commit this is no longer a pin bump — it needs a
new guest shim and a new fixture format. `797df554` is the last point where the
bump is only the rev.

Within that window it is also the right end of it: between 2026-08-07 and the
boundary there are 14 commits and **no `perf(levm)`**. The two that matter —
`perf(levm): monomorphize the dispatch loop on the validation observer` (#7105)
and `perf(l1): give branch-node hashing a monomorphic RLP encoder` (#7104) — are
already in.

Release tags do not help here. `v24.0.0`'s commit predates both, and it is
*diverged* from `main` (4 commits of its own, 66 behind) because ethrex cuts
releases from a side branch rather than from the tip — so a tag can lack work that
is already merged. `v25.0.0` is past the API boundary.

## Method

`vm-benchmarks-1`, 96 cores / 125 GB, rustc 1.94.0. The real-block step of
`.github/workflows/benchmark-pr.yml` verbatim:

```
cli prove $ELF --private-input $REAL_INPUT \
  --continuations --epoch-size-log2 22 -o /tmp/real_proof.bin --time
```

3 runs per side, reported as median plus spread.

Both sides started on an idle box (load 0.61 and 1.99). The baseline was held back
until load fell below 3 after the first side finished — starting it hot would have
inflated the baseline and flattered the bump.

Each side used **its own canonical fixture**: the baseline fetched the published
`_4f658c2b` release asset, the bumped side used the newly generated `_797df554`
one. They cannot share a file — the rkyv layout differs — and each side's own
asset is what CI would fetch, which is the honest comparison rather than two
ad-hoc conversions.

Cycle counts are deterministic: the bumped side reported 30,498,818 identically on
a laptop and on the bench box.

Raw logs in `bench_ethrex_bump_20260826/`.

## Before merging

`ethrex_mainnet_25368371_797df554.bin` (in the repo root) must be uploaded to the
`bench-fixtures-v1` release of **yetanotherco/lambda_vm**. The Makefile already
points at it with its sha256; until the asset exists, `make
ethrex-real-block-fixture` works locally (the file is in place and verifies) but
CI cannot fetch it.

This also stops being a routine bump: it leaves `feat/lambdavm-prover-backend` and
consumes ethrex `main` as a library. That branch is 69 commits behind `main` and
has not moved since 2026-08-04, so waiting for it to merge was itself costing the
−22.9% above. The PR's existing approval predates this change.
