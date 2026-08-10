# ethrex-block-converter

Converts a **real Ethereum block** into a serialized `ProgramInput` `.bin` for
the lambda-vm ethrex guest, reading an [`ethrex-replay`][replay] cache JSON.

This complements `tooling/ethrex-fixtures`, which builds *synthetic* blocks of N
plain ETH transfers. Those are cheap and deterministic but not representative:
they execute no contract code, touch a genesis state trie only a couple of
levels deep, and carry no bytecode in the witness. This tool produces the
opposite — a block that actually looks like Ethereum.

Figures below are for the **current default** real block, mainnet 25368371; a
repoint replaces them (see [Adopting a different block](#adopting-a-different-block)).
All are measured.

| | `ethrex_bench_20.bin` (synthetic) | `ethrex_mainnet_25368371.bin` (real, default) |
|---|---|---|
| gas used | 420,000 | **2,428,684** |
| transactions | 20 (all plain transfers) | 29 (real mix) |
| serialized size | 32,766 B | 1,110,156 B |
| cycles | 8,734,622 | **50,781,557** |
| keccak / ecsm calls | 411 / 80 | **10,478** / 116 |
| keccaks per ECSM call | 5.1 | **90** |

The last row counts keccaks per **ECSM call**, which is the ratio of the two rows
above it. One ecrecover issues four ECSM ecalls — `lincomb2_with_oracle` in
`crypto/ethrex-crypto/src/lib.rs` makes four oracle queries, each an `ecsm_mul` — so
per *ecrecover* the figures are 20.6 and 361, and the 80 and 116 above are 4x the 20
synthetic and 29 real signature checks.

Note the real block uses ~5.8x the gas and costs ~5.8x the cycles here, and that it
inverts the crypto mix: the synthetic block is ecrecover-bound, the real one keccak-
and trie-bound. That inversion is the point — a prover change can move the two
numbers in opposite directions. (The gas and cycle ratios agreeing is a coincidence
of this pair, not a rule: cycles/gas is 20.9 for this block on this ELF, and spans
29.7–38.2 across the three candidate blocks when all three are measured on one
common pre-LTO ELF — a ~29% spread.)

**Pin the ELF whenever you quote a cycle count.** Counts above were measured on this
branch at merge `fdb92f67` (main @ `9ccdaf2`), guest built with **clang 21.1.8**.
Two things move them:
- **Guest optimisation.** #861 gave the guest thin LTO; this block read 74,819,518
  on a mid-July pre-LTO ELF, so anything quoting ~74.8M or ~65.6M is **superseded**.
- **clang major version**, by ~2%. The guest embeds C (secp256k1-sys) and the
  Makefile pins target flags but not the compiler, so `cc` picks up whatever `clang`
  is on PATH. The RTX 5090 box (clang-18) measured **50,713,534** for this block on
  main @ `9ccdaf2` — 0.13% below the number above, same commit, different compiler.

Why this block specifically: it was the **only** block in a 90-day Dune sweep that
matched the shape constraints in the 1.6–2.6M gas band — exactly 2 heavy
transactions, no single whale transaction dominating, and a sane plain-transfer
share. Its composition is 11.94 tx/Mgas, 44.3% of gas in heavy transactions, 22.5%
in the top transaction, p50 transaction gas 41,297. A block that is merely *large*
is easy to find; one that is structurally typical is not.

The synthetic column is `ethrex_bench_20.bin` as the benchmark scripts actually
generate it — `ethrex-fixtures 20 … distinct`, i.e. 20 distinct genesis-funded
senders to 20 distinct recipients (`scripts/bench_verify.sh`,
`scripts/bench_recursion_scaling.sh`). The same block in `same` mode (one sender,
one recipient) serializes to 16,811 B.

Block 25368371 is **verified to run on the guest's precompile surface** (see
[Validation](#validation)); it needs no accelerator we don't have. Any
replacement block must clear the same check — that is what makes it usable, not
just realistic.

## Getting the fixture

The fixture is **fetched, not built**:

```bash
make ethrex-real-block-fixture
```

That downloads the finished `.bin` from `ETHREX_REAL_BLOCK_FIXTURE_URL` and
verifies `ETHREX_REAL_BLOCK_FIXTURE_SHA256` before moving it into place —
the same contract as `prepare-sysroot`. The file is gitignored (~1 MB; see
`executor/.gitignore`), and a corrupt or interrupted download is discarded rather
than left looking valid.

The digest of whatever is already on disk is re-checked on every invocation, not
only when the file is missing — so a stale copy left over from a re-upload under the
same block number, a corrupted file, or a hand-placed one is all caught and
re-fetched. That check is the reason these are phony targets rather than file rules.

Artifacts live in the **[`bench-fixtures-v1`][release]** release on
`yetanotherco/lambda_vm`, fetched unauthenticated:

| asset | sha256 | read by |
|---|---|---|
| `ethrex_mainnet_25368371_4f658c2b.bin` | `0a301731…` | every benchmark (**current default**) |
| `cache_mainnet_25368371.json` | `7aa88a5f…` | `regen-real-block-fixture` |
| `ethrex_mainnet_25453112.bin` | `0298663d…` | alternate candidate |
| `cache_mainnet_25453112.json` | `20ffbbc1…` | alternate candidate |

> The asset name carries the ethrex rev because the bytes are a function of it: the
> archived `ProgramInput` rkyv layout moves with the pin, so one block has one
> fixture per rev. The pre-bump bytes stay hosted under the original name
> (`ethrex_mainnet_25368371.bin`, `61eba49b…`) so older `main`s — whose Makefile
> pins that sha256 — keep fetching their own artifact.

Each block has two assets: the fixture and the **cache** it was converted from
(`make ethrex-real-block-cache`, ~2 MB, same verify-then-move contract). Only
`regen-real-block-fixture` reads the cache. Note it is *not* the cache the
converter's own tests use — see [Validation](#validation).

This crate is not on the fixture's path at all. Fetching a verified binary takes the
converter, the ~335-package ethrex host dependency tree and an ethrex-replay `rev`
pin off the critical path of everyone who just wants to run a benchmark. It also
decouples the benchmark block from what upstream hosts: ethrex-replay publishes a
cache for Hoodi and nothing else, so any mainnet block is unreachable by the
convert-locally route — producing its cache takes ~4 minutes and ~700 calls against
an archive RPC — and trivial by this one.

[release]: https://github.com/yetanotherco/lambda_vm/releases/tag/bench-fixtures-v1

## Regenerating the fixture (ethrex rev bumps)

Needed roughly twice a year, when the guest's ethrex `rev` moves and the rkyv
layout changes with it:

```bash
make regen-real-block-fixture     # fetches that block's cache, rebuilds the .bin
sha256sum "$(make -s print-real-block-fixture)"
```

Then upload the result and update `ETHREX_REAL_BLOCK_FIXTURE_SHA256` and its URL.

Directly, against any cache file:

```bash
cd tooling/ethrex-block-converter
cargo run --release -- <cache.json> <output_path>
```

Output is deterministic for a given cache file:

```text
wrote ../../executor/tests/ethrex_mainnet_25368371.bin (1110165 bytes): 1 block(s) \
  from mainnet starting at #25368371, 29 transaction(s), 2428684 gas
```

Verified at the current ethrex rev: regenerating from the hosted cache reproduces
`0a301731…` byte for byte, and the result passes `test_ethrex_real_block_native` —
which is what proves the hosted cache and the fixture the Makefile expects describe
the same block. (Byte count and digest are both rev-dependent: the `4f658c2b` bump
moved them from 1,110,156 B / `61eba49b…`, the bytes still hosted in the release.)

The converter's `conversion_is_reproducible` test enforces the same property, but
against its own pinned block rather than this one — see [Validation](#validation).

## What benchmarks with it

Costs below are for the **current default block** (mainnet 25368371) and move with
it — see [Measured cost of candidate blocks](#measured-cost-of-candidate-blocks).

| Where | How to run it | Cost (current default) |
|---|---|---|
| `benchmark-pr.yml` | **`/bench`** on a PR — also push to main and `workflow_dispatch` | 3 runs, 158.8 s each (~8 min of proving) |
| `bench-abba.yml` | **`/bench-abba [N]`** on a PR | ~72 min at the default 12 pairs (see below) |
| `benchmark-gpu.yml` | **`/bench-gpu [N]`** on a PR | 59.87 s/prove on an RTX 5090 (see below) |
| `scripts/bench_verify.sh` | `scripts/bench_verify.sh <ref>` | ~2.6 min per side, then cached |
| `scripts/perf_diff.sh` | `scripts/perf_diff.sh <ref>` | 5 recordings, so ~13 min of proving |
| `scripts/bench_abba.sh` | `scripts/bench_abba.sh <ref> [base] [pairs]` | 2 x 158.8 s per pair |

This block is what all three scripts prove by default (`WORKLOAD=real`); pass
`WORKLOAD=synthetic` for the N-plain-transfer fixture instead. `/bench-verify` is the
one flow that pins `synthetic`, because it reports a monolithic arm as well as a
continuation one and a real block does not fit monolithically.

None of them hardcode the fixture path or a block number — they read the path from
`make -s print-real-block-fixture` and run `make ethrex-real-block-fixture` on every
invocation, so the digest is re-checked rather than trusted.

**Every bench flow proves this block.** `/bench` runs it sampled on the shared
runner, against the cached baseline main publishes; `/bench-abba [N]` runs it as
N A/B/B/A pairs on that same runner; `/bench-gpu [N]` runs the same pairs on a
rented box, comparing PR vs main on the same machine — absolute GPU times are
host-CPU-dependent, so only same-box deltas are meaningful.

**Escalating from `/bench` to `/bench-abba`.** `/bench` resolves about 3%: three
runs of a 158.8 s prove, so it reports 3–10% as unresolved rather than as a
verdict. The paired test resolves a 95% delta of `t* × sd / sqrt(N)`, where `sd`
is the pair-delta standard deviation on the runner. **That sd is not measured
yet.** The columns below bracket it between 1.0% — the GPU box's measured 0.64%
pair sd plus margin — and 2.0%, which is `sqrt(2) ×` this runner's measured 1.43%
single-run CV:

| pairs | wall | resolves (sd 2.0%) | resolves (sd 1.0%) |
|---|---|---|---|
| 8 | ~50 min | 1.7% | 0.8% |
| **12** | **~72 min** | **1.3%** | **0.6%** |
| 20 | ~1h55m | 0.9% | 0.5% |
| 32 | ~3h | 0.7% | 0.4% |

12 pairs is the default. Wall is two 158.8 s proves per pair plus ~8 min of
setup. The first real `/bench-abba` run **measures** the sd — it is the `sd`
field of the paired-t line in the result comment — and this table should be
re-pinned to that value once it exists.

**GPU baseline (measured), and why the GPU epoch is 2^22.** On an RTX 5090 (32,607 MiB)
against main @ `9ccdaf2`, same fixture and CLI, one prove per setting:

| epoch | wall | VRAM | epochs | proof |
|---|---|---|---|---|
| 2^21 | 70.52 s | 19,193 MiB (58.9%) | 25 | 1.65 GB |
| **2^22** | **59.87 s** | 23,193 MiB (71.1%) | 13 | 1.12 GB |
| 2^23 | OOM after 9.7 s | 32,079 MiB (98.4%) — needs ~44 GiB | — | — |

**VRAM is the binding constraint**, so 2^22 is simply the largest setting that fits a
32 GiB card — and it is ~15% faster than 2^21 (equivalently, 2^21 is ~18% slower) with
28.9% headroom to spare. 2^23 is out of reach for every card below 48 GiB, not just
this one. `benchmark-gpu.yml` defaults the real-block path to 2^22 for this reason;
raw traces are in `~/workspace/lambda_vm_bench_cache/gpu_epoch_calib_2026-07-31/`
(`PROVENANCE.txt`).

2^22 is also what the CPU runner uses, but the two arrive there for different reasons
and must not be derived from each other: VRAM binds on the GPU path and host RAM on the
CPU one. The workflows pin it on both sides (`GPU_REAL_EPOCH_LOG2` here,
`REAL_BLOCK_EPOCH_LOG2` in `benchmark-pr.yml`, `ABBA_REAL_EPOCH_LOG2` in
`bench-abba.yml`); `bench_abba.sh` and `bench_verify.sh` still *default* to 2^20, as
does the CLI's `DEFAULT_CONTINUATION_EPOCH_SIZE_LOG2`, because 2^22 needs ~32 GiB of
host memory on a CPU build (peak RSS on the calibration box) and would break laptops.
See [Choosing the epoch size](#choosing-the-epoch-size) for the CPU tiers.

The CPU bench runner is roughly **2.65x** the GPU wall time for the same block: 158.8 s
median against the calibration RTX 5090's 59.87 s, both at epoch 2^22.

Do not derive one from the other in general: the CPU rate (3.13 s/Mcycle on this
block) does not transfer to the GPU, and the RTX 5090 sweep found the prover
CPU-bound at the serial producer above epoch 2^21, so GPU time lands closer to CPU
time than a naive device-throughput estimate suggests.

**Continuations are mandatory, not a tuning choice.** Peak heap on a monolithic
prove grows ~4.9 GB per million cycles on this workload family (measured on the
bench server: `10,728 MB + 2,007 MB/transfer`, R² = 0.998 across 4→20 transfers),
so this block would need **~240 GB** monolithically — and a heavier candidate far
more. `--continuations` makes peak heap a function of the epoch size instead of the
trace length, so the same block fits in **~32 GiB** at epoch 2^22 on the calibration
box — the setting both the CPU bench runner and the GPU path now use, picked by host
RAM on one and by VRAM on the other. Host peaks are machine-specific: the bench runner
itself measured **~52 GB** of peak heap for the same block and epoch. See
[Choosing the epoch size](#choosing-the-epoch-size) for the full curve and the other
tiers. The bundle on disk is ~1.15 GB (1.12 GB on the GPU path); a block would have to
be ~1.9x heavier to push it past the 2 GiB (2.147 GB) rkyv offset limit, which needs
`pointer_width_64` to serialize.

**`/bench` proves this block and nothing else** — 3 sampled runs against the cached
3-run baseline main publishes on every push. `/bench N` changes the sample count
(clamped to 5).

The synthetic N-transfer screen that used to run alongside it was **removed**. Two
reasons, recorded here because "add a cheap screen back" is an easy suggestion to
make twice:

1. Its only unique coverage was the **monolithic** prove path, which is vestigial —
   reportedly slower than a single-epoch continuation. Spending runner time to cover
   a path we intend to delete is not a trade worth making.
2. Its crypto mix is one **no real block has**: 9.16 ECSM per Mcycle, against a real
   block's 2.28. Screening against it tunes the prover for a worst case that
   cannot occur.

The synthetic fixtures themselves are not gone — `/bench-growth` still sweeps them for
a heap-vs-block-size slope, which needs a family of blocks and so cannot come from one
real one, and `/bench-verify` still proves the 20-transfer block so it can report a
monolithic arm as well as a continuation one.

**The cost is a shared resource.** One runner carries every `/bench`, `/bench-abba`
and `/bench-verify` in the repo, and a `/bench` now occupies it for **~15 min**, on
every comment and every push to main — ~8 min of that is the three proves, the rest is
checkout, a two-sided build, the fixture fetch and the guest ELF. `BENCH_RUNS_REAL` in
`benchmark-pr.yml` is the dial. A run measured **158.8 s** on that runner (median of 3,
2.8% spread, 13 epochs, epoch 2^22), so 3 runs is the right count; the dial is there
for a future block or prover change that takes a run past ~6 min.

Cycle counts here (8.73M synthetic, 50.78M real) are from merge `fdb92f67` (main @
`9ccdaf2`, clang 21). They move with guest optimisation and ~2% with the clang major,
so pin the ELF whenever you quote one, or it will look like a regression the next time
someone measures.

## Where validation runs

The checks themselves are described under [Validation](#validation) below; this is
where each one executes.

`.github/workflows/ethrex-block-converter.yml` runs them on changes to this crate,
`tooling/ethrex-tests`, or the `Makefile` — **not** on every PR. The fixture is a
benchmark input, read by no product code; it has to be right when it changes, not on
every commit, and running it per-PR put a network fetch and a cold build of ~335
packages in the required gate.

`no_kzg_backend_linked` is the exception and stays in the required gate
(`pr_main.yaml`): it is a pure unit test costing microseconds, and it is the property
the usability screen depends on, so it should fail on the PR that breaks it rather
than on some later, unrelated one.

## Prerequisites (for regeneration only)
Rust (stable) and network access on first run (cargo fetches the pinned ethrex
crates; `make` downloads the cache). **No RV64 target or sysroot needed** — this
is a host tool.

## Getting a cache for a different block (regeneration only)

The cache format is `ethrex-replay`'s, so use that tool to produce one — it
handles the RPC fetching, multiple client backends, and the `eth_getProof`
fallback, none of which is worth reimplementing here:

```bash
# In a checkout of https://github.com/lambdaclass/ethrex-replay
ethrex-replay cache <block-number> --rpc-url <url>
```

`debug_executionWitness` requires a **reth or ethrex** node; public providers
(Alchemy, Infura) do not serve it. `ethrex-replay` also supports `eth_getProof`
for geth/nethermind.

Only **mainnet, Hoodi and Sepolia** caches are accepted. ethrex-replay writes
`network: "LocalDevnet"` for any other chain, and that resolves to a test chain
(chain_id 9, every fork active from timestamp 0) — so converting it would replay
the block under invented rules while still passing every check here, since the
witness is only ever validated against whichever config we chose. The converter
refuses instead; `unmappable_network_is_rejected` pins that.

## Adopting a different block

The benchmark block and this crate's test block are **independent** — the fetch is
what decouples them — so a repoint touches two files and neither is this crate's
source.

**1. The Makefile — the only place a block number appears.** Five lines:

```make
ETHREX_REAL_BLOCK_NETWORK        := <mainnet|hoodi|sepolia>
ETHREX_REAL_BLOCK                := <block number>
ETHREX_REAL_BLOCK_FIXTURE_URL    := <release asset URL for the .bin>
ETHREX_REAL_BLOCK_FIXTURE_SHA256 := <sha256 of the .bin>
ETHREX_REAL_BLOCK_CACHE_URL / _SHA256 := <same, for its source cache>
```

**2. `REAL_BLOCK_FIXTURE` in `tooling/ethrex-tests`**, which points the usability
screen at the block actually being benchmarked. That is the whole of it.

**Nothing in this crate moves.** Its test constants stay pinned to Hoodi 1265656
across every repoint — what they exercise is the conversion, not the workload, and
Hoodi's is the one cache ethrex-replay publishes, so pinning there costs no hosting
and cannot drift.

Everything else derives from the Makefile — the fixture name, and through
`make -s print-real-block-fixture` the benchmark scripts and `benchmark-pr.yml`. No
workflow, script or env var names a block. Nothing in `executor/.gitignore` needs
touching either: it already ignores every accepted network's fixture name, so a
repointed ~1 MB fixture cannot become committable by accident.

### Measured cost of candidate blocks

Cost is a property of the block, so it changes with the repoint. All figures are
measured, never derived from gas — **cycles per gas is not constant** (20.9 for the
current default), so sizing a candidate from its gas mispredicts cost.

**Current default — main-vintage (merge `fdb92f67`, main @ `9ccdaf2`):**

These figures were measured on the pre-bump fixture (`61eba49b…`, 1,110,156 B) and are
left as measured rather than restamped. The ethrex `4f658c2b` bump changed the fixture
bytes, so they are a baseline for a workload that no longer exists byte-for-byte.

Post-bump CPU counterparts, measured ABBA on `vm-benchmarks-1` at the same epoch 2^22:
**45,074,552 cycles** (−11.24%), **142.37 s** CPU prove (−10.87%), **936.7 MB** proof
(−12.22%), peak RSS flat at ~48 GB. The GPU column has no post-bump counterpart yet.

| block | gas | cycles | GPU prove (RTX 5090) | CPU prove | proof | fixture |
|---|---|---|---|---|---|---|
| **mainnet 25368371** | 2.43M | **50,781,557** (clang 21)<br>50,713,534 (clang 18) | **59.87 s** @ epoch 2^22 | **158.8 s** @ epoch 2^22 (2.65x the GPU wall) | 1.15 GB CPU / 1.12 GB GPU | 1,110,156 B, `61eba49b…` |

Epoch 2^22 is the GPU recommendation: VRAM binds, 2^22 leaves 28.9% headroom on a
32 GiB card and 2^23 does not fit one at all. See
[Choosing the epoch size](#choosing-the-epoch-size) for the CPU tiers.

**Alternates — PRE-LTO vintage, superseded, re-measure before quoting.** These were
taken on a mid-July guest ELF, before #861 gave the guest thin LTO; the same build
change took the current default from 74,819,518 to ~50.7M, so expect these to fall by
a comparable factor. Kept because they are the selection evidence, not because the
numbers are current:

| block | gas | cycles (pre-LTO) | fixture |
|---|---|---|---|
| mainnet 25453112 | 4.24M | 125,932,956 | 2,019,747 B, `0298663d…` |
| hoodi 1265656 | 4.40M | 168,319,360 | 1,021,207 B, `1f7d4c4c…` |

All three clear the usability screen. Add a row rather than editing the wiring, and
say which ELF a number came from.

Two things these numbers show that a gas-based estimate would have got wrong, and both
survive the vintage change because they are same-ELF comparisons. **Gas does not size
cost:** on one common pre-LTO ELF the three blocks run at 30.8, 29.7
and 38.2 cycles per gas, so budgeting a candidate from gas alone is off by up to ~29%
— 25453112 and hoodi 1265656 sit within 4% of each other on gas (4.24M vs 4.40M) yet
25453112 costs ~25% fewer cycles (125.9M vs 168.3M). Gas happens to *order* these
three correctly; it does not size them. And **fixture size does not track cost**
either: the current default is the cheapest block and the middle-sized fixture.

The default is the cheapest of the three, which matters because the CPU workload sits
on a single shared bench runner. It was also the only block in a 90-day Dune sweep
matching the shape constraints (2 heavy transactions, no whale, sane transfer share)
in the 1.6–2.6M gas band — so it is cheap *and* structurally typical, not cheap
because it is degenerate.

### Choosing the epoch size

`--epoch-size-log2` trades memory for speed, and **the right value is a property of the
machine, not of the block**. Three tiers, all measured:

| where | epoch | why |
|---|---|---|
| GPU, 32 GiB card | **2^22** | VRAM-bound — 2^23 does not fit |
| CPU bench runner (≥64 GiB) | **2^22** | host-RAM-bound — it already peaks at ~52 GB here, and 2^23 measured 60 GiB on a roomier box |
| CPU server, 128 GiB class | **2^23** | the knee; 2^24 fits but is not worth it |
| laptops (CLI default) | **2^20** | unchanged, so a plain `cli prove` still works |

CPU sweep, 2026-07-31, on a 124 GiB / 32-core box, real block, **branch vintage**
(53,757,588 cycles on that box's clang-21 pre-LTO ELF):

| epoch | epochs | wall | peak RSS | proof |
|---|---|---|---|---|
| 2^20 | 52 | 616.90 s | 14.56 GiB | 2.83 GB |
| 2^21 | 26 | 464.26 s | 18.43 GiB | 1.72 GB |
| 2^22 | 13 | 397.88 s | 32.21 GiB | 1.15 GB |
| **2^23** | 7 | **356.47 s** | **60.01 GiB** | 0.90 GB |
| 2^24 | 4 | ~334 s | ~97–105 GiB *(provisional)* | — |

**There is a real knee.** Speed gained per doubling shrinks — 24.7%, 14.3%, 10.4%,
~6% — while memory roughly doubles at each step near the top. 2^23 uses 48% of a
124 GiB box (~52% headroom); 2^24 buys only ~6% more speed for ~15% headroom, so it
**fits but is not recommended on a shared box**, where one co-tenant turns a tight fit
into an OOM. The 2^24 RSS figure is provisional pending the calibration agent's formal
report.

A main-vintage anchor also ran on the same box: 2^23 = 344.12 s / 58.87 GiB, i.e. ~3%
faster than branch vintage at the same memory — as expected, since #861 cut cycles and
peak RSS is set by the epoch size rather than the trace length.

**Do not read absolute seconds or peak memory off this table for another machine.** This
box took 397.88 s at 2^22 on its branch-vintage ELF; the bench runner takes 158.8 s at
the same epoch on the main-vintage one — a 2.5x gap, against the ~6% the two cycle
counts differ by. The *ratios* between epochs transfer; the wall times do not. Memory
transfers no better: the bench runner peaks at ~52 GB of heap at 2^22 where this box
measured 32.21 GiB RSS.

### Verifying a repoint

Run **both**, in this order:

```bash
make ethrex-real-block-fixture          # fetch + verify the new .bin
make test-ethrex                        # the block is USABLE on the guest
```

`make test-ethrex` is the one that matters here, and the converter's tests cannot
replace it — they run against a different block. A new block is only
usable if it needs no accelerator the guest lacks, and this crate cannot tell you
that — its graph links a working c-kzg, so a block calling point evaluation (0x0a)
passes here and fails in the guest. `test_ethrex_real_block_native` in
`tooling/ethrex-tests` is the screen. See [Validation](#validation).

Benchmark comparability does not survive the swap, and that is intentional rather
than a wrinkle to work around: `benchmark-pr.yml` records which block it measured
and refuses to diff a PR against a baseline that measured a different one, so the
first run after a repoint reports one-sided numbers until main republishes.

## Why the JSON and not ethrex-replay's own `.bin`

`ethrex-replay` can already emit a rkyv `ProgramInput`, but it tracks ethrex
`main` while we pin a branch off it, and the type has diverged between the two
before: against the previously pinned rev (`156cb8d6…`) `main` carried an extra
`fee_configs` field and had moved the type from `l1::` to `input::`, so replay's
binary would not deserialize in our guest at all.

At the currently pinned rev (`4f658c2b…`, rebased onto recent `main`) the
`ProgramInput` definition matches `main`'s again, and that gap has closed for
now. What has not closed: replay resolves rkyv itself from ethrex's `^0.8.10`
rather than our exact `=0.8.16`, `main` has no `lambdavm` feature to build the
guest side against, and the next bump can reopen the type gap without warning.

The cache JSON carries only `blocks` + `witness` + `network` as plain serde, so
it survives that drift. This tool re-reads it with **our** pinned ethrex types
and re-serializes with **our** rkyv, which is what keeps the output layout
correct by construction. When the guest's ethrex `rev` is bumped, bump it here
too (and in `tooling/ethrex-fixtures` and the guest) and regenerate.

A previous real-block fixture (`ethrex_hoodi.bin`) was lost exactly to this kind
of drift — it predated the `Crypto` trait and stopped deserializing. Reading the
version-tolerant JSON instead of a pinned binary is the mitigation.

## Validation

Six checks, ordered so the argument builds: the host-side parity test first, then
what it does *not* cover, then the checks in `tooling/ethrex-tests` that close the
gap. Five run on the host and need no RV64 toolchain, and each executes in
milliseconds.

What costs time is a cold build of the ethrex host dependency tree — ~335
packages, including the `blst`, `c-kzg` and `secp256k1-sys` C builds, `malachite`
and `ark-ff/asm` — not the tests. That build is why these checks live in their own
path-filtered workflow rather than the PR gate (see [Where validation
runs](#where-validation-runs)); `ethrex-block-converter.yml` caches this workspace's
`target/` under its own key, so only cold runs pay it.

These run against this crate's own pinned block (Hoodi 1265656), **not** the
benchmark block — they test the conversion, which any real block exercises equally,
and Hoodi's is the one cache ethrex-replay publishes. Only
`test_ethrex_real_block_native` follows the benchmark block.

**`cargo test` here — `real_block_executes_under_guest_crypto`.** Executes the
block through `LambdaVmEcsmCrypto`, the `Crypto` impl the guest injects, so it is
exercised via the guest's own trait dispatch. Stateless re-execution ends in a
post-state-root check, so any divergence from consensus fails here.

**It does not screen KZG.** Declaring `ethrex-guest-program` with
`default-features = false, features = ["lambdavm"]` is necessary but not
sufficient: the `ethrex-config` dependency (used only for
`Network::get_genesis()`) pulls `ethrex-p2p`, whose `default = ["c-kzg"]`
propagates down to `ethrex-crypto/c-kzg` — and `default-features = false` cannot
switch it off, because ethrex's own workspace declares `ethrex-p2p` with defaults
on. Verify with `cargo tree -e features -i ethrex-crypto`. So point evaluation
(0x0a) resolves to a working c-kzg here and to nothing in the guest.

Scope of the gap: in `ethrex-crypto`, KZG is the **only** precompile whose
*availability* is feature-gated. The other two gates swap between working
implementations — `secp256k1` picks libsecp256k1 over k256, `std` picks malachite
over num-bigint for modexp — so they change which code runs, not whether a block
can execute. Dropping `ethrex-config` would therefore be a CI-time improvement
(it also sheds `ethrex-p2p`, `ethrex-blockchain`, `ethrex-storage` and the c-kzg
and secp256k1 C builds), not a correctness fix.

**`cargo test` here — `unmappable_network_is_rejected`.** Refuses a cache whose
`network` cannot be mapped to real chain rules rather than converting it under
substituted ones — see [above](#getting-a-cache-for-a-different-block) for why
that matters.

**`cargo test` here — `conversion_is_reproducible`.** Pins the block's stats and
the fixture's **sha256**. A length assert would not do: `ChainConfig` is
fixed-size, so a substituted chain config yields a byte-length-identical fixture,
and rkyv's `big_endian` feature would byte-swap in place — neither changes the
byte count. This is what catches an ethrex rev bump that moves the rkyv layout
instead of silently producing a fixture the guest can't read.

**`tooling/ethrex-tests` — `no_kzg_backend_linked`.** Asserts that crate links no
KZG backend. That was incidental to its dependency graph, and it is the property
the next check relies on, so it is pinned here rather than assumed.

**`tooling/ethrex-tests` — `test_ethrex_real_block_native`.** The one check that
follows the BENCHMARK block. Checks the serialized `.bin` itself deserializes and
executes. Since that crate links no KZG
backend, this is also **what screens point evaluation (0x0a)**: a block reaching
it diverges from consensus and fails here.

**`tooling/ethrex-tests` — `test_ethrex_real_block_vm`** (`#[ignore]`, excluded
from PR CI). The block through the guest ELF, comparing the VM's committed
output against the native reference. Needs the RV64 toolchain and its runtime is
unmeasured; run it on a build server, not a laptop:

```bash
cd tooling/ethrex-tests && cargo test --release test_ethrex_real_block_vm -- --ignored
```

[replay]: https://github.com/lambdaclass/ethrex-replay
