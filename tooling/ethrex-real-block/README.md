# ethrex-real-block

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
| cycles | 9,063,727 | **~65.6M** |
| keccak / ecsm calls | 411 / 80 | **10,478** / 116 |
| keccaks per ecrecover | 5.1 | **90** |

Note the real block uses only ~5.8x the gas while costing ~7.2x the cycles, and
that it inverts the crypto mix: the synthetic block is ecrecover-bound, the real one
keccak- and trie-bound. That inversion is the entire point — a prover change can
move the two numbers in opposite directions.

Cycle counts are for a **current guest ELF** and move ~14% with ELF vintage (this
block reads 74,819,518 on a mid-July ELF); pin the ELF whenever you quote one.

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

> Both URLs are currently **unset**. Until the artifacts are hosted, these targets
> fail with an actionable message and every consumer degrades gracefully: the
> benchmark workflow warns and skips its real-block section, and the usability
> screen in `.github/workflows/ethrex-real-block.yml` skips too.

The second artifact is the **cache** — the ethrex-replay JSON the fixture was
converted from (`make ethrex-real-block-cache`, ~2 MB, same verify-then-move
contract). Only the converter's tests and `regen-real-block-fixture` read it.

This crate is *not* on the fixture's path. Fetching a verified binary takes the
converter, the ~335-package ethrex host dependency tree and an ethrex-replay `rev`
pin off the critical path of everyone who just wants to run a benchmark. It also
decouples the block from what upstream hosts: ethrex-replay publishes a cache for
Hoodi and nothing else, so any mainnet block is unreachable by the convert-locally
route — producing its cache takes ~4 minutes and ~700 calls against an archive RPC —
and trivial by this one. Hosting our own cache is what lets the converter's tests
assert against the *same* block the benchmarks prove.

## Regenerating it (ethrex rev bumps)

Needed roughly twice a year, when the guest's ethrex `rev` moves and the rkyv
layout changes with it:

```bash
make regen-real-block-fixture     # fetches the cache, rebuilds the .bin in place
sha256sum "$(make -s print-real-block-fixture)"
```

Then upload the result and update `ETHREX_REAL_BLOCK_FIXTURE_SHA256` and its URL.

Directly, against any cache file:

```bash
cd tooling/ethrex-real-block
cargo run --release -- <cache.json> <output_path>
```

Output is deterministic for a given cache file:

```text
wrote ../../executor/tests/ethrex_mainnet_25368371.bin (1110156 bytes): 1 block(s) \
  from mainnet starting at #25368371, 29 transaction(s), 2428684 gas
```

That determinism is enforced, not just documented: `conversion_is_reproducible`
pins the block's stats **and** the fixture's sha256, so the digest keeps the
derivation honest. Verified: regenerating from the cache reproduces
`61eba49b…` byte for byte, which is also what proves the hosted `.bin` and the
hosted cache describe the same block.

## What benchmarks with it

Costs below are for the **current default block** (mainnet 25368371) and move with
it — see [Measured cost of candidate blocks](#measured-cost-of-candidate-blocks).

| Where | How to run it | Cost (current default) |
|---|---|---|
| `benchmark-pr.yml` | `/bench-real` on a PR; automatic on push to main and `workflow_dispatch` | ~6 min |
| `scripts/bench_verify.sh` | `WORKLOAD=real scripts/bench_verify.sh <ref>` | ~6 min per side, then cached |
| `scripts/perf_diff.sh` | `WORKLOAD=real scripts/perf_diff.sh <ref>` | 5 recordings, so ~30 min |

None of them hardcode the fixture path or a block number — they read the path from
`make -s print-real-block-fixture` and run `make ethrex-real-block-fixture` when
the `.bin` is absent.

**Continuations are mandatory, not a tuning choice.** Peak heap on a monolithic
prove grows ~4.9 GB per million cycles on this workload family (measured on the
bench server: `10,728 MB + 2,007 MB/transfer`, R² = 0.998 across 4→20 transfers),
so this block would need **~330 GB** monolithically — and a heavier candidate up to
~700 GB. `--continuations` makes peak heap a function of the epoch size instead of
the trace length, so the same block fits in **~18 GB** at epoch 2^21: a 15.8 GB
epoch working set plus ~60 MB per epoch of accumulated proofs over ~32 epochs. The
bundle on disk is ~1.9 GB; note that a heavier block pushes it past 2 GiB, which
needs rkyv `pointer_width_64` to serialize.

Plain `/bench` deliberately stays on the synthetic 20-transfer block: it proves in
~25s against ~6 min, and one runner carries every `/bench`, `/bench-abba` and
`/bench-verify` in the repo. The synthetic number is a fast screen; the real block
is the number that means something.

Cycle counts here (9.06M synthetic, ~65.6M real) are for a **current guest ELF**.
They move ~14% with ELF vintage — this block reads 74,819,518 on a mid-July ELF — so
pin the ELF whenever you quote one, or it will look like a regression the next time
someone measures.

## Where validation runs

The checks themselves are described under [Validation](#validation) below; this is
where each one executes.

`.github/workflows/ethrex-real-block.yml` runs them on changes to this crate,
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

Because we host both artifacts, the converter's tests assert against the **same**
block the benchmarks prove, so a repoint moves them together. Two files:

**1. The Makefile — the only place a block number appears.** Six lines:

```make
ETHREX_REAL_BLOCK_NETWORK        := <mainnet|hoodi|sepolia>
ETHREX_REAL_BLOCK                := <block number>
ETHREX_REAL_BLOCK_FIXTURE_SHA256 := <sha256 of the .bin>
ETHREX_REAL_BLOCK_CACHE_SHA256   := <sha256 of the cache JSON>
ETHREX_REAL_BLOCK_FIXTURE_URL    := <where you uploaded the .bin>
ETHREX_REAL_BLOCK_CACHE_URL      := <where you uploaded the cache>
```

**2. The test constants**, which are what prove the repoint is self-consistent:
- `src/main.rs` — `CACHE`, `CACHE_MISSING`, the chain-ID import and assert
  (`MAINNET_CHAIN_ID` / `HOODI_CHAIN_ID` / `SEPOLIA_CHAIN_ID` from
  `ethrex_config::networks`), and in `conversion_is_reproducible` the
  `first_block_number` / `transactions` / `gas_used` / digest asserts.
- `tooling/ethrex-tests` — `REAL_BLOCK_FIXTURE`, which points the usability screen
  at the block actually being benchmarked.

Then run `make test-ethrex-real-block-converter`. It passing means the cache, the
`.bin`, the digest and the block stats all describe one block.

Everything else derives from the Makefile — the fixture name, and through
`make -s print-real-block-fixture` the benchmark scripts and `benchmark-pr.yml`. No
workflow, script or env var names a block. Nothing in `executor/.gitignore` needs
touching either: it already ignores every accepted network's fixture name, so a
repointed ~1 MB fixture cannot become committable by accident.

### Measured cost of candidate blocks

Cost is a property of the block, so it changes with the repoint. Cycles are given
for a **current guest ELF**; prove time and heap are for the CPU bench runner at
epoch 2^21, using the 5.31–5.62 s per Mcycle measured on that box.

| block | gas | cycles | prove | peak heap | proof | fixture |
|---|---|---|---|---|---|---|
| **mainnet 25368371** — *current default* | 2.43M | **~65.6M** | **~6 min** | ~18 GB | ~1.9 GB | 1,110,156 B, `61eba49b…` |
| mainnet 25453112 | 4.24M | ~110M | ~10 min | ~19 GB | ~3.7 GB | 2,019,747 B, `0298663d…` |
| hoodi 1265656 | 4.40M | ~147.5M | ~13 min | ~21 GB | ~4.7 GB | 1,021,207 B, `1f7d4c4c…` |

These are measurements, not estimates. All three clear the usability screen. Add a
row rather than editing the wiring.

Two things the table shows that a gas-based estimate would have got wrong. **Cycles
per gas is not constant** — it ranges ~26–38 across these blocks — so ranking
candidates by gas mispredicts cost; 25453112 has *more* gas than hoodi 1265656 yet
costs ~25% fewer cycles. And **fixture size does not track cost** either: the current
default is the cheapest block and the middle-sized fixture.

The default is the cheapest of the three, which matters because this workload sits on
a single shared bench runner. It was also the only block in a 90-day Dune sweep
matching the shape constraints (2 heavy transactions, no whale, sane transfer share)
in the 1.6–2.6M gas band — so it is cheap *and* structurally typical, not cheap
because it is degenerate.

### Verifying a repoint

Run **both**, in this order:

```bash
make test-ethrex-real-block-converter   # cache, .bin, digest and stats agree
make test-ethrex                        # the block is USABLE on the guest
```

The second is not optional and the first cannot replace it: a new block is only
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
`main`, where the type has diverged from the rev our guest pins
(`156cb8d6…`): `main` has an extra `fee_configs` field, moved the type from
`l1::` to `input::`, and uses rkyv 0.8.10 against our exact `=0.8.16`. Its
binary would not deserialize in our guest.

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
runs](#where-validation-runs)); `ethrex-real-block.yml` caches this workspace's
`target/` under its own key, so only cold runs pay it.

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

**`tooling/ethrex-tests` — `test_ethrex_real_block_native`.** Checks the
serialized `.bin` itself deserializes and executes. Since that crate links no KZG
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
