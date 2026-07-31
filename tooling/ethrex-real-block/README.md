# ethrex-real-block

Converts a **real Ethereum block** into a serialized `ProgramInput` `.bin` for
the lambda-vm ethrex guest, reading an [`ethrex-replay`][replay] cache JSON.

This complements `tooling/ethrex-fixtures`, which builds *synthetic* blocks of N
plain ETH transfers. Those are cheap and deterministic but not representative:
they execute no contract code, touch a genesis state trie only a couple of
levels deep, and carry no bytecode in the witness. This tool produces the
opposite — a block that actually looks like Ethereum.

| | `ethrex_bench_20.bin` (synthetic) | `ethrex_hoodi_1265656.bin` (real) |
|---|---|---|
| gas used | 420,000 | **4,402,947** |
| transactions | 20 (all plain transfers) | 11 (7× EIP-1559, 4× EIP-4844 blob) |
| contract calls | 0 | 5, at ~830–955k gas each |
| contract bytecode in witness | 0 | 22 contracts, ~124 KB |
| state trie | 40 accounts, fresh genesis | 1,705 nodes |
| storage keys | 0 | 422 |
| serialized size | 32,766 B | 1,021,207 B |

Note it has *fewer* transactions than the synthetic fixture while being far more
representative — transaction count is not the axis that matters.

The synthetic column is `ethrex_bench_20.bin` as the benchmark scripts actually
generate it — `ethrex-fixtures 20 … distinct`, i.e. 20 distinct genesis-funded
senders to 20 distinct recipients (`scripts/bench_verify.sh`,
`scripts/bench_recursion_scaling.sh`). The same block in `same` mode (one sender,
one recipient) serializes to 16,811 B.

Block 1265656 is **verified to run on the guest's precompile surface** (see
[Validation](#validation)); it needs no accelerator we don't have. Any
replacement block must clear the same check — that is what makes it usable, not
just realistic.

## Prerequisites
Rust (stable) and network access on first run (cargo fetches the pinned ethrex
crates; `make` downloads the cache). **No RV64 target or sysroot needed** — this
is a host tool.

## How to run

From the repo root, the committed default (Hoodi block 1265656):

```bash
make ethrex-real-block-fixture
```

That downloads the cache to `caches/` (gitignored) and writes
`executor/tests/ethrex_hoodi_1265656.bin` (gitignored — ~1 MB, too large to
commit; see `executor/.gitignore`).

The cache URL is pinned to an ethrex-replay **commit**, not to `main`
(`ETHREX_REPLAY_REV` in the Makefile), matching how the guest pins ethrex. A
branch ref would let the benchmark's input change under a fixed fixture name,
which would quietly destroy comparability between runs.

Directly, against any cache file:

```bash
cd tooling/ethrex-real-block
cargo run --release -- <cache.json> <output_path>
```

Output is deterministic for a given cache file:

```text
ethrex_hoodi_1265656.bin
  block:  hoodi #1265656 — 11 transactions, 4,402,947 gas
  sha256: 1f7d4c4cdf9bd52472d9ebafdb4038f57a88c3c92d65c96fd86d7e323db87142
  source: ethrex-replay caches/cache_hoodi_1265656.json @ 2693e018
```

That checksum is enforced, not just documented: `conversion_is_reproducible`
asserts it. The pinned commit keeps the *input* immutable; the digest keeps the
*derivation* honest.

## What benchmarks with it

| Where | How to run it | Cost |
|---|---|---|
| `benchmark-pr.yml` | `/bench-real` on a PR; automatic on push to main and `workflow_dispatch` | ~15 min |
| `scripts/bench_verify.sh` | `WORKLOAD=real scripts/bench_verify.sh <ref>` | ~15 min per side, then cached |
| `scripts/perf_diff.sh` | `WORKLOAD=real scripts/perf_diff.sh <ref>` | 5 recordings, so >1 h |

None of them hardcode the fixture path — they read it from
`make -s print-real-block-fixture` and run `make ethrex-real-block-fixture` when
the (gitignored) `.bin` is absent.

**Continuations are mandatory, not a tuning choice.** Peak heap on a monolithic
prove grows ~3.1 GB per million cycles on this workload family (measured on the
bench server: 2007 MB per transfer at R² = 0.998 across 4→20 transfers), and the
real block is 168.3M cycles — roughly 500 GB. `--continuations` makes peak heap a
function of the epoch size instead of the trace length, so the same block fits in
~16 GB at epoch 2^21.

Plain `/bench` deliberately stays on the synthetic 20-transfer block: it proves in
~25s against ~15 min, and the bench runner is single and serialized across all PRs.
The synthetic number is a fast screen; the real block is the number that means
something.

## Getting a cache for a different block

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

Then point this tool at the resulting JSON. Adopting a different block as the
default is an edit, not a flag. Three of the four edits are the block's identity;
the fourth is the digest that proves the conversion still matches.

**1. The Makefile — this is what moves every benchmark.** Two lines:

```make
ETHREX_REAL_BLOCK_NETWORK := hoodi      # -> mainnet
ETHREX_REAL_BLOCK         := 1265656    # -> 25453112
```

Everything downstream is derived from them: the cache path, the download URL, the
fixture name, and — through `make -s print-real-block-fixture` — the benchmark
scripts and `benchmark-pr.yml`. Nothing else names the block. Re-pin
`ETHREX_REPLAY_REV` in the same edit if the new cache landed in a later commit.

**2. `src/main.rs`** — `CACHE` and the block assertions in
`conversion_is_reproducible`, including the **sha256**. The digest cannot be
guessed: run `make ethrex-real-block-fixture`, take the printed `sha256:` line,
and paste it in. That is the check that keeps the derivation honest.

**3. `tooling/ethrex-tests`** — `REAL_BLOCK_FIXTURE`.

**4. Nothing in `executor/.gitignore`** — it already ignores every accepted
network's fixture name, so a repointed ~1 MB fixture cannot become committable by
accident.

Benchmark comparability does not survive the swap, and that is intentional rather
than a wrinkle to work around: `benchmark-pr.yml` records which block it measured
and refuses to diff a PR against a baseline that measured a different one, so the
first run after a repoint reports one-sided numbers until main republishes.

Then run **`make test-ethrex`**, not just this crate's tests. A new block is only
usable if it needs no accelerator the guest lacks, and this crate cannot tell you
that — its graph links a working c-kzg, so a block calling point evaluation (0x0a)
passes here and fails in the guest. `test_ethrex_real_block_native` in
`tooling/ethrex-tests` is the screen. See [Validation](#validation).

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
and `ark-ff/asm` — not the tests. CI caches this workspace's `target/` (see the
`workspaces:` list on the `test-executor` job's `rust-cache` step), so only cold
runs pay it.

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
