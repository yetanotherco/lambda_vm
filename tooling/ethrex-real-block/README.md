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
| state-trie nodes | a handful | 1,705 |
| storage keys | 0 | 422 |
| serialized size | 17 KB | 1,021,207 B |

Note it has *fewer* transactions than the synthetic fixture while being far more
representative — transaction count is not the axis that matters.

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

That checksum is documentation of what the fixture should be, not an enforced
gate — the pinned commit is what makes the input immutable, and
`conversion_is_reproducible` pins the block's stats and serialized length.

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

Then point this tool at the resulting JSON. To make a new block the default,
override `ETHREX_REAL_BLOCK` (Makefile) or add a rule alongside it.

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

Three checks, cheapest first. The first two run on the host in milliseconds and
need no RV64 toolchain.

**`cargo test` here — `real_block_executes_under_guest_crypto`.** The screen for
*"does this block need an accelerator we don't have?"*. It builds
`ethrex-guest-program` with `default-features = false, features = ["lambdavm"]`
and executes with `LambdaVmEcsmCrypto` — byte-for-byte the guest's own
configuration. Stateless re-execution ends in a post-state-root check, so a
block reaching a precompile the guest doesn't link diverges from consensus and
fails here. KZG point evaluation (0x0a) is the notable omission: `lambdavm` does
not pull `ethrex-crypto/kzg-rs` (only `sp1` does).

Getting this configuration right matters. Under the default `secp256k1` feature
the same test passes while exercising a *richer* precompile set than the guest
ships — green, and meaningless.

**`cargo test` here — `conversion_is_reproducible`.** Pins the block's stats and
the serialized length, so an ethrex rev bump that moves the rkyv layout is
caught rather than silently producing a fixture the guest can't read.

**`tooling/ethrex-tests` — `test_ethrex_real_block_native`.** Checks the
serialized `.bin` itself deserializes and executes. That crate builds
`ethrex-guest-program` with default features, so this covers the artifact, not
the guest's precompile surface.

**`tooling/ethrex-tests` — `test_ethrex_real_block_vm`** (`#[ignore]`, excluded
from PR CI). The block through the guest ELF, comparing the VM's committed
output against the native reference. Needs the RV64 toolchain and its runtime is
unmeasured; run it on a build server, not a laptop:

```bash
cd tooling/ethrex-tests && cargo test --release test_ethrex_real_block_vm -- --ignored
```

[replay]: https://github.com/lambdaclass/ethrex-replay
