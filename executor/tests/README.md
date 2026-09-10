# Executor Test Fixtures

The `ethrex_*.bin` files are schema-prefixed SSZ stateless inputs consumed by
`ethrex_guest_program::l1::run_stateless_guest`. The first two bytes are the
Amsterdam schema ID (`0x1501`); the body contains the payload, public keys, and
execution witness.

The guest, native reference tests, and fixture generator all use this ethrex
commit:

```text
https://github.com/lambdaclass/ethrex.git
2cb18b0b95b27a2555d3debffdebc43c9685d6e3
```

Five manifests carry that pin, not one. `scripts/set_ethrex_rev.sh --show` prints
it and fails if they ever disagree.

The generator enables Amsterdam in its synthetic genesis and includes the two
EIP-8282 request predeploys required by ethrex 25.

### Generation

These blobs are generated reproducibly by the in-repo tool
`tooling/ethrex-fixtures` (in-memory, offline — no RPC). It builds a synthetic
block with N signed ETH transfers from a funded genesis account and serializes
the resulting SSZ stateless input:

```bash
cd tooling/ethrex-fixtures
cargo run --release -- 0  ../../executor/tests/ethrex_empty_block.bin   # empty block
cargo run --release -- 1  ../../executor/tests/ethrex_simple_tx.bin     # 1 transfer
cargo run --release -- 10 ../../executor/tests/ethrex_10_transfers.bin  # 10 transfers
cargo run --release -- 4  ../../executor/tests/ethrex_bench_4.bin distinct  # recursion profile
```

or all four at once with `make regen-ethrex-fixtures` from the repo root.

`ethrex_bench_4.bin` is the odd one out: `distinct` mode, and it is read by the
recursion profile target rather than the executor tests (see the Makefile's
`recursion-profile-block-input`). It is committed like the rest, so it is
regenerated and checksummed with them — a rev bump makes every one of these
undecodable, not just the three the executor reads.

To regenerate after an ethrex rev bump, re-pin with
`scripts/set_ethrex_rev.sh <40-char-sha>` (all five manifests at once), regenerate
the five locks, then run `make regen-ethrex-fixtures`.

The checksums below are refreshed by that same run, so they catch a hand-edited
`.bin` but never one that is stale against the pinned rev. `--show` is what
catches the cause.

Known fixtures:

```text
ethrex_empty_block.bin
  sha256: d914d36e673dc0e24bc4e105f3037e78305e63f6121e1937058dcc704fabbb8e
  contents: stateless ethrex empty block (0 transactions)

ethrex_simple_tx.bin
  sha256: 4dd4ab89d904981844f28b093fde0ed18ffa4d61273482eb8592d41db6a38e7d
  contents: stateless ethrex block with one plain ETH transfer

ethrex_10_transfers.bin
  sha256: e86c5fc80b8b603c4a58fd6ab6ce5bbb40d378c67c8b65f4d89a15b01f69fc6f
  contents: stateless ethrex block with ten plain ETH transfers

ethrex_bench_4.bin
  sha256: dbfe0d808ff9476ef70bfd4459b82330a2dc038bdfdf2808447ed04012556386
  contents: stateless ethrex block with four distinct plain ETH transfers
```

## Real-block fixture

The blocks above are synthetic. For a representative workload — real contract
execution, real trie depth, real bytecode — `make ethrex-real-block-fixture`
BUILDS `ethrex_mainnet_25453112.bin` from the block's replay cache, which is the
only fetched artifact; nothing about the fixture is published, because ethrex 25's
guest decodes only the Amsterdam schema and no hosted artifact for a pre-Amsterdam
block can be valid. It is gitignored rather than committed, so its digest lives
next to the block pin in the Makefile rather than in the table above, and it is
verified on every use. See `tooling/ethrex-fixtures/README.md` for what that
workload is, what it costs, and what it is not.
