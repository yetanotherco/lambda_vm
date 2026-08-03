# Executor Test Fixtures

## Ethrex private inputs

The `ethrex_*.bin` files are rkyv-serialized `ethrex_guest_program::l1::ProgramInput`
values consumed by the ethrex guest (`executor/programs/rust/ethrex`).

The native-reference tests live in `tooling/ethrex-tests` (a detached
workspace: ethrex pins rkyv `unaligned`, which must not feature-unify with the
main workspace's aligned proof format).

The ethrex guest, the native test reference, and the fixture generator are all
pinned to the same ethrex revision (the open LambdaVM-backend PR branch, until it
merges to `main`):

```text
https://github.com/lambdaclass/ethrex.git
4f658c2b3d10e3f21d35ce546870f55ca3f940fc
```

### Generation

These blobs are generated reproducibly by the in-repo tool `tooling/ethrex-fixtures`
(in-memory, offline — no RPC). It builds a synthetic block with N signed ETH
transfers from a funded genesis account and serializes the resulting
`ProgramInput`:

```bash
cd tooling/ethrex-fixtures
cargo run --release -- 0  ../../executor/tests/ethrex_empty_block.bin   # empty block
cargo run --release -- 1  ../../executor/tests/ethrex_simple_tx.bin     # 1 transfer
cargo run --release -- 10 ../../executor/tests/ethrex_10_transfers.bin  # 10 transfers
```

To regenerate after an ethrex rev bump, update the `rev` in
`tooling/ethrex-fixtures/Cargo.toml` (and the guest's), then run
`make regen-ethrex-fixtures` from the repo root. The target rebuilds the
committed fixtures and refreshes the checksums below.

Known fixtures:

```text
ethrex_empty_block.bin
  sha256: ca7454142f13db5d04356366e3917d264cbea709ddd1b5e6526720dbaa12064d
  contents: stateless ethrex empty block ProgramInput (0 transactions)

ethrex_simple_tx.bin
  sha256: 29c1607297d21d88accb33767c8e03eb6536d746e6b2399930afdc9d67f1fe3f
  contents: stateless ethrex block with one plain ETH transfer transaction

ethrex_10_transfers.bin
  sha256: d04cfed35bd16c8248ab8ebf2f3ca2ff01c08269271b17e5d5db3b1f22ea03ad
  contents: stateless ethrex block with ten plain ETH transfer transactions
```

## Real-block fixtures

The blocks above are synthetic (N plain ETH transfers over a small genesis).
For a representative workload — real contract execution, real trie depth, real
bytecode — `make ethrex-real-block-fixture` downloads
`ethrex_mainnet_25368371.bin` (1,110,165 B) from the `bench-fixtures-v1` release
and verifies it against `ETHREX_REAL_BLOCK_FIXTURE_SHA256` in the Makefile before
moving it into place. It is gitignored rather than committed, so the checksum
lives next to the URL in the Makefile rather than in the table above (the checksum
script only covers committed fixtures). See
`tooling/ethrex-block-converter/README.md` for how the fixture is produced and
repointed.
