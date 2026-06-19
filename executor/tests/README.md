# Executor Test Fixtures

## Ethrex private inputs

The `ethrex_*.bin` files are rkyv-serialized `ethrex_guest_program::l1::ProgramInput`
values consumed by the ethrex guest (`executor/programs/rust/ethrex`).

The ethrex guest, the native test reference, and the fixture generator are all
pinned to the same ethrex revision (the open LambdaVM-backend PR branch, until it
merges to `main`):

```text
https://github.com/lambdaclass/ethrex.git
156cb8d6a3974f411d71622eecd1b249ee37ff1c
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
  sha256: d3e594f07cc74e4ddc9db9e9db220a65a2d2e578b619fc3ce06e346007b3ca43
  contents: stateless ethrex empty block ProgramInput (0 transactions)

ethrex_simple_tx.bin
  sha256: 15e3b3efa434186682537755d828ac8bbdde4be3fc7cbe34f26687b618a6c6ab
  contents: stateless ethrex block with one plain ETH transfer transaction

ethrex_10_transfers.bin
  sha256: 38901ee4d40b99cf0aa7f642a92f0fc8db76d974bf43033a1673839020c3c28e
  contents: stateless ethrex block with ten plain ETH transfer transactions
```
