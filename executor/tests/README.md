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
cargo run --release -- 4  ../../executor/tests/ethrex_bench_4.bin distinct  # recursion profile
```

`ethrex_bench_4.bin` is the odd one out: `distinct` mode, and it is read by the
recursion profile target rather than the executor tests (see the Makefile's
`recursion-profile-block-input`). It is committed like the rest, so it is
regenerated and checksummed with them — a rev bump makes every one of these
undecodable, not just the three the executor reads.

To regenerate after an ethrex rev bump, update the `rev` in
`tooling/ethrex-fixtures/Cargo.toml` (and the guest's), then run
`make regen-ethrex-fixtures` from the repo root. The target rebuilds the
committed fixtures and refreshes the checksums below.

Known fixtures:

```text
ethrex_empty_block.bin
  sha256: 8d6f6061c71c23fad1d5dee26242d631efe0bff8d7f49422c2ba4cde9d4be919
  contents: stateless ethrex empty block ProgramInput (0 transactions)

ethrex_simple_tx.bin
  sha256: c40bce364f22758ab7fa6fe8b45ce4c305dee5add4536ef6dca0e74e410e2729
  contents: stateless ethrex block with one plain ETH transfer transaction

ethrex_10_transfers.bin
  sha256: 4d862e8537284729ff11c7bcf91c971e562dd6bbce2a1e181ba5bf48cb6b65cf
  contents: stateless ethrex block with ten plain ETH transfer transactions

ethrex_bench_4.bin
  sha256: 03ed0d175622af6ef9a981d7652ba7c86630b9473f49cae17edf649724b704e1
  contents: stateless ethrex block with four plain ETH transfers, `distinct` mode
            (N senders -> N recipients); read by the recursion profile target
```

## Real-block fixtures

The blocks above are synthetic (N plain ETH transfers over a small genesis).
For a representative workload — real contract execution, real trie depth, real
bytecode — `make ethrex-real-block-fixture` downloads
`ethrex_mainnet_25368371_4f658c2b.bin` (1,110,165 B) from the `bench-fixtures-v1` release
and verifies it against `ETHREX_REAL_BLOCK_FIXTURE_SHA256` in the Makefile before
moving it into place. It is gitignored rather than committed, so the checksum
lives next to the URL in the Makefile rather than in the table above (the checksum
script only covers committed fixtures). See
`tooling/ethrex-block-converter/README.md` for how the fixture is produced and
repointed.
