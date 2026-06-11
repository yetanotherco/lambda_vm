# Executor Test Fixtures

## Ethrex private inputs

The `ethrex_*.bin` files are rkyv-serialized `guest_program::input::ProgramInput`
values consumed by `executor/programs/rust/ethrex`.

The ethrex guest and the native test reference are pinned to:

```text
https://github.com/lambdaclass/ethrex.git
a9de3e8b405dbf406cac31b930fd1ffdc216a429
```

Known fixtures:

```text
ethrex_empty_block.bin
  sha256: 06626a051c07844570feae3cc6dc3831e0143ca81dbb1a56d4bf4e195c0b9411
  contents: stateless ethrex empty block ProgramInput

ethrex_simple_tx.bin
  sha256: 82998bea989ed4aa98b4f4b1476a7d0a4828a4f446cd7edff670418ba330e94b
  contents: stateless ethrex block with one plain ETH transfer transaction
```

The original generation command for these blobs is not recorded in this
repository. A follow-up should add an in-repo crate for generating custom ethrex
block fixtures.
