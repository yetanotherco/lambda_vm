# ethrex-block-converter

Converts an `ethrex-replay` cache into the schema-prefixed SSZ stateless input
consumed by `ethrex_guest_program::l1::run_stateless_guest`.

The converter reads the block, raw witness preimages, and public network from
the cache. It builds the Amsterdam `NewPayloadRequest`, recovers one public key
per transaction, carries the raw state/code/header witness, and validates the
serialized result with ethrex's native guest before writing it.

```bash
cd tooling/ethrex-block-converter
cargo run --release -- <cache.json> <output_path>
```

The output starts with the two-byte big-endian schema ID `0x1501`. The current
pinned stateless schema validates one Amsterdam block at a time. A replay
cache must therefore contain one block, its `slot_number`, its
`block_access_list_hash`, and the raw BAL when the BAL is non-empty. Caches
created before Amsterdam are rejected rather than silently rewriting their
block hash or chain rules.

The converter supports empty execution requests, which is the format currently
written by `ethrex-replay` for the public L1 cache. Caches with non-empty
requests are rejected because replay does not persist those request bodies.

## Revision pin

The converter, fixture generator, guest, and host tests all pin:

```text
https://github.com/lambdaclass/ethrex.git
8effcb0671c5d0b12fe0161ea37c174ec4466b6a
```

Keep these pins together. The SSZ wire format and the guest implementation are
coupled to the ethrex revision.

## Real-block fixture

The benchmark fixture is generated rather than fetched, and not by this crate:
`tooling/ethrex-fixtures --bin real_block` rebuilds a real mainnet block as an
Amsterdam block. What is hosted is that block's replay cache, which is
fork-independent. `make ethrex-real-block-fixture` produces the fixture when it is
missing; after an ethrex revision bump, rebuild it and re-baseline its digest:

```bash
make regen-real-block-fixture
sha256sum "$(make -s print-real-block-fixture)"   # -> ETHREX_REAL_BLOCK_FIXTURE_SHA256
```

There is no artifact left to publish: the generator is deterministic, so re-running
it is how the fixture is replaced. The old release assets are rkyv `ProgramInput`
artifacts and are incompatible with the pinned ethrex.

## What the tests cover

Both tests assert a rejection. The pinned Hoodi cache predates Amsterdam and is
refused rather than rewritten, and an unmappable network is refused too. The
success path — cache in, valid SSZ out — has no automated coverage: it needs a
cache from a network that runs Amsterdam, and none is published yet. Until one is,
the encoder is exercised by hand through the `cargo run` above.
