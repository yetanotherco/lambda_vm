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

The benchmark fixture is a fetched release artifact, not a build dependency:

```bash
make ethrex-real-block-fixture
```

After an ethrex revision bump, regenerate it from the matching cache:

```bash
make regen-real-block-fixture
sha256sum "$(make -s print-real-block-fixture)"
```

Publish the resulting SSZ artifact and matching Amsterdam cache, then update
their URLs and checksums in the Makefile before enabling the real-block
benchmark. The old release assets are rkyv `ProgramInput` artifacts and are
incompatible with the pinned ethrex.

The converter's tests use the Hoodi cache under `caches/` and verify that an
unmapped network is rejected. The checked release cache predates Amsterdam, so
it is intentionally rejected until a cache containing the new payload fields
is published.
