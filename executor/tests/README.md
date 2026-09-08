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

Generate the committed fixtures with:

```bash
make regen-ethrex-fixtures
```

The generator enables Amsterdam in its synthetic genesis and includes the two
EIP-8282 request predeploys required by ethrex 25.

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

The real-block fixture and its source cache are separate, fetched artifacts.
Both must be regenerated and published for the new ethrex revision before the
real-block benchmark and acceptance test can run.
