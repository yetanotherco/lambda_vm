# ethrex-fixtures

Generates synthetic **ethrex block fixtures** — serialized `ProgramInput` `.bin`
files — for the lambda-vm prover tests and benchmarks. Fully in-memory and
offline (no RPC, no node): it builds a genesis chain, creates a block with a
chosen number of signed ETH-transfer transactions, runs ethrex's stateless
witness generation, and writes the rkyv-encoded `ProgramInput`.

The ethrex dependency is pinned to the **same revision as the guest**
(`executor/programs/rust/ethrex`), so the produced fixtures deserialize and
execute in the guest. When you bump the guest's ethrex rev, bump the `rev` in
this crate's `Cargo.toml` too and regenerate.

## Prerequisites
- Rust (stable) and network access (the first build fetches the pinned ethrex
  crates). **No RV64 target or sysroot needed** — this is a host tool.

## How to run

```bash
cd tooling/ethrex-fixtures
cargo run --release -- <n_transfers> <output_path> [mode]
```

- `<n_transfers>` — how many ETH transfers to include in the block (`0` = empty
  block).
- `<output_path>` — where to write the `.bin` (relative to this directory).
- `[mode]` — account diversity (optional, default `same`):
  - `same` — one funded sender (`RICH_PK`) → one fixed recipient (`0xdeadbeef`).
  - `recipients` — one funded sender → N distinct recipients (1 → N fan-out).
  - `distinct` — N distinct, genesis-funded senders → N distinct recipients
    (N independent 1-1 pairs; senders are deterministic synthetic keys injected
    into the genesis allocation). This is what the CI benchmark uses, since the
    state-trie witness for many distinct accounts is closer to a real block.

It prints the output size, the number of transactions, and the mode, e.g.:

```
wrote ../../executor/tests/ethrex_simple_tx.bin (12745 bytes): block #1 with 1/1 transfer(s) [1 sender -> 1 recipient]
```

Output is deterministic for a given `(n_transfers, mode)`.

## Creating blocks with different numbers of transactions

Just change the first argument:

```bash
# empty block (0 transactions)
cargo run --release -- 0 ../../executor/tests/ethrex_empty_block.bin

# 1 transfer
cargo run --release -- 1 ../../executor/tests/ethrex_simple_tx.bin

# 10 transfers
cargo run --release -- 10 ../../executor/tests/ethrex_10_transfers.bin

# 50 transfers (custom)
cargo run --release -- 50 /tmp/ethrex_50_transfers.bin
```

For committed fixtures, prefer `make regen-ethrex-fixtures` from the repo root;
it regenerates the standard fixtures and refreshes
`executor/tests/README.md` checksums.

> Note: bigger blocks cost ~4M cycles per transfer (software ecrecover
> dominates), so they execute fine but may be too heavy to *prove* on a typical
> machine — e.g. 10 transfers ≈ 42M cycles.

## Details
- Transactions are plain ETH transfers. In `same`/`recipients` mode they are
  signed by a funded dev account from `genesis.json` (well-known load-test key —
  not a secret); in `distinct` mode each is signed by its own synthetic key,
  funded by injecting an entry into the genesis allocation. Output is
  deterministic in all modes.
- Currently only ETH transfers are supported. (ERC20 / contract calls would be
  a future extension.)
- Once the upstream LambdaVM-backend ethrex PR merges, this tool can be replaced
  by `ethrex-replay custom block` on ethrex `main`.
