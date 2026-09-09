# ethrex-fixtures

Generates deterministic synthetic ethrex blocks and writes the
schema-prefixed SSZ stateless input consumed by the LambdaVM guest. The tool
builds an in-memory chain, creates signed ETH transfers, generates the raw
execution witness, and validates the resulting input with ethrex's native
stateless guest before writing it.

The ethrex revision is pinned to the same commit as the guest. After changing
that pin, regenerate the committed fixtures with `make regen-ethrex-fixtures`.

```bash
cd tooling/ethrex-fixtures
cargo run --release -- <n_transfers> <output_path> [same|recipients|distinct]
```

`same` uses one funded sender, `recipients` sends to distinct recipients, and
`distinct` uses deterministic funded senders. The standard fixtures are:

```bash
cargo run --release -- 0  ../../executor/tests/ethrex_empty_block.bin
cargo run --release -- 1  ../../executor/tests/ethrex_simple_tx.bin
cargo run --release -- 10 ../../executor/tests/ethrex_10_transfers.bin
cargo run --release -- 4  ../../executor/tests/ethrex_bench_4.bin distinct
```

The generator is host-only; it needs no RV64 target or sysroot. Output is
deterministic for a given transfer count and mode.

## The benchmark block: `--bin real_block`

The synthetic blocks above are transfers, which is not what a real block costs:
a mainnet block is keccak- and trie-bound, and a prover change can move the two
numbers in opposite directions. `real_block` produces the benchmark workload
from a real block instead — its own transactions, its own accounts, its own
contract code — rebuilt as an Amsterdam block the guest accepts:

```bash
cargo run --release --bin real_block -- <ethrex-replay-cache.json> <output.bin>
# or, with the cache fetched for you:
make regen-real-block-fixture
```

It reads an ethrex-replay cache, installs the block's pre-state into an
in-memory store **keyed the way the tries are keyed** — `keccak(address)` and
`keccak(slot)`, so no key preimages are needed and every account and slot the
block touches arrives intact — registers a parent header at the block's real
height, replays the transactions in the block's own order through the t8n
entry point, and validates the result through the native guest before writing.

Output for mainnet 25453112 (38 transactions, 4,238,394 gas):

```
installed   132 accounts / 261 storage slots / 132 codes
rebuilt     #25453112 (38/38 txs, 3761976 gas, 10 reverted)
```

### Why 10 transactions revert, and why that is not a defect here

Those transactions were signed with gas limits computed under Osaka. Amsterdam
changes the gas model: cold account access goes from 2600 to 3000, and EIP-8037
carves a state-gas reservoir out of the transaction's excess gas limit. A
transfer sent with a limit of exactly 21,000 has no excess to carve, so it runs
out of gas. Running the same rebuild under Osaka rules — `REAL_BLOCK_FORK=osaka`,
which prints the totals and exits without writing, since the guest only decodes
the Amsterdam schema — is what separates the fork from the reconstruction: on
block 25368371 it reverts 2 transactions where Amsterdam reverts 12.

Every pre-Amsterdam block loses transactions this way. A screen of twelve real
mainnet blocks (both of our release caches plus ethrex's curated zkevm_bench
corpus) put the revert share between 26% and 50% with no exceptions, so it is not
a criterion for picking one.

Reverting is separate from being *dropped*: a reverted transaction was applied and
paid for its gas, while a dropped one never entered the block. This block applies
all 38 of its transactions, and the generator refuses to write a fixture that
applies fewer, because every other guard would still pass — a block with fewer
transactions is a valid block, so the loss would show up only as a smaller
benchmark. Screening candidate blocks does need the partial ones, so pass
`REAL_BLOCK_ALLOW_DROPS=1` for that; note also that the generator is not universal
(block 25087563 fails with `StateRootMismatch`), which is why screening is a
required step before pinning a different block. What the screen was for is weight: the retired
fixture cost 30.50M cycles on today's guest, and this block rebuilds to 37.14M
(+22%), the closest of the twelve — 25368371 comes in at -33% and the next
candidate up, 25087308, at +197%. Several blocks end up consuming *more* gas than
they did on mainnet (+7% to +22%), because Amsterdam makes the surviving
transactions dearer.

So what this fixture is, is a *real-mix* Amsterdam block — real contract code,
real calldata, real signatures, real trie depth — and not a replay of mainnet
economics. For workloads whose gas limits were computed for Amsterdam, use the
EEST benchmark fixtures; `ETHREX_BENCH_WORKLOAD_AFTER_BUMP.md` in the repository
root has both sets measured side by side.

Measured on the guest ELF at ethrex `2cb18b0b`: 37,137,386 cycles, 6,003 keccak
calls, 164 ECSM calls. Fixture: 549,144 bytes.

### Choosing the epoch size, and what the workload costs

Measured on the bench runner (`vm-benchmarks-1`, which is also the CI
self-hosted `bench` runner: 96 cores / 125 GB, idle) on 2026-09-08, with the
guest ELF at ethrex `2cb18b0b`:

| | mean | sd | CV | peak RSS | proof | verify |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| epoch 2^22, 5 proves | **125.17 s** | 0.71 s | 0.57 % | 43.00 GiB | 790 MB | 12.0 s |

2^22 is what `/bench`, `/bench-abba` and the GPU bench pin. The epoch trade-off
itself was swept on the previous block (2^21 costs +13.7 % of wall to save
12 GiB; 2^23 buys −7.5 % for +16 GiB): peak RSS is set by the epoch size rather
than by the block, so that shape carries over even though the seconds do not.
2^23 would take this workload past 50 GiB against the runner's 64 GiB floor,
which is why memory and not speed picks the default.

With that sd, a two-sided 95 % comparison resolves ~2.00 % at three runs per
side, ~1.00 % at five and ~0.58 % at ten — which is why a sub-2 % claim needs
`/bench N` or the ABBA tiebreaker rather than a re-read of a three-run table.

Continuations are not optional here: monolithic proving costs ~4.9 GB of peak
heap per million cycles on this family, so 37.14M cycles would need ~182 GB.
