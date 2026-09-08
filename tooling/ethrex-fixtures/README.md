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

Output for mainnet 25368371 (29 transactions, 2,428,684 gas):

```
installed   91 accounts / 117 storage slots / 91 codes
rebuilt     #25368371 (29/29 txs, 1803938 gas, 12 reverted)
```

### Why 12 transactions revert, and why that is not a defect here

Those transactions were signed with gas limits computed under Osaka. Amsterdam
changes the gas model: cold account access goes from 2600 to 3000, and EIP-8037
carves a state-gas reservoir out of the transaction's excess gas limit. A
transfer sent with a limit of exactly 21,000 has no excess to carve, so it runs
out of gas. The same rebuild under Osaka rules — `REAL_BLOCK_FORK=osaka`, which
prints the totals and exits without writing, since the guest only decodes the
Amsterdam schema — reverts **2** transactions instead of 12:

| rules | txs | gas | reverted |
| --- | --- | ---: | ---: |
| Osaka (what the block was built for) | 29/29 | 1,473,522 | 2 |
| Amsterdam | 29/29 | 1,803,938 | 12 |
| mainnet, as mined | 29 | 2,428,684 | — |

So the gap is the fork, not the reconstruction: a pre-Amsterdam block cannot be
a faithful Amsterdam workload at any level of tooling effort. What this fixture
is, therefore, is a *real-mix* Amsterdam block — real contract code, real
calldata, real signatures (the accelerator counters agree exactly with the
retired fixture at 116 ECSM calls), real trie depth — and not a replay of
mainnet economics. For workloads whose gas limits were computed for Amsterdam,
use the EEST benchmark fixtures; `ETHREX_BENCH_WORKLOAD_AFTER_BUMP.md` in the
repository root has both sets measured side by side.

Measured on the guest ELF at ethrex `2cb18b0b`: 20,360,647 cycles, 2,701 keccak
calls, 116 ECSM calls.
