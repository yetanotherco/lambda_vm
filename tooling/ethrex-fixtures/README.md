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
`distinct` uses deterministic funded senders. All three emit plain value transfers and
nothing else, which is what keeps the generated fixtures out of reach of a precompile the
guest does not implement — this crate links a working c-kzg that the guest has no backend
for, and only `tooling/ethrex-tests` (no KZG linked) screens for that. Adding a mode that
executes contract code means adding that screen; see the invariant in `src/main.rs`.

The standard fixtures are:

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

It reads an ethrex-replay cache, installs the block's own witness tries into an
in-memory store **as they are** — every node at its path, pruned siblings left as
hashes, so the installed state trie hashes to the real parent's state root and
keeps mainnet's depth — adds the two EIP-8282 predeploys Amsterdam requires,
registers a parent header at the block's real height, replays the transactions in
the block's own order through the t8n entry point, and validates the result
through the native guest before writing.

The depth is the point. The guest hashes every node on the path from each
touched account and slot to the root, and mainnet's trie puts accounts ~9 levels
down. Re-inserting only the block's leaves into a fresh trie, which is what this
tool did before, leaves ~275 accounts 3-4 levels deep: on this block that is
20.36M cycles instead of 30.50M, with the keccak share cut by more than half.

Output for mainnet 25368371, whose 29 transactions consumed 2,428,684 gas on mainnet
under Osaka (the `rebuilt` line reports what they consume here, under Amsterdam):

```
installed   93 accounts / 117 storage slots / 91 codes
rebuilt     #25368371 (29/29 txs, 1803938 gas, 12 reverted)
```

### Why 12 transactions revert, and why that is not a defect here

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
all 29 of its transactions, and the generator refuses to write a fixture that
applies fewer, because every other guard would still pass — a block with fewer
transactions is a valid block, so the loss would show up only as a smaller
benchmark. Screening candidate blocks does need the partial ones, so pass
`REAL_BLOCK_ALLOW_DROPS=1` for that; note also that the generator is not universal,
which is why screening is a required step before pinning a different block. A
transaction that under Amsterdam reads state its mainnet execution never touched
hits a pruned node, and the generator refuses the block (25087407, 25087416 and
25087554 do this); 25087563 reads a BLOCKHASH deeper than the cache's headers.
Several blocks end up consuming *more* gas than they did on mainnet (+7% to +22%),
because Amsterdam makes the surviving transactions dearer.

What the screen measures is shape. Rebuilt on their witness tries, five 24-60M-gas
blocks put keccak/Mcycle · ecsm/Mcycle · keccak/ecsm at a geometric mean of
174 · 2.88 · 60.4, each 0.10-0.37 from it (Euclidean distance in log space). This
block lands at 213 · 3.80 · 56.0, 0.35 away, inside that spread; 25453112, the
previous pin, lands at 0.26. Small blocks sit ~20% high on keccak/Mcycle, likely
because a big block's touched keys share the top of the trie.

So what this fixture is, is a *real-mix* Amsterdam block — real contract code,
real calldata, real signatures, mainnet's trie depth — and not a replay of mainnet
economics. The rebuilt parent is not the real one (the two predeploys change its
state root), so the two MEV transactions that check `blockhash(n-1)` revert. For workloads whose gas limits were computed for Amsterdam, use the
EEST benchmark fixtures upstream publishes with `tests-zkevm-benchmark`: they carry
`statelessInputBytes` the guest reads as-is, one dimension stressed per fixture,
which is the part a real block does not give.

Measured on the guest ELF at ethrex `8effcb06`: **30,497,198 cycles**, 6,492
keccak calls, 116 ECSM calls, identical on macOS/arm64 and the Linux x86-64 runner.
Fixture: 506,993 bytes.

**Pin the ELF whenever you quote a cycle count.** The three counts are
deterministic for a given ELF and input, and they move with anything that changes
the guest — including changes that touch none of the source it compiles. Moving
the pin from `2cb18b0b` to `8effcb06` cost **+362 cycles, 0.001 %**, with keccak
and ECSM identical, and that was the whole difference: no `.rs` file in the
guest's dependency graph differs between the two revs and no crates.io dependency
moved, so what shifted is the version metadata the ethrex crates carry. Immaterial
next to the ~1 % this workload can resolve, but it does mean an exact count
belongs to an exact rev.

The compiler is not pinned either — the guest embeds C (`secp256k1-sys`) and the
Makefile pins target flags but not `cc` — so in principle two boxes with different
clang majors can disagree. In practice the effect is small and not well
characterised: the one figure recorded in this repo is 0.13 % on a different
block, while the previous pin (25453112, shallow rebuild) came out at exactly
37,137,386 on both macOS/arm64 and the Linux x86-64 runner, from ELF binaries that
were themselves different. Quote the rev; do not assume the machine matters, and do not assume it
does not.

### Choosing the epoch size, and what the workload costs

Measured on the bench runner (`vm-benchmarks-1`, which is also the CI
self-hosted `bench` runner: 96 cores / 125 GB) with the guest ELF at ethrex
`8effcb06`, the cli built with `jemalloc-stats`, 5 proves interleaved with 5 of
the previous workload (25453112 on the shallow rebuild, 37.14M cycles) on the same
binary, one verify after each:

| epoch 2^22 | wall mean | sd | CV | epochs | peak RSS | proof | verify |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **this block**, 5 proves | **110.31 s** | 0.77 s | 0.70 % | 8 | 42.0-46.8 GiB | 641.7 MB | 10.65 s |
| previous workload, 5 proves | 126.08 s | 0.61 s | 0.49 % | 9 | 40.9-43.9 GiB | 748.5 MB | 12.08 s |

Every pair came out 12.0-13.6 % faster on this block, and the proof is 14.3 %
smaller; peak RSS and jemalloc peak heap overlap between the two, as expected when
the epoch size and not the block sets them. An earlier 14-prove sitting on the
previous workload at `2cb18b0b` gave 125.33 s (sd 1.58 s), in line with its row.

2^22 is what `/bench`, `/bench-abba` and the GPU bench pin. The epoch trade-off
itself was swept on the previous block (2^21 costs +13.7 % of wall to save
12 GiB; 2^23 buys −7.5 % for +16 GiB): peak RSS is set by the epoch size rather
than by the block, so that shape carries over even though the seconds do not.
2^23 would take this workload past 50 GiB against the runner's 64 GiB floor,
which is why memory and not speed picks the default.

Both rows ran on a quiet box (every prove got 74-76 cores' worth of CPU); on a
shared one the spread widens with how much of it each prove gets, as the earlier
sitting showed. Wall time here is a function of CPU share,
not of the prover, so quote a spread only together with the condition it was
measured under — `scripts/bench_abba.sh` records the CPU share of every prove and
flags a contended batch, and its comments carry the measurement. A two-sided 95 %
comparison resolves ~0.6 % at three runs per side on a quiet box and ~2.0 % on a
busy one, which is why a sub-2 % claim needs `/bench N` or the ABBA tiebreaker
rather than a re-read of a three-run table.

Continuations are not optional here: monolithic proving costs ~4.9 GB of peak
heap per million cycles on this family, so 30.50M cycles would need ~150 GB.
