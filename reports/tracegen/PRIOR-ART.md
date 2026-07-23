# How SP1 / ZisK / RISC0 / OpenVM / Jolt handle the memory walk + precompiles

Research (fetched from primary sources) into how leading RISC-V zkVMs solve exactly our
bottleneck: recovering each memory access's predecessor `(old_value, old_timestamp)` for the
offline-memory-checking argument without a sequential re-walk, and handling precompiles.

## The decisive correction
**None of them do a separate sequential re-walk (which is what we do).** They recover
predecessors either:
- **(A) inline, during the single forward execution pass** — the runtime keeps a live
  `addr → (value, timestamp)` map and writes `prev_value`/`prev_timestamp` into each access
  event *at access time* (an O(1) map write, not a second pass). Used by **SP1** and **OpenVM**.
- **(B) by a parallel sort** — emit all accesses unsorted `(addr, ts, value, rw)`, then
  **sort by (addr, ts)**; predecessor = the previous row per address; a permutation/LogUp
  argument ties the sorted stream to the emitted one. Used by **RISC Zero** (and **ZisK**).

The address-sort that appears in every design is a property of the **AIR/circuit** (how it
checks adjacency), *not* a runtime re-walk. Our mistake is doing predecessor recovery as a
dedicated **second sequential pass**; the industry folds it into the forward pass or a sort.

## Per-system (code/doc-verified)
- **SP1** (Plonky3): fast pass records `MemValue{clk,value}` (the value+ts *before* each access);
  `MemoryReadRecord{prev_timestamp}` / `MemoryWriteRecord{prev_value, prev_timestamp}` filled
  live from the `addr→MemoryEntry` map at access time. Witness-gen phase is a **stateless,
  per-shard, parallel** transform over these events — no re-walk. Precompiles execute **once**,
  emit events carrying **all** their `MemoryRead/WriteRecord`s → never re-read live memory,
  never recomputed. Parallelism = sharding (~2M-cycle shards); cross-shard consistency via an
  associative EC multiset hash + Init/Finalize boundary events.
- **ZisK**: emulator logs the **values read from memory** (a read-log) + register checkpoints;
  pass 2 is **memoryless** (a LOAD pops the next logged read value, doesn't touch live memory)
  → parallel across chunks. Memory consistency by **sorting** accesses `(addr, step)`;
  predecessor = previous sorted row; range-checked increments. Precompiles: input state
  recorded during execution → recomputed **out-of-band in parallel** SMs (not mid-walk).
- **RISC Zero**: keeps memory columns twice (execution order + `(location, cycle)`-sorted);
  grand-product permutation proves they match; predecessor falls out of sorted adjacency.
  Precompiles ("accelerators") share the same sorted memory argument.
- **OpenVM**: `TracingMemory` reads live per-cell `AccessMetadata` at each access to get the
  previous ts/value, updates in place (inline capture, like SP1). Memory bus + LogUp; boundary
  chips per segment (Merkle roots) → segments prove independently.
- **Jolt**: no sort; four multiset fingerprints (init/final/read/write) from committed columns;
  `write ts = read-cts + 1`.

## Precompiles — the uniform rule
Whichever predecessor route, **precompiles record their memory I/O as events *during
execution*** so witness-gen never re-reads or re-simulates live memory. Crucially this is done
by the **generic memory-access instrumentation** (precompiles use the same read/write
primitives), so it is **not a precompile-specific code change** — precompiles are captured for
free. This is exactly what removes our sequential dependency without touching precompile logic.

## Recommendation for us (respecting: crates separate, executor+log changes OK, no precompile changes)
**Adopt SP1/OpenVM route (A): inline predecessor capture on the executor's forward pass.**
- We already thread memory/registers forward in the executor. Add a live `addr→(value,ts)`
  shadow (+ 34 register entries) and record `prev_value`/`prev_timestamp` into each access at
  access time. The executor's forward pass stays O(1)/access.
- This **deletes our `p2a` re-walk**: the prover consumes recorded predecessors and phase 2
  becomes a stateless, per-chunk **parallel** transform.
- **Precompiles untouched** — captured by the same generic memory instrumentation.
- **Memory AIR unchanged** — SP1/OpenVM keep the timestamp-based argument; no constraint change
  (unlike the ZisK/RISC0 sort, which would change the argument).
- Smallest diff for a VM that already threads state; biggest win (removes the whole second pass).

**Alternative route (B): prover-side parallel sort** (RISC0/ZisK) if we prefer zero executor
change — but it still needs the access *values* (precompile inputs) from somewhere, so it
doesn't fully avoid an executor read-log; and verifying the sort adds a permutation argument.
Route (A) is the cleaner fit.

Sources: SP1 (`crates/core/executor/src/events/memory.rs`, jit `context.rs`/`risc.rs`, `vm.rs`,
`record.rs`, precompile events), ZisK (`emulator/src/emu.rs`, `state-machines/mem/*`,
`precompiles/*`), RISC Zero proof-system whitepaper §2.1, OpenVM whitepaper §4.6 +
`system/memory/online.rs`, Jolt (zksecurity "How Jolt works").
