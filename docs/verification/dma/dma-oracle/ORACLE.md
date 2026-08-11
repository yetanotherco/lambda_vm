# DMA memcpy oracle

Independent reference model for the DMA memcpy ecall, and the record of what it
is anchored on.

## 1. Validation status: **VALIDATED**

Run 2026-08-11, `python3 test_oracle.py` (full, no `--quick`):

```
[1] libc memmove                     PASS  3855 cases x overlap/alignment
[2] CPython slice assignment         PASS  3855 cases x overlap/alignment
[3] row/bus level <-> byte level     PASS  257 lengths x 15 overlaps
[4] guest stub chunking              PASS  1100 lengths
[5] mutation sweep                   PASS  all 6 mutants caught
VALIDATION STATUS: VALIDATED
  emitted 10 canonical vectors -> canonical_dma_vectors.json
```

Anchors 1 and 2 are **genuinely non-circular**: the platform C library and
CPython's `bytearray` slice assignment are two implementations of `memmove` that
share no code with `dma_ref.py` and no code with each other. libc in particular
is the definition the guest's `compiler_builtins` `memcpy` was replacing, which
is what makes it the right anchor for this campaign rather than a convenient one.

Anchor 3 is the one the chip depends on, and it has no external counterpart: it
is the only check that the **row sequence the AIR proves** is the **byte copy the
guest asked for**. Anchor 5 is what makes 1–4 worth running.

The harness reports what actually ran. A missing dependency SKIPs only its own
anchor, never cascades, and the banner names the anchors it is *not* anchored on
— the two defects the BLAKE3 harness had to have fixed after the fact
(PR #903's `thoughts/blake3/README.md`, "Harness defects — FIXED").

## 2. The four levels

`dma_ref.py` deliberately writes each level as its own function rather than
sharing a helper, so they can be checked against each other.

### 2.1 Byte level — `memcpy_ref(memory, dst, src, n)`

The C `memmove` contract on a sparse byte-addressed memory. Snapshot the whole
source, then write. Unwritten memory reads as zero, matching the VM.

Preconditions, in the executor's own order (`validate`):

```
n > 256                 -> reject (DmaMemcpyChunkTooLarge)
dst + n >= 2^64         -> reject (AddressOverflow)
src + n >= 2^64         -> reject (AddressOverflow)
```

The order matters: an oversized call that would *also* wrap reports the chunk
error, and the audit script asserts the Rust rejects in that order too.

### 2.2 Row level — `row_widths(n)`, `row_decomposition(...)`

```python
widths, remaining = [], n
while remaining != 0:
    width = 8 if remaining >= 8 else 1
    widths.append(width)
    remaining -= width
```

Written as the greedy loop, **not** as the closed form
`[8]*(n//8) + [1]*(n%8)`, because the AIR decides one row at a time from the
remaining count (`tail = count < 8`). That the closed form agrees is a property
anchor 3 checks (`(d)`), not an assumption the model makes.

`row_decomposition` adds the two flag columns and one terminal row:

| | `src` | `dst` | `count` | `first` | `end` |
|---|---|---|---|---|---|
| data row k | `src + Σw<k` | `dst + Σw<k` | `n − Σw<k` | k = 0 | 0 |
| terminal | `src + n` | `dst + n` | `0` | only if `n = 0` | 1 |

`tail` and `width` are **derived** properties, never stored — in the AIR they are
a lookup output, and a model that stored them could not disagree with itself.

The `n = 0` case is one row that is both `first` and `end`; both its bus
multiplicities (`mu − end`, `mu − first`) are zero, so it neither sends nor
receives on `DmaNext`, and it emits no memory operations. It is a real trace row
nonetheless, and the completeness sweep pins it.

### 2.3 Bus level — `memw_ops(...)`, `replay_memw(...)`

| ops | timestamp | what |
|---|---|---|
| 3 register reads | `T` | x10 = dst, x11 = src, x12 = n (base address `2·reg`) |
| one per data row | `T+1` | read `width` bytes at `src + offset` |
| one per data row | `T+2` | write the same bytes to `dst + offset` |

`replay_memw` applies an op list in timestamp order and **raises if a read's
recorded value disagrees with memory at that timestamp** — the memory-consistency
argument, modelled just enough that a mis-ordered op list fails loudly instead
of quietly producing the right answer. That is what catches the
`write_before_read` mutant, which is otherwise indistinguishable on
non-overlapping inputs.

### 2.4 Guest level — `chunk_ecalls(...)`, `guest_memcpy(...)`

Transcribed from the inline assembly in `syscalls/src/syscalls.rs`: while bytes
remain, take `min(remaining, 256)`, ecall, advance both pointers. `n = 0` issues
no ecall (the leading `beqz`), and the C return value `dst` is preserved in `t0`.

## 3. Chip-contract map

| oracle concept | AIR realisation |
|---|---|
| `validate`'s `n > 256` | `Alu[count, 257, LT] → 1`, multiplicity `first` |
| `validate`'s wrap checks | `emit_add_pair_no_overflow` on `src` and `dst` |
| `row_widths`' `remaining >= 8` | `Alu[count, 8, LT] → tail`, multiplicity `mu` |
| `width` | `step = 8 − 7·tail` |
| row k → row k+1 | `DmaNext` send `[ts, src_incr, dst_incr, count_decr]` / receive `[ts, src, dst, count]` |
| terminal row | `Zero[4·65535 − Σ count_decr] → end` |
| snapshot ordering | the AIR constants `T+1` (reads) and `T+2` (writes) |
| "the same bytes" | one set of `value` columns feeding both `Memw` tuples |
| zero-padded `value` | `tail · value[i] = 0`, `i = 1..7` |

## 4. Canonical vectors

`canonical_dma_vectors.json`, regenerated by the harness. Ten cases chosen so
every structural case appears exactly once:

| name | dst | src | n | rows | MEMW ops |
|---|---|---|---|---|---|
| empty | 0x1000 | 0x2000 | 0 | 1 | 3 |
| single byte | 0x1000 | 0x2000 | 1 | 2 | 5 |
| one wide row | 0x1000 | 0x2000 | 8 | 2 | 5 |
| wide plus tail | 0x1000 | 0x2000 | 9 | 3 | 7 |
| widest tail | 0x1000 | 0x2000 | 7 | 8 | 17 |
| unaligned body and tail | 0x2005 | 0x1003 | 27 | 7 | 15 |
| forward overlap | 0x3004 | 0x3000 | 24 | 4 | 9 |
| backward overlap | 0x3000 | 0x3004 | 24 | 4 | 9 |
| page crossing | 0x0FFC | 0x1FFC | 16 | 3 | 7 |
| maximum chunk | 0x1000 | 0x2000 | 256 | 33 | 67 |

"widest tail" is the expensive shape: `n = 7` is seven one-byte rows plus a
terminal, eight rows to move seven bytes. "maximum chunk" is the only case with
**no tail row at all**, which is why it is pinned — 256 is 8-aligned, so
`n % 8 = 0` and the last data row is a wide one.

Each vector carries its full row-and-column expansion (`row_columns`), so the
Rust side can be checked against this model without re-deriving it — see
`prover/src/tests/dma_tests.rs::dma_trace_matches_oracle_row_decomposition`.

## 5. Mutation sweep (anchor 5)

Every mutant must be caught by the anchor it targets. All six are:

| mutant | caught by |
|---|---|
| `row_widths` = all ones | anchor 3(d): disagrees with the closed form |
| `row_widths` = always wide | anchor 3: widths do not sum to `n` |
| `row_widths` tail off by one (`>` for `>=`) | anchor 3(d) |
| MEMW write before read | anchor 3(a) via `replay_memw`, **only on the overlapping deltas** |
| MEMW reads/writes interleaved per chunk | anchor 3(a), same |
| `chunk_ecalls` at 257 | anchor 4: a chunk exceeds the executor's bound |

The two timestamp mutants are the reason `DELTAS` includes `0, ±1, ±7, ±8, ±9`
and not only disjoint ranges: on non-overlapping buffers, reading after writing
is indistinguishable from reading before.

## 6. Open questions and known limitations

**O1 — the model does not model the memory table.** `replay_memw` enforces
read faithfulness at its own timestamp; it does not model per-address ordering
of *multi-byte* accesses, unaligned 8-byte operations, or the `Memw` width
decode. Those are the memory argument's own obligations.

**O2 — the per-ecall snapshot is not a `memcpy`-level `memmove`.** Chunk *k+1*
reads memory chunk *k* already wrote, so `guest_memcpy` is a forward copy. For
`dst < src` and for non-overlapping ranges it agrees with `memmove`; for
`dst > src` with an overlap wider than 256 bytes it does not. In contract for
`memcpy`, and anchor 4 deliberately excludes overlap for this reason — but the
claim "the DMA ecall has memmove semantics" must not be repeated at the C level.

**O3 — `value` bytes are modelled as integers, not range-checked bytes.** The
oracle emits `0..255` because it reads them out of a byte memory. The AIR gets
its byte range from the `Memw` receiver, which is outside both the oracle and the
gate; the audit script checks the wiring.

**O4 — the anchors test the *semantics*, not the executor.** `dma_ref` is a
model of `execution.rs`, checked against libc; that `execution.rs` matches it is
covered by the PR's own 256-case proptest plus the vector test in
`executor/src/tests/dma_tests.rs`, not by anything here.

**O5 — register reads are modelled as three ops at `T` and nothing more.** The
old-value/old-timestamp fields, and the fact that the DMA table *writes back* the
same value it read, are not modelled. They are not part of the copy semantics,
but they are part of the trace, and the audit script is the only thing looking at
them.

## 7. File manifest

| file | what it is |
|---|---|
| `dma_ref.py` | the four-level reference model |
| `test_oracle.py` | five-anchor validation harness; emits the canonical vectors |
| `canonical_dma_vectors.json` | 10 pinned vectors with full column expansions |
| `ORACLE.md` | this file |
