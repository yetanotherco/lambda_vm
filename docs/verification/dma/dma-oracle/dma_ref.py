"""
Independent reference model for the DMA memcpy ecall (PR #874).

Three levels, deliberately written as three separate functions so they can be
checked against each other rather than sharing a helper:

  1. BYTE level   -- `memcpy_ref`: the C `memcpy`/`memmove` contract. Snapshot
                     the source, then write. This is the semantics a guest is
                     entitled to, and the only level a guest can observe.
  2. ROW level    -- `row_decomposition`: the row sequence the DMA AIR table is
                     obliged to contain for one ecall. Eight bytes per row while
                     `count >= 8`, then one byte per row, then one terminal row.
  3. BUS level    -- `memw_ops`: the MEMW multiset a correct trace must emit --
                     three register reads at T, every source read at T+1, every
                     destination write at T+2.

`replay_memw` runs level 3 back down to level 1, which is what makes the row
decomposition falsifiable: `replay_memw(memw_ops(...)) == memcpy_ref(...)` must
hold for every length and every overlap configuration, and the mutants in
`test_oracle.py` must break it.

`chunk_ecalls` is the fourth level above all of these: the guest's strong
`memcpy` symbol (`syscalls/src/syscalls.rs`) is a loop that issues one ecall per
<= `DMA_MEMCPY_MAX_BYTES` bytes, so a guest-visible `memcpy` of arbitrary length
is a *composition* of the above.

NOTHING here reads the Rust implementation. The constants and the ABI are
transcribed from it (`executor/src/vm/instruction/execution.rs`,
`prover/src/tables/{dma.rs,trace_builder.rs}`); the transcription is audited in
`../TRANSCRIPTION-AUDIT.md`.
"""

from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Constants (transcribed; asserted against the Rust source by the audit script)
# ---------------------------------------------------------------------------

#: `executor::vm::instruction::execution::DMA_MEMCPY_SYSCALL_NUMBER`
DMA_MEMCPY_SYSCALL_NUMBER = (1 << 64) - 3  # u64::MAX - 2

#: `executor::vm::instruction::execution::DMA_MEMCPY_MAX_BYTES`
DMA_MEMCPY_MAX_BYTES = 256

#: Address space; both `src + n` and `dst + n` must stay inside it.
ADDRESS_SPACE = 1 << 64

#: The wide row width, and the tail row width.
WIDE_WIDTH = 8
TAIL_WIDTH = 1

#: Argument registers. memcpy(dst = x10, src = x11, n = x12).
REG_DST, REG_SRC, REG_COUNT = 10, 11, 12

#: MEMW timestamp offsets relative to the ecall timestamp T.
TS_REGISTERS = 0
TS_READ = 1
TS_WRITE = 2


class DmaRejected(Exception):
    """The executor refuses the ecall (`n` too large, or a wrapping range)."""


# ---------------------------------------------------------------------------
# Level 1 -- byte semantics
# ---------------------------------------------------------------------------

def validate(dst: int, src: int, n: int) -> None:
    """The executor's three preconditions, in its own order.

    `n > MAX` is rejected first, so the chunk-bound error is what a guest sees
    for an oversized call even if the range would also have wrapped.
    """
    if n > DMA_MEMCPY_MAX_BYTES:
        raise DmaRejected(f"chunk has {n} bytes; maximum per ecall is {DMA_MEMCPY_MAX_BYTES}")
    if dst + n >= ADDRESS_SPACE:
        raise DmaRejected("destination range wraps the address space")
    if src + n >= ADDRESS_SPACE:
        raise DmaRejected("source range wraps the address space")


def memcpy_ref(memory: dict, dst: int, src: int, n: int) -> dict:
    """`memmove(dst, src, n)` on a sparse byte-addressed memory.

    Reads the whole source before writing anything, so overlapping regions get
    snapshot semantics -- the executor copies through a fixed scratch buffer for
    exactly this reason. Unwritten memory reads as zero, matching the VM.

    Returns a NEW memory; the input is not mutated.
    """
    validate(dst, src, n)
    snapshot = [memory.get(src + i, 0) for i in range(n)]
    out = dict(memory)
    for i, byte in enumerate(snapshot):
        out[dst + i] = byte
    return out


# ---------------------------------------------------------------------------
# Level 2 -- row decomposition
# ---------------------------------------------------------------------------

def row_widths(n: int) -> list:
    """The width of each data row, in order.

    Greedy and deliberately *not* `[8]*(n//8) + [1]*(n%8)`: the AIR decides one
    row at a time from the remaining count (`tail = count < 8`), so the model
    decides one row at a time too. That the closed form agrees is a property the
    harness checks, not an assumption the model makes.
    """
    widths, remaining = [], n
    while remaining != 0:
        width = WIDE_WIDTH if remaining >= WIDE_WIDTH else TAIL_WIDTH
        widths.append(width)
        remaining -= width
    return widths


@dataclass
class DmaRow:
    """One row of the DMA table. Mirrors `prover::tables::dma::DmaOperation`."""
    timestamp: int
    src: int
    dst: int
    count: int
    first: bool
    end: bool
    value: list = field(default_factory=lambda: [0] * 8)

    @property
    def tail(self) -> bool:
        """`tail` is a *derived* column: the AIR pins it with an LT lookup."""
        return self.count < WIDE_WIDTH

    @property
    def width(self) -> int:
        return TAIL_WIDTH if self.tail else WIDE_WIDTH


def row_decomposition(timestamp: int, dst: int, src: int, n: int, memory: dict = None) -> list:
    """The rows a correct DMA trace must contain for one ecall, in chain order.

    One data row per copied chunk plus exactly one terminal row (`count == 0`,
    `end = 1`). `first` marks the head. `value` holds the copied bytes,
    zero-padded past the row's width -- the AIR forces those lanes to zero on
    tail rows, so the model must produce them zeroed too.

    `memory` is only needed to fill `value`; omit it for a shape-only model.
    """
    validate(dst, src, n)
    memory = memory or {}
    rows, offset, remaining = [], 0, n
    while remaining != 0:
        width = WIDE_WIDTH if remaining >= WIDE_WIDTH else TAIL_WIDTH
        value = [memory.get(src + offset + i, 0) for i in range(width)] + [0] * (8 - width)
        rows.append(DmaRow(
            timestamp=timestamp,
            src=src + offset,
            dst=dst + offset,
            count=remaining,
            first=not rows,
            end=False,
            value=value,
        ))
        offset += width
        remaining -= width
    rows.append(DmaRow(
        timestamp=timestamp,
        src=src + n,
        dst=dst + n,
        count=0,
        first=not rows,      # true only for n == 0: one row that is both
        end=True,
        value=[0] * 8,
    ))
    return rows


# ---------------------------------------------------------------------------
# Level 3 -- the MEMW multiset
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MemwOp:
    """One memory-bus operation. `is_write=False` leaves memory unchanged."""
    is_register: bool
    address: int
    timestamp: int
    width: int
    value: tuple
    is_write: bool


def memw_ops(timestamp: int, dst: int, src: int, n: int, memory: dict) -> list:
    """Every MEMW operation one DMA ecall must put on the bus.

    Order matters only through the timestamps: registers at T, *all* source
    reads at T+1, *all* destination writes at T+2. The two-phase split is what
    makes overlap well defined -- see `test_oracle.py`'s `write_before_read`
    mutant, which is caught only by the overlapping cases.
    """
    validate(dst, src, n)
    ops = [
        MemwOp(True, 2 * REG_DST, timestamp + TS_REGISTERS, 2, (dst,), False),
        MemwOp(True, 2 * REG_SRC, timestamp + TS_REGISTERS, 2, (src,), False),
        MemwOp(True, 2 * REG_COUNT, timestamp + TS_REGISTERS, 2, (n,), False),
    ]
    reads, writes, offset = [], [], 0
    for width in row_widths(n):
        chunk = tuple(memory.get(src + offset + i, 0) for i in range(width))
        reads.append(MemwOp(False, src + offset, timestamp + TS_READ, width, chunk, False))
        writes.append(MemwOp(False, dst + offset, timestamp + TS_WRITE, width, chunk, True))
        offset += width
    return ops + reads + writes


def replay_memw(ops: list, memory: dict) -> dict:
    """Apply a MEMW list to memory in timestamp order. Reads must be faithful.

    Raises if a read op's recorded value disagrees with memory at its
    timestamp -- that is the memory-consistency argument the MEMW table proves,
    modelled here so a mis-ordered op list fails loudly instead of quietly
    producing the right answer.
    """
    out = dict(memory)
    for op in sorted(ops, key=lambda o: o.timestamp):
        if op.is_register:
            continue
        if op.is_write:
            for i, byte in enumerate(op.value):
                out[op.address + i] = byte
        else:
            seen = tuple(out.get(op.address + i, 0) for i in range(op.width))
            if seen != op.value:
                raise AssertionError(
                    f"read at {op.address:#x}@{op.timestamp} recorded {op.value}, memory has {seen}"
                )
    return out


# ---------------------------------------------------------------------------
# Level 4 -- the guest stub's chunking loop
# ---------------------------------------------------------------------------

def chunk_ecalls(dst: int, src: int, n: int) -> list:
    """The `(dst, src, count)` triples the guest's `memcpy` stub issues.

    Transcribed from the inline assembly in `syscalls/src/syscalls.rs`: while
    bytes remain, take `min(remaining, MAX)`, ecall, then advance both pointers
    by the chunk. `n == 0` issues no ecall at all (the leading `beqz`).
    """
    calls, offset, remaining = [], 0, n
    while remaining != 0:
        chunk = min(remaining, DMA_MEMCPY_MAX_BYTES)
        calls.append((dst + offset, src + offset, chunk))
        offset += chunk
        remaining -= chunk
    return calls


def guest_memcpy(memory: dict, dst: int, src: int, n: int) -> tuple:
    """What a guest calling `memcpy` observes: the memory effect and the return.

    NOTE the semantics change at this level. Per chunk the copy is a snapshot,
    but *across* chunks it is not: chunk k+1 reads memory chunk k already wrote.
    That is plain forward `memmove`, correct for `dst < src` and for
    non-overlapping ranges, and NOT a `memmove` for `dst > src` with an overlap
    of more than `MAX` bytes. `memcpy`'s contract does not cover overlap, so
    this is in-contract -- but it means the DMA ecall's per-call snapshot is not
    a `memmove` guarantee at the C level. Recorded in ORACLE.md as O2.
    """
    out = dict(memory)
    for chunk_dst, chunk_src, chunk_n in chunk_ecalls(dst, src, n):
        out = memcpy_ref(out, chunk_dst, chunk_src, chunk_n)
    return out, dst


# ---------------------------------------------------------------------------
# Column encodings -- the AIR's view of a row
# ---------------------------------------------------------------------------

def dword_wl(value: int) -> list:
    """`DWordWL`: two 32-bit words, little-endian. `set_dword_wl`."""
    return [value & 0xFFFF_FFFF, (value >> 32) & 0xFFFF_FFFF]


def dword_hl(value: int) -> list:
    """`DWordHL`: four 16-bit halfwords, little-endian. `set_dword_hl`."""
    return [(value >> (16 * i)) & 0xFFFF for i in range(4)]


def row_columns(row: DmaRow) -> dict:
    """Every committed column of one DMA row, by name.

    This is the object the z3 gate pins for its positive controls and the
    object the Rust trace generator must produce; keeping it here (rather than
    inside the gate) is what lets the gate's completeness sweep be an
    *oracle-driven* check instead of a self-consistency check.
    """
    width = row.width
    return {
        "timestamp": dword_wl(row.timestamp),
        "src": dword_wl(row.src),
        "src_incr": dword_hl((row.src + width) % ADDRESS_SPACE),
        "dst": dword_wl(row.dst),
        "dst_incr": dword_hl((row.dst + width) % ADDRESS_SPACE),
        "count": dword_wl(row.count),
        "count_decr": dword_hl((row.count - width) % ADDRESS_SPACE),
        "first": int(row.first),
        "end": int(row.end),
        "tail": int(row.tail),
        "value": list(row.value),
        "mu": 1,
    }


def padding_columns() -> dict:
    """The padding row the trace generator emits (`generate_dma_trace`).

    `mu = 0` kills every bus interaction, but the arithmetic constraints on
    `count_decr` are unconditional, so padding must still satisfy them:
    `count = 1`, `tail = 1` (width 1), `count_decr = 0`. `src_incr`/`dst_incr`
    are 1 so their low carry is zero rather than `-1`.
    """
    return {
        "timestamp": [0, 0],
        "src": [0, 0],
        "src_incr": [1, 0, 0, 0],
        "dst": [0, 0],
        "dst_incr": [1, 0, 0, 0],
        "count": [1, 0],
        "count_decr": [0, 0, 0, 0],
        "first": 0,
        "end": 0,
        "tail": 1,
        "value": [0] * 8,
        "mu": 0,
    }
