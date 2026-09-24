"""
Tamper tests: are the oracle's anchors sensitive, or do they pass vacuously?

Split out of `test_ref.py` so the two jobs are separable, following the shape
PR #923 established for `formal_verification/` (`test_ref.py` checks the model,
`tamper_test.py` checks the checks). Each mutant below is a deliberate defect in
`dma_ref`; every one must be caught by the anchor it targets, and a mutant whose
anchor SKIPped is reported `NOT RUN` rather than credited as caught.

    python3 tamper_test.py            # full
    python3 tamper_test.py --quick    # shorter sweeps

Imported by `test_ref.py` so `python3 test_ref.py` still runs the whole board.
"""

import sys

import dma_ref as ref
from test_ref import (
    MAX, anchor_chunking, anchor_libc, anchor_row_level, anchor_slice_assign,
)

# ---------------------------------------------------------------------------
# [5] mutation sweep -- are the anchors above sensitive?
# ---------------------------------------------------------------------------

def _mutant_all_ones(n):
    return [1] * n


def _mutant_always_wide(n):
    return [8] * ((n + 7) // 8)


def _mutant_off_by_one_tail(n):
    widths, remaining = [], n
    while remaining != 0:
        width = 8 if remaining > 8 else 1      # `>` instead of `>=`
        widths.append(min(width, remaining))
        remaining -= widths[-1]
    return widths


def _mutant_write_before_read(timestamp, dst, src, n, memory):
    """Reads at T+2, writes at T+1: the copy stops being a snapshot."""
    ops = ref.memw_ops(timestamp, dst, src, n, memory)
    return [
        op if op.is_register else
        type(op)(op.is_register, op.address,
                 timestamp + (1 if op.is_write else 2),
                 op.width, op.value, op.is_write)
        for op in ops
    ]


def _mutant_interleaved(timestamp, dst, src, n, memory):
    """Each chunk written immediately after it is read (per-chunk timestamps)."""
    out = [op for op in ref.memw_ops(timestamp, dst, src, n, memory) if op.is_register]
    offset = 0
    for i, width in enumerate(ref.row_widths(n)):
        chunk = tuple(memory.get(src + offset + j, 0) for j in range(width))
        out.append(ref.MemwOp(False, src + offset, timestamp + 1 + 2 * i, width, chunk, False))
        out.append(ref.MemwOp(False, dst + offset, timestamp + 2 + 2 * i, width, chunk, True))
        offset += width
    return out


def _mutant_no_snapshot(memory, dst, src, n):
    """Copy byte-by-byte with no snapshot: correct for disjoint ranges, wrong for
    a backward overlap. The control for anchors 1 and 2, which had none --
    every other mutant targets `row_widths`/`memw_ops`/`chunk_ecalls`, i.e.
    anchors 3 and 4, so nothing demonstrated the two external differentials can
    fail at all."""
    ref.validate(dst, src, n)
    out = dict(memory)
    for i in range(n):
        out[dst + i] = out.get(src + i, 0)
    return out


def _mutant_chunk_257(dst, src, n):
    calls, offset, remaining = [], 0, n
    while remaining != 0:
        c = min(remaining, MAX + 1)            # one byte over the executor's bound
        calls.append((dst + offset, src + offset, c))
        offset += c
        remaining -= c
    return calls


def _with_memcpy_ref(replacement, run):
    """Temporarily swap `dma_ref.memcpy_ref`, so anchors 1/2 can be mutated too.

    Those two anchors call it through the module rather than via an injection
    point, so unlike `row_widths`/`memw_ops` they cannot be parameterised.
    """
    original = ref.memcpy_ref
    ref.memcpy_ref = replacement
    try:
        return run()
    finally:
        ref.memcpy_ref = original


def anchor_mutations(quick: bool):
    """Every mutant must be caught by the anchor it targets."""
    mutants = [
        ("memcpy_ref without snapshot", lambda: _with_memcpy_ref(
            _mutant_no_snapshot, lambda: anchor_libc(quick))),
        ("memcpy_ref without snapshot (slice)", lambda: _with_memcpy_ref(
            _mutant_no_snapshot, lambda: anchor_slice_assign(quick))),
        ("row_widths = all ones", lambda: anchor_row_level(quick, widths=_mutant_all_ones)),
        ("row_widths = always wide", lambda: anchor_row_level(quick, widths=_mutant_always_wide)),
        ("row_widths tail off by one", lambda: anchor_row_level(quick, widths=_mutant_off_by_one_tail)),
        ("memw write before read", lambda: anchor_row_level(quick, ops=_mutant_write_before_read)),
        ("memw read/write interleaved", lambda: anchor_row_level(quick, ops=_mutant_interleaved)),
        ("chunk_ecalls at MAX+1", lambda: anchor_chunking(quick, chunk=_mutant_chunk_257)),
    ]
    survivors, not_run = [], []
    for name, run in mutants:
        try:
            ok, _ = run()
        except (AssertionError, ref.DmaRejected):
            ok = False              # replay_memw or the executor bound caught it
        # THREE states, not two. An anchor that SKIPped returns `ok is None`, and
        # `if ok:` would score that as "caught" -- a mutant credited to a check
        # that never ran. That is exactly the cascade this module's docstring
        # promises cannot happen, and it bit the two `memcpy_ref` mutants, whose
        # anchors (libc, CPython) are the ones that can be unavailable.
        if ok is None:
            not_run.append(name)
            verdict = "NOT RUN (anchor skipped)"
        elif ok:
            survivors.append(name)
            verdict = "SURVIVED (bad)"
        else:
            verdict = "caught"
        print(f"      mutant {name:32s} -> {verdict}")
    if survivors:
        return False, f"{len(survivors)} mutant(s) survived: {', '.join(survivors)}"
    if not_run:
        return None, (f"{len(mutants) - len(not_run)}/{len(mutants)} caught; "
                      f"{len(not_run)} not run: {', '.join(not_run)}")
    return True, f"all {len(mutants)} mutants caught"



def main():
    quick = "--quick" in sys.argv
    print("=" * 72)
    print("DMA memcpy oracle -- tamper tests" + ("  (--quick)" if quick else ""))
    print("=" * 72)
    ok, detail = anchor_mutations(quick)
    label = {True: "PASS", False: "FAIL", None: "PARTIAL"}[ok]
    print(f"\n  {label}  {detail}")
    sys.exit(0 if ok is True else (2 if ok is None else 1))


if __name__ == "__main__":
    main()
