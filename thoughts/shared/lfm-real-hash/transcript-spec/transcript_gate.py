"""
GATE EXTENSION for the transcript tag — what is executable NOW, and what is
pre-committed for the build.

Design note: this harness IMPORTS `../gate-oracle/` rather than editing it. That
directory's model is PINNED to the committed chip (`artifact_pin.json`), and a
costing or spec exercise must not move a pinned instrument. Everything here is
additive.

WHAT IS EXECUTABLE NOW. The transcript step uses the frozen socket with a
different constant in `m[8]`, and `Framing.tag_word` already parameterises
exactly that — so the existing theorems apply to the transcript step today,
before any Rust exists:
  G1  the message schedule under TAG_LFMT, symbolic, all 7 rounds  -> UNSAT
  G2  the full pipeline, concrete, vs the transcript KATs           -> SAT
  G3  the same pipeline EXCLUDES a wrong digest                     -> UNSAT
  G4  tag controls: LFMC used where LFMT belongs, and vice versa    -> SAT

WHAT IS PRE-COMMITTED, NOT YET RUNNABLE. The mode-selected tag
`m[8] = MODE_C*TAG_LFMC + MODE_T*TAG_LFMT` needs a chip that has MODE_T. Those
controls are listed in `TRANSCRIPT.md` §5 and stubbed at the bottom of this file
so the build agent inherits them as a checklist rather than inventing them.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import replace

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "gate-oracle"))

import gate                          # noqa: E402
import socket_ref as sk              # noqa: E402
import transcript_ref as tr          # noqa: E402

TAG_T = sk.Framing(rounds=7, tag_word=tr.TAG_LFMT)


def board():
    rows = []

    def add(name, got, want, secs=0.0):
        ok = str(got) == want
        rows.append(ok)
        mark = "PASS" if ok else "**FAIL**"
        print(f"  [{mark:8s}] {name:52s} -> {str(got):6s} (want {want})"
              f"{f'  {secs:.1f}s' if secs > 0.3 else ''}")

    print("=" * 78)
    print("TRANSCRIPT GATE — executable now (tag = LFMT on the frozen socket)")
    print("=" * 78)

    # G1 — the schedule under the transcript tag, symbolic.
    r, t = gate.theorem_schedule(7, chip_framing=TAG_T, ref_framing=TAG_T)
    add("G1  message schedule @LFMT, 7 rounds, symbolic", r, "unsat", t)

    # G2/G3 — the full pipeline against the transcript KATs.
    with open(os.path.join(HERE, "transcript_kats.json")) as f:
        kats = json.load(f)
    vec = kats["rounds"]["7"]["step_vectors"][2]     # ramp_state_ramp_operand
    a, b, want = vec["state"], vec["operand"], vec["result"]

    r, t = gate.concrete_pipeline(7, a, b, want, chip_framing=TAG_T,
                                  timeout_ms=900_000)
    add("G2  full 7-round pipeline == transcript KAT", r, "sat", t)
    r, t = gate.concrete_pipeline(7, a, b, want, negate=True,
                                  chip_framing=TAG_T, timeout_ms=900_000)
    add("G3  same pipeline EXCLUDES a wrong digest", r, "unsat", t)

    # G4 — tag controls on the NEW surface, both directions.
    r, t = gate.concrete_control(7, a, b, want,
                                 chip_framing=sk.Framing(rounds=7),  # LFMC
                                 timeout_ms=900_000)
    add("G4a Merkle tag (LFMC) used for a transcript step", r, "sat", t)

    merkle_want = sk.socket_digest_wordlevel(a, b, sk.Framing(rounds=7))
    r, t = gate.concrete_control(7, a, b, merkle_want, chip_framing=TAG_T,
                                 timeout_ms=900_000)
    add("G4b transcript tag (LFMT) used for a Merkle parent", r, "sat", t)

    # G5 — the squeeze operand is load-bearing at the gate level too.
    st = [0x01020304, 0x05060708, 0x090A0B0C, 0x0D0E0F10]
    want0 = tr.compress_t(st, tr.squeeze_operand(0))
    r, t = gate.concrete_control(7, st, tr.squeeze_operand(1), want0,
                                 chip_framing=TAG_T, timeout_ms=900_000)
    add("G5  squeeze counter i=1 cannot produce squeeze i=0", r, "sat", t)

    print("\n" + "-" * 78)
    print(f"TRANSCRIPT GATE: {'PASS' if all(rows) else 'FAIL'}  "
          f"({sum(rows)}/{len(rows)})")
    return all(rows)


# ---------------------------------------------------------------------------
# PRE-COMMITTED — cannot run until the chip has MODE_T. The build agent must
# make each of these fire before the transcript arm is considered gated.
# ---------------------------------------------------------------------------

PRECOMMITTED_CONTROLS = [
    ("M1", "m[8] pinned to TAG_LFMC while MODE_T = 1",
     "SAT — a transcript row computing the Merkle tag is a live confusion bug"),
    ("M2", "m[8] pinned to TAG_LFMT while MODE_C = 1",
     "SAT — the mirror image; a Merkle parent computing the transcript tag"),
    ("M3", "MODE_C and MODE_T both 1 on one row",
     "UNSAT — excluded by the generalised mode-sum booleanity (idx 4)"),
    ("M4", "MODE_C = MODE_T = 0 on a row with MU = 1",
     "UNSAT — MU is defined as MODE_C + MODE_T, so this is not a real row"),
    ("M5", "drop the mode-sum booleanity",
     "SAT — modes become arbitrary felts, so m[8] becomes a prover-chosen "
     "linear combination of the two tags: the domain separation evaporates"),
    ("M6", "MODE_T treated as a MAIN (prover-chosen) column instead of "
           "preprocessed",
     "SAT — the whole soundness argument for the tag rests on MODE_* being "
     "preprocessed; this control is what proves that dependency is real"),
    ("M7", "capacity prefix idx 0-3 still pins S_k with the generalised form "
           "S_k = MODE_P*IN + (MODE_C + MODE_T)*IV_k",
     "UNSAT with the form present; SAT with it dropped"),
]


def print_precommitted():
    print("\n" + "=" * 78)
    print("PRE-COMMITTED CONTROLS (need MODE_T; the build agent must run these)")
    print("=" * 78)
    for tag, what, want in PRECOMMITTED_CONTROLS:
        print(f"  {tag}: {what}\n      expect: {want}")


if __name__ == "__main__":
    ok = board()
    print_precommitted()
    sys.exit(0 if ok else 1)
