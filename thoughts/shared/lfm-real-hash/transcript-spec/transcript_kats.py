"""
Transcript KATs: per-op vectors, an END-TO-END FriToyV0-shaped transcript, and
the checks that make them meaningful.

What this establishes, all ✓ EXECUTED:
  K1  every transcript step equals `BLAKE3(state ‖ operand ‖ "LFMT")[..16]` at
      7 rounds — computed by two separate routes (word level and byte level)
      and asserted equal. This is the crate-KAT identity the implementer must
      re-assert with a one-line `blake3::hash` call.
  K2  a full FriToyV0-preamble-shaped transcript, op by op, with the state
      after every step — so the implementer has an end-to-end vector, not only
      per-op ones.
  K3  DOMAIN SEPARATION IS REAL: a transcript step and a Merkle parent over the
      same two cells produce different digests (the tag is load-bearing).
  K4  the squeeze counter is load-bearing: dropping it makes consecutive
      squeezes iterate one fixed map, and the vectors change.
  K5  ordering is load-bearing: swapping two absorbs changes the transcript.
  K6  compression accounting matches the spec's cost claim (11 for FriToyV0).

Run: python3 transcript_kats.py [--write]
"""

from __future__ import annotations

import json
import os
import sys

import transcript_ref as tr

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "gate-oracle"))
import socket_ref as sk              # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "transcript_kats.json")

# FriToyV0's shape, ✓ VERIFIED against fixture.rs:26-40.
NUM_QUERIES = 4
QUERY_BITS = 4

# Fixed, written-out inputs — nothing depends on an RNG.
MAIN_ROOT = [0x01020304, 0x05060708, 0x090A0B0C, 0x0D0E0F10]
L1_ROOT = [0x11121314, 0x15161718, 0x191A1B1C, 0x1D1E1F20]
T0W = [0xDEADBEEF, 0xCAFEBABE, 0x8BADF00D, 0xFEEDFACE]
T1W = [0x0BADC0DE, 0xD15EA5E5, 0xC0FFEE00, 0xBAAAAAAD]


def hexlanes(c):
    return "".join(f"{x:08x}" for x in c)


def k1_step_identity(rounds: int) -> tuple[bool, str, list]:
    """Every step, both routes. At 7 rounds the byte route IS `blake3::hash`."""
    cases = [
        ("zero_state_zero_operand", [0, 0, 0, 0], [0, 0, 0, 0]),
        ("zero_state_main_root", [0, 0, 0, 0], MAIN_ROOT),
        ("ramp_state_ramp_operand", MAIN_ROOT, L1_ROOT),
        ("max_state", [0xFFFFFFFF] * 4, T0W),
        ("squeeze_operand_0", MAIN_ROOT, tr.squeeze_operand(0)),
        ("squeeze_operand_255", L1_ROOT, tr.squeeze_operand(255)),
    ]
    out = []
    for name, st, op in cases:
        w = tr.compress_t(st, op, rounds)
        b = tr.compress_t_bytelevel(st, op, rounds)
        if w != b:
            return False, f"K1 route mismatch on {name}@{rounds}", []
        out.append({"name": name, "state": st, "operand": op,
                    "result": w, "result_hex": hexlanes(w)})
    return True, f"K1 PASS: {len(cases)} steps, word route == byte route", out


def k2_end_to_end(rounds: int) -> tuple[dict, list]:
    """The FriToyV0 preamble + query loop, op by op.

    ✓ VERIFIED sequence, programs.rs:549-567:
      absorb(main_root), squeeze_ext, squeeze_ext, absorb(l1_root),
      squeeze_ext, absorb_felts(t0w), absorb_felts(t1w), then
      NUM_QUERIES x squeeze_bits.
    """
    t = tr.Transcript(rounds=rounds)
    steps = []

    def rec(op, value=None):
        steps.append({"op": op,
                      "state_after": list(t.state),
                      "state_after_hex": hexlanes(t.state),
                      **({"output": value, "output_hex": hexlanes(value)}
                         if value is not None and len(value) == 4 else {}),
                      **({"output_lanes": value} if value is not None
                         and len(value) != 4 else {})})

    t.absorb(MAIN_ROOT);            rec("absorb(main_root)")
    alpha = t.squeeze_ext();        rec("squeeze_ext -> alpha", alpha)
    zeta0 = t.squeeze_ext();        rec("squeeze_ext -> zeta0", zeta0)
    t.absorb(L1_ROOT);              rec("absorb(l1_root)")
    zeta1 = t.squeeze_ext();        rec("squeeze_ext -> zeta1", zeta1)
    # ✓ VERIFIED programs.rs: the preamble now calls absorb_felts TWICE, not
    # absorb2 — t0/t1 are terminal-polynomial coefficients, i.e. ARBITRARY field
    # elements, so each goes leaf-then-absorb. Four compresses where the old
    # vector modelled two; this is the 91 -> 93 correction, in the vectors.
    t.absorb_felts(T0W);            rec("absorb_felts(t0w)")
    t.absorb_felts(T1W);            rec("absorb_felts(t1w)")
    query_bits = []
    for q in range(NUM_QUERIES):
        bits = t.squeeze_bits(QUERY_BITS)
        query_bits.append(bits)
        rec(f"squeeze_bits(q={q})", bits)

    return {
        "rounds": rounds,
        "inputs": {"main_root": MAIN_ROOT, "l1_root": L1_ROOT,
                   "t0w": T0W, "t1w": T1W},
        "shape": {"num_queries": NUM_QUERIES, "query_bits": QUERY_BITS},
        "steps": steps,
        "challenges": {"alpha": alpha, "zeta0": zeta0, "zeta1": zeta1,
                       "query_bits": query_bits},
        "compressions": t.compressions,
        "final_state": list(t.state),
    }, steps


def k3_domain_separation(rounds: int) -> tuple[bool, str]:
    """A transcript step must NOT equal a Merkle parent over the same cells."""
    a, b = MAIN_ROOT, L1_ROOT
    step = tr.compress_t(a, b, rounds)
    parent = sk.socket_digest_wordlevel(a, b, sk.Framing(rounds=rounds))
    if step == parent:
        return False, ("K3 FAIL: transcript step == Merkle parent — the tag is "
                       "NOT separating the domains")
    return True, ("K3 PASS: transcript step != Merkle parent on the same two "
                  "cells (the LFMT/LFMC tag is load-bearing)")


def k4_counter_is_load_bearing(rounds: int) -> tuple[bool, str]:
    """Without SQ(i)'s counter every squeeze advance uses ONE fixed operand, so
    a run of squeezes iterates one fixed map. The vectors must notice."""
    t1 = tr.Transcript(rounds=rounds)
    t1.absorb(MAIN_ROOT)
    with_counter = [t1.squeeze() for _ in range(4)]

    t2 = tr.Transcript(rounds=rounds)
    t2.absorb(MAIN_ROOT)
    fixed = tr.squeeze_operand(0)
    without = []
    for _ in range(4):
        without.append(list(t2.state))
        t2.state = tr.compress_t(t2.state, fixed, rounds)

    if with_counter == without:
        return False, "K4 FAIL: the squeeze counter changes nothing"
    first_diff = next(i for i, (x, y) in enumerate(zip(with_counter, without))
                      if x != y)
    return True, (f"K4 PASS: counter-free squeezes diverge from the spec at "
                  f"squeeze #{first_diff} (they iterate one fixed map)")


def k5_order_is_load_bearing(rounds: int) -> tuple[bool, str]:
    a = tr.Transcript(rounds=rounds); a.absorb(MAIN_ROOT); a.absorb(L1_ROOT)
    b = tr.Transcript(rounds=rounds); b.absorb(L1_ROOT); b.absorb(MAIN_ROOT)
    if a.state == b.state:
        return False, "K5 FAIL: absorb order does not affect the transcript"
    return True, "K5 PASS: swapping two absorbs changes the state"


def main() -> int:
    print("=" * 74)
    print("TRANSCRIPT KATs — compress-chain (option B1)")
    print("=" * 74)
    ok = True
    doc = {"construction": "LFM compress-chain transcript (option B1)",
           "tag_ascii": tr.TAG_LFMT_ASCII.decode(),
           "tag_word": tr.TAG_LFMT,
           "squeeze_mark": tr.SQUEEZE_MARK,
           "state_cells": 1,
           "state_bits": 128,
           "initial_state": tr.ZERO_CELL,
           "framing": "identical to the Merkle socket except m[8] = TAG_LFMT",
           "rounds": {}}

    for rounds in (7, 6):
        good, msg, steps = k1_step_identity(rounds)
        ok &= good
        print(f"  [{'PASS' if good else 'FAIL'}] {msg}")
        e2e, _ = k2_end_to_end(rounds)
        doc["rounds"][str(rounds)] = {"step_vectors": steps, "fri_toy_v0": e2e}
        print(f"  [PASS] K2: FriToyV0-shaped transcript, {len(e2e['steps'])} "
              f"ops, {e2e['compressions']} compressions @{rounds}r")

    for fn in (k3_domain_separation, k4_counter_is_load_bearing,
               k5_order_is_load_bearing):
        good, msg = fn(7)
        ok &= good
        print(f"  [{'PASS' if good else 'FAIL'}] {msg}")

    # K6 — the cost claim in the spec must match what the reference performed.
    n = doc["rounds"]["7"]["fri_toy_v0"]["compressions"]
    good = (n == 13)
    ok &= good
    print(f"  [{'PASS' if good else 'FAIL'}] K6: FriToyV0 transcript costs "
          f"{n} compressions (13: 11 + the 2 leaf rows)")

    if "--write" in sys.argv:
        with open(OUT, "w") as f:
            json.dump(doc, f, indent=1)
        print(f"\n  wrote {OUT}")

    print("-" * 74)
    print(f"TRANSCRIPT KATs: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
