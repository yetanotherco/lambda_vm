"""
WHICH ARTIFACT WAS GATED -- a re-checkable pin, AND a framing-conformance check.

The chip is UNCOMMITTED and is being edited by a concurrent reviewer, so "the
file at path X" is not an identification and line numbers are not either.

WHAT THIS DOES
  1. hashes the NORMALIZED content (comments and whitespace stripped) of the
     four constraint- and framing-bearing regions, so the hash tracks semantics
     and is stable under prose edits and line drift;
  2. RESOLVES the chip's framing constants and checks them against
     `socket_ref.py`'s specification -- so the pin answers "does the chip still
     compute the socket the oracle specifies?", not merely "has this text
     changed?".

## Why (2) exists: this file's first version had a fail-open, and it fired

v1 hashed three regions -- `eval`, `bitwise_interactions`, `cols` -- and recorded
constants as their EXPRESSION TEXT. When the implementer's second wave landed it
reported "artifact matches the pin". That was a FALSE PASS, for two reasons, and
both are the exact failure mode this whole gate is built to prevent:

  * `SOCKET_ROUNDS` changed definition (to an alias of `BLAKE3_ROUNDS`). v1
    recorded `NUM_G = "SOCKET_ROUNDS * 8"`, which is stable under that change,
    so the pin could not see it. Hashing an expression is not hashing a value.
  * worse: `SOCKET_ROUNDS`, `TAG_LFMC`, `FLAGS_LFMC`, `BLOCK_LEN_LFMC`,
    `COUNTER_LFMC` and `OUT_WINDOW` are top-level constants that live in NONE of
    the three hashed regions. They are precisely the framing degrees of freedom
    the negative-control board tests. A change of `FLAGS_LFMC` from `0x0B` to
    anything else -- a live control, `flags_parent` -- would have passed silently.

The change that exposed it was benign. The hole was not. A drift detector that
answers PASS without looking at the thing that matters is worse than no detector,
because it is trusted.

Run: python3 artifact_pin.py           # record
     python3 artifact_pin.py --check   # verify against the record
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys

import socket_ref as sk

CHIP = ("/Users/maurofab/workspace/lambda_vm-blake3-impl/"
        "prover/src/lfm/blake3_socket.rs")
PRIMITIVE = ("/Users/maurofab/workspace/lambda_vm-blake3-impl/"
             "prover/src/lfm/blake3.rs")

HERE = os.path.dirname(os.path.abspath(__file__))
PIN_FILE = os.path.join(HERE, "artifact_pin.json")

BRACE_REGIONS = {
    "eval": "pub fn eval<B: ConstraintBuilder",
    "bitwise_interactions": "pub fn bitwise_interactions()",
    "cols": "pub mod cols",
}


def _brace_region(src: str, start_pat: str) -> str:
    i = src.index(start_pat)
    j = src.index("{", i)
    depth, k = 0, j
    while True:
        if src[k] == "{":
            depth += 1
        elif src[k] == "}":
            depth -= 1
            if depth == 0:
                break
        k += 1
    return src[i:k + 1]


def _normalize(s: str) -> str:
    """Hash the SEMANTICS, not the prose: strip line comments (including `//!`
    module docs and `///` item docs) and collapse whitespace. A doc rewrite must
    not invalidate the gate; a changed constraint must."""
    s = re.sub(r"//[^\n]*", "", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def _const_region(src: str) -> str:
    """REGION 4, added after the v1 fail-open: every top-level const declaration.
    This is where the framing constants live -- outside `eval`, outside `cols`,
    and therefore outside v1's coverage entirely."""
    lines = [ln for ln in src.splitlines()
             if re.match(r"\s*(pub(\([^)]*\))?\s+)?const\s+[A-Z_0-9]+\s*:", ln)
             or re.match(r"\s*(pub(\([^)]*\))?\s+)?const\s+_\s*:", ln)]
    return _normalize("\n".join(lines))


# ---------------------------------------------------------------------------
# Resolve the framing constants and check them against the ORACLE's spec.
# Every extraction is MANDATORY: a constant we cannot find is a FAILURE, never
# a silent skip. (Silently skipping is how v1 passed.)
# ---------------------------------------------------------------------------

def _find(src: str, pattern: str, name: str) -> str:
    m = re.search(pattern, src)
    if not m:
        raise LookupError(f"could not resolve `{name}` -- the pin cannot vouch "
                          f"for a constant it cannot find")
    return m.group(1).strip()


def resolve_framing(chip_src: str, prim_src: str) -> dict:
    def _tag(name: str) -> tuple[str, int]:
        expr = _find(chip_src, rf"pub const {name}:\s*u32\s*=\s*([^;]+);", name)
        mm = re.match(r'u32::from_le_bytes\(\*b"(\w{4})"\)', expr)
        if not mm:
            raise LookupError(f"{name} has an unexpected form: {expr!r}")
        return mm.group(1), int.from_bytes(mm.group(1).encode(), "little")

    m_ascii, tag_val = _tag("TAG_LFMC")
    class m:                       # keep the existing .group(1) call site working
        @staticmethod
        def group(_):
            return m_ascii
    # POST-B1: the transcript tag is part of the chip's framing too, so the pin
    # must resolve and check it. v1's lesson was that a framing value living
    # outside the hashed regions passes silently; a SECOND tag that the pin does
    # not know about is the same hole one tag over.
    t_ascii, tag_t_val = _tag("TAG_LFMT")
    l_ascii, tag_l_val = _tag("TAG_LFML")

    flags = int(_find(chip_src, r"pub const FLAGS_LFMC:\s*u32\s*=\s*([^;]+);",
                      "FLAGS_LFMC"), 0)
    blen = int(_find(chip_src, r"pub const BLOCK_LEN_LFMC:\s*u32\s*=\s*([^;]+);",
                     "BLOCK_LEN_LFMC"), 0)
    counter = int(_find(chip_src, r"pub const COUNTER_LFMC:\s*u64\s*=\s*([^;]+);",
                        "COUNTER_LFMC"), 0)
    out_window = _find(chip_src, r"pub const OUT_WINDOW:\s*usize\s*=\s*([^;]+);",
                       "OUT_WINDOW")
    num_g = _find(chip_src, r"pub const NUM_G:\s*usize\s*=\s*([^;]+);", "NUM_G")
    g_size = int(_find(chip_src, r"pub const G_SIZE:\s*usize\s*=\s*([^;]+);",
                       "G_SIZE"), 0)
    num_lanes = int(_find(chip_src, r"pub const NUM_LANES:\s*usize\s*=\s*([^;]+);",
                          "NUM_LANES"), 0)
    socket_rounds = _find(chip_src, r"pub const SOCKET_ROUNDS:\s*usize\s*=\s*([^;]+);",
                          "SOCKET_ROUNDS")
    full_output = _find(chip_src, r"full_output:\s*(\w+)", "FLOW.full_output")

    # SOCKET_ROUNDS resolves through BLAKE3_ROUNDS, whose value is cfg-dependent.
    # Record BOTH arms: the chip compiles to exactly one, the gate covers both.
    std = _find(prim_src, r'#\[cfg\(not\(feature = "blake3-6round"\)\)\]\s*'
                          r'pub const BLAKE3_ROUNDS:\s*usize\s*=\s*([^;]+);',
                "BLAKE3_ROUNDS (default arm)")
    six = _find(prim_src, r'#\[cfg\(feature = "blake3-6round"\)\]\s*'
                          r'pub const BLAKE3_ROUNDS:\s*usize\s*=\s*([^;]+);',
                "BLAKE3_ROUNDS (6round arm)")
    std_v = int(_find(prim_src, r"pub const BLAKE3_STANDARD_ROUNDS:\s*usize\s*=\s*([^;]+);",
                      "BLAKE3_STANDARD_ROUNDS"), 0)
    six_v = int(_find(prim_src, r"pub const BLAKE3_SIX_ROUNDS:\s*usize\s*=\s*([^;]+);",
                      "BLAKE3_SIX_ROUNDS"), 0)
    rounds_default = std_v if "STANDARD" in std else six_v
    rounds_feature = six_v if "SIX" in six else std_v

    return {
        "tag_ascii": m.group(1),
        "tag_word": tag_val,
        "flags": flags,
        "block_len": blen,
        "counter": counter,
        "out_window_expr": out_window,
        "num_g_expr": num_g,
        "g_size": g_size,
        "num_lanes": num_lanes,
        "tag_t_ascii": t_ascii,
        "tag_t_word": tag_t_val,
        "tag_l_ascii": l_ascii,
        "tag_l_word": tag_l_val,
        "socket_rounds_expr": socket_rounds,
        "full_output": full_output,
        "rounds_default": rounds_default,
        "rounds_under_blake3_6round": rounds_feature,
    }


def check_against_oracle(fr: dict) -> list[str]:
    """The pin's real job: does the chip's framing EQUAL the oracle's spec?"""
    bad = []
    if fr["tag_word"] != sk.TAG_LFMC:
        bad.append(f"TAG_LFMC {fr['tag_word']:#x} != oracle {sk.TAG_LFMC:#x}")
    if fr["tag_ascii"].encode() != sk.TAG_LFMC_ASCII:
        bad.append(f"tag ascii {fr['tag_ascii']!r} != oracle "
                   f"{sk.TAG_LFMC_ASCII.decode()!r}")
    if fr["tag_t_ascii"] != "LFMT" or fr["tag_t_word"] != 0x544D464C:
        bad.append(f"TAG_LFMT {fr['tag_t_ascii']!r}/{fr['tag_t_word']:#x} != "
                   f"'LFMT'/0x544D464C (transcript-spec/TRANSCRIPT.md §2)")
    if fr["tag_l_ascii"] != "LFML" or fr["tag_l_word"] != 0x4C4D464C:
        bad.append(f"TAG_LFML {fr['tag_l_ascii']!r}/{fr['tag_l_word']:#x} != "
                   f"'LFML'/0x4C4D464C (leaf-spec/LEAF.md §1)")
    # PAIRWISE distinct across all three -- one clash is one collapsed domain.
    tags = {"LFMC": fr["tag_word"], "LFMT": fr["tag_t_word"],
            "LFML": fr["tag_l_word"]}
    for x in tags:
        for y in tags:
            if x < y and tags[x] == tags[y]:
                bad.append(f"TAG_{x} == TAG_{y} -- that domain separation is gone")
    if fr["flags"] != sk.FLAGS_LFMC:
        bad.append(f"FLAGS_LFMC {fr['flags']:#x} != oracle {sk.FLAGS_LFMC:#x}")
    if fr["block_len"] != sk.BLOCK_LEN_LFMC:
        bad.append(f"BLOCK_LEN_LFMC {fr['block_len']} != oracle {sk.BLOCK_LEN_LFMC}")
    if fr["counter"] != sk.HONEST_7.counter:
        bad.append(f"COUNTER_LFMC {fr['counter']} != oracle {sk.HONEST_7.counter}")
    if fr["num_lanes"] != 2 * sk.DIGEST_LANES:
        bad.append(f"NUM_LANES {fr['num_lanes']} != 2 cells x {sk.DIGEST_LANES} lanes")
    if fr["g_size"] != 60:
        bad.append(f"G_SIZE {fr['g_size']} != 60 (the gated per-G cell count)")
    if fr["full_output"] != "false":
        bad.append(f"FLOW.full_output = {fr['full_output']}, expected false "
                   f"(requirement R3: only the window's 4 words are built)")
    if fr["num_g_expr"].replace(" ", "") != "SOCKET_ROUNDS*8":
        bad.append(f"NUM_G = {fr['num_g_expr']!r}, expected SOCKET_ROUNDS * 8")
    gated = {6, 7}
    got = {fr["rounds_default"], fr["rounds_under_blake3_6round"]}
    if got != gated:
        bad.append(f"round counts {sorted(got)} are not the gated pair "
                   f"{sorted(gated)} -- the board covers only 6 and 7")
    return bad


def compute() -> dict:
    with open(CHIP) as f:
        chip_src = f.read()
    with open(PRIMITIVE) as f:
        prim_src = f.read()
    with open(CHIP, "rb") as f:
        raw = f.read()

    regions = {}
    for name, pat in BRACE_REGIONS.items():
        body = _normalize(_brace_region(chip_src, pat))
        regions[name] = {"sha256": hashlib.sha256(body.encode()).hexdigest(),
                         "normalized_len": len(body)}
    cb = _const_region(chip_src)
    regions["framing_consts"] = {"sha256": hashlib.sha256(cb.encode()).hexdigest(),
                                 "normalized_len": len(cb)}

    return {
        "path": CHIP,
        "file_sha256": hashlib.sha256(raw).hexdigest(),
        "regions": regions,
        "framing": resolve_framing(chip_src, prim_src),
    }


def main() -> int:
    try:
        cur = compute()
    except LookupError as exc:
        print(f"PIN FAILED: {exc}")
        return 1

    conformance = check_against_oracle(cur["framing"])

    if "--check" in sys.argv:
        if not os.path.exists(PIN_FILE):
            print("no pin recorded; run without --check first")
            return 1
        with open(PIN_FILE) as f:
            old = json.load(f)
        drift = [n for n, v in cur["regions"].items()
                 if old["regions"].get(n, {}).get("sha256") != v["sha256"]]
        fdrift = {k: (old["framing"].get(k), v)
                  for k, v in cur["framing"].items()
                  if old["framing"].get(k) != v}
        ok = True
        if drift:
            print(f"REGION DRIFT: {drift}")
            print("  The gate verdict does NOT carry over. Re-transcribe the "
                  "changed region into chip_model.py and re-run gate.py.")
            ok = False
        if fdrift:
            print(f"FRAMING CONSTANT DRIFT: {fdrift}")
            ok = False
        if conformance:
            print("FRAMING NO LONGER MATCHES THE ORACLE SPEC:")
            for b in conformance:
                print(f"  - {b}")
            ok = False
        if ok:
            print("artifact matches the pin AND its framing still equals the "
                  "oracle spec; the gate verdict applies")
            if old["file_sha256"] != cur["file_sha256"]:
                print(f"  (whole-file hash moved {old['file_sha256'][:12]} -> "
                      f"{cur['file_sha256'][:12]}, but only outside the four "
                      f"hashed regions -- i.e. in comments/docs)")
        return 0 if ok else 1

    if conformance:
        print("REFUSING TO PIN -- the chip's framing does not match the oracle:")
        for b in conformance:
            print(f"  - {b}")
        return 1

    with open(PIN_FILE, "w") as f:
        json.dump(cur, f, indent=1)
    print(f"pinned -> {PIN_FILE}")
    print(f"  file            {cur['file_sha256']}")
    for name, v in cur["regions"].items():
        print(f"  {name:22s}  {v['sha256']}")
    print("  framing (resolved values, checked against socket_ref.py):")
    for k, v in cur["framing"].items():
        print(f"      {k:28s} {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
