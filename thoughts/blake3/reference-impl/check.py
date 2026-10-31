"""
SECOND-SOURCE validation of the 6-round BLAKE3 vectors.

Source 1 is `thoughts/blake3/blake3-oracle/blake3_ref.py` -- an in-repo Python
oracle written from the spec, anchored on the official vectors at 7 rounds.
Source 2 is upstream BLAKE3's own portable C implementation (crate `blake3`
1.8.5, `c/blake3_portable.c`) with its round loop parameterised; see
PARAMETERISATION.diff for the entire edit.

The two sources are independent in the ways that matter:
  - different authors (the BLAKE3 team vs this repo) and different languages;
  - different message-schedule CONSTRUCTION: the C indexes a precomputed
    MSG_SCHEDULE[7][16] table, the Python/Rust iteratively apply a single
    permutation between rounds. A bug in the iterative composition -- exactly
    the kind of thing a single source cannot catch -- shows up here;
  - the C drives the FULL tree hasher through the parameterised compression,
    so its 7-round run is a direct external anchor rather than a borrowed one.

Checks, in order:
  [A] parameterised C at rounds=7 reproduces official_test_vectors.json
      (35 cases x 3 modes) -- the parameterisation is inert.
  [B] rounds=6 actually changes the function (negative control).
  [C] MSG_SCHEDULE[r] == permute^r(identity) -- the two schedule
      constructions denote the same thing.
  [D] C at rounds=6 reproduces all ten CANONICAL_VECTORS byte for byte,
      compared against BOTH canonical_6round_vectors.json AND the Rust
      constants in prover/src/lfm/blake3.rs.
  [E] randomised differential, C vs Python oracle, at rounds 7 and 6.

Run:  python3 check.py      (after ./build.sh)
"""

import json
import os
import random
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ORACLE_DIR = os.path.join(HERE, "..", "blake3-oracle")
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
# Where the canonical vectors live. This was `prover/src/lfm/blake3.rs` until
# P-a Stage 1 sank the primitive into `crypto`, so that the CUDA kernels, the
# commitment backends and the LFM chip could all be checked against ONE
# definition; `prover::lfm::blake3` is now a re-export and no longer holds the
# table. Checks [D] and [E] were silently dead between that move and 2026-08-15,
# because `parse_rust_vectors` failed with a bare `ValueError` from `str.index`
# rather than saying what had happened — see `read_rust_primitive`.
RUST_PRIMITIVE = os.path.join(
    REPO_ROOT, "crypto", "crypto", "src", "hash", "blake3", "vectors.rs"
)


def read_rust_primitive():
    """The vectors file, or a diagnosis of where it went.

    A harness that dies on `ValueError: substring not found` when the code it
    validates is refactored is a harness that gets deleted instead of fixed. If
    this ever fires again, grep for `CANONICAL_VECTORS` and update the path
    above — the parser itself is layout-independent and needs no change.
    """
    if not os.path.exists(RUST_PRIMITIVE):
        sys.exit(
            f"check.py: {RUST_PRIMITIVE} does not exist.\n"
            "The canonical vectors have moved again. Find them with\n"
            "  grep -rn 'pub const CANONICAL_VECTORS' --include='*.rs' .\n"
            "and update RUST_PRIMITIVE at the top of this file."
        )
    src = open(RUST_PRIMITIVE).read()
    if "pub const CANONICAL_VECTORS" not in src:
        sys.exit(
            f"check.py: {RUST_PRIMITIVE} exists but no longer defines "
            "CANONICAL_VECTORS.\nFind them with\n"
            "  grep -rn 'pub const CANONICAL_VECTORS' --include='*.rs' .\n"
            "and update RUST_PRIMITIVE at the top of this file."
        )
    return src

sys.path.insert(0, ORACLE_DIR)
import blake3_ref as ref  # noqa: E402

B3REF7 = os.path.join(HERE, "b3ref7")
B3REF6 = os.path.join(HERE, "b3ref6")

FAILURES = []


def check(name, cond, detail=""):
    if cond:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name}   {detail}")
        FAILURES.append(name)


def pattern_input(n):
    return bytes(i % 251 for i in range(n))


def run(binary, *args, stdin=None):
    r = subprocess.run([binary, *[str(a) for a in args]], input=stdin,
                       capture_output=True, text=True, check=True)
    return r.stdout


# ---------------------------------------------------------------------------
# [A] the parameterisation is inert at rounds = 7
# ---------------------------------------------------------------------------

def check_official_vectors():
    data = json.load(open(os.path.join(ORACLE_DIR, "official_test_vectors.json")))
    key_hex = data["key"].encode("utf-8").hex()
    context = data["context_string"]
    cases = data["cases"]

    bad = []
    for c in cases:
        n = c["input_len"]
        out_len = len(c["hash"]) // 2
        got = run(B3REF7, "hash", n, out_len).strip()
        if got != c["hash"]:
            bad.append(("hash", n))
        got = run(B3REF7, "keyed", key_hex, n, out_len).strip()
        if got != c["keyed_hash"]:
            bad.append(("keyed", n))
        got = run(B3REF7, "derive", context, n, out_len).strip()
        if got != c["derive_key"]:
            bad.append(("derive_key", n))
    check(f"[A] parameterised C @ rounds=7 vs official vectors "
          f"({len(cases)} cases x 3 modes)", not bad, str(bad[:5]))


# ---------------------------------------------------------------------------
# [B] rounds = 6 is genuinely a different function
# ---------------------------------------------------------------------------

def check_six_differs():
    diffs = 0
    total = 0
    for n in (0, 1, 63, 64, 65, 1024, 1025, 4096):
        total += 1
        if run(B3REF6, "hash", n, 32).strip() != run(B3REF7, "hash", n, 32).strip():
            diffs += 1
    check(f"[B] rounds=6 differs from rounds=7 on all {total} probe lengths",
          diffs == total, f"only {diffs}/{total} differed")


# ---------------------------------------------------------------------------
# [C] the two message-schedule constructions denote the same thing
# ---------------------------------------------------------------------------

def check_schedule_equivalence():
    """C uses a precomputed MSG_SCHEDULE table; Python composes one permutation
    repeatedly. Confirm row r equals permute applied r times to the identity."""
    text = open(os.path.join(HERE, "upstream", "blake3_impl.h")).read()
    blob = re.search(r"MSG_SCHEDULE\[7\]\[16\]\s*=\s*\{(.*?)\n\};", text, re.S).group(1)
    rows = [[int(x) for x in re.findall(r"\d+", row)]
            for row in blob.strip().split("\n") if "{" in row]
    assert len(rows) == 7 and all(len(r) == 16 for r in rows), rows

    cur = list(range(16))
    ok = True
    for r in range(7):
        if rows[r] != cur:
            ok = False
            print(f"        row {r}: table={rows[r]} composed={cur}")
        cur = ref.permute(cur)
    check("[C] MSG_SCHEDULE[r] == permute^r(identity) for r in 0..7", ok)

    check("[C] MSG_SCHEDULE[1] == the repo's BLAKE3_MSG_PERMUTATION",
          rows[1] == ref.MSG_PERMUTATION,
          f"{rows[1]} vs {ref.MSG_PERMUTATION}")


# ---------------------------------------------------------------------------
# [D] the ten canonical 6-round vectors, from the C, vs JSON and vs Rust
# ---------------------------------------------------------------------------

def parse_rust_vectors():
    src = read_rust_primitive()
    start = src.index("pub const CANONICAL_VECTORS")
    blob = src[start:src.index("\n];", start)]
    out = []
    for part in blob.split("Vector {")[1:]:
        v = {}
        for field in ("h", "m", "out"):
            body = re.search(field + r":\s*\[(.*?)\]", part, re.S).group(1)
            v[field] = [int(x, 0) for x in re.findall(r"0x[0-9A-Fa-f]+", body)]
        v["t"] = int(re.search(r"\bt:\s*(0x[0-9A-Fa-f]+|\d+)", part).group(1), 0)
        v["block_len"] = int(re.search(r"block_len:\s*(0x[0-9A-Fa-f]+|\d+)", part).group(1), 0)
        v["flags"] = int(re.search(r"flags:\s*(0x[0-9A-Fa-f]+|\d+)", part).group(1), 0)
        out.append(v)
    return out


def encode_record(v):
    words = [f"{w:08x}" for w in v["h"]] + [f"{w:08x}" for w in v["m"]]
    return " ".join(words) + f" {v['t']:016x} {v['block_len']} {v['flags']}\n"


def check_canonical_vectors():
    js = json.load(open(os.path.join(ORACLE_DIR, "canonical_6round_vectors.json")))
    rust = parse_rust_vectors()
    check("[D] Rust CANONICAL_VECTORS count == JSON count == 10",
          len(rust) == len(js) == 10, f"{len(rust)} / {len(js)}")

    # The C driver takes block_len and flags as uint8_t; confirm lossless.
    check("[D] all vector block_len/flags fit in u8 (driver is lossless here)",
          all(v["block_len"] < 256 and v["flags"] < 256 for v in js))

    # Inputs come from the JSON; OUTPUTS come from the C.
    stdin = "".join(encode_record(v) for v in js)
    lines = run(B3REF6, "compress", stdin=stdin).strip().split("\n")
    check("[D] C emitted one output per vector", len(lines) == 10, str(len(lines)))

    c_out = [[int(line[8 * i:8 * i + 8], 16) for i in range(16)] for line in lines]

    bad_json = [i for i in range(10) if c_out[i] != js[i]["out"]]
    check("[D] C @ rounds=6 == canonical_6round_vectors.json (all 10, 16 words)",
          not bad_json, f"vectors {bad_json}")

    bad_rust = [i for i in range(10) if c_out[i] != rust[i]["out"]]
    check("[D] C @ rounds=6 == Rust CANONICAL_VECTORS in "
          "crypto/crypto/src/hash/blake3/vectors.rs",
          not bad_rust, f"vectors {bad_rust}")

    # Inputs must match too, or the output agreement is about different things.
    bad_in = [i for i in range(10)
              if any(rust[i][f] != js[i][f] for f in ("h", "m", "t", "block_len", "flags"))]
    check("[D] Rust vector INPUTS == JSON vector inputs", not bad_in, f"vectors {bad_in}")

    # Negative control: the same inputs at 7 rounds must NOT match.
    lines7 = run(B3REF7, "compress", stdin=stdin).strip().split("\n")
    c7 = [[int(line[8 * i:8 * i + 8], 16) for i in range(16)] for line in lines7]
    check("[D] negative control: C @ rounds=7 matches none of the 10 vectors",
          all(c7[i] != js[i]["out"] for i in range(10)))


# ---------------------------------------------------------------------------
# [E] randomised differential against the Python oracle
# ---------------------------------------------------------------------------

def check_differential(n=5000):
    rng = random.Random(0x5EC0D)
    recs = []
    expect7 = []
    expect6 = []
    for _ in range(n):
        v = {
            "h": [rng.randrange(1 << 32) for _ in range(8)],
            "m": [rng.randrange(1 << 32) for _ in range(16)],
            "t": rng.randrange(1 << 64),
            "block_len": rng.randrange(0, 65),
            "flags": rng.randrange(0, 256),
        }
        recs.append(encode_record(v))
        expect7.append(ref.compress(v["h"], v["m"], v["t"], v["block_len"],
                                    v["flags"], rounds=7))
        expect6.append(ref.compress(v["h"], v["m"], v["t"], v["block_len"],
                                    v["flags"], rounds=6))

    stdin = "".join(recs)
    for binary, expect, label in ((B3REF7, expect7, 7), (B3REF6, expect6, 6)):
        lines = run(binary, "compress", stdin=stdin).strip().split("\n")
        got = [[int(ln[8 * i:8 * i + 8], 16) for i in range(16)] for ln in lines]
        bad = [i for i in range(n) if got[i] != expect[i]]
        check(f"[E] C vs Python oracle @ rounds={label} ({n} random compressions)",
              len(got) == n and not bad, f"{len(bad)} mismatches, first={bad[:3]}")


def main():
    if not (os.path.exists(B3REF7) and os.path.exists(B3REF6)):
        print("binaries missing -- run ./build.sh first")
        return 2

    print("=" * 74)
    print("SECOND-SOURCE CHECK: upstream BLAKE3 C (round-parameterised)")
    print("=" * 74)
    check_official_vectors()
    check_six_differs()
    check_schedule_equivalence()
    check_canonical_vectors()
    check_differential()
    print("=" * 74)
    if FAILURES:
        print(f"RESULT: {len(FAILURES)} FAILURE(S): {FAILURES}")
        return 1
    print("RESULT: ALL GREEN -- two independent sources agree on the ten")
    print("        6-round vectors, and the 7-round anchor is external.")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    sys.exit(main())
