"""
LAYER 1 ANCHOR: certify `blake3_oracle.py` at rounds = 7 against external truth.

Three independent anchors, in decreasing order of strength:

  A1  OFFICIAL BLAKE3 test vectors (upstream `test_vectors.json`), all three
      modes (hash / keyed_hash / derive_key), 35 input lengths, extended output.
      This is external to this repo and to this project.
  A2  The upstream-published `known` digests for the empty string and "abc".
  A3  Differential against the independently-written in-repo reference
      `thoughts/blake3/blake3-oracle/blake3_ref.py` (if reachable), on random
      compression inputs at BOTH round counts.

A1/A2 certify the 7-round code path.  A3 additionally certifies that the SAME
code path at rounds = 6 agrees with a second implementation -- which is all that
can be said for 6 rounds, since no external anchor for it exists (assumption A6R).

Run: python3 anchor_check.py
"""

from __future__ import annotations

import json
import os
import random
import sys

import blake3_oracle as ora

HERE = os.path.dirname(os.path.abspath(__file__))

# Search paths for the restored phase-1 artifacts (worktree first, then repo).
_CANDIDATE_ROOTS = [
    "/Users/maurofab/workspace/lambda_vm-blake3-impl/thoughts/blake3",
    os.path.join(HERE, "..", "..", "..", "blake3"),
]


def _find(rel: str):
    for root in _CANDIDATE_ROOTS:
        p = os.path.join(root, rel)
        if os.path.exists(p):
            return p
    return None


def official_input(length: int) -> bytes:
    """Upstream's test input: the repeating byte pattern i % 251."""
    return bytes((i % 251) for i in range(length))


class _Xorshift64Star:
    """The `random` block's inputs are not the 251-pattern -- they come from a
    self-contained xorshift64* stream (`ground-truth/src/main.rs`, struct Rng),
    deliberately re-implemented here rather than shared, so the Python and Rust
    sides agree only if both are right."""

    M64 = (1 << 64) - 1

    def __init__(self, seed: int):
        self.x = seed & self.M64

    def next_u64(self) -> int:
        x = self.x
        x ^= x >> 12
        x = (x ^ (x << 25)) & self.M64
        x ^= x >> 27
        self.x = x
        return (x * 0x2545F4914F6CDD1D) & self.M64

    def byte(self) -> int:
        return (self.next_u64() >> 33) & 0xFF

    def bytes_(self, n: int) -> bytes:
        return bytes(self.byte() for _ in range(n))


def anchor_official_vectors() -> tuple[bool, str]:
    path = _find("blake3-oracle/official_test_vectors.json")
    if path is None:
        return False, "official_test_vectors.json NOT FOUND -- anchor A1 CANNOT RUN"
    with open(path) as f:
        vec = json.load(f)

    key = vec["key"].encode("utf-8")
    assert len(key) == 32, "official key must be 32 bytes"
    ctx = vec["context_string"]

    n_hash = n_keyed = n_derive = 0
    for case in vec["cases"]:
        data = official_input(case["input_len"])
        want_hash = bytes.fromhex(case["hash"])
        got = ora.hash_bytes(data, len(want_hash))
        if got != want_hash:
            return False, (f"HASH mismatch at input_len={case['input_len']}: "
                           f"got {got.hex()[:64]} want {want_hash.hex()[:64]}")
        n_hash += 1

        want_keyed = bytes.fromhex(case["keyed_hash"])
        got = ora.Hasher.new_keyed(key).update(data).finalize(len(want_keyed))
        if got != want_keyed:
            return False, f"KEYED mismatch at input_len={case['input_len']}"
        n_keyed += 1

        want_derive = bytes.fromhex(case["derive_key"])
        got = ora.Hasher.new_derive_key(ctx).update(data).finalize(len(want_derive))
        if got != want_derive:
            return False, f"DERIVE_KEY mismatch at input_len={case['input_len']}"
        n_derive += 1

    # The `random` block: independent seeds, short XOF windows, all three modes.
    n_rand = 0
    for case in vec.get("random", []):
        data = _Xorshift64Star(case["seed"]).bytes_(case["len"])
        xof = case["xof"]
        if bytes.fromhex(case["hash"]) != ora.hash_bytes(data, xof):
            return False, f"random HASH mismatch seed={case['seed']} len={case['len']}"
        k = bytes.fromhex(case["key"])
        if bytes.fromhex(case["keyed"]) != ora.Hasher.new_keyed(k).update(data).finalize(xof):
            return False, f"random KEYED mismatch seed={case['seed']}"
        if bytes.fromhex(case["derive"]) != (
                ora.Hasher.new_derive_key(case["ctx"]).update(data).finalize(xof)):
            return False, f"random DERIVE mismatch seed={case['seed']}"
        n_rand += 1

    known = vec.get("known", {})
    for name, want in known.items():
        data = b"" if name == "empty" else name.encode()
        if ora.hash_bytes(data, len(want) // 2).hex() != want:
            return False, f"known-digest mismatch: {name}"

    return True, (f"A1 PASS: {n_hash} hash + {n_keyed} keyed + {n_derive} derive_key "
                  f"cases, {n_rand} random cases, {len(known)} known digests")


def anchor_differential(trials: int = 200) -> tuple[bool, str]:
    path = _find("blake3-oracle/blake3_ref.py")
    if path is None:
        return False, "blake3_ref.py NOT FOUND -- anchor A3 CANNOT RUN"
    sys.path.insert(0, os.path.dirname(path))
    try:
        import blake3_ref as other  # type: ignore
    except Exception as exc:  # pragma: no cover
        return False, f"blake3_ref.py import failed: {exc}"

    if not hasattr(other, "compress"):
        return False, "blake3_ref.py has no `compress` -- differential CANNOT RUN"

    rng = random.Random(0xB3_0A_11)
    for rounds in (6, 7):
        for _ in range(trials):
            h = [rng.randrange(1 << 32) for _ in range(8)]
            m = [rng.randrange(1 << 32) for _ in range(16)]
            t = rng.randrange(1 << 64)
            bl = rng.randrange(65)
            fl = rng.randrange(128)
            mine = ora.compress(h, m, t, bl, fl, rounds=rounds)
            theirs = other.compress(h, m, t, bl, fl, rounds=rounds)
            if list(mine) != list(theirs):
                return False, (f"DIFFERENTIAL mismatch at rounds={rounds}\n"
                               f"  mine  ={[hex(x) for x in mine]}\n"
                               f"  theirs={[hex(x) for x in theirs]}")
    return True, (f"A3 PASS: {trials} random compressions x rounds in (6,7) agree "
                  f"with {os.path.relpath(path, HERE)}")


def negative_control_anchor() -> tuple[bool, str]:
    """A1 is only meaningful if a perturbed oracle FAILS it.  Four perturbations,
    each breaking exactly one convention, must each break the official vectors."""
    data = official_input(1024 + 5)
    good = ora.hash_bytes(data)

    fails = []

    # (i) wrong round count.
    if ora.hash_bytes(data, rounds=6) != good:
        fails.append("rounds=6")

    # (ii) message permutation perturbed.
    saved = list(ora.MSG_PERMUTATION)
    ora.MSG_PERMUTATION[0], ora.MSG_PERMUTATION[1] = saved[1], saved[0]
    try:
        if ora.hash_bytes(data) != good:
            fails.append("msg_permutation_swapped")
    finally:
        ora.MSG_PERMUTATION[:] = saved

    # (iii) IV perturbed.
    saved_iv = list(ora.IV)
    ora.IV[0] ^= 1
    try:
        if ora.hash_bytes(data) != good:
            fails.append("iv_bit_flipped")
    finally:
        ora.IV[:] = saved_iv

    # (iv) counter halves swapped (only observable with >1 chunk, hence the size).
    saved_compress = ora.compress

    def swapped(cv, bw, counter, bl, fl, rounds=ora.STANDARD_ROUNDS):
        c = ((counter & ora.MASK32) << 32) | ((counter >> 32) & ora.MASK32)
        return saved_compress(cv, bw, c, bl, fl, rounds)

    ora.compress = swapped
    try:
        # Rebuild the tree path through the patched compress.
        if ora.Hasher().update(data).finalize() != good:
            fails.append("counter_halves_swapped")
    finally:
        ora.compress = saved_compress

    want = {"rounds=6", "msg_permutation_swapped", "iv_bit_flipped",
            "counter_halves_swapped"}
    missing = want - set(fails)
    if missing:
        return False, f"NEGATIVE CONTROL FAILED -- these perturbations went undetected: {sorted(missing)}"
    return True, f"NC PASS: all 4 single-convention perturbations break the anchor"


def main() -> int:
    print("=" * 74)
    print("LAYER 1 ANCHOR CHECK -- blake3_oracle.py")
    print("=" * 74)
    results = []
    for name, fn in (("A1 official vectors", anchor_official_vectors),
                     ("A3 differential", anchor_differential),
                     ("NC anchor sensitivity", negative_control_anchor)):
        ok, msg = fn()
        results.append(ok)
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {msg}")
    ok = all(results)
    print("-" * 74)
    print(f"LAYER 1: {'ANCHORED' if ok else 'NOT ANCHORED -- do not build on this'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
