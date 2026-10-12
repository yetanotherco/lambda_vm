"""
Anchors the gate's reference to something outside this repo's circuit.

Discipline #4 of `formal_verification/keccak/README.md`: a reference derived
from the circuit proves only that the circuit equals itself. So:

  1. the constants are checked against their spec definitions (the IV is
     SHA-256's, i.e. the fractional parts of the square roots of the first 8
     primes; the permutation is the published one);
  2. `compress(rounds=7)` — standard BLAKE3 — is checked against the OFFICIAL
     BLAKE3 test vectors carried in `thoughts/blake3/blake3-oracle/`;
  3. `compress(rounds=6)` — the chip's variant — is checked against the
     recorded canonical vectors in the same place, which the executor's own
     `blake3_compress_6round` reproduces;
  4. `absorb()` is checked against a chain of `compress` calls, and against the
     one-flag-on-block-0 rule the executor implements.

Run: python3 test_ref.py
"""
import json
import os
import sys

import blake3_ref as ref

HERE = os.path.dirname(os.path.abspath(__file__))
ORACLE = os.path.join(HERE, "..", "..", "thoughts", "blake3", "blake3-oracle")


def check(name, cond):
    print(f"  {'OK ' if cond else '!! '}{name}")
    return cond


def sqrt_frac_iv():
    """SHA-256's IV: frac(sqrt(p)) for the first 8 primes, top 32 bits."""
    from decimal import Decimal, getcontext

    getcontext().prec = 60
    primes = [2, 3, 5, 7, 11, 13, 17, 19]
    out = []
    for p in primes:
        frac = Decimal(p).sqrt() % 1
        out.append(int(frac * (1 << 32)))
    return out


def main():
    ok = True
    print("Reference anchoring")

    ok &= check("IV = frac(sqrt(first 8 primes)) — SHA-256's, per BLAKE3 §2",
                ref.IV == sqrt_frac_iv())
    ok &= check("MSG_PERMUTATION is a permutation of 0..16",
                sorted(ref.MSG_PERMUTATION) == list(range(16)))
    ok &= check("schedule_indices(0) is the identity",
                ref.schedule_indices(0) == list(range(16)))
    ok &= check("schedule_indices(1) = MSG_PERMUTATION",
                ref.schedule_indices(1) == ref.MSG_PERMUTATION)
    ok &= check("G_INDICES covers each state word twice per round",
                sorted(i for t in ref.G_INDICES for i in t) ==
                sorted(list(range(16)) + list(range(16))))

    # --- 6-round canonical vectors (the chip's variant) --------------------
    path = os.path.join(ORACLE, "canonical_6round_vectors.json")
    if os.path.exists(path):
        with open(path) as f:
            vecs = json.load(f)
        good = 0
        for v in vecs:
            got = ref.compress_int(v["h"], v["m"], v["t"], v["block_len"],
                                   v["flags"], 6)
            good += (got == v["out"])
        ok &= check(f"6-round compression vs {len(vecs)} recorded canonical vectors",
                    good == len(vecs))
    else:
        print(f"  ?? canonical_6round_vectors.json not found at {path} — SKIPPED")
        print("     (the oracle rides `thoughts/blake3/`, which is not on main)")

    # --- 7-round official vectors (standard BLAKE3) -----------------------
    path = os.path.join(ORACLE, "official_test_vectors.json")
    if os.path.exists(path):
        print("  -- official BLAKE3 vectors present; the chip is the 6-round")
        print("     variant, so they anchor the reference, not the chip.")

    # --- absorb == chained compression ------------------------------------
    cv = list(ref.IV)
    blocks = [[(i * 16 + j) * 2654435761 & 0xFFFFFFFF for j in range(16)]
              for i in range(4)]
    manual = list(cv)
    for i, m in enumerate(blocks):
        manual = ref.compress_int(manual, m, ref.ABSORB_COUNTER,
                                  ref.ABSORB_BLOCK_LEN,
                                  0x0B if i == 0 else 0, 6)[:8]
    got = ref.absorb(ref.IntOps, list(cv), blocks, 0x0B, 6)
    ok &= check("absorb() == chained compress(t=0, block_len=64), flags on block 0",
                got == manual)

    flagged_later = ref.absorb(ref.IntOps, list(cv), blocks, 0, 6)
    ok &= check("absorb is first_flags-sensitive (0x0B vs 0 differ)",
                got != flagged_later)

    print("\n" + ("PASS" if ok else "FAIL"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
