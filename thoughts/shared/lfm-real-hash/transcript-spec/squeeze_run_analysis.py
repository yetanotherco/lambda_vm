"""
THE SQUEEZE-RUN ANALYSIS — the FSE-2014 lesson, written into this construction's
own spec rather than left in an options-paper appendix.

A "squeeze run" is a maximal sequence of consecutive squeezes with no absorb
between them. Within a run the transcript advances by repeatedly applying a
non-injective map to a 128-bit state, so the reachable state set shrinks. This
is the same phenomenon that broke GLUON-64 (Collision Spectrum, Entropy Loss,
T-Sponges, FSE 2014), and option A was NOT the only construction exposed to it —
option B is too, which is why it belongs here.

MODEL. `compress_T(·, operand)` with a fixed operand is a map on 2^128 points;
model it as random. For a random map on N points, the image after one
application has expected size N(1 - e^{-1}); iterating gives the recursion

    alpha_{j+1} = 1 - exp(-alpha_j),    alpha_0 = 1

with alpha_k ~ 2/k asymptotically (Flajolet-Odlyzko). Entropy loss after a run of
length k is -log2(alpha_k) bits of the state's 128.

COMPOSING DISTINCT MAPS DOES NOT ESCAPE THIS. With the squeeze counter each step
is a different map, but the same recursion governs the image of a composition of
independent random maps, so the bit-counting is unchanged. What the counter
removes is the *attack structure* — a single fixed public map has ONE functional
graph (rho-shapes, deep nodes, cycles) that an adversary can precompute and that
the T-sponge attacks exploit. That distinction is the whole point and it is why
the counter is in the spec even though the numbers below say the loss is
irrelevant either way.
"""

from __future__ import annotations

import math

STATE_BITS = 128


def alpha(k: int) -> float:
    """Fraction of the state space still reachable after a run of length k."""
    a = 1.0
    for _ in range(k):
        a = 1.0 - math.exp(-a)
    return a


def loss_bits(k: int) -> float:
    return -math.log2(alpha(k)) if k > 0 else 0.0


# --- (b) run lengths in the ACTUAL programs, ✓ VERIFIED ---------------------
# FriToyV0's sponge sequence (programs.rs:549-567):
#   absorb, squeeze, squeeze, absorb, squeeze, absorb2, then NUM_QUERIES
#   squeezes with NO absorb in the loop body.
FRI_TOY_RUNS = [2, 1, 4]          # NUM_QUERIES = 4 (fixture.rs:37)
FRI_TOY_MAX_RUN = max(FRI_TOY_RUNS)

# TrivialV0 has no sponge at all (it calls b.permute directly; see the spec's
# TrivialV0 section).
TRIVIAL_RUNS: list[int] = []


def main() -> None:
    print("=" * 74)
    print("SQUEEZE-RUN ANALYSIS — entropy loss vs run length")
    print("=" * 74)
    print(f"  state = {STATE_BITS} bits (one cell)\n")
    print("   run k | reachable fraction | loss (bits) | state left")
    print("  -------+--------------------+-------------+-----------")
    for k in (1, 2, 4, 8, 16, 64, 256, 1024, 4096, 65536):
        a, l = alpha(k), loss_bits(k)
        print(f"  {k:6d} | {a:18.6f} | {l:11.2f} | {STATE_BITS - l:9.2f}")

    print(f"\n  asymptotic check, alpha_k ~ 2/k:")
    for k in (1024, 65536):
        print(f"    k={k:6d}: alpha={alpha(k):.3e}  2/k={2/k:.3e}")

    print("\n" + "-" * 74)
    print("(b) RUN LENGTHS IN THE PROGRAMS AS THEY EXIST  ✓ VERIFIED")
    print(f"  FriToyV0 runs: {FRI_TOY_RUNS}  -> MAX RUN = {FRI_TOY_MAX_RUN}")
    print(f"     loss at k={FRI_TOY_MAX_RUN}: {loss_bits(FRI_TOY_MAX_RUN):.2f} bits "
          f"of {STATE_BITS}")
    print("  TrivialV0: no sponge (raw permute; see spec)")
    print("  NOTE: the max run IS NUM_QUERIES — it scales with the query count,")
    print("        so a production FRI (100-200 queries) would have a run that")
    print("        long. That is the regime worth stating a bound for.")
    for k in (128, 256):
        print(f"     hypothetical production run k={k}: {loss_bits(k):.2f} bits")

    print("\n" + "-" * 74)
    print("(c) GUIDANCE BOUND")
    for target in (1.0, 4.0, 8.0):
        k = 1
        while loss_bits(k) < target:
            k *= 2
        print(f"  loss stays under {target:.0f} bit(s) for runs up to k ~ {k//2}")
    print(f"  The birthday bound on a {STATE_BITS}-bit state (2^{STATE_BITS//2}) "
          f"dominates long")
    print("  before image shrinkage matters: even k = 2^16 costs under 16 bits.")


if __name__ == "__main__":
    main()
