"""
The pure-combinatorial premises the theta and rho necessity arguments rest on.

No solver: these are facts about cols::pi_src_cols, cols::cxz_right_bit_for_byte
and KECCAK_RHO that must hold before any claim about "this range check is
implied by the ByteAlu operand" can mean anything. The load-bearing one is
READ-ONCE (sections 3 and 6): `operand_summand_window` bounds a column from the
single operand byte that reads it, and a column read twice would need the
intersection of two windows instead.

`premises()` is imported and run by necessity_theta.py and necessity_rho.py, so
the checks below cannot be skipped by forgetting to run this file first.
"""
from keccak_ref import RHO
from field_model import rho_pi_offsets, theta_carry_source


def premises(verbose=True):
    """Assert every premise. Raises AssertionError naming the ones that fail."""
    failed = []

    def check(cond, msg):
        if verbose:
            print(f"  {'OK  ' if cond else 'FAIL'} {msg}")
        if not cond:
            failed.append(msg)

    def say(msg):
        if verbose:
            print(msg)

    say("=== (1) pi is a bijection on the 25 lanes ===")
    src_of = {(X, Y): ((X + 3 * Y) % 5, X) for X in range(5) for Y in range(5)}
    images = list(src_of.values())
    check(len(set(images)) == 25,
          f"(X,Y) -> ((X+3Y)%5, X) covers {len(set(images))}/25 source lanes, no repeats")

    say("\n=== (2) every source lane is read by exactly one output lane, via 8 bytes ===")
    readers = {}
    for (X, Y), src in src_of.items():
        readers.setdefault(src, []).append((X, Y))
    check(all(len(v) == 1 for v in readers.values()),
          "each source lane has exactly one reader lane")

    say("\n=== (3) every rot_left and rot_right byte column is read EXACTLY once ===")
    bad = []
    for src, ((X, Y),) in ((s, tuple(r)) for s, r in readers.items()):
        a = rho_pi_offsets(RHO[src[0]][src[1]] // 16)
        left_hits = [0] * 8
        right_hits = [0] * 8
        for z in range(8):
            left_hits[(z + a) % 8] += 1
            right_hits[(z + a - 2) % 8] += 1
        if left_hits != [1] * 8 or right_hits != [1] * 8:
            bad.append((src, left_hits, right_hits))
    check(not bad, "400/400 rho byte columns read exactly once (none zero times, none twice)"
                   f"{'' if not bad else f' — {bad[:2]}'}")

    say("\n=== (4) the pi byte offsets are EVEN, so a pi halfword reads one source halfword ===")
    odd = [(x, y) for x in range(5) for y in range(5) if rho_pi_offsets(RHO[x][y] // 16) % 2]
    check(not odd, "a in {0,6,4,2} is always even -> P_h = L_(h+A) + R_(h+A-1), A = a/2")
    mism = []
    for x in range(5):
        for y in range(5):
            a = rho_pi_offsets(RHO[x][y] // 16)
            A = a // 2
            for h in range(4):
                if ((2 * h + a) % 8) // 2 != (h + A) % 4 or ((2 * h + a - 2) % 8) // 2 != (h + A - 1) % 4:
                    mism.append((x, y, h))
    check(not mism, "the packed relation verified for all 25 lanes x 4 halfwords")

    say("\n=== (5) theta = all-ones saturates every pi halfword, for EVERY rotation ===")
    # left + right = 0xFFFF whatever rnc is, which is why config C forges on all 25.
    sat = all(((0xFFFF << (RHO[x][y] % 16)) & 0xFFFF) + (0xFFFF >> (16 - (RHO[x][y] % 16))
              if RHO[x][y] % 16 else 0) == 0xFFFF for x in range(5) for y in range(5))
    check(sat, "left + right = 0xFFFF for all 25 lanes -> pi = 0xFF..FF, the saturation config C needs")

    say("\n=== (6) the theta analogue: every Cxz_right carry column is read EXACTLY once ===")
    # cols::cxz_right_bit_for_byte sends the carry of halfword h-1 to the LOW byte
    # of halfword h and nothing to the odd bytes, so the four carries of one x are
    # a permutation of the four rotated_C low bytes. Without this, the carry has no
    # single operand window and theta's config C says nothing.
    sources = [theta_carry_source(h) for h in range(4)]
    check(sorted(sources) == [0, 1, 2, 3],
          f"theta_carry_source is a bijection on the 4 halfwords ({sources})")
    check(all(theta_carry_source(h) != h for h in range(4)),
          "no carry lands on its own halfword -> the cycle has no fixed point")

    assert not failed, failed
    if verbose:
        print("\nALL COMBINATORIAL PREMISES HOLD")


if __name__ == "__main__":
    premises()
