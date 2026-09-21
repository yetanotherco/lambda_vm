"""Validate the circuit model FORWARD against the reference, and confirm every
negative control genuinely perturbs the output.

A symbolic UNSAT cannot reveal a wholesale wrong model — if the equations I
hand-transcribed from sha256_round.rs describe some *other* function, z3 will
happily prove the model equals itself. This file is the guard against that.
"""
import random
from model_dataflow import round_dataflow, sigma_from_bits, word_to_bits
from sha256_ref import K, round_fn, big_sigma0, big_sigma1, ch, maj, MASK32

print("=== circuit model vs reference, random rounds ===")
rng = random.Random(0xBEEF)
for _ in range(2000):
    st = [rng.randrange(1 << 32) for _ in range(8)]
    w, t = rng.randrange(1 << 32), rng.randrange(64)
    mine, _ = round_dataflow(st, K[t], w)
    assert mine == round_fn(st, w, K[t]), (st, w, t)
print("  2000/2000 match")

print("\n=== structured inputs ===")
structured = [
    [0] * 8,
    [MASK32] * 8,
    [1 << i for i in range(8)],
    [MASK32 ^ (1 << i) for i in range(8)],
    [0x80000000] * 8,
    [0, MASK32, 0, MASK32, 0, MASK32, 0, MASK32],
]
for st in structured:
    for t in (0, 1, 16, 63):
        for w in (0, MASK32, 1, 0x80000000):
            mine, _ = round_dataflow(st, K[t], w)
            assert mine == round_fn(st, w, K[t]), (st, t, w)
print(f"  {len(structured)} states x 4 rounds x 4 words all match")

print("\n=== Sigma over bit columns equals the spec's rotate-xor ===")
for _ in range(2000):
    v = rng.randrange(1 << 32)
    bits = word_to_bits(v)
    assert sigma_from_bits(bits, 2) == big_sigma0(v), hex(v)
    assert sigma_from_bits(bits, 3) == big_sigma1(v), hex(v)
print("  2000/2000 Sigma0 and Sigma1 match")

print("\n=== the circuit sums Ch and Maj where the spec XORs them ===")
# The AIR adds the two BYTE_ALU results instead of XOR-ing. That is only valid
# because the operands are bit-disjoint. Check the identity rather than assume.
for _ in range(5000):
    x, y, z = (rng.randrange(1 << 32) for _ in range(3))
    assert ((x & y) + ((~x & MASK32) & z)) == ch(x, y, z)
    assert ((x & y) + (z & (x ^ y))) == maj(x, y, z)
print("  5000/5000: the + and ^ forms agree, so the disjointness holds")

print("\n=== negative controls perturb the output (falsifiability) ===")
bugs = ["maj_uses_b_not_c", "ch_no_not", "sigma0_off_by_one", "sigma1_dropped",
        "sigma1_is_sigma0", "drop_h_from_temp1", "carry_dropped"]
rows = [([rng.randrange(1 << 32) for _ in range(8)], rng.randrange(1 << 32), rng.randrange(64))
        for _ in range(200)]
for bug in bugs:
    caught = 0
    for st, w, t in rows:
        good, gaux = round_dataflow(st, K[t], w)
        bad, baux = round_dataflow(st, K[t], w, bug=bug)
        if bad != good or baux["carry_a"] != gaux["carry_a"]:
            caught += 1
    print(f"  {bug:<22} perturbs {caught}/{len(rows)} rows")
    assert caught > 0, f"{bug} does not change anything — the control is vacuous"

print("\nALL DATAFLOW VALIDATIONS PASSED")
