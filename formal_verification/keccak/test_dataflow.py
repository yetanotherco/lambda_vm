"""Validate the concrete contract-dataflow against the independent reference."""
import random
from keccak_ref import RC, keccak_round, keccak_f1600
from model_dataflow import round_dataflow

rng = random.Random(0xC0FFEE)
M = (1 << 64) - 1

print("=== per-round: contract dataflow vs FIPS-202 reference (random states) ===")
ok = True
for trial in range(200):
    st = [rng.randrange(0, 1 << 64) for _ in range(25)]
    r = rng.randrange(0, 24)
    got = round_dataflow(st, r)
    exp = keccak_round(st, RC[r])
    if got != exp:
        ok = False
        print(f"  MISMATCH trial={trial} round={r}")
        break
print("  200 random single rounds match:", ok)
assert ok

print("\n=== 24-round chain (full permutation) via dataflow vs reference ===")
ok2 = True
for trial in range(20):
    st = [rng.randrange(0, 1 << 64) for _ in range(25)]
    s = list(st)
    for r in range(24):
        s = round_dataflow(s, r)
    if s != keccak_f1600(st):
        ok2 = False
        print(f"  MISMATCH trial={trial}")
        break
print("  20 full 24-round permutations match:", ok2)
assert ok2

# Also: all-zero and specific structured inputs
for st in ([0] * 25, [1] * 25, list(range(25)), [M] * 25):
    s = list(st)
    for r in range(24):
        s = round_dataflow(s, r)
    assert s == keccak_f1600(st), st[:3]
print("  structured inputs (zeros/ones/range/all-FF) match: True")

print("\n=== sanity: each injected bug DOES change output (concrete) ===")
st = [rng.randrange(0, 1 << 64) for _ in range(25)]
for bug in ["theta_no_rot", "rho_swap", "chi_no_not", "chi_swap", "iota_no_rc"]:
    changed = any(round_dataflow(st, r, bug=bug) != keccak_round(st, RC[r]) for r in range(24))
    print(f"  bug={bug:14s} perturbs output: {changed}")
    assert changed, bug

print("\nALL DATAFLOW VALIDATIONS PASSED")
