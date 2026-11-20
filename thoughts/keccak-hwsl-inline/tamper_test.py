"""Tamper demo: CHANGED constraints and REMOVED constraints must both flip
the gate from UNSAT (verified) to SAT (forgeable). Clean model run first as
the control."""
from z3 import Solver, And, Or, Concat, Extract, sat, unsat, is_true
import z3_verify as zv
from keccak_ref import RC

ROUND = 1

print("=== control: clean model, round 1 ===")
r = zv.check_round(ROUND)
print(f"  clean              -> {r}   (want unsat)  {'OK' if r == unsat else '!!! BROKEN'}")
assert r == unsat

CASES = [
    ("iota_wrong_rc",     "CHANGED"),
    ("rho_off_by_one",    "CHANGED"),
    ("drop_chi_xor_byte", "REMOVED"),
    ("drop_hwsl_carry",   "REMOVED"),
]

print("\n=== tampered models: gate must catch every one (SAT) ===")
for bug, kind in CASES:
    r = zv.check_round(ROUND, bug=bug)
    ok = "OK — gate catches it" if r == sat else "!!! GATE MISSED THE BUG"
    print(f"  {kind}  {bug:18s} -> {r}   {ok}")
    assert r == sat, f"gate failed to catch {bug}"

# ---- exhibit the forged witnesses for the two REMOVED cases -----------------
def exhibit(bug):
    C, out_byte, start = zv.build_circuit(ROUND, f"cex_{bug}", bug=bug)
    lanes = [[Concat(*[start[(x, y, b)] for b in reversed(range(8))])
              for y in range(5)] for x in range(5)]
    ref = zv.zref_round(lanes, RC[ROUND])
    rb = lambda x, y, b: Extract(8 * b + 7, 8 * b, ref[x][y])
    s = Solver()
    s.add(And(*C))
    s.add(Or(*[out_byte(x, y, b) != rb(x, y, b)
               for x in range(5) for y in range(5) for b in range(8)]))
    assert s.check() == sat
    m = s.model()
    bad = [(x, y, b) for x in range(5) for y in range(5) for b in range(8)
           if is_true(m.evaluate(out_byte(x, y, b) != rb(x, y, b)))]
    lanes_hit = sorted(set((x, y) for x, y, _ in bad))
    print(f"  {bug}: forged witness makes {len(bad)} output bytes wrong; "
          f"lanes hit: {lanes_hit[:8]}{'...' if len(lanes_hit) > 8 else ''}")

print("\n=== forged-witness exhibits (REMOVED cases) ===")
exhibit("drop_chi_xor_byte")
exhibit("drop_hwsl_carry")

print("\nVERDICT: clean=UNSAT, all 4 tampers=SAT -> gate is sensitive to both "
      "changed AND removed constraints.")
