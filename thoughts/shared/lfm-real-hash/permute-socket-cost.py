"""
Cost model for the candidate permute sockets, VALIDATED against the gated census.

The gate's own model (`gate-oracle/chip_model.py`) is pinned to the committed
chip, so it must not be edited for a costing exercise. Instead the per-item costs
are re-expressed as a closed formula here and the formula is CHECKED against the
gated numbers first: if it cannot reproduce compress at both round counts, it is
not allowed to price anything else.

Per-item costs, all from the gated model:
  frozen socket prefix   28 cells   (12 IN + 4 S + 12 OUT; MU is preprocessed)
  input lane             4 cells,  2 AreBytes sends   (a lane's 4 byte columns)
  G-instance             60 cells, 24 sends           (16 ByteAlu[XOR] + 8 AreBytes)
  output word            4 cells,  4 ByteAlu[XOR] sends
  host LfmMem tuples     6 sends
  aux                    3 * ceil(sends / 2)
"""

import math

PREFIX = 28
CELLS_PER_LANE, SENDS_PER_LANE = 4, 2
CELLS_PER_G, SENDS_PER_G = 60, 24
CELLS_PER_OUTW, SENDS_PER_OUTW = 4, 4
IO_SENDS = 6


def census(rounds: int, lanes: int, out_words: int) -> dict:
    num_g = 8 * rounds
    main = (PREFIX + lanes * CELLS_PER_LANE + num_g * CELLS_PER_G
            + out_words * CELLS_PER_OUTW)
    sends = (lanes * SENDS_PER_LANE + num_g * SENDS_PER_G
             + out_words * SENDS_PER_OUTW + IO_SENDS)
    aux = 3 * math.ceil(sends / 2)
    return {"main": main, "sends": sends, "aux": aux, "cell_equiv": main + aux}


# --- VALIDATION: the formula must reproduce the GATED compress census ---------
GATED = {7: {"main": 3436, "sends": 1382, "aux": 2073, "cell_equiv": 5509},
         6: {"main": 2956, "sends": 1190, "aux": 1785, "cell_equiv": 4741}}

print("VALIDATION -- formula vs the gated compress census (lanes=8, out=4)")
ok = True
for r, want in GATED.items():
    got = census(r, lanes=8, out_words=4)
    match = got == want
    ok &= match
    print(f"  {r}r: {got}  {'MATCH' if match else 'MISMATCH vs ' + str(want)}")
if not ok:
    raise SystemExit("formula does not reproduce the gated census -- refusing to price")

print("\nCOMPRESS socket (as built): lanes=8 (two cells), out=4 (one cell)")
for r in (7, 6):
    print(f"  {r}r: {census(r, 8, 4)}")

print("\nOPTION A permute socket: lanes=12 (three cells), out=12 (three cells)")
for r in (7, 6):
    c = census(r, 12, 12)
    ratio = c["cell_equiv"] / census(r, 8, 4)["cell_equiv"]
    print(f"  {r}r: {c}   = {ratio:.3f} x one compress")

print("\nPER-PROGRAM (FriToyV0: 10 permutes + 56 compresses; counted from "
      "programs.rs)")
for r in (7, 6):
    comp = census(r, 8, 4)["cell_equiv"]
    perm = census(r, 12, 12)["cell_equiv"]
    a = 10 * perm + 56 * comp
    # Option B: the sponge becomes compress-based; 11 compresses replace the
    # 10 permutes (see the options paper for the op-by-op derivation).
    b = (11 + 56) * comp
    print(f"  {r}r  option A: {a:,} cell-equiv   option B: {b:,}   "
          f"B/A = {b/a:.3f}")

print("\nTrivialV0: 2 compresses + 1 permute")
for r in (7, 6):
    comp = census(r, 8, 4)["cell_equiv"]
    perm = census(r, 12, 12)["cell_equiv"]
    print(f"  {r}r  option A: {2*comp + perm:,}   option B (3 compresses): "
          f"{3*comp:,}")
