"""
Pricing for the O1 leaf-convention options, from the gated census formulas.

Validated the same way as `permute-socket-cost.py`: the compress formula must
reproduce the GATED census before it is allowed to price anything else.
"""

import math

# --- gated compress census (CHIP-GATE.md §3, reconciled to the built chip) ---
PREFIX, CELLS_PER_LANE, SENDS_PER_LANE = 28, 4, 2
CELLS_PER_G, SENDS_PER_G = 60, 24
CELLS_PER_OUTW, SENDS_PER_OUTW = 4, 4
IO_SENDS = 6


def compress_ce(rounds=7, extra_main=0):
    g = 8 * rounds
    main = PREFIX + 8 * CELLS_PER_LANE + g * CELLS_PER_G + 4 * CELLS_PER_OUTW + extra_main
    sends = 8 * SENDS_PER_LANE + g * SENDS_PER_G + 4 * SENDS_PER_OUTW + IO_SENDS
    return main + 3 * math.ceil(sends / 2)


assert compress_ce(7) == 5509 and compress_ce(6) == 4741, "formula must match the gate"
print(f"validated: compress = {compress_ce(7)} @7r / {compress_ce(6)} @6r\n")

# --- LFM_BITDEC, ✓ VERIFIED from chips.rs/layout.rs -------------------------
# main = NUM_COLUMNS - PREP_WIDTH = (130+64+2) - 130 = 66
# sends = 1 receiver + 64 bit senders = 65
BITDEC_CE = 66 + 3 * math.ceil(65 / 2)
# LFM_BALU: 4 main (A,B,C,OUT); ~4 LfmMem interactions
BALU_CE = 4 + 3 * math.ceil(4 / 2)
# felt_be_halves = 1 bit_dec + 64 mul/mul_add (32 per half x 2 halves)
FELT_BE_HALVES_CE = BITDEC_CE + 64 * BALU_CE
print(f"LFM_BITDEC per felt      : {BITDEC_CE}")
print(f"LFM_BALU per op          : {BALU_CE}")
print(f"felt_be_halves per felt  : {FELT_BE_HALVES_CE}  (option A's per-felt tax)\n")

# --- FriToyV0 shape, ✓ VERIFIED from programs.rs / fixture.rs ---------------
NUM_QUERIES = 4
# per query today: 3 leaves (1 compress each) + 4 + 4 + 3 path compresses
TODAY_PER_QUERY = 3 + 4 + 4 + 3
TODAY_TRANSCRIPT = 11
TODAY_TOTAL = NUM_QUERIES * TODAY_PER_QUERY + TODAY_TRANSCRIPT

# A leaf covers 2 trace rows = 8 FIELD ELEMENTS. Four felts fill one compress
# input (2 cells x 4 lanes = 8 lanes = 4 felts x 2 halves), so a leaf becomes
# 2 felt-mode compresses + 1 combine = 3.
LEAF_AFTER = 3
AFTER_PER_QUERY = 3 * LEAF_AFTER + 4 + 4 + 3
AFTER_TOTAL = NUM_QUERIES * AFTER_PER_QUERY + TODAY_TRANSCRIPT

# Felts needing a decomposition: leaf data only. Siblings and internal nodes are
# DIGESTS, already u32-laned by obligation O2.
LEAF_FELTS = NUM_QUERIES * 24 + 8        # 6 cells x 4 felts per query, + t0w/t1w

print(f"FriToyV0 compresses  today(counterfactual)={TODAY_TOTAL}  after={AFTER_TOTAL}")
print(f"felts needing decomposition: {LEAF_FELTS}\n")

base = TODAY_TOTAL * compress_ce(7)
opt_a = AFTER_TOTAL * compress_ce(7) + LEAF_FELTS * FELT_BE_HALVES_CE
# Option C: the canonicity gate lives IN the socket. Per felt: Z + GINV = 2
# witness columns; 4 felts per row => +8 main columns, ZERO extra sends.
opt_c = AFTER_TOTAL * compress_ce(7, extra_main=8)

print("FriToyV0 end-to-end, cell-equiv @7r")
print(f"  counterfactual 'if felts fit'  {base:>9,}   (todays 369,103 shape)")
print(f"  (A) felt_be_halves precedent   {opt_a:>9,}   {100*(opt_a/base-1):+.1f}%")
print(f"  (C) in-socket felt mode        {opt_c:>9,}   {100*(opt_c/base-1):+.1f}%")
print(f"  (B) stay off BLAKE3            {'n/a':>9}    0%  (nothing is proved)")
print(f"\n  C is {100*(1-opt_c/opt_a):.1f}% cheaper than A, and adds "
      f"{compress_ce(7,8)-compress_ce(7)} cells/row rather than "
      f"{FELT_BE_HALVES_CE} per felt.")
