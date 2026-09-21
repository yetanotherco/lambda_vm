"""Re-check every `sha256_*.rs:NN` citation in this directory against the files.

The citations are the gate's only audit trail into the Rust: the README makes
faithfulness of the model to the circuit a human obligation, and the citations
are how a reviewer discharges it. PR #950 found every citation in the keccak
gate stale — one had been wrong for three days because an unrelated PR grew the
file. This script makes that a check instead of a hope.

It verifies the cited line ranges still contain the identifier the comment
names, not merely that the file is long enough.
"""
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
RS = {
    "sha256_round.rs": ROOT / "prover/src/tables/sha256_round.rs",
    "sha256_schedule.rs": ROOT / "prover/src/tables/sha256_schedule.rs",
    "sha256_common.rs": ROOT / "prover/src/tables/sha256_common.rs",
    "sha256.rs": ROOT / "executor/src/sha256.rs",
    "templates.rs": ROOT / "prover/src/constraints/templates.rs",
    "core.rs": ROOT / "prover/src/tables/sha256.rs",
}

# what each cited range must still contain, so a citation that drifted onto
# unrelated code is caught rather than merely one that ran off the end
EXPECT = [
    ("sha256_round.rs", 28, 55, "pub const MU"),
    ("sha256_round.rs", 136, 153, "BusId::ByteAlu"),
    ("sha256_round.rs", 163, 166, "BusId::IsHalfword"),
    ("sha256_round.rs", 168, 168, "BusId::AreBytes"),
    ("sha256_round.rs", 192, 193, "BusId::ShaRound"),
    ("sha256_round.rs", 208, 212, "1u64 << (8 * j)"),
    ("sha256_round.rs", 216, 218, "check_bits"),
    ("sha256_round.rs", 222, 223, "let maj"),
    ("sha256_round.rs", 227, 235, "CARRY_E"),
    ("sha256_schedule.rs", 28, 43, "pub const MU"),
    ("sha256_schedule.rs", 98, 100, "BusId::IsHalfword"),
    ("sha256_schedule.rs", 101, 106, "BusId::AreBytes"),
    ("sha256_schedule.rs", 166, 172, "CARRY"),
    ("sha256_schedule.rs", 174, 174, "AMOUNT"),
    ("sha256_schedule.rs", 158, 160, "check_bits"),
    ("sha256.rs", 3, 12, "pub const K"),
    ("sha256.rs", 13, 15, "pub const IV"),
    ("templates.rs", 334, 374, "pub fn emit_add_pair"),
    ("templates.rs", 342, 347, "INV_SHIFT_32"),
]

bad = 0
for fname, lo, hi, needle in EXPECT:
    path = RS[fname]
    if not path.exists():
        print(f"MISSING  {fname}: {path}")
        bad += 1
        continue
    lines = path.read_text().splitlines()
    if hi > len(lines):
        print(f"STALE    {fname}:{lo}-{hi} — file has {len(lines)} lines")
        bad += 1
        continue
    window = "\n".join(lines[lo - 1:hi])
    if needle not in window:
        print(f"STALE    {fname}:{lo}-{hi} — no {needle!r} in the cited range")
        actual = [i + 1 for i, l in enumerate(lines) if needle in l]
        print(f"         {needle!r} is at line(s) {actual[:6]}")
        bad += 1
    else:
        print(f"ok       {fname}:{lo}-{hi} contains {needle!r}")

print()
print("ALL CITATIONS FRESH" if not bad else f"{bad} STALE CITATION(S) — fix before trusting the gate")
sys.exit(1 if bad else 0)
