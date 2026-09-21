"""Run the whole SHA-256 gate and print the board.

    python3 run_gate.py
"""
import subprocess
import sys
import time

STEPS = [
    ("citations into the Rust are fresh",     "check_citations.py"),
    ("reference vs hashlib + repo constants", "test_ref.py"),
    ("circuit model vs reference, forward",   "test_dataflow.py"),
    ("lemma 1: sigma/Sigma are rotate-xor",   "lemma_sigma.py"),
    ("lemma 2: Ch and Maj from BYTE_ALU",     "lemma_chmaj.py"),
    ("lemma 3: the additions force out_a/e",  "lemma_add.py"),
    ("lemma 4: the round grouping is spec's", "lemma_compose.py"),
    ("lemma 5: the schedule recurrence",      "lemma_schedule.py"),
]

fail = 0
print("=" * 66)
for label, script in STEPS:
    t0 = time.time()
    p = subprocess.run([sys.executable, script], capture_output=True, text=True)
    dt = time.time() - t0
    bad = p.returncode != 0 or "MISSED" in p.stdout or " sat" in p.stdout.split("the lemma")[-1]
    status = "FAIL" if bad else "ok"
    fail += bad
    print(f"{status:>4}  {label:<42} {dt:5.1f}s")
    if bad:
        print(p.stdout[-1500:])
        print(p.stderr[-800:])
print("=" * 66)
print("GATE: VERIFIED" if not fail else f"GATE: {fail} STEP(S) FAILED")
print("""
Covered, given the BYTE_ALU / IsHalfword / AreBytes / ShaK / ShaM contracts:
  - SHA256ROUND's per-round transition      (prover/src/tables/sha256_round.rs)
  - SHA256MSGSCHED's schedule recurrence    (prover/src/tables/sha256_schedule.rs)
NOT covered: the SHA256 core chip's memory and feed-forward wiring, the helper
chips themselves, and whether the range bounds suffice mod p — the last is
outside QF-BV entirely and is the same scope gap #950 documents for keccak.""")
sys.exit(1 if fail else 0)
