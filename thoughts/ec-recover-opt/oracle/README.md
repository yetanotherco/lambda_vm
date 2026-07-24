# EC RECOVER / ECSM oracle (secp256k1)

Independent oracle for the ECSM accelerator (`xr = x(k·P)`, `P = (xG, even-y)`)
plus a differential double-check of the repo's `crypto/ecsm` reference.
Built 2026-07-24. All anchors PASS.

## Verdicts

| Check | Verdict | Coverage |
|---|---|---|
| Curve constants vs SEC2 + repo `P_BYTES/N_BYTES/R_BYTES/B` | PASS | 8/8 (`check_constants.py`) |
| Anchor A — Wycheproof ECDH secp256k1 (authoritative) | PASS | 669 valid vectors matched, 19 invalid-x rejected, 0 fail (`anchor_a.log`) |
| Anchor B — PyPI `ecdsa` differential (independent lineage) | PASS | 500 random (k,P) + 60 k·G + 36 edge, 0 fail (`anchor_b.log`) |
| Anchor C — repo `crypto/ecsm` differential | PASS | 212 muls, 14 error paths (kind+order), 65 y-recoveries, 48 replays / **11,289 steps** verified, 0 fail (`anchor_c.log`) |
| vectors.json cross-check vs repo | PASS | 53/53 |
| Bonus 1 — ecrecover 3-way (ec_ref vs signer key vs `ecdsa` lib recovery) | PASS | 40 sigs (`bonus.log`) |
| Bonus 2 — guest `solve_y` λ-linear identity | PASS | 200 random cases (`bonus.log`) |

coincurve (libsecp256k1, would be a 4th lineage) has no wheel for python3.14
and fails to build from source — skipped; lineage count is still 3
(Wycheproof vectors, pure-python `ecdsa`, repo `k256`), plus SEC2 constants.

## Files

- `ec_ref.py` — independent reference (from SEC2/textbook formulas only):
  affine group law, MSB-first double-and-add, even-y recovery via
  `a^((p+1)/4)`, ABI mirror `x_only_mul`, documented-schedule generator
  `expected_schedule`, per-step replay `replay_schedule`, `ecrecover`.
- `check_constants.py` — SEC2 recomputation + repo constant diff.
- `anchor_a_wycheproof.py` + `wycheproof_ecdh_secp256k1.json` (752 tests,
  C2SP/wycheproof `main` `testvectors_v1`).
- `anchor_b_pypi.py` — venv differential (`./venv/bin/python`).
- `anchor_c_repo.py` + `repo-harness/` — cargo crate (path-dep on
  `crypto/ecsm`) speaking a line protocol; build with
  `cargo build --release` inside `repo-harness/`.
- `gen_vectors.py` → `vectors.json` — 53 canonical vectors (42 valid, 11
  error-path) with provenance + rationale tags, for the z3 gate.
- `bonus_ecrecover.py` — end-to-end ecrecover + solve_y identity.

Re-run everything:

```sh
python3 check_constants.py
python3 anchor_a_wycheproof.py
./venv/bin/python anchor_b_pypi.py
(cd repo-harness && cargo build --release)
python3 anchor_c_repo.py
python3 gen_vectors.py
./venv/bin/python bonus_ecrecover.py
```

## ECSM syscall ABI + error semantics (for the z3 gate)

Verified by reading the code (✓ VERIFIED, citations):

- **Syscall number**: `ECSM_SYSCALL_NUMBER = u64::MAX - 10`
  (`executor/src/vm/instruction/execution.rs:32`).
- **Registers**: `x10/a0` = addr to WRITE xR, `x11/a1` = addr of xG,
  `x12/a2` = addr of k (`execution.rs:429-431`, `syscalls/src/syscalls.rs:172-182`).
- **Values**: all three are 32-byte **little-endian**; loaded/stored as four
  unaligned doublewords at `addr + 8i` (`execution.rs:86-94`).
- **Address guard**: each of the three addrs must satisfy
  `(addr mod 2^32) + 31 < 2^32` (`LOW_LIMB = 1<<32`, `execution.rs:36,97-99`)
  else `ExecutionError::EcsmAddressOverflow` (`execution.rs:432-437`).
- **Overlap guard**: `|addr_xG − addr_k| ≥ 32` required, else
  `ExecutionError::EcsmOperandOverlap`. This is a *trace-provability* guard
  (MEMW access-chain), not a correctness one; xR may alias either input
  (`execution.rs:438-447`).
- **Input contract & check order** (`crypto/ecsm/src/lib.rs:103-120`,
  `prepare`): `k == 0` → `ScalarIsZero`; `k >= N` → `ScalarOutOfRange`;
  `xG >= p` → `CoordinateOutOfRange`; `x³+7` non-residue → `NotOnCurve`.
  Anchor C verified the *order* too (combined-invalid inputs).
- **On error the VM TRAPS**: `ecsm::scalar_mul_x(&k, &xg)?` at
  `execution.rs:450` propagates `EcsmError` into
  `ExecutionError::Ecsm(#[from])` (`execution.rs:638`) — execution fails
  before any write to xR. No garbage result exists; invalid inputs are
  *unexecutable*, hence never reach the trace.
- **Canonical y**: the accelerator lifts xG to the **even** y
  (`crypto/ecsm/src/curve.rs:26-34`, SEC1 `0x02` prefix). Sound because
  `x(k·P) = x(k·(−P))`; parity/sign is the guest's responsibility
  (`crypto/ethrex-crypto/src/lib.rs` lifts R from r + recid parity itself).
- **k=1 echo**: result is xG itself; `replay_double_and_add` returns an
  empty step list (`curve.rs:165-168`). The prover-side `xR < p` range check
  is what makes a non-canonical xG unprovable (`lib.rs:70-73` comment).

## Notes that matter for the z3 gate

1. **Error inputs never produce rows** — the executor traps, so the chip only
   ever sees `1 ≤ k < N`, `xG < p`, residue x. The gate should assume that
   domain but must confirm the *chip* enforces its own range checks
   (`XG_SUB_P`, `K_SUB_N`, `XR_SUB_P` columns in `prover/src/tables/ecsm.rs`)
   rather than inheriting soundness from the executor.
2. **Schedule semantics** (verified identical between repo and independent
   statement on 48 replays): rows are MSB-first; for each bit below the MSB:
   a double row, plus an add row iff the bit is set; `round` = bit index
   (add rows share their double's round); `next_op` = op of the following
   row, 0 on the last row. k=1 → zero rows.
3. **λ convention**: add rows use λ = (yG−yA)/(xG−xA) (chord from the
   *accumulator* to the *base*), double rows λ = 3xA²/(2yA). Repo per-step
   λ, a, r matched the independent replay on all 11,289 steps.
4. **Endianness trap**: ABI is little-endian bytes; Wycheproof/SEC1/ecdsa are
   big-endian. `vectors.json` carries both (`x`/`k` BE ints, `x_le`/`k_le`
   ABI bytes).
5. **x-only scope**: 9 Wycheproof "invalid" cases have off-curve (x,y) whose
   x still lifts to a valid point; the x-only precompile correctly accepts
   such x. y-validity is out of the accelerator's contract.
6. `ecrecover` composition (guest, `crypto/ethrex-crypto/src/lib.rs`):
   4 x-only queries (k1·P1, (k1+1)·P1, k2·P2, (k2+1)·P2), y recovered via the
   λ-linear identity (validated here, 200/200), wrong-sign excluded by the
   scalar-edge guards; falls back to software `lincomb` when any guard trips.
