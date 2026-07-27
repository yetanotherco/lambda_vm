# EC RECOVER / ECSM oracle (secp256k1)

Independent oracle for the ECSM accelerator (`xr = x(k·P)`, `P = (xG, even-y)`)
plus a differential double-check of the repo's `crypto/ecsm` reference.
Built 2026-07-24; extended the same day for the **lincomb2** precompile
(`Q = u1·P1 + u2·P2`, phase D0). All anchors PASS.

## Verdicts — ECSM (single-scalar, x-only)

| Check | Verdict | Coverage |
|---|---|---|
| Curve constants vs SEC2 + repo `P_BYTES/N_BYTES/R_BYTES/B` | PASS | 8/8 (`check_constants.py`) |
| Anchor A — Wycheproof ECDH secp256k1 (authoritative) | PASS | 669 valid vectors matched, 19 invalid-x rejected, 0 fail (`anchor_a.log`) |
| Anchor B — PyPI `ecdsa` differential (independent lineage) | PASS | 500 random (k,P) + 60 k·G + 36 edge, 0 fail (`anchor_b.log`) |
| Anchor C — repo `crypto/ecsm` differential | PASS | 212 muls, 14 error paths (kind+order), 65 y-recoveries, 48 replays / **11,289 steps** verified, 0 fail (`anchor_c.log`) |
| vectors.json cross-check vs repo | PASS | 53/53 |
| Bonus 1 — ecrecover 3-way (ec_ref vs signer key vs `ecdsa` lib recovery) | PASS | 40 sigs (`bonus.log`) |
| Bonus 2 — parity-authority identity | PASS | 200 random cases (`bonus.log`). **Re-aimed 2026-07-24**: phase G deleted `solve_y`, so the old λ-linear check had become a green test of code that no longer exists. It now pins the recid→parity convention and the fact that a `< p` test cannot separate a point from its negation — the numeric basis of the gate's N7 redundancy result. |

## Verdicts — lincomb2 (phase D0)

Logs live in `../lincomb2/` alongside the design docs.

| Check | Verdict | Coverage |
|---|---|---|
| Anchor L-A — 3-way lincomb2 differential + blinded-chain row re-derivation | PASS | 611 cases × 3 lineages, 0 fail (`../lincomb2/anchor_lincomb2_differential.log`) |
| Anchor L-B — small joint scalars `u1,u2 ∈ [1,16]`, fully-enumerated ground truth | PASS | 1,536 cases over 6 point pairs, 0 fail (same log; vectors in `lincomb2_small_vectors.json`) |
| Anchor A-ECDSA — Wycheproof ECDSA-verify secp256k1 (the `u1·G + u2·PK` shape) | PASS | 489 verdicts matched (335 valid / 154 invalid), 360 through the blinded chain, 0 fail (`../lincomb2/anchor_wycheproof_ecdsa.log`) |
| Anchor D — repo `ecsm::lincomb2_witness` vs the Python oracle | PASS | 161 witnesses, **69,431 rows** field-by-field, 8 error paths, 0 fail (`../lincomb2/anchor_repo_lincomb2.log`) |
| Bonus 3 — the 40-signature ecrecover differential re-run through lincomb2 | PASS | 40 sigs, all recovered through the blinded joint chain (`../lincomb2/anchor_ecrecover_lincomb2.log`) |

Lineage count for lincomb2 is 4: the affine `ec_ref` path, the Jacobian/LSB-first
`jacobian_ref` path, the `ecdsa` PyPI package, and Wycheproof's own verdicts —
plus the repo's Rust witness as the object under test.

### One negative result

`nums_blinding_probe.py` is not an anchor but a counterexample generator, and it
FINDS what it looks for: the NUMS blind of DESIGN.md §4 does **not** close the
incomplete-addition edge when `P2` is prover-chosen, because the prover can set
`P2 = μ·T₀` for a `μ` it knows and cancel `T₀`'s coefficient out of the collision
equation. 5/5 constructions land a degenerate add with `λ` unconstrained, each
packaged as a full `ecrecover` input the guest's own decomposition reproduces.
Write-up: `../lincomb2/FINDING-nums-blinding.log`; raw run:
`../lincomb2/nums_blinding_probe.log`. Phase A is unaffected — the Rust witness
and the Python reference both refuse these inputs.

coincurve (libsecp256k1, would be a 4th lineage) has no wheel for python3.14
and fails to build from source — skipped; lineage count is still 3
(Wycheproof vectors, pure-python `ecdsa`, repo `k256`), plus SEC2 constants.

## Files

- `ec_ref.py` — independent reference (from SEC2/textbook formulas only):
  affine group law, MSB-first double-and-add, even-y recovery via
  `a^((p+1)/4)`, ABI mirror `x_only_mul`, documented-schedule generator
  `expected_schedule`, per-step replay `replay_schedule`, `ecrecover`.
- `check_constants.py` — SEC2 recomputation + repo constant diff.
- `anchor_a_wycheproof.py` — Wycheproof driver. No argument = the ECDH anchor
  (`wycheproof_ecdh_secp256k1.json`, 752 tests); `ecdsa` = the ECDSA-verify
  anchor for lincomb2 (`ecdsa_secp256k1_sha256_p1363_test.json` 252 tests +
  `ecdsa_secp256k1_sha256_test.json` 476 tests). All three vector files are
  C2SP/wycheproof `main`, `testvectors_v1`, vendored here.
- `anchor_b_pypi.py` — venv differential (`./venv/bin/python`).
- `anchor_c_repo.py` + `repo-harness/` — cargo crate (path-dep on
  `crypto/ecsm`) speaking a line protocol; build with
  `cargo build --release` inside `repo-harness/`.
- `gen_vectors.py` → `vectors.json` — 53 canonical vectors (42 valid, 11
  error-path) with provenance + rationale tags, for the z3 gate.
- `bonus_ecrecover.py` — end-to-end ecrecover + the parity-authority identity + (part 3) the
  same 40 signatures re-run through the lincomb2 path.

lincomb2 (phase D0):

- `lincomb2_ref.py` — the lincomb2 reference on top of `ec_ref.py`: T₀
  derivation, `lincomb2`, the blinded-trace self-check, and `lincomb2_rows`
  (the full joint-chain row list, mirroring `ecsm::lincomb2_witness`).
- `jacobian_ref.py` — a SECOND independent implementation, deliberately a
  different code path: Jacobian coordinates, inversion-free formulas, LSB-first
  scalar multiplication, infinity representable. Also carries a textbook
  ECDSA verify.
- `lincomb2_anchors.py` → `lincomb2_small_vectors.json` — anchors L-A (≥500
  random differentials) and L-B (small-joint-scalar unrollings, the phase-E L7
  anchor).
- `anchor_d_lincomb2_repo.py` — repo `ecsm::lincomb2_witness` differential
  through the harness's `lincomb2` command.
- `nums_blinding_probe.py` — the counterexample generator for the DESIGN §4
  blinding argument (see above).

`lincomb2_ref.lincomb2_rows` tracks the CURRENT witness row format, which
includes `nb = d1 | d2` on double rows and the digit bits on both the double
and the add of a round. If `crypto/ecsm` changes that format again, anchor D is
what will notice.

Re-run everything:

```sh
python3 check_constants.py
python3 anchor_a_wycheproof.py                  # ECDH  -> anchor_a.log
./venv/bin/python anchor_b_pypi.py
(cd repo-harness && cargo build --release)
python3 anchor_c_repo.py
python3 gen_vectors.py
./venv/bin/python bonus_ecrecover.py            # incl. part 3 (lincomb2)
# lincomb2 (phase D0) — all need the venv (`ecdsa` package):
./venv/bin/python lincomb2_anchors.py
./venv/bin/python anchor_a_wycheproof.py ecdsa
./venv/bin/python anchor_d_lincomb2_repo.py
```

The venv is just `python3 -m venv venv && ./venv/bin/pip install ecdsa`
(add `z3-solver sympy` for the gate scripts in `../gate/`).

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
