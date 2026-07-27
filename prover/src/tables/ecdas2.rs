//! ECDAS2 chip — one step of the lincomb2 joint (Shamir/Straus) double-add chain.
//!
//! The joint sibling of [`ecdas`](super::ecdas). Its λ/xR/yR convolution core is
//! **byte-for-byte** the same machinery (same quotients `q0/q1/q2`, same 64-entry
//! carry arrays with the same offsets, same shifted-quotient `r = 3p` term), and
//! the witness reuses `ecsm::witness::build_step` unchanged to produce it. What
//! differs:
//!
//! * a **fourth relation** proving the addition is non-degenerate (see below);
//! * the addend `(XB, YB)` varies per row and arrives on the [`Addend`](BusId::Addend)
//!   bus instead of being a loop-invariant generator carried in the chain tuple;
//! * the chain is split into three separately-keyed **phases** so the two rows
//!   that break the `a = prev.r` telescoping cannot be confused with main-chain
//!   rows (both are emitted at `round = 0`, which the main loop also produces);
//! * two scalar-digit streams are counted instead of one.
//!
//! ## The non-degeneracy relation `D_INV·(xB − xA) ≡ 1 (mod p)`
//!
//! **This is the check that makes the chip sound, and it is unconditional — it
//! rests on no computational assumption.**
//!
//! When `xB = xA` the λ relation `Σλ(xB − xA) + yA − yB` degenerates: with
//! `yB = yA` it reads `0 = 0` for *every* λ, so `xR = λ² − xA − xB` and
//! `yR = λ(xA − xR) − yA` produce a point of the prover's choosing that the rest
//! of the chain then accepts. Everything downstream — the Addend balance, the
//! digit counting, the phase pinning — is satisfied. The row proves nothing.
//!
//! DESIGN §4 argued the NUMS blind `T₀` closes this by making a collision imply
//! a known linear relation on `dlog_G(T₀)`. **That argument does not hold, and
//! the edge is cheaply reachable.** The prover chooses `P2` — for ecrecover it is
//! `lift_x(r)` and `r` is free signature bytes — so setting `P2 = μ·T₀` cancels
//! `T₀`'s coefficient and leaves a collision condition solvable with *no*
//! knowledge of `dlog(T₀)`: with `P1 = G` and `u1 = 1` it reduces to
//! `μ·(c2 − 1) ≡ −2^j (mod N)`, one modular inversion. Verified 5/5 over
//! `len ∈ {8, 12, 16, 32, 256}`, each packaged as a well-formed `(z, v, r, s)`
//! (`thoughts/ec-recover-opt/oracle/nums_blinding_probe.py`,
//! `thoughts/ec-recover-opt/lincomb2/FINDING-nums-blinding.log`). The blind made
//! the edge *easier* to aim at than the unblinded chain's ~2^−j search.
//!
//! Phase A is unaffected: `lincomb2_witness` rejects all of these with
//! `ResultInfinity` (status ≠ 0 ⇒ guest fallback). The forgery lives entirely on
//! the malicious-prover side, where the row never passes through the witness
//! generator, so **only this constraint catches it**.
//!
//! The blind survives for a different, non-soundness reason: it lets any
//! `len ∈ [max_msb + 1, 256]` yield the correct `Q`, which is what drops the
//! exact-MSB sub-lemma. It is a convenience, not a defence — do not describe it
//! as one.
//!
//! The relation is gated to rows that actually consume an addend — the same
//! `S1 + S2 + S3 + S_CORR` sum as the Addend receive, so the precompute and
//! correction rows (both chord adds) are covered and doublings are not. Gating is
//! a cost choice rather than a correctness one: `x = 0` is not on secp256k1
//! (`7` is a non-residue mod `p`), so `xB − xA = −xA` would be invertible on
//! doublings too — but there is no addend there, so the cells would be wasted.
//!
//! ## Phases
//!
//! `PHASE = PH1 + 2·PH2` rides the chain tuple. ECSM2 pins every segment at both
//! ends, at multiplicity `OK`:
//!
//! | phase | rows | seeded with | drained to |
//! |---|---|---|---|
//! | 0 precompute | exactly 1 | `a = P1 = G`, addend `P2` | `X_P12/Y_P12` |
//! | 1 main chain | `len` doubles + their adds | `a = T₀`, `round = LEN_M1` | `ACC_X/ACC_Y` |
//! | 2 correction | exactly 1 | `a = ACC`, addend `−2^len·T₀` | `X_Q/Y_Q` |
//!
//! The phase-1 → phase-2 hand-off deliberately goes **through ECSM2** (drain then
//! re-send on the same columns) rather than along the chain: the outgoing tuple
//! pins the successor's `op` to this row's `NB`, and the last main row has
//! `NB = 0` while the correction row is an add (`op = 1`), so a direct hand-off
//! is not expressible.
//!
//! ## Round bookkeeping
//!
//! A doubling and its optional add share a `round`, so the successor round is
//! `round − 1 + NB` and the successor `op` is `NB` — exactly the `NEXT_OP`
//! mechanism of the single-scalar chain (`ecdas.rs`), under the joint name
//! `nb` ("an add follows me at this round"). Its two defining constraints are
//! op-gated: an add row carries its round's real digits (it needs them to pick
//! the addend) but always has `NB = 0`.
//!
//! ## Why double rows may carry a zero addend
//!
//! On `OP = 0` **no** convolution constraint reads `XB`/`YB`: in the λ relation
//! every occurrence sits inside the `op·(…)` product; in the xR relation the
//! `−xg(i)` cancels exactly against the `+xg(i)` released by `−(1−op)(xa−xg)`;
//! the yR relation never mentions them; and the non-degeneracy relation's whole
//! `d·(xB − xA)` term sits inside its `ΣS` gate, which `OP = ΣS` pins to zero.
//! So the addend columns are zero on doublings and the `Addend` receive stays
//! silent — but only because `OP = S1 + S2 + S3 + S_CORR` forces every selector
//! to zero there. Without that constraint the cancellation is still real and the
//! *gating* is forgeable: a prover would set `S2 = 1` on a doubling and mint a
//! spurious receive.
//!
//! ## Padding
//!
//! Padding rows are all-zero (including `OP = 0`, unlike `ecdas.rs` which pads
//! with `OP = 1`). All four relations then close at zero carries: the `μ`-gated
//! `R·P` term vanishes and every byte limb is zero. Keeping `OP = 0` lets
//! `OP = ΣS` stay an ungated degree-1 constraint.
//!
//! A padding row must also be inert on every *bus*, and "the columns are zero as
//! generated" is **not** an argument — a malicious prover fills padding rows
//! freely. The question to ask of each interaction is not "is it gated?" but
//! "which column supplies its multiplicity, and what forces that column to
//! zero?":
//!
//! | interaction | multiplicity | inert because |
//! |---|---|---|
//! | `Ecdas` receive + send | `MU` | by construction |
//! | 131 `AreBytes`, 252 `IsHalfword` | `MU` | by construction |
//! | `Addend` receive | `S1 + S2 + S3 + S_CORR` — **raw columns** | idx 24..=27 |
//! | 2 `JointBit` sends | `D1` / `D2` — **raw columns** | idx 22, 23 |
//!
//! The bottom two rows are where the live hole was: `(1 − MU)·{D1, D2, S1, S2,
//! S3, S_CORR} = 0` is what closes it, and idx 22..=27 writes out both forgeries
//! in full. Note that `NB` needs no companion — with every selector zero,
//! `OP = ΣS` forces `OP = 0` and idx 13 then reads `NB = D1 ∨ D2 = 0`.

use stark::lookup::{BusInteraction, BusValue, LinearTerm, Multiplicity, Packing};
use stark::trace::TraceTable;

use super::types::{BusId, FE, GoldilocksExtension, GoldilocksField, VmTable};
use ecsm::P_BYTES;
use ecsm::witness::{JointSel, JointStep};

pub(crate) use ecsm::R_BYTES;

// Bias signed convolution carries into IsHalfword [0, 2^16). Identical to
// `ecdas.rs`: the relations, and therefore the carry ranges, are the same.
pub(crate) use super::ecdas::{CARRY_OFFSET_LAMBDA, CARRY_OFFSET_XR, CARRY_OFFSET_YR};

/// Carry bias for the non-degeneracy relation. It needs no window of its own:
/// its honest carries measure `[-581, 6041]` (`WIDTH-AUDIT.md` §4, 6,346 real
/// rows) and its worst-case soundness magnitude is the *smallest* of the four
/// relations, so `CARRY_OFFSET_XR`'s window `[-8161, 57374]` holds it with an
/// order of magnitude of slack. Named separately only so the use sites read.
pub(crate) const CARRY_OFFSET_DINV: i64 = CARRY_OFFSET_XR;

/// Worst-case joint-chain rows for one lincomb2 evaluation.
///
/// `1 precompute + 256 doublings + 256 adds + 1 correction`. Reached at
/// `(u1, u2) = (2^255, 2^255 − 1)`: both lie in `[1, N)` and their bit patterns
/// are **complementary**, so every one of the 256 rounds carries a nonzero joint
/// digit and therefore an add. Complementarity is what maximises, not popcount —
/// `(N−1, N−1)` shares every add and reaches only 449.
///
/// **This is the number for anything capacity-shaped** (padding bounds,
/// per-ecall row allowances, any rows-per-call assertion). The measured mean of
/// 449.1 governs the *cost model* and nothing else: a random corpus tops out
/// around 471, but a submitter can construct the worst case deliberately, so no
/// bound may be derived from a sample maximum.
///
/// Exercised end to end by `test_prove_elfs_ecsm_lincomb2_full_size`, whose
/// guest uses exactly those scalars and asserts the emitted row count.
pub const MAX_ROWS_PER_EVALUATION: usize = 1 + 256 + 256 + 1;

/// The chain identifier carried in tuple position 0 of every
/// [`Ecdas`](BusId::Ecdas) interaction of the joint chain.
///
/// The single-scalar chain pins that position to the constant `0`
/// (`ecsm::ecdas_tuple`, "id = 0 (secp256k1)"), so the two chains sharing bus 28
/// differ in the α¹ coefficient of their fingerprints — a non-zero difference at
/// a fixed power, which Schwartz–Zippel closes. An old-chain tuple therefore
/// cannot be received by ECDAS2 or vice versa.
///
/// The separation is *not* about zeros shifting tuple positions: they do not.
/// `alpha_offset` advances by `num_bus_elements()` unconditionally
/// (`lookup.rs:1651-1663`), and the `if result != zero` guard at `lookup.rs:679`
/// only skips the multiply for a zero element. Nothing re-aligns.
pub const JOINT_CHAIN_ID: u64 = 1;

// =========================================================================
// Column indices (658 columns; keep in sync with NUM_COLUMNS below)
// =========================================================================

pub mod cols {
    pub const TIMESTAMP_0: usize = 0;
    pub const TIMESTAMP_1: usize = 1;
    /// Addend x, received on the `Addend` bus (zero on doublings).
    pub const XB: usize = 2; // U256BL (32)
    /// Addend y.
    pub const YB: usize = 34; // U256BL (32)
    /// Accumulator in.
    pub const XA: usize = 66; // U256BL (32)
    pub const YA: usize = 98; // U256BL (32)
    pub const ROUND: usize = 130; // Byte
    pub const OP: usize = 131; // Bit: 0 = double, 1 = add
    /// Result out.
    pub const XR: usize = 132; // U256BL (32)
    pub const YR: usize = 164; // U256BL (32)
    pub const LAMBDA: usize = 196; // U256BL (32)
    pub const Q0: usize = 228; // Byte[33]  — λ relation quotient
    pub const C0: usize = 261; // BaseField[64]
    pub const Q1: usize = 325; // Byte[33]  — xR relation quotient
    pub const C1: usize = 358; // BaseField[64]
    pub const Q2: usize = 422; // Byte[33]  — yR relation quotient
    pub const C2: usize = 455; // BaseField[64]
    /// "An add follows me at this round" — pins the successor `(round, op)`.
    pub const NB: usize = 519; // Bit
    /// `u1`'s digit at this row's round; set on both the doubling and its add.
    pub const D1: usize = 520; // Bit
    /// `u2`'s digit at this row's round.
    pub const D2: usize = 521; // Bit
    /// One-hot addend selector: P1 / P2 / P12.
    pub const S1: usize = 522; // Bit
    pub const S2: usize = 523; // Bit
    pub const S3: usize = 524; // Bit
    /// Selects the `−2^len·T₀` correction constant.
    pub const S_CORR: usize = 525; // Bit
    /// `PHASE = PH1 + 2·PH2`, with `PH1·PH2 = 0` — so `PHASE ∈ {0, 1, 2}` and
    /// `PH1` is a degree-1 "this is a main-chain row" gate.
    pub const PH1: usize = 526; // Bit
    pub const PH2: usize = 527; // Bit
    pub const MU: usize = 528; // Bit

    /// `(xB − xA)^{-1} mod p` — the non-degeneracy witness, live on every row
    /// that consumes an addend and zero elsewhere.
    pub const D_INV: usize = 529; // U256BL (32)
    /// Quotient of the non-degeneracy relation.
    pub const Q3: usize = 561; // Byte[33]
    pub const C3: usize = 594; // BaseField[64]

    pub const NUM_COLUMNS: usize = 658;

    #[inline]
    pub const fn c0(i: usize) -> usize {
        C0 + i
    }
    #[inline]
    pub const fn c1(i: usize) -> usize {
        C1 + i
    }
    #[inline]
    pub const fn c2(i: usize) -> usize {
        C2 + i
    }
    #[inline]
    pub const fn c3(i: usize) -> usize {
        C3 + i
    }
}

// =========================================================================
// Operation struct
// =========================================================================

/// One ECDAS2 row: a joint-chain step witness plus its ECALL timestamp.
#[derive(Debug, Clone)]
pub struct Ecdas2Operation {
    pub timestamp: u64,
    pub step: JointStep,
}

impl Ecdas2Operation {
    /// `(PH1, PH2)` for this row's phase.
    pub fn phase_bits(&self) -> (u8, u8) {
        match self.step.sel {
            JointSel::Precompute => (0, 0),
            JointSel::Correction => (0, 1),
            JointSel::Double | JointSel::AddP1 | JointSel::AddP2 | JointSel::AddP12 => (1, 0),
        }
    }

    /// `(S1, S2, S3, S_CORR)` for this row. The precompute row genuinely adds
    /// `P2`, so it reuses `S2` (and hence `sel = 2` on the Addend bus).
    pub fn selector_bits(&self) -> (u8, u8, u8, u8) {
        match self.step.sel {
            JointSel::Double => (0, 0, 0, 0),
            JointSel::AddP1 => (1, 0, 0, 0),
            JointSel::AddP2 | JointSel::Precompute => (0, 1, 0, 0),
            JointSel::AddP12 => (0, 0, 1, 0),
            JointSel::Correction => (0, 0, 0, 1),
        }
    }
}

// =========================================================================
// Trace generation
// =========================================================================

fn fe_from_i64(c: i64) -> FE {
    if c >= 0 {
        FE::from(c as u64)
    } else {
        FE::zero() - FE::from((-c) as u64)
    }
}

// =========================================================================
// The non-degeneracy witness
// =========================================================================
//
// `lincomb2_witness` does not carry these columns: the schedule it emits already
// implies `xB ≠ xA` on every add, because it refuses (`ResultInfinity`) any
// input whose chain would add a point to itself. So the inverse is a
// *derivation* from values it already publishes, not new information — but the
// chip needs it regardless, since a malicious prover's row never passes through
// that generator at all.
//
// The derivation lives in `ecsm::witness` beside `ext64`/`conv`/`limb_carries`,
// NOT here. A second copy of that limb arithmetic in the prover is the classic
// silent divergence: one side gets edited, the other does not, and nothing
// notices until a proof is wrong. Re-exported so callers keep saying
// `ecdas2::dinv_witness`.
pub use ecsm::witness::{DinvWitness, dinv_witness};

pub fn generate_ecdas2_trace(
    ops: &[Ecdas2Operation],
) -> TraceTable<GoldilocksField, GoldilocksExtension> {
    let n = ops.len();
    let num_rows = n.next_power_of_two().max(4);
    let mut trace = TraceTable::new_main(
        crate::tables::types::zeroed_fe_vec(num_rows * cols::NUM_COLUMNS),
        cols::NUM_COLUMNS,
        1,
    );
    let table = &mut trace.main_table;

    for (row_idx, op) in ops.iter().enumerate() {
        let j = &op.step;
        let s = &j.step;

        table.set_dword_wl(row_idx, cols::TIMESTAMP_0, op.timestamp);
        table.set_bytes(row_idx, cols::XB, &s.x_g);
        table.set_bytes(row_idx, cols::YB, &s.y_g);
        table.set_bytes(row_idx, cols::XA, &s.x_a);
        table.set_bytes(row_idx, cols::YA, &s.y_a);
        table.set_byte(row_idx, cols::ROUND, s.round);
        table.set_byte(row_idx, cols::OP, s.op);
        table.set_bytes(row_idx, cols::XR, &s.x_r);
        table.set_bytes(row_idx, cols::YR, &s.y_r);
        table.set_bytes(row_idx, cols::LAMBDA, &s.lambda);
        table.set_bytes(row_idx, cols::Q0, &s.q0);
        table.set_bytes(row_idx, cols::Q1, &s.q1);
        table.set_bytes(row_idx, cols::Q2, &s.q2);
        for i in 0..64 {
            debug_assert!((0..1 << 16).contains(&(s.c0[i] + CARRY_OFFSET_LAMBDA)));
            debug_assert!((0..1 << 16).contains(&(s.c1[i] + CARRY_OFFSET_XR)));
            debug_assert!((0..1 << 16).contains(&(s.c2[i] + CARRY_OFFSET_YR)));
            table.set_fe(row_idx, cols::c0(i), fe_from_i64(s.c0[i]));
            table.set_fe(row_idx, cols::c1(i), fe_from_i64(s.c1[i]));
            table.set_fe(row_idx, cols::c2(i), fe_from_i64(s.c2[i]));
        }

        let (ph1, ph2) = op.phase_bits();
        let (s1, s2, s3, s_corr) = op.selector_bits();
        table.set_byte(row_idx, cols::NB, j.nb);
        table.set_byte(row_idx, cols::D1, j.d1);
        table.set_byte(row_idx, cols::D2, j.d2);
        table.set_byte(row_idx, cols::S1, s1);
        table.set_byte(row_idx, cols::S2, s2);
        table.set_byte(row_idx, cols::S3, s3);
        table.set_byte(row_idx, cols::S_CORR, s_corr);
        table.set_byte(row_idx, cols::PH1, ph1);
        table.set_byte(row_idx, cols::PH2, ph2);
        table.set_fe(row_idx, cols::MU, FE::one());

        let d = dinv_witness(j);
        table.set_bytes(row_idx, cols::D_INV, &d.d_inv);
        table.set_bytes(row_idx, cols::Q3, &d.q3);
        for i in 0..64 {
            debug_assert!((0..1 << 16).contains(&(d.c3[i] + CARRY_OFFSET_DINV)));
            table.set_fe(row_idx, cols::c3(i), fe_from_i64(d.c3[i]));
        }
    }

    // Padding rows stay entirely zero (see the module header): with OP = 0 and
    // every limb zero, all four convolution relations close at zero carries and
    // `OP = ΣS` holds trivially.

    trace
}

// =========================================================================
// Bus value helpers
// =========================================================================

fn packed(col: usize) -> BusValue {
    BusValue::Packed {
        start_column: col,
        packing: Packing::Direct,
    }
}

/// The 32 bytes of a U256BL coordinate as bus elements. Same shape ECSM/ECDAS
/// use, so publisher and consumer pack identically.
pub fn coord(col: usize) -> Vec<BusValue> {
    super::ecsm::point_coord_busvalues(col)
}

/// The joint chain tuple
/// `[JOINT_CHAIN_ID, ts_lo, ts_hi, phase, accX(32), accY(32), round, op]`.
///
/// Deliberately narrower than the single-scalar `ecdas_tuple`: the addend is no
/// longer part of the accumulator state (it varies per row and arrives on the
/// `Addend` bus), so `genX`/`genY` are gone and `phase` takes their place.
pub fn joint_tuple(
    acc_x: Vec<BusValue>,
    acc_y: Vec<BusValue>,
    phase: BusValue,
    round: BusValue,
    op: BusValue,
    ts_lo: BusValue,
    ts_hi: BusValue,
) -> Vec<BusValue> {
    debug_assert_eq!(acc_x.len(), 32);
    debug_assert_eq!(acc_y.len(), 32);
    let mut v = Vec::with_capacity(1 + 2 + 1 + 2 * 32 + 2);
    v.push(BusValue::constant(JOINT_CHAIN_ID));
    v.push(ts_lo);
    v.push(ts_hi);
    v.push(phase);
    v.extend(acc_x);
    v.extend(acc_y);
    v.push(round);
    v.push(op);
    v
}

/// The `Addend` tuple `[ts_lo, ts_hi, sel, x(32), y(32)]`.
pub fn addend_tuple(
    ts_lo: BusValue,
    ts_hi: BusValue,
    sel: BusValue,
    x: Vec<BusValue>,
    y: Vec<BusValue>,
) -> Vec<BusValue> {
    debug_assert_eq!(x.len(), 32);
    debug_assert_eq!(y.len(), 32);
    let mut v = Vec::with_capacity(3 + 64);
    v.push(ts_lo);
    v.push(ts_hi);
    v.push(sel);
    v.extend(x);
    v.extend(y);
    v
}

// =========================================================================
// Bus interactions
// =========================================================================

/// `PHASE = PH1 + 2·PH2`.
fn phase_expr() -> BusValue {
    BusValue::linear(vec![
        LinearTerm::Column {
            coefficient: 1,
            column: cols::PH1,
        },
        LinearTerm::Column {
            coefficient: 2,
            column: cols::PH2,
        },
    ])
}

pub fn bus_interactions() -> Vec<BusInteraction> {
    let mu = || Multiplicity::Column(cols::MU);
    let ts_lo = || packed(cols::TIMESTAMP_0);
    let ts_hi = || packed(cols::TIMESTAMP_1);
    let mut out = Vec::new();

    // Receive the incoming accumulator [chain_id, ts, phase, xA, yA, round, op].
    out.push(BusInteraction::receiver(
        BusId::Ecdas,
        mu(),
        joint_tuple(
            coord(cols::XA),
            coord(cols::YA),
            phase_expr(),
            packed(cols::ROUND),
            packed(cols::OP),
            ts_lo(),
            ts_hi(),
        ),
    ));

    // Receive the addend [ts, sel, x, y] once per add row (silent on doublings,
    // where `OP = ΣS = 0`). `sel = S1 + 2·S2 + 3·S3 + 4·S_CORR ∈ {1, 2, 3, 4}`.
    //
    // `Multiplicity::Linear`, not `Sum3`: the correction row's `S_CORR` is a
    // fourth term. Still one interaction.
    out.push(BusInteraction::receiver(
        BusId::Addend,
        Multiplicity::Linear(vec![
            LinearTerm::Column {
                coefficient: 1,
                column: cols::S1,
            },
            LinearTerm::Column {
                coefficient: 1,
                column: cols::S2,
            },
            LinearTerm::Column {
                coefficient: 1,
                column: cols::S3,
            },
            LinearTerm::Column {
                coefficient: 1,
                column: cols::S_CORR,
            },
        ]),
        addend_tuple(
            ts_lo(),
            ts_hi(),
            BusValue::linear(vec![
                LinearTerm::Column {
                    coefficient: 1,
                    column: cols::S1,
                },
                LinearTerm::Column {
                    coefficient: 2,
                    column: cols::S2,
                },
                LinearTerm::Column {
                    coefficient: 3,
                    column: cols::S3,
                },
                LinearTerm::Column {
                    coefficient: 4,
                    column: cols::S_CORR,
                },
            ]),
            coord(cols::XB),
            coord(cols::YB),
        ),
    ));

    // ARE_BYTES range checks, paired: `ARE_BYTES[X, Y]` checks BOTH elements, so
    // adjacent bytes share one send — the identical layout `ecdas.rs` uses. The
    // 32-byte prefixes of LAMBDA, XR, YR, Q0, Q1, Q2, D_INV, Q3 pair internally
    // as (2i, 2i+1); of the five odd bytes, four pair as (ROUND, Q0[32]) and
    // (Q1[32], Q2[32]) and Q3[32] rides alone as [b, 0] — the shape ECSM2 uses
    // for `mem_q1[32]`. `collect_bitwise_from_ecdas2` mirrors this exactly.
    //
    // D_INV and Q3 are checked here for the same reason every other convolution
    // operand is: the integer-lifting argument (`WIDTH-AUDIT.md` §2) holds only
    // because every limb entering `S_i` is a byte, and these two are read by the
    // non-degeneracy relation.
    //
    // XB/YB are deliberately absent: they inherit byte-ness from the publisher's
    // already-checked columns through Addend tuple equality.
    let pair = |col_x: usize, col_y: usize, out: &mut Vec<BusInteraction>| {
        out.push(BusInteraction::sender(
            BusId::AreBytes,
            Multiplicity::Column(cols::MU),
            vec![packed(col_x), packed(col_y)],
        ));
    };
    for base in [
        cols::LAMBDA,
        cols::XR,
        cols::YR,
        cols::Q0,
        cols::Q1,
        cols::Q2,
        cols::D_INV,
        cols::Q3,
    ] {
        for i in 0..16 {
            pair(base + 2 * i, base + 2 * i + 1, &mut out);
        }
    }
    pair(cols::ROUND, cols::Q0 + 32, &mut out);
    pair(cols::Q1 + 32, cols::Q2 + 32, &mut out);
    out.push(BusInteraction::sender(
        BusId::AreBytes,
        Multiplicity::Column(cols::MU),
        vec![packed(cols::Q3 + 32), BusValue::constant(0)],
    ));

    // IS_HALF range checks on the carries (offsets keep them in [0, 2^16)).
    let half = |col: usize, off: i64| {
        BusValue::linear(vec![
            LinearTerm::Column {
                coefficient: 1,
                column: col,
            },
            LinearTerm::Constant(off),
        ])
    };
    for (base, off) in [
        (cols::C0, CARRY_OFFSET_LAMBDA),
        (cols::C1, CARRY_OFFSET_XR),
        (cols::C2, CARRY_OFFSET_YR),
        (cols::C3, CARRY_OFFSET_DINV),
    ] {
        for i in 0..63 {
            out.push(BusInteraction::sender(
                BusId::IsHalfword,
                mu(),
                vec![half(base + i, off)],
            ));
        }
    }

    // Per-stream digit sends. ECSM2 receives at multiplicity `2·bit`, because a
    // set digit is carried by BOTH the round's doubling and its add.
    for (stream, col) in [(1u64, cols::D1), (2u64, cols::D2)] {
        out.push(BusInteraction::sender(
            BusId::JointBit,
            Multiplicity::Column(col),
            vec![
                ts_lo(),
                ts_hi(),
                packed(cols::ROUND),
                BusValue::constant(stream),
            ],
        ));
    }

    // Send the updated accumulator: [chain_id, ts, phase, xR, yR, round - 1 + NB, NB].
    // `phase` is unchanged — each segment is drained and re-seeded by ECSM2.
    out.push(BusInteraction::sender(
        BusId::Ecdas,
        mu(),
        joint_tuple(
            coord(cols::XR),
            coord(cols::YR),
            phase_expr(),
            BusValue::linear(vec![
                LinearTerm::Column {
                    coefficient: 1,
                    column: cols::ROUND,
                },
                LinearTerm::Column {
                    coefficient: 1,
                    column: cols::NB,
                },
                LinearTerm::Constant(-1),
            ]),
            packed(cols::NB),
            ts_lo(),
            ts_hi(),
        ),
    ));

    out
}

/// Which convolution relation an ECDAS2 carry constraint enforces.
#[derive(Clone, Copy)]
pub enum Relation {
    Lambda,
    Xr,
    Yr,
    /// `D_INV·(xB − xA) ≡ 1 (mod p)` — the addition is a genuine chord.
    Dinv,
}

// =========================================================================
// Single-body constraint set (ConstraintSet front-end)
// =========================================================================
//
// Constraint indices 0..=287 (288 total):
//   0..=10  : IS_BIT on MU, OP, NB, D1, D2, S1, S2, S3, S_CORR, PH1, PH2
//   11      : PH1 · PH2                            (PHASE ∈ {0, 1, 2})
//   12      : OP · NB                              (an add is never followed by an add)
//   13      : (1 − OP)·(NB − D1 − D2 + D1·D2)      (NB = D1 ∨ D2 on doublings)
//   14      : OP − S1 − S2 − S3 − S_CORR           (adds pick exactly one addend)
//   15, 16  : (1 − PH1)·D1, (1 − PH1)·D2           (digits only on the main chain)
//   17      : PH1 · S_CORR                         (T₀ constant is not a main addend)
//   18      : PH1 · (S1 + S3 − OP·D1)              (addend matches u1's digit)
//   19      : PH1 · (S2 + S3 − OP·D2)              (addend matches u2's digit)
//   20      : MU·(1 − PH1 − PH2)·(S2 − 1)          (the precompute row adds P2)
//   21      : PH2 · (S_CORR − 1)                   (the correction row adds −2^len·T₀)
//   22..=27 : (1 − MU)·{D1, D2, S1, S2, S3, S_CORR} (padding rows emit nothing)
//   28..=287: per relation (Lambda, Xr, Yr, Dinv): 64 ConvCarry + 1 ColIsZero(c_63)
//
// The four relation blocks are contiguous and in that order, so the `Dinv` block
// (223..=287) can be ablated as a unit — which is what the phase-E negative
// control does: drop it, feed the gate the `nums_blinding_probe` construction,
// and the forgery must reappear as SAT.

use stark::constraints::builder::{ConstraintBuilder, ConstraintSet};

/// ECDAS2 transition constraints as a single-source [`ConstraintSet`] (288
/// total). No column configuration needed (the layout is fixed via `cols`).
pub struct Ecdas2Constraints;

impl Ecdas2Constraints {
    /// Byte `m` of the field prime `P` (zero beyond 32 bytes).
    fn p_byte_expr<B: ConstraintBuilder<GoldilocksField, GoldilocksExtension>>(
        b: &B,
        m: usize,
    ) -> B::Expr {
        if m < 32 {
            b.const_base(P_BYTES[m] as u64)
        } else {
            b.zero()
        }
    }

    /// Byte `m` of `R = 3p` (zero beyond 33 bytes).
    fn r_byte_expr<B: ConstraintBuilder<GoldilocksField, GoldilocksExtension>>(
        b: &B,
        m: usize,
    ) -> B::Expr {
        if m < 33 {
            b.const_base(R_BYTES[m] as u64)
        } else {
            b.zero()
        }
    }

    /// `bytes[base + j]` for `j < len`, else zero.
    fn byte_at<B: ConstraintBuilder<GoldilocksField, GoldilocksExtension>>(
        b: &B,
        base: usize,
        len: usize,
        j: usize,
    ) -> B::Expr {
        if j < len {
            b.main(0, base + j)
        } else {
            b.zero()
        }
    }

    /// The `μ·R·P − q·P` convolution term, shared by all four relations. The
    /// `μ`-gate makes it vanish on padding rows (`μ = 0`, `q = 0`), keeping every
    /// relation at zero carries.
    fn rq<B: ConstraintBuilder<GoldilocksField, GoldilocksExtension>>(
        b: &B,
        i: usize,
        qbase: usize,
    ) -> B::Expr {
        let mu = b.main(0, cols::MU);
        let mut r_p = b.zero();
        let mut q_p = b.zero();
        for j in 0..=i {
            r_p = r_p + Self::r_byte_expr(b, j) * Self::p_byte_expr(b, i - j);
            q_p = q_p + Self::byte_at(b, qbase, 33, j) * Self::p_byte_expr(b, i - j);
        }
        mu * r_p - q_p
    }

    /// `S_i` for `relation` at limb `i`. Identical to `ecdas.rs`'s body with
    /// `XG`/`YG` renamed to the per-row addend `XB`/`YB`.
    fn s_i<B: ConstraintBuilder<GoldilocksField, GoldilocksExtension>>(
        b: &B,
        relation: Relation,
        i: usize,
    ) -> B::Expr {
        let lam = |j: usize| Self::byte_at(b, cols::LAMBDA, 32, j);
        let xg = |j: usize| Self::byte_at(b, cols::XB, 32, j);
        let xa = |j: usize| Self::byte_at(b, cols::XA, 32, j);
        let ya = |j: usize| Self::byte_at(b, cols::YA, 32, j);
        let yg = |j: usize| Self::byte_at(b, cols::YB, 32, j);
        let xr = |j: usize| Self::byte_at(b, cols::XR, 32, j);
        let yr = |j: usize| Self::byte_at(b, cols::YR, 32, j);
        let op = b.main(0, cols::OP);
        let one = b.one();

        match relation {
            Relation::Lambda => {
                // op·(Σ λ_j(xB-xA)_{i-j} + (yA_i - yB_i))
                let mut op_branch = ya(i) - yg(i);
                for j in 0..=i {
                    op_branch = op_branch + lam(j) * (xg(i - j) - xa(i - j));
                }
                // (1-op)·Σ (2 λ_j yA_{i-j} - 3 xA_j xA_{i-j})
                let mut notop_branch = b.zero();
                for j in 0..=i {
                    let two = b.const_base(2);
                    let three = b.const_base(3);
                    notop_branch =
                        notop_branch + two * lam(j) * ya(i - j) - three * xa(j) * xa(i - j);
                }
                op.clone() * op_branch + (one - op) * notop_branch + Self::rq(b, i, cols::Q0)
            }
            Relation::Xr => {
                // Σ λ_j λ_{i-j} − xA_i − xB_i − xR_i − (1-op)(xA_i − xB_i) + rq
                let mut s = b.zero();
                for j in 0..=i {
                    s = s + lam(j) * lam(i - j);
                }
                s - xa(i) - xg(i) - xr(i) - (one - op) * (xa(i) - xg(i)) + Self::rq(b, i, cols::Q1)
            }
            Relation::Yr => {
                // Σ λ_j(xA-xR)_{i-j} − yA_i − yR_i + rq
                let mut s = b.zero();
                for j in 0..=i {
                    s = s + lam(j) * (xa(i - j) - xr(i - j));
                }
                s - ya(i) - yr(i) + Self::rq(b, i, cols::Q2)
            }
            Relation::Dinv => {
                // g·(Σ d_j(xB−xA)_{i−j} − [i = 0]) + rq, with
                // g = S1 + S2 + S3 + S_CORR.
                //
                // `g` rather than `OP` deliberately: idx 14 makes the two equal,
                // but this way the non-degeneracy proof is tied to the same
                // expression that receives the addend, so the check can never
                // drift away from the rows that consume one.
                //
                // On a gated-off row (`g = 0`) only `rq` remains, which closes at
                // `q3 = 3p` with zero carries on a real doubling and at `q3 = 0`
                // on an all-zero padding row.
                let d = |j: usize| Self::byte_at(b, cols::D_INV, 32, j);
                let mut s = b.zero();
                for j in 0..=i {
                    s = s + d(j) * (xg(i - j) - xa(i - j));
                }
                if i == 0 {
                    s = s - one;
                }
                let g = b.main(0, cols::S1)
                    + b.main(0, cols::S2)
                    + b.main(0, cols::S3)
                    + b.main(0, cols::S_CORR);
                g * s + Self::rq(b, i, cols::Q3)
            }
        }
    }

    /// `256·c_i − c_{i-1} − S_i`.
    fn conv_carry<B: ConstraintBuilder<GoldilocksField, GoldilocksExtension>>(
        b: &B,
        relation: Relation,
        i: usize,
    ) -> B::Expr {
        let c_base = match relation {
            Relation::Lambda => cols::C0,
            Relation::Xr => cols::C1,
            Relation::Yr => cols::C2,
            Relation::Dinv => cols::C3,
        };
        let c_i = b.main(0, c_base + i);
        let c_prev = if i == 0 {
            b.zero()
        } else {
            b.main(0, c_base + i - 1)
        };
        let two_pow_8 = b.const_base(256);
        two_pow_8 * c_i - c_prev - Self::s_i(b, relation, i)
    }
}

impl ConstraintSet<GoldilocksField, GoldilocksExtension> for Ecdas2Constraints {
    // The Lambda ConvCarry has the op·(λ·Δx) term, making it degree 3; so are
    // the NB and precompute-selector constraints.
    fn max_degree(&self) -> usize {
        3
    }

    fn eval<B: ConstraintBuilder<GoldilocksField, GoldilocksExtension>>(&self, b: &mut B) {
        // idx 0..=10: unconditional IS_BIT `x·(1−x)` on every boolean column.
        for (i, col) in [
            cols::MU,
            cols::OP,
            cols::NB,
            cols::D1,
            cols::D2,
            cols::S1,
            cols::S2,
            cols::S3,
            cols::S_CORR,
            cols::PH1,
            cols::PH2,
        ]
        .into_iter()
        .enumerate()
        {
            let x = b.main(0, col);
            let one = b.one();
            b.emit_base(i, x.clone() * (one - x));
        }

        // idx 11: PH1·PH2 = 0, so PHASE = PH1 + 2·PH2 ranges over {0, 1, 2}.
        let ph1 = b.main(0, cols::PH1);
        let ph2 = b.main(0, cols::PH2);
        b.emit_base(11, ph1 * ph2);

        // idx 12: OP·NB = 0. An add row never announces another add at its own
        // round, so a round is visited by at most one add.
        let op = b.main(0, cols::OP);
        let nb = b.main(0, cols::NB);
        b.emit_base(12, op * nb);

        // idx 13: (1 − OP)·(NB − D1 − D2 + D1·D2) = 0. On a doubling `NB` is the
        // OR of that round's two digits, which is exactly "an add follows me".
        // Op-gated: an add row carries the same digits but always has NB = 0.
        let op = b.main(0, cols::OP);
        let nb = b.main(0, cols::NB);
        let d1 = b.main(0, cols::D1);
        let d2 = b.main(0, cols::D2);
        let one = b.one();
        b.emit_base(13, (one - op) * (nb - d1.clone() - d2.clone() + d1 * d2));

        // idx 14: OP = S1 + S2 + S3 + S_CORR. Adds consume exactly one addend;
        // doublings consume none. Without this the double-row addend
        // cancellation is real but its *gating* is forgeable — a prover would
        // set S2 = 1 on a doubling and mint a spurious Addend receive.
        let op = b.main(0, cols::OP);
        let s1 = b.main(0, cols::S1);
        let s2 = b.main(0, cols::S2);
        let s3 = b.main(0, cols::S3);
        let s_corr = b.main(0, cols::S_CORR);
        b.emit_base(14, op - s1 - s2 - s3 - s_corr);

        // idx 15, 16: (1 − PH1)·D = 0. Digits live only on main-chain rows.
        // The precompute and correction rows are both emitted at `round = 0`, so
        // without this a prover sets D1 = 1 on both of them and satisfies the
        // `2·u1_bit(0)` JointBit receive with no round-0 add at all — u1's low
        // bit would be consumed without ever being added.
        let one = b.one();
        let ph1 = b.main(0, cols::PH1);
        let d1 = b.main(0, cols::D1);
        b.emit_base(15, (one - ph1) * d1);
        let one = b.one();
        let ph1 = b.main(0, cols::PH1);
        let d2 = b.main(0, cols::D2);
        b.emit_base(16, (one - ph1) * d2);

        // idx 17: PH1·S_CORR = 0. The T₀ correction constant is not available to
        // the main chain.
        let ph1 = b.main(0, cols::PH1);
        let s_corr = b.main(0, cols::S_CORR);
        b.emit_base(17, ph1 * s_corr);

        // idx 18, 19: on the main chain the addend is exactly the one the two
        // digits select. Written as the two degree-3 sums
        //   S1 + S3 = OP·D1     ("the addend includes P1 iff u1's digit is set")
        //   S2 + S3 = OP·D2
        // rather than the three one-hot products, which would need degree 4 to
        // carry the OP gate. Together with idx 14 they force
        // (D1,D2) = (1,0) ⇒ S1, (0,1) ⇒ S2, (1,1) ⇒ S3, and make (0,0) with
        // OP = 1 unsatisfiable — so no spurious add can be inserted. On a
        // doubling OP = 0 makes both sides zero, consistent with idx 14.
        let ph1 = b.main(0, cols::PH1);
        let s1 = b.main(0, cols::S1);
        let s3 = b.main(0, cols::S3);
        let op = b.main(0, cols::OP);
        let d1 = b.main(0, cols::D1);
        b.emit_base(18, ph1 * (s1 + s3 - op * d1));
        let ph1 = b.main(0, cols::PH1);
        let s2 = b.main(0, cols::S2);
        let s3 = b.main(0, cols::S3);
        let op = b.main(0, cols::OP);
        let d2 = b.main(0, cols::D2);
        b.emit_base(19, ph1 * (s2 + s3 - op * d2));

        // idx 20: MU·(1 − PH1 − PH2)·(S2 − 1) = 0. The single phase-0 row adds
        // P2. Without it a prover could point the precompute at P1, making the
        // chord `P1 + P1` — whose λ relation degenerates to 0 = 0 and admits any
        // λ, i.e. an arbitrary "P12". MU-gated so all-zero padding rows are free.
        let mu = b.main(0, cols::MU);
        let one = b.one();
        let ph1 = b.main(0, cols::PH1);
        let ph2 = b.main(0, cols::PH2);
        let s2 = b.main(0, cols::S2);
        let one2 = b.one();
        b.emit_base(20, mu * (one - ph1 - ph2) * (s2 - one2));

        // idx 21: PH2·(S_CORR − 1) = 0. The single phase-2 row adds the T₀
        // constant.
        let ph2 = b.main(0, cols::PH2);
        let s_corr = b.main(0, cols::S_CORR);
        let one = b.one();
        b.emit_base(21, ph2 * (s_corr - one));

        // idx 22..=27: (1 − MU)·x = 0 for every column that is a bus
        // *multiplicity*. A padding row is inert on the μ-gated interactions by
        // construction, but these two are not μ-gated — the digit sends count
        // `D1`/`D2` and the Addend receive counts `S1 + S2 + S3 + S_CORR` — so
        // without this a padding row still emits. The single-scalar chip carries
        // the same defence for its Bit-bus sender (`ecdas.rs`, idx 4:
        // `NEXT_OP·(1 − MU)`).
        //
        // Both holes are live forgeries, not hygiene:
        //
        // * a `MU = 0, PH1 = 1, NB = 1, D1 = 1, ROUND = r` row satisfies every
        //   other constraint and emits a real `JointBit[ts, r, 1]`. Two of them
        //   supply the `2·u1_bit(r)` an honest round pays with its doubling *and*
        //   its add, so the prover can drop the add at round `r` entirely (with
        //   `D1 = D2 = 0` the doubling's `NB` is 0 and nothing demands one). The
        //   chain then computes `(u1 − 2^r)·P1 + u2·P2`; back-solving the
        //   signature for a chosen target needs one modular inversion and no
        //   discrete log.
        // * a `MU = 0, OP = 1, S2 = 1` row keeps `OP = ΣS` satisfied and mints a
        //   spurious Addend receive.
        //
        // `NB` needs no companion: with every selector zero, `OP = ΣS` forces
        // `OP = 0`, and idx 13 then reads `NB = D1 ∨ D2 = 0`.
        for (i, col) in [
            cols::D1,
            cols::D2,
            cols::S1,
            cols::S2,
            cols::S3,
            cols::S_CORR,
        ]
        .into_iter()
        .enumerate()
        {
            let one = b.one();
            let mu = b.main(0, cols::MU);
            let x = b.main(0, col);
            b.emit_base(22 + i, (one - mu) * x);
        }

        // Per relation: 64 ConvCarry (i = 0..64) + 1 ColIsZero(c_63).
        let mut idx = 28;
        for (relation, c_base) in [
            (Relation::Lambda, cols::C0),
            (Relation::Xr, cols::C1),
            (Relation::Yr, cols::C2),
            (Relation::Dinv, cols::C3),
        ] {
            for i in 0..64 {
                let root = Self::conv_carry(b, relation, i);
                b.emit_base(idx, root);
                idx += 1;
            }
            let c_last = b.main(0, c_base + 63);
            b.emit_base(idx, c_last); // ColIsZero c_63
            idx += 1;
        }

        debug_assert_eq!(idx, 288);
    }
}
