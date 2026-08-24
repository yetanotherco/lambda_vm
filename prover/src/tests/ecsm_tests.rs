//! Tests for the ECSM core table — constraint satisfaction on generated traces,
//! the single-source constraint count, and isolated negative checks for the
//! padding closure, the scalar-bit padding guard and the `xG < p` overflow.

use crate::tables::ecsm::{
    EcsmConstraints, EcsmOperation, bus_interactions, cols, generate_ecsm_trace,
};
use crate::tables::types::{BusId, FE, GoldilocksExtension, GoldilocksField};
use ecsm::{N_BYTES, P_BYTES, compute_witness};
use math::field::element::FieldElement;
use stark::constraints::builder::{ConstraintSet, ProverEvalFolder};
use stark::frame::Frame;
use stark::lookup::{BusValue, LinearTerm};
use stark::table::TableView;
use stark::trace::TraceTable;
use stark::traits::TransitionEvaluationContext;

// Constraint indices in the single `EcsmConstraints::eval` body (see the index map there).
const IDX_KBITS_ZERO: usize = 257; // KBitsZeroOnPadding
const IDX_X2_CONV0: usize = 258; // ConvCarry(X2, 0)
const IDX_YG_CONV0: usize = 323; // ConvCarry(Yg, 0)
const IDX_Q1_BIT32: usize = 388; // IS_BIT(q1[32])
const IDX_XG_CARRY0: usize = 389; // CarryBit(XgLtP, 0)
const IDX_XG_OVERFLOW: usize = 396; // OverflowRequired(XgLtP)
const IDX_YR_OVERFLOW: usize = 420; // OverflowRequired(YrLtP)
const IDX_YG_OVERFLOW: usize = 428; // OverflowRequired(YgLtP)

fn gx_le() -> [u8; 32] {
    // secp256k1 Gx, little-endian.
    let mut be = [
        0x79, 0xBE, 0x66, 0x7E, 0xF9, 0xDC, 0xBB, 0xAC, 0x55, 0xA0, 0x62, 0x95, 0xCE, 0x87, 0x0B,
        0x07, 0x02, 0x9B, 0xFC, 0xDB, 0x2D, 0xCE, 0x28, 0xD9, 0x59, 0xF2, 0x81, 0x5B, 0x16, 0xF8,
        0x17, 0x98,
    ];
    be.reverse();
    be
}

fn k_le(v: u64) -> [u8; 32] {
    let mut k = [0u8; 32];
    k[..8].copy_from_slice(&v.to_le_bytes());
    k
}

fn op_for(k: u64) -> EcsmOperation {
    let witness = compute_witness(&k_le(k), &gx_le()).unwrap();
    EcsmOperation {
        timestamp: 444,
        addr_xg: 0x2000,
        addr_k: 0x3000,
        addr_xr: 0x1000,
        witness,
    }
}

/// Evaluate the ECSM [`ConstraintSet`] on a single main-trace row (the compiled
/// prover folder path), returning every base-field constraint value.
fn eval_main_row(main: Vec<FE>) -> Vec<FE> {
    let n = EcsmConstraints.meta().len();
    let frame = Frame::<GoldilocksField, GoldilocksExtension>::new(vec![TableView::new(
        vec![main],
        vec![vec![]],
    )]);
    let no_e: Vec<FieldElement<GoldilocksExtension>> = vec![];
    let offset_e = FieldElement::<GoldilocksExtension>::zero();
    let ctx =
        TransitionEvaluationContext::new_prover(frame.as_row_frame(), &no_e, &no_e, &offset_e);
    let mut base = vec![FE::zero(); n];
    let mut ext = vec![FieldElement::<GoldilocksExtension>::zero(); n];
    let mut folder = ProverEvalFolder::new(&ctx, &mut base, &mut ext);
    EcsmConstraints.eval(&mut folder);
    base
}

/// Evaluate the constraint set on trace row `row`.
fn eval_row(trace: &TraceTable<GoldilocksField, GoldilocksExtension>, row: usize) -> Vec<FE> {
    let main: Vec<FE> = (0..cols::NUM_COLUMNS)
        .map(|c| *trace.main_table.get(row, c))
        .collect();
    eval_main_row(main)
}

/// Every ECSM constraint evaluates to zero on a generated trace (real + padding
/// rows). Exercises the all-zero padding closure (µ-gated `p²` and `b`) end to end.
#[test]
fn constraints_hold_on_generated_trace() {
    let ops: Vec<EcsmOperation> = [1u64, 2, 5, 0xFFFF, 1_000_003]
        .iter()
        .map(|&k| op_for(k))
        .collect();
    let trace = generate_ecsm_trace(&ops);

    for row in 0..trace.num_rows() {
        for (i, v) in eval_row(&trace, row).iter().enumerate() {
            assert_eq!(*v, FE::zero(), "constraint {i} must hold at row {row}");
        }
    }
}

#[test]
fn constraint_set_count() {
    assert_eq!(EcsmConstraints.meta().len(), 429);
}

/// The yG carry recurrence closes on all-zero padding because both the `µ·p²` offset and the
/// curve constant `µ·b` are multiplied by `µ`, so they vanish when `µ = 0`. This checks the
/// closing argument (Yg limb-0 ConvCarry = constraint `IDX_YG_CONV0`) and its two ingredients.
#[test]
fn yg_padding_closes_via_mu_gated_p2_and_b() {
    // yG limb-0 ConvCarry residual on a one-off row with the given `µ` and `q1`.
    let yg_residual = |mu: u64, q1_is_p: bool| {
        let mut main = vec![FE::zero(); cols::NUM_COLUMNS];
        main[cols::MU] = FE::from(mu);
        if q1_is_p {
            for (i, &b) in P_BYTES.iter().enumerate() {
                main[cols::Q1 + i] = FE::from(b as u64);
            }
        }
        eval_main_row(main)[IDX_YG_CONV0]
    };

    // Padding row (µ=0, q1=0): µ gates away both p² and b → closes trivially.
    assert_eq!(
        yg_residual(0, false),
        FE::zero(),
        "padding row (µ=0, q1=0) must close"
    );

    // µ=0 but q1=p: µ·p² is gated away, so q1·p is unmatched → residual = +P_0² = +47² = 2209.
    assert_eq!(
        yg_residual(0, true),
        FE::from(2209u64),
        "µ=0 with q1=p leaves +P_0² = +47² residual"
    );

    // µ=1, q1=0: s_i = µ·P_0² − µ·b = 2209 − 7 = 2202 → residual (256·c − c_prev − s_i) = −2202.
    assert_eq!(
        yg_residual(1, false),
        FE::zero() - FE::from(2202u64),
        "µ=1, q1=0: residual = −(P_0² − b) = −2202"
    );

    // x² has no standalone constant → closes on an all-zero padding row regardless.
    let mut zero = vec![FE::zero(); cols::NUM_COLUMNS];
    zero[cols::MU] = FE::zero();
    assert_eq!(
        eval_main_row(zero)[IDX_X2_CONV0],
        FE::zero(),
        "x² closes on all-zero padding (no standalone constant)"
    );
}

/// A µ=0 padding row with any k_bit set must violate KBitsZeroOnPadding.
/// Guards against a prover injecting phantom `Bit` bus receives on padding rows.
#[test]
fn k_bits_zero_on_padding_rejects_forged_row() {
    // Single forged bit on padding row: sum=1, (1−µ)=1 → 1.
    let mut main = vec![FE::zero(); cols::NUM_COLUMNS];
    main[cols::k_bit(0)] = FE::one();
    assert_eq!(
        eval_main_row(main.clone())[IDX_KBITS_ZERO],
        FE::one(),
        "k_bit[0]=1 on µ=0 must fire (residual=1)"
    );

    // Same bit on an active row (µ=1): constraint holds.
    main[cols::MU] = FE::one();
    assert_eq!(
        eval_main_row(main)[IDX_KBITS_ZERO],
        FE::zero(),
        "k_bit[0]=1 on µ=1 must not fire"
    );

    // Multiple forged bits: sum=3, residual=3.
    let mut main_multi = vec![FE::zero(); cols::NUM_COLUMNS];
    main_multi[cols::k_bit(0)] = FE::one();
    main_multi[cols::k_bit(7)] = FE::one();
    main_multi[cols::k_bit(255)] = FE::one();
    assert_eq!(
        eval_main_row(main_multi)[IDX_KBITS_ZERO],
        FE::from(3u64),
        "3 forged k_bits → residual=3"
    );
}

/// OverflowRequired for XgLtP evaluates non-zero when xG = p (no valid xg_sub_p exists).
/// All CarryBit constraints still hold (c_i=0 is a valid bit), but the carry chain never
/// reaches c_7=1, so OverflowRequired = µ·(1−c_7) = 1 fires.
#[test]
fn xg_ge_p_overflow_required_fires() {
    let mut main = vec![FE::zero(); cols::NUM_COLUMNS];
    main[cols::MU] = FE::one();
    // xG = p, xg_sub_p = 0 (invalid subtraction witness — fine for this isolation test).
    for (i, &b) in P_BYTES.iter().enumerate() {
        main[cols::xg(i)] = FE::from(b as u64);
    }
    let row = eval_main_row(main);

    for i in 0..7 {
        assert_eq!(
            row[IDX_XG_CARRY0 + i],
            FE::zero(),
            "carry bit {i}: c_i=0 is a valid bit"
        );
    }
    assert_ne!(
        row[IDX_XG_OVERFLOW],
        FE::zero(),
        "OverflowRequired must fire when xG = p"
    );
}

/// `yR` and `yG` are published to guest memory, and the caller resolves the root by comparing
/// the echoed `yG` against its own `y`. A non-canonical representative would break that: `p`
/// is odd, so `y` and `p − y` differ in parity, but `y + p` — a second 256-bit representative
/// whenever `y < 2^256 − p ≈ 2^32`, and reachable, since `3 | p−1` leaves a third of the small
/// `y` with a curve `x` — carries the opposite parity. `OverflowRequired` is what forbids it;
/// `y = p` is the boundary where the addition stops overflowing.
#[test]
fn yr_and_yg_ge_p_overflow_required_fires() {
    for (coord, idx, name) in [
        (cols::YR, IDX_YR_OVERFLOW, "yR"),
        (cols::YG, IDX_YG_OVERFLOW, "yG"),
    ] {
        let mut main = vec![FE::zero(); cols::NUM_COLUMNS];
        main[cols::MU] = FE::one();
        // y = p, y_sub_p = 0 (invalid subtraction witness — fine for this isolation test).
        for (i, &b) in P_BYTES.iter().enumerate() {
            main[coord + i] = FE::from(b as u64);
        }
        let row = eval_main_row(main);
        assert_ne!(
            row[idx],
            FE::zero(),
            "OverflowRequired must fire when {name} = p"
        );
    }
}

fn five_g_x_le() -> [u8; 32] {
    // x-coordinate of 5·G (secp256k1), little-endian.
    // 0x2f8bde4d1a07209355b4a7250a5c5128e88b84bddc619ab7cba8d569b240efe4
    [
        0xe4, 0xef, 0x40, 0xb2, 0x69, 0xd5, 0xa8, 0xcb, 0xb7, 0x9a, 0x61, 0xdc, 0xbd, 0x84, 0x8b,
        0xe8, 0x28, 0x51, 0x5c, 0x0a, 0x25, 0xa7, 0xb4, 0x55, 0x93, 0x20, 0x07, 0x1a, 0x4d, 0xde,
        0x8b, 0x2f,
    ]
}

/// Exercises the q1[32]=1 path by using x(5·G) as the base point, which produces a yG
/// quotient whose high byte (index 32) equals 1. IS_BIT(q1[32]) must still hold.
#[test]
fn q1_bit32_equals_one_path() {
    let witness =
        compute_witness(&k_le(1), &five_g_x_le()).expect("k=1, xG=x(5·G) is a valid ECSM input");
    assert_eq!(
        witness.q1[32], 1,
        "sanity: q1[32] should be 1 for x(5·G) as base point"
    );

    let op = EcsmOperation {
        timestamp: 100,
        addr_xg: 0x2000,
        addr_k: 0x3000,
        addr_xr: 0x1000,
        witness,
    };
    let trace = generate_ecsm_trace(&[op]);

    for row in 0..trace.num_rows() {
        assert_eq!(
            eval_row(&trace, row)[IDX_Q1_BIT32],
            FE::zero(),
            "IS_BIT(q1[32]) must hold (value=1) at row {row}"
        );
    }
}

/// End-to-end constraint check for k = N−1, the maximum valid scalar (len_k = 255).
#[test]
fn constraints_hold_for_k_eq_n_minus_one() {
    let mut k_bytes = N_BYTES;
    k_bytes[0] -= 1; // N is odd, so N_BYTES[0] >= 1; this gives N-1 in little-endian.
    let witness = compute_witness(&k_bytes, &gx_le()).expect("N-1 is a valid scalar");
    assert_eq!(witness.len_k, 255, "N-1 has MSB at bit 255");

    let op = EcsmOperation {
        timestamp: 999,
        addr_xg: 0x2000,
        addr_k: 0x3000,
        addr_xr: 0x1000,
        witness,
    };
    let trace = generate_ecsm_trace(&[op]);

    for row in 0..trace.num_rows() {
        for (i, v) in eval_row(&trace, row).iter().enumerate() {
            assert_eq!(*v, FE::zero(), "constraint {i} must hold at row {row}");
        }
    }
}

// =========================================================================
// Structural: the 96-byte output buffer and its ECDAS backing
// =========================================================================

/// The single column of a `Packed` bus value.
fn packed_col(v: &BusValue) -> Option<usize> {
    match v {
        BusValue::Packed { start_column, .. } => Some(*start_column),
        _ => None,
    }
}

/// `(column, constant)` of a `Linear` bus value shaped `1·col + k`.
fn linear_col_plus(v: &BusValue) -> Option<(usize, i64)> {
    let BusValue::Linear(terms) = v else {
        return None;
    };
    let mut col = None;
    let mut constant = 0i64;
    for t in terms {
        match t {
            LinearTerm::Column {
                coefficient: 1,
                column,
            } => col = Some(*column),
            LinearTerm::Constant(k) => constant += k,
            _ => return None,
        }
    }
    col.map(|c| (c, constant))
}

/// The chip publishes `[xR ‖ yR ‖ yG]` as twelve doubleword MEMW writes, and the aliasing
/// argument rests on WHEN they happen: the operands are read at `T` and `T + 1`, so every
/// write must land strictly later, or an output buffer overlapping `xG`/`k` would touch one
/// address twice at the same timestamp. `yR` and `yG` share `T + 3`, which is legal only
/// because their address ranges are disjoint — and `T + 3` is the last sub-timestamp of the
/// instruction's stride-4 window, so there is no room for a fourth group.
///
/// Pin the layout: nothing else fails loudly if a write moves back onto a read's timestamp
/// or one of the two new coordinates stops being published.
#[test]
fn output_buffer_memw_writes_are_twelve_at_the_expected_offsets() {
    let want: Vec<(usize, i64, i64)> =
        [(cols::XR, 0i64, 2i64), (cols::YR, 32, 3), (cols::YG, 64, 3)]
            .iter()
            .flat_map(|&(col, off, ts)| (0..4).map(move |i| (col + 8 * i, off + 8 * i as i64, ts)))
            .collect();

    // MEMW write tuple: [is_register, base_lo, base_hi, value[8], ts_lo, ts_hi, w2, w4, w8].
    let got: Vec<(usize, i64, i64)> = bus_interactions()
        .iter()
        .filter(|b| b.is_sender && b.bus_id == BusId::Memw as u64 && b.values.len() == 16)
        .filter_map(|b| {
            // Every other MEMW access this chip makes is a read, and the length filter
            // already dropped those (24-element tuple), so these twelve are the buffer.
            let (addr_col, addr_off) = linear_col_plus(&b.values[1])?;
            assert_eq!(addr_col, cols::ADDR_XR_0);
            assert_eq!(packed_col(&b.values[2]), Some(cols::ADDR_XR_1));
            let (ts_col, ts_off) = linear_col_plus(&b.values[11])?;
            assert_eq!(ts_col, cols::TIMESTAMP_0);
            Some((packed_col(&b.values[3])?, addr_off, ts_off))
        })
        .collect();

    assert_eq!(
        got, want,
        "the [xR ‖ yR ‖ yG] write group changed shape (column, address offset, ts offset)"
    );
}

/// `yR` is not a free column, and neither is `yG`.
///
/// The ECDAS final receiver carries `[id, ts, xR, yR, xG, yG, −1, 0]`, so the published `yR`
/// has to match the constrained double-and-add output; the start sender carries the same
/// `(xG, yG)` the `Relation::Yg` convolution binds to the curve. The caller-side sign fix-up
/// (`ŷ = ±y` ⇒ negate or not) is sound only while both hold, so pin the tuple offsets.
#[test]
fn ecdas_tuples_carry_the_published_coordinates() {
    // ECDAS tuple: [id, ts_lo, ts_hi, accX(32), accY(32), genX(32), genY(32), round, op].
    const ACC_X: usize = 3;
    const ACC_Y: usize = ACC_X + 32;
    const GEN_X: usize = ACC_Y + 32;
    const GEN_Y: usize = GEN_X + 32;

    let coord_is = |values: &[BusValue], at: usize, base: usize| {
        (0..32).all(|b| packed_col(&values[at + b]) == Some(base + b))
    };

    let ecdas: Vec<_> = bus_interactions()
        .into_iter()
        .filter(|b| b.bus_id == BusId::Ecdas as u64)
        .collect();
    assert_eq!(
        ecdas.len(),
        2,
        "ECSM sends the start tuple and receives the result"
    );

    let start = ecdas
        .iter()
        .find(|b| b.is_sender)
        .expect("ECDAS start sender");
    assert!(coord_is(&start.values, ACC_X, cols::XG));
    assert!(coord_is(&start.values, ACC_Y, cols::YG));

    let result = ecdas
        .iter()
        .find(|b| !b.is_sender)
        .expect("ECDAS final receiver");
    assert!(
        coord_is(&result.values, ACC_X, cols::XR),
        "xR must come from the ECDAS accumulator"
    );
    assert!(
        coord_is(&result.values, ACC_Y, cols::YR),
        "yR must come from the ECDAS accumulator, not be a free witness"
    );
    assert!(coord_is(&result.values, GEN_X, cols::XG));
    assert!(coord_is(&result.values, GEN_Y, cols::YG));
}
