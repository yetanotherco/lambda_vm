//! Tests for the ECSM core table — constraint satisfaction on generated traces,
//! the single-source constraint count, and isolated negative checks for the
//! padding closure, the scalar-bit padding guard and the `xG < p` overflow.

use crate::tables::ecsm::{EcsmConstraints, EcsmOperation, cols, generate_ecsm_trace};
use crate::tables::types::{FE, GoldilocksExtension, GoldilocksField};
use ecsm::{N_BYTES, P_BYTES, compute_witness};
use math::field::element::FieldElement;
use stark::constraints::builder::{ConstraintSet, ProverEvalFolder};
use stark::frame::Frame;
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
const IDX_ADDR_XG_ADD: usize = 413; // AddCarryPair(addr_xG[1]), 2 per address, i = 1..=3
const IDX_ADDR_XR_ADD: usize = 425; // AddCarryPair(addr_xR[1])
const IDX_ADDR_XG_NOWRAP: usize = 431; // µ·carry_1 = 0 on addr_xG[3]
const IDX_ADDR_XR_NOWRAP: usize = 433; // µ·carry_1 = 0 on addr_xR[3]

/// Halfword accessor for one operand's per-access address columns.
type AccFn = fn(usize, usize) -> usize;

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
    assert_eq!(EcsmConstraints.meta().len(), 434);
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

/// `ec:c:extrapolate_addr_*`: a per-access address that is not `addr[0] + 8i` breaks the
/// addition. Without these constraints the twelve MEMW accesses could be sent at twelve
/// unrelated addresses, which is the deviation #902 reports.
///
/// Run over all three operands: the constraint block pairs each operand's base column with its
/// own accessor, and a copy/paste slip there (xG's base against k's columns, say) would satisfy
/// every test that only exercises xG.
#[test]
fn extrapolate_addr_rejects_a_wrong_per_access_address() {
    let trace = generate_ecsm_trace(&[op_for(5)]);
    let clean = eval_row(&trace, 0);
    for (i, v) in clean.iter().enumerate() {
        assert_eq!(*v, FE::zero(), "constraint {i} must hold on the clean row");
    }

    // (accessor, first ADD constraint of that operand's block). Blocks are operand-major with
    // stride 6: xG 413..418, k 419..424, xR 425..430; within a block, i = 1,2,3 x (carry_0, carry_1).
    let operands: [(AccFn, usize, &str); 3] = [
        (cols::addr_xg_acc, IDX_ADDR_XG_ADD, "xG"),
        (cols::addr_k_acc, IDX_ADDR_XG_ADD + 6, "k"),
        (cols::addr_xr_acc, IDX_ADDR_XR_ADD, "xR"),
    ];
    for (acc, block, name) in operands {
        for i in 1..4 {
            let mut main: Vec<FE> = (0..cols::NUM_COLUMNS)
                .map(|c| *trace.main_table.get(0, c))
                .collect();
            // Move addr_*[i] by one byte.
            main[acc(i, 0)] = main[acc(i, 0)] + FE::one();
            let row = eval_main_row(main);
            let idx = block + (i - 1) * 2;
            assert_ne!(
                row[idx],
                FE::zero(),
                "{name}: a wrong addr[{i}] must break constraint {idx}"
            );
        }
    }
}

/// The `µ·carry_1 = 0` addition, which the spec's `ADD` template does not give us: without it
/// `addr[3] = addr[0] + 24 − 2^64` satisfies the carry pair, so an operand could wrap the
/// address space in-circuit while the executor refuses it. The wrapped address is a small,
/// matchable one (`0x10` here), so what this blocks is a collision with legitimate memory, not
/// merely an unsatisfiable trace. Checked on all three operands.
#[test]
fn addr_wrapping_past_u64_is_rejected() {
    let operands: [(usize, AccFn, usize, usize, &str); 3] = [
        (
            cols::ADDR_XG_0,
            cols::addr_xg_acc,
            IDX_ADDR_XG_ADD,
            IDX_ADDR_XG_NOWRAP,
            "xG",
        ),
        (
            cols::ADDR_K_0,
            cols::addr_k_acc,
            IDX_ADDR_XG_ADD + 6,
            IDX_ADDR_XG_NOWRAP + 1,
            "k",
        ),
        (
            cols::ADDR_XR_0,
            cols::addr_xr_acc,
            IDX_ADDR_XR_ADD,
            IDX_ADDR_XR_NOWRAP,
            "xR",
        ),
    ];
    for (base_col, acc, add_block, nowrap, name) in operands {
        let mut main = vec![FE::zero(); cols::NUM_COLUMNS];
        main[cols::MU] = FE::one();
        // base = u64::MAX - 7, so addr[3] = base + 24 wraps to 0x10.
        let base = u64::MAX - 7;
        main[base_col] = FE::from(base & 0xFFFF_FFFF);
        main[base_col + 1] = FE::from(base >> 32);
        for i in 1..4u64 {
            let a = base.wrapping_add(8 * i);
            for hw in 0..4 {
                main[acc(i as usize, hw)] = FE::from((a >> (16 * hw)) & 0xFFFF);
            }
        }
        let row = eval_main_row(main);
        // Every carry bit of the three additions is satisfied by the wrapped witness...
        for offset in 0..6 {
            assert_eq!(
                row[add_block + offset],
                FE::zero(),
                "{name}: the wrapped addition still satisfies carry bit {offset}"
            );
        }
        // ...so only the no-wrap constraint catches it.
        assert_ne!(
            row[nowrap],
            FE::zero(),
            "{name}: an operand whose last address wraps u64 must be rejected"
        );
    }
}

/// Padding rows stay valid: every new constraint is µ-gated, so an all-zero row closes.
#[test]
fn addr_constraints_close_on_padding() {
    let main = vec![FE::zero(); cols::NUM_COLUMNS];
    let row = eval_main_row(main);
    for (offset, v) in row[IDX_ADDR_XG_ADD..=IDX_ADDR_XR_NOWRAP].iter().enumerate() {
        let idx = IDX_ADDR_XG_ADD + offset;
        assert_eq!(*v, FE::zero(), "constraint {idx} must close at µ = 0");
    }
}

/// The invariant the MSB16 bus bug broke, as a fast test instead of only end-to-end proving:
/// every `IsHalfword` interaction the ECSM row sends must have exactly one matching lookup in
/// `collect_bitwise_from_ecsm`, which is what feeds the receive multiplicity. The two live in
/// different files and are written by hand, so nothing but a count keeps them in step.
#[test]
fn is_half_sends_match_the_collector() {
    use crate::tables::bitwise::BitwiseOperationType;

    let sends = crate::tables::ecsm::bus_interactions()
        .iter()
        .filter(|b| b.is_sender && b.bus_id == u64::from(crate::tables::types::BusId::IsHalfword))
        .count();

    let op = op_for(5);
    let collected = crate::tables::trace_builder::collect_bitwise_from_ecsm(&[op])
        .iter()
        .filter(|o| o.lookup_type == BitwiseOperationType::IsHalf)
        .count();

    assert_eq!(
        sends, collected,
        "each IsHalfword send must have one collected lookup: {sends} sends vs {collected} lookups"
    );
    // The 36 address halfwords are part of that total; a regression that dropped them would
    // still balance if the collector lost them too, so pin the address share explicitly.
    assert!(
        sends >= 36,
        "the per-access address halfwords must be among the sends"
    );
}
