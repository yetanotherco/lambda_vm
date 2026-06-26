//! Tests for the ECDAS double/add table — the `R_BYTES` offset constant, constraint
//! satisfaction on generated traces across many scalars, and the constraint count.

use crate::constraints::templates::IsBitConstraint;
use crate::tables::ecdas::{
    ColIsZero, ConvCarry, EcdasOperation, MulZero, R_BYTES, Relation, cols, create_constraints,
    generate_ecdas_trace,
};
use crate::tables::types::{FE, GoldilocksExtension, GoldilocksField};
use ecsm::compute_witness;
use stark::constraints::transition::TransitionConstraint;
use stark::table::TableView;
use stark::trace::TraceTable;

fn gx_le() -> [u8; 32] {
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

fn ops_for_bytes(k_le: &[u8; 32]) -> Vec<EcdasOperation> {
    let w = compute_witness(k_le, &gx_le()).unwrap();
    w.steps
        .into_iter()
        .map(|step| EcdasOperation {
            timestamp: 444,
            step,
        })
        .collect()
}

fn ops_for(k: u64) -> Vec<EcdasOperation> {
    ops_for_bytes(&k_le(k))
}

fn row_view(
    trace: &TraceTable<GoldilocksField, GoldilocksExtension>,
    row: usize,
) -> TableView<GoldilocksField, GoldilocksExtension> {
    let main: Vec<FE> = (0..cols::NUM_COLUMNS)
        .map(|c| *trace.main_table.get(row, c))
        .collect();
    TableView::new(vec![main], vec![])
}

#[test]
fn r_bytes_is_three_p() {
    // 3·p as 33 little-endian bytes, cross-checked against the ecsm field modulus.
    // R_BYTES encodes 3p as 33 LE bytes; compute 3*P_BYTES using u128 carry arithmetic.
    let p = ecsm::P_BYTES;
    let mut three_p = [0u8; 33];
    let mut carry: u16 = 0;
    for i in 0..32 {
        let s = p[i] as u16 * 3 + carry;
        three_p[i] = s as u8;
        carry = s >> 8;
    }
    three_p[32] = carry as u8;
    assert_eq!(&three_p[..], &R_BYTES[..]);
}

/// Every ECDAS constraint evaluates to zero on a generated trace across many scalars
/// (which exercise both double and add steps), including padding rows.
#[test]
fn constraints_hold_on_generated_trace() {
    for k in [2u64, 3, 5, 7, 0xFF, 0xABCD, 1_000_003] {
        let ops = ops_for(k);
        assert!(!ops.is_empty(), "k={k} should have steps");
        let trace = generate_ecdas_trace(&ops);

        for row in 0..trace.num_rows() {
            let view = row_view(&trace, row);
            assert_eq!(
                IsBitConstraint::unconditional(cols::MU, 0).evaluate(&view),
                FE::zero(),
                "is_bit(mu) k={k} row {row}"
            );
            assert_eq!(
                IsBitConstraint::unconditional(cols::NEXT_OP, 0).evaluate(&view),
                FE::zero()
            );
            assert_eq!(
                IsBitConstraint::unconditional(cols::OP, 0).evaluate(&view),
                FE::zero()
            );
            assert_eq!(
                MulZero {
                    a: cols::OP,
                    b: cols::NEXT_OP,
                    b_complement: false,
                    constraint_idx: 0
                }
                .evaluate(&view),
                FE::zero(),
                "op·next_op k={k} row {row}"
            );
            assert_eq!(
                MulZero {
                    a: cols::NEXT_OP,
                    b: cols::MU,
                    b_complement: true,
                    constraint_idx: 0
                }
                .evaluate(&view),
                FE::zero()
            );
            for relation in [Relation::Lambda, Relation::Xr, Relation::Yr] {
                for i in 0..64 {
                    let v = ConvCarry {
                        relation,
                        i,
                        constraint_idx: 0,
                    }
                    .evaluate(&view);
                    assert_eq!(v, FE::zero(), "conv k={k} i={i} row {row}");
                }
            }
            for c_base in [cols::C0, cols::C1, cols::C2] {
                assert_eq!(
                    ColIsZero {
                        col: c_base + 63,
                        constraint_idx: 0
                    }
                    .evaluate(&view),
                    FE::zero()
                );
            }
        }
    }
}

/// Worst-case carries: N-1 (largest valid scalar) runs the full 256-bit ladder.
#[test]
fn constraints_hold_for_near_order_scalar() {
    let mut k = ecsm::N_BYTES;
    k[0] -= 1;
    let ops = ops_for_bytes(&k);
    assert!(!ops.is_empty());
    let trace = generate_ecdas_trace(&ops);
    for row in 0..trace.num_rows() {
        let view = row_view(&trace, row);
        for relation in [Relation::Lambda, Relation::Xr, Relation::Yr] {
            for i in 0..64 {
                assert_eq!(
                    ConvCarry {
                        relation,
                        i,
                        constraint_idx: 0
                    }
                    .evaluate(&view),
                    FE::zero(),
                    "conv N-1 i={i} row {row}"
                );
            }
        }
    }
}

#[test]
fn create_constraints_count() {
    let (constraints, next) = create_constraints(0);
    assert_eq!(constraints.len(), 200);
    assert_eq!(next, 200);
}
