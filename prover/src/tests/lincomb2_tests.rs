//! ECSM2 / ECDAS2 (lincomb2) chip tests.
//!
//! The end-to-end gate lives in `prove_elfs_tests` (a real guest proving and
//! verifying). This file is the trace-level gate: over a random corpus it checks
//! that every in-chip transition constraint holds on every emitted row, that the
//! chip's `Q` is the witness's `Q`, and that the four buses the joint chain owns
//! balance exactly. Plus layout/bus-shape pins and negative controls, following
//! the `ec_t0_tests` precedent of pinning a chip's contract in its own file.

use std::collections::HashMap;

use math::field::element::FieldElement;
use math::field::traits::IsPrimeField;
use num_bigint::BigUint;
use stark::lookup::{BusValue, LinearTerm, Multiplicity};
use stark::trace::TraceTable;

use ecsm::witness::{JointSel, Lincomb2Witness, lincomb2_witness};
use ecsm::{AffinePoint, n, replay_double_and_add, to_le_32};

use crate::tables::types::{BusId, GoldilocksExtension, GoldilocksField};
use crate::tables::{ecdas2, ecsm2};
use crate::test_utils::{busless_air, validate_busless};

type F = GoldilocksField;
type E = GoldilocksExtension;

/// Number of random lincomb2 evaluations the corpus tests cover.
const CASES: usize = 100;

// =============================================================================
// Corpus helpers
// =============================================================================

/// The secp256k1 generator, as the executor pins it.
fn generator() -> AffinePoint {
    let g = executor::vm::instruction::execution::GENERATOR_LE;
    AffinePoint {
        x: BigUint::from_bytes_le(&g[..32]),
        y: BigUint::from_bytes_le(&g[32..]),
    }
}

/// Deterministic pseudo-random scalar in `[1, N)` from a seed (the same
/// splitmix64 expansion the phase-A witness corpus uses).
fn scalar(seed: u64) -> BigUint {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
    let mut bytes = [0u8; 32];
    for chunk in bytes.chunks_mut(8) {
        s ^= s >> 30;
        s = s.wrapping_mul(0xBF58_476D_1CE4_E5B9);
        s ^= s >> 27;
        s = s.wrapping_mul(0x94D0_49BB_1331_11EB);
        s ^= s >> 31;
        chunk.copy_from_slice(&s.to_le_bytes());
    }
    BigUint::from_bytes_le(&bytes) % (n() - 1u32) + 1u32
}

/// A pseudo-random on-curve point `k·G`, standing in for ecrecover's `R`.
fn random_point(seed: u64) -> AffinePoint {
    replay_double_and_add(&scalar(seed), &generator()).1
}

/// `CASES` witnesses at the ecrecover shape: `P1 = G`, `P2` a random point, both
/// scalars random in `[1, N)`.
fn corpus() -> Vec<Lincomb2Witness> {
    let g = generator();
    (0..CASES)
        .map(|i| {
            let i = i as u64;
            lincomb2_witness(
                &to_le_32(&scalar(2 * i + 1)),
                &to_le_32(&scalar(2 * i + 2)),
                &g,
                &random_point(5000 + i),
            )
            .expect("random lincomb2 corpus must be non-degenerate")
        })
        .collect()
}

/// Wraps a witness into the two chips' operation structs at timestamp `ts`.
fn ops_for(ts: u64, w: &Lincomb2Witness) -> (ecsm2::Ecsm2Operation, Vec<ecdas2::Ecdas2Operation>) {
    let ecsm2_op = ecsm2::Ecsm2Operation {
        timestamp: ts,
        addr_q: 0x1000,
        addr_p1: 0x2000,
        addr_p2: 0x3000,
        addr_u: 0x4000,
        status: 0,
        witness: Some(Box::new(w.clone())),
    };
    let ecdas2_ops = w
        .steps
        .iter()
        .cloned()
        .map(|step| ecdas2::Ecdas2Operation {
            timestamp: ts,
            step,
        })
        .collect();
    (ecsm2_op, ecdas2_ops)
}

/// Canonical `u64` of a Goldilocks trace cell.
fn canonical(x: &FieldElement<F>) -> u64 {
    F::canonical(x.value())
}

fn read_bytes(trace: &TraceTable<F, E>, row: usize, col: usize, len: usize) -> Vec<u8> {
    (0..len)
        .map(|i| {
            let v = canonical(trace.main_table.get(row, col + i));
            u8::try_from(v).expect("byte column out of range")
        })
        .collect()
}

// =============================================================================
// Layout and bus-shape pins
// =============================================================================

#[test]
fn ecsm2_layout_is_as_documented() {
    use ecsm2::cols as c;
    assert_eq!(c::TIMESTAMP_0, 0);
    assert_eq!(c::ADDR_Q_0, 2);
    assert_eq!(c::X_P2, 10);
    assert_eq!(c::Y_P2, 42);
    assert_eq!(c::MEM_X2, 74);
    assert_eq!(c::MEM_Q0, 106);
    assert_eq!(c::MEM_C0, 138);
    assert_eq!(c::MEM_Q1, 202);
    assert_eq!(c::MEM_C1, 235);
    assert_eq!(c::Y_P2_SUB_P, 299);
    assert_eq!(c::X_P12, 315);
    assert_eq!(c::U1, 379);
    assert_eq!(c::U2, 635);
    assert_eq!(c::U1_SUB_N, 891);
    assert_eq!(c::U2_SUB_N, 907);
    assert_eq!(c::LEN_M1, 923);
    assert_eq!(c::X_T0N, 924);
    assert_eq!(c::ACC_X, 988);
    assert_eq!(c::X_Q, 1052);
    assert_eq!(c::X_Q_SUB_P, 1116);
    assert_eq!(c::N1, 1148);
    assert_eq!(c::STATUS, 1151);
    assert_eq!(c::OK, 1153);
    assert_eq!(c::MU, 1154);
    assert_eq!(c::NUM_COLUMNS, 1155);
}

/// Every declared column block, as `(name, offset, width)`.
///
/// Kept as data so [`assert_layout_is_a_partition`] can check the whole block
/// map at once. Widths are restated here on purpose: this is a pin, and a pin
/// that derived its widths from the thing it pins would assert nothing.
fn ecdas2_blocks() -> Vec<(&'static str, usize, usize)> {
    use ecdas2::cols as c;
    vec![
        ("TIMESTAMP_0", c::TIMESTAMP_0, 1),
        ("TIMESTAMP_1", c::TIMESTAMP_1, 1),
        ("XB", c::XB, 32),
        ("YB", c::YB, 32),
        ("XA", c::XA, 32),
        ("YA", c::YA, 32),
        ("ROUND", c::ROUND, 1),
        ("OP", c::OP, 1),
        ("XR", c::XR, 32),
        ("YR", c::YR, 32),
        ("LAMBDA", c::LAMBDA, 32),
        ("Q0", c::Q0, 33),
        ("C0", c::C0, 64),
        ("Q1", c::Q1, 33),
        ("C1", c::C1, 64),
        ("Q2", c::Q2, 33),
        ("C2", c::C2, 64),
        ("NB", c::NB, 1),
        ("D1", c::D1, 1),
        ("D2", c::D2, 1),
        ("S1", c::S1, 1),
        ("S2", c::S2, 1),
        ("S3", c::S3, 1),
        ("S_CORR", c::S_CORR, 1),
        ("PH1", c::PH1, 1),
        ("PH2", c::PH2, 1),
        ("MU", c::MU, 1),
        ("D_INV", c::D_INV, 32),
        ("Q3", c::Q3, 33),
        ("C3", c::C3, 64),
    ]
}

/// The same for ECSM2.
fn ecsm2_blocks() -> Vec<(&'static str, usize, usize)> {
    use ecsm2::cols as c;
    vec![
        ("TIMESTAMP_0", c::TIMESTAMP_0, 1),
        ("TIMESTAMP_1", c::TIMESTAMP_1, 1),
        ("ADDR_Q_0", c::ADDR_Q_0, 1),
        ("ADDR_Q_1", c::ADDR_Q_1, 1),
        ("ADDR_P1_0", c::ADDR_P1_0, 1),
        ("ADDR_P1_1", c::ADDR_P1_1, 1),
        ("ADDR_P2_0", c::ADDR_P2_0, 1),
        ("ADDR_P2_1", c::ADDR_P2_1, 1),
        ("ADDR_U_0", c::ADDR_U_0, 1),
        ("ADDR_U_1", c::ADDR_U_1, 1),
        ("X_P2", c::X_P2, 32),
        ("Y_P2", c::Y_P2, 32),
        ("MEM_X2", c::MEM_X2, 32),
        ("MEM_Q0", c::MEM_Q0, 32),
        ("MEM_C0", c::MEM_C0, 64),
        ("MEM_Q1", c::MEM_Q1, 33),
        ("MEM_C1", c::MEM_C1, 64),
        ("Y_P2_SUB_P", c::Y_P2_SUB_P, 16),
        ("X_P12", c::X_P12, 32),
        ("Y_P12", c::Y_P12, 32),
        ("U1", c::U1, 256),
        ("U2", c::U2, 256),
        ("U1_SUB_N", c::U1_SUB_N, 16),
        ("U2_SUB_N", c::U2_SUB_N, 16),
        ("LEN_M1", c::LEN_M1, 1),
        ("X_T0N", c::X_T0N, 32),
        ("Y_T0N", c::Y_T0N, 32),
        ("ACC_X", c::ACC_X, 32),
        ("ACC_Y", c::ACC_Y, 32),
        ("X_Q", c::X_Q, 32),
        ("Y_Q", c::Y_Q, 32),
        ("X_Q_SUB_P", c::X_Q_SUB_P, 16),
        ("Y_Q_SUB_P", c::Y_Q_SUB_P, 16),
        ("N1", c::N1, 1),
        ("N2", c::N2, 1),
        ("N3", c::N3, 1),
        ("STATUS", c::STATUS, 1),
        ("S_INV", c::S_INV, 1),
        ("OK", c::OK, 1),
        ("MU", c::MU, 1),
    ]
}

/// Asserts the declared blocks tile `[0, num_columns)` exactly: nothing
/// overlaps, nothing is unclaimed, and the highest block ends at `NUM_COLUMNS`.
///
/// This is the check that catches the failure mode a hand-maintained `cols`
/// module actually has — a block added at the wrong offset, silently aliasing
/// another and corrupting the trace with no compile error. It caught nothing on
/// the day it was written, which is the point: it is cheap enough to keep
/// standing.
fn assert_layout_is_a_partition(
    chip: &str,
    blocks: &[(&'static str, usize, usize)],
    num_columns: usize,
) {
    let mut owner: Vec<Option<&str>> = vec![None; num_columns];
    for (name, offset, width) in blocks {
        assert!(
            offset + width <= num_columns,
            "{chip}: block {name} runs past NUM_COLUMNS ({num_columns})",
        );
        for (cell, slot) in owner[*offset..*offset + *width].iter_mut().enumerate() {
            assert!(
                slot.is_none(),
                "{chip}: column {} is claimed by both {} and {name}",
                offset + cell,
                slot.unwrap(),
            );
            *slot = Some(name);
        }
    }
    let unclaimed: Vec<usize> = (0..num_columns).filter(|c| owner[*c].is_none()).collect();
    assert!(
        unclaimed.is_empty(),
        "{chip}: {} column(s) belong to no block, first at {}",
        unclaimed.len(),
        unclaimed[0],
    );
}

#[test]
fn ecdas2_columns_tile_exactly() {
    assert_layout_is_a_partition("ECDAS2", &ecdas2_blocks(), ecdas2::cols::NUM_COLUMNS);
}

#[test]
fn ecsm2_columns_tile_exactly() {
    assert_layout_is_a_partition("ECSM2", &ecsm2_blocks(), ecsm2::cols::NUM_COLUMNS);
}

/// The worst case is 514 rows, not the ~471 a random corpus reaches — and the
/// chip's constant must say so, since capacity bounds are derived from it.
#[test]
fn worst_case_schedule_is_514_rows() {
    assert_eq!(ecdas2::MAX_ROWS_PER_EVALUATION, 514);

    // (2^255, 2^255 − 1): complementary bit patterns, so every round adds.
    let mut u1 = [0u8; 32];
    u1[31] = 0x80;
    let mut u2 = [0xFFu8; 32];
    u2[31] = 0x7F;

    let w = lincomb2_witness(&u1, &u2, &generator(), &random_point(1234))
        .expect("the worst-case scalars are valid");
    assert_eq!(w.len, 256);
    assert_eq!(
        w.steps.len(),
        ecdas2::MAX_ROWS_PER_EVALUATION,
        "1 precompute + 256 doublings + 256 adds + 1 correction",
    );
    let adds = w
        .steps
        .iter()
        .filter(|s| s.sel != JointSel::Double && s.sel != JointSel::Precompute)
        .count();
    assert_eq!(adds, 257, "256 main-chain adds + the correction row");

    // And it is genuinely the maximum: identical patterns share their adds.
    let all_ones = lincomb2_witness(&u2, &u2, &generator(), &random_point(1234)).unwrap();
    assert!(
        all_ones.steps.len() < w.steps.len(),
        "complementarity maximises, not popcount: got {} vs {}",
        all_ones.steps.len(),
        w.steps.len(),
    );
}

#[test]
fn ecdas2_layout_is_as_documented() {
    use ecdas2::cols as c;
    // The convolution core sits at the same offsets as the single-scalar chip,
    // which is what makes the relation body a rename rather than a rewrite.
    assert_eq!(c::XB, crate::tables::ecdas::cols::XG);
    assert_eq!(c::YB, crate::tables::ecdas::cols::YG);
    assert_eq!(c::XA, crate::tables::ecdas::cols::XA);
    assert_eq!(c::ROUND, crate::tables::ecdas::cols::ROUND);
    assert_eq!(c::OP, crate::tables::ecdas::cols::OP);
    assert_eq!(c::Q0, crate::tables::ecdas::cols::Q0);
    assert_eq!(c::C2, crate::tables::ecdas::cols::C2);
    // The joint bookkeeping block replaces NEXT_OP.
    assert_eq!(c::NB, 519);
    assert_eq!(c::D1, 520);
    assert_eq!(c::D2, 521);
    assert_eq!(c::S1, 522);
    assert_eq!(c::S_CORR, 525);
    assert_eq!(c::PH1, 526);
    assert_eq!(c::PH2, 527);
    assert_eq!(c::MU, 528);
    // The non-degeneracy block.
    assert_eq!(c::D_INV, 529);
    assert_eq!(c::Q3, 561);
    assert_eq!(c::C3, 594);
    assert_eq!(c::NUM_COLUMNS, 658);
}

#[test]
fn ecdas2_bus_interaction_shape() {
    let buses = ecdas2::bus_interactions();
    let count = |id: BusId, sender: bool| {
        buses
            .iter()
            .filter(|b| b.bus_id == id as u64 && b.is_sender == sender)
            .count()
    };
    assert_eq!(count(BusId::Ecdas, false), 1, "one chain receive");
    assert_eq!(count(BusId::Ecdas, true), 1, "one chain send");
    assert_eq!(count(BusId::Addend, false), 1, "one addend receive");
    assert_eq!(
        count(BusId::Addend, true),
        0,
        "ECDAS2 never publishes addends"
    );
    assert_eq!(count(BusId::JointBit, true), 2, "one send per digit stream");
    assert_eq!(
        count(BusId::AreBytes, true),
        8 * 16 + 2 + 1,
        "the paired ARE_BYTES layout: 8 blocks of 32, ROUND + three odd quotient \
         bytes paired, and Q3[32] alone",
    );
    assert_eq!(count(BusId::IsHalfword, true), 4 * 63, "4 x 63 carries");
    assert_eq!(buses.len(), 388);

    // Every ARE_BYTES / IS_HALF send must be MU-gated, and the only two
    // interactions that are NOT are the ones whose multiplicity is a live
    // count — the Addend receive and the two digit sends. A MU = 0 row emitting
    // on either of those is a forgery, which is why `(1 − MU)·D = 0` and
    // `(1 − MU)·S = 0` exist; this pins the set of interactions that argument
    // has to cover.
    let not_mu_gated: Vec<_> = buses
        .iter()
        .filter(|b| !matches!(b.multiplicity, Multiplicity::Column(c) if c == ecdas2::cols::MU))
        .map(|b| (b.bus_id, b.is_sender))
        .collect();
    assert_eq!(
        not_mu_gated,
        vec![
            (BusId::Addend as u64, false),
            (BusId::JointBit as u64, true),
            (BusId::JointBit as u64, true),
        ],
    );

    // The addend receive must be gated by the four selectors, not by MU: it has
    // to stay silent on doublings.
    let addend = buses
        .iter()
        .find(|b| b.bus_id == BusId::Addend as u64)
        .unwrap();
    match &addend.multiplicity {
        Multiplicity::Linear(terms) => {
            assert_eq!(terms.len(), 4, "S1 + S2 + S3 + S_CORR");
            let cols: Vec<usize> = terms
                .iter()
                .map(|t| match t {
                    LinearTerm::Column {
                        coefficient: 1,
                        column,
                    } => *column,
                    other => panic!("unexpected addend multiplicity term {other:?}"),
                })
                .collect();
            assert_eq!(
                cols,
                vec![
                    ecdas2::cols::S1,
                    ecdas2::cols::S2,
                    ecdas2::cols::S3,
                    ecdas2::cols::S_CORR
                ]
            );
        }
        other => panic!("addend multiplicity must be Linear, got {other:?}"),
    }

    // The joint chain tuple must lead with a NON-ZERO chain id. That constant is
    // the only thing separating it from the single-scalar chain on bus 28: a
    // zero would be skipped by the fingerprint and the two chains could alias.
    for bus in buses.iter().filter(|b| b.bus_id == BusId::Ecdas as u64) {
        match bus.values[0] {
            BusValue::Linear(ref terms) => match terms.as_slice() {
                [LinearTerm::Constant(v)] => {
                    assert_eq!(*v, ecdas2::JOINT_CHAIN_ID as i64)
                }
                other => panic!("chain id must be one constant, got {other:?}"),
            },
            ref other => panic!("chain id must be a constant, got {other:?}"),
        }
    }
    assert_ne!(ecdas2::JOINT_CHAIN_ID, 0);
}

#[test]
fn ecsm2_bus_interaction_shape() {
    let buses = ecsm2::bus_interactions();
    let count = |id: BusId, sender: bool| {
        buses
            .iter()
            .filter(|b| b.bus_id == id as u64 && b.is_sender == sender)
            .count()
    };
    assert_eq!(count(BusId::Ecall, false), 1);
    assert_eq!(
        count(BusId::Memw, true),
        1 + 3 + 8 + 8 + 8 + 8,
        "x10 read+write, three register reads, three operand reads, one result write",
    );
    assert_eq!(count(BusId::JointBit, false), 512, "256 bits x 2 streams");
    assert_eq!(count(BusId::Addend, true), 4, "P1, P2, P12, -2^len.T0");
    assert_eq!(count(BusId::EcT0, true), 1);
    assert_eq!(count(BusId::Ecdas, true), 3, "three segment seeds");
    assert_eq!(count(BusId::Ecdas, false), 3, "three segment drains");
    assert_eq!(count(BusId::AreBytes, true), 49);
    assert_eq!(count(BusId::IsHalfword, true), 63 + 63 + 5 * 16);
    assert_eq!(count(BusId::Zero, true), 2, "u1 != 0 and u2 != 0");

    // Only the ECALL receive and the status write may be MU-gated. Everything
    // else — in particular the reads at `a1`, which assert the constant G — must
    // be OK-gated, or the `P1 != G` error path becomes unprovable.
    let mu_gated: Vec<_> = buses
        .iter()
        .filter(|b| matches!(b.multiplicity, Multiplicity::Column(c) if c == ecsm2::cols::MU))
        .collect();
    assert_eq!(mu_gated.len(), 2);
    assert_eq!(mu_gated[0].bus_id, BusId::Ecall as u64);
    assert_eq!(mu_gated[1].bus_id, BusId::Memw as u64);

    // The EC_T0 send must carry a plain `len`, i.e. `LEN_M1 + 1`: the table's
    // receive key re-adds the 1 and its 256 unpadded rows are what bound `len`.
    let ec_t0 = buses
        .iter()
        .find(|b| b.bus_id == BusId::EcT0 as u64)
        .unwrap();
    match &ec_t0.values[0] {
        BusValue::Linear(terms) => {
            assert!(terms.iter().any(|t| matches!(
                t,
                LinearTerm::Column {
                    coefficient: 1,
                    column
                } if *column == ecsm2::cols::LEN_M1
            )));
            assert!(
                terms.iter().any(|t| matches!(t, LinearTerm::Constant(1))),
                "the key must add the +1 that turns LEN_M1 back into len",
            );
        }
        other => panic!("EC_T0 key must be linear, got {other:?}"),
    }

    // Digit receives are 2x the bit, because a set digit is carried by BOTH the
    // round's doubling and its add and both send.
    let jointbit: Vec<_> = buses
        .iter()
        .filter(|b| b.bus_id == BusId::JointBit as u64)
        .collect();
    for bus in &jointbit {
        match &bus.multiplicity {
            Multiplicity::Linear(terms) => assert!(matches!(
                terms.as_slice(),
                [LinearTerm::Column { coefficient: 2, .. }]
            )),
            other => panic!("JointBit multiplicity must be 2*bit, got {other:?}"),
        }
    }
}

// =============================================================================
// The corpus gate: constraints + result
// =============================================================================

/// Every emitted row of both chips satisfies every in-chip transition
/// constraint, over `CASES` random ecrecover-shaped evaluations, and the chip's
/// `Q` is the witness's `Q`.
#[test]
fn random_corpus_satisfies_every_constraint() {
    let ecsm2_air = busless_air(ecsm2::cols::NUM_COLUMNS, ecsm2::Ecsm2Constraints);
    let ecdas2_air = busless_air(ecdas2::cols::NUM_COLUMNS, ecdas2::Ecdas2Constraints);

    let mut rows_sum = 0usize;
    let mut rows_max = 0usize;
    for (i, w) in corpus().iter().enumerate() {
        let (ecsm2_op, ecdas2_ops) = ops_for(4 * i as u64 + 4, w);
        rows_sum += ecdas2_ops.len();
        rows_max = rows_max.max(ecdas2_ops.len());

        let ecsm2_trace = ecsm2::generate_ecsm2_trace(std::slice::from_ref(&ecsm2_op));
        let ecdas2_trace = ecdas2::generate_ecdas2_trace(&ecdas2_ops);

        assert!(
            validate_busless(&ecsm2_air, &ecsm2_trace),
            "case {i}: an ECSM2 transition constraint failed",
        );
        assert!(
            validate_busless(&ecdas2_air, &ecdas2_trace),
            "case {i}: an ECDAS2 transition constraint failed",
        );

        assert_eq!(
            read_bytes(&ecsm2_trace, 0, ecsm2::cols::X_Q, 32),
            w.x_q.to_vec(),
            "case {i}: chip xQ",
        );
        assert_eq!(
            read_bytes(&ecsm2_trace, 0, ecsm2::cols::Y_Q, 32),
            w.y_q.to_vec(),
            "case {i}: chip yQ",
        );
    }
    println!(
        "lincomb2 joint-chain rows: mean {:.1}, max {} over {CASES} cases",
        rows_sum as f64 / CASES as f64,
        rows_max,
    );
}

/// The non-degeneracy relation reuses `CARRY_OFFSET_XR`'s window instead of
/// carrying an offset of its own. That is a completeness claim — a carry outside
/// `[-offset, 2^16 − 1 − offset]` costs the *honest* prover a proof — and the
/// generator only checks it under `debug_assert`, so measure it here.
#[test]
fn the_non_degeneracy_carries_fit_the_reused_window() {
    let offset = ecdas2::CARRY_OFFSET_DINV;
    let (mut lo, mut hi) = (i64::MAX, i64::MIN);
    let mut rows = 0usize;
    for w in corpus() {
        for step in &w.steps {
            let d = ecdas2::dinv_witness(step);
            rows += 1;
            for c in d.c3 {
                lo = lo.min(c);
                hi = hi.max(c);
                assert!(
                    (0..1 << 16).contains(&(c + offset)),
                    "carry {c} escapes the window at offset {offset}",
                );
            }
        }
    }
    println!(
        "d_inv carries over {rows} rows: [{lo}, {hi}], window [{}, {}]",
        -offset,
        (1 << 16) - 1 - offset,
    );
}

/// A padding-only trace (no ops) must satisfy every constraint on both chips —
/// the argument that lets error rows zero out and still close at zero carries.
#[test]
fn padding_rows_satisfy_every_constraint() {
    let ecsm2_air = busless_air(ecsm2::cols::NUM_COLUMNS, ecsm2::Ecsm2Constraints);
    let ecdas2_air = busless_air(ecdas2::cols::NUM_COLUMNS, ecdas2::Ecdas2Constraints);
    assert!(validate_busless(
        &ecsm2_air,
        &ecsm2::generate_ecsm2_trace(&[])
    ));
    assert!(validate_busless(
        &ecdas2_air,
        &ecdas2::generate_ecdas2_trace(&[])
    ));
}

/// An error row (`status != 0`, no witness) satisfies every constraint: `OK = 0`
/// zeroes every relation, and the witnessed inverse binds the non-zero status.
#[test]
fn error_row_satisfies_every_constraint() {
    let air = busless_air(ecsm2::cols::NUM_COLUMNS, ecsm2::Ecsm2Constraints);
    for status in 1..=7u64 {
        let op = ecsm2::Ecsm2Operation {
            timestamp: 4,
            addr_q: 0x1000,
            addr_p1: 0x2000,
            addr_p2: 0x3000,
            addr_u: 0x4000,
            status,
            witness: None,
        };
        let trace = ecsm2::generate_ecsm2_trace(&[op]);
        assert!(
            validate_busless(&air, &trace),
            "error row with status {status} must satisfy every constraint",
        );
    }
}

// =============================================================================
// The corpus gate: bus balance
// =============================================================================

/// One row's contribution to a bus: the concrete element vector plus a signed
/// multiplicity.
fn row_contribution(
    interaction: &stark::lookup::BusInteraction,
    trace: &TraceTable<F, E>,
    row: usize,
) -> (Vec<u64>, i128) {
    let get = |col: usize| *trace.main_table.get(row, col);
    let mult = match &interaction.multiplicity {
        Multiplicity::Column(c) => canonical(&get(*c)) as i128,
        Multiplicity::Linear(terms) => terms
            .iter()
            .map(|t| match t {
                LinearTerm::Column {
                    coefficient,
                    column,
                } => *coefficient as i128 * canonical(&get(*column)) as i128,
                LinearTerm::ColumnUnsigned {
                    coefficient,
                    column,
                } => *coefficient as i128 * canonical(&get(*column)) as i128,
                LinearTerm::Constant(v) => *v as i128,
            })
            .sum(),
        other => panic!("unsupported multiplicity in the lincomb2 chips: {other:?}"),
    };
    let mut elements = vec![interaction.bus_id];
    for value in &interaction.values {
        for element in value.combine_from(get) {
            elements.push(canonical(&element));
        }
    }
    (elements, mult)
}

/// Accumulates every row of `trace` into `ledger`, signed by sender/receiver,
/// for the interactions whose bus is in `buses`.
fn accumulate(
    ledger: &mut HashMap<Vec<u64>, i128>,
    interactions: &[stark::lookup::BusInteraction],
    trace: &TraceTable<F, E>,
    buses: &[BusId],
) {
    let wanted: Vec<u64> = buses.iter().map(|b| *b as u64).collect();
    for interaction in interactions {
        if !wanted.contains(&interaction.bus_id) {
            continue;
        }
        for row in 0..trace.num_rows() {
            let (key, mult) = row_contribution(interaction, trace, row);
            if mult == 0 {
                continue;
            }
            let sign = if interaction.is_sender { 1 } else { -1 };
            *ledger.entry(key).or_insert(0) += sign * mult;
        }
    }
}

/// The four buses the joint chain owns balance exactly, over the whole corpus.
///
/// `Ecdas` (28) is shared with the single-scalar chain but the joint tuples lead
/// with a non-zero chain id, so nothing here can be matched by the old chips.
/// `EcT0` (32) is included with the real preprocessed table on the other side, so
/// this also checks the `LEN_M1 + 1` keying against the table's 256 rows.
#[test]
fn random_corpus_balances_the_joint_buses() {
    use crate::tables::ec_t0;

    let ecsm2_buses = ecsm2::bus_interactions();
    let ecdas2_buses = ecdas2::bus_interactions();
    let ec_t0_buses = ec_t0::bus_interactions();
    let tracked = [BusId::Ecdas, BusId::Addend, BusId::EcT0, BusId::JointBit];

    let witnesses = corpus();
    let mut ecsm2_ops = Vec::new();
    let mut ecdas2_ops = Vec::new();
    for (i, w) in witnesses.iter().enumerate() {
        let (op, rows) = ops_for(4 * i as u64 + 4, w);
        ecsm2_ops.push(op);
        ecdas2_ops.extend(rows);
    }

    let ecsm2_trace = ecsm2::generate_ecsm2_trace(&ecsm2_ops);
    let ecdas2_trace = ecdas2::generate_ecdas2_trace(&ecdas2_ops);
    let mut ec_t0_trace = ec_t0::generate_ec_t0_trace();
    ec_t0::update_multiplicities(&mut ec_t0_trace, witnesses.iter().map(|w| w.len));

    let mut ledger: HashMap<Vec<u64>, i128> = HashMap::new();
    accumulate(&mut ledger, &ecsm2_buses, &ecsm2_trace, &tracked);
    accumulate(&mut ledger, &ecdas2_buses, &ecdas2_trace, &tracked);
    accumulate(&mut ledger, &ec_t0_buses, &ec_t0_trace, &tracked);

    let unbalanced: Vec<_> = ledger.iter().filter(|(_, v)| **v != 0).collect();
    assert!(
        unbalanced.is_empty(),
        "{} unbalanced tuple(s); first: bus {} net {}",
        unbalanced.len(),
        unbalanced[0].0[0],
        unbalanced[0].1,
    );
    assert!(!ledger.is_empty(), "the ledger must not be trivially empty");
}

// =============================================================================
// Negative controls
// =============================================================================

/// Helper: build a single-case ECDAS2 trace and return it with its witness.
fn one_case_ecdas2() -> (Lincomb2Witness, TraceTable<F, E>) {
    let w = lincomb2_witness(
        &to_le_32(&scalar(11)),
        &to_le_32(&scalar(12)),
        &generator(),
        &random_point(77),
    )
    .unwrap();
    let (_, ops) = ops_for(4, &w);
    let trace = ecdas2::generate_ecdas2_trace(&ops);
    (w, trace)
}

/// `OP = S1 + S2 + S3 + S_CORR` is what keeps the Addend receive silent on
/// doublings. Setting a selector on a doubling must break a constraint — without
/// it the double-row addend cancellation is real but its gating is forgeable.
#[test]
fn selector_on_a_doubling_is_rejected() {
    let air = busless_air(ecdas2::cols::NUM_COLUMNS, ecdas2::Ecdas2Constraints);
    let (w, mut trace) = one_case_ecdas2();
    let double_row = w
        .steps
        .iter()
        .position(|s| s.sel == JointSel::Double)
        .expect("the corpus always contains a doubling");
    assert!(validate_busless(&air, &trace), "baseline must be valid");
    trace
        .main_table
        .set(double_row, ecdas2::cols::S2, FieldElement::<F>::one());
    assert!(
        !validate_busless(&air, &trace),
        "a selector set on a doubling must be rejected",
    );
}

/// The addend an add row consumes is pinned to the two digits it carries. Moving
/// `S1` to `S2` on a `(d1, d2) = (1, 0)` add must break a constraint — otherwise
/// the chain could add `P2` where the scalar says `P1`.
#[test]
fn addend_not_matching_the_digits_is_rejected() {
    let air = busless_air(ecdas2::cols::NUM_COLUMNS, ecdas2::Ecdas2Constraints);
    let (w, mut trace) = one_case_ecdas2();
    let add_p1 = w
        .steps
        .iter()
        .position(|s| s.sel == JointSel::AddP1)
        .expect("the corpus always contains a P1 add");
    trace
        .main_table
        .set(add_p1, ecdas2::cols::S1, FieldElement::<F>::zero());
    trace
        .main_table
        .set(add_p1, ecdas2::cols::S2, FieldElement::<F>::one());
    assert!(
        !validate_busless(&air, &trace),
        "an addend that disagrees with the digits must be rejected",
    );
}

/// The precompute row must add `P2`. Pointing it at `P1` makes the chord
/// `P1 + P1`, whose λ relation degenerates to `0 = 0` and would admit an
/// arbitrary "P12".
#[test]
fn precompute_pointed_at_p1_is_rejected() {
    let air = busless_air(ecdas2::cols::NUM_COLUMNS, ecdas2::Ecdas2Constraints);
    let (w, mut trace) = one_case_ecdas2();
    let pre = w
        .steps
        .iter()
        .position(|s| s.sel == JointSel::Precompute)
        .expect("row 0 is the precompute");
    trace
        .main_table
        .set(pre, ecdas2::cols::S2, FieldElement::<F>::zero());
    trace
        .main_table
        .set(pre, ecdas2::cols::S1, FieldElement::<F>::one());
    assert!(
        !validate_busless(&air, &trace),
        "the precompute row must be pinned to P2",
    );
}

/// Digits may only live on main-chain rows. A digit on the precompute or
/// correction row — both emitted at `round = 0` — would let a prover satisfy the
/// `2·u_bit(0)` receive with no round-0 add at all.
#[test]
fn digits_outside_the_main_chain_are_rejected() {
    let air = busless_air(ecdas2::cols::NUM_COLUMNS, ecdas2::Ecdas2Constraints);
    let (w, base) = one_case_ecdas2();
    for sel in [JointSel::Precompute, JointSel::Correction] {
        let row = w.steps.iter().position(|s| s.sel == sel).unwrap();
        let mut trace = base.clone();
        trace
            .main_table
            .set(row, ecdas2::cols::D1, FieldElement::<F>::one());
        assert!(
            !validate_busless(&air, &trace),
            "a digit on the {sel:?} row must be rejected",
        );
    }
}

/// A padding row (`MU = 0`) is inert on every bus *except* the per-stream digit
/// send, whose multiplicity is the raw `D1`/`D2` column. Without `(1 − MU)·D = 0`
/// a prover drops the real add at some round `r` (a doubling with `D1 = D2 = 0`
/// has `NB = 0`, so nothing forces the add to exist) and supplies the required
/// `2·u_bit(r)` JointBit count from two phantom padding rows — the chain then
/// computes `(u1 − 2^r)·P1 + u2·P2` while the proof claims `u1·P1 + u2·P2`, which
/// is an arbitrary chosen recovered key.
#[test]
fn a_digit_on_a_padding_row_is_rejected() {
    let air = busless_air(ecdas2::cols::NUM_COLUMNS, ecdas2::Ecdas2Constraints);
    let (w, base) = one_case_ecdas2();
    assert!(validate_busless(&air, &base), "baseline must be valid");
    let pad = w.steps.len();
    assert!(pad < base.num_rows(), "the case must leave padding rows");

    for digit in [ecdas2::cols::D1, ecdas2::cols::D2] {
        let mut trace = base.clone();
        let one = FieldElement::<F>::one();
        // The forgery shape, not a random poke: `PH1 = 1` satisfies
        // `(1 − PH1)·D = 0`, `NB = 1` satisfies the doubling's `NB = D1 ∨ D2`,
        // and every other column stays zero so all three convolution relations
        // still close at zero carries.
        trace.main_table.set(pad, ecdas2::cols::PH1, one);
        trace.main_table.set(pad, ecdas2::cols::NB, one);
        trace.main_table.set(pad, digit, one);
        trace
            .main_table
            .set(pad, ecdas2::cols::ROUND, FieldElement::<F>::from(5u64));
        assert!(
            !validate_busless(&air, &trace),
            "a live digit on a MU = 0 row must be rejected (column {digit})",
        );
    }
}

/// The same hole on the Addend receive: its multiplicity is `S1 + S2 + S3 +
/// S_CORR`, not `MU`. `MU = 0, OP = 1, S2 = 1` keeps `OP = ΣS` satisfied and
/// mints a spurious addend *publish* consumer out of a padding row.
#[test]
fn an_addend_receive_on_a_padding_row_is_rejected() {
    let air = busless_air(ecdas2::cols::NUM_COLUMNS, ecdas2::Ecdas2Constraints);
    let (w, base) = one_case_ecdas2();
    let pad = w.steps.len();
    assert!(pad < base.num_rows(), "the case must leave padding rows");

    for sel in [
        ecdas2::cols::S1,
        ecdas2::cols::S2,
        ecdas2::cols::S3,
        ecdas2::cols::S_CORR,
    ] {
        let mut trace = base.clone();
        let one = FieldElement::<F>::one();
        trace.main_table.set(pad, ecdas2::cols::OP, one);
        trace.main_table.set(pad, sel, one);
        assert!(
            !validate_busless(&air, &trace),
            "an addend selector on a MU = 0 row must be rejected (column {sel})",
        );
    }
}

/// Builds the degenerate add row `A + A` out of a real doubling row's witness.
///
/// With the addend set to the accumulator the chord λ is undefined, so a prover
/// may pick **any** λ; picking the doubling's own λ makes the row's other two
/// relations reuse that row's quotients and carries verbatim:
///
/// * λ relation, `op = 1`: `Σ λ_j(xB − xA)_{i−j} + (yA_i − yB_i)` is identically
///   zero, so `q0 = 3p` closes it at zero carries — this is exactly the `0 = 0`
///   collapse that leaves λ free.
/// * xR relation: with `xB = xA` the `op = 1` form `λ² − xA − xB − xR` is the
///   same integer expression as the doubling's `λ² − xA − 0 − xR − (xA − 0)`.
/// * yR relation never reads the addend at all.
fn degenerate_add_step(w: &Lincomb2Witness) -> ecsm::EcdasStep {
    let double = w
        .steps
        .iter()
        .find(|s| s.sel == JointSel::Double)
        .expect("the corpus always contains a doubling");
    let mut s = double.step.clone();
    s.x_g = s.x_a;
    s.y_g = s.y_a;
    s.op = 1;
    s.next_op = 0;
    s.q0 = ecsm::R_BYTES;
    s.c0 = [0; 64];
    s
}

/// An add row whose addend equals its accumulator must be rejected.
///
/// This is the NUMS-blinding forgery: the prover picks `P2 = μ·T₀`, which makes
/// `acc == addend` (with equal `y`) reachable at a chosen round by one modular
/// inversion and no discrete log. The λ relation then collapses to `0 = 0`, λ is
/// unconstrained, and the rest of the chain follows deterministically to a `Q`
/// that is not `u1·P1 + u2·P2`.
#[test]
fn a_degenerate_add_is_rejected() {
    let air = busless_air(ecdas2::cols::NUM_COLUMNS, ecdas2::Ecdas2Constraints);
    let (w, base) = one_case_ecdas2();
    assert!(validate_busless(&air, &base), "baseline must be valid");

    // `AddP2` with `(d1, d2) = (0, 1)` on the main chain: every structural
    // constraint (`OP = ΣS`, the digit/addend agreement, the phase rules) is
    // satisfied, so nothing but the non-degeneracy check can reject this row.
    let step = ecsm::witness::JointStep {
        step: degenerate_add_step(&w),
        sel: JointSel::AddP2,
        d1: 0,
        d2: 1,
        nb: 0,
    };
    let trace = ecdas2::generate_ecdas2_trace(&[ecdas2::Ecdas2Operation {
        timestamp: 4,
        step: step.clone(),
    }]);
    assert!(
        !validate_busless(&air, &trace),
        "an add row whose addend equals its accumulator must be rejected",
    );
}

/// The ablation control for [`a_degenerate_add_is_rejected`]: the same row,
/// checked against the *single-scalar* ECDAS chip — which shares ECDAS2's λ/xR/yR
/// core byte-for-byte at the same column offsets and has no non-degeneracy
/// relation. It **accepts**, which is what makes the test above a test of the new
/// relation rather than of a botched reconstruction.
#[test]
fn the_degenerate_add_passes_the_chip_without_the_non_degeneracy_check() {
    use crate::tables::ecdas;

    let air = busless_air(ecdas::cols::NUM_COLUMNS, ecdas::EcdasConstraints);
    let (w, _) = one_case_ecdas2();
    let trace = ecdas::generate_ecdas_trace(&[ecdas::EcdasOperation {
        timestamp: 4,
        step: degenerate_add_step(&w),
    }]);
    assert!(
        validate_busless(&air, &trace),
        "the convolution core alone must accept the degenerate add — otherwise \
         `a_degenerate_add_is_rejected` proves nothing about the new relation",
    );
}

/// `status == 0` must oblige the proof. A row that claims a zero status while
/// `OK = 0` (proving nothing) must be rejected, or the guest reads a fabricated
/// `Q` out of memory.
#[test]
fn zero_status_without_ok_is_rejected() {
    let air = busless_air(ecsm2::cols::NUM_COLUMNS, ecsm2::Ecsm2Constraints);
    let op = ecsm2::Ecsm2Operation {
        timestamp: 4,
        addr_q: 0x1000,
        addr_p1: 0x2000,
        addr_p2: 0x3000,
        addr_u: 0x4000,
        status: 3,
        witness: None,
    };
    let mut trace = ecsm2::generate_ecsm2_trace(&[op]);
    assert!(
        validate_busless(&air, &trace),
        "baseline error row is valid"
    );

    // Rewrite the status to 0 while leaving OK = 0. No choice of S_INV can
    // satisfy `STATUS·S_INV = 1 − OK`.
    trace
        .main_table
        .set(0, ecsm2::cols::STATUS, FieldElement::<F>::zero());
    assert!(
        !validate_busless(&air, &trace),
        "status 0 with OK = 0 must be rejected",
    );
}

/// The converse: `OK = 1` forces the status to 0.
#[test]
fn nonzero_status_with_ok_is_rejected() {
    let air = busless_air(ecsm2::cols::NUM_COLUMNS, ecsm2::Ecsm2Constraints);
    let w = lincomb2_witness(
        &to_le_32(&scalar(3)),
        &to_le_32(&scalar(4)),
        &generator(),
        &random_point(99),
    )
    .unwrap();
    let (op, _) = ops_for(4, &w);
    let mut trace = ecsm2::generate_ecsm2_trace(&[op]);
    assert!(validate_busless(&air, &trace), "baseline OK row is valid");
    trace
        .main_table
        .set(0, ecsm2::cols::STATUS, FieldElement::<F>::one());
    assert!(
        !validate_busless(&air, &trace),
        "a non-zero status with OK = 1 must be rejected",
    );
}

/// Tampering `yP2` breaks the curve-membership convolution.
#[test]
fn off_curve_p2_is_rejected() {
    let air = busless_air(ecsm2::cols::NUM_COLUMNS, ecsm2::Ecsm2Constraints);
    let w = lincomb2_witness(
        &to_le_32(&scalar(5)),
        &to_le_32(&scalar(6)),
        &generator(),
        &random_point(123),
    )
    .unwrap();
    let (op, _) = ops_for(4, &w);
    let mut trace = ecsm2::generate_ecsm2_trace(&[op]);
    let orig = *trace.main_table.get(0, ecsm2::cols::Y_P2);
    trace
        .main_table
        .set(0, ecsm2::cols::Y_P2, orig + FieldElement::<F>::one());
    assert!(
        !validate_busless(&air, &trace),
        "an off-curve P2 must be rejected",
    );
}

/// The correction addend must be the negated blind the table stores, never the
/// positive `2^len·T₀` the witness also records — the two differ only in `y`, so
/// mixing them is a silent sign flip.
#[test]
fn correction_addend_is_the_negated_blind() {
    use ecsm::p;
    for i in 0..8u64 {
        let w = lincomb2_witness(
            &to_le_32(&scalar(2 * i + 21)),
            &to_le_32(&scalar(2 * i + 22)),
            &generator(),
            &random_point(300 + i),
        )
        .unwrap();
        let (x, y) = ecsm2::correction_addend(&w);
        assert_eq!(x, w.x_t0_pow, "x is shared: x(-P) = x(P)");
        let y_pos = BigUint::from_bytes_le(&w.y_t0_pow);
        assert_eq!(
            BigUint::from_bytes_le(&y),
            p() - &y_pos,
            "y must be the NEGATED blind, not y_t0_pow",
        );
    }
}

/// The witnessed addend counts are exactly the number of Addend receives, per
/// selector — the precompute row counts towards `P2` because it genuinely adds
/// `P2`.
#[test]
fn addend_counts_match_the_emitted_rows() {
    for i in 0..8u64 {
        let w = lincomb2_witness(
            &to_le_32(&scalar(2 * i + 31)),
            &to_le_32(&scalar(2 * i + 32)),
            &generator(),
            &random_point(400 + i),
        )
        .unwrap();
        let (n1, n2, n3) = ecsm2::addend_counts(&w);
        let count =
            |f: &dyn Fn(JointSel) -> bool| w.steps.iter().filter(|s| f(s.sel)).count() as u64;
        assert_eq!(n1, count(&|s| s == JointSel::AddP1));
        assert_eq!(
            n2,
            count(&|s| s == JointSel::AddP2 || s == JointSel::Precompute),
        );
        assert_eq!(n3, count(&|s| s == JointSel::AddP12));
        assert_eq!(
            n1 + n2 + n3 + 1,
            count(&|s| s != JointSel::Double),
            "every non-doubling row receives exactly one addend (+1 correction)",
        );
    }
}
