//! LT's `μ` must be a bounded, non-negative multiplicity (chip-level
//! regression; the end-to-end one is `multiplicity_forgery_poc::lt_*`).
//!
//! LT weights its ALU receive and all of its range lookups (2× MSB16,
//! 6× IS_HALFWORD) by the same `μ`. A row with `μ = −1` therefore acts as an ALU
//! *sender* and a range-check *receiver*. Paired with an honest `μ = +1` copy
//! carrying the same ALU tuple, the two cancel on the ALU bus, but their range
//! lookups do not have to match: the `−1` twin can hold `sub_0 = X` (any
//! `X < 2^32`) with `sub_1 = 0` and still satisfy the carry constraints, so one
//! IS_HALFWORD receive of X appears from nowhere. That absorbs an out-of-range
//! limb of *any* sender, another LT row included — which forges the LT result.
//! The fix range-checks μ (`IS_HALF[μ]`, weighted by μ), so a twin's `μ = −1`
//! itself has no receiver.
//!
//! Harness, with the real LT AIR (`lt::bus_interactions()` + `LtConstraints`):
//! - CPU: a mock ALU sender for the LT lookups the program "made".
//! - VICTIM: an optional mock IS_HALFWORD sender.
//! - RANGE / MSB16: stand-ins for the preprocessed BITWISE receivers, built
//!   honestly (exactly the halfwords 0..2^16). Their multiplicities are the net
//!   demand of every other table, *evaluated from the real interactions* — so a
//!   lookup added by a fix is tallied automatically, and a value outside the
//!   table can only balance if the other tables cancel it themselves.

use std::collections::HashMap;

use crypto::fiat_shamir::default_transcript::DefaultTranscript;
use math::field::element::FieldElement;
use stark::constraints::builder::EmptyConstraints;
use stark::lookup::{
    AirWithBuses, AuxiliaryTraceBuildData, BusInteraction, BusValue, LinearTerm, Multiplicity,
    NullBoundaryConstraintBuilder, Packing,
};
use stark::proof::options::ProofOptions;
use stark::trace::TraceTable;
use stark::traits::AIR;
use stark::verifier::{IsStarkVerifier, Verifier};

use crate::tables::lt::{LtOperation, bus_interactions, cols, generate_lt_trace};
use crate::tables::types::{BusId, FE, GoldilocksExtension, GoldilocksField, alu_op};
use crate::test_utils::{create_lt_air, multi_prove_ram};

type F = GoldilocksField;
type E = GoldilocksExtension;
type MockAir = AirWithBuses<F, E, NullBoundaryConstraintBuilder, (), EmptyConstraints>;
pub(crate) type Row = Vec<FE>;

const HALF: u64 = 1 << 16;

/// A mock table with one interaction: columns `0..n` are the tuple, column `n`
/// the multiplicity.
fn lookup_air(opts: &ProofOptions, bus: BusId, n: usize, sender: bool) -> MockAir {
    let values = (0..n)
        .map(|c| BusValue::Packed {
            start_column: c,
            packing: Packing::Direct,
        })
        .collect();
    let i = if sender {
        BusInteraction::sender(bus, Multiplicity::Column(n), values)
    } else {
        BusInteraction::receiver(bus, Multiplicity::Column(n), values)
    };
    AirWithBuses::new(
        n + 1,
        AuxiliaryTraceBuildData {
            interactions: vec![i],
        },
        opts,
        1,
        EmptyConstraints,
    )
}

/// Trace for `lookup_air`: one row per `(tuple, multiplicity)`, zero-padded.
fn lookup_trace(rows: &[(Vec<u64>, FE)], n: usize) -> TraceTable<F, E> {
    let len = rows.len().next_power_of_two().max(4);
    let mut data = vec![FE::zero(); len * (n + 1)];
    for (r, (tuple, m)) in rows.iter().enumerate() {
        for (c, v) in tuple.iter().enumerate() {
            data[r * (n + 1) + c] = FE::from(*v);
        }
        data[r * (n + 1) + n] = *m;
    }
    TraceTable::new_main(data, n + 1, 1)
}

fn eval_mult(m: &Multiplicity, g: &dyn Fn(usize) -> FE) -> FE {
    match m {
        Multiplicity::One => FE::one(),
        Multiplicity::Column(c) => g(*c),
        Multiplicity::Sum(a, b) => g(*a) + g(*b),
        Multiplicity::Negated(c) => FE::one() - g(*c),
        Multiplicity::Diff(a, b) => g(*a) - g(*b),
        Multiplicity::Sum3(a, b, c) => g(*a) + g(*b) + g(*c),
        Multiplicity::Linear(ts) => ts.iter().fold(FE::zero(), |acc, t| match *t {
            LinearTerm::Column {
                coefficient,
                column,
            } => acc + g(column) * FE::from(coefficient),
            LinearTerm::ColumnUnsigned {
                coefficient,
                column,
            } => acc + g(column) * FE::from(coefficient),
            LinearTerm::Constant(v) => acc + FE::from(v),
        }),
    }
}

/// Net sends per `(bus, tuple)` of LT's real interactions over `rows`.
pub(crate) fn lt_net(rows: &[Row]) -> HashMap<(u64, Vec<u64>), FE> {
    let mut net: HashMap<(u64, Vec<u64>), FE> = HashMap::new();
    for bi in bus_interactions() {
        for row in rows {
            let g = |c: usize| row[c];
            let m = eval_mult(&bi.multiplicity, &g);
            if m == FE::zero() {
                continue;
            }
            let tuple = bi
                .values
                .iter()
                .flat_map(|v| v.combine_from(g))
                .map(|x: FE| x.canonical_u64())
                .collect();
            let e = net.entry((bi.bus_id, tuple)).or_insert(FE::zero());
            *e = if bi.is_sender { *e + m } else { *e - m };
        }
    }
    net
}

/// The ALU tuple an LT lookup carries: `[lhs_lo, lhs_hi, rhs_lo, rhs_hi, flags, out, 0]`.
fn alu_tuple(op: &LtOperation, out: u64) -> Vec<u64> {
    let flags = alu_op::LT as u64 + 32 * op.signed as u64 + 64 * op.invert as u64;
    vec![
        op.lhs & 0xFFFF_FFFF,
        op.lhs >> 32,
        op.rhs & 0xFFFF_FFFF,
        op.rhs >> 32,
        flags,
        out,
        0,
    ]
}

/// The honest LT row for `op`.
pub(crate) fn honest_row(op: &LtOperation) -> Row {
    let t = generate_lt_trace(std::slice::from_ref(op));
    let row = t.main_table.get_row(0).to_vec();
    assert_eq!(row[cols::MU], FE::one());
    row
}

/// A `μ = −1` copy of `op`'s honest row whose `sub_0` is `x` (with `sub_1 = 0`,
/// so `sub_0 + 2^16·sub_1` and every carry are unchanged). Needs `lhs − rhs = x`
/// on the low word, `x < 2^32`.
pub(crate) fn twin_row(op: &LtOperation, x: u64) -> Row {
    let mut row = honest_row(op);
    assert_eq!(op.lhs.wrapping_sub(op.rhs) & 0xFFFF_FFFF, x);
    row[cols::MU] = -FE::one();
    row[cols::LHS_SUB_RHS_0] = FE::from(x);
    row[cols::LHS_SUB_RHS_1] = FE::zero();
    row
}

/// Proves CPU + LT + VICTIM + RANGE + MSB16 and returns whether it verifies.
///
/// `cpu`: the ALU lookups the CPU sends (once each). `victim`: IS_HALFWORD
/// values sent by another table. RANGE/MSB16 absorb the net demand for every
/// value they have a row for; anything else is left for the bus to reject.
fn prove_and_verify(lt_rows: &[Row], cpu: &[Vec<u64>], victim: &[u64]) -> bool {
    let opts = ProofOptions::default_test_options();
    let half_id: u64 = BusId::IsHalfword.into();
    let msb_id: u64 = BusId::Msb16.into();

    let mut net = lt_net(lt_rows);
    for &v in victim {
        *net.entry((half_id, vec![v])).or_insert(FE::zero()) += FE::one();
    }
    let range: Vec<(Vec<u64>, FE)> = (0..HALF)
        .map(|v| {
            let m = net.get(&(half_id, vec![v])).copied().unwrap_or(FE::zero());
            (vec![v], m)
        })
        .collect();
    let msb16: Vec<(Vec<u64>, FE)> = (0..HALF)
        .map(|v| {
            let t = vec![v, v >> 15];
            let m = net.get(&(msb_id, t.clone())).copied().unwrap_or(FE::zero());
            (t, m)
        })
        .collect();
    let cpu_rows: Vec<(Vec<u64>, FE)> = cpu.iter().map(|t| (t.clone(), FE::one())).collect();
    let victim_rows: Vec<(Vec<u64>, FE)> = victim.iter().map(|&v| (vec![v], FE::one())).collect();

    let lt_len = lt_rows.len().next_power_of_two().max(4);
    let mut lt_data = vec![FE::zero(); lt_len * cols::NUM_COLUMNS];
    for (r, row) in lt_rows.iter().enumerate() {
        lt_data[r * cols::NUM_COLUMNS..(r + 1) * cols::NUM_COLUMNS].copy_from_slice(row);
    }
    let mut lt_trace = TraceTable::new_main(lt_data, cols::NUM_COLUMNS, 1);
    let mut cpu_trace = lookup_trace(&cpu_rows, 7);
    let mut victim_trace = lookup_trace(&victim_rows, 1);
    let mut range_trace = lookup_trace(&range, 1);
    let mut msb16_trace = lookup_trace(&msb16, 2);

    let lt_air = create_lt_air(&opts);
    let cpu_air = lookup_air(&opts, BusId::Alu, 7, true);
    let victim_air = lookup_air(&opts, BusId::IsHalfword, 1, true);
    let range_air = lookup_air(&opts, BusId::IsHalfword, 1, false);
    let msb16_air = lookup_air(&opts, BusId::Msb16, 2, false);

    let pairs: Vec<(
        &dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>,
        _,
        _,
    )> = vec![
        (&cpu_air, &mut cpu_trace, &()),
        (&victim_air, &mut victim_trace, &()),
        (&lt_air, &mut lt_trace, &()),
        (&range_air, &mut range_trace, &()),
        (&msb16_air, &mut msb16_trace, &()),
    ];
    let proof = multi_prove_ram(pairs, &mut DefaultTranscript::<E>::new(&[])).unwrap();
    let airs: Vec<&dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>> =
        vec![&cpu_air, &victim_air, &lt_air, &range_air, &msb16_air];
    Verifier::multi_verify(
        &airs,
        &proof,
        &mut DefaultTranscript::<E>::new(&[]),
        &FieldElement::zero(),
    )
}

/// The honest LT row's out, as the CPU would receive it.
fn honest_out(op: &LtOperation) -> u64 {
    honest_row(op)[cols::OUT].canonical_u64()
}

/// Positive control: honest LT rows answering the CPU verify. Without this, a
/// "rejected" below would say nothing.
#[test]
fn control_honest_rows_verify() {
    let ops = [
        LtOperation::new(5, 3, false),
        LtOperation::new(HALF, 0, false),
    ];
    let rows: Vec<Row> = ops.iter().map(honest_row).collect();
    let cpu: Vec<_> = ops.iter().map(|op| alu_tuple(op, honest_out(op))).collect();
    assert!(prove_and_verify(&rows, &cpu, &[]));
}

/// Control: an out-of-range IS_HALFWORD send with no twin has no receiver.
#[test]
fn control_out_of_range_victim_is_rejected() {
    let op = LtOperation::new(HALF, 0, false);
    assert!(!prove_and_verify(
        &[honest_row(&op)],
        &[alu_tuple(&op, honest_out(&op))],
        &[HALF]
    ));
}

/// Control: a twin whose carry no longer holds is rejected by `LtConstraints`.
#[test]
fn control_twin_with_broken_carry_is_rejected() {
    let op = LtOperation::new(HALF, 0, false);
    let mut twin = twin_row(&op, HALF);
    twin[cols::LHS_SUB_RHS_1] = FE::one();
    assert!(!prove_and_verify(&[honest_row(&op), twin], &[], &[HALF]));
}

/// Regression: a `μ = −1` twin absorbing another table's out-of-range
/// IS_HALFWORD[2^16] (accepted before LT's μ was bounded).
#[test]
fn regression_negative_mu_twin_cannot_absorb_victim() {
    let op = LtOperation::new(HALF, 0, false);
    assert!(!prove_and_verify(
        &[honest_row(&op), twin_row(&op, HALF)],
        &[],
        &[HALF]
    ));
}

/// Regression: the victim can be LT itself — `5 < 3` claimed true (accepted
/// before LT's μ was bounded).
///
/// Row A (μ = 1) answers the CPU's `LT(5, 3) = 1` with `carry_1 = 1`: its high
/// `lhs − rhs` word becomes `2^32`, split as `sub_2 = 0, sub_3 = 2^16` — every
/// LT constraint holds, but `sub_3` is not a halfword. Twin C (`μ = −1`, op
/// `2^16 − 0`) absorbs IS_HALFWORD[2^16]; D, the honest `2^16 − 0` row, cancels
/// C on the ALU bus.
#[test]
fn regression_negative_mu_twin_cannot_forge_5_lt_3() {
    let op = LtOperation::new(5, 3, false);
    assert_eq!(honest_out(&op), 0);
    let mut a = honest_row(&op);
    a[cols::LT] = FE::one();
    a[cols::OUT] = FE::one();
    a[cols::LHS_SUB_RHS_2] = FE::zero();
    a[cols::LHS_SUB_RHS_3] = FE::from(HALF);

    let carrier = LtOperation::new(HALF, 0, false);
    let rows = [a, twin_row(&carrier, HALF), honest_row(&carrier)];
    assert!(!prove_and_verify(&rows, &[alu_tuple(&op, 1)], &[]));
}
