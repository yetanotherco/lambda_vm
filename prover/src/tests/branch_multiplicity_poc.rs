//! BRANCH's `μ` must be a bounded, non-negative multiplicity — same bug and fix
//! as LT (`lt_multiplicity_poc`).
//!
//! BRANCH weights its BRANCH receive and all range lookups (ARE_BYTES,
//! BYTE_ALU[AND], IS_HALFWORD ×3) by one free `μ`. A `μ = −1` twin of an honest
//! row cancels it on the BRANCH bus while *receiving* its range lookups; split
//! the high word non-canonically (`high_1 = 2^16, high_2 -= 1`, same packed
//! `next_pc`) and the twin absorbs an out-of-range `IS_HALFWORD[2^16]` sent by
//! another table. The fix range-checks `μ` (`IS_HALF[μ]`, weighted by μ), so the
//! twin's `μ = −1` itself lands at `p−1` with no receiver.
//!
//! Real BRANCH AIR + real bus balance; the surrounding tables are faithful mocks
//! built from BRANCH's own interactions (CPU sends what BRANCH receives; the
//! range buses receive what BRANCH sends, in-range only). A VICTIM mock sends an
//! out-of-range `IS_HALFWORD` value.

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

use crate::tables::branch::{BranchOperation, bus_interactions, cols, generate_branch_trace};
use crate::tables::types::{BusId, FE, GoldilocksExtension, GoldilocksField};
use crate::test_utils::{create_branch_air, multi_prove_ram};

type F = GoldilocksField;
type E = GoldilocksExtension;
type MockAir = AirWithBuses<F, E, NullBoundaryConstraintBuilder, (), EmptyConstraints>;
type Row = Vec<FE>;

const HALF: u64 = 1 << 16;

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

/// Net sends per `(bus, tuple)` of BRANCH's real interactions over `rows`.
fn branch_net(rows: &[Row]) -> HashMap<(u64, Vec<u64>), FE> {
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

fn mock(opts: &ProofOptions, bus: u64, n: usize, sender: bool) -> MockAir {
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

fn mock_trace(rows: &[(Vec<u64>, FE)], n: usize) -> TraceTable<F, E> {
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

/// An honest BRANCH row whose `next_pc = 2^48` (so `high_1 = 0, high_2 = 1`).
fn honest_row() -> Row {
    let t = generate_branch_trace(&[BranchOperation::new(1 << 48, 0, 0, false)]);
    let row = t.main_table.get_row(0).to_vec();
    assert_eq!(row[cols::MU], FE::one());
    assert_eq!(row[cols::NEXT_PC_HIGH_1], FE::zero());
    assert_eq!(row[cols::NEXT_PC_HIGH_2], FE::one());
    row
}

/// A `μ = −1` copy of the honest row with the high word split non-canonically
/// (`high_1 = 2^16, high_2 = 0`): same packed `next_pc`, so BRANCH and its
/// constraints agree, but `high_1` is out of range.
fn twin_row() -> Row {
    let mut row = honest_row();
    row[cols::MU] = -FE::one();
    row[cols::NEXT_PC_HIGH_1] = FE::from(HALF);
    row[cols::NEXT_PC_HIGH_2] = FE::zero();
    row
}

/// Proves CPU + BRANCH + range-bus mocks (+ VICTIM) and returns whether it
/// verifies. `victim` sends one out-of-range `IS_HALFWORD` value.
fn prove_and_verify(rows: &[Row], victim: Option<u64>) -> bool {
    let opts = ProofOptions::default_test_options();
    let branch_id: u64 = BusId::Branch.into();
    let half_id: u64 = BusId::IsHalfword.into();
    let widths: HashMap<u64, usize> = [
        (branch_id, 9usize),
        (BusId::AreBytes.into(), 2),
        (BusId::ByteAlu.into(), 4),
        (half_id, 1),
    ]
    .into_iter()
    .collect();

    let mut net = branch_net(rows);
    if let Some(v) = victim {
        *net.entry((half_id, vec![v])).or_insert(FE::zero()) += FE::one();
    }

    // One mock per bus. BRANCH is received by the chip, so the CPU mock sends it;
    // the range buses are sent by the chip, so their mocks receive. Out-of-range
    // IS_HALFWORD values get no receiver row (only the victim/μ bound reach them).
    let mut per_bus: HashMap<u64, Vec<(Vec<u64>, FE)>> = HashMap::new();
    for ((bus, tuple), m) in &net {
        if *m == FE::zero() {
            continue;
        }
        if *bus == half_id && tuple[0] >= HALF {
            continue;
        }
        per_bus.entry(*bus).or_default().push((tuple.clone(), *m));
    }

    let mut airs: Vec<MockAir> = Vec::new();
    let mut traces: Vec<TraceTable<F, E>> = Vec::new();
    let mut is_sender: Vec<bool> = Vec::new();
    for (&bus, rows) in &per_bus {
        let n = widths[&bus];
        let sender = bus == branch_id; // CPU sends BRANCH lookups; others receive.
        // BRANCH net is negative (chip receives); the sender supplies −net.
        let rows: Vec<(Vec<u64>, FE)> = rows
            .iter()
            .map(|(t, m)| (t.clone(), if sender { -*m } else { *m }))
            .collect();
        airs.push(mock(&opts, bus, n, sender));
        traces.push(mock_trace(&rows, n));
        is_sender.push(sender);
    }
    let _ = is_sender;
    if let Some(v) = victim {
        airs.push(mock(&opts, half_id, 1, true));
        traces.push(mock_trace(&[(vec![v], FE::one())], 1));
    }

    let branch_len = rows.len().next_power_of_two().max(4);
    let mut branch_data = vec![FE::zero(); branch_len * cols::NUM_COLUMNS];
    for (r, row) in rows.iter().enumerate() {
        branch_data[r * cols::NUM_COLUMNS..(r + 1) * cols::NUM_COLUMNS].copy_from_slice(row);
    }
    let mut branch_trace = TraceTable::new_main(branch_data, cols::NUM_COLUMNS, 1);
    let branch_air = create_branch_air(&opts);

    let mut pairs: Vec<(
        &dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>,
        _,
        _,
    )> = vec![(&branch_air, &mut branch_trace, &())];
    for (a, t) in airs.iter().zip(traces.iter_mut()) {
        pairs.push((a, t, &()));
    }
    let proof = multi_prove_ram(pairs, &mut DefaultTranscript::<E>::new(&[])).unwrap();

    let mut air_refs: Vec<&dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>> =
        vec![&branch_air];
    for a in &airs {
        air_refs.push(a);
    }
    Verifier::multi_verify(
        &air_refs,
        &proof,
        &mut DefaultTranscript::<E>::new(&[]),
        &FieldElement::zero(),
    )
}

/// Positive control: an honest BRANCH row answering the CPU verifies.
#[test]
fn control_honest_row_verifies() {
    assert!(prove_and_verify(&[honest_row()], None));
}

/// Control: an out-of-range IS_HALFWORD send with no twin has no receiver.
#[test]
fn control_out_of_range_victim_is_rejected() {
    assert!(!prove_and_verify(&[honest_row()], Some(HALF)));
}

/// Regression: the `μ = −1` twin can no longer absorb the victim's
/// IS_HALFWORD[2^16] (accepted before BRANCH's μ was bounded).
#[test]
fn regression_negative_mu_twin_cannot_absorb_victim() {
    assert!(!prove_and_verify(&[honest_row(), twin_row()], Some(HALF)));
}
