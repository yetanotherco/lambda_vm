//! Adversarial check of the MUL `μ` bound (`fix/multiplicity-soundness`).
//!
//! The fix bounds `μ_lo`/`μ_hi` with `IS_HALF[μ]` weighted *by μ itself*
//! (`Multiplicity::Column`), because the skip-empty-tables invariant (#977)
//! forbids a constant `Multiplicity::One` on an optional table. The open
//! question (raised reviewing BRANCH): does weighting by μ leave a hole that
//! `One` would not? A malicious `μ_hi = −1` row's bound send lands at value
//! `p−1` with weight `p−1 = −1`; the proof then balances iff *something else*
//! supplies `+1` at `p−1` on the IS_HALFWORD bus.
//!
//! This test pins down exactly that, with the real fixed MUL AIR and real bus
//! balance. Every surrounding table is a faithful mock built by evaluating
//! MUL's own interactions: the CPU sends what MUL receives on ALU; RANGE /
//! RANGE20 receive what MUL sends to IS_HALFWORD / IS_B20, but *only for values
//! a real range table has a row for* (`< 2^16` / `< 2^20`). A `CARRIER` mock
//! optionally sends `IS_HALFWORD[p−1]`, standing in for "some other table emits
//! an out-of-range halfword with +1".
//!
//! Result characterises the fix:
//! - no carrier  → the `μ_hi = −1` forgery is REJECTED (the known attack is
//!   blocked; same as `multiplicity_forgery_poc`);
//! - with carrier → whether it verifies says whether the bound is unconditional
//!   or only holds while no table leaks an out-of-range halfword.

use std::collections::HashMap;

use crypto::fiat_shamir::default_transcript::DefaultTranscript;
use math::field::element::FieldElement;
use stark::constraints::builder::EmptyConstraints;
use stark::lookup::{
    AirWithBuses, AuxiliaryTraceBuildData, BusInteraction, BusValue, Multiplicity,
    NullBoundaryConstraintBuilder, Packing,
};
use stark::proof::options::ProofOptions;
use stark::trace::TraceTable;
use stark::traits::AIR;
use stark::verifier::{IsStarkVerifier, Verifier};

use crate::tables::mul::{MulOperation, bus_interactions, cols, generate_mul_trace};
use crate::tables::types::{BusId, FE, GoldilocksExtension, GoldilocksField};
use crate::test_utils::{create_mul_air, multi_prove_ram};

type F = GoldilocksField;
type E = GoldilocksExtension;
type MockAir = AirWithBuses<F, E, NullBoundaryConstraintBuilder, (), EmptyConstraints>;

const HALF: u64 = 1 << 16;
const B20: u64 = 1 << 20;
/// `p − 1`, i.e. the field value of `−1` — where a `μ = −1` bound send lands.
fn minus_one() -> u64 {
    (-FE::one()).canonical_u64()
}

fn eval_mult(m: &Multiplicity, g: &dyn Fn(usize) -> FE) -> FE {
    use stark::lookup::LinearTerm;
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

/// Net sends per `(bus, tuple)` of MUL's real interactions over `rows`.
fn mul_net(rows: &[Vec<FE>]) -> HashMap<(u64, Vec<u64>), FE> {
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

fn mock(opts: &ProofOptions, bus: BusId, n: usize, sender: bool) -> MockAir {
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

/// Forged MUL rows for `3 * 5 = 99`: row A (`μ_lo = 1, μ_hi = −1`, `lo` forged)
/// plus its honest `μ_lo = 0, μ_hi = 1` copy cancelling A on ALU-hi.
fn forged_mul_rows() -> Vec<Vec<FE>> {
    let honest = generate_mul_trace(&[(MulOperation::new(3, false, 5, false), false)]);
    let a = honest.main_table.get_row(0).to_vec();
    assert_eq!(a[cols::MU_LO], FE::one());
    assert_eq!(a[cols::MU_HI], FE::zero());
    assert_eq!(a[cols::LO_0], FE::from(15u64));

    let mut forged = a.clone();
    forged[cols::LO_0] = FE::from(99u64);
    forged[cols::MU_HI] = -FE::one();

    let mut copy = a.clone();
    copy[cols::MU_LO] = FE::zero();
    copy[cols::MU_HI] = FE::one();
    vec![forged, copy]
}

/// Proves CPU + MUL + RANGE(+RANGE20)(+CARRIER) and returns whether it verifies.
///
/// RANGE/RANGE20 receive MUL's IS_HALFWORD/IS_B20 demand only for values a real
/// table has a row for; `extra_half` lets a carrier send one out-of-range value.
fn prove_and_verify(rows: &[Vec<FE>], extra_half: Option<u64>) -> bool {
    let opts = ProofOptions::default_test_options();
    let alu: u64 = BusId::Alu.into();
    let half: u64 = BusId::IsHalfword.into();
    let b20: u64 = BusId::IsB20.into();

    let net = mul_net(rows);
    let mut cpu = Vec::new();
    let mut range = Vec::new();
    let mut range20 = Vec::new();
    for ((bus, tuple), m) in &net {
        if *m == FE::zero() {
            continue;
        }
        if *bus == alu {
            // MUL receives on ALU (net < 0); the CPU sends −net.
            cpu.push((tuple.clone(), -*m));
        } else if *bus == half {
            // MUL sends to IS_HALFWORD; RANGE receives it, if it is a halfword.
            if tuple[0] < HALF {
                range.push((tuple.clone(), *m));
            }
        } else if *bus == b20 && tuple[0] < B20 {
            range20.push((tuple.clone(), *m));
        }
    }
    let mut carrier = Vec::new();
    if let Some(v) = extra_half {
        carrier.push((vec![v], FE::one()));
    }

    let mul_len = rows.len().next_power_of_two().max(4);
    let mut mul_data = vec![FE::zero(); mul_len * cols::NUM_COLUMNS];
    for (r, row) in rows.iter().enumerate() {
        mul_data[r * cols::NUM_COLUMNS..(r + 1) * cols::NUM_COLUMNS].copy_from_slice(row);
    }
    let mut mul_trace = TraceTable::new_main(mul_data, cols::NUM_COLUMNS, 1);
    let mut cpu_trace = mock_trace(&cpu, 7);
    let mut range_trace = mock_trace(&range, 1);
    let mut range20_trace = mock_trace(&range20, 1);
    let mut carrier_trace = mock_trace(&carrier, 1);

    let mul_air = create_mul_air(&opts);
    let cpu_air = mock(&opts, BusId::Alu, 7, true);
    let range_air = mock(&opts, BusId::IsHalfword, 1, false);
    let range20_air = mock(&opts, BusId::IsB20, 1, false);
    let carrier_air = mock(&opts, BusId::IsHalfword, 1, true);

    let pairs: Vec<(
        &dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>,
        _,
        _,
    )> = vec![
        (&cpu_air, &mut cpu_trace, &()),
        (&mul_air, &mut mul_trace, &()),
        (&range_air, &mut range_trace, &()),
        (&range20_air, &mut range20_trace, &()),
        (&carrier_air, &mut carrier_trace, &()),
    ];
    let proof = multi_prove_ram(pairs, &mut DefaultTranscript::<E>::new(&[])).unwrap();
    let airs: Vec<&dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>> =
        vec![&cpu_air, &mul_air, &range_air, &range20_air, &carrier_air];
    Verifier::multi_verify(
        &airs,
        &proof,
        &mut DefaultTranscript::<E>::new(&[]),
        &FieldElement::zero(),
    )
}

/// Positive control: honest `3 * 5 = 15` answering the CPU verifies.
#[test]
fn control_honest_mul_verifies() {
    let honest = generate_mul_trace(&[(MulOperation::new(3, false, 5, false), false)]);
    let row = honest.main_table.get_row(0).to_vec();
    assert!(prove_and_verify(&[row], None));
}

/// The forgery leaves exactly a `−1` at value `p−1` on IS_HALFWORD, and nowhere
/// else out of range — so a single carrier `+1` at `p−1` is all it would need.
#[test]
fn forgery_leaves_only_minus_one_at_p_minus_1() {
    let half: u64 = BusId::IsHalfword.into();
    let net = mul_net(&forged_mul_rows());
    for ((bus, tuple), m) in &net {
        if *bus == half && tuple[0] >= HALF && *m != FE::zero() {
            assert_eq!(tuple[0], minus_one());
            assert_eq!(*m, -FE::one());
        }
    }
}

/// Known attack, no carrier: rejected (matches `multiplicity_forgery_poc`).
#[test]
fn forgery_without_carrier_is_rejected() {
    assert!(!prove_and_verify(&forged_mul_rows(), None));
}

/// The decisive case: does a carrier that supplies `+1` at `p−1` let the
/// `μ_hi = −1` product forgery through the weighted-by-μ bound?
#[test]
fn forgery_with_carrier_at_p_minus_1() {
    let verified = prove_and_verify(&forged_mul_rows(), Some(minus_one()));
    // Asserting the observed behaviour so the harness is not vacuous; the point
    // of the test is the printed verdict, interpreted in the module docs.
    println!("MUL forgery with p-1 carrier verified = {verified}");
    assert!(
        verified,
        "if this fails the weighted-by-μ bound resisted the carrier"
    );
}
