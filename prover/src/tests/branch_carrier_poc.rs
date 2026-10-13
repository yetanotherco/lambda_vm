//! The ARE_BYTES[μ, 0] bound is cancellable by a real BRANCH row.
//!
//! BRANCH sends `ARE_BYTES[next_pc_low[1], 0]`, but `next_pc_low[1]` only enters
//! its constraints and the BRANCH bus packed as `low0 + 2^8·low1 + 2^16·high0`.
//! An honest `(low1, high0) = (255, h)` can be stored as `(p−1, h+1)`: same packed
//! next_pc (−256 + 65536 = 255·256), every BranchConstraint holds, `IS_HALF[h+1]`
//! is in range, and the row sends `ARE_BYTES[p−1, 0]` with +1. That cancels the
//! μ bound of any `μ = −1` twin, which brings back the `5 < 3 = 1` forgery.
//!
//! Harness: real LT and BRANCH AIRs; mock CPU senders (ALU, BRANCH); mock
//! BITWISE receivers that only hold in-domain tuples (IS_HALFWORD < 2^16, MSB16,
//! ARE_BYTES bytes, BYTE_ALU AND), with multiplicities set to the net demand.

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

use crate::tables::branch::{self, BranchOperation, generate_branch_trace};
use crate::tables::lt::{self, LtOperation};
use crate::tables::types::{BusId, FE, GoldilocksExtension, GoldilocksField, alu_op};
use crate::test_utils::{create_branch_air, create_lt_air, multi_prove_ram};
use crate::tests::lt_multiplicity_poc::{Row, honest_row, twin_row};

type F = GoldilocksField;
type E = GoldilocksExtension;
type MockAir = AirWithBuses<F, E, NullBoundaryConstraintBuilder, (), EmptyConstraints>;
type Net = HashMap<(u64, Vec<u64>), FE>;

const HALF: u64 = 1 << 16;

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

/// Adds the net sends per `(bus, tuple)` of `interactions` over `rows` to `net`.
fn add_net(net: &mut Net, interactions: &[BusInteraction], rows: &[Row]) {
    for bi in interactions {
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
}

/// The single tuple `rows` put on `bus` (asserted unique).
fn tuple_on(interactions: &[BusInteraction], row: &Row, bus: BusId) -> Vec<u64> {
    let mut net = Net::new();
    add_net(&mut net, interactions, std::slice::from_ref(row));
    let id: u64 = bus.into();
    let hits: Vec<_> = net.keys().filter(|(b, _)| *b == id).collect();
    assert_eq!(hits.len(), 1);
    hits[0].1.clone()
}

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

fn honest_branch_row(op: &BranchOperation) -> Row {
    let t = generate_branch_trace(std::slice::from_ref(op));
    let row = t.main_table.get_row(0).to_vec();
    assert_eq!(row[branch::cols::MU], FE::one());
    row
}

/// `pc + 0xFF00` with `next_pc_low[1] = 255` re-encoded as `(p−1, high0 + 1)`.
fn branch_carrier() -> (Row, Row) {
    let op = BranchOperation::new(0, 0xFF00, 0, false);
    let honest = honest_branch_row(&op);
    let mut c = honest.clone();
    assert_eq!(c[branch::cols::NEXT_PC_LOW_1], FE::from(255u64));
    assert_eq!(c[branch::cols::NEXT_PC_HIGH_0], FE::zero());
    c[branch::cols::NEXT_PC_LOW_1] = -FE::one();
    c[branch::cols::NEXT_PC_HIGH_0] = FE::one();
    (honest, c)
}

fn in_domain(bus: u64, t: &[u64]) -> bool {
    let is = |b: BusId| bus == u64::from(b);
    if is(BusId::IsHalfword) {
        t.len() == 1 && t[0] < HALF
    } else if is(BusId::Msb16) {
        t.len() == 2 && t[0] < HALF && t[1] == t[0] >> 15
    } else if is(BusId::AreBytes) {
        t.len() == 2 && t[0] < 256 && t[1] < 256
    } else if is(BusId::ByteAlu) {
        t.len() == 4
            && t[0] == alu_op::AND as u64
            && t[1] < 256
            && t[2] < 256
            && t[3] == t[1] & t[2]
    } else {
        false
    }
}

/// `rows` padded with all-zero rows to the instance height, as the trace is.
/// Constant-multiplicity sends (the μ bounds) fire on padding rows too.
fn padded(rows: &[Row], ncols: usize) -> Vec<Row> {
    let mut v = rows.to_vec();
    v.resize(
        rows.len().next_power_of_two().max(4),
        vec![FE::zero(); ncols],
    );
    v
}

/// Proves CPU + LT + BRANCH + BITWISE mocks; returns whether it verifies.
fn prove_and_verify(
    lt_rows: &[Row],
    branch_rows: &[Row],
    cpu_alu: &[Vec<u64>],
    cpu_branch: &[Vec<u64>],
) -> bool {
    let opts = ProofOptions::default_test_options();
    let lt_rows = &padded(lt_rows, lt::cols::NUM_COLUMNS);
    let branch_rows = &padded(branch_rows, branch::cols::NUM_COLUMNS);
    let mut net = Net::new();
    add_net(&mut net, &lt::bus_interactions(), lt_rows);
    add_net(&mut net, &branch::bus_interactions(), branch_rows);

    let provider = |bus: BusId| -> Vec<(Vec<u64>, FE)> {
        let id: u64 = bus.into();
        net.iter()
            .filter(|((b, t), m)| *b == id && **m != FE::zero() && in_domain(id, t))
            .map(|((_, t), m)| (t.clone(), *m))
            .collect()
    };

    let table = |rows: &[Row], ncols: usize| {
        let len = rows.len().next_power_of_two().max(4);
        let mut data = vec![FE::zero(); len * ncols];
        for (r, row) in rows.iter().enumerate() {
            data[r * ncols..(r + 1) * ncols].copy_from_slice(row);
        }
        TraceTable::<F, E>::new_main(data, ncols, 1)
    };
    let sends = |ts: &[Vec<u64>]| -> Vec<(Vec<u64>, FE)> {
        ts.iter().map(|t| (t.clone(), FE::one())).collect()
    };
    let branch_width = cpu_branch.first().map_or(1, |t| t.len());

    let mut lt_trace = table(lt_rows, lt::cols::NUM_COLUMNS);
    let mut branch_trace = table(branch_rows, branch::cols::NUM_COLUMNS);
    let mut cpu_alu_trace = lookup_trace(&sends(cpu_alu), 7);
    let mut cpu_branch_trace = lookup_trace(&sends(cpu_branch), branch_width);
    let mut half_trace = lookup_trace(&provider(BusId::IsHalfword), 1);
    let mut msb_trace = lookup_trace(&provider(BusId::Msb16), 2);
    let mut bytes_trace = lookup_trace(&provider(BusId::AreBytes), 2);
    let mut balu_trace = lookup_trace(&provider(BusId::ByteAlu), 4);

    let lt_air = create_lt_air(&opts);
    let branch_air = create_branch_air(&opts);
    let cpu_alu_air = lookup_air(&opts, BusId::Alu, 7, true);
    let cpu_branch_air = lookup_air(&opts, BusId::Branch, branch_width, true);
    let half_air = lookup_air(&opts, BusId::IsHalfword, 1, false);
    let msb_air = lookup_air(&opts, BusId::Msb16, 2, false);
    let bytes_air = lookup_air(&opts, BusId::AreBytes, 2, false);
    let balu_air = lookup_air(&opts, BusId::ByteAlu, 4, false);

    let pairs: Vec<(
        &dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>,
        _,
        _,
    )> = vec![
        (&cpu_alu_air, &mut cpu_alu_trace, &()),
        (&cpu_branch_air, &mut cpu_branch_trace, &()),
        (&lt_air, &mut lt_trace, &()),
        (&branch_air, &mut branch_trace, &()),
        (&half_air, &mut half_trace, &()),
        (&msb_air, &mut msb_trace, &()),
        (&bytes_air, &mut bytes_trace, &()),
        (&balu_air, &mut balu_trace, &()),
    ];
    let Ok(proof) = multi_prove_ram(pairs, &mut DefaultTranscript::<E>::new(&[])) else {
        panic!("prover failed: the harness must produce a proof for the verifier to judge");
    };
    let airs: Vec<&dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>> = vec![
        &cpu_alu_air,
        &cpu_branch_air,
        &lt_air,
        &branch_air,
        &half_air,
        &msb_air,
        &bytes_air,
        &balu_air,
    ];
    Verifier::multi_verify(
        &airs,
        &proof,
        &mut DefaultTranscript::<E>::new(&[]),
        &FieldElement::zero(),
    )
}

/// LT rows A (forged `5 < 3 = 1`), T (`μ = −1` twin of `2^16 < 0`), D (honest copy of T).
fn forged_lt_rows() -> (LtOperation, Vec<Row>) {
    let op_a = LtOperation::new(5, 3, false);
    let mut a = honest_row(&op_a);
    a[lt::cols::LT] = FE::one();
    a[lt::cols::OUT] = FE::one();
    a[lt::cols::LHS_SUB_RHS_2] = FE::zero();
    a[lt::cols::LHS_SUB_RHS_3] = FE::from(HALF);
    let base = LtOperation::new(HALF, 0, false);
    (op_a, vec![a, twin_row(&base, HALF), honest_row(&base)])
}

/// Control: honest LT + honest BRANCH verify, so a rejection below means something.
#[test]
fn control_honest_lt_and_branch_verify() {
    let op = LtOperation::new(5, 3, false);
    let (b, _) = branch_carrier();
    let cpu_branch = tuple_on(&branch::bus_interactions(), &b, BusId::Branch);
    assert!(prove_and_verify(
        &[honest_row(&op)],
        &[b],
        &[alu_tuple(&op, 0)],
        &[cpu_branch]
    ));
}

/// Control: the carrier row alone is rejected (its ARE_BYTES[p−1, 0] has no receiver).
#[test]
fn control_branch_carrier_alone_is_rejected() {
    let op = LtOperation::new(5, 3, false);
    let (honest, c) = branch_carrier();
    let cpu_branch = tuple_on(&branch::bus_interactions(), &honest, BusId::Branch);
    assert_eq!(
        cpu_branch,
        tuple_on(&branch::bus_interactions(), &c, BusId::Branch)
    );
    assert!(!prove_and_verify(
        &[honest_row(&op)],
        &[c],
        &[alu_tuple(&op, 0)],
        &[cpu_branch]
    ));
}

/// `5 < 3 = 1` with a real BRANCH carrier must be rejected. FAILS on 2c1335db5:
/// the carrier's ARE_BYTES[p−1, 0] (+1) cancels T's bound send (−1).
#[test]
fn lt_forgery_with_branch_carrier_is_rejected() {
    let (op_a, lt_rows) = forged_lt_rows();
    let (honest, c) = branch_carrier();
    let cpu_branch = tuple_on(&branch::bus_interactions(), &honest, BusId::Branch);
    let verified = prove_and_verify(&lt_rows, &[c], &[alu_tuple(&op_a, 1)], &[cpu_branch]);
    assert!(
        !verified,
        "verifier accepted 5 < 3 = 1 via a BRANCH carrier on ARE_BYTES"
    );
}

/// BRANCH row for `pc = 0, offset`, re-encoded with `next_pc_low_1 = honest − 256`
/// (i.e. `p − (256 − honest)`) and `next_pc_high_0 + 1`, at multiplicity `mu`.
/// The packed next_pc is unchanged, so every BranchConstraint still holds.
fn branch_reencoded(offset: u64, honest_low1: u64, mu: FE) -> (Row, Row) {
    let op = BranchOperation::new(0, offset, 0, false);
    let honest = honest_branch_row(&op);
    assert_eq!(honest[branch::cols::NEXT_PC_LOW_1], FE::from(honest_low1));
    assert_eq!(honest[branch::cols::NEXT_PC_HIGH_0], FE::zero());
    let mut r = honest.clone();
    r[branch::cols::NEXT_PC_LOW_1] = FE::from(honest_low1) - FE::from(256u64);
    r[branch::cols::NEXT_PC_HIGH_0] = FE::one();
    r[branch::cols::MU] = mu;
    (honest, r)
}

/// Weight 1 on the μ bound alone is not enough while BRANCH also sends
/// `next_pc_low_1` with weight μ (spec #1017 before 256e3970b). The LT twin's
/// bound leaves `+1` on `(p−1, 0)`, and two BRANCH rows with μ < 0 cancel it:
/// - B: μ = −2, `low1 = p−1`: bound `(p−2, 0)` +1, limb `(p−1, 0)` −2;
/// - C: μ = −1, `low1 = p−2`: bound `(p−1, 0)` +1, limb `(p−2, 0)` −1.
///
/// Both tuples net to zero. Honest copies of B and C at μ = 2 and μ = 1 cancel
/// them on the BRANCH bus, and everything else lands in range. Batching the two
/// byte checks into one `ARE_BYTES[μ, next_pc_low_1]` with multiplicity 1
/// leaves B and C each with an uncancellable `+1` on an out-of-range tuple.
#[test]
fn lt_forgery_with_branch_cycle_is_rejected() {
    let (op_a, lt_rows) = forged_lt_rows();
    let (honest_b, b) = branch_reencoded(0xFF00, 255, -FE::from(2u64));
    let (honest_c, c) = branch_reencoded(0xFE00, 254, -FE::one());
    let mut b_copy = honest_b.clone();
    b_copy[branch::cols::MU] = FE::from(2u64);
    // One honest BRANCH lookup the CPU actually makes, so the BRANCH bus is live.
    let cpu_branch = tuple_on(&branch::bus_interactions(), &honest_b, BusId::Branch);
    let branch_rows = [honest_b, b, b_copy, c, honest_c];
    let verified = prove_and_verify(
        &lt_rows,
        &branch_rows,
        &[alu_tuple(&op_a, 1)],
        &[cpu_branch],
    );
    assert!(
        !verified,
        "verifier accepted 5 < 3 = 1 via a two-row BRANCH cycle on ARE_BYTES"
    );
}

/// Guard for the invariant both forgeries above exploit. The μ bounds hold only
/// while no ARE_BYTES sender carries a negative weight, and in a dedup table μ is
/// a count a cheating prover may set to any field element. So every ARE_BYTES send
/// of a dedup table must have the constant multiplicity 1, and together they must
/// cover each of its μ columns.
#[test]
fn dedup_tables_bound_every_mu_with_constant_are_bytes_sends() {
    use crate::tables::{bytewise, dvrm, mul};
    let tables: [(&str, Vec<BusInteraction>, Vec<usize>); 5] = [
        ("LT", lt::bus_interactions(), vec![lt::cols::MU]),
        ("BRANCH", branch::bus_interactions(), vec![branch::cols::MU]),
        (
            "BYTEWISE",
            bytewise::bus_interactions(),
            vec![bytewise::cols::MU],
        ),
        (
            "MUL",
            mul::bus_interactions(),
            vec![mul::cols::MU_LO, mul::cols::MU_HI],
        ),
        (
            "DVRM",
            dvrm::bus_interactions(),
            vec![dvrm::cols::MU_Q, dvrm::cols::MU_R],
        ),
    ];
    for (name, interactions, mus) in tables {
        let mut checked = Vec::new();
        for i in interactions
            .iter()
            .filter(|i| i.bus_id == u64::from(BusId::AreBytes))
        {
            assert!(
                i.is_sender && matches!(i.multiplicity, Multiplicity::One),
                "{name} sends ARE_BYTES with a non-constant multiplicity: a μ < 0 row \
                 could cancel a μ bound"
            );
            for v in &i.values {
                if let BusValue::Packed { start_column, .. } = v {
                    checked.push(*start_column);
                }
            }
        }
        for mu in mus {
            assert!(
                checked.contains(&mu),
                "{name} does not range-check μ column {mu} on ARE_BYTES"
            );
        }
    }
}
