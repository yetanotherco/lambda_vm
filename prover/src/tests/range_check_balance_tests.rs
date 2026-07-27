//! Direct balance check on the **range-check buses** (`AreBytes`, `IsHalfword`,
//! `IsB20`, `Msb8`, `Msb16`, `Zero`, `Hwsl`).
//!
//! # Why this exists
//!
//! Every chip that range-checks a column sends on one of these buses, and the
//! BITWISE table's receive multiplicities come from a *separate* hand-written
//! collector (`collect_bitwise_from_*` in `trace_builder.rs`). The two must
//! agree send-for-send; several of those collectors carry a comment saying so.
//! Nothing enforced it:
//!
//! * the `debug-checks` bus tracker reports buses 14..33 only — the range-check
//!   buses are **not** in its report at all;
//! * a mismatch surfaces as a whole-proof "LogUp bus does not balance" with no
//!   indication of which bus, which table, or which tuple.
//!
//! This test closes that. It evaluates every table's declared interactions
//! against its generated trace and asserts the signed multiplicities cancel per
//! tuple, so a drifted mirror names the offending bus and value immediately.
//!
//! It is cheap (a small guest, well under a second) and it is a pure
//! cross-check: it re-derives nothing, it only reads what the chips declare.

use std::collections::HashMap;

use math::field::element::FieldElement;
use math::field::traits::IsPrimeField;
use stark::lookup::{BusInteraction, LinearTerm, Multiplicity};
use stark::trace::TraceTable;

use crate::tables::trace_builder::Traces;
use crate::tables::types::{BusId, GoldilocksExtension, GoldilocksField};

use executor::elf::Elf;

type F = GoldilocksField;
type E = GoldilocksExtension;

/// The buses the BITWISE table serves. These are exactly the ones the
/// `debug-checks` tracker leaves out.
const RANGE_CHECK_BUSES: [BusId; 7] = [
    BusId::AreBytes,
    BusId::IsHalfword,
    BusId::IsB20,
    BusId::Msb8,
    BusId::Msb16,
    BusId::Zero,
    BusId::Hwsl,
];

fn canonical(x: &FieldElement<F>) -> u64 {
    F::canonical(x.value())
}

/// Accumulates one table's range-check interactions into `ledger`, signed by
/// sender/receiver.
fn accumulate(
    ledger: &mut HashMap<Vec<u64>, i128>,
    interactions: &[BusInteraction],
    trace: &TraceTable<F, E>,
) {
    let wanted: Vec<u64> = RANGE_CHECK_BUSES.iter().map(|b| *b as u64).collect();
    for interaction in interactions {
        if !wanted.contains(&interaction.bus_id) {
            continue;
        }
        for row in 0..trace.num_rows() {
            let get = |col: usize| *trace.main_table.get(row, col);
            let value = |col: usize| canonical(&get(col)) as i128;
            let multiplicity: i128 = match &interaction.multiplicity {
                Multiplicity::One => 1,
                Multiplicity::Column(c) => value(*c),
                Multiplicity::Negated(c) => 1 - value(*c),
                Multiplicity::Sum(a, b) => value(*a) + value(*b),
                Multiplicity::Diff(a, b) => value(*a) - value(*b),
                Multiplicity::Sum3(a, b, c) => value(*a) + value(*b) + value(*c),
                Multiplicity::Linear(terms) => terms
                    .iter()
                    .map(|term| match term {
                        LinearTerm::Column {
                            coefficient,
                            column,
                        } => *coefficient as i128 * value(*column),
                        LinearTerm::ColumnUnsigned {
                            coefficient,
                            column,
                        } => *coefficient as i128 * value(*column),
                        LinearTerm::Constant(v) => *v as i128,
                    })
                    .sum(),
            };
            if multiplicity == 0 {
                continue;
            }
            let mut key = vec![interaction.bus_id];
            for bus_value in &interaction.values {
                for element in bus_value.combine_from(get) {
                    key.push(canonical(&element));
                }
            }
            let sign = if interaction.is_sender { 1 } else { -1 };
            *ledger.entry(key).or_insert(0) += sign * multiplicity;
        }
    }
}

/// Every range-check bus balances tuple-for-tuple across every table.
///
/// The guest is the lincomb2 one because that is the newest set of mirrors, but
/// the check is whole-machine: it covers all 30-odd tables, so it also guards
/// the pre-existing collectors against drift.
#[test]
fn range_check_buses_balance_across_every_table() {
    let _ = env_logger::builder().is_test(true).try_init();

    let elf_bytes = crate::test_utils::asm_elf_bytes("test_ecsm_lincomb2");
    let elf = Elf::load(&elf_bytes).expect("Failed to load ELF");
    let executor =
        executor::vm::execution::Executor::new(&elf, vec![]).expect("Failed to create executor");
    let result = executor.run().expect("Failed to run program");
    let traces =
        Traces::from_elf_and_logs_minimal(&elf, &result.logs, &Default::default(), &[]).unwrap();

    let mut ledger: HashMap<Vec<u64>, i128> = HashMap::new();

    macro_rules! table {
        ($module:path, $trace:expr) => {{
            use $module as m;
            accumulate(&mut ledger, &m::bus_interactions(), $trace);
        }};
    }
    macro_rules! tables {
        ($module:path, $traces:expr) => {{
            for t in $traces {
                table!($module, t);
            }
        }};
    }

    table!(crate::tables::bitwise, &traces.bitwise);
    table!(crate::tables::decode, &traces.decode);
    table!(crate::tables::register, &traces.register);
    table!(crate::tables::halt, &traces.halt);
    table!(crate::tables::commit, &traces.commit);
    table!(crate::tables::keccak, &traces.keccak);
    table!(crate::tables::keccak_rnd, &traces.keccak_rnd);
    table!(crate::tables::keccak_rc, &traces.keccak_rc);
    table!(crate::tables::ecsm, &traces.ecsm);
    table!(crate::tables::ecdas, &traces.ecdas);
    table!(crate::tables::ecsm2, &traces.ecsm2);
    table!(crate::tables::ecdas2, &traces.ecdas2);
    table!(crate::tables::ec_t0, &traces.ec_t0);
    tables!(crate::tables::cpu, &traces.cpus);
    tables!(crate::tables::cpu32, &traces.cpu32s);
    tables!(crate::tables::lt, &traces.lts);
    tables!(crate::tables::shift, &traces.shifts);
    tables!(crate::tables::mul, &traces.muls);
    tables!(crate::tables::dvrm, &traces.dvrms);
    tables!(crate::tables::branch, &traces.branches);
    tables!(crate::tables::load, &traces.loads);
    tables!(crate::tables::memw, &traces.memws);
    tables!(crate::tables::memw_aligned, &traces.memw_aligneds);
    tables!(crate::tables::memw_register, &traces.memw_registers);
    tables!(crate::tables::eq, &traces.eqs);
    tables!(crate::tables::bytewise, &traces.bytewises);
    tables!(crate::tables::store, &traces.stores);
    for (trace, config) in traces.pages.iter().zip(traces.page_configs.iter()) {
        accumulate(
            &mut ledger,
            &crate::tables::page::bus_interactions(config.page_base),
            trace,
        );
    }

    assert!(
        !ledger.is_empty(),
        "the ledger must not be trivially empty — no range-check sends were seen"
    );
    let mut unbalanced: Vec<_> = ledger.iter().filter(|(_, net)| **net != 0).collect();
    unbalanced.sort();
    for (key, net) in unbalanced.iter().take(10) {
        let bus = BusId::try_from(key[0]).map(|b| b.name()).unwrap_or("?");
        eprintln!("  {bus}{:?} net {net}", &key[1..]);
    }
    assert!(
        unbalanced.is_empty(),
        "{} range-check tuple(s) do not balance — a `collect_bitwise_from_*` \
         mirror has drifted from its chip's `bus_interactions()`",
        unbalanced.len(),
    );
}
