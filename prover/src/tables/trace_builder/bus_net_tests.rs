//! ★ The buses the tables close among themselves balance on the traces the
//! builders make, row by row, with no proof: every tuple a table sends is
//! received as often, padding rows included. The multiplicities BITWISE answers
//! with are counted by collectors that mirror each table's rows (the μ bounds
//! the deduplicating tables send once a row, the byte checks SHIFT's and
//! KECCAK_RND's padding rows send); a collector that miscounts leaves a tuple
//! here that only a proof would otherwise find.
//!
//! The buses a block's statement closes from outside the tables (the committed
//! output, memory's and registers' first and last values) are left out.

use std::collections::{BTreeMap, HashMap};

use executor::elf::Elf;
use stark::lookup::{BusInteraction, LinearTerm, Multiplicity};
use stark::proof::options::ProofOptions;

use super::finish_tests::{run, windowed};
use super::*;
use crate::VmAirs;
use crate::tables::MaxRowsConfig;
use crate::tables::types::{BusId, FE};

/// The buses a statement closes from outside the tables: a net on them is the
/// verifier's to cancel, not a miscount.
const OUTSIDE: [BusId; 4] = [
    BusId::Commit,
    BusId::Memory,
    BusId::GlobalMemory,
    BusId::Decode,
];

fn multiplicity(m: &Multiplicity, g: &dyn Fn(usize) -> FE) -> FE {
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

/// Every bus's tuples with a nonzero net over all of `traces`' tables, by bus,
/// the buses in [`OUTSIDE`] left out: `(bus, tuples)`.
fn unbalanced(program: &Elf, traces: &mut Traces) -> BTreeMap<u64, Vec<(Vec<u64>, i128)>> {
    let options = ProofOptions::default_test_options();
    let table_counts = traces.table_counts();
    let airs = VmAirs::new(
        program,
        &options,
        false,
        &traces.page_configs,
        &table_counts,
        &crate::cpu_heights_from_traces(traces),
        None,
        true,
        None,
        crate::test_utils::halted_register_commitment(
            &options,
            program,
            &traces.public_output_bytes,
        )
        .preprocessed(),
    );
    let outside: Vec<u64> = OUTSIDE.iter().map(|b| u64::from(*b)).collect();
    let mut net: HashMap<(u64, Vec<u64>), FE> = HashMap::new();
    // Each table's share of each tuple, to name the tables of a tuple left over.
    let mut by_table: HashMap<(u64, Vec<u64>), BTreeMap<String, i128>> = HashMap::new();
    for (air, trace, _) in airs.air_trace_pairs(traces) {
        let interactions: &[BusInteraction] = air.bus_interactions();
        for row in 0..trace.num_rows() {
            let g = |c: usize| *trace.get_main(row, c);
            for bi in interactions
                .iter()
                .filter(|bi| !outside.contains(&bi.bus_id))
            {
                let m = multiplicity(&bi.multiplicity, &g);
                if m == FE::zero() {
                    continue;
                }
                let tuple: Vec<u64> = bi
                    .values
                    .iter()
                    .flat_map(|v| v.combine_from(g))
                    .map(|x: FE| x.canonical_u64())
                    .collect();
                let family = air.name().split('[').next().unwrap_or("").to_string();
                let share = by_table.entry((bi.bus_id, tuple.clone())).or_default();
                let small = m.canonical_u64().min(1 << 20) as i128;
                *share.entry(family).or_insert(0) += if bi.is_sender { small } else { -small };
                let e = net.entry((bi.bus_id, tuple)).or_insert(FE::zero());
                *e = if bi.is_sender { *e + m } else { *e - m };
            }
        }
    }
    let mut by_bus: BTreeMap<u64, Vec<(Vec<u64>, i128)>> = BTreeMap::new();
    for ((bus, tuple), v) in net {
        if v != FE::zero() {
            // The net as a small signed count: a send left over is positive.
            let v = v.canonical_u64();
            let signed = if v > u64::MAX / 2 {
                -((FE::zero() - FE::from(v)).canonical_u64() as i128)
            } else {
                v as i128
            };
            let tables = by_table
                .get(&(bus, tuple.clone()))
                .map(|shares| {
                    shares
                        .iter()
                        .filter(|(_, n)| **n != 0)
                        .map(|(t, n)| format!("{t} {n:+}"))
                        .collect::<Vec<_>>()
                        .join(" ")
                })
                .unwrap_or_default();
            eprintln!("left over: bus {bus} {tuple:?} net {signed:+} ({tables})");
            by_bus.entry(bus).or_default().push((tuple, signed));
        }
    }
    by_bus
}

fn named(by_bus: &BTreeMap<u64, Vec<(Vec<u64>, i128)>>) -> String {
    by_bus
        .iter()
        .map(|(bus, tuples)| {
            let name = BusId::try_from(*bus)
                .map_or_else(|_| format!("bus {bus}"), |b| b.name().to_string());
            let mut tuples = tuples.clone();
            tuples.sort();
            let shown: Vec<String> = tuples
                .iter()
                .take(4)
                .map(|(tuple, net)| format!("{tuple:?} net {net:+}"))
                .collect();
            format!("{name}: {} tuple(s): {}", tuples.len(), shown.join(", "))
        })
        .collect::<Vec<_>>()
        .join(" · ")
}

/// The programs: a loop that repeats its ops past a row's `MU_MAX` (BRANCH, LT
/// and the ALU tables split rows), every instruction, the shifts, divisions,
/// loads and comparisons main's multiplicity fix bounded, KECCAK (KECCAK_RND's
/// padding; a state across a 64 KiB boundary, two IS_WH keys), COPY's writes.
const PROGRAMS: [&str; 11] = [
    "loop_4096",
    "all_instructions_64",
    "misalign_sd",
    "poc_mul_commit",
    "poc_divu_commit",
    "poc_sll_commit",
    "poc_slt_commit",
    "poc_lb_commit",
    "test_keccak_multi",
    "test_keccak_64k",
    "test_commit_split",
];

/// Whole-run builds at small chunks (every chunked table split) and at the
/// production sizes.
#[test]
fn the_whole_run_tables_balance_every_inner_bus() {
    for max_rows in [MaxRowsConfig::small(), MaxRowsConfig::default()] {
        for name in PROGRAMS {
            let (program, logs) = run(name);
            let mut traces = Traces::from_elf_and_logs(
                &program,
                &logs,
                &max_rows,
                &[],
                #[cfg(feature = "disk-spill")]
                StorageMode::Ram,
            )
            .expect("the build");
            let left = unbalanced(&program, &mut traces);
            assert!(left.is_empty(), "{name}: {}", named(&left));
        }
    }
}

/// Windowed builds: streamed chunks counted as they are handed out, KECCAK_RND
/// cut at the finish under either padding rule, KECCAK / ECSM / ECDAS / COPY cut.
#[test]
fn the_windowed_tables_balance_every_inner_bus() {
    type Configure = Box<dyn Fn(WindowedTraceBuilder<'_>) -> WindowedTraceBuilder<'_>>;
    let configs: Vec<(&str, Configure)> = vec![
        ("plain", Box::new(|b| b)),
        (
            "dropped",
            Box::new(|b| b.drop_streamed_ops().expect("before any window")),
        ),
        (
            "dropped, KECCAK_RND at the finish, cuts, last table padded",
            Box::new(|b| {
                b.keccak_rnd_chunks_at_finish(32)
                    .expect("rows")
                    .cuts_at_finish(2, 4, 2, 2)
                    .expect("rows")
                    .cut_padding(CutPadding::LastTable)
                    .drop_streamed_ops()
                    .expect("before any window")
            }),
        ),
        (
            "KECCAK_RND streamed, whole run padded",
            Box::new(|b| {
                b.keccak_rnd_chunks(32)
                    .expect("rows")
                    .cut_padding(CutPadding::WholeRun)
                    .drop_streamed_ops()
                    .expect("before any window")
            }),
        ),
    ];
    let mut failed = Vec::new();
    for (sizes, max_rows) in [
        ("small", MaxRowsConfig::small()),
        ("2^4", MaxRowsConfig::uniform(4)),
    ] {
        for name in PROGRAMS {
            let (program, logs) = run(name);
            for window in [7, 1000] {
                for (config, configure) in &configs {
                    let mut traces =
                        windowed(&program, &logs, &max_rows, window, configure.as_ref());
                    let left = unbalanced(&program, &mut traces);
                    if !left.is_empty() {
                        failed.push(format!(
                            "{name} {sizes} window {window} {config}: {}",
                            named(&left)
                        ));
                    }
                }
            }
        }
    }
    assert!(failed.is_empty(), "{}", failed.join("\n"));
}
