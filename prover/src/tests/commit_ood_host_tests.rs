//! The commit-time answers read on the host from tables held narrow
//! (`stacked_eval::ood_claims`): each column's value comes from its table's
//! one host widen, which the evaluation must not enter from inside its own
//! parallel map.

use std::sync::mpsc;
use std::time::Duration;

use multilinear::constraint_argument::{FactorKind, TraceData};
use multilinear::narrow::{ColumnOf, NarrowColumns};
use multilinear::stacking::StackedLayout;

use crate::tables::types::{FE, FEE, GoldilocksExtension, GoldilocksField};

type F = GoldilocksField;
type E = GoldilocksExtension;

/// Columns a narrow table holds: enough that the evaluation's jobs for it
/// outnumber the workers.
const COLUMNS: usize = 64;
const NUM_VARS: usize = 10;
/// Fresh tables, each evaluated once.
const ROUNDS: usize = 16;

fn narrow_table(seed: u64) -> TraceData<F, E> {
    let words: Vec<Vec<u64>> = (0..COLUMNS as u64)
        .map(|c| {
            (0..1u64 << NUM_VARS)
                .map(|r| {
                    (r.wrapping_mul(0x9e37_79b9) ^ (c + seed).wrapping_mul(0x85eb_ca6b)) % 65_521
                })
                .collect()
        })
        .collect();
    let raw: Vec<&[u64]> = words.iter().map(Vec::as_slice).collect();
    let packed = NarrowColumns::pack(&raw).expect("one height");
    let kinds = (0..COLUMNS).map(FactorKind::direct).collect();
    TraceData::new_narrow(packed, kinds, Vec::new()).expect("a narrow table")
}

/// ★ Every column of a table held narrow answers on the host, each the value
/// of its widened column at its point, and the evaluation ends: the widen runs
/// once, and no job of the evaluation waits on it from inside the pool. Bounded
/// by a timeout, so a regression fails instead of hanging the run.
#[test]
fn a_narrow_tables_answers_end_and_match_its_widened_columns() {
    let (done, finished) = mpsc::channel();
    std::thread::spawn(move || {
        for round in 0..ROUNDS {
            let table = narrow_table(round as u64);
            let handles: Vec<ColumnOf<'_, TraceData<F, E>>> = (0..COLUMNS)
                .map(|index| ColumnOf {
                    columns: &table,
                    index,
                })
                .collect();
            let columns: Vec<&ColumnOf<'_, TraceData<F, E>>> = handles.iter().collect();
            let layout =
                StackedLayout::build(&[NUM_VARS; COLUMNS], NUM_VARS + 6).expect("the layout packs");
            let z0 = FEE::new([FE::from(7 + round as u64), FE::from(11), FE::from(13)]);
            let (claims, counts) =
                multilinear::stacked_eval::ood_claims::<F, E, _>(&layout, &columns, None, &z0)
                    .expect("the answers");
            assert_eq!(counts.on_host, COLUMNS, "no card: every column on the host");
            let point = multilinear::whir_chain::ood_point(&z0, layout.n_stack());
            let widened = table.columns();
            for (c, value) in claims.columns.iter().enumerate() {
                let at = multilinear::commit_ood::column_point(&layout, c, &point);
                assert_eq!(
                    *value,
                    widened[c]
                        .evaluate_in(at)
                        .expect("a column takes its point"),
                    "round {round} column {c}"
                );
            }
        }
        let _ = done.send(());
    });
    finished.recv_timeout(Duration::from_secs(60)).expect(
        "the answers over narrow tables never ended: the evaluation waits on its own widen",
    );
}
