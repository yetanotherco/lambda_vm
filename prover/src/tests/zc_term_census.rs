//! A census of what the batched zerocheck walks a row, per base table kind:
//! the constraint program's steps against the bus claims' linear terms (every
//! interaction's multiplicity and values, which the argue folds into the same
//! sumcheck as the GKR input claims), and the distinct columns those terms
//! read. Analysis only; prints `ZC CENSUS` lines.
//!
//! ```text
//! cargo test --release -p lambda-vm-prover --lib -- --exact --nocapture tests::zc_term_census::zc_term_census_of_the_block_tables
//! ```

use std::collections::BTreeSet;

use stark::lookup::{BusValue, LinearTerm, Multiplicity};
use stark::proof::options::ProofOptions;
use stark::traits::AIR;

use crate::block_whir::BlockFormat;

type DynAir<'a> = dyn AIR<
        Field = crate::tables::types::GoldilocksField,
        FieldExtension = crate::tables::types::GoldilocksExtension,
        PublicInputs = (),
    > + 'a;

/// `(terms, columns read)` of one multiplicity.
fn multiplicity_terms(m: &Multiplicity, cols: &mut BTreeSet<usize>) -> usize {
    match m {
        Multiplicity::One => 0,
        Multiplicity::Column(c) | Multiplicity::Negated(c) => {
            cols.insert(*c);
            1
        }
        Multiplicity::Sum(a, b) | Multiplicity::Diff(a, b) => {
            cols.extend([*a, *b]);
            2
        }
        Multiplicity::Sum3(a, b, c) => {
            cols.extend([*a, *b, *c]);
            3
        }
        Multiplicity::Linear(ts) => linear_terms(ts, cols),
    }
}

fn linear_terms(ts: &[LinearTerm], cols: &mut BTreeSet<usize>) -> usize {
    ts.iter()
        .filter(|t| match t {
            LinearTerm::Column { column, .. } | LinearTerm::ColumnUnsigned { column, .. } => {
                cols.insert(*column);
                true
            }
            LinearTerm::Constant(_) => false,
        })
        .count()
}

/// One kind's row: columns, constraint steps, interactions, bus terms, the
/// distinct columns the bus reads.
fn census(air: &DynAir<'_>) -> (String, usize, usize, usize, usize, usize) {
    let width = air.trace_layout().0;
    let layout = crate::multilinear_prove::layout_of(air, width, 4).expect("the table lays out");
    let steps = layout.statement().shape.steps_as_ops().len();
    let mut cols = BTreeSet::new();
    let mut terms = 0usize;
    for bi in air.bus_interactions() {
        terms += multiplicity_terms(&bi.multiplicity, &mut cols);
        for v in &bi.values {
            terms += match v {
                BusValue::Packed {
                    start_column,
                    packing,
                } => {
                    let n = packing.num_columns();
                    cols.extend(*start_column..*start_column + n);
                    n
                }
                BusValue::Linear(ts) => linear_terms(ts, &mut cols),
            };
        }
    }
    (
        air.name().to_string(),
        width,
        steps,
        air.bus_interactions().len(),
        terms,
        cols.len(),
    )
}

#[test]
fn zc_term_census_of_the_block_tables() {
    let opts = ProofOptions::default_test_options();
    let elf = crate::test_utils::asm_elf_bytes("keep_alive_regs");
    let program = executor::elf::Elf::load(&elf).expect("the ELF loads");
    let pages = crate::tables::trace_builder::Traces::page_configs_from_elf_and_runtime(
        &program,
        &[],
        0,
        1 << 20,
    )
    .expect("pages");
    let heights = vec![1 << 4];
    let counts = crate::TableCounts {
        cpu: 1,
        lt: 1,
        memw: 1,
        memw_aligned: 1,
        load: 1,
        mul: 1,
        dvrm: 1,
        shift: 1,
        branch: 1,
        memw_register: 1,
        eq: 1,
        bytewise: 1,
        store: 1,
        cpu32: 1,
        keccak: 1,
        keccak_rnd: 1,
        ecsm: 1,
        ecdas: 1,
        hint: 1,
        copy: 1,
        blake3: 1,
        is_wh: 1,
        ecrec: 1,
    };
    let inputs = crate::BlockAirInputs::new(&opts, &program, heights, &[]).expect("the AIR inputs");
    let format = BlockFormat::production();
    let airs = crate::block_airs(
        &program,
        &opts,
        &pages,
        &counts,
        &inputs,
        format.keccak_rnd,
        format.memw_aligned,
        format.registers,
    );
    let mut seen = BTreeSet::new();
    println!(
        "ZC CENSUS: kind · columns · constraint steps · interactions · bus terms · bus columns · terms / (steps + terms)"
    );
    for air in airs.air_refs() {
        let (name, width, steps, inter, terms, cols) = census(air);
        if !seen.insert(name.clone()) {
            continue;
        }
        println!(
            "ZC CENSUS {name} · {width} · {steps} · {inter} · {terms} · {cols} · {:.2}",
            terms as f64 / (steps + terms).max(1) as f64
        );
    }
}
