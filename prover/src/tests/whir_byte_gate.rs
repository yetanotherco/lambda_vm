//! ★ THE BYTE GATE — the WHIR identity line over a canonically sorted EQ trace.
//!
//! A printing measurement, not an assertion. It exists to be run on two
//! revisions and have its output compared, which is a thing a test harness
//! cannot do for you, so it prints and is `#[ignore]`d.
//!
//! ```text
//! cargo test --release -p lambda-vm-prover --lib \
//!     the_whir_identity_line_over_a_canonically_sorted_eq_trace \
//!     -- --ignored --nocapture
//! ```
//!
//! # (a) The sort is MEASUREMENT-ONLY
//!
//! The rows are canonically ordered HERE, after `generate_eq_trace` returns,
//! and none of the six trace builders is touched. Row order is free to the
//! argument — the bus is a multiset — so a sorted trace is still a valid EQ
//! trace that proves and verifies; it is simply a reproducible one.
//!
//! # (b) Why it exists at all, which is a mistake worth not repeating
//!
//! The obvious version of this measurement — hash the proof of
//! `generate_eq_trace`'s output — **is not reproducible across processes**, and
//! quoting it across two revisions produces a number that looks like evidence
//! and is not. `generate_eq_trace` deduplicates through a
//! `std::collections::HashMap` and lays its rows out in iteration order
//! (`prover/src/tables/eq.rs:128`), which `RandomState` randomises per map;
//! five sibling generators do the same (`bytewise.rs:107`, `branch.rs:166`,
//! `dvrm.rs:298`, `lt.rs:168`, `mul.rs:306`).
//!
//! Four consecutive runs of the unsorted version, on ONE unchanged tree and one
//! unchanged binary, observed 2026-09-15:
//!
//! ```text
//! 9147a1b9ad34e92248608c997506b4b6c06228654fb8717eca04c09f17236bc5
//! 7b8afea2618350600e99bb67200bb4447d962f753b6e858ee0982336436e6dd3
//! f3c9671a29d0f8fe02aaaf9d1ea85e86e13577989489f914c5c89802cb70970e
//! d778a2de322c96e1733757cfa7668ea7180907914af82d79cf6c6a72e81ceca2
//! ```
//!
//! Two of those agreeing by chance across two revisions is roughly a one-in-ten
//! event, and it happened. **Run any instrument twice on one tree before
//! quoting it across two.**
//!
//! # (c) The expected value
//!
//! With the sort, three consecutive runs agree, and the line is identical at
//! PR #988's head and at every commit of this branch:
//!
//! ```text
//! 307d7c00  7b8afea2618350600e99bb67200bb4447d962f753b6e858ee0982336436e6dd3   6880 bytes
//! bcdd3dd2  7b8afea2618350600e99bb67200bb4447d962f753b6e858ee0982336436e6dd3   6880 bytes
//! 29fbb45d  7b8afea2618350600e99bb67200bb4447d962f753b6e858ee0982336436e6dd3   6880 bytes
//! ```
//!
//! That is the gate: **the keccak arm's bytes do not move.** A commit that
//! changes this line has changed the proof PR #988 produces, and owes an
//! explanation.
//!
//! # What it does NOT cover
//!
//! Grinding is off, so no nonce reaches the transcript and nothing here
//! exercises the proof-of-work path — see `whir_identity_tests` for why a
//! ground proof cannot be gated this way at all. One table, one group, one
//! stacked polynomial: this is a canary for the seam, not a block-level
//! measurement.

use digest::Digest;
use math::field::element::FieldElement;
use math::field::{
    extensions_goldilocks::Degree3GoldilocksExtensionField as Ext,
    goldilocks::GoldilocksField as Fp,
};

use crypto::fiat_shamir::default_transcript::DefaultTranscript;
use multilinear::whir_chain::{ChainConfig, GrindBits};
use multilinear::whir_hash::KeccakWhir;
use stark::multilinear_air::Uniforms;
use stark::multilinear_table::{self, CommittedTable, CommittedTables, TableLayout};
use stark::proof::options::ProofOptions;
use stark::traits::AIR;

use crate::tables::eq::{EqOperation, generate_eq_trace};
use crate::test_utils::{ConcreteVmAir, create_eq_air};

/// The EQ trace's columns with the ROWS sorted into a canonical order.
///
/// Sorted by the whole row read as canonical `u64`s, so the result does not
/// depend on the incoming order — which is the entire point.
fn canonically_sorted_columns() -> Vec<Vec<FieldElement<Fp>>> {
    let ops = vec![
        EqOperation::new(7, 7, false),
        EqOperation::new(7, 9, false),
        EqOperation::new(3, 3, true),
        EqOperation::new(3, 5, true),
    ];
    let columns: Vec<Vec<FieldElement<Fp>>> = generate_eq_trace(&ops).columns_main();
    let rows = columns[0].len();

    let key = |r: usize| -> Vec<u64> { columns.iter().map(|c| *c[r].value()).collect() };
    let mut order: Vec<usize> = (0..rows).collect();
    order.sort_by_key(|r| key(*r));

    columns
        .iter()
        .map(|c| order.iter().map(|r| c[*r]).collect())
        .collect()
}

/// Prints the identity line and the serialized length. See the module header.
#[test]
#[ignore = "a printing measurement: run it on two revisions and compare the output"]
fn the_whir_identity_line_over_a_canonically_sorted_eq_trace() {
    let config = ChainConfig {
        log_blowup: 2,
        log_folding: 2,
        num_queries: 3,
        grind: GrindBits::default(),
    };

    let options = ProofOptions::default_test_options();
    let air: ConcreteVmAir<_> = create_eq_air(&options);
    let columns = canonically_sorted_columns();
    let num_vars = columns[0].len().trailing_zeros() as usize;

    let layout = TableLayout::<Fp, Ext>::new(
        air.constraint_program(),
        air.constraints_meta(),
        air.bus_interactions(),
        columns.len(),
        num_vars,
        Uniforms::default(),
    )
    .expect("layout");
    let table = CommittedTable::from_layout(layout, |col| columns[col as usize].clone())
        .expect("committed table");
    let committed =
        CommittedTables::<_, _, KeccakWhir>::commit(vec![table], &config).expect("commit");

    let mut transcript = DefaultTranscript::<Ext>::new(b"whir-identity");
    let proof =
        multilinear_table::multi_prove(&committed, &config, &mut transcript).expect("prove");

    let bytes = rkyv::to_bytes::<rkyv::rancor::Error>(&proof).expect("serialize");
    let line: String = crypto::hash::platform_keccak::PlatformKeccak256::digest(bytes.as_ref())
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();

    println!("IDENTITY-LINE {line}");
    println!("IDENTITY-LEN  {}", bytes.len());
}
