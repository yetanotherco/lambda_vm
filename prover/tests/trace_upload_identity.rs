//! The zero-tail column upload (`LAMBDA_VM_TRACE_UPLOAD=zerotail`) proves the
//! same bytes.
//!
//! ```text
//! cargo test --release -p lambda-vm-prover --features cuda \
//!     --test trace_upload_identity -- --ignored --test-threads=1
//! ```
//!
//! The knob changes how the epoch's columns reach the card: each column's
//! all-zero tail is zeroed on the card instead of sent. The card must end up
//! holding the same values, so a `multi_prove` over the same columns must
//! serialise to the same bytes with the knob on as off.
//!
//! Three things keep this from being a check that cannot fail:
//!
//! * the control: two proves with the knob off agree, so the prover is
//!   deterministic over one set of columns (no grinding here; the RV64 VM's own
//!   traces are not reproducible, so the columns are built once and cloned);
//! * the path: the upload counters show the zero-tail arm left tails behind and
//!   the plain arms did not — on a host without a card the upload is declined
//!   and this fails rather than comparing two host proofs;
//! * the mutation: another valid table of the same shape changes the bytes.
//!
//! Needs a card, hence `--ignored`.

#![cfg(feature = "cuda")]

use math::field::element::FieldElement;
use math::field::{
    extensions_goldilocks::Degree3GoldilocksExtensionField as Ext,
    goldilocks::GoldilocksField as Fp,
};

use lambda_vm_prover::tables::eq::{EqOperation, generate_eq_trace};
use lambda_vm_prover::test_utils::create_eq_air;
use math_cuda::columns::{set_trace_upload_override, upload_totals};
use multilinear::whir_chain::{ChainConfig, GrindBits};
use multilinear::whir_hash::RpxWhir;
use stark::multilinear_air::Uniforms;
use stark::multilinear_table::{self, CommittedTable, CommittedTables, TableLayout};
use stark::proof::options::ProofOptions;
use stark::traits::AIR;

type Columns = Vec<Vec<FieldElement<Fp>>>;

/// The upload counters are process-wide; tests here take turns.
static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// No grinding, so nothing but the columns decides the proof.
fn config() -> ChainConfig {
    ChainConfig {
        log_blowup: 2,
        log_folding: 2,
        num_queries: 3,
        grind: GrindBits::uniform(0),
        format: multilinear::whir_chain::ChainFormat::DEFAULT,
    }
}

/// An EQ table of `real` distinct operations, padded to a power of two: at
/// 40,000 of 65,536 rows every column carries a 204,288-byte zero tail, over the
/// 64 KiB the upload cuts at.
fn eq_columns(real: u64, salt: u64) -> Columns {
    let ops: Vec<EqOperation> = (0..real)
        .map(|i| EqOperation::new(i + salt, if i % 3 == 0 { i + salt } else { i }, i % 2 == 0))
        .collect();
    generate_eq_trace(&ops).columns_main()
}

/// The proof over `tables`' columns, serialised, with the upload pinned to the
/// zero-tail path or not.
fn proof_bytes(tables: &[Columns], zero_tails: bool) -> Vec<u8> {
    let air = Box::leak(Box::new(create_eq_air(
        &ProofOptions::default_test_options(),
    )));
    let committed_tables = tables
        .iter()
        .map(|columns| {
            let layout = TableLayout::<Fp, Ext>::new(
                air.constraint_program(),
                air.constraints_meta(),
                air.bus_interactions(),
                columns.len(),
                columns[0].len().trailing_zeros() as usize,
                Uniforms::default(),
            )
            .expect("layout");
            CommittedTable::from_layout(layout, |col| columns[col as usize].clone())
                .expect("committed table")
        })
        .collect();
    set_trace_upload_override(Some(zero_tails));
    let committed =
        CommittedTables::<_, _, RpxWhir>::commit(committed_tables, &config()).expect("commit");
    set_trace_upload_override(None);
    let mut transcript = crypto::fiat_shamir::default_transcript::DefaultTranscript::<
        Ext,
        <RpxWhir as multilinear::whir_hash::WhirHash>::Transcript,
    >::new(b"trace-upload-identity");
    let proof = multilinear_table::multi_prove(&committed, &config(), &mut transcript, None)
        .expect("prove");
    rkyv::to_bytes::<rkyv::rancor::Error>(&proof)
        .expect("serialise")
        .to_vec()
}

#[test]
#[ignore = "needs a card"]
fn the_zero_tail_upload_proves_the_same_bytes() {
    let _turn = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    // Three tables of 12 columns: more columns than scanner threads, and tails
    // of three lengths.
    let tables = vec![
        eq_columns(40_000, 1 << 40),
        eq_columns(20_000, 2 << 40),
        eq_columns(9_000, 3 << 40),
    ];

    let before = upload_totals();
    let off = proof_bytes(&tables, false);
    let after_off = upload_totals();
    assert!(after_off.0 > before.0, "the plain arm uploaded no columns");
    assert_eq!(after_off.1, before.1, "the plain arm left tails behind");
    assert_eq!(after_off.3, before.3, "the plain arm skipped bytes");

    let on = proof_bytes(&tables, true);
    let after_on = upload_totals();
    assert!(
        after_on.1 > after_off.1,
        "the zero-tail arm did not take its path"
    );
    assert!(
        after_on.3 > after_off.3,
        "the zero-tail arm left no tail behind, so the tail path went untested"
    );

    let again = proof_bytes(&tables, false);
    assert_eq!(off, again, "CONTROL: two plain proves disagree");
    assert_eq!(on, off, "the zero-tail upload changed the proof");

    // The mutation: another valid first table (different operands, the same
    // shape) moves the bytes.
    let mut mutated = tables.clone();
    mutated[0] = eq_columns(40_000, 4 << 40);
    assert_ne!(mutated[0], tables[0], "the mutation changed nothing");
    assert_ne!(
        proof_bytes(&mutated, true),
        off,
        "MUTATION: a changed column left the proof unchanged"
    );
}
