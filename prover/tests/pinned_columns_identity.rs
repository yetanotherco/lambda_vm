//! Columns in a pinned slot prove the same bytes as columns in `Vec`s (D-TRACE
//! stage 1b, S3).
//!
//! ```text
//! cargo test --release -p lambda-vm-prover --features cuda \
//!     --test pinned_columns_identity -- --ignored --test-threads=1
//! ```
//!
//! With `LAMBDA_VM_WHIR_PINNED_COLUMNS=1` each committed column is a view into
//! an epoch's page-locked slot (`Mle::shared`), and the upload sends each
//! column up to its last raw nonzero value and zeroes the rest on the card. The
//! card must end up holding the same values, so a `multi_prove` over the same
//! columns must serialise to the same bytes either way.
//!
//! What keeps these from being checks that cannot fail:
//!
//! * the control: two `Vec` proves agree, so nothing but the columns decides
//!   the bytes (no grinding here; the RV64 VM's own traces are not
//!   reproducible, so the columns are built once and cloned);
//! * the path: the upload counters show the pinned arm's upload was pinned and
//!   left tails behind, and the `Vec` arms' neither — on a host without a card
//!   the upload is declined and this fails rather than comparing host proofs;
//! * the mutation: another valid table of the same shape moves the bytes;
//! * the negative: a tail claimed one value early leaves a nonzero value zero
//!   on the card, and the proof is no longer the honest one;
//! * the refusal: with the upload declined, the views are copied out before
//!   anything reads them — the slot is back in the pool before the prove — and
//!   the bytes are the `Vec` columns' under the same refusal.
//!
//! Needs a card, hence `--ignored`.

#![cfg(feature = "cuda")]

use std::sync::{Arc, OnceLock};

use math::field::element::FieldElement;
use math::field::{
    extensions_goldilocks::Degree3GoldilocksExtensionField as Ext,
    goldilocks::GoldilocksField as Fp,
};

use lambda_vm_prover::tables::eq::{EqOperation, generate_eq_trace};
use lambda_vm_prover::test_utils::create_eq_air;
use math_cuda::columns::upload_totals;
use multilinear::mle::{HostColumns, Mle};
use multilinear::pinned::Pool;
use multilinear::whir_chain::{ChainConfig, GrindBits};
use multilinear::whir_hash::RpxWhir;
use stark::multilinear_air::Uniforms;
use stark::multilinear_table::{self, CommittedTable, CommittedTables, TableLayout};
use stark::proof::options::ProofOptions;
use stark::traits::AIR;

type Columns = Vec<Vec<FieldElement<Fp>>>;

/// The upload counters and the pool are process-wide; tests here take turns.
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
/// 40,000 of 65,536 rows every column carries a zero tail.
fn eq_columns(real: u64, salt: u64) -> Columns {
    let ops: Vec<EqOperation> = (0..real)
        .map(|i| EqOperation::new(i + salt, if i % 3 == 0 { i + salt } else { i }, i % 2 == 0))
        .collect();
    generate_eq_trace(&ops).columns_main()
}

fn tables() -> Vec<Columns> {
    vec![
        eq_columns(40_000, 1 << 40),
        eq_columns(20_000, 2 << 40),
        eq_columns(9_000, 3 << 40),
    ]
}

/// One slot, big enough for every fixture here, allocated once.
fn pool() -> &'static Pool {
    static POOL: OnceLock<Pool> = OnceLock::new();
    POOL.get_or_init(|| {
        let pool = Pool::new(1, 64 << 20);
        let took = pool.fill().expect("a pinned slot on the card");
        assert_eq!(took.len(), 1);
        pool
    })
}

/// The tables written into a slot the way the producer writes an epoch: back
/// to back, each table's columns in order, each column's tail found by the raw
/// word. `early` claims that one column's tail starts one value sooner.
struct Laid {
    host: Arc<dyn HostColumns<Fp>>,
    /// Per table: where its columns start, their height, each one's tail.
    tables: Vec<(usize, usize, Vec<usize>)>,
}

fn lay_out(tables: &[Columns], early: Option<(usize, usize)>) -> Laid {
    let cells: usize = tables.iter().map(|t| t.len() * t[0].len()).sum();
    let mut lease = pool().try_lease(cells).expect("the slot is free");
    let values = lease.values_mut();
    let mut laid = Vec::with_capacity(tables.len());
    let mut at = 0usize;
    for (t, columns) in tables.iter().enumerate() {
        let height = columns[0].len();
        let mut nonzero = Vec::with_capacity(columns.len());
        for (c, column) in columns.iter().enumerate() {
            values[at + c * height..at + (c + 1) * height].clone_from_slice(column);
            let mut n = column
                .iter()
                .rposition(|v| *v.value() != 0)
                .map_or(0, |i| i + 1);
            if early == Some((t, c)) {
                assert!(n > 0, "the early tail needs a column with a nonzero value");
                n -= 1;
            }
            nonzero.push(n);
        }
        laid.push((at, height, nonzero));
        at += columns.len() * height;
    }
    Laid {
        host: lease.freeze(),
        tables: laid,
    }
}

/// The committed tables over `tables`' columns: cloned `Vec`s, or views into
/// the slot `laid` describes. Takes `laid`, so the views hold the only
/// handles to the slot.
fn committed_tables(
    air: &'static impl AIR<Field = Fp, FieldExtension = Ext>,
    tables: &[Columns],
    laid: Option<Laid>,
) -> Vec<CommittedTable<'static, Fp, Ext>> {
    tables
        .iter()
        .enumerate()
        .map(|(t, columns)| {
            let layout = TableLayout::<Fp, Ext>::new(
                air.constraint_program(),
                air.constraints_meta(),
                air.bus_interactions(),
                columns.len(),
                columns[0].len().trailing_zeros() as usize,
                Uniforms::default(),
            )
            .expect("layout");
            match &laid {
                None => CommittedTable::from_layout(layout, |col| columns[col as usize].clone()),
                Some(laid) => {
                    let (start, height, nonzero) = &laid.tables[t];
                    CommittedTable::from_layout_mles(layout, |col| {
                        let col = col as usize;
                        Mle::shared(
                            laid.host.clone(),
                            start + col * height,
                            *height,
                            nonzero[col],
                        )
                    })
                }
            }
            .expect("committed table")
        })
        .collect()
}

fn air() -> &'static impl AIR<Field = Fp, FieldExtension = Ext> {
    Box::leak(Box::new(create_eq_air(
        &ProofOptions::default_test_options(),
    )))
}

/// The proof over `committed`, serialised, or why there is none.
fn prove(committed: &CommittedTables<'_, Fp, Ext, RpxWhir>) -> Result<Vec<u8>, String> {
    let mut transcript = crypto::fiat_shamir::default_transcript::DefaultTranscript::<
        Ext,
        <RpxWhir as multilinear::whir_hash::WhirHash>::Transcript,
    >::new(b"pinned-columns-identity");
    let proof = multilinear_table::multi_prove(committed, &config(), &mut transcript, None)
        .map_err(|e| format!("{e:?}"))?;
    Ok(rkyv::to_bytes::<rkyv::rancor::Error>(&proof)
        .expect("serialise")
        .to_vec())
}

fn proof_bytes(tables: &[Columns], laid: Option<Laid>) -> Result<Vec<u8>, String> {
    let committed =
        CommittedTables::<_, _, RpxWhir>::commit(committed_tables(air(), tables, laid), &config())
            .map_err(|e| format!("{e:?}"))?;
    prove(&committed)
}

#[test]
#[ignore = "needs a card"]
fn pinned_columns_prove_the_same_bytes() {
    let _turn = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let tables = tables();

    let before = upload_totals();
    let owned = proof_bytes(&tables, None).expect("the Vec prove");
    let after_owned = upload_totals();
    assert!(after_owned.0 > before.0, "the Vec arm uploaded no columns");
    assert_eq!(after_owned.1, before.1, "the Vec arm's upload was pinned");
    assert_eq!(after_owned.3, before.3, "the Vec arm left tails behind");

    let pinned = proof_bytes(&tables, Some(lay_out(&tables, None))).expect("the pinned prove");
    let after_pinned = upload_totals();
    assert_eq!(
        after_pinned.1,
        after_owned.1 + 1,
        "the pinned arm's upload was not the one pinned upload"
    );
    assert!(
        after_pinned.3 > after_owned.3,
        "the pinned arm left no tail behind, so the tail path went untested"
    );
    assert_eq!(pool().stats().free, 1, "the slot did not come back");

    let again = proof_bytes(&tables, None).expect("the second Vec prove");
    assert_eq!(owned, again, "CONTROL: two Vec proves disagree");
    assert_eq!(pinned, owned, "the pinned columns changed the proof");

    // The mutation: another valid first table (other operands, the same
    // shape) moves the bytes, pinned as well.
    let mut mutated = tables.clone();
    mutated[0] = eq_columns(40_000, 4 << 40);
    assert_ne!(mutated[0], tables[0], "the mutation changed nothing");
    assert_ne!(
        proof_bytes(&mutated, Some(lay_out(&mutated, None))).expect("the mutated prove"),
        owned,
        "MUTATION: a changed column left the proof unchanged"
    );
}

/// The negative: the first table's first column claims its zero tail starts
/// one value before its last nonzero one. The card then holds a zero there
/// while the host holds the value, and the prove either refuses or produces a
/// proof that is not the honest one — never the honest bytes.
#[test]
#[ignore = "needs a card"]
fn a_tail_claimed_early_is_not_the_honest_proof() {
    let _turn = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let tables = tables();
    let honest = proof_bytes(&tables, None).expect("the Vec prove");
    let early = proof_bytes(&tables, Some(lay_out(&tables, Some((0, 0)))));
    match &early {
        Ok(bytes) => assert_ne!(
            *bytes, honest,
            "a value left zero on the card did not change the proof"
        ),
        Err(why) => println!("the prove refused the early tail: {why}"),
    }
    assert_eq!(pool().stats().free, 1, "the slot did not come back");
}

/// The refusal: with this thread's upload declined, the commit copies every
/// view out before the stacks read anything, so the slot is back in the pool
/// before the prove starts, and the bytes are the `Vec` columns' under the
/// same refusal.
#[test]
#[ignore = "needs a card"]
fn a_refused_upload_copies_the_views_out_first() {
    let _turn = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let tables = tables();
    multilinear::gpu::refuse_column_uploads_on_this_thread(true);
    let before = upload_totals();
    let owned = proof_bytes(&tables, None).expect("the Vec prove, refused");

    let detached = multilinear_table::host_backing_detaches();
    let committed = CommittedTables::<_, _, RpxWhir>::commit(
        committed_tables(air(), &tables, Some(lay_out(&tables, None))),
        &config(),
    )
    .expect("the pinned commit, refused");
    assert_eq!(
        multilinear_table::host_backing_detaches(),
        detached + 1,
        "the refused commit did not copy the views out"
    );
    assert_eq!(
        pool().stats().free,
        1,
        "the slot is still held after the views were copied out"
    );
    let pinned = prove(&committed).expect("the pinned prove, refused");
    drop(committed);
    multilinear::gpu::refuse_column_uploads_on_this_thread(false);
    assert_eq!(
        upload_totals().0,
        before.0,
        "a refused arm uploaded columns"
    );
    assert_eq!(pinned, owned, "copying the views out changed the proof");
}
