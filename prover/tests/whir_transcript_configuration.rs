//! ★★★ A prove under `H` squeezes `H`'s sponge, and no other.
//!
//! ```text
//! cargo test -p lambda-vm-prover --features hash-metrics \
//!     --test whir_transcript_configuration
//! ```
//!
//! # The claim a test finally makes
//!
//! `whir_hash_tests` has carried a test called
//! `the_transcript_follows_the_configuration` since the seam landed. It
//! constructed both transcript types itself and showed they draw different
//! challenges — true, and it never mentioned the prover. It was therefore true
//! throughout the period in which **no WHIR call site built an RPX transcript
//! at all**, and would have stayed true if none ever did. It is renamed to what
//! its body asserts; this file is the claim its old name made.
//!
//! Here the transcript is the production one: a real `multi_prove` through the
//! same entry point the prover uses, with the counters read afterwards. What is
//! observed is the prover's choice, not the test's.
//!
//! # Why every assertion is two-sided
//!
//! A single "squeezes" number cannot tell "the other hash ran" from "nothing
//! ran", and under the defect this exists to catch the RPX arm squeezed
//! thousands of times — all of them keccak. So each arm asserts both that its
//! own counters moved AND that the other's are zero:
//!
//! * keccak: every squeeze and absorb is keccak's;
//! * rpx: transcript work happened AND none of it was keccak.
//!
//! Without the second half the RPX assertion passed before the wiring landed.
//! Without the first, a prover that stopped squeezing would pass it too.
//!
//! # ⚠ Its own binary
//!
//! The counters are process-global and the prover's lib-test binary runs 600+
//! tests in parallel, most of which hash. An early version of this test lived
//! there and read `215` squeezes of which `200` were keccak — the other 15 were
//! a neighbour's. A binary of its own, and a lock within it, are both needed.

#![cfg(feature = "hash-metrics")]

use math::field::element::FieldElement;
use math::field::{
    extensions_goldilocks::Degree3GoldilocksExtensionField as Ext,
    goldilocks::GoldilocksField as Fp,
};

use crypto::hash_metrics;
use lambda_vm_prover::tables::eq::{EqOperation, generate_eq_trace};
use lambda_vm_prover::test_utils::create_eq_air;
use multilinear::whir_chain::{ChainConfig, GrindBits};
use multilinear::whir_hash::{KeccakWhir, RpxWhir, WhirHash};
use stark::multilinear_air::Uniforms;
use stark::multilinear_table::{self, CommittedTable, CommittedTables, TableLayout};
use stark::proof::options::ProofOptions;
use stark::traits::AIR;

/// Counters are global; tests here take turns.
static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn serialise() -> std::sync::MutexGuard<'static, ()> {
    LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

/// Grinding ON: the grind is one of the three hash consumers the seam names,
/// so a fixture that left it out could not see a half-flip. Four bits, so the
/// search costs nothing.
fn config() -> ChainConfig {
    ChainConfig {
        log_blowup: 2,
        log_folding: 2,
        num_queries: 3,
        grind: GrindBits::uniform(4),
    }
}

/// One real prove under `H`, through the production entry point.
fn prove<H: WhirHash>() {
    let ops = vec![
        EqOperation::new(7, 7, false),
        EqOperation::new(7, 9, false),
        EqOperation::new(3, 3, true),
        EqOperation::new(3, 5, true),
    ];
    let columns: Vec<Vec<FieldElement<Fp>>> = generate_eq_trace(&ops).columns_main();

    // Leaked so the layout borrows nothing from a temporary, as in the sibling
    // fixture; this is a test binary that exits immediately after.
    let air = Box::leak(Box::new(create_eq_air(
        &ProofOptions::default_test_options(),
    )));
    let layout = TableLayout::<Fp, Ext>::new(
        air.constraint_program(),
        air.constraints_meta(),
        air.bus_interactions(),
        columns.len(),
        columns[0].len().trailing_zeros() as usize,
        Uniforms::default(),
    )
    .expect("layout");

    let table = CommittedTable::from_layout(layout, |col| columns[col as usize].clone())
        .expect("committed table");
    let committed = CommittedTables::<_, _, H>::commit(vec![table], &config()).expect("commit");

    let mut transcript = crypto::fiat_shamir::default_transcript::DefaultTranscript::<
        Ext,
        H::Transcript,
    >::new(b"whir-transcript-configuration");
    multilinear_table::multi_prove(&committed, &config(), &mut transcript).expect("prove");
}

#[test]
fn a_keccak_prove_squeezes_only_keccak() {
    let _serialised = serialise();

    hash_metrics::reset();
    prove::<KeccakWhir>();
    let c = hash_metrics::snapshot();

    assert!(
        c.transcript_squeezes > 0 && c.transcript_absorbs > 0,
        "the prove did no transcript work at all ({} absorbs, {} squeezes), so \
         the zeros below would mean nothing",
        c.transcript_absorbs,
        c.transcript_squeezes
    );
    assert_eq!(
        (c.transcript_absorbs_keccak, c.transcript_squeezes_keccak),
        (c.transcript_absorbs, c.transcript_squeezes),
        "a keccak prove squeezed something that was not keccak"
    );
    assert_eq!(
        (c.transcript_absorbs_rpx, c.transcript_squeezes_rpx),
        (0, 0)
    );
    assert_eq!(c.transcript_unattributed(), (0, 0, 0));
}

#[test]
fn an_rpx_prove_squeezes_only_rpx() {
    let _serialised = serialise();

    hash_metrics::reset();
    prove::<RpxWhir>();
    let c = hash_metrics::snapshot();

    assert!(
        c.transcript_squeezes > 0 && c.transcript_absorbs > 0,
        "the prove did no transcript work at all, so the keccak zero below is \
         not evidence of anything"
    );
    assert_eq!(
        (c.transcript_absorbs_rpx, c.transcript_squeezes_rpx),
        (c.transcript_absorbs, c.transcript_squeezes),
        "an RPX prove did transcript work on some other sponge"
    );
    assert_eq!(
        (c.transcript_absorbs_keccak, c.transcript_squeezes_keccak),
        (0, 0),
        "the prover ran a KECCAK transcript under an RPX configuration — {} \
         absorbs and {} squeezes of it. This is the defect this file exists \
         for; it stood through four measured A/Bs.",
        c.transcript_absorbs_keccak,
        c.transcript_squeezes_keccak
    );
    assert_eq!(c.transcript_unattributed(), (0, 0, 0));
}

/// ★ The two configurations do the same amount of transcript WORK.
///
/// Only the bucket may differ. If a hash swap changed the absorb or squeeze
/// count, the per-arm bench line would report a protocol difference as a hash
/// difference, and a reader comparing arms would misattribute it.
#[test]
fn the_two_configurations_do_the_same_transcript_work() {
    let _serialised = serialise();

    hash_metrics::reset();
    prove::<KeccakWhir>();
    let k = hash_metrics::snapshot();

    hash_metrics::reset();
    prove::<RpxWhir>();
    let r = hash_metrics::snapshot();

    assert_eq!(
        (k.transcript_absorbs, k.transcript_squeezes),
        (r.transcript_absorbs, r.transcript_squeezes),
        "the two arms do different amounts of transcript work"
    );
}
