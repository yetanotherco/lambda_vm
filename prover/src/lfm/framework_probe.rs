//! B0 de-risk probe (Milestone B entry gate).
//!
//! Every LFM chip stakes its instruction column group on one framework
//! pattern no in-tree chip exercises today: a **preprocessed column used as a
//! LogUp `Multiplicity`** (plus preprocessed bus values, which KECCAK_RC does
//! exercise). This probe round-trips a minimal sender/receiver pair through
//! the real `multi_prove` / `multi_verify_views`, with the sender's value and
//! multiplicity columns both preprocessed, and pins the tamper behavior:
//! a flipped preprocessed root is rejected by the prover (recommit mismatch)
//! and by the verifier (root equality), and a tampered witness value breaks
//! the bus balance.

use crypto::fiat_shamir::default_transcript::DefaultTranscript;
use crypto::fiat_shamir::is_transcript::IsTranscript;
use stark::config::Commitment;
use stark::constraints::builder::EmptyConstraints;
use stark::lookup::{
    AirWithBuses, AuxiliaryTraceBuildData, BusInteraction, BusValue, Multiplicity,
    NullBoundaryConstraintBuilder, Packing,
};
use stark::proof::options::{GoldilocksCubicProofOptions, ProofOptions};
use stark::proof::view::MultiProofView;
use stark::prover::{IsStarkProver, Prover};
use stark::trace::TraceTable;
use stark::traits::AIR;
use stark::verifier::{IsStarkVerifier, Verifier};

use crate::tables::types::{FE, FEE, GoldilocksExtension, GoldilocksField};

use super::commit::commit_columns;

type F = GoldilocksField;
type E = GoldilocksExtension;
type ProbeAir = AirWithBuses<F, E, NullBoundaryConstraintBuilder, (), EmptyConstraints>;
type DynAir<'a> = &'a dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>;

/// Scratch bus id, far above the live `BusId` range.
const PROBE_BUS: u64 = 63;
const PROBE_TAG: &[u8] = b"LFM_B0_PROBE_V1";
const NUM_ROWS: usize = 256;

fn fe(v: u64) -> FE {
    FE::from(v)
}

fn options() -> ProofOptions {
    GoldilocksCubicProofOptions::with_blowup(2).expect("probe options")
}

fn values() -> Vec<FE> {
    (0..NUM_ROWS as u64).map(|i| fe(1_000 + 3 * i)).collect()
}

/// Sender: cols `[VAL (prep 0), MULT (prep 1), PAD (main 2)]` — both the bus
/// value AND the multiplicity read preprocessed columns.
fn sender_air(prep_root: Commitment, opts: &ProofOptions) -> ProbeAir {
    let interactions = vec![BusInteraction::sender(
        PROBE_BUS,
        Multiplicity::Column(1),
        vec![BusValue::Packed {
            start_column: 0,
            packing: Packing::Direct,
        }],
    )];
    AirWithBuses::new(
        3,
        AuxiliaryTraceBuildData { interactions },
        opts,
        1,
        EmptyConstraints,
    )
    .with_name("B0_SEND")
    .with_preprocessed(prep_root, 2)
}

/// Receiver: cols `[VAL (main 0), MULT (main 1)]` — plain witness echo.
fn receiver_air(opts: &ProofOptions) -> ProbeAir {
    let interactions = vec![BusInteraction::receiver(
        PROBE_BUS,
        Multiplicity::Column(1),
        vec![BusValue::Packed {
            start_column: 0,
            packing: Packing::Direct,
        }],
    )];
    AirWithBuses::new(
        2,
        AuxiliaryTraceBuildData { interactions },
        opts,
        1,
        EmptyConstraints,
    )
    .with_name("B0_RECV")
}

fn sender_trace() -> TraceTable<F, E> {
    let mut data = Vec::with_capacity(NUM_ROWS * 3);
    for v in values() {
        data.extend([v, FE::one(), FE::zero()]);
    }
    TraceTable::new_main(data, 3, 1)
}

fn receiver_trace() -> TraceTable<F, E> {
    let mut data = Vec::with_capacity(NUM_ROWS * 2);
    for v in values() {
        data.extend([v, FE::one()]);
    }
    TraceTable::new_main(data, 2, 1)
}

fn prep_root(opts: &ProofOptions) -> Commitment {
    commit_columns(&[values(), vec![FE::one(); NUM_ROWS]], opts)
}

fn transcript() -> DefaultTranscript<E> {
    let mut t = DefaultTranscript::<E>::new(&[]);
    t.append_bytes(PROBE_TAG);
    t
}

fn prove(
    sender: &ProbeAir,
    receiver: &ProbeAir,
) -> Result<stark::proof::stark::MultiProof<F, E, ()>, stark::prover::ProvingError> {
    let mut st = sender_trace();
    let mut rt = receiver_trace();
    let pairs: Vec<(DynAir, &mut TraceTable<F, E>, &())> =
        vec![(sender, &mut st, &()), (receiver, &mut rt, &())];
    let mut t = transcript();
    Prover::multi_prove(
        pairs,
        &mut t,
        #[cfg(feature = "disk-spill")]
        Default::default(),
    )
}

#[test]
fn b0_preprocessed_multiplicity_round_trips() {
    let opts = options();
    let root = prep_root(&opts);
    let sender = sender_air(root, &opts);
    let receiver = receiver_air(&opts);
    let proof = prove(&sender, &receiver).expect("prove with preprocessed multiplicity");

    let refs: Vec<DynAir> = vec![&sender, &receiver];
    let mut vt = transcript();
    assert!(
        Verifier::multi_verify_views(&refs, MultiProofView::Owned(&proof), &mut vt, &FEE::zero(),),
        "honest proof must verify"
    );
}

#[test]
fn b0_prover_rejects_mismatched_preprocessed_root() {
    let opts = options();
    let mut root = prep_root(&opts);
    root[0] ^= 1;
    let sender = sender_air(root, &opts);
    let receiver = receiver_air(&opts);
    assert!(
        prove(&sender, &receiver).is_err(),
        "prover must reject a trace that does not recommit to the supplied root"
    );
}

#[test]
fn b0_verifier_rejects_wrong_preprocessed_root() {
    let opts = options();
    let root = prep_root(&opts);
    let sender = sender_air(root, &opts);
    let receiver = receiver_air(&opts);
    let proof = prove(&sender, &receiver).expect("honest prove");

    let mut bad_root = root;
    bad_root[0] ^= 1;
    let bad_sender = sender_air(bad_root, &opts);
    let refs: Vec<DynAir> = vec![&bad_sender, &receiver];
    let mut vt = transcript();
    assert!(
        !Verifier::multi_verify_views(&refs, MultiProofView::Owned(&proof), &mut vt, &FEE::zero(),),
        "a supplied root differing from the proof's must reject"
    );
}

#[test]
fn b0_tampered_witness_value_breaks_balance() {
    let opts = options();
    let root = prep_root(&opts);
    let sender = sender_air(root, &opts);
    let receiver = receiver_air(&opts);

    // Receiver echoes one wrong value: prove succeeds locally (no constraint
    // relates the two tables directly) but the bus no longer balances to 0.
    let mut st = sender_trace();
    let mut rt = receiver_trace();
    rt.set_main(0, 0, fe(999_999));
    let pairs: Vec<(DynAir, &mut TraceTable<F, E>, &())> =
        vec![(&sender, &mut st, &()), (&receiver, &mut rt, &())];
    let mut t = transcript();
    let proof = Prover::multi_prove(
        pairs,
        &mut t,
        #[cfg(feature = "disk-spill")]
        Default::default(),
    )
    .expect("locally consistent");

    let refs: Vec<DynAir> = vec![&sender, &receiver];
    let mut vt = transcript();
    assert!(
        !Verifier::multi_verify_views(&refs, MultiProofView::Owned(&proof), &mut vt, &FEE::zero(),),
        "unbalanced bus must reject"
    );
}
