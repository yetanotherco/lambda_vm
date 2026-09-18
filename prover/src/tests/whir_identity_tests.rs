//! ★ The byte gate for the WHIR hash seam, and the proof that it is a gate.
//!
//! `whir_identity::identity_line` is the instrument the coordinator diffs
//! across arms, so what it can and cannot see has to be pinned rather than
//! described. Five propositions, each constructed rather than observed:
//!
//! 1. the line is STABLE when the proof is — two independent runs of the same
//!    prover on the same input give the same line;
//! 2. the line is SENSITIVE — moving one byte of one Merkle root moves it, and
//!    so does moving one opened codeword value;
//! 3. the line is BLIND TO NONCES, and to nothing else — changing only a
//!    grinding nonce leaves it alone, which is exactly the exclusion the
//!    instrument claims;
//! 4. the SERIALIZED LENGTH is unaffected by the nonce normalisation, so a
//!    length comparison between two arms measures the proof and not the
//!    instrument;
//! 5. ⚠ **the normalisation does NOT make a GROUND proof reproducible** — the
//!    thing this gate was first designed to do, which it cannot.
//!
//! Proposition 2 is what makes 1 and 3 worth anything: a digest that could not
//! change would satisfy 1 and 3 vacuously.
//!
//! # ⚠ The correction proposition 5 records
//!
//! The gate was pre-registered as "hash the proof with every grinding nonce
//! zeroed", on the reasoning that the nonce is the only nondeterministic field.
//! That reasoning is wrong, and the test below is what found it: the nonce is
//! **absorbed into the transcript** (`multilinear::whir_chain::grind`), so every
//! challenge drawn after the first grind depends on which valid nonce the
//! search returned. Two honest runs diverge in every root and every opening
//! from that point on, and the divergence is not in the nonce fields, so
//! zeroing them cannot remove it.
//!
//! What a byte gate needs instead is a search that returns the SAME valid nonce
//! — `crypto::grinding::generate_nonce_smallest`, reached in production by
//! setting `LAMBDA_VM_DETERMINISTIC_GRIND`. The nonce normalisation is kept
//! anyway, because it costs nothing and because it makes the line insensitive
//! to the one field that still legitimately varies between a CPU arm and a
//! device arm.
//!
//! # ⚠ A SECOND cause of irreproducibility, found the same way
//!
//! Even with no proof of work at all, two `generate_eq_trace` calls on the same
//! operations produce DIFFERENT TRACES: the generator deduplicates through a
//! `std::collections::HashMap` and lays the rows out in iteration order
//! (`prover/src/tables/eq.rs:128`), which `RandomState` randomises per map.
//! Five sibling generators do the same — BYTEWISE, BRANCH, DVRM, LT and MUL.
//! Row order is free to the argument (the bus is a multiset), so this is not a
//! soundness defect, but it means **a proof of the same program is not
//! byte-reproducible across processes for reasons that have nothing to do with
//! the hash**.
//!
//! So the fixture below builds its trace ONCE and proves it twice. What these
//! tests pin is the property the seam is responsible for — the WHIR prove path
//! is a function of its input — and not a property of the trace builders, which
//! is someone else's to fix.

use math::field::element::FieldElement;
use math::field::{
    extensions_goldilocks::Degree3GoldilocksExtensionField as Ext,
    goldilocks::GoldilocksField as Fp,
};

use crypto::fiat_shamir::default_transcript::DefaultTranscript;
use multilinear::whir_chain::{ChainConfig, GrindBits};
use multilinear::whir_hash::KeccakWhir;
use stark::multilinear_air::Uniforms;
use stark::multilinear_table::{self, CommittedTable, CommittedTables, MultiProof, TableLayout};
use stark::proof::options::ProofOptions;
use stark::traits::AIR;

use crate::tables::eq::{EqOperation, generate_eq_trace};
use crate::test_utils::{ConcreteVmAir, create_eq_air};
use crate::whir_identity::{identity_line, serialized_len};

type Proof = MultiProof<Fp, Ext>;

/// Grinding ON, deliberately: the nonces are what the instrument normalises
/// away, so a fixture without them could not exercise propositions 3 and 5.
/// Four bits, so the search costs nothing.
fn config() -> ChainConfig {
    ChainConfig {
        log_blowup: 2,
        log_folding: 2,
        num_queries: 3,
        grind: GrindBits::uniform(4),
    }
}

/// The same posture with no proof of work — the deterministic control.
fn config_unground() -> ChainConfig {
    ChainConfig {
        grind: GrindBits::default(),
        ..config()
    }
}

fn eq_operations() -> Vec<EqOperation> {
    vec![
        EqOperation::new(7, 7, false),
        EqOperation::new(7, 9, false),
        EqOperation::new(3, 3, true),
        EqOperation::new(3, 5, true),
    ]
}

/// The trace, built ONCE — see the header: the generator's row order follows a
/// `HashMap`, so building it per call would make every comparison below a test
/// of the trace builder instead of a test of the prover.
fn fixture_columns() -> Vec<Vec<FieldElement<Fp>>> {
    generate_eq_trace(&eq_operations()).columns_main()
}

/// One real table, proved end to end — the smallest thing that has roots,
/// openings and nonces all at once.
fn prove_once(seed: &[u8], columns: &[Vec<FieldElement<Fp>>]) -> Proof {
    prove_with(seed, columns, &config())
}

fn prove_once_unground(seed: &[u8], columns: &[Vec<FieldElement<Fp>>]) -> Proof {
    prove_with(seed, columns, &config_unground())
}

fn prove_with(seed: &[u8], columns: &[Vec<FieldElement<Fp>>], config: &ChainConfig) -> Proof {
    let options = ProofOptions::default_test_options();
    let air: ConcreteVmAir<_> = create_eq_air(&options);
    let num_main = columns.len();
    let num_vars = columns[0].len().trailing_zeros() as usize;

    let layout = TableLayout::<Fp, Ext>::new(
        air.constraint_program(),
        air.constraints_meta(),
        air.bus_interactions(),
        num_main,
        num_vars,
        Uniforms::default(),
    )
    .expect("layout");
    let table = CommittedTable::from_layout(layout, |col| columns[col as usize].clone())
        .expect("committed table");
    let committed =
        CommittedTables::<_, _, KeccakWhir>::commit(vec![table], config).expect("commit");

    let mut transcript = DefaultTranscript::<Ext>::new(seed);
    multilinear_table::multi_prove(&committed, config, &mut transcript, None).expect("prove")
}

/// The fixture is worth using only if it actually carries what the propositions
/// are about.
fn assert_fixture_is_not_degenerate(proof: &Proof) {
    assert!(!proof.roots.is_empty(), "the fixture has no Merkle root");
    let rounds = proof
        .columns
        .iter()
        .flat_map(|stacked| &stacked.polys)
        .flat_map(|chain| &chain.rounds)
        .count();
    assert!(rounds > 0, "the fixture has no WHIR rounds");
    let ground: u64 = proof
        .columns
        .iter()
        .flat_map(|stacked| &stacked.polys)
        .flat_map(|chain| &chain.rounds)
        .map(|r| r.nonces.folding | r.nonces.ood | r.nonces.query)
        .fold(0, |a, b| a | b);
    assert!(
        ground != 0,
        "the fixture ground no nonce, so the normalisation cannot be exercised"
    );
    let opened: usize = proof
        .columns
        .iter()
        .flat_map(|stacked| &stacked.polys)
        .map(|chain| chain.opened_elements())
        .sum();
    assert!(opened > 0, "the fixture opened no codeword value");
}

/// (1) The same input gives the same line, twice, when the proof itself is
/// deterministic.
#[test]
fn the_identity_line_is_stable_across_runs() {
    let columns = fixture_columns();
    let a = prove_once_unground(b"whir-identity", &columns);
    assert!(!a.roots.is_empty(), "the control fixture has no root");
    let b = prove_once_unground(b"whir-identity", &columns);

    assert_eq!(
        identity_line(&a).unwrap(),
        identity_line(&b).unwrap(),
        "two runs of the same prover on the same input must give one line"
    );
}

/// (5) ⚠ **The correction, pinned so nobody re-derives the wrong gate.**
///
/// With grinding on, two honest runs differ in far more than their nonces, and
/// the normalised line differs too. This is the proposition that killed the
/// original byte-gate design; it is here so the next reader is told by a test
/// rather than by a comment.
///
/// It compares the two proofs' NON-nonce content directly rather than asserting
/// two digests differ: the digests differing is the consequence, the roots
/// differing is the cause, and a test that asserted only the consequence would
/// pass for the wrong reason if the instrument broke. If the search ever became
/// deterministic by default, this assertion would fail LOUDLY and name the
/// reason — which is the correct outcome, not a flake.
#[test]
fn zeroing_the_nonces_does_not_make_a_ground_proof_reproducible() {
    let columns = fixture_columns();
    let a = prove_once(b"whir-identity", &columns);
    assert_fixture_is_not_degenerate(&a);
    let b = prove_once(b"whir-identity", &columns);

    let nonces_of = |p: &Proof| {
        p.columns
            .iter()
            .flat_map(|s| &s.polys)
            .flat_map(|c| &c.rounds)
            .map(|r| r.nonces)
            .collect::<Vec<_>>()
    };
    if nonces_of(&a) == nonces_of(&b) {
        // The two searches happened to agree — at four bits that is common.
        // Nothing is being claimed about this run.
        return;
    }

    // The COMMITTED trace's root is drawn before any grind, so it does not
    // move — naming that explicitly, because it is the thing that makes this
    // failure mode easy to miss. What moves is everything the transcript
    // produced after the first grind.
    assert_eq!(
        a.roots, b.roots,
        "the trace commitment precedes the first grind and cannot depend on it"
    );

    let after_the_grind = |p: &Proof| {
        p.columns
            .iter()
            .flat_map(|s| &s.polys)
            .map(|c| {
                (
                    c.final_value,
                    c.rounds
                        .iter()
                        .map(|r| (r.next_root, r.ood_value))
                        .collect::<Vec<_>>(),
                )
            })
            .collect::<Vec<_>>()
    };
    assert_ne!(
        after_the_grind(&a),
        after_the_grind(&b),
        "the nonce is absorbed, so a different nonce must move every challenge \
         drawn after it — the successor roots and the out-of-domain values"
    );
    assert_ne!(
        identity_line(&a).unwrap(),
        identity_line(&b).unwrap(),
        "and the normalised line moves with them: the nonce fields are not where \
         the divergence lives, so zeroing them cannot remove it"
    );
}

/// (2a) One byte of one root moves the line.
#[test]
fn the_identity_line_moves_when_a_root_moves() {
    let columns = fixture_columns();
    let proof = prove_once(b"whir-identity", &columns);
    assert_fixture_is_not_degenerate(&proof);
    let before = identity_line(&proof).unwrap();

    let mut tampered = proof.clone();
    tampered.roots[0][0] ^= 1;

    assert_ne!(
        before,
        identity_line(&tampered).unwrap(),
        "a changed Merkle root must change the line"
    );
}

/// (2b) One opened codeword value moves the line.
///
/// The roots are the obvious field; the openings are the bulk of the bytes and
/// the thing a leaf-hash defect would corrupt, so they are checked separately.
#[test]
fn the_identity_line_moves_when_an_opened_value_moves() {
    use multilinear::whir_chain::RoundOpenings;

    let columns = fixture_columns();
    let proof = prove_once(b"whir-identity", &columns);
    assert_fixture_is_not_degenerate(&proof);
    let before = identity_line(&proof).unwrap();

    let mut tampered = proof.clone();
    let round = &mut tampered.columns[0].polys[0].rounds[0];
    match &mut round.openings {
        RoundOpenings::Base(p) => p.current[0].values[0] += FieldElement::<Fp>::one(),
        RoundOpenings::Extension(p) => {
            p.current[0].values[0] += FieldElement::<Ext>::one();
        }
    }

    assert_ne!(
        before,
        identity_line(&tampered).unwrap(),
        "a changed opened codeword value must change the line"
    );
}

/// (3) ★ The exclusion, stated exactly: a nonce and nothing but a nonce.
#[test]
fn the_identity_line_is_blind_to_grinding_nonces_and_to_nothing_else() {
    let columns = fixture_columns();
    let proof = prove_once(b"whir-identity", &columns);
    assert_fixture_is_not_degenerate(&proof);
    let before = identity_line(&proof).unwrap();

    // A different, arbitrary nonce in every slot of one round.
    let mut renonced = proof.clone();
    {
        let nonces = &mut renonced.columns[0].polys[0].rounds[0].nonces;
        nonces.folding ^= 0xdead_beef;
        nonces.ood ^= 0x0bad_f00d;
        nonces.query ^= 0xfeed_face;
    }
    assert_ne!(
        renonced.columns[0].polys[0].rounds[0].nonces, proof.columns[0].polys[0].rounds[0].nonces,
        "the tamper must actually have changed the nonces"
    );
    assert_eq!(
        before,
        identity_line(&renonced).unwrap(),
        "the line excludes grinding nonces, by construction"
    );

    // …and the very next field along does move it, so the blindness is a
    // targeted exclusion rather than a broken digest.
    let mut moved = proof.clone();
    moved.columns[0].polys[0].final_value += FieldElement::<Ext>::one();
    assert_ne!(
        before,
        identity_line(&moved).unwrap(),
        "only the nonces are excluded"
    );
}

/// (4) The nonce normalisation does not change how long the proof is, so a
/// length comparison across arms measures the proof.
#[test]
fn the_serialized_length_does_not_depend_on_the_nonces() {
    let columns = fixture_columns();
    let proof = prove_once(b"whir-identity", &columns);
    assert_fixture_is_not_degenerate(&proof);

    let mut renonced = proof.clone();
    for stacked in &mut renonced.columns {
        for chain in &mut stacked.polys {
            for round in &mut chain.rounds {
                round.nonces.folding ^= u64::MAX;
                round.nonces.ood ^= u64::MAX;
                round.nonces.query ^= u64::MAX;
            }
        }
    }

    assert_eq!(
        serialized_len(&proof).unwrap(),
        serialized_len(&renonced).unwrap(),
        "a nonce is a fixed-width u64; the length cannot depend on its value"
    );
}
