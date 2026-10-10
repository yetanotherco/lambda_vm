//! The block's LFM proofs under the block pin (P3b, D-P3B-1013 §5 T2–T5).
//!
//! The no-epoch block's recursion commits, transcribes and grinds its LFM
//! proofs under [`crate::hash_pin::Block`] (Poseidon1); the legacy paths keep
//! [`crate::hash_pin::Legacy`] (RPX). These tests pin, on one small `Hash16`
//! program at both configurations:
//!
//! - the LFM statement tag each configuration absorbs first;
//! - the program id naming the hash its roots were committed with;
//! - a block LFM proof that round-trips, refused by the legacy pin and the
//!   reverse, and a prove or verify whose options or artifacts name another
//!   hash refused before any work;
//! - a tampered block LFM proof refused (root, opening sibling, FRI sibling,
//!   final polynomial, nonce);
//! - the tag absorbed: a proof made without it is accepted only by a verifier
//!   that also drops it;
//! - in-guest (T6): a node under the block pin, derived from its children's
//!   artifacts as production derives it, verifies two block children when
//!   executed; each tampered arena (a root, an opening, a FRI path, the nonce,
//!   an OOD value, the published words) and a legacy child are refused.

use std::sync::Mutex;
use std::sync::atomic::Ordering;

use stark::config::CommitmentHash;
use stark::proof::options::{GoldilocksCubicProofOptions, ProofFormat, ProofOptions};
use stark::prover::ProvingError;

use crate::hash_pin::{
    BLOCK_LFM, BLOCK_LFM_CAP, BLOCK_SOCKET, BLOCK_WRAP, Block, BlockHash, Legacy,
};
use crate::tables::types::{FE, FEE, GoldilocksExtension, GoldilocksField};

use super::builder::{Cell, LfmBuilder};
use super::compiler::{LfmProgram, compile};
use super::executor::execute;
use super::harvest::{HarvestedChild, harvest_child_verified_under, try_child_arena_words};
use super::hash::HasherKind;
use super::per_table_aggregator::{DerivedChild, declare_leg_arenas, emit_leg};
use super::proof::{
    LfmProof, LfmProveError, lfm_prove, lfm_prove_under, verify_against_artifacts,
    verify_against_artifacts_under,
};
use super::registry::{LfmArtifacts, build_artifacts_with_hasher};
use super::statement::lfm_program_id_chunked;

/// The tag mutation is process-global; the tests that prove under the block
/// pin take this in turn so none of them proves while another drops the tag.
static SERIAL: Mutex<()> = Mutex::new(());

const KIND: HasherKind = HasherKind::Poseidon1W16;

/// Blowup 4 as the block's LFM proofs, with a short grind so the host search
/// stays cheap; the nonce is still there to tamper.
fn legacy_options() -> ProofOptions {
    GoldilocksCubicProofOptions::with_params(4, 128, 8).expect("options")
}

/// [`legacy_options`] under the block's LFM format.
fn block_options() -> ProofOptions {
    let mut o = legacy_options();
    o.format.base = BLOCK_LFM;
    o
}

fn word(seed: u64) -> [FE; 4] {
    core::array::from_fn(|i| FE::from(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ i as u64))
}

/// Two chained width-16 permutations, every output of the second published:
/// a block program's socket rows in miniature.
fn program() -> LfmProgram {
    let mut b = LfmBuilder::new();
    let ins: [Cell; 4] = core::array::from_fn(|k| b.digest_const(word(k as u64 + 1)).as_cell());
    let first = b.hash16(ins);
    let zero = b.digest_const([FE::zero(); 4]).as_cell();
    let second = b.hash16([first[0], zero, first[0], ins[3]]);
    for c in second {
        b.public(c);
    }
    let program = compile(b.finish());
    super::validator::validate(&program).expect("admissible");
    program
}

fn artifacts(opts: &ProofOptions) -> LfmArtifacts {
    build_artifacts_with_hasher(&program(), opts, KIND)
}

fn prove_block(artifacts: &LfmArtifacts, opts: &ProofOptions) -> Result<LfmProof, LfmProveError> {
    lfm_prove_under::<Block>(&program(), artifacts, &[], opts)
}

#[test]
fn the_lfm_statement_tags_name_the_configuration() {
    let legacy = ProofFormat::LEGACY;
    let block = block_options().format;
    assert_eq!(
        Legacy::lfm_statement_tag(&legacy),
        b"LAMBDAVM_LFM_STATEMENT_V1".to_vec(),
        "the legacy LFM tag is today's, byte for byte"
    );
    assert_eq!(
        Block::lfm_statement_tag(&block),
        format!("LAMBDAVM_LFM_STATEMENT_V1/P1W16/C{BLOCK_LFM_CAP}").into_bytes(),
        "the block's LFM tag names the hash and the cap"
    );
    assert_eq!(
        crate::hash_pin::lfm_statement_tag(&legacy),
        Legacy::lfm_statement_tag(&legacy)
    );
    assert_eq!(
        crate::hash_pin::lfm_statement_tag(&block),
        Block::lfm_statement_tag(&block)
    );
}

#[test]
fn the_program_id_names_the_commitment_hash() {
    let legacy = artifacts(&legacy_options());
    let block = artifacts(&block_options());
    assert_eq!(legacy.commitment, crate::hash_pin::LEGACY_COMMITMENT_HASH);
    assert_eq!(block.commitment, CommitmentHash::Poseidon1);
    assert_ne!(legacy.roots, block.roots, "the roots are another hash's");
    assert_ne!(legacy.program_id, block.program_id);
    // The tag alone moves the id: the block's roots named as the legacy hash's
    // are another program.
    let renamed = lfm_program_id_chunked(
        &block.roots,
        &block.log_heights,
        block.keccak_rnd_chunks,
        block.hasher,
        block.chip_set,
        &block.blake3_chunk_roots,
        &block.blake3_chunk_log_heights,
        &block.hash_chunk_roots,
        &block.hash_chunk_log_heights,
        crate::hash_pin::LEGACY_COMMITMENT_HASH,
    );
    assert_ne!(renamed, block.program_id);
}

#[test]
fn a_block_lfm_proof_round_trips_and_refuses_the_legacy_pin() {
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let (lopts, bopts) = (legacy_options(), block_options());
    let (lart, bart) = (artifacts(&lopts), artifacts(&bopts));
    let program = program();
    let words = execute(&program, &[], &KIND).expect("execute").public_words;

    let proof = prove_block(&bart, &bopts).expect("the block pin proves");
    assert!(
        verify_against_artifacts_under::<Block>(&bart, &proof.proof, &words, &bopts),
        "the block pin verifies its proof"
    );
    let legacy_proof = lfm_prove(&program, &lart, &[], &lopts).expect("the legacy pin proves");
    assert!(verify_against_artifacts(
        &lart,
        &legacy_proof.proof,
        &words,
        &lopts
    ));

    // Each refused by the other's verifier, by the configuration check …
    assert!(!verify_against_artifacts(
        &bart,
        &proof.proof,
        &words,
        &bopts
    ));
    assert!(!verify_against_artifacts(
        &lart,
        &proof.proof,
        &words,
        &lopts
    ));
    assert!(!verify_against_artifacts_under::<Block>(
        &lart,
        &legacy_proof.proof,
        &words,
        &lopts
    ));
    // … and by the proof itself: a legacy proof read as a block one, with the
    // block's artifacts and options (the check passes), fails the verify.
    assert!(!verify_against_artifacts_under::<Block>(
        &bart,
        &legacy_proof.proof,
        &words,
        &bopts
    ));

    // Options or artifacts naming another hash: refused before any work.
    for (art, opts, what) in [
        (&lart, &bopts, "legacy roots under the block's options"),
        (&bart, &lopts, "the block's roots under legacy options"),
    ] {
        assert!(
            matches!(
                prove_block(art, opts),
                Err(LfmProveError::Prover(ProvingError::WrongParameter(_)))
            ),
            "{what}"
        );
        assert!(
            !verify_against_artifacts_under::<Block>(art, &proof.proof, &words, opts),
            "{what}"
        );
    }
    assert!(matches!(
        lfm_prove(&program, &bart, &[], &bopts),
        Err(LfmProveError::Prover(ProvingError::WrongParameter(_)))
    ));
}

#[test]
fn a_tampered_block_lfm_proof_is_refused() {
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let bopts = block_options();
    let bart = artifacts(&bopts);
    let words = execute(&program(), &[], &KIND)
        .expect("execute")
        .public_words;
    let honest = prove_block(&bart, &bopts).expect("the block pin proves");
    assert!(verify_against_artifacts_under::<Block>(
        &bart,
        &honest.proof,
        &words,
        &bopts
    ));
    // The sub-proof with the longest opening paths: every mutation below lands
    // on a real sibling there.
    let k = (0..honest.proof.proofs.len())
        .max_by_key(|&k| {
            honest.proof.proofs[k].deep_poly_openings[1]
                .main_trace_polys
                .proof
                .merkle_path
                .len()
        })
        .expect("sub-proofs");
    type Proof = stark::proof::stark::MultiProof<GoldilocksField, GoldilocksExtension, ()>;
    type Mutation = (&'static str, fn(&mut Proof, usize));
    let mutations: [Mutation; 5] = [
        ("the main root", |p, k| {
            p.proofs[k].lde_trace_main_merkle_root[3] ^= 1
        }),
        ("a main-trace opening sibling", |p, k| {
            p.proofs[k].deep_poly_openings[1]
                .main_trace_polys
                .proof
                .merkle_path[0][5] ^= 1
        }),
        ("a FRI layer sibling", |p, k| {
            p.proofs[k].query_list[1].layers_auth_paths[0].merkle_path[0][7] ^= 1
        }),
        ("the final polynomial", |p, k| {
            p.proofs[k].fri_final_poly_coeffs[0] += FEE::one()
        }),
        ("the nonce", |p, k| {
            p.proofs[k].nonce = p.proofs[k].nonce.map(|n| n ^ 1);
        }),
    ];
    for (what, mutate) in mutations {
        let mut forged = honest.proof.clone();
        mutate(&mut forged, k);
        assert!(
            !verify_against_artifacts_under::<Block>(&bart, &forged, &words, &bopts),
            "a tampered {what} was accepted"
        );
    }
}

#[test]
fn the_block_lfm_statement_tag_is_absorbed() {
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let bopts = block_options();
    let bart = artifacts(&bopts);
    let words = execute(&program(), &[], &KIND)
        .expect("execute")
        .public_words;
    let honest = prove_block(&bart, &bopts).expect("the block pin proves");
    // Mutation: the block's statements under the legacy tag.
    crate::hash_pin::P1_LFM_TAG_OMITTED.store(true, Ordering::SeqCst);
    let untagged = prove_block(&bart, &bopts);
    let accepted_without_tag = untagged
        .as_ref()
        .is_ok_and(|p| verify_against_artifacts_under::<Block>(&bart, &p.proof, &words, &bopts));
    let honest_without_tag =
        verify_against_artifacts_under::<Block>(&bart, &honest.proof, &words, &bopts);
    crate::hash_pin::P1_LFM_TAG_OMITTED.store(false, Ordering::SeqCst);
    let untagged = untagged.expect("the block pin proves without its tag");
    assert!(
        accepted_without_tag,
        "a verifier that also drops the tag accepts it"
    );
    assert!(
        !honest_without_tag,
        "a verifier without the tag accepted a tagged proof"
    );
    assert!(
        !verify_against_artifacts_under::<Block>(&bart, &untagged.proof, &words, &bopts),
        "the block's verifier accepted a proof without its tag"
    );
}

/// A node under the block pin over `n` copies of `child`, emitted as the block
/// tree emits its nodes: each child's arenas declared, then its leg (the LFM
/// statement under the child's tag, Phase A, every sub-proof, the closure).
fn node_over(child: &DerivedChild, n: usize) -> LfmProgram {
    let mut b = LfmBuilder::new().with_wrap_hash(BLOCK_WRAP);
    let shape = child.shape();
    let arenas: Vec<_> = (0..n).map(|_| declare_leg_arenas(&mut b, &shape)).collect();
    for a in &arenas {
        let leg = emit_leg(&mut b, &shape, a);
        b.public(leg.publics[0].lanes[0].as_cell());
    }
    let program = compile(b.finish());
    super::validator::validate(&program).expect("admissible");
    assert!(
        program.hash16(),
        "a block node hashes on the width-16 socket"
    );
    program
}

/// The arenas' names, in `try_child_arena_words`' order.
fn arena_labels(c: &HarvestedChild) -> Vec<String> {
    let mut out = vec!["publics".to_string(), "main roots".to_string()];
    for (t, (h, leg)) in c.tables.iter().zip(&c.legs).enumerate() {
        let mut push = |what: &str| out.push(format!("t{t} {what}"));
        if h.aux_root.is_some() {
            push("aux root");
        }
        if h.contribution.is_some() {
            push("contribution");
        }
        for what in [
            "composition root",
            "ood current",
            "ood next",
            "parts",
            "fri roots",
            "fri coeffs",
        ] {
            push(what);
        }
        if h.nonce.is_some() {
            push("nonce");
        }
        push("opening");
        push("fri");
        if leg.try_caps_arena().expect("caps").is_some() {
            push("caps");
        }
    }
    out
}

#[test]
fn a_block_node_verifies_block_children_and_refuses_tampers() {
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let (lopts, bopts) = (legacy_options(), block_options());
    let (lart, bart) = (artifacts(&lopts), artifacts(&bopts));
    let program = program();
    let proved = prove_block(&bart, &bopts).expect("the block pin proves");
    let (child, _) = harvest_child_verified_under::<Block>(bart.clone(), bopts.clone(), &proved)
        .expect("the block child harvests");
    let derived = DerivedChild::from_artifacts(&bart, &bopts, proved.public_words.len())
        .expect("the child derives from its artifacts");
    let node = node_over(&derived, 2);
    let one = try_child_arena_words(&child).expect("the child's arenas");
    let labels = arena_labels(&child);
    assert_eq!(labels.len(), one.len(), "one name per arena");
    let both = |a: &[Vec<super::word::LfmWord>]| -> Vec<Vec<super::word::LfmWord>> {
        a.iter().chain(&one).cloned().collect()
    };
    assert!(
        execute(&node, &both(&one), &BLOCK_SOCKET).is_ok(),
        "a block node accepts two honest block children"
    );

    // Each arena kind tampered in the sub-proof where it is longest, so the
    // tamper lands on a real sibling, FRI path or coefficient.
    let longest = |what: &str| -> String {
        (0..child.tables.len())
            .map(|t| format!("t{t} {what}"))
            .filter_map(|l| {
                labels
                    .iter()
                    .position(|x| *x == l)
                    .map(|at| (one[at].len(), l))
            })
            .max()
            .filter(|(len, _)| *len > 0)
            .unwrap_or_else(|| panic!("no sub-proof has a {what} arena"))
            .1
    };
    let mut targets = vec!["publics".to_string(), "main roots".to_string()];
    for what in [
        "composition root",
        "ood current",
        "fri roots",
        "fri coeffs",
        "nonce",
        "opening",
        "fri",
        "caps",
    ] {
        targets.push(longest(what));
    }
    for what in targets {
        let at = labels
            .iter()
            .position(|l| *l == what)
            .expect("a named arena");
        let mut forged = one.clone();
        assert!(!forged[at].is_empty(), "the {what} arena is empty");
        let word = forged[at].len() / 2;
        forged[at][word][0] += FE::one();
        assert!(
            execute(&node, &both(&forged), &BLOCK_SOCKET).is_err(),
            "a tampered {what} was accepted in-guest"
        );
    }

    // A legacy child (RPX-committed) of the same program, fed to the block node.
    let legacy_proof = lfm_prove(&program, &lart, &[], &lopts).expect("the legacy pin proves");
    let (legacy_child, _) = super::harvest::harvest_child_verified(lart, lopts, &legacy_proof)
        .expect("the legacy child harvests");
    let legacy = try_child_arena_words(&legacy_child).expect("the legacy arenas");
    assert!(
        execute(&node, &both(&legacy), &BLOCK_SOCKET).is_err(),
        "a block node accepted a legacy child"
    );
}
