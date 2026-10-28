//! An LFM proof's commitment-time answers (`stark::commit_ood`), on the host
//! verifier and in the node's leg (`per_table_aggregator::emit_leg`), over the
//! Poseidon1 child of [`super::p1_node_tests`]: every sub-proof answers its
//! non-precomputed main columns; the leg draws `z0` and the shared LogUp pair
//! where the host does, after the answers; an answer altered, dropped or extra
//! is refused by both; a false answer the prover carries into its DEEP
//! polynomial is refused by both; and with the answers' DEEP quotients left
//! out, that false answer passes the host and a leg without the quotients but
//! not the leg with them.

use crypto::fiat_shamir::is_transcript::IsTranscript;
use stark::commit_ood::faults::{self, Faults};
use stark::proof::view::MultiProofView;

use crate::hash_pin::{P1Family, ProofFamily};
use crate::tables::types::FEE;

use super::builder::LfmBuilder;
use super::compiler::{LfmProgram, compile};
use super::executor::{LfmExecError, execute};
use super::harvest::try_child_arena_words;
use super::hash::HasherKind;
use super::p1_node_tests::{
    P1Child, arenas, harvest, node, options, p1_child, p1_child_with, refused_by_a_check, runs,
};
use super::per_table_aggregator::{DerivedChild, declare_leg_arenas, emit_leg_answered};
use super::proof::{LfmProof, lfm_prove_under, verify_against_artifacts_under};
use super::word::ext_word;

fn host_verifies(c: &P1Child, proof: &LfmProof) -> bool {
    verify_against_artifacts_under::<P1Family>(
        &c.artifacts,
        &proof.proof,
        &proof.public_words,
        &c.opts,
    )
}

/// The child at enough queries that a false answer's DEEP quotient — far from
/// every codeword — is caught by FRI with overwhelming probability, whatever
/// the grinding nonce makes the queries.
fn child_for_false_answers() -> P1Child {
    let mut opts = options();
    opts.fri_number_of_queries = 24;
    p1_child_with(opts)
}

/// The child's program proved again under `faults` (the prover's fault set,
/// read on this thread).
fn prove_with(c: &P1Child, faults: Faults) -> LfmProof {
    let program = super::programs::trivial_program();
    faults::with(faults, || {
        lfm_prove_under::<P1Family>(&program, &c.artifacts, &arenas(), &c.opts)
    })
    .expect("the child proves")
}

/// The child with `proof` in place of its honest one.
fn with_proof(c: &P1Child, proof: LfmProof) -> P1Child {
    P1Child {
        artifacts: c.artifacts.clone(),
        proof,
        opts: c.opts.clone(),
    }
}

/// One leg over the child that publishes its `z`, `α` and `z0`.
fn publishing_node(c: &P1Child) -> LfmProgram {
    let derived = DerivedChild::from_artifacts_under::<P1Family>(
        &c.artifacts,
        &c.opts,
        c.proof.public_words.len(),
    )
    .unwrap_or_else(|e| panic!("{e}"));
    let shape = derived.shape();
    let mut b = LfmBuilder::new().with_wrap_hash(P1Family::WRAP);
    let a = declare_leg_arenas(&mut b, &shape);
    let (cells, z0) = emit_leg_answered(&mut b, &shape, &a);
    for e in [cells.z_alpha.0, cells.z_alpha.1, z0] {
        b.public(e.as_cell());
    }
    compile(b.finish())
}

/// The host's `z0`: the statement, every root, then the draw over the
/// largest domain — the verifier's Phase A up to the answers.
fn host_z0(c: &P1Child) -> FEE {
    let airs = super::airs::LfmAirs::for_artifacts(&c.artifacts, &c.opts);
    let refs = airs.air_refs();
    let view = MultiProofView::Owned(&c.proof.proof.multi);
    let mut t = P1Family::transcript(&[]);
    super::statement::absorb_lfm_statement_tagged(
        &mut t,
        P1Family::STATEMENT_TAG,
        &c.artifacts.program_id,
        &c.proof.public_words,
        c.opts.fri_final_poly_log_degree,
    );
    for (idx, air) in refs.iter().enumerate() {
        let v = view.get(idx);
        if air.is_preprocessed() {
            t.append_bytes(
                &super::epoch_verify::layout_precomputed_commitment(*air, v.trace_length())
                    .expect("a preprocessed root"),
            );
        }
        t.append_bytes(v.lde_trace_main_merkle_root());
    }
    let domain = stark::commit_ood::point_domain(refs.iter().enumerate().map(|(idx, air)| {
        (
            view.get(idx).trace_length(),
            air.options().blowup_factor as usize,
            air.options().coset_offset,
        )
    }))
    .expect("one coset offset");
    stark::commit_ood::draw_point(&mut t, domain)
}

/// ★ Every sub-proof answers exactly its main columns past the precomputed
/// ones (the LFM program itself is precomputed and answers nothing), and the
/// proof verifies with them.
#[test]
fn every_lfm_sub_proof_answers_its_non_precomputed_main_columns() {
    let c = p1_child();
    assert!(host_verifies(&c, &c.proof));
    let airs = super::airs::LfmAirs::for_artifacts(&c.artifacts, &c.opts);
    let refs = airs.air_refs();
    assert_eq!(
        c.proof.proof.commit_ood.len(),
        refs.len(),
        "one list per sub-proof"
    );
    let mut total = 0;
    for (air, answers) in refs.iter().zip(&c.proof.proof.commit_ood) {
        let (main, _) = air.trace_layout();
        let precomputed = if air.is_preprocessed() {
            air.num_precomputed_columns()
        } else {
            0
        };
        assert_eq!(answers.len(), main - precomputed, "{}", air.name());
        assert!(!answers.is_empty(), "{} answers something", air.name());
        total += answers.len();
        println!(
            "COMMIT OOD ANSWERS {}: {} (main {main}, precomputed {precomputed})",
            air.name(),
            answers.len()
        );
    }
    println!("COMMIT OOD ANSWERS total: {total} ({} bytes)", 24 * total);
}

/// ★ The leg draws `z0` where the host does — after every root — and the
/// shared LogUp pair where the host does — after every answer: the honest
/// child executes and publishes the host's three values. A leg that absorbed
/// the answers after the pair would publish another `z`.
#[test]
fn the_leg_draws_z0_and_the_shared_pair_where_the_host_does() {
    let c = p1_child();
    let honest = harvest(&c);
    let program = publishing_node(&c);
    let arenas = try_child_arena_words(&honest).unwrap_or_else(|e| panic!("{e}"));
    let exec = execute(&program, &arenas, &HasherKind::Poseidon1W16).expect("the honest child");
    let published: Vec<_> = exec.public_words.iter().map(|(_, w)| *w).collect();
    let (z, alpha) = honest.z_alpha;
    assert_eq!(
        published,
        vec![ext_word(&z), ext_word(&alpha), ext_word(&host_z0(&c))],
        "z, α and z0 are the host's"
    );
}

/// An altered answer is refused by the host verifier and has no execution in
/// the leg, at every sub-proof.
#[test]
fn an_altered_answer_is_refused_on_both_sides() {
    let c = p1_child();
    let program = node(&c, P1Family::STATEMENT_TAG);
    assert!(runs(&program, &harvest(&c)), "the honest child executes");
    for t in 0..c.proof.proof.commit_ood.len() {
        let last = c.proof.proof.commit_ood[t].len() - 1;
        for i in [0, last] {
            let mut proof = LfmProof {
                proof: c.proof.proof.clone(),
                public_words: c.proof.public_words.clone(),
            };
            proof.proof.commit_ood[t][i] += FEE::one();
            assert!(!host_verifies(&c, &proof), "host: table {t} answer {i}");
            let mut bad = harvest(&c);
            bad.commit_ood[t][i] += FEE::one();
            assert!(
                refused_by_a_check(&program, &bad),
                "leg: table {t} answer {i}"
            );
        }
    }
}

/// A dropped or extra answer: the host refuses the count before absorbing
/// anything; the leg's answers arena is shape, so the arenas do not fit it.
#[test]
fn a_dropped_or_extra_answer_is_refused() {
    let c = p1_child();
    let program = node(&c, P1Family::STATEMENT_TAG);
    type Edit<'a> = (&'a str, &'a dyn Fn(&mut Vec<Vec<FEE>>));
    let edits: [Edit<'_>; 4] = [
        ("an answer dropped", &|a| {
            a[0].pop();
        }),
        ("an answer extra", &|a| a[1].push(FEE::zero())),
        ("a sub-proof's answers dropped", &|a| {
            a.pop();
        }),
        ("no answers", &|a| a.clear()),
    ];
    for (what, edit) in edits {
        let mut proof = LfmProof {
            proof: c.proof.proof.clone(),
            public_words: c.proof.public_words.clone(),
        };
        edit(&mut proof.proof.commit_ood);
        assert!(!host_verifies(&c, &proof), "host: {what}");
        let mut bad = harvest(&c);
        edit(&mut bad.commit_ood);
        let arenas = try_child_arena_words(&bad).unwrap_or_else(|e| panic!("{e}"));
        assert!(
            matches!(
                execute(&program, &arenas, &HasherKind::Poseidon1W16),
                Err(LfmExecError::ArenaLenMismatch { .. })
            ),
            "leg: {what}"
        );
    }
}

/// A false answer the prover absorbs and carries into its DEEP polynomial (a
/// consistent proof of the false value) is refused by the host and has no
/// execution in the leg.
#[test]
fn a_false_answer_carried_into_deep_is_refused_on_both_sides() {
    let c = child_for_false_answers();
    let program = node(&c, P1Family::STATEMENT_TAG);
    let last = c.proof.proof.commit_ood.len() - 1;
    for at in [(0, 0), (last, 0)] {
        let false_child = with_proof(
            &c,
            prove_with(
                &c,
                Faults {
                    false_answer: Some(at),
                    without_deep_term: false,
                },
            ),
        );
        assert!(
            !host_verifies(&false_child, &false_child.proof),
            "host: {at:?}"
        );
        assert!(
            refused_by_a_check(&program, &harvest(&false_child)),
            "leg: {at:?}"
        );
    }
}

/// ★ Load-bearing, both sides: a false answer proved with the answers' DEEP
/// quotients left out passes the host verifier that leaves them out too, and
/// a leg emitted without them; the leg with them refuses it. So the quotients
/// are what refuse the false answer above, on the host and in the leg.
#[test]
fn the_answers_quotients_are_what_refuse_a_false_answer() {
    let c = child_for_false_answers();
    let fault = Faults {
        false_answer: Some((0, 0)),
        without_deep_term: true,
    };
    let false_child = with_proof(&c, prove_with(&c, fault));
    assert!(
        faults::with(fault, || host_verifies(&false_child, &false_child.proof)),
        "host without the quotients accepts it"
    );
    assert!(
        !host_verifies(&false_child, &false_child.proof),
        "host with them refuses it"
    );
    let child = harvest(&false_child);
    let with_term = node(&c, P1Family::STATEMENT_TAG);
    assert!(refused_by_a_check(&with_term, &child), "the leg refuses it");
    super::deep::WITHOUT_ANSWERS_TERM.with(|w| w.set(true));
    let without_term = node(&c, P1Family::STATEMENT_TAG);
    super::deep::WITHOUT_ANSWERS_TERM.with(|w| w.set(false));
    assert!(
        runs(&without_term, &child),
        "a leg without the quotients accepts it"
    );
}
