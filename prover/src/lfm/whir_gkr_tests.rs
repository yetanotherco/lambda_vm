//! Gates for the GKR closure.
//!
//! The proof under test is a real one: `gkr::prove` over a real `FractionTree`,
//! so the layer relation actually holds and the emitted assert is exercised on
//! a proof that passes as well as on one that must not.

use crypto::fiat_shamir::default_transcript::DefaultTranscript;
use crypto::fiat_shamir::is_transcript::IsTranscript;
use multilinear::gkr::{FractionLayer, FractionTree, GkrProof, prove, verify};
use multilinear::mle::Mle;

use crate::tables::types::{FE, FEE, GoldilocksExtension};

use super::builder::{Ext, LfmBuilder};
use super::compiler::{LfmProgram, compile};
use super::executor::execute;
use super::validator::validate;
use super::whir_gkr::{
    GKR_SUMCHECK_DEGREE, GkrLayerChallenges, GkrLayerWires, emit_gkr_verify, gkr_verify_consts,
    gkr_verify_rows,
};
use super::word::{LfmWord, ext_word, word_as_ext};

/// A transcript that records what it hands out.
///
/// The GKR verifier's challenges are drawn interleaved with its sumchecks and
/// never returned, and re-deriving them would be a mirror of the host rather
/// than a reading of it. This observes instead: every call is delegated, and
/// `sample_field_element`'s answers are kept in order.
struct Recording {
    inner: DefaultTranscript<GoldilocksExtension>,
    sampled: Vec<FEE>,
}

impl Recording {
    fn new(label: &[u8]) -> Self {
        Self {
            inner: DefaultTranscript::<GoldilocksExtension>::new(label),
            sampled: Vec::new(),
        }
    }
}

impl IsTranscript<GoldilocksExtension> for Recording {
    fn append_field_element(&mut self, element: &FEE) {
        self.inner.append_field_element(element);
    }
    fn append_bytes(&mut self, new_bytes: &[u8]) {
        self.inner.append_bytes(new_bytes);
    }
    fn state(&self) -> [u8; 32] {
        self.inner.state()
    }
    fn sample_field_element(&mut self) -> FEE {
        let drawn = self.inner.sample_field_element();
        self.sampled.push(drawn);
        drawn
    }
    fn sample_u64(&mut self, upper_bound: u64) -> u64 {
        self.inner.sample_u64(upper_bound)
    }
}

fn fee(v: u64) -> FEE {
    FEE::new([
        FE::from(v),
        FE::from(v ^ 0x9E37),
        FE::from(v.wrapping_mul(31)),
    ])
}

/// A fraction tree over `num_vars` variables, with no zero denominator.
fn tree(num_vars: usize, seed: u64) -> FractionTree<GoldilocksExtension> {
    let size = 1usize << num_vars;
    let p = Mle::new((0..size).map(|i| fee(seed + i as u64)).collect()).expect("p is a cube");
    let q = Mle::new(
        (0..size)
            .map(|i| fee(seed.wrapping_mul(7) + 1 + i as u64))
            .collect(),
    )
    .expect("q is a cube");
    FractionTree::build(FractionLayer::new(p, q).expect("equal widths")).expect("the tree builds")
}

/// The arena the ladder reads: the output fraction, then per layer its
/// sumcheck evaluations, its four bound values, λ, its round challenges and
/// `c`. The program below hints in exactly this order.
fn gkr_arena_words(layers: usize) -> usize {
    2 + (0..layers)
        .map(|i| i * GKR_SUMCHECK_DEGREE + 4 + 1 + i + 1)
        .sum::<usize>()
}

/// Publics: `p`, `q`, and the point the ladder leaves.
fn gkr_public_words(layers: usize) -> usize {
    2 + if layers == 0 { 0 } else { layers }
}

fn gkr_program(layers: usize) -> LfmProgram {
    let mut b = LfmBuilder::new().with_wrap_hash(super::edsl::WrapHash::production());
    let arena = b.declare_arena(gkr_arena_words(layers) as u32);
    let mut idx = 0u32;
    let next = |b: &mut LfmBuilder, idx: &mut u32| -> Ext {
        let cell = b.hint_word(arena, *idx).as_ext();
        *idx += 1;
        cell
    };

    let output_p = next(&mut b, &mut idx);
    let output_q = next(&mut b, &mut idx);

    let mut wires = Vec::with_capacity(layers);
    let mut drawn = Vec::with_capacity(layers);
    for layer in 0..layers {
        let mut sumcheck = Vec::with_capacity(layer);
        for _ in 0..layer {
            let mut round = Vec::with_capacity(GKR_SUMCHECK_DEGREE);
            for _ in 0..GKR_SUMCHECK_DEGREE {
                round.push(next(&mut b, &mut idx));
            }
            sumcheck.push(round);
        }
        let p_lo = next(&mut b, &mut idx);
        let p_hi = next(&mut b, &mut idx);
        let q_lo = next(&mut b, &mut idx);
        let q_hi = next(&mut b, &mut idx);
        let lambda = next(&mut b, &mut idx);
        let mut rounds = Vec::with_capacity(layer);
        for _ in 0..layer {
            rounds.push(next(&mut b, &mut idx));
        }
        let c = next(&mut b, &mut idx);
        wires.push(GkrLayerWires {
            sumcheck,
            p_lo,
            p_hi,
            q_lo,
            q_hi,
        });
        drawn.push(GkrLayerChallenges { lambda, rounds, c });
    }

    let claim = emit_gkr_verify(&mut b, (output_p, output_q), &wires, &drawn);
    b.public(claim.p.as_cell());
    b.public(claim.q.as_cell());
    for coordinate in &claim.point {
        b.public(coordinate.as_cell());
    }

    let program = compile(b.finish());
    validate(&program).expect("the GKR leg must be admissible");
    program
}

/// The arena values, in the order [`gkr_program`] hints them. `sampled` is the
/// recorder's challenge stream, which `gkr::verify` draws as λ, then one per
/// sumcheck round, then `c`, layer by layer.
fn gkr_arena(
    output: (FEE, FEE),
    proof: &GkrProof<GoldilocksExtension>,
    sampled: &[FEE],
) -> Vec<LfmWord> {
    let mut words = vec![ext_word(&output.0), ext_word(&output.1)];
    let mut drawn = sampled.iter();
    for (layer, proof_layer) in proof.layers.iter().enumerate() {
        for round in &proof_layer.sumcheck.rounds {
            for value in &round.evaluations {
                words.push(ext_word(value));
            }
        }
        for value in [
            &proof_layer.p_lo,
            &proof_layer.p_hi,
            &proof_layer.q_lo,
            &proof_layer.q_hi,
        ] {
            words.push(ext_word(value));
        }
        // λ, one challenge per round, then c.
        for _ in 0..(layer + 2) {
            words.push(ext_word(drawn.next().expect("the recorder saw every draw")));
        }
    }
    assert!(
        drawn.next().is_none(),
        "the recorder's stream must be exactly the ladder's challenges"
    );
    words
}

/// ★ F1 for the ladder: every layer's rows, against the closed form.
#[test]
fn the_gkr_ladder_emits_its_closed_form() {
    for layers in [1usize, 2, 3, 6, 12] {
        let program = gkr_program(layers);
        let measured = program.instrs.len() - gkr_arena_words(layers) - gkr_public_words(layers);
        let predicted = gkr_verify_rows(layers) + gkr_verify_consts(layers);
        println!(
            "gkr over {layers:>2} layers: {measured:>5} rows emitted, {predicted:>5} predicted"
        );
        assert_eq!(
            measured, predicted,
            "a {layers}-layer ladder must emit its closed form"
        );
    }
}

/// ★ The leg computes what `gkr::verify` computes — all three parts of the
/// claim it returns, on a proof `gkr::prove` actually produced.
#[test]
fn the_gkr_leg_computes_what_the_host_computes() {
    for num_vars in [1usize, 2, 4] {
        let tree = tree(num_vars, 0x1234);
        let output = tree.output();

        let mut proving = Recording::new(b"v1-gkr-gate");
        let proved = prove(&tree, &mut proving).expect("the tree proves");

        let mut verifying = Recording::new(b"v1-gkr-gate");
        let host = verify(&proved.proof, output, &mut verifying).expect("the host accepts");

        let layers = proved.proof.layers.len();
        let program = gkr_program(layers);
        let arena = gkr_arena(output, &proved.proof, &verifying.sampled);
        let exec = execute(&program, &[arena], &crate::hash_pin::BLOCK_HASHER)
            .expect("the GKR leg executes on a proof the host accepts");

        let published: Vec<FEE> = exec
            .public_words
            .iter()
            .map(|(_, word)| word_as_ext(word).expect("a published extension value"))
            .collect();
        assert_eq!(published.len(), gkr_public_words(layers));
        assert_eq!(
            published[0], host.p,
            "{num_vars} variables: the p the ladder leaves"
        );
        assert_eq!(
            published[1], host.q,
            "{num_vars} variables: the q the ladder leaves"
        );
        assert_eq!(
            published[2..],
            host.point[..],
            "{num_vars} variables: the point the ladder leaves"
        );
    }
}

/// ★ THE TAMPER ARM. One corrupted proof value and the emitted program cannot
/// execute — the layer-relation assert has no satisfying assignment.
///
/// The forged arena keeps the HONEST challenge stream, which is the sharper
/// model of what a forger can do here: every input is hinted, so nothing stops
/// one from leaving the transcript alone and changing a proof value. (The host
/// cannot be asked for the forgery's challenges at all — it rejects at the
/// tampered layer and stops drawing, which is how this arm first failed.)
///
/// Three halves matter, not two: the untouched proof must EXECUTE, or the arm
/// is passing because the program never runs; the host must REJECT the same
/// forged proof under its own transcript, so the arm is about a proof that is
/// genuinely bad; and the machine must refuse it.
#[test]
fn a_corrupted_proof_value_cannot_execute() {
    let tree = tree(3, 0x99);
    let output = tree.output();
    let mut proving = Recording::new(b"v1-gkr-tamper");
    let proved = prove(&tree, &mut proving).expect("the tree proves");

    let mut verifying = Recording::new(b"v1-gkr-tamper");
    verify(&proved.proof, output, &mut verifying).expect("the control proof must verify");

    let layers = proved.proof.layers.len();
    let program = gkr_program(layers);
    let honest = gkr_arena(output, &proved.proof, &verifying.sampled);
    assert!(
        execute(&program, &[honest], &crate::hash_pin::BLOCK_HASHER).is_ok(),
        "the untouched proof must execute, or the arm below proves nothing"
    );

    for layer in [0usize, 1, layers - 1] {
        for which in ["p_lo", "a sumcheck evaluation"] {
            let mut forged = proved.proof.clone();
            match which {
                "p_lo" => forged.layers[layer].p_lo += FEE::one(),
                _ => match forged.layers[layer].sumcheck.rounds.first_mut() {
                    Some(round) => round.evaluations[0] += FEE::one(),
                    // Layer 0 has no sumcheck round to corrupt.
                    None => continue,
                },
            }

            let mut fresh = Recording::new(b"v1-gkr-tamper");
            assert!(
                verify(&forged, output, &mut fresh).is_err(),
                "layer {layer}, {which}: the host must reject the forgery"
            );

            let arena = gkr_arena(output, &forged, &verifying.sampled);
            assert!(
                execute(&program, &[arena], &crate::hash_pin::BLOCK_HASHER).is_err(),
                "layer {layer}, {which}: the machine must refuse the forgery too"
            );
        }
    }
}
