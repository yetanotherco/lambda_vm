use crypto::fiat_shamir::is_transcript::{IsStarkTranscript, IsTranscript};
use math::field::element::FieldElement;
use math::field::traits::{IsFFTField, IsField, IsSubFieldOf};
use math::traits::AsBytes;

use crate::config::{FriLayerMerkleTree, FriLayerMerkleTreeBackend};
use crate::fri::fri_commitment::FriLayer;
use crate::fri::fri_functions::{
    compute_coset_twiddles_inv, fold_evaluations_in_place, update_twiddles_in_place,
};

/// Combine DEEP polynomial codewords by their FRI height for batched FRI.
///
/// Each element of `inputs` is a pair `(codeword, height)` where `height` is
/// the log₂ of the codeword length (i.e. `codeword.len() == 2^height`).
/// The global index `i` into `inputs` is used to derive the mixing power
/// `alpha^i` (index 0 → alpha^0 = 1, index 1 → alpha^1, …).
///
/// Returns a `Vec` of length `max_height + 1`.  Index `h` contains
/// `Some(combined)` where `combined[j] = Σ_{i : height_i == h} alpha^i * codeword_i[j]`,
/// or `None` when no input has height `h`.
pub fn combine_by_height<E>(
    inputs: &[(Vec<FieldElement<E>>, usize)],
    alpha: &FieldElement<E>,
) -> Vec<Option<Vec<FieldElement<E>>>>
where
    E: IsField,
    FieldElement<E>: Clone,
{
    if inputs.is_empty() {
        return vec![];
    }

    let max_height = inputs
        .iter()
        .map(|(_, h)| *h)
        .max()
        .expect("inputs is non-empty so max height exists");

    let mut out: Vec<Option<Vec<FieldElement<E>>>> = vec![None; max_height + 1];

    // Precompute alpha^0, alpha^1, …, alpha^(n-1) via repeated multiplication.
    let mut alpha_pows: Vec<FieldElement<E>> = Vec::with_capacity(inputs.len());
    let mut cur = FieldElement::one();
    for _ in 0..inputs.len() {
        alpha_pows.push(cur.clone());
        cur = &cur * alpha;
    }

    for (i, (codeword, height)) in inputs.iter().enumerate() {
        let h = *height;
        let expected_len = 1usize << h;
        assert_eq!(
            codeword.len(),
            expected_len,
            "codeword at index {i} has length {} but height {h} expects {expected_len}",
            codeword.len()
        );

        let a_i = &alpha_pows[i];

        match &mut out[h] {
            None => {
                let combined: Vec<FieldElement<E>> = codeword.iter().map(|x| a_i * x).collect();
                out[h] = Some(combined);
            }
            Some(acc) => {
                for (j, x) in codeword.iter().enumerate() {
                    acc[j] = &acc[j] + &(a_i * x);
                }
            }
        }
    }

    out
}

/// FRI commit phase that operates on the bucketed output of [`combine_by_height`].
///
/// `combined[h]` is `Some(codeword)` when there are DEEP polynomial contributions
/// at height `h` (i.e., with codeword length `2^h`), or `None` otherwise.
///
/// The function starts from the tallest bucket (index `h_max`) and folds
/// downward.  After each fold to height `h`, any bucket at `combined[h]` is
/// *injected* into the running codeword with coefficient `β²` (where `β` is
/// the current folding challenge), before the layer is committed to the
/// transcript.  Termination mirrors [`crate::fri::commit_phase_from_evaluations`]:
/// the last fold produces a single scalar (`last_value`) that is appended to
/// the transcript rather than committed as a Merkle tree layer.
pub fn batched_commit_phase<F, E, T>(
    mut combined: Vec<Option<Vec<FieldElement<E>>>>,
    transcript: &mut T,
    coset_offset: &FieldElement<F>,
) -> (
    FieldElement<E>,
    Vec<FriLayer<E, FriLayerMerkleTreeBackend<E>>>,
)
where
    F: IsFFTField + IsSubFieldOf<E> + 'static,
    E: IsField + 'static,
    T: IsStarkTranscript<E, F> + Clone,
    FieldElement<F>: AsBytes + Sync + Send,
    FieldElement<E>: AsBytes + Sync + Send,
{
    // Find h_max: the largest index that has a Some(codeword).
    let h_max = combined
        .iter()
        .enumerate()
        .rev()
        .find_map(|(h, slot)| if slot.is_some() { Some(h) } else { None })
        .expect("batched_commit_phase: combined must have at least one Some entry");

    // Take the starting codeword — NOT committed; it plays the role of layer 0.
    let mut running = combined[h_max]
        .take()
        .expect("combined[h_max] is Some by construction");

    let domain_size = 1usize << h_max;
    debug_assert_eq!(
        running.len(),
        domain_size,
        "starting codeword length must equal 2^h_max"
    );

    // Inverse twiddle factors for the initial domain size.
    let mut inv_twiddles = compute_coset_twiddles_inv(coset_offset, domain_size);

    // Commit (h_max − 1) layers; one final fold yields the last_value scalar.
    let num_committed_layers = h_max.saturating_sub(1);
    let mut fri_layer_list = Vec::with_capacity(num_committed_layers);

    for _ in 0..num_committed_layers {
        // <<<< Receive challenge β
        let beta = transcript.sample_field_element();

        // Fold evaluations in-place; running halves in length.
        fold_evaluations_in_place(&mut running, &beta, &inv_twiddles);

        // Height h of the folded codeword.
        let h = running.len().trailing_zeros() as usize;

        // Inject oracle at height h (if any): running[j] += β² · ro[j]
        if let Some(ro) = combined[h].take() {
            let beta_sq = beta.square();
            for (j, val) in ro.iter().enumerate() {
                running[j] = &running[j] + &(&beta_sq * val);
            }
        }

        // Build the row-pair Merkle tree over the current running codeword.
        let leaves: Vec<[FieldElement<E>; 2]> = running
            .chunks_exact(2)
            .map(|chunk| [chunk[0].clone(), chunk[1].clone()])
            .collect();
        let merkle_tree = FriLayerMerkleTree::build(&leaves)
            .expect("FRI batched commit: Merkle tree construction must succeed");
        let root = merkle_tree.root;
        fri_layer_list.push(FriLayer::new(&running, merkle_tree));

        // >>>> Send commitment: append root to transcript.
        transcript.append_bytes(&root);

        // Update twiddles for the next (halved) level.
        update_twiddles_in_place(&mut inv_twiddles);
    }

    // <<<< Receive the final folding challenge.
    let beta = transcript.sample_field_element();

    // Final fold: running goes from length 2 → length 1.
    fold_evaluations_in_place(&mut running, &beta, &inv_twiddles);

    let last_value = running
        .first()
        .expect("FRI evals are non-empty after final fold")
        .clone();

    // >>>> Send value: append the last scalar to the transcript.
    transcript.append_field_element(&last_value);

    (last_value, fri_layer_list)
}

/// Canonical, order-deterministic absorption of an epoch's table-height histogram
/// into the transcript. Single source of truth for the structural binding: the
/// multiset of `lde_log_height`s across an epoch's tables fully determines the fold
/// order and injection points of the batched FRI (arity is uniformly 2 — #729 is not
/// on this branch, see the task-3 brief override). Binding this histogram is
/// therefore equivalent to binding the whole arity/injection schedule.
///
/// Encoding (fixed-width, length-prefixed, order-preserving):
/// `u64::to_le_bytes(heights.len())` followed by `u64::to_le_bytes(h)` for each `h`
/// in `heights`, in the exact order given. Caller (prover and verifier alike) must
/// pass `heights` in the same canonical per-epoch table order — this function does
/// not sort or deduplicate.
pub fn absorb_height_histogram<E, T>(transcript: &mut T, heights: &[usize])
where
    E: IsField,
    T: IsTranscript<E>,
{
    transcript.append_bytes(&(heights.len() as u64).to_le_bytes());
    for h in heights {
        transcript.append_bytes(&(*h as u64).to_le_bytes());
    }
}

/// Challenges derived from replaying the shared batched round-4 transcript sequence.
/// See [`derive_batched_fri_challenges`].
#[derive(Debug, Clone)]
pub struct BatchedFriChallenges<E: IsField> {
    /// Sampled once after the height histogram (and, at the call site, after all
    /// per-table OOD evaluations have been absorbed).
    pub alpha: FieldElement<E>,
    /// One per committed layer plus one final challenge for the last fold.
    /// `betas.len() == layer_roots.len() + 1`.
    pub betas: Vec<FieldElement<E>>,
    /// Transcript state right before the grinding nonce bytes are appended.
    /// All-zero when `grinding_factor == 0` or `nonce` is `None` (mirrors
    /// the per-table convention in `verifier.rs`).
    pub grinding_seed: [u8; 32],
    /// One `sample_u64(domain_size >> 1)` draw per query.
    pub iotas: Vec<usize>,
}

/// Replays the shared batched round-4 transcript sequence (histogram, alpha,
/// per-layer beta/root, final beta/last_value, grinding, query iotas) and returns
/// the derived challenges. The single routine the prover (T4) and verifier (T5)
/// both call so they provably derive identical challenges; calls
/// `absorb_height_histogram` internally instead of duplicating the encoding.
#[allow(clippy::too_many_arguments)]
pub fn derive_batched_fri_challenges<E, T>(
    transcript: &mut T,
    heights: &[usize],
    layer_roots: &[[u8; 32]],
    last_value: &FieldElement<E>,
    grinding_factor: u8,
    nonce: Option<u64>,
    num_queries: usize,
    domain_size: usize,
) -> BatchedFriChallenges<E>
where
    E: IsField,
    T: IsTranscript<E>,
{
    absorb_height_histogram(transcript, heights);

    let alpha = transcript.sample_field_element();

    let mut betas = Vec::with_capacity(layer_roots.len() + 1);
    for root in layer_roots {
        let beta = transcript.sample_field_element();
        transcript.append_bytes(root);
        betas.push(beta);
    }

    let final_beta = transcript.sample_field_element();
    transcript.append_field_element(last_value);
    betas.push(final_beta);

    let mut grinding_seed = [0u8; 32];
    if grinding_factor > 0
        && let Some(nonce_value) = nonce
    {
        grinding_seed = transcript.state();
        transcript.append_bytes(&nonce_value.to_be_bytes());
    }

    let iotas = (0..num_queries)
        .map(|_| transcript.sample_u64((domain_size as u64) >> 1) as usize)
        .collect();

    BatchedFriChallenges {
        alpha,
        betas,
        grinding_seed,
        iotas,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fri::fri_functions::{compute_coset_twiddles_inv, fold_evaluations_in_place};
    use crypto::fiat_shamir::default_transcript::DefaultTranscript;
    use crypto::fiat_shamir::is_transcript::IsTranscript;
    use math::field::element::FieldElement;
    use math::field::goldilocks::GoldilocksField;

    type FE = FieldElement<GoldilocksField>;
    type Transcript = DefaultTranscript<GoldilocksField>;

    #[test]
    fn combine_by_height_two_height3_one_height2() {
        // Three codewords: indices 0, 1 have height 3 (length 8);
        //                  index 2 has height 2 (length 4).
        let cw0: Vec<FE> = (1u64..=8).map(FE::from).collect();
        let cw1: Vec<FE> = (10u64..=17).map(FE::from).collect();
        let cw2: Vec<FE> = (100u64..=103).map(FE::from).collect();

        let alpha = FE::from(7u64);

        let inputs: Vec<(Vec<FE>, usize)> =
            vec![(cw0.clone(), 3), (cw1.clone(), 3), (cw2.clone(), 2)];

        let out = combine_by_height(&inputs, &alpha);

        // Output vec length = max_height + 1 = 4 (indices 0..=3 only).
        assert_eq!(out.len(), 4, "output length should be max_height+1 = 4");

        // Heights 0 and 1 have no inputs.
        assert!(out[0].is_none(), "height 0 should be None");
        assert!(out[1].is_none(), "height 1 should be None");

        // Height 3: combined[j] = alpha^0 * cw0[j] + alpha^1 * cw1[j]
        let alpha0 = FE::one();
        let alpha1 = alpha.clone();
        let expected3: Vec<FE> = cw0
            .iter()
            .zip(cw1.iter())
            .map(|(a, b)| &(&alpha0 * a) + &(&alpha1 * b))
            .collect();

        let got3 = out[3].as_ref().expect("height 3 should be Some");
        assert_eq!(
            got3.len(),
            8,
            "height-3 combined codeword should have length 8"
        );
        assert_eq!(got3, &expected3, "height-3 combined values mismatch");

        // Height 2: combined[j] = alpha^2 * cw2[j]
        let alpha2 = &alpha * &alpha;
        let expected2: Vec<FE> = cw2.iter().map(|x| &alpha2 * x).collect();

        let got2 = out[2].as_ref().expect("height 2 should be Some");
        assert_eq!(
            got2.len(),
            4,
            "height-2 combined codeword should have length 4"
        );
        assert_eq!(got2, &expected2, "height-2 combined values mismatch");
    }

    /// Verify that after the first fold in `batched_commit_phase`:
    ///   - The committed layer[0] evaluation equals fold(combined[4], β₀) + β₀² · combined[3]
    ///   - The total number of committed layers is h_max − 1 = 3
    #[test]
    fn batched_commit_phase_first_layer_matches_manual_fold_and_inject() {
        // Build synthetic codewords for h=4 (len 16) and h=3 (len 8).
        let data_h4: Vec<FE> = (1u64..=16).map(FE::from).collect();
        let data_h3: Vec<FE> = (101u64..=108).map(FE::from).collect();

        // combined = [None, None, None, Some(data_h3), Some(data_h4)]
        let combined: Vec<Option<Vec<FE>>> = vec![
            None,
            None,
            None,
            Some(data_h3.clone()),
            Some(data_h4.clone()),
        ];

        let coset_offset = FE::from(3u64);

        // Create transcript; clone before mutating so we can replay independently.
        let mut transcript = Transcript::new(b"batched_fri_test");
        let mut transcript_check = transcript.clone();

        // Run the commit phase.
        let (_last_val, layers) = batched_commit_phase(combined, &mut transcript, &coset_offset);

        // h_max = 4, so we expect h_max − 1 = 3 committed layers.
        assert_eq!(
            layers.len(),
            3,
            "expected 3 committed FRI layers for h_max=4"
        );

        // --- Independent recomputation of layer[0] ---

        // The first beta is the first thing sampled from the transcript.
        let beta_0 = transcript_check.sample_field_element();

        // Fold data_h4 with beta_0 using the same twiddles as the function.
        let inv_twiddles_h4 = compute_coset_twiddles_inv::<GoldilocksField>(&coset_offset, 16);
        let mut expected = data_h4.clone();
        fold_evaluations_in_place(&mut expected, &beta_0, &inv_twiddles_h4);
        // expected now has length 8 (height 3)

        // Inject combined[3]: expected[j] += beta_0² · data_h3[j]
        let beta_0_sq = beta_0.square();
        for (j, val) in data_h3.iter().enumerate() {
            expected[j] = &expected[j] + &(&beta_0_sq * val);
        }

        // The first committed layer's evaluation vector must match.
        assert_eq!(
            layers[0].evaluation, expected,
            "layer[0] evaluation does not match manual fold+inject"
        );
    }

    /// The prover, by hand, runs exactly the sequence documented in the design spec's
    /// "Transcript binding" section (restricted to round 4, batched-arity-2, no
    /// final_poly / no log_arities — see task-3 override): absorb the height
    /// histogram, sample α, then per committed layer sample β and append its root,
    /// then a final β and append `last_value`, then grinding, then query indices.
    /// `derive_batched_fri_challenges` must reproduce byte-identical outputs when
    /// fed the same inputs, starting from a transcript in the same state.
    #[test]
    fn batched_round4_prover_inline_matches_verifier_replay() {
        // Canonical per-epoch table heights (the multiset bound into the transcript).
        let heights: Vec<usize> = vec![10, 10, 8, 8, 8, 5];

        // Fake committed layer roots (K = 4) — only their bytes/order matter here.
        let layer_roots: Vec<[u8; 32]> = (0u8..4).map(|i| [i; 32]).collect();
        let last_value = FE::from(999u64);

        let grinding_factor: u8 = 4;
        let num_queries = 3;
        let domain_size = 1usize << 10;

        let seed_transcript = Transcript::new(b"batched_round4_test");
        let mut transcript_a = seed_transcript.clone();
        let mut transcript_b = seed_transcript.clone();

        // --- Clone A: prover-inline sequence, by hand ---
        absorb_height_histogram(&mut transcript_a, &heights);
        let alpha_a = transcript_a.sample_field_element();

        let mut betas_a = Vec::with_capacity(layer_roots.len() + 1);
        for root in &layer_roots {
            let beta = transcript_a.sample_field_element();
            transcript_a.append_bytes(root);
            betas_a.push(beta);
        }
        let final_beta_a = transcript_a.sample_field_element();
        transcript_a.append_field_element(&last_value);
        betas_a.push(final_beta_a);
        assert_eq!(
            betas_a.len(),
            layer_roots.len() + 1,
            "beta count must be K+1 to match batched_commit_phase"
        );

        let grinding_seed_a = transcript_a.state();
        // Test-only: derive a real PoW nonce so the grinding step is exercised
        // identically by both sides (the nonce search itself is not under test).
        let nonce = crate::grinding::generate_nonce(&grinding_seed_a, grinding_factor)
            .expect("a valid grinding nonce exists for this small grinding_factor");
        transcript_a.append_bytes(&nonce.to_be_bytes());

        let iotas_a: Vec<usize> = (0..num_queries)
            .map(|_| transcript_a.sample_u64((domain_size as u64) >> 1) as usize)
            .collect();

        // --- Clone B: shared replay routine ---
        let result = derive_batched_fri_challenges(
            &mut transcript_b,
            &heights,
            &layer_roots,
            &last_value,
            grinding_factor,
            Some(nonce),
            num_queries,
            domain_size,
        );

        assert_eq!(result.alpha, alpha_a, "alpha mismatch");
        assert_eq!(result.betas, betas_a, "beta vector mismatch");
        assert_eq!(
            result.grinding_seed, grinding_seed_a,
            "grinding seed mismatch"
        );
        assert_eq!(result.iotas, iotas_a, "iotas mismatch");
    }

    /// Tampering with the height histogram (without changing anything else) must
    /// change the derived batching challenge α — this is the structural binding
    /// that protects the fold/injection schedule (spec trap #4).
    #[test]
    fn absorb_height_histogram_binds_heights_into_alpha() {
        let heights_a: Vec<usize> = vec![10, 10, 8, 8, 8, 5];
        let heights_b: Vec<usize> = vec![10, 10, 8, 8, 8, 6]; // one height differs

        let mut transcript_a = Transcript::new(b"histogram_binding_test");
        let mut transcript_b = Transcript::new(b"histogram_binding_test");

        absorb_height_histogram(&mut transcript_a, &heights_a);
        absorb_height_histogram(&mut transcript_b, &heights_b);

        let alpha_a = transcript_a.sample_field_element();
        let alpha_b = transcript_b.sample_field_element();

        assert_ne!(
            alpha_a, alpha_b,
            "different height histograms must yield different alpha"
        );
    }
}
