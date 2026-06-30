use crypto::fiat_shamir::is_transcript::IsStarkTranscript;
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
}
