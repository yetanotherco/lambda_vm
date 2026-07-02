pub mod fri_commitment;
pub mod fri_decommit;
pub(crate) mod fri_functions;
pub(crate) mod terminal;

use crypto::fiat_shamir::is_transcript::IsStarkTranscript;
use math::field::element::FieldElement;
use math::field::traits::{IsFFTField, IsField, IsSubFieldOf};
use math::traits::AsBytes;

use crate::config::{FriLayerMerkleTree, FriLayerMerkleTreeBackend};

use self::fri_commitment::FriLayer;
use self::fri_decommit::FriDecommitment;
use self::fri_functions::{
    compute_coset_twiddles_inv, fold_evaluations_in_place, update_twiddles_in_place,
};

/// FRI commit phase from pre-computed bit-reversed evaluations, skipping the
/// initial FFT. Stops folding when the remaining codeword encodes a polynomial
/// of degree < 2^`final_poly_log_degree` with blowup 2^`blowup_log`, and
/// returns the coefficient vector of that terminal polynomial.
///
/// The `T: Clone` and `F/E: 'static` bounds are required by the cuda GPU
/// fast path (`try_fri_commit_gpu` snapshots the transcript and TypeId-
/// checks the field types). They are present unconditionally (including
/// in builds without the `cuda` feature) to keep one stable signature.
#[allow(clippy::type_complexity)]
pub fn commit_phase_from_evaluations<
    F: IsFFTField + IsSubFieldOf<E> + 'static,
    E: IsField + 'static + Send + Sync,
    T: IsStarkTranscript<E, F> + Clone,
>(
    // `_number_layers`: retained for signature stability with the cuda fast path; termination is now driven by blowup_log + final_poly_log_degree.
    _number_layers: usize,
    mut evals: Vec<FieldElement<E>>,
    transcript: &mut T,
    coset_offset: &FieldElement<F>,
    domain_size: usize,
    blowup_log: u32,
    final_poly_log_degree: u32,
) -> (
    Vec<FieldElement<E>>,
    Vec<FriLayer<E, FriLayerMerkleTreeBackend<E>>>,
)
where
    FieldElement<F>: AsBytes + Sync + Send,
    FieldElement<E>: AsBytes + Sync + Send,
{
    // GPU fast path: drives the entire commit phase device-side (per-layer
    // fold + Keccak leaves + pair-hash tree, only D2H'ing each layer's root
    // + evals + nodes for FriLayer construction). Returns `None` on any
    // failure: precondition misses skip cleanly, and `try_fri_commit_gpu`
    // snapshots the transcript before mutating it so a mid-loop cudarc
    // error restores state and lets the CPU loop below run as if the GPU
    // had never been tried.
    #[cfg(feature = "cuda")]
    {
        // Try the GPU early-termination FRI commit first. `try_fri_commit_gpu`
        // drives the same commit phase on-device (Goldilocks + Ext3, above the
        // LDE size threshold, and only when folding actually happens) and returns
        // `Some` with the final-polynomial coefficients. It returns `None` on any
        // precondition miss or cudarc error — restoring the transcript first — so
        // the CPU path below then runs as if the GPU had never been tried.
        if let Some(result) = crate::gpu_lde::try_fri_commit_gpu::<F, E, T>(
            _number_layers,
            &evals,
            transcript,
            coset_offset,
            domain_size,
            blowup_log,
            final_poly_log_degree,
        ) {
            return result;
        }
    }

    // Determine how many total folds are needed to reach the terminal codeword.
    // terminal_len = 2^(blowup_log + k), clamped to initial_len for tiny inputs.
    let initial_len = evals.len();
    let k = final_poly_log_degree as usize;
    let terminal_len = ((1usize << blowup_log) << k).min(initial_len);
    let total_folds = (initial_len / terminal_len).trailing_zeros() as usize;
    let num_committed = total_folds.saturating_sub(1);

    // Inverse twiddle factors for evaluation-form folding.
    let mut inv_twiddles = compute_coset_twiddles_inv(coset_offset, domain_size);
    let mut fri_layer_list = Vec::with_capacity(num_committed);
    // Track the coset offset as it squares with each fold (needed for iFFT in terminal).
    let mut terminal_offset = coset_offset.clone();

    // Commit `num_committed` folded layers to the transcript.
    for _ in 0..num_committed {
        // <<<< Receive challenge 𝜁ₖ
        let zeta = transcript.sample_field_element();

        // Fold evaluations in-place (no FFT needed).
        fold_evaluations_in_place(&mut evals, &zeta, &inv_twiddles);

        // Build the Merkle tree from consecutive pairs.
        let leaves: Vec<[FieldElement<E>; 2]> = evals
            .chunks_exact(2)
            .map(|chunk| [chunk[0].clone(), chunk[1].clone()])
            .collect();
        let merkle_tree = FriLayerMerkleTree::build(&leaves)
            .expect("FRI commit: Merkle tree construction must succeed");
        let root = merkle_tree.root;
        fri_layer_list.push(FriLayer::new(&evals, merkle_tree));

        // >>>> Send commitment: [pₖ]
        transcript.append_bytes(&root);

        // Update twiddles and offset for the next level.
        update_twiddles_in_place(&mut inv_twiddles);
        terminal_offset = terminal_offset.square();
    }

    // One final fold to reach the terminal codeword (size terminal_len), unless
    // already there (total_folds == 0 means initial_len == terminal_len).
    if total_folds > 0 {
        // <<<< Receive challenge: 𝜁_final
        let zeta = transcript.sample_field_element();
        fold_evaluations_in_place(&mut evals, &zeta, &inv_twiddles);
        terminal_offset = terminal_offset.square();
    }
    debug_assert_eq!(evals.len(), terminal_len, "terminal codeword size mismatch");

    // Recover the low-degree polynomial coefficients from the terminal codeword
    // and send them to the verifier.
    //
    // The number of coefficients is determined by the *actual* terminal codeword,
    // not the requested `final_poly_log_degree`: for tiny inputs `terminal_len`
    // is clamped to `initial_len`, so the terminal polynomial has degree
    // < terminal_len / 2^blowup_log = 2^(log2(terminal_len) - blowup_log). Using
    // this clamped exponent keeps the coefficient count in lockstep with what the
    // verifier reconstructs (`expected_k = min(k, trace_bits)`); passing the raw
    // `final_poly_log_degree` would over-pad with zeros and break the round-trip.
    let effective_log_degree = terminal_len.trailing_zeros() - blowup_log;
    let final_poly_coeffs = crate::fri::terminal::coeffs_from_terminal_codeword::<F, E>(
        &evals,
        &terminal_offset,
        effective_log_degree,
    );
    for c in &final_poly_coeffs {
        transcript.append_field_element(c);
    }

    (final_poly_coeffs, fri_layer_list)
}

pub fn query_phase<F: IsField>(
    fri_layers: &[FriLayer<F, FriLayerMerkleTreeBackend<F>>],
    iotas: &[usize],
) -> Vec<FriDecommitment<F>>
where
    FieldElement<F>: AsBytes + Sync + Send,
{
    // GPU fast path: gather every layer's authentication paths on device (the
    // layer trees stay resident from the GPU commit). Falls back to the host
    // walk below if any layer lacks a device tree.
    #[cfg(feature = "cuda")]
    if let Some(decommits) = crate::gpu_lde::try_fri_query_phase_gpu::<F>(fri_layers, iotas) {
        return decommits;
    }

    if !fri_layers.is_empty() {
        let num_layers = fri_layers.len();
        iotas
            .iter()
            .map(|iota_s| {
                let mut layers_evaluations_sym = Vec::with_capacity(num_layers);
                let mut layers_auth_paths = Vec::with_capacity(num_layers);

                let mut index = *iota_s;
                for layer in fri_layers {
                    // symmetric element
                    let evaluation_sym = layer.evaluation[index ^ 1].clone();
                    let auth_path_sym = layer.merkle_tree.get_proof_by_pos(index >> 1).unwrap();
                    layers_evaluations_sym.push(evaluation_sym);
                    layers_auth_paths.push(auth_path_sym);

                    index >>= 1;
                }

                FriDecommitment {
                    layers_auth_paths,
                    layers_evaluations_sym,
                }
            })
            .collect()
    } else {
        // For 0 FRI layers (small traces), return empty decommitments for each query.
        // The verifier still needs one decommitment entry per query, even if the
        // FRI layer data is empty.
        iotas
            .iter()
            .map(|_| FriDecommitment {
                layers_auth_paths: vec![],
                layers_evaluations_sym: vec![],
            })
            .collect()
    }
}
