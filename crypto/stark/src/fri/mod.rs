#[cfg(any(test, feature = "test-utils"))]
pub mod capture;
#[cfg(all(feature = "cuda", any(test, feature = "test-utils")))]
pub mod device_parity;
pub mod fri_commitment;
pub mod fri_decommit;
pub(crate) mod fri_functions;
pub(crate) mod group;
pub mod schedule;
pub(crate) mod terminal;
#[cfg(any(test, feature = "test-utils"))]
pub mod vectors;

use crypto::fiat_shamir::is_transcript::IsStarkTranscript;
use crypto::merkle_tree::merkle::MerkleTree;
use math::field::element::FieldElement;
use math::field::traits::{IsFFTField, IsField, IsSubFieldOf};
use math::traits::AsBytes;

use crate::config::StarkHash;
use crate::fri::terminal::FriFoldLayout;

use self::fri_commitment::FriLayer;
use self::fri_decommit::FriDecommitment;
use self::fri_functions::{fold_evaluations_in_place, update_twiddles_in_place};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// FRI commit phase from pre-computed bit-reversed evaluations, skipping the
/// initial FFT. Stops folding when the remaining codeword encodes a polynomial
/// of degree < 2^`final_poly_log_degree` with blowup 2^`blowup_log`, and
/// returns the coefficient vector of that terminal polynomial.
///
/// Layer trees are built with `H::Pair` — the commitment configuration's
/// FRI-layer backend, the same `H` the caller's prover and verifier are
/// instantiated at. That is what makes the layer roots this returns
/// authenticable by [`crate::verifier::IsStarkVerifier::verify`], which
/// re-hashes each opened pair through `H::Batched`: the two are one hash by
/// [`StarkHash`]'s two-element invariant, so agreement is a property of naming
/// one configuration rather than of two call sites happening to match.
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
    H: StarkHash,
>(
    evals: Vec<FieldElement<E>>,
    transcript: &mut T,
    coset_offset: &FieldElement<F>,
    domain_size: usize,
    blowup_log: u32,
    final_poly_log_degree: u32,
    inv_twiddles: &[FieldElement<F>],
) -> (Vec<FieldElement<E>>, Vec<FriLayer<E, H::Pair<E>>>)
where
    FieldElement<F>: AsBytes + Sync + Send,
    FieldElement<E>: AsBytes + Sync + Send,
{
    // Today's layout: pair layers, the all-ones schedule.
    let layout = FriFoldLayout::new(
        evals.len().trailing_zeros(),
        blowup_log,
        final_poly_log_degree,
    );
    commit_phase_with_layout::<F, E, T, H>(
        evals,
        transcript,
        coset_offset,
        domain_size,
        blowup_log,
        final_poly_log_degree,
        &layout,
        inv_twiddles,
    )
}

/// [`commit_phase_from_evaluations`] under an explicit fold layout (the proof
/// format's; see [`FriFoldLayout::for_options`]).
///
/// Transcript, per committed layer `j` with fold exponent `d_j`: sample `ζ`,
/// fold `d_{j−1}` times with `ζ, ζ², …` (`d_{−1} = 1`: fold 0 is the binary
/// fold of the DEEP pair), commit the result with leaves of `2^{d_j}` values,
/// append the root. Then, when anything folds, sample the final `ζ` and fold
/// `d_last` times into the terminal codeword. At the all-ones schedule this is
/// exactly today's loop (sample, fold once, commit pairs, append).
///
/// Leaves: the legacy encoding commits `[a, b]` pairs with `H::Pair`; the
/// group encoding hashes each `2^d`-value group with `H::Batched` (the two
/// agree on a two-element leaf, `StarkHash`'s invariant) and builds the tree
/// from those leaf hashes with `H::Pair`'s parent hash — the parent hash both
/// families share, which today's layer trees already rely on (built with
/// `H::Pair`, verified with `H::Batched`).
///
/// The device arm (`try_fri_commit_gpu`) runs both encodings: today's loop for
/// the legacy one and its group twin otherwise. One-row layouts are not
/// implemented on the device and always take the CPU loop
/// ([`commit_phase_cpu_with_layout`]).
#[allow(clippy::type_complexity, clippy::too_many_arguments)]
pub(crate) fn commit_phase_with_layout<
    F: IsFFTField + IsSubFieldOf<E> + 'static,
    E: IsField + 'static + Send + Sync,
    T: IsStarkTranscript<E, F> + Clone,
    H: StarkHash,
>(
    evals: Vec<FieldElement<E>>,
    transcript: &mut T,
    coset_offset: &FieldElement<F>,
    domain_size: usize,
    blowup_log: u32,
    final_poly_log_degree: u32,
    layout: &FriFoldLayout,
    inv_twiddles: &[FieldElement<F>],
) -> (Vec<FieldElement<E>>, Vec<FriLayer<E, H::Pair<E>>>)
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
    if !layout.one_row {
        // Try the GPU early-termination FRI commit first. `try_fri_commit_gpu`
        // drives the same commit phase on-device (Goldilocks + Ext3, above the
        // LDE size threshold, and only when folding actually happens) and returns
        // `Some` with the final-polynomial coefficients. It returns `None` on any
        // precondition miss or cudarc error — restoring the transcript first — so
        // the CPU path below then runs as if the GPU had never been tried.
        if let Some(result) = crate::gpu_lde::try_fri_commit_gpu::<F, E, T, H::Pair<E>>(
            &evals,
            transcript,
            coset_offset,
            domain_size,
            blowup_log,
            final_poly_log_degree,
            layout,
            inv_twiddles,
        ) {
            return result;
        }
    }
    commit_phase_cpu_with_layout::<F, E, T, H>(
        evals,
        transcript,
        coset_offset,
        domain_size,
        blowup_log,
        final_poly_log_degree,
        layout,
        inv_twiddles,
    )
}

/// The CPU loop of [`commit_phase_with_layout`], with no device arm: what every
/// build runs when the device declines, and the host reference the device
/// parity tests compare against.
#[allow(clippy::type_complexity, clippy::too_many_arguments)]
pub(crate) fn commit_phase_cpu_with_layout<
    F: IsFFTField + IsSubFieldOf<E> + 'static,
    E: IsField + 'static + Send + Sync,
    T: IsStarkTranscript<E, F> + Clone,
    H: StarkHash,
>(
    mut evals: Vec<FieldElement<E>>,
    transcript: &mut T,
    coset_offset: &FieldElement<F>,
    domain_size: usize,
    blowup_log: u32,
    final_poly_log_degree: u32,
    layout: &FriFoldLayout,
    inv_twiddles: &[FieldElement<F>],
) -> (Vec<FieldElement<E>>, Vec<FriLayer<E, H::Pair<E>>>)
where
    FieldElement<F>: AsBytes + Sync + Send,
    FieldElement<E>: AsBytes + Sync + Send,
{
    debug_assert_eq!(evals.len(), domain_size);
    // Caller-enforced twiddle sizing (Domain::fri_inv_twiddles): the folding
    // loop below indexes `inv_twiddles[..len/2]` per layer.
    debug_assert_eq!(inv_twiddles.len(), evals.len() / 2);
    // The fold layout, shared with the GPU prover and the verifier — see
    // `FriFoldLayout`. It was built for this codeword's size.
    let _ = (blowup_log, final_poly_log_degree);
    debug_assert_eq!(
        layout.total_folds,
        evals.len().trailing_zeros() - layout.terminal_len.trailing_zeros()
    );
    // One-row layouts (S2) commit the DEEP codeword itself as layer 0; they are
    // refused before a layout is built (`FriFormat::from_options`).
    debug_assert!(!layout.one_row, "one-row FRI layouts are not implemented");
    let num_committed = layout.num_committed;

    // Inverse twiddle factors for evaluation-form folding: per-layer working
    // copy of the per-domain cached set (`Domain::fri_inv_twiddles`).
    let mut inv_twiddles = inv_twiddles.to_vec();
    let mut fri_layer_list = Vec::with_capacity(num_committed);

    // Folds still owed before the next commit: fold 0 is the binary fold of
    // the DEEP pair, so one; after committing layer `j`, `d_j`.
    let mut pending: u32 = 1;

    // Commit `num_committed` folded layers to the transcript.
    for &d in &layout.schedule {
        // <<<< Receive challenge 𝜁ₖ
        let zeta = transcript.sample_field_element();

        // Fold `pending` times with 𝜁, 𝜁², … (evaluation form, no FFT).
        fold_times(&mut evals, &zeta, pending, &mut inv_twiddles);

        let merkle_tree = if layout.is_legacy() {
            // Build the Merkle tree from consecutive pairs.
            let leaves: Vec<[FieldElement<E>; 2]> = evals
                .chunks_exact(2)
                .map(|chunk| [chunk[0].clone(), chunk[1].clone()])
                .collect();
            MerkleTree::<H::Pair<E>>::build(&leaves)
        } else {
            group_tree::<E, H>(&evals, 1usize << d)
        }
        .expect("FRI commit: Merkle tree construction must succeed");
        let root = merkle_tree.root;
        fri_layer_list.push(FriLayer::new(&evals, merkle_tree));

        // >>>> Send commitment: [pₖ]
        transcript.append_bytes(&root);

        pending = u32::from(d);
    }

    // The final folds to reach the terminal codeword (size terminal_len),
    // unless already there (total_folds == 0 means initial_len == terminal_len).
    if layout.total_folds > 0 {
        // <<<< Receive challenge: 𝜁_final
        let zeta = transcript.sample_field_element();
        fold_times(&mut evals, &zeta, pending, &mut inv_twiddles);
    }
    debug_assert_eq!(
        evals.len(),
        layout.terminal_len,
        "terminal codeword size mismatch"
    );

    // Recover the low-degree polynomial coefficients from the terminal codeword
    // and send them to the verifier.
    //
    // The coefficient count follows the *actual* terminal codeword via
    // `layout.effective_k` (`min(k, trace_bits)`), not the requested
    // `final_poly_log_degree`: for tiny inputs the codeword is clamped to the
    // full LDE, so passing the raw `k` would over-pad with zeros and break the
    // round-trip against the verifier's own `expected_k` reconstruction.
    // The terminal coset offset is `coset_offset^(2^total_folds)` — the offset
    // after `total_folds` squarings (matches the GPU prover and the verifier).
    let terminal_offset = coset_offset.pow(1u64 << layout.total_folds);
    let final_poly_coeffs = crate::fri::terminal::coeffs_from_terminal_codeword::<F, E>(
        &evals,
        &terminal_offset,
        layout.effective_k,
    );
    for c in &final_poly_coeffs {
        transcript.append_field_element(c);
    }

    (final_poly_coeffs, fri_layer_list)
}

/// `n` binary folds of `evals` with `ζ, ζ², …, ζ^{2^{n−1}}`, each followed by
/// the twiddle update for the halved domain. `n = 1` is one plain fold (the
/// trailing twiddle update only prepares a fold that may never come).
pub(crate) fn fold_times<F: IsField + IsSubFieldOf<E>, E: IsField>(
    evals: &mut Vec<FieldElement<E>>,
    zeta: &FieldElement<E>,
    n: u32,
    inv_twiddles: &mut Vec<FieldElement<F>>,
) {
    let mut z = zeta.clone();
    for level in 0..n {
        fold_evaluations_in_place(evals, &z, inv_twiddles);
        update_twiddles_in_place(inv_twiddles);
        if level + 1 < n {
            z = z.square();
        }
    }
}

/// A group-leaf layer tree: leaf `g` = `H::Batched` over `evals[g·n .. (g+1)·n]`.
pub(crate) fn group_tree<E, H>(
    evals: &[FieldElement<E>],
    n: usize,
) -> Option<MerkleTree<H::Pair<E>>>
where
    E: IsField + 'static + Send + Sync,
    FieldElement<E>: AsBytes + Sync + Send,
    H: StarkHash,
{
    use crypto::merkle_tree::traits::IsStreamingLeafBackend;
    let hash = |g: &[FieldElement<E>]| {
        <H::Batched<E> as IsStreamingLeafBackend<E>>::hash_data_from_slices(g, &[])
    };
    #[cfg(feature = "parallel")]
    let leaves: Vec<_> = evals.par_chunks_exact(n).map(hash).collect();
    #[cfg(not(feature = "parallel"))]
    let leaves: Vec<_> = evals.chunks_exact(n).map(hash).collect();
    MerkleTree::<H::Pair<E>>::build_from_hashed_leaves(leaves)
}

/// Open every committed layer at each query index, producing one
/// [`FriDecommitment`] per query.
///
/// Takes the layers [`commit_phase_from_evaluations`] built, so it is generic
/// over the same configuration `H`: the authentication paths it walks are only
/// meaningful against roots the verifier re-derives through `H`.
pub fn query_phase<F: IsField + 'static, H: StarkHash>(
    fri_layers: &[FriLayer<F, H::Pair<F>>],
    iotas: &[usize],
) -> Vec<FriDecommitment<F>>
where
    FieldElement<F>: AsBytes + Sync + Send,
{
    // GPU fast path: gather every layer's authentication paths on device (the
    // layer trees stay resident from the GPU commit). Falls back to the host
    // walk below if any layer lacks a device tree.
    #[cfg(feature = "cuda")]
    if let Some(decommits) =
        crate::gpu_lde::try_fri_query_phase_gpu::<F, H::Pair<F>>(fri_layers, iotas)
    {
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

/// [`query_phase`] under an explicit fold layout. The legacy encoding is
/// [`query_phase`] itself (device arm included); the group encoding opens, per
/// committed layer `j`, the whole group `evaluation[leaf·2^{d_j} ..][..2^{d_j}]`
/// (the query's own value included, FRI.md §3.4) and the path of
/// `leaf = p >> d_j`, then moves to `p >> d_j` — on the device when the layers
/// are device-resident (`try_fri_query_phase_gpu_groups`), else by the host
/// walk ([`query_phase_groups_host`]).
pub(crate) fn query_phase_with_layout<F: IsField + 'static, H: StarkHash>(
    fri_layers: &[FriLayer<F, H::Pair<F>>],
    iotas: &[usize],
    layout: &FriFoldLayout,
) -> Vec<FriDecommitment<F>>
where
    FieldElement<F>: AsBytes + Sync + Send,
{
    if layout.is_legacy() {
        return query_phase::<F, H>(fri_layers, iotas);
    }
    #[cfg(feature = "cuda")]
    if let Some(decommits) =
        crate::gpu_lde::try_fri_query_phase_gpu_groups::<F, H::Pair<F>>(fri_layers, iotas, layout)
    {
        return decommits;
    }
    query_phase_groups_host::<F, H>(fri_layers, iotas, layout)
}

/// The host walk of [`query_phase_with_layout`]'s group encoding over host
/// layer trees (the device parity tests' reference).
pub(crate) fn query_phase_groups_host<F: IsField + 'static, H: StarkHash>(
    fri_layers: &[FriLayer<F, H::Pair<F>>],
    iotas: &[usize],
    layout: &FriFoldLayout,
) -> Vec<FriDecommitment<F>>
where
    FieldElement<F>: AsBytes + Sync + Send,
{
    debug_assert_eq!(fri_layers.len(), layout.num_committed);
    iotas
        .iter()
        .map(|&iota| {
            let mut values = Vec::with_capacity(layout.opened_values_per_query());
            let mut paths = Vec::with_capacity(fri_layers.len());
            let mut index = iota;
            for (layer, &d) in fri_layers.iter().zip(&layout.schedule) {
                let n = 1usize << d;
                let leaf = index >> d;
                values.extend_from_slice(&layer.evaluation[leaf * n..(leaf + 1) * n]);
                paths.push(
                    layer
                        .merkle_tree
                        .get_proof_by_pos(leaf)
                        .expect("FRI query: leaf index within the layer tree"),
                );
                index = leaf;
            }
            FriDecommitment {
                layers_auth_paths: paths,
                layers_evaluations_sym: values,
            }
        })
        .collect()
}
