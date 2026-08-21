//! Batched FRI: one FRI instance over an epoch's DEEP codewords instead of one
//! per table.
//!
//! Codewords are bucketed by height, mixed within a bucket with powers of a
//! single `alpha`, and then folded from the tallest bucket downward, each
//! shorter bucket being *injected* into the running codeword at the layer whose
//! length matches it. One set of query indices, drawn from the tallest domain,
//! tests the whole chain.
//!
//! # Termination
//!
//! Folding stops at the same terminal the unbatched
//! [`crate::fri::commit_phase_from_evaluations`] stops at — the codeword that
//! encodes a polynomial of degree `< 2^fri_final_poly_log_degree` — and sends
//! that polynomial's coefficients, rather than folding all the way down to a
//! scalar. [`BatchedFriLayout`] derives the fold count through the shared
//! [`FriFoldLayout`], with one batched-only floor: the terminal may not sit
//! above the SHORTEST injected codeword, or that codeword would never reach the
//! running word. So the early stop is `min(blowup_log + k, h_min)`.

use crypto::fiat_shamir::is_transcript::{IsStarkTranscript, IsTranscript};
use crypto::merkle_tree::merkle::MerkleTree;
use math::field::element::FieldElement;
use math::field::traits::{IsFFTField, IsField, IsSubFieldOf};
use math::traits::AsBytes;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crate::config::StarkHash;
use crate::fri::fri_commitment::FriLayer;
use crate::fri::fri_functions::{
    compute_coset_twiddles_inv, fold_evaluations_in_place, update_twiddles_in_place,
};
use crate::fri::terminal::{FriFoldLayout, coeffs_from_terminal_codeword};

/// Accumulates DEEP codewords into per-height buckets as they are produced,
/// mixing the `i`-th absorbed codeword with `alpha^i`.
///
/// The point of absorbing one codeword at a time is memory: a caller that
/// produces a table's quotient, absorbs it and drops it retains only one bucket
/// per distinct height (`O(2^h_max)` in total), where handing
/// [`combine_by_height`] a fully-materialized `Vec` of every table's codeword
/// retains `O(N_tables · 2^h)`. The result is identical either way — absorption
/// order defines the `alpha` powers, so the caller must absorb in the same
/// canonical per-epoch order the verifier assumes.
pub struct HeightCombiner<E: IsField> {
    buckets: Vec<Option<Vec<FieldElement<E>>>>,
    alpha: FieldElement<E>,
    /// `alpha^i` for the next codeword to be absorbed.
    next_power: FieldElement<E>,
}

impl<E: IsField> HeightCombiner<E> {
    pub fn new(alpha: FieldElement<E>) -> Self {
        Self {
            buckets: Vec::new(),
            alpha,
            next_power: FieldElement::one(),
        }
    }

    /// Absorb one codeword of length `2^height`, scaled by the next power of
    /// `alpha`.
    pub fn absorb(&mut self, codeword: &[FieldElement<E>], height: usize) {
        let expected_len = 1usize << height;
        assert_eq!(
            codeword.len(),
            expected_len,
            "codeword has length {} but height {height} expects {expected_len}",
            codeword.len()
        );

        if self.buckets.len() <= height {
            self.buckets.resize_with(height + 1, || None);
        }
        let scaled = &self.next_power;
        // Data-parallel under `parallel`: the scale and the scale-accumulate
        // are elementwise over up to 2^h_max elements, and this loop has no
        // per-table overlap to hide behind — it was serial wall time once per
        // absorbed table. Same arithmetic in both arms, identical result.
        #[cfg(feature = "parallel")]
        match &mut self.buckets[height] {
            None => {
                self.buckets[height] = Some(
                    codeword
                        .par_iter()
                        .map(|x| scaled * x)
                        .collect::<Vec<FieldElement<E>>>(),
                );
            }
            Some(acc) => {
                acc.par_iter_mut()
                    .zip(codeword.par_iter())
                    .for_each(|(a, x)| {
                        *a = &*a + &(scaled * x);
                    });
            }
        }
        #[cfg(not(feature = "parallel"))]
        match &mut self.buckets[height] {
            None => {
                self.buckets[height] = Some(codeword.iter().map(|x| scaled * x).collect());
            }
            Some(acc) => {
                for (a, x) in acc.iter_mut().zip(codeword.iter()) {
                    *a = &*a + &(scaled * x);
                }
            }
        }
        self.next_power = &self.next_power * &self.alpha;
    }

    /// The per-height buckets. Index `h` is `Some(combined)` when at least one
    /// codeword of height `h` was absorbed, `None` otherwise; the `Vec` is
    /// `max_absorbed_height + 1` long, or empty if nothing was absorbed.
    pub fn finish(self) -> Vec<Option<Vec<FieldElement<E>>>> {
        self.buckets
    }
}

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
///
/// This is [`HeightCombiner`] with every codeword already materialized. Prefer
/// the combiner in the prover, where holding all of them at once is the whole
/// memory cost the batching is meant to remove.
pub fn combine_by_height<E>(
    inputs: &[(Vec<FieldElement<E>>, usize)],
    alpha: &FieldElement<E>,
) -> Vec<Option<Vec<FieldElement<E>>>>
where
    E: IsField,
{
    let mut combiner = HeightCombiner::new(alpha.clone());
    for (codeword, height) in inputs {
        combiner.absorb(codeword, *height);
    }
    combiner.finish()
}

/// How far a batched FRI instance folds, and what it sends at the end.
///
/// Mirrors [`FriFoldLayout`] — same early stop, same terminal codeword, same
/// coefficient count — with the one difference batching forces: the terminal is
/// additionally floored at the SHORTEST injected codeword's height, since a
/// bucket below the terminal would never be folded into the running word. In a
/// real epoch the shortest table is normally well above `blowup_log + k`, so the
/// floor is inert and the layout is exactly the unbatched one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BatchedFriLayout {
    /// Folds from the tallest bucket down to the terminal codeword.
    pub total_folds: u32,
    /// Committed (Merkle-rooted) FRI layers.
    pub num_committed: usize,
    /// Terminal codeword length.
    pub terminal_len: usize,
    /// `log2` of the terminal polynomial's degree bound — the number of
    /// coefficients sent is `2^effective_k`.
    pub effective_k: u32,
}

impl BatchedFriLayout {
    /// Derive the layout from the epoch's codeword heights.
    ///
    /// * `h_max` / `h_min` — the tallest and shortest codeword heights present.
    /// * `blowup_log` — log2 of the LDE blowup factor.
    /// * `final_poly_log_degree` — the requested `fri_final_poly_log_degree`.
    ///
    /// Panics if `h_min < blowup_log` (a codeword shorter than the blowup is not
    /// a Reed-Solomon word of any positive rate) or if `h_min > h_max`.
    pub fn new(h_max: usize, h_min: usize, blowup_log: u32, final_poly_log_degree: u32) -> Self {
        assert!(h_min <= h_max, "h_min {h_min} exceeds h_max {h_max}");
        assert!(
            h_min as u32 >= blowup_log,
            "codeword height {h_min} is below the blowup {blowup_log}"
        );
        // Deriving at `h_min` is what applies the floor: `FriFoldLayout` clamps
        // the terminal to its `lde_log` argument, so the terminal comes out at
        // `min(blowup_log + k, h_min)`. Its terminal_len / effective_k are then
        // exactly what the unbatched prover would send for that codeword.
        let shortest = FriFoldLayout::new(h_min as u32, blowup_log, final_poly_log_degree);
        let terminal_log = shortest.terminal_len.trailing_zeros();
        // The running codeword starts at h_max, not h_min, so the fold count is
        // re-derived from where folding actually begins.
        let total_folds = h_max as u32 - terminal_log;
        Self {
            total_folds,
            num_committed: total_folds.saturating_sub(1) as usize,
            terminal_len: shortest.terminal_len,
            effective_k: shortest.effective_k,
        }
    }
}

/// Which of an epoch's tables enter the ONE batched FRI instance, and which keep
/// a terminal-only instance of their own.
///
/// # Why there are two classes
///
/// A table whose own FRI would commit ZERO layers gains nothing from being
/// batched — there is no layer for the batch to share — while it pays the full
/// cost of being lifted to the tallest domain, which is where the proximity-gaps
/// term's `|D0|²` lives. At the measured epoch that is 13 of 28 legs carrying 92%
/// of the batch's width, so excluding them is a correction rather than a
/// compromise: it recovers ~3.6 bits of soundness AND removes work.
///
/// A zero-layer table's FRI is degenerate in the useful sense — its terminal
/// codeword IS its deep-composition codeword — so its "own instance" is one
/// terminal polynomial and no layers at all.
///
/// # ★ The index rule BETWEEN the classes — a hard precondition
///
/// Both classes are opened at the SAME query indices, because the mixed-height
/// MMCS is unaffected by this split: it still commits every table, and the point
/// of one shared authentication path survives whole. What differs is the index
/// SPACE each class reads them in:
///
/// ```text
/// batched class:    iota, used directly (it is an index in the tallest domain)
/// standalone table: iota >> (h_max - h_t)
/// ```
///
/// This is the same reduction [`crate::fri::mmcs`]'s index-convention section
/// documents for a short round, and it fails the same silent way: prover and
/// verifier derive it from the shape, so a wrong shift is self-consistent —
/// honest proofs still verify while the short tables end up checked at positions
/// the FRI join never reaches. `each_instance_class_is_tamper_checked` is the
/// control, and it tampers a table of EACH class, because a control that only
/// touched the batched class would pass under any convention for the other.
///
/// # Determinism
///
/// The plan is a pure function of `(heights, blowup_log, final_poly_log_degree)`,
/// all of which the transcript has bound before any challenge that depends on it.
/// Prover and verifier therefore derive the SAME partition without it being sent,
/// which is why the split adds nothing to the wire and nothing to the shape
/// binding.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FriInstancePlan {
    /// Table indices whose codewords are mixed into the batched instance, in
    /// input order — the order that defines the `alpha` powers.
    pub batched: Vec<usize>,
    /// Table indices that keep a terminal-only instance, in input order.
    pub standalone: Vec<usize>,
    /// Tallest and shortest height WITHIN the batched class — the layout is
    /// derived from these, not from the whole epoch.
    pub h_max: usize,
    pub h_min: usize,
}

impl FriInstancePlan {
    /// Partition an epoch's tables. `None` when `heights` is empty or carries a
    /// height that cannot be a codeword length — both are proof-supplied, so both
    /// are rejections rather than panics.
    ///
    /// The TALLEST table is always batched, even if it would classify as
    /// standalone on its own. That keeps the batched class non-empty, so the
    /// layout is always well defined; an epoch whose tallest table folds nothing
    /// degenerates to a single terminal-only instance, which is what it should be.
    pub fn new(heights: &[usize], blowup_log: u32, final_poly_log_degree: u32) -> Option<Self> {
        if heights.is_empty() {
            return None;
        }
        let &h_max_epoch = heights.iter().max()?;
        if h_max_epoch == 0 || h_max_epoch >= u32::BITS as usize {
            return None;
        }
        let tallest = heights.iter().position(|h| *h == h_max_epoch)?;

        let mut batched = Vec::with_capacity(heights.len());
        let mut standalone = Vec::new();
        for (t, &h) in heights.iter().enumerate() {
            if h < blowup_log as usize {
                return None;
            }
            let folds_a_layer =
                FriFoldLayout::new(h as u32, blowup_log, final_poly_log_degree).num_committed > 0;
            if folds_a_layer || t == tallest {
                batched.push(t);
            } else {
                standalone.push(t);
            }
        }

        let h_max = batched.iter().map(|&t| heights[t]).max()?;
        let h_min = batched.iter().map(|&t| heights[t]).min()?;
        Some(Self {
            batched,
            standalone,
            h_max,
            h_min,
        })
    }
}

/// FRI commit phase over the bucketed output of [`combine_by_height`] /
/// [`HeightCombiner::finish`].
///
/// `combined[h]` is `Some(codeword)` when there are DEEP contributions at height
/// `h` (codeword length `2^h`), or `None` otherwise.
///
/// Folding starts from the tallest bucket. After each fold to height `h`, the
/// bucket at `combined[h]` is injected into the running codeword with
/// coefficient `β²` (β being the fold challenge just used), before the layer is
/// committed. Termination follows [`BatchedFriLayout`]: the running codeword is
/// folded to the terminal length and the terminal polynomial's coefficients are
/// appended to the transcript, exactly as
/// [`crate::fri::commit_phase_from_evaluations`] does — not folded down to a
/// single scalar.
///
/// Layer trees are built with `H::Pair`, the same commitment configuration the
/// unbatched [`crate::fri::commit_phase_from_evaluations`] uses — so a batched
/// prover and the verifier that authenticates its openings through `H::Batched`
/// agree on the hash by naming one configuration, not by two call sites
/// coinciding.
#[allow(clippy::type_complexity)]
pub fn batched_commit_phase<F, E, T, H>(
    mut combined: Vec<Option<Vec<FieldElement<E>>>>,
    transcript: &mut T,
    coset_offset: &FieldElement<F>,
    blowup_log: u32,
    final_poly_log_degree: u32,
) -> (Vec<FieldElement<E>>, Vec<FriLayer<E, H::Pair<E>>>)
where
    F: IsFFTField + IsSubFieldOf<E> + 'static,
    E: IsField + 'static + Send + Sync,
    T: IsStarkTranscript<E, F> + Clone,
    H: StarkHash,
    FieldElement<F>: AsBytes + Sync + Send,
    FieldElement<E>: AsBytes + Sync + Send,
{
    let (h_min, h_max) = bucket_height_range(&combined)
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

    let layout = BatchedFriLayout::new(h_max, h_min, blowup_log, final_poly_log_degree);

    // Inverse twiddle factors for the initial domain size.
    let mut inv_twiddles = compute_coset_twiddles_inv(coset_offset, domain_size);

    let mut fri_layer_list = Vec::with_capacity(layout.num_committed);

    for _ in 0..layout.num_committed {
        // <<<< Receive challenge β
        let beta = transcript.sample_field_element();

        // Fold evaluations in-place; running halves in length.
        fold_evaluations_in_place(&mut running, &beta, &inv_twiddles);
        inject_bucket(&mut running, &mut combined, &beta);

        // Build the row-pair Merkle tree over the current running codeword.
        let leaves: Vec<[FieldElement<E>; 2]> = running
            .chunks_exact(2)
            .map(|chunk| [chunk[0].clone(), chunk[1].clone()])
            .collect();
        let merkle_tree = MerkleTree::<H::Pair<E>>::build(&leaves)
            .expect("FRI batched commit: Merkle tree construction must succeed");
        let root = merkle_tree.root;
        fri_layer_list.push(FriLayer::new(&running, merkle_tree));

        // >>>> Send commitment: append root to transcript.
        transcript.append_bytes(&root);

        // Update twiddles for the next (halved) level.
        update_twiddles_in_place(&mut inv_twiddles);
    }

    // One final fold to reach the terminal codeword, unless already there. The
    // bucket AT the terminal height is injected here: it is the last one that can
    // still enter the running word, which is why the layout floors the terminal
    // at the shortest height rather than at `blowup_log + k` alone.
    if layout.total_folds > 0 {
        let beta = transcript.sample_field_element();
        fold_evaluations_in_place(&mut running, &beta, &inv_twiddles);
        inject_bucket(&mut running, &mut combined, &beta);
    }
    debug_assert_eq!(
        running.len(),
        layout.terminal_len,
        "terminal codeword size mismatch"
    );
    debug_assert!(
        combined.iter().all(Option::is_none),
        "every bucket must have been injected before the terminal"
    );

    // Recover the terminal polynomial's coefficients and send them, mirroring
    // `commit_phase_from_evaluations`: the coefficient count follows
    // `layout.effective_k` (the actual terminal), and the terminal coset offset
    // is `coset_offset^(2^total_folds)`.
    let terminal_offset = coset_offset.pow(1u64 << layout.total_folds);
    let final_poly_coeffs =
        coeffs_from_terminal_codeword::<F, E>(&running, &terminal_offset, layout.effective_k);
    for c in &final_poly_coeffs {
        transcript.append_field_element(c);
    }

    (final_poly_coeffs, fri_layer_list)
}

/// The `(h_min, h_max)` of the occupied buckets, or `None` when none are.
fn bucket_height_range<E: IsField>(
    combined: &[Option<Vec<FieldElement<E>>>],
) -> Option<(usize, usize)> {
    let mut occupied = combined
        .iter()
        .enumerate()
        .filter_map(|(h, slot)| slot.as_ref().map(|_| h));
    let first = occupied.next()?;
    Some((first, occupied.next_back().unwrap_or(first)))
}

/// `running += β² · combined[h]` for the running codeword's current height `h`,
/// consuming that bucket. A no-op when the bucket is empty.
fn inject_bucket<E: IsField>(
    running: &mut [FieldElement<E>],
    combined: &mut [Option<Vec<FieldElement<E>>>],
    beta: &FieldElement<E>,
) {
    let h = running.len().trailing_zeros() as usize;
    let Some(bucket) = combined.get_mut(h).and_then(Option::take) else {
        return;
    };
    debug_assert_eq!(
        bucket.len(),
        running.len(),
        "a bucket at height {h} must match the running codeword's length"
    );
    let beta_sq = beta.square();
    for (val, contribution) in running.iter_mut().zip(bucket.iter()) {
        *val = &*val + &(&beta_sq * contribution);
    }
}

/// Canonical, order-deterministic absorption of an epoch's table-SHAPE histogram
/// into the transcript. Single source of truth for the structural binding.
///
/// The multiset of `lde_log_height`s across an epoch's tables fully determines
/// the fold order and injection points of the batched FRI (arity is uniformly
/// 2), so binding the heights binds the whole injection schedule. The widths are
/// bound alongside them because they are what makes the mixed-height MMCS leaf
/// parse unambiguous (see [`crate::fri::mmcs`]'s width-binding section) — the
/// verifier derives widths from the AIR set rather than the proof, so this is
/// defence in depth rather than the primary binding, and it costs one field per
/// table.
///
/// Encoding (fixed-width, length-prefixed, order-preserving):
/// `u64::to_le_bytes(len)` followed by `u64::to_le_bytes(h)`, `u64::to_le_bytes(w)`
/// for each `(h, w)` pair, in the exact order given. Caller (prover and verifier
/// alike) must pass the shape in the same canonical per-epoch table order — this
/// function does not sort or deduplicate.
///
/// Panics if `heights` and `widths` differ in length; both sides construct them
/// from the same table list.
pub fn absorb_shape_histogram<E, T>(transcript: &mut T, heights: &[usize], widths: &[usize])
where
    E: IsField,
    T: IsTranscript<E>,
{
    assert_eq!(
        heights.len(),
        widths.len(),
        "the shape histogram needs one width per height"
    );
    transcript.append_bytes(&(heights.len() as u64).to_le_bytes());
    for (h, w) in heights.iter().zip(widths.iter()) {
        transcript.append_bytes(&(*h as u64).to_le_bytes());
        transcript.append_bytes(&(*w as u64).to_le_bytes());
    }
}

/// Challenges derived from replaying the shared batched round-4 transcript
/// sequence. See [`derive_batched_fri_challenges`].
#[derive(Debug, Clone)]
pub struct BatchedFriChallenges<E: IsField> {
    /// Sampled once after the shape histogram (and, at the call site, after all
    /// per-table OOD evaluations have been absorbed).
    pub alpha: FieldElement<E>,
    /// One per committed layer, plus one for the final fold when there is one:
    /// `betas.len() == layout.num_committed + (layout.total_folds > 0) as usize`.
    pub betas: Vec<FieldElement<E>>,
    /// The layout the betas and the terminal were derived under.
    pub layout: BatchedFriLayout,
    /// Transcript state right before the grinding nonce bytes are appended.
    /// All-zero when `grinding_factor == 0` or `nonce` is `None`.
    pub grinding_seed: [u8; 32],
    /// One `sample_u64(2^(h_max - 1))` draw per query — a row-PAIR index in the
    /// tallest domain. A round whose own `h_max` is lower must reduce these; see
    /// [`crate::fri::mmcs`]'s index-convention section.
    pub iotas: Vec<usize>,
    /// Which tables the batched instance carries and which keep a terminal-only
    /// instance of their own. Derived from the shape, never sent.
    pub plan: FriInstancePlan,
}

/// Replays the shared batched round-4 transcript sequence (shape histogram,
/// alpha, per-layer beta/root, final beta, terminal coefficients, grinding, query
/// iotas) and returns the derived challenges. The one routine the prover and the
/// verifier both call, so they provably derive identical challenges.
///
/// `standalone_coeffs[t]` is table `t`'s terminal-only polynomial, `Some`
/// exactly for the standalone class — presence is checked against the derived
/// plan and every coefficient is ABSORBED, right after `α` and before the
/// first `ζ`. That absorb is load-bearing: the standalone check evaluates the
/// sent polynomial at the query indices drawn BELOW, so a polynomial that
/// were not bound here could be chosen after the indices are known, and each
/// query's proximity test would bind nothing until the queries saturate the
/// table's domain. The unbatched path absorbs its terminal before sampling
/// queries for the same reason; this keeps the batched path's binding equal.
///
/// Returns `None` when the proof's layer-root count disagrees with the layout the
/// epoch's shape implies, when the terminal coefficient count is wrong, or when
/// a standalone polynomial is present for the wrong class — all prover-supplied,
/// all rejections, not panics.
#[allow(clippy::too_many_arguments)]
pub fn derive_batched_fri_challenges<E, T>(
    transcript: &mut T,
    heights: &[usize],
    widths: &[usize],
    layer_roots: &[[u8; 32]],
    final_poly_coeffs: &[FieldElement<E>],
    standalone_coeffs: &[Option<&[FieldElement<E>]>],
    blowup_log: u32,
    final_poly_log_degree: u32,
    grinding_factor: u8,
    nonce: Option<u64>,
    num_queries: usize,
) -> Option<BatchedFriChallenges<E>>
where
    E: IsField,
    T: IsTranscript<E>,
{
    // The partition is derived, not sent: it is a pure function of the shape the
    // histogram below binds, so both sides reach the same one. `None` on any
    // height that cannot be a codeword length — heights come from proof-supplied
    // trace lengths, so a bogus one is a rejection, never a panic on the
    // verifier's path.
    let plan = FriInstancePlan::new(heights, blowup_log, final_poly_log_degree)?;
    let (h_max, h_min) = (plan.h_max, plan.h_min);
    let layout = BatchedFriLayout::new(h_max, h_min, blowup_log, final_poly_log_degree);
    if layer_roots.len() != layout.num_committed
        || final_poly_coeffs.len() != 1usize << layout.effective_k
        || standalone_coeffs.len() != heights.len()
    {
        return None;
    }

    absorb_shape_histogram(transcript, heights, widths);

    let alpha = transcript.sample_field_element();

    // The standalone class's terminal polynomials, bound before any query can
    // depend on them — per table ascending, each coefficient in order. The
    // length pin (`2^(h_t − blowup_log)`, exactly) stays with
    // `verify_epoch_commitments`.
    for (table, coeffs) in standalone_coeffs.iter().enumerate() {
        if coeffs.is_some() != plan.standalone.contains(&table) {
            return None;
        }
        if let Some(coeffs) = coeffs {
            for c in coeffs.iter() {
                transcript.append_field_element(c);
            }
        }
    }

    let mut betas = Vec::with_capacity(layout.num_committed + 1);
    for root in layer_roots {
        let beta = transcript.sample_field_element();
        transcript.append_bytes(root);
        betas.push(beta);
    }

    if layout.total_folds > 0 {
        betas.push(transcript.sample_field_element());
    }
    for c in final_poly_coeffs {
        transcript.append_field_element(c);
    }

    let mut grinding_seed = [0u8; 32];
    if grinding_factor > 0
        && let Some(nonce_value) = nonce
    {
        grinding_seed = transcript.state();
        transcript.append_bytes(&nonce_value.to_be_bytes());
    }

    let iotas = (0..num_queries)
        .map(|_| transcript.sample_u64(1u64 << (h_max - 1)) as usize)
        .collect();

    Some(BatchedFriChallenges {
        alpha,
        betas,
        layout,
        grinding_seed,
        iotas,
        plan,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::DefaultStarkHash;
    use crate::fri::commit_phase_from_evaluations;
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
        let alpha1 = alpha;
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

    /// Absorbing codewords one at a time — the shape a prover uses so it never
    /// holds every table's quotient at once — must land on the same buckets as
    /// handing them all over materialized.
    #[test]
    fn streaming_absorption_matches_materialized_combine() {
        let inputs: Vec<(Vec<FE>, usize)> = vec![
            ((1u64..=16).map(FE::from).collect(), 4),
            ((50u64..=57).map(FE::from).collect(), 3),
            ((90u64..=105).map(FE::from).collect(), 4),
            ((200u64..=203).map(FE::from).collect(), 2),
            ((300u64..=307).map(FE::from).collect(), 3),
        ];
        let alpha = FE::from(11u64);

        let eager = combine_by_height(&inputs, &alpha);

        let mut combiner = HeightCombiner::new(alpha);
        for (codeword, height) in &inputs {
            combiner.absorb(codeword, *height);
        }
        assert_eq!(
            combiner.finish(),
            eager,
            "streaming absorption must equal the materialized combine"
        );
    }

    /// After the first fold in `batched_commit_phase`, the committed layer[0]
    /// evaluation must equal `fold(combined[4], β₀) + β₀² · combined[3]`.
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
        let (blowup_log, k) = (1u32, 1u32);

        // Create transcript; clone before mutating so we can replay independently.
        let mut transcript = Transcript::new(b"batched_fri_test");
        let mut transcript_check = transcript.clone();

        let (_coeffs, layers) = batched_commit_phase::<_, _, _, DefaultStarkHash>(
            combined,
            &mut transcript,
            &coset_offset,
            blowup_log,
            k,
        );

        // Terminal at min(blowup_log + k, h_min) = min(2, 3) = 2, so folds run
        // 4 -> 2: two folds, one committed layer.
        let layout = BatchedFriLayout::new(4, 3, blowup_log, k);
        assert_eq!(layout.total_folds, 2);
        assert_eq!(
            layers.len(),
            layout.num_committed,
            "committed layers must follow the layout"
        );

        // --- Independent recomputation of layer[0] ---
        let beta_0 = transcript_check.sample_field_element();

        let inv_twiddles_h4 = compute_coset_twiddles_inv::<GoldilocksField>(&coset_offset, 16);
        let mut expected = data_h4.clone();
        fold_evaluations_in_place(&mut expected, &beta_0, &inv_twiddles_h4);
        // expected now has length 8 (height 3)

        // Inject combined[3]: expected[j] += beta_0² · data_h3[j]
        let beta_0_sq = beta_0.square();
        for (j, val) in data_h3.iter().enumerate() {
            expected[j] = &expected[j] + &(&beta_0_sq * val);
        }

        assert_eq!(
            layers[0].evaluation, expected,
            "layer[0] evaluation does not match manual fold+inject"
        );
    }

    /// ★ M-12: the batched commit phase must terminate where the unbatched one
    /// does. With a single bucket the two are the same protocol, so they must
    /// agree on the committed-layer count, the terminal coefficients, and the
    /// resulting transcript state — pinning that batching did not silently switch
    /// to folding all the way to a scalar (which for this input would commit
    /// `h_max - 1 = 9` layers instead of 4).
    #[test]
    fn single_bucket_terminal_matches_the_unbatched_commit_phase() {
        let h = 10usize;
        let (blowup_log, k) = (1u32, 5u32);
        let coset_offset = FE::from(3u64);
        let evals: Vec<FE> = (0..(1u64 << h)).map(|i| FE::from(i * 7 + 1)).collect();
        let inv_twiddles = compute_coset_twiddles_inv::<GoldilocksField>(&coset_offset, 1 << h);

        let mut t_unbatched = Transcript::new(b"terminal_parity");
        let (unbatched_coeffs, unbatched_layers) = commit_phase_from_evaluations::<
            GoldilocksField,
            GoldilocksField,
            Transcript,
            DefaultStarkHash,
        >(
            evals.clone(),
            &mut t_unbatched,
            &coset_offset,
            1 << h,
            blowup_log,
            k,
            &inv_twiddles,
        );

        let mut combined: Vec<Option<Vec<FE>>> = vec![None; h + 1];
        combined[h] = Some(evals);
        let mut t_batched = Transcript::new(b"terminal_parity");
        let (batched_coeffs, batched_layers) = batched_commit_phase::<_, _, _, DefaultStarkHash>(
            combined,
            &mut t_batched,
            &coset_offset,
            blowup_log,
            k,
        );

        // total_folds = 10 - (1 + 5) = 4, so 3 committed layers — not h_max-1 = 9.
        assert_eq!(unbatched_layers.len(), 3);
        assert_eq!(
            batched_layers.len(),
            unbatched_layers.len(),
            "batched and unbatched must commit the same number of layers"
        );
        assert_eq!(
            batched_coeffs.len(),
            1usize << k,
            "the terminal polynomial must carry 2^k coefficients"
        );
        assert_eq!(
            batched_coeffs, unbatched_coeffs,
            "batched and unbatched must send the same terminal polynomial"
        );
        for (b, u) in batched_layers.iter().zip(unbatched_layers.iter()) {
            assert_eq!(b.merkle_tree.root, u.merkle_tree.root);
        }
        assert_eq!(
            t_batched.state(),
            t_unbatched.state(),
            "the two commit phases must leave the transcript in the same state"
        );
    }

    /// The batched-only floor: the terminal may not sit above the shortest
    /// injected codeword, or that bucket would never enter the running word.
    #[test]
    fn terminal_is_floored_at_the_shortest_codeword() {
        let (blowup_log, k) = (1u32, 5u32);

        // Shortest codeword above blowup_log + k = 6: the floor is inert and the
        // layout is the unbatched one for h_max.
        let inert = BatchedFriLayout::new(10, 8, blowup_log, k);
        assert_eq!(inert.total_folds, 4, "10 -> 6");
        assert_eq!(inert.effective_k, k);

        // Shortest codeword BELOW blowup_log + k: folding must continue down to
        // it, and the terminal polynomial shrinks accordingly.
        let floored = BatchedFriLayout::new(10, 4, blowup_log, k);
        assert_eq!(floored.total_folds, 6, "10 -> 4");
        assert_eq!(floored.effective_k, 3, "terminal_log 4 - blowup_log 1");

        // And the commit phase really does consume that low bucket.
        let coset_offset = FE::from(3u64);
        let mut combined: Vec<Option<Vec<FE>>> = vec![None; 8];
        combined[7] = Some((0..128u64).map(|i| FE::from(i + 1)).collect());
        combined[4] = Some((0..16u64).map(|i| FE::from(i * 3 + 5)).collect());
        let mut transcript = Transcript::new(b"floor_test");
        let (coeffs, layers) = batched_commit_phase::<_, _, _, DefaultStarkHash>(
            combined,
            &mut transcript,
            &coset_offset,
            blowup_log,
            k,
        );
        let layout = BatchedFriLayout::new(7, 4, blowup_log, k);
        assert_eq!(layers.len(), layout.num_committed);
        assert_eq!(coeffs.len(), 1usize << layout.effective_k);
    }

    /// The prover, by hand, runs exactly the round-4 sequence; the shared replay
    /// routine must reproduce byte-identical outputs from the same start state.
    #[test]
    fn batched_round4_prover_inline_matches_verifier_replay() {
        let heights: Vec<usize> = vec![10, 10, 8, 8, 8, 7];
        let widths: Vec<usize> = vec![3, 5, 2, 2, 9, 1];
        let (blowup_log, k) = (1u32, 5u32);
        // total_folds = 10 - 6 = 4 -> 3 committed layers, 4 betas.
        let layout = BatchedFriLayout::new(10, 7, blowup_log, k);
        assert_eq!((layout.num_committed, layout.total_folds), (3, 4));

        let layer_roots: Vec<[u8; 32]> = (0u8..3).map(|i| [i; 32]).collect();
        let final_poly_coeffs: Vec<FE> = (0..(1u64 << layout.effective_k)).map(FE::from).collect();
        // Height 7 folds no layer at these parameters, so table 5 is standalone
        // and its terminal polynomial is part of the round-4 sequence.
        let standalone_terminal: Vec<FE> = (0..(1u64 << (7 - blowup_log))).map(FE::from).collect();

        let grinding_factor: u8 = 4;
        let num_queries = 3;

        let seed_transcript = Transcript::new(b"batched_round4_test");
        let mut transcript_a = seed_transcript.clone();
        let mut transcript_b = seed_transcript.clone();

        // --- Clone A: prover-inline sequence, by hand ---
        absorb_shape_histogram(&mut transcript_a, &heights, &widths);
        let alpha_a = transcript_a.sample_field_element();
        for c in &standalone_terminal {
            transcript_a.append_field_element(c);
        }

        let mut betas_a = Vec::with_capacity(layer_roots.len() + 1);
        for root in &layer_roots {
            let beta = transcript_a.sample_field_element();
            transcript_a.append_bytes(root);
            betas_a.push(beta);
        }
        betas_a.push(transcript_a.sample_field_element());
        for c in &final_poly_coeffs {
            transcript_a.append_field_element(c);
        }
        assert_eq!(
            betas_a.len(),
            layout.total_folds as usize,
            "one beta per fold, matching batched_commit_phase"
        );

        let grinding_seed_a = transcript_a.state();
        // Test-only: derive a real PoW nonce so the grinding step is exercised
        // identically by both sides (the nonce search itself is not under test).
        let nonce = crate::grinding::generate_nonce::<
            crate::config::GrindingDigest<DefaultStarkHash>,
        >(&grinding_seed_a, grinding_factor)
        .expect("a valid grinding nonce exists for this small grinding_factor");
        transcript_a.append_bytes(&nonce.to_be_bytes());

        let iotas_a: Vec<usize> = (0..num_queries)
            .map(|_| transcript_a.sample_u64(1u64 << 9) as usize)
            .collect();

        // --- Clone B: shared replay routine ---
        let standalone: Vec<Option<&[FE]>> = vec![
            None,
            None,
            None,
            None,
            None,
            Some(standalone_terminal.as_slice()),
        ];
        let result = derive_batched_fri_challenges(
            &mut transcript_b,
            &heights,
            &widths,
            &layer_roots,
            &final_poly_coeffs,
            &standalone,
            blowup_log,
            k,
            grinding_factor,
            Some(nonce),
            num_queries,
        )
        .expect("a well-formed layer-root and coefficient count");

        assert_eq!(result.alpha, alpha_a, "alpha mismatch");
        assert_eq!(result.betas, betas_a, "beta vector mismatch");
        assert_eq!(result.layout, layout, "layout mismatch");
        assert_eq!(
            result.grinding_seed, grinding_seed_a,
            "grinding seed mismatch"
        );
        assert_eq!(result.iotas, iotas_a, "iotas mismatch");
        assert!(
            result.iotas.iter().all(|&i| i < 1usize << 9),
            "iotas must be row-pair indices in the tallest domain"
        );
    }

    /// A layer-root or coefficient count that disagrees with the shape's layout is
    /// prover-supplied, so it is a rejection rather than a panic.
    #[test]
    fn derive_rejects_a_layer_count_that_contradicts_the_shape() {
        let heights: Vec<usize> = vec![10, 8];
        let widths: Vec<usize> = vec![2, 3];
        let (blowup_log, k) = (1u32, 5u32);
        let layout = BatchedFriLayout::new(10, 8, blowup_log, k);
        let coeffs: Vec<FE> = vec![FE::one(); 1usize << layout.effective_k];
        let roots: Vec<[u8; 32]> = vec![[0u8; 32]; layout.num_committed];

        let no_standalone: Vec<Option<&[FE]>> = vec![None; heights.len()];
        let mut ok = Transcript::new(b"reject");
        assert!(
            derive_batched_fri_challenges(
                &mut ok,
                &heights,
                &widths,
                &roots,
                &coeffs,
                &no_standalone,
                blowup_log,
                k,
                0,
                None,
                1
            )
            .is_some()
        );

        let mut too_few = Transcript::new(b"reject");
        assert!(
            derive_batched_fri_challenges(
                &mut too_few,
                &heights,
                &widths,
                &roots[..roots.len() - 1],
                &coeffs,
                &no_standalone,
                blowup_log,
                k,
                0,
                None,
                1
            )
            .is_none(),
            "one fewer layer root than the shape implies must be rejected"
        );

        let mut bad_coeffs = Transcript::new(b"reject");
        assert!(
            derive_batched_fri_challenges(
                &mut bad_coeffs,
                &heights,
                &widths,
                &roots,
                &coeffs[..coeffs.len() - 1],
                &no_standalone,
                blowup_log,
                k,
                0,
                None,
                1
            )
            .is_none(),
            "a short terminal polynomial must be rejected"
        );
    }

    /// `heights` comes from proof-supplied trace lengths, so every out-of-range
    /// value is a rejection rather than a shift overflow or a layout assert.
    #[test]
    fn derive_rejects_out_of_range_heights_without_panicking() {
        let widths = vec![2usize, 3];
        let (blowup_log, k) = (1u32, 5u32);
        let coeffs: Vec<FE> = vec![FE::one(); 1usize << k];
        let roots: Vec<[u8; 32]> = vec![[0u8; 32]; 3];

        let derive = |heights: &[usize]| {
            let no_standalone: Vec<Option<&[FE]>> = vec![None; heights.len()];
            derive_batched_fri_challenges(
                &mut Transcript::new(b"range"),
                heights,
                &widths,
                &roots,
                &coeffs,
                &no_standalone,
                blowup_log,
                k,
                0,
                None,
                1,
            )
            .is_some()
        };

        assert!(derive(&[10, 8]), "a well-formed shape is accepted");
        assert!(!derive(&[0, 0]), "a zero height must be rejected");
        assert!(
            !derive(&[10, 0]),
            "a height below the blowup must be rejected"
        );
        assert!(
            !derive(&[u32::BITS as usize, 8]),
            "a height at the shift width must be rejected"
        );
        assert!(
            !derive(&[usize::MAX, 8]),
            "an absurd height must be rejected, not wrapped by the u32 cast"
        );
        let empty: [usize; 0] = [];
        assert!(!derive(&empty), "an empty epoch must be rejected");
    }

    /// Tampering the shape histogram (without changing anything else) must change
    /// the derived batching challenge α — the structural binding that protects the
    /// fold/injection schedule. Heights and widths are both bound (M-13a), so a
    /// change to either alone must move α.
    #[test]
    fn absorb_shape_histogram_binds_heights_and_widths_into_alpha() {
        let heights: Vec<usize> = vec![10, 10, 8, 8, 8, 5];
        let widths: Vec<usize> = vec![4, 4, 2, 2, 2, 1];

        let alpha_of = |h: &[usize], w: &[usize]| {
            let mut t = Transcript::new(b"histogram_binding_test");
            absorb_shape_histogram(&mut t, h, w);
            t.sample_field_element()
        };

        let base = alpha_of(&heights, &widths);

        let mut other_height = heights.clone();
        other_height[5] = 6;
        assert_ne!(
            base,
            alpha_of(&other_height, &widths),
            "different height histograms must yield different alpha"
        );

        let mut other_width = widths.clone();
        other_width[5] = 2;
        assert_ne!(
            base,
            alpha_of(&heights, &other_width),
            "different width histograms must yield different alpha"
        );

        // The length prefix plus fixed-width fields make the encoding injective:
        // swapping a (height, width) pair between tables also moves alpha.
        let swapped_h = vec![10, 10, 8, 8, 5, 8];
        let swapped_w = vec![4, 4, 2, 2, 1, 2];
        assert_ne!(
            base,
            alpha_of(&swapped_h, &swapped_w),
            "table order must be bound, not just the multiset"
        );
    }
}
