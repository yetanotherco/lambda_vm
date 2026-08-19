//! Round 4 of the batched path: ONE FRI instance over the epoch's height-combined
//! DEEP codewords.
//!
//! # The transcript sequence, and why it has one owner
//!
//! ```text
//! shape histogram → α → (β, layer root)* → β_final → terminal coeffs → grinding → iotas
//! ```
//!
//! [`commit_batched_fri`] walks it on the prover's side;
//! [`crate::fri::batched::derive_batched_fri_challenges`] walks it on the
//! verifier's. The two are pinned to each other by
//! `prover_commit_matches_verifier_derivation`, not by review of two call sites.
//! α is sampled AFTER the shape is absorbed and BEFORE any codeword is combined,
//! which is why this function takes a `combine` closure rather than the codewords:
//! the prover cannot mix with α until the transcript has produced it, and the
//! closure is where a caller streams table by table (see
//! [`crate::fri::batched::HeightCombiner`]).
//!
//! # ★ TWO instance classes, and the index rule between them
//!
//! Not every table belongs in the batch. A table whose own FRI commits ZERO
//! layers gains nothing from being batched — there is no layer for the batch to
//! share — while it pays the full lift to the tallest domain, which is where the
//! proximity-gaps term's `|D0|^2` lives. At the measured epoch that is 13 of 28
//! legs carrying 92% of the batch's width. [`FriInstancePlan`] partitions them,
//! and the excluded tables keep a terminal-only instance
//! ([`verify_standalone_fri_query`]) that costs one polynomial and no layers.
//!
//! The MMCS is untouched by this split — it still commits every table, so the
//! one-shared-authentication-path win survives whole. What differs is the index
//! SPACE: the batched class reads `iota` directly, a standalone table at height
//! `h` reads `iota >> (h_max - h)`. Both classes need a tamper control, since a
//! control that only touched the batched one would pass under any convention for
//! the other.
//!
//! # Query indices and the injection convention
//!
//! One `iota` per query, drawn from `[0, 2^(h_max-1))` — a row-PAIR index in the
//! TALLEST codeword's domain. Every shorter object is located by shifting it
//! down, which is what makes "one index, shared across all tables" true rather
//! than aspirational:
//!
//! - a matrix of height `h` in a round whose own tallest matrix is `h_max_round`
//!   is opened at MMCS leaf `iota >> (h_max_fri - h)` — but note that
//!   [`crate::fri::mmcs::MixedMmcs::verify_batch`] wants an index in ITS OWN
//!   space, so a round whose `h_max_round` is below the FRI's must first reduce
//!   (see that module's index-convention section, and [`reduce_iota_to_round`]).
//! - the codeword bucket at height `h` is read at position
//!   [`injection_position`], which is exactly one of the two rows of the pair the
//!   MMCS opened. That coincidence is not luck: both are the same row-pair
//!   layout, which is why a single opening serves both the authentication and the
//!   FRI join.
//!
//! # What "injection" costs the verifier
//!
//! The prover's [`crate::fri::batched::batched_commit_phase`] folds, then adds
//! `β² · bucket_h` to the running codeword before committing the layer. So the
//! verifier's per-query recursion adds the same term to the value it computed by
//! folding — and only to that value. The symmetric value at each layer comes from
//! the proof and is Merkle-authenticated against the layer root, so it already
//! carries its own injection; re-adding one would double it.

use crypto::fiat_shamir::is_transcript::{IsStarkTranscript, IsTranscript};
use crypto::merkle_tree::proof::verify_merkle_path;
use math::field::element::FieldElement;
use math::field::traits::{IsFFTField, IsField, IsSubFieldOf};
use math::traits::AsBytes;

use crate::config::{Commitment, StarkHash};
use crate::fri::batched::{
    BatchedFriLayout, FriInstancePlan, absorb_shape_histogram, batched_commit_phase,
    derive_batched_fri_challenges,
};
use crate::fri::fri_commitment::FriLayer;
use crate::fri::fri_decommit::FriDecommitment;
use crate::grinding;

/// What the prover produced in the batched round 4, plus the challenges it drew
/// on the way. The layers are kept so the caller can run the query phase over
/// them; everything else is what goes on the wire.
pub struct BatchedFriCommit<E: IsField + 'static, H: StarkHash>
where
    FieldElement<E>: AsBytes + Sync + Send,
{
    pub layers: Vec<FriLayer<E, H::Pair<E>>>,
    pub layer_roots: Vec<Commitment>,
    pub final_poly_coeffs: Vec<FieldElement<E>>,
    pub layout: BatchedFriLayout,
    /// The grinding nonce, `None` when `grinding_factor == 0`.
    pub nonce: Option<u64>,
    /// Row-pair indices in the tallest domain, one per query.
    pub iotas: Vec<usize>,
    /// The mixing challenge the codewords were combined with. Kept because the
    /// query phase needs it to rebuild each table's contribution.
    pub alpha: FieldElement<E>,
    /// Which tables this instance carries, and which keep a terminal-only
    /// instance of their own. See [`FriInstancePlan`].
    pub plan: FriInstancePlan,
}

/// Prover side of the batched round-4 sequence.
///
/// `heights[t]` is `log2` of table `t`'s LDE length and `widths[t]` its committed
/// column count, both in the epoch's canonical table order — the same order the
/// verifier rebuilds from the AIR set, and the same order `combine` must absorb
/// codewords in, since absorption order is what defines the α powers.
///
/// `combine` receives α and returns the per-height buckets (see
/// [`crate::fri::batched::HeightCombiner::finish`]). It is a closure rather than
/// a materialized `Vec` so a caller can produce one table's DEEP codeword,
/// absorb it and drop it: holding all of them at once is the memory cost
/// batching exists to remove.
#[allow(clippy::too_many_arguments)]
pub fn commit_batched_fri<F, E, T, H, C>(
    transcript: &mut T,
    heights: &[usize],
    widths: &[usize],
    combine: C,
    coset_offset: &FieldElement<F>,
    blowup_log: u32,
    final_poly_log_degree: u32,
    grinding_factor: u8,
    num_queries: usize,
) -> BatchedFriCommit<E, H>
where
    F: IsFFTField + IsSubFieldOf<E> + 'static,
    E: IsField + 'static + Send + Sync,
    T: IsStarkTranscript<E, F> + Clone,
    H: StarkHash,
    C: FnOnce(&FieldElement<E>, &FriInstancePlan) -> Vec<Option<Vec<FieldElement<E>>>>,
    FieldElement<F>: AsBytes + Sync + Send,
    FieldElement<E>: AsBytes + Sync + Send,
{
    // Derived from the shape, exactly as the verifier derives it — the partition
    // is never sent. The tables whose own FRI commits no layer are left out of the
    // batch: they gain nothing from it and pay the full lift to the tallest
    // domain, which is where the proximity-gaps term's `|D0|^2` lives.
    let plan = FriInstancePlan::new(heights, blowup_log, final_poly_log_degree)
        .expect("commit_batched_fri: the epoch's shape is the prover's own");
    let h_max = plan.h_max;

    absorb_shape_histogram::<E, T>(transcript, heights, widths);
    let alpha = transcript.sample_field_element();

    let combined = combine(&alpha, &plan);
    let (final_poly_coeffs, layers) = batched_commit_phase::<F, E, T, H>(
        combined,
        transcript,
        coset_offset,
        blowup_log,
        final_poly_log_degree,
    );
    let layer_roots: Vec<Commitment> = layers.iter().map(|layer| layer.merkle_tree.root).collect();

    // Grinding runs on the CONFIGURATION's transcript hash, not a hard-wired
    // one — the same rule the unbatched `prover.rs` follows. `H` names both the
    // commitment family and the Fiat-Shamir hash, so a batched proof committed
    // with BLAKE3 grinds with BLAKE3 and one committed with keccak grinds with
    // keccak, without either side being told twice.
    let nonce = (grinding_factor > 0).then(|| {
        let value = grinding::generate_nonce::<crate::config::GrindingDigest<H>>(
            &transcript.state(),
            grinding_factor,
        )
        .expect("nonce not found");
        transcript.append_bytes(&value.to_be_bytes());
        value
    });

    let iotas = (0..num_queries)
        .map(|_| transcript.sample_u64(1u64 << (h_max - 1)) as usize)
        .collect();

    BatchedFriCommit {
        layers,
        layer_roots,
        final_poly_coeffs,
        layout: BatchedFriLayout::new(plan.h_max, plan.h_min, blowup_log, final_poly_log_degree),
        nonce,
        iotas,
        alpha,
        plan,
    }
}

/// Verify one query against a STANDALONE table's terminal-only instance.
///
/// A table whose own FRI commits no layer has a terminal codeword that IS its
/// deep-composition codeword, so there is nothing to fold and nothing to
/// authenticate: the check is that the value the query opened is the value the
/// sent terminal polynomial encodes at that position.
///
/// ★ `iota` is the SHARED batched query index and is reduced here — the two
/// instance classes read the same index in different spaces (see
/// [`FriInstancePlan`]). `deep` is the table's own deep-composition pair at its
/// reduced row pair, which the caller reconstructs from authenticated openings.
///
/// Returns `false` on every malformed input; it never panics.
pub fn verify_standalone_fri_query<E>(
    iota: usize,
    h_max_fri: usize,
    h_table: usize,
    deep: (&FieldElement<E>, &FieldElement<E>),
    terminal_codeword: &[FieldElement<E>],
) -> bool
where
    E: IsField + 'static,
{
    let Some(reduced) = reduce_iota_to_round(iota, h_max_fri, h_table) else {
        return false;
    };
    terminal_codeword
        .get(reduced * 2)
        .is_some_and(|t| deep.0 == t)
        && terminal_codeword
            .get(reduced * 2 + 1)
            .is_some_and(|t| deep.1 == t)
}

/// Position, inside the codeword of height `h`, that query `iota` reads.
///
/// `iota` is a row-pair index in the tallest domain (height `h_max`); the layer
/// whose codeword has height `h` is reached after `h_max - h` folds, and the
/// query's position there is `iota >> (h_max - h - 1)`. Both rows of the pair a
/// height-`h` MMCS opening returns — leaf `iota >> (h_max - h)`, i.e. LDE rows
/// `2k` and `2k+1` — are candidates, and the low bit of this position picks
/// between them; see [`injected_value_at_query`].
///
/// Not defined at `h == h_max`: the tallest codeword is the FRI's layer 0, which
/// the query reads as a PAIR (`2·iota`, `2·iota+1`) rather than at one position.
#[inline]
pub fn injection_position(iota: usize, h_max: usize, h: usize) -> usize {
    debug_assert!(
        h < h_max,
        "the tallest codeword is read as a pair, not at a position"
    );
    iota >> (h_max - h - 1)
}

/// The value a height-`h` matrix contributes to its injection layer, chosen from
/// the row pair its MMCS opening returned.
///
/// `evaluation` is the opening's row `2k` and `evaluation_sym` its row `2k+1`,
/// with `k = iota >> (h_max - h)`. The pair straddles the injection position, so
/// the choice is exactly that position's low bit.
#[inline]
pub fn injected_value_at_query<'a, E: IsField>(
    iota: usize,
    h_max: usize,
    h: usize,
    evaluation: &'a FieldElement<E>,
    evaluation_sym: &'a FieldElement<E>,
) -> &'a FieldElement<E> {
    if injection_position(iota, h_max, h) & 1 == 0 {
        evaluation
    } else {
        evaluation_sym
    }
}

/// Reduce a FRI query index to the index space of a round whose tallest matrix
/// is shorter than the FRI's.
///
/// [`crate::fri::mmcs::MixedMmcs::verify_batch`] walks its path with the LOW bits
/// of the index it is given, while it locates a short matrix inside the tree by
/// the HIGH bits — consistent only when the index comes from that tree's own
/// `h_max`. The batched preprocessed round is the case that breaks it (its
/// tallest matrix sits below the FRI's), so every caller reduces here rather than
/// each writing the shift out. Returns `None` when the round claims to be TALLER
/// than the FRI, which no honest shape can be.
#[inline]
pub fn reduce_iota_to_round(iota: usize, h_max_fri: usize, h_max_round: usize) -> Option<usize> {
    (h_max_round <= h_max_fri).then(|| iota >> (h_max_fri - h_max_round))
}

/// Verify one query of the batched FRI: the fold-with-injection recursion, every
/// committed layer's opening, and the terminal check.
///
/// `p0` is the query's pair of values in the tallest codeword — the α-mixed DEEP
/// evaluations of the tables at height `h_max`, at LDE positions `2·iota` and
/// `2·iota + 1`. `bucket_at_height[h]` is `Some(v)` when at least one table has
/// height `h < h_max`, with `v` that height group's α-mixed value at
/// [`injection_position`]; `None` when no table sits at `h`. Both are the
/// caller's to reconstruct from authenticated openings — this function does no
/// authentication of trace data, only of FRI layers.
///
/// Returns `false` on every malformed input; it never panics.
#[allow(clippy::too_many_arguments)]
pub fn verify_batched_fri_query<F, E, H>(
    layer_roots: &[Commitment],
    betas: &[FieldElement<E>],
    layout: &BatchedFriLayout,
    h_max: usize,
    iota: usize,
    decommitment: &FriDecommitment<E>,
    evaluation_point_inv: &FieldElement<F>,
    p0: (&FieldElement<E>, &FieldElement<E>),
    bucket_at_height: &[Option<FieldElement<E>>],
    terminal_codeword: &[FieldElement<E>],
) -> bool
where
    F: IsFFTField + IsSubFieldOf<E> + 'static,
    E: IsField + 'static,
    H: StarkHash,
    FieldElement<F>: AsBytes + Sync + Send,
    FieldElement<E>: AsBytes + Sync + Send,
{
    // The decommitment vectors are prover-supplied and are NOT bound into the
    // transcript, so their lengths are pinned here before anything zips them —
    // the same reason `step_3_verify_fri` pins them in the unbatched path. A
    // short vector would make the fold loop run fewer rounds and accept the query
    // without ever reaching the terminal.
    if layer_roots.len() != layout.num_committed
        || decommitment.layers_auth_paths.len() != layout.num_committed
        || decommitment.layers_evaluations_sym.len() != layout.num_committed
        || betas.len() != layout.num_committed + usize::from(layout.total_folds > 0)
    {
        return false;
    }
    if h_max == 0 || h_max >= usize::BITS as usize || iota >= 1usize << (h_max - 1) {
        return false;
    }
    if bucket_at_height.len() < h_max {
        return false;
    }

    // No-fold case: the codeword never folds, so the terminal IS the tallest
    // codeword and the query's two points sit at `2·iota` and `2·iota + 1`. No
    // bucket can exist below `h_max` here — `h_min == h_max` is what makes
    // `total_folds` zero — so there is nothing to inject.
    if layout.total_folds == 0 {
        return terminal_codeword.get(iota * 2).is_some_and(|t| p0.0 == t)
            && terminal_codeword
                .get(iota * 2 + 1)
                .is_some_and(|t| p0.1 == t);
    }

    // First fold: layer 0 (the tallest codeword) is not committed, so this fold
    // consumes `p0` rather than an authenticated opening. Then the height just
    // below joins, exactly as `batched_commit_phase` does before it commits.
    let mut point_inv = evaluation_point_inv.clone();
    let mut v = (p0.0 + p0.1) + &point_inv * &betas[0] * (p0.0 - p0.1);
    let mut index = iota;
    inject(&mut v, &betas[0], bucket_at_height, h_max - 1);

    let mut openings_ok = true;
    for i in 0..layout.num_committed {
        let evaluation_sym = &decommitment.layers_evaluations_sym[i];
        openings_ok &= verify_layer_opening::<E, H>(
            &layer_roots[i],
            decommitment.layers_auth_paths[i].merkle_path.as_slice(),
            &v,
            evaluation_sym,
            index,
        );

        point_inv = point_inv.square();
        v = (&v + evaluation_sym) + &point_inv * &betas[i + 1] * (&v - evaluation_sym);
        index >>= 1;
        // The injection height descends with the running codeword. `checked_sub`
        // rather than `h_max - 2 - i`: `layout`'s fields are only consistent with
        // `h_max` when the layout was DERIVED from the same heights, and this
        // function is on the verifier's path, where an overflow panic is not a
        // rejection. An inconsistent layout simply injects nothing and fails at
        // the terminal.
        if let Some(height) = (h_max - 1).checked_sub(i + 1) {
            inject(&mut v, &betas[i + 1], bucket_at_height, height);
        }
    }

    // `v` is now the query's value in the terminal codeword and `index` its
    // position there. `.get` fails closed on an out-of-range index.
    openings_ok & terminal_codeword.get(index).is_some_and(|t| &v == t)
}

/// `running += β² · bucket_h` for the height the running codeword has just
/// reached. A no-op when no table sits at that height, and when the height is
/// below the terminal (`bucket_at_height` is indexed by height, so a fold that
/// runs past index 0 has nothing to read).
fn inject<E: IsField>(
    value: &mut FieldElement<E>,
    beta: &FieldElement<E>,
    bucket_at_height: &[Option<FieldElement<E>>],
    height: usize,
) {
    if let Some(Some(contribution)) = bucket_at_height.get(height) {
        *value = &*value + &(beta.square() * contribution);
    }
}

/// Authenticate a committed FRI layer's row pair against its root. `index` is the
/// query's position in that layer; the leaf is the pair at `index >> 1`, ordered
/// by `index`'s low bit — the same convention the unbatched
/// `verify_fri_layer_openings` uses, and the same one `query_phase` opens with.
fn verify_layer_opening<E, H>(
    root: &Commitment,
    auth_path: &[Commitment],
    evaluation: &FieldElement<E>,
    evaluation_sym: &FieldElement<E>,
    index: usize,
) -> bool
where
    E: IsField + 'static,
    H: StarkHash,
    FieldElement<E>: AsBytes + Sync + Send,
{
    let leaf = if index % 2 == 1 {
        vec![evaluation_sym.clone(), evaluation.clone()]
    } else {
        vec![evaluation.clone(), evaluation_sym.clone()]
    };
    verify_merkle_path::<H::Batched<E>>(auth_path, root, index >> 1, &leaf)
}

/// Replay the batched round-4 transcript sequence and return the challenges,
/// or `None` when the proof's shape contradicts the epoch's.
///
/// A thin alias for [`derive_batched_fri_challenges`], re-exported here so the
/// verifier reaches the sequence through the same module the prover's
/// [`commit_batched_fri`] lives in — the two are one protocol, and splitting them
/// across modules is how they drift.
#[allow(clippy::too_many_arguments)]
pub fn replay_batched_fri<E, T>(
    transcript: &mut T,
    heights: &[usize],
    widths: &[usize],
    layer_roots: &[Commitment],
    final_poly_coeffs: &[FieldElement<E>],
    blowup_log: u32,
    final_poly_log_degree: u32,
    grinding_factor: u8,
    nonce: Option<u64>,
    num_queries: usize,
) -> Option<crate::fri::batched::BatchedFriChallenges<E>>
where
    E: IsField,
    T: IsTranscript<E>,
{
    derive_batched_fri_challenges(
        transcript,
        heights,
        widths,
        layer_roots,
        final_poly_coeffs,
        blowup_log,
        final_poly_log_degree,
        grinding_factor,
        nonce,
        num_queries,
    )
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::config::KeccakStarkHash;
    use crate::fri::batched::{HeightCombiner, combine_by_height};
    use crate::fri::terminal::terminal_codeword_from_coeffs;
    use crypto::fiat_shamir::default_transcript::DefaultTranscript;
    use math::fft::bit_reversing::{in_place_bit_reverse_permute, reverse_index};
    use math::field::goldilocks::GoldilocksField;
    use math::polynomial::Polynomial;

    pub(crate) type F = GoldilocksField;
    pub(crate) type FE = FieldElement<GoldilocksField>;
    pub(crate) type Transcript = DefaultTranscript<GoldilocksField>;

    pub(crate) const BLOWUP_LOG: u32 = 1;
    pub(crate) const FINAL_POLY_LOG_DEGREE: u32 = 1;
    pub(crate) const COSET_OFFSET: u64 = 3;

    /// One synthetic table: a genuinely low-degree codeword at its own height.
    pub(crate) struct FakeTable {
        pub height: usize,
        pub width: usize,
        pub codeword: Vec<FE>,
    }

    /// A codeword of height `h` that IS a Reed-Solomon word of rate `2^-BLOWUP_LOG`
    /// on the coset the batched FRI will read it at.
    ///
    /// The coset matters and is the one thing easy to get wrong here: folding
    /// squares the offset, so the layer a height-`h` bucket is injected into lives
    /// on `offset^(2^(h_max-h))·⟨ω⟩`, not on `offset·⟨ω⟩`. A word built on the
    /// wrong coset is still low degree — the map is a rescaling of the argument —
    /// so it would pass a degree check while making the terminal reconstruction
    /// disagree, which is exactly the failure the honest-path test has to be able
    /// to see.
    pub(crate) fn low_degree_codeword(h: usize, h_max: usize, seed: u64) -> Vec<FE> {
        let num_coeffs = 1usize << (h as u32 - BLOWUP_LOG);
        let coeffs: Vec<FE> = (0..num_coeffs)
            .map(|i| FE::from(seed.wrapping_mul(97).wrapping_add(i as u64 * 31 + 1)))
            .collect();
        let offset = FE::from(COSET_OFFSET).pow(1u64 << (h_max - h));
        let mut natural = Polynomial::evaluate_offset_fft::<F>(
            &Polynomial::new(&coeffs),
            1usize << BLOWUP_LOG,
            Some(num_coeffs),
            &offset,
        )
        .expect("coset evaluation");
        in_place_bit_reverse_permute(&mut natural);
        natural
    }

    /// Four tables over three heights, the shape the batched path has to handle:
    /// several tables sharing the tallest height (so the base group batches), one
    /// at an intermediate height (so an injection lands on a committed layer) and
    /// one at the terminal height (so the FINAL fold's injection is exercised —
    /// the case #768's loop missed).
    pub(crate) fn fixture() -> Vec<FakeTable> {
        let h_max = 5;
        vec![
            FakeTable {
                height: 5,
                width: 3,
                codeword: low_degree_codeword(5, h_max, 11),
            },
            FakeTable {
                height: 4,
                width: 2,
                codeword: low_degree_codeword(4, h_max, 23),
            },
            FakeTable {
                height: 5,
                width: 7,
                codeword: low_degree_codeword(5, h_max, 41),
            },
            FakeTable {
                height: 2,
                width: 1,
                codeword: low_degree_codeword(2, h_max, 59),
            },
        ]
    }

    pub(crate) fn heights_of(tables: &[FakeTable]) -> Vec<usize> {
        tables.iter().map(|t| t.height).collect()
    }

    pub(crate) fn widths_of(tables: &[FakeTable]) -> Vec<usize> {
        tables.iter().map(|t| t.width).collect()
    }

    /// Run the prover's batched round 4 over `tables`, streaming the codewords
    /// into the combiner one at a time — the shape a real prover uses.
    pub(crate) fn commit_fixture(
        tables: &[FakeTable],
        transcript: &mut Transcript,
        grinding_factor: u8,
        num_queries: usize,
    ) -> BatchedFriCommit<F, KeccakStarkHash> {
        let heights = heights_of(tables);
        let widths = widths_of(tables);
        commit_batched_fri::<F, F, Transcript, KeccakStarkHash, _>(
            transcript,
            &heights,
            &widths,
            |alpha, plan| {
                // Only the batched class is mixed in, and in the plan's order —
                // absorption order is what defines the alpha powers, so a caller
                // that absorbed the standalone tables too would shift every
                // power and agree with no verifier.
                let mut combiner = HeightCombiner::new(*alpha);
                for &t in &plan.batched {
                    combiner.absorb(&tables[t].codeword, tables[t].height);
                }
                combiner.finish()
            },
            &FE::from(COSET_OFFSET),
            BLOWUP_LOG,
            FINAL_POLY_LOG_DEGREE,
            grinding_factor,
            num_queries,
        )
    }

    /// υ⁻¹ for query `iota`: the inverse of the tallest coset's element at
    /// FRI-order position `2·iota`, matching the unbatched verifier's
    /// `query_challenge_to_evaluation_point`.
    pub(crate) fn evaluation_point_inv(iota: usize, h_max: usize) -> FE {
        let n = 1usize << h_max;
        let omega = F::get_primitive_root_of_unity(h_max as u64).expect("root of unity");
        let point = FE::from(COSET_OFFSET) * omega.pow(reverse_index(iota * 2, n as u64));
        point.inv().expect("query point is never zero")
    }

    /// What the verifier must reconstruct from authenticated openings: the α-mixed
    /// value of every height group at this query's position. Here it is read
    /// straight off the combined buckets, which is the oracle — `combine_by_height`
    /// has its own tests, and the point of this one is the fold recursion.
    pub(crate) fn query_inputs(
        tables: &[FakeTable],
        alpha: &FE,
        iota: usize,
    ) -> ((FE, FE), Vec<Option<FE>>) {
        let plan = FriInstancePlan::new(&heights_of(tables), BLOWUP_LOG, FINAL_POLY_LOG_DEGREE)
            .expect("the fixture's shape partitions");
        let h_max = plan.h_max;
        let inputs: Vec<(Vec<FE>, usize)> = plan
            .batched
            .iter()
            .map(|&t| (tables[t].codeword.clone(), tables[t].height))
            .collect();
        let combined = combine_by_height(&inputs, alpha);

        let tallest = combined[h_max].as_ref().expect("tallest bucket exists");
        let p0 = (tallest[iota * 2], tallest[iota * 2 + 1]);

        let buckets = (0..h_max)
            .map(|h| {
                combined
                    .get(h)
                    .and_then(|slot| slot.as_ref())
                    .map(|codeword| codeword[injection_position(iota, h_max, h)])
            })
            .collect();
        (p0, buckets)
    }

    /// Verify one query end to end against the committed layers.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn verify_one_query(
        commit: &BatchedFriCommit<F, KeccakStarkHash>,
        betas: &[FE],
        h_max: usize,
        iota: usize,
        decommitment: &FriDecommitment<F>,
        p0: (&FE, &FE),
        buckets: &[Option<FE>],
        layer_roots: &[Commitment],
        final_poly_coeffs: &[FE],
    ) -> bool {
        let terminal_offset = FE::from(COSET_OFFSET).pow(1u64 << commit.layout.total_folds);
        let terminal = terminal_codeword_from_coeffs::<F, F>(
            final_poly_coeffs,
            &terminal_offset,
            commit.layout.terminal_len,
        );
        verify_batched_fri_query::<F, F, KeccakStarkHash>(
            layer_roots,
            betas,
            &commit.layout,
            h_max,
            iota,
            decommitment,
            &evaluation_point_inv(iota, h_max),
            p0,
            buckets,
            &terminal,
        )
    }

    /// The prover's inline sequence and the verifier's replay are ONE protocol;
    /// this is what pins them together. Every challenge, not only the iotas —
    /// α gates the height combination and the βs gate every fold, so an
    /// agreement that held only at the query indices would still be a broken
    /// proof system.
    #[test]
    fn prover_commit_matches_verifier_derivation() {
        let tables = fixture();
        let mut prover_transcript = Transcript::new(b"batched_round4");
        let mut verifier_transcript = prover_transcript.clone();

        let commit = commit_fixture(&tables, &mut prover_transcript, 4, 6);

        let replay = replay_batched_fri::<F, Transcript>(
            &mut verifier_transcript,
            &heights_of(&tables),
            &widths_of(&tables),
            &commit.layer_roots,
            &commit.final_poly_coeffs,
            BLOWUP_LOG,
            FINAL_POLY_LOG_DEGREE,
            4,
            commit.nonce,
            6,
        )
        .expect("an honest shape must derive");

        assert_eq!(replay.alpha, commit.alpha, "α must agree");
        assert_eq!(replay.layout, commit.layout, "the fold layout must agree");
        assert_eq!(replay.iotas, commit.iotas, "the query indices must agree");
        assert_eq!(
            replay.betas.len(),
            commit.layout.num_committed + 1,
            "one β per committed layer plus the final fold"
        );
        assert!(
            crate::grinding::is_valid_nonce::<crate::config::GrindingDigest<KeccakStarkHash>>(
                &replay.grinding_seed,
                commit.nonce.expect("grinding was requested"),
                4
            ),
            "the replayed grinding seed must accept the prover's nonce"
        );
        assert_eq!(
            prover_transcript.state(),
            verifier_transcript.state(),
            "both sides must end in the same transcript state"
        );
    }

    /// The honest path, and it is not vacuous: the fixture spans three heights,
    /// so this exercises the base group, an injection into a committed layer and
    /// an injection at the final fold. If the injection convention or the
    /// position derivation were wrong, the terminal check would fail.
    #[test]
    fn honest_batched_queries_verify() {
        let tables = fixture();
        let h_max = 5;
        let mut transcript = Transcript::new(b"batched_round4");
        let commit = commit_fixture(&tables, &mut transcript, 0, 8);

        let decommitments =
            crate::fri::query_phase::<F, KeccakStarkHash>(&commit.layers, &commit.iotas);

        let mut verifier_transcript = Transcript::new(b"batched_round4");
        let replay = replay_batched_fri::<F, Transcript>(
            &mut verifier_transcript,
            &heights_of(&tables),
            &widths_of(&tables),
            &commit.layer_roots,
            &commit.final_poly_coeffs,
            BLOWUP_LOG,
            FINAL_POLY_LOG_DEGREE,
            0,
            None,
            8,
        )
        .expect("an honest shape must derive");

        assert!(commit.layout.num_committed >= 1, "the fixture must fold");
        for (query, &iota) in commit.iotas.iter().enumerate() {
            let (p0, buckets) = query_inputs(&tables, &replay.alpha, iota);
            assert!(
                verify_one_query(
                    &commit,
                    &replay.betas,
                    h_max,
                    iota,
                    &decommitments[query],
                    (&p0.0, &p0.1),
                    &buckets,
                    &commit.layer_roots,
                    &commit.final_poly_coeffs,
                ),
                "honest query {query} (iota {iota}) must verify"
            );
        }
    }

    /// The MMCS row pair a query opens at height `h` and the FRI position the
    /// injection reads must be the SAME two rows. That coincidence is what lets
    /// one opening serve both the authentication and the FRI join, and it is a
    /// property of the two index derivations, so it is worth pinning exhaustively
    /// rather than sampling.
    #[test]
    fn injection_position_lands_inside_the_mmcs_row_pair() {
        let h_max = 6;
        for iota in 0..(1usize << (h_max - 1)) {
            for h in 1..h_max {
                let position = injection_position(iota, h_max, h);
                let mmcs_leaf = iota >> (h_max - h);
                assert_eq!(
                    position >> 1,
                    mmcs_leaf,
                    "height {h}, iota {iota}: the injection position must sit in the opened leaf"
                );
                assert!(
                    position < (1usize << h),
                    "height {h}, iota {iota}: position must stay inside the codeword"
                );
            }
        }
    }

    /// `reduce_iota_to_round` is the documented remedy for the one case where a
    /// round's tallest matrix is below the FRI's. Pin both that it is the shift
    /// the MMCS wants and that it refuses the impossible direction rather than
    /// shifting by a negative amount.
    #[test]
    fn reduce_iota_to_round_matches_the_mmcs_index_space() {
        let h_max_fri = 6;
        for iota in 0..(1usize << (h_max_fri - 1)) {
            for h_max_round in 1..=h_max_fri {
                let reduced =
                    reduce_iota_to_round(iota, h_max_fri, h_max_round).expect("round is shorter");
                assert!(
                    reduced < (1usize << (h_max_round - 1)),
                    "the reduced index must land in the round's own leaf range"
                );
            }
        }
        assert!(
            reduce_iota_to_round(0, 4, 5).is_none(),
            "a round taller than the FRI is not a shape any honest epoch has"
        );
    }

    /// A width the epoch did not commit to moves α, and therefore every fold and
    /// every query index. This is the shape binding doing its job one level up
    /// from the leaf: the leaf header binds a mis-parse, this binds a mis-shaped
    /// epoch.
    #[test]
    fn a_tampered_shape_moves_the_derived_challenges() {
        let tables = fixture();
        let mut prover_transcript = Transcript::new(b"batched_round4");
        let commit = commit_fixture(&tables, &mut prover_transcript, 0, 4);

        let mut widths = widths_of(&tables);
        widths[1] += 1;
        let mut verifier_transcript = Transcript::new(b"batched_round4");
        let replay = replay_batched_fri::<F, Transcript>(
            &mut verifier_transcript,
            &heights_of(&tables),
            &widths,
            &commit.layer_roots,
            &commit.final_poly_coeffs,
            BLOWUP_LOG,
            FINAL_POLY_LOG_DEGREE,
            0,
            None,
            4,
        )
        .expect("the shape is still structurally consistent");

        assert_ne!(
            replay.alpha, commit.alpha,
            "a width the prover did not commit to must move α"
        );
        assert_ne!(
            replay.iotas, commit.iotas,
            "a width the prover did not commit to must move the query indices"
        );
    }
}
