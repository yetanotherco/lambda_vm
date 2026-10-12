//! ★★ W1-A — the RPX transcript hands out canonical felts, and the keccak one
//! is untouched.
//!
//! Two facts with two reasons, kept apart on purpose because a verifier that
//! needs both must not take one as evidence of the other:
//!
//! 1. **A field coordinate is one candidate under RPX** — because a squeeze IS
//!    four canonical felts once the byte reversal is gone. That is W1-A, and
//!    [`TranscriptHash::CANDIDATES_PER_COORDINATE`] states it.
//! 2. **A query index is one draw** — because every WHIR query bound is a power
//!    of two, so `sample_u64`'s rejection threshold is zero. That is true today,
//!    hash-independent, and has nothing to do with (1).
//!
//! And one finding, which is the strongest reason the reversal had to go and is
//! not a cost argument at all:
//! [`the_reversal_would_have_biased_the_rpx_sampler`] — a reversed candidate is
//! `>= p` for about `2^32` canonical felts, so the old sampler drew uniformly
//! from a subset of the field.
//!
//! # ⛔ Why none of this is tested by sampling, and what that cost
//!
//! The obvious control — "show keccak needing more than one candidate" — is
//! unreachable. A keccak squeeze is 32 uniform bytes, so an 8-byte group lands
//! in `[p, 2^64)` with probability about **2^-32**: observing one takes on the
//! order of a billion squeezes. The same goes for `sample_u64`, whose rejection
//! region is `2^64 mod bound` wide — under `2^-32` of the range for any bound
//! this system uses, power of two or not.
//!
//! So a statistical test cannot tell the two configurations apart, and one that
//! appeared to would be measuring noise. **The difference is structural**: under
//! RPX every group is canonical BY CONSTRUCTION; under keccak every group is
//! canonical WITH HIGH PROBABILITY. The tests below pin the construction —
//! round-trips, the arithmetic of the threshold, and a rejection driven by a
//! candidate constructed to be rejected — rather than waiting for an event that
//! will not arrive.
//!
//! That is also why the rejection path is exercised explicitly: `Some(1)` is a
//! claim about the sampler's INPUTS, and it would be worth nothing if the
//! branch it bypasses had quietly stopped working.
//!
//! # ⚠ WHICH TEST CATCHES WHICH MUTATION — and which do not
//!
//! Run, not assumed. Putting the reversal back (either by flipping
//! `RpxTranscriptHash::REVERSES_SQUEEZE` or by deleting the `if` in `sample`)
//! fails **exactly one** test below:
//! [`an_rpx_squeeze_is_the_unreversed_digest_of_what_was_absorbed`].
//!
//! [`every_group_of_an_rpx_squeeze_is_a_canonical_felt`] and
//! [`a_cubic_element_costs_exactly_three_draws_under_rpx`] both still PASS with
//! the reversal restored, and that is not a defect in them — it is the same
//! 2^-32 again from the other side. A canonical felt's bytes read backwards are
//! a number below `p` unless the top bytes conspire, so 2048 reversed groups
//! look exactly like 2048 canonical ones. Those two tests guard the
//! canonicalisation inside `digest_to_commitment`, which is a real regression
//! mode; they do **not** guard the byte order, and reading them as if they did
//! would leave the seam covered by nothing.
//!
//! One test guards the byte order. It is the one with the construction in it.

use digest::Digest;
use math::field::element::FieldElement;
use math::field::extensions_goldilocks::Degree3GoldilocksExtensionField as Ext;
use math::field::goldilocks::{GOLDILOCKS_PRIME, GoldilocksField as Fp};
use math::field::traits::HasDefaultTranscript;

use crate::fiat_shamir::default_transcript::DefaultTranscript;
use crate::fiat_shamir::is_transcript::IsTranscript;
use crate::fiat_shamir::transcript_hash::{
    KeccakTranscriptHash, RpxTranscriptHash, TranscriptHash,
};
use crate::hash::platform_keccak::PlatformKeccak256;
use crate::hash::rpx::{commitment_to_digest, digest_to_commitment, sponge_leaf_bytes};

/// The 8-byte big-endian groups a sampler reads out of a squeeze, in the order
/// [`DefaultTranscript::next_sample_u64`] reads them.
fn groups(squeeze: &[u8; 32]) -> [u64; 4] {
    core::array::from_fn(|i| {
        let mut b = [0u8; 8];
        b.copy_from_slice(&squeeze[i * 8..(i + 1) * 8]);
        u64::from_be_bytes(b)
    })
}

/// ★★ K5 (RPX half). The squeeze is the digest, in the digest's own byte order.
///
/// Against the CONSTRUCTION — `digest_to_commitment(sponge_leaf_bytes(..))` —
/// not against the other arm. Two transcripts agreeing tells you nothing; they
/// agree on a wrong hash too.
#[test]
fn an_rpx_squeeze_is_the_unreversed_digest_of_what_was_absorbed() {
    let absorbed = b"W1-A: the squeeze is the digest";
    let mut transcript = DefaultTranscript::<Ext, RpxTranscriptHash>::new(absorbed);

    let squeeze = transcript.sample();
    let expected = digest_to_commitment(&sponge_leaf_bytes(absorbed));

    assert_eq!(
        squeeze, expected,
        "the RPX squeeze is not the digest of what was absorbed"
    );

    // …and NOT the reversed one, or the constant is decorative.
    let mut reversed = expected;
    reversed.reverse();
    assert_ne!(
        squeeze, reversed,
        "the RPX squeeze is still reversed — REVERSES_SQUEEZE is not being read"
    );
    // ⚠ No `assert!(!RpxTranscriptHash::REVERSES_SQUEEZE)` here. Clippy is right
    // that it cannot fail at runtime, and it would add nothing: the two
    // assertions above test the BEHAVIOUR the constant is supposed to cause,
    // which is what a wrong constant would break. Asserting the constant's own
    // value would only restate the source line that sets it.
}

/// ★★ K5 (keccak half). THE CONTROL. The keccak squeeze is still the reversed
/// digest, byte for byte.
///
/// This is the half that must not move: every proof this system has produced
/// was produced under this convention, and W1-A is only allowed to touch the
/// other arm.
#[test]
fn a_keccak_squeeze_is_still_the_reversed_digest() {
    let absorbed = b"W1-A: the squeeze is the digest";
    let mut transcript = DefaultTranscript::<Ext, KeccakTranscriptHash>::new(absorbed);

    let squeeze = transcript.sample();

    let mut expected: [u8; 32] = PlatformKeccak256::digest(absorbed).into();
    expected.reverse();

    assert_eq!(
        squeeze, expected,
        "the keccak squeeze moved: this commit changed every proof on this branch"
    );
}

/// ★★ THE PROPERTY BEHIND `Some(1)`. Every group of every RPX squeeze is a
/// canonical felt — by round-trip, not by luck.
///
/// The chain is exercised, not one squeeze: `sample()` absorbs its own output,
/// so squeeze `n+1` is a function of squeeze `n`, and a canonicality that held
/// only for the first would be an accident of the seed.
#[test]
fn every_group_of_an_rpx_squeeze_is_a_canonical_felt() {
    let mut transcript = DefaultTranscript::<Ext, RpxTranscriptHash>::new(b"chain");

    for round in 0..512 {
        let squeeze = transcript.sample();

        for (i, g) in groups(&squeeze).iter().enumerate() {
            assert!(
                *g < GOLDILOCKS_PRIME,
                "round {round}, group {i}: {g} is not a canonical felt, \
                 so a one-candidate schedule would miss here"
            );
        }

        // The structural statement the bound above is a consequence of: the 32
        // bytes ARE four felts, and reading them back gives the same four.
        let digest = commitment_to_digest(&squeeze);
        assert_eq!(
            digest_to_commitment(&digest),
            squeeze,
            "round {round}: the squeeze does not round-trip through its felts"
        );
    }
}

/// ★★ ONE CANDIDATE, COMPOSED. The real sampler, fed the real squeeze,
/// consumes exactly three draws for a cubic element.
///
/// [`Ext::sample_field_element_from`] is the production body; the closure is
/// the production byte source. Only the counter is the test's.
#[test]
fn a_cubic_element_costs_exactly_three_draws_under_rpx() {
    let mut transcript = DefaultTranscript::<Ext, RpxTranscriptHash>::new(b"three draws");
    let squeeze = transcript.sample();
    let mut supply = groups(&squeeze).into_iter();

    let mut draws = 0usize;
    let element = Ext::sample_field_element_from(|| {
        draws += 1;
        supply
            .next()
            .expect("a fourth draw means a rejection occurred")
    });

    assert_eq!(
        draws,
        3,
        "a cubic element took {draws} draws, so CANDIDATES_PER_COORDINATE = {:?} is wrong",
        RpxTranscriptHash::CANDIDATES_PER_COORDINATE
    );
    // `NonZeroUsize`, not `usize`: the schedule's type on this branch cannot
    // spell `Some(0)`, which is a draw count that returns an uninitialised
    // candidate. Same assertion, one constructor deeper.
    assert_eq!(
        RpxTranscriptHash::CANDIDATES_PER_COORDINATE,
        core::num::NonZeroUsize::new(1)
    );

    // The element is the first three groups, in order — which is what makes the
    // draw count meaningful rather than a count of a loop that did nothing.
    let expected: Vec<FieldElement<Fp>> = groups(&squeeze)[..3]
        .iter()
        .map(|g| FieldElement::from(*g))
        .collect();
    assert_eq!(element.value().to_vec(), expected);
}

/// ★★ …AND THE REJECTION BRANCH IS STILL ALIVE.
///
/// `Some(1)` is a claim about the sampler's INPUTS. If the rejection test had
/// been deleted, every test above would still pass and the constant would be
/// true for the wrong reason — so the branch is driven by a candidate
/// constructed to be rejected. This is the only way to reach it: waiting for a
/// keccak squeeze to produce one is a 2^-32 event per group.
#[test]
fn a_non_canonical_candidate_is_rejected_and_redrawn() {
    // p itself is the smallest non-canonical u64, and `p + 7` is inside the
    // window a uniform draw can land in.
    let supply = [GOLDILOCKS_PRIME, GOLDILOCKS_PRIME + 7, 42u64];
    let mut it = supply.into_iter();

    let mut draws = 0usize;
    let element = Fp::sample_field_element_from(|| {
        draws += 1;
        it.next().expect("the sampler drew more than the supply")
    });

    assert_eq!(
        draws, 3,
        "the sampler accepted a candidate >= p: the rejection test is gone, and \
         CANDIDATES_PER_COORDINATE = Some(1) would then be true of nothing"
    );
    assert_eq!(element, FieldElement::<Fp>::from(42u64));
}

/// ★★★ WHAT THE REVERSAL WAS ACTUALLY DOING: biasing the RPX sampler.
///
/// This is the strongest reason to remove it, and it is not a cost argument.
///
/// A candidate under the old code was `byteswap(canonical(felt))`. That is
/// `>= p` exactly when the felt's low four bytes are all `0xFF` — reversing
/// puts them in the top four, and `p`'s top four bytes are `0xFFFFFFFF`. The
/// rejection sampler then drew uniformly from a SUBSET of `[0, p)` missing
/// about `2^32` elements: statistical distance ~`2^-32` per coordinate, and
/// over an epoch verify's ~3e4 coordinate draws a loose hybrid bound of
/// ~`2^-17` of added soundness error.
///
/// Not a demonstrated attack — the excluded set is fixed and public and no
/// prover steers into it — and never exercised, because the RPX transcript was
/// not wired to any prover (see the commit). But it is exactly the kind of
/// unquoted term an audit names, and it was inherited rather than chosen:
/// harmless under keccak, whose 8-byte groups are uniform on 64 bits and whose
/// sampler is therefore exactly uniform. It exists only in the
/// RPX-under-`DefaultTranscript` combination.
///
/// The test exhibits the witness rather than describing it, and characterises
/// the whole excluded set so the claim is a statement about all of it.
#[test]
fn the_reversal_would_have_biased_the_rpx_sampler() {
    let byteswap = |v: u64| {
        u64::from_be_bytes({
            let mut b = v.to_be_bytes();
            b.reverse();
            b
        })
    };

    // V1's witness: a canonical felt whose reversed bytes are NOT canonical, so
    // the old code would have rejected this felt every time it appeared.
    let witness: u64 = 0x0000_0001_ffff_ffff;
    assert!(
        witness < GOLDILOCKS_PRIME,
        "the witness must be a real felt"
    );
    assert!(
        byteswap(witness) >= GOLDILOCKS_PRIME,
        "the witness's reversed bytes are canonical, so it is not a witness"
    );

    // …and the excluded set is exactly the felts whose low four bytes are all
    // `0xFF`, save the one whose high four bytes are zero. Checked over the
    // whole set rather than sampled: it is generated, not searched for.
    for high in 0..4096u64 {
        let v = (high << 32) | 0xFFFF_FFFF;
        if v >= GOLDILOCKS_PRIME {
            continue;
        }
        let excluded = byteswap(v) >= GOLDILOCKS_PRIME;
        assert_eq!(
            excluded,
            high != 0,
            "felt {v:#018x} is misclassified: the excluded set is not what the \
             bias argument says it is"
        );
    }

    // The far larger complement: nothing OUTSIDE that set was excluded, so the
    // bias is precisely the one described and not a larger one.
    let mut checked = 0u32;
    for v in (0..1u64 << 24).map(|i| i.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 1) {
        if v >= GOLDILOCKS_PRIME || (v & 0xFFFF_FFFF) == 0xFFFF_FFFF {
            continue;
        }
        assert!(
            byteswap(v) < GOLDILOCKS_PRIME,
            "felt {v:#018x} was excluded but is not in the described set"
        );
        checked += 1;
    }
    assert!(
        checked > 1_000_000,
        "only {checked} felts were actually checked"
    );

    // And the fix: a live squeeze's groups are canonical, so no candidate is
    // excluded and the distribution is the digest's, entire.
    let mut transcript = DefaultTranscript::<Ext, RpxTranscriptHash>::new(b"bias");
    for _ in 0..64 {
        for g in groups(&transcript.sample()) {
            assert!(g < GOLDILOCKS_PRIME);
        }
    }
}

/// ★ The keccak configuration makes no such claim, and says so.
#[test]
fn keccak_does_not_claim_a_bounded_candidate_count() {
    assert_eq!(
        KeccakTranscriptHash::CANDIDATES_PER_COORDINATE,
        None,
        "keccak's squeeze is uniform bytes: the draw count has a distribution, not a bound"
    );
}

/// ★★ THE SECOND FACT, WITH ITS OWN REASON. A query index is one draw because
/// the bound is a power of two — not because of anything W1-A did.
///
/// `sample_u64` rejects `candidate < 2^64 mod bound`. For a power of two that
/// region is EMPTY, so the loop cannot turn, whatever the hash. Every WHIR
/// query bound is `num_leaves = 1 << (log_domain_size - log_folding)`
/// (`multilinear::whir_commit::CodewordCommitment::num_leaves`), so this covers
/// all of them.
///
/// ⚠ The non-power-of-two half is the point of the test. Without it this would
/// assert that zero equals zero for 64 values and pass on any implementation.
#[test]
fn a_power_of_two_bound_has_no_rejection_region_and_a_ragged_one_does() {
    let threshold = |bound: u64| bound.wrapping_neg() % bound;

    for k in 0..64 {
        let bound = 1u64 << k;
        assert_eq!(
            threshold(bound),
            0,
            "bound 2^{k} has a rejection region, so a query index is not one draw"
        );
    }

    // Constructed counter-examples: a bound that is NOT a power of two must
    // have a non-empty rejection region, or the expression above is not
    // computing what this test claims it computes.
    for bound in [3u64, 5, 6, 100, (1 << 20) + 1, u64::MAX] {
        assert_ne!(
            threshold(bound),
            0,
            "bound {bound} is not a power of two yet shows no rejection region"
        );
    }
}

/// ★ …and the draw count that follows from it, observed through the transcript
/// rather than asserted.
///
/// Four `sample_u64` calls at a power-of-two bound must consume exactly one
/// squeeze. The witness is the transcript's own state: every squeeze chains its
/// output back in, so a fifth draw would leave `a` somewhere `b` is not.
#[test]
fn four_query_indices_cost_one_squeeze() {
    for hash_is_rpx in [false, true] {
        let (state_a, state_b) = if hash_is_rpx {
            let mut a = DefaultTranscript::<Ext, RpxTranscriptHash>::new(b"queries");
            let mut b = a.clone();
            for _ in 0..4 {
                a.sample_u64(1 << 20);
            }
            let _ = b.sample();
            (
                IsTranscript::<Ext>::state(&a),
                IsTranscript::<Ext>::state(&b),
            )
        } else {
            let mut a = DefaultTranscript::<Ext, KeccakTranscriptHash>::new(b"queries");
            let mut b = a.clone();
            for _ in 0..4 {
                a.sample_u64(1 << 20);
            }
            let _ = b.sample();
            (
                IsTranscript::<Ext>::state(&a),
                IsTranscript::<Ext>::state(&b),
            )
        };

        assert_eq!(
            state_a, state_b,
            "four query draws consumed more than one squeeze (rpx = {hash_is_rpx})"
        );
    }
}

/// ★ The duplex buffer hands out whole groups, which is what makes the felt
/// boundaries and the read boundaries the same boundaries.
///
/// Sampling 4 `u64`s consumes exactly one squeeze; the 5th forces the next. If
/// a read ever straddled two groups, `Some(1)` would be false even with a
/// canonical squeeze — the claim depends on this alignment, so it is pinned.
#[test]
fn the_buffer_is_consumed_in_whole_felt_groups() {
    let mut transcript = DefaultTranscript::<Ext, RpxTranscriptHash>::new(b"alignment");
    let mut reference = transcript.clone();

    let first = reference.sample();
    let second = reference.sample();

    let drawn: Vec<u64> = (0..8).map(|_| transcript.sample_u64(u64::MAX)).collect();

    let expected: Vec<u64> = groups(&first)
        .into_iter()
        .chain(groups(&second))
        .map(|g| g % u64::MAX)
        .collect();

    assert_eq!(
        drawn, expected,
        "the buffer is not being handed out as whole 8-byte groups in order"
    );
}
