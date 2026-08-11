//! The `"LFML"` leaf mode (option C): its vectors, its canonicity gate, and the
//! milestone it unblocks — `FriToyV0` proving under BLAKE3.
//!
//! ## What pins what
//!
//! 1. **The felt boundary** — halves and canonicity — against
//!    [`super::leaf_kats`]'s boundary table, `p − 1` and the non-canonical
//!    aliases included.
//! 2. **The step function** against the `blake3` crate: at 7 rounds a leaf is
//!    `blake3::hash(LE32(lanes) ‖ "LFML")` truncated, so the leaf domain
//!    inherits the socket's external anchor rather than claiming a new one.
//! 3. **The chip**: M9 (mode confusion, six ordered pairs) and M10 (a `MODE_L`
//!    row that skips canonicity), which are what make "`MODE_L` implies
//!    felt-input semantics" a constraint rather than a convention.
//! 4. **The assembled program**: `FriToyV0` proves and verifies under BLAKE3 —
//!    the F3.4 milestone — with a NEGATIVE leg showing the canonicity gate does
//!    work in the real proof and not only in a unit test.
//!
//! Every rejection test is paired with an honest-path assertion.

use math::field::traits::IsPrimeField;
use stark::proof::options::{GoldilocksCubicProofOptions, ProofOptions};

use crate::tables::types::{FE, GoldilocksField};

use super::blake3_socket::{
    self, FELTS_PER_LEAF, SOCKET_ROUNDS, TAG_LFMC, TAG_LFML, TAG_LFMT, cols, felt_halves,
    is_canonical, leaf_digest_rounds, leaf_lanes, socket_digest_rounds_tagged, word_of,
};
use super::hash::HasherKind;
use super::instr::{HashMode, Instr};
use super::leaf_kats::{BOUNDARY_FELTS, FRI_LEAF, LEAF_VECTORS, NON_CANONICAL};
use super::proof::{lfm_prove_with_hasher, verify_against};
use super::registry::build_artifacts_with_hasher;
use super::word::LfmWord;

const KIND: HasherKind = HasherKind::Blake3;

/// Goldilocks `p`.
const P: u64 = 0xFFFF_FFFF_0000_0001;

fn options() -> ProofOptions {
    GoldilocksCubicProofOptions::with_blowup(2).expect("blowup=2 is valid")
}

fn felts_of(vals: &[u64; 4]) -> LfmWord {
    core::array::from_fn(|i| FE::from(vals[i]))
}

// =========================================================================
// L2/L3/L4 — the felt boundary
// =========================================================================

/// Every boundary felt splits into the halves the spec pins, `p − 1` included —
/// the tight case, where `hi` is maximal and `lo` is exactly zero.
#[test]
fn every_boundary_felt_round_trips_through_its_halves() {
    for v in BOUNDARY_FELTS.iter() {
        let (lo, hi) = felt_halves(v.felt).unwrap_or_else(|| panic!("{} is canonical", v.name));
        assert_eq!((lo, hi), (v.lo, v.hi), "halves of {}", v.name);
        assert!(is_canonical(lo, hi), "{} must pass the predicate", v.name);
        assert_eq!(
            u64::from(lo) + (u64::from(hi) << 32),
            v.felt,
            "halves must recompose {}",
            v.name
        );
    }
}

/// ★ Non-canonical values are REJECTED, never reduced — and the test says why
/// it matters: each one ALIASES a canonical felt, so reducing would give one
/// field element two leaf digests.
#[test]
fn non_canonical_values_are_rejected_not_reduced() {
    for v in NON_CANONICAL.iter() {
        assert!(!is_canonical(v.lo, v.hi), "{} is non-canonical", v.name);
        assert_eq!(felt_halves(v.value), None, "{} must be refused", v.name);

        // The alias, spelled out: hi maximal makes 2^32·hi = p − 1 ≡ −1, so the
        // pair encodes `lo − 1`, which has its own ordinary encoding. THAT is
        // the collision the canonicity block prevents.
        let aliased = (u128::from(v.lo) + (u128::from(v.hi) << 32)) % u128::from(P);
        assert_eq!(aliased, u128::from(v.lo) - 1);
        let (clo, chi) = felt_halves(aliased as u64).expect("the alias target is canonical");
        assert_ne!(
            (clo, chi),
            (v.lo, v.hi),
            "{}: two half-pairs for one felt is exactly the hazard",
            v.name
        );
    }
}

/// L4 — the predicate IS `v < p`, over every boundary and a dense sweep.
///
/// The spec ran 300,007 cases; this runs the same boundaries plus a sweep near
/// the wrap, which is where a predicate that is merely *nearly* right fails.
#[test]
fn the_canonicity_predicate_is_exactly_less_than_p() {
    let check = |v: u64| {
        let (lo, hi) = (v as u32, (v >> 32) as u32);
        assert_eq!(
            is_canonical(lo, hi),
            v < P,
            "predicate disagrees with v < p at {v:#x}"
        );
    };
    for v in [
        0u64,
        1,
        u64::from(u32::MAX),
        1 << 32,
        P - 2,
        P - 1,
        P,
        P + 1,
        u64::MAX,
    ] {
        check(v);
    }
    // The whole neighbourhood of the wrap, both sides.
    for d in 0..2_000u64 {
        check(P.wrapping_sub(d));
        check(P.wrapping_add(d));
    }
    // A stride across the space, so the sweep is not only local.
    for k in 0..20_000u64 {
        check(k.wrapping_mul(0x9E37_79B9_7F4A_7C15));
    }
}

// =========================================================================
// L1 — the leaf digest, and its crate anchor
// =========================================================================

/// Every leaf vector reproduces at BOTH round counts, lanes included.
#[test]
fn every_leaf_vector_reproduces_at_both_round_counts() {
    for v in LEAF_VECTORS.iter() {
        let felts = felts_of(&v.felts);
        assert_eq!(
            leaf_lanes(&felts).expect("canonical"),
            v.lanes,
            "lanes of {}",
            v.name
        );
        assert_eq!(
            leaf_digest_rounds(&felts, 6).expect("canonical"),
            v.digest_6,
            "6-round leaf {}",
            v.name
        );
        assert_eq!(
            leaf_digest_rounds(&felts, 7).expect("canonical"),
            v.digest_7,
            "7-round leaf {}",
            v.name
        );
    }
}

/// ★ **The external anchor, direct.** At 7 rounds a leaf is literally
/// `blake3::hash(LE32(lo0)‖LE32(hi0)‖…‖LE32(hi3)‖"LFML")` truncated to 16 bytes.
///
/// The message is rebuilt from the byte-level specification rather than from
/// `socket_message`, so the word-level and byte-level forms can disagree. This
/// is the property option C was chosen to preserve: putting the felt encoding
/// INSIDE the socket keeps the message layout byte-identical to a digest-mode
/// compress, so the crate stays a direct KAT for the leaf domain too.
#[test]
fn seven_rounds_is_blake3_of_the_leaf_message() {
    for v in LEAF_VECTORS.iter() {
        let mut msg = Vec::with_capacity(36);
        for lane in v.lanes.iter() {
            msg.extend_from_slice(&lane.to_le_bytes());
        }
        msg.extend_from_slice(b"LFML");
        assert_eq!(msg.len(), 36, "a leaf row is one 36-byte block");

        let full = blake3::hash(&msg);
        let want: [u32; 4] = core::array::from_fn(|i| {
            u32::from_le_bytes(full.as_bytes()[4 * i..4 * i + 4].try_into().unwrap())
        });
        assert_eq!(want, v.digest_7, "leaf {} must be blake3::hash", v.name);
    }
}

/// The tag word is the ASCII, little-endian, and the three live domains are
/// pairwise distinct as VALUES — the cheap check that a typo cannot pass.
#[test]
fn the_leaf_tag_is_lfml_and_the_three_domains_are_distinct() {
    assert_eq!(TAG_LFML, u32::from_le_bytes(*b"LFML"));
    assert_eq!(TAG_LFML.to_le_bytes(), *b"LFML");
    assert_ne!(TAG_LFML, TAG_LFMC);
    assert_ne!(TAG_LFML, TAG_LFMT);
    assert_ne!(TAG_LFMC, TAG_LFMT);
}

/// L5 — the three domains produce three different digests from the SAME eight
/// lanes. Distinct tag values are necessary; distinct digests are the property.
#[test]
fn the_three_domains_differ_on_the_same_lanes() {
    for v in LEAF_VECTORS.iter() {
        let (a, b) = (
            [v.lanes[0], v.lanes[1], v.lanes[2], v.lanes[3]],
            [v.lanes[4], v.lanes[5], v.lanes[6], v.lanes[7]],
        );
        for rounds in [6, 7] {
            let d: Vec<[u32; 4]> = [TAG_LFMC, TAG_LFMT, TAG_LFML]
                .iter()
                .map(|t| socket_digest_rounds_tagged(&a, &b, rounds, *t))
                .collect();
            assert_ne!(d[0], d[1], "{} @{rounds}: LFMC == LFMT", v.name);
            assert_ne!(d[0], d[2], "{} @{rounds}: LFMC == LFML", v.name);
            assert_ne!(d[1], d[2], "{} @{rounds}: LFMT == LFML", v.name);
        }
    }
}

/// L6 — an eight-felt leaf is exactly three compressions: two `LFML` rows and
/// one ordinary `LFMC` parent, in that association.
#[test]
fn an_eight_felt_leaf_is_two_leaf_rows_and_one_parent() {
    let lo: LfmWord = core::array::from_fn(|i| FE::from(FRI_LEAF.felts[i]));
    let hi: LfmWord = core::array::from_fn(|i| FE::from(FRI_LEAF.felts[4 + i]));
    for (rounds, want) in [(6, FRI_LEAF.digest_6), (7, FRI_LEAF.digest_7)] {
        let d0 = leaf_digest_rounds(&lo, rounds).expect("canonical");
        let d1 = leaf_digest_rounds(&hi, rounds).expect("canonical");
        let parent = socket_digest_rounds_tagged(&d0, &d1, rounds, TAG_LFMC);
        assert_eq!(parent, want, "the 8-felt leaf at {rounds} rounds");
    }
    assert_eq!(FRI_LEAF.compresses, 3);

    // The HOST path agrees with the reference — `host_leaf_hash_pair` is what
    // the fixture builds its trees with, so a divergence here is a fixture that
    // the machine cannot authenticate.
    let host = super::fixture::host_leaf_hash_pair(KIND, &lo, &hi);
    let want = if SOCKET_ROUNDS == 7 {
        FRI_LEAF.digest_7
    } else {
        FRI_LEAF.digest_6
    };
    assert_eq!(host, word_of(&want));
}

// =========================================================================
// M9 / M10 — the chip-level controls the leaf spec pre-committed
// =========================================================================

/// A hash row in `mode` over `felts`/`lanes`, exactly as the trace filler builds
/// one — for the controls, which need to force a mismatch the filler cannot.
fn leaf_row(felts: &LfmWord) -> Vec<FE> {
    let mut row = vec![FE::zero(); cols::NUM_COLUMNS];
    row[cols::MODE_L] = FE::one();
    row[cols::IN0..cols::IN0 + FELTS_PER_LEAF].copy_from_slice(felts);
    for (k, iv) in super::blake3::BLAKE3_IV.iter().take(4).enumerate() {
        row[cols::S8 + k] = FE::from(u64::from(*iv));
    }
    let digest = leaf_digest_rounds(felts, SOCKET_ROUNDS).expect("canonical");
    row[cols::OUT0..cols::OUT0 + 4].copy_from_slice(&word_of(&digest));
    blake3_socket::fill_socket_witness(&mut row);
    row
}

/// **M9 — mode confusion, all six ordered pairs.** A row in one domain whose
/// witness computes another domain's digest must be rejected.
///
/// Three tags means six ordered confusions, and the leaf domain adds four of
/// them. `L5` above shows the three functions differ; this shows the CHIP
/// notices, which is a different claim.
#[test]
fn m9_no_domain_can_compute_another_domains_digest() {
    let felts = felts_of(&LEAF_VECTORS[3].felts);
    let lanes = leaf_lanes(&felts).expect("canonical");
    let (a, b) = (
        [lanes[0], lanes[1], lanes[2], lanes[3]],
        [lanes[4], lanes[5], lanes[6], lanes[7]],
    );

    for (mode, own) in [
        (HashMode::Compress, TAG_LFMC),
        (HashMode::Transcript, TAG_LFMT),
        (HashMode::Leaf, TAG_LFML),
    ] {
        for other in [TAG_LFMC, TAG_LFMT, TAG_LFML] {
            let mut row = vec![FE::zero(); cols::NUM_COLUMNS];
            row[super::blake3_socket_tests::mode_col(mode)] = FE::one();
            if mode == HashMode::Leaf {
                row[cols::IN0..cols::IN0 + 4].copy_from_slice(&felts);
            } else {
                row[cols::IN0..cols::IN0 + 4].copy_from_slice(&word_of(&a));
                row[cols::IN0 + 4..cols::IN0 + 8].copy_from_slice(&word_of(&b));
            }
            for (k, iv) in super::blake3::BLAKE3_IV.iter().take(4).enumerate() {
                row[cols::S8 + k] = FE::from(u64::from(*iv));
            }
            let digest = socket_digest_rounds_tagged(&a, &b, SOCKET_ROUNDS, other);
            row[cols::OUT0..cols::OUT0 + 4].copy_from_slice(&word_of(&digest));
            blake3_socket::fill_socket_witness_tagged(&mut row, other);

            let violated = super::blake3_socket_tests::violations(&row);
            if other == own {
                assert_eq!(
                    violated,
                    Vec::<usize>::new(),
                    "HONEST CONTROL: {mode:?} in its own domain must be accepted"
                );
            } else {
                assert!(
                    !violated.is_empty(),
                    "{mode:?} computing the {other:#010x} domain must be rejected"
                );
            }
        }
    }
}

/// **M10 — `MODE_L` implies felt-input semantics, as a CONSTRAINT.**
///
/// A leaf row that skips the canonicity block must be rejected. Two ways to
/// skip it, and both are tried: zero the witnesses, and install the
/// non-canonical alias with a witness that would satisfy every constraint the
/// canonicity block does not impose.
#[test]
fn m10_a_leaf_row_cannot_skip_canonicity() {
    let felts = felts_of(&LEAF_VECTORS[1].felts);
    let base = leaf_row(&felts);
    assert_eq!(
        super::blake3_socket_tests::violations(&base),
        Vec::<usize>::new(),
        "HONEST CONTROL: a canonical leaf row satisfies every constraint"
    );

    // (a) blank the canonicity witnesses. `canon_b` pins `Z` from `G`, so a
    // zeroed `Z` is only satisfiable when `G` is invertible AND `GINV` matches;
    // blanking both breaks it.
    let mut blanked = base.clone();
    for i in 0..FELTS_PER_LEAF {
        blanked[cols::canon_z(i)] = FE::zero();
        blanked[cols::canon_ginv(i)] = FE::zero();
    }
    assert!(
        !super::blake3_socket_tests::violations(&blanked).is_empty(),
        "a leaf row with no canonicity witness must be rejected"
    );

    // (b) ★ THE ALIAS. Re-encode felt 0 as (lo + 1, hi = 2^32 − 1), which is the
    // SAME field element — the binding constraint is satisfied — and let the
    // witness be otherwise consistent. Only canonicity can catch this, and the
    // test asserts it is `canon-c` that does.
    let target = felts_of(&[0, 0, 0, 0]);
    let mut alias = leaf_row(&target);
    // lane 0 = lo0 becomes 1, lane 1 = hi0 becomes 2^32 − 1.
    for (lane, v) in [(0usize, 1u32), (1, u32::MAX)] {
        for byte in 0..4 {
            alias[cols::lane_byte(lane, byte)] = FE::from(u64::from((v >> (8 * byte)) as u8));
        }
    }
    // The witness the alias would need: hi is maximal, so G = 0 and Z = 1.
    alias[cols::canon_z(0)] = FE::one();
    alias[cols::canon_ginv(0)] = FE::zero();

    let violated = super::blake3_socket_tests::violations(&alias);
    // canon-c for felt 0 — located from the arm's own indices rather than by a
    // literal, so growing the framing cannot silently point this at another
    // constraint.
    const CANON_C_FELT0: usize =
        blake3_socket::LEAF_IDX + blake3_socket::LEAF_CONSTRAINTS_PER_FELT - 1;
    assert!(
        violated.contains(&CANON_C_FELT0),
        "the alias must be caught by canon-c (idx {CANON_C_FELT0}), got {violated:?}"
    );

    // And the alias really is the same field element, so nothing ELSE could
    // have caught it — that is what makes canonicity load-bearing rather than
    // redundant with the binding constraint.
    assert_eq!(
        (u128::from(1u32) + (u128::from(u32::MAX) << 32)) % u128::from(P),
        0,
        "the alias encodes felt 0"
    );
}

// =========================================================================
// ★ The F3.4 milestone: FriToyV0 under BLAKE3
// =========================================================================

fn fri_arenas(inner: &super::fixture::FriToyProof) -> Vec<Vec<LfmWord>> {
    vec![inner.commitments.clone(), inner.openings.clone()]
}

/// ★★ **`FriToyV0` proves and verifies under BLAKE3.** The milestone the whole
/// campaign was for: a real verification program, over real FRI data — LDE
/// evaluations and folded extension elements, none of them `u32` — proved under
/// the machine's real hash.
///
/// This replaces `blake3_socket_tests::fri_toy_is_still_blocked_by_o1…`, whose
/// own doc required a prove+verify rather than an execute when O1 closed.
#[test]
fn fri_toy_proves_and_verifies_under_blake3() {
    let opts = options();
    let program = super::programs::fri_toy_program();
    let inner = super::fixture::fixture_prove_with_hasher(KIND);
    let artifacts = build_artifacts_with_hasher(&program, &opts, KIND);
    let proved = lfm_prove_with_hasher(&program, &artifacts, &fri_arenas(&inner), &opts, KIND)
        .expect("FriToyV0 must prove under BLAKE3");
    assert!(
        verify_against(
            &artifacts.roots,
            &artifacts.program_id,
            artifacts.keccak_rnd_chunks,
            &proved.proof,
            &proved.public_words,
            &opts,
            artifacts.hasher,
        ),
        "an honest BLAKE3 proof of FriToyV0 must verify"
    );
    // The attested output is the inner proof's identity: both roots.
    assert_eq!(proved.public_words[0].1, inner.commitments[0]);
    assert_eq!(proved.public_words[1].1, inner.commitments[1]);
}

/// The same program under the other two hashers — B1 and option C both changed
/// shared constructions, so all three must stay green.
#[test]
fn fri_toy_proves_and_verifies_under_every_hasher() {
    let opts = options();
    let program = super::programs::fri_toy_program();
    for kind in [HasherKind::Test, HasherKind::Poseidon, HasherKind::Blake3] {
        let inner = super::fixture::fixture_prove_with_hasher(kind);
        let artifacts = build_artifacts_with_hasher(&program, &opts, kind);
        let proved = lfm_prove_with_hasher(&program, &artifacts, &fri_arenas(&inner), &opts, kind)
            .unwrap_or_else(|e| panic!("prove under {kind:?}: {e:?}"));
        assert!(
            verify_against(
                &artifacts.roots,
                &artifacts.program_id,
                artifacts.keccak_rnd_chunks,
                &proved.proof,
                &proved.public_words,
                &opts,
                artifacts.hasher,
            ),
            "an honest proof of FriToyV0 must verify under {kind:?}"
        );
    }
}

/// ⚠ **The NEGATIVE leg — the criterion most likely to be skipped.**
///
/// A fixture whose tree was built under a DIFFERENT hasher must not
/// authenticate. This is what shows the leaf digests are load-bearing in the
/// assembled program: every opened row is authenticated by re-deriving its leaf,
/// so a leaf computed by another hash breaks the walk.
#[test]
fn fri_toy_rejects_a_fixture_built_under_another_hasher() {
    let opts = options();
    let program = super::programs::fri_toy_program();
    let artifacts = build_artifacts_with_hasher(&program, &opts, KIND);
    let mismatched = super::fixture::fixture_prove_with_hasher(HasherKind::Test);
    assert!(
        lfm_prove_with_hasher(&program, &artifacts, &fri_arenas(&mismatched), &opts, KIND).is_err(),
        "a Test-hashed fixture must not authenticate under BLAKE3"
    );

    // HONEST CONTROL: the matching fixture does prove, so the rejection is about
    // the hasher and not about the program.
    let matching = super::fixture::fixture_prove_with_hasher(KIND);
    assert!(
        lfm_prove_with_hasher(&program, &artifacts, &fri_arenas(&matching), &opts, KIND).is_ok()
    );
}

/// The leaf mode is what closed O1 for this program: the fixture's committed
/// values are still not `u32`-laned, and it proves anyway.
///
/// The old tripwire asserted the opposite conclusion from the same premise. It
/// is kept as a positive statement because the premise is what makes the
/// milestone meaningful — proving over `u32`-shaped data would have proved
/// nothing about the leaf mode.
#[test]
fn the_fixture_data_is_still_not_u32_and_that_is_the_point() {
    let over = super::fixture::fixture_columns()
        .iter()
        .flatten()
        .filter(|v| GoldilocksField::canonical(v.value()) >= 1u64 << 32)
        .count();
    assert!(
        over > 0,
        "if the fixture became u32-laned the milestone would be vacuous"
    );

    // Every leaf row in the emitted program is a `Leaf`, and every Merkle-walk
    // step is a `Compress` — the split the O5 retirement rests on.
    let program = super::programs::fri_toy_program();
    let leaves = program
        .instrs
        .iter()
        .filter(|i| matches!(i, Instr::Hash { mode, .. } if *mode == HashMode::Leaf))
        .count();
    assert_eq!(
        leaves, 26,
        "4 queries × 3 data leaves × 2 LFML rows, plus the two terminal \
         coefficients the transcript absorbs as data"
    );
}

// =========================================================================
// D1 — the unread input cells, on EVERY arm
// =========================================================================

/// Evaluates a hash row against `kind`'s constraint set and returns the
/// violated indices.
fn violations_under(kind: HasherKind, row: &[FE]) -> Vec<usize> {
    use math::field::element::FieldElement;
    use stark::constraints::builder::{ConstraintSet, ProverEvalFolder};
    use stark::frame::Frame;
    use stark::table::TableView;
    use stark::traits::TransitionEvaluationContext;

    use crate::tables::types::{GoldilocksExtension, GoldilocksField};

    let set = super::chips::hash::HashConstraints { kind };
    let n = ConstraintSet::<GoldilocksField, GoldilocksExtension>::meta(&set).len();
    let no_ch: Vec<FieldElement<GoldilocksExtension>> = vec![];
    let offset = FieldElement::<GoldilocksExtension>::zero();
    let frame = Frame::<GoldilocksField, GoldilocksExtension>::new(vec![TableView::new(
        vec![row.to_vec()],
        vec![vec![]],
    )]);
    let ctx =
        TransitionEvaluationContext::new_prover(frame.as_row_frame(), &no_ch, &no_ch, &offset);
    let mut base_out = vec![FE::zero(); n];
    let mut ext_out = vec![FieldElement::<GoldilocksExtension>::zero(); n];
    let mut folder = ProverEvalFolder::new(&ctx, &mut base_out, &mut ext_out);
    set.eval(&mut folder);
    folder.assert_all_emitted();
    base_out
        .iter()
        .enumerate()
        .filter(|(_, v)| **v != FE::zero())
        .map(|(i, _)| i)
        .collect()
}

/// A `MODE_L` row for `kind` whose SECOND input cell carries `extra`.
///
/// Everything downstream is derived from the two cells, by each arm's own rule,
/// so the row is internally consistent whatever `extra` is: `extra = 0` is the
/// honest row a trace filler would write, and any other `extra` is the forgery a
/// prover controlling the whole trace would actually submit. Building both the
/// same way is what makes "only the pins fire" a meaningful assertion — a
/// half-built forgery would trip the round constraints instead and prove
/// nothing about the pins.
fn leaf_row_with_second_cell(kind: HasherKind, felts: &LfmWord, extra: &LfmWord) -> Vec<FE> {
    use super::hash::{HASH_STATE_FELTS, LfmHasher};

    let mut row = vec![FE::zero(); super::chips::hash::num_columns(kind)];
    row[cols::MODE_L] = FE::one();
    row[cols::IN0..cols::IN0 + 4].copy_from_slice(felts);
    row[cols::IN0 + 4..cols::IN0 + 8].copy_from_slice(extra);
    let iv = kind.compress_iv();
    row[cols::S8..cols::S8 + iv.len()].copy_from_slice(&iv);

    match kind {
        // BLAKE3 reads four felts and nothing else, so its output does not
        // depend on the second cell at all — which is exactly why only a pin
        // can notice junk there.
        HasherKind::Blake3 => {
            let out = kind.leaf_out(felts);
            row[cols::OUT0..cols::OUT0 + out.len()].copy_from_slice(&out);
            blake3_socket::fill_socket_witness(&mut row);
        }
        // The field-native arms permute the whole state, second cell included.
        HasherKind::Test => {
            let mut state = [FE::zero(); HASH_STATE_FELTS];
            state[0..4].copy_from_slice(felts);
            state[4..8].copy_from_slice(extra);
            state[8..12].copy_from_slice(&iv);
            let out = kind.permute(state);
            row[cols::OUT0..cols::OUT0 + out.len()].copy_from_slice(&out);
        }
        HasherKind::Poseidon => {
            // The filler reads `IN`/`S` back out of the row and writes every
            // round intermediate AND `OUT`, so the whole witness follows the
            // junk rather than only the final output.
            super::trace::fill_poseidon_witness(&mut row);
        }
    }
    row
}

/// ★★ **D1 — a leaf row's UNREAD input cells are pinned on every arm.**
///
/// `MODE_L` reads one cell. The other two receive nothing from `LfmMem`, so
/// unless a constraint pins them they are free — and `Test`'s and `Poseidon`'s
/// round 0 reads `A_i = IN_i` for `i < 8`, so on those arms the four free felts
/// were consumed by the permutation the AIR proves. `leaf(c)` stopped being a
/// function of `c`, which is a Fiat–Shamir break for any program that absorbs
/// data through `absorb_felts`.
///
/// It shipped that way and an adversarial review executed it: Poseidon proved
/// AND verified with attacker junk in those columns. This is the regression
/// test, and it runs on all three arms because the defect was that one arm had
/// the pin and two did not.
///
/// **Shaped like WA9**: it does not merely show the junk row is rejected, it
/// shows the pins are what rejects it — the violated set is exactly those four
/// constraints, so a set without them accepts the row. Necessary, not just
/// present.
#[test]
fn d1_the_unread_input_pins_are_load_bearing_under_every_hasher() {
    let felts = felts_of(&LEAF_VECTORS[3].felts);
    let zero: LfmWord = [FE::zero(); 4];

    for kind in [HasherKind::Test, HasherKind::Poseidon, HasherKind::Blake3] {
        // HONEST CONTROL FIRST: the pin must not reject honest rows. It cannot —
        // every arm's `leaf_out` leaves the unread cells zero — but a fix that
        // rejected everything would pass the negative leg on its own.
        let honest = leaf_row_with_second_cell(kind, &felts, &zero);
        assert_eq!(
            violations_under(kind, &honest),
            Vec::<usize>::new(),
            "{kind:?}: an honest leaf row must still satisfy every constraint"
        );

        // ★ The forgery, built the way an attacker would: junk in the cell the
        // mode does not read, and the rest of the row made CONSISTENT with it —
        // a prover controls the whole trace, so they would never leave a
        // detectable inconsistency behind. That is what makes the assertion
        // below exact: with the row otherwise honest, the pins are the only
        // constraints that can fire, so `== 4` says the PIN caught it rather
        // than something downstream noticing the junk by accident.
        let junk: LfmWord = core::array::from_fn(|j| FE::from(0x9E37_79B9_u64 + j as u64));
        let forged = leaf_row_with_second_cell(kind, &felts, &junk);

        // The forgery really is a different hash — otherwise the pin would be
        // guarding nothing on this arm.
        if kind != HasherKind::Blake3 {
            assert_ne!(
                forged[cols::OUT0],
                honest[cols::OUT0],
                "{kind:?}: the junk must actually move the digest"
            );
        }

        // ★ THE WA9 SHAPE. Not "the row is rejected" — that would pass for a
        // constraint set that rejected it for some incidental reason, and would
        // say nothing about whether the pins are needed. What is asserted is
        // that the violated set IS EXACTLY the pins for the cell that was
        // forged, which carries both legs at once:
        //
        //   - WITH the pins, the row is rejected;
        //   - WITHOUT them — delete those four constraints and every other
        //     constraint in the set still evaluates to zero on this row — it is
        //     ACCEPTED. That is the dropped-leg, and it is what makes the pins
        //     load-bearing rather than merely present.
        //
        // On `Test` and `Poseidon` that acceptance was the shipped behaviour and
        // an executed Fiat–Shamir break; on BLAKE3 the row is inert either way,
        // which is why the same assertion means "hygiene" there and "soundness"
        // on the two arms whose round 0 reads `IN4..8`.
        let base = super::chips::hash::unread_input_pin_base(kind);
        let expected: Vec<usize> = (base..base + 4).collect();
        let violated = violations_under(kind, &forged);
        assert_eq!(
            violated, expected,
            "{kind:?}: the violated set must be EXACTLY the four pins for the \
             forged cell — anything else and the dropped-leg claim does not hold"
        );
    }
}

/// The pins are derived from `HashMode::num_input_cells`, not written per arm —
/// so a mode added later cannot acquire free columns by an arm forgetting it.
///
/// Structural, and deliberately so: the test above shows the pins fire on the
/// three arms that exist, this shows a fourth arm could not miss them.
#[test]
fn d1_the_pins_come_from_one_derivation() {
    use super::chips::hash::{MODE_SELECTORS, NUM_UNREAD_INPUT_PINS};

    // Every selector is in the table exactly once, and the table agrees with the
    // layout's contiguous one-hot span.
    assert_eq!(MODE_SELECTORS.len(), super::layout::hash::NUM_SELECTORS);
    let mut cols_seen: Vec<usize> = MODE_SELECTORS.iter().map(|(c, _)| *c).collect();
    cols_seen.sort_unstable();
    let span: Vec<usize> = (super::layout::hash::MODE_C
        ..super::layout::hash::MODE_C + super::layout::hash::NUM_SELECTORS)
        .collect();
    assert_eq!(cols_seen, span, "the selectors are the one-hot span");

    // Four pins per unread cell, over the two cells some mode does not read.
    let unread: usize = (1..3)
        .filter(|slot| {
            MODE_SELECTORS
                .iter()
                .any(|(_, m)| m.num_input_cells() <= *slot)
        })
        .count();
    assert_eq!(NUM_UNREAD_INPUT_PINS, 4 * unread);

    // And the mode that motivated them really does read one cell.
    assert_eq!(HashMode::Leaf.num_input_cells(), 1);
    assert_eq!(HashMode::Compress.num_input_cells(), 2);
    assert_eq!(HashMode::Permute.num_input_cells(), 3);
}
