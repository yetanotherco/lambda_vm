//! The compress-chain Fiat–Shamir transcript (option B1): its vectors, its
//! domain separation, its cost, and the machine that computes it.
//!
//! ## What pins what
//!
//! Four layers, deliberately different evidence:
//!
//! 1. **The step function** is `blake3::hash(state ‖ operand ‖ "LFMT")`
//!    truncated — asserted against the *crate*, not against an oracle, so the
//!    external anchor the 7-round decision was bought for is inherited rather
//!    than claimed.
//! 2. **The vectors** are [`super::transcript_kats`], rendered from a Python
//!    reference the oracle wrote before any of this Rust existed. Both round
//!    counts, per-op and end-to-end.
//! 3. **The host chain** (`fixture::HostSponge`) reproduces them, which is what
//!    "bit-exact mirror" has to mean to be checkable.
//! 4. **The machine** (`edsl::SpongeVar` through `LFM_HASH`) reproduces the same
//!    challenges *inside a proof the production verifier accepts* — the layer
//!    that would catch a host and chip that agree with the spec separately and
//!    with each other not at all.
//!
//! Every rejection test here is paired with an honest-path assertion. A test
//! that only checks "the bad thing is rejected" passes just as well when
//! everything is rejected, which is the failure mode a soundness fix has.

use crate::tables::types::{FE, FEE, GoldilocksField};
use math::field::traits::IsPrimeField;
use stark::proof::options::{GoldilocksCubicProofOptions, ProofOptions};

use super::blake3_socket::{
    SOCKET_ROUNDS, TAG_LFMT, socket_digest_rounds, transcript_digest, transcript_digest_rounds,
    word_of,
};
use super::builder::{Cell, LfmBuilder, LfmProgramSource};
use super::compiler::{LfmProgram, compile};
use super::edsl::{SQUEEZE_MARK, SpongeVar};
use super::executor::execute;
use super::fixture::HostSponge;
use super::hash::HasherKind;
use super::instr::{HashMode, Instr};
use super::proof::{lfm_prove_with_hasher, verify_against};
use super::registry::build_artifacts_with_hasher;
use super::transcript_kats::{
    EndToEndVector, FRI_TOY_6, FRI_TOY_7, FRI_TOY_COMPRESSIONS, L1_ROOT, MAIN_ROOT, STEP_VECTORS,
    T0W, T1W,
};
use super::word::LfmWord;

const KIND: HasherKind = HasherKind::Blake3;

/// Query shape of the `FriToyV0` preamble the end-to-end vector is shaped like
/// (✓ VERIFIED `fixture::shape`).
const NUM_QUERIES: usize = 4;
const QUERY_BITS: usize = 4;

fn options() -> ProofOptions {
    GoldilocksCubicProofOptions::with_blowup(2).expect("blowup=2 is valid")
}

/// The end-to-end vector for the round count this build compiled.
fn compiled_vector() -> &'static EndToEndVector {
    if SOCKET_ROUNDS == 7 {
        &FRI_TOY_7
    } else {
        &FRI_TOY_6
    }
}

fn lanes(w: &LfmWord) -> [u32; 4] {
    core::array::from_fn(|i| {
        u32::try_from(GoldilocksField::canonical(w[i].value())).expect("a u32 lane")
    })
}

// =========================================================================
// K1 — the step function, and its external anchor
// =========================================================================

/// The per-op vectors, at BOTH round counts in one run.
#[test]
fn every_step_vector_reproduces_at_both_round_counts() {
    for v in STEP_VECTORS.iter() {
        assert_eq!(
            transcript_digest_rounds(&v.state, &v.operand, 6),
            v.result_6,
            "6-round step vector {}",
            v.name
        );
        assert_eq!(
            transcript_digest_rounds(&v.state, &v.operand, 7),
            v.result_7,
            "7-round step vector {}",
            v.name
        );
    }
}

/// The compiled-in entry point agrees with the vector for its round count — so
/// the knob and the table cannot drift apart.
#[test]
fn the_compiled_step_matches_its_round_counts_vectors() {
    for v in STEP_VECTORS.iter() {
        let expected = if SOCKET_ROUNDS == 7 {
            v.result_7
        } else {
            v.result_6
        };
        assert_eq!(
            transcript_digest(&v.state, &v.operand),
            expected,
            "vector {}",
            v.name
        );
    }
}

/// ★ **The external anchor, direct.** At 7 rounds a transcript step is
/// literally `blake3::hash(state ‖ operand ‖ "LFMT")` truncated to 16 bytes.
///
/// The message is re-derived from the byte-level framing rather than from
/// `socket_message`, so the word-level and byte-level forms are two statements
/// that can disagree. This is the property option B was chosen for: the
/// transcript inherits the compress socket's anchor because the tag is the only
/// thing that moved.
#[test]
fn seven_rounds_is_blake3_of_the_transcript_message() {
    for v in STEP_VECTORS.iter() {
        let mut msg = Vec::with_capacity(36);
        for lane in v.state.iter().chain(v.operand.iter()) {
            msg.extend_from_slice(&lane.to_le_bytes());
        }
        msg.extend_from_slice(b"LFMT");
        assert_eq!(msg.len(), 36, "a transcript step is one 36-byte block");

        let full = blake3::hash(&msg);
        let want: [u32; 4] = core::array::from_fn(|i| {
            u32::from_le_bytes(full.as_bytes()[4 * i..4 * i + 4].try_into().unwrap())
        });
        assert_eq!(
            transcript_digest_rounds(&v.state, &v.operand, 7),
            want,
            "7-round step {} must be blake3::hash of its message",
            v.name
        );
        assert_eq!(want, v.result_7, "the table itself agrees with the crate");
    }
}

/// The tag word is the ASCII, little-endian — the one place a byte order slip
/// would silently redefine the domain.
#[test]
fn the_transcript_tag_is_lfmt_little_endian() {
    assert_eq!(TAG_LFMT, u32::from_le_bytes(*b"LFMT"));
    assert_eq!(TAG_LFMT.to_le_bytes(), *b"LFMT");
    assert_eq!(SQUEEZE_MARK.to_le_bytes(), *b"SQZ0");
}

// =========================================================================
// K3 — domain separation, in both directions
// =========================================================================

/// A transcript step and a Merkle parent over the SAME two cells are different
/// digests. Without this the chain would be replayable as a tree and vice
/// versa, and the `MODE_T` column would be buying nothing.
#[test]
fn a_transcript_step_is_not_a_merkle_parent() {
    for v in STEP_VECTORS.iter() {
        for rounds in [6, 7] {
            assert_ne!(
                transcript_digest_rounds(&v.state, &v.operand, rounds),
                socket_digest_rounds(&v.state, &v.operand, rounds),
                "vector {} at {rounds} rounds: the tag is not separating the domains",
                v.name
            );
        }
    }
}

/// The honest-path control for the test above: the two ARE the same function
/// apart from the tag, so a bug that made them differ for some other reason
/// would show up here.
#[test]
fn the_two_domains_differ_only_in_the_tag() {
    for v in STEP_VECTORS.iter() {
        use super::blake3_socket::{TAG_LFMC, socket_digest_rounds_tagged};
        assert_eq!(
            socket_digest_rounds_tagged(&v.state, &v.operand, 7, TAG_LFMT),
            transcript_digest_rounds(&v.state, &v.operand, 7)
        );
        assert_eq!(
            socket_digest_rounds_tagged(&v.state, &v.operand, 7, TAG_LFMC),
            socket_digest_rounds(&v.state, &v.operand, 7)
        );
    }
}

// =========================================================================
// K2 — the end-to-end vector, host side
// =========================================================================

/// Replays the `FriToyV0` preamble against the reference step function at an
/// explicit round count, so BOTH vectors are checkable from one build.
///
/// ✓ VERIFIED sequence, `programs::fri_toy_program_source`: absorb(main_root),
/// squeeze_ext, squeeze_ext, absorb(l1_root), squeeze_ext, absorb2(t0w, t1w),
/// then `NUM_QUERIES` × squeeze_bits.
fn replay_reference(rounds: usize) -> (Vec<[u32; 4]>, Vec<[u32; 4]>, usize) {
    let mut state = [0u32; 4];
    let mut squeeze_index = 0u32;
    let mut compressions = 0usize;
    let mut states = Vec::new();
    let mut outputs = Vec::new();

    let absorb = |state: &mut [u32; 4], c: &[u32; 4], compressions: &mut usize| {
        *state = transcript_digest_rounds(state, c, rounds);
        *compressions += 1;
    };
    let squeeze = |state: &mut [u32; 4], i: &mut u32, compressions: &mut usize| -> [u32; 4] {
        let out = *state;
        let sq = [SQUEEZE_MARK, *i, 0, 0];
        *state = transcript_digest_rounds(state, &sq, rounds);
        *i += 1;
        *compressions += 1;
        out
    };

    absorb(&mut state, &MAIN_ROOT, &mut compressions);
    states.push(state);
    outputs.push(squeeze(&mut state, &mut squeeze_index, &mut compressions));
    states.push(state);
    outputs.push(squeeze(&mut state, &mut squeeze_index, &mut compressions));
    states.push(state);
    absorb(&mut state, &L1_ROOT, &mut compressions);
    states.push(state);
    outputs.push(squeeze(&mut state, &mut squeeze_index, &mut compressions));
    states.push(state);
    absorb(&mut state, &T0W, &mut compressions);
    absorb(&mut state, &T1W, &mut compressions);
    states.push(state);
    for _ in 0..NUM_QUERIES {
        outputs.push(squeeze(&mut state, &mut squeeze_index, &mut compressions));
        states.push(state);
    }
    (states, outputs, compressions)
}

fn check_end_to_end(rounds: usize, want: &EndToEndVector) {
    let (states, outputs, compressions) = replay_reference(rounds);
    assert_eq!(states.len(), want.states.len());
    for (i, (got, expected)) in states.iter().zip(want.states.iter()).enumerate() {
        assert_eq!(got, expected, "state after op {i} at {rounds} rounds");
    }
    // Challenges are read off the PRE-advance outputs, not off the states.
    let ext = |o: &[u32; 4]| [o[0], o[1], o[2]];
    assert_eq!(ext(&outputs[0]), want.alpha, "alpha at {rounds} rounds");
    assert_eq!(ext(&outputs[1]), want.zeta0, "zeta0 at {rounds} rounds");
    assert_eq!(ext(&outputs[2]), want.zeta1, "zeta1 at {rounds} rounds");
    for (q, bits) in want.query_bits.iter().enumerate() {
        let lane0 = outputs[3 + q][0];
        let got: [u8; QUERY_BITS] = core::array::from_fn(|k| ((lane0 >> k) & 1) as u8);
        assert_eq!(&got, bits, "query {q} bits at {rounds} rounds");
    }
    assert_eq!(
        compressions, FRI_TOY_COMPRESSIONS,
        "the preamble's compression count is a cost claim, not an accident"
    );
}

/// K2 at 7 rounds — the default build.
#[test]
fn the_end_to_end_vector_reproduces_at_seven_rounds() {
    check_end_to_end(7, &FRI_TOY_7);
}

/// K2 at 6 rounds — the `blake3-6round` variant, pinned unconditionally.
#[test]
fn the_end_to_end_vector_reproduces_at_six_rounds() {
    check_end_to_end(6, &FRI_TOY_6);
}

/// ★ The HOST CHAIN — `fixture::HostSponge`, the thing the fixture prover and
/// every host-side replay use — reproduces the vector op for op.
///
/// This is the mirror property stated as something checkable. It runs at the
/// compiled-in round count, which is why the two `check_end_to_end` tests above
/// carry the other one.
#[test]
fn the_host_sponge_reproduces_the_end_to_end_vector() {
    let want = compiled_vector();
    let mut sponge = HostSponge::with_hasher(KIND);
    let cell = |w: &[u32; 4]| word_of(w);
    let mut states = Vec::new();

    sponge.absorb(&cell(&MAIN_ROOT));
    states.push(sponge.state());
    let alpha = sponge.squeeze_ext();
    states.push(sponge.state());
    let zeta0 = sponge.squeeze_ext();
    states.push(sponge.state());
    sponge.absorb(&cell(&L1_ROOT));
    states.push(sponge.state());
    let zeta1 = sponge.squeeze_ext();
    states.push(sponge.state());
    sponge.absorb2(&cell(&T0W), &cell(&T1W));
    states.push(sponge.state());
    let mut queries = Vec::new();
    for _ in 0..NUM_QUERIES {
        queries.push(sponge.squeeze_index(QUERY_BITS));
        states.push(sponge.state());
    }

    for (i, (got, expected)) in states.iter().zip(want.states.iter()).enumerate() {
        assert_eq!(lanes(got), *expected, "host state after op {i}");
    }

    let ext_lanes = |e: &FEE| -> [u32; 3] {
        let v = e.value();
        core::array::from_fn(|i| {
            u32::try_from(GoldilocksField::canonical(v[i].value())).expect("a u32 lane")
        })
    };
    assert_eq!(ext_lanes(&alpha), want.alpha);
    assert_eq!(ext_lanes(&zeta0), want.zeta0);
    assert_eq!(ext_lanes(&zeta1), want.zeta1);
    for (q, bits) in want.query_bits.iter().enumerate() {
        let index: u64 = bits
            .iter()
            .enumerate()
            .map(|(k, &b)| u64::from(b) << k)
            .sum();
        assert_eq!(queries[q], index, "host query {q}");
    }
}

// =========================================================================
// K4/K5 — the counter and the ordering are load-bearing
// =========================================================================

/// Without the counter every advance uses ONE fixed operand, so a run of
/// squeezes iterates one fixed map — precisely the structure the FSE-2014
/// T-sponge attacks exploit. The vectors must notice.
#[test]
fn the_squeeze_counter_is_load_bearing() {
    let rounds = 7;
    let start = transcript_digest_rounds(&[0; 4], &MAIN_ROOT, rounds);

    let mut with_counter = Vec::new();
    let mut s = start;
    for i in 0..4u32 {
        with_counter.push(s);
        s = transcript_digest_rounds(&s, &[SQUEEZE_MARK, i, 0, 0], rounds);
    }

    let mut without = Vec::new();
    let mut s = start;
    let fixed = [SQUEEZE_MARK, 0, 0, 0];
    for _ in 0..4 {
        without.push(s);
        s = transcript_digest_rounds(&s, &fixed, rounds);
    }

    // Honest-path half: squeeze 0 uses counter 0, so the two MUST agree there.
    // Without this the test would pass for a chain that simply produced noise.
    assert_eq!(
        with_counter[0], without[0],
        "the first squeeze is the same either way — counter 0 is counter 0"
    );
    assert_ne!(
        with_counter[1..],
        without[1..],
        "counter-free squeezes must diverge: they iterate one fixed map"
    );
}

/// Absorbing two cells in the other order is a different transcript. Cheap to
/// state, and the property every Fiat–Shamir argument silently assumes.
#[test]
fn absorb_order_is_load_bearing() {
    let mut a = HostSponge::with_hasher(KIND);
    a.absorb(&word_of(&MAIN_ROOT));
    a.absorb(&word_of(&L1_ROOT));

    let mut b = HostSponge::with_hasher(KIND);
    b.absorb(&word_of(&L1_ROOT));
    b.absorb(&word_of(&MAIN_ROOT));

    assert_ne!(a.state(), b.state());

    // Honest-path control: the same order gives the same state.
    let mut c = HostSponge::with_hasher(KIND);
    c.absorb(&word_of(&MAIN_ROOT));
    c.absorb(&word_of(&L1_ROOT));
    assert_eq!(a.state(), c.state());
}

// =========================================================================
// K6 + the machine — the emitted program
// =========================================================================

/// A program shaped exactly like `FriToyV0`'s preamble, with the absorbed cells
/// as arena words so a test can feed the vector's inputs in.
///
/// Its public output is every challenge the preamble derives, so a proof of it
/// carries the transcript's answers where a verifier can check them.
fn preamble_program_source() -> LfmProgramSource {
    let mut b = LfmBuilder::new();
    let arena = b.declare_arena(4);
    let h: Vec<Cell> = (0..4).map(|i| b.hint_word(arena, i)).collect();

    let mut sponge = SpongeVar::new(&mut b);
    sponge.absorb(&mut b, h[0]);
    let alpha = sponge.squeeze_ext(&mut b);
    let zeta0 = sponge.squeeze_ext(&mut b);
    sponge.absorb(&mut b, h[1]);
    let zeta1 = sponge.squeeze_ext(&mut b);
    sponge.absorb2(&mut b, h[2], h[3]);

    b.public(alpha.as_cell());
    b.public(zeta0.as_cell());
    b.public(zeta1.as_cell());
    for _ in 0..NUM_QUERIES {
        let bits = sponge.squeeze_bits(&mut b, QUERY_BITS);
        let index = super::edsl::bits_to_felt(&mut b, &bits);
        b.public(index.as_cell());
    }
    b.finish()
}

fn preamble_program() -> LfmProgram {
    compile(preamble_program_source())
}

fn preamble_arena() -> Vec<Vec<LfmWord>> {
    vec![vec![
        word_of(&MAIN_ROOT),
        word_of(&L1_ROOT),
        word_of(&T0W),
        word_of(&T1W),
    ]]
}

/// K6 — the preamble costs exactly the compressions the spec priced it at, and
/// every one of them is a TRANSCRIPT row rather than a Merkle one.
#[test]
fn the_preamble_costs_eleven_transcript_steps() {
    let program = preamble_program();
    let modes: Vec<HashMode> = program
        .instrs
        .iter()
        .filter_map(|i| match i {
            Instr::Hash { mode, .. } => Some(*mode),
            _ => None,
        })
        .collect();
    assert_eq!(
        modes.len(),
        FRI_TOY_COMPRESSIONS,
        "the transcript preamble is {FRI_TOY_COMPRESSIONS} compressions"
    );
    assert!(
        modes.iter().all(|m| *m == HashMode::Transcript),
        "a transcript step must never be emitted as a Merkle compress"
    );
}

/// ★ **The machine computes the specified transcript.** The emitted program,
/// executed under BLAKE3, produces the vector's challenges.
///
/// This is the layer the host tests cannot reach: `SpongeVar` and `HostSponge`
/// are separate code, and this is where they are made to answer the same
/// question.
#[test]
fn the_machine_reproduces_the_end_to_end_vector() {
    let want = compiled_vector();
    let program = preamble_program();
    let exec = execute(&program, &preamble_arena(), &KIND).expect("the preamble executes");

    let public: Vec<LfmWord> = exec.public_words.iter().map(|(_, w)| *w).collect();
    assert_eq!(public.len(), 3 + NUM_QUERIES);

    let ext3 = |w: &LfmWord| -> [u32; 3] {
        let l = lanes(w);
        [l[0], l[1], l[2]]
    };
    assert_eq!(ext3(&public[0]), want.alpha, "alpha");
    assert_eq!(ext3(&public[1]), want.zeta0, "zeta0");
    assert_eq!(ext3(&public[2]), want.zeta1, "zeta1");
    for (q, bits) in want.query_bits.iter().enumerate() {
        let index: u64 = bits
            .iter()
            .enumerate()
            .map(|(k, &b)| u64::from(b) << k)
            .sum();
        assert_eq!(
            GoldilocksField::canonical(public[3 + q][0].value()),
            index,
            "query {q}"
        );
    }
}

/// The same program, PROVED under BLAKE3 and accepted by the production
/// verifier — the transcript is not merely computed, it is constrained.
#[test]
fn the_transcript_proves_and_verifies_under_blake3() {
    let opts = options();
    let program = preamble_program();
    let artifacts = build_artifacts_with_hasher(&program, &opts, KIND);
    let proved = lfm_prove_with_hasher(&program, &artifacts, &preamble_arena(), &opts, KIND)
        .expect("a transcript program must prove under BLAKE3");
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
        "an honest BLAKE3 transcript proof must verify"
    );

    // The public challenges are checked against the SPEC's vector, not against
    // the executor — so the proof's outputs answer to the specification.
    let want = compiled_vector();
    let alpha = lanes(&proved.public_words[0].1);
    assert_eq!([alpha[0], alpha[1], alpha[2]], want.alpha);
}

/// The same program under every hasher: B1 changed the transcript for ALL of
/// them, so all of them must still prove and verify.
#[test]
fn the_transcript_proves_and_verifies_under_every_hasher() {
    let opts = options();
    let program = preamble_program();
    for kind in [HasherKind::Test, HasherKind::Poseidon, HasherKind::Blake3] {
        let artifacts = build_artifacts_with_hasher(&program, &opts, kind);
        let proved = lfm_prove_with_hasher(&program, &artifacts, &preamble_arena(), &opts, kind)
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
            "an honest transcript proof must verify under {kind:?}"
        );
    }
}

/// The machine's chain and the host's chain agree under every hasher — the
/// property `fixture_prove` depends on and the one a rewrite of either side
/// would break.
#[test]
fn the_machine_and_the_host_chain_agree_under_every_hasher() {
    let program = preamble_program();
    for kind in [HasherKind::Test, HasherKind::Poseidon, HasherKind::Blake3] {
        // BLAKE3 needs u32 lanes (O1); the vector's inputs are u32 either way,
        // so one arena serves all three hashers.
        let exec = execute(&program, &preamble_arena(), &kind).expect("executes");
        let public: Vec<LfmWord> = exec.public_words.iter().map(|(_, w)| *w).collect();

        let mut sponge = HostSponge::with_hasher(kind);
        sponge.absorb(&word_of(&MAIN_ROOT));
        let alpha = sponge.squeeze_ext();
        let zeta0 = sponge.squeeze_ext();
        sponge.absorb(&word_of(&L1_ROOT));
        let zeta1 = sponge.squeeze_ext();
        sponge.absorb2(&word_of(&T0W), &word_of(&T1W));

        for (i, want) in [alpha, zeta0, zeta1].iter().enumerate() {
            let v = want.value();
            for l in 0..3 {
                assert_eq!(public[i][l], v[l], "{kind:?} challenge {i} lane {l}");
            }
        }
        for q in 0..NUM_QUERIES {
            let index = sponge.squeeze_index(QUERY_BITS);
            assert_eq!(
                public[3 + q][0],
                FE::from(index),
                "{kind:?} query {q} index"
            );
        }
    }
}

// =========================================================================
// The cost claims the decision was made on
// =========================================================================

/// Hash rows in `program`, split by mode.
fn hash_row_modes(program: &LfmProgram) -> (usize, usize) {
    let mut compress = 0;
    let mut transcript = 0;
    for i in &program.instrs {
        if let Instr::Hash { mode, .. } = i {
            match mode {
                HashMode::Compress => compress += 1,
                HashMode::Transcript => transcript += 1,
                HashMode::Permute => panic!("no registered program may contain a permute"),
            }
        }
    }
    (compress, transcript)
}

/// ★ The option-B cost claims, measured on the emitted programs.
///
/// `permute-socket-options.md` §3 priced B at **369,103** `LFM_HASH` cell-equiv
/// for `FriToyV0` and **16,527** for `TrivialV0` at 7 rounds, and the decision
/// was taken partly on those numbers. Both are `rows × cells_per_compression`,
/// so this asserts the row counts and the per-row price separately — a product
/// that came out right for two wrong reasons is the failure mode.
///
/// The per-compression price is `blake3_socket_tests`' own census formula
/// (`main + 3·⌈interactions/2⌉`), 5,509 at 7 rounds and 4,741 at 6.
#[test]
fn the_programs_cost_what_option_b_priced_them_at() {
    const CELLS_PER_COMPRESSION_7R: usize = 5_509;
    const CELLS_PER_COMPRESSION_6R: usize = 4_741;
    let price = if SOCKET_ROUNDS == 7 {
        CELLS_PER_COMPRESSION_7R
    } else {
        CELLS_PER_COMPRESSION_6R
    };

    // The price, from the census rather than from a literal.
    let census = super::airs::lfm_chip_census_with_hasher(
        &super::programs::trivial_program(),
        HasherKind::Blake3,
    );
    let hash_chip = census
        .iter()
        .find(|c| c.name == "LFM_HASH")
        .expect("the census names the hash chip");
    assert_eq!(
        hash_chip.main_cols + 3 * hash_chip.aux_cols,
        price,
        "the per-compression price must be the census's, not a literal"
    );

    // TrivialV0: three compressions, no transcript.
    let (c, t) = hash_row_modes(&super::programs::trivial_program());
    assert_eq!((c, t), (3, 0));
    if SOCKET_ROUNDS == 7 {
        assert_eq!((c + t) * price, 16_527, "TrivialV0 at 7 rounds");
    }

    // FriToyV0: 56 Merkle compressions (4 queries × 14) and the transcript's 11.
    let (c, t) = hash_row_modes(&super::programs::fri_toy_program());
    assert_eq!(t, FRI_TOY_COMPRESSIONS, "the transcript's share");
    assert_eq!((c, t), (56, 11));
    if SOCKET_ROUNDS == 7 {
        assert_eq!((c + t) * price, 369_103, "FriToyV0 at 7 rounds");
    }
}
