//! `LFM_BLAKE3` as a MEMBER of the machine — the chip group, its emitters and
//! the framing above them.
//!
//! [`super::blake3_probe`] proves the chip standalone against a synthetic
//! `LfmMem` mirror, which is what prices it. This suite is the other half: the
//! chip reached through [`super::builder::LfmBuilder::blake3_compress`], the
//! `Blake3Chain` framing emitted over it, and the constructions the wrap builds
//! from that — against the host hash, with a control for every framing choice
//! the emitter makes.
//!
//! ## What the oracle is, and why it is worth more than a KAT table
//!
//! `crypto::hash::blake3::chain::blake3_chain` is the host hash the RV64 prover
//! commits with, and at **seven** rounds it is the `blake3` crate bit for bit
//! over every message up to one chunk (`seven_round_chain_is_the_blake3_crate`).
//! So comparing the in-machine digest against it anchors this emitter to the
//! published hash through one hop, not to a table this repository generated.
//! The 6-round arm differs from that by a loop bound alone.
//!
//! ## What this suite does NOT establish
//!
//! Nothing here ratifies A6R (the 6-round assumption) or the `Blake3Chain`
//! DRAFT forks F1–F3 — `t = 0` throughout, the flag schedule, and no leaf/parent
//! domain separation. The emitter ENCODES all three; a reversal is an emitter
//! rewrite plus a re-bless, and these tests would then pin the new schedule
//! exactly as they pin this one.

use stark::constraints::builder::{ConstraintSet, check_dense_index_set};
use stark::proof::options::{GoldilocksCubicProofOptions, ProofOptions};

use crate::tables::types::{FE, GoldilocksExtension, GoldilocksField, VmTable};

use super::blake3::chain::{BLOCK_LEN, blake3_chain, block_flags, block_len_of, num_blocks};
use super::blake3::{BLAKE3_IV, BLAKE3_ROUNDS, blake3_compress_rounds};
use super::blake3_chip::{self, Blake3LfmConstraints, NUM_CONSTRAINTS, cols};
use super::builder::{Cell, LfmBuilder};
use super::compiler::compile;
use super::edsl::{self, WrapHash};
use super::executor::{LfmExecError, execute};
use super::hash::TestPermutation;
use super::keccak_host::{num_stream_halves, pack_stream};
use super::programs::{blake3_sponge_program, keccak_sponge_program};
use super::proof::{lfm_prove, verify_against};
use super::registry::build_artifacts;
use super::word::{LfmWord, base_word};

type F = GoldilocksField;
type E = GoldilocksExtension;

fn options() -> ProofOptions {
    GoldilocksCubicProofOptions::with_blowup(2).expect("blowup=2 is valid")
}

/// The message the framing tests are taken over: byte `i` is `37i + 11`, the
/// generator `crypto`'s own chain KATs use, so a length here and a length there
/// are the same bytes.
fn message(len: usize) -> Vec<u8> {
    (0..len)
        .map(|i| (i as u8).wrapping_mul(37).wrapping_add(11))
        .collect()
}

fn sponge_arenas(msg: &[u8]) -> Vec<Vec<LfmWord>> {
    vec![pack_stream(msg).into_iter().map(base_word).collect()]
}

/// The 32 digest bytes out of two published machine words.
fn digest_bytes(public: &[(u32, LfmWord)]) -> [u8; 32] {
    use math::field::traits::IsPrimeField;
    let mut out = [0u8; 32];
    for (w, (_, word)) in public.iter().enumerate().take(2) {
        for (l, lane) in word.iter().enumerate() {
            let v = GoldilocksField::canonical(lane.value()) as u32;
            out[4 * (4 * w + l)..4 * (4 * w + l) + 4].copy_from_slice(&v.to_le_bytes());
        }
    }
    out
}

/// The lengths every framing boundary lives at, and what each discriminates:
/// the empty message is ONE block (0); `block_len` is the true length and the
/// tail is zero-padded (1, 31, 63); 64 bytes is the parent form; 65 moves
/// `CHUNK_END | ROOT` off block 0; an exact multiple emits no spurious final
/// block (128); interior blocks carry no flags at all (192, 256).
///
/// ★ **Through the chunk boundary, deliberately.** The list stopped at 256,
/// which left the construction's most consequential seam untested by any digest
/// comparison: at 1024 this chain is still standard BLAKE3, and past it the
/// construction knowingly leaves the standard (the standard would start chunk 1
/// with `t = 1` and a reset chaining value; this keeps one unbounded chunk).
/// 1023/1024/1025 and 1087/1088 bracket that seam from both sides, 511/512/513
/// bracket an interior multiple, and 2048 is two chunks past it. A framing bug
/// that only bites after the first chunk would have passed the old list.
const FRAMING_LENS: [usize; 19] = [
    0, 1, 31, 63, 64, 65, 127, 128, 192, 256, 511, 512, 513, 1023, 1024, 1025, 1087, 1088, 2048,
];

// =========================================================================
// The framing, against the host hash
// =========================================================================

/// ★ THE ANCHOR: the emitted `Blake3Chain` IS the host's, at every length a
/// block boundary can be wrong at.
///
/// Execution only — what it establishes is that the EMITTER computes the right
/// function; that the CHIP computes what the executor mirrors is the
/// prove-and-verify test below (standing method rule 2).
#[test]
fn the_emitted_chain_is_the_host_blake3_chain() {
    for len in FRAMING_LENS {
        let msg = message(len);
        let program = blake3_sponge_program(len);
        let exec = execute(&program, &sponge_arenas(&msg), &TestPermutation)
            .unwrap_or_else(|e| panic!("len {len}: the emitted chain must execute: {e:?}"));
        assert_eq!(
            digest_bytes(&exec.public_words),
            blake3_chain(&msg),
            "len {len}: the machine's digest must be the host chain's"
        );
    }
}

/// NON-VACUITY for the anchor: distinct lengths must give distinct digests, or
/// the test above would pass for an emitter that ignored the message.
///
/// It is also **P4** in its cheapest observable form: lengths sharing a padded
/// block (31 vs 32, 64 vs 65's first block) must not collide, which is what
/// says `block_len` and the flag schedule are consumed.
#[test]
fn distinct_lengths_give_distinct_machine_digests() {
    let mut seen: Vec<(usize, [u8; 32])> = Vec::new();
    for len in FRAMING_LENS {
        let msg = message(len);
        let program = blake3_sponge_program(len);
        let exec = execute(&program, &sponge_arenas(&msg), &TestPermutation).expect("execute");
        let digest = digest_bytes(&exec.public_words);
        for (other, d) in &seen {
            assert_ne!(*d, digest, "lengths {other} and {len} collide");
        }
        seen.push((len, digest));
    }
}

/// ★ The highest-value falsification: break ONE framing choice at a time and
/// watch the digest move.
///
/// The emitter reads its schedule from `crypto` (`num_blocks`, `block_flags`,
/// `block_len_of`) so it cannot restate it wrongly — which means the thing left
/// to check is that the CHIP consumes each field at all. A chip that ignored
/// `flags`, or `block_len`, or the counter would pass every KAT above at a
/// single length and produce a valid proof of a different hash the moment the
/// message crossed a block.
///
/// Driven through the raw builder rather than the framing, because breaking a
/// framing choice is precisely what the framing will not do.
#[test]
fn breaking_one_framing_choice_at_a_time_breaks_the_digest() {
    let h: [u32; 8] = BLAKE3_IV;
    let m: [u32; 16] = core::array::from_fn(|i| 0x0123_4567u32.wrapping_mul(i as u32 + 1));

    let honest = blake3_compress_rounds(
        &h,
        &m,
        0,
        BLOCK_LEN as u32,
        block_flags(0, 1),
        BLAKE3_ROUNDS,
    );

    // Each perturbation names the framing decision it breaks.
    let broken: [(&str, u64, u32, u32); 4] = [
        (
            "the counter is not zero (F1)",
            1,
            BLOCK_LEN as u32,
            block_flags(0, 1),
        ),
        (
            "the counter's HIGH half is not zero",
            1u64 << 32,
            BLOCK_LEN as u32,
            block_flags(0, 1),
        ),
        (
            "block_len is the shape, not the true byte count",
            0,
            (BLOCK_LEN - 1) as u32,
            block_flags(0, 1),
        ),
        (
            "the flag schedule is an interior block, not a lone one",
            0,
            BLOCK_LEN as u32,
            block_flags(1, 3),
        ),
    ];
    for (what, t, block_len, flags) in broken {
        assert_ne!(
            blake3_compress_rounds(&h, &m, t, block_len, flags, BLAKE3_ROUNDS),
            honest,
            "the compression must depend on every framing field — {what}"
        );
    }

    // And the machine agrees with the primitive on the honest framing, through
    // a real `Instr::Blake3` rather than through the framing that produced it.
    let program = raw_compress_program(&h, &m, 0, BLOCK_LEN as u32, block_flags(0, 1));
    let exec = execute(&program, &[], &TestPermutation).expect("a raw compression executes");
    assert_eq!(
        published_words(&exec.public_words),
        honest,
        "the chip's output must be the primitive's, field for field"
    );
}

/// A program that issues ONE raw compression over compile-time constants and
/// publishes all sixteen output words.
fn raw_compress_program(
    h: &[u32; 8],
    m: &[u32; 16],
    t: u64,
    block_len: u32,
    flags: u32,
) -> super::compiler::LfmProgram {
    let mut b = LfmBuilder::new().with_wrap_hash(WrapHash::Blake3);
    let word = |b: &mut LfmBuilder, lanes: [u32; 4]| -> Cell {
        b.digest_const(core::array::from_fn(|i| FE::from(u64::from(lanes[i]))))
            .as_cell()
    };
    let h_cells = [
        word(&mut b, [h[0], h[1], h[2], h[3]]),
        word(&mut b, [h[4], h[5], h[6], h[7]]),
    ];
    let m_cells: [Cell; 4] =
        core::array::from_fn(|w| word(&mut b, core::array::from_fn(|l| m[4 * w + l])));
    let params = word(&mut b, [t as u32, (t >> 32) as u32, block_len, flags]);
    let out = b.blake3_compress(h_cells, m_cells, params);
    for cell in out {
        b.public(cell);
    }
    compile(b.finish())
}

/// The sixteen `u32`s behind four published machine words.
fn published_words(public: &[(u32, LfmWord)]) -> [u32; 16] {
    use math::field::traits::IsPrimeField;
    core::array::from_fn(|i| {
        let (w, l) = (i / 4, i % 4);
        GoldilocksField::canonical(public[w].1[l].value()) as u32
    })
}

/// The closed-form schedule the emitter reads is the one it emits — the
/// emitter's own accounting, checked against `crypto`'s.
///
/// A cheap statement with a real job: it is what says the eDSL and the host
/// hasher agree about how many compressions a message of a given length costs,
/// which is the number the census projects from.
#[test]
fn the_emitted_block_count_is_the_chains() {
    for len in FRAMING_LENS.iter().chain(&[1024usize, 1088]) {
        let program = blake3_sponge_program(*len);
        let compressions = program
            .instrs
            .iter()
            .filter(|i| matches!(i, super::instr::Instr::Blake3(_)))
            .count();
        assert_eq!(
            compressions,
            num_blocks(*len),
            "len {len}: one compression per block, and the empty message is ONE"
        );
        // The last block carries the true remaining byte count, every earlier
        // one a full block.
        assert_eq!(
            block_len_of(num_blocks(*len) - 1, *len) as usize,
            len - BLOCK_LEN * (num_blocks(*len) - 1),
            "len {len}: the final block_len is the remainder"
        );
    }
}

// =========================================================================
// The chip as a member of the machine
// =========================================================================

/// ★ **MANDATORY, and release-visible.** Every constraint index is emitted
/// exactly once.
///
/// `EmitTracker`'s duplicate assert is `#[cfg(debug_assertions)]` and this
/// workspace has no `[profile.release]` override, so under the house
/// `cargo test --release` convention a second `emit_base(idx, …)` **silently
/// overwrites the first** — counts still fill `0..N`, so `NUM_CONSTRAINTS`, the
/// predicted-count tests and `assert_complete` all pass with a constraint
/// deleted. `check_dense_index_set` runs with no cfg, which is what makes this
/// test see the release build's actual emission.
///
/// The RATE-4 near-miss is the precedent: a hardcoded index there would have
/// overwritten four output pins with nothing failing.
#[test]
fn every_constraint_index_is_emitted_exactly_once() {
    let set = Blake3LfmConstraints;
    let meta = <Blake3LfmConstraints as ConstraintSet<F, E>>::meta(&set);
    check_dense_index_set(&meta, NUM_CONSTRAINTS)
        .unwrap_or_else(|e| panic!("LFM_BLAKE3 constraint body: {e}"));
}

/// The chip is a registered member of the fixed set, at the slot the roots and
/// the digest are built against.
///
/// Pinned by NAME and by INDEX together: the census maps array slots onto
/// `LFM_CHIP_NAMES` across the `KECCAK_RND` slot, and a placement that moved
/// one without the other would leave every root one position out with nothing
/// in the arithmetic to notice.
#[test]
fn the_chip_occupies_its_registered_slot() {
    use super::airs::{KECCAK_RND_SLOT, LFM_CHIP_NAMES, NUM_LFM_CHIPS};
    assert_eq!(NUM_LFM_CHIPS, 15, "the promotion is 14 -> 15");
    assert_eq!(LFM_CHIP_NAMES[11], "LFM_BLAKE3");
    assert_eq!(
        KECCAK_RND_SLOT, 12,
        "LFM_BLAKE3 sits before the hosted keccak family, so the family moved up"
    );
    assert_eq!(LFM_CHIP_NAMES[KECCAK_RND_SLOT], "KECCAK_RND");
    assert_eq!(LFM_CHIP_NAMES[13], "KECCAK_RC");
    assert_eq!(LFM_CHIP_NAMES[14], "BITWISE");
    // Every registry entry carries a root and a height for the new slot.
    for entry in super::registry::LFM_REGISTRY {
        assert_eq!(
            entry.log_heights[11], 2,
            "{:?}: a program with no compression still commits the chip's empty \
             group, padded to the 4-row minimum — the fixed-machine principle",
            entry.kind
        );
    }
}

/// ★ The registry blessing is INVARIANT under `BLAKE3_ROUNDS`, and the chip's
/// witness is NOT — the two halves of one claim.
///
/// Without this, one registry entry describes two machines: `blake3-6round`
/// moves `NUM_COLUMNS` 3556 → 3076, `NUM_CONSTRAINTS` 897 → 769 and the
/// interaction count 1453 → 1261. A proof built under one round count and
/// verified under the other fails CLOSED — the OOD width check rejects it
/// before a constraint is evaluated — but it fails on a width mismatch rather
/// than on the axis being NAMED, and named is exactly what `hasher` gets a
/// `program_id` fold for and what `CommitmentHash` gets a compile-time guard
/// for. The round count is the third such axis and has neither.
///
/// What makes blessing once nonetheless correct is that the round count lives
/// **entirely in the value columns**: the preprocessed instruction group is
/// addresses, multiplicities, the reversed-digest pair and `MU`, and no term of
/// it mentions `NUM_G`. So no root and no log-height can move with the feature,
/// and the one table is readable under either build.
///
/// Both directions are asserted, because the prefix not moving means nothing
/// unless something else does. Confirmed empirically as well: regenerating
/// `LFM_REGISTRY` under `--features blake3-6round` reproduces the committed
/// table exactly — all 3,072 root and digest bytes, all six `log_heights`, all
/// six `program_id`s.
#[test]
fn the_registry_blessing_is_round_count_invariant() {
    // ---- the prefix is a function of the I/O shape alone.
    assert_eq!(
        cols::PREP_WIDTH,
        blake3_chip::IN_WORDS + 2 * blake3_chip::OUT_WORDS + 2 * cols::DIGEST_WORDS + 1,
        "the instruction group is 7 input addresses, 4 output addresses and their \
         multiplicities, 2 reversed-digest addresses and theirs, and MU — not one \
         term of it is a function of the round count"
    );
    // And the committed group IS that prefix, so no root can move with it.
    let program = blake3_sponge_program(65);
    assert_eq!(
        program.groups.blake3.width,
        cols::PREP_WIDTH,
        "the committed group is the preprocessed prefix and nothing else"
    );

    // ---- NON-VACUITY: the witness DOES move, so the separation is real.
    assert_eq!(blake3_chip::NUM_G, BLAKE3_ROUNDS * 8);
    let (value_columns, constraints, interactions, other_value_columns) = if BLAKE3_ROUNDS == 6 {
        (3_056usize, 769usize, 1_261usize, 3_536usize)
    } else {
        (3_536, 897, 1_453, 3_056)
    };
    assert_eq!(
        cols::NUM_COLUMNS - cols::PREP_WIDTH,
        value_columns,
        "the value columns are round-dependent"
    );
    assert_eq!(NUM_CONSTRAINTS, constraints);
    assert_eq!(blake3_chip::bus_interactions().len(), interactions);
    assert_ne!(
        value_columns, other_value_columns,
        "the two round counts must be two different machines, or this test \
         asserts an invariance with nothing to be invariant under"
    );
}

/// An input lane at or above `2^32` is REJECTED, not reduced.
///
/// The chip recomposes each `u32` from four BITWISE-range-checked byte columns,
/// so no such value exists on the AIR side and the program would be unprovable.
/// Failing in the executor, with a reason, is what turns that into a diagnosable
/// error instead of an unbalanced bus.
#[test]
fn an_out_of_range_lane_is_rejected_rather_than_reduced() {
    let mut b = LfmBuilder::new().with_wrap_hash(WrapHash::Blake3);
    let ok = b
        .digest_const([FE::zero(), FE::zero(), FE::zero(), FE::zero()])
        .as_cell();
    // Lane 0 of the chaining value is 2^32 — one past the largest `u32`.
    let bad = b
        .digest_const([FE::from(1u64 << 32), FE::zero(), FE::zero(), FE::zero()])
        .as_cell();
    let out = b.blake3_compress([bad, ok], [ok, ok, ok, ok], ok);
    b.public(out[0]);
    let program = compile(b.finish());

    match execute(&program, &[], &TestPermutation) {
        Err(LfmExecError::NotU32Half { lane, .. }) => assert_eq!(lane, 0),
        other => panic!("a non-u32 lane must be rejected, got {other:?}"),
    }

    // HONEST CONTROL: the same shape with every lane a legal `u32` executes.
    let mut b = LfmBuilder::new().with_wrap_hash(WrapHash::Blake3);
    let ok = b
        .digest_const([FE::zero(), FE::zero(), FE::zero(), FE::zero()])
        .as_cell();
    let good = b
        .digest_const([
            FE::from(u64::from(u32::MAX)),
            FE::zero(),
            FE::zero(),
            FE::zero(),
        ])
        .as_cell();
    let out = b.blake3_compress([good, ok], [ok, ok, ok, ok], ok);
    b.public(out[0]);
    let program = compile(b.finish());
    assert!(
        execute(&program, &[], &TestPermutation).is_ok(),
        "u32::MAX is a legal lane and must execute"
    );
}

/// ★ The chip PROVES and VERIFIES as a member of the set — the only kind of
/// test that says anything about the constraints (method rule 2).
#[test]
fn the_blake3_chip_proves_and_verifies() {
    let opts = options();
    // 65 bytes: two blocks, so the flag schedule and the chaining value are
    // both exercised rather than collapsing into the single-block case.
    for len in [0usize, 64, 65, 200] {
        let msg = message(len);
        let program = blake3_sponge_program(len);
        let artifacts = build_artifacts(&program, &opts);
        let proved = lfm_prove(&program, &artifacts, &sponge_arenas(&msg), &opts)
            .unwrap_or_else(|e| panic!("len {len}: prove failed: {e:?}"));
        assert_eq!(
            digest_bytes(&proved.public_words),
            blake3_chain(&msg),
            "len {len}: the PROVED digest must be the host chain's"
        );
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
            "len {len}: the machine proof of Blake3Chain must verify"
        );
    }
}

/// Tampering with the chip's witness is not accepted.
///
/// The output columns are corrupted after trace generation, so the trace is
/// internally inconsistent in exactly the way a prover trying to claim a
/// different digest would make it.
#[test]
fn tampering_with_the_blake3_witness_is_not_accepted() {
    let opts = options();
    let msg = message(65);
    let program = blake3_sponge_program(65);
    let artifacts = build_artifacts(&program, &opts);
    let exec = execute(&program, &sponge_arenas(&msg), &TestPermutation).expect("execute");
    let mut traces = super::trace::build_traces(&program, &exec.records);

    // One output byte of the first compression.
    let col = cols::out_word(0, 0);
    let old = traces.blake3.main_table.get_row(0)[col];
    traces.blake3.main_table.set_fe(0, col, old + FE::one());

    let proved = super::proof::prove_traces(&artifacts, &mut traces, &exec.public_words, &opts);
    match proved {
        Err(_) => {}
        Ok(proof) => assert!(
            !verify_against(
                &artifacts.roots,
                &artifacts.program_id,
                artifacts.keccak_rnd_chunks,
                &proof,
                &exec.public_words,
                &opts,
                artifacts.hasher,
            ),
            "a tampered LFM_BLAKE3 witness must not verify"
        ),
    }
}

/// The reversed digest is the digest backwards, and a row that does not ask for
/// one emits nothing.
///
/// Both halves matter. The first is the value; the second is that the two extra
/// sends really are inert when unused, which is what makes the feature free
/// rather than a cost every compression pays.
#[test]
fn the_reversed_digest_is_the_digest_backwards() {
    use super::layout::blake3 as l;

    let len = 100usize;
    let msg = message(len);

    let mut b = LfmBuilder::new().with_wrap_hash(WrapHash::Blake3);
    let num_halves = num_stream_halves(len) as u32;
    let arena = b.declare_arena(num_halves);
    let stream: Vec<_> = (0..num_halves).map(|i| b.hint_felt(arena, i)).collect();
    let (plain, rev) = edsl::blake3_256_with_rev(&mut b, &stream, len);
    b.public(plain[0]);
    b.public(plain[1]);
    b.public(rev[0]);
    b.public(rev[1]);
    let program = compile(b.finish());

    let exec = execute(&program, &sponge_arenas(&msg), &TestPermutation).expect("execute");
    let plain_bytes = digest_bytes(&exec.public_words[..2]);
    let rev_bytes = digest_bytes(&exec.public_words[2..]);
    assert_eq!(plain_bytes, blake3_chain(&msg), "the plain digest");
    let mut expected = plain_bytes;
    expected.reverse();
    assert_eq!(
        rev_bytes, expected,
        "the reversed digest is the plain one, backwards"
    );
    // And it is not vacuous: the message is chosen so the digest is not a
    // palindrome.
    assert_ne!(
        rev_bytes, plain_bytes,
        "a palindromic digest proves nothing"
    );

    // The rows that did NOT request a reversal carry zero rev multiplicities,
    // so their two extra sends contribute nothing to the bus.
    let group = &program.groups.blake3;
    let last = group.real_rows - 1;
    for row in 0..last {
        for w in 0..l::DIGEST_WORDS {
            assert_eq!(
                *group.at(row, l::rev_mult(w)),
                FE::zero(),
                "row {row} did not request a reversed digest"
            );
        }
    }
    assert!(
        (0..l::DIGEST_WORDS).any(|w| *group.at(last, l::rev_mult(w)) != FE::zero()),
        "the final row DID request one, or the control above is vacuous"
    );
}

// =========================================================================
// Cross-hash: the two configurations are two machines
// =========================================================================

/// ★ **Required in BOTH directions.** A program under one wrap hash is not a
/// program under the other, and neither one's proof verifies against the
/// other's artifacts.
///
/// This is the property that says a flip is a flip rather than a relabelling.
/// The two programs have the same SHAPE — same length, same public output — so
/// what separates them is the emitted hash and nothing else.
#[test]
fn the_two_wrap_hashes_produce_mutually_unverifiable_proofs() {
    let opts = options();
    let len = 200usize;
    let msg = message(len);

    let keccak_program = keccak_sponge_program(len);
    let blake3_program = blake3_sponge_program(len);
    let keccak_artifacts = build_artifacts(&keccak_program, &opts);
    let blake3_artifacts = build_artifacts(&blake3_program, &opts);

    assert_ne!(
        keccak_artifacts.program_id, blake3_artifacts.program_id,
        "the two hashes must be two program identities"
    );

    let keccak_proof = lfm_prove(
        &keccak_program,
        &keccak_artifacts,
        &sponge_arenas(&msg),
        &opts,
    )
    .expect("keccak prove");
    let blake3_proof = lfm_prove(
        &blake3_program,
        &blake3_artifacts,
        &sponge_arenas(&msg),
        &opts,
    )
    .expect("blake3 prove");

    // The digests differ, which is the cheapest statement that the emitters
    // really are different functions.
    assert_ne!(
        digest_bytes(&keccak_proof.public_words),
        digest_bytes(&blake3_proof.public_words),
        "the two hashes of the same message must differ"
    );

    let verifies = |artifacts: &super::registry::LfmArtifacts, proved: &super::proof::LfmProof| {
        verify_against(
            &artifacts.roots,
            &artifacts.program_id,
            artifacts.keccak_rnd_chunks,
            &proved.proof,
            &proved.public_words,
            &opts,
            artifacts.hasher,
        )
    };
    // The honest control comes first: each proof verifies under its own.
    assert!(
        verifies(&keccak_artifacts, &keccak_proof),
        "keccak honest path"
    );
    assert!(
        verifies(&blake3_artifacts, &blake3_proof),
        "blake3 honest path"
    );
    // And neither crosses.
    assert!(
        !verifies(&blake3_artifacts, &keccak_proof),
        "a keccak proof must not verify against the BLAKE3 program's artifacts"
    );
    assert!(
        !verifies(&keccak_artifacts, &blake3_proof),
        "a BLAKE3 proof must not verify against the keccak program's artifacts"
    );
}

/// The Merkle constructions agree with the host tree under BOTH hashes, through
/// ONE emitter.
///
/// `WrapHash::merkle_walk` and `merkle_tree_root` are written once and dispatch
/// only at the parent hash, so this is what says the parameterization did not
/// silently specialize to one of them.
#[test]
fn the_merkle_constructions_agree_with_the_host_under_both_hashes() {
    use super::keccak_host;

    // A four-leaf tree over 32-byte nodes, built the way `hash_new_parent`
    // does: `hash(left ‖ right)`, no domain separation, no ordering flag.
    let leaves: [[u8; 32]; 4] =
        core::array::from_fn(|i| core::array::from_fn(|j| (i * 32 + j) as u8));

    for hash in [WrapHash::Keccak, WrapHash::Blake3] {
        let host_parent = |l: &[u8; 32], r: &[u8; 32]| -> [u8; 32] {
            let mut bytes = Vec::with_capacity(64);
            bytes.extend_from_slice(l);
            bytes.extend_from_slice(r);
            match hash {
                WrapHash::Keccak => keccak_host::keccak256(&bytes),
                WrapHash::Blake3 => blake3_chain(&bytes),
            }
        };
        let host_root = host_parent(
            &host_parent(&leaves[0], &leaves[1]),
            &host_parent(&leaves[2], &leaves[3]),
        );

        // The same tree, in the machine, from hinted leaf digests.
        let mut b = LfmBuilder::new().with_wrap_hash(hash);
        let arena = b.declare_arena(8);
        let cells: Vec<[Cell; 2]> = (0..4)
            .map(|i| [b.hint_word(arena, 2 * i), b.hint_word(arena, 2 * i + 1)])
            .collect();
        let root = edsl::wrap_merkle_tree_root(&mut b, &cells);
        b.public(root[0]);
        b.public(root[1]);
        let program = compile(b.finish());

        let arena_words: Vec<LfmWord> = leaves
            .iter()
            .flat_map(super::proof_arena::commitment_words)
            .collect();
        let exec = execute(&program, &[arena_words], &TestPermutation)
            .unwrap_or_else(|e| panic!("{hash:?}: the tree build must execute: {e:?}"));
        assert_eq!(
            digest_bytes(&exec.public_words),
            host_root,
            "{hash:?}: the machine's root must be the host tree's"
        );
    }
}

// =========================================================================
// Rider 1 — the draw schedule, not just the hash
// =========================================================================

/// ★ The BLAKE3 configuration draws TWO candidates per base coordinate, and the
/// keccak one draws one.
///
/// Emit-time bookkeeping, which is where the schedule lives: `out_pos` is what
/// decides when a refill lands, so a machine that drew a different number would
/// squeeze at different points and reproduce different challenges.
#[test]
fn the_configurations_draw_different_numbers_of_candidates() {
    use super::transcript_replay::TranscriptReplay;

    for (hash, per_coordinate) in [(WrapHash::Keccak, 8usize), (WrapHash::Blake3, 16usize)] {
        let mut b = LfmBuilder::new().with_wrap_hash(hash);
        let mut t = TranscriptReplay::new(b"seed");
        let before = t.out_pos();
        let _ = t.sample_felt(&mut b);
        let after = t.out_pos();
        // The first draw forces a refill, so the buffer position after one
        // coordinate IS the bytes that coordinate consumed.
        assert_eq!(
            (after + super::keccak_host::SQUEEZE_LEN - before) % super::keccak_host::SQUEEZE_LEN,
            per_coordinate,
            "{hash:?}: one base coordinate must consume {per_coordinate} bytes"
        );
    }
}

/// ★ The in-range predicate is the one `assert_canonical` encodes, as a BIT —
/// including the two boundary candidates that decide it.
///
/// `candidate ≥ p ⟺ hi = 2^32 − 1 ∧ lo ≠ 0`. The vectors are the exact corners:
/// `p − 1` (in range, and the largest that is), `p` (the smallest that is not),
/// and a candidate with the maximal `hi` but a zero `lo` (in range, and the case
/// a naive `hi = 2^32 − 1 ⇒ reject` would get wrong).
#[test]
fn the_in_range_predicate_is_canonicity_as_a_bit() {
    use super::transcript_replay::{Candidate, candidate_in_range};
    use math::field::traits::IsPrimeField;

    const HI_MAX: u64 = 0xFFFF_FFFF;
    // (hi, lo, expected in_range)
    let vectors: [(u64, u64, bool); 6] = [
        (0, 0, true),
        (0, 1, true),
        (HI_MAX - 1, HI_MAX, true),
        (HI_MAX, 0, true),  // p − 1 exactly
        (HI_MAX, 1, false), // p exactly
        (HI_MAX, HI_MAX, false),
    ];

    for (hi, lo, want) in vectors {
        let mut b = LfmBuilder::new().with_wrap_hash(WrapHash::Blake3);
        let arena = b.declare_arena(2);
        let c = Candidate {
            lo: b.hint_felt(arena, 0),
            hi: b.hint_felt(arena, 1),
        };
        let bit = candidate_in_range(&mut b, c);
        b.public(bit.as_cell());
        let program = compile(b.finish());
        let arenas = vec![vec![base_word(FE::from(lo)), base_word(FE::from(hi))]];
        let exec = execute(&program, &arenas, &TestPermutation)
            .unwrap_or_else(|e| panic!("({hi:#x}, {lo:#x}): must execute: {e:?}"));
        let got = GoldilocksField::canonical(exec.public_words[0].1[0].value());
        assert_eq!(
            got,
            u64::from(want),
            "candidate hi={hi:#x} lo={lo:#x}: in_range must be {want}"
        );
    }
}

/// HONEST-PATH CONTROL for the whole stage: keccak still proves and verifies
/// through the rewritten emitters.
///
/// Without this, an over-broad emitter change reads as a pass — every BLAKE3
/// test above would go green on a tree where the keccak path had been broken or
/// silently switched.
#[test]
fn keccak_still_proves_and_verifies_through_the_switched_emitters() {
    let opts = options();
    let len = 202usize;
    let msg = message(len);
    let program = keccak_sponge_program(len);
    let artifacts = build_artifacts(&program, &opts);
    let proved = lfm_prove(&program, &artifacts, &sponge_arenas(&msg), &opts).expect("prove");
    assert_eq!(
        digest_bytes(&proved.public_words),
        super::keccak_host::keccak256(&msg),
        "the default path must still be keccak256, byte for byte"
    );
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
        "the keccak proof must still verify"
    );
}

/// The default is keccak, everywhere, and nothing selects BLAKE3 by omission.
///
/// One line, and it is the statement the whole stage rests on: the chip and the
/// emitters landed, the flip did not.
#[test]
fn the_default_wrap_hash_is_keccak() {
    assert_eq!(WrapHash::default(), WrapHash::Keccak);
    assert_eq!(LfmBuilder::new().wrap_hash(), WrapHash::Keccak);
}

/// The chip's layout assigns every column exactly once, prefix included.
///
/// A layout with a hole is a column no constraint and no send reads — free
/// witness. A layout with an overlap is two things sharing a cell, which the
/// single-dataflow rule cannot see because both interpretations agree.
#[test]
fn the_layout_assigns_every_column_exactly_once() {
    let mut seen = vec![0usize; cols::NUM_COLUMNS];
    let mut mark = |col: usize, what: &str| {
        assert!(col < cols::NUM_COLUMNS, "{what} at {col} is past the width");
        seen[col] += 1;
    };

    for j in 0..blake3_chip::IN_WORDS {
        mark(cols::in_addr(j), "in_addr");
    }
    for j in 0..blake3_chip::OUT_WORDS {
        mark(cols::out_addr(j), "out_addr");
        mark(cols::mult(j), "mult");
    }
    for w in 0..cols::DIGEST_WORDS {
        mark(cols::rev_addr(w), "rev_addr");
        mark(cols::rev_mult(w), "rev_mult");
    }
    mark(cols::MU, "MU");
    for i in 0..blake3_chip::IN_U32 {
        for b in 0..4 {
            mark(cols::in_word(i, b), "in byte");
        }
    }
    for g in 0..blake3_chip::NUM_G {
        for k in 0..cols::G_SIZE {
            mark(cols::g_base(g) + k, "G cell");
        }
    }
    for i in 0..blake3_chip::OUT_U32 {
        for b in 0..4 {
            mark(cols::out_word(i, b), "out byte");
        }
    }

    let unassigned: Vec<usize> = (0..cols::NUM_COLUMNS).filter(|c| seen[*c] == 0).collect();
    let doubled: Vec<usize> = (0..cols::NUM_COLUMNS).filter(|c| seen[*c] > 1).collect();
    assert!(
        unassigned.is_empty(),
        "columns assigned to nothing: {unassigned:?}"
    );
    assert!(doubled.is_empty(), "columns assigned twice: {doubled:?}");
}

// =========================================================================
// ★ The transcript replay, against the REAL host transcript
// =========================================================================
//
// ## Why the single-coordinate tests above are not enough
//
// `the_configurations_draw_different_numbers_of_candidates` checks that ONE
// base coordinate consumes 16 bytes under BLAKE3 and 8 under keccak, and
// `the_in_range_predicate_is_canonicity_as_a_bit` checks the select's predicate
// at its corners. Both are structurally blind to the bug that matters, and the
// blindness is demonstrable rather than suspected: mutate the replay's schedule
// from n = 2 back to n = 1 and the FIRST challenge is still correct — candidate
// 0 is in range with probability 1 − 2⁻³², so it is the answer either way. The
// divergence appears at the SECOND challenge, because a schedule that consumed
// 8 bytes instead of 16 reads it from the wrong buffer offset and refills at
// the wrong point.
//
// ✓ EXECUTED, not argued: forcing `candidates_per_coordinate` to 1 for the
// BLAKE3 arm leaves `the_in_range_predicate_is_canonicity_as_a_bit` PASSING and
// fails the oracle below on `blake3: second base challenge` — the first is
// still right. That is the whole case for this test existing.
//
// So a consumption-schedule bug is invisible to any test that draws once. What
// sees it is a SCRIPT — several draws of different kinds, an absorb in the
// middle to invalidate the buffer, a raw `sample()` whose reversed digest is
// re-absorbed — with every published value compared against the real
// `DefaultTranscript` under the same configuration. That is what follows, and
// it is the BLAKE3 counterpart of the keccak oracle in
// `machine_tests::transcript_replay_matches_the_host`.

/// The script's seed and absorb shapes. `ABSORB_A` is a digest-sized absorb and
/// `ABSORB_B` a rate-sized one, so the segment crosses a block boundary between
/// the two draw runs under both hashes.
const ORACLE_SEED: &[u8] = b"lfm-transcript-replay-v0";
const ORACLE_ABSORB_A: usize = 32;
const ORACLE_ABSORB_B: usize = 136;
const ORACLE_QUERY_BITS: usize = 20;

fn oracle_absorbs() -> (Vec<u8>, Vec<u8>) {
    let a = (0..ORACLE_ABSORB_A)
        .map(|i| (i as u8).wrapping_mul(31).wrapping_add(7))
        .collect();
    let b = (0..ORACLE_ABSORB_B)
        .map(|i| (i as u8).wrapping_mul(17).wrapping_add(3))
        .collect();
    (a, b)
}

fn oracle_arenas() -> Vec<Vec<LfmWord>> {
    let (a, b) = oracle_absorbs();
    let mut bytes = a;
    bytes.extend_from_slice(&b);
    vec![pack_stream(&bytes).into_iter().map(base_word).collect()]
}

/// The script, emitted under a chosen wrap hash.
///
/// Deliberately mixed: two base draws, an extension draw (three coordinates, so
/// the schedule is exercised across a refill), an absorb that invalidates the
/// output buffer, a `sample_u64` draw (which takes the RAW candidate stream in
/// both configurations and must therefore NOT change), a further base draw, a
/// raw `sample()` whose reversed digest becomes the next segment, and one last
/// base draw on the far side of it.
fn oracle_replay_program(hash: WrapHash) -> super::compiler::LfmProgram {
    use super::builder::Felt;
    use super::edsl::bits_to_felt;
    use super::keccak_host::BYTES_PER_HALF;
    use super::transcript_replay::TranscriptReplay;

    let total_halves = ((ORACLE_ABSORB_A + ORACLE_ABSORB_B) / BYTES_PER_HALF) as u32;
    let halves_a = ORACLE_ABSORB_A / BYTES_PER_HALF;

    let mut b = LfmBuilder::new().with_wrap_hash(hash);
    let arena = b.declare_arena(total_halves);
    let halves: Vec<Felt> = (0..total_halves).map(|i| b.hint_felt(arena, i)).collect();
    let (absorb_a, absorb_b) = halves.split_at(halves_a);

    let mut t = TranscriptReplay::new(ORACLE_SEED);
    t.append_halves(absorb_a);
    let f0 = t.sample_felt(&mut b);
    let f1 = t.sample_felt(&mut b);
    let e = t.sample_ext(&mut b);
    t.append_halves(absorb_b);
    let q = t.sample_u64_pow2(&mut b, ORACLE_QUERY_BITS);
    let qf = bits_to_felt(&mut b, &q);
    let f2 = t.sample_felt(&mut b);
    let s = t.sample(&mut b);
    let f3 = t.sample_felt(&mut b);

    b.public(f0.as_cell());
    b.public(f1.as_cell());
    b.public(e.as_cell());
    b.public(qf.as_cell());
    b.public(f2.as_cell());
    b.public(s[0]);
    b.public(s[1]);
    b.public(f3.as_cell());
    compile(b.finish())
}

struct OracleExpectation {
    f0: FE,
    f1: FE,
    e: [FE; 3],
    q: u64,
    f2: FE,
    s: [u8; 32],
    f3: FE,
}

/// The oracle: the REAL `DefaultTranscript` under configuration `T`, driven
/// through the same script.
///
/// Instantiated over the BASE field so `sample_field_element` is one coordinate,
/// matching the machine's `sample_felt`; the extension draw in the middle is
/// three consecutive base draws, which is what the host's ext sampler does
/// (`core::array::from_fn` evaluates in index order).
fn oracle_expectation<T: crypto::fiat_shamir::transcript_hash::TranscriptHash>() -> OracleExpectation
{
    use crypto::fiat_shamir::default_transcript::DefaultTranscript;
    use crypto::fiat_shamir::is_transcript::IsTranscript;

    let (a, b) = oracle_absorbs();
    let mut h = DefaultTranscript::<GoldilocksField, T>::new(ORACLE_SEED);
    h.append_bytes(&a);
    let f0 = h.sample_field_element();
    let f1 = h.sample_field_element();
    let e: [FE; 3] = core::array::from_fn(|_| h.sample_field_element());
    h.append_bytes(&b);
    let q = h.sample_u64(1 << ORACLE_QUERY_BITS);
    let f2 = h.sample_field_element();
    let s = h.sample();
    let f3 = h.sample_field_element();
    OracleExpectation {
        f0,
        f1,
        e,
        q,
        f2,
        s,
        f3,
    }
}

fn check_against_oracle(public: &[(u32, LfmWord)], x: &OracleExpectation, what: &str) {
    assert_eq!(public.len(), 8, "{what}: published word count");
    assert_eq!(public[0].1[0], x.f0, "{what}: first base challenge");
    assert_eq!(public[1].1[0], x.f1, "{what}: second base challenge");
    for i in 0..3 {
        assert_eq!(public[2].1[i], x.e[i], "{what}: ext coordinate {i}");
    }
    assert_eq!(public[3].1[0], FE::from(x.q), "{what}: sample_u64 draw");
    assert_eq!(public[4].1[0], x.f2, "{what}: post-absorb challenge");
    assert_eq!(digest_bytes(&public[5..7]), x.s, "{what}: raw sample()");
    assert_eq!(public[7].1[0], x.f3, "{what}: post-sample challenge");
}

/// HONEST CONTROL: the keccak arm of the same harness reproduces the host.
///
/// Without it a BLAKE3 failure below says nothing — it could be the harness,
/// the script, or the comparison. This is also the statement that threading the
/// wrap hash through `TranscriptReplay` did not move the default configuration.
#[test]
fn the_keccak_replay_matches_the_host_transcript() {
    use crypto::fiat_shamir::transcript_hash::KeccakTranscriptHash;
    let exec = execute(
        &oracle_replay_program(WrapHash::Keccak),
        &oracle_arenas(),
        &TestPermutation,
    )
    .expect("the keccak replay must execute");
    check_against_oracle(
        &exec.public_words,
        &oracle_expectation::<KeccakTranscriptHash>(),
        "keccak",
    );
}

/// ★ THE ORACLE: the BLAKE3 configuration's in-machine replay reproduces the
/// host `Blake3TranscriptHash`, value for value.
///
/// This is the one test that exercises the three Stage-5 transcript changes
/// TOGETHER — the n = 2 consumption schedule across refill boundaries, the
/// select chain that picks the first in-range candidate, and the BLAKE3 squeeze
/// with its reversed digest re-absorbed as the next segment. Any one of them
/// wrong by a byte moves a published challenge, and the module note above says
/// why no single-coordinate test can see it.
#[test]
fn the_blake3_replay_matches_the_host_transcript() {
    use crypto::fiat_shamir::transcript_hash::Blake3TranscriptHash;
    let exec = execute(
        &oracle_replay_program(WrapHash::Blake3),
        &oracle_arenas(),
        &TestPermutation,
    )
    .expect("the blake3 replay must execute");
    check_against_oracle(
        &exec.public_words,
        &oracle_expectation::<Blake3TranscriptHash>(),
        "blake3",
    );
}

/// NON-VACUITY for the pair above: the two configurations must publish
/// DIFFERENT values, or both tests would pass against one oracle.
#[test]
fn the_two_replays_publish_different_challenges() {
    let keccak = execute(
        &oracle_replay_program(WrapHash::Keccak),
        &oracle_arenas(),
        &TestPermutation,
    )
    .expect("keccak replay");
    let blake3 = execute(
        &oracle_replay_program(WrapHash::Blake3),
        &oracle_arenas(),
        &TestPermutation,
    )
    .expect("blake3 replay");
    assert_ne!(
        keccak.public_words, blake3.public_words,
        "the two transcript configurations must not agree on this script"
    );
}

/// The select chain's RULE is the host's fallback rule, at every pattern of two
/// candidates including both-miss.
///
/// ⚠ What this pins and what it does not. It compares the host's
/// `candidate_under_fixed_schedule` against a transcription of the rule
/// `sample_felt` emits — it is a statement about the SPEC the emitter follows,
/// not about the instructions it emits. That the emitted instructions really
/// implement this rule is the oracle test above, end to end. Both are needed:
/// the oracle cannot reach the both-miss corner (probability ≈ 2⁻⁶⁴), and this
/// cannot see an emitter that computes the right rule over the wrong operands.
#[test]
fn the_select_chain_rule_is_the_hosts_fallback() {
    use math::field::traits::HasDefaultTranscript;
    const P: u64 = 0xFFFF_FFFF_0000_0001;

    // Verbatim from `default_transcript::candidate_under_fixed_schedule`, which
    // is `pub(crate)` in `crypto` and so cannot be called from here.
    fn candidate_under_fixed_schedule<Fld: HasDefaultTranscript>(
        n: usize,
        mut next: impl FnMut() -> u64,
    ) -> u64 {
        let mut chosen: Option<u64> = None;
        let mut last = 0u64;
        for _ in 0..n {
            let candidate = next();
            last = candidate;
            if chosen.is_none() && Fld::candidate_in_range(candidate) {
                chosen = Some(candidate);
            }
        }
        chosen.unwrap_or(last)
    }

    let vectors: [(u64, u64, &str); 4] = [
        (7, 9, "both in range -> the first"),
        (P + 5, 9, "the first misses -> the second"),
        (7, P + 5, "the second misses -> still the first"),
        (
            P + 5,
            P + 11,
            "both miss -> the LAST, which is then rejected",
        ),
    ];
    for (c0, c1, what) in vectors {
        let mut it = [c0, c1].into_iter();
        let host = candidate_under_fixed_schedule::<GoldilocksField>(2, || {
            it.next().expect("two candidates")
        });
        // `sample_felt`'s rule, transcribed: the first in range, else the last.
        let machine = if c0 < P { c0 } else { c1 };
        assert_eq!(host, machine, "{what}: c0={c0:#x} c1={c1:#x}");
    }
}

/// ★ THE FLIP INVENTORY, pinned — which registered programs the Stage-6 flip
/// moves, and which it must NOT.
///
/// Review finding rev-emit E3: every `LfmProgramSource` constructor in
/// `programs.rs` builds its own `LfmBuilder`, and only two of the twenty pass a
/// hash. At the flip each production constructor is edited individually and
/// nothing catches one left on the default — which composes with the
/// undischarged `CommitmentHash` guard into the residual R-3 risk (a valid
/// proof of the wrong digest). The full site list and the flip procedure live
/// in PA-PLAN's Stage-6 section; this is the half that can go stale silently,
/// so it is executable.
///
/// The classification is MEASURED, not asserted: a program that emits no hash
/// instruction at all is flip-inert whatever its constructor says, and a
/// program that emits `KeccakF` either must flip or is deliberately pinned. The
/// two categories are named per entry so adding a registered program forces the
/// question rather than inheriting an answer.
#[test]
fn the_flip_inventory_of_registered_programs_is_pinned() {
    use super::instr::Instr;
    use super::programs::{
        KECCAK_SPONGE_LEN, fri_toy_program, keccak_chain_program, keccak_sponge_program,
        statement_replay_program, transcript_replay_program, trivial_program,
    };

    /// What the Stage-6 flip owes each registered program.
    #[derive(Debug, PartialEq, Eq)]
    enum Fate {
        /// Emits no hash at all — the flip cannot move it.
        Inert,
        /// Emits the wrap hash. Its constructor MUST take the flip.
        MustFlip,
        /// Emits keccak deliberately: an instrument that is ABOUT keccak, whose
        /// identity is pinned in `LFM_REGISTRY`. A BLAKE3 twin would be a new
        /// program and a new row, never a re-blessing of this one.
        PinnedKeccak,
    }

    let cases: [(&str, super::compiler::LfmProgram, Fate, usize); 6] = [
        ("TrivialV0", trivial_program(), Fate::Inert, 0),
        ("FriToyV0", fri_toy_program(), Fate::Inert, 0),
        (
            "KeccakChainV0",
            keccak_chain_program(),
            Fate::PinnedKeccak,
            2,
        ),
        (
            "KeccakSpongeV0",
            keccak_sponge_program(KECCAK_SPONGE_LEN),
            Fate::PinnedKeccak,
            2,
        ),
        (
            "TranscriptReplayV0",
            transcript_replay_program(),
            Fate::MustFlip,
            6,
        ),
        (
            "StatementReplayV0",
            statement_replay_program(),
            Fate::MustFlip,
            5,
        ),
    ];

    assert_eq!(
        cases.len(),
        super::registry::LFM_REGISTRY.len(),
        "every registered program must have a stated fate — a new row without \
         one is a program the flip would move or miss by accident"
    );

    for (name, program, fate, keccak_rows) in &cases {
        let keccak = program
            .instrs
            .iter()
            .filter(|i| matches!(i, Instr::KeccakF(_)))
            .count();
        let blake3 = program
            .instrs
            .iter()
            .filter(|i| matches!(i, Instr::Blake3(_)))
            .count();
        assert_eq!(
            keccak, *keccak_rows,
            "{name}: the emitted keccak count is what the fate below is a \
             judgement about"
        );
        assert_eq!(blake3, 0, "{name}: nothing selects BLAKE3 before the flip");
        match fate {
            Fate::Inert => assert_eq!(
                keccak, 0,
                "{name} is classified flip-inert but emits {keccak} hash rows"
            ),
            Fate::MustFlip | Fate::PinnedKeccak => assert!(
                keccak > 0,
                "{name} is classified as hashing but emits none — the fate is \
                 wrong, or the program is"
            ),
        }
    }

    // NON-VACUITY: the classification must actually split the set, or "every
    // program has a fate" is satisfied by giving them all the same one.
    assert!(cases.iter().any(|c| c.2 == Fate::Inert));
    assert!(cases.iter().any(|c| c.2 == Fate::MustFlip));
    assert!(cases.iter().any(|c| c.2 == Fate::PinnedKeccak));
}

/// ★ The REVERSED-DIGEST send, PROVED — the chip's last surface that execution
/// alone does not reach.
///
/// `the_reversed_digest_is_the_digest_backwards` executes the send and checks
/// its value; `the_blake3_chip_proves_and_verifies` proves the chip but through
/// `blake3_256`, which sets no reversed-digest multiplicity, so the two extra
/// `LfmMem` sends are inert in every proof either of them builds. That leaves
/// the flipped-coefficient `Linear` — the one piece of column arithmetic
/// transcribed from `chips::keccak` onto a different chip's OUT block —
/// exercised by the executor's mirror and by nothing on the AIR side.
///
/// This proves it, at several lengths so the digest being reversed is a
/// different 32 bytes each time. A transcription error in
/// `reversed_lane_value`'s `OUT + 31 − 4l − k` would leave the executor and the
/// bus disagreeing about what was written, which is an unbalanced `LfmMem`
/// multiset: the proof fails to build, or fails to verify.
#[test]
fn the_reversed_digest_send_proves_and_verifies() {
    let opts = options();
    // 0 exercises the single-block case, 65 the chain, 200 an interior block.
    for len in [0usize, 65, 200] {
        let msg = message(len);

        let mut b = LfmBuilder::new().with_wrap_hash(WrapHash::Blake3);
        let num_halves = num_stream_halves(len) as u32;
        let arena = b.declare_arena(num_halves);
        let stream: Vec<_> = (0..num_halves).map(|i| b.hint_felt(arena, i)).collect();
        let (plain, rev) = edsl::blake3_256_with_rev(&mut b, &stream, len);
        b.public(plain[0]);
        b.public(plain[1]);
        b.public(rev[0]);
        b.public(rev[1]);
        let program = compile(b.finish());

        // The multiplicities really are nonzero, or the send under test is the
        // inert one again and this proves nothing new.
        let group = &program.groups.blake3;
        let last = group.real_rows - 1;
        assert!(
            (0..cols::DIGEST_WORDS).any(|w| *group.at(last, cols::rev_mult(w)) != FE::zero()),
            "len {len}: the reversed digest must be READ, or its send is inert"
        );

        let artifacts = build_artifacts(&program, &opts);
        let proved = lfm_prove(&program, &artifacts, &sponge_arenas(&msg), &opts)
            .unwrap_or_else(|e| panic!("len {len}: prove failed: {e:?}"));

        let plain_bytes = digest_bytes(&proved.public_words[..2]);
        let rev_bytes = digest_bytes(&proved.public_words[2..]);
        assert_eq!(plain_bytes, blake3_chain(&msg), "len {len}: plain digest");
        let mut expected = plain_bytes;
        expected.reverse();
        assert_eq!(rev_bytes, expected, "len {len}: reversed digest");
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
            "len {len}: a proof carrying the reversed-digest send must verify"
        );
    }
}

/// ★ R-8: BOTH BLAKE3 surfaces active in one machine, and the shared BITWISE
/// table balances for the right reason.
///
/// Registering `LFM_BLAKE3` while D0 keeps the socket arm means one program can
/// carry two BLAKE3 AIRs — the socket (`LFM_HASH` under `HasherKind::Blake3`,
/// a 52-byte fixed message, 128-bit digest) and the chip (a raw compression,
/// 256-bit) — and both feed lookups into the SAME 2^20-row BITWISE table. They
/// run the same `run_flow` under different `FlowConfig`s, which is exactly the
/// shape where two producers could balance the shared table between them rather
/// than each against the table: a socket row's missing lookup absorbed by a chip
/// row's spare one would leave the bus balanced and one of the two computations
/// unconstrained.
///
/// The control is the honest-path one done twice over: each surface alone
/// proves and verifies, and the two TOGETHER prove and verify. A cross-balance
/// would show up as the combined program proving while one of the singles does
/// not, or as the combined proof failing to verify — the histogram is built from
/// the senders' own enumeration, so a mismatch is an unbalanced multiset either
/// way.
#[test]
fn both_blake3_surfaces_in_one_machine_balance_bitwise() {
    use super::hash::HasherKind;
    use super::proof::lfm_prove_with_hasher;

    // Three programs: socket only, chip only, and both — the same builder
    // shapes, so what differs between them is which surface is present.
    let socket_only = |with_chip: bool, with_socket: bool| {
        let mut b = LfmBuilder::new().with_wrap_hash(WrapHash::Blake3);
        let zero = b
            .digest_const([FE::zero(), FE::zero(), FE::zero(), FE::zero()])
            .as_cell();
        if with_socket {
            // The socket: an `LFM_HASH` compress, whose lanes must be u32.
            let a = b
                .digest_const(core::array::from_fn(|i| FE::from(1_000u64 + i as u64)))
                .as_cell();
            let d = b.compress(a.as_digest(), zero.as_digest());
            b.public(d.as_cell());
        }
        if with_chip {
            // The chip: one raw compression at the parent framing.
            let m: [Cell; 4] = core::array::from_fn(|w| {
                b.digest_const(core::array::from_fn(|l| {
                    FE::from(7_000u64 + (4 * w + l) as u64)
                }))
                .as_cell()
            });
            let h = [
                b.digest_const(core::array::from_fn(|i| FE::from(u64::from(BLAKE3_IV[i]))))
                    .as_cell(),
                b.digest_const(core::array::from_fn(|i| {
                    FE::from(u64::from(BLAKE3_IV[4 + i]))
                }))
                .as_cell(),
            ];
            let params = b
                .digest_const([
                    FE::zero(),
                    FE::zero(),
                    FE::from(BLOCK_LEN as u64),
                    FE::from(u64::from(block_flags(0, 1))),
                ])
                .as_cell();
            let out = b.blake3_compress(h, m, params);
            b.public(out[0]);
        }
        compile(b.finish())
    };

    let opts = options();
    for (what, program, socket_rows, chip_rows) in [
        ("socket alone", socket_only(false, true), 1, 0),
        ("chip alone", socket_only(true, false), 0, 1),
        ("★ both surfaces", socket_only(true, true), 1, 1),
    ] {
        // NON-VACUITY: the three programs must really differ in which surface
        // they carry, or "both together verify" is one surface tested thrice.
        assert_eq!(
            program.groups.hash.real_rows, socket_rows,
            "{what}: LFM_HASH (socket) rows"
        );
        assert_eq!(
            program.groups.blake3.real_rows, chip_rows,
            "{what}: LFM_BLAKE3 (chip) rows"
        );
        let artifacts =
            super::registry::build_artifacts_with_hasher(&program, &opts, HasherKind::Blake3);
        let proved = lfm_prove_with_hasher(&program, &artifacts, &[], &opts, HasherKind::Blake3)
            .unwrap_or_else(|e| panic!("{what}: must prove: {e:?}"));
        assert!(
            verify_against(
                &artifacts.roots,
                &artifacts.program_id,
                artifacts.keccak_rnd_chunks,
                &proved.proof,
                &proved.public_words,
                &opts,
                HasherKind::Blake3,
            ),
            "{what}: must verify — a shared-BITWISE cross-balance between the \
             socket and the chip would show up here"
        );
    }
}
