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
const FRAMING_LENS: [usize; 10] = [0, 1, 31, 63, 64, 65, 127, 128, 192, 256];

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
