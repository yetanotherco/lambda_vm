//! GAP-FIX REC: the recursion-proof shape fixes, behind their `LAMBDA_VM_GAP_*`
//! knobs.
//!
//! R1 — `BITWISE` leaves every program whose instantiated chips send it no
//! lookup ([`ChipSet::bitwise`]). R2 — `LFM_HASH` splits into two power-of-two
//! instances where one table would be mostly padding ([`HashChunking`]).
//! The premise tests read interaction lists, split rules and digests and prove
//! nothing; the tests that prove (`*_proves_*`, `*_refused`, `an_emitted_leg_*`)
//! belong on the box.

use stark::proof::options::ProofOptions;
use stark::proof::view::MultiProofView;

use crate::GoldilocksCubicProofOptions;
use crate::tables::types::FE;

use stark::config::Commitment;

use super::airs::{
    BITWISE_SLOT, ChipSet, HASH_SLOT, LFM_CHIP_NAMES, gap_knob_value, gap_r1_enabled,
    lfm_chip_census_masked,
};
use super::chunking::HashChunking;
use super::hash::HasherKind;
use super::programs::{keccak_chain_program, trivial_program};
use super::proof::{lfm_prove, lfm_prove_with_hasher, verify_against, verify_against_artifacts};
use super::registry::{LfmArtifacts, build_artifacts, build_artifacts_with_hasher};
use super::statement::{lfm_program_id, lfm_program_id_chunked};
use super::word::LfmWord;

const HASHERS: [HasherKind; 5] = [
    HasherKind::Test,
    HasherKind::Poseidon,
    HasherKind::Blake3,
    HasherKind::Rpo,
    HasherKind::Rpx,
];

fn options() -> ProofOptions {
    GoldilocksCubicProofOptions::with_blowup(2).expect("options")
}

/// The four family combinations, each keeping `BITWISE`.
fn family_masks() -> [ChipSet; 4] {
    [(false, false), (true, false), (false, true), (true, true)].map(|(keccak, blake3)| ChipSet {
        keccak,
        blake3,
        bitwise: true,
    })
}

#[test]
fn the_gap_knob_reads_zero_one_and_unset() {
    assert!(!gap_knob_value("K", None), "unset is off");
    assert!(!gap_knob_value("K", Some("")), "empty is unset");
    assert!(!gap_knob_value("K", Some("0")));
    assert!(!gap_knob_value("K", Some(" 0 ")));
    assert!(gap_knob_value("K", Some("1")));
    assert!(gap_knob_value("K", Some(" 1\n")));
}

/// A typo stops the run rather than proving the default under the lever's name.
#[test]
#[should_panic(expected = "must be 0 or 1")]
fn a_malformed_gap_knob_stops_the_run() {
    let _ = gap_knob_value("LAMBDA_VM_GAP_R1", Some("yes"));
}

/// ★ THE PREMISE R1 RESTS ON, read off the interaction lists: the only chips
/// that touch a bus `BITWISE` receives are the keccak family, `LFM_BLAKE3`, and
/// the hash chip under the BLAKE3 socket. If any other chip gains a byte lookup,
/// this fails — and so does every mask that would have dropped its receiver.
#[test]
fn only_the_byte_lookup_chips_send_to_bitwise() {
    for mask in family_masks() {
        for hasher in HASHERS {
            assert_eq!(
                mask.bitwise_required(hasher),
                mask.keccak || mask.blake3 || hasher == HasherKind::Blake3,
                "{mask:?} under {hasher:?}"
            );
        }
    }
}

/// Keeping `BITWISE` is the machine as it was: the tag every digest folds and
/// the sub-proof count are unchanged. Dropping it sets tag bit 2 and removes
/// exactly one sub-proof.
#[test]
fn keeping_bitwise_keeps_every_tag_and_air_count() {
    for kept in family_masks() {
        assert_eq!(
            kept.as_tag(),
            (kept.keccak as u8) | ((kept.blake3 as u8) << 1),
            "{kept:?}: the tag of a mask that keeps BITWISE must not move"
        );
        let dropped = ChipSet {
            bitwise: false,
            ..kept
        };
        assert_eq!(dropped.as_tag(), kept.as_tag() | 0b100, "{dropped:?}");
        for (rnd, b3) in [(0, 0), (1, 1), (3, 2)] {
            let rnd = kept.keccak_rnd_chunks(rnd.max(1));
            let b3 = kept.blake3_chunks(b3.max(1));
            assert_eq!(
                dropped.num_airs(rnd, b3) + 1,
                kept.num_airs(rnd, b3),
                "{kept:?}"
            );
        }
    }
    assert_eq!(ChipSet::FULL.as_tag(), 0b011);
    assert!(ChipSet::FULL.bitwise);
}

/// The mask follows the program: a program with no byte lookups drops
/// `BITWISE` under R1 and keeps it without; a keccak program keeps it either
/// way; the BLAKE3 socket keeps it for a program with no family at all.
#[test]
fn the_r1_mask_follows_the_program() {
    let trivial = trivial_program();
    let chain = keccak_chain_program();
    for hasher in [HasherKind::Test, HasherKind::Rpx] {
        let off = ChipSet::for_program_under(&trivial, hasher, false);
        let on = ChipSet::for_program_under(&trivial, hasher, true);
        assert!(
            !off.keccak && !off.blake3,
            "the trivial program uses no family"
        );
        assert!(off.bitwise, "R1 off keeps BITWISE");
        assert!(
            !on.bitwise,
            "R1 on drops it from a program that sends it nothing"
        );
        assert_eq!((on.keccak, on.blake3), (off.keccak, off.blake3));
        assert!(ChipSet::for_program_under(&chain, hasher, true).bitwise);
    }
    assert!(ChipSet::for_program_under(&trivial, HasherKind::Blake3, true).bitwise);
    // The process mask is the same rule at the process's own setting.
    assert_eq!(
        ChipSet::for_program_with_hasher(&trivial, HasherKind::Rpx),
        ChipSet::for_program_under(&trivial, HasherKind::Rpx, gap_r1_enabled())
    );
}

/// The census describes the machine the mask proves: with `BITWISE` dropped it
/// has no `BITWISE` row, one entry fewer, and still one entry per sub-proof.
#[test]
fn the_census_follows_the_mask() {
    let program = trivial_program();
    let hasher = HasherKind::Rpx;
    let kept = ChipSet::for_program_under(&program, hasher, false);
    let dropped = ChipSet::for_program_under(&program, hasher, true);
    let with = lfm_chip_census_masked(&program, hasher, kept);
    let without = lfm_chip_census_masked(&program, hasher, dropped);
    let bitwise = LFM_CHIP_NAMES[BITWISE_SLOT];
    assert_eq!(with.iter().filter(|c| c.name == bitwise).count(), 1);
    assert_eq!(without.iter().filter(|c| c.name == bitwise).count(), 0);
    assert_eq!(without.len() + 1, with.len());
    let blake3 = |m: ChipSet| m.blake3_chunks(program.blake3_chunk_count());
    assert_eq!(with.len(), kept.num_airs(0, blake3(kept)));
    assert_eq!(without.len(), dropped.num_airs(0, blake3(dropped)));
    let cells = |census: &[super::airs::LfmChipCells]| {
        census
            .iter()
            .map(|c| c.main_cells() + 3 * c.aux_cells())
            .sum::<u64>()
    };
    let bitwise_cells = with
        .iter()
        .filter(|c| c.name == bitwise)
        .map(|c| c.main_cells() + 3 * c.aux_cells())
        .sum::<u64>();
    assert_eq!(cells(&with) - cells(&without), bitwise_cells);
}

// ============================ proving (box) ================================

fn trivial_arenas() -> Vec<Vec<LfmWord>> {
    vec![
        (0..4u64)
            .map(|i| core::array::from_fn(|j| FE::from(1_000 * (i + 1) + j as u64)))
            .collect(),
    ]
}

fn keccak_arenas() -> Vec<Vec<LfmWord>> {
    let state: [u64; 25] = core::array::from_fn(|i| 7u64.wrapping_mul(i as u64 + 1) ^ 0x5A5A);
    vec![super::keccak_adapter::state_to_words(&state).to_vec()]
}

/// `artifacts` under another mask: the same committed roots and heights, the
/// program id re-derived over the new mask — exactly what the artifact build
/// produces for that mask, since the mask moves no root.
fn under_mask(artifacts: &LfmArtifacts, chip_set: ChipSet) -> LfmArtifacts {
    let mut out = artifacts.clone();
    out.chip_set = chip_set;
    out.program_id = lfm_program_id_chunked(
        &artifacts.roots,
        &artifacts.log_heights,
        artifacts.keccak_rnd_chunks,
        artifacts.hasher,
        chip_set,
        &artifacts.blake3_chunk_roots,
        &artifacts.blake3_chunk_log_heights,
        &artifacts.hash_chunk_roots,
        &artifacts.hash_chunk_log_heights,
    );
    out
}

/// ★ R1 end to end on a program with no byte lookups: it proves and verifies
/// with `BITWISE` dropped, carries one sub-proof fewer, is a different program
/// identity, and neither proof verifies as the other program.
#[test]
fn a_program_without_byte_lookups_proves_and_verifies_without_bitwise() {
    let opts = options();
    let program = trivial_program();
    let full = build_artifacts(&program, &opts);
    let mask = ChipSet::for_program_under(&program, full.hasher, true);
    assert!(!mask.bitwise, "the trivial program sends BITWISE nothing");
    let full = under_mask(
        &full,
        ChipSet {
            bitwise: true,
            ..mask
        },
    );
    let r1 = under_mask(&full, mask);
    assert_ne!(r1.program_id, full.program_id, "the mask is bound");

    let with = lfm_prove(&program, &full, &trivial_arenas(), &opts).expect("proves with BITWISE");
    let without = lfm_prove(&program, &r1, &trivial_arenas(), &opts).expect("proves without it");
    assert_eq!(with.public_words, without.public_words, "same statement");
    assert_eq!(
        MultiProofView::Owned(&without.proof).len() + 1,
        MultiProofView::Owned(&with.proof).len(),
        "one sub-proof fewer"
    );
    assert!(verify_against_artifacts(
        &full,
        &with.proof,
        &with.public_words,
        &opts
    ));
    assert!(verify_against_artifacts(
        &r1,
        &without.proof,
        &without.public_words,
        &opts
    ));
    assert!(
        !verify_against_artifacts(&full, &without.proof, &without.public_words, &opts),
        "a proof without BITWISE is not a proof of the program that keeps it"
    );
    assert!(
        !verify_against_artifacts(&r1, &with.proof, &with.public_words, &opts),
        "and the program that dropped it accepts no proof carrying it"
    );
}

/// ★ The other direction, which is the one that matters: a mask that drops
/// `BITWISE` from a program whose chips DO send it lookups is refused, and so is
/// every proof against it — the keccak family's byte lookups would otherwise
/// have no receiver.
#[test]
fn dropping_bitwise_under_a_byte_lookup_family_is_refused() {
    let opts = options();
    let program = keccak_chain_program();
    let full = build_artifacts(&program, &opts);
    assert!(full.chip_set.keccak && full.chip_set.bitwise);
    let forged = under_mask(
        &full,
        ChipSet {
            bitwise: false,
            ..full.chip_set
        },
    );
    assert!(forged.chip_set.bitwise_required(forged.hasher));

    let honest = lfm_prove(&program, &full, &keccak_arenas(), &opts).expect("proves");
    assert!(verify_against_artifacts(
        &full,
        &honest.proof,
        &honest.public_words,
        &opts
    ));
    assert!(
        !verify_against_artifacts(&forged, &honest.proof, &honest.public_words, &opts),
        "an honest proof does not verify against a mask that drops a needed BITWISE"
    );
    // A proof made under the forged mask — the keccak lookups sent, their
    // receiver absent — is refused too.
    if let Ok(dropped) = lfm_prove(&program, &forged, &keccak_arenas(), &opts) {
        assert!(
            !verify_against_artifacts(&forged, &dropped.proof, &dropped.public_words, &opts),
            "a proof whose byte lookups have no receiver must not verify"
        );
    }
}

// ================================= R2 ======================================
//
// `LFM_HASH` split into two power-of-two instances when one table would be
// mostly padding (`chunking::HashChunking`).

/// The rule on the block's own heights (`thoughts/zf/gap/W2-RECURSION.md`,
/// `CENSUS.md`): split where one table would pad at least `2^15` rows, never
/// where the split saves nothing.
#[test]
fn the_hash_split_rule_on_the_blocks_heights() {
    for (rows, want) in [
        (293_778, Some(1usize << 18)), // the typical STARK wrap: 294,912 for 524,288
        (299_131, Some(1 << 18)),      // STARK wrap 5
        (358_634, Some(1 << 18)),      // STARK wrap 6: 393,216 for 524,288
        (295_573, Some(1 << 18)),      // STARK L2 node
        (173_628, Some(1 << 17)),      // WHIR wrap 0: 196,608 for 262,144
        (412_149, None),               // STARK L1 node: 2^18 + 2^18 is one table
        (224_938, None),               // 2^17 + 2^17 is one table
        (1 << 18, None),               // already full
        (40_000, None),                // would save 24,576 < 2^15 rows
        (70_000, Some(1 << 16)),       // saves 57,344 rows
        (0, None),
        (1, None),
        (3, None),
    ] {
        let got = HashChunking::for_rows(rows);
        let want = want.map_or(HashChunking::unbounded(), HashChunking::split_at);
        assert_eq!(got, want, "{rows} hash rows");
    }
}

/// `chunk_count` and `chunk_range` are one rule: the ranges are disjoint, in
/// order, and cover every row exactly once.
#[test]
fn the_hash_chunk_ranges_cover_every_row_once() {
    for policy in [
        HashChunking::unbounded(),
        HashChunking::split_at(1),
        HashChunking::split_at(4),
        HashChunking::split_at(1 << 18),
    ] {
        for n in [0usize, 1, 3, 4, 5, 9, (1 << 18) + 31_634] {
            let count = policy.chunk_count(n);
            assert!(count == 1 || count == 2, "{policy:?} n={n}");
            let mut next = 0;
            for c in 0..count {
                let r = policy.chunk_range(n, c);
                assert_eq!(r.start, next, "{policy:?} n={n} chunk {c}");
                next = r.end;
            }
            assert_eq!(next, n, "{policy:?} n={n}: every row in one chunk");
            assert!(policy.chunk_range(n, count).is_empty());
        }
    }
}

/// An unsplit `LFM_HASH` keeps the digest it always had; a split one is a
/// different program by name.
#[test]
fn an_unsplit_hash_keeps_the_program_digest() {
    let program = trivial_program();
    let artifacts = build_artifacts(&program, &options());
    assert_eq!(artifacts.hash_chunk_roots, vec![artifacts.roots[HASH_SLOT]]);
    let a = &artifacts;
    let digest = |roots: &[Commitment], heights: &[u8]| {
        lfm_program_id_chunked(
            &a.roots,
            &a.log_heights,
            a.keccak_rnd_chunks,
            a.hasher,
            a.chip_set,
            &a.blake3_chunk_roots,
            &a.blake3_chunk_log_heights,
            roots,
            heights,
        )
    };
    let legacy = lfm_program_id(
        &a.roots,
        &a.log_heights,
        a.keccak_rnd_chunks,
        a.hasher,
        a.chip_set,
        &a.blake3_chunk_roots,
        &a.blake3_chunk_log_heights,
    );
    assert_eq!(a.program_id, legacy);
    assert_eq!(digest(&[], &[]), legacy);
    assert_eq!(
        digest(&a.hash_chunk_roots, &a.hash_chunk_log_heights),
        legacy
    );
    let tail = [a.roots[HASH_SLOT], [7u8; 32]];
    assert_ne!(digest(&tail, &[2, 2]), legacy, "a tail moves the digest");
    assert_ne!(
        digest(&tail, &[2, 2]),
        digest(&tail, &[2, 3]),
        "and the tail's heights are bound"
    );
}

/// A split program describes two `LFM_HASH` tables everywhere its shape is read:
/// the chunk groups, the census, and one more sub-proof.
#[test]
fn a_split_program_describes_two_hash_tables() {
    let whole = trivial_program();
    assert_eq!(
        whole.groups.hash.real_rows, 3,
        "the fixture's three compressions"
    );
    assert_eq!(whole.hash_chunk_count(), 1);
    let split = trivial_program().with_hash_chunking(HashChunking::split_at(2));
    assert_eq!(split.hash_chunk_count(), 2);
    assert_eq!(split.hash_chunk_real_rows(), vec![2, 1]);
    let (g0, g1) = (split.hash_chunk_group(0), split.hash_chunk_group(1));
    let g = &split.groups.hash;
    assert_eq!(g0.data[..2 * g.width], g.data[..2 * g.width]);
    assert_eq!(g1.data[..g.width], g.data[2 * g.width..3 * g.width]);
    assert!(g0.data[2 * g.width..].iter().all(|v| *v == FE::zero()));
    assert!(g1.data[g.width..].iter().all(|v| *v == FE::zero()));

    let hasher = HasherKind::Rpx;
    let mask = ChipSet::for_program_under(&split, hasher, false);
    let one = lfm_chip_census_masked(&whole, hasher, mask);
    let two = lfm_chip_census_masked(&split, hasher, mask);
    let hash = LFM_CHIP_NAMES[HASH_SLOT];
    assert_eq!(one.iter().filter(|c| c.name == hash).count(), 1);
    let rows: Vec<(u64, u64)> = two
        .iter()
        .filter(|c| c.name == hash)
        .map(|c| (c.real_rows, c.rows))
        .collect();
    assert_eq!(rows, vec![(2, 4), (1, 4)]);
    assert_eq!(two.len(), one.len() + 1);
}

// ============================ proving (box) ================================

/// ★ R2 end to end on the host: a split program proves and verifies, carries
/// one sub-proof more, is a different program identity, and neither proof
/// verifies as the other program. A forged tail root, and the single-table
/// door, reject the split proof.
#[test]
fn a_split_hash_proves_and_verifies_as_its_own_program() {
    let opts = options();
    let hasher = HasherKind::Rpx;
    let whole = trivial_program();
    let split = trivial_program().with_hash_chunking(HashChunking::split_at(2));
    let a1 = build_artifacts_with_hasher(&whole, &opts, hasher);
    let a2 = build_artifacts_with_hasher(&split, &opts, hasher);
    assert_eq!(a1.hash_chunk_roots.len(), 1);
    assert_eq!(a2.hash_chunk_roots.len(), 2);
    assert_eq!(a2.hash_chunk_roots[0], a2.roots[HASH_SLOT]);
    assert_ne!(a1.program_id, a2.program_id);

    let p1 = lfm_prove_with_hasher(&whole, &a1, &trivial_arenas(), &opts, hasher).expect("whole");
    let p2 = lfm_prove_with_hasher(&split, &a2, &trivial_arenas(), &opts, hasher).expect("split");
    assert_eq!(p1.public_words, p2.public_words, "same statement");
    assert_eq!(
        MultiProofView::Owned(&p2.proof).len(),
        MultiProofView::Owned(&p1.proof).len() + 1,
        "one sub-proof more"
    );
    assert!(verify_against_artifacts(
        &a1,
        &p1.proof,
        &p1.public_words,
        &opts
    ));
    assert!(verify_against_artifacts(
        &a2,
        &p2.proof,
        &p2.public_words,
        &opts
    ));
    assert!(!verify_against_artifacts(
        &a1,
        &p2.proof,
        &p2.public_words,
        &opts
    ));
    assert!(!verify_against_artifacts(
        &a2,
        &p1.proof,
        &p1.public_words,
        &opts
    ));

    let mut forged = a2.clone();
    forged.hash_chunk_roots[1] = forged.hash_chunk_roots[0];
    assert!(
        !verify_against_artifacts(&forged, &p2.proof, &p2.public_words, &opts),
        "chunk 1 is checked against its own root"
    );
    assert!(
        !verify_against(
            &a2.roots,
            &a2.program_id,
            a2.keccak_rnd_chunks,
            &p2.proof,
            &p2.public_words,
            &opts,
            a2.hasher,
            a2.chip_set,
        ),
        "the single-table door rejects a split proof on the AIR count"
    );
}

/// ★ The in-guest side, which is what a node runs: an emitted verify leg
/// accepts a child whose `LFM_HASH` is split AND whose mask drops `BITWISE`,
/// and rejects the same proof when the leg is emitted with chunk 1's root
/// replaced — the root Phase A absorbs is the chunk's own.
#[test]
fn an_emitted_leg_verifies_a_split_child_without_bitwise() {
    use super::builder::LfmBuilder;
    use super::per_table_aggregator::{declare_leg_arenas, emit_leg};
    use super::per_table_aggregator_tests::{child_arena_words, child_shape, real_child};

    let opts = super::proof::aggregation_wrap_options();
    let hasher = crate::hash_pin::BLOCK_HASHER;
    let program = trivial_program().with_hash_chunking(HashChunking::split_at(2));
    let built = build_artifacts_with_hasher(&program, &opts, hasher);
    let artifacts = under_mask(
        &built,
        ChipSet {
            bitwise: false,
            ..built.chip_set
        },
    );
    assert!(!artifacts.chip_set.bitwise_required(hasher));
    let proved = lfm_prove_with_hasher(&program, &artifacts, &trivial_arenas(), &opts, hasher)
        .expect("the split, BITWISE-free child proves");
    let child = real_child(artifacts, opts.clone(), &proved);
    let words = child_arena_words(&child);

    let leg_program = |shape: &super::per_table_aggregator::ChildShape<'_>| {
        let mut b = LfmBuilder::new().with_wrap_hash(super::edsl::WrapHash::production());
        let arenas = declare_leg_arenas(&mut b, shape);
        let _ = emit_leg(&mut b, shape, &arenas);
        super::compiler::compile(b.finish())
    };

    let shape = child_shape(&child);
    let honest = leg_program(&shape);
    assert!(
        super::executor::execute(&honest, &words, &hasher).is_ok(),
        "the leg must accept the split child"
    );

    // The same proof, the leg emitted against chunk 0's root in chunk 1's place.
    let hash = LFM_CHIP_NAMES[HASH_SLOT];
    let airs = super::airs::LfmAirs::for_artifacts(&child.artifacts, &opts);
    let tail = airs
        .air_refs()
        .iter()
        .enumerate()
        .filter(|(_, air)| air.name() == hash)
        .map(|(i, _)| i)
        .nth(1)
        .expect("a split child has two LFM_HASH sub-proofs");
    let wrong = child.artifacts.hash_chunk_roots[0];
    let mut forged = child_shape(&child);
    forged.tables[tail].precomputed_root = Some(&wrong);
    assert!(
        super::executor::execute(&leg_program(&forged), &words, &hasher).is_err(),
        "a leg bound to the wrong chunk root must reject the proof"
    );
}
