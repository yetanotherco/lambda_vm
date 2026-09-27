//! GAP-FIX REC: the recursion-proof shape fixes, behind their `LAMBDA_VM_GAP_*`
//! knobs.
//!
//! R1 — `BITWISE` leaves every program whose instantiated chips send it no
//! lookup ([`ChipSet::bitwise`]). The premise tests below read the chips'
//! interaction lists and the mask arithmetic and prove nothing; the tests that
//! prove (`*_proves_*`, `*_refused`) belong on the box.

use stark::proof::options::ProofOptions;
use stark::proof::view::MultiProofView;

use crate::GoldilocksCubicProofOptions;
use crate::tables::types::FE;

use super::airs::{
    BITWISE_SLOT, ChipSet, LFM_CHIP_NAMES, gap_knob_value, gap_r1_enabled, lfm_chip_census_masked,
};
use super::hash::HasherKind;
use super::programs::{keccak_chain_program, trivial_program};
use super::proof::{lfm_prove, verify_against_artifacts};
use super::registry::{LfmArtifacts, build_artifacts};
use super::statement::lfm_program_id;
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
    out.program_id = lfm_program_id(
        &artifacts.roots,
        &artifacts.log_heights,
        artifacts.keccak_rnd_chunks,
        artifacts.hasher,
        chip_set,
        &artifacts.blake3_chunk_roots,
        &artifacts.blake3_chunk_log_heights,
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
