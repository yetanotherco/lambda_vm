//! Where a W-LFM build commits its prepared stacks ([`StackPlace`]) is the
//! prover's choice and never the proof's: held on the card, committed and let
//! go at once (revived at each group's opening), or on the host, every place
//! proves the same bytes, and a build whose stack columns are not the
//! committed ones is refused in each. Without a card (the laptop) the card
//! places commit on the host's fallback, so these pin the retire and revive
//! logic itself; the card's kernels are the box's.

use super::compiler::LfmProgram;
use super::hash::HasherKind;
use super::levels::parse_stacks;
use super::wlfm::{
    StackPlace, WlfmBuild, WlfmFamily, WlfmLevers, WlfmProof, WlfmRate, build_wlfm_placed,
    prepare_wlfm, prove_wlfm, test_format, verify_wlfm,
};

fn bytes(proof: &WlfmProof) -> Vec<u8> {
    rkyv::to_bytes::<rkyv::rancor::Error>(proof)
        .expect("a W-LFM proof serializes")
        .to_vec()
}

fn placed(
    program: &LfmProgram,
    hasher: HasherKind,
    family: WlfmFamily,
    place: StackPlace,
) -> WlfmBuild {
    build_wlfm_placed(
        program,
        hasher,
        WlfmRate::Two,
        WlfmLevers::with_family(family),
        place,
    )
    .unwrap_or_else(|e| panic!("the {place:?} build: {e}"))
}

/// Every place's build of `program` proves and verifies, with the same
/// identity and the same proof bytes as the held build's.
fn every_place_proves_the_same_bytes(
    program: &LfmProgram,
    hasher: HasherKind,
    family: WlfmFamily,
    arenas: &[Vec<super::word::LfmWord>],
) {
    let held = placed(program, hasher, family, StackPlace::Card);
    assert!(
        !held.artifacts.stacks.is_empty(),
        "the program has prepared stacks"
    );
    let want = prove_wlfm(program, &held, arenas).expect("the held build proves");
    verify_wlfm(&held.artifacts, &want, &want.public_words).expect("and verifies");
    for place in [StackPlace::CardRetired, StackPlace::Host] {
        let build = placed(program, hasher, family, place);
        assert_eq!(build.place(), place);
        assert_eq!(
            build.artifacts.program_id, held.artifacts.program_id,
            "{place:?}: the same identity"
        );
        let proof = prove_wlfm(program, &build, arenas)
            .unwrap_or_else(|e| panic!("the {place:?} build proves: {e}"));
        verify_wlfm(&build.artifacts, &proof, &proof.public_words)
            .unwrap_or_else(|e| panic!("the {place:?} proof verifies: {e}"));
        assert!(
            bytes(&proof) == bytes(&want),
            "{place:?}: the held build's proof bytes"
        );
        // A retired build proves again: each prove revives its own copy.
        if place == StackPlace::CardRetired {
            let again = prove_wlfm(program, &build, arenas).expect("a second prove");
            assert!(bytes(&again) == bytes(&want), "a second prove's bytes");
        }
    }
}

/// ★ The Poseidon1 fixture (every field chip, two hash rows): held, retired
/// and host stacks prove one proof, byte for byte.
#[test]
fn every_stack_place_proves_the_same_bytes_under_poseidon1() {
    let _unground = test_format::set(None);
    let program = super::wlfm_tests::honest();
    every_place_proves_the_same_bytes(
        &program,
        super::wlfm_tests::HASHER,
        WlfmFamily::P1,
        &super::wlfm_tests::arenas(),
    );
}

/// ★ The lean BLAKE3 fixture under the BLAKE3 family (the levels' own): the
/// same, with the lean tables' prepared stacks among them.
#[test]
fn every_stack_place_proves_the_same_bytes_under_blake3() {
    let _fast = test_format::set(Some(2));
    let program = super::b3l_wlfm_tests::honest();
    every_place_proves_the_same_bytes(&program, super::wlfm_tests::HASHER, WlfmFamily::B3, &[]);
}

/// A prover whose prepared stack's columns moved after the commit is refused
/// in every place: a retired stack revived from them no longer matches its
/// kept tree, and a held one opens values the tables did not argue. The
/// unmoved build of the same place is the control.
#[test]
fn a_prepared_column_moved_after_its_commit_is_refused_in_every_place() {
    let _unground = test_format::set(None);
    let program = super::wlfm_tests::honest();
    let arenas = super::wlfm_tests::arenas();
    for place in [StackPlace::CardRetired, StackPlace::Card, StackPlace::Host] {
        let build = placed(&program, super::wlfm_tests::HASHER, WlfmFamily::P1, place);
        let honest = prove_wlfm(&program, &build, &arenas).expect("the control proves");
        verify_wlfm(&build.artifacts, &honest, &honest.public_words).expect("the control verifies");
        let (moved, done) = build.with_prepared_column_moved();
        assert!(done, "the fixture has a prepared column");
        let refused = match prove_wlfm(&program, &moved, &arenas) {
            Err(_) => true,
            Ok(proof) => verify_wlfm(&moved.artifacts, &proof, &proof.public_words).is_err(),
        };
        assert!(refused, "{place:?}: a moved prepared column is refused");
    }
}

/// A build whose prover half was let go (a proved level's) proves no more,
/// and keeps the artifacts a verifier reads.
#[test]
fn a_released_build_proves_no_more_and_keeps_its_artifacts() {
    let _unground = test_format::set(None);
    let program = super::wlfm_tests::honest();
    let arenas = super::wlfm_tests::arenas();
    let mut build = placed(
        &program,
        super::wlfm_tests::HASHER,
        WlfmFamily::P1,
        StackPlace::CardRetired,
    );
    let proof = prove_wlfm(&program, &build, &arenas).expect("proves");
    build.release_prover_half();
    assert!(
        prove_wlfm(&program, &build, &arenas).is_err(),
        "a released build is refused"
    );
    verify_wlfm(&build.artifacts, &proof, &proof.public_words).expect("its artifacts verify");
}

/// The host part of a build counts the cells its commit takes: every
/// stack's columns at their heights, within the stacked slots they fill.
#[test]
fn a_prebuild_counts_its_prepared_cells() {
    let program = super::wlfm_tests::honest();
    let pre = prepare_wlfm(
        &program,
        super::wlfm_tests::HASHER,
        WlfmRate::Two,
        WlfmLevers::with_family(WlfmFamily::P1),
    )
    .expect("prepares");
    let cells = pre.prepared_cells();
    let build = pre.commit(StackPlace::Host).expect("commits");
    let want: u64 = build
        .artifacts
        .stacks
        .iter()
        .map(|s| s.layout.num_polys() as u64 * (1u64 << s.layout.n_stack()))
        .sum();
    assert!(
        cells > 0 && cells <= want,
        "{cells} prepared cells in {want} stacked slots"
    );
}

/// The placement knob's words: `auto` (or unset) leaves it to the room; the
/// three places by name; anything else is an error, never a silent default.
#[test]
fn the_stacks_knob_reads_its_words_and_refuses_nonsense() {
    assert_eq!(parse_stacks(None), Ok(None));
    assert_eq!(parse_stacks(Some("")), Ok(None));
    assert_eq!(parse_stacks(Some(" auto ")), Ok(None));
    assert_eq!(parse_stacks(Some("held")), Ok(Some(StackPlace::Card)));
    assert_eq!(
        parse_stacks(Some("retired")),
        Ok(Some(StackPlace::CardRetired))
    );
    assert_eq!(parse_stacks(Some("host")), Ok(Some(StackPlace::Host)));
    assert!(parse_stacks(Some("card")).is_err());
    assert!(parse_stacks(Some("1")).is_err());
}

/// The chip prefixes a level's host part packs before its commit are the
/// ones its build would pack after it, table for table: the same words.
#[test]
fn a_prebuilds_chip_prefixes_are_the_builds() {
    for (program, family) in [
        (super::wlfm_tests::honest(), WlfmFamily::P1),
        (super::b3l_wlfm_tests::honest(), WlfmFamily::B3),
    ] {
        let levers = WlfmLevers::with_family(family);
        let mut pre = prepare_wlfm(&program, super::wlfm_tests::HASHER, WlfmRate::Two, levers)
            .expect("prepares");
        pre.hold_chip_prefixes().expect("packs before the commit");
        let early = pre.commit(StackPlace::CardRetired).expect("commits");
        let mut late = placed(
            &program,
            super::wlfm_tests::HASHER,
            family,
            StackPlace::Host,
        );
        late.hold_chip_prefixes().expect("packs after the commit");
        assert!(
            early.chip_prefixes().iter().any(Option::is_some),
            "{family:?}: the fixture has chip tables"
        );
        assert!(
            early.chip_prefixes() == late.chip_prefixes(),
            "{family:?}: the same prefixes"
        );
    }
}

/// The levels' card pad knob: unset or empty is no pad, a whole number of
/// MiB is that many bytes, anything else is refused.
#[test]
fn the_card_pad_knob_reads_mib_and_refuses_nonsense() {
    use super::levels::parse_card_pad;
    assert_eq!(parse_card_pad(None), Ok(0));
    assert_eq!(parse_card_pad(Some(" ")), Ok(0));
    assert_eq!(parse_card_pad(Some("0")), Ok(0));
    assert_eq!(parse_card_pad(Some("3072")), Ok(3072u64 << 20));
    assert!(parse_card_pad(Some("3G")).is_err());
    assert!(parse_card_pad(Some("-1")).is_err());
}
