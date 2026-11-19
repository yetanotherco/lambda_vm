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

/// A prove over retired prepared stacks opens each group alone (its revive
/// gets the room the next group's argue would hold beside it) and uploads no
/// group's columns beside another's commit or argue; over held or host stacks
/// the groups open beside the next argue and the next columns go up beside. The bytes are the same
/// (the place tests above, and the stark block's retired-stacks test under
/// both overlaps).
#[test]
fn a_retired_build_opens_each_group_alone() {
    use stark::multilinear_block::PhaseBOverlap;
    let program = super::b3l_wlfm_tests::honest();
    for (place, want) in [
        (StackPlace::Card, PhaseBOverlap::On),
        (StackPlace::Host, PhaseBOverlap::On),
        (StackPlace::CardRetired, PhaseBOverlap::Off),
    ] {
        let build = placed(&program, super::wlfm_tests::HASHER, WlfmFamily::B3, place);
        assert!(
            build.phase_b_overlap() == want,
            "{place:?}: phase B's overlap"
        );
        assert_eq!(
            build.uploads_beside(),
            want == PhaseBOverlap::On,
            "{place:?}: the next group's columns beside a commit and an argue"
        );
    }
}

/// The levels' card budget knob: unset or empty is the default 85 % of the
/// card, a whole percent from 50 to 95 is that share, anything else is
/// refused.
#[test]
fn the_card_budget_knob_reads_a_percent_and_refuses_nonsense() {
    use super::levels::{CARD_BUDGET_ENV, parse_card_budget};
    assert_eq!(CARD_BUDGET_ENV, "LAMBDA_VM_LEVELS_CARD_BUDGET_PCT");
    assert_eq!(super::levels::LEVELS_CARD_BUDGET_DEFAULT_PCT, 85);
    assert_eq!(parse_card_budget(None), Ok(Some(85)));
    assert_eq!(parse_card_budget(Some("")), Ok(Some(85)));
    assert_eq!(parse_card_budget(Some("80")), Ok(Some(80)));
    assert_eq!(parse_card_budget(Some("85")), Ok(Some(85)));
    assert_eq!(parse_card_budget(Some(" 50 ")), Ok(Some(50)));
    assert_eq!(parse_card_budget(Some("95")), Ok(Some(95)));
    for bad in ["49", "96", "100", "0", "85%", "0.85", "-1", "x"] {
        assert!(parse_card_budget(Some(bad)).is_err(), "{bad}");
    }
}

/// Restores this thread's planned-overlap knob and phase-B faults.
struct OverlapRestore;

impl Drop for OverlapRestore {
    fn drop(&mut self) {
        super::wlfm::set_overlap_planned_for_test(None);
        super::wlfm::set_overlap_faults_for_test(Default::default());
    }
}

/// `program`'s build streamed under the planned-overlap knob `on` and phase-B
/// `faults`: its bytes and its groups' stamps; refused proofs and failed
/// verifications are the error.
fn streamed_planned(
    program: &LfmProgram,
    build: &WlfmBuild,
    arenas: &[Vec<super::word::LfmWord>],
    on: bool,
    faults: stark::multilinear_block::OverlapFaults,
) -> Result<(Vec<u8>, Vec<stark::multilinear_block::GroupStamps>), String> {
    super::wlfm::set_overlap_planned_for_test(Some(on));
    super::wlfm::set_overlap_faults_for_test(faults);
    let _ = super::wlfm::take_wlfm_stamps();
    let filled = super::wlfm::execute_and_fill_unmerged(program, arenas, build.artifacts.hasher)?;
    let proof =
        super::wlfm::prove_wlfm_traces_as(build, filled, true, super::wlfm::WlfmProving::Streamed)?;
    verify_wlfm(&build.artifacts, &proof, &proof.public_words)?;
    let groups = super::wlfm::take_wlfm_stamps()
        .map(|s| s.groups)
        .unwrap_or_default();
    Ok((bytes(&proof), groups))
}

/// The level-shaped probe at one stacked polynomial a group (two groups, each
/// with a prepared stack), its build in `place` under `family`.
fn probe_placed(
    family: WlfmFamily,
    place: StackPlace,
) -> (LfmProgram, Vec<Vec<super::word::LfmWord>>, WlfmBuild) {
    let (program, arenas) = super::levels_probe::level_program(4, 256);
    let build = placed(&program, super::wlfm_tests::HASHER, family, place);
    (program, arenas, build)
}

/// ★ Phase B's planned overlap under retired stacks
/// (`LAMBDA_VM_LEVELS_OVERLAP_PLANNED`) moves no byte in any place: under both
/// families, the level-shaped probe (two groups) streamed with its retired
/// build's groups opened beside the next group's argue proves the held build's
/// bytes, and its stamps say every group but the last opened beside (no card:
/// the plan always fits) with none planned out; held and host builds overlap
/// as they always have, and the knob off leaves a retired build serial.
#[test]
fn the_planned_overlap_proves_the_same_bytes_in_every_place() {
    let _restore = OverlapRestore;
    let _fast = test_format::set(Some(2));
    let _groups = test_format::set_group_polys(1);
    for family in [WlfmFamily::P1, WlfmFamily::B3] {
        let (program, arenas, held) = probe_placed(family, StackPlace::Card);
        let (want, _) = streamed_planned(&program, &held, &arenas, false, Default::default())
            .expect("the held build proves");
        for place in [StackPlace::CardRetired, StackPlace::Card, StackPlace::Host] {
            let (_, _, build) = probe_placed(family, place);
            for on in [false, true] {
                let (got, groups) =
                    streamed_planned(&program, &build, &arenas, on, Default::default())
                        .unwrap_or_else(|e| panic!("{family:?} {place:?} {on}: {e}"));
                assert!(
                    got == want,
                    "{family:?} {place:?} planned {on}: the held bytes"
                );
                assert!(groups.len() >= 2, "{family:?}: several groups");
                let beside = groups.iter().filter(|s| s.overlapped).count();
                assert!(
                    groups.iter().all(|s| !s.overlap_planned_out),
                    "{family:?} {place:?} {on}: nothing planned out without a card"
                );
                if place == StackPlace::CardRetired && !on {
                    assert_eq!(beside, 0, "{family:?}: retired and off, every group alone");
                } else {
                    assert_eq!(
                        beside,
                        groups.len() - 1,
                        "{family:?} {place:?} {on}: every group but the last beside"
                    );
                }
            }
        }
    }
}

/// The planned overlap falls back to the same bytes in the W-LFM prover: a
/// transition the plan refuses opens alone; an opening that fails beside an
/// argue is the prove's error, not a panic; a moved prepared column is
/// refused with the overlap on. (The refusal latch over several transitions:
/// stark's `the_planned_overlaps_fallbacks_prove_the_same_bytes`.)
#[test]
fn a_refused_planned_overlap_falls_back_to_the_same_bytes() {
    use stark::multilinear_block::OverlapFaults;
    let _restore = OverlapRestore;
    let _fast = test_format::set(Some(2));
    let _groups = test_format::set_group_polys(1);
    let (program, arenas, build) = probe_placed(WlfmFamily::B3, StackPlace::CardRetired);
    let (want, groups) = streamed_planned(&program, &build, &arenas, true, Default::default())
        .expect("the retired build proves with the planned overlap");
    assert!(
        groups.len() >= 2 && groups[0].overlapped,
        "group 0 opened beside"
    );
    let (got, groups) = streamed_planned(
        &program,
        &build,
        &arenas,
        true,
        OverlapFaults {
            overlap_refused_at: Some(0),
            ..Default::default()
        },
    )
    .expect("proves with the transition planned out");
    assert!(got == want, "a planned-out transition: the same bytes");
    assert!(
        groups[0].overlap_planned_out && !groups[0].overlapped,
        "group 0 opened alone"
    );
    let failed = streamed_planned(
        &program,
        &build,
        &arenas,
        true,
        OverlapFaults {
            open_fails: Some(0),
            ..Default::default()
        },
    );
    assert!(
        failed.is_err(),
        "a failed opening beside an argue is the prove's error"
    );
    let (moved, done) = build.with_prepared_column_moved();
    assert!(done, "the probe has a prepared column");
    assert!(
        streamed_planned(&program, &moved, &arenas, true, Default::default()).is_err(),
        "a moved prepared column is refused with the planned overlap"
    );
}

/// The planned-overlap knob's name; it is on by default, and only `0` turns
/// it off.
#[test]
fn the_overlap_planned_knob_is_named_and_on_by_default() {
    use super::wlfm::{OVERLAP_PLANNED_ENV, overlap_planned};
    use super::wlfm_knobs::knob_word_off;
    assert_eq!(OVERLAP_PLANNED_ENV, "LAMBDA_VM_LEVELS_OVERLAP_PLANNED");
    if std::env::var_os(OVERLAP_PLANNED_ENV).is_none() {
        assert!(overlap_planned());
    }
    assert!(knob_word_off(Some("0")) && !knob_word_off(Some("1")) && !knob_word_off(None));
}

/// The encode split's prepared-bits reading counts every prepared column once,
/// in one bucket, with all its cells: the buckets sum to the stacks' columns
/// and cells.
#[test]
fn the_prepared_bits_reading_counts_every_prepared_column_once() {
    let _fast = test_format::set(Some(2));
    let program = super::b3l_wlfm_tests::honest();
    let build = placed(
        &program,
        super::wlfm_tests::HASHER,
        WlfmFamily::B3,
        StackPlace::Card,
    );
    let [columns, cells] = super::wlfm::prepared_bits_of(&build);
    assert!(
        columns.iter().sum::<u64>() > 0,
        "the fixture has prepared columns"
    );
    assert!(cells.iter().sum::<u64>() >= columns.iter().sum::<u64>());
    assert!(
        columns
            .iter()
            .zip(&cells)
            .all(|(c, n)| (*c == 0) == (*n == 0)),
        "a bucket's cells come with its columns"
    );
}
