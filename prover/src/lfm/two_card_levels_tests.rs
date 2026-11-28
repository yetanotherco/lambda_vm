//! The recursion levels' phase B over two cards (`LAMBDA_VM_TWO_CARD_LEVELS`):
//! a level's groups split over two drivers, each group on its own fork of the
//! post-roots transcript and its share put back by group index, so the level
//! proof is the same.
//!
//! Host tests (no card): a small level proved streamed with its phase B split
//! over two drivers on the host is the bytes of one driver, verifies, and put
//! groups on the second driver; a root of a group the second driver proved,
//! swapped for another group's, is refused; with the knob off nothing splits.
//! Card tests (`#[ignore]`, run alone): the same small level split onto the
//! second card (card 1 with two cards, a second backend of card 0 forced with
//! one), a root of a group proved there swapped, is refused. The driver's
//! byte identities on a real BLAKE3 block, one card and two, are in
//! `levels_tests`.

use super::wlfm::{
    StackPlace, WlfmBuild, WlfmFamily, WlfmLevers, WlfmProof, WlfmProving, WlfmRate,
    build_wlfm_placed, execute_and_fill_unmerged, prove_wlfm_traces_as,
    set_levels_split_on_host_for_test, take_wlfm_stamps, test_format, verify_wlfm,
};
use crate::lfm::HasherKind;

/// Restores this thread's levels' split on the host when dropped.
struct OnHost;

impl Drop for OnHost {
    fn drop(&mut self) {
        set_levels_split_on_host_for_test(false);
    }
}

/// The level-shaped probe program (lean compressions over hinted leaves and
/// paths, extension multiply-adds; one stacked polynomial a group, so several
/// groups; its prepared stacks retired), proved streamed (the levels' path) with its phase B split over
/// two drivers on the host or not; the build, the proof, verified, and each
/// group's (card, polynomials).
fn proved(split_on_host: bool) -> (WlfmBuild, WlfmProof, Vec<(u8, usize)>) {
    let hasher = HasherKind::Poseidon1W16;
    let (level, arenas) = super::levels_probe::level_program(4, 256);
    // Its prepared stacks retired, as level 1's are on a real block: a held
    // stack pins its group to card 0, a retired one does not.
    let built = build_wlfm_placed(
        &level,
        hasher,
        WlfmRate::Two,
        WlfmLevers::with_family(WlfmFamily::B3),
        StackPlace::CardRetired,
    )
    .expect("the W-LFM build");
    let filled = execute_and_fill_unmerged(&level, &arenas, hasher).expect("executes");
    let _on_host = OnHost;
    set_levels_split_on_host_for_test(split_on_host);
    let _ = take_wlfm_stamps();
    let proof = prove_wlfm_traces_as(&built, filled, true, WlfmProving::Streamed).expect("proves");
    let groups: Vec<(u8, usize)> = take_wlfm_stamps()
        .expect("the prove's stamps")
        .groups
        .iter()
        .map(|g| (g.card, g.polys))
        .collect();
    verify_wlfm(&built.artifacts, &proof, &proof.public_words).expect("verifies");
    (built, proof, groups)
}

fn cards(groups: &[(u8, usize)]) -> Vec<u8> {
    groups.iter().map(|g| g.0).collect()
}

fn bytes(proof: &WlfmProof) -> Vec<u8> {
    rkyv::to_bytes::<rkyv::rancor::Error>(proof)
        .expect("serializes")
        .to_vec()
}

/// Each group's first root in the proof's roots block: the block holds every
/// group's round-0 trees in group order.
fn first_roots(built: &WlfmBuild, proof: &WlfmProof, groups: &[(u8, usize)]) -> Vec<usize> {
    let mut at = 0usize;
    let mut first = Vec::with_capacity(groups.len());
    for &(_, polys) in groups {
        first.push(at);
        at += multilinear::stacked_eval::round0_trees(polys, &built.artifacts.config).len();
    }
    assert_eq!(at, proof.proof.roots.len(), "the roots block's length");
    first
}

/// With the levels' split on over two drivers on the host, a level proves the
/// bytes of one driver and verifies, and the second driver proved some of its
/// groups; with it off, every group is card 0's.
#[test]
fn a_level_split_over_two_drivers_proves_the_same_bytes() {
    let _fast = test_format::set(Some(2));
    // One stacked polynomial a group: a level of several groups to split.
    let _groups = test_format::set_group_polys(1);
    // The second driver's count is the process's (a test beside this one may
    // split too): off is read from the stamps, on from the count's growth.
    let (_, one, one_groups) = proved(false);
    let before = stark::multilinear_block::split_second_groups();
    let one_cards = cards(&one_groups);
    assert!(
        one_cards.iter().all(|&c| c == 0),
        "off: every group on card 0, read {one_cards:?}"
    );
    let (_, two, two_groups) = proved(true);
    let two_cards = cards(&two_groups);
    assert!(two_cards.len() >= 2, "a level of several groups");
    assert!(
        two_cards.contains(&1),
        "the second driver proved a group: {two_cards:?}"
    );
    assert!(
        stark::multilinear_block::split_second_groups() > before,
        "the split counted its second driver's groups"
    );
    assert_eq!(bytes(&two), bytes(&one), "the split moved a byte");
}

/// A level proved over two drivers, one root of a group the second driver
/// proved swapped for another group's root, is refused by the level's
/// verifier.
#[test]
fn a_level_group_from_the_second_driver_with_its_root_swapped_is_refused() {
    let _fast = test_format::set(Some(2));
    // One stacked polynomial a group: a level of several groups to split.
    let _groups = test_format::set_group_polys(1);
    let (built, two, groups) = proved(true);
    let cards = cards(&groups);
    let first = first_roots(&built, &two, &groups);
    let roots = &two.proof.roots;
    let g = (0..cards.len())
        .find(|&g| cards[g] == 1)
        .expect("a group on the second driver");
    let other = (0..cards.len())
        .find(|&h| roots[first[h]] != roots[first[g]])
        .expect("another group's root");
    let mut swapped = two.clone();
    swapped.proof.roots[first[g]] = roots[first[other]];
    assert!(
        verify_wlfm(&built.artifacts, &swapped, &swapped.public_words).is_err(),
        "group {g}'s root swapped for group {other}'s was accepted"
    );
}

/// ★ Card, run alone. The small level split onto the second card under the
/// levels' knob (`auto` on card 1 with two cards, forced on a second backend
/// of card 0 with one) proves and verifies, a group was proved there, and one
/// root of such a group swapped for another group's is refused.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "card: run alone"]
fn a_level_proof_split_onto_the_second_card_with_a_root_swapped_is_refused() {
    use math_cuda::device::{
        TwoCardSplit, card_count, set_two_card_levels_for_test, set_two_card_split_for_test,
    };
    struct Restore;
    impl Drop for Restore {
        fn drop(&mut self) {
            set_two_card_levels_for_test(None);
            set_two_card_split_for_test(None);
        }
    }
    let _restore = Restore;
    assert!(multilinear::gpu::reserve_budget() > 0, "needs a card");
    let _fast = test_format::set(Some(2));
    // One stacked polynomial a group: a level of several groups to split.
    let _groups = test_format::set_group_polys(1);
    set_two_card_split_for_test(Some(TwoCardSplit::Off));
    set_two_card_levels_for_test(Some(if card_count() >= 2 {
        TwoCardSplit::Auto
    } else {
        TwoCardSplit::Forced
    }));
    assert!(
        multilinear::gpu::two_card_levels_on(),
        "the levels' split on"
    );
    assert!(
        !multilinear::gpu::two_card_split_on(),
        "the base's split stays off"
    );
    assert!(
        multilinear::gpu::two_card_split_ready(),
        "the second card ready"
    );
    let (built, two, groups) = proved(false);
    let cards = cards(&groups);
    println!("TWO-CARD LEVELS on the second card: groups by card {cards:?}");
    let first = first_roots(&built, &two, &groups);
    let roots = &two.proof.roots;
    let g = (0..cards.len())
        .find(|&g| cards[g] == 1)
        .expect("a group on the second card");
    let other = (0..cards.len())
        .find(|&h| roots[first[h]] != roots[first[g]])
        .expect("another group's root");
    let mut swapped = two.clone();
    swapped.proof.roots[first[g]] = roots[first[other]];
    assert!(
        verify_wlfm(&built.artifacts, &swapped, &swapped.public_words).is_err(),
        "group {g}'s root swapped for group {other}'s was accepted"
    );
}

/// The probe level proved streamed with its phase B split over two drivers on
/// the host, its stacks committed at `place`, under `faults`: the prove's
/// answer (no verification).
fn prove_split_on_host(
    place: StackPlace,
    faults: stark::multilinear_block::OverlapFaults,
) -> Result<WlfmProof, String> {
    struct Restore;
    impl Drop for Restore {
        fn drop(&mut self) {
            super::wlfm::set_overlap_faults_for_test(
                stark::multilinear_block::OverlapFaults::default(),
            );
            set_levels_split_on_host_for_test(false);
        }
    }
    let hasher = HasherKind::Poseidon1W16;
    let (level, arenas) = super::levels_probe::level_program(4, 256);
    let built = build_wlfm_placed(
        &level,
        hasher,
        WlfmRate::Two,
        WlfmLevers::with_family(WlfmFamily::B3),
        place,
    )
    .expect("the W-LFM build");
    let filled = execute_and_fill_unmerged(&level, &arenas, hasher).expect("executes");
    let _restore = Restore;
    set_levels_split_on_host_for_test(true);
    super::wlfm::set_overlap_faults_for_test(faults);
    prove_wlfm_traces_as(&built, filled, true, WlfmProving::Streamed)
}

/// A failed argue and a failed opening at any place of a level split over two
/// drivers are the level prove's error: never a panic, never a proof. (The
/// probe level's groups are one a driver, each opened at its driver's end, not
/// on an opening thread, so a panicking opening thread is not reached here;
/// `phase_b_split_tests` covers it on the drivers both paths share.)
#[test]
fn a_fault_on_either_driver_of_a_split_level_is_the_proves_error() {
    use stark::multilinear_block::OverlapFaults;
    let _fast = test_format::set(Some(2));
    let _groups = test_format::set_group_polys(1);
    let (_, _, groups) = proved(true);
    let n = groups.len();
    assert!(
        cards(&groups).contains(&1),
        "the unfaulted level split: {groups:?}"
    );
    for place in 0..n {
        for (what, faults) in [
            (
                "argue",
                OverlapFaults {
                    argue_fails: Some(place),
                    ..OverlapFaults::default()
                },
            ),
            (
                "opening",
                OverlapFaults {
                    open_fails: Some(place),
                    ..OverlapFaults::default()
                },
            ),
        ] {
            let proved =
                std::panic::catch_unwind(|| prove_split_on_host(StackPlace::CardRetired, faults));
            match proved {
                Ok(Err(_)) => {}
                Ok(Ok(_)) => panic!("{what} failing at place {place}: the level proved"),
                Err(_) => panic!("{what} failing at place {place}: the prove panicked"),
            }
        }
    }
}

/// A level whose prepared stacks are held (level 2's on a real block) keeps
/// every group on card 0 under the split: a held stack pins its group, so no
/// two groups are free and the level proves on one driver, the same bytes.
#[test]
fn a_level_with_its_stacks_held_stays_on_card_0_under_the_split() {
    let _fast = test_format::set(Some(2));
    let _groups = test_format::set_group_polys(1);
    let hasher = HasherKind::Poseidon1W16;
    let (level, arenas) = super::levels_probe::level_program(4, 256);
    let built = build_wlfm_placed(
        &level,
        hasher,
        WlfmRate::Two,
        WlfmLevers::with_family(WlfmFamily::B3),
        StackPlace::Card,
    )
    .expect("the W-LFM build");
    let prove = |split: bool| {
        let filled = execute_and_fill_unmerged(&level, &arenas, hasher).expect("executes");
        let _on_host = OnHost;
        set_levels_split_on_host_for_test(split);
        let _ = take_wlfm_stamps();
        let proof =
            prove_wlfm_traces_as(&built, filled, true, WlfmProving::Streamed).expect("proves");
        let groups: Vec<(u8, usize)> = take_wlfm_stamps()
            .expect("the prove's stamps")
            .groups
            .iter()
            .map(|g| (g.card, g.polys))
            .collect();
        verify_wlfm(&built.artifacts, &proof, &proof.public_words).expect("verifies");
        (proof, groups)
    };
    let (one, _) = prove(false);
    let (split, groups) = prove(true);
    assert!(groups.len() >= 2, "a level of several groups");
    assert!(
        cards(&groups).iter().all(|&c| c == 0),
        "held stacks: every group on card 0, read {:?}",
        cards(&groups)
    );
    assert_eq!(bytes(&split), bytes(&one), "held: the same bytes");
}

/// The probe level's stacks held (`StackPlace::Card`), each group's stack
/// committed on the card `assign` picks from the groups' prices
/// ([`super::wlfm::WlfmPrebuild::commit_on`]; `None`: today's commit), and
/// proved streamed with phase B split over two drivers on the host or not:
/// the build, the cards the stacks were given, the prove's answer and each
/// group's (card, polynomials).
#[allow(clippy::type_complexity)]
fn held_by_card(
    assign: Option<&dyn Fn(&[u64]) -> Vec<u8>>,
    split_on_host: bool,
) -> (
    WlfmBuild,
    Option<Vec<u8>>,
    Result<WlfmProof, String>,
    Vec<(u8, usize)>,
) {
    let hasher = HasherKind::Poseidon1W16;
    let (level, arenas) = super::levels_probe::level_program(4, 256);
    let pre = super::wlfm::prepare_wlfm(
        &level,
        hasher,
        WlfmRate::Two,
        WlfmLevers::with_family(WlfmFamily::B3),
    )
    .expect("the W-LFM host part");
    let cards = assign.map(|f| f(&pre.group_prices()));
    let built = pre
        .commit_on(StackPlace::Card, cards.clone())
        .expect("the W-LFM build");
    let filled = execute_and_fill_unmerged(&level, &arenas, hasher).expect("executes");
    let _on_host = OnHost;
    set_levels_split_on_host_for_test(split_on_host);
    let _ = take_wlfm_stamps();
    let proof = prove_wlfm_traces_as(&built, filled, true, WlfmProving::Streamed);
    let groups: Vec<(u8, usize)> = take_wlfm_stamps()
        .map(|s| s.groups.iter().map(|g| (g.card, g.polys)).collect())
        .unwrap_or_default();
    (built, cards, proof, groups)
}

/// The driver's assignment: longest first over the groups' prices.
fn by_price(prices: &[u64]) -> Vec<u8> {
    stark::phase_b_split::assign_cards(prices, &[])
}

/// Every group's stack on card 1.
fn all_on_card_1(prices: &[u64]) -> Vec<u8> {
    vec![1; prices.len()]
}

/// ★ With each held stack committed on its group's card
/// (`LAMBDA_VM_LEVELS_STACKS_BY_CARD`), a level whose stacks are all held
/// splits over two drivers where it otherwise stays on card 0: every group
/// with its stack on card 1 is argued by the second driver and every other by
/// the first, the program's identity (hashed over the stacks in group order)
/// and the proof's bytes are the one-card level's, and the proof verifies.
/// Every stack on card 1 too: the first driver then argues none.
#[test]
fn stacks_held_by_card_split_the_level_with_the_same_bytes() {
    let _fast = test_format::set(Some(2));
    let _groups = test_format::set_group_polys(1);
    let (one_built, _, one, one_groups) = held_by_card(None, true);
    let one = one.expect("the held level proves");
    assert!(
        cards(&one_groups).iter().all(|&c| c == 0),
        "held, today: every group on card 0"
    );
    for assign in [&by_price as &dyn Fn(&[u64]) -> Vec<u8>, &all_on_card_1] {
        let (built, stacks, two, groups) = held_by_card(Some(assign), true);
        let stacks = stacks.expect("cards given");
        let two = two.expect("the level held by card proves");
        assert!(stacks.contains(&1), "a stack on card 1: {stacks:?}");
        assert_eq!(
            cards(&groups),
            stacks,
            "each group argued on its stack's card"
        );
        assert_eq!(
            built.artifacts.program_id, one_built.artifacts.program_id,
            "the program's identity over the stacks in group order"
        );
        verify_wlfm(&built.artifacts, &two, &two.public_words).expect("verifies");
        assert_eq!(bytes(&two), bytes(&one), "the stacks by card moved a byte");
    }
}

/// A held stack on card 1 whose level is not split (the split not asked, as
/// when the second card was marked failed after the commit) is a clean error
/// of the level's prove, never a panic or a proof.
#[test]
fn a_stack_held_on_card_1_without_the_split_is_a_clean_error() {
    let _fast = test_format::set(Some(2));
    let _groups = test_format::set_group_polys(1);
    let proved = std::panic::catch_unwind(|| held_by_card(Some(&by_price), false).2);
    match proved {
        Ok(Err(e)) => assert!(e.contains("second card"), "the error says why: {e}"),
        Ok(Ok(_)) => panic!("a stack held on card 1 proved without the split"),
        Err(_) => panic!("the prove panicked"),
    }
}

/// The second card failing in its stack commits (a test's fault before its
/// first stack, and before its second, after one was made there; every stack
/// given to card 1): every stack is committed on card 0 (the one made there
/// turned back into its columns first), and the level proves the one-card
/// level's bytes with every group on card 0.
#[test]
fn a_second_card_fault_in_the_stack_commits_commits_them_on_card_0() {
    use super::wlfm::SecondStackFault;
    struct Restore;
    impl Drop for Restore {
        fn drop(&mut self) {
            super::wlfm::set_second_stack_fault_for_test(None);
        }
    }
    let _restore = Restore;
    let _fast = test_format::set(Some(2));
    let _groups = test_format::set_group_polys(1);
    let (one_built, _, one, _) = held_by_card(None, true);
    let one = one.expect("the held level proves");
    for at in [0usize, 1] {
        super::wlfm::set_second_stack_fault_for_test(Some((at, SecondStackFault::BeforeCommit)));
        let (built, stacks, proof, groups) = held_by_card(Some(&all_on_card_1), true);
        super::wlfm::set_second_stack_fault_for_test(None);
        assert!(
            stacks.is_some_and(|s| s.len() > at),
            "fault at {at}: a stack there to fault"
        );
        let proof = proof.unwrap_or_else(|e| panic!("fault at {at}: the level proves: {e}"));
        assert!(
            cards(&groups).iter().all(|&c| c == 0),
            "fault at {at}: every stack on card 0, so every group: {groups:?}"
        );
        assert_eq!(
            built.artifacts.program_id, one_built.artifacts.program_id,
            "fault at {at}: the same roots"
        );
        assert_eq!(bytes(&proof), bytes(&one), "fault at {at}: the same bytes");
    }
}

#[cfg(feature = "test-cuda-faults")]
fn assert_cuda_stack_fault_fallback(
    fault: multilinear::whir_commit::DeviceStackFault,
    place: StackPlace,
) {
    use super::wlfm::{SecondStackFault, prepare_wlfm, set_second_stack_fault_for_test};
    use math_cuda::device::{
        TwoCardSplit, backend, card_count, second_backend_started, set_two_card_levels_for_test,
        set_two_card_split_for_test, two_card_split_ready,
    };

    struct Restore;
    impl Drop for Restore {
        fn drop(&mut self) {
            set_second_stack_fault_for_test(None);
            set_two_card_levels_for_test(None);
            set_two_card_split_for_test(None);
        }
    }

    let _restore = Restore;
    let _fast = test_format::set(Some(2));
    let _stack = test_format::set_stack_log(15);
    let _groups = test_format::set_group_polys(1);
    let _chunks = test_format::set_chunk_log(13);
    set_two_card_split_for_test(Some(TwoCardSplit::Off));
    set_two_card_levels_for_test(Some(if card_count() == 1 {
        TwoCardSplit::Forced
    } else {
        TwoCardSplit::Auto
    }));
    assert!(two_card_split_ready());
    let main = backend().expect("the main CUDA backend");
    let second = second_backend_started().expect("the second CUDA backend");
    let before = (main.reserved_bytes(), second.reserved_bytes());

    let hasher = HasherKind::Poseidon1W16;
    let (level, arenas) = super::levels_probe::level_program(4, 256);
    let prepare = || {
        prepare_wlfm(
            &level,
            hasher,
            WlfmRate::Two,
            WlfmLevers {
                first_fold: Some(4),
                ..WlfmLevers::with_family(WlfmFamily::B3)
            },
        )
        .expect("the level prepares")
    };
    let prove = |build: &WlfmBuild| {
        let filled = execute_and_fill_unmerged(&level, &arenas, hasher).expect("executes");
        let proof =
            prove_wlfm_traces_as(build, filled, true, WlfmProving::Streamed).expect("proves");
        let stamps = take_wlfm_stamps().expect("the proof's group stamps");
        assert!(stamps.groups.iter().all(|g| g.card == 0));
        verify_wlfm(&build.artifacts, &proof, &proof.public_words).expect("verifies");
        proof
    };
    let pre = prepare();
    let cells = pre.prepared_cells_by_group();
    let groups: Vec<_> = (0..cells.len()).filter(|&g| cells[g] != 0).collect();
    // Leave a card-0 stack between each pair of card-1 stacks. Fault the
    // middle card-1 stack, whose codeword must be large enough for CUDA.
    let target = (2..groups.len().saturating_sub(2))
        .find(|&i| cells[groups[i]] >= 1 << 15)
        .unwrap_or_else(|| {
            panic!("need five prepared stacks with a CUDA stack in between: {cells:?}")
        });
    let mut assigned = vec![0; cells.len()];
    for i in [target - 2, target, target + 2] {
        assigned[groups[i]] = 1;
    }

    set_second_stack_fault_for_test(Some((1, SecondStackFault::Device(fault))));
    let recovered = pre
        .commit_on(place, Some(assigned))
        .expect("card 0 recovers the failed stack");
    set_second_stack_fault_for_test(None);
    assert!(!two_card_split_ready(), "the failed backend was disabled");
    assert_eq!(
        second.reserved_bytes(),
        before.1,
        "the second backend gave back its stack reservations"
    );
    let actual = prove(&recovered);
    let program_id = recovered.artifacts.program_id;
    drop(recovered);
    let reference = prepare().commit(place).expect("card 0 commits every stack");
    assert_eq!(
        reference.artifacts.program_id, program_id,
        "the prepared roots in stack order"
    );
    let expected = prove(&reference);
    drop(reference);
    let actual = bytes(&actual);
    assert_eq!(actual, bytes(&expected));
    assert_eq!(
        (main.reserved_bytes(), second.reserved_bytes()),
        before,
        "both ledgers after proving"
    );
    println!(
        "CUDA STACK FALLBACK: {} B equal, ledger {before:?}",
        actual.len()
    );
}

#[cfg(feature = "test-cuda-faults")]
#[test]
#[ignore = "card: run alone in a fresh process"]
fn a_cuda_stack_commit_fault_replays_interleaved_stacks_on_card_0() {
    assert_cuda_stack_fault_fallback(
        multilinear::whir_commit::DeviceStackFault::Commit,
        StackPlace::Card,
    );
}

#[cfg(feature = "test-cuda-faults")]
#[test]
#[ignore = "card: run alone in a fresh process"]
fn a_cuda_stack_retirement_fault_replays_interleaved_stacks_on_card_0() {
    assert_cuda_stack_fault_fallback(
        multilinear::whir_commit::DeviceStackFault::Retire,
        StackPlace::CardRetired,
    );
}

/// Restores this thread's levels' phase-A lanes when dropped.
struct LanesOff;

impl Drop for LanesOff {
    fn drop(&mut self) {
        super::levels::set_commits_by_card_on_thread_for_test(None);
    }
}

/// The probe level, its prepared stacks at `place`, proved streamed with its
/// phase A on `lanes` (one lane for `None`): the proof, verified, and each
/// group's (commit lane, phase-B card).
fn proved_on_lanes(
    place: StackPlace,
    lanes: Option<stark::multilinear_block::PhaseALanes>,
) -> (WlfmProof, Vec<(u8, u8)>) {
    let hasher = HasherKind::Poseidon1W16;
    let (level, arenas) = super::levels_probe::level_program(4, 256);
    let built = build_wlfm_placed(
        &level,
        hasher,
        WlfmRate::Two,
        WlfmLevers::with_family(WlfmFamily::B3),
        place,
    )
    .expect("the W-LFM build");
    let filled = execute_and_fill_unmerged(&level, &arenas, hasher).expect("executes");
    let _off = LanesOff;
    super::levels::set_commits_by_card_on_thread_for_test(lanes);
    let _ = take_wlfm_stamps();
    let proof = prove_wlfm_traces_as(&built, filled, true, WlfmProving::Streamed).expect("proves");
    let groups = take_wlfm_stamps()
        .expect("the prove's stamps")
        .groups
        .iter()
        .map(|g| (g.commit_card, g.card))
        .collect();
    verify_wlfm(&built.artifacts, &proof, &proof.public_words).expect("verifies");
    (proof, groups)
}

/// ★ A level's phase A on two commit lanes (on the host) proves the bytes of
/// one lane, its prepared stacks retired (level 1's on a real block: no
/// upload beside a commit) or held (level 2's: the next group uploaded
/// beside each commit): every group on the lane the assignment gave it, each
/// lane taking the next group when free, and the second lane failing before
/// its first commit (its groups then committed on card 0).
#[test]
fn a_level_committed_on_two_lanes_proves_the_same_bytes() {
    use stark::multilinear_block::{LanePolicy, PhaseALanes};
    let _fast = test_format::set(Some(2));
    // One stacked polynomial a group: a level of several groups.
    let _groups = test_format::set_group_polys(1);
    for place in [StackPlace::CardRetired, StackPlace::Card] {
        let (one, one_groups) = proved_on_lanes(place, None);
        let n = one_groups.len();
        assert!(n >= 2, "{place:?}: a level of several groups");
        assert!(
            one_groups.iter().all(|g| g.0 == 0),
            "{place:?}: one lane commits on card 0"
        );
        let alternating: Vec<u8> = (0..n).map(|g| (g % 2) as u8).collect();
        let on_host = |policy: LanePolicy, fault: Option<usize>| PhaseALanes {
            knob: None,
            policy,
            on_host: true,
            fault,
        };
        for (what, lanes, want) in [
            (
                "alternating",
                on_host(LanePolicy::Fixed(alternating.clone()), None),
                Some(alternating.clone()),
            ),
            ("next free", on_host(LanePolicy::NextFree, None), None),
            (
                "the second lane failing",
                on_host(LanePolicy::Fixed(alternating.clone()), Some(0)),
                Some(vec![0; n]),
            ),
        ] {
            let (two, groups) = proved_on_lanes(place, Some(lanes));
            assert_eq!(bytes(&two), bytes(&one), "{place:?} {what}: a byte moved");
            let commit_cards: Vec<u8> = groups.iter().map(|g| g.0).collect();
            if let Some(want) = want {
                assert_eq!(commit_cards, want, "{place:?} {what}: each group's lane");
            }
        }
    }
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "card: run alone in a fresh process with deterministic grind"]
fn a_phase_a_fault_recovers_held_stacks_from_the_second_card() {
    use math_cuda::device::{
        TwoCardSplit, backend, second_backend_started, set_two_card_levels_for_test,
        set_two_card_split_for_test,
    };
    use stark::multilinear_block::{PhaseALanes, TwoCardKnob};

    struct Restore;
    impl Drop for Restore {
        fn drop(&mut self) {
            set_two_card_levels_for_test(None);
            set_two_card_split_for_test(None);
        }
    }
    let _restore = Restore;
    let _lanes = LanesOff;
    let _fast = test_format::set(Some(2));
    let _groups = test_format::set_group_polys(1);
    assert!(crypto::grinding::deterministic());
    set_two_card_split_for_test(Some(TwoCardSplit::Off));
    set_two_card_levels_for_test(Some(TwoCardSplit::Forced));
    assert!(multilinear::gpu::two_card_split_ready());
    let main = backend().expect("the main CUDA backend");
    let second = second_backend_started().expect("the second CUDA backend");
    let ledger = || (main.reserved_bytes(), second.reserved_bytes());
    let before = ledger();
    let declines = multilinear::gpu::cross_card_declines();

    let (build, _, expected, _) = held_by_card(None, false);
    let program_id = build.artifacts.program_id;
    let expected = expected.expect("one card proves");
    drop(build);
    super::levels::set_commits_by_card_on_thread_for_test(Some(PhaseALanes {
        knob: Some(TwoCardKnob::Levels),
        fault: Some(1),
        ..PhaseALanes::default()
    }));
    let (build, assignment, recovered, groups) = held_by_card(Some(&all_on_card_1), false);
    assert!(assignment.is_some_and(|cards| cards.len() > 1));
    assert!(!multilinear::gpu::two_card_split_ready());
    assert_eq!(build.artifacts.program_id, program_id);
    assert!(groups.iter().all(|g| g.0 == 0));
    let recovered = recovered.expect("card 0 recovers the phase-A groups and prepared stacks");
    verify_wlfm(&build.artifacts, &recovered, &recovered.public_words).expect("verifies");
    drop(build);
    let recovered = bytes(&recovered);
    assert_eq!(recovered, bytes(&expected));
    assert_eq!(ledger(), before);
    assert_eq!(multilinear::gpu::cross_card_declines(), declines);
    println!(
        "PHASE A STACK RECOVERY: {} B equal, ledger {before:?}",
        recovered.len()
    );
}
