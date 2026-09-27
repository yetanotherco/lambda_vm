//! What a stacked group promises the card, end to end on the device.
//!
//! ```text
//! cargo test --release -p multilinear --features cuda --test whir_room_device -- --test-threads=1
//! ```
//!
//! Needs a GPU. Every arm is asserted by the counter of the path it is about,
//! in both directions, so a card that declined could not turn a comparison
//! into one of a path with itself. One `#[test]`: the arms switch process-wide
//! overrides and read the process-wide reservation total, and a parallel test
//! would move both.
#![cfg(feature = "cuda")]

use crypto::fiat_shamir::default_transcript::DefaultTranscript;
use math::field::element::FieldElement;
use math::field::extensions_goldilocks::Degree3GoldilocksExtensionField as Ext;
use math::field::goldilocks::GoldilocksField as F;
use multilinear::gpu;
use multilinear::mle::Mle;
use multilinear::stacked_eval::{self, Claimed, StackedCommitment, StackedProof};
use multilinear::stacking::StackedLayout;
use multilinear::whir_chain::{
    CapPolicy, ChainConfig, ChainFormat, FirstFold, GrindBits, WhirFolds,
};
use multilinear::whir_hash::{RpxWhir, WhirHash};

type FE = FieldElement<F>;
type EE = FieldElement<Ext>;
type H = RpxWhir;

fn column(num_vars: usize, seed: u64) -> Mle<F> {
    Mle::new(
        (0..(1u64 << num_vars))
            .map(|i| {
                FE::from(
                    i.wrapping_add(seed)
                        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                        .wrapping_add(seed.wrapping_mul(6364136223846793005))
                        >> 11,
                )
            })
            .collect(),
    )
    .unwrap()
}

/// The production shape of a chain — first6 then uniform4, caps on every tree
/// — without the grind, whose nonce is not reproducible across two searches.
fn config() -> ChainConfig {
    ChainConfig {
        log_blowup: 2,
        log_folding: 4,
        num_queries: 20,
        grind: GrindBits::default(),
        format: ChainFormat {
            cap: CapPolicy::Auto,
            folds: WhirFolds::First(FirstFold::new(6).unwrap()),
        },
    }
}

/// Columns of mixed heights, each claimed at its own point, spilling over
/// `polys` stacked polynomials of `2^n_stack` — above the device commit and
/// sumcheck thresholds.
struct Case {
    layout: StackedLayout,
    columns: Vec<Mle<F>>,
    points: Vec<Vec<EE>>,
    values: Vec<EE>,
}

fn case(n_stack: usize, polys: usize) -> Case {
    // Each polynomial is filled by a column of `n_stack − 1` variables and the
    // halves below it, so the layout is `polys` full stacks.
    let mut heights = Vec::new();
    for _ in 0..polys {
        heights.extend([n_stack - 1, n_stack - 2, n_stack - 3, n_stack - 4]);
        heights.extend([n_stack - 5, n_stack - 6, n_stack - 7, n_stack - 7]);
    }
    let columns: Vec<Mle<F>> = heights
        .iter()
        .enumerate()
        .map(|(i, &h)| column(h, 7 + 31 * i as u64))
        .collect();
    let layout = StackedLayout::build(&heights, n_stack).unwrap();
    assert_eq!(layout.num_polys(), polys);
    let points: Vec<Vec<EE>> = heights
        .iter()
        .enumerate()
        .map(|(i, &h)| {
            (0..h)
                .map(|j| EE::from(101 + 13 * i as u64 + j as u64))
                .collect()
        })
        .collect();
    let values = columns
        .iter()
        .zip(&points)
        .map(|(c, z)| c.evaluate_in(z).unwrap())
        .collect();
    Case {
        layout,
        columns,
        points,
        values,
    }
}

fn commit(c: &Case, resident: Option<&gpu::ResidentColumns>) -> StackedCommitment<F, H> {
    let columns = multilinear::stacking::borrow(&c.columns);
    StackedCommitment::<F, H>::commit(
        c.layout.clone(),
        &columns,
        resident.map(|store| (store, 0)),
        &config(),
    )
    .unwrap()
}

/// Opens `stacked` and checks the host verifier takes it.
fn open(
    c: &Case,
    stacked: &StackedCommitment<F, H>,
    resident: Option<&gpu::ResidentColumns>,
) -> StackedProof<F, Ext> {
    let columns = multilinear::stacking::borrow(&c.columns);
    let proof = stacked_eval::prove::<F, Ext, _, H>(
        stacked,
        &columns,
        resident.map(|store| (store, 0)),
        &Claimed::PerColumn(&c.points),
        &c.values,
        &config(),
        &mut DefaultTranscript::<Ext, <H as WhirHash>::Transcript>::new(b"whir-room-device"),
    )
    .unwrap();
    stacked_eval::verify::<F, Ext, _, H>(
        &proof,
        &c.layout,
        &stacked.roots(),
        &Claimed::PerColumn(&c.points),
        &c.values,
        stacked.domain(),
        &config(),
        &mut DefaultTranscript::<Ext, <H as WhirHash>::Transcript>::new(b"whir-room-device"),
    )
    .unwrap_or_else(|e| panic!("the host verifier refused the proof: {e:?}"));
    proof
}

fn bytes(proof: &StackedProof<F, Ext>) -> Vec<u8> {
    rkyv::to_bytes::<rkyv::rancor::Error>(proof)
        .unwrap()
        .to_vec()
}

/// Bytes promised across the process — what every `reserve` is checked against.
fn reserved() -> u64 {
    math_cuda::device::backend()
        .expect("needs a GPU")
        .reserved_bytes()
}

/// The room a group promised before any of this: one base codeword.
fn codeword_bytes(c: &Case) -> u64 {
    (1u64 << (c.layout.n_stack() + config().log_blowup)) * 8
}

/// Counts of the room's three events since `from`.
fn room_counts(from: (u64, u64, u64)) -> (u64, u64, u64) {
    (
        gpu::room_parks() - from.0,
        gpu::room_turns() - from.1,
        gpu::room_turn_refusals() - from.2,
    )
}

fn room_marks() -> (u64, u64, u64) {
    (
        gpu::room_parks(),
        gpu::room_turns(),
        gpu::room_turn_refusals(),
    )
}

/// ★ The room goes back when the commits end and comes back for the openings.
///
/// Both arms commit and open the same group; the park arm must hold EXACTLY
/// one room less than the held arm after its commits and after its openings,
/// give it back once and take it back once, and prove the same bytes. Every
/// other term — the codewords, the leaf layers they retain — is the same in
/// both, so the difference is the room and nothing else.
fn the_room_is_given_back_between_the_commits_and_the_openings(c: &Case, room: u64) {
    let arm = |park: bool| {
        gpu::force_room_park(Some(park));
        let marks = room_marks();
        let before = reserved();
        let stacked = commit(c, None);
        let committed = reserved() - before;
        let proof = open(c, &stacked, None);
        let opened = reserved() - before;
        let counts = room_counts(marks);
        drop(stacked);
        assert_eq!(
            reserved(),
            before,
            "park {park}: the group gave back less than it promised"
        );
        // The openings leave nothing promised behind them: every chain's
        // folds, and the leaf layers those kept, went back with the chain.
        assert_eq!(
            opened,
            committed,
            "park {park}: the openings left {} B promised past their end",
            opened.saturating_sub(committed)
        );
        (bytes(&proof), committed, opened, counts)
    };
    let (held, held_committed, held_opened, held_counts) = arm(false);
    let (parked, park_committed, park_opened, park_counts) = arm(true);
    gpu::force_room_park(None);
    assert_eq!(
        held_counts,
        (0, 0, 0),
        "the held arm gave its room back or took a turn"
    );
    assert_eq!(
        park_counts,
        (1, 1, 0),
        "the park arm must give its room back once and take it back once, unrefused"
    );
    assert_eq!(
        held_committed - park_committed,
        room,
        "between the commits and the openings the park arm must hold exactly one room less"
    );
    assert_eq!(
        held_opened - park_opened,
        room,
        "after the openings the park arm must have given its turn back"
    );
    assert_eq!(
        parked, held,
        "where the room sits must not move a byte of the proof"
    );
}

/// ⛔ A refused turn is counted, and the openings run on the device anyway — no
/// host fallback of either kind — proving the bytes a granted turn proves.
fn a_refused_turn_is_counted_and_still_opens_on_the_device(c: &Case) {
    gpu::force_room_park(Some(true));
    let stacked = commit(c, None);
    let granted = open(c, &stacked, None);

    // Empty the retention first — `reserve` evicts it on a miss, and a layer
    // given back would let the turn through — then promise every byte the
    // budget has left to nothing.
    assert!(math_cuda::device::reserve(u64::MAX / 4).is_none());
    let be = math_cuda::device::backend().expect("needs a GPU");
    let filler =
        math_cuda::device::reserve(be.vram_budget_bytes().saturating_sub(be.reserved_bytes()));

    let marks = room_marks();
    let (open_fallbacks, commit_fallbacks, lean) = (
        gpu::open_host_fallbacks(),
        gpu::host_fallbacks(),
        gpu::lean_open_calls(),
    );
    let refused = open(c, &stacked, None);
    let counts = room_counts(marks);
    drop(filler);
    gpu::force_room_park(None);

    assert_eq!(
        counts,
        (0, 0, 1),
        "a full budget must refuse the turn, and count it"
    );
    assert_eq!(
        gpu::open_host_fallbacks() - open_fallbacks,
        0,
        "a refused turn must not send the factors to the host"
    );
    assert_eq!(gpu::host_fallbacks() - commit_fallbacks, 0);
    assert_eq!(
        gpu::lean_open_calls() - lean,
        c.layout.num_polys() as u64,
        "the refused group's openings must still have run on the device"
    );
    assert_eq!(
        bytes(&refused),
        bytes(&granted),
        "a refused turn must prove the bytes a granted one does"
    );
}

/// An opening whose device factors decline over a codeword the device holds
/// builds them on the host, proves the same bytes, and is COUNTED — once per
/// chain, and never on the path that did not decline.
fn a_declined_opening_is_counted(c: &Case) {
    let stacked = commit(c, None);
    let polys = c.layout.num_polys() as u64;

    let before = gpu::open_host_fallbacks();
    let on_device = open(c, &stacked, None);
    assert_eq!(
        gpu::open_host_fallbacks() - before,
        0,
        "an opening the device built counted as a host fallback"
    );

    gpu::force_shared_open_declined(true);
    let before = gpu::open_host_fallbacks();
    let declined = open(c, &stacked, None);
    gpu::force_shared_open_declined(false);
    assert_eq!(
        gpu::open_host_fallbacks() - before,
        polys,
        "every chain whose factors the device declined must be counted, once"
    );
    assert_eq!(
        bytes(&declined),
        bytes(&on_device),
        "the host-built factors must prove the device's bytes"
    );
}

/// What `f` took from the device's memory pool at its peak, beyond what the
/// pool had handed out when it started.
fn peak_of<R>(f: impl FnOnce() -> R) -> (R, u64) {
    math_cuda::device::drain_and_trim().unwrap();
    let (before, _) = math_cuda::device::pool_used_bytes().unwrap();
    math_cuda::device::reset_pool_high_water().unwrap();
    let out = f();
    math_cuda::device::drain_and_trim().unwrap();
    let (_, high) = math_cuda::device::pool_used_bytes().unwrap();
    (out, high.max(before) - before)
}

const MIB: f64 = 1024.0 * 1024.0;

/// ★ THE RESIZE, MEASURED: a group's commits and its openings take no more from
/// the card than the rooms the resize promises for them — and those rooms are
/// smaller than the codeword the group used to promise.
///
/// Read off the memory pool's own high-water mark, which sees every
/// allocation the turns make, not only the ones somebody reserved. The commits
/// leave their codewords and the leaf layers they retain behind, so theirs is
/// checked with those added; the openings leave nothing. Each bound is checked
/// from below too — by the one buffer the phase cannot avoid — so a probe that
/// read nothing could not pass.
fn a_groups_turns_hold_what_their_rooms_promise() {
    let c = case(22, 2);
    let polys = c.layout.num_polys() as u64;
    let codeword = codeword_bytes(&c);
    gpu::force_room_park(Some(true));
    gpu::force_room_resize(Some(true));
    let turns = gpu::room_turns_for(&c.layout, &config(), true);

    // The transform's twiddles are cached for the process the first time a
    // domain is used, outside every promise: one commit first, so the
    // measured one does not pay for them.
    drop(commit(&c, None));

    let (stacked, committed) = peak_of(|| commit(&c, None));
    let (_, opened) = peak_of(|| open(&c, &stacked, None));
    drop(stacked);
    gpu::force_room_park(None);
    gpu::force_room_resize(None);

    // At the production first fold of six a leaf layer is a sixteenth of the
    // codeword, and the commits keep one each.
    let layers = polys * codeword / 16;
    let slack = |bytes: u64| bytes / 20 + (8 << 20);
    println!(
        "2^22 x{polys}: commit peak {:.1} MiB = {polys} codewords + layers + {:.1} MiB against a \
         {:.1} MiB turn; opening peak {:.1} MiB against a {:.1} MiB turn; codeword {:.1} MiB",
        committed as f64 / MIB,
        committed.saturating_sub(polys * codeword + layers) as f64 / MIB,
        turns.commit as f64 / MIB,
        opened as f64 / MIB,
        turns.open as f64 / MIB,
        codeword as f64 / MIB,
    );
    assert!(
        turns.commit < codeword && turns.open < codeword,
        "the resized turns must be smaller than the codeword they replace"
    );
    assert!(
        committed <= polys * codeword + layers + turns.commit + slack(turns.commit),
        "the commits took more than their codewords, their layers and their turn"
    );
    assert!(
        committed >= polys * codeword + (8 << c.layout.n_stack()),
        "the probe missed the commits' coefficients beside their codewords"
    );
    assert!(
        opened <= turns.open + slack(turns.open),
        "an opening took more than its turn"
    );
    assert!(
        opened >= math_cuda::whir::tree_bytes(c.layout.n_stack() + config().log_blowup - 6),
        "the probe missed the tree the first round's paths are read from"
    );
}

/// Rooms sized, and of those sized to their turn, since `from`.
fn sizings_since(from: (u64, u64)) -> (u64, u64) {
    (
        gpu::room_sizings() - from.0,
        gpu::rooms_turn_sized() - from.1,
    )
}

#[test]
fn a_group_promises_the_card_what_its_turns_take() {
    let c = case(16, 2);
    a_declined_opening_is_counted(&c);
    gpu::force_room_resize(Some(false));
    let from = (gpu::room_sizings(), gpu::rooms_turn_sized());
    the_room_is_given_back_between_the_commits_and_the_openings(&c, codeword_bytes(&c));
    let (sized, turn_sized) = sizings_since(from);
    assert!(
        sized > 0 && turn_sized == 0,
        "resize off: {sized} rooms sized, {turn_sized} to their turn (want some, and none)"
    );
    gpu::force_room_resize(Some(true));
    let turns = gpu::room_turns_for(&c.layout, &config(), true);
    let from = (gpu::room_sizings(), gpu::rooms_turn_sized());
    the_room_is_given_back_between_the_commits_and_the_openings(&c, turns.commit.max(turns.open));
    let (sized, turn_sized) = sizings_since(from);
    assert!(
        sized > 0 && turn_sized == sized,
        "resize on: {sized} rooms sized, {turn_sized} to their turn (want all)"
    );
    a_refused_turn_is_counted_and_still_opens_on_the_device(&c);
    gpu::force_room_resize(None);
    a_groups_turns_hold_what_their_rooms_promise();
}
