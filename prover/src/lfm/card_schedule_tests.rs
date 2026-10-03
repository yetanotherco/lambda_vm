//! The gates for [`super::card_schedule`]'s knobs: each one moves where or when
//! work runs, and none may move a byte a proof commits to.
//!
//! - SCOPE: [`build_artifacts_sectioned`] builds exactly what
//!   [`build_artifacts_with_hasher`] builds, over every branch of the walk, and
//!   its device commits all land inside the section.
//! - AHEAD: [`lfm_prepare`] + [`lfm_prove_prepared`] is [`lfm_prove`].
//! - NICE: the pool's own gates are in `card_schedule`; the fill's identity on
//!   the pool is in `trace_identity_tests`.

use std::cell::Cell;

use stark::proof::options::{GoldilocksCubicProofOptions, OneRowMode, ProofOptions};

use super::chunking::{Blake3Chunking, HashChunking};
use super::compiler::LfmProgram;
use super::hash::HasherKind;
use super::programs::{blake3_sponge_program, fri_toy_program, trivial_program};
use super::registry::{build_artifacts_sectioned, build_artifacts_with_hasher};

fn options(one_row: OneRowMode) -> ProofOptions {
    let mut o = GoldilocksCubicProofOptions::with_blowup(2).expect("options");
    o.format.one_row = one_row;
    o
}

/// Every branch of the build's walk: an unsplit and a split `LFM_HASH` (a hash
/// tail or none), one `LFM_BLAKE3` chunk and three.
fn programs() -> Vec<(&'static str, LfmProgram)> {
    vec![
        ("TrivialV0", trivial_program()),
        (
            "TrivialV0, LFM_HASH split",
            trivial_program().with_hash_chunking(HashChunking::split_at(2)),
        ),
        ("FriToyV0", fri_toy_program()),
        (
            "Blake3Chain(768), three LFM_BLAKE3 chunks",
            // 12 compressions at 5 per chunk: 5 + 5 + 2.
            blake3_sponge_program(768).with_blake3_chunking(Blake3Chunking::from_compressions(5)),
        ),
    ]
}

/// ★★★ THE SCOPE GATE: the sectioned build IS the whole build, field by field,
/// under both leaf layouts — and the split accounts for every commit.
///
/// On a build without a card every commit is a host commit and the section is
/// never entered; on the box the groups above the device floor (`LFM_RANGE`,
/// 65,536 rows, in every program) go through it. The equality holds either way,
/// which is the point: where a commit runs is not part of what it commits to.
#[test]
fn the_sectioned_build_is_the_whole_build() {
    let mut saw_tail = false;
    let mut saw_chunks = false;
    for one_row in [OneRowMode::Off, OneRowMode::Auto, OneRowMode::On] {
        let opts = options(one_row);
        for (name, program) in programs() {
            let whole = build_artifacts_with_hasher(&program, &opts, HasherKind::Test);
            let entered = Cell::new(0usize);
            let (sectioned, split) =
                build_artifacts_sectioned(&program, &opts, HasherKind::Test, || {
                    entered.set(entered.get() + 1)
                });
            assert_eq!(
                sectioned, whole,
                "{name} @ {one_row:?}: the sectioned build drifted from the whole one"
            );
            // Slots 0..=10, every BLAKE3 chunk, the hash tail — per layout.
            let per_layout =
                11 + whole.blake3_chunk_roots.len() + (whole.hash_chunk_roots.len() - 1);
            let layouts = if one_row == OneRowMode::Off { 1 } else { 2 };
            assert_eq!(
                split.host_commits + split.device_commits,
                per_layout * layouts,
                "{name} @ {one_row:?}: the split does not account for every commit: {split:?}"
            );
            assert_eq!(
                entered.get(),
                usize::from(split.device_commits > 0),
                "{name} @ {one_row:?}: the section must be entered once exactly when there \
                 is device work, and never otherwise: {split:?}"
            );
            assert_eq!(
                whole.one_row_roots.is_some(),
                one_row != OneRowMode::Off,
                "{name} @ {one_row:?}: the one-row walk ran under the wrong format"
            );
            saw_tail |= whole.hash_chunk_roots.len() > 1;
            saw_chunks |= whole.blake3_chunk_roots.len() > 1;
        }
    }
    // Vacuity guard: without these the walk's tail and chunk branches could go
    // uncompared while the test still passed.
    assert!(saw_tail, "no case exercised the LFM_HASH tail");
    assert!(
        saw_chunks,
        "no case exercised more than one LFM_BLAKE3 chunk"
    );
}

/// ★★ THE SECTION COVERS EVERY DEVICE COMMIT, read off the device layer's own
/// counters rather than trusted from the code's shape: no device commit before
/// the section is entered, exactly `device_commits` of them inside it, and the
/// host commits all before it.
///
/// ⚠ The counters are process-wide, so another thread building artifacts at the
/// same moment could only make this fail, never pass. It is exact under nextest
/// (one process per test) and under `--test-threads=1`, which is how the box
/// runs it.
#[cfg(feature = "cuda")]
#[test]
fn every_device_commit_lands_inside_the_section() {
    use super::commit::device_host_group_counts;
    let opts = options(OneRowMode::On);
    let program = fri_toy_program();
    let before = device_host_group_counts();
    let at_entry = Cell::new(None);
    let (_, split) = build_artifacts_sectioned(&program, &opts, HasherKind::Test, || {
        at_entry.set(Some(device_host_group_counts()))
    });
    let after = device_host_group_counts();
    let Some(entry) = at_entry.get() else {
        assert_eq!(
            split.device_commits, 0,
            "device work was routed but the section never entered"
        );
        assert!(
            stark::gpu_lde::device_vram_budget_bytes().is_none(),
            "a card is present, yet no commit reached it: LFM_RANGE (65,536 rows) \
             clears the device floor in every program"
        );
        return;
    };
    assert_eq!(
        (entry.0 - before.0, entry.1 - before.1),
        (0, split.host_commits as u64),
        "before the section: no device commit, and every host commit"
    );
    assert_eq!(
        (after.0 - entry.0, after.1 - entry.1),
        (split.device_commits as u64, 0),
        "inside the section: every device commit, and no host commit"
    );
}

/// ★ AHEAD's premise: the prepare/prove split proves what `lfm_prove` proves —
/// the same public words, and a proof the artifacts' verifier accepts.
///
/// Proof BYTES are not compared: the grinding nonce is found by a parallel
/// search, so two proves of one statement differ there and at every query it
/// seeds (memory grinding-nonce-nondeterminism). The traces, the roots and the
/// statement are deterministic, and those are what the split could have moved.
///
/// ⚠ A PROVE: box-scale (the fixed 2^20-row BITWISE table is in every proof).
#[test]
fn a_prepared_prove_is_the_prove() {
    use super::proof::{lfm_prepare, lfm_prove, lfm_prove_prepared, verify_against_artifacts};
    use super::word::LfmWord;
    use crate::tables::types::FE;

    let opts = options(OneRowMode::Off);
    let program = trivial_program();
    let arenas: Vec<Vec<LfmWord>> = vec![
        (0..4u64)
            .map(|i| core::array::from_fn(|j| FE::from(1_000 * (i + 1) + j as u64)))
            .collect(),
    ];
    let artifacts = build_artifacts_with_hasher(&program, &opts, HasherKind::Test);
    let plain = lfm_prove(&program, &artifacts, &arenas, &opts).expect("the plain prove");
    let prepared = lfm_prepare(&program, &arenas, HasherKind::Test).expect("the prepare");
    let split = lfm_prove_prepared(&artifacts, prepared, &opts).expect("the prepared prove");
    assert_eq!(
        split.public_words, plain.public_words,
        "the same execution publishes the same words"
    );
    assert!(verify_against_artifacts(
        &artifacts,
        &plain.proof,
        &plain.public_words,
        &opts
    ));
    assert!(
        verify_against_artifacts(&artifacts, &split.proof, &split.public_words, &opts),
        "the prepared prove must verify against the same artifacts"
    );
}

/// ⛔ The hasher check `lfm_prove_with_hasher` makes, made on the split too.
#[test]
#[should_panic(expected = "program_id binds the hasher")]
fn a_prepared_prove_refuses_another_hashers_artifacts() {
    use super::proof::{lfm_prepare, lfm_prove_prepared};
    use super::word::LfmWord;
    use crate::tables::types::FE;

    let opts = options(OneRowMode::Off);
    let program = trivial_program();
    let arenas: Vec<Vec<LfmWord>> = vec![
        (0..4u64)
            .map(|i| core::array::from_fn(|j| FE::from(1_000 * (i + 1) + j as u64)))
            .collect(),
    ];
    let artifacts = build_artifacts_with_hasher(&program, &opts, HasherKind::Rpx);
    let prepared = lfm_prepare(&program, &arenas, HasherKind::Test).expect("the prepare");
    let _ = lfm_prove_prepared(&artifacts, prepared, &opts);
}
