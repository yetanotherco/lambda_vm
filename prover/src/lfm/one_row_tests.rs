//! S2 (one-row openings) on the LFM side, host only (design/FRI.md §7.5.2–4,
//! REVIEW-FRI F8): the commit helpers at both leaf layouts, the registry
//! policy (a one-row format never reads `LFM_REGISTRY`), the one-row roots of
//! a program's artifacts, and the in-circuit register commitment against its
//! host twin at BOTH layouts. Execute-only and artifact builds; nothing here
//! proves.

use stark::leaf_layout::LeafLayout;
use stark::proof::options::{GoldilocksCubicProofOptions, OneRowMode, ProofOptions};

use crate::tables::types::{FE, GoldilocksField};

use super::commit::{
    commit_columns_with, commit_lde_columns, commit_lde_columns_with, group_columns,
};
use super::programs::{RegisterDerivationShape, register_derivation_program};
use super::registry::{
    LfmProgramKind, PROGRAM_GROUP_SLOTS, REGISTRY_READS, build_artifacts, program_groups, resolve,
    resolve_artifacts,
};
use super::validator::validate;
use super::word::LfmWord;

fn options(blowup: u8, one_row: OneRowMode) -> ProofOptions {
    let mut o = GoldilocksCubicProofOptions::with_blowup(blowup).expect("options");
    o.format.one_row = one_row;
    o
}

fn splitmix(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// `lfm::commit` at one row is `stark::commitment` at `rows_per_leaf = 1`,
/// under the block pin; at row pairs it is today's helper.
#[test]
fn the_commit_helpers_follow_the_layout() {
    let mut st = 11u64;
    let cols: Vec<Vec<FE>> = (0..3)
        .map(|_| (0..64).map(|_| FE::from(splitmix(&mut st))).collect())
        .collect();
    type B =
        <crate::hash_pin::BlockStarkHash as stark::config::StarkHash>::Batched<GoldilocksField>;
    let (_, row) =
        stark::commitment::commit_bit_reversed_with::<GoldilocksField, B>(&cols, 1).expect("tree");
    let (_, pair) =
        stark::commitment::commit_bit_reversed_with::<GoldilocksField, B>(&cols, 2).expect("tree");
    assert_eq!(commit_lde_columns_with(&cols, LeafLayout::Row), row);
    assert_eq!(commit_lde_columns_with(&cols, LeafLayout::RowPair), pair);
    assert_eq!(
        commit_lde_columns(&cols),
        pair,
        "today's helper is the row-pair one"
    );
    assert_ne!(row, pair);
}

/// ★ The registry policy (FRI.md §7.5.4): `LFM_REGISTRY` stays row-pair only.
/// At the default format `resolve_artifacts` IS the registry row; under a
/// one-row format (`On` or `Auto`) it never reads the registry and builds the
/// program's artifacts at run time — with the SAME row-pair roots and program
/// id (so the identity is unchanged) plus the one-row roots.
#[test]
fn a_one_row_format_never_reads_the_registry() {
    let kind = LfmProgramKind::TrivialV0;
    let reads = || REGISTRY_READS.with(|c| c.get());

    let before = reads();
    let default = resolve_artifacts(kind, &options(2, OneRowMode::Off)).expect("registered");
    assert_eq!(reads(), before + 1, "the default format reads the registry");
    assert_eq!(default, resolve(kind, 2).expect("row").artifacts());
    assert!(default.one_row_roots.is_none());

    for mode in [OneRowMode::On, OneRowMode::Auto] {
        let before = reads();
        let built = resolve_artifacts(kind, &options(2, mode)).expect("built");
        assert_eq!(reads(), before, "{mode:?}: LFM_REGISTRY must not be read");
        assert_eq!(
            built.roots, default.roots,
            "{mode:?}: row-pair roots unchanged"
        );
        assert_eq!(
            built.program_id, default.program_id,
            "{mode:?}: identity unchanged"
        );
        let one_row = built.one_row_roots.as_ref().expect("one-row roots built");
        for slot in 0..=10 {
            let root = one_row.roots[slot].expect("every committed group has a one-row root");
            assert_ne!(root, built.roots[slot], "slot {slot}: layouts differ");
        }
        // Blowup 2 has no one-row static twin: the hosted KECCAK_RC and
        // BITWISE roots are hard misses (RULINGS 14), not recomputes.
        assert_eq!(one_row.roots[13], None);
        assert_eq!(one_row.roots[14], None);
    }
}

/// The one-row roots are each group's own one-row commitment, and the hosted
/// static tables take their shipped twins (blowup 4, the knob's blowup).
#[test]
fn artifacts_carry_each_groups_one_row_root() {
    let program = LfmProgramKind::TrivialV0.program();
    let opts = options(4, OneRowMode::On);
    let artifacts = build_artifacts(&program, &opts);
    let one_row = artifacts.one_row_roots.as_ref().expect("one-row roots");
    let groups = program_groups(&program);
    for (slot, group) in groups.iter().enumerate().take(PROGRAM_GROUP_SLOTS) {
        assert_eq!(
            one_row.roots[slot],
            Some(commit_columns_with(
                &group_columns(group),
                &opts,
                LeafLayout::Row
            )),
            "slot {slot}"
        );
    }
    assert_eq!(
        one_row.roots[13],
        crate::tables::keccak_rc::preprocessed_commitment_for(&opts, LeafLayout::Row)
    );
    assert!(one_row.roots[13].is_some() && one_row.roots[14].is_some());
    assert_eq!(
        one_row.roots[12], None,
        "KECCAK_RND has no preprocessed columns"
    );
    assert_eq!(
        one_row.roots[super::airs::BLAKE3_SLOT],
        one_row.blake3_chunk_roots.first().copied()
    );
    // The default format builds none, and its artifacts are unchanged.
    let default = build_artifacts(&program, &options(4, OneRowMode::Off));
    assert!(default.one_row_roots.is_none());
    assert_eq!(default.roots, artifacts.roots);
    assert_eq!(default.program_id, artifacts.program_id);
}

fn register_file(seed: u64) -> Vec<u32> {
    let mut st = seed;
    (0..crate::tables::register::NUM_REGISTER_ADDRESSES)
        .map(|_| (splitmix(&mut st) >> 32) as u32)
        .collect()
}

fn digest_bytes(public: &[(u32, LfmWord)]) -> [u8; 32] {
    use math::field::traits::IsPrimeField;
    if public.len() == 1 {
        return super::algebraic_commit::digest_to_commitment(&public[0].1);
    }
    assert_eq!(
        public.len(),
        2,
        "a digest is one algebraic word or two byte words"
    );
    let mut out = [0u8; 32];
    for h in 0..8 {
        let lane = public[h / 4].1[h % 4];
        let half = GoldilocksField::canonical(lane.value()) as u32;
        out[4 * h..4 * h + 4].copy_from_slice(&half.to_le_bytes());
    }
    out
}

/// ★ The in-circuit register commitment against its host twin at BOTH leaf
/// layouts (FRI.md §7.5.3). A mismatch would show only as a runtime
/// `DivByZero` deep in a node, so each layout gets its own root equality. One
/// emitter, two constants (`RegisterDerivationShape::rows_per_leaf`).
#[test]
fn the_register_derivation_matches_its_host_twin_at_both_layouts() {
    for blowup in [2usize, 4] {
        let opts = GoldilocksCubicProofOptions::with_blowup(blowup as u8).expect("options");
        for layout in [LeafLayout::RowPair, LeafLayout::Row] {
            let shape = RegisterDerivationShape {
                blowup,
                coset_offset: opts.coset_offset,
                rows_per_leaf: layout.rows_per_leaf(),
            };
            assert_eq!(shape.leaves(), 128 * blowup / layout.rows_per_leaf());
            let program = register_derivation_program(shape);
            validate(&program).expect("admission");
            let (init, fini) = (register_file(1), register_file(2));
            let column = |v: &[u32]| {
                v.iter()
                    .map(|&x| super::word::base_word(FE::from(x as u64)))
                    .collect::<Vec<_>>()
            };
            let arenas = vec![column(&init), column(&fini)];
            let exec = super::executor::execute(&program, &arenas, &crate::hash_pin::BLOCK_HASHER)
                .unwrap_or_else(|e| panic!("blowup {blowup} {layout:?}: {e:?}"));
            let host = crate::tables::register::compute_precomputed_commitment_with_fini_layout(
                &opts, &init, &fini, layout,
            );
            assert_eq!(
                digest_bytes(&exec.public_words),
                host,
                "blowup {blowup} {layout:?}: the emitted root must equal the host twin's"
            );
            if layout == LeafLayout::RowPair {
                assert_eq!(
                    host,
                    crate::tables::register::compute_precomputed_commitment_with_fini(
                        &opts, &init, &fini
                    ),
                    "row pairs are today's commitment"
                );
            }
        }
    }
}
