use super::*;
use crate::block_whir::{AnyBlockWhirProof, prove_range_reference};
use crate::lfm::{
    LfmWord,
    b3l_tables::B3lShape,
    block_node::{BlockBindings, bind_and_publish_with},
    executor::execute,
    last_level::{CLUSTER_LAST_FILE_VERSION, LAST_FILE_HEADER, LAST_FILE_MAGIC, LAST_FILE_VERSION},
    whir_block::{LeafChecks, leaf_program_as_carrier, leaf_program_with},
    wire::WireFormat,
    wlfm::LeafArtifacts,
};
use crate::tables::types::FE;
use digest::Digest;
use stark::multilinear_table::batched::VerifierChecks;
use std::sync::OnceLock;

const FAMILY: WlfmFamily = WlfmFamily::B3;
const LAST: LastLevel = LastLevel::pinned(FAMILY);

#[test]
fn range_leaf_uses_the_local_levels_stack_placement() {
    use crate::lfm::builder::{B3Block, B3Cv};
    use multilinear::gpu;

    let mut builder = wlfm_leg_builder(FAMILY);
    let zero = builder.digest_const([FE::zero(); 4]).as_cell();
    for word in builder.b3_compress(
        B3Cv::Iv,
        B3Block::Lanes([zero; 4]),
        0,
        crypto::hash::blake3::chain::FLAGS_ONE_BLOCK as u8,
    ) {
        builder.public(word);
    }
    let program = finish_program(builder).unwrap();
    let local = levels::prepare_level(&program, FAMILY, None).unwrap();
    assert!(local.prepared_cells() > 0);
    let base_budget = gpu::reserve_budget();
    LeafPrebuild::new(program, None, LAST, Clock::default())
        .unwrap()
        .prove(0, Clock::default(), |leaf| {
            let local = levels::commit_level(local, 1)?;
            assert_eq!(leaf.build.place(), local.place());
            assert_eq!(
                leaf.build.artifacts.derived_roots(),
                local.artifacts.derived_roots()
            );
            Ok(())
        })
        .unwrap();
    assert_eq!(gpu::reserve_budget(), base_budget);
}

fn options() -> BlockOptions {
    let mut options = crate::lfm::whir_block_tests::small_block_options();
    options.window_log2 = Some(2);
    options.max_rows.cpu = 8;
    options.drop_streamed_ops = true;
    options
}

fn fixture_block(name: &str, input: &[u8]) -> (Vec<u8>, AnyBlockWhirProof) {
    fixture_block_cut(name, input, &[2, 4])
}

fn fixture_block_cut(name: &str, input: &[u8], cuts: &[usize]) -> (Vec<u8>, AnyBlockWhirProof) {
    let elf = crate::test_utils::asm_elf_bytes(name);
    let format = pinned_format(FAMILY).unwrap();
    assert!(format.local_lookups());
    assert!(matches!(
        format.argue,
        multilinear::whir_chain::ArgueFormat::Batched {
            ladder: multilinear::whir_chain::GkrLadder::Double,
            ..
        }
    ));
    let opts = crate::lfm::proof::block_base_options_under(&format.zf);
    let (proof, _, _) = prove_range_reference(
        &elf,
        input,
        &opts,
        &format,
        &options(),
        cuts,
        Clock::default(),
    )
    .unwrap();
    (elf, proof)
}

fn plan_for(elf: &[u8], base: &AnyBlockWhirProof, map: Option<&LeafMap>) -> WhirBlockPlan {
    let format = pinned_format(FAMILY).unwrap();
    let opts = crate::lfm::proof::block_base_options_under(&format.zf);
    cluster_plan(elf, &opts, &format, base.statement(), None, map).unwrap()
}

struct Fixture {
    elf: Vec<u8>,
    base: AnyBlockWhirProof,
    map: LeafMap,
    plan: WhirBlockPlan,
    words: [Vec<LfmWord>; 2],
}

fn fixture() -> &'static Fixture {
    static FIXTURE: OnceLock<Fixture> = OnceLock::new();
    FIXTURE.get_or_init(|| {
        let (elf, base) = fixture_block("test_commit_split", &[0x11]);
        assert!(
            base.groups.len() >= 3,
            "fixture needs a group that can move"
        );
        let mut map = vec![0; base.groups.len()];
        map[1] = 1;
        let map = LeafMap::decode(&map, base.groups.len()).unwrap();
        let plan = plan_for(&elf, &base, Some(&map));
        let (argue, _) = plan.group_stack(0);
        assert!(argue > 0);
        let words = std::array::from_fn(|i| {
            leaf_words(
                &plan.leaf_program(i).unwrap(),
                &block_leaf_arena(&plan, &base, i).unwrap(),
            )
        });
        Fixture {
            elf,
            base,
            map,
            plan,
            words,
        }
    })
}

fn leaf_words(program: &LfmProgram, arena: &[Vec<LfmWord>]) -> Vec<LfmWord> {
    execute(program, arena, &crate::hash_pin::BLOCK_HASHER)
        .unwrap()
        .public_words
        .into_iter()
        .map(|(_, word)| word)
        .collect()
}

fn bindings(
    layout: &BlockLayout,
    words: &[Vec<LfmWord>],
    checks: BlockBindings,
) -> Result<Vec<LfmWord>, String> {
    use crate::lfm::per_table_aggregator::{HintedPublicWord, LegCells};
    let mut b = wlfm_leg_builder(FAMILY);
    let zero = b.ext_const(&crate::tables::types::FEE::zero());
    let legs: Vec<_> = words
        .iter()
        .map(|words| {
            let arena = b.declare_arena(words.len() as u32);
            let publics = (0..words.len())
                .map(|index| {
                    let word = b.hint_word(arena, index as u32);
                    HintedPublicWord {
                        index: index as u32,
                        halves: Vec::new(),
                        lanes: b.unpack(word).to_vec(),
                    }
                })
                .collect();
            LegCells {
                publics,
                z_alpha: (zero, zero),
            }
        })
        .collect();
    bind_and_publish_with(&mut b, &legs, layout, true, checks);
    let program = finish_program(b)?;
    execute(&program, words, &crate::hash_pin::BLOCK_HASHER)
        .map(|e| e.public_words.into_iter().map(|(_, word)| word).collect())
        .map_err(|e| format!("{e:?}"))
}

fn refused_binding(layout: &BlockLayout, words: &[Vec<LfmWord>], check: &str) {
    assert!(
        bindings(layout, words, BlockBindings::ALL).is_err(),
        "{check} binding"
    );
    assert!(
        bindings(layout, words, BlockBindings::without(check)).is_ok(),
        "without {check}"
    );
}

fn headers(map: &[u8], body: &[u8]) -> Vec<u8> {
    let mut file = LAST_FILE_MAGIC.to_vec();
    file.extend_from_slice(&[CLUSTER_LAST_FILE_VERSION, LAST.code().unwrap(), 2]);
    file.extend_from_slice(&(map.len() as u32).to_le_bytes());
    file.extend_from_slice(map);
    file.extend_from_slice(&crypto::hash::blake3::chain::Blake3Std::digest(map));
    file.extend_from_slice(body);
    file
}

#[test]
fn a_leaf_map_of_another_length_is_refused() {
    assert!(
        cluster_file_parts(&headers(&[0, 1], &[]), LAST, 3)
            .unwrap_err()
            .contains("length")
    );
    assert!(LeafMap::decode(&[0, 1], 1).unwrap_err().contains("length"));
    assert!(
        LeafMap::decode(&[0, 1], 2).is_ok(),
        "mutant uses the file's length"
    );
}

#[test]
fn a_leaf_map_past_the_cap_is_refused() {
    for byte in [8, 255] {
        let map: Vec<_> = (0..=byte).collect();
        assert!(
            LeafMap::decode(&map, map.len())
                .unwrap_err()
                .contains("cap")
        );
        assert!(
            LeafMap::decode(&[0, 1, byte], 3)
                .unwrap_err()
                .contains("cap")
        );
        let uncapped = map.iter().enumerate().map(|(g, _)| vec![g]).collect();
        assert!(BlockPartition::new(uncapped, map.len()).is_ok());
    }
}

#[test]
fn a_leaf_map_that_skips_a_leaf_is_refused() {
    assert!(LeafMap::decode(&[0, 2], 2).unwrap_err().contains("skips"));
    assert!(
        LeafMap::decode(&[1, 0], 2)
            .unwrap_err()
            .contains("first-group")
    );
    assert!(
        LeafMap::decode(&[0, 1], 2).is_ok(),
        "compacting skipped ids would admit it"
    );
}

#[test]
fn a_one_leaf_map_has_one_encoding() {
    let file = headers(&[0, 0], &[]);
    assert!(
        cluster_file_parts(&file, LAST, 2)
            .unwrap_err()
            .contains("version 1")
    );
    assert!(
        BlockPartition::new(vec![vec![0, 1]], 2).is_ok(),
        "coverage alone accepts"
    );
    let mut v1 = file[..LAST_FILE_HEADER].to_vec();
    v1[4] = crate::lfm::last_level::LAST_FILE_VERSION;
    assert!(cluster_file_parts(&v1, LAST, 2).unwrap().0.is_none());
}

#[test]
fn a_file_with_several_leaves_refuses_every_stop_but_the_default() {
    let f = fixture();
    let file = headers(f.map.as_bytes(), &[]);
    let verify = |stop, file: &[u8]| {
        verify_cluster_block_levels_file_at(&f.elf, f.base.statement(), FAMILY, stop, LAST, file)
            .unwrap_err()
    };
    let mut stops: Vec<_> = (0..=levels::MAX_LEVELS + 1)
        .map(LevelStop::Levels)
        .collect();
    stops.extend([0, 1, 2, 3, 4, usize::MAX].map(LevelStop::Groups));
    for stop in stops.into_iter().filter(|&stop| stop != LevelStop::DEFAULT) {
        let error = verify(stop, &file);
        assert!(
            error.contains("only the default chain stop")
                && error.contains(&format!("refused {stop:?}")),
            "{stop:?}: {error}"
        );
        // A version-1 file is the one-leaf verifier's, under any stop.
        let mut v1 = file[..LAST_FILE_HEADER].to_vec();
        v1[4] = LAST_FILE_VERSION;
        assert!(!verify(stop, &v1).contains("chain stop"), "{stop:?}");
    }
    let error = verify(LevelStop::DEFAULT, &file);
    assert!(!error.contains("chain stop"), "the default stop: {error}");
    // The join's own stop reads both leaves' group counts.
    assert!(is_last_level(LevelStop::Groups(4), 1, 2 + 2));
    assert!(check_leaf_stop(LevelStop::Groups(4), 2 + 2).is_err());
    assert!(
        check_leaf_stop(LevelStop::Groups(3), 2 + 2).is_ok(),
        "the stop reads both leaves"
    );
}

#[test]
fn a_junk_body_is_refused_before_any_program_is_derived() {
    let f = fixture();
    let derived = || DERIVATIONS.with(std::cell::Cell::get);
    let honest = join_file();
    let body = LAST_FILE_HEADER + 4 + f.map.as_bytes().len() + 32;
    for junk in [&[][..], &[0xab; 64], &honest[body..honest.len() - 1]] {
        let before = derived();
        let file = headers(f.map.as_bytes(), junk);
        let error = verify_cluster_block_levels_file(&f.elf, f.base.statement(), FAMILY, &file)
            .unwrap_err();
        assert_eq!(
            check_shipment_file(&f.elf, f.base.statement(), Some(&f.map), &file).unwrap_err(),
            error
        );
        assert_eq!(derived(), before, "a program was derived for: {error}");
    }
    let before = derived();
    verify_cluster_block_levels_file(&f.elf, f.base.statement(), FAMILY, &honest).unwrap();
    assert_eq!(
        derived(),
        before + 1,
        "the honest file derives its chain once"
    );
}

#[test]
fn poseidon_range_levels_are_refused() {
    let f = fixture();
    let file = headers(f.map.as_bytes(), &[]);
    let err = verify_cluster_block_levels_file_at(
        &f.elf,
        f.base.statement(),
        WlfmFamily::P1,
        LevelStop::DEFAULT,
        LAST,
        &file,
    )
    .unwrap_err();
    assert!(err.contains("only BLAKE3"));
}

#[test]
fn an_altered_or_truncated_map_is_an_error_before_any_decode() {
    let mut file = headers(&[0, 1, 0], &[]);
    let map_byte = LAST_FILE_HEADER + 4 + 2;
    for end in [LAST_FILE_HEADER, map_byte, map_byte + 1, file.len() - 1] {
        assert!(
            cluster_file_parts(&file[..end], LAST, 3)
                .unwrap_err()
                .contains("truncated")
        );
    }
    file[map_byte] = 1;
    assert!(
        cluster_file_parts(&file, LAST, 3)
            .unwrap_err()
            .contains("checksum")
    );
    assert!(
        LeafMap::decode(&file[map_byte - 2..=map_byte], 3).is_ok(),
        "without checksum the map decodes"
    );
}

#[test]
fn the_join_refuses_a_tamper_in_every_lane_of_the_state() {
    let f = fixture();
    let layout = f.plan.child_layout();
    let honest = bindings(&layout, &f.words, BlockBindings::ALL).unwrap();
    assert_eq!(honest, f.words[0][..layout.total() - 1]);
    assert_eq!(layout.state_words, 2);
    for word in 0..2 {
        for lane in 0..4 {
            let mut tampered = f.words.clone();
            tampered[1][layout.state(word)][lane] += FE::from(1u64);
            refused_binding(&layout, &tampered, "state");
        }
    }
}

fn other_proof() -> &'static (Vec<u8>, AnyBlockWhirProof) {
    static OTHER: OnceLock<(Vec<u8>, AnyBlockWhirProof)> = OnceLock::new();
    OTHER.get_or_init(|| fixture_block("test_commit_split", &[0x22]))
}

#[test]
fn the_join_refuses_a_leaf_of_another_proof_of_the_same_block() {
    let f = fixture();
    let (elf, base) = other_proof();
    let other = plan_for(elf, base, Some(&f.map));
    assert_eq!(other.id(), f.plan.id());
    assert_ne!(f.base.proof.roots, base.proof.roots);
    let layout = f.plan.child_layout();
    let arena = block_leaf_arena(&other, base, 1).unwrap();
    let foreign_leaf = prove_leaf(&other, 1, &arena).unwrap();
    let [a, b] = leaves();
    assert_eq!(
        foreign_leaf.0.artifacts.program_id,
        b.0.artifacts.program_id
    );
    let children = [a.0.clone(), b.0.clone()];
    let join = join_program(&children, &layout).unwrap();
    assert!(run_join(&children, &[&a.1, &foreign_leaf.1], &join).is_err());
    let mut foreign = leaf_words(
        &other.leaf_program(1).unwrap(),
        &block_leaf_arena(&other, base, 1).unwrap(),
    );
    // Isolate the state check even when the other challenge set also moved
    // the numeric bus share.
    foreign[layout.sum()] = f.words[1][layout.sum()];
    refused_binding(&layout, &[f.words[0].clone(), foreign], "state");
}

#[test]
fn the_join_refuses_a_leaf_of_another_block() {
    let f = fixture();
    let (elf, base) = fixture_block("test_commit_4", &[0x11]);
    let mut map = vec![0; base.groups.len()];
    map[1] = 1;
    let other = plan_for(
        &elf,
        &base,
        Some(&LeafMap::decode(&map, map.len()).unwrap()),
    );
    assert_ne!(other.id(), f.plan.id());
    let layout = f.plan.child_layout();
    let arena = block_leaf_arena(&other, &base, 1).unwrap();
    let foreign_leaf = prove_leaf(&other, 1, &arena).unwrap();
    let a = &leaves()[0];
    let children = [a.0.clone(), foreign_leaf.0];
    let join = join_program(&children, &layout).unwrap();
    assert!(run_join(&children, &[&a.1, &foreign_leaf.1], &join).is_err());
    let foreign = leaf_words(
        &other.leaf_program(1).unwrap(),
        &block_leaf_arena(&other, &base, 1).unwrap(),
    );
    assert_eq!(foreign.len(), layout.total());
    assert!(
        bindings(
            &layout,
            &[f.words[0].clone(), foreign.clone()],
            BlockBindings::ALL
        )
        .is_err()
    );
    let mut isolated = f.words.clone();
    for half in 0..2 {
        isolated[1][layout.id(half)] = foreign[layout.id(half)];
    }
    refused_binding(&layout, &isolated, "id");
}

#[test]
fn the_join_refuses_a_root_changed_in_another_leafs_group() {
    let f = fixture();
    let (_, other) = other_proof();
    let root = f
        .base
        .proof
        .roots
        .iter()
        .zip(&other.proof.roots)
        .position(|(a, b)| a != b)
        .expect("private input changes a root");
    let group = (0..f.base.groups.len())
        .find(|&g| (f.plan.root_start(g)..f.plan.root_start(g + 1)).contains(&root))
        .unwrap();
    let owner = f.map.as_bytes()[group] as usize;
    let leaf = 1 - owner;
    let mut arena = block_leaf_arena(&f.plan, &f.base, leaf).unwrap();
    let replacement =
        crate::lfm::proof_arena::commitment_words_for(&other.proof.roots[root], f.plan.wrap_hash());
    arena[0][2 * root..2 * root + 2].copy_from_slice(&replacement);
    let program = f.plan.leaf_program(leaf).unwrap();
    assert!(execute(&program, &arena, &crate::hash_pin::BLOCK_HASHER).is_err());

    let other_plan = plan_for(&f.elf, other, Some(&f.map));
    let mut mixed = f.words.clone();
    mixed[leaf] = leaf_words(
        &other_plan.leaf_program(leaf).unwrap(),
        &block_leaf_arena(&other_plan, other, leaf).unwrap(),
    );
    mixed[leaf][f.plan.child_layout().sum()] = f.words[leaf][f.plan.child_layout().sum()];
    refused_binding(&f.plan.child_layout(), &mixed, "state");
}

#[test]
fn the_join_refuses_an_altered_share_and_an_altered_output() {
    let f = fixture();
    let layout = f.plan.child_layout();
    for (index, check) in [(layout.sum(), "bus"), (layout.out_half(0), "out")] {
        let mut changed = f.words.clone();
        changed[1][index][0] += FE::from(1u64);
        refused_binding(&layout, &changed, check);
    }
}

#[test]
fn the_join_refuses_no_carrier_and_two_carriers() {
    let f = fixture();
    let layout = f.plan.child_layout();
    for (leaf, carrier) in [(0, false), (1, true)] {
        let program = leaf_program_as_carrier(&f.plan, leaf, carrier).unwrap();
        let mut words = f.words.clone();
        words[leaf] = leaf_words(&program, &block_leaf_arena(&f.plan, &f.base, leaf).unwrap());
        refused_binding(&layout, &words, "bus");
    }
}

type ProvedLeaf = (WlfmChild, WlfmProof);

fn leaves() -> &'static [ProvedLeaf; 2] {
    static LEAVES: OnceLock<[ProvedLeaf; 2]> = OnceLock::new();
    LEAVES.get_or_init(|| {
        let f = fixture();
        std::array::from_fn(|leaf| {
            prove_leaf(
                &f.plan,
                leaf,
                &block_leaf_arena(&f.plan, &f.base, leaf).unwrap(),
            )
            .unwrap()
        })
    })
}

fn run_join(
    children: &[WlfmChild],
    proofs: &[&WlfmProof],
    program: &LfmProgram,
) -> Result<(), String> {
    let arenas = children
        .iter()
        .zip(proofs)
        .map(|(child, proof)| wlfm_leg_arena(child, proof))
        .collect::<Result<Vec<_>, _>>()?;
    execute(program, &arenas, &crate::hash_pin::BLOCK_HASHER)
        .map(|_| ())
        .map_err(|e| format!("{e:?}"))
}

#[test]
fn the_join_refuses_leaf_proofs_in_another_order() {
    let f = fixture();
    let [a, b] = leaves();
    let children = [a.0.clone(), b.0.clone()];
    let join = join_program(&children, &f.plan.child_layout()).unwrap();
    assert!(run_join(&children, &[&b.1, &a.1], &join).is_err());
    let reversed = [b.0.clone(), a.0.clone()];
    let mutant = join_program(&reversed, &f.plan.child_layout()).unwrap();
    assert!(
        run_join(&reversed, &[&b.1, &a.1], &mutant).is_ok(),
        "prover-controlled leg order"
    );
}

#[test]
fn the_join_refuses_todays_closed_level_1_as_a_leaf() {
    let f = fixture();
    let plan = plan_for(&f.elf, &f.base, None);
    let program = plan.closed_block_program().unwrap();
    let level = build_leaf_program(program, level_rounds(&plan), false).unwrap();
    let proof = prove_level(&level, &block_leaf_arena(&plan, &f.base, 0).unwrap()).unwrap();
    let closed = levels::child_of(&plan, &level);
    let other = leaves()[1].0.clone();
    assert!(
        join_program(&[closed.clone(), other.clone()], &f.plan.child_layout())
            .err()
            .unwrap()
            .contains("word count")
    );
    assert!(
        wlfm_leg_arena(&closed, &proof).is_ok(),
        "the same proof is an admissible ordinary leg"
    );
    let lied = WlfmChild {
        num_public_words: f.plan.child_layout().total(),
        ..closed
    };
    assert!(
        wlfm_leg_arena(&lied, &proof).is_err(),
        "padding the declared child layout does not pad its proof"
    );
}

fn mutant_join_file(plan: &WhirBlockPlan, leaves: [ProvedLeaf; 2], check: &str) -> Vec<u8> {
    let f = fixture();
    let children = leaves.each_ref().map(|leaf| leaf.0.clone());
    let arenas = children
        .iter()
        .zip(&leaves)
        .map(|(child, leaf)| wlfm_leg_arena(child, &leaf.1).unwrap())
        .collect::<Vec<_>>();
    let program = emit_join(&children, &plan.child_layout(), |b, legs, layout| {
        bind_and_publish_with(b, legs, layout, true, BlockBindings::without(check))
    })
    .unwrap();
    let level = build_level(
        program,
        FAMILY,
        level_rounds(plan),
        false,
        (LevelStop::DEFAULT, 2),
        LAST,
    )
    .unwrap();
    let proof = prove_level(&level, &arenas).unwrap();
    verify_wlfm(&level.build.artifacts, &proof, &proof.public_words)
        .expect("mutant verifier accepts");
    assert!(top_claims(&f.plan, &proof.public_words));
    encode_file(&proof, LAST, WireFormat::Dedup, Some(&f.map)).unwrap()
}

fn refuse_coverage(duplicate: bool) {
    let f = fixture();
    let mut leaves = f.plan.partition().leaves().to_vec();
    if duplicate {
        leaves[1].push(2);
    } else {
        leaves[0].retain(|&g| g != 2);
    }
    assert!(BlockPartition::new(leaves.clone(), f.map.as_bytes().len()).is_err());
    let mutation = BlockPartition::unvalidated(leaves, f.map.as_bytes().len());
    let plan = plan_for(&f.elf, &f.base, Some(&f.map)).with_partition(mutation);
    let proofs = std::array::from_fn(|leaf| {
        prove_leaf(
            &plan,
            leaf,
            &block_leaf_arena(&plan, &f.base, leaf).unwrap(),
        )
        .unwrap()
    });
    let file = mutant_join_file(&plan, proofs, "bus");
    assert!(
        verify_cluster_block_levels_file(&f.elf, f.base.statement(), FAMILY, &file)
            .unwrap_err()
            .contains("does not verify")
    );
}

#[test]
fn a_group_proved_in_no_leaf_is_refused_at_the_public_entry() {
    refuse_coverage(false);
}

#[test]
fn a_group_proved_in_two_leaves_is_refused_at_the_public_entry() {
    refuse_coverage(true);
}

fn join_proof() -> &'static WlfmProof {
    static JOIN: OnceLock<WlfmProof> = OnceLock::new();
    JOIN.get_or_init(|| {
        let f = fixture();
        prove_join(
            &f.plan.child_layout(),
            level_rounds(&f.plan),
            leaves().to_vec(),
            LAST,
        )
        .unwrap()
    })
}

fn join_file() -> Vec<u8> {
    encode_file(join_proof(), LAST, WireFormat::Dedup, Some(&fixture().map)).unwrap()
}

#[test]
fn a_file_is_refused_under_any_other_map() {
    let f = fixture();
    let proof = join_proof();
    let file = join_file();
    verify_cluster_block_levels_file(&f.elf, f.base.statement(), FAMILY, &file).unwrap();
    check_shipment_file(&f.elf, f.base.statement(), Some(&f.map), &file).unwrap();
    assert!(
        levels::verify_block_levels_file(&f.elf, f.base.statement(), FAMILY, &file)
            .unwrap_err()
            .contains("version")
    );
    let mut moved = f.map.as_bytes().to_vec();
    moved[2] = 1;
    let moved = LeafMap::decode(&moved, moved.len()).unwrap();
    let altered = encode_file(proof, LAST, WireFormat::Dedup, Some(&moved)).unwrap();
    assert!(
        verify_cluster_block_levels_file(&f.elf, f.base.statement(), FAMILY, &altered)
            .unwrap_err()
            .contains("does not verify")
    );
    let before = DERIVATIONS.with(std::cell::Cell::get);
    assert!(
        check_shipment_file(&f.elf, f.base.statement(), Some(&f.map), &altered)
            .unwrap_err()
            .contains("disagrees with the coordinator's map")
    );
    assert_eq!(DERIVATIONS.with(std::cell::Cell::get), before);
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "GPU tier: independently derives the two-leaf fixture twice"]
fn two_leaf_card_artifacts_match_host() {
    let f = fixture();
    assert_card_artifacts_match_host(&f.elf, f.base.statement(), &f.map);
}

#[test]
fn a_weakened_leaf_program_from_a_worker_is_refused_before_publication() {
    let f = fixture();
    let arena = block_leaf_arena(&f.plan, &f.base, 1).unwrap();
    let weak = prebuild_weakened_leaf(&f.plan, 1, Clock::default()).unwrap();
    let weak = prove_prebuilt_leaf(&f.plan, 1, weak, &arena, Clock::default()).unwrap();
    assert_ne!(
        weak.0.artifacts.program_id,
        leaves()[1].0.artifacts.program_id
    );
    let from_worker = crate::lfm::wlfm::LeafArtifacts::from_build(&weak.0.artifacts);
    let worker = WlfmChild {
        artifacts: from_worker.into_artifacts(level_rounds(&f.plan)).unwrap(),
        ..weak.0
    };
    let proof = prove_join(
        &f.plan.child_layout(),
        level_rounds(&f.plan),
        vec![leaves()[0].clone(), (worker, weak.1)],
        LAST,
    )
    .expect("prover's join accepts the worker's own artifacts");
    let err = verify_shipment(&f.elf, f.base.statement().to_owned(), &proof, Some(&f.map))
        .err()
        .expect("no publishable file");
    assert!(
        err.contains("before publication") && err.contains("does not verify"),
        "{err}"
    );
}

#[test]
fn a_malformed_leaf_proof_is_an_error_not_a_panic() {
    let f = fixture();
    let [_, (child, proof)] = leaves();
    for part in ["roots", "answers", "chains", "rounds"] {
        let mut malformed = proof.clone();
        match part {
            "roots" => {
                malformed.proof.roots.pop().unwrap();
            }
            "answers" => {
                malformed.commit_ood.pop().unwrap();
            }
            "chains" => {
                malformed.proof.columns[0].polys.clear();
            }
            "rounds" => {
                malformed.proof.columns[0].polys[0].rounds.clear();
            }
            _ => unreachable!(),
        }
        assert!(wlfm_leg_arena(child, &malformed).is_err(), "{part}");
    }
    let program = leaf_program_with(&f.plan, 1, LeafChecks::ALL).unwrap();
    let arena = block_leaf_arena(&f.plan, &f.base, 1).unwrap();
    assert_eq!(leaf_words(&program, &arena), f.words[1]);
}

/// test_ecrec's three ECREC calls in windows of 32 cycles: the first call
/// before cycle 160, the second and third after it.
const ECREC_CUTS: [usize; 1] = [5];

fn ecrec_options() -> BlockOptions {
    let mut options = options();
    options.window_log2 = Some(5);
    options.max_rows.cpu = 64;
    options
}

/// test_ecrec's base cut at [`ECREC_CUTS`] by the one-process range
/// reference under `memory`, its ECREC tables tampered so (`None`: honest),
/// and its block verifier's verdict with `checks` (a refusal's reason, or an
/// error before the checks). The caller holds the ECREC forgery lock.
fn ecrec_base_verdict(
    memory: crate::tables::fused_memory::MemoryFormat,
    tamper: Option<crate::block_whir::EcrecTamper>,
    checks: &[VerifierChecks],
) -> Vec<Result<(), String>> {
    let elf = crate::test_utils::asm_elf_bytes(ECREC_PROGRAM);
    let format = pinned_format(FAMILY).unwrap().with_memory(memory);
    let opts = crate::lfm::proof::block_base_options_under(&format.zf);
    let previous = crate::block_whir::ECREC_TAMPER.with(|t| t.replace(tamper));
    let proved = prove_range_reference(
        &elf,
        &[],
        &opts,
        &format,
        &ecrec_options(),
        &ECREC_CUTS,
        Clock::default(),
    );
    crate::block_whir::ECREC_TAMPER.with(|t| t.set(previous));
    let (base, _, (map, _)) = proved.expect("the range reference proves what it built");
    assert_eq!(base.table_counts.ecrec, 2, "an ECREC table in each range");
    assert_eq!(
        map.expect("two ranges, two leaves").partition.num_leaves(),
        2
    );
    checks
        .iter()
        .map(|&checks| {
            crate::base_only::with_blake3_levels(|| {
                crate::block_whir::verify_block_whir_traced(
                    &base, &elf, &opts, &format, false, checks,
                )
            })
            .map_err(|e| format!("error: {e}"))?
            .map_err(|refusal| format!("refused: {refusal:?}"))
        })
        .collect()
}

const ECREC_PROGRAM: &str = "test_ecrec";

/// ★ A run cut into two ranges that each hold an ECREC call states two ECREC
/// tables, and its base verifies. The control of the three tampers below.
///
/// Mutation: ECREC left at one table in a block and the range assembly
/// refuses the count.
#[test]
#[ignore = "GPU tier: a two-range base of test_ecrec"]
fn two_ecrec_tables_from_two_ranges_verify() {
    let _free = crate::tables::trace_builder::ecrec_forge_free();
    assert_eq!(
        ecrec_base_verdict(SPLIT, None, &[VerifierChecks::ALL]),
        [Ok(())]
    );
}

const SPLIT: crate::tables::fused_memory::MemoryFormat =
    crate::tables::fused_memory::MemoryFormat::Split;

/// A tamper of the two-range ECREC layout under `memory`, proved over:
/// refused by the full verifier, and accepted with the bus balance switched
/// off, so that check is what refuses it (each group's local balance holds
/// either way).
fn refused_ecrec_tamper(
    memory: crate::tables::fused_memory::MemoryFormat,
    tamper: crate::block_whir::EcrecTamper,
) {
    let unbalanced = VerifierChecks {
        balance: false,
        ..VerifierChecks::ALL
    };
    let verdicts = ecrec_base_verdict(memory, Some(tamper), &[VerifierChecks::ALL, unbalanced]);
    println!("ECREC TAMPER {memory:?} {tamper:?}: {verdicts:?}");
    assert_eq!(
        verdicts[0],
        Err("refused: Proof(BusImbalance)".to_string()),
        "{tamper:?}"
    );
    assert_eq!(
        verdicts[1],
        Ok(()),
        "{tamper:?}: only the bus balance refuses it"
    );
}

/// ★ A call the CPU made in the second range with no ECREC row there (its
/// row made padding) is refused.
#[test]
#[ignore = "GPU tier: a two-range base of test_ecrec"]
fn an_ecrec_call_with_its_row_dropped_is_refused() {
    let _free = crate::tables::trace_builder::ecrec_forge_free();
    refused_ecrec_tamper(SPLIT, crate::block_whir::EcrecTamper::DropRow);
}

/// ★ A call with a row in both ECREC tables (the first range's row copied
/// into the second's padding) is refused.
#[test]
#[ignore = "GPU tier: a two-range base of test_ecrec"]
fn an_ecrec_row_in_both_tables_is_refused() {
    let _free = crate::tables::trace_builder::ecrec_forge_free();
    refused_ecrec_tamper(SPLIT, crate::block_whir::EcrecTamper::RowInBoth);
}

/// ★ A wrong key in the second ECREC table (a call's row keyed one CPU step
/// later) is refused.
#[test]
#[ignore = "GPU tier: a two-range base of test_ecrec"]
fn a_wrong_key_in_the_second_ecrec_table_is_refused() {
    let _free = crate::tables::trace_builder::ecrec_forge_free();
    refused_ecrec_tamper(SPLIT, crate::block_whir::EcrecTamper::WrongKey);
}

/// ★ Under the word memory format (the tables are ECREC_W's): a call with a
/// row in both ECREC_W tables (the first range's row, its `BAIL` with it,
/// copied into the second's padding) is refused by the bus balance, and only
/// by it.
#[test]
#[ignore = "GPU tier: a two-range base of test_ecrec"]
fn an_ecrec_w_row_in_both_tables_is_refused() {
    let _free = crate::tables::trace_builder::ecrec_forge_free();
    refused_ecrec_tamper(
        crate::tables::fused_memory::MemoryFormat::Words,
        crate::block_whir::EcrecTamper::RowInBoth,
    );
}

fn digest(bytes: &[u8]) -> String {
    format!("{:x}", sha2::Sha256::digest(bytes))
}

// The archive pin runs under deterministic settings even when the ordinary
// test-security suite was started without them.
fn deterministic_pin_process(test: &str) -> bool {
    if super::pin_process::deterministic() && last_wire() == Ok(WireFormat::Dedup) {
        return false;
    }
    super::pin_process::run_alone(test);
    true
}

#[test]
fn the_pin_process_fails_when_its_child_runs_no_test() {
    use super::pin_process::one_test_passed;
    let ran = "running 1 test\ntest a::b ... ok\n\ntest result: ok. 1 passed; 0 failed; 0 ignored; \
               0 measured; 3707 filtered out; finished in 1.00s\n";
    assert!(one_test_passed(ran));
    let none = "running 0 tests\n\ntest result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; \
                3708 filtered out; finished in 0.00s\n";
    assert!(!one_test_passed(none));
    let two = ran.replace("1 passed", "2 passed");
    assert!(!one_test_passed(&two));
    let missing = std::panic::catch_unwind(|| {
        super::pin_process::run_alone("lfm::cluster_levels::tests::no_test_has_this_name")
    });
    assert!(missing.is_err(), "a child that ran no test passed the pin");
}

#[test]
fn one_leaf_matches_pinned_level_digests_and_file_bytes() {
    if deterministic_pin_process(
        "lfm::cluster_levels::tests::one_leaf_matches_pinned_level_digests_and_file_bytes",
    ) {
        return;
    }
    let elf = crate::test_utils::asm_elf_bytes("test_commit_split");
    super::pin_process::check_pinned_elf(&elf);
    let format = pinned_format(FAMILY).unwrap();
    let opts = crate::lfm::proof::block_base_options_under(&format.zf);
    let run = levels::prove_block_levels(
        &elf,
        &[],
        &opts,
        &format,
        &options(),
        FAMILY,
        LevelStop::DEFAULT,
        true,
        true,
    )
    .unwrap();
    let default = encode_file(&run.proofs[1], LAST, WireFormat::Dedup, None).unwrap();
    let checks = FILE_CHECKS.with(std::cell::Cell::get);
    let reference = prove_range_levels_reference(&elf, &[], &options(), FAMILY, &[]).unwrap();
    assert_eq!(FILE_CHECKS.with(std::cell::Cell::get), checks + 1);
    assert_eq!(reference.file, default);
    assert_eq!(reference.file[4], crate::lfm::last_level::LAST_FILE_VERSION);
    let mut observed = run
        .proofs
        .iter()
        .map(|p| digest(&rkyv::to_bytes::<rkyv::rancor::Error>(p).unwrap()))
        .collect::<Vec<_>>();
    observed.push(digest(&default));
    assert_eq!(
        observed,
        [
            "85a8c12bec574d78a606018366dda3cec0ee16cb3fc2ceafaefbf88cbad12c5a",
            "9979ad7a0939d8b3815bfff2ed35ea6aae82a3d58d4f00792472f4d69c1235fa",
            "3c6ef0ba020dbf4f111aeb3b9099986bc16b90b1b909ff88c00979d32e8b55cf",
        ]
    );
}

#[test]
fn two_leaf_program_ids_are_pinned() {
    if deterministic_pin_process("lfm::cluster_levels::tests::two_leaf_program_ids_are_pinned") {
        return;
    }
    let f = fixture();
    super::pin_process::check_pinned_elf(&f.elf);
    let children = (0..2)
        .map(|leaf| leaf_child(&f.plan, &build_leaf(&f.plan, leaf, true).unwrap().build))
        .collect::<Vec<_>>();
    let join = join_level(
        &children,
        &f.plan.child_layout(),
        LevelStop::DEFAULT,
        LAST,
        level_rounds(&f.plan),
        true,
    )
    .unwrap();
    let ids = children
        .iter()
        .map(|c| c.artifacts.program_id)
        .chain([join.build.artifacts.program_id])
        .map(|id| {
            id.iter()
                .map(|byte| format!("{byte:02x}"))
                .collect::<String>()
        })
        .collect::<Vec<_>>();
    assert_eq!(
        ids,
        [
            "353c00c462c0b4983c001c866a96665c1ac0bb60e82921f53c0dbf594b2ef55e",
            "0326289ec24917f75885b3222348a7da70e28a63f68d5b0ab8241aaa6bdc2674",
            "71a74a4f8a71c1fd904c7d0baba67911a87932214ced7d7045c788eb7134c243",
        ]
    );
}

fn leaf_map(groups: usize, leaves: usize) -> LeafMap {
    let bytes: Vec<u8> = (0..groups).map(|g| g.min(leaves - 1) as u8).collect();
    LeafMap::decode(&bytes, groups).unwrap()
}

fn proved_leaves(plan: &WhirBlockPlan, base: &AnyBlockWhirProof) -> Vec<ProvedLeaf> {
    (0..plan.partition().num_leaves())
        .map(|leaf| prove_leaf(plan, leaf, &block_leaf_arena(plan, base, leaf).unwrap()).unwrap())
        .collect()
}

#[test]
fn a_proof_under_the_other_versions_header_is_refused() {
    let f = fixture();
    let verify_v1 =
        |file: &[u8]| levels::verify_block_levels_file(&f.elf, f.base.statement(), FAMILY, file);
    let verify =
        |file: &[u8]| verify_cluster_block_levels_file(&f.elf, f.base.statement(), FAMILY, file);
    // The join's proof under a version-1 header, at both entries.
    let join_v1 = encode_file(join_proof(), LAST, WireFormat::Dedup, None).unwrap();
    assert_eq!(join_v1[4], LAST_FILE_VERSION);
    for error in [
        verify_v1(&join_v1).unwrap_err(),
        verify(&join_v1).unwrap_err(),
    ] {
        assert!(
            !error.contains("version"),
            "refused past the header: {error}"
        );
    }
    verify(&join_file()).expect("the same proof under its own header");
    // A one-leaf proof under a version-2 header with a valid map.
    let plan = plan_for(&f.elf, &f.base, None);
    let single = prove_single(&plan, &block_leaf_arena(&plan, &f.base, 0).unwrap(), LAST).unwrap();
    let single_v1 = encode_file(&single, LAST, WireFormat::Dedup, None).unwrap();
    verify_v1(&single_v1).expect("the one-leaf proof under its own header");
    verify(&single_v1).expect("the cluster entry reads version 1 as one leaf");
    let single_v2 = encode_file(&single, LAST, WireFormat::Dedup, Some(&f.map)).unwrap();
    let error = verify(&single_v2).unwrap_err();
    assert!(error.contains("does not verify"), "{error}");
}

#[test]
fn three_leaves_refuse_a_tamper_in_the_last() {
    let f = fixture();
    let map = leaf_map(f.base.groups.len(), 3);
    let plan = plan_for(&f.elf, &f.base, Some(&map));
    assert_eq!(plan.partition().num_leaves(), 3);
    let layout = plan.child_layout();
    let words: Vec<_> = (0..3)
        .map(|leaf| {
            leaf_words(
                &plan.leaf_program(leaf).unwrap(),
                &block_leaf_arena(&plan, &f.base, leaf).unwrap(),
            )
        })
        .collect();
    let honest = bindings(&layout, &words, BlockBindings::ALL).unwrap();
    assert_eq!(honest, words[0][..layout.total() - 1]);
    let mut tampers = Vec::new();
    for word in 0..2 {
        for lane in 0..4 {
            tampers.push((layout.state(word), lane, "state"));
        }
    }
    for half in 0..2 {
        tampers.push((layout.id(half), 0, "id"));
    }
    tampers.push((layout.sum(), 0, "bus"));
    tampers.push((layout.out_half(0), 0, "out"));
    for (index, lane, check) in tampers {
        let mut tampered = words.clone();
        tampered[2][index][lane] += FE::from(1u64);
        refused_binding(&layout, &tampered, check);
    }
    // Whole proofs: the last leaf proved for another proof of the block.
    let (elf, base) = other_proof();
    let other = plan_for(elf, base, Some(&map));
    assert_eq!(other.id(), plan.id());
    let proved = proved_leaves(&plan, &f.base);
    let foreign = prove_leaf(&other, 2, &block_leaf_arena(&other, base, 2).unwrap()).unwrap();
    let children: Vec<_> = proved.iter().map(|leaf| leaf.0.clone()).collect();
    let join = join_program(&children, &layout).unwrap();
    run_join(
        &children,
        &[&proved[0].1, &proved[1].1, &proved[2].1],
        &join,
    )
    .expect("three honest leaves");
    assert!(run_join(&children, &[&proved[0].1, &proved[1].1, &foreign.1], &join).is_err());
}

fn many_groups() -> &'static (Vec<u8>, AnyBlockWhirProof) {
    static MANY: OnceLock<(Vec<u8>, AnyBlockWhirProof)> = OnceLock::new();
    MANY.get_or_init(|| fixture_block_cut("all_instructions_64", &[], &[1, 2, 3, 4, 5, 6, 7]))
}

#[test]
fn eight_leaves_are_accepted() {
    let (elf, base) = many_groups();
    let groups = base.groups.len();
    assert!(groups >= MAX_CLUSTER_LEAVES, "{groups} groups");
    let map = leaf_map(groups, MAX_CLUSTER_LEAVES);
    let plan = plan_for(elf, base, Some(&map));
    assert_eq!(plan.partition().num_leaves(), MAX_CLUSTER_LEAVES);
    let proof = prove_join(
        &plan.child_layout(),
        level_rounds(&plan),
        proved_leaves(&plan, base),
        LAST,
    )
    .unwrap();
    let file = encode_file(&proof, LAST, WireFormat::Dedup, Some(&map)).unwrap();
    verify_cluster_block_levels_file(elf, base.statement(), FAMILY, &file).unwrap();
    let mut moved = map.as_bytes().to_vec();
    moved[groups - 1] = 0;
    let moved = LeafMap::decode(&moved, groups).unwrap();
    let altered = encode_file(&proof, LAST, WireFormat::Dedup, Some(&moved)).unwrap();
    assert!(verify_cluster_block_levels_file(elf, base.statement(), FAMILY, &altered).is_err());
}

#[test]
fn a_malformed_file_is_an_error_at_the_cluster_entry() {
    let f = fixture();
    let verify =
        |file: &[u8]| verify_cluster_block_levels_file(&f.elf, f.base.statement(), FAMILY, file);
    let file = join_file();
    verify(&file).unwrap();
    let body = LAST_FILE_HEADER + 4 + f.map.as_bytes().len() + 32;
    for end in [body, body + 1, (body + file.len()) / 2, file.len() - 1] {
        assert!(verify(&file[..end]).is_err(), "a body cut at {end}");
    }
    for wire in [WireFormat::Rkyv, WireFormat::Compact, WireFormat::Dedup] {
        let file = encode_file(join_proof(), LAST, wire, Some(&f.map)).unwrap();
        verify(&file).unwrap_or_else(|e| panic!("{wire:?}: {e}"));
        for extra in [&[0u8][..], &[0xff; 64]] {
            let mut long = file.clone();
            long.extend_from_slice(extra);
            assert!(
                verify(&long).is_err(),
                "{wire:?} with {} trailing bytes",
                extra.len()
            );
        }
    }
    let proof = join_proof();
    let mut fewer = proof.clone();
    fewer.public_words.pop();
    let mut more = proof.clone();
    let extra = (more.public_words.len() as u32, more.public_words[0].1);
    more.public_words.push(extra);
    let mut none = proof.clone();
    none.public_words.clear();
    for (name, proof) in [("fewer", fewer), ("more", more), ("no", none)] {
        let file = encode_file(&proof, LAST, WireFormat::Dedup, Some(&f.map)).unwrap();
        assert!(verify(&file).is_err(), "{name} public words");
    }
    for wire in [0u8, 1, 3, 255] {
        let mut other = file.clone();
        other[6] = wire;
        assert!(
            verify(&other).is_err(),
            "a dedup body read as encoding {wire}"
        );
    }
}

/// Range mode has no one-leaf file: the first range always owns its cycles'
/// groups and the last range the finishing tables', so even a first range
/// that runs every cycle ships two leaves. One active owner would ship
/// version 1, with no map.
#[test]
fn a_range_that_runs_every_cycle_still_ships_two_leaves() {
    let elf = crate::test_utils::asm_elf_bytes("test_commit_split");
    let checks = FILE_CHECKS.with(std::cell::Cell::get);
    let shipped = prove_range_levels_reference(&elf, &[], &options(), FAMILY, &[64]).unwrap();
    assert_eq!(FILE_CHECKS.with(std::cell::Cell::get), checks + 1);
    assert_eq!(shipped.file[4], CLUSTER_LAST_FILE_VERSION);
    let (map, _, _) =
        cluster_file_parts(&shipped.file, LAST, shipped.statement.groups.len()).unwrap();
    assert_eq!(map.unwrap().partition.num_leaves(), 2);
    verify_cluster_block_levels_file(&elf, shipped.statement.view(), FAMILY, &shipped.file)
        .expect("the cluster entry accepts");
    assert!(
        levels::verify_block_levels_file(&elf, shipped.statement.view(), FAMILY, &shipped.file)
            .unwrap_err()
            .contains("version")
    );
    let (map, order) = LeafMap::from_owners(&[1, 1, 1], 3).unwrap();
    assert!(map.is_none());
    assert_eq!(order, [1]);
}

/// Every lean and LFM chunk height of a worker's stated shapes.
fn rows_mut<'a>(
    chunks: &'a mut [Vec<usize>],
    lean: &'a mut B3lShape,
) -> impl Iterator<Item = &'a mut usize> {
    chunks.iter_mut().flatten().chain(
        lean.rounds
            .iter_mut()
            .chain(&mut lean.heads)
            .chain(&mut lean.muxes)
            .chain(&mut lean.caps)
            .chain(&mut lean.cap_reads),
    )
}

type Reshape = fn(&mut usize, &mut [bool; 3], &mut Vec<Vec<usize>>, &mut B3lShape);

/// Shapes a hostile worker may state for its leaf; the first is the honest
/// leaf's shapes with placeholder prepared roots.
fn hostile_reshapes() -> [(&'static str, Reshape); 6] {
    [
        ("placeholder prepared roots", |_, _, _, _| {}),
        ("one row a table", |_, _, chunks, lean| {
            rows_mut(chunks, lean).for_each(|n| *n = 1)
        }),
        ("the tallest table", |_, _, chunks, lean| {
            rows_mut(chunks, lean)
                .for_each(|n| *n = 1 << crate::multilinear_prove::MAX_STATED_TABLE_VARS)
        }),
        ("the most chunks", |_, _, chunks, lean| {
            let rows = rows_mut(chunks, lean).count();
            chunks[0] = vec![2; 4096 - rows + chunks[0].len()];
        }),
        ("the other hash chunk count", |hash, _, _, _| {
            *hash = 3 - *hash
        }),
        ("every chip", |_, chips, _, _| *chips = [true; 3]),
    ]
}

#[test]
fn hostile_self_consistent_worker_shapes_are_an_error_not_a_panic() {
    let f = fixture();
    let layout = f.plan.child_layout();
    let rounds = level_rounds(&f.plan);
    let [a, b] = leaves();
    // Control: the honest worker's artifacts ship.
    let honest = LeafArtifacts::from_build(&b.0.artifacts)
        .into_artifacts(rounds)
        .unwrap();
    let honest = WlfmChild {
        artifacts: honest,
        ..b.0.clone()
    };
    let proof = prove_join(
        &layout,
        rounds,
        vec![a.clone(), (honest, b.1.clone())],
        LAST,
    )
    .unwrap();
    verify_shipment(&f.elf, f.base.statement().to_owned(), &proof, Some(&f.map)).unwrap();
    let mut admitted = 0;
    for (name, reshape) in hostile_reshapes() {
        let forged =
            match LeafArtifacts::from_build(&b.0.artifacts).self_consistent(rounds, reshape) {
                Ok(forged) => forged,
                Err(error) => {
                    eprintln!("{name}: not admitted ({error})");
                    continue;
                }
            };
        let artifacts = forged
            .into_artifacts(rounds)
            .unwrap_or_else(|e| panic!("{name}: self-consistent shapes refused: {e}"));
        admitted += 1;
        let child = WlfmChild {
            artifacts,
            num_public_words: layout.total(),
            cap_mux: Some(CapMux::Packed),
        };
        // The join compiler returns a program or an error.
        let compiled = join_program(&[a.0.clone(), child.clone()], &layout).map(|_| ());
        eprintln!("{name}: the join compiler returned {compiled:?}");
        let shipped = prove_join(&layout, rounds, vec![a.clone(), (child, b.1.clone())], LAST)
            .and_then(|proof| {
                verify_shipment(&f.elf, f.base.statement().to_owned(), &proof, Some(&f.map))
                    .map(|_| ())
            });
        assert!(shipped.is_err(), "{name}: a file shipped");
    }
    assert!(admitted >= 4, "only {admitted} shapes were admitted");
}

/// This process's resident set and its peak since the last reset, in KiB
/// (Linux only).
fn resident_kib() -> Option<(u64, u64)> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    let field = |name: &str| {
        status
            .lines()
            .find_map(|line| line.strip_prefix(name))?
            .trim()
            .strip_suffix(" kB")?
            .trim()
            .parse()
            .ok()
    };
    Some((field("VmRSS:")?, field("VmHWM:")?))
}

/// Every stated table at `2^VARS` rows.
#[allow(clippy::ptr_arg)] // a `Reshape`
fn tallest_at<const VARS: u32>(
    _: &mut usize,
    _: &mut [bool; 3],
    chunks: &mut Vec<Vec<usize>>,
    lean: &mut B3lShape,
) {
    rows_mut(chunks, lean).for_each(|n| *n = 1 << VARS);
}

/// `TABLES` stated tables, each at `2^VARS` rows.
fn stated_at<const TABLES: usize, const VARS: u32>(
    hash: &mut usize,
    chips: &mut [bool; 3],
    chunks: &mut Vec<Vec<usize>>,
    lean: &mut B3lShape,
) {
    tallest_at::<VARS>(hash, chips, chunks, lean);
    let others = rows_mut(chunks, lean).count() - chunks[0].len();
    chunks[0] = vec![1 << VARS; TABLES.saturating_sub(others).max(1)];
}

/// 4096 stated tables, each at `2^VARS` rows.
fn most_chunks_at<const VARS: u32>(
    hash: &mut usize,
    chips: &mut [bool; 3],
    chunks: &mut Vec<Vec<usize>>,
    lean: &mut B3lShape,
) {
    stated_at::<4096, VARS>(hash, chips, chunks, lean);
}

/// The early join builds the join program from the shapes a worker states,
/// before any proof and beside the coordinator's own leaf. This prints the
/// cost of `build_join` (compile and build) on each shape `into_artifacts`
/// admits, the largest at once included: seconds and the peak resident set
/// above the start (Linux, after `/proc/self/clear_refs` resets the peak).
#[test]
#[ignore = "measurement: the join build on every admitted worker shape"]
fn the_join_build_cost_of_every_admitted_worker_shape() {
    let f = fixture();
    let layout = f.plan.child_layout();
    let rounds = level_rounds(&f.plan);
    let [a, b] = leaves();
    let sized: [(&str, Reshape); 16] = [
        ("59 stated tables at 2^22 rows", stated_at::<59, 22>),
        ("118 stated tables at 2^22 rows", stated_at::<118, 22>),
        ("110 stated tables at 2^24 rows", stated_at::<110, 24>),
        ("118 stated tables at 2^24 rows", stated_at::<118, 24>),
        ("every table at 2^2 rows", tallest_at::<2>),
        ("every table at 2^8 rows", tallest_at::<8>),
        ("every table at 2^16 rows", tallest_at::<16>),
        ("every table at 2^20 rows", tallest_at::<20>),
        ("every table at 2^24 rows", tallest_at::<24>),
        ("every table at 2^29 rows", tallest_at::<29>),
        ("every table at 2^30 rows", tallest_at::<30>),
        ("the most chunks at 2^2 rows", most_chunks_at::<2>),
        ("the most chunks at 2^8 rows", most_chunks_at::<8>),
        ("the most chunks at 2^16 rows", most_chunks_at::<16>),
        ("the most chunks at 2^20 rows", most_chunks_at::<20>),
        ("the most chunks at 2^29 rows", most_chunks_at::<29>),
    ];
    let cases = [("the honest leaf's own artifacts", None)]
        .into_iter()
        .chain(
            hostile_reshapes()
                .into_iter()
                .chain(sized)
                .map(|(name, reshape)| (name, Some(reshape))),
        );
    // One process a case past a memory cap: `LV_JOIN_COST_ONLY` names it.
    let only = std::env::var("LV_JOIN_COST_ONLY").ok();
    for (name, reshape) in cases {
        if only.as_deref().is_some_and(|only| only != name) {
            continue;
        }
        let artifacts = match reshape {
            None => Ok(LeafArtifacts::from_build(&b.0.artifacts)),
            Some(reshape) => {
                LeafArtifacts::from_build(&b.0.artifacts).self_consistent(rounds, reshape)
            }
        }
        .and_then(|artifacts| artifacts.into_artifacts(rounds));
        let artifacts = match artifacts {
            Ok(artifacts) => artifacts,
            Err(error) => {
                println!("JOIN BUILD COST {name}: not admitted ({error})");
                continue;
            }
        };
        let child = WlfmChild {
            artifacts,
            num_public_words: layout.total(),
            cap_mux: Some(CapMux::Packed),
        };
        let children = [a.0.clone(), child];
        let reset = std::fs::write("/proc/self/clear_refs", "5").is_ok();
        let before = resident_kib();
        let start = std::time::Instant::now();
        let built = build_join(&children, &layout, rounds, LAST, Clock::default());
        let seconds = start.elapsed().as_secs_f64();
        let after = resident_kib();
        let peak = match (reset, before, after) {
            (true, Some((rss, _)), Some((_, hwm))) => {
                format!("{:.1} MiB", hwm.saturating_sub(rss) as f64 / 1024.0)
            }
            _ => "unknown".into(),
        };
        let outcome = built.as_ref().map(|_| "built").map_err(String::as_str);
        println!(
            "JOIN BUILD COST {name}: {outcome:?} in {seconds:.3} s, peak +{peak}, \
             {} tables, tallest 2^{}",
            children[1].artifacts.table_num_vars.len(),
            children[1]
                .artifacts
                .table_num_vars
                .iter()
                .max()
                .copied()
                .unwrap_or(0)
        );
        drop(built);
    }
}

#[cfg(feature = "cuda")]
mod tcp {
    use super::*;
    use crate::block_whir::cluster_tcp::{Topology, serve};
    use std::{
        fs::{self, File},
        net::TcpListener,
        path::PathBuf,
        process::{Child, Command, Stdio},
    };

    struct Workers {
        dir: PathBuf,
        children: Vec<Child>,
    }

    impl Drop for Workers {
        fn drop(&mut self) {
            for child in &mut self.children {
                let _ = child.kill();
                let _ = child.wait();
            }
            let _ = fs::remove_dir_all(&self.dir);
        }
    }

    fn proof(cuts: &[usize], fault: Option<&str>) -> Result<VerifiedClusterFile, String> {
        proof_with(cuts, fault, false)
    }

    /// With `fallback`, the topology names the local fallback and a faulted
    /// worker may exit with its error.
    fn proof_with(
        cuts: &[usize],
        fault: Option<&str>,
        fallback: bool,
    ) -> Result<VerifiedClusterFile, String> {
        proof_of("test_commit_split", &options(), cuts, fault, fallback)
    }

    /// [`proof_with`] of program `name` under `options`.
    fn proof_of(
        name: &str,
        options: &BlockOptions,
        cuts: &[usize],
        fault: Option<&str>,
        fallback: bool,
    ) -> Result<VerifiedClusterFile, String> {
        let listeners = (0..=cuts.len())
            .map(|_| TcpListener::bind("127.0.0.1:0").unwrap())
            .collect::<Vec<_>>();
        let addr = |i: usize| listeners[i].local_addr().unwrap();
        let mut text = format!("coordinator {}\n", addr(0));
        for id in 0..cuts.len() {
            text.push_str(&format!("worker {} 0 {:064x}\n", addr(id + 1), id + 1));
        }
        text.push_str(&format!(
            "windows {}\ndeadlines-ms 12000 60000 90000\n",
            cuts.iter()
                .map(usize::to_string)
                .collect::<Vec<_>>()
                .join(",")
        ));
        if fallback {
            text.push_str("fallback local\n");
        }
        let dir = std::env::temp_dir().join(format!(
            "lv-s5b-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir(&dir).unwrap();
        let mut workers = Workers {
            dir,
            children: Vec::new(),
        };
        let path = workers.dir.join("topology");
        fs::write(&path, &text).unwrap();
        let topology = Topology::parse(&text).unwrap();
        drop(listeners);
        for id in 0..cuts.len() {
            let log = File::create(workers.dir.join(format!("worker-{id}.log"))).unwrap();
            let mut command = Command::new(std::env::current_exe().unwrap());
            command.args([
                "--ignored",
                "--exact",
                "lfm::cluster_levels::tests::tcp::range_levels_worker",
                "--test-threads=1",
                "--nocapture",
            ]);
            command
                .env("LV_S5B_TOPOLOGY", &path)
                .env("LV_S5B_WORKER", id.to_string())
                .env("LAMBDA_VM_CARDS", "0")
                .env("RAYON_NUM_THREADS", "2")
                .env_remove("LAMBDA_VM_CLUSTER_TEST_FAULT")
                .stdout(Stdio::from(log.try_clone().unwrap()))
                .stderr(Stdio::from(log));
            if id == 0
                && let Some(fault) = fault
            {
                command.env("LAMBDA_VM_CLUSTER_TEST_FAULT", fault);
            }
            workers.children.push(command.spawn().unwrap());
        }
        let elf = crate::test_utils::asm_elf_bytes(name);
        let result = prove_cluster_block_levels(&elf, &[], options, FAMILY, &topology);
        for (id, child) in workers.children.iter_mut().enumerate() {
            let status = child.wait().unwrap();
            if !status.success() || result.is_err() {
                let log = fs::read_to_string(workers.dir.join(format!("worker-{id}.log"))).unwrap();
                eprintln!("STAGE 5B worker {id}: {status}\n{log}");
            }
            assert!(
                status.success() || fallback && fault.is_some(),
                "worker {id}: {status}, coordinator {:?}",
                result.as_ref().err()
            );
        }
        result
    }

    #[test]
    #[ignore = "cluster subprocess entry"]
    fn range_levels_worker() {
        let path = std::env::var_os("LV_S5B_TOPOLOGY").unwrap();
        let topology = Topology::read(std::path::Path::new(&path)).unwrap();
        let id = std::env::var("LV_S5B_WORKER").unwrap().parse().unwrap();
        let format = pinned_format(FAMILY).unwrap();
        let opts = crate::lfm::proof::block_base_options_under(&format.zf);
        serve(&topology, id, &opts, &format).unwrap();
    }

    fn identity(cuts: &[usize], leaves: usize) {
        assert_eq!(
            std::env::var("LAMBDA_VM_DETERMINISTIC_GRIND").as_deref(),
            Ok("1")
        );
        assert_eq!(
            std::env::var("LAMBDA_VM_FIXED_TRACE_HASH").as_deref(),
            Ok("1")
        );
        let checks = FILE_CHECKS.with(std::cell::Cell::get);
        let actual = proof(cuts, None).unwrap();
        assert_eq!(FILE_CHECKS.with(std::cell::Cell::get), checks + 1);
        let elf = crate::test_utils::asm_elf_bytes("test_commit_split");
        let expected = prove_range_levels_reference(&elf, &[], &options(), FAMILY, cuts).unwrap();
        assert_eq!(FILE_CHECKS.with(std::cell::Cell::get), checks + 2);
        assert_eq!(actual.statement.public_output, [0xaa, 0xbb, 0xcc, 0xdd]);
        assert_eq!(
            rkyv::to_bytes::<rkyv::rancor::Error>(&actual.statement)
                .unwrap()
                .as_slice(),
            rkyv::to_bytes::<rkyv::rancor::Error>(&expected.statement)
                .unwrap()
                .as_slice()
        );
        assert_eq!(digest(&actual.file), digest(&expected.file));
        assert_eq!(actual.file, expected.file);
        let (map, _, _) =
            cluster_file_parts(&actual.file, LAST, actual.statement.groups.len()).unwrap();
        assert_eq!(map.map_or(1, |map| map.partition.num_leaves()), leaves);
    }

    #[test]
    #[ignore = "GPU tier: two one-card processes and the serial reference"]
    fn two_leaf_tcp_matches_one_process_reference() {
        identity(&[2], 2);
    }

    #[test]
    #[ignore = "GPU tier: four one-card processes and the serial reference"]
    fn four_leaf_tcp_matches_one_process_reference() {
        identity(&[2, 4, 6], 4);
    }

    /// ★ test_ecrec cut so each of two ranges holds an ECREC call: over TCP
    /// and in one process the statement states two ECREC tables, the base,
    /// both leaves, the join and the version-2 file prove and verify, and the
    /// statement and file bytes are equal.
    ///
    /// Mutation: ECREC left at one table in a block and both runs refuse the
    /// count at range assembly.
    #[test]
    #[ignore = "GPU tier: two one-card processes and the serial reference"]
    fn ecrec_in_two_ranges_tcp_matches_one_process_reference() {
        let _free = crate::tables::trace_builder::ecrec_forge_free();
        for var in [
            "LAMBDA_VM_DETERMINISTIC_GRIND",
            "LAMBDA_VM_FIXED_TRACE_HASH",
        ] {
            assert_eq!(std::env::var(var).as_deref(), Ok("1"), "{var}");
        }
        let checks = FILE_CHECKS.with(std::cell::Cell::get);
        let actual = proof_of(ECREC_PROGRAM, &ecrec_options(), &ECREC_CUTS, None, false).unwrap();
        assert_eq!(FILE_CHECKS.with(std::cell::Cell::get), checks + 1);
        let elf = crate::test_utils::asm_elf_bytes(ECREC_PROGRAM);
        let expected =
            prove_range_levels_reference(&elf, &[], &ecrec_options(), FAMILY, &ECREC_CUTS).unwrap();
        assert_eq!(FILE_CHECKS.with(std::cell::Cell::get), checks + 2);
        assert_eq!(actual.statement.table_counts.ecrec, 2);
        assert_eq!(actual.statement.public_output.len(), 3 * 72);
        assert_eq!(
            rkyv::to_bytes::<rkyv::rancor::Error>(&actual.statement)
                .unwrap()
                .as_slice(),
            rkyv::to_bytes::<rkyv::rancor::Error>(&expected.statement)
                .unwrap()
                .as_slice()
        );
        assert_eq!(digest(&actual.file), digest(&expected.file));
        assert_eq!(actual.file, expected.file);
        assert_eq!(&actual.file[..LAST_FILE_MAGIC.len()], LAST_FILE_MAGIC);
        assert_eq!(
            actual.file[LAST_FILE_MAGIC.len()],
            CLUSTER_LAST_FILE_VERSION
        );
        let (map, _, _) =
            cluster_file_parts(&actual.file, LAST, actual.statement.groups.len()).unwrap();
        assert_eq!(map.map_or(1, |map| map.partition.num_leaves()), 2);
        verify_cluster_block_levels_file(&elf, actual.statement.view(), FAMILY, &actual.file)
            .expect("the file verifies");
        println!(
            "ECREC TWO RANGES: file {} B, {}",
            actual.file.len(),
            digest(&actual.file)
        );
    }

    #[test]
    #[ignore = "GPU tier: a two-leaf one-process reference (134 s on the host)"]
    fn the_range_reference_checks_its_file_before_returning_it() {
        let elf = crate::test_utils::asm_elf_bytes("test_commit_split");
        let checks = FILE_CHECKS.with(std::cell::Cell::get);
        let reference = prove_range_levels_reference(&elf, &[], &options(), FAMILY, &[2]).unwrap();
        assert_eq!(FILE_CHECKS.with(std::cell::Cell::get), checks + 1);
        let (map, _, _) =
            cluster_file_parts(&reference.file, LAST, reference.statement.groups.len()).unwrap();
        assert_eq!(map.map_or(1, |map| map.partition.num_leaves()), 2);
        verify_cluster_block_levels_file(&elf, reference.statement.view(), FAMILY, &reference.file)
            .expect("the reference file verifies");
    }

    #[test]
    #[ignore = "GPU tier: a failed worker and the one-process fallback"]
    fn a_failed_range_worker_falls_back_to_a_checked_reference() {
        let checks = FILE_CHECKS.with(std::cell::Cell::get);
        let fallback = proof_with(&[2], Some("missing-range"), true)
            .expect("the local fallback returns a file");
        // The failed attempt returned no file. The fallback reference comes
        // back unchecked, and the checked entry checks it once.
        assert_eq!(FILE_CHECKS.with(std::cell::Cell::get), checks + 1);
        let elf = crate::test_utils::asm_elf_bytes("test_commit_split");
        verify_cluster_block_levels_file(&elf, fallback.statement.view(), FAMILY, &fallback.file)
            .expect("the fallback file verifies");
        let (map, _, _) =
            cluster_file_parts(&fallback.file, LAST, fallback.statement.groups.len()).unwrap();
        assert_eq!(map.map_or(1, |map| map.partition.num_leaves()), 2);
    }

    #[test]
    #[ignore = "GPU tier: a real worker proves a weakened program"]
    fn tcp_weakened_worker_returns_no_file() {
        let error = proof(&[2], Some("weakened-leaf"))
            .err()
            .expect("nothing to publish");
        assert!(
            error.contains("before publication") && error.contains("does not verify"),
            "{error}"
        );
    }

    fn knob(name: &str) -> bool {
        std::env::var(name).as_deref() == Ok("1")
    }

    /// The early-frame cases run in the tier's pass with the opt-in knobs set.
    fn assert_early_join() {
        use crate::block_whir::cluster_tcp::EARLY_JOIN_ENV;
        assert!(knob(EARLY_JOIN_ENV), "run with {EARLY_JOIN_ENV}=1");
        for var in [
            "LAMBDA_VM_DETERMINISTIC_GRIND",
            "LAMBDA_VM_FIXED_TRACE_HASH",
        ] {
            assert!(knob(var), "{var}");
        }
    }

    /// The one-process reference of the two-leaf cut, proved once a process.
    fn reference_file() -> &'static [u8] {
        static FILE: OnceLock<Vec<u8>> = OnceLock::new();
        FILE.get_or_init(|| {
            let elf = crate::test_utils::asm_elf_bytes("test_commit_split");
            prove_range_levels_reference(&elf, &[], &options(), FAMILY, &[2])
                .unwrap()
                .file
        })
    }

    /// The attempt with worker 0's `fault` fails with `reason`, and the
    /// fallback's checked file is the one-process reference, byte for byte.
    fn falls_back(fault: Option<&str>, reason: &str) {
        use crate::block_whir::cluster_tcp::local_fallbacks;
        let fallbacks = local_fallbacks();
        let seen = FALLBACK_ERRORS.with(|errors| errors.borrow().len());
        let checks = FILE_CHECKS.with(std::cell::Cell::get);
        let file = proof_with(&[2], fault, true)
            .unwrap_or_else(|e| panic!("{fault:?}: the fallback returned no file: {e}"));
        assert_eq!(FILE_CHECKS.with(std::cell::Cell::get), checks + 1);
        assert_eq!(local_fallbacks(), fallbacks + 1, "{fault:?}: no fallback");
        let errors = FALLBACK_ERRORS.with(|errors| errors.borrow()[seen..].to_vec());
        assert!(
            errors.len() == 1 && errors[0].contains(reason),
            "{fault:?}: expected {reason:?}, fell back on {errors:?}"
        );
        assert_eq!(file.file, reference_file(), "{fault:?}");
        let elf = crate::test_utils::asm_elf_bytes("test_commit_split");
        verify_cluster_block_levels_file(&elf, file.statement.view(), FAMILY, &file.file)
            .expect("the fallback file verifies");
    }

    /// Sets the coordinator's early join build fault for one test.
    struct JoinBuildFault;

    impl JoinBuildFault {
        fn set(fault: u8) -> Self {
            use crate::block_whir::cluster_tcp::JOIN_BUILD_FAULT;
            JOIN_BUILD_FAULT.store(fault, std::sync::atomic::Ordering::SeqCst);
            Self
        }
    }

    impl Drop for JoinBuildFault {
        fn drop(&mut self) {
            use crate::block_whir::cluster_tcp::JOIN_BUILD_FAULT;
            JOIN_BUILD_FAULT.store(0, std::sync::atomic::Ordering::SeqCst);
        }
    }

    #[test]
    #[ignore = "GPU tier (default knobs): early artifacts nobody asked for"]
    fn early_artifacts_are_refused_when_early_join_is_off() {
        use crate::block_whir::cluster_tcp::EARLY_JOIN_ENV;
        assert!(!knob(EARLY_JOIN_ENV), "run with {EARLY_JOIN_ENV} unset");
        falls_back(Some("early-unasked"), "unexpected early leaf artifacts");
    }

    /// Worker 0 states one table above the shape cap, then more tables than
    /// the cap (both self-consistent): each attempt is refused for the cap,
    /// no join is built from the worker's shapes, and the fallback ships the
    /// reference.
    ///
    /// Mutation: without the cap the join is built from those shapes.
    fn refused_above_the_shape_cap() {
        use crate::block_whir::cluster_tcp::WORKER_JOIN_BUILDS;
        use std::sync::atomic::Ordering::SeqCst;
        for fault in ["oversized-leaf", "too-many-tables-leaf"] {
            let builds = WORKER_JOIN_BUILDS.load(SeqCst);
            falls_back(Some(fault), "worker leaf shapes above the cap");
            assert_eq!(
                WORKER_JOIN_BUILDS.load(SeqCst),
                builds,
                "{fault}: a join was built from the worker's shapes"
            );
        }
    }

    #[test]
    #[ignore = "GPU tier (default knobs): a worker leaf above the shape cap"]
    fn a_worker_leaf_above_the_shape_cap_never_reaches_the_join() {
        use crate::block_whir::cluster_tcp::EARLY_JOIN_ENV;
        assert!(!knob(EARLY_JOIN_ENV), "run with {EARLY_JOIN_ENV} unset");
        refused_above_the_shape_cap();
    }

    #[test]
    #[ignore = "GPU tier (opt-in knobs): early artifacts above the shape cap"]
    fn early_artifacts_above_the_shape_cap_never_reach_the_join() {
        assert_early_join();
        refused_above_the_shape_cap();
    }

    #[test]
    #[ignore = "GPU tier (opt-in knobs): early artifacts sent twice"]
    fn early_join_refuses_duplicate_artifacts() {
        assert_early_join();
        falls_back(
            Some("early-duplicate"),
            "unexpected, duplicate or stale worker leaf",
        );
    }

    #[test]
    #[ignore = "GPU tier (opt-in knobs): the leaf proof before its artifacts"]
    fn early_join_refuses_a_proof_before_its_artifacts() {
        assert_early_join();
        falls_back(
            Some("early-proof-first"),
            "leaf proof arrived before its artifacts",
        );
    }

    #[test]
    #[ignore = "GPU tier (opt-in knobs): early artifacts bound to another plan"]
    fn early_join_refuses_artifacts_for_another_plan() {
        assert_early_join();
        falls_back(
            Some("early-other-plan"),
            "unexpected, duplicate or stale worker leaf",
        );
    }

    #[test]
    #[ignore = "GPU tier (opt-in knobs): early artifacts and no proof"]
    fn a_worker_that_sends_artifacts_and_no_proof_falls_back_to_a_checked_reference() {
        assert_early_join();
        falls_back(Some("early-artifacts-only"), "missing worker 0 leaf");
    }

    #[test]
    #[ignore = "GPU tier (opt-in knobs): early artifacts, then the worker exits"]
    fn a_worker_that_exits_after_its_artifacts_falls_back_to_a_checked_reference() {
        assert_early_join();
        falls_back(Some("early-artifacts-exit"), "worker 0 leaf");
    }

    /// Self-consistent artifacts of another program (placeholder prepared
    /// roots, the id they hash to), then the honest proof: the join of those
    /// artifacts does not run over that proof, and the attempt falls back.
    #[test]
    #[ignore = "GPU tier (opt-in knobs): early artifacts of another program"]
    fn early_artifacts_of_another_program_never_ship() {
        assert_early_join();
        falls_back(Some("early-other-program"), "");
    }

    #[test]
    #[ignore = "GPU tier (opt-in knobs): the early join build panics"]
    fn a_panicking_early_join_build_falls_back_to_a_checked_reference() {
        use crate::block_whir::cluster_tcp::JOIN_BUILD_PANICS;
        assert_early_join();
        let _fault = JoinBuildFault::set(JOIN_BUILD_PANICS);
        falls_back(None, "join build thread panicked");
    }

    #[test]
    #[ignore = "GPU tier (opt-in knobs): the early join build fails"]
    fn a_failing_early_join_build_falls_back_to_a_checked_reference() {
        use crate::block_whir::cluster_tcp::JOIN_BUILD_FAILS;
        assert_early_join();
        let _fault = JoinBuildFault::set(JOIN_BUILD_FAILS);
        falls_back(None, "the early join build fails");
    }

    /// ★ The join build panics while the leaf phase has already failed (the
    /// worker exited after its artifacts). Before the fix the leaf phase's
    /// error returned past the join thread and the scope re-raised its panic:
    /// the process died with no fallback.
    ///
    /// Mutation: drop both the catch in the join thread and the hand join on
    /// the error path, and this test's process panics.
    #[test]
    #[ignore = "GPU tier (opt-in knobs): a join build panic beside a dead worker"]
    fn a_panicking_join_build_beside_a_dead_worker_is_an_error_not_a_crash() {
        use crate::block_whir::cluster_tcp::JOIN_BUILD_PANICS;
        assert_early_join();
        let _fault = JoinBuildFault::set(JOIN_BUILD_PANICS);
        falls_back(Some("early-artifacts-exit"), "worker 0 leaf");
    }
}
