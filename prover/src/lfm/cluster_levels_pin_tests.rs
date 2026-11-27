//! The two-leaf programs and shipped files at the production posture.

use super::*;
use crate::lfm::wire::WireFormat;

/// The two leaves' and the join's program ids for the test fixture's block
/// and map, under the production parameters every level proves at.
#[test]
#[ignore = "GPU tier: proves a small range block at the production posture"]
fn two_leaf_program_ids_are_pinned_at_the_production_posture() {
    assert!(!std::hint::black_box(crate::security_posture::TEST_POSTURE));
    if !pin_process::deterministic() || last_wire() != Ok(WireFormat::Dedup) {
        pin_process::run_alone(
            "lfm::cluster_levels::production_pin::\
             two_leaf_program_ids_are_pinned_at_the_production_posture",
        );
        return;
    }
    let family = WlfmFamily::B3;
    let elf = crate::test_utils::asm_elf_bytes("test_commit_split");
    pin_process::check_pinned_elf(&elf);
    let format = pinned_format(family).unwrap();
    let opts = super::super::proof::block_base_options_under(&format.zf);
    let mut options = super::super::whir_block_tests::small_block_options();
    options.window_log2 = Some(2);
    options.max_rows.cpu = 8;
    options.drop_streamed_ops = true;
    let (base, _, _) = crate::block_whir::prove_range_reference(
        &elf,
        &[0x11],
        &opts,
        &format,
        &options,
        &[2, 4],
        Clock::default(),
    )
    .unwrap();
    let mut map = vec![0; base.groups.len()];
    map[1] = 1;
    let map = LeafMap::decode(&map, map.len()).unwrap();
    let plan = cluster_plan(&elf, &opts, &format, base.statement(), None, Some(&map)).unwrap();
    let children = (0..2)
        .map(|leaf| leaf_child(&plan, &build_leaf(&plan, leaf, true).unwrap().build))
        .collect::<Vec<_>>();
    let join = join_level(
        &children,
        &plan.child_layout(),
        LevelStop::DEFAULT,
        LastLevel::pinned(family),
        level_rounds(&plan),
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
            "687b664d27e38dd80bf0994761f56723d7fcd00be4d46ccadc12fcef94f34b30",
            "5338a3fa2baec57649cf5acd1f39d7531c65c28316b62a1b10f492e326aec3f1",
            "e0ab2c9b1ba811fd69087e6c591cf1e4ac58747e7e94b0ccda37c9db5243880e",
        ]
    );
}

/// The ELF and the runner's published (statement, file) archive.
fn read_shipped_cluster_file() -> (Vec<u8>, OwnedBlockStatement, Vec<u8>) {
    let var = |name| std::env::var_os(name).unwrap_or_else(|| panic!("{name} is unset"));
    let elf = std::fs::read(var("LAMBDA_VM_CLUSTER_CHECK_ELF")).unwrap();
    let output = std::fs::read(var("LAMBDA_VM_CLUSTER_CHECK_OUTPUT")).unwrap();
    let mut aligned = rkyv::util::AlignedVec::<16>::with_capacity(output.len());
    aligned.extend_from_slice(&output);
    let (statement, file) =
        rkyv::from_bytes::<(OwnedBlockStatement, Vec<u8>), rkyv::rancor::Error>(&aligned).unwrap();
    (elf, statement, file)
}

/// The file verifies, and the same proof under a map with one group moved
/// to another leaf is refused. Prints each verification's seconds.
#[test]
#[ignore = "GPU tier: reads a cluster runner's output named in the environment"]
fn a_shipped_cluster_file_verifies_and_refuses_a_moved_map() {
    let (elf, statement, file) = read_shipped_cluster_file();
    let family = WlfmFamily::B3;
    let last = LastLevel::pinned(family);
    let start = std::time::Instant::now();
    verify_cluster_block_levels_file(&elf, statement.view(), family, &file).unwrap();
    eprintln!(
        "CLUSTER SHIPPED FILE VERIFY {:.3}s ({} bytes)",
        start.elapsed().as_secs_f64(),
        file.len()
    );
    let (map, wire, body) = cluster_file_parts(&file, last, statement.groups.len()).unwrap();
    let map = map.expect("a file with several leaves");
    let leaves = map.partition(statement.groups.len()).unwrap().num_leaves();
    let mut moved = map.as_bytes().to_vec();
    let group = moved.len() - 1;
    moved[group] = if moved[group] == 0 { 1 } else { 0 };
    let moved = LeafMap::decode(&moved, moved.len()).unwrap();
    let wire = match wire {
        0 => WireFormat::Rkyv,
        1 => WireFormat::Compact,
        _ => WireFormat::Dedup,
    };
    let altered = encode_file(
        &decode_body(file[6], body).unwrap(),
        last,
        wire,
        Some(&moved),
    )
    .unwrap();
    let start = std::time::Instant::now();
    let error =
        verify_cluster_block_levels_file(&elf, statement.view(), family, &altered).unwrap_err();
    eprintln!(
        "CLUSTER MOVED MAP REFUSED {:.3}s: {leaves} leaves, group {group} moved: {error}",
        start.elapsed().as_secs_f64()
    );
    assert!(error.contains("does not verify"), "{error}");
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "GPU tier: requires a production cluster reference 58 archive and its ELF"]
fn median_reference_card_artifacts_match_host() {
    let (elf, statement, file) = read_shipped_cluster_file();
    let last = LastLevel::pinned(WlfmFamily::B3);
    let (map, _, _) = cluster_file_parts(&file, last, statement.groups.len()).unwrap();
    let map = map.expect("two-leaf range reference");
    assert_eq!(map.partition.num_leaves(), 2);
    assert_card_artifacts_match_host(&elf, statement.view(), &map);
}
