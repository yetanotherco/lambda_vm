//! Exercises the shape profile against the real VM AIRs.
//!
//! The unit tests in `shape_profile` cover the schema mapping with plain data;
//! this covers the other half — that `AirShapeInput::from_air` reads sane
//! shapes off all ~29 live tables, and that a captured segment stays
//! self-consistent.

use crate::shape_profile::{AirShapeInput, BusIndexMap, SegmentProfile, build_segment};
use crate::tables::trace_builder::Traces;
use crate::test_utils::run_asm_elf;

/// Builds a real segment profile from a small program's tables.
fn profile_for(program: &str) -> SegmentProfile {
    let (elf, logs, _instructions) = run_asm_elf(program);
    let mut traces =
        Traces::from_elf_and_logs_minimal(&elf, &logs, &Default::default(), &[]).unwrap();

    let proof_options = stark::proof::options::ProofOptions::default_test_options();
    let table_counts = traces.table_counts();
    let airs = crate::VmAirs::new(
        &elf,
        &proof_options,
        true,
        &traces.page_configs,
        &table_counts,
        None,
        true,
        None,
        None,
        None,
    );

    let pairs = airs.air_trace_pairs(&mut traces);
    let mut buses = BusIndexMap::new();
    build_segment(
        0,
        pairs
            .iter()
            .map(|(air, trace, _)| AirShapeInput::from_air(*air, trace.num_rows())),
        &mut buses,
    )
}

#[test]
fn real_airs_produce_a_self_consistent_segment() {
    let profile = profile_for("sub");

    assert!(
        profile.airs.len() >= 10,
        "expected the full table set, got {}",
        profile.airs.len()
    );

    for air in &profile.airs {
        assert_eq!(
            air.buses.len(),
            air.num_interactions,
            "{}: bus count disagrees with num_interactions",
            air.air_name
        );
        assert_eq!(
            air.interaction_message_lens.len(),
            air.num_interactions,
            "{}: message-len count disagrees with num_interactions",
            air.air_name
        );
        assert_eq!(
            air.interaction_count_weights.len(),
            air.num_interactions,
            "{}: count-weight count disagrees with num_interactions",
            air.air_name
        );
        assert_eq!(
            1usize << air.log_height,
            air.height,
            "{}: log_height does not match height",
            air.air_name
        );
        assert!(air.width.common_main > 0, "{}: empty main", air.air_name);
        assert!(
            air.max_constraint_degree >= 1,
            "{}: degree below 1",
            air.air_name
        );
        // Every message carries at least one field beyond the bus id.
        for (i, len) in air.interaction_message_lens.iter().enumerate() {
            assert!(
                *len > 0,
                "{}: interaction {i} has an empty message",
                air.air_name
            );
        }
    }
}

#[test]
fn air_ids_are_dense_and_ordered() {
    let profile = profile_for("sub");
    let ids: Vec<usize> = profile.airs.iter().map(|a| a.air_id).collect();
    assert_eq!(ids, (0..profile.airs.len()).collect::<Vec<_>>());
}

#[test]
fn global_degree_dominates_every_air() {
    let profile = profile_for("sub");
    for air in &profile.airs {
        assert!(
            air.max_constraint_degree <= profile.global_max_constraint_degree,
            "{} exceeds the global degree",
            air.air_name
        );
    }
}

#[test]
fn bus_indices_are_dense_from_zero() {
    let profile = profile_for("sub");
    let distinct: std::collections::HashSet<u16> = profile
        .airs
        .iter()
        .flat_map(|a| a.buses.iter().copied())
        .collect();
    assert!(!distinct.is_empty(), "no buses captured");

    // Indices are handed out 0, 1, 2, ... as bus ids are first seen, so the set
    // must be exactly 0..distinct.len().
    let max = distinct.iter().copied().max().unwrap() as usize;
    assert_eq!(
        max,
        distinct.len() - 1,
        "bus indices are not dense from zero"
    );
}

#[test]
fn interaction_totals_are_nonzero_across_the_segment() {
    let profile = profile_for("sub");
    let total: usize = profile.airs.iter().map(|a| a.num_interactions).sum();
    assert!(total > 0, "captured no bus interactions at all");
}

/// Not an assertion — writes the real profile so it can be eyeballed against
/// `max_rows` and handed to the replay runner. Run with `--ignored`.
#[test]
#[ignore]
fn dump_profile_for_inspection() {
    let profile = profile_for("sub");
    // Never inside the repo: this is throwaway inspection output.
    let path = std::env::temp_dir().join("lambda-vm-shape-profile-sub.jsonl");
    let _ = std::fs::remove_file(&path);
    crate::shape_profile::append_jsonl(&path, &profile).unwrap();
    eprintln!("wrote {} AIRs to {}", profile.airs.len(), path.display());
}
