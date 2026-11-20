//! Where a block node's program emission spends its time: the builder's walk,
//! the source's finish, the compile and the admission validator, for a level-1
//! node over three median-shaped leaves and the top over two such nodes. The
//! children are heights only (`registry::census_artifacts` with the leaves'
//! padded heights): a node's instructions depend on its children's shapes, never
//! on a root, so no commit and no proof are needed.
//!
//! Run alone: `cargo test --release -p lambda-vm-prover --lib -- --ignored
//! --exact lfm::node_emit_timing_tests::node_emission_split --nocapture`.
//! Prints one `NODE EMIT …` line per node.

use std::time::Instant;

use crate::hash_pin::{BlockFamily, ProofFamily};

use super::block_node::{BlockLayout, BlockNodeInputs, emit_block_node};
use super::builder::LfmBuilder;
use super::compiler::{LfmProgram, compile};
use super::per_table_aggregator::DerivedChild;
use super::registry::{LfmArtifacts, census_artifacts};

/// A median leaf's padded heights by chip slot (a median block's `W3 LEAF CENSUS`):
/// `LFM_CONST` 2^11, base ALU 2^20, extension ALU 2^21, `SELECT` 2^20,
/// bit-dec 2^15, the socket 2^18, lanes and hints 2^20, publics 2^6,
/// `LFM_RANGE` 2^16. The keccak and BLAKE3 groups keep the trivial program's.
const LEAF_LOG_HEIGHTS: [(usize, u8); 10] = [
    (0, 11),
    (1, 20),
    (2, 21),
    (3, 20),
    (4, 15),
    (5, 18),
    (7, 20),
    (8, 20),
    (9, 6),
    (10, 16),
];

fn leaf_artifacts() -> LfmArtifacts {
    let program = super::programs::trivial_program();
    let mut a = census_artifacts(&program, <BlockFamily as ProofFamily>::SOCKET);
    for (slot, h) in LEAF_LOG_HEIGHTS {
        a.log_heights[slot] = h;
    }
    a.hash_chunk_log_heights = vec![a.log_heights[super::airs::HASH_SLOT]];
    a.hash_chunk_roots = vec![[0u8; 32]];
    a
}

/// Emit one node over `kids` and print where the time went; returns the
/// compiled program.
fn timed_node(label: &str, kids: &[&DerivedChild], layout: BlockLayout, top: bool) -> LfmProgram {
    let shapes: Vec<_> = kids.iter().map(|c| c.shape()).collect();
    let t = Instant::now();
    let mut b = LfmBuilder::new().with_wrap_hash(<BlockFamily as ProofFamily>::WRAP);
    emit_block_node(
        &mut b,
        &BlockNodeInputs {
            children: &shapes,
            layout,
            top,
        },
    );
    let walk = t.elapsed().as_secs_f64();
    let t = Instant::now();
    let source = b.finish();
    let finish = t.elapsed().as_secs_f64();
    let t = Instant::now();
    let program = compile(source);
    let compiled = t.elapsed().as_secs_f64();
    let t = Instant::now();
    super::validator::validate(&program).expect("the node is admissible");
    let validated = t.elapsed().as_secs_f64();
    let split = super::validator::validate_split(&program)
        .iter()
        .map(|(n, s)| format!("{n} {s:.3}"))
        .collect::<Vec<_>>()
        .join(" · ");
    println!("NODE VALIDATE {label}: {split}");
    let rows = super::whir_chain::ChipRows::of(&program);
    println!(
        "NODE EMIT {label}: walk {walk:.3}s · finish {finish:.3}s · compile {compiled:.3}s · \
         validate {validated:.3}s · total {:.3}s · {} instructions · rows {rows:?}",
        walk + finish + compiled + validated,
        program.instrs.len(),
    );
    program
}

#[test]
#[ignore = "instrument: run alone with --exact"]
fn node_emission_split() {
    let wrap = super::proof::aggregation_wrap_options();
    let layout = BlockLayout::child(1, 8);
    let words = layout.total();
    let leaf = DerivedChild::from_artifacts_under::<BlockFamily>(&leaf_artifacts(), &wrap, words)
        .expect("the leaf derives");
    let level1 = timed_node("L1 (3 leaves)", &[&leaf, &leaf, &leaf], layout, false);
    let node = census_artifacts(&level1, <BlockFamily as ProofFamily>::SOCKET);
    drop(level1);
    let node = DerivedChild::from_artifacts_under::<BlockFamily>(&node, &wrap, words)
        .expect("the node derives");
    timed_node("top (2 nodes)", &[&node, &node], layout, true);
}
