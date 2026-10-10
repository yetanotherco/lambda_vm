//! The no-epoch block's recursion (D-NOEPOCH §12): the shared front against the
//! host transcript, the partition, the node bindings with a mutation per check,
//! block leaves over a real block proof, and the box-tier trees.
//!
//! Tiers:
//! - laptop: the front (synthetic statement and roots), the partition, the
//!   bindings over hinted words — programs of a few thousand instructions,
//!   executed, never proved;
//! - box, fixture scale: block leaves over a real monolithic proof of a small
//!   guest, proved and verified, the nodes above them, and the negatives that need
//!   real child proofs;
//! - box, production scale: the whole block, base to top node, timed.

use stark::config::Commitment;

use crate::tables::types::{FE, FEE};

use super::LfmArtifacts;
use super::block_leaf::{BlockPartition, emit_block_leaf_over, partition_by_rule};
use super::block_node::{
    BlockBindings, BlockLayout, BlockNodeInputs, bind_and_publish_with, emit_block_node_with,
};
use super::block_plan::{BlockShape, BlockTreePlan, le_halves, top_claims};
use super::block_replay::{BlockStatementShape, replay_block_front};
use super::block_tree::{BlockTreeSink, StdoutSink};
use super::builder::LfmBuilder;
use super::compiler::{LfmProgram, compile};
use super::edsl::WrapHash;
use super::executor::execute;
use super::per_table_aggregator::{DerivedChild, LegCells, hint_public_words, publics_arena};
use super::per_table_aggregator_tests::{RealChild, child_shape};
use super::statement_replay::{NUM_TABLE_COUNTS, PhaseAPreprocessed, PhaseATable};
use super::word::{LfmWord, base_word, ext_word, word_as_ext};

fn production_builder() -> LfmBuilder {
    LfmBuilder::new().with_wrap_hash(WrapHash::legacy())
}

use super::block_tree::block_seed;

// ================================ (a) the front ===========================

/// ★ (a) The machine's block front draws production's `z, α` and reaches the
/// host transcript's state, from the same statement and the same roots.
///
/// Synthetic data on purpose: a public output whose length is not a multiple of
/// four, distinct counts in declaration order, two page ranges, and preprocessed
/// roots on some tables only — every loop and branch of the absorb sequence. A
/// coalesced or miscounted call moves every value below it.
#[test]
fn the_block_front_replays_the_hosts_transcript() {
    use crypto::fiat_shamir::is_transcript::IsTranscript;

    assert_eq!(
        WrapHash::legacy(),
        WrapHash::Algebraic,
        "the block transcript is algebraic, and this compares its state word"
    );
    const OUTPUT_LEN: usize = 37;
    let elf_digest: [u8; 32] = core::array::from_fn(|i| 0x31 ^ (i as u8).wrapping_mul(7));
    let public_output: Vec<u8> = (0..OUTPUT_LEN)
        .map(|i| 0x5a ^ (i as u8).wrapping_mul(13))
        .collect();
    let page_ranges = [
        crate::RuntimePageRange {
            base: 0x2000,
            count: 2,
        },
        crate::RuntimePageRange {
            base: 0xa000,
            count: 1,
        },
    ];
    let counts = crate::TableCounts {
        cpu: 3,
        lt: 4,
        memw: 5,
        memw_aligned: 6,
        load: 7,
        mul: 8,
        dvrm: 9,
        shift: 10,
        branch: 11,
        memw_register: 12,
        eq: 13,
        bytewise: 14,
        store: 15,
        cpu32: 16,
        keccak: 1,
        keccak_rnd: 4,
        ecsm: 1,
        ecdas: 0,
        hint: 1,
        commit: 1,
        blake3: 0,
    };
    // An independent statement of the encoding, not `table_count_values`.
    let count_array: [u64; NUM_TABLE_COUNTS] = [
        3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 1, 4, 1, 0, 1, 1, 0,
    ];
    const PAGES: usize = 3;
    const FPLD: u8 = 5;
    let root = |k: u64| -> Commitment {
        super::algebraic_commit::digest_to_commitment(&[
            FE::from(k),
            FE::from(k.wrapping_mul(0x9e37_79b9)),
            FE::from(k + 17),
            FE::from(k ^ 0x5555),
        ])
    };
    const N: usize = 7;
    let prep: Vec<Option<Commitment>> = (0..N as u64)
        .map(|i| (i % 3 == 0).then(|| root(1000 + i)))
        .collect();
    let main: Vec<Commitment> = (0..N as u64).map(|i| root(2000 + i)).collect();

    // HOST.
    let mut host = block_seed(
        &elf_digest,
        &public_output,
        &counts,
        PAGES,
        &page_ranges,
        FPLD,
    );
    for i in 0..N {
        if let Some(p) = &prep[i] {
            host.append_bytes(p);
        }
        host.append_bytes(&main[i]);
    }
    let z = host.sample_field_element();
    let alpha = host.sample_field_element();
    let state = host.state_word();

    // MACHINE.
    let shape = BlockStatementShape {
        domain_tag: crate::statement::DOMAIN_TAG.to_vec(),
        public_output_len: OUTPUT_LEN,
        table_counts: count_array,
        num_private_input_pages: PAGES as u64,
        fri_final_poly_log_degree: FPLD,
        page_ranges: page_ranges.iter().map(|r| (r.base, r.count)).collect(),
    };
    let mut b = production_builder();
    let a_out = b.declare_arena(shape.out_halves() as u32);
    let per_root = super::epoch::RootCells::words_per_root(&b);
    let a_main = b.declare_arena(per_root * N as u32);
    let out: Vec<_> = (0..shape.out_halves() as u32)
        .map(|i| b.hint_felt(a_out, i))
        .collect();
    let cells: Vec<_> = (0..N)
        .map(|i| super::epoch::RootCells::hint(&mut b, a_main, per_root * i as u32))
        .collect();
    let lanes: Vec<Vec<_>> = cells
        .iter()
        .map(super::epoch::RootCells::lanes_flat)
        .collect();
    let tables: Vec<PhaseATable> = (0..N)
        .map(|i| PhaseATable {
            preprocessed_root: prep[i].as_ref().map(PhaseAPreprocessed::Constant),
            main_root: &lanes[i][..],
        })
        .collect();
    let front = replay_block_front(&mut b, &shape, &elf_digest, &out, &tables);
    b.public(front.z.as_cell());
    b.public(front.alpha.as_cell());
    for c in front.state.cells() {
        b.public(*c);
    }
    let program = compile(b.finish());
    let arenas = vec![
        le_halves(&public_output)
            .into_iter()
            .map(base_word)
            .collect(),
        super::proof_arena::commitments_to_arena(&main),
    ];
    let exec = execute(
        &program,
        &arenas,
        &program.hasher(crate::hash_pin::LEGACY_HASHER),
    )
    .expect("the block front must execute");
    let got = |i: usize| word_as_ext(&exec.public_words[i].1).expect("an ext challenge");
    assert_eq!(got(0), z, "z must be production's");
    assert_eq!(got(1), alpha, "α must be production's");
    assert_eq!(
        exec.public_words[2].1, state,
        "the state digest must be the host transcript's after z, α"
    );
    println!(
        "BLOCK FRONT: {} instructions, z α and state = host over {N} tables ({} preprocessed)",
        program.instrs.len(),
        prep.iter().flatten().count()
    );

    // ---- what the front does NOT check: it absorbs each half's LIVE bytes only
    // (`bit_dec(half, 8·live)` reads the low bits of a row that recomposes all
    // 64), so a nonzero pad byte, or a half at or over 2^32, executes and draws
    // the same z, α and state. The halves' canonicity is pinned elsewhere: the
    // carrier leaf's COMMIT-bus target (`emit_output_bytes`, refused in
    // `the_commit_target_pins_every_output_half_to_its_bytes`), carried to every
    // leaf by the nodes' `out` binding, and the verifier's exact `top_claims`.
    let last = arenas[0].len() - 1;
    let live = OUTPUT_LEN % 4;
    let v: u64 = arenas[0][last][0].canonical();
    let first: u64 = arenas[0][0][0].canonical();
    let mut padded = arenas.clone();
    padded[0][last] = base_word(FE::from(v | (1u64 << (8 * live))));
    let mut over = arenas.clone();
    over[0][0] = base_word(FE::from(first + (1u64 << 32)));
    for (what, tampered) in [("a nonzero pad byte", padded), ("a half over 2^32", over)] {
        let e = execute(
            &program,
            &tampered,
            &program.hasher(crate::hash_pin::LEGACY_HASHER),
        )
        .unwrap_or_else(|e| panic!("the front alone takes {what}: {e:?}"));
        let words: Vec<LfmWord> = e.public_words.iter().map(|(_, w)| *w).collect();
        let honest: Vec<LfmWord> = exec.public_words.iter().map(|(_, w)| *w).collect();
        assert_eq!(
            words, honest,
            "{what} is not absorbed: the front draws the same z, α, state"
        );
    }
}

/// ★ The output halves' canonicity is the COMMIT-bus target's check
/// (`epoch::emit_output_bytes`, run by the carrier leaf): each half must equal
/// the recomposition of its four bytes and every pad byte must be zero, so a
/// nonzero pad byte and a half at or over 2^32 are refused. The block front does
/// not check it (`the_block_front_replays_the_hosts_transcript`); the nodes'
/// `out` binding carries the carrier's halves to every leaf.
#[test]
fn the_commit_target_pins_every_output_half_to_its_bytes() {
    const LEN: usize = 37;
    let output: Vec<u8> = (0..LEN)
        .map(|i| 0x5a ^ (i as u8).wrapping_mul(13))
        .collect();
    let halves = le_halves(&output);
    let mut b = production_builder();
    let a_out = b.declare_arena(halves.len() as u32);
    let hinted: Vec<_> = (0..halves.len() as u32)
        .map(|i| b.hint_felt(a_out, i))
        .collect();
    let bytes = super::epoch::emit_output_bytes(&mut b, &hinted, LEN);
    for byte in bytes {
        b.public(byte.as_cell());
    }
    let program = compile(b.finish());
    let honest: Vec<Vec<LfmWord>> = vec![halves.iter().copied().map(base_word).collect()];
    let exec = execute(
        &program,
        &honest,
        &program.hasher(crate::hash_pin::LEGACY_HASHER),
    )
    .expect("canonical halves execute");
    let got: Vec<u8> = exec
        .public_words
        .iter()
        .map(|(_, w)| w[0].canonical() as u8)
        .collect();
    assert_eq!(got, output, "the target's bytes are the output");
    let last = halves.len() - 1;
    let v: u64 = halves[last].canonical();
    let first: u64 = halves[0].canonical();
    let mut padded = honest.clone();
    padded[0][last] = base_word(FE::from(v | (1u64 << (8 * (LEN % 4)))));
    let mut over = honest.clone();
    over[0][0] = base_word(FE::from(first + (1u64 << 32)));
    for (what, tampered) in [("a nonzero pad byte", padded), ("a half over 2^32", over)] {
        assert!(
            execute(
                &program,
                &tampered,
                &program.hasher(crate::hash_pin::LEGACY_HASHER)
            )
            .is_err(),
            "{what} must not execute"
        );
    }
}

// ============================== (b) the partition =========================

/// ★ The partition refuses every list set that does not cover each instance
/// exactly once — the only place coverage can go wrong, since the lists are
/// program constants after this. Each arm fails if its check is deleted from
/// `BlockPartition::new`.
#[test]
fn the_partition_refuses_a_gap_an_overlap_and_an_empty_leaf() {
    assert!(BlockPartition::new(vec![vec![0, 2], vec![1, 3]], 4).is_ok());
    let missing = BlockPartition::new(vec![vec![0, 2], vec![3]], 4).unwrap_err();
    assert!(missing.contains("instance 1 is in no leaf"), "{missing}");
    let twice = BlockPartition::new(vec![vec![0, 1, 2], vec![2, 3]], 4).unwrap_err();
    assert!(
        twice.contains("instance 2 is in leaf 0 and leaf 1"),
        "{twice}"
    );
    let dup = BlockPartition::new(vec![vec![0, 1, 1], vec![2, 3]], 4).unwrap_err();
    assert!(dup.contains("instance 1 is in leaf 0 and leaf 0"), "{dup}");
    let range = BlockPartition::new(vec![vec![0, 1], vec![2, 4]], 4).unwrap_err();
    assert!(range.contains("out of range"), "{range}");
    let empty = BlockPartition::new(vec![vec![0, 1, 2, 3], vec![]], 4).unwrap_err();
    assert!(empty.contains("verifies no instance"), "{empty}");
}

/// The rule's seeds and its min-load fill, on a toy instance list; KECCAK, ECSM
/// and ECDAS chunks past the first fill by load with the rest.
#[test]
fn the_partition_rule_seeds_the_wide_tables_and_fills_by_load() {
    let names = [
        "BITWISE",
        "DECODE",
        "KECCAK_RC",
        "REGISTER",
        "HALT",
        "COMMIT[0]",
        "KECCAK[0]",
        "KECCAK_RND[0]",
        "KECCAK_RND[1]",
        "ECSM[0]",
        "ECDAS[0]",
        "HINT[0]",
        "CPU[0]",
        "CPU[1]",
        "PAGE:0x0",
        "MEMW_R[0]",
    ];
    let costs = [1, 1, 1, 1, 1, 1, 50, 60, 60, 40, 40, 1, 30, 30, 10, 30];
    let p = partition_by_rule(&names, &costs, 4).expect("four leaves fill");
    // Seeds: KECCAK_RND 0/1 → leaves 0/1, ECDAS → 1, ECSM → 2, KECCAK → 3,
    // the fixed and tiny tables → 0. Loads after seeding: 67, 100, 40, 50.
    // Fill in AIR order, PAGE last: CPU[0] → 2 (40 → 70), CPU[1] → 3 (50 → 80),
    // MEMW_R[0] → 0 (67 → 97), PAGE → 2 (70 → 80).
    assert_eq!(
        p.leaves(),
        &[
            vec![0, 1, 2, 3, 4, 5, 7, 11, 15],
            vec![8, 10],
            vec![9, 12, 14],
            vec![6, 13],
        ]
    );
    // One leaf takes everything; the seeds wrap modulo the leaf count.
    let one = partition_by_rule(&names, &costs, 1).expect("one leaf fills");
    assert_eq!(one.leaves(), &[(0..names.len()).collect::<Vec<_>>()]);
    // More leaves than the rule can fill: refused, never an empty leaf.
    assert!(partition_by_rule(&names, &costs, names.len() + 1).is_err());

    // KECCAK[1], ECSM[1] and ECDAS[1] (instances 7, 11 and 13 now) are not
    // seeded: they fill by load in AIR order, before the CPU chunks. Loads after
    // seeding: 67, 100, 40, 50 as above. KECCAK[1] → 2 (40 → 90), ECSM[1] → 3
    // (50 → 90), ECDAS[1] → 0 (67 → 107), CPU[0] → 2 (90 → 120), CPU[1] → 3
    // (90 → 120), MEMW_R[0] → 1 (100 → 130), PAGE → 0 (107 → 117).
    let chunked = [
        "BITWISE",
        "DECODE",
        "KECCAK_RC",
        "REGISTER",
        "HALT",
        "COMMIT[0]",
        "KECCAK[0]",
        "KECCAK[1]",
        "KECCAK_RND[0]",
        "KECCAK_RND[1]",
        "ECSM[0]",
        "ECSM[1]",
        "ECDAS[0]",
        "ECDAS[1]",
        "HINT[0]",
        "CPU[0]",
        "CPU[1]",
        "PAGE:0x0",
        "MEMW_R[0]",
    ];
    let costs = [
        1, 1, 1, 1, 1, 1, 50, 50, 60, 60, 40, 40, 40, 40, 1, 30, 30, 10, 30,
    ];
    let p = partition_by_rule(&chunked, &costs, 4).expect("four leaves fill");
    assert_eq!(
        p.leaves(),
        &[
            vec![0, 1, 2, 3, 4, 5, 8, 13, 14, 17],
            vec![9, 12, 18],
            vec![7, 10, 15],
            vec![6, 11, 16],
        ]
    );
}

// ============================ (M1) the plan's shape ========================

/// A block shape for `poc_rodata_commit` the host's pre-checks accept: one CPU
/// and one MEMW_R table, the ELF's pages, every trace 2^5 rows.
fn honest_fixture_shape(elf: &executor::elf::Elf) -> BlockShape {
    let counts = crate::TableCounts {
        cpu: 1,
        lt: 0,
        memw: 0,
        memw_aligned: 0,
        load: 0,
        mul: 0,
        dvrm: 0,
        shift: 0,
        branch: 0,
        memw_register: 1,
        eq: 0,
        bytewise: 0,
        store: 0,
        cpu32: 0,
        keccak: 0,
        keccak_rnd: 0,
        ecsm: 0,
        ecdas: 0,
        hint: 0,
        commit: 0,
        blake3: 0,
    };
    let pages = crate::tables::trace_builder::Traces::page_configs_from_elf(elf).len();
    let n = counts.total().expect("fits") + crate::FIXED_TABLE_COUNT + pages;
    BlockShape {
        table_counts: counts,
        runtime_page_ranges: Vec::new(),
        num_private_input_pages: 0,
        public_output_len: 4,
        trace_lengths: vec![32; n],
    }
}

/// ★ M1: the plan refuses every shape the host verifier refuses, before it
/// builds an AIR ([`super::block_plan::check_shape`], `verify_proof_parts`'
/// pre-checks): a non-chunked accelerator counted twice, a KECCAK_RND count the
/// trace lengths do not cover, more private-input pages than the bound, a
/// runtime page range over an ELF page, a trace length that is not a power of
/// two, and a KECCAK, KECCAK_RND, ECSM or ECDAS instance over its cap. Each
/// tamper is the only change to a shape the checks accept.
#[test]
fn the_plan_refuses_a_shape_the_host_refuses() {
    use super::block_plan::check_shape;
    let opts = fixture_block_options();
    let elf_bytes = crate::test_utils::asm_elf_bytes("poc_rodata_commit");
    let elf = executor::elf::Elf::load(&elf_bytes).expect("load the ELF");
    let honest = honest_fixture_shape(&elf);
    assert!(
        check_shape(&elf, &opts, &honest).is_ok(),
        "the honest shape passes, so each refusal below is its tamper's"
    );
    let elf_page = crate::tables::trace_builder::Traces::page_configs_from_elf(&elf)
        .first()
        .expect("the ELF has a page")
        .page_base;
    type Tamper = Box<dyn Fn(&mut BlockShape)>;
    let tampers: Vec<(&str, Tamper)> = vec![
        (
            "a non-chunked accelerator counted twice",
            Box::new(|s| {
                s.table_counts.hint = 2;
                s.trace_lengths.extend([32, 32]);
            }),
        ),
        (
            "a KECCAK_RND count the trace lengths do not cover",
            Box::new(|s| s.table_counts.keccak_rnd = 3),
        ),
        (
            "more private-input pages than the bound",
            Box::new(|s| {
                s.num_private_input_pages = crate::tables::page::max_private_input_pages() + 1;
            }),
        ),
        (
            "a runtime page range over an ELF page",
            Box::new(move |s| {
                s.runtime_page_ranges.push(crate::RuntimePageRange {
                    base: elf_page,
                    count: 1,
                });
                s.trace_lengths.push(32);
            }),
        ),
        (
            "a trace length that is not a power of two",
            Box::new(|s| s.trace_lengths[0] = 48),
        ),
        (
            "an ECDAS instance over its cap",
            Box::new(|s| {
                // ECDAS[1] is instance 6: the five fixed tables, then ECDAS[0].
                s.table_counts.ecdas = 2;
                s.trace_lengths.extend([32, 32]);
                s.trace_lengths[6] = 2 * crate::BLOCK_ECDAS_MAX_ROWS;
            }),
        ),
        (
            "a KECCAK_RND instance over its cap",
            Box::new(|s| {
                s.table_counts.keccak_rnd = 2;
                s.trace_lengths.extend([32, 32]);
                s.trace_lengths[5] = 2 * crate::BLOCK_KECCAK_RND_MAX_ROWS;
            }),
        ),
        (
            "a KECCAK instance over its cap",
            Box::new(|s| {
                // KECCAK[1] is instance 6: the five fixed tables, then KECCAK[0].
                s.table_counts.keccak = 2;
                s.trace_lengths.extend([32, 32]);
                s.trace_lengths[6] = 2 * crate::BLOCK_KECCAK_MAX_ROWS;
            }),
        ),
        (
            "an ECSM instance over its cap",
            Box::new(|s| {
                s.table_counts.ecsm = 2;
                s.trace_lengths.extend([32, 32]);
                s.trace_lengths[5] = 2 * crate::BLOCK_ECSM_MAX_ROWS;
            }),
        ),
    ];
    for (what, tamper) in tampers {
        let mut shape = honest.clone();
        tamper(&mut shape);
        let refused = check_shape(&elf, &opts, &shape);
        assert!(refused.is_err(), "the plan must refuse {what}");
        assert!(
            BlockTreePlan::derive(&elf_bytes, &opts, &shape).is_err(),
            "the derivation refuses {what}"
        );
        println!("PLAN refuses {what}: {}", refused.err().unwrap_or_default());
    }
    // KECCAK_RND is chunked under the block's accelerator shape: a count the
    // trace lengths DO cover passes the pre-checks (the sub-proof count is its
    // bound, as in `verify_proof_parts`).
    let mut chunked = honest.clone();
    chunked.table_counts.keccak_rnd = 3;
    chunked.trace_lengths.extend([32, 32, 32]);
    assert!(check_shape(&elf, &opts, &chunked).is_ok());
    // So are ECDAS, KECCAK and ECSM, each instance at its cap at most.
    chunked.table_counts.ecdas = 2;
    chunked.table_counts.keccak = 2;
    chunked.table_counts.ecsm = 2;
    chunked.trace_lengths.extend([32; 6]);
    assert!(check_shape(&elf, &opts, &chunked).is_ok());
    // KECCAK 5–6, KECCAK_RND 7–9, ECSM 10–11, ECDAS 12–13, each at its cap.
    chunked.trace_lengths[5..7].fill(crate::BLOCK_KECCAK_MAX_ROWS);
    chunked.trace_lengths[7..10].fill(crate::BLOCK_KECCAK_RND_MAX_ROWS);
    chunked.trace_lengths[10..12].fill(crate::BLOCK_ECSM_MAX_ROWS);
    chunked.trace_lengths[12..14].fill(crate::BLOCK_ECDAS_MAX_ROWS);
    assert!(check_shape(&elf, &opts, &chunked).is_ok());
    // The instance after the last ECDAS (CPU[0], index 14) is not capped: an
    // off-by-one in the ranges would refuse it.
    chunked.trace_lengths[14] = 1 << 22;
    assert!(check_shape(&elf, &opts, &chunked).is_ok());
}

/// ★ KECCAK, ECSM and ECDAS chunks past the first are spread by load, so no
/// count of them overfills a leaf: 13 KECCAK instances at their 2^18 cap (≈ 284
/// k permutations of in-guest verification, over
/// [`super::block_plan::LEAF_PERMS_CAP`] on one leaf), 13 ECSM at 2^17 (≈ 423 k)
/// and 13 ECDAS at 2^17 (≈ 346 k) beside 41 CPU chunks derive a plan whose
/// every leaf is under the cap, with each table's chunks on more than one leaf.
/// Pinned to one leaf each, no leaf count can bring that leaf under the cap and
/// the plan is refused.
#[test]
fn the_plan_spreads_chunked_accelerators_by_load() {
    let opts = super::proof::block_base_options();
    let elf_bytes = crate::test_utils::asm_elf_bytes("poc_rodata_commit");
    let elf = executor::elf::Elf::load(&elf_bytes).expect("load the ELF");
    const CHUNKS: usize = 13;
    const CPUS: usize = 40;
    let shape = spread_fixture_shape(&elf, CHUNKS, CPUS);
    // In AIR order after the five fixed tables: KECCAK 5–17, ECSM 18–30, ECDAS
    // 31–43, then CPU[0] (32 rows) and the 40 CPU chunks added at 2^21.
    let first_cpu = crate::FIXED_TABLE_COUNT + 3 * CHUNKS;
    let plan = BlockTreePlan::derive(&elf_bytes, &opts, &shape).expect("the plan derives");
    assert!(plan.instance(first_cpu - 1).name.starts_with("ECDAS["));
    assert!(plan.instance(first_cpu).name.starts_with("CPU["));
    let costs = plan.costs();
    let partition = plan.partition();
    let load = |leaf: &[usize]| leaf.iter().map(|&i| costs[i]).sum::<usize>();
    let heaviest = partition.leaves().iter().map(|l| load(l)).max().unwrap();
    assert!(
        heaviest <= super::block_plan::LEAF_PERMS_CAP,
        "a leaf of {heaviest} permutations"
    );
    let leaves_of = |kind: &str| {
        let prefix = format!("{kind}[");
        partition
            .leaves()
            .iter()
            .filter(|l| {
                l.iter()
                    .any(|&i| plan.instance(i).name.starts_with(&prefix))
            })
            .count()
    };
    let (keccak_leaves, ecsm_leaves) = (leaves_of("KECCAK"), leaves_of("ECSM"));
    let ecdas_leaves = leaves_of("ECDAS");
    assert!(keccak_leaves > 1 && ecsm_leaves > 1 && ecdas_leaves > 1);
    let one_kind = |kind: &str| {
        let prefix = format!("{kind}[");
        (0..shape.trace_lengths.len())
            .filter(|&i| plan.instance(i).name.starts_with(&prefix))
            .map(|i| costs[i])
            .sum::<usize>()
    };
    println!(
        "PLAN SPREAD: {CHUNKS} KECCAK ({} perms) on {keccak_leaves} leaves, {CHUNKS} ECSM ({} perms) \
         on {ecsm_leaves}, {CHUNKS} ECDAS ({} perms) on {ecdas_leaves}; {} leaves, heaviest {heaviest}",
        one_kind("KECCAK"),
        one_kind("ECSM"),
        one_kind("ECDAS"),
        partition.num_leaves()
    );
}

/// [`honest_fixture_shape`] at production heights: `chunks` KECCAK, ECSM and
/// ECDAS instances at their block caps, and `cpus` more CPU chunks at 2^21 rows
/// after `CPU[0]`.
fn spread_fixture_shape(elf: &executor::elf::Elf, chunks: usize, cpus: usize) -> BlockShape {
    let mut shape = honest_fixture_shape(elf);
    shape.table_counts.keccak = chunks;
    shape.table_counts.ecsm = chunks;
    shape.table_counts.ecdas = chunks;
    shape.table_counts.cpu += cpus;
    let fixed = crate::FIXED_TABLE_COUNT;
    let mut lengths: Vec<usize> = shape.trace_lengths[..fixed].to_vec();
    lengths.extend(std::iter::repeat_n(crate::BLOCK_KECCAK_MAX_ROWS, chunks));
    lengths.extend(std::iter::repeat_n(crate::BLOCK_ECSM_MAX_ROWS, chunks));
    lengths.extend(std::iter::repeat_n(crate::BLOCK_ECDAS_MAX_ROWS, chunks));
    lengths.push(shape.trace_lengths[fixed]);
    lengths.extend(std::iter::repeat_n(1 << 21, cpus));
    lengths.extend_from_slice(&shape.trace_lengths[fixed + 1..]);
    shape.trace_lengths = lengths;
    shape
}

// ======================= (S5) the leaf programs' host bytes =================

use super::block_tree::{JEMALLOC_OVERSIZE, ProgramBytes, program_groups};

/// [`program_groups`]' chip names, in its order.
const PROGRAM_GROUP_NAMES: [&str; 11] = [
    "const", "balu", "xalu", "select", "bitdec", "hash", "keccak", "blake3", "lanes", "hint",
    "public",
];

/// The heap's live bytes (jemalloc `stats.allocated`, the epoch turned first).
fn heap_allocated() -> usize {
    use tikv_jemalloc_ctl::{epoch, stats};
    epoch::advance().expect("the jemalloc epoch turns");
    stats::allocated::read().expect("stats.allocated reads")
}

/// The pages jemalloc holds: live, freed-and-dirty, and its metadata
/// (`stats.resident`, an upper bound: a fresh extent counts before it is
/// touched).
fn heap_resident() -> usize {
    use tikv_jemalloc_ctl::{epoch, stats};
    epoch::advance().expect("the jemalloc epoch turns");
    stats::resident::read().expect("stats.resident reads")
}

fn gib(bytes: usize) -> f64 {
    bytes as f64 / (1u64 << 30) as f64
}

/// The production-height plan the S5 instruments emit from:
/// [`spread_fixture_shape`] with 13 chunks of each accelerator and 40 CPU
/// chunks, under the block base's options, every leaf filled toward
/// [`super::block_plan::LEAF_PERMS_CAP`] as the median block's are.
fn production_height_plan() -> BlockTreePlan {
    spread_plan(13, 40)
}

/// [`spread_fixture_shape`]'s plan under the block base's options.
fn spread_plan(chunks: usize, cpus: usize) -> BlockTreePlan {
    spread_plan_under(chunks, cpus, &super::proof::block_base_options())
}

/// [`spread_plan`] at the default format with the LogUp pair layout — the
/// format the pinned tree ids were recorded under, before k4 became the
/// default. The pins guard the program form, which no format choice changes,
/// so they keep holding at the rollback format.
fn pinned_spread_plan(chunks: usize, cpus: usize) -> BlockTreePlan {
    let pair = crate::zf_format::ZfFormat {
        logup: stark::proof::options::LogUpPolicy::Pair,
        ..crate::zf_format::ZfFormat::DEFAULT
    };
    spread_plan_under(
        chunks,
        cpus,
        &pair.base_options(crate::recursion::Preset::Blowup4.options()),
    )
}

fn spread_plan_under(chunks: usize, cpus: usize, opts: &crate::ProofOptions) -> BlockTreePlan {
    let elf_bytes = crate::test_utils::asm_elf_bytes("poc_rodata_commit");
    let elf = executor::elf::Elf::load(&elf_bytes).expect("load the ELF");
    let shape = spread_fixture_shape(&elf, chunks, cpus);
    BlockTreePlan::derive(&elf_bytes, opts, &shape).expect("the plan derives")
}

/// ★ D-ANYBLOCK S5 §4.13 instrument (laptop): a block leaf program's host bytes
/// by part, at production heights ([`production_height_plan`]) — the ≈ 0.34 GiB
/// a leaf (BIG 480) the whole-block harness holds beside the base. Per leaf:
/// the instruction vector (`size_of::<Instr>()` a row), `BitDec`'s bit lists
/// and the boxed operands, the column groups held (padded, at capacity) against
/// their real rows, and the share in allocations at or over jemalloc's
/// oversize threshold. The heap's own count of each retained program (jemalloc
/// `stats.allocated` across its emission, one leaf at a time) checks the sum.
#[test]
#[ignore = "laptop instrument: run alone (--exact), so the heap's delta is the program's"]
fn a_block_leaf_programs_host_bytes_by_part() {
    use super::instr::{Addr, Instr};
    let plan = production_height_plan();
    let costs = plan.costs();
    let partition = plan.partition().clone();
    let threshold: usize = unsafe { tikv_jemalloc_ctl::raw::read(b"opt.oversize_threshold\0") }
        .expect("opt.oversize_threshold reads");
    println!(
        "LEAF BYTES: size_of::<Instr>() {} · Addr {} · (Addr, u64) {} · FE {} · jemalloc \
         oversize threshold {threshold} B (the instruments assume {JEMALLOC_OVERSIZE}) · {} leaves",
        size_of::<Instr>(),
        size_of::<Addr>(),
        size_of::<(Addr, u64)>(),
        size_of::<FE>(),
        partition.num_leaves()
    );
    assert_eq!(
        threshold, JEMALLOC_OVERSIZE,
        "jemalloc's oversize threshold moved"
    );
    // The first emission makes the process-wide caches; drop it.
    drop(plan.leaf_program(0).expect("leaf 0 emits"));

    let mib = |b: usize| b as f64 / (1u64 << 20) as f64;
    let mut sum = ProgramBytes::default();
    let (mut perms_sum, mut heap_sum, mut instrs_sum) = (0usize, 0usize, 0usize);
    let mut variants = std::collections::BTreeMap::<&str, usize>::new();
    let (mut bitdecs, mut bits, mut contiguous, mut one_mult) = (0usize, 0usize, 0usize, 0usize);
    let mut groups_held = [0usize; 11];
    let mut groups_real = [0usize; 11];
    for (k, leaf) in partition.leaves().iter().enumerate() {
        let perms: usize = leaf.iter().map(|&i| costs[i]).sum();
        let before = heap_allocated();
        let program = plan.leaf_program(k).expect("the leaf emits");
        let heap = heap_allocated().saturating_sub(before);
        let b = ProgramBytes::of(&program);
        println!(
            "LEAF BYTES leaf {k}: {} instances · {perms} perms · {} instrs · held {:.1} MiB (instrs \
             {:.1} for {:.1} used · bitdec bits {:.1} · boxed {:.1} · groups {:.1}, padded rows {:.1}, \
             real rows {:.1} · other {:.2}) · ≥ oversize {:.1} % · {} allocations · heap Δ {:.1} MiB \
             ({:+.1} % on the sum)",
            leaf.len(),
            program.instrs.len(),
            mib(b.total()),
            mib(b.instrs),
            mib(program.instrs.len() * size_of::<Instr>()),
            mib(b.bitdec_heap),
            mib(b.boxed),
            mib(b.groups),
            mib(b.groups_padded),
            mib(b.groups_real),
            mib(b.other),
            100.0 * b.large as f64 / b.total() as f64,
            b.allocs,
            mib(heap),
            100.0 * (heap as f64 / b.total() as f64 - 1.0),
        );
        for instr in &program.instrs {
            let name = match instr {
                Instr::Const { .. } => "Const",
                Instr::BaseAlu { .. } => "BaseAlu",
                Instr::ExtAlu { .. } => "ExtAlu",
                Instr::Select { .. } => "Select",
                Instr::BitDec { bits: list, .. } => {
                    bitdecs += 1;
                    bits += list.len();
                    if list.windows(2).all(|w| w[1].0.0 == w[0].0.0 + 1) {
                        contiguous += 1;
                    }
                    if list.windows(2).all(|w| w[1].1 == w[0].1) {
                        one_mult += 1;
                    }
                    "BitDec"
                }
                Instr::Hash { .. } => "Hash",
                Instr::Hash16(_) => "Hash16",
                Instr::Hint { .. } => "Hint",
                Instr::Pack { .. } => "Pack",
                Instr::Unpack { .. } => "Unpack",
                Instr::KeccakF(_) => "KeccakF",
                Instr::Blake3(_) => "Blake3",
                Instr::Public { .. } => "Public",
            };
            *variants.entry(name).or_default() += 1;
        }
        for (j, g) in program_groups(&program).iter().enumerate() {
            groups_held[j] += g.data.capacity() * size_of::<FE>();
            groups_real[j] += g.real_rows * g.width * size_of::<FE>();
        }
        sum.add(&b);
        perms_sum += perms;
        heap_sum += heap;
        instrs_sum += program.instrs.len();
    }
    let t = sum.total() as f64;
    let pct = |b: usize| 100.0 * b as f64 / t;
    println!(
        "LEAF BYTES Σ: {perms_sum} perms · {instrs_sum} instrs · held {:.1} MiB = instrs {:.1} % · \
         bitdec bits {:.1} % · boxed {:.1} % · groups {:.1} % (capacity slack {:.1} % and row \
         padding {:.1} % of the total) · instrs' capacity slack {:.1} % · other {:.2} % · ≥ oversize \
         {:.1} % · heap Δ {:.1} MiB ({:+.1} %) · {:.0} B a perm, {:.1} B an instr",
        mib(sum.total()),
        pct(sum.instrs),
        pct(sum.bitdec_heap),
        pct(sum.boxed),
        pct(sum.groups),
        pct(sum.groups - sum.groups_padded),
        pct(sum.groups_padded - sum.groups_real),
        pct(sum.instrs - instrs_sum * size_of::<Instr>()),
        pct(sum.other),
        pct(sum.large),
        mib(heap_sum),
        100.0 * (heap_sum as f64 / t - 1.0),
        t / perms_sum as f64,
        t / instrs_sum as f64,
    );
    let groups: Vec<String> = PROGRAM_GROUP_NAMES
        .iter()
        .zip(groups_held.iter().zip(&groups_real))
        .filter(|(_, (held, _))| **held > 0)
        .map(|(name, (held, real))| format!("{name} {:.1}/{:.1}", mib(*held), mib(*real)))
        .collect();
    println!(
        "LEAF BYTES groups (MiB held/real rows, Σ leaves): {}",
        groups.join(" · ")
    );
    let mix: Vec<String> = variants
        .iter()
        .map(|(name, n)| format!("{name} {:.1} %", 100.0 * *n as f64 / instrs_sum as f64))
        .collect();
    println!(
        "LEAF BYTES instrs: {} · BitDec {bitdecs}: {:.1} bits each, contiguous addresses {:.1} %, \
         one multiplicity {:.1} %",
        mix.join(" · "),
        bits as f64 / bitdecs.max(1) as f64,
        100.0 * contiguous as f64 / bitdecs.max(1) as f64,
        100.0 * one_mult as f64 / bitdecs.max(1) as f64,
    );
}

/// The S5 late-emission instruments' probe: under the posture's never-purge
/// jemalloc, `gen` threads allocate and touch trace-like buffers of
/// `buffer_sizes` (cycled) to ≈ 2 GiB and free them, as phase B retires the
/// base's traces; then 4 threads of their own emit leaf programs to ≈ 1.75 GiB and
/// hold them, as the harness's ELF-beside pool does. Reports how much of the
/// programs' heap took pages the allocator did not already hold
/// (`stats.resident`'s rise): the bytes a late emission still adds to the
/// base's high-water.
fn late_emission_reuse_probe(arm: &str, buffer_sizes: &[usize]) {
    let dirty: isize = unsafe { tikv_jemalloc_ctl::raw::read(b"opt.dirty_decay_ms\0") }
        .expect("opt.dirty_decay_ms reads");
    assert_eq!(
        dirty, -1,
        "run under the posture's allocator: _RJEM_MALLOC_CONF=dirty_decay_ms:-1,muzzy_decay_ms:-1"
    );
    let plan = production_height_plan();
    let first = plan.leaf_program(0).expect("leaf 0 emits");
    let per_leaf = ProgramBytes::of(&first);
    drop(first);
    let leaves = (((7usize << 28) / per_leaf.total()).max(1)).min(plan.partition().num_leaves());
    let pool = |name: &'static str, threads: usize| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .thread_name(move |i| format!("{name}-{i}"))
            .build()
            .expect("the pool builds")
    };
    let (gen_pool, emit_pool) = (pool("probe-gen", 6), pool("probe-emit", 4));
    let target = 2usize << 30;
    let mut sizes = Vec::new();
    let mut at = 0usize;
    while sizes.iter().sum::<usize>() < target {
        sizes.push(buffer_sizes[at % buffer_sizes.len()]);
        at += 1;
    }
    let buffers: Vec<Vec<u8>> = gen_pool.install(|| {
        use rayon::prelude::*;
        sizes.par_iter().map(|&n| vec![1u8; n]).collect()
    });
    let (resident0, live0) = (heap_resident(), heap_allocated());
    gen_pool.install(|| {
        use rayon::prelude::*;
        buffers.into_par_iter().for_each(drop);
    });
    let (resident1, live1) = (heap_resident(), heap_allocated());
    let programs: Vec<LfmProgram> = emit_pool.install(|| {
        use rayon::prelude::*;
        (0..leaves)
            .into_par_iter()
            .map(|k| plan.leaf_program(k).expect("the leaf emits"))
            .collect()
    });
    let (resident2, live2) = (heap_resident(), heap_allocated());
    let mut held = ProgramBytes::default();
    for p in &programs {
        held.add(&ProgramBytes::of(p));
    }
    let heap = live2.saturating_sub(live1);
    let fresh = resident2.saturating_sub(resident1);
    println!(
        "LATE PROBE {arm}: {} buffers ({:.2} GiB, sizes {:?} MiB) freed: live {:.2} → {:.2} GiB, \
         resident {:.2} → {:.2} · {leaves} leaf programs emitted on 4 threads: heap {:.2} GiB \
         (held by count {:.2}, ≥ oversize {:.1} %) · resident {:.2} → {:.2} (+{:.2} GiB) · \
         fresh pages {:.1} % of the programs' heap, reused {:.1} %",
        sizes.len(),
        gib(sizes.iter().sum()),
        buffer_sizes.iter().map(|b| b >> 20).collect::<Vec<_>>(),
        gib(live0),
        gib(live1),
        gib(resident0),
        gib(resident1),
        gib(heap),
        gib(held.total()),
        100.0 * held.large as f64 / held.total() as f64,
        gib(resident1),
        gib(resident2),
        gib(fresh),
        100.0 * fresh as f64 / heap as f64,
        100.0 * (1.0 - fresh as f64 / heap as f64),
    );
    drop(programs);
}

/// ★ S5 late emission's mechanism, the trace arm: the base's packed traces are
/// tens to hundreds of MiB an instance (median: 56.5 GiB over 941), so phase B
/// frees them into the shared oversize arena. Laptop, the posture's allocator:
/// `_RJEM_MALLOC_CONF=dirty_decay_ms:-1,muzzy_decay_ms:-1`, run alone.
#[test]
#[ignore = "laptop instrument: run alone under _RJEM_MALLOC_CONF=dirty_decay_ms:-1,muzzy_decay_ms:-1"]
fn late_emission_probe_over_freed_trace_sized_buffers() {
    late_emission_reuse_probe("trace-sized", &[16 << 20, 48 << 20, 128 << 20, 24 << 20]);
}

/// ★ The contrast arm: 1 MiB buffers, under the oversize threshold, so their
/// pages stay in the freeing threads' own arenas.
#[test]
#[ignore = "laptop instrument: run alone under _RJEM_MALLOC_CONF=dirty_decay_ms:-1,muzzy_decay_ms:-1"]
fn late_emission_probe_over_freed_small_buffers() {
    late_emission_reuse_probe("small", &[1 << 20]);
}

/// Every program id of `plan`'s tree under the tree's wrap options, as hex:
/// each leaf's, then each node level's in node order, the top's last. Derived
/// one program at a time (emit, build, keep the child's derived shape, drop the
/// program), so one program is held at once. Prints the ELF digest the plan
/// absorbs and each program's preprocessed roots by slot (`TREE ROOTS`), so two
/// runs that disagree can be compared group by group.
fn tree_ids(plan: &BlockTreePlan) -> Vec<String> {
    let wrap = super::proof::aggregation_wrap_options();
    let words = plan.child_layout().total();
    let hex = |id: &Commitment| id.iter().map(|b| format!("{b:02x}")).collect::<String>();
    let roots = |j: usize, a: &LfmArtifacts| {
        let slots: Vec<String> = a
            .roots
            .iter()
            .enumerate()
            .map(|(s, r)| format!("{s}:{}", hex(r)))
            .collect();
        let chunks = |c: &[Commitment]| c.iter().map(hex).collect::<Vec<_>>().join(" ");
        println!(
            "TREE ROOTS {j}: {} · hash chunks {} · blake3 chunks {}",
            slots.join(" "),
            chunks(&a.hash_chunk_roots),
            chunks(&a.blake3_chunk_roots)
        );
    };
    println!("TREE ELF digest {}", hex(plan.elf_digest()));
    let mut ids = Vec::new();
    let mut level: Vec<DerivedChild> = Vec::new();
    for k in 0..plan.partition().num_leaves() {
        let program = plan.leaf_program(k).expect("the leaf emits");
        let artifacts = super::block_plan::artifacts_of(&program, &wrap);
        roots(ids.len(), &artifacts);
        ids.push(hex(&artifacts.program_id));
        level.push(
            DerivedChild::from_artifacts(&artifacts, &wrap, words).expect("the leaf derives"),
        );
    }
    let levels = plan.levels();
    for (lv, arities) in levels.iter().enumerate() {
        let top = lv + 1 == levels.len();
        let mut kids = level.into_iter();
        let mut next = Vec::new();
        for &a in &arities.arities {
            let group: Vec<DerivedChild> = kids.by_ref().take(a).collect();
            let program = plan.node_program(&group, top).expect("the node emits");
            let artifacts = super::block_plan::artifacts_of(&program, &wrap);
            roots(ids.len(), &artifacts);
            ids.push(hex(&artifacts.program_id));
            next.push(
                DerivedChild::from_artifacts(&artifacts, &wrap, words).expect("the node derives"),
            );
        }
        level = next;
    }
    ids
}

/// The pinned ids of `plan`'s tree for the ELF the plan absorbs, checked against
/// [`tree_ids`]: one pin set per fixture-ELF digest, since the fixture ELF's
/// bytes depend on the clang that assembled it (laptop `af87f637…`, 1264 B;
/// FAST `bfb782e1…`, 1272 B) and every leaf absorbs the digest (FAST 670). An
/// ELF with no pins refuses, naming its digest.
fn check_pinned_tree_ids(label: &str, plan: &BlockTreePlan, pins: &[(&str, &[&str])]) {
    let digest = plan
        .elf_digest()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect::<String>();
    let Some((_, want)) = pins.iter().find(|(d, _)| *d == digest) else {
        panic!(
            "no pinned ids for the fixture ELF's digest {digest}: its bytes depend on the clang that \
             assembled it; record this machine's ids at a base sha and add them"
        );
    };
    let ids = tree_ids(plan);
    for (j, id) in ids.iter().enumerate() {
        println!("{label} {j}: {id}");
    }
    assert_eq!(
        ids, *want,
        "a program of the tree changed (ELF digest {digest})"
    );
}

/// An ELF the pins do not know is refused by name, before any program is
/// derived, rather than read as a changed program.
#[test]
#[should_panic(expected = "no pinned ids for the fixture ELF's digest")]
fn pinned_tree_ids_refuse_an_elf_they_do_not_know() {
    check_pinned_tree_ids("NONE", &pinned_spread_plan(2, 4), &[("00", &[])]);
}

/// The fixture ELF's digest on the laptop (macOS clang) and on FAST.
const LAPTOP_ELF_DIGEST: &str = "3b39e219ec8e9d539067daf43e89cf56c58677134c0fdbc7373ece7d042eadde";
const FAST_ELF_DIGEST: &str = "f5e120d0eaa369d7ebdec2767b8d0c0559ceeac138edf0970d47753105df8069";

/// ★ The compact program form changes no program: every id of a small spread
/// plan's tree equals the one recorded before the form changed — on the laptop
/// at 541f4bdc2 (#1013's d740eb5d5 + the instruments), on FAST at fe1fb1691
/// (FAST 672, where the new sha's ids matched it, device and host).
#[test]
#[ignore = "laptop or box: run with --exact"]
fn the_compact_program_form_keeps_a_small_trees_ids() {
    check_pinned_tree_ids("SMALL TREE IDS", &pinned_spread_plan(2, 4), &SMALL_TREE_IDS);
}

/// The same at production heights ([`production_height_plan`]: 7 leaves, 2
/// nodes and the top), on a box: ≈ 4.7 min and 4.6 GiB on the laptop's host
/// commit. The box gates add the 1× and median trees' top ids.
#[test]
#[ignore = "box tier: production-height artifacts, ≈ 4.7 min and 4.6 GiB on the laptop's host commit"]
fn the_compact_program_form_keeps_every_production_height_tree_id() {
    check_pinned_tree_ids(
        "TREE IDS",
        &pinned_spread_plan(13, 40),
        &PRODUCTION_HEIGHT_TREE_IDS,
    );
}

/// [`pinned_spread_plan`]'s format with a Poseidon1 base at 4-ary cap
/// `cap` (P3a).
fn p1_spread_options(cap: stark::proof::options::CapPolicy) -> crate::ProofOptions {
    let pair = crate::zf_format::ZfFormat {
        logup: stark::proof::options::LogUpPolicy::Pair,
        base: stark::proof::options::BaseFormat {
            arity4_cap: cap,
            ..stark::proof::options::BaseFormat::P1
        },
        ..crate::zf_format::ZfFormat::DEFAULT
    };
    pair.base_options(crate::recursion::Preset::Blowup4.options())
}

/// ★ P3a on the laptop, no proof: a small spread plan under a Poseidon1 base
/// (cap 1, and uncapped, whose odd-depth trees walk a padded top) derives its
/// whole tree. Every leaf's hash rows are width-16 socket rows (so its
/// artifacts take the socket), every node and the top emit over those
/// children on the pin, the plan names the P1 cost model, and no program id
/// of the tree is one of the RPX tree's (the base is bound into every id).
#[test]
#[ignore = "laptop or box: run with --exact (≈ 1 min)"]
fn a_p1_base_small_tree_derives() {
    use super::hash::HasherKind;
    use stark::proof::options::CapPolicy;
    let rpx = tree_ids(&pinned_spread_plan(2, 4));
    for cap in [CapPolicy::Fixed(1), CapPolicy::Off] {
        let plan = spread_plan_under(2, 4, &p1_spread_options(cap));
        assert_eq!(
            plan.cost_model(),
            super::block_plan::P1_PARTITION_COST_MODEL
        );
        for k in 0..plan.partition().num_leaves() {
            let program = plan.leaf_program(k).expect("the P1 leaf emits");
            assert!(program.hash16(), "leaf {k} hashes on the width-16 socket");
            assert_eq!(
                program.hasher(crate::hash_pin::LEGACY_HASHER),
                HasherKind::Poseidon1W16
            );
            println!(
                "P1 LEAF {k} (cap {cap}): {} socket rows over {} instances",
                program.groups.hash.real_rows,
                plan.partition().leaf(k).len()
            );
        }
        // The socket review's A1: a parent takes a P1W16 child's hasher from the
        // child's artifacts — the hasher the pinned program id names — and
        // builds the child's `LFM_HASH` as the socket from them.
        let wrap = super::proof::aggregation_wrap_options();
        let leaf = super::block_plan::artifacts_of(&plan.leaf_program(0).expect("emits"), &wrap);
        assert_eq!(
            leaf.hasher,
            HasherKind::Poseidon1W16,
            "the leaf's id names the socket"
        );
        let airs = super::airs::LfmAirs::for_artifacts(&leaf, &wrap);
        let hash = airs
            .air_refs()
            .into_iter()
            .find(|a| a.name() == "LFM_HASH")
            .expect("LFM_HASH in the child's set");
        assert_eq!(
            hash.constraints_meta().len(),
            super::p1w16_socket::SOCKET_FORM.num_constraints() + 4,
            "the parent derives the child's LFM_HASH as the socket (its constraints and LogUp)"
        );
        let ids = tree_ids(&plan);
        println!("P1 TREE IDS (cap {cap}): {ids:?}");
        assert!(
            ids.iter().all(|id| !rpx.contains(id)),
            "a P1 tree shares no program with the RPX tree"
        );
    }
}

/// The leaf's shared front alone — public output and main-root hints, the
/// statement, Phase A, `z, α` and the state — in `leaf`'s builder: its hash
/// rows are a leaf's overhead beside its instances.
fn front_hash_rows(plan: &BlockTreePlan, wrap: WrapHash) -> usize {
    let mut b = LfmBuilder::new().with_wrap_hash(wrap);
    let statement = plan.statement();
    let n = plan.num_instances();
    let per_root = super::epoch::RootCells::words_per_root(&b);
    let a_out = b.declare_arena(statement.out_halves() as u32);
    let a_main = b.declare_arena(per_root * n as u32);
    let out: Vec<_> = (0..statement.out_halves() as u32)
        .map(|i| b.hint_felt(a_out, i))
        .collect();
    let main: Vec<super::epoch::RootCells> = (0..n)
        .map(|i| super::epoch::RootCells::hint(&mut b, a_main, per_root * i as u32))
        .collect();
    let lanes: Vec<Vec<_>> = main
        .iter()
        .map(super::epoch::RootCells::lanes_flat)
        .collect();
    let roots: Vec<Option<Commitment>> = plan
        .instances()
        .iter()
        .map(|i| i.precomputed_root)
        .collect();
    let phase_a: Vec<PhaseATable> = (0..n)
        .map(|i| PhaseATable {
            preprocessed_root: roots[i].as_ref().map(PhaseAPreprocessed::Constant),
            main_root: &lanes[i][..],
        })
        .collect();
    let front = replay_block_front(&mut b, statement, plan.elf_digest(), &out, &phase_a);
    for c in front.state.cells() {
        b.public(*c);
    }
    compile(b.finish()).groups.hash.real_rows
}

/// ★ P3a's census (laptop instrument, no proof): at production heights
/// ([`production_height_plan`]'s shape) under RPX (the control) and a
/// Poseidon1 base at caps 1 and 4, every leaf's hash rows split into the shared
/// front, its instances' legs (the closed form, `table_permutations_for`) and
/// the forks (what is left): the mean fork per instance is the cost model's
/// fork constant (RPX's `FORK_PERMS`, 694, is the control). Also each leaf's
/// other rows, where the emulated width-8 grind lands.
#[test]
#[ignore = "laptop instrument: production-height leaves, emitted one at a time (≈ 0.4 GiB each)"]
fn p1_leaf_census_at_production_heights() {
    use stark::proof::options::{BaseFormat, CapPolicy};
    let arms = [
        ("rpx", BaseFormat::RPX),
        (
            "p1-c1",
            BaseFormat {
                arity4_cap: CapPolicy::Fixed(1),
                ..BaseFormat::P1
            },
        ),
        ("p1-c4", BaseFormat::P1),
    ];
    for (name, base) in arms {
        let opts = super::proof::block_base_options_for(base);
        let plan = spread_plan_under(13, 40, &opts);
        let wrap = WrapHash::for_base(&base);
        let front = front_hash_rows(&plan, wrap);
        let (mut rows, mut legs, mut instances, mut other) = (0usize, 0usize, 0usize, 0usize);
        let mut law = 0.0f64;
        for k in 0..plan.partition().num_leaves() {
            let program = plan.leaf_program(k).expect("the leaf emits");
            let leaf: &[usize] = plan.partition().leaf(k);
            let leg: usize = leaf
                .iter()
                .map(|&i| {
                    super::epoch_verify::table_permutations_for(&plan.instance(i).verify, wrap)
                })
                .sum();
            let hash = program.groups.hash.real_rows;
            let rest: usize = program_groups(&program)
                .iter()
                .map(|g| g.real_rows)
                .sum::<usize>()
                - hash;
            println!(
                "CENSUS {name} leaf {k}: {} instances · hash rows {hash} = front {front} + legs \
                 {leg} + forks {} · other rows {rest}",
                leaf.len(),
                hash as i64 - front as i64 - leg as i64
            );
            let groups: Vec<String> = PROGRAM_GROUP_NAMES
                .iter()
                .zip(program_groups(&program))
                .map(|(g, c)| format!("{g} {}×{}", c.real_rows, c.width))
                .collect();
            println!("CENSUS {name} leaf {k} groups: {}", groups.join(" · "));
            let (main, aux) = super::airs::lfm_cell_counts_with_hasher(
                &program,
                program.hasher(crate::hash_pin::LEGACY_HASHER),
            );
            let instrs = program.instrs.len();
            println!(
                "CENSUS {name} leaf {k} cost: {instrs} instructions · cells main {main} + aux {aux} · \
                 law {:.2} s (0.059 + 421 ns/instr + 5.63 ns/cell, FAST)",
                0.059 + 421e-9 * instrs as f64 + 5.63e-9 * (main + aux) as f64
            );
            law += 0.059 + 421e-9 * instrs as f64 + 5.63e-9 * (main + aux) as f64;
            rows += hash;
            legs += leg;
            instances += leaf.len();
            other += rest;
        }
        let leaves = plan.partition().num_leaves();
        let forks = rows as i64 - (leaves * front) as i64 - legs as i64;
        println!(
            "CENSUS {name}: {leaves} leaves · {instances} instances · hash rows {rows} · front \
             {front}/leaf · legs {legs} · fork mean {:.1}/instance · other rows {other} · cost \
             model {:#x} (Σ costs {}) · law Σ {law:.2} s",
            forks as f64 / instances as f64,
            plan.cost_model(),
            plan.costs().iter().sum::<usize>()
        );
    }
}

/// P3a's interior attribution (laptop instrument, no proof): what a node pays
/// for each table of a leaf child, for leaf 0 of the production-height plan
/// under RPX and under a P1 base at cap 1. Per child table: its padded height
/// (the chip census), its width, its composition parts and committed FRI
/// layers at the wrap options, and its legs' closed form
/// (`table_permutations_for`) with the opening words a query reads. RYZEN 010's
/// G4 found the P1 tree's nodes +15 % instructions and +19 % cells.
#[test]
#[ignore = "laptop instrument: production-height leaves, emission only (≈ 0.4 GiB a leaf)"]
fn p1_node_leg_census() {
    use super::airs::{ChipSet, LfmAirs, NUM_LFM_CHIPS, lfm_chip_census_with_hasher};
    use stark::proof::options::{BaseFormat, CapPolicy};
    let wrap = super::proof::aggregation_wrap_options();
    let arms = [
        ("rpx", BaseFormat::RPX),
        (
            "p1-c1",
            BaseFormat {
                arity4_cap: CapPolicy::Fixed(1),
                ..BaseFormat::P1
            },
        ),
    ];
    for (name, base) in arms {
        let plan = spread_plan_under(13, 40, &super::proof::block_base_options_for(base));
        let program = plan.leaf_program(0).expect("the leaf emits");
        let hasher = program.hasher(crate::hash_pin::LEGACY_HASHER);
        let census = lfm_chip_census_with_hasher(&program, hasher);
        let airs = LfmAirs::new_with_hasher(
            &[[0u8; 32]; NUM_LFM_CHIPS],
            &wrap,
            1,
            hasher,
            ChipSet::for_program_with_hasher(&program, hasher),
        );
        let refs = airs.air_refs();
        let dw = super::edsl::digest_words(&LfmBuilder::new().with_wrap_hash(WrapHash::legacy()))
            as usize;
        let (mut perms, mut words) = (0usize, 0usize);
        for c in &census {
            let Some(air) = refs.iter().find(|a| a.name() == c.name) else {
                println!("NODE LEGS {name} {}: no AIR in the set", c.name);
                continue;
            };
            let rows = (c.rows as usize).max(1);
            let (v, _) = super::epoch_verify::TableVerifyShape::derive(*air, rows)
                .expect("the child table's shape derives");
            let p = super::epoch_verify::table_permutations_for(&v, WrapHash::legacy());
            let w = v.opening_words(dw) + v.fri_words(dw);
            perms += p;
            words += w;
            println!(
                "NODE LEGS {name} {:<12} rows 2^{:<2} · cols {} + {} aux · parts {} · FRI layers {} · legs {p} perms \
                 · query words {w}",
                c.name,
                rows.trailing_zeros(),
                c.main_cols,
                c.aux_cols,
                v.quotient.num_composition_parts,
                v.fri.num_committed()
            );
        }
        println!("NODE LEGS {name}: one leaf child's legs {perms} perms · {words} query words");
    }
}

/// P3a's interior, measured on the emitted programs (laptop instrument): the
/// small spread plan's top node over its one leaf, under RPX and under a P1 base
/// at cap 1 — the node's instructions and rows by group, and each child table's
/// height, parts and closed-form legs.
#[test]
#[ignore = "laptop instrument: run with --exact --nocapture (≈ 2 min)"]
fn p1_node_program_census() {
    use stark::proof::options::{BaseFormat, CapPolicy};
    let wrap = super::proof::aggregation_wrap_options();
    for (name, opts) in [
        ("rpx", pinned_spread_plan_options()),
        ("p1-c1", p1_spread_options(CapPolicy::Fixed(1))),
    ] {
        let _ = BaseFormat::RPX;
        let plan = spread_plan_under(2, 4, &opts);
        let leaf = plan.leaf_program(0).expect("the leaf emits");
        let artifacts = super::block_plan::artifacts_of(&leaf, &wrap);
        let words = plan.child_layout().total();
        let airs = super::airs::LfmAirs::for_artifacts(&artifacts, &wrap);
        let heights = super::airs::LfmAirs::log_heights_in_air_order(&artifacts).expect("heights");
        let dw = 1usize;
        for (air, h) in airs.air_refs().into_iter().zip(heights) {
            let (v, an) =
                super::epoch_verify::TableVerifyShape::derive(air, 1usize << h).expect("derives");
            let r = an.report();
            println!(
                "NODE CHILD {name} {:<12} 2^{h:<2} · cols {} · eval points {} · parts {} · alpha {} · \
                 constraint IR nodes {} ext-alu {} mul-base {} consts {} · legs {} perms · words {}",
                air.name(),
                v.sub.deep.num_total_cols,
                v.sub.deep.num_eval_points,
                v.quotient.num_composition_parts,
                v.num_alpha_powers,
                r.nodes,
                r.ext_alu,
                r.mul_base,
                r.constants,
                super::epoch_verify::table_permutations_for(&v, WrapHash::legacy()),
                v.opening_words(dw) + v.fri_words(dw)
            );
        }
        let child = DerivedChild::from_artifacts(&artifacts, &wrap, words).expect("derives");
        let top = plan.node_program(&[child], true).expect("the top emits");
        let groups: Vec<String> = PROGRAM_GROUP_NAMES
            .iter()
            .zip(program_groups(&top))
            .map(|(g, c)| format!("{g} {}", c.real_rows))
            .collect();
        println!(
            "NODE PROGRAM {name}: leaf {} instructions · top {} instructions · {}",
            leaf.instrs.len(),
            top.instrs.len(),
            groups.join(" · ")
        );
    }
}

/// A block's shape from a list of `AIR-name rows` lines (one per instance, any
/// order; reconstructed from a box run's census and table walk), over `elf`: the
/// table counts by kind, the ELF's pages, the private-input pages and the
/// runtime ranges from the PAGE names, the lengths in the AIR order.
fn shape_from_instances(
    elf: &executor::elf::Elf,
    lines: &str,
    public_output_len: usize,
) -> BlockShape {
    use std::collections::BTreeMap;
    let rows: BTreeMap<String, usize> = lines
        .lines()
        .filter(|l| !l.trim().is_empty() && !l.starts_with('#'))
        .map(|l| {
            let (name, r) = l.trim().rsplit_once(' ').expect("`name rows`");
            (name.to_string(), r.parse().expect("rows"))
        })
        .collect();
    let count = |kind: &str| {
        rows.keys()
            .filter(|n| n.split('[').next() == Some(kind) && n.contains('['))
            .count()
    };
    let counts = crate::TableCounts {
        cpu: count("CPU"),
        lt: count("LT"),
        memw: count("MEMW"),
        memw_aligned: count("MEMW_A"),
        load: count("LOAD"),
        mul: count("MUL"),
        dvrm: count("DVRM"),
        shift: count("SHIFT"),
        branch: count("BRANCH"),
        memw_register: count("MEMW_R"),
        eq: count("EQ"),
        bytewise: count("BYTEWISE"),
        store: count("STORE"),
        cpu32: count("CPU32"),
        keccak: count("KECCAK"),
        keccak_rnd: count("KECCAK_RND"),
        ecsm: count("ECSM"),
        ecdas: count("ECDAS"),
        hint: count("HINT"),
        commit: count("COMMIT"),
        blake3: count("BLAKE3"),
    };
    let elf_pages: std::collections::BTreeSet<u64> =
        crate::tables::trace_builder::Traces::page_configs_from_elf(elf)
            .iter()
            .map(|c| c.page_base)
            .collect();
    let private: std::collections::BTreeSet<u64> = crate::tables::page::private_input_page_bases(
        crate::tables::page::max_private_input_pages(),
    )
    .collect();
    let page = crate::tables::page::DEFAULT_PAGE_SIZE as u64;
    let mut runtime: Vec<u64> = Vec::new();
    let mut num_private = 0;
    for name in rows.keys().filter(|n| n.starts_with("PAGE:")) {
        let base =
            u64::from_str_radix(name.trim_start_matches("PAGE:0x"), 16).expect("a page base");
        if private.contains(&base) {
            num_private += 1;
        } else if !elf_pages.contains(&base) {
            runtime.push(base);
        }
    }
    runtime.sort_unstable();
    let mut ranges: Vec<crate::RuntimePageRange> = Vec::new();
    for base in runtime {
        match ranges.last_mut() {
            Some(r) if r.base.checked_add(r.count * page) == Some(base) => r.count += 1,
            _ => ranges.push(crate::RuntimePageRange { base, count: 1 }),
        }
    }
    let n = counts.total().expect("fits")
        + crate::FIXED_TABLE_COUNT
        + rows.keys().filter(|n| n.starts_with("PAGE:")).count();
    let mut shape = BlockShape {
        table_counts: counts,
        runtime_page_ranges: ranges,
        num_private_input_pages: num_private,
        public_output_len,
        trace_lengths: vec![32; n],
    };
    let opts = super::proof::block_base_options();
    let configs =
        super::block_plan::check_shape(elf, &opts, &shape).expect("the reconstructed shape checks");
    let airs = crate::VmAirs::new(
        elf,
        &opts,
        false,
        &configs,
        &shape.table_counts,
        None,
        true,
        None,
        None,
        None,
    );
    let names: Vec<String> = airs
        .air_refs()
        .iter()
        .map(|a| a.name().to_string())
        .collect();
    assert_eq!(names.len(), n, "one AIR per instance");
    shape.trace_lengths = names
        .iter()
        .map(|a| *rows.get(a).unwrap_or_else(|| panic!("no rows for {a}")))
        .collect();
    shape
}

/// ★ One table per socket leaf, sized (laptop instrument, emission only, no
/// proof): a real block's shape (`P3_SHAPE`, `AIR-name rows` lines from a box
/// run; `P3_ELF`, its ELF) under a P1 base at cap 1, partitioned at each leaf
/// cap in `P3_CAPS` (comma-separated; 162000 is today's). Per cap: the leaves,
/// their socket tables (one, or split by `HashChunking`), Σ instructions, Σ
/// cells (main + aux), Σ the FAST cost law, and Σ the node legs a parent pays
/// for its children's tables; and the no-split rule at that cap (each split
/// socket table one padded table: its cells and legs).
#[test]
#[ignore = "laptop instrument: P3_SHAPE, P3_ELF, P3_CAPS; emission only, one leaf at a time"]
fn p1_one_table_per_leaf_sizing() {
    use super::airs::{ChipSet, LfmAirs, NUM_LFM_CHIPS, lfm_chip_census_with_hasher};
    use stark::proof::options::{BaseFormat, CapPolicy};
    let read = |var: &str| std::env::var(var).unwrap_or_else(|_| panic!("{var} must be set"));
    let elf_bytes = std::fs::read(read("P3_ELF")).expect("P3_ELF reads");
    let elf = executor::elf::Elf::load(&elf_bytes).expect("load the ELF");
    let lines = std::fs::read_to_string(read("P3_SHAPE")).expect("P3_SHAPE reads");
    let caps: Vec<usize> = read("P3_CAPS")
        .split(',')
        .map(|c| c.trim().parse().expect("a cap"))
        .collect();
    let shape = shape_from_instances(&elf, &lines, 4);
    let base = BaseFormat {
        arity4_cap: CapPolicy::Fixed(1),
        ..BaseFormat::P1
    };
    let opts = super::proof::block_base_options_for(base);
    let wrap = super::proof::aggregation_wrap_options();
    let mut plan = BlockTreePlan::derive(&elf_bytes, &opts, &shape).expect("the plan derives");
    let law = |instrs: usize, cells: u64| 0.059 + 421e-9 * instrs as f64 + 5.63e-9 * cells as f64;
    let hash = super::airs::LFM_CHIP_NAMES[super::airs::HASH_SLOT];
    let total: usize = plan.costs().iter().sum();
    println!(
        "ONE TABLE: {} instances · Σ {total} perms (closed form) · caps {caps:?}",
        shape.trace_lengths.len()
    );
    for cap in caps {
        let partition = plan.partition_at_cap(cap).expect("the rule partitions");
        plan = plan.with_partition(partition);
        let leaves = plan.partition().num_leaves();
        let (mut instrs, mut cells, mut cost, mut legs) = (0usize, 0u64, 0.0f64, 0usize);
        let (mut split, mut cells_ns, mut cost_ns, mut legs_ns) = (0usize, 0u64, 0.0f64, 0usize);
        let mut heights = std::collections::BTreeMap::<String, usize>::new();
        let (mut lightest, mut heaviest) = (u64::MAX, 0u64);
        let costs = plan.costs();
        let mut over = i64::MIN;
        for k in 0..leaves {
            let program = plan.leaf_program(k).expect("the leaf emits");
            let hasher = program.hasher(crate::hash_pin::LEGACY_HASHER);
            let (main, aux) = super::airs::lfm_cell_counts_with_hasher(&program, hasher);
            let census = lfm_chip_census_with_hasher(&program, hasher);
            let airs = LfmAirs::new_with_hasher(
                &[[0u8; 32]; NUM_LFM_CHIPS],
                &wrap,
                1,
                hasher,
                ChipSet::for_program_with_hasher(&program, hasher),
            );
            let refs = airs.air_refs();
            let leg = |name: &str, rows: u64| -> usize {
                let air = refs
                    .iter()
                    .find(|a| a.name() == name)
                    .expect("the chip's AIR");
                let (v, _) =
                    super::epoch_verify::TableVerifyShape::derive(*air, rows.max(1) as usize)
                        .expect("the child table's shape derives");
                super::epoch_verify::table_permutations_for(&v, WrapHash::legacy())
            };
            let chunks: Vec<_> = census.iter().filter(|c| c.name == hash).collect();
            let leaf_legs: usize = census.iter().map(|c| leg(c.name, c.rows)).sum();
            let n = program.instrs.len();
            instrs += n;
            cells += main + aux;
            cost += law(n, main + aux);
            legs += leaf_legs;
            let label = chunks
                .iter()
                .map(|c| c.rows.to_string())
                .collect::<Vec<_>>()
                .join("+");
            *heights.entry(label).or_default() += 1;
            // The no-split rule: the socket table one padded table.
            let real: u64 = chunks.iter().map(|c| c.real_rows).sum();
            lightest = lightest.min(real);
            heaviest = heaviest.max(real);
            let closed: usize = plan.partition().leaf(k).iter().map(|&i| costs[i]).sum();
            over = over.max(real as i64 - closed as i64);
            let padded: u64 = chunks.iter().map(|c| c.rows).sum();
            let one = real.next_power_of_two();
            let width = (chunks[0].main_cols + chunks[0].aux_cols) as u64;
            let ns_cells = main + aux + (one - padded) * width;
            if chunks.len() > 1 {
                split += 1;
            }
            cells_ns += ns_cells;
            cost_ns += law(n, ns_cells);
            legs_ns += leaf_legs - chunks.iter().map(|c| leg(c.name, c.rows)).sum::<usize>()
                + leg(hash, one);
        }
        let l1 = leaves.div_ceil(super::block_plan::BLOCK_FAN_IN);
        println!(
            "ONE TABLE cap {cap}: {leaves} leaves ({l1} level-1 nodes) · socket tables {heights:?} ({split} split) · socket \
             rows {lightest}..{heaviest} (real − closed form ≤ {over}) · Σ {instrs} instructions · Σ {cells} cells · law Σ {cost:.2} s · node legs Σ \
             {legs} perms"
        );
        println!(
            "ONE TABLE cap {cap} no-split: Σ {cells_ns} cells (+{}) · law Σ {cost_ns:.2} s (+{:.2}) · node legs Σ {legs_ns} perms ({:+})",
            cells_ns - cells,
            cost_ns - cost,
            legs_ns as i64 - legs as i64
        );
    }
}

/// [`pinned_spread_plan`]'s options.
fn pinned_spread_plan_options() -> crate::ProofOptions {
    let pair = crate::zf_format::ZfFormat {
        logup: stark::proof::options::LogUpPolicy::Pair,
        ..crate::zf_format::ZfFormat::DEFAULT
    };
    pair.base_options(crate::recursion::Preset::Blowup4.options())
}

/// [`the_compact_program_form_keeps_a_small_trees_ids`]'s ids by ELF digest.
const SMALL_TREE_IDS: [(&str, &[&str]); 2] = [
    (
        LAPTOP_ELF_DIGEST,
        &[
            "ec0cf01af432b895627d8f30c4242baf927bec4152776dbfb72fc7549207bc6e",
            "56e894da228b62995082187277db3dbcb3a8a5dc8a4e5d3ecbef27d36f2c076b",
        ],
    ),
    (
        FAST_ELF_DIGEST,
        &[
            "739667a9adcb14185565133c71d8c788215133379a6d8eb6c371c2c6233ff9a8",
            "71600bdb4c4ee9098105cd0166d4580fd64cdf8f21509b6e054e71ccc5eb79a1",
        ],
    ),
];

/// [`the_compact_program_form_keeps_every_production_height_tree_id`]'s ids by
/// ELF digest.
const PRODUCTION_HEIGHT_TREE_IDS: [(&str, &[&str]); 2] = [
    (
        LAPTOP_ELF_DIGEST,
        &[
            "193da2aac2d21a96430f5571d56bf290f34d8865d1fa1eeaf10d6717058538c0",
            "b8a136ce48d98a671236a4a0531b4765ce9446f134a1258d814020b56d8c56d2",
            "cbdc9f20884aa0227e7f6515e75592256ea1be9908e5d29416ddb7c00c1c18ba",
            "a71a9f6384d06fa55ff73affe25f9c0a10f2d9e2087bbfe44347d06f7f16db33",
            "56ac2dd62a1e75fc96ac5d600fc664ebc69c29bedb4973a02add0779c58f64f0",
            "bbc20a4e8ebabb8b8789ec97817b8f8702e23bd1c54923cba0a961f62a9d3ce6",
            "8eb7e4a80e9b8ada89769aba3cfb74df56ac57a8db11f21f4e8d009f5da834a6",
            "7706e7ceca21108335c57bff62d1ac964898ddebcb5f221671420610d31c4183",
            "a032aae4d0d21619c2c4ef89fe387759733b670c55c1c3e4ff65a350a5760735",
            "eceac60369a194dc12f3a83321f88d15f990ba05cf1f882d8ea9239692d63219",
        ],
    ),
    (
        FAST_ELF_DIGEST,
        &[
            "93cc53931a1160fc8c109eb6d5f81ecc0cd828223367fcb0a6d2851c0810bb14",
            "2b0b474e7b7f771bb37329b9b752b9c5625e98ba4293b6a6cbc164005cb6a1c1",
            "0ed61e2212f08cf3fb039335af80d5c3145b65a2c74d057ea6a2692398bdef08",
            "092289d251a74532e006eedfd89d111fe5420e20d3eb99f83cbce71d620b812d",
            "7033ec9d5130aa89831052e30c027ac7cb6aeede9bba093654a19705d0ae9fb0",
            "4b398b930a6f65e84762f90909eec00eced87f6159271d986e7e287b32a79cd6",
            "100d981243118b7bded7b4989e377827124c9c435286f23713cb882d46473dea",
            "b9fa1c3e7fd580f0d7fc87f733d282de0fbaab94b6f535475852fae48186be15",
            "95e17686fd5a5b11dab21fd18e1cdd6cc06bf8d185ffbd72f31dad322be40b30",
            "6e1111f64f288f6b4b94ef9baa1277a184ccc976b876ab9ec3e3519d31e64241",
        ],
    ),
];

/// The ELF constants a plan may take from ahead of time (beside the base) are
/// tied to the ELF and the options they were computed under: another ELF's, or
/// another blowup's, are refused, and the matching ones derive the plan.
#[test]
fn the_plan_refuses_elf_constants_of_another_elf_or_options() {
    use super::block_plan::ElfConstants;
    let opts = fixture_block_options();
    let a = crate::test_utils::asm_elf_bytes("poc_rodata_commit");
    let b = crate::test_utils::asm_elf_bytes("test_commit_4");
    let consts = ElfConstants::compute(&a, &opts).expect("the constants compute");
    let shape = honest_fixture_shape(&executor::elf::Elf::load(&a).expect("load the ELF"));
    assert!(
        BlockTreePlan::derive_with(&b, &opts, &shape, &consts).is_err(),
        "another ELF's constants are refused"
    );
    let mut other = opts.clone();
    other.blowup_factor *= 2;
    assert!(
        BlockTreePlan::derive_with(&a, &other, &shape, &consts).is_err(),
        "constants computed under other options are refused"
    );
    let plan = BlockTreePlan::derive_with(&a, &opts, &shape, &consts).expect("the plan derives");
    let inline = BlockTreePlan::derive(&a, &opts, &shape).expect("the plan derives");
    assert_eq!(
        plan.attested(),
        inline.attested(),
        "constants ahead and inline give one plan"
    );
}

/// The ELF constants on a pool of their own, DECODE and the pages at once
/// (`NOEPOCH_ELF_CONSTS`), are the ones computed a page at a time on the pool
/// beside the base, at any width, and they are refused for another ELF or other
/// options as those are.
#[test]
fn the_elf_constants_at_once_are_the_page_at_a_time_ones_and_still_refused() {
    use super::block_plan::ElfConstants;
    let opts = fixture_block_options();
    let a = crate::test_utils::asm_elf_bytes("poc_rodata_commit");
    let b = crate::test_utils::asm_elf_bytes("test_commit_4");
    let serial = ElfConstants::compute(&a, &opts).expect("the constants compute");
    for threads in [1, 3, 8] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .expect("a pool");
        let at_once = pool
            .install(|| ElfConstants::compute_parallel(&a, &opts))
            .expect("the constants compute");
        assert_eq!(at_once, serial, "{threads} thread(s)");
    }
    assert_ne!(
        ElfConstants::compute_parallel(&b, &opts).expect("the constants compute"),
        serial,
        "another ELF's constants differ"
    );
    let at_once = ElfConstants::compute_parallel(&a, &opts).expect("the constants compute");
    let shape = honest_fixture_shape(&executor::elf::Elf::load(&a).expect("load the ELF"));
    assert!(
        BlockTreePlan::derive_with(&b, &opts, &shape, &at_once).is_err(),
        "another ELF's constants are refused"
    );
    let mut other = opts.clone();
    other.blowup_factor *= 2;
    assert!(
        BlockTreePlan::derive_with(&a, &other, &shape, &at_once).is_err(),
        "constants computed under other options are refused"
    );
    let plan = BlockTreePlan::derive_with(&a, &opts, &shape, &at_once).expect("the plan derives");
    let inline = BlockTreePlan::derive(&a, &opts, &shape).expect("the plan derives");
    assert_eq!(
        plan.attested(),
        inline.attested(),
        "constants at once and inline give one plan"
    );
}

/// The block ELF's constants a page at a time on four threads (the pool beside
/// the base, before `NOEPOCH_ELF_CONSTS`) and at once on pools of 4, 8 and 16:
/// the same constants at every width, each with its wall on an idle host.
#[test]
#[ignore = "box tier, production scale: the block ELF's constants (NOEPOCH_ELF)"]
fn the_block_elfs_constants_at_once_are_the_same_at_every_width() {
    use super::block_plan::ElfConstants;
    let path =
        std::env::var("NOEPOCH_ELF").unwrap_or_else(|_| panic!("NOEPOCH_ELF must name a file"));
    let elf = std::fs::read(&path).unwrap_or_else(|e| panic!("NOEPOCH_ELF {path}: {e}"));
    let opts = super::proof::block_base_options();
    let on = |threads: usize, at_once: bool| {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .expect("a pool");
        let t = std::time::Instant::now();
        let consts = pool
            .install(|| {
                if at_once {
                    ElfConstants::compute_parallel(&elf, &opts)
                } else {
                    ElfConstants::compute(&elf, &opts)
                }
            })
            .expect("the constants compute");
        (consts, t.elapsed().as_secs_f64())
    };
    let (serial, secs) = on(4, false);
    println!("ELF CONSTANTS: a page at a time on 4 threads in {secs:.2}s");
    for threads in [4, 8, 16] {
        let (at_once, secs) = on(threads, true);
        println!("ELF CONSTANTS: at once on {threads} threads in {secs:.2}s");
        assert_eq!(at_once, serial, "at once on {threads} threads");
    }
}

/// P1's cost model v2 keeps every production-height leaf's socket rows in one
/// `LFM_HASH` table: over ¾ · 2^18 rows, so `HashChunking` does not split it,
/// and at most 2^18 (v1's 162,000 cap split every one).
#[test]
fn no_production_height_p1_leaf_splits_its_hash_table() {
    use super::airs::lfm_chip_census_with_hasher;
    use stark::proof::options::{BaseFormat, CapPolicy};
    let base = BaseFormat {
        arity4_cap: CapPolicy::Fixed(1),
        ..BaseFormat::P1
    };
    let plan = spread_plan_under(13, 40, &super::proof::block_base_options_for(base));
    assert_eq!(
        plan.cost_model(),
        super::block_plan::P1_PARTITION_COST_MODEL
    );
    let hash = super::airs::LFM_CHIP_NAMES[super::airs::HASH_SLOT];
    for k in 0..plan.partition().num_leaves() {
        let program = plan.leaf_program(k).expect("the leaf emits");
        let rows = program.groups.hash.real_rows;
        assert!(
            rows > 3 << 16 && rows <= 1 << 18,
            "leaf {k}: {rows} socket rows, outside one 2^18 table's unsplit window"
        );
        let census =
            lfm_chip_census_with_hasher(&program, program.hasher(crate::hash_pin::LEGACY_HASHER));
        let tables: Vec<u64> = census
            .iter()
            .filter(|c| c.name == hash)
            .map(|c| c.rows)
            .collect();
        assert_eq!(tables, vec![1u64 << 18], "leaf {k}: its LFM_HASH tables");
    }
}

/// The partition's cost model is part of the verifier's identity (it decides
/// every leaf's instance list), so its output is pinned to its version: a change
/// to the closed form, the fork constant, the cap or the rule's seeds fails
/// here until `PARTITION_COST_MODEL` is bumped with it.
#[test]
fn the_partition_cost_model_is_pinned_to_its_version() {
    // The block base's preset without the environment's format knobs, so the
    // queries — the closed form's main term — are production's.
    let opts = crate::recursion::Preset::Blowup4.options();
    let elf_bytes = crate::test_utils::asm_elf_bytes("poc_rodata_commit");
    let shape = honest_fixture_shape(&executor::elf::Elf::load(&elf_bytes).expect("load the ELF"));
    let plan = BlockTreePlan::derive(&elf_bytes, &opts, &shape).expect("the plan derives");
    assert_eq!(
        (plan.cost_model(), plan.costs()),
        (2, vec![4654, 3884, 3994, 3884, 4544, 4874, 3994, 3554]),
        "the cost model's output moved: bump PARTITION_COST_MODEL with it"
    );
    // v2's rule: the first ECDAS, ECSM and KECCAK seed leaves 1, 2 and 3; their
    // second chunks fill by load with the rest (v1 put them on the same leaves:
    // [[0, 7], [5, 6], [3, 4], [1, 2]]).
    let names = [
        "BITWISE",
        "KECCAK[0]",
        "KECCAK[1]",
        "ECSM[0]",
        "ECSM[1]",
        "ECDAS[0]",
        "ECDAS[1]",
        "CPU[0]",
    ];
    let rule = partition_by_rule(&names, &[1, 50, 50, 40, 40, 40, 40, 30], 4).expect("fills");
    assert_eq!(
        (plan.cost_model(), rule.leaves()),
        (2, &[vec![0, 2], vec![4, 5], vec![3, 6], vec![1, 7]][..]),
        "the rule's output moved: bump PARTITION_COST_MODEL with it"
    );
}

/// The block tree's fan-in is a tree-format constant (every node program and
/// so the top depend on it): four by default, apart from the epoch tree's two.
/// The record block's 8 leaves lay out as two nodes of four under a two-child
/// top; a block of up to four leaves is the top alone.
#[test]
fn the_block_tree_fans_in_four_by_default() {
    use super::block_plan::{BLOCK_FAN_IN, block_fan_in};
    use super::per_table_aggregator::{FAN_IN, tree_shape};
    assert_eq!(BLOCK_FAN_IN, 4, "the block tree's default fan-in");
    assert_eq!(FAN_IN, 2, "the epoch tree keeps its own fan-in");
    if std::env::var_os("NOEPOCH_BLOCK_FAN_IN").is_none() {
        assert_eq!(block_fan_in(), BLOCK_FAN_IN, "no knob: the plan's fan-in");
    }
    let arities = |leaves: usize| -> Vec<Vec<usize>> {
        tree_shape(leaves, BLOCK_FAN_IN)
            .into_iter()
            .map(|l| l.arities)
            .collect()
    };
    assert_eq!(
        arities(8),
        vec![vec![4, 4], vec![2]],
        "the record block's 8 leaves"
    );
    assert_eq!(arities(4), vec![vec![4]], "four leaves: the top alone");
    assert_eq!(arities(3), vec![vec![3]], "three leaves: the top alone");
    assert_eq!(arities(9), vec![vec![4, 4, 1], vec![3]], "a leftover leaf");
}

/// m5: a nonzero pad byte in the public output's last half is refused, over a
/// real block. `test_commit_3` commits three bytes, so the last half carries one
/// pad byte, which the carrier leaf's COMMIT-bus target (`emit_output_bytes`)
/// keeps zero; the honest arenas execute.
#[test]
#[ignore = "proves a VM block (BITWISE is 2^20 rows); box tier"]
fn a_nonzero_pad_byte_in_the_output_is_refused() {
    let opts = fixture_block_options();
    let (elf_bytes, proof) = small_block("test_commit_3", &[], &opts);
    let (rb, ..) = harvest_block(&opts, &elf_bytes, &proof).expect("harvest");
    assert_eq!(
        rb.public_output.len() % 4,
        3,
        "a ragged output: three live bytes in the last half"
    );
    let partition = rb.plan.partition().clone();
    let program = block_leaf_program(&rb, 0);
    let arenas = block_leaf_arenas(&rb, &partition, 0);
    execute(
        &program,
        &arenas,
        &program.hasher(crate::hash_pin::LEGACY_HASHER),
    )
    .unwrap_or_else(|e| panic!("the honest leaf executes: {e:?}"));
    let last = arenas[0].len() - 1;
    let v: u64 = arenas[0][last][0].canonical();
    let mut padded = arenas.clone();
    padded[0][last] = base_word(FE::from(v | (1u64 << 24)));
    assert!(
        execute(
            &program,
            &padded,
            &program.hasher(crate::hash_pin::LEGACY_HASHER)
        )
        .is_err(),
        "a nonzero pad byte in the last output half must not execute"
    );
    println!("BLOCK FIXTURE: a nonzero pad byte in the last output half is refused");
}

// =============================== (c) the bindings =========================

/// Two children's published words under [`BlockLayout::child`], honest: the same
/// id, state and output, and shares that cancel.
fn honest_children(layout: &BlockLayout) -> Vec<Vec<LfmWord>> {
    let id = super::programs::program_id_words(&core::array::from_fn(|i| 3 + i as u8));
    let state: Vec<LfmWord> = (0..layout.state_words as u64)
        .map(|w| [FE::from(11 + w), FE::from(12), FE::from(13), FE::from(14)])
        .collect();
    let out: Vec<LfmWord> = (0..layout.out_halves as u64)
        .map(|i| base_word(FE::from(0x100 + i)))
        .collect();
    let share = FEE::new([FE::from(5u64), FE::from(6u64), FE::from(7u64)]);
    [share, -share]
        .iter()
        .map(|s| {
            let mut words = id.to_vec();
            words.extend(state.iter().copied());
            words.extend(out.iter().copied());
            words.push(ext_word(s));
            assert_eq!(words.len(), layout.total());
            words
        })
        .collect()
}

/// A node's binding step over HINTED child words — the checks and publishes
/// `emit_block_node` runs after verifying its children, with the verification
/// left out so each check can be driven on its own.
fn bindings_program(
    layout: &BlockLayout,
    children: usize,
    top: bool,
    checks: BlockBindings,
) -> LfmProgram {
    let mut b = production_builder();
    let arenas: Vec<_> = (0..children)
        .map(|_| b.declare_arena(8 * layout.total() as u32))
        .collect();
    let zero = b.ext_const(&FEE::zero());
    let legs: Vec<LegCells> = arenas
        .iter()
        .map(|&a| LegCells {
            publics: hint_public_words(&mut b, a, layout.total()),
            z_alpha: (zero, zero),
        })
        .collect();
    bind_and_publish_with(&mut b, &legs, layout, top, checks);
    compile(b.finish())
}

fn run_bindings(
    layout: &BlockLayout,
    words: &[Vec<LfmWord>],
    top: bool,
    checks: BlockBindings,
) -> Result<Vec<LfmWord>, String> {
    let program = bindings_program(layout, words.len(), top, checks);
    let arenas: Vec<Vec<LfmWord>> = words.iter().map(|w| publics_arena(w)).collect();
    execute(
        &program,
        &arenas,
        &program.hasher(crate::hash_pin::LEGACY_HASHER),
    )
    .map(|e| e.public_words.iter().map(|(_, w)| *w).collect())
    .map_err(|e| format!("{e:?}"))
}

/// ★ (c)/(e) Every cross-child check refuses its tamper, and each is the one
/// doing the refusing: with that check removed (`BlockBindings::without`) the same
/// tampered words execute.
///
/// - `id`: a child attesting another ELF's inputs;
/// - `state`: a child that forked from another transcript — another block's
///   statement or roots (a leaf over another block's proof);
/// - `out`: a child claiming another public output;
/// - `bus`: shares that do not cancel at the top — a tampered contribution, or a
///   target subtracted twice or not at all.
#[test]
fn every_block_binding_refuses_its_tamper_and_is_load_bearing() {
    let layout = BlockLayout::child(super::proof_arena::words_per_root(), 3);
    let honest = honest_children(&layout);
    let top_words = run_bindings(&layout, &honest, true, BlockBindings::ALL)
        .expect("honest children bind at the top");
    assert_eq!(
        top_words.len(),
        BlockLayout::top(layout.state_words, 3).total()
    );
    assert_eq!(
        top_words[..],
        honest[0][..layout.total() - 1],
        "the top republishes the claim"
    );
    let mid = run_bindings(&layout, &honest, false, BlockBindings::ALL)
        .expect("honest children bind below the top");
    assert_eq!(
        word_as_ext(mid.last().expect("a sum")).expect("an ext sum"),
        FEE::zero(),
        "a non-top node publishes the children's summed shares"
    );

    let tampers: [(&str, usize); 4] = [
        ("id", layout.id(1)),
        ("state", layout.state(0)),
        ("out", layout.out_half(2)),
        ("bus", layout.sum()),
    ];
    for (check, at) in tampers {
        let mut words = honest.clone();
        words[1][at][0] += FE::from(1u64);
        let refused = run_bindings(&layout, &words, true, BlockBindings::ALL);
        assert!(
            refused.is_err(),
            "`{check}`: a tampered word must be refused by the full binding set"
        );
        let mutant = run_bindings(&layout, &words, true, BlockBindings::without(check));
        assert!(
            mutant.is_ok(),
            "`{check}`: with the check removed the tamper executes — otherwise another \
             check is doing the refusing and this one is not load-bearing: {mutant:?}"
        );
    }
    // Below the top the bus is not closed, so a share that does not cancel is
    // carried up rather than refused.
    let mut words = honest.clone();
    words[1][layout.sum()][0] += FE::from(1u64);
    assert!(run_bindings(&layout, &words, false, BlockBindings::ALL).is_ok());
}

// =============================== the block harvest ========================

/// One block proof, production-accepted, read for leaf emission
/// ([`super::block_tree::BlockWitness`]).
pub(super) use super::block_tree::BlockWitness as RealBlock;

/// Harvest a block proof that production's block verifier accepts.
///
/// The plan comes first ([`BlockTreePlan::derive`], from the shape the proof
/// claims), and the harvest verifies and reads the proof over the plan's own
/// AIR set — `verify_proof_parts`' set. Returns the harvest and the seconds of
/// its two halves — the plan and the host verify, which only refuses (a harness
/// assert, as in every tree driver), and the replay a driver needs.
pub(super) fn harvest_block(
    opts: &crate::ProofOptions,
    elf_bytes: &[u8],
    proof: &crate::VmProof,
) -> Result<(RealBlock, f64, f64), String> {
    harvest_block_with(opts, elf_bytes, proof, true)
}

/// [`harvest_block`], with the host verify run or skipped. Skipping it is for a
/// caller that runs the verifying harvest beside its own work and joins it
/// before it reports anything (`the_block_tree_composes_to_a_top_node`): the
/// assert still decides whether the run counts, it only stops gating the leaves.
pub(super) fn harvest_block_with(
    opts: &crate::ProofOptions,
    elf_bytes: &[u8],
    proof: &crate::VmProof,
    verify: bool,
) -> Result<(RealBlock, f64, f64), String> {
    harvest_block_over(opts, elf_bytes, proof, verify, None)
}

/// [`harvest_block_with`] over ELF constants computed ahead (beside the base),
/// or computed here when `None` ([`super::block_tree::harvest_block_over`]).
/// Prints the split of its first half: the plan, the COMMIT-bus target, the
/// host verify.
pub(super) fn harvest_block_over(
    opts: &crate::ProofOptions,
    elf_bytes: &[u8],
    proof: &crate::VmProof,
    verify: bool,
    consts: Option<&super::block_plan::ElfConstants>,
) -> Result<(RealBlock, f64, f64), String> {
    super::block_tree::harvest_block_over(opts, elf_bytes, proof, verify, consts, &StdoutSink)
}

/// Leaf `k` of the plan — the program the verifier derives.
pub(super) fn block_leaf_program(rb: &RealBlock, k: usize) -> LfmProgram {
    super::block_tree::block_leaf_program(rb, k).unwrap_or_else(|e| panic!("{e}"))
}

/// Leaf `k` of another partition, carrying the COMMIT-bus target when
/// `carries` — a tree the verifier did not derive, for the negatives.
pub(super) fn block_leaf_program_over(
    rb: &RealBlock,
    partition: &BlockPartition,
    k: usize,
    carries: bool,
) -> LfmProgram {
    // A leaf verifies the base under its format's wrap hash, as the plan's do.
    let mut b = LfmBuilder::new().with_wrap_hash(WrapHash::for_base(rb.plan.base()));
    emit_block_leaf_over(&mut b, &rb.plan, partition, k, carries);
    let program = compile(b.finish());
    super::validator::validate(&program).expect("a block leaf must be admissible");
    program
}

/// Leaf `k`'s arenas, in `emit_block_leaf`'s declaration order.
pub(super) fn block_leaf_arenas(
    rb: &RealBlock,
    partition: &BlockPartition,
    k: usize,
) -> Vec<Vec<LfmWord>> {
    super::block_tree::block_leaf_arenas(rb, partition, k).unwrap_or_else(|e| panic!("{e}"))
}

/// A block node over `children` from their PROOF-read shapes, under a
/// weakened binding set — a mutation, for the negatives. Production nodes come
/// from the plan over the children's derived shapes ([`compose_block_tree`]).
pub(super) fn block_node_program_with(
    children: &[&RealChild],
    layout: BlockLayout,
    top: bool,
    checks: BlockBindings,
) -> LfmProgram {
    let shapes: Vec<_> = children.iter().map(|c| child_shape(c)).collect();
    let mut b = production_builder();
    emit_block_node_with(
        &mut b,
        &BlockNodeInputs {
            children: &shapes,
            layout,
            top,
        },
        checks,
    );
    let program = compile(b.finish());
    super::validator::validate(&program).expect("a block node must be admissible");
    program
}

/// A block node's arenas: its children's, in child order.
pub(super) fn block_node_arenas(children: &[&RealChild]) -> Vec<Vec<LfmWord>> {
    super::block_tree::block_node_arenas(children).unwrap_or_else(|e| panic!("{e}"))
}

/// Prove `program` over `arenas` and read the proof back as a child — production
/// must accept it (`real_child_timed`'s assert). Prints one timing line.
pub(super) fn prove_as_child(
    label: &str,
    program: &LfmProgram,
    arenas: &[Vec<LfmWord>],
    opts: &crate::ProofOptions,
) -> RealChild {
    prove_program(label, program, arenas, opts).0
}

/// [`prove_as_child`], keeping the proof — what the tree's final check reads.
pub(super) fn prove_program(
    label: &str,
    program: &LfmProgram,
    arenas: &[Vec<LfmWord>],
    opts: &crate::ProofOptions,
) -> (RealChild, super::proof::LfmProof) {
    prove_program_with(label, program, None, arenas, opts, true)
}

/// [`prove_program`] over the artifacts built ahead, when given
/// ([`super::block_tree::prove_program_with`]). Without `verify_inline` the
/// proof is read back unverified: the caller verifies it elsewhere before
/// anything is reported.
pub(super) fn prove_program_with(
    label: &str,
    program: &LfmProgram,
    built: Option<LfmArtifacts>,
    arenas: &[Vec<LfmWord>],
    opts: &crate::ProofOptions,
    verify_inline: bool,
) -> (RealChild, super::proof::LfmProof) {
    super::block_tree::prove_program_with(
        label,
        program,
        built,
        arenas,
        opts,
        verify_inline,
        &StdoutSink,
    )
    .unwrap_or_else(|e| panic!("{e}"))
}

/// ★ The block tree's FINAL check over the plan the harness holds: the top
/// proof against `expected`, the artifacts of the top program the plan derives
/// with no proof ([`BlockTreePlan::derive_top`]) — never against a program named
/// by whoever produced the proof. The partition is bound only through program
/// identity (a leaf's id commits to its list, each node interns its children's
/// ids), so this is where a tree over any other partition is refused: its top is
/// another program, and its proof fails here. Production's whole verifier,
/// which derives the plan itself, is [`super::block_plan::verify_block_tree`].
pub(super) fn verify_block_top(
    expected: &super::registry::LfmArtifacts,
    top: &super::proof::LfmProof,
    opts: &crate::ProofOptions,
) -> bool {
    super::proof::verify_against_artifacts(expected, &top.proof, &top.public_words, opts)
}

/// Levels above the leaves, as the plan lays them out
/// ([`BlockTreePlan::levels`], fan-in [`super::block_plan::block_fan_in`]), the
/// last level's single node the top. A one-leaf block still gets a top node:
/// the bus closes there.
/// Returns the top child, its proof and each level's wall.
pub(super) fn compose_block_tree(
    plan: &BlockTreePlan,
    leaves: Vec<RealChild>,
    opts: &crate::ProofOptions,
    siblings: usize,
) -> (RealChild, super::proof::LfmProof, Vec<f64>) {
    super::block_tree::compose_block_tree_with(
        plan,
        leaves,
        opts,
        siblings,
        None,
        None,
        &StdoutSink,
    )
    .unwrap_or_else(|e| panic!("{e}"))
}

/// The top node's published words, against the host's: the id, the state, the
/// public output.
pub(super) fn assert_top_claims_the_block(top: &RealChild, rb: &RealBlock) {
    let words: Vec<LfmWord> = top.public_words.iter().map(|(_, w)| *w).collect();
    let mut want = super::programs::program_id_words(&rb.plan.attested().program_id()).to_vec();
    want.push(rb.state);
    want.extend(le_halves(&rb.public_output).into_iter().map(base_word));
    assert_eq!(
        words, want,
        "the top node publishes the block's claim: the attestation id, the state and \
         the public output"
    );
    assert!(
        top_claims(&rb.plan, &top.public_words, &rb.public_output),
        "the block verifier's claim check accepts the honest top"
    );
}

// ============================ fixture-scale blocks ========================

/// The small block's options: a real format at blowup 4 (every table's LDE pairs
/// index at least one bit) with two queries, so the leaves stay small. The base
/// is the harness's (`NOEPOCH_BASE`, `NOEPOCH_P1_CAP`: RPX unset), so the same
/// suites run a Poseidon1 base's leaves (P3a).
fn fixture_block_options() -> crate::ProofOptions {
    let mut opts = super::epoch_tests::from_proof_gate_options();
    opts.format.base = crate::tests::noepoch_block_tests::noepoch_harness_options()
        .format
        .base;
    opts
}

/// A small guest proved as one no-epoch block on the host: every table cut at
/// 2^5 rows (`MaxRowsConfig::small`), so most types run several instances.
fn small_block(name: &str, input: &[u8], opts: &crate::ProofOptions) -> (Vec<u8>, crate::VmProof) {
    small_block_at(name, input, opts, &crate::tables::MaxRowsConfig::small())
}

/// [`small_block`] at other table caps.
fn small_block_at(
    name: &str,
    input: &[u8],
    opts: &crate::ProofOptions,
    max_rows: &crate::tables::MaxRowsConfig,
) -> (Vec<u8>, crate::VmProof) {
    use executor::vm::execution::Executor;
    let elf_bytes = crate::test_utils::asm_elf_bytes(name);
    let program = executor::elf::Elf::load(&elf_bytes).expect("load the ELF");
    let run = Executor::new(&program, input.to_vec())
        .expect("executor")
        .run()
        .expect("run");
    let mut traces = crate::tables::trace_builder::Traces::from_elf_and_logs(
        &program,
        &run.logs,
        max_rows,
        input,
        #[cfg(feature = "disk-spill")]
        stark::storage_mode::StorageMode::Ram,
    )
    .expect("build the traces");
    let mut observed = None;
    let proof = crate::block::prove_block_traces(
        &elf_bytes,
        &program,
        &mut traces,
        opts,
        None,
        stark::residency_mode::ResidencyMode::Retain,
        Vec::new(),
        &mut crate::block::BlockTimes::default(),
        &mut |shape| observed = Some(shape.clone()),
    )
    .expect("the small block proves");
    assert_eq!(
        format!("{:?}", observed.expect("the prover hands out the shape")),
        format!("{:?}", BlockShape::of_proof(&proof)),
        "the shape handed out before the prove is the proof's"
    );
    (elf_bytes, proof)
}

/// ★ (b) Block leaves over a real block proof EXECUTE, and publish exactly the
/// host's values: the attestation id, the transcript state after `z, α`, the
/// public output, and shares of the bus that sum to zero over the leaves.
///
/// `poc_rodata_commit` carries an ELF data page (an ELF-derived preprocessed root
/// besides DECODE's) and a public output (a nonzero COMMIT-bus target). Three
/// leaves (a test partition: the plan's own is one leaf at this size), so the
/// shares are split and the target sits in one of them.
///
/// Then, per leaf, the arena census (every declared word is a proof word the
/// shapes prescribe, and nothing more), and the leaf-level refusals: a tampered
/// `L`, a non-canonical output half, and a leaf emitted from any other part
/// count or contribution flag than the AIR's is another program.
#[test]
#[ignore = "proves a VM block (BITWISE is 2^20 rows); box tier"]
fn block_leaves_execute_over_a_real_block_proof() {
    let opts = fixture_block_options();
    println!(
        "BLOCK FIXTURE BASE: {:?} cap {}",
        opts.format.base.hash, opts.format.base.arity4_cap
    );
    let wrap_opts = super::proof::aggregation_wrap_options();
    let (elf_bytes, proof) = small_block("poc_rodata_commit", &[], &opts);
    assert!(
        crate::block::verify_block(&proof, &elf_bytes, &opts).expect("verifies"),
        "the block verifier accepts the small block"
    );
    let (mut rb, verify, replay) = harvest_block(&opts, &elf_bytes, &proof).expect("harvest");
    assert!(!rb.public_output.is_empty(), "a nonzero COMMIT-bus target");
    assert!(
        !rb.plan.attested().pages.is_empty(),
        "an ELF data page is attested"
    );
    println!(
        "BLOCK FIXTURE: the plan's own partition is {} leaf(s)",
        rb.plan.partition().num_leaves()
    );
    let three = rb.partition_over(3);
    rb.plan = rb.plan.with_partition(three);
    let partition = rb.plan.partition().clone();
    println!(
        "BLOCK FIXTURE: {} instances, verify {verify:.2}s · replay {replay:.2}s, leaves {:?}",
        rb.num_instances(),
        partition.leaves()
    );
    let dw = super::proof_arena::words_per_root();
    let mut total = FEE::zero();
    for k in 0..partition.num_leaves() {
        let program = block_leaf_program(&rb, k);
        let arenas = block_leaf_arenas(&rb, &partition, k);
        let exec = execute(
            &program,
            &arenas,
            &program.hasher(crate::hash_pin::LEGACY_HASHER),
        )
        .unwrap_or_else(|e| panic!("leaf {k} must execute: {e:?}"));
        let got: Vec<LfmWord> = exec.public_words.iter().map(|(_, w)| *w).collect();
        assert_eq!(
            got,
            rb.expected_leaf_publics(&partition, k, k == rb.plan.carrier()),
            "leaf {k} publishes the host's id, state, output and share"
        );
        total += word_as_ext(got.last().expect("a share")).expect("an ext share");

        // ---- the census: the schema is the proof's words, counted from the
        // proof's own blocks, and not one more — a surplus word is where an
        // unbound second copy of a value hides.
        let declared: usize = program.arena_schema.lens.iter().map(|l| *l as usize).sum();
        let mut want = rb.public_output.len().div_ceil(4) + dw * rb.num_instances();
        for &i in partition.leaf(k) {
            let (h, leg) = (&rb.tables[i], &rb.legs[i]);
            want += dw * usize::from(h.aux_root.is_some())
                + usize::from(h.contribution.is_some())
                + dw
                + h.ood_current.len()
                + h.ood_next.len()
                + h.parts.len()
                + dw * h.fri_roots.len()
                + h.fri_coeffs.len()
                + usize::from(h.nonce.is_some())
                + leg.opening_arena().len()
                + leg.fri_arena().len()
                + leg.caps_arena().iter().map(Vec::len).sum::<usize>();
        }
        assert_eq!(
            declared, want,
            "leaf {k}: the arena schema is exactly the proof's words"
        );

        // ---- a tampered `L`: the fork absorbs it, so every later challenge
        // moves and the leaf does not execute.
        if let Some((i, at)) = contribution_arena(&rb, &partition, k) {
            let mut bad = arenas.clone();
            assert_eq!(
                bad[at],
                vec![ext_word(&rb.tables[i].contribution.expect("an L"))],
                "the census found instance {i}'s L"
            );
            bad[at][0][0] += FE::from(1u64);
            assert!(
                execute(
                    &program,
                    &bad,
                    &program.hasher(crate::hash_pin::LEGACY_HASHER)
                )
                .is_err(),
                "leaf {k}: a tampered L of instance {i} must not execute"
            );
        }

        // ---- a moved Merkle word of the leaf's last instance: the last sibling
        // of its openings and of its FRI paths, and its first cap node when a
        // tree is capped (P3a: every arity-4 hint and cap node is bound too).
        let last = arenas.len() - 1;
        let caps = rb.legs[*partition.leaf(k).last().expect("an instance")]
            .caps_arena()
            .is_some();
        let (openings, fri) = if caps {
            (last - 2, last - 1)
        } else {
            (last - 1, last)
        };
        let mut moved = vec![("an opening sibling", openings), ("a FRI sibling", fri)];
        if caps {
            moved.push(("a cap node", last));
        }
        for (what, at) in moved {
            if arenas[at].is_empty() {
                continue;
            }
            let mut bad = arenas.clone();
            let w = if what == "a cap node" {
                0
            } else {
                bad[at].len() - 1
            };
            bad[at][w][1] += FE::from(1u64);
            assert!(
                execute(
                    &program,
                    &bad,
                    &program.hasher(crate::hash_pin::LEGACY_HASHER)
                )
                .is_err(),
                "leaf {k}: {what} moved and the leaf executed"
            );
        }
        println!(
            "BLOCK FIXTURE leaf {k}: {} instances, {} instructions, {declared} arena words, \
             executes",
            partition.leaf(k).len(),
            program.instrs.len()
        );
    }
    assert_eq!(total, FEE::zero(), "the leaves' shares close the bus");

    // ---- a non-canonical output half in the carrier leaf (leaf 0): its COMMIT-bus
    // target pins each half to its four bytes and every pad byte to zero, so a
    // half at or over 2^32, or a nonzero pad byte, is refused.
    let program = block_leaf_program(&rb, 0);
    let arenas = block_leaf_arenas(&rb, &partition, 0);
    let half0: u64 = arenas[0][0][0].canonical();
    let mut over = arenas.clone();
    over[0][0] = base_word(FE::from(half0 + (1u64 << 32)));
    assert!(
        execute(
            &program,
            &over,
            &program.hasher(crate::hash_pin::LEGACY_HASHER)
        )
        .is_err(),
        "an output half at or over 2^32 must not execute"
    );
    let live = rb.public_output.len() % 4;
    if live != 0 {
        let last = arenas[0].len() - 1;
        let v: u64 = arenas[0][last][0].canonical();
        let mut padded = arenas.clone();
        padded[0][last] = base_word(FE::from(v | (1u64 << (8 * live))));
        assert!(
            execute(
                &program,
                &padded,
                &program.hasher(crate::hash_pin::LEGACY_HASHER)
            )
            .is_err(),
            "a nonzero pad byte in the last output half must not execute"
        );
    } else {
        println!("BLOCK FIXTURE: the output fills its last half; no pad byte to tamper");
    }

    // ---- M1: a leaf emitted from any other part count or contribution flag
    // than the AIR's is another program — so a tree over it has another top,
    // refused at the final check (the fixture tree's partition negatives show
    // that refusal on proved trees).
    let (k, i) = (0..partition.num_leaves())
        .flat_map(|k| partition.leaf(k).iter().map(move |&i| (k, i)))
        .find(|&(_, i)| rb.plan.instance(i).challenge.has_contribution)
        .expect("an instance with a bus contribution");
    let honest =
        super::block_plan::artifacts_of(&block_leaf_program(&rb, k), &wrap_opts).program_id;
    type Mutation = fn(&mut super::block_plan::PlannedInstance);
    let mutations: [(&str, Mutation); 2] = [
        ("an inflated part count", |p| {
            p.challenge.num_parts += 1;
            p.verify.quotient.num_composition_parts += 1;
            p.verify.sub.deep.num_composition_parts += 1;
        }),
        ("a dropped contribution", |p| {
            p.challenge.has_contribution = false
        }),
    ];
    for (what, mutate) in mutations {
        let mut plan = BlockTreePlan::derive(&elf_bytes, &opts, &BlockShape::of_proof(&proof))
            .expect("the plan derives")
            .with_partition(partition.clone());
        mutate(plan.instance_mut(i));
        let program = plan.leaf_program(k).expect("the mutant emits");
        let id = super::block_plan::artifacts_of(&program, &wrap_opts).program_id;
        assert_ne!(
            id, honest,
            "leaf {k} emitted from {what} on instance {i} is another program"
        );
        println!("BLOCK FIXTURE: {what} on instance {i} makes leaf {k} another program");
    }
}

/// Instance and arena index of the first `L` leaf `k`'s arenas carry, in
/// [`block_leaf_arenas`]' order.
fn contribution_arena(
    rb: &RealBlock,
    partition: &BlockPartition,
    k: usize,
) -> Option<(usize, usize)> {
    let mut at = 2;
    for &i in partition.leaf(k) {
        let (h, leg) = (&rb.tables[i], &rb.legs[i]);
        at += usize::from(h.aux_root.is_some());
        if h.contribution.is_some() {
            return Some((i, at));
        }
        at += 6 + usize::from(h.nonce.is_some()) + 2 + usize::from(leg.caps_arena().is_some());
    }
    None
}

/// ★ (b)–(e) at fixture scale, PROVED: three leaves, the nodes above them and the
/// top, every proof accepted by production's LFM verifier, the top claiming the
/// block; then the two refusals that need real child proofs, each with its
/// mutation:
///
/// - a leaf proved over ANOTHER block — the same guest and shape, a private input
///   that differs outside the committed bytes, so the public output is equal and
///   only the transcript state tells the blocks apart — refused at the first
///   node by the state check, and accepted with that check removed;
/// - leaves none of which subtracts the COMMIT-bus target (their shares sum to
///   the target, not zero), refused by the top's bus assert, and accepted with
///   that assert removed.
#[test]
#[ignore = "proves two VM blocks, their leaves and nodes; box tier"]
fn the_block_fixture_tree_proves_and_refuses_another_blocks_leaf() {
    let opts = fixture_block_options();
    println!(
        "BLOCK FIXTURE BASE: {:?} cap {}",
        opts.format.base.hash, opts.format.base.arity4_cap
    );
    let wrap_opts = super::proof::aggregation_wrap_options();
    let input_a: Vec<u8> = (0u8..16).collect();
    let mut input_b = input_a.clone();
    input_b[13] ^= 0xff;
    let (elf_bytes, proof_a) = small_block("test_private_input_xpage", &input_a, &opts);
    let (_, proof_b) = small_block("test_private_input_xpage", &input_b, &opts);
    assert_eq!(
        proof_a.public_output, proof_b.public_output,
        "the same claim"
    );
    assert_eq!(
        crate::statement::table_count_values(&proof_a.table_counts),
        crate::statement::table_count_values(&proof_b.table_counts),
        "the same shape"
    );
    let (mut a, ..) = harvest_block(&opts, &elf_bytes, &proof_a).expect("harvest A");
    let (mut b, ..) = harvest_block(&opts, &elf_bytes, &proof_b).expect("harvest B");
    assert_ne!(a.state, b.state, "the blocks differ in their roots");
    // A test partition of three leaves (the plan's own is one at this size), in
    // both plans: the final check below derives over it.
    let partition = a.partition_over(3);
    assert_eq!(partition, b.partition_over(3), "the same partition");
    a.plan = a.plan.with_partition(partition.clone());
    b.plan = b.plan.with_partition(partition.clone());
    let layout = a.plan.child_layout();

    // Leaves of A, and of B; the same programs, since the shapes are equal. The
    // plan's carrier unless `carriers` names others (the negatives).
    let leaves = |rb: &RealBlock, carriers: &[usize], tag: &str| -> Vec<RealChild> {
        (0..partition.num_leaves())
            .map(|k| {
                let carries = carriers.contains(&k);
                let program = if carriers == [rb.plan.carrier()] {
                    block_leaf_program(rb, k)
                } else {
                    block_leaf_program_over(rb, &partition, k, carries)
                };
                let child = prove_as_child(
                    &format!("{tag} leaf {k}"),
                    &program,
                    &block_leaf_arenas(rb, &partition, k),
                    &wrap_opts,
                );
                let words: Vec<LfmWord> = child.public_words.iter().map(|(_, w)| *w).collect();
                assert_eq!(words, rb.expected_leaf_publics(&partition, k, carries));
                child
            })
            .collect()
    };
    let leaves_a = leaves(&a, &[0], "A");
    let leaves_b = leaves(&b, &[0], "B");
    assert_eq!(
        leaves_a[1].artifacts.program_id, leaves_b[1].artifacts.program_id,
        "a leaf's program is a function of the shape, so B's leaf 1 is A's program"
    );

    // ---- another block's leaf, beside A's leaf 0, at the first node.
    let mixed = [&leaves_a[0], &leaves_b[1]];
    let node = |checks: BlockBindings| {
        let program = block_node_program_with(&mixed, layout, false, checks);
        execute(
            &program,
            &block_node_arenas(&mixed),
            &crate::hash_pin::LEGACY_HASHER,
        )
    };
    assert!(
        node(BlockBindings::ALL).is_err(),
        "a node over two blocks' leaves must not execute"
    );
    assert!(
        node(BlockBindings::without("state")).is_ok(),
        "with the state check removed it executes: the state check is what refuses it"
    );

    // ---- no leaf, or two, carry the target: the shares sum to E or to −E, and
    // the top refuses.
    for (what, carriers) in [("no leaf", &[][..]), ("two leaves", &[0, 1][..])] {
        let wrong_leaves = leaves(&a, carriers, &format!("A carried by {what}"));
        let wrong: Vec<&RealChild> = wrong_leaves.iter().collect();
        let top_over = |checks: BlockBindings| {
            let program = block_node_program_with(&wrong, layout, true, checks);
            execute(
                &program,
                &block_node_arenas(&wrong),
                &crate::hash_pin::LEGACY_HASHER,
            )
        };
        assert!(
            top_over(BlockBindings::ALL).is_err(),
            "a top over shares {what} carries must not execute: they do not close the bus"
        );
        assert!(
            top_over(BlockBindings::without("bus")).is_ok(),
            "{what} carrying: with the bus assert removed it executes, so the bus assert \
             is what refuses it"
        );
    }

    // ---- the honest tree over A, and the final check against the top program
    // the verifier derives from the partition.
    let (top, top_proof, walls) = compose_block_tree(&a.plan, leaves_a, &wrap_opts, 1);
    assert_top_claims_the_block(&top, &a);
    let expected = a.plan.derive_top(&wrap_opts).expect("the top derives");
    assert_eq!(
        expected.program_id, top.artifacts.program_id,
        "the plan derives, with no proof, the top program the harness proved"
    );
    assert!(verify_block_top(&expected, &top_proof, &wrap_opts));
    println!("BLOCK FIXTURE TREE: levels {walls:?}, the top claims block A");

    // ---- a tree over ANOTHER partition of the same block: one instance moved from
    // leaf 1 to leaf 2. Every proof in it is honest and its top claims the same
    // block — only its program identity differs, and the final check refuses it.
    let mut lists = partition.leaves().to_vec();
    let moved = lists[1].pop().expect("leaf 1 has instances");
    assert!(!lists[1].is_empty(), "leaf 1 keeps an instance");
    lists[2].push(moved);
    let other = BlockPartition::new(lists, a.num_instances()).expect("still a partition");
    let other_leaves: Vec<RealChild> = (0..other.num_leaves())
        .map(|k| {
            prove_as_child(
                &format!("A other-partition leaf {k}"),
                &block_leaf_program_over(&a, &other, k, k == 0),
                &block_leaf_arenas(&a, &other, k),
                &wrap_opts,
            )
        })
        .collect();
    let (other_top, other_proof, _) = compose_block_tree(&a.plan, other_leaves, &wrap_opts, 1);
    assert_eq!(
        other_top.public_words, top.public_words,
        "the other partition's top claims the same block"
    );
    assert_ne!(other_top.artifacts.program_id, top.artifacts.program_id);
    assert!(
        !verify_block_top(&expected, &other_proof, &wrap_opts),
        "a tree over another partition must be refused at the final check"
    );
    assert!(
        verify_block_top(&other_top.artifacts, &other_proof, &wrap_opts),
        "checked against the program its own prover built, it verifies: the final check's \
         program, derived by the verifier from the partition constant, is what refuses it"
    );
    println!("BLOCK FIXTURE TREE: another partition's tree is refused at the final check");

    // ---- the dangerous trees: one that SKIPS an instance (its constraints are
    // never verified) and one that verifies an instance TWICE. `BlockPartition::new`
    // refuses both list sets, but nothing binds a prover to it. The instance has a
    // zero bus contribution, so leaving it out or counting it twice keeps the bus
    // closed: every in-circuit check passes, the tree proves, and its top claims
    // the same block. Only the final check's program, derived from the verifier's
    // own partition, refuses it.
    let n = a.num_instances();
    let lists = partition.leaves().to_vec();
    let (k_leaf, skipped) = lists
        .iter()
        .enumerate()
        .filter(|(_, l)| l.len() > 1)
        .flat_map(|(j, l)| l.iter().map(move |&i| (j, i)))
        .find(|&(_, i)| a.tables[i].contribution.is_none_or(|c| c == FEE::zero()))
        .expect(
            "the skip/duplicate negatives need an instance with a zero bus contribution in a \
             leaf of two or more",
        );
    println!(
        "BLOCK FIXTURE TREE: instance {skipped} ({}) of leaf {k_leaf} has a zero bus contribution",
        a.names()[skipped]
    );
    let mut skip = lists.clone();
    skip[k_leaf].retain(|&i| i != skipped);
    let mut twice = lists.clone();
    let other_leaf = (k_leaf + 1) % twice.len();
    twice[other_leaf].push(skipped);
    twice[other_leaf].sort_unstable();
    for (what, lists) in [("skips", skip), ("verifies twice", twice)] {
        assert!(
            BlockPartition::new(lists.clone(), n).is_err(),
            "the emitter's partition refuses a tree that {what} an instance"
        );
        let bad = BlockPartition::unvalidated(lists, n);
        let bad_leaves: Vec<RealChild> = (0..bad.num_leaves())
            .map(|k| {
                prove_as_child(
                    &format!("A tree that {what} instance {skipped}, leaf {k}"),
                    &block_leaf_program_over(&a, &bad, k, k == 0),
                    &block_leaf_arenas(&a, &bad, k),
                    &wrap_opts,
                )
            })
            .collect();
        let (bad_top, bad_proof, _) = compose_block_tree(&a.plan, bad_leaves, &wrap_opts, 1);
        assert_eq!(
            bad_top.public_words, top.public_words,
            "a tree that {what} an instance claims the same block"
        );
        assert_ne!(bad_top.artifacts.program_id, top.artifacts.program_id);
        assert!(
            !verify_block_top(&expected, &bad_proof, &wrap_opts),
            "a tree that {what} an instance must be refused at the final check"
        );
        assert!(
            verify_block_top(&bad_top.artifacts, &bad_proof, &wrap_opts),
            "checked against the program its own prover built, a tree that {what} an \
             instance verifies: the derived top program is what refuses it"
        );
        println!(
            "BLOCK FIXTURE TREE: a tree that {what} instance {skipped} is refused at the final check"
        );
    }
}

/// ★ M1 end to end at fixture scale: the block VERIFIER
/// ([`super::block_plan::verify_block_tree`]) derives the plan and the top
/// program from the trusted ELF, the options and the claimed shape — no proof
/// read — and accepts the honest tree's top proof over the plan's OWN partition
/// and carrier. It refuses the same top proof under another ELF, another
/// claimed output, and a shape that lies about one trace length (each derives
/// another top or another claim). Over ELF constants computed ahead
/// ([`super::block_plan::verify_block_tree_with`]'s path) it accepts the same
/// top, and refuses another ELF's constants.
#[test]
#[ignore = "proves a VM block and its tree; box tier"]
fn the_block_verifier_derives_the_tree_and_accepts_only_its_top() {
    let opts = fixture_block_options();
    let wrap_opts = super::proof::aggregation_wrap_options();
    let (elf_bytes, proof) = small_block("poc_rodata_commit", &[], &opts);
    let shape = BlockShape::of_proof(&proof);
    let (rb, ..) = harvest_block(&opts, &elf_bytes, &proof).expect("harvest");
    let partition = rb.plan.partition().clone();
    println!(
        "BLOCK VERIFIER FIXTURE: the plan's partition is {} leaf(s), {} instances",
        partition.num_leaves(),
        rb.num_instances()
    );
    let leaves: Vec<RealChild> = (0..partition.num_leaves())
        .map(|k| {
            prove_as_child(
                &format!("plan leaf {k}"),
                &block_leaf_program(&rb, k),
                &block_leaf_arenas(&rb, &partition, k),
                &wrap_opts,
            )
        })
        .collect();
    let leaf_artifacts: Vec<LfmArtifacts> = leaves.iter().map(|c| c.artifacts.clone()).collect();
    let (top, top_proof, _) = compose_block_tree(&rb.plan, leaves, &wrap_opts, 1);
    assert_top_claims_the_block(&top, &rb);

    // The tree derived ahead (`NOEPOCH_TREE_AHEAD`'s path: every program and its
    // artifacts, built on host-only threads) is the tree the harness proved: each
    // leaf's artifacts equal the device-built ones field by field, and the top is
    // the proved top.
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(2)
        .start_handler(|_| super::commit::mark_thread_host_only())
        .build()
        .expect("a host-only pool");
    let (tree, _) = pool
        .install(|| {
            rb.plan.derive_tree(&wrap_opts, &|program| {
                super::registry::build_artifacts_with_hasher(
                    program,
                    &wrap_opts,
                    crate::hash_pin::LEGACY_HASHER,
                )
            })
        })
        .expect("the tree derives ahead");
    let ahead_leaves: Vec<&LfmArtifacts> = tree[0].iter().map(|(_, a)| a).collect();
    assert_eq!(
        ahead_leaves,
        leaf_artifacts.iter().collect::<Vec<_>>(),
        "the host-built leaf artifacts ahead are the proved leaves'"
    );
    let ahead_top = &tree.last().expect("a top level")[0].1;
    assert_eq!(
        ahead_top.program_id, top.artifacts.program_id,
        "the tree ahead closes to the proved top"
    );
    println!(
        "BLOCK VERIFIER FIXTURE: the tree derived ahead on the host is the proved tree ({} \
         programs; leaf artifacts equal field by field; top id equal)",
        tree.iter().map(Vec::len).sum::<usize>()
    );

    let verify = |elf: &[u8], shape: &BlockShape, output: &[u8]| {
        super::block_plan::verify_block_tree_under(
            elf, &opts, &wrap_opts, None, shape, output, &top_proof,
        )
    };
    let derived =
        verify(&elf_bytes, &shape, &proof.public_output).expect("the verifier accepts the block");
    assert_eq!(
        derived, top.artifacts.program_id,
        "the verifier's top is the program the harness proved"
    );

    let other_elf = crate::test_utils::asm_elf_bytes("test_private_input_xpage");
    let other = verify(&other_elf, &shape, &proof.public_output);
    assert!(
        other.is_err(),
        "another ELF's verifier must refuse the block"
    );
    let mut output = proof.public_output.clone();
    output[0] ^= 1;
    assert!(
        verify(&elf_bytes, &shape, &output).is_err(),
        "another claimed output must be refused"
    );
    let mut lie = shape.clone();
    let i = (0..lie.trace_lengths.len())
        .find(|&i| !rb.plan.instance(i).name.starts_with("PAGE") && lie.trace_lengths[i] < 1 << 10)
        .expect("a small table");
    lie.trace_lengths[i] *= 2;
    assert!(
        verify(&elf_bytes, &lie, &proof.public_output).is_err(),
        "a shape that lies about instance {i}'s trace length must be refused"
    );
    println!(
        "BLOCK VERIFIER FIXTURE: accepts the plan's tree; refuses another ELF ({}), another \
         output, a trace-length lie",
        other.err().unwrap_or_default()
    );

    let with = |consts: &super::block_plan::ElfConstants| {
        super::block_plan::verify_block_tree_under(
            &elf_bytes,
            &opts,
            &wrap_opts,
            Some(consts),
            &shape,
            &proof.public_output,
            &top_proof,
        )
    };
    let consts = super::block_plan::ElfConstants::compute(&elf_bytes, &opts).expect("constants");
    assert_eq!(
        with(&consts).expect("the verifier accepts the block over cached constants"),
        derived,
        "cached constants derive the same top"
    );
    let theirs = super::block_plan::ElfConstants::compute(&other_elf, &opts).expect("constants");
    let refused = with(&theirs);
    assert!(refused.is_err(), "another ELF's constants must be refused");
    println!(
        "BLOCK VERIFIER FIXTURE: cached ELF constants accepted; another ELF's constants refused \
         ({})",
        refused.err().unwrap_or_default()
    );
}

/// ★ LogUp k4 through the block's recursion (D-LOGUP S2, G2 and N6 at fixture
/// scale): a small block proved under `LAMBDA_VM_ZF_LOGUP=k4` (tables committing
/// four interactions per aux column, four composition parts) verifies on the
/// host under the k4 options and is refused under the pair options (shape: the
/// verifier's format, never the proof's). Its leaves verify the k4 base proof
/// in the guest; the block verifier derives the tree from the k4 options and
/// accepts the proved top, while the same top derived under the pair options is
/// another program and is refused.
#[test]
#[ignore = "proves a VM block and its tree; box tier"]
fn the_block_tree_verifies_logup_k4_and_refuses_it_under_pairs() {
    use stark::proof::options::{LogUpPolicy, ProofFormat};
    let pair = fixture_block_options();
    let k4 = crate::ProofOptions {
        format: ProofFormat {
            logup: LogUpPolicy::K4,
            ..pair.format
        },
        ..pair.clone()
    };
    let wrap_opts = super::proof::aggregation_wrap_options();
    assert_eq!(
        wrap_opts.format.logup,
        LogUpPolicy::Pair,
        "the LFM chips keep pairs"
    );
    let (elf_bytes, proof) = small_block("poc_rodata_commit", &[], &k4);
    let four = proof
        .proof
        .proofs
        .iter()
        .filter(|p| p.composition_poly_parts_ood_evaluation.len() == 4)
        .count();
    assert!(
        four > 0,
        "no sub-proof carries four composition parts under k4"
    );
    assert!(
        matches!(
            crate::block::verify_block(&proof, &elf_bytes, &k4),
            Ok(true)
        ),
        "the k4 block proof verifies under the k4 options"
    );
    assert!(
        !matches!(
            crate::block::verify_block(&proof, &elf_bytes, &pair),
            Ok(true)
        ),
        "N3: the k4 block proof must be refused under the pair options"
    );
    println!(
        "BLOCK LOGUP K4 FIXTURE: {four} of {} sub-proofs carry 4 parts; host verify under k4          accepts, under pairs refuses",
        proof.proof.proofs.len()
    );

    let shape = BlockShape::of_proof(&proof);
    let (rb, ..) = harvest_block(&k4, &elf_bytes, &proof).expect("harvest under k4");
    let partition = rb.plan.partition().clone();
    let leaves: Vec<RealChild> = (0..partition.num_leaves())
        .map(|k| {
            prove_as_child(
                &format!("k4 leaf {k}"),
                &block_leaf_program(&rb, k),
                &block_leaf_arenas(&rb, &partition, k),
                &wrap_opts,
            )
        })
        .collect();
    let (top, top_proof, _) = compose_block_tree(&rb.plan, leaves, &wrap_opts, 1);
    assert_top_claims_the_block(&top, &rb);
    let verify = |opts: &crate::ProofOptions| {
        super::block_plan::verify_block_tree_under(
            &elf_bytes,
            opts,
            &wrap_opts,
            None,
            &shape,
            &proof.public_output,
            &top_proof,
        )
    };
    let derived = verify(&k4).expect("the verifier accepts the k4 block tree");
    assert_eq!(
        derived, top.artifacts.program_id,
        "the verifier's k4 top is the program the harness proved"
    );
    let under_pairs = verify(&pair);
    assert!(
        under_pairs.is_err(),
        "N6: the k4 tree's top must be refused against the plan derived under pairs"
    );
    println!(
        "BLOCK LOGUP K4 FIXTURE: {} leaf(s) verify the k4 base in the guest; the verifier derives          the k4 tree and accepts its top (derived = proved); under pairs it refuses ({})",
        partition.num_leaves(),
        under_pairs.err().unwrap_or_default()
    );
}

/// ★ ECDAS chunked through the block's recursion: test_ecsm_multi's 42 ECDAS
/// steps at 16 rows a chunk (three instances; the 0xABCDEF call runs through
/// all three) prove as a block whose tree the block verifier derives and
/// accepts. The same proof's shape with one ECDAS instance declared over
/// [`crate::BLOCK_ECDAS_MAX_ROWS`] is refused before any program is derived.
#[test]
#[ignore = "proves a VM block and its tree; box tier"]
fn the_block_tree_verifies_a_chunked_ecdas() {
    let opts = fixture_block_options();
    let wrap_opts = super::proof::aggregation_wrap_options();
    let max_rows = crate::tables::MaxRowsConfig {
        ecdas: 16,
        ..crate::tables::MaxRowsConfig::small()
    };
    let (elf_bytes, proof) = small_block_at("test_ecsm_multi", &[], &opts, &max_rows);
    let shape = BlockShape::of_proof(&proof);
    assert_eq!(shape.table_counts.ecdas, 3, "three ECDAS instances");
    let (rb, ..) = harvest_block(&opts, &elf_bytes, &proof).expect("harvest");
    let partition = rb.plan.partition().clone();
    let leaves: Vec<RealChild> = (0..partition.num_leaves())
        .map(|k| {
            prove_as_child(
                &format!("plan leaf {k}"),
                &block_leaf_program(&rb, k),
                &block_leaf_arenas(&rb, &partition, k),
                &wrap_opts,
            )
        })
        .collect();
    let (top, top_proof, _) = compose_block_tree(&rb.plan, leaves, &wrap_opts, 1);
    assert_top_claims_the_block(&top, &rb);
    let verify = |shape: &BlockShape| {
        super::block_plan::verify_block_tree_under(
            &elf_bytes,
            &opts,
            &wrap_opts,
            None,
            shape,
            &proof.public_output,
            &top_proof,
        )
    };
    let derived = verify(&shape).expect("the verifier accepts the chunked block");
    assert_eq!(derived, top.artifacts.program_id);

    let c = &shape.table_counts;
    let first_ecdas = crate::FIXED_TABLE_COUNT + c.commit + c.keccak + c.keccak_rnd + c.ecsm;
    assert!(rb.plan.instance(first_ecdas).name.starts_with("ECDAS["));
    let mut tall = shape.clone();
    tall.trace_lengths[first_ecdas] = 2 * crate::BLOCK_ECDAS_MAX_ROWS;
    let refused = verify(&tall);
    assert!(
        refused.is_err(),
        "an ECDAS instance declared over its cap must be refused"
    );
    println!(
        "BLOCK TREE ECDAS CHUNKED: {} leaf(s), 3 ECDAS instances, accepted; an ECDAS over its cap \
         refused ({})",
        partition.num_leaves(),
        refused.err().unwrap_or_default()
    );
}

/// ★ KECCAK and ECSM chunked through the block's recursion: test_keccak_multi's
/// three permutations and test_ecsm_multi's three scalar multiplications, one
/// call a chunk (three instances each), prove as blocks whose trees the block
/// verifier derives and accepts. Each proof's shape with its first chunk
/// declared over its cap ([`crate::BLOCK_KECCAK_MAX_ROWS`],
/// [`crate::BLOCK_ECSM_MAX_ROWS`]) is refused before any program is derived.
#[test]
#[ignore = "proves two VM blocks and their trees; box tier"]
fn the_block_tree_verifies_chunked_keccak_and_ecsm() {
    let opts = fixture_block_options();
    let wrap_opts = super::proof::aggregation_wrap_options();
    let small = crate::tables::MaxRowsConfig::small;
    for (guest, kind, max_rows, cap) in [
        (
            "test_keccak_multi",
            "KECCAK",
            crate::tables::MaxRowsConfig {
                keccak: 1,
                ..small()
            },
            crate::BLOCK_KECCAK_MAX_ROWS,
        ),
        (
            "test_ecsm_multi",
            "ECSM",
            crate::tables::MaxRowsConfig { ecsm: 1, ..small() },
            crate::BLOCK_ECSM_MAX_ROWS,
        ),
    ] {
        let (elf_bytes, proof) = small_block_at(guest, &[], &opts, &max_rows);
        let shape = BlockShape::of_proof(&proof);
        let c = &shape.table_counts;
        let (count, first) = match kind {
            "KECCAK" => (c.keccak, crate::FIXED_TABLE_COUNT + c.commit),
            _ => (
                c.ecsm,
                crate::FIXED_TABLE_COUNT + c.commit + c.keccak + c.keccak_rnd,
            ),
        };
        assert_eq!(count, 3, "{kind}: one call a chunk");
        let (rb, ..) = harvest_block(&opts, &elf_bytes, &proof).expect("harvest");
        let partition = rb.plan.partition().clone();
        let leaves: Vec<RealChild> = (0..partition.num_leaves())
            .map(|k| {
                prove_as_child(
                    &format!("plan leaf {k}"),
                    &block_leaf_program(&rb, k),
                    &block_leaf_arenas(&rb, &partition, k),
                    &wrap_opts,
                )
            })
            .collect();
        let (top, top_proof, _) = compose_block_tree(&rb.plan, leaves, &wrap_opts, 1);
        assert_top_claims_the_block(&top, &rb);
        let verify = |shape: &BlockShape| {
            super::block_plan::verify_block_tree_under(
                &elf_bytes,
                &opts,
                &wrap_opts,
                None,
                shape,
                &proof.public_output,
                &top_proof,
            )
        };
        let derived = verify(&shape).expect("the verifier accepts the chunked block");
        assert_eq!(derived, top.artifacts.program_id);

        assert_eq!(rb.plan.instance(first).name, format!("{kind}[0]"));
        let mut tall = shape.clone();
        tall.trace_lengths[first] = 2 * cap;
        let refused = verify(&tall);
        assert!(
            refused.is_err(),
            "a {kind} instance declared over its cap must be refused"
        );
        println!(
            "BLOCK TREE {kind} CHUNKED: {} leaf(s), 3 {kind} instances, accepted; a {kind} over \
             its cap refused ({})",
            partition.num_leaves(),
            refused.err().unwrap_or_default()
        );
    }
}

// ============================ production scale ============================

/// D-NOEPOCH §12.2's partition of block 25368371's 137 instances, as the
/// designer's script printed it: the rule over `legmodel.py`'s costs, which
/// differ from the closed form the plan prices with. Reported beside the plan's
/// partition, never substituted for it.
const D12_LEAVES: [&[usize]; 8] = [
    &[
        0, 1, 2, 3, 4, 5, 7, 13, 49, 61, 69, 77, 85, 93, 101, 109, 117, 125, 133,
    ],
    &[
        8, 12, 37, 45, 53, 62, 70, 78, 86, 94, 103, 111, 119, 127, 136,
    ],
    &[
        9, 11, 35, 42, 50, 59, 67, 75, 83, 91, 99, 105, 113, 121, 129,
    ],
    &[
        6, 10, 34, 41, 48, 54, 60, 68, 76, 84, 92, 100, 108, 116, 124, 132,
    ],
    &[
        14, 18, 22, 26, 31, 38, 46, 55, 63, 71, 79, 87, 95, 106, 114, 122, 130, 134,
    ],
    &[
        15, 19, 23, 27, 32, 39, 43, 51, 58, 66, 74, 82, 90, 98, 107, 115, 123, 131,
    ],
    &[
        16, 20, 24, 28, 33, 40, 47, 56, 57, 65, 73, 81, 89, 97, 104, 112, 120, 128,
    ],
    &[
        17, 21, 25, 29, 30, 36, 44, 52, 64, 72, 80, 88, 96, 102, 110, 118, 126, 135,
    ],
];

/// The harness's sink: stdout, and D-NOEPOCH §12.2's partition compared with
/// the plan's on the `BLOCK PARTITION SOURCE` line.
struct HarnessSink;

/// The precomputed-tree download counters when the base proved, so the run's
/// `TREE DOWNLOAD` line splits the base's downloads from the recursion's.
#[cfg(feature = "cuda")]
static TREE_DOWNLOADS_AT_BASE: std::sync::Mutex<Option<math_cuda::device::TreeDownloadTotals>> =
    std::sync::Mutex::new(None);

impl BlockTreeSink for HarnessSink {
    fn write(&self, text: &str) {
        print!("{text}");
    }

    fn base_proved(&self, _proof: &crate::VmProof) {
        #[cfg(feature = "cuda")]
        {
            *TREE_DOWNLOADS_AT_BASE
                .lock()
                .unwrap_or_else(|e| e.into_inner()) =
                Some(math_cuda::device::tree_download_totals());
        }
    }

    fn partition_note(&self, names: &[&str], partition: &BlockPartition) -> Option<String> {
        let d12_block = names.len() == 137 && names[42] == "MEMW[0]" && names[100] == "MEMW_R[0]";
        Some(
            if !d12_block {
                "n/a (not the 137-instance block)"
            } else if partition
                .leaves()
                .iter()
                .map(Vec::as_slice)
                .eq(D12_LEAVES.iter().copied())
            {
                "equal"
            } else {
                "differ (reported, not used)"
            }
            .to_string(),
        )
    }
}

/// ★★★ THE WHOLE NO-EPOCH BLOCK, base to top node, through the one driver the
/// CLI's `prove-block` runs too ([`super::block_tree::prove_block_tree`]):
/// `block::prove_block`, the harvest, the leaves (level 0), the interior and
/// the top — the block's recursion, and the number the epoch/no-epoch decision
/// is taken on.
///
/// Its `★★★ WHOLE RUN` line is measured as the epoch tree's
/// ([`super::per_table_aggregator_tests::the_production_tree_composes_to_a_root`]):
/// from before the base to the last proof, the in-run verifies inside. That
/// tree's number stops at its interior (its block-artifact root is a separate
/// arm); this one's top node is its last proof, and it closes the block's bus.
/// After the run, off its clock: the block verifier, cold and warm.
#[test]
#[ignore = "box tier, production scale: the whole no-epoch block and its recursion (NOEPOCH_ELF, NOEPOCH_INPUT, --features cuda)"]
fn the_block_tree_composes_to_a_top_node() {
    use std::time::Instant;

    // A refused harvest names its table and its check: the verifier's `error!` lines.
    let _ = env_logger::builder().is_test(true).try_init();

    let read = |var: &str| -> Vec<u8> {
        let p = std::env::var(var).unwrap_or_else(|_| panic!("{var} must name a file"));
        std::fs::read(&p).unwrap_or_else(|e| panic!("{var} {p}: {e}"))
    };
    let elf_bytes = read("NOEPOCH_ELF");
    let input = read("NOEPOCH_INPUT");
    let wrap_opts = super::proof::aggregation_wrap_options();
    let mut cfg = super::block_tree::BlockTreeConfig::from_env().unwrap_or_else(|e| panic!("{e}"));
    // The base format the harness's environment names (`NOEPOCH_BASE`,
    // `NOEPOCH_P1_CAP`; RPX unset): test code, the library reads no such
    // variable. Under P1 the static preprocessed roots are computed here,
    // before the run's clock, as the base harness does.
    let base_opts = crate::tests::noepoch_block_tests::noepoch_harness_options();
    cfg.base = base_opts.format.base;
    let p1 = crate::hash_pin::base_of(&base_opts.format) == crate::hash_pin::BaseHash::P1;
    println!(
        "BASE HASH: {} (format {:?}, cap {}) · statement tag {} · cost model {:#x}",
        if p1 { "p1" } else { "rpx" },
        cfg.base.hash,
        cfg.base.arity4_cap,
        String::from_utf8_lossy(&crate::hash_pin::statement_tag(&base_opts.format)),
        super::block_plan::CostModel::for_base(&cfg.base).id
    );
    if p1 {
        crate::hash_pin::warm_base_statics(&base_opts);
    }
    #[cfg(feature = "cuda")]
    let tree_downloads_at_start = math_cuda::device::tree_download_totals();
    let run = super::block_tree::prove_block_tree(
        &elf_bytes,
        &input,
        &cfg,
        std::sync::Arc::new(HarnessSink),
    )
    .unwrap_or_else(|e| panic!("{e}"));
    // The precomputed trees the run downloaded on cache misses, by path, the
    // base's apart from the recursion's (I-COPIES L1).
    #[cfg(feature = "cuda")]
    {
        let end = math_cuda::device::tree_download_totals();
        let at_base = TREE_DOWNLOADS_AT_BASE
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .unwrap_or(tree_downloads_at_start);
        println!(
            "   TREE DOWNLOAD ({}): base {} · recursion {} · staging pair waits {}",
            if math_cuda::device::tree_download_staged() {
                "staged"
            } else {
                "pageable"
            },
            at_base.since(&tree_downloads_at_start).line(),
            end.since(&at_base).line(),
            math_cuda::device::staging_totals().pair_waits
        );
    }
    let super::block_tree::BlockTreeRun {
        shape,
        top_proof,
        top,
        witness: rb,
        consts,
        forced,
        ..
    } = run;
    // `NOEPOCH_SHAPE_OUT=<file>`: the shape a consumer receives, for a verifier
    // run in a process of its own
    // (`the_block_verifiers_derivation_from_a_saved_shape`).
    if let Ok(path) = std::env::var("NOEPOCH_SHAPE_OUT")
        && !path.is_empty()
    {
        std::fs::write(&path, shape_to_text(&shape))
            .unwrap_or_else(|e| panic!("NOEPOCH_SHAPE_OUT {path}: {e}"));
        println!("   BLOCK SHAPE written to {path}");
    }
    // Off the run's clock: the top proof's size, and the plan's partition.
    println!(
        "   BLOCK TOP PROOF: {} B · {} leaves · cost model {:#x}",
        rkyv::to_bytes::<rkyv::rancor::Error>(&top_proof.proof)
            .map(|b| b.len())
            .unwrap_or(0),
        rb.plan.partition().num_leaves(),
        rb.plan.cost_model()
    );
    // The base's main-trace roots in AIR order, as one digest: two runs that
    // build the same traces (under `LAMBDA_VM_FIXED_TRACE_HASH=1`) print the
    // same one, whatever their grinding.
    let mut roots = blake3::Hasher::new();
    for root in &rb.main_roots {
        roots.update(root);
    }
    println!(
        "   BLOCK MAIN ROOTS: {} ({} roots)",
        &roots.finalize().to_hex()[..32],
        rb.main_roots.len()
    );
    // `LAMBDA_VM_ALLOC_PURGE=tree` (or `all`): the tree's freed pages back to
    // the OS before the verifier derives its programs (after the whole run's
    // stopwatch).
    crate::alloc_purge::purge_point("tree");

    // ---- the block VERIFIER, outside the whole run (a consumer's work, not the
    // prover's): `verify_block_tree` under the block presets derives the plan and
    // the top program from the ELF and the claimed shape with no proof read,
    // verifies the top proof against that program, and checks its words against
    // the ELF's id and the output. A forced partition is another tree: then the
    // harness's own plan derives its top instead.
    // Its own device high-water, from the pool (a drain and a trim first, then
    // the high-water restarted), not the card: what the run left reserved
    // would otherwise read as the verifier's.
    #[cfg(feature = "cuda")]
    let pool_at_start = {
        let _ = math_cuda::device::drain_and_trim();
        let live = math_cuda::device::pool_used_bytes()
            .ok()
            .map(|(now, _)| now);
        let _ = math_cuda::device::reset_pool_high_water();
        let _ = super::derive_gate::take_summary();
        live
    };
    let t = Instant::now();
    let mut split = None;
    let derived = match forced {
        None => {
            let (id, times) = super::block_plan::verify_block_tree_timed(
                cfg.base,
                &elf_bytes,
                None,
                &shape,
                &rb.public_output,
                &top_proof,
            )
            .expect("the block verifier accepts the tree");
            split = Some(times);
            id
        }
        Some(_) => {
            let top = rb
                .plan
                .derive_top(&wrap_opts)
                .expect("the forced top derives");
            assert!(verify_block_top(&top, &top_proof, &wrap_opts));
            assert!(top_claims(
                &rb.plan,
                &top_proof.public_words,
                &rb.public_output
            ));
            top.program_id
        }
    };
    assert_eq!(
        derived, top.artifacts.program_id,
        "the verifier derives, with no proof, the top program the harness proved"
    );
    println!(
        "   BLOCK VERIFIER: {} (plan + top program derived, no proof read; the top proof \
         verifies against it and claims the block) in {:.2}s",
        if forced.is_some() {
            "forced-partition"
        } else {
            "verify_block_tree under the block presets"
        },
        t.elapsed().as_secs_f64()
    );
    if let Some(split) = split {
        println!("   BLOCK VERIFIER split (cold): {split}");
    }
    #[cfg(feature = "cuda")]
    {
        let gib = |b: u64| format!("{:.2}", b as f64 / (1u64 << 30) as f64);
        let show = |b: Option<u64>| b.map_or("?".to_string(), gib);
        let (now, high) = math_cuda::device::pool_used_bytes()
            .ok()
            .map_or((None, None), |(now, high)| {
                (Some(now), Some(high.max(pool_at_start.unwrap_or(0))))
            });
        println!(
            "   BLOCK VERIFIER DEVICE (cold): pool high-water {} GiB (live {} at the start after a \
             drain and a trim, {} at the end, reserved {}) · derive gate: {}",
            show(high),
            show(pool_at_start),
            show(now),
            show(math_cuda::device::pool_reserved_bytes().ok()),
            super::derive_gate::take_summary().map_or("not armed".to_string(), |s| s.to_string())
        );
    }
    // Warm: the ELF constants a consumer caches per ELF (the harness's, computed
    // beside the base under the same options).
    if let (None, Some(consts)) = (forced, consts.as_deref()) {
        let t = Instant::now();
        let (id, warm) = super::block_plan::verify_block_tree_timed(
            cfg.base,
            &elf_bytes,
            Some(consts),
            &shape,
            &rb.public_output,
            &top_proof,
        )
        .expect("the block verifier accepts the tree over cached constants");
        assert_eq!(id, derived, "cached constants derive the same top");
        println!(
            "   BLOCK VERIFIER warm (ELF constants cached): {:.2}s ({warm})",
            t.elapsed().as_secs_f64()
        );
    }
}

// ============ the block verifier's derivation in a process of its own ==========

/// The fields of [`crate::TableCounts`], in declaration order, for
/// [`shape_to_text`] and [`shape_from_text`].
fn table_count_fields(c: &mut crate::TableCounts) -> [&mut usize; 21] {
    [
        &mut c.cpu,
        &mut c.lt,
        &mut c.memw,
        &mut c.memw_aligned,
        &mut c.load,
        &mut c.mul,
        &mut c.dvrm,
        &mut c.shift,
        &mut c.branch,
        &mut c.memw_register,
        &mut c.eq,
        &mut c.bytewise,
        &mut c.store,
        &mut c.cpu32,
        &mut c.keccak,
        &mut c.keccak_rnd,
        &mut c.ecsm,
        &mut c.ecdas,
        &mut c.hint,
        &mut c.commit,
        &mut c.blake3,
    ]
}

/// A [`BlockShape`] as five lines of text: the table counts, the runtime page
/// ranges (`base:count`), the private-input pages, the public output's length
/// and the trace lengths.
fn shape_to_text(shape: &BlockShape) -> String {
    let mut counts = shape.table_counts.clone();
    let join = |v: Vec<String>| v.join(" ");
    format!(
        "counts {}\nranges {}\nprivate {}\noutput {}\nlengths {}\n",
        join(
            table_count_fields(&mut counts)
                .iter()
                .map(|c| c.to_string())
                .collect()
        ),
        join(
            shape
                .runtime_page_ranges
                .iter()
                .map(|r| format!("{}:{}", r.base, r.count))
                .collect()
        ),
        shape.num_private_input_pages,
        shape.public_output_len,
        join(shape.trace_lengths.iter().map(|l| l.to_string()).collect()),
    )
}

/// [`shape_to_text`]'s inverse.
fn shape_from_text(text: &str) -> BlockShape {
    let line = |key: &str| -> Vec<&str> {
        let l = text
            .lines()
            .find(|l| l.split(' ').next() == Some(key))
            .unwrap_or_else(|| panic!("the shape has no `{key}` line"));
        l.split(' ').skip(1).filter(|w| !w.is_empty()).collect()
    };
    let num = |w: &str| -> u64 {
        w.parse()
            .unwrap_or_else(|_| panic!("`{w}` is not a number"))
    };
    let mut table_counts = crate::TableCounts {
        cpu: 0,
        lt: 0,
        memw: 0,
        memw_aligned: 0,
        load: 0,
        mul: 0,
        dvrm: 0,
        shift: 0,
        branch: 0,
        memw_register: 0,
        eq: 0,
        bytewise: 0,
        store: 0,
        cpu32: 0,
        keccak: 0,
        keccak_rnd: 0,
        ecsm: 0,
        ecdas: 0,
        hint: 0,
        commit: 0,
        blake3: 0,
    };
    let counts = line("counts");
    assert_eq!(counts.len(), 21, "the shape's counts line has 21 fields");
    for (field, w) in table_count_fields(&mut table_counts)
        .into_iter()
        .zip(counts)
    {
        *field = num(w) as usize;
    }
    BlockShape {
        table_counts,
        runtime_page_ranges: line("ranges")
            .into_iter()
            .map(|w| {
                let (base, count) = w.split_once(':').expect("a range is base:count");
                crate::RuntimePageRange {
                    base: num(base),
                    count: num(count),
                }
            })
            .collect(),
        num_private_input_pages: num(line("private")[0]) as usize,
        public_output_len: num(line("output")[0]) as usize,
        trace_lengths: line("lengths")
            .into_iter()
            .map(|w| num(w) as usize)
            .collect(),
    }
}

/// A shape survives its text form: [`shape_from_text`] of [`shape_to_text`] is
/// the shape, page ranges included.
#[test]
fn a_block_shape_round_trips_through_its_text() {
    let elf_bytes = crate::test_utils::asm_elf_bytes("poc_rodata_commit");
    let elf = executor::elf::Elf::load(&elf_bytes).expect("load the ELF");
    let mut shape = honest_fixture_shape(&elf);
    shape.table_counts.keccak = 3;
    shape.table_counts.blake3 = 1;
    shape.runtime_page_ranges = vec![
        crate::RuntimePageRange {
            base: 0x7000_0000,
            count: 2,
        },
        crate::RuntimePageRange {
            base: 0x8000_0000,
            count: 1,
        },
    ];
    shape.num_private_input_pages = 5;
    shape.trace_lengths.extend([1 << 18, 1 << 17]);
    let back = shape_from_text(&shape_to_text(&shape));
    assert_eq!(format!("{back:?}"), format!("{shape:?}"));
}

/// A fixture plan with eight more CPU instances, cut into six leaves, so a
/// leaf level, an interior level and the top.
fn six_leaf_fixture_plan() -> BlockTreePlan {
    let opts = fixture_block_options();
    let elf_bytes = crate::test_utils::asm_elf_bytes("poc_rodata_commit");
    let elf = executor::elf::Elf::load(&elf_bytes).expect("load the ELF");
    let mut shape = honest_fixture_shape(&elf);
    // The CPU instances follow `CPU[0]`, right after the fixed tables.
    shape.table_counts.cpu += 8;
    let at = crate::FIXED_TABLE_COUNT + 1;
    shape.trace_lengths.splice(at..at, [32; 8]);
    let plan = BlockTreePlan::derive(&elf_bytes, &opts, &shape).expect("the plan derives");
    let names: Vec<&str> = plan.instances().iter().map(|i| i.name.as_str()).collect();
    let partition = partition_by_rule(&names, &plan.costs(), 6).expect("six leaves fill");
    let plan = plan.with_partition(partition);
    assert_eq!(
        plan.levels().len(),
        2,
        "six leaves at fan-in 4: one interior level and the top"
    );
    plan
}

/// The verifier's streaming derivation (each program and its artifacts dropped
/// once its child is derived) derives the top of the tree that keeps every
/// program ([`BlockTreePlan::derive_tree`]), over [`six_leaf_fixture_plan`].
#[test]
#[ignore = "box tier: two derivations of a six-leaf tree under the wrap preset, ≈ 2 min on the laptop"]
fn the_streaming_derivation_derives_the_kept_trees_top() {
    let wrap_opts = super::proof::aggregation_wrap_options();
    let plan = six_leaf_fixture_plan();
    let streamed = plan.derive_top(&wrap_opts).expect("the top derives");
    let (tree, _) = plan
        .derive_tree(&wrap_opts, &|p| {
            super::block_plan::artifacts_of(p, &wrap_opts)
        })
        .expect("the tree derives");
    let kept = &tree.last().expect("a top level")[0].1;
    assert_eq!(streamed.program_id, kept.program_id);
}

/// ★ The derive gate on the card ([`super::derive_gate`]), over
/// [`six_leaf_fixture_plan`]: derived open (a budget nothing reaches), then
/// under a fixed budget of half the open peak, never under the largest set.
/// The gated derivation's running total stays within its budget and waits,
/// and its top is the open one's; each derivation's pool high-water is
/// printed. Then the mutation: a permit held across each build's forks is
/// counted there, and the tree still derives.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "box tier: --features cuda, three derivations of a six-leaf tree"]
fn the_derive_gate_bounds_the_derivation_on_the_card() {
    use super::derive_gate::{Setting, pin_setting, take_summary};
    let gib = |b: u64| b as f64 / (1u64 << 30) as f64;
    let wrap_opts = super::proof::aggregation_wrap_options();
    let plan = six_leaf_fixture_plan();
    let derive = |setting: Setting| {
        pin_setting(Some(setting));
        let _ = math_cuda::device::drain_and_trim();
        let start = math_cuda::device::pool_used_bytes().map_or(0, |(now, _)| now);
        let _ = math_cuda::device::reset_pool_high_water();
        let top = plan.derive_top(&wrap_opts).expect("the top derives");
        let high = math_cuda::device::pool_used_bytes().map_or(0, |(_, high)| high.max(start));
        let summary = take_summary().expect("the derivation armed the gate");
        println!(
            "DERIVE GATE {setting:?}: {summary} · pool high-water {:.3} GiB (live {:.3} at the start)",
            gib(high),
            gib(start)
        );
        (top.program_id, summary)
    };
    let (open_top, open) = derive(Setting::Fixed(u64::MAX));
    assert!(
        open.dispatches > 0,
        "no device set reached the card: the gate is untested here"
    );
    let largest = super::commit::device_artifact_peak_bytes();
    let budget = (open.peak / 2).max(largest);
    assert!(
        open.peak > budget,
        "the open peak {:.3} GiB is one set ({:.3} GiB): no budget binds",
        gib(open.peak),
        gib(largest)
    );
    let (gated_top, gated) = derive(Setting::Fixed(budget));
    assert!(gated.peak <= budget, "the gate held {gated}");
    assert!(gated.waits > 0, "nothing waited: {gated}");
    assert_eq!(gated.dispatches, open.dispatches);
    assert_eq!(gated_top, open_top, "the gate is scheduling only");

    assert_eq!((gated.reentries, gated.forks_held), (0, 0), "{gated}");

    // ⛔ The mutation: each build holds a permit across its walk's forks. Every
    // fork is counted, and a job stolen there that asks again passes through:
    // the tree still derives, to the same top.
    pin_setting(Some(Setting::Fixed(u64::MAX)));
    let armed = super::derive_gate::arm().expect("the gate arms");
    let (tree, _) = plan
        .derive_tree(&wrap_opts, &|p| {
            let _held = super::derive_gate::admit(0).expect("armed");
            super::block_plan::artifacts_of(p, &wrap_opts)
        })
        .expect("the tree derives under the mutation");
    drop(armed);
    pin_setting(None);
    let held = take_summary().expect("the mutation armed the gate");
    println!("DERIVE GATE mutation (a permit around each build): {held}");
    assert!(
        held.forks_held > 0,
        "no fork under a held permit was counted: {held}"
    );
    assert_eq!(
        tree.last().expect("a top level")[0].1.program_id,
        open_top,
        "the mutation changes scheduling only"
    );
}

/// ★ The block verifier's derivation ([`BlockTreePlan::derive_top`] under the
/// block presets, what [`super::block_plan::verify_block_tree`] runs before its
/// final check) in a process of its own, over a shape a whole-block run saved
/// (`NOEPOCH_SHAPE_OUT`): its host peak, where no page the tree freed hides it,
/// and the top id, which must be the whole run's. `LAMBDA_VM_BLOCK_DERIVE_HOLD`
/// picks how a level holds its programs.
#[test]
#[ignore = "box tier: NOEPOCH_ELF and NOEPOCH_SHAPE (a whole-block run's NOEPOCH_SHAPE_OUT), --features cuda"]
fn the_block_verifiers_derivation_from_a_saved_shape() {
    use super::per_table_aggregator_tests::HostSampler;
    use std::time::Instant;
    let path = |var: &str| std::env::var(var).unwrap_or_else(|_| panic!("{var} must name a file"));
    let elf_bytes = std::fs::read(path("NOEPOCH_ELF")).expect("read NOEPOCH_ELF");
    let shape = shape_from_text(
        &std::fs::read_to_string(path("NOEPOCH_SHAPE")).expect("read NOEPOCH_SHAPE"),
    );
    let hold = std::env::var("LAMBDA_VM_BLOCK_DERIVE_HOLD").unwrap_or_default();
    let opts = super::proof::block_base_options();
    let wrap_opts = super::proof::aggregation_wrap_options();
    let sampler = HostSampler::start();
    let t = Instant::now();
    let consts =
        super::block_plan::ElfConstants::compute(&elf_bytes, &opts).expect("the constants compute");
    let constants = t.elapsed().as_secs_f64();
    let t = Instant::now();
    let plan =
        BlockTreePlan::derive_with(&elf_bytes, &opts, &shape, &consts).expect("the plan derives");
    let planned = t.elapsed().as_secs_f64();
    let t = Instant::now();
    let (top, levels) = plan.derive_top_timed(&wrap_opts).expect("the top derives");
    let derived = t.elapsed().as_secs_f64();
    let (peak, _) = sampler.stop();
    let split: Vec<String> = levels
        .iter()
        .map(|p| {
            format!(
                "{} in {:.2} (emit Σ {:.2}, build Σ {:.2})",
                p.programs, p.wall, p.emit, p.build
            )
        })
        .collect();
    println!(
        "BLOCK VERIFIER ONLY: hold `{}` · {} leaves · constants {constants:.2}s · plan {planned:.2}s · derive \
         {derived:.2}s ({}) · host peak {peak:.3} GiB · top program id {}",
        if hold.is_empty() {
            "stream"
        } else {
            hold.as_str()
        },
        plan.partition().num_leaves(),
        split.join(" · "),
        top.program_id
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect::<String>()
    );
}
