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
use stark::proof::view::MultiProofView;

use crate::tables::types::{FE, FEE, GoldilocksExtension, GoldilocksField};

use super::LfmArtifacts;
use super::block_leaf::{BlockPartition, emit_block_leaf_over, partition_by_rule};
use super::block_node::{
    BlockBindings, BlockLayout, BlockNodeInputs, bind_and_publish_with, emit_block_node_with,
};
use super::block_plan::{BlockShape, BlockTreePlan, le_halves, top_claims};
use super::block_replay::{BlockStatementShape, replay_block_front};
use super::block_tree_pipeline::Pipe;
use super::builder::LfmBuilder;
use super::compiler::{LfmProgram, compile};
use super::edsl::WrapHash;
use super::epoch_tests::HostTable;
use super::epoch_verify_tests::TableLegs;
use super::executor::execute;
use super::per_table_aggregator::{DerivedChild, LegCells, hint_public_words, publics_arena};
use super::per_table_aggregator_tests::{RealChild, child_arena_words, child_shape};
use super::statement_replay::{NUM_TABLE_COUNTS, PhaseAPreprocessed, PhaseATable};
use super::word::{LfmWord, base_word, ext_word, word_as_ext};

type Gl = GoldilocksField;
type Ext3 = GoldilocksExtension;

fn production_builder() -> LfmBuilder {
    LfmBuilder::new().with_wrap_hash(WrapHash::production())
}

/// Bytes as the arena's `u32` halves: four bytes each, little-endian, the last
/// zero-padded.
/// The host transcript every block prover and verifier starts from:
/// `StatementKind::Monolithic` over the block's statement fields.
fn block_seed(
    elf_digest: &[u8; 32],
    public_output: &[u8],
    table_counts: &crate::TableCounts,
    num_private_input_pages: usize,
    runtime_page_ranges: &[crate::RuntimePageRange],
    fri_final_poly_log_degree: u8,
) -> crate::hash_pin::BlockTranscript {
    let mut t = crate::hash_pin::block_transcript(&[]);
    crate::statement::absorb_statement_with_digest(
        &mut t,
        crate::statement::StatementKind::Monolithic,
        elf_digest,
        public_output,
        table_counts,
        num_private_input_pages,
        runtime_page_ranges,
        fri_final_poly_log_degree,
    );
    t
}

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
        WrapHash::production(),
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
    let exec = execute(&program, &arenas, &crate::hash_pin::BLOCK_HASHER)
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
        let e = execute(&program, &tampered, &crate::hash_pin::BLOCK_HASHER)
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
    let exec = execute(&program, &honest, &crate::hash_pin::BLOCK_HASHER)
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
            execute(&program, &tampered, &crate::hash_pin::BLOCK_HASHER).is_err(),
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

/// jemalloc's `opt.oversize_threshold` (5.3's default; the instruments read it
/// back): an allocation of at least this many bytes comes from one arena every
/// thread shares, a smaller one from its thread's own arena. The pages a freed
/// buffer leaves behind serve only its arena's later requests.
const JEMALLOC_OVERSIZE: usize = 8 << 20;

/// A program's host bytes as held (capacities, not lengths), by part, and the
/// share in allocations at or over [`JEMALLOC_OVERSIZE`]: the part that can
/// take the pages a freed trace left in the shared arena.
#[derive(Default, Clone, Copy)]
struct ProgramBytes {
    /// The instruction vector itself.
    instrs: usize,
    /// `Instr::BitDec`'s bit lists.
    bitdec_heap: usize,
    /// `KeccakF`'s and `Blake3`'s boxed operands.
    boxed: usize,
    /// The column groups as held: padded rows, at capacity.
    groups: usize,
    /// The column groups' padded rows (the committed matrices).
    groups_padded: usize,
    /// The column groups' real rows alone.
    groups_real: usize,
    /// The arena schema.
    other: usize,
    /// Of the total, the bytes in allocations at or over [`JEMALLOC_OVERSIZE`].
    large: usize,
    allocs: usize,
}

impl ProgramBytes {
    fn of(p: &LfmProgram) -> Self {
        use super::instr::{Addr, Blake3Operands, Instr, KeccakOperands};
        let mut b = Self::default();
        b.instrs = b.note(p.instrs.capacity() * size_of::<Instr>());
        for instr in &p.instrs {
            match instr {
                Instr::BitDec { bits, .. } => {
                    b.bitdec_heap += b.note(bits.capacity() * size_of::<(Addr, u64)>());
                }
                Instr::KeccakF(_) => b.boxed += b.note(size_of::<KeccakOperands>()),
                Instr::Blake3(_) => b.boxed += b.note(size_of::<Blake3Operands>()),
                _ => {}
            }
        }
        for g in program_groups(p) {
            b.groups += b.note(g.data.capacity() * size_of::<FE>());
            b.groups_padded += g.padded_rows * g.width * size_of::<FE>();
            b.groups_real += g.real_rows * g.width * size_of::<FE>();
        }
        b.other = b.note(p.arena_schema.lens.capacity() * size_of::<u32>());
        b
    }

    /// Counts one allocation of `bytes` and returns them.
    fn note(&mut self, bytes: usize) -> usize {
        if bytes > 0 {
            self.allocs += 1;
            if bytes >= JEMALLOC_OVERSIZE {
                self.large += bytes;
            }
        }
        bytes
    }

    fn total(&self) -> usize {
        self.instrs + self.bitdec_heap + self.boxed + self.groups + self.other
    }

    fn add(&mut self, o: &Self) {
        self.instrs += o.instrs;
        self.bitdec_heap += o.bitdec_heap;
        self.boxed += o.boxed;
        self.groups += o.groups;
        self.groups_padded += o.groups_padded;
        self.groups_real += o.groups_real;
        self.other += o.other;
        self.large += o.large;
        self.allocs += o.allocs;
    }
}

/// A program's column groups with their chip names, in the frozen chip order.
fn program_groups(p: &LfmProgram) -> [&super::compiler::ColumnGroup; 11] {
    let g = &p.groups;
    [
        &g.const_, &g.balu, &g.xalu, &g.select, &g.bitdec, &g.hash, &g.keccak, &g.blake3, &g.lanes,
        &g.hint, &g.public,
    ]
}

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
    let opts = super::proof::block_base_options();
    let elf_bytes = crate::test_utils::asm_elf_bytes("poc_rodata_commit");
    let elf = executor::elf::Elf::load(&elf_bytes).expect("load the ELF");
    let shape = spread_fixture_shape(&elf, chunks, cpus);
    BlockTreePlan::derive(&elf_bytes, &opts, &shape).expect("the plan derives")
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

/// ★ The compact program form changes no program: every id of a small spread
/// plan's tree equals the one recorded at 541f4bdc2 (#1013's d740eb5d5 + the
/// instruments), before the form changed.
///
/// ⚠ The pins are the LAPTOP's: the plan absorbs the ELF's digest, and the
/// fixture ELF's bytes depend on the clang that assembled it (laptop
/// `af87f637…`, 1264 B; FAST `bfb782e1…`, 1272 B), so a box derives other ids
/// for the same programs (FAST 670). The box gate compares two shas on one box.
#[test]
#[ignore = "laptop: run with --exact"]
fn the_compact_program_form_keeps_a_small_trees_ids() {
    let ids = tree_ids(&spread_plan(2, 4));
    for (j, id) in ids.iter().enumerate() {
        println!("SMALL TREE IDS {j}: {id}");
    }
    assert_eq!(ids, SMALL_TREE_IDS, "a program of the tree changed");
}

/// The same at production heights ([`production_height_plan`]: 7 leaves, 2
/// nodes and the top), on a box: ≈ 4.7 min and 4.6 GiB on the laptop's host
/// commit. The box gates add the 1× and median trees' top ids.
#[test]
#[ignore = "box tier: production-height artifacts on the host, ≈ 4.7 min and 4.6 GiB"]
fn the_compact_program_form_keeps_every_production_height_tree_id() {
    let ids = tree_ids(&production_height_plan());
    for (j, id) in ids.iter().enumerate() {
        println!("TREE IDS {j}: {id}");
    }
    assert_eq!(
        ids, PRODUCTION_HEIGHT_TREE_IDS,
        "a program of the tree changed"
    );
}

/// [`the_compact_program_form_keeps_a_small_trees_ids`]'s ids at 541f4bdc2.
const SMALL_TREE_IDS: [&str; 2] = [
    "ec0cf01af432b895627d8f30c4242baf927bec4152776dbfb72fc7549207bc6e",
    "56e894da228b62995082187277db3dbcb3a8a5dc8a4e5d3ecbef27d36f2c076b",
];

/// [`the_compact_program_form_keeps_every_production_height_tree_id`]'s ids at
/// 541f4bdc2.
const PRODUCTION_HEIGHT_TREE_IDS: [&str; 10] = [
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
    execute(&program, &arenas, &crate::hash_pin::BLOCK_HASHER)
        .unwrap_or_else(|e| panic!("the honest leaf executes: {e:?}"));
    let last = arenas[0].len() - 1;
    let v: u64 = arenas[0][last][0].canonical();
    let mut padded = arenas.clone();
    padded[0][last] = base_word(FE::from(v | (1u64 << 24)));
    assert!(
        execute(&program, &padded, &crate::hash_pin::BLOCK_HASHER).is_err(),
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
    execute(&program, &arenas, &crate::hash_pin::BLOCK_HASHER)
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

/// One block proof, production-accepted, read for leaf emission.
///
/// `plan` is the verifier's: every shape and constant a leaf is emitted from,
/// derived from the ELF, the options and the shape the proof claims — never
/// from its data. The other fields are the proof's DATA, which the leaves'
/// arenas carry: every instance's main root, the shared state, and per instance
/// the fork's data ([`HostTable`]) and the legs' openings ([`TableLegs`]), each
/// checked against the AIR's shape as it is read.
pub(super) struct RealBlock {
    pub(super) plan: BlockTreePlan,
    pub(super) public_output: Vec<u8>,
    pub(super) main_roots: Vec<Commitment>,
    pub(super) tables: Vec<HostTable>,
    pub(super) legs: Vec<TableLegs>,
    /// The host transcript's state after `z, α`.
    pub(super) state: LfmWord,
    /// The COMMIT-bus target production computed.
    pub(super) expected_bus_balance: FEE,
}

impl RealBlock {
    pub(super) fn num_instances(&self) -> usize {
        self.plan.num_instances()
    }

    /// AIR names, in AIR order.
    pub(super) fn names(&self) -> Vec<&str> {
        self.plan
            .instances()
            .iter()
            .map(|i| i.name.as_str())
            .collect()
    }

    /// The rule over `num_leaves` leaves: a TEST partition, another tree than
    /// the plan's unless the count is the plan's own.
    pub(super) fn partition_over(&self, num_leaves: usize) -> BlockPartition {
        partition_by_rule(&self.names(), &self.plan.costs(), num_leaves)
            .expect("the rule fills every leaf")
    }

    /// What leaf `k` of `partition` must publish, from the host's own values:
    /// the id, the state, the output halves, and its share of the bus — less
    /// the target when it `carries` it.
    pub(super) fn expected_leaf_publics(
        &self,
        partition: &BlockPartition,
        k: usize,
        carries: bool,
    ) -> Vec<LfmWord> {
        let mut words =
            super::programs::program_id_words(&self.plan.attested().program_id()).to_vec();
        words.push(self.state);
        words.extend(le_halves(&self.public_output).into_iter().map(base_word));
        let mut share = partition
            .leaf(k)
            .iter()
            .filter_map(|&i| self.tables[i].contribution)
            .fold(FEE::zero(), |acc, l| acc + l);
        if carries {
            share = share - self.expected_bus_balance;
        }
        words.push(ext_word(&share));
        words
    }
}

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
/// or computed here when `None`. Prints the split of its first half: the plan,
/// the COMMIT-bus target, the host verify.
pub(super) fn harvest_block_over(
    opts: &crate::ProofOptions,
    elf_bytes: &[u8],
    proof: &crate::VmProof,
    verify: bool,
    consts: Option<&super::block_plan::ElfConstants>,
) -> Result<(RealBlock, f64, f64), String> {
    use crypto::fiat_shamir::is_transcript::IsTranscript;
    use rayon::prelude::*;
    use stark::verifier::IsStarkVerifier;

    let t_verify = std::time::Instant::now();
    let shape = BlockShape::of_proof(proof);
    let plan = match consts {
        Some(c) => BlockTreePlan::derive_with(elf_bytes, opts, &shape, c)?,
        None => BlockTreePlan::derive(elf_bytes, opts, &shape)?,
    };
    let plan_secs = t_verify.elapsed().as_secs_f64();
    let view = MultiProofView::Owned(&proof.proof);
    let refs = plan.airs().air_refs();
    let seed = || {
        block_seed(
            plan.elf_digest(),
            &proof.public_output,
            &proof.table_counts,
            proof.num_private_input_pages,
            &proof.runtime_page_ranges,
            opts.fri_final_poly_log_degree,
        )
    };
    let expected = crate::compute_expected_commit_bus_balance_view(
        &refs,
        view,
        &proof.public_output,
        0,
        &mut seed(),
    )
    .ok_or("the COMMIT bus target must compute")?;
    let target_secs = t_verify.elapsed().as_secs_f64() - plan_secs;
    if verify
        && !crate::hash_pin::BlockVerifier::<Gl, Ext3, ()>::multi_verify_views(
            &refs,
            view,
            &mut seed(),
            &expected,
        )
    {
        return Err("production's verifier rejects the block".to_string());
    }
    let verify_secs = t_verify.elapsed().as_secs_f64();
    println!(
        "   harvest split ({}): plan {plan_secs:.2}s ({}) · bus target {target_secs:.2}s · \
         host verify {:.2}s",
        if verify { "verifying" } else { "reading" },
        if consts.is_some() {
            "ELF constants ahead"
        } else {
            "ELF constants inline"
        },
        verify_secs - plan_secs - target_secs
    );

    // ---- Phase A, as `multi_verify_views` absorbs it: the plan's preprocessed
    // roots (the AIR's, never the proof's), then the proof's main roots.
    let t_replay = std::time::Instant::now();
    let n = refs.len();
    let mut transcript = seed();
    let mut main_roots = Vec::with_capacity(n);
    for idx in 0..n {
        let v = view.get(idx);
        if let Some(p) = &plan.instance(idx).precomputed_root {
            transcript.append_bytes(p);
        }
        transcript.append_bytes(v.lde_trace_main_merkle_root());
        main_roots.push(*v.lde_trace_main_merkle_root());
    }
    let lookup: Vec<FEE> = (0..stark::lookup::LOGUP_NUM_CHALLENGES)
        .map(|_| transcript.sample_field_element())
        .collect();
    let state = transcript.state_word();

    // ---- one fork per instance, and the legs' reading of the same sub-proof.
    let per_instance: Vec<(HostTable, TableLegs)> = (0..n)
        .into_par_iter()
        .map(|idx| {
            let air = refs[idx];
            let v = view.get(idx);
            let mut fork = transcript.clone();
            if n > 1 {
                fork.append_bytes(&(idx as u64).to_le_bytes());
            }
            if let Some(root) = v.lde_trace_aux_merkle_root() {
                fork.append_bytes(root);
            }
            if let Some(c) = v.bus_table_contribution() {
                fork.append_field_element(&c);
            }
            let table = super::epoch_tests::host_table_forked(air, v, idx, n, &mut fork, &lookup);
            let legs = super::epoch_verify_tests::build_table_legs(air, v, &lookup);
            (table, legs)
        })
        .collect();
    let (tables, legs): (Vec<_>, Vec<_>) = per_instance.into_iter().unzip();
    // The proof-reading path and the plan derive one shape per instance.
    for (i, (t, l)) in tables.iter().zip(&legs).enumerate() {
        let planned = plan.instance(i);
        assert_eq!(
            t.precomputed_root, planned.precomputed_root,
            "instance {i}: the fork's and the plan's preprocessed roots are one value"
        );
        assert_eq!(
            format!("{:?}", t.shape),
            format!("{:?}", planned.challenge),
            "instance {i}: the plan's challenge shape is the one the proof is read against"
        );
        assert_eq!(
            format!("{:?}", l.verify),
            format!("{:?}", planned.verify),
            "instance {i}: the plan's legs shape is the one the proof is read against"
        );
    }
    let replay_secs = t_replay.elapsed().as_secs_f64();

    Ok((
        RealBlock {
            plan,
            public_output: proof.public_output.clone(),
            main_roots,
            tables,
            legs,
            state,
            expected_bus_balance: expected,
        },
        verify_secs,
        replay_secs,
    ))
}

/// Leaf `k` of the plan — the program the verifier derives.
pub(super) fn block_leaf_program(rb: &RealBlock, k: usize) -> LfmProgram {
    rb.plan
        .leaf_program(k)
        .unwrap_or_else(|e| panic!("leaf {k} must emit: {e}"))
}

/// Leaf `k` of another partition, carrying the COMMIT-bus target when
/// `carries` — a tree the verifier did not derive, for the negatives.
pub(super) fn block_leaf_program_over(
    rb: &RealBlock,
    partition: &BlockPartition,
    k: usize,
    carries: bool,
) -> LfmProgram {
    let mut b = production_builder();
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
    let mut arenas: Vec<Vec<LfmWord>> = vec![
        le_halves(&rb.public_output)
            .into_iter()
            .map(base_word)
            .collect(),
        super::proof_arena::commitments_to_arena(&rb.main_roots),
    ];
    for &i in partition.leaf(k) {
        let (h, leg) = (&rb.tables[i], &rb.legs[i]);
        if let Some(root) = &h.aux_root {
            arenas.push(super::proof_arena::commitments_to_arena(&[*root]));
        }
        if let Some(l) = &h.contribution {
            arenas.push(vec![ext_word(l)]);
        }
        arenas.push(super::proof_arena::commitments_to_arena(&[
            h.composition_root
        ]));
        arenas.push(h.ood_current.iter().map(ext_word).collect());
        arenas.push(h.ood_next.iter().map(ext_word).collect());
        arenas.push(h.parts.iter().map(ext_word).collect());
        arenas.push(super::proof_arena::commitments_to_arena(&h.fri_roots));
        arenas.push(h.fri_coeffs.iter().map(ext_word).collect());
        if let Some(nonce) = h.nonce {
            arenas.push(vec![base_word(FE::from(nonce))]);
        }
        arenas.push(leg.opening_arena());
        arenas.push(leg.fri_arena());
        arenas.extend(leg.caps_arena());
    }
    arenas
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

/// The plan's node over `children`, from the shapes their ARTIFACTS give — no
/// child proof — checked against the heights their proofs carry.
pub(super) fn block_node_program(
    plan: &BlockTreePlan,
    children: &[&RealChild],
    top: bool,
) -> LfmProgram {
    let words = plan.child_layout().total();
    let derived: Vec<DerivedChild> = children
        .iter()
        .map(|c| {
            let d = DerivedChild::from_artifacts(&c.artifacts, &c.opts, words)
                .unwrap_or_else(|e| panic!("a child's shapes derive from its artifacts: {e}"));
            let proved: Vec<u32> = c.tables.iter().map(|t| t.shape.log2_trace_length).collect();
            assert_eq!(
                d.log2_trace_lengths(),
                proved,
                "the artifacts' heights are the child proof's trace lengths"
            );
            d
        })
        .collect();
    plan.node_program(&derived, top)
        .unwrap_or_else(|e| panic!("a block node must emit: {e}"))
}

/// A block node's arenas: its children's, in child order.
pub(super) fn block_node_arenas(children: &[&RealChild]) -> Vec<Vec<LfmWord>> {
    children.iter().flat_map(|c| child_arena_words(c)).collect()
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

/// [`prove_program`] over the artifacts built ahead, when given. Without
/// `verify_inline` the proof is read back unverified: the caller verifies it
/// elsewhere ([`BesideVerifies`]) before anything is reported.
pub(super) fn prove_program_with(
    label: &str,
    program: &LfmProgram,
    built: Option<LfmArtifacts>,
    arenas: &[Vec<LfmWord>],
    opts: &crate::ProofOptions,
    verify_inline: bool,
) -> (RealChild, super::proof::LfmProof) {
    let t = std::time::Instant::now();
    let artifacts = built.unwrap_or_else(|| {
        super::program_census::build_artifacts_counted(program, opts, crate::hash_pin::BLOCK_HASHER)
    });
    let t_artifacts = t.elapsed().as_secs_f64();
    let t = std::time::Instant::now();
    let proved = super::proof::lfm_prove(program, &artifacts, arenas, opts)
        .unwrap_or_else(|e| panic!("{label} must prove: {e:?}"));
    let t_prove = t.elapsed().as_secs_f64();
    super::per_table_aggregator_tests::print_prove_split(label);
    let t = std::time::Instant::now();
    let (child, verified) = if verify_inline {
        let (child, t_verify) =
            super::per_table_aggregator_tests::real_child_timed(artifacts, opts.clone(), &proved);
        (child, format!("verify {t_verify:.2}"))
    } else {
        let child = super::per_table_aggregator_tests::real_child_unverified(
            artifacts,
            opts.clone(),
            &proved,
        );
        (child, "verified beside".to_string())
    };
    println!(
        "   {label} TIMING: {} instructions · build_artifacts {t_artifacts:.2}s · prove \
         {t_prove:.2}s · harvest {:.2}s ({verified})",
        program.instrs.len(),
        t.elapsed().as_secs_f64()
    );
    (child, proved)
}

/// Child proofs verified on helper threads beside the timed path (the default;
/// `NOEPOCH_CHILD_VERIFY=inline` is the old order): production's verify of each, every one
/// joined, a refusal failing the run, before anything is reported.
#[derive(Default)]
pub(super) struct BesideVerifies(std::sync::Mutex<Vec<(String, std::thread::JoinHandle<bool>)>>);

impl BesideVerifies {
    fn spawn(
        &self,
        label: &str,
        artifacts: LfmArtifacts,
        proof: super::proof::LfmProof,
        opts: crate::ProofOptions,
    ) {
        let handle = std::thread::spawn(move || {
            super::proof::verify_against_artifacts(
                &artifacts,
                &proof.proof,
                &proof.public_words,
                &opts,
            )
        });
        self.0
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push((label.to_string(), handle));
    }

    /// Joins every verify, panicking on a refusal; returns how many and the
    /// seconds the join waited.
    fn join_all(&self) -> (usize, f64) {
        let t = std::time::Instant::now();
        let jobs = std::mem::take(&mut *self.0.lock().unwrap_or_else(|e| e.into_inner()));
        let n = jobs.len();
        for (label, handle) in jobs {
            let accepted = handle
                .join()
                .unwrap_or_else(|_| panic!("{label}: the verify beside panicked"));
            assert!(
                accepted,
                "{label}: production refuses the child (verified beside)"
            );
        }
        (n, t.elapsed().as_secs_f64())
    }
}

/// The tree's child proofs verified beside the timed path, by default (FAST
/// 398: −0.62 s on level 0 + interior, the join waiting 0.00 s);
/// `NOEPOCH_CHILD_VERIFY=inline` verifies each before its parent, as before.
fn child_verify_beside_knob() -> bool {
    std::env::var("NOEPOCH_CHILD_VERIFY").map_or(true, |v| v != "inline")
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
    compose_block_tree_with(plan, leaves, opts, siblings, None, None)
}

/// How the tree's programs come ahead (`NOEPOCH_TREE_AHEAD`). Unset or `pipe`,
/// the default (FAST 451: recursion −2.56 s): the leaf programs are emitted
/// beside the base on the host and the artifacts built on the card during level
/// 0, each node program emitted as its children's artifacts exist
/// ([`super::block_tree_pipeline`]). `1` derives every program and its
/// artifacts beside the base on the host; `0` derives each inline, before it
/// proves.
#[derive(Clone, Copy)]
enum AheadMode {
    Host,
    Pipe,
}

fn tree_ahead_mode() -> Option<AheadMode> {
    match std::env::var("NOEPOCH_TREE_AHEAD").ok().as_deref() {
        Some("0") => None,
        Some("1") => Some(AheadMode::Host),
        None | Some("" | "pipe") => Some(AheadMode::Pipe),
        Some(v) => panic!("NOEPOCH_TREE_AHEAD must be 0, 1 or pipe, got `{v}`"),
    }
}

/// Threads of the pipeline builder's own pool for the node emission, by
/// default: one per level-1 node of the block's 8 leaves at fan-in 2, where it
/// was measured (FAST 455: level 0 −0.40 s, recursion −0.29 s; on the global
/// pool a leaf proof's join could steal an emission and hold the card idle). At
/// fan-in 4 level 1 has two nodes to emit.
const EMIT_POOL_THREADS: usize = 4;

/// `NOEPOCH_EMIT_POOL=<threads>`: the pipeline's builder emits the node
/// programs on a host-only pool of its own with that many threads, unset
/// meaning [`EMIT_POOL_THREADS`]; `0` emits them on the global pool the provers
/// use.
fn emit_pool_knob() -> usize {
    match std::env::var("NOEPOCH_EMIT_POOL") {
        Ok(v) if !v.is_empty() => v
            .parse::<usize>()
            .unwrap_or_else(|_| panic!("NOEPOCH_EMIT_POOL must be a thread count, got `{v}`")),
        _ => EMIT_POOL_THREADS,
    }
}

/// `NOEPOCH_TREE_NODE_EMIT`: in the pipeline mode, `early` or unset (the
/// default) emits each node's program as soon as its children's artifacts
/// exist, during the level below; `level` emits a level's programs together
/// once the whole level below is built. Per level, the median block's card sat
/// idle 8.9 s between level 0's last hold and level 1's first (BIG 481); early
/// takes the median's recursion −7.32 s (BIG 483, 2 + 2) and 1× −0.15 s (FAST
/// 668, 4 + 4), the programs and the top unchanged.
fn node_emit_early_knob() -> bool {
    match std::env::var("NOEPOCH_TREE_NODE_EMIT").ok().as_deref() {
        None | Some("" | "early") => true,
        Some("level") => false,
        Some(v) => panic!("NOEPOCH_TREE_NODE_EMIT must be early or level, got `{v}`"),
    }
}

/// `NOEPOCH_TREE_EMIT_WINDOW=<W>`: in the pipeline mode, only the first W leaf
/// programs are emitted beside the base; the builder emits the rest in leaf
/// order, at most `2 × W` ahead of its artifact builds, on its own pool. Unset
/// (the default) emits every leaf program beside the base, which holds them all
/// through the base's prove (≈ 338 MiB a leaf at the median block, BIG 480).
fn emit_window_knob() -> Option<usize> {
    std::env::var("NOEPOCH_TREE_EMIT_WINDOW")
        .ok()
        .filter(|v| !v.is_empty())
        .map(|v| {
            v.parse()
                .ok()
                .filter(|w: &usize| *w >= 1)
                .unwrap_or_else(|| {
                    panic!("NOEPOCH_TREE_EMIT_WINDOW must be a positive integer, got `{v}`")
                })
        })
}

/// How the pipeline emits the leaf programs beside the base
/// (`NOEPOCH_TREE_EMIT_LATE`).
///
/// Late: leaf 0 at the shape, the rest once the heap's live bytes have fallen
/// `margin` × their estimated bytes below the shape's (phase B frees each trace
/// as its table proves), or have stopped falling, or the base has returned
/// ([`late_trigger`]). Freed trace buffers sit in jemalloc's shared oversize
/// arena and 99.5 % of a program's bytes are allocations that size, so the
/// programs take the freed pages instead of raising the base's high-water. At
/// the median, spill off, that is base-phase VmRSS −10.37 GiB, recursion −0.31
/// s, whole −2.15 s, 98 % of the programs' bytes on reused pages (BIG 585, 2 +
/// 2); with spill on it is inert (−1.28 GiB, +0.55 s), the room phase A leaves
/// being elsewhere.
#[derive(Clone, Copy, Debug, PartialEq)]
enum LateMode {
    /// `0` or `off`: every leaf program at the shape.
    Off,
    /// Unset, empty or `auto` (the default): late at [`LATE_MARGIN`] when the
    /// programs' estimate reaches [`LATE_MIN_ESTIMATE`], at the shape otherwise.
    Auto,
    /// `<margin>`: late at that margin, whatever the estimate.
    Margin(f64),
}

fn emit_late_knob() -> LateMode {
    match std::env::var("NOEPOCH_TREE_EMIT_LATE").ok().as_deref() {
        None | Some("" | "auto") => LateMode::Auto,
        Some("0" | "off") => LateMode::Off,
        Some(v) => LateMode::Margin(
            v.parse()
                .ok()
                .filter(|m: &f64| m.is_finite() && *m > 0.0)
                .unwrap_or_else(|| {
                    panic!(
                        "NOEPOCH_TREE_EMIT_LATE must be auto, off or a positive margin, got `{v}`"
                    )
                }),
        ),
    }
}

/// The default late emission's margin over the programs' estimated bytes: the
/// pages they need plus phase B's own transients (BIG 585).
const LATE_MARGIN: f64 = 1.25;

/// The default late emission's floor: below this estimate the programs are
/// emitted at the shape. A small tree's programs barely move the peak, and its
/// short phase B may not free `margin` × their bytes before the base returns,
/// which would leave the harvest waiting on them (the record block's 7 more
/// leaves are ≈ 3.8 GiB [I]; the median's 55 are 30.6 GiB).
const LATE_MIN_ESTIMATE: usize = 8 << 30;

/// The margin a late emission waits at, or `None` to emit at the shape.
fn late_margin(mode: LateMode, estimate: usize) -> Option<f64> {
    match mode {
        LateMode::Off => None,
        LateMode::Margin(m) => Some(m),
        LateMode::Auto => (estimate >= LATE_MIN_ESTIMATE).then_some(LATE_MARGIN),
    }
}

/// The other `beside - 1` leaves' programs' bytes, from leaf 0's (`first`)
/// bytes a permutation of in-guest verification.
fn late_estimate(plan: &BlockTreePlan, first: &LfmProgram, beside: usize) -> usize {
    let costs = plan.costs();
    let perms =
        |k: usize| -> usize { plan.partition().leaves()[k].iter().map(|&i| costs[i]).sum() };
    let per_perm = ProgramBytes::of(first).total() as f64 / perms(0).max(1) as f64;
    (per_perm * (1..beside).map(perms).sum::<usize>() as f64) as usize
}

/// The default emits late only above its floor, a margin forces it, `off` never.
#[test]
fn the_default_late_emission_needs_its_floor() {
    const G: usize = 1 << 30;
    assert_eq!(late_margin(LateMode::Auto, 30 * G), Some(LATE_MARGIN));
    assert_eq!(
        late_margin(LateMode::Auto, LATE_MIN_ESTIMATE),
        Some(LATE_MARGIN)
    );
    assert_eq!(late_margin(LateMode::Auto, 4 * G), None);
    assert_eq!(late_margin(LateMode::Margin(2.0), G), Some(2.0));
    assert_eq!(late_margin(LateMode::Off, 30 * G), None);
}

/// [`late_trigger`]'s settle rule: the heap's live bytes have made no new low
/// by [`LATE_LOW_STEP`] for this long (spill on: phase B holds only its
/// read-back window, so the live bytes stop falling long before the programs'
/// bytes are freed).
const LATE_SETTLE_SECS: f64 = 30.0;

/// The step a new low of the heap's live bytes must beat the last one by.
const LATE_LOW_STEP: usize = 1 << 30;

/// When the late emission (`NOEPOCH_TREE_EMIT_LATE`) starts: once the heap's
/// live bytes are `need` below their value at the shape (`fell`), or have made
/// no new low for [`LATE_SETTLE_SECS`] (`settled`), or the base has returned
/// (`base returned`); `None` keeps waiting.
fn late_trigger(
    live_at_shape: usize,
    live: usize,
    need: usize,
    since_low_secs: f64,
    base_done: bool,
) -> Option<&'static str> {
    if live_at_shape.saturating_sub(live) >= need {
        Some("fell")
    } else if base_done {
        Some("base returned")
    } else if since_low_secs >= LATE_SETTLE_SECS {
        Some("settled")
    } else {
        None
    }
}

/// The late emission's wait and what the heap did around it.
struct LateWait {
    margin: f64,
    trigger: &'static str,
    waited: f64,
    /// Seconds since the base started, when the wait ended.
    at: f64,
    estimate: usize,
    need: usize,
    live_at_shape: usize,
    live_at_start: usize,
    resident_at_start: usize,
}

/// Waits for [`late_trigger`], polling the heap every 200 ms, for a fall of
/// `margin` × `estimate` ([`late_estimate`]).
fn wait_for_late_emission(
    estimate: usize,
    margin: f64,
    live_at_shape: usize,
    base_done: &std::sync::atomic::AtomicBool,
    t_base0: std::time::Instant,
) -> LateWait {
    use std::time::Instant;
    let need = (margin * estimate as f64) as usize;
    let t = Instant::now();
    let (mut low, mut low_at) = (live_at_shape, Instant::now());
    let trigger = loop {
        let live = heap_allocated();
        if live + LATE_LOW_STEP <= low {
            (low, low_at) = (live, Instant::now());
        }
        let done = base_done.load(std::sync::atomic::Ordering::SeqCst);
        if let Some(why) = late_trigger(
            live_at_shape,
            live,
            need,
            low_at.elapsed().as_secs_f64(),
            done,
        ) {
            break why;
        }
        std::thread::sleep(std::time::Duration::from_millis(200));
    };
    LateWait {
        margin,
        trigger,
        waited: t.elapsed().as_secs_f64(),
        at: t_base0.elapsed().as_secs_f64(),
        estimate,
        need,
        live_at_shape,
        live_at_start: heap_allocated(),
        resident_at_start: heap_resident(),
    }
}

/// `NOEPOCH_TREE_EMIT_LATE`'s trigger: a fall of `need` below the shape's live
/// bytes starts the emission; otherwise the base's return or a heap that has
/// settled does; nothing else does.
#[test]
fn the_late_emission_waits_for_its_room() {
    const G: usize = 1 << 30;
    // The median, spill off: 68 GiB live at the shape, a fall of 30 GiB needed.
    assert_eq!(late_trigger(68 * G, 60 * G, 30 * G, 0.5, false), None);
    assert_eq!(
        late_trigger(68 * G, 38 * G, 30 * G, 0.5, false),
        Some("fell")
    );
    // Phase B's transients lifting the live bytes over the shape's are no fall.
    assert_eq!(late_trigger(68 * G, 70 * G, 30 * G, 0.5, false), None);
    let settled = LATE_SETTLE_SECS;
    assert_eq!(
        late_trigger(68 * G, 60 * G, 30 * G, settled - 0.1, false),
        None
    );
    assert_eq!(
        late_trigger(68 * G, 60 * G, 30 * G, settled, false),
        Some("settled")
    );
    assert_eq!(
        late_trigger(68 * G, 60 * G, 30 * G, 0.5, true),
        Some("base returned")
    );
    // The fall is reported over the other two.
    assert_eq!(
        late_trigger(68 * G, 38 * G, 30 * G, settled, true),
        Some("fell")
    );
}

/// [`compose_block_tree`], each node proved from the program and artifacts
/// derived ahead when `ahead` holds the node levels, and each node below the top
/// verified on `beside` when given (the top is verified inline).
pub(super) fn compose_block_tree_with(
    plan: &BlockTreePlan,
    leaves: Vec<RealChild>,
    opts: &crate::ProofOptions,
    siblings: usize,
    ahead: Option<&Pipe>,
    beside: Option<&BesideVerifies>,
) -> (RealChild, super::proof::LfmProof, Vec<f64>) {
    let fan_in = super::block_plan::block_fan_in();
    let shape = plan.levels();
    let mut level = leaves;
    let mut top_proof = None;
    let mut walls = Vec::with_capacity(shape.len());
    for (lv, arities) in shape.iter().enumerate() {
        let top = lv + 1 == shape.len();
        let t = std::time::Instant::now();
        let mut starts = Vec::with_capacity(arities.arities.len());
        let mut at = 0usize;
        for a in &arities.arities {
            starts.push(at..at + a);
            at += a;
        }
        assert_eq!(at, level.len(), "the level's arities cover its children");
        let next = super::per_table_aggregator_tests::in_index_order(starts.len(), siblings, |j| {
            let kids: Vec<&RealChild> = level[starts[j].clone()].iter().collect();
            let label = format!(
                "BLOCK L{}N{j} ({} child{}){}",
                lv + 1,
                kids.len(),
                if kids.len() == 1 { "" } else { "ren" },
                if top { " TOP" } else { "" }
            );
            let te = std::time::Instant::now();
            let (program, built) = match ahead {
                Some(p) => {
                    let (program, artifacts) = p.take_node(lv, j, &label);
                    (program, Some(artifacts))
                }
                None => (block_node_program(plan, &kids, top), None),
            };
            println!(
                "   {label}: {} in {:.2}s",
                if built.is_some() {
                    "program ahead"
                } else {
                    "emitted"
                },
                te.elapsed().as_secs_f64()
            );
            super::per_table_aggregator_tests::census_and_panel(&program, &label, fan_in);
            let inline = top || beside.is_none();
            let (child, proof) = prove_program_with(
                &label,
                &program,
                built,
                &block_node_arenas(&kids),
                opts,
                inline,
            );
            if let (false, Some(b)) = (inline, beside) {
                b.spawn(&label, child.artifacts.clone(), proof, opts.clone());
                return (child, None);
            }
            (child, top.then_some(proof))
        });
        let (next, mut proofs): (Vec<RealChild>, Vec<Option<super::proof::LfmProof>>) =
            next.into_iter().unzip();
        if top {
            top_proof = proofs.pop().flatten();
        }
        let wall = t.elapsed().as_secs_f64();
        println!(
            "   BLOCK LEVEL {}: {} node(s) in {wall:.2}s{}",
            lv + 1,
            next.len(),
            if top { " (the top)" } else { "" }
        );
        walls.push(wall);
        level = next;
    }
    assert_eq!(level.len(), 1, "the tree closes to one node");
    (
        level.pop().expect("one"),
        top_proof.expect("the top level keeps its proof"),
        walls,
    )
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
/// index at least one bit) with two queries, so the leaves stay small.
fn fixture_block_options() -> crate::ProofOptions {
    super::epoch_tests::from_proof_gate_options()
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
        let exec = execute(&program, &arenas, &crate::hash_pin::BLOCK_HASHER)
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
                execute(&program, &bad, &crate::hash_pin::BLOCK_HASHER).is_err(),
                "leaf {k}: a tampered L of instance {i} must not execute"
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
        execute(&program, &over, &crate::hash_pin::BLOCK_HASHER).is_err(),
        "an output half at or over 2^32 must not execute"
    );
    let live = rb.public_output.len() % 4;
    if live != 0 {
        let last = arenas[0].len() - 1;
        let v: u64 = arenas[0][last][0].canonical();
        let mut padded = arenas.clone();
        padded[0][last] = base_word(FE::from(v | (1u64 << (8 * live))));
        assert!(
            execute(&program, &padded, &crate::hash_pin::BLOCK_HASHER).is_err(),
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
            &crate::hash_pin::BLOCK_HASHER,
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
                &crate::hash_pin::BLOCK_HASHER,
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
                    crate::hash_pin::BLOCK_HASHER,
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

/// Threads for the ELF constants beside the base by default: FAST 393 measured
/// the harvest −1.87 s and the whole −1.80 s at four, the base +0.02 s.
const ELF_BESIDE_THREADS: usize = 4;

/// `NOEPOCH_ELF_BESIDE`: threads for the ELF constants beside the base, unset
/// meaning [`ELF_BESIDE_THREADS`]; `0` computes them inline in the harvest.
fn elf_beside_knob() -> Option<usize> {
    match std::env::var("NOEPOCH_ELF_BESIDE") {
        Ok(v) if !v.is_empty() => Some(
            v.parse::<usize>()
                .unwrap_or_else(|_| panic!("NOEPOCH_ELF_BESIDE must be a thread count, got `{v}`")),
        ),
        _ => Some(ELF_BESIDE_THREADS),
    }
    .filter(|&t| t > 0)
}

/// `NOEPOCH_LEAVES`: the leaf count; unset is the rule's `⌈Σ cost / 279 000⌉`.
fn leaves_knob() -> Option<usize> {
    std::env::var("NOEPOCH_LEAVES")
        .ok()
        .filter(|v| !v.is_empty())
        .map(|v| {
            v.parse()
                .ok()
                .filter(|k: &usize| *k >= 1)
                .unwrap_or_else(|| panic!("NOEPOCH_LEAVES must be a positive integer, got `{v}`"))
        })
}

/// ★★★ THE WHOLE NO-EPOCH BLOCK, base to top node: `block::prove_block`, the
/// harvest, the leaves (level 0), the interior and the top — the block's
/// recursion, and the number the epoch/no-epoch decision is taken on.
///
/// Its `★★★ WHOLE RUN` line is measured as the epoch tree's
/// ([`super::per_table_aggregator_tests::the_production_tree_composes_to_a_root`]):
/// from before the base to the last proof, the harness's verifies inside. That
/// tree's number stops at its interior (its block-artifact root is a separate
/// arm); this one's top node is its last proof, and it closes the block's bus.
#[test]
#[ignore = "box tier, production scale: the whole no-epoch block and its recursion (NOEPOCH_ELF, NOEPOCH_INPUT, --features cuda)"]
fn the_block_tree_composes_to_a_top_node() {
    use super::per_table_aggregator_tests::{
        HostSampler, cgroup_limit_gib, in_index_order, tree_siblings, tree_siblings_l0,
    };
    use std::time::Instant;

    // A refused harvest names its table and its check: the verifier's `error!` lines.
    let _ = env_logger::builder().is_test(true).try_init();

    if !cfg!(feature = "cuda") {
        panic!("the production block tree requires `--features cuda`");
    }
    let read = |var: &str| -> Vec<u8> {
        let p = std::env::var(var).unwrap_or_else(|_| panic!("{var} must name a file"));
        std::fs::read(&p).unwrap_or_else(|e| panic!("{var} {p}: {e}"))
    };
    let elf_bytes = read("NOEPOCH_ELF");
    let input = read("NOEPOCH_INPUT");
    let inner = super::proof::block_base_options();
    let wrap_opts = super::proof::aggregation_wrap_options();
    let ceiling = cgroup_limit_gib();
    let pct = |g: f64| match &ceiling {
        Ok(c) => format!(" ({:.1}% of {c:.2})", 100.0 * g / c),
        Err(_) => String::new(),
    };
    println!(
        "★★★ NO-EPOCH BLOCK TREE (base + leaves + interior + top)\n   \
         {} input bytes · inner blowup {} / {} q · wrap blowup {} / {} q · cgroup {}",
        input.len(),
        inner.blowup_factor,
        inner.fri_number_of_queries,
        wrap_opts.blowup_factor,
        wrap_opts.fri_number_of_queries,
        match &ceiling {
            Ok(g) => format!("{g:.2} GiB"),
            Err(why) => format!("UNKNOWN — {why}"),
        },
    );

    let whole = HostSampler::start();
    let t_all = Instant::now();

    // ---- the ELF constants beside the base (`NOEPOCH_ELF_BESIDE=<threads>`, four
    // by default): the plan's ELF-only input (DECODE's root, recomputed on the
    // host) computed on a pool of its own while the base proves, joined by the
    // harvest. `NOEPOCH_ELF_BESIDE=0`: the harvest computes it inline.
    let elf_beside = elf_beside_knob();
    // `NOEPOCH_TREE_AHEAD` (the pipeline by default): the same pool then emits the
    // leaf programs (with `1`, every tree program and its artifacts, on the host)
    // from the shape the base hands out before its prove, and the levels prove
    // from them.
    let tree_ahead = elf_beside.and(tree_ahead_mode());
    let emit_threads = emit_pool_knob();
    let emit_window = emit_window_knob();
    let node_emit_early = node_emit_early_knob();
    let emit_late = emit_late_knob();
    let base_done = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let (shape_tx, shape_rx) = std::sync::mpsc::channel::<BlockShape>();
    // The thread hands its results back on `ready` and, in the pipeline mode,
    // stays on as the tree's builder once `go` says the base is done.
    let (ready_tx, ready_rx) = std::sync::mpsc::channel();
    let (go_tx, go_rx) = std::sync::mpsc::channel::<()>();
    let t_base0 = Instant::now();
    let consts_beside = elf_beside.map(|threads| {
        let (elf, opts, wrap) = (elf_bytes.clone(), inner.clone(), wrap_opts.clone());
        let base_done = base_done.clone();
        std::thread::spawn(move || {
            let t = Instant::now();
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .thread_name(|i| format!("elf-beside-{i}"))
                .start_handler(|_| super::commit::mark_thread_host_only())
                .build()
                .expect("the ELF constants pool builds");
            let consts = pool.install(|| super::block_plan::ElfConstants::compute(&elf, &opts));
            let secs = t.elapsed().as_secs_f64();
            let mut job = None;
            let mut late_line = None;
            let ahead = match (&consts, tree_ahead) {
                (Ok(c), Some(mode)) => shape_rx.recv().ok().map(|shape| {
                    let live_at_shape = (emit_late != LateMode::Off).then(heap_allocated);
                    let t = Instant::now();
                    let derived = pool.install(|| -> Result<_, String> {
                        let plan = BlockTreePlan::derive_with(&elf, &opts, &shape, c)?;
                        Ok(match mode {
                            AheadMode::Host => {
                                let (tree, phases) = plan.derive_tree(&wrap, &|program| {
                                    super::registry::build_artifacts_with_hasher(
                                        program,
                                        &wrap,
                                        crate::hash_pin::BLOCK_HASHER,
                                    )
                                })?;
                                (std::sync::Arc::new(Pipe::filled(tree)), phases)
                            }
                            AheadMode::Pipe => {
                                use rayon::prelude::*;
                                let n = plan.partition().num_leaves();
                                let beside = emit_window.map_or(n, |w| w.min(n));
                                // `NOEPOCH_TREE_EMIT_LATE`: leaf 0 now, sizing the
                                // wait; the rest once phase B has freed their room.
                                let mut leaves = Vec::with_capacity(beside);
                                let late = match live_at_shape {
                                    Some(live) if beside > 1 => {
                                        leaves.push(plan.leaf_program(0)?);
                                        let estimate = late_estimate(&plan, &leaves[0], beside);
                                        match late_margin(emit_late, estimate) {
                                            Some(margin) => Some(wait_for_late_emission(
                                                estimate, margin, live, &base_done, t_base0,
                                            )),
                                            None => {
                                                late_line = Some(format!(
                                                    "   TREE LATE: auto, the other {} leaves' programs \
                                                     estimated {:.2} GiB, under the {:.0} GiB floor: \
                                                     emitted at the shape",
                                                    beside - 1,
                                                    gib(estimate),
                                                    gib(LATE_MIN_ESTIMATE)
                                                ));
                                                None
                                            }
                                        }
                                    }
                                    _ => None,
                                };
                                let te = Instant::now();
                                let first = leaves.len();
                                leaves.extend(
                                    (first..beside)
                                        .into_par_iter()
                                        .map(|k| plan.leaf_program(k))
                                        .collect::<Result<Vec<_>, String>>()?,
                                );
                                if let Some(w) = late {
                                    let mut held = ProgramBytes::default();
                                    for p in &leaves {
                                        held.add(&ProgramBytes::of(p));
                                    }
                                    let resident = heap_resident();
                                    late_line = Some(format!(
                                        "   TREE LATE: leaf 0 at the shape, {} more after {:.2}s ({}) \
                                         at {:.2}s of the base, emitted in {:.2}s, done at {:.2}s · heap \
                                         live {:.2} GiB at the shape → {:.2} at the start (need a fall of \
                                         {:.2} = {} × {:.2} estimated) → {:.2} at the end · resident \
                                         {:.2} → {:.2} GiB over the emission (+{:.2}) · programs held \
                                         {:.2} GiB, ≥ oversize {:.1} %",
                                        beside - 1,
                                        w.waited,
                                        w.trigger,
                                        w.at,
                                        te.elapsed().as_secs_f64(),
                                        t_base0.elapsed().as_secs_f64(),
                                        gib(w.live_at_shape),
                                        gib(w.live_at_start),
                                        gib(w.need),
                                        w.margin,
                                        gib(w.estimate),
                                        gib(heap_allocated()),
                                        gib(w.resident_at_start),
                                        gib(resident),
                                        gib(resident.saturating_sub(w.resident_at_start)),
                                        gib(held.total()),
                                        100.0 * held.large as f64 / held.total().max(1) as f64,
                                    ));
                                }
                                let pipe = std::sync::Arc::new(Pipe::new(n, &plan.levels()));
                                let emitted = super::block_plan::PhaseTimes {
                                    programs: leaves.len(),
                                    wall: te.elapsed().as_secs_f64(),
                                    emit: te.elapsed().as_secs_f64(),
                                    build: 0.0,
                                };
                                job = Some((pipe.clone(), plan, leaves));
                                (pipe, vec![emitted])
                            }
                        })
                    });
                    (
                        shape,
                        derived,
                        t.elapsed().as_secs_f64(),
                        t_base0.elapsed().as_secs_f64(),
                    )
                }),
                _ => None,
            };
            let _ = ready_tx.send((consts, secs, ahead, late_line));
            let (pipe, plan, leaves) = job?;
            go_rx.recv().ok()?;
            Some(pipe.run_builder(
                &plan,
                leaves,
                &wrap,
                emit_threads,
                emit_window.unwrap_or(0),
                node_emit_early,
            ))
        })
    });

    // ---- the base.
    let base_sampler = HostSampler::start();
    let t = Instant::now();
    let mut shape_at = None;
    let (proof, times) = crate::block::prove_block_observed(&elf_bytes, &input, &inner, &mut |s| {
        shape_at = Some(t.elapsed().as_secs_f64());
        let _ = shape_tx.send(s.clone());
    })
    .expect("the block must prove");
    base_done.store(true, std::sync::atomic::Ordering::SeqCst);
    drop(shape_tx);
    let base = t.elapsed().as_secs_f64();
    let (base_peak, _) = base_sampler.stop();
    println!(
        "   base: {} sub-proofs in {base:.2}s (execute {:.2} · build {:.2} · setup {:.2} · \
         prove {:.2}) · host peak {base_peak:.3} GiB{}",
        proof.proof.proofs.len(),
        times.execute,
        times.build,
        times.setup,
        times.prove,
        pct(base_peak)
    );

    // ---- the harvest. Production's verify of the base is a harness assert, not
    // work a driver does, so by default it runs on a helper thread beside level 0
    // and is joined before anything is reported: a refused block still fails the
    // run, it only stops delaying the leaves (the epoch tree's per-epoch verifies
    // likewise run beside its other wraps). `NOEPOCH_HARVEST_VERIFY=inline`
    // verifies first, as before.
    let inline_verify = std::env::var("NOEPOCH_HARVEST_VERIFY").is_ok_and(|v| v == "inline");
    let t = Instant::now();
    let mut pipe = None;
    let consts = consts_beside.as_ref().map(|_| {
        let tj = Instant::now();
        let (consts, secs, ahead, late_line) =
            ready_rx.recv().expect("the ELF constants thread stopped");
        println!(
            "   ELF constants beside the base: {secs:.2}s on {} thread(s), joined after the base, \
             waited {:.2}s (counted in the harvest)",
            elf_beside.unwrap_or(0),
            tj.elapsed().as_secs_f64()
        );
        if let Some((shape, derived, secs, done_at)) = ahead {
            let (filled, phases) = derived.expect("the tree derives ahead");
            assert_eq!(
                format!("{shape:?}"),
                format!("{:?}", BlockShape::of_proof(&proof)),
                "the shape handed out before the prove is the proof's"
            );
            let split: Vec<String> = phases
                .iter()
                .map(|p| format!("{} in {:.2} (emit Σ {:.2}, build Σ {:.2})", p.programs, p.wall, p.emit, p.build))
                .collect();
            println!(
                "   TREE AHEAD: {} programs derived beside the base in {secs:.2}s on the host (shape at \
                 {:.2}s, done at {done_at:.2}s of a {base:.2}s base) · {}",
                phases.iter().map(|p| p.programs).sum::<usize>(),
                shape_at.unwrap_or(f64::NAN),
                split.join(" · ")
            );
            pipe = Some(filled);
        }
        if let Some(line) = late_line {
            println!("{line}");
        }
        std::sync::Arc::new(consts.expect("the ELF constants compute"))
    });
    assert!(
        tree_ahead.is_none() || pipe.is_some(),
        "NOEPOCH_TREE_AHEAD derived no tree"
    );
    let shape = BlockShape::of_proof(&proof);
    let (mut rb, verify, replay, beside) = if inline_verify {
        let (rb, verify, replay) =
            harvest_block_over(&inner, &elf_bytes, &proof, true, consts.as_deref())
                .expect("harvest");
        drop(proof);
        (rb, Some(verify), replay, None)
    } else {
        let proof = std::sync::Arc::new(proof);
        let beside = {
            let (proof, opts, elf) = (proof.clone(), inner.clone(), elf_bytes.clone());
            let consts = consts.clone();
            std::thread::spawn(move || {
                harvest_block_over(&opts, &elf, &proof, true, consts.as_deref())
                    .map(|(rb, verify, _)| (rb.state, verify))
            })
        };
        let (rb, _, replay) =
            harvest_block_over(&inner, &elf_bytes, &proof, false, consts.as_deref())
                .expect("harvest");
        drop(proof);
        (rb, None, replay, Some(beside))
    };
    let mut harvest = t.elapsed().as_secs_f64();
    println!(
        "   harvest: {harvest:.2}s (production verify {}, harness-only · replay {replay:.2}s) · \
         {} instances · public output {} bytes",
        match verify {
            Some(v) => format!("{v:.2}s inline"),
            None => "beside level 0".to_string(),
        },
        rb.num_instances(),
        rb.public_output.len()
    );

    // ---- the partition: the plan's — the rule over the closed-form costs, a
    // pure function of the block's shape. `NOEPOCH_LEAVES` forces another leaf
    // count: another tree, which the final check then derives over.
    let forced = leaves_knob();
    if let Some(k) = forced {
        let p = rb.partition_over(k);
        rb.plan = rb.plan.with_partition(p);
    }
    let partition = rb.plan.partition().clone();
    let names = rb.names();
    let d12_block = names.len() == 137 && names[42] == "MEMW[0]" && names[100] == "MEMW_R[0]";
    println!(
        "   BLOCK PARTITION SOURCE: {} · D-NOEPOCH §12.2's lists (legmodel.py costs): {}",
        if forced.is_some() {
            "NOEPOCH_LEAVES (forced: NOT the plan's tree)"
        } else {
            "the plan (the rule)"
        },
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
    );
    let costs = rb.plan.costs();
    let k = partition.num_leaves();
    for (j, leaf) in partition.leaves().iter().enumerate() {
        let perms: usize = leaf.iter().map(|&i| costs[i]).sum();
        println!(
            "   BLOCK LEAF {j}: {} instances · {perms} perms (closed form) · idx {leaf:?}",
            leaf.len()
        );
    }
    println!(
        "   BLOCK PARTITION: {k} leaves, Σ {} perms, heaviest {} (cap {}, cost model v{})",
        costs.iter().sum::<usize>(),
        partition
            .leaves()
            .iter()
            .map(|l| l.iter().map(|&i| costs[i]).sum::<usize>())
            .max()
            .unwrap_or(0),
        super::block_plan::LEAF_PERMS_CAP,
        rb.plan.cost_model()
    );
    // Which leaves verify the chunked accelerators' instances (the rule seeds
    // each table's first instance and fills its later chunks by load).
    let leaves_of = |kind: &str| -> Vec<usize> {
        let prefix = format!("{kind}[");
        (0..k)
            .filter(|&j| {
                partition
                    .leaf(j)
                    .iter()
                    .any(|&i| names[i].starts_with(&prefix))
            })
            .collect()
    };
    println!(
        "   BLOCK PARTITION CHUNKED: KECCAK on leaves {:?} · ECSM {:?} · ECDAS {:?} · KECCAK_RND {:?}",
        leaves_of("KECCAK"),
        leaves_of("ECSM"),
        leaves_of("ECDAS"),
        leaves_of("KECCAK_RND")
    );

    // ---- level 0: the leaves.
    let beside_verifies = child_verify_beside_knob().then(BesideVerifies::default);
    let l0 = tree_siblings_l0().min(k);
    println!("   ★ LEVEL-0 CONCURRENCY: {l0} leaf proof(s) at once (LFM_TREE_SIBLINGS_L0)");
    super::device_permit::arm(l0);
    // The pipeline's builder starts on the card now that the base is done.
    let _ = go_tx.send(());
    let l0_sampler = HostSampler::start();
    let t = Instant::now();
    let leaves = in_index_order(k, l0, |j| {
        let label = format!("BLOCK L0 leaf {j}");
        let te = Instant::now();
        let (program, built) = match &pipe {
            Some(p) => {
                let (program, artifacts) = p.take_leaf(j, &label);
                (program, Some(artifacts))
            }
            None => (block_leaf_program(&rb, j), None),
        };
        let arenas = block_leaf_arenas(&rb, &partition, j);
        println!(
            "   {label}: {} + arenas in {:.2}s",
            if built.is_some() {
                "program ahead"
            } else {
                "emitted"
            },
            te.elapsed().as_secs_f64()
        );
        super::per_table_aggregator_tests::census_and_panel(
            &program,
            &label,
            super::block_plan::block_fan_in(),
        );
        let (child, proof) = prove_program_with(
            &label,
            &program,
            built,
            &arenas,
            &wrap_opts,
            beside_verifies.is_none(),
        );
        if let Some(b) = &beside_verifies {
            b.spawn(&label, child.artifacts.clone(), proof, wrap_opts.clone());
        }
        let words: Vec<LfmWord> = child.public_words.iter().map(|(_, w)| *w).collect();
        assert_eq!(
            words,
            rb.expected_leaf_publics(&partition, j, j == rb.plan.carrier()),
            "{label} publishes the host's id, state, output and share"
        );
        child
    });
    let level0 = t.elapsed().as_secs_f64();
    let (l0_peak, _) = l0_sampler.stop();
    println!(
        "   BLOCK LEVEL 0: {k} leaves in {level0:.2}s · host peak {l0_peak:.3} GiB{}",
        pct(l0_peak)
    );

    // ---- the interior and the top.
    let siblings = tree_siblings();
    println!("   ★ SIBLING CONCURRENCY: {siblings} node proof(s) at once (LFM_TREE_SIBLINGS)");
    super::device_permit::arm(siblings);
    let t = Instant::now();
    let (top, top_proof, walls) = compose_block_tree_with(
        &rb.plan,
        leaves,
        &wrap_opts,
        siblings,
        pipe.as_deref(),
        beside_verifies.as_ref(),
    );
    // The pipeline's builder is done by now (every node took its program).
    if let Some(handle) = consts_beside {
        let built = handle.join().expect("the ELF constants thread panicked");
        if let Some(times) = built {
            let times = times.expect("the tree's builder");
            println!(
                "   TREE PIPE: leaf artifacts built by {:.2}s of level 0, node levels by {:?} s · node \
                 emission Σ {:.2}s ({}) · artifact builds Σ {:.2}s (holding the card permit) · leaf \
                 programs {}",
                times.leaves,
                times.levels,
                times.emit,
                if emit_threads > 0 {
                    format!("own pool of {emit_threads}")
                } else {
                    "global pool".to_string()
                },
                times.build,
                match emit_window {
                    Some(w) =>
                        format!("emitted in a window of {w} (the first {w} beside the base)"),
                    None => "all emitted beside the base".to_string(),
                }
            );
            println!(
                "   TREE PIPE node emission: {}",
                if node_emit_early {
                    "early (each node once its children's artifacts exist)"
                } else {
                    "per level (each level once the level below is built)"
                }
            );
        }
    }
    super::device_permit::arm(1);
    // Every child verified beside is joined here, inside the interior's time: a
    // refusal fails the run before anything is reported.
    if let Some(b) = &beside_verifies {
        let (n, waited) = b.join_all();
        println!(
            "   CHILD VERIFIES beside the timed path: {n} accepted (production's verify), the join \
             waited {waited:.2}s (counted in the interior)"
        );
    }
    let interior = t.elapsed().as_secs_f64();
    // The verify beside level 0 must have accepted the block, over the same
    // reconstruction the leaves read (one state word), before anything counts.
    // What the join waits is on the critical path, so it is the harvest's.
    if let Some(beside) = beside {
        let t = Instant::now();
        let (state, verify) = beside
            .join()
            .expect("the harvest verify beside level 0 panicked")
            .expect("harvest (verified beside level 0)");
        assert_eq!(
            state, rb.state,
            "the verified harvest and the leaves' harvest read one transcript"
        );
        let waited = t.elapsed().as_secs_f64();
        harvest += waited;
        println!(
            "   harvest verify beside level 0: production verify {verify:.2}s, harness-only · \
             joined after the top, waited {waited:.2}s (counted in the harvest)"
        );
    }
    assert_top_claims_the_block(&top, &rb);
    let t = Instant::now();
    assert!(
        verify_block_top(&top.artifacts, &top_proof, &wrap_opts),
        "the top proof must verify against the top program the harness emitted"
    );
    println!(
        "   BLOCK FINAL CHECK (harness): the top verifies against its emitted program \
         ({:.2}s, program id {})",
        t.elapsed().as_secs_f64(),
        top.artifacts
            .program_id
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect::<String>()
    );

    let total = t_all.elapsed().as_secs_f64();
    let (peak, at) = whole.stop();
    println!(
        "★★★ NO-EPOCH BLOCK: base {base:.2}s · harvest {harvest:.2}s · level 0 {level0:.2}s \
         ({k} leaves) · interior {interior:.2}s (levels {}) · recursion {:.2}s · whole {total:.2}s",
        walls
            .iter()
            .map(|w| format!("{w:.2}"))
            .collect::<Vec<_>>()
            .join(" + "),
        harvest + level0 + interior
    );
    println!("★★★ WHOLE RUN: host peak {peak:.3} GiB at t={at:.1}, {total:.1}s total");

    // ---- the block VERIFIER, outside the whole run (a consumer's work, not the
    // prover's): `verify_block_tree` under the block presets derives the plan and
    // the top program from the ELF and the claimed shape with no proof read,
    // verifies the top proof against that program, and checks its words against
    // the ELF's id and the output. A forced partition is another tree: then the
    // harness's own plan derives its top instead.
    let t = Instant::now();
    let mut split = None;
    let derived = match forced {
        None => {
            let (id, times) = super::block_plan::verify_block_tree_timed(
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
    // Warm: the ELF constants a consumer caches per ELF (the harness's, computed
    // beside the base under the same options).
    if let (None, Some(consts)) = (forced, consts.as_deref()) {
        let t = Instant::now();
        let (id, warm) = super::block_plan::verify_block_tree_timed(
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
