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

use super::block_leaf::{
    BlockInstance, BlockLeafInputs, BlockPartition, emit_block_leaf, partition_by_rule,
};
use super::block_node::{
    BlockBindings, BlockLayout, BlockNodeInputs, bind_and_publish, emit_block_node,
};
use super::block_replay::{BlockStatementShape, replay_block_front};
use super::builder::LfmBuilder;
use super::compiler::{LfmProgram, compile};
use super::edsl::WrapHash;
use super::epoch_tests::HostTable;
use super::epoch_verify_tests::TableLegs;
use super::executor::execute;
use super::per_table_aggregator::{LegCells, hint_public_words, publics_arena};
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
fn le_halves(bytes: &[u8]) -> Vec<FE> {
    bytes
        .chunks(4)
        .map(|c| {
            let mut le = [0u8; 4];
            le[..c.len()].copy_from_slice(c);
            FE::from(u64::from(u32::from_le_bytes(le)))
        })
        .collect()
}

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

/// The rule's seeds and its min-load fill, on a toy instance list.
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
    let p = partition_by_rule(&names, &costs, 4);
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
    let one = partition_by_rule(&names, &costs, 1);
    assert_eq!(one.leaves(), &[(0..names.len()).collect::<Vec<_>>()]);
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
    bind_and_publish(&mut b, &legs, layout, top, checks);
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

/// One block proof, production-accepted, read for leaf emission: the statement,
/// every instance's roots, the shared challenges and state, and per instance the
/// fork's shape and data ([`HostTable`]) and the legs' shapes and openings
/// ([`TableLegs`]).
pub(super) struct RealBlock {
    pub(super) statement: BlockStatementShape,
    pub(super) elf_digest: [u8; 32],
    pub(super) public_output: Vec<u8>,
    /// AIR names, in AIR order.
    pub(super) names: Vec<String>,
    /// Per instance, the preprocessed root at its leaf layout — the AIR's, never
    /// the proof's.
    pub(super) precomputed_roots: Vec<Option<Commitment>>,
    pub(super) main_roots: Vec<Commitment>,
    pub(super) tables: Vec<HostTable>,
    pub(super) legs: Vec<TableLegs>,
    /// The host transcript's state after `z, α`.
    pub(super) state: LfmWord,
    /// The COMMIT-bus target production computed.
    pub(super) expected_bus_balance: FEE,
    /// The ELF-derived inputs the leaves' attestation id covers: the ELF digest,
    /// the entry point, DECODE's root and the ELF data pages' roots.
    pub(super) attested: super::programs::AttestedInputs,
}

/// Permutations of transcript and grinding per instance fork, on top of its
/// legs' closed form: D-NOEPOCH §12.2's 694 (690 transcript + 4 PoW, the mean
/// over 337 measured sub-proofs).
const FORK_PERMS: usize = 694;

/// The leaf-load cap: today's wrap LFM_HASH shape, 2^18 + 2^15 rows, less the
/// headroom wrap 2 runs at (D-NOEPOCH §12.2).
const LEAF_PERMS_CAP: usize = 279_000;

impl RealBlock {
    pub(super) fn num_instances(&self) -> usize {
        self.names.len()
    }

    /// Per instance, its in-guest verification cost: the legs' closed form plus
    /// the fork's transcript and grinding.
    pub(super) fn leg_costs(&self) -> Vec<usize> {
        self.legs
            .iter()
            .map(|l| {
                super::epoch_verify::table_permutations_for(&l.verify, WrapHash::production())
                    + FORK_PERMS
            })
            .collect()
    }

    /// The §12.2 partition, over `num_leaves` leaves or `⌈Σ cost / cap⌉`.
    pub(super) fn partition(&self, num_leaves: Option<usize>) -> BlockPartition {
        let costs = self.leg_costs();
        let k = num_leaves.unwrap_or_else(|| costs.iter().sum::<usize>().div_ceil(LEAF_PERMS_CAP));
        let names: Vec<&str> = self.names.iter().map(String::as_str).collect();
        partition_by_rule(&names, &costs, k.max(1))
    }

    /// What leaf `k` must publish, from the host's own values: the id, the
    /// state, the output halves, and its share of the bus.
    pub(super) fn expected_leaf_publics(
        &self,
        partition: &BlockPartition,
        k: usize,
        carrier: usize,
    ) -> Vec<LfmWord> {
        let mut words = super::programs::program_id_words(&self.attested.program_id()).to_vec();
        words.push(self.state);
        words.extend(le_halves(&self.public_output).into_iter().map(base_word));
        let mut share = partition
            .leaf(k)
            .iter()
            .filter_map(|&i| self.tables[i].contribution)
            .fold(FEE::zero(), |acc, l| acc + l);
        if k == carrier {
            share = share - self.expected_bus_balance;
        }
        words.push(ext_word(&share));
        words
    }
}

/// Harvest a block proof that production's block verifier accepts.
///
/// The AIR set is the monolithic verifier's (`verify_proof_parts`): page
/// configs from the ELF and the proof's ranges, `VmAirs::new` with HALT and the
/// production BITWISE, DECODE's root from the ELF. Returns the harvest and the
/// seconds of its two halves — the host verify, which only refuses (a harness
/// assert, as in every tree driver), and the replay a driver needs.
pub(super) fn harvest_block(
    opts: &crate::ProofOptions,
    elf_bytes: &[u8],
    proof: &crate::VmProof,
) -> Result<(RealBlock, f64, f64), String> {
    use crypto::fiat_shamir::is_transcript::IsTranscript;
    use rayon::prelude::*;
    use stark::verifier::IsStarkVerifier;

    let t_verify = std::time::Instant::now();
    let elf = executor::elf::Elf::load(elf_bytes).map_err(|e| format!("ELF: {e}"))?;
    let elf_digest = crate::statement::elf_digest(elf_bytes);
    let view = MultiProofView::Owned(&proof.proof);
    proof
        .table_counts
        .validate_for(crate::AcceleratorShape::KeccakRndChunked)
        .map_err(|e| format!("table counts: {e:?}"))?;
    let page_configs = crate::tables::trace_builder::Traces::page_configs_from_elf_and_runtime(
        &elf,
        &proof.runtime_page_ranges,
        proof.num_private_input_pages,
        view.len(),
    )
    .map_err(|e| format!("page configs: {e:?}"))?;
    let expected_count = proof.table_counts.total().expect("counts fit")
        + crate::FIXED_TABLE_COUNT
        + page_configs.len();
    if expected_count != view.len() {
        return Err(format!(
            "{expected_count} sub-proofs declared, {} in the proof",
            view.len()
        ));
    }
    let decode = crate::tables::decode::commitment_from_elf(&elf, opts)
        .map_err(|e| format!("DECODE commitment: {e:?}"))?;
    let airs = crate::VmAirs::new(
        &elf,
        opts,
        false,
        &page_configs,
        &proof.table_counts,
        Some(decode),
        true,
        None,
        None,
        None,
    );
    let refs = airs.air_refs();
    let seed = || {
        block_seed(
            &elf_digest,
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
    if !crate::hash_pin::BlockVerifier::<Gl, Ext3, ()>::multi_verify_views(
        &refs,
        view,
        &mut seed(),
        &expected,
    ) {
        return Err("production's verifier rejects the block".to_string());
    }
    let verify_secs = t_verify.elapsed().as_secs_f64();

    // ---- Phase A, as `multi_verify_views` absorbs it.
    let t_replay = std::time::Instant::now();
    let n = refs.len();
    let mut transcript = seed();
    let mut names = Vec::with_capacity(n);
    let mut precomputed_roots = Vec::with_capacity(n);
    let mut main_roots = Vec::with_capacity(n);
    let mut decode_root = None;
    let mut data_pages = Vec::new();
    let mut page = 0usize;
    for (idx, air) in refs.iter().enumerate() {
        let v = view.get(idx);
        names.push(air.name().to_string());
        let prep = air.is_preprocessed().then(|| {
            super::epoch_verify_tests::layout_precomputed_commitment(*air, v.trace_length())
        });
        if let Some(p) = &prep {
            transcript.append_bytes(p);
        }
        transcript.append_bytes(v.lde_trace_main_merkle_root());
        if air.name() == "DECODE" {
            decode_root = prep;
        }
        if air.name().starts_with("PAGE:") {
            let config = &page_configs[page];
            page += 1;
            if config.init_values.is_some() && !config.is_private_input {
                data_pages.push((config.page_base, prep.expect("a data page is preprocessed")));
            }
        }
        precomputed_roots.push(prep);
        main_roots.push(*v.lde_trace_main_merkle_root());
    }
    assert_eq!(
        page,
        page_configs.len(),
        "one PAGE sub-proof per page config"
    );
    assert!(refs.iter().any(|a| a.has_aux_trace()), "a block uses LogUp");
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
    for (i, t) in tables.iter().enumerate() {
        assert_eq!(
            t.precomputed_root, precomputed_roots[i],
            "instance {i}: the fork's and Phase A's preprocessed roots are one value"
        );
    }
    let replay_secs = t_replay.elapsed().as_secs_f64();

    let attested = super::programs::AttestedInputs {
        elf_digest,
        pc_start: elf.entry_point,
        decode: decode_root.ok_or("a block has a DECODE sub-proof")?,
        pages: data_pages,
    };
    Ok((
        RealBlock {
            statement: BlockStatementShape {
                public_output_len: proof.public_output.len(),
                table_counts: crate::statement::table_count_values(&proof.table_counts),
                num_private_input_pages: proof.num_private_input_pages as u64,
                fri_final_poly_log_degree: opts.fri_final_poly_log_degree,
                page_ranges: proof
                    .runtime_page_ranges
                    .iter()
                    .map(|r| (r.base, r.count))
                    .collect(),
            },
            elf_digest,
            public_output: proof.public_output.clone(),
            names,
            precomputed_roots,
            main_roots,
            tables,
            legs,
            state,
            expected_bus_balance: expected,
            attested,
        },
        verify_secs,
        replay_secs,
    ))
}

/// Leaf `k` of `partition`, `carrier` being the leaf that subtracts the
/// COMMIT-bus target.
pub(super) fn block_leaf_program(
    rb: &RealBlock,
    partition: &BlockPartition,
    k: usize,
    carrier: usize,
) -> LfmProgram {
    let instances: Vec<BlockInstance> = partition
        .leaf(k)
        .iter()
        .map(|&i| BlockInstance {
            challenge: &rb.tables[i].shape,
            verify: &rb.legs[i].verify,
            analysis: &rb.legs[i].analysis,
        })
        .collect();
    let id = rb.attested.program_id();
    let mut b = production_builder();
    emit_block_leaf(
        &mut b,
        &BlockLeafInputs {
            statement: &rb.statement,
            elf_digest: &rb.elf_digest,
            precomputed_roots: &rb.precomputed_roots,
            partition,
            leaf: k,
            instances: &instances,
            program_id: &id,
            carries_commit_target: k == carrier,
        },
    );
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

/// The layout a block's leaves and non-top nodes publish.
pub(super) fn child_layout(rb: &RealBlock) -> BlockLayout {
    BlockLayout::child(
        super::proof_arena::words_per_root(),
        rb.statement.out_halves(),
    )
}

/// A block node over `children` (leaves or nodes), emitted and validated.
pub(super) fn block_node_program(
    children: &[&RealChild],
    layout: BlockLayout,
    top: bool,
    checks: BlockBindings,
) -> LfmProgram {
    let shapes: Vec<_> = children.iter().map(|c| child_shape(c)).collect();
    let mut b = production_builder();
    emit_block_node(
        &mut b,
        &BlockNodeInputs {
            children: &shapes,
            layout,
            top,
            checks,
        },
    );
    let program = compile(b.finish());
    super::validator::validate(&program).expect("a block node must be admissible");
    program
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
    let t = std::time::Instant::now();
    let artifacts = super::program_census::build_artifacts_counted(
        program,
        opts,
        crate::hash_pin::BLOCK_HASHER,
    );
    let t_artifacts = t.elapsed().as_secs_f64();
    let t = std::time::Instant::now();
    let proved = super::proof::lfm_prove(program, &artifacts, arenas, opts)
        .unwrap_or_else(|e| panic!("{label} must prove: {e:?}"));
    let t_prove = t.elapsed().as_secs_f64();
    super::per_table_aggregator_tests::print_prove_split(label);
    let t = std::time::Instant::now();
    let (child, t_verify) =
        super::per_table_aggregator_tests::real_child_timed(artifacts, opts.clone(), &proved);
    println!(
        "   {label} TIMING: {} instructions · build_artifacts {t_artifacts:.2}s · prove \
         {t_prove:.2}s · harvest {:.2}s (verify {t_verify:.2})",
        program.instrs.len(),
        t.elapsed().as_secs_f64()
    );
    child
}

/// Levels above the leaves: fan-in [`super::per_table_aggregator::FAN_IN`],
/// leftovers wrapped in arity-1 nodes (`tree_shape`), the last level's single
/// node the top. A one-leaf block still gets a top node: the bus closes there.
/// Returns the top child and each level's wall.
pub(super) fn compose_block_tree(
    leaves: Vec<RealChild>,
    layout: BlockLayout,
    opts: &crate::ProofOptions,
    siblings: usize,
) -> (RealChild, Vec<f64>) {
    use super::per_table_aggregator::{FAN_IN, Level, tree_shape};
    let mut shape = tree_shape(leaves.len(), FAN_IN);
    if shape.is_empty() {
        shape.push(Level { arities: vec![1] });
    }
    let mut level = leaves;
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
            let program = block_node_program(&kids, layout, top, BlockBindings::ALL);
            println!("   {label}: emitted in {:.2}s", te.elapsed().as_secs_f64());
            super::per_table_aggregator_tests::census_and_panel(&program, &label, FAN_IN);
            prove_as_child(&label, &program, &block_node_arenas(&kids), opts)
        });
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
    (level.pop().expect("one"), walls)
}

/// The top node's published words, against the host's: the id, the state, the
/// public output.
pub(super) fn assert_top_claims_the_block(top: &RealChild, rb: &RealBlock) {
    let words: Vec<LfmWord> = top.public_words.iter().map(|(_, w)| *w).collect();
    let mut want = super::programs::program_id_words(&rb.attested.program_id()).to_vec();
    want.push(rb.state);
    want.extend(le_halves(&rb.public_output).into_iter().map(base_word));
    assert_eq!(
        words, want,
        "the top node publishes the block's claim: the attestation id, the state and \
         the public output"
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
        &crate::tables::MaxRowsConfig::small(),
        input,
        #[cfg(feature = "disk-spill")]
        stark::storage_mode::StorageMode::Ram,
    )
    .expect("build the traces");
    let proof = crate::block::prove_block_traces(
        &elf_bytes,
        &program,
        &mut traces,
        opts,
        None,
        stark::residency_mode::ResidencyMode::Retain,
        Vec::new(),
        &mut crate::block::BlockTimes::default(),
    )
    .expect("the small block proves");
    (elf_bytes, proof)
}

/// ★ (b) Block leaves over a real block proof EXECUTE, and publish exactly the
/// host's values: the attestation id, the transcript state after `z, α`, the
/// public output, and shares of the bus that sum to zero over the leaves.
///
/// `poc_rodata_commit` carries an ELF data page (an ELF-derived preprocessed root
/// besides DECODE's) and a public output (a nonzero COMMIT-bus target). Three
/// leaves, so the shares are split and the target sits in one of them.
#[test]
#[ignore = "proves a VM block (BITWISE is 2^20 rows); box tier"]
fn block_leaves_execute_over_a_real_block_proof() {
    let opts = fixture_block_options();
    let (elf_bytes, proof) = small_block("poc_rodata_commit", &[], &opts);
    assert!(
        crate::block::verify_block(&proof, &elf_bytes, &opts).expect("verifies"),
        "the block verifier accepts the small block"
    );
    let (rb, verify, replay) = harvest_block(&opts, &elf_bytes, &proof).expect("harvest");
    assert!(!rb.public_output.is_empty(), "a nonzero COMMIT-bus target");
    assert!(
        !rb.attested.pages.is_empty(),
        "an ELF data page is attested"
    );
    let partition = rb.partition(Some(3));
    println!(
        "BLOCK FIXTURE: {} instances, verify {verify:.2}s · replay {replay:.2}s, leaves {:?}",
        rb.num_instances(),
        partition.leaves()
    );
    let mut total = FEE::zero();
    for k in 0..partition.num_leaves() {
        let program = block_leaf_program(&rb, &partition, k, 0);
        let arenas = block_leaf_arenas(&rb, &partition, k);
        let exec = execute(&program, &arenas, &crate::hash_pin::BLOCK_HASHER)
            .unwrap_or_else(|e| panic!("leaf {k} must execute: {e:?}"));
        let got: Vec<LfmWord> = exec.public_words.iter().map(|(_, w)| *w).collect();
        assert_eq!(
            got,
            rb.expected_leaf_publics(&partition, k, 0),
            "leaf {k} publishes the host's id, state, output and share"
        );
        total += word_as_ext(got.last().expect("a share")).expect("an ext share");
        println!(
            "BLOCK FIXTURE leaf {k}: {} instances, {} instructions, executes",
            partition.leaf(k).len(),
            program.instrs.len()
        );
    }
    assert_eq!(total, FEE::zero(), "the leaves' shares close the bus");
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
    let (a, ..) = harvest_block(&opts, &elf_bytes, &proof_a).expect("harvest A");
    let (b, ..) = harvest_block(&opts, &elf_bytes, &proof_b).expect("harvest B");
    assert_ne!(a.state, b.state, "the blocks differ in their roots");
    let partition = a.partition(Some(3));
    assert_eq!(partition, b.partition(Some(3)), "the same partition");
    let layout = child_layout(&a);

    // Leaves of A, and of B; the same programs, since the shapes are equal.
    let leaves = |rb: &RealBlock, carrier: usize, tag: &str| -> Vec<RealChild> {
        (0..partition.num_leaves())
            .map(|k| {
                let program = block_leaf_program(rb, &partition, k, carrier);
                let child = prove_as_child(
                    &format!("{tag} leaf {k}"),
                    &program,
                    &block_leaf_arenas(rb, &partition, k),
                    &wrap_opts,
                );
                let words: Vec<LfmWord> = child.public_words.iter().map(|(_, w)| *w).collect();
                assert_eq!(words, rb.expected_leaf_publics(&partition, k, carrier));
                child
            })
            .collect()
    };
    let leaves_a = leaves(&a, 0, "A");
    let leaves_b = leaves(&b, 0, "B");
    assert_eq!(
        leaves_a[1].artifacts.program_id, leaves_b[1].artifacts.program_id,
        "a leaf's program is a function of the shape, so B's leaf 1 is A's program"
    );

    // ---- another block's leaf, beside A's leaf 0, at the first node.
    let mixed = [&leaves_a[0], &leaves_b[1]];
    let node = |checks: BlockBindings| {
        let program = block_node_program(&mixed, layout, false, checks);
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

    // ---- no leaf carries the target: the shares sum to it, and the top refuses.
    let uncarried_leaves = leaves(&a, usize::MAX, "A uncarried");
    let uncarried: Vec<&RealChild> = uncarried_leaves.iter().collect();
    let top_over = |checks: BlockBindings| {
        let program = block_node_program(&uncarried, layout, true, checks);
        execute(
            &program,
            &block_node_arenas(&uncarried),
            &crate::hash_pin::BLOCK_HASHER,
        )
    };
    assert!(
        top_over(BlockBindings::ALL).is_err(),
        "a top over shares that do not close the bus must not execute"
    );
    assert!(
        top_over(BlockBindings::without("bus")).is_ok(),
        "with the bus assert removed it executes: the bus assert is what refuses it"
    );

    // ---- the honest tree over A.
    let (top, walls) = compose_block_tree(leaves_a, layout, &wrap_opts, 1);
    assert_top_claims_the_block(&top, &a);
    println!("BLOCK FIXTURE TREE: levels {walls:?}, the top claims block A");
}

// ============================ production scale ============================

/// D-NOEPOCH §12.2's partition of block 25368371's 137 instances, as the
/// designer's script printed it — compared against the rule's own output on the
/// block, never substituted for it.
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

    // ---- the base.
    let base_sampler = HostSampler::start();
    let t = Instant::now();
    let (proof, times) =
        crate::block::prove_block(&elf_bytes, &input, &inner).expect("the block must prove");
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

    // ---- the harvest.
    let t = Instant::now();
    let (rb, verify, replay) = harvest_block(&inner, &elf_bytes, &proof).expect("harvest");
    drop(proof);
    let harvest = t.elapsed().as_secs_f64();
    println!(
        "   harvest: {harvest:.2}s (production verify {verify:.2}s, harness-only · replay \
         {replay:.2}s) · {} instances · public output {} bytes",
        rb.num_instances(),
        rb.public_output.len()
    );

    // ---- the partition.
    let partition = rb.partition(leaves_knob());
    let costs = rb.leg_costs();
    let k = partition.num_leaves();
    for (j, leaf) in partition.leaves().iter().enumerate() {
        let perms: usize = leaf.iter().map(|&i| costs[i]).sum();
        println!(
            "   BLOCK LEAF {j}: {} instances · {perms} perms (closed form) · idx {leaf:?}",
            leaf.len()
        );
    }
    println!(
        "   BLOCK PARTITION: {k} leaves, Σ {} perms; = D-NOEPOCH §12.2: {}",
        costs.iter().sum::<usize>(),
        if rb.num_instances() == 137 && k == D12_LEAVES.len() {
            if partition
                .leaves()
                .iter()
                .zip(D12_LEAVES)
                .all(|(a, b)| a[..] == b[..])
            {
                "yes".to_string()
            } else {
                "NO (the rule's costs differ from legmodel's)".to_string()
            }
        } else {
            "n/a (not the 137-instance block)".to_string()
        }
    );

    // ---- level 0: the leaves.
    let l0 = tree_siblings_l0().min(k);
    println!("   ★ LEVEL-0 CONCURRENCY: {l0} leaf proof(s) at once (LFM_TREE_SIBLINGS_L0)");
    super::device_permit::arm(l0);
    let l0_sampler = HostSampler::start();
    let t = Instant::now();
    let leaves = in_index_order(k, l0, |j| {
        let label = format!("BLOCK L0 leaf {j}");
        let te = Instant::now();
        let program = block_leaf_program(&rb, &partition, j, 0);
        let arenas = block_leaf_arenas(&rb, &partition, j);
        println!(
            "   {label}: emitted + arenas in {:.2}s",
            te.elapsed().as_secs_f64()
        );
        super::per_table_aggregator_tests::census_and_panel(
            &program,
            &label,
            super::per_table_aggregator::FAN_IN,
        );
        let child = prove_as_child(&label, &program, &arenas, &wrap_opts);
        let words: Vec<LfmWord> = child.public_words.iter().map(|(_, w)| *w).collect();
        assert_eq!(
            words,
            rb.expected_leaf_publics(&partition, j, 0),
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
    let (top, walls) = compose_block_tree(leaves, child_layout(&rb), &wrap_opts, siblings);
    super::device_permit::arm(1);
    let interior = t.elapsed().as_secs_f64();
    assert_top_claims_the_block(&top, &rb);

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
}
