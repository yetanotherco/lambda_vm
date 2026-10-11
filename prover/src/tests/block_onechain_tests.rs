//! D-ONECHAIN B1 at the block: one WHIR chain per run of a group's stacked
//! polynomials ([`GroupFold::Indexed`]) through the block's frame, prover and
//! verifier (REV-B2-JUDGE B1-1 (a), B1-5). The chain-level gates are
//! `multilinear`'s `whir_group_tests`.
//!
//! On the host build only: a grouped round 0 folds host codewords, and the
//! card's grouped round 0 is D-ONECHAIN's stage B3. Until then a card prover
//! refuses an `Indexed` group by name (`DeviceFailed`), and a card build
//! leaves these out.

use crate::block_whir::{
    self, BlockFormat, BlockOptions, BlockWhirProof, Deviations, prove_block_whir_with,
    verify_block_whir,
};
use crate::tables::MaxRowsConfig;
use crate::test_utils::asm_elf_bytes;
use crate::zf_format::ZfFormat;
use multilinear::stacked_eval::index_chains;
use multilinear::whir_chain::{
    ArgueFormat, GroupFold, MAX_INDEX_POLYS, MAX_INDEX_VARS, RoundOpenings, StackVars,
    check_index_domain, index_vars,
};
use stark::proof::options::{BaseFormat, ProofOptions};

/// A stack of 2^10 and `group_polys` polynomials a group under `base`, so a
/// small program spans several groups, opened per `group`.
fn format(base: BaseFormat, group_polys: usize, group: GroupFold) -> BlockFormat {
    let mut zf = ZfFormat::DEFAULT.with_base(base);
    zf.whir_stack = StackVars::new(10).expect("a valid stack");
    BlockFormat {
        zf,
        group_polys,
        max_groups: block_whir::BLOCK_MAX_GROUPS,
        prepared: true,
        argue: ArgueFormat::PerTable,
        group,
    }
}

/// The production format, opened per `group`: one group holds a small program.
fn production(group: GroupFold) -> BlockFormat {
    BlockFormat {
        zf: ZfFormat::DEFAULT,
        group_polys: block_whir::BLOCK_GROUP_POLYS,
        max_groups: block_whir::BLOCK_MAX_GROUPS,
        prepared: true,
        argue: ArgueFormat::PerTable,
        group,
    }
}

/// `block_whir_tests`' small-program options at `max_rows`: groups retire and
/// revive between the phases (3 binary levels dropped).
fn options(max_rows: MaxRowsConfig) -> BlockOptions {
    BlockOptions {
        max_rows,
        keccak_rnd_rows_log2: 16,
        ecdas_rows_log2: block_whir::BLOCK_ECDAS_ROWS_LOG2,
        keccak_rows_log2: block_whir::BLOCK_KECCAK_ROWS_LOG2,
        ecsm_rows_log2: block_whir::BLOCK_ECSM_ROWS_LOG2,
        drop_levels: multilinear::whir_commit::TreeDrop::uniform(3),
        window_log2: None,
        stream_keccak_rnd: false,
        stream_memw_lt: false,
        drop_streamed_ops: false,
        layout_workers: 0,
        layout_ahead: Some(2),
        pack_rest_as_laid_out: false,
        narrow: stark::multilinear_block::Narrowing::Card { min_cells: 0 },
        upload_ahead: true,
        memlog: false,
        finish_keccak_rnd_chunks: true,
        finish_cuts: true,
        finish_stream: true,
        rest_layout_bytes: Some(1 << 20),
        pack_finished: true,
        gpack: true,
        spill: block_whir::BlockSpillPolicy::Off,
        regen: None,
    }
}

fn prove(elf: &[u8], format: &BlockFormat, max_rows: MaxRowsConfig) -> BlockWhirProof {
    prove_block_whir_with(
        elf,
        &[],
        &ProofOptions::default_test_options(),
        format,
        &options(max_rows),
        &Deviations::default(),
    )
    .expect("prove")
    .0
}

fn verify(proof: &BlockWhirProof, elf: &[u8], format: &BlockFormat) -> bool {
    verify_block_whir(proof, elf, &ProofOptions::default_test_options(), format).unwrap_or(false)
}

/// The width of every grouped chain's run in the proof, groups and prepared
/// stacks alike: its round-0 blocks a query.
fn grouped_widths(proof: &BlockWhirProof) -> Vec<usize> {
    proof
        .proof
        .columns
        .iter()
        .chain(&proof.prepared)
        .flat_map(|stack| &stack.polys)
        .filter_map(|chain| match &chain.rounds.first()?.openings {
            RoundOpenings::BaseGrouped(g) => g.current.first().map(Vec::len),
            _ => None,
        })
        .collect()
}

/// REV-B2-JUDGE B1-1 (a), REV-B2-A's frame probe under `Indexed`: a table
/// stated at 2^27 rows alone in its group (the frame admits it, exempt from
/// `group_polys`) stacks into one polynomial per column — 38 for the CPU — and
/// the verifier's split cuts them into nine runs of four and one of two, each
/// within `MAX_INDEX_VARS` index variables and the two-adicity.
#[test]
fn a_wide_singleton_group_splits_into_runs_of_four() {
    let elf = asm_elf_bytes("sub");
    let indexed = production(GroupFold::Indexed);
    let proof = prove(
        &elf,
        &production(GroupFold::PerPoly),
        MaxRowsConfig::default(),
    );
    let opts = ProofOptions::default_test_options();
    let honest = block_whir::block_frame(proof.statement(), &elf, &opts, &indexed).expect("frame");
    let chunked: Vec<std::ops::Range<usize>> =
        block_whir::chunked_table_ranges(&proof.table_counts)
            .iter()
            .map(|(_, r, _)| r.clone())
            .collect();
    let (t, width) = honest
        .shapes
        .iter()
        .enumerate()
        .filter(|(i, _)| !chunked.iter().any(|r| r.contains(i)))
        .map(|(i, &(w, _))| (i, w))
        .max_by_key(|&(_, w)| w)
        .expect("a table outside the chunked families");
    let n = proof.table_num_vars.len();
    let mut heights = proof.table_num_vars.clone();
    heights[t] = block_whir::BLOCK_MAX_TABLE_VARS as u8;
    let groups: Vec<Vec<u32>> = vec![
        vec![t as u32],
        (0..n as u32).filter(|&i| i as usize != t).collect(),
    ];
    let statement = block_whir::BlockStatement {
        table_num_vars: &heights,
        groups: &groups,
        ..proof.statement()
    };
    let frame = block_whir::block_frame(statement, &elf, &opts, &indexed)
        .expect("the verifier's frame admits the shape");
    assert_eq!(frame.config.format.group, GroupFold::Indexed);
    let layout = &frame.stack_layouts[0];
    assert_eq!(layout.n_stack(), block_whir::BLOCK_MAX_TABLE_VARS);
    let polys = layout.num_polys();
    println!(
        "ONECHAIN B1-1a: table {t} ({}) width {width} stated at 2^{}: group 0 stacks into \
         {polys} polynomials",
        honest.airs.air_refs()[t].name(),
        block_whir::BLOCK_MAX_TABLE_VARS
    );
    assert_eq!(polys, width, "one 2^27 polynomial per column");
    let runs: Vec<usize> = index_chains(polys, frame.config.format.group)
        .iter()
        .map(|r| r.len())
        .collect();
    let mut want = vec![MAX_INDEX_POLYS; polys / MAX_INDEX_POLYS];
    if !polys.is_multiple_of(MAX_INDEX_POLYS) {
        want.push(polys % MAX_INDEX_POLYS);
    }
    assert_eq!(runs, want);
    if width == 38 {
        assert_eq!(runs, [vec![4; 9], vec![2]].concat());
    }
    for &run in &runs {
        assert!(index_vars(run) <= MAX_INDEX_VARS);
        check_index_domain(
            layout.n_stack(),
            index_vars(run),
            frame.config.log_blowup,
            32,
        )
        .expect("the virtual domain exists");
    }
}

/// REV-B2-JUDGE B1-5 at the block, and the grouped chain's round trip
/// through it: at up to two, three and four polynomials a group, under RPX
/// and under Poseidon1's 4-ary capped trees, an `Indexed` block proves and
/// verifies with grouped chains, its phase-A roots are the per-polynomial
/// block's (B2-sep), and neither mode's proof verifies under the other's
/// verifier.
#[test]
fn an_indexed_block_round_trips_and_refuses_the_other_mode() {
    let elf = asm_elf_bytes("all_instructions_64");
    for base in [BaseFormat::RPX, BaseFormat::P1_WHIR] {
        for group_polys in 2..=MAX_INDEX_POLYS {
            let indexed = format(base, group_polys, GroupFold::Indexed);
            let per_poly = format(base, group_polys, GroupFold::PerPoly);
            let a = prove(&elf, &indexed, MaxRowsConfig::small());
            let b = prove(&elf, &per_poly, MaxRowsConfig::small());
            let widths = grouped_widths(&a);
            println!(
                "ONECHAIN B1-5 {base:?} group_polys {group_polys}: {} groups, grouped runs \
                 {widths:?}",
                a.proof.columns.len()
            );
            assert!(
                widths.iter().any(|&w| w >= 2),
                "{base:?} {group_polys}: no grouped chain"
            );
            assert!(widths.iter().all(|&w| (2..=group_polys).contains(&w)));
            assert!(grouped_widths(&b).is_empty());
            assert_eq!(a.proof.roots, b.proof.roots, "phase A's roots");
            assert!(verify(&a, &elf, &indexed), "{base:?} {group_polys}");
            assert!(verify(&b, &elf, &per_poly), "{base:?} {group_polys}");
            assert!(!verify(&a, &elf, &per_poly), "{base:?} {group_polys}");
            assert!(!verify(&b, &elf, &indexed), "{base:?} {group_polys}");
        }
    }
}
