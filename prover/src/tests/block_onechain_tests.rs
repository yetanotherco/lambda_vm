//! D-ONECHAIN B1 at the block: one WHIR chain per run of a group's stacked
//! polynomials ([`GroupFold::Indexed`]) through the block's frame, prover and
//! verifier (REV-B2-JUDGE B1-1 (a), B1-5). The chain-level gates are
//! `multilinear`'s `whir_group_tests`, and on the card its
//! `whir_group_device`.
//!
//! On either build: a card build runs a grouped round 0 on the card where it
//! takes the stack, and on the host where it does not.

use crate::block_whir::{
    self, BlockFormat, BlockOptions, BlockRefusal, BlockWhirProof, Deviations,
    prove_block_whir_with,
};
use crate::tables::MaxRowsConfig;
use crate::test_utils::asm_elf_bytes;
use crate::zf_format::ZfFormat;
use multilinear::stacked_eval::index_chains;
use multilinear::whir_chain::{
    ArgueFormat, GroupFold, MAX_INDEX_POLYS, MAX_INDEX_VARS, RoundOpenings, StackVars,
    check_index_domain, index_vars,
};
use stark::multilinear_table::batched::VerifierChecks;
use stark::proof::options::{BaseFormat, ProofOptions};

/// A stack of 2^10 and `group_polys` polynomials a group under `base`, so a
/// small program spans several groups, argued per `argue` and opened per
/// `group`. A card declines to open a stack this small (its sumcheck
/// threshold is 2^12), so on a card build a coset round 0 here encodes its
/// successor on the host ([`multilinear::gpu::coset_host_encodes`]).
fn format(
    base: BaseFormat,
    group_polys: usize,
    group: GroupFold,
    argue: ArgueFormat,
) -> BlockFormat {
    format_at(base, group_polys, group, argue, 10)
}

/// [`format`] at a stack of `2^stack_vars`.
fn format_at(
    base: BaseFormat,
    group_polys: usize,
    group: GroupFold,
    argue: ArgueFormat,
    stack_vars: usize,
) -> BlockFormat {
    let mut zf = ZfFormat::DEFAULT.with_base(base);
    zf.whir_stack = StackVars::new(stack_vars).expect("a valid stack");
    BlockFormat {
        zf,
        group_polys,
        max_groups: block_whir::BLOCK_MAX_GROUPS,
        prepared: true,
        argue,
        group,
        leaf_scale: crate::lfm::whir_block::LeafScale::Fixed,
        leaf_query: crate::lfm::whir_chain::LeafQuery::V1,
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
        leaf_scale: crate::lfm::whir_block::LeafScale::Fixed,
        leaf_query: crate::lfm::whir_chain::LeafQuery::V1,
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
        copy_rows_log2: block_whir::BLOCK_COPY_ROWS_LOG2,
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
        rest_build_bytes: None,
        pack_finished: true,
        gpack: true,
        regen: None,
        phase_b_overlap: stark::multilinear_block::PhaseBOverlap::Off,
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

/// Each chain's width in one stack's opening, in proof order: a grouped
/// chain's round-0 blocks a query, 1 for a per-polynomial chain.
fn chain_widths<F, E>(stack: &multilinear::stacked_eval::StackedProof<F, E>) -> Vec<usize>
where
    F: math::field::traits::IsField,
    E: math::field::traits::IsField,
{
    stack
        .polys
        .iter()
        .map(|chain| match chain.rounds.first().map(|r| &r.openings) {
            Some(RoundOpenings::BaseGrouped(g)) => g.current.first().map_or(0, Vec::len),
            _ => 1,
        })
        .collect()
}

/// REV-B2-JUDGE B1-1 (a), REV-B2-A's frame probe under `Indexed`: a table
/// stated at 2^27 rows alone in its group (the frame admits it, exempt from
/// `group_polys`) stacks into one polynomial per column, and the verifier's
/// split cuts them into runs of four and one of the rest, each within
/// `MAX_INDEX_VARS` index variables and the two-adicity. The table is the
/// widest outside the chunked families and the CPU's: a CPU table stated past
/// `BLOCK_CPU_MAX_VARS` (2^21 rows, the block's chunk height) is refused by
/// the frame, which this checks first on the widest CPU table.
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
    let cpu_start = crate::cpu_proof_start(true, &proof.table_counts);
    let cpus = cpu_start..cpu_start + proof.table_counts.cpu;
    let widest = |keep: &dyn Fn(usize) -> bool| {
        honest
            .shapes
            .iter()
            .enumerate()
            .filter(|&(i, _)| keep(i) && !chunked.iter().any(|r| r.contains(&i)))
            .map(|(i, &(w, _))| (i, w))
            .max_by_key(|&(_, w)| w)
    };
    // Table `t` stated at 2^27 rows, alone in its group.
    let n = proof.table_num_vars.len();
    let singleton = |t: usize| {
        let mut heights = proof.table_num_vars.clone();
        heights[t] = block_whir::BLOCK_MAX_TABLE_VARS as u8;
        let groups: Vec<Vec<u32>> = vec![
            vec![t as u32],
            (0..n as u32).filter(|&i| i as usize != t).collect(),
        ];
        (heights, groups)
    };
    let (cpu, _) = widest(&|i| cpus.contains(&i)).expect("a CPU table");
    let (heights, groups) = singleton(cpu);
    let statement = block_whir::BlockStatement {
        table_num_vars: &heights,
        groups: &groups,
        ..proof.statement()
    };
    assert!(
        matches!(
            block_whir::block_frame(statement, &elf, &opts, &indexed),
            Err(crate::Error::InvalidTableCounts(m)) if m.starts_with("CPU[")
        ),
        "a CPU table stated at 2^27 rows is refused by the frame"
    );
    let (t, width) = widest(&|i| !cpus.contains(&i))
        .expect("a table outside the chunked families and the CPU's");
    assert!(
        width > MAX_INDEX_POLYS,
        "table {t} is {width} columns wide: one run"
    );
    let (heights, groups) = singleton(t);
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
/// through it: at up to two, three and four polynomials a group, under the
/// block's base (Poseidon1's 4-ary capped trees), an `Indexed` block proves and
/// verifies with grouped chains — each group's chains exactly the verifier's
/// split of its stack, runs of at most `MAX_INDEX_POLYS` whatever the group
/// budget is (the budget bounds a group of several tables; a table that alone
/// needs more is a group of its own, and its runs are of four) — runs of two,
/// three and four each occur, its phase-A roots are the per-polynomial
/// block's (B2-sep), and neither mode's proof verifies under the other's
/// verifier — refused by the STATEMENT: the two modes' statement bytes differ
/// in one bit, the fold word's bit 47, so the transcript parts at the first
/// absorb and the first group's argue refuses, before any opening's shape is
/// read (the shapes' own refusals, `ChainCount` and the per-polynomial count,
/// are `whir_group_tests`'). Each group's tables argued one by one.
#[test]
fn an_indexed_block_round_trips_and_refuses_the_other_mode() {
    let seen = round_trips_and_refuses_the_other_mode(ArgueFormat::PerTable, 2..=MAX_INDEX_POLYS);
    // Runs of two, three and four each round-tripped and each was refused
    // across the modes: the three budgets pack the program into all of them.
    assert_eq!(
        seen.into_iter().collect::<Vec<_>>(),
        (2..=MAX_INDEX_POLYS).collect::<Vec<_>>(),
        "a grouped run of every width from two to {MAX_INDEX_POLYS}"
    );
}

/// [`an_indexed_block_round_trips_and_refuses_the_other_mode`] under the
/// argue production runs, each group's tables argued together
/// (`ArgueFormat::BATCHED`), at the production budget of three polynomials a
/// group: the grouped opening takes its points and values from whichever
/// argue ran, and this is the pairing a default would ship. The same round
/// trip, the same equal roots, the statements one bit apart, and both
/// cross-mode refusals at the statement, never at an opening's shape — over
/// runs of two, three and four.
#[test]
fn an_indexed_block_round_trips_under_the_batched_argue() {
    let seen = round_trips_and_refuses_the_other_mode(
        ArgueFormat::BATCHED,
        block_whir::BLOCK_GROUP_POLYS..=block_whir::BLOCK_GROUP_POLYS,
    );
    // The production budget packs the program into runs of every width: the
    // tables of a group, and two tables alone in theirs split into fours.
    assert_eq!(
        seen.into_iter().collect::<Vec<_>>(),
        (2..=MAX_INDEX_POLYS).collect::<Vec<_>>(),
        "a grouped run of every width under the batched argue"
    );
}

/// ★ The base code's rate (`LAMBDA_VM_ZF_WHIR_BLOWUP`) under one chain a run
/// and the batched argue — the composed format, at the production budget of
/// three polynomials a group: a block at blowup 2 has grouped chains and
/// verifies; a verifier at blowup 4 refuses it, and a verifier at blowup 2
/// refuses a block at blowup 4; and at blowup 2 a per-polynomial verifier
/// refuses the grouped block.
#[test]
fn an_indexed_block_at_blowup_2_is_refused_by_a_verifier_at_blowup_4() {
    let elf = asm_elf_bytes("all_instructions_64");
    let opts = ProofOptions::default_test_options();
    let at = |blowup: u8, group: GroupFold| {
        let mut f = format(
            crate::hash_pin::BLOCK_BASE,
            block_whir::BLOCK_GROUP_POLYS,
            group,
            ArgueFormat::BATCHED,
        );
        f.zf.whir_blowup = blowup;
        f
    };
    let verify = |proof: &BlockWhirProof, format: &BlockFormat| {
        block_whir::verify_block_whir(proof, &elf, &opts, format).unwrap_or(false)
    };
    let half = prove(&elf, &at(2, GroupFold::Indexed), MaxRowsConfig::small());
    assert!(
        grouped_widths(&half).iter().any(|&w| w >= 2),
        "a grouped chain at blowup 2"
    );
    assert!(verify(&half, &at(2, GroupFold::Indexed)), "the control");
    assert!(
        !verify(&half, &at(4, GroupFold::Indexed)),
        "a verifier at blowup 4 must refuse a block at blowup 2"
    );
    assert!(
        !verify(&half, &at(2, GroupFold::PerPoly)),
        "a per-polynomial verifier must refuse the grouped block at blowup 2"
    );
    let four = prove(&elf, &at(4, GroupFold::Indexed), MaxRowsConfig::small());
    assert!(
        verify(&four, &at(4, GroupFold::Indexed)),
        "the control at blowup 4"
    );
    assert!(
        !verify(&four, &at(2, GroupFold::Indexed)),
        "a verifier at blowup 2 must refuse a block at blowup 4"
    );
}

/// ★ The coset round-0 leaf order (`LAMBDA_VM_ZF_WHIR_LEAVES`) under one chain
/// a run and the batched argue, at the production budget of three polynomials
/// a group, at either blowup: a coset block has grouped chains and verifies
/// under its own format; a verifier of the natural order refuses it, and so
/// does a coset verifier at the other blowup; a coset verifier refuses the
/// natural block.
#[test]
fn an_indexed_coset_block_is_refused_by_a_natural_verifier() {
    use multilinear::whir_chain::LeafOrder;
    let elf = asm_elf_bytes("all_instructions_64");
    let opts = ProofOptions::default_test_options();
    let at = |blowup: u8, leaves: LeafOrder| {
        let mut f = format(
            crate::hash_pin::BLOCK_BASE,
            block_whir::BLOCK_GROUP_POLYS,
            GroupFold::Indexed,
            ArgueFormat::BATCHED,
        );
        f.zf.whir_blowup = blowup;
        f.zf.whir_leaves = leaves;
        f
    };
    let verify = |proof: &BlockWhirProof, format: &BlockFormat| {
        block_whir::verify_block_whir(proof, &elf, &opts, format).unwrap_or(false)
    };
    for (blowup, other) in [(4, 2), (2, 4)] {
        let encodes = multilinear::gpu::coset_host_encodes();
        let coset = prove(&elf, &at(blowup, LeafOrder::Coset), MaxRowsConfig::small());
        assert!(
            grouped_widths(&coset).iter().any(|&w| w >= 2),
            "a grouped chain under the coset order at blowup {blowup}"
        );
        // On a card the 2^10 stack's openings fall to the host, and each coset
        // round 0 encodes its successor there: counted, the prove goes on.
        if cfg!(feature = "cuda") {
            assert!(
                multilinear::gpu::coset_host_encodes() > encodes,
                "the host encodes counted at blowup {blowup}"
            );
        }
        assert!(
            verify(&coset, &at(blowup, LeafOrder::Coset)),
            "the control at blowup {blowup}"
        );
        assert!(
            !verify(&coset, &at(blowup, LeafOrder::Natural)),
            "a natural verifier must refuse the coset block at blowup {blowup}"
        );
        assert!(
            !verify(&coset, &at(other, LeafOrder::Coset)),
            "a coset verifier at blowup {other} must refuse the coset block at blowup {blowup}"
        );
    }
    let natural = prove(&elf, &at(4, LeafOrder::Natural), MaxRowsConfig::small());
    assert!(
        verify(&natural, &at(4, LeafOrder::Natural)),
        "the natural control"
    );
    assert!(
        !verify(&natural, &at(4, LeafOrder::Coset)),
        "a coset verifier must refuse the natural block"
    );
}

/// ★ The composed format at the block: one chain a run, the coset round-0
/// order and blowup 2, under the batched argue at the production budget. A
/// block proved in it verifies, and is refused by the verifier of each format
/// one setting away — blowup 4, the natural order, one chain a polynomial —
/// and a block one setting away (the natural order) is refused by the composed
/// verifier. Each setting is a verifier constant; the proof carries none.
#[test]
fn a_composed_block_is_refused_one_setting_away() {
    use multilinear::whir_chain::LeafOrder;
    let elf = asm_elf_bytes("all_instructions_64");
    let opts = ProofOptions::default_test_options();
    let at = |group: GroupFold, leaves: LeafOrder, blowup: u8| {
        let mut f = format(
            crate::hash_pin::BLOCK_BASE,
            block_whir::BLOCK_GROUP_POLYS,
            group,
            ArgueFormat::BATCHED,
        );
        f.zf.whir_leaves = leaves;
        f.zf.whir_blowup = blowup;
        f
    };
    let verify = |proof: &BlockWhirProof, format: &BlockFormat| {
        block_whir::verify_block_whir(proof, &elf, &opts, format).unwrap_or(false)
    };
    let composed = at(GroupFold::Indexed, LeafOrder::Coset, 2);
    let encodes = multilinear::gpu::coset_host_encodes();
    let proof = prove(&elf, &composed, MaxRowsConfig::small());
    assert!(
        grouped_widths(&proof).iter().any(|&w| w >= 2),
        "a grouped chain in the composed block"
    );
    // On a card the 2^10 stack's openings fall to the host: each coset round
    // 0 encodes its successor there, counted, and the prove goes on.
    if cfg!(feature = "cuda") {
        assert!(
            multilinear::gpu::coset_host_encodes() > encodes,
            "the host encodes counted"
        );
    }
    assert!(verify(&proof, &composed), "the composed format's control");
    for (label, other) in [
        ("blowup 4", at(GroupFold::Indexed, LeafOrder::Coset, 4)),
        (
            "the natural order",
            at(GroupFold::Indexed, LeafOrder::Natural, 2),
        ),
        (
            "one chain a polynomial",
            at(GroupFold::PerPoly, LeafOrder::Coset, 2),
        ),
    ] {
        assert!(
            !verify(&proof, &other),
            "a verifier at {label} must refuse the composed block"
        );
    }
    let natural = at(GroupFold::Indexed, LeafOrder::Natural, 2);
    let neighbour = prove(&elf, &natural, MaxRowsConfig::small());
    assert!(verify(&neighbour, &natural), "the natural order's control");
    assert!(
        !verify(&neighbour, &composed),
        "the composed verifier must refuse a block in the natural order"
    );
}

/// The smallest stack a card opens, in variables: its sumcheck threshold
/// (`multilinear::gpu`'s 2^12). A smaller one's openings fall to the host.
const CARD_OPEN_VARS: usize = 12;

/// Every stack the block opens, its groups' then its prepared ones, as the
/// verifier frames them: the stack's height in variables and its chains.
fn stack_chains(elf: &[u8], proof: &BlockWhirProof, format: &BlockFormat) -> Vec<(usize, usize)> {
    let opts = ProofOptions::default_test_options();
    let frame = block_whir::block_frame(proof.statement(), elf, &opts, format).expect("frame");
    let columns = block_whir::prepared_tables(&frame.airs, &frame.page_configs, format)
        .expect("prepared tables");
    let prepared = block_whir::group_stacks(columns, proof.statement().groups)
        .expect("prepared stacks")
        .iter()
        .map(|stack| {
            stark::multilinear_table::global_layout(&stack.shapes(), frame.config.format.stack)
                .expect("prepared layout")
                .n_stack()
        })
        .collect::<Vec<_>>();
    assert_eq!(frame.stack_layouts.len(), proof.proof.columns.len());
    assert_eq!(prepared.len(), proof.prepared.len());
    frame
        .stack_layouts
        .iter()
        .map(|layout| layout.n_stack())
        .chain(prepared)
        .zip(
            proof
                .proof
                .columns
                .iter()
                .chain(&proof.prepared)
                .map(|stack| stack.polys.len()),
        )
        .collect()
}

/// ★ The defaults' block at a stack cap the card opens (2^12, its sumcheck
/// threshold): one chain a run under the coset order, at blowup 4 and under the
/// composed format at blowup 2, proves and verifies, and on a card build no
/// chain of a stack of 2^12 or more encodes its round-0 successor on the host.
/// A cap is not a floor: a stack of fewer cells is lower (this program has one
/// at 2^10), the card declines its openings and each of its chains may encode
/// once on the host (the 2^10 tests above), so the count is bounded by those
/// chains, printed, and is 0 when there are none.
#[test]
fn a_coset_block_at_a_card_sized_stack_encodes_no_successor_on_the_host() {
    use multilinear::whir_chain::LeafOrder;
    let elf = asm_elf_bytes("all_instructions_64");
    let opts = ProofOptions::default_test_options();
    for blowup in [4, 2] {
        let mut format = format_at(
            crate::hash_pin::BLOCK_BASE,
            block_whir::BLOCK_GROUP_POLYS,
            GroupFold::Indexed,
            ArgueFormat::BATCHED,
            CARD_OPEN_VARS,
        );
        format.zf.whir_blowup = blowup;
        format.zf.whir_leaves = LeafOrder::Coset;
        let encodes = multilinear::gpu::coset_host_encodes();
        let proof = prove(&elf, &format, MaxRowsConfig::small());
        let host = multilinear::gpu::coset_host_encodes() - encodes;
        assert!(
            block_whir::verify_block_whir(&proof, &elf, &opts, &format).unwrap_or(false),
            "the block verifies at blowup {blowup}"
        );
        let stacks = stack_chains(&elf, &proof, &format);
        let below: usize = stacks
            .iter()
            .filter(|&&(vars, _)| vars < CARD_OPEN_VARS)
            .map(|&(_, chains)| chains)
            .sum();
        println!(
            "ONECHAIN CARD-SIZED blowup {blowup}: stacks (vars, chains) {stacks:?}; {below} \
             chains below 2^{CARD_OPEN_VARS}; host encodes {host}"
        );
        // Not vacuous: a grouped chain at a stack the card opens.
        let grouped_at_size = proof
            .proof
            .columns
            .iter()
            .chain(&proof.prepared)
            .zip(&stacks)
            .any(|(stack, &(vars, _))| {
                vars >= CARD_OPEN_VARS && chain_widths(stack).iter().any(|&w| w >= 2)
            });
        assert!(
            grouped_at_size,
            "a grouped chain at a stack of 2^{CARD_OPEN_VARS} or more, blowup {blowup}"
        );
        // A chain encodes its successor at most once, at round 0; the card
        // opens every stack of 2^12 or more, so only the chains below may.
        assert!(
            host <= below as u64,
            "no host encode at a stack the card opens, blowup {blowup}: {host} host encodes, \
             {below} chains below 2^{CARD_OPEN_VARS}"
        );
    }
}

/// The round trip and the cross-mode refusals at each budget of `budgets`
/// polynomials a group, under `argue`; returns the grouped run widths seen.
fn round_trips_and_refuses_the_other_mode(
    argue: ArgueFormat,
    budgets: std::ops::RangeInclusive<usize>,
) -> std::collections::BTreeSet<usize> {
    let elf = asm_elf_bytes("all_instructions_64");
    let opts = ProofOptions::default_test_options();
    let traced = |proof: &BlockWhirProof, format: &BlockFormat| {
        block_whir::verify_block_whir_traced(proof, &elf, &opts, format, false, VerifierChecks::ALL)
            .expect("the statement frames under either mode")
    };
    let base = crate::hash_pin::BLOCK_BASE;
    let mut seen = std::collections::BTreeSet::new();
    for group_polys in budgets {
        let indexed = format(base, group_polys, GroupFold::Indexed, argue);
        let per_poly = format(base, group_polys, GroupFold::PerPoly, argue);
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
        // Each group's chains are the verifier's split of its stack, exactly:
        // runs of at most `MAX_INDEX_POLYS`, whatever `group_polys` is. The
        // budget bounds a group of SEVERAL tables; a table that alone needs
        // more is a group of its own (the frame's rule and the packing's).
        let framed = block_whir::block_frame(a.statement(), &elf, &opts, &indexed).expect("frame");
        assert_eq!(a.proof.columns.len(), framed.stack_layouts.len());
        assert_eq!(b.proof.columns.len(), framed.stack_layouts.len());
        for (g, layout) in framed.stack_layouts.iter().enumerate() {
            let polys = layout.num_polys();
            let got = chain_widths(&a.proof.columns[g]);
            println!(
                "ONECHAIN B1-5 {base:?} group_polys {group_polys} group {g}: {} tables, \
                 {polys} polynomials, chains {got:?}",
                framed.sizes[g]
            );
            let split: Vec<usize> = index_chains(polys, GroupFold::Indexed)
                .iter()
                .map(|run| run.len())
                .collect();
            assert_eq!(
                got, split,
                "{base:?} {group_polys} group {g}: the proof's chains are the split's"
            );
            assert!(
                got.iter().all(|w| (1..=MAX_INDEX_POLYS).contains(w)),
                "{base:?} {group_polys} group {g}: a run of at most {MAX_INDEX_POLYS}"
            );
            assert!(
                polys <= group_polys || framed.sizes[g] == 1,
                "{base:?} {group_polys} group {g}: {polys} polynomials over {} tables — only a \
                 table alone may need more than the group budget",
                framed.sizes[g]
            );
            assert_eq!(
                chain_widths(&b.proof.columns[g]),
                vec![1; polys],
                "{base:?} {group_polys} group {g}: one chain a polynomial under PerPoly"
            );
        }
        for (k, stack) in a.prepared.iter().enumerate() {
            let got = chain_widths(stack);
            println!(
                "ONECHAIN B1-5 {base:?} group_polys {group_polys} prepared stack {k}: chains \
                 {got:?}"
            );
            assert!(got.iter().all(|w| (1..=MAX_INDEX_POLYS).contains(w)));
        }
        seen.extend(widths.iter().copied());
        assert!(grouped_widths(&b).is_empty());
        println!(
            "ONECHAIN B1-5 {base:?} group_polys {group_polys}: phase-A roots {} (Indexed), {} \
             (PerPoly), equal: {}",
            a.proof.roots.len(),
            b.proof.roots.len(),
            a.proof.roots == b.proof.roots
        );
        assert_same_roots(&elf, &a, &b, &indexed);
        let (own_a, own_b) = (traced(&a, &indexed), traced(&b, &per_poly));
        println!(
            "ONECHAIN B1-5 {base:?} group_polys {group_polys}: Indexed under Indexed {own_a:?}, \
             PerPoly under PerPoly {own_b:?}"
        );
        assert!(own_a.is_ok(), "{base:?} {group_polys}");
        assert!(own_b.is_ok(), "{base:?} {group_polys}");

        // The statement: one bit apart, the fold word's 47.
        let frame = |format: &BlockFormat| {
            block_whir::block_frame(a.statement(), &elf, &opts, format).expect("frame")
        };
        let tag = block_whir::block_statement_tag(&base);
        let digest = crate::statement::elf_digest(&elf);
        let (sa, sb) = (
            block_whir::block_statement_bytes(
                a.statement(),
                &tag,
                &digest,
                &frame(&indexed).config,
            ),
            block_whir::block_statement_bytes(
                a.statement(),
                &tag,
                &digest,
                &frame(&per_poly).config,
            ),
        );
        assert_eq!(sa.len(), sb.len());
        let differ: Vec<(usize, u8)> = sa
            .iter()
            .zip(&sb)
            .enumerate()
            .filter(|(_, (x, y))| x != y)
            .map(|(i, (x, y))| (i, x ^ y))
            .collect();
        println!(
            "ONECHAIN B1-5 {base:?} group_polys {group_polys}: statements of {} bytes differ at \
             (byte, xor) {differ:?}",
            sa.len()
        );
        assert_eq!(differ.len(), 1, "{base:?} {group_polys}: {differ:?}");
        assert_eq!(differ[0].1, 0x80, "bit 47 is byte 5's top bit");

        for (label, proof, format) in [
            ("Indexed under PerPoly", &a, &per_poly),
            ("PerPoly under Indexed", &b, &indexed),
        ] {
            match traced(proof, format) {
                Err(BlockRefusal::Proof(e)) => {
                    println!("ONECHAIN B1-5 {base:?} {group_polys} {label}: refused with {e:?}");
                    assert!(
                        !matches!(
                            e,
                            multilinear::Error::QueryCountMismatch { .. }
                                | multilinear::Error::GroupedShape(_)
                                | multilinear::Error::ColumnOpeningRejected { .. }
                        ),
                        "{base:?} {group_polys} {label}: refused at an opening's shape or \
                             chain, not by the statement: {e:?}"
                    );
                }
                other => panic!("{base:?} {group_polys} {label}: {other:?}"),
            }
        }
    }
    println!("ONECHAIN B1-5 {argue:?} run widths seen: {seen:?}");
    seen
}

/// The two modes' phase-A roots are one list (B2-sep). On a mismatch, names the
/// first differing root's group, polynomial and tables, then fails: a
/// difference is a design finding, not a test to adjust.
fn assert_same_roots(elf: &[u8], a: &BlockWhirProof, b: &BlockWhirProof, format: &BlockFormat) {
    let Some(first) = (0..a.proof.roots.len().max(b.proof.roots.len()))
        .find(|&i| a.proof.roots.get(i) != b.proof.roots.get(i))
    else {
        return;
    };
    let opts = ProofOptions::default_test_options();
    let frame = block_whir::block_frame(a.statement(), elf, &opts, format).expect("frame");
    let mut at = first;
    let mut g = 0;
    while g < frame.stack_layouts.len() && at >= frame.stack_layouts[g].num_polys() {
        at -= frame.stack_layouts[g].num_polys();
        g += 1;
    }
    let tables: Vec<String> = a.groups.get(g).map_or_else(Vec::new, |group| {
        group
            .iter()
            .map(|&t| format!("{t} {}", frame.airs.air_refs()[t as usize].name()))
            .collect()
    });
    panic!(
        "phase A's roots differ first at root {first} of {}: group {g}, polynomial {at} \
         (tables {tables:?})",
        a.proof.roots.len()
    );
}
