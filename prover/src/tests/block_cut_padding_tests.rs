//! The block proved with its cut tables stopped at the last real row
//! (`CutPadding::LastTable`, `LAMBDA_VM_BLOCK_CUT_PADDING=last`): the verifier,
//! unchanged, accepts the block without its tables of padding rows alone, and
//! refuses a block one table short of the real rows on the bus balance, which
//! is its only objection. The streamed prove counts the tables of the rule
//! the process runs under, and phase B builds every one it dropped again.

use executor::elf::Elf;
use executor::vm::execution::Executor;
use multilinear::whir_chain::{ArgueFormat, StackVars};
use stark::multilinear_table::batched::VerifierChecks;
use stark::proof::options::ProofOptions;

use crate::block_whir::{
    self, BlockFormat, BlockOptions, BlockRefusal, BlockWhirProof, Deviations,
    prove_block_whir_with, verify_block_whir, verify_block_whir_traced,
};
use crate::tables::MaxRowsConfig;
use crate::tables::trace_builder::{CutPadding, Traces, WindowedTraceBuilder};
use crate::test_utils::asm_elf_bytes;
use crate::zf_format::ZfFormat;

/// A stack of 2^10 and two polynomials a group: a small program spans several
/// groups.
fn many_groups(argue: ArgueFormat) -> BlockFormat {
    let mut zf = ZfFormat::DEFAULT;
    zf.whir_stack = StackVars::new(10).expect("a valid stack");
    BlockFormat {
        zf,
        group_polys: 2,
        max_groups: block_whir::BLOCK_MAX_GROUPS,
        prepared: true,
        argue,
        group: multilinear::whir_chain::GroupFold::PerPoly,
    }
}

/// The production options at a test's size: chunked tables of `max_rows`,
/// KECCAK_RND cut at 2^5 rows, KECCAK and ECSM at one row, ECDAS at 2^4, so
/// each cut family of a small program has a table of padding rows alone with
/// the whole run padded.
fn options(
    max_rows: MaxRowsConfig,
    window_log2: Option<usize>,
    layout_workers: usize,
) -> BlockOptions {
    BlockOptions {
        max_rows,
        keccak_rnd_rows_log2: 5,
        keccak_rows_log2: 0,
        ecsm_rows_log2: 0,
        ecdas_rows_log2: 4,
        drop_levels: multilinear::whir_commit::TreeDrop::uniform(3),
        window_log2,
        drop_streamed_ops: layout_workers > 0,
        layout_workers,
        layout_ahead: Some(2),
        narrow: stark::multilinear_block::Narrowing::Card { min_cells: 0 },
        memlog: false,
        rest_layout_bytes: Some(1 << 20),
        spill: block_whir::BlockSpillPolicy::Off,
        regen: None,
        ..BlockOptions::production()
    }
}

/// `program` run, and its traces as the block's windowed build makes them at
/// [`options`]' cuts, the cut tables padded as `padding`, held wide.
fn traces(program: &str, padding: CutPadding) -> (Elf, Vec<u8>, Traces) {
    let elf = asm_elf_bytes(program);
    let program = Elf::load(&elf).expect("the ELF loads");
    let logs = Executor::new(&program, Vec::new())
        .expect("the executor starts")
        .run()
        .expect("the program runs")
        .logs;
    let o = options(MaxRowsConfig::default(), None, 0);
    let mut builder = WindowedTraceBuilder::new(&program, &[], &o.max_rows)
        .expect("the builder")
        .keccak_rnd_chunks_at_finish(1 << o.keccak_rnd_rows_log2)
        .expect("rows")
        .cuts_at_finish(
            1 << o.keccak_rows_log2,
            1 << o.ecsm_rows_log2,
            1 << o.ecdas_rows_log2,
        )
        .expect("rows")
        .cut_padding(padding);
    let window = 1000;
    let body = logs.len() - 1;
    let cut = body - body % window;
    let mut chunks = Vec::new();
    for w in logs[..cut].chunks(window) {
        chunks.extend(builder.push(w).expect("a window"));
    }
    let mut traces = builder.finish(&logs[cut..]).expect("the last window");
    traces
        .insert_streamed(chunks)
        .expect("every chunk has a placeholder");
    (program, elf, traces)
}

/// The block over `traces`.
fn prove(program: &Elf, elf: &[u8], traces: &mut Traces, format: &BlockFormat) -> BlockWhirProof {
    block_whir::prove_traces(
        program,
        elf,
        traces,
        &ProofOptions::default_test_options(),
        format,
        &options(MaxRowsConfig::default(), None, 0),
        &Deviations::default(),
        false,
        &|_, _| {},
        &mut Default::default(),
    )
    .expect("prove")
}

/// The four cut families' counts in a statement: KECCAK, KECCAK_RND, ECSM,
/// ECDAS.
fn cut_counts(proof: &BlockWhirProof) -> [usize; 4] {
    let c = &proof.table_counts;
    [c.keccak, c.keccak_rnd, c.ecsm, c.ecdas]
}

/// ★ The verifier, as it is, accepts a block without its tables of padding
/// rows alone. test_keccak_multi (3 permutations: 3 KECCAK rows, 72 KECCAK_RND
/// rows), per table and batched, and test_ecsm_multi (3 scalar
/// multiplications: 3 ECSM rows, 42 ECDAS steps), batched, each proved with
/// the whole run padded and cut at the last real row: both verify, and the
/// second states one table fewer of each cut family the program has.
#[test]
fn a_block_without_its_padding_tables_proves_and_verifies() {
    let opts = ProofOptions::default_test_options();
    for (program, argue, whole_run, last_table) in [
        (
            "test_keccak_multi",
            ArgueFormat::PerTable,
            [4, 4, 0, 0],
            [3, 3, 0, 0],
        ),
        (
            "test_keccak_multi",
            ArgueFormat::BATCHED,
            [4, 4, 0, 0],
            [3, 3, 0, 0],
        ),
        (
            "test_ecsm_multi",
            ArgueFormat::BATCHED,
            [0, 0, 4, 4],
            [0, 0, 3, 3],
        ),
    ] {
        let format = many_groups(argue);
        let what = format!("{program} {argue:?}");
        let mut tables = Vec::new();
        for (padding, counts) in [
            (CutPadding::WholeRun, whole_run),
            (CutPadding::LastTable, last_table),
        ] {
            let (elf_program, elf, mut traces) = traces(program, padding);
            let proof = prove(&elf_program, &elf, &mut traces, &format);
            assert_eq!(cut_counts(&proof), counts, "{what} {padding:?}: the counts");
            assert_eq!(
                verify_block_whir(&proof, &elf, &opts, &format).map_err(|e| format!("{e:?}")),
                Ok(true),
                "{what} {padding:?}"
            );
            tables.push(proof.table_num_vars.len());
        }
        assert_eq!(
            tables[0] - tables[1],
            2,
            "{what}: two tables of padding rows alone left out"
        );
        println!(
            "BLOCK CUT PADDING ACCEPTED: {what}: {} tables with the whole run padded, {} cut at \
             the last real row, both verified",
            tables[0], tables[1]
        );
    }
}

/// ★ One table short of the real rows is refused, and on the bus balance
/// alone: for each cut family (KECCAK_RND per table and batched, the other
/// three batched), the block cut at the last real row with that family's last
/// table — which holds real rows — left out of the traces, the counts and the
/// heights. The verifier's refusal is `BusImbalance`; with the balance check
/// off (the mutation) it accepts the same proof, so no other check objects to
/// the missing table.
#[test]
fn a_block_one_table_short_of_the_real_rows_is_refused_on_the_bus() {
    let opts = ProofOptions::default_test_options();
    type Family = fn(
        &mut Traces,
    ) -> &mut Vec<
        stark::trace::TraceTable<crate::test_utils::F, crate::test_utils::E>,
    >;
    let families: [(&str, &str, ArgueFormat, Family); 5] = [
        (
            "test_keccak_multi",
            "KECCAK_RND",
            ArgueFormat::PerTable,
            |t| &mut t.keccak_rnds,
        ),
        (
            "test_keccak_multi",
            "KECCAK_RND",
            ArgueFormat::BATCHED,
            |t| &mut t.keccak_rnds,
        ),
        ("test_keccak_multi", "KECCAK", ArgueFormat::BATCHED, |t| {
            &mut t.keccaks
        }),
        ("test_ecsm_multi", "ECSM", ArgueFormat::BATCHED, |t| {
            &mut t.ecsms
        }),
        ("test_ecsm_multi", "ECDAS", ArgueFormat::BATCHED, |t| {
            &mut t.ecdases
        }),
    ];
    for (program, family, argue, tables) in families {
        let format = many_groups(argue);
        let what = format!("{program} {family} {argue:?}");
        let (elf_program, elf, mut traces) = traces(program, CutPadding::LastTable);
        assert_eq!(
            tables(&mut traces).len(),
            3,
            "{what}: tables with a real row"
        );
        tables(&mut traces).pop();
        let short = prove(&elf_program, &elf, &mut traces, &format);
        let refusal =
            verify_block_whir_traced(&short, &elf, &opts, &format, false, VerifierChecks::ALL)
                .unwrap_or_else(|e| panic!("{what}: the statement did not frame: {e:?}"));
        assert!(
            matches!(
                refusal,
                Err(BlockRefusal::Proof(multilinear::Error::BusImbalance))
            ),
            "{what}: refused for another reason, or accepted: {refusal:?}"
        );
        let unbalanced = verify_block_whir_traced(
            &short,
            &elf,
            &opts,
            &format,
            false,
            VerifierChecks {
                balance: false,
                ..VerifierChecks::ALL
            },
        )
        .unwrap_or_else(|e| panic!("{what}: the statement did not frame: {e:?}"));
        assert!(
            unbalanced.is_ok(),
            "{what}: with the balance check off another check refused: {unbalanced:?}"
        );
        println!(
            "BLOCK CUT PADDING REFUSED: {what}: one table short of the real rows, refused as \
             BusImbalance; accepted with the balance check off"
        );
    }
}

/// ★ The streamed prove, as the block runs it, counts the tables of the rule
/// the process runs under (`LAMBDA_VM_BLOCK_CUT_PADDING`): test_keccak_multi's
/// KECCAK_RND is four tables of 2^5 rows with the whole run padded and three
/// cut at the last real row (KECCAK, ECSM and ECDAS at their production cuts:
/// one KECCAK table). Proved with the finish monolithic, with the finish
/// streamed, and with live regeneration dropping every table it can — which
/// then builds each dropped KECCAK_RND table again with no mismatch — and
/// verified each time. Run once per value of the variable.
#[test]
fn a_streamed_block_counts_the_tables_of_the_rule_in_effect() {
    let padding = CutPadding::from_env().expect("the variable reads");
    let tables = match padding {
        CutPadding::WholeRun => 4,
        CutPadding::LastTable => 3,
    };
    let elf = asm_elf_bytes("test_keccak_multi");
    let opts = ProofOptions::default_test_options();
    let format = many_groups(ArgueFormat::BATCHED);
    for (what, layout_workers, regen) in [
        ("the finish monolithic", 0, None),
        ("the finish streamed", 3, None),
        (
            "live regeneration",
            3,
            Some(block_whir::regen::RegenMode::Always),
        ),
    ] {
        let o = BlockOptions {
            keccak_rows_log2: block_whir::BLOCK_KECCAK_ROWS_LOG2,
            ecsm_rows_log2: block_whir::BLOCK_ECSM_ROWS_LOG2,
            ecdas_rows_log2: block_whir::BLOCK_ECDAS_ROWS_LOG2,
            regen,
            ..options(MaxRowsConfig::small(), Some(3), layout_workers)
        };
        let (proof, stamps) =
            prove_block_whir_with(&elf, &[], &opts, &format, &o, &Deviations::default())
                .unwrap_or_else(|e| panic!("{padding:?}, {what}: {e:?}"));
        assert_eq!(
            cut_counts(&proof),
            [1, tables, 0, 0],
            "{padding:?}, {what}: the counts"
        );
        assert_eq!(
            verify_block_whir(&proof, &elf, &opts, &format).map_err(|e| format!("{e:?}")),
            Ok(true),
            "{padding:?}, {what}"
        );
        if regen.is_some() {
            let r = stamps.regen.expect("live regeneration's readout");
            let lines = r.lines.join("\n");
            assert!(r.rest_dropped > 0, "no table of the rest dropped\n{lines}");
            assert_eq!(r.regenerated, r.dropped, "every table back\n{lines}");
            assert_eq!(r.rest_regenerated, r.rest_dropped, "{lines}");
            assert_eq!((r.mismatches, r.failed), (0, 0), "{lines}");
            assert_eq!(r.error, None, "{lines}");
            // KECCAK_RND's tables built again: every one phase A dropped, and
            // none the rule does not build.
            let again: usize = r
                .lines
                .iter()
                .find_map(|line| line.strip_prefix("BLOCK REGEN rest: "))
                .and_then(|rest| rest.split("· KECCAK_RND ").nth(1))
                .and_then(|count| count.split(' ').next())
                .and_then(|n| n.parse().ok())
                .unwrap_or_else(|| panic!("no KECCAK_RND count in the rest's line\n{lines}"));
            assert!(
                (1..=tables).contains(&again),
                "{again} KECCAK_RND tables built again of {tables}\n{lines}"
            );
            println!(
                "BLOCK CUT PADDING REBUILT: {padding:?}: {again} of {tables} KECCAK_RND tables \
                 dropped and built again, {} of the rest in all, 0 mismatches",
                r.rest_regenerated
            );
        }
        println!("BLOCK CUT PADDING STREAMED: {padding:?}, {what}: KECCAK_RND {tables}, verified");
    }
}
