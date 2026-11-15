//! The in-row register format (`tables::register::RegisterFormat::InRow`):
//! its AIRs' shapes, the verifier's cap on the residual table's rows, and blocks proved and
//! verified under it (the proving tests run where a block can be proved: the
//! box's lib run and CI). The inner buses' balance on the builders' traces is
//! `tables::trace_builder::bus_net_tests`'.

use crate::block_whir::{
    self, BlockFormat, Deviations, check_residual_rows, prove_block_whir_with, verify_block_whir,
};
use crate::tables::register::RegisterFormat;
use crate::tables::types::BusId;
use crate::tables::{MaxRowsConfig, cpu, halt, register_residual};
use crate::test_utils::asm_elf_bytes;
use multilinear::whir_chain::GkrPack;
use stark::lookup::BusInteraction;
use stark::multilinear_table::batched::{ArgueShape, VerifierChecks, chunking};
use stark::proof::options::ProofOptions;

/// The GKR input slots a table of `2^num_vars` rows and these interactions
/// takes under the production packing.
fn slots(interactions: &[BusInteraction], num_vars: usize) -> usize {
    let n = interactions.len();
    let shape = ArgueShape {
        num_vars,
        input_vars: multilinear::logup::input_layer_vars(n, num_vars),
        degree: 2,
        shifted: false,
        interactions: n,
    };
    let c = chunking(&shape, GkrPack::PRODUCTION);
    c.count << c.slot_vars
}

#[test]
fn the_in_row_cpu_row_takes_todays_gkr_slots_and_reads_its_columns() {
    let today = cpu::bus_interactions(0);
    let in_row = cpu::bus_interactions_inrow(0);
    assert_eq!((today.len(), in_row.len()), (20, 24));
    for num_vars in [21, 20, 10] {
        assert_eq!(slots(&in_row, num_vars), slots(&today, num_vars));
    }
    assert_eq!(slots(&in_row, 21), 24);
    let reg = u64::from(BusId::Reg);
    assert_eq!(in_row.iter().filter(|i| i.bus_id == reg).count(), 8);
    assert!(
        in_row
            .iter()
            .all(|i| i.bus_id != u64::from(BusId::Memw) && i.bus_id != u64::from(BusId::Memory)),
        "no MEMW message and no Memory token is left in the row"
    );
    let width = cpu::inrow_cols::NUM_COLUMNS_INROW;
    assert_eq!(width, 42);
    let read: std::collections::BTreeSet<usize> = in_row
        .iter()
        .flat_map(BusInteraction::columns_read)
        .collect();
    assert!((cpu::cols::NUM_COLUMNS..width).all(|c| read.contains(&c)));
    assert!(read.iter().all(|&c| c < width));
}

#[test]
fn the_residual_table_halt_and_register_take_the_reg_bus() {
    let reg = u64::from(BusId::Reg);
    let residual = register_residual::bus_interactions();
    assert_eq!(residual.len(), 6);
    assert_eq!(slots(&residual, 21), 8);
    assert_eq!(residual.iter().filter(|i| i.bus_id == reg).count(), 2);
    let halt = halt::bus_interactions_inrow();
    assert_eq!(halt.len(), 1 + 31 + 2);
    assert_eq!(halt.iter().filter(|i| i.bus_id == reg).count(), 2);
    assert!(halt.iter().all(|i| i.bus_id != u64::from(BusId::Memory)));
    let register = crate::tables::register::inrow::bus_interactions(9);
    assert!(register.iter().all(|i| i.bus_id == reg));
    let files = crate::tables::register::halted_fini(7);
    let columns = crate::tables::register::inrow::preprocessed_columns(&files, &files);
    assert_eq!(
        columns.len(),
        crate::tables::register::inrow::NUM_PREPROCESSED_COLS
    );
    assert!(columns.iter().all(|c| c.len() == 64));
}

/// The verifier refuses a statement whose residual register tables hold more than
/// `register_residual::MAX_ROWS_TOTAL` rows, and takes one at the cap.
#[test]
fn the_verifier_caps_the_residual_tables_rows() {
    assert_eq!(register_residual::MAX_ROWS_TOTAL, 1 << 28);
    assert!(check_residual_rows(std::iter::repeat_n(21u8, 128)).is_ok());
    assert!(check_residual_rows(std::iter::repeat_n(21u8, 129)).is_err());
    assert!(check_residual_rows([28u8]).is_ok());
    assert!(check_residual_rows([28u8, 2]).is_err());
}

/// The block verifier's statement checks cap the residual register tables'
/// rows: a statement whose three residual tables (MEMW_R's slot under the
/// in-row register format) state 2^27 + 2^27 + 2^2 rows is refused by that
/// cap, and one at 2^27 + 2^26 + 2^26 rows frames. The tables are found by
/// their exact names, so the frame's own selection of them is what is read:
/// a selection that missed one would pass the refused statement, and one that
/// took any other table would refuse the statement at the cap.
#[test]
fn the_block_frame_caps_the_rows_its_residual_register_tables_state() {
    use crate::tables::{bitwise, register_residual::cols as residual_cols};
    let elf = asm_elf_bytes("all_instructions_64");
    let program = executor::elf::Elf::load(&elf).expect("the ELF loads");
    let opts = ProofOptions::default_test_options();
    let counts = crate::TableCounts {
        cpu: 1,
        memw_register: 3,
        lt: 0,
        memw: 0,
        memw_aligned: 0,
        load: 0,
        mul: 0,
        dvrm: 0,
        shift: 0,
        branch: 0,
        eq: 0,
        bytewise: 0,
        store: 0,
        cpu32: 0,
        keccak: 0,
        keccak_rnd: 0,
        ecsm: 0,
        ecdas: 0,
        hint: 0,
        copy: 0,
        blake3: 0,
        is_wh: 0,
        ecrec: 0,
    };
    let pages = crate::tables::trace_builder::Traces::page_configs_from_elf_and_runtime(
        &program,
        &[],
        0,
        usize::MAX,
    )
    .expect("the program's pages")
    .len();
    let n = crate::FIXED_TABLE_COUNT + counts.total().expect("fits") + pages;
    let groups: Vec<Vec<u32>> = (0..n as u32).map(|i| vec![i]).collect();
    let mut heights = vec![10u8; n];
    heights[block_whir::BITWISE_TABLE] = bitwise::NUM_VARS as u8;
    let format = in_row();
    let frame = |heights: &[u8]| {
        block_whir::block_frame(
            block_whir::BlockStatement {
                table_num_vars: heights,
                runtime_page_ranges: &[],
                table_counts: &counts,
                public_output: &[],
                num_private_input_pages: 0,
                groups: &groups,
                // The split memory format (`in_row()`): no fused tables.
                fused_memory: [0, 0],
            },
            &elf,
            &opts,
            &format,
        )
    };
    let honest = frame(&heights).expect("the small statement frames");
    let residual: Vec<usize> = (0..counts.memw_register)
        .map(|k| {
            let refs = honest.airs.air_refs();
            let at = refs
                .iter()
                .position(|air| air.name() == format!("MEMW_R[{k}]"))
                .expect("a residual table in MEMW_R's slot");
            assert_eq!(
                refs[at].trace_layout().0,
                residual_cols::NUM_COLUMNS,
                "MEMW_R[{k}] is the residual register table"
            );
            at
        })
        .collect();
    let stated = |vars: [u8; 3]| {
        let mut h = heights.clone();
        for (&at, v) in residual.iter().zip(vars) {
            h[at] = v;
        }
        h
    };
    let at_cap = frame(&stated([27, 26, 26]));
    assert!(
        at_cap.is_ok(),
        "2^28 residual rows, the cap, frame: {:?}",
        at_cap.err()
    );
    let over = frame(&stated([27, 27, 2]))
        .err()
        .expect("2^28 + 4 residual rows are refused");
    let message = over.to_string();
    assert!(
        message.contains(&format!(
            "the residual register table states {} rows",
            (1u64 << 28) + 4
        )),
        "refused by the residual cap: {message}"
    );
}

/// The production format under the in-row register format.
fn in_row() -> BlockFormat {
    BlockFormat {
        registers: RegisterFormat::InRow,
        memory: crate::tables::fused_memory::MemoryFormat::Split,
        ..BlockFormat::production()
    }
}

fn prove(elf: &[u8], format: &BlockFormat) -> block_whir::BlockWhirProof {
    let options = super::block_whir_tests::streamed(MaxRowsConfig::small(), 3, 9);
    prove_block_whir_with(
        elf,
        &[],
        &ProofOptions::default_test_options(),
        format,
        &options,
        &Deviations::default(),
    )
    .expect("prove")
    .0
}

fn verifies(proof: &block_whir::BlockWhirProof, elf: &[u8], format: &BlockFormat) -> bool {
    verify_block_whir(proof, elf, &ProofOptions::default_test_options(), format).unwrap_or(false)
}

/// Blocks proved under the in-row register format verify: every instruction,
/// KECCAK's chips (register reads through the residual register table), the commit syscall (x254),
/// and registers idle past a CPU slot's reach (keep-alives).
#[test]
fn an_in_row_block_proves_and_verifies() {
    for name in [
        "all_instructions_64",
        "test_keccak_multi",
        "test_commit_split",
        "keep_alive_regs",
    ] {
        let elf = asm_elf_bytes(name);
        let proof = prove(&elf, &in_row());
        assert!(verifies(&proof, &elf, &in_row()), "{name}");
    }
}

/// Each register format refuses the other's proof: the format is the
/// verifier's constant, never the proof's.
#[test]
fn each_register_format_refuses_the_others_proof() {
    let elf = asm_elf_bytes("keep_alive_regs");
    let today = BlockFormat {
        registers: RegisterFormat::Tokens32,
        memory: crate::tables::fused_memory::MemoryFormat::Split,
        ..BlockFormat::production()
    };
    let in_row_proof = prove(&elf, &in_row());
    let today_proof = prove(&elf, &today);
    assert!(verifies(&in_row_proof, &elf, &in_row()));
    assert!(verifies(&today_proof, &elf, &today));
    assert!(!verifies(&in_row_proof, &elf, &today));
    assert!(!verifies(&today_proof, &elf, &in_row()));
}

/// The in-row build of `logs` as the block prover's windowed build makes it
/// under `format` (its MEMW_A and KECCAK_RND forms), at the small chunks.
fn in_row_build_as(
    program: &executor::elf::Elf,
    logs: &[executor::vm::logs::Log],
    format: &BlockFormat,
) -> crate::tables::trace_builder::Traces {
    use crate::tables::trace_builder::WindowedTraceBuilder;
    let mut builder = WindowedTraceBuilder::new(program, &[], &MaxRowsConfig::small())
        .and_then(|b| b.register_format(RegisterFormat::InRow))
        .and_then(|b| b.memw_aligned_order(format.memw_aligned))
        .map(|b| b.keccak_rnd_form(format.keccak_rnd))
        .expect("the builder");
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
    traces
}

/// `traces` proved as a block under `format`: the prover takes any trace.
fn prove_built(
    program: &executor::elf::Elf,
    elf: &[u8],
    traces: &mut crate::tables::trace_builder::Traces,
    format: &BlockFormat,
) -> Result<block_whir::BlockWhirProof, crate::Error> {
    prove_built_with(program, elf, traces, format, &Deviations::default())
}

/// [`prove_built`] under the prover's `deviations`.
fn prove_built_with(
    program: &executor::elf::Elf,
    elf: &[u8],
    traces: &mut crate::tables::trace_builder::Traces,
    format: &BlockFormat,
    deviations: &Deviations,
) -> Result<block_whir::BlockWhirProof, crate::Error> {
    let options = super::block_whir_tests::streamed(MaxRowsConfig::small(), 3, 9);
    let mut stamps = block_whir::BlockStamps::default();
    block_whir::prove_traces(
        program,
        elf,
        traces,
        &ProofOptions::default_test_options(),
        format,
        &options,
        deviations,
        true,
        &|_, _| {},
        &mut stamps,
    )
}

/// The verifier's check a forged block is meant to fail.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Check {
    /// The buses' balance.
    Balance,
    /// The constraint sumcheck's final equation.
    Constraint,
    /// The preprocessed columns' openings.
    Preprocessed,
}

impl Check {
    /// The error the multilinear verifier refuses with at this check.
    fn error(self) -> multilinear::Error {
        match self {
            Check::Balance => multilinear::Error::BusImbalance,
            Check::Constraint => multilinear::Error::BatchMismatch,
            Check::Preprocessed => multilinear::Error::EvaluationMismatch,
        }
    }

    /// Every check but this one.
    fn off(self) -> VerifierChecks {
        let all = VerifierChecks::ALL;
        match self {
            Check::Balance => VerifierChecks {
                balance: false,
                ..all
            },
            Check::Constraint => VerifierChecks {
                constraint: false,
                ..all
            },
            Check::Preprocessed => VerifierChecks {
                preprocessed: false,
                ..all
            },
        }
    }
}

/// How the verifier reads a forged block: with every check, and with every
/// check but `check`. `None` when it is refused by `check` alone (that check's
/// error with every check on, and verified with it off: the forgery is whole
/// save for what `check` reads); otherwise both verdicts, to report.
fn refused_by(
    proof: &block_whir::BlockWhirProof,
    elf: &[u8],
    format: &BlockFormat,
    check: Check,
) -> Option<String> {
    let verdict = |checks: VerifierChecks| {
        block_whir::verify_block_whir_traced(
            proof,
            elf,
            &ProofOptions::default_test_options(),
            format,
            false,
            checks,
        )
        .map_err(|e| format!("{e:?}"))
    };
    let all = verdict(VerifierChecks::ALL);
    let off = verdict(check.off());
    let named = matches!(
        &all,
        Ok(Err(block_whir::BlockRefusal::Proof(e))) if *e == check.error()
    );
    let whole = matches!(&off, Ok(Ok(())));
    (!(named && whole)).then(|| format!("every check: {all:?} · without {check:?}: {off:?}"))
}

/// ★ Box: the in-row forgeries proved (`tables::trace_builder::
/// in_row_forgery_tests`, which reads each on the buses): every forged build
/// proves, and the block verifier refuses it at the check it targets, and
/// verifies it with that check off; the honest build of each program proves
/// and verifies. Each forged build has BITWISE's `IS_HALF` counts recounted,
/// as its prover would, so a stale count cannot be what refuses it. With the
/// PC chain run past the exit call (HALT's pc `(1, 0)`) among them, against
/// the honest exit 0 of the program without its first exit.
#[test]
fn in_row_forgeries_are_refused_by_the_block_verifier() {
    use crate::tables::trace_builder::in_row_forgery_tests::{
        Forgery, forge, program_of, recount_is_half, with_verifier_files,
    };
    use crate::tests::halt_pc_forgery_poc::{
        FORGED_ROWS, GUEST, forged_logs, patched_elf, run_logs,
    };
    let format = in_row();
    let run = |elf: &[u8], logs: Option<Vec<executor::vm::logs::Log>>| {
        let program = executor::elf::Elf::load(elf).expect("the ELF loads");
        let logs = logs.unwrap_or_else(|| run_logs(elf));
        let mut traces = in_row_build_as(&program, &logs, &format);
        with_verifier_files(&program, &mut traces);
        (program, traces)
    };
    let mut wrong = Vec::new();
    let mut refused = 0;
    for forgery in [
        Forgery::SlotSelfLoop,
        Forgery::SlotCycle,
        Forgery::SlotPastReach,
        Forgery::ResidualSelfLoop,
        Forgery::KeepAliveChangesValue,
        Forgery::WrongOldRd,
        Forgery::KeepAliveInventsValue,
        Forgery::ReadWithoutRequester,
    ] {
        let check = match forgery {
            Forgery::KeepAliveChangesValue => Check::Constraint,
            _ => Check::Balance,
        };
        let elf = asm_elf_bytes(program_of(forgery));
        let (program, mut honest) = run(&elf, None);
        let proof =
            prove_built(&program, &elf, &mut honest, &format).expect("the honest build proves");
        assert!(
            verifies(&proof, &elf, &format),
            "{forgery:?}: the honest build verifies"
        );
        let (program, mut forged) = run(&elf, None);
        forge(forgery, &mut forged);
        let recounted = recount_is_half(&program, &mut forged);
        let proof =
            prove_built(&program, &elf, &mut forged, &format).expect("the forged build proves");
        match refused_by(&proof, &elf, &format, check) {
            None => refused += 1,
            Some(read) => wrong.push(format!("{forgery:?}: {read}")),
        }
        println!("IN-ROW FORGERY {forgery:?}: {recounted} IS_HALF value(s) recounted, {check:?}");
    }
    // The PC chain past the exit call.
    let patched = patched_elf(&asm_elf_bytes(GUEST));
    let (program, mut honest) = run(&patched, None);
    let proof =
        prove_built(&program, &patched, &mut honest, &format).expect("the honest exit proves");
    assert!(
        verifies(&proof, &patched, &format),
        "the patched program's exit 0 verifies"
    );
    let elf = asm_elf_bytes(GUEST);
    let (program, mut forged) = run(&elf, Some(forged_logs(&elf, FORGED_ROWS)));
    crate::tables::types::VmTable::set_dword_wl(
        &mut forged.halt.main_table,
        0,
        halt::cols::PC_0,
        1,
    );
    with_verifier_files(&program, &mut forged);
    let recounted = recount_is_half(&program, &mut forged);
    let proof = prove_built(&program, &elf, &mut forged, &format).expect("the forged run proves");
    match refused_by(&proof, &elf, &format, Check::Balance) {
        None => refused += 1,
        Some(read) => wrong.push(format!("the PC chain past the exit: {read}")),
    }
    println!("IN-ROW FORGERY PC chain past the exit: {recounted} IS_HALF value(s) recounted");
    assert!(
        wrong.is_empty(),
        "forgeries not refused by their check alone:\n{}",
        wrong.join("\n")
    );
    println!("IN-ROW FORGERIES REFUSED: {refused}, each by its check alone");
}

/// ★ Box: REGISTER's files are the verifier's under the in-row register
/// format. A block whose register that the run never touches starts and ends
/// at another value (its initial and final words edited alike, so every bus
/// balances), proved without the prover's own column check, is refused at
/// REGISTER's preprocessed openings, and verifies with that check off: were
/// REGISTER's columns not the verifier's, the files would be the prover's.
#[test]
fn an_in_row_block_with_other_register_files_is_refused_at_its_preprocessed_openings() {
    use crate::tables::register::inrow;
    use crate::tables::trace_builder::in_row_forgery_tests::with_verifier_files;
    use crate::tests::halt_pc_forgery_poc::run_logs;
    let format = in_row();
    let elf = asm_elf_bytes("keep_alive_regs");
    let program = executor::elf::Elf::load(&elf).expect("the ELF loads");
    let mut traces = in_row_build_as(&program, &run_logs(&elf), &format);
    with_verifier_files(&program, &mut traces);
    let table = &mut traces.register;
    use crate::tables::types::{GoldilocksExtension, GoldilocksField};
    let cell = |t: &stark::trace::TraceTable<GoldilocksField, GoldilocksExtension>,
                r: usize,
                c: usize| t.get_main(r, c).canonical_u64();
    let row = (0..crate::tables::register::NUM_REGISTERS)
        .find(|&r| cell(table, r, inrow::cols::IS_PC) == 0 && cell(table, r, inrow::cols::TS) == 1)
        .expect("a register the run never touches");
    assert_eq!(
        cell(table, row, inrow::cols::INIT_0),
        cell(table, row, inrow::cols::FINI_0),
        "an untouched register ends where it starts"
    );
    let other = crate::tables::types::FE::from(cell(table, row, inrow::cols::INIT_0) + 1);
    table.main_table.set(row, inrow::cols::INIT_0, other);
    table.main_table.set(row, inrow::cols::FINI_0, other);
    let proof = prove_built_with(
        &program,
        &elf,
        &mut traces,
        &format,
        &Deviations {
            unchecked_preprocessed: true,
            ..Default::default()
        },
    )
    .expect("the edited build proves");
    let read = refused_by(&proof, &elf, &format, Check::Preprocessed);
    assert!(
        read.is_none(),
        "register x{}'s files edited: {}",
        inrow::register_of_row(row),
        read.unwrap_or_default()
    );
    println!(
        "IN-ROW REGISTER FILES: x{} at {other:?} refused at REGISTER's preprocessed openings alone",
        inrow::register_of_row(row)
    );
}
