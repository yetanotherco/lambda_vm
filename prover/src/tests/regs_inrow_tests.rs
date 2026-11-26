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
                word_memory: [0; 7],
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
fn assert_in_row_block_verifies(name: &str) {
    let elf = asm_elf_bytes(name);
    let proof = prove(&elf, &in_row());
    assert!(verifies(&proof, &elf, &in_row()), "{name}");
}

mod an_in_row_block_proves_and_verifies {
    #[test]
    fn all_instructions_64() {
        super::assert_in_row_block_verifies("all_instructions_64");
    }

    #[test]
    fn test_keccak_multi() {
        super::assert_in_row_block_verifies("test_keccak_multi");
    }

    #[test]
    fn test_commit_split() {
        super::assert_in_row_block_verifies("test_commit_split");
    }

    #[test]
    fn keep_alive_regs() {
        super::assert_in_row_block_verifies("keep_alive_regs");
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
        .map(|b| b.ecdas_form(format.ecdas))
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
/// verifies it with that check off. Cases on the same program share one
/// honest proof. Each forged build has BITWISE's `IS_HALF` counts recounted,
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
    let mut honest_controls = std::collections::BTreeMap::new();
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
        let (elf, honest_proof) = honest_controls
            .entry(program_of(forgery))
            .or_insert_with(|| {
                let elf = asm_elf_bytes(program_of(forgery));
                let (program, mut honest) = run(&elf, None);
                let proof = prove_built(&program, &elf, &mut honest, &format)
                    .expect("the honest build proves");
                (elf, proof)
            });
        assert!(
            verifies(honest_proof, elf, &format),
            "{forgery:?}: the honest build verifies"
        );
        let (program, mut forged) = run(elf, None);
        forge(forgery, &mut forged);
        let recounted = recount_is_half(&program, &mut forged);
        let proof =
            prove_built(&program, elf, &mut forged, &format).expect("the forged build proves");
        match refused_by(&proof, elf, &format, check) {
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

// ---- censuses of the block verifier's AIR sets under the in-row format ----------------------------------------------
//
// The one-path argument for registers takes two premises from the AIR set as a whole rather than
// from any one table's constraints: which tables touch the REG bus (and how), and that no table
// but the MEMW and MEMW_A relays holds a register token on the Memory bus. A third, that no MEMW
// requester names the PC's register x255, keeps a CPU row with `pc_double_read = 1` from having
// its PC token rewritten inside the `[ts − 3, ts]` gap. These censuses read every table kind of a
// block under each memory format, so a new table or a changed interaction fails here.

use crate::tables::fused_memory::MemoryFormat;
use stark::lookup::Multiplicity;

const MEMORY_FORMATS: [MemoryFormat; 3] = [
    MemoryFormat::Split,
    MemoryFormat::Fused,
    MemoryFormat::Words,
];

/// The block verifier's AIRs under the in-row register format and `memory`, with every table
/// kind present: two CPU tables, two residual register tables, two MEMW_A tables, one of every
/// other family, and the fused or word tables of `memory`.
fn every_table_airs(memory: MemoryFormat) -> crate::VmAirs {
    let elf = asm_elf_bytes("all_instructions_64");
    let program = executor::elf::Elf::load(&elf).expect("the ELF loads");
    let opts = ProofOptions::default_test_options();
    let counts = crate::TableCounts {
        cpu: 2,
        lt: 1,
        memw: 1,
        memw_aligned: 2,
        load: 1,
        mul: 1,
        dvrm: 1,
        shift: 1,
        branch: 1,
        memw_register: 2,
        eq: 1,
        bytewise: 1,
        store: 1,
        cpu32: 1,
        keccak: 1,
        keccak_rnd: 1,
        ecsm: 1,
        ecdas: 1,
        hint: 1,
        copy: 1,
        blake3: 1,
        is_wh: 1,
        ecrec: 1,
    };
    let pages = crate::tables::trace_builder::Traces::page_configs_from_elf_and_runtime(
        &program,
        &[],
        0,
        1 << 20,
    )
    .expect("pages");
    let inputs = crate::BlockAirInputs::new(&opts, &program, vec![1 << 10, 1 << 4], &[])
        .expect("the AIR inputs");
    let pinned = BlockFormat::pinned();
    assert_eq!(
        pinned.registers,
        RegisterFormat::InRow,
        "the pinned register format"
    );
    let mut airs = crate::block_airs(
        &program,
        &opts,
        &pages,
        &counts,
        &inputs,
        pinned.keccak_rnd,
        pinned.memw_aligned,
        RegisterFormat::InRow,
        pinned.ecdas,
    );
    match memory {
        MemoryFormat::Split => {}
        MemoryFormat::Fused => airs.with_fused_memory(&opts, [1, 1]),
        MemoryFormat::Words => airs
            .with_word_memory(&opts, [1; 7], &pages)
            .expect("the word tables"),
    }
    airs
}

/// A table's family: its AIR's name without the index (`CPU[1]`) or the page
/// base (`PAGE:0x0`).
fn family(name: &str) -> String {
    name.split(['[', ':']).next().unwrap_or("").to_string()
}

/// An affine form over the main columns: `(constant, [(column, coefficient)])`.
type Affine = (u64, Vec<(usize, u64)>);

/// `i`'s message elements (the bus id excluded) as affine forms, read off the real evaluator.
fn affine_elements(i: &BusInteraction) -> Vec<Affine> {
    use crate::tables::types::FE;
    let eval = |g: &dyn Fn(usize) -> FE| -> Vec<FE> {
        i.values.iter().flat_map(|v| v.combine_from(g)).collect()
    };
    let zero = eval(&|_| FE::zero());
    let mut out: Vec<Affine> = zero
        .iter()
        .map(|c| (c.canonical_u64(), Vec::new()))
        .collect();
    for c in i.columns_read() {
        let e = eval(&|j| if j == c { FE::one() } else { FE::zero() });
        for (k, v) in e.iter().enumerate() {
            let coefficient = *v - zero[k];
            if coefficient != FE::zero() {
                out[k].1.push((c, coefficient.canonical_u64()));
            }
        }
    }
    out
}

fn constant(c: u64) -> Affine {
    (c, Vec::new())
}

/// ★ The REG bus, over every table kind of an in-row block under each memory format: only the
/// CPU, the residual register table (MEMW_R's slot), REGISTER and HALT touch it, each with an
/// exact count of interactions; every tuple has four elements; every multiplicity is a bit its
/// AIR fixes (one, a CPU flag in the CPU's IS_BIT set, the residual's three-flag sum, REGISTER's
/// PC flag or its negation); and each node's consume and produce name one register at one
/// weight. A new table on the bus, or a weight that is not a bit, would let a register's tokens
/// leave its one path.
#[test]
fn only_the_cpu_the_residual_table_register_and_halt_touch_the_reg_bus() {
    use crate::tables::register::inrow::cols as reg_cols;
    use crate::tables::register_residual::cols as res;
    let reg = u64::from(BusId::Reg);
    for memory in MEMORY_FORMATS {
        let airs = every_table_airs(memory);
        // Per family: (senders, receivers) on the REG bus, over all its tables.
        let mut users: std::collections::BTreeMap<String, (usize, usize)> = Default::default();
        for air in airs.air_refs() {
            let name = air.name();
            let fam = family(name);
            let on_reg: Vec<&BusInteraction> = air
                .bus_interactions()
                .iter()
                .filter(|i| i.bus_id == reg)
                .collect();
            for i in &on_reg {
                let e = users.entry(fam.clone()).or_default();
                if i.is_sender {
                    e.0 += 1;
                } else {
                    e.1 += 1;
                }
                // `num_bus_elements` counts the bus id too.
                assert_eq!(
                    i.num_bus_elements(),
                    1 + 4,
                    "{memory:?}: {name}: a REG tuple"
                );
                let bit = match (&i.multiplicity, fam.as_str()) {
                    (Multiplicity::One, _) => true,
                    (Multiplicity::Column(c), "CPU") => {
                        crate::constraints::cpu::BIT_FLAG_COLUMNS.contains(c)
                    }
                    (Multiplicity::Sum3(a, b, c), "MEMW_R") => {
                        (*a, *b, *c) == (res::MU_READ, res::MU_WRITE, res::MU_KEEP)
                    }
                    (Multiplicity::Column(c) | Multiplicity::Negated(c), "REGISTER") => {
                        *c == reg_cols::IS_PC
                    }
                    _ => false,
                };
                assert!(
                    bit,
                    "{memory:?}: {name}: REG multiplicity {:?}",
                    i.multiplicity
                );
            }
            // A node is a consume followed by its produce: both name one register (the same
            // first element) at one weight.
            let mut k = 0;
            while k + 1 < on_reg.len() {
                if on_reg[k].is_sender && !on_reg[k + 1].is_sender {
                    assert_eq!(
                        affine_elements(on_reg[k])[0],
                        affine_elements(on_reg[k + 1])[0],
                        "{memory:?}: {name}: a node's two tokens name one register"
                    );
                    assert_eq!(
                        on_reg[k].multiplicity,
                        on_reg[k + 1].multiplicity,
                        "{memory:?}: {name}: a node's two tokens at one weight"
                    );
                    k += 2;
                } else {
                    k += 1;
                }
            }
        }
        println!("REG BUS USERS {memory:?}: {users:?}");
        let expected: std::collections::BTreeMap<String, (usize, usize)> = [
            // Two CPU tables: three slots and the PC edge, a consume and a produce each.
            ("CPU", (8, 8)),
            // Two residual tables: one consume and one produce each.
            ("HALT", (1, 1)),
            ("MEMW_R", (2, 2)),
            // Every register's initial token, and its final one (PC and the rest).
            ("REGISTER", (2, 1)),
        ]
        .into_iter()
        .map(|(f, n)| (f.to_string(), n))
        .collect();
        assert_eq!(users, expected, "{memory:?}: the REG bus's users");
    }
}

/// ★ The Memory bus, over every table kind of an in-row block under each memory format: its
/// users are pinned by family; every token's `is_register` element is the constant 0 except in
/// the MEMW and MEMW_A relays, whose element is their own `IS_REGISTER` column (taken from the
/// MEMW message they receive); no table holds a constant non-zero `is_register`; and PAGE's
/// first token, the only token a table produces without a requester, sits at the timestamp
/// `(0, 0)`. So a register token on the Memory bus can only be produced by a relay row that
/// consumed an earlier one: with REGISTER, the CPU and HALT off the bus, a register access routed
/// to a relay has no chain to start from.
#[test]
fn no_table_but_the_relays_holds_a_register_token_on_the_memory_bus() {
    let memory_bus = u64::from(BusId::Memory);
    for memory in MEMORY_FORMATS {
        let airs = every_table_airs(memory);
        let mut users: std::collections::BTreeSet<String> = Default::default();
        let mut page_produced = 0;
        for air in airs.air_refs() {
            let name = air.name();
            let fam = family(name);
            for i in air
                .bus_interactions()
                .iter()
                .filter(|i| i.bus_id == memory_bus)
            {
                if users.insert(fam.clone()) {
                    println!("MEMORY BUS USER {memory:?}: {name}");
                }
                let elements = affine_elements(i);
                let is_register = &elements[0];
                if fam == "MEMW" || fam == "MEMW_A" {
                    let column = match fam.as_str() {
                        "MEMW" => crate::tables::memw::cols::IS_REGISTER,
                        _ => crate::tables::memw_aligned::cols::IS_REGISTER,
                    };
                    assert_eq!(
                        *is_register,
                        (0, vec![(column, 1)]),
                        "{memory:?}: {name}: a relay's token carries its IS_REGISTER column"
                    );
                } else {
                    assert_eq!(
                        *is_register,
                        constant(0),
                        "{memory:?}: {name} holds a Memory token whose is_register is not 0"
                    );
                }
                // PAGE's produce (a receive on this bus) is the first token of an address.
                if fam == "PAGE" && !i.is_sender {
                    assert_eq!(
                        (&elements[3], &elements[4]),
                        (&constant(0), &constant(0)),
                        "{memory:?}: {name}: the first token's timestamp"
                    );
                    page_produced += 1;
                }
            }
        }
        println!("MEMORY BUS USERS {memory:?}: {users:?}");
        // Under the word memory format RAM is on its own word buses (PAGE_W mints there), so
        // the Memory bus keeps only the two relays and no table produces a first token on it.
        assert_eq!(
            page_produced > 0,
            memory != MemoryFormat::Words,
            "{memory:?}: PAGE produces first tokens"
        );
        let expected: &[&str] = match memory {
            MemoryFormat::Split => &["MEMW", "MEMW_A", "PAGE"],
            MemoryFormat::Fused => &["MEMW", "MEMW_A", "MEM_LOAD", "MEM_STORE", "PAGE"],
            MemoryFormat::Words => &["MEMW", "MEMW_A"],
        };
        assert_eq!(
            users.iter().map(String::as_str).collect::<Vec<_>>(),
            expected,
            "{memory:?}: the Memory bus's users"
        );
    }
}

/// ★ MEMW messages that address a register, over every table kind of an in-row block under each
/// memory format. No sender's `is_register` is column-valued (each is the constant 0 or 1). A
/// register message (`is_register` = 1) names its register with `base_address[1] = 0` and either
/// a constant `2·r` for r in 1..=31 or x254 (the commit buffer), or exactly one of the column
/// forms the senders are known to use: CPU32's `2·rs1`, `2·rs2`, `2·rd` (DECODE-bound, below 32)
/// and COPY's `20 + 488·IS_WRITE` (x10 or x254). No sender may name x255, the PC: a requester
/// write into the PC's `[ts − 3, ts]` gap on a row with `pc_double_read = 1` would rewrite it.
/// Every register message's width flags are the constants `(1, 0, 0)`. Receivers: MEMW_R takes
/// register messages at the constant `is_register` = 1; MEMW and MEMW_A take any at their
/// `IS_REGISTER` column; nobody else receives.
#[test]
fn every_register_message_names_x1_to_x31_or_x254_by_a_known_form() {
    use std::collections::{BTreeMap, BTreeSet};
    let memw = u64::from(BusId::Memw);
    for memory in MEMORY_FORMATS {
        let airs = every_table_airs(memory);
        let mut column_forms: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
        let mut constants: BTreeMap<String, BTreeSet<u64>> = BTreeMap::new();
        let mut free_is_register: Vec<String> = Vec::new();
        let mut receivers: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
        for air in airs.air_refs() {
            let fam = family(air.name());
            for i in air.bus_interactions().iter().filter(|i| i.bus_id == memw) {
                let n = i.num_bus_elements() - 1;
                // `is_register` is element 8 of a 24-element access, 0 of a 16-element write.
                let at = match n {
                    24 => 8,
                    16 => 0,
                    other => panic!("{memory:?}: {fam}: a MEMW message of {other} elements"),
                };
                let elements = affine_elements(i);
                let is_register = &elements[at];
                if !i.is_sender {
                    receivers
                        .entry(fam.clone())
                        .or_default()
                        .insert(format!("{is_register:?}"));
                    continue;
                }
                match is_register {
                    (0, t) if t.is_empty() => continue,
                    (1, t) if t.is_empty() => {}
                    other => {
                        free_is_register.push(format!("{fam}: {other:?}"));
                        continue;
                    }
                }
                let address = &elements[at + 1];
                assert_eq!(
                    elements[at + 2],
                    constant(0),
                    "{memory:?}: {fam}: a register message's base_address[1]"
                );
                for (k, want) in [(n - 3, 1u64), (n - 2, 0), (n - 1, 0)] {
                    assert_eq!(
                        elements[k],
                        constant(want),
                        "{memory:?}: {fam}: a register message's width flag {k}"
                    );
                }
                if address.1.is_empty() {
                    constants.entry(fam.clone()).or_default().insert(address.0);
                } else {
                    column_forms
                        .entry(fam.clone())
                        .or_default()
                        .insert(format!("{address:?}"));
                }
            }
        }
        println!("MEMW RECEIVERS {memory:?} (is_register forms): {receivers:?}");
        println!("MEMW REGISTER MESSAGES {memory:?}, constant addresses: {constants:?}");
        println!("MEMW REGISTER MESSAGES {memory:?}, column addresses: {column_forms:?}");
        assert!(
            free_is_register.is_empty(),
            "{memory:?}: a MEMW sender with a column-valued is_register: {free_is_register:?}"
        );
        for (fam, addresses) in &constants {
            for &a in addresses {
                assert!(
                    a % 2 == 0 && ((2..=62).contains(&a) || a == 2 * 254),
                    "{memory:?}: {fam} addresses register word {a}"
                );
            }
        }
        let copy_select = format!(
            "{:?}",
            (20u64, vec![(crate::tables::copy::cols::IS_WRITE, 488u64)])
        );
        let cpu32_fields: BTreeSet<String> = [
            crate::tables::cpu32::cols::RS1,
            crate::tables::cpu32::cols::RS2,
            crate::tables::cpu32::cols::RD,
        ]
        .iter()
        .map(|&c| format!("{:?}", (0u64, vec![(c, 2u64)])))
        .collect();
        // Under the word memory format COPY_W names x10 and x254 by constants.
        let mut expected: BTreeMap<String, BTreeSet<String>> =
            [("CPU32".to_string(), cpu32_fields)].into_iter().collect();
        if memory != MemoryFormat::Words {
            expected.insert("COPY".to_string(), [copy_select].into_iter().collect());
        }
        assert_eq!(
            column_forms, expected,
            "{memory:?}: the column-valued register addresses"
        );
        let relay = |column: usize| -> BTreeSet<String> {
            [format!("{:?}", (0u64, vec![(column, 1u64)]))]
                .into_iter()
                .collect()
        };
        let expected: BTreeMap<String, BTreeSet<String>> = [
            (
                "MEMW".to_string(),
                relay(crate::tables::memw::cols::IS_REGISTER),
            ),
            (
                "MEMW_A".to_string(),
                relay(crate::tables::memw_aligned::cols::IS_REGISTER),
            ),
            (
                "MEMW_R".to_string(),
                [format!("{:?}", constant(1))].into_iter().collect(),
            ),
        ]
        .into_iter()
        .collect();
        assert_eq!(receivers, expected, "{memory:?}: the MEMW receivers");
    }
}

/// ★ The wrap budget from the code's own constants: along any chain of REG-bus nodes a
/// register's timestamp advances by at most 2^16 a CPU slot (three a row, `IS_HALF[step − 1]`),
/// 4 a PC edge (one a row), and 2^32 a residual row; with at most `MAX_CPU_ROWS` CPU rows and
/// `MAX_ROWS_TOTAL` residual rows the total stays below p, so no chain closes on a timestamp it
/// passed. And the verifier's u128 sum of the residual tables' rows cannot overflow at the
/// frame's bounds.
#[test]
fn the_register_clock_cannot_wrap_the_field_at_the_codes_bounds() {
    let p: u128 = 0xFFFF_FFFF_0000_0001;
    let rows = u128::from(cpu::MAX_CPU_ROWS);
    let residual = u128::from(register_residual::MAX_ROWS_TOTAL);
    let total = 3 * rows * (1 << 16) + 4 * rows + residual * (1 << 32);
    println!(
        "REGISTER CLOCK BUDGET: 2^{:.3} against p = 2^{:.3}",
        (total as f64).log2(),
        (p as f64).log2()
    );
    assert!(
        1 + total < p,
        "the register clock's total advance stays below p"
    );
    let most = (block_whir::BLOCK_MAX_TABLES as u128) << block_whir::BLOCK_MAX_TABLE_VARS;
    assert!(most < u128::MAX / 2);
}
