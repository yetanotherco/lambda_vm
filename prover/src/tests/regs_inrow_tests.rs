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
use stark::multilinear_table::batched::{ArgueShape, chunking};
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

/// The production format under the in-row register format.
fn in_row() -> BlockFormat {
    BlockFormat {
        registers: RegisterFormat::InRow,
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
    let options = super::block_whir_tests::streamed(MaxRowsConfig::small(), 3, 9);
    let mut stamps = block_whir::BlockStamps::default();
    block_whir::prove_traces(
        program,
        elf,
        traces,
        &ProofOptions::default_test_options(),
        format,
        &options,
        &Deviations::default(),
        true,
        &|_, _| {},
        &mut stamps,
    )
}

/// ★ Box: the in-row forgeries proved (`tables::trace_builder::
/// in_row_forgery_tests`, which reads each on the buses): every forged build
/// proves, and the block verifier refuses it; the honest build of each
/// program proves and verifies. With the PC chain run past the exit call
/// (HALT's pc `(1, 0)`) among them, against the honest exit 0 of the program
/// without its first exit.
#[test]
fn in_row_forgeries_are_refused_by_the_block_verifier() {
    use crate::tables::trace_builder::in_row_forgery_tests::{
        Forgery, forge, program_of, with_verifier_files,
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
    let mut refused = Vec::new();
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
        let proof =
            prove_built(&program, &elf, &mut forged, &format).expect("the forged build proves");
        assert!(
            !verifies(&proof, &elf, &format),
            "{forgery:?}: the forged build is refused"
        );
        refused.push(forgery);
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
    let proof = prove_built(&program, &elf, &mut forged, &format).expect("the forged run proves");
    assert!(
        !verifies(&proof, &elf, &format),
        "the PC chain past the exit call is refused"
    );
    println!(
        "IN-ROW FORGERIES REFUSED: {} and the PC chain past the exit",
        refused.len()
    );
}
