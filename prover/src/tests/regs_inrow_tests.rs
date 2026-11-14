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
    let today = BlockFormat::production();
    let in_row_proof = prove(&elf, &in_row());
    let today_proof = prove(&elf, &today);
    assert!(verifies(&in_row_proof, &elf, &in_row()));
    assert!(verifies(&today_proof, &elf, &today));
    assert!(!verifies(&in_row_proof, &elf, &today));
    assert!(!verifies(&today_proof, &elf, &in_row()));
}
