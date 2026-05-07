//! CPU_BITWISE chip — handles AND, OR, XOR (and their `*W` 32-bit variants).
//!
//! After Phase 2 step C3 the chip owns its column layout and bus interactions
//! independently from the base CPU chip. The base CPU layout dropped the
//! per-byte `ARG1[0..7]` and `RES[0..7]` cells (which only the BITWISE bus
//! ever needed); CPU_BITWISE keeps them locally so the AND/OR/XOR byte
//! lookups still fire. `CpuOperation` is shared (the source data is the
//! same for both chips); only the column layout, bus interactions and
//! trace generator are chip-specific.

use super::cpu::CPU_PADDING_PC;
pub use super::cpu::CpuOperation;

use stark::lookup::{BusInteraction, BusValue, LinearTerm, Multiplicity, Packing};
use stark::trace::TraceTable;

use super::types::{BusId, FE, GoldilocksExtension, GoldilocksField};

// =========================================================================
// Column layout
// =========================================================================

/// Column layout for the CPU_BITWISE chip.
///
/// Cols 0-70 are the base CPU layout (re-exported by name). The byte cells
/// `ARG1[0..7]` and `RES[0..7]` live at indices 71-86, beyond the base CPU
/// width. The HALF_HI aux columns (`ARG1_LO_HALF_HI`, `ARG1_HI_HALF_HI`,
/// `RES_HI_HALF_HI`) are present at indices 68-70 because of the base
/// re-export but are unread by CPU_BITWISE constraints/sends. The trace
/// generator still populates them with the correct halfword values for
/// uniformity.
pub mod cols {
    pub use super::super::cpu::cols::{
        ADD, AND, ARG1_HI, ARG1_HI_HALF_HI, ARG1_LO, ARG1_LO_HALF_HI, ARG2, ARG2_0, ARG2_1, ARG2_2,
        ARG2_3, ARG2_4, ARG2_5, ARG2_6, ARG2_7, ARG2_HI, ARG2_LO, BEQ, BLT, BRANCH_COND,
        C_TYPE_INSTRUCTION, DIVREM, EBREAK, ECALL, IMM_0, IMM_1, IS_EQUAL, JALR, LOAD,
        MEMORY_2BYTES, MEMORY_4BYTES, MEMORY_8BYTES, MP_SELECTOR, MUL, MULDIV_SELECTOR, NEXT_PC_0,
        NEXT_PC_1, OR, PC_0, PC_1, PC_DOUBLE_READ, PREV_PC_TIMESTAMP_BORROW, RD, READ_REGISTER1,
        READ_REGISTER2, RES_EXT_BIT, RES_HI, RES_HI_HALF_HI, RES_INV, RES_LO, RES_LO_HALF_HI, RS1,
        RS2, RV1_0, RV1_1, RV1_2, RV1_EXT_BIT, RV2_0, RV2_1, RV2_2, RV2_EXT_BIT, RVD_0, RVD_1,
        SHIFT, SIGNED, SLT, STORE, SUB, TIMESTAMP, WORD_INSTR, WRITE_REGISTER, XOR,
    };

    /// arg1[0..8]: Extended rv1 as DWordBL (8 bytes), only present on the
    /// CPU_BITWISE layout (Phase 2 step C3).
    pub const ARG1_0: usize = 71;
    pub const ARG1_1: usize = 72;
    pub const ARG1_2: usize = 73;
    pub const ARG1_3: usize = 74;
    pub const ARG1_4: usize = 75;
    pub const ARG1_5: usize = 76;
    pub const ARG1_6: usize = 77;
    pub const ARG1_7: usize = 78;

    /// res[0..8]: ALU result as DWordBL (8 bytes), CPU_BITWISE only.
    pub const RES_0: usize = 79;
    pub const RES_1: usize = 80;
    pub const RES_2: usize = 81;
    pub const RES_3: usize = 82;
    pub const RES_4: usize = 83;
    pub const RES_5: usize = 84;
    pub const RES_6: usize = 85;
    pub const RES_7: usize = 86;

    /// Total number of columns on the CPU_BITWISE table.
    pub const NUM_COLUMNS: usize = 87;

    /// ARG1 byte columns as array
    pub const ARG1: [usize; 8] = [
        ARG1_0, ARG1_1, ARG1_2, ARG1_3, ARG1_4, ARG1_5, ARG1_6, ARG1_7,
    ];

    /// RES byte columns as array
    pub const RES: [usize; 8] = [RES_0, RES_1, RES_2, RES_3, RES_4, RES_5, RES_6, RES_7];
}

// =========================================================================
// Bus interactions
// =========================================================================

/// Bus interactions for the CPU_BITWISE chip — only the buses an AND/OR/XOR
/// row actually fires.
///
/// Builds on `super::cpu::bus_interactions()` (the base CPU send list) by:
/// - keeping only Decode, IsByte, IsHalfword, Bitwise, Msb16, Memw, Memory;
/// - dropping the 6 limb-form IS_HALFWORD sends for ARG1_LO/HI and RES_HI
///   (they reference HALF_HI aux cells that the bitwise chip doesn't use);
/// - adding 8 byte-pair IS_HALFWORD sends for ARG1 and RES (sourced from
///   the chip-local byte cells); and
/// - adding 8 unified BITWISE bus sends, one per byte index.
pub fn bus_interactions() -> Vec<BusInteraction> {
    let keep_buses: [u64; 7] = [
        BusId::Decode.into(),
        BusId::IsByte.into(),
        BusId::IsHalfword.into(),
        BusId::Bitwise.into(),
        BusId::Msb16.into(),
        BusId::Memw.into(),
        BusId::Memory.into(),
    ];

    let mut interactions: Vec<BusInteraction> = super::cpu::bus_interactions()
        .into_iter()
        .filter(|i| keep_buses.contains(&i.bus_id))
        .filter(|i| !is_arg1_or_res_hi_limb_halfword(i))
        .collect();

    // Byte-pair IS_HALFWORD range checks for ARG1 and RES (4 each).
    for arr in [&cols::ARG1, &cols::RES] {
        for i in 0..4 {
            interactions.push(BusInteraction::sender(
                BusId::IsHalfword,
                Multiplicity::One,
                vec![BusValue::linear(vec![
                    LinearTerm::Column {
                        coefficient: 1,
                        column: arr[2 * i],
                    },
                    LinearTerm::Column {
                        coefficient: 256,
                        column: arr[2 * i + 1],
                    },
                ])],
            ));
        }
    }

    // Unified BITWISE bus: 8 sends, one per byte position. Token format
    // (op_id, X, Y, RESULT) with disjoint-bit op_id = AND + 2*OR + 4*XOR
    // and multiplicity = AND + OR + XOR (at-most-one).
    for i in 0..8 {
        interactions.push(BusInteraction::sender(
            BusId::Bitwise,
            Multiplicity::Sum3(cols::AND, cols::OR, cols::XOR),
            vec![
                BusValue::linear(vec![
                    LinearTerm::Column {
                        coefficient: 1,
                        column: cols::AND,
                    },
                    LinearTerm::Column {
                        coefficient: 2,
                        column: cols::OR,
                    },
                    LinearTerm::Column {
                        coefficient: 4,
                        column: cols::XOR,
                    },
                ]),
                BusValue::Packed {
                    start_column: cols::ARG1[i],
                    packing: Packing::Direct,
                },
                BusValue::Packed {
                    start_column: cols::ARG2[i],
                    packing: Packing::Direct,
                },
                BusValue::Packed {
                    start_column: cols::RES[i],
                    packing: Packing::Direct,
                },
            ],
        ));
    }

    interactions
}

/// True iff `i` is one of the limb-form IS_HALFWORD sends emitted by base
/// CPU's `bus_interactions` for the ARG1_LO/HI or RES_HI halfword
/// decompositions. CPU_BITWISE substitutes byte-pair sends for these.
fn is_arg1_or_res_hi_limb_halfword(i: &BusInteraction) -> bool {
    if i.bus_id != u64::from(BusId::IsHalfword) {
        return false;
    }
    if i.values.len() != 1 {
        return false;
    }
    let target_aux = [
        cols::ARG1_LO_HALF_HI,
        cols::ARG1_HI_HALF_HI,
        cols::RES_HI_HALF_HI,
    ];
    match &i.values[0] {
        BusValue::Packed { start_column, .. } => target_aux.contains(start_column),
        BusValue::Linear(terms) => terms.iter().any(|t| match t {
            LinearTerm::Column { column, .. } => target_aux.contains(column),
            _ => false,
        }),
    }
}

// =========================================================================
// Trace generation
// =========================================================================

/// Generates the CPU_BITWISE trace table from a list of CPU operations.
///
/// The CPU_BITWISE chip processes only AND/OR/XOR rows. The trace layout
/// extends the base CPU layout with byte cells for `ARG1` and `RES`
/// (cols 71-86) needed by the BITWISE bus and ARG1/RES IS_HALFWORD
/// byte-pair sends.
pub fn generate_cpu_trace_bitwise(
    operations: &[CpuOperation],
) -> TraceTable<GoldilocksField, GoldilocksExtension> {
    let n = operations.len();
    let num_rows = n.next_power_of_two().max(4);
    let mut data = vec![FE::zero(); num_rows * cols::NUM_COLUMNS];

    for (row_idx, op) in operations.iter().enumerate() {
        let base = row_idx * cols::NUM_COLUMNS;
        let d = &op.decode;

        data[base + cols::TIMESTAMP] = FE::from(op.timestamp);
        data[base + cols::PC_0] = FE::from(d.pc & 0xFFFF_FFFF);
        data[base + cols::PC_1] = FE::from(d.pc >> 32);
        data[base + cols::RS1] = FE::from(d.rs1 as u64);
        data[base + cols::RS2] = FE::from(d.rs2 as u64);
        data[base + cols::RD] = FE::from(d.rd as u64);
        data[base + cols::READ_REGISTER1] = FE::from((d.read_register1 && d.rs1 != 0) as u64);
        data[base + cols::READ_REGISTER2] = FE::from((d.read_register2 && d.rs2 != 0) as u64);
        data[base + cols::WRITE_REGISTER] = FE::from((d.write_register && d.rd != 0) as u64);
        data[base + cols::MEMORY_2BYTES] = FE::from(d.memory_2bytes as u64);
        data[base + cols::MEMORY_4BYTES] = FE::from(d.memory_4bytes as u64);
        data[base + cols::MEMORY_8BYTES] = FE::from(d.memory_8bytes as u64);
        data[base + cols::C_TYPE_INSTRUCTION] = FE::from(d.c_type as u64);
        data[base + cols::IMM_0] = FE::from(d.imm & 0xFFFF_FFFF);
        data[base + cols::IMM_1] = FE::from(d.imm >> 32);
        data[base + cols::SIGNED] = FE::from(d.signed as u64);
        data[base + cols::MP_SELECTOR] = FE::from(d.mp_selector as u64);
        data[base + cols::MULDIV_SELECTOR] = FE::from(d.muldiv_selector as u64);
        data[base + cols::WORD_INSTR] = FE::from(d.word_instr as u64);

        data[base + cols::ADD] = FE::from(d.op_add as u64);
        data[base + cols::SUB] = FE::from(d.op_sub as u64);
        data[base + cols::SLT] = FE::from(d.op_slt as u64);
        data[base + cols::AND] = FE::from(d.op_and as u64);
        data[base + cols::OR] = FE::from(d.op_or as u64);
        data[base + cols::XOR] = FE::from(d.op_xor as u64);
        data[base + cols::SHIFT] = FE::from(d.op_shift as u64);
        data[base + cols::JALR] = FE::from(d.op_jalr as u64);
        data[base + cols::BEQ] = FE::from(d.op_beq as u64);
        data[base + cols::BLT] = FE::from(d.op_blt as u64);
        data[base + cols::LOAD] = FE::from(d.op_load as u64);
        data[base + cols::STORE] = FE::from(d.op_store as u64);
        data[base + cols::MUL] = FE::from(d.op_mul as u64);
        data[base + cols::DIVREM] = FE::from(d.op_divrem as u64);
        data[base + cols::ECALL] = FE::from(d.op_ecall as u64);
        data[base + cols::EBREAK] = FE::from(d.op_ebreak as u64);

        data[base + cols::NEXT_PC_0] = FE::from(op.next_pc & 0xFFFF_FFFF);
        data[base + cols::NEXT_PC_1] = FE::from(op.next_pc >> 32);

        let rvd = if d.op_load { op.rvd } else { op.compute_rvd() };
        data[base + cols::RVD_0] = FE::from(rvd & 0xFFFF_FFFF);
        data[base + cols::RVD_1] = FE::from(rvd >> 32);

        data[base + cols::RV1_0] = FE::from(op.rv1 & 0xFFFF);
        data[base + cols::RV1_1] = FE::from((op.rv1 >> 16) & 0xFFFF);
        data[base + cols::RV1_2] = FE::from(op.rv1 >> 32);
        data[base + cols::RV2_0] = FE::from(op.rv2 & 0xFFFF);
        data[base + cols::RV2_1] = FE::from((op.rv2 >> 16) & 0xFFFF);
        data[base + cols::RV2_2] = FE::from(op.rv2 >> 32);

        let rv1_ext_bit = d.word_instr && CpuOperation::sign_bit_32(op.rv1);
        data[base + cols::RV1_EXT_BIT] = FE::from(rv1_ext_bit as u64);

        let arg1 = op.compute_arg1();
        data[base + cols::ARG1_LO] = FE::from(arg1 & 0xFFFF_FFFF);
        data[base + cols::ARG1_HI] = FE::from(arg1 >> 32);
        for i in 0..8 {
            data[base + cols::ARG1[i]] = FE::from((arg1 >> (i * 8)) & 0xFF);
        }

        let arg2 = op.compute_arg2();
        let rv2_ext_bit = d.word_instr && CpuOperation::sign_bit_32(op.rv2);
        data[base + cols::RV2_EXT_BIT] = FE::from(rv2_ext_bit as u64);
        for i in 0..8 {
            data[base + cols::ARG2[i]] = FE::from((arg2 >> (i * 8)) & 0xFF);
        }
        data[base + cols::ARG2_LO] = FE::from(arg2 & 0xFFFF_FFFF);
        data[base + cols::ARG2_HI] = FE::from(arg2 >> 32);

        let res = op.compute_res();
        let res_ext_bit = d.word_instr && CpuOperation::sign_bit_32(res);
        data[base + cols::RES_EXT_BIT] = FE::from(res_ext_bit as u64);
        for i in 0..8 {
            data[base + cols::RES[i]] = FE::from((res >> (i * 8)) & 0xFF);
        }
        data[base + cols::RES_LO] = FE::from(res & 0xFFFF_FFFF);
        data[base + cols::RES_HI] = FE::from(res >> 32);

        // Halfword aux cells. RES_LO_HALF_HI is read by the MSB16 res-ext-bit
        // sender that CPU_BITWISE inherits; the ARG1/RES_HI HALF_HI cells are
        // unread on this chip (CPU_BITWISE uses byte-pair IS_HALFWORD sends
        // for ARG1 and RES) but they're populated for trace-table uniformity
        // with the base CPU layout that exposes the same constants.
        data[base + cols::RES_LO_HALF_HI] = FE::from((res >> 16) & 0xFFFF);
        data[base + cols::ARG1_LO_HALF_HI] = FE::from((arg1 >> 16) & 0xFFFF);
        data[base + cols::ARG1_HI_HALF_HI] = FE::from((arg1 >> 48) & 0xFFFF);
        data[base + cols::RES_HI_HALF_HI] = FE::from((res >> 48) & 0xFFFF);

        let res_lo = res & 0xFFFF_FFFF;
        let res_hi = res >> 32;
        let sum = res_lo + res_hi;
        let res_inv = if sum != 0 {
            FE::from(sum).inv().expect("nonzero element has inverse")
        } else {
            FE::zero()
        };
        data[base + cols::RES_INV] = res_inv;

        data[base + cols::IS_EQUAL] = FE::from(op.is_equal as u64);
        data[base + cols::BRANCH_COND] = FE::from(op.branch_cond as u64);

        let pc_double_read = (d.read_register1 && d.rs1 == 255) as u64;
        let ts_lo = op.timestamp & 0xFFFF_FFFF;
        let prev_pc_ts_borrow = if pc_double_read == 0 && ts_lo < 3 {
            1u64
        } else {
            0u64
        };
        data[base + cols::PC_DOUBLE_READ] = FE::from(pc_double_read);
        data[base + cols::PREV_PC_TIMESTAMP_BORROW] = FE::from(prev_pc_ts_borrow);
    }

    // Padding: pc=1 (odd, unreachable), next_pc=5; CO69 NextPcAdd carry=0.
    for row_idx in n..num_rows {
        let base = row_idx * cols::NUM_COLUMNS;
        data[base + cols::PC_0] = FE::from(CPU_PADDING_PC);
        data[base + cols::NEXT_PC_0] = FE::from(CPU_PADDING_PC + 4);
    }

    TraceTable::new_main(data, cols::NUM_COLUMNS, 1)
}
