//! Tests for RV64C (compressed instruction) decompression.

use crate::vm::instruction::decoding::{ArithOp, Instruction, InstructionError, LoadStoreWidth};
use crate::vm::instruction::decompress::{decompress, instr_len};

#[test]
fn instr_len_distinguishes_compressed_from_full() {
    assert_eq!(instr_len(0x0000), 2); // low bits 00
    assert_eq!(instr_len(0x4515), 2); // low bits 01 (c.li)
    assert_eq!(instr_len(0x8082), 2); // low bits 10 (c.jr ra)
    assert_eq!(instr_len(0x0033), 4); // low bits 11 (a base instruction)
}

#[test]
fn all_zero_halfword_is_illegal() {
    assert!(matches!(
        decompress(0x0000),
        Err(InstructionError::IllegalCompressed(0x0000))
    ));
}

#[test]
fn c_li_expands_to_addi_from_x0() {
    // c.li x10, 5
    match decompress(0x4515).unwrap() {
        Instruction::ArithImm {
            dst,
            src,
            imm,
            op: ArithOp::Add,
        } => {
            assert_eq!((dst, src, imm), (10, 0, 5));
        }
        other => panic!("expected ArithImm, got {other:?}"),
    }
}

#[test]
fn c_li_sign_extends_negative_immediate() {
    // c.li x10, -1 (6-bit immediate, all bits set) exercises the sign-extension path.
    let enc: u16 = (0b010 << 13) | (1 << 12) | (10 << 7) | (0b11111 << 2) | 0b01;
    match decompress(enc).unwrap() {
        Instruction::ArithImm {
            dst,
            src,
            imm,
            op: ArithOp::Add,
        } => {
            assert_eq!((dst, src, imm), (10, 0, -1));
        }
        other => panic!("expected ArithImm, got {other:?}"),
    }
}

#[test]
fn c_addi4spn_expands_to_addi_from_sp() {
    // c.addi4spn x8, x2, 4
    match decompress(0x0040).unwrap() {
        Instruction::ArithImm {
            dst,
            src,
            imm,
            op: ArithOp::Add,
        } => {
            assert_eq!((dst, src, imm), (8, 2, 4));
        }
        other => panic!("expected ArithImm, got {other:?}"),
    }
}

#[test]
fn c_mv_and_c_add() {
    // c.mv x10, x11  ->  add x10, x0, x11
    match decompress(0x852E).unwrap() {
        Instruction::Arith {
            dst,
            src1,
            src2,
            op: ArithOp::Add,
        } => assert_eq!((dst, src1, src2), (10, 0, 11)),
        other => panic!("expected Arith (c.mv), got {other:?}"),
    }
    // c.add x10, x11  ->  add x10, x10, x11
    match decompress(0x952E).unwrap() {
        Instruction::Arith {
            dst,
            src1,
            src2,
            op: ArithOp::Add,
        } => assert_eq!((dst, src1, src2), (10, 10, 11)),
        other => panic!("expected Arith (c.add), got {other:?}"),
    }
}

#[test]
fn c_jr_and_c_jalr() {
    // c.jr ra (0x8082) -> jalr x0, 0(x1)
    match decompress(0x8082).unwrap() {
        Instruction::JumpAndLinkRegister { base, dst, offset } => {
            assert_eq!((base, dst, offset), (1, 0, 0))
        }
        other => panic!("expected JALR (c.jr), got {other:?}"),
    }
    // c.jalr ra (0x9082) -> jalr x1, 0(x1)
    match decompress(0x9082).unwrap() {
        Instruction::JumpAndLinkRegister { base, dst, offset } => {
            assert_eq!((base, dst, offset), (1, 1, 0))
        }
        other => panic!("expected JALR (c.jalr), got {other:?}"),
    }
}

#[test]
fn c_ebreak_expands_to_ecall_ebreak() {
    assert!(matches!(decompress(0x9002), Ok(Instruction::EcallEbreak)));
}

#[test]
fn c_sub_expands_to_register_sub_not_imm() {
    // c.sub x8, x9 -> sub x8, x8, x9 (must be Arith, never ArithImm).
    // funct3=100, funct2=11, rd'=x8 (field 0), sub-bits=00 (C.SUB), rs2'=x9 (field 1)
    let enc: u16 = (0b100 << 13) | (0b11 << 10) | (1 << 2) | 0b01;
    match decompress(enc).unwrap() {
        Instruction::Arith {
            dst,
            src1,
            src2,
            op: ArithOp::Sub,
        } => assert_eq!((dst, src1, src2), (8, 8, 9)),
        other => panic!("expected Arith Sub, got {other:?}"),
    }
}

#[test]
fn c_lwsp_x0_is_reserved() {
    // c.lwsp with rd == 0 is reserved.
    let enc: u16 = (0b010 << 13) | 0b10;
    assert!(matches!(
        decompress(enc),
        Err(InstructionError::ReservedCompressed(_))
    ));
}

#[test]
fn c_lwsp_loads_from_sp() {
    // c.lwsp x10, 0(x2)
    let enc: u16 = (0b010 << 13) | (10 << 7) | 0b10;
    match decompress(enc).unwrap() {
        Instruction::Load {
            dst,
            offset,
            base,
            width: LoadStoreWidth::Word,
        } => assert_eq!((dst, offset, base), (10, 0, 2)),
        other => panic!("expected Load (c.lwsp), got {other:?}"),
    }
}

#[test]
fn float_compressed_is_excluded() {
    // c.fld (Q0 funct3=001) must not decode to an integer instruction.
    let enc: u16 = 0b001 << 13;
    assert!(matches!(
        decompress(enc),
        Err(InstructionError::IllegalCompressed(_))
    ));
}
