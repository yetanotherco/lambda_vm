use executor::{elf::Elf, vm::execution::run_program};

// NOTE: These tests require 64-bit RISC-V ELF files (RV64IM).
// The test ELF files need to be recompiled with a 64-bit toolchain.
// Until then, these tests will fail with "Not a 64-bit ELF" error.
fn run_program_and_check_output(elf_path: &str, expected_output: i64) {
    println!("Testing {}", elf_path);
    let elf_data = std::fs::read(elf_path).unwrap();
    let program = Elf::load(&elf_data).unwrap();
    println!("Program entry: 0x{:016x}", program.entry_point);
    program.image.iter().for_each(|(addr, word)| {
        println!("0x{:016x}: 0x{:08x}", addr, word);
    });
    let result =
        run_program(program.image, program.entry_point, vec![]).expect("Failed to run program");

    assert!(result.return_values.register_values.0 == expected_output);
}

#[test]
fn test_basic_program() {
    run_program_and_check_output("./program_artifacts/asm/basic_program.elf", 0);
}

#[test]
fn test_addi_one() {
    run_program_and_check_output("./program_artifacts/asm/addi_one.elf", 1);
}

#[test]
fn test_addi_minus_one() {
    run_program_and_check_output("./program_artifacts/asm/addi_minus_one.elf", -1);
}

#[test]
fn test_addi_max() {
    run_program_and_check_output("./program_artifacts/asm/addi_max.elf", 2047);
}

#[test]
fn test_addi_min() {
    run_program_and_check_output("./program_artifacts/asm/addi_min.elf", -2048);
}

#[test]
fn test_addi_reg() {
    run_program_and_check_output("./program_artifacts/asm/addi_reg.elf", 30);
}

#[test]
fn test_addi_reg_max() {
    run_program_and_check_output("./program_artifacts/asm/addi_reg_max.elf", 2080);
}

#[test]
fn test_addi_reg_min() {
    run_program_and_check_output("./program_artifacts/asm/addi_reg_min.elf", -2070);
}

#[test]
fn test_addi_255() {
    run_program_and_check_output("./program_artifacts/asm/addi_255.elf", 255);
}

#[test]
fn test_add() {
    run_program_and_check_output("./program_artifacts/asm/add.elf", 30);
}

#[test]
fn test_add_neg() {
    run_program_and_check_output("./program_artifacts/asm/add_neg.elf", 10);
}

#[test]
fn test_add_max() {
    run_program_and_check_output("./program_artifacts/asm/add_max.elf", i32::MAX as i64);
}

#[test]
fn test_add_max_plus_one() {
    // i64::MAX + 1 overflows to i64::MIN
    run_program_and_check_output("./program_artifacts/asm/add_max_plus_one.elf", i64::MIN);
}

#[test]
fn test_add_min() {
    run_program_and_check_output("./program_artifacts/asm/add_min.elf", i32::MIN as i64);
}

#[test]
fn test_add_min_minus_one() {
    // i64::MIN - 1 overflows to i64::MAX
    run_program_and_check_output("./program_artifacts/asm/add_min_minus_one.elf", i64::MAX);
}

#[test]
fn test_andi() {
    run_program_and_check_output("./program_artifacts/asm/andi.elf", 0x00);
}

#[test]
fn test_andi_one() {
    run_program_and_check_output("./program_artifacts/asm/andi_one.elf", 0x01);
}

#[test]
fn test_andi_one_and_zero() {
    run_program_and_check_output("./program_artifacts/asm/andi_one_and_zero.elf", 0x00);
}

#[test]
fn test_andi_one_and_two() {
    run_program_and_check_output("./program_artifacts/asm/andi_one_and_two.elf", 0x00);
}

#[test]
fn test_andi_max() {
    // andi with -1 immediate on -1 value = -1
    run_program_and_check_output("./program_artifacts/asm/andi_max.elf", -1);
}

#[test]
fn test_ori() {
    run_program_and_check_output("./program_artifacts/asm/ori.elf", 0x00);
}

#[test]
fn test_ori_one() {
    run_program_and_check_output("./program_artifacts/asm/ori_one.elf", 0x01);
}

#[test]
fn test_ori_one_and_one() {
    run_program_and_check_output("./program_artifacts/asm/ori_one_and_one.elf", 0x01);
}

#[test]
fn test_ori_two_and_one() {
    run_program_and_check_output("./program_artifacts/asm/ori_two_and_one.elf", 0x03);
}

#[test]
fn test_ori_five_and_four() {
    run_program_and_check_output("./program_artifacts/asm/ori_five_and_four.elf", 0x05);
}

#[test]
fn test_ori_three_and_five() {
    run_program_and_check_output("./program_artifacts/asm/ori_three_and_five.elf", 0x07);
}

#[test]
fn test_ori_max() {
    // ori with -1 immediate on -1 value = -1
    run_program_and_check_output("./program_artifacts/asm/ori_max.elf", -1);
}

#[test]
fn test_xori() {
    run_program_and_check_output("./program_artifacts/asm/xori.elf", 0x00);
}

#[test]
fn test_xori_one() {
    run_program_and_check_output("./program_artifacts/asm/xori_one.elf", 0x01);
}

#[test]
fn test_xori_one_and_one() {
    run_program_and_check_output("./program_artifacts/asm/xori_one_and_one.elf", 0x00);
}

#[test]
fn test_xori_max() {
    // xori zero with -1 immediate = -1
    run_program_and_check_output("./program_artifacts/asm/xori_max.elf", -1);
}

#[test]
fn test_xori_negate() {
    run_program_and_check_output("./program_artifacts/asm/xori_negate.elf", 0x01);
}

#[test]
fn test_slti() {
    run_program_and_check_output("./program_artifacts/asm/slti.elf", 0);
}

#[test]
fn test_slti_one() {
    run_program_and_check_output("./program_artifacts/asm/slti_one.elf", 1);
}

#[test]
fn test_slti_minus_one() {
    run_program_and_check_output("./program_artifacts/asm/slti_minus_one.elf", 0);
}

#[test]
fn test_slti_negative() {
    run_program_and_check_output("./program_artifacts/asm/slti_negative.elf", 1);
}

#[test]
fn test_slti_negative_minus() {
    run_program_and_check_output("./program_artifacts/asm/slti_negative_minus.elf", 0);
}

#[test]
fn test_sltiu() {
    run_program_and_check_output("./program_artifacts/asm/sltiu.elf", 0);
}

#[test]
fn test_sltiu_one() {
    run_program_and_check_output("./program_artifacts/asm/sltiu_one.elf", 1);
}

#[test]
fn test_sltiu_negative() {
    run_program_and_check_output("./program_artifacts/asm/sltiu_negative.elf", 0);
}

#[test]
fn test_sltiu_two_negatives() {
    run_program_and_check_output("./program_artifacts/asm/sltiu_two_negatives.elf", 1);
}

#[test]
fn test_slli() {
    run_program_and_check_output("./program_artifacts/asm/slli.elf", 0);
}

#[test]
fn test_slli_one() {
    run_program_and_check_output("./program_artifacts/asm/slli_one.elf", 0);
}

#[test]
fn test_slli_one_one() {
    run_program_and_check_output("./program_artifacts/asm/slli_one_one.elf", 2);
}

#[test]
fn test_slli_one_zero() {
    run_program_and_check_output("./program_artifacts/asm/slli_one_zero.elf", 1);
}

#[test]
fn test_slli_ff_four() {
    run_program_and_check_output("./program_artifacts/asm/slli_ff_four.elf", 0xFF0);
}

#[test]
fn test_slli_max() {
    // -1 << 4 = -16 in 64-bit
    run_program_and_check_output("./program_artifacts/asm/slli_max.elf", -16);
}

#[test]
fn test_slli_max_half() {
    // -1 << 15 = -32768 in 64-bit
    run_program_and_check_output("./program_artifacts/asm/slli_max_half.elf", -32768);
}

#[test]
fn test_slli_max_max() {
    // -1 << 31 = -2147483648 in 64-bit
    run_program_and_check_output("./program_artifacts/asm/slli_max_max.elf", -2147483648);
}

#[test]
fn test_slli_not_arith() {
    run_program_and_check_output("./program_artifacts/asm/slli_not_arith.elf", 2);
}

#[test]
fn test_srli() {
    run_program_and_check_output("./program_artifacts/asm/srli.elf", 0);
}

#[test]
fn test_srli_one() {
    run_program_and_check_output("./program_artifacts/asm/srli_one.elf", 0);
}

#[test]
fn test_srli_one_zero() {
    run_program_and_check_output("./program_artifacts/asm/srli_one_zero.elf", 1);
}

#[test]
fn test_srli_one_one() {
    run_program_and_check_output("./program_artifacts/asm/srli_one_one.elf", 0);
}

#[test]
fn test_srli_two_one() {
    run_program_and_check_output("./program_artifacts/asm/srli_two_one.elf", 1);
}

#[test]
fn test_srli_max() {
    // -1 (as unsigned) >> 4 = 0x0FFFFFFFFFFFFFFF in 64-bit
    run_program_and_check_output(
        "./program_artifacts/asm/srli_max.elf",
        0x0FFFFFFFFFFFFFFFu64 as i64,
    );
}

#[test]
fn test_srli_max_max() {
    // -1 (as unsigned) >> 31 = 0x1FFFFFFFF in 64-bit
    run_program_and_check_output(
        "./program_artifacts/asm/srli_max_max.elf",
        0x1FFFFFFFFu64 as i64,
    );
}

#[test]
fn test_srai() {
    run_program_and_check_output("./program_artifacts/asm/srai.elf", 0);
}

#[test]
fn test_srai_one() {
    run_program_and_check_output("./program_artifacts/asm/srai_one.elf", 0);
}

#[test]
fn test_srai_one_one() {
    run_program_and_check_output("./program_artifacts/asm/srai_one_one.elf", 0);
}

#[test]
fn test_srai_two_one() {
    run_program_and_check_output("./program_artifacts/asm/srai_two_one.elf", 1);
}

#[test]
fn test_srai_max() {
    // -1 >> 1 (arithmetic) = -1 in 64-bit
    run_program_and_check_output("./program_artifacts/asm/srai_max.elf", -1);
}

#[test]
fn test_srai_negative() {
    // -16 >> 1 (arithmetic) = -8 in 64-bit
    run_program_and_check_output("./program_artifacts/asm/srai_negative.elf", -8);
}

#[test]
fn test_jal() {
    run_program_and_check_output("./program_artifacts/asm/jal.elf", 1);
}

#[test]
fn test_jal_next() {
    run_program_and_check_output("./program_artifacts/asm/jal_next.elf", 0);
}

#[test]
fn test_jal_prev() {
    run_program_and_check_output("./program_artifacts/asm/jal_prev.elf", 2);
}

#[test]
fn test_jal_ret() {
    run_program_and_check_output("./program_artifacts/asm/jal_ret.elf", 0x11160);
}

#[test]
fn test_jalr() {
    run_program_and_check_output("./program_artifacts/asm/jalr.elf", 1);
}

#[test]
fn test_jalr_neg() {
    run_program_and_check_output("./program_artifacts/asm/jalr_neg.elf", 2);
}

#[test]
fn test_jalr_ret() {
    run_program_and_check_output("./program_artifacts/asm/jalr_ret.elf", 0x11160);
}

#[test]
fn test_jalr_odd() {
    run_program_and_check_output("./program_artifacts/asm/jalr_odd.elf", 1);
}

#[test]
fn test_jalr_odd_reg() {
    run_program_and_check_output("./program_artifacts/asm/jalr_odd_reg.elf", 1);
}

#[test]
fn test_bne() {
    run_program_and_check_output("./program_artifacts/asm/bne.elf", 2);
}

#[test]
fn test_bne_true() {
    run_program_and_check_output("./program_artifacts/asm/bne_true.elf", 1);
}

#[test]
fn test_bne_neg() {
    run_program_and_check_output("./program_artifacts/asm/bne_neg.elf", 3);
}

#[test]
fn test_loop_5() {
    run_program_and_check_output("./program_artifacts/asm/loop_5.elf", 5);
}

#[test]
fn test_lw_sw() {
    run_program_and_check_output("./program_artifacts/asm/lw_sw.elf", 1);
}

#[test]
fn test_lw_sw_offset() {
    run_program_and_check_output("./program_artifacts/asm/lw_sw_offset.elf", 1);
}

#[ignore = "Unaligned memory access not properly implemented yet"]
#[test]
fn test_lw_sw_offset_odd() {
    run_program_and_check_output("./program_artifacts/asm/lw_sw_offset_odd.elf", 1);
}

#[test]
fn test_auipc() {
    run_program_and_check_output("./program_artifacts/asm/auipc.elf", 0x11158);
}

#[test]
fn test_auipc_offset() {
    run_program_and_check_output("./program_artifacts/asm/auipc_offset.elf", 0x12158);
}

#[test]
fn test_mul() {
    run_program_and_check_output("./program_artifacts/asm/mul.elf", -200);
}

#[test]
fn test_mul_max() {
    run_program_and_check_output("./program_artifacts/asm/mul_max.elf", 1);
}

#[test]
fn test_mulh_max() {
    run_program_and_check_output("./program_artifacts/asm/mulh_max.elf", 99);
}

#[test]
fn test_mulhu_max() {
    // mulhu of -1 * -1 in 64-bit gives the upper 64 bits of 128-bit result
    // (-1) * (-1) as unsigned = 0xFFFFFFFFFFFFFFFE_0000000000000001
    // Upper 64 bits = 0xFFFFFFFFFFFFFFFE = -2
    run_program_and_check_output("./program_artifacts/asm/mulhu_max.elf", -2);
}
#[test]
fn test_mulhsu_max() {
    run_program_and_check_output("./program_artifacts/asm/mulhsu_max.elf", -100);
}

#[test]
fn test_div_zero() {
    // Division by zero returns -1 in RISC-V
    run_program_and_check_output("./program_artifacts/asm/div_zero.elf", -1);
}

#[test]
fn test_divu_zero() {
    // Division by zero returns all-ones in RISC-V
    run_program_and_check_output("./program_artifacts/asm/divu_zero.elf", -1);
}

#[test]
fn test_divu() {
    // -1 (as unsigned 64-bit) / 2 = 0x7FFFFFFFFFFFFFFF
    run_program_and_check_output("./program_artifacts/asm/divu.elf", i64::MAX);
}

#[test]
fn test_rem_zero() {
    run_program_and_check_output("./program_artifacts/asm/rem_zero.elf", 10);
}

#[test]
fn test_rem() {
    run_program_and_check_output("./program_artifacts/asm/rem.elf", -13);
}

#[test]
fn test_rem_overflow() {
    // i64::MIN % -1 = 0 (division would overflow, but remainder is 0)
    run_program_and_check_output("./program_artifacts/asm/rem_overflow.elf", 0);
}

#[test]
fn test_remu_zero() {
    run_program_and_check_output("./program_artifacts/asm/remu_zero.elf", 10);
}

#[test]
fn test_remu() {
    // -1 (as unsigned 64-bit) % 55 = 0xFFFFFFFFFFFFFFFF % 55 = 15
    run_program_and_check_output("./program_artifacts/asm/remu.elf", 15);
}

// ==================== W-suffix Instructions (RV64 specific) ====================

#[test]
fn test_addw() {
    // 0x7FFFFFFF + 1 = 0x80000000, sign-extends to 0xFFFFFFFF80000000
    run_program_and_check_output("./program_artifacts/asm/addw.elf", i32::MIN as i64);
}

#[test]
fn test_addw_pos() {
    // Simple positive: 10 + 20 = 30
    run_program_and_check_output("./program_artifacts/asm/addw_pos.elf", 30);
}

#[test]
fn test_subw() {
    // 10 - 20 = -10
    run_program_and_check_output("./program_artifacts/asm/subw.elf", -10);
}

#[test]
fn test_subw_overflow() {
    // 0x80000000 - 1 = 0x7FFFFFFF
    run_program_and_check_output("./program_artifacts/asm/subw_overflow.elf", i32::MAX as i64);
}

#[test]
fn test_addiw() {
    // 0x7FFFFFFF + 1 = 0x80000000, sign-extends to 0xFFFFFFFF80000000
    run_program_and_check_output("./program_artifacts/asm/addiw.elf", i32::MIN as i64);
}

#[test]
fn test_addiw_neg() {
    // 100 + (-50) = 50
    run_program_and_check_output("./program_artifacts/asm/addiw_neg.elf", 50);
}

#[test]
fn test_sllw() {
    // 1 << 31 = 0x80000000, sign-extends to 0xFFFFFFFF80000000
    run_program_and_check_output("./program_artifacts/asm/sllw.elf", i32::MIN as i64);
}

#[test]
fn test_sllw_wrap() {
    // shift amount uses only lower 5 bits: 33 & 0x1F = 1, so 1 << 1 = 2
    run_program_and_check_output("./program_artifacts/asm/sllw_wrap.elf", 2);
}

#[test]
fn test_srlw() {
    // 0x80000000 >> 1 = 0x40000000 (logical, no sign extension of shift)
    run_program_and_check_output("./program_artifacts/asm/srlw.elf", 0x40000000);
}

#[test]
fn test_sraw() {
    // 0x80000000 >> 1 (arithmetic) = 0xC0000000, sign-extends to 0xFFFFFFFFC0000000
    run_program_and_check_output(
        "./program_artifacts/asm/sraw.elf",
        0xFFFFFFFFC0000000u64 as i64,
    );
}

#[test]
fn test_slliw() {
    // 1 << 31 = 0x80000000, sign-extends to 0xFFFFFFFF80000000
    run_program_and_check_output("./program_artifacts/asm/slliw.elf", i32::MIN as i64);
}

#[test]
fn test_srliw() {
    // 0x80000000 >> 1 = 0x40000000
    run_program_and_check_output("./program_artifacts/asm/srliw.elf", 0x40000000);
}

#[test]
fn test_sraiw() {
    // 0x80000000 >> 1 (arithmetic) = 0xC0000000, sign-extends to 0xFFFFFFFFC0000000
    run_program_and_check_output(
        "./program_artifacts/asm/sraiw.elf",
        0xFFFFFFFFC0000000u64 as i64,
    );
}

#[test]
fn test_mulw() {
    // 100000 * 30000 = 3000000000 = 0xB2D05E00, sign-extends to negative
    run_program_and_check_output(
        "./program_artifacts/asm/mulw.elf",
        0xFFFFFFFFB2D05E00u64 as i64,
    );
}

#[test]
fn test_mulw_neg() {
    // -10 * 20 = -200
    run_program_and_check_output("./program_artifacts/asm/mulw_neg.elf", -200);
}

#[test]
fn test_divw() {
    // -100 / 7 = -14
    run_program_and_check_output("./program_artifacts/asm/divw.elf", -14);
}

#[test]
fn test_divw_zero() {
    // Division by zero returns -1
    run_program_and_check_output("./program_artifacts/asm/divw_zero.elf", -1);
}

#[test]
fn test_divw_overflow() {
    // i32::MIN / -1 would overflow, RISC-V returns i32::MIN
    run_program_and_check_output("./program_artifacts/asm/divw_overflow.elf", i32::MIN as i64);
}

#[test]
fn test_divuw_zero() {
    // DIVUW by zero returns 0xFFFFFFFF, sign-extended to -1
    run_program_and_check_output("./program_artifacts/asm/divuw_zero.elf", -1);
}

#[test]
fn test_divuw() {
    // 0xFFFFFFFF / 2 = 0x7FFFFFFF (as 32-bit unsigned)
    run_program_and_check_output("./program_artifacts/asm/divuw.elf", 0x7FFFFFFF);
}

#[test]
fn test_remw() {
    // -100 % 7 = -2
    run_program_and_check_output("./program_artifacts/asm/remw.elf", -2);
}

#[test]
fn test_remw_zero() {
    // REMW by zero returns dividend
    run_program_and_check_output("./program_artifacts/asm/remw_zero.elf", 42);
}

#[test]
fn test_remw_overflow() {
    // i32::MIN % -1 = 0 (division would overflow, but remainder is 0)
    run_program_and_check_output("./program_artifacts/asm/remw_overflow.elf", 0);
}

#[test]
fn test_remuw_zero() {
    // REMUW by zero returns dividend
    run_program_and_check_output("./program_artifacts/asm/remuw_zero.elf", 42);
}

#[test]
fn test_remuw() {
    // 0xFFFFFFFF % 7 = 3
    run_program_and_check_output("./program_artifacts/asm/remuw.elf", 3);
}

// ==================== 64-bit Load/Store ====================

#[test]
fn test_ld_sd() {
    // Store and load 0x123456789ABCDEF0
    run_program_and_check_output(
        "./program_artifacts/asm/ld_sd.elf",
        0x123456789ABCDEF0u64 as i64,
    );
}

#[test]
fn test_ld_sd_offset() {
    // Store and load with offset
    run_program_and_check_output(
        "./program_artifacts/asm/ld_sd_offset.elf",
        0xDEADBEEFCAFEBABEu64 as i64,
    );
}

#[test]
fn test_ld_sd_neg() {
    // Store and load -1
    run_program_and_check_output("./program_artifacts/asm/ld_sd_neg.elf", -1);
}

#[test]
fn test_lwu() {
    // LWU zero-extends: 0xFFFFFFFF -> 0x00000000FFFFFFFF
    run_program_and_check_output("./program_artifacts/asm/lwu.elf", 0xFFFFFFFF);
}

#[test]
fn test_lw_sign_extend() {
    // LW sign-extends: 0x80000000 -> 0xFFFFFFFF80000000
    run_program_and_check_output(
        "./program_artifacts/asm/lw_sign_extend.elf",
        i32::MIN as i64,
    );
}

#[test]
fn test_lwu_vs_lw() {
    // LWU zero-extends: 0x80000000 -> 0x0000000080000000
    run_program_and_check_output(
        "./program_artifacts/asm/lwu_vs_lw.elf",
        0x80000000u64 as i64,
    );
}

// ==================== Missing Branch Instructions ====================

#[test]
fn test_beq() {
    // BEQ taken: 10 == 10, result = 3
    run_program_and_check_output("./program_artifacts/asm/beq.elf", 3);
}

#[test]
fn test_beq_false() {
    // BEQ not taken: 10 != 20, result = 2
    run_program_and_check_output("./program_artifacts/asm/beq_false.elf", 2);
}

#[test]
fn test_blt() {
    // BLT taken: -10 < 10, result = 3
    run_program_and_check_output("./program_artifacts/asm/blt.elf", 3);
}

#[test]
fn test_blt_false() {
    // BLT not taken: 10 is not < 10, result = 2
    run_program_and_check_output("./program_artifacts/asm/blt_false.elf", 2);
}

#[test]
fn test_bge() {
    // BGE taken: 10 >= 10, result = 3
    run_program_and_check_output("./program_artifacts/asm/bge.elf", 3);
}

#[test]
fn test_bge_greater() {
    // BGE taken: 20 >= 10, result = 3
    run_program_and_check_output("./program_artifacts/asm/bge_greater.elf", 3);
}

#[test]
fn test_bge_false() {
    // BGE not taken: -10 < 10, result = 2
    run_program_and_check_output("./program_artifacts/asm/bge_false.elf", 2);
}

#[test]
fn test_bltu() {
    // BLTU taken: 5 < 10 (unsigned), result = 3
    run_program_and_check_output("./program_artifacts/asm/bltu.elf", 3);
}

#[test]
fn test_bltu_neg() {
    // BLTU not taken: -1 (0xFFFF...) is NOT < 10 unsigned, result = 2
    run_program_and_check_output("./program_artifacts/asm/bltu_neg.elf", 2);
}

#[test]
fn test_bgeu() {
    // BGEU taken: 10 >= 10 (unsigned), result = 3
    run_program_and_check_output("./program_artifacts/asm/bgeu.elf", 3);
}

#[test]
fn test_bgeu_neg() {
    // BGEU taken: -1 (0xFFFF...) >= 10 unsigned, result = 3
    run_program_and_check_output("./program_artifacts/asm/bgeu_neg.elf", 3);
}

// ==================== LUI Instruction ====================

#[test]
fn test_lui() {
    // LUI: 0x12345 << 12 = 0x12345000
    run_program_and_check_output("./program_artifacts/asm/lui.elf", 0x12345000);
}

#[test]
fn test_lui_neg() {
    // LUI: 0x80000 << 12 = 0x80000000, sign-extends to 0xFFFFFFFF80000000
    run_program_and_check_output("./program_artifacts/asm/lui_neg.elf", i32::MIN as i64);
}

#[test]
fn test_lui_max() {
    // LUI: 0x7FFFF << 12 = 0x7FFFF000
    run_program_and_check_output("./program_artifacts/asm/lui_max.elf", 0x7FFFF000);
}

// ==================== 64-bit Edge Cases ====================

#[test]
fn test_add_64bit() {
    // 0x100000000 + 0x100000000 = 0x200000000
    run_program_and_check_output("./program_artifacts/asm/add_64bit.elf", 0x200000000i64);
}

#[test]
fn test_slli_64() {
    // 1 << 32 = 0x100000000
    run_program_and_check_output("./program_artifacts/asm/slli_64.elf", 0x100000000i64);
}

#[test]
fn test_slli_63() {
    // 1 << 63 = i64::MIN
    run_program_and_check_output("./program_artifacts/asm/slli_63.elf", i64::MIN);
}

#[test]
fn test_srli_64() {
    // 0x123456789ABCDEF0 >> 32 = 0x12345678
    run_program_and_check_output("./program_artifacts/asm/srli_64.elf", 0x12345678);
}

#[test]
fn test_srai_64() {
    // 0x8000000000000000 >> 32 (arithmetic) = 0xFFFFFFFF80000000
    run_program_and_check_output(
        "./program_artifacts/asm/srai_64.elf",
        0xFFFFFFFF80000000u64 as i64,
    );
}

#[test]
fn test_mul_64bit() {
    // 0x100000000 * 2 = 0x200000000
    run_program_and_check_output("./program_artifacts/asm/mul_64bit.elf", 0x200000000i64);
}

#[test]
fn test_div_overflow() {
    // i64::MIN / -1 returns i64::MIN (RISC-V behavior for overflow)
    run_program_and_check_output("./program_artifacts/asm/div_overflow.elf", i64::MIN);
}

#[test]
fn test_mulh_64bit() {
    // 0x100000000 * 0x100000000 = 0x10000000000000000 (128-bit), upper 64 = 1
    run_program_and_check_output("./program_artifacts/asm/mulh_64bit.elf", 1);
}

// ==================== SUB Register-Register ====================

#[test]
fn test_sub() {
    // 30 - 10 = 20
    run_program_and_check_output("./program_artifacts/asm/sub.elf", 20);
}

#[test]
fn test_sub_neg_result() {
    // 10 - 30 = -20
    run_program_and_check_output("./program_artifacts/asm/sub_neg_result.elf", -20);
}

#[test]
fn test_sub_64bit() {
    // 0x200000000 - 0x100000000 = 0x100000000
    run_program_and_check_output("./program_artifacts/asm/sub_64bit.elf", 0x100000000i64);
}

#[test]
fn test_sub_underflow() {
    // 0 - 1 = -1
    run_program_and_check_output("./program_artifacts/asm/sub_underflow.elf", -1);
}
