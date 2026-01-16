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
    let (results, _logs) =
        run_program(program.image, program.entry_point, vec![]).expect("Failed to run program");

    assert!(results.register_values.0 == expected_output);
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
fn test_remu_zero() {
    run_program_and_check_output("./program_artifacts/asm/remu_zero.elf", 10);
}

#[test]
fn test_remu() {
    // -1 (as unsigned 64-bit) % 55 = 0xFFFFFFFFFFFFFFFF % 55 = 15
    run_program_and_check_output("./program_artifacts/asm/remu.elf", 15);
}
