use executor::{
    elf::Elf,
    vm::execution::{Executor, ExecutorError},
};

/// Run a program and verify it exits successfully (exit code 0).
///
/// Programs use Linux syscall 93 (exit) with a0=0 for successful termination.
/// The computation results are verified by the prover via execution traces,
/// not by checking register values after halt.
fn run_program(elf_path: &str) {
    println!("Testing {}", elf_path);
    let elf_data = std::fs::read(elf_path).unwrap();
    let program = Elf::load(&elf_data).unwrap();
    let mut executor = Executor::new(&program, vec![]).expect("Failed to create executor");

    while let Some(_logs) = executor.resume().expect("Failed to execute") {}

    let result = executor.finish().expect("Failed to get return values");
    assert_eq!(
        result.register_values.0, 0,
        "Program {} exited with non-zero exit code: {}",
        elf_path, result.register_values.0
    );
}

/// Test that the memory-mapped private input region is readable by guest programs.
/// The ASM program reads from 0xFF000000 and commits 8 bytes of data.
#[test]
fn test_private_input_memory_mapped() {
    let elf_data = std::fs::read("./program_artifacts/asm/test_private_input_xpage.elf").unwrap();
    let program = Elf::load(&elf_data).unwrap();
    let input: Vec<u8> = (0u8..16).collect();
    let executor = Executor::new(&program, input.clone()).unwrap();
    let result = executor.run().unwrap();
    // Committed bytes are at 0xFF000010 = data bytes [0..8]
    assert_eq!(result.return_values.memory_values, input[0..8].to_vec());
}

#[test]
fn test_basic_program() {
    run_program("./program_artifacts/asm/basic_program.elf");
}

#[test]
fn test_addi_one() {
    run_program("./program_artifacts/asm/addi_one.elf");
}

#[test]
fn test_addi_minus_one() {
    run_program("./program_artifacts/asm/addi_minus_one.elf");
}

#[test]
fn test_addi_max() {
    run_program("./program_artifacts/asm/addi_max.elf");
}

#[test]
fn test_addi_min() {
    run_program("./program_artifacts/asm/addi_min.elf");
}

#[test]
fn test_addi_reg() {
    run_program("./program_artifacts/asm/addi_reg.elf");
}

#[test]
fn test_addi_reg_max() {
    run_program("./program_artifacts/asm/addi_reg_max.elf");
}

#[test]
fn test_addi_reg_min() {
    run_program("./program_artifacts/asm/addi_reg_min.elf");
}

#[test]
fn test_addi_255() {
    run_program("./program_artifacts/asm/addi_255.elf");
}

#[test]
fn test_add() {
    run_program("./program_artifacts/asm/add.elf");
}

#[test]
fn test_add_neg() {
    run_program("./program_artifacts/asm/add_neg.elf");
}

#[test]
fn test_add_max() {
    run_program("./program_artifacts/asm/add_max.elf");
}

#[test]
fn test_add_max_plus_one() {
    run_program("./program_artifacts/asm/add_max_plus_one.elf");
}

#[test]
fn test_add_min() {
    run_program("./program_artifacts/asm/add_min.elf");
}

#[test]
fn test_add_min_minus_one() {
    run_program("./program_artifacts/asm/add_min_minus_one.elf");
}

#[test]
fn test_andi() {
    run_program("./program_artifacts/asm/andi.elf");
}

#[test]
fn test_andi_one() {
    run_program("./program_artifacts/asm/andi_one.elf");
}

#[test]
fn test_andi_one_and_zero() {
    run_program("./program_artifacts/asm/andi_one_and_zero.elf");
}

#[test]
fn test_andi_one_and_two() {
    run_program("./program_artifacts/asm/andi_one_and_two.elf");
}

#[test]
fn test_andi_max() {
    run_program("./program_artifacts/asm/andi_max.elf");
}

#[test]
fn test_ori() {
    run_program("./program_artifacts/asm/ori.elf");
}

#[test]
fn test_ori_one() {
    run_program("./program_artifacts/asm/ori_one.elf");
}

#[test]
fn test_ori_one_and_one() {
    run_program("./program_artifacts/asm/ori_one_and_one.elf");
}

#[test]
fn test_ori_two_and_one() {
    run_program("./program_artifacts/asm/ori_two_and_one.elf");
}

#[test]
fn test_ori_five_and_four() {
    run_program("./program_artifacts/asm/ori_five_and_four.elf");
}

#[test]
fn test_ori_three_and_five() {
    run_program("./program_artifacts/asm/ori_three_and_five.elf");
}

#[test]
fn test_ori_max() {
    run_program("./program_artifacts/asm/ori_max.elf");
}

#[test]
fn test_xori() {
    run_program("./program_artifacts/asm/xori.elf");
}

#[test]
fn test_xori_one() {
    run_program("./program_artifacts/asm/xori_one.elf");
}

#[test]
fn test_xori_one_and_one() {
    run_program("./program_artifacts/asm/xori_one_and_one.elf");
}

#[test]
fn test_xori_max() {
    run_program("./program_artifacts/asm/xori_max.elf");
}

#[test]
fn test_xori_negate() {
    run_program("./program_artifacts/asm/xori_negate.elf");
}

#[test]
fn test_slti() {
    run_program("./program_artifacts/asm/slti.elf");
}

#[test]
fn test_slti_one() {
    run_program("./program_artifacts/asm/slti_one.elf");
}

#[test]
fn test_slti_minus_one() {
    run_program("./program_artifacts/asm/slti_minus_one.elf");
}

#[test]
fn test_slti_negative() {
    run_program("./program_artifacts/asm/slti_negative.elf");
}

#[test]
fn test_slti_negative_minus() {
    run_program("./program_artifacts/asm/slti_negative_minus.elf");
}

#[test]
fn test_sltiu() {
    run_program("./program_artifacts/asm/sltiu.elf");
}

#[test]
fn test_sltiu_one() {
    run_program("./program_artifacts/asm/sltiu_one.elf");
}

#[test]
fn test_sltiu_negative() {
    run_program("./program_artifacts/asm/sltiu_negative.elf");
}

#[test]
fn test_sltiu_two_negatives() {
    run_program("./program_artifacts/asm/sltiu_two_negatives.elf");
}

#[test]
fn test_slli() {
    run_program("./program_artifacts/asm/slli.elf");
}

#[test]
fn test_slli_one() {
    run_program("./program_artifacts/asm/slli_one.elf");
}

#[test]
fn test_slli_one_one() {
    run_program("./program_artifacts/asm/slli_one_one.elf");
}

#[test]
fn test_slli_one_zero() {
    run_program("./program_artifacts/asm/slli_one_zero.elf");
}

#[test]
fn test_slli_ff_four() {
    run_program("./program_artifacts/asm/slli_ff_four.elf");
}

#[test]
fn test_slli_max() {
    run_program("./program_artifacts/asm/slli_max.elf");
}

#[test]
fn test_slli_max_half() {
    run_program("./program_artifacts/asm/slli_max_half.elf");
}

#[test]
fn test_slli_max_max() {
    run_program("./program_artifacts/asm/slli_max_max.elf");
}

#[test]
fn test_slli_not_arith() {
    run_program("./program_artifacts/asm/slli_not_arith.elf");
}

#[test]
fn test_srli() {
    run_program("./program_artifacts/asm/srli.elf");
}

#[test]
fn test_srli_one() {
    run_program("./program_artifacts/asm/srli_one.elf");
}

#[test]
fn test_srli_one_zero() {
    run_program("./program_artifacts/asm/srli_one_zero.elf");
}

#[test]
fn test_srli_one_one() {
    run_program("./program_artifacts/asm/srli_one_one.elf");
}

#[test]
fn test_srli_two_one() {
    run_program("./program_artifacts/asm/srli_two_one.elf");
}

#[test]
fn test_srli_max() {
    run_program("./program_artifacts/asm/srli_max.elf");
}

#[test]
fn test_srli_max_max() {
    run_program("./program_artifacts/asm/srli_max_max.elf");
}

#[test]
fn test_srai() {
    run_program("./program_artifacts/asm/srai.elf");
}

#[test]
fn test_srai_one() {
    run_program("./program_artifacts/asm/srai_one.elf");
}

#[test]
fn test_srai_one_one() {
    run_program("./program_artifacts/asm/srai_one_one.elf");
}

#[test]
fn test_srai_two_one() {
    run_program("./program_artifacts/asm/srai_two_one.elf");
}

#[test]
fn test_srai_max() {
    run_program("./program_artifacts/asm/srai_max.elf");
}

#[test]
fn test_srai_negative() {
    run_program("./program_artifacts/asm/srai_negative.elf");
}

#[test]
fn test_jal() {
    run_program("./program_artifacts/asm/jal.elf");
}

#[test]
fn test_jal_next() {
    run_program("./program_artifacts/asm/jal_next.elf");
}

#[test]
fn test_jal_prev() {
    run_program("./program_artifacts/asm/jal_prev.elf");
}

#[test]
fn test_jal_ret() {
    run_program("./program_artifacts/asm/jal_ret.elf");
}

#[test]
fn test_jalr() {
    run_program("./program_artifacts/asm/jalr.elf");
}

#[test]
fn test_jalr_neg() {
    run_program("./program_artifacts/asm/jalr_neg.elf");
}

#[test]
fn test_jalr_ret() {
    run_program("./program_artifacts/asm/jalr_ret.elf");
}

#[test]
fn test_jalr_odd() {
    run_program("./program_artifacts/asm/jalr_odd.elf");
}

#[test]
fn test_jalr_odd_reg() {
    run_program("./program_artifacts/asm/jalr_odd_reg.elf");
}

#[test]
fn test_bne() {
    run_program("./program_artifacts/asm/bne.elf");
}

#[test]
fn test_bne_true() {
    run_program("./program_artifacts/asm/bne_true.elf");
}

#[test]
fn test_bne_neg() {
    run_program("./program_artifacts/asm/bne_neg.elf");
}

#[test]
fn test_loop_5() {
    run_program("./program_artifacts/asm/loop_5.elf");
}

#[test]
fn test_lw_sw() {
    run_program("./program_artifacts/asm/lw_sw.elf");
}

#[test]
fn test_lw_sw_offset() {
    run_program("./program_artifacts/asm/lw_sw_offset.elf");
}

#[test]
fn test_lw_sw_offset_odd() {
    run_program("./program_artifacts/asm/lw_sw_offset_odd.elf");
}

#[test]
fn test_misalign_lh() {
    run_program("./program_artifacts/asm/misalign_lh.elf");
}

#[test]
fn test_misalign_lhu() {
    run_program("./program_artifacts/asm/misalign_lhu.elf");
}

#[test]
fn test_misalign_lw() {
    run_program("./program_artifacts/asm/misalign_lw.elf");
}

#[test]
fn test_misalign_lwu() {
    run_program("./program_artifacts/asm/misalign_lwu.elf");
}

#[test]
fn test_misalign_ld() {
    run_program("./program_artifacts/asm/misalign_ld.elf");
}

#[test]
fn test_misalign_sh() {
    run_program("./program_artifacts/asm/misalign_sh.elf");
}

#[test]
fn test_misalign_sw() {
    run_program("./program_artifacts/asm/misalign_sw.elf");
}

#[test]
fn test_misalign_sd() {
    run_program("./program_artifacts/asm/misalign_sd.elf");
}

#[test]
fn test_misaligned_pc_traps() {
    let elf_data = std::fs::read("./program_artifacts/asm/misaligned_pc.elf").unwrap();
    let program = Elf::load(&elf_data).unwrap();
    let mut executor = Executor::new(&program, vec![]).expect("Failed to create executor");
    let err = loop {
        match executor.resume() {
            Ok(Some(_)) => continue,
            Ok(None) => panic!("expected misaligned PC trap, program halted normally"),
            Err(e) => break e,
        }
    };
    assert!(
        matches!(err, ExecutorError::InstructionAddressMisaligned(2)),
        "expected InstructionAddressMisaligned(2), got {:?}",
        err
    );
}

#[test]
fn test_auipc() {
    run_program("./program_artifacts/asm/auipc.elf");
}

#[test]
fn test_auipc_offset() {
    run_program("./program_artifacts/asm/auipc_offset.elf");
}

#[test]
fn test_mul() {
    run_program("./program_artifacts/asm/mul.elf");
}

#[test]
fn test_mul_max() {
    run_program("./program_artifacts/asm/mul_max.elf");
}

#[test]
fn test_mulh_max() {
    run_program("./program_artifacts/asm/mulh_max.elf");
}

#[test]
fn test_mulhu_max() {
    run_program("./program_artifacts/asm/mulhu_max.elf");
}

#[test]
fn test_mulhsu_max() {
    run_program("./program_artifacts/asm/mulhsu_max.elf");
}

#[test]
fn test_div_zero() {
    run_program("./program_artifacts/asm/div_zero.elf");
}

#[test]
fn test_divu_zero() {
    run_program("./program_artifacts/asm/divu_zero.elf");
}

#[test]
fn test_divu() {
    run_program("./program_artifacts/asm/divu.elf");
}

#[test]
fn test_rem_zero() {
    run_program("./program_artifacts/asm/rem_zero.elf");
}

#[test]
fn test_rem() {
    run_program("./program_artifacts/asm/rem.elf");
}

#[test]
fn test_rem_overflow() {
    run_program("./program_artifacts/asm/rem_overflow.elf");
}

#[test]
fn test_remu_zero() {
    run_program("./program_artifacts/asm/remu_zero.elf");
}

#[test]
fn test_remu() {
    run_program("./program_artifacts/asm/remu.elf");
}

// ==================== W-suffix Instructions (RV64 specific) ====================

#[test]
fn test_addw() {
    run_program("./program_artifacts/asm/addw.elf");
}

#[test]
fn test_addw_pos() {
    run_program("./program_artifacts/asm/addw_pos.elf");
}

#[test]
fn test_subw() {
    run_program("./program_artifacts/asm/subw.elf");
}

#[test]
fn test_subw_overflow() {
    run_program("./program_artifacts/asm/subw_overflow.elf");
}

#[test]
fn test_addiw() {
    run_program("./program_artifacts/asm/addiw.elf");
}

#[test]
fn test_addiw_neg() {
    run_program("./program_artifacts/asm/addiw_neg.elf");
}

#[test]
fn test_sllw() {
    run_program("./program_artifacts/asm/sllw.elf");
}

#[test]
fn test_sllw_wrap() {
    run_program("./program_artifacts/asm/sllw_wrap.elf");
}

#[test]
fn test_srlw() {
    run_program("./program_artifacts/asm/srlw.elf");
}

#[test]
fn test_sraw() {
    run_program("./program_artifacts/asm/sraw.elf");
}

#[test]
fn test_slliw() {
    run_program("./program_artifacts/asm/slliw.elf");
}

#[test]
fn test_srliw() {
    run_program("./program_artifacts/asm/srliw.elf");
}

#[test]
fn test_sraiw() {
    run_program("./program_artifacts/asm/sraiw.elf");
}

#[test]
fn test_mulw() {
    run_program("./program_artifacts/asm/mulw.elf");
}

#[test]
fn test_mulw_neg() {
    run_program("./program_artifacts/asm/mulw_neg.elf");
}

#[test]
fn test_mulw_overflow() {
    run_program("./program_artifacts/asm/mulw_overflow.elf");
}

#[test]
fn test_divw() {
    run_program("./program_artifacts/asm/divw.elf");
}

#[test]
fn test_divw_zero() {
    run_program("./program_artifacts/asm/divw_zero.elf");
}

#[test]
fn test_divw_overflow() {
    run_program("./program_artifacts/asm/divw_overflow.elf");
}

#[test]
fn test_divuw_zero() {
    run_program("./program_artifacts/asm/divuw_zero.elf");
}

#[test]
fn test_divuw() {
    run_program("./program_artifacts/asm/divuw.elf");
}

#[test]
fn test_divuw_high_bit() {
    run_program("./program_artifacts/asm/divuw_high_bit.elf");
}

#[test]
fn test_remw() {
    run_program("./program_artifacts/asm/remw.elf");
}

#[test]
fn test_remw_zero() {
    run_program("./program_artifacts/asm/remw_zero.elf");
}

#[test]
fn test_remw_overflow() {
    run_program("./program_artifacts/asm/remw_overflow.elf");
}

#[test]
fn test_remuw_zero() {
    run_program("./program_artifacts/asm/remuw_zero.elf");
}

#[test]
fn test_remuw() {
    run_program("./program_artifacts/asm/remuw.elf");
}

#[test]
fn test_remuw_high_bit() {
    run_program("./program_artifacts/asm/remuw_high_bit.elf");
}

// ==================== 64-bit Load/Store ====================

#[test]
fn test_ld_sd() {
    run_program("./program_artifacts/asm/ld_sd.elf");
}

#[test]
fn test_ld_sd_offset() {
    run_program("./program_artifacts/asm/ld_sd_offset.elf");
}

#[test]
fn test_ld_sd_neg() {
    run_program("./program_artifacts/asm/ld_sd_neg.elf");
}

#[test]
fn test_lwu() {
    run_program("./program_artifacts/asm/lwu.elf");
}

#[test]
fn test_lw_sign_extend() {
    run_program("./program_artifacts/asm/lw_sign_extend.elf");
}

#[test]
fn test_lwu_vs_lw() {
    run_program("./program_artifacts/asm/lwu_vs_lw.elf");
}

// ==================== Branch Instructions ====================

#[test]
fn test_beq() {
    run_program("./program_artifacts/asm/beq.elf");
}

#[test]
fn test_beq_false() {
    run_program("./program_artifacts/asm/beq_false.elf");
}

#[test]
fn test_blt() {
    run_program("./program_artifacts/asm/blt.elf");
}

#[test]
fn test_blt_false() {
    run_program("./program_artifacts/asm/blt_false.elf");
}

#[test]
fn test_bge() {
    run_program("./program_artifacts/asm/bge.elf");
}

#[test]
fn test_bge_greater() {
    run_program("./program_artifacts/asm/bge_greater.elf");
}

#[test]
fn test_bge_false() {
    run_program("./program_artifacts/asm/bge_false.elf");
}

#[test]
fn test_bltu() {
    run_program("./program_artifacts/asm/bltu.elf");
}

#[test]
fn test_bltu_neg() {
    run_program("./program_artifacts/asm/bltu_neg.elf");
}

#[test]
fn test_bgeu() {
    run_program("./program_artifacts/asm/bgeu.elf");
}

#[test]
fn test_bgeu_neg() {
    run_program("./program_artifacts/asm/bgeu_neg.elf");
}

// ==================== LUI Instruction ====================

#[test]
fn test_lui() {
    run_program("./program_artifacts/asm/lui.elf");
}

#[test]
fn test_lui_neg() {
    run_program("./program_artifacts/asm/lui_neg.elf");
}

#[test]
fn test_lui_max() {
    run_program("./program_artifacts/asm/lui_max.elf");
}

// ==================== 64-bit Edge Cases ====================

#[test]
fn test_add_64bit() {
    run_program("./program_artifacts/asm/add_64bit.elf");
}

#[test]
fn test_slli_64() {
    run_program("./program_artifacts/asm/slli_64.elf");
}

#[test]
fn test_slli_63() {
    run_program("./program_artifacts/asm/slli_63.elf");
}

#[test]
fn test_srli_64() {
    run_program("./program_artifacts/asm/srli_64.elf");
}

#[test]
fn test_srai_64() {
    run_program("./program_artifacts/asm/srai_64.elf");
}

#[test]
fn test_mul_64bit() {
    run_program("./program_artifacts/asm/mul_64bit.elf");
}

#[test]
fn test_div_overflow() {
    run_program("./program_artifacts/asm/div_overflow.elf");
}

#[test]
fn test_mulh_64bit() {
    run_program("./program_artifacts/asm/mulh_64bit.elf");
}

// ==================== SUB Register-Register ====================

#[test]
fn test_sub() {
    run_program("./program_artifacts/asm/sub.elf");
}

#[test]
fn test_sub_neg_result() {
    run_program("./program_artifacts/asm/sub_neg_result.elf");
}

#[test]
fn test_sub_64bit() {
    run_program("./program_artifacts/asm/sub_64bit.elf");
}

#[test]
fn test_sub_underflow() {
    run_program("./program_artifacts/asm/sub_underflow.elf");
}

// ==================== Keccak Precompile ====================

#[test]
fn test_keccak() {
    // Runs keccak-f[1600] on a zeroed state and commits the 200-byte result.
    // Expected output is the FIPS-202 zero-input KAT.
    let elf_data = std::fs::read("./program_artifacts/asm/test_keccak.elf").unwrap();
    let program = Elf::load(&elf_data).unwrap();
    let executor = Executor::new(&program, vec![]).expect("Failed to create executor");
    let result = executor.run().expect("Failed to run program");

    let expected_state: [u64; 25] = [
        0xF1258F7940E1DDE7,
        0x84D5CCF933C0478A,
        0xD598261EA65AA9EE,
        0xBD1547306F80494D,
        0x8B284E056253D057,
        0xFF97A42D7F8E6FD4,
        0x90FEE5A0A44647C4,
        0x8C5BDA0CD6192E76,
        0xAD30A6F71B19059C,
        0x30935AB7D08FFC64,
        0xEB5AA93F2317D635,
        0xA9A6E6260D712103,
        0x81A57C16DBCF555F,
        0x43B831CD0347C826,
        0x01F22F1A11A5569F,
        0x05E5635A21D9AE61,
        0x64BEFEF28CC970F2,
        0x613670957BC46611,
        0xB87C5A554FD00ECB,
        0x8C3EE88A1CCF32C8,
        0x940C7922AE3A2614,
        0x1841F924A2C509E4,
        0x16F53526E70465C2,
        0x75F644E97F30A13B,
        0xEAF1FF7B5CECA249,
    ];
    let mut expected_bytes = Vec::with_capacity(200);
    for lane in expected_state {
        expected_bytes.extend_from_slice(&lane.to_le_bytes());
    }
    assert_eq!(result.return_values.memory_values, expected_bytes);
    assert_eq!(result.return_values.register_values.0, 0);
}

#[test]
fn test_run_epochs_splits_execution_into_n_cycle_epochs() {
    let elf_data = std::fs::read("./program_artifacts/asm/basic_program.elf").unwrap();
    let program = Elf::load(&elf_data).unwrap();

    // Reference: full single-pass run.
    let full = Executor::new(&program, vec![]).unwrap().run().unwrap();

    // Pick an epoch size that splits this program into a few epochs, whatever
    // its exact length.
    let total_cycles = full.logs.len();
    assert!(total_cycles >= 2);
    let epoch_size = (total_cycles / 3).max(1);

    let epochs = Executor::new(&program, vec![])
        .unwrap()
        .run_epochs(epoch_size)
        .unwrap();

    // The program is long enough to span several epochs.
    assert!(epochs.len() >= 2);

    // Concatenated epoch logs reproduce the full run's instruction stream.
    let concat: Vec<u64> = epochs
        .iter()
        .flat_map(|e| e.logs.iter().map(|l| l.current_pc))
        .collect();
    let expected: Vec<u64> = full.logs.iter().map(|l| l.current_pc).collect();
    assert_eq!(concat, expected);

    // Every epoch except the last runs exactly `epoch_size` cycles.
    for epoch in &epochs[..epochs.len() - 1] {
        assert_eq!(epoch.logs.len(), epoch_size);
    }
    let last = epochs.last().unwrap();
    assert!(!last.logs.is_empty() && last.logs.len() <= epoch_size);

    // The program finished, so the final epoch's boundary pc is 0.
    assert_eq!(last.end_pc, 0);
}
