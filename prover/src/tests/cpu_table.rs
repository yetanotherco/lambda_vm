use lambdaworks_math::field::{
    element::FieldElement, fields::fft_friendly::babybear_u32::Babybear31PrimeField,
};

type FE = FieldElement<Babybear31PrimeField>;

// In this example we build a cpu table with four rows. Each row has the following instruction:
// ADD, LOAD, STORE, SUB.
// ADD: rv1 + rv2.
// LOAD: Copy value from address `rs1 + imm` to `rsd`.
// STORE: Copy rv2 to address `rs1 + imm`.
// SUB: rv2 - rv1.
pub fn build_cpu_columns_example() -> Vec<Vec<FE>> {
    let mut columns = Vec::new();
    // Timestamp: A word2L column containing the values 4 * i for i = 1,...
    // Column index: 0
    let timestamp_1 = vec![
        FE::from(&4u32),
        FE::from(&8u32),
        FE::from(&12u32),
        FE::from(&16u32),
    ];
    // Column index: 1
    let timestamp_2 = vec![FE::zero(); 4];

    columns.push(timestamp_1);
    columns.push(timestamp_2);

    // ----- 30 uncompressed decode columns -----

    // Word2L pc.
    // Column index: 2
    let pc_1 = vec![
        FE::from(&4u32),
        FE::from(&8u32),
        FE::from(&12u32),
        FE::from(&16u32),
    ];
    // Column index: 3
    let pc_2 = vec![FE::zero(); 4];

    columns.push(pc_1);
    columns.push(pc_2);

    // Index of source register 1.
    // Column index: 4
    let rs_1 = vec![FE::from(&1), FE::from(&2), FE::from(&3), FE::from(&4)];
    columns.push(rs_1);

    // Index of source register 2.
    // Column index: 5
    let rs_2 = vec![FE::from(&5), FE::from(&6), FE::from(&7), FE::from(&8)];
    columns.push(rs_2);

    // Index of destination register.
    // Column index: 6
    let rd = vec![FE::from(&9), FE::from(&10), FE::from(&11), FE::from(&12)];
    columns.push(rd);

    // Flag: Whether the result should be written to `rd`.
    // In our example:
    // ADD and SUB write to `rd` (See Page 29).
    // LOAD writes and STORE doesn't `rd` (See Page 34).
    // https://docs.riscv.org/reference/isa/_attachments/riscv-unprivileged.pdf>
    // Column index: 7
    let write_register = vec![FE::one(), FE::one(), FE::zero(), FE::one()];
    columns.push(write_register);

    // Does the memory access (read or write) touch at least 2 bytes.
    // Flag.
    // Column index: 8
    let memory_2_bytes = vec![FE::zero(), FE::zero(), FE::zero(), FE::zero()];
    columns.push(memory_2_bytes);

    // Does the memory access (read or write) touch 4 bytes.
    // Flag.
    // Column index: 9
    let memory_4_bytes = vec![FE::zero(), FE::zero(), FE::one(), FE::zero()];
    columns.push(memory_4_bytes);

    // The 32-bit version of the immediate.
    // In the example:
    // ADD and SUB between registers: No immedaite.
    // LOAD and STORE: The address is obtained by adding rs1 to the sign-extended 12-bit offset immediate.
    // Column index: 10
    let imm_1 = vec![
        FE::zero(),
        FE::one(),
        FE::from(&((1 << 12) - 1)),
        FE::zero(),
    ];
    // Column index: 11
    let imm_2 = vec![FE::zero(); 4];

    columns.push(imm_1);
    columns.push(imm_2);

    // Flag to indicate signed or unsigned input interpretation.
    // Column index: 12
    let signed = vec![FE::zero(), FE::one(), FE::one(), FE::zero()];
    columns.push(signed);

    // Flag: multi-purpose selector used by different ALU operations for different purposes.
    // In our example: Currently is not used for neither ADD, SUB, LOAD or STORE.
    // Column index: 13
    let mp_selector = vec![FE::zero(), FE::zero(), FE::zero(), FE::zero()];
    columns.push(mp_selector);

    // Flag that selects which output of MUL (lo/hi) or DIV (quo/rem) is wanted.
    // Column index: 14
    let muldiv_selector = vec![FE::zero(), FE::zero(), FE::zero(), FE::zero()];
    columns.push(muldiv_selector);

    // One-hot 17 columns of flags:
    // Instructions in this example: add, load, store, sub.
    // Column index: 15
    let add = vec![FE::one(), FE::zero(), FE::zero(), FE::zero()];
    columns.push(add);
    // Column index: 16
    let sub = vec![FE::zero(), FE::zero(), FE::zero(), FE::one()];
    columns.push(sub);
    // Column index: 17
    let slt = vec![FE::zero(), FE::zero(), FE::zero(), FE::zero()];
    columns.push(slt);
    // Column index: 18
    let and = vec![FE::zero(), FE::zero(), FE::zero(), FE::zero()];
    columns.push(and);
    // Column index: 19
    let or = vec![FE::zero(), FE::zero(), FE::zero(), FE::zero()];
    columns.push(or);
    // Column index: 20
    let xor = vec![FE::zero(), FE::zero(), FE::zero(), FE::zero()];
    columns.push(xor);
    // Column index: 21
    let sl = vec![FE::zero(), FE::zero(), FE::zero(), FE::zero()];
    columns.push(sl);
    // Column index: 22
    let sr = vec![FE::zero(), FE::zero(), FE::zero(), FE::zero()];
    columns.push(sr);
    // Column index: 23
    let jalr = vec![FE::zero(), FE::zero(), FE::zero(), FE::zero()];
    columns.push(jalr);
    // Column index: 24
    let beq = vec![FE::zero(), FE::zero(), FE::zero(), FE::zero()];
    columns.push(beq);
    // Column index: 25
    let blt = vec![FE::zero(), FE::zero(), FE::zero(), FE::zero()];
    columns.push(blt);
    // Column index: 26
    let load = vec![FE::zero(), FE::one(), FE::zero(), FE::zero()];
    columns.push(load);
    // Column index: 27
    let store = vec![FE::zero(), FE::zero(), FE::one(), FE::zero()];
    columns.push(store);
    // Column index: 28
    let mul = vec![FE::zero(), FE::zero(), FE::zero(), FE::zero()];
    columns.push(mul);
    // Column index: 29
    let divrem = vec![FE::zero(), FE::zero(), FE::zero(), FE::zero()];
    columns.push(divrem);
    // Column index: 30
    let ecall = vec![FE::zero(), FE::zero(), FE::zero(), FE::zero()];
    columns.push(ecall);
    // Column index: 31
    let ebreak = vec![FE::zero(), FE::zero(), FE::zero(), FE::zero()];
    columns.push(ebreak);

    // ----- End Decode Columns -----

    // Column index: 32
    let next_pc_1 = vec![FE::from(8), FE::from(12), FE::from(16), FE::from(20)];
    // Column index: 333
    let next_pc_2 = vec![FE::zero(); 4];

    columns.push(next_pc_1);
    columns.push(next_pc_2);

    // rv1 Word4L
    // Column index: 34
    let rv1_1 = vec![
        FE::from(&10u32),
        FE::from(&20u32),
        FE::from(&30u32),
        FE::from(&40u32),
    ];
    // Column index: 35
    let rv1_2 = vec![FE::zero(); 4];
    // Column index: 36
    let rv1_3 = vec![FE::zero(); 4];
    // Column index: 37
    let rv1_4 = vec![FE::zero(); 4];

    columns.push(rv1_1);
    columns.push(rv1_2);
    columns.push(rv1_3);
    columns.push(rv1_4);

    // rv2 Word4L
    // Column index: 38
    let rv2_1 = vec![
        FE::from(&50u32),
        FE::from(&60u32),
        FE::from(&70u32),
        FE::from(&10u32),
    ];
    // Column index: 39
    let rv2_2 = vec![FE::zero(); 4];
    // Column index: 10
    let rv2_3 = vec![FE::zero(); 4];
    // Column index: 41
    let rv2_4 = vec![FE::zero(); 4];

    columns.push(rv2_1);
    columns.push(rv2_2);
    columns.push(rv2_3);
    columns.push(rv2_4);

    // rvd Word2L
    // Column index: 42
    let rvd_1 = vec![
        FE::from(&60u32), // rv1 + rv2 = 10 + 50
        FE::from(&1u32),  // LOAD: copy a value from memory to `rd`.
        FE::zero(),
        FE::from(&30u32), // rv1 - rv2 = 40 - 10
    ];
    // Column index: 43
    let rvd_2 = vec![FE::zero(); 4];

    columns.push(rvd_1);
    columns.push(rvd_2);

    // The second argument of the (ALU) operation being performed; a multiplex between rv2 and imm.
    // Definition: (1 - STORE - LOAD)·rv2 + (1 - BEQ - BLT)·imm
    // Column index: 44
    let arg2_1 = vec![
        FE::from(&50),              // ADD -> rv2
        FE::one(),                  // LOAD -> imm
        FE::from(&((1 << 12) - 1)), // STORE -> imm
        FE::from(&10),              // SUB -> rv2
    ];
    // Column index: 45
    let arg2_2 = vec![FE::zero(); 4];
    // Column index: 46
    let arg2_3 = vec![FE::zero(); 4];
    // Column index: 47
    let arg2_4 = vec![FE::zero(); 4];

    columns.push(arg2_1);
    columns.push(arg2_2);
    columns.push(arg2_3);
    columns.push(arg2_4);

    // The word2L ALU result.
    // Column index: 48
    let res_1 = vec![
        FE::from(&60u32),   // rv1 + rv2 = 10 + 50
        FE::from(&21u32),   // rv1 + imm = 20 + 1.
        FE::from(&4125u32), // rv1 + imm = 30 + 2^12 - 1 = 4125.
        FE::from(&30u32),   // rv2 - rv1 = 40 - 10 = 30
    ];
    // Column index: 49
    let res_2 = vec![FE::zero(); 4];
    // Column index: 50
    let res_3 = vec![FE::zero(); 4];
    // Column index: 51
    let res_4 = vec![FE::zero(); 4];

    columns.push(res_1);
    columns.push(res_2);
    columns.push(res_3);
    columns.push(res_4);

    // Flag: Wether rv1 and arg2 are equal.
    // Column index: 52
    let is_equal = vec![FE::zero(), FE::zero(), FE::zero(), FE::zero()];
    columns.push(is_equal);

    // Flag: Whether a branch is taken.
    // Column index: 53
    let branch_cond = vec![FE::zero(), FE::zero(), FE::zero(), FE::zero()];
    columns.push(branch_cond);

    columns
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::air::cpu_air::{CPUTableAIR, build_cpu_trace};
    use lambdaworks_crypto::fiat_shamir::default_transcript::DefaultTranscript;
    use lambdaworks_math::field::fields::fft_friendly::quartic_babybear_u32::Degree4BabyBearU32ExtensionField;
    use stark_platinum_prover::{
        proof::options::ProofOptions,
        prover::{IsStarkProver, Prover},
        verifier::{IsStarkVerifier, Verifier},
    };
    #[test]
    fn test_prove_cpu_table() {
        let columns = build_cpu_columns_example();
        let mut trace = build_cpu_trace(columns);
        let proof_options = ProofOptions::default_test_options();

        let proof = Prover::<CPUTableAIR>::prove(
            &mut trace,
            &(),
            &proof_options,
            DefaultTranscript::<Degree4BabyBearU32ExtensionField>::new(&[]),
        )
        .unwrap();

        assert!(Verifier::<CPUTableAIR>::verify(
            &proof,
            &(),
            &proof_options,
            DefaultTranscript::<Degree4BabyBearU32ExtensionField>::new(&[]),
        ));
    }
}
