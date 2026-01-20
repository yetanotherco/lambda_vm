use crate::vm::instruction::decoding::Instruction;

/// Log containing the executed instruction and the new value of the updated register
/// Uses zero as default value if the instruction doesn't use either of src1, src2 or dst
/// Note that values written to dst register zero will be ignored
/// In case of JALR instruction: value of base will be at src1_val
/// In case of Store instruction: value of base will be at src1_val and value to be stored will be at src2_val
/// In case of Load instruction: value of base will be at src1_val
#[derive(Debug, Clone)]
pub struct Log {
    /// Executed Instruction
    pub instruction: Instruction,
    /// PC before instruction execution
    pub current_pc: u64,
    /// PC after instruction execution
    pub next_pc: u64,
    /// Value of src1 register before execution (if used by the instruction)
    pub src1_val: u64,
    /// Value of src2 register before execution (if used by the instruction)
    pub src2_val: u64,
    /// Value of dst register after execution (if used by the instruction)
    pub dst_val: u64,
}
