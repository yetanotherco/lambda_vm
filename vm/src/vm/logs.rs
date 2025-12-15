use crate::vm::instruction::decoding::Instruction;

/// Log containing the executed instruction and the new value of the updated register
/// In case of JALR instruction: value of base will be at src1_val
/// In case of Store instruction: value of base will be at src1_val and value to be stored will be at src2_val
/// In case of Load instruction: value of base will be at src1_val
pub struct Log {
    pub instruction: Instruction,
    pub current_pc: u32,
    pub next_pc: u32,
    pub src1_val: u32,
    pub src2_val: u32,
    pub dst_val: u32,
}
