use crate::cpu::instruction::decoding::Instruction;

/// Log containing the executed instruction and the new value of the updated register
pub struct Log {
    pub instruction: Instruction,
    pub updated_register_value: u32,
    pub updated_pc: u32,
}
