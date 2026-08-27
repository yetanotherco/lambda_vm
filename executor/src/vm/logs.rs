/// Log containing the execution state for one instruction
/// Uses zero as default value if the instruction doesn't use either of src1, src2 or dst
/// Note that values written to dst register zero will be ignored
/// In case of JALR instruction: value of base will be at src1_val
/// In case of Store instruction: value of base will be at src1_val and value to be stored will be at src2_val
/// In case of Load instruction: value of base will be at src1_val
/// The instruction itself is not stored here - use current_pc to look it up from the predecoded instructions map
///
/// For ECALL instructions, these fields are repurposed (since decode sets read_register1/2=false,
/// write_register=false, so src/dst are unconstrained):
/// - `src1_val` = syscall number (from x17): 64=Commit, 93=Halt, etc.
/// - `src2_val` = Commit: buf_addr (x11); Keccak: state_addr; ECSM: addr_xG;
///   Hint: input addr; DMA memcpy: src. 0 for every other syscall.
/// - `dst_val` = Commit: count (x12); ECSM: addr_k; Hint: output addr;
///   DMA memcpy: byte count. 0 for every other syscall, Keccak included.
#[derive(Debug, Clone)]
pub struct Log {
    /// PC before instruction execution (use this to look up the instruction)
    pub current_pc: u64,
    /// PC after instruction execution
    pub next_pc: u64,
    /// Value of src1 register before execution (if used by the instruction).
    /// For ECALL: syscall number from x17.
    pub src1_val: u64,
    /// Value of src2 register before execution (if used by the instruction).
    /// For ECALL: see the per-syscall table above.
    pub src2_val: u64,
    /// Value of dst register after execution (if used by the instruction).
    /// For ECALL: see the per-syscall table above.
    pub dst_val: u64,
}
