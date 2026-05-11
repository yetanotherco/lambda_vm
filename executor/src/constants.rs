/// VM memory layout constants shared between prover and verifier code paths.
///
/// These live outside `vm/` because the verifier needs them even when the full
/// VM executor is not compiled in (e.g. inside a RISC-V guest verifying a proof).

/// Initial value of the stack pointer register (SP, x2).
/// 64-bit max, aligned to 16 bytes per RV64 ABI.
pub const STACK_TOP: u64 = 0xFFFFFFFFFFFFFFF0;

/// Maximum byte length of the private-input region.
pub const MAX_PRIVATE_INPUT_SIZE: u64 = 6700000;

/// Memory address where the private-input region starts.
/// Layout: 4-byte LE length prefix at this address, then payload at +4.
pub const PRIVATE_INPUT_START_INDEX: u64 = 0xFF000000;
