//! VM memory layout constants shared between prover and verifier code paths.
//!
//! These live outside `vm/` because the verifier needs them even when the full
//! VM executor is not compiled in (e.g. inside a RISC-V guest verifying a proof).

/// Initial value of the stack pointer register (SP, x2).
/// 64-bit max, aligned to 16 bytes per RV64 ABI.
pub const STACK_TOP: u64 = 0xFFFFFFFFFFFFFFF0;

/// Maximum byte length of the private-input region.
///
/// Bumped from 6.7 MB to 64 MB to accommodate serialized STARK proofs as
/// private input for the naive recursion experiment.
pub const MAX_PRIVATE_INPUT_SIZE: u64 = 64 * 1024 * 1024;

/// Memory address where the private-input region starts.
/// Layout: 4-byte LE length prefix at this address, then payload at +4.
pub const PRIVATE_INPUT_START_INDEX: u64 = 0xFF000000;

/// Syscall number for the Keccak-f[1600] precompile.
pub const KECCAK_SYSCALL_NUMBER: u64 = u64::MAX - 1;

/// Syscall number for the Goldilocks Fp3 multiply precompile.
/// Multiplies two cubic extension field elements (x³ - 2) over Goldilocks in O(1) VM cycles.
pub const FP3_MUL_SYSCALL_NUMBER: u64 = u64::MAX - 2;

/// Syscall number for the Goldilocks Fp3 fused multiply-add precompile.
/// Computes `acc += lhs × rhs` for Fp3 elements in one VM cycle.
/// ABI: a7=FP3_FMA_SYSCALL_NUMBER, a0=acc_ptr (in/out, [u64;3]), a1=lhs_ptr ([u64;3]), a2=rhs_ptr ([u64;3])
pub const FP3_FMA_SYSCALL_NUMBER: u64 = u64::MAX - 3;

/// Syscall number for the Goldilocks scalar×Fp3 fused multiply-add precompile.
/// Computes `acc += scalar × fp3_rhs` where scalar is a single Goldilocks element.
/// Costs 3 Goldilocks muls (vs 9 for Fp3×Fp3) while still issuing one ecall.
/// ABI: a7=FP3_SCALAR_FMA_SYSCALL_NUMBER, a0=acc_ptr (in/out, [u64;3]), a1=scalar_ptr ([u64;1]), a2=rhs_ptr ([u64;3])
pub const FP3_SCALAR_FMA_SYSCALL_NUMBER: u64 = u64::MAX - 4;

/// Syscall number for the Goldilocks scalar-Fp3 dot product precompile.
/// Computes `acc += scalars[0]*fp3[0] + scalars[1]*fp3[1] + ... + scalars[n-1]*fp3[n-1]`
/// in a single ecall. Replaces n calls to FP3_SCALAR_FMA_SYSCALL.
/// ABI: a7=FP3_SCALAR_DOT_SYSCALL_NUMBER, a0=acc_ptr (in/out, [u64;3]),
///      a1=scalars_ptr ([u64;n]), a2=fp3_ptr ([u64;3*n]), a3=n (count)
pub const FP3_SCALAR_DOT_SYSCALL_NUMBER: u64 = u64::MAX - 5;

/// Syscall number for the Goldilocks Fp3-Fp3 dot product precompile.
/// Computes `acc += lhs[0]*rhs[0] + lhs[1]*rhs[1] + ... + lhs[n-1]*rhs[n-1]`
/// in a single ecall. Replaces n calls to FP3_FMA_SYSCALL.
/// ABI: a7=FP3_DOT_SYSCALL_NUMBER, a0=acc_ptr (in/out, [u64;3]),
///      a1=lhs_ptr ([u64;3*n]), a2=rhs_ptr ([u64;3*n]), a3=n (count)
pub const FP3_DOT_SYSCALL_NUMBER: u64 = u64::MAX - 6;

/// Round constants for Keccak-f[1600] (24 rounds).
pub const KECCAK_RC: [u64; 24] = [
    0x0000000000000001,
    0x0000000000008082,
    0x800000000000808A,
    0x8000000080008000,
    0x000000000000808B,
    0x0000000080000001,
    0x8000000080008081,
    0x8000000000008009,
    0x000000000000008A,
    0x0000000000000088,
    0x0000000080008009,
    0x000000008000000A,
    0x000000008000808B,
    0x800000000000008B,
    0x8000000000008089,
    0x8000000000008003,
    0x8000000000008002,
    0x8000000000000080,
    0x000000000000800A,
    0x800000008000000A,
    0x8000000080008081,
    0x8000000000008080,
    0x0000000080000001,
    0x8000000080008008,
];

/// Rotation offsets R[x][y] for the rho step of Keccak-f[1600].
pub const KECCAK_RHO: [[u32; 5]; 5] = [
    [0, 36, 3, 41, 18],
    [1, 44, 10, 45, 2],
    [62, 6, 43, 15, 61],
    [28, 55, 25, 21, 56],
    [27, 20, 39, 8, 14],
];
