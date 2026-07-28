#[cfg(target_arch = "riscv64")]
use core::arch::asm;

/// Memory-mapped private input region start address.
/// Layout: 4-byte LE length prefix at this address, data at +4.
/// The host pre-loads the input; the guest reads directly (no ecall).
/// Must match `executor::vm::memory::PRIVATE_INPUT_START_INDEX`.
#[cfg(target_arch = "riscv64")]
pub const PRIVATE_INPUT_START: usize = 0xFF000000;

/// Maximum private-input length the guest will read, in bytes (512 MiB).
/// The host caps stored input at this size in `Memory::store_private_inputs`,
/// so an honest length prefix is always `<=` this bound; a larger value can only
/// come from a malformed or forged prefix. The reader clamps to this cap so a
/// bogus length can never make the guest fabricate an arbitrarily long slice.
/// Must match `executor::vm::memory::MAX_PRIVATE_INPUT_SIZE`.
#[cfg(target_arch = "riscv64")]
const MAX_PRIVATE_INPUT_SIZE: usize = 512 * 1024 * 1024;

#[cfg(target_arch = "riscv64")]
pub enum SyscallNumbers {
    Print = 1,
    Panic = 2,
    Commit = 64,
    Halt = 93,
}

/// Syscall number for KeccakPermute (u64::MAX - 1).
#[cfg(target_arch = "riscv64")]
const KECCAK_SYSCALL_NUMBER: usize = usize::MAX - 1;

/// Syscall number for the x-only ECSM secp256k1 scalar-multiply accelerator (-11 as usize).
#[cfg(target_arch = "riscv64")]
const ECSM_SYSCALL_NUMBER: usize = usize::MAX - 10;

/// Syscall number for the affine ECSM variant (full point in/out).
/// Must match `executor::...::execution::ECSM_AFFINE_SYSCALL_NUMBER` (u64::MAX - 11).
#[cfg(target_arch = "riscv64")]
const ECSM_AFFINE_SYSCALL_NUMBER: usize = usize::MAX - 11;

/// Syscall number for the non-constraining Hint ecall (BENCH ONLY).
/// Must match `executor::...::execution::HINT_SYSCALL_NUMBER` (u64::MAX - 20).
#[cfg(target_arch = "riscv64")]
const HINT_SYSCALL_NUMBER: usize = usize::MAX - 20;

/// Hint selectors passed in `a0` (must match the executor's `HINT_*`).
pub const HINT_FIELD_INV: usize = 0;
pub const HINT_SCALAR_INV: usize = 1;
pub const HINT_FIELD_SQRT: usize = 2;

/// No-op. The `Print` ecall (a7=1) has no receiver on the Ecall bus, so emitting
/// it makes the LogUp bus unbalance and the proof fail to verify. Printing isn't
/// needed in provable programs, so `print_string` does nothing on every target.
/// Keep it as a no-op rather than deleting call sites: that way no guest path can
/// ever reintroduce an unmatched Print ecall. (See `SyscallNumbers::Print`.)
pub fn print_string(_s: &str) {}

/// # Safety
///
/// This function should not be called by the user
/// It is only for rust std internal uses
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sys_write(_fildes: i32, buf: *const u8, size: usize) -> isize {
    print_string("sys_write called\n");
    let content = unsafe { core::slice::from_raw_parts(buf, size) };
    print_string(&("SYS_WRITE: ".to_owned() + str::from_utf8(content).unwrap_or("<invalid utf8>"))); // Does the print of the sys write
    size.try_into().unwrap_or(-1)
}

#[cfg(target_arch = "riscv64")]
/// # Safety
///
/// This function should not be called by the user
/// It is only for rust std internal uses
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sys_panic(msg_ptr: *const u8, len: usize) {
    print_string("Sys panic called\n");
    unsafe {
        asm!(
            "ecall",
            in("a0") msg_ptr,
            in("a1") len,
            in("a7") SyscallNumbers::Panic as usize,
        )
    }
}

#[cfg(target_arch = "riscv64")]
pub fn commit(slice: &[u8]) {
    unsafe {
        asm!(
            "ecall",
            in("a0") 1usize,
            in("a1") slice.as_ptr(),
            in("a2") slice.len(),
            in("a7") SyscallNumbers::Commit as usize,
        )
    }
}

/// Read private input bytes from the memory-mapped region at
/// `PRIVATE_INPUT_START = 0xFF000000`.
///
/// The host pre-loads the input before execution; this function reads the
/// 4-byte LE length prefix and then copies the data bytes into a new `Vec`.
/// No ecall is performed — it's a plain memory read (ZisK-style).
#[cfg(target_arch = "riscv64")]
pub fn get_private_input() -> Vec<u8> {
    // Copy the borrowed private-input bytes into an owned `Vec`. The raw-pointer
    // read (length prefix + data slice) and its single `unsafe` block live in
    // `get_private_input_slice`, so the memory layout is defined in one place.
    get_private_input_slice().to_vec()
}

#[cfg(not(target_arch = "riscv64"))]
pub fn get_private_input() -> Vec<u8> {
    unimplemented!("syscalls are only implemented for riscv64 targets");
}

/// Borrow the private input bytes in place from the memory-mapped region —
/// no copy, no allocation. Same layout as [`get_private_input`]; the returned
/// slice starts at `PRIVATE_INPUT_START + 4` (a 4-aligned address) and lives
/// for the whole execution (the host never remaps the region).
#[cfg(target_arch = "riscv64")]
pub fn get_private_input_slice() -> &'static [u8] {
    // SAFETY: The host pre-loads private input at PRIVATE_INPUT_START before
    // execution and never remaps it afterward, so the returned slice is valid
    // for the `'static` lifetime of the guest's single-threaded execution
    // region, which stays mapped and unmodified for the whole execution.
    let len_ptr = PRIVATE_INPUT_START as *const u32;
    // Clamp the prover-written length prefix to `MAX_PRIVATE_INPUT_SIZE`. An
    // honest prefix (written by the host, which caps stored input at this size)
    // is always within bound, so clamping never changes behavior for real
    // inputs — it only bounds the slice length when a malformed or forged prefix
    // claims more, keeping the read deterministic.
    let len = (unsafe { core::ptr::read_volatile(len_ptr) } as usize).min(MAX_PRIVATE_INPUT_SIZE);
    let data_ptr = (PRIVATE_INPUT_START + 4) as *const u8;
    unsafe { core::slice::from_raw_parts(data_ptr, len) }
}

#[cfg(not(target_arch = "riscv64"))]
pub fn get_private_input_slice() -> &'static [u8] {
    unimplemented!("syscalls are only implemented for riscv64 targets");
}

#[cfg(target_arch = "riscv64")]
pub fn sys_halt() -> ! {
    // NOTE: no print_string here — the Print ecall is unmatched on the Ecall bus
    // and would cause a verification failure.
    unsafe {
        asm!(
            "ecall",
            in("a0") 0usize, // exit_code = 0 (enforced by HALT read on x10)
            in("a7") SyscallNumbers::Halt as usize,
        );
    }
    unreachable!()
}

#[cfg(not(target_arch = "riscv64"))]
pub fn sys_halt() -> ! {
    unimplemented!("syscalls are only implemented for riscv64 targets");
}

#[cfg(target_arch = "riscv64")]
/// Apply the Keccak-f[1600] permutation to a 25-element u64 state in-place.
pub fn keccak_permute(state: &mut [u64; 25]) {
    unsafe {
        asm!(
            "ecall",
            in("a0") state.as_mut_ptr(),
            in("a7") KECCAK_SYSCALL_NUMBER,
        )
    }
}

#[cfg(not(target_arch = "riscv64"))]
/// Apply the Keccak-f[1600] permutation to a 25-element u64 state in-place.
pub fn keccak_permute(_state: &mut [u64; 25]) {
    unimplemented!("syscalls are only implemented for riscv64 targets");
}

#[cfg(target_arch = "riscv64")]
/// Compute `xR = (k·G)_x` on secp256k1 via the ECSM accelerator. All values are 32-byte
/// little-endian. Requires `0 < k < N` and a canonical valid `xG` curve coordinate.
/// `xG` and `k` must not overlap; `xR` may alias either input.
pub fn ecsm_mul(xr: &mut [u8; 32], xg: &[u8; 32], k: &[u8; 32]) {
    unsafe {
        asm!(
            "ecall",
            in("a0") xr.as_mut_ptr(), // x10 = address to write xR
            in("a1") xg.as_ptr(),     // x11 = address of xG
            in("a2") k.as_ptr(),      // x12 = address of k
            in("a7") ECSM_SYSCALL_NUMBER,
        )
    }
}

/// AFFINE: compute `k·(xG, yG)` on secp256k1 and write BOTH result coordinates into a
/// contiguous 64-byte buffer (`xR` at `out[0..32]`, `yR` at `out[32..64]`). The input is
/// the full affine point as a contiguous 64-byte buffer (`xG` at `in[0..32]`, `yG` at
/// `in[32..64]`); `k` is 32 bytes. All values 32-byte little-endian. Passing the full point
/// (not just `xG`) means the returned `yR` is the y of the caller's actual point — no
/// parity convention or caller-side sign flip. Lets ECDSA recovery avoid the second
/// `(k+1)·P` query and the x-only y-reconstruction.
#[cfg(target_arch = "riscv64")]
pub fn ecsm_mul_affine(out: &mut [u8; 64], input: &[u8; 64], k: &[u8; 32]) {
    unsafe {
        asm!(
            "ecall",
            in("a0") out.as_mut_ptr(),  // x10 = address to write [xR‖yR] (64 bytes)
            in("a1") input.as_ptr(),    // x11 = address of [xG‖yG] (64 bytes)
            in("a2") k.as_ptr(),        // x12 = address of k
            in("a7") ECSM_AFFINE_SYSCALL_NUMBER,
        )
    }
}

#[cfg(not(target_arch = "riscv64"))]
pub fn ecsm_mul_affine(_out: &mut [u8; 64], _input: &[u8; 64], _k: &[u8; 32]) {
    unimplemented!("syscalls are only implemented for riscv64 targets");
}

#[cfg(not(target_arch = "riscv64"))]
/// Compute `xR = (k·G)_x` on secp256k1 via the ECSM accelerator (32-byte little-endian values).
pub fn ecsm_mul(_xr: &mut [u8; 32], _xg: &[u8; 32], _k: &[u8; 32]) {
    unimplemented!("syscalls are only implemented for riscv64 targets");
}

/// BENCH ONLY. Ask the host for a non-constraining hint (modular inverse/sqrt).
/// `hint_id` selects the operation ([`HINT_FIELD_INV`]/[`HINT_SCALAR_INV`]/
/// [`HINT_FIELD_SQRT`]); `input`/`out` are 32-byte **big-endian** field/scalar
/// elements — k256's own serialization, so consumers pass `to_bytes()` straight
/// through. Note this differs from [`ecsm_mul`], which is little-endian.
/// The result is UNVERIFIED — the caller MUST check it in-guest
/// (e.g. `x·inv == 1`), since this ecall adds no correctness constraint.
#[cfg(target_arch = "riscv64")]
pub fn hint(hint_id: usize, out: &mut [u8; 32], input: &[u8; 32]) {
    unsafe {
        asm!(
            "ecall",
            in("a0") hint_id,           // x10 = hint selector
            in("a1") input.as_ptr(),    // x11 = input address (32-byte BE)
            in("a2") out.as_mut_ptr(),  // x12 = output address (32-byte BE)
            in("a7") HINT_SYSCALL_NUMBER,
        )
    }
}

#[cfg(not(target_arch = "riscv64"))]
pub fn hint(_hint_id: usize, _out: &mut [u8; 32], _input: &[u8; 32]) {
    unimplemented!("syscalls are only implemented for riscv64 targets");
}

// =============================================================================
// Stub implementations for unsupported std functions
// These functions are required by Rust's std zkvm module but are not supported
// in Lambda VM. They will panic at runtime if called.
// =============================================================================

/// # Safety
///
/// This function is not supported in Lambda VM.
/// It will panic if called.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sys_read(_fd: u32, _buf: *mut u8, _nbytes: usize) -> usize {
    panic!("sys_read is not supported: io::Read for Stdin is not implemented in Lambda VM");
}

/// # Safety
///
/// This function is not supported in Lambda VM.
/// It will panic if called.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sys_argc() -> usize {
    panic!("sys_argc is not supported: command-line arguments are not available in Lambda VM");
}

/// # Safety
///
/// This function is not supported in Lambda VM.
/// It will panic if called.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sys_argv(_buf: *mut u32, _buf_nwords: usize, _arg_idx: usize) -> usize {
    panic!("sys_argv is not supported: command-line arguments are not available in Lambda VM");
}
