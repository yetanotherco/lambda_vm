//! EF zkVM IO interface: <https://github.com/eth-act/zkvm-standards/blob/main/standards/io-interface/README.md>
//!
//! Two C-callable functions that match the EF standard so portable applications
//! compile unchanged across zkVMs:
//!
//! - `read_input`: returns a zero-copy pointer + size to the private input.
//! - `write_output`: appends bytes to the public output. Multiple calls
//!   concatenate.
//!
//! On Lambda VM these map to:
//! - `read_input` → memory-mapped private input region at `0xFF000000`
//!   (4-byte LE length prefix at base, data at `+4`).
//! - `write_output` → ECALL #64 (Commit). The trace builder maintains a
//!   running commitment index in synthetic register `x254`, so multiple
//!   ECALLs naturally concatenate at the proof level.

#[cfg(target_arch = "riscv64")]
use core::arch::asm;

#[cfg(target_arch = "riscv64")]
use crate::syscalls::SyscallNumbers;

/// Memory-mapped private input region start address.
/// Must match `executor::vm::memory::PRIVATE_INPUT_START_INDEX`.
#[cfg(target_arch = "riscv64")]
const PRIVATE_INPUT_LEN_ADDR: usize = 0xFF000000;
#[cfg(target_arch = "riscv64")]
const PRIVATE_INPUT_DATA_ADDR: usize = 0xFF000004;

/// EF IO: return a zero-copy pointer and size for the private input.
///
/// Per the spec this function is idempotent, callable multiple times, and
/// cannot fail. If `buf_size` is 0, the value of `buf_ptr` is unspecified.
///
/// # Safety
///
/// `buf_ptr` and `buf_size` must be valid, writable pointers.
#[cfg(target_arch = "riscv64")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn read_input(buf_ptr: *mut *const u8, buf_size: *mut usize) {
    let len_ptr = PRIVATE_INPUT_LEN_ADDR as *const u32;
    let len = unsafe { core::ptr::read_volatile(len_ptr) } as usize;
    unsafe {
        *buf_ptr = PRIVATE_INPUT_DATA_ADDR as *const u8;
        *buf_size = len;
    }
}

/// EF IO: append `size` bytes from `output` to the public output.
///
/// Multiple calls concatenate. Per the spec this function cannot fail; in
/// practice the executor enforces a total-output cap (see
/// `MAX_PUBLIC_OUTPUT_TOTAL_SIZE` in `executor::vm::memory`). Exceeding it
/// causes the executor to return an error and abort proving — not a graceful
/// failure mode at the C boundary, but consistent with "cannot fail" for
/// well-formed programs that stay under the limit.
///
/// # Safety
///
/// `output` must point to `size` readable bytes within guest memory.
#[cfg(target_arch = "riscv64")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn write_output(output: *const u8, size: usize) {
    unsafe {
        asm!(
            "ecall",
            in("a0") 1usize, // fd = 1 (stdout) — required by the COMMIT chip
            in("a1") output,
            in("a2") size,
            in("a7") SyscallNumbers::Commit as usize,
        );
    }
}

#[cfg(not(target_arch = "riscv64"))]
/// # Safety
///
/// Host-side stub. Lambda VM's IO interface is only implemented for the
/// `riscv64` guest target.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn read_input(_buf_ptr: *mut *const u8, _buf_size: *mut usize) {
    unimplemented!("read_input is only implemented for riscv64 targets");
}

#[cfg(not(target_arch = "riscv64"))]
/// # Safety
///
/// Host-side stub. Lambda VM's IO interface is only implemented for the
/// `riscv64` guest target.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn write_output(_output: *const u8, _size: usize) {
    unimplemented!("write_output is only implemented for riscv64 targets");
}
