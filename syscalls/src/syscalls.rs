#[cfg(target_arch = "riscv64")]
use core::arch::asm;

/// Memory-mapped private input region start address.
/// Layout: 4-byte LE length prefix at this address, data at +4.
/// The host pre-loads the input; the guest reads directly (no ecall).
#[cfg(target_arch = "riscv64")]
const PRIVATE_INPUT_START: usize = 0xFF000000;

#[cfg(target_arch = "riscv64")]
enum SyscallNumbers {
    Print = 1,
    Panic = 2,
    Commit = 64,
    Halt = 93,
}

#[cfg(target_arch = "riscv64")]
/// This is a template for printing in the vm
pub fn print_string(s: &str) {
    unsafe {
        asm!(
            "ecall",
            in("a0") s.as_ptr(),
            in("a1") s.len(),
            in("a7") SyscallNumbers::Print as usize,
        );
    }
}

#[cfg(not(target_arch = "riscv64"))]
/// This is a template for printing in the vm
pub fn print_string(_: &str) {
    unimplemented!("syscalls are only implemented for riscv64 targets");
}

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
pub fn get_private_input() -> Result<Vec<u8>, SyscallError> {
    // Read length prefix (4 bytes LE at PRIVATE_INPUT_START)
    let len_ptr = PRIVATE_INPUT_START as *const u32;
    let len = unsafe { core::ptr::read_volatile(len_ptr) } as usize;
    // Read data bytes starting at PRIVATE_INPUT_START + 4
    let data_ptr = (PRIVATE_INPUT_START + 4) as *const u8;
    let slice = unsafe { core::slice::from_raw_parts(data_ptr, len) };
    Ok(slice.to_vec())
}

#[cfg(not(target_arch = "riscv64"))]
pub fn get_private_input() -> Result<Vec<u8>, SyscallError> {
    unimplemented!("syscalls are only implemented for riscv64 targets");
}

#[derive(Debug)]
pub enum SyscallError {
    WrongPrivateInputSize,
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
