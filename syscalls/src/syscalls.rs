use core::arch::asm;

// TODO: This should be properly defined
const MAX_PRIVATE_INPUT_SIZE: usize = 1024;

enum SyscallNumbers {
    Print = 1,
    Panic = 2,
    Commit = 3,
    GetPrivateInputs = 4,
    Halt = 5,
}

/// This is a template for printing in the vm
pub fn print_string(s: &str) {
    unsafe {
        asm!(
            "mv a0, {ptr}",
            "mv a1, {len}",
            "mv a7, {syscall_number}", // syscall number for print
            "ecall",
            ptr = in(reg) s.as_ptr(),
            len = in(reg) s.len(),
            syscall_number = in(reg) SyscallNumbers::Print as usize,
        );
    }
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

/// # Safety
///
/// This function should not be called by the user
/// It is only for rust std internal uses
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sys_panic(msg_ptr: *const u8, len: usize) {
    print_string("Sys panic called\n");
    unsafe {
        asm!(
            "mv a0, {ptr}",
            "mv a1, {len}",
            "mv a7, {syscall_number}", // syscall number for panic
            "ecall",
            ptr = in(reg) msg_ptr,
            len = in(reg) len,
            syscall_number = in(reg) SyscallNumbers::Panic as usize,
        )
    }
}

/// # Safety
///
/// This function should not be called by the user
/// It is only for rust std internal uses
#[unsafe(no_mangle)]
pub extern "C" fn sys_rand(_buf: *mut u8, _len: usize) -> isize {
    print_string("sys_rand called\n");
    0
}

pub fn commit(slice: &[u8]) {
    print_string("commit called\n");
    unsafe {
        asm!(
            "mv a0, {ptr}",
            "mv a1, {len}",
            "mv a7, {syscall_number}", // syscall number for commit
            "ecall",
            ptr = in(reg) slice.as_ptr(),
            len = in(reg) slice.len(),
            syscall_number = in(reg) SyscallNumbers::Commit as usize,
        )
    }
}

pub fn get_private_input() -> Result<Vec<u8>, SyscallError> {
    print_string("get_private_input called\n");
    let mut dest = vec![0u8; MAX_PRIVATE_INPUT_SIZE];
    unsafe {
        asm!(
            "mv a0, {ptr}",
            "mv a7, {syscall_number}", // syscall number for get_private_input
            "ecall",
            ptr = in(reg) dest.as_mut_ptr(),
            syscall_number = in(reg) SyscallNumbers::GetPrivateInputs as usize,
        )
    }
    let len = u32::from_le_bytes(
        dest[0..4]
            .try_into()
            .map_err(|_| SyscallError::WrongPrivateInputSize)?,
    ) as usize;
    dest.drain(0..4);
    dest.truncate(len);

    Ok(dest)
}

#[derive(thiserror::Error, Debug)]
pub enum SyscallError {
    #[error("Wrong private input size")]
    WrongPrivateInputSize,
}

pub fn sys_halt() -> ! {
    print_string("sys_halt called\n");
    unsafe {
        asm!(
            "mv a7, {syscall_number}", // syscall number for halt
            "ecall",
            syscall_number = in(reg) SyscallNumbers::Halt as usize,
        );
    }
    unreachable!()
}
