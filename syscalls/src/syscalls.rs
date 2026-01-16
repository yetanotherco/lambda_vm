#[cfg(target_arch = "riscv64")]
use core::arch::asm;

#[cfg(target_arch = "riscv64")]
// TODO: This should be properly defined
const MAX_PRIVATE_INPUT_SIZE: usize = 6700000;

#[cfg(target_arch = "riscv64")]
enum SyscallNumbers {
    Print = 1,
    Panic = 2,
    Commit = 3,
    GetPrivateInputs = 4,
    Halt = 5,
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
    print_string("commit called\n");
    unsafe {
        asm!(
            "ecall",
            in("a0") slice.as_ptr(),
            in("a1") slice.len(),
            in("a7") SyscallNumbers::Commit as usize,
        )
    }
}

#[cfg(target_arch = "riscv64")]
pub fn get_private_input() -> Result<Vec<u8>, SyscallError> {
    print_string("get_private_input called\n");
    let mut dest = vec![0u8; MAX_PRIVATE_INPUT_SIZE];
    unsafe {
        asm!(
            "ecall",
            in("a0") dest.as_mut_ptr(),
            in("a7") SyscallNumbers::GetPrivateInputs as usize,
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

#[derive(Debug)]
pub enum SyscallError {
    WrongPrivateInputSize,
}

#[cfg(target_arch = "riscv64")]
pub fn sys_halt() -> ! {
    print_string("sys_halt called\n");
    unsafe {
        asm!(
            "ecall",
            in("a7") SyscallNumbers::Halt as usize,
        );
    }
    unreachable!()
}

#[cfg(not(target_arch = "riscv64"))]
pub fn sys_halt() -> ! {
    unimplemented!("syscalls are only implemented for riscv64 targets");
}
