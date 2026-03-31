#![no_std]
#![no_main]

use core::arch::asm;
use core::panic::PanicInfo;

const SYSCALL_GET_PRIVATE_INPUTS: u64 = 4;
const SYSCALL_COMMIT: u64 = 64;
const SYSCALL_HALT: u64 = 93;

#[panic_handler]
fn panic(_info: &PanicInfo) -> ! {
    loop {}
}

fn read_n() -> u64 {
    let mut input = [0u8; 12];

    unsafe {
        asm!(
            "ecall",
            in("a0") input.as_mut_ptr(),
            in("a7") SYSCALL_GET_PRIVATE_INPUTS,
        );
    }

    let mut n_bytes = [0u8; 8];
    n_bytes.copy_from_slice(&input[4..12]);
    u64::from_le_bytes(n_bytes)
}

fn commit(bytes: &[u8]) {
    unsafe {
        asm!(
            "ecall",
            in("a0") 1u64,
            in("a1") bytes.as_ptr(),
            in("a2") bytes.len(),
            in("a7") SYSCALL_COMMIT,
        );
    }
}

fn halt() -> ! {
    unsafe {
        asm!(
            "ecall",
            in("a0") 0u64,
            in("a7") SYSCALL_HALT,
            options(noreturn),
        );
    }
}

#[unsafe(no_mangle)]
pub fn main() -> ! {
    let n = read_n();
    let mut a: u64 = 0;
    let mut b: u64 = 1;
    for _ in 0..n {
        let c = a.wrapping_add(b);
        a = b;
        b = c;
    }

    commit(&b.to_le_bytes());
    halt()
}
