#![no_std]
#![no_main]

use core::arch::asm;
use core::panic::PanicInfo;

const PRIVATE_INPUT_START: usize = 0xFF000000;
const SYSCALL_COMMIT: u64 = 64;
const SYSCALL_HALT: u64 = 93;

#[panic_handler]
fn panic(_info: &PanicInfo) -> ! {
    loop {}
}

fn read_n() -> u64 {
    // Layout matches `syscalls::get_private_input`: 4-byte LE length prefix at
    // PRIVATE_INPUT_START, payload at +4. We only need the first 8 bytes (u64).
    let mut n_bytes = [0u8; 8];

    debug_assert!(
        unsafe { core::ptr::read_volatile(PRIVATE_INPUT_START as *const u32) } >= 8,
        "private input too short to contain a u64"
    );

    let input_data = (PRIVATE_INPUT_START + 4) as *const u8;
    n_bytes.copy_from_slice(unsafe { core::slice::from_raw_parts(input_data, 8) });

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
