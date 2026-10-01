#![no_std]
#![no_main]

use core::arch::asm;
use core::panic::PanicInfo;

const SYSCALL_HALT: u64 = 93;

#[panic_handler]
fn panic(_info: &PanicInfo) -> ! {
    loop {}
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
    halt()
}
