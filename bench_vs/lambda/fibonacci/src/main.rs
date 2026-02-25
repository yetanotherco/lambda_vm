#![no_std]
#![no_main]

use core::panic::PanicInfo;

#[panic_handler]
fn panic(_info: &PanicInfo) -> ! {
    loop {}
}

const N: u64 = include!(concat!(env!("OUT_DIR"), "/n.txt"));

#[inline(never)]
fn halt(code: u64) -> ! {
    unsafe {
        core::arch::asm!(
            "ecall",
            in("a0") code,
            in("a7") 5u64,
            options(noreturn),
        );
    }
}

#[unsafe(no_mangle)]
pub fn main() -> ! {
    let mut a: u64 = 0;
    let mut b: u64 = 1;
    for _ in 0..N {
        let c = a.wrapping_add(b);
        a = b;
        b = c;
    }
    halt(b)
}
