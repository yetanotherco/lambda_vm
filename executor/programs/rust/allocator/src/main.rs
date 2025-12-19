#![no_std]
#![no_main]

extern crate alloc;
use core::panic::PanicInfo;
use alloc::string::String;
use embedded_alloc::LlffHeap as Heap;

#[global_allocator]
static HEAP: Heap = Heap::empty();

use riscv as _;

#[panic_handler]
fn panic(_info: &PanicInfo) -> ! {
    loop {}
}

#[unsafe(export_name = "main")]
pub fn main() -> u32 {
    {
        use core::mem::MaybeUninit;
        use core::ptr::addr_of_mut;
        const HEAP_SIZE: usize = 1024;
        static mut HEAP_MEM: [MaybeUninit<u8>; HEAP_SIZE] = [MaybeUninit::uninit(); HEAP_SIZE];
        let heap_ptr = addr_of_mut!(HEAP_MEM) as *mut u8;
        unsafe { HEAP.init(heap_ptr as usize, HEAP_SIZE) }
    }
    let hello = String::from("Hello World");
    return hello.len() as u32;
}
