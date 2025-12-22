#![no_std]
#![no_main]

use ethereum_types::U256;
use core::panic::PanicInfo;
use embedded_alloc::LlffHeap as Heap;

#[global_allocator]
static HEAP: Heap = Heap::empty();

use riscv as _;

#[panic_handler]
fn panic(_info: &PanicInfo) -> ! {
    loop {}
}

pub fn u256_from_big_endian(slice: &[u8]) -> U256 {
    let mut padded = [0u8; 32];
    padded[32 - slice.len()..32].copy_from_slice(slice);

    let mut ret = [0; 4];

    let mut u64_bytes = [0u8; 8];
    for i in 0..4 {
        u64_bytes.copy_from_slice(&padded[8 * i..(8 * i + 8)]);
        ret[4 - i - 1] = u64::from_be_bytes(u64_bytes);
    }

    U256(ret)
}

pub fn u256_to_big_endian(value: U256) -> [u8; 32] {
    let mut bytes = [0u8; 32];

    for i in 0..4 {
        let u64_be = value.0[4 - i - 1].to_be_bytes();
        bytes[8 * i..(8 * i + 8)].copy_from_slice(&u64_be);
    }

    bytes
}

#[unsafe(export_name = "main")]
pub fn main() -> u8 {
    {
        use core::mem::MaybeUninit;
        use core::ptr::addr_of_mut;
        const HEAP_SIZE: usize = 1024;
        static mut HEAP_MEM: [MaybeUninit<u8>; HEAP_SIZE] = [MaybeUninit::uninit(); HEAP_SIZE];
        let heap_ptr = addr_of_mut!(HEAP_MEM) as *mut u8;
        unsafe { HEAP.init(heap_ptr as usize, HEAP_SIZE) }
    }
    
    let a = u256_to_big_endian(U256::one());
    let b = U256::one().to_big_endian();
    let new_a = u256_from_big_endian(&a);
    let new_b = U256::from_big_endian(&b);

    if a != b || new_a != U256::one() || new_b != U256::one(){
        return 0;
    }
    return 1;
}
