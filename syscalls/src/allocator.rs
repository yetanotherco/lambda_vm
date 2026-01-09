use embedded_alloc::LlffHeap as Heap;
use riscv as _;

#[global_allocator]
static HEAP: Heap = Heap::empty();

pub fn init_allocator() {
    {
        use core::mem::MaybeUninit;
        use core::ptr::addr_of_mut;
        const HEAP_SIZE: usize = 1024 * 64;
        static mut HEAP_MEM: [MaybeUninit<u8>; HEAP_SIZE] = [MaybeUninit::uninit(); HEAP_SIZE];
        let heap_ptr = addr_of_mut!(HEAP_MEM) as *mut u8;
        unsafe { HEAP.init(heap_ptr as usize, HEAP_SIZE) }
    }
}
