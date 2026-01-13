use embedded_alloc::TlsfHeap as Heap;
use riscv as _;

#[global_allocator]
static HEAP: Heap = Heap::empty();

const MAX_MEMORY_SIZE: usize = 0xC000_0000;

pub fn init_allocator() {
    {
        unsafe extern "C" {
            static _end: u8;
        }
        let heap_pos: usize = unsafe { (&_end) as *const u8 as usize };
        unsafe { HEAP.init(heap_pos, MAX_MEMORY_SIZE - heap_pos) }
    }
}
