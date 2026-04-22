use embedded_alloc::TlsfHeap as Heap;
use riscv as _;

#[global_allocator]
static HEAP: Heap = Heap::empty();

const MAX_MEMORY_SIZE: usize = 0xC000_0000;
const WORD_SIZE: usize = 4;

pub fn init_allocator() {
    {
        unsafe extern "C" {
            static _end: u8;
        }
        let heap_pos: usize = unsafe { (&_end) as *const u8 as usize };
        unsafe { HEAP.init(heap_pos, MAX_MEMORY_SIZE - heap_pos) }
    }
}

/// # Safety
///
/// This function should not be called by the user
/// It is only for rust std internal uses
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sys_alloc_aligned(bytes: usize, align: usize) -> *mut u8 {
    use core::alloc::GlobalAlloc;
    unsafe { HEAP.alloc(core::alloc::Layout::from_size_align(bytes, align).unwrap()) }
}

/// # Safety
///
/// This function should not be called by the user
/// It is only for rust std internal uses
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sys_alloc_words(nwords: usize) -> *mut u32 {
    unsafe { sys_alloc_aligned(WORD_SIZE * nwords, WORD_SIZE) as *mut u32 }
}

/// # Safety
///
/// This function should not be called by the user
/// It is only for rust std internal uses
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sys_getenv(
    _out_words: *mut u32,
    _out_nwords: usize,
    _varname: *const u8,
    _varname_len: usize,
) -> usize {
    // NOTE: no print_string here — the Print ecall (#1) has no receiver on the
    // Ecall bus and would cause a verification failure.
    usize::MAX
}
