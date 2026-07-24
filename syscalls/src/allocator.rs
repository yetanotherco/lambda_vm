#[cfg(not(all(target_arch = "riscv64", feature = "bump-alloc")))]
use embedded_alloc::TlsfHeap as Heap;
use riscv as _;

// Only the guest routes Rust allocations through this heap; on host (e.g.
// `cargo test` for the sponge's differential tests) the attribute would hijack
// the test harness's allocator with a never-initialized heap and abort.
#[cfg(not(all(target_arch = "riscv64", feature = "bump-alloc")))]
#[cfg_attr(target_arch = "riscv64", global_allocator)]
static HEAP: Heap = Heap::empty();

// Bump allocator (opt-in via the `bump-alloc` feature). See `bump::BumpAlloc`.
// SOUND only when the guest's TOTAL allocated bytes fit the heap region, because
// `dealloc` never reclaims. The recursion verifier verifies this bound (~49 MiB
// total vs a ~3 GiB heap) before opting in.
#[cfg(all(target_arch = "riscv64", feature = "bump-alloc"))]
#[global_allocator]
static HEAP: bump::BumpAlloc = bump::BumpAlloc::new();

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

/// Bump allocator: a monotonic pointer over the heap region with a no-op
/// `dealloc`. Every allocation returns fresh, correctly-aligned memory that is
/// never handed out again, so it is always correct; the only failure mode is
/// running out of heap. That makes it SOUND to use only for guests whose TOTAL
/// allocated bytes (summed over every allocation, realloc churn included) fit
/// the region — nothing is ever reclaimed. It trades that memory for an
/// allocate in a handful of instructions (align-up + bounds-check + advance)
/// instead of the TLSF free-list walk/split/merge, and a zero-cost free.
///
/// Single-hart guest: interior mutability via `UnsafeCell` with a hand-written
/// `Sync` is sound because the VM never executes two allocations concurrently.
#[cfg(all(target_arch = "riscv64", feature = "bump-alloc"))]
mod bump {
    use core::alloc::{GlobalAlloc, Layout};
    use core::cell::UnsafeCell;
    use core::ptr;

    pub struct BumpAlloc {
        /// Address of the next byte to hand out.
        next: UnsafeCell<usize>,
        /// One past the last usable heap byte.
        end: UnsafeCell<usize>,
    }

    // SAFETY: the guest is single-threaded (single hart), so the interior
    // `UnsafeCell`s are never touched concurrently.
    unsafe impl Sync for BumpAlloc {}

    impl BumpAlloc {
        pub const fn new() -> Self {
            Self {
                next: UnsafeCell::new(0),
                end: UnsafeCell::new(0),
            }
        }

        /// # Safety
        /// Call once, before any allocation, with a valid `[start, start+size)`
        /// heap region.
        pub unsafe fn init(&self, start: usize, size: usize) {
            unsafe {
                *self.next.get() = start;
                *self.end.get() = start + size;
            }
        }
    }

    unsafe impl GlobalAlloc for BumpAlloc {
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            let next_ptr = self.next.get();
            let cur = unsafe { *next_ptr };
            let align = layout.align();
            // Align up. `align` is a power of two, `cur` is a small guest
            // address, so `cur + align - 1` cannot overflow usize.
            let aligned = (cur + (align - 1)) & !(align - 1);
            let new_next = match aligned.checked_add(layout.size()) {
                Some(v) => v,
                None => return ptr::null_mut(),
            };
            if new_next > unsafe { *self.end.get() } {
                // Out of heap — signal OOM to the allocation error handler.
                return ptr::null_mut();
            }
            unsafe { *next_ptr = new_next };
            aligned as *mut u8
        }

        // Bump allocator never reclaims; freeing is a no-op.
        unsafe fn dealloc(&self, _ptr: *mut u8, _layout: Layout) {}
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
