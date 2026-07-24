use riscv as _;

const MAX_MEMORY_SIZE: usize = 0xC000_0000;
const WORD_SIZE: usize = 4;

// Guest global allocator, selectable at build time:
//   - default: embedded-alloc TLSF — reclaims freed memory (safe for arbitrary churn).
//   - `bump-alloc` feature (BENCH): a monotonic bump allocator that never frees
//     (no-op `dealloc`) and skips zeroing in `alloc_zeroed` (guest memory is
//     zero-initialized and bump never reuses a region, so fresh bytes read as 0).
//     Removes the free-list bookkeeping the profile attributes ~15% of guest cycles
//     to, at the cost of a monotonically growing footprint — see the roadmap's §1.2
//     risk analysis (OOM + touched-page growth). Kept behind a feature so TLSF stays
//     the default.
//
// Only the guest installs a #[global_allocator]; on host (e.g. `cargo test` for the
// sponge's differential tests) the attribute would hijack the test harness's
// allocator with a never-initialized heap and abort.

#[cfg(not(feature = "bump-alloc"))]
mod imp {
    use embedded_alloc::TlsfHeap as Heap;

    #[cfg_attr(target_arch = "riscv64", global_allocator)]
    static HEAP: Heap = Heap::empty();

    pub fn init(heap_start: usize, heap_end: usize) {
        unsafe { HEAP.init(heap_start, heap_end - heap_start) }
    }
}

#[cfg(feature = "bump-alloc")]
mod imp {
    use core::alloc::{GlobalAlloc, Layout};
    use core::sync::atomic::{AtomicUsize, Ordering};

    struct BumpAlloc;

    // Single-hart guest → `Relaxed` atomics are contention-free and avoid the
    // `static mut` edition-2024 lints.
    static HEAP_POS: AtomicUsize = AtomicUsize::new(0);
    static HEAP_END: AtomicUsize = AtomicUsize::new(0);

    #[cfg_attr(target_arch = "riscv64", global_allocator)]
    static ALLOC: BumpAlloc = BumpAlloc;

    pub fn init(heap_start: usize, heap_end: usize) {
        HEAP_POS.store(heap_start, Ordering::Relaxed);
        HEAP_END.store(heap_end, Ordering::Relaxed);
    }

    unsafe impl GlobalAlloc for BumpAlloc {
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            let align = layout.align();
            let pos = HEAP_POS.load(Ordering::Relaxed);
            // `align` is a power of two per the Layout contract.
            let aligned = pos.wrapping_add(align - 1) & !(align - 1);
            match aligned.checked_add(layout.size()) {
                Some(new_pos) if new_pos <= HEAP_END.load(Ordering::Relaxed) => {
                    HEAP_POS.store(new_pos, Ordering::Relaxed);
                    aligned as *mut u8
                }
                // Out of heap → null makes the caller's `handle_alloc_error` abort.
                _ => core::ptr::null_mut(),
            }
        }

        unsafe fn dealloc(&self, _ptr: *mut u8, _layout: Layout) {
            // A bump allocator never reclaims.
        }

        unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
            // Guest memory is zero-initialized and bump never reuses a freed region,
            // so freshly bumped memory already reads as zero — skip the memset.
            unsafe { self.alloc(layout) }
        }
    }
}

pub fn init_allocator() {
    unsafe extern "C" {
        static _end: u8;
    }
    let heap_pos: usize = unsafe { (&_end) as *const u8 as usize };
    imp::init(heap_pos, MAX_MEMORY_SIZE);
}

/// # Safety
///
/// This function should not be called by the user
/// It is only for rust std internal uses
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sys_alloc_aligned(bytes: usize, align: usize) -> *mut u8 {
    // Route through whichever `#[global_allocator]` is installed (TLSF or bump).
    unsafe { std::alloc::alloc(core::alloc::Layout::from_size_align(bytes, align).unwrap()) }
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
