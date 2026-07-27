use riscv as _;

const MAX_MEMORY_SIZE: usize = 0xC000_0000;
const WORD_SIZE: usize = 4;

// Guest global allocator, selectable at build time (at most one BENCH feature):
//   - default: Doug Lea's malloc (dlmalloc), backed by a bump "system" provider that
//     hands dlmalloc page-aligned chunks from the guest heap. dlmalloc does all
//     sub-allocation churn itself, so — unlike a raw bump allocator — it reuses freed
//     memory (no OOM) while executing fewer guest instructions per alloc/free than TLSF
//     and cutting memory operations (smaller memory table = cheaper proof).
//   - `tlsf-alloc` feature: embedded-alloc's TLSF heap — the previous default, kept a
//     flag away for A/B or fallback (reclaims freed memory, safe for arbitrary churn).
//   - `bump-alloc` feature (BENCH): a monotonic bump allocator that never frees (no-op
//     `dealloc`) and skips zeroing in `alloc_zeroed` (guest memory is zero-initialized
//     and bump never reuses a region, so fresh bytes read as 0). Fastest per-op but the
//     footprint grows monotonically — OOM-prone (see the roadmap's §1.2 risk analysis).
//
// Only the guest installs a #[global_allocator]; on host (e.g. `cargo test` for the
// sponge's differential tests) the attribute would hijack the test harness's
// allocator with a never-initialized heap and abort.

#[cfg(all(feature = "tlsf-alloc", feature = "bump-alloc"))]
compile_error!("`tlsf-alloc` and `bump-alloc` are mutually exclusive allocator features");

#[cfg(not(any(feature = "tlsf-alloc", feature = "bump-alloc")))]
mod imp {
    use core::alloc::{GlobalAlloc, Layout};
    use core::cell::RefCell;
    use core::sync::atomic::{AtomicUsize, Ordering};
    use critical_section::Mutex;
    use dlmalloc::{Allocator, Dlmalloc};

    // Page granularity dlmalloc requests memory in. Must be a power of two; the guest
    // heap region is 3 GiB so the value only affects the segment rounding below.
    const PAGE_SIZE: usize = 4096;

    // The "system" side of dlmalloc: instead of mmap/sbrk (absent on the guest) it
    // bump-allocates page-aligned segments from the single contiguous heap region
    // [_end, MAX_MEMORY_SIZE). It never releases a segment (`free`/`free_part`/
    // `remap` all decline) — dlmalloc itself owns all reuse of freed *user*
    // allocations against this fixed backing store, which is what keeps churny
    // workloads OOM-free unlike a raw bump allocator.
    struct BumpSystem;

    // Single-hart guest → `Relaxed` atomics are contention-free.
    static HEAP_POS: AtomicUsize = AtomicUsize::new(0);
    static HEAP_END: AtomicUsize = AtomicUsize::new(0);

    unsafe impl Allocator for BumpSystem {
        fn alloc(&self, size: usize) -> (*mut u8, usize, u32) {
            // Round up to a page so consecutive segments stay page-aligned.
            let size = size.wrapping_add(PAGE_SIZE - 1) & !(PAGE_SIZE - 1);
            let pos = HEAP_POS.load(Ordering::Relaxed);
            match pos.checked_add(size) {
                Some(new_pos) if new_pos <= HEAP_END.load(Ordering::Relaxed) => {
                    HEAP_POS.store(new_pos, Ordering::Relaxed);
                    // flags = 0: a plain external segment (never partially released).
                    (pos as *mut u8, size, 0)
                }
                // Out of heap → null makes dlmalloc return null → handle_alloc_error.
                _ => (core::ptr::null_mut(), 0, 0),
            }
        }

        fn remap(&self, _ptr: *mut u8, _old: usize, _new: usize, _can_move: bool) -> *mut u8 {
            core::ptr::null_mut()
        }

        fn free_part(&self, _ptr: *mut u8, _old: usize, _new: usize) -> bool {
            false
        }

        fn free(&self, _ptr: *mut u8, _size: usize) -> bool {
            false
        }

        fn can_release_part(&self, _flags: u32) -> bool {
            false
        }

        fn allocates_zeros(&self) -> bool {
            // Guest memory is zero-initialized and this provider never reuses a
            // segment, so system-fresh bytes read as 0 → dlmalloc's calloc skips the
            // memset for system-fresh memory (it still zeroes recycled blocks itself).
            true
        }

        fn page_size(&self) -> usize {
            PAGE_SIZE
        }
    }

    // Dlmalloc is Send but !Sync, so it can't sit in a static directly. A single-hart
    // critical section serializes access and supplies the Sync a #[global_allocator]
    // static requires — the same primitive embedded-alloc's TLSF heap uses.
    static DLMALLOC: Mutex<RefCell<Dlmalloc<BumpSystem>>> =
        Mutex::new(RefCell::new(Dlmalloc::new_with_allocator(BumpSystem)));

    struct DlGlobal;

    #[cfg_attr(target_arch = "riscv64", global_allocator)]
    static ALLOC: DlGlobal = DlGlobal;

    pub fn init(heap_start: usize, heap_end: usize) {
        HEAP_POS.store(heap_start, Ordering::Relaxed);
        HEAP_END.store(heap_end, Ordering::Relaxed);
    }

    unsafe impl GlobalAlloc for DlGlobal {
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            critical_section::with(|cs| unsafe {
                DLMALLOC
                    .borrow(cs)
                    .borrow_mut()
                    .malloc(layout.size(), layout.align())
            })
        }

        unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
            critical_section::with(|cs| unsafe {
                DLMALLOC
                    .borrow(cs)
                    .borrow_mut()
                    .free(ptr, layout.size(), layout.align())
            })
        }

        unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
            critical_section::with(|cs| unsafe {
                DLMALLOC
                    .borrow(cs)
                    .borrow_mut()
                    .calloc(layout.size(), layout.align())
            })
        }

        unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
            critical_section::with(|cs| unsafe {
                DLMALLOC.borrow(cs).borrow_mut().realloc(
                    ptr,
                    layout.size(),
                    layout.align(),
                    new_size,
                )
            })
        }
    }
}

#[cfg(feature = "tlsf-alloc")]
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
    // Route through whichever `#[global_allocator]` is installed (dlmalloc/TLSF/bump).
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
