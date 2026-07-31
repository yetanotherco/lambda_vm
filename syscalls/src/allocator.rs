use riscv as _;

const MAX_MEMORY_SIZE: usize = 0xC000_0000;
const WORD_SIZE: usize = 4;

// Guest global allocator, selectable at build time. The default was chosen on a measured
// three-way A/B (ethrex blocks of 1..1500 transfers, both 1-to-1 and distinct-account
// fixtures; guest cycles, trace elements, proving time and peak RSS):
//
//   - default: a monotonic bump allocator. No free lists and no coalescing -- `alloc`
//     moves a cursor, `dealloc` is empty -- so it spends the fewest guest instructions
//     per allocation of the three. Against dlmalloc that measured ~9% fewer guest
//     cycles, ~8% fewer main-trace elements, ~6% faster proving and ~8% lower peak RSS,
//     flat across every block size tried. It never reuses a freed region, so its
//     footprint grows monotonically -- see the ceiling note below.
//   - `dlmalloc-alloc` feature: Doug Lea's malloc on a bump "system" provider that hands
//     it page-aligned segments. Its footprint is bounded by live bytes instead of total
//     bytes ever allocated, which makes it the only one of the three safe under
//     unbounded churn -- use it for continuations, where one execution spans many blocks
//     and bump's growth has no bound.
//   - `tlsf-alloc` feature: embedded-alloc's TLSF heap. The original default, and the
//     slowest of the three (bump proved ~11% faster, dlmalloc ~6%). Kept as a fallback.
//
// Bump's ceiling: the guest heap is [_end, MAX_MEMORY_SIZE), about 3 GiB, and a block's
// allocation is bounded by its gas. A gas-full block of the cheapest transactions (1500
// transfers, 31.5M gas, 523M cycles) executes without exhausting it, and any other
// composition fits fewer transactions into the same gas. Contract-heavy blocks allocate
// more per transaction and are not covered by that bound; if one ever exhausts the heap,
// `dlmalloc-alloc` is a one-flag fallback.
//
// Only the guest installs a #[global_allocator]; on host (e.g. `cargo test` for the
// sponge's differential tests) the attribute would hijack the test harness's
// allocator with a never-initialized heap and abort.

// Off riscv only `init` is reachable (no `#[global_allocator]` is installed and
// `sys_alloc_aligned` goes through `std::alloc`), so the plumbing is dead there.
#[cfg(not(any(feature = "tlsf-alloc", feature = "dlmalloc-alloc")))]
#[cfg_attr(not(target_arch = "riscv64"), allow(dead_code))]
mod imp {
    use core::alloc::{GlobalAlloc, Layout};
    use core::sync::atomic::{AtomicUsize, Ordering};

    struct BumpAlloc;

    // Single-hart guest -> `Relaxed` atomics are contention-free and avoid the
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
                // Out of heap -> null makes the caller's `handle_alloc_error` abort.
                _ => core::ptr::null_mut(),
            }
        }

        unsafe fn dealloc(&self, _ptr: *mut u8, _layout: Layout) {
            // A bump allocator never reclaims.
        }

        unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
            // Guest memory is zero-initialized and bump never reuses a freed region,
            // so freshly bumped memory already reads as zero -- skip the memset.
            unsafe { self.alloc(layout) }
        }
    }

    // Host tests. `BumpAlloc`'s cursor is global, so they serialize on `HEAP_LOCK` and
    // each re-points it at its own leaked, page-aligned buffer.
    #[cfg(test)]
    mod tests {
        use super::*;
        use std::sync::{Mutex, MutexGuard};

        static HEAP_LOCK: Mutex<()> = Mutex::new(());

        // Leaks on purpose: `BumpAlloc` hands out raw addresses into this region, so it
        // must outlive every pointer derived from it.
        fn with_heap(bytes: usize) -> MutexGuard<'static, ()> {
            let guard = HEAP_LOCK.lock().unwrap_or_else(|e| e.into_inner());
            let l = Layout::from_size_align(bytes, 4096).unwrap();
            // Zeroed, like guest memory: reads of never-written heap return 0 there.
            let base = unsafe { std::alloc::alloc_zeroed(l) };
            assert!(!base.is_null());
            init(base as usize, base as usize + bytes);
            guard
        }

        fn layout(size: usize, align: usize) -> Layout {
            Layout::from_size_align(size, align).unwrap()
        }

        /// `alloc_zeroed` skips the memset, which is only sound because bump never hands
        /// back a region it already served. Dirty a block, free it, and check the next
        /// `alloc_zeroed` gets fresh (still-zero) memory rather than the dirt.
        #[test]
        fn alloc_zeroed_never_returns_a_dirtied_region() {
            let _guard = with_heap(1024 * 1024);
            let l = layout(256, 8);
            let dirty = unsafe { BumpAlloc.alloc(l) };
            assert!(!dirty.is_null());
            unsafe { core::ptr::write_bytes(dirty, 0xAA, 256) };
            unsafe { BumpAlloc.dealloc(dirty, l) };

            let fresh = unsafe { BumpAlloc.alloc_zeroed(l) };
            assert!(!fresh.is_null());
            assert_ne!(fresh, dirty, "bump must not re-serve a freed region");
            let bytes = unsafe { core::slice::from_raw_parts(fresh, 256) };
            assert!(bytes.iter().all(|&b| b == 0), "alloc_zeroed returned dirt");
        }

        /// The defining property, and what bounds the footprint: a free is a no-op, so
        /// the cursor only ever moves forward.
        #[test]
        fn dealloc_does_not_reclaim() {
            let _guard = with_heap(1024 * 1024);
            let l = layout(4096, 8);
            let first = unsafe { BumpAlloc.alloc(l) };
            unsafe { BumpAlloc.dealloc(first, l) };
            let second = unsafe { BumpAlloc.alloc(l) };
            assert_eq!(
                second as usize,
                first as usize + 4096,
                "the cursor must not rewind over a freed block"
            );
        }

        #[test]
        fn alignment_requests_are_honored() {
            let _guard = with_heap(1024 * 1024);
            // Start off-alignment so the padding path is exercised.
            let _ = unsafe { BumpAlloc.alloc(layout(1, 1)) };
            for align in [16usize, 64, 256, 4096] {
                let p = unsafe { BumpAlloc.alloc(layout(align * 3, align)) };
                assert!(!p.is_null(), "alloc with align {align} failed");
                assert_eq!(p as usize % align, 0, "align {align} not honored");
            }
        }

        /// Exhaustion must return null (which becomes `handle_alloc_error` on the guest),
        /// never a pointer past `HEAP_END`.
        #[test]
        fn exhaustion_returns_null_instead_of_running_past_the_heap() {
            let _guard = with_heap(8192);
            let l = layout(4096, 8);
            assert!(!unsafe { BumpAlloc.alloc(l) }.is_null());
            assert!(!unsafe { BumpAlloc.alloc(l) }.is_null());
            assert!(
                unsafe { BumpAlloc.alloc(l) }.is_null(),
                "handed out memory past HEAP_END"
            );
            // An absurd size declines too. It declines on the bounds check rather than
            // on the `checked_add`: `Layout` requires size rounded up to align to fit in
            // `isize::MAX`, so a size that would overflow the cursor arithmetic can't be
            // constructed in the first place.
            let huge = layout(isize::MAX as usize - 7, 8);
            assert!(unsafe { BumpAlloc.alloc(huge) }.is_null());
        }

        /// Before `init_allocator` runs HEAP_END is 0 -- allocation must fail closed
        /// rather than hand out address 0.
        #[test]
        fn uninitialized_allocator_hands_out_nothing() {
            let _guard = HEAP_LOCK.lock().unwrap_or_else(|e| e.into_inner());
            init(0, 0);
            assert!(unsafe { BumpAlloc.alloc(layout(1, 1)) }.is_null());
        }
    }
}

#[cfg(all(feature = "dlmalloc-alloc", not(feature = "tlsf-alloc")))]
#[cfg_attr(not(target_arch = "riscv64"), allow(dead_code))]
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
            // Round up to a page so consecutive segments stay page-aligned. Checked, so
            // a size near `usize::MAX` declines instead of wrapping to a small one.
            let Some(size) = size
                .checked_add(PAGE_SIZE - 1)
                .map(|rounded| rounded & !(PAGE_SIZE - 1))
            else {
                return (core::ptr::null_mut(), 0, 0);
            };
            let pos = HEAP_POS.load(Ordering::Relaxed);
            match pos.checked_add(size) {
                Some(new_pos) if new_pos <= HEAP_END.load(Ordering::Relaxed) => {
                    HEAP_POS.store(new_pos, Ordering::Relaxed);
                    // flags = 0: no `EXTERN` bit, so dlmalloc may coalesce a new segment
                    // onto the previous one (ours are contiguous, so it usually just
                    // extends `top`). Releasing is gated on `can_release_part` below,
                    // which declines, so `sys_trim`/`release_unused_segments` are no-ops.
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
            // Guest memory is zero-initialized and this provider never reuses a segment,
            // so system-fresh bytes read as 0. dlmalloc consults this only in
            // `calloc_must_clear` = `!allocates_zeros() || !mmapped(chunk)`, i.e. it may
            // skip calloc's memset only for a chunk it marked mmapped. Two independent
            // reasons that is safe here: the Rust port has no mmap path at all (nothing
            // ever sets the mmapped marker, so calloc always zeroes), and even if it
            // grew one, freeing an mmapped chunk whose system `free` declines drops the
            // chunk instead of re-binning it — so a recycled block is never mmapped.
            // Locked by `calloc_zeroes_recycled_dirty_blocks` below.
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

    // Host tests for the provider and for dlmalloc's behaviour on top of it. They drive
    // a local `Dlmalloc<BumpSystem>` rather than the `DLMALLOC` static: the static's
    // `critical_section::with` has no implementation off riscv (the impl comes from
    // `riscv`'s `critical-section-single-hart`), and a local instance exercises the same
    // allocator code. `BumpSystem`'s cursor is global, so the tests serialize on
    // `HEAP_LOCK` and each re-points it at its own leaked, page-aligned buffer.
    #[cfg(test)]
    mod tests {
        use super::*;
        use std::sync::{Mutex, MutexGuard};

        static HEAP_LOCK: Mutex<()> = Mutex::new(());

        // Leaks on purpose: the buffer must outlive every pointer dlmalloc derives from
        // it, and `BumpSystem` hands segments out by raw address.
        fn with_heap(bytes: usize) -> (MutexGuard<'static, ()>, Dlmalloc<BumpSystem>) {
            let guard = HEAP_LOCK.lock().unwrap_or_else(|e| e.into_inner());
            let layout = core::alloc::Layout::from_size_align(bytes, PAGE_SIZE).unwrap();
            // Zeroed, like guest memory: reads of never-written heap return 0 there.
            let base = unsafe { std::alloc::alloc_zeroed(layout) };
            assert!(!base.is_null());
            init(base as usize, base as usize + bytes);
            (guard, Dlmalloc::new_with_allocator(BumpSystem))
        }

        fn layout(size: usize) -> (usize, usize) {
            (size, core::mem::align_of::<usize>())
        }

        /// The load-bearing consequence of `allocates_zeros() == true`: dlmalloc's
        /// `calloc` may skip its memset when it believes a block is system-fresh, so
        /// recycling a dirtied block through `calloc` must still come back zeroed.
        /// Checked at a small size and at one past dlmalloc's 64 KiB granularity (the
        /// size class the C original would serve from a fresh mmap).
        #[test]
        fn calloc_zeroes_recycled_dirty_blocks() {
            for size in [64usize, 512 * 1024] {
                let (_guard, mut dl) = with_heap(8 * 1024 * 1024);
                let (sz, al) = layout(size);

                let dirty = unsafe { dl.malloc(sz, al) };
                assert!(!dirty.is_null(), "malloc({size}) failed");
                unsafe { core::ptr::write_bytes(dirty, 0xAA, size) };
                unsafe { dl.free(dirty, sz, al) };

                let fresh = unsafe { dl.calloc(sz, al) };
                assert!(!fresh.is_null(), "calloc({size}) failed");
                let bytes = unsafe { core::slice::from_raw_parts(fresh, size) };
                assert!(
                    bytes.iter().all(|&b| b == 0),
                    "calloc({size}) returned dirty memory: {} non-zero bytes",
                    bytes.iter().filter(|&&b| b != 0).count()
                );
            }
        }

        /// What dlmalloc buys over a raw bump allocator: churn is served out of freed
        /// blocks, so a heap far smaller than the total allocated volume never runs out.
        #[test]
        fn freed_blocks_are_reused_so_churn_does_not_exhaust_the_heap() {
            let (_guard, mut dl) = with_heap(1024 * 1024);
            let (sz, al) = layout(4096);
            // 40 MiB of traffic through a 1 MiB heap.
            for i in 0..10_000 {
                let p = unsafe { dl.malloc(sz, al) };
                assert!(!p.is_null(), "malloc failed on iteration {i} — no reuse");
                unsafe { dl.free(p, sz, al) };
            }
        }

        #[test]
        fn segments_are_page_aligned_disjoint_and_page_rounded() {
            let (_guard, _dl) = with_heap(1024 * 1024);
            let (first, first_size, flags) = BumpSystem.alloc(PAGE_SIZE + 1);
            assert!(!first.is_null());
            assert_eq!(flags, 0);
            assert_eq!(first as usize % PAGE_SIZE, 0);
            assert_eq!(first_size, 2 * PAGE_SIZE, "size must round up to a page");

            let (second, second_size, _) = BumpSystem.alloc(1);
            assert_eq!(second as usize % PAGE_SIZE, 0);
            assert_eq!(second_size, PAGE_SIZE);
            assert_eq!(
                second as usize,
                first as usize + first_size,
                "segments must be contiguous and non-overlapping"
            );
        }

        #[test]
        fn provider_declines_instead_of_handing_out_memory_past_the_heap() {
            let (_guard, _dl) = with_heap(2 * PAGE_SIZE);
            assert!(!BumpSystem.alloc(PAGE_SIZE).0.is_null());
            assert!(!BumpSystem.alloc(PAGE_SIZE).0.is_null());
            let (ptr, size, _) = BumpSystem.alloc(1);
            assert!(ptr.is_null(), "handed out memory past HEAP_END");
            assert_eq!(size, 0);

            // A request that would overflow the page rounding must also decline, not
            // wrap to a small size and succeed.
            let (ptr, size, _) = BumpSystem.alloc(usize::MAX - 8);
            assert!(ptr.is_null());
            assert_eq!(size, 0);
        }

        /// dlmalloc must return null rather than a bogus pointer once the provider is
        /// exhausted — that null is what reaches `handle_alloc_error` on the guest.
        #[test]
        fn allocation_fails_cleanly_when_the_heap_is_exhausted() {
            let (_guard, mut dl) = with_heap(64 * PAGE_SIZE);
            let (sz, al) = layout(1024 * 1024);
            let mut last = core::ptr::null_mut();
            for _ in 0..8 {
                last = unsafe { dl.malloc(sz, al) };
                if last.is_null() {
                    break;
                }
            }
            assert!(
                last.is_null(),
                "1 MiB allocations never exhausted a 256 KiB heap"
            );
        }

        /// Nothing calls `init` before `init_allocator` on the guest, but a stray
        /// allocation before it must fail closed (HEAP_END == 0) rather than write to
        /// address 0.
        #[test]
        fn uninitialized_provider_hands_out_nothing() {
            let _guard = HEAP_LOCK.lock().unwrap_or_else(|e| e.into_inner());
            init(0, 0);
            assert!(BumpSystem.alloc(1).0.is_null());
        }

        #[test]
        fn realloc_preserves_contents_when_growing() {
            let (_guard, mut dl) = with_heap(1024 * 1024);
            let (sz, al) = layout(128);
            let p = unsafe { dl.malloc(sz, al) };
            assert!(!p.is_null());
            unsafe { core::ptr::write_bytes(p, 0x5A, 128) };

            let grown = unsafe { dl.realloc(p, sz, al, 4096) };
            assert!(!grown.is_null());
            let kept = unsafe { core::slice::from_raw_parts(grown, 128) };
            assert!(
                kept.iter().all(|&b| b == 0x5A),
                "realloc lost the old bytes"
            );
            unsafe { dl.free(grown, 4096, al) };
        }

        #[test]
        fn alignment_requests_are_honored() {
            let (_guard, mut dl) = with_heap(1024 * 1024);
            for align in [16usize, 64, 256, 4096] {
                let p = unsafe { dl.malloc(align * 3, align) };
                assert!(!p.is_null(), "malloc with align {align} failed");
                assert_eq!(p as usize % align, 0, "align {align} not honored");
                unsafe { dl.free(p, align * 3, align) };
            }
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
    // Route through whichever `#[global_allocator]` is installed (dlmalloc or TLSF).
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
