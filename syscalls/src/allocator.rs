use riscv as _;

const MAX_MEMORY_SIZE: usize = 0xC000_0000;
const WORD_SIZE: usize = 4;

// Guest global allocator, selectable at build time. The default was chosen on measured A/Bs
// against embedded-alloc's TLSF heap (the previous default, now removed) on guest cycles, and
// against dlmalloc on cycles, proving time, proof size and peak RSS: bump spends the fewest
// guest cycles and proves fastest at the epochs worth running, and against dlmalloc it pays
// for that with a 0.6..1.0% larger proof bundle at every epoch and a loss at epoch 2^20, where
// eight epochs amplify the pages its non-reuse touches. Numbers, fixtures and method are in
// #869, which reports the dlmalloc arm; the real block does not resolve the difference.
//
//   - default: a monotonic bump allocator. No free lists and no coalescing -- `alloc` moves
//     a cursor, `dealloc` is empty -- so it spends the fewest guest instructions per
//     allocation. It never reuses a freed region, so its footprint grows monotonically and
//     the proof pays PAGE rows for every page that footprint touches. Touches, not spans: a
//     page the guest allocates but never loads or stores costs nothing, which is why the
//     `alloc_zeroed` memset skip below can leave a large zeroed buffer cheaper here than
//     under an allocator that writes it. See the ceiling note below.
//   - `dlmalloc-alloc` feature: Doug Lea's malloc on a bump "system" provider that hands it
//     page-aligned segments. Slower to prove at the epochs worth running, but its footprint
//     is bounded by live bytes rather than total bytes ever allocated, and it can grow a
//     buried block in place, which bump cannot. Select it for an execution whose churn has
//     no per-block bound, and when proof size or epoch 2^20 is what counts. Nothing selects
//     it today: CI builds and tests the feature on host, but no guest manifest or Makefile
//     rule turns it on, so the riscv64 `#[global_allocator]` below is a fallback with no
//     consumer yet.
//
// Bump's footprint is cumulative allocation, and no gas rule bounds that, so the fit below
// describes honest blocks and is not a safety margin. Measured execute-only over eight ethrex
// fixtures from 0.42M to 63M gas, allocation is linear in gas: 2.55 MB + 2.213 B/gas, marginal
// rate flat (2.18..2.29) across that 150x range, so an honest block has no superlinear term.
// 1500 transfers (31.5M gas) allocate 72.1 MB. The two contract-heavy fixtures average up to
// 3.87 B/gas, but both are small blocks (2.4M and 4.2M gas), so that average still carries the
// ~2.5 MB constant, and no gas-full contract-heavy block has been measured.
//
// What the fit does not bound is an adversarial block, because bytes per gas is chosen by the
// bytecode rather than by the schedule. Every CALL copies its argument region into a fresh heap
// buffer -- levm's `get_calldata` -> `Memory::load_range` -> `Bytes::copy_from_slice` -- sized
// by the caller, fully written, and never reclaimed here; memory expansion is charged once as
// `max(args, retdata)`, so each further warm CALL costs ~100 gas whatever `args_len` is. That
// reaches ~561 B/gas, ~145x the 3.87 above, and `modexp` allocates its operand buffers before
// it charges for them, under a size cap that is fork-gated.
//
// So the operative limit is not the ~3 GiB of [_end, MAX_MEMORY_SIZE) but prover cost, which
// climbs continuously well before it: every touched 256 KiB page adds a 2^18-row PAGE table,
// and on the continuation path a GLOBAL_MEMORY table per page ever touched, which does not
// reset per epoch. Peak prover RAM and bundle size are therefore what decide when to switch,
// not a gas figure.
//
// What spends that budget faster than live bytes suggest is that nothing is ever reclaimed:
// `dealloc` is a no-op, and a grow that cannot extend in place -- the block is not the one the
// cursor sits on -- abandons the old block on top of that. A guest program that processes many
// blocks in one execution has no per-block bound at all, which is what `dlmalloc-alloc` is for.
//
// Exhausting the heap does not fail cleanly today, and what it does instead depends on the
// guest. `alloc` returns null, which reaches `handle_alloc_error`. Every guest that can exhaust
// this heap is a std guest with no `#[panic_handler]` of its own -- ethrex included -- so it
// does not reach a panic handler at all: it goes `__rust_alloc_error_handler` ->
// `default_alloc_error_hook` -> `unimp`, and this VM decodes `unimp` as a write to the
// read-only `cycle` CSR and executes it as a no-op. The hook's epilogue restores `ra` to that
// `unimp` and returns onto it, so execution spins. The `loop {}` panic handlers are in the
// `no_std` guests, none of which allocates.
//
// The sibling abort paths are worse than a spin. `abort()`, `panic_any` with a payload that is
// neither `&str` nor `String`, an empty panic message, and a double panic all reach a bare
// `unimp` that falls through into whatever the linker placed next, and control can reach
// `pc == 0`, which the executor treats as ordinary completion -- a guest that aborted then
// looks like a guest that finished. Nothing on the proving path bounds cycles either
// (`--cycle-budget` is opt-in and only on `execute`). So the fallback matters.
//
// Returning null on exhaustion predates the bump default -- TLSF returned null and hung too --
// and fixing it is not allocator-local: `HALT` constrains `exit_code = 0`, so a nonzero exit
// cannot be proved at all, and a clean abort needs either a committed failure marker or a
// non-provable abort ecall. A host-side cycle bound on `prove` needs neither and bounds the
// spin, but only rejecting writes to read-only CSRs in the decoder turns the fall-through into
// an error instead of a silent success.
//
// Only the guest installs a #[global_allocator]; on host (e.g. `cargo test` for the
// sponge's differential tests) the attribute would hijack the test harness's
// allocator with a never-initialized heap and abort.

// Off riscv only `init` is reachable (no `#[global_allocator]` is installed and
// `sys_alloc_aligned` goes through `std::alloc`), so the plumbing is dead there.
#[cfg(not(feature = "dlmalloc-alloc"))]
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

    /// Idempotent: a later call must not rewind the cursor over live allocations. See
    /// `init_allocator` for why that would be silent corruption and why nothing calls
    /// this twice today. `HEAP_END` doubles as the initialized flag -- `init_allocator`
    /// always passes the nonzero `MAX_MEMORY_SIZE`.
    pub fn init(heap_start: usize, heap_end: usize) {
        let initialized = HEAP_END.load(Ordering::Relaxed) != 0;
        debug_assert!(
            !initialized,
            "allocator init called twice; the cursor would rewind over live allocations"
        );
        if initialized {
            return;
        }
        HEAP_POS.store(heap_start, Ordering::Relaxed);
        HEAP_END.store(heap_end, Ordering::Relaxed);
    }

    // Test-only: `init` is idempotent, so the tests must clear the flag to re-point the
    // global cursor at their own heap.
    #[cfg(test)]
    fn reset() {
        HEAP_END.store(0, Ordering::Relaxed);
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
                // Out of heap -> null, which the caller turns into `handle_alloc_error`.
                // See the module note on why that spins rather than aborting.
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

        unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
            // `GlobalAlloc`'s default allocates a fresh block, copies, and `dealloc`s the old
            // one -- a no-op here, so every grow would abandon its previous buffer. When the
            // block is the one the cursor sits on, extend it in place instead: no copy and
            // nothing abandoned, which makes growing by a constant cost the final size rather
            // than the sum of every intermediate one.
            if (ptr as usize).wrapping_add(layout.size()) == HEAP_POS.load(Ordering::Relaxed) {
                // Shrinking gives the tail up rather than rewinding the cursor: `alloc_zeroed`
                // skips its memset because a region is never served twice, which holds only
                // while the cursor is monotonic.
                if new_size <= layout.size() {
                    return ptr;
                }
                return match (ptr as usize).checked_add(new_size) {
                    Some(end) if end <= HEAP_END.load(Ordering::Relaxed) => {
                        HEAP_POS.store(end, Ordering::Relaxed);
                        ptr
                    }
                    // A fresh block would start at or past `ptr`, so it cannot fit either --
                    // decline without copying.
                    _ => core::ptr::null_mut(),
                };
            }

            // SAFETY: `realloc`'s contract puts `new_size` within the bounds a `Layout` with
            // this align accepts, which is what the default implementation relies on too.
            let new_layout = unsafe { Layout::from_size_align_unchecked(new_size, layout.align()) };
            let new_ptr = unsafe { self.alloc(new_layout) };
            if !new_ptr.is_null() {
                unsafe {
                    core::ptr::copy_nonoverlapping(ptr, new_ptr, layout.size().min(new_size))
                };
            }
            new_ptr
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
            reset();
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

        /// What the in-place path buys, and the reason it exists: growing one buffer by a
        /// constant 1024 times consumes the final size. Under `GlobalAlloc`'s default
        /// `realloc` it would consume the sum of every step -- ~33 MiB here, so this heap
        /// would run out.
        #[test]
        fn incremental_growth_costs_only_the_final_size() {
            let _guard = with_heap(1024 * 1024);
            let base = unsafe { BumpAlloc.alloc(layout(64, 8)) };
            assert!(!base.is_null());
            let mut size = 64usize;
            for _ in 0..1024 {
                let grown = unsafe { BumpAlloc.realloc(base, layout(size, 8), size + 64) };
                assert_eq!(
                    grown, base,
                    "grow past {size} bytes did not extend in place"
                );
                size += 64;
            }
            assert_eq!(
                HEAP_POS.load(Ordering::Relaxed),
                base as usize + size,
                "growth consumed more heap than the final buffer"
            );
        }

        #[test]
        fn growing_the_top_block_keeps_its_contents() {
            let _guard = with_heap(1024 * 1024);
            let base = unsafe { BumpAlloc.alloc(layout(64, 8)) };
            unsafe { core::ptr::write_bytes(base, 0x5A, 64) };

            let grown = unsafe { BumpAlloc.realloc(base, layout(64, 8), 4096) };
            assert_eq!(grown, base);
            let kept = unsafe { core::slice::from_raw_parts(grown, 64) };
            assert!(kept.iter().all(|&b| b == 0x5A), "in-place grow lost bytes");
        }

        /// A block with something allocated after it cannot be extended, so it falls back
        /// to the allocate-and-copy the default `realloc` does.
        #[test]
        fn growing_a_buried_block_copies_it() {
            let _guard = with_heap(1024 * 1024);
            let buried = unsafe { BumpAlloc.alloc(layout(64, 8)) };
            unsafe { core::ptr::write_bytes(buried, 0x5A, 64) };
            let top = unsafe { BumpAlloc.alloc(layout(64, 8)) };
            assert!(!top.is_null());

            let grown = unsafe { BumpAlloc.realloc(buried, layout(64, 8), 128) };
            assert!(!grown.is_null());
            assert_ne!(grown, buried, "a buried block cannot grow in place");
            let kept = unsafe { core::slice::from_raw_parts(grown, 64) };
            assert!(
                kept.iter().all(|&b| b == 0x5A),
                "realloc lost the old bytes"
            );
        }

        /// Shrinking must not rewind the cursor: that would re-serve bytes the guest already
        /// wrote, and `alloc_zeroed` skips its memset on the promise that never happens.
        #[test]
        fn shrinking_does_not_rewind_the_cursor_onto_dirty_bytes() {
            let _guard = with_heap(1024 * 1024);
            let l = layout(4096, 8);
            let block = unsafe { BumpAlloc.alloc(l) };
            unsafe { core::ptr::write_bytes(block, 0xAA, 4096) };
            let cursor = HEAP_POS.load(Ordering::Relaxed);

            let shrunk = unsafe { BumpAlloc.realloc(block, l, 64) };
            assert_eq!(shrunk, block, "a shrink should keep the block where it is");
            assert_eq!(
                HEAP_POS.load(Ordering::Relaxed),
                cursor,
                "the cursor must not rewind over bytes the guest wrote"
            );

            let fresh = unsafe { BumpAlloc.alloc_zeroed(l) };
            assert!(!fresh.is_null());
            let bytes = unsafe { core::slice::from_raw_parts(fresh, 4096) };
            assert!(bytes.iter().all(|&b| b == 0), "alloc_zeroed returned dirt");
        }

        /// Exhaustion on the in-place path declines rather than handing out memory past
        /// `HEAP_END`.
        #[test]
        fn growing_past_the_heap_end_returns_null() {
            let _guard = with_heap(8192);
            let l = layout(4096, 8);
            let block = unsafe { BumpAlloc.alloc(l) };
            assert!(!block.is_null());
            assert!(unsafe { BumpAlloc.realloc(block, l, 16384) }.is_null());
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
            // An absurd size declines too, and on the bounds check rather than on the
            // `checked_add`. The `Layout` invariant alone does not get you there: it
            // gives `size <= isize::MAX - (align - 1)`, and with
            // `aligned <= pos + align - 1` that bounds
            // `aligned + size <= pos + isize::MAX` -- which is `< 2^64` only if
            // `pos < 2^63`. The second half comes from the cursor being heap-bounded:
            // `alloc` stores `new_pos` only when `new_pos <= HEAP_END`, so
            // `pos <= HEAP_END`, and on the guest that is `MAX_MEMORY_SIZE` =
            // 0xC000_0000. The `checked_add` stays: it keeps the no-overflow argument
            // local to `alloc` instead of resting on both of those.
            let huge = layout(isize::MAX as usize - 7, 8);
            assert!(unsafe { BumpAlloc.alloc(huge) }.is_null());
        }

        /// Before `init_allocator` runs HEAP_END is 0 -- allocation must fail closed
        /// rather than hand out address 0.
        #[test]
        fn uninitialized_allocator_hands_out_nothing() {
            let _guard = HEAP_LOCK.lock().unwrap_or_else(|e| e.into_inner());
            reset();
            init(0, 0);
            assert!(unsafe { BumpAlloc.alloc(layout(1, 1)) }.is_null());
        }

        /// A second `init` would rewind the cursor over live allocations, which
        /// `alloc_zeroed`'s missing memset turns into silently dirty memory. In debug
        /// builds the `debug_assert!` is what catches that.
        #[test]
        #[cfg(debug_assertions)]
        #[should_panic(expected = "init called twice")]
        fn a_second_init_is_loud_in_debug() {
            let _guard = with_heap(1024 * 1024);
            init(0, 0);
        }

        /// Guests are built in release, where the `debug_assert!` is compiled out and the
        /// early return is the only thing holding the invariant up.
        #[test]
        #[cfg(not(debug_assertions))]
        fn a_second_init_leaves_the_cursor_alone() {
            let _guard = with_heap(1024 * 1024);
            let first = unsafe { BumpAlloc.alloc(layout(4096, 8)) };
            assert!(!first.is_null());
            init(first as usize, first as usize + 4096);
            let second = unsafe { BumpAlloc.alloc(layout(4096, 8)) };
            assert_eq!(
                second as usize,
                first as usize + 4096,
                "init rewound the cursor over a live allocation"
            );
        }
    }
}

#[cfg(feature = "dlmalloc-alloc")]
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
            // so system-fresh bytes read as 0.
            //
            // This setting is INERT, not a performance win. dlmalloc consults it only
            // through `calloc_must_clear(ptr)` =
            // `!allocates_zeros() || !mmapped(Chunk::from_mem(ptr))`, and `mmapped` is
            // not a marker bit anyone sets — it is `(*p).head & INUSE == 0`, the absence
            // of both in-use bits (dlmalloc 0.2.14 `src/dlmalloc.rs:1805`). Every path
            // that returns a pointer to a caller goes through `set_inuse` /
            // `set_inuse_and_pinuse` / `set_size_and_pinuse_of_inuse_chunk`, all of which
            // set `CINUSE`, and `calloc_must_clear` is only ever evaluated on a user
            // pointer. So no *user* chunk is ever `mmapped`, `calloc_must_clear` is
            // always true, `calloc` always memsets, and flipping this to `false` would
            // change nothing.
            //
            // Flagless heads do exist, so don't reason from "nothing is ever mmapped":
            // `init_top` (dlmalloc.rs:789) writes a segment-end sentinel with
            // `head = top_foot_size()` = 80 on 64-bit, and `80 & INUSE == 0`, so that
            // sentinel *is* `mmapped()`-true. Harmless — it is never returned to a
            // caller, so it never reaches `calloc_must_clear`.
            //
            // Kept `true` for correctness-by-construction if upstream ever grows an mmap
            // path. Locked by `calloc_zeroes_recycled_dirty_blocks` below.
            true
        }

        fn page_size(&self) -> usize {
            PAGE_SIZE
        }
    }

    // Dlmalloc is Send but !Sync, so it can't sit in a static directly. A single-hart
    // critical section serializes access and supplies the Sync a #[global_allocator]
    // static requires. Its single-hart implementation comes from the `riscv` crate.
    //
    // An initialized `Dlmalloc` is address-sensitive and must never be moved:
    // `smallbin_at` returns a pointer into `self.smallbins` and `init_bins` writes
    // self-pointers into that array, so relocating it after first use — into a `Box`, a
    // `OnceCell`, or a local — silently corrupts the bins. Safe as a `static`; the note
    // is for whoever refactors this.
    static DLMALLOC: Mutex<RefCell<Dlmalloc<BumpSystem>>> =
        Mutex::new(RefCell::new(Dlmalloc::new_with_allocator(BumpSystem)));

    struct DlGlobal;

    #[cfg_attr(target_arch = "riscv64", global_allocator)]
    static ALLOC: DlGlobal = DlGlobal;

    /// Idempotent: a later call must not rewind the segment cursor, which would hand
    /// dlmalloc segments overlapping ones it is already using. See `init_allocator` for
    /// the full argument and for why nothing calls this twice today. `HEAP_END` doubles
    /// as the initialized flag -- `init_allocator` always passes the nonzero
    /// `MAX_MEMORY_SIZE`.
    pub fn init(heap_start: usize, heap_end: usize) {
        let initialized = HEAP_END.load(Ordering::Relaxed) != 0;
        debug_assert!(
            !initialized,
            "allocator init called twice; the segment cursor would rewind over live segments"
        );
        if initialized {
            return;
        }
        HEAP_POS.store(heap_start, Ordering::Relaxed);
        HEAP_END.store(heap_end, Ordering::Relaxed);
    }

    // Test-only: `init` is idempotent, so the tests must clear the flag to re-point the
    // global segment cursor at their own heap.
    #[cfg(test)]
    fn reset() {
        HEAP_END.store(0, Ordering::Relaxed);
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
            reset();
            init(base as usize, base as usize + bytes);
            // Moved out by value, which is only sound because it is untouched: an
            // initialized `Dlmalloc` is address-sensitive (see the `DLMALLOC` static).
            // `new_with_allocator` is const and `init_bins` runs on first malloc, which
            // has not happened yet.
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
            reset();
            init(0, 0);
            assert!(BumpSystem.alloc(1).0.is_null());
        }

        /// A second `init` would rewind the segment cursor and hand dlmalloc segments that
        /// overlap ones it is already using. Debug builds catch it on the `debug_assert!`.
        #[test]
        #[cfg(debug_assertions)]
        #[should_panic(expected = "init called twice")]
        fn a_second_init_is_loud_in_debug() {
            let (_guard, _dl) = with_heap(1024 * 1024);
            init(0, 0);
        }

        /// The release path, which is what the guest runs: the early return is the whole
        /// protection.
        #[test]
        #[cfg(not(debug_assertions))]
        fn a_second_init_leaves_the_segment_cursor_alone() {
            let (_guard, _dl) = with_heap(1024 * 1024);
            let (first, size, _) = BumpSystem.alloc(PAGE_SIZE);
            assert!(!first.is_null());
            init(first as usize, first as usize + size);
            let (second, _, _) = BumpSystem.alloc(PAGE_SIZE);
            assert_eq!(
                second as usize,
                first as usize + size,
                "init rewound the segment cursor over a live segment"
            );
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

/// Points the guest allocator at `[_end, MAX_MEMORY_SIZE)`.
///
/// Must run exactly once per execution, and `imp::init` enforces that by ignoring any
/// later call rather than trusting its callers. A second call rewinds the cursor back
/// over live allocations, and because the bump arm's `alloc_zeroed` skips the memset --
/// sound only because bump never re-serves a region -- the next `alloc_zeroed` would
/// then hand back dirty bytes. The guest would compute on garbage and the prover would
/// produce a perfectly valid proof of that wrong execution: no crash, no diagnostic,
/// which is why this is guarded rather than merely documented.
///
/// What makes it once today is an entry-point flag, not the call sites. The six guests
/// that call this explicitly all also override the ELF entry with
/// `-C link-arg=-e -C link-arg=main` in their `.cargo/config.toml`, so `_start` -- the
/// only other caller, in `src/entrypoint.rs` -- never runs for them; guests that do
/// enter through `_start` never call it explicitly. A guest that dropped `-e main` while
/// keeping its explicit call would therefore call this twice, which is why the guard
/// lives in `imp::init` rather than in a comment here.
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
    // Route through whichever `#[global_allocator]` is installed (bump or dlmalloc).
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
