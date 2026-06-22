#![no_std]
#![no_main]

extern crate alloc;

use core::alloc::{GlobalAlloc, Layout};
use core::arch::asm;
use core::panic::PanicInfo;
use core::sync::atomic::{AtomicUsize, Ordering};

const PRIVATE_INPUT_START: usize = 0xFF000000;
const SYSCALL_COMMIT: u64 = 64;
const SYSCALL_HALT: u64 = 93;
const MAX_MEMORY_SIZE: usize = 0xC000_0000;

/// A trivial bump allocator for the single-threaded zkVM guest.
///
/// `verify_recursion_blob` allocates once (rkyv metadata, `VmAirs` table
/// constraints, FRI/transition scratch) and then halts — it never frees an
/// individual object. TLSF's free-list bookkeeping is therefore pure overhead
/// (the profile showed `TlsfHeap::alloc` at 43% of TraceCost). This allocator
/// just bumps a pointer: align up, advance, return. `dealloc` is a no-op.
///
/// The arena lives in the address range `[_end, MAX_MEMORY_SIZE)` — exactly
/// where `TlsfHeap` was initialized — so it neither bloats the ELF BSS nor
/// collides with the private-input region at `PRIVATE_INPUT_START`.
struct BumpAllocator {
    /// Next free address (the bump cursor). 0 until `init`.
    next: AtomicUsize,
    /// One-past-the-end of the arena (`MAX_MEMORY_SIZE`). 0 until `init`.
    end: AtomicUsize,
}

impl BumpAllocator {
    const fn new() -> Self {
        Self {
            next: AtomicUsize::new(0),
            end: AtomicUsize::new(0),
        }
    }

    /// Point the arena at `[start, end)`. Called once at guest entry.
    fn init(&self, start: usize, end: usize) {
        self.next.store(start, Ordering::Relaxed);
        self.end.store(end, Ordering::Relaxed);
    }
}

// SAFETY: the guest is single-threaded (single hart). The atomics are used only
// to satisfy the `&self` / interior-mutability requirement of `GlobalAlloc`;
// there is no concurrent contention.
unsafe impl Sync for BumpAllocator {}

unsafe impl GlobalAlloc for BumpAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let end = self.end.load(Ordering::Relaxed);
        let mut cur = self.next.load(Ordering::Relaxed);
        loop {
            let align = layout.align();
            // Align the cursor up to the requested alignment.
            let aligned = (cur + align - 1) & !(align - 1);
            // Bounds check with overflow safety: bail if the request would run
            // off the end of the arena.
            let new_next = match aligned.checked_add(layout.size()) {
                Some(n) if n <= end => n,
                _ => return core::ptr::null_mut(),
            };
            match self.next.compare_exchange_weak(
                cur,
                new_next,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => return aligned as *mut u8,
                Err(observed) => cur = observed,
            }
        }
    }

    unsafe fn dealloc(&self, _ptr: *mut u8, _layout: Layout) {
        // Bump allocator never frees: the guest allocates once and halts.
    }
}

#[global_allocator]
static HEAP: BumpAllocator = BumpAllocator::new();

/// Halt the VM via the `sys_halt` ecall. Used both for normal termination and
/// from the panic handler.
fn halt() -> ! {
    unsafe {
        asm!(
            "ecall",
            in("a0") 0u64,
            in("a7") SYSCALL_HALT,
            options(noreturn),
        );
    }
}

// A guest panic must HALT immediately, not `loop {}`. The executor faithfully
// runs an infinite loop forever — turning any panic-triggering input into an
// unbounded-cycle DoS on the prover. Halting terminates in O(1) cycles (the run
// simply produces no success commitment).
#[panic_handler]
fn panic(_info: &PanicInfo) -> ! {
    halt()
}

fn init_allocator() {
    unsafe extern "C" {
        static _end: u8;
    }
    let heap_pos = (&raw const _end) as usize;
    HEAP.init(heap_pos, MAX_MEMORY_SIZE);
}

/// Read the entire private-input region as a byte slice.
///
/// Layout (per `syscalls::get_private_input`): 4-byte LE length prefix at
/// `PRIVATE_INPUT_START`, payload at +4.
fn read_private_input() -> &'static [u8] {
    let len = unsafe { core::ptr::read_volatile(PRIVATE_INPUT_START as *const u32) } as usize;
    let data = (PRIVATE_INPUT_START + 4) as *const u8;
    unsafe { core::slice::from_raw_parts(data, len) }
}

fn commit(bytes: &[u8]) {
    unsafe {
        asm!(
            "ecall",
            in("a0") 1u64,
            in("a1") bytes.as_ptr(),
            in("a2") bytes.len(),
            in("a7") SYSCALL_COMMIT,
        );
    }
}

/// Private input layout: a 12-byte aligning magic/version prefix followed by an
/// rkyv-archived `lambda_vm_prover::RecursionInput` `{ vm_proof, inner_elf,
/// options, vkey }`. `inner_elf` holds the inner program's ELF bytes, `options`
/// the parameters the inner prover used, and `vkey` the host-derived bitwise
/// preprocessed commitment so the guest can skip the ~87% of verifier cycles
/// that would otherwise be spent recomputing it from scratch. The blob is read
/// zero-copy via `verify_recursion_blob` (which validates the prefix, then reads
/// the 16-aligned archive in place).
#[unsafe(no_mangle)]
pub fn main() -> ! {
    init_allocator();

    let blob = read_private_input();
    // Zero-copy read of the proof bundle: `verify_recursion_blob` validates the
    // aligning prefix and reads the archive in place — no deserialization pass.
    //
    // On any failure (bad prefix, verify error, or proof rejected) we HALT
    // without committing the success marker, rather than panicking — a panic
    // would spin the executor forever (unbounded-cycle DoS on the prover).
    match lambda_vm_prover::verify_recursion_blob(blob) {
        Ok(true) => commit(&[1u8]),
        // Verify errored or the inner proof was rejected: halt with no marker.
        Ok(false) | Err(_) => {}
    }

    halt()
}
