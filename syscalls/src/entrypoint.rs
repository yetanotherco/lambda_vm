use core::arch::global_asm;

use crate::{
    allocator::init_allocator,
    syscalls::{
        DMA_MEMCPY_MAX_BYTES, DMA_MEMCPY_SYSCALL_NUMBER, DMA_MEMSET_SYSCALL_NUMBER, sys_halt,
    },
};

/// # Safety
///
/// This function is the default entrypoint and should not be called directly.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn _start() -> ! {
    unsafe extern "C" {
        unsafe fn main();
    }
    init_allocator();
    unsafe {
        main();
        sys_halt();
    }
}

// ---------------------------------------------------------------------------
// DMA memcpy symbol override
//
// `memcpy` is defined next to `_start` on purpose, and not in `syscalls.rs`.
// `compiler_builtins` defines `memcpy` weakly, and a linker extracts an archive
// member only to satisfy an undefined symbol — a weak definition already
// satisfies it, so a strong definition sitting in a member nothing else pulls in
// is silently dropped, with no duplicate-symbol diagnostic. The object defining
// `_start` is always extracted, so co-locating the symbol makes it win
// resolution without `--whole-archive` or any guest link flag. This is the
// "always-linked runtime" mechanism the accelerated-memory-operations standard
// requires vendors to pick and document; see `docs/general_flow.md`.
//
// This placement is insurance, not a repair for an observed failure: in
// `syscalls.rs` the symbol also won resolution, and not by luck — `_start` calls
// `sys_halt` from that module and it is not `#[inline]`, so every guest carries
// an undefined reference that forces the object out of the archive, whatever the
// guest itself names. What the move buys is not depending on that: neither on
// `_start` continuing to call into `syscalls.rs`, nor on rustc's codegen-unit
// merging keeping the two modules together. Only same-module items are
// guaranteed to share an object (partitioning places them together and merging
// never splits), so co-locating with `_start` — the one symbol the linker is
// obliged to resolve — makes the guarantee local.
// `test_dma_memcpy_compiler_emitted_copies` is what detects a regression: a guest
// that falls back still produces correct output, only its ecall count drops.
//
// A Rust `#[no_mangle] fn memcpy` did not reliably override compiler-builtins in
// optimized guests: the final ELF still jumped to compiler_builtins'
// implementation. LLVM still inlines statically-sized tiny copies. Remaining
// out-of-line copies are split into bounded DMA ecalls so a single guest
// instruction cannot create an unbounded continuation trace.
//
// `.p2align 2` is load-bearing: a bare `.section` gives sh_addralign = 1, so the
// linker is free to place `memcpy` at an address that is not a multiple of 4 and
// the VM, which fetches one 4-byte instruction per pc, could not decode it.
// ---------------------------------------------------------------------------

global_asm!(
    r#"
    .section .text.memcpy,"ax",@progbits
    .p2align 2
    .globl memcpy
    .type memcpy,@function
memcpy:
    mv t0, a0
    mv t1, a2
    beqz t1, .Ldma_memcpy_done
.Ldma_memcpy_loop:
    li a2, {max_bytes}
    bgeu t1, a2, .Ldma_memcpy_call
    mv a2, t1
.Ldma_memcpy_call:
    li a7, {syscall}
    ecall
    sub t1, t1, a2
    add a0, a0, a2
    add a1, a1, a2
    bnez t1, .Ldma_memcpy_loop
.Ldma_memcpy_done:
    mv a0, t0
    ret
    .size memcpy, .-memcpy
"#,
    syscall = const DMA_MEMCPY_SYSCALL_NUMBER,
    max_bytes = const DMA_MEMCPY_MAX_BYTES,
);

// ---------------------------------------------------------------------------
// DMA memmove symbol override
//
// Here rather than in `syscalls.rs` for the same reason as `memcpy` above:
// `compiler_builtins` defines `memmove` weakly too, so the strong definition has
// to sit in the object the linker is obliged to extract.
//
// Reuses the memcpy ecall unchanged — no new table, no new syscall. Each ecall
// already snapshots its whole source range before writing (all reads at T+1,
// all writes at T+2), so a single chunk has memmove semantics for free.
//
// Chunking is what breaks it: copying [0,256) -> [4,260) clobbers source bytes
// that a later forward chunk still needs. So when the destination starts inside
// the source range (src < dst < src+n) the chunks are walked from the END
// backwards; every chunk then reads bytes no earlier chunk has written yet.
// Otherwise (disjoint, or dst below src) forward chunking is already safe.
// ---------------------------------------------------------------------------

global_asm!(
    r#"
    .section .text.memmove,"ax",@progbits
    .p2align 2
    .globl memmove
    .type memmove,@function
memmove:
    mv t0, a0
    beqz a2, .Ldma_memmove_done
    bgeu a1, a0, .Ldma_memmove_fwd      // src >= dst: forward is safe
    add t2, a1, a2
    bgeu a0, t2, .Ldma_memmove_fwd      // dst >= src+n: disjoint
    // Overlapping with dst inside [src, src+n): walk chunks from the end.
    add a0, a0, a2
    add a1, a1, a2
    mv t1, a2
.Ldma_memmove_back_loop:
    li a2, {max_bytes}
    bgeu t1, a2, .Ldma_memmove_back_call
    mv a2, t1
.Ldma_memmove_back_call:
    sub a0, a0, a2
    sub a1, a1, a2
    li a7, {syscall}
    ecall
    sub t1, t1, a2
    bnez t1, .Ldma_memmove_back_loop
    j .Ldma_memmove_done
.Ldma_memmove_fwd:
    mv t1, a2
.Ldma_memmove_fwd_loop:
    li a2, {max_bytes}
    bgeu t1, a2, .Ldma_memmove_fwd_call
    mv a2, t1
.Ldma_memmove_fwd_call:
    li a7, {syscall}
    ecall
    sub t1, t1, a2
    add a0, a0, a2
    add a1, a1, a2
    bnez t1, .Ldma_memmove_fwd_loop
.Ldma_memmove_done:
    mv a0, t0
    ret
    .size memmove, .-memmove
"#,
    syscall = const DMA_MEMCPY_SYSCALL_NUMBER,
    max_bytes = const DMA_MEMCPY_MAX_BYTES,
);

// ---------------------------------------------------------------------------
// DMA memset symbol override
//
// Here rather than in `syscalls.rs` for the same reason as `memcpy` above.
//
// memset is expressed as a *propagating* memmove, so it needs no accelerator of its
// own: the stub seeds the first eight bytes with an ordinary store and then calls the
// copy accelerator with `dst = seed_end`, `src = seed_start`. The chip runs that call
// with the read/write timestamp order inverted — it writes at T+1 and reads at T+2 —
// so every step observes the previous step's write and the seed propagates across the
// range. The ecall number is what selects the order; the guest never chooses it.
//
// `a1` therefore carries a source address here, not the fill byte. Fills shorter than
// sixteen bytes take a plain store loop: they cannot amortise the seed, and below eight
// bytes there is nothing left to propagate.
// ---------------------------------------------------------------------------

global_asm!(
    r#"
    .section .text.memset,"ax",@progbits
    .p2align 2
    .globl memset
    .type memset,@function
memset:
    mv t0, a0
    andi a1, a1, 255
    beqz a2, .Ldma_memset_done
    li t2, 16
    bltu a2, t2, .Ldma_memset_bytewise
    // Seed the first eight bytes one at a time. A doubleword store would be shorter
    // but would assume an alignment `dst` does not have: a byte array on the stack is
    // 1-aligned, and seeding it with `sd` is silently wrong there.
    sb a1, 0(a0)
    sb a1, 1(a0)
    sb a1, 2(a0)
    sb a1, 3(a0)
    sb a1, 4(a0)
    sb a1, 5(a0)
    sb a1, 6(a0)
    sb a1, 7(a0)
    mv   t1, a2
    addi t1, t1, -8
    mv   a1, a0
    addi a0, a0, 8
.Ldma_memset_loop:
    li a2, {max_bytes}
    bgeu t1, a2, .Ldma_memset_call
    mv a2, t1
.Ldma_memset_call:
    li a7, {syscall}
    ecall
    sub t1, t1, a2
    add a0, a0, a2
    add a1, a1, a2
    bnez t1, .Ldma_memset_loop
    j .Ldma_memset_done
.Ldma_memset_bytewise:
    mv t1, a2
.Ldma_memset_byte_loop:
    sb a1, 0(a0)
    addi a0, a0, 1
    addi t1, t1, -1
    bnez t1, .Ldma_memset_byte_loop
.Ldma_memset_done:
    mv a0, t0
    ret
    .size memset, .-memset
"#,
    syscall = const DMA_MEMSET_SYSCALL_NUMBER,
    max_bytes = const DMA_MEMCPY_MAX_BYTES,
);
