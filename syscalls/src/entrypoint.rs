use core::arch::global_asm;

use crate::{
    allocator::init_allocator,
    syscalls::{DMA_MEMCPY_MAX_BYTES, DMA_MEMCPY_SYSCALL_NUMBER, sys_halt},
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
// `syscalls.rs` the symbol also won resolution, because rustc merged that module
// into a codegen unit every guest already pulled in for `commit` and `sys_halt`.
// What it buys is not depending on that — codegen-unit merging is an internal
// rustc decision, and a guest that referenced nothing else from the module would
// silently get the weak definition. Only same-module items are guaranteed to
// share an object (partitioning places them together and merging never splits),
// so `_start` is what makes the guarantee, and
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
