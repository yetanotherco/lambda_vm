#[cfg(not(feature = "fast-memcpy"))]
use core::arch::global_asm;

#[cfg(not(feature = "fast-memcpy"))]
use crate::syscalls::{DMA_MEMCPY_MAX_BYTES, DMA_MEMCPY_SYSCALL_NUMBER, DMA_MEMSET_SYSCALL_NUMBER};
use crate::{allocator::init_allocator, syscalls::sys_halt};

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

// The software memory routines, for guests built with the `fast-memcpy` feature: this
// `memcpy`, and compiler_builtins' `memmove` and `memset`. Without the feature the three
// are the copy accelerator's stubs further down, and this routine is not linked.
//
// compiler_builtins' routine copies a word in 5 instructions when source and destination
// share their alignment and in 10 when they do not (it reassembles every misaligned word
// with shifts), and the ethrex guest spends about a fifth of its instructions in it. This
// VM executes an unaligned doubleword load or store as one instruction, so this routine
// aligns the destination with one (possibly unaligned) doubleword copy, moves four
// doublewords in 11 instructions per iteration with plain loads and stores, and finishes a
// partial word with one overlapping doubleword copy instead of a byte loop. Its stores stay
// on the aligned path inside the 32-byte loop; the head, the tail and every load may be
// unaligned, and the prover's memory argument covers those accesses like any other.
//
// It is defined here, next to `_start`, so that it lands in the object the linker always
// takes, and it is a strong global, so it wins over compiler_builtins' weak `memcpy`.
// `memmove`, `memset` and `memcmp` stay the library's under this feature.
//
// Contract: the C one and no more. The two regions must not overlap, except that they may be
// the very same region (the one overlap the compiler emits, for an object assigned to
// itself). The library routine copies forward a byte or word at a time, so for a destination
// starting below the source it happened to give the right bytes; this routine does not (its
// first and last doublewords re-read source bytes the copy may already have overwritten).
// Rust never asks for that: `copy_from_slice` and `copy_nonoverlapping` cannot overlap in safe
// code, and `ptr::copy` / `copy_within` lower to `memmove`. A C dependency linked into a guest
// would be the first caller that might.
//
// Tests: `executor/programs/rust/mem_sweep` runs it in the VM at every length to 1100 and
// every source and destination offset mod 8 (and more, see that guest), and
// `memcpy_model.rs` checks a transliteration of it on the host for out-of-range accesses.
// Change the two together.
#[cfg(feature = "fast-memcpy")]
core::arch::global_asm!(
    r#"
    .section .text.memcpy,"ax",@progbits
    .p2align 2
    .globl memcpy
    .type memcpy,@function
memcpy:
    # a0 = dst (returned unchanged), a1 = src, a2 = n; t0 walks dst.
    mv   t0, a0
    li   t1, 32
    bltu a2, t1, 5f
    # n >= 32: copy the first doubleword, then advance by 1..=8 so that t0 is aligned.
    ld   t2, 0(a1)
    sd   t2, 0(t0)
    andi t3, t0, 7
    li   t4, 8
    sub  t3, t4, t3
    add  t0, t0, t3
    add  a1, a1, t3
    sub  a2, a2, t3
    # 32 bytes an iteration while at least 32 are left; t5 = where the loop stops.
    andi t5, a2, -32
    beqz t5, 2f
    add  t5, t0, t5
1:
    ld   t2, 0(a1)
    ld   t3, 8(a1)
    ld   t4, 16(a1)
    ld   t6, 24(a1)
    sd   t2, 0(t0)
    sd   t3, 8(t0)
    sd   t4, 16(t0)
    sd   t6, 24(t0)
    addi a1, a1, 32
    addi t0, t0, 32
    bltu t0, t5, 1b
    andi a2, a2, 31
2:
    # Fewer than 32 left: whole doublewords.
    li   t1, 8
    bltu a2, t1, 4f
3:
    ld   t2, 0(a1)
    sd   t2, 0(t0)
    addi a1, a1, 8
    addi t0, t0, 8
    addi a2, a2, -8
    bgeu a2, t1, 3b
4:
    # 1..=7 left: the last doubleword of the copy, overlapping bytes already written.
    beqz a2, 9f
    add  a1, a1, a2
    add  t0, t0, a2
    ld   t2, -8(a1)
    sd   t2, -8(t0)
    ret
5:
    # n < 32: doublewords if n >= 8, then the overlapping last one; else bytes.
    li   t1, 8
    bltu a2, t1, 7f
    add  t5, a1, a2
    add  t6, t0, a2
6:
    ld   t2, 0(a1)
    sd   t2, 0(t0)
    addi a1, a1, 8
    addi t0, t0, 8
    addi a2, a2, -8
    bgeu a2, t1, 6b
    beqz a2, 9f
    ld   t2, -8(t5)
    sd   t2, -8(t6)
    ret
7:
    beqz a2, 9f
8:
    lbu  t2, 0(a1)
    sb   t2, 0(t0)
    addi a1, a1, 1
    addi t0, t0, 1
    addi a2, a2, -1
    bnez a2, 8b
9:
    ret
    .size memcpy, .-memcpy
"#
);

// ---------------------------------------------------------------------------
// DMA memcpy symbol override
//
// The default for every guest: `memcpy`, `memmove` and `memset` below hand the copy
// to the COPY accelerator. A guest that turns on `fast-memcpy` (above) keeps all three
// in software instead, and none of the three stubs is defined.
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

#[cfg(not(feature = "fast-memcpy"))]
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
// already snapshots its whole source range before writing (all reads at T,
// all writes at T+1), so a single chunk has memmove semantics for free.
//
// Chunking is what breaks it: copying [0,256) -> [4,260) clobbers source bytes
// that a later forward chunk still needs. So when the destination starts inside
// the source range (src < dst < src+n) the chunks are walked from the END
// backwards; every chunk then reads bytes no earlier chunk has written yet.
// Otherwise (disjoint, or dst below src) forward chunking is already safe.
// ---------------------------------------------------------------------------

#[cfg(not(feature = "fast-memcpy"))]
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
// with the read/write timestamp order inverted — it writes at T and reads at T+1 —
// so every step observes the previous step's write and the seed propagates across the
// range. The ecall number is what selects the order; the guest never chooses it.
//
// `a1` therefore carries a source address here, not the fill byte, and it is only ever
// read by the `sb`s that lay down the seed.
//
// No range check for the 2^32 limb boundary here, and none is needed: the accelerator
// pins `dst = src + 8` with a single 64-bit `ADD`, carries and all, so a range that
// crosses the boundary is as provable as any other. `sb` writes the low byte of its
// source, so C's `(unsigned char)c` truncation comes for free and needs no masking of
// its own -- a wide or negative `int` fill lands as the right byte either way.
//
// Fills shorter than sixteen bytes take a plain store loop: they cannot amortise the
// seed, and below eight bytes there is nothing left to propagate.
// ---------------------------------------------------------------------------

#[cfg(not(feature = "fast-memcpy"))]
global_asm!(
    r#"
    .section .text.memset,"ax",@progbits
    .p2align 2
    .globl memset
    .type memset,@function
memset:
    mv t0, a0
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
