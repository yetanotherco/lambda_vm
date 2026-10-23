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

// `memcpy` for guests built with the `fast-memcpy` feature.
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
// `memmove`, `memset` and `memcmp` stay the library's.
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
