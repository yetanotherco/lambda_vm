// Word comparisons on LambdaVM, which supports unaligned scalar loads.
// Bounds are checked before each load; zero length reads no memory.
.section .text.lambda_memcmp,"ax",@progbits
.globl memcmp
.type memcmp,@function
memcmp:
    li t0, 8
    bltu a2, t0, .Lcmp_tail
.Lcmp_word:
    ld t1, 0(a0)
    ld t2, 0(a1)
    bne t1, t2, .Lcmp_byte
    addi a0, a0, 8
    addi a1, a1, 8
    addi a2, a2, -8
    bgeu a2, t0, .Lcmp_word
.Lcmp_tail:
    beqz a2, .Lcmp_equal
.Lcmp_byte:
    lbu t1, 0(a0)
    lbu t2, 0(a1)
    bne t1, t2, .Lcmp_diff
    addi a0, a0, 1
    addi a1, a1, 1
    addi a2, a2, -1
    bnez a2, .Lcmp_byte
.Lcmp_equal:
    li a0, 0
    ret
.Lcmp_diff:
    sub a0, t1, t2
    ret
.size memcmp, .-memcmp
