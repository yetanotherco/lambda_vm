.attribute 5, "rv64i2p1_m2p0"
.globl main
main:
    # Registers left alone for longer than a CPU row's register step can
    # reach (2^16 timestamps, 16384 cycles), then read and written: under the
    # in-row register format each such access needs a keep-alive first.
    addi t2, zero, 5          # t2: read after the loop
    addi t3, zero, 7          # t3: written after the loop
    lui  t0, 5                # counter = 20480
loop:
    addi t0, t0, -1
    bne  t0, zero, loop       # 40960 cycles
    add  t4, t2, t2           # reads t2 (a keep-alive) and writes t4 (first access)
    addi t3, t3, 1            # reads and writes t3 (a keep-alive)
    li	a0, 0
    li	a7, 93
	ecall
