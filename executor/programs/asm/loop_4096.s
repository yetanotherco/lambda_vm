.attribute 5, "rv64i2p1_m2p0"
.globl main
main:
    # Program with exactly 4096 (2^12) instructions
    # Setup: 4 instructions
    # Loop: 1363 iterations * 3 instructions = 4089 instructions
    # Halt: 3 instructions (li a0, li a7, ecall)
    # Total: 4 + 4089 + 3 = 4096

    addi t0, zero, 1363       # 1: counter = 1363
    addi t1, zero, 0          # 2: accumulator = 0
    addi zero, zero, 0        # 3: NOP (padding)
    addi zero, zero, 0        # 4: NOP (padding)

loop:
    addi t1, t1, 1            # acc += 1
    addi t0, t0, -1           # counter--
    bne  t0, zero, loop       # if counter != 0, continue
    # 1363 * 3 = 4089 instructions

    li	a0, 0
    li	a7, 93
	ecall        # halt
