.attribute 5, "rv64i2p1_m2p0"
.globl main
main:
    # Program with exactly 32768 (2^15) instructions
    # Setup: 4 instructions
    # Loop: 10921 iterations * 3 instructions = 32763 instructions
    # Halt: 1 instruction
    # Total: 4 + 32763 + 1 = 32768

    lui  t0, 3                # 1: t0 = 3 << 12 = 12288
    addi t0, t0, -1367        # 2: t0 = 12288 - 1367 = 10921
    addi t1, zero, 0          # 3: accumulator = 0
    addi zero, zero, 0        # 4: NOP (padding)

loop:
    addi t1, t1, 1            # acc += 1
    addi t0, t0, -1           # counter--
    bne  t0, zero, loop       # if counter != 0, continue
    # 10921 * 3 = 32763 instructions

    jalr zero, 0(zero)        # halt
