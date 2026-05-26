	.attribute 5, "rv64i2p1_m2p0_c2p0"
	.option rvc
	.globl main
# Self-checking RV64C test. Each check computes a difference that is zero when the
# compressed instruction behaved correctly, and ORs it into the error accumulator
# s0 (x8). The program exits with a0 = s0, so a0 == 0 iff every check passed.
#
# Compressed register-register ops (c.and/c.or/c.xor/c.sub/c.add/c.mv) only address
# x8..x15, so working values live in s0,s1,a0..a5.
main:
	li      s0, 0                   # error accumulator

	# --- C.LI, C.ADDI, C.MV, C.SUB ---
	c.li    a0, 10
	c.addi  a0, 5                   # a0 = 15
	c.li    a1, 15
	c.mv    a2, a0
	c.sub   a2, a1                  # a2 = 0
	c.or    s0, a2

	# --- C.ADD ---
	c.li    a0, 7
	c.li    a1, 3
	c.add   a0, a1                  # a0 = 10
	c.li    a1, 10
	c.mv    a2, a0
	c.sub   a2, a1
	c.or    s0, a2

	# --- C.AND / C.OR / C.XOR ---
	c.li    a0, 12
	c.li    a1, 10
	c.mv    a2, a0
	c.and   a2, a1                  # 12 & 10 = 8
	c.li    a3, 8
	c.mv    a4, a2
	c.sub   a4, a3
	c.or    s0, a4

	c.mv    a2, a0
	c.or    a2, a1                  # 12 | 10 = 14
	c.li    a3, 14
	c.mv    a4, a2
	c.sub   a4, a3
	c.or    s0, a4

	c.mv    a2, a0
	c.xor   a2, a1                  # 12 ^ 10 = 6
	c.li    a3, 6
	c.mv    a4, a2
	c.sub   a4, a3
	c.or    s0, a4

	# --- C.ANDI / C.SLLI / C.SRLI / C.SRAI ---
	c.li    a0, 15
	c.andi  a0, 10                  # 15 & 10 = 10
	c.li    a1, 10
	c.mv    a2, a0
	c.sub   a2, a1
	c.or    s0, a2

	c.li    a0, 1
	c.slli  a0, 4                   # 1 << 4 = 16
	c.li    a1, 16
	c.mv    a2, a0
	c.sub   a2, a1
	c.or    s0, a2

	c.li    a0, 28
	c.srli  a0, 2                   # 28 >> 2 = 7
	c.li    a1, 7
	c.mv    a2, a0
	c.sub   a2, a1
	c.or    s0, a2

	li      a0, -16
	c.srai  a0, 2                   # -16 >> 2 = -4 (arithmetic)
	li      a1, -4
	c.mv    a2, a0
	c.sub   a2, a1
	c.or    s0, a2

	# --- C.LUI ---
	c.lui   a0, 1                   # a0 = 0x1000
	li      a1, 0x1000
	c.mv    a2, a0
	c.sub   a2, a1
	c.or    s0, a2

	# --- C.SWSP / C.LWSP and C.SDSP / C.LDSP ---
	li      sp, 0x2000
	li      a0, 1234
	c.swsp  a0, 0(sp)
	c.lwsp  a1, 0(sp)               # a1 = 1234
	c.mv    a2, a1
	li      a3, 1234
	c.sub   a2, a3
	c.or    s0, a2

	li      a0, 0x123456789
	c.sdsp  a0, 8(sp)
	c.ldsp  a1, 8(sp)               # full 64-bit round trip
	c.mv    a2, a1
	li      a3, 0x123456789
	c.sub   a2, a3
	c.or    s0, a2

	# --- C.SW / C.LW and C.SD / C.LD (base in x8..x15) ---
	li      a3, 0x3000
	c.li    a0, 30
	c.sw    a0, 0(a3)
	c.lw    a1, 0(a3)               # a1 = 30
	c.mv    a2, a1
	c.li    a4, 30
	c.sub   a2, a4
	c.or    s0, a2

	li      a0, 0x5678abcd
	c.sd    a0, 8(a3)
	c.ld    a1, 8(a3)               # 64-bit round trip
	c.mv    a2, a1
	li      a4, 0x5678abcd
	c.sub   a2, a4
	c.or    s0, a2

	# --- C.ADDI4SPN / C.ADDI16SP ---
	li      sp, 0x4000
	c.addi4spn a0, sp, 8            # a0 = sp + 8 = 0x4008
	li      a1, 0x4008
	c.mv    a2, a0
	c.sub   a2, a1
	c.or    s0, a2

	c.addi16sp sp, 16              # sp = 0x4010
	li      a1, 0x4010
	c.mv    a2, sp
	c.sub   a2, a1
	c.or    s0, a2

	# --- branches: C.BEQZ / C.BNEZ / C.J (with a straddling region) ---
	c.li    a0, 0
	c.beqz  a0, beqz_ok             # taken
	c.li    s1, 1
	c.or    s0, s1                  # only reached on failure
beqz_ok:
	c.li    a0, 5
	c.bnez  a0, bnez_ok             # taken
	c.li    s1, 1
	c.or    s0, s1
bnez_ok:
	c.j     after_j
	c.li    s1, 1
	c.or    s0, s1
after_j:

	# --- call/return: jal (4-byte) + C.JR (return) ---
	c.li    a0, 0
	jal     ra, func                # func sets a0 = 1
	c.li    a1, 1
	c.mv    a2, a0
	c.sub   a2, a1
	c.or    s0, a2

	# --- exit: a0 = s0 (0 on success) ---
	c.mv    a0, s0
	li      a7, 93
	ecall

func:
	c.li    a0, 1
	c.jr    ra
