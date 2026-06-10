	.attribute	5, "rv64i2p1_m2p0_zmmul1p0"
.Lfunc_end0:
	.globl	main
main:
	# Allocate 200 bytes on the stack for the Keccak state (25 × u64).
	addi	sp, sp, -200

	# Initialize a non-zero, deterministic state: lane[i] = i + 1.
	# Used by the host test as the initial state for tiny-keccak::keccakf
	# cross-checking.
	mv	t0, sp
	li	t1, 1
	li	t2, 26
.Linit_loop:
	sd	t1, 0(t0)
	addi	t0, t0, 8
	addi	t1, t1, 1
	bne	t1, t2, .Linit_loop

	# First keccak-f[1600] call.
	mv	a0, sp
	li	a7, -2
	ecall

	# Second keccak-f[1600] call on the result.
	mv	a0, sp
	li	a7, -2
	ecall

	# Third keccak-f[1600] call on the result.
	mv	a0, sp
	li	a7, -2
	ecall

	# Commit the final 200-byte state.
	li	a0, 1
	mv	a1, sp
	li	a2, 200
	li	a7, 64
	ecall

	# Restore stack and halt.
	addi	sp, sp, 200
	li	a0, 0
	li	a7, 93
	ecall
.Lfunc_end1:
	.size	main, .Lfunc_end1-main
