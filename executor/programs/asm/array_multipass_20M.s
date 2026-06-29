	.attribute	5, "rv64i2p1"
	.globl	main
main:
	# Multi-pass array: P passes over an N-word array, each element
	# load+add+store. Touches a LARGE distinct RAM footprint (N words)
	# and REUSES it every pass (so each cell is touched in multiple
	# epochs) -> worst-case stress for the local-to-global table.
	#
	# Footprint = N words = 4*N bytes (here 262144 words = 1 MiB).
	# Steps ~= P * N * 6  (here 13 * 262144 * 6 ~= 20.4M).
	#
	# Tuning knobs:
	#   t5 init (N)  -> distinct footprint (bytes = 4*N)
	#   t6 init (P)  -> number of passes (cross-epoch reuse)
	#   keep P*N*6 ~= target step count.

	li	t3, 1			# increment k
	li	t6, 13			# P = passes
	li	t0, 0x40000000		# BASE = array address (free RAM)

.outer:
	mv	t1, t0			# ptr = BASE
	li	t5, 262144		# N = words per pass
.inner:
	lw	t4, 0(t1)		# t4 = a[i]
	add	t4, t4, t3		# a[i] += k
	sw	t4, 0(t1)		# a[i] = t4
	addi	t1, t1, 4		# ptr += 4
	addi	t5, t5, -1		# i--
	bnez	t5, .inner
	addi	t6, t6, -1		# pass--
	bnez	t6, .outer

	li	a0, 0
	li	a7, 93
	ecall
