	.data
	.align 3
# `data_page_touch`, but on a page whose genesis is DENSE.
#
# That fixture's .data page holds one .dword, so its INIT column has 112 nonzero
# entries of 262,144 — far below the threshold at which a prepared opening is
# worth taking (`prover/src/genesis_stack.rs`: 9,725 at 18 variables). It
# therefore exercises the SPARSE genesis route and leaves the prepared route
# with no table at fixture scale.
#
# This one surrounds the touched cell with non-zero bytes so the page it lives
# on crosses the threshold whatever offset the linker puts it at.
#
# ⚠ WHY THE FILL IS ON BOTH SIDES. The counter has to be on a DENSE page, and
# where .data starts inside its page is the linker's business. With `n` bytes
# before and `n` after, the counter's own page holds at least `n` of them
# whatever its offset: a counter at the start of a page keeps the trailing fill,
# one at the end keeps the leading fill, and one in the middle keeps both. A
# single trailing fill would leave a counter near the end of a page with almost
# none of it.
#
# 32 KiB a side puts the floor at 32,768 nonzero entries, 3.4x the threshold, so
# the fixture is not sized to just barely qualify.
	.fill 32768, 1, 0xA5
counter:
	.dword 0x123456789ABCDEF0
	.fill 32768, 1, 0x5A

	.text
	.attribute	5, "rv64i2p1"
	.globl	main
main:
	# The same five instructions `data_page_touch` runs: load, mutate, store
	# back a static global, so the page is genuinely ELF-backed and the cell
	# crosses an epoch boundary. Only the page's genesis differs.
	la	t0, counter		# 1: t0 = &counter
	ld	t1, 0(t0)		# 2: t1 = counter (0x123456789ABCDEF0)
	addi	t1, t1, 1		# 3: t1 += 1
	sd	t1, 0(t0)		# 4: counter = t1

	li	a0, 0
	li	a7, 93
	ecall		# 5: Halt
