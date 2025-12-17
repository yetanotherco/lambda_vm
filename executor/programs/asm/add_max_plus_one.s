	.attribute	5, "rv32i2p1_m2p0_zmmul1p0"
.Lfunc_end0:
	.globl	main
main:
	addi	a2, zero, 2047 # 2_047
	add     a2, a2, a2     # 4_094
	add     a2, a2, a2     # 8_188
	add     a2, a2, a2     # 16_376
	add     a2, a2, a2     # 32_752
	add     a2, a2, a2     # 65_504
	add     a2, a2, a2     # 131_008
	add     a2, a2, a2     # 262_016
	add     a2, a2, a2     # 524_032
	add     a3, a2, a2     # 1_048_064
	add     a2, a3, a3     # 2_096_128
	add     a2, a2, a2     # 4_192_256
	add     a2, a2, a2     # 8_384_512
	add     a2, a2, a2     # 16_769_024
	add     a2, a2, a2     # 33_538_048
	add     a2, a2, a2     # 67_076_096
	add     a2, a2, a2     # 134_152_192
	add     a2, a2, a2     # 268_304_384
	add     a2, a2, a2     # 536_608_768
	add     a2, a2, a2     # 1_073_217_536
	add     a2, a2, a2     # 2_146_435_072
	add     a2, a2, a3     # 2_147_483_136
	addi    a0, a2, 511    # 2_147_483_647
	addi    a0, a0, 1      # -2_147_483_648 (overflow)
	jalr	zero, 0(ra)
.Lfunc_end1:
	.size	main, .Lfunc_end1-main
