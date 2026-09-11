# About ECALL

ECALLs provide system-level functionalities to the guest program.

When `ECALL` is executed, it is assumed that: - register `A7` contains the system call number

- the arguments are located in registers `A0`-`A6`, and - the return value is written to `A0`, where `A0`-`A7` are symbolic names for the registers `x10`-`x17`

## ECALL number overview

We provide a list of supported ECALL numbers. Negative numbers (represented as 2s complement 64-bit numbers), are used for our own custom accelerators/extensions.

/ 64: `write` ([commit]) / 93: `exit` ([halt]) / -1: `SHA256` ([sha256]) / -2: `KECCAK` ([keccak])