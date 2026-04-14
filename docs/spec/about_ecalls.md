# About ECALL

ECALLs provide system-level functionalities to the guest program.

When `ECALL` is executed, it is assumed that: - register `A7` contains the system call number

- the arguments are located in registers `A0`-`A6`, and - the return value is written to `A0`, where `A0`-`A7` are symbolic names for the registers `x10`-`x17`