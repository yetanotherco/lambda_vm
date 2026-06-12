# Signatures

The following lists signatures of the 29 interactions in this VM.

| Signature | Bus size |
|-----------|----------|
| `DECODE[DWordWL, DWordWL, BaseField]` | 5 |
| `CPU32[Byte; DWordWL, DWordWL]` | 5 |
| `ALU[DWordWL; DWordWL, DWordWL, Byte]` | 7 |
| `MEMOP[DWordWL; DWordWL, DWordWL, DWordWL, Byte]` | 9 |
| `BRANCH[DWordWL; DWordWL, DWordWL, DWordWL, Bit]` | 9 |
| `MEMW[BaseField[8]; Bit, DWordWL, BaseField[8], DWordWL, Bit, Bit, Bit]` | 24 |
| `MEMW[Bit, DWordWL, BaseField[8], DWordWL, Bit, Bit, Bit]` | 16 |
| `LOAD[DWordWL; DWordWL, DWordWL, Byte]` | 7 |
| `ECALL[DWordWL, DWordWL]` | 4 |
| `CNB[DWordWL, BaseField, DWordWL, DWordWL]` | 7 |
| `COMMIT[BaseField, Byte]` | 2 |
| `BYTE_ALU[Byte; Byte, Byte, Byte]` | 4 |
| `MSB8[Bit; Byte]` | 2 |
| `MSB16[Bit; Half]` | 2 |
| `ZERO[Bit; B20]` | 2 |
| `ARE_BYTES[Byte, Byte]` | 2 |
| `IS_HALF[Half]` | 1 |
| `IS_B20[B20]` | 1 |
| `HWSL[Half[2]; Half, B4]` | 4 |
| `memory[Bit, DWordWL, DWordWL, BaseField]` | 6 |
| `SHA256_K[Word; BaseField]` | 2 |
| `SHA256_M[Word; DWordWL, BaseField]` | 4 |
| `SHA256ROUND[DWordWL, Word[8], BaseField]` | 11 |
| `ROTXOR[Word; Word, Byte, Byte, Byte, Bit]` | 6 |
| `KECCAK[DWordWL, BaseField, Byte[8][5][5]]` | 203 |
| `KECCAK_RC[Byte[8]; BaseField]` | 9 |
| `ECDAS[DWordWL, U256BL, U256BL, U256BL, U256BL, Byte, Bit]` | 132 |
| `SERVE_K[DWordWL, DWordWL, Byte]` | 5 |
| `BIT[DWordWL, Byte]` | 3 |

Below, we list the signatures of the 6 templates in this VM.

| Signature |
|-----------|
| `BaseField => IS_BIT<BaseField>` |
| `BaseField => IS_BYTE<BaseField>` |
| `BaseField => ADD<DWordWL; DWordWL, DWordWL>` |
| `BaseField => SUB<DWordWL; DWordWL, DWordWL>` |
| `Bit => NEG<DWordWL; DWordHL>` |
| `SIGN<Bit; Half, Bit>` |