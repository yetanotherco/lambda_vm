# Spec Deviations Report

This document details every deviation between the prover implementation and the spec found during the audit of 9 tables (Bitwise, Branch, LT, Load, MUL, DVRM, HALT, Decode, CPU). Each deviation is classified by severity and whether it affects soundness.

**12 deviations found**: DVRM (2, LOW), Decode (4, LOW), CPU (6: 4 CRITICAL + 2 MODERATE). Tables Bitwise, Branch, LT, Load, MUL, and HALT had zero deviations.

## Severity Levels

- **CRITICAL**: Could allow invalid proofs (soundness bug)
- **MODERATE**: Functional difference from spec that could cause issues in cross-table interactions
- **LOW**: Cosmetic or over-constrained difference, no soundness impact

---

## DEV-1: DVRM sends IS_HALF for input assumptions (DVRM table)

**Severity**: LOW (overconstrained, sound)

**Spec**: DVRM-A1.i and DVRM-A2.i list `IS_HALF[n[i]]` and `IS_HALF[d[i]]` as **assumptions** — meaning they should be enforced by the sender (CPU), not by the DVRM table itself.

**Code** (`prover/src/tables/dvrm.rs:391-414`): The DVRM table sends 8 extra IS_HALF bus interactions for `n[0..3]` and `d[0..3]` with multiplicity `μ_sum = μ_q + μ_r`.

**Impact**: The Bitwise table (IS_HALF receiver) absorbs these extra lookups. Bus balance still holds because the Bitwise table receives with multiplicity equal to the sum of all senders. The extra sends increase the Bitwise table's multiplicity for these values but do not break soundness — they only add redundant range checks.

If the CPU also sends IS_HALF for these values (as the spec assumption implies), the Bitwise table will see double the multiplicity for DVRM inputs. This is harmless but wasteful.

**Recommendation**: Either (a) remove the 8 extra IS_HALF from DVRM and rely on the CPU to enforce, or (b) update the spec to list these as constraints (C-tags) rather than assumptions (A-tags).

---

## DEV-2: DVRM constraint tag numbering mismatch (DVRM table)

**Severity**: LOW (documentation only)

**Spec**: Constraint tags C1-C22 with specific meanings (e.g., C12 = carry IS_BIT, C9 = MUL lo).

**Code** (`prover/src/tables/dvrm.rs`): Comments use different tag numbers:

| Spec Tag | Code Comment Tag | Constraint |
|----------|-----------------|------------|
| C9 (MUL lo) | C13 | MUL sender lower 64 bits |
| C10 (MUL hi) | C14 | MUL sender upper 64 bits |
| C11 (IS_HALF q) | C15 | IS_HALF for quotient |
| C12 (carry IS_BIT) | C9 | Virtual carry IS_BIT |
| C13 (IS_HALF r) | C10 | IS_HALF for remainder |
| C14 (IS_HALF n_sub_r) | C11 | IS_HALF for n-r |
| C15 (sign_n_sub_r IS_BIT) | C12 | sign_n_sub_r IS_BIT |
| C16 (div_by_zero q) | C19 | div_by_zero ⇒ q=0xFFFF |
| C17 (ZERO div_by_zero) | C20 | ZERO for div_by_zero |
| C18-C20 (SIGN) | C16-C18 | MSB16 + unsigned sign |

**Impact**: No functional impact. All constraints are correctly implemented regardless of tag labels. The mismatch likely arose from the spec evolving after the code was written.

**Recommendation**: Update code comments to use current spec tag numbers, or add a mapping table.

---

## DEV-3: MULW sets `signed=0` instead of `signed=1` (Decode table)

**Severity**: LOW (no correctness impact for current MUL chip)

**Spec** (`docs/spec/decode.md`): The decode table row for `MUL[W]` shows `signed=1` for both MUL and MULW variants.

**Code** (`prover/src/tables/types.rs:738-744`):
```rust
ArithOp::Mul => {
    entry.op_mul = true;
    entry.mp_selector = true;
    if !is_word {
        entry.signed = true;  // Only set for 64-bit MUL, not MULW
    }
}
```

For MULW, `signed=false` because `is_word=true` skips the `signed=true` assignment.

**Analysis**: In RISC-V, MULW multiplies the lower 32 bits of rs1 and rs2, then sign-extends the lower 32 bits of the result to 64 bits. The lower 32 bits of a product are identical regardless of whether the operands are treated as signed or unsigned. Therefore, the MUL chip produces the correct result for MULW regardless of the `signed` flag.

However, the `signed` flag is part of `packed_decode`, which is sent on the DECODE bus. Both CPU and DECODE must agree on this value. Since both use the same `DecodeEntry::packed_decode()` function, they are consistent. The deviation only matters if:
1. The spec is used to generate test vectors independently, or
2. The MUL chip changes to use the `signed` flag differently for word instructions

**Recommendation**: Either update the code to set `signed=true` for MULW (matching spec), or update the spec to reflect `signed=0` for MULW.

---

## DEV-4: ECALL sets `rs2=10, rd=10` instead of `rs2=0, rd=0` (Decode table)

**Severity**: LOW (internally consistent)

**Spec** (`docs/spec/decode.md`): The decode table for ECALL specifies `rs1 := x17` and uses default values for rs2 and rd (which are 0 per the spec's rules: "when a value is not specified by an instruction it defaults to 0").

**Code** (`prover/src/tables/types.rs:688-695`):
```rust
Instruction::EcallEbreak => {
    entry.op_ecall = true;
    entry.rs1 = 17;              // a7 (syscall number)
    entry.read_register1 = true;
    entry.rs2 = 10;              // a0 — NOT in spec
    entry.rd = 10;               // a0 — NOT in spec
    // read_register2, write_register remain false
}
```

**Impact on packed_decode**: Spec would produce bits [35:43]=0 (rs2) and [43:51]=0 (rd). Code produces bits [35:43]=10 and [43:51]=10. These are different packed_decode values.

However, `read_register2=false` and `write_register=false`, so the CPU doesn't actually read rs2 or write rd for ECALL. The registers x10 (a0) and a7 are accessed by the HALT/COMMIT chips via MEMW interactions, not through the CPU's register read/write mechanism.

The CPU table constructs the same packed_decode using the same `DecodeEntry`, so both CPU and DECODE agree. The DECODE bus balances.

**Recommendation**: Either remove `rs2=10, rd=10` from ECALL decoding (matching spec defaults), or update the spec to explicitly list them. Since these values are inert (read_register2=false, write_register=false), removing them would be cleaner.

---

## DEV-5: `read_register1` excludes `rs1=255` (Decode table, CPU table, trace_builder)

**Severity**: LOW (internally consistent, architecturally motivated)

**Spec** (`docs/spec/decode.md`): "read_register1 [...] is set to 1 when [...] rs1 ≠ 0". This implies `read_register1=1` when `rs1=255` (since 255 ≠ 0).

**Code** (three locations, all consistent):
- `prover/src/tables/types.rs:444`: `read_reg1_physical = self.read_register1 && self.rs1 != 0 && self.rs1 != 255`
- `prover/src/tables/cpu.rs:750`: same logic
- `prover/src/tables/trace_builder.rs:479`: same logic

**Analysis**: Register x255 is a **virtual register** representing the program counter (PC). It is used in:
- **AUIPC**: `ADDI rd, x255, imm` — reads PC, adds immediate
- **JAL**: `JALR rd, x255, imm` — reads PC, adds offset

The CPU already has the PC value in its own column — it does not need a MEMW read to obtain it. Setting `read_register1=0` for `rs1=255` means the CPU won't send a MEMW read interaction for the PC register, which is correct because:
1. The CPU has `pc` as a direct column (no memory lookup needed)
2. Sending a MEMW read for x255 would require the MEMW table to have a corresponding entry, adding unnecessary complexity

Since all three locations (DECODE packed_decode, CPU trace, trace_builder) use the same condition, the DECODE bus balances.

**Recommendation**: Update the spec to explicitly state: "`read_register1` is set to 1 when `rs1 ≠ 0` and `rs1 ≠ 255`".

---

## DEV-6: `c_type` (RV64C compressed instructions) not implemented (Decode table)

**Severity**: LOW (feature gap, not a bug)

**Spec** (`docs/spec/decode.md`): Documents the `c_type` flag at packed_decode bit [6] and explains that "the c_type flag should be set to 1 whenever the decoded instruction is provided in compressed form and 0 otherwise." The spec describes full RV64IMC support.

**Code** (`prover/src/tables/types.rs`): The `c_type` field exists in `DecodeEntry` and is correctly placed at bit position 6 in `packed_decode`, but `from_instruction()` never sets `c_type=true`. The executor (`executor/src/`) has no handling for compressed instructions.

**Impact**: Programs using RV64C compressed instructions will fail to parse or execute. The prover currently only supports RV64IM. This is a known feature gap, not a correctness issue for supported instructions.

**Recommendation**: Implement RV64C support or document explicitly that only RV64IM is currently supported.

---

## DEV-7: Missing IS_BIT for `read_register1` (CPU table)

**Severity**: **CRITICAL** (potential soundness bug)

**Spec** (`docs/spec/cpu.md`): Constraint tag `CPU-CR2` specifies `IS_BIT<read_register1>`, requiring `read_register1 * (1 - read_register1) = 0` to constrain the column to binary values {0, 1}.

**Code** (`prover/src/constraints/cpu.rs:59-93`): The `BIT_FLAG_COLUMNS` list includes `write_register` (CR4) but does NOT include `read_register1` (CR2) or `read_register2` (CR3):
```rust
pub const BIT_FLAG_COLUMNS: &[usize] = &[
    cols::WRITE_REGISTER,  // CR4 ✓
    cols::MEMORY_2BYTES,   // CR5 ✓
    // ... (no READ_REGISTER1 or READ_REGISTER2)
];
```

**Impact**: The `read_register1` column is used as the multiplicity for the MEMW register read interaction (CM47). Without IS_BIT, a dishonest prover can set `read_register1` to any field element.

The `packed_decode` verification (C1) provides one linear equation: `2^0 * read_register1 + 2^1 * read_register2 = K - (other constrained terms)`. Since all other terms are constrained (IS_BIT for write_register and all flags, IS_BYTE for rs1/rs2/rd), this gives one equation with two unknowns (`read_register1`, `read_register2`), leaving a degree of freedom.

A malicious prover could manipulate MEMW read multiplicities (e.g., setting `read_register1` to a negative field value to "absorb" sends from other bus participants), potentially compromising the memory argument's soundness.

**Recommendation**: Add `cols::READ_REGISTER1` to `BIT_FLAG_COLUMNS`.

---

## DEV-8: Missing IS_BIT for `read_register2` (CPU table)

**Severity**: **CRITICAL** (potential soundness bug)

**Spec** (`docs/spec/cpu.md`): Constraint tag `CPU-CR3` specifies `IS_BIT<read_register2>`.

**Code** (`prover/src/constraints/cpu.rs:59-93`): `READ_REGISTER2` is absent from `BIT_FLAG_COLUMNS`.

**Impact**: Same as DEV-7. The `read_register2` column is the multiplicity for the MEMW rs2 read interaction (CM49). Without IS_BIT, combined with the packed_decode single-equation constraint shared with `read_register1`, the prover has freedom to set non-binary values.

**Recommendation**: Add `cols::READ_REGISTER2` to `BIT_FLAG_COLUMNS`.

---

## DEV-9: Missing rv1 zero-forcing polynomial (CPU table)

**Severity**: **CRITICAL** (potential soundness bug)

**Spec** (`docs/spec/cpu.md`): Constraint tag `CPU-CM48` specifies:
```
(1 - read_register1) * rv1[i] = 0   for i ∈ [0, 2]
```
This forces rv1 = 0 when `read_register1 = 0` (i.e., when the CPU does not read from rs1).

**Code** (`prover/src/constraints/cpu.rs`): No such constraint exists. The constraint list (56 total) does not include any polynomial enforcing rv1 = 0 when read_register1 = 0.

**Impact**: When `read_register1 = 0`, rv1 is **completely unconstrained**:
- No MEMW read fires (multiplicity = 0), so rv1 is not pinned by the memory argument.
- No zero-forcing polynomial, so rv1 can be any value.

This affects:
1. **x0 (zero register)**: Instructions with `rs1 = 0` have `read_register1 = 0`. rv1 should be 0 (hardwired zero), but the prover can set rv1 to any value, forging the result of `ADDI rd, x0, imm` (and similar).
2. **x255 (virtual PC register)**: `AUIPC` and `JAL` use `rs1 = 255`, `read_register1 = 0`. rv1 should equal `pc`, but the prover can set rv1 to anything, forging the result of `AUIPC` (rd = pc + imm becomes rd = X + imm for arbitrary X).

**Attack scenario**: For `ADDI x1, x0, 42`, the prover sets rv1 = 999. Then arg1 = 999, res = 999 + 42 = 1041, rvd = 1041 is written to x1 via MEMW. All constraints are satisfied. The prover has forged the register value.

**Recommendation**: Add 3 polynomial constraints: `(1 - read_register1) * rv1[i] = 0` for i ∈ {0, 1, 2}.

---

## DEV-10: Missing rv2 zero-forcing polynomial (CPU table)

**Severity**: **CRITICAL** (potential soundness bug)

**Spec** (`docs/spec/cpu.md`): Constraint tag `CPU-CM50` specifies:
```
(1 - read_register2) * rv2[i] = 0   for i ∈ [0, 2]
```

**Code** (`prover/src/constraints/cpu.rs`): No such constraint exists.

**Impact**: When `read_register2 = 0`, rv2 is unconstrained. This primarily affects I-type instructions where `rs2 = 0`:

The arg2 constraint (CE62) computes: `arg2[:4] = (1-LOAD)*rv2[:2] + (1-BEQ-BLT-STORE)*imm[0]`. For a non-LOAD, non-branch, non-STORE instruction (e.g., `ADDI`): arg2 = rv2 + imm. If rv2 ≠ 0, the prover adds an arbitrary offset to arg2.

**Attack scenario**: For `ADDI x1, x2, 10` where `rs2 = 0`: arg2 should be `imm = 10`. Without CM50, the prover sets rv2 = 100, making arg2 = 100 + 10 = 110, which propagates into res and rvd.

**Recommendation**: Add 3 polynomial constraints: `(1 - read_register2) * rv2[i] = 0` for i ∈ {0, 1, 2}.

---

## DEV-11: MUL interaction sends `rvd` instead of `res` (CPU table)

**Severity**: MODERATE (functional difference for word instructions)

**Spec** (`docs/spec/cpu.md`): CPU-CA45 specifies:
```
MUL[res::DWordWL; arg1::DWordHL, signed, arg2::DWordHL, mp_selector, muldiv_selector] | MUL
```
The output is `res` (the raw ALU result).

**Code** (`prover/src/tables/cpu.rs:1233-1268`): The MUL sender uses `RVD_0::DWordWL` as the result:
```rust
// result (rvd) as DWordWL (2 words → 2 elements)
BusValue::Packed {
    start_column: cols::RVD_0,
    packing: Packing::DWordWL,
},
```

**Analysis**: For non-word MUL instructions (MUL, MULH, MULHSU, MULHU), `rvd = res` due to CE65/CE66 constraints (with word_instr=0). No issue.

For MULW (word_instr=1): `rvd = sign_extend(res[31:0], 64)`, which differs from `res` in the upper 32 bits when bit 31 is set. The MUL chip computes the full 64-bit product and is unaware of word_instr. If the MUL chip's result has different upper 32 bits than the sign-extended rvd, the bus won't balance.

Example: arg1 = 0x7FFFFFFF, arg2 = 2 → product lo = 0x00000000FFFFFFFE. rvd = sign_extend(0xFFFFFFFE) = 0xFFFFFFFFFFFFFFFE. These differ → potential bus mismatch.

**Note**: The MUL chip may handle this internally or tests may not exercise this case. Requires cross-table verification.

**Recommendation**: Either send `res::DWordBL` (raw result) instead of `rvd`, or verify the MUL chip produces sign-extended results for word instructions.

---

## DEV-12: DVRM interaction sends `rvd` instead of `res` (CPU table)

**Severity**: MODERATE (functional difference for word instructions)

**Spec** (`docs/spec/cpu.md`): CPU-CA46 specifies:
```
DVRM[res::DWordWL; arg1::DWordHL, arg2::DWordHL, signed, muldiv_selector] | DIVREM
```

**Code** (`prover/src/tables/cpu.rs:1275-1305`): Same pattern as DEV-11 — sends `RVD_0::DWordWL` instead of the raw result.

**Impact**: Same analysis as DEV-11, applied to DIVW and REMW. For word division/remainder, the DVRM chip computes the full 64-bit quotient/remainder, while the CPU sends rvd (sign-extended from 32 bits). Upper 32 bits may differ.

**Recommendation**: Same as DEV-11.

---

## Summary Table

| ID | Table | Description | Severity | Soundness Impact |
|----|-------|-------------|----------|-----------------|
| DEV-1 | DVRM | 8 extra IS_HALF for input assumptions | LOW | None (overconstrained) |
| DEV-2 | DVRM | Constraint tag numbering in comments | LOW | None (documentation) |
| DEV-3 | Decode | MULW `signed=0` vs spec `signed=1` | LOW | None (lower 32-bit multiply identical) |
| DEV-4 | Decode | ECALL `rs2=10, rd=10` vs spec `0, 0` | LOW | None (flags inactive, consistent) |
| DEV-5 | Decode/CPU | `read_register1` excludes `rs1=255` | LOW | None (x255 is virtual PC, consistent) |
| DEV-6 | Decode | RV64C `c_type` not implemented | LOW | None (feature gap) |
| DEV-7 | CPU | **Missing IS_BIT for `read_register1`** | **CRITICAL** | **MEMW multiplicity unconstrained** |
| DEV-8 | CPU | **Missing IS_BIT for `read_register2`** | **CRITICAL** | **MEMW multiplicity unconstrained** |
| DEV-9 | CPU | **Missing rv1 zero-forcing polynomial (CM48)** | **CRITICAL** | **rv1 forgeable for x0/x255 instructions** |
| DEV-10 | CPU | **Missing rv2 zero-forcing polynomial (CM50)** | **CRITICAL** | **rv2 forgeable for I-type instructions** |
| DEV-11 | CPU | MUL sends `rvd` instead of `res` | MODERATE | Potential bus mismatch for MULW |
| DEV-12 | CPU | DVRM sends `rvd` instead of `res` | MODERATE | Potential bus mismatch for DIVW/REMW |

**CRITICAL: 4 findings** (DEV-7 through DEV-10) — all in the CPU table, all missing constraints that the spec requires. DEV-9 and DEV-10 enable register value forgery. DEV-7 and DEV-8 enable multiplicity manipulation.

**MODERATE: 2 findings** (DEV-11, DEV-12) — CPU sends sign-extended `rvd` instead of raw `res` to MUL/DVRM chips, potentially causing bus mismatches for word instructions.

**LOW: 6 findings** (DEV-1 through DEV-6) — no soundness impact.
