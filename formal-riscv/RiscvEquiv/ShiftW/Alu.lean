import RiscvEquiv.ShiftW.Bridge

/-!
# Word shifts: what CPU32 learns from the ALU bus is Sail's 32-bit result

The SHIFT clause of `Alu.Guarantees` at `word = 1` (flag byte `opsel(SHIFT) + 1 + 32·signed +
64·direction`). CPU32 sends the low word of `rs1`, zero- or sign-extended (hypothesis `ha`),
and keeps the low word of the result, which it sign-extends to 64 bits before writing `rd`.
So the theorems conclude about `outLo`: it is Sail's 32-bit result. The sign extension is the
CPU32 side and is not covered here.

`shift_word_alu_low` reads `outLo` once; each theorem instantiates it at the flags of its
instruction. The immediate versions are the register ones with `b = shamt`.
-/

namespace RiscvEquiv

open LambdaVM

/-- With `word = 1` only the shift amount mod 32 matters. -/
lemma shiftResult_word_amt (s d x s₁ s₂ : ℕ) (h : s₁ % 32 = s₂ % 32) :
    Alu.shiftResult 1 s d x s₁ = Alu.shiftResult 1 s d x s₂ := by
  simp [Alu.shiftResult, h]

/-- A SHIFT message with `word = 1` and flags `signed = s`, `direction = d` carries, in `outLo`,
the low word of `shiftResult` of its operands. -/
lemma shift_word_alu_low (m : Alu.Msg (F p)) (hg : Alu.Guarantees m)
    (h1 : m.in1Lo.val < 2^32) (h2 : m.in1Hi.val < 2^32)
    (h3 : m.in2Lo.val < 2^32) (h4 : m.in2Hi.val < 2^32)
    (s d : ℕ) (hs : s < 2) (hd : d < 2) (hflags : m.flags.val = opselSHIFT + 1 + 32 * s + 64 * d)
    (x : ℕ) (b : BitVec 64) (ha : dword m.in1Lo m.in1Hi = x)
    (hb : dword m.in2Lo m.in2Hi = b.toNat) :
    m.outLo.val = Alu.shiftResult 1 s d x b.toNat % 2^32 := by
  obtain ⟨-, -, hshift⟩ := hg h1 h2 h3 h4
  obtain ⟨hlo, -, hout⟩ := hshift 1 s d (by norm_num) hs hd hflags
  -- SHIFT reads only the low byte of the shift amount; mod 32 it is the low bits of `b`.
  have hamt : m.in2Lo.val % 256 % 32 = b.toNat % 32 := by
    rw [← hb, dword]
    omega
  have hlow : m.outLo.val = dword m.outLo m.outHi % 2^32 := by
    rw [dword]
    omega
  rw [hlow, hout, ha, shiftResult_word_amt _ _ _ _ _ hamt]

/-- An ALU message with the flags of `SLLW` whose first operand is the zero-extended low word
of `a` carries, in `outLo`, Sail's 32-bit `SLLW` result. -/
theorem sllw_alu_matches_sail (m : Alu.Msg (F p)) (hg : Alu.Guarantees m)
    (h1 : m.in1Lo.val < 2^32) (h2 : m.in1Hi.val < 2^32)
    (h3 : m.in2Lo.val < 2^32) (h4 : m.in2Hi.val < 2^32)
    (hflags : m.flags.val = opselSHIFT + 1)
    (a b : BitVec 64) (ha : dword m.in1Lo m.in1Hi = a.toNat % 2^32)
    (hb : dword m.in2Lo m.in2Hi = b.toNat) :
    m.outLo.val = (sllwPure a b).toNat := by
  rw [sllwPure_toNat]
  exact shift_word_alu_low m hg h1 h2 h3 h4 0 0 (by norm_num) (by norm_num)
    (by simpa using hflags) _ b ha hb

/-- An ALU message with the flags of `SRLW` whose first operand is the zero-extended low word
of `a` carries, in `outLo`, Sail's 32-bit `SRLW` result. -/
theorem srlw_alu_matches_sail (m : Alu.Msg (F p)) (hg : Alu.Guarantees m)
    (h1 : m.in1Lo.val < 2^32) (h2 : m.in1Hi.val < 2^32)
    (h3 : m.in2Lo.val < 2^32) (h4 : m.in2Hi.val < 2^32)
    (hflags : m.flags.val = opselSHIFT + 1 + 64)
    (a b : BitVec 64) (ha : dword m.in1Lo m.in1Hi = a.toNat % 2^32)
    (hb : dword m.in2Lo m.in2Hi = b.toNat) :
    m.outLo.val = (srlwPure a b).toNat := by
  rw [srlwPure_toNat]
  exact shift_word_alu_low m hg h1 h2 h3 h4 0 1 (by norm_num) (by norm_num)
    (by simpa using hflags) _ b ha hb

/-- An ALU message with the flags of `SRAW` whose first operand is the sign-extended low word
of `a` carries, in `outLo`, Sail's 32-bit `SRAW` result. -/
theorem sraw_alu_matches_sail (m : Alu.Msg (F p)) (hg : Alu.Guarantees m)
    (h1 : m.in1Lo.val < 2^32) (h2 : m.in1Hi.val < 2^32)
    (h3 : m.in2Lo.val < 2^32) (h4 : m.in2Hi.val < 2^32)
    (hflags : m.flags.val = opselSHIFT + 1 + 32 + 64)
    (a b : BitVec 64) (ha : dword m.in1Lo m.in1Hi = ((a.setWidth 32).signExtend 64).toNat)
    (hb : dword m.in2Lo m.in2Hi = b.toNat) :
    m.outLo.val = (srawPure a b).toNat := by
  rw [srawPure_toNat]
  exact shift_word_alu_low m hg h1 h2 h3 h4 1 1 (by norm_num) (by norm_num)
    (by simpa using hflags) _ b ha hb

/-- `SLLIW`: `SLLW` with the immediate `shamt` as the second operand. -/
theorem slliw_alu_matches_sail (m : Alu.Msg (F p)) (hg : Alu.Guarantees m)
    (h1 : m.in1Lo.val < 2^32) (h2 : m.in1Hi.val < 2^32)
    (h3 : m.in2Lo.val < 2^32) (h4 : m.in2Hi.val < 2^32)
    (hflags : m.flags.val = opselSHIFT + 1)
    (a : BitVec 64) (shamt : BitVec 5) (ha : dword m.in1Lo m.in1Hi = a.toNat % 2^32)
    (hb : dword m.in2Lo m.in2Hi = shamt.toNat) :
    m.outLo.val = (sllwPure a (shamt.setWidth 64)).toNat :=
  sllw_alu_matches_sail m hg h1 h2 h3 h4 hflags a _ ha (by simpa using hb)

/-- `SRLIW`: `SRLW` with the immediate `shamt` as the second operand. -/
theorem srliw_alu_matches_sail (m : Alu.Msg (F p)) (hg : Alu.Guarantees m)
    (h1 : m.in1Lo.val < 2^32) (h2 : m.in1Hi.val < 2^32)
    (h3 : m.in2Lo.val < 2^32) (h4 : m.in2Hi.val < 2^32)
    (hflags : m.flags.val = opselSHIFT + 1 + 64)
    (a : BitVec 64) (shamt : BitVec 5) (ha : dword m.in1Lo m.in1Hi = a.toNat % 2^32)
    (hb : dword m.in2Lo m.in2Hi = shamt.toNat) :
    m.outLo.val = (srlwPure a (shamt.setWidth 64)).toNat :=
  srlw_alu_matches_sail m hg h1 h2 h3 h4 hflags a _ ha (by simpa using hb)

/-- `SRAIW`: `SRAW` with the immediate `shamt` as the second operand. -/
theorem sraiw_alu_matches_sail (m : Alu.Msg (F p)) (hg : Alu.Guarantees m)
    (h1 : m.in1Lo.val < 2^32) (h2 : m.in1Hi.val < 2^32)
    (h3 : m.in2Lo.val < 2^32) (h4 : m.in2Hi.val < 2^32)
    (hflags : m.flags.val = opselSHIFT + 1 + 32 + 64)
    (a : BitVec 64) (shamt : BitVec 5)
    (ha : dword m.in1Lo m.in1Hi = ((a.setWidth 32).signExtend 64).toNat)
    (hb : dword m.in2Lo m.in2Hi = shamt.toNat) :
    m.outLo.val = (srawPure a (shamt.setWidth 64)).toNat :=
  sraw_alu_matches_sail m hg h1 h2 h3 h4 hflags a _ ha (by simpa using hb)

end RiscvEquiv
