module

public import LambdaVM.Tables.Bitwise
public import Clean.Air.CountedChannel
public import LambdaVM.Buses.Alu
public import LambdaVM.Mu
public import LambdaVM.Templates.IsByte

@[expose] public section

/-!
# BYTEWISE (spec/bytewise.typ)

One row per `AND`/`OR`/`XOR` of two 64-bit operands. Columns follow `spec/src/bytewise.toml`:
`a`, `b` and `res` as `DWordBL` (eight bytes, little endian), the operation selector `op`, and
the multiplicity `μ`.

The row looks each of the eight byte triples up in `BYTE_ALU` with the same `op`, and pushes
`ALU[res; a, b, op]` with the operands re-limbed to `DWordWL` (four bytes per word). The
arithmetic that justifies the re-limbing is `Bytewise.compose`: an operation that acts on each
bit independently commutes with reading a list of bytes as a little-endian number.

Two things the chip does *not* constrain, both on purpose:

* **`op` itself.** `BYTE_ALU` only has rows for `AND`, `OR` and `XOR`, so an active row can only
  ever push one of those three on the `ALU` bus. The lookup is the range check.
* **The bytes of `a`, `b` and `res`.** They come back as byte-ranged from `BYTE_ALU`'s own
  guarantee, so soundness needs no assumption about them.

`IS_BYTE[μ]` is a constraint of this chip, not an assumption: it is what makes every `BYTE_ALU`
pull a counted pull of multiplicity at most `muMax`. `spec/main` added it; the older copy of the
spec in this repo still types `μ` as `BaseField` and omits it, and with an unbounded `μ` the
counted pulls would have no sound bound at all.
-/

namespace LambdaVM.BYTEWISE

open Bitwise (ByteAlu ByteAluMsg byteAluTable AreBytes areBytesTable)

/-- `BYTE_ALU` with pulls of multiplicity up to `muMax`. A BYTEWISE row's own `μ` is a byte, as
`spec/src/bytewise.toml` types it and as `MU_MAX` fixes it in `prover/src/tables/types.rs`; the
channel's bound is the model-wide one of `LambdaVM.muMax`. -/
abbrev ByteAluC : RawChannel (F p) := ByteAlu.counted muMax

/-! ## Bitwise operations compose from bytes

`BYTE_ALU` gives the operation on each byte. The `ALU` bus carries the operation on the whole
64-bit value. These lemmas are the bridge: an operation that acts on each bit independently
commutes with reading bytes as a little-endian number. -/

namespace Bits

/-- An operation on naturals that acts on each bit independently, as `f` on the bits. -/
def Bitwise (g : ℕ → ℕ → ℕ) (f : Bool → Bool → Bool) : Prop :=
  f false false = false ∧ ∀ a b j, (g a b).testBit j = f (a.testBit j) (b.testBit j)

theorem bitwise_and : Bitwise (· &&& ·) Bool.and := ⟨rfl, fun a b j => Nat.testBit_and a b j⟩
theorem bitwise_or : Bitwise (· ||| ·) Bool.or := ⟨rfl, fun a b j => Nat.testBit_or a b j⟩
theorem bitwise_xor : Bitwise (· ^^^ ·) bne := ⟨rfl, fun a b j => Nat.testBit_xor a b j⟩

theorem Bitwise.zero {g f} (h : Bitwise g f) : g 0 0 = 0 := by
  refine Nat.eq_of_testBit_eq fun j => ?_
  rw [h.2]; simp [h.1]

/-- A bitwise operation splits off the low byte. -/
theorem Bitwise.cons {g f} (h : Bitwise g f) {x y X Y : ℕ} (hx : x < 2^8) (hy : y < 2^8)
    (hxy : g x y < 2^8) : g (2^8 * X + x) (2^8 * Y + y) = 2^8 * g X Y + g x y := by
  refine Nat.eq_of_testBit_eq fun j => ?_
  rw [h.2, Nat.testBit_two_pow_mul_add _ hx, Nat.testBit_two_pow_mul_add _ hy,
    Nat.testBit_two_pow_mul_add _ hxy]
  split <;> rw [h.2]

theorem and_lt {x y : ℕ} (hx : x < 2^8) : x &&& y < 2^8 := by
  rw [Nat.and_comm]; exact Nat.and_lt_two_pow _ hx

theorem or_lt {x y : ℕ} (hx : x < 2^8) (hy : y < 2^8) : x ||| y < 2^8 := Nat.or_lt_two_pow hx hy

theorem xor_lt {x y : ℕ} (hx : x < 2^8) (hy : y < 2^8) : x ^^^ y < 2^8 := Nat.xor_lt_two_pow hx hy

/-- The operation that `BYTE_ALU`'s `opsel` selects, on naturals. Outside the three values the
table has rows for, the choice is immaterial: an active row can never push such an `op`. -/
def appOf (op x y : ℕ) : ℕ :=
  if op = opselAND then x &&& y else if op = opselOR then x ||| y else x ^^^ y

/-- Eight bytes read as a 64-bit little-endian number. -/
def ofBytes (x0 x1 x2 x3 x4 x5 x6 x7 : ℕ) : ℕ :=
  x0 + 2^8 * x1 + 2^16 * x2 + 2^24 * x3 + 2^32 * x4 + 2^40 * x5 + 2^48 * x6 + 2^56 * x7

/-- A list of bytes read as a little-endian number. -/
def horner : List ℕ → ℕ
  | [] => 0
  | x :: xs => 2^8 * horner xs + x

/-- A bitwise operation that keeps bytes as bytes commutes with `horner`. The induction is on the
list, so the eight-byte case below is one instance of it. -/
theorem horner_zipWith {g f} (h : Bitwise g f) (hlt : ∀ x y, x < 2^8 → y < 2^8 → g x y < 2^8) :
    ∀ (l m : List ℕ), l.length = m.length → (∀ x ∈ l, x < 2^8) → (∀ y ∈ m, y < 2^8) →
      horner (List.zipWith g l m) = g (horner l) (horner m)
  | [], [], _, _, _ => by simp [horner, h.zero]
  | x :: xs, y :: ys, hlen, hl, hm => by
    have hx := hl x (by simp)
    have hy := hm y (by simp)
    simp only [List.zipWith_cons_cons, horner]
    rw [horner_zipWith h hlt xs ys (by simpa using hlen) (fun z hz => hl z (by simp [hz]))
      (fun z hz => hm z (by simp [hz])), h.cons hx hy (hlt x y hx hy)]

/-- `ofBytes` is `horner` of the eight bytes. -/
theorem ofBytes_eq_horner (x0 x1 x2 x3 x4 x5 x6 x7 : ℕ) :
    ofBytes x0 x1 x2 x3 x4 x5 x6 x7 = horner [x0, x1, x2, x3, x4, x5, x6, x7] := by
  simp only [ofBytes, horner]; ring

/-- A bitwise operation that keeps bytes as bytes commutes with `ofBytes`: applying it to each of
the eight byte pairs is applying it to the two 64-bit values. -/
theorem compose {g f} (h : Bitwise g f) (hlt : ∀ x y, x < 2^8 → y < 2^8 → g x y < 2^8)
    {a0 a1 a2 a3 a4 a5 a6 a7 b0 b1 b2 b3 b4 b5 b6 b7 : ℕ}
    (ha0 : a0 < 2^8) (ha1 : a1 < 2^8) (ha2 : a2 < 2^8) (ha3 : a3 < 2^8)
    (ha4 : a4 < 2^8) (ha5 : a5 < 2^8) (ha6 : a6 < 2^8) (ha7 : a7 < 2^8)
    (hb0 : b0 < 2^8) (hb1 : b1 < 2^8) (hb2 : b2 < 2^8) (hb3 : b3 < 2^8)
    (hb4 : b4 < 2^8) (hb5 : b5 < 2^8) (hb6 : b6 < 2^8) (hb7 : b7 < 2^8) :
    g (ofBytes a0 a1 a2 a3 a4 a5 a6 a7) (ofBytes b0 b1 b2 b3 b4 b5 b6 b7)
      = ofBytes (g a0 b0) (g a1 b1) (g a2 b2) (g a3 b3)
          (g a4 b4) (g a5 b5) (g a6 b6) (g a7 b7) := by
  rw [ofBytes_eq_horner a0, ofBytes_eq_horner b0, ofBytes_eq_horner (g a0 b0),
    ← horner_zipWith h hlt [a0, a1, a2, a3, a4, a5, a6, a7] [b0, b1, b2, b3, b4, b5, b6, b7]
      rfl (by simp only [List.mem_cons, List.not_mem_nil, or_false]; omega)
      (by simp only [List.mem_cons, List.not_mem_nil, or_false]; omega)]
  simp only [List.zipWith_cons_cons, List.zipWith_nil_right]

/-- Every operation the table selects keeps bytes as bytes. -/
theorem appOf_lt (op : ℕ) {x y : ℕ} (hx : x < 2^8) (hy : y < 2^8) : appOf op x y < 2^8 := by
  simp only [appOf]
  split_ifs
  · exact and_lt hx
  · exact or_lt hx hy
  · exact xor_lt hx hy

/-- `appOf op` composes from bytes, whichever of the three operations `op` selects. This is the
form the soundness proof uses: `BYTE_ALU` gives one byte at a time, the `ALU` bus wants the
whole 64-bit value. -/
theorem compose_appOf (op : ℕ)
    {a0 a1 a2 a3 a4 a5 a6 a7 b0 b1 b2 b3 b4 b5 b6 b7 : ℕ}
    (ha0 : a0 < 2^8) (ha1 : a1 < 2^8) (ha2 : a2 < 2^8) (ha3 : a3 < 2^8)
    (ha4 : a4 < 2^8) (ha5 : a5 < 2^8) (ha6 : a6 < 2^8) (ha7 : a7 < 2^8)
    (hb0 : b0 < 2^8) (hb1 : b1 < 2^8) (hb2 : b2 < 2^8) (hb3 : b3 < 2^8)
    (hb4 : b4 < 2^8) (hb5 : b5 < 2^8) (hb6 : b6 < 2^8) (hb7 : b7 < 2^8) :
    appOf op (ofBytes a0 a1 a2 a3 a4 a5 a6 a7) (ofBytes b0 b1 b2 b3 b4 b5 b6 b7)
      = ofBytes (appOf op a0 b0) (appOf op a1 b1) (appOf op a2 b2) (appOf op a3 b3)
          (appOf op a4 b4) (appOf op a5 b5) (appOf op a6 b6) (appOf op a7 b7) := by
  simp only [appOf]
  split_ifs
  · exact compose bitwise_and (fun _ _ hx _ => and_lt hx)
      ha0 ha1 ha2 ha3 ha4 ha5 ha6 ha7 hb0 hb1 hb2 hb3 hb4 hb5 hb6 hb7
  · exact compose bitwise_or (fun _ _ hx hy => or_lt hx hy)
      ha0 ha1 ha2 ha3 ha4 ha5 ha6 ha7 hb0 hb1 hb2 hb3 hb4 hb5 hb6 hb7
  · exact compose bitwise_xor (fun _ _ hx hy => xor_lt hx hy)
      ha0 ha1 ha2 ha3 ha4 ha5 ha6 ha7 hb0 hb1 hb2 hb3 hb4 hb5 hb6 hb7

end Bits

/-- One BYTEWISE row, fields in the column order of `spec/src/bytewise.toml`: `a`, `b` and `res`
as `DWordBL` bytes, little endian. -/
structure Row (F : Type) where
  a0 : F
  a1 : F
  a2 : F
  a3 : F
  a4 : F
  a5 : F
  a6 : F
  a7 : F
  b0 : F
  b1 : F
  b2 : F
  b3 : F
  b4 : F
  b5 : F
  b6 : F
  b7 : F
  op : F
  res0 : F
  res1 : F
  res2 : F
  res3 : F
  res4 : F
  res5 : F
  res6 : F
  res7 : F
  mu : F
deriving ProvableStruct

/-- The first operand as a 64-bit value. -/
def Row.aVal (r : Row (F p)) : ℕ :=
  Bits.ofBytes r.a0.val r.a1.val r.a2.val r.a3.val r.a4.val r.a5.val r.a6.val r.a7.val

/-- The second operand as a 64-bit value. -/
def Row.bVal (r : Row (F p)) : ℕ :=
  Bits.ofBytes r.b0.val r.b1.val r.b2.val r.b3.val r.b4.val r.b5.val r.b6.val r.b7.val

/-- The result as a 64-bit value. -/
def Row.resVal (r : Row (F p)) : ℕ :=
  Bits.ofBytes r.res0.val r.res1.val r.res2.val r.res3.val r.res4.val r.res5.val r.res6.val
    r.res7.val

/-- Four bytes as the `DWordWL` word the `ALU` bus carries, little endian. -/
@[circuit_norm]
def word (x0 x1 x2 x3 : Expression (F p)) : Expression (F p) :=
  x0 + 256 * x1 + 65536 * x2 + 16777216 * x3

def main (r : Var Row (F p)) : Circuit (F p) Unit := do
  -- ARE_BYTES[μ, 0]: what makes every BYTE_ALU pull below a counted pull of at most `muMax`.
  -- Spelt as `bytewise.rs` and `lt.rs` send it, operand first, rather than as `IS_BYTE<1, μ>`,
  -- which is the same statement but the other message and so a different bus entry.
  areBytes r.mu 0
  -- BYTE_ALU[op, a[i], b[i]] -> res[i], pulled μ times, for i = 0 … 7
  ByteAlu.pullUpTo muMax r.mu ⟨r.op, r.a0, r.b0, r.res0⟩
  ByteAlu.pullUpTo muMax r.mu ⟨r.op, r.a1, r.b1, r.res1⟩
  ByteAlu.pullUpTo muMax r.mu ⟨r.op, r.a2, r.b2, r.res2⟩
  ByteAlu.pullUpTo muMax r.mu ⟨r.op, r.a3, r.b3, r.res3⟩
  ByteAlu.pullUpTo muMax r.mu ⟨r.op, r.a4, r.b4, r.res4⟩
  ByteAlu.pullUpTo muMax r.mu ⟨r.op, r.a5, r.b5, r.res5⟩
  ByteAlu.pullUpTo muMax r.mu ⟨r.op, r.a6, r.b6, r.res6⟩
  ByteAlu.pullUpTo muMax r.mu ⟨r.op, r.a7, r.b7, r.res7⟩
  -- ALU[res; a, b, op], pushed μ times, operands and result re-limbed to `DWordWL`
  Alu.channel.pushIf r.mu
    ⟨word r.a0 r.a1 r.a2 r.a3, word r.a4 r.a5 r.a6 r.a7,
     word r.b0 r.b1 r.b2 r.b3, word r.b4 r.b5 r.b6 r.b7,
     r.op,
     word r.res0 r.res1 r.res2 r.res3, word r.res4 r.res5 r.res6 r.res7⟩

/-- The honest prover's row, as the trace generator fills it: a byte multiplicity, byte limbs,
and a result that is the selected operation applied byte by byte. -/
def Honest (r : Row (F p)) : Prop :=
  r.mu.val < 2^8 ∧
  r.a0.val < 2^8 ∧ r.a1.val < 2^8 ∧ r.a2.val < 2^8 ∧ r.a3.val < 2^8 ∧
  r.a4.val < 2^8 ∧ r.a5.val < 2^8 ∧ r.a6.val < 2^8 ∧ r.a7.val < 2^8 ∧
  r.b0.val < 2^8 ∧ r.b1.val < 2^8 ∧ r.b2.val < 2^8 ∧ r.b3.val < 2^8 ∧
  r.b4.val < 2^8 ∧ r.b5.val < 2^8 ∧ r.b6.val < 2^8 ∧ r.b7.val < 2^8 ∧
  (r.op.val = opselAND ∨ r.op.val = opselOR ∨ r.op.val = opselXOR) ∧
  r.res0.val = Bits.appOf r.op.val r.a0.val r.b0.val ∧
  r.res1.val = Bits.appOf r.op.val r.a1.val r.b1.val ∧
  r.res2.val = Bits.appOf r.op.val r.a2.val r.b2.val ∧
  r.res3.val = Bits.appOf r.op.val r.a3.val r.b3.val ∧
  r.res4.val = Bits.appOf r.op.val r.a4.val r.b4.val ∧
  r.res5.val = Bits.appOf r.op.val r.a5.val r.b5.val ∧
  r.res6.val = Bits.appOf r.op.val r.a6.val r.b6.val ∧
  r.res7.val = Bits.appOf r.op.val r.a7.val r.b7.val

/-- What an active row establishes: the result is the selected bitwise operation of the two
64-bit operands. The operand ranges are not hypotheses here — `BYTE_ALU` supplies them. -/
def RowSpec (r : Row (F p)) : Prop :=
  r.mu ≠ 0 → r.resVal = Bits.appOf r.op.val r.aVal r.bVal

/-- The three `opsel` values are distinct in the field: `p > 2^34`, so `0`, `1` and `2` do not
collide. Used to resolve the disjunction each `BYTE_ALU` guarantee carries. -/
theorem opsel_val_of_eq {op : F p} (h : op = ((opselAND : ℕ) : F p) ∨ op = ((opselOR : ℕ) : F p)
    ∨ op = ((opselXOR : ℕ) : F p)) :
    op.val = opselAND ∨ op.val = opselOR ∨ op.val = opselXOR := by
  rcases h with h | h | h
  · exact Or.inl (val_eq_of_eq_natCast h (by norm_num [opselAND]))
  · exact Or.inr (Or.inl (val_eq_of_eq_natCast h (by norm_num [opselOR])))
  · exact Or.inr (Or.inr (val_eq_of_eq_natCast h (by norm_num [opselXOR])))

/-- Four bytes as a word, read on naturals: the sum never wraps. -/
lemma val_word {x0 x1 x2 x3 : F p} (h0 : x0.val < 2^8) (h1 : x1.val < 2^8) (h2 : x2.val < 2^8)
    (h3 : x3.val < 2^8) :
    (x0 + 256 * x1 + 65536 * x2 + 16777216 * x3).val
      = x0.val + 256 * x1.val + 65536 * x2.val + 16777216 * x3.val :=
  val_eq_of_eq_natCast (by push_cast; simp only [ZMod.natCast_zmod_val]) (by omega)

/-- One `BYTE_ALU` guarantee, read as `appOf` applied to the byte values. The three branches of
the table's disjunction are exactly the three branches of `appOf`. -/
theorem appOf_of_guarantee {op a b res : F p}
    (h : (op = ((opselAND : ℕ) : F p) ∧ res.val = a.val &&& b.val) ∨
         (op = ((opselOR : ℕ) : F p) ∧ res.val = a.val ||| b.val) ∨
         (op = ((opselXOR : ℕ) : F p) ∧ res.val = a.val ^^^ b.val)) :
    res.val = Bits.appOf op.val a.val b.val := by
  rcases h with ⟨hop, hres⟩ | ⟨hop, hres⟩ | ⟨hop, hres⟩ <;>
    rw [hres, val_eq_of_eq_natCast hop (by norm_num [opselAND, opselOR, opselXOR])] <;>
    simp [Bits.appOf, opselAND, opselOR, opselXOR]

/-- The converse of `appOf_of_guarantee`: an honest row's byte equation is one branch of the
`BYTE_ALU` disjunction. Used for completeness. -/
theorem guarantee_of_appOf {op a b res : F p}
    (hop : op.val = opselAND ∨ op.val = opselOR ∨ op.val = opselXOR)
    (hres : res.val = Bits.appOf op.val a.val b.val) :
    (op = ((opselAND : ℕ) : F p) ∧ res.val = a.val &&& b.val) ∨
    (op = ((opselOR : ℕ) : F p) ∧ res.val = a.val ||| b.val) ∨
    (op = ((opselXOR : ℕ) : F p) ∧ res.val = a.val ^^^ b.val) := by
  have hv : ∀ n : ℕ, op.val = n → op = (n : F p) := fun n h => by rw [← h, ZMod.natCast_zmod_val]
  rcases hop with h | h | h
  · exact Or.inl ⟨hv _ h, by rw [hres, h]; simp [Bits.appOf, opselAND]⟩
  · exact Or.inr (Or.inl ⟨hv _ h, by rw [hres, h]; simp [Bits.appOf, opselAND, opselOR]⟩)
  · exact Or.inr (Or.inr ⟨hv _ h, by
      rw [hres, h]; simp [Bits.appOf, opselAND, opselOR, opselXOR]⟩)

def circuit : GeneralFormalCircuit (F p) Row unit where
  name := "BYTEWISE"
  main
  Spec r _ _ := RowSpec r
  channelsWithRequirements := [Alu.channel.toRaw, ByteAluC, Bitwise.AreBytes.toRaw]

  requirementsChannelsLawful input_var i₀ := by
    obtain ⟨a0, a1, a2, a3, a4, a5, a6, a7, b0, b1, b2, b3, b4, b5, b6, b7, op,
      res0, res1, res2, res3, res4, res5, res6, res7, mu⟩ := input_var
    simp only [circuit_norm, main, areBytes, word, ByteAluC, Bitwise.ByteAlu, Bitwise.AreBytes,
      Alu.channel]
    intro channel h
    tauto

  soundness := by
    circuit_proof_start [main, areBytes, word, Alu.channel, Alu.Guarantees, RowSpec,
      Row.aVal, Row.bVal, Row.resVal, Bits.ofBytes, dword, Bitwise.areBytesTable,
      Bitwise.byteAluTable]
    obtain ⟨⟨hmu, -⟩, g0, g1, g2, g3, g4, g5, g6, g7⟩ := h_holds
    -- an inactive row owes nothing: no lookup fires and nothing is pushed
    by_cases h0 : input_mu = 0
    · simp [h0]
    -- IS_BYTE[μ] bounds μ, so every BYTE_ALU pull is a counted pull and delivers
    have hcount : Channel.CountUpTo muMax input_mu :=
      Channel.countUpTo_of_val h0 (by simp only [muMax]; omega)
    obtain ⟨ha0, hb0, hd0⟩ := g0 hcount
    obtain ⟨ha1, hb1, hd1⟩ := g1 hcount
    obtain ⟨ha2, hb2, hd2⟩ := g2 hcount
    obtain ⟨ha3, hb3, hd3⟩ := g3 hcount
    obtain ⟨ha4, hb4, hd4⟩ := g4 hcount
    obtain ⟨ha5, hb5, hd5⟩ := g5 hcount
    obtain ⟨ha6, hb6, hd6⟩ := g6 hcount
    obtain ⟨ha7, hb7, hd7⟩ := g7 hcount
    -- each lookup reads as `appOf` on its byte pair
    have e0 := appOf_of_guarantee hd0
    have e1 := appOf_of_guarantee hd1
    have e2 := appOf_of_guarantee hd2
    have e3 := appOf_of_guarantee hd3
    have e4 := appOf_of_guarantee hd4
    have e5 := appOf_of_guarantee hd5
    have e6 := appOf_of_guarantee hd6
    have e7 := appOf_of_guarantee hd7
    -- the eight share `op`, so one disjunction fixes which operation it is
    have hop : input_op.val = opselAND ∨ input_op.val = opselOR ∨ input_op.val = opselXOR :=
      opsel_val_of_eq (by
        rcases hd0 with ⟨h, -⟩ | ⟨h, -⟩ | ⟨h, -⟩
        exacts [Or.inl h, Or.inr (Or.inl h), Or.inr (Or.inr h)])
    -- the result bytes are bytes, so the two result words are words
    have hr0 : input_res0.val < 2^8 := e0 ▸ Bits.appOf_lt _ ha0 hb0
    have hr1 : input_res1.val < 2^8 := e1 ▸ Bits.appOf_lt _ ha1 hb1
    have hr2 : input_res2.val < 2^8 := e2 ▸ Bits.appOf_lt _ ha2 hb2
    have hr3 : input_res3.val < 2^8 := e3 ▸ Bits.appOf_lt _ ha3 hb3
    have hr4 : input_res4.val < 2^8 := e4 ▸ Bits.appOf_lt _ ha4 hb4
    have hr5 : input_res5.val < 2^8 := e5 ▸ Bits.appOf_lt _ ha5 hb5
    have hr6 : input_res6.val < 2^8 := e6 ▸ Bits.appOf_lt _ ha6 hb6
    have hr7 : input_res7.val < 2^8 := e7 ▸ Bits.appOf_lt _ ha7 hb7
    -- the eight byte operations assemble into the operation on the 64-bit values
    have hcomp := Bits.compose_appOf input_op.val ha0 ha1 ha2 ha3 ha4 ha5 ha6 ha7
      hb0 hb1 hb2 hb3 hb4 hb5 hb6 hb7
    simp only [Bits.ofBytes] at hcomp
    have hspec : input_res0.val + 2^8 * input_res1.val + 2^16 * input_res2.val
        + 2^24 * input_res3.val + 2^32 * input_res4.val + 2^40 * input_res5.val
        + 2^48 * input_res6.val + 2^56 * input_res7.val
        = Bits.appOf input_op.val
          (input_a0.val + 2^8 * input_a1.val + 2^16 * input_a2.val + 2^24 * input_a3.val
            + 2^32 * input_a4.val + 2^40 * input_a5.val + 2^48 * input_a6.val
            + 2^56 * input_a7.val)
          (input_b0.val + 2^8 * input_b1.val + 2^16 * input_b2.val + 2^24 * input_b3.val
            + 2^32 * input_b4.val + 2^40 * input_b5.val + 2^48 * input_b6.val
            + 2^56 * input_b7.val) := by
      rw [hcomp, e0, e1, e2, e3, e4, e5, e6, e7]
    -- the word values of the operands and the result, read on naturals
    have wa0 := val_word ha0 ha1 ha2 ha3
    have wa1 := val_word ha4 ha5 ha6 ha7
    have wb0 := val_word hb0 hb1 hb2 hb3
    have wb1 := val_word hb4 hb5 hb6 hb7
    have wr0 := val_word hr0 hr1 hr2 hr3
    have wr1 := val_word hr4 hr5 hr6 hr7
    -- the selector is one of the three, so it is below every other opsel on the bus
    have hop3 : input_op.val < 3 := by
      rcases hop with h | h | h <;> rw [h] <;> simp [opselAND, opselOR, opselXOR]
    refine ⟨fun _ => hspec, .inr hcount, .inr hcount, .inr hcount, .inr hcount, .inr hcount,
      .inr hcount, .inr hcount, .inr hcount, ?_⟩
    intro _ _ _ _ _
    -- `op` is 0, 1 or 2, so only the three BYTEWISE clauses can fire
    refine ⟨fun s i hs hi h => by have : opselLT = 4 := rfl; omega,
      fun i hi h => by have : opselEQ = 3 := rfl; omega,
      fun w s d hw hs hd h => by have : opselSHIFT = 5 := rfl; omega,
      fun h => ?_, fun h => ?_, fun h => ?_,
      fun ls rs hg hls hrs hhg h => by have : opselMUL = 7 := rfl; omega⟩
    all_goals
      refine ⟨by rw [wr0]; omega, by rw [wr1]; omega, ?_⟩
      rw [wr0, wr1, wa0, wa1, wb0, wb1]
      simp only [Bits.appOf, h, opselAND, opselOR, opselXOR, reduceIte] at hspec
      ring_nf
      ring_nf at hspec
      exact hspec

  ProverAssumptions r _ _ := Honest r

  completeness := by
    circuit_proof_start [main, areBytes, word, Alu.channel, Alu.Guarantees, Honest,
      Bitwise.areBytesTable, Bitwise.byteAluTable]
    obtain ⟨hmu, ha0, ha1, ha2, ha3, ha4, ha5, ha6, ha7,
      hb0, hb1, hb2, hb3, hb4, hb5, hb6, hb7, hop,
      hr0, hr1, hr2, hr3, hr4, hr5, hr6, hr7⟩ := h_assumptions
    exact ⟨⟨hmu, by simp⟩,
      fun _ => ⟨ha0, hb0, guarantee_of_appOf hop hr0⟩,
      fun _ => ⟨ha1, hb1, guarantee_of_appOf hop hr1⟩,
      fun _ => ⟨ha2, hb2, guarantee_of_appOf hop hr2⟩,
      fun _ => ⟨ha3, hb3, guarantee_of_appOf hop hr3⟩,
      fun _ => ⟨ha4, hb4, guarantee_of_appOf hop hr4⟩,
      fun _ => ⟨ha5, hb5, guarantee_of_appOf hop hr5⟩,
      fun _ => ⟨ha6, hb6, guarantee_of_appOf hop hr6⟩,
      fun _ => ⟨ha7, hb7, guarantee_of_appOf hop hr7⟩⟩

end LambdaVM.BYTEWISE
