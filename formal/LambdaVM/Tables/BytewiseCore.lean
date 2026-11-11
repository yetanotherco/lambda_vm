module

public import LambdaVM.AluOp
public import Mathlib.Tactic.Ring
public import Mathlib.Data.Nat.Bitwise

@[expose] public section

/-!
# The arithmetic of the BYTEWISE chip, on naturals

`BYTE_ALU` gives the operation on each byte. The `ALU` bus carries the operation on the whole
64-bit value. These lemmas are the bridge: an operation that acts on each bit independently
commutes with reading bytes as a little-endian number. None of it is about circuits, which is
why it lives here, the way `ShiftCore`, `MulCore` and `DvrmCore` carry the arithmetic of their
chips.

* `Bitwise` — an operation on naturals that acts on each bit independently.
* `appOf` — the operation `BYTE_ALU`'s `opsel` selects: `AND`, `OR` or `XOR`.
* `ofBytes` — eight bytes read as a 64-bit little-endian number.
* `compose` / `compose_appOf` — the operation on each byte pair is the operation on the
  two 64-bit values (soundness).
* `appOf_lt` — every operation the table selects keeps bytes as bytes.
-/

namespace LambdaVM.BytewiseCore

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

end LambdaVM.BytewiseCore
