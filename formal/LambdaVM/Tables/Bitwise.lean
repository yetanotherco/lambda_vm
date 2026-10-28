module

public import LambdaVM.Field
public import LambdaVM.AluOp

@[expose] public section

/-!
# BITWISE (spec/bitwise.typ)

Preprocessed table with one row per `(X, Y, Z) ∈ Byte × Byte × B4`, indexed by
`i = X + 2^8·Y + 2^16·Z`, exactly as `generate_bitwise_row` in `prover/src/tables/bitwise.rs`.
Each bus the table serves is a static lookup channel whose rows are a projection of `row i`;
the channel's `Guarantees` is the semantic reading of that projection, and `guarantees_iff`
proves the two agree. Bus names follow `BusId` in `prover/src/tables/types.rs`.
-/

namespace LambdaVM.Bitwise

/-- The eleven precomputed columns of one row, as naturals. -/
structure Row where
  x : ℕ
  y : ℕ
  z : ℕ
  and : ℕ
  or : ℕ
  xor : ℕ
  msb8 : ℕ
  msb16 : ℕ
  zero : ℕ
  sll : ℕ
  sllc : ℕ

def numRows : ℕ := 2^20

/-- Row index of `(x, y, z)`. -/
def index (x y z : ℕ) : ℕ := x + 2^8 * y + 2^16 * z

/-- `SLL`: `h` shifted left by `z`, kept to 16 bits. -/
def sll (h z : ℕ) : ℕ := if z = 0 then h else (h <<< z) % 2^16

/-- `SLLC`: the bits `SLL` drops, `h` shifted right by `16 − z`. -/
def sllc (h z : ℕ) : ℕ := if z = 0 then 0 else h >>> (16 - z)

/-- On a 16-bit `h` the two shifts are the spec's formulas, with no case on `z`. -/
lemma sll_eq {h z : ℕ} (hh : h < 2^16) : sll h z = (h <<< z) % 2^16 := by
  unfold sll; split_ifs with hz
  · rw [hz, Nat.shiftLeft_zero, Nat.mod_eq_of_lt hh]
  · rfl

lemma sllc_eq {h z : ℕ} (hh : h < 2^16) : sllc h z = h >>> (16 - z) := by
  unfold sllc; split_ifs with hz
  · rw [hz, Nat.sub_zero, Nat.shiftRight_eq_div_pow, Nat.div_eq_of_lt hh]
  · rfl

lemma sll_of_eq {x h z : ℕ} (hh : h < 2^16) (hx : x = (h <<< z) % 2^16) : x = sll h z := by
  rw [hx, sll_eq hh]

lemma sllc_of_eq {x h z : ℕ} (hh : h < 2^16) (hx : x = h >>> (16 - z)) : x = sllc h z := by
  rw [hx, sllc_eq hh]

/-- The precomputed columns of `(x, y, z)`: `generate_bitwise_row`. -/
def rowOf (x y z : ℕ) : Row :=
  let halfword := x + 2^8 * y
  { x, y, z
    and := x &&& y
    or := x ||| y
    xor := x ^^^ y
    msb8 := x / 2^7
    msb16 := halfword / 2^15
    zero := if x = 0 ∧ y = 0 ∧ z = 0 then 1 else 0
    sll := sll halfword z
    sllc := sllc halfword z }

/-- Row `i` of the table, with `/` and `%` where Rust masks and shifts the index. -/
def row (i : ℕ) : Row := rowOf (i % 2^8) (i / 2^8 % 2^8) (i / 2^16 % 16)

def rows : List Row := (List.range numRows).map row

lemma mem_rows_iff {r : Row} : r ∈ rows ↔ ∃ i < numRows, row i = r := by
  simp [rows, List.mem_map, List.mem_range]

lemma index_lt {x y z : ℕ} (hx : x < 2^8) (hy : y < 2^8) (hz : z < 16) :
    index x y z < numRows := by
  unfold index numRows; omega

lemma row_x_lt (i : ℕ) : (row i).x < 2^8 := Nat.mod_lt _ (by norm_num)
lemma row_y_lt (i : ℕ) : (row i).y < 2^8 := Nat.mod_lt _ (by norm_num)
lemma row_z_lt (i : ℕ) : (row i).z < 16 := Nat.mod_lt _ (by norm_num)

lemma row_index {x y z : ℕ} (hx : x < 2^8) (hy : y < 2^8) (hz : z < 16) :
    row (index x y z) = rowOf x y z := by
  have h1 : index x y z % 2^8 = x := by unfold index; omega
  have h2 : index x y z / 2^8 % 2^8 = y := by unfold index; omega
  have h3 : index x y z / 2^16 % 16 = z := by unfold index; omega
  simp only [row, h1, h2, h3]

/-- Everything in the table is below `2^20 < p`, so casting to the field is injective. -/
lemma lt_p_of_lt_two_pow_20 {n : ℕ} (h : n < 2^20) : n < p := lt_trans h (by decide)

lemma val_natCast_of_lt_two_pow_20 {n : ℕ} (h : n < 2^20) : ((n : F p)).val = n :=
  ZMod.val_natCast_of_lt (lt_p_of_lt_two_pow_20 h)

/-! ## Range-check channels -/

/-- `ARE_BYTES[X, Y]` (`BusId::AreBytes`): both values are bytes. -/
def areBytesTable : StaticLookupChannel (F p) fieldPair where
  name := "ARE_BYTES"
  table := rows.map fun r => ((r.x : F p), (r.y : F p))
  Guarantees | (x, y) => x.val < 2^8 ∧ y.val < 2^8
  guarantees_iff := by
    rintro ⟨x, y⟩
    simp only [List.mem_map, mem_rows_iff, Prod.mk.injEq]
    constructor
    · rintro ⟨hx, hy⟩
      refine ⟨row (index x.val y.val 0), ⟨_, index_lt hx hy (by norm_num), rfl⟩, ?_, ?_⟩
      · rw [row_index hx hy (by norm_num)]; exact ZMod.natCast_zmod_val x
      · rw [row_index hx hy (by norm_num)]; exact ZMod.natCast_zmod_val y
    · rintro ⟨r, ⟨i, _, rfl⟩, rfl, rfl⟩
      exact ⟨by rw [val_natCast_of_lt_two_pow_20 (by linarith [row_x_lt i])]; exact row_x_lt i,
             by rw [val_natCast_of_lt_two_pow_20 (by linarith [row_y_lt i])]; exact row_y_lt i⟩

abbrev AreBytes : Channel (F p) fieldPair := .fromStatic _ _ areBytesTable

/-- `IS_HALF[X + 2^8·Y]` (`BusId::IsHalfword`): the value is a halfword. -/
def isHalfTable : StaticLookupChannel (F p) field where
  name := "IS_HALF"
  table := rows.map fun r => ((r.x + 2^8 * r.y : ℕ) : F p)
  Guarantees h := h.val < 2^16
  guarantees_iff := by
    intro h
    simp only [List.mem_map, mem_rows_iff]
    constructor
    · intro hh
      refine ⟨row (index (h.val % 2^8) (h.val / 2^8) 0),
        ⟨_, index_lt (by omega) (by omega) (by norm_num), rfl⟩, ?_⟩
      rw [row_index (by omega) (by omega) (by norm_num)]
      simp only [rowOf]
      rw [show h.val % 2^8 + 2^8 * (h.val / 2^8) = h.val by omega]
      exact ZMod.natCast_zmod_val h
    · rintro ⟨r, ⟨i, _, rfl⟩, rfl⟩
      have := row_x_lt i; have := row_y_lt i
      rw [val_natCast_of_lt_two_pow_20 (by omega)]; omega

abbrev IsHalf : Channel (F p) field := .fromStatic _ _ isHalfTable

/-- `IS_B20[X + 2^8·Y + 2^16·Z]` (`BusId::IsB20`): the value is below `2^20`. -/
def isB20Table : StaticLookupChannel (F p) field where
  name := "IS_B20"
  table := rows.map fun r => ((r.x + 2^8 * r.y + 2^16 * r.z : ℕ) : F p)
  Guarantees v := v.val < 2^20
  guarantees_iff := by
    intro v
    simp only [List.mem_map, mem_rows_iff]
    constructor
    · intro hv
      refine ⟨row (index (v.val % 2^8) (v.val / 2^8 % 2^8) (v.val / 2^16)),
        ⟨_, index_lt (by omega) (by omega) (by omega), rfl⟩, ?_⟩
      rw [row_index (by omega) (by omega) (by omega)]
      simp only [rowOf]
      rw [show v.val % 2^8 + 2^8 * (v.val / 2^8 % 2^8) + 2^16 * (v.val / 2^16) = v.val by omega]
      exact ZMod.natCast_zmod_val v
    · rintro ⟨r, ⟨i, _, rfl⟩, rfl⟩
      have := row_x_lt i; have := row_y_lt i; have := row_z_lt i
      rw [val_natCast_of_lt_two_pow_20 (by omega)]; omega

abbrev IsB20 : Channel (F p) field := .fromStatic _ _ isB20Table

/-! ## Bit-extraction channels -/

lemma row_msb8 (i : ℕ) : (row i).msb8 = (row i).x / 2^7 := rfl
lemma row_msb16 (i : ℕ) : (row i).msb16 = ((row i).x + 2^8 * (row i).y) / 2^15 := rfl
lemma row_zero (i : ℕ) :
    (row i).zero = if (row i).x = 0 ∧ (row i).y = 0 ∧ (row i).z = 0 then 1 else 0 := rfl

/-- `MSB8[X] -> MSB8` (`BusId::Msb8`): `X` is a byte and the output is its top bit. -/
def msb8Table : StaticLookupChannel (F p) fieldPair where
  name := "MSB8"
  table := rows.map fun r => ((r.x : F p), (r.msb8 : F p))
  Guarantees | (x, m) => x.val < 2^8 ∧ m.val = x.val / 2^7
  guarantees_iff := by
    rintro ⟨x, m⟩
    simp only [List.mem_map, mem_rows_iff, Prod.mk.injEq]
    constructor
    · rintro ⟨hx, hm⟩
      refine ⟨row (index x.val 0 0), ⟨_, index_lt hx (by norm_num) (by norm_num), rfl⟩, ?_, ?_⟩
      · rw [row_index hx (by norm_num) (by norm_num)]; exact ZMod.natCast_zmod_val x
      · rw [row_index hx (by norm_num) (by norm_num)]
        simp only [rowOf]
        rw [← hm]; exact ZMod.natCast_zmod_val m
    · rintro ⟨r, ⟨i, _, rfl⟩, rfl, rfl⟩
      have := row_x_lt i
      rw [val_natCast_of_lt_two_pow_20 (by omega), val_natCast_of_lt_two_pow_20 (by rw [row_msb8]; omega),
        row_msb8]
      exact ⟨this, rfl⟩

abbrev Msb8 : Channel (F p) fieldPair := .fromStatic _ _ msb8Table

/-- `MSB16[X + 2^8·Y] -> MSB16` (`BusId::Msb16`): a halfword and its top bit. -/
def msb16Table : StaticLookupChannel (F p) fieldPair where
  name := "MSB16"
  table := rows.map fun r => (((r.x + 2^8 * r.y : ℕ) : F p), (r.msb16 : F p))
  Guarantees | (h, m) => h.val < 2^16 ∧ m.val = h.val / 2^15
  guarantees_iff := by
    rintro ⟨h, m⟩
    simp only [List.mem_map, mem_rows_iff, Prod.mk.injEq]
    constructor
    · rintro ⟨hh, hm⟩
      refine ⟨row (index (h.val % 2^8) (h.val / 2^8) 0),
        ⟨_, index_lt (by omega) (by omega) (by norm_num), rfl⟩, ?_, ?_⟩
      · rw [row_index (by omega) (by omega) (by norm_num)]
        simp only [rowOf]
        rw [show h.val % 2^8 + 2^8 * (h.val / 2^8) = h.val by omega]
        exact ZMod.natCast_zmod_val h
      · rw [row_index (by omega) (by omega) (by norm_num)]
        simp only [rowOf]
        rw [show h.val % 2^8 + 2^8 * (h.val / 2^8) = h.val by omega, ← hm]
        exact ZMod.natCast_zmod_val m
    · rintro ⟨r, ⟨i, _, rfl⟩, rfl, rfl⟩
      have := row_x_lt i; have := row_y_lt i
      rw [val_natCast_of_lt_two_pow_20 (by omega), val_natCast_of_lt_two_pow_20 (by rw [row_msb16]; omega),
        row_msb16]
      exact ⟨by omega, rfl⟩

abbrev Msb16 : Channel (F p) fieldPair := .fromStatic _ _ msb16Table

/-- `ZERO[X + 2^8·Y + 2^16·Z] -> ZERO` (`BusId::Zero`): a 20-bit value and its zero flag. -/
def zeroTable : StaticLookupChannel (F p) fieldPair where
  name := "ZERO"
  table := rows.map fun r => (((r.x + 2^8 * r.y + 2^16 * r.z : ℕ) : F p), (r.zero : F p))
  Guarantees | (v, z) => v.val < 2^20 ∧ z = if v = 0 then 1 else 0
  guarantees_iff := by
    rintro ⟨v, z⟩
    simp only [List.mem_map, mem_rows_iff, Prod.mk.injEq]
    constructor
    · rintro ⟨hv, hz⟩
      refine ⟨row (index (v.val % 2^8) (v.val / 2^8 % 2^8) (v.val / 2^16)),
        ⟨_, index_lt (by omega) (by omega) (by omega), rfl⟩, ?_, ?_⟩
      · rw [row_index (by omega) (by omega) (by omega)]
        simp only [rowOf]
        rw [show v.val % 2^8 + 2^8 * (v.val / 2^8 % 2^8) + 2^16 * (v.val / 2^16) = v.val by omega]
        exact ZMod.natCast_zmod_val v
      · rw [row_index (by omega) (by omega) (by omega)]
        simp only [rowOf]
        have hiff : (v.val % 2^8 = 0 ∧ v.val / 2^8 % 2^8 = 0 ∧ v.val / 2^16 = 0) ↔ v = 0 := by
          rw [← ZMod.val_eq_zero]; omega
        rw [hz]
        simp only [hiff]
        split_ifs <;> simp
    · rintro ⟨r, ⟨i, _, rfl⟩, rfl, rfl⟩
      have := row_x_lt i; have := row_y_lt i; have := row_z_lt i
      rw [val_natCast_of_lt_two_pow_20 (by omega), row_zero]
      refine ⟨by omega, ?_⟩
      have hn : (((row i).x + 2^8 * (row i).y + 2^16 * (row i).z : ℕ) : F p) = 0 ↔
          ((row i).x = 0 ∧ (row i).y = 0 ∧ (row i).z = 0) := by
        rw [← ZMod.val_eq_zero, val_natCast_of_lt_two_pow_20 (by omega)]; omega
      simp only [hn]
      split_ifs <;> simp

abbrev Zero : Channel (F p) fieldPair := .fromStatic _ _ zeroTable

/-! ## Byte ALU and halfword shift channels -/

/-- Message of `BYTE_ALU[opsel, X, Y] -> out` (`BusId::ByteAlu`), `opsel` = AND 0 / OR 1 / XOR 2. -/
structure ByteAluMsg (F : Type) where
  opsel : F
  x : F
  y : F
  out : F
deriving ProvableStruct

lemma row_and (i : ℕ) : (row i).and = (row i).x &&& (row i).y := rfl
lemma row_or (i : ℕ) : (row i).or = (row i).x ||| (row i).y := rfl
lemma row_xor (i : ℕ) : (row i).xor = (row i).x ^^^ (row i).y := rfl

lemma and_lt_256 {x y : ℕ} (hy : y < 2^8) : x &&& y < 2^8 := Nat.and_le_right.trans_lt hy
lemma or_lt_256 {x y : ℕ} (hx : x < 2^8) (hy : y < 2^8) : x ||| y < 2^8 := by
  have := Nat.or_lt_two_pow (n := 8) (by norm_num; exact hx) (by norm_num; exact hy)
  norm_num at this; exact this
lemma xor_lt_256 {x y : ℕ} (hx : x < 2^8) (hy : y < 2^8) : x ^^^ y < 2^8 := by
  have := Nat.xor_lt_two_pow (n := 8) (by norm_num; exact hx) (by norm_num; exact hy)
  norm_num at this; exact this

def byteAluTable : StaticLookupChannel (F p) ByteAluMsg where
  name := "BYTE_ALU"
  table := rows.flatMap fun r =>
    [ ⟨opselAND, r.x, r.y, r.and⟩, ⟨opselOR, r.x, r.y, r.or⟩, ⟨opselXOR, r.x, r.y, r.xor⟩ ]
  Guarantees
  | { opsel, x, y, out } =>
    x.val < 2^8 ∧ y.val < 2^8 ∧
      ((opsel = opselAND ∧ out.val = x.val &&& y.val) ∨
       (opsel = opselOR ∧ out.val = x.val ||| y.val) ∨
       (opsel = opselXOR ∧ out.val = x.val ^^^ y.val))
  guarantees_iff := by
    rintro ⟨opsel, x, y, out⟩
    simp only [List.mem_flatMap, mem_rows_iff, List.mem_cons,
      List.not_mem_nil, or_false, ByteAluMsg.mk.injEq]
    constructor
    · rintro ⟨hx, hy, hop⟩
      refine ⟨row (index x.val y.val 0), ⟨_, index_lt hx hy (by norm_num), rfl⟩, ?_⟩
      rw [row_index hx hy (by norm_num)]
      simp only [rowOf]
      rcases hop with ⟨rfl, hout⟩ | ⟨rfl, hout⟩ | ⟨rfl, hout⟩
      · left; refine ⟨by simp, (ZMod.natCast_zmod_val x).symm, (ZMod.natCast_zmod_val y).symm, ?_⟩
        rw [← hout]; exact (ZMod.natCast_zmod_val out).symm
      · right; left; refine ⟨by simp, (ZMod.natCast_zmod_val x).symm, (ZMod.natCast_zmod_val y).symm, ?_⟩
        rw [← hout]; exact (ZMod.natCast_zmod_val out).symm
      · right; right; refine ⟨by simp, (ZMod.natCast_zmod_val x).symm, (ZMod.natCast_zmod_val y).symm, ?_⟩
        rw [← hout]; exact (ZMod.natCast_zmod_val out).symm
    · rintro ⟨r, ⟨i, _, rfl⟩, ⟨rfl, rfl, rfl, rfl⟩ | ⟨rfl, rfl, rfl, rfl⟩ | ⟨rfl, rfl, rfl, rfl⟩⟩
      all_goals
        have hx := row_x_lt i; have hy := row_y_lt i
        rw [val_natCast_of_lt_two_pow_20 (by omega), val_natCast_of_lt_two_pow_20 (by omega)]
        refine ⟨hx, hy, ?_⟩
      · left; rw [row_and, val_natCast_of_lt_two_pow_20 (by linarith [and_lt_256 (x := (row i).x) hy])]
        simp
      · right; left; rw [row_or, val_natCast_of_lt_two_pow_20 (by linarith [or_lt_256 hx hy])]
        simp
      · right; right; rw [row_xor, val_natCast_of_lt_two_pow_20 (by linarith [xor_lt_256 hx hy])]
        simp

abbrev ByteAlu : Channel (F p) ByteAluMsg := .fromStatic _ _ byteAluTable

/-- Message of `HWSL[X + 2^8·Y, Z] -> [SLL, SLLC]` (`BusId::Hwsl`). -/
structure HwslMsg (F : Type) where
  h : F
  z : F
  sll : F
  sllc : F
deriving ProvableStruct

lemma row_sll (i : ℕ) : (row i).sll = sll ((row i).x + 2^8 * (row i).y) (row i).z := rfl
lemma row_sllc (i : ℕ) : (row i).sllc = sllc ((row i).x + 2^8 * (row i).y) (row i).z := rfl

def hwslTable : StaticLookupChannel (F p) HwslMsg where
  name := "HWSL"
  table := rows.map fun r => ⟨((r.x + 2^8 * r.y : ℕ) : F p), r.z, r.sll, r.sllc⟩
  Guarantees
  | { h, z, sll, sllc } =>
    h.val < 2^16 ∧ z.val < 16 ∧
      sll.val = (h.val <<< z.val) % 2^16 ∧ sllc.val = h.val >>> (16 - z.val)
  guarantees_iff := by
    rintro ⟨h, z, sll, sllc⟩
    simp only [List.mem_map, mem_rows_iff, HwslMsg.mk.injEq]
    constructor
    · rintro ⟨hh, hz, hsll, hsllc⟩
      refine ⟨row (index (h.val % 2^8) (h.val / 2^8) z.val),
        ⟨_, index_lt (by omega) (by omega) hz, rfl⟩, ?_⟩
      rw [row_index (by omega) (by omega) hz]
      simp only [rowOf]
      rw [show h.val % 2^8 + 2^8 * (h.val / 2^8) = h.val by omega, sll_eq hh, sllc_eq hh]
      refine ⟨ZMod.natCast_zmod_val h, ZMod.natCast_zmod_val z, ?_, ?_⟩
      · rw [← hsll]; exact ZMod.natCast_zmod_val sll
      · rw [← hsllc]; exact ZMod.natCast_zmod_val sllc
    · rintro ⟨r, ⟨i, _, rfl⟩, rfl, rfl, rfl, rfl⟩
      have hx := row_x_lt i; have hy := row_y_lt i; have hz := row_z_lt i
      have hh : (row i).x + 2^8 * (row i).y < 2^16 := by omega
      have hsll_lt : (row i).sll < 2^16 := by
        rw [row_sll, sll_eq hh]; exact Nat.mod_lt _ (by norm_num)
      have hsllc_lt : (row i).sllc < 2^16 := by
        rw [row_sllc, sllc_eq hh]; exact (Nat.shiftRight_le _ _).trans_lt hh
      rw [val_natCast_of_lt_two_pow_20 (by omega), val_natCast_of_lt_two_pow_20 (by omega),
        val_natCast_of_lt_two_pow_20 (by omega), val_natCast_of_lt_two_pow_20 (by omega),
        row_sll, row_sllc, sll_eq hh, sllc_eq hh]
      exact ⟨hh, hz, rfl, rfl⟩

abbrev Hwsl : Channel (F p) HwslMsg := .fromStatic _ _ hwslTable

end LambdaVM.Bitwise
