module

public import LambdaVM.Field

@[expose] public section

/-!
# BITWISE (spec/bitwise.typ)

Preprocessed table with one row per `(X, Y, Z) ∈ Byte × Byte × B4`, indexed by
`i = X + 256·Y + 65536·Z`, exactly as `generate_bitwise_row` in `prover/src/tables/bitwise.rs`.
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
def index (x y z : ℕ) : ℕ := x + 256 * y + 65536 * z

/-- `generate_bitwise_row`, with `/` and `%` where Rust masks and shifts the index. -/
def row (i : ℕ) : Row :=
  let x := i % 256
  let y := i / 256 % 256
  let z := i / 65536 % 16
  let halfword := x + 256 * y
  { x, y, z
    and := x &&& y
    or := x ||| y
    xor := x ^^^ y
    msb8 := x / 128
    msb16 := halfword / 32768
    zero := if x = 0 ∧ y = 0 ∧ z = 0 then 1 else 0
    sll := if z = 0 then halfword else (halfword <<< z) % 65536
    sllc := if z = 0 then 0 else halfword >>> (16 - z) }

def rows : List Row := (List.range numRows).map row

lemma mem_rows_iff {r : Row} : r ∈ rows ↔ ∃ i < numRows, row i = r := by
  simp [rows, List.mem_map, List.mem_range]

lemma index_lt {x y z : ℕ} (hx : x < 256) (hy : y < 256) (hz : z < 16) :
    index x y z < numRows := by
  unfold index numRows; omega

lemma row_x_lt (i : ℕ) : (row i).x < 256 := Nat.mod_lt _ (by norm_num)
lemma row_y_lt (i : ℕ) : (row i).y < 256 := Nat.mod_lt _ (by norm_num)
lemma row_z_lt (i : ℕ) : (row i).z < 16 := Nat.mod_lt _ (by norm_num)

lemma row_index {x y z : ℕ} (hx : x < 256) (hy : y < 256) (hz : z < 16) :
    row (index x y z) =
      let halfword := x + 256 * y
      { x, y, z
        and := x &&& y
        or := x ||| y
        xor := x ^^^ y
        msb8 := x / 128
        msb16 := halfword / 32768
        zero := if x = 0 ∧ y = 0 ∧ z = 0 then 1 else 0
        sll := if z = 0 then halfword else (halfword <<< z) % 65536
        sllc := if z = 0 then 0 else halfword >>> (16 - z) } := by
  have h1 : index x y z % 256 = x := by unfold index; omega
  have h2 : index x y z / 256 % 256 = y := by unfold index; omega
  have h3 : index x y z / 65536 % 16 = z := by unfold index; omega
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
  Guarantees | (x, y) => x.val < 256 ∧ y.val < 256
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

/-- `IS_HALF[X + 256·Y]` (`BusId::IsHalfword`): the value is a halfword. -/
def isHalfTable : StaticLookupChannel (F p) field where
  name := "IS_HALF"
  table := rows.map fun r => ((r.x + 256 * r.y : ℕ) : F p)
  Guarantees h := h.val < 65536
  guarantees_iff := by
    intro h
    simp only [List.mem_map, mem_rows_iff]
    constructor
    · intro hh
      refine ⟨row (index (h.val % 256) (h.val / 256) 0),
        ⟨_, index_lt (by omega) (by omega) (by norm_num), rfl⟩, ?_⟩
      rw [row_index (by omega) (by omega) (by norm_num)]
      simp only
      rw [show h.val % 256 + 256 * (h.val / 256) = h.val by omega]
      exact ZMod.natCast_zmod_val h
    · rintro ⟨r, ⟨i, _, rfl⟩, rfl⟩
      have := row_x_lt i; have := row_y_lt i
      rw [val_natCast_of_lt_two_pow_20 (by omega)]; omega

abbrev IsHalf : Channel (F p) field := .fromStatic _ _ isHalfTable

/-- `IS_B20[X + 256·Y + 65536·Z]` (`BusId::IsB20`): the value is below `2^20`. -/
def isB20Table : StaticLookupChannel (F p) field where
  name := "IS_B20"
  table := rows.map fun r => ((r.x + 256 * r.y + 65536 * r.z : ℕ) : F p)
  Guarantees v := v.val < 2^20
  guarantees_iff := by
    intro v
    simp only [List.mem_map, mem_rows_iff]
    constructor
    · intro hv
      refine ⟨row (index (v.val % 256) (v.val / 256 % 256) (v.val / 65536)),
        ⟨_, index_lt (by omega) (by omega) (by omega), rfl⟩, ?_⟩
      rw [row_index (by omega) (by omega) (by omega)]
      simp only
      rw [show v.val % 256 + 256 * (v.val / 256 % 256) + 65536 * (v.val / 65536) = v.val by omega]
      exact ZMod.natCast_zmod_val v
    · rintro ⟨r, ⟨i, _, rfl⟩, rfl⟩
      have := row_x_lt i; have := row_y_lt i; have := row_z_lt i
      rw [val_natCast_of_lt_two_pow_20 (by omega)]; omega

abbrev IsB20 : Channel (F p) field := .fromStatic _ _ isB20Table

/-! ## Bit-extraction channels -/

lemma row_msb8 (i : ℕ) : (row i).msb8 = (row i).x / 128 := rfl
lemma row_msb16 (i : ℕ) : (row i).msb16 = ((row i).x + 256 * (row i).y) / 32768 := rfl
lemma row_zero (i : ℕ) :
    (row i).zero = if (row i).x = 0 ∧ (row i).y = 0 ∧ (row i).z = 0 then 1 else 0 := rfl

/-- `MSB8[X] -> MSB8` (`BusId::Msb8`): `X` is a byte and the output is its top bit. -/
def msb8Table : StaticLookupChannel (F p) fieldPair where
  name := "MSB8"
  table := rows.map fun r => ((r.x : F p), (r.msb8 : F p))
  Guarantees | (x, m) => x.val < 256 ∧ m.val = x.val / 128
  guarantees_iff := by
    rintro ⟨x, m⟩
    simp only [List.mem_map, mem_rows_iff, Prod.mk.injEq]
    constructor
    · rintro ⟨hx, hm⟩
      refine ⟨row (index x.val 0 0), ⟨_, index_lt hx (by norm_num) (by norm_num), rfl⟩, ?_, ?_⟩
      · rw [row_index hx (by norm_num) (by norm_num)]; exact ZMod.natCast_zmod_val x
      · rw [row_index hx (by norm_num) (by norm_num)]
        simp only
        rw [← hm]; exact ZMod.natCast_zmod_val m
    · rintro ⟨r, ⟨i, _, rfl⟩, rfl, rfl⟩
      have := row_x_lt i
      rw [val_natCast_of_lt_two_pow_20 (by omega), val_natCast_of_lt_two_pow_20 (by rw [row_msb8]; omega),
        row_msb8]
      exact ⟨this, rfl⟩

abbrev Msb8 : Channel (F p) fieldPair := .fromStatic _ _ msb8Table

/-- `MSB16[X + 256·Y] -> MSB16` (`BusId::Msb16`): a halfword and its top bit. -/
def msb16Table : StaticLookupChannel (F p) fieldPair where
  name := "MSB16"
  table := rows.map fun r => (((r.x + 256 * r.y : ℕ) : F p), (r.msb16 : F p))
  Guarantees | (h, m) => h.val < 65536 ∧ m.val = h.val / 32768
  guarantees_iff := by
    rintro ⟨h, m⟩
    simp only [List.mem_map, mem_rows_iff, Prod.mk.injEq]
    constructor
    · rintro ⟨hh, hm⟩
      refine ⟨row (index (h.val % 256) (h.val / 256) 0),
        ⟨_, index_lt (by omega) (by omega) (by norm_num), rfl⟩, ?_, ?_⟩
      · rw [row_index (by omega) (by omega) (by norm_num)]
        simp only
        rw [show h.val % 256 + 256 * (h.val / 256) = h.val by omega]
        exact ZMod.natCast_zmod_val h
      · rw [row_index (by omega) (by omega) (by norm_num)]
        simp only
        rw [show h.val % 256 + 256 * (h.val / 256) = h.val by omega, ← hm]
        exact ZMod.natCast_zmod_val m
    · rintro ⟨r, ⟨i, _, rfl⟩, rfl, rfl⟩
      have := row_x_lt i; have := row_y_lt i
      rw [val_natCast_of_lt_two_pow_20 (by omega), val_natCast_of_lt_two_pow_20 (by rw [row_msb16]; omega),
        row_msb16]
      exact ⟨by omega, rfl⟩

abbrev Msb16 : Channel (F p) fieldPair := .fromStatic _ _ msb16Table

/-- `ZERO[X + 256·Y + 65536·Z] -> ZERO` (`BusId::Zero`): a 20-bit value and its zero flag. -/
def zeroTable : StaticLookupChannel (F p) fieldPair where
  name := "ZERO"
  table := rows.map fun r => (((r.x + 256 * r.y + 65536 * r.z : ℕ) : F p), (r.zero : F p))
  Guarantees | (v, z) => v.val < 2^20 ∧ z = if v = 0 then 1 else 0
  guarantees_iff := by
    rintro ⟨v, z⟩
    simp only [List.mem_map, mem_rows_iff, Prod.mk.injEq]
    constructor
    · rintro ⟨hv, hz⟩
      refine ⟨row (index (v.val % 256) (v.val / 256 % 256) (v.val / 65536)),
        ⟨_, index_lt (by omega) (by omega) (by omega), rfl⟩, ?_, ?_⟩
      · rw [row_index (by omega) (by omega) (by omega)]
        simp only
        rw [show v.val % 256 + 256 * (v.val / 256 % 256) + 65536 * (v.val / 65536) = v.val by omega]
        exact ZMod.natCast_zmod_val v
      · rw [row_index (by omega) (by omega) (by omega)]
        simp only
        have hiff : (v.val % 256 = 0 ∧ v.val / 256 % 256 = 0 ∧ v.val / 65536 = 0) ↔ v = 0 := by
          rw [← ZMod.val_eq_zero]; omega
        rw [hz]
        simp only [hiff]
        split_ifs <;> simp
    · rintro ⟨r, ⟨i, _, rfl⟩, rfl, rfl⟩
      have := row_x_lt i; have := row_y_lt i; have := row_z_lt i
      rw [val_natCast_of_lt_two_pow_20 (by omega), row_zero]
      refine ⟨by omega, ?_⟩
      have hn : (((row i).x + 256 * (row i).y + 65536 * (row i).z : ℕ) : F p) = 0 ↔
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

lemma and_lt_256 {x y : ℕ} (hy : y < 256) : x &&& y < 256 := Nat.and_le_right.trans_lt hy
lemma or_lt_256 {x y : ℕ} (hx : x < 256) (hy : y < 256) : x ||| y < 256 := by
  have := Nat.or_lt_two_pow (n := 8) (by norm_num; exact hx) (by norm_num; exact hy)
  norm_num at this; exact this
lemma xor_lt_256 {x y : ℕ} (hx : x < 256) (hy : y < 256) : x ^^^ y < 256 := by
  have := Nat.xor_lt_two_pow (n := 8) (by norm_num; exact hx) (by norm_num; exact hy)
  norm_num at this; exact this

def byteAluTable : StaticLookupChannel (F p) ByteAluMsg where
  name := "BYTE_ALU"
  table := rows.flatMap fun r =>
    [ ⟨0, r.x, r.y, r.and⟩, ⟨1, r.x, r.y, r.or⟩, ⟨2, r.x, r.y, r.xor⟩ ]
  Guarantees
  | { opsel, x, y, out } =>
    x.val < 256 ∧ y.val < 256 ∧
      ((opsel = 0 ∧ out.val = x.val &&& y.val) ∨
       (opsel = 1 ∧ out.val = x.val ||| y.val) ∨
       (opsel = 2 ∧ out.val = x.val ^^^ y.val))
  guarantees_iff := by
    rintro ⟨opsel, x, y, out⟩
    simp only [List.mem_flatMap, mem_rows_iff, List.mem_cons,
      List.not_mem_nil, or_false, ByteAluMsg.mk.injEq]
    constructor
    · rintro ⟨hx, hy, hop⟩
      refine ⟨row (index x.val y.val 0), ⟨_, index_lt hx hy (by norm_num), rfl⟩, ?_⟩
      rw [row_index hx hy (by norm_num)]
      simp only
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

/-- Message of `HWSL[X + 256·Y, Z] -> [SLL, SLLC]` (`BusId::Hwsl`). -/
structure HwslMsg (F : Type) where
  h : F
  z : F
  sll : F
  sllc : F
deriving ProvableStruct

lemma row_sll (i : ℕ) : (row i).sll =
    if (row i).z = 0 then (row i).x + 256 * (row i).y
    else (((row i).x + 256 * (row i).y) <<< (row i).z) % 65536 := rfl
lemma row_sllc (i : ℕ) : (row i).sllc =
    if (row i).z = 0 then 0 else ((row i).x + 256 * (row i).y) >>> (16 - (row i).z) := rfl

def hwslTable : StaticLookupChannel (F p) HwslMsg where
  name := "HWSL"
  table := rows.map fun r => ⟨((r.x + 256 * r.y : ℕ) : F p), r.z, r.sll, r.sllc⟩
  Guarantees
  | { h, z, sll, sllc } =>
    h.val < 65536 ∧ z.val < 16 ∧
      sll.val = (if z.val = 0 then h.val else (h.val <<< z.val) % 65536) ∧
      sllc.val = (if z.val = 0 then 0 else h.val >>> (16 - z.val))
  guarantees_iff := by
    rintro ⟨h, z, sll, sllc⟩
    simp only [List.mem_map, mem_rows_iff, HwslMsg.mk.injEq]
    constructor
    · rintro ⟨hh, hz, hsll, hsllc⟩
      refine ⟨row (index (h.val % 256) (h.val / 256) z.val),
        ⟨_, index_lt (by omega) (by omega) hz, rfl⟩, ?_⟩
      rw [row_index (by omega) (by omega) hz]
      simp only
      rw [show h.val % 256 + 256 * (h.val / 256) = h.val by omega]
      refine ⟨ZMod.natCast_zmod_val h, ZMod.natCast_zmod_val z, ?_, ?_⟩
      · rw [← hsll]; exact ZMod.natCast_zmod_val sll
      · rw [← hsllc]; exact ZMod.natCast_zmod_val sllc
    · rintro ⟨r, ⟨i, _, rfl⟩, rfl, rfl, rfl, rfl⟩
      have hx := row_x_lt i; have hy := row_y_lt i; have hz := row_z_lt i
      have hsll_lt : (row i).sll < 65536 := by
        rw [row_sll]; split_ifs
        · omega
        · exact Nat.mod_lt _ (by norm_num)
      have hsllc_lt : (row i).sllc < 65536 := by
        rw [row_sllc]; split_ifs
        · norm_num
        · exact (Nat.shiftRight_le _ _).trans_lt (by omega)
      rw [val_natCast_of_lt_two_pow_20 (by omega), val_natCast_of_lt_two_pow_20 (by omega),
        val_natCast_of_lt_two_pow_20 (by omega), val_natCast_of_lt_two_pow_20 (by omega),
        row_sll, row_sllc]
      exact ⟨by omega, hz, rfl, rfl⟩

abbrev Hwsl : Channel (F p) HwslMsg := .fromStatic _ _ hwslTable

end LambdaVM.Bitwise
