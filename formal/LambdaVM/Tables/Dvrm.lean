module

public import Clean.Air.CountedChannel
public import LambdaVM.Buses.Alu
public import LambdaVM.Mu
public import LambdaVM.Tables.Bitwise
public import LambdaVM.Tables.DvrmCore
public import LambdaVM.Tables.MulCore
public import LambdaVM.Templates.IsBit
public import LambdaVM.Templates.IsByte
public import LambdaVM.Templates.Neg
public import LambdaVM.Templates.Sign

@[expose] public section

/-!
# DVRM (`spec/src/dvrm.toml`)

Truncated division of two 64-bit operands, read signed or unsigned by one flag, so the chip
serves `DIV`, `DIVU`, `REM` and `REMU`. A row holds both operands and both results as 16-bit
limbs, the absolute values of `r` and `d` as words, `n - r` as limbs, five sign bits, and the
`div_by_zero` and `overflow` flags.

The chip is the first here that **pulls and pushes the same bus**. It pulls three `ALU` messages
to borrow work from other chips — one `LT` for `|r| < |d|`, two `MUL` for `d * q = n - r` — and
pushes two `DIVREM` messages with its own results. Ordering by whole channels cannot place that;
ordering the `ALU` messages in classes can, which is what `Ensemble.tableSoundness_of_stagedOn`
is for. That is an ensemble concern: the chip's own soundness only receives on its pulls and owes
on its pushes.

Both multiplicities are bytes and every range check fires with their sum, so a DVRM row reaches
`2 * 255`, which is what fixes `LambdaVM.muMax` (see `LambdaVM/Mu.lean`).

Two readings follow the newer spec and `prover/src/tables/dvrm.rs` rather than the TOML vendored
under `spec/`, as MUL already does:

* `μ_q` and `μ_r` are bytes, range-checked. The vendored TOML types them `BaseField` and declares
  no check; `dvrm.rs` explains that without it a `μ_q = 1, μ_r = -1` row receives a quotient
  lookup while sending none of its range checks.
* `div_by_zero` pads to `0`, not `1`. `dvrm.rs` builds the trace from `zeroed_fe_vec`, so a
  padding row has it zero.

One count in `dvrm.rs` is wrong and this file follows the code: its module comment says it sends
`ZERO` five times, but it emits six — two per `NEG` template, plus `overflow` and `div_by_zero`.
-/

namespace LambdaVM.DVRM
open Bitwise (IsHalf Zero)

/-- `IS_HALF` with pulls of multiplicity up to `muMax`. A DVRM row pulls it `μ_q + μ_r` times. -/
abbrev IsHalfC : RawChannel (F p) := Bitwise.IsHalf.counted muMax
/-- `ZERO` and the `ALU` bus, likewise counted: every pull of a DVRM row fires with the sum. -/
abbrev ZeroC : RawChannel (F p) := Bitwise.Zero.counted muMax
abbrev Msb16C : RawChannel (F p) := Bitwise.Msb16.counted muMax
abbrev AluC : RawChannel (F p) := Alu.channelC

/-- Four 16-bit halves: the `DWordHL` of the spec. The row groups its operands this way rather
than holding 34 flat fields, because `deriving ProvableStruct` does not reach that many
components; the flattened order is unchanged, so it is still the column order of `dvrm.rs`. -/
structure Halves4 (F : Type) where
  x0 : F
  x1 : F
  x2 : F
  x3 : F
deriving ProvableStruct

/-- A DVRM row. Flattened, its fields are the column order of `cols` in
`prover/src/tables/dvrm.rs`: `n`, `d`, `signed`, `q`, `r`, `div_by_zero`, `overflow`, `abs_r`,
`abs_d`, `n_sub_r`, the five sign bits and the two multiplicities, 34 columns in all. -/
structure Row (F : Type) where
  n : Halves4 F
  d : Halves4 F
  signed : F
  q : Halves4 F
  r : Halves4 F
  divByZero : F
  overflow : F
  absR : DWordWL F
  absD : DWordWL F
  nSubR : Halves4 F
  signNSubR : F
  signN : F
  signD : F
  signQ : F
  signR : F
  muQ : F
  muR : F
deriving ProvableStruct

/-- Two 16-bit halves as the 32-bit word the cast to `QuadWL` makes. -/
@[circuit_norm]
def word (x0 x1 : Expression (F p)) : Expression (F p) := x0 + 65536 * x1

/-- A sign-extension word: both halves are `65535 * s`. -/
@[circuit_norm]
def fill (s : Expression (F p)) : Expression (F p) := 65535 * s + 65536 * (65535 * s)

def main (r : Var Row (F p)) : Circuit (F p) Unit := do
  -- range: IS_BYTE<1, μ_q> and IS_BYTE<1, μ_r>, batched as `dvrm.rs` emits them
  areBytes r.muQ r.muR
  let mu := r.muQ + r.muR
  IsHalf.pullUpTo muMax mu r.n.x0
  IsHalf.pullUpTo muMax mu r.n.x1
  IsHalf.pullUpTo muMax mu r.n.x2
  IsHalf.pullUpTo muMax mu r.n.x3
  IsHalf.pullUpTo muMax mu r.d.x0
  IsHalf.pullUpTo muMax mu r.d.x1
  IsHalf.pullUpTo muMax mu r.d.x2
  IsHalf.pullUpTo muMax mu r.d.x3
  isBit r.signed
  -- sign_equality: r ≠ 0 → sign_r = sign_n
  assertZero ((r.r.x0 + r.r.x1 + r.r.x2 + r.r.x3) * (r.signR - r.signN))
  -- abs_diff: ALU[abs_r, abs_d, opsel(LT)] -> [not div_by_zero, 0], so |r| < |d| unless d = 0
  Alu.channel.pullUpTo muMax mu
    ⟨r.absR.lo, r.absR.hi, r.absD.lo, r.absD.hi, (opselLT : F p), 1 - r.divByZero, 0⟩
  -- abs_r is −r when r is negative and r otherwise, and the same for d
  neg r.signR r.r.x0 r.r.x1 r.r.x2 r.r.x3 r.absR
  assertZero ((1 - r.signR) * (r.absR.lo - word r.r.x0 r.r.x1))
  assertZero ((1 - r.signR) * (r.absR.hi - word r.r.x2 r.r.x3))
  neg r.signD r.d.x0 r.d.x1 r.d.x2 r.d.x3 r.absD
  assertZero ((1 - r.signD) * (r.absD.lo - word r.d.x0 r.d.x1))
  assertZero ((1 - r.signD) * (r.absD.hi - word r.d.x2 r.d.x3))
  -- overflow: sign_q = signed·(1 − overflow), and overflow is `n = −2^63 ∧ d = −1`
  assertZero (r.signed * (1 - r.overflow) - r.signQ)
  Zero.pullUpTo muMax mu
    (r.n.x0 + r.n.x1 + r.n.x2 + (r.n.x3 - 32768 * r.signN) + (1 - r.signN)
      + (65535 - r.d.x0) + (65535 - r.d.x1) + (65535 - r.d.x2) + (65535 - r.d.x3), r.overflow)
  -- n_sub_r: the carries of extended(n − r) + extended(r) = extended(n), over four words
  IsHalf.pullUpTo muMax mu r.r.x0
  IsHalf.pullUpTo muMax mu r.r.x1
  IsHalf.pullUpTo muMax mu r.r.x2
  IsHalf.pullUpTo muMax mu r.r.x3
  IsHalf.pullUpTo muMax mu r.nSubR.x0
  IsHalf.pullUpTo muMax mu r.nSubR.x1
  IsHalf.pullUpTo muMax mu r.nSubR.x2
  IsHalf.pullUpTo muMax mu r.nSubR.x3
  let en0 := word r.n.x0 r.n.x1
  let en1 := word r.n.x2 r.n.x3
  let enF := fill r.signN
  let er0 := word r.r.x0 r.r.x1
  let er1 := word r.r.x2 r.r.x3
  let erF := fill r.signR
  let es0 := word r.nSubR.x0 r.nSubR.x1
  let es1 := word r.nSubR.x2 r.nSubR.x3
  let esF := fill r.signNSubR
  let c0 := (es0 + er0 - en0) * inv32
  let c1 := (es1 + er1 + c0 - en1) * inv32
  let c2 := (esF + erF + c1 - enF) * inv32
  let c3 := (esF + erF + c2 - enF) * inv32
  isBit c0
  isBit c1
  isBit c2
  isBit c3
  isBit r.signNSubR
  -- equality: ALU[d, q, opsel(MUL) + 32·signed + 64·sign_q] -> n − r, low and high halves
  let mulFlags := (opselMUL : F p) + 32 * r.signed + 64 * r.signQ
  Alu.channel.pullUpTo muMax mu
    ⟨word r.d.x0 r.d.x1, word r.d.x2 r.d.x3, word r.q.x0 r.q.x1, word r.q.x2 r.q.x3, mulFlags, es0, es1⟩
  Alu.channel.pullUpTo muMax mu
    ⟨word r.d.x0 r.d.x1, word r.d.x2 r.d.x3, word r.q.x0 r.q.x1, word r.q.x2 r.q.x3, mulFlags + 128, esF, esF⟩
  IsHalf.pullUpTo muMax mu r.q.x0
  IsHalf.pullUpTo muMax mu r.q.x1
  IsHalf.pullUpTo muMax mu r.q.x2
  IsHalf.pullUpTo muMax mu r.q.x3
  -- defs: the three sign bits
  sign r.signed r.n.x3 r.signN
  sign r.signed r.r.x3 r.signR
  sign r.signed r.d.x3 r.signD
  -- div_by_zero: d = 0 forces every limb of the quotient to 65535, so q = 2^64 − 1
  assertZero (r.divByZero * (r.q.x0 - 65535))
  assertZero (r.divByZero * (r.q.x1 - 65535))
  assertZero (r.divByZero * (r.q.x2 - 65535))
  assertZero (r.divByZero * (r.q.x3 - 65535))
  Zero.pullUpTo muMax mu (r.d.x0 + r.d.x1 + r.d.x2 + r.d.x3, r.divByZero)
  -- output: ALU[n, d, opsel(DIVREM) + 32·signed (+128)] -> q, r
  let dvFlags := (opselDIVREM : F p) + 32 * r.signed
  Alu.channel.emitUpTo muMax r.muQ
    ⟨en0, en1, word r.d.x0 r.d.x1, word r.d.x2 r.d.x3, dvFlags, word r.q.x0 r.q.x1, word r.q.x2 r.q.x3⟩
  Alu.channel.emitUpTo muMax r.muR
    ⟨en0, en1, word r.d.x0 r.d.x1, word r.d.x2 r.d.x3, dvFlags + 128, er0, er1⟩

/-! ## What a row means -/

/-- Four halves as a 64-bit value. -/
def Halves4.value (x : Halves4 (F p)) : ℕ :=
  MulCore.ofHalves4 x.x0.val x.x1.val x.x2.val x.x3.val

@[inline] def Row.nVal (row : Row (F p)) : ℕ := row.n.value
@[inline] def Row.dVal (row : Row (F p)) : ℕ := row.d.value
@[inline] def Row.qVal (row : Row (F p)) : ℕ := row.q.value
@[inline] def Row.rVal (row : Row (F p)) : ℕ := row.r.value

/-- An active row holds the truncated quotient and remainder of its two operands, each read as
signed or unsigned by the one flag. `Alu.divremResult` carries the RISC-V special cases: division
by zero gives quotient `2^64 - 1` and remainder the numerator, and `-2^63 / -1` needs none, since
`Int.tdiv` already wraps to `-2^63` with remainder `0`. The halves of `q` and `r` are 16-bit
limbs, so the result columns are determined, the way `SHIFT.RowSpec` bounds `out0` and `out1`. -/
def RowSpec (row : Row (F p)) : Prop :=
  row.muQ + row.muR ≠ 0 →
    row.q.x0.val < 2^16 ∧ row.q.x1.val < 2^16 ∧ row.q.x2.val < 2^16 ∧ row.q.x3.val < 2^16 ∧
    row.r.x0.val < 2^16 ∧ row.r.x1.val < 2^16 ∧ row.r.x2.val < 2^16 ∧ row.r.x3.val < 2^16 ∧
    row.qVal = Alu.divremResult row.signed.val 0 row.nVal row.dVal ∧
    row.rVal = Alu.divremResult row.signed.val 1 row.nVal row.dVal

/-- The honest prover's row, as `dedup_dvrm_rows` and the trace generator of `dvrm.rs` fill it.
Every column below the operands is a function of `n`, `d` and `signed`, and `DvrmCore.honest_nat`
is the proof that these values satisfy everything the constraints ask.

A padding row is NOT honest: `dvrm.rs` builds the trace from `zeroed_fe_vec`, so its padding
rows are all zero, while an honest row with `d = 0` must carry `div_by_zero = 1` and
`q = 2^64 - 1`. `Padding` below covers them, the way `SHIFT` does. -/
def Honest (row : Row (F p)) : Prop :=
  row.muQ.val < 2^8 ∧ row.muR.val < 2^8 ∧ IsBool row.signed ∧
  row.n.x0.val < 2^16 ∧ row.n.x1.val < 2^16 ∧ row.n.x2.val < 2^16 ∧ row.n.x3.val < 2^16 ∧
  row.d.x0.val < 2^16 ∧ row.d.x1.val < 2^16 ∧ row.d.x2.val < 2^16 ∧ row.d.x3.val < 2^16 ∧
  row.q.x0.val < 2^16 ∧ row.q.x1.val < 2^16 ∧ row.q.x2.val < 2^16 ∧ row.q.x3.val < 2^16 ∧
  row.r.x0.val < 2^16 ∧ row.r.x1.val < 2^16 ∧ row.r.x2.val < 2^16 ∧ row.r.x3.val < 2^16 ∧
  row.nSubR.x0.val < 2^16 ∧ row.nSubR.x1.val < 2^16 ∧
  row.nSubR.x2.val < 2^16 ∧ row.nSubR.x3.val < 2^16 ∧
  row.absR.lo.val < 2^32 ∧ row.absR.hi.val < 2^32 ∧
  row.absD.lo.val < 2^32 ∧ row.absD.hi.val < 2^32 ∧
  -- the results are the honest ones
  row.qVal = Alu.divremResult row.signed.val 0 row.nVal row.dVal ∧
  row.rVal = Alu.divremResult row.signed.val 1 row.nVal row.dVal ∧
  -- n - r, as a 64-bit residue, and its sign
  row.nSubR.value
    = ((Alu.sgn64 row.signed.val row.nVal - Alu.sgn64 row.signed.val row.rVal) % 2^64).toNat ∧
  row.signNSubR.val
    = (if Alu.sgn64 row.signed.val row.nVal - Alu.sgn64 row.signed.val row.rVal < 0
        then 1 else 0) ∧
  -- the flags
  row.overflow.val
    = (if row.signed.val = 1 ∧ row.nVal = 2^63 ∧ row.dVal = 2^64 - 1 then 1 else 0) ∧
  row.divByZero.val = (if row.dVal = 0 then 1 else 0) ∧
  row.signQ.val = row.signed.val * (1 - row.overflow.val) ∧
  -- the three SIGN outputs
  row.signN.val = (if row.signed.val = 1 ∧ 2^63 ≤ row.nVal then 1 else 0) ∧
  row.signD.val = (if row.signed.val = 1 ∧ 2^63 ≤ row.dVal then 1 else 0) ∧
  row.signR.val = (if row.signed.val = 1 ∧ 2^63 ≤ row.rVal then 1 else 0) ∧
  -- the two absolute values
  row.absR.lo.val + 2^32 * row.absR.hi.val = (Alu.sgn64 row.signed.val row.rVal).natAbs ∧
  row.absD.lo.val + 2^32 * row.absD.hi.val = (Alu.sgn64 row.signed.val row.dVal).natAbs

/-- A padding row: every column zero, which is what `zeroed_fe_vec` leaves in `dvrm.rs`. Both
multiplicities are zero, so every counted pull is inactive and every push is suppressed. -/
def Padding (row : Row (F p)) : Prop :=
  row.n.x0 = 0 ∧ row.n.x1 = 0 ∧ row.n.x2 = 0 ∧ row.n.x3 = 0 ∧
  row.d.x0 = 0 ∧ row.d.x1 = 0 ∧ row.d.x2 = 0 ∧ row.d.x3 = 0 ∧ row.signed = 0 ∧
  row.q.x0 = 0 ∧ row.q.x1 = 0 ∧ row.q.x2 = 0 ∧ row.q.x3 = 0 ∧
  row.r.x0 = 0 ∧ row.r.x1 = 0 ∧ row.r.x2 = 0 ∧ row.r.x3 = 0 ∧
  row.divByZero = 0 ∧ row.overflow = 0 ∧
  row.absR.lo = 0 ∧ row.absR.hi = 0 ∧ row.absD.lo = 0 ∧ row.absD.hi = 0 ∧
  row.nSubR.x0 = 0 ∧ row.nSubR.x1 = 0 ∧ row.nSubR.x2 = 0 ∧ row.nSubR.x3 = 0 ∧
  row.signNSubR = 0 ∧ row.signN = 0 ∧ row.signD = 0 ∧ row.signQ = 0 ∧ row.signR = 0 ∧
  row.muQ = 0 ∧ row.muR = 0

-- A 35-interaction circuit: without the precomputed instance every proof below elaborates with
-- metavariables and times out. See "Lessons for big chips" in formal/README.md.
@[reducible]
instance elaborated : ElaboratedCircuit (F p) Row unit main := by elaborate_circuit

/-! ## Reading a row back as truncated division

Every self-contained step below is a standalone lemma over opaque variables, so that each is
kernel-checked once instead of being inlined into the circuit proof's term; see
`doc/performance-problems.md`. For the same reason `soundness` leaves `Alu.Guarantees` folded. -/

/-- `a = k·2^32 − W` in the field, with `k` a bit and `W ≤ 2^32` zero exactly when `k` is: then
`a` holds the natural difference, which is a 32-bit word. Both words of a `NEG` output take this
shape once the two carries of `x + neg = 0` are known. -/
lemma word_sub_val {a : F p} {W k : ℕ} (hW : W ≤ 2^32) (hk : k < 2)
    (h0 : k = 0 → W = 0) (h1 : k = 1 → 1 ≤ W) (h : a = (k : F p) * 2^32 - (W : F p)) :
    a.val = k * 2^32 - W ∧ a.val < 2^32 := by
  have hb : W ≤ k * 2^32 ∧ k * 2^32 - W < 2^32 := by
    rcases (by omega : k = 0 ∨ k = 1) with rfl | rfl
    · have := h0 rfl; omega
    · have := h1 rfl; omega
  have hcast : a = ((k * 2^32 - W : ℕ) : F p) := by
    rw [Nat.cast_sub hb.1, h]; push_cast; ring
  have hv := val_eq_of_eq_natCast hcast (by omega)
  exact ⟨hv, by omega⟩

/-- Four 16-bit halves make a 64-bit value. -/
lemma ofHalves4_lt {a b c d : ℕ} (ha : a < 2^16) (hb : b < 2^16) (hc : c < 2^16) (hd : d < 2^16) :
    MulCore.ofHalves4 a b c d < 2^64 := by
  simp only [MulCore.ofHalves4]
  omega

/-- The `SIGN` output of a 64-bit value held as four halves: zero when the value is read
unsigned, and the top bit of the top half otherwise, which is the sign bit `Alu.sgn64` reads. -/
lemma sign_val {s n h0 h1 h2 h3 : ℕ} (hb0 : h0 < 2^16) (hb1 : h1 < 2^16)
    (hb2 : h2 < 2^16) (hb3 : h3 < 2^16)
    (h : (s = 0 ∧ n = 0) ∨ (s = 1 ∧ n = h3 / 2^15)) :
    n = if s = 1 ∧ 2^63 ≤ MulCore.ofHalves4 h0 h1 h2 h3 then 1 else 0 := by
  have hx : (2^63 ≤ MulCore.ofHalves4 h0 h1 h2 h3) ↔ 2^15 ≤ h3 := by
    simp only [MulCore.ofHalves4]; omega
  simp only [hx]
  rcases h with ⟨rfl, rfl⟩ | ⟨rfl, rfl⟩
  · rw [ite_eq_right (by simp)]
  · by_cases hc : 2^15 ≤ h3
    · rw [ite_eq_left ⟨rfl, hc⟩]; omega
    · rw [ite_eq_right (fun hn => hc hn.2)]; omega

/-- **The `NEG` template of a row, read on naturals.** From the two `ZERO` lookups of `neg` and
the two `(1 − sign)` constraints, `abs` is a pair of 32-bit words: the four halves when the sign
bit is clear, and their two's complement when it is set. -/
lemma abs_val {x0 x1 x2 x3 alo ahi s : F p}
    (hx0 : x0.val < 2^16) (hx1 : x1.val < 2^16) (hx2 : x2.val < 2^16) (hx3 : x3.val < 2^16)
    (hs : IsBool s)
    (hz0 : s = 1 → 1 - (x0 + 65536 * x1 + alo - 0) * inv32 = if x0 + x1 = 0 then 1 else 0)
    (hz1 : s = 1 → 1 - (x2 + 65536 * x3 + ahi + (x0 + 65536 * x1 + alo - 0) * inv32 - 0) * inv32
      = if x0 + x1 + x2 + x3 = 0 then 1 else 0)
    (he0 : (1 - s) * (alo - (x0 + 65536 * x1)) = 0)
    (he1 : (1 - s) * (ahi - (x2 + 65536 * x3)) = 0) :
    alo.val < 2^32 ∧ ahi.val < 2^32 ∧
      dword alo ahi
        = if s.val = 1 then (2^64 - MulCore.ofHalves4 x0.val x1.val x2.val x3.val) % 2^64
          else MulCore.ofHalves4 x0.val x1.val x2.val x3.val := by
  have hv2 : (x0 + x1).val = x0.val + x1.val :=
    val_eq_of_eq_natCast (by push_cast; simp only [ZMod.natCast_zmod_val]) (by omega)
  have hv4 : (x0 + x1 + x2 + x3).val = x0.val + x1.val + x2.val + x3.val :=
    val_eq_of_eq_natCast (by push_cast; simp only [ZMod.natCast_zmod_val]) (by omega)
  have hi2 : (x0 + x1 = 0) ↔ x0.val + x1.val = 0 := by rw [← ZMod.val_eq_zero, hv2]
  have hi4 : (x0 + x1 + x2 + x3 = 0) ↔ x0.val + x1.val + x2.val + x3.val = 0 := by
    rw [← ZMod.val_eq_zero, hv4]
  rcases hs with rfl | rfl
  · -- read unsigned: the two constraints copy the halves into `abs`
    simp only [sub_zero, one_mul] at he0 he1
    have w0 : alo.val = x0.val + 65536 * x1.val := by
      rw [sub_eq_zero.mp he0]
      exact val_eq_of_eq_natCast (by push_cast; simp only [ZMod.natCast_zmod_val]) (by omega)
    have w1 : ahi.val = x2.val + 65536 * x3.val := by
      rw [sub_eq_zero.mp he1]
      exact val_eq_of_eq_natCast (by push_cast; simp only [ZMod.natCast_zmod_val]) (by omega)
    rw [ZMod.val_zero, ite_eq_right (by norm_num : ¬ (0:ℕ) = 1)]
    simp only [dword, MulCore.ofHalves4]
    omega
  · -- read negative: `NEG`'s two carries make `abs` the two's complement
    have key : ∀ E c : F p, c = E * inv32 → E = c * 2^32 := by
      intro E c h; rw [h, mul_assoc, inv32_mul_two_pow, mul_one]
    have hC0 : (x0 + 65536 * x1 + alo - 0) * inv32 = if x0 + x1 = 0 then 0 else 1 := by
      have h := hz0 rfl
      split_ifs at h ⊢ <;> linear_combination -h
    have hC1 : (x2 + 65536 * x3 + ahi + (x0 + 65536 * x1 + alo - 0) * inv32 - 0) * inv32
        = if x0 + x1 + x2 + x3 = 0 then 0 else 1 := by
      have h := hz1 rfl
      split_ifs at h ⊢ <;> linear_combination -h
    obtain ⟨k0, hk0b, hk0, hk0c, hc0⟩ : ∃ k : ℕ, k < 2 ∧ (k = 0 ↔ x0.val + 65536 * x1.val = 0)
        ∧ (if x0 + x1 = 0 then (0 : F p) else 1) = (k : F p)
        ∧ alo = (k : F p) * 2^32 - ((x0.val + 65536 * x1.val : ℕ) : F p) := by
      have hE := key _ _ hC0.symm
      rw [show x0 + 65536 * x1 = ((x0.val + 65536 * x1.val : ℕ) : F p) by
        push_cast; simp only [ZMod.natCast_zmod_val]] at hE
      by_cases hb : x0 + x1 = 0
      · have hw := hi2.mp hb
        rw [ite_eq_left hb] at hE ⊢
        push_cast at hE
        exact ⟨0, by norm_num, by constructor <;> intro <;> omega, by norm_num,
          by push_cast; linear_combination hE⟩
      · have hw : ¬ (x0.val + x1.val = 0) := fun h' => hb (hi2.mpr h')
        rw [ite_eq_right hb] at hE ⊢
        push_cast at hE
        exact ⟨1, by norm_num, by constructor <;> intro <;> omega, by norm_num,
          by push_cast; linear_combination hE⟩
    obtain ⟨k1, hk1b, hk1, hc1⟩ : ∃ k : ℕ, k < 2
        ∧ (k = 0 ↔ x2.val + 65536 * x3.val + k0 = 0)
        ∧ ahi = (k : F p) * 2^32 - ((x2.val + 65536 * x3.val + k0 : ℕ) : F p) := by
      have hE := key _ _ hC1.symm
      rw [hC0, hk0c] at hE
      rw [show ((x2.val + 65536 * x3.val + k0 : ℕ) : F p) = x2 + 65536 * x3 + (k0 : F p) by
        push_cast; simp only [ZMod.natCast_zmod_val]]
      by_cases hb : x0 + x1 + x2 + x3 = 0
      · have hw := hi4.mp hb
        rw [ite_eq_left hb] at hE
        exact ⟨0, by norm_num, by constructor <;> intro <;> omega, by
          push_cast; linear_combination hE⟩
      · have hw : ¬ (x0.val + x1.val + x2.val + x3.val = 0) := fun h' => hb (hi4.mpr h')
        rw [ite_eq_right hb] at hE
        exact ⟨1, by norm_num, by constructor <;> intro <;> omega, by
          push_cast; linear_combination hE⟩
    obtain ⟨vlo, blo⟩ := word_sub_val (W := x0.val + 65536 * x1.val) (k := k0)
      (by omega) hk0b (by omega) (by omega) hc0
    obtain ⟨vhi, bhi⟩ := word_sub_val (W := x2.val + 65536 * x3.val + k0) (k := k1)
      (by omega) hk1b (by omega) (by omega) hc1
    refine ⟨blo, bhi, ?_⟩
    rw [val_one_eq, ite_eq_left rfl]
    simp only [dword, MulCore.ofHalves4]
    omega

/-- **The chip's four carry equations, assembled.** The two 32-bit limb additions and the two
sign-extension ones give `(n − r) + r = n` on the 128-bit sign-extended values, the top carry
being what leaves the 128th bit. -/
lemma add128_of_limbs {s0 s1 s2 s3 r0 r1 r2 r3 n0 n1 n2 n3 sm sr sn k0 k1 k2 k3 : ℕ}
    (_hk1 : k1 < 2) (_hk2 : k2 < 2) (_hk3 : k3 < 2)
    (e0 : s0 + 65536 * s1 + (r0 + 65536 * r1) = n0 + 65536 * n1 + k0 * 2^32)
    (e1 : s2 + 65536 * s3 + (r2 + 65536 * r3) + k0 = n2 + 65536 * n3 + k1 * 2^32)
    (e2 : (2^32 - 1) * sm + (2^32 - 1) * sr + k1 = (2^32 - 1) * sn + k2 * 2^32)
    (e3 : (2^32 - 1) * sm + (2^32 - 1) * sr + k2 = (2^32 - 1) * sn + k3 * 2^32) :
    MulCore.ofHalves4 s0 s1 s2 s3 + sm * (2^128 - 2^64)
        + (MulCore.ofHalves4 r0 r1 r2 r3 + sr * (2^128 - 2^64))
      = MulCore.ofHalves4 n0 n1 n2 n3 + sn * (2^128 - 2^64) + k3 * 2^128 := by
  simp only [MulCore.ofHalves4]
  omega

/-- **The two `MUL` messages, assembled.** The low half carries `n − r` and the high half is the
sign-extension word `(2^32 − 1)·s` twice, so together they are the 128-bit product. -/
lemma prod128_of_halves {lo0 lo1 h : F p} {s M : ℕ} (hs : h.val = (2^32 - 1) * s)
    (hlo : dword lo0 lo1 = M % 2^64) (hhi : dword h h = M / 2^64) :
    dword lo0 lo1 + 2^64 * ((2^64 - 1) * s) = M := by
  simp only [dword] at hlo hhi ⊢
  omega

/-! ### The `ALU` bus, one lemma per clause the chip uses -/

/-- The `LT` clause at `opsel(LT)` with both flag bits clear: the output is the comparison bit
of the two operands, read unsigned. -/
lemma lt_of_guarantees {a0 a1 b0 b1 o : F p} (ha0 : a0.val < 2^32) (ha1 : a1.val < 2^32)
    (hb0 : b0.val < 2^32) (hb1 : b1.val < 2^32)
    (h : Alu.Guarantees (⟨a0, a1, b0, b1, 4, o, 0⟩ : Alu.Msg (F p))) :
    o = ((if dword a0 a1 < dword b0 b1 then 1 else 0 : ℕ) : F p) := by
  obtain ⟨-, ho⟩ := h.lt ha0 ha1 hb0 hb1 (s := 0) (i := 0) (by norm_num) (by norm_num)
    (by rw [val_ofNat 4 (by norm_num)]; simp [opselLT])
  dsimp only at ho
  have hL : dword a0 a1 < 2^64 := by simp only [dword]; omega
  have hR : dword b0 b1 < 2^64 := by simp only [dword]; omega
  have hu := Alu.ult_ofNat_iff hL hR
  have hbit : Alu.ltBit 0 0 (dword a0 a1) (dword b0 b1)
      = if dword a0 a1 < dword b0 b1 then 1 else 0 := by
    unfold Alu.ltBit Alu.ltRaw
    by_cases hc : dword a0 a1 < dword b0 b1
    · rw [ite_eq_right (by norm_num : ¬ (0:ℕ) = 1), ite_eq_left (hu.mpr hc), ite_eq_left hc]
      norm_num
    · rw [ite_eq_right (by norm_num : ¬ (0:ℕ) = 1), ite_eq_right (fun hx => hc (hu.mp hx)),
        ite_eq_right hc]
      norm_num
  rw [ho, hbit]

/-- The `MUL` clause at `opsel(MUL) + 32·ls + 64·rs + 128·high`. -/
lemma mul_of_guarantees {a0 a1 b0 b1 o0 o1 f : F p} {ls rs hi : ℕ}
    (ha0 : a0.val < 2^32) (ha1 : a1.val < 2^32) (hb0 : b0.val < 2^32) (hb1 : b1.val < 2^32)
    (hls : ls < 2) (hrs : rs < 2) (hhi : hi < 2) (hf : f.val = 7 + 32 * ls + 64 * rs + 128 * hi)
    (h : Alu.Guarantees (⟨a0, a1, b0, b1, f, o0, o1⟩ : Alu.Msg (F p))) :
    dword o0 o1 = Alu.mulResult ls rs hi (dword a0 a1) (dword b0 b1) :=
  (h.mul ha0 ha1 hb0 hb1 hls hrs hhi (by rw [hf]; simp only [opselMUL])).2.2

/-- What a `DIVREM` pusher owes: at `opsel(DIVREM) + 32·signed + 128·rem` no other clause of the
bus can fire, and that one says the output is `Alu.divremResult` of the operands. -/
lemma divrem_guarantees {a0 a1 b0 b1 o0 o1 f : F p} {s rm : ℕ} (hs : s < 2) (hrm : rm < 2)
    (hf : f.val = 8 + 32 * s + 128 * rm) (ho0 : o0.val < 2^32) (ho1 : o1.val < 2^32)
    (hout : dword o0 o1 = Alu.divremResult s rm (dword a0 a1) (dword b0 b1)) :
    Alu.Guarantees (⟨a0, a1, b0, b1, f, o0, o1⟩ : Alu.Msg (F p)) :=
  Alu.Guarantees.of_divrem hs hrm (by rw [hf]; simp only [opselDIVREM]) fun _ _ _ _ =>
    ⟨ho0, ho1, hout⟩

theorem soundness :
    GeneralFormalCircuit.Soundness (F p) (Input := Row) (Output := unit) main (fun _ _ => True)
      (fun r _ _ => RowSpec r) := by
  circuit_proof_start [main, areBytes, isBit, sign, neg, word, fill, RowSpec,
    Row.nVal, Row.dVal, Row.qVal, Row.rVal, Halves4.value, Alu.channel,
    Bitwise.areBytesTable, Bitwise.isHalfTable, Bitwise.msb16Table, Bitwise.zeroTable,
    Neg.toWL, Neg.zeroWL, Add.carry0, Add.carry1]
  obtain ⟨⟨hmuQ, hmuR⟩, h_n0, h_n1, h_n2, h_n3, h_d0, h_d1, h_d2, h_d3,
    h_sgb, h_sgeq, h_ltp, h_nr0, h_nr1, h_ar0, h_ar1, h_nd0, h_nd1, h_ad0, h_ad1,
    h_sq, h_ovf, h_r0, h_r1, h_r2, h_r3, h_s0, h_s1, h_s2, h_s3,
    hcb0, hcb1, hcb2, hcb3, h_snsr, h_mulA, h_mulB, h_q0, h_q1, h_q2, h_q3,
    h_sgnN, h_sgnN0, h_sgnR, h_sgnR0, h_sgnD, h_sgnD0,
    hqz0, hqz1, hqz2, hqz3, h_dbz⟩ := h_holds
  clear h_input h_assumptions
  -- the flags a row pins down whether or not it is active
  have hsignedb : IsBool input_signed := mul_one_sub_eq_zero_iff.mp h_sgb
  have hsigned2 := IsBool.val_lt_two hsignedb
  have hsignRb : IsBool input_signR := by
    rcases hsignedb with h | h
    · rw [h] at h_sgnR0; simp only [sub_zero, one_mul] at h_sgnR0; exact Or.inl h_sgnR0
    · obtain ⟨hb, hv⟩ := h_sgnR (countUpTo_of_eq_one h)
      exact isBool_of_val_lt_two (by omega)
  have hsignDb : IsBool input_signD := by
    rcases hsignedb with h | h
    · rw [h] at h_sgnD0; simp only [sub_zero, one_mul] at h_sgnD0; exact Or.inl h_sgnD0
    · obtain ⟨hb, hv⟩ := h_sgnD (countUpTo_of_eq_one h)
      exact isBool_of_val_lt_two (by omega)
  have hsignNb : IsBool input_signN := by
    rcases hsignedb with h | h
    · rw [h] at h_sgnN0; simp only [sub_zero, one_mul] at h_sgnN0; exact Or.inl h_sgnN0
    · obtain ⟨hb, hv⟩ := h_sgnN (countUpTo_of_eq_one h)
      exact isBool_of_val_lt_two (by omega)
  have hsnsrb : IsBool input_signNSubR := mul_one_sub_eq_zero_iff.mp h_snsr
  -- `ARE_BYTES[μ_q, μ_r]` makes the sum of the two multiplicities a small natural
  have hsumval : (input_muQ + input_muR).val = input_muQ.val + input_muR.val :=
    val_eq_of_eq_natCast (by push_cast; simp only [ZMod.natCast_zmod_val]) (by omega)
  by_cases hmu : input_muQ + input_muR = 0
  · -- an inactive row owes nothing: no range check fires and neither result is pushed
    have hz : input_muQ.val + input_muR.val = 0 := by rw [← hsumval, hmu, ZMod.val_zero]
    have hzq : input_muQ = 0 := by rw [← ZMod.val_eq_zero]; omega
    have hzr : input_muR = 0 := by rw [← ZMod.val_eq_zero]; omega
    constructorm* _ ∧ _
    all_goals first
      | exact fun h => absurd hmu h
      | exact Or.inl hmu
      | exact eq_zero_or_countUpTo hsignRb
      | exact eq_zero_or_countUpTo hsignDb
      | exact eq_zero_or_countUpTo hsignedb
      | exact fun h => absurd hzq h
      | exact fun h => absurd hzr h
  -- an active row: the sum is a count of at most `muMax`, so every range check delivers
  have hcount : Channel.CountUpTo muMax (input_muQ + input_muR) :=
    Channel.countUpTo_of_val hmu (by simp only [hsumval, muMax]; omega)
  have hn0 := h_n0 hcount
  have hn1 := h_n1 hcount
  have hn2 := h_n2 hcount
  have hn3 := h_n3 hcount
  have hd0 := h_d0 hcount
  have hd1 := h_d1 hcount
  have hd2 := h_d2 hcount
  have hd3 := h_d3 hcount
  have hr0 := h_r0 hcount
  have hr1 := h_r1 hcount
  have hr2 := h_r2 hcount
  have hr3 := h_r3 hcount
  have hs0 := h_s0 hcount
  have hs1 := h_s1 hcount
  have hs2 := h_s2 hcount
  have hs3 := h_s3 hcount
  have hq0 := h_q0 hcount
  have hq1 := h_q1 hcount
  have hq2 := h_q2 hcount
  have hq3 := h_q3 hcount
  clear h_n0 h_n1 h_n2 h_n3 h_d0 h_d1 h_d2 h_d3 h_r0 h_r1 h_r2 h_r3
  clear h_s0 h_s1 h_s2 h_s3 h_q0 h_q1 h_q2 h_q3
  -- the 32-bit words the `ALU` bus carries, read on naturals
  have hwv : ∀ a b : F p, a.val < 2^16 → b.val < 2^16 →
      (a + 65536 * b).val = a.val + 65536 * b.val := fun a b ha hb =>
    val_eq_of_eq_natCast (by push_cast; simp only [ZMod.natCast_zmod_val]) (by omega)
  have hadd2 : ∀ a b : F p, a.val < 2^33 → b.val < 2^33 → (a + b).val = a.val + b.val :=
    fun a b ha hb =>
      val_eq_of_eq_natCast (by push_cast; simp only [ZMod.natCast_zmod_val]) (by omega)
  have hfillv : ∀ b : F p, IsBool b →
      (65535 * b + 65536 * (65535 * b)).val = (2^32 - 1) * b.val := by
    intro b hb
    rcases hb with rfl | rfl
    · simp
    · rw [show (65535 * (1 : F p) + 65536 * (65535 * 1)) = ((4294967295 : ℕ) : F p) by norm_num,
        val_natCast_of_lt_two_pow_40 (by norm_num), val_one_eq]
      norm_num
  have hw_n0 := hwv input_n_x0 input_n_x1 hn0 hn1
  have hw_n1 := hwv input_n_x2 input_n_x3 hn2 hn3
  have hw_d0 := hwv input_d_x0 input_d_x1 hd0 hd1
  have hw_d1 := hwv input_d_x2 input_d_x3 hd2 hd3
  have hw_r0 := hwv input_r_x0 input_r_x1 hr0 hr1
  have hw_r1 := hwv input_r_x2 input_r_x3 hr2 hr3
  have hw_q0 := hwv input_q_x0 input_q_x1 hq0 hq1
  have hw_q1 := hwv input_q_x2 input_q_x3 hq2 hq3
  have hw_s0 := hwv input_nSubR_x0 input_nSubR_x1 hs0 hs1
  have hw_s1 := hwv input_nSubR_x2 input_nSubR_x3 hs2 hs3
  clear hwv
  have hNw : dword (input_n_x0 + 65536 * input_n_x1) (input_n_x2 + 65536 * input_n_x3)
      = MulCore.ofHalves4 input_n_x0.val input_n_x1.val input_n_x2.val input_n_x3.val := by
    simp only [dword, MulCore.ofHalves4, hw_n0, hw_n1]; ring
  have hDw : dword (input_d_x0 + 65536 * input_d_x1) (input_d_x2 + 65536 * input_d_x3)
      = MulCore.ofHalves4 input_d_x0.val input_d_x1.val input_d_x2.val input_d_x3.val := by
    simp only [dword, MulCore.ofHalves4, hw_d0, hw_d1]; ring
  have hQw : dword (input_q_x0 + 65536 * input_q_x1) (input_q_x2 + 65536 * input_q_x3)
      = MulCore.ofHalves4 input_q_x0.val input_q_x1.val input_q_x2.val input_q_x3.val := by
    simp only [dword, MulCore.ofHalves4, hw_q0, hw_q1]; ring
  have hRw : dword (input_r_x0 + 65536 * input_r_x1) (input_r_x2 + 65536 * input_r_x3)
      = MulCore.ofHalves4 input_r_x0.val input_r_x1.val input_r_x2.val input_r_x3.val := by
    simp only [dword, MulCore.ofHalves4, hw_r0, hw_r1]; ring
  have hSw : dword (input_nSubR_x0 + 65536 * input_nSubR_x1)
        (input_nSubR_x2 + 65536 * input_nSubR_x3)
      = MulCore.ofHalves4 input_nSubR_x0.val input_nSubR_x1.val input_nSubR_x2.val
        input_nSubR_x3.val := by
    simp only [dword, MulCore.ofHalves4, hw_s0, hw_s1]; ring
  -- `div_by_zero` is the zero flag of the divisor
  obtain ⟨-, hdbze⟩ := h_dbz hcount
  have hdsum : (input_d_x0 + input_d_x1 + input_d_x2 + input_d_x3).val
      = input_d_x0.val + input_d_x1.val + input_d_x2.val + input_d_x3.val :=
    val_eq_of_eq_natCast (by push_cast; simp only [ZMod.natCast_zmod_val]) (by omega)
  have hdiff : (input_d_x0 + input_d_x1 + input_d_x2 + input_d_x3 = 0)
      ↔ MulCore.ofHalves4 input_d_x0.val input_d_x1.val input_d_x2.val input_d_x3.val = 0 := by
    rw [← ZMod.val_eq_zero, hdsum]; simp only [MulCore.ofHalves4]
    constructor <;> intro <;> omega
  have hdbzval : input_divByZero.val
      = if MulCore.ofHalves4 input_d_x0.val input_d_x1.val input_d_x2.val input_d_x3.val = 0
        then 1 else 0 := by
    by_cases hc : input_d_x0 + input_d_x1 + input_d_x2 + input_d_x3 = 0
    · rw [hdbze, ite_eq_left hc, ite_eq_left (hdiff.mp hc), val_one_eq]
    · rw [hdbze, ite_eq_right hc, ite_eq_right (fun h => hc (hdiff.mpr h)), ZMod.val_zero]
  clear h_dbz hdbze hdsum hdiff
  -- `overflow` is a bit, so `sign_q = signed · (1 − overflow)` is one too
  obtain ⟨-, hovfe⟩ := h_ovf hcount
  have hsignQb : IsBool input_signQ := by
    have hovfb : IsBool input_overflow := by
      rw [hovfe]; split_ifs; exacts [Or.inr rfl, Or.inl rfl]
    have hsqe : input_signQ = input_signed * (1 - input_overflow) := by linear_combination -h_sq
    rcases hsignedb with h | h <;> rcases hovfb with h' | h' <;> rw [hsqe, h, h'] <;> simp [IsBool]
  have hsignQ2 := IsBool.val_lt_two hsignQb
  clear h_ovf hovfe h_sq
  -- the three `SIGN` outputs are the sign bits `Alu.sgn64` reads
  have hsignNval : input_signN.val
      = if input_signed.val = 1 ∧
          2^63 ≤ MulCore.ofHalves4 input_n_x0.val input_n_x1.val input_n_x2.val input_n_x3.val
        then 1 else 0 := by
    refine sign_val hn0 hn1 hn2 hn3 ?_
    rcases hsignedb with h | h
    · rw [h] at h_sgnN0; simp only [sub_zero, one_mul] at h_sgnN0
      exact Or.inl ⟨by rw [h, ZMod.val_zero], by rw [h_sgnN0, ZMod.val_zero]⟩
    · exact Or.inr ⟨by rw [h, val_one_eq], (h_sgnN (countUpTo_of_eq_one h)).2⟩
  have hsignRval : input_signR.val
      = if input_signed.val = 1 ∧
          2^63 ≤ MulCore.ofHalves4 input_r_x0.val input_r_x1.val input_r_x2.val input_r_x3.val
        then 1 else 0 := by
    refine sign_val hr0 hr1 hr2 hr3 ?_
    rcases hsignedb with h | h
    · rw [h] at h_sgnR0; simp only [sub_zero, one_mul] at h_sgnR0
      exact Or.inl ⟨by rw [h, ZMod.val_zero], by rw [h_sgnR0, ZMod.val_zero]⟩
    · exact Or.inr ⟨by rw [h, val_one_eq], (h_sgnR (countUpTo_of_eq_one h)).2⟩
  have hsignDval : input_signD.val
      = if input_signed.val = 1 ∧
          2^63 ≤ MulCore.ofHalves4 input_d_x0.val input_d_x1.val input_d_x2.val input_d_x3.val
        then 1 else 0 := by
    refine sign_val hd0 hd1 hd2 hd3 ?_
    rcases hsignedb with h | h
    · rw [h] at h_sgnD0; simp only [sub_zero, one_mul] at h_sgnD0
      exact Or.inl ⟨by rw [h, ZMod.val_zero], by rw [h_sgnD0, ZMod.val_zero]⟩
    · exact Or.inr ⟨by rw [h, val_one_eq], (h_sgnD (countUpTo_of_eq_one h)).2⟩
  clear h_sgnN h_sgnN0 h_sgnR h_sgnR0 h_sgnD h_sgnD0 h_sgb
  -- `abs_r` and `abs_d`, from the two `NEG` templates
  obtain ⟨hARlo, hARhi, hARval⟩ := abs_val hr0 hr1 hr2 hr3 hsignRb
    (fun h => (h_nr0 (countUpTo_of_eq_one h)).2)
    (fun h => (h_nr1 (countUpTo_of_eq_one h)).2) h_ar0 h_ar1
  obtain ⟨hADlo, hADhi, hADval⟩ := abs_val hd0 hd1 hd2 hd3 hsignDb
    (fun h => (h_nd0 (countUpTo_of_eq_one h)).2)
    (fun h => (h_nd1 (countUpTo_of_eq_one h)).2) h_ad0 h_ad1
  clear h_nr0 h_nr1 h_ar0 h_ar1 h_nd0 h_nd1 h_ad0 h_ad1
  have hARcond : dword input_absR_lo input_absR_hi
      = if input_signed.val = 1 ∧
          2^63 ≤ MulCore.ofHalves4 input_r_x0.val input_r_x1.val input_r_x2.val input_r_x3.val
        then (2^64 - MulCore.ofHalves4 input_r_x0.val input_r_x1.val input_r_x2.val
          input_r_x3.val) % 2^64
        else MulCore.ofHalves4 input_r_x0.val input_r_x1.val input_r_x2.val input_r_x3.val := by
    rw [hARval]
    by_cases hc : input_signed.val = 1 ∧
        2^63 ≤ MulCore.ofHalves4 input_r_x0.val input_r_x1.val input_r_x2.val input_r_x3.val
    · rw [ite_eq_left (by rw [hsignRval, ite_eq_left hc]), ite_eq_left hc]
    · rw [ite_eq_right (by rw [hsignRval, ite_eq_right hc]; norm_num), ite_eq_right hc]
  have hADcond : dword input_absD_lo input_absD_hi
      = if input_signed.val = 1 ∧
          2^63 ≤ MulCore.ofHalves4 input_d_x0.val input_d_x1.val input_d_x2.val input_d_x3.val
        then (2^64 - MulCore.ofHalves4 input_d_x0.val input_d_x1.val input_d_x2.val
          input_d_x3.val) % 2^64
        else MulCore.ofHalves4 input_d_x0.val input_d_x1.val input_d_x2.val input_d_x3.val := by
    rw [hADval]
    by_cases hc : input_signed.val = 1 ∧
        2^63 ≤ MulCore.ofHalves4 input_d_x0.val input_d_x1.val input_d_x2.val input_d_x3.val
    · rw [ite_eq_left (by rw [hsignDval, ite_eq_left hc]), ite_eq_left hc]
    · rw [ite_eq_right (by rw [hsignDval, ite_eq_right hc]; norm_num), ite_eq_right hc]
  clear hARval hADval hsignDval
  -- the `LT` pull: `|r| < |d|` unless the divisor is zero
  have hlteq := lt_of_guarantees hARlo hARhi hADlo hADhi (h_ltp hcount)
  clear h_ltp
  have hltfact : input_divByZero.val = 0 →
      (if input_signed.val = 1 ∧
          2^63 ≤ MulCore.ofHalves4 input_r_x0.val input_r_x1.val input_r_x2.val input_r_x3.val
        then (2^64 - MulCore.ofHalves4 input_r_x0.val input_r_x1.val input_r_x2.val
          input_r_x3.val) % 2^64
        else MulCore.ofHalves4 input_r_x0.val input_r_x1.val input_r_x2.val input_r_x3.val)
      < (if input_signed.val = 1 ∧
          2^63 ≤ MulCore.ofHalves4 input_d_x0.val input_d_x1.val input_d_x2.val input_d_x3.val
        then (2^64 - MulCore.ofHalves4 input_d_x0.val input_d_x1.val input_d_x2.val
          input_d_x3.val) % 2^64
        else MulCore.ofHalves4 input_d_x0.val input_d_x1.val input_d_x2.val input_d_x3.val) := by
    intro h0
    rw [← hARcond, ← hADcond]
    by_contra hge
    rw [show input_divByZero = 0 by rw [← ZMod.val_eq_zero]; exact h0, sub_zero,
      ite_eq_right hge] at hlteq
    simp at hlteq
  clear hlteq hARcond hADcond hARlo hARhi hADlo hADhi
  -- `div_by_zero = 1` forces every half of the quotient to `65535`
  have hqdbz : input_divByZero.val = 1 →
      MulCore.ofHalves4 input_q_x0.val input_q_x1.val input_q_x2.val input_q_x3.val = 2^64 - 1 := by
    intro h1
    have hne : input_divByZero ≠ 0 := by
      intro h; rw [h, ZMod.val_zero] at h1; exact absurd h1 (by norm_num)
    have g : ∀ x : F p, input_divByZero * (x - 65535) = 0 → x.val = 65535 := by
      intro x hx
      rw [sub_eq_zero.mp ((mul_eq_zero.mp hx).resolve_left hne)]
      exact val_ofNat 65535 (by norm_num)
    rw [g _ hqz0, g _ hqz1, g _ hqz2, g _ hqz3]
    simp only [MulCore.ofHalves4]
    norm_num
  clear hqz0 hqz1 hqz2 hqz3
  -- the remainder has the sign of the numerator
  have hrsgn : MulCore.ofHalves4 input_r_x0.val input_r_x1.val input_r_x2.val input_r_x3.val ≠ 0 →
      (input_signed.val = 1 ∧
          2^63 ≤ MulCore.ofHalves4 input_r_x0.val input_r_x1.val input_r_x2.val input_r_x3.val
        ↔ input_signed.val = 1 ∧
          2^63 ≤ MulCore.ofHalves4 input_n_x0.val input_n_x1.val input_n_x2.val
            input_n_x3.val) := by
    intro hR0
    have hrsum : (input_r_x0 + input_r_x1 + input_r_x2 + input_r_x3).val
        = input_r_x0.val + input_r_x1.val + input_r_x2.val + input_r_x3.val :=
      val_eq_of_eq_natCast (by push_cast; simp only [ZMod.natCast_zmod_val]) (by omega)
    have hne : input_r_x0 + input_r_x1 + input_r_x2 + input_r_x3 ≠ 0 := by
      rw [Ne, ← ZMod.val_eq_zero, hrsum]
      simp only [MulCore.ofHalves4] at hR0
      omega
    have heq : input_signR.val = input_signN.val := by
      rw [sub_eq_zero.mp ((mul_eq_zero.mp h_sgeq).resolve_left hne)]
    rw [hsignRval, hsignNval] at heq
    constructor
    · intro hA
      by_contra hB
      rw [ite_eq_left hA, ite_eq_right hB] at heq
      exact absurd heq (by norm_num)
    · intro hB
      by_contra hA
      rw [ite_eq_right hA, ite_eq_left hB] at heq
      exact absurd heq (by norm_num)
  clear h_sgeq
  -- the two `MUL` pulls: `d · q` is the 128-bit product that `n − r` names
  have hfl : (7 + 32 * input_signed + 64 * input_signQ : F p).val
      = 7 + 32 * input_signed.val + 64 * input_signQ.val :=
    val_eq_of_eq_natCast (by push_cast; simp only [ZMod.natCast_zmod_val]) (by omega)
  have hfl1 : (7 + 32 * input_signed + 64 * input_signQ + 128 : F p).val
      = 7 + 32 * input_signed.val + 64 * input_signQ.val + 128 :=
    val_eq_of_eq_natCast (by push_cast; simp only [ZMod.natCast_zmod_val]) (by omega)
  have hlow := mul_of_guarantees (ls := input_signed.val) (rs := input_signQ.val) (hi := 0)
    (by omega) (by omega) (by omega) (by omega) hsigned2 hsignQ2 (by norm_num)
    (by simp [hfl]) (h_mulA hcount)
  have hhigh := mul_of_guarantees (ls := input_signed.val) (rs := input_signQ.val) (hi := 1)
    (by omega) (by omega) (by omega) (by omega) hsigned2 hsignQ2 (by norm_num)
    (by simp [hfl1]) (h_mulB hcount)
  clear h_mulA h_mulB hfl hfl1
  rw [show ∀ ls rs x y : ℕ, Alu.mulResult ls rs 0 x y
      = (Alu.ext128 ls x * Alu.ext128 rs y) % 2^128 % 2^64 from
    fun _ _ _ _ => by simp [Alu.mulResult], hDw, hQw] at hlow
  rw [show ∀ ls rs x y : ℕ, Alu.mulResult ls rs 1 x y
      = (Alu.ext128 ls x * Alu.ext128 rs y) % 2^128 / 2^64 from
    fun _ _ _ _ => by simp [Alu.mulResult], hDw, hQw] at hhigh
  have hFs := hfillv input_signNSubR hsnsrb
  have hmulfact : MulCore.ofHalves4 input_nSubR_x0.val input_nSubR_x1.val input_nSubR_x2.val
        input_nSubR_x3.val + 2^64 * ((2^64 - 1) * input_signNSubR.val)
      = (Alu.ext128 input_signed.val
            (MulCore.ofHalves4 input_d_x0.val input_d_x1.val input_d_x2.val input_d_x3.val)
          * Alu.ext128 input_signQ.val
            (MulCore.ofHalves4 input_q_x0.val input_q_x1.val input_q_x2.val input_q_x3.val))
        % 2^128 := by
    rw [← hSw]
    exact prod128_of_halves hFs hlow hhigh
  clear hlow hhigh
  -- the 128-bit addition `(n − r) + r = n`, carry by carry. The carries are named through an
  -- existential, not `set`: a let-bound local stays zeta-reducible and the big row polynomials
  -- come back in every later `whnf`.
  obtain ⟨C0, hC0d⟩ : ∃ c : F p, c = (input_nSubR_x0 + 65536 * input_nSubR_x1
      + (input_r_x0 + 65536 * input_r_x1) - (input_n_x0 + 65536 * input_n_x1)) * inv32 := ⟨_, rfl⟩
  rw [← hC0d] at hcb0 hcb1 hcb2 hcb3
  obtain ⟨C1, hC1d⟩ : ∃ c : F p, c = (input_nSubR_x2 + 65536 * input_nSubR_x3
      + (input_r_x2 + 65536 * input_r_x3) + C0 - (input_n_x2 + 65536 * input_n_x3)) * inv32 :=
    ⟨_, rfl⟩
  rw [← hC1d] at hcb1 hcb2 hcb3
  obtain ⟨C2, hC2d⟩ : ∃ c : F p, c = (65535 * input_signNSubR + 65536 * (65535 * input_signNSubR)
      + (65535 * input_signR + 65536 * (65535 * input_signR)) + C1
      - (65535 * input_signN + 65536 * (65535 * input_signN))) * inv32 := ⟨_, rfl⟩
  rw [← hC2d] at hcb2 hcb3
  obtain ⟨C3, hC3d⟩ : ∃ c : F p, c = (65535 * input_signNSubR + 65536 * (65535 * input_signNSubR)
      + (65535 * input_signR + 65536 * (65535 * input_signR)) + C2
      - (65535 * input_signN + 65536 * (65535 * input_signN))) * inv32 := ⟨_, rfl⟩
  rw [← hC3d] at hcb3
  have hb0 := IsBool.val_lt_two (mul_one_sub_eq_zero_iff.mp hcb0)
  have hb1 := IsBool.val_lt_two (mul_one_sub_eq_zero_iff.mp hcb1)
  have hb2 := IsBool.val_lt_two (mul_one_sub_eq_zero_iff.mp hcb2)
  have hb3 := IsBool.val_lt_two (mul_one_sub_eq_zero_iff.mp hcb3)
  have hFr := hfillv input_signR hsignRb
  have hFn := hfillv input_signN hsignNb
  have hsnsr2 := IsBool.val_lt_two hsnsrb
  have hsignR2 := IsBool.val_lt_two hsignRb
  have hsignN2 := IsBool.val_lt_two hsignNb
  have hp0 := hadd2 (input_nSubR_x0 + 65536 * input_nSubR_x1) (input_r_x0 + 65536 * input_r_x1)
    (by omega) (by omega)
  have hp1a := hadd2 (input_nSubR_x2 + 65536 * input_nSubR_x3) (input_r_x2 + 65536 * input_r_x3)
    (by omega) (by omega)
  have hp1 := hadd2 (input_nSubR_x2 + 65536 * input_nSubR_x3 + (input_r_x2 + 65536 * input_r_x3))
    C0 (by omega) (by omega)
  have hp2a := hadd2 (65535 * input_signNSubR + 65536 * (65535 * input_signNSubR))
    (65535 * input_signR + 65536 * (65535 * input_signR)) (by omega) (by omega)
  have hp2 := hadd2 (65535 * input_signNSubR + 65536 * (65535 * input_signNSubR)
    + (65535 * input_signR + 65536 * (65535 * input_signR))) C1 (by omega) (by omega)
  have hp3 := hadd2 (65535 * input_signNSubR + 65536 * (65535 * input_signNSubR)
    + (65535 * input_signR + 65536 * (65535 * input_signR))) C2 (by omega) (by omega)
  have e0 := carry_lift (a := input_nSubR_x0 + 65536 * input_nSubR_x1
    + (input_r_x0 + 65536 * input_r_x1)) (b := input_n_x0 + 65536 * input_n_x1)
    (by omega) (by omega) (mul_one_sub_eq_zero_iff.mp hcb0) hC0d
  have e1 := carry_lift (a := input_nSubR_x2 + 65536 * input_nSubR_x3
    + (input_r_x2 + 65536 * input_r_x3) + C0) (b := input_n_x2 + 65536 * input_n_x3)
    (by omega) (by omega) (mul_one_sub_eq_zero_iff.mp hcb1) hC1d
  have e2 := carry_lift (a := 65535 * input_signNSubR + 65536 * (65535 * input_signNSubR)
    + (65535 * input_signR + 65536 * (65535 * input_signR)) + C1)
    (b := 65535 * input_signN + 65536 * (65535 * input_signN))
    (by omega) (by omega) (mul_one_sub_eq_zero_iff.mp hcb2) hC2d
  have e3 := carry_lift (a := 65535 * input_signNSubR + 65536 * (65535 * input_signNSubR)
    + (65535 * input_signR + 65536 * (65535 * input_signR)) + C2)
    (b := 65535 * input_signN + 65536 * (65535 * input_signN))
    (by omega) (by omega) (mul_one_sub_eq_zero_iff.mp hcb3) hC3d
  clear hcb0 hcb1 hcb2 hcb3 hC0d hC1d hC2d hC3d hadd2 hfillv
  have haddfact : ∃ c : ℕ,
      MulCore.ofHalves4 input_nSubR_x0.val input_nSubR_x1.val input_nSubR_x2.val input_nSubR_x3.val
        + input_signNSubR.val * (2^128 - 2^64)
        + Alu.ext128 input_signed.val
          (MulCore.ofHalves4 input_r_x0.val input_r_x1.val input_r_x2.val input_r_x3.val)
      = Alu.ext128 input_signed.val
          (MulCore.ofHalves4 input_n_x0.val input_n_x1.val input_n_x2.val input_n_x3.val)
        + c * 2^128 := by
    refine ⟨C3.val, ?_⟩
    rw [DvrmCore.ext128_eq, DvrmCore.ext128_eq, ← hsignRval, ← hsignNval]
    -- the four limb equations are 32-bit facts; `omega` is given no hypothesis carrying a
    -- `2^64` or `2^128`, which would make it split on coefficients that large
    clear hmulfact hltfact hqdbz hrsgn hdbzval hNw hDw hQw hRw hSw hsignRval hsignNval
    exact add128_of_limbs (k0 := C0.val) hb1 hb2 hb3 (by omega) (by omega) (by omega) (by omega)
  clear e0 e1 e2 e3 hp0 hp1a hp1 hp2a hp2 hp3 hFs hFr hFn hb0 hb1 hb2 hb3
  clear hw_n0 hw_n1 hw_d0 hw_d1 hw_s0 hw_s1 hsignRval hsignNval
  -- everything `DvrmCore` asks for. Bound with `have`, not destructured: `rcases` on an
  -- application whose conjuncts are this large builds a `casesOn` motive the kernel chokes on
  have hQR := DvrmCore.dvrm_core
    (n := MulCore.ofHalves4 input_n_x0.val input_n_x1.val input_n_x2.val input_n_x3.val)
    (d := MulCore.ofHalves4 input_d_x0.val input_d_x1.val input_d_x2.val input_d_x3.val)
    (q := MulCore.ofHalves4 input_q_x0.val input_q_x1.val input_q_x2.val input_q_x3.val)
    (r := MulCore.ofHalves4 input_r_x0.val input_r_x1.val input_r_x2.val input_r_x3.val)
    (nsr := MulCore.ofHalves4 input_nSubR_x0.val input_nSubR_x1.val input_nSubR_x2.val
      input_nSubR_x3.val)
    (s := input_signNSubR.val) (signed := input_signed.val) (sq := input_signQ.val)
    (dbz := input_divByZero.val)
    (ofHalves4_lt hn0 hn1 hn2 hn3) (ofHalves4_lt hd0 hd1 hd2 hd3)
    (ofHalves4_lt hq0 hq1 hq2 hq3) (ofHalves4_lt hr0 hr1 hr2 hr3)
    (ofHalves4_lt hs0 hs1 hs2 hs3)
    hdbzval hqdbz hrsgn hltfact haddfact hmulfact
  clear hdbzval hqdbz hrsgn hltfact haddfact hmulfact hSw
  have hdv : (8 + 32 * input_signed : F p).val = 8 + 32 * input_signed.val :=
    val_eq_of_eq_natCast (by push_cast; simp only [ZMod.natCast_zmod_val]) (by omega)
  have hdv1 : (8 + 32 * input_signed + 128 : F p).val = 8 + 32 * input_signed.val + 128 :=
    val_eq_of_eq_natCast (by push_cast; simp only [ZMod.natCast_zmod_val]) (by omega)
  exact ⟨fun _ => ⟨hq0, hq1, hq2, hq3, hr0, hr1, hr2, hr3, hQR.1, hQR.2⟩,
    .inr hcount, .inr hcount, .inr hcount, .inr hcount, .inr hcount, .inr hcount, .inr hcount,
    .inr hcount, .inr hcount,
    eq_zero_or_countUpTo hsignRb, eq_zero_or_countUpTo hsignRb,
    eq_zero_or_countUpTo hsignDb, eq_zero_or_countUpTo hsignDb,
    .inr hcount, .inr hcount, .inr hcount, .inr hcount, .inr hcount, .inr hcount, .inr hcount,
    .inr hcount, .inr hcount, .inr hcount, .inr hcount, .inr hcount, .inr hcount, .inr hcount,
    .inr hcount,
    eq_zero_or_countUpTo hsignedb, eq_zero_or_countUpTo hsignedb,
    eq_zero_or_countUpTo hsignedb,
    .inr hcount,
    fun _ => divrem_guarantees (s := input_signed.val) (rm := 0) hsigned2 (by norm_num)
      (by simp [hdv]) (by omega) (by omega) (by rw [hQw, hNw, hDw]; exact hQR.1),
    fun _ => divrem_guarantees (s := input_signed.val) (rm := 1) hsigned2 (by norm_num)
      (by simp [hdv1]) (by omega) (by omega) (by rw [hRw, hNw, hDw]; exact hQR.2)⟩

/-- The value of a sign-extension word `fill s`, for a bit `s`. -/
lemma val_fill {s : F p} (hs : s.val < 2) :
    (65535 * s + 65536 * (65535 * s)).val = 4294967295 * s.val := by
  rcases eq_zero_or_one_of_val hs with ⟨h, -⟩ | ⟨h, -⟩
  · rw [h]; simp
  · rw [h, val_one_eq, mul_one,
      show (65535 : F p) + 65536 * 65535 = ((4294967295 : ℕ) : F p) by norm_num,
      val_natCast_of_lt_two_pow_40 (by norm_num)]

/-- **The four carries of `n_sub_r`.** On the four 32-bit limbs of the honest 128-bit addition
`extended(n − r) + extended(r) = extended(n)`, each of the chip's four virtual carries is the bit
its `IS_BIT` constraint asks for. -/
lemma carries_of_add128 {s0 s1 sF r0 r1 rF n0 n1 nF : F p}
    (hs0 : s0.val < 2^32) (hs1 : s1.val < 2^32) (hsF : sF.val < 2^32)
    (hr0 : r0.val < 2^32) (hr1 : r1.val < 2^32) (hrF : rF.val < 2^32)
    (hn0 : n0.val < 2^32) (hn1 : n1.val < 2^32) (hnF : nF.val < 2^32)
    (hadd : ∃ k : ℕ,
      s0.val + 2^32 * s1.val + 2^64 * sF.val + 2^96 * sF.val
          + (r0.val + 2^32 * r1.val + 2^64 * rF.val + 2^96 * rF.val)
        = n0.val + 2^32 * n1.val + 2^64 * nF.val + 2^96 * nF.val + k * 2^128) :
    ∃ k0 k1 k2 k3 : ℕ, k0 < 2 ∧ k1 < 2 ∧ k2 < 2 ∧ k3 < 2 ∧
      (s0 + r0 - n0) * inv32 = (k0 : F p) ∧
      (s1 + r1 + (s0 + r0 - n0) * inv32 - n1) * inv32 = (k1 : F p) ∧
      (sF + rF + (s1 + r1 + (s0 + r0 - n0) * inv32 - n1) * inv32 - nF) * inv32 = (k2 : F p) ∧
      (sF + rF + (sF + rF + (s1 + r1 + (s0 + r0 - n0) * inv32 - n1) * inv32 - nF) * inv32 - nF)
          * inv32 = (k3 : F p) := by
  obtain ⟨e0, e1, e2, e3, hb0, hb1, hb2, hb3⟩ :=
    DvrmCore.add128_limbs hs0 hs1 hsF hsF hr0 hr1 hrF hrF hn0 hn1 hnF hnF hadd
  set K0 := (ZMod.val s0 + ZMod.val r0) / 2^32 with hK0
  set K1 := (ZMod.val s1 + ZMod.val r1 + K0) / 2^32 with hK1
  set K2 := (ZMod.val sF + ZMod.val rF + K1) / 2^32 with hK2
  set K3 := (ZMod.val sF + ZMod.val rF + K2) / 2^32 with hK3
  have hpair : ∀ a b : F p, a + b = ((a.val + b.val : ℕ) : F p) := by
    intro a b; push_cast; simp only [ZMod.natCast_zmod_val]
  have hc0 : (s0 + r0 - n0) * inv32 = (K0 : F p) :=
    MulCore.carry_cast (ZMod.natCast_zmod_val s0).symm (ZMod.natCast_zmod_val r0).symm
      (ZMod.natCast_zmod_val n0).symm (by omega)
  have hc1 : (s1 + r1 + (s0 + r0 - n0) * inv32 - n1) * inv32 = (K1 : F p) :=
    MulCore.carry_cast (hpair s1 r1) hc0 (ZMod.natCast_zmod_val n1).symm (by omega)
  have hc2 : (sF + rF + (s1 + r1 + (s0 + r0 - n0) * inv32 - n1) * inv32 - nF) * inv32
      = (K2 : F p) :=
    MulCore.carry_cast (hpair sF rF) hc1 (ZMod.natCast_zmod_val nF).symm (by omega)
  have hc3 : (sF + rF + (sF + rF + (s1 + r1 + (s0 + r0 - n0) * inv32 - n1) * inv32 - nF) * inv32
      - nF) * inv32 = (K3 : F p) :=
    MulCore.carry_cast (hpair sF rF) hc2 (ZMod.natCast_zmod_val nF).symm (by omega)
  exact ⟨K0, K1, K2, K3, hb0, hb1, hb2, hb3, hc0, hc1, hc2, hc3⟩

/-- **The two `ZERO` lookups of the `NEG` template.** On an operand held in four halves whose
two's complement is the word pair `(lo, hi)`, the first carry is `1` exactly when the low word of
the operand is nonzero, the second exactly when the operand is. -/
lemma neg_zero_pulls {x0 x1 x2 x3 lo hi : F p}
    (hx0 : x0.val < 2^16) (hx1 : x1.val < 2^16) (hx2 : x2.val < 2^16) (hx3 : x3.val < 2^16)
    (hlo : lo.val < 2^32) (hhi : hi.val < 2^32)
    (hneg : lo.val + 2^32 * hi.val
      = (2^64 - (x0.val + 2^16 * x1.val + 2^32 * x2.val + 2^48 * x3.val)) % 2^64) :
    (ZMod.val (x0 + x1) < 2^20 ∧
        1 - (x0 + 65536 * x1 + lo - 0) * inv32 = if x0 + x1 = 0 then 1 else 0) ∧
      (ZMod.val (x0 + x1 + x2 + x3) < 2^20 ∧
        1 - (x2 + 65536 * x3 + hi + (x0 + 65536 * x1 + lo - 0) * inv32 - 0) * inv32
          = if x0 + x1 + x2 + x3 = 0 then 1 else 0) := by
  have hwlo := Neg.val_wl hx0 hx1
  have hwhi := Neg.val_wl hx2 hx3
  have hsum2 := Neg.val_sum2 hx0 hx1
  have hsum4 := Neg.val_sum4 hx0 hx1 hx2 hx3
  have hsum : (dword (x0 + 65536 * x1) (x2 + 65536 * x3) + dword lo hi) % 2^64 = dword 0 0 := by
    simp only [dword, hwlo, hwhi, ZMod.val_zero]
    omega
  obtain ⟨k0, k1, hk0, hk1, e0, e1, hc0, hc1⟩ :=
    Add.carries_of_sum (l0 := x0 + 65536 * x1) (l1 := x2 + 65536 * x3) (r0 := lo) (r1 := hi)
      (s0 := 0) (s1 := 0) (by omega) (by omega) hlo hhi (by simp) (by simp) hsum
  simp only [ZMod.val_zero] at e0 e1
  refine ⟨⟨by omega, ?_⟩, ⟨by omega, ?_⟩⟩
  · rw [hc0]
    rcases eq_or_ne (x0 + x1) 0 with h | h
    · have hv : (x0 + x1).val = 0 := by rw [h]; simp
      have hk : k0 = 0 := by omega
      simp [h, hk]
    · have hv : (x0 + x1).val ≠ 0 := by rw [Ne, ZMod.val_eq_zero]; exact h
      have hk : k0 = 1 := by omega
      simp [h, hk]
  · rw [hc1]
    rcases eq_or_ne (x0 + x1 + x2 + x3) 0 with h | h
    · have hv : (x0 + x1 + x2 + x3).val = 0 := by rw [h]; simp
      have hk : k1 = 0 := by omega
      simp [h, hk]
    · have hv : (x0 + x1 + x2 + x3).val ≠ 0 := by rw [Ne, ZMod.val_eq_zero]; exact h
      have hk : k1 = 1 := by omega
      simp [h, hk]

/-- **The `LT` message a row pulls.** Its flag byte is `opsel(LT)`, so only the `LT` clause of
the bus fires, and it says `|r| < |d|` exactly when `div_by_zero` is zero. -/
lemma alu_lt_message {rlo rhi dlo dhi dz : F p} {data : ProverData (F p)}
    (hrlo : rlo.val < 2^32) (hrhi : rhi.val < 2^32)
    (hdlo : dlo.val < 2^32) (hdhi : dhi.val < 2^32) (hdz : dz.val < 2)
    (hbit : (if rlo.val + 2^32 * rhi.val < dlo.val + 2^32 * dhi.val then 1 else 0) = 1 - dz.val) :
    Alu.channel.Guarantees ⟨rlo, rhi, dlo, dhi, 4, 1 - dz, 0⟩ data := by
  have hflags : (4 : F p).val = 4 := val_ofNat 4 (by norm_num)
  have hlt : ∀ L R : ℕ, L < 2^64 → R < 2^64 →
      Alu.ltBit 0 0 L R = if L < R then 1 else 0 := by
    intro L R hL hR
    have hu := Alu.ult_ofNat_iff hL hR
    by_cases h : L < R
    · simp [Alu.ltBit, Alu.ltRaw, hu.mpr h, h]
    · have hf : (BitVec.ofNat 64 L).ult (BitVec.ofNat 64 R) = false := by
        rcases Bool.eq_false_or_eq_true ((BitVec.ofNat 64 L).ult (BitVec.ofNat 64 R)) with h' | h'
        · exact absurd (hu.mp h') h
        · exact h'
      simp [Alu.ltBit, Alu.ltRaw, hf, h]
  show Alu.Guarantees _
  refine Alu.Guarantees.of_lt (s := 0) (i := 0) (by norm_num) (by norm_num)
    (by simp [hflags, opselLT]) fun _ _ _ _ => ⟨rfl, ?_⟩
  dsimp only [dword]
  rw [hlt _ _ (by omega) (by omega), hbit]
  rcases eq_zero_or_one_of_val hdz with ⟨h, hv⟩ | ⟨h, hv⟩ <;> rw [hv, h] <;> norm_num

/-- **A `MUL` message a row pulls.** Its flag byte is `opsel(MUL)` plus the two sign bits and the
half selector, so only the `MUL` clause of the bus fires. -/
lemma alu_mul_message {dlo dhi qlo qhi olo ohi flags : F p} {data : ProverData (F p)}
    {ls rs high : ℕ} (hls : ls < 2) (hrs : rs < 2) (hhigh : high < 2)
    (hflags : flags.val = 7 + 32 * ls + 64 * rs + 128 * high)
    (holo : olo.val < 2^32) (hohi : ohi.val < 2^32)
    (hres : olo.val + 2^32 * ohi.val
      = Alu.mulResult ls rs high (dlo.val + 2^32 * dhi.val) (qlo.val + 2^32 * qhi.val)) :
    Alu.channel.Guarantees ⟨dlo, dhi, qlo, qhi, flags, olo, ohi⟩ data :=
  Alu.Guarantees.of_mul hls hrs hhigh (by simp [hflags, opselMUL]) fun _ _ _ _ =>
    ⟨holo, hohi, hres⟩

/-- The two halves of the `MUL` bus result, read off the 128-bit product. -/
lemma mulResult_of_mul128 {ls rs x y lo sm : ℕ} (hlo : lo < 2^64)
    (h : lo + 2^64 * ((2^64 - 1) * sm) = Alu.ext128 ls x * Alu.ext128 rs y % 2^128) :
    Alu.mulResult ls rs 0 x y = lo ∧
      Alu.mulResult ls rs 1 x y = 4294967295 * sm + 2^32 * (4294967295 * sm) := by
  constructor
  · show Alu.ext128 ls x * Alu.ext128 rs y % 2^128 % 2^64 = lo
    rw [← h]; omega
  · show Alu.ext128 ls x * Alu.ext128 rs y % 2^128 / 2^64 = _
    rw [← h]; omega

/-- **The `ZERO` lookup that computes `overflow`.** The sum the chip looks up stays below `2^20`,
and it is zero exactly when the numerator is `−2^63` and the divisor `−1`, which is the signed
division overflow. -/
lemma overflow_zero_pull {n0 n1 n2 n3 d0 d1 d2 d3 sn ov : F p} {s : ℕ}
    (hn0 : n0.val < 2^16) (hn1 : n1.val < 2^16) (hn2 : n2.val < 2^16) (hn3 : n3.val < 2^16)
    (hd0 : d0.val < 2^16) (hd1 : d1.val < 2^16) (hd2 : d2.val < 2^16) (hd3 : d3.val < 2^16)
    (hsn : sn.val
      = if s = 1 ∧ 2^63 ≤ n0.val + 2^16 * n1.val + 2^32 * n2.val + 2^48 * n3.val then 1 else 0)
    (hov : ov.val = if s = 1 ∧ n0.val + 2^16 * n1.val + 2^32 * n2.val + 2^48 * n3.val = 2^63
        ∧ d0.val + 2^16 * d1.val + 2^32 * d2.val + 2^48 * d3.val = 2^64 - 1 then 1 else 0) :
    ZMod.val (n0 + n1 + n2 + (n3 - 32768 * sn) + (1 - sn) + (65535 - d0) + (65535 - d1)
        + (65535 - d2) + (65535 - d3)) < 2^20 ∧
      ov = if n0 + n1 + n2 + (n3 - 32768 * sn) + (1 - sn) + (65535 - d0) + (65535 - d1)
        + (65535 - d2) + (65535 - d3) = 0 then 1 else 0 := by
  have hsn2 : sn.val < 2 := by rw [hsn]; split_ifs <;> omega
  have hsn_iff : sn.val = 1 ↔ (s = 1 ∧ 2^63 ≤ n0.val + 2^16 * n1.val + 2^32 * n2.val
      + 2^48 * n3.val) := by
    rw [hsn]
    refine ⟨fun h => ?_, fun h => by rw [ite_eq_left h]⟩
    by_contra hc
    rw [ite_eq_right hc] at h
    exact absurd h (by norm_num)
  have hge : 32768 * sn.val ≤ n3.val := by
    rcases (by omega : sn.val = 0 ∨ sn.val = 1) with h | h
    · omega
    · have := (hsn_iff.mp h).2; omega
  obtain ⟨A, hA⟩ : ∃ A : ℕ, n3.val = A + 32768 * sn.val := ⟨n3.val - 32768 * sn.val, by omega⟩
  obtain ⟨B, hB⟩ : ∃ B : ℕ, 1 = B + sn.val := ⟨1 - sn.val, by omega⟩
  obtain ⟨C0, hC0⟩ : ∃ C : ℕ, 65535 = C + d0.val := ⟨65535 - d0.val, by omega⟩
  obtain ⟨C1, hC1⟩ : ∃ C : ℕ, 65535 = C + d1.val := ⟨65535 - d1.val, by omega⟩
  obtain ⟨C2, hC2⟩ : ∃ C : ℕ, 65535 = C + d2.val := ⟨65535 - d2.val, by omega⟩
  obtain ⟨C3, hC3⟩ : ∃ C : ℕ, 65535 = C + d3.val := ⟨65535 - d3.val, by omega⟩
  have eA : ((A : ℕ) : F p) = n3 - 32768 * sn := by
    have h := congrArg (fun m : ℕ => (m : F p)) hA
    push_cast at h; simp only [ZMod.natCast_zmod_val] at h; linear_combination -h
  have eB : ((B : ℕ) : F p) = 1 - sn := by
    have h := congrArg (fun m : ℕ => (m : F p)) hB
    push_cast at h; simp only [ZMod.natCast_zmod_val] at h; linear_combination -h
  have eC0 : ((C0 : ℕ) : F p) = 65535 - d0 := by
    have h := congrArg (fun m : ℕ => (m : F p)) hC0
    push_cast at h; simp only [ZMod.natCast_zmod_val] at h; linear_combination -h
  have eC1 : ((C1 : ℕ) : F p) = 65535 - d1 := by
    have h := congrArg (fun m : ℕ => (m : F p)) hC1
    push_cast at h; simp only [ZMod.natCast_zmod_val] at h; linear_combination -h
  have eC2 : ((C2 : ℕ) : F p) = 65535 - d2 := by
    have h := congrArg (fun m : ℕ => (m : F p)) hC2
    push_cast at h; simp only [ZMod.natCast_zmod_val] at h; linear_combination -h
  have eC3 : ((C3 : ℕ) : F p) = 65535 - d3 := by
    have h := congrArg (fun m : ℕ => (m : F p)) hC3
    push_cast at h; simp only [ZMod.natCast_zmod_val] at h; linear_combination -h
  have hWcast : ((n0.val + n1.val + n2.val + A + B + C0 + C1 + C2 + C3 : ℕ) : F p)
      = n0 + n1 + n2 + (n3 - 32768 * sn) + (1 - sn) + (65535 - d0) + (65535 - d1)
        + (65535 - d2) + (65535 - d3) := by
    push_cast
    rw [eA, eB, eC0, eC1, eC2, eC3]
    simp only [ZMod.natCast_zmod_val]
  have hval : ZMod.val (n0 + n1 + n2 + (n3 - 32768 * sn) + (1 - sn) + (65535 - d0) + (65535 - d1)
      + (65535 - d2) + (65535 - d3)) = n0.val + n1.val + n2.val + A + B + C0 + C1 + C2 + C3 := by
    rw [← hWcast]; exact val_natCast_of_lt_two_pow_40 (by omega)
  have hE0 : (n0 + n1 + n2 + (n3 - 32768 * sn) + (1 - sn) + (65535 - d0) + (65535 - d1)
        + (65535 - d2) + (65535 - d3) = 0)
      ↔ n0.val + n1.val + n2.val + A + B + C0 + C1 + C2 + C3 = 0 := by
    rw [← ZMod.val_eq_zero, hval]
  have hiff : (s = 1 ∧ n0.val + 2^16 * n1.val + 2^32 * n2.val + 2^48 * n3.val = 2^63
        ∧ d0.val + 2^16 * d1.val + 2^32 * d2.val + 2^48 * d3.val = 2^64 - 1)
      ↔ n0.val + n1.val + n2.val + A + B + C0 + C1 + C2 + C3 = 0 := by
    refine ⟨fun h => ?_, fun h0 => ?_⟩
    · have hs1 : sn.val = 1 := hsn_iff.mpr ⟨h.1, by omega⟩
      omega
    · have hs1 : sn.val = 1 := by omega
      exact ⟨(hsn_iff.mp hs1).1, by omega, by omega⟩
  refine ⟨by omega, ?_⟩
  rw [if_congr hE0 rfl rfl, if_congr hiff.symm rfl rfl, ← ZMod.natCast_zmod_val ov, hov]
  split_ifs <;> norm_num

/-- The honest 128-bit addition, rewritten on the four 32-bit limbs the carry constraints use. -/
lemma add128_limbs_of_honest {m0 m1 m2 m3 r0 r1 r2 r3 n0 n1 n2 n3 sm sr sn k : ℕ}
    (h : m0 + 2^16 * m1 + 2^32 * m2 + 2^48 * m3 + sm * (2^128 - 2^64)
        + (r0 + 2^16 * r1 + 2^32 * r2 + 2^48 * r3 + sr * (2^128 - 2^64))
      = n0 + 2^16 * n1 + 2^32 * n2 + 2^48 * n3 + sn * (2^128 - 2^64) + k * 2^128) :
    m0 + 65536 * m1 + 2^32 * (m2 + 65536 * m3) + 2^64 * (4294967295 * sm)
          + 2^96 * (4294967295 * sm)
        + (r0 + 65536 * r1 + 2^32 * (r2 + 65536 * r3) + 2^64 * (4294967295 * sr)
          + 2^96 * (4294967295 * sr))
      = n0 + 65536 * n1 + 2^32 * (n2 + 65536 * n3) + 2^64 * (4294967295 * sn)
        + 2^96 * (4294967295 * sn) + k * 2^128 := by
  omega

theorem completeness :
    GeneralFormalCircuit.Completeness (F p) (Input := Row) (Output := unit) main
      (fun r _ _ => Honest r ∨ Padding r) (fun _ _ _ => True) := by
  circuit_proof_start [main, Honest, Halves4.value, Row.nVal, Row.dVal, Row.qVal, Row.rVal,
    Padding, MulCore.ofHalves4, word, fill, neg, sign, isBit, areBytes, Neg.toWL, Neg.zeroWL,
    Add.carry0, Add.carry1, Bitwise.areBytesTable, Bitwise.isHalfTable, Bitwise.msb16Table,
    Bitwise.zeroTable]
  rcases h_assumptions with
    ⟨hmuQ, hmuR, hsgb, hn0, hn1, hn2, hn3, hd0, hd1, hd2, hd3, hq0, hq1, hq2, hq3,
      hr0, hr1, hr2, hr3, hm0, hm1, hm2, hm3, harlo, harhi, hadlo, hadhi,
      hqv, hrv, hmv, hsmv, hovv, hdzv, hsqv, hsnv, hsdv, hsrv, harv, hadv⟩ |
    ⟨pn0, pn1, pn2, pn3, pd0, pd1, pd2, pd3, psg, pq0, pq1, pq2, pq3,
      pr0, pr1, pr2, pr3, pdz, pov, parlo, parhi, padlo, padhi,
      pm0, pm1, pm2, pm3, psm, psn, psd, psq, psr, pmq, pmr⟩
  case inr =>
    -- a padding row: every column is zero, so every counted pull is inactive, every push is
    -- suppressed, and every constraint is a product with a zero factor
    simp only [pn0, pn1, pn2, pn3, pd0, pd1, pd2, pd3, psg, pq0, pq1, pq2, pq3,
      pr0, pr1, pr2, pr3, pdz, pov, parlo, parhi, padlo, padhi, pm0, pm1, pm2, pm3,
      psm, psn, psd, psq, psr, pmq, pmr, circuit_norm]
    -- a zero multiplicity is not a count, so every counted pull's obligation is vacuous
    norm_num [not_countUpTo_zero]
  have hveq : ∀ a b : F p, ZMod.val a = ZMod.val b → a = b := fun a b h => by
    rw [← ZMod.natCast_zmod_val a, ← ZMod.natCast_zmod_val b, h]
  have hsv2 : ZMod.val input_signed < 2 := IsBool.val_lt_two hsgb
  set NV := ZMod.val input_n_x0 + 2^16 * ZMod.val input_n_x1 + 2^32 * ZMod.val input_n_x2
    + 2^48 * ZMod.val input_n_x3 with hNV
  set DV := ZMod.val input_d_x0 + 2^16 * ZMod.val input_d_x1 + 2^32 * ZMod.val input_d_x2
    + 2^48 * ZMod.val input_d_x3 with hDV
  set QV := ZMod.val input_q_x0 + 2^16 * ZMod.val input_q_x1 + 2^32 * ZMod.val input_q_x2
    + 2^48 * ZMod.val input_q_x3 with hQV
  set RV := ZMod.val input_r_x0 + 2^16 * ZMod.val input_r_x1 + 2^32 * ZMod.val input_r_x2
    + 2^48 * ZMod.val input_r_x3 with hRV
  set MV := ZMod.val input_nSubR_x0 + 2^16 * ZMod.val input_nSubR_x1
    + 2^32 * ZMod.val input_nSubR_x2 + 2^48 * ZMod.val input_nSubR_x3 with hMV
  simp only [← hNV, ← hDV, ← hRV] at hsmv hovv hsnv hsdv hsrv hdzv
  obtain ⟨hq64, hr64, hnsr64, hsm1, har64, had64, hsq1, hdzq, hsrsn, har1, har0, had1, had0,
      hltd, hmul128, hadd128⟩ :=
    DvrmCore.honest_nat (n := NV) (d := DV) (s := ZMod.val input_signed)
      (by omega) (by omega) hsv2
  simp only [← hqv, ← hrv] at hq64 hr64 hnsr64 hsm1 har64 hdzq hsrsn har1 har0 hltd hmul128 hadd128
  simp only [← hmv, ← hsmv, ← hovv, ← hsqv] at hnsr64 hsm1 hmul128 hadd128
  -- every flag of the row is a bit
  have hsm2 : ZMod.val input_signNSubR < 2 := by omega
  have hsr2 : ZMod.val input_signR < 2 := by rw [hsrv]; split_ifs <;> omega
  have hsn2 : ZMod.val input_signN < 2 := by rw [hsnv]; split_ifs <;> omega
  have hsd2 : ZMod.val input_signD < 2 := by rw [hsdv]; split_ifs <;> omega
  have hov2 : ZMod.val input_overflow < 2 := by rw [hovv]; split_ifs <;> omega
  have hdz2 : ZMod.val input_divByZero < 2 := by rw [hdzv]; split_ifs <;> omega
  have hsq2 : ZMod.val input_signQ < 2 := by
    rw [hsqv]
    rcases (by omega : ZMod.val input_signed = 0 ∨ ZMod.val input_signed = 1) with h | h <;>
      rw [h] <;> omega
  -- the 32-bit words the row is read in
  have hwn0 := Neg.val_wl hn0 hn1
  have hwn1 := Neg.val_wl hn2 hn3
  have hwd0 := Neg.val_wl hd0 hd1
  have hwd1 := Neg.val_wl hd2 hd3
  have hwq0 := Neg.val_wl hq0 hq1
  have hwq1 := Neg.val_wl hq2 hq3
  have hwr0 := Neg.val_wl hr0 hr1
  have hwr1 := Neg.val_wl hr2 hr3
  have hwm0 := Neg.val_wl hm0 hm1
  have hwm1 := Neg.val_wl hm2 hm3
  have hfm := val_fill hsm2
  have hfr := val_fill hsr2
  have hfn := val_fill hsn2
  -- a negative operand's absolute value is its two's complement. The `NEG` pulls are counted,
  -- so the hypothesis arrives as a count rather than as `sign = 1`; for a bit the two say the
  -- same thing.
  have hnegR : Channel.CountUpTo muMax input_signR →
      ZMod.val input_absR_lo + 2^32 * ZMod.val input_absR_hi = (2^64 - RV) % 2^64 := fun h1 => by
    rw [harv]
    exact har1 (by
      rw [← hsrv, eq_one_of_countUpTo (isBool_of_val_lt_two hsr2) h1]; exact val_one_eq)
  have hnegD : Channel.CountUpTo muMax input_signD →
      ZMod.val input_absD_lo + 2^32 * ZMod.val input_absD_hi = (2^64 - DV) % 2^64 := fun h1 => by
    rw [hadv]
    exact had1 (by
      rw [← hsdv, eq_one_of_countUpTo (isBool_of_val_lt_two hsd2) h1]; exact val_one_eq)
  -- the 128-bit addition `extended(n − r) + extended(r) = extended(n)`, limb by limb
  have hadd' : ∃ k : ℕ,
      ZMod.val (input_nSubR_x0 + 65536 * input_nSubR_x1)
          + 2^32 * ZMod.val (input_nSubR_x2 + 65536 * input_nSubR_x3)
          + 2^64 * ZMod.val (65535 * input_signNSubR + 65536 * (65535 * input_signNSubR))
          + 2^96 * ZMod.val (65535 * input_signNSubR + 65536 * (65535 * input_signNSubR))
          + (ZMod.val (input_r_x0 + 65536 * input_r_x1)
            + 2^32 * ZMod.val (input_r_x2 + 65536 * input_r_x3)
            + 2^64 * ZMod.val (65535 * input_signR + 65536 * (65535 * input_signR))
            + 2^96 * ZMod.val (65535 * input_signR + 65536 * (65535 * input_signR)))
        = ZMod.val (input_n_x0 + 65536 * input_n_x1)
            + 2^32 * ZMod.val (input_n_x2 + 65536 * input_n_x3)
            + 2^64 * ZMod.val (65535 * input_signN + 65536 * (65535 * input_signN))
            + 2^96 * ZMod.val (65535 * input_signN + 65536 * (65535 * input_signN))
            + k * 2^128 := by
    obtain ⟨k, hk⟩ := hadd128
    simp only [DvrmCore.ext128_eq, ← hsrv, ← hsnv] at hk
    rw [hMV, hRV, hNV] at hk
    exact ⟨k, by
      rw [hwm0, hwm1, hfm, hwr0, hwr1, hfr, hwn0, hwn1, hfn]
      exact add128_limbs_of_honest hk⟩
  -- the two `MUL` messages
  have hDVeq : ZMod.val (input_d_x0 + 65536 * input_d_x1)
      + 2^32 * ZMod.val (input_d_x2 + 65536 * input_d_x3) = DV := by rw [hwd0, hwd1]; omega
  have hQVeq : ZMod.val (input_q_x0 + 65536 * input_q_x1)
      + 2^32 * ZMod.val (input_q_x2 + 65536 * input_q_x3) = QV := by rw [hwq0, hwq1]; omega
  have hflo : ZMod.val (7 + 32 * input_signed + 64 * input_signQ)
      = 7 + 32 * ZMod.val input_signed + 64 * ZMod.val input_signQ :=
    val_eq_of_eq_natCast (by push_cast; simp only [ZMod.natCast_zmod_val]) (by omega)
  have hfhi : ZMod.val (7 + 32 * input_signed + 64 * input_signQ + 128)
      = 7 + 32 * ZMod.val input_signed + 64 * ZMod.val input_signQ + 128 :=
    val_eq_of_eq_natCast (by push_cast; simp only [ZMod.natCast_zmod_val]) (by omega)
  obtain ⟨hlow, hhigh⟩ := mulResult_of_mul128 hnsr64 hmul128
  -- dividing by zero forces every limb of the quotient
  have hqall : ZMod.val input_divByZero = 1 →
      input_q_x0 = 65535 ∧ input_q_x1 = 65535 ∧ input_q_x2 = 65535 ∧ input_q_x3 = 65535 := by
    intro hv
    have hq : QV = 2^64 - 1 := hdzq (by rw [← hdzv]; exact hv)
    refine ⟨hveq _ _ ?_, hveq _ _ ?_, hveq _ _ ?_, hveq _ _ ?_⟩ <;>
      rw [show ZMod.val (65535 : F p) = 65535 from val_ofNat 65535 (by norm_num)] <;> omega
  -- everything the goals below need is in hand; drop the heavy arithmetic so that the many
  -- small `omega` calls do not have to re-read it
  clear hmul128 hadd128 hq64 hr64 hnsr64 hsm1 har64 had64 hsq1 har1 had1 hdzq hqv hrv hmv h_env
    h_input
  obtain ⟨k0, k1, k2, k3, hk0, hk1, hk2, hk3, hc0, hc1, hc2, hc3⟩ :=
    carries_of_add128 (by omega) (by omega) (by omega) (by omega) (by omega) (by omega)
      (by omega) (by omega) (by omega) hadd'
  clear hadd'
  refine ⟨⟨hmuQ, hmuR⟩,
    fun _ => hn0, fun _ => hn1, fun _ => hn2, fun _ => hn3,
    fun _ => hd0, fun _ => hd1, fun _ => hd2, fun _ => hd3,
    mul_one_sub_eq_zero_iff.mpr hsgb, ?_, fun _ => ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_,
    ?_, fun _ => ?_,
    fun _ => hr0, fun _ => hr1, fun _ => hr2, fun _ => hr3,
    fun _ => hm0, fun _ => hm1, fun _ => hm2, fun _ => hm3,
    ?_, ?_, ?_, ?_, ?_, fun _ => ?_, fun _ => ?_,
    fun _ => hq0, fun _ => hq1, fun _ => hq2, fun _ => hq3,
    ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, fun _ => ?_⟩
  · -- sign_equality: `r ≠ 0` forces `sign_r = sign_n`
    by_cases hrz : RV = 0
    · have e0 : input_r_x0 = 0 := by rw [← ZMod.val_eq_zero]; omega
      have e1 : input_r_x1 = 0 := by rw [← ZMod.val_eq_zero]; omega
      have e2 : input_r_x2 = 0 := by rw [← ZMod.val_eq_zero]; omega
      have e3 : input_r_x3 = 0 := by rw [← ZMod.val_eq_zero]; omega
      rw [e0, e1, e2, e3]; ring
    · rw [hveq input_signR input_signN (by rw [hsrv, hsnv]; exact hsrsn hrz)]; ring
  · -- `LT`: `|r| < |d|` unless `d = 0`
    exact alu_lt_message harlo harhi hadlo hadhi hdz2 (by rw [harv, hadv, hdzv]; exact hltd)
  · exact fun h1 => (neg_zero_pulls hr0 hr1 hr2 hr3 harlo harhi (hnegR h1)).1
  · exact fun h1 => (neg_zero_pulls hr0 hr1 hr2 hr3 harlo harhi (hnegR h1)).2
  · -- `abs_r = r` when `r` is not negative
    rcases eq_zero_or_one_of_val hsr2 with ⟨h, hv⟩ | ⟨h, -⟩
    · have har : (Alu.sgn64 (ZMod.val input_signed) RV).natAbs = RV :=
        har0 (by rw [← hsrv]; exact hv)
      rw [hveq input_absR_lo (input_r_x0 + 65536 * input_r_x1) (by rw [hwr0]; omega),
        sub_self, mul_zero]
    · rw [h, sub_self, zero_mul]
  · rcases eq_zero_or_one_of_val hsr2 with ⟨h, hv⟩ | ⟨h, -⟩
    · have har : (Alu.sgn64 (ZMod.val input_signed) RV).natAbs = RV :=
        har0 (by rw [← hsrv]; exact hv)
      rw [hveq input_absR_hi (input_r_x2 + 65536 * input_r_x3) (by rw [hwr1]; omega),
        sub_self, mul_zero]
    · rw [h, sub_self, zero_mul]
  · exact fun h1 => (neg_zero_pulls hd0 hd1 hd2 hd3 hadlo hadhi (hnegD h1)).1
  · exact fun h1 => (neg_zero_pulls hd0 hd1 hd2 hd3 hadlo hadhi (hnegD h1)).2
  · rcases eq_zero_or_one_of_val hsd2 with ⟨h, hv⟩ | ⟨h, -⟩
    · have had : (Alu.sgn64 (ZMod.val input_signed) DV).natAbs = DV :=
        had0 (by rw [← hsdv]; exact hv)
      rw [hveq input_absD_lo (input_d_x0 + 65536 * input_d_x1) (by rw [hwd0]; omega),
        sub_self, mul_zero]
    · rw [h, sub_self, zero_mul]
  · rcases eq_zero_or_one_of_val hsd2 with ⟨h, hv⟩ | ⟨h, -⟩
    · have had : (Alu.sgn64 (ZMod.val input_signed) DV).natAbs = DV :=
        had0 (by rw [← hsdv]; exact hv)
      rw [hveq input_absD_hi (input_d_x2 + 65536 * input_d_x3) (by rw [hwd1]; omega),
        sub_self, mul_zero]
    · rw [h, sub_self, zero_mul]
  · -- `sign_q = signed · (1 − overflow)`
    have hsq : input_signQ
        = ((ZMod.val input_signed * (1 - ZMod.val input_overflow) : ℕ) : F p) := by
      rw [← hsqv, ZMod.natCast_zmod_val]
    rcases eq_zero_or_one_of_val hsv2 with ⟨hs, hvs⟩ | ⟨hs, hvs⟩ <;>
      rcases eq_zero_or_one_of_val hov2 with ⟨ho, hvo⟩ | ⟨ho, hvo⟩ <;>
      rw [hsq, hvs, hvo, hs, ho] <;> norm_num
  · -- `ZERO`: the overflow flag is `n = −2^63 ∧ d = −1`
    exact overflow_zero_pull hn0 hn1 hn2 hn3 hd0 hd1 hd2 hd3 hsnv hovv
  · rw [hc0]; rcases (by omega : k0 = 0 ∨ k0 = 1) with rfl | rfl <;> simp
  · rw [hc1]; rcases (by omega : k1 = 0 ∨ k1 = 1) with rfl | rfl <;> simp
  · rw [hc2]; rcases (by omega : k2 = 0 ∨ k2 = 1) with rfl | rfl <;> simp
  · rw [hc3]; rcases (by omega : k3 = 0 ∨ k3 = 1) with rfl | rfl <;> simp
  · exact mul_one_sub_eq_zero_iff.mpr (isBool_of_val_lt_two hsm2)
  · -- `MUL`, low half
    refine alu_mul_message (ls := ZMod.val input_signed) (rs := ZMod.val input_signQ) (high := 0)
      hsv2 hsq2 (by norm_num) (by omega) (by omega) (by omega) ?_
    rw [hwm0, hwm1, hDVeq, hQVeq, hlow]; omega
  · -- `MUL`, high half
    refine alu_mul_message (ls := ZMod.val input_signed) (rs := ZMod.val input_signQ) (high := 1)
      hsv2 hsq2 (by norm_num) (by omega) (by omega) (by omega) ?_
    rw [hfm, hDVeq, hQVeq, hhigh]
  · -- `SIGN` on the numerator
    intro h1'
    have h1 : input_signed = 1 := eq_one_of_countUpTo hsgb h1'
    have hv : ZMod.val input_signed = 1 := by rw [h1]; exact val_one_eq
    refine ⟨hn3, ?_⟩
    rw [hsnv]
    by_cases hge : 2^63 ≤ NV
    · rw [ite_eq_left ⟨hv, hge⟩]; omega
    · rw [ite_eq_right (fun hc => hge hc.2)]; omega
  · rcases hsgb with h | h
    · have hne : ZMod.val input_signed ≠ 1 := by rw [h, ZMod.val_zero]; norm_num
      rw [show input_signN = 0 by rw [← ZMod.val_eq_zero, hsnv, ite_eq_right (fun hc => hne hc.1)],
        mul_zero]
    · rw [h, sub_self, zero_mul]
  · intro h1'
    have h1 : input_signed = 1 := eq_one_of_countUpTo hsgb h1'
    have hv : ZMod.val input_signed = 1 := by rw [h1]; exact val_one_eq
    refine ⟨hr3, ?_⟩
    rw [hsrv]
    by_cases hge : 2^63 ≤ RV
    · rw [ite_eq_left ⟨hv, hge⟩]; omega
    · rw [ite_eq_right (fun hc => hge hc.2)]; omega
  · rcases hsgb with h | h
    · have hne : ZMod.val input_signed ≠ 1 := by rw [h, ZMod.val_zero]; norm_num
      rw [show input_signR = 0 by rw [← ZMod.val_eq_zero, hsrv, ite_eq_right (fun hc => hne hc.1)],
        mul_zero]
    · rw [h, sub_self, zero_mul]
  · intro h1'
    have h1 : input_signed = 1 := eq_one_of_countUpTo hsgb h1'
    have hv : ZMod.val input_signed = 1 := by rw [h1]; exact val_one_eq
    refine ⟨hd3, ?_⟩
    rw [hsdv]
    by_cases hge : 2^63 ≤ DV
    · rw [ite_eq_left ⟨hv, hge⟩]; omega
    · rw [ite_eq_right (fun hc => hge hc.2)]; omega
  · rcases hsgb with h | h
    · have hne : ZMod.val input_signed ≠ 1 := by rw [h, ZMod.val_zero]; norm_num
      rw [show input_signD = 0 by rw [← ZMod.val_eq_zero, hsdv, ite_eq_right (fun hc => hne hc.1)],
        mul_zero]
    · rw [h, sub_self, zero_mul]
  · rcases eq_zero_or_one_of_val hdz2 with ⟨h, -⟩ | ⟨-, hv⟩
    · rw [h, zero_mul]
    · rw [(hqall hv).1, sub_self, mul_zero]
  · rcases eq_zero_or_one_of_val hdz2 with ⟨h, -⟩ | ⟨-, hv⟩
    · rw [h, zero_mul]
    · rw [(hqall hv).2.1, sub_self, mul_zero]
  · rcases eq_zero_or_one_of_val hdz2 with ⟨h, -⟩ | ⟨-, hv⟩
    · rw [h, zero_mul]
    · rw [(hqall hv).2.2.1, sub_self, mul_zero]
  · rcases eq_zero_or_one_of_val hdz2 with ⟨h, -⟩ | ⟨-, hv⟩
    · rw [h, zero_mul]
    · rw [(hqall hv).2.2.2, sub_self, mul_zero]
  · -- `ZERO`: the div-by-zero flag
    have hs4 := Neg.val_sum4 hd0 hd1 hd2 hd3
    refine ⟨by omega, ?_⟩
    have hiff : (input_d_x0 + input_d_x1 + input_d_x2 + input_d_x3 = 0) ↔ DV = 0 := by
      rw [← ZMod.val_eq_zero, hs4]; omega
    rw [if_congr hiff rfl rfl, ← ZMod.natCast_zmod_val input_divByZero, hdzv]
    split_ifs <;> norm_num

/-- `RequirementsChannelsLawful` from a cover of the circuit's channels plus a local proof for
the ones left out of the declared list. A channel not declared has to be one the circuit receives
on, and its interactions have to meet their requirement here rather than at an ensemble.

This is what lets a chip stop deferring a bus whose requirement it can already meet, which is not
a tidiness point: `ARE_BYTES` bounds every counted multiplicity, so it has to be the first channel
finished in any ensemble, and a requirement on the first channel can only be discharged from
soundness -- which wants every guarantee the chip has, and at that point has none. A chip that
defers `ARE_BYTES` cannot sit in such an ensemble at all.

Going through `shallowChannels` for the cover is also what keeps the obligation affordable on a
chip with 35 interactions. Proving `InChannelsOrRequirements` in place means `simp only
[circuit_norm, main]` expands it into a 35-fold conjunction, each conjunct an `n`-way disjunction
`channel = c₁ ∨ … ∨ channel = cₙ ∨ Requirements` over *closed* channel terms, and the closing
`tauto` then has to settle a few hundred `RawChannel` equalities by `whnf`, comparing structures
whose `Guarantees`/`Requirements` fields are closures over `Alu.Guarantees` -- an eight-clause
predicate on seven 64-bit operands. Here the cover leaves one goal with the channel as a
*variable*, where both sides came out of the same normalization pass and each case closes
syntactically, and only the undeclared channels ever reach an interaction. -/
theorem requirementsChannelsLawful_of_cover {F : Type} [FiniteField F]
    {ops : Operations F} {guarantees requirements : List (RawChannel F)}
    (h_subcircuits : ops.subcircuitChannelsWithRequirements ⊆ requirements)
    (h_cover : ∀ channel ∈ ops.shallowChannels,
      channel ∈ guarantees ∨ channel ∈ requirements)
    (h_local : ∀ env, ConstraintsHold.Shallow env ops →
      ∀ i ∈ ops.shallowInteractions, i.channel ∉ requirements → i.Requirements env) :
    ops.RequirementsChannelsLawful guarantees requirements := by
  refine ⟨h_subcircuits, h_cover, fun env hc => ?_⟩
  rw [Operations.inChannelsOrRequirements_iff_forall_mem]
  intro i hi
  by_cases h : i.channel ∈ requirements
  · exact Or.inl h
  · exact Or.inr (h_local env hc i hi h)

/-- The chip touches five raw channels and defers three of them. `pullUpTo` puts the range
checks, the two `ZERO` checks and the three borrowed `ALU` messages on the *counted* channels
`IsHalfC`, `ZeroC` and `AluC`, whose pull requirement is a bound on the multiplicity that only
`ARE_BYTES` can give, and that bound is what an ensemble has to supply.

The other two are met here. `ARE_BYTES` itself is pulled once with multiplicity `1`, which is
already `mult ∈ {0, 1}`; and the three `MSB16` pulls come from the `SIGN` template, whose
multiplicity is the sign flag, which `isBit r.signed` constrains to a bit. Declaring either one
would be worse than redundant: `ARE_BYTES` bounds every counted multiplicity, so it has to be the
first channel finished in any ensemble, and a requirement on a channel finished first can only be
discharged from soundness — which wants every guarantee the chip has, including the ones that are
not finished yet. A chip that defers `ARE_BYTES` cannot sit in such an ensemble at all. -/
theorem requirementsChannelsLawful : ∀ (input : Var Row (F p)) (offset : ℕ),
    ((main input).operations offset).RequirementsChannelsLawful
      elaborated.channelsWithGuarantees
      [AluC, IsHalfC, ZeroC] := by
  intro input offset
  apply requirementsChannelsLawful_of_cover
  · -- the chip has no subcircuits
    simp only [circuit_norm, main]
  · -- the 35 channels, read off the operations without touching a single message
    simp only [circuit_norm, main]
    intro channel h
    tauto
  · -- the one `ARE_BYTES` pull and the three `MSB16` pulls, met in place
    intro env hc
    simp only [circuit_norm, main] at hc
    simp only [circuit_norm, main, List.not_mem_nil]
    intro i hi
    repeat' rcases hi with rfl | hi
    all_goals intro hne
    all_goals try (revert hne; simp only [circuit_norm, AluC, Alu.channelC, IsHalfC, ZeroC,
      Bitwise.IsHalf, Bitwise.Zero, Alu.channel, StaticLookupChannel.toChannel])
    all_goals exact fun _ => eq_zero_or_countUpTo (mul_one_sub_eq_zero_iff.mp (by tauto))

/-! ## What an ensemble reads off a row

The chip's multiplicities come in three kinds: the row's `mu = mu_q + mu_r`, which `ARE_BYTES`
bounds; the literal `1` of the `ARE_BYTES` pull itself; and the sign flags, under which `NEG`
pulls `ZERO` twice per call. The third kind is the one an ensemble cannot see from `ARE_BYTES`,
so the facts it needs about those flags are stated here, where the circuit is in scope. -/

/-- `signed` is a bit, and the quotient's sign is `signed` unless the row overflows. These are
the two constraints an ensemble needs to read the opcode of an `ALU` message off a row: every one
the chip touches carries `base + 32 * signed` or `base + 32 * signed + 64 * sign_q`, plus `128`
for the high half. The chip says `overflow` is a bit nowhere; `ZERO` does.

The `obtain` reads the nineteen `assertZero`s in the order `main` writes them: `isBit signed`
first and the overflow definition seventh. -/
theorem flags_of_constraints {env : Environment (F p)} {input : Var Row (F p)} {offset : ℕ}
    (hc : ((main input).operations offset).ConstraintsHold env) :
    IsBool (Expression.eval env input.signed) ∧
      Expression.eval env input.signed * (1 - Expression.eval env input.overflow)
        - Expression.eval env input.signQ = 0 := by
  simp only [circuit_norm, main, isBit, sign, neg, forall_eq_or_imp, forall_eq] at hc
  obtain ⟨hs,-,-,-,-,-,hq,-,-,-,-,-,-,-,-,-,-,-,-⟩ := hc
  exact ⟨mul_one_sub_eq_zero_iff.mp hs, hq⟩

/-- The row's two multiplicities are bytes, which is the chip's own `ARE_BYTES` pull saying so.
Stated here, in the chip's own spelling, so an ensemble can feed it to the counting lemmas
without first normalizing a thirty-three-fold conjunction. -/
theorem bytes_mu {env : Environment (F p)} {input : Var Row (F p)} {offset : ℕ}
    (g : ((main input).operations offset).ChannelGuarantees Bitwise.AreBytes.toRaw env) :
    (Expression.eval env input.muQ).val < 256 ∧ (Expression.eval env input.muR).val < 256 := by
  simp only [circuit_norm, main, isBit, sign, neg, areBytes, Bitwise.AreBytes,
    Bitwise.areBytesTable, StaticLookupChannel.toChannel, forall_eq_or_imp, forall_eq] at g
  exact g.1

/-- The quotient's sign is a bit once `overflow` is one, which is what `ZERO` says. -/
lemma isBool_signQ {signed overflow signQ : F p} (hb : IsBool signed) (ho : IsBool overflow)
    (h : signed * (1 - overflow) - signQ = 0) : IsBool signQ := by
  have hq : signQ = signed * (1 - overflow) := by linear_combination -h
  rcases hb with rfl | rfl <;> rcases ho with rfl | rfl <;> rw [hq] <;> simp [IsBool]

/-- `overflow` is a bit, which the chip never says: it is the answer `ZERO` returned, and the
answer only arrives when the pull is active, which is what `hmu` carries. The `obtain` reads the
thirty-three pulls in the order `main` emits them, where the overflow check is the fifteenth. -/
theorem isBool_overflow {env : Environment (F p)} {input : Var Row (F p)} {offset : ℕ}
    (hmu : Channel.CountUpTo muMax
      (Expression.eval env input.muQ + Expression.eval env input.muR))
    (g : ((main input).operations offset).ChannelGuarantees ZeroC env) :
    IsBool (Expression.eval env input.overflow) := by
  simp only [circuit_norm, main, isBit, sign, neg, ZeroC, Bitwise.Zero,
    StaticLookupChannel.toChannel, forall_eq_or_imp, forall_eq] at g
  obtain ⟨-,-,-,-,-,-,-,-,-,-,-,-,-,-,gov,-,-,-,-,-,-,-,-,-,-,-,-,-,-,-,-,-,-⟩ := g
  exact Bitwise.isBool_of_zero (gov hmu)

/-- A sign flag is a bit: zero on an unsigned row by the chip's own constraint, and otherwise the
top bit `MSB16` returned. -/
lemma isBool_sign {signed x s : F p} (hb : IsBool signed) (hz : (1 - signed) * s = 0)
    (g : Channel.CountUpTo muMax signed → Bitwise.msb16Table.Guarantees (x, s)) : IsBool s := by
  rcases hb with rfl | rfl
  · left; simpa using hz
  · exact Bitwise.isBool_of_msb16 (g countUpTo_one)

/-- The three sign flags are bits. The chip does not say so on its own: `SIGN` reads the top bit
of a half off `MSB16` and only forces the flag to zero on an unsigned row, so on a signed row the
bit comes from the bus. An ensemble has this once `MSB16` is finished, and it is what makes the
`ZERO` pulls that `NEG` inlines legal on a counted channel: their multiplicity is a sign flag,
not the row's `mu`, so `ARE_BYTES` says nothing about them.

The two `obtain`s read the conjunctions `circuit_norm` leaves off `main`, in the order `main`
writes them: the thirty-three pulls for the guarantee (the two `ALU` pushes drop out, a pull
guarantee being vacuous on a push), and the nineteen `assertZero`s for the constraints, where the
first is `isBit signed` and the three `SIGN` templates are the thirteenth to fifteenth. -/
theorem isBool_signs {env : Environment (F p)} {input : Var Row (F p)} {offset : ℕ}
    (hc : ((main input).operations offset).ConstraintsHold env)
    (g : ((main input).operations offset).ChannelGuarantees Msb16C env) :
    IsBool (Expression.eval env input.signN) ∧ IsBool (Expression.eval env input.signR) ∧
      IsBool (Expression.eval env input.signD) := by
  simp only [circuit_norm, main, isBit, sign, neg, Msb16C, Bitwise.Msb16,
    StaticLookupChannel.toChannel, forall_eq_or_imp, forall_eq] at hc g
  obtain ⟨-,-,-,-,-,-,-,-,-,-,-,-,-,-,-,-,-,-,-,-,-,-,-,-,-,-,-,-,-,gN,gR,gD,-⟩ := g
  obtain ⟨hs,-,-,-,-,-,-,-,-,-,-,-,hzN,hzR,hzD,-,-,-,-⟩ := hc
  have hb : IsBool (Expression.eval env input.signed) := mul_one_sub_eq_zero_iff.mp hs
  exact ⟨isBool_sign hb hzN gN, isBool_sign hb hzR gR, isBool_sign hb hzD gD⟩

def circuit : GeneralFormalCircuit (F p) Row unit where
  name := "DVRM"
  main
  elaborated
  Spec r _ _ := RowSpec r
  channelsWithRequirements :=
    [AluC, IsHalfC, ZeroC]

  requirementsChannelsLawful := requirementsChannelsLawful
  soundness := soundness
  ProverAssumptions r _ _ := Honest r ∨ Padding r
  completeness := completeness

end LambdaVM.DVRM
