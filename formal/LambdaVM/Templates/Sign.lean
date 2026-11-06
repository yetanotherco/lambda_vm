module

public import LambdaVM.Tables.Bitwise
public import LambdaVM.Templates.IsBit
public import LambdaVM.Mu
public import Clean.Air.CountedChannel

@[expose] public section

/-!
# SIGN (spec/src/sign.toml)

`SIGN<sign; X, signed>`: when `signed = 1`, `sign` is the top bit of the halfword `X`, read
from one `MSB16[X] = sign` lookup with multiplicity `signed`; when `signed = 0`, the second
constraint `(1 − signed)·sign = 0` forces `sign = 0`.

`IS_BIT<signed>` is an assumption of the template, not one of its constraints, so a chip that
inlines `sign` constrains the flag itself. `SignCheck.circuit` adds that constraint to pin the
meaning down on its own, the way `IsByteCheck` does.

`Tables/Shift.lean` does not use this template: `shift.toml` declares its `is_negative` column
as a direct `MSB16` interaction, not as `SIGN<>`. The chips that do use it are `dvrm.toml`,
`cpu32.toml` and `mul.toml`.
-/

namespace LambdaVM

open Bitwise (Msb16 msb16Table)

/-- `SIGN<s; x, signed>`: `s` is the top bit of the half `x` when `signed ≠ 0`, and `0`
otherwise.

The pull goes to `Msb16.counted muMax` rather than `Msb16.toRaw` even though `signed` is a bit
and `toRaw` would take it. LT pulls the same bus with multiplicity `μ`, which reaches `muMax`,
so it has to use the counted channel; two spellings would be two `RawChannel`s named `"MSB16"`,
which `unique_channel_names` rejects, and no ensemble could hold LT together with MUL or DVRM.
It is the same agreement the model makes on `Alu.channelC` and on `ZERO`, and like those it is
invisible in the AIR. -/
@[circuit_norm]
def sign (signed x s : Expression (F p)) : Circuit (F p) Unit := do
  Msb16.pullUpTo muMax signed (x, s)
  assertZero ((1 - signed) * s)

-- Circuit inference does not unfold user wrappers; bridge it to its body.
instance (signed x s : Expression (F p)) : ExplicitCircuit (sign signed x s) :=
  .of_eq (circuit' := do Msb16.pullUpTo muMax signed (x, s); assertZero ((1 - signed) * s)) rfl
    (by infer_explicit_circuit)

/-! The template as a stand-alone assertion, to pin down its meaning. -/

namespace SignCheck

structure Inputs (F : Type) where
  signed : F
  x : F
  s : F
deriving ProvableStruct

def circuit : FormalAssertion (F p) Inputs where
  main input := do
    -- IS_BIT<1, signed>: in a chip this is one of the chip's own flag constraints
    isBit input.signed
    sign input.signed input.x input.s

  Assumptions := fun { signed, .. } => IsBool signed

  Spec := fun { signed, x, s } =>
    (signed ≠ 0 → x.val < 2^16 ∧ s.val = x.val / 2^15) ∧ (signed = 0 → s = 0)

  -- the conditional pull is lawful because the circuit constrains `signed` to be a bit
  requirementsChannelsLawful input_var i₀ := by
    obtain ⟨signed, x, s⟩ := input_var
    simp only [circuit_norm, sign, isBit, msb16Table]
    intro env hc
    exact eq_zero_or_countUpTo (mul_one_sub_eq_zero_iff.mp hc.1)

  soundness := by
    circuit_proof_start [sign, isBit, msb16Table]
    obtain ⟨h_bool, h_pull, h_unsigned⟩ := h_holds
    rcases mul_one_sub_eq_zero_iff.mp h_bool with rfl | rfl <;> simp_all
    -- an active row: the pull carries a count of one, which is both what unlocks the lookup's
    -- guarantee and what the counted channel asks back
    exact ⟨h_pull countUpTo_one, countUpTo_one⟩

  completeness := by
    circuit_proof_start [sign, isBit, msb16Table]
    rcases h_assumptions with rfl | rfl <;> simp_all
    -- an inactive row owes nothing: a zero multiplicity is not a count
    exact fun h => absurd h not_countUpTo_zero

end SignCheck

end LambdaVM
