
-- Experimental lean backend for Hax
-- The Hax prelude library can be found in hax/proof-libs/lean
import Hax
import CoreModelsSupplement
import Std.Tactic.Do
import Std.Do.Triple
import Std.Tactic.Do.Syntax
open Std.Do
open Std.Tactic

set_option mvcgen.warning false
set_option linter.unusedVariables false


namespace math.errors

inductive ByteConversionError : Type
| FromBEBytesError : ByteConversionError
| FromLEBytesError : ByteConversionError
| ValueNotReduced : ByteConversionError

@[spec]
def ByteConversionError_cast_to_repr (x : ByteConversionError) :
    RustM isize := do
  match x with
    | (ByteConversionError.FromBEBytesError ) => do (pure (0 : isize))
    | (ByteConversionError.FromLEBytesError ) => do (pure (1 : isize))
    | (ByteConversionError.ValueNotReduced ) => do (pure (2 : isize))

@[instance] opaque Impl_1.AssociatedTypes :
  core_models.fmt.Debug.AssociatedTypes ByteConversionError :=
  by constructor <;> exact Inhabited.default

@[instance] opaque Impl_1 :
  core_models.fmt.Debug ByteConversionError :=
  by constructor <;> exact Inhabited.default

@[instance] opaque Impl_2.AssociatedTypes :
  core_models.marker.StructuralPartialEq.AssociatedTypes ByteConversionError :=
  by constructor <;> exact Inhabited.default

@[instance] opaque Impl_2 :
  core_models.marker.StructuralPartialEq ByteConversionError :=
  by constructor <;> exact Inhabited.default

@[instance] opaque Impl_3.AssociatedTypes :
  core_models.cmp.PartialEq.AssociatedTypes
  ByteConversionError
  ByteConversionError :=
  by constructor <;> exact Inhabited.default

@[instance] opaque Impl_3 :
  core_models.cmp.PartialEq ByteConversionError ByteConversionError :=
  by constructor <;> exact Inhabited.default

@[instance] opaque Impl_4.AssociatedTypes :
  core_models.cmp.Eq.AssociatedTypes ByteConversionError :=
  by constructor <;> exact Inhabited.default

@[instance] opaque Impl_4 :
  core_models.cmp.Eq ByteConversionError :=
  by constructor <;> exact Inhabited.default

inductive CreationError : Type
| InvalidHexString : CreationError
| HexStringIsTooBig : CreationError
| CanonicalOutOfRange : CreationError
| EmptyString : CreationError

@[spec]
def CreationError_cast_to_repr (x : CreationError) : RustM isize := do
  match x with
    | (CreationError.InvalidHexString ) => do (pure (0 : isize))
    | (CreationError.HexStringIsTooBig ) => do (pure (1 : isize))
    | (CreationError.CanonicalOutOfRange ) => do (pure (2 : isize))
    | (CreationError.EmptyString ) => do (pure (3 : isize))

@[instance] opaque Impl_5.AssociatedTypes :
  core_models.fmt.Debug.AssociatedTypes CreationError :=
  by constructor <;> exact Inhabited.default

@[instance] opaque Impl_5 :
  core_models.fmt.Debug CreationError :=
  by constructor <;> exact Inhabited.default

@[instance] opaque Impl_6.AssociatedTypes :
  core_models.marker.StructuralPartialEq.AssociatedTypes CreationError :=
  by constructor <;> exact Inhabited.default

@[instance] opaque Impl_6 :
  core_models.marker.StructuralPartialEq CreationError :=
  by constructor <;> exact Inhabited.default

@[instance] opaque Impl_7.AssociatedTypes :
  core_models.cmp.PartialEq.AssociatedTypes CreationError CreationError :=
  by constructor <;> exact Inhabited.default

@[instance] opaque Impl_7 :
  core_models.cmp.PartialEq CreationError CreationError :=
  by constructor <;> exact Inhabited.default

@[instance] opaque Impl_8.AssociatedTypes :
  core_models.cmp.Eq.AssociatedTypes CreationError :=
  by constructor <;> exact Inhabited.default

@[instance] opaque Impl_8 :
  core_models.cmp.Eq CreationError :=
  by constructor <;> exact Inhabited.default

inductive DeserializationError : Type
| InvalidAmountOfBytes : DeserializationError
| FieldFromBytesError : DeserializationError
| PointerSizeError : DeserializationError
| InvalidValue : DeserializationError

@[spec]
def DeserializationError_cast_to_repr (x : DeserializationError) :
    RustM isize := do
  match x with
    | (DeserializationError.InvalidAmountOfBytes ) => do (pure (0 : isize))
    | (DeserializationError.FieldFromBytesError ) => do (pure (1 : isize))
    | (DeserializationError.PointerSizeError ) => do (pure (2 : isize))
    | (DeserializationError.InvalidValue ) => do (pure (3 : isize))

@[instance] opaque Impl_9.AssociatedTypes :
  core_models.fmt.Debug.AssociatedTypes DeserializationError :=
  by constructor <;> exact Inhabited.default

@[instance] opaque Impl_9 :
  core_models.fmt.Debug DeserializationError :=
  by constructor <;> exact Inhabited.default

@[instance] opaque Impl_10.AssociatedTypes :
  core_models.marker.StructuralPartialEq.AssociatedTypes DeserializationError :=
  by constructor <;> exact Inhabited.default

@[instance] opaque Impl_10 :
  core_models.marker.StructuralPartialEq DeserializationError :=
  by constructor <;> exact Inhabited.default

@[instance] opaque Impl_11.AssociatedTypes :
  core_models.cmp.PartialEq.AssociatedTypes
  DeserializationError
  DeserializationError :=
  by constructor <;> exact Inhabited.default

@[instance] opaque Impl_11 :
  core_models.cmp.PartialEq DeserializationError DeserializationError :=
  by constructor <;> exact Inhabited.default

@[instance] opaque Impl_12.AssociatedTypes :
  core_models.cmp.Eq.AssociatedTypes DeserializationError :=
  by constructor <;> exact Inhabited.default

@[instance] opaque Impl_12 :
  core_models.cmp.Eq DeserializationError :=
  by constructor <;> exact Inhabited.default

@[spec]
def Impl.from_hoisted (error : ByteConversionError) :
    RustM DeserializationError := do
  match error with
    | (ByteConversionError.FromBEBytesError ) => do
      (pure DeserializationError.FieldFromBytesError)
    | (ByteConversionError.FromLEBytesError ) => do
      (pure DeserializationError.FieldFromBytesError)
    | _ => do (pure DeserializationError.InvalidValue)

@[reducible] instance Impl.AssociatedTypes :
  core_models.convert.From.AssociatedTypes
  DeserializationError
  ByteConversionError
  where

instance Impl :
  core_models.convert.From DeserializationError ByteConversionError
  where
  _from := (Impl.from_hoisted)

end math.errors


namespace math.field.errors

inductive FieldError : Type
| DivisionByZero : FieldError
| --  Returns order of the calculated root of unity
    RootOfUnityError : u64 -> FieldError
| --  Can't calculate inverse of zero
    InvZeroError : FieldError

@[instance] opaque Impl.AssociatedTypes :
  core_models.fmt.Debug.AssociatedTypes FieldError :=
  by constructor <;> exact Inhabited.default

@[instance] opaque Impl :
  core_models.fmt.Debug FieldError :=
  by constructor <;> exact Inhabited.default

end math.field.errors


namespace math.field.extensions_goldilocks

def Impl.BYTE_LEN_hoisted : usize := (16 : usize)

def Impl_1.BYTE_LEN_hoisted : usize := (24 : usize)

def Impl_1.from_bytes_be.N : usize := (8 : usize)

def Impl_1.from_bytes_le.N : usize := (8 : usize)

--  Degree 2 extension field of Goldilocks
structure Degree2GoldilocksExtensionField where
  -- no fields

@[instance] opaque Impl_12.AssociatedTypes :
  core_models.clone.Clone.AssociatedTypes Degree2GoldilocksExtensionField :=
  by constructor <;> exact Inhabited.default

@[instance] opaque Impl_12 :
  core_models.clone.Clone Degree2GoldilocksExtensionField :=
  by constructor <;> exact Inhabited.default

@[instance] opaque Impl_11.AssociatedTypes :
  core_models.marker.Copy.AssociatedTypes Degree2GoldilocksExtensionField :=
  by constructor <;> exact Inhabited.default

@[instance] opaque Impl_11 :
  core_models.marker.Copy Degree2GoldilocksExtensionField :=
  by constructor <;> exact Inhabited.default

@[instance] opaque Impl_13.AssociatedTypes :
  core_models.fmt.Debug.AssociatedTypes Degree2GoldilocksExtensionField :=
  by constructor <;> exact Inhabited.default

@[instance] opaque Impl_13 :
  core_models.fmt.Debug Degree2GoldilocksExtensionField :=
  by constructor <;> exact Inhabited.default

--  Degree 3 extension field of Goldilocks
structure Degree3GoldilocksExtensionField where
  -- no fields

@[instance] opaque Impl_15.AssociatedTypes :
  core_models.clone.Clone.AssociatedTypes Degree3GoldilocksExtensionField :=
  by constructor <;> exact Inhabited.default

@[instance] opaque Impl_15 :
  core_models.clone.Clone Degree3GoldilocksExtensionField :=
  by constructor <;> exact Inhabited.default

@[instance] opaque Impl_14.AssociatedTypes :
  core_models.marker.Copy.AssociatedTypes Degree3GoldilocksExtensionField :=
  by constructor <;> exact Inhabited.default

@[instance] opaque Impl_14 :
  core_models.marker.Copy Degree3GoldilocksExtensionField :=
  by constructor <;> exact Inhabited.default

@[instance] opaque Impl_16.AssociatedTypes :
  core_models.fmt.Debug.AssociatedTypes Degree3GoldilocksExtensionField :=
  by constructor <;> exact Inhabited.default

@[instance] opaque Impl_16 :
  core_models.fmt.Debug Degree3GoldilocksExtensionField :=
  by constructor <;> exact Inhabited.default

def Impl_8.BYTE_LEN_hoisted : usize := (24 : usize)

def Impl_8.from_bytes_be.BYTES_PER_FIELD : usize := (8 : usize)

def Impl_8.from_bytes_le.BYTES_PER_FIELD : usize := (8 : usize)

end math.field.extensions_goldilocks


namespace math.field.goldilocks

--  Inform the compiler that a condition is always true.
-- 
--  # Safety
--  The caller must guarantee that `p` is true.
@[spec]
def assume (p : Bool) : RustM rust_primitives.hax.Tuple0 := do
  let _ ←
    if true then do
      let _ ← (hax_lib.assert p);
      (pure rust_primitives.hax.Tuple0.mk)
    else do
      (pure rust_primitives.hax.Tuple0.mk);
  if (← (!? p)) then do
    (rust_primitives.hax.never_to_any
      (← (core_models.hint.unreachable_unchecked
        rust_primitives.hax.Tuple0.mk)))
  else do
    (pure rust_primitives.hax.Tuple0.mk)

--  The Goldilocks prime: p = 2^64 - 2^32 + 1
def GOLDILOCKS_PRIME : u64 := (18446744069414584321 : u64)

--  EPSILON = 2^32 - 1 = p - 2^64 (i.e., -2^64 mod p)
--  This is the key constant for fast reduction.
def EPSILON : u64 := (4294967295 : u64)

--  Native Goldilocks field using direct u64 representation.
-- 
--  Values are stored as u64 in the range [0, 2^64), not necessarily canonical.
--  Canonicalization to [0, p) happens only when needed (comparison, serialization).
structure GoldilocksField where
  -- no fields

@[instance] opaque Impl_7.AssociatedTypes :
  core_models.fmt.Debug.AssociatedTypes GoldilocksField :=
  by constructor <;> exact Inhabited.default

@[instance] opaque Impl_7 :
  core_models.fmt.Debug GoldilocksField :=
  by constructor <;> exact Inhabited.default

@[instance] opaque Impl_8.AssociatedTypes :
  core_models.clone.Clone.AssociatedTypes GoldilocksField :=
  by constructor <;> exact Inhabited.default

@[instance] opaque Impl_8 :
  core_models.clone.Clone GoldilocksField :=
  by constructor <;> exact Inhabited.default

@[instance] opaque Impl_9.AssociatedTypes :
  core_models.marker.Copy.AssociatedTypes GoldilocksField :=
  by constructor <;> exact Inhabited.default

@[instance] opaque Impl_9 :
  core_models.marker.Copy GoldilocksField :=
  by constructor <;> exact Inhabited.default

@[instance] opaque Impl_10.AssociatedTypes :
  core_models.marker.StructuralPartialEq.AssociatedTypes GoldilocksField :=
  by constructor <;> exact Inhabited.default

@[instance] opaque Impl_10 :
  core_models.marker.StructuralPartialEq GoldilocksField :=
  by constructor <;> exact Inhabited.default

@[instance] opaque Impl_11.AssociatedTypes :
  core_models.cmp.PartialEq.AssociatedTypes GoldilocksField GoldilocksField :=
  by constructor <;> exact Inhabited.default

@[instance] opaque Impl_11 :
  core_models.cmp.PartialEq GoldilocksField GoldilocksField :=
  by constructor <;> exact Inhabited.default

@[instance] opaque Impl_12.AssociatedTypes :
  core_models.cmp.Eq.AssociatedTypes GoldilocksField :=
  by constructor <;> exact Inhabited.default

@[instance] opaque Impl_12 :
  core_models.cmp.Eq GoldilocksField :=
  by constructor <;> exact Inhabited.default

@[instance] opaque Impl_13.AssociatedTypes :
  core_models.hash.Hash.AssociatedTypes GoldilocksField :=
  by constructor <;> exact Inhabited.default

@[instance] opaque Impl_13 :
  core_models.hash.Hash GoldilocksField :=
  by constructor <;> exact Inhabited.default

@[instance] opaque Impl_14.AssociatedTypes :
  core_models.default.Default.AssociatedTypes GoldilocksField :=
  by constructor <;> exact Inhabited.default

@[instance] opaque Impl_14 :
  core_models.default.Default GoldilocksField :=
  by constructor <;> exact Inhabited.default

--  Addition with branch hint for rare double-overflow.
--  Compiles to a 3-instruction common path (add + csel + adds on ARM)
--  with a predicted-not-taken branch for the exceedingly rare double overflow.
@[spec]
def Impl.add_hoisted (a : u64) (b : u64) : RustM u64 := do
  let ⟨sum, over⟩ ← (core_models.num.Impl_9.overflowing_add a b);
  let ⟨sum, over⟩ ←
    (core_models.num.Impl_9.overflowing_add
      sum
      (← ((← (rust_primitives.hax.cast_op over : RustM u64)) *? EPSILON)));
  let sum : u64 ←
    if over then do
      let _ ←
        (assume
          (← ((← (a >? GOLDILOCKS_PRIME)) &&? (← (b >? GOLDILOCKS_PRIME)))));
      let _ := rust_primitives.hax.Tuple0.mk;
      let _ ← (core_models.hint.cold_path rust_primitives.hax.Tuple0.mk);
      let sum : u64 ← (sum +? EPSILON);
      (pure sum)
    else do
      (pure sum);
  (pure sum)

--  Subtraction with branch hint for rare double-underflow.
@[spec]
def Impl.sub_hoisted (a : u64) (b : u64) : RustM u64 := do
  let ⟨diff, under⟩ ← (core_models.num.Impl_9.overflowing_sub a b);
  let ⟨diff, under⟩ ←
    (core_models.num.Impl_9.overflowing_sub
      diff
      (← ((← (rust_primitives.hax.cast_op under : RustM u64)) *? EPSILON)));
  let diff : u64 ←
    if under then do
      let _ ←
        (assume
          (← ((← (a <? (← (EPSILON -? (1 : u64)))))
            &&? (← (b >? GOLDILOCKS_PRIME)))));
      let _ := rust_primitives.hax.Tuple0.mk;
      let _ ← (core_models.hint.cold_path rust_primitives.hax.Tuple0.mk);
      let diff : u64 ← (diff -? EPSILON);
      (pure diff)
    else do
      (pure diff);
  (pure diff)

@[spec]
def Impl.zero_hoisted (_ : rust_primitives.hax.Tuple0) : RustM u64 := do
  (pure (0 : u64))

@[spec]
def Impl.one_hoisted (_ : rust_primitives.hax.Tuple0) : RustM u64 := do
  (pure (1 : u64))

@[spec]
def Impl.from_u64_hoisted (x : u64) : RustM u64 := do
  if (← (x >=? GOLDILOCKS_PRIME)) then do
    (x -? GOLDILOCKS_PRIME)
  else do
    (pure x)

@[spec]
def Impl.from_base_type_hoisted (x : u64) : RustM u64 := do (pure x)

@[spec]
def Impl.double_hoisted (a : u64) : RustM u64 := do (Impl.add_hoisted a a)

@[spec]
def add_no_canonicalize_trashing_input (x : u64) (y : u64) : RustM u64 := do
  let ⟨res_wrapped, carry⟩ ← (core_models.num.Impl_9.overflowing_add x y);
  (core_models.num.Impl_9.wrapping_add
    res_wrapped
    (← (EPSILON *? (← (rust_primitives.hax.cast_op carry : RustM u64)))))

--  Reduce a 128-bit value to a 64-bit Goldilocks field element.
-- 
--  Uses the identities: 2^64 ≡ 2^32 - 1 (mod p), 2^96 ≡ -1 (mod p).
--  Branch hints mark rare borrow/carry paths for better branch prediction.
@[spec]
def reduce128 (x : u128) : RustM u64 := do
  let ⟨x_lo, x_hi⟩ :=
    (rust_primitives.hax.Tuple2.mk
      (← (rust_primitives.hax.cast_op x : RustM u64))
      (← (rust_primitives.hax.cast_op (← (x >>>? (64 : i32))) : RustM u64)));
  let x_hi_hi : u64 ← (x_hi >>>? (32 : i32));
  let x_hi_lo : u64 ← (x_hi &&&? EPSILON);
  let ⟨t0, borrow⟩ ← (core_models.num.Impl_9.overflowing_sub x_lo x_hi_hi);
  let t0 : u64 ←
    if borrow then do
      let _ ← (core_models.hint.cold_path rust_primitives.hax.Tuple0.mk);
      let t0 : u64 ← (t0 -? EPSILON);
      (pure t0)
    else do
      (pure t0);
  let t1 : u64 ←
    (core_models.num.Impl_9.wrapping_sub (← (x_hi_lo <<<? (32 : i32))) x_hi_lo);
  (add_no_canonicalize_trashing_input t0 t1)

--  Multiplication using 128-bit intermediate and fast reduction.
--  LLVM generates optimal MUL+UMULH code on ARM64.
@[spec]
def Impl.mul_hoisted (a : u64) (b : u64) : RustM u64 := do
  let product : u128 ←
    ((← (rust_primitives.hax.cast_op a : RustM u128))
      *? (← (rust_primitives.hax.cast_op b : RustM u128)));
  (reduce128 product)

--  Squaring using 128-bit intermediate and fast reduction.
@[spec]
def Impl.square_hoisted (a : u64) : RustM u64 := do
  let a_val : u64 := a;
  let product : u128 ←
    ((← (rust_primitives.hax.cast_op a_val : RustM u128))
      *? (← (rust_primitives.hax.cast_op a_val : RustM u128)));
  (reduce128 product)

def dot_product_2.EPSILON_SQ : u64 :=
  RustM.of_isOk
    (do (core_models.num.Impl_9.wrapping_mul EPSILON EPSILON))
    (by rfl)

--  Compute a0*b0 + a1*b1 mod p in a single reduction pass.
-- 
--  Instead of reducing each product separately (2 reduce128 calls),
--  this sums the u128 products and reduces once. When the sum overflows u128,
--  we correct by adding 2^128 mod p = EPSILON^2 = (2^32 - 1)^2.
-- 
--  This is the critical building block for extension field multiplication:
--  each Fp2 mul needs two dot products instead of three separate mul+reduce.
@[spec]
def dot_product_2 (a0 : u64) (b0 : u64) (a1 : u64) (b1 : u64) : RustM u64 := do
  let prod0 : u128 ←
    ((← (rust_primitives.hax.cast_op a0 : RustM u128))
      *? (← (rust_primitives.hax.cast_op b0 : RustM u128)));
  let prod1 : u128 ←
    ((← (rust_primitives.hax.cast_op a1 : RustM u128))
      *? (← (rust_primitives.hax.cast_op b1 : RustM u128)));
  let ⟨sum, overflow⟩ ← (core_models.num.Impl_10.overflowing_add prod0 prod1);
  let reduced : u64 ← (reduce128 sum);
  if overflow then do
    let _ ← (core_models.hint.cold_path rust_primitives.hax.Tuple0.mk);
    (add_no_canonicalize_trashing_input reduced dot_product_2.EPSILON_SQ)
  else do
    (pure reduced)

def dot_product_3.EPSILON_SQ : u64 :=
  RustM.of_isOk
    (do (core_models.num.Impl_9.wrapping_mul EPSILON EPSILON))
    (by rfl)

--  Compute a0*b0 + a1*b1 + a2*b2 mod p in a single reduction pass.
-- 
--  Accumulates three u128 products, tracking overflow count (at most 2).
--  Each overflow adds 2^128 mod p = EPSILON^2 to the result.
--  This is the critical building block for Fp3 multiplication (the extension
--  field used by the VM's STARK prover).
@[spec]
def dot_product_3
    (a0 : u64)
    (b0 : u64)
    (a1 : u64)
    (b1 : u64)
    (a2 : u64)
    (b2 : u64) :
    RustM u64 := do
  let prod0 : u128 ←
    ((← (rust_primitives.hax.cast_op a0 : RustM u128))
      *? (← (rust_primitives.hax.cast_op b0 : RustM u128)));
  let prod1 : u128 ←
    ((← (rust_primitives.hax.cast_op a1 : RustM u128))
      *? (← (rust_primitives.hax.cast_op b1 : RustM u128)));
  let prod2 : u128 ←
    ((← (rust_primitives.hax.cast_op a2 : RustM u128))
      *? (← (rust_primitives.hax.cast_op b2 : RustM u128)));
  let ⟨sum01, over1⟩ ← (core_models.num.Impl_10.overflowing_add prod0 prod1);
  let ⟨sum012, over2⟩ ← (core_models.num.Impl_10.overflowing_add sum01 prod2);
  let overflow_count : u64 ←
    ((← (rust_primitives.hax.cast_op over1 : RustM u64))
      +? (← (rust_primitives.hax.cast_op over2 : RustM u64)));
  let reduced : u64 ← (reduce128 sum012);
  let reduced : u64 ←
    if (← (overflow_count >? (0 : u64))) then do
      let _ ← (core_models.hint.cold_path rust_primitives.hax.Tuple0.mk);
      let reduced : u64 ←
        (add_no_canonicalize_trashing_input reduced dot_product_3.EPSILON_SQ);
      if (← (overflow_count >? (1 : u64))) then do
        let _ ← (core_models.hint.cold_path rust_primitives.hax.Tuple0.mk);
        let reduced : u64 ←
          (add_no_canonicalize_trashing_input reduced dot_product_3.EPSILON_SQ);
        (pure reduced)
      else do
        (pure reduced)
    else do
      (pure reduced);
  (pure reduced)

def Impl_2.BYTE_LEN_hoisted : usize := (8 : usize)

@[spec]
def Impl_4.canonical_hoisted (a : u64) : RustM u64 := do
  if (← (a >=? GOLDILOCKS_PRIME)) then do
    (a -? GOLDILOCKS_PRIME)
  else do
    (pure a)

@[spec]
def Impl_4.from_hex_hoisted (hex_string : String) :
    RustM (core_models.result.Result u64 math.errors.CreationError) := do
  let hex : String ←
    (core_models.option.Impl.unwrap_or String
      (← (core_models.str.Impl.strip_prefix String hex_string "0x"))
      hex_string);
  (core_models.result.Impl.map_err
    u64
    core_models.num.error.ParseIntError
    math.errors.CreationError
    (core_models.num.error.ParseIntError -> RustM math.errors.CreationError)
    (← (core_models.result.Impl.map
      u64
      core_models.num.error.ParseIntError
      u64
      (u64 -> RustM u64)
      (← (core_models.num.Impl_40.from_str_radix hex (16 : u32)))
      math.field.traits.IsField.from_u64))
    (fun _ =>
      (do
      (pure math.errors.CreationError.InvalidHexString) :
      RustM math.errors.CreationError)))

@[spec]
def Impl_4.field_bit_size_hoisted (_ : rust_primitives.hax.Tuple0) :
    RustM usize := do
  (pure (64 : usize))

--  Two-adicity of Goldilocks: p - 1 = 2^32 * (2^32 - 1)
def Impl_5.TWO_ADICITY_hoisted : u64 := (32 : u64)

--  Primitive 2^32-th root of unity.
--  This is the same value used in Plonky3.
def Impl_5.TWO_ADIC_PRIMITVE_ROOT_OF_UNITY_hoisted : u64 :=
  (1753635133440165772 : u64)

@[spec]
def Impl_5.field_name_hoisted (_ : rust_primitives.hax.Tuple0) :
    RustM String := do
  (pure "Goldilocks")

end math.field.goldilocks


namespace math.field.traits

--  Represents different configurations that powers of roots of unity can be in. Some of these may
--  be necessary for FFT (as twiddle factors).
inductive RootsConfig : Type
| Natural : RootsConfig
| NaturalInversed : RootsConfig
| BitReverse : RootsConfig
| BitReverseInversed : RootsConfig

@[spec]
def RootsConfig_cast_to_repr (x : RootsConfig) : RustM isize := do
  match x with
    | (RootsConfig.Natural ) => do (pure (0 : isize))
    | (RootsConfig.NaturalInversed ) => do (pure (1 : isize))
    | (RootsConfig.BitReverse ) => do (pure (2 : isize))
    | (RootsConfig.BitReverseInversed ) => do (pure (3 : isize))

@[instance] opaque Impl_1.AssociatedTypes :
  core_models.clone.Clone.AssociatedTypes RootsConfig :=
  by constructor <;> exact Inhabited.default

@[instance] opaque Impl_1 :
  core_models.clone.Clone RootsConfig :=
  by constructor <;> exact Inhabited.default

@[instance] opaque Impl_2.AssociatedTypes :
  core_models.marker.Copy.AssociatedTypes RootsConfig :=
  by constructor <;> exact Inhabited.default

@[instance] opaque Impl_2 :
  core_models.marker.Copy RootsConfig :=
  by constructor <;> exact Inhabited.default

--  Provides the Legendre symbol for an element modulo p
--  The Legendre symbol is Zero if a is congruent to 0 modulo p
--  It is equal to One if a is a square modulo p (which means it has a square root)
--  It is equal to MinusOne if a is not a square modulo p
--  For example, p - 1 is not a square modulo p if p is congruent to 3 modulo 4
--  This applies to Mersenne primes, for example
inductive LegendreSymbol : Type
| MinusOne : LegendreSymbol
| Zero : LegendreSymbol
| One : LegendreSymbol

@[spec]
def LegendreSymbol_cast_to_repr (x : LegendreSymbol) : RustM isize := do
  match x with
    | (LegendreSymbol.MinusOne ) => do (pure (0 : isize))
    | (LegendreSymbol.Zero ) => do (pure (1 : isize))
    | (LegendreSymbol.One ) => do (pure (2 : isize))

@[instance] opaque Impl_3.AssociatedTypes :
  core_models.marker.StructuralPartialEq.AssociatedTypes LegendreSymbol :=
  by constructor <;> exact Inhabited.default

@[instance] opaque Impl_3 :
  core_models.marker.StructuralPartialEq LegendreSymbol :=
  by constructor <;> exact Inhabited.default

@[instance] opaque Impl_4.AssociatedTypes :
  core_models.cmp.PartialEq.AssociatedTypes LegendreSymbol LegendreSymbol :=
  by constructor <;> exact Inhabited.default

@[instance] opaque Impl_4 :
  core_models.cmp.PartialEq LegendreSymbol LegendreSymbol :=
  by constructor <;> exact Inhabited.default

end math.field.traits


namespace math.spill_safe

--  # Safety
--  Implementer asserts `Self`'s memory representation contains no padding,
--  every bit pattern is a valid value of `Self`, and `Self` carries no
--  indirection (heap pointers, references, etc.). Adding this `unsafe impl`
--  for a type that violates these invariants is UB at any byte cast.
class SpillSafe.AssociatedTypes (Self : Type) where
  [trait_constr_SpillSafe_i0 : core_models.marker.Copy.AssociatedTypes Self]

attribute [instance_reducible, instance]
  SpillSafe.AssociatedTypes.trait_constr_SpillSafe_i0

class SpillSafe (Self : Type)
  [associatedTypes : outParam (SpillSafe.AssociatedTypes (Self : Type))]
  where
  [trait_constr_SpillSafe_i0 : core_models.marker.Copy Self]

attribute [instance_reducible, instance] SpillSafe.trait_constr_SpillSafe_i0

@[reducible] instance Impl.AssociatedTypes : SpillSafe.AssociatedTypes u8 where

instance Impl : SpillSafe u8 where

@[reducible] instance Impl_1.AssociatedTypes :
  SpillSafe.AssociatedTypes u16
  where

instance Impl_1 : SpillSafe u16 where

@[reducible] instance Impl_2.AssociatedTypes :
  SpillSafe.AssociatedTypes u32
  where

instance Impl_2 : SpillSafe u32 where

@[reducible] instance Impl_3.AssociatedTypes :
  SpillSafe.AssociatedTypes u64
  where

instance Impl_3 : SpillSafe u64 where

@[reducible] instance Impl_4.AssociatedTypes :
  SpillSafe.AssociatedTypes u128
  where

instance Impl_4 : SpillSafe u128 where

@[reducible] instance Impl_5.AssociatedTypes :
  SpillSafe.AssociatedTypes i8
  where

instance Impl_5 : SpillSafe i8 where

@[reducible] instance Impl_6.AssociatedTypes :
  SpillSafe.AssociatedTypes i16
  where

instance Impl_6 : SpillSafe i16 where

@[reducible] instance Impl_7.AssociatedTypes :
  SpillSafe.AssociatedTypes i32
  where

instance Impl_7 : SpillSafe i32 where

@[reducible] instance Impl_8.AssociatedTypes :
  SpillSafe.AssociatedTypes i64
  where

instance Impl_8 : SpillSafe i64 where

@[reducible] instance Impl_9.AssociatedTypes :
  SpillSafe.AssociatedTypes i128
  where

instance Impl_9 : SpillSafe i128 where

@[reducible] instance Impl_10.AssociatedTypes
  (T : Type)
  (N : usize)
  [trait_constr_Impl_10_associated_type_i0 : SpillSafe.AssociatedTypes T]
  [trait_constr_Impl_10_i0 : SpillSafe T ] :
  SpillSafe.AssociatedTypes (RustArray T N)
  where

instance Impl_10
  (T : Type)
  (N : usize)
  [trait_constr_Impl_10_associated_type_i0 : SpillSafe.AssociatedTypes T]
  [trait_constr_Impl_10_i0 : SpillSafe T ] :
  SpillSafe (RustArray T N)
  where

end math.spill_safe


namespace math.traits

--  Serialize function without args
--  Used for serialization when formatting options are not relevant
class AsBytes.AssociatedTypes (Self : Type) where

class AsBytes (Self : Type)
  [associatedTypes : outParam (AsBytes.AssociatedTypes (Self : Type))]
  where
  as_bytes (Self) : (Self -> RustM (alloc.vec.Vec u8 alloc.alloc.Global))

@[spec]
def Impl.as_bytes_hoisted (self : u32) :
    RustM (alloc.vec.Vec u8 alloc.alloc.Global) := do
  (alloc.slice.Impl.to_vec u8
    (← (rust_primitives.unsize (← (core_models.num.Impl_8.to_le_bytes self)))))

@[reducible] instance Impl.AssociatedTypes : AsBytes.AssociatedTypes u32 where

instance Impl : AsBytes u32 where
  as_bytes := (Impl.as_bytes_hoisted)

@[spec]
def Impl_1.as_bytes_hoisted (self : u64) :
    RustM (alloc.vec.Vec u8 alloc.alloc.Global) := do
  (alloc.slice.Impl.to_vec u8
    (← (rust_primitives.unsize (← (core_models.num.Impl_9.to_le_bytes self)))))

@[reducible] instance Impl_1.AssociatedTypes : AsBytes.AssociatedTypes u64 where

instance Impl_1 : AsBytes u64 where
  as_bytes := (Impl_1.as_bytes_hoisted)

def Impl_2.BYTE_LEN_hoisted : usize := (8 : usize)

@[spec]
def Impl_2.to_bytes_be_hoisted (self : u64) :
    RustM (alloc.vec.Vec u8 alloc.alloc.Global) := do
  (alloc.slice.Impl.to_vec u8
    (← (rust_primitives.unsize (← (core_models.num.Impl_9.to_be_bytes self)))))

@[spec]
def Impl_2.to_bytes_le_hoisted (self : u64) :
    RustM (alloc.vec.Vec u8 alloc.alloc.Global) := do
  (alloc.slice.Impl.to_vec u8
    (← (rust_primitives.unsize (← (core_models.num.Impl_9.to_le_bytes self)))))

@[spec]
def Impl_2.from_bytes_be_hoisted (bytes : (RustSlice u8)) :
    RustM (core_models.result.Result u64 math.errors.ByteConversionError) := do
  match
    (← (core_models.option.Impl.ok_or
      (RustSlice u8)
      math.errors.ByteConversionError
      (← (core_models.slice.Impl.get u8 (core_models.ops.range.Range usize)
        bytes
        (core_models.ops.range.Range.mk
          (start := (0 : usize))
          (_end := (8 : usize)))))
      math.errors.ByteConversionError.FromBEBytesError))
  with
    | (core_models.result.Result.Ok  needed_bytes) => do
      match
        (← (core_models.result.Impl.map_err
          (RustArray u8 8)
          core_models.array.TryFromSliceError
          math.errors.ByteConversionError
          (core_models.array.TryFromSliceError ->
          RustM math.errors.ByteConversionError)
          (← (core_models.convert.TryInto.try_into
            (RustSlice u8)
            (RustArray u8 8) needed_bytes))
          (fun _ =>
            (do
            (pure math.errors.ByteConversionError.FromBEBytesError) :
            RustM math.errors.ByteConversionError))))
      with
        | (core_models.result.Result.Ok  hoist15) => do
          (pure (core_models.result.Result.Ok
            (← (core_models.num.Impl_9.from_be_bytes hoist15))))
        | (core_models.result.Result.Err  err) => do
          (pure (core_models.result.Result.Err err))
    | (core_models.result.Result.Err  err) => do
      (pure (core_models.result.Result.Err err))

@[spec]
def Impl_2.from_bytes_le_hoisted (bytes : (RustSlice u8)) :
    RustM (core_models.result.Result u64 math.errors.ByteConversionError) := do
  match
    (← (core_models.option.Impl.ok_or
      (RustSlice u8)
      math.errors.ByteConversionError
      (← (core_models.slice.Impl.get u8 (core_models.ops.range.Range usize)
        bytes
        (core_models.ops.range.Range.mk
          (start := (0 : usize))
          (_end := (8 : usize)))))
      math.errors.ByteConversionError.FromLEBytesError))
  with
    | (core_models.result.Result.Ok  needed_bytes) => do
      match
        (← (core_models.result.Impl.map_err
          (RustArray u8 8)
          core_models.array.TryFromSliceError
          math.errors.ByteConversionError
          (core_models.array.TryFromSliceError ->
          RustM math.errors.ByteConversionError)
          (← (core_models.convert.TryInto.try_into
            (RustSlice u8)
            (RustArray u8 8) needed_bytes))
          (fun _ =>
            (do
            (pure math.errors.ByteConversionError.FromLEBytesError) :
            RustM math.errors.ByteConversionError))))
      with
        | (core_models.result.Result.Ok  hoist17) => do
          (pure (core_models.result.Result.Ok
            (← (core_models.num.Impl_9.from_le_bytes hoist17))))
        | (core_models.result.Result.Err  err) => do
          (pure (core_models.result.Result.Err err))
    | (core_models.result.Result.Err  err) => do
      (pure (core_models.result.Result.Err err))

--  Deserialize function without args
class Deserializable.AssociatedTypes (Self : Type) where

class Deserializable (Self : Type)
  [associatedTypes : outParam (Deserializable.AssociatedTypes (Self : Type))]
  where
  deserialize (Self) :
    ((RustSlice u8) ->
    RustM (core_models.result.Result Self math.errors.DeserializationError))

end math.traits


namespace math.unsigned_integer.traits

sorry

@[reducible] instance Impl.AssociatedTypes :
  IsUnsignedInteger.AssociatedTypes u128
  where

instance Impl : IsUnsignedInteger u128 where

@[reducible] instance Impl_1.AssociatedTypes :
  IsUnsignedInteger.AssociatedTypes u64
  where

instance Impl_1 : IsUnsignedInteger u64 where

@[reducible] instance Impl_2.AssociatedTypes :
  IsUnsignedInteger.AssociatedTypes u32
  where

instance Impl_2 : IsUnsignedInteger u32 where

@[reducible] instance Impl_3.AssociatedTypes :
  IsUnsignedInteger.AssociatedTypes u16
  where

instance Impl_3 : IsUnsignedInteger u16 where

@[reducible] instance Impl_4.AssociatedTypes :
  IsUnsignedInteger.AssociatedTypes usize
  where

instance Impl_4 : IsUnsignedInteger usize where

end math.unsigned_integer.traits


namespace math.fft.bit_reversing

--  Reverses the `log2(size)` first bits of `i`
@[spec]
def reverse_index (i : usize) (size : u64) : RustM usize := do
  if (← (size ==? (1 : u64))) then do
    (pure i)
  else do
    ((← (core_models.num.Impl_11.reverse_bits i))
      >>>? (← (core_models.num.Impl_11.BITS
        -? (← (core_models.num.Impl_9.trailing_zeros size)))))

--  In-place bit-reverse permutation algorithm. Requires input length to be a power of two.
@[spec]
def in_place_bit_reverse_permute (E : Type) (input : (RustSlice E)) :
    RustM (RustSlice E) := do
  let input : (RustSlice E) ←
    (rust_primitives.hax.folds.fold_range
      (0 : usize)
      (← (core_models.slice.Impl.len E input))
      (fun input _ => (do (pure true) : RustM Bool))
      input
      (fun input i =>
        (do
        let bit_reversed_index : usize ←
          (reverse_index
            i
            (← (rust_primitives.hax.cast_op
              (← (core_models.slice.Impl.len E input)) :
              RustM u64)));
        if (← (bit_reversed_index >? i)) then do
          let input : (RustSlice E) ←
            (core_models.slice.Impl.swap E input i bit_reversed_index);
          (pure input)
        else do
          (pure input) :
        RustM (RustSlice E))));
  (pure input)

end math.fft.bit_reversing


namespace math.fft.bowers_fft

--  Maximum supported FFT order to prevent integer overflow.
--  With order 63, n = 2^63 which is the largest power of 2 that fits in usize on 64-bit.
--  For 32-bit systems, max order is 31.
def MAX_FFT_ORDER : u64 := (63 : u64)

end math.fft.bowers_fft


namespace math.fft.errors

inductive FFTError : Type
| RootOfUnityError : u64 -> FFTError
| InputError : usize -> FFTError
| OrderError : u64 -> FFTError
| DomainSizeError : usize -> FFTError
| --  A coset offset of zero was supplied; it has no multiplicative inverse.
    InvalidCosetOffset : FFTError

@[instance] opaque Impl_2.AssociatedTypes :
  core_models.fmt.Debug.AssociatedTypes FFTError :=
  by constructor <;> exact Inhabited.default

@[instance] opaque Impl_2 :
  core_models.fmt.Debug FFTError :=
  by constructor <;> exact Inhabited.default

@[spec]
def Impl.fmt_hoisted (self : FFTError) (f : core_models.fmt.Formatter) :
    RustM
    (rust_primitives.hax.Tuple2
      core_models.fmt.Formatter
      (core_models.result.Result
        rust_primitives.hax.Tuple0
        core_models.fmt.Error))
    := do
  let ⟨f, hax_temp_output⟩ ←
    match self with
      | (FFTError.RootOfUnityError  _) => do
        let ⟨tmp0, out⟩ ←
          (core_models.fmt.Impl_11.write_fmt
            f
            (← (core_models.fmt.rt.Impl_1.new_const ((1 : usize))
              (RustArray.ofVec #v["Could not calculate root of unity"]))));
        let f : core_models.fmt.Formatter := tmp0;
        (pure (rust_primitives.hax.Tuple2.mk f out))
      | (FFTError.InputError  v) => do
        let args : (rust_primitives.hax.Tuple1 usize) :=
          (rust_primitives.hax.Tuple1.mk v);
        let args : (RustArray core_models.fmt.rt.Argument 1) :=
          (RustArray.ofVec #v[(← (core_models.fmt.rt.Impl.new_display usize
                                  (rust_primitives.hax.Tuple1._0 args)))]);
        let ⟨tmp0, out⟩ ←
          (core_models.fmt.Impl_11.write_fmt
            f
            (← (core_models.fmt.rt.Impl_1.new_v1 ((2 : usize)) ((1 : usize))
              (RustArray.ofVec #v["Input length is ",
                                    ", which is not a power of two"])
              args)));
        let f : core_models.fmt.Formatter := tmp0;
        (pure (rust_primitives.hax.Tuple2.mk f out))
      | (FFTError.OrderError  v) => do
        let args : (rust_primitives.hax.Tuple1 u64) :=
          (rust_primitives.hax.Tuple1.mk v);
        let args : (RustArray core_models.fmt.rt.Argument 1) :=
          (RustArray.ofVec #v[(← (core_models.fmt.rt.Impl.new_display u64
                                  (rust_primitives.hax.Tuple1._0 args)))]);
        let ⟨tmp0, out⟩ ←
          (core_models.fmt.Impl_11.write_fmt
            f
            (← (core_models.fmt.rt.Impl_1.new_v1 ((1 : usize)) ((1 : usize))
              (RustArray.ofVec #v["Order should be less than or equal to 63, but is "])
              args)));
        let f : core_models.fmt.Formatter := tmp0;
        (pure (rust_primitives.hax.Tuple2.mk f out))
      | (FFTError.DomainSizeError  _) => do
        let ⟨tmp0, out⟩ ←
          (core_models.fmt.Impl_11.write_fmt
            f
            (← (core_models.fmt.rt.Impl_1.new_const ((1 : usize))
              (RustArray.ofVec #v["Domain size exceeds two adicity of the field"]))));
        let f : core_models.fmt.Formatter := tmp0;
        (pure (rust_primitives.hax.Tuple2.mk f out))
      | (FFTError.InvalidCosetOffset ) => do
        let ⟨tmp0, out⟩ ←
          (core_models.fmt.Impl_11.write_fmt
            f
            (← (core_models.fmt.rt.Impl_1.new_const ((1 : usize))
              (RustArray.ofVec #v["Coset offset is zero, which is not invertible"]))));
        let f : core_models.fmt.Formatter := tmp0;
        (pure (rust_primitives.hax.Tuple2.mk f out));
  (pure (rust_primitives.hax.Tuple2.mk f hax_temp_output))

@[reducible] instance Impl.AssociatedTypes :
  core_models.fmt.Display.AssociatedTypes FFTError
  where

instance Impl : core_models.fmt.Display FFTError where
  fmt := (Impl.fmt_hoisted)

@[spec]
def Impl_1.from_hoisted (error : math.field.errors.FieldError) :
    RustM FFTError := do
  match error with
    | (math.field.errors.FieldError.DivisionByZero ) => do
      (rust_primitives.hax.never_to_any
        (← (core_models.panicking.panic_fmt
          (← (core_models.fmt.rt.Impl_1.new_const ((1 : usize))
            (RustArray.ofVec #v["Can\'t divide by zero during FFT"]))))))
    | (math.field.errors.FieldError.InvZeroError ) => do
      (rust_primitives.hax.never_to_any
        (← (core_models.panicking.panic_fmt
          (← (core_models.fmt.rt.Impl_1.new_const ((1 : usize))
            (RustArray.ofVec #v["Can\'t calculate inverse of zero during FFT"]))))))
    | (math.field.errors.FieldError.RootOfUnityError  order) => do
      (pure (FFTError.RootOfUnityError order))

@[reducible] instance Impl_1.AssociatedTypes :
  core_models.convert.From.AssociatedTypes FFTError math.field.errors.FieldError
  where

instance Impl_1 :
  core_models.convert.From FFTError math.field.errors.FieldError
  where
  _from := (Impl_1.from_hoisted)

end math.fft.errors


namespace math.polynomial

--  Represents the polynomial c_0 + c_1 * X + c_2 * X^2 + ... + c_n * X^n
--  as a vector of coefficients `[c_0, c_1, ... , c_n]`
structure Polynomial (FE : Type) where
  coefficients : (alloc.vec.Vec FE alloc.alloc.Global)

@[instance] opaque Impl_2.AssociatedTypes
  (FE : Type)
  [trait_constr_Impl_2_associated_type_i0 :
    core_models.fmt.Debug.AssociatedTypes
    FE]
  [trait_constr_Impl_2_i0 : core_models.fmt.Debug FE ] :
  core_models.fmt.Debug.AssociatedTypes (Polynomial FE) :=
  by constructor <;> exact Inhabited.default

@[instance] opaque Impl_2
  (FE : Type)
  [trait_constr_Impl_2_associated_type_i0 :
    core_models.fmt.Debug.AssociatedTypes
    FE]
  [trait_constr_Impl_2_i0 : core_models.fmt.Debug FE ] :
  core_models.fmt.Debug (Polynomial FE) :=
  by constructor <;> exact Inhabited.default

@[instance] opaque Impl_3.AssociatedTypes
  (FE : Type)
  [trait_constr_Impl_3_associated_type_i0 :
    core_models.clone.Clone.AssociatedTypes
    FE]
  [trait_constr_Impl_3_i0 : core_models.clone.Clone FE ] :
  core_models.clone.Clone.AssociatedTypes (Polynomial FE) :=
  by constructor <;> exact Inhabited.default

@[instance] opaque Impl_3
  (FE : Type)
  [trait_constr_Impl_3_associated_type_i0 :
    core_models.clone.Clone.AssociatedTypes
    FE]
  [trait_constr_Impl_3_i0 : core_models.clone.Clone FE ] :
  core_models.clone.Clone (Polynomial FE) :=
  by constructor <;> exact Inhabited.default

@[instance] opaque Impl_4.AssociatedTypes (FE : Type) :
  core_models.marker.StructuralPartialEq.AssociatedTypes (Polynomial FE) :=
  by constructor <;> exact Inhabited.default

@[instance] opaque Impl_4 (FE : Type) :
  core_models.marker.StructuralPartialEq (Polynomial FE) :=
  by constructor <;> exact Inhabited.default

@[instance] opaque Impl_5.AssociatedTypes
  (FE : Type)
  [trait_constr_Impl_5_associated_type_i0 :
    core_models.cmp.PartialEq.AssociatedTypes
    FE
    FE]
  [trait_constr_Impl_5_i0 : core_models.cmp.PartialEq FE FE ] :
  core_models.cmp.PartialEq.AssociatedTypes (Polynomial FE) (Polynomial FE) :=
  by constructor <;> exact Inhabited.default

@[instance] opaque Impl_5
  (FE : Type)
  [trait_constr_Impl_5_associated_type_i0 :
    core_models.cmp.PartialEq.AssociatedTypes
    FE
    FE]
  [trait_constr_Impl_5_i0 : core_models.cmp.PartialEq FE FE ] :
  core_models.cmp.PartialEq (Polynomial FE) (Polynomial FE) :=
  by constructor <;> exact Inhabited.default

@[instance] opaque Impl_6.AssociatedTypes
  (FE : Type)
  [trait_constr_Impl_6_associated_type_i0 : core_models.cmp.Eq.AssociatedTypes
    FE]
  [trait_constr_Impl_6_i0 : core_models.cmp.Eq FE ] :
  core_models.cmp.Eq.AssociatedTypes (Polynomial FE) :=
  by constructor <;> exact Inhabited.default

@[instance] opaque Impl_6
  (FE : Type)
  [trait_constr_Impl_6_associated_type_i0 : core_models.cmp.Eq.AssociatedTypes
    FE]
  [trait_constr_Impl_6_i0 : core_models.cmp.Eq FE ] :
  core_models.cmp.Eq (Polynomial FE) :=
  by constructor <;> exact Inhabited.default

end math.polynomial


namespace math.traits

--  A trait for converting an element to and from its byte representation and
--  for getting an element from its byte representation in big-endian or
--  little-endian order.
class ByteConversion.AssociatedTypes (Self : Type) where

class ByteConversion (Self : Type)
  [associatedTypes : outParam (ByteConversion.AssociatedTypes (Self : Type))]
  where
  BYTE_LEN (Self) : usize
  to_bytes_be (Self) : (Self -> RustM (alloc.vec.Vec u8 alloc.alloc.Global))
  to_bytes_le (Self) : (Self -> RustM (alloc.vec.Vec u8 alloc.alloc.Global))
  from_bytes_be (Self) :
    ((RustSlice u8) ->
    RustM (core_models.result.Result Self math.errors.ByteConversionError))
  from_bytes_le (Self) :
    ((RustSlice u8) ->
    RustM (core_models.result.Result Self math.errors.ByteConversionError))
  write_bytes_be (Self) (self : Self) (buf : (RustSlice u8)) :RustM (RustSlice
    u8) := do
    let bytes : (alloc.vec.Vec u8 alloc.alloc.Global) ←
      (ByteConversion.to_bytes_be Self self);
    let buf : (RustSlice u8) ←
      (rust_primitives.hax.monomorphized_update_at.update_at_range_to
        buf
        (core_models.ops.range.RangeTo.mk
          (_end := (← (alloc.vec.Impl_1.len u8 alloc.alloc.Global bytes))))
        (← (core_models.slice.Impl.copy_from_slice u8
          (← buf[
            (core_models.ops.range.RangeTo.mk
              (_end := (← (alloc.vec.Impl_1.len u8 alloc.alloc.Global bytes))))
            ]_?)
          (← (core_models.ops.deref.Deref.deref
            (alloc.vec.Vec u8 alloc.alloc.Global) bytes)))));
    (pure buf)

@[reducible] instance Impl_2.AssociatedTypes :
  ByteConversion.AssociatedTypes u64
  where

instance Impl_2 : ByteConversion u64 where
  BYTE_LEN := (Impl_2.BYTE_LEN_hoisted)
  to_bytes_be := (Impl_2.to_bytes_be_hoisted)
  to_bytes_le := (Impl_2.to_bytes_le_hoisted)
  from_bytes_be := (Impl_2.from_bytes_be_hoisted)
  from_bytes_le := (Impl_2.from_bytes_le_hoisted)

end math.traits


namespace math.field.traits

--  Trait to add field behaviour to a struct.
class IsField.AssociatedTypes (Self : Type) where
  [trait_constr_IsField_i0 : core_models.fmt.Debug.AssociatedTypes Self]
  [trait_constr_IsField_i1 : core_models.clone.Clone.AssociatedTypes Self]
  BaseType : Type

attribute [instance_reducible, instance]
  IsField.AssociatedTypes.trait_constr_IsField_i0

attribute [instance_reducible, instance]
  IsField.AssociatedTypes.trait_constr_IsField_i1

attribute [reducible] IsField.AssociatedTypes.BaseType

abbrev IsField.BaseType :=
  IsField.AssociatedTypes.BaseType

class IsField (Self : Type)
  [associatedTypes : outParam (IsField.AssociatedTypes (Self : Type))]
  where
  [trait_constr_IsField_i0 : core_models.fmt.Debug Self]
  [trait_constr_IsField_i1 : core_models.clone.Clone Self]
  [trait_constr_BaseType_associated_type_i1 :
    core_models.clone.Clone.AssociatedTypes
    associatedTypes.BaseType]
  [trait_constr_BaseType_i1 : core_models.clone.Clone associatedTypes.BaseType ]
  [trait_constr_BaseType_associated_type_i2 :
    core_models.fmt.Debug.AssociatedTypes
    associatedTypes.BaseType]
  [trait_constr_BaseType_i2 : core_models.fmt.Debug associatedTypes.BaseType ]
  [trait_constr_BaseType_associated_type_i3 :
    math.traits.ByteConversion.AssociatedTypes
    associatedTypes.BaseType]
  [trait_constr_BaseType_i3 : math.traits.ByteConversion
    associatedTypes.BaseType
    ]
  [trait_constr_BaseType_associated_type_i4 :
    core_models.default.Default.AssociatedTypes
    associatedTypes.BaseType]
  [trait_constr_BaseType_i4 : core_models.default.Default
    associatedTypes.BaseType
    ]
  [trait_constr_BaseType_associated_type_i5 :
    core_models.marker.Send.AssociatedTypes
    associatedTypes.BaseType]
  [trait_constr_BaseType_i5 : core_models.marker.Send associatedTypes.BaseType ]
  [trait_constr_BaseType_associated_type_i6 :
    core_models.marker.Sync.AssociatedTypes
    associatedTypes.BaseType]
  [trait_constr_BaseType_i6 : core_models.marker.Sync associatedTypes.BaseType ]
  add (Self) :
    (associatedTypes.BaseType ->
    associatedTypes.BaseType ->
    RustM associatedTypes.BaseType)
  double (Self) (a : associatedTypes.BaseType) :RustM associatedTypes.BaseType
    := do
    (IsField.add Self a a)
  mul (Self) :
    (associatedTypes.BaseType ->
    associatedTypes.BaseType ->
    RustM associatedTypes.BaseType)
  square (Self) (a : associatedTypes.BaseType) :RustM associatedTypes.BaseType
    := do
    (IsField.mul Self a a)
  pow (Self)
    (T : Type)
    [trait_constr_pow_associated_type_i1 :
      math.unsigned_integer.traits.IsUnsignedInteger.AssociatedTypes
      T]
    [trait_constr_pow_i1 : math.unsigned_integer.traits.IsUnsignedInteger T ]
    (a : associatedTypes.BaseType)
    (exponent : T) :RustM associatedTypes.BaseType := do
    let zero : T ← (core_models.convert.From._from T u16 (0 : u16));
    let one : T ← (core_models.convert.From._from T u16 (1 : u16));
    if (← (core_models.cmp.PartialEq.eq T T exponent zero)) then do
      (IsField.one Self rust_primitives.hax.Tuple0.mk)
    else do
      if (← (core_models.cmp.PartialEq.eq T T exponent one)) then do
        (core_models.clone.Clone.clone associatedTypes.BaseType a)
      else do
        let result : associatedTypes.BaseType ←
          (core_models.clone.Clone.clone associatedTypes.BaseType a);
        let ⟨exponent, result⟩ ←
          (rust_primitives.hax.while_loop
            (fun ⟨exponent, result⟩ => (do (pure true) : RustM Bool))
            (fun ⟨exponent, result⟩ =>
              (do
              (core_models.cmp.PartialEq.eq
                T
                T (← (core_models.ops.bit.BitAnd.bitand T T exponent one)) zero)
              :
              RustM Bool))
            (fun ⟨exponent, result⟩ =>
              (do
              (rust_primitives.hax.int.from_machine (0 : u32)) :
              RustM hax_lib.int.Int))
            (rust_primitives.hax.Tuple2.mk exponent result)
            (fun ⟨exponent, result⟩ =>
              (do
              let result : associatedTypes.BaseType ←
                (IsField.square Self result);
              let exponent : T ←
                (core_models.ops.bit.ShrAssign.shr_assign
                  T
                  usize exponent (1 : usize));
              (pure (rust_primitives.hax.Tuple2.mk exponent result)) :
              RustM (rust_primitives.hax.Tuple2 T associatedTypes.BaseType))));
        if (← (core_models.cmp.PartialEq.eq T T exponent zero)) then do
          (pure result)
        else do
          let base : associatedTypes.BaseType ←
            (core_models.clone.Clone.clone associatedTypes.BaseType result);
          let exponent : T ←
            (core_models.ops.bit.ShrAssign.shr_assign
              T
              usize exponent (1 : usize));
          let ⟨base, exponent, result⟩ ←
            (rust_primitives.hax.while_loop
              (fun ⟨base, exponent, result⟩ => (do (pure true) : RustM Bool))
              (fun ⟨base, exponent, result⟩ =>
                (do
                (core_models.cmp.PartialEq.ne T T exponent zero) : RustM Bool))
              (fun ⟨base, exponent, result⟩ =>
                (do
                (rust_primitives.hax.int.from_machine (0 : u32)) :
                RustM hax_lib.int.Int))
              (rust_primitives.hax.Tuple3.mk base exponent result)
              (fun ⟨base, exponent, result⟩ =>
                (do
                let base : associatedTypes.BaseType ←
                  (IsField.square Self base);
                let result : associatedTypes.BaseType ←
                  if
                  (← (core_models.cmp.PartialEq.eq
                    T
                    T
                    (← (core_models.ops.bit.BitAnd.bitand T T exponent one))
                    one)) then do
                    let result : associatedTypes.BaseType ←
                      (IsField.mul Self result base);
                    (pure result)
                  else do
                    (pure result);
                let exponent : T ←
                  (core_models.ops.bit.ShrAssign.shr_assign
                    T
                    usize exponent (1 : usize));
                (pure (rust_primitives.hax.Tuple3.mk base exponent result)) :
                RustM
                (rust_primitives.hax.Tuple3
                  associatedTypes.BaseType
                  T
                  associatedTypes.BaseType))));
          (pure result)
  sub (Self) :
    (associatedTypes.BaseType ->
    associatedTypes.BaseType ->
    RustM associatedTypes.BaseType)
  neg (Self) : (associatedTypes.BaseType -> RustM associatedTypes.BaseType)
  inv (Self) :
    (associatedTypes.BaseType ->
    RustM (core_models.result.Result
      associatedTypes.BaseType
      math.field.errors.FieldError))
  div (Self) :
    (associatedTypes.BaseType ->
    associatedTypes.BaseType ->
    RustM (core_models.result.Result
      associatedTypes.BaseType
      math.field.errors.FieldError))
  eq (Self) :
    (associatedTypes.BaseType -> associatedTypes.BaseType -> RustM Bool)
  zero (Self) (_ : rust_primitives.hax.Tuple0) :RustM associatedTypes.BaseType
    := do
    (core_models.default.Default.default
      associatedTypes.BaseType rust_primitives.hax.Tuple0.mk)
  one (Self) : (rust_primitives.hax.Tuple0 -> RustM associatedTypes.BaseType)
  from_u64 (Self) : (u64 -> RustM associatedTypes.BaseType)
  from_base_type (Self) :
    (associatedTypes.BaseType -> RustM associatedTypes.BaseType)

attribute [instance_reducible, instance] IsField.trait_constr_IsField_i0

attribute [instance_reducible, instance] IsField.trait_constr_IsField_i1

end math.field.traits


namespace math.field.element

--  A field element with operations algorithms defined in `F`
-- 
--  `#[repr(transparent)]` makes `FieldElement<F>` byte-identical to
--  `F::BaseType`, which [`SpillSafe`](crate::spill_safe::SpillSafe)
--  requires. Changing the `repr` or adding fields breaks this and
--  is UB in any function that requires `T: SpillSafe`.
structure FieldElement
  (F : Type)
  [trait_constr_FieldElement_associated_type_i0 :
    math.field.traits.IsField.AssociatedTypes
    F]
  [trait_constr_FieldElement_i0 : math.field.traits.IsField F ]
  where
  value : (math.field.traits.IsField.BaseType F)

@[instance] opaque Impl_34.AssociatedTypes
  (F : Type)
  [trait_constr_Impl_34_associated_type_i0 :
    core_models.fmt.Debug.AssociatedTypes
    F]
  [trait_constr_Impl_34_i0 : core_models.fmt.Debug F ]
  [trait_constr_Impl_34_associated_type_i1 :
    math.field.traits.IsField.AssociatedTypes
    F]
  [trait_constr_Impl_34_i1 : math.field.traits.IsField F ]
  [trait_constr_Impl_34_associated_type_i2 :
    core_models.fmt.Debug.AssociatedTypes
    (math.field.traits.IsField.BaseType F)]
  [trait_constr_Impl_34_i2 : core_models.fmt.Debug
    (math.field.traits.IsField.BaseType F)
    ] :
  core_models.fmt.Debug.AssociatedTypes (FieldElement F) :=
  by constructor <;> exact Inhabited.default

@[instance] opaque Impl_34
  (F : Type)
  [trait_constr_Impl_34_associated_type_i0 :
    core_models.fmt.Debug.AssociatedTypes
    F]
  [trait_constr_Impl_34_i0 : core_models.fmt.Debug F ]
  [trait_constr_Impl_34_associated_type_i1 :
    math.field.traits.IsField.AssociatedTypes
    F]
  [trait_constr_Impl_34_i1 : math.field.traits.IsField F ]
  [trait_constr_Impl_34_associated_type_i2 :
    core_models.fmt.Debug.AssociatedTypes
    (math.field.traits.IsField.BaseType F)]
  [trait_constr_Impl_34_i2 : core_models.fmt.Debug
    (math.field.traits.IsField.BaseType F)
    ] :
  core_models.fmt.Debug (FieldElement F) :=
  by constructor <;> exact Inhabited.default

@[instance] opaque Impl_35.AssociatedTypes
  (F : Type)
  [trait_constr_Impl_35_associated_type_i0 :
    core_models.clone.Clone.AssociatedTypes
    F]
  [trait_constr_Impl_35_i0 : core_models.clone.Clone F ]
  [trait_constr_Impl_35_associated_type_i1 :
    math.field.traits.IsField.AssociatedTypes
    F]
  [trait_constr_Impl_35_i1 : math.field.traits.IsField F ]
  [trait_constr_Impl_35_associated_type_i2 :
    core_models.clone.Clone.AssociatedTypes
    (math.field.traits.IsField.BaseType F)]
  [trait_constr_Impl_35_i2 : core_models.clone.Clone
    (math.field.traits.IsField.BaseType F)
    ] :
  core_models.clone.Clone.AssociatedTypes (FieldElement F) :=
  by constructor <;> exact Inhabited.default

@[instance] opaque Impl_35
  (F : Type)
  [trait_constr_Impl_35_associated_type_i0 :
    core_models.clone.Clone.AssociatedTypes
    F]
  [trait_constr_Impl_35_i0 : core_models.clone.Clone F ]
  [trait_constr_Impl_35_associated_type_i1 :
    math.field.traits.IsField.AssociatedTypes
    F]
  [trait_constr_Impl_35_i1 : math.field.traits.IsField F ]
  [trait_constr_Impl_35_associated_type_i2 :
    core_models.clone.Clone.AssociatedTypes
    (math.field.traits.IsField.BaseType F)]
  [trait_constr_Impl_35_i2 : core_models.clone.Clone
    (math.field.traits.IsField.BaseType F)
    ] :
  core_models.clone.Clone (FieldElement F) :=
  by constructor <;> exact Inhabited.default

@[instance] opaque Impl_36.AssociatedTypes
  (F : Type)
  [trait_constr_Impl_36_associated_type_i0 :
    core_models.hash.Hash.AssociatedTypes
    F]
  [trait_constr_Impl_36_i0 : core_models.hash.Hash F ]
  [trait_constr_Impl_36_associated_type_i1 :
    math.field.traits.IsField.AssociatedTypes
    F]
  [trait_constr_Impl_36_i1 : math.field.traits.IsField F ]
  [trait_constr_Impl_36_associated_type_i2 :
    core_models.hash.Hash.AssociatedTypes
    (math.field.traits.IsField.BaseType F)]
  [trait_constr_Impl_36_i2 : core_models.hash.Hash
    (math.field.traits.IsField.BaseType F)
    ] :
  core_models.hash.Hash.AssociatedTypes (FieldElement F) :=
  by constructor <;> exact Inhabited.default

@[instance] opaque Impl_36
  (F : Type)
  [trait_constr_Impl_36_associated_type_i0 :
    core_models.hash.Hash.AssociatedTypes
    F]
  [trait_constr_Impl_36_i0 : core_models.hash.Hash F ]
  [trait_constr_Impl_36_associated_type_i1 :
    math.field.traits.IsField.AssociatedTypes
    F]
  [trait_constr_Impl_36_i1 : math.field.traits.IsField F ]
  [trait_constr_Impl_36_associated_type_i2 :
    core_models.hash.Hash.AssociatedTypes
    (math.field.traits.IsField.BaseType F)]
  [trait_constr_Impl_36_i2 : core_models.hash.Hash
    (math.field.traits.IsField.BaseType F)
    ] :
  core_models.hash.Hash (FieldElement F) :=
  by constructor <;> exact Inhabited.default

@[instance] opaque Impl_37.AssociatedTypes
  (F : Type)
  [trait_constr_Impl_37_associated_type_i0 :
    core_models.marker.Copy.AssociatedTypes
    F]
  [trait_constr_Impl_37_i0 : core_models.marker.Copy F ]
  [trait_constr_Impl_37_associated_type_i1 :
    math.field.traits.IsField.AssociatedTypes
    F]
  [trait_constr_Impl_37_i1 : math.field.traits.IsField F ]
  [trait_constr_Impl_37_associated_type_i2 :
    core_models.marker.Copy.AssociatedTypes
    (math.field.traits.IsField.BaseType F)]
  [trait_constr_Impl_37_i2 : core_models.marker.Copy
    (math.field.traits.IsField.BaseType F)
    ] :
  core_models.marker.Copy.AssociatedTypes (FieldElement F) :=
  by constructor <;> exact Inhabited.default

@[instance] opaque Impl_37
  (F : Type)
  [trait_constr_Impl_37_associated_type_i0 :
    core_models.marker.Copy.AssociatedTypes
    F]
  [trait_constr_Impl_37_i0 : core_models.marker.Copy F ]
  [trait_constr_Impl_37_associated_type_i1 :
    math.field.traits.IsField.AssociatedTypes
    F]
  [trait_constr_Impl_37_i1 : math.field.traits.IsField F ]
  [trait_constr_Impl_37_associated_type_i2 :
    core_models.marker.Copy.AssociatedTypes
    (math.field.traits.IsField.BaseType F)]
  [trait_constr_Impl_37_i2 : core_models.marker.Copy
    (math.field.traits.IsField.BaseType F)
    ] :
  core_models.marker.Copy (FieldElement F) :=
  by constructor <;> exact Inhabited.default

@[spec]
def Impl_1.from_hoisted
    (F : Type)
    [trait_constr_from_hoisted_associated_type_i0 :
      core_models.clone.Clone.AssociatedTypes
      (math.field.traits.IsField.BaseType F)]
    [trait_constr_from_hoisted_i0 : core_models.clone.Clone
      (math.field.traits.IsField.BaseType F)
      ]
    [trait_constr_from_hoisted_associated_type_i1 :
      math.field.traits.IsField.AssociatedTypes
      F]
    [trait_constr_from_hoisted_i1 : math.field.traits.IsField F ]
    (value : (math.field.traits.IsField.BaseType F)) :
    RustM (FieldElement F) := do
  (pure (FieldElement.mk
    (value := (← (math.field.traits.IsField.from_base_type
      F
      (← (core_models.clone.Clone.clone
        (math.field.traits.IsField.BaseType F) value)))))))

--  From overloading for field elements
@[reducible] instance Impl_1.AssociatedTypes
  (F : Type)
  [trait_constr_Impl_1_associated_type_i0 :
    core_models.clone.Clone.AssociatedTypes
    (math.field.traits.IsField.BaseType F)]
  [trait_constr_Impl_1_i0 : core_models.clone.Clone
    (math.field.traits.IsField.BaseType F)
    ]
  [trait_constr_Impl_1_associated_type_i1 :
    math.field.traits.IsField.AssociatedTypes
    F]
  [trait_constr_Impl_1_i1 : math.field.traits.IsField F ] :
  core_models.convert.From.AssociatedTypes
  (FieldElement F)
  (math.field.traits.IsField.BaseType F)
  where

instance Impl_1
  (F : Type)
  [trait_constr_Impl_1_associated_type_i0 :
    core_models.clone.Clone.AssociatedTypes
    (math.field.traits.IsField.BaseType F)]
  [trait_constr_Impl_1_i0 : core_models.clone.Clone
    (math.field.traits.IsField.BaseType F)
    ]
  [trait_constr_Impl_1_associated_type_i1 :
    math.field.traits.IsField.AssociatedTypes
    F]
  [trait_constr_Impl_1_i1 : math.field.traits.IsField F ] :
  core_models.convert.From
  (FieldElement F)
  (math.field.traits.IsField.BaseType F)
  where
  _from := (Impl_1.from_hoisted F)

@[spec]
def Impl_2.from_hoisted
    (F : Type)
    [trait_constr_from_hoisted_associated_type_i0 :
      math.field.traits.IsField.AssociatedTypes
      F]
    [trait_constr_from_hoisted_i0 : math.field.traits.IsField F ]
    (value : u64) :
    RustM (FieldElement F) := do
  (pure (FieldElement.mk
    (value := (← (math.field.traits.IsField.from_u64 F value)))))

--  From overloading for U64
@[reducible] instance Impl_2.AssociatedTypes
  (F : Type)
  [trait_constr_Impl_2_associated_type_i0 :
    math.field.traits.IsField.AssociatedTypes
    F]
  [trait_constr_Impl_2_i0 : math.field.traits.IsField F ] :
  core_models.convert.From.AssociatedTypes (FieldElement F) u64
  where

instance Impl_2
  (F : Type)
  [trait_constr_Impl_2_associated_type_i0 :
    math.field.traits.IsField.AssociatedTypes
    F]
  [trait_constr_Impl_2_i0 : math.field.traits.IsField F ] :
  core_models.convert.From (FieldElement F) u64
  where
  _from := (Impl_2.from_hoisted F)

@[spec]
def Impl_6.from_raw
    (F : Type)
    [trait_constr_from_raw_associated_type_i0 :
      core_models.clone.Clone.AssociatedTypes
      (math.field.traits.IsField.BaseType F)]
    [trait_constr_from_raw_i0 : core_models.clone.Clone
      (math.field.traits.IsField.BaseType F)
      ]
    [trait_constr_from_raw_associated_type_i1 :
      math.field.traits.IsField.AssociatedTypes
      F]
    [trait_constr_from_raw_i1 : math.field.traits.IsField F ]
    (value : (math.field.traits.IsField.BaseType F)) :
    RustM (FieldElement F) := do
  (pure (FieldElement.mk (value := value)))

@[spec]
def Impl_6.const_from_raw
    (F : Type)
    [trait_constr_const_from_raw_associated_type_i0 :
      core_models.clone.Clone.AssociatedTypes
      (math.field.traits.IsField.BaseType F)]
    [trait_constr_const_from_raw_i0 : core_models.clone.Clone
      (math.field.traits.IsField.BaseType F)
      ]
    [trait_constr_const_from_raw_associated_type_i1 :
      math.field.traits.IsField.AssociatedTypes
      F]
    [trait_constr_const_from_raw_i1 : math.field.traits.IsField F ]
    (value : (math.field.traits.IsField.BaseType F)) :
    RustM (FieldElement F) := do
  (pure (FieldElement.mk (value := value)))

@[spec]
def Impl_7.eq_hoisted
    (F : Type)
    [trait_constr_eq_hoisted_associated_type_i0 :
      math.field.traits.IsField.AssociatedTypes
      F]
    [trait_constr_eq_hoisted_i0 : math.field.traits.IsField F ]
    (self : (FieldElement F))
    (other : (FieldElement F)) :
    RustM Bool := do
  (math.field.traits.IsField.eq
    F (FieldElement.value self) (FieldElement.value other))

--  Equality operator overloading for field elements
@[reducible] instance Impl_7.AssociatedTypes
  (F : Type)
  [trait_constr_Impl_7_associated_type_i0 :
    math.field.traits.IsField.AssociatedTypes
    F]
  [trait_constr_Impl_7_i0 : math.field.traits.IsField F ] :
  core_models.cmp.PartialEq.AssociatedTypes (FieldElement F) (FieldElement F)
  where

instance Impl_7
  (F : Type)
  [trait_constr_Impl_7_associated_type_i0 :
    math.field.traits.IsField.AssociatedTypes
    F]
  [trait_constr_Impl_7_i0 : math.field.traits.IsField F ] :
  core_models.cmp.PartialEq (FieldElement F) (FieldElement F)
  where
  eq := (Impl_7.eq_hoisted F)

@[reducible] instance Impl_8.AssociatedTypes
  (F : Type)
  [trait_constr_Impl_8_associated_type_i0 :
    math.field.traits.IsField.AssociatedTypes
    F]
  [trait_constr_Impl_8_i0 : math.field.traits.IsField F ] :
  core_models.cmp.Eq.AssociatedTypes (FieldElement F)
  where

instance Impl_8
  (F : Type)
  [trait_constr_Impl_8_associated_type_i0 :
    math.field.traits.IsField.AssociatedTypes
    F]
  [trait_constr_Impl_8_i0 : math.field.traits.IsField F ] :
  core_models.cmp.Eq (FieldElement F)
  where

@[spec]
def Impl_29.neg_hoisted
    (F : Type)
    [trait_constr_neg_hoisted_associated_type_i0 :
      math.field.traits.IsField.AssociatedTypes
      F]
    [trait_constr_neg_hoisted_i0 : math.field.traits.IsField F ]
    (self : (FieldElement F)) :
    RustM (FieldElement F) := do
  (pure (FieldElement.mk
    (value := (← (math.field.traits.IsField.neg F (FieldElement.value self))))))

--  Negation operator overloading for field elements*/
@[reducible] instance Impl_29.AssociatedTypes
  (F : Type)
  [trait_constr_Impl_29_associated_type_i0 :
    math.field.traits.IsField.AssociatedTypes
    F]
  [trait_constr_Impl_29_i0 : math.field.traits.IsField F ] :
  core_models.ops.arith.Neg.AssociatedTypes (FieldElement F)
  where
  Output := (FieldElement F)

instance Impl_29
  (F : Type)
  [trait_constr_Impl_29_associated_type_i0 :
    math.field.traits.IsField.AssociatedTypes
    F]
  [trait_constr_Impl_29_i0 : math.field.traits.IsField F ] :
  core_models.ops.arith.Neg (FieldElement F)
  where
  neg := (Impl_29.neg_hoisted F)

@[spec]
def Impl_30.neg_hoisted
    (F : Type)
    [trait_constr_neg_hoisted_associated_type_i0 :
      math.field.traits.IsField.AssociatedTypes
      F]
    [trait_constr_neg_hoisted_i0 : math.field.traits.IsField F ]
    (self : (FieldElement F)) :
    RustM (FieldElement F) := do
  (core_models.field.element.Impl_30.neg_hoisted self)

@[reducible] instance Impl_30.AssociatedTypes
  (F : Type)
  [trait_constr_Impl_30_associated_type_i0 :
    math.field.traits.IsField.AssociatedTypes
    F]
  [trait_constr_Impl_30_i0 : math.field.traits.IsField F ] :
  core_models.ops.arith.Neg.AssociatedTypes (FieldElement F)
  where
  Output := (FieldElement F)

instance Impl_30
  (F : Type)
  [trait_constr_Impl_30_associated_type_i0 :
    math.field.traits.IsField.AssociatedTypes
    F]
  [trait_constr_Impl_30_i0 : math.field.traits.IsField F ] :
  core_models.ops.arith.Neg (FieldElement F)
  where
  neg := (Impl_30.neg_hoisted F)

@[spec]
def Impl_3.from_hoisted
    (F : Type)
    [trait_constr_from_hoisted_associated_type_i0 :
      math.field.traits.IsField.AssociatedTypes
      F]
    [trait_constr_from_hoisted_i0 : math.field.traits.IsField F ]
    (value : i64) :
    RustM (FieldElement F) := do
  if (← (value >=? (0 : i64))) then do
    (core_models.convert.From._from
      (FieldElement F)
      u64 (← (rust_primitives.hax.cast_op value : RustM u64)))
  else do
    (core_models.ops.arith.Neg.neg
      (FieldElement F)
      (← (core_models.convert.From._from
        (FieldElement F)
        u64 (← (core_models.num.Impl_3.unsigned_abs value)))))

--  From overloading for i64.
--  Negative values are converted to their field equivalents: -x becomes p - x.
@[reducible] instance Impl_3.AssociatedTypes
  (F : Type)
  [trait_constr_Impl_3_associated_type_i0 :
    math.field.traits.IsField.AssociatedTypes
    F]
  [trait_constr_Impl_3_i0 : math.field.traits.IsField F ] :
  core_models.convert.From.AssociatedTypes (FieldElement F) i64
  where

instance Impl_3
  (F : Type)
  [trait_constr_Impl_3_associated_type_i0 :
    math.field.traits.IsField.AssociatedTypes
    F]
  [trait_constr_Impl_3_i0 : math.field.traits.IsField F ] :
  core_models.convert.From (FieldElement F) i64
  where
  _from := (Impl_3.from_hoisted F)

@[spec]
def Impl_4.from_hoisted
    (F : Type)
    [trait_constr_from_hoisted_associated_type_i0 :
      math.field.traits.IsField.AssociatedTypes
      F]
    [trait_constr_from_hoisted_i0 : math.field.traits.IsField F ]
    (value : i32) :
    RustM (FieldElement F) := do
  (core_models.convert.From._from
    (FieldElement F)
    i64 (← (rust_primitives.hax.cast_op value : RustM i64)))

--  From overloading for i32 (convenience for integer literals).
@[reducible] instance Impl_4.AssociatedTypes
  (F : Type)
  [trait_constr_Impl_4_associated_type_i0 :
    math.field.traits.IsField.AssociatedTypes
    F]
  [trait_constr_Impl_4_i0 : math.field.traits.IsField F ] :
  core_models.convert.From.AssociatedTypes (FieldElement F) i32
  where

instance Impl_4
  (F : Type)
  [trait_constr_Impl_4_associated_type_i0 :
    math.field.traits.IsField.AssociatedTypes
    F]
  [trait_constr_Impl_4_i0 : math.field.traits.IsField F ] :
  core_models.convert.From (FieldElement F) i32
  where
  _from := (Impl_4.from_hoisted F)

@[spec]
def Impl_31.default_hoisted
    (F : Type)
    [trait_constr_default_hoisted_associated_type_i0 :
      math.field.traits.IsField.AssociatedTypes
      F]
    [trait_constr_default_hoisted_i0 : math.field.traits.IsField F ]
    (_ : rust_primitives.hax.Tuple0) :
    RustM (FieldElement F) := do
  (pure (FieldElement.mk
    (value := (← (math.field.traits.IsField.zero
      F rust_primitives.hax.Tuple0.mk)))))

@[reducible] instance Impl_31.AssociatedTypes
  (F : Type)
  [trait_constr_Impl_31_associated_type_i0 :
    math.field.traits.IsField.AssociatedTypes
    F]
  [trait_constr_Impl_31_i0 : math.field.traits.IsField F ] :
  core_models.default.Default.AssociatedTypes (FieldElement F)
  where

instance Impl_31
  (F : Type)
  [trait_constr_Impl_31_associated_type_i0 :
    math.field.traits.IsField.AssociatedTypes
    F]
  [trait_constr_Impl_31_i0 : math.field.traits.IsField F ] :
  core_models.default.Default (FieldElement F)
  where
  default := (Impl_31.default_hoisted F)

--  Creates a field element from `value`
@[spec]
def Impl_32.new
    (F : Type)
    [trait_constr_new_associated_type_i0 :
      math.field.traits.IsField.AssociatedTypes
      F]
    [trait_constr_new_i0 : math.field.traits.IsField F ]
    (value : (math.field.traits.IsField.BaseType F)) :
    RustM (FieldElement F) := do
  (pure (FieldElement.mk
    (value := (← (math.field.traits.IsField.from_base_type F value)))))

--  Returns the underlying `value`
@[spec]
def Impl_32.value
    (F : Type)
    [trait_constr_value_associated_type_i0 :
      math.field.traits.IsField.AssociatedTypes
      F]
    [trait_constr_value_i0 : math.field.traits.IsField F ]
    (self : (FieldElement F)) :
    RustM (math.field.traits.IsField.BaseType F) := do
  (pure (FieldElement.value self))

--  Returns the multiplicative inverse of `self`
@[spec]
def Impl_32.inv
    (F : Type)
    [trait_constr_inv_associated_type_i0 :
      math.field.traits.IsField.AssociatedTypes
      F]
    [trait_constr_inv_i0 : math.field.traits.IsField F ]
    (self : (FieldElement F)) :
    RustM
    (core_models.result.Result (FieldElement F) math.field.errors.FieldError)
    := do
  match (← (math.field.traits.IsField.inv F (FieldElement.value self))) with
    | (core_models.result.Result.Ok  value) => do
      (pure (core_models.result.Result.Ok (FieldElement.mk (value := value))))
    | (core_models.result.Result.Err  err) => do
      (pure (core_models.result.Result.Err err))

--  Returns the square of `self`
@[spec]
def Impl_32.square
    (F : Type)
    [trait_constr_square_associated_type_i0 :
      math.field.traits.IsField.AssociatedTypes
      F]
    [trait_constr_square_i0 : math.field.traits.IsField F ]
    (self : (FieldElement F)) :
    RustM (FieldElement F) := do
  (pure (FieldElement.mk
    (value := (← (math.field.traits.IsField.square
      F (FieldElement.value self))))))

--  Returns the double of `self`
@[spec]
def Impl_32.double
    (F : Type)
    [trait_constr_double_associated_type_i0 :
      math.field.traits.IsField.AssociatedTypes
      F]
    [trait_constr_double_i0 : math.field.traits.IsField F ]
    (self : (FieldElement F)) :
    RustM (FieldElement F) := do
  (pure (FieldElement.mk
    (value := (← (math.field.traits.IsField.double
      F (FieldElement.value self))))))

--  Returns `self` raised to the power of `exponent`
@[spec]
def Impl_32.pow
    (F : Type)
    (T : Type)
    [trait_constr_pow_associated_type_i0 :
      math.field.traits.IsField.AssociatedTypes
      F]
    [trait_constr_pow_i0 : math.field.traits.IsField F ]
    [trait_constr_pow_associated_type_i1 :
      math.unsigned_integer.traits.IsUnsignedInteger.AssociatedTypes
      T]
    [trait_constr_pow_i1 : math.unsigned_integer.traits.IsUnsignedInteger T ]
    (self : (FieldElement F))
    (exponent : T) :
    RustM (FieldElement F) := do
  (pure (FieldElement.mk
    (value := (← (math.field.traits.IsField.pow
      F T (FieldElement.value self) exponent)))))

--  Returns the multiplicative neutral element of the field.
@[spec]
def Impl_32.one
    (F : Type)
    [trait_constr_one_associated_type_i0 :
      math.field.traits.IsField.AssociatedTypes
      F]
    [trait_constr_one_i0 : math.field.traits.IsField F ]
    (_ : rust_primitives.hax.Tuple0) :
    RustM (FieldElement F) := do
  (pure (FieldElement.mk
    (value := (← (math.field.traits.IsField.one
      F rust_primitives.hax.Tuple0.mk)))))

--  Returns the additive neutral element of the field.
@[spec]
def Impl_32.zero
    (F : Type)
    [trait_constr_zero_associated_type_i0 :
      math.field.traits.IsField.AssociatedTypes
      F]
    [trait_constr_zero_i0 : math.field.traits.IsField F ]
    (_ : rust_primitives.hax.Tuple0) :
    RustM (FieldElement F) := do
  (pure (FieldElement.mk
    (value := (← (math.field.traits.IsField.zero
      F rust_primitives.hax.Tuple0.mk)))))

--  Returns the raw base type
@[spec]
def Impl_32.to_raw
    (F : Type)
    [trait_constr_to_raw_associated_type_i0 :
      math.field.traits.IsField.AssociatedTypes
      F]
    [trait_constr_to_raw_i0 : math.field.traits.IsField F ]
    (self : (FieldElement F)) :
    RustM (math.field.traits.IsField.BaseType F) := do
  (pure (FieldElement.value self))

--  Converts a field element into a BigUint.
@[spec]
def Impl_32.to_big_uint
    (F : Type)
    [trait_constr_to_big_uint_associated_type_i0 :
      math.field.traits.IsField.AssociatedTypes
      F]
    [trait_constr_to_big_uint_i0 : math.field.traits.IsField F ]
    [trait_constr_to_big_uint_associated_type_i1 :
      math.traits.ByteConversion.AssociatedTypes
      (FieldElement F)]
    [trait_constr_to_big_uint_i1 : math.traits.ByteConversion (FieldElement F) ]
    (self : (FieldElement F)) :
    RustM num_bigint.biguint.BigUint := do
  (num_bigint.biguint.Impl_19.from_bytes_be
    (← (core_models.ops.deref.Deref.deref
      (alloc.vec.Vec u8 alloc.alloc.Global)
      (← (math.traits.ByteConversion.to_bytes_be (FieldElement F) self)))))

--  Converts a field element into a hex string.
@[spec]
def Impl_32.to_hex_str
    (F : Type)
    [trait_constr_to_hex_str_associated_type_i0 :
      math.field.traits.IsField.AssociatedTypes
      F]
    [trait_constr_to_hex_str_i0 : math.field.traits.IsField F ]
    [trait_constr_to_hex_str_associated_type_i1 :
      math.traits.ByteConversion.AssociatedTypes
      (FieldElement F)]
    [trait_constr_to_hex_str_i1 : math.traits.ByteConversion (FieldElement F) ]
    (self : (FieldElement F)) :
    RustM alloc.string.String := do
  let args : (rust_primitives.hax.Tuple1 num_bigint.biguint.BigUint) :=
    (rust_primitives.hax.Tuple1.mk (← (Impl_32.to_big_uint F self)));
  let args : (RustArray core_models.fmt.rt.Argument 1) :=
    (RustArray.ofVec #v[(← (core_models.fmt.rt.Impl.new_upper_hex
                            num_bigint.biguint.BigUint
                            (rust_primitives.hax.Tuple1._0 args)))]);
  (core_models.hint.must_use alloc.string.String
    (← (alloc.fmt.format
      (← (core_models.fmt.rt.Impl_1.new_v1_formatted
        (← (rust_primitives.unsize (RustArray.ofVec #v["0x"])))
        (← (rust_primitives.unsize args))
        (← (rust_primitives.unsize
          (RustArray.ofVec #v[(core_models.fmt.rt.Placeholder.mk
                                  (position := (0 : usize))
                                  (flags := (3909091360 : u32))
                                  (precision :=
                                  core_models.fmt.rt.Count.Implied)
                                  (width := (core_models.fmt.rt.Count.Is
                                    (2 : u16))))]))))))))

end math.field.element


namespace math.field.extensions_goldilocks

@[spec]
def Impl.to_bytes_be_hoisted
    (self :
    (RustArray
    (math.field.element.FieldElement math.field.goldilocks.GoldilocksField)
    2)) :
    RustM (alloc.vec.Vec u8 alloc.alloc.Global) := do
  (rust_primitives.hax.never_to_any
    (← (core_models.panicking.panic "not implemented")))

@[spec]
def Impl.to_bytes_le_hoisted
    (self :
    (RustArray
    (math.field.element.FieldElement math.field.goldilocks.GoldilocksField)
    2)) :
    RustM (alloc.vec.Vec u8 alloc.alloc.Global) := do
  (rust_primitives.hax.never_to_any
    (← (core_models.panicking.panic "not implemented")))

@[spec]
def Impl.from_bytes_be_hoisted (_bytes : (RustSlice u8)) :
    RustM
    (core_models.result.Result
      (RustArray
      (math.field.element.FieldElement math.field.goldilocks.GoldilocksField)
      2)
      math.errors.ByteConversionError)
    := do
  (rust_primitives.hax.never_to_any
    (← (core_models.panicking.panic "not implemented")))

@[spec]
def Impl.from_bytes_le_hoisted (_bytes : (RustSlice u8)) :
    RustM
    (core_models.result.Result
      (RustArray
      (math.field.element.FieldElement math.field.goldilocks.GoldilocksField)
      2)
      math.errors.ByteConversionError)
    := do
  (rust_primitives.hax.never_to_any
    (← (core_models.panicking.panic "not implemented")))

@[reducible] instance Impl.AssociatedTypes :
  math.traits.ByteConversion.AssociatedTypes
  (RustArray
  (math.field.element.FieldElement math.field.goldilocks.GoldilocksField)
  2)
  where

instance Impl :
  math.traits.ByteConversion
  (RustArray
  (math.field.element.FieldElement math.field.goldilocks.GoldilocksField)
  2)
  where
  BYTE_LEN := (Impl.BYTE_LEN_hoisted)
  to_bytes_be := (Impl.to_bytes_be_hoisted)
  to_bytes_le := (Impl.to_bytes_le_hoisted)
  from_bytes_be := (Impl.from_bytes_be_hoisted)
  from_bytes_le := (Impl.from_bytes_le_hoisted)

abbrev FpE :
  Type :=
  (math.field.element.FieldElement math.field.goldilocks.GoldilocksField)

@[spec]
def Impl_2.from_base_type_hoisted
    (x :
    (RustArray
    (math.field.element.FieldElement math.field.goldilocks.GoldilocksField)
    2)) :
    RustM
    (RustArray
    (math.field.element.FieldElement math.field.goldilocks.GoldilocksField)
    2)
    := do
  (pure x)

--  Field element type for the quadratic extension of native Goldilocks
abbrev Fp2E :
  Type :=
  (math.field.element.FieldElement Degree2GoldilocksExtensionField)

@[spec]
def Impl_5.from_base_type_hoisted
    (x :
    (RustArray
    (math.field.element.FieldElement math.field.goldilocks.GoldilocksField)
    3)) :
    RustM
    (RustArray
    (math.field.element.FieldElement math.field.goldilocks.GoldilocksField)
    3)
    := do
  (pure x)

--  Field element type for the cubic extension of native Goldilocks
abbrev Fp3E :
  Type :=
  (math.field.element.FieldElement Degree3GoldilocksExtensionField)

end math.field.extensions_goldilocks


namespace math.field.goldilocks

--  Type alias for Goldilocks field elements
abbrev GoldilocksElement :
  Type :=
  (math.field.element.FieldElement GoldilocksField)

end math.field.goldilocks


namespace math.field.traits

--  Represents the subfield relation between two fields.
class IsSubFieldOf.AssociatedTypes (Self : Type) (F : Type) where
  [trait_constr_IsSubFieldOf_i0 : IsField.AssociatedTypes Self]
  [trait_constr_IsSubFieldOf_i1 : IsField.AssociatedTypes F]

attribute [instance_reducible, instance]
  IsSubFieldOf.AssociatedTypes.trait_constr_IsSubFieldOf_i0

attribute [instance_reducible, instance]
  IsSubFieldOf.AssociatedTypes.trait_constr_IsSubFieldOf_i1

class IsSubFieldOf (Self : Type) (F : Type)
  [associatedTypes : outParam (IsSubFieldOf.AssociatedTypes (Self : Type) (F :
      Type))]
  where
  [trait_constr_IsSubFieldOf_i0 : IsField Self]
  [trait_constr_IsSubFieldOf_i1 : IsField F]
  mul (Self) (F) :
    ((IsField.BaseType Self) ->
    (IsField.BaseType F) ->
    RustM (IsField.BaseType F))
  add (Self) (F) :
    ((IsField.BaseType Self) ->
    (IsField.BaseType F) ->
    RustM (IsField.BaseType F))
  div (Self) (F) :
    ((IsField.BaseType Self) ->
    (IsField.BaseType F) ->
    RustM (core_models.result.Result
      (IsField.BaseType F)
      math.field.errors.FieldError))
  sub (Self) (F) :
    ((IsField.BaseType Self) ->
    (IsField.BaseType F) ->
    RustM (IsField.BaseType F))
  embed (Self) (F) : ((IsField.BaseType Self) -> RustM (IsField.BaseType F))
  to_subfield_vec (Self) (F) :
    ((IsField.BaseType F) ->
    RustM (alloc.vec.Vec (IsField.BaseType Self) alloc.alloc.Global))

attribute [instance_reducible, instance]
  IsSubFieldOf.trait_constr_IsSubFieldOf_i0

attribute [instance_reducible, instance]
  IsSubFieldOf.trait_constr_IsSubFieldOf_i1

end math.field.traits


namespace math.field.element

@[spec]
def Impl.to_subfield_vec
    (F : Type)
    (S : Type)
    [trait_constr_to_subfield_vec_associated_type_i0 :
      math.field.traits.IsField.AssociatedTypes
      F]
    [trait_constr_to_subfield_vec_i0 : math.field.traits.IsField F ]
    [trait_constr_to_subfield_vec_associated_type_i1 :
      math.field.traits.IsSubFieldOf.AssociatedTypes
      S
      F]
    [trait_constr_to_subfield_vec_i1 : math.field.traits.IsSubFieldOf S F ]
    (self : (FieldElement F)) :
    RustM (alloc.vec.Vec (FieldElement S) alloc.alloc.Global) := do
  let
    raws : (alloc.vec.Vec
      (math.field.traits.IsField.BaseType S)
      alloc.alloc.Global) ←
    (math.field.traits.IsSubFieldOf.to_subfield_vec
      S
      F (FieldElement.value self));
  let out : (alloc.vec.Vec (FieldElement S) alloc.alloc.Global) ←
    (alloc.vec.Impl.with_capacity (FieldElement S)
      (← (alloc.vec.Impl_1.len
        (math.field.traits.IsField.BaseType S)
        alloc.alloc.Global raws)));
  let out : (alloc.vec.Vec (FieldElement S) alloc.alloc.Global) ←
    (core_models.iter.traits.iterator.Iterator.fold
      (← (core_models.iter.traits.collect.IntoIterator.into_iter
        (alloc.vec.Vec
          (math.field.traits.IsField.BaseType S)
          alloc.alloc.Global) raws))
      out
      (fun out x =>
        (do
        (alloc.vec.Impl_1.push (FieldElement S) alloc.alloc.Global
          out
          (← (Impl_6.from_raw S x))) :
        RustM (alloc.vec.Vec (FieldElement S) alloc.alloc.Global))));
  (pure out)

@[spec]
def Impl_9.add_hoisted
    (F : Type)
    (L : Type)
    [trait_constr_add_hoisted_associated_type_i0 :
      math.field.traits.IsSubFieldOf.AssociatedTypes
      F
      L]
    [trait_constr_add_hoisted_i0 : math.field.traits.IsSubFieldOf F L ]
    [trait_constr_add_hoisted_associated_type_i1 :
      math.field.traits.IsField.AssociatedTypes
      L]
    [trait_constr_add_hoisted_i1 : math.field.traits.IsField L ]
    (self : (FieldElement F))
    (rhs : (FieldElement L)) :
    RustM (FieldElement L) := do
  (pure (FieldElement.mk
    (value := (← (math.field.traits.IsSubFieldOf.add
      F
      L (FieldElement.value self) (FieldElement.value rhs))))))

--  Addition operator overloading for field elements
@[reducible] instance Impl_9.AssociatedTypes
  (F : Type)
  (L : Type)
  [trait_constr_Impl_9_associated_type_i0 :
    math.field.traits.IsSubFieldOf.AssociatedTypes
    F
    L]
  [trait_constr_Impl_9_i0 : math.field.traits.IsSubFieldOf F L ]
  [trait_constr_Impl_9_associated_type_i1 :
    math.field.traits.IsField.AssociatedTypes
    L]
  [trait_constr_Impl_9_i1 : math.field.traits.IsField L ] :
  core_models.ops.arith.Add.AssociatedTypes (FieldElement F) (FieldElement L)
  where
  Output := (FieldElement L)

instance Impl_9
  (F : Type)
  (L : Type)
  [trait_constr_Impl_9_associated_type_i0 :
    math.field.traits.IsSubFieldOf.AssociatedTypes
    F
    L]
  [trait_constr_Impl_9_i0 : math.field.traits.IsSubFieldOf F L ]
  [trait_constr_Impl_9_associated_type_i1 :
    math.field.traits.IsField.AssociatedTypes
    L]
  [trait_constr_Impl_9_i1 : math.field.traits.IsField L ] :
  core_models.ops.arith.Add (FieldElement F) (FieldElement L)
  where
  add := (Impl_9.add_hoisted F L)

@[spec]
def Impl_10.add_hoisted
    (F : Type)
    (L : Type)
    [trait_constr_add_hoisted_associated_type_i0 :
      math.field.traits.IsSubFieldOf.AssociatedTypes
      F
      L]
    [trait_constr_add_hoisted_i0 : math.field.traits.IsSubFieldOf F L ]
    [trait_constr_add_hoisted_associated_type_i1 :
      math.field.traits.IsField.AssociatedTypes
      L]
    [trait_constr_add_hoisted_i1 : math.field.traits.IsField L ]
    (self : (FieldElement F))
    (rhs : (FieldElement L)) :
    RustM (FieldElement L) := do
  (core_models.field.element.Impl_10.add_hoisted (FieldElement L) self rhs)

@[reducible] instance Impl_10.AssociatedTypes
  (F : Type)
  (L : Type)
  [trait_constr_Impl_10_associated_type_i0 :
    math.field.traits.IsSubFieldOf.AssociatedTypes
    F
    L]
  [trait_constr_Impl_10_i0 : math.field.traits.IsSubFieldOf F L ]
  [trait_constr_Impl_10_associated_type_i1 :
    math.field.traits.IsField.AssociatedTypes
    L]
  [trait_constr_Impl_10_i1 : math.field.traits.IsField L ] :
  core_models.ops.arith.Add.AssociatedTypes (FieldElement F) (FieldElement L)
  where
  Output := (FieldElement L)

instance Impl_10
  (F : Type)
  (L : Type)
  [trait_constr_Impl_10_associated_type_i0 :
    math.field.traits.IsSubFieldOf.AssociatedTypes
    F
    L]
  [trait_constr_Impl_10_i0 : math.field.traits.IsSubFieldOf F L ]
  [trait_constr_Impl_10_associated_type_i1 :
    math.field.traits.IsField.AssociatedTypes
    L]
  [trait_constr_Impl_10_i1 : math.field.traits.IsField L ] :
  core_models.ops.arith.Add (FieldElement F) (FieldElement L)
  where
  add := (Impl_10.add_hoisted F L)

@[spec]
def Impl_11.add_hoisted
    (F : Type)
    (L : Type)
    [trait_constr_add_hoisted_associated_type_i0 :
      math.field.traits.IsSubFieldOf.AssociatedTypes
      F
      L]
    [trait_constr_add_hoisted_i0 : math.field.traits.IsSubFieldOf F L ]
    [trait_constr_add_hoisted_associated_type_i1 :
      math.field.traits.IsField.AssociatedTypes
      L]
    [trait_constr_add_hoisted_i1 : math.field.traits.IsField L ]
    (self : (FieldElement F))
    (rhs : (FieldElement L)) :
    RustM (FieldElement L) := do
  (core_models.field.element.Impl_11.add_hoisted (FieldElement L) self rhs)

@[reducible] instance Impl_11.AssociatedTypes
  (F : Type)
  (L : Type)
  [trait_constr_Impl_11_associated_type_i0 :
    math.field.traits.IsSubFieldOf.AssociatedTypes
    F
    L]
  [trait_constr_Impl_11_i0 : math.field.traits.IsSubFieldOf F L ]
  [trait_constr_Impl_11_associated_type_i1 :
    math.field.traits.IsField.AssociatedTypes
    L]
  [trait_constr_Impl_11_i1 : math.field.traits.IsField L ] :
  core_models.ops.arith.Add.AssociatedTypes (FieldElement F) (FieldElement L)
  where
  Output := (FieldElement L)

instance Impl_11
  (F : Type)
  (L : Type)
  [trait_constr_Impl_11_associated_type_i0 :
    math.field.traits.IsSubFieldOf.AssociatedTypes
    F
    L]
  [trait_constr_Impl_11_i0 : math.field.traits.IsSubFieldOf F L ]
  [trait_constr_Impl_11_associated_type_i1 :
    math.field.traits.IsField.AssociatedTypes
    L]
  [trait_constr_Impl_11_i1 : math.field.traits.IsField L ] :
  core_models.ops.arith.Add (FieldElement F) (FieldElement L)
  where
  add := (Impl_11.add_hoisted F L)

@[spec]
def Impl_12.add_hoisted
    (F : Type)
    (L : Type)
    [trait_constr_add_hoisted_associated_type_i0 :
      math.field.traits.IsSubFieldOf.AssociatedTypes
      F
      L]
    [trait_constr_add_hoisted_i0 : math.field.traits.IsSubFieldOf F L ]
    [trait_constr_add_hoisted_associated_type_i1 :
      math.field.traits.IsField.AssociatedTypes
      L]
    [trait_constr_add_hoisted_i1 : math.field.traits.IsField L ]
    (self : (FieldElement F))
    (rhs : (FieldElement L)) :
    RustM (FieldElement L) := do
  (core_models.field.element.Impl_12.add_hoisted (FieldElement L) self rhs)

@[reducible] instance Impl_12.AssociatedTypes
  (F : Type)
  (L : Type)
  [trait_constr_Impl_12_associated_type_i0 :
    math.field.traits.IsSubFieldOf.AssociatedTypes
    F
    L]
  [trait_constr_Impl_12_i0 : math.field.traits.IsSubFieldOf F L ]
  [trait_constr_Impl_12_associated_type_i1 :
    math.field.traits.IsField.AssociatedTypes
    L]
  [trait_constr_Impl_12_i1 : math.field.traits.IsField L ] :
  core_models.ops.arith.Add.AssociatedTypes (FieldElement F) (FieldElement L)
  where
  Output := (FieldElement L)

instance Impl_12
  (F : Type)
  (L : Type)
  [trait_constr_Impl_12_associated_type_i0 :
    math.field.traits.IsSubFieldOf.AssociatedTypes
    F
    L]
  [trait_constr_Impl_12_i0 : math.field.traits.IsSubFieldOf F L ]
  [trait_constr_Impl_12_associated_type_i1 :
    math.field.traits.IsField.AssociatedTypes
    L]
  [trait_constr_Impl_12_i1 : math.field.traits.IsField L ] :
  core_models.ops.arith.Add (FieldElement F) (FieldElement L)
  where
  add := (Impl_12.add_hoisted F L)

@[spec]
def Impl_13.add_assign_hoisted
    (F : Type)
    (L : Type)
    [trait_constr_add_assign_hoisted_associated_type_i0 :
      math.field.traits.IsSubFieldOf.AssociatedTypes
      F
      L]
    [trait_constr_add_assign_hoisted_i0 : math.field.traits.IsSubFieldOf F L ]
    [trait_constr_add_assign_hoisted_associated_type_i1 :
      math.field.traits.IsField.AssociatedTypes
      L]
    [trait_constr_add_assign_hoisted_i1 : math.field.traits.IsField L ]
    (self : (FieldElement L))
    (rhs : (FieldElement F)) :
    RustM (FieldElement L) := do
  let self : (FieldElement L) :=
    {self
    with value := (← (math.field.traits.IsSubFieldOf.add
      F
      L (FieldElement.value rhs) (FieldElement.value self)))};
  (pure self)

--  AddAssign operator overloading for field elements
@[reducible] instance Impl_13.AssociatedTypes
  (F : Type)
  (L : Type)
  [trait_constr_Impl_13_associated_type_i0 :
    math.field.traits.IsSubFieldOf.AssociatedTypes
    F
    L]
  [trait_constr_Impl_13_i0 : math.field.traits.IsSubFieldOf F L ]
  [trait_constr_Impl_13_associated_type_i1 :
    math.field.traits.IsField.AssociatedTypes
    L]
  [trait_constr_Impl_13_i1 : math.field.traits.IsField L ] :
  core_models.ops.arith.AddAssign.AssociatedTypes
  (FieldElement L)
  (FieldElement F)
  where

instance Impl_13
  (F : Type)
  (L : Type)
  [trait_constr_Impl_13_associated_type_i0 :
    math.field.traits.IsSubFieldOf.AssociatedTypes
    F
    L]
  [trait_constr_Impl_13_i0 : math.field.traits.IsSubFieldOf F L ]
  [trait_constr_Impl_13_associated_type_i1 :
    math.field.traits.IsField.AssociatedTypes
    L]
  [trait_constr_Impl_13_i1 : math.field.traits.IsField L ] :
  core_models.ops.arith.AddAssign (FieldElement L) (FieldElement F)
  where
  add_assign := (Impl_13.add_assign_hoisted F L)

@[spec]
def Impl_15.sub_hoisted
    (F : Type)
    (L : Type)
    [trait_constr_sub_hoisted_associated_type_i0 :
      math.field.traits.IsSubFieldOf.AssociatedTypes
      F
      L]
    [trait_constr_sub_hoisted_i0 : math.field.traits.IsSubFieldOf F L ]
    [trait_constr_sub_hoisted_associated_type_i1 :
      math.field.traits.IsField.AssociatedTypes
      L]
    [trait_constr_sub_hoisted_i1 : math.field.traits.IsField L ]
    (self : (FieldElement F))
    (rhs : (FieldElement L)) :
    RustM (FieldElement L) := do
  (pure (FieldElement.mk
    (value := (← (math.field.traits.IsSubFieldOf.sub
      F
      L (FieldElement.value self) (FieldElement.value rhs))))))

--  Subtraction operator overloading for field elements*/
@[reducible] instance Impl_15.AssociatedTypes
  (F : Type)
  (L : Type)
  [trait_constr_Impl_15_associated_type_i0 :
    math.field.traits.IsSubFieldOf.AssociatedTypes
    F
    L]
  [trait_constr_Impl_15_i0 : math.field.traits.IsSubFieldOf F L ]
  [trait_constr_Impl_15_associated_type_i1 :
    math.field.traits.IsField.AssociatedTypes
    L]
  [trait_constr_Impl_15_i1 : math.field.traits.IsField L ] :
  core_models.ops.arith.Sub.AssociatedTypes (FieldElement F) (FieldElement L)
  where
  Output := (FieldElement L)

instance Impl_15
  (F : Type)
  (L : Type)
  [trait_constr_Impl_15_associated_type_i0 :
    math.field.traits.IsSubFieldOf.AssociatedTypes
    F
    L]
  [trait_constr_Impl_15_i0 : math.field.traits.IsSubFieldOf F L ]
  [trait_constr_Impl_15_associated_type_i1 :
    math.field.traits.IsField.AssociatedTypes
    L]
  [trait_constr_Impl_15_i1 : math.field.traits.IsField L ] :
  core_models.ops.arith.Sub (FieldElement F) (FieldElement L)
  where
  sub := (Impl_15.sub_hoisted F L)

@[spec]
def Impl_16.sub_hoisted
    (F : Type)
    (L : Type)
    [trait_constr_sub_hoisted_associated_type_i0 :
      math.field.traits.IsSubFieldOf.AssociatedTypes
      F
      L]
    [trait_constr_sub_hoisted_i0 : math.field.traits.IsSubFieldOf F L ]
    [trait_constr_sub_hoisted_associated_type_i1 :
      math.field.traits.IsField.AssociatedTypes
      L]
    [trait_constr_sub_hoisted_i1 : math.field.traits.IsField L ]
    (self : (FieldElement F))
    (rhs : (FieldElement L)) :
    RustM (FieldElement L) := do
  (core_models.field.element.Impl_16.sub_hoisted (FieldElement L) self rhs)

@[reducible] instance Impl_16.AssociatedTypes
  (F : Type)
  (L : Type)
  [trait_constr_Impl_16_associated_type_i0 :
    math.field.traits.IsSubFieldOf.AssociatedTypes
    F
    L]
  [trait_constr_Impl_16_i0 : math.field.traits.IsSubFieldOf F L ]
  [trait_constr_Impl_16_associated_type_i1 :
    math.field.traits.IsField.AssociatedTypes
    L]
  [trait_constr_Impl_16_i1 : math.field.traits.IsField L ] :
  core_models.ops.arith.Sub.AssociatedTypes (FieldElement F) (FieldElement L)
  where
  Output := (FieldElement L)

instance Impl_16
  (F : Type)
  (L : Type)
  [trait_constr_Impl_16_associated_type_i0 :
    math.field.traits.IsSubFieldOf.AssociatedTypes
    F
    L]
  [trait_constr_Impl_16_i0 : math.field.traits.IsSubFieldOf F L ]
  [trait_constr_Impl_16_associated_type_i1 :
    math.field.traits.IsField.AssociatedTypes
    L]
  [trait_constr_Impl_16_i1 : math.field.traits.IsField L ] :
  core_models.ops.arith.Sub (FieldElement F) (FieldElement L)
  where
  sub := (Impl_16.sub_hoisted F L)

@[spec]
def Impl_17.sub_hoisted
    (F : Type)
    (L : Type)
    [trait_constr_sub_hoisted_associated_type_i0 :
      math.field.traits.IsSubFieldOf.AssociatedTypes
      F
      L]
    [trait_constr_sub_hoisted_i0 : math.field.traits.IsSubFieldOf F L ]
    [trait_constr_sub_hoisted_associated_type_i1 :
      math.field.traits.IsField.AssociatedTypes
      L]
    [trait_constr_sub_hoisted_i1 : math.field.traits.IsField L ]
    (self : (FieldElement F))
    (rhs : (FieldElement L)) :
    RustM (FieldElement L) := do
  (core_models.field.element.Impl_17.sub_hoisted (FieldElement L) self rhs)

@[reducible] instance Impl_17.AssociatedTypes
  (F : Type)
  (L : Type)
  [trait_constr_Impl_17_associated_type_i0 :
    math.field.traits.IsSubFieldOf.AssociatedTypes
    F
    L]
  [trait_constr_Impl_17_i0 : math.field.traits.IsSubFieldOf F L ]
  [trait_constr_Impl_17_associated_type_i1 :
    math.field.traits.IsField.AssociatedTypes
    L]
  [trait_constr_Impl_17_i1 : math.field.traits.IsField L ] :
  core_models.ops.arith.Sub.AssociatedTypes (FieldElement F) (FieldElement L)
  where
  Output := (FieldElement L)

instance Impl_17
  (F : Type)
  (L : Type)
  [trait_constr_Impl_17_associated_type_i0 :
    math.field.traits.IsSubFieldOf.AssociatedTypes
    F
    L]
  [trait_constr_Impl_17_i0 : math.field.traits.IsSubFieldOf F L ]
  [trait_constr_Impl_17_associated_type_i1 :
    math.field.traits.IsField.AssociatedTypes
    L]
  [trait_constr_Impl_17_i1 : math.field.traits.IsField L ] :
  core_models.ops.arith.Sub (FieldElement F) (FieldElement L)
  where
  sub := (Impl_17.sub_hoisted F L)

@[spec]
def Impl_18.sub_hoisted
    (F : Type)
    (L : Type)
    [trait_constr_sub_hoisted_associated_type_i0 :
      math.field.traits.IsSubFieldOf.AssociatedTypes
      F
      L]
    [trait_constr_sub_hoisted_i0 : math.field.traits.IsSubFieldOf F L ]
    [trait_constr_sub_hoisted_associated_type_i1 :
      math.field.traits.IsField.AssociatedTypes
      L]
    [trait_constr_sub_hoisted_i1 : math.field.traits.IsField L ]
    (self : (FieldElement F))
    (rhs : (FieldElement L)) :
    RustM (FieldElement L) := do
  (core_models.field.element.Impl_18.sub_hoisted (FieldElement L) self rhs)

@[reducible] instance Impl_18.AssociatedTypes
  (F : Type)
  (L : Type)
  [trait_constr_Impl_18_associated_type_i0 :
    math.field.traits.IsSubFieldOf.AssociatedTypes
    F
    L]
  [trait_constr_Impl_18_i0 : math.field.traits.IsSubFieldOf F L ]
  [trait_constr_Impl_18_associated_type_i1 :
    math.field.traits.IsField.AssociatedTypes
    L]
  [trait_constr_Impl_18_i1 : math.field.traits.IsField L ] :
  core_models.ops.arith.Sub.AssociatedTypes (FieldElement F) (FieldElement L)
  where
  Output := (FieldElement L)

instance Impl_18
  (F : Type)
  (L : Type)
  [trait_constr_Impl_18_associated_type_i0 :
    math.field.traits.IsSubFieldOf.AssociatedTypes
    F
    L]
  [trait_constr_Impl_18_i0 : math.field.traits.IsSubFieldOf F L ]
  [trait_constr_Impl_18_associated_type_i1 :
    math.field.traits.IsField.AssociatedTypes
    L]
  [trait_constr_Impl_18_i1 : math.field.traits.IsField L ] :
  core_models.ops.arith.Sub (FieldElement F) (FieldElement L)
  where
  sub := (Impl_18.sub_hoisted F L)

@[spec]
def Impl_19.mul_hoisted
    (F : Type)
    (L : Type)
    [trait_constr_mul_hoisted_associated_type_i0 :
      math.field.traits.IsSubFieldOf.AssociatedTypes
      F
      L]
    [trait_constr_mul_hoisted_i0 : math.field.traits.IsSubFieldOf F L ]
    [trait_constr_mul_hoisted_associated_type_i1 :
      math.field.traits.IsField.AssociatedTypes
      L]
    [trait_constr_mul_hoisted_i1 : math.field.traits.IsField L ]
    (self : (FieldElement F))
    (rhs : (FieldElement L)) :
    RustM (FieldElement L) := do
  (pure (FieldElement.mk
    (value := (← (math.field.traits.IsSubFieldOf.mul
      F
      L (FieldElement.value self) (FieldElement.value rhs))))))

--  Multiplication operator overloading for field elements*/
@[reducible] instance Impl_19.AssociatedTypes
  (F : Type)
  (L : Type)
  [trait_constr_Impl_19_associated_type_i0 :
    math.field.traits.IsSubFieldOf.AssociatedTypes
    F
    L]
  [trait_constr_Impl_19_i0 : math.field.traits.IsSubFieldOf F L ]
  [trait_constr_Impl_19_associated_type_i1 :
    math.field.traits.IsField.AssociatedTypes
    L]
  [trait_constr_Impl_19_i1 : math.field.traits.IsField L ] :
  core_models.ops.arith.Mul.AssociatedTypes (FieldElement F) (FieldElement L)
  where
  Output := (FieldElement L)

instance Impl_19
  (F : Type)
  (L : Type)
  [trait_constr_Impl_19_associated_type_i0 :
    math.field.traits.IsSubFieldOf.AssociatedTypes
    F
    L]
  [trait_constr_Impl_19_i0 : math.field.traits.IsSubFieldOf F L ]
  [trait_constr_Impl_19_associated_type_i1 :
    math.field.traits.IsField.AssociatedTypes
    L]
  [trait_constr_Impl_19_i1 : math.field.traits.IsField L ] :
  core_models.ops.arith.Mul (FieldElement F) (FieldElement L)
  where
  mul := (Impl_19.mul_hoisted F L)

@[spec]
def Impl_20.mul_hoisted
    (F : Type)
    (L : Type)
    [trait_constr_mul_hoisted_associated_type_i0 :
      math.field.traits.IsSubFieldOf.AssociatedTypes
      F
      L]
    [trait_constr_mul_hoisted_i0 : math.field.traits.IsSubFieldOf F L ]
    [trait_constr_mul_hoisted_associated_type_i1 :
      math.field.traits.IsField.AssociatedTypes
      L]
    [trait_constr_mul_hoisted_i1 : math.field.traits.IsField L ]
    (self : (FieldElement F))
    (rhs : (FieldElement L)) :
    RustM (FieldElement L) := do
  (core_models.field.element.Impl_20.mul_hoisted (FieldElement L) self rhs)

@[reducible] instance Impl_20.AssociatedTypes
  (F : Type)
  (L : Type)
  [trait_constr_Impl_20_associated_type_i0 :
    math.field.traits.IsSubFieldOf.AssociatedTypes
    F
    L]
  [trait_constr_Impl_20_i0 : math.field.traits.IsSubFieldOf F L ]
  [trait_constr_Impl_20_associated_type_i1 :
    math.field.traits.IsField.AssociatedTypes
    L]
  [trait_constr_Impl_20_i1 : math.field.traits.IsField L ] :
  core_models.ops.arith.Mul.AssociatedTypes (FieldElement F) (FieldElement L)
  where
  Output := (FieldElement L)

instance Impl_20
  (F : Type)
  (L : Type)
  [trait_constr_Impl_20_associated_type_i0 :
    math.field.traits.IsSubFieldOf.AssociatedTypes
    F
    L]
  [trait_constr_Impl_20_i0 : math.field.traits.IsSubFieldOf F L ]
  [trait_constr_Impl_20_associated_type_i1 :
    math.field.traits.IsField.AssociatedTypes
    L]
  [trait_constr_Impl_20_i1 : math.field.traits.IsField L ] :
  core_models.ops.arith.Mul (FieldElement F) (FieldElement L)
  where
  mul := (Impl_20.mul_hoisted F L)

@[spec]
def Impl_21.mul_hoisted
    (F : Type)
    (L : Type)
    [trait_constr_mul_hoisted_associated_type_i0 :
      math.field.traits.IsSubFieldOf.AssociatedTypes
      F
      L]
    [trait_constr_mul_hoisted_i0 : math.field.traits.IsSubFieldOf F L ]
    [trait_constr_mul_hoisted_associated_type_i1 :
      math.field.traits.IsField.AssociatedTypes
      L]
    [trait_constr_mul_hoisted_i1 : math.field.traits.IsField L ]
    (self : (FieldElement F))
    (rhs : (FieldElement L)) :
    RustM (FieldElement L) := do
  (core_models.field.element.Impl_21.mul_hoisted (FieldElement L) self rhs)

@[reducible] instance Impl_21.AssociatedTypes
  (F : Type)
  (L : Type)
  [trait_constr_Impl_21_associated_type_i0 :
    math.field.traits.IsSubFieldOf.AssociatedTypes
    F
    L]
  [trait_constr_Impl_21_i0 : math.field.traits.IsSubFieldOf F L ]
  [trait_constr_Impl_21_associated_type_i1 :
    math.field.traits.IsField.AssociatedTypes
    L]
  [trait_constr_Impl_21_i1 : math.field.traits.IsField L ] :
  core_models.ops.arith.Mul.AssociatedTypes (FieldElement F) (FieldElement L)
  where
  Output := (FieldElement L)

instance Impl_21
  (F : Type)
  (L : Type)
  [trait_constr_Impl_21_associated_type_i0 :
    math.field.traits.IsSubFieldOf.AssociatedTypes
    F
    L]
  [trait_constr_Impl_21_i0 : math.field.traits.IsSubFieldOf F L ]
  [trait_constr_Impl_21_associated_type_i1 :
    math.field.traits.IsField.AssociatedTypes
    L]
  [trait_constr_Impl_21_i1 : math.field.traits.IsField L ] :
  core_models.ops.arith.Mul (FieldElement F) (FieldElement L)
  where
  mul := (Impl_21.mul_hoisted F L)

@[spec]
def Impl_22.mul_hoisted
    (F : Type)
    (L : Type)
    [trait_constr_mul_hoisted_associated_type_i0 :
      math.field.traits.IsSubFieldOf.AssociatedTypes
      F
      L]
    [trait_constr_mul_hoisted_i0 : math.field.traits.IsSubFieldOf F L ]
    [trait_constr_mul_hoisted_associated_type_i1 :
      math.field.traits.IsField.AssociatedTypes
      L]
    [trait_constr_mul_hoisted_i1 : math.field.traits.IsField L ]
    (self : (FieldElement F))
    (rhs : (FieldElement L)) :
    RustM (FieldElement L) := do
  (core_models.field.element.Impl_22.mul_hoisted (FieldElement L) self rhs)

@[reducible] instance Impl_22.AssociatedTypes
  (F : Type)
  (L : Type)
  [trait_constr_Impl_22_associated_type_i0 :
    math.field.traits.IsSubFieldOf.AssociatedTypes
    F
    L]
  [trait_constr_Impl_22_i0 : math.field.traits.IsSubFieldOf F L ]
  [trait_constr_Impl_22_associated_type_i1 :
    math.field.traits.IsField.AssociatedTypes
    L]
  [trait_constr_Impl_22_i1 : math.field.traits.IsField L ] :
  core_models.ops.arith.Mul.AssociatedTypes (FieldElement F) (FieldElement L)
  where
  Output := (FieldElement L)

instance Impl_22
  (F : Type)
  (L : Type)
  [trait_constr_Impl_22_associated_type_i0 :
    math.field.traits.IsSubFieldOf.AssociatedTypes
    F
    L]
  [trait_constr_Impl_22_i0 : math.field.traits.IsSubFieldOf F L ]
  [trait_constr_Impl_22_associated_type_i1 :
    math.field.traits.IsField.AssociatedTypes
    L]
  [trait_constr_Impl_22_i1 : math.field.traits.IsField L ] :
  core_models.ops.arith.Mul (FieldElement F) (FieldElement L)
  where
  mul := (Impl_22.mul_hoisted F L)

@[spec]
def Impl_23.mul_assign_hoisted
    (F : Type)
    (L : Type)
    [trait_constr_mul_assign_hoisted_associated_type_i0 :
      math.field.traits.IsSubFieldOf.AssociatedTypes
      F
      L]
    [trait_constr_mul_assign_hoisted_i0 : math.field.traits.IsSubFieldOf F L ]
    [trait_constr_mul_assign_hoisted_associated_type_i1 :
      math.field.traits.IsField.AssociatedTypes
      L]
    [trait_constr_mul_assign_hoisted_i1 : math.field.traits.IsField L ]
    (self : (FieldElement L))
    (rhs : (FieldElement F)) :
    RustM (FieldElement L) := do
  let self : (FieldElement L) :=
    {self
    with value := (← (math.field.traits.IsSubFieldOf.mul
      F
      L (FieldElement.value rhs) (FieldElement.value self)))};
  (pure self)

--  MulAssign operator overloading for field elements
@[reducible] instance Impl_23.AssociatedTypes
  (F : Type)
  (L : Type)
  [trait_constr_Impl_23_associated_type_i0 :
    math.field.traits.IsSubFieldOf.AssociatedTypes
    F
    L]
  [trait_constr_Impl_23_i0 : math.field.traits.IsSubFieldOf F L ]
  [trait_constr_Impl_23_associated_type_i1 :
    math.field.traits.IsField.AssociatedTypes
    L]
  [trait_constr_Impl_23_i1 : math.field.traits.IsField L ] :
  core_models.ops.arith.MulAssign.AssociatedTypes
  (FieldElement L)
  (FieldElement F)
  where

instance Impl_23
  (F : Type)
  (L : Type)
  [trait_constr_Impl_23_associated_type_i0 :
    math.field.traits.IsSubFieldOf.AssociatedTypes
    F
    L]
  [trait_constr_Impl_23_i0 : math.field.traits.IsSubFieldOf F L ]
  [trait_constr_Impl_23_associated_type_i1 :
    math.field.traits.IsField.AssociatedTypes
    L]
  [trait_constr_Impl_23_i1 : math.field.traits.IsField L ] :
  core_models.ops.arith.MulAssign (FieldElement L) (FieldElement F)
  where
  mul_assign := (Impl_23.mul_assign_hoisted F L)

@[spec]
def Impl_24.mul_assign_hoisted
    (F : Type)
    (L : Type)
    [trait_constr_mul_assign_hoisted_associated_type_i0 :
      math.field.traits.IsSubFieldOf.AssociatedTypes
      F
      L]
    [trait_constr_mul_assign_hoisted_i0 : math.field.traits.IsSubFieldOf F L ]
    [trait_constr_mul_assign_hoisted_associated_type_i1 :
      math.field.traits.IsField.AssociatedTypes
      L]
    [trait_constr_mul_assign_hoisted_i1 : math.field.traits.IsField L ]
    (self : (FieldElement L))
    (rhs : (FieldElement F)) :
    RustM (FieldElement L) := do
  let self : (FieldElement L) :=
    {self
    with value := (← (math.field.traits.IsSubFieldOf.mul
      F
      L (FieldElement.value rhs) (FieldElement.value self)))};
  (pure self)

--  MulAssign operator overloading for field elements
@[reducible] instance Impl_24.AssociatedTypes
  (F : Type)
  (L : Type)
  [trait_constr_Impl_24_associated_type_i0 :
    math.field.traits.IsSubFieldOf.AssociatedTypes
    F
    L]
  [trait_constr_Impl_24_i0 : math.field.traits.IsSubFieldOf F L ]
  [trait_constr_Impl_24_associated_type_i1 :
    math.field.traits.IsField.AssociatedTypes
    L]
  [trait_constr_Impl_24_i1 : math.field.traits.IsField L ] :
  core_models.ops.arith.MulAssign.AssociatedTypes
  (FieldElement L)
  (FieldElement F)
  where

instance Impl_24
  (F : Type)
  (L : Type)
  [trait_constr_Impl_24_associated_type_i0 :
    math.field.traits.IsSubFieldOf.AssociatedTypes
    F
    L]
  [trait_constr_Impl_24_i0 : math.field.traits.IsSubFieldOf F L ]
  [trait_constr_Impl_24_associated_type_i1 :
    math.field.traits.IsField.AssociatedTypes
    L]
  [trait_constr_Impl_24_i1 : math.field.traits.IsField L ] :
  core_models.ops.arith.MulAssign (FieldElement L) (FieldElement F)
  where
  mul_assign := (Impl_24.mul_assign_hoisted F L)

@[spec]
def Impl_25.div_hoisted
    (F : Type)
    (L : Type)
    [trait_constr_div_hoisted_associated_type_i0 :
      math.field.traits.IsSubFieldOf.AssociatedTypes
      F
      L]
    [trait_constr_div_hoisted_i0 : math.field.traits.IsSubFieldOf F L ]
    [trait_constr_div_hoisted_associated_type_i1 :
      math.field.traits.IsField.AssociatedTypes
      L]
    [trait_constr_div_hoisted_i1 : math.field.traits.IsField L ]
    (self : (FieldElement F))
    (rhs : (FieldElement L)) :
    RustM
    (core_models.result.Result (FieldElement L) math.field.errors.FieldError)
    := do
  match
    (← (math.field.traits.IsSubFieldOf.div
      F
      L (FieldElement.value self) (FieldElement.value rhs)))
  with
    | (core_models.result.Result.Ok  value) => do
      (pure (core_models.result.Result.Ok (FieldElement.mk (value := value))))
    | (core_models.result.Result.Err  err) => do
      (pure (core_models.result.Result.Err err))

--  Division operator overloading for field elements*/
@[reducible] instance Impl_25.AssociatedTypes
  (F : Type)
  (L : Type)
  [trait_constr_Impl_25_associated_type_i0 :
    math.field.traits.IsSubFieldOf.AssociatedTypes
    F
    L]
  [trait_constr_Impl_25_i0 : math.field.traits.IsSubFieldOf F L ]
  [trait_constr_Impl_25_associated_type_i1 :
    math.field.traits.IsField.AssociatedTypes
    L]
  [trait_constr_Impl_25_i1 : math.field.traits.IsField L ] :
  core_models.ops.arith.Div.AssociatedTypes (FieldElement F) (FieldElement L)
  where
  Output := (core_models.result.Result
    (FieldElement L)
    math.field.errors.FieldError)

instance Impl_25
  (F : Type)
  (L : Type)
  [trait_constr_Impl_25_associated_type_i0 :
    math.field.traits.IsSubFieldOf.AssociatedTypes
    F
    L]
  [trait_constr_Impl_25_i0 : math.field.traits.IsSubFieldOf F L ]
  [trait_constr_Impl_25_associated_type_i1 :
    math.field.traits.IsField.AssociatedTypes
    L]
  [trait_constr_Impl_25_i1 : math.field.traits.IsField L ] :
  core_models.ops.arith.Div (FieldElement F) (FieldElement L)
  where
  div := (Impl_25.div_hoisted F L)

@[spec]
def Impl_26.div_hoisted
    (F : Type)
    (L : Type)
    [trait_constr_div_hoisted_associated_type_i0 :
      math.field.traits.IsSubFieldOf.AssociatedTypes
      F
      L]
    [trait_constr_div_hoisted_i0 : math.field.traits.IsSubFieldOf F L ]
    [trait_constr_div_hoisted_associated_type_i1 :
      math.field.traits.IsField.AssociatedTypes
      L]
    [trait_constr_div_hoisted_i1 : math.field.traits.IsField L ]
    (self : (FieldElement F))
    (rhs : (FieldElement L)) :
    RustM
    (core_models.result.Result (FieldElement L) math.field.errors.FieldError)
    := do
  (core_models.field.element.Impl_26.div_hoisted (FieldElement L) self rhs)

@[reducible] instance Impl_26.AssociatedTypes
  (F : Type)
  (L : Type)
  [trait_constr_Impl_26_associated_type_i0 :
    math.field.traits.IsSubFieldOf.AssociatedTypes
    F
    L]
  [trait_constr_Impl_26_i0 : math.field.traits.IsSubFieldOf F L ]
  [trait_constr_Impl_26_associated_type_i1 :
    math.field.traits.IsField.AssociatedTypes
    L]
  [trait_constr_Impl_26_i1 : math.field.traits.IsField L ] :
  core_models.ops.arith.Div.AssociatedTypes (FieldElement F) (FieldElement L)
  where
  Output := (core_models.result.Result
    (FieldElement L)
    math.field.errors.FieldError)

instance Impl_26
  (F : Type)
  (L : Type)
  [trait_constr_Impl_26_associated_type_i0 :
    math.field.traits.IsSubFieldOf.AssociatedTypes
    F
    L]
  [trait_constr_Impl_26_i0 : math.field.traits.IsSubFieldOf F L ]
  [trait_constr_Impl_26_associated_type_i1 :
    math.field.traits.IsField.AssociatedTypes
    L]
  [trait_constr_Impl_26_i1 : math.field.traits.IsField L ] :
  core_models.ops.arith.Div (FieldElement F) (FieldElement L)
  where
  div := (Impl_26.div_hoisted F L)

@[spec]
def Impl_27.div_hoisted
    (F : Type)
    (L : Type)
    [trait_constr_div_hoisted_associated_type_i0 :
      math.field.traits.IsSubFieldOf.AssociatedTypes
      F
      L]
    [trait_constr_div_hoisted_i0 : math.field.traits.IsSubFieldOf F L ]
    [trait_constr_div_hoisted_associated_type_i1 :
      math.field.traits.IsField.AssociatedTypes
      L]
    [trait_constr_div_hoisted_i1 : math.field.traits.IsField L ]
    (self : (FieldElement F))
    (rhs : (FieldElement L)) :
    RustM
    (core_models.result.Result (FieldElement L) math.field.errors.FieldError)
    := do
  (core_models.field.element.Impl_27.div_hoisted (FieldElement L) self rhs)

@[reducible] instance Impl_27.AssociatedTypes
  (F : Type)
  (L : Type)
  [trait_constr_Impl_27_associated_type_i0 :
    math.field.traits.IsSubFieldOf.AssociatedTypes
    F
    L]
  [trait_constr_Impl_27_i0 : math.field.traits.IsSubFieldOf F L ]
  [trait_constr_Impl_27_associated_type_i1 :
    math.field.traits.IsField.AssociatedTypes
    L]
  [trait_constr_Impl_27_i1 : math.field.traits.IsField L ] :
  core_models.ops.arith.Div.AssociatedTypes (FieldElement F) (FieldElement L)
  where
  Output := (core_models.result.Result
    (FieldElement L)
    math.field.errors.FieldError)

instance Impl_27
  (F : Type)
  (L : Type)
  [trait_constr_Impl_27_associated_type_i0 :
    math.field.traits.IsSubFieldOf.AssociatedTypes
    F
    L]
  [trait_constr_Impl_27_i0 : math.field.traits.IsSubFieldOf F L ]
  [trait_constr_Impl_27_associated_type_i1 :
    math.field.traits.IsField.AssociatedTypes
    L]
  [trait_constr_Impl_27_i1 : math.field.traits.IsField L ] :
  core_models.ops.arith.Div (FieldElement F) (FieldElement L)
  where
  div := (Impl_27.div_hoisted F L)

@[spec]
def Impl_28.div_hoisted
    (F : Type)
    (L : Type)
    [trait_constr_div_hoisted_associated_type_i0 :
      math.field.traits.IsSubFieldOf.AssociatedTypes
      F
      L]
    [trait_constr_div_hoisted_i0 : math.field.traits.IsSubFieldOf F L ]
    [trait_constr_div_hoisted_associated_type_i1 :
      math.field.traits.IsField.AssociatedTypes
      L]
    [trait_constr_div_hoisted_i1 : math.field.traits.IsField L ]
    (self : (FieldElement F))
    (rhs : (FieldElement L)) :
    RustM
    (core_models.result.Result (FieldElement L) math.field.errors.FieldError)
    := do
  (core_models.field.element.Impl_28.div_hoisted (FieldElement L) self rhs)

@[reducible] instance Impl_28.AssociatedTypes
  (F : Type)
  (L : Type)
  [trait_constr_Impl_28_associated_type_i0 :
    math.field.traits.IsSubFieldOf.AssociatedTypes
    F
    L]
  [trait_constr_Impl_28_i0 : math.field.traits.IsSubFieldOf F L ]
  [trait_constr_Impl_28_associated_type_i1 :
    math.field.traits.IsField.AssociatedTypes
    L]
  [trait_constr_Impl_28_i1 : math.field.traits.IsField L ] :
  core_models.ops.arith.Div.AssociatedTypes (FieldElement F) (FieldElement L)
  where
  Output := (core_models.result.Result
    (FieldElement L)
    math.field.errors.FieldError)

instance Impl_28
  (F : Type)
  (L : Type)
  [trait_constr_Impl_28_associated_type_i0 :
    math.field.traits.IsSubFieldOf.AssociatedTypes
    F
    L]
  [trait_constr_Impl_28_i0 : math.field.traits.IsSubFieldOf F L ]
  [trait_constr_Impl_28_associated_type_i1 :
    math.field.traits.IsField.AssociatedTypes
    L]
  [trait_constr_Impl_28_i1 : math.field.traits.IsField L ] :
  core_models.ops.arith.Div (FieldElement F) (FieldElement L)
  where
  div := (Impl_28.div_hoisted F L)

@[spec]
def Impl_32.to_extension
    (F : Type)
    (L : Type)
    [trait_constr_to_extension_associated_type_i0 :
      math.field.traits.IsField.AssociatedTypes
      F]
    [trait_constr_to_extension_i0 : math.field.traits.IsField F ]
    [trait_constr_to_extension_associated_type_i1 :
      math.field.traits.IsField.AssociatedTypes
      L]
    [trait_constr_to_extension_i1 : math.field.traits.IsField L ]
    [trait_constr_to_extension_associated_type_i2 :
      math.field.traits.IsSubFieldOf.AssociatedTypes
      F
      L]
    [trait_constr_to_extension_i2 : math.field.traits.IsSubFieldOf F L ]
    (self : (FieldElement F)) :
    RustM (FieldElement L) := do
  (pure (FieldElement.mk
    (value := (← (math.field.traits.IsSubFieldOf.embed
      F
      L (FieldElement.value self))))))

--  Compute `self - rhs` where `rhs` is in a subfield `S` of `F`.
-- 
--  Uses mixed F-S arithmetic: computes `self - embed(rhs)` without
--  explicitly converting rhs to the extension field.
@[spec]
def Impl_32.sub_subfield
    (F : Type)
    (S : Type)
    [trait_constr_sub_subfield_associated_type_i0 :
      math.field.traits.IsField.AssociatedTypes
      F]
    [trait_constr_sub_subfield_i0 : math.field.traits.IsField F ]
    [trait_constr_sub_subfield_associated_type_i1 :
      math.field.traits.IsSubFieldOf.AssociatedTypes
      S
      F]
    [trait_constr_sub_subfield_i1 : math.field.traits.IsSubFieldOf S F ]
    (self : (FieldElement F))
    (rhs : (FieldElement S)) :
    RustM (FieldElement F) := do
  (pure (FieldElement.mk
    (value := (← (math.field.traits.IsField.neg
      F
      (← (math.field.traits.IsSubFieldOf.sub
        S
        F (FieldElement.value rhs) (FieldElement.value self))))))))

end math.field.element


namespace math.field.traits

@[spec]
def Impl.mul_hoisted
    (F : Type)
    [trait_constr_mul_hoisted_associated_type_i0 : IsField.AssociatedTypes F]
    [trait_constr_mul_hoisted_i0 : IsField F ]
    (a : (IsField.BaseType F))
    (b : (IsField.BaseType F)) :
    RustM (IsField.BaseType F) := do
  (IsField.mul F a b)

@[spec]
def Impl.add_hoisted
    (F : Type)
    [trait_constr_add_hoisted_associated_type_i0 : IsField.AssociatedTypes F]
    [trait_constr_add_hoisted_i0 : IsField F ]
    (a : (IsField.BaseType F))
    (b : (IsField.BaseType F)) :
    RustM (IsField.BaseType F) := do
  (IsField.add F a b)

@[spec]
def Impl.sub_hoisted
    (F : Type)
    [trait_constr_sub_hoisted_associated_type_i0 : IsField.AssociatedTypes F]
    [trait_constr_sub_hoisted_i0 : IsField F ]
    (a : (IsField.BaseType F))
    (b : (IsField.BaseType F)) :
    RustM (IsField.BaseType F) := do
  (IsField.sub F a b)

@[spec]
def Impl.div_hoisted
    (F : Type)
    [trait_constr_div_hoisted_associated_type_i0 : IsField.AssociatedTypes F]
    [trait_constr_div_hoisted_i0 : IsField F ]
    (a : (IsField.BaseType F))
    (b : (IsField.BaseType F)) :
    RustM
    (core_models.result.Result
      (IsField.BaseType F)
      math.field.errors.FieldError)
    := do
  (IsField.div F a b)

@[spec]
def Impl.embed_hoisted
    (F : Type)
    [trait_constr_embed_hoisted_associated_type_i0 : IsField.AssociatedTypes F]
    [trait_constr_embed_hoisted_i0 : IsField F ]
    (a : (IsField.BaseType F)) :
    RustM (IsField.BaseType F) := do
  (pure a)

@[spec]
def Impl.to_subfield_vec_hoisted
    (F : Type)
    [trait_constr_to_subfield_vec_hoisted_associated_type_i0 :
      IsField.AssociatedTypes
      F]
    [trait_constr_to_subfield_vec_hoisted_i0 : IsField F ]
    (b : (IsField.BaseType F)) :
    RustM (alloc.vec.Vec (IsField.BaseType F) alloc.alloc.Global) := do
  (alloc.slice.Impl.into_vec (IsField.BaseType F) alloc.alloc.Global
    (← (rust_primitives.unsize (RustArray.ofVec #v[b]))))

@[reducible] instance Impl.AssociatedTypes
  (F : Type)
  [trait_constr_Impl_associated_type_i0 : IsField.AssociatedTypes F]
  [trait_constr_Impl_i0 : IsField F ] :
  IsSubFieldOf.AssociatedTypes F F
  where

instance Impl
  (F : Type)
  [trait_constr_Impl_associated_type_i0 : IsField.AssociatedTypes F]
  [trait_constr_Impl_i0 : IsField F ] :
  IsSubFieldOf F F
  where
  mul := (Impl.mul_hoisted F)
  add := (Impl.add_hoisted F)
  sub := (Impl.sub_hoisted F)
  div := (Impl.div_hoisted F)
  embed := (Impl.embed_hoisted F)
  to_subfield_vec := (Impl.to_subfield_vec_hoisted F)

end math.field.traits


namespace math.field.element

@[spec]
def Impl.inplace_batch_inverse_sequential
    (F : Type)
    [trait_constr_inplace_batch_inverse_sequential_associated_type_i0 :
      math.field.traits.IsField.AssociatedTypes
      F]
    [trait_constr_inplace_batch_inverse_sequential_i0 :
      math.field.traits.IsField
      F
      ]
    (numbers : (RustSlice (FieldElement F))) :
    RustM
    (rust_primitives.hax.Tuple2
      (RustSlice (FieldElement F))
      (core_models.result.Result
        rust_primitives.hax.Tuple0
        math.field.errors.FieldError))
    := do
  if (← (core_models.slice.Impl.is_empty (FieldElement F) numbers)) then do
    (pure (rust_primitives.hax.Tuple2.mk
      numbers
      (core_models.result.Result.Ok rust_primitives.hax.Tuple0.mk)))
  else do
    let count : usize ← (core_models.slice.Impl.len (FieldElement F) numbers);
    let prod_prefix : (alloc.vec.Vec (FieldElement F) alloc.alloc.Global) ←
      (alloc.vec.Impl.with_capacity (FieldElement F) count);
    let prod_prefix : (alloc.vec.Vec (FieldElement F) alloc.alloc.Global) ←
      (alloc.vec.Impl_1.push (FieldElement F) alloc.alloc.Global
        prod_prefix
        (← (core_models.clone.Clone.clone
          (FieldElement F) (← numbers[(0 : usize)]_?))));
    let prod_prefix : (alloc.vec.Vec (FieldElement F) alloc.alloc.Global) ←
      (rust_primitives.hax.folds.fold_range
        (1 : usize)
        count
        (fun prod_prefix _ => (do (pure true) : RustM Bool))
        prod_prefix
        (fun prod_prefix i =>
          (do
          (alloc.vec.Impl_1.push (FieldElement F) alloc.alloc.Global
            prod_prefix
            (← (core_models.ops.arith.Mul.mul
              (FieldElement F)
              (FieldElement F)
              (← prod_prefix[(← (i -? (1 : usize)))]_?)
              (← numbers[i]_?)))) :
          RustM (alloc.vec.Vec (FieldElement F) alloc.alloc.Global))));
    match (← (Impl_32.inv F (← prod_prefix[(← (count -? (1 : usize)))]_?))) with
      | (core_models.result.Result.Ok  bi_inv) => do
        let ⟨bi_inv, numbers⟩ ←
          (core_models.iter.traits.iterator.Iterator.fold
            (← (core_models.iter.traits.collect.IntoIterator.into_iter
              (core_models.iter.adapters.rev.Rev
                (core_models.ops.range.Range usize))
              (← (core_models.iter.traits.iterator.Iterator.rev
                (core_models.ops.range.Range usize)
                (core_models.ops.range.Range.mk
                  (start := (1 : usize))
                  (_end := count))))))
            (rust_primitives.hax.Tuple2.mk bi_inv numbers)
            (fun ⟨bi_inv, numbers⟩ i =>
              (do
              let ai_inv : (FieldElement F) ←
                (core_models.ops.arith.Mul.mul
                  (FieldElement F)
                  (FieldElement F)
                  bi_inv
                  (← prod_prefix[(← (i -? (1 : usize)))]_?));
              let bi_inv : (FieldElement F) ←
                (core_models.ops.arith.Mul.mul
                  (FieldElement F)
                  (FieldElement F) bi_inv (← numbers[i]_?));
              let numbers : (RustSlice (FieldElement F)) ←
                (rust_primitives.hax.monomorphized_update_at.update_at_usize
                  numbers
                  i
                  ai_inv);
              (pure (rust_primitives.hax.Tuple2.mk bi_inv numbers)) :
              RustM
              (rust_primitives.hax.Tuple2
                (FieldElement F)
                (RustSlice (FieldElement F))))));
        let numbers : (RustSlice (FieldElement F)) ←
          (rust_primitives.hax.monomorphized_update_at.update_at_usize
            numbers
            (0 : usize)
            bi_inv);
        let
          hax_temp_output : (core_models.result.Result
            rust_primitives.hax.Tuple0
            math.field.errors.FieldError) :=
          (core_models.result.Result.Ok rust_primitives.hax.Tuple0.mk);
        (pure (rust_primitives.hax.Tuple2.mk numbers hax_temp_output))
      | (core_models.result.Result.Err  err) => do
        (pure (rust_primitives.hax.Tuple2.mk
          numbers
          (core_models.result.Result.Err err)))

--  Computes the multiplicative inverses of a slice of field elements
--  The algorithm just performs one inversion and several multiplications and should be used
--  when wanting to invert several elements together.
-- 
--  On `Err(InvZeroError)` the input slice is left unchanged (all-or-nothing).
--  The parallel path enforces this with a zero pre-scan; the sequential
--  path checks before any mutation.
@[spec]
def Impl.inplace_batch_inverse
    (F : Type)
    [trait_constr_inplace_batch_inverse_associated_type_i0 :
      math.field.traits.IsField.AssociatedTypes
      F]
    [trait_constr_inplace_batch_inverse_i0 : math.field.traits.IsField F ]
    (numbers : (RustSlice (FieldElement F))) :
    RustM
    (rust_primitives.hax.Tuple2
      (RustSlice (FieldElement F))
      (core_models.result.Result
        rust_primitives.hax.Tuple0
        math.field.errors.FieldError))
    := do
  let ⟨tmp0, out⟩ ← (Impl.inplace_batch_inverse_sequential F numbers);
  let numbers : (RustSlice (FieldElement F)) := tmp0;
  let
    hax_temp_output : (core_models.result.Result
      rust_primitives.hax.Tuple0
      math.field.errors.FieldError) :=
    out;
  (pure (rust_primitives.hax.Tuple2.mk numbers hax_temp_output))

@[spec]
def Impl_14.sum_hoisted
    (F : Type)
    (I : Type)
    [trait_constr_sum_hoisted_associated_type_i0 :
      math.field.traits.IsField.AssociatedTypes
      F]
    [trait_constr_sum_hoisted_i0 : math.field.traits.IsField F ]
    [trait_constr_sum_hoisted_associated_type_i1 :
      core_models.iter.traits.iterator.Iterator.AssociatedTypes
      I]
    [trait_constr_sum_hoisted_i1 : core_models.iter.traits.iterator.Iterator
      I
      (associatedTypes := {
        show core_models.iter.traits.iterator.Iterator.AssociatedTypes I
        by infer_instance
        with Item := (FieldElement F)})]
    (iter : I) :
    RustM (FieldElement F) := do
  (core_models.iter.traits.iterator.Iterator.fold
    I
    (FieldElement F)
    ((FieldElement F) -> (FieldElement F) -> RustM (FieldElement F))
    iter
    (← (Impl_32.zero F rust_primitives.hax.Tuple0.mk))
    (fun augend addend =>
      (do
      (core_models.ops.arith.Add.add
        (FieldElement F)
        (FieldElement F) augend addend) :
      RustM (FieldElement F))))

--  Sum operator for field elements
@[reducible] instance Impl_14.AssociatedTypes
  (F : Type)
  [trait_constr_Impl_14_associated_type_i0 :
    math.field.traits.IsField.AssociatedTypes
    F]
  [trait_constr_Impl_14_i0 : math.field.traits.IsField F ] :
  core_models.iter.traits.accum.Sum.AssociatedTypes
  (FieldElement F)
  (FieldElement F)
  where

instance Impl_14
  (F : Type)
  [trait_constr_Impl_14_associated_type_i0 :
    math.field.traits.IsField.AssociatedTypes
    F]
  [trait_constr_Impl_14_i0 : math.field.traits.IsField F ] :
  core_models.iter.traits.accum.Sum (FieldElement F) (FieldElement F)
  where
  sum :=
    fun
      
      (I : Type)
      [trait_constr__associated_type_i1 :
        core_models.iter.traits.iterator.Iterator.AssociatedTypes
        I]
      [trait_constr__i1 : core_models.iter.traits.iterator.Iterator
        I
        (associatedTypes := {
          show core_models.iter.traits.iterator.Iterator.AssociatedTypes I
          by infer_instance
          with Item := (FieldElement F)})]
      =>
    (Impl_14.sum_hoisted F I)

end math.field.element


namespace math.field.traits

--  This trait is necessary for sampling a random field element with a uniform distribution.
class HasDefaultTranscript.AssociatedTypes (Self : Type) where
  [trait_constr_HasDefaultTranscript_i0 : IsField.AssociatedTypes Self]

attribute [instance_reducible, instance]
  HasDefaultTranscript.AssociatedTypes.trait_constr_HasDefaultTranscript_i0

class HasDefaultTranscript (Self : Type)
  [associatedTypes : outParam (HasDefaultTranscript.AssociatedTypes (Self :
      Type))]
  where
  [trait_constr_HasDefaultTranscript_i0 : IsField Self]
  get_random_field_element_from_rng (Self)
    (impl_rand::Rng : Type)
    [trait_constr_get_random_field_element_from_rng_associated_type_i1 :
      rand.rng.Rng.AssociatedTypes
      impl_rand::Rng]
    [trait_constr_get_random_field_element_from_rng_i1 : rand.rng.Rng
      impl_rand::Rng
      ] :
    (impl_rand::Rng ->
    RustM (rust_primitives.hax.Tuple2
      impl_rand::Rng
      (math.field.element.FieldElement Self)))

attribute [instance_reducible, instance]
  HasDefaultTranscript.trait_constr_HasDefaultTranscript_i0

end math.field.traits


namespace math.spill_safe

@[reducible] instance Impl_11.AssociatedTypes
  (F : Type)
  [trait_constr_Impl_11_associated_type_i0 :
    math.field.traits.IsField.AssociatedTypes
    F]
  [trait_constr_Impl_11_i0 : math.field.traits.IsField F ]
  [trait_constr_Impl_11_associated_type_i1 :
    core_models.marker.Copy.AssociatedTypes
    F]
  [trait_constr_Impl_11_i1 : core_models.marker.Copy F ]
  [trait_constr_Impl_11_associated_type_i2 : SpillSafe.AssociatedTypes
    (math.field.traits.IsField.BaseType F)]
  [trait_constr_Impl_11_i2 : SpillSafe (math.field.traits.IsField.BaseType F) ]
  :
  SpillSafe.AssociatedTypes (math.field.element.FieldElement F)
  where

instance Impl_11
  (F : Type)
  [trait_constr_Impl_11_associated_type_i0 :
    math.field.traits.IsField.AssociatedTypes
    F]
  [trait_constr_Impl_11_i0 : math.field.traits.IsField F ]
  [trait_constr_Impl_11_associated_type_i1 :
    core_models.marker.Copy.AssociatedTypes
    F]
  [trait_constr_Impl_11_i1 : core_models.marker.Copy F ]
  [trait_constr_Impl_11_associated_type_i2 : SpillSafe.AssociatedTypes
    (math.field.traits.IsField.BaseType F)]
  [trait_constr_Impl_11_i2 : SpillSafe (math.field.traits.IsField.BaseType F) ]
  :
  SpillSafe (math.field.element.FieldElement F)
  where

end math.spill_safe


namespace math.fft.bowers_fft

--  Pre-computed twiddle factors organized by layer for cache-friendly access.
-- 
--  # Why LayerTwiddles?
-- 
--  Standard FFT implementations access twiddles with strided patterns like `twiddles[j * 2^layer]`.
--  This causes cache misses because the stride grows exponentially with each layer, leading to
--  random memory access patterns.
-- 
--  LayerTwiddles reorganizes twiddles so that each layer's values are stored contiguously.
--  During FFT computation, we iterate sequentially through `layer_twiddles[layer][0..count]`,
--  achieving O(N) sequential memory access instead of O(N log N) strided access.
-- 
--  This optimization can provide 10-30% speedup on large inputs where memory bandwidth
--  is the bottleneck.
-- 
--  # Memory Layout
-- 
--  For an FFT of size n = 2^order:
--  - Layer 0: n/2 twiddles (w^0, w^1, w^2, ...)
--  - Layer 1: n/4 twiddles (w^0, w^2, w^4, ...)
--  - Layer k: n/2^(k+1) twiddles (w^0, w^(2^k), w^(2*2^k), ...)
-- 
--  Total memory: n - 1 twiddles (same as flat storage, but organized for locality).
-- 
--  # Reusability
-- 
--  LayerTwiddles can be computed once and reused for multiple FFTs of the same size.
--  This amortizes the precomputation cost when processing many polynomials.
structure LayerTwiddles
  (F : Type)
  [trait_constr_LayerTwiddles_associated_type_i0 :
    math.field.traits.IsField.AssociatedTypes
    F]
  [trait_constr_LayerTwiddles_i0 : math.field.traits.IsField F ]
  where
  layers : (alloc.vec.Vec
      (alloc.vec.Vec (math.field.element.FieldElement F) alloc.alloc.Global)
      alloc.alloc.Global)

@[instance] opaque Impl_1.AssociatedTypes
  (F : Type)
  [trait_constr_Impl_1_associated_type_i0 :
    core_models.clone.Clone.AssociatedTypes
    F]
  [trait_constr_Impl_1_i0 : core_models.clone.Clone F ]
  [trait_constr_Impl_1_associated_type_i1 :
    math.field.traits.IsField.AssociatedTypes
    F]
  [trait_constr_Impl_1_i1 : math.field.traits.IsField F ] :
  core_models.clone.Clone.AssociatedTypes (LayerTwiddles F) :=
  by constructor <;> exact Inhabited.default

@[instance] opaque Impl_1
  (F : Type)
  [trait_constr_Impl_1_associated_type_i0 :
    core_models.clone.Clone.AssociatedTypes
    F]
  [trait_constr_Impl_1_i0 : core_models.clone.Clone F ]
  [trait_constr_Impl_1_associated_type_i1 :
    math.field.traits.IsField.AssociatedTypes
    F]
  [trait_constr_Impl_1_i1 : math.field.traits.IsField F ] :
  core_models.clone.Clone (LayerTwiddles F) :=
  by constructor <;> exact Inhabited.default

end math.fft.bowers_fft


namespace math.polynomial

--  Creates a new polynomial with the given coefficients
@[spec]
def Impl.new
    (F : Type)
    [trait_constr_new_associated_type_i0 :
      math.field.traits.IsField.AssociatedTypes
      F]
    [trait_constr_new_i0 : math.field.traits.IsField F ]
    (coefficients : (RustSlice (math.field.element.FieldElement F))) :
    RustM (Polynomial (math.field.element.FieldElement F)) := do
  let zero : (math.field.element.FieldElement F) ←
    (math.field.element.Impl_32.zero F rust_primitives.hax.Tuple0.mk);
  let len : usize ←
    (core_models.slice.Impl.len (math.field.element.FieldElement F)
      coefficients);
  let len : usize ←
    (rust_primitives.hax.while_loop
      (fun len => (do (pure true) : RustM Bool))
      (fun len =>
        (do
        ((← (len >? (0 : usize)))
          &&? (← (core_models.cmp.PartialEq.eq
            (math.field.element.FieldElement F)
            (math.field.element.FieldElement F)
            (← coefficients[(← (len -? (1 : usize)))]_?)
            zero))) :
        RustM Bool))
      (fun len =>
        (do
        (rust_primitives.hax.int.from_machine (0 : u32)) :
        RustM hax_lib.int.Int))
      len
      (fun len =>
        (do let len : usize ← (len -? (1 : usize)); (pure len) : RustM usize)));
  let
    unpadded_coefficients : (alloc.vec.Vec
      (math.field.element.FieldElement F)
      alloc.alloc.Global) ←
    (alloc.vec.Impl.with_capacity (math.field.element.FieldElement F) len);
  let
    unpadded_coefficients : (alloc.vec.Vec
      (math.field.element.FieldElement F)
      alloc.alloc.Global) ←
    (core_models.iter.traits.iterator.Iterator.fold
      (← (core_models.iter.traits.collect.IntoIterator.into_iter
        (RustSlice (math.field.element.FieldElement F))
        (← coefficients[(core_models.ops.range.RangeTo.mk (_end := len))]_?)))
      unpadded_coefficients
      (fun unpadded_coefficients coeff =>
        (do
        (alloc.vec.Impl_1.push
          (math.field.element.FieldElement F)
          alloc.alloc.Global
          unpadded_coefficients
          (← (core_models.clone.Clone.clone
            (math.field.element.FieldElement F) coeff))) :
        RustM
        (alloc.vec.Vec
          (math.field.element.FieldElement F)
          alloc.alloc.Global))));
  (pure (Polynomial.mk (coefficients := unpadded_coefficients)))

--  Creates a new monomial term coefficient*x^degree
@[spec]
def Impl.new_monomial
    (F : Type)
    [trait_constr_new_monomial_associated_type_i0 :
      math.field.traits.IsField.AssociatedTypes
      F]
    [trait_constr_new_monomial_i0 : math.field.traits.IsField F ]
    (coefficient : (math.field.element.FieldElement F))
    (degree : usize) :
    RustM (Polynomial (math.field.element.FieldElement F)) := do
  let
    coefficients : (alloc.vec.Vec
      (math.field.element.FieldElement F)
      alloc.alloc.Global) ←
    (alloc.vec.from_elem (math.field.element.FieldElement F)
      (← (math.field.element.Impl_32.zero F rust_primitives.hax.Tuple0.mk))
      degree);
  let
    coefficients : (alloc.vec.Vec
      (math.field.element.FieldElement F)
      alloc.alloc.Global) ←
    (alloc.vec.Impl_1.push
      (math.field.element.FieldElement F)
      alloc.alloc.Global coefficients coefficient);
  (Impl.new F
    (← (core_models.ops.deref.Deref.deref
      (alloc.vec.Vec (math.field.element.FieldElement F) alloc.alloc.Global)
      coefficients)))

--  Creates the null polynomial
@[spec]
def Impl.zero
    (F : Type)
    [trait_constr_zero_associated_type_i0 :
      math.field.traits.IsField.AssociatedTypes
      F]
    [trait_constr_zero_i0 : math.field.traits.IsField F ]
    (_ : rust_primitives.hax.Tuple0) :
    RustM (Polynomial (math.field.element.FieldElement F)) := do
  (Impl.new F (← (rust_primitives.unsize (RustArray.ofVec #v[]))))

--  Evaluates a polynomial P(t) at a point x, using Horner's algorithm
--  Returns y = P(x)
@[spec]
def Impl.evaluate
    (F : Type)
    (E : Type)
    [trait_constr_evaluate_associated_type_i0 :
      math.field.traits.IsField.AssociatedTypes
      F]
    [trait_constr_evaluate_i0 : math.field.traits.IsField F ]
    [trait_constr_evaluate_associated_type_i1 :
      math.field.traits.IsField.AssociatedTypes
      E]
    [trait_constr_evaluate_i1 : math.field.traits.IsField E ]
    [trait_constr_evaluate_associated_type_i2 :
      math.field.traits.IsSubFieldOf.AssociatedTypes
      F
      E]
    [trait_constr_evaluate_i2 : math.field.traits.IsSubFieldOf F E ]
    (self : (Polynomial (math.field.element.FieldElement F)))
    (x : (math.field.element.FieldElement E)) :
    RustM (math.field.element.FieldElement E) := do
  (core_models.iter.traits.iterator.Iterator.fold
    (core_models.iter.adapters.rev.Rev
      (core_models.slice.iter.Iter (math.field.element.FieldElement F)))
    (math.field.element.FieldElement E)
    ((math.field.element.FieldElement E) ->
    (math.field.element.FieldElement F) ->
    RustM (math.field.element.FieldElement E))
    (← (core_models.iter.traits.iterator.Iterator.rev
      (core_models.slice.iter.Iter (math.field.element.FieldElement F))
      (← (core_models.slice.Impl.iter (math.field.element.FieldElement F)
        (← (core_models.ops.deref.Deref.deref
          (alloc.vec.Vec (math.field.element.FieldElement F) alloc.alloc.Global)
          (Polynomial.coefficients self)))))))
    (← (math.field.element.Impl_32.zero E rust_primitives.hax.Tuple0.mk))
    (fun acc coeff =>
      (do
      (core_models.ops.arith.Add.add
        (math.field.element.FieldElement F)
        (math.field.element.FieldElement E)
        coeff
        (← (core_models.ops.arith.Mul.mul
          (math.field.element.FieldElement E)
          (math.field.element.FieldElement E)
          acc
          (← (alloc.borrow.ToOwned.to_owned
            (math.field.element.FieldElement E) x))))) :
      RustM (math.field.element.FieldElement E))))

--  Returns the degree of a polynomial, which corresponds to the highest power of x^d
--  with non-zero coefficient
@[spec]
def Impl.degree
    (F : Type)
    [trait_constr_degree_associated_type_i0 :
      math.field.traits.IsField.AssociatedTypes
      F]
    [trait_constr_degree_i0 : math.field.traits.IsField F ]
    (self : (Polynomial (math.field.element.FieldElement F))) :
    RustM usize := do
  if
  (← (alloc.vec.Impl_1.is_empty
    (math.field.element.FieldElement F)
    alloc.alloc.Global (Polynomial.coefficients self))) then do
    (pure (0 : usize))
  else do
    ((← (alloc.vec.Impl_1.len
        (math.field.element.FieldElement F)
        alloc.alloc.Global (Polynomial.coefficients self)))
      -? (1 : usize))

--  Returns coefficients of the polynomial as an array
--  \[c_0, c_1, c_2, ..., c_n\]
--  that represents the polynomial
--  c_0 + c_1 * X + c_2 * X^2 + ... + c_n * X^n
@[spec]
def Impl.coefficients
    (F : Type)
    [trait_constr_coefficients_associated_type_i0 :
      math.field.traits.IsField.AssociatedTypes
      F]
    [trait_constr_coefficients_i0 : math.field.traits.IsField F ]
    (self : (Polynomial (math.field.element.FieldElement F))) :
    RustM (RustSlice (math.field.element.FieldElement F)) := do
  (core_models.ops.deref.Deref.deref
    (alloc.vec.Vec (math.field.element.FieldElement F) alloc.alloc.Global)
    (Polynomial.coefficients self))

--  Returns the length of the vector of coefficients
@[spec]
def Impl.coeff_len
    (F : Type)
    [trait_constr_coeff_len_associated_type_i0 :
      math.field.traits.IsField.AssociatedTypes
      F]
    [trait_constr_coeff_len_i0 : math.field.traits.IsField F ]
    (self : (Polynomial (math.field.element.FieldElement F))) :
    RustM usize := do
  (core_models.slice.Impl.len (math.field.element.FieldElement F)
    (← (Impl.coefficients F self)))

@[spec]
def Impl.mul_with_ref
    (F : Type)
    [trait_constr_mul_with_ref_associated_type_i0 :
      math.field.traits.IsField.AssociatedTypes
      F]
    [trait_constr_mul_with_ref_i0 : math.field.traits.IsField F ]
    (self : (Polynomial (math.field.element.FieldElement F)))
    (factor : (Polynomial (math.field.element.FieldElement F))) :
    RustM (Polynomial (math.field.element.FieldElement F)) := do
  let degree : usize ← ((← (Impl.degree F self)) +? (← (Impl.degree F factor)));
  let
    coefficients : (alloc.vec.Vec
      (math.field.element.FieldElement F)
      alloc.alloc.Global) ←
    (alloc.vec.from_elem (math.field.element.FieldElement F)
      (← (math.field.element.Impl_32.zero F rust_primitives.hax.Tuple0.mk))
      (← (degree +? (1 : usize))));
  if
  (← ((← (alloc.vec.Impl_1.is_empty
      (math.field.element.FieldElement F)
      alloc.alloc.Global (Polynomial.coefficients self)))
    ||? (← (alloc.vec.Impl_1.is_empty
      (math.field.element.FieldElement F)
      alloc.alloc.Global (Polynomial.coefficients factor))))) then do
    (Impl.new F
      (← (rust_primitives.unsize
        (RustArray.ofVec #v[(← (math.field.element.Impl_32.zero F
                                rust_primitives.hax.Tuple0.mk))]))))
  else do
    let
      coefficients : (alloc.vec.Vec
        (math.field.element.FieldElement F)
        alloc.alloc.Global) ←
      (core_models.iter.traits.iterator.Iterator.fold
        (← (core_models.iter.traits.collect.IntoIterator.into_iter
          (core_models.ops.range.RangeInclusive usize)
          (← (core_models.ops.range.Impl_7.new usize
            (0 : usize)
            (← (Impl.degree F factor))))))
        coefficients
        (fun coefficients i =>
          (do
          if
          (← (core_models.cmp.PartialEq.ne
            (math.field.element.FieldElement F)
            (math.field.element.FieldElement F)
            (← (Polynomial.coefficients factor)[i]_?)
            (← (math.field.element.Impl_32.zero F
              rust_primitives.hax.Tuple0.mk)))) then do
            (core_models.iter.traits.iterator.Iterator.fold
              (← (core_models.iter.traits.collect.IntoIterator.into_iter
                (core_models.ops.range.RangeInclusive usize)
                (← (core_models.ops.range.Impl_7.new usize
                  (0 : usize)
                  (← (Impl.degree F self))))))
              coefficients
              (fun coefficients j =>
                (do
                if
                (← (core_models.cmp.PartialEq.ne
                  (math.field.element.FieldElement F)
                  (math.field.element.FieldElement F)
                  (← (Polynomial.coefficients self)[j]_?)
                  (← (math.field.element.Impl_32.zero F
                    rust_primitives.hax.Tuple0.mk)))) then do
                  let
                    coefficients : (alloc.vec.Vec
                      (math.field.element.FieldElement F)
                      alloc.alloc.Global) ←
                    (alloc.slice.Impl.to_vec
                      (←
                      (rust_primitives.hax.monomorphized_update_at.update_at_usize
                        (← (alloc.vec.Impl_1.as_slice coefficients))
                        (← (i +? j))
                        (← (core_models.ops.arith.AddAssign.add_assign
                          (math.field.element.FieldElement F)
                          (math.field.element.FieldElement F)
                          (← coefficients[(← (i +? j))]_?)
                          (← (core_models.ops.arith.Mul.mul
                            (math.field.element.FieldElement F)
                            (math.field.element.FieldElement F)
                            (← (Polynomial.coefficients factor)[i]_?)
                            (← (Polynomial.coefficients self)[j]_?))))))));
                  (pure coefficients)
                else do
                  (pure coefficients) :
                RustM
                (alloc.vec.Vec
                  (math.field.element.FieldElement F)
                  alloc.alloc.Global))))
          else do
            (pure coefficients) :
          RustM
          (alloc.vec.Vec
            (math.field.element.FieldElement F)
            alloc.alloc.Global))));
    (Impl.new F
      (← (core_models.ops.deref.Deref.deref
        (alloc.vec.Vec (math.field.element.FieldElement F) alloc.alloc.Global)
        coefficients)))

--  Scales the coefficients of a polynomial P by a factor
--  Returns P(factor * x)
@[spec]
def Impl.scale
    (F : Type)
    (S : Type)
    [trait_constr_scale_associated_type_i0 :
      math.field.traits.IsField.AssociatedTypes
      F]
    [trait_constr_scale_i0 : math.field.traits.IsField F ]
    [trait_constr_scale_associated_type_i1 :
      math.field.traits.IsSubFieldOf.AssociatedTypes
      S
      F]
    [trait_constr_scale_i1 : math.field.traits.IsSubFieldOf S F ]
    (self : (Polynomial (math.field.element.FieldElement F)))
    (factor : (math.field.element.FieldElement S)) :
    RustM (Polynomial (math.field.element.FieldElement F)) := do
  let
    scaled_coefficients : (alloc.vec.Vec
      (math.field.element.FieldElement F)
      alloc.alloc.Global) ←
    (alloc.vec.Impl.with_capacity (math.field.element.FieldElement F)
      (← (alloc.vec.Impl_1.len
        (math.field.element.FieldElement F)
        alloc.alloc.Global (Polynomial.coefficients self))));
  let power : (math.field.element.FieldElement S) ←
    (math.field.element.Impl_32.one S rust_primitives.hax.Tuple0.mk);
  let ⟨power, scaled_coefficients⟩ ←
    (core_models.iter.traits.iterator.Iterator.fold
      (← (core_models.iter.traits.collect.IntoIterator.into_iter
        (core_models.slice.iter.Iter (math.field.element.FieldElement F))
        (← (core_models.slice.Impl.iter (math.field.element.FieldElement F)
          (← (core_models.ops.deref.Deref.deref
            (alloc.vec.Vec
              (math.field.element.FieldElement F)
              alloc.alloc.Global) (Polynomial.coefficients self)))))))
      (rust_primitives.hax.Tuple2.mk power scaled_coefficients)
      (fun ⟨power, scaled_coefficients⟩ coeff =>
        (do
        let
          scaled_coefficients : (alloc.vec.Vec
            (math.field.element.FieldElement F)
            alloc.alloc.Global) ←
          (alloc.vec.Impl_1.push
            (math.field.element.FieldElement F)
            alloc.alloc.Global
            scaled_coefficients
            (← (core_models.ops.arith.Mul.mul
              (math.field.element.FieldElement S)
              (math.field.element.FieldElement F) power coeff)));
        let power : (math.field.element.FieldElement S) ←
          (core_models.ops.arith.Mul.mul
            (math.field.element.FieldElement S)
            (math.field.element.FieldElement S) power factor);
        (pure (rust_primitives.hax.Tuple2.mk power scaled_coefficients)) :
        RustM
        (rust_primitives.hax.Tuple2
          (math.field.element.FieldElement S)
          (alloc.vec.Vec
            (math.field.element.FieldElement F)
            alloc.alloc.Global)))));
  (pure (Polynomial.mk (coefficients := scaled_coefficients)))

--  Multiplies all coefficients by a factor
@[spec]
def Impl.scale_coeffs
    (F : Type)
    [trait_constr_scale_coeffs_associated_type_i0 :
      math.field.traits.IsField.AssociatedTypes
      F]
    [trait_constr_scale_coeffs_i0 : math.field.traits.IsField F ]
    (self : (Polynomial (math.field.element.FieldElement F)))
    (factor : (math.field.element.FieldElement F)) :
    RustM (Polynomial (math.field.element.FieldElement F)) := do
  let
    scaled_coefficients : (alloc.vec.Vec
      (math.field.element.FieldElement F)
      alloc.alloc.Global) ←
    (core_models.iter.traits.iterator.Iterator.collect
      (core_models.iter.adapters.map.Map
        (core_models.slice.iter.Iter (math.field.element.FieldElement F))
        ((math.field.element.FieldElement F) ->
        RustM (math.field.element.FieldElement F)))
      (alloc.vec.Vec (math.field.element.FieldElement F) alloc.alloc.Global)
      (← (core_models.iter.traits.iterator.Iterator.map
        (core_models.slice.iter.Iter (math.field.element.FieldElement F))
        (math.field.element.FieldElement F)
        ((math.field.element.FieldElement F) ->
        RustM (math.field.element.FieldElement F))
        (← (core_models.slice.Impl.iter (math.field.element.FieldElement F)
          (← (core_models.ops.deref.Deref.deref
            (alloc.vec.Vec
              (math.field.element.FieldElement F)
              alloc.alloc.Global) (Polynomial.coefficients self)))))
        (fun coeff =>
          (do
          (core_models.ops.arith.Mul.mul
            (math.field.element.FieldElement F)
            (math.field.element.FieldElement F) factor coeff) :
          RustM (math.field.element.FieldElement F))))));
  (pure (Polynomial.mk (coefficients := scaled_coefficients)))

--  Returns a vector of polynomials [p₀, p₁, ..., p_{d-1}], where d is `number_of_parts`, such that `self` equals
--  p₀(Xᵈ) + Xp₁(Xᵈ) + ... + X^(d-1)p_{d-1}(Xᵈ).
-- 
--  Example: if d = 2 and `self` is 3 X^3 + X^2 + 2X + 1, then `poly.break_in_parts(2)`
--  returns a vector with two polynomials `(p₀, p₁)`, where p₀ = X + 1 and p₁ = 3X + 2.
@[spec]
def Impl.break_in_parts
    (F : Type)
    [trait_constr_break_in_parts_associated_type_i0 :
      math.field.traits.IsField.AssociatedTypes
      F]
    [trait_constr_break_in_parts_i0 : math.field.traits.IsField F ]
    (self : (Polynomial (math.field.element.FieldElement F)))
    (number_of_parts : usize) :
    RustM
    (alloc.vec.Vec
      (Polynomial (math.field.element.FieldElement F))
      alloc.alloc.Global)
    := do
  let coef : (RustSlice (math.field.element.FieldElement F)) ←
    (Impl.coefficients F self);
  let
    parts : (alloc.vec.Vec
      (Polynomial (math.field.element.FieldElement F))
      alloc.alloc.Global) ←
    (alloc.vec.Impl.with_capacity
      (Polynomial (math.field.element.FieldElement F)) number_of_parts);
  let
    parts : (alloc.vec.Vec
      (Polynomial (math.field.element.FieldElement F))
      alloc.alloc.Global) ←
    (rust_primitives.hax.folds.fold_range
      (0 : usize)
      number_of_parts
      (fun parts _ => (do (pure true) : RustM Bool))
      parts
      (fun parts i =>
        (do
        let
          coeffs : (alloc.vec.Vec
            (math.field.element.FieldElement F)
            alloc.alloc.Global) ←
          (alloc.vec.Impl.new (math.field.element.FieldElement F)
            rust_primitives.hax.Tuple0.mk);
        let j : usize := i;
        let ⟨coeffs, j⟩ ←
          (rust_primitives.hax.while_loop
            (fun ⟨coeffs, j⟩ => (do (pure true) : RustM Bool))
            (fun ⟨coeffs, j⟩ =>
              (do
              (j
                <? (← (core_models.slice.Impl.len
                  (math.field.element.FieldElement F) coef))) :
              RustM Bool))
            (fun ⟨coeffs, j⟩ =>
              (do
              (rust_primitives.hax.int.from_machine (0 : u32)) :
              RustM hax_lib.int.Int))
            (rust_primitives.hax.Tuple2.mk coeffs j)
            (fun ⟨coeffs, j⟩ =>
              (do
              let
                coeffs : (alloc.vec.Vec
                  (math.field.element.FieldElement F)
                  alloc.alloc.Global) ←
                (alloc.vec.Impl_1.push
                  (math.field.element.FieldElement F)
                  alloc.alloc.Global
                  coeffs
                  (← (core_models.clone.Clone.clone
                    (math.field.element.FieldElement F) (← coef[j]_?))));
              let j : usize ← (j +? number_of_parts);
              (pure (rust_primitives.hax.Tuple2.mk coeffs j)) :
              RustM
              (rust_primitives.hax.Tuple2
                (alloc.vec.Vec
                  (math.field.element.FieldElement F)
                  alloc.alloc.Global)
                usize))));
        let
          parts : (alloc.vec.Vec
            (Polynomial (math.field.element.FieldElement F))
            alloc.alloc.Global) ←
          (alloc.vec.Impl_1.push
            (Polynomial (math.field.element.FieldElement F))
            alloc.alloc.Global
            parts
            (← (Impl.new F
              (← (core_models.ops.deref.Deref.deref
                (alloc.vec.Vec
                  (math.field.element.FieldElement F)
                  alloc.alloc.Global) coeffs)))));
        (pure parts) :
        RustM
        (alloc.vec.Vec
          (Polynomial (math.field.element.FieldElement F))
          alloc.alloc.Global))));
  (pure parts)

--  Pads a polynomial with zeros until the desired length
--  This function can be useful when evaluating polynomials with the FFT
@[spec]
def pad_with_zero_coefficients_to_length
    (F : Type)
    [trait_constr_pad_with_zero_coefficients_to_length_associated_type_i0 :
      math.field.traits.IsField.AssociatedTypes
      F]
    [trait_constr_pad_with_zero_coefficients_to_length_i0 :
      math.field.traits.IsField
      F
      ]
    (pa : (Polynomial (math.field.element.FieldElement F)))
    (n : usize) :
    RustM (Polynomial (math.field.element.FieldElement F)) := do
  let pa : (Polynomial (math.field.element.FieldElement F)) :=
    {pa
    with coefficients := (← (alloc.vec.Impl_2.resize
      (math.field.element.FieldElement F)
      alloc.alloc.Global
      (Polynomial.coefficients pa)
      n
      (← (math.field.element.Impl_32.zero F rust_primitives.hax.Tuple0.mk))))};
  (pure pa)

--  Pads polynomial representations with minimum number of zeros to match lengths.
@[spec]
def pad_with_zero_coefficients
    (L : Type)
    (F : Type)
    [trait_constr_pad_with_zero_coefficients_associated_type_i0 :
      math.field.traits.IsField.AssociatedTypes
      L]
    [trait_constr_pad_with_zero_coefficients_i0 : math.field.traits.IsField L ]
    [trait_constr_pad_with_zero_coefficients_associated_type_i1 :
      math.field.traits.IsSubFieldOf.AssociatedTypes
      F
      L]
    [trait_constr_pad_with_zero_coefficients_i1 : math.field.traits.IsSubFieldOf
      F
      L
      ]
    (pa : (Polynomial (math.field.element.FieldElement F)))
    (pb : (Polynomial (math.field.element.FieldElement L))) :
    RustM
    (rust_primitives.hax.Tuple2
      (Polynomial (math.field.element.FieldElement F))
      (Polynomial (math.field.element.FieldElement L)))
    := do
  let pa : (Polynomial (math.field.element.FieldElement F)) ←
    (core_models.clone.Clone.clone
      (Polynomial (math.field.element.FieldElement F)) pa);
  let pb : (Polynomial (math.field.element.FieldElement L)) ←
    (core_models.clone.Clone.clone
      (Polynomial (math.field.element.FieldElement L)) pb);
  let ⟨pa, pb⟩ ←
    if
    (← ((← (alloc.vec.Impl_1.len
        (math.field.element.FieldElement F)
        alloc.alloc.Global (Polynomial.coefficients pa)))
      >? (← (alloc.vec.Impl_1.len
        (math.field.element.FieldElement L)
        alloc.alloc.Global (Polynomial.coefficients pb))))) then do
      let pb : (Polynomial (math.field.element.FieldElement L)) ←
        (pad_with_zero_coefficients_to_length L
          pb
          (← (alloc.vec.Impl_1.len
            (math.field.element.FieldElement F)
            alloc.alloc.Global (Polynomial.coefficients pa))));
      (pure (rust_primitives.hax.Tuple2.mk pa pb))
    else do
      let pa : (Polynomial (math.field.element.FieldElement F)) ←
        (pad_with_zero_coefficients_to_length F
          pa
          (← (alloc.vec.Impl_1.len
            (math.field.element.FieldElement L)
            alloc.alloc.Global (Polynomial.coefficients pb))));
      (pure (rust_primitives.hax.Tuple2.mk pa pb));
  (pure (rust_primitives.hax.Tuple2.mk pa pb))

--  Precompute `1/(z - point_i)` for each coset point, using batch inversion.
-- 
--  Given an evaluation point `z` (in extension field E) and coset points in base field F,
--  returns the vector of inverse denominators needed for barycentric interpolation.
--  Uses Montgomery's trick: 1 field inversion + O(N) multiplications.
@[spec]
def barycentric_inv_denoms
    (F : Type)
    (E : Type)
    [trait_constr_barycentric_inv_denoms_associated_type_i0 :
      math.field.traits.IsSubFieldOf.AssociatedTypes
      F
      E]
    [trait_constr_barycentric_inv_denoms_i0 : math.field.traits.IsSubFieldOf
      F
      E
      ]
    [trait_constr_barycentric_inv_denoms_associated_type_i1 :
      math.field.traits.IsField.AssociatedTypes
      E]
    [trait_constr_barycentric_inv_denoms_i1 : math.field.traits.IsField E ]
    (z : (math.field.element.FieldElement E))
    (coset_points : (RustSlice (math.field.element.FieldElement F))) :
    RustM
    (alloc.vec.Vec (math.field.element.FieldElement E) alloc.alloc.Global)
    := do
  let
    denoms : (alloc.vec.Vec
      (math.field.element.FieldElement E)
      alloc.alloc.Global) ←
    (core_models.iter.traits.iterator.Iterator.collect
      (core_models.iter.adapters.map.Map
        (core_models.slice.iter.Iter (math.field.element.FieldElement F))
        ((math.field.element.FieldElement F) ->
        RustM (math.field.element.FieldElement E)))
      (alloc.vec.Vec (math.field.element.FieldElement E) alloc.alloc.Global)
      (← (core_models.iter.traits.iterator.Iterator.map
        (core_models.slice.iter.Iter (math.field.element.FieldElement F))
        (math.field.element.FieldElement E)
        ((math.field.element.FieldElement F) ->
        RustM (math.field.element.FieldElement E))
        (← (core_models.slice.Impl.iter (math.field.element.FieldElement F)
          coset_points))
        (fun p =>
          (do
          (core_models.ops.arith.Neg.neg
            (math.field.element.FieldElement E)
            (← (core_models.ops.arith.Sub.sub
              (math.field.element.FieldElement F)
              (math.field.element.FieldElement E) p z))) :
          RustM (math.field.element.FieldElement E))))));
  let ⟨tmp0, out⟩ ←
    (math.field.element.Impl.inplace_batch_inverse E
      (← (alloc.vec.Impl_1.as_slice denoms)));
  let
    denoms : (alloc.vec.Vec
      (math.field.element.FieldElement E)
      alloc.alloc.Global) ←
    (alloc.slice.Impl.to_vec tmp0);
  let _ ←
    (core_models.result.Impl.expect
      rust_primitives.hax.Tuple0
      math.field.errors.FieldError
      out
      "z is sampled to avoid coset points, so z - g*w^i is never zero");
  (pure denoms)

--  Like `interpolate_coset_eval_ext` but takes a precomputed `g_n_inv = (g^N)^{-1}`.
-- 
--  Both `coset_offset_pow_n` and `g_n_inv` stay in the base field F.
@[spec]
def interpolate_coset_eval_ext_with_g_n_inv
    (F : Type)
    (E : Type)
    [trait_constr_interpolate_coset_eval_ext_with_g_n_inv_associated_type_i0 :
      math.field.traits.IsSubFieldOf.AssociatedTypes
      F
      E]
    [trait_constr_interpolate_coset_eval_ext_with_g_n_inv_i0 :
      math.field.traits.IsSubFieldOf
      F
      E
      ]
    [trait_constr_interpolate_coset_eval_ext_with_g_n_inv_associated_type_i1 :
      math.field.traits.IsField.AssociatedTypes
      E]
    [trait_constr_interpolate_coset_eval_ext_with_g_n_inv_i1 :
      math.field.traits.IsField
      E
      ]
    (z_pow_n : (math.field.element.FieldElement E))
    (coset_offset_pow_n : (math.field.element.FieldElement F))
    (n_inv : (math.field.element.FieldElement F))
    (g_n_inv : (math.field.element.FieldElement F))
    (coset_points : (RustSlice (math.field.element.FieldElement F)))
    (evaluations : (RustSlice (math.field.element.FieldElement E)))
    (inv_denoms : (RustSlice (math.field.element.FieldElement E))) :
    RustM (math.field.element.FieldElement E) := do
  let _ ←
    if true then do
      let _ ←
        match
          (rust_primitives.hax.Tuple2.mk
            (← (core_models.slice.Impl.len (math.field.element.FieldElement F)
              coset_points))
            (← (core_models.slice.Impl.len (math.field.element.FieldElement E)
              evaluations)))
        with
          | ⟨left_val, right_val⟩ => do
            (hax_lib.assert (← (left_val ==? right_val)));
      (pure rust_primitives.hax.Tuple0.mk)
    else do
      (pure rust_primitives.hax.Tuple0.mk);
  let _ ←
    if true then do
      let _ ←
        match
          (rust_primitives.hax.Tuple2.mk
            (← (core_models.slice.Impl.len (math.field.element.FieldElement F)
              coset_points))
            (← (core_models.slice.Impl.len (math.field.element.FieldElement E)
              inv_denoms)))
        with
          | ⟨left_val, right_val⟩ => do
            (hax_lib.assert (← (left_val ==? right_val)));
      (pure rust_primitives.hax.Tuple0.mk)
    else do
      (pure rust_primitives.hax.Tuple0.mk);
  let sum : (math.field.element.FieldElement E) ←
    (math.field.element.Impl_32.zero E rust_primitives.hax.Tuple0.mk);
  let sum : (math.field.element.FieldElement E) ←
    (rust_primitives.hax.folds.fold_range
      (0 : usize)
      (← (core_models.slice.Impl.len (math.field.element.FieldElement F)
        coset_points))
      (fun sum _ => (do (pure true) : RustM Bool))
      sum
      (fun sum i =>
        (do
        let numerator : (math.field.element.FieldElement E) ←
          (core_models.ops.arith.Mul.mul
            (math.field.element.FieldElement F)
            (math.field.element.FieldElement E)
            (← coset_points[i]_?)
            (← evaluations[i]_?));
        let sum : (math.field.element.FieldElement E) ←
          (core_models.ops.arith.Add.add
            (math.field.element.FieldElement E)
            (math.field.element.FieldElement E)
            sum
            (← (core_models.ops.arith.Mul.mul
              (math.field.element.FieldElement E)
              (math.field.element.FieldElement E)
              numerator
              (← inv_denoms[i]_?))));
        (pure sum) :
        RustM (math.field.element.FieldElement E))));
  let vanishing : (math.field.element.FieldElement E) ←
    (math.field.element.Impl_32.sub_subfield E F z_pow_n coset_offset_pow_n);
  let scalar : (math.field.element.FieldElement F) ←
    (core_models.ops.arith.Mul.mul
      (math.field.element.FieldElement F)
      (math.field.element.FieldElement F) n_inv g_n_inv);
  (core_models.ops.arith.Mul.mul
    (math.field.element.FieldElement F)
    (math.field.element.FieldElement E)
    scalar
    (← (core_models.ops.arith.Mul.mul
      (math.field.element.FieldElement E)
      (math.field.element.FieldElement E) vanishing sum)))

end math.polynomial


namespace math.field.traits

--  Trait to define necessary parameters for FFT-friendly Fields.
--  Two-Adic fields are ones whose order is of the form  $2^n k + 1$.
--  Here $n$ is usually called the *two-adicity* of the field. The
--  reason we care about it is that in an $n$-adic field there are $2^j$-roots
--  of unity for every `j` between 1 and n, which is needed to do Fast Fourier.
--  A two-adic primitive root of unity is a number w that satisfies w^(2^n) = 1
--  and w^(j) != 1 for every j below 2^n. With this primitive root we can generate
--  any other root of unity we need to perform FFT.
class IsFFTField.AssociatedTypes (Self : Type) where
  [trait_constr_IsFFTField_i0 : IsField.AssociatedTypes Self]

attribute [instance_reducible, instance]
  IsFFTField.AssociatedTypes.trait_constr_IsFFTField_i0

class IsFFTField (Self : Type)
  [associatedTypes : outParam (IsFFTField.AssociatedTypes (Self : Type))]
  where
  [trait_constr_IsFFTField_i0 : IsField Self]
  TWO_ADICITY (Self) : u64
  TWO_ADIC_PRIMITVE_ROOT_OF_UNITY (Self) : (IsField.BaseType Self)
  field_name (Self) (_ : rust_primitives.hax.Tuple0) :RustM String := do
    (pure "")
  get_primitive_root_of_unity (Self) (order : u64) :RustM
    (core_models.result.Result
      (math.field.element.FieldElement Self)
      math.field.errors.FieldError) := do
    let
      two_adic_primitive_root_of_unity : (math.field.element.FieldElement
        Self) ←
      (math.field.element.Impl_32.new Self
        (IsFFTField.TWO_ADIC_PRIMITVE_ROOT_OF_UNITY Self));
    if (← (order ==? (0 : u64))) then do
      (pure (core_models.result.Result.Ok
        (← (math.field.element.Impl_32.one Self
          rust_primitives.hax.Tuple0.mk))))
    else do
      if (← (order >? (IsFFTField.TWO_ADICITY Self))) then do
        (pure (core_models.result.Result.Err
          (math.field.errors.FieldError.RootOfUnityError order)))
      else do
        let log_power : u64 ← ((IsFFTField.TWO_ADICITY Self) -? order);
        let root : (math.field.element.FieldElement Self) ←
          (core_models.iter.traits.iterator.Iterator.fold
            (core_models.ops.range.Range u64)
            (math.field.element.FieldElement Self)
            ((math.field.element.FieldElement Self) ->
            u64 ->
            RustM (math.field.element.FieldElement Self))
            (core_models.ops.range.Range.mk
              (start := (0 : u64))
              (_end := log_power))
            two_adic_primitive_root_of_unity
            (fun acc _ =>
              (do
              (math.field.element.Impl_32.square Self acc) :
              RustM (math.field.element.FieldElement Self))));
        (pure (core_models.result.Result.Ok root))

attribute [instance_reducible, instance] IsFFTField.trait_constr_IsFFTField_i0

class IsPrimeField.AssociatedTypes (Self : Type) where
  [trait_constr_IsPrimeField_i0 : IsField.AssociatedTypes Self]
  CanonicalType : Type

attribute [instance_reducible, instance]
  IsPrimeField.AssociatedTypes.trait_constr_IsPrimeField_i0

attribute [reducible] IsPrimeField.AssociatedTypes.CanonicalType

abbrev IsPrimeField.CanonicalType :=
  IsPrimeField.AssociatedTypes.CanonicalType

class IsPrimeField (Self : Type)
  [associatedTypes : outParam (IsPrimeField.AssociatedTypes (Self : Type))]
  where
  [trait_constr_IsPrimeField_i0 : IsField Self]
  [trait_constr_CanonicalType_associated_type_i1 :
    math.unsigned_integer.traits.IsUnsignedInteger.AssociatedTypes
    associatedTypes.CanonicalType]
  [trait_constr_CanonicalType_i1 :
    math.unsigned_integer.traits.IsUnsignedInteger
    associatedTypes.CanonicalType
    ]
  canonical (Self) :
    ((IsField.BaseType Self) -> RustM associatedTypes.CanonicalType)
  modulus_minus_one (Self) (_ : rust_primitives.hax.Tuple0) :RustM
    associatedTypes.CanonicalType := do
    (IsPrimeField.canonical
      Self
      (← (IsField.neg
        Self (← (IsField.one Self rust_primitives.hax.Tuple0.mk)))))
  from_hex (Self) :
    (String ->
    RustM (core_models.result.Result
      (IsField.BaseType Self)
      math.errors.CreationError))
  field_bit_size (Self) : (rust_primitives.hax.Tuple0 -> RustM usize)
  legendre_symbol (Self) (a : (IsField.BaseType Self)) :RustM LegendreSymbol :=
    do
    let symbol : (IsField.BaseType Self) ←
      (IsField.pow
        Self associatedTypes.CanonicalType
        a
        (← (core_models.ops.bit.Shr.shr
          associatedTypes.CanonicalType
          usize
          (← (IsPrimeField.modulus_minus_one
            Self rust_primitives.hax.Tuple0.mk))
          (1 : usize))));
    match
      (← match symbol with
        | x => do
          match
            (← (IsField.eq
              Self x (← (IsField.zero Self rust_primitives.hax.Tuple0.mk))))
          with
            | true => do
              (pure (core_models.option.Option.Some LegendreSymbol.Zero))
            | _ => do (pure core_models.option.Option.None)
        | _ => do (pure core_models.option.Option.None))
    with
      | (core_models.option.Option.Some  x) => do (pure x)
      | (core_models.option.Option.None ) => do
        match
          (← match symbol with
            | x => do
              match
                (← (IsField.eq
                  Self x (← (IsField.one Self rust_primitives.hax.Tuple0.mk))))
              with
                | true => do
                  (pure (core_models.option.Option.Some LegendreSymbol.One))
                | _ => do (pure core_models.option.Option.None)
            | _ => do (pure core_models.option.Option.None))
        with
          | (core_models.option.Option.Some  x) => do (pure x)
          | (core_models.option.Option.None ) => do
            (pure LegendreSymbol.MinusOne)
  sqrt (Self) (a : (IsField.BaseType Self)) :RustM (core_models.option.Option
      (rust_primitives.hax.Tuple2
        (IsField.BaseType Self)
        (IsField.BaseType Self))) := do
    match (← (IsPrimeField.legendre_symbol Self a)) with
      | (LegendreSymbol.Zero ) => do
        (pure (core_models.option.Option.Some
          (rust_primitives.hax.Tuple2.mk
            (← (IsField.zero Self rust_primitives.hax.Tuple0.mk))
            (← (IsField.zero Self rust_primitives.hax.Tuple0.mk)))))
      | (LegendreSymbol.MinusOne ) => do (pure core_models.option.Option.None)
      | (LegendreSymbol.One ) => do
        let _ := rust_primitives.hax.Tuple0.mk;
        let integer_one : associatedTypes.CanonicalType ←
          (core_models.convert.From._from
            associatedTypes.CanonicalType
            u16 (1 : u16));
        let s : usize := (0 : usize);
        let q : associatedTypes.CanonicalType ←
          (IsPrimeField.modulus_minus_one Self rust_primitives.hax.Tuple0.mk);
        let ⟨q, s⟩ ←
          (rust_primitives.hax.while_loop
            (fun ⟨q, s⟩ => (do (pure true) : RustM Bool))
            (fun ⟨q, s⟩ =>
              (do
              (core_models.cmp.PartialEq.ne
                associatedTypes.CanonicalType
                associatedTypes.CanonicalType
                (← (core_models.ops.bit.BitAnd.bitand
                  associatedTypes.CanonicalType
                  associatedTypes.CanonicalType q integer_one))
                integer_one) :
              RustM Bool))
            (fun ⟨q, s⟩ =>
              (do
              (rust_primitives.hax.int.from_machine (0 : u32)) :
              RustM hax_lib.int.Int))
            (rust_primitives.hax.Tuple2.mk q s)
            (fun ⟨q, s⟩ =>
              (do
              let s : usize ← (s +? (1 : usize));
              let q : associatedTypes.CanonicalType ←
                (core_models.ops.bit.ShrAssign.shr_assign
                  associatedTypes.CanonicalType
                  usize q (1 : usize));
              (pure (rust_primitives.hax.Tuple2.mk q s)) :
              RustM
              (rust_primitives.hax.Tuple2
                associatedTypes.CanonicalType
                usize))));
        let non_qr : (IsField.BaseType Self) ←
          (IsField.from_u64 Self (2 : u64));
        let non_qr : (IsField.BaseType Self) ←
          (rust_primitives.hax.while_loop
            (fun non_qr => (do (pure true) : RustM Bool))
            (fun non_qr =>
              (do
              (core_models.cmp.PartialEq.ne
                LegendreSymbol
                LegendreSymbol
                (← (IsPrimeField.legendre_symbol Self non_qr))
                LegendreSymbol.MinusOne) :
              RustM Bool))
            (fun non_qr =>
              (do
              (rust_primitives.hax.int.from_machine (0 : u32)) :
              RustM hax_lib.int.Int))
            non_qr
            (fun non_qr =>
              (do
              let non_qr : (IsField.BaseType Self) ←
                (IsField.add
                  Self
                  non_qr
                  (← (IsField.one Self rust_primitives.hax.Tuple0.mk)));
              (pure non_qr) :
              RustM (IsField.BaseType Self))));
        let c : (IsField.BaseType Self) ←
          (IsField.pow Self associatedTypes.CanonicalType non_qr q);
        let x : (IsField.BaseType Self) ←
          (IsField.pow
            Self associatedTypes.CanonicalType
            a
            (← (core_models.ops.bit.Shr.shr
              associatedTypes.CanonicalType
              usize
              (← (core_models.ops.arith.Add.add
                associatedTypes.CanonicalType
                associatedTypes.CanonicalType q integer_one))
              (1 : usize))));
        let t : (IsField.BaseType Self) ←
          (IsField.pow Self associatedTypes.CanonicalType a q);
        let m : usize := s;
        let one : (IsField.BaseType Self) ←
          (IsField.one Self rust_primitives.hax.Tuple0.mk);
        let ⟨c, m, x⟩ ←
          (rust_primitives.hax.while_loop
            (fun ⟨c, m, x⟩ => (do (pure true) : RustM Bool))
            (fun ⟨c, m, x⟩ =>
              (do (!? (← (IsField.eq Self t one))) : RustM Bool))
            (fun ⟨c, m, x⟩ =>
              (do
              (rust_primitives.hax.int.from_machine (0 : u32)) :
              RustM hax_lib.int.Int))
            (rust_primitives.hax.Tuple3.mk c m x)
            (fun ⟨c, m, x⟩ =>
              (do
              let i : usize := (0 : usize);
              let t : (IsField.BaseType Self) ←
                (core_models.clone.Clone.clone (IsField.BaseType Self) t);
              let minus_one : (IsField.BaseType Self) ←
                (IsField.neg
                  Self (← (IsField.one Self rust_primitives.hax.Tuple0.mk)));
              let ⟨i, t⟩ ←
                (rust_primitives.hax.while_loop
                  (fun ⟨i, t⟩ => (do (pure true) : RustM Bool))
                  (fun ⟨i, t⟩ =>
                    (do (!? (← (IsField.eq Self t minus_one))) : RustM Bool))
                  (fun ⟨i, t⟩ =>
                    (do
                    (rust_primitives.hax.int.from_machine (0 : u32)) :
                    RustM hax_lib.int.Int))
                  (rust_primitives.hax.Tuple2.mk i t)
                  (fun ⟨i, t⟩ =>
                    (do
                    let i : usize ← (i +? (1 : usize));
                    let t : (IsField.BaseType Self) ← (IsField.mul Self t t);
                    (pure (rust_primitives.hax.Tuple2.mk i t)) :
                    RustM
                    (rust_primitives.hax.Tuple2
                      usize
                      (IsField.BaseType Self)))));
              let i : usize ← (i +? (1 : usize));
              let b : (IsField.BaseType Self) ←
                (core_models.iter.traits.iterator.Iterator.fold
                  (core_models.ops.range.Range usize)
                  (IsField.BaseType Self)
                  ((IsField.BaseType Self) ->
                  usize ->
                  RustM (IsField.BaseType Self))
                  (core_models.ops.range.Range.mk
                    (start := (0 : usize))
                    (_end := (← ((← (m -? i)) -? (1 : usize)))))
                  c
                  (fun acc _ =>
                    (do
                    (IsField.square Self acc) :
                    RustM (IsField.BaseType Self))));
              let c : (IsField.BaseType Self) ← (IsField.mul Self b b);
              let x : (IsField.BaseType Self) ← (IsField.mul Self x b);
              let t : (IsField.BaseType Self) ← (IsField.mul Self t c);
              let m : usize := i;
              (pure (rust_primitives.hax.Tuple3.mk c m x)) :
              RustM
              (rust_primitives.hax.Tuple3
                (IsField.BaseType Self)
                usize
                (IsField.BaseType Self)))));
        let neg_x : (IsField.BaseType Self) ← (IsField.neg Self x);
        (pure (core_models.option.Option.Some
          (rust_primitives.hax.Tuple2.mk x neg_x)))

attribute [instance_reducible, instance]
  IsPrimeField.trait_constr_IsPrimeField_i0

end math.field.traits


namespace math.field.element

--  Creates a field element from a BigUint that is smaller than the modulus.
--  Returns error if the value is bigger than the modulus.
@[spec]
def Impl_32.from_reduced_big_uint
    (F : Type)
    [trait_constr_from_reduced_big_uint_associated_type_i0 :
      math.field.traits.IsField.AssociatedTypes
      F]
    [trait_constr_from_reduced_big_uint_i0 : math.field.traits.IsField F ]
    [trait_constr_from_reduced_big_uint_associated_type_i1 :
      math.traits.ByteConversion.AssociatedTypes
      (FieldElement F)]
    [trait_constr_from_reduced_big_uint_i1 : math.traits.ByteConversion
      (FieldElement F)
      ]
    [trait_constr_from_reduced_big_uint_associated_type_i2 :
      math.field.traits.IsPrimeField.AssociatedTypes
      F]
    [trait_constr_from_reduced_big_uint_i2 : math.field.traits.IsPrimeField F ]
    (value : num_bigint.biguint.BigUint) :
    RustM
    (core_models.result.Result (FieldElement F) math.errors.ByteConversionError)
    := do
  let
    args : (rust_primitives.hax.Tuple1
      (math.field.traits.IsPrimeField.CanonicalType F)) :=
    (rust_primitives.hax.Tuple1.mk
      (← (math.field.traits.IsPrimeField.modulus_minus_one
        F rust_primitives.hax.Tuple0.mk)));
  let args : (RustArray core_models.fmt.rt.Argument 1) :=
    (RustArray.ofVec #v[(← (core_models.fmt.rt.Impl.new_lower_hex
                            (math.field.traits.IsPrimeField.CanonicalType F)
                            (rust_primitives.hax.Tuple1._0 args)))]);
  let mod_minus_one : alloc.string.String ←
    (core_models.hint.must_use alloc.string.String
      (← (alloc.fmt.format
        (← (core_models.fmt.rt.Impl_1.new_v1 ((1 : usize)) ((1 : usize))
          (RustArray.ofVec #v[""])
          args)))));
  let modulus : num_bigint.biguint.BigUint ←
    (core_models.ops.arith.Add.add
      num_bigint.biguint.BigUint
      u32
      (← (core_models.result.Impl.expect
        num_bigint.biguint.BigUint
        num_bigint.ParseBigIntError
        (← (num_traits.Num.from_str_radix
          num_bigint.biguint.BigUint
          (← (core_models.ops.deref.Deref.deref
            alloc.string.String mod_minus_one))
          (16 : u32)))
        "invalid modulus representation"))
      (1 : u32));
  if
  (← (core_models.cmp.PartialOrd.ge
    num_bigint.biguint.BigUint
    num_bigint.biguint.BigUint value modulus)) then do
    (pure (core_models.result.Result.Err
      math.errors.ByteConversionError.ValueNotReduced))
  else do
    let bytes : (alloc.vec.Vec u8 alloc.alloc.Global) ←
      (num_bigint.biguint.Impl_19.to_bytes_le value);
    let bytes : (alloc.vec.Vec u8 alloc.alloc.Global) ←
      (alloc.vec.Impl_2.resize u8 alloc.alloc.Global
        bytes
        (← (core_models.mem.size_of (math.field.traits.IsField.BaseType F)
          rust_primitives.hax.Tuple0.mk))
        (0 : u8));
    (math.traits.ByteConversion.from_bytes_le
      (FieldElement F)
      (← (core_models.ops.deref.Deref.deref
        (alloc.vec.Vec u8 alloc.alloc.Global) bytes)))

@[spec]
def Impl_5.try_from_hoisted
    (F : Type)
    [trait_constr_try_from_hoisted_associated_type_i0 :
      math.traits.ByteConversion.AssociatedTypes
      (FieldElement F)]
    [trait_constr_try_from_hoisted_i0 : math.traits.ByteConversion
      (FieldElement F)
      ]
    [trait_constr_try_from_hoisted_associated_type_i1 :
      math.field.traits.IsPrimeField.AssociatedTypes
      F]
    [trait_constr_try_from_hoisted_i1 : math.field.traits.IsPrimeField F ]
    (value : num_bigint.biguint.BigUint) :
    RustM
    (core_models.result.Result (FieldElement F) math.errors.ByteConversionError)
    := do
  (Impl_32.from_reduced_big_uint F value)

--  From overloading for BigUint.
--  Creates a field element from a BigUint that is smaller than the modulus.
--  Returns error if the BigUint value is bigger than the modulus.
@[reducible] instance Impl_5.AssociatedTypes
  (F : Type)
  [trait_constr_Impl_5_associated_type_i0 :
    math.traits.ByteConversion.AssociatedTypes
    (FieldElement F)]
  [trait_constr_Impl_5_i0 : math.traits.ByteConversion (FieldElement F) ]
  [trait_constr_Impl_5_associated_type_i1 :
    math.field.traits.IsPrimeField.AssociatedTypes
    F]
  [trait_constr_Impl_5_i1 : math.field.traits.IsPrimeField F ] :
  core_models.convert.TryFrom.AssociatedTypes
  (FieldElement F)
  num_bigint.biguint.BigUint
  where
  Error := math.errors.ByteConversionError

instance Impl_5
  (F : Type)
  [trait_constr_Impl_5_associated_type_i0 :
    math.traits.ByteConversion.AssociatedTypes
    (FieldElement F)]
  [trait_constr_Impl_5_i0 : math.traits.ByteConversion (FieldElement F) ]
  [trait_constr_Impl_5_associated_type_i1 :
    math.field.traits.IsPrimeField.AssociatedTypes
    F]
  [trait_constr_Impl_5_i1 : math.field.traits.IsPrimeField F ] :
  core_models.convert.TryFrom (FieldElement F) num_bigint.biguint.BigUint
  where
  try_from := (Impl_5.try_from_hoisted F)

--  Converts a hex string into a field element.
--  It returns error if the hex value is larger than the modulus.
@[spec]
def Impl_32.from_hex_str
    (F : Type)
    [trait_constr_from_hex_str_associated_type_i0 :
      math.field.traits.IsField.AssociatedTypes
      F]
    [trait_constr_from_hex_str_i0 : math.field.traits.IsField F ]
    [trait_constr_from_hex_str_associated_type_i1 :
      math.traits.ByteConversion.AssociatedTypes
      (FieldElement F)]
    [trait_constr_from_hex_str_i1 : math.traits.ByteConversion
      (FieldElement F)
      ]
    [trait_constr_from_hex_str_associated_type_i2 :
      math.field.traits.IsPrimeField.AssociatedTypes
      F]
    [trait_constr_from_hex_str_i2 : math.field.traits.IsPrimeField F ]
    (hex : String) :
    RustM
    (core_models.result.Result (FieldElement F) math.errors.CreationError)
    := do
  let hex_str : String ←
    (core_models.option.Impl.unwrap_or String
      (← (core_models.str.Impl.strip_prefix String hex "0x"))
      hex);
  if (← (core_models.str.Impl.is_empty hex_str)) then do
    (pure (core_models.result.Result.Err math.errors.CreationError.EmptyString))
  else do
    match
      (← (core_models.result.Impl.map_err
        num_bigint.biguint.BigUint
        num_bigint.ParseBigIntError
        math.errors.CreationError
        (num_bigint.ParseBigIntError -> RustM math.errors.CreationError)
        (← (num_traits.Num.from_str_radix
          num_bigint.biguint.BigUint hex_str (16 : u32)))
        (fun _ =>
          (do
          (pure math.errors.CreationError.InvalidHexString) :
          RustM math.errors.CreationError))))
    with
      | (core_models.result.Result.Ok  value) => do
        (core_models.result.Impl.map_err
          (FieldElement F)
          math.errors.ByteConversionError
          math.errors.CreationError
          (math.errors.ByteConversionError -> RustM math.errors.CreationError)
          (← (Impl_32.from_reduced_big_uint F value))
          (fun _ =>
            (do
            (pure math.errors.CreationError.InvalidHexString) :
            RustM math.errors.CreationError)))
      | (core_models.result.Result.Err  err) => do
        (pure (core_models.result.Result.Err err))

--  Returns the canonical form of the value stored
@[spec]
def Impl_33.canonical
    (F : Type)
    [trait_constr_canonical_associated_type_i0 :
      math.field.traits.IsPrimeField.AssociatedTypes
      F]
    [trait_constr_canonical_i0 : math.field.traits.IsPrimeField F ]
    (self : (FieldElement F)) :
    RustM (math.field.traits.IsPrimeField.CanonicalType F) := do
  (math.field.traits.IsPrimeField.canonical F (← (Impl_32.value F self)))

--  Returns the two square roots of a field element, provided it exists
--  The function returns the roots whenever the field element is a quadratic residue modulo p
@[spec]
def Impl_33.sqrt
    (F : Type)
    [trait_constr_sqrt_associated_type_i0 :
      math.field.traits.IsPrimeField.AssociatedTypes
      F]
    [trait_constr_sqrt_i0 : math.field.traits.IsPrimeField F ]
    (self : (FieldElement F)) :
    RustM
    (core_models.option.Option
      (rust_primitives.hax.Tuple2 (FieldElement F) (FieldElement F)))
    := do
  let
    sqrts : (core_models.option.Option
      (rust_primitives.hax.Tuple2
        (math.field.traits.IsField.BaseType F)
        (math.field.traits.IsField.BaseType F))) ←
    (math.field.traits.IsPrimeField.sqrt F (FieldElement.value self));
  (core_models.option.Impl.map
    (rust_primitives.hax.Tuple2
      (math.field.traits.IsField.BaseType F)
      (math.field.traits.IsField.BaseType F))
    (rust_primitives.hax.Tuple2 (FieldElement F) (FieldElement F))
    ((rust_primitives.hax.Tuple2
      (math.field.traits.IsField.BaseType F)
      (math.field.traits.IsField.BaseType F)) ->
    RustM (rust_primitives.hax.Tuple2 (FieldElement F) (FieldElement F)))
    sqrts
    (fun ⟨sqrt1, sqrt2⟩ =>
      (do
      (pure (rust_primitives.hax.Tuple2.mk
        (FieldElement.mk (value := sqrt1))
        (FieldElement.mk (value := sqrt2)))) :
      RustM (rust_primitives.hax.Tuple2 (FieldElement F) (FieldElement F)))))

--  Returns the Legendre symbol of a field element modulo p
@[spec]
def Impl_33.legendre_symbol
    (F : Type)
    [trait_constr_legendre_symbol_associated_type_i0 :
      math.field.traits.IsPrimeField.AssociatedTypes
      F]
    [trait_constr_legendre_symbol_i0 : math.field.traits.IsPrimeField F ]
    (self : (FieldElement F)) :
    RustM math.field.traits.LegendreSymbol := do
  (math.field.traits.IsPrimeField.legendre_symbol F (FieldElement.value self))

--  Creates a `FieldElement` from a hexstring. It can contain `0x` or not.
--  Returns an `CreationError::InvalidHexString`if the value is not a hexstring.
--  Returns a `CreationError::EmptyString` if the input string is empty.
--  Returns a `CreationError::HexStringIsTooBig` if the the input hex string is bigger than the
--  maximum amount of characters for this element.
--  Returns a `CreationError::CanonicalOutOfRange` if the canonical form of the value is
--  out of the range [0, p-1] where p is the modulus.
@[spec]
def Impl_33.from_hex
    (F : Type)
    [trait_constr_from_hex_associated_type_i0 :
      math.field.traits.IsPrimeField.AssociatedTypes
      F]
    [trait_constr_from_hex_i0 : math.field.traits.IsPrimeField F ]
    (hex_string : String) :
    RustM
    (core_models.result.Result (FieldElement F) math.errors.CreationError)
    := do
  if (← (core_models.str.Impl.is_empty hex_string)) then do
    (pure (core_models.result.Result.Err math.errors.CreationError.EmptyString))
  else do
    match (← (math.field.traits.IsPrimeField.from_hex F hex_string)) with
      | (core_models.result.Result.Ok  value) => do
        (pure (core_models.result.Result.Ok (FieldElement.mk (value := value))))
      | (core_models.result.Result.Err  err) => do
        (pure (core_models.result.Result.Err err))

end math.field.element


namespace math.fft.bowers_fft

--  Process a single block with 2-layer fusion (DIF butterfly).
@[spec]
def process_fused_block
    (F : Type)
    (E : Type)
    [trait_constr_process_fused_block_associated_type_i0 :
      math.field.traits.IsFFTField.AssociatedTypes
      F]
    [trait_constr_process_fused_block_i0 : math.field.traits.IsFFTField F ]
    [trait_constr_process_fused_block_associated_type_i1 :
      math.field.traits.IsSubFieldOf.AssociatedTypes
      F
      E]
    [trait_constr_process_fused_block_i1 : math.field.traits.IsSubFieldOf F E ]
    [trait_constr_process_fused_block_associated_type_i2 :
      math.field.traits.IsField.AssociatedTypes
      E]
    [trait_constr_process_fused_block_i2 : math.field.traits.IsField E ]
    (block : (RustSlice (math.field.element.FieldElement E)))
    (twiddles_l0 : (RustSlice (math.field.element.FieldElement F)))
    (twiddles_l1 : (RustSlice (math.field.element.FieldElement F))) :
    RustM (RustSlice (math.field.element.FieldElement E)) := do
  let block_size : usize ←
    (core_models.slice.Impl.len (math.field.element.FieldElement E) block);
  let quarter : usize ← (block_size >>>? (2 : i32));
  let _ ←
    if true then do
      let _ ←
        if
        (← (!? (← ((← (core_models.slice.Impl.len
            (math.field.element.FieldElement F) twiddles_l0))
          >=? (← ((2 : usize) *? quarter)))))) then do
          let args : (rust_primitives.hax.Tuple2 usize usize) :=
            (rust_primitives.hax.Tuple2.mk
              (← (core_models.slice.Impl.len (math.field.element.FieldElement F)
                twiddles_l0))
              (← ((2 : usize) *? quarter)));
          let args : (RustArray core_models.fmt.rt.Argument 2) :=
            (RustArray.ofVec #v[(← (core_models.fmt.rt.Impl.new_display usize
                                    (rust_primitives.hax.Tuple2._0 args))),
                                  (← (core_models.fmt.rt.Impl.new_display usize
                                    (rust_primitives.hax.Tuple2._1 args)))]);
          (rust_primitives.hax.never_to_any
            (← (core_models.panicking.panic_fmt
              (← (core_models.fmt.rt.Impl_1.new_v1 ((2 : usize)) ((2 : usize))
                (RustArray.ofVec #v["twiddles_l0 too short: ", " < "])
                args)))))
        else do
          (pure rust_primitives.hax.Tuple0.mk);
      (pure rust_primitives.hax.Tuple0.mk)
    else do
      (pure rust_primitives.hax.Tuple0.mk);
  let _ ←
    if true then do
      let _ ←
        if
        (← (!? (← ((← (core_models.slice.Impl.len
            (math.field.element.FieldElement F) twiddles_l1))
          >=? quarter)))) then do
          let args : (rust_primitives.hax.Tuple2 usize usize) :=
            (rust_primitives.hax.Tuple2.mk
              (← (core_models.slice.Impl.len (math.field.element.FieldElement F)
                twiddles_l1))
              quarter);
          let args : (RustArray core_models.fmt.rt.Argument 2) :=
            (RustArray.ofVec #v[(← (core_models.fmt.rt.Impl.new_display usize
                                    (rust_primitives.hax.Tuple2._0 args))),
                                  (← (core_models.fmt.rt.Impl.new_display usize
                                    (rust_primitives.hax.Tuple2._1 args)))]);
          (rust_primitives.hax.never_to_any
            (← (core_models.panicking.panic_fmt
              (← (core_models.fmt.rt.Impl_1.new_v1 ((2 : usize)) ((2 : usize))
                (RustArray.ofVec #v["twiddles_l1 too short: ", " < "])
                args)))))
        else do
          (pure rust_primitives.hax.Tuple0.mk);
      (pure rust_primitives.hax.Tuple0.mk)
    else do
      (pure rust_primitives.hax.Tuple0.mk);
  let block : (RustSlice (math.field.element.FieldElement E)) ←
    (rust_primitives.hax.folds.fold_range
      (0 : usize)
      quarter
      (fun block _ => (do (pure true) : RustM Bool))
      block
      (fun block j =>
        (do
        let i0 : usize := j;
        let i1 : usize ← (j +? quarter);
        let i2 : usize ← (j +? (← ((2 : usize) *? quarter)));
        let i3 : usize ← (j +? (← ((3 : usize) *? quarter)));
        let w0 : (math.field.element.FieldElement F) ← twiddles_l0[j]_?;
        let w1 : (math.field.element.FieldElement F) ←
          twiddles_l0[(← (j +? quarter))]_?;
        let sum_02 : (math.field.element.FieldElement E) ←
          (core_models.ops.arith.Add.add
            (math.field.element.FieldElement E)
            (math.field.element.FieldElement E)
            (← block[i0]_?)
            (← block[i2]_?));
        let diff_02 : (math.field.element.FieldElement E) ←
          (core_models.ops.arith.Sub.sub
            (math.field.element.FieldElement E)
            (math.field.element.FieldElement E)
            (← block[i0]_?)
            (← block[i2]_?));
        let diff_02_w : (math.field.element.FieldElement E) ←
          (core_models.ops.arith.Mul.mul
            (math.field.element.FieldElement F)
            (math.field.element.FieldElement E) w0 diff_02);
        let sum_13 : (math.field.element.FieldElement E) ←
          (core_models.ops.arith.Add.add
            (math.field.element.FieldElement E)
            (math.field.element.FieldElement E)
            (← block[i1]_?)
            (← block[i3]_?));
        let diff_13 : (math.field.element.FieldElement E) ←
          (core_models.ops.arith.Sub.sub
            (math.field.element.FieldElement E)
            (math.field.element.FieldElement E)
            (← block[i1]_?)
            (← block[i3]_?));
        let diff_13_w : (math.field.element.FieldElement E) ←
          (core_models.ops.arith.Mul.mul
            (math.field.element.FieldElement F)
            (math.field.element.FieldElement E) w1 diff_13);
        let w2 : (math.field.element.FieldElement F) ← twiddles_l1[j]_?;
        let final_0 : (math.field.element.FieldElement E) ←
          (core_models.ops.arith.Add.add
            (math.field.element.FieldElement E)
            (math.field.element.FieldElement E) sum_02 sum_13);
        let diff_sums : (math.field.element.FieldElement E) ←
          (core_models.ops.arith.Sub.sub
            (math.field.element.FieldElement E)
            (math.field.element.FieldElement E) sum_02 sum_13);
        let final_1 : (math.field.element.FieldElement E) ←
          (core_models.ops.arith.Mul.mul
            (math.field.element.FieldElement F)
            (math.field.element.FieldElement E) w2 diff_sums);
        let final_2 : (math.field.element.FieldElement E) ←
          (core_models.ops.arith.Add.add
            (math.field.element.FieldElement E)
            (math.field.element.FieldElement E) diff_02_w diff_13_w);
        let diff_diffs : (math.field.element.FieldElement E) ←
          (core_models.ops.arith.Sub.sub
            (math.field.element.FieldElement E)
            (math.field.element.FieldElement E) diff_02_w diff_13_w);
        let final_3 : (math.field.element.FieldElement E) ←
          (core_models.ops.arith.Mul.mul
            (math.field.element.FieldElement F)
            (math.field.element.FieldElement E) w2 diff_diffs);
        let block : (RustSlice (math.field.element.FieldElement E)) ←
          (rust_primitives.hax.monomorphized_update_at.update_at_usize
            block
            i0
            final_0);
        let block : (RustSlice (math.field.element.FieldElement E)) ←
          (rust_primitives.hax.monomorphized_update_at.update_at_usize
            block
            i1
            final_1);
        let block : (RustSlice (math.field.element.FieldElement E)) ←
          (rust_primitives.hax.monomorphized_update_at.update_at_usize
            block
            i2
            final_2);
        let block : (RustSlice (math.field.element.FieldElement E)) ←
          (rust_primitives.hax.monomorphized_update_at.update_at_usize
            block
            i3
            final_3);
        (pure block) :
        RustM (RustSlice (math.field.element.FieldElement E)))));
  (pure block)

--  Process a single block with 3-layer fusion (DIF radix-8 butterfly).
-- 
--  Processes 8 elements through 3 DIF butterfly layers at once, keeping all
--  intermediate values in registers. Reduces memory round-trips compared to
--  2-layer fusion: 8 reads + 8 writes instead of 8+8+8+8 for separate layers.
@[spec]
def process_triple_fused_block
    (F : Type)
    (E : Type)
    [trait_constr_process_triple_fused_block_associated_type_i0 :
      math.field.traits.IsFFTField.AssociatedTypes
      F]
    [trait_constr_process_triple_fused_block_i0 : math.field.traits.IsFFTField
      F
      ]
    [trait_constr_process_triple_fused_block_associated_type_i1 :
      math.field.traits.IsSubFieldOf.AssociatedTypes
      F
      E]
    [trait_constr_process_triple_fused_block_i1 : math.field.traits.IsSubFieldOf
      F
      E
      ]
    [trait_constr_process_triple_fused_block_associated_type_i2 :
      math.field.traits.IsField.AssociatedTypes
      E]
    [trait_constr_process_triple_fused_block_i2 : math.field.traits.IsField E ]
    (block : (RustSlice (math.field.element.FieldElement E)))
    (twiddles_l0 : (RustSlice (math.field.element.FieldElement F)))
    (twiddles_l1 : (RustSlice (math.field.element.FieldElement F)))
    (twiddles_l2 : (RustSlice (math.field.element.FieldElement F))) :
    RustM (RustSlice (math.field.element.FieldElement E)) := do
  let block_size : usize ←
    (core_models.slice.Impl.len (math.field.element.FieldElement E) block);
  let eighth : usize ← (block_size >>>? (3 : i32));
  let block : (RustSlice (math.field.element.FieldElement E)) ←
    (rust_primitives.hax.folds.fold_range
      (0 : usize)
      eighth
      (fun block _ => (do (pure true) : RustM Bool))
      block
      (fun block j =>
        (do
        let i0 : usize := j;
        let i1 : usize ← (j +? eighth);
        let i2 : usize ← (j +? (← ((2 : usize) *? eighth)));
        let i3 : usize ← (j +? (← ((3 : usize) *? eighth)));
        let i4 : usize ← (j +? (← ((4 : usize) *? eighth)));
        let i5 : usize ← (j +? (← ((5 : usize) *? eighth)));
        let i6 : usize ← (j +? (← ((6 : usize) *? eighth)));
        let i7 : usize ← (j +? (← ((7 : usize) *? eighth)));
        let w0_0 : (math.field.element.FieldElement F) ← twiddles_l0[j]_?;
        let w0_1 : (math.field.element.FieldElement F) ←
          twiddles_l0[(← (j +? eighth))]_?;
        let w0_2 : (math.field.element.FieldElement F) ←
          twiddles_l0[(← (j +? (← ((2 : usize) *? eighth))))]_?;
        let w0_3 : (math.field.element.FieldElement F) ←
          twiddles_l0[(← (j +? (← ((3 : usize) *? eighth))))]_?;
        let s04 : (math.field.element.FieldElement E) ←
          (core_models.ops.arith.Add.add
            (math.field.element.FieldElement E)
            (math.field.element.FieldElement E)
            (← block[i0]_?)
            (← block[i4]_?));
        let d04 : (math.field.element.FieldElement E) ←
          (core_models.ops.arith.Mul.mul
            (math.field.element.FieldElement F)
            (math.field.element.FieldElement E)
            w0_0
            (← (core_models.ops.arith.Sub.sub
              (math.field.element.FieldElement E)
              (math.field.element.FieldElement E)
              (← block[i0]_?)
              (← block[i4]_?))));
        let s15 : (math.field.element.FieldElement E) ←
          (core_models.ops.arith.Add.add
            (math.field.element.FieldElement E)
            (math.field.element.FieldElement E)
            (← block[i1]_?)
            (← block[i5]_?));
        let d15 : (math.field.element.FieldElement E) ←
          (core_models.ops.arith.Mul.mul
            (math.field.element.FieldElement F)
            (math.field.element.FieldElement E)
            w0_1
            (← (core_models.ops.arith.Sub.sub
              (math.field.element.FieldElement E)
              (math.field.element.FieldElement E)
              (← block[i1]_?)
              (← block[i5]_?))));
        let s26 : (math.field.element.FieldElement E) ←
          (core_models.ops.arith.Add.add
            (math.field.element.FieldElement E)
            (math.field.element.FieldElement E)
            (← block[i2]_?)
            (← block[i6]_?));
        let d26 : (math.field.element.FieldElement E) ←
          (core_models.ops.arith.Mul.mul
            (math.field.element.FieldElement F)
            (math.field.element.FieldElement E)
            w0_2
            (← (core_models.ops.arith.Sub.sub
              (math.field.element.FieldElement E)
              (math.field.element.FieldElement E)
              (← block[i2]_?)
              (← block[i6]_?))));
        let s37 : (math.field.element.FieldElement E) ←
          (core_models.ops.arith.Add.add
            (math.field.element.FieldElement E)
            (math.field.element.FieldElement E)
            (← block[i3]_?)
            (← block[i7]_?));
        let d37 : (math.field.element.FieldElement E) ←
          (core_models.ops.arith.Mul.mul
            (math.field.element.FieldElement F)
            (math.field.element.FieldElement E)
            w0_3
            (← (core_models.ops.arith.Sub.sub
              (math.field.element.FieldElement E)
              (math.field.element.FieldElement E)
              (← block[i3]_?)
              (← block[i7]_?))));
        let w1_0 : (math.field.element.FieldElement F) ← twiddles_l1[j]_?;
        let w1_1 : (math.field.element.FieldElement F) ←
          twiddles_l1[(← (j +? eighth))]_?;
        let ss02 : (math.field.element.FieldElement E) ←
          (core_models.ops.arith.Add.add
            (math.field.element.FieldElement E)
            (math.field.element.FieldElement E) s04 s26);
        let ds02 : (math.field.element.FieldElement E) ←
          (core_models.ops.arith.Mul.mul
            (math.field.element.FieldElement F)
            (math.field.element.FieldElement E)
            w1_0
            (← (core_models.ops.arith.Sub.sub
              (math.field.element.FieldElement E)
              (math.field.element.FieldElement E) s04 s26)));
        let ss13 : (math.field.element.FieldElement E) ←
          (core_models.ops.arith.Add.add
            (math.field.element.FieldElement E)
            (math.field.element.FieldElement E) s15 s37);
        let ds13 : (math.field.element.FieldElement E) ←
          (core_models.ops.arith.Mul.mul
            (math.field.element.FieldElement F)
            (math.field.element.FieldElement E)
            w1_1
            (← (core_models.ops.arith.Sub.sub
              (math.field.element.FieldElement E)
              (math.field.element.FieldElement E) s15 s37)));
        let sd02 : (math.field.element.FieldElement E) ←
          (core_models.ops.arith.Add.add
            (math.field.element.FieldElement E)
            (math.field.element.FieldElement E) d04 d26);
        let dd02 : (math.field.element.FieldElement E) ←
          (core_models.ops.arith.Mul.mul
            (math.field.element.FieldElement F)
            (math.field.element.FieldElement E)
            w1_0
            (← (core_models.ops.arith.Sub.sub
              (math.field.element.FieldElement E)
              (math.field.element.FieldElement E) d04 d26)));
        let sd13 : (math.field.element.FieldElement E) ←
          (core_models.ops.arith.Add.add
            (math.field.element.FieldElement E)
            (math.field.element.FieldElement E) d15 d37);
        let dd13 : (math.field.element.FieldElement E) ←
          (core_models.ops.arith.Mul.mul
            (math.field.element.FieldElement F)
            (math.field.element.FieldElement E)
            w1_1
            (← (core_models.ops.arith.Sub.sub
              (math.field.element.FieldElement E)
              (math.field.element.FieldElement E) d15 d37)));
        let w2 : (math.field.element.FieldElement F) ← twiddles_l2[j]_?;
        let block : (RustSlice (math.field.element.FieldElement E)) ←
          (rust_primitives.hax.monomorphized_update_at.update_at_usize
            block
            i0
            (← (core_models.ops.arith.Add.add
              (math.field.element.FieldElement E)
              (math.field.element.FieldElement E) ss02 ss13)));
        let block : (RustSlice (math.field.element.FieldElement E)) ←
          (rust_primitives.hax.monomorphized_update_at.update_at_usize
            block
            i1
            (← (core_models.ops.arith.Mul.mul
              (math.field.element.FieldElement F)
              (math.field.element.FieldElement E)
              w2
              (← (core_models.ops.arith.Sub.sub
                (math.field.element.FieldElement E)
                (math.field.element.FieldElement E) ss02 ss13)))));
        let block : (RustSlice (math.field.element.FieldElement E)) ←
          (rust_primitives.hax.monomorphized_update_at.update_at_usize
            block
            i2
            (← (core_models.ops.arith.Add.add
              (math.field.element.FieldElement E)
              (math.field.element.FieldElement E) ds02 ds13)));
        let block : (RustSlice (math.field.element.FieldElement E)) ←
          (rust_primitives.hax.monomorphized_update_at.update_at_usize
            block
            i3
            (← (core_models.ops.arith.Mul.mul
              (math.field.element.FieldElement F)
              (math.field.element.FieldElement E)
              w2
              (← (core_models.ops.arith.Sub.sub
                (math.field.element.FieldElement E)
                (math.field.element.FieldElement E) ds02 ds13)))));
        let block : (RustSlice (math.field.element.FieldElement E)) ←
          (rust_primitives.hax.monomorphized_update_at.update_at_usize
            block
            i4
            (← (core_models.ops.arith.Add.add
              (math.field.element.FieldElement E)
              (math.field.element.FieldElement E) sd02 sd13)));
        let block : (RustSlice (math.field.element.FieldElement E)) ←
          (rust_primitives.hax.monomorphized_update_at.update_at_usize
            block
            i5
            (← (core_models.ops.arith.Mul.mul
              (math.field.element.FieldElement F)
              (math.field.element.FieldElement E)
              w2
              (← (core_models.ops.arith.Sub.sub
                (math.field.element.FieldElement E)
                (math.field.element.FieldElement E) sd02 sd13)))));
        let block : (RustSlice (math.field.element.FieldElement E)) ←
          (rust_primitives.hax.monomorphized_update_at.update_at_usize
            block
            i6
            (← (core_models.ops.arith.Add.add
              (math.field.element.FieldElement E)
              (math.field.element.FieldElement E) dd02 dd13)));
        let block : (RustSlice (math.field.element.FieldElement E)) ←
          (rust_primitives.hax.monomorphized_update_at.update_at_usize
            block
            i7
            (← (core_models.ops.arith.Mul.mul
              (math.field.element.FieldElement F)
              (math.field.element.FieldElement E)
              w2
              (← (core_models.ops.arith.Sub.sub
                (math.field.element.FieldElement E)
                (math.field.element.FieldElement E) dd02 dd13)))));
        (pure block) :
        RustM (RustSlice (math.field.element.FieldElement E)))));
  (pure block)

--  Shared implementation for `new` and `new_inverse`.
@[spec]
def Impl.build
    (F : Type)
    [trait_constr_build_associated_type_i0 :
      math.field.traits.IsFFTField.AssociatedTypes
      F]
    [trait_constr_build_i0 : math.field.traits.IsFFTField F ]
    (order : u64)
    (root : (math.field.element.FieldElement F)) :
    RustM (core_models.option.Option (LayerTwiddles F)) := do
  if (← (order >? MAX_FFT_ORDER)) then do
    (pure core_models.option.Option.None)
  else do
    let n : usize ← ((1 : usize) <<<? order);
    let
      layers : (alloc.vec.Vec
        (alloc.vec.Vec (math.field.element.FieldElement F) alloc.alloc.Global)
        alloc.alloc.Global) ←
      (alloc.vec.Impl.with_capacity
        (alloc.vec.Vec (math.field.element.FieldElement F) alloc.alloc.Global)
        (← (rust_primitives.hax.cast_op order : RustM usize)));
    let
      layers : (alloc.vec.Vec
        (alloc.vec.Vec (math.field.element.FieldElement F) alloc.alloc.Global)
        alloc.alloc.Global) ←
      (rust_primitives.hax.folds.fold_range
        (0 : usize)
        (← (rust_primitives.hax.cast_op order : RustM usize))
        (fun layers _ => (do (pure true) : RustM Bool))
        layers
        (fun layers layer =>
          (do
          let _ ←
            if true then do
              let _ ←
                if
                (← (!? (← (layer
                  <? (← (rust_primitives.hax.cast_op
                    core_models.num.Impl_11.BITS :
                    RustM usize)))))) then do
                  (rust_primitives.hax.never_to_any
                    (← (core_models.panicking.panic_fmt
                      (← (core_models.fmt.rt.Impl_1.new_const ((1 : usize))
                        (RustArray.ofVec #v["Layer index exceeds shift limit"]))))))
                else do
                  (pure rust_primitives.hax.Tuple0.mk);
              (pure rust_primitives.hax.Tuple0.mk)
            else do
              (pure rust_primitives.hax.Tuple0.mk);
          let stride : usize ← ((1 : usize) <<<? layer);
          let count : usize ← (n >>>? (← (layer +? (1 : usize))));
          let
            layer_twiddles : (alloc.vec.Vec
              (math.field.element.FieldElement F)
              alloc.alloc.Global) ←
            (alloc.vec.Impl.with_capacity (math.field.element.FieldElement F)
              count);
          let w_stride : (math.field.element.FieldElement F) ←
            (math.field.element.Impl_32.pow F u64
              root
              (← (rust_primitives.hax.cast_op stride : RustM u64)));
          let current : (math.field.element.FieldElement F) ←
            (math.field.element.Impl_32.one F rust_primitives.hax.Tuple0.mk);
          let ⟨current, layer_twiddles⟩ ←
            (rust_primitives.hax.folds.fold_range
              (0 : usize)
              count
              (fun ⟨current, layer_twiddles⟩ _ => (do (pure true) : RustM Bool))
              (rust_primitives.hax.Tuple2.mk current layer_twiddles)
              (fun ⟨current, layer_twiddles⟩ _ =>
                (do
                let
                  layer_twiddles : (alloc.vec.Vec
                    (math.field.element.FieldElement F)
                    alloc.alloc.Global) ←
                  (alloc.vec.Impl_1.push
                    (math.field.element.FieldElement F)
                    alloc.alloc.Global
                    layer_twiddles
                    (← (core_models.clone.Clone.clone
                      (math.field.element.FieldElement F) current)));
                let current : (math.field.element.FieldElement F) ←
                  (core_models.ops.arith.Mul.mul
                    (math.field.element.FieldElement F)
                    (math.field.element.FieldElement F) current w_stride);
                (pure (rust_primitives.hax.Tuple2.mk current layer_twiddles)) :
                RustM
                (rust_primitives.hax.Tuple2
                  (math.field.element.FieldElement F)
                  (alloc.vec.Vec
                    (math.field.element.FieldElement F)
                    alloc.alloc.Global)))));
          let
            layers : (alloc.vec.Vec
              (alloc.vec.Vec
                (math.field.element.FieldElement F)
                alloc.alloc.Global)
              alloc.alloc.Global) ←
            (alloc.vec.Impl_1.push
              (alloc.vec.Vec
                (math.field.element.FieldElement F)
                alloc.alloc.Global)
              alloc.alloc.Global layers layer_twiddles);
          (pure layers) :
          RustM
          (alloc.vec.Vec
            (alloc.vec.Vec
              (math.field.element.FieldElement F)
              alloc.alloc.Global)
            alloc.alloc.Global))));
    (pure (core_models.option.Option.Some
      (LayerTwiddles.mk (layers := layers))))

--  Compute layer-specific twiddles from primitive root of unity.
-- 
--  For an FFT of size n = 2^order, layer k needs n/2^(k+1) twiddles.
--  The twiddles for layer k are: w^0, w^(2^k), w^(2*2^k), w^(3*2^k), ...
-- 
--  # Errors
--  Returns `None` if:
--  - `order` exceeds the maximum supported value (would cause integer overflow)
--  - The field doesn't have a primitive root of unity for the given order
-- 
--  # Example
--  ```ignore
--  let layer_twiddles = LayerTwiddles::<GoldilocksField>::new(10)
--      .expect("Failed to create twiddles for order 10");
--  ```
@[spec]
def Impl.new
    (F : Type)
    [trait_constr_new_associated_type_i0 :
      math.field.traits.IsFFTField.AssociatedTypes
      F]
    [trait_constr_new_i0 : math.field.traits.IsFFTField F ]
    (order : u64) :
    RustM (core_models.option.Option (LayerTwiddles F)) := do
  match
    (← (core_models.result.Impl.ok
      (math.field.element.FieldElement F)
      math.field.errors.FieldError
      (← (math.field.traits.IsFFTField.get_primitive_root_of_unity F order))))
  with
    | (core_models.option.Option.Some  root) => do (Impl.build F order root)
    | (core_models.option.Option.None ) => do
      (pure core_models.option.Option.None)

--  Compute layer-specific twiddles from the **inverse** primitive root of unity.
-- 
--  This is used for the inverse FFT (IFFT). The inverse twiddles are computed
--  from w^(-1) where w is the primitive root of unity.
-- 
--  # Errors
--  Returns `None` if:
--  - `order` exceeds the maximum supported value (would cause integer overflow)
--  - The field doesn't have a primitive root of unity for the given order
-- 
--  # Example
--  ```ignore
--  let inv_twiddles = LayerTwiddles::<GoldilocksField>::new_inverse(10)
--      .expect("Failed to create inverse twiddles for order 10");
--  ```
@[spec]
def Impl.new_inverse
    (F : Type)
    [trait_constr_new_inverse_associated_type_i0 :
      math.field.traits.IsFFTField.AssociatedTypes
      F]
    [trait_constr_new_inverse_i0 : math.field.traits.IsFFTField F ]
    (order : u64) :
    RustM (core_models.option.Option (LayerTwiddles F)) := do
  match
    (← (core_models.result.Impl.ok
      (math.field.element.FieldElement F)
      math.field.errors.FieldError
      (← (math.field.traits.IsFFTField.get_primitive_root_of_unity F order))))
  with
    | (core_models.option.Option.Some  root) => do
      match
        (← (core_models.result.Impl.ok
          (math.field.element.FieldElement F)
          math.field.errors.FieldError
          (← (math.field.element.Impl_32.inv F root))))
      with
        | (core_models.option.Option.Some  inv_root) => do
          (Impl.build F order inv_root)
        | (core_models.option.Option.None ) => do
          (pure core_models.option.Option.None)
    | (core_models.option.Option.None ) => do
      (pure core_models.option.Option.None)

--  Get the twiddles for a specific layer.
-- 
--  # Panics
--  Panics if `layer >= self.layers.len()`.
@[spec]
def Impl.get_layer
    (F : Type)
    [trait_constr_get_layer_associated_type_i0 :
      math.field.traits.IsFFTField.AssociatedTypes
      F]
    [trait_constr_get_layer_i0 : math.field.traits.IsFFTField F ]
    (self : (LayerTwiddles F))
    (layer : usize) :
    RustM (RustSlice (math.field.element.FieldElement F)) := do
  let _ ←
    if
    (← (!? (← (layer
      <? (← (alloc.vec.Impl_1.len
        (alloc.vec.Vec (math.field.element.FieldElement F) alloc.alloc.Global)
        alloc.alloc.Global (LayerTwiddles.layers self))))))) then do
      let args : (rust_primitives.hax.Tuple2 usize usize) :=
        (rust_primitives.hax.Tuple2.mk
          layer
          (← (alloc.vec.Impl_1.len
            (alloc.vec.Vec
              (math.field.element.FieldElement F)
              alloc.alloc.Global)
            alloc.alloc.Global (LayerTwiddles.layers self))));
      let args : (RustArray core_models.fmt.rt.Argument 2) :=
        (RustArray.ofVec #v[(← (core_models.fmt.rt.Impl.new_display usize
                                (rust_primitives.hax.Tuple2._0 args))),
                              (← (core_models.fmt.rt.Impl.new_display usize
                                (rust_primitives.hax.Tuple2._1 args)))]);
      (rust_primitives.hax.never_to_any
        (← (core_models.panicking.panic_fmt
          (← (core_models.fmt.rt.Impl_1.new_v1 ((2 : usize)) ((2 : usize))
            (RustArray.ofVec #v["Layer index out of bounds: ", " >= "])
            args)))))
    else do
      (pure rust_primitives.hax.Tuple0.mk);
  (core_models.ops.deref.Deref.deref
    (alloc.vec.Vec (math.field.element.FieldElement F) alloc.alloc.Global)
    (← (LayerTwiddles.layers self)[layer]_?))

--  Returns the number of layers (equal to the FFT order).
@[spec]
def Impl.num_layers
    (F : Type)
    [trait_constr_num_layers_associated_type_i0 :
      math.field.traits.IsFFTField.AssociatedTypes
      F]
    [trait_constr_num_layers_i0 : math.field.traits.IsFFTField F ]
    (self : (LayerTwiddles F)) :
    RustM usize := do
  (alloc.vec.Impl_1.len
    (alloc.vec.Vec (math.field.element.FieldElement F) alloc.alloc.Global)
    alloc.alloc.Global (LayerTwiddles.layers self))

--  Process a single block with 2-layer IFFT fusion (DIT butterfly).
-- 
--  Processes two consecutive IFFT layers in a single pass. The `twiddles_hi` are
--  for the higher-numbered layer (processed first in DIT order) and `twiddles_lo`
--  are for the lower-numbered layer (processed second).
@[spec]
def process_ifft_fused_block
    (F : Type)
    (E : Type)
    [trait_constr_process_ifft_fused_block_associated_type_i0 :
      math.field.traits.IsFFTField.AssociatedTypes
      F]
    [trait_constr_process_ifft_fused_block_i0 : math.field.traits.IsFFTField F ]
    [trait_constr_process_ifft_fused_block_associated_type_i1 :
      math.field.traits.IsSubFieldOf.AssociatedTypes
      F
      E]
    [trait_constr_process_ifft_fused_block_i1 : math.field.traits.IsSubFieldOf
      F
      E
      ]
    [trait_constr_process_ifft_fused_block_associated_type_i2 :
      math.field.traits.IsField.AssociatedTypes
      E]
    [trait_constr_process_ifft_fused_block_i2 : math.field.traits.IsField E ]
    (block : (RustSlice (math.field.element.FieldElement E)))
    (twiddles_hi : (RustSlice (math.field.element.FieldElement F)))
    (twiddles_lo : (RustSlice (math.field.element.FieldElement F))) :
    RustM (RustSlice (math.field.element.FieldElement E)) := do
  let block_size : usize ←
    (core_models.slice.Impl.len (math.field.element.FieldElement E) block);
  let quarter : usize ← (block_size >>>? (2 : i32));
  let _ ←
    if true then do
      let _ ←
        if
        (← (!? (← ((← (core_models.slice.Impl.len
            (math.field.element.FieldElement F) twiddles_hi))
          >=? quarter)))) then do
          let args : (rust_primitives.hax.Tuple2 usize usize) :=
            (rust_primitives.hax.Tuple2.mk
              (← (core_models.slice.Impl.len (math.field.element.FieldElement F)
                twiddles_hi))
              quarter);
          let args : (RustArray core_models.fmt.rt.Argument 2) :=
            (RustArray.ofVec #v[(← (core_models.fmt.rt.Impl.new_display usize
                                    (rust_primitives.hax.Tuple2._0 args))),
                                  (← (core_models.fmt.rt.Impl.new_display usize
                                    (rust_primitives.hax.Tuple2._1 args)))]);
          (rust_primitives.hax.never_to_any
            (← (core_models.panicking.panic_fmt
              (← (core_models.fmt.rt.Impl_1.new_v1 ((2 : usize)) ((2 : usize))
                (RustArray.ofVec #v["twiddles_hi too short: ", " < "])
                args)))))
        else do
          (pure rust_primitives.hax.Tuple0.mk);
      (pure rust_primitives.hax.Tuple0.mk)
    else do
      (pure rust_primitives.hax.Tuple0.mk);
  let _ ←
    if true then do
      let _ ←
        if
        (← (!? (← ((← (core_models.slice.Impl.len
            (math.field.element.FieldElement F) twiddles_lo))
          >=? (← ((2 : usize) *? quarter)))))) then do
          let args : (rust_primitives.hax.Tuple2 usize usize) :=
            (rust_primitives.hax.Tuple2.mk
              (← (core_models.slice.Impl.len (math.field.element.FieldElement F)
                twiddles_lo))
              (← ((2 : usize) *? quarter)));
          let args : (RustArray core_models.fmt.rt.Argument 2) :=
            (RustArray.ofVec #v[(← (core_models.fmt.rt.Impl.new_display usize
                                    (rust_primitives.hax.Tuple2._0 args))),
                                  (← (core_models.fmt.rt.Impl.new_display usize
                                    (rust_primitives.hax.Tuple2._1 args)))]);
          (rust_primitives.hax.never_to_any
            (← (core_models.panicking.panic_fmt
              (← (core_models.fmt.rt.Impl_1.new_v1 ((2 : usize)) ((2 : usize))
                (RustArray.ofVec #v["twiddles_lo too short: ", " < "])
                args)))))
        else do
          (pure rust_primitives.hax.Tuple0.mk);
      (pure rust_primitives.hax.Tuple0.mk)
    else do
      (pure rust_primitives.hax.Tuple0.mk);
  let block : (RustSlice (math.field.element.FieldElement E)) ←
    (rust_primitives.hax.folds.fold_range
      (0 : usize)
      quarter
      (fun block _ => (do (pure true) : RustM Bool))
      block
      (fun block j =>
        (do
        let i0 : usize := j;
        let i1 : usize ← (j +? quarter);
        let i2 : usize ← (j +? (← ((2 : usize) *? quarter)));
        let i3 : usize ← (j +? (← ((3 : usize) *? quarter)));
        let w_hi : (math.field.element.FieldElement F) ← twiddles_hi[j]_?;
        let bw0 : (math.field.element.FieldElement E) ←
          (core_models.ops.arith.Mul.mul
            (math.field.element.FieldElement F)
            (math.field.element.FieldElement E) w_hi (← block[i1]_?));
        let a0 : (math.field.element.FieldElement E) ←
          (core_models.ops.arith.Add.add
            (math.field.element.FieldElement E)
            (math.field.element.FieldElement E) (← block[i0]_?) bw0);
        let b0 : (math.field.element.FieldElement E) ←
          (core_models.ops.arith.Sub.sub
            (math.field.element.FieldElement E)
            (math.field.element.FieldElement E) (← block[i0]_?) bw0);
        let bw1 : (math.field.element.FieldElement E) ←
          (core_models.ops.arith.Mul.mul
            (math.field.element.FieldElement F)
            (math.field.element.FieldElement E) w_hi (← block[i3]_?));
        let a1 : (math.field.element.FieldElement E) ←
          (core_models.ops.arith.Add.add
            (math.field.element.FieldElement E)
            (math.field.element.FieldElement E) (← block[i2]_?) bw1);
        let b1 : (math.field.element.FieldElement E) ←
          (core_models.ops.arith.Sub.sub
            (math.field.element.FieldElement E)
            (math.field.element.FieldElement E) (← block[i2]_?) bw1);
        let w_lo_0 : (math.field.element.FieldElement F) ← twiddles_lo[j]_?;
        let w_lo_1 : (math.field.element.FieldElement F) ←
          twiddles_lo[(← (j +? quarter))]_?;
        let bw2 : (math.field.element.FieldElement E) ←
          (core_models.ops.arith.Mul.mul
            (math.field.element.FieldElement F)
            (math.field.element.FieldElement E) w_lo_0 a1);
        let block : (RustSlice (math.field.element.FieldElement E)) ←
          (rust_primitives.hax.monomorphized_update_at.update_at_usize
            block
            i0
            (← (core_models.ops.arith.Add.add
              (math.field.element.FieldElement E)
              (math.field.element.FieldElement E) a0 bw2)));
        let block : (RustSlice (math.field.element.FieldElement E)) ←
          (rust_primitives.hax.monomorphized_update_at.update_at_usize
            block
            i2
            (← (core_models.ops.arith.Sub.sub
              (math.field.element.FieldElement E)
              (math.field.element.FieldElement E) a0 bw2)));
        let bw3 : (math.field.element.FieldElement E) ←
          (core_models.ops.arith.Mul.mul
            (math.field.element.FieldElement F)
            (math.field.element.FieldElement E) w_lo_1 b1);
        let block : (RustSlice (math.field.element.FieldElement E)) ←
          (rust_primitives.hax.monomorphized_update_at.update_at_usize
            block
            i1
            (← (core_models.ops.arith.Add.add
              (math.field.element.FieldElement E)
              (math.field.element.FieldElement E) b0 bw3)));
        let block : (RustSlice (math.field.element.FieldElement E)) ←
          (rust_primitives.hax.monomorphized_update_at.update_at_usize
            block
            i3
            (← (core_models.ops.arith.Sub.sub
              (math.field.element.FieldElement E)
              (math.field.element.FieldElement E) b0 bw3)));
        (pure block) :
        RustM (RustSlice (math.field.element.FieldElement E)))));
  (pure block)

--  Process a single block with 3-layer IFFT fusion (DIT radix-8 butterfly).
@[spec]
def process_ifft_triple_fused_block
    (F : Type)
    (E : Type)
    [trait_constr_process_ifft_triple_fused_block_associated_type_i0 :
      math.field.traits.IsFFTField.AssociatedTypes
      F]
    [trait_constr_process_ifft_triple_fused_block_i0 :
      math.field.traits.IsFFTField
      F
      ]
    [trait_constr_process_ifft_triple_fused_block_associated_type_i1 :
      math.field.traits.IsSubFieldOf.AssociatedTypes
      F
      E]
    [trait_constr_process_ifft_triple_fused_block_i1 :
      math.field.traits.IsSubFieldOf
      F
      E
      ]
    [trait_constr_process_ifft_triple_fused_block_associated_type_i2 :
      math.field.traits.IsField.AssociatedTypes
      E]
    [trait_constr_process_ifft_triple_fused_block_i2 : math.field.traits.IsField
      E
      ]
    (block : (RustSlice (math.field.element.FieldElement E)))
    (twiddles_hi : (RustSlice (math.field.element.FieldElement F)))
    (twiddles_mid : (RustSlice (math.field.element.FieldElement F)))
    (twiddles_lo : (RustSlice (math.field.element.FieldElement F))) :
    RustM (RustSlice (math.field.element.FieldElement E)) := do
  let block_size : usize ←
    (core_models.slice.Impl.len (math.field.element.FieldElement E) block);
  let eighth : usize ← (block_size >>>? (3 : i32));
  let block : (RustSlice (math.field.element.FieldElement E)) ←
    (rust_primitives.hax.folds.fold_range
      (0 : usize)
      eighth
      (fun block _ => (do (pure true) : RustM Bool))
      block
      (fun block j =>
        (do
        let i0 : usize := j;
        let i1 : usize ← (j +? eighth);
        let i2 : usize ← (j +? (← ((2 : usize) *? eighth)));
        let i3 : usize ← (j +? (← ((3 : usize) *? eighth)));
        let i4 : usize ← (j +? (← ((4 : usize) *? eighth)));
        let i5 : usize ← (j +? (← ((5 : usize) *? eighth)));
        let i6 : usize ← (j +? (← ((6 : usize) *? eighth)));
        let i7 : usize ← (j +? (← ((7 : usize) *? eighth)));
        let w_hi : (math.field.element.FieldElement F) ← twiddles_hi[j]_?;
        let bw01 : (math.field.element.FieldElement E) ←
          (core_models.ops.arith.Mul.mul
            (math.field.element.FieldElement F)
            (math.field.element.FieldElement E) w_hi (← block[i1]_?));
        let a01 : (math.field.element.FieldElement E) ←
          (core_models.ops.arith.Add.add
            (math.field.element.FieldElement E)
            (math.field.element.FieldElement E) (← block[i0]_?) bw01);
        let b01 : (math.field.element.FieldElement E) ←
          (core_models.ops.arith.Sub.sub
            (math.field.element.FieldElement E)
            (math.field.element.FieldElement E) (← block[i0]_?) bw01);
        let bw23 : (math.field.element.FieldElement E) ←
          (core_models.ops.arith.Mul.mul
            (math.field.element.FieldElement F)
            (math.field.element.FieldElement E) w_hi (← block[i3]_?));
        let a23 : (math.field.element.FieldElement E) ←
          (core_models.ops.arith.Add.add
            (math.field.element.FieldElement E)
            (math.field.element.FieldElement E) (← block[i2]_?) bw23);
        let b23 : (math.field.element.FieldElement E) ←
          (core_models.ops.arith.Sub.sub
            (math.field.element.FieldElement E)
            (math.field.element.FieldElement E) (← block[i2]_?) bw23);
        let bw45 : (math.field.element.FieldElement E) ←
          (core_models.ops.arith.Mul.mul
            (math.field.element.FieldElement F)
            (math.field.element.FieldElement E) w_hi (← block[i5]_?));
        let a45 : (math.field.element.FieldElement E) ←
          (core_models.ops.arith.Add.add
            (math.field.element.FieldElement E)
            (math.field.element.FieldElement E) (← block[i4]_?) bw45);
        let b45 : (math.field.element.FieldElement E) ←
          (core_models.ops.arith.Sub.sub
            (math.field.element.FieldElement E)
            (math.field.element.FieldElement E) (← block[i4]_?) bw45);
        let bw67 : (math.field.element.FieldElement E) ←
          (core_models.ops.arith.Mul.mul
            (math.field.element.FieldElement F)
            (math.field.element.FieldElement E) w_hi (← block[i7]_?));
        let a67 : (math.field.element.FieldElement E) ←
          (core_models.ops.arith.Add.add
            (math.field.element.FieldElement E)
            (math.field.element.FieldElement E) (← block[i6]_?) bw67);
        let b67 : (math.field.element.FieldElement E) ←
          (core_models.ops.arith.Sub.sub
            (math.field.element.FieldElement E)
            (math.field.element.FieldElement E) (← block[i6]_?) bw67);
        let w_mid_0 : (math.field.element.FieldElement F) ← twiddles_mid[j]_?;
        let w_mid_1 : (math.field.element.FieldElement F) ←
          twiddles_mid[(← (j +? eighth))]_?;
        let bw_m0 : (math.field.element.FieldElement E) ←
          (core_models.ops.arith.Mul.mul
            (math.field.element.FieldElement F)
            (math.field.element.FieldElement E) w_mid_0 a23);
        let aa0 : (math.field.element.FieldElement E) ←
          (core_models.ops.arith.Add.add
            (math.field.element.FieldElement E)
            (math.field.element.FieldElement E) a01 bw_m0);
        let ab0 : (math.field.element.FieldElement E) ←
          (core_models.ops.arith.Sub.sub
            (math.field.element.FieldElement E)
            (math.field.element.FieldElement E) a01 bw_m0);
        let bw_m1 : (math.field.element.FieldElement E) ←
          (core_models.ops.arith.Mul.mul
            (math.field.element.FieldElement F)
            (math.field.element.FieldElement E) w_mid_1 b23);
        let ba0 : (math.field.element.FieldElement E) ←
          (core_models.ops.arith.Add.add
            (math.field.element.FieldElement E)
            (math.field.element.FieldElement E) b01 bw_m1);
        let bb0 : (math.field.element.FieldElement E) ←
          (core_models.ops.arith.Sub.sub
            (math.field.element.FieldElement E)
            (math.field.element.FieldElement E) b01 bw_m1);
        let bw_m2 : (math.field.element.FieldElement E) ←
          (core_models.ops.arith.Mul.mul
            (math.field.element.FieldElement F)
            (math.field.element.FieldElement E) w_mid_0 a67);
        let aa1 : (math.field.element.FieldElement E) ←
          (core_models.ops.arith.Add.add
            (math.field.element.FieldElement E)
            (math.field.element.FieldElement E) a45 bw_m2);
        let ab1 : (math.field.element.FieldElement E) ←
          (core_models.ops.arith.Sub.sub
            (math.field.element.FieldElement E)
            (math.field.element.FieldElement E) a45 bw_m2);
        let bw_m3 : (math.field.element.FieldElement E) ←
          (core_models.ops.arith.Mul.mul
            (math.field.element.FieldElement F)
            (math.field.element.FieldElement E) w_mid_1 b67);
        let ba1 : (math.field.element.FieldElement E) ←
          (core_models.ops.arith.Add.add
            (math.field.element.FieldElement E)
            (math.field.element.FieldElement E) b45 bw_m3);
        let bb1 : (math.field.element.FieldElement E) ←
          (core_models.ops.arith.Sub.sub
            (math.field.element.FieldElement E)
            (math.field.element.FieldElement E) b45 bw_m3);
        let w_lo_0 : (math.field.element.FieldElement F) ← twiddles_lo[j]_?;
        let w_lo_1 : (math.field.element.FieldElement F) ←
          twiddles_lo[(← (j +? eighth))]_?;
        let w_lo_2 : (math.field.element.FieldElement F) ←
          twiddles_lo[(← (j +? (← ((2 : usize) *? eighth))))]_?;
        let w_lo_3 : (math.field.element.FieldElement F) ←
          twiddles_lo[(← (j +? (← ((3 : usize) *? eighth))))]_?;
        let bw_l0 : (math.field.element.FieldElement E) ←
          (core_models.ops.arith.Mul.mul
            (math.field.element.FieldElement F)
            (math.field.element.FieldElement E) w_lo_0 aa1);
        let block : (RustSlice (math.field.element.FieldElement E)) ←
          (rust_primitives.hax.monomorphized_update_at.update_at_usize
            block
            i0
            (← (core_models.ops.arith.Add.add
              (math.field.element.FieldElement E)
              (math.field.element.FieldElement E) aa0 bw_l0)));
        let block : (RustSlice (math.field.element.FieldElement E)) ←
          (rust_primitives.hax.monomorphized_update_at.update_at_usize
            block
            i4
            (← (core_models.ops.arith.Sub.sub
              (math.field.element.FieldElement E)
              (math.field.element.FieldElement E) aa0 bw_l0)));
        let bw_l1 : (math.field.element.FieldElement E) ←
          (core_models.ops.arith.Mul.mul
            (math.field.element.FieldElement F)
            (math.field.element.FieldElement E) w_lo_1 ba1);
        let block : (RustSlice (math.field.element.FieldElement E)) ←
          (rust_primitives.hax.monomorphized_update_at.update_at_usize
            block
            i1
            (← (core_models.ops.arith.Add.add
              (math.field.element.FieldElement E)
              (math.field.element.FieldElement E) ba0 bw_l1)));
        let block : (RustSlice (math.field.element.FieldElement E)) ←
          (rust_primitives.hax.monomorphized_update_at.update_at_usize
            block
            i5
            (← (core_models.ops.arith.Sub.sub
              (math.field.element.FieldElement E)
              (math.field.element.FieldElement E) ba0 bw_l1)));
        let bw_l2 : (math.field.element.FieldElement E) ←
          (core_models.ops.arith.Mul.mul
            (math.field.element.FieldElement F)
            (math.field.element.FieldElement E) w_lo_2 ab1);
        let block : (RustSlice (math.field.element.FieldElement E)) ←
          (rust_primitives.hax.monomorphized_update_at.update_at_usize
            block
            i2
            (← (core_models.ops.arith.Add.add
              (math.field.element.FieldElement E)
              (math.field.element.FieldElement E) ab0 bw_l2)));
        let block : (RustSlice (math.field.element.FieldElement E)) ←
          (rust_primitives.hax.monomorphized_update_at.update_at_usize
            block
            i6
            (← (core_models.ops.arith.Sub.sub
              (math.field.element.FieldElement E)
              (math.field.element.FieldElement E) ab0 bw_l2)));
        let bw_l3 : (math.field.element.FieldElement E) ←
          (core_models.ops.arith.Mul.mul
            (math.field.element.FieldElement F)
            (math.field.element.FieldElement E) w_lo_3 bb1);
        let block : (RustSlice (math.field.element.FieldElement E)) ←
          (rust_primitives.hax.monomorphized_update_at.update_at_usize
            block
            i3
            (← (core_models.ops.arith.Add.add
              (math.field.element.FieldElement E)
              (math.field.element.FieldElement E) bb0 bw_l3)));
        let block : (RustSlice (math.field.element.FieldElement E)) ←
          (rust_primitives.hax.monomorphized_update_at.update_at_usize
            block
            i7
            (← (core_models.ops.arith.Sub.sub
              (math.field.element.FieldElement E)
              (math.field.element.FieldElement E) bb0 bw_l3)));
        (pure block) :
        RustM (RustSlice (math.field.element.FieldElement E)))));
  (pure block)

--  Optimized Bowers IFFT with 2-layer fusion and sequential twiddle access.
-- 
--  **Note**: This performs the inverse butterfly structure but does NOT apply
--  the 1/n scaling factor. The caller must:
--  1. Pass inverse twiddles from `LayerTwiddles::new_inverse(order)`
--  2. Scale results by n^(-1) after the transform
-- 
--  Using forward twiddles (from `LayerTwiddles::new()`) will produce incorrect results.
-- 
--  # Example
--  ```ignore
--  let order = 10u64;
--  let n = 1 << order;
-- 
--  // Create inverse twiddles for IFFT
--  let inv_twiddles = LayerTwiddles::<F>::new_inverse(order).unwrap();
-- 
--  // Apply inverse FFT (after bit-reversing FFT output)
--  in_place_bit_reverse_permute(&mut data);
--  bowers_ifft_opt(&mut data, &inv_twiddles)?;
-- 
--  // Scale by 1/n to complete the inverse transform
--  let n_inv = FieldElement::<F>::from(n as u64).inv().unwrap();
--  for val in data.iter_mut() {
--      *val = &*val * &n_inv;
--  }
--  ```
-- 
--  # Errors
--  Returns `FFTError::InputError` if:
--  - Input length is not a power of two
--  - Twiddle table size doesn't match input size
@[spec]
def bowers_ifft_opt
    (F : Type)
    (E : Type)
    [trait_constr_bowers_ifft_opt_associated_type_i0 :
      math.field.traits.IsFFTField.AssociatedTypes
      F]
    [trait_constr_bowers_ifft_opt_i0 : math.field.traits.IsFFTField F ]
    [trait_constr_bowers_ifft_opt_associated_type_i1 :
      math.field.traits.IsSubFieldOf.AssociatedTypes
      F
      E]
    [trait_constr_bowers_ifft_opt_i1 : math.field.traits.IsSubFieldOf F E ]
    [trait_constr_bowers_ifft_opt_associated_type_i2 :
      math.field.traits.IsField.AssociatedTypes
      E]
    [trait_constr_bowers_ifft_opt_i2 : math.field.traits.IsField E ]
    (input : (RustSlice (math.field.element.FieldElement E)))
    (layer_twiddles : (LayerTwiddles F)) :
    RustM
    (rust_primitives.hax.Tuple2
      (RustSlice (math.field.element.FieldElement E))
      (core_models.result.Result
        rust_primitives.hax.Tuple0
        math.fft.errors.FFTError))
    := do
  let n : usize ←
    (core_models.slice.Impl.len (math.field.element.FieldElement E) input);
  if (← (!? (← (core_models.num.Impl_11.is_power_of_two n)))) then do
    (pure (rust_primitives.hax.Tuple2.mk
      input
      (core_models.result.Result.Err (math.fft.errors.FFTError.InputError n))))
  else do
    if (← (n <=? (1 : usize))) then do
      (pure (rust_primitives.hax.Tuple2.mk
        input
        (core_models.result.Result.Ok rust_primitives.hax.Tuple0.mk)))
    else do
      let log_n : usize ←
        (rust_primitives.hax.cast_op
          (← (core_models.num.Impl_11.trailing_zeros n)) :
          RustM usize);
      if (← ((← (Impl.num_layers F layer_twiddles)) !=? log_n)) then do
        (pure (rust_primitives.hax.Tuple2.mk
          input
          (core_models.result.Result.Err
            (math.fft.errors.FFTError.InputError n))))
      else do
        if (← (n <=? (4 : usize))) then do
          let input : (RustSlice (math.field.element.FieldElement E)) ←
            (core_models.iter.traits.iterator.Iterator.fold
              (← (core_models.iter.traits.collect.IntoIterator.into_iter
                (core_models.iter.adapters.rev.Rev
                  (core_models.ops.range.Range usize))
                (← (core_models.iter.traits.iterator.Iterator.rev
                  (core_models.ops.range.Range usize)
                  (core_models.ops.range.Range.mk
                    (start := (0 : usize))
                    (_end := log_n))))))
              input
              (fun input layer =>
                (do
                let block_size : usize ← (n >>>? layer);
                let half_block : usize ← (block_size >>>? (1 : i32));
                let twiddles : (RustSlice (math.field.element.FieldElement F)) ←
                  (Impl.get_layer F layer_twiddles layer);
                (rust_primitives.hax.folds.fold_range_step_by
                  (0 : usize)
                  n
                  block_size
                  (fun input _ => (do (pure true) : RustM Bool))
                  input
                  (fun input block_start =>
                    (do
                    (rust_primitives.hax.folds.fold_range
                      (0 : usize)
                      half_block
                      (fun input _ => (do (pure true) : RustM Bool))
                      input
                      (fun input j =>
                        (do
                        let i0 : usize ← (block_start +? j);
                        let i1 : usize ← (i0 +? half_block);
                        let w : (math.field.element.FieldElement F) ←
                          twiddles[j]_?;
                        let bw : (math.field.element.FieldElement E) ←
                          (core_models.ops.arith.Mul.mul
                            (math.field.element.FieldElement F)
                            (math.field.element.FieldElement E)
                            w
                            (← input[i1]_?));
                        let sum : (math.field.element.FieldElement E) ←
                          (core_models.ops.arith.Add.add
                            (math.field.element.FieldElement E)
                            (math.field.element.FieldElement E)
                            (← input[i0]_?)
                            bw);
                        let diff : (math.field.element.FieldElement E) ←
                          (core_models.ops.arith.Sub.sub
                            (math.field.element.FieldElement E)
                            (math.field.element.FieldElement E)
                            (← input[i0]_?)
                            bw);
                        let
                          input : (RustSlice
                          (math.field.element.FieldElement E)) ←
                          (rust_primitives.hax.monomorphized_update_at.update_at_usize
                            input
                            i0
                            sum);
                        let
                          input : (RustSlice
                          (math.field.element.FieldElement E)) ←
                          (rust_primitives.hax.monomorphized_update_at.update_at_usize
                            input
                            i1
                            diff);
                        (pure input) :
                        RustM (RustSlice (math.field.element.FieldElement E)))))
                    :
                    RustM (RustSlice (math.field.element.FieldElement E))))) :
                RustM (RustSlice (math.field.element.FieldElement E)))));
          (pure (rust_primitives.hax.Tuple2.mk
            input
            (core_models.result.Result.Ok rust_primitives.hax.Tuple0.mk)))
        else do
          let layer : usize := log_n;
          let layer : usize ←
            (rust_primitives.hax.while_loop
              (fun layer => (do (pure true) : RustM Bool))
              (fun layer => (do (layer >=? (2 : usize)) : RustM Bool))
              (fun layer =>
                (do
                (rust_primitives.hax.int.from_machine (0 : u32)) :
                RustM hax_lib.int.Int))
              layer
              (fun layer =>
                (do
                let layer_hi : usize ← (layer -? (1 : usize));
                let layer_lo : usize ← (layer -? (2 : usize));
                let block_size : usize ← (n >>>? layer_lo);
                let _ ←
                  if true then do
                    let _ ← (hax_lib.assert (← (block_size >=? (4 : usize))));
                    (pure rust_primitives.hax.Tuple0.mk)
                  else do
                    (pure rust_primitives.hax.Tuple0.mk);
                let
                  twiddles_hi : (RustSlice
                  (math.field.element.FieldElement F)) ←
                  (Impl.get_layer F layer_twiddles layer_hi);
                let
                  twiddles_lo : (RustSlice
                  (math.field.element.FieldElement F)) ←
                  (Impl.get_layer F layer_twiddles layer_lo);
                let _ ←
                  (rust_primitives.hax.folds.fold_range_step_by
                    (0 : usize)
                    n
                    block_size
                    (fun _ _ => (do (pure true) : RustM Bool))
                    rust_primitives.hax.Tuple0.mk
                    (fun _ block_start =>
                      (do (pure sorry) : RustM rust_primitives.hax.Tuple0)));
                let layer : usize ← (layer -? (2 : usize));
                (pure layer) :
                RustM usize)));
          let input : (RustSlice (math.field.element.FieldElement E)) ←
            if (← (layer >=? (1 : usize))) then do
              let remaining_layer : usize ← (layer -? (1 : usize));
              let block_size : usize ← (n >>>? remaining_layer);
              let half_block : usize ← (block_size >>>? (1 : i32));
              let twiddles : (RustSlice (math.field.element.FieldElement F)) ←
                (Impl.get_layer F layer_twiddles remaining_layer);
              (rust_primitives.hax.folds.fold_range_step_by
                (0 : usize)
                n
                block_size
                (fun input _ => (do (pure true) : RustM Bool))
                input
                (fun input block_start =>
                  (do
                  (rust_primitives.hax.folds.fold_range
                    (0 : usize)
                    half_block
                    (fun input _ => (do (pure true) : RustM Bool))
                    input
                    (fun input j =>
                      (do
                      let i0 : usize ← (block_start +? j);
                      let i1 : usize ← (i0 +? half_block);
                      let w : (math.field.element.FieldElement F) ←
                        twiddles[j]_?;
                      let bw : (math.field.element.FieldElement E) ←
                        (core_models.ops.arith.Mul.mul
                          (math.field.element.FieldElement F)
                          (math.field.element.FieldElement E)
                          w
                          (← input[i1]_?));
                      let sum : (math.field.element.FieldElement E) ←
                        (core_models.ops.arith.Add.add
                          (math.field.element.FieldElement E)
                          (math.field.element.FieldElement E)
                          (← input[i0]_?)
                          bw);
                      let diff : (math.field.element.FieldElement E) ←
                        (core_models.ops.arith.Sub.sub
                          (math.field.element.FieldElement E)
                          (math.field.element.FieldElement E)
                          (← input[i0]_?)
                          bw);
                      let
                        input : (RustSlice
                        (math.field.element.FieldElement E)) ←
                        (rust_primitives.hax.monomorphized_update_at.update_at_usize
                          input
                          i0
                          sum);
                      let
                        input : (RustSlice
                        (math.field.element.FieldElement E)) ←
                        (rust_primitives.hax.monomorphized_update_at.update_at_usize
                          input
                          i1
                          diff);
                      (pure input) :
                      RustM (RustSlice (math.field.element.FieldElement E))))) :
                  RustM (RustSlice (math.field.element.FieldElement E)))))
            else do
              (pure input);
          let
            hax_temp_output : (core_models.result.Result
              rust_primitives.hax.Tuple0
              math.fft.errors.FFTError) :=
            (core_models.result.Result.Ok rust_primitives.hax.Tuple0.mk);
          (pure (rust_primitives.hax.Tuple2.mk input hax_temp_output))

--  Optimized Bowers FFT with 2-layer fusion and sequential twiddle access.
-- 
--  This is the recommended single-threaded FFT. It combines:
-- 
--  1. **Sequential twiddle access**: LayerTwiddles stores twiddles per layer
--     for cache-friendly sequential reads
--  2. **2-layer fusion**: Processes two FFT layers at once, keeping intermediate
--     values in registers to reduce memory traffic
-- 
--  For multi-threaded execution, use `bowers_fft_opt_fused_parallel` instead.
-- 
--  # Errors
--  Returns `FFTError::InputError` if:
--  - Input length is not a power of two
--  - Twiddle table size doesn't match input size
@[spec]
def bowers_fft_opt_fused
    (F : Type)
    (E : Type)
    [trait_constr_bowers_fft_opt_fused_associated_type_i0 :
      math.field.traits.IsFFTField.AssociatedTypes
      F]
    [trait_constr_bowers_fft_opt_fused_i0 : math.field.traits.IsFFTField F ]
    [trait_constr_bowers_fft_opt_fused_associated_type_i1 :
      math.field.traits.IsSubFieldOf.AssociatedTypes
      F
      E]
    [trait_constr_bowers_fft_opt_fused_i1 : math.field.traits.IsSubFieldOf F E ]
    [trait_constr_bowers_fft_opt_fused_associated_type_i2 :
      math.field.traits.IsField.AssociatedTypes
      E]
    [trait_constr_bowers_fft_opt_fused_i2 : math.field.traits.IsField E ]
    (input : (RustSlice (math.field.element.FieldElement E)))
    (layer_twiddles : (LayerTwiddles F)) :
    RustM
    (rust_primitives.hax.Tuple2
      (RustSlice (math.field.element.FieldElement E))
      (core_models.result.Result
        rust_primitives.hax.Tuple0
        math.fft.errors.FFTError))
    := do
  let n : usize ←
    (core_models.slice.Impl.len (math.field.element.FieldElement E) input);
  if (← (!? (← (core_models.num.Impl_11.is_power_of_two n)))) then do
    (pure (rust_primitives.hax.Tuple2.mk
      input
      (core_models.result.Result.Err (math.fft.errors.FFTError.InputError n))))
  else do
    if (← (n <=? (1 : usize))) then do
      (pure (rust_primitives.hax.Tuple2.mk
        input
        (core_models.result.Result.Ok rust_primitives.hax.Tuple0.mk)))
    else do
      let log_n : usize ←
        (rust_primitives.hax.cast_op
          (← (core_models.num.Impl_11.trailing_zeros n)) :
          RustM usize);
      if (← ((← (Impl.num_layers F layer_twiddles)) !=? log_n)) then do
        (pure (rust_primitives.hax.Tuple2.mk
          input
          (core_models.result.Result.Err
            (math.fft.errors.FFTError.InputError n))))
      else do
        if (← (n <=? (4 : usize))) then do
          let input : (RustSlice (math.field.element.FieldElement E)) ←
            (rust_primitives.hax.folds.fold_range
              (0 : usize)
              log_n
              (fun input _ => (do (pure true) : RustM Bool))
              input
              (fun input layer =>
                (do
                let block_size : usize ← (n >>>? layer);
                let half_block : usize ← (block_size >>>? (1 : i32));
                let twiddles : (RustSlice (math.field.element.FieldElement F)) ←
                  (Impl.get_layer F layer_twiddles layer);
                (rust_primitives.hax.folds.fold_range_step_by
                  (0 : usize)
                  n
                  block_size
                  (fun input _ => (do (pure true) : RustM Bool))
                  input
                  (fun input block_start =>
                    (do
                    (rust_primitives.hax.folds.fold_range
                      (0 : usize)
                      half_block
                      (fun input _ => (do (pure true) : RustM Bool))
                      input
                      (fun input j =>
                        (do
                        let i0 : usize ← (block_start +? j);
                        let i1 : usize ← (i0 +? half_block);
                        let w : (math.field.element.FieldElement F) ←
                          twiddles[j]_?;
                        let sum : (math.field.element.FieldElement E) ←
                          (core_models.ops.arith.Add.add
                            (math.field.element.FieldElement E)
                            (math.field.element.FieldElement E)
                            (← input[i0]_?)
                            (← input[i1]_?));
                        let diff : (math.field.element.FieldElement E) ←
                          (core_models.ops.arith.Sub.sub
                            (math.field.element.FieldElement E)
                            (math.field.element.FieldElement E)
                            (← input[i0]_?)
                            (← input[i1]_?));
                        let diff_w : (math.field.element.FieldElement E) ←
                          (core_models.ops.arith.Mul.mul
                            (math.field.element.FieldElement F)
                            (math.field.element.FieldElement E) w diff);
                        let
                          input : (RustSlice
                          (math.field.element.FieldElement E)) ←
                          (rust_primitives.hax.monomorphized_update_at.update_at_usize
                            input
                            i0
                            sum);
                        let
                          input : (RustSlice
                          (math.field.element.FieldElement E)) ←
                          (rust_primitives.hax.monomorphized_update_at.update_at_usize
                            input
                            i1
                            diff_w);
                        (pure input) :
                        RustM (RustSlice (math.field.element.FieldElement E)))))
                    :
                    RustM (RustSlice (math.field.element.FieldElement E))))) :
                RustM (RustSlice (math.field.element.FieldElement E)))));
          (pure (rust_primitives.hax.Tuple2.mk
            input
            (core_models.result.Result.Ok rust_primitives.hax.Tuple0.mk)))
        else do
          let layer : usize := (0 : usize);
          let layer : usize ←
            (rust_primitives.hax.while_loop_cf
              (fun layer => (do (pure true) : RustM Bool))
              (fun layer =>
                (do ((← (layer +? (1 : usize))) <? log_n) : RustM Bool))
              (fun layer =>
                (do
                (rust_primitives.hax.int.from_machine (0 : u32)) :
                RustM hax_lib.int.Int))
              layer
              (fun layer =>
                (do
                let block_size : usize ← (n >>>? layer);
                if (← (block_size >=? (4 : usize))) then do
                  let
                    twiddles_l0 : (RustSlice
                    (math.field.element.FieldElement F)) ←
                    (Impl.get_layer F layer_twiddles layer);
                  let
                    twiddles_l1 : (RustSlice
                    (math.field.element.FieldElement F)) ←
                    (Impl.get_layer F
                      layer_twiddles
                      (← (layer +? (1 : usize))));
                  let _ ←
                    (rust_primitives.hax.folds.fold_range_step_by
                      (0 : usize)
                      n
                      block_size
                      (fun _ _ => (do (pure true) : RustM Bool))
                      rust_primitives.hax.Tuple0.mk
                      (fun _ block_start =>
                        (do (pure sorry) : RustM rust_primitives.hax.Tuple0)));
                  let layer : usize ← (layer +? (2 : usize));
                  (pure (core_models.ops.control_flow.ControlFlow.Continue
                    layer))
                else do
                  (pure (core_models.ops.control_flow.ControlFlow.Break
                    (rust_primitives.hax.Tuple2.mk
                      rust_primitives.hax.Tuple0.mk
                      layer))) :
                RustM
                (core_models.ops.control_flow.ControlFlow
                  (rust_primitives.hax.Tuple2 rust_primitives.hax.Tuple0 usize)
                  usize))));
          let input : (RustSlice (math.field.element.FieldElement E)) ←
            if (← (layer <? log_n)) then do
              let block_size : usize ← (n >>>? layer);
              let half_block : usize ← (block_size >>>? (1 : i32));
              let twiddles : (RustSlice (math.field.element.FieldElement F)) ←
                (Impl.get_layer F layer_twiddles layer);
              (rust_primitives.hax.folds.fold_range_step_by
                (0 : usize)
                n
                block_size
                (fun input _ => (do (pure true) : RustM Bool))
                input
                (fun input block_start =>
                  (do
                  (rust_primitives.hax.folds.fold_range
                    (0 : usize)
                    half_block
                    (fun input _ => (do (pure true) : RustM Bool))
                    input
                    (fun input j =>
                      (do
                      let i0 : usize ← (block_start +? j);
                      let i1 : usize ← (i0 +? half_block);
                      let w : (math.field.element.FieldElement F) ←
                        twiddles[j]_?;
                      let sum : (math.field.element.FieldElement E) ←
                        (core_models.ops.arith.Add.add
                          (math.field.element.FieldElement E)
                          (math.field.element.FieldElement E)
                          (← input[i0]_?)
                          (← input[i1]_?));
                      let diff : (math.field.element.FieldElement E) ←
                        (core_models.ops.arith.Sub.sub
                          (math.field.element.FieldElement E)
                          (math.field.element.FieldElement E)
                          (← input[i0]_?)
                          (← input[i1]_?));
                      let diff_w : (math.field.element.FieldElement E) ←
                        (core_models.ops.arith.Mul.mul
                          (math.field.element.FieldElement F)
                          (math.field.element.FieldElement E) w diff);
                      let
                        input : (RustSlice
                        (math.field.element.FieldElement E)) ←
                        (rust_primitives.hax.monomorphized_update_at.update_at_usize
                          input
                          i0
                          sum);
                      let
                        input : (RustSlice
                        (math.field.element.FieldElement E)) ←
                        (rust_primitives.hax.monomorphized_update_at.update_at_usize
                          input
                          i1
                          diff_w);
                      (pure input) :
                      RustM (RustSlice (math.field.element.FieldElement E))))) :
                  RustM (RustSlice (math.field.element.FieldElement E)))))
            else do
              (pure input);
          let
            hax_temp_output : (core_models.result.Result
              rust_primitives.hax.Tuple0
              math.fft.errors.FFTError) :=
            (core_models.result.Result.Ok rust_primitives.hax.Tuple0.mk);
          (pure (rust_primitives.hax.Tuple2.mk input hax_temp_output))

end math.fft.bowers_fft


namespace math.fft.roots_of_unity

--  Returns a `Vec` of the powers of a `2^n`th primitive root of unity, scaled `offset` times,
--  in a Natural configuration.
@[spec]
def get_powers_of_primitive_root_coset
    (F : Type)
    [trait_constr_get_powers_of_primitive_root_coset_associated_type_i0 :
      math.field.traits.IsFFTField.AssociatedTypes
      F]
    [trait_constr_get_powers_of_primitive_root_coset_i0 :
      math.field.traits.IsFFTField
      F
      ]
    (n : u64)
    (count : usize)
    (offset : (math.field.element.FieldElement F)) :
    RustM
    (core_models.result.Result
      (alloc.vec.Vec (math.field.element.FieldElement F) alloc.alloc.Global)
      math.fft.errors.FFTError)
    := do
  if (← (count ==? (0 : usize))) then do
    (pure (core_models.result.Result.Ok
      (← (alloc.vec.Impl.new (math.field.element.FieldElement F)
        rust_primitives.hax.Tuple0.mk))))
  else do
    match
      (← (math.field.traits.IsFFTField.get_primitive_root_of_unity F n))
    with
      | (core_models.result.Result.Ok  root) => do
        let
          results : (alloc.vec.Vec
            (math.field.element.FieldElement F)
            alloc.alloc.Global) ←
          (alloc.vec.Impl.with_capacity (math.field.element.FieldElement F)
            count);
        let current : (math.field.element.FieldElement F) ←
          (core_models.clone.Clone.clone
            (math.field.element.FieldElement F) offset);
        let ⟨current, results⟩ ←
          (rust_primitives.hax.folds.fold_range
            (0 : usize)
            count
            (fun ⟨current, results⟩ _ => (do (pure true) : RustM Bool))
            (rust_primitives.hax.Tuple2.mk current results)
            (fun ⟨current, results⟩ _ =>
              (do
              let
                results : (alloc.vec.Vec
                  (math.field.element.FieldElement F)
                  alloc.alloc.Global) ←
                (alloc.vec.Impl_1.push
                  (math.field.element.FieldElement F)
                  alloc.alloc.Global
                  results
                  (← (core_models.clone.Clone.clone
                    (math.field.element.FieldElement F) current)));
              let current : (math.field.element.FieldElement F) ←
                (core_models.ops.arith.Mul.mul
                  (math.field.element.FieldElement F)
                  (math.field.element.FieldElement F) current root);
              (pure (rust_primitives.hax.Tuple2.mk current results)) :
              RustM
              (rust_primitives.hax.Tuple2
                (math.field.element.FieldElement F)
                (alloc.vec.Vec
                  (math.field.element.FieldElement F)
                  alloc.alloc.Global)))));
        (pure (core_models.result.Result.Ok results))
      | (core_models.result.Result.Err  err) => do
        (pure (core_models.result.Result.Err
          (← (core_models.convert.From._from
            math.fft.errors.FFTError
            math.field.errors.FieldError err))))

end math.fft.roots_of_unity


namespace math.polynomial

--  Dispatch forward FFT (DIF) to parallel or sequential implementation based on buffer size.
@[spec]
def dispatch_fft
    (F : Type)
    (E : Type)
    [trait_constr_dispatch_fft_associated_type_i0 :
      math.field.traits.IsFFTField.AssociatedTypes
      F]
    [trait_constr_dispatch_fft_i0 : math.field.traits.IsFFTField F ]
    [trait_constr_dispatch_fft_associated_type_i1 :
      math.field.traits.IsSubFieldOf.AssociatedTypes
      F
      E]
    [trait_constr_dispatch_fft_i1 : math.field.traits.IsSubFieldOf F E ]
    [trait_constr_dispatch_fft_associated_type_i2 :
      math.field.traits.IsField.AssociatedTypes
      E]
    [trait_constr_dispatch_fft_i2 : math.field.traits.IsField E ]
    [trait_constr_dispatch_fft_associated_type_i3 :
      core_models.marker.Send.AssociatedTypes
      E]
    [trait_constr_dispatch_fft_i3 : core_models.marker.Send E ]
    [trait_constr_dispatch_fft_associated_type_i4 :
      core_models.marker.Sync.AssociatedTypes
      E]
    [trait_constr_dispatch_fft_i4 : core_models.marker.Sync E ]
    (buffer : (RustSlice (math.field.element.FieldElement E)))
    (twiddles : (math.fft.bowers_fft.LayerTwiddles F)) :
    RustM
    (rust_primitives.hax.Tuple2
      (RustSlice (math.field.element.FieldElement E))
      (core_models.result.Result
        rust_primitives.hax.Tuple0
        math.fft.errors.FFTError))
    := do
  let ⟨tmp0, out⟩ ←
    (math.fft.bowers_fft.bowers_fft_opt_fused F E buffer twiddles);
  let buffer : (RustSlice (math.field.element.FieldElement E)) := tmp0;
  let
    hax_temp_output : (core_models.result.Result
      rust_primitives.hax.Tuple0
      math.fft.errors.FFTError) :=
    out;
  (pure (rust_primitives.hax.Tuple2.mk buffer hax_temp_output))

--  Dispatch inverse FFT (DIT) to parallel or sequential implementation based on buffer size.
@[spec]
def dispatch_ifft
    (F : Type)
    (E : Type)
    [trait_constr_dispatch_ifft_associated_type_i0 :
      math.field.traits.IsFFTField.AssociatedTypes
      F]
    [trait_constr_dispatch_ifft_i0 : math.field.traits.IsFFTField F ]
    [trait_constr_dispatch_ifft_associated_type_i1 :
      math.field.traits.IsSubFieldOf.AssociatedTypes
      F
      E]
    [trait_constr_dispatch_ifft_i1 : math.field.traits.IsSubFieldOf F E ]
    [trait_constr_dispatch_ifft_associated_type_i2 :
      math.field.traits.IsField.AssociatedTypes
      E]
    [trait_constr_dispatch_ifft_i2 : math.field.traits.IsField E ]
    [trait_constr_dispatch_ifft_associated_type_i3 :
      core_models.marker.Send.AssociatedTypes
      E]
    [trait_constr_dispatch_ifft_i3 : core_models.marker.Send E ]
    [trait_constr_dispatch_ifft_associated_type_i4 :
      core_models.marker.Sync.AssociatedTypes
      E]
    [trait_constr_dispatch_ifft_i4 : core_models.marker.Sync E ]
    (buffer : (RustSlice (math.field.element.FieldElement E)))
    (twiddles : (math.fft.bowers_fft.LayerTwiddles F)) :
    RustM
    (rust_primitives.hax.Tuple2
      (RustSlice (math.field.element.FieldElement E))
      (core_models.result.Result
        rust_primitives.hax.Tuple0
        math.fft.errors.FFTError))
    := do
  let ⟨tmp0, out⟩ ← (math.fft.bowers_fft.bowers_ifft_opt F E buffer twiddles);
  let buffer : (RustSlice (math.field.element.FieldElement E)) := tmp0;
  let
    hax_temp_output : (core_models.result.Result
      rust_primitives.hax.Tuple0
      math.fft.errors.FFTError) :=
    out;
  (pure (rust_primitives.hax.Tuple2.mk buffer hax_temp_output))

--  Returns a new polynomial that interpolates `(w^i, fft_evals[i])`, with `w` being a
--  Nth primitive root of unity in a subfield F of E, and `i in 0..N`, with `N = fft_evals.len()`.
--  This is considered to be the inverse operation of [Self::evaluate_fft()].
@[spec]
def Impl_1.interpolate_fft
    (E : Type)
    (F : Type)
    [trait_constr_interpolate_fft_associated_type_i0 :
      math.field.traits.IsField.AssociatedTypes
      E]
    [trait_constr_interpolate_fft_i0 : math.field.traits.IsField E ]
    [trait_constr_interpolate_fft_associated_type_i1 :
      math.field.traits.IsFFTField.AssociatedTypes
      F]
    [trait_constr_interpolate_fft_i1 : math.field.traits.IsFFTField F ]
    [trait_constr_interpolate_fft_associated_type_i2 :
      math.field.traits.IsSubFieldOf.AssociatedTypes
      F
      E]
    [trait_constr_interpolate_fft_i2 : math.field.traits.IsSubFieldOf F E ]
    [trait_constr_interpolate_fft_associated_type_i3 :
      core_models.marker.Send.AssociatedTypes
      E]
    [trait_constr_interpolate_fft_i3 : core_models.marker.Send E ]
    [trait_constr_interpolate_fft_associated_type_i4 :
      core_models.marker.Sync.AssociatedTypes
      E]
    [trait_constr_interpolate_fft_i4 : core_models.marker.Sync E ]
    (fft_evals : (RustSlice (math.field.element.FieldElement E))) :
    RustM
    (core_models.result.Result
      (Polynomial (math.field.element.FieldElement E))
      math.fft.errors.FFTError)
    := do
  let n : usize ←
    (core_models.slice.Impl.len (math.field.element.FieldElement E) fft_evals);
  if (← (!? (← (core_models.num.Impl_11.is_power_of_two n)))) then do
    (pure (core_models.result.Result.Err
      (math.fft.errors.FFTError.InputError n)))
  else do
    let order : u64 ←
      (rust_primitives.hax.cast_op
        (← (core_models.num.Impl_11.trailing_zeros n)) :
        RustM u64);
    match
      (← (core_models.option.Impl.ok_or
        (math.fft.bowers_fft.LayerTwiddles F)
        math.fft.errors.FFTError
        (← (math.fft.bowers_fft.Impl.new_inverse F order))
        (math.fft.errors.FFTError.DomainSizeError
          (← (rust_primitives.hax.cast_op order : RustM usize)))))
    with
      | (core_models.result.Result.Ok  inv_twiddles) => do
        let
          coeffs : (alloc.vec.Vec
            (math.field.element.FieldElement E)
            alloc.alloc.Global) ←
          (alloc.slice.Impl.to_vec (math.field.element.FieldElement E)
            fft_evals);
        let
          coeffs : (alloc.vec.Vec
            (math.field.element.FieldElement E)
            alloc.alloc.Global) ←
          (alloc.slice.Impl.to_vec
            (← (math.fft.bit_reversing.in_place_bit_reverse_permute
              (math.field.element.FieldElement E)
              (← (alloc.vec.Impl_1.as_slice coeffs)))));
        let ⟨tmp0, out⟩ ←
          (dispatch_ifft F E
            (← (alloc.vec.Impl_1.as_slice coeffs))
            inv_twiddles);
        let
          coeffs : (alloc.vec.Vec
            (math.field.element.FieldElement E)
            alloc.alloc.Global) ←
          (alloc.slice.Impl.to_vec tmp0);
        match out with
          | (core_models.result.Result.Ok  _) => do
            let scale_factor : (math.field.element.FieldElement E) ←
              (core_models.result.Impl.unwrap
                (math.field.element.FieldElement E)
                math.field.errors.FieldError
                (← (math.field.element.Impl_32.inv E
                  (← (core_models.convert.From._from
                    (math.field.element.FieldElement E)
                    u64 (← (rust_primitives.hax.cast_op n : RustM u64)))))));
            (pure (core_models.result.Result.Ok
              (← (Impl.scale_coeffs E
                (← (Impl.new E
                  (← (core_models.ops.deref.Deref.deref
                    (alloc.vec.Vec
                      (math.field.element.FieldElement E)
                      alloc.alloc.Global) coeffs))))
                scale_factor))))
          | (core_models.result.Result.Err  err) => do
            (pure (core_models.result.Result.Err err))
      | (core_models.result.Result.Err  err) => do
        (pure (core_models.result.Result.Err err))

--  Returns a new polynomial that interpolates offset `(w^i, fft_evals[i])`, with `w` being a
--  Nth primitive root of unity in a subfield F of E, and `i in 0..N`, with `N = fft_evals.len()`.
--  This is considered to be the inverse operation of [Self::evaluate_offset_fft()].
@[spec]
def Impl_1.interpolate_offset_fft
    (E : Type)
    (F : Type)
    [trait_constr_interpolate_offset_fft_associated_type_i0 :
      math.field.traits.IsField.AssociatedTypes
      E]
    [trait_constr_interpolate_offset_fft_i0 : math.field.traits.IsField E ]
    [trait_constr_interpolate_offset_fft_associated_type_i1 :
      math.field.traits.IsFFTField.AssociatedTypes
      F]
    [trait_constr_interpolate_offset_fft_i1 : math.field.traits.IsFFTField F ]
    [trait_constr_interpolate_offset_fft_associated_type_i2 :
      math.field.traits.IsSubFieldOf.AssociatedTypes
      F
      E]
    [trait_constr_interpolate_offset_fft_i2 : math.field.traits.IsSubFieldOf
      F
      E
      ]
    [trait_constr_interpolate_offset_fft_associated_type_i3 :
      core_models.marker.Send.AssociatedTypes
      E]
    [trait_constr_interpolate_offset_fft_i3 : core_models.marker.Send E ]
    [trait_constr_interpolate_offset_fft_associated_type_i4 :
      core_models.marker.Sync.AssociatedTypes
      E]
    [trait_constr_interpolate_offset_fft_i4 : core_models.marker.Sync E ]
    (fft_evals : (RustSlice (math.field.element.FieldElement E)))
    (offset : (math.field.element.FieldElement F)) :
    RustM
    (core_models.result.Result
      (Polynomial (math.field.element.FieldElement E))
      math.fft.errors.FFTError)
    := do
  match (← (Impl_1.interpolate_fft E F fft_evals)) with
    | (core_models.result.Result.Ok  scaled) => do
      match
        (← (core_models.result.Impl.map_err
          (math.field.element.FieldElement F)
          math.field.errors.FieldError
          math.fft.errors.FFTError
          (math.field.errors.FieldError -> RustM math.fft.errors.FFTError)
          (← (math.field.element.Impl_32.inv F offset))
          (fun _ =>
            (do
            (pure math.fft.errors.FFTError.InvalidCosetOffset) :
            RustM math.fft.errors.FFTError))))
      with
        | (core_models.result.Result.Ok  offset_inv) => do
          (pure (core_models.result.Result.Ok
            (← (Impl.scale E F scaled offset_inv))))
        | (core_models.result.Result.Err  err) => do
          (pure (core_models.result.Result.Err err))
    | (core_models.result.Result.Err  err) => do
      (pure (core_models.result.Result.Err err))

--  Compute the coset LDE into a caller-provided buffer, avoiding allocation when
--  `buffer.capacity() >= n * blowup_factor`.
-- 
--  Same as [`coset_lde_full`], but writes into `buffer` instead of allocating a new Vec.
--  The buffer is cleared and reused: `buffer.clear(); buffer.extend_from_slice(evals);
--  buffer.resize(lde_size, zero)`. When the capacity is sufficient, no heap allocation occurs.
--  Weights are in the base field F — the scaling `w * coeff` uses mixed F×E multiplication.
@[spec]
def Impl_1.coset_lde_full_into
    (E : Type)
    (F : Type)
    [trait_constr_coset_lde_full_into_associated_type_i0 :
      math.field.traits.IsField.AssociatedTypes
      E]
    [trait_constr_coset_lde_full_into_i0 : math.field.traits.IsField E ]
    [trait_constr_coset_lde_full_into_associated_type_i1 :
      math.field.traits.IsFFTField.AssociatedTypes
      F]
    [trait_constr_coset_lde_full_into_i1 : math.field.traits.IsFFTField F ]
    [trait_constr_coset_lde_full_into_associated_type_i2 :
      math.field.traits.IsSubFieldOf.AssociatedTypes
      F
      E]
    [trait_constr_coset_lde_full_into_i2 : math.field.traits.IsSubFieldOf F E ]
    [trait_constr_coset_lde_full_into_associated_type_i3 :
      core_models.marker.Send.AssociatedTypes
      F]
    [trait_constr_coset_lde_full_into_i3 : core_models.marker.Send F ]
    [trait_constr_coset_lde_full_into_associated_type_i4 :
      core_models.marker.Sync.AssociatedTypes
      F]
    [trait_constr_coset_lde_full_into_i4 : core_models.marker.Sync F ]
    [trait_constr_coset_lde_full_into_associated_type_i5 :
      core_models.marker.Send.AssociatedTypes
      E]
    [trait_constr_coset_lde_full_into_i5 : core_models.marker.Send E ]
    [trait_constr_coset_lde_full_into_associated_type_i6 :
      core_models.marker.Sync.AssociatedTypes
      E]
    [trait_constr_coset_lde_full_into_i6 : core_models.marker.Sync E ]
    (evals : (RustSlice (math.field.element.FieldElement E)))
    (blowup_factor : usize)
    (weights : (RustSlice (math.field.element.FieldElement F)))
    (inv_twiddles : (math.fft.bowers_fft.LayerTwiddles F))
    (fwd_twiddles : (math.fft.bowers_fft.LayerTwiddles F))
    (buffer :
    (alloc.vec.Vec (math.field.element.FieldElement E) alloc.alloc.Global)) :
    RustM
    (rust_primitives.hax.Tuple2
      (alloc.vec.Vec (math.field.element.FieldElement E) alloc.alloc.Global)
      (core_models.result.Result
        rust_primitives.hax.Tuple0
        math.fft.errors.FFTError))
    := do
  let n : usize ←
    (core_models.slice.Impl.len (math.field.element.FieldElement E) evals);
  if (← (n ==? (0 : usize))) then do
    let
      buffer : (alloc.vec.Vec
        (math.field.element.FieldElement E)
        alloc.alloc.Global) ←
      (alloc.vec.Impl_1.clear
        (math.field.element.FieldElement E)
        alloc.alloc.Global buffer);
    (pure (rust_primitives.hax.Tuple2.mk
      buffer
      (core_models.result.Result.Ok rust_primitives.hax.Tuple0.mk)))
  else do
    if (← (!? (← (core_models.num.Impl_11.is_power_of_two n)))) then do
      (pure (rust_primitives.hax.Tuple2.mk
        buffer
        (core_models.result.Result.Err
          (math.fft.errors.FFTError.InputError n))))
    else do
      let lde_size : usize ← (n *? blowup_factor);
      if
      (← ((← (rust_primitives.hax.cast_op
          (← (core_models.num.Impl_11.trailing_zeros lde_size)) :
          RustM u64))
        >? (math.field.traits.IsFFTField.TWO_ADICITY F))) then do
        (pure (rust_primitives.hax.Tuple2.mk
          buffer
          (core_models.result.Result.Err
            (math.fft.errors.FFTError.DomainSizeError
              (← (rust_primitives.hax.cast_op
                (← (core_models.num.Impl_11.trailing_zeros lde_size)) :
                RustM usize))))))
      else do
        let
          buffer : (alloc.vec.Vec
            (math.field.element.FieldElement E)
            alloc.alloc.Global) ←
          (alloc.vec.Impl_1.clear
            (math.field.element.FieldElement E)
            alloc.alloc.Global buffer);
        let
          buffer : (alloc.vec.Vec
            (math.field.element.FieldElement E)
            alloc.alloc.Global) ←
          (alloc.vec.Impl_2.extend_from_slice
            (math.field.element.FieldElement E)
            alloc.alloc.Global buffer evals);
        let
          buffer : (alloc.vec.Vec
            (math.field.element.FieldElement E)
            alloc.alloc.Global) ←
          (alloc.vec.Impl_2.resize
            (math.field.element.FieldElement E)
            alloc.alloc.Global
            buffer
            lde_size
            (← (math.field.element.Impl_32.zero E
              rust_primitives.hax.Tuple0.mk)));
        let
          buffer : (alloc.vec.Vec
            (math.field.element.FieldElement E)
            alloc.alloc.Global) ←
          (alloc.slice.Impl.to_vec
            (← (rust_primitives.hax.monomorphized_update_at.update_at_range_to
              (← (alloc.vec.Impl_1.as_slice buffer))
              (core_models.ops.range.RangeTo.mk (_end := n))
              (← (math.fft.bit_reversing.in_place_bit_reverse_permute
                (math.field.element.FieldElement E)
                (← buffer[
                  (core_models.ops.range.RangeTo.mk (_end := n))
                  ]_?))))));
        let ⟨tmp0, out⟩ ←
          (dispatch_ifft F E
            (← buffer[(core_models.ops.range.RangeTo.mk (_end := n))]_?)
            inv_twiddles);
        let
          buffer : (alloc.vec.Vec
            (math.field.element.FieldElement E)
            alloc.alloc.Global) ←
          (alloc.slice.Impl.to_vec
            (← (rust_primitives.hax.monomorphized_update_at.update_at_range_to
              (← (alloc.vec.Impl_1.as_slice buffer))
              (core_models.ops.range.RangeTo.mk (_end := n))
              tmp0)));
        match out with
          | (core_models.result.Result.Ok  _) => do
            let
              buffer : (alloc.vec.Vec
                (math.field.element.FieldElement E)
                alloc.alloc.Global) ←
              (rust_primitives.hax.folds.fold_range
                (0 : usize)
                n
                (fun buffer _ => (do (pure true) : RustM Bool))
                buffer
                (fun buffer i =>
                  (do
                  (alloc.slice.Impl.to_vec
                    (←
                    (rust_primitives.hax.monomorphized_update_at.update_at_usize
                      (← (alloc.vec.Impl_1.as_slice buffer))
                      i
                      (← (core_models.ops.arith.Mul.mul
                        (math.field.element.FieldElement F)
                        (math.field.element.FieldElement E)
                        (← weights[i]_?)
                        (← buffer[i]_?)))))) :
                  RustM
                  (alloc.vec.Vec
                    (math.field.element.FieldElement E)
                    alloc.alloc.Global))));
            let ⟨tmp0, out⟩ ←
              (dispatch_fft F E
                (← (alloc.vec.Impl_1.as_slice buffer))
                fwd_twiddles);
            let
              buffer : (alloc.vec.Vec
                (math.field.element.FieldElement E)
                alloc.alloc.Global) ←
              (alloc.slice.Impl.to_vec tmp0);
            match out with
              | (core_models.result.Result.Ok  _) => do
                let
                  buffer : (alloc.vec.Vec
                    (math.field.element.FieldElement E)
                    alloc.alloc.Global) ←
                  (alloc.slice.Impl.to_vec
                    (← (math.fft.bit_reversing.in_place_bit_reverse_permute
                      (math.field.element.FieldElement E)
                      (← (alloc.vec.Impl_1.as_slice buffer)))));
                let
                  hax_temp_output : (core_models.result.Result
                    rust_primitives.hax.Tuple0
                    math.fft.errors.FFTError) :=
                  (core_models.result.Result.Ok rust_primitives.hax.Tuple0.mk);
                (pure (rust_primitives.hax.Tuple2.mk buffer hax_temp_output))
              | (core_models.result.Result.Err  err) => do
                (pure (rust_primitives.hax.Tuple2.mk
                  buffer
                  (core_models.result.Result.Err err)))
          | (core_models.result.Result.Err  err) => do
            (pure (rust_primitives.hax.Tuple2.mk
              buffer
              (core_models.result.Result.Err err)))

--  Compute the coset LDE with pre-computed twiddle factors and pre-computed weights.
-- 
--  Same as [`coset_lde_with_twiddles`], but also accepts pre-computed `weights[i] = offset^i / n`
--  so that the scaling step avoids the running product across columns.
--  Weights are in the base field F — the scaling `w * coeff` uses mixed F×E multiplication.
@[spec]
def Impl_1.coset_lde_full
    (E : Type)
    (F : Type)
    [trait_constr_coset_lde_full_associated_type_i0 :
      math.field.traits.IsField.AssociatedTypes
      E]
    [trait_constr_coset_lde_full_i0 : math.field.traits.IsField E ]
    [trait_constr_coset_lde_full_associated_type_i1 :
      math.field.traits.IsFFTField.AssociatedTypes
      F]
    [trait_constr_coset_lde_full_i1 : math.field.traits.IsFFTField F ]
    [trait_constr_coset_lde_full_associated_type_i2 :
      math.field.traits.IsSubFieldOf.AssociatedTypes
      F
      E]
    [trait_constr_coset_lde_full_i2 : math.field.traits.IsSubFieldOf F E ]
    [trait_constr_coset_lde_full_associated_type_i3 :
      core_models.marker.Send.AssociatedTypes
      F]
    [trait_constr_coset_lde_full_i3 : core_models.marker.Send F ]
    [trait_constr_coset_lde_full_associated_type_i4 :
      core_models.marker.Sync.AssociatedTypes
      F]
    [trait_constr_coset_lde_full_i4 : core_models.marker.Sync F ]
    [trait_constr_coset_lde_full_associated_type_i5 :
      core_models.marker.Send.AssociatedTypes
      E]
    [trait_constr_coset_lde_full_i5 : core_models.marker.Send E ]
    [trait_constr_coset_lde_full_associated_type_i6 :
      core_models.marker.Sync.AssociatedTypes
      E]
    [trait_constr_coset_lde_full_i6 : core_models.marker.Sync E ]
    (evals : (RustSlice (math.field.element.FieldElement E)))
    (blowup_factor : usize)
    (weights : (RustSlice (math.field.element.FieldElement F)))
    (inv_twiddles : (math.fft.bowers_fft.LayerTwiddles F))
    (fwd_twiddles : (math.fft.bowers_fft.LayerTwiddles F)) :
    RustM
    (core_models.result.Result
      (alloc.vec.Vec (math.field.element.FieldElement E) alloc.alloc.Global)
      math.fft.errors.FFTError)
    := do
  let n : usize ←
    (core_models.slice.Impl.len (math.field.element.FieldElement E) evals);
  if (← (n ==? (0 : usize))) then do
    (pure (core_models.result.Result.Ok
      (← (alloc.vec.Impl.new (math.field.element.FieldElement E)
        rust_primitives.hax.Tuple0.mk))))
  else do
    let lde_size : usize ← (n *? blowup_factor);
    let
      buffer : (alloc.vec.Vec
        (math.field.element.FieldElement E)
        alloc.alloc.Global) ←
      (alloc.vec.Impl.with_capacity (math.field.element.FieldElement E)
        lde_size);
    let ⟨tmp0, out⟩ ←
      (Impl_1.coset_lde_full_into E F
        evals
        blowup_factor
        weights
        inv_twiddles
        fwd_twiddles
        buffer);
    let
      buffer : (alloc.vec.Vec
        (math.field.element.FieldElement E)
        alloc.alloc.Global) :=
      tmp0;
    match out with
      | (core_models.result.Result.Ok  _) => do
        (pure (core_models.result.Result.Ok buffer))
      | (core_models.result.Result.Err  err) => do
        (pure (core_models.result.Result.Err err))

--  In-place coset LDE: the buffer already contains N evaluation points at `[0..N]`.
-- 
--  This expands the buffer from N elements to `N * blowup_factor` by performing:
--  1. iFFT on buffer[..N]
--  2. Scale by pre-computed weights
--  3. Zero-pad to N * blowup_factor
--  4. Forward FFT on the full buffer
-- 
--  Unlike `coset_lde_full_into`, this skips the `clear + extend_from_slice` step
--  since data is already in the buffer. Used for transpose elimination: columns are
--  extracted directly into owned buffers, then expanded in-place.
@[spec]
def Impl_1.coset_lde_full_expand
    (E : Type)
    (F : Type)
    [trait_constr_coset_lde_full_expand_associated_type_i0 :
      math.field.traits.IsField.AssociatedTypes
      E]
    [trait_constr_coset_lde_full_expand_i0 : math.field.traits.IsField E ]
    [trait_constr_coset_lde_full_expand_associated_type_i1 :
      math.field.traits.IsFFTField.AssociatedTypes
      F]
    [trait_constr_coset_lde_full_expand_i1 : math.field.traits.IsFFTField F ]
    [trait_constr_coset_lde_full_expand_associated_type_i2 :
      math.field.traits.IsSubFieldOf.AssociatedTypes
      F
      E]
    [trait_constr_coset_lde_full_expand_i2 : math.field.traits.IsSubFieldOf
      F
      E
      ]
    [trait_constr_coset_lde_full_expand_associated_type_i3 :
      core_models.marker.Send.AssociatedTypes
      F]
    [trait_constr_coset_lde_full_expand_i3 : core_models.marker.Send F ]
    [trait_constr_coset_lde_full_expand_associated_type_i4 :
      core_models.marker.Sync.AssociatedTypes
      F]
    [trait_constr_coset_lde_full_expand_i4 : core_models.marker.Sync F ]
    [trait_constr_coset_lde_full_expand_associated_type_i5 :
      core_models.marker.Send.AssociatedTypes
      E]
    [trait_constr_coset_lde_full_expand_i5 : core_models.marker.Send E ]
    [trait_constr_coset_lde_full_expand_associated_type_i6 :
      core_models.marker.Sync.AssociatedTypes
      E]
    [trait_constr_coset_lde_full_expand_i6 : core_models.marker.Sync E ]
    (buffer :
    (alloc.vec.Vec (math.field.element.FieldElement E) alloc.alloc.Global))
    (blowup_factor : usize)
    (weights : (RustSlice (math.field.element.FieldElement F)))
    (inv_twiddles : (math.fft.bowers_fft.LayerTwiddles F))
    (fwd_twiddles : (math.fft.bowers_fft.LayerTwiddles F)) :
    RustM
    (rust_primitives.hax.Tuple2
      (alloc.vec.Vec (math.field.element.FieldElement E) alloc.alloc.Global)
      (core_models.result.Result
        rust_primitives.hax.Tuple0
        math.fft.errors.FFTError))
    := do
  let n : usize ←
    (alloc.vec.Impl_1.len (math.field.element.FieldElement E) alloc.alloc.Global
      buffer);
  if (← (n ==? (0 : usize))) then do
    (pure (rust_primitives.hax.Tuple2.mk
      buffer
      (core_models.result.Result.Ok rust_primitives.hax.Tuple0.mk)))
  else do
    if (← (!? (← (core_models.num.Impl_11.is_power_of_two n)))) then do
      (pure (rust_primitives.hax.Tuple2.mk
        buffer
        (core_models.result.Result.Err
          (math.fft.errors.FFTError.InputError n))))
    else do
      let lde_size : usize ← (n *? blowup_factor);
      if
      (← ((← (rust_primitives.hax.cast_op
          (← (core_models.num.Impl_11.trailing_zeros lde_size)) :
          RustM u64))
        >? (math.field.traits.IsFFTField.TWO_ADICITY F))) then do
        (pure (rust_primitives.hax.Tuple2.mk
          buffer
          (core_models.result.Result.Err
            (math.fft.errors.FFTError.DomainSizeError
              (← (rust_primitives.hax.cast_op
                (← (core_models.num.Impl_11.trailing_zeros lde_size)) :
                RustM usize))))))
      else do
        let
          buffer : (alloc.vec.Vec
            (math.field.element.FieldElement E)
            alloc.alloc.Global) ←
          (alloc.slice.Impl.to_vec
            (← (rust_primitives.hax.monomorphized_update_at.update_at_range_to
              (← (alloc.vec.Impl_1.as_slice buffer))
              (core_models.ops.range.RangeTo.mk (_end := n))
              (← (math.fft.bit_reversing.in_place_bit_reverse_permute
                (math.field.element.FieldElement E)
                (← buffer[
                  (core_models.ops.range.RangeTo.mk (_end := n))
                  ]_?))))));
        let ⟨tmp0, out⟩ ←
          (dispatch_ifft F E
            (← buffer[(core_models.ops.range.RangeTo.mk (_end := n))]_?)
            inv_twiddles);
        let
          buffer : (alloc.vec.Vec
            (math.field.element.FieldElement E)
            alloc.alloc.Global) ←
          (alloc.slice.Impl.to_vec
            (← (rust_primitives.hax.monomorphized_update_at.update_at_range_to
              (← (alloc.vec.Impl_1.as_slice buffer))
              (core_models.ops.range.RangeTo.mk (_end := n))
              tmp0)));
        match out with
          | (core_models.result.Result.Ok  _) => do
            let
              buffer : (alloc.vec.Vec
                (math.field.element.FieldElement E)
                alloc.alloc.Global) ←
              (rust_primitives.hax.folds.fold_range
                (0 : usize)
                n
                (fun buffer _ => (do (pure true) : RustM Bool))
                buffer
                (fun buffer i =>
                  (do
                  (alloc.slice.Impl.to_vec
                    (←
                    (rust_primitives.hax.monomorphized_update_at.update_at_usize
                      (← (alloc.vec.Impl_1.as_slice buffer))
                      i
                      (← (core_models.ops.arith.Mul.mul
                        (math.field.element.FieldElement F)
                        (math.field.element.FieldElement E)
                        (← weights[i]_?)
                        (← buffer[i]_?)))))) :
                  RustM
                  (alloc.vec.Vec
                    (math.field.element.FieldElement E)
                    alloc.alloc.Global))));
            let
              buffer : (alloc.vec.Vec
                (math.field.element.FieldElement E)
                alloc.alloc.Global) ←
              (alloc.vec.Impl_2.resize
                (math.field.element.FieldElement E)
                alloc.alloc.Global
                buffer
                lde_size
                (← (math.field.element.Impl_32.zero E
                  rust_primitives.hax.Tuple0.mk)));
            let ⟨tmp0, out⟩ ←
              (dispatch_fft F E
                (← (alloc.vec.Impl_1.as_slice buffer))
                fwd_twiddles);
            let
              buffer : (alloc.vec.Vec
                (math.field.element.FieldElement E)
                alloc.alloc.Global) ←
              (alloc.slice.Impl.to_vec tmp0);
            match out with
              | (core_models.result.Result.Ok  _) => do
                let
                  buffer : (alloc.vec.Vec
                    (math.field.element.FieldElement E)
                    alloc.alloc.Global) ←
                  (alloc.slice.Impl.to_vec
                    (← (math.fft.bit_reversing.in_place_bit_reverse_permute
                      (math.field.element.FieldElement E)
                      (← (alloc.vec.Impl_1.as_slice buffer)))));
                let
                  hax_temp_output : (core_models.result.Result
                    rust_primitives.hax.Tuple0
                    math.fft.errors.FFTError) :=
                  (core_models.result.Result.Ok rust_primitives.hax.Tuple0.mk);
                (pure (rust_primitives.hax.Tuple2.mk buffer hax_temp_output))
              | (core_models.result.Result.Err  err) => do
                (pure (rust_primitives.hax.Tuple2.mk
                  buffer
                  (core_models.result.Result.Err err)))
          | (core_models.result.Result.Err  err) => do
            (pure (rust_primitives.hax.Tuple2.mk
              buffer
              (core_models.result.Result.Err err)))

@[spec]
def evaluate_fft_cpu_raw
    (F : Type)
    (E : Type)
    [trait_constr_evaluate_fft_cpu_raw_associated_type_i0 :
      math.field.traits.IsFFTField.AssociatedTypes
      F]
    [trait_constr_evaluate_fft_cpu_raw_i0 : math.field.traits.IsFFTField F ]
    [trait_constr_evaluate_fft_cpu_raw_associated_type_i1 :
      math.field.traits.IsSubFieldOf.AssociatedTypes
      F
      E]
    [trait_constr_evaluate_fft_cpu_raw_i1 : math.field.traits.IsSubFieldOf F E ]
    [trait_constr_evaluate_fft_cpu_raw_associated_type_i2 :
      math.field.traits.IsField.AssociatedTypes
      E]
    [trait_constr_evaluate_fft_cpu_raw_i2 : math.field.traits.IsField E ]
    [trait_constr_evaluate_fft_cpu_raw_associated_type_i3 :
      core_models.marker.Send.AssociatedTypes
      E]
    [trait_constr_evaluate_fft_cpu_raw_i3 : core_models.marker.Send E ]
    [trait_constr_evaluate_fft_cpu_raw_associated_type_i4 :
      core_models.marker.Sync.AssociatedTypes
      E]
    [trait_constr_evaluate_fft_cpu_raw_i4 : core_models.marker.Sync E ]
    (coeffs : (RustSlice (math.field.element.FieldElement E)))
    (permute_to_natural : Bool) :
    RustM
    (core_models.result.Result
      (alloc.vec.Vec (math.field.element.FieldElement E) alloc.alloc.Global)
      math.fft.errors.FFTError)
    := do
  let n : usize ←
    (core_models.slice.Impl.len (math.field.element.FieldElement E) coeffs);
  if (← (!? (← (core_models.num.Impl_11.is_power_of_two n)))) then do
    (pure (core_models.result.Result.Err
      (math.fft.errors.FFTError.InputError n)))
  else do
    let order : u64 ←
      (rust_primitives.hax.cast_op
        (← (core_models.num.Impl_11.trailing_zeros n)) :
        RustM u64);
    match
      (← (core_models.option.Impl.ok_or
        (math.fft.bowers_fft.LayerTwiddles F)
        math.fft.errors.FFTError
        (← (math.fft.bowers_fft.Impl.new F order))
        (math.fft.errors.FFTError.DomainSizeError
          (← (rust_primitives.hax.cast_op order : RustM usize)))))
    with
      | (core_models.result.Result.Ok  layer_twiddles) => do
        let
          result : (alloc.vec.Vec
            (math.field.element.FieldElement E)
            alloc.alloc.Global) ←
          (alloc.slice.Impl.to_vec (math.field.element.FieldElement E) coeffs);
        let ⟨tmp0, out⟩ ←
          (dispatch_fft F E
            (← (alloc.vec.Impl_1.as_slice result))
            layer_twiddles);
        let
          result : (alloc.vec.Vec
            (math.field.element.FieldElement E)
            alloc.alloc.Global) ←
          (alloc.slice.Impl.to_vec tmp0);
        match out with
          | (core_models.result.Result.Ok  _) => do
            let result : rust_primitives.hax.Tuple0 ←
              if permute_to_natural then do
                let
                  result : (alloc.vec.Vec
                    (math.field.element.FieldElement E)
                    alloc.alloc.Global) ←
                  (alloc.slice.Impl.to_vec
                    (← (math.fft.bit_reversing.in_place_bit_reverse_permute
                      (math.field.element.FieldElement E)
                      (← (alloc.vec.Impl_1.as_slice result)))));
                (pure result)
              else do
                (pure result);
            (pure (core_models.result.Result.Ok result))
          | (core_models.result.Result.Err  err) => do
            (pure (core_models.result.Result.Err err))
      | (core_models.result.Result.Err  err) => do
        (pure (core_models.result.Result.Err err))

--  Returns `N` evaluations of this polynomial using FFT over a domain in a subfield F of E (so the results
--  are P(w^i), with w being a primitive root of unity).
--  `N = max(self.coeff_len(), domain_size).next_power_of_two() * blowup_factor`.
--  If `domain_size` is `None`, it defaults to 0.
@[spec]
def Impl_1.evaluate_fft
    (E : Type)
    (F : Type)
    [trait_constr_evaluate_fft_associated_type_i0 :
      math.field.traits.IsField.AssociatedTypes
      E]
    [trait_constr_evaluate_fft_i0 : math.field.traits.IsField E ]
    [trait_constr_evaluate_fft_associated_type_i1 :
      math.field.traits.IsFFTField.AssociatedTypes
      F]
    [trait_constr_evaluate_fft_i1 : math.field.traits.IsFFTField F ]
    [trait_constr_evaluate_fft_associated_type_i2 :
      math.field.traits.IsSubFieldOf.AssociatedTypes
      F
      E]
    [trait_constr_evaluate_fft_i2 : math.field.traits.IsSubFieldOf F E ]
    [trait_constr_evaluate_fft_associated_type_i3 :
      core_models.marker.Send.AssociatedTypes
      E]
    [trait_constr_evaluate_fft_i3 : core_models.marker.Send E ]
    [trait_constr_evaluate_fft_associated_type_i4 :
      core_models.marker.Sync.AssociatedTypes
      E]
    [trait_constr_evaluate_fft_i4 : core_models.marker.Sync E ]
    (poly : (Polynomial (math.field.element.FieldElement E)))
    (blowup_factor : usize)
    (domain_size : (core_models.option.Option usize)) :
    RustM
    (core_models.result.Result
      (alloc.vec.Vec (math.field.element.FieldElement E) alloc.alloc.Global)
      math.fft.errors.FFTError)
    := do
  let domain_size : usize ←
    (core_models.option.Impl.unwrap_or usize domain_size (0 : usize));
  let len : usize ←
    ((← (core_models.num.Impl_11.next_power_of_two
        (← (core_models.cmp.max usize
          (← (Impl.coeff_len E poly))
          domain_size))))
      *? blowup_factor);
  if
  (← ((← (rust_primitives.hax.cast_op
      (← (core_models.num.Impl_11.trailing_zeros len)) :
      RustM u64))
    >? (math.field.traits.IsFFTField.TWO_ADICITY F))) then do
    (pure (core_models.result.Result.Err
      (math.fft.errors.FFTError.DomainSizeError
        (← (rust_primitives.hax.cast_op
          (← (core_models.num.Impl_11.trailing_zeros len)) :
          RustM usize)))))
  else do
    if
    (← (core_models.slice.Impl.is_empty (math.field.element.FieldElement E)
      (← (Impl.coefficients E poly)))) then do
      (pure (core_models.result.Result.Ok
        (← (alloc.vec.from_elem (math.field.element.FieldElement E)
          (← (math.field.element.Impl_32.zero E rust_primitives.hax.Tuple0.mk))
          len))))
    else do
      let
        coeffs : (alloc.vec.Vec
          (math.field.element.FieldElement E)
          alloc.alloc.Global) ←
        (alloc.slice.Impl.to_vec (math.field.element.FieldElement E)
          (← (Impl.coefficients E poly)));
      let
        coeffs : (alloc.vec.Vec
          (math.field.element.FieldElement E)
          alloc.alloc.Global) ←
        (alloc.vec.Impl_2.resize
          (math.field.element.FieldElement E)
          alloc.alloc.Global
          coeffs
          len
          (← (math.field.element.Impl_32.zero E
            rust_primitives.hax.Tuple0.mk)));
      (evaluate_fft_cpu_raw F E
        (← (core_models.ops.deref.Deref.deref
          (alloc.vec.Vec (math.field.element.FieldElement E) alloc.alloc.Global)
          coeffs))
        true)

--  Returns `N` evaluations with an offset of this polynomial using FFT over a domain in a subfield F of E
--  (so the results are P(w^i), with w being a primitive root of unity).
--  `N = max(self.coeff_len(), domain_size).next_power_of_two() * blowup_factor`.
--  If `domain_size` is `None`, it defaults to 0.
@[spec]
def Impl_1.evaluate_offset_fft
    (E : Type)
    (F : Type)
    [trait_constr_evaluate_offset_fft_associated_type_i0 :
      math.field.traits.IsField.AssociatedTypes
      E]
    [trait_constr_evaluate_offset_fft_i0 : math.field.traits.IsField E ]
    [trait_constr_evaluate_offset_fft_associated_type_i1 :
      math.field.traits.IsFFTField.AssociatedTypes
      F]
    [trait_constr_evaluate_offset_fft_i1 : math.field.traits.IsFFTField F ]
    [trait_constr_evaluate_offset_fft_associated_type_i2 :
      math.field.traits.IsSubFieldOf.AssociatedTypes
      F
      E]
    [trait_constr_evaluate_offset_fft_i2 : math.field.traits.IsSubFieldOf F E ]
    [trait_constr_evaluate_offset_fft_associated_type_i3 :
      core_models.marker.Send.AssociatedTypes
      E]
    [trait_constr_evaluate_offset_fft_i3 : core_models.marker.Send E ]
    [trait_constr_evaluate_offset_fft_associated_type_i4 :
      core_models.marker.Sync.AssociatedTypes
      E]
    [trait_constr_evaluate_offset_fft_i4 : core_models.marker.Sync E ]
    (poly : (Polynomial (math.field.element.FieldElement E)))
    (blowup_factor : usize)
    (domain_size : (core_models.option.Option usize))
    (offset : (math.field.element.FieldElement F)) :
    RustM
    (core_models.result.Result
      (alloc.vec.Vec (math.field.element.FieldElement E) alloc.alloc.Global)
      math.fft.errors.FFTError)
    := do
  let scaled : (Polynomial (math.field.element.FieldElement E)) ←
    (Impl.scale E F poly offset);
  (Impl_1.evaluate_fft E F scaled blowup_factor domain_size)

end math.polynomial


namespace math.field.extensions_goldilocks

--  Returns the component-wise addition of `a` and `b`
@[spec]
def Impl_2.add_hoisted
    (a :
    (RustArray
    (math.field.element.FieldElement math.field.goldilocks.GoldilocksField)
    2))
    (b :
    (RustArray
    (math.field.element.FieldElement math.field.goldilocks.GoldilocksField)
    2)) :
    RustM
    (RustArray
    (math.field.element.FieldElement math.field.goldilocks.GoldilocksField)
    2)
    := do
  (pure (RustArray.ofVec #v[(← (core_models.ops.arith.Add.add
                                (math.field.element.FieldElement
                                  math.field.goldilocks.GoldilocksField)
                                (math.field.element.FieldElement
                                  math.field.goldilocks.GoldilocksField)
                                (← a[(0 : usize)]_?)
                                (← b[(0 : usize)]_?))),
                              (← (core_models.ops.arith.Add.add
                                (math.field.element.FieldElement
                                  math.field.goldilocks.GoldilocksField)
                                (math.field.element.FieldElement
                                  math.field.goldilocks.GoldilocksField)
                                (← a[(1 : usize)]_?)
                                (← b[(1 : usize)]_?)))]))

--  Returns the component-wise subtraction of `a` and `b`
@[spec]
def Impl_2.sub_hoisted
    (a :
    (RustArray
    (math.field.element.FieldElement math.field.goldilocks.GoldilocksField)
    2))
    (b :
    (RustArray
    (math.field.element.FieldElement math.field.goldilocks.GoldilocksField)
    2)) :
    RustM
    (RustArray
    (math.field.element.FieldElement math.field.goldilocks.GoldilocksField)
    2)
    := do
  (pure (RustArray.ofVec #v[(← (core_models.ops.arith.Sub.sub
                                (math.field.element.FieldElement
                                  math.field.goldilocks.GoldilocksField)
                                (math.field.element.FieldElement
                                  math.field.goldilocks.GoldilocksField)
                                (← a[(0 : usize)]_?)
                                (← b[(0 : usize)]_?))),
                              (← (core_models.ops.arith.Sub.sub
                                (math.field.element.FieldElement
                                  math.field.goldilocks.GoldilocksField)
                                (math.field.element.FieldElement
                                  math.field.goldilocks.GoldilocksField)
                                (← a[(1 : usize)]_?)
                                (← b[(1 : usize)]_?)))]))

--  Returns the component-wise negation of `a`
@[spec]
def Impl_2.neg_hoisted
    (a :
    (RustArray
    (math.field.element.FieldElement math.field.goldilocks.GoldilocksField)
    2)) :
    RustM
    (RustArray
    (math.field.element.FieldElement math.field.goldilocks.GoldilocksField)
    2)
    := do
  (pure (RustArray.ofVec #v[(← (core_models.ops.arith.Neg.neg
                                (math.field.element.FieldElement
                                  math.field.goldilocks.GoldilocksField)
                                (← a[(0 : usize)]_?))),
                              (← (core_models.ops.arith.Neg.neg
                                (math.field.element.FieldElement
                                  math.field.goldilocks.GoldilocksField)
                                (← a[(1 : usize)]_?)))]))

@[spec]
def Impl_2.eq_hoisted
    (a :
    (RustArray
    (math.field.element.FieldElement math.field.goldilocks.GoldilocksField)
    2))
    (b :
    (RustArray
    (math.field.element.FieldElement math.field.goldilocks.GoldilocksField)
    2)) :
    RustM Bool := do
  ((← (core_models.cmp.PartialEq.eq
      (math.field.element.FieldElement math.field.goldilocks.GoldilocksField)
      (math.field.element.FieldElement math.field.goldilocks.GoldilocksField)
      (← a[(0 : usize)]_?)
      (← b[(0 : usize)]_?)))
    &&? (← (core_models.cmp.PartialEq.eq
      (math.field.element.FieldElement math.field.goldilocks.GoldilocksField)
      (math.field.element.FieldElement math.field.goldilocks.GoldilocksField)
      (← a[(1 : usize)]_?)
      (← b[(1 : usize)]_?))))

@[spec]
def Impl_2.zero_hoisted (_ : rust_primitives.hax.Tuple0) :
    RustM
    (RustArray
    (math.field.element.FieldElement math.field.goldilocks.GoldilocksField)
    2)
    := do
  (pure (RustArray.ofVec #v[(← (math.field.element.Impl_32.zero
                                math.field.goldilocks.GoldilocksField
                                rust_primitives.hax.Tuple0.mk)),
                              (← (math.field.element.Impl_32.zero
                                math.field.goldilocks.GoldilocksField
                                rust_primitives.hax.Tuple0.mk))]))

@[spec]
def Impl_2.one_hoisted (_ : rust_primitives.hax.Tuple0) :
    RustM
    (RustArray
    (math.field.element.FieldElement math.field.goldilocks.GoldilocksField)
    2)
    := do
  (pure (RustArray.ofVec #v[(← (math.field.element.Impl_32.one
                                math.field.goldilocks.GoldilocksField
                                rust_primitives.hax.Tuple0.mk)),
                              (← (math.field.element.Impl_32.zero
                                math.field.goldilocks.GoldilocksField
                                rust_primitives.hax.Tuple0.mk))]))

@[spec]
def Impl_2.from_u64_hoisted (x : u64) :
    RustM
    (RustArray
    (math.field.element.FieldElement math.field.goldilocks.GoldilocksField)
    2)
    := do
  (pure (RustArray.ofVec #v[(← (core_models.convert.From._from
                                (math.field.element.FieldElement
                                  math.field.goldilocks.GoldilocksField)
                                u64 x)),
                              (← (math.field.element.Impl_32.zero
                                math.field.goldilocks.GoldilocksField
                                rust_primitives.hax.Tuple0.mk))]))

@[spec]
def Impl_2.double_hoisted
    (a :
    (RustArray
    (math.field.element.FieldElement math.field.goldilocks.GoldilocksField)
    2)) :
    RustM
    (RustArray
    (math.field.element.FieldElement math.field.goldilocks.GoldilocksField)
    2)
    := do
  (pure (RustArray.ofVec #v[(← (math.field.element.Impl_32.double
                                math.field.goldilocks.GoldilocksField
                                (← a[(0 : usize)]_?))),
                              (← (math.field.element.Impl_32.double
                                math.field.goldilocks.GoldilocksField
                                (← a[(1 : usize)]_?)))]))

@[spec]
def Impl_3.mul_hoisted
    (a : u64)
    (b :
    (RustArray
    (math.field.element.FieldElement math.field.goldilocks.GoldilocksField)
    2)) :
    RustM
    (RustArray
    (math.field.element.FieldElement math.field.goldilocks.GoldilocksField)
    2)
    := do
  let
    c0 : (math.field.element.FieldElement
      math.field.goldilocks.GoldilocksField) ←
    (math.field.element.Impl_6.from_raw math.field.goldilocks.GoldilocksField
      (← (math.field.traits.IsField.mul
        math.field.goldilocks.GoldilocksField
        a
        (← (math.field.element.Impl_32.value
          math.field.goldilocks.GoldilocksField (← b[(0 : usize)]_?))))));
  let
    c1 : (math.field.element.FieldElement
      math.field.goldilocks.GoldilocksField) ←
    (math.field.element.Impl_6.from_raw math.field.goldilocks.GoldilocksField
      (← (math.field.traits.IsField.mul
        math.field.goldilocks.GoldilocksField
        a
        (← (math.field.element.Impl_32.value
          math.field.goldilocks.GoldilocksField (← b[(1 : usize)]_?))))));
  (pure (RustArray.ofVec #v[c0, c1]))

@[spec]
def Impl_3.add_hoisted
    (a : u64)
    (b :
    (RustArray
    (math.field.element.FieldElement math.field.goldilocks.GoldilocksField)
    2)) :
    RustM
    (RustArray
    (math.field.element.FieldElement math.field.goldilocks.GoldilocksField)
    2)
    := do
  let
    c0 : (math.field.element.FieldElement
      math.field.goldilocks.GoldilocksField) ←
    (math.field.element.Impl_6.from_raw math.field.goldilocks.GoldilocksField
      (← (math.field.traits.IsField.add
        math.field.goldilocks.GoldilocksField
        a
        (← (math.field.element.Impl_32.value
          math.field.goldilocks.GoldilocksField (← b[(0 : usize)]_?))))));
  (pure (RustArray.ofVec #v[c0, (← b[(1 : usize)]_?)]))

@[spec]
def Impl_3.sub_hoisted
    (a : u64)
    (b :
    (RustArray
    (math.field.element.FieldElement math.field.goldilocks.GoldilocksField)
    2)) :
    RustM
    (RustArray
    (math.field.element.FieldElement math.field.goldilocks.GoldilocksField)
    2)
    := do
  let
    c0 : (math.field.element.FieldElement
      math.field.goldilocks.GoldilocksField) ←
    (math.field.element.Impl_6.from_raw math.field.goldilocks.GoldilocksField
      (← (math.field.traits.IsField.sub
        math.field.goldilocks.GoldilocksField
        a
        (← (math.field.element.Impl_32.value
          math.field.goldilocks.GoldilocksField (← b[(0 : usize)]_?))))));
  let
    c1 : (math.field.element.FieldElement
      math.field.goldilocks.GoldilocksField) ←
    (math.field.element.Impl_6.from_raw math.field.goldilocks.GoldilocksField
      (← (math.field.traits.IsField.neg
        math.field.goldilocks.GoldilocksField
        (← (math.field.element.Impl_32.value
          math.field.goldilocks.GoldilocksField (← b[(1 : usize)]_?))))));
  (pure (RustArray.ofVec #v[c0, c1]))

@[spec]
def Impl_3.embed_hoisted (a : u64) :
    RustM
    (RustArray
    (math.field.element.FieldElement math.field.goldilocks.GoldilocksField)
    2)
    := do
  (pure (RustArray.ofVec #v[(← (math.field.element.Impl_6.from_raw
                                math.field.goldilocks.GoldilocksField a)),
                              (← (math.field.element.Impl_32.zero
                                math.field.goldilocks.GoldilocksField
                                rust_primitives.hax.Tuple0.mk))]))

@[spec]
def Impl_3.to_subfield_vec_hoisted
    (b :
    (RustArray
    (math.field.element.FieldElement math.field.goldilocks.GoldilocksField)
    2)) :
    RustM (alloc.vec.Vec u64 alloc.alloc.Global) := do
  let out : (alloc.vec.Vec u64 alloc.alloc.Global) ←
    (alloc.vec.Impl.with_capacity u64
      (← (core_models.slice.Impl.len
        (math.field.element.FieldElement math.field.goldilocks.GoldilocksField)
        (← (rust_primitives.unsize b)))));
  let out : (alloc.vec.Vec u64 alloc.alloc.Global) ←
    (rust_primitives.hax.folds.fold_range
      (0 : usize)
      (← (core_models.slice.Impl.len
        (math.field.element.FieldElement math.field.goldilocks.GoldilocksField)
        (← (rust_primitives.unsize b))))
      (fun out _ => (do (pure true) : RustM Bool))
      out
      (fun out i =>
        (do
        (alloc.vec.Impl_1.push u64 alloc.alloc.Global
          out
          (← (math.field.element.Impl_32.to_raw
            math.field.goldilocks.GoldilocksField (← b[i]_?)))) :
        RustM (alloc.vec.Vec u64 alloc.alloc.Global))));
  (pure out)

--  Returns the component-wise addition of `a` and `b`
@[spec]
def Impl_5.add_hoisted
    (a :
    (RustArray
    (math.field.element.FieldElement math.field.goldilocks.GoldilocksField)
    3))
    (b :
    (RustArray
    (math.field.element.FieldElement math.field.goldilocks.GoldilocksField)
    3)) :
    RustM
    (RustArray
    (math.field.element.FieldElement math.field.goldilocks.GoldilocksField)
    3)
    := do
  (pure (RustArray.ofVec #v[(← (core_models.ops.arith.Add.add
                                (math.field.element.FieldElement
                                  math.field.goldilocks.GoldilocksField)
                                (math.field.element.FieldElement
                                  math.field.goldilocks.GoldilocksField)
                                (← a[(0 : usize)]_?)
                                (← b[(0 : usize)]_?))),
                              (← (core_models.ops.arith.Add.add
                                (math.field.element.FieldElement
                                  math.field.goldilocks.GoldilocksField)
                                (math.field.element.FieldElement
                                  math.field.goldilocks.GoldilocksField)
                                (← a[(1 : usize)]_?)
                                (← b[(1 : usize)]_?))),
                              (← (core_models.ops.arith.Add.add
                                (math.field.element.FieldElement
                                  math.field.goldilocks.GoldilocksField)
                                (math.field.element.FieldElement
                                  math.field.goldilocks.GoldilocksField)
                                (← a[(2 : usize)]_?)
                                (← b[(2 : usize)]_?)))]))

--  Multiplication using schoolbook with fused dot products.
--  (a0 + a1*w + a2*w^2) * (b0 + b1*w + b2*w^2) mod (w^3 - 2)
-- 
--  Expanding and applying w^3 = 2:
--    c0 = a0*b0 + 2*(a1*b2 + a2*b1)
--    c1 = a0*b1 + a1*b0 + 2*a2*b2
--    c2 = a0*b2 + a1*b1 + a2*b0
-- 
--  Each component is computed as a single dot_product_3 (9 raw muls,
--  3 reduce128 calls) instead of Karatsuba (6 muls, 6 reduce128 + many
--  add/sub). The reduction savings outweigh the extra multiplications.
@[spec]
def Impl_5.mul_hoisted
    (a :
    (RustArray
    (math.field.element.FieldElement math.field.goldilocks.GoldilocksField)
    3))
    (b :
    (RustArray
    (math.field.element.FieldElement math.field.goldilocks.GoldilocksField)
    3)) :
    RustM
    (RustArray
    (math.field.element.FieldElement math.field.goldilocks.GoldilocksField)
    3)
    := do
  let ⟨a0, a1, a2⟩ :=
    (rust_primitives.hax.Tuple3.mk
      (← (math.field.element.Impl_32.value math.field.goldilocks.GoldilocksField
        (← a[(0 : usize)]_?)))
      (← (math.field.element.Impl_32.value math.field.goldilocks.GoldilocksField
        (← a[(1 : usize)]_?)))
      (← (math.field.element.Impl_32.value math.field.goldilocks.GoldilocksField
        (← a[(2 : usize)]_?))));
  let ⟨b0, b1, b2⟩ :=
    (rust_primitives.hax.Tuple3.mk
      (← (math.field.element.Impl_32.value math.field.goldilocks.GoldilocksField
        (← b[(0 : usize)]_?)))
      (← (math.field.element.Impl_32.value math.field.goldilocks.GoldilocksField
        (← b[(1 : usize)]_?)))
      (← (math.field.element.Impl_32.value math.field.goldilocks.GoldilocksField
        (← b[(2 : usize)]_?))));
  let b1_2 : u64 ←
    (math.field.traits.IsField.double math.field.goldilocks.GoldilocksField b1);
  let b2_2 : u64 ←
    (math.field.traits.IsField.double math.field.goldilocks.GoldilocksField b2);
  let c0 : u64 ← (math.field.goldilocks.dot_product_3 a0 b0 a1 b2_2 a2 b1_2);
  let c1 : u64 ← (math.field.goldilocks.dot_product_3 a0 b1 a1 b0 a2 b2_2);
  let c2 : u64 ← (math.field.goldilocks.dot_product_3 a0 b2 a1 b1 a2 b0);
  (pure (RustArray.ofVec #v[(← (math.field.element.Impl_6.from_raw
                                math.field.goldilocks.GoldilocksField c0)),
                              (← (math.field.element.Impl_6.from_raw
                                math.field.goldilocks.GoldilocksField c1)),
                              (← (math.field.element.Impl_6.from_raw
                                math.field.goldilocks.GoldilocksField c2))]))

--  Squaring using fused dot products.
--  (a0 + a1*w + a2*w^2)^2 mod (w^3 - 2):
--    c0 = a0^2 + 4*a1*a2
--    c1 = 2*a0*a1 + 2*a2^2
--    c2 = 2*a0*a2 + a1^2
@[spec]
def Impl_5.square_hoisted
    (a :
    (RustArray
    (math.field.element.FieldElement math.field.goldilocks.GoldilocksField)
    3)) :
    RustM
    (RustArray
    (math.field.element.FieldElement math.field.goldilocks.GoldilocksField)
    3)
    := do
  let ⟨a0, a1, a2⟩ :=
    (rust_primitives.hax.Tuple3.mk
      (← (math.field.element.Impl_32.value math.field.goldilocks.GoldilocksField
        (← a[(0 : usize)]_?)))
      (← (math.field.element.Impl_32.value math.field.goldilocks.GoldilocksField
        (← a[(1 : usize)]_?)))
      (← (math.field.element.Impl_32.value math.field.goldilocks.GoldilocksField
        (← a[(2 : usize)]_?))));
  let a0_2 : u64 ←
    (math.field.traits.IsField.double math.field.goldilocks.GoldilocksField a0);
  let a2_4 : u64 ←
    (math.field.traits.IsField.double
      math.field.goldilocks.GoldilocksField
      (← (math.field.traits.IsField.double
        math.field.goldilocks.GoldilocksField a2)));
  let c0 : u64 ← (math.field.goldilocks.dot_product_2 a0 a0 a1 a2_4);
  let a2_2 : u64 ←
    (math.field.traits.IsField.double math.field.goldilocks.GoldilocksField a2);
  let c1 : u64 ← (math.field.goldilocks.dot_product_2 a0_2 a1 a2_2 a2);
  let c2 : u64 ← (math.field.goldilocks.dot_product_2 a1 a1 a0_2 a2);
  (pure (RustArray.ofVec #v[(← (math.field.element.Impl_6.from_raw
                                math.field.goldilocks.GoldilocksField c0)),
                              (← (math.field.element.Impl_6.from_raw
                                math.field.goldilocks.GoldilocksField c1)),
                              (← (math.field.element.Impl_6.from_raw
                                math.field.goldilocks.GoldilocksField c2))]))

--  Returns the component-wise subtraction of `a` and `b`
@[spec]
def Impl_5.sub_hoisted
    (a :
    (RustArray
    (math.field.element.FieldElement math.field.goldilocks.GoldilocksField)
    3))
    (b :
    (RustArray
    (math.field.element.FieldElement math.field.goldilocks.GoldilocksField)
    3)) :
    RustM
    (RustArray
    (math.field.element.FieldElement math.field.goldilocks.GoldilocksField)
    3)
    := do
  (pure (RustArray.ofVec #v[(← (core_models.ops.arith.Sub.sub
                                (math.field.element.FieldElement
                                  math.field.goldilocks.GoldilocksField)
                                (math.field.element.FieldElement
                                  math.field.goldilocks.GoldilocksField)
                                (← a[(0 : usize)]_?)
                                (← b[(0 : usize)]_?))),
                              (← (core_models.ops.arith.Sub.sub
                                (math.field.element.FieldElement
                                  math.field.goldilocks.GoldilocksField)
                                (math.field.element.FieldElement
                                  math.field.goldilocks.GoldilocksField)
                                (← a[(1 : usize)]_?)
                                (← b[(1 : usize)]_?))),
                              (← (core_models.ops.arith.Sub.sub
                                (math.field.element.FieldElement
                                  math.field.goldilocks.GoldilocksField)
                                (math.field.element.FieldElement
                                  math.field.goldilocks.GoldilocksField)
                                (← a[(2 : usize)]_?)
                                (← b[(2 : usize)]_?)))]))

--  Returns the component-wise negation of `a`
@[spec]
def Impl_5.neg_hoisted
    (a :
    (RustArray
    (math.field.element.FieldElement math.field.goldilocks.GoldilocksField)
    3)) :
    RustM
    (RustArray
    (math.field.element.FieldElement math.field.goldilocks.GoldilocksField)
    3)
    := do
  (pure (RustArray.ofVec #v[(← (core_models.ops.arith.Neg.neg
                                (math.field.element.FieldElement
                                  math.field.goldilocks.GoldilocksField)
                                (← a[(0 : usize)]_?))),
                              (← (core_models.ops.arith.Neg.neg
                                (math.field.element.FieldElement
                                  math.field.goldilocks.GoldilocksField)
                                (← a[(1 : usize)]_?))),
                              (← (core_models.ops.arith.Neg.neg
                                (math.field.element.FieldElement
                                  math.field.goldilocks.GoldilocksField)
                                (← a[(2 : usize)]_?)))]))

--  Returns the multiplicative inverse of `a`
@[spec]
def Impl_5.inv_hoisted
    (a :
    (RustArray
    (math.field.element.FieldElement math.field.goldilocks.GoldilocksField)
    3)) :
    RustM
    (core_models.result.Result
      (RustArray
      (math.field.element.FieldElement math.field.goldilocks.GoldilocksField)
      3)
      math.field.errors.FieldError)
    := do
  let
    a0_sq : (math.field.element.FieldElement
      math.field.goldilocks.GoldilocksField) ←
    (math.field.element.Impl_32.square math.field.goldilocks.GoldilocksField
      (← a[(0 : usize)]_?));
  let
    a1_sq : (math.field.element.FieldElement
      math.field.goldilocks.GoldilocksField) ←
    (math.field.element.Impl_32.square math.field.goldilocks.GoldilocksField
      (← a[(1 : usize)]_?));
  let
    a2_sq : (math.field.element.FieldElement
      math.field.goldilocks.GoldilocksField) ←
    (math.field.element.Impl_32.square math.field.goldilocks.GoldilocksField
      (← a[(2 : usize)]_?));
  let
    a0_cubed : (math.field.element.FieldElement
      math.field.goldilocks.GoldilocksField) ←
    (core_models.ops.arith.Mul.mul
      (math.field.element.FieldElement math.field.goldilocks.GoldilocksField)
      (math.field.element.FieldElement math.field.goldilocks.GoldilocksField)
      a0_sq
      (← a[(0 : usize)]_?));
  let
    a1_cubed : (math.field.element.FieldElement
      math.field.goldilocks.GoldilocksField) ←
    (core_models.ops.arith.Mul.mul
      (math.field.element.FieldElement math.field.goldilocks.GoldilocksField)
      (math.field.element.FieldElement math.field.goldilocks.GoldilocksField)
      a1_sq
      (← a[(1 : usize)]_?));
  let
    a2_cubed : (math.field.element.FieldElement
      math.field.goldilocks.GoldilocksField) ←
    (core_models.ops.arith.Mul.mul
      (math.field.element.FieldElement math.field.goldilocks.GoldilocksField)
      (math.field.element.FieldElement math.field.goldilocks.GoldilocksField)
      a2_sq
      (← a[(2 : usize)]_?));
  let
    a0a1a2 : (math.field.element.FieldElement
      math.field.goldilocks.GoldilocksField) ←
    (core_models.ops.arith.Mul.mul
      (math.field.element.FieldElement math.field.goldilocks.GoldilocksField)
      (math.field.element.FieldElement math.field.goldilocks.GoldilocksField)
      (← (core_models.ops.arith.Mul.mul
        (math.field.element.FieldElement math.field.goldilocks.GoldilocksField)
        (math.field.element.FieldElement math.field.goldilocks.GoldilocksField)
        (← a[(0 : usize)]_?)
        (← a[(1 : usize)]_?)))
      (← a[(2 : usize)]_?));
  let
    norm : (math.field.element.FieldElement
      math.field.goldilocks.GoldilocksField) ←
    (core_models.ops.arith.Sub.sub
      (math.field.element.FieldElement math.field.goldilocks.GoldilocksField)
      (math.field.element.FieldElement math.field.goldilocks.GoldilocksField)
      (← (core_models.ops.arith.Add.add
        (math.field.element.FieldElement math.field.goldilocks.GoldilocksField)
        (math.field.element.FieldElement math.field.goldilocks.GoldilocksField)
        (← (core_models.ops.arith.Add.add
          (math.field.element.FieldElement
            math.field.goldilocks.GoldilocksField)
          (math.field.element.FieldElement
            math.field.goldilocks.GoldilocksField)
          a0_cubed
          (← (math.field.element.Impl_32.double
            math.field.goldilocks.GoldilocksField a1_cubed))))
        (← (math.field.element.Impl_32.double
          math.field.goldilocks.GoldilocksField
          (← (math.field.element.Impl_32.double
            math.field.goldilocks.GoldilocksField a2_cubed))))))
      (← (math.field.element.Impl_32.double
        math.field.goldilocks.GoldilocksField
        (← (core_models.ops.arith.Add.add
          (math.field.element.FieldElement
            math.field.goldilocks.GoldilocksField)
          (math.field.element.FieldElement
            math.field.goldilocks.GoldilocksField)
          (← (math.field.element.Impl_32.double
            math.field.goldilocks.GoldilocksField a0a1a2))
          a0a1a2)))));
  match
    (← (math.field.element.Impl_32.inv math.field.goldilocks.GoldilocksField
      norm))
  with
    | (core_models.result.Result.Ok  norm_inv) => do
      let
        a1a2 : (math.field.element.FieldElement
          math.field.goldilocks.GoldilocksField) ←
        (core_models.ops.arith.Mul.mul
          (math.field.element.FieldElement
            math.field.goldilocks.GoldilocksField)
          (math.field.element.FieldElement
            math.field.goldilocks.GoldilocksField)
          (← a[(1 : usize)]_?)
          (← a[(2 : usize)]_?));
      let
        a0a1 : (math.field.element.FieldElement
          math.field.goldilocks.GoldilocksField) ←
        (core_models.ops.arith.Mul.mul
          (math.field.element.FieldElement
            math.field.goldilocks.GoldilocksField)
          (math.field.element.FieldElement
            math.field.goldilocks.GoldilocksField)
          (← a[(0 : usize)]_?)
          (← a[(1 : usize)]_?));
      let
        a0a2 : (math.field.element.FieldElement
          math.field.goldilocks.GoldilocksField) ←
        (core_models.ops.arith.Mul.mul
          (math.field.element.FieldElement
            math.field.goldilocks.GoldilocksField)
          (math.field.element.FieldElement
            math.field.goldilocks.GoldilocksField)
          (← a[(0 : usize)]_?)
          (← a[(2 : usize)]_?));
      (pure (core_models.result.Result.Ok
        (RustArray.ofVec #v[(← (core_models.ops.arith.Mul.mul
                                (math.field.element.FieldElement
                                  math.field.goldilocks.GoldilocksField)
                                (math.field.element.FieldElement
                                  math.field.goldilocks.GoldilocksField)
                                (← (core_models.ops.arith.Sub.sub
                                  (math.field.element.FieldElement
                                    math.field.goldilocks.GoldilocksField)
                                  (math.field.element.FieldElement
                                    math.field.goldilocks.GoldilocksField)
                                  a0_sq
                                  (← (math.field.element.Impl_32.double
                                    math.field.goldilocks.GoldilocksField
                                    a1a2))))
                                norm_inv)),
                              (← (core_models.ops.arith.Mul.mul
                                (math.field.element.FieldElement
                                  math.field.goldilocks.GoldilocksField)
                                (math.field.element.FieldElement
                                  math.field.goldilocks.GoldilocksField)
                                (← (core_models.ops.arith.Sub.sub
                                  (math.field.element.FieldElement
                                    math.field.goldilocks.GoldilocksField)
                                  (math.field.element.FieldElement
                                    math.field.goldilocks.GoldilocksField)
                                  (← (math.field.element.Impl_32.double
                                    math.field.goldilocks.GoldilocksField
                                    a2_sq))
                                  a0a1))
                                norm_inv)),
                              (← (core_models.ops.arith.Mul.mul
                                (math.field.element.FieldElement
                                  math.field.goldilocks.GoldilocksField)
                                (math.field.element.FieldElement
                                  math.field.goldilocks.GoldilocksField)
                                (← (core_models.ops.arith.Sub.sub
                                  (math.field.element.FieldElement
                                    math.field.goldilocks.GoldilocksField)
                                  (math.field.element.FieldElement
                                    math.field.goldilocks.GoldilocksField)
                                  a1_sq
                                  a0a2))
                                norm_inv))])))
    | (core_models.result.Result.Err  err) => do
      (pure (core_models.result.Result.Err err))

@[spec]
def Impl_5.div_hoisted
    (a :
    (RustArray
    (math.field.element.FieldElement math.field.goldilocks.GoldilocksField)
    3))
    (b :
    (RustArray
    (math.field.element.FieldElement math.field.goldilocks.GoldilocksField)
    3)) :
    RustM
    (core_models.result.Result
      (RustArray
      (math.field.element.FieldElement math.field.goldilocks.GoldilocksField)
      3)
      math.field.errors.FieldError)
    := do
  match (← (Impl_5.inv_hoisted b)) with
    | (core_models.result.Result.Ok  b_inv) => do
      (pure (core_models.result.Result.Ok (← (Impl_5.mul_hoisted a b_inv))))
    | (core_models.result.Result.Err  err) => do
      (pure (core_models.result.Result.Err err))

@[spec]
def Impl_5.eq_hoisted
    (a :
    (RustArray
    (math.field.element.FieldElement math.field.goldilocks.GoldilocksField)
    3))
    (b :
    (RustArray
    (math.field.element.FieldElement math.field.goldilocks.GoldilocksField)
    3)) :
    RustM Bool := do
  ((← ((← (core_models.cmp.PartialEq.eq
        (math.field.element.FieldElement math.field.goldilocks.GoldilocksField)
        (math.field.element.FieldElement math.field.goldilocks.GoldilocksField)
        (← a[(0 : usize)]_?)
        (← b[(0 : usize)]_?)))
      &&? (← (core_models.cmp.PartialEq.eq
        (math.field.element.FieldElement math.field.goldilocks.GoldilocksField)
        (math.field.element.FieldElement math.field.goldilocks.GoldilocksField)
        (← a[(1 : usize)]_?)
        (← b[(1 : usize)]_?)))))
    &&? (← (core_models.cmp.PartialEq.eq
      (math.field.element.FieldElement math.field.goldilocks.GoldilocksField)
      (math.field.element.FieldElement math.field.goldilocks.GoldilocksField)
      (← a[(2 : usize)]_?)
      (← b[(2 : usize)]_?))))

@[spec]
def Impl_5.zero_hoisted (_ : rust_primitives.hax.Tuple0) :
    RustM
    (RustArray
    (math.field.element.FieldElement math.field.goldilocks.GoldilocksField)
    3)
    := do
  (pure (RustArray.ofVec #v[(← (math.field.element.Impl_32.zero
                                math.field.goldilocks.GoldilocksField
                                rust_primitives.hax.Tuple0.mk)),
                              (← (math.field.element.Impl_32.zero
                                math.field.goldilocks.GoldilocksField
                                rust_primitives.hax.Tuple0.mk)),
                              (← (math.field.element.Impl_32.zero
                                math.field.goldilocks.GoldilocksField
                                rust_primitives.hax.Tuple0.mk))]))

@[spec]
def Impl_5.one_hoisted (_ : rust_primitives.hax.Tuple0) :
    RustM
    (RustArray
    (math.field.element.FieldElement math.field.goldilocks.GoldilocksField)
    3)
    := do
  (pure (RustArray.ofVec #v[(← (math.field.element.Impl_32.one
                                math.field.goldilocks.GoldilocksField
                                rust_primitives.hax.Tuple0.mk)),
                              (← (math.field.element.Impl_32.zero
                                math.field.goldilocks.GoldilocksField
                                rust_primitives.hax.Tuple0.mk)),
                              (← (math.field.element.Impl_32.zero
                                math.field.goldilocks.GoldilocksField
                                rust_primitives.hax.Tuple0.mk))]))

@[spec]
def Impl_5.from_u64_hoisted (x : u64) :
    RustM
    (RustArray
    (math.field.element.FieldElement math.field.goldilocks.GoldilocksField)
    3)
    := do
  (pure (RustArray.ofVec #v[(← (core_models.convert.From._from
                                (math.field.element.FieldElement
                                  math.field.goldilocks.GoldilocksField)
                                u64 x)),
                              (← (math.field.element.Impl_32.zero
                                math.field.goldilocks.GoldilocksField
                                rust_primitives.hax.Tuple0.mk)),
                              (← (math.field.element.Impl_32.zero
                                math.field.goldilocks.GoldilocksField
                                rust_primitives.hax.Tuple0.mk))]))

@[spec]
def Impl_5.double_hoisted
    (a :
    (RustArray
    (math.field.element.FieldElement math.field.goldilocks.GoldilocksField)
    3)) :
    RustM
    (RustArray
    (math.field.element.FieldElement math.field.goldilocks.GoldilocksField)
    3)
    := do
  (pure (RustArray.ofVec #v[(← (math.field.element.Impl_32.double
                                math.field.goldilocks.GoldilocksField
                                (← a[(0 : usize)]_?))),
                              (← (math.field.element.Impl_32.double
                                math.field.goldilocks.GoldilocksField
                                (← a[(1 : usize)]_?))),
                              (← (math.field.element.Impl_32.double
                                math.field.goldilocks.GoldilocksField
                                (← a[(2 : usize)]_?)))]))

@[spec]
def Impl_6.mul_hoisted
    (a : u64)
    (b :
    (RustArray
    (math.field.element.FieldElement math.field.goldilocks.GoldilocksField)
    3)) :
    RustM
    (RustArray
    (math.field.element.FieldElement math.field.goldilocks.GoldilocksField)
    3)
    := do
  let
    c0 : (math.field.element.FieldElement
      math.field.goldilocks.GoldilocksField) ←
    (math.field.element.Impl_6.from_raw math.field.goldilocks.GoldilocksField
      (← (math.field.traits.IsField.mul
        math.field.goldilocks.GoldilocksField
        a
        (← (math.field.element.Impl_32.value
          math.field.goldilocks.GoldilocksField (← b[(0 : usize)]_?))))));
  let
    c1 : (math.field.element.FieldElement
      math.field.goldilocks.GoldilocksField) ←
    (math.field.element.Impl_6.from_raw math.field.goldilocks.GoldilocksField
      (← (math.field.traits.IsField.mul
        math.field.goldilocks.GoldilocksField
        a
        (← (math.field.element.Impl_32.value
          math.field.goldilocks.GoldilocksField (← b[(1 : usize)]_?))))));
  let
    c2 : (math.field.element.FieldElement
      math.field.goldilocks.GoldilocksField) ←
    (math.field.element.Impl_6.from_raw math.field.goldilocks.GoldilocksField
      (← (math.field.traits.IsField.mul
        math.field.goldilocks.GoldilocksField
        a
        (← (math.field.element.Impl_32.value
          math.field.goldilocks.GoldilocksField (← b[(2 : usize)]_?))))));
  (pure (RustArray.ofVec #v[c0, c1, c2]))

@[spec]
def Impl_6.add_hoisted
    (a : u64)
    (b :
    (RustArray
    (math.field.element.FieldElement math.field.goldilocks.GoldilocksField)
    3)) :
    RustM
    (RustArray
    (math.field.element.FieldElement math.field.goldilocks.GoldilocksField)
    3)
    := do
  let
    c0 : (math.field.element.FieldElement
      math.field.goldilocks.GoldilocksField) ←
    (math.field.element.Impl_6.from_raw math.field.goldilocks.GoldilocksField
      (← (math.field.traits.IsField.add
        math.field.goldilocks.GoldilocksField
        a
        (← (math.field.element.Impl_32.value
          math.field.goldilocks.GoldilocksField (← b[(0 : usize)]_?))))));
  (pure (RustArray.ofVec #v[c0, (← b[(1 : usize)]_?), (← b[(2 : usize)]_?)]))

@[spec]
def Impl_6.sub_hoisted
    (a : u64)
    (b :
    (RustArray
    (math.field.element.FieldElement math.field.goldilocks.GoldilocksField)
    3)) :
    RustM
    (RustArray
    (math.field.element.FieldElement math.field.goldilocks.GoldilocksField)
    3)
    := do
  let
    c0 : (math.field.element.FieldElement
      math.field.goldilocks.GoldilocksField) ←
    (math.field.element.Impl_6.from_raw math.field.goldilocks.GoldilocksField
      (← (math.field.traits.IsField.sub
        math.field.goldilocks.GoldilocksField
        a
        (← (math.field.element.Impl_32.value
          math.field.goldilocks.GoldilocksField (← b[(0 : usize)]_?))))));
  let
    c1 : (math.field.element.FieldElement
      math.field.goldilocks.GoldilocksField) ←
    (math.field.element.Impl_6.from_raw math.field.goldilocks.GoldilocksField
      (← (math.field.traits.IsField.neg
        math.field.goldilocks.GoldilocksField
        (← (math.field.element.Impl_32.value
          math.field.goldilocks.GoldilocksField (← b[(1 : usize)]_?))))));
  let
    c2 : (math.field.element.FieldElement
      math.field.goldilocks.GoldilocksField) ←
    (math.field.element.Impl_6.from_raw math.field.goldilocks.GoldilocksField
      (← (math.field.traits.IsField.neg
        math.field.goldilocks.GoldilocksField
        (← (math.field.element.Impl_32.value
          math.field.goldilocks.GoldilocksField (← b[(2 : usize)]_?))))));
  (pure (RustArray.ofVec #v[c0, c1, c2]))

@[spec]
def Impl_6.embed_hoisted (a : u64) :
    RustM
    (RustArray
    (math.field.element.FieldElement math.field.goldilocks.GoldilocksField)
    3)
    := do
  (pure (RustArray.ofVec #v[(← (math.field.element.Impl_6.from_raw
                                math.field.goldilocks.GoldilocksField a)),
                              (← (math.field.element.Impl_32.zero
                                math.field.goldilocks.GoldilocksField
                                rust_primitives.hax.Tuple0.mk)),
                              (← (math.field.element.Impl_32.zero
                                math.field.goldilocks.GoldilocksField
                                rust_primitives.hax.Tuple0.mk))]))

@[spec]
def Impl_6.to_subfield_vec_hoisted
    (b :
    (RustArray
    (math.field.element.FieldElement math.field.goldilocks.GoldilocksField)
    3)) :
    RustM (alloc.vec.Vec u64 alloc.alloc.Global) := do
  let out : (alloc.vec.Vec u64 alloc.alloc.Global) ←
    (alloc.vec.Impl.with_capacity u64
      (← (core_models.slice.Impl.len
        (math.field.element.FieldElement math.field.goldilocks.GoldilocksField)
        (← (rust_primitives.unsize b)))));
  let out : (alloc.vec.Vec u64 alloc.alloc.Global) ←
    (rust_primitives.hax.folds.fold_range
      (0 : usize)
      (← (core_models.slice.Impl.len
        (math.field.element.FieldElement math.field.goldilocks.GoldilocksField)
        (← (rust_primitives.unsize b))))
      (fun out _ => (do (pure true) : RustM Bool))
      out
      (fun out i =>
        (do
        (alloc.vec.Impl_1.push u64 alloc.alloc.Global
          out
          (← (math.field.element.Impl_32.to_raw
            math.field.goldilocks.GoldilocksField (← b[i]_?)))) :
        RustM (alloc.vec.Vec u64 alloc.alloc.Global))));
  (pure out)

end math.field.extensions_goldilocks


namespace math.field.goldilocks

--  Multiply a raw u64 field element by 7 (the Fp2 non-residue).
--  Uses 7 = 8 - 1 for a straight-line computation.
@[spec]
def mul_by_7_raw (a : u64) : RustM u64 := do
  let a2 : u64 ← (math.field.traits.IsField.double GoldilocksField a);
  let a4 : u64 ← (math.field.traits.IsField.double GoldilocksField a2);
  let a8 : u64 ← (math.field.traits.IsField.double GoldilocksField a4);
  (math.field.traits.IsField.sub GoldilocksField a8 a)

end math.field.goldilocks


namespace math.field.extensions_goldilocks

--  Multiplication using fused dot products for fewer reductions.
--  (a0 + a1*w) * (b0 + b1*w) = (a0*b0 + 7*a1*b1) + (a0*b1 + a1*b0)*w
-- 
--  Uses dot_product_2 to compute each output component with a single
--  reduce128 instead of separate mul + reduce per product.
@[spec]
def Impl_2.mul_hoisted
    (a :
    (RustArray
    (math.field.element.FieldElement math.field.goldilocks.GoldilocksField)
    2))
    (b :
    (RustArray
    (math.field.element.FieldElement math.field.goldilocks.GoldilocksField)
    2)) :
    RustM
    (RustArray
    (math.field.element.FieldElement math.field.goldilocks.GoldilocksField)
    2)
    := do
  let ⟨a0, a1⟩ :=
    (rust_primitives.hax.Tuple2.mk
      (← (math.field.element.Impl_32.value math.field.goldilocks.GoldilocksField
        (← a[(0 : usize)]_?)))
      (← (math.field.element.Impl_32.value math.field.goldilocks.GoldilocksField
        (← a[(1 : usize)]_?))));
  let ⟨b0, b1⟩ :=
    (rust_primitives.hax.Tuple2.mk
      (← (math.field.element.Impl_32.value math.field.goldilocks.GoldilocksField
        (← b[(0 : usize)]_?)))
      (← (math.field.element.Impl_32.value math.field.goldilocks.GoldilocksField
        (← b[(1 : usize)]_?))));
  let b1_7 : u64 ← (math.field.goldilocks.mul_by_7_raw b1);
  let c0 : u64 ← (math.field.goldilocks.dot_product_2 a0 b0 a1 b1_7);
  let c1 : u64 ← (math.field.goldilocks.dot_product_2 a0 b1 a1 b0);
  (pure (RustArray.ofVec #v[(← (math.field.element.Impl_6.from_raw
                                math.field.goldilocks.GoldilocksField c0)),
                              (← (math.field.element.Impl_6.from_raw
                                math.field.goldilocks.GoldilocksField c1))]))

--  Squaring using fused dot product for the first component.
--  (a0 + a1*w)^2 = (a0^2 + 7*a1^2) + 2*a0*a1*w
@[spec]
def Impl_2.square_hoisted
    (a :
    (RustArray
    (math.field.element.FieldElement math.field.goldilocks.GoldilocksField)
    2)) :
    RustM
    (RustArray
    (math.field.element.FieldElement math.field.goldilocks.GoldilocksField)
    2)
    := do
  let ⟨a0, a1⟩ :=
    (rust_primitives.hax.Tuple2.mk
      (← (math.field.element.Impl_32.value math.field.goldilocks.GoldilocksField
        (← a[(0 : usize)]_?)))
      (← (math.field.element.Impl_32.value math.field.goldilocks.GoldilocksField
        (← a[(1 : usize)]_?))));
  let a1_7 : u64 ← (math.field.goldilocks.mul_by_7_raw a1);
  let c0 : u64 ← (math.field.goldilocks.dot_product_2 a0 a0 a1 a1_7);
  let c1 : u64 ←
    (math.field.traits.IsField.mul math.field.goldilocks.GoldilocksField a0 a1);
  let c1 : u64 ←
    (math.field.traits.IsField.double math.field.goldilocks.GoldilocksField c1);
  (pure (RustArray.ofVec #v[(← (math.field.element.Impl_6.from_raw
                                math.field.goldilocks.GoldilocksField c0)),
                              (← (math.field.element.Impl_6.from_raw
                                math.field.goldilocks.GoldilocksField c1))]))

--  Multiply a field element by 7 (the quadratic non-residue).
--  Wraps the raw u64 implementation for use with FieldElement types.
@[spec]
def mul_by_7
    (a :
    (math.field.element.FieldElement math.field.goldilocks.GoldilocksField)) :
    RustM
    (math.field.element.FieldElement math.field.goldilocks.GoldilocksField)
    := do
  (math.field.element.Impl_6.from_raw math.field.goldilocks.GoldilocksField
    (← (math.field.goldilocks.mul_by_7_raw
      (← (math.field.element.Impl_32.value math.field.goldilocks.GoldilocksField
        a)))))

--  Returns the multiplicative inverse of `a`:
--  (a0 + a1*w)^-1 = (a0 - a1*w) / (a0^2 - W*a1^2)
@[spec]
def Impl_2.inv_hoisted
    (a :
    (RustArray
    (math.field.element.FieldElement math.field.goldilocks.GoldilocksField)
    2)) :
    RustM
    (core_models.result.Result
      (RustArray
      (math.field.element.FieldElement math.field.goldilocks.GoldilocksField)
      2)
      math.field.errors.FieldError)
    := do
  let
    a0_sq : (math.field.element.FieldElement
      math.field.goldilocks.GoldilocksField) ←
    (math.field.element.Impl_32.square math.field.goldilocks.GoldilocksField
      (← a[(0 : usize)]_?));
  let
    a1_sq : (math.field.element.FieldElement
      math.field.goldilocks.GoldilocksField) ←
    (math.field.element.Impl_32.square math.field.goldilocks.GoldilocksField
      (← a[(1 : usize)]_?));
  let
    w_a1_sq : (math.field.element.FieldElement
      math.field.goldilocks.GoldilocksField) ←
    (mul_by_7 a1_sq);
  let
    norm : (math.field.element.FieldElement
      math.field.goldilocks.GoldilocksField) ←
    (core_models.ops.arith.Sub.sub
      (math.field.element.FieldElement math.field.goldilocks.GoldilocksField)
      (math.field.element.FieldElement math.field.goldilocks.GoldilocksField)
      a0_sq
      w_a1_sq);
  match
    (← (math.field.element.Impl_32.inv math.field.goldilocks.GoldilocksField
      norm))
  with
    | (core_models.result.Result.Ok  norm_inv) => do
      (pure (core_models.result.Result.Ok
        (RustArray.ofVec #v[(← (core_models.ops.arith.Mul.mul
                                (math.field.element.FieldElement
                                  math.field.goldilocks.GoldilocksField)
                                (math.field.element.FieldElement
                                  math.field.goldilocks.GoldilocksField)
                                (← a[(0 : usize)]_?)
                                norm_inv)),
                              (← (core_models.ops.arith.Mul.mul
                                (math.field.element.FieldElement
                                  math.field.goldilocks.GoldilocksField)
                                (math.field.element.FieldElement
                                  math.field.goldilocks.GoldilocksField)
                                (← (core_models.ops.arith.Neg.neg
                                  (math.field.element.FieldElement
                                    math.field.goldilocks.GoldilocksField)
                                  (← a[(1 : usize)]_?)))
                                norm_inv))])))
    | (core_models.result.Result.Err  err) => do
      (pure (core_models.result.Result.Err err))

@[spec]
def Impl_2.div_hoisted
    (a :
    (RustArray
    (math.field.element.FieldElement math.field.goldilocks.GoldilocksField)
    2))
    (b :
    (RustArray
    (math.field.element.FieldElement math.field.goldilocks.GoldilocksField)
    2)) :
    RustM
    (core_models.result.Result
      (RustArray
      (math.field.element.FieldElement math.field.goldilocks.GoldilocksField)
      2)
      math.field.errors.FieldError)
    := do
  match (← (Impl_2.inv_hoisted b)) with
    | (core_models.result.Result.Ok  b_inv) => do
      (pure (core_models.result.Result.Ok (← (Impl_2.mul_hoisted a b_inv))))
    | (core_models.result.Result.Err  err) => do
      (pure (core_models.result.Result.Err err))

@[reducible] instance Impl_2.AssociatedTypes :
  math.field.traits.IsField.AssociatedTypes Degree2GoldilocksExtensionField
  where
  BaseType := (RustArray
  (math.field.element.FieldElement math.field.goldilocks.GoldilocksField)
  2)

instance Impl_2 :
  math.field.traits.IsField Degree2GoldilocksExtensionField
  where
  add := (Impl_2.add_hoisted)
  mul := (Impl_2.mul_hoisted)
  square := (Impl_2.square_hoisted)
  sub := (Impl_2.sub_hoisted)
  neg := (Impl_2.neg_hoisted)
  inv := (Impl_2.inv_hoisted)
  div := (Impl_2.div_hoisted)
  eq := (Impl_2.eq_hoisted)
  zero := (Impl_2.zero_hoisted)
  one := (Impl_2.one_hoisted)
  from_u64 := (Impl_2.from_u64_hoisted)
  from_base_type := (Impl_2.from_base_type_hoisted)
  double := (Impl_2.double_hoisted)

@[spec]
def Impl_3.div_hoisted
    (a : u64)
    (b :
    (RustArray
    (math.field.element.FieldElement math.field.goldilocks.GoldilocksField)
    2)) :
    RustM
    (core_models.result.Result
      (RustArray
      (math.field.element.FieldElement math.field.goldilocks.GoldilocksField)
      2)
      math.field.errors.FieldError)
    := do
  match
    (← (math.field.traits.IsField.inv Degree2GoldilocksExtensionField b))
  with
    | (core_models.result.Result.Ok  b_inv) => do
      (pure (core_models.result.Result.Ok
        (← (Impl_3.mul_hoisted Degree2GoldilocksExtensionField a b_inv))))
    | (core_models.result.Result.Err  err) => do
      (pure (core_models.result.Result.Err err))

@[reducible] instance Impl_3.AssociatedTypes :
  math.field.traits.IsSubFieldOf.AssociatedTypes
  math.field.goldilocks.GoldilocksField
  Degree2GoldilocksExtensionField
  where

instance Impl_3 :
  math.field.traits.IsSubFieldOf
  math.field.goldilocks.GoldilocksField
  Degree2GoldilocksExtensionField
  where
  mul := (Impl_3.mul_hoisted)
  add := (Impl_3.add_hoisted)
  div := (Impl_3.div_hoisted)
  sub := (Impl_3.sub_hoisted)
  embed := (Impl_3.embed_hoisted)
  to_subfield_vec := (Impl_3.to_subfield_vec_hoisted)

--  Returns the conjugate of self: conjugate(a0 + a1*w) = a0 - a1*w
@[spec]
def Impl_4.conjugate
    (self : (math.field.element.FieldElement Degree2GoldilocksExtensionField)) :
    RustM
    (math.field.element.FieldElement Degree2GoldilocksExtensionField)
    := do
  (math.field.element.Impl_32.new Degree2GoldilocksExtensionField
    (RustArray.ofVec #v[(← (← (math.field.element.Impl_32.value
                              Degree2GoldilocksExtensionField self))[
                            (0 : usize)
                            ]_?),
                          (← (core_models.ops.arith.Neg.neg
                            (math.field.element.FieldElement
                              math.field.goldilocks.GoldilocksField)
                            (← (← (math.field.element.Impl_32.value
                                Degree2GoldilocksExtensionField self))[
                              (1 : usize)
                              ]_?)))]))

--  Create a field element from an i64.
--  Negative values are converted to their field equivalents: -x becomes p - x.
@[spec]
def Impl_4.from_i64 (value : i64) :
    RustM
    (math.field.element.FieldElement Degree2GoldilocksExtensionField)
    := do
  (core_models.convert.From._from
    (math.field.element.FieldElement Degree2GoldilocksExtensionField)
    i64 value)

end math.field.extensions_goldilocks


namespace math.field.goldilocks

@[spec]
def inv_addition_chain.exp_acc (base : u64) (tail : u64) (n : u32) :
    RustM u64 := do
  let result : u64 := base;
  let result : u64 ←
    (rust_primitives.hax.folds.fold_range
      (0 : u32)
      n
      (fun result _ => (do (pure true) : RustM Bool))
      result
      (fun result _ =>
        (do
        (math.field.traits.IsField.square GoldilocksField result) :
        RustM u64)));
  (math.field.traits.IsField.mul GoldilocksField result tail)

--  Inversion using optimized addition chain for a^(p-2).
--  Based on Plonky2's approach.
-- 
--  p - 2 = 0xFFFFFFFE_FFFFFFFF = 2^64 - 2^32 - 1
--  Binary structure: 32 ones, one zero, 31 ones
-- 
--  This uses approximately 72 multiplications (vs ~96 for binary exp).
@[spec]
def inv_addition_chain (base : u64) : RustM u64 := do
  let x : u64 := base;
  let x2 : u64 ← (math.field.traits.IsField.square GoldilocksField x);
  let x3 : u64 ← (math.field.traits.IsField.mul GoldilocksField x2 x);
  let x7 : u64 ← (inv_addition_chain.exp_acc x3 x (1 : u32));
  let x63 : u64 ← (inv_addition_chain.exp_acc x7 x7 (3 : u32));
  let x12m1 : u64 ← (inv_addition_chain.exp_acc x63 x63 (6 : u32));
  let x24m1 : u64 ← (inv_addition_chain.exp_acc x12m1 x12m1 (12 : u32));
  let x30m1 : u64 ← (inv_addition_chain.exp_acc x24m1 x63 (6 : u32));
  let x31m1 : u64 ← (inv_addition_chain.exp_acc x30m1 x (1 : u32));
  let x32m1 : u64 ← (inv_addition_chain.exp_acc x31m1 x (1 : u32));
  let t : u64 := x31m1;
  let t : u64 ←
    (rust_primitives.hax.folds.fold_range
      (0 : i32)
      (33 : i32)
      (fun t _ => (do (pure true) : RustM Bool))
      t
      (fun t _ =>
        (do (math.field.traits.IsField.square GoldilocksField t) : RustM u64)));
  (math.field.traits.IsField.mul GoldilocksField t x32m1)

--  Compute a^(p-2) for field inversion using the optimized addition chain.
@[spec]
def exp_p_minus_2 (base : u64) : RustM u64 := do (inv_addition_chain base)

@[reducible] instance Impl_4.AssociatedTypes :
  math.field.traits.IsPrimeField.AssociatedTypes GoldilocksField
  where
  CanonicalType := u64

instance Impl_4 : math.field.traits.IsPrimeField GoldilocksField where
  canonical := (Impl_4.canonical_hoisted)
  from_hex := (Impl_4.from_hex_hoisted)
  field_bit_size := (Impl_4.field_bit_size_hoisted)

--  Negation: -a = p - a (or 0 if a = 0)
@[spec]
def Impl.neg_hoisted (a : u64) : RustM u64 := do
  let canonical : u64 ←
    (math.field.traits.IsPrimeField.canonical GoldilocksField a);
  if (← (canonical ==? (0 : u64))) then do
    (pure (0 : u64))
  else do
    (GOLDILOCKS_PRIME -? canonical)

--  Multiplicative inverse using Fermat's little theorem: a^(-1) = a^(p-2)
@[spec]
def Impl.inv_hoisted (a : u64) :
    RustM (core_models.result.Result u64 math.field.errors.FieldError) := do
  let canonical : u64 ←
    (math.field.traits.IsPrimeField.canonical GoldilocksField a);
  if (← (canonical ==? (0 : u64))) then do
    (pure (core_models.result.Result.Err
      math.field.errors.FieldError.InvZeroError))
  else do
    (pure (core_models.result.Result.Ok (← (exp_p_minus_2 canonical))))

@[spec]
def Impl.eq_hoisted (a : u64) (b : u64) : RustM Bool := do
  ((← (math.field.traits.IsPrimeField.canonical GoldilocksField a))
    ==? (← (math.field.traits.IsPrimeField.canonical GoldilocksField b)))

@[spec]
def Impl.div_hoisted (a : u64) (b : u64) :
    RustM (core_models.result.Result u64 math.field.errors.FieldError) := do
  match (← (Impl.inv_hoisted b)) with
    | (core_models.result.Result.Ok  b_inv) => do
      (pure (core_models.result.Result.Ok (← (Impl.mul_hoisted a b_inv))))
    | (core_models.result.Result.Err  err) => do
      (pure (core_models.result.Result.Err err))

@[reducible] instance Impl.AssociatedTypes :
  math.field.traits.IsField.AssociatedTypes GoldilocksField
  where
  BaseType := u64

instance Impl : math.field.traits.IsField GoldilocksField where
  add := (Impl.add_hoisted)
  sub := (Impl.sub_hoisted)
  mul := (Impl.mul_hoisted)
  square := (Impl.square_hoisted)
  neg := (Impl.neg_hoisted)
  inv := (Impl.inv_hoisted)
  div := (Impl.div_hoisted)
  eq := (Impl.eq_hoisted)
  zero := (Impl.zero_hoisted)
  one := (Impl.one_hoisted)
  from_u64 := (Impl.from_u64_hoisted)
  from_base_type := (Impl.from_base_type_hoisted)
  double := (Impl.double_hoisted)

--  Create a new field element from a u64.
@[spec]
def Impl_1.from_canonical_u64 (n : u64) :
    RustM (math.field.element.FieldElement GoldilocksField) := do
  (core_models.convert.From._from
    (math.field.element.FieldElement GoldilocksField)
    u64 n)

--  Get the canonical u64 representation in [0, p).
@[spec]
def Impl_1.canonical_u64
    (self : (math.field.element.FieldElement GoldilocksField)) :
    RustM u64 := do
  (math.field.traits.IsPrimeField.canonical
    GoldilocksField (← (math.field.element.Impl_32.value GoldilocksField self)))

--  Convert to little-endian bytes.
@[spec]
def Impl_1.to_bytes_le
    (self : (math.field.element.FieldElement GoldilocksField)) :
    RustM (RustArray u8 8) := do
  (core_models.num.Impl_9.to_le_bytes (← (Impl_1.canonical_u64 self)))

--  Convert to big-endian bytes.
@[spec]
def Impl_1.to_bytes_be
    (self : (math.field.element.FieldElement GoldilocksField)) :
    RustM (RustArray u8 8) := do
  (core_models.num.Impl_9.to_be_bytes (← (Impl_1.canonical_u64 self)))

--  Create a field element from an i64.
--  Negative values are converted to their field equivalents: -x becomes p - x.
@[spec]
def Impl_1.from_i64 (value : i64) :
    RustM (math.field.element.FieldElement GoldilocksField) := do
  (core_models.convert.From._from
    (math.field.element.FieldElement GoldilocksField)
    i64 value)

@[spec]
def Impl_2.write_bytes_be_hoisted
    (self : (math.field.element.FieldElement GoldilocksField))
    (buf : (RustSlice u8)) :
    RustM (RustSlice u8) := do
  let _ ←
    if true then do
      let _ ←
        (hax_lib.assert
          (← ((← (core_models.slice.Impl.len u8 buf)) >=? (8 : usize))));
      (pure rust_primitives.hax.Tuple0.mk)
    else do
      (pure rust_primitives.hax.Tuple0.mk);
  let buf : (RustSlice u8) ←
    (rust_primitives.hax.monomorphized_update_at.update_at_range_to
      buf
      (core_models.ops.range.RangeTo.mk (_end := (8 : usize)))
      (← (core_models.slice.Impl.copy_from_slice u8
        (← buf[(core_models.ops.range.RangeTo.mk (_end := (8 : usize)))]_?)
        (← (rust_primitives.unsize
          (← (core_models.num.Impl_9.to_be_bytes
            (← (Impl_1.canonical_u64 self)))))))));
  (pure buf)

@[spec]
def Impl_2.to_bytes_be_hoisted
    (self : (math.field.element.FieldElement GoldilocksField)) :
    RustM (alloc.vec.Vec u8 alloc.alloc.Global) := do
  (alloc.slice.Impl.to_vec u8
    (← (rust_primitives.unsize
      (← (core_models.num.Impl_9.to_be_bytes
        (← (Impl_1.canonical_u64 self)))))))

@[spec]
def Impl_2.to_bytes_le_hoisted
    (self : (math.field.element.FieldElement GoldilocksField)) :
    RustM (alloc.vec.Vec u8 alloc.alloc.Global) := do
  (alloc.slice.Impl.to_vec u8
    (← (rust_primitives.unsize
      (← (core_models.num.Impl_9.to_le_bytes
        (← (Impl_1.canonical_u64 self)))))))

@[spec]
def Impl_2.from_bytes_be_hoisted (bytes : (RustSlice u8)) :
    RustM
    (core_models.result.Result
      (math.field.element.FieldElement GoldilocksField)
      math.errors.ByteConversionError)
    := do
  match
    (← (core_models.option.Impl.ok_or
      (RustSlice u8)
      math.errors.ByteConversionError
      (← (core_models.slice.Impl.get u8 (core_models.ops.range.Range usize)
        bytes
        (core_models.ops.range.Range.mk
          (start := (0 : usize))
          (_end := (8 : usize)))))
      math.errors.ByteConversionError.FromBEBytesError))
  with
    | (core_models.result.Result.Ok  needed_bytes) => do
      match
        (← (core_models.result.Impl.map_err
          (RustArray u8 8)
          core_models.array.TryFromSliceError
          math.errors.ByteConversionError
          (core_models.array.TryFromSliceError ->
          RustM math.errors.ByteConversionError)
          (← (core_models.convert.TryInto.try_into
            (RustSlice u8)
            (RustArray u8 8) needed_bytes))
          (fun _ =>
            (do
            (pure math.errors.ByteConversionError.FromBEBytesError) :
            RustM math.errors.ByteConversionError))))
      with
        | (core_models.result.Result.Ok  hoist9) => do
          let value : u64 ← (core_models.num.Impl_9.from_be_bytes hoist9);
          (pure (core_models.result.Result.Ok
            (← (core_models.convert.From._from
              (math.field.element.FieldElement GoldilocksField)
              u64 value))))
        | (core_models.result.Result.Err  err) => do
          (pure (core_models.result.Result.Err err))
    | (core_models.result.Result.Err  err) => do
      (pure (core_models.result.Result.Err err))

@[spec]
def Impl_2.from_bytes_le_hoisted (bytes : (RustSlice u8)) :
    RustM
    (core_models.result.Result
      (math.field.element.FieldElement GoldilocksField)
      math.errors.ByteConversionError)
    := do
  match
    (← (core_models.option.Impl.ok_or
      (RustSlice u8)
      math.errors.ByteConversionError
      (← (core_models.slice.Impl.get u8 (core_models.ops.range.Range usize)
        bytes
        (core_models.ops.range.Range.mk
          (start := (0 : usize))
          (_end := (8 : usize)))))
      math.errors.ByteConversionError.FromLEBytesError))
  with
    | (core_models.result.Result.Ok  needed_bytes) => do
      match
        (← (core_models.result.Impl.map_err
          (RustArray u8 8)
          core_models.array.TryFromSliceError
          math.errors.ByteConversionError
          (core_models.array.TryFromSliceError ->
          RustM math.errors.ByteConversionError)
          (← (core_models.convert.TryInto.try_into
            (RustSlice u8)
            (RustArray u8 8) needed_bytes))
          (fun _ =>
            (do
            (pure math.errors.ByteConversionError.FromLEBytesError) :
            RustM math.errors.ByteConversionError))))
      with
        | (core_models.result.Result.Ok  hoist10) => do
          let value : u64 ← (core_models.num.Impl_9.from_le_bytes hoist10);
          (pure (core_models.result.Result.Ok
            (← (core_models.convert.From._from
              (math.field.element.FieldElement GoldilocksField)
              u64 value))))
        | (core_models.result.Result.Err  err) => do
          (pure (core_models.result.Result.Err err))
    | (core_models.result.Result.Err  err) => do
      (pure (core_models.result.Result.Err err))

@[reducible] instance Impl_2.AssociatedTypes :
  math.traits.ByteConversion.AssociatedTypes
  (math.field.element.FieldElement GoldilocksField)
  where

instance Impl_2 :
  math.traits.ByteConversion (math.field.element.FieldElement GoldilocksField)
  where
  BYTE_LEN := (Impl_2.BYTE_LEN_hoisted)
  write_bytes_be := (Impl_2.write_bytes_be_hoisted)
  to_bytes_be := (Impl_2.to_bytes_be_hoisted)
  to_bytes_le := (Impl_2.to_bytes_le_hoisted)
  from_bytes_be := (Impl_2.from_bytes_be_hoisted)
  from_bytes_le := (Impl_2.from_bytes_le_hoisted)

end math.field.goldilocks


namespace math.field.extensions_goldilocks

@[spec]
def Impl_1.to_bytes_be_hoisted
    (self :
    (RustArray
    (math.field.element.FieldElement math.field.goldilocks.GoldilocksField)
    3)) :
    RustM (alloc.vec.Vec u8 alloc.alloc.Global) := do
  let bytes : (alloc.vec.Vec u8 alloc.alloc.Global) ←
    (math.traits.ByteConversion.to_bytes_be
      (math.field.element.FieldElement math.field.goldilocks.GoldilocksField)
      (← self[(2 : usize)]_?));
  let bytes : (alloc.vec.Vec u8 alloc.alloc.Global) ←
    (core_models.iter.traits.collect.Extend.extend
      (alloc.vec.Vec u8 alloc.alloc.Global)
      u8 (alloc.vec.Vec u8 alloc.alloc.Global)
      bytes
      (← (math.traits.ByteConversion.to_bytes_be
        (math.field.element.FieldElement math.field.goldilocks.GoldilocksField)
        (← self[(1 : usize)]_?))));
  let bytes : (alloc.vec.Vec u8 alloc.alloc.Global) ←
    (core_models.iter.traits.collect.Extend.extend
      (alloc.vec.Vec u8 alloc.alloc.Global)
      u8 (alloc.vec.Vec u8 alloc.alloc.Global)
      bytes
      (← (math.traits.ByteConversion.to_bytes_be
        (math.field.element.FieldElement math.field.goldilocks.GoldilocksField)
        (← self[(0 : usize)]_?))));
  (pure bytes)

@[spec]
def Impl_1.to_bytes_le_hoisted
    (self :
    (RustArray
    (math.field.element.FieldElement math.field.goldilocks.GoldilocksField)
    3)) :
    RustM (alloc.vec.Vec u8 alloc.alloc.Global) := do
  let bytes : (alloc.vec.Vec u8 alloc.alloc.Global) ←
    (math.traits.ByteConversion.to_bytes_le
      (math.field.element.FieldElement math.field.goldilocks.GoldilocksField)
      (← self[(0 : usize)]_?));
  let bytes : (alloc.vec.Vec u8 alloc.alloc.Global) ←
    (core_models.iter.traits.collect.Extend.extend
      (alloc.vec.Vec u8 alloc.alloc.Global)
      u8 (alloc.vec.Vec u8 alloc.alloc.Global)
      bytes
      (← (math.traits.ByteConversion.to_bytes_le
        (math.field.element.FieldElement math.field.goldilocks.GoldilocksField)
        (← self[(1 : usize)]_?))));
  let bytes : (alloc.vec.Vec u8 alloc.alloc.Global) ←
    (core_models.iter.traits.collect.Extend.extend
      (alloc.vec.Vec u8 alloc.alloc.Global)
      u8 (alloc.vec.Vec u8 alloc.alloc.Global)
      bytes
      (← (math.traits.ByteConversion.to_bytes_le
        (math.field.element.FieldElement math.field.goldilocks.GoldilocksField)
        (← self[(2 : usize)]_?))));
  (pure bytes)

@[spec]
def Impl_1.from_bytes_be_hoisted (bytes : (RustSlice u8)) :
    RustM
    (core_models.result.Result
      (RustArray
      (math.field.element.FieldElement math.field.goldilocks.GoldilocksField)
      3)
      math.errors.ByteConversionError)
    := do
  if
  (← ((← (core_models.slice.Impl.len u8 bytes))
    <? (← (Impl_1.from_bytes_be.N *? (3 : usize))))) then do
    (pure (core_models.result.Result.Err
      math.errors.ByteConversionError.FromBEBytesError))
  else do
    match
      (← (math.traits.ByteConversion.from_bytes_be
        (math.field.element.FieldElement math.field.goldilocks.GoldilocksField)
        (← bytes[
          (core_models.ops.range.Range.mk
            (start := (0 : usize))
            (_end := Impl_1.from_bytes_be.N))
          ]_?)))
    with
      | (core_models.result.Result.Ok  x2) => do
        match
          (← (math.traits.ByteConversion.from_bytes_be
            (math.field.element.FieldElement
              math.field.goldilocks.GoldilocksField)
            (← bytes[
              (core_models.ops.range.Range.mk
                (start := Impl_1.from_bytes_be.N)
                (_end := (← (Impl_1.from_bytes_be.N *? (2 : usize)))))
              ]_?)))
        with
          | (core_models.result.Result.Ok  x1) => do
            match
              (← (math.traits.ByteConversion.from_bytes_be
                (math.field.element.FieldElement
                  math.field.goldilocks.GoldilocksField)
                (← bytes[
                  (core_models.ops.range.Range.mk
                    (start := (← (Impl_1.from_bytes_be.N *? (2 : usize))))
                    (_end := (← (Impl_1.from_bytes_be.N *? (3 : usize)))))
                  ]_?)))
            with
              | (core_models.result.Result.Ok  x0) => do
                (pure (core_models.result.Result.Ok
                  (RustArray.ofVec #v[x0, x1, x2])))
              | (core_models.result.Result.Err  err) => do
                (pure (core_models.result.Result.Err err))
          | (core_models.result.Result.Err  err) => do
            (pure (core_models.result.Result.Err err))
      | (core_models.result.Result.Err  err) => do
        (pure (core_models.result.Result.Err err))

@[spec]
def Impl_1.from_bytes_le_hoisted (bytes : (RustSlice u8)) :
    RustM
    (core_models.result.Result
      (RustArray
      (math.field.element.FieldElement math.field.goldilocks.GoldilocksField)
      3)
      math.errors.ByteConversionError)
    := do
  if
  (← ((← (core_models.slice.Impl.len u8 bytes))
    <? (← (Impl_1.from_bytes_le.N *? (3 : usize))))) then do
    (pure (core_models.result.Result.Err
      math.errors.ByteConversionError.FromLEBytesError))
  else do
    match
      (← (math.traits.ByteConversion.from_bytes_le
        (math.field.element.FieldElement math.field.goldilocks.GoldilocksField)
        (← bytes[
          (core_models.ops.range.Range.mk
            (start := (0 : usize))
            (_end := Impl_1.from_bytes_le.N))
          ]_?)))
    with
      | (core_models.result.Result.Ok  x0) => do
        match
          (← (math.traits.ByteConversion.from_bytes_le
            (math.field.element.FieldElement
              math.field.goldilocks.GoldilocksField)
            (← bytes[
              (core_models.ops.range.Range.mk
                (start := Impl_1.from_bytes_le.N)
                (_end := (← (Impl_1.from_bytes_le.N *? (2 : usize)))))
              ]_?)))
        with
          | (core_models.result.Result.Ok  x1) => do
            match
              (← (math.traits.ByteConversion.from_bytes_le
                (math.field.element.FieldElement
                  math.field.goldilocks.GoldilocksField)
                (← bytes[
                  (core_models.ops.range.Range.mk
                    (start := (← (Impl_1.from_bytes_le.N *? (2 : usize))))
                    (_end := (← (Impl_1.from_bytes_le.N *? (3 : usize)))))
                  ]_?)))
            with
              | (core_models.result.Result.Ok  x2) => do
                (pure (core_models.result.Result.Ok
                  (RustArray.ofVec #v[x0, x1, x2])))
              | (core_models.result.Result.Err  err) => do
                (pure (core_models.result.Result.Err err))
          | (core_models.result.Result.Err  err) => do
            (pure (core_models.result.Result.Err err))
      | (core_models.result.Result.Err  err) => do
        (pure (core_models.result.Result.Err err))

@[reducible] instance Impl_1.AssociatedTypes :
  math.traits.ByteConversion.AssociatedTypes
  (RustArray
  (math.field.element.FieldElement math.field.goldilocks.GoldilocksField)
  3)
  where

instance Impl_1 :
  math.traits.ByteConversion
  (RustArray
  (math.field.element.FieldElement math.field.goldilocks.GoldilocksField)
  3)
  where
  BYTE_LEN := (Impl_1.BYTE_LEN_hoisted)
  to_bytes_be := (Impl_1.to_bytes_be_hoisted)
  to_bytes_le := (Impl_1.to_bytes_le_hoisted)
  from_bytes_be := (Impl_1.from_bytes_be_hoisted)
  from_bytes_le := (Impl_1.from_bytes_le_hoisted)

@[reducible] instance Impl_5.AssociatedTypes :
  math.field.traits.IsField.AssociatedTypes Degree3GoldilocksExtensionField
  where
  BaseType := (RustArray
  (math.field.element.FieldElement math.field.goldilocks.GoldilocksField)
  3)

instance Impl_5 :
  math.field.traits.IsField Degree3GoldilocksExtensionField
  where
  add := (Impl_5.add_hoisted)
  mul := (Impl_5.mul_hoisted)
  square := (Impl_5.square_hoisted)
  sub := (Impl_5.sub_hoisted)
  neg := (Impl_5.neg_hoisted)
  inv := (Impl_5.inv_hoisted)
  div := (Impl_5.div_hoisted)
  eq := (Impl_5.eq_hoisted)
  zero := (Impl_5.zero_hoisted)
  one := (Impl_5.one_hoisted)
  from_u64 := (Impl_5.from_u64_hoisted)
  from_base_type := (Impl_5.from_base_type_hoisted)
  double := (Impl_5.double_hoisted)

@[spec]
def Impl_6.div_hoisted
    (a : u64)
    (b :
    (RustArray
    (math.field.element.FieldElement math.field.goldilocks.GoldilocksField)
    3)) :
    RustM
    (core_models.result.Result
      (RustArray
      (math.field.element.FieldElement math.field.goldilocks.GoldilocksField)
      3)
      math.field.errors.FieldError)
    := do
  match
    (← (math.field.traits.IsField.inv Degree3GoldilocksExtensionField b))
  with
    | (core_models.result.Result.Ok  b_inv) => do
      (pure (core_models.result.Result.Ok
        (← (Impl_6.mul_hoisted Degree3GoldilocksExtensionField a b_inv))))
    | (core_models.result.Result.Err  err) => do
      (pure (core_models.result.Result.Err err))

@[reducible] instance Impl_6.AssociatedTypes :
  math.field.traits.IsSubFieldOf.AssociatedTypes
  math.field.goldilocks.GoldilocksField
  Degree3GoldilocksExtensionField
  where

instance Impl_6 :
  math.field.traits.IsSubFieldOf
  math.field.goldilocks.GoldilocksField
  Degree3GoldilocksExtensionField
  where
  mul := (Impl_6.mul_hoisted)
  add := (Impl_6.add_hoisted)
  div := (Impl_6.div_hoisted)
  sub := (Impl_6.sub_hoisted)
  embed := (Impl_6.embed_hoisted)
  to_subfield_vec := (Impl_6.to_subfield_vec_hoisted)

--  Create a field element from an i64.
--  Negative values are converted to their field equivalents: -x becomes p - x.
@[spec]
def Impl_7.from_i64 (value : i64) :
    RustM
    (math.field.element.FieldElement Degree3GoldilocksExtensionField)
    := do
  (core_models.convert.From._from
    (math.field.element.FieldElement Degree3GoldilocksExtensionField)
    i64 value)

@[spec]
def Impl_8.write_bytes_be_hoisted
    (self : (math.field.element.FieldElement Degree3GoldilocksExtensionField))
    (buf : (RustSlice u8)) :
    RustM (RustSlice u8) := do
  let _ ←
    if true then do
      let _ ←
        (hax_lib.assert
          (← ((← (core_models.slice.Impl.len u8 buf)) >=? (24 : usize))));
      (pure rust_primitives.hax.Tuple0.mk)
    else do
      (pure rust_primitives.hax.Tuple0.mk);
  let
    components : (RustArray
    (math.field.element.FieldElement math.field.goldilocks.GoldilocksField)
    3) ←
    (math.field.element.Impl_32.value Degree3GoldilocksExtensionField self);
  let buf : (RustSlice u8) ←
    (rust_primitives.hax.monomorphized_update_at.update_at_range
      buf
      (core_models.ops.range.Range.mk
        (start := (0 : usize))
        (_end := (8 : usize)))
      (← (math.traits.ByteConversion.write_bytes_be
        (math.field.element.FieldElement math.field.goldilocks.GoldilocksField)
        (← components[(0 : usize)]_?)
        (← buf[
          (core_models.ops.range.Range.mk
            (start := (0 : usize))
            (_end := (8 : usize)))
          ]_?))));
  let buf : (RustSlice u8) ←
    (rust_primitives.hax.monomorphized_update_at.update_at_range
      buf
      (core_models.ops.range.Range.mk
        (start := (8 : usize))
        (_end := (16 : usize)))
      (← (math.traits.ByteConversion.write_bytes_be
        (math.field.element.FieldElement math.field.goldilocks.GoldilocksField)
        (← components[(1 : usize)]_?)
        (← buf[
          (core_models.ops.range.Range.mk
            (start := (8 : usize))
            (_end := (16 : usize)))
          ]_?))));
  let buf : (RustSlice u8) ←
    (rust_primitives.hax.monomorphized_update_at.update_at_range
      buf
      (core_models.ops.range.Range.mk
        (start := (16 : usize))
        (_end := (24 : usize)))
      (← (math.traits.ByteConversion.write_bytes_be
        (math.field.element.FieldElement math.field.goldilocks.GoldilocksField)
        (← components[(2 : usize)]_?)
        (← buf[
          (core_models.ops.range.Range.mk
            (start := (16 : usize))
            (_end := (24 : usize)))
          ]_?))));
  (pure buf)

@[spec]
def Impl_8.to_bytes_be_hoisted
    (self : (math.field.element.FieldElement Degree3GoldilocksExtensionField)) :
    RustM (alloc.vec.Vec u8 alloc.alloc.Global) := do
  let byte_slice : (alloc.vec.Vec u8 alloc.alloc.Global) ←
    (math.traits.ByteConversion.to_bytes_be
      (math.field.element.FieldElement math.field.goldilocks.GoldilocksField)
      (← (← (math.field.element.Impl_32.value Degree3GoldilocksExtensionField
          self))[
        (0 : usize)
        ]_?));
  let byte_slice : (alloc.vec.Vec u8 alloc.alloc.Global) ←
    (core_models.iter.traits.collect.Extend.extend
      (alloc.vec.Vec u8 alloc.alloc.Global)
      u8 (alloc.vec.Vec u8 alloc.alloc.Global)
      byte_slice
      (← (math.traits.ByteConversion.to_bytes_be
        (math.field.element.FieldElement math.field.goldilocks.GoldilocksField)
        (← (← (math.field.element.Impl_32.value Degree3GoldilocksExtensionField
            self))[
          (1 : usize)
          ]_?))));
  let byte_slice : (alloc.vec.Vec u8 alloc.alloc.Global) ←
    (core_models.iter.traits.collect.Extend.extend
      (alloc.vec.Vec u8 alloc.alloc.Global)
      u8 (alloc.vec.Vec u8 alloc.alloc.Global)
      byte_slice
      (← (math.traits.ByteConversion.to_bytes_be
        (math.field.element.FieldElement math.field.goldilocks.GoldilocksField)
        (← (← (math.field.element.Impl_32.value Degree3GoldilocksExtensionField
            self))[
          (2 : usize)
          ]_?))));
  (pure byte_slice)

@[spec]
def Impl_8.to_bytes_le_hoisted
    (self : (math.field.element.FieldElement Degree3GoldilocksExtensionField)) :
    RustM (alloc.vec.Vec u8 alloc.alloc.Global) := do
  let byte_slice : (alloc.vec.Vec u8 alloc.alloc.Global) ←
    (math.traits.ByteConversion.to_bytes_le
      (math.field.element.FieldElement math.field.goldilocks.GoldilocksField)
      (← (← (math.field.element.Impl_32.value Degree3GoldilocksExtensionField
          self))[
        (0 : usize)
        ]_?));
  let byte_slice : (alloc.vec.Vec u8 alloc.alloc.Global) ←
    (core_models.iter.traits.collect.Extend.extend
      (alloc.vec.Vec u8 alloc.alloc.Global)
      u8 (alloc.vec.Vec u8 alloc.alloc.Global)
      byte_slice
      (← (math.traits.ByteConversion.to_bytes_le
        (math.field.element.FieldElement math.field.goldilocks.GoldilocksField)
        (← (← (math.field.element.Impl_32.value Degree3GoldilocksExtensionField
            self))[
          (1 : usize)
          ]_?))));
  let byte_slice : (alloc.vec.Vec u8 alloc.alloc.Global) ←
    (core_models.iter.traits.collect.Extend.extend
      (alloc.vec.Vec u8 alloc.alloc.Global)
      u8 (alloc.vec.Vec u8 alloc.alloc.Global)
      byte_slice
      (← (math.traits.ByteConversion.to_bytes_le
        (math.field.element.FieldElement math.field.goldilocks.GoldilocksField)
        (← (← (math.field.element.Impl_32.value Degree3GoldilocksExtensionField
            self))[
          (2 : usize)
          ]_?))));
  (pure byte_slice)

@[spec]
def Impl_8.from_bytes_be_hoisted (bytes : (RustSlice u8)) :
    RustM
    (core_models.result.Result
      (math.field.element.FieldElement Degree3GoldilocksExtensionField)
      math.errors.ByteConversionError)
    := do
  if
  (← ((← (core_models.slice.Impl.len u8 bytes))
    <? (← (Impl_8.from_bytes_be.BYTES_PER_FIELD *? (3 : usize))))) then do
    (pure (core_models.result.Result.Err
      math.errors.ByteConversionError.FromBEBytesError))
  else do
    match
      (← (math.traits.ByteConversion.from_bytes_be
        (math.field.element.FieldElement math.field.goldilocks.GoldilocksField)
        (← bytes[
          (core_models.ops.range.Range.mk
            (start := (0 : usize))
            (_end := Impl_8.from_bytes_be.BYTES_PER_FIELD))
          ]_?)))
    with
      | (core_models.result.Result.Ok  x0) => do
        match
          (← (math.traits.ByteConversion.from_bytes_be
            (math.field.element.FieldElement
              math.field.goldilocks.GoldilocksField)
            (← bytes[
              (core_models.ops.range.Range.mk
                (start := Impl_8.from_bytes_be.BYTES_PER_FIELD)
                (_end := (← (Impl_8.from_bytes_be.BYTES_PER_FIELD
                  *? (2 : usize)))))
              ]_?)))
        with
          | (core_models.result.Result.Ok  x1) => do
            match
              (← (math.traits.ByteConversion.from_bytes_be
                (math.field.element.FieldElement
                  math.field.goldilocks.GoldilocksField)
                (← bytes[
                  (core_models.ops.range.Range.mk
                    (start := (← (Impl_8.from_bytes_be.BYTES_PER_FIELD
                      *? (2 : usize))))
                    (_end := (← (Impl_8.from_bytes_be.BYTES_PER_FIELD
                      *? (3 : usize)))))
                  ]_?)))
            with
              | (core_models.result.Result.Ok  x2) => do
                (pure (core_models.result.Result.Ok
                  (← (math.field.element.Impl_32.new
                    Degree3GoldilocksExtensionField
                    (RustArray.ofVec #v[x0, x1, x2])))))
              | (core_models.result.Result.Err  err) => do
                (pure (core_models.result.Result.Err err))
          | (core_models.result.Result.Err  err) => do
            (pure (core_models.result.Result.Err err))
      | (core_models.result.Result.Err  err) => do
        (pure (core_models.result.Result.Err err))

@[spec]
def Impl_8.from_bytes_le_hoisted (bytes : (RustSlice u8)) :
    RustM
    (core_models.result.Result
      (math.field.element.FieldElement Degree3GoldilocksExtensionField)
      math.errors.ByteConversionError)
    := do
  if
  (← ((← (core_models.slice.Impl.len u8 bytes))
    <? (← (Impl_8.from_bytes_le.BYTES_PER_FIELD *? (3 : usize))))) then do
    (pure (core_models.result.Result.Err
      math.errors.ByteConversionError.FromLEBytesError))
  else do
    match
      (← (math.traits.ByteConversion.from_bytes_le
        (math.field.element.FieldElement math.field.goldilocks.GoldilocksField)
        (← bytes[
          (core_models.ops.range.Range.mk
            (start := (0 : usize))
            (_end := Impl_8.from_bytes_le.BYTES_PER_FIELD))
          ]_?)))
    with
      | (core_models.result.Result.Ok  x0) => do
        match
          (← (math.traits.ByteConversion.from_bytes_le
            (math.field.element.FieldElement
              math.field.goldilocks.GoldilocksField)
            (← bytes[
              (core_models.ops.range.Range.mk
                (start := Impl_8.from_bytes_le.BYTES_PER_FIELD)
                (_end := (← (Impl_8.from_bytes_le.BYTES_PER_FIELD
                  *? (2 : usize)))))
              ]_?)))
        with
          | (core_models.result.Result.Ok  x1) => do
            match
              (← (math.traits.ByteConversion.from_bytes_le
                (math.field.element.FieldElement
                  math.field.goldilocks.GoldilocksField)
                (← bytes[
                  (core_models.ops.range.Range.mk
                    (start := (← (Impl_8.from_bytes_le.BYTES_PER_FIELD
                      *? (2 : usize))))
                    (_end := (← (Impl_8.from_bytes_le.BYTES_PER_FIELD
                      *? (3 : usize)))))
                  ]_?)))
            with
              | (core_models.result.Result.Ok  x2) => do
                (pure (core_models.result.Result.Ok
                  (← (math.field.element.Impl_32.new
                    Degree3GoldilocksExtensionField
                    (RustArray.ofVec #v[x0, x1, x2])))))
              | (core_models.result.Result.Err  err) => do
                (pure (core_models.result.Result.Err err))
          | (core_models.result.Result.Err  err) => do
            (pure (core_models.result.Result.Err err))
      | (core_models.result.Result.Err  err) => do
        (pure (core_models.result.Result.Err err))

@[reducible] instance Impl_8.AssociatedTypes :
  math.traits.ByteConversion.AssociatedTypes
  (math.field.element.FieldElement Degree3GoldilocksExtensionField)
  where

instance Impl_8 :
  math.traits.ByteConversion
  (math.field.element.FieldElement Degree3GoldilocksExtensionField)
  where
  BYTE_LEN := (Impl_8.BYTE_LEN_hoisted)
  write_bytes_be := (Impl_8.write_bytes_be_hoisted)
  to_bytes_be := (Impl_8.to_bytes_be_hoisted)
  to_bytes_le := (Impl_8.to_bytes_le_hoisted)
  from_bytes_be := (Impl_8.from_bytes_be_hoisted)
  from_bytes_le := (Impl_8.from_bytes_le_hoisted)

@[spec]
def Impl_9.as_bytes_hoisted
    (self : (math.field.element.FieldElement Degree3GoldilocksExtensionField)) :
    RustM (alloc.vec.Vec u8 alloc.alloc.Global) := do
  (math.traits.ByteConversion.to_bytes_be
    (math.field.element.FieldElement Degree3GoldilocksExtensionField) self)

@[reducible] instance Impl_9.AssociatedTypes :
  math.traits.AsBytes.AssociatedTypes
  (math.field.element.FieldElement Degree3GoldilocksExtensionField)
  where

instance Impl_9 :
  math.traits.AsBytes
  (math.field.element.FieldElement Degree3GoldilocksExtensionField)
  where
  as_bytes := (Impl_9.as_bytes_hoisted)

@[spec]
def Impl_10.get_random_field_element_from_rng_hoisted
    (impl_rand::Rng : Type)
    [trait_constr_get_random_field_element_from_rng_hoisted_associated_type_i0 :
      rand.rng.Rng.AssociatedTypes
      impl_rand::Rng]
    [trait_constr_get_random_field_element_from_rng_hoisted_i0 : rand.rng.Rng
      impl_rand::Rng
      ]
    (rng : impl_rand::Rng) :
    RustM
    (rust_primitives.hax.Tuple2
      impl_rand::Rng
      (math.field.element.FieldElement Degree3GoldilocksExtensionField))
    := do
  let sample : (RustArray u8 8) ←
    (rust_primitives.hax.repeat (0 : u8) (8 : usize));
  let
    coeffs : (RustArray
    (math.field.element.FieldElement math.field.goldilocks.GoldilocksField)
    3) :=
    (RustArray.ofVec #v[(← (math.field.element.Impl_32.zero
                            math.field.goldilocks.GoldilocksField
                            rust_primitives.hax.Tuple0.mk)),
                          (← (math.field.element.Impl_32.zero
                            math.field.goldilocks.GoldilocksField
                            rust_primitives.hax.Tuple0.mk)),
                          (← (math.field.element.Impl_32.zero
                            math.field.goldilocks.GoldilocksField
                            rust_primitives.hax.Tuple0.mk))]);
  let _ := sorry;
  let
    hax_temp_output : (math.field.element.FieldElement
      Degree3GoldilocksExtensionField) ←
    (math.field.element.Impl_32.new Degree3GoldilocksExtensionField coeffs);
  (pure (rust_primitives.hax.Tuple2.mk rng hax_temp_output))

@[reducible] instance Impl_10.AssociatedTypes :
  math.field.traits.HasDefaultTranscript.AssociatedTypes
  Degree3GoldilocksExtensionField
  where

instance Impl_10 :
  math.field.traits.HasDefaultTranscript Degree3GoldilocksExtensionField
  where
  get_random_field_element_from_rng :=
    fun
      
      (impl_rand::Rng : Type)
      [trait_constr__associated_type_i0 : rand.rng.Rng.AssociatedTypes
        impl_rand::Rng]
      [trait_constr__i0 : rand.rng.Rng impl_rand::Rng ]
      =>
    (Impl_10.get_random_field_element_from_rng_hoisted impl_rand::Rng)

end math.field.extensions_goldilocks


namespace math.field.goldilocks

@[spec]
def Impl_3.as_bytes_hoisted
    (self : (math.field.element.FieldElement GoldilocksField)) :
    RustM (alloc.vec.Vec u8 alloc.alloc.Global) := do
  (math.traits.ByteConversion.to_bytes_be
    (math.field.element.FieldElement GoldilocksField) self)

@[reducible] instance Impl_3.AssociatedTypes :
  math.traits.AsBytes.AssociatedTypes
  (math.field.element.FieldElement GoldilocksField)
  where

instance Impl_3 :
  math.traits.AsBytes (math.field.element.FieldElement GoldilocksField)
  where
  as_bytes := (Impl_3.as_bytes_hoisted)

@[reducible] instance Impl_5.AssociatedTypes :
  math.field.traits.IsFFTField.AssociatedTypes GoldilocksField
  where

instance Impl_5 : math.field.traits.IsFFTField GoldilocksField where
  TWO_ADICITY := (Impl_5.TWO_ADICITY_hoisted)
  TWO_ADIC_PRIMITVE_ROOT_OF_UNITY :=
  (Impl_5.TWO_ADIC_PRIMITVE_ROOT_OF_UNITY_hoisted)
  field_name := (Impl_5.field_name_hoisted)

@[spec]
def Impl_6.get_random_field_element_from_rng_hoisted
    (impl_rand::Rng : Type)
    [trait_constr_get_random_field_element_from_rng_hoisted_associated_type_i0 :
      rand.rng.Rng.AssociatedTypes
      impl_rand::Rng]
    [trait_constr_get_random_field_element_from_rng_hoisted_i0 : rand.rng.Rng
      impl_rand::Rng
      ]
    (rng : impl_rand::Rng) :
    RustM
    (rust_primitives.hax.Tuple2
      impl_rand::Rng
      (math.field.element.FieldElement GoldilocksField))
    := do
  let sample : (RustArray u8 8) ←
    (rust_primitives.hax.repeat (0 : u8) (8 : usize));
  let int_sample : u64 := core_models.num.Impl_9.MAX;
  let ⟨int_sample, rng, sample⟩ ←
    (rust_primitives.hax.while_loop
      (fun ⟨int_sample, rng, sample⟩ => (do (pure true) : RustM Bool))
      (fun ⟨int_sample, rng, sample⟩ =>
        (do (int_sample >=? GOLDILOCKS_PRIME) : RustM Bool))
      (fun ⟨int_sample, rng, sample⟩ =>
        (do
        (rust_primitives.hax.int.from_machine (0 : u32)) :
        RustM hax_lib.int.Int))
      (rust_primitives.hax.Tuple3.mk int_sample rng sample)
      (fun ⟨int_sample, rng, sample⟩ =>
        (do
        let ⟨tmp0, tmp1⟩ ←
          (rand.rng.Rng.fill impl_rand::Rng (RustArray u8 8) rng sample);
        let rng : impl_rand::Rng := tmp0;
        let sample : (RustArray u8 8) := tmp1;
        let _ := rust_primitives.hax.Tuple0.mk;
        let int_sample : u64 ← (core_models.num.Impl_9.from_be_bytes sample);
        (pure (rust_primitives.hax.Tuple3.mk int_sample rng sample)) :
        RustM
        (rust_primitives.hax.Tuple3 u64 impl_rand::Rng (RustArray u8 8)))));
  let hax_temp_output : (math.field.element.FieldElement GoldilocksField) ←
    (core_models.convert.From._from
      (math.field.element.FieldElement GoldilocksField)
      u64 int_sample);
  (pure (rust_primitives.hax.Tuple2.mk rng hax_temp_output))

@[reducible] instance Impl_6.AssociatedTypes :
  math.field.traits.HasDefaultTranscript.AssociatedTypes GoldilocksField
  where

instance Impl_6 : math.field.traits.HasDefaultTranscript GoldilocksField where
  get_random_field_element_from_rng :=
    fun
      
      (impl_rand::Rng : Type)
      [trait_constr__associated_type_i0 : rand.rng.Rng.AssociatedTypes
        impl_rand::Rng]
      [trait_constr__i0 : rand.rng.Rng impl_rand::Rng ]
      =>
    (Impl_6.get_random_field_element_from_rng_hoisted impl_rand::Rng)

end math.field.goldilocks

