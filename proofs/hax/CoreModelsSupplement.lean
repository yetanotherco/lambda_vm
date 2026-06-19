-- Supplement to the Hax Lean proof-lib's `core_models`.
--
-- The hax-extracted `math.lean` references several `core::`/`std::` models that
-- the pinned Hax proof-lib (proofs/hax/.lake/packages/Hax, rev a1f1f97) does not
-- yet provide. The Hax Lean backend docs (docs/manual/lean/quick_start.md) note
-- this is expected: "the extracted code can fail to build if it uses definitions
-- from Rust `core`/`std` libraries that are missing in our Lean model."
--
-- We supply the missing ones here as `opaque` declarations, matching how the
-- proof-lib already models `leading_zeros`/`ilog2`/`mem.*` (opaque, no body).
-- These are stubs sufficient to make `math.lean` well-formed Lean; they are NOT
-- computational specs. Any later theorem that needs to reason THROUGH these
-- (e.g. the field add/mul overflow correction via `overflowing_add`) must give
-- them a real spec — opaque means they are currently assumed, not proven.
--
-- Impl_9 = u64, Impl_10 = u128, Impl_11 = usize (per core_models.lean).
import Hax

namespace core_models.hint
--  See [`core::hint::unreachable_unchecked`]
opaque unreachable_unchecked (_ : rust_primitives.hax.Tuple0) :
  RustM rust_primitives.hax.Never
--  See [`core::hint::cold_path`] (stable since Rust 1.95; pure branch hint).
opaque cold_path (_ : rust_primitives.hax.Tuple0) :
  RustM rust_primitives.hax.Tuple0
end core_models.hint

namespace core_models.num
--  overflowing_add: (sum, carry). Returned as a Tuple2 so math.lean's `⟨sum, over⟩`
--  destructuring matches. (See [`u64::overflowing_add`] etc.)
opaque Impl_9.overflowing_add (x : u64) (y : u64) :
  RustM (rust_primitives.hax.Tuple2 u64 Bool)
opaque Impl_10.overflowing_add (x : u128) (y : u128) :
  RustM (rust_primitives.hax.Tuple2 u128 Bool)
opaque Impl_11.overflowing_add (x : usize) (y : usize) :
  RustM (rust_primitives.hax.Tuple2 usize Bool)

opaque Impl_9.overflowing_sub (x : u64) (y : u64) :
  RustM (rust_primitives.hax.Tuple2 u64 Bool)
opaque Impl_10.overflowing_sub (x : u128) (y : u128) :
  RustM (rust_primitives.hax.Tuple2 u128 Bool)
opaque Impl_11.overflowing_sub (x : usize) (y : usize) :
  RustM (rust_primitives.hax.Tuple2 usize Bool)

--  See [`usize::is_multiple_of`] (Rust 1.87). Real (computable) definition so
--  the panic-freedom proof can see it always returns a value.
def Impl_11.is_multiple_of (x : usize) (n : usize) : RustM Bool :=
  pure (decide (x.toNat % n.toNat = 0))

--  byte (de)serialization (See [`u64::from_le_bytes`] / [`from_be_bytes`] / [`to_*_bytes`])
opaque Impl_9.from_le_bytes (x : RustArray u8 8) : RustM u64
opaque Impl_9.from_be_bytes (x : RustArray u8 8) : RustM u64
opaque Impl_9.to_le_bytes (x : u64) : RustM (RustArray u8 8)
opaque Impl_9.to_be_bytes (x : u64) : RustM (RustArray u8 8)

--  bit ops (See [`u64::trailing_zeros`], [`usize::trailing_zeros`],
--  [`usize::reverse_bits`], [`usize::BITS`])
opaque Impl_9.trailing_zeros (x : u64) : RustM u32
opaque Impl_11.trailing_zeros (x : usize) : RustM u32
opaque Impl_11.reverse_bits (x : usize) : RustM usize
opaque Impl_11.BITS : RustM usize

--  hex parsing. `Impl_40` is hax's index for u64 in math.lean's from_hex path.
--  (See [`u64::from_str_radix`].)
opaque Impl_40.from_str_radix (src : String) (radix : u32) :
  RustM (core_models.result.Result u64 core_models.num.error.ParseIntError)
end core_models.num

namespace core_models.str
--  See [`str::strip_prefix`]. Off the verifier path (hex parsing); opaque.
opaque Impl.strip_prefix (P : Type) (s : String) (prefix_ : P) :
  RustM (core_models.option.Option String)
end core_models.str

namespace rust_primitives.hax.monomorphized_update_at
--  Slice update over a RangeTo index. The proof-lib provides `update_at_usize`
--  but not the range-to variant math.lean uses (ByteConversion::write_bytes_be).
opaque update_at_range_to (α : Type) (s : RustSlice α)
  (r : core_models.ops.range.RangeTo usize) (v : RustSlice α) :
  RustM (RustSlice α)
end rust_primitives.hax.monomorphized_update_at

namespace core_models.slice
--  See [`<[T]>::swap`]. Returns the mutated slice (functional form).
opaque Impl.swap (T : Type) (s : RustSlice T) (i : usize) (j : usize) :
  RustM (RustSlice T)
end core_models.slice

-- `Vec::with_capacity` and `Vec::push` — missing from the proof-lib's
-- alloc.lean (which has `new`/`len`/`extend_from_slice`). Vec α = Seq α; these
-- are computable (real) so loops building vectors are provable. Capacity is a
-- hint only, so `with_capacity` is an empty Vec.
open rust_primitives.sequence in
@[spec]
def alloc.vec.Impl.with_capacity (α : Type) (_capacity : usize) :
    RustM (alloc.vec.Vec α alloc.alloc.Global) :=
  pure ⟨(List.nil).toArray, by grind⟩

open rust_primitives.sequence in
@[spec]
def alloc.vec.Impl_1.push (α : Type) (_Allocator : Type)
    (x : alloc.vec.Vec α alloc.alloc.Global) (v : α) :
    RustM (alloc.vec.Vec α alloc.alloc.Global) :=
  if h : x.val.size + 1 < USize64.size then
    pure ⟨x.val.push v, by simp [h]⟩
  else
    .fail .maximumSizeExceeded
