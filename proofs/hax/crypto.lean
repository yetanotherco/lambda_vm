
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


namespace crypto.merkle_tree.proof

--  Stores a merkle path to some leaf.
--  Internally, the necessary hashes are stored from root to leaf in the
--  `merkle_path` field, in such a way that, if the merkle tree is of height `n`, the
--  `i`-th element of `merkle_path` is the sibling node in the `n - 1 - i`-th check
--  when verifying.
structure Proof
  (T : Type)
  [trait_constr_Proof_associated_type_i0 :
    core_models.cmp.PartialEq.AssociatedTypes
    T
    T]
  [trait_constr_Proof_i0 : core_models.cmp.PartialEq T T ]
  [trait_constr_Proof_associated_type_i1 : core_models.cmp.Eq.AssociatedTypes T]
  [trait_constr_Proof_i1 : core_models.cmp.Eq T ]
  where
  merkle_path : (alloc.vec.Vec T alloc.alloc.Global)

end crypto.merkle_tree.proof


namespace crypto.merkle_tree.traits

--  A backend for Merkle trees. This defines raw `Data` from which the Merkle
--  tree is built from. It also defines the `Node` type and the hash function
--  used to build parent nodes from children nodes.
class IsMerkleTreeBackend.AssociatedTypes (Self : Type) where
  Node : Type
  Data : Type

attribute [reducible] IsMerkleTreeBackend.AssociatedTypes.Node

attribute [reducible] IsMerkleTreeBackend.AssociatedTypes.Data

abbrev IsMerkleTreeBackend.Node :=
  IsMerkleTreeBackend.AssociatedTypes.Node

abbrev IsMerkleTreeBackend.Data :=
  IsMerkleTreeBackend.AssociatedTypes.Data

class IsMerkleTreeBackend (Self : Type)
  [associatedTypes : outParam (IsMerkleTreeBackend.AssociatedTypes (Self :
      Type))]
  where
  [trait_constr_Node_associated_type_i1 :
    core_models.cmp.PartialEq.AssociatedTypes
    associatedTypes.Node
    associatedTypes.Node]
  [trait_constr_Node_i1 : core_models.cmp.PartialEq
    associatedTypes.Node
    associatedTypes.Node
    ]
  [trait_constr_Node_associated_type_i2 : core_models.cmp.Eq.AssociatedTypes
    associatedTypes.Node]
  [trait_constr_Node_i2 : core_models.cmp.Eq associatedTypes.Node ]
  [trait_constr_Node_associated_type_i3 :
    core_models.clone.Clone.AssociatedTypes
    associatedTypes.Node]
  [trait_constr_Node_i3 : core_models.clone.Clone associatedTypes.Node ]
  [trait_constr_Node_associated_type_i4 :
    core_models.marker.Sync.AssociatedTypes
    associatedTypes.Node]
  [trait_constr_Node_i4 : core_models.marker.Sync associatedTypes.Node ]
  [trait_constr_Node_associated_type_i5 :
    core_models.marker.Send.AssociatedTypes
    associatedTypes.Node]
  [trait_constr_Node_i5 : core_models.marker.Send associatedTypes.Node ]
  [trait_constr_Data_associated_type_i1 :
    core_models.marker.Sync.AssociatedTypes
    associatedTypes.Data]
  [trait_constr_Data_i1 : core_models.marker.Sync associatedTypes.Data ]
  [trait_constr_Data_associated_type_i2 :
    core_models.marker.Send.AssociatedTypes
    associatedTypes.Data]
  [trait_constr_Data_i2 : core_models.marker.Send associatedTypes.Data ]
  hash_data (Self) : (associatedTypes.Data -> RustM associatedTypes.Node)
  hash_leaves (Self) :
    ((RustSlice associatedTypes.Data) ->
    RustM (alloc.vec.Vec associatedTypes.Node alloc.alloc.Global))
  hash_new_parent (Self) :
    (associatedTypes.Node -> associatedTypes.Node -> RustM associatedTypes.Node)

end crypto.merkle_tree.traits


namespace crypto.merkle_tree.proof

--  Verifies a Merkle inclusion proof for the value contained at leaf index.
def Impl_2.verify
    (T : Type)
    (B : Type)
    [trait_constr_verify_associated_type_i0 :
      core_models.cmp.PartialEq.AssociatedTypes
      T
      T]
    [trait_constr_verify_i0 : core_models.cmp.PartialEq T T ]
    [trait_constr_verify_associated_type_i1 : core_models.cmp.Eq.AssociatedTypes
      T]
    [trait_constr_verify_i1 : core_models.cmp.Eq T ]
    [trait_constr_verify_associated_type_i2 :
      crypto.merkle_tree.traits.IsMerkleTreeBackend.AssociatedTypes
      B]
    [trait_constr_verify_i2 : crypto.merkle_tree.traits.IsMerkleTreeBackend
      B
      (associatedTypes := {
        show crypto.merkle_tree.traits.IsMerkleTreeBackend.AssociatedTypes B
        by infer_instance
        with Node := T})]
    (self : (Proof T))
    (root_hash : T)
    (index : usize)
    (value : (crypto.merkle_tree.traits.IsMerkleTreeBackend.Data B)) :
    RustM Bool := do
  let hashed_value : T ←
    (crypto.merkle_tree.traits.IsMerkleTreeBackend.hash_data B value);
  let ⟨hashed_value, index⟩ ←
    (rust_primitives.hax.folds.fold_range
      (0 : usize)
      (← (alloc.vec.Impl_1.len T alloc.alloc.Global (Proof.merkle_path self)))
      (fun ⟨hashed_value, index⟩ _ => (do (pure true) : RustM Bool))
      (rust_primitives.hax.Tuple2.mk hashed_value index)
      (fun ⟨hashed_value, index⟩ i =>
        (do
        let sibling_node : T ← (Proof.merkle_path self)[i]_?;
        let hashed_value : T ←
          if
          (← (core_models.num.Impl_11.is_multiple_of index (2 : usize))) then do
            let hashed_value : T ←
              (crypto.merkle_tree.traits.IsMerkleTreeBackend.hash_new_parent
                B hashed_value sibling_node);
            (pure hashed_value)
          else do
            let hashed_value : T ←
              (crypto.merkle_tree.traits.IsMerkleTreeBackend.hash_new_parent
                B sibling_node hashed_value);
            (pure hashed_value);
        let index : usize ← (index >>>? (1 : i32));
        (pure (rust_primitives.hax.Tuple2.mk hashed_value index)) :
        RustM (rust_primitives.hax.Tuple2 T usize))));
  (core_models.cmp.PartialEq.eq T T root_hash hashed_value)

set_option hax_mvcgen.specset "int" in
@[hax_spec]
def
      Impl_2.verify.spec
      (T : Type)
      (B : Type)
      [trait_constr_verify_associated_type_i0 :
        core_models.cmp.PartialEq.AssociatedTypes
        T
        T]
      [trait_constr_verify_i0 : core_models.cmp.PartialEq T T ]
      [trait_constr_verify_associated_type_i1 :
        core_models.cmp.Eq.AssociatedTypes
        T]
      [trait_constr_verify_i1 : core_models.cmp.Eq T ]
      [trait_constr_verify_associated_type_i2 :
        crypto.merkle_tree.traits.IsMerkleTreeBackend.AssociatedTypes
        B]
      [trait_constr_verify_i2 : crypto.merkle_tree.traits.IsMerkleTreeBackend
        B
        (associatedTypes := {
          show crypto.merkle_tree.traits.IsMerkleTreeBackend.AssociatedTypes B
          by infer_instance
          with Node := T})]
      (self : (Proof T))
      (root_hash : T)
      (index : usize)
      (value : (crypto.merkle_tree.traits.IsMerkleTreeBackend.Data B)) :
    Spec
      (requires := do (pure true))
      (ensures := fun _res => do (pure true))
      (Impl_2.verify
        (T : Type)
        (B : Type)
        (self : (Proof T))
        (root_hash : T)
        (index : usize)
        (value : (crypto.merkle_tree.traits.IsMerkleTreeBackend.Data B))) := {
  pureRequires := by hax_construct_pure <;> grind
  pureEnsures := by hax_construct_pure <;> grind
  contract := by hax_mvcgen [Impl_2.verify] <;> grind
}

end crypto.merkle_tree.proof

