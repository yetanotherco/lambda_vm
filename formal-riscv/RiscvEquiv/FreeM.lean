import Sail

/-!
# The monad laws for Sail's `FreeM`

The Sail model runs in `SailM`, the free monad `FreeM` of `Sail.ArchSem`: an instruction is a
tree of effects, not a state transformer. Unlike a state monad, reassociating a `do` block is not
definitional there, since `bind` recurses on the tree and `rX_bits r` is a stuck term. lean-sail
gives `FreeM` a `Monad` instance but no `LawfulMonad` one; this file proves the laws, so the
`Sail.lean` files can flatten Sail's nested `do` blocks with `bind_assoc` and `pure_bind`.
-/

namespace RiscvEquiv

open Sail.ArchSem

universe u v w

variable {Eff : Type v} [Effect.{u, v} Eff]

theorem FreeM.bind_assoc {α β γ : Type w} (x : FreeM Eff α) (f : α → FreeM Eff β)
    (g : β → FreeM Eff γ) : (x >>= f) >>= g = x >>= fun a => f a >>= g := by
  induction x with
  | pure a => rfl
  | impure op cont ih =>
    show FreeM.impure op _ = FreeM.impure op _
    congr
    funext r
    exact ih r

theorem FreeM.id_map {α : Type w} (x : FreeM Eff α) : id <$> x = x := by
  induction x with
  | pure a => rfl
  | impure op cont ih =>
    show FreeM.impure op _ = FreeM.impure op _
    congr
    funext r
    exact ih r

instance : LawfulMonad (FreeM Eff) :=
  LawfulMonad.mk' _ FreeM.id_map (fun _ _ => rfl) FreeM.bind_assoc

end RiscvEquiv
