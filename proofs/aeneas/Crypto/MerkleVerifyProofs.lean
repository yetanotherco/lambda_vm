-- Panic-freedom proofs for the carved single-leaf Merkle `Proof::verify`
-- (see Crypto/MerkleVerify.lean).
--
-- "Panic-free" in the aeneas `Result` model = the function evaluates to `.ok _`,
-- never `.fail _` (panic / out-of-bounds / overflow) nor `.div`
-- (non-termination). The spec notation `f ⦃ _ => True ⦄` says exactly this:
-- `spec` is `False` on `.fail`/`.div`, so proving `f ⦃ _ => True ⦄` IS proving
-- panic-(and-divergence-)freedom.
--
-- SCOPE / what is and isn't provable here:
--   * `verify` is generic over an arbitrary `IsMerkleTreeBackend`, whose
--     `hash_data`/`hash_new_parent` return `Result` and MAY fail for an
--     adversarial backend. So UNCONDITIONAL panic-freedom of `verify` is FALSE.
--     Provable: CONDITIONAL panic-freedom, given the backend's hash ops total.
--   * The `merkle_path[i]` index is in bounds because `i` ranges over
--     `0..merkle_path.len()`; discharging it needs the loop invariant.
--   * `verify_loop` uses aeneas's `loop` fixed-point; proving it `.ok` needs a
--     decreasing-measure + invariant (`loop.spec_decr_nat`). The range iterator
--     `IteratorRange.next` has no registered `@[progress]` spec, so `step*` can
--     not drive the loop automatically — this is the open obligation below.
import Aeneas
import Crypto.MerkleVerify
open Aeneas Aeneas.Std Result

namespace MerkleVerifyProofs

open merkle_tree.proof merkle_tree.traits

/-- `@[progress]` spec for the usize range iterator's `next` (the aeneas Std
proof-lib ships none, which is what blocks `step*` on range loops). It always
succeeds (`.ok`); on a non-empty range it yields the current `start` and advances
`start` by one, otherwise `none`. With this registered, `step` can drive any
`for _ in a..b` loop body. -/
@[progress]
theorem iter_range_next_spec (r : core.ops.range.Range Std.Usize) :
    core.iter.range.IteratorRange.next core.iter.range.StepUsize r ⦃ res =>
      (r.start.val < r.end.val →
        ∃ r', res = (some r.start, r') ∧ r'.start.val = r.start.val + 1 ∧ r'.«end» = r.«end») ∧
      (¬ r.start.val < r.end.val → res = (none, r)) ⦄ := by
  unfold core.iter.range.IteratorRange.next
  have hadd := Usize.checked_add_bv_spec r.start 1#usize
  simp only [core.iter.range.StepUsize.forward_checked,
    core.cmp.impls.PartialOrdUsize.lt, core.clone.impls.CloneUsize.clone,
    liftFun2, liftFun1]
  by_cases hlt : r.start.val < r.end.val
  · -- non-empty: checked_add succeeds (no overflow), advances start by 1.
    have hmax : r.end.val ≤ Usize.max := by scalar_tac
    cases hc : r.start.checked_add 1#usize with
    | none => rw [hc] at hadd; simp at hadd; scalar_tac
    | some n =>
      rw [hc] at hadd
      obtain ⟨_, hnval, _⟩ := hadd
      simp [hlt, hc, WP.spec_ok]
      scalar_tac
  · -- empty: yields none.
    simp [hlt, WP.spec_ok]

/-- Totality hypotheses on a backend: its hash operations never fail. This is
the honest formal counterpart of "assume the hash function is total" — the
weakest assumption under which `verify` can be panic-free. -/
structure HashTotal {B T D : Type} (inst : IsMerkleTreeBackend B T D) : Prop where
  hash_data_ok   : ∀ d, ∃ n, inst.hash_data d = .ok n
  hash_parent_ok : ∀ a b, ∃ n, inst.hash_new_parent a b = .ok n

/-- Panic-freedom of the verification loop, given total backend hashes.

`loop.spec_decr_nat` with:
  * measure = remaining range length (`end - start`), strictly decreasing each
    step because `next` advances `start` by 1 when `start < end`;
  * invariant = `start ≤ end ≤ v.len`, which keeps the index `i = start`
    in bounds so `Vec.index` (the `merkle_path[i]` lookup) succeeds. -/
theorem verify_loop_ok {T B D : Type}
    (inst : IsMerkleTreeBackend B T D) (hT : HashTotal inst)
    (v : alloc.vec.Vec T) (iter : core.ops.range.Range Std.Usize)
    (index : Std.Usize) (hashed_value : T)
    (hi : iter.start.val ≤ iter.«end».val ∧ iter.«end».val ≤ v.length) :
    Proof.verify_loop inst iter v index hashed_value ⦃ _ => True ⦄ := by
  unfold Proof.verify_loop
  apply loop.spec_decr_nat
    (measure := fun (s : core.ops.range.Range Std.Usize × Std.Usize × T) =>
      s.1.«end».val - s.1.start.val)
    (inv := fun (s : core.ops.range.Range Std.Usize × Std.Usize × T) =>
      s.1.start.val ≤ s.1.«end».val ∧ s.1.«end».val ≤ v.length)
  · rintro ⟨r, idx, hv⟩ ⟨hse, hev⟩
    simp only at hse hev ⊢
    unfold Proof.verify_loop.body
    simp only [alloc.vec.Vec.index_slice_index]
    -- `next` is driven by the @[progress] `iter_range_next_spec`; its precondition
    -- (r.start < max when the range is non-empty) holds since r.end ≤ len ≤ max.
    step as ⟨ o, iter1, hne, hemp ⟩
    -- case on the iterator output
    by_cases hlt : r.start.val < r.end.val
    · obtain ⟨r', ho, hr'start, hr'end⟩ := hne hlt
      have hidx : r.start.val < v.length := by scalar_tac
      injection ho with ho_o ho_it; subst ho_o
      -- `iter1 = r'`; keep the facts about it by rewriting iter1 := r'.
      subst ho_it
      simp only
      step as ⟨ sib, _ ⟩            -- Vec.index_usize, succeeds by hidx
      obtain ⟨p, hp⟩ := hT.hash_parent_ok hv sib
      obtain ⟨p', hp'⟩ := hT.hash_parent_ok sib hv
      -- is_multiple_of is our total computable model (reduces to `ok _`); then the
      -- branch hash call is total by hT, and the shift is total (1 < 64).
      simp only [core.num.Usize.is_multiple_of, bind_tc_ok, hp, hp']
      -- now: `if (decide ..) then ok p else ok p'` >>= shift >>= ok cont
      split <;>
        (step as ⟨ idx', _ ⟩
         case _ =>             -- 1 < numBits (32 or 64)
           have h1 : (1#i32).val = 1 := by decide
           have := System.Platform.numBits_eq
           omega
         simp only [hr'start, hr'end]
         scalar_tac)
    · obtain ⟨ho_o, ho_it⟩ := Prod.mk.inj (hemp hlt)
      subst ho_o; subst ho_it; simp
  · exact hi

/-- Conditional panic-freedom of `verify`, FACTORED through the loop: if the
backend's `hash_data` is total and the verification loop is itself panic-free on
the relevant arguments, then `verify` as a whole is panic-free.

This isolates exactly the two open obligations (total `hash_data`, panic-free
loop) from the rest of `verify`, which is a straight-line `hash_data` → loop →
`eq` and has no other fallible operations. The `eq` (`PartialEq::eq`) is total
(returns `Bool`, no `Result` failure). -/
theorem verify_ok_of_loop_ok {T B D : Type}
    (inst : IsMerkleTreeBackend B T D)
    (corecmpPartialEqInst : core.cmp.PartialEq T T)
    (corecmpEqInst : core.cmp.Eq T)
    (self : Proof T) (root_hash : T) (index : Std.Usize) (value : D)
    -- `hash_data` succeeds on `value`:
    (hData : ∃ n, inst.hash_data value = .ok n)
    -- the loop is panic-free for every starting hashed value (equational form):
    (hLoop : ∀ hv, ∃ r,
      Proof.verify_loop inst { start := 0#usize, «end» := self.merkle_path.len }
        self.merkle_path index hv = .ok r)
    -- the node equality used for the final root check is total:
    (hEq : ∀ a b, ∃ c, corecmpPartialEqInst.eq a b = .ok c) :
    Proof.verify corecmpPartialEqInst corecmpEqInst inst self root_hash index value
      ⦃ _ => True ⦄ := by
  -- `spec _ (fun _ => True)` is `True` whenever the computation is `.ok`, so it
  -- suffices to rewrite the three `Result`-producing subterms to `.ok`.
  unfold Proof.verify
  obtain ⟨n, hn⟩ := hData
  obtain ⟨r, hr⟩ := hLoop n
  obtain ⟨c, hc⟩ := hEq root_hash r
  simp [hn, hr, hc]

/-- A `f ⦃ _ => True ⦄` panic-freedom fact (`spec` form) gives the equational
`∃ r, f = .ok r` form, by case analysis on the `Result`. -/
theorem ok_of_spec_true {α} {f : Result α} (h : f ⦃ _ => True ⦄) :
    ∃ r, f = .ok r := by
  cases f with
  | ok r => exact ⟨r, rfl⟩
  | fail e => simp only [WP.spec, WP.theta] at h
  | div => simp only [WP.spec, WP.theta] at h

/-- **Conditional panic-freedom of `Proof::verify`** (no remaining loop
hypothesis): given a backend whose `hash_data`/`hash_new_parent`/node-`eq` are
total, and a starting `index`, `verify` evaluates to `.ok` — it never panics,
overflows, indexes out of bounds, or diverges. The `merkle_path[i]` lookups are
all in bounds (the loop ranges over `0..merkle_path.len`); this is discharged by
`verify_loop_ok`. UNCONDITIONAL panic-freedom is false: an adversarial backend's
hash/eq could fail — which is exactly what the totality hypotheses rule out. -/
theorem verify_ok {T B D : Type}
    (inst : IsMerkleTreeBackend B T D) (hT : HashTotal inst)
    (corecmpPartialEqInst : core.cmp.PartialEq T T)
    (corecmpEqInst : core.cmp.Eq T)
    (self : Proof T) (root_hash : T) (index : Std.Usize) (value : D)
    (hEq : ∀ a b, ∃ c, corecmpPartialEqInst.eq a b = .ok c) :
    Proof.verify corecmpPartialEqInst corecmpEqInst inst self root_hash index value
      ⦃ _ => True ⦄ := by
  apply verify_ok_of_loop_ok inst corecmpPartialEqInst corecmpEqInst self root_hash index value
    (hT.hash_data_ok value)
  · -- the loop is panic-free: `verify_loop_ok` with the initial range `0..len`,
    -- whose invariant `0 ≤ len ≤ len` holds trivially.
    intro hv
    apply ok_of_spec_true
    apply verify_loop_ok inst hT
    exact ⟨by scalar_tac, by scalar_tac⟩
  · exact hEq

/-! ## Index-algebra utilities

The non-looping node-index helpers (`sibling_index`, `parent_index`,
`get_sibling_pos`, `get_parent_pos`, `is_power_of_two`) carved alongside
`verify`. We prove, for each, exactly the precondition under which it is
panic-free, AND — where it matters for the completeness proof — the closed-form
value it returns. The arithmetic is `usize`, so subtraction underflows at 0 and
addition overflows at `Usize.max`; the preconditions are precisely the bounds
that avoid those.

`Usize` indices use a binary-heap layout: node `n`'s children are `2n+1`, `2n+2`
and its parent is `(n-1)/2` (which equals `parent_index n` for both parities,
since for odd `n = 2k+1` we have `n/2 = k = (n-1)/2`). The sibling of `n` is
`n-1` if `n` is even (right child, sibling is the left), `n+1` if odd (left
child, sibling is the right). These closed forms are what the path traversal in
`verify`/`build_merkle_path` walks. -/

open merkle_tree.utils

/-- `is_power_of_two` is panic-free iff `x ≥ 1`: it computes `x - 1` (usize), which
underflows at `x = 0`. The bitwise `&&&` is total. -/
theorem is_power_of_two_ok (x : Std.Usize) (hx : 1 ≤ x.val) :
    is_power_of_two x ⦃ _ => True ⦄ := by
  unfold is_power_of_two
  step as ⟨ i, _ ⟩            -- x - 1, succeeds since x ≥ 1
  step as ⟨ i1, _ ⟩          -- lift (x &&& i), total

/-- Closed form + panic-freedom of `parent_index`, given `n ≥ 1`. The only
fallible op is the `n - 1` in the even branch (underflows at `0`); division by 2
never fails. The returned value is `(n-1)/2` for BOTH parities (the binary-heap
parent of `n`): for odd `n = 2k+1`, `n/2 = k = (n-1)/2`. -/
@[progress]
theorem parent_index_ok (n : Std.Usize) (hn : 1 ≤ n.val) :
    parent_index n ⦃ r => r.val = (n.val - 1) / 2 ⦄ := by
  unfold parent_index
  simp only [core.num.Usize.is_multiple_of, bind_tc_ok]
  split
  · -- even: (n - 1) / 2
    step as ⟨ i, hi ⟩          -- n - 1
    step as ⟨ q, hq ⟩          -- i / 2
    scalar_tac
  · -- odd: n / 2, and for odd n, n / 2 = (n - 1) / 2
    rename_i hb
    step as ⟨ q, hq ⟩
    -- n odd ⇒ n % 2 = 1 ⇒ n / 2 = (n - 1) / 2
    simp only [beq_iff_eq] at hb
    have h2 : (2#usize).val = 2 := by decide
    rw [h2] at hb
    have : n.val % 2 = 1 := by omega
    omega

/-- Closed form + panic-freedom of `sibling_index`, given `1 ≤ n < Usize.max`.
Even `n` → `n - 1` (needs `n ≥ 1`); odd `n` → `n + 1` (needs `n < max`). -/
@[progress]
theorem sibling_index_ok (n : Std.Usize) (hlo : 1 ≤ n.val) (hhi : n.val < Std.Usize.max) :
    sibling_index n ⦃ r =>
      r.val = (if n.val % 2 = 0 then n.val - 1 else n.val + 1) ⦄ := by
  have h2 : (2#usize).val = 2 := by decide
  unfold sibling_index
  simp only [core.num.Usize.is_multiple_of, bind_tc_ok]
  split
  · rename_i hb
    simp only [beq_iff_eq] at hb; rw [h2] at hb
    step as ⟨ i, hi ⟩          -- n - 1
    rw [if_pos hb]; scalar_tac
  · rename_i hb
    simp only [beq_iff_eq] at hb; rw [h2] at hb
    step as ⟨ i, hi ⟩          -- n + 1
    rw [if_neg hb]; scalar_tac

/-- `get_parent_pos` is TOTAL: the `node_index = 0` guard returns early before any
arithmetic, so the `n - 1` in the even branch only runs when `n ≥ 1`. -/
theorem get_parent_pos_ok (n : Std.Usize) :
    get_parent_pos n ⦃ _ => True ⦄ := by
  unfold get_parent_pos
  split
  · simp
  · rename_i hne
    simp only [core.num.Usize.is_multiple_of, bind_tc_ok]
    have hn : 1 ≤ n.val := by scalar_tac
    split
    · step as ⟨ i, _ ⟩; step as ⟨ q, _ ⟩
    · step as ⟨ q, _ ⟩

/-- `get_sibling_pos` is panic-free iff `n < Usize.max`: the `n = 0` guard handles
the even-branch underflow, but the odd branch does `n + 1`, which overflows at
`n = max`. Returns `none` for the root (`n = 0`), `some` otherwise. -/
theorem get_sibling_pos_ok (n : Std.Usize) (hhi : n.val < Std.Usize.max) :
    get_sibling_pos n ⦃ _ => True ⦄ := by
  unfold get_sibling_pos
  split
  · simp
  · rename_i hne
    simp only [core.num.Usize.is_multiple_of, bind_tc_ok]
    have hn : 1 ≤ n.val := by scalar_tac
    split
    · step as ⟨ i, _ ⟩        -- n - 1, n ≥ 1
    · step as ⟨ i, _ ⟩        -- n + 1, n < max

/-! ## Completeness, Phase A: the verifier fold reconstructs the root

`verify` folds the Merkle path: starting from `hash_data value`, for each sibling
`s` along the path it forms `hash_new_parent acc s` (when the running index is
even — the running node is a left child, sibling on the right) or
`hash_new_parent s acc` (odd — running node is a right child, sibling on the
left), then halves the index, and finally compares the result with `root_hash`.

`foldPath` below is the pure (monadic) specification of exactly that fold over a
list of siblings. `IsMerklePath root index value path` says the path hashes up to
`root`. The theorem `verify_path_complete` then states: if a path satisfies that
spec, `verify` returns `.ok true` — i.e. an honest proof verifies. This is the
verifier-correctness half of completeness; it makes NO reference to how the path
was built (that is Phase B: showing `build`/`get_proof_by_pos` produce a
spec-correct path). It needs no totality/injectivity hypotheses — the spec
already supplies the successful hashes. -/

open merkle_tree.traits

/-- Pure monadic specification of the Merkle-path fold that `verify_loop` runs.
`idx` is the running node index (Nat); `acc` the running hash. -/
def foldPath {B T D : Type} (inst : IsMerkleTreeBackend B T D) :
    T → Nat → List T → Result T
  | acc, _,   []        => ok acc
  | acc, idx, s :: rest => do
      let acc' ← (if idx % 2 = 0 then inst.hash_new_parent acc s
                  else inst.hash_new_parent s acc)
      foldPath inst acc' (idx / 2) rest

/-- `verify_loop` over `v` with range `[k, len)`, starting index `index` and
running hash `acc`, agrees with `foldPath` over the remaining siblings
`v.val.drop k`: if the fold from here lands (successfully) on `root`, so does the
loop.

This is the loop⇄fold characterization. Driven by `loop.spec_decr_nat`: the
invariant carries "the fold of the as-yet-unconsumed suffix lands on `root`",
which each step peels by one element (matching the body's parent-hash to the
`foldPath` step), with the index halving in lockstep; the measure `end - start`
decreases. No totality hypotheses: the `foldPath = ok root` premise already
witnesses every hash succeeding. -/
theorem verify_loop_eq_foldPath {T B D : Type}
    (inst : IsMerkleTreeBackend B T D)
    (v : alloc.vec.Vec T) (k : Std.Usize) (index : Std.Usize) (acc root : T)
    (hk : k.val ≤ v.length)
    (hfold : foldPath inst acc index.val (v.val.drop k.val) = .ok root) :
    Proof.verify_loop inst { start := k, «end» := v.len } v index acc
      ⦃ r => r = root ⦄ := by
  unfold Proof.verify_loop
  apply loop.spec_decr_nat
    (measure := fun (s : core.ops.range.Range Std.Usize × Std.Usize × T) =>
      s.1.«end».val - s.1.start.val)
    (inv := fun (s : core.ops.range.Range Std.Usize × Std.Usize × T) =>
      s.1.«end».val = v.length ∧ s.1.start.val ≤ v.length ∧
      foldPath inst s.2.2 s.2.1.val (v.val.drop s.1.start.val) = .ok root)
  · rintro ⟨r, idx, hv⟩ ⟨hend, hsv, hfd⟩
    simp only at hend hsv hfd ⊢
    unfold Proof.verify_loop.body
    simp only [alloc.vec.Vec.index_slice_index]
    step as ⟨ o, iter1, hne, hemp ⟩
    by_cases hlt : r.start.val < r.end.val
    · -- consume v[start]: the fold's head element
      obtain ⟨r', ho, hr'start, hr'end⟩ := hne hlt
      injection ho with ho_o ho_it; subst ho_o; subst ho_it
      have hstart_lt : r.start.val < v.val.length := by
        have : v.length = v.val.length := rfl; omega
      -- the remaining suffix is non-empty; expose its head = v[start]
      rw [List.drop_eq_getElem_cons hstart_lt] at hfd
      simp only [foldPath] at hfd
      simp only
      step as ⟨ sib, hsib ⟩            -- Vec.index_usize = v.val[start]
      -- align the fold's head element with the body's read `sib`
      rw [← hsib] at hfd
      simp only [core.num.Usize.is_multiple_of, bind_tc_ok]
      -- index parity: `is_multiple_of idx 2` ↔ `idx.val % 2 = 0`, and the body
      -- shifts `idx >>> 1 = idx.val / 2`, matching `foldPath`'s `idx / 2`.
      have h2 : (2#usize).val = 2 := by decide
      split
      · rename_i hb
        simp only [beq_iff_eq] at hb; rw [h2] at hb
        rw [if_pos hb] at hfd
        -- body computes hash_new_parent acc sib (idx even); fold's head did too
        obtain ⟨p, hp, hrest⟩ : ∃ p, inst.hash_new_parent hv sib = .ok p ∧
            foldPath inst p (idx.val / 2) (v.val.drop (r.start.val + 1)) = .ok root := by
          revert hfd; cases hh : inst.hash_new_parent hv sib <;> simp_all
        rw [hp]; simp only [bind_tc_ok]
        step as ⟨ idx', hidx' ⟩        -- idx >>> 1
        case _ =>                       -- shift precondition: 1 < numBits
          have h1 : (1#i32).val = 1 := by decide
          have := System.Platform.numBits_eq; omega
        refine ⟨by rw [hr'end]; exact hend, by rw [hr'start]; omega, ?_, by
          rw [hr'start, hr'end]; omega⟩
        -- idx' = idx / 2 (shift), iter1.start = r.start+1: matches the fold tail
        have hidxv : idx'.val = idx.val / 2 := by rw [hidx']; simpa using Nat.shiftRight_one _
        rw [hidxv, hr'start]; exact hrest
      · rename_i hb
        simp only [beq_iff_eq] at hb; rw [h2] at hb
        rw [if_neg hb] at hfd
        obtain ⟨p, hp, hrest⟩ : ∃ p, inst.hash_new_parent sib hv = .ok p ∧
            foldPath inst p (idx.val / 2) (v.val.drop (r.start.val + 1)) = .ok root := by
          revert hfd; cases hh : inst.hash_new_parent sib hv <;> simp_all
        rw [hp]; simp only [bind_tc_ok]
        step as ⟨ idx', hidx' ⟩
        case _ =>
          have h1 : (1#i32).val = 1 := by decide
          have := System.Platform.numBits_eq; omega
        refine ⟨by rw [hr'end]; exact hend, by rw [hr'start]; omega, ?_, by
          rw [hr'start, hr'end]; omega⟩
        have hidxv : idx'.val = idx.val / 2 := by rw [hidx']; simpa using Nat.shiftRight_one _
        rw [hidxv, hr'start]; exact hrest
    · -- empty suffix: r.start ≥ r.end = len, so the drop is empty; fold = ok acc
      have hdrop_nil : v.val.drop r.start.val = [] := by
        apply List.drop_eq_nil_of_le
        have hlen : v.length = v.val.length := rfl; omega
      rw [hdrop_nil] at hfd; simp only [foldPath] at hfd
      -- the iterator returned `none`, so the body breaks with `done hv`
      obtain ⟨ho_o, ho_it⟩ := Prod.mk.inj (hemp hlt)
      subst ho_o
      simp only [Result.ok.injEq] at hfd
      simpa using hfd
  · exact ⟨rfl, hk, by simpa using hfold⟩

/-- An honest Merkle path for `(root, index, value)`: `value` hashes to a leaf,
and folding that leaf up the path (per `foldPath`) lands exactly on `root`. This
is the verifier's success condition expressed purely on the trait's hash ops —
no reference to tree construction (that link is Phase B). -/
def IsMerklePath {B T D : Type} (inst : IsMerkleTreeBackend B T D)
    (root : T) (index : Std.Usize) (value : D) (path : alloc.vec.Vec T) : Prop :=
  ∃ leaf, inst.hash_data value = .ok leaf ∧
    foldPath inst leaf index.val path.val = .ok root

/-- **Completeness (verifier-fold half):** if `(root_hash, index, value, path)` is
an honest Merkle path, `Proof.verify` returns `.ok true`. I.e. an honest proof
always verifies. The only hypothesis beyond the path spec is that the node
equality instance is *lawful on equal inputs* (`a = b → eq a b = ok true`) —
abstract `PartialEq` is otherwise unconstrained. No totality/injectivity needed:
`IsMerklePath` already witnesses every hash succeeding and pinpoints the root, so
the final comparison is of a value with itself. -/
theorem verify_path_complete {T B D : Type}
    (inst : IsMerkleTreeBackend B T D)
    (corecmpPartialEqInst : core.cmp.PartialEq T T)
    (corecmpEqInst : core.cmp.Eq T)
    (self : Proof T) (root_hash : T) (index : Std.Usize) (value : D)
    (hEqRefl : ∀ a, corecmpPartialEqInst.eq a a = .ok true)
    (hpath : IsMerklePath inst root_hash index value self.merkle_path) :
    Proof.verify corecmpPartialEqInst corecmpEqInst inst self root_hash index value
      = .ok true := by
  obtain ⟨leaf, hleaf, hfold⟩ := hpath
  unfold Proof.verify
  -- hash_data value = ok leaf
  rw [hleaf]; simp only [bind_tc_ok]
  -- the loop reconstructs `root_hash` from `leaf` over the whole path `[0, len)`
  have hloop : Proof.verify_loop inst
      { start := 0#usize, «end» := self.merkle_path.len } self.merkle_path index leaf
      = .ok root_hash := by
    have hspec := verify_loop_eq_foldPath inst self.merkle_path 0#usize index leaf root_hash
      (by scalar_tac) (by simpa using hfold)
    -- a `⦃ r => r = root_hash ⦄` spec on a `Result` forces `.ok root_hash`
    cases hc : Proof.verify_loop inst
        { start := 0#usize, «end» := self.merkle_path.len } self.merkle_path index leaf with
    | ok r => rw [hc] at hspec; simp only [WP.spec, WP.theta] at hspec; rw [hspec]
    | fail e => rw [hc] at hspec; simp only [WP.spec, WP.theta] at hspec
    | div => rw [hc] at hspec; simp only [WP.spec, WP.theta] at hspec
  rw [hloop]; simp only [bind_tc_ok]
  -- final comparison: root_hash with itself
  unfold core.cmp.impls.PartialEqShared.eq
  rw [hEqRefl]

/-! ## Completeness, Phase B: `build_merkle_path` / `get_proof_by_pos` produce an honest path

Phase A proved: an `IsMerklePath` is accepted by `verify`. Phase B closes the
loop: the path that `get_proof_by_pos` reads out of a well-formed tree IS an
`IsMerklePath`, so honest proofs verify end-to-end.

The structural fact a Merkle tree satisfies is the *binary-heap invariant*: a
node at position `n ≥ 1` is one of the two children of its parent
`parent_index n`, and the parent equals `hash_new_parent` of its (left, right)
children — in the order dictated by `n`'s side. `build_merkle_path` walks `n` up
to the root via `parent_index`, collecting `node[sibling_index n]` at each step;
`verify`'s `foldPath` then re-applies exactly those parent hashes. We capture the
per-step heap fact as a hypothesis `HeapStep` over the climb (its discharge from
`build`'s construction loop is the deepest remaining layer, isolated below), and
prove the loop⇄climb correspondence: the collected path folds back to the root.

`open` the carved tree namespace. -/

open merkle_tree.merkle merkle_tree.utils

/-- node-index sibling (pure Nat mirror of `sibling_index`): even → n-1, odd → n+1. -/
def sibIdx (n : Nat) : Nat := if n % 2 = 0 then n - 1 else n + 1
/-- node-index parent (pure Nat mirror of `parent_index`): `(n-1)/2` both parities. -/
def parentIdx (n : Nat) : Nat := (n - 1) / 2

/-- The reachable climb from leaf node-position `n` to the root, under the
binary-heap invariant. `g` is the tree's `node_get` as a partial map
(`g i = some (node at i)`); `bit` is the running leaf-ordinal whose low bit picks
the child side. `HeapClimb g n bit acc` says: folding `acc` (the running hash at
node `n`, expected `= node n`) up the path collected by `build_merkle_path` from
`n` lands on `node ROOT`. It is the exact precondition under which the loop's
output is an honest path. Defined by well-founded recursion on `n` (strictly
decreasing via `parentIdx`). -/
def HeapClimb {B T D : Type} (inst : IsMerkleTreeBackend B T D)
    (g : Nat → Option T) : Nat → Nat → T → Prop
  | 0,     _,   acc => g 0 = some acc           -- at root: acc must be the root node
  | n + 1, bit, acc =>
      -- one climb step: sibling exists, parent-hash holds in the parity order,
      -- and the climb continues from the parent.
      ∃ sib p,
        g (sibIdx (n + 1)) = some sib ∧
        (if bit % 2 = 0 then inst.hash_new_parent acc sib
         else inst.hash_new_parent sib acc) = .ok p ∧
        g (parentIdx (n + 1)) = some p ∧
        HeapClimb inst g (parentIdx (n + 1)) (bit / 2) p
decreasing_by
  · simp only [parentIdx]; omega

/-- Bridge: the carved `sibling_index`/`parent_index` agree (as `Result`-values)
with the pure `sibIdx`/`parentIdx` on `n ≥ 1` (in range). Reuses the Phase-#2
`@[progress]` specs. -/
theorem sibling_index_eq (n : Std.Usize) (hlo : 1 ≤ n.val) (hhi : n.val < Std.Usize.max) :
    ∃ s, sibling_index n = .ok s ∧ s.val = sibIdx n.val := by
  obtain ⟨s, hs⟩ : ∃ s, sibling_index n = .ok s ∧
      s.val = (if n.val % 2 = 0 then n.val - 1 else n.val + 1) := by
    have := sibling_index_ok n hlo hhi
    cases h : sibling_index n with
    | ok s => rw [h] at this; simp only [WP.spec, WP.theta] at this; exact ⟨s, rfl, this⟩
    | fail e => rw [h] at this; simp only [WP.spec, WP.theta] at this
    | div => rw [h] at this; simp only [WP.spec, WP.theta] at this
  exact ⟨s, hs.1, by rw [hs.2]; simp only [sibIdx]⟩

theorem parent_index_eq (n : Std.Usize) (hn : 1 ≤ n.val) :
    ∃ p, parent_index n = .ok p ∧ p.val = parentIdx n.val := by
  have := parent_index_ok n hn
  cases h : parent_index n with
  | ok p => rw [h] at this; simp only [WP.spec, WP.theta] at this
            exact ⟨p, rfl, by rw [this]; simp only [parentIdx]⟩
  | fail e => rw [h] at this; simp only [WP.spec, WP.theta] at this
  | div => rw [h] at this; simp only [WP.spec, WP.theta] at this

/-- `node_get self idx = ok (self.nodes.val[idx.val]?)` — the carved `node_get`
unfolds to a slice `get`. -/
theorem node_get_eq {T B D : Type} (inst : IsMerkleTreeBackend B T D)
    (self : MerkleTree B T D) (idx : Std.Usize) :
    MerkleTree.node_get inst self idx = .ok (self.nodes.val[idx.val]?) := by
  unfold MerkleTree.node_get core.slice.Slice.get core.slice.index.SliceIndexUsizeSlice
    core.slice.index.Usize.get
  simp only [alloc.vec.Vec.deref]
  rfl

/-- **Core Phase-B loop lemma.** With `g = self.nodes.val[·]?`, if the climb from
node `pos` satisfies `HeapClimb` with running hash `acc = node pos`, then
`build_merkle_path_loop` from `(path, pos)` succeeds with `.Ok finalPath`, and the
siblings it appended fold (via `foldPath`, starting `acc`, leaf-bit `bit`) back to
`node ROOT`. By strong induction on the climb height `pos.val`; each step peels
one `HeapClimb`/`foldPath` layer, the index halving in lockstep. The `hbound`
invariant (`|path| + pos < max`) keeps `sibling_index`'s `n+1` and every
`Vec::push` in range. -/
theorem build_merkle_path_loop_folds {T B D : Type}
    (inst : IsMerkleTreeBackend B T D) (self : MerkleTree B T D) (root : T)
    (hClone : ∀ a, inst.corecloneCloneInst.clone a = .ok a)
    (hroot : self.nodes.val[0]? = some root)
    (pos : Std.Usize) (bit : Nat) (acc : T) (path : alloc.vec.Vec T)
    -- invariant bound: path-so-far plus the remaining climb height fits in usize.
    -- Preserved across steps (path grows by 1, pos drops to ≤ (pos-1)/2, and
    -- 1 + (pos-1)/2 ≤ pos for pos ≥ 1), so each `Vec::push` stays in range.
    (hbound : path.val.length + pos.val < Std.Usize.max)
    (hclimb : HeapClimb inst (fun i => self.nodes.val[i]?) pos.val bit acc) :
    ∃ finalPath, MerkleTree.build_merkle_path_loop inst self path pos
        = .ok (.Ok finalPath) ∧
      (∃ suffix, finalPath.val = path.val ++ suffix) ∧
      foldPath inst acc bit (finalPath.val.drop path.val.length) = .ok root := by
  -- strong induction on the climb height `pos.val`
  generalize hm : pos.val = m
  induction m using Nat.strong_induction_on generalizing pos bit acc path with
  | _ m IH =>
    unfold MerkleTree.build_merkle_path_loop
    rw [loop]
    unfold MerkleTree.build_merkle_path_loop.body
    by_cases h0 : pos.val = 0
    · -- at the root: loop breaks with `.Ok path`; HeapClimb base gives acc = root
      have hpos0 : pos = 0#usize := by scalar_tac
      subst hm
      rw [h0] at hclimb
      simp only [HeapClimb] at hclimb           -- g 0 = some acc
      rw [hroot] at hclimb                       -- acc = root
      have hacc : acc = root := by injection hclimb with h; exact h.symm
      -- pos == ROOT is true
      simp only [hpos0, merkle_tree.merkle.ROOT, bne_self_eq_false, Bool.false_eq_true,
        if_false, ↓reduceIte, ne_eq, not_true_eq_false]
      refine ⟨path, by rfl, ⟨[], by simp⟩, ?_⟩
      simp only [List.drop_length, foldPath, hacc]
    · -- climb one level: pos = n+1
      obtain ⟨n, hn⟩ : ∃ n, pos.val = n + 1 := ⟨pos.val - 1, by omega⟩
      rw [hn] at hclimb
      simp only [HeapClimb] at hclimb
      obtain ⟨sib, p, hsib_g, hhash, hp_g, hrec⟩ := hclimb
      -- pos != ROOT
      have hne : pos ≠ 0#usize := by intro h; apply h0; rw [h]; rfl
      simp only [merkle_tree.merkle.ROOT, ne_eq]
      rw [if_pos (by simp only [bne_iff_ne, ne_eq]; intro hc; exact hne (by scalar_tac))]
      -- sibling_index pos = ok (sibIdx pos)
      obtain ⟨s, hsib_eq, hsib_val⟩ := sibling_index_eq pos (by omega) (by omega)
      simp only [hsib_eq, bind_tc_ok]
      -- node_get(s) = ok (nodes[s]?) = ok (some sib)  (s.val = sibIdx pos = sibIdx (n+1))
      rw [node_get_eq]
      have hsidx : s.val = sibIdx (n + 1) := by rw [hsib_val, hn]
      rw [hsidx, hsib_g]
      simp only [bind_tc_ok]
      -- clone sib = ok sib
      rw [hClone]
      simp only [bind_tc_ok]
      -- push path sib = ok path' with path'.val = path.val ++ [sib]
      obtain ⟨path', hpush, hpushval⟩ : ∃ path', alloc.vec.Vec.push path sib = .ok path' ∧
          path'.val = path.val ++ [sib] := by
        have := alloc.vec.Vec.push_spec path sib (by omega)
        cases hc : alloc.vec.Vec.push path sib with
        | ok p' => rw [hc] at this; simp only [WP.spec, WP.theta] at this; exact ⟨p', rfl, this⟩
        | fail e => rw [hc] at this; simp only [WP.spec, WP.theta] at this
        | div => rw [hc] at this; simp only [WP.spec, WP.theta] at this
      rw [hpush]; simp only [bind_tc_ok]
      -- parent_index pos = ok (parentIdx pos)
      obtain ⟨pp, hpp_eq, hpp_val⟩ := parent_index_eq pos (by omega)
      rw [hpp_eq]; simp only [bind_tc_ok]
      -- the loop continues: `loop body (path', pp)` = build_merkle_path_loop self path' pp
      show ∃ finalPath, MerkleTree.build_merkle_path_loop inst self path' pp
          = .ok (.Ok finalPath) ∧ _ ∧ _
      -- apply IH at pp (height parentIdx (n+1) < n+1 = pos.val)
      have hpp_lt : pp.val < pos.val := by
        rw [hpp_val]; unfold parentIdx; have : pos.val = n + 1 := hn; omega
      have hpp_climb : HeapClimb inst (fun i => self.nodes.val[i]?) pp.val (bit / 2) p := by
        rw [hpp_val, hn]; exact hrec
      -- bound preserved: |path'| + pp = (|path|+1) + (pos-1)/2 ≤ |path| + pos < max
      have hbound' : path'.val.length + pp.val < Std.Usize.max := by
        rw [hpushval, hpp_val]; unfold parentIdx
        simp only [List.length_append, List.length_cons, List.length_nil]
        have : pos.val = n + 1 := hn; omega
      obtain ⟨finalPath, hfp_loop, ⟨suf, hsuf⟩, hfp_fold⟩ :=
        IH pp.val (by rw [← hm]; exact hpp_lt) pp (bit / 2) p path' hbound' hpp_climb rfl
      -- finalPath.val = path'.val ++ suf = path.val ++ (sib :: suf)
      have hfinal : finalPath.val = path.val ++ (sib :: suf) := by
        rw [hsuf, hpushval]; simp
      refine ⟨finalPath, hfp_loop, ⟨sib :: suf, hfinal⟩, ?_⟩
      -- drop |path| of finalPath = sib :: drop |path'| of finalPath
      have hdroplen : path'.val.length = path.val.length + 1 := by rw [hpushval]; simp
      have hdrop : finalPath.val.drop path.val.length
          = sib :: finalPath.val.drop (path.val.length + 1) := by
        rw [hfinal]; simp [List.drop_append_of_le_length]
      rw [hdrop]
      simp only [foldPath, hhash, bind_tc_ok]
      rw [← hdroplen]; exact hfp_fold

/-- `build_merkle_path` (the wrapper) starts from an empty path (`with_capacity`),
so the produced path's *entire* contents fold back to the root. -/
theorem build_merkle_path_folds {T B D : Type}
    (inst : IsMerkleTreeBackend B T D) (self : MerkleTree B T D) (root : T)
    (hClone : ∀ a, inst.corecloneCloneInst.clone a = .ok a)
    (hcap : self.nodes.val.length < Std.Usize.max)
    (hroot : self.nodes.val[0]? = some root)
    (pos : Std.Usize) (bit : Nat) (acc : T)
    (hbound : pos.val < Std.Usize.max)
    (hclimb : HeapClimb inst (fun i => self.nodes.val[i]?) pos.val bit acc) :
    ∃ finalPath, MerkleTree.build_merkle_path inst self pos = .ok (.Ok finalPath) ∧
      foldPath inst acc bit finalPath.val = .ok root := by
  unfold MerkleTree.build_merkle_path MerkleTree.node_count core.num.Usize.ilog2
  simp only [bind_tc_ok]
  -- `nodes.len + 1` does not overflow (nodes.length < max), so reduces to ok _
  obtain ⟨s1, hs1⟩ : ∃ s1, self.nodes.len + 1#usize = Result.ok s1 := by
    have := Std.Usize.add_spec (x := self.nodes.len) (y := 1#usize)
      (by simp only [alloc.vec.Vec.len]; scalar_tac)
    cases hc : self.nodes.len + 1#usize with
    | ok s1 => exact ⟨s1, rfl⟩
    | fail e => rw [hc] at this; simp only [WP.spec, WP.theta] at this
    | div => rw [hc] at this; simp only [WP.spec, WP.theta] at this
  rw [hs1]; simp only [bind_tc_ok]
  -- with_capacity gives an empty Vec; UScalar.cast of 0 succeeds
  obtain ⟨td, htd⟩ : ∃ td, lift (UScalar.cast .Usize 0#u32) = Result.ok td := by
    simp only [UScalar.cast]; exact ⟨_, rfl⟩
  rw [htd]; simp only [bind_tc_ok]
  obtain ⟨finalPath, hloop, _, hfold⟩ :=
    build_merkle_path_loop_folds inst self root hClone hroot pos bit acc
      (alloc.vec.Vec.with_capacity T td)
      (by
        have : (alloc.vec.Vec.with_capacity T td).val.length = 0 := by
          simp only [alloc.vec.Vec.with_capacity, alloc.vec.Vec.new, List.length_nil]
        rw [this]; omega)
      hclimb
  refine ⟨finalPath, hloop, ?_⟩
  have hlen0 : (alloc.vec.Vec.with_capacity T td).val.length = 0 := by
    simp only [alloc.vec.Vec.with_capacity, alloc.vec.Vec.new, List.length_nil]
  rw [hlen0] at hfold; simpa using hfold

/-- `get_proof_by_pos pos` reads the path for the leaf at node-position
`leafPos = pos + node_count/2`. If that leaf's climb satisfies `HeapClimb` with
leaf-ordinal `bit` and running hash `acc`, the produced proof's `merkle_path`
folds (from `acc`, `bit`) back to the root. -/
theorem get_proof_by_pos_folds {T B D : Type}
    (inst : IsMerkleTreeBackend B T D) (self : MerkleTree B T D) (root : T)
    (hClone : ∀ a, inst.corecloneCloneInst.clone a = .ok a)
    (hcap : self.nodes.val.length < Std.Usize.max)
    (hroot : self.nodes.val[0]? = some root)
    (pos : Std.Usize) (bit : Nat) (acc : T) (half leafPos : Std.Usize)
    (hhalf : alloc.vec.Vec.len self.nodes / 2#usize = Result.ok half)
    (hleafPos : pos + half = Result.ok leafPos)
    (hbound : leafPos.val < Std.Usize.max)
    (hclimb : HeapClimb inst (fun i => self.nodes.val[i]?) leafPos.val bit acc) :
    ∃ proof, MerkleTree.get_proof_by_pos inst self pos = .ok (some proof) ∧
      foldPath inst acc bit proof.merkle_path.val = .ok root := by
  unfold MerkleTree.get_proof_by_pos MerkleTree.node_count
  simp only [bind_tc_ok]
  -- node_count/2 = half, then pos + half = leafPos
  rw [hhalf]; simp only [bind_tc_ok]
  rw [hleafPos]; simp only [bind_tc_ok]
  obtain ⟨finalPath, hbmp, hfold⟩ :=
    build_merkle_path_folds inst self root hClone hcap hroot leafPos bit acc hbound hclimb
  rw [hbmp]; simp only [MerkleTree.create_proof, bind_tc_ok]
  exact ⟨{ merkle_path := finalPath }, rfl, hfold⟩

/-! ## Completeness, end-to-end: an honest proof verifies

Composing Phase B (`get_proof_by_pos` produces a fold-correct path) with Phase A
(`verify` accepts any fold-correct path) gives the headline result: a proof
generated by `get_proof_by_pos` from a well-formed Merkle tree is accepted by
`verify`. The structural hypothesis on the tree is `HeapClimb` (the binary-heap
invariant along the leaf's climb); the backend hypotheses are the honest "lawful
hash/eq/clone" assumptions (clone is identity, eq is reflexive). -/
theorem honest_proof_verifies {T B D : Type}
    (inst : IsMerkleTreeBackend B T D)
    (corecmpPartialEqInst : core.cmp.PartialEq T T)
    (corecmpEqInst : core.cmp.Eq T)
    (self : MerkleTree B T D) (root : T) (index : Std.Usize) (value : D)
    (half leafPos : Std.Usize) (leaf : T)
    -- backend laws (honest assumptions, as in Phase A):
    (hClone : ∀ a, inst.corecloneCloneInst.clone a = .ok a)
    (hEqRefl : ∀ a, corecmpPartialEqInst.eq a a = .ok true)
    -- tree shape: < max nodes; the root is `node 0`; the leaf at `leafPos` is the
    -- hash of `value`; and `leafPos = index + node_count/2` (the standard layout):
    (hcap : self.nodes.val.length < Std.Usize.max)
    (hroot : self.nodes.val[0]? = some root)
    (hleaf : inst.hash_data value = .ok leaf)
    (hhalf : alloc.vec.Vec.len self.nodes / 2#usize = Result.ok half)
    (hleafPos : index + half = Result.ok leafPos)
    (hbound : leafPos.val < Std.Usize.max)
    -- the binary-heap invariant along this leaf's climb, with the running hash
    -- starting at the leaf node `= hash_data value`:
    (hclimb : HeapClimb inst (fun i => self.nodes.val[i]?) leafPos.val index.val leaf) :
    -- THEN: the generated proof exists and `verify` accepts it.
    ∃ proof, MerkleTree.get_proof_by_pos inst self index = .ok (some proof) ∧
      Proof.verify corecmpPartialEqInst corecmpEqInst inst proof root index value = .ok true := by
  obtain ⟨proof, hgp, hfold⟩ :=
    get_proof_by_pos_folds inst self root hClone hcap hroot index index.val leaf
      half leafPos hhalf hleafPos hbound hclimb
  refine ⟨proof, hgp, ?_⟩
  -- the produced path is an `IsMerklePath` (leaf = hash_data value, folds to root)
  apply verify_path_complete inst corecmpPartialEqInst corecmpEqInst proof root index value hEqRefl
  exact ⟨leaf, hleaf, hfold⟩

end MerkleVerifyProofs
