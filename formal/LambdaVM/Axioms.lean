module

public import LambdaVM

/-!
# Axiom check

Every circuit of the model and every ensemble theorem must rest on nothing beyond Lean's three
standard axioms: `propext`, `Classical.choice`, `Quot.sound`. In particular none of them may
depend on `sorryAx`.

`#print axioms` reports transitively, so a declaration reached by one already listed needs no
line of its own. The templates and the two static tables below are *not* reached by any of them,
which is why they are listed explicitly. The `sorry` check on `lake build` in the workflow is the
backstop for everything else in the library.

`.github/workflows/formal.yml` runs this file with `lake env lean` and fails if any report
names `sorryAx` or an axiom outside those three. It also counts the `#print axioms` lines
here and requires one report for each, so a declaration that stops existing fails the build
rather than silently dropping out of the check. Add a line for every new circuit, ensemble
soundness theorem or global row theorem.
-/

-- The modelled chips. Each `circuit` bundles its own soundness and completeness proofs.
#print axioms LambdaVM.LT.circuit
#print axioms LambdaVM.EQ.circuit
#print axioms LambdaVM.SHIFT.circuit
#print axioms LambdaVM.BYTEWISE.circuit
#print axioms LambdaVM.MUL.circuit
#print axioms LambdaVM.DVRM.circuit

-- One RawChannel per bus name, so a chip that moves a pull between spellings fails here
-- rather than at the next ensemble somebody writes.
#print axioms LambdaVM.Buses.model_channels_agree

-- The ensembles.
#print axioms LambdaVM.Ensembles.eq_rows_correct
#print axioms LambdaVM.Ensembles.LtStaged.tableSoundness
#print axioms LambdaVM.Ensembles.DvrmStaged.tableSoundness
#print axioms LambdaVM.Ensembles.DvrmStaged.dvrm_rows_correct
#print axioms LambdaVM.Ensembles.DvrmStaged.lt_rows_correct

-- The templates the chips are built from. They are `FormalAssertion`s that nothing above
-- reaches, so without these lines a `sorry` in one of them would pass the check.
#print axioms LambdaVM.IsBit.circuit
#print axioms LambdaVM.Add.circuit
#print axioms LambdaVM.IsByteCheck.circuit
#print axioms LambdaVM.AreBytesCheck.circuit
#print axioms LambdaVM.SignCheck.circuit
#print axioms LambdaVM.NegCheck.circuit

-- The static tables. `msb8Table` is the one no modelled chip pulls, so without this line a
-- `sorry` in it would pass the check; `isB20Table` is pulled by MUL and reached anyway, and is
-- printed next to it so the pair reads as one list. Printing the table covers its
-- `guarantees_iff`.
#print axioms LambdaVM.Bitwise.isB20Table
#print axioms LambdaVM.Bitwise.msb8Table

