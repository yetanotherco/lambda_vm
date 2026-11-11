module

public import LambdaVM

/-!
# Axiom check

Every modelled circuit and every ensemble theorem must rest on nothing beyond Lean's three
standard axioms: `propext`, `Classical.choice`, `Quot.sound`. In particular none of them may
depend on `sorryAx`.

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
#print axioms LambdaVM.Ensembles.DvrmStaged.rows_correct_of
#print axioms LambdaVM.Ensembles.DvrmStaged.dvrm_rows_correct
#print axioms LambdaVM.Ensembles.DvrmStaged.lt_rows_correct
#print axioms LambdaVM.Ensembles.DvrmStaged.mul_rows_correct
