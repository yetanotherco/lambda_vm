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

-- The ensembles.
#print axioms LambdaVM.Ensembles.eq_rows_correct
#print axioms LambdaVM.Ensembles.LtStaged.tableSoundness
