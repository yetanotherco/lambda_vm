import LambdaVM
import Clean.Circuit.Json

/-!
Exports the operations of each modeled chip as JSON, for the differential check against the
Rust prover (`prover/src/tests/lean_diff.rs`). Variables are numbered in `Row` field order,
which is the Rust column order.
-/

open Lean LambdaVM

def exportCircuit (name : String) {Row : TypeMap} [ProvableType Row]
    (main : Var Row (F p) → Circuit (F p) Unit) : IO Unit := do
  let ops := (main (varFromOffset Row 0)).operations 0
  let json := Json.mkObj [("name", name), ("columns", size Row), ("operations", toJson ops)]
  IO.FS.writeFile s!"export/{name}.json" (json.pretty 100)
  IO.println s!"wrote export/{name}.json"

def main : IO Unit := do
  exportCircuit "lt" LT.main
  exportCircuit "eq" EQ.main
  exportCircuit "shift" SHIFT.main
