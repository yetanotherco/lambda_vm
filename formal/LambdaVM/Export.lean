module

public meta import LambdaVM
public meta import Clean.Circuit.Json

/-!
Exports the operations of each modeled chip as JSON, for the differential check against the
Rust prover (`prover/src/tests/lean_diff.rs`). Variables are numbered in `Row` field order,
which is the Rust column order.

BITWISE has no row circuit: its export is a sample of the table, the precomputed columns of
`Bitwise.row i` and the message every static lookup channel serves at row `i`.
-/

open Lean LambdaVM

meta def exportCircuit (name : String) {Row : TypeMap} [ProvableType Row]
    (main : Var Row (F p) → Circuit (F p) Unit) : IO Unit := do
  let ops := (main (varFromOffset Row 0)).operations 0
  let json := Json.mkObj [("name", name), ("columns", size Row), ("operations", toJson ops)]
  IO.FS.writeFile s!"export/{name}.json" (json.pretty 100)
  IO.println s!"wrote export/{name}.json"

/-- BITWISE rows to sample: every `z` with edge bytes for `x` and `y`, the last row, and
pseudo-random rows. -/
meta def bitwiseIndices : List ℕ :=
  let edge := [0, 1, 0x7F, 0x80, 0xFF]
  let grid := (List.range 16).flatMap fun z =>
    edge.flatMap fun y => edge.map fun x => Bitwise.index x y z
  let random := (List.range 256).map fun k => (k * 2654435761 + 12345) % Bitwise.numRows
  grid ++ [Bitwise.numRows - 1] ++ random

meta def rowJson (r : Bitwise.Row) : Json :=
  toJson [r.x, r.y, r.z, r.and, r.or, r.xor, r.msb8, r.msb16, r.zero, r.sll, r.sllc]

/-- The messages a static lookup channel serves at each sampled row: `perRow` consecutive
entries of its table per BITWISE row (BYTE_ALU serves three, one per `opsel`). -/
meta def channelJson {M : TypeMap} [ProvableType M] (slc : StaticLookupChannel (F p) M)
    (perRow : ℕ) (indices : List ℕ) : Json :=
  let table := slc.table.toArray
  let messages := indices.map fun i => (List.range perRow).map fun k =>
    match table[perRow * i + k]? with
    | some m => (toElements m).toList.map fun e : F p => e.val
    | none => []
  Json.mkObj [("channel", slc.name), ("messages", toJson messages)]

meta def exportBitwise : IO Unit := do
  let indices := bitwiseIndices
  let channels := [
    channelJson Bitwise.areBytesTable 1 indices,
    channelJson Bitwise.isHalfTable 1 indices,
    channelJson Bitwise.isB20Table 1 indices,
    channelJson Bitwise.msb8Table 1 indices,
    channelJson Bitwise.msb16Table 1 indices,
    channelJson Bitwise.zeroTable 1 indices,
    channelJson Bitwise.hwslTable 1 indices,
    channelJson Bitwise.byteAluTable 3 indices]
  let json := Json.mkObj [("name", "bitwise"), ("indices", toJson indices),
    ("rows", Json.arr (indices.map (rowJson ∘ Bitwise.row)).toArray),
    ("channels", Json.arr channels.toArray)]
  IO.FS.writeFile "export/bitwise.json" json.compress
  IO.println "wrote export/bitwise.json"

public meta def main : IO Unit := do
  exportCircuit "lt" LT.main
  exportCircuit "eq" EQ.main
  exportCircuit "shift" SHIFT.main
  exportBitwise
