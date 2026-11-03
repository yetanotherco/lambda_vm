module

public import Clean.Circuit.Channel

@[expose] public section

/-!
# Static lookup channels

A table the verifier knows, given as a list of rows, read through a channel whose guarantee is
the semantic reading of a row. Clean main replaced its `StaticLookupChannel` by `Air.Flat.Lookup`,
whose rows are witness programs of the row index. The formal model keeps its tables as lists, with
`guarantees_iff` relating each list to its reading, so the structure lives here.
-/

namespace LambdaVM

structure StaticLookupChannel (F : Type) [Field F] (Message : TypeMap) [ProvableType Message] where
  name : String
  table : List (Message F)
  Guarantees : Message F → Prop
  guarantees_iff : ∀ msg, Guarantees msg ↔ msg ∈ table

/-- The channel through which other components read the table. -/
@[circuit_norm]
def StaticLookupChannel.toChannel {F : Type} [Field F] {Message : TypeMap} [ProvableType Message]
    (slc : StaticLookupChannel F Message) : Channel F Message where
  name := slc.name
  Guarantees msg _ := slc.Guarantees msg

end LambdaVM
