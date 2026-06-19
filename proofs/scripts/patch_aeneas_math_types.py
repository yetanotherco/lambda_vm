#!/usr/bin/env python3
"""Post-process aeneas-generated proofs/aeneas/Math/Types.lean.

Aeneas (rev pinned in proofs/aeneas/lake-manifest.json) emits three classes of
defect in the generated `Math/Types.lean` that prevent `lake build`:

  1. Duplicate structure field names. Lean rejects two fields with the same name
     in one `structure`. `IsField` carries `core.clone.Clone` / `core.fmt.Debug`
     instances for both `Self` and `Self_BaseType` under the SAME generated name;
     `IsSubFieldOf` carries two `IsFieldInst` fields. We rename the second
     occurrence in each. These fields are never projected by name in Funs.lean
     (verified), so renaming the declaration is safe.

  2. `field.traits.IsField` "unknown identifier" downstream — a CASCADE from (1):
     when the `IsField` structure fails to elaborate, the identifier is never
     defined, so every later reference fails. Fixing (1) resolves these.

  3. The `rand::rng::Rng` <-> `rand::rng::Fill` traits are mutually recursive and
     emitted as two sequential `structure`s, so each forward-references the other
     (and `mutual structure` fails the kernel positivity check — the cross type
     appears in negative position). RNG is out of proof scope (prover/transcript
     side, never the verifier path), and Funs.lean only uses these as opaque
     instance-binder types (verified — never constructed/projected), so we model
     them as opaque `axiom ... : Type`, matching how aeneas already axiomatizes
     other `rand`/`rand_core` externals in TypesExternal.lean.

The `parameterize_trait_types` aeneas flag (Config.ml) targets the assoc-type
issue behind (2) directly, but it is NOT exposed as a CLI option in Main.ml, so
it cannot be enabled without rebuilding aeneas from OCaml source. Hence this
deterministic post-process instead.

The script is IDEMPOTENT (safe to re-run) and FAILS LOUDLY if an expected
snippet is absent — so a future aeneas version that changes its output trips an
error here rather than silently leaving the build broken.
"""

import sys
from pathlib import Path

MARKER = "-- [patched by patch_aeneas_math_types.py]"

# (description, exact text to find, replacement). Applied in order; each must
# match exactly once unless already patched.
REPLACEMENTS = [
    (
        "IsField dup field: corecloneCloneInst (Self_BaseType)",
        "  corecloneCloneInst : core.clone.Clone Self_BaseType",
        "  corecloneCloneBaseTypeInst : core.clone.Clone Self_BaseType",
    ),
    (
        "IsField dup field: corefmtDebugInst (Self_BaseType)",
        "  corefmtDebugInst : core.fmt.Debug Self_BaseType",
        "  corefmtDebugBaseTypeInst : core.fmt.Debug Self_BaseType",
    ),
    (
        "IsSubFieldOf dup field: second IsFieldInst (F)",
        "  IsFieldInst : field.traits.IsField F Self_Clause1_BaseType",
        "  IsFieldClause1Inst : field.traits.IsField F Self_Clause1_BaseType",
    ),
    (
        "rand.rng.Rng mutually-recursive structure -> opaque axiom",
        """@[rust_trait "rand::rng::Rng" (parentClauses := ["rand_coreRngCoreInst"])]
structure rand.rng.Rng (Self : Type) where
  rand_coreRngCoreInst : rand_core.RngCore Self
  fill : forall {T : Type} (FillInst : rand.rng.Fill T), Self → T → Result
    (Self × T)""",
        '@[rust_trait "rand::rng::Rng"] axiom rand.rng.Rng (Self : Type) : Type',
    ),
    (
        "rand.rng.Fill mutually-recursive structure -> opaque axiom",
        """@[rust_trait "rand::rng::Fill"]
structure rand.rng.Fill (Self : Type) where
  try_fill : forall {R : Type} (RngInst : rand.rng.Rng R), Self → R →
    Result ((core.result.Result Unit rand_core.error.Error) × Self × R)""",
        '@[rust_trait "rand::rng::Fill"] axiom rand.rng.Fill (Self : Type) : Type',
    ),
]


def main() -> int:
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <path/to/Math/Types.lean>", file=sys.stderr)
        return 2
    path = Path(sys.argv[1])
    if not path.is_file():
        print(f"error: {path} not found", file=sys.stderr)
        return 1

    text = path.read_text()
    if MARKER in text:
        print(f"patch_aeneas_math_types: {path} already patched, skipping")
        return 0

    for desc, find, repl in REPLACEMENTS:
        count = text.count(find)
        if count == 0:
            print(
                f"error: expected snippet not found ({desc}).\n"
                f"  aeneas output may have changed — review and update this script.\n"
                f"  searched for:\n{find}",
                file=sys.stderr,
            )
            return 1
        if count > 1:
            print(
                f"error: snippet matched {count}x, expected 1 ({desc}). "
                f"Refusing to patch ambiguously.",
                file=sys.stderr,
            )
            return 1
        text = text.replace(find, repl)

    text = f"{MARKER}\n{text}"
    path.write_text(text)
    print(f"patch_aeneas_math_types: applied {len(REPLACEMENTS)} fixes to {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
