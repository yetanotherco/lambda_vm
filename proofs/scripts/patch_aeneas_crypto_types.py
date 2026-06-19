#!/usr/bin/env python3
"""Post-process aeneas-generated proofs/aeneas/Crypto/{Types,Funs}.lean.

Same class of aeneas codegen defect as Math/Types.lean (see
patch_aeneas_math_types.py): Lean rejects duplicate field names in a
`structure`. Crypto's split-files output re-declares the `math` field traits it
depends on, so the `IsField` / `IsSubFieldOf` duplicate-field defects reappear
here, PLUS a crypto-specific one: `IsStarkTranscript` carries two
`mathfieldtraitsIsFieldInst` fields (for the `F` and `S` field params).

Renaming the second field of each affected `structure` (in Types.lean) requires
renaming the SECOND assignment in every instance literal that CONSTRUCTS that
structure (in Funs.lean) to match — otherwise Lean reports the field as
specified twice / the renamed field as missing. We patch both files.

The `math.field.traits.IsField` "unknown identifier" errors downstream are a
CASCADE of the Types.lean duplicate-field failure; fixing it resolves them.

Idempotent (skips if the marker is present); fails loudly if a snippet is
absent or ambiguous.
"""

import sys
from pathlib import Path

MARKER = "-- [patched by patch_aeneas_crypto_types.py]"

# Per-file (snippet -> replacement). Each must match exactly once.
TYPES_REPLACEMENTS = [
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
        "IsSubFieldOf dup field decl: second IsFieldInst (F)",
        "  IsFieldInst : math.field.traits.IsField F Self_Clause1_BaseType",
        "  IsFieldClause1Inst : math.field.traits.IsField F Self_Clause1_BaseType",
    ),
    (
        "IsStarkTranscript dup field decl: second mathfieldtraitsIsFieldInst (S)",
        "  mathfieldtraitsIsFieldInst : math.field.traits.IsField S\n    Self_Clause2_BaseType",
        "  mathfieldtraitsIsFieldSInst : math.field.traits.IsField S\n    Self_Clause2_BaseType",
    ),
]

FUNS_REPLACEMENTS = [
    (
        "IsSubFieldOf.Blanket literal: second IsFieldInst assignment",
        "  IsFieldInst := IsFieldInst1\n  IsFieldInst := IsFieldInst1",
        "  IsFieldInst := IsFieldInst1\n  IsFieldClause1Inst := IsFieldInst1",
    ),
    (
        "IsStarkTranscript literal: second mathfieldtraitsIsFieldInst assignment",
        "  mathfieldtraitsIsFieldInst := mathfieldtraitsIsFieldInst1",
        "  mathfieldtraitsIsFieldSInst := mathfieldtraitsIsFieldInst1",
    ),
]


def patch_file(path: Path, replacements) -> int:
    if not path.is_file():
        print(f"error: {path} not found", file=sys.stderr)
        return 1
    text = path.read_text()
    if MARKER in text:
        print(f"patch_aeneas_crypto_types: {path} already patched, skipping")
        return 0
    for desc, find, repl in replacements:
        count = text.count(find)
        if count != 1:
            print(
                f"error: snippet matched {count}x, expected 1 ({desc}) in {path}.\n"
                f"  aeneas output may have changed — review and update this script.\n"
                f"  searched for:\n{find}",
                file=sys.stderr,
            )
            return 1
        text = text.replace(find, repl)
    path.write_text(f"{MARKER}\n{text}")
    print(f"patch_aeneas_crypto_types: applied {len(replacements)} fixes to {path}")
    return 0


def main() -> int:
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <path/to/Crypto-dir>", file=sys.stderr)
        return 2
    d = Path(sys.argv[1])
    rc = patch_file(d / "Types.lean", TYPES_REPLACEMENTS)
    if rc != 0:
        return rc
    return patch_file(d / "Funs.lean", FUNS_REPLACEMENTS)


if __name__ == "__main__":
    sys.exit(main())
