#!/usr/bin/env python3
"""Post-process the hax-extracted proofs/hax/math.lean.

The pinned Hax Lean proof-lib does not provide some `core::`/`std::` models that
math.lean references (overflowing_add, from_le/be_bytes, trailing_zeros,
reverse_bits, usize::BITS, slice swap, hint::unreachable_unchecked). We provide
them as opaque stubs in CoreModelsSupplement.lean (a sibling lean_lib), but
hax's generated math.lean only emits `import Hax` and has no knowledge of our
supplement — so we inject the import here, right after the `import Hax` line.

Idempotent (skips if already imported); fails loudly if the anchor line is
absent (so a hax output change is caught rather than silently ignored).
"""

import sys
from pathlib import Path

ANCHOR = "import Hax\n"
INJECT = "import CoreModelsSupplement\n"


def main() -> int:
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <path/to/math.lean>", file=sys.stderr)
        return 2
    path = Path(sys.argv[1])
    if not path.is_file():
        print(f"error: {path} not found", file=sys.stderr)
        return 1

    text = path.read_text()
    if INJECT in text:
        print(f"patch_hax_math: {path} already imports supplement, skipping")
        return 0

    count = text.count(ANCHOR)
    if count != 1:
        print(
            f"error: expected exactly one `import Hax` line, found {count}.\n"
            f"  hax output may have changed — review and update this script.",
            file=sys.stderr,
        )
        return 1

    text = text.replace(ANCHOR, ANCHOR + INJECT, 1)
    path.write_text(text)
    print(f"patch_hax_math: injected `import CoreModelsSupplement` into {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
