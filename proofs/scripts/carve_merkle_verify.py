#!/usr/bin/env python3
"""Carve the Merkle single-leaf `Proof::verify` subset out of the aeneas-generated
Crypto Lean into a self-contained, ZERO-MAINTENANCE module.

The full generated `Crypto/Funs.lean` does not compile (out-of-scope poseidon /
transcript / batch / concrete-backend code hits upstream aeneas codegen limits),
but the single-leaf `Proof::verify` and its dependencies are well-formed and
depend only on the abstract `IsMerkleTreeBackend` trait + Aeneas Std primitives.

This script extracts, from the generated `Types.lean` + `Funs.lean`, the
transitive dependency closure of a small set of ROOT declarations, and writes
`MerkleVerify.lean` containing exactly that closure (types first, then funs, in
source order). Because the closure is recomputed from the generated files on
every run, regeneration needs no hand-editing — re-run this and the carved file
is current. If a root or its closure ever pulls in something that doesn't exist,
the script fails loudly.

Roots: the three `verify` defs. Their closure is expected to be just those plus
the `IsMerkleTreeBackend` structure (everything else they touch is Aeneas Std,
which is imported, not carved).
"""

import re
import sys
from pathlib import Path

ROOTS = [
    "merkle_tree.proof.Proof.verify",
    "merkle_tree.proof.Proof.verify_loop",
    "merkle_tree.proof.Proof.verify_loop.body",
    # Index-algebra utilities: the non-looping node-index helpers that relate a
    # node to its sibling/parent. Carved alongside `verify` because they are the
    # arithmetic backbone of the Merkle path traversal (sibling/parent inverses)
    # that the completeness proof reasons about, and their (precondition'd)
    # panic-freedom is proved in MerkleVerifyProofs.lean. Their closure is just
    # themselves + the `is_multiple_of` model (no trait, no loop).
    "merkle_tree.utils.sibling_index",
    "merkle_tree.utils.parent_index",
    "merkle_tree.utils.get_sibling_pos",
    "merkle_tree.utils.get_parent_pos",
    "merkle_tree.utils.is_power_of_two",
    # Proof generation (Phase B of completeness): `build_merkle_path` /
    # `get_proof_by_pos` read the sibling path out of an already-built tree. We
    # prove they produce an `IsMerklePath` (given the binary-heap tree invariant
    # as a hypothesis), then compose with `verify_path_complete` for end-to-end
    # completeness ("an honest proof verifies"). Closures stay bounded:
    # MerkleTree/Error structs, node_get/node_count, the path loop. Models for
    # `merkle_tree.merkle.ROOT` (an inline-attr one-liner the block parser doesn't
    # split cleanly) and `Usize.ilog2` (missing from this aeneas Std, only used
    # for a capacity hint) are injected into the header below.
    "merkle_tree.merkle.MerkleTree.build_merkle_path",
    "merkle_tree.merkle.MerkleTree.build_merkle_path_loop",
    "merkle_tree.merkle.MerkleTree.build_merkle_path_loop.body",
    "merkle_tree.merkle.MerkleTree.get_proof_by_pos",
    "merkle_tree.merkle.MerkleTree.create_proof",
    "merkle_tree.merkle.MerkleTree.node_count",
    "merkle_tree.merkle.MerkleTree.node_get",
]

# A top-level declaration head. Attribute lines (`@[...]`) and doc comments that
# immediately precede it are attached to the block.
DECL_RE = re.compile(
    r"^(?:def|abbrev|structure|inductive|impl_def|instance|opaque|theorem|class)\b"
)
# Identifier characters in these generated names: dotted, alnum, underscore, prime.
NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_.']*")


class Block:
    __slots__ = ("name", "text", "kind", "order")

    def __init__(self, name, text, kind, order):
        self.name = name
        self.text = text
        self.kind = kind  # "type" or "fun"
        self.order = order


def strip_trailing_doc(block_text):
    """Drop a dangling doc comment / attribute lines at the END of a block.

    A block's text runs until the next declaration head, so it may include the
    doc comment (and `@[...]` attrs) that actually belong to the FOLLOWING
    declaration — which is excluded from the carve. Left in place, that orphan
    `/-- ... -/` (with no decl after it) is a parse error. Remove any trailing
    run of blank / doc-comment / attribute lines.
    """
    lines = block_text.splitlines(keepends=True)
    end = len(lines)
    while end > 0:
        s = lines[end - 1].strip()
        if s == "" or s.startswith("@[") or s.startswith('"'):
            end -= 1
            continue
        if s.endswith("-/"):
            # swallow the whole doc comment upward to its `/-` opener
            k = end - 1
            while k > 0 and not lines[k].lstrip().startswith("/-"):
                k -= 1
            end = k
            continue
        break
    return "".join(lines[:end])


def parse_blocks(text, kind, order_base):
    """Split a generated Lean file into named top-level declaration blocks.

    A block starts at a declaration head (optionally preceded by `@[...]`
    attribute lines and/or a `/-- ... -/` doc comment) and runs until the next
    block start. The declared name is the first identifier after the head
    keyword.
    """
    lines = text.splitlines(keepends=True)
    blocks = []
    # Index the line numbers where a real declaration head appears.
    heads = [i for i, ln in enumerate(lines) if DECL_RE.match(ln)]
    if not heads:
        return blocks
    # The line index where the *previous* block ended, so we never absorb lines
    # belonging to it. Blocks are contiguous; a block's preamble starts right
    # after the previous block's last non-blank line.
    for h_idx, line_no in enumerate(heads):
        prev_end = heads[h_idx - 1] if h_idx > 0 else -1
        # Walk upward from the head, absorbing attribute lines and a complete
        # doc comment, but never past the previous declaration head.
        j = line_no - 1
        while j > prev_end:
            s = lines[j].strip()
            if s == "":
                j -= 1
                continue
            if s.startswith("@[") or s.startswith('"'):
                # attribute line (possibly wrapped onto continuation lines)
                j -= 1
                continue
            if s.endswith("-/"):
                # end of a doc comment: swallow up through its `/-` opener
                j -= 1
                while j > prev_end and not lines[j].lstrip().startswith("/-"):
                    j -= 1
                j -= 1  # also take the `/-` opener line
                continue
            break
        start = j + 1
        end = heads[h_idx + 1] if h_idx + 1 < len(heads) else len(lines)
        # Trim trailing blank lines from the previous block's tail that we may
        # have swept into `start` of the next; keep it simple: include as-is.
        block_text = "".join(lines[start:end])
        # The declared name: first identifier after the head keyword.
        head_line = lines[line_no]
        m_kw = DECL_RE.match(head_line)
        rest = head_line[m_kw.end():]
        # name may be on the next line if the head wraps (e.g. `def\n  name`).
        nm = NAME_RE.search(rest)
        k = line_no
        while nm is None and k + 1 < end:
            k += 1
            nm = NAME_RE.search(lines[k])
        if nm is None:
            continue
        blocks.append(Block(nm.group(0), block_text, kind, order_base + line_no))
    return blocks


def main() -> int:
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <path/to/Crypto-dir>", file=sys.stderr)
        return 2
    d = Path(sys.argv[1])
    types_path = d / "Types.lean"
    funs_path = d / "Funs.lean"
    for p in (types_path, funs_path):
        if not p.is_file():
            print(f"error: {p} not found", file=sys.stderr)
            return 1

    types_text = types_path.read_text()
    funs_text = funs_path.read_text()

    blocks = parse_blocks(types_text, "type", 0)
    blocks += parse_blocks(funs_text, "fun", 10_000_000)

    by_name = {}
    for b in blocks:
        # keep the first definition of a name (defs are unique in practice)
        by_name.setdefault(b.name, b)

    names = set(by_name)

    for r in ROOTS:
        if r not in by_name:
            print(f"error: root {r!r} not found in generated Crypto Lean.", file=sys.stderr)
            return 1

    # Transitive closure: a block depends on every local declaration name that
    # appears as an identifier in its text (excluding its own name).
    def deps_of(block):
        found = set()
        for m in NAME_RE.finditer(block.text):
            tok = m.group(0)
            if tok in names and tok != block.name:
                found.add(tok)
            # also catch the leading prefix of a longer dotted projection like
            # `Foo.bar` where `Foo` is a local type: NAME_RE already grabs the
            # whole dotted token; add prefixes that are themselves declarations.
            parts = tok.split(".")
            for i in range(1, len(parts)):
                pref = ".".join(parts[:i])
                if pref in names and pref != block.name:
                    found.add(pref)
        return found

    closure = set()
    worklist = list(ROOTS)
    while worklist:
        nm = worklist.pop()
        if nm in closure:
            continue
        closure.add(nm)
        for dep in deps_of(by_name[nm]):
            if dep not in closure:
                worklist.append(dep)

    selected = sorted((by_name[n] for n in closure), key=lambda b: b.order)
    type_blocks = [b for b in selected if b.kind == "type"]
    fun_blocks = [b for b in selected if b.kind == "fun"]

    header = (
        "-- AUTO-GENERATED by proofs/scripts/carve_merkle_verify.py — DO NOT EDIT.\n"
        "-- The single-leaf Merkle `Proof::verify` subset, carved from the aeneas\n"
        "-- output by dependency closure so it compiles standalone (the full\n"
        "-- Crypto/Funs.lean does not, due to out-of-scope upstream-blocked code).\n"
        "-- Re-run the carve script after regenerating; no manual maintenance.\n"
        "import Aeneas\n"
        "open Aeneas Aeneas.Std Result ControlFlow Error\n"
        "set_option linter.dupNamespace false\n"
        "set_option linter.hashCommand false\n"
        "set_option linter.unusedVariables false\n"
        "set_option maxHeartbeats 1000000\n\n"
        "-- Missing Aeneas Std model: `usize::is_multiple_of` (Rust 1.87) is\n"
        "-- referenced by the generated code but not provided by this aeneas\n"
        "-- version's Std. Defined here computably (mirrors the Rust semantics)\n"
        "-- so the carved module is self-contained and the predicate is provable.\n"
        "def core.num.Usize.is_multiple_of (x : Std.Usize) (n : Std.Usize) :\n"
        "    Result Bool := ok (x.val % n.val == 0)\n\n"
        "-- `merkle_tree.merkle.ROOT = 0`. The generated def is an inline-attr\n"
        "-- one-liner (`@[global_simps, irreducible] def ROOT := 0#usize`) that the\n"
        "-- block-carver does not split cleanly, so it is supplied here verbatim.\n"
        "def merkle_tree.merkle.ROOT : Std.Usize := 0#usize\n\n"
        "-- Missing Aeneas Std model: `usize::ilog2` (-> u32). Used ONLY by\n"
        "-- `build_merkle_path` to size a `Vec::with_capacity` hint — the value has\n"
        "-- no effect on the result (capacity, not length), so it is modelled as the\n"
        "-- total constant `0` to keep the carve self-contained and panic-free.\n"
        "def core.num.Usize.ilog2 (_x : Std.Usize) : Result Std.U32 := ok 0#u32\n\n"
        "noncomputable section\n\n"
    )
    body = "".join(strip_trailing_doc(b.text).rstrip("\n") + "\n\n"
                   for b in type_blocks + fun_blocks)
    body += "end\n"
    out = d / "MerkleVerify.lean"
    out.write_text(header + body)
    print(
        f"carve_merkle_verify: wrote {out} "
        f"({len(type_blocks)} types + {len(fun_blocks)} funs; closure of {len(ROOTS)} roots)"
    )
    print("  closure: " + ", ".join(sorted(closure)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
