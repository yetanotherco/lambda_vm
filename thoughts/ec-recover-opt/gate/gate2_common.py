"""Shared model for the lincomb2 (ECSM2/ECDAS2) z3 gate — phase E.

Layered on `gate_common.py`: the three convolution relations are BYTE-FOR-BYTE
the old chip's (proved mechanically by `relation_bodies_identical()` below), so
the S_i builders, interval machinery and constants are imported rather than
re-transcribed. What is new here is the SCHEDULE model — phases, selectors,
digit streams — and the chip-state detector.

## Why there is a chip-state detector

Two soundness fixes were outstanding when this gate was first written — the
padding-row digit gate and the non-degeneracy relation. **Both have since
landed** (`ecdas2.rs` idx 22..=27 and the `Relation::Dinv` block). The detector
stays, because its job was never "report the state on the day this was written":
it is what stops a negative control from being scored as a passing *ablation*
when the defence it ablates is actually missing. A gate whose controls go SAT
for the real reason, and record it as an expected result, is worse than no gate.

## What the detector must NOT be

`TRANSCRIPTION-AUDIT.md` F1/F2 documented the failure mode this file used to
have: every predicate was a *token or comment* match. Deleting the emitting loop
while leaving its header comment still reported the defence present; narrowing
the `D_INV` gate by one term — which yields a working forgery on the correction
row — changed no verdict anywhere on the board.

So every predicate here is now parsed from the **emitted expression** or the
**multiplicity expression**, and every parser **fails closed**: an unrecognised
shape reports the defence ABSENT with a reason, never present. Two structural
invariants are checked in both directions:

  * `padding_gate_state()` — the set of columns carrying `(1 − MU)·X = 0` must
    EQUAL the set of columns supplying a multiplicity in
    `ecdas2::bus_interactions()`. This is the invariant the chip header claims
    ("which column supplies its multiplicity, and what forces that column to
    zero?"), and it catches a *new ungated multiplicity* — the original JointBit
    bug — from the side that bug actually appeared on.
  * `dinv_gate_state()` — the `Relation::Dinv` arm's gate expression must be
    exactly the term list of the `Addend` receive's `Multiplicity::Linear`.
    That is what `RESULTS-lincomb2.md` §2 asserts in prose ("it ties the check
    to the very expression that counts the Addend receive").

Column indices are deliberately NOT hardcoded anywhere that matters: the model
is written against constraint identities and bus multiplicities.
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from gate_common import (  # noqa: F401  (re-exported for the phase-E scripts)
    B, BIT, BYTE, GEN_X, GEN_Y, Iv, N, OFF, P, PG, P_BYTES, R3P, R_BYTES,
    compose, decompose, s_ecdas_lambda, s_ecdas_xr, s_ecdas_yr,
)

REPO = Path(__file__).resolve().parents[3]
ECDAS2 = REPO / "prover/src/tables/ecdas2.rs"
ECSM2 = REPO / "prover/src/tables/ecsm2.rs"
ECDAS1 = REPO / "prover/src/tables/ecdas.rs"
ECSM1 = REPO / "prover/src/tables/ecsm.rs"

# The four Addend-bus selectors (ecsm2.rs `SEL_*`).
SEL_P1, SEL_P2, SEL_P12, SEL_CORRECTION = 1, 2, 3, 4


# ── source extraction primitives ────────────────────────────────────────────

def _match_from(s, i, opener="{", closer="}"):
    """Brace/bracket-matched span starting at the first `opener` at or after i."""
    j = s.index(opener, i)
    depth, k = 0, j
    while True:
        if s[k] == opener:
            depth += 1
        elif s[k] == closer:
            depth -= 1
        if depth == 0:
            return j, k
        k += 1


def _fn_body_text(src, fn):
    """The full text of `fn <name>` including its brace-matched body.

    Accepts both the generic (`fn s_i<B: …>`) and plain (`fn bus_interactions()`)
    declaration forms.
    """
    for marker in (f"fn {fn}<B:", f"fn {fn}("):
        i = src.find(marker)
        if i != -1:
            _, k = _match_from(src, i)
            return src[i:k + 1]
    raise ValueError(f"no `fn {fn}` in this source")


def _fn_body(path, fn):
    return _fn_body_text(path.read_text(), fn)


def _norm(t):
    """Strip comments/whitespace and undo the documented XB/YB rename."""
    t = re.sub(r"//.*", "", t)
    t = re.sub(r"\s+", " ", t)
    return t.replace("cols::XB", "cols::XG").replace("cols::YB", "cols::YG").strip()


def _match_arm_text(src, variant, fn="s_i"):
    """The `Relation::<variant> => { … }` arm of `fn`, brace-matched."""
    s = _fn_body_text(src, fn)
    i = s.index(f"Relation::{variant} => {{")
    j, k = _match_from(s, i)
    return s[j:k + 1]


def _match_arm(path, variant):
    return _match_arm_text(path.read_text(), variant)


def _prologue(src, fn="s_i"):
    """The part of `fn` before its `match relation {` — the operand bindings.

    `TRANSCRIPTION-AUDIT.md` F3: rebinding an operand here (e.g. `xa` to
    `cols::XR`) falsifies every value lemma while leaving all three arms
    textually identical, so this must be compared too.
    """
    body = _fn_body_text(src, fn)
    return body[: body.index("match relation")]


def _strip_dinv_arm(t):
    """Remove ECDAS2's extra `Relation::Dinv => cols::…,` dispatch line."""
    return re.sub(r"\s*Relation::Dinv\s*=>\s*cols::\w+,", "", t)


# ── mechanical port arguments ───────────────────────────────────────────────

def relation_bodies_identical():
    """L1/L2a/L3/L4/L5a port argument, checked rather than asserted.

    Every lemma of the original gate that speaks only about the three original
    convolution relations transfers verbatim IF those relations are built by the
    same expressions. Compared PER ARM, because ECDAS2 has since gained a fourth
    relation (`Dinv`) — an additive block that cannot affect a lemma quantified
    over the other three.

    Covers, since `TRANSCRIPTION-AUDIT.md` F3:

      * the three arms (as before);
      * the `s_i` PROLOGUE, i.e. every operand → column binding. Without this
        the check passes on a chip whose relations read the wrong columns;
      * `conv_carry`, with ECDAS2's `Dinv` dispatch line excised — this is the
        carry recurrence L1 telescopes;
      * the four shared helpers, compared whole.

    Returns {name: bool}.
    """
    src1, src2 = ECDAS1.read_text(), ECDAS2.read_text()
    out = {}
    for variant in ("Lambda", "Xr", "Yr"):
        out[f"s_i::{variant}"] = (_norm(_match_arm_text(src1, variant))
                                  == _norm(_match_arm_text(src2, variant)))
    out["s_i::prologue"] = _norm(_prologue(src1)) == _norm(_prologue(src2))
    out["conv_carry"] = (_norm(_fn_body_text(src1, "conv_carry"))
                         == _norm(_strip_dinv_arm(_fn_body_text(src2, "conv_carry"))))
    for fn in ("rq", "p_byte_expr", "r_byte_expr", "byte_at"):
        out[fn] = _norm(_fn_body_text(src1, fn)) == _norm(_fn_body_text(src2, fn))
    return out


# ECSM2's membership relations are ECSM's, modulo the column rename and the
# `µ → OK` gate swap. The soundness theorem's "P2 is on the curve" clause rests
# on this port, and nothing checked it (`TRANSCRIPTION-AUDIT.md`, gap 2).
_MEMBERSHIP_RENAME = (
    ("cols::MEM_X2", "cols::X2"),
    ("cols::MEM_Q0", "cols::Q0"),
    ("cols::MEM_Q1", "cols::Q1"),
    ("cols::MEM_C0", "cols::C0"),
    ("cols::MEM_C1", "cols::C1"),
    ("cols::X_P2", "cols::XG"),
    ("cols::Y_P2", "cols::YG"),
    ("cols::OK", "cols::MU"),
)


def _norm_membership(t):
    t = re.sub(r"//.*", "", t)
    for a, b in _MEMBERSHIP_RENAME:
        t = t.replace(a, b)
    t = re.sub(r"\bok\b", "mu", t)          # the local binding, after cols::OK
    t = re.sub(r"\s+", " ", t)
    return t.strip()


def membership_bodies_identical():
    """ECSM2's `Relation::X2`/`Yg` are ECSM's, modulo rename + `µ → OK`.

    The gate swap is not cosmetic — it is what lets an ECSM2 error row
    (`MU = 1, OK = 0`) close at zero carries like a padding row — but it is
    sound because `OK` is IS_BIT and `OK·(1 − MU) = 0` (`ecsm2.rs` idx 1, 2), so
    `OK = 1` rows are a subset of `MU = 1` rows and every ECSM lemma applies to
    them verbatim.

    `carry_chain` is deliberately NOT compared: ECSM2 has five `OverflowKind`s
    to ECSM's three, so the bodies differ by design. Its five overflow checks
    are covered by L8 N3 / N7 instead.

    Returns {name: bool}.
    """
    src1, src2 = ECSM1.read_text(), ECSM2.read_text()
    out = {}
    for variant in ("X2", "Yg"):
        out[f"s_i::{variant}"] = (_norm_membership(_match_arm_text(src1, variant))
                                  == _norm_membership(_match_arm_text(src2, variant)))
    out["s_i::prologue"] = (_norm_membership(_prologue(src1))
                            == _norm_membership(_prologue(src2)))
    for fn in ("conv_carry", "p_byte_expr", "byte_at"):
        out[fn] = (_norm_membership(_fn_body_text(src1, fn))
                   == _norm_membership(_fn_body_text(src2, fn)))
    return out


# ── multiplicity extraction (the F2 invariant) ──────────────────────────────

_MULT_RE = re.compile(r"Multiplicity::(Column|Sum3|Sum|Diff|Negated|Linear)\s*\(")
_FOR_TUPLE_RE = re.compile(r"for\s*\(([^)]*)\)\s*in\s*\[")


def _resolve_loop_ident(body, pos, ident):
    """`cols::` names bound to `ident` by the nearest enclosing `for (…) in […]`.

    `Multiplicity::Column(col)` inside
    `for (stream, col) in [(1u64, cols::D1), (2u64, cols::D2)]` supplies TWO
    multiplicity columns; a regex for `cols::` alone would miss both.
    """
    best = None
    for m in _FOR_TUPLE_RE.finditer(body, 0, pos):
        if ident in [p.strip() for p in m.group(1).split(",")]:
            best = m
    if best is None:
        return None
    j, k = _match_from(body, best.end() - 1, "[", "]")
    return set(re.findall(r"cols::(\w+)", body[j:k + 1]))


def multiplicity_columns(src, fn="bus_interactions"):
    """Every column that supplies a bus multiplicity, parsed from the source.

    Returns `(columns, unresolved)`. `unresolved` lists multiplicity expressions
    the parser did not understand — a non-empty list must be read as "the
    invariant could not be checked", never as "no extra columns".
    """
    body = _fn_body_text(src, fn)
    cols, unresolved = set(), []
    for m in _MULT_RE.finditer(body):
        kind = m.group(1)
        j, k = _match_from(body, m.end() - 1, "(", ")")
        arg = body[j + 1:k]
        if kind == "Linear":
            found = re.findall(r"column:\s*cols::(\w+)", arg)
            cols.update(found)
            n_terms = len(re.findall(r"LinearTerm::Column\w*\s*\{", arg))
            if n_terms != len(found):
                unresolved.append(f"Linear with {n_terms - len(found)} non-`cols::` terms")
            continue
        for piece in arg.split(","):
            piece = piece.strip()
            if not piece:
                continue
            direct = re.fullmatch(r"cols::(\w+)", piece)
            if direct:
                cols.add(direct.group(1))
                continue
            if re.fullmatch(r"\w+", piece):
                resolved = _resolve_loop_ident(body, m.start(), piece)
                if resolved:
                    cols.update(resolved)
                else:
                    unresolved.append(f"{kind}({piece})")
            else:
                unresolved.append(f"{kind}({piece[:40]})")
    return cols, unresolved


def padding_gated_columns(src):
    """Columns X for which the chip emits `(1 − MU)·X = 0`, from the EXPRESSION.

    Returns `(columns, reason)`. `reason` is None on success; on any
    unrecognised shape the column set is empty and `reason` says why — the
    detector fails CLOSED, so an un-parsed chip is reported undefended.
    """
    body = _fn_body_text(src, "eval")
    cols = set()

    # (a) unrolled form: `b.emit_base(i, (one - mu) * b.main(0, cols::X))`
    cols.update(re.findall(
        r"\(\s*(?:one|1)\s*-\s*mu\s*\)\s*\*\s*b\.main\(0,\s*cols::(\w+)\)", body))
    cols.update(re.findall(
        r"b\.main\(0,\s*cols::(\w+)\)\s*\*\s*\(\s*(?:one|1)\s*-\s*mu\s*\)", body))

    # (b) loop form, which is what the chip uses:
    #     for (i, col) in [cols::…] { let x = b.main(0, col);
    #                                 b.emit_base(22 + i, (one - mu) * x); }
    loop_reason = None
    for m in re.finditer(r"for\s*(\(?[^)\n]*\)?)\s*in\s*\[", body):
        try:
            bj, bk = _match_from(body, m.end() - 1, "[", "]")
            hj, hk = _match_from(body, bk)
        except (ValueError, IndexError):
            continue
        loop_body = body[hj:hk + 1]
        if "emit_base" not in loop_body:
            continue
        prod = re.search(r"\(\s*(?:one|1)\s*-\s*mu\s*\)\s*\*\s*(\w+)", loop_body)
        if not prod:
            continue
        var = prod.group(1)
        bind = re.search(rf"let\s+{re.escape(var)}\s*=\s*b\.main\(0,\s*(\w+)\)", loop_body)
        if not bind:
            loop_reason = (f"`(1 - mu) * {var}` in a loop, but `{var}` is not bound "
                           f"by `b.main(0, <loop var>)`")
            continue
        bindings = [p.strip() for p in m.group(1).strip("()").split(",")]
        if bind.group(1) not in bindings:
            loop_reason = (f"`(1 - mu) * {var}` reads `{bind.group(1)}`, not a "
                           f"binding of this loop {bindings}")
            continue
        found = set(re.findall(r"cols::(\w+)", body[bj:bk + 1]))
        if not found:
            loop_reason = "gate loop header lists no `cols::` columns"
            continue
        cols.update(found)

    if not cols:
        return set(), (loop_reason or "no `(1 - mu) * <column>` emission found")
    return cols, None


def padding_gate_state(src=None):
    """The F2 invariant: gated columns == raw-multiplicity columns.

    `MU` is excluded from the multiplicity side: it gates itself by
    construction, and `(1 − MU)·MU = 0` is implied by `IS_BIT(MU)`.
    """
    src = ECDAS2.read_text() if src is None else src
    gated, reason = padding_gated_columns(src)
    mult, unresolved = multiplicity_columns(src)
    raw = mult - {"MU"}
    return {
        "present": bool(gated) and reason is None,
        "reason": reason,
        "gated": gated,
        "raw_multiplicity": raw,
        "unresolved_multiplicities": unresolved,
        "exact": (not unresolved) and reason is None and gated == raw,
        "ungated_multiplicities": raw - gated,
        "gated_non_multiplicities": gated - raw,
    }


# ── the D_INV gate (the F1 invariant) ───────────────────────────────────────

def emitted_relations(src):
    """`(Relation, carry-column)` pairs the `eval` body actually emits."""
    return set(re.findall(r"\(Relation::(\w+),\s*cols::(\w+)\)",
                          _fn_body_text(src, "eval")))


def addend_multiplicity_columns(src):
    """The term list of the `Addend` receive's `Multiplicity::Linear`.

    Returns `(columns, problem)`. Every term must be a unit-coefficient plain
    column: a `coefficient: 2` would double the receive without changing the
    column set, so the set alone is not enough to characterise the multiplicity.
    """
    body = _fn_body_text(src, "bus_interactions")
    i = body.index("BusId::Addend")
    m = _MULT_RE.search(body, i)
    if not m or m.group(1) != "Linear":
        return set(), f"Addend receive multiplicity is {m and m.group(1)}, not Linear"
    j, k = _match_from(body, m.end() - 1, "(", ")")
    arg = body[j:k + 1]
    terms = re.findall(
        r"LinearTerm::Column\w*\s*\{\s*coefficient:\s*(-?\d+),\s*column:\s*cols::(\w+)",
        arg)
    n_all = len(re.findall(r"LinearTerm::\w+", arg))
    if len(terms) != n_all:
        return set(), f"{n_all - len(terms)} Addend term(s) are not plain `cols::` columns"
    bad = [c for coeff, c in terms if coeff != "1"]
    if bad:
        return set(), f"Addend terms with coefficient != 1: {sorted(bad)}"
    return {c for _, c in terms}, None


def _is_plain_column_sum(expr):
    """Is `expr` literally `b.main(0, cols::A) + b.main(0, cols::B) + …`?

    Set equality on the `cols::` names is NOT enough: an opaque factor
    (`gate_expr(b) * b.main(0, cols::S1) + …`) leaves the set unchanged while
    letting the whole relation be gated off on rows of the tamperer's choosing.
    Found by `audit_transcription.py` §G against the first version of this
    check, which compared sets only.
    """
    norm = re.sub(r"\s+", " ", expr).strip()
    terms = [t.strip() for t in norm.split("+")]
    return all(re.fullmatch(r"b\.main\(0, cols::\w+\)", t) for t in terms)


def dinv_gate_state(src=None):
    """The F1 invariant: `Relation::Dinv`'s gate == the Addend receive's terms.

    Narrowing the gate by one term (dropping `S_CORR`) leaves the correction row
    with no non-degeneracy check, which is a working forgery reachable by one
    point subtraction — `TRANSCRIPTION-AUDIT.md` F1. Nothing used to read this
    expression, so no verdict on the board moved.
    """
    src = ECDAS2.read_text() if src is None else src
    state = {"present": False, "reason": None, "gate": set(), "addend": set(),
             "emitted": False, "applied": False, "matches_addend": False,
             "plain_sum": False}

    state["emitted"] = any(r == "Dinv" for r, _ in emitted_relations(src))
    try:
        arm = _match_arm_text(src, "Dinv")
    except ValueError:
        state["reason"] = "no `Relation::Dinv` arm in `s_i`"
        return state

    m = re.search(r"let\s+g\s*=\s*(.*?);", arm, re.S)
    if not m:
        state["reason"] = "no `let g = …;` gate expression in the Dinv arm"
        return state
    expr = re.sub(r"//.*", "", m.group(1))
    state["gate"] = set(re.findall(r"cols::(\w+)", expr))
    state["plain_sum"] = _is_plain_column_sum(expr)
    state["applied"] = re.search(r"\bg\s*\*\s*s\b", arm) is not None
    addend, addend_problem = addend_multiplicity_columns(src)
    state["addend"] = addend
    state["matches_addend"] = bool(state["gate"]) and state["gate"] == addend

    if not state["emitted"]:
        state["reason"] = "the Dinv block is never emitted by `eval`"
    elif not state["applied"]:
        state["reason"] = "the gate expression `g` does not multiply the relation"
    elif not state["plain_sum"]:
        state["reason"] = ("the gate is not a plain sum of columns: "
                           f"`{re.sub(chr(10) + r'\s*', ' ', expr).strip()[:90]}`")
    elif addend_problem:
        state["reason"] = addend_problem
    elif not state["matches_addend"]:
        state["reason"] = (f"gate {sorted(state['gate'])} != Addend receive "
                           f"{sorted(state['addend'])}")
    state["present"] = state["reason"] is None
    return state


# ── the JointSel → PH*/S* mapping (RESULTS §7's modelled step) ──────────────

_ARM_RE = re.compile(r"((?:JointSel::\w+\s*\|?\s*)+)=>\s*\(([^)]*)\)")


def _sel_map(src, fn):
    """`{JointSel variant: tuple}` for a `match self.step.sel` mapping fn."""
    out = {}
    for m in _ARM_RE.finditer(_fn_body_text(src, fn)):
        bits = tuple(int(x.strip()) for x in m.group(2).split(","))
        for v in re.findall(r"JointSel::(\w+)", m.group(1)):
            out[v] = bits
    return out


def joint_sel_maps(src=None):
    """`Ecdas2Operation::phase_bits` / `selector_bits`, parsed from the chip.

    `PH*`/`S*` are not fields of `Lincomb2Witness`; the chip derives them from
    `JointSel`, and `positive_real_witness2.py` reproduces that derivation by
    hand. `RESULTS-lincomb2.md` §7 flagged the hand copy as the anchor's one
    remaining modelled step — this is what makes it machine-checked instead.

    Also returns the enum's variant list, so a NEW variant (which would be a
    silent gap in the Python dicts) is visible rather than merely absent.
    """
    src = ECDAS2.read_text() if src is None else src
    witness = (REPO / "crypto/ecsm/src/witness.rs").read_text()
    i = witness.index("pub enum JointSel {")
    j, k = _match_from(witness, i)
    variants = re.findall(r"^\s*(\w+),", witness[j:k + 1], re.M)
    return {
        "phase_bits": _sel_map(src, "phase_bits"),
        "selector_bits": _sel_map(src, "selector_bits"),
        "variants": variants,
    }


# ── chip-state detection ────────────────────────────────────────────────────

def chip_state(ecdas2_src=None, ecsm2_src=None):
    """What is actually in the chip right now. Parsed, not pattern-matched.

    `padding_digit_gate` and `dinv_relation` keep their names and meanings for
    the phase-E scripts, but are now derived from the emitted expressions, carry
    the two structural invariants, and fail closed.
    """
    ecdas2 = ECDAS2.read_text() if ecdas2_src is None else ecdas2_src
    ecsm2 = ECSM2.read_text() if ecsm2_src is None else ecsm2_src

    pad = padding_gate_state(ecdas2)
    dinv = dinv_gate_state(ecdas2)

    n_cols = int(re.search(r"pub const NUM_COLUMNS: usize = (\d+);", ecdas2).group(1))
    m = re.search(r"debug_assert_eq!\(idx, (\d+)\);", ecdas2)
    n_constraints = int(m.group(1)) if m else None

    jb = re.search(r"BusId::JointBit,\s*\n\s*(Multiplicity::[A-Za-z]+)", ecdas2)
    jb_mult = jb.group(1) if jb else "?"

    return {
        "ecdas2_columns": n_cols,
        "ecdas2_constraints": n_constraints,
        "padding_digit_gate": pad["present"],
        "padding_gate_detail": pad,
        "dinv_relation": dinv["present"],
        "dinv_gate_detail": dinv,
        "jointbit_multiplicity": jb_mult,
        "ecsm2_columns": int(
            re.search(r"pub const NUM_COLUMNS: usize = (\d+);", ecsm2).group(1)
        ),
    }


def print_chip_state(state=None):
    st = state or chip_state()
    pad, dinv = st["padding_gate_detail"], st["dinv_gate_detail"]
    print("chip state (PARSED from prover/src/tables/ at run time):")
    print(f"   ECDAS2 : {st['ecdas2_columns']} columns, "
          f"{st['ecdas2_constraints']} constraints")
    print(f"   ECSM2  : {st['ecsm2_columns']} columns")
    print(f"   JointBit send multiplicity      : {st['jointbit_multiplicity']}")
    print(f"   (1−MU)·X padding gate present   : {st['padding_digit_gate']}"
          + ("" if st["padding_digit_gate"] else f"   [{pad['reason']}]"))
    print(f"      gated columns                : {sorted(pad['gated'])}")
    print(f"      raw bus multiplicities       : {sorted(pad['raw_multiplicity'])}")
    print(f"      sets are EQUAL               : {pad['exact']}")
    if pad["ungated_multiplicities"]:
        print("      *** UNGATED MULTIPLICITY *** : "
              f"{sorted(pad['ungated_multiplicities'])}")
    if pad["gated_non_multiplicities"]:
        print("      (gated but not a multiplicity, harmless): "
              f"{sorted(pad['gated_non_multiplicities'])}")
    if pad["unresolved_multiplicities"]:
        print("      *** UNPARSED MULTIPLICITY ** : "
              f"{pad['unresolved_multiplicities']}")
    print(f"   D_INV non-degeneracy present    : {st['dinv_relation']}"
          + ("" if st["dinv_relation"] else f"   [{dinv['reason']}]"))
    print(f"      gate expression columns      : {sorted(dinv['gate'])}")
    print(f"      Addend receive multiplicity  : {sorted(dinv['addend'])}")
    print(f"      gate == Addend receive       : {dinv['matches_addend']}")
    return st


# ── ECDAS2 schedule model (z3) ──────────────────────────────────────────────

def bit_var(s, name):
    import z3
    v = z3.Int(name)
    s.add(z3.Or(v == 0, v == 1))
    return v


class Ecdas2Row:
    """The schedule columns of one ECDAS2 row, with `ecdas2.rs` idx 11..=21.

    `ablate` names a constraint to DROP (negative controls); `padding_gate`
    toggles idx 22..=27.

    `padding_gate` defaults to **True**, i.e. to the chip as it is. It used to
    default to False — a model strictly WEAKER than the chip on `MU = 0` rows,
    which produced one spurious forgery (`TRANSCRIPTION-AUDIT.md` F4). Ablations
    must now pass `padding_gate=False` explicitly.
    """

    SCHEDULE_IDX = {
        11: "PH1·PH2",
        12: "OP·NB",
        13: "(1−OP)(NB−D1−D2+D1·D2)",
        14: "OP−ΣS",
        15: "(1−PH1)·D1",
        16: "(1−PH1)·D2",
        17: "PH1·S_CORR",
        18: "PH1·(S1+S3−OP·D1)",
        19: "PH1·(S2+S3−OP·D2)",
        20: "MU·(1−PH1−PH2)·(S2−1)",
        21: "PH2·(S_CORR−1)",
    }

    def __init__(self, s, tag, ablate=(), padding_gate=True):
        self.tag = tag
        for nm in ("mu", "op", "nb", "d1", "d2", "s1", "s2", "s3", "sc", "ph1", "ph2"):
            setattr(self, nm, bit_var(s, f"{nm.upper()}_{tag}"))
        import z3
        self.round = z3.Int(f"ROUND_{tag}")
        # AreBytes(ROUND) is MU-gated, so this is asserted on padding rows where
        # the chip does NOT enforce it (`TRANSCRIPTION-AUDIT.md` F5). Verified
        # benign: with idx 22..=27 a MU=0 row's ROUND reaches no bus at all.
        s.add(self.round >= 0, self.round <= 255)

        def add(idx, expr):
            if idx not in ablate:
                s.add(expr == 0)

        add(11, self.ph1 * self.ph2)
        add(12, self.op * self.nb)
        add(13, (1 - self.op) * (self.nb - self.d1 - self.d2 + self.d1 * self.d2))
        add(14, self.op - self.s1 - self.s2 - self.s3 - self.sc)
        add(15, (1 - self.ph1) * self.d1)
        add(16, (1 - self.ph1) * self.d2)
        add(17, self.ph1 * self.sc)
        add(18, self.ph1 * (self.s1 + self.s3 - self.op * self.d1))
        add(19, self.ph1 * (self.s2 + self.s3 - self.op * self.d2))
        add(20, self.mu * (1 - self.ph1 - self.ph2) * (self.s2 - 1))
        add(21, self.ph2 * (self.sc - 1))

        # idx 22..=27: (1 − MU)·x = 0 for every column that is a bus
        # MULTIPLICITY — the two digit sends and the four Addend selectors.
        # Ablated as the unit "pad".
        if padding_gate and "pad" not in ablate:
            for col in (self.d1, self.d2, self.s1, self.s2, self.s3, self.sc):
                s.add((1 - self.mu) * col == 0)

    def digit_send(self, stream):
        """JointBit send multiplicity — `Multiplicity::Column(D1/D2)`, ungated."""
        return self.d1 if stream == 1 else self.d2

    def addend_receive(self):
        """Addend receive multiplicity — S1+S2+S3+S_CORR, also ungated."""
        return self.s1 + self.s2 + self.s3 + self.sc


# ── real lincomb2 witnesses via the oracle harness ──────────────────────────

HARNESS = Path(__file__).resolve().parents[1] / "oracle/repo-harness/target/release/ecsm-oracle-harness"


def lincomb2_witness(u1, u2, p1, p2):
    """One `ecsm::lincomb2_witness` dump, as a dict. Raises on the error path."""
    import json
    import subprocess
    line = f"lincomb2 {u1:x} {u2:x} {p1[0]:x} {p1[1]:x} {p2[0]:x} {p2[1]:x}\n"
    r = subprocess.run([str(HARNESS)], input=line, capture_output=True,
                       text=True, check=True)
    out = r.stdout.strip()
    if not out.startswith("lincomb2_json "):
        raise RuntimeError(f"harness rejected: {out[:80]}")
    return json.loads(out[len("lincomb2_json "):])
