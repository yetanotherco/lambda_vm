"""Shared model for the lincomb2 (ECSM2/ECDAS2) z3 gate — phase E.

Layered on `gate_common.py`: the three convolution relations are BYTE-FOR-BYTE
the old chip's (proved mechanically by `relation_bodies_identical()` below), so
the S_i builders, interval machinery and constants are imported rather than
re-transcribed. What is new here is the SCHEDULE model — phases, selectors,
digit streams — and the chip-state detector.

## Why there is a chip-state detector

Two soundness fixes were outstanding when this gate was written:

  * `(1−MU)·D1 = 0` / `(1−MU)·D2 = 0` — closes the padding-row phantom-digit
    forgery (`../lincomb2/L6-COUNTING.md`);
  * `D_INV·(xB − xA) ≡ 1 (mod p)` — closes the degenerate-add forgery
    (`../lincomb2/FINDING-nums-blinding.log`).

Until they land, negative controls 1 and 2 are not *ablations* — they are LIVE
HOLES, and the gate must say so rather than quietly reporting the expected SAT.
`chip_state()` reads the chip source at run time and reports which are present,
so this file keeps telling the truth as `ecdas2.rs` changes underneath it.

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

# The four Addend-bus selectors (ecsm2.rs `SEL_*`).
SEL_P1, SEL_P2, SEL_P12, SEL_CORRECTION = 1, 2, 3, 4


# ── mechanical port argument ────────────────────────────────────────────────

def _fn_body(path, fn):
    """The full text of `fn <name><B: ...>` including its brace-matched body."""
    s = path.read_text()
    i = s.index(f"fn {fn}<B:")
    j = s.index("{", i)
    depth, k = 0, j
    while True:
        if s[k] == "{":
            depth += 1
        elif s[k] == "}":
            depth -= 1
        if depth == 0:
            break
        k += 1
    return s[i:k + 1]


def _norm(t):
    """Strip comments/whitespace and undo the documented XB/YB rename."""
    t = re.sub(r"//.*", "", t)
    t = re.sub(r"\s+", " ", t)
    return t.replace("cols::XB", "cols::XG").replace("cols::YB", "cols::YG").strip()


def _match_arm(path, variant):
    """The `Relation::<variant> => { … }` arm of `s_i`, brace-matched."""
    s = _fn_body(path, "s_i")
    i = s.index(f"Relation::{variant} => {{")
    j = s.index("{", i)
    depth, k = 0, j
    while True:
        if s[k] == "{":
            depth += 1
        elif s[k] == "}":
            depth -= 1
        if depth == 0:
            break
        k += 1
    return s[j:k + 1]


def relation_bodies_identical():
    """L1/L2a/L3/L4/L5a port argument, checked rather than asserted.

    Every lemma of the original gate that speaks only about the three original
    convolution relations transfers verbatim IF those relations are built by the
    same expressions. Compared PER ARM, because ECDAS2 has since gained a fourth
    relation (`Dinv`) — an additive block that cannot affect a lemma quantified
    over the other three. The shared helpers are compared whole.

    Returns {name: bool}.
    """
    out = {}
    for variant in ("Lambda", "Xr", "Yr"):
        out[f"s_i::{variant}"] = (_norm(_match_arm(ECDAS1, variant))
                                  == _norm(_match_arm(ECDAS2, variant)))
    for fn in ("rq", "p_byte_expr", "r_byte_expr", "byte_at"):
        out[fn] = _norm(_fn_body(ECDAS1, fn)) == _norm(_fn_body(ECDAS2, fn))
    return out


# ── chip-state detection ────────────────────────────────────────────────────

def chip_state():
    """What is actually in the chip right now. Read, not assumed."""
    ecdas2 = ECDAS2.read_text()
    ecsm2 = ECSM2.read_text()

    # The padding gate. Detected from the CONSTRAINT MAP rather than by pattern-
    # matching the emitting expression, which the chip may spell as a loop: the
    # header documents each index, so look for `(1 − MU)` applied to the two
    # digit columns. Both the comment form and a `(one - mu)`/`(1 - mu)` product
    # are accepted.
    pad_gate = bool(
        re.search(r"\(1\s*[−-]\s*MU\)\s*·\s*\{?[^}\n]*D1", ecdas2)
        or re.search(r"(one|1)\s*-\s*mu\s*\)?\s*\*\s*d1", ecdas2)
        or re.search(r"d1\s*\*\s*\(\s*(one|1)\s*-\s*mu", ecdas2)
    )
    # The non-degeneracy relation: a D_INV column block plus its relation arm.
    dinv = "D_INV" in ecdas2 and "Dinv" in ecdas2

    n_cols = int(re.search(r"pub const NUM_COLUMNS: usize = (\d+);", ecdas2).group(1))
    n_constraints = None
    m = re.search(r"debug_assert_eq!\(idx, (\d+)\);", ecdas2)
    if m:
        n_constraints = int(m.group(1))

    # JointBit send multiplicity — the L6 break's mechanism.
    jb = re.search(r"BusId::JointBit,\s*\n\s*(Multiplicity::[A-Za-z]+)", ecdas2)
    jb_mult = jb.group(1) if jb else "?"

    return {
        "ecdas2_columns": n_cols,
        "ecdas2_constraints": n_constraints,
        "padding_digit_gate": pad_gate,
        "dinv_relation": dinv,
        "jointbit_multiplicity": jb_mult,
        "ecsm2_columns": int(
            re.search(r"pub const NUM_COLUMNS: usize = (\d+);", ecsm2).group(1)
        ),
    }


def print_chip_state(state=None):
    st = state or chip_state()
    print("chip state (read from prover/src/tables/ at run time):")
    print(f"   ECDAS2 : {st['ecdas2_columns']} columns, "
          f"{st['ecdas2_constraints']} constraints")
    print(f"   ECSM2  : {st['ecsm2_columns']} columns")
    print(f"   JointBit send multiplicity      : {st['jointbit_multiplicity']}")
    print(f"   (1−MU)·D padding gate present   : {st['padding_digit_gate']}")
    print(f"   D_INV non-degeneracy present    : {st['dinv_relation']}")
    return st


# ── ECDAS2 schedule model (z3) ──────────────────────────────────────────────

def bit_var(s, name):
    import z3
    v = z3.Int(name)
    s.add(z3.Or(v == 0, v == 1))
    return v


class Ecdas2Row:
    """The schedule columns of one ECDAS2 row, with `ecdas2.rs` idx 11..=21.

    `ablate` names a constraint to DROP (negative controls); `extra` enables
    constraints that are proposed but may not have landed yet.
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

    def __init__(self, s, tag, ablate=(), padding_gate=False):
        self.tag = tag
        for nm in ("mu", "op", "nb", "d1", "d2", "s1", "s2", "s3", "sc", "ph1", "ph2"):
            setattr(self, nm, bit_var(s, f"{nm.upper()}_{tag}"))
        import z3
        self.round = z3.Int(f"ROUND_{tag}")
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
