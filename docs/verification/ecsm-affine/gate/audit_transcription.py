"""A5 — the transcription audit: does the gate's model match the code it claims to model?

This is the pass the BLAKE3 campaign identified as highest-value and the EC campaign proved
the point with — its equivalent audit
(`thoughts/ec-recover-opt/gate/TRANSCRIPTION-AUDIT.md`, branch feat/ec-lincomb2) "found three
premises the gate asserted about the chip and never read, one of them hiding a working
forgery".

The dangerous direction is a model STRONGER than the object it models: it yields UNSAT where
the real chip is forgeable, and no positive anchor can catch it, because honest witnesses
satisfy a correct model and an over-strong one equally well. So every premise the lemma
scripts rely on is listed here as a `Premise` and CHECKED against the source text — not
re-derived, read.

Two kinds of entry:

  * **assumed** — the model relies on this being true of the code. Failing one invalidates
    a lemma.
  * **negative-space** — the model relies on something being ABSENT (e.g. "nothing constrains
    `yG`'s parity"). These are the ones a reader cannot verify by looking at what is written,
    so they are the ones most worth mechanising.

Each premise is also MUTATION-TESTED: the source is perturbed in memory and the premise must
then fail. A premise whose check passes on mutated source is checking nothing, which is the
failure mode this file exists to prevent.

Run: `python audit_transcription.py`
"""

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from affine_common import (  # noqa: E402
    ADDR_LIMB_BOUND_32B,
    ADDR_LIMB_BOUND_64B,
    CARRY_OFFSET_X2,
    CARRY_OFFSET_YG,
    ECSM_AFFINE_SYSCALL_NUMBER,
    ECSM_SYSCALL_NUMBER,
    INSTRUCTION_TS_STRIDE,
)


def find_repo_root(start=None):
    """Walk up until a directory looks like the lambda_vm root (a workspace `Cargo.toml`
    next to `prover/`).

    Deliberately NOT `parents[N]`. A hard-coded depth breaks silently when the campaign
    moves — and this audit reads repo source by path, so a wrong root would report premises
    against the wrong tree. (It happens to fail loudly here, because `load_sources` checks
    existence and names every missing file, but a marker walk removes the hazard instead of
    relying on the guard.)"""
    here = (start or Path(__file__)).resolve()
    for cand in here.parents:
        if (cand / "Cargo.toml").is_file() and (cand / "prover").is_dir():
            if "[workspace]" in (cand / "Cargo.toml").read_text():
                return cand
    raise RuntimeError(f"no lambda_vm repo root above {here}")


REPO = find_repo_root()
ECSM_RS = REPO / "prover" / "src" / "tables" / "ecsm.rs"
EXEC_RS = REPO / "executor" / "src" / "vm" / "instruction" / "execution.rs"
WITNESS_RS = REPO / "crypto" / "ecsm" / "src" / "witness.rs"
TEMPLATES_RS = REPO / "prover" / "src" / "constraints" / "templates.rs"
TRACE_BUILDER_RS = REPO / "prover" / "src" / "tables" / "trace_builder.rs"

results = []


def report(name, verdict, detail=""):
    results.append((name, verdict, detail))
    print(f"[{verdict:10}] {name}  {detail}")


@dataclass
class Premise:
    """One thing the gate assumes about the code."""
    key: str
    lemma: str            # which lemma consumes it
    what: str             # the assumption, in words
    check: object         # (sources) -> (ok, detail)
    kind: str = "assumed"  # or "negative-space"
    mutations: list = field(default_factory=list)  # (file, pattern, replacement, why)


def _src(sources, path):
    return sources[path]


# ── the premises ────────────────────────────────────────────────────────────

def p_columns(s):
    src = _src(s, ECSM_RS)
    got = {n: int(m) for n, m in re.findall(
        r"pub const (IS_AFFINE|YR_SUB_P|NUM_COLUMNS|MU): usize = (\d+);", src)}
    want = {"MU": 666, "IS_AFFINE": 667, "YR_SUB_P": 668, "NUM_COLUMNS": 684}
    ok = got == want and got["YR_SUB_P"] + 16 == got["NUM_COLUMNS"]
    return ok, (f"{got}; YR_SUB_P + 16 halfwords = {got.get('YR_SUB_P', 0) + 16} == "
                f"NUM_COLUMNS {got.get('NUM_COLUMNS')}")


def p_constraint_count(s):
    src = _src(s, ECSM_RS)
    m = re.search(r"debug_assert_eq!\(idx, (\d+)\);", src)
    ok = m is not None and int(m.group(1)) == 423
    # and the header index map mentions the new blocks
    ok &= "//   413..420 : CarryBit(YrLtP, 0..7)" in src
    ok &= "//   421      : IS_BIT(IS_AFFINE)" in src
    ok &= "//   422      : AffineZeroOnPadding" in src
    return ok, f"idx closes at {m.group(1) if m else '?'} == 423; header map lists 413..422"


def p_addr_bounds(s):
    src = _src(s, ECSM_RS)
    b32 = re.search(r"ADDR_LIMB_BOUND_32B: u64 = \(1 << 32\) - (\d+);", src)
    b64 = re.search(r"ADDR_LIMB_BOUND_64B: u64 = \(1 << 32\) - (\d+);", src)
    ok = (b32 and b64
          and (1 << 32) - int(b32.group(1)) == ADDR_LIMB_BOUND_32B
          and (1 << 32) - int(b64.group(1)) == ADDR_LIMB_BOUND_64B)
    return ok, (f"2^32−{b32.group(1) if b32 else '?'} / 2^32−{b64.group(1) if b64 else '?'} "
                "match the model's constants")


def p_syscall_numbers(s):
    src = _src(s, EXEC_RS)
    x = re.search(r"ECSM_SYSCALL_NUMBER: u64 = u64::MAX - (\d+);", src)
    a = re.search(r"ECSM_AFFINE_SYSCALL_NUMBER: u64 = u64::MAX - (\d+);", src)
    ok = (x and a
          and 2**64 - 1 - int(x.group(1)) == ECSM_SYSCALL_NUMBER
          and 2**64 - 1 - int(a.group(1)) == ECSM_AFFINE_SYSCALL_NUMBER)
    return ok, f"u64::MAX−{x.group(1) if x else '?'} / u64::MAX−{a.group(1) if a else '?'}"


def p_lowword_assert(s):
    """A1c's premise: the low-word inequality is guarded at COMPILE time, so a future
    variant cannot silently un-pin `IS_AFFINE`."""
    src = _src(s, EXEC_RS)
    ok = bool(re.search(
        r"const _: \(\) = assert!\(\s*ECSM_SYSCALL_NUMBER & 0xFFFF_FFFF"
        r"\s*!=\s*ECSM_AFFINE_SYSCALL_NUMBER & 0xFFFF_FFFF", src))
    return ok, "the low-32-bit-word inequality is a compile-time assert"


def p_syscall_word_linear(s):
    """A1c's other premise: the Ecall receiver's syscall words really are
    `xonly + IS_AFFINE·(affine − xonly)`, per word."""
    src = _src(s, ECSM_RS)
    ok = ("let syscall_word = |xonly: i64, affine: i64|" in src
          and "LinearTerm::Constant(xonly)" in src
          and "coefficient: affine - xonly," in src
          and "column: cols::IS_AFFINE," in src
          and "syscall_word(xonly_lo, affine_lo)" in src
          and "syscall_word(xonly_hi, affine_hi)" in src)
    return ok, "both received words are the IS_AFFINE interpolation of the two numbers"


def p_affine_gated_buses(s):
    """A1/A3's premise: the yG read and yR write fire with multiplicity IS_AFFINE, four
    doublewords each, at `+32 + 8i`, and the yR write at `ts + 3`."""
    src = _src(s, ECSM_RS)
    ok = "let affine = || Multiplicity::Column(cols::IS_AFFINE);" in src
    ok &= src.count("            affine(),") == 2          # one per for-loop body
    ok &= len(re.findall(r"LinearTerm::Constant\(\(32 \+ 8 \* i\) as i64\)", src)) == 2
    ok &= "memw_read(\n                dword_bytes(cols::YG, i)," in src
    ok &= "memw_write(\n                dword_bytes(cols::YR, i)," in src
    ok &= "ts_lo_plus(3)," in src
    return ok, ("2 affine-gated bus blocks (4 dwords each), offsets +32+8i, yG via "
                "memw_read at ts and yR via memw_write at ts+3")


def p_yrltp_wiring(s):
    """A2's premise: `OverflowKind::YrLtP` uses `P_BYTES` as the constant, `YR_SUB_P` as the
    halfword addend, `YR` as the byte-stored sum."""
    src = _src(s, ECSM_RS)
    ok = ("OverflowKind::YrLtP => &P_BYTES," in src
          and "OverflowKind::YrLtP => cols::YR_SUB_P," in src
          and "OverflowKind::YrLtP => cols::YR," in src
          and "fn sum_is_bits(self) -> bool {\n        matches!(self, OverflowKind::KLtN)" in src)
    return ok, "YrLtP → (P_BYTES, YR_SUB_P, YR), byte-stored sum (not bits)"


def p_yrltp_mu_gated(s):
    """A2d's premise, and the reason it is an OBSERVATION rather than a bug: `YrLtP` sits in
    the same `for kind in [...]` loop as the other three, so its constraints are µ-gated —
    NOT IS_AFFINE-gated. It therefore binds on x-only rows too."""
    src = _src(s, ECSM_RS)
    m = re.search(r"for kind in \[\s*OverflowKind::XgLtP,\s*OverflowKind::KLtN,\s*"
                  r"OverflowKind::XrLtP,\s*OverflowKind::YrLtP,\s*\] \{(.*?)\n        \}",
                  src, re.S)
    ok = m is not None
    if ok:
        body = m.group(1)
        ok = ("let mu = b.main(0, cols::MU);" in body
              and "mu * ci.clone() * (one - ci.clone())" in body
              and "mu * (one - c[7].clone())" in body
              and "cols::IS_AFFINE" not in body)
    return ok, ("all four chains share one µ-gated loop; IS_AFFINE does not appear in it ⇒ "
                "YrLtP binds x-only rows too (strictly stronger; A2d)")


def p_yr_sub_p_halfword_checks(s):
    """A2's C2 premise: the 16 `YR_SUB_P` halfwords are IsHalfword-checked, µ-gated."""
    src = _src(s, ECSM_RS)
    ok = bool(re.search(
        r"for i in 0\.\.16 \{\s*out\.push\(BusInteraction::sender\(\s*BusId::IsHalfword,"
        r"\s*mu\(\),\s*vec!\[packed\(cols::yr_sub_p\(i\)\)\],", src))
    return ok, "16 µ-gated IsHalfword sends on yr_sub_p(i)"


def p_alu_lt_senders(s):
    """A4's premise: three µ-gated `Alu` LT senders, xG/xR against the mode-dependent bound
    and k against the flat one, each asserting `result = 1` with a literal zero high word."""
    src = _src(s, ECSM_RS)
    ok = src.count("out.push(alu_lt(") == 3
    ok &= "out.push(alu_lt(packed(cols::ADDR_XG_0), addr_bound_by_mode()));" in src
    ok &= "out.push(alu_lt(packed(cols::ADDR_XR_0), addr_bound_by_mode()));" in src
    ok &= re.search(r"out\.push\(alu_lt\(\s*packed\(cols::ADDR_K_0\),\s*"
                    r"BusValue::constant\(ADDR_LIMB_BOUND_32B\),", src) is not None
    ok &= "BusValue::constant(alu_op::LT as u64)" in src
    ok &= re.search(r"let alu_lt = \|lhs_lo: BusValue, rhs_lo: BusValue\|", src) is not None
    return ok, "3 senders: xG/xR vs addr_bound_by_mode(), k vs the flat 32-byte bound, LT/1"


def p_executor_abi(s):
    """A4's premise about the executor arm: both 64-byte spans checked at 63, k at 31, and
    the overlap guard computed in u128."""
    src = _src(s, EXEC_RS)
    ok = bool(re.search(r"if !addr_limb_ok\(addr_xg, 63\)\s*\|\|\s*!addr_limb_ok\(addr_xr, 63\)"
                        r"\s*\|\|\s*!addr_limb_ok\(addr_k, 31\)", src))
    ok &= bool(re.search(r"if \(addr_k as u128\) < addr_xg as u128 \+ 64\s*"
                         r"&& \(addr_xg as u128\) < addr_k as u128 \+ 32", src))
    ok &= "load_u256_le(memory, addr_xg.wrapping_add(32))?" in src
    ok &= "store_u256_le(memory, addr_xr.wrapping_add(32), &yr)?" in src
    return ok, "spans 63/63/31, u128 overlap guard, yG at +32 in, yR at +32 out"


def p_witness_yr_sub_p(s):
    """A2e's premise: the honest `y_r_sub_p` is `(2^256 + yR − p) mod 2^256`, filled on BOTH
    paths (the shared `compute_witness_inner`)."""
    src = _src(s, WITNESS_RS)
    ok = "let y_r_sub_p = to_le_32(&((&two_256 + &result.y) - p()));" in src
    ok &= "fn compute_witness_inner(" in src
    ok &= "pub fn compute_witness_with_y(" in src
    # both public entry points funnel into the shared inner fn
    ok &= src.count("compute_witness_inner(k_le, k, g)") == 2
    return ok, "y_r_sub_p = (2^256 + yR − p), computed in the shared inner fn used by both paths"


def p_carry_offsets(s):
    src = _src(s, ECSM_RS)
    x2 = re.search(r"CARRY_OFFSET_X2: i64 = (\d+);", src)
    yg = re.search(r"CARRY_OFFSET_YG: i64 = (\d+);", src)
    ok = (x2 and yg and int(x2.group(1)) == CARRY_OFFSET_X2
          and int(yg.group(1)) == CARRY_OFFSET_YG)
    return ok, f"X2 {x2.group(1) if x2 else '?'} / YG {yg.group(1) if yg else '?'}"


def p_inv_shift_32(s):
    src = _src(s, TEMPLATES_RS)
    m = re.search(r"INV_SHIFT_32: u64 = (\d+);", src)
    ok = m is not None and int(m.group(1)) == 18446744065119617026
    return ok, f"INV_SHIFT_32 = {m.group(1) if m else '?'} = 2^-32 mod p_g"


# ── negative-space premises ────────────────────────────────────────────────

def p_no_parity_constraint(s):
    """A3's central premise, and pure negative space: NOTHING in the chip constrains `yG`'s
    parity. If some constraint did, A3b's forgery would be blocked by it and A3d's
    "load-bearing" verdict would be wrong.

    Checked by enumerating every appearance of `cols::YG` and confirming each is one of the
    four known, parity-blind uses. A new appearance fails the audit — which is the point:
    the premise stops being true silently otherwise."""
    src = _src(s, ECSM_RS)
    lines = {i + 1: ln for i, ln in enumerate(src.splitlines()) if "cols::YG" in ln}
    allowed = {
        "table.set_bytes(row_idx, cols::YG, &w.y_g);": "trace fill",
        "dword_bytes(cols::YG, i),": "the affine yG MEMW read (the fix itself)",
        "is_byte(cols::YG, 32, &mut out);": "AreBytes range check (parity-blind)",
        "cols::YG,": "Ecdas seed/drain bus tuples (parity-blind)",
        "s = s + byte(cols::YG, 32, j) * byte(cols::YG, 32, i - j);":
            "the Yg relation's yG² term — satisfied by BOTH roots",
    }
    unknown = {ln: txt.strip() for ln, txt in lines.items()
               if txt.strip() not in allowed}
    ok = not unknown and len(lines) == 7
    return ok, (f"{len(lines)} uses of cols::YG, all parity-blind "
                f"({', '.join(sorted(set(allowed.values())))})"
                if ok else f"UNRECOGNISED uses: {unknown}")


def p_yr_not_byte_checked(s):
    """A2d's premise (contract C4-YR), also negative space: `ecsm.rs` does NOT byte-check
    `YR`, so `YrLtP`'s byte hypothesis is inherited through the Ecdas bus rather than emitted
    locally. If a future commit adds `is_byte(cols::YR, 32, ...)`, C4-YR stops being a
    contract and becomes a local fact — better, but the audit should notice."""
    src = _src(s, ECSM_RS)
    checked = set(re.findall(r"is_byte\(cols::(\w+), \d+, &mut out\);", src))
    ok = checked == {"X2", "Q0", "YG", "Q1"} and "YR" not in checked
    return ok, (f"ecsm.rs byte-checks {sorted(checked)}; YR (and XR) are NOT among them ⇒ "
                "C4-YR is a contract, not a local emission")


def p_ts_stride(s):
    """A4f's premise: one instruction consumes 4 sub-timestamps, so `ts + 3` is free.

    Read from the CODE, not from ecsm.rs's comment claiming it. The builder assigns
    `timestamp = i·4 + 4` per CPU op, which IS the stride; and the ECSM interactions use
    exactly offsets {0, +1, +2, +3}, so `ts + 3` is the last slot and nothing spills into the
    next instruction. An earlier form of this premise matched the comment text and was
    therefore checking documentation rather than behaviour."""
    tb = _src(s, TRACE_BUILDER_RS)
    src = _src(s, ECSM_RS)
    # PARSE the stride; do not compare the source against a hard-coded 4, or the check is
    # blind to a change in the very number it is about (this is what the mutation control
    # caught on the first version of this premise).
    strides = {int(m) for m in re.findall(r"let timestamp = \(i as u64\) \* (\d+) \+ \d+;", tb)}
    offsets = {int(m) for m in re.findall(r"ts_lo_plus\((\d+)\)", src)}
    ok = len(strides) == 1 and strides == {INSTRUCTION_TS_STRIDE}
    ok &= offsets == {1, 2, 3}          # plus the bare `ts_lo()` reads at offset 0
    ok &= bool(strides) and max(offsets) == max(strides) - 1
    return ok, (f"builder stride parsed as {sorted(strides)} (model says "
                f"{INSTRUCTION_TS_STRIDE}); ECSM uses ts offsets {{0}} ∪ {sorted(offsets)}, "
                f"max {max(offsets)} == stride−1 ⇒ ts+3 is free and nothing spills")


PREMISES = [
    Premise("P1 column layout", "A1/A2", "IS_AFFINE=667, YR_SUB_P=668..684, NUM_COLUMNS=684",
            p_columns,
            mutations=[(ECSM_RS, r"pub const NUM_COLUMNS: usize = 684;",
                        "pub const NUM_COLUMNS: usize = 683;", "off-by-one column count")]),
    Premise("P2 constraint count + index map", "A1", "idx closes at 423; 413..422 documented",
            p_constraint_count,
            mutations=[(ECSM_RS, r"debug_assert_eq!\(idx, 423\);",
                        "debug_assert_eq!(idx, 421);", "wrong constraint total")]),
    Premise("P3 address-limb bounds", "A4", "2^32−31 and 2^32−63", p_addr_bounds,
            mutations=[(ECSM_RS, r"ADDR_LIMB_BOUND_64B: u64 = \(1 << 32\) - 63;",
                        "ADDR_LIMB_BOUND_64B: u64 = (1 << 32) - 64;", "off-by-one bound")]),
    Premise("P4 syscall numbers", "A1c", "u64::MAX−10 and u64::MAX−11", p_syscall_numbers,
            mutations=[(EXEC_RS, r"ECSM_AFFINE_SYSCALL_NUMBER: u64 = u64::MAX - 11;",
                        "ECSM_AFFINE_SYSCALL_NUMBER: u64 = u64::MAX - 12;",
                        "different affine number")]),
    Premise("P5 low-word compile-time assert", "A1c",
            "the pinning's only separating word is guarded at compile time", p_lowword_assert,
            mutations=[(EXEC_RS, r"const _: \(\) = assert!\(",
                        "const _UNUSED: () = ((), assert!(", "assert removed/renamed")]),
    Premise("P6 syscall word is linear in IS_AFFINE", "A1c",
            "both received words interpolate the two numbers", p_syscall_word_linear,
            mutations=[(ECSM_RS, r"coefficient: affine - xonly,",
                        "coefficient: 0,", "coefficient zeroed ⇒ selector unpinned")]),
    Premise("P7 affine-gated bus layout", "A1/A3",
            "4+4 IS_AFFINE-gated dwords at +32+8i, yR at ts+3", p_affine_gated_buses,
            mutations=[(ECSM_RS, r"LinearTerm::Constant\(\(32 \+ 8 \* i\) as i64\)",
                        "LinearTerm::Constant((8 * i) as i64)", "offset +32 dropped")]),
    Premise("P8 YrLtP wiring", "A2", "YrLtP → (P_BYTES, YR_SUB_P, YR), byte-stored",
            p_yrltp_wiring,
            mutations=[(ECSM_RS, r"OverflowKind::YrLtP => cols::YR,",
                        "OverflowKind::YrLtP => cols::XR,", "sum column swapped to XR")]),
    Premise("P9 YrLtP is µ-gated, not IS_AFFINE-gated", "A2d",
            "the four chains share one µ-gated loop", p_yrltp_mu_gated,
            mutations=[(ECSM_RS, r"let mu = b\.main\(0, cols::MU\);\n                let one = b\.one\(\);\n                b\.emit_base\(idx, mu \* ci\.clone\(\)",
                        "let mu = b.main(0, cols::IS_AFFINE);\n                let one = b.one();\n                b.emit_base(idx, mu * ci.clone()",
                        "carry bits re-gated on IS_AFFINE")]),
    Premise("P10 YR_SUB_P halfword checks", "A2", "16 µ-gated IsHalfword sends",
            p_yr_sub_p_halfword_checks,
            mutations=[(ECSM_RS, r"vec!\[packed\(cols::yr_sub_p\(i\)\)\],",
                        "vec![packed(cols::xr_sub_p(i))],", "halfword checks aimed at XR")]),
    Premise("P11 Alu LT senders", "A4", "3 senders with the right bounds and LT/1",
            p_alu_lt_senders,
            mutations=[(ECSM_RS, r"out\.push\(alu_lt\(packed\(cols::ADDR_XR_0\), addr_bound_by_mode\(\)\)\);",
                        "", "xR's address bound removed")]),
    Premise("P12 executor ABI arm", "A4", "spans 63/63/31 and the u128 overlap guard",
            p_executor_abi,
            mutations=[(EXEC_RS, r"if \(addr_k as u128\) < addr_xg as u128 \+ 64",
                        "if (addr_k as u64) < addr_xg as u64 + 64",
                        "u128 widening reverted to the wrapping form")]),
    Premise("P13 honest y_r_sub_p", "A2e", "(2^256 + yR − p), shared by both paths",
            p_witness_yr_sub_p,
            mutations=[(WITNESS_RS, r"let y_r_sub_p = to_le_32\(&\(\(&two_256 \+ &result\.y\) - p\(\)\)\);",
                        "let y_r_sub_p = to_le_32(&((&two_256 + &result.x) - p()));",
                        "addend built from x instead of y")]),
    Premise("P14 carry offsets", "A3", "8160 / 16319", p_carry_offsets,
            mutations=[(ECSM_RS, r"CARRY_OFFSET_YG: i64 = 16319;",
                        "CARRY_OFFSET_YG: i64 = 16320;", "offset perturbed")]),
    Premise("P15 INV_SHIFT_32", "A2a", "2^-32 mod p_g", p_inv_shift_32,
            mutations=[(TEMPLATES_RS, r"INV_SHIFT_32: u64 = 18446744065119617026;",
                        "INV_SHIFT_32: u64 = 18446744065119617025;", "inverse perturbed")]),
    Premise("P16 nothing constrains yG's parity", "A3", "every cols::YG use is parity-blind",
            p_no_parity_constraint, kind="negative-space",
            mutations=[(ECSM_RS, r"is_byte\(cols::YG, 32, &mut out\);",
                        "is_byte(cols::YG, 32, &mut out);\n    let _parity = cols::YG;",
                        "an unrecognised cols::YG use appears")]),
    Premise("P17 YR is not byte-checked in ecsm.rs", "A2d",
            "C4-YR is inherited, not emitted", p_yr_not_byte_checked, kind="negative-space",
            mutations=[(ECSM_RS, r"is_byte\(cols::Q1, 33, &mut out\);",
                        "is_byte(cols::Q1, 33, &mut out);\n    is_byte(cols::YR, 32, &mut out);",
                        "YR gains a local byte check")]),
    Premise("P18 instruction timestamp stride", "A4f", "4 sub-timestamps per instruction",
            p_ts_stride,
            mutations=[(TRACE_BUILDER_RS, r"let timestamp = \(i as u64\) \* 4 \+ 4;",
                        "let timestamp = (i as u64) * 3 + 3;",
                        "stride reduced to 3 ⇒ ts+3 collides with the next instruction")]),
]


# ── the audit ──────────────────────────────────────────────────────────────

def load_sources():
    paths = (ECSM_RS, EXEC_RS, WITNESS_RS, TEMPLATES_RS, TRACE_BUILDER_RS)
    missing = [p for p in paths if not p.exists()]
    if missing:
        return None, missing
    return {p: p.read_text() for p in paths}, []


def main():
    sources, missing = load_sources()
    if sources is None:
        report("audit", "FAIL", f"missing sources: {missing}")
        return 1

    n_ok = 0
    for pr in PREMISES:
        ok, detail = pr.check(sources)
        n_ok += ok
        tag = "READ" if pr.kind == "assumed" else "READ(neg)"
        report(f"{pr.key} [{pr.lemma}]", tag if ok else "FAIL", detail)

    print()
    # Mutation testing: each premise must FAIL on perturbed source, or it checks nothing.
    blind = []
    n_mut = 0
    for pr in PREMISES:
        for path, pattern, repl, why in pr.mutations:
            mutated = dict(sources)
            new, count = re.subn(pattern, repl, mutated[path], count=1)
            if count == 0:
                blind.append(f"{pr.key}: mutation pattern did not apply ({why})")
                continue
            mutated[path] = new
            ok, _ = pr.check(mutated)
            n_mut += 1
            if ok:
                blind.append(f"{pr.key}: survives mutation '{why}' ⇒ the check is BLIND")
    report("mutation testing", "PROVED" if not blind else "FAIL",
           f"{n_mut} mutations applied; every premise's check fails on its mutant"
           if not blind else "; ".join(blind))

    print()
    failed = [n for n, v, _ in results if v == "FAIL"]
    print(f"TRANSCRIPTION AUDIT: {n_ok}/{len(PREMISES)} premises read from source, "
          f"{n_mut} mutation controls, {len(failed)} failures")
    if failed:
        print("  FAILURES: " + ", ".join(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
