"""
Executable half of the transcription audit: does the gate model the Rust that
was actually written?

The gate (`dma-chip/z3_dma_verify.py`) proves things about a MODEL. Everything
it proves is worthless if the model and `prover/src/tables/dma.rs` have drifted,
and the dangerous drift direction is a model STRONGER than the object it
models -- it yields UNSAT where the real table is forgeable, and no positive
anchor can catch it, because honest inputs satisfy a correct model and an
over-strong one equally well. (In the EC campaign the equivalent audit found
three premises the gate asserted about the chip and never read, one of them
hiding a working forgery.)

So this script reads the Rust and asserts, textually and structurally:

  A. CONSTANTS   -- every number the oracle and gate hard-code appears in the
                    Rust with that value.
  B. COLUMNS     -- the column layout the gate assumes is the layout `dma::cols`
                    declares, including `NUM_COLUMNS`.
  C. CONSTRAINTS -- each constraint index the gate models is emitted, at that
                    index, by the template the gate modelled, with the operands
                    the gate used; and no constraint index exists that the gate
                    does not model.
  D. BUSES       -- the 23 interactions, their bus ids, their multiplicities and
                    the wiring facts the gate explicitly CANNOT see: that the
                    source read and the destination write reference the SAME
                    `value` columns, that their timestamp offsets are +1 and +2,
                    that `w8 = 1 - tail` on both, and that a read carries
                    `old == value`.
  E. EXECUTOR    -- the ecall validates what the oracle validates, in that order.
  F. GENERATOR   -- `generate_dma_trace`'s padding row is the row the oracle's
                    `padding_columns()` describes.

It is deliberately textual (regex over the source) rather than a Rust test: the
point is to catch a change in `dma.rs` that nobody reflected here, and a Rust
test would be edited in the same commit as the code it guards.

    python3 audit_gate_transcription.py [--repo /path/to/lambda_vm]
"""

import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
#: This script lives at `docs/verification/dma/`, so the repo root is three up.
DEFAULT_REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))

sys.path.insert(0, os.path.join(HERE, "dma-oracle"))
sys.path.insert(0, os.path.join(HERE, "dma-chip"))
import dma_ref as ref                                             # noqa: E402

#: Gate constants this audit cross-checks, duplicated rather than imported:
#: importing `z3_dma_verify` drags in `z3`, and EVERY claim here is textual and
#: needs no solver -- so a machine without z3 can still run the whole audit.
#: The duplication is the cost of that, and it is a real one: nothing detects
#: drift between these values and the gate's. They are all derived constants
#: (Goldilocks, its 2^32 inverse, the executor's chunk bound), so drift would
#: mean one of the two is simply wrong, and `audit_constants` checks each against
#: the Rust independently -- which is the property that matters.
GATE_P = 2**64 - 2**32 + 1
GATE_INV_2_32 = pow(2**32, -1, GATE_P)
GATE_MAX_BYTES = 256
GATE_ZERO_SUM = 4 * 65535
GATE_ZERO_DOMAIN = 2**20


class Audit:
    """A findings collector. Nothing raises; everything is reported."""

    def __init__(self):
        self.checks = 0
        self.findings = []

    def ok(self, claim, condition, detail=""):
        self.checks += 1
        if not condition:
            self.findings.append((claim, detail))
        return condition

    def report(self):
        print("=" * 76)
        print(f"{self.checks} claims checked, {len(self.findings)} finding(s)")
        print("=" * 76)
        for claim, detail in self.findings:
            print(f"  FINDING  {claim}")
            if detail:
                print(f"           {detail}")
        if not self.findings:
            print("  no drift between the Rust, the oracle and the gate")
        return not self.findings


def read(repo, relative):
    """Read a source file, whitespace-normalised for literal matching.

    Two things this fixes. (a) ENCODING: every Rust file here contains non-ASCII
    (em-dashes), and the locale default is not always UTF-8, so a bare `open()`
    can die with `UnicodeDecodeError` under `LC_ALL=C` with coercion disabled.
    (b) FORMATTING: the literal checks below match source fragments like
    `if tail { 1 } else { 8 }`, and `rustfmt` reflows those across lines the
    moment a line grows past `max_width`. An earlier version tried
    `src.replace("\n", " ")`, which collapses the newline but leaves the
    indentation, so it could never match a reflowed form -- the guard was dead
    code and a purely cosmetic reformat produced spurious findings. Since this
    script is meant to run in CI, a spurious red is how it gets deleted.

    Collapsing all runs of whitespace to one space makes every literal check
    reflow-insensitive. Line-oriented claims use `read_raw` instead.
    """
    return re.sub(r"\s+", " ", read_raw(repo, relative))


def read_raw(repo, relative):
    """The file verbatim, for claims that depend on line structure."""
    path = os.path.join(repo, relative)
    with open(path, encoding="utf-8") as f:
        return f.read()


# ---------------------------------------------------------------------------
# A. Constants
# ---------------------------------------------------------------------------

def audit_constants(a, repo):
    execution = read(repo, "executor/src/vm/instruction/execution.rs")
    dma = read(repo, "prover/src/tables/dma.rs")
    templates = read(repo, "prover/src/constraints/templates.rs")
    syscalls = read(repo, "syscalls/src/syscalls.rs")

    m = re.search(r"pub const DMA_MEMCPY_MAX_BYTES:\s*u64\s*=\s*(\d+)", execution)
    a.ok("DMA_MEMCPY_MAX_BYTES matches the oracle and gate", m and
         int(m.group(1)) == ref.DMA_MEMCPY_MAX_BYTES == GATE_MAX_BYTES,
         f"rust={m.group(1) if m else '?'} oracle={ref.DMA_MEMCPY_MAX_BYTES} gate={GATE_MAX_BYTES}")

    m = re.search(r"pub const DMA_MEMCPY_SYSCALL_NUMBER:\s*u64\s*=\s*u64::MAX\s*-\s*(\d+)", execution)
    a.ok("DMA_MEMCPY_SYSCALL_NUMBER is u64::MAX - 2", m and
         (2**64 - 1 - int(m.group(1))) == ref.DMA_MEMCPY_SYSCALL_NUMBER)

    m = re.search(r"const DMA_MEMCPY_MAX_BYTES:\s*usize\s*=\s*(\d+)", syscalls)
    a.ok("the guest stub's chunk bound equals the executor's", m and
         int(m.group(1)) == ref.DMA_MEMCPY_MAX_BYTES,
         "a stub that chunks larger than the executor accepts would abort the guest")

    m = re.search(r"pub const INV_SHIFT_32:\s*u64\s*=\s*(\d+)", templates)
    a.ok("INV_SHIFT_32 is the true inverse of 2^32 mod p, and the gate has it",
         m and int(m.group(1)) == GATE_INV_2_32
         and (int(m.group(1)) * 2**32) % GATE_P == 1)

    # The table takes its bound FROM the executor rather than restating it --
    # the property that makes the AIR bound and the execution bound un-driftable.
    a.ok("dma.rs re-exports the executor's bound instead of restating it",
         "DMA_MEMCPY_MAX_BYTES as EXECUTOR_DMA_MEMCPY_MAX_BYTES" in dma
         and re.search(r"pub const DMA_MEMCPY_MAX_BYTES:\s*u64\s*=\s*"
                       r"EXECUTOR_DMA_MEMCPY_MAX_BYTES", dma) is not None)

    a.ok("the Zero sender's constant is 4 * 65535, as the gate assumes",
         "LinearTerm::Constant(4 * 65535)" in dma
         and GATE_ZERO_SUM == 4 * 65535)

    # The Zero receiver's domain: bitwise packs x + 256y + 65536z with z 4 bits.
    bitwise = read(repo, "prover/src/tables/bitwise.rs")
    a.ok("the Zero send stays inside the bitwise table's ZERO domain",
         "65536 * z" in bitwise.replace("65536 * cols::Z", "65536 * z")
         or "coefficient: 65536" in bitwise,
         "the receiver packs x + 256y + 65536z with z < 16, i.e. arguments < 2^20")
    a.ok("4 * 65535 fits that domain", GATE_ZERO_SUM < GATE_ZERO_DOMAIN)

    a.ok("the row widths the gate uses are the widths dma.rs uses",
         "if tail { 1 } else { 8 }" in dma.replace("\n", " ")
         or re.search(r"let width = if tail \{ 1 \} else \{ 8 \}", dma) is not None,
         f"gate uses {1}/{8}")
    a.ok("the AIR's step expression is 8 - 7*tail",
         "AddLinearTerm::Constant(8)" in dma and "coefficient: -7" in dma)


# ---------------------------------------------------------------------------
# B. Column layout
# ---------------------------------------------------------------------------

EXPECTED_COLUMNS = {
    "TIMESTAMP_0": 0, "TIMESTAMP_1": 1,
    "SRC_0": 2, "SRC_1": 3,
    "SRC_INCR_0": 4, "SRC_INCR_1": 5, "SRC_INCR_2": 6, "SRC_INCR_3": 7,
    "DST_0": 8, "DST_1": 9,
    "DST_INCR_0": 10, "DST_INCR_1": 11, "DST_INCR_2": 12, "DST_INCR_3": 13,
    "COUNT_0": 14, "COUNT_1": 15,
    "COUNT_DECR_0": 16, "COUNT_DECR_1": 17, "COUNT_DECR_2": 18, "COUNT_DECR_3": 19,
    "FIRST": 20, "END": 21, "TAIL": 22, "VALUE_0": 23, "MU": 31,
    "NUM_COLUMNS": 32,
}


def audit_columns(a, repo):
    dma = read(repo, "prover/src/tables/dma.rs")
    declared = {m.group(1): int(m.group(2)) for m in
                re.finditer(r"pub const (\w+):\s*usize\s*=\s*(\d+);", dma)}
    for name, index in EXPECTED_COLUMNS.items():
        a.ok(f"column {name} is at {index}", declared.get(name) == index,
             f"declared at {declared.get(name)}")
    a.ok("VALUE is the eight columns starting at VALUE_0",
         re.search(r"pub const VALUE:\s*\[usize;\s*8\]", dma) is not None
         and dma.count("VALUE_0 +") == 7)
    # Every column the gate models, and nothing more. `mu` at 31 with `value`
    # at 23..30 means the layout is exactly full: 32 columns, none spare.
    a.ok("the layout is dense: 24 named + 8 value = NUM_COLUMNS",
         declared.get("NUM_COLUMNS") == declared.get("MU") + 1 == 32)


# ---------------------------------------------------------------------------
# C. Constraints
# ---------------------------------------------------------------------------

#: (index, what the gate models at that index)
EXPECTED_CONSTRAINTS = [
    (0, "emit_is_bit FIRST"),
    (1, "emit_is_bit END"),
    (2, "emit_is_bit TAIL"),
    (3, "emit_is_bit MU"),
    (4, "(first + end) * (1 - mu)"),
    (5, "emit_add_pair_no_overflow src + step = src_incr"),
    (7, "emit_add_pair_no_overflow dst + step = dst_incr"),
    (9, "emit_add_pair count_decr + step = count"),
    (11, "tail * value[i] for i in 1..8"),
]


def audit_constraints(a, repo):
    dma = read(repo, "prover/src/tables/dma.rs")
    body = dma.split("impl ConstraintSet")[1]

    a.ok("idx 0-3 are the four booleanity constraints, in the gate's order",
         re.search(r"emit_is_bit\(b, 0, cols::FIRST", body) and
         re.search(r"emit_is_bit\(b, 1, cols::END", body) and
         re.search(r"emit_is_bit\(b, 2, cols::TAIL", body) and
         re.search(r"emit_is_bit\(b, 3, cols::MU", body))

    a.ok("idx 4 is (first + end) * (1 - mu)",
         re.search(r"emit_base\(4,\s*\(first \+ end\) \* \(one - mu\)\)", body) is not None,
         "the gate rewrites this as Implies(mu == 0, first == 0 and end == 0)")

    a.ok("idx 5 is the NO-OVERFLOW add on src, gated by (MU, END)",
         re.search(r"emit_add_pair_no_overflow\(\s*b,\s*5,\s*cols::MU,\s*cols::END,",
                   body) is not None)
    a.ok("idx 7 is the NO-OVERFLOW add on dst, gated by (MU, END)",
         re.search(r"emit_add_pair_no_overflow\(\s*b,\s*7,\s*cols::MU,\s*cols::END,",
                   body) is not None)
    a.ok("idx 9 is the PLAIN add on count (wrap permitted, unconditional)",
         re.search(r"emit_add_pair\(\s*b,\s*9,\s*&\[\],", body) is not None,
         "the gate relies on this being the plain form: the terminal row holds 0 - 1")

    a.ok("src/dst adds read src as DWordWL and src_incr as DWordHL",
         "AddOperand::dword(cols::SRC_0)" in body
         and "AddOperand::from_dword_hl(cols::SRC_INCR_0)" in body
         and "AddOperand::dword(cols::DST_0)" in body
         and "AddOperand::from_dword_hl(cols::DST_INCR_0)" in body)
    a.ok("the count add has count_decr on the LHS and count as the SUM",
         re.search(r"emit_add_pair\(\s*b,\s*9,\s*&\[\],\s*"
                   r"&AddOperand::from_dword_hl\(cols::COUNT_DECR_0\),\s*"
                   r"&step,\s*&AddOperand::dword\(cols::COUNT_0\),", body) is not None,
         "reversing it would make count_decr the sum and break the terminal row")

    a.ok("idx 11..17 zero the seven unused value lanes on a tail row",
         re.search(r"emit_base\(11 \+ i - 1,\s*tail\.clone\(\) \* b\.main\(0, column\)\)",
                   body) is not None
         and ".skip(1)" in body)

    # No constraint index outside what the gate models.
    emitted = sorted({int(m.group(1)) for m in re.finditer(r"emit_base\((\d+)", body)}
                     | {int(m.group(1)) for m in
                        re.finditer(r"emit_is_bit\(b, (\d+)", body)}
                     | {int(m.group(1)) for m in
                        re.finditer(r"emit_add_pair(?:_no_overflow)?\(\s*b,\s*(\d+)", body)})
    a.ok("DmaConstraints does not raise max_degree above the default 2",
         "fn max_degree" not in body,
         "the gate's encoding rewrites `boolean * expr` products as implications, "
         "which is exact only while every such product has a boolean factor -- a "
         "degree-3 constraint would mean that rewrite lost something")

    a.ok("no constraint index exists that the gate does not model",
         emitted == [0, 1, 2, 3, 4, 5, 7, 9, 11],
         f"emitted anchors: {emitted}; the pairs also occupy 6, 8, 10 and the "
         f"lane loop 12..17")


# ---------------------------------------------------------------------------
# D. Buses -- including the wiring the gate cannot see
# ---------------------------------------------------------------------------

def audit_buses(a, repo):
    dma = read(repo, "prover/src/tables/dma.rs")
    buses = dma.split("pub fn bus_interactions")[1].split("/// An `IsHalfword`")[0]

    # The twelve IsHalfword sends are built by the `halfword()` helper below the
    # list, so they appear as calls rather than as literal `BusInteraction::`s.
    inline = buses.count("BusInteraction::")
    via_helper = len(re.findall(r"\bhalfword\(cols::\w+\)", buses))
    a.ok("there are 23 bus interactions", inline + via_helper == 23,
         f"found {inline} inline + {via_helper} via halfword() = {inline + via_helper}")

    counts = {bus: len(re.findall(rf"BusId::{bus}\b", buses)) for bus in
              ("Ecall", "DmaNext", "Zero", "Memw", "Alu")}
    counts["IsHalfword"] = via_helper
    a.ok("bus mix is 1 Ecall, 2 DmaNext, 12 IsHalfword, 1 Zero, 5 Memw, 2 Alu",
         counts == {"Ecall": 1, "DmaNext": 2, "IsHalfword": 12, "Zero": 1,
                    "Memw": 5, "Alu": 2}, str(counts))

    a.ok("the Ecall interaction is a RECEIVER with multiplicity `first`",
         re.search(r"BusInteraction::receiver\(\s*BusId::Ecall,\s*"
                   r"Multiplicity::Column\(cols::FIRST\)", buses) is not None)
    a.ok("DmaNext sends with `mu - end` and receives with `mu - first`",
         "let mu_minus_end = Multiplicity::Diff(cols::MU, cols::END);" in dma
         and "let mu_minus_first = Multiplicity::Diff(cols::MU, cols::FIRST);" in dma
         and re.search(r"sender\(\s*BusId::DmaNext,\s*mu_minus_end", buses)
         and re.search(r"receiver\(\s*BusId::DmaNext,\s*mu_minus_first", buses))

    a.ok("both DmaNext tuples carry the timestamp",
         buses.split("BusId::DmaNext")[1].count("TIMESTAMP_0") == 1
         and buses.split("BusId::DmaNext")[2].count("TIMESTAMP_0") == 1,
         "without it, rows of two different ecalls could be spliced -- the "
         "exact hole the BLAKE3 design review found in its internal bus")

    a.ok("the send carries the INCREMENTED triple and the receive the plain one",
         all(name in buses.split("BusId::DmaNext")[1] for name in
             ("SRC_INCR_0", "DST_INCR_0", "COUNT_DECR_0"))
         and all(name in buses.split("BusId::DmaNext")[2] for name in
                 ("SRC_0", "DST_0", "COUNT_0")))

    a.ok("all twelve IsHalfword sends are on count_decr, src_incr and dst_incr",
         sorted(re.findall(r"halfword\(cols::(\w+)\)", buses)) ==
         sorted([f"COUNT_DECR_{i}" for i in range(4)]
                + [f"SRC_INCR_{i}" for i in range(4)]
                + [f"DST_INCR_{i}" for i in range(4)]),
         "these are the range checks MAIN 1 and the width audit prove "
         "load-bearing; losing one is a forgeable end flag or a wrapped address")
    a.ok("IsHalfword sends have multiplicity `mu`",
         re.search(r"BusId::IsHalfword,\s*Multiplicity::Column\(cols::MU\)", dma)
         is not None)

    a.ok("the Zero send has multiplicity `mu` and pairs the sum with END",
         re.search(r"sender\(\s*BusId::Zero,\s*Multiplicity::Column\(cols::MU\)",
                   buses) is not None
         and "column: cols::COUNT_DECR_3" in buses
         and "start_column: cols::END" in buses)
    a.ok("all four count_decr halfwords enter the Zero sum with coefficient -1",
         len(re.findall(r"coefficient: -1,\s*column: cols::COUNT_DECR_\d", buses)) == 4)

    a.ok("the three register reads are x10=dst, x11=src, x12=count",
         re.search(r"memw_register_read\(20, cols::DST_0, cols::DST_1\)", buses)
         and re.search(r"memw_register_read\(22, cols::SRC_0, cols::SRC_1\)", buses)
         and re.search(r"memw_register_read\(24, cols::COUNT_0, cols::COUNT_1\)", buses),
         "base_address = 2*reg; these are the sends REG-32 is discharged by")
    a.ok("register reads have multiplicity `first`",
         len(re.findall(r"BusId::Memw,\s*Multiplicity::Column\(cols::FIRST\)", buses)) == 3)

    a.ok("the tail LT lookup is count vs 8 with output `tail`, multiplicity mu",
         re.search(r"BusId::Alu,\s*Multiplicity::Column\(cols::MU\)", buses)
         and "BusValue::constant(8)" in buses
         and "start_column: cols::TAIL" in buses)
    a.ok("the bound LT lookup is count vs MAX+1 with output pinned to 1, "
         "multiplicity first",
         "BusValue::constant(DMA_MEMCPY_MAX_BYTES + 1)" in buses
         and re.search(r"BusId::Alu,\s*Multiplicity::Column\(cols::FIRST\)", buses)
         is not None)

    # ---- the wiring facts the gate explicitly cannot see -------------------
    read_tuple = buses.split("// 22. MEMW read")[1].split("// 23.")[0]
    write_tuple = buses.split("// 23. MEMW write")[1]

    a.ok("the read tuple carries value_columns() TWICE (old and value)",
         read_tuple.count("value_columns()") == 1
         and "tuple.extend(values.iter().cloned())" in read_tuple
         and "tuple.append(&mut values)" in read_tuple,
         "old == value is what makes the source read non-mutating")
    a.ok("the write tuple carries the SAME value_columns()",
         "tuple.extend(value_columns())" in write_tuple,
         "THIS is why a copied byte cannot change: one set of columns feeds "
         "both memory tuples, so the gate never has to prove read == write")
    a.ok("value_columns() is exactly cols::VALUE, packed Direct",
         re.search(r"fn value_columns\(\).*?cols::VALUE \.iter\(\)"
                   r".*?packing: Packing::Direct", dma) is not None,
         "note the source is whitespace-normalised by `read`, so this pattern "
         "matches the reflow-insensitive form")

    a.ok("the read is at T+1 and the write at T+2",
         "timestamp_with_offset(1)" in read_tuple
         and "timestamp_with_offset(2)" in write_tuple,
         "all reads strictly before all writes is what gives an overlapping "
         "copy snapshot semantics; the gate cannot see timestamps")
    a.ok("timestamp_with_offset only offsets the LOW limb",
         re.search(r"fn timestamp_with_offset.*?cols::TIMESTAMP_0.*?"
                   r"LinearTerm::Constant\(offset\)", dma, re.S) is not None
         and read_tuple.count("cols::TIMESTAMP_1") == 1,
         "a +1/+2 that could carry into the high limb would break ordering")

    a.ok("both data tuples set w2 = 0, w4 = 0 and w8 = 1 - tail",
         read_tuple.count("BusValue::constant(0)") >= 2
         and write_tuple.count("BusValue::constant(0)") >= 2
         and read_tuple.count("column: cols::TAIL") == 1
         and write_tuple.count("column: cols::TAIL") == 1,
         "w8 = 1 - tail is the only link between the width the AIR proves and "
         "the number of bytes the memory table moves")
    a.ok("both data tuples have multiplicity `mu - end`",
         len(re.findall(r"BusInteraction::sender\(BusId::Memw, mu_minus_end", buses)) == 2,
         "an `end` row therefore emits NO memory operation -- the premise the "
         "truncation forgeries in MAIN 1 and the width audit turn on")
    a.ok("the read addresses src and the write addresses dst",
         "start_column: cols::SRC_0" in read_tuple
         and "start_column: cols::DST_0" in write_tuple)
    a.ok("both data tuples are non-register accesses",
         "// is_register" in read_tuple and "// is_register" in write_tuple)


# ---------------------------------------------------------------------------
# G. Bus packing -- element counts and tuple alignment
# ---------------------------------------------------------------------------

def audit_packing(a, repo):
    """How many BUS ELEMENTS each packing produces, and whether the two DmaNext
    tuples align element-for-element.

    THIS SECTION EXISTS BECAUSE ITS ABSENCE HID A FALSE FINDING. The audit used
    to check only that the strings `DWordWL`/`DWordHL` appeared in the sender and
    receiver tuples. It never checked how many bus elements those packings
    produce -- and the gate had assumed a 64-bit value crosses the bus as ONE
    field element. It does not: both are 2 elements with separate alpha powers,
    so the binding is per 32-bit limb. The gate's weaker model manufactured an
    alias the real bus rejects, and that phantom was published as the campaign's
    headline residual. A model weaker than the AIR yields false alarms; the
    lesson is that "how wide is one bus element" is a premise like any other and
    must be read from the source, not assumed.
    """
    lookup = read(repo, "crypto/stark/src/lookup.rs")

    body = lookup[lookup.index("pub fn num_bus_elements"):]
    body = body[:body.index("pub fn columns")]
    expected = {"Direct": 1, "Word2L": 1, "Word4L": 1, "DWordWL": 2,
                "DWordHHW": 2, "DWordWHH": 2, "DWordHL": 2, "DWordBL": 2,
                "QuadHL": 4, "QuadWL": 4}
    for name, count in expected.items():
        a.ok(f"num_bus_elements(Packing::{name}) == {count}",
             re.search(rf"Packing::{name} => {count},", body) is not None)

    # NOTE the multiplication sign: the source comments use U+00D7 ("2× Direct"),
    # not ASCII "2x". An earlier version of this claim searched for the ASCII form,
    # so the regex could never match and the predicate was tautologically true --
    # a dead guard inside the very section added to close the packing-assumption
    # gap. `2 → 1` on any compound arm now produces TWO findings, not one.
    a.ok("no Packing variant folds a 64-bit value into one bus element",
         not re.search(r"Packing::\w+ => 1, // 2×", body),
         "if one ever did, DmaNext would bind packed values and the gate's link "
         "model would have to change with it")

    # Each element gets its own alpha power.
    accum = lookup[lookup.index("Packing::DWordHL => {"):]
    accum = accum[:accum.index("// 2× Word4L")]
    a.ok("DWordHL accumulates two Word2L halves at consecutive alpha powers",
         "alpha_powers[alpha_offset]" in accum
         and "alpha_powers[alpha_offset + 1]" in accum
         and "shifts.shift_16" in accum)

    # The two DmaNext tuples must have equal element counts and align pairwise.
    dma = read(repo, "prover/src/tables/dma.rs")
    buses = dma.split("pub fn bus_interactions")[1]
    send = buses[buses.index("sender( BusId::DmaNext"):]
    send = send[:send.index("BusInteraction::receiver( BusId::DmaNext")]
    recv = buses[buses.index("receiver( BusId::DmaNext"):]
    recv = recv[:recv.index("// 4-7.")]

    def elements(tup):
        n = 0
        for packing, count in (("Packing::DWordHL", 2), ("Packing::DWordWL", 2),
                               ("Packing::Direct", 1)):
            n += tup.count(packing) * count
        return n

    a.ok("both DmaNext tuples carry the same number of bus elements",
         elements(send) == elements(recv) == 8,
         f"sender={elements(send)} receiver={elements(recv)}; a mismatch would "
         f"misalign every field and silently change what the bus binds")
    # ORDER, not just membership. §D checks that the right column names appear in
    # each tuple and the block above checks the element counts -- neither pins the
    # PAIRING, while the gate's `dmanext_link()` hard-codes it (sender low word <->
    # receiver low word, src<->src, dst<->dst, count<->count). Swapping SRC_0 and
    # DST_0 in the receiver used to leave this audit at "0 findings" while the gate
    # kept asserting src_incr<->src about a table that now binds src_incr<->dst.
    # That is the same class as the assumption that produced the retracted R1: a
    # premise about the bus taken on faith rather than read from the source.
    def ordinal(tup, names):
        """The order in which `names` first appear in a tuple's source text."""
        seen = [(tup.index(n), n) for n in names if n in tup]
        return [n for _, n in sorted(seen)]

    a.ok("the DmaNext sender orders its values ts, src_incr, dst_incr, count_decr",
         ordinal(send, ("cols::SRC_INCR_0", "cols::DST_INCR_0", "cols::COUNT_DECR_0"))
         == ["cols::SRC_INCR_0", "cols::DST_INCR_0", "cols::COUNT_DECR_0"],
         "the gate pairs these positionally with the receiver's src/dst/count")
    a.ok("the DmaNext receiver orders its values ts, src, dst, count",
         ordinal(recv, ("cols::SRC_0", "cols::DST_0", "cols::COUNT_0"))
         == ["cols::SRC_0", "cols::DST_0", "cols::COUNT_0"],
         "a swap here silently re-pairs every field the gate models")
    a.ok("both DmaNext tuples put the timestamp first, in the same order",
         ordinal(send, ("cols::TIMESTAMP_0", "cols::TIMESTAMP_1"))
         == ordinal(recv, ("cols::TIMESTAMP_0", "cols::TIMESTAMP_1"))
         == ["cols::TIMESTAMP_0", "cols::TIMESTAMP_1"])

    a.ok("the sender uses DWordHL x3 and the receiver DWordWL x3",
         send.count("Packing::DWordHL") == 3
         and recv.count("Packing::DWordWL") == 3,
         "so the aligned pairs are (incr low word, src low word) and "
         "(incr high word, src high word) -- a per-limb binding")


# ---------------------------------------------------------------------------
# E. Executor
# ---------------------------------------------------------------------------

def audit_executor(a, repo):
    execution = read(repo, "executor/src/vm/instruction/execution.rs")
    body = execution.split("SyscallNumbers::DmaMemcpy => {")[1].split("SyscallNumbers::Hint")[0]

    a.ok("the operands are read from x10, x11, x12 as dst, src, n",
         re.search(r"let dst = registers\.read\(10\)", body)
         and re.search(r"let src = registers\.read\(11\)", body)
         and re.search(r"let n = registers\.read\(12\)", body))
    a.ok("the chunk bound is an actual guard on n, not just a reachable error",
         re.search(r"if n > DMA_MEMCPY_MAX_BYTES\s*\{", body) is not None,
         "checking only that the error variant is mentioned would pass for a "
         "guard rewritten to `if false`")
    a.ok("the chunk bound is rejected BEFORE the range checks, as the oracle's "
         "`validate` orders it",
         body.index("DmaMemcpyChunkTooLarge") < body.index("checked_add"))
    a.ok("both ranges are checked for wrap",
         "dst.checked_add(n)" in body and "src.checked_add(n)" in body)
    a.ok("the copy goes through a snapshot buffer, reads before writes",
         body.index("memory.load_byte") < body.index("memory.store_byte")
         and "let mut bytes = [0u8; DMA_MEMCPY_MAX_BYTES as usize]" in body,
         "this is the implementation choice that makes an overlapping copy a "
         "memmove; the oracle's write_before_read mutant is its negative control")


# ---------------------------------------------------------------------------
# F. Trace generator
# ---------------------------------------------------------------------------

def audit_generator(a, repo):
    dma = read(repo, "prover/src/tables/dma.rs")
    gen = dma.split("pub fn generate_dma_trace")[1].split("/// Helper: a MEMW")[0]

    a.ok("rows are padded to a power of two, minimum 4",
         "next_power_of_two().max(4)" in gen)
    a.ok("width selection is `tail = count < 8` then 1 or 8",
         "let tail = op.count < 8;" in gen and "if tail { 1 } else { 8 }" in gen)
    a.ok("src_incr/dst_incr use wrapping_add and count_decr wrapping_sub",
         "op.src.wrapping_add(width)" in gen
         and "op.dst.wrapping_add(width)" in gen
         and "op.count.wrapping_sub(width)" in gen,
         "wrapping is correct here BECAUSE the AIR rejects the wraps that "
         "matter: no_overflow on src/dst, and the count wrap only on `end`")

    padding = gen.split("for row_idx in n..num_rows")[1]
    expected = ref.padding_columns()
    a.ok("the padding row sets COUNT_0 = 1", "cols::COUNT_0, FE::one()" in padding
         and expected["count"] == [1, 0])
    a.ok("the padding row sets SRC_INCR_0 = DST_INCR_0 = 1",
         "cols::SRC_INCR_0, FE::one()" in padding
         and "cols::DST_INCR_0, FE::one()" in padding
         and expected["src_incr"][0] == expected["dst_incr"][0] == 1)
    a.ok("the padding row sets TAIL = 1", "cols::TAIL, FE::one()" in padding
         and expected["tail"] == 1)
    a.ok("the padding row leaves MU, FIRST, END and COUNT_DECR at zero",
         "cols::MU" not in padding and "cols::FIRST" not in padding
         and "cols::END" not in padding and "cols::COUNT_DECR" not in padding
         and expected["mu"] == 0 and expected["count_decr"] == [0, 0, 0, 0],
         "the gate's completeness sweep pins exactly this row; if the "
         "generator changes it, the sweep must be re-run")


# ---------------------------------------------------------------------------
# H. Fixture pinning
# ---------------------------------------------------------------------------

def audit_fixture(a, repo):
    """The Rust test consumes the oracle's emitted table, not a transcription.

    Previously `prover/src/tests/dma_tests.rs` carried a hand-typed copy of the
    canonical vectors with a comment saying "do not edit by hand: rerun the
    oracle and re-transcribe" -- and nothing enforced it, so regenerating the
    vectors from a changed model left the Rust literals stale and green.
    """
    tests = read(repo, "prover/src/tests/dma_tests.rs")
    a.ok("dma_tests.rs embeds the oracle's row table with include_str!",
         "include_str!" in tests and "canonical_dma_rows.txt" in tests,
         "otherwise a regenerated oracle is a silent no-op on the Rust side")
    a.ok("dma_tests.rs drives the real decomposition, not the trace formatter",
         "dma_ops_for_test" in tests,
         "`generate_dma_trace` only formats an already-decomposed op list into "
         "columns, so asserting against it proves nothing about the row split")
    a.ok("the emitted row table exists and is non-trivial",
         len(read_raw(repo, "docs/verification/dma/dma-oracle/"
                            "canonical_dma_rows.txt").splitlines()) > 20)

    # FRESHNESS, not just presence. Greping for `include_str!` proves the Rust
    # reads a fixture; it does not prove the fixture is what the current oracle
    # emits. And `test_oracle.py` regenerates it only on a fully-green run, so an
    # oracle regression leaves a stale fixture behind with the Rust test still
    # green. Re-derive the table here and compare.
    sys.path.insert(0, os.path.join(repo, "docs/verification/dma/dma-oracle"))
    committed = read_raw(repo, "docs/verification/dma/dma-oracle/canonical_dma_rows.txt")
    try:
        import test_oracle as harness
        rows = []
        for name, dst, src, n in harness.CANONICAL_CASES:
            memory = {src + i: (i * 7 + 3) & 0xFF for i in range(n)}
            decomposed = ref.row_decomposition(0x30, dst, src, n, memory)
            data = [r for r in decomposed if not r.end]
            rows.append(f"vector|{name}|{dst}|{src}|{n}|{len(data)}")
            for r in data:
                rows.append(f"row|{r.src}|{r.dst}|{r.count}|"
                            f"{1 if r.tail else 0}|{r.width}")
        a.ok("the checked-in row table matches what the oracle emits today",
             [line for line in committed.splitlines() if not line.startswith("#")] == rows,
             "regenerate with `python3 dma-oracle/test_oracle.py`")
    except ImportError as exc:
        a.ok("the oracle harness is importable so the fixture can be re-derived",
             False, f"could not import test_oracle: {exc}")


# ---------------------------------------------------------------------------

def main():
    repo = DEFAULT_REPO
    if "--repo" in sys.argv:
        at = sys.argv.index("--repo") + 1
        if at >= len(sys.argv):
            sys.exit("--repo needs a path")
        repo = sys.argv[at]
    # Fail with a diagnosis rather than a bare FileNotFoundError deep in a check.
    for marker in ("prover/src/tables/dma.rs", "crypto/stark/src/lookup.rs"):
        if not os.path.exists(os.path.join(repo, marker)):
            sys.exit(f"{repo} does not look like a lambda_vm checkout "
                     f"(missing {marker})")
    print(f"auditing {repo}")

    a = Audit()
    # The per-section claim counts are PRINTED, not documented by hand. An
    # earlier version stated them in TRANSCRIPTION-AUDIT.md and got five of six
    # wrong -- apportioned to sum to the real total instead of measured, which is
    # the "declared, not derived" defect this file exists to catch. Now the doc
    # quotes this output.
    for name, fn in (("A. constants", audit_constants),
                     ("B. columns", audit_columns),
                     ("C. constraints", audit_constraints),
                     ("D. buses", audit_buses),
                     ("E. executor", audit_executor),
                     ("F. generator", audit_generator),
                     ("G. bus packing", audit_packing),
                     ("H. fixture pinning", audit_fixture)):
        before, before_checks = len(a.findings), a.checks
        fn(a, repo)
        n = a.checks - before_checks
        status = "ok" if len(a.findings) == before else f"{len(a.findings) - before} finding(s)"
        print(f"  {name:20s} {n:3d} claims   {status}")
    sys.exit(0 if a.report() else 1)


if __name__ == "__main__":
    main()
