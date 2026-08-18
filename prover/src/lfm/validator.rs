//! The registry-admission validator — release-mode, always on.
//!
//! A program digest enters the `LFM_REGISTRY` only after this passes. The
//! AIR checks per-op algebra and bus balance; the *registrar* vouches for the
//! structural well-formedness below, and this validator is what makes that
//! vouching real (the reference machine checks less, and only in dev builds).
//! Together: uniqueness + acyclicity + bounded multiplicities + balance ⇒
//! every read observes the unique written value.
//!
//! "Bounded multiplicities" is load-bearing and easy to lose: balance is a
//! *field* identity, so a send gated by `p − 1` subtracts a token, and a
//! subtract-then-re-add pair over one address keeps the count while changing
//! the value. Every check below that reads `program.instrs` is therefore
//! backed by one that reads the committed `program.groups` directly — the
//! groups are what the AIR sees, and the two are joined by nothing but the
//! emitter.
//!
//! The compiler's invariant panics are tripwires; this validator is the gate;
//! the registry is the record. There is no off-switch, and there must never
//! be one.

use std::collections::{HashMap, HashSet};

use math::field::traits::IsPrimeField;

use crate::tables::types::{FE, GoldilocksField};

use super::compiler::{ColumnGroup, LfmColumnGroups, LfmProgram};
use super::instr::{Addr, Instr};
use super::layout;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LfmViolation {
    /// Check 1 — write-once uniqueness.
    DoubleWrite { addr: u64 },
    /// Check 2 — every read has a writer.
    ReadOfUnwritten { addr: u64 },
    /// Check 2 (range) — an address outside the allocated space.
    AddressOutOfRange { addr: u64 },
    /// Check 3 — acyclicity: an operand not strictly below its destination.
    CyclicRead { instr: usize, addr: u64 },
    /// Check 4 — a write's `mult` differs from the emitted read count.
    MultMismatch {
        addr: u64,
        expected: u64,
        found: u64,
    },
    /// Check 4 — a `Compress` row whose spare output slots are not the
    /// documented placeholders.
    ///
    /// `Instr::writes()` reports only `outs[0]` for `Compress` (`instr.rs`),
    /// so checks 1 and 4 never see slots 1–2 — but `emit_column_groups` copies
    /// them into the committed group unconditionally and `chips::hash` sends
    /// all three slots. `instr.rs`'s field conventions call them "`Addr(0)`
    /// placeholders with `mults` fixed to 0"; this is what makes that a rule
    /// rather than a comment.
    CompressSlotNotPlaceholder { instr: usize },
    /// Check 5 — opcode selectors not one-hot / flags not boolean on a real row.
    NonOneHotSelector { chip: &'static str, row: usize },
    /// Check 6 — nonzero data beyond the program length.
    DirtyPadding { chip: &'static str, row: usize },
    /// Check 7 — a `Hint` outside the declared arena schema.
    ArenaOutOfBounds { arena: u32, index: u32 },
    /// Check 8 — two `LFM_KECCAK` rows carry the same tag.
    ///
    /// The tag is the only thing binding a permutation's request token to its
    /// reply token: with a duplicate, a prover can swap the two rows' output
    /// states and the `Keccak` bus still balances. This is not theoretical —
    /// `keccak_probe::duplicate_tag_output_swap_accepts_demonstrating_hazard`
    /// exhibits the forgery against the raw family. Tags are preprocessed
    /// program data, so this check is what makes them trustworthy.
    DuplicateKeccakTag { tag: (u64, u64) },
    /// Check 8 — a tag half at or above `2^32`, so it cannot equal the
    /// `DWordWL` timestamp any `KECCAK_RND` row carries.
    MalformedKeccakTag { row: usize },
    /// Check 9 — a multiplicity column outside the legal range.
    ///
    /// A multiplicity is a static read count: a small non-negative integer.
    /// The field is not ordered, so nothing about `Multiplicity::Column` stops
    /// a preprocessed mult from holding `p − 1`, which the bus reads as `−1` —
    /// a **negative** send. That is the one shape the LogUp count argument
    /// cannot see: a negative ghost send cancels an honest write and a
    /// positive one replaces it at the same address, leaving the token count
    /// exactly balanced while the cell's value is prover-chosen. Surplus
    /// *positive* sends are already denied by the count (they have no matching
    /// receive); this check is what denies the negative half.
    MultOutOfRange {
        chip: &'static str,
        row: usize,
        col: usize,
        mult: u64,
        bound: u64,
    },
    /// Cross-check — group shape does not match the instruction partition.
    GroupShapeMismatch { chip: &'static str },
}

pub fn validate(program: &LfmProgram) -> Result<(), LfmViolation> {
    check_writes_and_reads(program)?;
    check_multiplicities(program)?;
    check_arenas(program)?;
    check_groups(program)?;
    check_keccak_tags(&program.groups.keccak)?;
    Ok(())
}

/// Check 8: `LFM_KECCAK` tags are well-formed and pairwise distinct.
///
/// Padding rows are skipped: their `IS_REAL` is zero, so they emit no bus
/// tokens and their all-zero tag binds nothing.
fn check_keccak_tags(group: &ColumnGroup) -> Result<(), LfmViolation> {
    let mut seen = HashSet::new();
    for row in 0..group.real_rows {
        let lo = GoldilocksField::canonical(group.at(row, layout::keccak::TAG_LO).value());
        let hi = GoldilocksField::canonical(group.at(row, layout::keccak::TAG_HI).value());
        if lo >= 1u64 << 32 || hi >= 1u64 << 32 {
            return Err(LfmViolation::MalformedKeccakTag { row });
        }
        if !seen.insert((lo, hi)) {
            return Err(LfmViolation::DuplicateKeccakTag { tag: (lo, hi) });
        }
    }
    Ok(())
}

/// Checks 1–3: uniqueness, read-has-writer, acyclicity.
fn check_writes_and_reads(program: &LfmProgram) -> Result<(), LfmViolation> {
    let n = program.num_addrs as usize;
    let mut written = vec![false; n];
    for instr in &program.instrs {
        for Addr(w) in instr.writes() {
            let slot = written
                .get_mut(w as usize)
                .ok_or(LfmViolation::AddressOutOfRange { addr: w })?;
            if *slot {
                return Err(LfmViolation::DoubleWrite { addr: w });
            }
            *slot = true;
        }
    }
    for (idx, instr) in program.instrs.iter().enumerate() {
        let min_write = instr.writes().iter().map(|a| a.0).min();
        for Addr(r) in instr.reads() {
            if !*written
                .get(r as usize)
                .ok_or(LfmViolation::AddressOutOfRange { addr: r })?
            {
                return Err(LfmViolation::ReadOfUnwritten { addr: r });
            }
            if let Some(w) = min_write
                && r >= w
            {
                return Err(LfmViolation::CyclicRead {
                    instr: idx,
                    addr: r,
                });
            }
        }
    }
    Ok(())
}

/// Check 4: every write's `mult` equals an independent recount of its reads.
fn check_multiplicities(program: &LfmProgram) -> Result<(), LfmViolation> {
    let mut counts: HashMap<Addr, u64> = HashMap::new();
    for instr in &program.instrs {
        for r in instr.reads() {
            *counts.entry(r).or_insert(0) += 1;
        }
    }
    let check = |addr: Addr, found: u64| -> Result<(), LfmViolation> {
        let expected = counts.get(&addr).copied().unwrap_or(0);
        if expected != found {
            return Err(LfmViolation::MultMismatch {
                addr: addr.0,
                expected,
                found,
            });
        }
        Ok(())
    };
    for (idx, instr) in program.instrs.iter().enumerate() {
        match instr {
            Instr::Const { out, mult, .. }
            | Instr::BaseAlu { out, mult, .. }
            | Instr::ExtAlu { out, mult, .. }
            | Instr::Hint { out, mult, .. }
            | Instr::Pack { out, mult, .. } => check(*out, *mult)?,
            Instr::Unpack { outs, mults, .. } => {
                for i in 0..4 {
                    check(outs[i], mults[i])?;
                }
            }
            Instr::Select {
                out_l,
                out_r,
                mult_l,
                mult_r,
                ..
            } => {
                check(*out_l, *mult_l)?;
                check(*out_r, *mult_r)?;
            }
            Instr::BitDec { bits, .. } => {
                for (addr, mult) in bits {
                    check(*addr, *mult)?;
                }
            }
            Instr::Hash {
                mode,
                ins,
                outs,
                mults,
            } => {
                let num_outs = mode.num_output_cells();
                for i in 0..num_outs {
                    check(outs[i], mults[i])?;
                }
                // A one-output row's spare slots are outside `writes()` and so
                // outside checks 1 and 4 — but they are inside the committed
                // group and inside the bus. Pin them to the placeholders
                // `instr.rs` documents, so "slot 0 only" is a checked property
                // of the program and not a convention the emitter happens to
                // follow.
                if mode.num_output_cells() == 1
                    && (mults[1] != 0 || mults[2] != 0 || outs[1] != Addr(0) || outs[2] != Addr(0))
                {
                    return Err(LfmViolation::CompressSlotNotPlaceholder { instr: idx });
                }
                // The same for the INPUT slots a mode does not read. A leaf row
                // reads one cell, so its second and third slots reach the bus as
                // addresses nothing receives; pinning them keeps "reads exactly
                // `num_input_cells`" a checked property too.
                if ins[mode.num_input_cells()..].iter().any(|a| *a != Addr(0)) {
                    return Err(LfmViolation::CompressSlotNotPlaceholder { instr: idx });
                }
            }
            Instr::KeccakF(k) => {
                for i in 0..layout::keccak::NUM_WORDS {
                    check(k.outs[i], k.mults[i])?;
                }
                if let Some(rev) = &k.rev {
                    for i in 0..layout::keccak::DIGEST_WORDS {
                        check(rev.outs[i], rev.mults[i])?;
                    }
                }
            }
            Instr::Blake3(k) => {
                for i in 0..layout::blake3::OUT_WORDS {
                    check(k.outs[i], k.mults[i])?;
                }
                if let Some(rev) = &k.rev {
                    for i in 0..layout::blake3::DIGEST_WORDS {
                        check(rev.outs[i], rev.mults[i])?;
                    }
                }
            }
            Instr::Public { .. } => {}
        }
    }
    Ok(())
}

/// Check 7: arena discipline.
fn check_arenas(program: &LfmProgram) -> Result<(), LfmViolation> {
    let lens = &program.arena_schema.lens;
    for instr in &program.instrs {
        if let Instr::Hint { arena, index, .. } = instr {
            let ok = lens.get(*arena as usize).is_some_and(|&len| *index < len);
            if !ok {
                return Err(LfmViolation::ArenaOutOfBounds {
                    arena: *arena,
                    index: *index,
                });
            }
        }
    }
    Ok(())
}

fn is_bool(v: &FE) -> bool {
    *v == FE::zero() || *v == FE::one()
}

/// Checks 5–6 on the emitted column groups: selector one-hot-ness on real
/// rows, all-zero padding beyond the program length — plus the shape
/// cross-check against the instruction partition.
fn check_groups(program: &LfmProgram) -> Result<(), LfmViolation> {
    let g = &program.groups;

    let chip_real = |chip: &'static str, group: &ColumnGroup, count: usize| {
        if group.real_rows == count {
            Ok(())
        } else {
            Err(LfmViolation::GroupShapeMismatch { chip })
        }
    };
    let counts = partition_counts(&program.instrs);
    chip_real("LFM_CONST", &g.const_, counts.const_)?;
    chip_real("LFM_BALU", &g.balu, counts.balu)?;
    chip_real("LFM_XALU", &g.xalu, counts.xalu)?;
    chip_real("LFM_SELECT", &g.select, counts.select)?;
    chip_real("LFM_BITDEC", &g.bitdec, counts.bitdec)?;
    chip_real("LFM_HASH", &g.hash, counts.hash)?;
    chip_real("LFM_KECCAK", &g.keccak, counts.keccak)?;
    chip_real("LFM_BLAKE3", &g.blake3, counts.blake3)?;
    chip_real("LFM_LANES", &g.lanes, counts.lanes)?;
    chip_real("LFM_HINT", &g.hint, counts.hint)?;
    chip_real("LFM_PUBLIC", &g.public, counts.public)?;

    // Selector one-hot / is_real flags on real rows.
    one_hot(
        &g.balu,
        "LFM_BALU",
        layout::balu::SEL_ADD,
        layout::balu::NUM_SELECTORS,
    )?;
    one_hot(
        &g.xalu,
        "LFM_XALU",
        layout::xalu::SEL_ADD,
        layout::xalu::NUM_SELECTORS,
    )?;
    one_hot(
        &g.hash,
        "LFM_HASH",
        layout::hash::MODE_C,
        layout::hash::NUM_SELECTORS,
    )?;
    one_hot(&g.lanes, "LFM_LANES", layout::lanes::MODE_PACK, 2)?;
    one_hot(&g.keccak, "LFM_KECCAK", layout::keccak::MODE_PERM, 2)?;
    // `LFM_BLAKE3` has ONE mode, so its is-real flag is a flag rather than a
    // one-hot run — the same shape `LFM_SELECT` and `LFM_BITDEC` have. The AIR
    // keeps its own ungated `IS_BIT(MU)` belt-over-suspenders; this is what
    // makes MU = 1 on every real row a checked property of the program.
    flag_is_one(&g.blake3, "LFM_BLAKE3", layout::blake3::MU)?;
    flag_is_one(&g.select, "LFM_SELECT", layout::select::IS_REAL)?;
    flag_is_one(&g.bitdec, "LFM_BITDEC", layout::bitdec::IS_REAL)?;
    flag_is_one(&g.public, "LFM_PUBLIC", layout::public::IS_REAL)?;

    // Padding: everything beyond the real rows is zero.
    for (chip, group) in [
        ("LFM_CONST", &g.const_),
        ("LFM_BALU", &g.balu),
        ("LFM_XALU", &g.xalu),
        ("LFM_SELECT", &g.select),
        ("LFM_BITDEC", &g.bitdec),
        ("LFM_HASH", &g.hash),
        ("LFM_KECCAK", &g.keccak),
        ("LFM_BLAKE3", &g.blake3),
        ("LFM_LANES", &g.lanes),
        ("LFM_HINT", &g.hint),
        ("LFM_PUBLIC", &g.public),
    ] {
        for row in group.real_rows..group.padded_rows {
            for col in 0..group.width {
                if *group.at(row, col) != FE::zero() {
                    return Err(LfmViolation::DirtyPadding { chip, row });
                }
            }
        }
    }

    check_mult_ranges(g)?;
    Ok(())
}

/// The hard ceiling on any multiplicity, independent of the per-chip
/// accounting below and of how that accounting might drift: `2^32`, the same
/// canonical-range shape check 8 uses on the keccak tags. Under it no
/// multiplicity can reach the half of the field the bus reads as negative, and
/// no sum of them over a program of any buildable size can wrap the modulus —
/// which is what the LogUp argument needs in order to mean what it says.
const MULT_HARD_CAP: u64 = 1 << 32;

/// The largest value a multiplicity may legally take.
///
/// A multiplicity is the number of times its cell is read, so it cannot exceed
/// the number of reads the *whole program* emits — which is bounded by the
/// committed row counts times the per-chip count of `LfmMem` receivers. Those
/// counts mirror `chips::*::bus_interactions`; over-counting only widens an
/// upper bound (safe), under-counting would reject honest programs, so where
/// two receivers are mutually exclusive the count rounds up.
fn mult_bound(g: &LfmColumnGroups) -> u64 {
    let receives_per_row: [(&ColumnGroup, u64); 11] = [
        (&g.const_, 0), // reads nothing
        (&g.balu, 3),   // A, B, C
        (&g.xalu, 3),   // A, B, C
        (&g.select, 3), // BIT, IN_L, IN_R
        (&g.bitdec, 1), // IN
        (&g.hash, 3),   // IN0, IN1, IN2
        (
            &g.keccak,
            (layout::keccak::NUM_WORDS + layout::keccak::BLOCK_WORDS) as u64,
        ), // state + rate block
        (&g.blake3, layout::blake3::IN_WORDS as u64), // h | m | params
        (&g.lanes, 5),  // the word (Unpack) or the four lanes (Pack)
        (&g.hint, 0),   // reads nothing
        (&g.public, 1), // the published cell
    ];
    let reads = receives_per_row.iter().fold(0u64, |acc, (group, per_row)| {
        acc.saturating_add((group.real_rows as u64).saturating_mul(*per_row))
    });
    reads.min(MULT_HARD_CAP - 1)
}

/// Every preprocessed column that gates an `LfmMem` **send** — i.e. every
/// write multiplicity — per chip, mirroring `chips::*::bus_interactions`.
///
/// `LFM_PUBLIC` contributes none: it only receives, gated by `IS_REAL`, which
/// check 5 pins to 1. Every other receive gate is likewise a selector already
/// pinned to `{0,1}` on real rows by check 5 and to 0 on padding rows by
/// check 6 — the send gates listed here are the only unbounded ones.
fn mult_columns(g: &LfmColumnGroups) -> Vec<(&'static str, &ColumnGroup, Vec<usize>)> {
    use layout::{balu, bitdec, blake3, const_, hash, hint, keccak, lanes, select, xalu};
    vec![
        ("LFM_CONST", &g.const_, vec![const_::MULT]),
        ("LFM_BALU", &g.balu, vec![balu::MULT]),
        ("LFM_XALU", &g.xalu, vec![xalu::MULT]),
        (
            "LFM_SELECT",
            &g.select,
            vec![select::MULT_L, select::MULT_R],
        ),
        (
            "LFM_BITDEC",
            &g.bitdec,
            (0..bitdec::NUM_BITS).map(bitdec::bit_mult).collect(),
        ),
        (
            "LFM_HASH",
            &g.hash,
            vec![hash::MULT0, hash::MULT1, hash::MULT2],
        ),
        (
            "LFM_KECCAK",
            &g.keccak,
            (0..keccak::NUM_WORDS)
                .map(keccak::mult)
                .chain((0..keccak::DIGEST_WORDS).map(keccak::rev_mult))
                .collect(),
        ),
        (
            "LFM_BLAKE3",
            &g.blake3,
            (0..blake3::OUT_WORDS)
                .map(blake3::mult)
                .chain((0..blake3::DIGEST_WORDS).map(blake3::rev_mult))
                .collect(),
        ),
        (
            "LFM_LANES",
            &g.lanes,
            (0..4)
                .map(|i| lanes::LANE_MULT0 + i)
                .chain([lanes::WORD_MULT])
                .collect(),
        ),
        ("LFM_HINT", &g.hint, vec![hint::MULT]),
    ]
}

/// Check 9: every multiplicity column of every group holds a canonically
/// small non-negative integer.
///
/// This is the one check on this list that is deliberately
/// **instrs-independent**. Checks 1–4 reach the committed columns only through
/// `program.instrs`, so any slot the `Instr` accessors do not expose is
/// invisible to them while still being emitted onto the bus, and the two
/// objects are joined by nothing but the emitter. Reading the committed
/// columns directly is what makes this hold whatever the instruction list
/// claims — and a field-negative multiplicity, the shape the LogUp count
/// argument cannot see (see [`LfmViolation::MultOutOfRange`]), is a canonical
/// value near `p` and dies here.
///
/// Padding rows are skipped: check 6 already pins them to all-zero.
fn check_mult_ranges(g: &LfmColumnGroups) -> Result<(), LfmViolation> {
    let bound = mult_bound(g);
    for (chip, group, cols) in mult_columns(g) {
        for row in 0..group.real_rows {
            for &col in &cols {
                let mult = GoldilocksField::canonical(group.at(row, col).value());
                if mult > bound {
                    return Err(LfmViolation::MultOutOfRange {
                        chip,
                        row,
                        col,
                        mult,
                        bound,
                    });
                }
            }
        }
    }
    Ok(())
}

fn one_hot(
    group: &ColumnGroup,
    chip: &'static str,
    first_sel: usize,
    num_sels: usize,
) -> Result<(), LfmViolation> {
    for row in 0..group.real_rows {
        let mut ones = 0usize;
        for s in 0..num_sels {
            let v = group.at(row, first_sel + s);
            if !is_bool(v) {
                return Err(LfmViolation::NonOneHotSelector { chip, row });
            }
            if *v == FE::one() {
                ones += 1;
            }
        }
        if ones != 1 {
            return Err(LfmViolation::NonOneHotSelector { chip, row });
        }
    }
    Ok(())
}

fn flag_is_one(group: &ColumnGroup, chip: &'static str, col: usize) -> Result<(), LfmViolation> {
    for row in 0..group.real_rows {
        if *group.at(row, col) != FE::one() {
            return Err(LfmViolation::NonOneHotSelector { chip, row });
        }
    }
    Ok(())
}

struct PartitionCounts {
    const_: usize,
    balu: usize,
    xalu: usize,
    select: usize,
    bitdec: usize,
    hash: usize,
    keccak: usize,
    blake3: usize,
    lanes: usize,
    hint: usize,
    public: usize,
}

fn partition_counts(instrs: &[Instr]) -> PartitionCounts {
    let mut c = PartitionCounts {
        const_: 0,
        balu: 0,
        xalu: 0,
        select: 0,
        bitdec: 0,
        hash: 0,
        keccak: 0,
        blake3: 0,
        lanes: 0,
        hint: 0,
        public: 0,
    };
    for i in instrs {
        match i {
            Instr::Const { .. } => c.const_ += 1,
            Instr::BaseAlu { .. } => c.balu += 1,
            Instr::ExtAlu { .. } => c.xalu += 1,
            Instr::Select { .. } => c.select += 1,
            Instr::BitDec { .. } => c.bitdec += 1,
            Instr::Hash { .. } => c.hash += 1,
            Instr::KeccakF(_) => c.keccak += 1,
            Instr::Blake3(_) => c.blake3 += 1,
            Instr::Pack { .. } | Instr::Unpack { .. } => c.lanes += 1,
            Instr::Hint { .. } => c.hint += 1,
            Instr::Public { .. } => c.public += 1,
        }
    }
    c
}
