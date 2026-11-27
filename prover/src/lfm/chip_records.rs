//! The LFM chips' tables held as their records, for the W-LFM prover
//! (`wlfm_knobs::CHIP_RECORDS_ENV`): a chip table is its program's
//! preprocessed prefix, packed narrow once a program at build time, followed by
//! value columns copied from one executor record a row. The fill keeps the
//! executor's value vectors as they are ([`LfmChipValues`]) instead of building
//! the chips' row-major traces, and each table goes to the prove as
//! [`LfmChipRecords`]: the card writes its prefix (the narrow widen) and its
//! values (`math_cuda::fill`, `fill_lfm_values` / `fill_lfm_bitdec`) into the
//! table's columns as the group uploads it; a host reader gets this module's
//! fill of the same columns.
//!
//! Both fills write `trace.rs`'s `chip_trace` columns word for word (canonical),
//! so no proof byte depends on which ran or on the holding: the prepared
//! opening binds the prefix the records carry, as it binds a trace's.
//!
//! The value records are the executor's own: `BaluRow`, `XaluRow`, `SelectRow`
//! and `LfmWord` are `#[repr(C)]` rows of `FE` in their chips' value-column
//! order, read as `u64` words with no copy. `LFM_BITDEC` is the one chip whose
//! record shrinks: the decomposed value, one word a row, from which the fill
//! makes the 64 bits, `Z` and `GINV`. `LFM_CONST` and `LFM_RANGE` have no
//! values (their one value column is zero).

// The W-LFM prover (and so everything here but the fill's hand-over) is built
// with the `parallel` feature only.
#![cfg_attr(not(feature = "parallel"), allow(dead_code))]

use std::sync::Arc;

use math::field::traits::IsPrimeField;
use multilinear::narrow::NarrowColumns;
use multilinear::pinned::PoolClass;
use multilinear::records::{LfmValues, RecordsFill, TableRecords};

use crate::tables::types::{FE, GoldilocksField};

#[cfg(test)]
use super::chips;
use super::executor::{BaluRow, LfmRecords, SelectRow, XaluRow, levels_values_pinned};
use super::layout;
use super::word::LfmWord;

/// An LFM chip whose tables the W-LFM prover can hold as records: every chip
/// of the WHIR recursion chip set but `LFM_HASH`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ChipKind {
    Const,
    Balu,
    Xalu,
    Select,
    Bitdec,
    Lanes,
    Hint,
    Public,
    Range,
}

impl ChipKind {
    pub(crate) const ALL: [ChipKind; 9] = [
        ChipKind::Const,
        ChipKind::Balu,
        ChipKind::Xalu,
        ChipKind::Select,
        ChipKind::Bitdec,
        ChipKind::Lanes,
        ChipKind::Hint,
        ChipKind::Public,
        ChipKind::Range,
    ];

    /// The chip an LFM AIR's name names, if its tables can be held as
    /// records (`None` for `LFM_HASH`, the keccak and BLAKE3 chips, and any
    /// other table).
    pub(crate) fn of_air(name: &str) -> Option<Self> {
        Some(match name {
            "LFM_CONST" => ChipKind::Const,
            "LFM_BALU" => ChipKind::Balu,
            "LFM_XALU" => ChipKind::Xalu,
            "LFM_SELECT" => ChipKind::Select,
            "LFM_BITDEC" => ChipKind::Bitdec,
            "LFM_LANES" => ChipKind::Lanes,
            "LFM_HINT" => ChipKind::Hint,
            "LFM_PUBLIC" => ChipKind::Public,
            "LFM_RANGE" => ChipKind::Range,
            _ => return None,
        })
    }

    /// The preprocessed prefix's columns.
    pub(crate) fn prep_width(self) -> usize {
        match self {
            ChipKind::Const => layout::const_::PREP_WIDTH,
            ChipKind::Balu => layout::balu::PREP_WIDTH,
            ChipKind::Xalu => layout::xalu::PREP_WIDTH,
            ChipKind::Select => layout::select::PREP_WIDTH,
            ChipKind::Bitdec => layout::bitdec::PREP_WIDTH,
            ChipKind::Lanes => layout::lanes::PREP_WIDTH,
            ChipKind::Hint => layout::hint::PREP_WIDTH,
            ChipKind::Public => layout::public::PREP_WIDTH,
            ChipKind::Range => layout::range::PREP_WIDTH,
        }
    }

    /// What the value columns are, from the records' words.
    pub(crate) fn values(self) -> LfmValues {
        match self {
            ChipKind::Balu => LfmValues::Words(4),
            ChipKind::Xalu => LfmValues::Words(12),
            ChipKind::Select => LfmValues::Words(17),
            ChipKind::Lanes | ChipKind::Hint | ChipKind::Public => LfmValues::Words(4),
            ChipKind::Bitdec => LfmValues::Bits,
            ChipKind::Const | ChipKind::Range => LfmValues::Zeros(1),
        }
    }

    /// The chip's columns: its prefix and its values.
    #[cfg(test)]
    pub(crate) fn num_columns(self) -> usize {
        self.prep_width() + self.values().cols()
    }

    /// The chip's columns as its AIR declares them (`chips::*::cols`).
    #[cfg(test)]
    pub(crate) fn air_columns(self) -> usize {
        match self {
            ChipKind::Const => chips::const_::cols::NUM_COLUMNS,
            ChipKind::Balu => chips::balu::cols::NUM_COLUMNS,
            ChipKind::Xalu => chips::xalu::cols::NUM_COLUMNS,
            ChipKind::Select => chips::select::cols::NUM_COLUMNS,
            ChipKind::Bitdec => chips::bitdec::cols::NUM_COLUMNS,
            ChipKind::Lanes => chips::lanes::cols::NUM_COLUMNS,
            ChipKind::Hint => chips::hint::cols::NUM_COLUMNS,
            ChipKind::Public => chips::public::cols::NUM_COLUMNS,
            ChipKind::Range => chips::range::cols::NUM_COLUMNS,
        }
    }
}

/// One chip's value records, as the executor wrote them (`LFM_BITDEC`'s as
/// the decomposed values).
enum ChipVec {
    Balu(Vec<BaluRow>),
    Xalu(Vec<XaluRow>),
    Select(Vec<SelectRow>),
    Words(Vec<LfmWord>),
    Values(Vec<u64>),
    Empty,
}

impl ChipVec {
    /// The records as row-major words, one a value cell.
    fn words(&self) -> &[u64] {
        /// `v`'s elements as `u64` words: `N` of them an element.
        ///
        /// SAFETY (the callers'): `T` is a `#[repr(C)]` struct or array of
        /// `FE` fields only, `N · 8` bytes with no padding (asserted at the
        /// types, `executor.rs`), and `FE` is `repr(transparent)` over `u64`.
        unsafe fn cast<T, const N: usize>(v: &[T]) -> &[u64] {
            debug_assert_eq!(std::mem::size_of::<T>(), N * 8);
            unsafe { core::slice::from_raw_parts(v.as_ptr().cast::<u64>(), v.len() * N) }
        }
        // SAFETY: each type is `N` `FE`s with no padding (see `cast`).
        unsafe {
            match self {
                ChipVec::Balu(v) => cast::<BaluRow, 4>(v),
                ChipVec::Xalu(v) => cast::<XaluRow, 12>(v),
                ChipVec::Select(v) => cast::<SelectRow, 17>(v),
                ChipVec::Words(v) => cast::<LfmWord, 4>(v),
                ChipVec::Values(v) => v,
                ChipVec::Empty => &[],
            }
        }
    }

    /// The records' rows.
    fn rows(&self) -> usize {
        match self {
            ChipVec::Balu(v) => v.len(),
            ChipVec::Xalu(v) => v.len(),
            ChipVec::Select(v) => v.len(),
            ChipVec::Words(v) => v.len(),
            ChipVec::Values(v) => v.len(),
            ChipVec::Empty => 0,
        }
    }
}

const _: () = {
    use std::mem::size_of;
    assert!(size_of::<BaluRow>() == 4 * 8);
    assert!(size_of::<XaluRow>() == 12 * 8);
    assert!(size_of::<SelectRow>() == 17 * 8);
    assert!(size_of::<LfmWord>() == 4 * 8);
};

impl std::fmt::Debug for ChipVec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} rows, {} words", self.rows(), self.words().len())
    }
}

/// The value records of every chip a W-LFM prove holds as records, moved out
/// of the executor's: what the fill keeps in place of those chips' traces.
pub(crate) struct LfmChipValues {
    chips: Vec<(ChipKind, Arc<ChipVec>)>,
}

impl std::fmt::Debug for LfmChipValues {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_list()
            .entries(self.chips.iter().map(|(k, v)| (k, v.rows())))
            .finish()
    }
}

impl LfmChipValues {
    /// Every held chip's records moved out of `records` (left empty), and
    /// `LFM_BITDEC`'s bits folded into their values.
    pub(crate) fn take(records: &mut LfmRecords) -> Self {
        let bitdec = core::mem::take(&mut records.bitdec);
        let value = |r: &super::executor::BitDecRow| -> u64 {
            r.bits
                .iter()
                .enumerate()
                .fold(0u64, |v, (i, b)| v | (b.canonical() & 1) << i)
        };
        let values: Vec<u64> = if levels_values_pinned() {
            let mut values = multilinear::pinned::levels_vec(bitdec.len());
            #[cfg(feature = "parallel")]
            {
                use rayon::prelude::*;
                bitdec.par_iter().map(value).collect_into_vec(&mut values);
            }
            #[cfg(not(feature = "parallel"))]
            values.extend(bitdec.iter().map(value));
            values
        } else {
            #[cfg(feature = "parallel")]
            {
                use rayon::prelude::*;
                bitdec.par_iter().map(value).collect()
            }
            #[cfg(not(feature = "parallel"))]
            bitdec.iter().map(value).collect()
        };
        drop(bitdec);
        let chips = ChipKind::ALL
            .iter()
            .map(|&kind| {
                let v = match kind {
                    ChipKind::Balu => ChipVec::Balu(core::mem::take(&mut records.balu)),
                    ChipKind::Xalu => ChipVec::Xalu(core::mem::take(&mut records.xalu)),
                    ChipKind::Select => ChipVec::Select(core::mem::take(&mut records.select)),
                    ChipKind::Bitdec => ChipVec::Values(Vec::new()),
                    ChipKind::Lanes => ChipVec::Words(core::mem::take(&mut records.lanes)),
                    ChipKind::Hint => ChipVec::Words(core::mem::take(&mut records.hint)),
                    ChipKind::Public => ChipVec::Words(core::mem::take(&mut records.public)),
                    ChipKind::Const | ChipKind::Range => ChipVec::Empty,
                };
                (kind, Arc::new(v))
            })
            .collect::<Vec<_>>();
        let mut held = LfmChipValues { chips };
        if let Some((_, v)) = held.chips.iter_mut().find(|(k, _)| *k == ChipKind::Bitdec) {
            *v = Arc::new(ChipVec::Values(values));
        }
        held
    }

    /// `kind`'s records, shared.
    fn of(&self, kind: ChipKind) -> Option<&Arc<ChipVec>> {
        self.chips.iter().find(|(k, _)| *k == kind).map(|(_, v)| v)
    }

    /// Each value vector's used bytes and whether its range is in the pool.
    pub(crate) fn upload_values(&self) -> String {
        self.chips
            .iter()
            .filter(|(_, v)| !matches!(**v, ChipVec::Empty))
            .map(|(kind, v)| {
                let words = v.words();
                let bytes = std::mem::size_of_val(words);
                let pinned =
                    bytes > 0 && multilinear::pinned::is_pinned(words.as_ptr().cast(), bytes);
                format!("{kind:?} {bytes} B pinned {pinned}")
            })
            .collect::<Vec<_>>()
            .join(", ")
    }

    /// The table of `kind`'s rows `[start, start + rows)`, its prefix
    /// `prefix` (`rows` rows of the chip's preprocessed columns): `None` when
    /// these values hold no such chip.
    pub(crate) fn table(
        &self,
        kind: ChipKind,
        prefix: Arc<NarrowColumns>,
        start: usize,
        rows: usize,
    ) -> Option<Result<LfmChipRecords, String>> {
        self.of(kind)
            .map(|values| LfmChipRecords::new(kind, prefix, Arc::clone(values), start, rows))
    }
}

/// One table of an LFM chip held as its records: rows `[start, start + rows)`
/// of the chip, its prefix packed narrow, its values the chip's records from
/// `start` (none past the chip's real rows).
#[derive(Debug)]
pub(crate) struct LfmChipRecords {
    kind: ChipKind,
    prefix: Arc<NarrowColumns>,
    values: Arc<ChipVec>,
    start: usize,
    real: usize,
    rows: usize,
}

impl LfmChipRecords {
    fn new(
        kind: ChipKind,
        prefix: Arc<NarrowColumns>,
        values: Arc<ChipVec>,
        start: usize,
        rows: usize,
    ) -> Result<Self, String> {
        if prefix.rows() != rows || prefix.cols() != kind.prep_width() {
            return Err(format!(
                "{kind:?}: a prefix of {} × {} for a table of {rows} rows and {} preprocessed \
                 columns",
                prefix.rows(),
                prefix.cols(),
                kind.prep_width()
            ));
        }
        let real = match kind.values() {
            LfmValues::Zeros(_) => 0,
            _ => values.rows().saturating_sub(start).min(rows),
        };
        Ok(LfmChipRecords {
            kind,
            prefix,
            values,
            start,
            real,
            rows,
        })
    }

    /// The real rows' value words.
    fn value_words(&self) -> &[u64] {
        let w = self.kind.values().words_per_row();
        &self.values.words()[self.start * w..(self.start + self.real) * w]
    }

    /// Every column as the host writes it, canonical: the prefix's, then the
    /// values'.
    pub(crate) fn host_columns(&self) -> Vec<Vec<u64>> {
        host_columns(
            &self.prefix,
            self.value_words(),
            self.kind.values(),
            self.real,
            self.rows,
        )
    }

    /// The chip, its first row and its real rows.
    #[cfg(test)]
    pub(crate) fn span(&self) -> (ChipKind, usize, usize) {
        (self.kind, self.start, self.real)
    }
}

impl TableRecords for LfmChipRecords {
    fn rows(&self) -> usize {
        self.rows
    }

    fn cols(&self) -> usize {
        self.prefix.cols() + self.kind.values().cols()
    }

    fn fill(&self) -> RecordsFill<'_> {
        RecordsFill::LfmChip {
            prefix: &self.prefix,
            values: self.value_words(),
            kind: self.kind.values(),
            real: self.real,
        }
    }

    fn host_fill(&self) -> Option<NarrowColumns> {
        let columns = self.host_columns();
        let refs: Vec<&[u64]> = columns.iter().map(Vec::as_slice).collect();
        NarrowColumns::pack(&refs)
    }
}

/// `LFM_BITDEC`'s value columns of one decomposed value: its 64 bits low
/// first, `Z` (top 32 bits all ones) and `GINV` (the inverse of
/// `(2^32 − 1) − top` when that is nonzero, else 0): the executor's
/// `BitDecRow`, canonical.
pub(crate) fn bitdec_row(value: u64) -> [u64; 66] {
    let v = GoldilocksField::canonical(&value);
    let mut row = [0u64; 66];
    for (i, bit) in row[..64].iter_mut().enumerate() {
        *bit = (v >> i) & 1;
    }
    let g = 0xFFFF_FFFFu64 - (v >> 32);
    if g == 0 {
        row[64] = 1;
    } else {
        row[65] = FE::from(g).inv().map(|i| i.canonical()).unwrap_or(0);
    }
    row
}

/// A chip table's columns as the host writes them (canonical words): `prefix`
/// (every row), then `kind`'s columns from `values` for the first `real` rows,
/// zero past them.
pub(crate) fn host_columns(
    prefix: &NarrowColumns,
    values: &[u64],
    kind: LfmValues,
    real: usize,
    rows: usize,
) -> Vec<Vec<u64>> {
    let mut columns: Vec<Vec<u64>> = (0..prefix.cols()).map(|c| prefix.column(c)).collect();
    let canonical = |w: u64| GoldilocksField::canonical(&w);
    match kind {
        LfmValues::Words(n) => {
            for c in 0..n {
                let mut column = vec![0u64; rows];
                for (r, v) in column.iter_mut().enumerate().take(real) {
                    *v = canonical(values[r * n + c]);
                }
                columns.push(column);
            }
        }
        LfmValues::Bits => {
            let mut value_columns = vec![vec![0u64; rows]; LfmValues::BITS_COLUMNS];
            for (r, &v) in values.iter().enumerate().take(real) {
                for (c, w) in bitdec_row(v).into_iter().enumerate() {
                    value_columns[c][r] = w;
                }
            }
            columns.extend(value_columns);
        }
        LfmValues::Zeros(n) => columns.extend((0..n).map(|_| vec![0u64; rows])),
    }
    columns
}

/// `LAMBDA_VM_LEVELS_POOL_CLASS=1` puts chip prefixes and narrow stack packs
/// in the levels' own pool arena. Read once per process.
pub(crate) fn levels_pool_class() -> PoolClass {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if *ON
        .get_or_init(|| std::env::var("LAMBDA_VM_LEVELS_POOL_CLASS").is_ok_and(|word| word == "1"))
    {
        PoolClass::Levels
    } else {
        PoolClass::Shared
    }
}

/// A table's prefix packed narrow from its columns (every one the table's
/// height): canonical words, each column at the bytes its largest word needs,
/// the columns in parallel. `None` when the columns are not one height.
pub(crate) fn pack_prefix(columns: &[&[FE]]) -> Option<NarrowColumns> {
    let rows = columns.first().map_or(0, |c| c.len());
    if columns.iter().any(|c| c.len() != rows) {
        return None;
    }
    let pack_one = |column: &&[FE]| -> (u8, Vec<u8>) {
        let words: Vec<u64> = column.iter().map(FE::canonical).collect();
        let max = words.iter().copied().max().unwrap_or(0);
        let w = multilinear_width(max);
        let mut bytes = Vec::with_capacity(rows * w as usize);
        for word in &words {
            bytes.extend_from_slice(&word.to_le_bytes()[..w as usize]);
        }
        (w, bytes)
    };
    #[cfg(feature = "parallel")]
    let packed: Vec<(u8, Vec<u8>)> = {
        use rayon::prelude::*;
        columns.par_iter().map(pack_one).collect()
    };
    #[cfg(not(feature = "parallel"))]
    let packed: Vec<(u8, Vec<u8>)> = columns.iter().map(pack_one).collect();
    let total: usize = packed.iter().map(|(_, b)| b.len()).sum();
    let mut data = levels_pool_class().with_pinned(total, || Vec::with_capacity(total));
    let mut widths = Vec::with_capacity(packed.len());
    for (w, bytes) in packed {
        widths.push(w);
        data.extend_from_slice(&bytes);
    }
    NarrowColumns::from_parts(rows, widths, data)
}

/// The bytes a word needs: 1, 2, 4 or 8 (`NarrowColumns`' rule).
fn multilinear_width(max: u64) -> u8 {
    match max {
        0..=0xff => 1,
        0x100..=0xffff => 2,
        0x1_0000..=0xffff_ffff => 4,
        _ => 8,
    }
}
