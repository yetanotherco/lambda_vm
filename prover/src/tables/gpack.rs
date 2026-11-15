//! G-pack: a generator writes its main trace straight into the packed form the
//! block holds it in (`stark::narrow`), with no 64-bit copy.
//!
//! A generator's fill is generic over [`VmTable`], and [`generate_main!`] runs
//! it on the writer its caller's [`TraceForm`] asks for:
//! - [`TraceForm::Wide`]: the table at 8 bytes a cell, as every build outside
//!   the block's narrow storage holds it;
//! - [`TraceForm::Narrow`]: the packed trace, the `NarrowMain::pack` of the same
//!   words byte for byte. When the table's kind has a [`WidthHint`] (the widths
//!   its earlier traces needed in this process), the fill writes the packed
//!   columns directly at those widths ([`NarrowWriter`]): a column that needed
//!   fewer bytes is narrowed in place, and a word wider than its column has the
//!   fill run again at the widths the trace needs. Without a hint (the kind's
//!   first trace) the fill writes the 64-bit table and packs it, as before,
//!   and the widths it needed become the hint.
//!
//! The packed bytes are a function of the words alone: a hint changes the
//! time a trace takes, never a byte of it.

use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};

use multilinear::narrow::NarrowColumns;
use stark::narrow::{NarrowRows, NarrowWriter};
use stark::table::TableRowsMut;
use stark::trace::TraceTable;

use super::types::{FE, GoldilocksExtension, GoldilocksField, VmTable};

/// A generator's trace.
pub type Trace = TraceTable<GoldilocksField, GoldilocksExtension>;

/// How a generator holds the main trace it writes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TraceForm {
    /// At 8 bytes a cell.
    Wide,
    /// Packed at the bytes its columns need (`TraceTable::is_main_narrow`),
    /// where a trace can be held packed; at 8 bytes a cell where it cannot (the
    /// `debug-checks` build).
    Narrow,
}

/// The widths a table kind's traces needed so far in this process, column by
/// column (the most each column needed): where a [`TraceForm::Narrow`]
/// generator writes the kind's next trace.
pub struct WidthHint(Mutex<Vec<u8>>);

impl WidthHint {
    /// No widths yet.
    pub const fn new() -> Self {
        Self(Mutex::new(Vec::new()))
    }

    /// The widths, when there are some for `cols` columns.
    fn get(&self, cols: usize) -> Option<Vec<u8>> {
        let widths = self.0.lock().unwrap_or_else(|e| e.into_inner());
        (widths.len() == cols && cols > 0).then(|| widths.clone())
    }

    /// Whether there are widths for `cols` columns.
    pub fn known(&self, cols: usize) -> bool {
        let widths = self.0.lock().unwrap_or_else(|e| e.into_inner());
        widths.len() == cols && cols > 0
    }

    /// The bytes a packed row takes at these widths (their sum), when there
    /// are widths.
    pub(crate) fn bytes_a_row(&self) -> Option<usize> {
        let widths = self.0.lock().unwrap_or_else(|e| e.into_inner());
        (!widths.is_empty()).then(|| widths.iter().map(|&w| usize::from(w)).sum())
    }

    /// Take `widths` (a trace's) into the most each column needed.
    pub fn learn(&self, widths: &[u8]) {
        let mut most = self.0.lock().unwrap_or_else(|e| e.into_inner());
        if most.len() == widths.len() {
            for (m, &w) in most.iter_mut().zip(widths) {
                *m = (*m).max(w);
            }
        } else {
            *most = widths.to_vec();
        }
    }
}

impl Default for WidthHint {
    fn default() -> Self {
        Self::new()
    }
}

/// The packed cells: the raw word of each element, as `NarrowMain` holds it.
impl VmTable for NarrowWriter {
    #[inline]
    fn set_fe(&mut self, row: usize, col: usize, value: FE) {
        self.set(row, col, *value.value());
    }
}

/// A part of a 64-bit table split by rows ([`generate_main_rows!`]).
impl VmTable for TableRowsMut<'_, GoldilocksField> {
    #[inline]
    fn set_fe(&mut self, row: usize, col: usize, value: FE) {
        self.set(row, col, value);
    }
}

/// A part of a packed writer split by rows ([`generate_main_rows!`]).
impl VmTable for NarrowRows<'_> {
    #[inline]
    fn set_fe(&mut self, row: usize, col: usize, value: FE) {
        self.set(row, col, *value.value());
    }
}

/// The fewest rows a part of a split fill holds ([`row_bounds`]).
pub(crate) const SPLIT_MIN_ROWS: usize = 1 << 16;

/// The most parts a split fill cuts a table into ([`row_bounds`]).
pub(crate) const SPLIT_MAX_PARTS: usize = 8;

/// The row ranges a fill of `rows` rows is split into: `[b[i], b[i + 1])`.
/// One range when `split` is off or the table has fewer than two parts'
/// rows; else up to [`SPLIT_MAX_PARTS`] near-equal ranges of at least
/// [`SPLIT_MIN_ROWS`] rows.
pub(crate) fn row_bounds(rows: usize, split: bool) -> Vec<usize> {
    if !split {
        return vec![0, rows];
    }
    row_bounds_at(rows, SPLIT_MIN_ROWS, SPLIT_MAX_PARTS)
}

/// [`row_bounds`] with parts of at least `min_rows` rows, at most `max_parts`.
pub(crate) fn row_bounds_at(rows: usize, min_rows: usize, max_parts: usize) -> Vec<usize> {
    let min_rows = min_rows.max(1);
    if rows < 2 * min_rows || max_parts < 2 {
        return vec![0, rows];
    }
    let parts = (rows / min_rows).min(max_parts);
    (0..=parts).map(|p| rows * p / parts).collect()
}

/// Fills [`generate_main_rows!`] ran as more than one part, since the process
/// began (a measurement, and the tests' proof that a split happened).
static SPLIT_FILLS: AtomicUsize = AtomicUsize::new(0);

/// [`SPLIT_FILLS`].
pub fn split_fills() -> usize {
    SPLIT_FILLS.load(Relaxed)
}

#[doc(hidden)]
pub fn note_split(parts: usize) {
    if parts > 1 {
        SPLIT_FILLS.fetch_add(1, Relaxed);
    }
}

/// `bounds` made a cover of the rows `[0, rows)`: rising, from 0 to `rows`, so
/// a split fill leaves no row unwritten whatever bounds it is handed.
#[doc(hidden)]
pub fn covering(bounds: &[usize], rows: usize) -> Vec<usize> {
    let mut out = Vec::with_capacity(bounds.len() + 2);
    out.push(0);
    for &b in bounds {
        let b = b.min(rows);
        if b > *out.last().unwrap_or(&0) {
            out.push(b);
        }
    }
    if out.last() != Some(&rows) {
        out.push(rows);
    }
    out
}

/// A part of a split fill: the rows it holds.
#[doc(hidden)]
pub trait RowPart: VmTable + Send {
    fn rows(&self) -> std::ops::Range<usize>;
}

impl RowPart for TableRowsMut<'_, GoldilocksField> {
    fn rows(&self) -> std::ops::Range<usize> {
        self.range()
    }
}

impl RowPart for NarrowRows<'_> {
    fn rows(&self) -> std::ops::Range<usize> {
        self.range()
    }
}

/// Run `fill` on every part, each over its own rows: on the pool the caller
/// runs on (never `install`), or in turn without `parallel`.
#[doc(hidden)]
pub fn run_parts<P: RowPart>(
    parts: &mut [P],
    fill: impl Fn(&mut P, std::ops::Range<usize>) + Sync,
) {
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        parts.par_iter_mut().for_each(|part| {
            let rows = part.rows();
            fill(part, rows);
        });
    }
    #[cfg(not(feature = "parallel"))]
    for part in parts.iter_mut() {
        let rows = part.rows();
        fill(part, rows);
    }
}

/// `segments`' items at the flat indices `range` (the segments one after
/// another), each with its index: a split fill's operations.
pub(crate) fn items_in<'s, T>(
    segments: &'s [&'s [T]],
    range: std::ops::Range<usize>,
) -> impl Iterator<Item = (usize, &'s T)> + 's {
    let mut base = 0usize;
    segments.iter().flat_map(move |segment| {
        let (start, end) = (base, base + segment.len());
        base = end;
        let (a, b) = (range.start.clamp(start, end), range.end.clamp(start, end));
        segment[a - start..b - start]
            .iter()
            .enumerate()
            .map(move |(i, item)| (a + i, item))
    })
}

/// How [`generate_main!`] builds one trace.
#[doc(hidden)]
pub enum Plan {
    /// The 64-bit table.
    Wide,
    /// The 64-bit table, then packed, its widths learned.
    WideThenPack,
    /// Written packed at these widths.
    Write(Vec<u8>),
}

impl Plan {
    #[doc(hidden)]
    pub fn new(form: TraceForm, hint: &WidthHint, cols: usize) -> Self {
        match form {
            TraceForm::Wide => Plan::Wide,
            // The debug checks read the 64-bit table (`TraceTable::pack_main_narrow`).
            TraceForm::Narrow if cfg!(feature = "debug-checks") => Plan::Wide,
            TraceForm::Narrow => hint.get(cols).map_or(Plan::WideThenPack, Plan::Write),
        }
    }
}

/// Traces a [`TraceForm::Narrow`] generator wrote packed and finished as
/// written, narrowed some columns of in place, wrote again after a word missed
/// its column, and built at 8 bytes a cell then packed (no hint yet); since
/// [`reset_counts`].
static COUNTS: [AtomicUsize; 4] = [const { AtomicUsize::new(0) }; 4];

/// G-pack's counts since [`reset_counts`]: traces written packed as they
/// were, written packed with some columns narrowed afterwards, written packed
/// a second time (a word missed its column), and built wide then packed.
pub fn counts() -> [usize; 4] {
    std::array::from_fn(|i| COUNTS[i].load(Relaxed))
}

/// Zero [`counts`].
pub fn reset_counts() {
    for c in &COUNTS {
        c.store(0, Relaxed);
    }
}

#[cfg(test)]
thread_local! {
    /// [`counts`] of this thread's traces, never reset.
    static THREAD_COUNTS: std::cell::Cell<[usize; 4]> = const { std::cell::Cell::new([0; 4]) };
}

/// [`counts`] of the traces this thread built, since the thread began.
#[cfg(test)]
pub(crate) fn thread_counts() -> [usize; 4] {
    THREAD_COUNTS.with(std::cell::Cell::get)
}

/// One more trace in [`counts`] slot `i`.
fn count(i: usize) {
    COUNTS[i].fetch_add(1, Relaxed);
    #[cfg(test)]
    THREAD_COUNTS.with(|c| {
        let mut n = c.get();
        n[i] += 1;
        c.set(n);
    });
}

/// A zeroed 64-bit trace of `rows` × `cols`.
#[doc(hidden)]
pub fn wide_trace(rows: usize, cols: usize) -> Trace {
    TraceTable::new_main(super::types::zeroed_fe_vec(rows * cols), cols, 1)
}

/// `trace` packed (where it can be), its widths taken into `hint`.
#[doc(hidden)]
pub fn packed(mut trace: Trace, hint: &WidthHint) -> Trace {
    if trace.pack_main_narrow()
        && let Some(narrow) = trace.narrow_main()
    {
        hint.learn(narrow.widths());
        count(3);
    }
    trace
}

/// The trace `writer` holds, its widths taken into `hint`; or the widths it
/// needs, when a word missed its column.
#[doc(hidden)]
pub fn written(writer: NarrowWriter, hint: &WidthHint, again: bool) -> Result<Trace, Vec<u8>> {
    let narrowed = writer.needed_widths() != writer.widths();
    match writer.finish() {
        Ok(narrow) => {
            hint.learn(narrow.widths());
            count(if again {
                2
            } else if narrowed {
                1
            } else {
                0
            });
            // The trace holds it as `multilinear`'s columns, the same layout.
            // An `Err` of no widths has the caller build the trace wide (not
            // reached: the writer's widths are 1, 2, 4 or 8 and its bytes are
            // rows × Σ widths).
            let (rows, widths, data) = narrow.into_parts();
            let columns = NarrowColumns::from_parts(rows, widths, data).ok_or_else(Vec::new)?;
            Ok(TraceTable::try_from_narrow_main(columns, 1)
                .unwrap_or_else(|columns| widened(&columns)))
        }
        Err(need) => {
            hint.learn(&need);
            Err(need)
        }
    }
}

/// `columns`' words as a 64-bit trace.
fn widened(columns: &NarrowColumns) -> Trace {
    let (rows, cols) = (columns.rows(), columns.cols());
    let mut data = vec![FE::zero(); rows * cols];
    for c in 0..cols {
        for (r, w) in columns.column(c).into_iter().enumerate() {
            data[r * cols + c] = FE::from(w);
        }
    }
    TraceTable::new_main(data, cols, 1)
}

/// Build a `rows` × `cols` main trace (step size 1) in `form`: `fill`, an
/// expression over `$t: &mut impl VmTable` that writes every non-zero cell
/// once, runs on the writer the [`Plan`] picks, and again on a second writer
/// when a word missed its column (so it must only read its inputs).
///
/// ```ignore
/// generate_main!(form, &WIDTHS, num_rows, cols::NUM_COLUMNS, |t| fill(t, ops))
/// ```
macro_rules! generate_main {
    ($form:expr, $hint:expr, $rows:expr, $cols:expr, |$t:ident| $fill:expr) => {{
        use $crate::tables::gpack::{Plan, packed, wide_trace, written};
        let (rows, cols, hint): (usize, usize, &$crate::tables::gpack::WidthHint) =
            ($rows, $cols, $hint);
        let wide = |then_pack: bool| {
            let mut trace = wide_trace(rows, cols);
            {
                let $t = &mut trace.main_table;
                $fill;
            }
            if then_pack {
                packed(trace, hint)
            } else {
                trace
            }
        };
        let write = |widths: &[u8], again: bool| {
            let mut writer = ::stark::narrow::NarrowWriter::new(rows, widths);
            {
                let $t = &mut writer;
                $fill;
            }
            written(writer, hint, again)
        };
        match Plan::new($form, hint, cols) {
            Plan::Wide => wide(false),
            Plan::WideThenPack => wide(true),
            Plan::Write(widths) => match write(&widths, false) {
                Ok(trace) => trace,
                Err(need) if need.len() == cols => match write(&need, true) {
                    Ok(trace) => trace,
                    // Not reached: every word fits the widths of every word.
                    Err(_) => wide(true),
                },
                Err(_) => wide(true),
            },
        }
    }};
}
pub(crate) use generate_main;

/// [`generate_main!`] with the fill split by rows: `fill`, an expression over
/// `$t: &mut impl VmTable` and `$range: Range<usize>` that writes every
/// non-zero cell of the rows `$range` once (and no other row), runs once a
/// range of `bounds` ([`row_bounds`]) on that range's part of the writer the
/// [`Plan`] picks, the parts at once ([`run_parts`]). A packed writer takes
/// back each part's OR of its words before it finishes, so the trace — its
/// widths, its narrowing, a second pass when a word missed its column — is
/// the trace [`generate_main!`] writes from the same cells, byte for byte.
///
/// ```ignore
/// generate_main_rows!(form, &WIDTHS, num_rows, cols::NUM_COLUMNS, &bounds, |t, rows| fill(t, ops, rows))
/// ```
macro_rules! generate_main_rows {
    ($form:expr, $hint:expr, $rows:expr, $cols:expr, $bounds:expr, |$t:ident, $range:ident| $fill:expr) => {{
        use $crate::tables::gpack::{Plan, packed, run_parts, wide_trace, written};
        let (rows, cols, hint): (usize, usize, &$crate::tables::gpack::WidthHint) =
            ($rows, $cols, $hint);
        let bounds = $crate::tables::gpack::covering(&$bounds, rows);
        let bounds = bounds.as_slice();
        $crate::tables::gpack::note_split(bounds.len() - 1);
        let wide = |then_pack: bool| {
            let mut trace = wide_trace(rows, cols);
            {
                let mut parts = trace.main_table.split_rows_mut(bounds);
                run_parts(&mut parts, |$t, $range| $fill);
            }
            if then_pack {
                packed(trace, hint)
            } else {
                trace
            }
        };
        let write = |widths: &[u8], again: bool| {
            let mut writer = ::stark::narrow::NarrowWriter::new(rows, widths);
            let seen: Vec<Vec<u64>> = {
                let mut parts = writer.split_rows(bounds);
                run_parts(&mut parts, |$t, $range| $fill);
                parts
                    .into_iter()
                    .map(::stark::narrow::NarrowRows::into_seen)
                    .collect()
            };
            for part in &seen {
                writer.absorb(part);
            }
            written(writer, hint, again)
        };
        match Plan::new($form, hint, cols) {
            Plan::Wide => wide(false),
            Plan::WideThenPack => wide(true),
            Plan::Write(widths) => match write(&widths, false) {
                Ok(trace) => trace,
                Err(need) if need.len() == cols => match write(&need, true) {
                    Ok(trace) => trace,
                    // Not reached: every word fits the widths of every word.
                    Err(_) => wide(true),
                },
                Err(_) => wide(true),
            },
        }
    }};
}
pub(crate) use generate_main_rows;

#[cfg(test)]
mod split_bounds_tests {
    use super::{SPLIT_MAX_PARTS, SPLIT_MIN_ROWS, covering, row_bounds, row_bounds_at};

    /// A table under two parts' rows stays one range; a 2^17-row table is two,
    /// a 2^20-row table eight, never more; off is one range.
    #[test]
    fn the_production_bounds_split_only_tall_tables() {
        assert_eq!(row_bounds(1 << 20, false), vec![0, 1 << 20]);
        assert_eq!(row_bounds((1 << 17) - 1, true), vec![0, (1 << 17) - 1]);
        assert_eq!(row_bounds(1 << 17, true), vec![0, 1 << 16, 1 << 17]);
        let b = row_bounds(1 << 20, true);
        assert_eq!(b.len(), SPLIT_MAX_PARTS + 1);
        assert!(b.windows(2).all(|w| w[1] - w[0] >= SPLIT_MIN_ROWS));
        assert_eq!((b[0], *b.last().unwrap()), (0, 1 << 20));
        assert_eq!(row_bounds_at(10, 3, 8), vec![0, 3, 6, 10]);
        assert_eq!(row_bounds_at(10, 0, 1), vec![0, 10]);
    }

    /// Any bounds become a rising cover of `[0, rows)`: no row is left out.
    #[test]
    fn any_bounds_cover_every_row() {
        assert_eq!(covering(&[], 5), vec![0, 5]);
        assert_eq!(covering(&[0, 5], 5), vec![0, 5]);
        assert_eq!(covering(&[2, 3], 5), vec![0, 2, 3, 5]);
        assert_eq!(covering(&[0, 3, 3, 1, 9], 5), vec![0, 3, 5]);
        assert_eq!(covering(&[0, 0], 0), vec![0]);
    }
}
