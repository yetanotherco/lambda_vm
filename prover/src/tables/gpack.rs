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

use stark::narrow::NarrowWriter;
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
            // Where a trace cannot be held packed (`stark`'s `debug-checks`),
            // its words at 8 bytes a cell.
            Ok(TraceTable::try_from_narrow_main(narrow, 1)
                .unwrap_or_else(|narrow| widened(&narrow)))
        }
        Err(need) => {
            hint.learn(&need);
            Err(need)
        }
    }
}

/// `narrow`'s words as a 64-bit trace.
fn widened(narrow: &stark::narrow::NarrowMain) -> Trace {
    let mut words = vec![0u64; narrow.rows() * narrow.cols()];
    narrow.widen_into(&mut words);
    TraceTable::new_main(words.into_iter().map(FE::from).collect(), narrow.cols(), 1)
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
                Err(need) => match write(&need, true) {
                    Ok(trace) => trace,
                    // Not reached: every word fits the widths of every word.
                    Err(_) => wide(true),
                },
            },
        }
    }};
}
pub(crate) use generate_main;
