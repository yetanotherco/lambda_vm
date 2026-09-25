//! Settling column claims against a **jagged** commitment: every table's real
//! rows packed back to back, with no power-of-two padding per table.
//!
//! [`stacked_eval`](crate::stacked_eval) gives each column a `2^m` subcube, so
//! a table of `h` rows pays for `2^⌈log h⌉`. Here a table is described by what
//! it actually holds: its first `h` rows, committed, and for every column a
//! **tail** — the value `c` its rows from `h` on repeat, stated in the clear. A
//! column that is constant all the way down commits nothing at all. The column
//! the table's argument is about is then
//!
//! ```text
//! column(z) = Σ_{r<h} eq(z, r)·column[r]  +  c·(1 − Σ_{r<h} eq(z, r))
//! ```
//!
//! The second term the verifier computes itself ([`lt_eval`]), so what is left
//! to prove is the first, a weighted sum over the committed dense polynomial:
//! one weighted WHIR opening per polynomial, the same [`whir_chain`] call the
//! stack uses.
//!
//! # The packing, and why the verifier's weight is cheap
//!
//! A table's committed columns are cut into **blocks** of `2^b` columns — the
//! binary digits of its width, so no column is padded — and each block is laid
//! out **row-major** at an offset aligned to `2^b`: row `r`, column `c` of the
//! block sits at `t + r·2^b + c`. Then the low `b` bits of a position are the
//! column and the high bits are `t/2^b + r`, so `eq(ρ, ·)` splits and the
//! block's whole share of the weight at `ρ` is
//!
//! ```text
//! [Σ_c γ^{i_c}·eq(ρ_lo, c)] · Σ_{r<h} eq(z, r)·eq(ρ_hi, t/2^b + r)
//! ```
//!
//! one multiplication per column and **one** branching program per block
//! ([`jagged_eval_range`]) rather than one per column. A block that does not fit
//! in what is left of a polynomial continues in the next, so only the last
//! polynomial has a tail of zeros.
//!
//! # Why the tails are sound to state
//!
//! The prover chooses `h` and `c`, and it could already choose every padding
//! row: they are rows like any other, held to the same constraints over the
//! whole cube. What must not happen is choosing them **after** a challenge, so
//! they are bound with the roots — [`tails_digest`] travels as one more root —
//! before anything is drawn.

use crypto::fiat_shamir::is_transcript::IsTranscript;
use crypto::hash::platform_keccak::PlatformKeccak256;
use math::{
    field::{
        element::FieldElement,
        traits::{IsFFTField, IsField, IsPrimeField, IsSubFieldOf},
    },
    traits::AsBytes,
};
use sha3::Digest;
use std::sync::atomic::{AtomicU64, Ordering};


use crate::{
    Error, challenge_powers,
    eq::eq_evals,
    mle::Mle,
    whir::Domain,
    whir_batch::{self, BatchProof},
    whir_chain::ChainConfig,
    whir_commit::Commitment,
};

/// Extension-field multiplications the verifier's closed forms have done — the
/// field work jagged adds to a recursive verifier, which a hash count does not
/// see.
pub static EVAL_MULS: AtomicU64 = AtomicU64::new(0);

/// Smallest `n` with `2^n >= x`.
fn ceil_log2(x: usize) -> usize {
    x.max(1).next_power_of_two().trailing_zeros() as usize
}

/// The smallest dense polynomial worth committing.
const MIN_VARS: usize = 2;

/// The most polynomials a packing is split into to save cells.
const MAX_POLYS: usize = 16;

/// `2^b` columns of one table, laid out row-major.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Block {
    pub table: usize,
    /// Real rows, shared by the block's columns.
    pub rows: usize,
    /// `log2` of the block's width.
    pub log_width: usize,
    /// The block's columns, in the group's column order.
    pub columns: Vec<usize>,
}

/// A run of one block's rows inside one polynomial.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Segment {
    pub poly: usize,
    /// Start inside the polynomial; a multiple of the block's width.
    pub offset: usize,
    pub block: usize,
    /// The block's first row this segment holds, and how many.
    pub row_start: usize,
    pub rows: usize,
}

/// A fixed, public packing of the tables' blocks into dense polynomials of
/// `2^num_vars` each. Both sides derive it from the stated tails and the
/// tables' widths, so it never travels.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct JaggedLayout {
    num_vars: usize,
    num_polys: usize,
    blocks: Vec<Block>,
    segments: Vec<Segment>,
}

impl JaggedLayout {
    /// The blocks of every table: its committed columns in order, cut at the
    /// binary digits of their count, widest first.
    fn cut_blocks(tables: &[usize], rows: &[usize]) -> Result<Vec<Block>, Error> {
        if tables.iter().sum::<usize>() != rows.len() {
            return Err(Error::QueryCountMismatch {
                expected: rows.len(),
                got: tables.iter().sum(),
            });
        }
        let mut blocks = Vec::new();
        let mut first = 0usize;
        for (table, &width) in tables.iter().enumerate() {
            let committed: Vec<usize> = (first..first + width).filter(|&c| rows[c] > 0).collect();
            // One height per table: that is what lets a block share one
            // branching program. A prover stating otherwise is refused rather
            // than handed a layout of one block per column.
            let height = committed.first().map_or(0, |&c| rows[c]);
            if committed.iter().any(|&c| rows[c] != height) {
                return Err(Error::VariableCountMismatch {
                    expected: height,
                    got: committed.iter().map(|&c| rows[c]).find(|&h| h != height).unwrap_or(0),
                });
            }
            let mut at = 0usize;
            for bit in (0..usize::BITS as usize).rev() {
                if committed.len() & (1usize << bit) != 0 {
                    blocks.push(Block {
                        table,
                        rows: height,
                        log_width: bit,
                        columns: committed[at..at + (1usize << bit)].to_vec(),
                    });
                    at += 1usize << bit;
                }
            }
            first += width;
        }
        // Widest first keeps every offset aligned to the width of the block
        // placed there: what came before is a multiple of a wider width.
        blocks.sort_by_key(|b| std::cmp::Reverse(b.log_width));
        Ok(blocks)
    }

    fn pack(blocks: &[Block], num_vars: usize) -> Self {
        let size = 1usize << num_vars;
        let (mut poly, mut at) = (0usize, 0usize);
        let mut segments = Vec::new();
        for (index, block) in blocks.iter().enumerate() {
            let mut row = 0usize;
            while row < block.rows {
                if at == size {
                    poly += 1;
                    at = 0;
                }
                let fit = (size - at) >> block.log_width;
                let take = fit.min(block.rows - row);
                segments.push(Segment {
                    poly,
                    offset: at,
                    block: index,
                    row_start: row,
                    rows: take,
                });
                at += take << block.log_width;
                row += take;
            }
        }
        Self {
            num_vars,
            num_polys: poly + 1,
            blocks: blocks.to_vec(),
            segments,
        }
    }

    /// The packing of `tables` (columns per table, in order) whose columns hold
    /// `rows[i]` real rows each, into polynomials of at most `2^max_vars`.
    ///
    /// The fewest cells, with at most [`MAX_POLYS`] polynomials. They are opened
    /// together ([`whir_batch`]), so another polynomial costs a wider leaf, not
    /// another authentication path; the cap keeps the leaves from growing
    /// without bound. Ties go to the larger size.
    pub fn build(tables: &[usize], rows: &[usize], max_vars: usize) -> Result<Self, Error> {
        let blocks = Self::cut_blocks(tables, rows)?;
        let widest = blocks.iter().map(|b| b.log_width).max().unwrap_or(0);
        let min_vars = widest.max(MIN_VARS);
        if min_vars > max_vars.max(MIN_VARS) {
            return Err(Error::ColumnTallerThanStack {
                column_vars: widest,
                n_stack: max_vars,
            });
        }
        let total: usize = blocks.iter().map(|b| b.rows << b.log_width).sum();
        let top = max_vars.max(min_vars).min(ceil_log2(total).max(min_vars));
        let mut best = Self::pack(&blocks, top);
        for n in (min_vars..top).rev() {
            let candidate = Self::pack(&blocks, n);
            if candidate.num_polys > MAX_POLYS {
                break;
            }
            if candidate.cells() < best.cells() {
                best = candidate;
            }
        }
        Ok(best)
    }

    pub fn num_vars(&self) -> usize {
        self.num_vars
    }

    pub fn num_polys(&self) -> usize {
        self.num_polys
    }

    pub fn blocks(&self) -> &[Block] {
        &self.blocks
    }

    pub fn segments(&self) -> &[Segment] {
        &self.segments
    }

    /// Everything committed, padding included.
    pub fn cells(&self) -> usize {
        self.num_polys << self.num_vars
    }

    fn segments_in(&self, poly: usize) -> impl Iterator<Item = &Segment> {
        self.segments.iter().filter(move |s| s.poly == poly)
    }

    /// Dense polynomial `poly`: its segments' rows, row-major, and zeros after
    /// the last.
    fn assemble<F: IsField + 'static>(
        &self,
        poly: usize,
        columns: &[&Mle<F>],
    ) -> Result<Mle<F>, Error> {
        let mut buffer = vec![FieldElement::<F>::zero(); 1usize << self.num_vars];
        for segment in self.segments_in(poly) {
            let block = &self.blocks[segment.block];
            let width = 1usize << block.log_width;
            let region = &mut buffer[segment.offset..segment.offset + (segment.rows << block.log_width)];
            for (c, &column) in block.columns.iter().enumerate() {
                let values = &columns[column].evals()[segment.row_start..segment.row_start + segment.rows];
                for (r, value) in values.iter().enumerate() {
                    region[r * width + c] = value.clone();
                }
            }
        }
        Mle::new(buffer)
    }
}

/// A column's tail: how many leading rows it really has, and the value every
/// row after them repeats.
#[derive(
    Clone,
    Debug,
    PartialEq,
    Eq,
    serde::Serialize,
    serde::Deserialize,
    rkyv::Archive,
    rkyv::Serialize,
    rkyv::Deserialize,
)]
#[serde(bound = "")]
pub struct Tail<F: IsField> {
    pub rows: u32,
    pub value: FieldElement<F>,
}

impl<F: IsField> Tail<F> {
    /// The shortest description of `column`: its last value, and the rows
    /// before the run of it that ends the column.
    pub fn of(column: &[FieldElement<F>]) -> Self {
        let value = column.last().cloned().unwrap_or_else(FieldElement::<F>::zero);
        let rows = column
            .iter()
            .rposition(|v| *v != value)
            .map_or(0, |last_other| last_other + 1);
        Self {
            rows: rows as u32,
            value,
        }
    }
}

/// Every table's tails at one height: the tallest of its columns', and zero for
/// a column that is constant all the way down. A block shares one height, so a
/// table's committed columns must too.
fn table_tails<F: IsField + 'static>(tables: &[usize], columns: &[&Mle<F>]) -> Vec<Tail<F>> {
    let mut tails: Vec<Tail<F>> = columns.iter().map(|c| Tail::of(c.evals())).collect();
    let mut first = 0usize;
    for &width in tables {
        let height = tails[first..first + width].iter().map(|t| t.rows).max().unwrap_or(0);
        for tail in &mut tails[first..first + width] {
            if tail.rows > 0 {
                tail.rows = height;
            }
        }
        first += width;
    }
    tails
}

/// `Σ_{r<h} eq(z, r)` — the weight a column's committed rows carry at `z`,
/// which is what its tail does not.
///
/// A two-state automaton over the bits of `r`, low to high, tracking whether
/// `r < h` on the bits seen so far. `z[0]` is the most significant variable,
/// as everywhere in this crate.
pub fn lt_eval<E: IsField>(z: &[FieldElement<E>], h: usize) -> FieldElement<E> {
    let m = z.len();
    if h >= (1usize << m) {
        return FieldElement::<E>::one();
    }
    let one = FieldElement::<E>::one();
    let mut dp = [one.clone(), FieldElement::<E>::zero()];
    EVAL_MULS.fetch_add(4 * m as u64, Ordering::Relaxed);
    for j in 0..m {
        let zj = &z[m - 1 - j];
        let hj = (h >> j) & 1;
        let mut next = [FieldElement::<E>::zero(), FieldElement::<E>::zero()];
        for (lt, w) in dp.iter().enumerate() {
            for rj in 0..2usize {
                let wz = if rj == 1 { zj.clone() } else { &one - zj };
                let lt_next = if rj != hj { (rj < hj) as usize } else { lt };
                next[lt_next] += w * &wz;
            }
        }
        dp = next;
    }
    dp[1].clone()
}

/// `J(z, ρ; t, h) = Σ_{r<h} eq(z, r)·eq(ρ, t + r)`, directly — the reference
/// [`jagged_eval_range`] is tested against.
pub fn jagged_eval<E: IsField>(
    z: &[FieldElement<E>],
    rho: &[FieldElement<E>],
    t: usize,
    h: usize,
) -> FieldElement<E> {
    jagged_eval_range(z, rho, 0, t, h)
}

/// `Σ_{r<len} eq(z, a + r)·eq(ρ, t + r)` — one segment's rows `a..a+len` of a
/// table claimed at `z`, sitting at `t..t+len` of a cube evaluated at `ρ`.
///
/// An automaton over the bits, low to high, with the carries of `a + r` and of
/// `t + r` and whether `r < len` so far: eight states, of which a segment
/// starting at row 0 only ever reaches four. Each bit's four weights
/// `eq1(z_j, u)·eq1(ρ_j, x)` are built once, so a transition is one
/// multiplication. Requires `a + len <= 2^z.len()` and `t + len <= 2^ρ.len()`.
pub fn jagged_eval_range<E: IsField>(
    z: &[FieldElement<E>],
    rho: &[FieldElement<E>],
    a: usize,
    t: usize,
    len: usize,
) -> FieldElement<E> {
    let (m, n) = (z.len(), rho.len());
    let one = FieldElement::<E>::one();
    let zero = FieldElement::<E>::zero();
    let bits = m.max(n) + 1;
    // dp[carry of a + r][carry of t + r][r < len so far]
    let mut dp = vec![zero.clone(); 8];
    dp[0] = one.clone();
    let mut muls = 0u64;
    for j in 0..bits {
        // Past its width a number's bit is zero, so its weight is the
        // indicator of 0.
        let wz = if j < m {
            let zj = &z[m - 1 - j];
            [&one - zj, zj.clone()]
        } else {
            [one.clone(), zero.clone()]
        };
        let wr = if j < n {
            let rj = &rho[n - 1 - j];
            [&one - rj, rj.clone()]
        } else {
            [one.clone(), zero.clone()]
        };
        let step = [
            [&wz[0] * &wr[0], &wz[0] * &wr[1]],
            [&wz[1] * &wr[0], &wz[1] * &wr[1]],
        ];
        muls += 4;
        let (aj, tj, lj) = ((a >> j) & 1, (t >> j) & 1, (len >> j) & 1);
        let mut next = vec![zero.clone(); 8];
        for state in 0..8usize {
            let w = &dp[state];
            if *w == zero {
                continue;
            }
            let (cu, cx, lt) = (state >> 2, (state >> 1) & 1, state & 1);
            for rj in 0..2usize {
                let (su, sx) = (aj + rj + cu, tj + rj + cx);
                let (uj, xj) = (su & 1, sx & 1);
                let weight = &step[uj][xj];
                if *weight == zero {
                    continue;
                }
                let lt_next = if rj != lj { (rj < lj) as usize } else { lt };
                let to = ((su >> 1) << 2) | ((sx >> 1) << 1) | lt_next;
                next[to] += w * weight;
                muls += 1;
            }
        }
        dp = next;
    }
    EVAL_MULS.fetch_add(muls, Ordering::Relaxed);
    dp[1].clone()
}

/// The digest the tails travel as: one more root, so whatever absorbs the roots
/// binds them too.
pub fn tails_digest<F: IsField>(tails: &[Tail<F>]) -> Commitment
where
    FieldElement<F>: AsBytes,
{
    let mut hasher = PlatformKeccak256::new();
    hasher.update(b"jagged-tails");
    hasher.update((tails.len() as u64).to_le_bytes());
    for tail in tails {
        hasher.update(tail.rows.to_le_bytes());
        hasher.update(tail.value.as_bytes());
    }
    hasher.finalize().into()
}

/// The dense polynomials, committed, and the tails that complete the columns.
pub struct JaggedCommitment<F: IsFFTField + IsPrimeField>
where
    FieldElement<F>: AsBytes + Sync + Send,
{
    layout: JaggedLayout,
    tables: Vec<usize>,
    tails: Vec<Tail<F>>,
    /// Every dense polynomial in one shared tree ([`whir_batch`]).
    commitment: whir_batch::WideCommitment<F>,
    domain: Domain<F>,
}

impl<F: IsFFTField + IsPrimeField + Send + Sync + 'static> JaggedCommitment<F>
where
    FieldElement<F>: AsBytes + Sync + Send,
{
    /// Trims every table to its tails, packs what is left and commits it.
    /// `tables` is how many of `columns` each table holds, in order.
    pub fn commit(
        columns: &[&Mle<F>],
        tables: &[usize],
        max_vars: usize,
        config: &ChainConfig,
    ) -> Result<Self, Error> {
        let tails = table_tails(tables, columns);
        let rows: Vec<usize> = tails.iter().map(|t| t.rows as usize).collect();
        let layout = JaggedLayout::build(tables, &rows, max_vars)?;
        let dense: Vec<Mle<F>> = (0..layout.num_polys())
            .map(|poly| layout.assemble(poly, columns))
            .collect::<Result<_, _>>()?;
        let (commitment, domain) = whir_batch::commit(&dense, config)?;
        Ok(Self {
            layout,
            tables: tables.to_vec(),
            tails,
            commitment,
            domain,
        })
    }

    /// The shared tree's root, then the tails' digest.
    pub fn roots(&self) -> Vec<Commitment> {
        vec![self.commitment.root(), tails_digest(&self.tails)]
    }

    pub fn layout(&self) -> &JaggedLayout {
        &self.layout
    }

    pub fn tails(&self) -> &[Tail<F>] {
        &self.tails
    }

    pub fn domain(&self) -> &Domain<F> {
        &self.domain
    }
}

/// The openings, one per dense polynomial, and the tails the verifier rebuilds
/// the layout from.
#[derive(
    Clone,
    Debug,
    serde::Serialize,
    serde::Deserialize,
    rkyv::Archive,
    rkyv::Serialize,
    rkyv::Deserialize,
)]
#[serde(bound = "")]
pub struct JaggedProof<F: IsField, E: IsField> {
    pub tails: Vec<Tail<F>>,
    /// One batched opening of every dense polynomial. Its claim is the total the
    /// columns owe their committed rows, so a block split across polynomials
    /// needs no apportioning.
    pub opening: BatchProof<F, E>,
}

impl<F: IsField, E: IsField> JaggedProof<F, E> {
    pub fn opened_elements(&self) -> usize {
        self.opening.opened_elements()
    }
}

/// How many roots a jagged group takes: the shared tree's, and the tails'
/// digest.
pub fn num_roots<F: IsField, E: IsField>(
    _proof: &JaggedProof<F, E>,
    _tables: &[usize],
    _max_vars: usize,
) -> Result<usize, Error> {
    Ok(2)
}

/// Each table's point: the one every column of it is claimed at.
fn table_points<'a, E: IsField>(
    tables: &[usize],
    points: &'a [Vec<FieldElement<E>>],
) -> Result<Vec<&'a [FieldElement<E>]>, Error> {
    let mut out = Vec::with_capacity(tables.len());
    let mut first = 0usize;
    for &width in tables {
        let point = points.get(first).map_or(&[][..], Vec::as_slice);
        if points[first..first + width].iter().any(|p| p.as_slice() != point) {
            return Err(Error::EvaluationMismatch);
        }
        out.push(point);
        first += width;
    }
    Ok(out)
}

/// Checks the claims line up with the columns, and every height fits its
/// table's cube.
fn check_shape<F: IsField, E: IsField>(
    tails: &[Tail<F>],
    points: &[Vec<FieldElement<E>>],
    values: &[FieldElement<E>],
) -> Result<(), Error> {
    if points.len() != tails.len() || values.len() != tails.len() {
        return Err(Error::QueryCountMismatch {
            expected: tails.len(),
            got: points.len().min(values.len()),
        });
    }
    for (tail, point) in tails.iter().zip(points) {
        if point.len() >= usize::BITS as usize || tail.rows as usize > (1usize << point.len()) {
            return Err(Error::VariableCountMismatch {
                expected: point.len(),
                got: tail.rows as usize,
            });
        }
    }
    Ok(())
}

/// Every table's `L(z, h)`, once per table.
fn table_lts<E: IsField>(
    tables: &[usize],
    tails_rows: &[usize],
    table_points: &[&[FieldElement<E>]],
) -> Vec<FieldElement<E>> {
    let mut first = 0usize;
    tables
        .iter()
        .zip(table_points)
        .map(|(&width, point)| {
            let height = tails_rows[first..first + width].iter().copied().max().unwrap_or(0);
            first += width;
            lt_eval(point, height)
        })
        .collect()
}

/// Proves every column takes its claimed value at its table's point.
#[allow(clippy::too_many_arguments)]
pub fn prove<F, E, T>(
    committed: &JaggedCommitment<F>,
    columns: &[&Mle<F>],
    points: &[Vec<FieldElement<E>>],
    values: &[FieldElement<E>],
    config: &ChainConfig,
    transcript: &mut T,
) -> Result<JaggedProof<F, E>, Error>
where
    F: IsFFTField + IsPrimeField + IsSubFieldOf<E> + Send + Sync + 'static,
    E: IsField + Send + Sync + 'static,
    FieldElement<F>: AsBytes + Sync + Send,
    FieldElement<E>: AsBytes + Sync + Send,
    T: IsTranscript<E>,
{
    let layout = &committed.layout;
    check_shape(&committed.tails, points, values)?;
    let table_points = table_points(&committed.tables, points)?;
    for value in values {
        transcript.append_field_element(value);
    }
    let weights = challenge_powers(&transcript.sample_field_element(), values.len());

    let mut weights_all = Vec::with_capacity(layout.num_polys());
    let mut dense = Vec::with_capacity(layout.num_polys());
    for poly in 0..layout.num_polys() {
        // W(x) = γ^{i_c}·eq(z, r) at row r, column c of every segment.
        let mut weight = vec![FieldElement::<E>::zero(); 1usize << layout.num_vars()];
        let mut eqs: Vec<Option<Vec<FieldElement<E>>>> = vec![None; committed.tables.len()];
        for segment in layout.segments_in(poly) {
            let block = &layout.blocks[segment.block];
            let eq = eqs[block.table].get_or_insert_with(|| eq_evals(table_points[block.table]));
            let width = 1usize << block.log_width;
            let region = &mut weight[segment.offset..segment.offset + (segment.rows << block.log_width)];
            for (r, row) in region.chunks_mut(width).enumerate() {
                let e = &eq[segment.row_start + r];
                for (cell, &column) in row.iter_mut().zip(&block.columns) {
                    *cell = &weights[column] * e;
                }
            }
        }
        weights_all.push(Mle::new(weight)?);
        dense.push(layout.assemble(poly, columns)?);
    }
    let opening = whir_batch::prove::<F, E, T>(
        &dense,
        weights_all,
        &committed.commitment,
        &committed.domain,
        config,
        transcript,
    )?;
    Ok(JaggedProof {
        tails: committed.tails.clone(),
        opening,
    })
}

/// Verifies the claims against the roots: `roots` is the polynomials' roots
/// followed by the tails' digest, exactly as [`JaggedCommitment::roots`] gave
/// them. `tables` is how many of the columns each table holds, in order.
#[allow(clippy::too_many_arguments)]
pub fn verify<F, E, T>(
    proof: &JaggedProof<F, E>,
    roots: &[Commitment],
    tables: &[usize],
    points: &[Vec<FieldElement<E>>],
    values: &[FieldElement<E>],
    max_vars: usize,
    config: &ChainConfig,
    transcript: &mut T,
) -> Result<(), Error>
where
    F: IsFFTField + IsPrimeField + IsSubFieldOf<E> + 'static,
    E: IsField + Send + Sync + 'static,
    FieldElement<F>: AsBytes + Sync + Send,
    FieldElement<E>: AsBytes + Sync + Send,
    T: IsTranscript<E>,
{
    check_shape(&proof.tails, points, values)?;
    let table_points = table_points(tables, points)?;
    let rows: Vec<usize> = proof.tails.iter().map(|t| t.rows as usize).collect();
    let layout = JaggedLayout::build(tables, &rows, max_vars)?;
    if roots.len() != 2 {
        return Err(Error::QueryCountMismatch {
            expected: 2,
            got: roots.len(),
        });
    }
    if roots[1] != tails_digest(&proof.tails) {
        return Err(Error::EvaluationMismatch);
    }
    for value in values {
        transcript.append_field_element(value);
    }
    let weights = challenge_powers(&transcript.sample_field_element(), values.len());

    // What each claim owes the committed rows: v − c·(1 − L(z, h)).
    let lts = table_lts(tables, &rows, &table_points);
    let mut table_of = Vec::with_capacity(rows.len());
    for (table, &width) in tables.iter().enumerate() {
        table_of.extend(std::iter::repeat_n(table, width));
    }
    let one = FieldElement::<E>::one();
    let parts: Vec<FieldElement<E>> = values
        .iter()
        .zip(&proof.tails)
        .zip(&table_of)
        .map(|((v, tail), &table)| {
            if tail.value == FieldElement::<F>::zero() {
                return v.clone();
            }
            // A column with no committed row owes all of itself to its tail.
            let rest = if tail.rows == 0 { one.clone() } else { &one - &lts[table] };
            v - tail.value.clone().to_extension::<E>() * rest
        })
        .collect();
    // What the committed rows owe, in total. A column with nothing committed
    // has no weight anywhere, so this is also what holds it to its tail.
    let owed = parts
        .iter()
        .zip(&weights)
        .fold(FieldElement::<E>::zero(), |acc, (part, w)| acc + w * part);

    let domain = Domain::<F>::new(layout.num_vars() + config.log_blowup)?;
    let n = layout.num_vars();
    whir_batch::verify::<F, E, T, _>(
        &proof.opening,
        &roots[0],
        layout.num_polys(),
        |poly: usize, rho: &[FieldElement<E>]| {
            // eq(ρ_lo, c) for each block width, once per polynomial.
            let mut low: Vec<Option<Vec<FieldElement<E>>>> = vec![None; n + 1];
            let mut total = FieldElement::<E>::zero();
            for segment in layout.segments_in(poly) {
                let block = &layout.blocks[segment.block];
                let b = block.log_width;
                let eq_low = low[b].get_or_insert_with(|| {
                    EVAL_MULS.fetch_add(1u64 << b, Ordering::Relaxed);
                    eq_evals(&rho[n - b..])
                });
                EVAL_MULS.fetch_add(block.columns.len() as u64, Ordering::Relaxed);
                let columns = block
                    .columns
                    .iter()
                    .zip(eq_low.iter())
                    .fold(FieldElement::<E>::zero(), |acc, (&c, e)| acc + &weights[c] * e);
                let rows = jagged_eval_range(
                    table_points[block.table],
                    &rho[..n - b],
                    segment.row_start,
                    segment.offset >> b,
                    segment.rows,
                );
                total += columns * rows;
            }
            Ok(total)
        },
        owed,
        n,
        &domain,
        config,
        transcript,
    )
    .map_err(|_| Error::ColumnOpeningRejected { column: 0 })?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eq::eq_eval;
    use crate::whir_chain::GrindBits;
    use crypto::fiat_shamir::default_transcript::DefaultTranscript;
    use math::field::goldilocks::GoldilocksField as F;

    type FE = FieldElement<F>;

    fn bits(x: usize, n: usize) -> Vec<FE> {
        (0..n)
            .map(|i| FE::from(((x >> (n - 1 - i)) & 1) as u64))
            .collect()
    }

    fn point(n: usize, seed: u64) -> Vec<FE> {
        (0..n)
            .map(|i| FE::from(seed.wrapping_mul(0x9E37_79B9) ^ (i as u64 * 7919 + 3)))
            .collect()
    }

    fn config() -> ChainConfig {
        ChainConfig {
            log_blowup: 2,
            log_folding: 2,
            num_queries: 3,
            grind: GrindBits::default(),
        }
    }

    #[test]
    fn lt_eval_matches_brute_force() {
        for m in 0..6 {
            let z = point(m, 11 + m as u64);
            for h in 0..=(1usize << m) {
                let brute = (0..h).fold(FE::zero(), |acc, r| acc + eq_eval(&z, &bits(r, m)).unwrap());
                assert_eq!(lt_eval(&z, h), brute, "m={m} h={h}");
            }
        }
    }

    #[test]
    fn jagged_eval_range_matches_brute_force() {
        for m in 0..4 {
            for n in 0..5 {
                let z = point(m, 5 + m as u64);
                let rho = point(n, 17 + n as u64);
                for a in 0..=(1usize << m) {
                    for len in 0..=((1usize << m) - a).min(1usize << n) {
                        for t in 0..=((1usize << n) - len) {
                            let brute = (0..len).fold(FE::zero(), |acc, r| {
                                acc + eq_eval(&z, &bits(a + r, m)).unwrap()
                                    * eq_eval(&rho, &bits(t + r, n)).unwrap()
                            });
                            assert_eq!(
                                jagged_eval_range(&z, &rho, a, t, len),
                                brute,
                                "m={m} n={n} a={a} t={t} len={len}"
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn tail_is_the_shortest_run() {
        let v = |xs: &[u64]| xs.iter().map(|&x| FE::from(x)).collect::<Vec<_>>();
        assert_eq!(Tail::of(&v(&[1, 2, 3, 0, 0, 0])).rows, 3);
        assert_eq!(Tail::of(&v(&[7, 7, 7, 7])).rows, 0);
        assert_eq!(Tail::of(&v(&[1, 2, 5, 5])).rows, 2);
        assert_eq!(Tail::of(&v(&[1, 2, 3, 4])).rows, 3);
        assert_eq!(Tail::of(&v(&[1, 2, 3, 4])).value, FE::from(4));
    }

    #[test]
    fn a_table_commits_at_one_height() {
        let v = |xs: &[u64]| Mle::new(xs.iter().map(|&x| FE::from(x)).collect()).unwrap();
        let cols = [v(&[1, 2, 0, 0]), v(&[1, 2, 3, 0]), v(&[5, 5, 5, 5])];
        let refs: Vec<&Mle<F>> = cols.iter().collect();
        let rows: Vec<u32> = table_tails(&[3], &refs).iter().map(|t| t.rows).collect();
        assert_eq!(rows, vec![3, 3, 0]);
    }

    /// Every committed cell lands exactly once, offsets are aligned to their
    /// block's width, and only the last polynomial has room left.
    fn check_layout(tables: &[usize], rows: &[usize], max_vars: usize) -> JaggedLayout {
        let layout = JaggedLayout::build(tables, rows, max_vars).unwrap();
        let size = 1usize << layout.num_vars();
        let mut used = vec![vec![false; size]; layout.num_polys()];
        let mut covered = std::collections::HashMap::new();
        for s in layout.segments() {
            let block = &layout.blocks()[s.block];
            assert_eq!(s.offset % (1 << block.log_width), 0);
            for r in 0..s.rows {
                for (c, &column) in block.columns.iter().enumerate() {
                    let x = s.offset + (r << block.log_width) + c;
                    assert!(!used[s.poly][x], "two cells collide");
                    used[s.poly][x] = true;
                    assert!(covered.insert((column, s.row_start + r), ()).is_none());
                }
            }
        }
        for (column, &h) in rows.iter().enumerate() {
            for r in 0..h {
                assert!(covered.contains_key(&(column, r)), "column {column} row {r} missing");
            }
        }
        for poly in &used[..layout.num_polys() - 1] {
            assert!(poly.iter().all(|&u| u), "a polynomial before the last is not full");
        }
        layout
    }

    #[test]
    fn layout_covers_every_row_once() {
        check_layout(&[3, 5, 1], &[13, 13, 13, 7, 7, 0, 7, 7, 64], 8);
    }

    #[test]
    fn layout_splits_a_block_across_polynomials() {
        // A cap of 2^4 forces the 5-wide table's blocks over several.
        let layout = check_layout(&[5, 2], &[9, 9, 9, 9, 9, 6, 6], 4);
        assert!(layout.num_polys() > 1);
        assert!(layout.segments().iter().any(|s| s.row_start > 0));
    }

    #[test]
    fn layout_refuses_two_heights_in_one_table() {
        assert!(JaggedLayout::build(&[2], &[5, 6], 6).is_err());
    }

    /// Three tables, widths 3, 5 and 2, with zero, nonzero and absent tails.
    fn tables() -> (Vec<usize>, Vec<Mle<F>>) {
        let make = |num_vars: usize, real: usize, tail: u64, seed: u64| {
            Mle::new(
                (0..(1usize << num_vars))
                    .map(|i| {
                        if i < real {
                            FE::from((i as u64 + 1).wrapping_mul(seed) % 1_000_003 + 1_000_000)
                        } else {
                            FE::from(tail)
                        }
                    })
                    .collect(),
            )
            .unwrap()
        };
        let columns = vec![
            make(4, 11, 0, 3),
            make(4, 11, 0, 5),
            make(4, 16, 0, 7),
            make(3, 5, 42, 11),
            make(3, 5, 0, 13),
            make(3, 0, 9, 17),
            make(3, 3, 1, 19),
            make(3, 5, 0, 23),
            make(5, 20, 0, 29),
            make(5, 32, 0, 31),
        ];
        (vec![3, 5, 2], columns)
    }

    fn claims(tables: &[usize], columns: &[Mle<F>]) -> (Vec<Vec<FE>>, Vec<FE>) {
        let mut points = Vec::new();
        let mut first = 0usize;
        for (t, &width) in tables.iter().enumerate() {
            let p = point(columns[first].num_vars(), 100 + t as u64);
            points.extend(std::iter::repeat_n(p, width));
            first += width;
        }
        let values = columns
            .iter()
            .zip(&points)
            .map(|(c, p)| c.evaluate(p).unwrap())
            .collect();
        (points, values)
    }

    fn run(
        max_vars: usize,
        tamper: impl FnOnce(&mut Vec<FE>, &mut JaggedProof<F, F>),
    ) -> Result<(), Error> {
        let (tables, columns) = tables();
        let refs: Vec<&Mle<F>> = columns.iter().collect();
        let committed = JaggedCommitment::<F>::commit(&refs, &tables, max_vars, &config())?;
        let roots = committed.roots();
        let (points, mut values) = claims(&tables, &columns);
        let mut prover = DefaultTranscript::<F>::new(b"jagged");
        let mut proof = prove::<F, F, _>(&committed, &refs, &points, &values, &config(), &mut prover)?;
        tamper(&mut values, &mut proof);
        let mut verifier = DefaultTranscript::<F>::new(b"jagged");
        verify::<F, F, _>(&proof, &roots, &tables, &points, &values, max_vars, &config(), &mut verifier)
    }

    #[test]
    fn honest_claims_verify() {
        run(7, |_, _| {}).unwrap();
    }

    #[test]
    fn honest_claims_verify_with_blocks_split_across_polynomials() {
        run(5, |_, _| {}).unwrap();
    }

    #[test]
    fn a_wrong_value_is_rejected() {
        assert!(run(7, |values, _| values[3] += FE::one()).is_err());
        assert!(run(5, |values, _| values[9] += FE::one()).is_err());
    }

    #[test]
    fn a_wrong_value_on_an_uncommitted_column_is_rejected() {
        // Column 5 is all tail.
        assert!(run(7, |values, _| values[5] += FE::one()).is_err());
    }

    #[test]
    fn a_restated_tail_is_rejected() {
        assert!(run(7, |_, proof| proof.tails[3].value += FE::one()).is_err());
        assert!(run(7, |_, proof| {
            for tail in &mut proof.tails[0..3] {
                tail.rows -= 1;
            }
        })
        .is_err());
    }

    #[test]
    fn every_polynomial_is_opened_at_once() {
        let (tables, columns) = tables();
        let refs: Vec<&Mle<F>> = columns.iter().collect();
        let committed = JaggedCommitment::<F>::commit(&refs, &tables, 5, &config()).unwrap();
        assert!(committed.layout().num_polys() > 1);
        assert_eq!(committed.roots().len(), 2);
    }
}
