//! The opening's two factors, resident across the groups of rounds.
//!
//! A chained WHIR does not run one sumcheck: it runs a group of rounds, folds
//! its codeword, commits the successor, answers an out-of-domain point and
//! folds the next group's weight into the one it carries. The factors — the
//! message and the weight, each the width of a stacked polynomial — are the
//! same two tables throughout, so they stay here and the host drives them.
//!
//! The message arrives in the base field, which is where it is committed, and
//! is lifted on the way in: the sumcheck runs in one field.

use std::sync::Arc;

use cudarc::driver::{CudaSlice, CudaStream, DevicePtr, LaunchConfig, PushKernelArg};

use crate::Result;
use crate::device::backend;
use crate::sumcheck::SumcheckSession;

/// The weight and the message, on device, with the cube they currently span.
pub struct OpeningSession {
    stream: Arc<CudaStream>,
    weight: Arc<CudaSlice<u64>>,
    message: Arc<CudaSlice<u64>>,
    len: usize,
}

impl OpeningSession {
    /// `weight` is interleaved ext3; `message` is the base-field polynomial
    /// being opened, lifted here.
    pub fn new(weight: &[u64], message: &[u64]) -> Result<Self> {
        assert!(weight.len().is_multiple_of(3), "three u64 per ext3 element");
        let len = weight.len() / 3;
        let be = backend()?;
        let stream = be.next_stream();
        let weight_dev = crate::device::htod_or_trim(&stream, weight)?;
        Self::with_weight(stream, weight_dev, len, message)
    }

    /// The same with the weight built here: each share is a column's subcube
    /// offset, the point it is claimed at, and the scale it carries, and its
    /// cells are `scale·eq(point, ·)`. The gaps stay zero.
    ///
    /// A stacked polynomial's weight is as wide as the polynomial — writing it
    /// on the host and sending it costs more than the rounds that read it.
    pub fn from_shares(
        shares: &[(usize, Vec<u64>, [u64; 3])],
        len: usize,
        message: &[u64],
    ) -> Result<Self> {
        let be = backend()?;
        let stream = be.next_stream();
        // Zeroed because the gaps between the shares' subcubes are part of the
        // weight and nothing writes them.
        let mut weight = crate::device::alloc_zeros_or_trim::<u64>(&stream, len * 3)?;
        crate::sumcheck::eq_expand_shares_ext3(&stream, &mut weight, shares)?;
        Self::with_weight(stream, weight, len, message)
    }

    /// The same again with the message given as the columns it is stacked from,
    /// so it is never assembled on the host — `(column, offset in elements)`.
    pub fn from_shares_and_parts(
        shares: &[(usize, Vec<u64>, [u64; 3])],
        len: usize,
        parts: &[(&[u64], usize)],
    ) -> Result<Self> {
        Self::from_shares_and_source(shares, len, |stream, base| {
            for (column, offset) in parts {
                let mut at = base.slice_mut(*offset..*offset + column.len());
                stream.memcpy_htod(*column, &mut at)?;
            }
            Ok(())
        })
    }

    /// The same for parts the card already holds: `(column index, offset)` into
    /// the epoch's columns, so the message is assembled there instead of
    /// crossing the bus a second time.
    pub fn from_shares_and_resident(
        shares: &[(usize, Vec<u64>, [u64; 3])],
        len: usize,
        store: &crate::columns::DeviceColumns,
        parts: &[(usize, usize)],
    ) -> Result<Self> {
        Self::from_shares_and_source(shares, len, |stream, base| {
            for (column, offset) in parts {
                store.copy_into(*column, base, *offset, stream)?;
            }
            Ok(())
        })
    }

    fn from_shares_and_source(
        shares: &[(usize, Vec<u64>, [u64; 3])],
        len: usize,
        write: impl FnOnce(&Arc<CudaStream>, &mut CudaSlice<u64>) -> Result<()>,
    ) -> Result<Self> {
        let be = backend()?;
        let stream = be.next_stream();
        let mut weight = crate::device::alloc_zeros_or_trim::<u64>(&stream, len * 3)?;
        crate::sumcheck::eq_expand_shares_ext3(&stream, &mut weight, shares)?;
        // Zeroed because what the parts do not cover is the stacking's padding.
        let mut base = crate::device::alloc_zeros_or_trim::<u64>(&stream, len)?;
        write(&stream, &mut base)?;
        Self::lift_into(stream, weight, len, base)
    }

    fn with_weight(
        stream: Arc<CudaStream>,
        weight_dev: CudaSlice<u64>,
        len: usize,
        message: &[u64],
    ) -> Result<Self> {
        assert_eq!(message.len(), len, "the message spans the weight's cube");
        let base = crate::device::htod_or_trim(&stream, message)?;
        Self::lift_into(stream, weight_dev, len, base)
    }

    fn lift_into(
        stream: Arc<CudaStream>,
        weight_dev: CudaSlice<u64>,
        len: usize,
        base: CudaSlice<u64>,
    ) -> Result<Self> {
        assert!(len.is_power_of_two(), "the cube is a power of two");

        let be = backend()?;
        // SAFETY: the kernel writes every element it is sized for.
        let mut lifted = unsafe { stream.alloc::<u64>(len * 3) }?;
        let count = len as u64;
        unsafe {
            stream
                .launch_builder(&be.mle_lift_base_ext3)
                .arg(&base)
                .arg(&count)
                .arg(&mut lifted)
                .launch(LaunchConfig::for_num_elems(len as u32))?;
        }

        Ok(Self {
            stream,
            weight: Arc::new(weight_dev),
            message: Arc::new(lifted),
            len,
        })
    }

    /// A session over factors already on the card, both extension-valued and
    /// `len` wide — what [`LeanRound0::materialize`] hands over once its
    /// rounds have bound the first variables.
    pub fn from_tables(
        stream: Arc<CudaStream>,
        weight: CudaSlice<u64>,
        message: CudaSlice<u64>,
        len: usize,
    ) -> Self {
        assert!(len.is_power_of_two(), "the cube is a power of two");
        assert!(
            weight.len() >= len * 3 && message.len() >= len * 3,
            "the tables span the cube"
        );
        Self {
            stream,
            weight: Arc::new(weight),
            message: Arc::new(message),
            len,
        }
    }

    /// What a session over `2^num_vars` built from shares holds — the weight
    /// and the lifted message, three limbs each — and the base-field staging it
    /// builds the message through, which is freed once the lift has run.
    pub const fn planned_bytes(num_vars: usize) -> (u64, u64) {
        (48u64 << num_vars, 8u64 << num_vars)
    }

    /// Device bytes the two tables occupy — their allocations, which the rounds
    /// fold in place and never shrink.
    pub fn device_bytes(&self) -> u64 {
        ((self.weight.len() + self.message.len()) * 8) as u64
    }

    /// The weight's and the message's live cube, `len` ext3 values each, read
    /// back — for the parity tests against the lean opening.
    pub fn tables_to_host(&self) -> Result<(Vec<u64>, Vec<u64>)> {
        let weight = self
            .stream
            .clone_dtoh(&self.weight.slice(0..self.len * 3))?;
        let message = self
            .stream
            .clone_dtoh(&self.message.slice(0..self.len * 3))?;
        self.stream.synchronize()?;
        Ok((weight, message))
    }

    /// Cube indices the factors still span.
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn num_vars(&self) -> usize {
        self.len.trailing_zeros() as usize
    }

    /// A sumcheck over `[weight, message]` as they stand.
    ///
    /// The rounds fold them in place, so the caller tells this session how far
    /// they got with [`bound`](Self::bound).
    pub fn sumcheck(
        &self,
        nodes: &[u64],
        consts: &[u64],
        num_slots: usize,
        root_slot: u32,
    ) -> Result<SumcheckSession> {
        let addresses = {
            let (weight, _weight_guard) = self.weight.device_ptr(&self.stream);
            let (message, _message_guard) = self.message.device_ptr(&self.stream);
            [weight, message]
        };
        SumcheckSession::from_device(
            self.stream.clone(),
            &addresses,
            self.len,
            vec![self.weight.clone(), self.message.clone()],
            nodes,
            consts,
            num_slots,
            root_slot,
        )
    }

    /// Records that `rounds` variables have been bound by a sumcheck over
    /// these factors.
    pub fn bound(&mut self, rounds: usize) {
        self.len >>= rounds;
    }

    /// The message's value at `point`, which the out-of-domain answer needs.
    ///
    /// Folds a copy: the message is the next group's factor and the rounds
    /// have not bound this point.
    pub fn evaluate_message(&self, point: &[u64]) -> Result<[u64; 3]> {
        assert_eq!(point.len(), self.num_vars() * 3, "the point spans the cube");
        let mut copy = self.stream.alloc_zeros::<u64>(self.len * 3)?;
        {
            let source = self.message.slice(0..self.len * 3);
            self.stream.memcpy_dtod(&source, &mut copy)?;
        }
        crate::sumcheck::evaluate_resident_ext3(&self.stream, &mut copy, self.len, point)
    }

    /// `weight += scale · eq(point, ·)`, the weight the next group carries.
    pub fn add_scaled_eq(&self, point: &[u64], scale: &[u64]) -> Result<()> {
        assert_eq!(point.len(), self.num_vars() * 3, "the point spans the cube");
        assert_eq!(scale.len(), 3, "an ext3 scale");
        let be = backend()?;
        let eq = crate::sumcheck::eq_table_ext3(&self.stream, point, self.len)?;
        let scale_dev = crate::device::htod_or_trim(&self.stream, scale)?;
        let count = self.len as u64;
        // The kernel writes through a shared reference, the way the tree's
        // layers are folded: what orders these is the stream, and the weight
        // has one owner.
        unsafe {
            self.stream
                .launch_builder(&be.add_scaled_ext3)
                .arg(self.weight.as_ref())
                .arg(&eq)
                .arg(&count)
                .arg(&scale_dev)
                .launch(LaunchConfig::for_num_elems(self.len as u32))?;
        }
        Ok(())
    }
}

/// Nodes a lean round evaluates at (`WHIR_LEAN_MAX_T` in `whir_fold.cu`): the
/// opening's product is degree two, so two.
const LEAN_MAX_T: usize = 4;
/// The partial-sum grid a lean round is capped at, as the sumcheck's rounds are.
const LEAN_MAX_GRID: u32 = 4096;
const LEAN_BLOCK: u32 = 256;
/// Share ids are `u16` on the card, and `0xFFFF` marks a position no column
/// covers.
const LEAN_MAX_SHARES: usize = 0xFFFF;
/// The round scratch a lean opening keeps: the nodes, one partial per node
/// and block, and the sums.
const LEAN_SCRATCH_BYTES: u64 =
    ((LEAN_MAX_T * 3 + LEAN_MAX_T * LEAN_MAX_GRID as usize * 3 + LEAN_MAX_T * 3) * 8) as u64;

/// One column of a stacked polynomial as the lean rounds read it: where it sits
/// in the stack, how many variables it spans, its weight's scale, and where its
/// point's `eq` table sits in the shared buffer, in two halves.
///
/// `eq(z, row) = hi[row >> lo_bits] · lo[row & (2^lo_bits − 1)]`: the high half
/// is the table of the point's first `num_vars − lo_bits` coordinates, the low
/// half of the rest, each indexed with its most significant bit on its first
/// coordinate, as every `eq` table here is.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LeanShare {
    /// The first stack index the column occupies.
    pub stack_offset: usize,
    /// Its variables: it spans `2^num_vars` stack indices.
    pub num_vars: usize,
    /// Variables in the low half of its point's `eq` table.
    pub lo_bits: usize,
    /// Where the high and the low half start in the `eq` buffer, in ext3
    /// elements.
    pub hi_at: usize,
    pub lo_at: usize,
    /// The weight's scale, as ext3 limbs.
    pub scale: [u64; 3],
}

/// Where a lean opening reads the message.
pub enum LeanMessage<'a> {
    /// In the epoch's resident columns: `(store column, stack offset)` per
    /// part. Nothing is copied.
    Resident {
        store: &'a crate::columns::DeviceColumns,
        parts: &'a [(usize, usize)],
    },
    /// Host columns at their stack offsets, staged once into a base-field
    /// buffer the width of the stack — what the materialised path uploads too,
    /// before lifting it.
    Parts(&'a [(&'a [u64], usize)]),
}

/// A WHIR chain's first rounds over its factors in SHARE form
/// (`whir_lean_*` in `whir_fold.cu`).
///
/// The materialised opening ([`OpeningSession`]) holds the weight and the
/// lifted message at the stack's full width — `2 · 2^n` extension values, 1.5×
/// the committed base codeword, plus the base staging while it is built — for
/// rounds that bind the top variables first. Here the weight stays a list of
/// shares and small `eq` tables, the message is read where the columns lie, a
/// `u16` per stack index says which column holds it, and each round's
/// evaluations are computed from those with the challenges bound so far. The
/// factors exist only once [`materialize`](Self::materialize) writes them at
/// `2^(n − bound)`.
///
/// ★ The evaluations and the bound tables EQUAL the materialised path's
/// (`tests/whir_lean.rs`): the same field values by a different sequence of
/// operations, so raw limbs agree whenever both are canonical — the parity the
/// device path already has with the host.
pub struct LeanRound0 {
    stream: Arc<CudaStream>,
    /// The stack's variables.
    num_vars: usize,
    colmap: CudaSlice<u16>,
    shares: CudaSlice<u64>,
    eq: CudaSlice<u64>,
    data: Arc<CudaSlice<u64>>,
    /// Whether `data` is a staging buffer this opening allocated.
    staged: bool,
    /// The challenges bound so far, three limbs each.
    bound: Vec<u64>,
    t_dev: CudaSlice<u64>,
    t_host: Vec<u64>,
    partials: CudaSlice<u64>,
    sums: CudaSlice<u64>,
}

impl LeanRound0 {
    /// `Ok(None)` when the shares do not describe a stack this can read — out
    /// of order, overlapping, past the end, more than a `u16` can name, or a
    /// part missing for one of them — and the caller materialises instead.
    pub fn new(
        shares: &[LeanShare],
        eq: &[u64],
        num_vars: usize,
        message: LeanMessage<'_>,
    ) -> Result<Option<Self>> {
        let len = 1usize << num_vars;
        if shares.is_empty() || shares.len() >= LEAN_MAX_SHARES || eq.is_empty() {
            return Ok(None);
        }
        let mut end = 0usize;
        for share in shares {
            let span = 1usize << share.num_vars;
            if share.stack_offset < end
                || share.stack_offset + span > len
                || share.lo_bits > share.num_vars
                || (share.hi_at + (span >> share.lo_bits)) * 3 > eq.len()
                || (share.lo_at + (1usize << share.lo_bits)) * 3 > eq.len()
            {
                return Ok(None);
            }
            end = share.stack_offset + span;
        }

        let be = backend()?;
        let stream = be.next_stream();
        // Each share's data offset: where its column starts in `data`.
        let (data, staged, data_offsets): (Arc<CudaSlice<u64>>, bool, Vec<usize>) = match message {
            LeanMessage::Resident { store, parts } => {
                let mut offsets = Vec::with_capacity(shares.len());
                for share in shares {
                    let Some(&(column, _)) = parts
                        .iter()
                        .find(|(_, offset)| *offset == share.stack_offset)
                    else {
                        return Ok(None);
                    };
                    let (at, rows) = store.span(column);
                    if rows != 1usize << share.num_vars {
                        return Ok(None);
                    }
                    offsets.push(at);
                }
                (store.buffer(), false, offsets)
            }
            LeanMessage::Parts(parts) => {
                // Zeroed: what the parts do not cover is the stacking's padding.
                let mut base = crate::device::alloc_zeros_or_trim::<u64>(&stream, len)?;
                for (column, offset) in parts {
                    if offset + column.len() > len {
                        return Ok(None);
                    }
                    let mut at = base.slice_mut(*offset..*offset + column.len());
                    stream.memcpy_htod(*column, &mut at)?;
                }
                let offsets = shares.iter().map(|s| s.stack_offset).collect();
                (Arc::new(base), true, offsets)
            }
        };

        let mut rows = Vec::with_capacity(shares.len() * 9);
        let mut starts = Vec::with_capacity(shares.len());
        let mut ends = Vec::with_capacity(shares.len());
        for (share, data_at) in shares.iter().zip(&data_offsets) {
            rows.extend_from_slice(&[
                share.stack_offset as u64,
                *data_at as u64,
                share.lo_bits as u64,
                share.hi_at as u64,
                share.lo_at as u64,
                share.scale[0],
                share.scale[1],
                share.scale[2],
                0,
            ]);
            starts.push(share.stack_offset as u64);
            ends.push((share.stack_offset + (1usize << share.num_vars)) as u64);
        }
        let shares_dev = crate::device::htod_or_trim(&stream, &rows)?;
        let eq_dev = crate::device::htod_or_trim(&stream, eq)?;
        let starts_dev = crate::device::htod_or_trim(&stream, &starts)?;
        let ends_dev = crate::device::htod_or_trim(&stream, &ends)?;
        // SAFETY: the kernel writes every position.
        let mut colmap = unsafe { crate::device::alloc_or_trim::<u16>(&stream, len) }?;
        let count = shares.len() as u32;
        let len_arg = len as u64;
        let grid = (len as u64)
            .div_ceil(LEAN_BLOCK as u64)
            .min(LEAN_MAX_GRID as u64) as u32;
        unsafe {
            stream
                .launch_builder(&be.whir_lean_colmap)
                .arg(&mut colmap)
                .arg(&len_arg)
                .arg(&starts_dev)
                .arg(&ends_dev)
                .arg(&count)
                .launch(LaunchConfig {
                    grid_dim: (grid.max(1), 1, 1),
                    block_dim: (LEAN_BLOCK, 1, 1),
                    shared_mem_bytes: 0,
                })?;
        }
        let t_dev = crate::device::alloc_zeros_or_trim::<u64>(&stream, LEAN_MAX_T * 3)?;
        let partials = crate::device::alloc_zeros_or_trim::<u64>(
            &stream,
            LEAN_MAX_T * LEAN_MAX_GRID as usize * 3,
        )?;
        let sums = crate::device::alloc_zeros_or_trim::<u64>(&stream, LEAN_MAX_T * 3)?;
        Ok(Some(Self {
            stream,
            num_vars,
            colmap,
            shares: shares_dev,
            eq: eq_dev,
            data,
            staged,
            bound: Vec::new(),
            t_dev,
            t_host: Vec::new(),
            partials,
            sums,
        }))
    }

    /// Variables still free.
    pub fn num_vars(&self) -> usize {
        self.num_vars - self.bound.len() / 3
    }

    /// Variables bound so far.
    pub fn bound(&self) -> usize {
        self.bound.len() / 3
    }

    /// What a lean opening of `2^num_vars` holds with `shares` columns and
    /// `eq_values` extension values of `eq` halves: a `u16` per position, nine
    /// u64 per share, the halves, the round scratch — and the message staged
    /// in the base field when it is not resident.
    pub const fn planned_bytes(
        num_vars: usize,
        shares: usize,
        eq_values: usize,
        staged: bool,
    ) -> u64 {
        let staging = if staged { 8u64 << num_vars } else { 0 };
        (2u64 << num_vars)
            + 72 * shares as u64
            + 24 * eq_values as u64
            + LEAN_SCRATCH_BYTES
            + staging
    }

    /// Device bytes this opening holds: the share map, the shares, the `eq`
    /// halves, the round scratch, and the staged message when it staged one.
    pub fn device_bytes(&self) -> u64 {
        let staged = if self.staged { self.data.len() * 8 } else { 0 };
        (self.colmap.len() * 2
            + (self.shares.len() + self.eq.len() + self.partials.len() + self.sums.len()) * 8
            + self.t_dev.len() * 8
            + staged) as u64
    }

    /// `eq(bound challenges, ·)` over the `2^bound` prefixes, on the card —
    /// `[1]` before any is bound.
    fn prefix_weights(&self) -> Result<CudaSlice<u64>> {
        if self.bound.is_empty() {
            return crate::device::htod_or_trim(&self.stream, &[1u64, 0, 0]);
        }
        crate::sumcheck::eq_table_ext3(&self.stream, &self.bound, 1usize << self.bound())
    }

    /// The next round's evaluations at the nodes `t` (ext3, three u64 each),
    /// as a [`SumcheckSession::round`] over the materialised factors would
    /// return them.
    pub fn round(&mut self, t: &[u64]) -> Result<Vec<u64>> {
        assert!(t.len().is_multiple_of(3), "three u64 per ext3 node");
        let num_t = t.len() / 3;
        assert!(num_t > 0 && num_t <= LEAN_MAX_T, "nodes per round");
        assert!(self.num_vars() >= 1, "a round needs a variable to bind");
        let be = backend()?;
        let round = self.bound() + 1;
        let eqc = self.prefix_weights()?;
        if self.t_host != t {
            let mut head = self.t_dev.slice_mut(0..t.len());
            self.stream.memcpy_htod(t, &mut head)?;
            self.t_host.clear();
            self.t_host.extend_from_slice(t);
        }
        let rests = 1u64 << (self.num_vars - round);
        let grid = rests.div_ceil(LEAN_BLOCK as u64).min(LEAN_MAX_GRID as u64) as u32;
        let n_arg = self.num_vars as u32;
        let s_arg = round as u32;
        let num_t_arg = num_t as u32;
        unsafe {
            self.stream
                .launch_builder(&be.whir_lean_round)
                .arg(&self.colmap)
                .arg(&self.shares)
                .arg(&self.eq)
                .arg(self.data.as_ref())
                .arg(&n_arg)
                .arg(&s_arg)
                .arg(&eqc)
                .arg(&self.t_dev)
                .arg(&num_t_arg)
                .arg(&mut self.partials)
                .launch(LaunchConfig {
                    grid_dim: (grid.max(1), 1, 1),
                    block_dim: (LEAN_BLOCK, 1, 1),
                    shared_mem_bytes: LEAN_BLOCK * 3 * 8,
                })?;
            self.stream
                .launch_builder(&be.sum_partials_ext3)
                .arg(&self.partials)
                .arg(&(grid.max(1) as u64))
                .arg(&mut self.sums)
                .launch(LaunchConfig {
                    grid_dim: (num_t as u32, 1, 1),
                    block_dim: (LEAN_BLOCK, 1, 1),
                    shared_mem_bytes: LEAN_BLOCK * 3 * 8,
                })?;
        }
        let sums = self.stream.clone_dtoh(&self.sums.slice(0..num_t * 3))?;
        self.stream.synchronize()?;
        Ok(sums)
    }

    /// Binds the round's variable to `r` (ext3, three u64). Nothing moves on
    /// the card: the next round reads the shares with one more prefix bound.
    pub fn bind(&mut self, r: &[u64]) {
        assert_eq!(r.len(), 3, "an ext3 challenge");
        assert!(self.num_vars() >= 1, "a bind needs a variable");
        self.bound.extend_from_slice(r);
    }

    /// The factors at the challenges bound so far, written out at
    /// `2^(n − bound)`: the session the rest of the chain runs on.
    pub fn materialize(self) -> Result<OpeningSession> {
        let be = backend()?;
        let bound = self.bound();
        let len = 1usize << (self.num_vars - bound);
        let eqfull = self.prefix_weights()?;
        // SAFETY: the kernel writes every element of both.
        let mut weight = unsafe { crate::device::alloc_or_trim::<u64>(&self.stream, len * 3) }?;
        let mut message = unsafe { crate::device::alloc_or_trim::<u64>(&self.stream, len * 3) }?;
        let n_arg = self.num_vars as u32;
        let bound_arg = bound as u32;
        let grid = (len as u64)
            .div_ceil(LEAN_BLOCK as u64)
            .min(LEAN_MAX_GRID as u64) as u32;
        unsafe {
            self.stream
                .launch_builder(&be.whir_lean_materialize)
                .arg(&self.colmap)
                .arg(&self.shares)
                .arg(&self.eq)
                .arg(self.data.as_ref())
                .arg(&n_arg)
                .arg(&bound_arg)
                .arg(&eqfull)
                .arg(&mut weight)
                .arg(&mut message)
                .launch(LaunchConfig {
                    grid_dim: (grid.max(1), 1, 1),
                    block_dim: (LEAN_BLOCK, 1, 1),
                    shared_mem_bytes: 0,
                })?;
        }
        Ok(OpeningSession::from_tables(
            self.stream.clone(),
            weight,
            message,
            len,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// ★ The working set the lean opening exists to shrink, in numbers, at the
    /// production stack (25) and at two fatter ones (27, 28). The materialised
    /// session is `1.5 C` held plus `0.25 C` staging (`C` the committed base
    /// codeword, `2^(n+2)·8` bytes); the lean one over resident columns is
    /// `C/16` for the share map plus kilobytes — 400 columns and 30 points'
    /// `eq` halves at `2^11` values each is generous.
    #[test]
    fn the_lean_opening_holds_a_sixteenth_of_a_codeword() {
        for n in [25usize, 27, 28] {
            let codeword = 8u64 << (n + 2);
            let (held, staging) = OpeningSession::planned_bytes(n);
            let lean = LeanRound0::planned_bytes(n, 400, 30 * 2 * 2048, false);
            let lean_staged = LeanRound0::planned_bytes(n, 400, 30 * 2 * 2048, true);
            println!(
                "2^{n}: materialised {:.2} C + {:.2} C staging; lean {:.4} C resident, {:.4} C staged",
                held as f64 / codeword as f64,
                staging as f64 / codeword as f64,
                lean as f64 / codeword as f64,
                lean_staged as f64 / codeword as f64,
            );
            assert_eq!(held * 2, codeword * 3);
            assert!(lean * 15 < codeword, "2^{n}: {lean} B");
            assert!(
                lean * 20 < held + staging,
                "2^{n}: {lean} B against {} B",
                held + staging
            );
        }
    }
}
