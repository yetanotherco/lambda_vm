//! ⛔ MEASUREMENT (on no proving path): the values a first-round query's kept
//! subtree holds, evaluated from the stacked multilinear instead of read out of
//! its re-encoded codeword.
//!
//! Today a retired first-round codeword is rebuilt whole
//! ([`crate::whir::encode_codeword_resident`]: `2^(n+b)` values) only for its
//! openings to read the `2^kept` leaves under each query's kept tree node. A
//! leaf `j` is the `2^fold` values `word[j + t·2^(D−fold)]`, `D = n + b`. If
//! the tree put leaf `j` at position `bitrev(j)`, the leaves under one kept
//! node would be `j ≡ c (mod 2^(D−L))`, `L = fold + kept`, and their values the
//! codeword on ONE coset `ω^c·⟨ζ⟩`, `ζ = ω^(2^(D−L))`, `2^L` points.
//!
//! The codeword is `P(ω^j)` with `P(x) = f̂(x^(2^(n−1)), …, x^2, x)`: hypercube
//! bit `s` goes to `x^(2^(n−1−s))` (`multilinear::whir::lift_coefficients`).
//! On the coset `x^(2^m)` is the constant `ω^(c·2^m)` for every `m ≥ L`, so
//! the bits `s < n − L` fold away against constants,
//! `f'(rest) = f(rest, 0) + γ·(f(rest, 1) − f(rest, 0))`,
//! `γ_s(c) = ω^(c·2^(n−1−s))`, and what is left is a multilinear of `L`
//! variables: its Möbius transform, a twist by `ω^(c·i)` and one NTT of `2^L`
//! over `ζ` give the coset's values. Bit `s`'s constant depends on `c` modulo
//! `2^(b+1+s)` only, so the low bits fold once per residue class for every
//! coset in it (the tile kernel), the rest per coset.
//!
//! No Möbius transform of the whole polynomial and no `2^(n+b)` transform.
//! [`CosetPlan::evaluate_on_host`] is the same algorithm on the host, and the
//! tests pin both against `encode(lift_coefficients(f))`.
//!
//! The module is loaded on first use from its own cubin, never by
//! [`crate::device::Backend::init`].

use std::sync::{Arc, OnceLock};

use cudarc::driver::{
    CudaFunction, CudaModule, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::Ptx;
use math::field::goldilocks::GoldilocksField;
use math::field::traits::{IsFFTField, IsField};

use crate::Result;
use crate::device::{alloc_or_trim, backend, htod_or_trim};

const WHIR_COSET_CUBIN: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/whir_coset.cubin"));

/// Tile bits the tile kernel folds at most: a 256-value tile, its block.
const MAX_TILE_LEVELS: usize = 8;
/// Bits the rest kernel folds at most: one warp's lanes.
const MAX_REST_LEVELS: usize = 5;
/// Tiles a block of the tile kernel folds (`COSET_TILES_PER_BLOCK`).
const TILES_PER_BLOCK: usize = 8;
/// NTT stages and Möbius levels the chunk kernel takes in shared memory.
const MAX_CHUNK_LEVELS: usize = 10;
/// Dynamic shared memory the tile kernel may ask for: under the 48 KiB a
/// block takes without an attribute, with room for its static arrays.
const SHARED_LIMIT: usize = 47 * 1024;
const BLOCK: u32 = 256;

struct Module {
    _module: Arc<CudaModule>,
    fold_tiles: CudaFunction,
    fold_tiles_parts: CudaFunction,
    tile_map: CudaFunction,
    fold_rest: CudaFunction,
    mobius_level: CudaFunction,
    chunk: CudaFunction,
    ntt_stage: CudaFunction,
    leaf_order: CudaFunction,
}

fn module() -> Result<&'static Module> {
    static MODULE: OnceLock<Module> = OnceLock::new();
    if let Some(m) = MODULE.get() {
        return Ok(m);
    }
    let be = backend()?;
    let module = be
        .ctx
        .load_module(Ptx::from_binary(WHIR_COSET_CUBIN.to_vec()))?;
    let m = Module {
        fold_tiles: module.load_function("coset_fold_tiles")?,
        fold_tiles_parts: module.load_function("coset_fold_tiles_parts")?,
        tile_map: module.load_function("coset_tile_map")?,
        fold_rest: module.load_function("coset_fold_rest")?,
        mobius_level: module.load_function("coset_mobius_level")?,
        chunk: module.load_function("coset_chunk")?,
        ntt_stage: module.load_function("coset_ntt_stage")?,
        leaf_order: module.load_function("coset_leaf_order")?,
        _module: module,
    };
    Ok(MODULE.get_or_init(|| m))
}

fn fmul(a: u64, b: u64) -> u64 {
    GoldilocksField::mul(&a, &b)
}

fn fadd(a: u64, b: u64) -> u64 {
    GoldilocksField::add(&a, &b)
}

fn fsub(a: u64, b: u64) -> u64 {
    GoldilocksField::sub(&a, &b)
}

fn fpow(mut base: u64, mut exp: u64) -> u64 {
    let mut acc = 1u64;
    while exp > 0 {
        if exp & 1 == 1 {
            acc = fmul(acc, base);
        }
        base = fmul(base, base);
        exp >>= 1;
    }
    acc
}

fn fold(f0: u64, f1: u64, gamma: u64) -> u64 {
    fadd(f0, fmul(gamma, fsub(f1, f0)))
}

fn reverse(value: u64, bits: usize) -> u64 {
    if bits == 0 {
        0
    } else {
        value.reverse_bits() >> (64 - bits)
    }
}

/// Everything the kernels read besides the polynomial, for one set of queried
/// leaves: the cosets they fall in and the folding constants.
#[derive(Clone, Debug)]
pub struct CosetPlan {
    num_vars: usize,
    log_blowup: usize,
    log_coset: usize,
    kept: usize,
    /// The cosets, as `c = j mod 2^(D−L)` over the queried leaves, ascending
    /// and distinct. Output block `q` is coset `classes[q]`.
    classes: Vec<u64>,
    tile_levels: usize,
    rest_levels: usize,
    level_count: Vec<u32>,
    level_offset: Vec<u32>,
    parents: Vec<u32>,
    gammas: Vec<u64>,
    buf_len: usize,
    slab_of: Vec<u32>,
    rest_gammas: Vec<u64>,
    twists: Vec<u64>,
    twiddles: Vec<u64>,
}

impl CosetPlan {
    /// The plan for a codeword of a `num_vars`-variable multilinear at
    /// `log_blowup`, leaves of `2^log_fold` values, `kept` binary levels under
    /// a kept node, and the queried `leaves` (each `< 2^(D−log_fold)`).
    ///
    /// `None` for a shape the kernels do not take: no bit to fold
    /// (`num_vars ≤ L`), more than 13 bits to fold (8 tile + 5 rest), a
    /// domain past the field's two-adicity, a leaf out of range, or no leaf.
    pub fn new(
        num_vars: usize,
        log_blowup: usize,
        log_fold: usize,
        kept: usize,
        leaves: &[u64],
    ) -> Option<Self> {
        let log_coset = log_fold + kept;
        let log_domain = num_vars + log_blowup;
        if leaves.is_empty()
            || log_coset < 1
            || num_vars <= log_coset
            || log_domain > GoldilocksField::TWO_ADICITY as usize
            || leaves.iter().any(|&j| j >> (log_domain - log_fold) != 0)
        {
            return None;
        }
        let folded = num_vars - log_coset;
        let tile_levels = folded.min(MAX_TILE_LEVELS);
        let rest_levels = folded - tile_levels;
        if rest_levels > MAX_REST_LEVELS {
            return None;
        }
        let omega = *GoldilocksField::get_primitive_root_of_unity(log_domain as u64)
            .ok()?
            .value();
        // ω^(2^m), m = 0..D.
        let mut omega_pow2 = Vec::with_capacity(log_domain + 1);
        let mut w = omega;
        for _ in 0..=log_domain {
            omega_pow2.push(w);
            w = fmul(w, w);
        }
        // Bit `s`'s constant for the class of `c`: ω^(c·2^(n−1−s)).
        let gamma_of = |c: u64, s: usize| fpow(omega_pow2[num_vars - 1 - s], c);

        let class_mod = 1u64 << (log_domain - log_coset);
        let mut classes: Vec<u64> = leaves.iter().map(|&j| j & (class_mod - 1)).collect();
        classes.sort_unstable();
        classes.dedup();

        // Level t lists the residues c mod 2^(b+1+t), ascending.
        let mut keys: Vec<Vec<u64>> = Vec::with_capacity(tile_levels);
        let mut level_count = Vec::with_capacity(tile_levels);
        let mut level_offset = Vec::with_capacity(tile_levels);
        let mut parents = Vec::new();
        let mut gammas = Vec::new();
        for t in 0..tile_levels {
            let modulus = 1u64 << (log_blowup + 1 + t);
            let mut level: Vec<u64> = classes.iter().map(|&c| c & (modulus - 1)).collect();
            level.sort_unstable();
            level.dedup();
            level_offset.push(parents.len() as u32);
            level_count.push(level.len() as u32);
            for &key in &level {
                let parent = if t == 0 {
                    0
                } else {
                    keys[t - 1]
                        .binary_search(&(key & ((modulus >> 1) - 1)))
                        .ok()?
                };
                parents.push(parent as u32);
                gammas.push(gamma_of(key, t));
            }
            keys.push(level);
        }
        let buf_len = (0..tile_levels)
            .map(|t| (level_count[t] as usize) << (tile_levels - 1 - t))
            .chain(std::iter::once(1usize << tile_levels))
            .max()
            .unwrap_or(1);

        let last = keys.last()?;
        let last_mod = 1u64 << (log_blowup + tile_levels);
        let mut slab_of = Vec::with_capacity(classes.len());
        let mut rest_gammas = Vec::with_capacity(classes.len() * rest_levels);
        for &c in &classes {
            slab_of.push(last.binary_search(&(c & (last_mod - 1))).ok()? as u32);
            for l in 0..rest_levels {
                rest_gammas.push(gamma_of(c, tile_levels + l));
            }
        }
        let twists = classes.iter().map(|&c| fpow(omega, c)).collect();
        let zeta = omega_pow2[log_domain - log_coset];
        let mut twiddles = Vec::with_capacity(1 << (log_coset - 1));
        let mut z = 1u64;
        for _ in 0..1usize << (log_coset - 1) {
            twiddles.push(z);
            z = fmul(z, zeta);
        }

        let plan = Self {
            num_vars,
            log_blowup,
            log_coset,
            kept,
            classes,
            tile_levels,
            rest_levels,
            level_count,
            level_offset,
            parents,
            gammas,
            buf_len,
            slab_of,
            rest_gammas,
            twists,
            twiddles,
        };
        (plan.tile_shared_bytes() <= SHARED_LIMIT).then_some(plan)
    }

    /// The cosets, ascending; output block `q` is `classes()[q]`.
    pub fn classes(&self) -> &[u64] {
        &self.classes
    }

    /// Values a coset holds: `2^L`.
    pub fn coset_len(&self) -> usize {
        1 << self.log_coset
    }

    /// The output block of the coset `leaf` falls in.
    pub fn block_of(&self, leaf: u64) -> Option<usize> {
        let class_mod = 1u64 << self.classes_bits();
        self.classes.binary_search(&(leaf & (class_mod - 1))).ok()
    }

    /// Where `leaf` sits among the blocks: its coset's output block `q` and
    /// its position `r` under that coset's kept node (`leaf =
    /// classes()[q] + bitrev_kept(r)·2^(D−L)`).
    pub fn slot_of(&self, leaf: u64) -> Option<(usize, usize)> {
        let q = self.block_of(leaf)?;
        let r = reverse(leaf >> self.classes_bits(), self.kept) as usize;
        Some((q, r))
    }

    /// The tree position, among the kept nodes, of output block `q`'s node:
    /// the coset's class with its `D − L` bits reversed.
    pub fn block_position(&self, q: usize) -> usize {
        reverse(self.classes[q], self.classes_bits()) as usize
    }

    /// Binary tree levels under a kept node: a block is `2^kept` leaves.
    pub fn kept(&self) -> usize {
        self.kept
    }

    /// `D − L`: there are `2^(D−L)` cosets of `2^L` points in the domain.
    fn classes_bits(&self) -> usize {
        self.num_vars + self.log_blowup - self.log_coset
    }

    /// The leaves of output block `q` in its order: entry `r` is the leaf at
    /// position `r` under the kept node, `c + bitrev_kept(r)·2^(D−L)`.
    pub fn block_leaves(&self, q: usize) -> Vec<u64> {
        let stride = 1u64 << self.classes_bits();
        (0..1u64 << self.kept)
            .map(|r| self.classes[q] + reverse(r, self.kept) * stride)
            .collect()
    }

    /// Tiles of `2^tile_levels` values the polynomial splits into.
    fn tiles(&self) -> usize {
        1 << (self.num_vars - self.tile_levels)
    }

    /// Classes the last tile level lists: the slab's rows.
    fn slab_rows(&self) -> usize {
        *self.level_count.last().unwrap_or(&0) as usize
    }

    /// Möbius levels and NTT stages the chunk kernel takes.
    fn chunk_levels(&self) -> usize {
        self.log_coset.min(MAX_CHUNK_LEVELS)
    }

    /// Bytes the tile kernel's dynamic shared memory takes: the constants,
    /// two level buffers, the block's last-level values, the parents.
    fn tile_shared_bytes(&self) -> usize {
        let total = self.gammas.len();
        8 * (total + 2 * self.buf_len + self.slab_rows() * TILES_PER_BLOCK) + 4 * total
    }

    /// Bytes the device evaluation allocates for a polynomial: the assembled
    /// stack, the classes' slabs, the cosets twice (transform and leaf order).
    pub fn device_bytes(&self) -> usize {
        8 * ((1 << self.num_vars)
            + self.slab_rows() * self.tiles()
            + 2 * self.classes.len() * self.coset_len())
    }

    /// The cosets' values from the multilinear's `2^n` hypercube values, on the
    /// host, by the kernels' algorithm: block `q` is `2^L` values in leaf order
    /// (leaf `r`'s `2^fold` values at `r·2^fold`).
    pub fn evaluate_on_host(&self, values: &[u64]) -> Vec<u64> {
        assert_eq!(
            values.len(),
            1 << self.num_vars,
            "one value a hypercube point"
        );
        let tile = 1usize << self.tile_levels;
        let tiles = self.tiles();
        let mut slab = vec![0u64; self.slab_rows() * tiles];
        for at in 0..tiles {
            let mut rows = vec![values[at * tile..(at + 1) * tile].to_vec()];
            for t in 0..self.tile_levels {
                let off = self.level_offset[t] as usize;
                rows = (0..self.level_count[t] as usize)
                    .map(|k| {
                        let row = &rows[self.parents[off + k] as usize];
                        row.chunks_exact(2)
                            .map(|p| fold(p[0], p[1], self.gammas[off + k]))
                            .collect()
                    })
                    .collect();
            }
            for (k, row) in rows.iter().enumerate() {
                slab[k * tiles + at] = row[0];
            }
        }

        let n = self.coset_len();
        let fold_bits = self.log_coset - self.kept;
        let mut out = Vec::with_capacity(self.classes.len() * n);
        for q in 0..self.classes.len() {
            let row = &slab[self.slab_of[q] as usize * tiles..][..tiles];
            let gammas = &self.rest_gammas[q * self.rest_levels..][..self.rest_levels];
            let mut g: Vec<u64> = row
                .chunks_exact(1 << self.rest_levels)
                .map(|chunk| {
                    let mut v = chunk.to_vec();
                    for &gamma in gammas {
                        v = v.chunks_exact(2).map(|p| fold(p[0], p[1], gamma)).collect();
                    }
                    v[0]
                })
                .collect();
            for level in 0..self.log_coset {
                for i in 0..n {
                    if i >> level & 1 == 1 {
                        g[i] = fsub(g[i], g[i ^ (1 << level)]);
                    }
                }
            }
            for (h, v) in g.iter_mut().enumerate() {
                *v = fmul(*v, fpow(self.twists[q], reverse(h as u64, self.log_coset)));
            }
            for s in 1..=self.log_coset {
                let half = 1usize << (s - 1);
                for block in (0..n).step_by(2 * half) {
                    for j in 0..half {
                        let w = self.twiddles[j << (self.log_coset - s)];
                        let u = g[block + j];
                        let v = fmul(g[block + j + half], w);
                        g[block + j] = fadd(u, v);
                        g[block + j + half] = fsub(u, v);
                    }
                }
            }
            let mut block = vec![0u64; n];
            for (u, &v) in g.iter().enumerate() {
                let r = reverse(u as u64 & ((1 << self.kept) - 1), self.kept) as usize;
                block[(r << fold_bits) | (u >> self.kept)] = v;
            }
            out.extend_from_slice(&block);
        }
        out
    }
}

/// The cosets' values on the card: block `q` (`2^L` values, leaf order) for
/// coset `plan.classes()[q]`, queued on `stream`.
pub struct CosetValues {
    buffer: CudaSlice<u64>,
    stream: Arc<CudaStream>,
}

impl CosetValues {
    /// Waits for the evaluation queued on this buffer's stream.
    pub fn synchronize(&self) -> Result<()> {
        self.stream.synchronize()
    }

    /// Every block, read back.
    pub fn to_host(&self) -> Result<Vec<u64>> {
        let values = crate::readback::clone_dtoh(&self.stream, &self.buffer)?;
        self.stream.synchronize()?;
        Ok(values)
    }
}

/// Wall-clock laps of an evaluation's phases, read only when asked for.
struct Laps<'a> {
    split: Option<&'a mut Vec<(&'static str, f64)>>,
    clock: std::time::Instant,
}

impl Laps<'_> {
    /// Synchronizes and records the phase just queued, when laps are kept.
    fn lap(&mut self, name: &'static str, stream: &Arc<CudaStream>) -> Result<()> {
        if let Some(split) = self.split.as_deref_mut() {
            stream.synchronize()?;
            split.push((name, self.clock.elapsed().as_secs_f64() * 1e3));
            self.clock = std::time::Instant::now();
        }
        Ok(())
    }
}

/// The plan's cosets over a stacked polynomial whose columns the card holds:
/// `(column, offset)` into `store`, zeros between, as
/// [`crate::whir::encode_codeword_resident`] assembles it.
///
/// `split`, when given, synchronizes after each phase and records its wall
/// milliseconds (assemble, tiles, rest, transform); the timed path passes
/// `None` and queues everything.
pub fn evaluate_resident(
    store: &crate::columns::DeviceColumns,
    parts: &[(usize, usize)],
    plan: &CosetPlan,
    split: Option<&mut Vec<(&'static str, f64)>>,
) -> Result<CosetValues> {
    let be = backend()?;
    let m = module()?;
    let stream = be.next_stream();
    let n = 1usize << plan.num_vars;
    let mut laps = Laps {
        split,
        clock: std::time::Instant::now(),
    };
    for &(column, offset) in parts {
        ensure!(column < store.num_columns() && offset + store.span(column).1 <= n);
    }
    // SAFETY: the memset writes every element.
    let mut f = unsafe { alloc_or_trim::<u64>(&stream, n) }?;
    stream.memset_zeros(&mut f)?;
    for &(column, offset) in parts {
        store.copy_into(column, &mut f, offset, &stream)?;
    }
    laps.lap("assemble", &stream)?;
    let len = plan.classes.len() * plan.coset_len();
    let out = evaluate_from(&stream, m, Tiles::Assembled(f), plan, &mut laps, 0, len)?;
    Ok(CosetValues {
        buffer: out,
        stream,
    })
}

/// [`evaluate_resident`] over a [`StackSlices`]: the stack read where it lies
/// when `in_place` and its parts allow ([`StackSlices::parts_in_place`]; the
/// kinds of its tiles returned), assembled into a fresh buffer otherwise
/// (`None`). The same values either way, word for word.
pub fn evaluate_slices(
    stack: &StackSlices,
    plan: &CosetPlan,
    in_place: bool,
    split: Option<&mut Vec<(&'static str, f64)>>,
) -> Result<(CosetValues, Option<TileKinds>)> {
    ensure!(plan.num_vars == stack.num_vars && plan.log_blowup == stack.log_blowup);
    let be = backend()?;
    let m = module()?;
    let stream = be.next_stream();
    let mut laps = Laps {
        split,
        clock: std::time::Instant::now(),
    };
    let parts = if in_place {
        stack.parts_in_place()
    } else {
        None
    };
    let (tiles, kinds) = match &parts {
        Some(parts) => (
            Tiles::InPlace {
                data: stack.buffer.as_ref(),
                parts,
            },
            Some(parts.tile_kinds(plan.num_vars, plan.tile_levels)),
        ),
        None => (Tiles::Assembled(stack.assemble(&stream)?), None),
    };
    laps.lap("assemble", &stream)?;
    let len = plan.classes.len() * plan.coset_len();
    let out = evaluate_from(&stream, m, tiles, plan, &mut laps, 0, len)?;
    Ok((
        CosetValues {
            buffer: out,
            stream,
        },
        kinds,
    ))
}

/// What the tile kernel reads: the assembled stack (spent by it), or the stack
/// where it lies, by its parts.
enum Tiles<'a> {
    Assembled(CudaSlice<u64>),
    InPlace {
        data: &'a CudaSlice<u64>,
        parts: &'a StackParts,
    },
}

/// A stack's tiles by kind ([`StackParts::tile_kinds`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TileKinds {
    /// Tiles one part holds all of: read straight from the store.
    pub full: u64,
    /// Tiles no part touches: folded to zero without being read.
    pub zero: u64,
    /// Tiles of shorter columns, a column's end or zeros: read value by value.
    pub mixed: u64,
}

/// A stack's parts as the in-place kernels read them: sorted by their offset
/// in the stack, disjoint, the empty ones dropped; part `p` is `len[p]` values
/// at `off[p]` in the stack and at `at[p]` in the store's buffer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StackParts {
    off: Vec<u64>,
    len: Vec<u64>,
    at: Vec<u64>,
}

impl StackParts {
    /// `parts` — `(element offset in the buffer, elements, offset in the
    /// stack)` — of a `2^num_vars`-value stack in a buffer of `buffer_len`
    /// elements. `None` for parts the kernels cannot place: overlapping, past
    /// the stack or past the buffer.
    pub fn new(
        parts: &[(usize, usize, usize)],
        num_vars: usize,
        buffer_len: usize,
    ) -> Option<Self> {
        let n = 1u64 << num_vars;
        let mut sorted: Vec<(u64, u64, u64)> = parts
            .iter()
            .filter(|&&(_, len, _)| len > 0)
            .map(|&(at, len, off)| (off as u64, len as u64, at as u64))
            .collect();
        sorted.sort_unstable();
        let mut end = 0u64;
        for &(off, len, at) in &sorted {
            if off < end || off.checked_add(len)? > n || at.checked_add(len)? > buffer_len as u64 {
                return None;
            }
            end = off + len;
        }
        // The map keeps two words for its kinds.
        (sorted.len() < u32::MAX as usize - 1).then(|| Self {
            off: sorted.iter().map(|p| p.0).collect(),
            len: sorted.iter().map(|p| p.1).collect(),
            at: sorted.iter().map(|p| p.2).collect(),
        })
    }

    /// How many of the stack's `2^tile_bits`-value tiles each kind takes, from
    /// the parts alone (no tile is visited): the tiles whole inside a part, the
    /// tiles no part touches, and the rest.
    pub fn tile_kinds(&self, num_vars: usize, tile_bits: usize) -> TileKinds {
        let tile = 1u64 << tile_bits;
        let tiles = 1u64 << (num_vars - tile_bits);
        let (mut full, mut touched, mut reached) = (0u64, 0u64, 0u64);
        for (&off, &len) in self.off.iter().zip(&self.len) {
            let end = off + len;
            full += (end / tile).saturating_sub(off.div_ceil(tile));
            // Sorted and disjoint: the touched spans, merged as they come.
            let (from, to) = ((off / tile).max(reached), end.div_ceil(tile));
            if to > from {
                touched += to - from;
                reached = to;
            }
        }
        TileKinds {
            full,
            zero: tiles - touched,
            mixed: touched - full,
        }
    }
}

/// The cosets from the stack `tiles` reads (an assembled one is spent), into a
/// buffer of `out_len` values laid out by `stride` (`coset_leaf_order`), on
/// `stream`.
fn evaluate_from(
    stream: &Arc<CudaStream>,
    m: &Module,
    tiles_from: Tiles<'_>,
    plan: &CosetPlan,
    laps: &mut Laps<'_>,
    stride: u64,
    out_len: usize,
) -> Result<CudaSlice<u64>> {
    let tiles = plan.tiles();
    let last = plan.slab_rows();
    let count = htod_or_trim(stream, &plan.level_count)?;
    let offset = htod_or_trim(stream, &plan.level_offset)?;
    let parents = htod_or_trim(stream, &plan.parents)?;
    let gammas = htod_or_trim(stream, &plan.gammas)?;
    // SAFETY: the tile kernel writes every (class, tile) entry.
    let mut slab = unsafe { alloc_or_trim::<u64>(stream, last * tiles) }?;
    let tiles_u64 = tiles as u64;
    let levels_u32 = plan.tile_levels as u32;
    let total_u32 = plan.gammas.len() as u32;
    let buf_u32 = plan.buf_len as u32;
    // The long kernels' grids are cut to the calling thread's cap, if any
    // (`device::grid_cap`); each strides over its work, the same values.
    let capped = crate::device::capped_grid;
    let tile_grid = LaunchConfig {
        grid_dim: (capped(tiles.div_ceil(TILES_PER_BLOCK) as u32), 1, 1),
        block_dim: (BLOCK, 1, 1),
        shared_mem_bytes: plan.tile_shared_bytes() as u32,
    };
    match tiles_from {
        Tiles::Assembled(f) => {
            unsafe {
                stream
                    .launch_builder(&m.fold_tiles)
                    .arg(&f)
                    .arg(&tiles_u64)
                    .arg(&levels_u32)
                    .arg(&count)
                    .arg(&offset)
                    .arg(&parents)
                    .arg(&gammas)
                    .arg(&total_u32)
                    .arg(&buf_u32)
                    .arg(&mut slab)
                    .launch(tile_grid)?;
            }
            // Spent; the free is stream-ordered.
            drop(f);
        }
        Tiles::InPlace { data, parts } => {
            // A stack with no part (all zeros) still uploads one word a
            // table: a zero-length upload is not an allocation.
            let upload = |v: &[u64]| {
                if v.is_empty() {
                    htod_or_trim(stream, &[0u64])
                } else {
                    htod_or_trim(stream, v)
                }
            };
            let (off, len, at) = (upload(&parts.off)?, upload(&parts.len)?, upload(&parts.at)?);
            let nparts = parts.off.len() as u32;
            let tile_bits = plan.tile_levels as u32;
            // SAFETY: the map kernel writes every tile's entry.
            let mut map = unsafe { alloc_or_trim::<u32>(stream, tiles) }?;
            unsafe {
                stream
                    .launch_builder(&m.tile_map)
                    .arg(&off)
                    .arg(&len)
                    .arg(&nparts)
                    .arg(&tile_bits)
                    .arg(&tiles_u64)
                    .arg(&mut map)
                    .launch(LaunchConfig {
                        grid_dim: (capped(tiles.div_ceil(BLOCK as usize) as u32), 1, 1),
                        block_dim: (BLOCK, 1, 1),
                        shared_mem_bytes: 0,
                    })?;
                stream
                    .launch_builder(&m.fold_tiles_parts)
                    .arg(data)
                    .arg(&map)
                    .arg(&off)
                    .arg(&len)
                    .arg(&at)
                    .arg(&nparts)
                    .arg(&tiles_u64)
                    .arg(&levels_u32)
                    .arg(&count)
                    .arg(&offset)
                    .arg(&parents)
                    .arg(&gammas)
                    .arg(&total_u32)
                    .arg(&buf_u32)
                    .arg(&mut slab)
                    .launch(tile_grid)?;
            }
        }
    }
    laps.lap("tiles", stream)?;

    let cosets = plan.classes.len();
    let len = plan.coset_len();
    let log_len = plan.log_coset as u32;
    let slab_of = htod_or_trim(stream, &plan.slab_of)?;
    // A zero-length upload is not an allocation the driver takes; the kernel
    // reads no constant when there is no bit left.
    let rest_gammas = if plan.rest_gammas.is_empty() {
        htod_or_trim(stream, &[0u64])?
    } else {
        htod_or_trim(stream, &plan.rest_gammas)?
    };
    // SAFETY: the rest kernel writes every coset value.
    let mut g = unsafe { alloc_or_trim::<u64>(stream, cosets * len) }?;
    let rest_u32 = plan.rest_levels as u32;
    let cosets_u32 = cosets as u32;
    let warps = (cosets * len) as u64;
    unsafe {
        stream
            .launch_builder(&m.fold_rest)
            .arg(&slab)
            .arg(&tiles_u64)
            .arg(&rest_u32)
            .arg(&cosets_u32)
            .arg(&slab_of)
            .arg(&rest_gammas)
            .arg(&log_len)
            .arg(&mut g)
            .launch(LaunchConfig {
                grid_dim: (capped((warps * 32).div_ceil(BLOCK as u64) as u32), 1, 1),
                block_dim: (BLOCK, 1, 1),
                shared_mem_bytes: 0,
            })?;
    }
    drop(slab);
    laps.lap("rest", stream)?;

    let twists = htod_or_trim(stream, &plan.twists)?;
    let twiddles = htod_or_trim(stream, &plan.twiddles)?;
    let chunk_levels = plan.chunk_levels();
    let pairs = (cosets * len / 2) as u64;
    let pair_grid = LaunchConfig {
        grid_dim: (pairs.div_ceil(BLOCK as u64) as u32, 1, 1),
        block_dim: (BLOCK, 1, 1),
        shared_mem_bytes: 0,
    };
    // The Möbius levels commute, so the ones past the chunk go first and the
    // chunk kernel ends the transform where the NTT starts.
    for level in chunk_levels..plan.log_coset {
        let level_u32 = level as u32;
        unsafe {
            stream
                .launch_builder(&m.mobius_level)
                .arg(&mut g)
                .arg(&pairs)
                .arg(&log_len)
                .arg(&level_u32)
                .launch(pair_grid)?;
        }
    }
    let chunk_u32 = chunk_levels as u32;
    unsafe {
        stream
            .launch_builder(&m.chunk)
            .arg(&mut g)
            .arg(&log_len)
            .arg(&chunk_u32)
            .arg(&twists)
            .arg(&twiddles)
            .launch(LaunchConfig {
                grid_dim: ((cosets << (plan.log_coset - chunk_levels)) as u32, 1, 1),
                block_dim: (1 << (chunk_levels - 1), 1, 1),
                shared_mem_bytes: (8usize << chunk_levels) as u32,
            })?;
    }
    for stage in chunk_levels + 1..=plan.log_coset {
        let stage_u32 = stage as u32;
        unsafe {
            stream
                .launch_builder(&m.ntt_stage)
                .arg(&mut g)
                .arg(&pairs)
                .arg(&log_len)
                .arg(&stage_u32)
                .arg(&twiddles)
                .launch(pair_grid)?;
        }
    }
    // SAFETY: the permutation writes every value of a coset; past them (a
    // strided layout padded to a power of two) the memset below does.
    let mut out = unsafe { alloc_or_trim::<u64>(stream, out_len) }?;
    if out_len > cosets * len {
        stream.memset_zeros(&mut out)?;
    }
    let total = (cosets * len) as u64;
    let kept_u32 = plan.kept as u32;
    unsafe {
        stream
            .launch_builder(&m.leaf_order)
            .arg(&g)
            .arg(&total)
            .arg(&log_len)
            .arg(&kept_u32)
            .arg(&stride)
            .arg(&mut out)
            .launch(LaunchConfig {
                grid_dim: (total.div_ceil(BLOCK as u64) as u32, 1, 1),
                block_dim: (BLOCK, 1, 1),
                shared_mem_bytes: 0,
            })?;
    }
    drop(g);
    laps.lap("transform", stream)?;
    Ok(out)
}

/// A stacked polynomial the card holds as slices of one buffer — `(element
/// offset in the buffer, elements, offset in the stack)`, zeros between —
/// with its own handle on the buffer, so a revived commitment can evaluate its
/// cosets after the borrow of the store it was revived from has ended.
#[derive(Clone)]
pub struct StackSlices {
    buffer: Arc<CudaSlice<u64>>,
    parts: Vec<(usize, usize, usize)>,
    num_vars: usize,
    log_blowup: usize,
    /// What a successor encoded from this stack's bound message lives inside,
    /// as a fold lives inside its codeword's room.
    room: Arc<crate::device::DeviceReservation>,
}

impl StackSlices {
    /// The stack `parts` — `(column, offset)` into `store` — make, at
    /// `2^num_vars` values and a code of `2^log_blowup` times that.
    pub fn from_resident(
        store: &crate::columns::DeviceColumns,
        parts: &[(usize, usize)],
        num_vars: usize,
        log_blowup: usize,
    ) -> Result<Self> {
        let n = 1usize << num_vars;
        let mut held = Vec::with_capacity(parts.len());
        for &(column, offset) in parts {
            ensure!(column < store.num_columns());
            let (at, len) = store.span(column);
            ensure!(offset + len <= n);
            held.push((at, len, offset));
        }
        let room = crate::device::reserve(0).ok_or_else(crate::invalid_value)?;
        Ok(Self {
            buffer: store.buffer(),
            parts: held,
            num_vars,
            log_blowup,
            room: Arc::new(room),
        })
    }

    /// Values the codeword would have: `2^(n + b)`.
    pub fn elements(&self) -> usize {
        1 << (self.num_vars + self.log_blowup)
    }

    pub fn num_vars(&self) -> usize {
        self.num_vars
    }

    pub fn log_blowup(&self) -> usize {
        self.log_blowup
    }

    pub fn room(&self) -> &Arc<crate::device::DeviceReservation> {
        &self.room
    }

    /// The whole codeword, as the re-encode makes it: what a fold, or a plan
    /// the kernels do not take, falls back to.
    pub fn encode(&self, transient: bool) -> Result<crate::whir::DeviceCodeword> {
        crate::whir::encode_codeword_slices(
            &self.buffer,
            &self.parts,
            self.num_vars,
            self.log_blowup,
            transient,
        )
    }

    /// The parts as the in-place kernels read them, or `None` when they cannot
    /// place them ([`StackParts::new`]): the caller assembles the stack.
    pub fn parts_in_place(&self) -> Option<StackParts> {
        StackParts::new(&self.parts, self.num_vars, self.buffer.len())
    }

    fn assemble(&self, stream: &Arc<CudaStream>) -> Result<CudaSlice<u64>> {
        // SAFETY: `copy_slices` writes every element (zeros, then the parts).
        let mut f = unsafe { alloc_or_trim::<u64>(stream, 1 << self.num_vars) }?;
        crate::whir::copy_slices(stream, &self.buffer, &self.parts, &mut f)?;
        Ok(f)
    }
}

/// What a retired round-0 tree in the coset order needs for its openings,
/// evaluated from the stack: one tree over the plan's blocks (block `q` = the
/// gathered leaves `[q·2^kept, (q+1)·2^kept)`, padded to `leaves`), in the
/// hash's node layout as `DeviceCodeword::block_subtrees` returns it, and the
/// queried blocks' values.
pub struct CosetOpenings {
    pub nodes: Vec<u8>,
    pub leaves: usize,
    /// `2^fold` values a query, in the order the queries were given.
    pub values: Vec<u64>,
    /// The stack's tiles by kind when it was read where it lies; `None` when
    /// it was assembled (not asked for, or parts the kernels cannot place).
    pub in_place: Option<TileKinds>,
}

/// The kept blocks' subtrees and the queried blocks' values of a round-0
/// tree in the coset order, from the stack: the cosets evaluated straight into
/// the strided layout the coset leaf kernel reads, hashed and built by the
/// commit's own kernels (so each block's subtree is the committed one, which
/// the caller checks against its kept node), and the queried blocks gathered
/// from the same buffer. `queried` are leaf indices, each in one of the plan's
/// cosets. Poseidon1 trees only. `in_place`: the stack read where it lies
/// ([`evaluate_slices`]); the same words either way.
pub fn coset_openings(
    stack: &StackSlices,
    plan: &CosetPlan,
    queried: &[u64],
    hash: crate::DeviceHash,
    in_place: bool,
) -> Result<CosetOpenings> {
    ensure!(hash == crate::DeviceHash::Poseidon1);
    ensure!(plan.num_vars == stack.num_vars && plan.log_blowup == stack.log_blowup);
    let be = backend()?;
    let m = module()?;
    let stream = be.next_stream();
    let block = 1usize << (plan.log_coset - plan.kept);
    let leaves = (plan.classes.len() << plan.kept).next_power_of_two().max(2);
    let mut at = Vec::with_capacity(queried.len());
    for &leaf in queried {
        let (q, r) = plan.slot_of(leaf).ok_or_else(crate::invalid_value)?;
        at.push(((q << plan.kept) | r) as u64);
    }

    let parts = if in_place {
        stack.parts_in_place()
    } else {
        None
    };
    let (tiles, kinds) = match &parts {
        Some(parts) => (
            Tiles::InPlace {
                data: stack.buffer.as_ref(),
                parts,
            },
            Some(parts.tile_kinds(plan.num_vars, plan.tile_levels)),
        ),
        None => (Tiles::Assembled(stack.assemble(&stream)?), None),
    };
    let mut laps = Laps {
        split: None,
        clock: std::time::Instant::now(),
    };
    let out = evaluate_from(
        &stream,
        m,
        tiles,
        plan,
        &mut laps,
        leaves as u64,
        leaves * block,
    )?;

    let total_nodes = crate::tree_nodes(hash, leaves);
    // SAFETY: every byte is written before it is read — the leaves by the
    // kernel below, the inner nodes by the level loop after it.
    let mut nodes = unsafe { alloc_or_trim::<u8>(&stream, total_nodes * 32) }?;
    let leaves_offset = crate::leaves_offset(hash, leaves) * 32;
    let leaves_u64 = leaves as u64;
    let block_u64 = block as u64;
    {
        let mut region = nodes.slice_mut(leaves_offset..leaves_offset + leaves * 32);
        let k = crate::p1_stark::kernels()?;
        unsafe {
            stream
                .launch_builder(&k.leaves_base_coset)
                .arg(&out)
                .arg(&leaves_u64)
                .arg(&block_u64)
                .arg(&mut region)
                .launch(crate::p1_stark::launch_cfg(leaves_u64))?;
        }
    }
    crate::merkle::build_inner_tree_levels(stream.as_ref(), be, &mut nodes, leaves, hash)?;
    let nodes_host = crate::readback::clone_dtoh(&stream, &nodes)?;

    let index_dev = htod_or_trim(&stream, &at)?;
    // SAFETY: the gather writes every value it is sized for.
    let mut values = unsafe { alloc_or_trim::<u64>(&stream, queried.len().max(1) * block) }?;
    if !queried.is_empty() {
        let queries = queried.len() as u64;
        let limbs = 1u64;
        unsafe {
            stream
                .launch_builder(&be.gather_cosets)
                .arg(&out)
                .arg(&index_dev)
                .arg(&queries)
                .arg(&leaves_u64)
                .arg(&block_u64)
                .arg(&limbs)
                .arg(&mut values)
                .launch(LaunchConfig::for_num_elems((queried.len() * block) as u32))?;
        }
    }
    let mut values_host = crate::readback::clone_dtoh(&stream, &values)?;
    stream.synchronize()?;
    values_host.truncate(queried.len() * block);
    Ok(CosetOpenings {
        nodes: nodes_host,
        leaves,
        values: values_host,
        in_place: kinds,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use math::field::element::FieldElement;
    use multilinear::mle::Mle;
    use multilinear::whir::{Domain, encode, lift_coefficients};

    type FE = FieldElement<GoldilocksField>;
    const P: u64 = 0xFFFF_FFFF_0000_0001;

    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
    }

    fn canonical(v: u64) -> u64 {
        if v >= P { v - P } else { v }
    }

    /// The codeword the commit encodes: `encode(lift_coefficients(f))` over
    /// the `2^(n+b)` subgroup.
    fn codeword(values: &[u64], num_vars: usize, log_blowup: usize) -> Vec<u64> {
        let mle = Mle::new(values.iter().map(|&v| FE::from(v)).collect()).expect("mle");
        let domain = Domain::<GoldilocksField>::new(num_vars + log_blowup).expect("domain");
        encode::<GoldilocksField, GoldilocksField>(&lift_coefficients(&mle), &domain)
            .expect("encode")
            .iter()
            .map(|v| canonical(*v.value()))
            .collect()
    }

    /// Every block equals the codeword at its leaves, leaf by leaf and value
    /// by value: block `q`'s entry `r·2^fold + t` is `word[leaf_r + t·2^(D−fold)]`.
    fn the_host_plan_is_the_codeword(
        num_vars: usize,
        log_blowup: usize,
        log_fold: usize,
        kept: usize,
        queries: usize,
        seed: u64,
    ) {
        let mut rng = Rng(seed);
        let values: Vec<u64> = (0..1usize << num_vars).map(|_| rng.next() % P).collect();
        let num_leaves = 1u64 << (num_vars + log_blowup - log_fold);
        let leaves: Vec<u64> = (0..queries).map(|_| rng.next() % num_leaves).collect();
        let plan = CosetPlan::new(num_vars, log_blowup, log_fold, kept, &leaves).expect("plan");
        let word = codeword(&values, num_vars, log_blowup);
        let got = plan.evaluate_on_host(&values);
        let n = plan.coset_len();
        assert_eq!(got.len(), plan.classes().len() * n);
        let emulated = emulate::kernels(&plan, &values);
        assert_eq!(
            emulated.iter().map(|&v| canonical(v)).collect::<Vec<_>>(),
            got.iter().map(|&v| canonical(v)).collect::<Vec<_>>(),
            "the kernels' emulation and the host algorithm"
        );
        // The long kernels cut to 1, 2 or 7 blocks give the full grid's words.
        for cap in [1, 2, 7] {
            assert_eq!(
                emulate::kernels_capped(&plan, &values, Some(cap)),
                emulated,
                "{cap} blocks a kernel"
            );
        }
        for (q, block) in got.chunks_exact(n).enumerate() {
            for (r, leaf) in plan.block_leaves(q).into_iter().enumerate() {
                for t in 0..1usize << log_fold {
                    let want = word[leaf as usize + t * num_leaves as usize];
                    assert_eq!(
                        canonical(block[(r << log_fold) + t]),
                        want,
                        "n {num_vars} b {log_blowup} fold {log_fold} kept {kept}: block {q} leaf {leaf} value {t}"
                    );
                }
            }
        }
        for &leaf in &leaves {
            let q = plan.block_of(leaf).expect("every query has a block");
            assert!(
                plan.block_leaves(q).contains(&leaf),
                "a query's block holds its leaf"
            );
        }
    }

    /// `kernels/whir_coset.cu` transcribed: each kernel's blocks, threads and
    /// warps run in turn (a phase between two barriers at a time), over the
    /// shared-memory layout and the launch parameters [`evaluate_resident`]
    /// passes. What the laptop can say about the device code without a card:
    /// an index, a layout or a launch-size slip shows here as a wrong value
    /// or an out-of-bounds panic.
    // A transcription keeps the kernels' thread loops, indices and all.
    #[allow(clippy::needless_range_loop)]
    mod emulate {
        use super::super::*;

        pub fn kernels(plan: &CosetPlan, values: &[u64]) -> Vec<u64> {
            kernels_capped(plan, values, None)
        }

        /// [`kernels`] with the long kernels' grids cut to `cap` blocks
        /// (`device::capped_grid`), each striding over its work.
        pub fn kernels_capped(plan: &CosetPlan, values: &[u64], cap: Option<usize>) -> Vec<u64> {
            let len = plan.classes.len() * plan.coset_len();
            kernels_into(plan, values, 0, len, cap)
        }

        /// The blocks a capped launch takes: `full`, or `cap` (at least one).
        fn grid_of(full: usize, cap: Option<usize>) -> usize {
            cap.map_or(full, |c| full.min(c.max(1)))
        }

        /// [`kernels`] into the strided layout `coset_openings` hashes:
        /// `leaves` gathered leaves (padded to a power of two), value `t` of
        /// leaf `i` at `i + t·leaves`.
        pub fn kernels_strided(plan: &CosetPlan, values: &[u64]) -> (Vec<u64>, usize) {
            let leaves = (plan.classes.len() << plan.kept).next_power_of_two().max(2);
            let block = 1usize << (plan.log_coset - plan.kept);
            (
                kernels_into(plan, values, leaves as u64, leaves * block, None),
                leaves,
            )
        }

        fn kernels_into(
            plan: &CosetPlan,
            values: &[u64],
            stride: u64,
            out_len: usize,
            cap: Option<usize>,
        ) -> Vec<u64> {
            let tiles = plan.tiles();
            let mut slab = vec![0u64; plan.slab_rows() * tiles];
            fold_tiles(plan, values, &mut slab, cap);
            finish(plan, &slab, stride, out_len, cap)
        }

        /// [`kernels`] over the stack read where it lies: `data` the store's
        /// buffer, `parts` its parts. Returns the cosets and the tile map.
        pub fn kernels_in_place(
            plan: &CosetPlan,
            parts: &StackParts,
            data: &[u64],
            cap: Option<usize>,
        ) -> (Vec<u64>, Vec<u32>) {
            let tiles = plan.tiles();
            let map = tile_map(parts, plan.tile_levels, tiles, cap);
            let mut slab = vec![0u64; plan.slab_rows() * tiles];
            fold_tiles_parts(plan, data, &map, parts, &mut slab, cap);
            let len = plan.classes.len() * plan.coset_len();
            (finish(plan, &slab, 0, len, cap), map)
        }

        pub const ZERO: u32 = 0xffff_ffff;
        pub const MIXED: u32 = 0xffff_fffe;

        fn parts_upto(off: &[u64], e: u64) -> usize {
            let (mut lo, mut hi) = (0usize, off.len());
            while lo < hi {
                let mid = (lo + hi) >> 1;
                if off[mid] <= e {
                    lo = mid + 1;
                } else {
                    hi = mid;
                }
            }
            lo
        }

        /// `coset_tile_map`, one thread a tile, the grid striding over them.
        /// An entry no thread writes stays `UNWRITTEN`.
        pub fn tile_map(
            parts: &StackParts,
            tile_bits: usize,
            num_tiles: usize,
            cap: Option<usize>,
        ) -> Vec<u32> {
            const UNWRITTEN: u32 = 0xdead_beef;
            let threads = grid_of(num_tiles.div_ceil(BLOCK as usize), cap) * BLOCK as usize;
            let mut map = vec![UNWRITTEN; num_tiles];
            for thread in 0..threads {
                for tile in (thread..num_tiles).step_by(threads) {
                    let tile = tile as u64;
                    let start = tile << tile_bits;
                    let end = start + (1 << tile_bits);
                    let upto = parts_upto(&parts.off, start);
                    let mut kind = ZERO;
                    if upto > 0 {
                        let p = upto - 1;
                        let past = parts.off[p] + parts.len[p];
                        if past >= end {
                            kind = p as u32;
                        } else if past > start {
                            kind = MIXED;
                        }
                    }
                    if kind == ZERO && upto < parts.off.len() && parts.off[upto] < end {
                        kind = MIXED;
                    }
                    map[tile as usize] = kind;
                }
            }
            map
        }

        /// `coset_fold_tiles_parts`, a block's tiles in turn as in
        /// [`fold_tiles`].
        fn fold_tiles_parts(
            plan: &CosetPlan,
            data: &[u64],
            map: &[u32],
            parts: &StackParts,
            slab: &mut [u64],
            cap: Option<usize>,
        ) {
            const TILES: usize = TILES_PER_BLOCK;
            let levels = plan.tile_levels;
            let num_tiles = plan.tiles();
            let buf_len = plan.buf_len;
            let last_count = plan.level_count[levels - 1] as usize;
            let grid = grid_of(num_tiles.div_ceil(TILES), cap);
            for block in 0..grid {
                // A block's shared memory, kept across its passes.
                let mut bufs = [vec![0u64; buf_len], vec![0u64; buf_len]];
                let mut fin = vec![0u64; last_count * TILES];
                let tile = 1usize << levels;
                for first in (block * TILES..num_tiles).step_by(grid * TILES) {
                    for i in 0..TILES {
                        let at = first + i;
                        if at >= num_tiles {
                            break;
                        }
                        let kind = map[at];
                        if kind == ZERO {
                            for k in 0..last_count {
                                fin[k * TILES + i] = 0;
                            }
                            continue;
                        }
                        let base = (at as u64) << levels;
                        if kind == MIXED {
                            for x in 0..tile {
                                let e = base + x as u64;
                                let upto = parts_upto(&parts.off, e);
                                let mut v = 0;
                                if upto > 0 {
                                    let p = upto - 1;
                                    if e < parts.off[p] + parts.len[p] {
                                        v = data[(parts.at[p] + (e - parts.off[p])) as usize];
                                    }
                                }
                                bufs[0][x] = v;
                            }
                        } else {
                            let p = kind as usize;
                            let src = (parts.at[p] + (base - parts.off[p])) as usize;
                            bufs[0][..tile].copy_from_slice(&data[src..src + tile]);
                        }
                        let (mut inp, mut out) = (0usize, 1usize);
                        let mut width = tile;
                        for t in 0..levels {
                            let cnt = plan.level_count[t] as usize;
                            let off = plan.level_offset[t] as usize;
                            let log_half = levels - 1 - t;
                            let half = 1usize << log_half;
                            let outputs = cnt << log_half;
                            let last = t + 1 == levels;
                            let mut wrote = vec![0u64; buf_len];
                            for o in 0..outputs {
                                let k = o >> log_half;
                                let j = o & (half - 1);
                                let row = plan.parents[off + k] as usize * width + 2 * j;
                                let r =
                                    fold(bufs[inp][row], bufs[inp][row + 1], plan.gammas[off + k]);
                                if last {
                                    fin[k * TILES + i] = r;
                                } else {
                                    wrote[o] = r;
                                }
                            }
                            if !last {
                                bufs[out][..outputs].copy_from_slice(&wrote[..outputs]);
                            }
                            std::mem::swap(&mut inp, &mut out);
                            width = half;
                        }
                    }
                    for x in 0..last_count * TILES {
                        let k = x / TILES;
                        let at = first + x % TILES;
                        if at < num_tiles {
                            slab[k * num_tiles + at] = fin[x];
                        }
                    }
                }
            }
        }

        fn finish(
            plan: &CosetPlan,
            slab: &[u64],
            stride: u64,
            out_len: usize,
            cap: Option<usize>,
        ) -> Vec<u64> {
            let cosets = plan.classes.len();
            let len = plan.coset_len();
            let mut g = vec![0u64; cosets * len];
            fold_rest(plan, slab, &mut g, cap);
            let cl = plan.chunk_levels();
            let pairs = cosets * len / 2;
            for level in cl..plan.log_coset {
                mobius_level(&mut g, pairs, plan.log_coset, level);
            }
            chunk(plan, &mut g, cl);
            for stage in cl + 1..=plan.log_coset {
                ntt_stage(&mut g, pairs, plan.log_coset, stage, &plan.twiddles);
            }
            let mut out = vec![0u64; out_len];
            leaf_order(&g, plan.log_coset, plan.kept, stride, &mut out);
            out
        }

        fn fold_tiles(plan: &CosetPlan, f: &[u64], slab: &mut [u64], cap: Option<usize>) {
            const TILES: usize = TILES_PER_BLOCK;
            let levels = plan.tile_levels;
            let num_tiles = plan.tiles();
            let total = plan.gammas.len();
            let buf_len = plan.buf_len;
            let last_count = plan.level_count[levels - 1] as usize;
            // The kernel's dynamic layout, in u64 words, and the bytes asked for.
            let fin_at = total + 2 * buf_len;
            let parent_at_bytes = 8 * (fin_at + last_count * TILES);
            assert_eq!(parent_at_bytes + 4 * total, plan.tile_shared_bytes());
            let grid = grid_of(num_tiles.div_ceil(TILES), cap);
            for block in 0..grid {
                // A block's shared memory, kept across its passes.
                let mut bufs = [vec![0u64; buf_len], vec![0u64; buf_len]];
                let mut fin = vec![0u64; last_count * TILES];
                let tile = 1usize << levels;
                for first in (block * TILES..num_tiles).step_by(grid * TILES) {
                    for i in 0..TILES {
                        let at = first + i;
                        if at >= num_tiles {
                            break;
                        }
                        bufs[0][..tile].copy_from_slice(&f[at * tile..(at + 1) * tile]);
                        let (mut inp, mut out) = (0usize, 1usize);
                        let mut width = tile;
                        for t in 0..levels {
                            let cnt = plan.level_count[t] as usize;
                            let off = plan.level_offset[t] as usize;
                            let log_half = levels - 1 - t;
                            let half = 1usize << log_half;
                            let outputs = cnt << log_half;
                            let last = t + 1 == levels;
                            let mut wrote = vec![0u64; buf_len];
                            for o in 0..outputs {
                                let k = o >> log_half;
                                let j = o & (half - 1);
                                let row = plan.parents[off + k] as usize * width + 2 * j;
                                let r =
                                    fold(bufs[inp][row], bufs[inp][row + 1], plan.gammas[off + k]);
                                if last {
                                    fin[k * TILES + i] = r;
                                } else {
                                    wrote[o] = r;
                                }
                            }
                            if !last {
                                bufs[out][..outputs].copy_from_slice(&wrote[..outputs]);
                            }
                            std::mem::swap(&mut inp, &mut out);
                            width = half;
                        }
                    }
                    for x in 0..last_count * TILES {
                        let k = x / TILES;
                        let at = first + x % TILES;
                        if at < num_tiles {
                            slab[k * num_tiles + at] = fin[x];
                        }
                    }
                }
            }
        }

        fn fold_rest(plan: &CosetPlan, slab: &[u64], g: &mut [u64], cap: Option<usize>) {
            let rest = plan.rest_levels;
            let num_tiles = plan.tiles();
            let log_free = plan.log_coset;
            let warps = plan.classes.len() << log_free;
            let threads = grid_of((warps * 32).div_ceil(BLOCK as usize), cap) * BLOCK as usize;
            let launched = threads / 32;
            for first in 0..launched {
                for warp in (first..warps).step_by(launched) {
                    let q = warp >> log_free;
                    let h = warp & ((1 << log_free) - 1);
                    let src = plan.slab_of[q] as usize * num_tiles + (h << rest);
                    let mut v: Vec<u64> = (0..32)
                        .map(|lane| {
                            if lane < 1 << rest {
                                slab[src + lane]
                            } else {
                                0
                            }
                        })
                        .collect();
                    for l in 0..rest {
                        let snap = v.clone();
                        for lane in 0..32 {
                            let partner = if lane + (1 << l) < 32 {
                                snap[lane + (1 << l)]
                            } else {
                                snap[lane]
                            };
                            if lane & ((2 << l) - 1) == 0 {
                                v[lane] = fold(snap[lane], partner, plan.rest_gammas[q * rest + l]);
                            }
                        }
                    }
                    g[warp] = v[0];
                }
            }
        }

        fn mobius_level(g: &mut [u64], pairs: usize, log_size: usize, level: usize) {
            for idx in 0..pairs {
                let q = idx >> (log_size - 1);
                let r = idx & ((1 << (log_size - 1)) - 1);
                let lo = ((r >> level) << (level + 1)) | (r & ((1 << level) - 1));
                let base = q << log_size;
                g[base + lo + (1 << level)] = fsub(g[base + lo + (1 << level)], g[base + lo]);
            }
        }

        fn chunk(plan: &CosetPlan, g: &mut [u64], cl: usize) {
            let log_size = plan.log_coset;
            let half = 1usize << (cl - 1);
            let chunks = 1usize << (log_size - cl);
            for block in 0..plan.classes.len() << (log_size - cl) {
                let q = block / chunks;
                let at = (block % chunks) << cl;
                let base = (q << log_size) + at;
                let mut x = g[base..base + 2 * half].to_vec();
                for l in 0..cl {
                    for tid in 0..half {
                        let lo = ((tid >> l) << (l + 1)) | (tid & ((1 << l) - 1));
                        x[lo + (1 << l)] = fsub(x[lo + (1 << l)], x[lo]);
                    }
                }
                let w = plan.twists[q];
                for tid in 0..half {
                    for e in [tid, tid + half] {
                        let i = reverse((at + e) as u64, log_size);
                        x[e] = fmul(x[e], fpow(w, i));
                    }
                }
                for s in 1..=cl {
                    let h = 1usize << (s - 1);
                    for tid in 0..half {
                        let i0 = ((tid >> (s - 1)) << s) | (tid & (h - 1));
                        let tw = plan.twiddles[(tid & (h - 1)) << (log_size - s)];
                        let u = x[i0];
                        let v = fmul(x[i0 + h], tw);
                        x[i0] = fadd(u, v);
                        x[i0 + h] = fsub(u, v);
                    }
                }
                g[base..base + 2 * half].copy_from_slice(&x);
            }
        }

        fn ntt_stage(g: &mut [u64], pairs: usize, log_size: usize, s: usize, twiddles: &[u64]) {
            for idx in 0..pairs {
                let q = idx >> (log_size - 1);
                let r = idx & ((1 << (log_size - 1)) - 1);
                let h = 1usize << (s - 1);
                let j = r & (h - 1);
                let i0 = ((r >> (s - 1)) << s) | j;
                let base = q << log_size;
                let u = g[base + i0];
                let v = fmul(g[base + i0 + h], twiddles[j << (log_size - s)]);
                g[base + i0] = fadd(u, v);
                g[base + i0 + h] = fsub(u, v);
            }
        }

        fn leaf_order(g: &[u64], log_size: usize, kept: usize, stride: u64, out: &mut [u64]) {
            for (idx, &v) in g.iter().enumerate() {
                let q = idx >> log_size;
                let u = idx & ((1 << log_size) - 1);
                let low = u & ((1 << kept) - 1);
                let r = if kept == 0 {
                    0
                } else {
                    reverse(low as u64, kept) as usize
                };
                let t = u >> kept;
                if stride == 0 {
                    out[(q << log_size) | (r << (log_size - kept)) | t] = v;
                } else {
                    out[((q << kept) | r) + t * stride as usize] = v;
                }
            }
        }
    }

    /// The strided layout the openings hash: value `t` of gathered leaf
    /// `q·2^kept + r` is the codeword at the leaf `slot_of` names, the
    /// padding is zero, and each block's tree position is its class reversed.
    #[test]
    fn the_strided_layout_holds_each_slot_s_leaf() {
        let (num_vars, log_blowup, log_fold, kept) = (14, 2, 2, 3);
        let mut rng = Rng(21);
        let values: Vec<u64> = (0..1usize << num_vars).map(|_| rng.next() % P).collect();
        let num_leaves = 1u64 << (num_vars + log_blowup - log_fold);
        let leaves: Vec<u64> = (0..13).map(|_| rng.next() % num_leaves).collect();
        let plan = CosetPlan::new(num_vars, log_blowup, log_fold, kept, &leaves).expect("plan");
        let word = codeword(&values, num_vars, log_blowup);
        let (strided, gathered) = emulate::kernels_strided(&plan, &values);
        let block = 1usize << log_fold;
        let real = plan.classes().len() << kept;
        assert_eq!(gathered, real.next_power_of_two());
        let depth = (num_vars + log_blowup - log_fold) as u32;
        for q in 0..plan.classes().len() {
            for (r, leaf) in plan.block_leaves(q).into_iter().enumerate() {
                assert_eq!(plan.slot_of(leaf), Some((q, r)), "leaf {leaf}");
                for t in 0..block {
                    assert_eq!(
                        canonical(strided[((q << kept) | r) + t * gathered]),
                        word[leaf as usize + t * num_leaves as usize]
                    );
                }
                // Block q's node sits at the position its class reversed
                // names: the leaf's tree position has those top bits.
                let position = leaf.reverse_bits() >> (64 - depth);
                assert_eq!(position as usize, (plan.block_position(q) << kept) | r);
            }
        }
        for i in real..gathered {
            for t in 0..block {
                assert_eq!(strided[i + t * gathered], 0, "padding");
            }
        }
    }

    /// Tile levels only, the rest kernel copying.
    #[test]
    fn cosets_equal_the_codeword_with_tile_levels_only() {
        the_host_plan_is_the_codeword(10, 2, 3, 4, 9, 1);
        the_host_plan_is_the_codeword(12, 2, 2, 3, 20, 2);
    }

    /// Tile levels and rest levels both, at blowups 2, 4 and 8.
    #[test]
    fn cosets_equal_the_codeword_with_rest_levels() {
        the_host_plan_is_the_codeword(14, 2, 2, 2, 17, 3);
        the_host_plan_is_the_codeword(15, 1, 3, 1, 11, 4);
        the_host_plan_is_the_codeword(16, 3, 2, 1, 33, 5);
    }

    /// The production geometry's coset (first fold 6, 256 leaves a kept node:
    /// 2^14 points) over the smallest stack that still folds bits.
    #[test]
    fn cosets_equal_the_codeword_at_the_production_coset() {
        the_host_plan_is_the_codeword(16, 2, 6, 8, 12, 6);
    }

    /// The production plan's shape: 13 bits fold (8 tile, 5 rest), the
    /// classes and the device footprint, without a polynomial.
    #[test]
    fn the_production_plan_folds_eight_tile_and_five_rest_bits() {
        let mut rng = Rng(7);
        let leaves: Vec<u64> = (0..114).map(|_| rng.next() % (1 << 23)).collect();
        let plan = CosetPlan::new(27, 2, 6, 8, &leaves).expect("plan");
        assert_eq!((plan.tile_levels, plan.rest_levels), (8, 5));
        assert_eq!(plan.coset_len(), 1 << 14);
        assert!(plan.classes().len() <= 114);
        // Level t lists at most the 2^(3+t) residues, and at most the queries.
        for (t, &count) in plan.level_count.iter().enumerate() {
            assert!(
                count as usize <= (8usize << t).min(114),
                "level {t}: {count}"
            );
        }
        assert!(plan.tile_shared_bytes() <= SHARED_LIMIT);
        assert!(plan.buf_len <= 1024);
        for &leaf in &leaves {
            let q = plan.block_of(leaf).expect("block");
            assert!(plan.block_leaves(q).contains(&leaf));
        }
    }

    /// A stack's parts as a store holds them: columns of the given heights
    /// placed as `StackedLayout::build` does (each at a multiple of its
    /// height, the cursor only growing), `short` values cut off every third
    /// column's end (a column ending mid-tile), then laid in the buffer in
    /// another order with junk between (a read past a part shows). Returns
    /// `(buffer offset, elements, stack offset)` parts and the buffer.
    fn store_layout(
        num_vars: usize,
        heights: &[usize],
        short: usize,
        seed: u64,
    ) -> (Vec<(usize, usize, usize)>, Vec<u64>) {
        let mut rng = Rng(seed);
        let mut placed = Vec::new();
        let mut cursor = 0usize;
        for (i, &h) in heights.iter().enumerate() {
            let len = 1usize << h;
            let off = cursor.next_multiple_of(len);
            if off + len > 1 << num_vars {
                break;
            }
            cursor = off + len;
            let len = if i % 3 == 2 {
                len - short.min(len - 1)
            } else {
                len
            };
            placed.push((len, off));
        }
        let mut order: Vec<usize> = (0..placed.len()).collect();
        for i in (1..order.len()).rev() {
            order.swap(i, rng.next() as usize % (i + 1));
        }
        let mut data = Vec::new();
        let mut parts = Vec::new();
        for &i in &order {
            let (len, off) = placed[i];
            for _ in 0..rng.next() % 5 {
                data.push(rng.next() % P);
            }
            parts.push((data.len(), len, off));
            data.extend((0..len).map(|_| rng.next() % P));
        }
        data.push(rng.next() % P);
        (parts, data)
    }

    /// The stack `parts` make out of `data`, zeros between.
    fn assembled(num_vars: usize, parts: &[(usize, usize, usize)], data: &[u64]) -> Vec<u64> {
        let mut f = vec![0u64; 1 << num_vars];
        for &(at, len, off) in parts {
            f[off..off + len].copy_from_slice(&data[at..at + len]);
        }
        f
    }

    /// Each tile's kind by looking at every value of it.
    fn kinds_by_hand(num_vars: usize, tile_bits: usize, parts: &StackParts) -> Vec<u32> {
        let owner = |e: u64| {
            (0..parts.off.len()).find(|&p| parts.off[p] <= e && e < parts.off[p] + parts.len[p])
        };
        (0..1u64 << (num_vars - tile_bits))
            .map(|tile| {
                let owners: Vec<Option<usize>> = (tile << tile_bits..(tile + 1) << tile_bits)
                    .map(owner)
                    .collect();
                if owners.iter().all(Option::is_none) {
                    emulate::ZERO
                } else if owners.iter().all(|o| *o == owners[0]) {
                    owners[0].unwrap() as u32
                } else {
                    emulate::MIXED
                }
            })
            .collect()
    }

    /// The stack read where it lies gives the assembled stack's words, raw,
    /// and the tile map gives each tile the kind its values have, the counts
    /// `tile_kinds` says without visiting a tile. Both paths give the same
    /// words, and the same map, with their grids cut to 1, 2 or 7 blocks.
    fn in_place_is_the_assembled_stack(
        plan: &CosetPlan,
        parts: &[(usize, usize, usize)],
        data: &[u64],
        what: &str,
    ) -> TileKinds {
        let n = plan.num_vars;
        let placed = StackParts::new(parts, n, data.len()).expect("the parts place");
        let (got, map) = emulate::kernels_in_place(plan, &placed, data, None);
        let whole = assembled(n, parts, data);
        let reference = emulate::kernels(plan, &whole);
        assert_eq!(
            got, reference,
            "{what}: in place and assembled, word for word"
        );
        for cap in [1, 2, 7] {
            let (capped, capped_map) = emulate::kernels_in_place(plan, &placed, data, Some(cap));
            assert_eq!(capped, got, "{what}: in place, {cap} blocks a kernel");
            assert_eq!(capped_map, map, "{what}: the map at {cap} blocks");
            assert_eq!(
                emulate::kernels_capped(plan, &whole, Some(cap)),
                reference,
                "{what}: assembled, {cap} blocks a kernel"
            );
        }
        assert_eq!(
            map,
            kinds_by_hand(n, plan.tile_levels, &placed),
            "{what}: the map"
        );
        let kinds = placed.tile_kinds(n, plan.tile_levels);
        let count = |want: &dyn Fn(u32) -> bool| map.iter().filter(|&&k| want(k)).count() as u64;
        assert_eq!(
            kinds,
            TileKinds {
                full: count(&|k| k < emulate::MIXED),
                zero: count(&|k| k == emulate::ZERO),
                mixed: count(&|k| k == emulate::MIXED),
            },
            "{what}: the kinds counted from the parts"
        );
        kinds
    }

    /// The production tile (8 bits), with and without rest bits, at blowups 4
    /// and 2, and a 5-bit tile, over the layouts the stacks have: columns down
    /// to a value, columns ending mid-tile, alignment gaps, a zero tail; one
    /// column; no column.
    #[test]
    fn the_stack_read_in_place_is_the_assembled_stack() {
        let mut rng = Rng(31);
        for (num_vars, log_blowup, log_fold, kept) in [(12, 2, 2, 2), (16, 1, 2, 3), (11, 2, 3, 3)]
        {
            let num_leaves = 1u64 << (num_vars + log_blowup - log_fold);
            let leaves: Vec<u64> = (0..19).map(|_| rng.next() % num_leaves).collect();
            let plan = CosetPlan::new(num_vars, log_blowup, log_fold, kept, &leaves).expect("plan");
            let tb = plan.tile_levels;
            assert!(tb >= 3, "{num_vars}: tile bits {tb}");
            // Production-like: tall columns first, then shorter ones down to a
            // value, the cursor stopping ≈ 15 % short of the stack's end.
            let mut heights = vec![num_vars - 2, num_vars - 3];
            heights.extend((0..40).map(|i| (num_vars - 4).saturating_sub(i % (num_vars - 3))));
            let (mut parts, data) = store_layout(num_vars, &heights, 7, rng.next());
            let cap = (1usize << num_vars) * 85 / 100;
            parts.retain(|&(_, _, off)| off < cap);
            let kinds = in_place_is_the_assembled_stack(&plan, &parts, &data, "production-like");
            assert!(
                kinds.full > 0 && kinds.zero > 0 && kinds.mixed > 0,
                "{kinds:?}"
            );
            // Shortened columns at unaligned ends, sub-tile columns only.
            let small: Vec<usize> = (0..300).map(|i| [0, 1, 2, 5, 3][i % 5]).collect();
            let (parts, data) = store_layout(num_vars, &small, 1, rng.next());
            in_place_is_the_assembled_stack(&plan, &parts, &data, "sub-tile columns");
            // One column, the whole stack.
            let (parts, data) = store_layout(num_vars, &[num_vars], 0, rng.next());
            let kinds = in_place_is_the_assembled_stack(&plan, &parts, &data, "one column");
            assert_eq!((kinds.zero, kinds.mixed), (0, 0));
            // Columns meeting inside a tile and at a tile's edge, and one a
            // value short of its last tile's end, a tile of junk after it.
            let t = 1usize << tb;
            let parts = [
                (0, t + 3, 0),
                (t + 3, t - 3, t + 3),
                (2 * t, t, 2 * t),
                (3 * t, 2 * t - 1, 4 * t),
            ];
            let data: Vec<u64> = (0..6 * t).map(|_| rng.next() % P).collect();
            in_place_is_the_assembled_stack(&plan, &parts, &data, "columns meeting");
            // No column: every tile zero, every coset value zero.
            let kinds = in_place_is_the_assembled_stack(&plan, &[], &[], "no column");
            assert_eq!((kinds.full, kinds.mixed), (0, 0));
        }
    }

    /// The parts the in-place kernels cannot place are refused (the caller
    /// assembles the stack); empty ones are dropped and the rest sorted.
    #[test]
    fn parts_the_in_place_kernels_cannot_place_are_refused() {
        let n = 10;
        assert!(
            StackParts::new(&[(0, 8, 0), (8, 8, 4)], n, 64).is_none(),
            "overlapping"
        );
        assert!(
            StackParts::new(&[(0, 8, 1020)], n, 64).is_none(),
            "past the stack"
        );
        assert!(
            StackParts::new(&[(60, 8, 0)], n, 64).is_none(),
            "past the buffer"
        );
        assert!(
            StackParts::new(&[(0, 8, 0), (8, 8, 8)], n, 16).is_some(),
            "adjacent"
        );
        let placed =
            StackParts::new(&[(40, 8, 512), (0, 0, 4), (8, 4, 16)], n, 64).expect("placed");
        assert_eq!(
            (placed.off, placed.len, placed.at),
            (vec![16, 512], vec![4, 8], vec![8, 40])
        );
    }

    #[test]
    fn shapes_the_kernels_do_not_take_are_refused() {
        assert!(
            CosetPlan::new(14, 2, 6, 8, &[0]).is_none(),
            "no bit to fold"
        );
        assert!(
            CosetPlan::new(28, 2, 6, 8, &[0]).is_none(),
            "14 bits to fold"
        );
        assert!(CosetPlan::new(20, 2, 6, 8, &[]).is_none(), "no query");
        assert!(
            CosetPlan::new(20, 2, 6, 8, &[1 << 16]).is_none(),
            "leaf out of range"
        );
        assert!(CosetPlan::new(20, 2, 6, 8, &[(1 << 16) - 1]).is_some());
    }
}
