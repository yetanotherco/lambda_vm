//! BRANCH ops as the builder keeps them until their tables are built: a stream
//! of zigzag LEB128 deltas instead of 32 bytes an op (S3c-BRANCH).
//!
//! An op is a flags byte (`jalr`, and how its register is coded) and three
//! varints: the pc from the previous op's, the offset itself, and the register
//! the cheapest of three ways (as it is, from the previous op's register, or
//! from the op's own pc: a JALR's return address sits near it). Every op takes
//! at most 31 bytes, so any op, whatever its values, is kept losslessly and
//! none is held raw. Every [`INDEX_EVERY`] ops a sparse index records where
//! the stream stands, so any range expands on its own; the stream is stored in
//! blocks of at most [`super::blocks::BLOCK_BYTES`] that are never
//! reallocated once full, and an op never straddles two blocks (as
//! [`super::CompactLt`]). The ops expand exactly as pushed, in order, so the
//! tables are the same.

use super::{BranchOperation, Expand, Part, Segmented, blocks};

/// Where the stream stands at op `k * INDEX_EVERY`: its block, the byte in
/// it, and the previous op's pc and register.
#[derive(Clone, Copy)]
struct Mark {
    block: u32,
    at: u32,
    pc: u64,
    register: u64,
}

#[derive(Default)]
pub(super) struct CompactBranch {
    bytes: Vec<Vec<u8>>,
    index: Vec<Mark>,
    len: usize,
    last_pc: u64,
    last_register: u64,
    /// Bytes a block of the stream holds; 0 is [`blocks::BLOCK_BYTES`] (tests
    /// set a small one, to cross blocks).
    block_bytes: usize,
}

const INDEX_EVERY: usize = 4096;

/// The most bytes one op takes: the flags byte and three varints of at most
/// ten.
const OP_BYTES: usize = 31;

/// `flags`: the op is a JALR.
const JALR: u8 = 1;
/// `flags` bits 1–2: how the register is coded.
const REGISTER_SHIFT: u8 = 1;
const REGISTER_ABSOLUTE: u8 = 0;
const REGISTER_FROM_LAST: u8 = 1;
const REGISTER_FROM_PC: u8 = 2;

fn zigzag(d: u64) -> u64 {
    (d << 1) ^ ((d as i64 >> 63) as u64)
}

fn unzigzag(z: u64) -> u64 {
    (z >> 1) ^ (z & 1).wrapping_neg()
}

/// The bytes LEB128 takes for `v`.
fn varint_len(v: u64) -> usize {
    (64 - (v | 1).leading_zeros() as usize).div_ceil(7)
}

fn put(out: &mut Vec<u8>, mut v: u64) {
    while v >= 0x80 {
        out.push(v as u8 | 0x80);
        v >>= 7;
    }
    out.push(v as u8);
}

fn get(bytes: &[u8], at: &mut usize) -> u64 {
    let (mut v, mut shift) = (0u64, 0u32);
    loop {
        let b = bytes[*at];
        *at += 1;
        v |= u64::from(b & 0x7f) << shift;
        if b < 0x80 {
            return v;
        }
        shift += 7;
    }
}

impl CompactBranch {
    /// A list whose stream is cut into blocks of `bytes` (at least an op's).
    #[cfg(test)]
    pub(super) fn with_block_bytes(bytes: usize) -> Self {
        Self {
            block_bytes: bytes.max(OP_BYTES),
            ..Self::default()
        }
    }

    /// `ops`, in order.
    pub(super) fn from_ops(ops: &[BranchOperation]) -> Self {
        let mut list = Self::default();
        list.extend(ops);
        list
    }

    fn block_bytes(&self) -> usize {
        if self.block_bytes == 0 {
            blocks::BLOCK_BYTES
        } else {
            self.block_bytes
        }
    }

    pub(super) fn len(&self) -> usize {
        self.len
    }

    /// Append `ops`, in order.
    pub(super) fn extend(&mut self, ops: &[BranchOperation]) {
        let block_bytes = self.block_bytes();
        for op in ops {
            // Room for the op in the last block: grown as a `Vec` grows up to
            // a block's bytes, else a new block, allocated whole.
            let room = self
                .bytes
                .last()
                .is_some_and(|last| last.len() + OP_BYTES <= block_bytes);
            if !room {
                let capacity = if self.bytes.is_empty() {
                    4096.min(block_bytes)
                } else {
                    block_bytes
                };
                self.bytes.push(Vec::with_capacity(capacity));
            }
            let block = self.bytes.len() - 1;
            let Some(last) = self.bytes.last_mut() else {
                continue;
            };
            if last.capacity() - last.len() < OP_BYTES {
                let want = (last.capacity() * 2)
                    .min(block_bytes)
                    .max(last.len() + OP_BYTES);
                last.reserve_exact(want - last.len());
            }
            if self.len.is_multiple_of(INDEX_EVERY) {
                self.index.push(Mark {
                    block: block as u32,
                    at: last.len() as u32,
                    pc: self.last_pc,
                    register: self.last_register,
                });
            }
            // The register, the cheapest of its three codings.
            let codings = [
                (REGISTER_ABSOLUTE, zigzag(op.register)),
                (
                    REGISTER_FROM_LAST,
                    zigzag(op.register.wrapping_sub(self.last_register)),
                ),
                (REGISTER_FROM_PC, zigzag(op.register.wrapping_sub(op.pc))),
            ];
            let (mode, register) = codings
                .into_iter()
                .min_by_key(|&(_, v)| varint_len(v))
                .unwrap_or((REGISTER_ABSOLUTE, zigzag(op.register)));
            last.push((u8::from(op.jalr) * JALR) | (mode << REGISTER_SHIFT));
            put(last, zigzag(op.pc.wrapping_sub(self.last_pc)));
            put(last, zigzag(op.offset));
            put(last, register);
            self.last_pc = op.pc;
            self.last_register = op.register;
            self.len += 1;
        }
    }

    /// The bytes it takes on the heap (capacities).
    pub(super) fn heap_bytes(&self) -> usize {
        self.bytes.iter().map(Vec::capacity).sum::<usize>()
            + self.bytes.capacity() * std::mem::size_of::<Vec<u8>>()
            + self.index.capacity() * std::mem::size_of::<Mark>()
    }

    /// The bytes its largest single allocation takes.
    pub(super) fn largest_bytes(&self) -> usize {
        let blocks = self.bytes.iter().map(Vec::capacity).max().unwrap_or(0);
        blocks.max(self.index.capacity() * std::mem::size_of::<Mark>())
    }

    /// The list as one compact segment.
    pub(super) fn segments(&self) -> Segmented<'_, BranchOperation> {
        Segmented {
            parts: vec![Part::Compact(self)],
        }
    }

    /// Ops `start..end`, expanded.
    pub(super) fn range(&self, start: usize, end: usize) -> Vec<BranchOperation> {
        let mut out = Vec::new();
        self.expand_into(start, end, &mut out);
        out
    }
}

impl Expand<BranchOperation> for CompactBranch {
    fn len(&self) -> usize {
        self.len
    }

    fn expand_into(&self, start: usize, end: usize, out: &mut Vec<BranchOperation>) {
        let end = end.min(self.len);
        if start >= end {
            return;
        }
        let first = start / INDEX_EVERY;
        let mark = self.index[first];
        let (mut block, mut at) = (mark.block as usize, mark.at as usize);
        let (mut pc, mut register) = (mark.pc, mark.register);
        out.reserve(end - start);
        for k in first * INDEX_EVERY..end {
            // An op lies in one block: the next op starts the next block once
            // this one ends.
            if at >= self.bytes[block].len() {
                block += 1;
                at = 0;
            }
            let bytes = &self.bytes[block];
            let flags = bytes[at];
            at += 1;
            pc = pc.wrapping_add(unzigzag(get(bytes, &mut at)));
            let offset = unzigzag(get(bytes, &mut at));
            let coded = unzigzag(get(bytes, &mut at));
            register = match (flags >> REGISTER_SHIFT) & 3 {
                REGISTER_FROM_LAST => register.wrapping_add(coded),
                REGISTER_FROM_PC => pc.wrapping_add(coded),
                _ => coded,
            };
            if k >= start {
                out.push(BranchOperation::new(
                    pc,
                    offset,
                    register,
                    flags & JALR != 0,
                ));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{CompactBranch, INDEX_EVERY, OP_BYTES};
    use crate::tables::branch::{BranchOperation, generate_branch_trace};

    /// Ops whose fields reach both ends of u64 and i64 and sit near each other,
    /// in a pattern that changes every few ops.
    fn ops(n: usize) -> Vec<BranchOperation> {
        let edges = [
            0u64,
            1,
            2,
            0x7fff_ffff_ffff_ffff,
            0x8000_0000_0000_0000,
            u64::MAX,
            u64::MAX - 1,
        ];
        (0..n)
            .map(|i| {
                let k = i as u64;
                let pc = match i % 5 {
                    0 => 0x1000 + 4 * k,
                    1 => edges[i % edges.len()],
                    2 => 0x2000_0000u64.wrapping_sub(4 * k),
                    _ => 0x1_0000_0000 + 8 * k,
                };
                let offset = match i % 3 {
                    0 => (-(k as i64 % 4096)) as u64,
                    1 => edges[(i / 3) % edges.len()],
                    _ => 4 * (k % 1024),
                };
                let register = match i % 7 {
                    0 => pc.wrapping_add(4),
                    1 => edges[(i / 7) % edges.len()],
                    2 => k.wrapping_mul(0x9e37_79b9_7f4a_7c15),
                    _ => k / 3,
                };
                BranchOperation::new(pc, offset, register, i % 4 == 1)
            })
            .collect()
    }

    /// Every range of ops, across the sparse index and across small blocks,
    /// expands to the ops pushed, fields at both ends of their ranges
    /// included; no op takes more than 31 bytes, so none needs a raw form.
    #[test]
    fn compact_branch_ops_expand_to_what_was_pushed() {
        let all = ops(3 * INDEX_EVERY + 77);
        for block_bytes in [0usize, 64, 1000] {
            let mut list = if block_bytes == 0 {
                CompactBranch::default()
            } else {
                CompactBranch::with_block_bytes(block_bytes)
            };
            // Appended in uneven windows, as the builder appends them.
            for window in all.chunks(997) {
                list.extend(window);
            }
            assert_eq!(list.len(), all.len());
            assert_eq!(
                list.range(0, all.len()),
                all,
                "the whole list, blocks of {block_bytes}"
            );
            for (start, end) in [
                (0, 1),
                (INDEX_EVERY - 1, INDEX_EVERY + 1),
                (INDEX_EVERY, 2 * INDEX_EVERY),
                (5, 3 * INDEX_EVERY + 77),
                (2 * INDEX_EVERY + 3, 2 * INDEX_EVERY + 3),
                (3 * INDEX_EVERY + 70, 9 * INDEX_EVERY),
            ] {
                assert_eq!(
                    list.range(start, end),
                    all[start.min(all.len())..end.min(all.len())],
                    "ops {start}..{end}, blocks of {block_bytes}"
                );
            }
            let stream: usize = list.bytes.iter().map(Vec::len).sum();
            assert!(
                stream <= OP_BYTES * all.len(),
                "{stream} bytes for {} ops",
                all.len()
            );
        }
    }

    /// The worst op (every field a full ten-byte varint) round trips, at 31
    /// bytes; the common op (a nearby pc, a short offset, a return address
    /// near the pc) takes a few.
    #[test]
    fn the_worst_op_is_kept_and_the_common_one_is_short() {
        let worst = BranchOperation::new(
            0x8000_0000_0000_0000,
            0x8000_0000_0000_0000,
            0x5555_5555_5555_5555,
            true,
        );
        let list = CompactBranch::from_ops(std::slice::from_ref(&worst));
        assert_eq!(list.range(0, 1), vec![worst]);
        assert!(list.bytes[0].len() <= OP_BYTES);
        let common: Vec<BranchOperation> = (0..1000u64)
            .map(|k| {
                BranchOperation::new(
                    0x10_0000 + 16 * k,
                    (-64i64) as u64,
                    0x10_0004 + 16 * k,
                    k % 2 == 0,
                )
            })
            .collect();
        let list = CompactBranch::from_ops(&common);
        assert_eq!(list.range(0, 1000), common);
        let stream: usize = list.bytes.iter().map(Vec::len).sum();
        assert!(
            stream <= 6 * common.len(),
            "{stream} bytes for 1000 common ops"
        );
    }

    /// The BRANCH table of the expanded ops is the table of the ops.
    #[test]
    fn the_table_of_compact_branch_ops_is_the_table_of_the_ops() {
        let all = ops(2 * INDEX_EVERY + 5);
        let list = CompactBranch::from_ops(&all);
        let words = |ops: &[BranchOperation]| {
            let t = generate_branch_trace(ops);
            let m = &t.main_table;
            (0..m.height)
                .flat_map(|r| (0..m.width).map(move |c| *m.get(r, c).value()))
                .collect::<Vec<u64>>()
        };
        assert_eq!(words(&list.range(0, all.len())), words(&all));
    }
}
