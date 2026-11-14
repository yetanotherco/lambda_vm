//! BRANCH ops as the builder keeps them until their tables are built: a
//! [`DeltaStream`] instead of 32 bytes an op (S3c-BRANCH).
//!
//! An op is a flags byte (`jalr`, and how its register is coded) and three
//! varints: the pc from the previous op's, the offset itself, and the register
//! the cheapest of three ways (as it is, from the previous op's register, or
//! from the op's own pc: a JALR's return address sits near it). Every op takes
//! at most 31 bytes, so any op, whatever its values, is kept losslessly and
//! none is held raw.

use super::BranchOperation;
use super::delta::{Codec, DeltaStream, cheapest, get, put, unzigzag, zigzag};

/// BRANCH's [`Codec`]: the previous op's pc and register.
#[derive(Clone, Copy, Default)]
pub(super) struct BranchCodec {
    pc: u64,
    register: u64,
}

pub(super) type CompactBranch = DeltaStream<BranchCodec>;

/// `flags`: the op is a JALR.
const JALR: u8 = 1;
/// `flags` bits 1–2: how the register is coded.
const REGISTER_SHIFT: u8 = 1;
const REGISTER_ABSOLUTE: u8 = 0;
const REGISTER_FROM_LAST: u8 = 1;
const REGISTER_FROM_PC: u8 = 2;

impl Codec for BranchCodec {
    type Op = BranchOperation;
    /// The flags byte and three varints of at most ten.
    const MAX_BYTES: usize = 31;

    fn encode(&mut self, op: &BranchOperation, out: &mut Vec<u8>) {
        let (mode, register) = cheapest([
            (REGISTER_ABSOLUTE, zigzag(op.register)),
            (
                REGISTER_FROM_LAST,
                zigzag(op.register.wrapping_sub(self.register)),
            ),
            (REGISTER_FROM_PC, zigzag(op.register.wrapping_sub(op.pc))),
        ]);
        out.push((u8::from(op.jalr) * JALR) | (mode << REGISTER_SHIFT));
        put(out, zigzag(op.pc.wrapping_sub(self.pc)));
        put(out, zigzag(op.offset));
        put(out, register);
        self.pc = op.pc;
        self.register = op.register;
    }

    fn decode(&mut self, bytes: &[u8], at: &mut usize) -> BranchOperation {
        let flags = bytes[*at];
        *at += 1;
        self.pc = self.pc.wrapping_add(unzigzag(get(bytes, at)));
        let offset = unzigzag(get(bytes, at));
        let coded = unzigzag(get(bytes, at));
        self.register = match (flags >> REGISTER_SHIFT) & 3 {
            REGISTER_FROM_LAST => self.register.wrapping_add(coded),
            REGISTER_FROM_PC => self.pc.wrapping_add(coded),
            _ => coded,
        };
        BranchOperation::new(self.pc, offset, self.register, flags & JALR != 0)
    }
}

#[cfg(test)]
mod tests {
    use super::super::delta::{Codec, with_index_every};
    use super::{BranchCodec, CompactBranch};

    const INDEX_EVERY: usize = 4096;
    const OP_BYTES: usize = <BranchCodec as Codec>::MAX_BYTES;
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
            let stream = list.stream_bytes();
            assert!(
                stream <= OP_BYTES * all.len(),
                "{stream} bytes for {} ops",
                all.len()
            );
        }
    }

    /// Chunks of every size around the index's spacing, each starting past an
    /// index mark, expand to their ops; so does a stream marked every three
    /// ops ([`with_index_every`]), where nearly every range starts past one.
    #[test]
    fn chunks_starting_past_an_index_mark_expand_to_their_ops() {
        let all = ops(3 * INDEX_EVERY + 77);
        let list = CompactBranch::from_ops(&all);
        assert_eq!(list.every(), INDEX_EVERY);
        for chunk in [1000usize, INDEX_EVERY - 1, INDEX_EVERY + 1, 5000] {
            for (k, want) in all.chunks(chunk).enumerate() {
                assert_eq!(
                    list.range(k * chunk, (k + 1) * chunk),
                    want,
                    "chunk {k} of {chunk}"
                );
            }
        }
        let dense = with_index_every(3, || CompactBranch::from_ops(&all));
        assert_eq!(dense.every(), 3);
        for chunk in [1usize, 2, 7, 100] {
            for (k, want) in all.chunks(chunk).enumerate().take(500) {
                assert_eq!(
                    dense.range(k * chunk, (k + 1) * chunk),
                    want,
                    "dense chunk {k} of {chunk}"
                );
            }
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
        assert!(list.stream_bytes() <= OP_BYTES);
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
        let stream = list.stream_bytes();
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
