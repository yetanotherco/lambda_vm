//! BYTEWISE ops as the builder keeps them until their tables are built: a
//! [`DeltaStream`] instead of 24 bytes an op (S3c-BYTEWISE).
//!
//! An op is a flags byte (how each operand is coded, and the opcode when it is
//! under 8; a larger one follows in a byte of its own) and two varints: `a` as
//! it is or from the previous op's `a`, and `b` as it is (a mask), from the
//! op's own `a` or from the previous op's `b`, each the cheapest. Every op
//! takes at most 22 bytes, so any op, whatever its values, is kept losslessly
//! and none is held raw.

use super::bytewise::BytewiseOperation;
use super::delta::{Codec, DeltaStream, cheapest, get, put, unzigzag, zigzag};

/// BYTEWISE's [`Codec`]: the previous op's operands.
#[derive(Clone, Copy, Default)]
pub(super) struct BytewiseCodec {
    a: u64,
    b: u64,
}

pub(super) type CompactBytewise = DeltaStream<BytewiseCodec>;

/// `flags` bit 0: how `a` is coded; bits 1–2: how `b` is; bits 3–5: the
/// opcode; bit 6: the opcode is too large for bits 3–5 and follows.
const A_SHIFT: u8 = 0;
const B_SHIFT: u8 = 1;
const OP_SHIFT: u8 = 3;
const OP_FOLLOWS: u8 = 1 << 6;
const ABSOLUTE: u8 = 0;
const FROM_LAST: u8 = 1;
const FROM_A: u8 = 2;

impl Codec for BytewiseCodec {
    type Op = BytewiseOperation;
    /// The flags byte, an opcode byte and two varints of at most ten.
    const MAX_BYTES: usize = 22;

    fn encode(&mut self, op: &BytewiseOperation, out: &mut Vec<u8>) {
        let (a_mode, a) = cheapest([
            (ABSOLUTE, zigzag(op.a)),
            (FROM_LAST, zigzag(op.a.wrapping_sub(self.a))),
        ]);
        let (b_mode, b) = cheapest([
            (ABSOLUTE, zigzag(op.b)),
            (FROM_A, zigzag(op.b.wrapping_sub(op.a))),
            (FROM_LAST, zigzag(op.b.wrapping_sub(self.b))),
        ]);
        let modes = (a_mode << A_SHIFT) | (b_mode << B_SHIFT);
        if op.op < 8 {
            out.push(modes | (op.op << OP_SHIFT));
        } else {
            out.push(modes | OP_FOLLOWS);
            out.push(op.op);
        }
        put(out, a);
        put(out, b);
        self.a = op.a;
        self.b = op.b;
    }

    fn decode(&mut self, bytes: &[u8], at: &mut usize) -> BytewiseOperation {
        let flags = bytes[*at];
        *at += 1;
        let opcode = if flags & OP_FOLLOWS != 0 {
            let opcode = bytes[*at];
            *at += 1;
            opcode
        } else {
            (flags >> OP_SHIFT) & 7
        };
        let a = unzigzag(get(bytes, at));
        self.a = match (flags >> A_SHIFT) & 1 {
            FROM_LAST => self.a.wrapping_add(a),
            _ => a,
        };
        let b = unzigzag(get(bytes, at));
        self.b = match (flags >> B_SHIFT) & 3 {
            FROM_A => self.a.wrapping_add(b),
            FROM_LAST => self.b.wrapping_add(b),
            _ => b,
        };
        BytewiseOperation::new(self.a, self.b, opcode)
    }
}

#[cfg(test)]
mod tests {
    use super::super::delta::{Codec, with_index_every};
    use super::{BytewiseCodec, CompactBytewise};
    use crate::tables::bytewise::{BytewiseOperation, generate_bytewise_trace};
    use crate::tables::types::alu_op;

    const INDEX_EVERY: usize = 4096;
    const OP_BYTES: usize = <BytewiseCodec as Codec>::MAX_BYTES;

    /// Operands at both ends of u64 / i64, masks, near and far, every opcode
    /// the table takes, in a pattern that changes every few ops.
    fn ops(n: usize) -> Vec<BytewiseOperation> {
        let edges = [
            0u64,
            1,
            0xff,
            0xffff,
            0x7fff_ffff_ffff_ffff,
            0x8000_0000_0000_0000,
            u64::MAX,
        ];
        let opcodes = [alu_op::AND, alu_op::OR, alu_op::XOR];
        (0..n)
            .map(|i| {
                let k = i as u64;
                let a = match i % 4 {
                    0 => k,
                    1 => edges[i % edges.len()],
                    2 => k.wrapping_mul(0x9e37_79b9_7f4a_7c15),
                    _ => 0x10_0000 + 8 * k,
                };
                let b = match i % 5 {
                    0 => 0xff,
                    1 => edges[(i / 5) % edges.len()],
                    2 => a,
                    3 => !a,
                    _ => k.wrapping_mul(0x2545_f491_4f6c_dd1d),
                };
                BytewiseOperation::new(a, b, opcodes[i % 3])
            })
            .collect()
    }

    /// Every range, across the sparse index and small blocks, and every chunk
    /// around the index's spacing expands to the ops pushed, an opcode too
    /// large for the flags byte included; no op takes more than 22 bytes, so
    /// none needs a raw form.
    #[test]
    fn compact_bytewise_ops_expand_to_what_was_pushed() {
        let mut all = ops(3 * INDEX_EVERY + 77);
        // Opcodes past XOR are not the table's, but the list keeps any byte:
        // the ones the flags byte holds (3–7) and the ones that follow it.
        for (i, op) in [3u8, 4, 5, 6, 7].into_iter().enumerate() {
            all[11 + 2 * i].op = op;
        }
        all[17].op = 8;
        all[INDEX_EVERY + 5].op = u8::MAX;
        for block_bytes in [0usize, 64, 1000] {
            let mut list = if block_bytes == 0 {
                CompactBytewise::default()
            } else {
                CompactBytewise::with_block_bytes(block_bytes)
            };
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
                (2 * INDEX_EVERY + 3, 2 * INDEX_EVERY + 3),
                (3 * INDEX_EVERY + 70, 9 * INDEX_EVERY),
            ] {
                assert_eq!(
                    list.range(start, end),
                    all[start.min(all.len())..end.min(all.len())],
                    "ops {start}..{end}, blocks of {block_bytes}"
                );
            }
            assert!(list.stream_bytes() <= OP_BYTES * all.len());
        }
        let list = CompactBytewise::from_ops(&all);
        for chunk in [1000usize, INDEX_EVERY - 1, INDEX_EVERY + 1, 5000] {
            for (k, want) in all.chunks(chunk).enumerate() {
                assert_eq!(
                    list.range(k * chunk, (k + 1) * chunk),
                    want,
                    "chunk {k} of {chunk}"
                );
            }
        }
        let dense = with_index_every(3, || CompactBytewise::from_ops(&all));
        for (k, want) in all.chunks(7).enumerate().take(500) {
            assert_eq!(dense.range(k * 7, (k + 1) * 7), want, "dense chunk {k}");
        }
    }

    /// The worst op round trips at 22 bytes; a byte mask on a small value
    /// takes three (the first op, with no mask before it, four).
    #[test]
    fn the_worst_bytewise_op_is_kept_and_the_common_one_is_short() {
        let worst = BytewiseOperation::new(0x5555_5555_5555_5555, 0x8000_0000_0000_0000, 200);
        let list = CompactBytewise::from_ops(std::slice::from_ref(&worst));
        assert_eq!(list.range(0, 1), vec![worst]);
        assert!(list.stream_bytes() <= OP_BYTES);
        let common: Vec<BytewiseOperation> = (0..1000u64)
            .map(|k| BytewiseOperation::new(k % 60, 0xff, alu_op::AND))
            .collect();
        let list = CompactBytewise::from_ops(&common);
        assert_eq!(list.range(0, 1000), common);
        assert!(
            list.stream_bytes() <= 3 * common.len() + 1,
            "{} bytes",
            list.stream_bytes()
        );
    }

    /// The BYTEWISE table of the expanded ops is the table of the ops.
    #[test]
    fn the_table_of_compact_bytewise_ops_is_the_table_of_the_ops() {
        let all = ops(2 * INDEX_EVERY + 5);
        let list = CompactBytewise::from_ops(&all);
        let words = |ops: &[BytewiseOperation]| {
            let t = generate_bytewise_trace(ops);
            let m = &t.main_table;
            (0..m.height)
                .flat_map(|r| (0..m.width).map(move |c| *m.get(r, c).value()))
                .collect::<Vec<u64>>()
        };
        assert_eq!(words(&list.range(0, all.len())), words(&all));
    }
}
