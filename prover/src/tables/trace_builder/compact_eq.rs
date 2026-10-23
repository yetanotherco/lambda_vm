//! EQ ops as the builder keeps them until their tables are built: a
//! [`DeltaStream`] instead of 24 bytes an op (S3c-EQ).
//!
//! An op is a flags byte (`invert`, and how each operand is coded) and two
//! varints: `a` as it is or from the previous op's `a`, and `b` as it is, from
//! the op's own `a` (equal operands cost a byte) or from the previous op's
//! `b`, each the cheapest. Every op takes at most 21 bytes, so any op,
//! whatever its values, is kept losslessly and none is held raw.

use super::delta::{Codec, DeltaStream, cheapest, get, put, unzigzag, zigzag};
use super::eq::EqOperation;

/// EQ's [`Codec`]: the previous op's operands.
#[derive(Clone, Copy, Default)]
pub(super) struct EqCodec {
    a: u64,
    b: u64,
}

pub(super) type CompactEq = DeltaStream<EqCodec>;

/// `flags`: the op inverts.
const INVERT: u8 = 1;
/// `flags` bits 1–2: how `a` is coded; bits 3–4: how `b` is.
const A_SHIFT: u8 = 1;
const B_SHIFT: u8 = 3;
const ABSOLUTE: u8 = 0;
const FROM_LAST: u8 = 1;
const FROM_A: u8 = 2;

impl Codec for EqCodec {
    type Op = EqOperation;
    /// The flags byte and two varints of at most ten.
    const MAX_BYTES: usize = 21;

    fn encode(&mut self, op: &EqOperation, out: &mut Vec<u8>) {
        let (a_mode, a) = cheapest([
            (ABSOLUTE, zigzag(op.a)),
            (FROM_LAST, zigzag(op.a.wrapping_sub(self.a))),
        ]);
        let (b_mode, b) = cheapest([
            (ABSOLUTE, zigzag(op.b)),
            (FROM_A, zigzag(op.b.wrapping_sub(op.a))),
            (FROM_LAST, zigzag(op.b.wrapping_sub(self.b))),
        ]);
        out.push((u8::from(op.invert) * INVERT) | (a_mode << A_SHIFT) | (b_mode << B_SHIFT));
        put(out, a);
        put(out, b);
        self.a = op.a;
        self.b = op.b;
    }

    fn decode(&mut self, bytes: &[u8], at: &mut usize) -> EqOperation {
        let flags = bytes[*at];
        *at += 1;
        let a = unzigzag(get(bytes, at));
        self.a = match (flags >> A_SHIFT) & 3 {
            FROM_LAST => self.a.wrapping_add(a),
            _ => a,
        };
        let b = unzigzag(get(bytes, at));
        self.b = match (flags >> B_SHIFT) & 3 {
            FROM_A => self.a.wrapping_add(b),
            FROM_LAST => self.b.wrapping_add(b),
            _ => b,
        };
        EqOperation::new(self.a, self.b, flags & INVERT != 0)
    }
}

#[cfg(test)]
mod tests {
    use super::super::delta::{Codec, with_index_every};
    use super::{CompactEq, EqCodec};
    use crate::tables::eq::{EqOperation, generate_eq_trace};

    const INDEX_EVERY: usize = 4096;
    const OP_BYTES: usize = <EqCodec as Codec>::MAX_BYTES;

    /// Operands at both ends of u64 / i64, equal, near each other and far
    /// apart, in a pattern that changes every few ops.
    fn ops(n: usize) -> Vec<EqOperation> {
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
                let a = match i % 4 {
                    0 => k,
                    1 => edges[i % edges.len()],
                    2 => k.wrapping_mul(0x9e37_79b9_7f4a_7c15),
                    _ => 0x10_0000 + 8 * k,
                };
                let b = match i % 5 {
                    0 => a,
                    1 => edges[(i / 5) % edges.len()],
                    2 => a.wrapping_add(1),
                    3 => 0,
                    _ => k.wrapping_mul(0x2545_f491_4f6c_dd1d),
                };
                EqOperation::new(a, b, i % 3 == 1)
            })
            .collect()
    }

    /// Every range, across the sparse index and small blocks, and every chunk
    /// around the index's spacing expands to the ops pushed; no op takes more
    /// than 21 bytes, so none needs a raw form.
    #[test]
    fn compact_eq_ops_expand_to_what_was_pushed() {
        let all = ops(3 * INDEX_EVERY + 77);
        for block_bytes in [0usize, 64, 1000] {
            let mut list = if block_bytes == 0 {
                CompactEq::default()
            } else {
                CompactEq::with_block_bytes(block_bytes)
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
        let list = CompactEq::from_ops(&all);
        for chunk in [1000usize, INDEX_EVERY - 1, INDEX_EVERY + 1, 5000] {
            for (k, want) in all.chunks(chunk).enumerate() {
                assert_eq!(
                    list.range(k * chunk, (k + 1) * chunk),
                    want,
                    "chunk {k} of {chunk}"
                );
            }
        }
        let dense = with_index_every(3, || CompactEq::from_ops(&all));
        for (k, want) in all.chunks(7).enumerate().take(500) {
            assert_eq!(dense.range(k * 7, (k + 1) * 7), want, "dense chunk {k}");
        }
    }

    /// The worst op round trips at 21 bytes; equal operands and small ones
    /// take three.
    #[test]
    fn the_worst_eq_op_is_kept_and_the_common_one_is_short() {
        let worst = EqOperation::new(0x5555_5555_5555_5555, 0x8000_0000_0000_0000, true);
        let list = CompactEq::from_ops(std::slice::from_ref(&worst));
        assert_eq!(list.range(0, 1), vec![worst]);
        assert!(list.stream_bytes() <= OP_BYTES);
        let common: Vec<EqOperation> = (0..1000u64)
            .map(|k| EqOperation::new(k % 50, if k % 2 == 0 { k % 50 } else { 0 }, k % 3 == 0))
            .collect();
        let list = CompactEq::from_ops(&common);
        assert_eq!(list.range(0, 1000), common);
        assert!(
            list.stream_bytes() <= 3 * common.len(),
            "{} bytes",
            list.stream_bytes()
        );
    }

    /// The EQ table of the expanded ops is the table of the ops.
    #[test]
    fn the_table_of_compact_eq_ops_is_the_table_of_the_ops() {
        let all = ops(2 * INDEX_EVERY + 5);
        let list = CompactEq::from_ops(&all);
        let words = |ops: &[EqOperation]| {
            let t = generate_eq_trace(ops);
            let m = &t.main_table;
            (0..m.height)
                .flat_map(|r| (0..m.width).map(move |c| *m.get(r, c).value()))
                .collect::<Vec<u64>>()
        };
        assert_eq!(words(&list.range(0, all.len())), words(&all));
    }
}
