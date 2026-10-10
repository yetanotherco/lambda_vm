//! The hand-out's two ways of taking a chunk out of a tail: as the window
//! parts it lies in ([`Tail::take`]) and copied into one list
//! ([`Tail::take_copy`]) give the same ops, and so does what stays behind.

use super::{Segments, Tail};

/// A fixed xorshift stream.
struct Rng(u64);

impl Rng {
    fn below(&mut self, n: u64) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x % n
    }
}

fn ops(segments: &Segments<u64>) -> Vec<u64> {
    segments.iter().copied().collect()
}

/// Parts of every length (empty ones too) pushed between takes of every size,
/// some a part exactly, some across many parts: the parts' chunks are the
/// copies, the reversed walk over them is the reversed copy, and the tails left
/// behind are equal.
#[test]
fn a_chunk_taken_as_parts_is_the_copied_chunk() {
    for seed in 1..=40u64 {
        let mut rng = Rng(seed.wrapping_mul(0x9e37_79b9_7f4a_7c15));
        let (mut shared, mut copied) = (Tail::new(), Tail::new());
        let mut next = 0u64;
        for _ in 0..60 {
            for _ in 0..rng.below(3) {
                let len = [0, 1, 7, 8, 9, 64][rng.below(6) as usize] + rng.below(40);
                let part: Vec<u64> = (next..next + len).collect();
                next += len;
                shared.push(part.clone());
                copied.push(part);
            }
            let n = [1, 8, 33, 64, 200][rng.below(5) as usize];
            while shared.len >= n && rng.below(4) != 0 {
                let parts = shared.take(n);
                let copy = copied.take_copy(n);
                assert_eq!(ops(&parts), copy, "seed {seed}");
                let reversed: Vec<u64> = parts.iter().rev().copied().collect();
                assert!(reversed.iter().eq(copy.iter().rev()), "seed {seed}");
                assert_eq!(parts.slices().iter().map(|s| s.len()).sum::<usize>(), n);
                assert_eq!(shared.held(), copied.held(), "seed {seed}");
            }
        }
        assert_eq!(shared.into_vec(), copied.into_vec(), "seed {seed}");
    }
}

/// A part a chunk uses only the front of stays shared with the tail; once the
/// next chunk uses up the rest, the tail no longer holds it.
#[test]
fn a_straddling_part_is_shared_until_used_up() {
    let mut tail = Tail::new();
    tail.push((0..10u64).collect());
    let first = tail.take(4);
    assert_eq!(ops(&first), vec![0, 1, 2, 3]);
    assert_eq!(tail.parts.len(), 1, "the part stays in the tail");
    let second = tail.take(6);
    assert_eq!(ops(&second), (4..10).collect::<Vec<_>>());
    assert!(tail.parts.is_empty(), "the part left with the last chunk");
    assert!(std::sync::Arc::ptr_eq(
        &first.parts[0].0,
        &second.parts[0].0
    ));
}
