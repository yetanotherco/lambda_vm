//! [`Tail`] hands chunks out as shared ranges of its window parts.

use std::sync::{Arc, Weak};

use super::{Segments, Tail};

fn ops(segments: &Segments<u32>) -> Vec<u32> {
    segments.iter().copied().collect()
}

/// Chunks taken across part boundaries are the consecutive ranges of the run's
/// list, whatever the parts' lengths, and so is what `into_vec` leaves.
#[test]
fn a_tail_hands_out_the_runs_list_in_order() {
    for sizes in [vec![5, 7, 1, 17], vec![4, 4, 4], vec![1; 10], vec![30]] {
        for chunk in [1, 3, 4, 7, 12] {
            let mut tail = Tail::new();
            let mut next = 0u32;
            for &size in &sizes {
                tail.push((next..next + size).collect());
                next += size;
            }
            let mut out = Vec::new();
            while tail.len >= chunk {
                let taken = tail.take(chunk);
                assert_eq!(taken.contiguous().len(), chunk);
                out.extend(ops(&taken));
            }
            out.extend(tail.into_vec());
            assert_eq!(out, (0..next).collect::<Vec<_>>(), "{sizes:?} / {chunk}");
        }
    }
}

/// A part shared by two chunks lives until both are dropped, and no longer: the
/// tail lets go of a part once a chunk has taken its last op, so nothing it
/// handed out stays resident through it.
#[test]
fn a_part_is_freed_once_its_chunks_are() {
    let mut tail = Tail::new();
    tail.push(vec![0u32; 10]);
    tail.push(vec![1u32; 10]);
    let first: Weak<Vec<u32>> = Arc::downgrade(&tail.parts[0]);
    let second: Weak<Vec<u32>> = Arc::downgrade(&tail.parts[1]);

    // A chunk ending inside the first part: the tail and the chunk share it.
    let a = tail.take(6);
    assert_eq!(first.strong_count(), 2);
    // The next chunk takes the first part's rest and some of the second: the
    // tail lets go of the first part.
    let b = tail.take(6);
    assert_eq!(first.strong_count(), 2, "chunks a and b, not the tail");
    assert_eq!(tail.parts.len(), 1);
    drop(a);
    assert_eq!(first.strong_count(), 1);
    drop(b);
    assert!(first.upgrade().is_none(), "the first part is freed");
    assert_eq!(second.strong_count(), 1, "the tail's");

    // The rest of the second part: the tail lets go of it, the chunk holds it
    // alone, and it is freed with the chunk.
    let c = tail.take(8);
    assert_eq!(tail.len, 0);
    assert_eq!(second.strong_count(), 1, "chunk c's");
    assert_eq!(ops(&c), vec![1u32; 8]);
    drop(c);
    assert!(second.upgrade().is_none(), "the second part is freed");
}

/// A chunk that is one part's range is generated from the part itself (a
/// borrow); one that spans parts is made one list for the generator.
#[test]
fn a_chunk_in_one_part_is_not_copied() {
    let mut tail = Tail::new();
    tail.push((0u32..8).collect());
    tail.push((8u32..16).collect());
    let whole = tail.take(8);
    assert!(matches!(whole.contiguous(), std::borrow::Cow::Borrowed(_)));
    tail.push((16u32..24).collect());
    let _inside = tail.take(4);
    let across = tail.take(8);
    assert!(matches!(across.contiguous(), std::borrow::Cow::Owned(_)));
    assert_eq!(
        across.contiguous().to_vec(),
        (12u32..20).collect::<Vec<_>>()
    );
}
