use ethrex_common::types::Code;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, Barrier};

fn targets(bytes: &[u8]) -> Vec<u32> {
    let mut result = Vec::new();
    let mut pc = 0;
    while pc < bytes.len() {
        let byte = bytes[pc];
        if byte == 0x5b { result.push(pc as u32); }
        pc += if (0x60..=0x7f).contains(&byte) { byte as usize - 0x5e } else { 1 };
    }
    result
}

fn check_queries(bytes: Vec<u8>, order: &[usize]) {
    let expected = targets(&bytes);
    // Do not call serialization, equality, Hash, or jump_targets until after
    // queries: those APIs may eagerly materialize the complete analysis.
    let code = Code::from_bytecode_unchecked(bytes.clone().into(), Default::default());
    let clone = code.clone();
    for (index, &pc) in order.iter().enumerate() {
        let instance = if index % 2 == 0 { &code } else { &clone };
        assert_eq!(instance.is_valid_jump_target(pc),
            pc < bytes.len() && expected.binary_search(&(pc as u32)).is_ok(),
            "bytes={bytes:x?}, pc={pc}");
    }
    assert!(!code.is_valid_jump_target(usize::MAX));
    assert_eq!(code.jump_targets(), expected);
    assert_eq!(clone.jump_targets(), expected);
    let eager = Code::from_parts_unchecked(Default::default(), &bytes, Arc::from(expected));
    assert_eq!(code, eager);
    assert_eq!(serde_json::to_vec(&code).unwrap(), serde_json::to_vec(&eager).unwrap());
    let digest = |value: &Code| {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        value.hash(&mut hasher);
        hasher.finish()
    };
    assert_eq!(digest(&code), digest(&eager));
}

#[test]
fn lazy_queries_match_linear_scanner_before_materialization() {
    let mut seed = 0xd253_8371u32;
    for len in [0, 1, 31, 32, 33, 63, 64, 65, 255, 256, 257, 511, 512, 513, 2048, 24576] {
        for pattern in 0..12 {
            let bytes: Vec<_> = (0..len).map(|i| {
                seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5;
                match pattern {
                    0 => 0x5b,
                    1 => 0x60,
                    2 => 0x7f,
                    3 => if i % 2 == 0 { 0x60 } else { 0x5b },
                    4 => if i % 33 == 0 { 0x7f } else { 0x5b },
                    5..=8 => [0x00, 0x5b, 0x60, 0x7f][seed as usize % 4],
                    _ => seed as u8,
                }
            }).collect();
            let ascending: Vec<_> = (0..=len).collect();
            check_queries(bytes.clone(), &ascending);
            check_queries(bytes.clone(), &ascending.iter().copied().rev().collect::<Vec<_>>());
            let shuffled: Vec<_> = (0..len + 100).map(|_| {
                seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5;
                seed as usize % (len + 33)
            }).collect();
            check_queries(bytes, &shuffled);
        }
    }
}

#[test]
fn lazy_queries_cover_push_and_chunk_boundaries() {
    for boundary in [32, 64, 99, 100, 231, 232, 256, 512] {
        for width in 1..=32 {
            for distance in 1..=33 {
                if distance > boundary { continue; }
                let start = boundary - distance;
                let mut bytes = vec![0x5b; boundary + 40];
                bytes[start] = 0x5f + width;
                let order = [boundary, start + 1, start, start + width as usize,
                             start + width as usize + 1, boundary - 1, 0];
                check_queries(bytes, &order);
                let mut truncated = vec![0x5b; boundary];
                truncated[start] = 0x5f + width;
                check_queries(truncated, &order);
            }
        }
    }
}

#[test]
fn lazy_analysis_is_shared_and_safe_under_concurrent_queries() {
    let bytes: Vec<u8> = (0..4096).map(|i| [0x5b, 0x7f, 0x60, 0x5b, 0x00][i % 5]).collect();
    let expected = Arc::new(targets(&bytes));
    let code = Code::from_bytecode_unchecked(bytes.into(), Default::default());
    let barrier = Arc::new(Barrier::new(8));
    let handles: Vec<_> = (0..8).map(|worker| {
        let code = code.clone();
        let expected = expected.clone();
        let barrier = barrier.clone();
        std::thread::spawn(move || {
            barrier.wait();
            for n in 0..4096 {
                let pc = (n * 31 + worker * 47) % 4096;
                assert_eq!(code.is_valid_jump_target(pc), expected.binary_search(&(pc as u32)).is_ok());
                if n == 1024 && worker % 2 == 0 {
                    assert_eq!(code.jump_targets(), expected.as_slice());
                }
            }
        })
    }).collect();
    for handle in handles { handle.join().unwrap(); }
}

#[test]
fn in_place_branches_drop_partial_children_on_error_and_unwind() {
    use ethrex_trie::{node::{Node, ArchivedNode, BranchNode, LeafNode}, NodeRef, Nibbles};
    use rkyv::{de::{Pool, Pooling, pooling::{PoolingState, ErasedPtr}}, rancor::Error};
    struct FaultPool {
        pool: Pool,
        calls: usize,
        fail_at: usize,
        unwind: bool,
    }
    impl Pooling<Error> for FaultPool {
        fn start_pooling(&mut self, address: usize) -> PoolingState {
            self.calls += 1;
            if self.calls == self.fail_at {
                if self.unwind { panic!("injected pooling failure"); }
                return PoolingState::Pending;
            }
            <Pool as Pooling<Error>>::start_pooling(&mut self.pool, address)
        }
        unsafe fn finish_pooling(&mut self, address: usize, ptr: ErasedPtr,
                                 drop: unsafe fn(ErasedPtr)) -> Result<(), Error> {
            // Forward the caller's pointer/drop pair unchanged.
            unsafe { <Pool as Pooling<Error>>::finish_pooling(&mut self.pool, address, ptr, drop) }
        }
    }
    let children = std::array::from_fn(|index| {
        NodeRef::from(Arc::new(Node::Leaf(LeafNode::new(
            Nibbles::from_bytes(&[index as u8]), vec![index as u8; index + 1]))))
    });
    let node = Node::Branch(Box::new(BranchNode::new_with_value(children, vec![0x99; 4096])));
    let bytes = rkyv::to_bytes::<Error>(&node).unwrap();
    let archived = rkyv::access::<ArchivedNode, Error>(&bytes).unwrap();
    let ArchivedNode::Branch(branch) = archived else { unreachable!() };
    for unwind in [false, true] {
        for fail_at in 1..=16 {
            let mut pool = FaultPool { pool: Pool::new(), calls: 0, fail_at, unwind };
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                rkyv::api::deserialize_using::<Node, _, Error>(archived, &mut pool)
            }));
            if unwind { assert!(result.is_err()); }
            else { assert!(result.unwrap().is_err()); }
            for child in &branch.choices[..fail_at - 1] {
                let restored: NodeRef = rkyv::api::deserialize_using::<_, _, Error>(child, &mut pool).unwrap();
                let NodeRef::Node(arc, _) = restored else { unreachable!() };
                // One owner is the pool, the other is restored. A leaked partial
                // branch would leave an extra strong owner here.
                assert_eq!(Arc::strong_count(&arc), 2, "fail_at={fail_at}, unwind={unwind}");
            }
        }
    }
    let restored: Node = rkyv::from_bytes::<_, Error>(&bytes).unwrap();
    assert_eq!(restored, node);
}
