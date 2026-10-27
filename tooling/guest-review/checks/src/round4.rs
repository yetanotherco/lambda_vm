use rkyv::rancor::Error;
use std::sync::{Arc, OnceLock};

#[path = "../../../../executor/programs/rust/ethrex/src/input.rs"]
mod guest_input;

#[test]
fn node_ref_archive_and_shared_deserialization_match_upstream() {
    use ethrex_trie::{NodeRef, NodeHash, Nibbles, node::{Node, LeafNode}};
    use reference_trie::{NodeRef as Ref, NodeHash as Hash, Nibbles as NibblesRef,
                         node::{Node as NodeRefType, LeafNode as LeafRef}};
    for size in [0, 1, 31, 32, 33, 256] {
        let shared = Arc::new(Node::Leaf(LeafNode::new(Nibbles::from_bytes(&[0x12, 0x34]), vec![0xa5; size])));
        let shared_ref = Arc::new(NodeRefType::Leaf(LeafRef::new(NibblesRef::from_bytes(&[0x12, 0x34]), vec![0xa5; size])));
        let cached = OnceLock::new();
        cached.set(NodeHash::Hashed([0x77;32].into())).unwrap();
        let reference_cached = OnceLock::new();
        reference_cached.set(Hash::Hashed([0x77;32].into())).unwrap();
        let mut nodes = vec![NodeRef::Node(shared.clone(), cached), NodeRef::Node(shared, OnceLock::new())];
        let mut reference = vec![Ref::Node(shared_ref.clone(), reference_cached), Ref::Node(shared_ref, OnceLock::new())];
        for n in 0..=31 {
            nodes.push(NodeRef::Hash(NodeHash::Inline(([n;31],n))));
            reference.push(Ref::Hash(Hash::Inline(([n;31],n))));
        }
        nodes.push(NodeRef::Hash(NodeHash::Hashed([0xef;32].into())));
        reference.push(Ref::Hash(Hash::Hashed([0xef;32].into())));
        let bytes = rkyv::to_bytes::<Error>(&nodes).unwrap();
        assert_eq!(&*bytes, &*rkyv::to_bytes::<Error>(&reference).unwrap());
        let expected: Vec<Ref> = rkyv::from_bytes::<_,Error>(&bytes).unwrap();
        let expected_json = serde_json::to_value(expected).unwrap();
        for capacity in [0, 1, 8, 64] {
            let archived = rkyv::access::<rkyv::Archived<Vec<NodeRef>>,Error>(&bytes).unwrap();
            let got: Vec<NodeRef> = rkyv::api::deserialize_using::<_,_,Error>(archived, &mut rkyv::de::Pool::with_capacity(capacity)).unwrap();
            assert_eq!(serde_json::to_value(&got).unwrap(), expected_json);
            let (NodeRef::Node(a, cache), NodeRef::Node(b, _)) = (&got[0], &got[1]) else { panic!() };
            assert!(Arc::ptr_eq(a,b), "shared nodes must remain shared");
            assert!(cache.get().is_none(), "Skip cache must deserialize empty");
        }
    }
}

#[test]
fn covered_memory_matches_upstream_for_boundaries_and_frame_reuse() {
    use ethrex_levm::memory::Memory;
    use reference_levm::memory::Memory as Reference;
    for initial in [0, 1, 31, 32, 33, 63, 64, 65, 255] {
        for request in 0..=129 {
            let mut actual = Memory::new();
            let mut expected = Reference::new();
            actual.resize(initial).unwrap(); expected.resize(initial).unwrap();
            actual.store_word(1, 0x1234u64.into()).unwrap();
            expected.store_word(1, 0x1234u64.into()).unwrap();
            actual.resize(request).unwrap(); expected.resize(request).unwrap();
            assert_eq!(actual.len(), expected.len());
            assert_eq!(actual.live_bytes(), expected.live_bytes());
            // The public length can be changed by callers; preserve rounding
            // semantics even for an unaligned logical length.
            actual.len = 33; expected.len = 33;
            actual.resize(request).unwrap(); expected.resize(request).unwrap();
            assert_eq!(actual.len(), expected.len());
            assert_eq!(actual.live_bytes(), expected.live_bytes());
            let mut child = actual.next_memory();
            let mut child_ref = expected.next_memory();
            child.store_word(31, 0xaabbccu64.into()).unwrap();
            child_ref.store_word(31, 0xaabbccu64.into()).unwrap();
            assert_eq!(child.live_bytes(), child_ref.live_bytes());
            child.clean_from_base(); child_ref.clean_from_base();
            assert_eq!(*actual.buffer.borrow(), *expected.buffer.borrow());
            actual.reset_for_reuse(); expected.reset_for_reuse();
            actual.resize(65).unwrap(); expected.resize(65).unwrap();
            assert_eq!(actual.live_bytes(), expected.live_bytes());
        }
    }
    for size in usize::MAX-30..=usize::MAX {
        let mut actual = Memory::new(); let mut expected = Reference::new();
        assert_eq!(actual.resize(size).unwrap_err().to_string(), expected.resize(size).unwrap_err().to_string());
    }
}

#[test]
fn reserved_decoder_matches_default_and_rejects_invalid_archives() {
    use ethrex_guest_program::l1::ProgramInput;
    // Exercise both sides of the reservation threshold with valid archives.
    for len in [0, 1, 65535, 65536, 65537, 131072] {
        let mut input = ProgramInput::default();
        input.execution_witness.codes.push(vec![0x5b; len]);
        let bytes = rkyv::to_bytes::<Error>(&input).unwrap();
        let expected = rkyv::from_bytes::<ProgramInput,Error>(&bytes).unwrap();
        let got = guest_input::decode_input(&bytes).unwrap();
        assert_eq!(serde_json::to_value(got).unwrap(), serde_json::to_value(expected).unwrap());
        for end in [0, 1, bytes.len()/2, bytes.len()-1] {
            let a = guest_input::decode_input(&bytes[..end]);
            let b = rkyv::from_bytes::<ProgramInput,Error>(&bytes[..end]);
            assert_eq!(a.is_ok(), b.is_ok());
        }
        let mut broken = bytes.to_vec();
        // Root Vec length / relative pointer corruption must undergo the same
        // checked-access path; compare acceptance with the default validator.
        for offset in 1..=32 {
            let index = broken.len()-offset;
            broken[index] ^= 0xff;
            let a = guest_input::decode_input(&broken);
            let b = rkyv::from_bytes::<ProgramInput,Error>(&broken);
            assert_eq!(a.is_ok(), b.is_ok(), "length {len}, byte {index}");
            if let (Ok(a), Ok(b)) = (a,b) { assert_eq!(serde_json::to_value(a).unwrap(),serde_json::to_value(b).unwrap()); }
            broken[index] ^= 0xff;
        }
    }
}

#[test]
fn empty_code_keeps_the_supplied_hash_and_wire_representation() {
    use ethrex_common::types::Code;
    for byte in [0, 1, 0xff] {
        let hash = [byte;32].into();
        let got = Code::from_bytecode_unchecked(Vec::new().into(), hash);
        let expected = Code::from_parts_unchecked(hash, &[], Arc::from(Vec::<u32>::new()));
        assert_eq!(got, expected);
        assert_eq!(serde_json::to_value(got).unwrap(), serde_json::to_value(expected).unwrap());
    }
}
