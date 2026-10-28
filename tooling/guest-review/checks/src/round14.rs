use ethrex_trie::{
    Nibbles, NodeHash, NodeRef,
    node::{BranchNode, LeafNode, Node},
};
use rkyv::rancor::Error;
use std::sync::{Arc, OnceLock};

#[test]
fn mixed_branch_children_preserve_archive_values_and_sharing() {
    for inline_len in 0..=31u8 {
        let leaf = Arc::new(Node::Leaf(LeafNode::new(
            Nibbles::from_bytes(&[0x12, 0x34]),
            vec![0x77; 40],
        )));
        let shared = Arc::new(Node::Branch(Box::new(BranchNode::new(
            std::array::from_fn(|i| {
                if i % 2 == 0 {
                    NodeRef::Node(leaf.clone(), OnceLock::new())
                } else {
                    NodeRef::Hash(NodeHash::Inline(([0x80; 31], inline_len)))
                }
            }),
        ))));
        for rotation in 0..16 {
            for value_len in [0, 1, 4096] {
                let children = std::array::from_fn(|i| match (i + rotation) % 4 {
                    0 | 1 => {
                        let cache = OnceLock::from(NodeHash::Hashed([0x55; 32].into()));
                        NodeRef::Node(shared.clone(), cache)
                    }
                    2 => NodeRef::Hash(NodeHash::Hashed([i as u8; 32].into())),
                    _ => NodeRef::Hash(NodeHash::Inline(([i as u8; 31], inline_len))),
                });
                let node = Node::Branch(Box::new(BranchNode::new_with_value(
                    children,
                    vec![0xaa; value_len],
                )));
                let bytes = rkyv::to_bytes::<Error>(&node).unwrap();
                let reference: reference_trie::node::Node =
                    rkyv::from_bytes::<_, Error>(&bytes).unwrap();
                assert_eq!(&*bytes, &*rkyv::to_bytes::<Error>(&reference).unwrap());
                let expected = serde_json::to_value(&reference).unwrap();
                let archived = rkyv::access::<rkyv::Archived<Node>, Error>(&bytes).unwrap();
                for capacity in [0, 1, 32] {
                    let got: Node = rkyv::api::deserialize_using::<_, _, Error>(
                        archived,
                        &mut rkyv::de::Pool::with_capacity(capacity),
                    )
                    .unwrap();
                    assert_eq!(serde_json::to_value(&got).unwrap(), expected);
                    assert_eq!(&*rkyv::to_bytes::<Error>(&got).unwrap(), &*bytes);
                    let Node::Branch(branch) = got else {
                        unreachable!()
                    };
                    let mut first = None;
                    for child in &branch.choices {
                        if let NodeRef::Node(arc, cache) = child {
                            assert!(cache.get().is_none(), "derived hash cache must stay empty");
                            if let Some(previous) = first {
                                assert!(Arc::ptr_eq(arc, previous), "shared child duplicated");
                            }
                            first = Some(arc);
                        }
                    }
                }
            }
        }
    }
}
