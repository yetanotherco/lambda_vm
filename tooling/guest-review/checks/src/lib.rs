#[cfg(test)]
mod tests {
    use ethrex_common::types::Code;
    use ethrex_crypto::NativeCrypto;
    use ethrex_rlp::encode::RLPEncode;
    use ethrex_trie::{node::BranchNode, NodeHash, NodeRef, Trie};

    fn reference_targets(code: &[u8]) -> Vec<u32> {
        let mut immediate = 0;
        let mut targets = Vec::new();
        for (i, &op) in code.iter().enumerate() {
            if immediate > 0 { immediate -= 1; }
            else if (0x60..=0x7f).contains(&op) { immediate = op as usize - 0x5f; }
            else if op == 0x5b { targets.push(i as u32); }
        }
        targets
    }

    fn check_code(bytes: Vec<u8>) {
        let expected = reference_targets(&bytes);
        let code = Code::from_bytecode_unchecked(bytes.clone().into(), Default::default());
        assert_eq!(code.jump_targets(), expected);
        for target in 0..=bytes.len()+33 {
            assert_eq!(code.is_valid_jump_target(target), expected.binary_search(&(target as u32)).is_ok());
        }
        assert!(!code.is_valid_jump_target(usize::MAX));
        assert_eq!(code.code(), bytes);
        assert_eq!(&code.dispatch_buf()[bytes.len()..], &[0;33]);
    }

    #[test]
    fn exhaustive_short_bytecode() {
        check_code(vec![]);
        for a in 0..=255 {
            check_code(vec![a]);
            for b in 0..=255 { check_code(vec![a,b]); }
        }
    }

    #[test]
    fn scanner_push_boundaries_and_random() {
        for width in 1..=32 {
            for available in 0..=width+1 {
                let mut code = vec![0x5b;31];
                code.push(0x5f + width as u8);
                code.extend(std::iter::repeat_n(0x5b,available));
                check_code(code);
            }
        }
        let mut seed = 0xda512179u32;
        for len in 0..2048 {
            check_code((0..len).map(|_| {
                seed ^= seed<<13; seed ^= seed>>17; seed ^= seed<<5; seed as u8
            }).collect());
        }
    }

    #[test]
    fn bitmap_precomputed_parts_preserve_old_membership_and_wire_format() {
        use std::sync::Arc;
        for len in [0,1,31,32,63,64,65,127,128,129,257] {
            let bytes: Vec<u8> = (0..len).map(|i| if i%3==0 {0x5b} else {0x60}).collect();
            let mut targets: Vec<u32> = (0..=len as u32+2).collect();
            targets.push(u32::MAX);
            let code = Code::from_parts_unchecked(Default::default(), &bytes, Arc::from(targets.clone()));
            for target in 0..len+66 {
                let expected = bytes.get(target) == Some(&0x5b) && targets.binary_search(&(target as u32)).is_ok();
                assert_eq!(code.is_valid_jump_target(target), expected);
            }
            assert!(!code.is_valid_jump_target(usize::MAX));
            let wire = serde_json::to_value(&code).unwrap();
            assert_eq!(wire["jump_targets"], serde_json::json!(targets));
            assert!(wire.get("jump_bitmap").is_none());
            let restored: Code = serde_json::from_value(wire).unwrap();
            assert_eq!(code, restored);
            assert_eq!(code, code.clone());
        }
        assert!(!Code::default().is_valid_jump_target(0));
    }

    #[test]
    fn code_serialization_preserves_unpadded_form() {
        for bytes in [vec![],vec![0x60,0x5b,0x5b],vec![0x7f],vec![0x5b;257]] {
            let code=Code::from_bytecode_unchecked(bytes.into(),Default::default());
            let json=serde_json::to_vec(&code).unwrap();
            let restored:Code=serde_json::from_slice(&json).unwrap();
            assert_eq!(code,restored);
            assert_eq!(json,serde_json::to_vec(&restored).unwrap());
        }
    }

    #[test]
    fn branch_encoding_matches_unchanged_trait_encoder() {
        for size in [0,1,2,31,32,36,37,38,39,40,55,56,127,128,255,256,1024,65535,65536] {
            for seed in 0..48 {
                let choices = core::array::from_fn(|i| match if seed<16 {
                    usize::from(i<seed)
                } else if seed<32 {
                    2*usize::from(i<seed-16)
                } else { (i+seed)%3 } {
                    0 => NodeRef::Hash(NodeHash::Inline(([0;31],0))),
                    1 => NodeRef::Hash(NodeHash::Hashed([i as u8;32].into())),
                    _ => {
                        let mut data=[0;31];data[..3].copy_from_slice(&[0xc2,0x20,0x01]);
                        NodeRef::Hash(NodeHash::Inline((data,3)))
                    }
                });
                let branch=BranchNode::new_with_value(choices,vec![seed as u8;size]);
                let expected=branch.encode_to_vec();
                for capacity in [0,1,64,2048] {
                    let mut actual=Vec::with_capacity(capacity);
                    actual.extend_from_slice(&[0x12,0x34]);
                    branch.encode_into_vec(&mut actual);
                    assert_eq!(&actual[..2],&[0x12,0x34]);
                    assert_eq!(&actual[2..],expected);
                    let expected_hash = NodeHash::from_encoded(&expected, &NativeCrypto);
                    assert_eq!(branch.compute_hash_no_alloc(&mut actual, &NativeCrypto), expected_hash);
                    assert!(actual.is_empty());
                }
            }
        }
    }

    #[test]
    fn trie_roots_match_unmodified_revision_after_mutation() {
        let mut actual=Trie::new_temp();
        let mut expected=reference_trie::Trie::new_temp();
        let keys:Vec<Vec<u8>>=(0..257u32).map(|i| {
            let mut key=vec![0x12;32];key[28..].copy_from_slice(&i.to_be_bytes());key
        }).collect();
        for (i,key) in keys.iter().enumerate() {
            let value=vec![(i%251) as u8;1+i%80];
            actual.insert(key.clone(),value.clone()).unwrap();
            expected.insert(key.clone(),value).unwrap();
            assert_eq!(actual.hash_no_commit(&NativeCrypto),expected.hash_no_commit(&reference_crypto::NativeCrypto));
            // Exercise already-memoized roots, not just initial traversal.
            assert_eq!(actual.hash_no_commit(&NativeCrypto),expected.hash_no_commit(&reference_crypto::NativeCrypto));
        }
        for (i,key) in keys.iter().enumerate().rev() {
            if i%3==0 {
                assert_eq!(actual.remove(key).unwrap(),expected.remove(key).unwrap());
            } else {
                let value=vec![0xa5;33];
                actual.insert(key.clone(),value.clone()).unwrap();expected.insert(key.clone(),value).unwrap();
            }
            assert_eq!(actual.hash_no_commit(&NativeCrypto),expected.hash_no_commit(&reference_crypto::NativeCrypto));
        }
        for key in keys {
            assert_eq!(actual.remove(&key).unwrap(),expected.remove(&key).unwrap());
            assert_eq!(actual.hash_no_commit(&NativeCrypto),expected.hash_no_commit(&reference_crypto::NativeCrypto));
        }
    }

    #[test]
    fn rlp_lengths_match_reference_at_all_prefix_widths() {
        let mut lengths: Vec<usize> = (0..=1024).collect();
        for bits in (8..usize::BITS).step_by(8) {
            let boundary = 1usize << bits;
            lengths.extend([boundary - 1, boundary, boundary + 1]);
        }
        lengths.push(usize::MAX);
        for len in lengths {
            let mut actual = vec![0x12, 0x34];
            let mut expected = actual.clone();
            ethrex_rlp::encode::encode_length(len, &mut actual);
            reference_rlp::encode::encode_length(len, &mut expected);
            assert_eq!(actual, expected, "list length {len}");
        }
        for len in [0,1,2,54,55,56,57,127,128,255,256,257,65535,65536,65537] {
            for byte in [0,1,0x7f,0x80,0xff] {
                let value = vec![byte; len];
                let actual = ethrex_rlp::encode::RLPEncode::encode_to_vec(value.as_slice());
                let expected = reference_rlp::encode::RLPEncode::encode_to_vec(value.as_slice());
                assert_eq!(actual, expected, "string length {len}, byte {byte}");
                assert_eq!(actual.len(), ethrex_rlp::encode::RLPEncode::length(value.as_slice()));
            }
        }
    }

    #[test]
    fn rlp_integers_match_reference() {
        let mut values: Vec<u64> = (0..=65536).collect();
        for bits in 1..64 {
            let boundary = 1u64 << bits;
            values.extend([boundary - 1, boundary, boundary + 1]);
        }
        values.push(u64::MAX);
        let mut seed = 0xa45ba781992fff11u64;
        for _ in 0..4096 {
            seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17;
            values.push(seed);
        }
        for value in values {
            let actual = ethrex_rlp::encode::RLPEncode::encode_to_vec(&value);
            let expected = reference_rlp::encode::RLPEncode::encode_to_vec(&value);
            assert_eq!(actual, expected, "integer {value}");
            assert_eq!(actual.len(), ethrex_rlp::encode::RLPEncode::length(&value));
            let value = value as usize;
            assert_eq!(ethrex_rlp::encode::RLPEncode::encode_to_vec(&value),
                reference_rlp::encode::RLPEncode::encode_to_vec(&value));
        }
    }

    #[test]
    fn branch_writer_maximum_children_and_uncached_nodes() {
        use ethrex_trie::{node::{LeafNode, Node}, Nibbles};
        use std::sync::{Arc, OnceLock};
        for kind in 0..4 {
            for value_len in [0,1,55,56,65536] {
                let choices = core::array::from_fn(|i| match kind {
                    0 => NodeRef::Hash(NodeHash::Hashed([i as u8;32].into())),
                    1 => NodeRef::Hash(NodeHash::Inline(([i as u8;31],31))),
                    2 => NodeRef::Hash(NodeHash::Inline(([0;31],0))),
                    _ => NodeRef::Node(Arc::new(Node::Leaf(LeafNode::new(
                        Nibbles::from_bytes(&[i as u8]), vec![i as u8;64],
                    ))),OnceLock::new()),
                });
                let branch=BranchNode::new_with_value(choices,vec![0xa5;value_len]);
                let mut scratch=Vec::new();
                // Hash before the reference encoding, so child caches start cold.
                let actual=branch.compute_hash_no_alloc(&mut scratch,&NativeCrypto);
                let expected=NodeHash::from_encoded(&branch.encode_to_vec(),&NativeCrypto);
                assert_eq!(actual,expected);
                assert!(scratch.is_empty());
                assert_eq!(branch.compute_hash_no_alloc(&mut scratch,&NativeCrypto),expected);
            }
        }
    }

    #[test]
    fn branch_writer_rejects_invalid_inline_lengths() {
        for len in [32,33,255] {
            let choices=core::array::from_fn(|_|NodeRef::Hash(NodeHash::Inline(([0;31],len))));
            let branch=BranchNode::new_with_value(choices,Vec::new());
            let mut scratch=Vec::new();
            let result=std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                branch.compute_hash_no_alloc(&mut scratch,&NativeCrypto)
            }));
            assert!(result.is_err());
        }
    }
}

#[cfg(test)]
mod round4;

#[cfg(test)]
mod round6;

#[cfg(test)]
mod round7;

#[cfg(test)]
mod round8;

#[cfg(test)]
mod round9;

#[cfg(test)]
mod round10;

#[cfg(test)]
mod round11;

#[cfg(test)]
mod round12;
