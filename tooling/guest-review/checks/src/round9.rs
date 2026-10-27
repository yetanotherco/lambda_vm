use ethrex_common::U256;
use ethrex_levm::call_frame::Stack;

#[test]
fn stack_pair_matches_previous_sequence_at_every_depth() {
    let mut base = Stack::default();
    for (i, slot) in base.values.iter_mut().enumerate() {
        *slot = U256([i as u64, !(i as u64), u64::MAX, 1 << (i % 64)]);
    }
    for offset in (0..=1025).chain([usize::MAX]) {
        let mut actual = base.clone();
        let mut expected = base.clone();
        actual.offset = offset;
        expected.offset = offset;
        let got = actual.pop1_and_top_mut().map(|(a,b)| { let out=(a,*b); *b=U256::MAX; out });
        let want = expected.pop1().and_then(|a| expected.top_mut().map(|b| { let out=(a,*b); *b=U256::MAX; out }));
        assert_eq!(got, want, "offset {offset}");
        assert_eq!(actual.offset, expected.offset);
        assert_eq!(actual.values, expected.values);
    }
}

fn check_dup<const N: usize>() {
    let mut actual = Stack::default();
    let mut expected = reference_levm::call_frame::Stack::default();
    for offset in 0..=1024 {
        for i in 0..1024 {
            let value=U256([i as u64, !(i as u64), 0x7fffffffffffffff, 1 << (i%64)]);
            actual.values[i]=value;
            expected.values[i]=value;
        }
        actual.offset=offset;
        expected.offset=offset;
        assert_eq!(format!("{:?}",actual.dup::<N>()), format!("{:?}",expected.dup::<N>()), "N={N}, offset={offset}");
        assert_eq!(actual.offset,expected.offset);
        assert_eq!(actual.values,expected.values);
    }
}

#[test]
fn dup_matches_pinned_stack_at_every_depth() {
    macro_rules! all { ($($n:expr),*) => { $(check_dup::<$n>();)* }; }
    all!(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,1023,1024,1025);
}

macro_rules! trie_scenario {
    ($name:ident,$trie:ident,$crypto:ident,$rlp:ident) => {
        mod $name {
            use $trie::{Trie, TrieDB, Nibbles, NodeHash, NodeRef, Node, error::TrieError, node::{BranchNode, ExtensionNode, LeafNode}};
            use $crypto::NativeCrypto;
            use $rlp::encode::RLPEncode;
            use std::sync::{Arc,Mutex};
            struct Db { reply: Vec<u8>, mode: u8, paths: Arc<Mutex<Vec<Vec<u8>>>> }
            impl TrieDB for Db {
                fn get(&self,key:Nibbles)->Result<Option<Vec<u8>>,TrieError>{
                    self.paths.lock().unwrap().push(key.into_vec());
                    match self.mode {0=>Ok(Some(self.reply.clone())),1=>Ok(None),2=>Ok(Some(vec![])),3=>Err(TrieError::InvalidInput),_=>Ok(Some(vec![0xff]))}
                }
                fn put_batch(&self,_:Vec<(Nibbles,Vec<u8>)>)->Result<(),TrieError>{unreachable!()}
            }
            pub fn run()->Vec<String> {
                let mut observations=Vec::new();
                for extension in [false,true] {
                    for resident in [false,true] {
                        for mode in 0..5 {
                            for operation in 0..3 {
                                let key=vec![0x12,0x34];
                                let suffix=if extension {vec![3,4,16]} else {vec![2,3,4,16]};
                                let leaf=Node::from(LeafNode::new(Nibbles::from_hex(suffix),vec![0xa5;40]));
                                let encoded=$trie::node::Node::encode_to_vec(&leaf);
                                let child=if resident {leaf.into()} else {NodeRef::Hash(NodeHash::Hashed([0x33;32].into()))};
                                let root:Node=if extension {
                                    ExtensionNode::new(Nibbles::from_hex(vec![1,2]),child).into()
                                } else {
                                    let mut choices=BranchNode::EMPTY_CHOICES;choices[1]=child;BranchNode::new(choices).into()
                                };
                                let paths=Arc::new(Mutex::new(vec![]));
                                let mut trie=Trie::new(Box::new(Db{reply:encoded,mode,paths:paths.clone()}));
                                trie.root=root.into();
                                // Force Arc::make_mut to preserve a shared snapshot on mutation.
                                let saved=trie.root.clone();
                                let before=saved.compute_hash(&NativeCrypto);
                                let result=match operation {
                                    0=>format!("{:?}",trie.get(&key)),
                                    1=>format!("{:?}",trie.insert(key.clone(),vec![0x5a;48])),
                                    _=>format!("{:?}",trie.remove(&key)),
                                };
                                assert_eq!(saved.compute_hash(&NativeCrypto),before);
                                observations.push(format!("{extension}/{resident}/{mode}/{operation}: {result}; paths={:?}; root={:?}",paths.lock().unwrap(),trie.hash_no_commit(&NativeCrypto)));
                            }
                        }
                    }
                }
                observations
            }
        }
    }
}
trie_scenario!(optimized,ethrex_trie,ethrex_crypto,ethrex_rlp);
trie_scenario!(reference,reference_trie,reference_crypto,reference_rlp);

#[test]
fn trie_resident_and_database_paths_match_pinned_revision() {
    assert_eq!(optimized::run(),reference::run());
}
