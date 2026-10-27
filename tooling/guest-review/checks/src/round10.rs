macro_rules! cursor_cases {
    ($name:ident,$trie:ident,$rlp:ident,$crypto:ident) => {
        mod $name {
            use $trie::{TrieDB,Nibbles,Node,NodeRef,NodeHash,error::TrieError,node::{BranchNode,ExtensionNode,LeafNode}};
            use $rlp::encode::RLPEncode;
            use std::sync::{Arc,Mutex};
            struct Db { mode:usize, reply:Vec<u8>, paths:Arc<Mutex<Vec<Vec<u8>>>> }
            impl TrieDB for Db {
                fn get(&self,path:Nibbles)->Result<Option<Vec<u8>>,TrieError>{
                    self.paths.lock().unwrap().push(path.into_vec());
                    match self.mode {0=>Ok(Some(self.reply.clone())),1=>Ok(None),2=>Ok(Some(vec![])),3=>Err(TrieError::InvalidInput),_=>Ok(Some(vec![0xff]))}
                }
                fn put_batch(&self,_:Vec<(Nibbles,Vec<u8>)>)->Result<(),TrieError>{unreachable!()}
            }
            fn branch(choice:usize,child:NodeRef)->Node {
                let mut children=BranchNode::EMPTY_CHOICES;
                children[choice]=child;
                BranchNode::new_with_value(children,vec![0x7e]).into()
            }
            pub fn run()->Vec<String>{
                let mut outputs=vec![];
                for depth in 0..=8 {
                    for extension in [false,true] {
                        for mode in 0..7 {
                            let leaf:Node=LeafNode::new(Nibbles::from_hex(vec![6,16]),vec![0xa5;3]).into();
                            let reply=leaf.encode_to_vec();
                            let child=match mode {
                                5=>leaf.into(),
                                6=>{let mut raw=[0u8;31];raw[..reply.len()].copy_from_slice(&reply);NodeRef::Hash(NodeHash::Inline((raw,reply.len() as u8)))},
                                _=>NodeRef::Hash(NodeHash::Hashed([0x33;32].into())),
                            };
                            let mut root=if extension {ExtensionNode::new(Nibbles::from_hex(vec![5]),child).into()} else {branch(5,child)};
                            let mut prefix=vec![];
                            for layer in 0..depth {
                                let mut head=if layer%2==0 {vec![1,2]} else {vec![3]};
                                root=if layer%2==0 {ExtensionNode::new(Nibbles::from_hex(head.clone()),root.into()).into()} else {branch(3,root.into())};
                                head.extend(prefix);prefix=head;
                            }
                            for consumed in [false,true] {
                                for tail in [vec![5,6,16],vec![],vec![17,6,16],vec![4,6,16],vec![5],vec![5,7,16],vec![16]] {
                                    let mut query=prefix.clone();query.extend(tail);
                                    let mut path=if consumed {let mut p=vec![9,8,7];p.extend(query);let mut p=Nibbles::from_hex(p);for _ in 0..3 {p.next();}p} else {Nibbles::from_hex(query)};
                                    // A pre-consumed path must survive a later database fallback unchanged.
                                    let paths=Arc::new(Mutex::new(vec![]));
                                    let db=Db{mode,reply:reply.clone(),paths:paths.clone()};
                                    let before=root.compute_hash(&$crypto::NativeCrypto);
                                    let result=root.get(&db,path.take());
                                    assert_eq!(before,root.compute_hash(&$crypto::NativeCrypto));
                                    outputs.push(format!("{depth}/{extension}/{mode}/{consumed}: {result:?} {:?}",paths.lock().unwrap()));
                                }
                            }
                        }
                    }
                }
                outputs
            }
        }
    }
}
cursor_cases!(actual,ethrex_trie,ethrex_rlp,ethrex_crypto);
cursor_cases!(expected,reference_trie,reference_rlp,reference_crypto);
#[test]
fn cursor_resident_prefixes_and_database_fallbacks_match_original() {
    assert_eq!(actual::run(),expected::run());
}

#[test]
fn repeated_trie_queries_and_mutations_match_original() {
    let mut actual=ethrex_trie::Trie::new_temp();
    let mut expected=reference_trie::Trie::new_temp();
    let keys:Vec<Vec<u8>>=(0..192u32).map(|i|{let mut key=vec![0x12;32];key[28..].copy_from_slice(&i.to_be_bytes());key}).collect();
    for round in 0..4 {
        for (i,key) in keys.iter().enumerate() {
            if (i+round)%4==0 {assert_eq!(actual.remove(key).unwrap(),expected.remove(key).unwrap());}
            else {let value=vec![i as u8;1+(i*7+round)%80];actual.insert(key.clone(),value.clone()).unwrap();expected.insert(key.clone(),value).unwrap();}
            for query in [key.clone(),vec![0xff;32],vec![]] {assert_eq!(actual.get(&query).unwrap(),expected.get(&query).unwrap());}
        }
        for key in &keys {assert_eq!(actual.get(key).unwrap(),expected.get(key).unwrap());}
        assert_eq!(actual.hash_no_commit(&ethrex_crypto::NativeCrypto),expected.hash_no_commit(&reference_crypto::NativeCrypto));
    }
}

#[test]
fn specialized_remainder_matches_full_width_reference() {
    use ethrex_common::U256;
    use ethrex_levm::opcode_handlers::arithmetic::rem_with_u64_fast_path;
    let mut values=vec![U256::zero(),U256::one(),U256::MAX];
    for bit in 0..256 {
        let power=U256::one()<<bit;
        values.extend([power,power-U256::one(),power.saturating_add(U256::one())]);
    }
    for &lhs in &values { for &rhs in &values {
        assert_eq!(rem_with_u64_fast_path(lhs,rhs),lhs.checked_rem(rhs).unwrap_or_default(),"lhs={lhs}, rhs={rhs}");
    }}
    let mut seed=0x25368371_25453112u64;
    let mut random=||{seed^=seed<<13;seed^=seed>>7;seed^=seed<<17;seed};
    for _ in 0..20_000 {
        let lhs=U256([random(),random(),random(),random()]);
        for rhs in [U256::from(random() as u32),U256::from(random()),U256([random(),random(),random(),random()])] {
            assert_eq!(rem_with_u64_fast_path(lhs,rhs),lhs.checked_rem(rhs).unwrap_or_default());
        }
    }
}

macro_rules! flat_key_cases {
    ($name:ident,$trie:ident) => {
        mod $name {
            use $trie::{Trie,TrieDB,Nibbles,Node,node::LeafNode,error::TrieError};
            use std::sync::{Arc,Mutex};
            struct Db {computed:bool,reply:usize,events:Arc<Mutex<Vec<String>>>}
            impl TrieDB for Db {
                // Intentionally implement only the legacy hook. The new default
                // borrowed hook must still call it, preserving downstream backends.
                fn flatkeyvalue_computed(&self,key:Nibbles)->bool {
                    self.events.lock().unwrap().push(format!("computed:{:?}",key.into_vec()));self.computed
                }
                fn get(&self,key:Nibbles)->Result<Option<Vec<u8>>,TrieError>{
                    self.events.lock().unwrap().push(format!("get:{:?}",key.into_vec()));
                    match self.reply {0=>Ok(Some(vec![0xa5])),1=>Ok(Some(vec![])),2=>Ok(None),_=>Err(TrieError::InvalidInput)}
                }
                fn put_batch(&self,_:Vec<(Nibbles,Vec<u8>)>)->Result<(),TrieError>{unreachable!()}
            }
            pub fn run()->Vec<String> {
                let mut out=vec![];
                for computed in [false,true] {for reply in 0..4 {
                    let events=Arc::new(Mutex::new(vec![]));
                    let mut trie=Trie::new(Box::new(Db{computed,reply,events:events.clone()}));
                    let key=vec![0x12,0x34];
                    let node:Node=LeafNode::new(Nibbles::from_bytes(&key),vec![0x33]).into();trie.root=node.into();
                    for query in [&key,&vec![0xab],&vec![]] {out.push(format!("{:?}",trie.get(query)));}
                    trie.insert(key.clone(),vec![0x44]).unwrap();
                    out.push(format!("{:?}",trie.get(&key)));
                    out.push(format!("{:?}",events.lock().unwrap()));
                }}
                out
            }
        }
    }
}
flat_key_cases!(flat_actual,ethrex_trie);
flat_key_cases!(flat_expected,reference_trie);
#[test]
fn borrowed_flat_key_hook_preserves_legacy_overrides_and_dirty_keys() {
    assert_eq!(flat_actual::run(),flat_expected::run());
}
