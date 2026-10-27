use ethrex_common::{U256, utils::{u256_to_big_endian, u256_from_big_endian, u256_from_big_endian_const}};
use ethrex_levm::{memory::Memory, opcode_handlers::arithmetic::div_with_u64_fast_path};
use std::hint::black_box;

fn check_width<const N: usize>(bytes: &[u8; 32]) {
    let mut input = [0u8; N];
    input.copy_from_slice(&bytes[32 - N..]);
    let mut expected = U256::zero();
    for &byte in &input {
        expected = (expected << 8) | U256::from(byte);
    }
    assert_eq!(u256_from_big_endian_const(black_box(input)), expected);
    assert_eq!(u256_from_big_endian(black_box(&input)), expected);
}

fn check_word(value: U256) {
    let mut expected = [0u8; 32];
    for (index, byte) in expected.iter_mut().enumerate() {
        let n = 31 - index;
        *byte = (value.0[n / 8] >> (8 * (n % 8))) as u8;
    }
    assert_eq!(u256_to_big_endian(black_box(value)), expected);
    macro_rules! widths { ($($n:expr),*) => { $(check_width::<$n>(&expected);)* }; }
    widths!(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,
            17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32);
}

pub fn main() {
    check_full_width_division();
    check_zero_memory_words();
    check_keccak();
    check_wide_division();
    check_nibbles();
    check_trie();
    check_stack();
    check_scanner();
    for lhs in [0u64, 1, 2, 255, u32::MAX as u64, u64::MAX] {
        for rhs in [0u64, 1, 2, 3, u32::MAX as u64, u64::MAX] {
            assert_eq!(div_with_u64_fast_path(black_box(lhs.into()), black_box(rhs.into())),
                U256::from(lhs.checked_div(rhs).unwrap_or(0)));
        }
    }
    check_word(U256::zero());
    check_word(U256::MAX);
    for bit in 0..256 {
        check_word(U256::one() << bit);
    }
    let mut seed = 0x25368371_25453112u64;
    let mut random = || {
        seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17; seed
    };
    let mut memory = Memory::new();
    for _ in 0..256 {
        let lhs = U256([random(), random(), random(), random()]);
        check_word(lhs);
        for offset in 0..64 {
            memory.store_word(offset, black_box(lhs)).unwrap();
            let expected = lhs.to_big_endian();
            assert_eq!(&memory.live_bytes()[offset..offset + 32], &expected);
            assert_eq!(memory.load_word(offset).unwrap(), lhs);
        }
        for rhs in [U256::zero(), U256::one(), U256::MAX,
                    U256::one() << (random() % 256),
                    U256::from(u32::MAX), U256::from(u64::MAX),
                    U256::from(random() as u32), U256::from(random()),
                    U256([random(), random(), random(), random()])] {
            assert_eq!(div_with_u64_fast_path(black_box(lhs), black_box(rhs)),
                lhs.checked_div(rhs).unwrap_or_default());
        }
    }
    for _ in 0..256 {
        let lhs=U256([random(),random(),random(),random()]);
        for rhs in [U256::zero(),U256::one(),U256::MAX,U256::one()<<(random()%256),
                    U256::from(u32::MAX),U256::from(u64::MAX),U256::from(random() as u32),
                    U256::from(random()),U256([random(),random(),random(),random()])] {
            assert_eq!(ethrex_levm::opcode_handlers::arithmetic::rem_with_u64_fast_path(black_box(lhs),black_box(rhs)),
                lhs.checked_rem(rhs).unwrap_or_default());
        }
    }
    // Output only after every assertion has completed inside the pinned RV64 VM.
    lambda_vm_syscalls::syscalls::commit(b"R12CHECK");
}


fn check_scanner() {
    use ethrex_common::types::Code;
    let mut seed = 0x25368371u32;
    for len in [1, 32, 99, 100, 231, 232, 255, 512, 2048, 24576] {
        for pattern in 0..4 {
            let bytes: Vec<u8> = (0..len).map(|i| {
                seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5;
                match pattern {
                    0 => 0x5b,
                    1 => if i % 33 == 0 { 0x7f } else { 0x5b },
                    2 => [0x00, 0x5b, 0x60, 0x7f][seed as usize % 4],
                    _ => seed as u8,
                }
            }).collect();
            let mut expected = vec![false; len];
            let mut pc = 0;
            while pc < len {
                let opcode = bytes[pc];
                expected[pc] = opcode == 0x5b;
                pc += if (0x60..=0x7f).contains(&opcode) { opcode as usize - 0x5e } else { 1 };
            }
            let probes: Vec<usize> = [0,1,31,32,33,63,64,65,98,99,100,230,231,232,255,256,len-1,len,usize::MAX]
                .into_iter().chain((0..64).map(|_| {
                    seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5;
                    seed as usize % len
                })).collect();
            for reverse in [false, true] {
                let code = Code::from_bytecode_unchecked(bytes.clone().into(), Default::default());
                let clone = code.clone();
                for i in 0..probes.len() {
                    let target = probes[if reverse { probes.len()-1-i } else { i }];
                    let instance = if i % 2 == 0 { &code } else { &clone };
                    assert_eq!(instance.is_valid_jump_target(black_box(target)), expected.get(target).copied().unwrap_or(false));
                }
                let targets: Vec<u32> = expected.iter().enumerate().filter_map(|(i, &valid)| valid.then_some(i as u32)).collect();
                assert_eq!(code.jump_targets(), targets);
            }
        }
    }
}

fn check_stack() {
    use ethrex_levm::{call_frame::Stack, errors::ExceptionalHalt};
    let mut stack = Stack::default();
    for (i, slot) in stack.values.iter_mut().enumerate() {
        *slot = U256([i as u64, !(i as u64), u64::MAX, 1 << (i % 64)]);
    }
    for offset in (0..=1025).chain([usize::MAX]) {
        stack.offset = black_box(offset);
        let expected = if offset < 1023 { Some((stack.values[offset], stack.values[offset+1])) } else { None };
        match stack.pop1_and_top_mut() {
            Ok((a,b)) => assert_eq!(Some((a,*b)),expected),
            Err(e) => {assert!(expected.is_none());assert_eq!(e,ExceptionalHalt::StackUnderflow);}
        }
        assert_eq!(stack.offset,if offset < 1024 {offset+1} else {offset});
    }
    macro_rules! depths { ($($n:expr),*) => { $(
        for offset in [0,1,2,1000,1008,1022,1023,1024] {
            stack.offset=black_box(offset);
            let expected=if offset+$n >= 1024 {Err(ExceptionalHalt::StackUnderflow)}
                else if offset==0 {Err(ExceptionalHalt::StackOverflow)} else {Ok(())};
            let before=if expected.is_ok() {stack.values[offset+$n]} else {U256::zero()};
            assert_eq!(stack.dup::<$n>(),expected);
            assert_eq!(stack.offset,if expected.is_ok() {offset-1} else {offset});
            if expected.is_ok() {assert_eq!(stack.values[offset-1],before);}
        }
    )* }; }
    depths!(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,1023,1024,1025);
}

fn compact_reference(data: &[u8]) -> Vec<u8> {
    let leaf=data.last()==Some(&16);
    let mut data=if leaf {&data[..data.len()-1]} else {data};
    let mut out=vec![];
    let mut prefix=if leaf {0x20u8} else {0};
    if data.len()%2==1 {prefix=prefix.wrapping_add(0x10).wrapping_add(data[0]);data=&data[1..];}
    out.push(prefix);
    for pair in data.chunks_exact(2) {out.push((pair[0]<<4)|pair[1]);}
    out
}
fn check_nibbles() {
    use ethrex_trie::Nibbles;
    // Exhaust all byte pairs, repeated across all four SWAR lanes.
    for pair in 0..=u16::MAX {
        let a=pair as u8;let b=(pair>>8) as u8;
        let bytes=black_box([a,b,b,a]);
        let expanded=Nibbles::from_raw(&bytes,false).into_vec();
        assert_eq!(expanded.as_slice(),&[a>>4,a&15,b>>4,b&15,b>>4,b&15,a>>4,a&15]);
        // End in zero to prevent a value of 16 from becoming the leaf marker.
        let nibbles=vec![a,b,a,b,a,b,a,b,0,0];
        let expected=compact_reference(&nibbles);
        assert_eq!(Nibbles::from_hex(black_box(nibbles)).encode_compact(),expected);
    }
    let mut seed=0x193a4172u32;
    for len in 0..=129 {
        for alignment in 0..8 {
            let bytes:Vec<u8>=(0..len+alignment).map(|_|{seed^=seed<<13;seed^=seed>>17;seed^=seed<<5;seed as u8}).collect();
            let bytes=black_box(&bytes[alignment..]);
            for leaf in [false,true] {
                let mut expected:Vec<u8>=bytes.iter().flat_map(|&b|[b>>4,b&15]).collect();
                if leaf {expected.push(16);}
                let path=Nibbles::from_raw(bytes,leaf);
                assert_eq!(path.as_ref(),expected.as_slice());
                assert_eq!(path.encode_compact(),compact_reference(&expected));
            }
            // All short nibble lengths and both odd/even packing offsets.
            let mut nibbles:Vec<u8>=bytes.iter().map(|b|b&15).collect();
            for leaf in [false,true] {
                if leaf {nibbles.push(16);}
                assert_eq!(Nibbles::from_hex(nibbles.clone()).encode_compact(),compact_reference(&nibbles));
            }
        }
    }
}
fn check_trie() {
    use ethrex_trie::{Trie,Nibbles,Node,node::{BranchNode,ExtensionNode,LeafNode}};
    for depth in 0..=16 {
        let mut root:Node=LeafNode::new(Nibbles::from_hex(vec![6,16]),vec![0xa5;3]).into();
        let mut path=vec![6,16];
        for i in 0..depth {
            if i%2==0 {
                root=ExtensionNode::new(Nibbles::from_hex(vec![1,2]),root.into()).into();
                let mut prefix=vec![1,2];prefix.extend(path);path=prefix;
            } else {
                let mut children=BranchNode::EMPTY_CHOICES;children[3]=root.into();
                root=BranchNode::new(children).into();path.insert(0,3);
            }
        }
        let trie=Trie::new_temp();
        assert_eq!(root.get(trie.db(),Nibbles::from_hex(path.clone())).unwrap(),Some(vec![0xa5;3]));
        path[0]=15;
        assert_eq!(root.get(trie.db(),Nibbles::from_hex(path)).unwrap(),None);
    }
}

fn check_keccak() {
    use ethrex_crypto::Crypto;
    use lambda_vm_ethrex_crypto::LambdaVmEcsmCrypto;
    let vectors=include_bytes!("keccak-vectors.bin");
    assert_eq!(vectors.len(),13344*40);
    for record in vectors.chunks_exact(40) {
        let len=u32::from_le_bytes(record[..4].try_into().unwrap()) as usize;
        let alignment=record[4] as usize;
        let pattern=record[5];
        let data:Vec<u8>=(0..len+alignment).map(|i|match pattern {
            0=>0,1=>255,2=>((i as u32).wrapping_mul(0x9e3779b9)>>24) as u8,_=>i as u8,
        }).collect();
        let actual=LambdaVmEcsmCrypto.keccak256(black_box(&data[alignment..]));
        assert_eq!(actual.as_slice(),&record[8..]);
    }
}
fn check_wide_division() {
    for bit in 32..64 {
        let power=1u64<<bit;
        for d in [power-1,power,power+1,power|0xffffffff,u64::MAX,10u64.pow(18)] {
            let denominator=U256::from(d);
            for r in [0,1,d/2,d-2,d-1] {
                for digit in [0,u32::MAX as u64-1,u32::MAX as u64] {
                    for position in [32,64,128,192] {
                        let numerator=(U256::from(r)<<position)|U256::from(digit);
                        assert_eq!(div_with_u64_fast_path(black_box(numerator),black_box(denominator)),numerator/denominator);
                    }
                }
            }
        }
    }
    let mut seed=0x1125368371254531u64;
    let mut random=||{seed^=seed<<13;seed^=seed>>7;seed^=seed<<17;seed};
    for _ in 0..1024 {
        let lhs=U256([random(),random(),random(),random()]);
        let rhs=U256::from(random() | (1u64<<32));
        assert_eq!(div_with_u64_fast_path(black_box(lhs),black_box(rhs)),lhs/rhs);
    }
}

fn check_full_width_division() {
    let one=U256::one();let base=one<<32;
    for digits in 3..=8 {
        for bit in 0..32 {
            let top=32*(digits-1)+bit;
            let d=(one<<top)+one;
            if top<255 {assert_eq!(div_with_u64_fast_path(black_box(d*U256::from(2)-one),black_box(d)),one);}
            if digits<8 {assert_eq!(div_with_u64_fast_path(black_box(d*base+d-one),black_box(d)),base);}
        }
    }
    let mut seed=0x1225368371254531u64;
    let mut random=||{seed^=seed<<13;seed^=seed>>7;seed^=seed<<17;seed};
    for width in 65..=256 {
        for _ in 0..8 {
            let raw=U256([random(),random(),random(),random()]);
            let d=(raw & (U256::MAX>>(256-width))) | (U256::one()<<(width-1));
            let n=U256([random(),random(),random(),random()]);
            assert_eq!(div_with_u64_fast_path(black_box(n),black_box(d)),n/d);
            let cap=U256::MAX/d;
            for q in [U256::zero(),U256::one(),cap,U256::from(random()).min(cap)] {
                let product=q*d;
                let r=(d-U256::one()).min(U256::MAX-product);
                for extra in [U256::zero(),r] {
                    assert_eq!(div_with_u64_fast_path(black_box(product+extra),black_box(d)),q);
                }
            }
        }
    }
}
fn check_zero_memory_words() {
    let mut memory=Memory::new();
    let mut seed=0x0abc_25368371u64;
    for mask in 0..16 {
        for _ in 0..8 {
            let mut value=U256::zero();
            for limb in 0..4 {
                seed^=seed<<13;seed^=seed>>7;seed^=seed<<17;
                if mask & (1<<limb)!=0 {value.0[limb]=seed|1;}
            }
            check_word(black_box(value));
            for offset in 0..8 {
                memory.store_word(offset,black_box(value)).unwrap();
                assert_eq!(&memory.live_bytes()[offset..offset+32],&value.to_big_endian());
                assert_eq!(memory.load_word(offset).unwrap(),value);
            }
        }
    }
}
