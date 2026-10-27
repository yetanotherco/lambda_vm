use reference_crypto::{Crypto, NativeCrypto};
use std::io::{self, Write};
fn main() {
    let mut out=io::BufWriter::new(io::stdout().lock());
    for len in (0u32..=410).chain([544,545,1024,4095,4096,16384]) {
        for alignment in 0u8..8 {
            for pattern in 0u8..4 {
                let data:Vec<u8>=(0..len+u32::from(alignment)).map(|i|match pattern {
                    0=>0,1=>255,2=>(i.wrapping_mul(0x9e3779b9)>>24) as u8,_=>i as u8,
                }).collect();
                let expected=NativeCrypto.keccak256(&data[alignment as usize..]);
                out.write_all(&len.to_le_bytes()).unwrap();
                out.write_all(&[alignment,pattern,0,0]).unwrap();
                out.write_all(&expected).unwrap();
            }
        }
    }
}
