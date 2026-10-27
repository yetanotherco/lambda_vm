use ethrex_common::types::{Block, Genesis};
use ethrex_common::types::block_execution_witness::{RpcExecutionWitness, decode_witness_headers};
use ethrex_guest_program::l1::{ProgramInput, execution_program};
use ethrex_crypto::NativeCrypto;
use serde::Deserialize;

#[derive(Deserialize)]
struct Cache { blocks: Vec<Block>, witness: RpcExecutionWitness, network: serde_json::Value }

fn main() {
    let args:Vec<_> = std::env::args().collect();
    let cache:Cache = serde_json::from_slice(&std::fs::read(&args[1]).unwrap()).unwrap();
    assert_eq!(cache.network, serde_json::json!({"PublicNetwork":"Mainnet"}));
    // Read the exact mainnet genesis shipped with the pinned ethrex revision.
    let genesis:Genesis = serde_json::from_slice(&std::fs::read(&args[2]).unwrap()).unwrap();
    let first = cache.blocks[0].header.number;
    let gas:u64 = cache.blocks.iter().map(|b|b.header.gas_used).sum();
    let txs:usize = cache.blocks.iter().map(|b|b.body.transactions.len()).sum();
    let headers = decode_witness_headers(&cache.witness.headers).unwrap();
    let witness = cache.witness.into_execution_witness(genesis.config,first,&headers,&NativeCrypto).unwrap();
    let input = ProgramInput::new(cache.blocks,witness);
    let bytes = rkyv::to_bytes::<rkyv::rancor::Error>(&input).unwrap();
    // The unmodified native guest must validate the canonical block before saving it.
    let expected = execution_program(input,std::sync::Arc::new(NativeCrypto)).unwrap().encode();
    assert_eq!(expected.len(),160);
    std::fs::write(&args[3],&bytes).unwrap();
    println!("block={first} gas={gas} transactions={txs} bytes={}",bytes.len());
}
