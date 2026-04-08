//! Generates a serialized ProgramInput for an empty Ethereum block.
//!
//! Strategy: Build an empty block, execute it to get the correct state root,
//! then reconstruct the block with the correct root. Finally generate the
//! execution witness and serialize.
//!
//! Output: `../../executor/tests/ethrex_empty_block.bin`

use ethrex_blockchain::{Blockchain, BlockchainOptions};
use ethrex_common::types::{Block, BlockBody, BlockHeader, Genesis, GenesisAccount};
use ethrex_common::{Address, H256, U256};
use ethrex_storage::{EngineType, Store};
use guest_program::input::ProgramInput;
use std::collections::BTreeMap;

#[tokio::main]
async fn main() {
    println!("=== Generating ProgramInput for an empty Ethereum block ===");

    // Step 1: Create genesis and blockchain
    println!("[1/4] Creating blockchain...");
    let genesis = create_minimal_genesis();

    let mut store = Store::new("", EngineType::InMemory).expect("Failed to create store");
    store
        .add_initial_state(genesis.clone())
        .await
        .expect("Failed to add genesis");

    let blockchain = Blockchain::new(store.clone(), BlockchainOptions::default());

    let genesis_header = store
        .get_block_header(0)
        .expect("Failed to get genesis header")
        .expect("Genesis header not found");

    println!("  Genesis hash: {:?}", genesis_header.compute_block_hash());
    println!("  Genesis state_root: {:?}", genesis_header.state_root);

    // Step 2: For an empty block with no transactions and no withdrawals,
    // the state doesn't change. State root = parent state root.
    println!("[2/4] Building empty block...");
    let state_root = genesis_header.state_root;

    let empty_block = build_empty_block(&genesis_header, state_root);
    println!(
        "  Block 1 hash: {:?}",
        empty_block.header.compute_block_hash()
    );

    // Step 3: Add block and generate witness
    println!("[3/4] Adding block and generating witness...");
    let blocks = vec![empty_block];

    for block in &blocks {
        blockchain
            .add_block(block.clone())
            .expect("Failed to add block");
    }

    let witness = blockchain
        .generate_witness_for_blocks(&blocks)
        .await
        .expect("Failed to generate witness");

    // Step 4: Serialize ProgramInput
    println!("[4/4] Serializing...");
    let program_input = ProgramInput {
        blocks,
        execution_witness: witness,
        elasticity_multiplier: 2,
        fee_configs: None,
    };

    let serialized =
        rkyv::to_bytes::<rkyv::rancor::Error>(&program_input).expect("Failed to serialize");

    let output_path = "../../executor/tests/ethrex_empty_block.bin";
    std::fs::write(output_path, &serialized).expect("Failed to write");

    println!("Done! {} bytes -> {}", serialized.len(), output_path);
}

fn create_minimal_genesis() -> Genesis {
    let mut alloc = BTreeMap::new();
    let address: Address = "0x1000000000000000000000000000000000000000"
        .parse()
        .unwrap();
    alloc.insert(
        address,
        GenesisAccount {
            balance: U256::from(1_000_000_000_000_000_000u128),
            code: Default::default(),
            storage: Default::default(),
            nonce: 0,
        },
    );

    Genesis {
        config: Default::default(),
        alloc,
        coinbase: Default::default(),
        difficulty: U256::zero(),
        extra_data: Default::default(),
        gas_limit: 30_000_000,
        nonce: 0,
        mix_hash: Default::default(),
        timestamp: 0,
        base_fee_per_gas: Some(1_000_000_000),
        blob_gas_used: None,
        excess_blob_gas: None,
        requests_hash: None,
    }
}

fn build_empty_block(parent: &BlockHeader, state_root: H256) -> Block {
    let gas_limit = parent.gas_limit;

    let base_fee = parent.base_fee_per_gas.map(|parent_fee| {
        let target = parent.gas_limit / 2;
        let deficit = target - parent.gas_used;
        let decrease = parent_fee * deficit / target / 8;
        parent_fee.saturating_sub(decrease).max(1)
    });

    // Well-known hashes for empty lists
    let empty_ommers: H256 = "0x1dcc4de8dec75d7aab85b567b6ccd41ad312451b948a7413f0a142fd40d49347"
        .parse()
        .unwrap();
    let empty_trie: H256 = "0x56e81f171bcc55a6ff8345e692c0f86e5b48e01b996cadc001622fb5e363b421"
        .parse()
        .unwrap();

    let header = BlockHeader {
        parent_hash: parent.compute_block_hash(),
        ommers_hash: empty_ommers,
        transactions_root: empty_trie,
        receipts_root: empty_trie,
        state_root,
        number: parent.number + 1,
        gas_limit,
        gas_used: 0,
        timestamp: parent.timestamp + 12,
        base_fee_per_gas: base_fee,
        difficulty: U256::zero(),
        ..Default::default()
    };

    Block {
        header,
        body: BlockBody {
            transactions: vec![],
            ommers: vec![],
            withdrawals: None,
        },
    }
}
