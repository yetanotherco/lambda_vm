//! Generate synthetic ethrex block fixtures (serialized `ProgramInput`) for the
//! lambda-vm prover/benchmarks — in-memory, offline, deterministic.
//!
//! Usage:
//!   cargo run -- <n_transfers> <out.bin>
//! e.g.
//!   cargo run -- 1  ../../executor/tests/ethrex_simple_tx.bin
//!   cargo run -- 10 ../../executor/tests/ethrex_10_transfers.bin
//!
//! TODO(ethrex-integration, PR #666): TEMPORARY. Delete this whole crate once
//! the LambdaVM-backend ethrex PR lands on ethrex `main` and fixtures are
//! generated via `ethrex-replay custom block` instead.
//!
//! Pinned to the same ethrex rev as the guest, so the rkyv `ProgramInput`
//! layout matches what the guest deserializes.

use bytes::Bytes;
use ethrex_blockchain::payload::{BuildPayloadArgs, create_payload};
use ethrex_blockchain::{Blockchain, BlockchainOptions};
use ethrex_common::types::{
    EIP1559Transaction, ELASTICITY_MULTIPLIER, Genesis, Transaction, TxKind,
};
use ethrex_common::{Address, H256, U256};
use ethrex_guest_program::l1::ProgramInput;
use ethrex_l2_rpc::signer::{LocalSigner, Signable, Signer};
use ethrex_storage::{EngineType, Store};
use secp256k1::SecretKey;

/// Well-known load-test rich account (funded in genesis.json). Key is public
/// dev material — not a secret.
const RICH_PK: &str = "bcdf20249abf0ed6d944c0288fad489e33f66b3960d9e6229c1cd214ed3bbe31";
const GENESIS_JSON: &str = include_str!("../genesis.json");

fn usage_and_exit(program: &str) -> ! {
    eprintln!("usage: {program} <n_transfers> <out.bin>");
    std::process::exit(2);
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args();
    let program = args.next().unwrap_or_else(|| "ethrex-fixtures".into());
    let Some(n_transfers) = args.next() else {
        usage_and_exit(&program);
    };
    let Some(out_path) = args.next() else {
        usage_and_exit(&program);
    };
    if args.next().is_some() {
        usage_and_exit(&program);
    }
    let n_transfers: u64 = n_transfers.parse()?;

    // --- 1. genesis -> in-memory store -------------------------------------
    let genesis: Genesis = serde_json::from_str(GENESIS_JSON)?;
    let chain_id = genesis.config.chain_id;
    let mut store = Store::new(".ethrex-fixtures-tmp", EngineType::InMemory)?;
    store.add_initial_state(genesis).await?;

    let head_number = store.get_latest_block_number().await?;
    let head = store
        .get_block_header(head_number)?
        .ok_or("missing genesis header")?;
    let parent_hash = head.hash();
    let parent_ts = head.timestamp;

    let blockchain = Blockchain::new(store.clone(), BlockchainOptions::default());

    // --- 2. build + sign N transfers, push to the mempool ------------------
    let signer: Signer = LocalSigner::new(SecretKey::from_slice(&hex::decode(RICH_PK)?)?).into();
    let recipient = Address::from_low_u64_be(0xdead_beef);
    for nonce in 0..n_transfers {
        let mut tx = Transaction::EIP1559Transaction(EIP1559Transaction {
            chain_id,
            nonce,
            max_priority_fee_per_gas: 1_000_000_000,
            max_fee_per_gas: 100_000_000_000,
            gas_limit: 21_000,
            to: TxKind::Call(recipient),
            value: U256::from(1u64),
            data: Bytes::new(),
            access_list: vec![],
            ..Default::default()
        });
        tx.sign_inplace(&signer).await?;
        blockchain.add_transaction_to_pool(tx).await?;
    }

    // --- 3. produce the block (fills + executes mempool txs) ---------------
    let payload_args = BuildPayloadArgs {
        parent: parent_hash,
        timestamp: parent_ts + 12,
        fee_recipient: Address::zero(),
        random: H256::zero(),
        withdrawals: Some(vec![]),
        beacon_root: Some(H256::zero()),
        slot_number: None,
        version: 3,
        elasticity_multiplier: ELASTICITY_MULTIPLIER,
        gas_ceil: 30_000_000,
    };
    let skeleton = create_payload(&payload_args, &store, Bytes::new())?;
    let result = blockchain.build_payload(skeleton)?;
    let block = result.payload;
    let included = block.body.transactions.len();
    assert_eq!(
        included as u64, n_transfers,
        "only {included}/{n_transfers} transactions made it into the block \
         (check gas limit / account balance / nonces)"
    );

    // --- 4. stateless witness -> ProgramInput -> rkyv ----------------------
    let witness = blockchain
        .generate_witness_for_blocks(&[block.clone()])
        .await?;
    let program_input = ProgramInput::new(vec![block], witness);
    let bytes = rkyv::to_bytes::<rkyv::rancor::Error>(&program_input)?;
    std::fs::write(&out_path, &bytes)?;

    println!(
        "wrote {out_path} ({} bytes): block #{} with {included}/{n_transfers} transfer(s)",
        bytes.len(),
        head_number + 1,
    );
    Ok(())
}
