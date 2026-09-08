//! Generate synthetic ethrex block fixtures (SSZ stateless inputs) for the
//! lambda-vm prover/benchmarks — in-memory, offline, deterministic.
//!
//! Usage:
//!   cargo run -- <n_transfers> <output_path> [mode]
//!
//! mode (optional, default `same`):
//!   same        all txs: the rich sender -> one fixed recipient (0xdeadbeef)
//!   recipients  the rich sender -> N distinct recipients (1 -> N fan-out)
//!   distinct    N distinct, genesis-funded senders -> N distinct recipients
//!               (N independent, unrelated 1-1 account pairs)
//! e.g.
//!   cargo run -- 1  ../../executor/tests/ethrex_simple_tx.bin
//!   cargo run -- 10 ../../executor/tests/ethrex_10_transfers.bin
//!   cargo run -- 20 /tmp/ethrex_20_distinct.bin distinct
//!
//! TODO(ethrex-integration, PR #666): TEMPORARY. Delete this whole crate once
//! the LambdaVM-backend ethrex PR lands on ethrex `main` and fixtures are
//! generated via `ethrex-replay custom block` instead.
//!
//! Pinned to the same ethrex rev as the guest, so the SSZ layout matches what
//! the guest deserializes.

use bytes::Bytes;
use ethrex_blockchain::payload::{BuildPayloadArgs, create_payload};
use ethrex_blockchain::{Blockchain, BlockchainOptions};
use ethrex_common::types::block_access_list::BlockAccessList;
use ethrex_common::types::block_execution_witness::RpcExecutionWitness;
use ethrex_common::types::stateless_ssz::{
    Bytes20, ExecutionPayload, ExecutionRequests, LogsBloom, NewPayloadRequest,
    STATELESS_INPUT_SCHEMA_ID, SszExecutionWitness, SszPublicKeys, SszStatelessInput,
};
use ethrex_common::types::{
    EIP1559Transaction, ELASTICITY_MULTIPLIER, Genesis, GenesisAccount, Transaction, TxKind,
};
use ethrex_common::{Address, H256, U256};
use ethrex_guest_program::crypto::NativeCrypto;
use ethrex_guest_program::l1::run_stateless_guest;
use ethrex_l2_rpc::signer::{LocalSigner, Signable, Signer};
use ethrex_storage::{EngineType, Store};
use libssz::SszEncode;
use libssz_types::{ProgressiveList, SszList, SszVector};
use secp256k1::SecretKey;

/// Well-known load-test rich account (funded in genesis.json). Key is public
/// dev material — not a secret.
const RICH_PK: &str = "bcdf20249abf0ed6d944c0288fad489e33f66b3960d9e6229c1cd214ed3bbe31";
const GENESIS_JSON: &str = include_str!("../genesis.json");

/// How the block's transactions distribute across accounts.
#[derive(Clone, Copy, PartialEq)]
enum Mode {
    /// All txs: the rich sender -> one fixed recipient (original behavior).
    Same,
    /// The rich sender -> N distinct recipients (1 -> N fan-out).
    Recipients,
    /// N distinct, genesis-funded senders -> N distinct recipients.
    Distinct,
}

fn parse_mode(s: &str) -> Option<Mode> {
    match s {
        "same" => Some(Mode::Same),
        "recipients" | "fanout" => Some(Mode::Recipients),
        "distinct" | "diverse" => Some(Mode::Distinct),
        _ => None,
    }
}

fn usage_and_exit(program: &str) -> ! {
    eprintln!("usage: {program} <n_transfers> <output_path> [same|recipients|distinct]");
    std::process::exit(2);
}

/// Deterministic, distinct, valid secp256k1 signer for sender index `i`.
/// Key = 0x01 ‖ 0…0 ‖ big-endian(i): always nonzero and far below the curve order.
fn deterministic_signer(i: u64) -> Signer {
    let mut sk = [0u8; 32];
    sk[0] = 1;
    sk[24..32].copy_from_slice(&i.to_be_bytes());
    LocalSigner::new(SecretKey::from_slice(&sk).expect("valid secret key")).into()
}

fn rich_signer() -> Result<Signer, Box<dyn std::error::Error>> {
    Ok(LocalSigner::new(SecretKey::from_slice(&hex::decode(RICH_PK)?)?).into())
}

fn empty_execution_requests() -> ExecutionRequests {
    ExecutionRequests {
        deposits: ProgressiveList::new(),
        withdrawals: ProgressiveList::new(),
        consolidations: ProgressiveList::new(),
        builder_deposits: ProgressiveList::new(),
        builder_exits: ProgressiveList::new(),
    }
}

fn build_stateless_input(
    block: &ethrex_common::types::Block,
    witness: &RpcExecutionWitness,
    block_access_list: Option<&BlockAccessList>,
    chain_id: u64,
) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let header = &block.header;
    let bal = block_access_list.ok_or("Amsterdam fixture has no block access list")?;
    let block_access_list_hash = header
        .block_access_list_hash
        .ok_or("Amsterdam fixture header has no block_access_list_hash")?;
    if bal.compute_hash(&NativeCrypto) != block_access_list_hash {
        return Err("fixture BAL does not match block_access_list_hash".into());
    }
    let slot_number = header
        .slot_number
        .ok_or("Amsterdam fixture header has no slot_number")?;

    let transactions = block
        .body
        .transactions
        .iter()
        .map(|tx| tx.encode_canonical_to_vec().into())
        .collect::<Vec<ProgressiveList<u8>>>()
        .into();
    let public_keys = block
        .body
        .transactions
        .iter()
        .enumerate()
        .map(|(i, tx)| {
            let key = tx
                .public_key(&NativeCrypto)
                .map_err(|e| format!("failed to recover public key for transaction {i}: {e}"))?
                .ok_or_else(|| format!("transaction {i} has no recoverable signature"))?;
            SszVector::try_from(key.to_vec())
                .map_err(|e| format!("public key for transaction {i} is invalid: {e:?}"))
        })
        .collect::<Result<Vec<_>, String>>()?;
    let withdrawals = block
        .body
        .withdrawals
        .clone()
        .unwrap_or_default()
        .into_iter()
        .map(
            |withdrawal| ethrex_common::types::stateless_ssz::Withdrawal {
                index: withdrawal.index,
                validator_index: withdrawal.validator_index,
                address: Bytes20(withdrawal.address.0),
                amount: withdrawal.amount,
            },
        )
        .collect::<Vec<_>>()
        .into();
    let logs_bloom: LogsBloom = SszVector::try_from(header.logs_bloom.0.to_vec())?;
    let extra_data = SszList::try_from(header.extra_data.to_vec())?;
    let base_fee = header.base_fee_per_gas.ok_or("fixture has no base fee")?;
    let mut base_fee_bytes = [0u8; 32];
    base_fee_bytes[..8].copy_from_slice(&base_fee.to_le_bytes());

    let execution_payload = ExecutionPayload {
        parent_hash: header.parent_hash.0,
        fee_recipient: Bytes20(header.coinbase.0),
        state_root: header.state_root.0,
        receipts_root: header.receipts_root.0,
        logs_bloom,
        prev_randao: header.prev_randao.0,
        block_number: header.number,
        gas_limit: header.gas_limit,
        gas_used: header.gas_used,
        timestamp: header.timestamp,
        extra_data,
        base_fee_per_gas: base_fee_bytes,
        block_hash: header.compute_block_hash(&NativeCrypto).0,
        transactions,
        withdrawals,
        blob_gas_used: header.blob_gas_used.unwrap_or_default(),
        excess_blob_gas: header.excess_blob_gas.unwrap_or_default(),
        block_access_list: ethrex_rlp::encode::RLPEncode::encode_to_vec(bal).into(),
        slot_number,
    };
    let ssz_witness = SszExecutionWitness {
        state: ProgressiveList::from(
            witness
                .state
                .iter()
                .map(|bytes| SszList::try_from(bytes.to_vec()))
                .collect::<Result<Vec<_>, _>>()?,
        ),
        codes: ProgressiveList::from(
            witness
                .codes
                .iter()
                .map(|bytes| SszList::try_from(bytes.to_vec()))
                .collect::<Result<Vec<_>, _>>()?,
        ),
        headers: SszList::try_from(
            witness
                .headers
                .iter()
                .map(|bytes| SszList::try_from(bytes.to_vec()))
                .collect::<Result<Vec<_>, _>>()?,
        )?,
    };
    let input = SszStatelessInput {
        new_payload_request: NewPayloadRequest {
            execution_payload,
            versioned_hashes: block
                .body
                .transactions
                .iter()
                .flat_map(|tx| tx.blob_versioned_hashes())
                .map(|hash| hash.0)
                .collect::<Vec<_>>()
                .into(),
            parent_beacon_block_root: header.parent_beacon_block_root.unwrap_or_default().0,
            execution_requests: empty_execution_requests(),
        },
        witness: ssz_witness,
        chain_id,
        public_keys: SszPublicKeys::from(public_keys),
    };
    let mut bytes = STATELESS_INPUT_SCHEMA_ID.to_be_bytes().to_vec();
    input.ssz_append(&mut bytes);
    Ok(bytes)
}

/// Distinct recipient address for tx index `i` (fresh account, no funding needed).
fn recipient_for(i: u64) -> Address {
    Address::from_low_u64_be(0xdead_0000_0000u64 + i)
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
    let mode = match args.next() {
        None => Mode::Same,
        Some(s) => match parse_mode(&s) {
            Some(m) => m,
            None => usage_and_exit(&program),
        },
    };
    if args.next().is_some() {
        usage_and_exit(&program);
    }
    let Ok(n_transfers) = n_transfers.parse::<u64>() else {
        usage_and_exit(&program);
    };

    // --- 1. genesis -> in-memory store -------------------------------------
    let mut genesis: Genesis = serde_json::from_str(GENESIS_JSON)?;
    // The stateless guest consumes the Amsterdam SSZ schema. Keep the local
    // chain's fork schedule aligned with that schema while preserving the
    // fixture's deterministic execution rules.
    genesis.config.amsterdam_time = Some(0);
    genesis.config.blob_schedule.amsterdam = Some(genesis.config.blob_schedule.bpo2);

    // For `distinct`, fund each synthetic sender in genesis so its tx is valid.
    if mode == Mode::Distinct {
        for i in 0..n_transfers {
            genesis.alloc.insert(
                deterministic_signer(i).address(),
                GenesisAccount {
                    code: Bytes::new(),
                    storage: Default::default(),
                    balance: U256::from(100_000_000_000_000_000_000u128), // 100 ETH
                    nonce: 0,
                },
            );
        }
    }

    let chain_id = genesis.config.chain_id;
    let mut store = Store::new(".ethrex-fixtures-tmp", EngineType::InMemory)?;
    store.add_initial_state(genesis).await?;

    let head_number = store.get_latest_block_number()?;
    let head = store
        .get_block_header(head_number)?
        .ok_or("missing genesis header")?;
    let parent_hash = head.hash();
    let parent_ts = head.timestamp;

    let blockchain = Blockchain::new(store.clone(), BlockchainOptions::default());

    // --- 2. build + sign N transfers, push to the mempool ------------------
    // `same`:       rich sender, nonce 0..N, fixed recipient.
    // `recipients`: rich sender, nonce 0..N, distinct recipient per tx.
    // `distinct`:   distinct sender per tx (nonce 0), distinct recipient per tx.
    for i in 0..n_transfers {
        // `distinct` senders all use nonce 0 with otherwise-identical fees, so the
        // payload builder would tie-break block order by the mempool's wall-clock
        // insertion time (and hash-map iteration order) — nondeterministic. A unique
        // per-index tip makes the order a strict function of `i` (tip descending),
        // so the output bytes are reproducible regardless of timing/platform. `same`
        // and `recipients` keep the constant tip (single sender, nonce-ordered), so
        // the committed same-mode fixtures' checksums are unaffected.
        let (signer, nonce, recipient, priority_fee) = match mode {
            Mode::Same => (
                rich_signer()?,
                i,
                Address::from_low_u64_be(0xdead_beef),
                1_000_000_000u64,
            ),
            Mode::Recipients => (rich_signer()?, i, recipient_for(i), 1_000_000_000u64),
            Mode::Distinct => (
                deterministic_signer(i),
                0,
                recipient_for(i),
                1_000_000_000u64 + i,
            ),
        };
        let mut tx = Transaction::EIP1559Transaction(EIP1559Transaction {
            chain_id,
            nonce,
            max_priority_fee_per_gas: priority_fee,
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
        slot_number: Some(1),
        version: 4,
        elasticity_multiplier: ELASTICITY_MULTIPLIER,
        gas_ceil: 60_000_000,
    };
    let skeleton = create_payload(&payload_args, &store, Bytes::new())?;
    let result = blockchain.build_payload(skeleton)?;
    let block = result.payload;
    let block_access_list = result.block_access_list;
    let included = block.body.transactions.len();
    assert_eq!(
        included as u64, n_transfers,
        "only {included}/{n_transfers} transactions made it into the block \
         (check gas limit / account balance / nonces)"
    );

    // --- 4. stateless witness -> SSZ ---------------------------------------
    let witness = blockchain
        .generate_witness_for_blocks(std::slice::from_ref(&block))
        .await?;
    let witness: RpcExecutionWitness = witness.try_into()?;
    let bytes = build_stateless_input(&block, &witness, block_access_list.as_ref(), chain_id)?;
    let output = run_stateless_guest(&bytes, std::sync::Arc::new(NativeCrypto));
    if output.len() != 43 || output[32] == 0 {
        return Err("generated stateless fixture failed native validation".into());
    }
    std::fs::write(&out_path, &bytes)?;

    let mode_label = match mode {
        Mode::Same => "1 sender -> 1 recipient",
        Mode::Recipients => "1 sender -> N recipients",
        Mode::Distinct => "N senders -> N recipients",
    };
    println!(
        "wrote {out_path} ({} bytes): block #{} with {included}/{n_transfers} transfer(s) [{mode_label}]",
        bytes.len(),
        head_number + 1,
    );
    Ok(())
}
