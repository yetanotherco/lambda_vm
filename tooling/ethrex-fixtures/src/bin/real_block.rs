//! Rebuild a real mainnet block as an Amsterdam block the stateless guest accepts.
//!
//! The retired benchmark fixture (`ethrex_mainnet_25368371.bin`) is an rkyv
//! `ProgramInput` from before execution-specs #3278. The pinned guest only
//! decodes schema `0x1501` (Amsterdam) and mainnet has no Amsterdam fork yet, so
//! that block cannot be re-serialized: it has no block access list hash and no
//! slot number, and its state carries none of the EIP-8282 predeploys Amsterdam
//! requires. What CAN be reused is its workload — the real transactions and the
//! real accounts they touch.
//!
//! This takes an ethrex-replay cache, seeds a local Amsterdam genesis with the
//! accounts from the block's own execution witness, replays the block's
//! transactions on top, and emits the SSZ stateless input for the resulting
//! block. The output is NOT the mainnet block: the state root, block hash and
//! gas schedule are this chain's. It is the same transaction mix, which is the
//! part no synthetic block reproduces.
//!
//! Usage:
//!   cargo run --release --bin real_block -- <cache.json> <out.bin>
//!
//! NOTE: `build_stateless_input` and its helpers are copied from `main.rs`
//! rather than shared. A real PR should lift them into a module; keeping this
//! binary self-contained leaves the committed generator untouched.

use bytes::Bytes;
use ethrex_blockchain::payload::{BuildPayloadArgs, create_payload};
use ethrex_blockchain::{Blockchain, BlockchainOptions};
use ethrex_common::H256;
use ethrex_common::types::block_execution_witness::{
    ExecutionWitness, RpcExecutionWitness, amsterdam_chain_config, decode_witness_headers,
};
use ethrex_common::types::{AccountState, Block, ELASTICITY_MULTIPLIER, Genesis};
use ethrex_fixtures::build_stateless_input;
use ethrex_guest_program::crypto::{Crypto, NativeCrypto};
use ethrex_guest_program::l1::run_stateless_guest;
use ethrex_rlp::decode::RLPDecode;
use ethrex_rlp::encode::RLPEncode;
use ethrex_storage::{EngineType, Store};
use ethrex_trie::{Nibbles, Node, NodeRef};
use std::collections::BTreeMap;

const GENESIS_JSON: &str = include_str!("../../genesis.json");
/// Mainnet: the cached transactions are signed for it, so the local chain has to
/// claim the same id or every signature fails.
const CHAIN_ID: u64 = 1;
/// Low enough that every cached transaction clears the block's base fee. The
/// real block's was 0.119 gwei; a transaction rejected for fee reasons would
/// silently shrink the workload.
const GENESIS_BASE_FEE: u64 = 1_000;
const GAS_CEIL: u64 = 60_000_000;

/// Walk an embedded witness trie and return every `(32-byte key, value)` leaf it
/// still holds. Pruned subtrees appear as `NodeRef::Hash` and are skipped: they
/// are the proof siblings, which no transaction in the block reads.
///
/// Returns the number of leaves that could NOT be turned into a 32-byte key. In
/// a hashed trie there is no such thing, so a non-zero count means a malformed
/// witness node — and the caller must refuse to build on it. Dropping one
/// silently loses an account or a storage slot, the transaction that reads it
/// then sees zero and reverts, and the fixture ends up carrying less work than
/// the block it claims to reproduce with the revert count as its only symptom.
/// The installed-vs-collected counts cannot catch it: both are derived from this
/// function's output, so they agree by construction.
fn collect_leaves(node: &Node, path: Nibbles, out: &mut Vec<(H256, Vec<u8>)>) -> usize {
    let descend = |child: &NodeRef, path: Nibbles, out: &mut Vec<(H256, Vec<u8>)>| -> usize {
        if let NodeRef::Node(child, _) = child {
            collect_leaves(child, path, out)
        } else {
            0
        }
    };
    match node {
        Node::Branch(branch) => {
            let mut unusable = 0;
            for (i, child) in branch.choices.iter().enumerate() {
                unusable += descend(child, path.append_new(i as u8), out);
            }
            unusable
        }
        Node::Extension(ext) => descend(&ext.child, path.concat(&ext.prefix), out),
        Node::Leaf(leaf) => {
            let bytes = path.concat(&leaf.partial).to_bytes();
            if bytes.len() == 32 {
                out.push((H256::from_slice(&bytes), leaf.value.clone()));
                0
            } else {
                1
            }
        }
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args();
    let program = args.next().unwrap_or_else(|| "real_block".into());
    let (Some(cache_path), Some(out_path)) = (args.next(), args.next()) else {
        eprintln!("usage: {program} <cache.json> <out.bin>");
        std::process::exit(2);
    };

    // --- 1. cache -> real block + its execution witness --------------------
    let cache: serde_json::Value = serde_json::from_slice(&std::fs::read(&cache_path)?)?;
    let blocks = cache
        .get("blocks")
        .and_then(|b| b.as_array())
        .ok_or("cache has no `blocks` array")?;
    if blocks.len() != 1 {
        return Err(format!("cache holds {} blocks; expected exactly 1", blocks.len()).into());
    }
    let real_block: Block = serde_json::from_value(blocks[0].clone())?;
    let rpc_witness: RpcExecutionWitness = serde_json::from_value(
        cache
            .get("witness")
            .ok_or("cache has no `witness`")?
            .clone(),
    )?;

    let decoded_headers = decode_witness_headers(&rpc_witness.headers)
        .map_err(|e| format!("decode witness headers: {e:?}"))?;
    let witness: ExecutionWitness = rpc_witness
        .clone()
        .into_execution_witness(
            amsterdam_chain_config(CHAIN_ID),
            real_block.header.number,
            &decoded_headers,
            &NativeCrypto,
        )
        .map_err(|e| format!("into_execution_witness: {e:?}"))?;

    let codes_by_hash: std::collections::HashMap<H256, Vec<u8>> = witness
        .codes
        .iter()
        .map(|code| (H256(NativeCrypto.keccak256(code)), code.clone()))
        .collect();
    // --- 2. a local Amsterdam chain to host the block ----------------------
    // The committed genesis carries the system contracts and the two EIP-8282
    // predeploys Amsterdam needs; only the chain id and fee floor change. The
    // block's own accounts do NOT go in the alloc — they are installed into the
    // tries below, which needs no key preimages.
    let mut genesis: Genesis = serde_json::from_str(GENESIS_JSON)?;
    let blob_schedule = genesis.config.blob_schedule;
    genesis.config = amsterdam_chain_config(CHAIN_ID);
    genesis.config.blob_schedule = blob_schedule;
    // `REAL_BLOCK_FORK=osaka` executes the same block under the rules it was
    // built for. The guest cannot consume the result (it only decodes the
    // Amsterdam schema), so this exists purely to attribute a gas difference to
    // the fork rather than to this tool's reconstruction.
    let osaka_probe = std::env::var("REAL_BLOCK_FORK").as_deref() == Ok("osaka");
    if osaka_probe {
        genesis.config.amsterdam_time = None;
    }
    genesis.gas_limit = GAS_CEIL;
    genesis.base_fee_per_gas = Some(GENESIS_BASE_FEE);
    genesis.timestamp = real_block.header.timestamp.saturating_sub(12);

    let mut accounts = Vec::new();
    let mut unusable_leaves = 0usize;
    if let Some(root) = &witness.state_trie_root {
        unusable_leaves += collect_leaves(root, Nibbles::default(), &mut accounts);
    }

    let mut store = Store::new(".ethrex-real-block-tmp", EngineType::InMemory)?;
    store.add_initial_state(genesis).await?;
    let genesis_header = store
        .get_block_header(store.get_latest_block_number()?)?
        .ok_or("missing genesis header")?;

    // Install the block's own pre-state into the store's tries, keyed the way the
    // tries themselves are keyed: `keccak(address)` and `keccak(slot)`. That is
    // what makes this faithful — a `Genesis.alloc` is keyed by the preimages, and
    // the witness cannot give those back (they are hashes; the replay cache's
    // `keys` field carries only the few the RPC returned, and upstream is removing
    // it). Writing the hashed keys directly needs no preimages at all, so every
    // account and every storage slot the block touches arrives intact.
    let genesis_root = genesis_header.state_root;
    // The layered opener's `put_batch` is `unimplemented!()` — the writable path is
    // the direct one, which talks to the storage backend without the reorg overlay.
    let mut state_trie = store.open_direct_state_trie(genesis_root)?;
    let mut installed_accounts = 0usize;
    let mut installed_slots = 0usize;
    let mut installed_codes = 0usize;
    for (hashed_address, encoded) in &accounts {
        let mut account = AccountState::decode(encoded)
            .map_err(|e| format!("decode account {hashed_address:#x}: {e:?}"))?;
        if let Some(storage_root) = witness.storage_trie_roots.get(hashed_address) {
            let mut leaves = Vec::new();
            unusable_leaves += collect_leaves(storage_root, Nibbles::default(), &mut leaves);
            let mut storage_trie = store.open_direct_storage_trie(
                *hashed_address,
                *ethrex_common::constants::EMPTY_TRIE_HASH,
            )?;
            for (hashed_slot, value) in &leaves {
                storage_trie.insert(hashed_slot.0.to_vec(), value.clone())?;
            }
            installed_slots += leaves.len();
            account.storage_root = storage_trie.hash(&NativeCrypto)?;
        }
        if let Some(code) = codes_by_hash.get(&account.code_hash) {
            store
                .add_account_code(ethrex_common::types::Code::from_bytecode(
                    Bytes::from(code.clone()),
                    &NativeCrypto,
                ))
                .await?;
            installed_codes += 1;
        }
        state_trie.insert(hashed_address.0.to_vec(), account.encode_to_vec())?;
        installed_accounts += 1;
    }
    let installed_root = state_trie.hash(&NativeCrypto)?;
    if unusable_leaves > 0 {
        return Err(format!(
            "{unusable_leaves} witness leaf/leaves did not yield a 32-byte trie key, so \
             that much of the block's pre-state was never installed. The transactions \
             reading it would see zero and revert, shrinking the workload silently."
        )
        .into());
    }

    // Build at the real height, not at #1. The seeded accounts hold the block's
    // own pre-state, but a contract that stores a block number and compares it
    // against `block.number` reads nonsense on a chain that just started — and
    // several of the block's transactions do exactly that. So register a parent
    // header at the real parent's height carrying the genesis state root (which
    // is the seeded state), and build on top of it. The guest only ever sees
    // this parent and the block itself, and those two are contiguous.
    let mut head = genesis_header.clone();
    head.hash = Default::default();
    head.number = real_block.header.number - 1;
    head.parent_hash = genesis_header.hash();
    head.timestamp = real_block.header.timestamp.saturating_sub(12);
    head.gas_limit = GAS_CEIL;
    head.base_fee_per_gas = Some(GENESIS_BASE_FEE);
    head.state_root = installed_root;
    let head_hash = head.hash();
    let head_number = head.number;
    store.add_block_header(head_hash, head.clone()).await?;
    store
        .add_block_body(head_hash, ethrex_common::types::BlockBody::empty())
        .await?;
    store.add_block_number(head_hash, head_number).await?;
    store
        .forkchoice_update(
            vec![(head_number, head_hash)],
            head_number,
            head_hash,
            None,
            None,
        )
        .await?;
    // Real blocks carry builder/MEV transactions with a zero tip cap, which the
    // default mempool floor rejects. Dropping them would silently shrink the
    // workload this fixture exists to reproduce.
    let blockchain = Blockchain::new(
        store.clone(),
        BlockchainOptions {
            min_tip_wei: 0,
            ..BlockchainOptions::default()
        },
    );

    // --- 3. replay the real transactions, in the block's own order ----------
    // Not through the mempool: `build_payload` would re-sort by effective tip and
    // an arbitrage transaction that ran second on mainnet reverts when it runs
    // first. The t8n entry point takes an explicit, ordered list, reports what it
    // could not apply instead of aborting, and seeds BLOCKHASH with the ancestor
    // hashes the witness carries.
    let total_txs = real_block.body.transactions.len();
    let payload_args = BuildPayloadArgs {
        parent: head_hash,
        timestamp: head.timestamp + 12,
        fee_recipient: real_block.header.coinbase,
        random: real_block.header.prev_randao,
        withdrawals: Some(vec![]),
        beacon_root: Some(H256::zero()),
        slot_number: Some(1),
        version: 4,
        elasticity_multiplier: ELASTICITY_MULTIPLIER,
        gas_ceil: GAS_CEIL,
    };
    let skeleton = create_payload(&payload_args, &store, Bytes::new())?;
    let mut block_hash_cache = BTreeMap::new();
    block_hash_cache.insert(head_number, head_hash);
    let (result, rejected, t8n_error) = blockchain.build_payload_t8n(
        skeleton,
        real_block.body.transactions.clone(),
        block_hash_cache,
        false,
    )?;
    if let Some(error) = t8n_error {
        return Err(format!("payload build reported: {error}").into());
    }

    // Amsterdam always emits one `EncodedRequests` per request type (a lone type
    // byte when the list is empty), so the count is not the signal — a non-empty
    // payload is. The SSZ input below declares no requests, which only matches
    // the header's `requests_hash` while every list is in fact empty.
    let non_empty_requests = result
        .requests
        .iter()
        .filter(|encoded| !encoded.is_empty())
        .count();
    if non_empty_requests > 0 {
        return Err(format!(
            "the rebuilt block produced {non_empty_requests} non-empty EIP-7685 request \
             list(s); the SSZ input this tool writes declares none"
        )
        .into());
    }
    if osaka_probe {
        let reverted = result.receipts.iter().filter(|r| !r.succeeded).count();
        println!(
            "osaka probe #{} ({}/{} txs, {} gas, {reverted} reverted) vs real {} gas",
            result.payload.header.number,
            result.payload.body.transactions.len(),
            total_txs,
            result.payload.header.gas_used,
            real_block.header.gas_used,
        );
        return Ok(());
    }

    let block = result.payload;
    let included = block.body.transactions.len();
    // The tx mix is the whole reason this fixture exists, so a builder that applied
    // fewer than all of them has produced a smaller workload than the block it claims
    // to reproduce -- and every guard below still passes, because a block with fewer
    // transactions is a perfectly valid block. The screen sets the escape: surveying
    // candidate blocks needs the partial ones reported, not refused.
    if included != total_txs && std::env::var_os("REAL_BLOCK_ALLOW_DROPS").is_none() {
        return Err(format!(
            "the payload builder applied only {included} of {total_txs} transactions; \
             rejected: {rejected:?}. Set REAL_BLOCK_ALLOW_DROPS=1 to write the fixture \
             anyway."
        )
        .into());
    }

    // --- 4. witness -> SSZ -> native validation ----------------------------
    let witness = blockchain
        .generate_witness_for_blocks(std::slice::from_ref(&block))
        .await?;
    let witness: RpcExecutionWitness = witness.try_into()?;
    let bytes = build_stateless_input(
        &block,
        &witness,
        result.block_access_list.as_ref(),
        CHAIN_ID,
    )?;
    let output = run_stateless_guest(&bytes, std::sync::Arc::new(NativeCrypto));
    if output.len() != 43 || output[32] == 0 {
        return Err("rebuilt block failed native stateless validation".into());
    }
    std::fs::write(&out_path, &bytes)?;

    // --- 5. report ---------------------------------------------------------
    let reverted = result.receipts.iter().filter(|r| !r.succeeded).count();
    let mut per_tx = String::new();
    let mut prev_cumulative = 0u64;
    for (i, receipt) in result.receipts.iter().enumerate() {
        let gas = receipt.cumulative_gas_used.saturating_sub(prev_cumulative);
        prev_cumulative = receipt.cumulative_gas_used;
        let to = match block.body.transactions.get(i).map(|tx| tx.to()) {
            Some(ethrex_common::types::TxKind::Call(address)) => format!("{address:#x}"),
            Some(ethrex_common::types::TxKind::Create) => "create".to_string(),
            None => "?".to_string(),
        };
        per_tx.push_str(&format!(
            "  tx {i:>2}: {:>9} gas  {:<8} -> {to}\n",
            gas,
            if receipt.succeeded { "ok" } else { "REVERTED" }
        ));
    }
    println!(
        "real block  #{} ({} txs, {} gas)",
        real_block.header.number, total_txs, real_block.header.gas_used
    );
    println!(
        "witness     {} account leaves / {} codes / {} trie nodes",
        accounts.len(),
        rpc_witness.codes.len(),
        rpc_witness.state.len()
    );
    println!(
        "installed   {installed_accounts} accounts / {installed_slots} storage slots / \
         {installed_codes} codes, state root {installed_root:#x}"
    );
    println!(
        "rebuilt     #{} ({included}/{total_txs} txs, {} gas, {reverted} reverted)",
        block.header.number, block.header.gas_used
    );
    print!("{per_tx}");
    if !rejected.is_empty() {
        println!("rejected ({} of {total_txs}):", rejected.len());
        for entry in &rejected {
            println!("  {entry:?}");
        }
    }
    println!(
        "gas vs real {:+.2}%",
        (block.header.gas_used as f64 / real_block.header.gas_used as f64 - 1.0) * 100.0
    );
    println!("wrote       {out_path} ({} bytes)", bytes.len());
    Ok(())
}
