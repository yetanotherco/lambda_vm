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
//! This takes an ethrex-replay cache, installs the block's own execution witness
//! tries as the pre-state of a local Amsterdam chain, replays the block's
//! transactions on top, and emits the SSZ stateless input for the resulting
//! block. The witness tries go in as they are, pruned siblings included, so the
//! pre-state keeps mainnet's depth and the guest hashes the same paths it would
//! on the real block. The output is NOT the mainnet block: the state root, block
//! hash and gas schedule are this chain's. It is the same transaction mix, which
//! is the part no synthetic block reproduces.
//!
//! Usage:
//!   cargo run --release --bin real_block -- <cache.json> <out.bin>

use bytes::Bytes;
use ethrex_blockchain::payload::{BuildPayloadArgs, create_payload};
use ethrex_blockchain::{Blockchain, BlockchainOptions};
use ethrex_common::H256;
use ethrex_common::types::block_execution_witness::{
    ExecutionWitness, RpcExecutionWitness, amsterdam_chain_config, decode_witness_headers,
};
use ethrex_common::types::{AccountState, Block, ELASTICITY_MULTIPLIER, Genesis};
use ethrex_guest_program::crypto::{Crypto, NativeCrypto};
use ethrex_rlp::decode::RLPDecode;
use ethrex_rlp::encode::RLPEncode;
use ethrex_ssz_input::{build_stateless_input, validate_natively};
use ethrex_storage::{EngineType, Store};
use ethrex_trie::node::{BranchNode, ExtensionNode, LeafNode};
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
            // A branch carries a value only when some key ends at it, which cannot happen
            // in these tries: every key is a 32-byte hash, so no key is a prefix of
            // another. Counted rather than ignored, because "cannot happen" is exactly
            // what this function reports instead of asserting.
            let mut unusable = usize::from(!branch.value.is_empty());
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

/// An account the witness does not carry, to be added to the installed state trie.
struct Graft {
    /// Nibbles of the hashed key, with the trailing leaf flag (16).
    key: Vec<u8>,
    /// The hashed key itself, for a plain `Trie::insert`.
    raw: Vec<u8>,
    value: Vec<u8>,
    grafted: bool,
}

/// Copy an embedded witness node with fresh hash caches, so `Trie::commit` writes
/// every node instead of skipping the ones whose hash is already memoized.
///
/// A pruned child stays a `NodeRef::Hash`: it is a proof sibling no transaction in
/// the block reads. The one exception is a pending graft whose path runs into it —
/// inserting there would need the pruned subtree, which the witness does not have,
/// so the graft replaces it with a leaf. That changes the state root but not the
/// depth of anything the block touches. `path` is the nibble path to `node`.
fn fresh(node: &Node, path: &mut Vec<u8>, grafts: &mut Vec<Graft>) -> Node {
    match node {
        Node::Branch(branch) => {
            let mut choices = BranchNode::EMPTY_CHOICES;
            for (i, child) in branch.choices.iter().enumerate() {
                path.push(i as u8);
                choices[i] = match child {
                    NodeRef::Node(child, _) => fresh(child, path, grafts).into(),
                    NodeRef::Hash(h) if child.is_valid() => {
                        let hit = grafts
                            .iter()
                            .position(|g| !g.grafted && g.key.starts_with(path));
                        match hit {
                            Some(j) => {
                                let g = &mut grafts[j];
                                g.grafted = true;
                                Node::Leaf(LeafNode {
                                    partial: Nibbles::from_hex(g.key[path.len()..].to_vec()),
                                    value: g.value.clone(),
                                })
                                .into()
                            }
                            None => NodeRef::Hash(*h),
                        }
                    }
                    NodeRef::Hash(h) => NodeRef::Hash(*h),
                };
                path.pop();
            }
            Node::Branch(Box::new(BranchNode::new_with_value(
                choices,
                branch.value.clone(),
            )))
        }
        Node::Extension(ext) => {
            let n = ext.prefix.len();
            path.extend_from_slice(ext.prefix.as_ref());
            let child = match &ext.child {
                NodeRef::Node(c, _) => fresh(c, path, grafts).into(),
                NodeRef::Hash(h) => NodeRef::Hash(*h),
            };
            path.truncate(path.len() - n);
            Node::Extension(ExtensionNode {
                prefix: ext.prefix.clone(),
                child,
            })
        }
        Node::Leaf(leaf) => Node::Leaf(leaf.clone()),
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
    // predeploys Amsterdam needs; its fork schedule is replaced wholesale by the
    // one the guest itself uses, so the two cannot drift. That includes the blob
    // schedule: `amsterdam_chain_config` activates BPO1/BPO2, and EIP-7892 has
    // Amsterdam inherit the highest activated BPO entry (target 14 / max 21),
    // which is what `get_fork_blob_schedule` resolves. Carrying genesis.json's
    // own `blobSchedule` over it would be a no-op — its two entries are the
    // serde defaults — so it is not carried over at all. The block's accounts do
    // NOT go in the alloc either; they are installed into the tries below, which
    // needs no key preimages.
    let mut genesis: Genesis = serde_json::from_str(GENESIS_JSON)?;
    genesis.config = amsterdam_chain_config(CHAIN_ID);
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

    let predeploys: Vec<(ethrex_common::Address, ethrex_common::types::GenesisAccount)> = genesis
        .alloc
        .iter()
        .filter(|(a, _)| format!("{a:#x}").ends_with("8282"))
        .map(|(a, g)| (*a, g.clone()))
        .collect();
    if predeploys.len() != 2 {
        return Err(format!(
            "expected 2 EIP-8282 predeploys in genesis, found {}",
            predeploys.len()
        )
        .into());
    }
    let mut store = Store::new(".ethrex-real-block-tmp", EngineType::InMemory)?;
    store.add_initial_state(genesis).await?;
    let genesis_header = store
        .get_block_header(store.get_latest_block_number()?)?
        .ok_or("missing genesis header")?;

    // Install the block's own pre-state: the witness tries AS THEY ARE, every node
    // written at its path and every pruned sibling left as a hash. Re-inserting the
    // witness leaves into a fresh trie would also be faithful key by key -- the
    // tries are keyed by `keccak(address)`/`keccak(slot)`, so no preimages are
    // needed either way -- but a trie holding only this block's ~100 accounts is 3-4
    // levels deep where mainnet's is ~9, and the guest would hash a fraction of the
    // nodes it hashes on the real block. The installed state trie must hash to the
    // real parent's state root; that is checked below, before anything is added.
    let parent_real_root = decoded_headers
        .iter()
        .find(|h| h.number + 1 == real_block.header.number)
        .map(|h| h.state_root)
        .ok_or("witness carries no parent header")?;
    let mut installed_accounts = accounts.len();
    let mut installed_slots = 0usize;
    let mut installed_codes = 0usize;
    // Two things the witness can leave out. Neither is an error here -- an account whose
    // storage or code this block never touches is simply absent from the witness, and
    // installing it is not this tool's job -- but an account that DECLARES either and
    // does not get it is installed pointing at nodes the local store does not have. That
    // surfaces downstream as a missing-node or missing-code failure if the block reaches
    // it, and as nothing at all if it does not, so the counts are reported.
    let empty_code_hash = H256(NativeCrypto.keccak256(&[]));
    let mut declared_storage_missing = 0usize;
    let mut declared_code_missing = 0usize;
    for (hashed_address, encoded) in &accounts {
        let account = AccountState::decode(encoded)
            .map_err(|e| format!("decode account {hashed_address:#x}: {e:?}"))?;
        if let Some(storage_root) = witness.storage_trie_roots.get(hashed_address) {
            let mut leaves = Vec::new();
            unusable_leaves += collect_leaves(storage_root, Nibbles::default(), &mut leaves);
            installed_slots += leaves.len();
            let mut storage_trie = store.open_direct_storage_trie(
                *hashed_address,
                *ethrex_common::constants::EMPTY_TRIE_HASH,
            )?;
            storage_trie.root = fresh(storage_root, &mut Vec::new(), &mut Vec::new()).into();
            let got = storage_trie.hash(&NativeCrypto)?;
            if got != account.storage_root {
                return Err(format!(
                    "storage trie of {hashed_address:#x} installed with root {got:#x}, \
                     account declares {:#x}",
                    account.storage_root
                )
                .into());
            }
        } else if account.storage_root != *ethrex_common::constants::EMPTY_TRIE_HASH {
            declared_storage_missing += 1;
        }
        if let Some(code) = codes_by_hash.get(&account.code_hash) {
            store
                .add_account_code(ethrex_common::types::Code::from_bytecode(
                    Bytes::from(code.clone()),
                    &NativeCrypto,
                ))
                .await?;
            installed_codes += 1;
        } else if account.code_hash != empty_code_hash {
            declared_code_missing += 1;
        }
    }
    let state_root_node = witness
        .state_trie_root
        .as_ref()
        .ok_or("witness has no state trie")?;
    let sanity = ethrex_trie::Trie::new_temp_with_root(
        fresh(state_root_node, &mut Vec::new(), &mut Vec::new()).into(),
    )
    .hash_no_commit(&NativeCrypto);
    if sanity != parent_real_root {
        return Err(format!(
            "witness state trie hashes to {sanity:#x}, real parent root {parent_real_root:#x}"
        )
        .into());
    }
    // Amsterdam requires the two EIP-8282 predeploys to carry code, and mainnet has
    // neither. They are the only accounts added, which is why the rebuilt parent
    // cannot be the real one: its state root differs by exactly these two.
    let mut grafts: Vec<Graft> = Vec::new();
    for (address, acct) in &predeploys {
        let code = ethrex_common::types::Code::from_bytecode(acct.code.clone(), &NativeCrypto);
        let state = AccountState {
            nonce: acct.nonce,
            balance: acct.balance,
            storage_root: *ethrex_common::constants::EMPTY_TRIE_HASH,
            code_hash: code.hash,
        };
        store.add_account_code(code).await?;
        let hashed = NativeCrypto.keccak256(address.as_bytes());
        if accounts.iter().any(|(h, _)| h.0 == hashed) {
            return Err(format!("predeploy {address:#x} already in the mainnet witness").into());
        }
        grafts.push(Graft {
            key: Nibbles::from_bytes(&hashed).into_vec(),
            raw: hashed.to_vec(),
            value: state.encode_to_vec(),
            grafted: false,
        });
        installed_accounts += 1;
    }
    // The layered opener's `put_batch` is `unimplemented!()` — the writable path is
    // the direct one, which talks to the storage backend without the reorg overlay.
    let mut state_trie =
        store.open_direct_state_trie(*ethrex_common::constants::EMPTY_TRIE_HASH)?;
    state_trie.root = fresh(state_root_node, &mut Vec::new(), &mut grafts).into();
    for g in grafts.iter().filter(|g| !g.grafted) {
        state_trie.insert(g.raw.clone(), g.value.clone())?;
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

    // Build at the real height, not at #1. The installed tries hold the block's
    // own pre-state, but a contract that stores a block number and compares it
    // against `block.number` reads nonsense on a chain that just started — and
    // several of the block's transactions do exactly that. So register a parent
    // header at the real parent's height carrying the installed state root, and
    // build on top of it. The guest only ever sees this parent and the block
    // itself, and those two are contiguous.
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
        // The real one, not zero: the EIP-4788 system call stores it in the ring
        // buffer, and storing zero DELETES the slot mainnet overwrote. Deleting
        // collapses a branch whose sibling the witness only carries as a hash.
        beacon_root: real_block.header.parent_beacon_block_root,
        slot_number: Some(1),
        version: 4,
        elasticity_multiplier: ELASTICITY_MULTIPLIER,
        gas_ceil: GAS_CEIL,
    };
    let skeleton = create_payload(&payload_args, &store, Bytes::new())?;
    // Every ancestor the witness carries, not just the parent: a transaction reading
    // BLOCKHASH at one of those depths would otherwise see zero where mainnet gave it a
    // hash, and the fixture would quietly execute something else. The local head
    // overwrites the real ancestor at its own height, because that is the block this one
    // builds on.
    //
    // The reach is whatever the cache's `headers` field carries, NOT the full 255 depths
    // EIP-2935 allows: this cache feeds `build_payload_t8n` here, and the witness the
    // guest gets is regenerated separately below. A read deeper than that sees zero on
    // both sides, so it cannot make the guest and the native reference disagree -- it
    // would just be a block whose BLOCKHASH behaviour is not mainnet's. Repointing to a
    // block whose transactions reach further needs that checked, not assumed.
    let mut block_hash_cache: BTreeMap<u64, H256> = decoded_headers
        .iter()
        .map(|header| (header.number, header.compute_block_hash(&NativeCrypto)))
        .collect();
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
    // A transaction that under Amsterdam reads state mainnet's execution never
    // touched hits a pruned node and is rejected with a DB error. Refused rather
    // than tolerated: levm keeps the nonce bump and gas charge of a transaction
    // that fails after `prepare_execution`, so the built block's state root
    // silently includes a transaction the block does not, and witness generation
    // fails later with a bare StateRootMismatch.
    let missing_state: Vec<_> = rejected
        .iter()
        .filter(|r| r.error.contains("Database access error"))
        .collect();
    if !missing_state.is_empty() {
        return Err(format!(
            "{} transaction(s) read state the block's witness does not carry: \
             {missing_state:?}. This block cannot be rebuilt on its witness trie.",
            missing_state.len()
        )
        .into());
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
    // fewer than all of them has produced a smaller block than the one it claims to
    // reproduce -- and every guard below still passes, because a block with fewer
    // transactions is a perfectly valid block. The screen sets the escape: surveying
    // candidate blocks needs the partial ones reported, not refused.
    //
    // This counts INCLUSION, not work. A transaction that runs out of gas is included
    // with a failed receipt and passes here while executing a fraction of its opcodes,
    // and its gas is charged in full either way, so neither this count nor gas_used can
    // stand in for the workload. What bounds that is the cycle floor the benchmark
    // entry points apply (scripts/assert_workload_cycles.sh).
    if included != total_txs && std::env::var_os("REAL_BLOCK_ALLOW_DROPS").is_none() {
        return Err(format!(
            "the payload builder applied only {included} of {total_txs} transactions; \
             rejected: {rejected:?}. Set REAL_BLOCK_ALLOW_DROPS=1 to write the fixture \
             anyway."
        )
        .into());
    }
    if included != total_txs {
        // The bytes carry no record of this: a fixture written with drops is
        // indistinguishable from one whose block really had that many transactions, and
        // the Makefile's sha256 would happily pin it. Say so where the operator sees it.
        eprintln!(
            "WARNING: REAL_BLOCK_ALLOW_DROPS is set and {} of {total_txs} transactions \
             were dropped. The output records nothing about that -- use it to screen a \
             candidate block, do NOT pin it as the benchmark fixture.",
            total_txs - included
        );
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
    validate_natively(&bytes)
        .map_err(|e| format!("rebuilt block failed native stateless validation: {e}"))?;
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
    if declared_storage_missing > 0 || declared_code_missing > 0 {
        println!(
            "note        {declared_storage_missing} account(s) declare storage and \
             {declared_code_missing} declare code the witness did not carry; they are \
             installed pointing at nodes this store does not have. Harmless unless the \
             block reaches them, in which case execution fails below rather than here."
        );
    }
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
