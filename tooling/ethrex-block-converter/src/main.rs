//! Convert an `ethrex-replay` cache into the SSZ stateless input consumed by
//! the lambda-vm ethrex guest.
//!
//! Usage:
//!   cargo run --release -- <cache.json> <output_path>

use ethrex_common::constants::{DEFAULT_REQUESTS_HASH, EMPTY_BLOCK_ACCESS_LIST_HASH};
use ethrex_common::types::Block;
use ethrex_common::types::block_access_list::BlockAccessList;
use ethrex_common::types::block_execution_witness::RpcExecutionWitness;
use ethrex_common::types::stateless_ssz::{
    Bytes20, ExecutionPayload, ExecutionRequests, LogsBloom, NewPayloadRequest,
    STATELESS_INPUT_SCHEMA_ID, SszExecutionWitness, SszPublicKeys, SszStatelessInput,
};
use ethrex_config::networks::Network;
use ethrex_crypto::NativeCrypto;
use ethrex_guest_program::l1::run_stateless_guest;
use libssz::SszEncode;
use libssz_types::{ProgressiveList, SszList, SszVector};
use serde::Deserialize;

/// The subset of an `ethrex-replay` cache needed by the stateless SSZ input.
///
/// Current replay caches carry the witness as raw RLP preimages. A future cache
/// may carry the Amsterdam block access list beside the block; accepting it here
/// keeps the converter independent of replay's Rust type layout.
///
/// Deliberately deserialized with *our* pinned ethrex types rather than by
/// depending on `ethrex-replay`: it tracks ethrex `main` while we pin a commit of
/// it, and the input type has diverged between the two before. This JSON carries
/// only `blocks` + `witness` + `network` as plain serde, so it stays the
/// version-tolerant interface between them. Extra fields in the file (L2 blob
/// data, custom `chain_config`) are ignored.
#[derive(Deserialize)]
struct Cache {
    blocks: Vec<Block>,
    witness: RpcExecutionWitness,
    network: Network,
    #[serde(default, alias = "blockAccessList")]
    block_access_list: Option<BlockAccessList>,
}

struct BlockSummary {
    network: String,
    first_block_number: u64,
    blocks: usize,
    transactions: usize,
    gas_used: u64,
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

fn ssz_witness(
    witness: &RpcExecutionWitness,
) -> Result<SszExecutionWitness, Box<dyn std::error::Error>> {
    let state = witness
        .state
        .iter()
        .enumerate()
        .map(|(i, bytes)| {
            SszList::try_from(bytes.to_vec())
                .map_err(|e| format!("witness state[{i}] is too large: {e:?}"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let codes = witness
        .codes
        .iter()
        .enumerate()
        .map(|(i, bytes)| {
            SszList::try_from(bytes.to_vec())
                .map_err(|e| format!("witness codes[{i}] is too large: {e:?}"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let headers = witness
        .headers
        .iter()
        .enumerate()
        .map(|(i, bytes)| {
            SszList::try_from(bytes.to_vec())
                .map_err(|e| format!("witness headers[{i}] is too large: {e:?}"))
        })
        .collect::<Result<Vec<_>, _>>()?;

    Ok(SszExecutionWitness {
        state: ProgressiveList::from(state),
        codes: ProgressiveList::from(codes),
        headers: SszList::try_from(headers)
            .map_err(|e| format!("witness has too many headers: {e:?}"))?,
    })
}

fn stateless_input_from_cache(
    cache_path: &str,
) -> Result<(Vec<u8>, BlockSummary), Box<dyn std::error::Error>> {
    let cache: Cache =
        serde_json::from_reader(std::io::BufReader::new(std::fs::File::open(cache_path)?))?;

    let Some(block) = cache.blocks.first() else {
        return Err("cache contains no blocks".into());
    };
    if cache.blocks.len() != 1 {
        return Err(format!(
            "cache contains {} blocks; the ethrex 25 stateless guest accepts one block",
            cache.blocks.len()
        )
        .into());
    }
    if !matches!(cache.network, Network::PublicNetwork(_)) {
        return Err(format!(
            "unsupported network `{}`: its chain rules would be guessed, not read",
            cache.network
        )
        .into());
    }

    let summary = BlockSummary {
        network: cache.network.to_string(),
        first_block_number: block.header.number,
        blocks: 1,
        transactions: block.body.transactions.len(),
        gas_used: block.header.gas_used,
    };
    let header = &block.header;

    // The new wire format is the Amsterdam payload. A pre-Amsterdam replay
    // cache has no BAL or slot and cannot be upgraded without replaying the
    // block, so reject it instead of changing its block hash or chain rules.
    let block_access_list_hash = header
        .block_access_list_hash
        .ok_or("cache block has no Amsterdam block_access_list_hash")?;
    let slot_number = header
        .slot_number
        .ok_or("cache block has no Amsterdam slot_number")?;
    let block_access_list = match cache.block_access_list.as_ref() {
        Some(bal) => bal.clone(),
        None if block_access_list_hash == *EMPTY_BLOCK_ACCESS_LIST_HASH => BlockAccessList::new(),
        None => {
            return Err("cache block has a non-empty BAL hash but no raw block access list".into());
        }
    };
    if block_access_list.compute_hash(&NativeCrypto) != block_access_list_hash {
        return Err("cache block access list does not match block_access_list_hash".into());
    }

    let requests_hash = header
        .requests_hash
        .ok_or("cache block has no requests_hash")?;
    if requests_hash != *DEFAULT_REQUESTS_HASH {
        return Err(
            "cache contains execution requests, but the replay cache has no request data".into(),
        );
    }
    let base_fee = header
        .base_fee_per_gas
        .ok_or("cache block has no base_fee_per_gas")?;
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
    let logs_bloom: LogsBloom = SszVector::try_from(header.logs_bloom.0.to_vec())
        .map_err(|e| format!("logs bloom is not 256 bytes: {e:?}"))?;
    let extra_data = SszList::try_from(header.extra_data.to_vec())
        .map_err(|e| format!("extra data exceeds 32 bytes: {e:?}"))?;
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
        block_access_list: ethrex_rlp::encode::RLPEncode::encode_to_vec(&block_access_list).into(),
        slot_number,
    };
    let new_payload_request = NewPayloadRequest {
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
    };
    let input = SszStatelessInput {
        new_payload_request,
        witness: ssz_witness(&cache.witness)?,
        chain_id: cache.network.get_genesis()?.config.chain_id,
        public_keys: SszPublicKeys::from(public_keys),
    };

    let mut bytes = STATELESS_INPUT_SCHEMA_ID.to_be_bytes().to_vec();
    input.ssz_append(&mut bytes);
    Ok((bytes, summary))
}

fn usage_and_exit(program: &str) -> ! {
    eprintln!("usage: {program} <cache.json> <output_path>");
    std::process::exit(2);
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args();
    let program = args
        .next()
        .unwrap_or_else(|| "ethrex-block-converter".into());
    let (Some(cache_path), Some(out_path)) = (args.next(), args.next()) else {
        usage_and_exit(&program);
    };
    if args.next().is_some() {
        usage_and_exit(&program);
    }

    let (bytes, summary) = stateless_input_from_cache(&cache_path)?;
    let output = run_stateless_guest(&bytes, std::sync::Arc::new(NativeCrypto));
    if output.len() != 43 || output[32] == 0 {
        return Err("converted stateless input failed native validation".into());
    }
    std::fs::write(&out_path, &bytes)?;

    println!(
        "wrote {out_path} ({} bytes): {} block(s) from {} starting at #{}, {} transaction(s), {} gas",
        bytes.len(),
        summary.blocks,
        summary.network,
        summary.first_block_number,
        summary.transactions,
        summary.gas_used,
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const CACHE: &str = "caches/cache_hoodi_1265656.json";
    const CACHE_MISSING: &str = "caches/cache_hoodi_1265656.json is missing — run `make ethrex-real-block-converter-cache` first";

    #[test]
    fn legacy_cache_is_rejected_without_amsterdam_fields() {
        let result = stateless_input_from_cache(CACHE);
        let Err(error) = result else {
            panic!("the checked-in replay cache unexpectedly has Amsterdam fields");
        };
        if !std::path::Path::new(CACHE).exists() {
            assert!(
                error.to_string().contains("No such file")
                    || error.to_string().contains("os error"),
                "{CACHE_MISSING}: {error}"
            );
        } else {
            assert!(
                error.to_string().contains("Amsterdam"),
                "unexpected error: {error}"
            );
        }
    }

    #[test]
    fn unmappable_network_is_rejected() {
        let mut cache: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(CACHE).expect(CACHE_MISSING)).unwrap();
        cache["network"] = serde_json::json!("LocalDevnet");
        let path =
            std::env::temp_dir().join(format!("ethrex_localdevnet_{}.json", std::process::id()));
        std::fs::write(&path, serde_json::to_vec(&cache).unwrap()).unwrap();
        let result = stateless_input_from_cache(path.to_str().unwrap());
        std::fs::remove_file(&path).ok();
        let Err(error) = result else {
            panic!("LocalDevnet must be rejected");
        };
        assert!(error.to_string().contains("unsupported network"), "{error}");
    }
}
