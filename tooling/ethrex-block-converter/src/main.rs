//! Convert an `ethrex-replay` cache into the SSZ stateless input consumed by
//! the lambda-vm ethrex guest.
//!
//! Usage:
//!   cargo run --release -- <cache.json> <output_path>

use ethrex_common::constants::{DEFAULT_REQUESTS_HASH, EMPTY_BLOCK_ACCESS_LIST_HASH};
use ethrex_common::types::Block;
use ethrex_common::types::block_access_list::BlockAccessList;
use ethrex_common::types::block_execution_witness::RpcExecutionWitness;
use ethrex_config::networks::Network;
use ethrex_crypto::NativeCrypto;
use ethrex_ssz_input::{build_stateless_input, validate_natively};
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
            "cache contains {} blocks; the pinned stateless guest accepts one block",
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
    if header.slot_number.is_none() {
        return Err("cache block has no Amsterdam slot_number".into());
    }
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
    if header.base_fee_per_gas.is_none() {
        return Err("cache block has no base_fee_per_gas".into());
    }

    // Everything above sources the block from the cache and rejects what the pinned schema
    // cannot express. The encoding itself is `ethrex-ssz-input`, shared with the fixture
    // generators, so an ethrex rev bump moves one copy of it instead of three.
    let chain_id = cache.network.get_genesis()?.config.chain_id;
    let bytes = build_stateless_input(block, &cache.witness, Some(&block_access_list), chain_id)?;
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
    validate_natively(&bytes)
        .map_err(|e| format!("converted stateless input failed native validation: {e}"))?;
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
        // Fail on the missing cache instead of accepting its I/O error as the rejection:
        // every error reads as a pass otherwise, and this test would go green on a clean
        // checkout having asserted nothing. `unmappable_network_is_rejected` reports the
        // same condition the same way.
        assert!(std::path::Path::new(CACHE).exists(), "{CACHE_MISSING}");
        let Err(error) = stateless_input_from_cache(CACHE) else {
            panic!("the checked-in replay cache unexpectedly has Amsterdam fields");
        };
        assert!(
            error.to_string().contains("Amsterdam"),
            "unexpected error: {error}"
        );
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
