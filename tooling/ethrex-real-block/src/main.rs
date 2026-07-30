//! Convert a real Ethereum block into the rkyv-serialized `ProgramInput` the
//! lambda-vm ethrex guest consumes, from an `ethrex-replay` cache JSON.
//!
//! Usage:
//!   cargo run --release -- <cache.json> <output_path>

use ethrex_common::types::Block;
use ethrex_common::types::block_execution_witness::RpcExecutionWitness;
use ethrex_config::networks::Network;
use ethrex_guest_program::l1::ProgramInput;
use serde::Deserialize;

/// The subset of `ethrex-replay`'s on-disk cache that a `ProgramInput` needs.
///
/// Deliberately deserialized with *our* pinned ethrex types rather than by
/// depending on `ethrex-replay`: it tracks ethrex `main`, where `ProgramInput`
/// has diverged (extra `fee_configs` field, different module path, rkyv 0.8.10
/// vs our `=0.8.16`), so its own `.bin` output would not deserialize in our
/// guest. This JSON is the version-tolerant interface between the two.
///
/// Extra fields in the file (L2 blob data, custom `chain_config`) are ignored.
#[derive(Deserialize)]
struct Cache {
    blocks: Vec<Block>,
    witness: RpcExecutionWitness,
    network: Network,
}

/// Summary of the converted block, for the CLI's one-line report.
struct BlockSummary {
    network: String,
    first_block_number: u64,
    blocks: usize,
    transactions: usize,
    gas_used: u64,
}

fn program_input_from_cache(
    cache_path: &str,
) -> Result<(ProgramInput, BlockSummary), Box<dyn std::error::Error>> {
    let cache: Cache =
        serde_json::from_reader(std::io::BufReader::new(std::fs::File::open(cache_path)?))?;

    let Some(first_block) = cache.blocks.first() else {
        return Err("cache contains no blocks".into());
    };
    let summary = BlockSummary {
        network: cache.network.to_string(),
        first_block_number: first_block.header.number,
        blocks: cache.blocks.len(),
        transactions: cache.blocks.iter().map(|b| b.body.transactions.len()).sum(),
        gas_used: cache.blocks.iter().map(|b| b.header.gas_used).sum(),
    };

    // `into_execution_witness` rebuilds the trie structures from the flat node
    // list and needs the parent header, which the cache carries inside `witness`.
    let chain_config = cache.network.get_genesis()?.config;
    let witness = cache
        .witness
        .into_execution_witness(chain_config, summary.first_block_number)?;

    Ok((ProgramInput::new(cache.blocks, witness), summary))
}

fn usage_and_exit(program: &str) -> ! {
    eprintln!("usage: {program} <cache.json> <output_path>");
    std::process::exit(2);
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args();
    let program = args.next().unwrap_or_else(|| "ethrex-real-block".into());
    let (Some(cache_path), Some(out_path)) = (args.next(), args.next()) else {
        usage_and_exit(&program);
    };
    if args.next().is_some() {
        usage_and_exit(&program);
    }

    let (program_input, summary) = program_input_from_cache(&cache_path)?;
    let bytes = rkyv::to_bytes::<rkyv::rancor::Error>(&program_input)?;
    std::fs::write(&out_path, &bytes)?;

    println!(
        "wrote {out_path} ({} bytes): {} block(s) from {} starting at #{}, \
         {} transaction(s), {} gas",
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

    /// Executes the converted block against the guest's exact precompile
    /// surface: `ethrex-guest-program` with `default-features = false,
    /// features = ["lambdavm"]` (see Cargo.toml) plus `LambdaVmEcsmCrypto`,
    /// the same `Crypto` impl the guest injects.
    ///
    /// This is the screen for "does the block need an accelerator we don't
    /// have". Stateless re-execution ends in a post-state-root check, so a
    /// block reaching an unlinked precompile — KZG point evaluation (0x0a) is
    /// the one `lambdavm` omits — diverges from consensus and fails here,
    /// on the host, without needing an RV64 toolchain or a proving run.
    #[test]
    fn real_block_executes_under_guest_crypto() {
        use ethrex_guest_program::l1::execution_program;
        use lambda_vm_ethrex_crypto::LambdaVmEcsmCrypto;
        use std::sync::Arc;

        let (program_input, _) = program_input_from_cache(CACHE).unwrap();
        execution_program(program_input, Arc::new(LambdaVmEcsmCrypto)).unwrap();
    }

    /// The serialized form is what the guest actually reads, so pin it: a
    /// change here means the rkyv layout moved and every consumer of the
    /// fixture (and its README checksum) needs regenerating.
    #[test]
    fn conversion_is_reproducible() {
        let (program_input, summary) = program_input_from_cache(CACHE).unwrap();
        let bytes = rkyv::to_bytes::<rkyv::rancor::Error>(&program_input).unwrap();

        assert_eq!(summary.first_block_number, 1_265_656);
        assert_eq!(summary.transactions, 11);
        assert_eq!(summary.gas_used, 4_402_947);

        // Pin the exact serialized bytes, not just length: HashMap iteration order
        // could vary the output while keeping size fixed.
        let expected: [u8; 32] = [
            0x1f, 0x7d, 0x4c, 0x4c, 0xdf, 0x9b, 0xd5, 0x24,
            0x72, 0xd9, 0xeb, 0xaf, 0xdb, 0x40, 0x38, 0xf5,
            0x7a, 0x88, 0xc3, 0xc9, 0x2d, 0x65, 0xc9, 0x6f,
            0xd8, 0x6d, 0x7e, 0x32, 0x3d, 0xb8, 0x71, 0x42,
        ];
        assert_eq!(sha2::Sha256::digest(&bytes).as_slice(), expected);
    }
}
