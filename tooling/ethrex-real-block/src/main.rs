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

    // The chain rules come from `network`, and ethrex-replay maps every chain it
    // doesn't recognise (anything but mainnet / Hoodi / Sepolia) onto
    // `LocalDevnet` — which resolves to a test chain: chain_id 9, every fork
    // active from timestamp 0. Converting under that would execute the block
    // against invented rules while still satisfying every check downstream (the
    // witness is replayed against whatever config we picked, and host and guest
    // read the same one), so refuse rather than guess.
    if !matches!(cache.network, Network::PublicNetwork(_)) {
        return Err(format!(
            "unsupported network `{}`: its chain rules would be guessed, not read \
             (ethrex-replay writes LocalDevnet for any chain it does not recognise)",
            cache.network
        )
        .into());
    }

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
    use ethrex_config::networks::HOODI_CHAIN_ID;

    const CACHE: &str = "caches/cache_hoodi_1265656.json";

    /// This crate is pinned to Hoodi 1265656 on purpose, independently of whichever
    /// block the benchmarks currently prove: it is the one cache ethrex-replay hosts
    /// upstream, so keeping the converter's test input there costs us no hosting and
    /// cannot drift. What is under test here is the CONVERSION, not the benchmark
    /// workload — see tooling/ethrex-real-block/README.md.
    ///
    /// `caches/` is gitignored and fetched on demand, so every test here fails on
    /// a clean checkout until the cache is downloaded. Say so instead of surfacing
    /// a bare `No such file or directory` from `unwrap()`.
    const CACHE_MISSING: &str = "caches/cache_hoodi_1265656.json is missing — run \
                                 `make ethrex-real-block-converter-cache` from the repo root first";

    /// Executes the converted block with `LambdaVmEcsmCrypto`, the `Crypto` impl
    /// the guest injects, so the block is exercised through the same trait
    /// dispatch the guest uses. Stateless re-execution ends in a post-state-root
    /// check, so any divergence from consensus fails here — on the host, with no
    /// RV64 toolchain and no proving run.
    ///
    /// It does NOT screen KZG: this crate's graph links `c-kzg` (via
    /// `ethrex-config` → `ethrex-p2p`, see Cargo.toml), so point evaluation
    /// (0x0a) resolves to a working implementation here while the guest has none.
    /// A block calling 0x0a would pass this test.
    ///
    /// `test_ethrex_real_block_native` in `tooling/ethrex-tests` is what covers
    /// 0x0a, and `no_kzg_backend_linked` there keeps it covered. That split is
    /// sufficient rather than a workaround: KZG is the only precompile in
    /// `ethrex-crypto` whose *availability* is feature-gated — the other gates
    /// swap between two working implementations.
    #[test]
    fn real_block_executes_under_guest_crypto() {
        use ethrex_guest_program::l1::execution_program;
        use lambda_vm_ethrex_crypto::LambdaVmEcsmCrypto;
        use std::sync::Arc;

        let (program_input, _) = program_input_from_cache(CACHE).expect(CACHE_MISSING);
        execution_program(program_input, Arc::new(LambdaVmEcsmCrypto)).unwrap();
    }

    /// A cache whose `network` we can't map to real chain rules must be refused,
    /// not converted under substituted ones. Only the `network` field is changed
    /// here, and the result is byte-length-identical to the real fixture — which
    /// is exactly why the other tests cannot catch this on their own.
    #[test]
    fn unmappable_network_is_rejected() {
        let mut cache: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(CACHE).expect(CACHE_MISSING)).unwrap();
        cache["network"] = serde_json::json!("LocalDevnet");

        let path = std::env::temp_dir().join(format!(
            "ethrex_real_block_localdevnet_{}.json",
            std::process::id()
        ));
        std::fs::write(&path, serde_json::to_vec(&cache).unwrap()).unwrap();
        let result = program_input_from_cache(path.to_str().unwrap());
        std::fs::remove_file(&path).ok();
        let Err(err) = result else {
            panic!("LocalDevnet resolves to chain_id 9 with all forks at 0; must not convert");
        };

        assert!(
            err.to_string().contains("unsupported network"),
            "wrong rejection reason: {err}"
        );
    }

    /// The serialized form is what the guest actually reads, so pin it: a
    /// change here means the rkyv layout moved and every consumer of the
    /// fixture (and its README checksum) needs regenerating.
    #[test]
    fn conversion_is_reproducible() {
        use sha2::Digest;

        let (program_input, summary) = program_input_from_cache(CACHE).expect(CACHE_MISSING);
        let bytes = rkyv::to_bytes::<rkyv::rancor::Error>(&program_input).unwrap();

        assert_eq!(summary.first_block_number, 1_265_656);
        assert_eq!(summary.transactions, 11);
        assert_eq!(summary.gas_used, 4_402_947);

        // Asserted separately from the digest below, not covered by it. The digest
        // is exactly what a legitimate ethrex rev bump forces someone to rewrite
        // (the layout moves, this goes red, a fresh digest gets pasted in) — and at
        // that moment it stops covering the substituted-chain-config case it was
        // chosen for. This assert survives that churn.
        assert_eq!(
            program_input.execution_witness.chain_config.chain_id, HOODI_CHAIN_ID,
            "chain config is not Hoodi's — the block would replay under other rules",
        );

        // Digest, not length: a layout change can preserve the byte count exactly
        // (`ChainConfig` is fixed-size, and rkyv's `big_endian` feature would only
        // byte-swap in place), so `len()` cannot pin the archived form.
        let digest: String = sha2::Sha256::digest(&bytes)
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect();
        assert_eq!(
            digest, "1f7d4c4cdf9bd52472d9ebafdb4038f57a88c3c92d65c96fd86d7e323db87142",
            "fixture bytes changed — regenerate it and update the README checksum",
        );
    }
}
