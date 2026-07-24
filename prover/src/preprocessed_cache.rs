//! Process-local cache for program-specific preprocessed commitments.
//!
//! DECODE and ELF-data PAGE commitments depend only on the ELF contents and
//! proof options, but building them requires FFTs and Merkle trees. Native
//! proving commonly proves the same program many times, so recomputing these
//! roots for every execution is pure overhead.

use std::collections::VecDeque;
use std::sync::{Arc, LazyLock, Mutex};

use executor::elf::Elf;

use crate::{Commitment, ProofOptions, tables};

const CACHE_CAPACITY: usize = 8;
const CACHE_TOGGLE: &str = "LAMBDA_VM_PREPROCESSED_CACHE";

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct CacheKey {
    elf_digest: [u8; 32],
    blowup_factor: u8,
    fri_number_of_queries: usize,
    coset_offset: u64,
    grinding_factor: u8,
    fri_final_poly_log_degree: u8,
}

impl CacheKey {
    fn new(elf_digest: [u8; 32], options: &ProofOptions) -> Self {
        Self {
            elf_digest,
            blowup_factor: options.blowup_factor,
            fri_number_of_queries: options.fri_number_of_queries,
            coset_offset: options.coset_offset,
            grinding_factor: options.grinding_factor,
            fri_final_poly_log_degree: options.fri_final_poly_log_degree,
        }
    }
}

pub(crate) struct PreprocessedCommitments {
    pub(crate) decode: Commitment,
    pub(crate) pages: Vec<(u64, Commitment)>,
}

type CacheEntry = (CacheKey, Arc<PreprocessedCommitments>);

static CACHE: LazyLock<Mutex<VecDeque<CacheEntry>>> =
    LazyLock::new(|| Mutex::new(VecDeque::with_capacity(CACHE_CAPACITY)));

fn cache_enabled() -> bool {
    !matches!(
        std::env::var(CACHE_TOGGLE).as_deref(),
        Ok("0" | "false" | "off")
    )
}

fn cache_lock() -> std::sync::MutexGuard<'static, VecDeque<CacheEntry>> {
    CACHE
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn compute(
    elf: &Elf,
    options: &ProofOptions,
    page_configs: &[tables::page::PageConfig],
) -> Arc<PreprocessedCommitments> {
    #[cfg(feature = "instruments")]
    let __sp = stark::instruments::span("air_preprocessed_compute");

    let decode = {
        #[cfg(feature = "instruments")]
        let __sp = stark::instruments::span("air_cached_decode_compute");
        tables::decode::commitment_from_elf(elf, options)
            .expect("failed to compute cached DECODE commitment")
    };

    let pages = {
        #[cfg(feature = "instruments")]
        let __sp = stark::instruments::span("air_cached_pages_compute");
        page_configs
            .iter()
            .filter(|config| !config.is_private_input && config.init_values.is_some())
            .map(|config| {
                (
                    config.page_base,
                    tables::page::compute_precomputed_commitment(config, options),
                )
            })
            .collect()
    };

    Arc::new(PreprocessedCommitments { decode, pages })
}

/// Return roots for this exact `(ELF contents, proof options)` pair.
///
/// The digest is the same Keccak-256 identity already bound into the proof's
/// Fiat-Shamir statement. Cache storage is bounded and contains only roots, not
/// ELF bytes or traces. A miss is computed outside the mutex so unrelated
/// programs can initialize concurrently; a same-key race may duplicate work
/// once, but both computations are deterministic.
pub(crate) fn get(
    elf: &Elf,
    elf_digest: [u8; 32],
    options: &ProofOptions,
    page_configs: &[tables::page::PageConfig],
) -> Arc<PreprocessedCommitments> {
    #[cfg(feature = "instruments")]
    let __sp = stark::instruments::span("air_preprocessed_cache");

    if !cache_enabled() {
        return compute(elf, options, page_configs);
    }

    let key = CacheKey::new(elf_digest, options);
    if let Some(hit) = cache_lock()
        .iter()
        .find(|(candidate, _)| *candidate == key)
        .map(|(_, roots)| Arc::clone(roots))
    {
        return hit;
    }

    let computed = compute(elf, options, page_configs);
    let mut cache = cache_lock();

    // Another prover may have filled this key while we computed outside the
    // lock. Prefer that entry and avoid storing a duplicate.
    if let Some(hit) = cache
        .iter()
        .find(|(candidate, _)| *candidate == key)
        .map(|(_, roots)| Arc::clone(roots))
    {
        return hit;
    }

    if cache.len() == CACHE_CAPACITY {
        cache.pop_front();
    }
    cache.push_back((key, Arc::clone(&computed)));
    computed
}

#[cfg(test)]
mod tests {
    use super::CacheKey;
    use crate::ProofOptions;

    fn options() -> ProofOptions {
        ProofOptions {
            blowup_factor: 2,
            fri_number_of_queries: 80,
            coset_offset: 3,
            grinding_factor: 20,
            fri_final_poly_log_degree: 7,
        }
    }

    #[test]
    fn cache_key_binds_elf_and_every_proof_option() {
        let base = options();
        let key = CacheKey::new([7; 32], &base);

        let mut changed = base.clone();
        changed.blowup_factor = 4;
        assert_ne!(key, CacheKey::new([7; 32], &changed));

        changed = base.clone();
        changed.fri_number_of_queries += 1;
        assert_ne!(key, CacheKey::new([7; 32], &changed));

        changed = base.clone();
        changed.coset_offset += 1;
        assert_ne!(key, CacheKey::new([7; 32], &changed));

        changed = base.clone();
        changed.grinding_factor += 1;
        assert_ne!(key, CacheKey::new([7; 32], &changed));

        changed = base.clone();
        changed.fri_final_poly_log_degree += 1;
        assert_ne!(key, CacheKey::new([7; 32], &changed));

        assert_ne!(key, CacheKey::new([8; 32], &base));
    }
}
