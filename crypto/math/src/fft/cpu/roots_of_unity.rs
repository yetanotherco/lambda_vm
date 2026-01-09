use crate::field::{
    element::FieldElement,
    traits::{IsFFTField, RootsConfig},
};
use alloc::vec::Vec;

use crate::fft::errors::FFTError;

use super::bit_reversing::in_place_bit_reverse_permute;

#[cfg(feature = "std")]
use std::collections::HashMap;
#[cfg(feature = "std")]
use std::sync::RwLock;

/// A cache for storing pre-computed twiddle factors.
/// This cache is thread-safe and can be shared across multiple FFT computations.
/// Twiddles are stored by (order, config) pairs.
#[cfg(feature = "std")]
pub struct TwiddleCache<F: IsFFTField> {
    cache: RwLock<HashMap<(u64, RootsConfig), Vec<FieldElement<F>>>>,
}

#[cfg(feature = "std")]
impl<F: IsFFTField> Default for TwiddleCache<F> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(feature = "std")]
impl<F: IsFFTField> TwiddleCache<F> {
    /// Creates a new empty twiddle cache.
    pub fn new() -> Self {
        Self {
            cache: RwLock::new(HashMap::new()),
        }
    }

    /// Gets twiddle factors from cache, computing and caching them if not present.
    /// This is the primary interface for twiddle factor retrieval.
    pub fn get_or_compute(
        &self,
        order: u64,
        config: RootsConfig,
    ) -> Result<Vec<FieldElement<F>>, FFTError> {
        // First, try to read from cache
        {
            let read_guard = self.cache.read().unwrap();
            if let Some(twiddles) = read_guard.get(&(order, config)) {
                return Ok(twiddles.clone());
            }
        }

        // Not in cache, compute and store
        let twiddles = get_twiddles(order, config)?;

        {
            let mut write_guard = self.cache.write().unwrap();
            write_guard.insert((order, config), twiddles.clone());
        }

        Ok(twiddles)
    }

    /// Pre-computes and caches twiddle factors for a given order.
    /// Useful for warming up the cache before performance-critical sections.
    pub fn precompute(&self, order: u64, config: RootsConfig) -> Result<(), FFTError> {
        let _ = self.get_or_compute(order, config)?;
        Ok(())
    }

    /// Clears all cached twiddle factors.
    pub fn clear(&self) {
        let mut write_guard = self.cache.write().unwrap();
        write_guard.clear();
    }

    /// Returns the number of cached twiddle factor sets.
    pub fn len(&self) -> usize {
        let read_guard = self.cache.read().unwrap();
        read_guard.len()
    }

    /// Returns true if the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Returns a `Vec` of the powers of a `2^n`th primitive root of unity in some configuration
/// `config`. For example, in a `Natural` config this would yield: w^0, w^1, w^2...
pub fn get_powers_of_primitive_root<F: IsFFTField>(
    n: u64,
    count: usize,
    config: RootsConfig,
) -> Result<Vec<FieldElement<F>>, FFTError> {
    if count == 0 {
        return Ok(Vec::new());
    }

    let root = match config {
        RootsConfig::Natural | RootsConfig::BitReverse => F::get_primitive_root_of_unity(n)?,
        _ => F::get_primitive_root_of_unity(n)?.inv().unwrap(),
    };
    let up_to = match config {
        RootsConfig::Natural | RootsConfig::NaturalInversed => count,
        // In bit reverse form we could need as many as `(1 << count.bits()) - 1` roots
        _ => count.next_power_of_two(),
    };

    let mut results = Vec::with_capacity(up_to);
    // NOTE: a nice version would be using `core::iter::successors`. However, this is 10% faster.
    results.extend((0..up_to).scan(FieldElement::one(), |state, _| {
        let res = state.clone();
        *state = &(*state) * &root;
        Some(res)
    }));

    if matches!(
        config,
        RootsConfig::BitReverse | RootsConfig::BitReverseInversed
    ) {
        in_place_bit_reverse_permute(&mut results);
    }

    Ok(results)
}

/// Returns a `Vec` of the powers of a `2^n`th primitive root of unity, scaled `offset` times,
/// in a Natural configuration.
pub fn get_powers_of_primitive_root_coset<F: IsFFTField>(
    n: u64,
    count: usize,
    offset: &FieldElement<F>,
) -> Result<Vec<FieldElement<F>>, FFTError> {
    let root = F::get_primitive_root_of_unity(n)?;
    let results = (0..count).map(|i| offset * root.pow(i));

    Ok(results.collect())
}

/// Returns 2^`order` / 2 twiddle factors for FFT in some configuration `config`.
/// Twiddle factors are powers of a primitive root of unity of the field, used for FFT
/// computations. FFT only requires the first half of all the powers
pub fn get_twiddles<F: IsFFTField>(
    order: u64,
    config: RootsConfig,
) -> Result<Vec<FieldElement<F>>, FFTError> {
    if order > 63 {
        return Err(FFTError::OrderError(order));
    }

    get_powers_of_primitive_root(order, (1 << order) / 2, config)
}

#[cfg(test)]
mod tests {
    use crate::{
        fft::{
            cpu::{bit_reversing::in_place_bit_reverse_permute, roots_of_unity::get_twiddles},
            errors::FFTError,
        },
        field::{test_fields::u64_test_field::U64TestField, traits::RootsConfig},
    };
    use proptest::prelude::*;

    type F = U64TestField;

    proptest! {
        #[test]
        fn test_gen_twiddles_bit_reversed_validity(n in 1..8_u64) {
            let twiddles = get_twiddles::<F>(n, RootsConfig::Natural).unwrap();
            let mut twiddles_to_reorder = get_twiddles(n, RootsConfig::BitReverse).unwrap();
            in_place_bit_reverse_permute(&mut twiddles_to_reorder); // so now should be naturally ordered

            prop_assert_eq!(twiddles, twiddles_to_reorder);
        }
    }

    #[test]
    fn gen_twiddles_with_order_greater_than_63_should_fail() {
        let twiddles = get_twiddles::<F>(64, RootsConfig::Natural);

        assert!(matches!(twiddles, Err(FFTError::OrderError(_))));
    }

    #[cfg(feature = "std")]
    mod cache_tests {
        use super::*;
        use crate::fft::cpu::roots_of_unity::TwiddleCache;

        #[test]
        fn test_twiddle_cache_basic() {
            let cache: TwiddleCache<F> = TwiddleCache::new();
            assert!(cache.is_empty());

            // First call computes and caches
            let twiddles1 = cache.get_or_compute(4, RootsConfig::BitReverse).unwrap();
            assert_eq!(cache.len(), 1);

            // Second call should hit cache
            let twiddles2 = cache.get_or_compute(4, RootsConfig::BitReverse).unwrap();
            assert_eq!(twiddles1, twiddles2);
            assert_eq!(cache.len(), 1);

            // Different config should create new entry
            let _twiddles3 = cache.get_or_compute(4, RootsConfig::Natural).unwrap();
            assert_eq!(cache.len(), 2);

            // Different order should create new entry
            let _twiddles4 = cache.get_or_compute(5, RootsConfig::BitReverse).unwrap();
            assert_eq!(cache.len(), 3);
        }

        #[test]
        fn test_twiddle_cache_clear() {
            let cache: TwiddleCache<F> = TwiddleCache::new();

            cache.precompute(4, RootsConfig::BitReverse).unwrap();
            cache.precompute(5, RootsConfig::Natural).unwrap();
            assert_eq!(cache.len(), 2);

            cache.clear();
            assert!(cache.is_empty());
        }

        #[test]
        fn test_twiddle_cache_matches_direct_computation() {
            let cache: TwiddleCache<F> = TwiddleCache::new();

            let direct = get_twiddles::<F>(6, RootsConfig::BitReverse).unwrap();
            let cached = cache.get_or_compute(6, RootsConfig::BitReverse).unwrap();

            assert_eq!(direct, cached);
        }
    }
}
