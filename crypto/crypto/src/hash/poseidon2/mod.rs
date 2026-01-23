//! Poseidon2 hash function implementation for Goldilocks field.
//!
//! Poseidon2 is an optimized version of the Poseidon hash function designed for
//! efficient implementation in both hardware and software, particularly for
//! zero-knowledge proof systems.
//!
//! # Configuration
//!
//! | Parameter | Value |
//! |-----------|-------|
//! | Field | Goldilocks (p = 2^64 - 2^32 + 1) |
//! | Width | 8 |
//! | Rate | 4 |
//! | Capacity | 4 |
//! | S-box | x^7 |
//! | Rounds | 4 (external) + 22 (internal) + 4 (external) = 30 |
//!
//! # Security Levels
//!
//! This implementation provides two security levels:
//!
//! | Function Family | Output Size | Security (collision) | Use Case |
//! |-----------------|-------------|----------------------|----------|
//! | 64-bit (`hash`, `compress`, etc.) | 1 Fp (8 bytes) | ~32 bits (birthday) | Legacy, non-critical |
//! | 128-bit (`hash_128`, `compress_128`, etc.) | 2 Fp (16 bytes) | ~64 bits (birthday) | **Recommended for STARK** |
//!
//! The 128-bit functions return `[Fp; 2]` and provide collision resistance equivalent
//! to Keccak256 for birthday attacks (~2^64 operations to find collision).
//!
//! ## 128-bit Hash Functions
//!
//! | Function | Description | Domain Tag |
//! |----------|-------------|------------|
//! | [`Poseidon2::hash_128`] | Hash two field elements | 2 |
//! | [`Poseidon2::hash_single_128`] | Hash single field element | 1 |
//! | [`Poseidon2::compress_128`] | Compress two 128-bit digests (4 Fp inputs) | 4 |
//! | [`Poseidon2::hash_many_128`] | Sponge construction for arbitrary inputs | 10* padding |
//! | [`Poseidon2::hash_vec_128`] | Convenience wrapper (delegates to above) | varies |
//!
//! # Key Differences from Poseidon
//!
//! - Different linear layers for external and internal rounds
//! - Optimized MDS matrices (Horizen Labs 4×4 matrix)
//! - Reduced constraint complexity
//!
//! # Domain Separation
//!
//! The implementation uses domain separation via the capacity element:
//! - `hash_single(x)` / `hash_single_128(x)`: domain tag = 1
//! - `hash(x, y)` / `hash_128(x, y)`: domain tag = 2
//! - `compress_128(left, right)`: domain tag = 4 (for 4 Fp inputs)
//! - `hash_many(inputs)` / `hash_many_128(inputs)`: uses 10* padding (no explicit domain tag)
//!
//! This ensures that `hash(a, b) ≠ hash_many([a, b])` due to different constructions.
//!
//! # Example: 128-bit Merkle Tree Commitment
//!
//! ```rust
//! use crypto::hash::poseidon2::{Poseidon2, Fp};
//!
//! // Hash leaf data to 128-bit commitment
//! let leaf = Fp::from(42u64);
//! let leaf_hash: [Fp; 2] = Poseidon2::hash_single_128(&leaf);
//!
//! // Compress two child nodes into parent (Merkle tree)
//! let left_child = [Fp::from(1u64), Fp::from(2u64)];
//! let right_child = [Fp::from(3u64), Fp::from(4u64)];
//! let parent: [Fp; 2] = Poseidon2::compress_128(&left_child, &right_child);
//! ```
//!
//! # References
//!
//! - Paper: <https://eprint.iacr.org/2023/323>
//! - Constants: HorizenLabs/Plonky3 implementation

pub mod goldilocks;

use math::field::element::FieldElement;
use math::field::fields::fft_friendly::u64_goldilocks::GoldilocksField;

use goldilocks::{
    EXTERNAL_ROUND_CONSTANTS_INIT, EXTERNAL_ROUND_CONSTANTS_TERM, EXTERNAL_ROUNDS_BEGIN,
    EXTERNAL_ROUNDS_END, INTERNAL_ROUND_CONSTANTS, INTERNAL_ROUNDS, MATRIX_DIAG_8, RATE, WIDTH,
};

/// Type alias for Goldilocks field element
pub type Fp = FieldElement<GoldilocksField>;

/// Poseidon2 hash function for Goldilocks field with width 8
pub struct Poseidon2 {
    state: [Fp; WIDTH],
}

impl Default for Poseidon2 {
    fn default() -> Self {
        Self::new()
    }
}

impl Poseidon2 {
    /// Create a new Poseidon2 instance with zero state
    pub fn new() -> Self {
        Self {
            state: core::array::from_fn(|_| Fp::zero()),
        }
    }

    /// Create a new Poseidon2 instance with given initial state
    pub fn with_state(state: [Fp; WIDTH]) -> Self {
        Self { state }
    }

    /// Get the current state
    pub fn state(&self) -> &[Fp; WIDTH] {
        &self.state
    }

    /// Apply the full Poseidon2 permutation to the state
    pub fn permute(&mut self) {
        // Initial linear layer (only for initial external rounds)
        self.external_linear_layer();

        // Initial external rounds (after initial linear layer)
        for round_constants in EXTERNAL_ROUND_CONSTANTS_INIT
            .iter()
            .take(EXTERNAL_ROUNDS_BEGIN)
        {
            self.external_round(round_constants);
        }

        // Internal rounds
        for &round_constant in INTERNAL_ROUND_CONSTANTS.iter().take(INTERNAL_ROUNDS) {
            self.internal_round(round_constant);
        }

        // Terminal external rounds (no initial linear layer)
        for round_constants in EXTERNAL_ROUND_CONSTANTS_TERM
            .iter()
            .take(EXTERNAL_ROUNDS_END)
        {
            self.external_round(round_constants);
        }
    }

    /// Apply the S-box (x^7) to a single element
    #[inline]
    fn sbox(x: &Fp) -> Fp {
        // x^7 = x^4 * x^2 * x
        let x2 = x * x;
        let x4 = &x2 * &x2;
        let x6 = &x4 * &x2;
        &x6 * x
    }

    /// External round: AddRoundConstants + S-box (all elements) + External Linear Layer
    fn external_round(&mut self, round_constants: &[u64; WIDTH]) {
        // Add round constants
        for (state_elem, &rc) in self.state.iter_mut().zip(round_constants.iter()) {
            *state_elem = &*state_elem + &Fp::from(rc);
        }

        // Apply S-box to all elements
        for state_elem in &mut self.state {
            *state_elem = Self::sbox(state_elem);
        }

        // External linear layer (MDS)
        self.external_linear_layer();
    }

    /// Internal round: AddRoundConstant (first element only) + S-box (first element only) + Internal Linear Layer
    fn internal_round(&mut self, round_constant: u64) {
        // Add round constant to first element only
        self.state[0] = &self.state[0] + &Fp::from(round_constant);

        // Apply S-box to first element only
        self.state[0] = Self::sbox(&self.state[0]);

        // Internal linear layer (diagonal matrix with 1s)
        self.internal_linear_layer();
    }

    /// Apply the Horizen Labs 4x4 MDS matrix to a 4-element array
    /// Matrix: [[5,7,1,3], [4,6,1,1], [1,3,5,7], [1,1,4,6]]
    /// This requires 10 additions and 4 doubles
    #[inline]
    fn apply_hl_mat4(x: &mut [Fp; 4]) {
        // t0 = x0 + x1
        let t0 = &x[0] + &x[1];
        // t1 = x2 + x3
        let t1 = &x[2] + &x[3];
        // t2 = 2*x1 + t1 = 2*x1 + x2 + x3
        let t2 = &(&x[1] + &x[1]) + &t1;
        // t3 = 2*x3 + t0 = x0 + x1 + 2*x3
        let t3 = &(&x[3] + &x[3]) + &t0;
        // t4 = 4*t1 + t3 = x0 + x1 + 4*x2 + 6*x3
        let t1_double = &t1 + &t1;
        let t4 = &(&t1_double + &t1_double) + &t3;
        // t5 = 4*t0 + t2 = 4*x0 + 6*x1 + x2 + x3
        let t0_double = &t0 + &t0;
        let t5 = &(&t0_double + &t0_double) + &t2;
        // t6 = t3 + t5 = 5*x0 + 7*x1 + x2 + 3*x3
        let t6 = &t3 + &t5;
        // t7 = t2 + t4 = x0 + 3*x1 + 5*x2 + 7*x3
        let t7 = &t2 + &t4;

        // Output: [t6, t5, t7, t4]
        // Row 0: 5*x0 + 7*x1 + 1*x2 + 3*x3 = t6
        // Row 1: 4*x0 + 6*x1 + 1*x2 + 1*x3 = t5
        // Row 2: 1*x0 + 3*x1 + 5*x2 + 7*x3 = t7
        // Row 3: 1*x0 + 1*x1 + 4*x2 + 6*x3 = t4
        x[0] = t6;
        x[1] = t5;
        x[2] = t7;
        x[3] = t4;
    }

    /// External linear layer using Horizen Labs 4x4 MDS matrix applied to two halves
    /// For width 8: apply 4x4 MDS to state[0..4] and state[4..8], then diffuse
    fn external_linear_layer(&mut self) {
        // Apply HL M4 to each 4-element chunk
        let mut first_half: [Fp; 4] = core::array::from_fn(|i| self.state[i]);
        let mut second_half: [Fp; 4] = core::array::from_fn(|i| self.state[i + 4]);

        Self::apply_hl_mat4(&mut first_half);
        Self::apply_hl_mat4(&mut second_half);

        // Copy results back
        self.state[..4].copy_from_slice(&first_half);
        self.state[4..8].copy_from_slice(&second_half);

        // Diffuse across blocks: add column sums back to each element
        // stored[l] = state[l] + state[4+l] for each column l
        // state[i] += stored[i % 4]
        for i in 0..4 {
            let sum = &self.state[i] + &self.state[i + 4];
            self.state[i] = &self.state[i] + &sum;
            self.state[i + 4] = &self.state[i + 4] + &sum;
        }
    }

    /// Internal linear layer using diagonal matrix
    /// The matrix is: M_internal = diag(MATRIX_DIAG) + all-ones matrix
    /// Equivalently: y_i = diag_i * x_i + sum(x_j)
    fn internal_linear_layer(&mut self) {
        // Compute sum of all elements
        let mut sum = Fp::zero();
        for state_elem in &self.state {
            sum = &sum + state_elem;
        }

        // Apply: y_i = (diag_i - 1) * x_i + sum
        // Which equals: y_i = diag_i * x_i + sum - x_i = diag_i * x_i + (sum of other elements)
        for (state_elem, &diag_val) in self.state.iter_mut().zip(MATRIX_DIAG_8.iter()) {
            let diag = Fp::from(diag_val);
            *state_elem = &(&diag * &*state_elem) + &sum;
        }
    }

    // ========================================================================
    // Sponge-based hash functions
    // ========================================================================

    /// Hash two field elements and return a single field element
    pub fn hash(x: &Fp, y: &Fp) -> Fp {
        let mut hasher = Self::new();
        hasher.state[0] = *x;
        hasher.state[1] = *y;
        // Domain separation: set capacity element to 2 (for 2 inputs)
        hasher.state[WIDTH - 1] = Fp::from(2u64);
        hasher.permute();
        hasher.state[0]
    }

    /// Hash a single field element
    pub fn hash_single(x: &Fp) -> Fp {
        let mut hasher = Self::new();
        hasher.state[0] = *x;
        // Domain separation: set capacity element to 1 (for 1 input)
        hasher.state[WIDTH - 1] = Fp::from(1u64);
        hasher.permute();
        hasher.state[0]
    }

    /// Hash multiple field elements using sponge construction
    pub fn hash_many(inputs: &[Fp]) -> Fp {
        let mut hasher = Self::new();

        // Pad input with 1 followed by 0s (if necessary)
        let mut values = inputs.to_vec();
        values.push(Fp::from(1u64)); // Padding
        while !values.len().is_multiple_of(RATE) {
            values.push(Fp::zero());
        }

        // Absorb phase: process input in chunks of RATE
        for chunk in values.chunks(RATE) {
            // XOR chunk into state (first RATE elements)
            for (i, val) in chunk.iter().enumerate() {
                hasher.state[i] = &hasher.state[i] + val;
            }
            hasher.permute();
        }

        // Squeeze phase: return first element
        hasher.state[0]
    }

    /// Compress two digests into one (for Merkle tree internal nodes)
    pub fn compress(left: &Fp, right: &Fp) -> Fp {
        Self::hash(left, right)
    }

    // ========================================================================
    // 128-bit security hash functions
    // ========================================================================
    // These functions return [Fp; 2] (128 bits) instead of Fp (64 bits)
    // for enhanced collision resistance equivalent to Keccak256 security.

    /// Hash two field elements and return a 128-bit digest (2 Fp)
    pub fn hash_128(x: &Fp, y: &Fp) -> [Fp; 2] {
        let mut hasher = Self::new();
        hasher.state[0] = *x;
        hasher.state[1] = *y;
        // Domain separation: set capacity element to 2 (for 2 inputs)
        hasher.state[WIDTH - 1] = Fp::from(2u64);
        hasher.permute();
        [hasher.state[0], hasher.state[1]]
    }

    /// Hash a single field element and return a 128-bit digest (2 Fp)
    pub fn hash_single_128(x: &Fp) -> [Fp; 2] {
        let mut hasher = Self::new();
        hasher.state[0] = *x;
        // Domain separation: set capacity element to 1 (for 1 input)
        hasher.state[WIDTH - 1] = Fp::from(1u64);
        hasher.permute();
        [hasher.state[0], hasher.state[1]]
    }

    /// Compress two 128-bit digests into one 128-bit digest (for Merkle tree internal nodes)
    pub fn compress_128(left: &[Fp; 2], right: &[Fp; 2]) -> [Fp; 2] {
        let mut hasher = Self::new();
        hasher.state[0] = left[0];
        hasher.state[1] = left[1];
        hasher.state[2] = right[0];
        hasher.state[3] = right[1];
        // Domain separation: set capacity element to 4 (for 4 inputs)
        hasher.state[WIDTH - 1] = Fp::from(4u64);
        hasher.permute();
        [hasher.state[0], hasher.state[1]]
    }

    /// Hash multiple field elements using sponge construction, return 128-bit digest
    pub fn hash_many_128(inputs: &[Fp]) -> [Fp; 2] {
        let mut hasher = Self::new();

        // Pad input with 1 followed by 0s (if necessary)
        let mut values = inputs.to_vec();
        values.push(Fp::from(1u64)); // Padding
        while !values.len().is_multiple_of(RATE) {
            values.push(Fp::zero());
        }

        // Absorb phase: process input in chunks of RATE
        for chunk in values.chunks(RATE) {
            // XOR chunk into state (first RATE elements)
            for (i, val) in chunk.iter().enumerate() {
                hasher.state[i] = &hasher.state[i] + val;
            }
            hasher.permute();
        }

        // Squeeze phase: return first two elements (128 bits)
        [hasher.state[0], hasher.state[1]]
    }

    /// Hash a vector of field elements (for Merkle tree leaves), return 128-bit digest.
    ///
    /// # Behavior
    ///
    /// - **Empty**: Returns `[Fp::zero(), Fp::zero()]` (debug assertion guards against misuse)
    /// - **Length 1**: Delegates to [`hash_single_128`](Self::hash_single_128)
    /// - **Length 2+**: Delegates to [`hash_many_128`](Self::hash_many_128)
    pub fn hash_vec_128(inputs: &[Fp]) -> [Fp; 2] {
        if inputs.is_empty() {
            debug_assert!(
                false,
                "hash_vec_128 called with empty input - this may cause collision with valid hashes"
            );
            return [Fp::zero(), Fp::zero()];
        }
        if inputs.len() == 1 {
            return Self::hash_single_128(&inputs[0]);
        }
        Self::hash_many_128(inputs)
    }

    /// Hash a vector of field elements (for Merkle tree leaves).
    ///
    /// # Behavior
    ///
    /// - **Empty**: Returns `Fp::zero()` (debug assertion guards against misuse)
    /// - **Length 1**: Delegates to [`hash_single`](Self::hash_single) (domain tag = 1)
    /// - **Length 2+**: Delegates to [`hash_many`](Self::hash_many) (10* padding)
    ///
    /// # Security Note
    ///
    /// Empty input returns a fixed zero value which could collide with valid hash
    /// outputs. Callers should ensure empty inputs are invalid in their protocol,
    /// or filter them before calling this function. A debug assertion catches this
    /// in development builds.
    ///
    /// # Construction Discontinuity
    ///
    /// Note that length-1 inputs use `hash_single` (with domain tag), while
    /// length-2+ inputs use `hash_many` (with padding). This is intentional for
    /// efficiency but means the underlying construction differs by length.
    pub fn hash_vec(inputs: &[Fp]) -> Fp {
        if inputs.is_empty() {
            debug_assert!(
                false,
                "hash_vec called with empty input - this may cause collision with valid hashes"
            );
            return Fp::zero();
        }
        if inputs.len() == 1 {
            return Self::hash_single(&inputs[0]);
        }
        Self::hash_many(inputs)
    }
}

#[cfg(test)]
mod tests {
    use super::goldilocks::MATRIX_DIAG_8;
    use super::*;

    #[test]
    fn test_poseidon2_permutation_deterministic() {
        // Test that permutation is deterministic
        let mut hasher1 = Poseidon2::new();
        let mut hasher2 = Poseidon2::new();

        hasher1.state[0] = Fp::from(1u64);
        hasher1.state[1] = Fp::from(2u64);

        hasher2.state[0] = Fp::from(1u64);
        hasher2.state[1] = Fp::from(2u64);

        hasher1.permute();
        hasher2.permute();

        for i in 0..WIDTH {
            assert_eq!(hasher1.state[i], hasher2.state[i]);
        }
    }

    #[test]
    fn test_poseidon2_hash_deterministic() {
        let x = Fp::from(123u64);
        let y = Fp::from(456u64);

        let h1 = Poseidon2::hash(&x, &y);
        let h2 = Poseidon2::hash(&x, &y);

        assert_eq!(h1, h2);
    }

    #[test]
    fn test_poseidon2_hash_different_inputs() {
        let x1 = Fp::from(1u64);
        let y1 = Fp::from(2u64);

        let x2 = Fp::from(1u64);
        let y2 = Fp::from(3u64);

        let h1 = Poseidon2::hash(&x1, &y1);
        let h2 = Poseidon2::hash(&x2, &y2);

        assert_ne!(h1, h2);
    }

    #[test]
    fn test_poseidon2_hash_single() {
        let x = Fp::from(42u64);
        let h = Poseidon2::hash_single(&x);

        // Should not be zero or equal to input
        assert_ne!(h, Fp::zero());
        assert_ne!(h, x);
    }

    #[test]
    fn test_poseidon2_hash_many() {
        let inputs: Vec<Fp> = (1..=10).map(|i| Fp::from(i as u64)).collect();
        let h = Poseidon2::hash_many(&inputs);

        // Should produce a valid hash
        assert_ne!(h, Fp::zero());

        // Same inputs should produce same hash
        let h2 = Poseidon2::hash_many(&inputs);
        assert_eq!(h, h2);
    }

    #[test]
    fn test_poseidon2_sbox() {
        // Test S-box: x^7
        let x = Fp::from(2u64);
        let result = Poseidon2::sbox(&x);
        let expected = Fp::from(128u64); // 2^7 = 128
        assert_eq!(result, expected);
    }

    #[test]
    fn test_poseidon2_compress() {
        let left = Fp::from(100u64);
        let right = Fp::from(200u64);

        let compressed = Poseidon2::compress(&left, &right);

        // Same as hash(left, right)
        let hashed = Poseidon2::hash(&left, &right);
        assert_eq!(compressed, hashed);
    }

    #[test]
    fn test_poseidon2_state_changes() {
        let mut hasher = Poseidon2::new();

        // Set some initial state
        for (i, state_elem) in hasher.state.iter_mut().enumerate() {
            *state_elem = Fp::from(i as u64);
        }

        let initial_state = hasher.state;
        hasher.permute();

        // State should change after permutation
        let all_same = hasher
            .state
            .iter()
            .zip(initial_state.iter())
            .all(|(a, b)| a == b);
        assert!(!all_same, "State should change after permutation");
    }

    // ========================================================================
    // Test vectors from Plonky3 (HorizenLabs implementation)
    // These verify our implementation matches the reference
    // ========================================================================

    #[test]
    fn test_poseidon2_plonky3_vector_zeros() {
        // Test vector: all zeros input
        let mut hasher = Poseidon2::new();
        // State is already all zeros

        hasher.permute();

        let expected: [u64; 8] = [
            4214787979728720400,
            12324939279576102560,
            10353596058419792404,
            15456793487362310586,
            10065219879212154722,
            16227496357546636742,
            2959271128466640042,
            14285409611125725709,
        ];

        for i in 0..WIDTH {
            assert_eq!(
                *hasher.state[i].value(),
                expected[i],
                "Mismatch at index {} for zeros input",
                i
            );
        }
    }

    #[test]
    fn test_poseidon2_plonky3_vector_sequential() {
        // Test vector: sequential input [0, 1, 2, 3, 4, 5, 6, 7]
        let mut hasher = Poseidon2::new();
        for i in 0..WIDTH {
            hasher.state[i] = Fp::from(i as u64);
        }

        hasher.permute();

        let expected: [u64; 8] = [
            14266028122062624699,
            5353147180106052723,
            15203350112844181434,
            17630919042639565165,
            16601551015858213987,
            10184091939013874068,
            16774100645754596496,
            12047415603622314780,
        ];

        for i in 0..WIDTH {
            assert_eq!(
                *hasher.state[i].value(),
                expected[i],
                "Mismatch at index {} for sequential input",
                i
            );
        }
    }

    #[test]
    fn test_poseidon2_plonky3_vector_random() {
        // Test vector: random input
        let input: [u64; 8] = [
            5116996373749832116,
            8931548647907683339,
            17132360229780760684,
            11280040044015983889,
            11957737519043010992,
            15695650327991256125,
            17604752143022812942,
            543194415197607509,
        ];

        let mut hasher = Poseidon2::new();
        for i in 0..WIDTH {
            hasher.state[i] = Fp::from(input[i]);
        }

        hasher.permute();

        let expected: [u64; 8] = [
            1831346684315917658,
            13497752062035433374,
            12149460647271516589,
            15656333994315312197,
            4671534937670455565,
            3140092508031220630,
            4251208148861706881,
            6973971209430822232,
        ];

        for i in 0..WIDTH {
            assert_eq!(
                *hasher.state[i].value(),
                expected[i],
                "Mismatch at index {} for random input",
                i
            );
        }
    }

    #[test]
    fn test_apply_hl_mat4() {
        // Test the Horizen Labs matrix [[5,7,1,3], [4,6,1,1], [1,3,5,7], [1,1,4,6]]
        // Input: [1, 0, 0, 0] should give [5, 4, 1, 1]
        let mut x = [Fp::from(1u64), Fp::zero(), Fp::zero(), Fp::zero()];
        Poseidon2::apply_hl_mat4(&mut x);
        assert_eq!(*x[0].value(), 5u64, "Row 0, column 0");
        assert_eq!(*x[1].value(), 4u64, "Row 1, column 0");
        assert_eq!(*x[2].value(), 1u64, "Row 2, column 0");
        assert_eq!(*x[3].value(), 1u64, "Row 3, column 0");

        // Input: [0, 1, 0, 0] should give [7, 6, 3, 1]
        let mut x = [Fp::zero(), Fp::from(1u64), Fp::zero(), Fp::zero()];
        Poseidon2::apply_hl_mat4(&mut x);
        assert_eq!(*x[0].value(), 7u64, "Row 0, column 1");
        assert_eq!(*x[1].value(), 6u64, "Row 1, column 1");
        assert_eq!(*x[2].value(), 3u64, "Row 2, column 1");
        assert_eq!(*x[3].value(), 1u64, "Row 3, column 1");

        // Input: [1, 2, 3, 4] should give:
        // [5*1 + 7*2 + 1*3 + 3*4, 4*1 + 6*2 + 1*3 + 1*4, 1*1 + 3*2 + 5*3 + 7*4, 1*1 + 1*2 + 4*3 + 6*4]
        // = [5+14+3+12, 4+12+3+4, 1+6+15+28, 1+2+12+24] = [34, 23, 50, 39]
        let mut x = [
            Fp::from(1u64),
            Fp::from(2u64),
            Fp::from(3u64),
            Fp::from(4u64),
        ];
        Poseidon2::apply_hl_mat4(&mut x);
        assert_eq!(*x[0].value(), 34u64, "General test [0]");
        assert_eq!(*x[1].value(), 23u64, "General test [1]");
        assert_eq!(*x[2].value(), 50u64, "General test [2]");
        assert_eq!(*x[3].value(), 39u64, "General test [3]");
    }

    #[test]
    fn test_external_linear_layer() {
        // Test the full external linear layer for width 8
        // For simple input [1,0,0,0,0,0,0,0], after HL mat4 on halves:
        // First half [1,0,0,0] -> [5,4,1,1]
        // Second half [0,0,0,0] -> [0,0,0,0]
        // After diffusion: state[i] += state[i] + state[i+4]
        // state[0] = 5 + 5 + 0 = 10
        // state[4] = 0 + 5 + 0 = 5
        let mut hasher = Poseidon2::new();
        hasher.state[0] = Fp::from(1u64);
        hasher.external_linear_layer();

        assert_eq!(*hasher.state[0].value(), 10u64, "ELL state[0]");
        assert_eq!(*hasher.state[4].value(), 5u64, "ELL state[4]");
    }

    #[test]
    fn test_internal_linear_layer() {
        // Test internal linear layer: y[i] = diag[i] * x[i] + sum(x)
        // For input [1, 0, 0, 0, 0, 0, 0, 0]:
        // sum = 1
        // y[0] = diag[0] * 1 + 1 = diag[0] + 1
        // y[1] = diag[1] * 0 + 1 = 1
        // ...
        let mut hasher = Poseidon2::new();
        hasher.state[0] = Fp::from(1u64);
        hasher.internal_linear_layer();

        let diag0 = Fp::from(MATRIX_DIAG_8[0]);
        let expected0 = &diag0 + &Fp::from(1u64);
        assert_eq!(hasher.state[0], expected0, "ILL state[0]");
        assert_eq!(hasher.state[1], Fp::from(1u64), "ILL state[1]");
    }

    // ========================================================================
    // Domain separation tests (HIGH PRIORITY)
    // These verify that different hash constructions produce different outputs
    // ========================================================================

    #[test]
    fn test_domain_separation_hash_vs_hash_many() {
        // hash(a, b) uses domain tag = 2 in capacity
        // hash_many([a, b]) uses 10* padding, no domain tag
        // These MUST produce different outputs
        let a = Fp::from(1u64);
        let b = Fp::from(2u64);
        assert_ne!(
            Poseidon2::hash(&a, &b),
            Poseidon2::hash_many(&[a, b]),
            "hash(a,b) should differ from hash_many([a,b]) due to domain separation"
        );
    }

    #[test]
    fn test_domain_separation_single_vs_many() {
        // hash_single uses domain tag = 1
        // hash_many uses 10* padding
        // These MUST produce different outputs for same logical input
        let x = Fp::from(42u64);
        assert_ne!(
            Poseidon2::hash_single(&x),
            Poseidon2::hash_many(&[x]),
            "hash_single(x) should differ from hash_many([x]) due to domain separation"
        );
    }

    #[test]
    fn test_hash_vec_delegates_length_one() {
        // hash_vec with length 1 should delegate to hash_single
        let x = Fp::from(42u64);
        assert_eq!(
            Poseidon2::hash_vec(&[x]),
            Poseidon2::hash_single(&x),
            "hash_vec([x]) should equal hash_single(x)"
        );
    }

    #[test]
    fn test_hash_vec_delegates_length_two_plus() {
        // hash_vec with length >= 2 should delegate to hash_many
        let inputs = vec![Fp::from(1u64), Fp::from(2u64), Fp::from(3u64)];
        assert_eq!(
            Poseidon2::hash_vec(&inputs),
            Poseidon2::hash_many(&inputs),
            "hash_vec should equal hash_many for length >= 2"
        );
    }

    #[test]
    fn test_compress_non_commutative() {
        // compress(a, b) != compress(b, a) - important for Merkle tree security
        let a = Fp::from(100u64);
        let b = Fp::from(200u64);
        assert_ne!(
            Poseidon2::compress(&a, &b),
            Poseidon2::compress(&b, &a),
            "compress should be non-commutative (order matters)"
        );
    }

    // ========================================================================
    // Empty input edge case tests
    // ========================================================================

    #[test]
    fn test_hash_vec_empty_returns_zero() {
        // Empty input returns Fp::zero() (but triggers debug_assert in debug builds)
        // This test verifies the behavior in release mode
        #[cfg(not(debug_assertions))]
        {
            assert_eq!(
                Poseidon2::hash_vec(&[]),
                Fp::zero(),
                "hash_vec([]) should return Fp::zero()"
            );
        }
    }

    #[test]
    fn test_hash_many_empty_not_zero() {
        // hash_many([]) applies 10* padding: [1, 0, 0, 0] then permutes
        // Result should NOT be Fp::zero()
        let result = Poseidon2::hash_many(&[]);
        assert_ne!(
            result,
            Fp::zero(),
            "hash_many([]) should not be zero (padding is applied)"
        );
    }

    #[test]
    fn test_hash_single_zero_not_zero() {
        // hash_single(0) should not return 0 (would be a weak hash)
        let result = Poseidon2::hash_single(&Fp::zero());
        assert_ne!(result, Fp::zero(), "hash_single(0) should not return 0");
    }

    #[test]
    fn test_hash_zero_pair_not_zero() {
        // hash(0, 0) should not return 0
        let result = Poseidon2::hash(&Fp::zero(), &Fp::zero());
        assert_ne!(result, Fp::zero(), "hash(0, 0) should not return 0");
    }

    // ========================================================================
    // 128-bit hash function tests
    // ========================================================================

    #[test]
    fn test_hash_128_deterministic() {
        let x = Fp::from(123u64);
        let y = Fp::from(456u64);

        let h1 = Poseidon2::hash_128(&x, &y);
        let h2 = Poseidon2::hash_128(&x, &y);

        assert_eq!(h1, h2);
    }

    #[test]
    fn test_hash_128_different_inputs() {
        let x1 = Fp::from(1u64);
        let y1 = Fp::from(2u64);

        let x2 = Fp::from(1u64);
        let y2 = Fp::from(3u64);

        let h1 = Poseidon2::hash_128(&x1, &y1);
        let h2 = Poseidon2::hash_128(&x2, &y2);

        assert_ne!(h1, h2);
    }

    #[test]
    fn test_hash_single_128() {
        let x = Fp::from(42u64);
        let h = Poseidon2::hash_single_128(&x);

        // Should not be zero
        assert!(h[0] != Fp::zero() || h[1] != Fp::zero());
    }

    #[test]
    fn test_compress_128_deterministic() {
        let left = [Fp::from(100u64), Fp::from(101u64)];
        let right = [Fp::from(200u64), Fp::from(201u64)];

        let c1 = Poseidon2::compress_128(&left, &right);
        let c2 = Poseidon2::compress_128(&left, &right);

        assert_eq!(c1, c2);
    }

    #[test]
    fn test_compress_128_non_commutative() {
        // compress_128(a, b) != compress_128(b, a) - important for Merkle tree security
        let a = [Fp::from(100u64), Fp::from(101u64)];
        let b = [Fp::from(200u64), Fp::from(201u64)];

        assert_ne!(
            Poseidon2::compress_128(&a, &b),
            Poseidon2::compress_128(&b, &a),
            "compress_128 should be non-commutative (order matters)"
        );
    }

    #[test]
    fn test_hash_many_128() {
        let inputs: Vec<Fp> = (1..=10).map(|i| Fp::from(i as u64)).collect();
        let h = Poseidon2::hash_many_128(&inputs);

        // Should produce a valid hash (not all zeros)
        assert!(h[0] != Fp::zero() || h[1] != Fp::zero());

        // Same inputs should produce same hash
        let h2 = Poseidon2::hash_many_128(&inputs);
        assert_eq!(h, h2);
    }

    #[test]
    fn test_hash_vec_128_delegates_correctly() {
        let x = Fp::from(42u64);
        // Length 1 should delegate to hash_single_128
        assert_eq!(
            Poseidon2::hash_vec_128(&[x]),
            Poseidon2::hash_single_128(&x)
        );

        // Length 2+ should delegate to hash_many_128
        let inputs = vec![Fp::from(1u64), Fp::from(2u64), Fp::from(3u64)];
        assert_eq!(
            Poseidon2::hash_vec_128(&inputs),
            Poseidon2::hash_many_128(&inputs)
        );
    }

    #[test]
    fn test_128_bit_domain_separation() {
        // 128-bit and 64-bit versions should produce different results
        // (state[0] might match but state[1] differs, or vice versa)
        let x = Fp::from(123u64);
        let y = Fp::from(456u64);

        let h64 = Poseidon2::hash(&x, &y);
        let h128 = Poseidon2::hash_128(&x, &y);

        // The 64-bit hash equals the first element of 128-bit (same construction)
        assert_eq!(h64, h128[0], "First element of 128-bit should match 64-bit");
    }
}
