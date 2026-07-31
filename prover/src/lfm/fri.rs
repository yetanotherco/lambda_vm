//! FRI: the compile-time shape of the emitted verifier.
//!
//! Slice 1 of the FRI leg — the arithmetic only. `others/lfm-fri-verify-spec.md`
//! is the verified account of the production verify path this mirrors; §2 is
//! the section this file implements.
//!
//! ## Why the shape is a struct and not a runtime computation
//!
//! Production derives the fold layout at verify time from the AIR's options and
//! domain (`FriFoldLayout::new`, `fri/terminal.rs:45`). The machine cannot: it
//! is straight-line, so the layer count fixes how many walks and folds are
//! EMITTED. Every field below is therefore program shape in the sense of
//! `others/lfm-target-shape.md`, and a program that read any of it from an
//! arena would let the prover choose how much FRI to verify — the degenerate
//! case being "none".
//!
//! ## What this module is checked against
//!
//! `FriFoldLayout` is `pub(crate)` inside `crypto/stark`, so this mirror cannot
//! be differentialled against the struct itself. The oracle is production's
//! observable BEHAVIOUR instead — the vector lengths a real proof carries and
//! the verifier structurally enforces before its query loop
//! (`verifier.rs:426-448`): `fri_layers_merkle_roots.len() == num_committed`
//! and `fri_final_poly_coeffs.len() == 1 << effective_k`. That is a stronger
//! check than reading the struct would be, because those are the lengths the
//! verifier actually rejects on.
//!
//! ## What this cannot see
//!
//! It is arithmetic over a shape; it says nothing about whether the emitted
//! walk or fold is correct, only about how many of each there should be. It
//! also mirrors the CPU layout only — `fri/mod.rs` has cuda fast paths that
//! claim the same layout, unverified here and never run by the machine.

use stark::proof::options::ProofOptions;

/// The compile-time shape of one sub-proof's FRI verification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FriShape {
    /// `log2` of the LDE (deep-composition) codeword length.
    pub log2_lde_length: u32,
    /// `log2` of the blowup factor.
    pub blowup_log: u32,
    /// The requested terminal log-degree, `ProofOptions::fri_final_poly_log_degree`.
    pub final_poly_log_degree: u32,
    /// The LDE coset offset. Carried rather than assumed: the emitter bakes
    /// domain constants derived from it, and the standing deferral on
    /// `coset_offset != 3` is about test COVERAGE, not about a hardcoded 3 —
    /// so the value has to come from the options, and [`Self::from_options`] is
    /// the only constructor that reads it.
    pub coset_offset: u64,
    /// Queries the sub-proof carries.
    pub num_queries: usize,
}

impl FriShape {
    /// Derive the shape from the inner proof's own options.
    ///
    /// Every FRI-relevant parameter comes from `options` — including the coset
    /// offset, which discharges the plumbing half of the `coset_offset != 3`
    /// deferral recorded in `others/lfm-assembly-obligations.md`.
    pub fn from_options(options: &ProofOptions, log2_lde_length: u32) -> Self {
        Self {
            log2_lde_length,
            blowup_log: (options.blowup_factor as u32).trailing_zeros(),
            final_poly_log_degree: options.fri_final_poly_log_degree as u32,
            coset_offset: options.coset_offset,
            num_queries: options.fri_number_of_queries,
        }
    }

    /// `log2` of the terminal codeword length, clamped to the full LDE for
    /// traces too small to fold that far (`terminal.rs:46`'s `.min(lde_log)`).
    pub fn terminal_log(self) -> u32 {
        (self.blowup_log + self.final_poly_log_degree).min(self.log2_lde_length)
    }

    /// Folds from the LDE codeword down to the terminal codeword.
    pub fn total_folds(self) -> u32 {
        self.log2_lde_length - self.terminal_log()
    }

    /// Committed (Merkle-rooted) layers — one root, one auth path per query,
    /// and one Merkle walk to emit, each.
    ///
    /// **`total_folds − 1`, not `total_folds`.** The final fold is performed
    /// and never committed (`fri/mod.rs:114-118`), so a query folds once more
    /// than it authenticates. This off-by-one is the readiest way to build a
    /// verifier that looks right and checks one layer too few.
    pub fn num_committed(self) -> usize {
        self.total_folds().saturating_sub(1) as usize
    }

    /// Folds a query performs: `num_committed + 1` whenever anything folds at
    /// all, and 0 when the codeword is already terminal.
    pub fn num_folds(self) -> usize {
        self.total_folds() as usize
    }

    /// Terminal codeword length.
    pub fn terminal_len(self) -> usize {
        1usize << self.terminal_log()
    }

    /// The terminal log-degree actually used — `min(k, trace_bits)`. Equals
    /// `final_poly_log_degree` except under the clamp.
    pub fn effective_k(self) -> u32 {
        self.terminal_log() - self.blowup_log
    }

    /// Coefficients the proof carries for the terminal polynomial.
    pub fn num_terminal_coeffs(self) -> usize {
        1usize << self.effective_k()
    }

    /// Merkle path length for committed layer `i`: that layer's codeword is
    /// `2^(n−i−1)` long and its leaves are pairs, so the tree has `2^(n−i−2)`
    /// leaves.
    pub fn layer_path_len(self, layer: usize) -> usize {
        (self.log2_lde_length as usize)
            .checked_sub(layer + 2)
            .expect("layer index must be below num_committed")
    }

    /// Merkle path steps one query walks across every committed layer.
    pub fn path_steps_per_query(self) -> usize {
        (0..self.num_committed())
            .map(|i| self.layer_path_len(i))
            .sum()
    }

    /// Keccak permutations one query costs: one leaf hash per committed layer
    /// (a 48-byte pair, one rate block) plus one per path step (64 bytes, one
    /// rate block).
    pub fn permutations_per_query(self) -> usize {
        self.num_committed() + self.path_steps_per_query()
    }

    /// Keccak permutations the whole sub-proof's FRI costs.
    pub fn permutations(self) -> usize {
        self.num_queries * self.permutations_per_query()
    }

    /// Invariants a caller cannot assemble their way out of.
    pub fn check(self) {
        assert!(
            self.blowup_log >= 1,
            "a blowup of 1 is not a low-degree extension"
        );
        assert!(
            self.log2_lde_length > self.blowup_log,
            "the LDE must be strictly larger than the blowup: a trace of one \
             row has no FRI to do"
        );
        assert!(
            self.terminal_log() <= self.log2_lde_length,
            "the terminal codeword cannot exceed the LDE"
        );
        assert!(
            self.effective_k() <= self.final_poly_log_degree,
            "the clamp can only lower the terminal degree, never raise it"
        );
        assert_eq!(
            self.terminal_len(),
            1usize << (self.blowup_log + self.effective_k()),
            "terminal_len must equal 2^(blowup_log + effective_k)"
        );
    }
}
