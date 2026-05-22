//! Fiat-Shamir transcripts for the STARK prove/verify protocols.
//!
//! A transcript absorbs bytes and field elements and produces uniformly
//! random challenges (field elements / indices) derived from everything
//! absorbed so far.

pub mod default_transcript;
pub mod is_transcript;
