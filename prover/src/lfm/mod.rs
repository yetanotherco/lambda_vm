//! LFM — the Lambda Field Machine.
//!
//! A fixed, straight-line, field-native recursion machine for verifying
//! Lambda VM STARK proofs: the SP1 v4 mechanism (the program is the machine's
//! preprocessed columns; write-once memory closed by pure LogUp balance; no
//! pc, no branches, no fetch/decode) with on-demand registration instead of
//! exhaustive shape enumeration — our framework has no keygen, so a program
//! is nothing but a vector of supplied preprocessed roots plus a registry
//! entry.
//!
//! Design authority: `others/lfm-design.md` (v0). This module is the
//! software layer (Milestone A): word model, instruction set, eDSL builder,
//! straight-line compiler, executor/witness generator, admission validator,
//! and the hash interface with a placeholder permutation. The chips and prover
//! integration follow (Milestone B); the fixed AIR set is 14 chips, the last
//! three being the production keccak family hosted unchanged (see `airs`).

pub mod airs;
pub mod batched_epoch;
pub mod blake3;
pub mod blake3_chip;
pub mod blake3_socket;
pub mod builder;
pub mod chips;
pub mod chunking;
pub mod commit;
pub mod compiler;
pub mod constraints;
pub mod deep;
pub mod edsl;
pub mod epoch;
pub mod epoch_verify;
pub mod executor;
pub mod fixture;
pub mod fri;
pub mod hash;
pub mod instr;
pub mod keccak_adapter;
pub mod keccak_host;
pub mod layout;
pub mod lde;
pub mod logup;
pub mod poseidon;
pub mod programs;
pub mod proof;
pub mod proof_arena;
pub mod proof_fixture;
pub mod registry;
pub mod statement;
pub mod statement_replay;
pub mod sub_proof;
pub mod trace;
pub mod transcript_replay;
pub mod validator;
pub mod word;

pub use airs::{LfmAirs, NUM_LFM_CHIPS, num_lfm_airs};
pub use builder::{ArenaSchema, LfmBuilder, LfmProgramSource};
pub use chunking::{KECCAK_RND_MAX_CHUNK_ROWS, KeccakChunking};
pub use commit::{PrepRoundBuilder, commit_columns, commit_group, commit_lde_columns, lde_columns};
pub use compiler::{ColumnGroup, LfmColumnGroups, LfmProgram, compile};
pub use executor::{LfmExecError, LfmExecution, LfmRecords, execute};
pub use hash::{HasherKind, LfmHasher, TestPermutation};
pub use instr::{Addr, ArenaId, BaseOp, ExtOp, HashMode, Instr};
pub use proof::{LfmProof, LfmProveError, lfm_prove, lfm_verify, verify_against_artifacts};
pub use registry::{
    LFM_REGISTRY, LfmArtifacts, LfmProgramKind, LfmRegistryEntry, LfmRegistryError,
    build_artifacts, build_artifacts_with_hasher, resolve,
};
pub use statement::{LFM_MACHINE_VERSION, lfm_program_id};
pub use transcript_replay::{Candidate, TranscriptReplay};
pub use validator::{LfmViolation, validate};
pub use word::{LfmWord, base_word, ext_word, pack_digest, unpack_digest};

#[cfg(test)]
mod blake3_chip_tests;
#[cfg(test)]
mod blake3_probe;
#[cfg(test)]
mod blake3_socket_kats;
#[cfg(test)]
mod blake3_socket_tests;
#[cfg(test)]
mod constraint_tests;
#[cfg(test)]
mod epoch_tests;
#[cfg(test)]
mod epoch_verify_tests;
#[cfg(test)]
mod framework_probe;
#[cfg(test)]
mod fri_tests;
#[cfg(test)]
mod join_tests;
#[cfg(test)]
mod keccak_probe;
#[cfg(test)]
mod leaf_kats;
#[cfg(test)]
mod leaf_tests;
#[cfg(test)]
mod logup_tests;
#[cfg(test)]
mod machine_tests;
#[cfg(test)]
mod poseidon_chip_tests;
#[cfg(test)]
mod step_size_tests;
#[cfg(test)]
mod tests;
#[cfg(test)]
mod transcript_kats;
#[cfg(test)]
mod transcript_tests;
#[cfg(test)]
mod wrap_tests;
