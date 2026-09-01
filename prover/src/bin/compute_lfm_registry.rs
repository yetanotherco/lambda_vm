//! Regenerates the `LFM_REGISTRY` constant table.
//!
//! Usage: `cargo run --bin compute_lfm_registry --release`, then paste the
//! output over the generated block in `prover/src/lfm/registry.rs`. Drift
//! tests recompute and compare on every PR; a drift failure is investigated,
//! never re-blessed (the `compute_static_commitments` policy).

use lambda_vm_prover::GoldilocksCubicProofOptions;
use lambda_vm_prover::lfm::programs::{
    KECCAK_SPONGE_LEN, fri_toy_program, keccak_chain_program, keccak_sponge_program,
    statement_replay_program, transcript_replay_program, trivial_program,
};
use lambda_vm_prover::lfm::registry::build_artifacts_with_hasher;
use lambda_vm_prover::lfm::validate;

/// Blowups registered in v0 (extend alongside `STATIC_BLOWUP_FACTORS` when
/// other presets come online).
const REGISTRY_BLOWUP_FACTORS: &[u8] = &[2];

// The permutation this table is blessed under is `registry::REGISTRY_HASHER` —
// a property of the TABLE rather than of this generator, and the same constant
// `build_artifacts` defaults to, so the two cannot drift apart.
use lambda_vm_prover::lfm::registry::REGISTRY_HASHER;

fn fmt_bytes(bytes: &[u8; 32]) -> String {
    let inner = bytes
        .iter()
        .map(|b| format!("{b:#04x}"))
        .collect::<Vec<_>>()
        .join(", ");
    format!("[{inner}]")
}

fn main() {
    let programs = [
        ("TrivialV0", trivial_program()),
        ("FriToyV0", fri_toy_program()),
        ("KeccakChainV0", keccak_chain_program()),
        ("KeccakSpongeV0", keccak_sponge_program(KECCAK_SPONGE_LEN)),
        ("TranscriptReplayV0", transcript_replay_program()),
        ("StatementReplayV0", statement_replay_program()),
    ];
    println!("pub static LFM_REGISTRY: &[LfmRegistryEntry] = &[");
    for (kind, program) in &programs {
        // A program digest enters the registry only after admission passes —
        // the gate `validator.rs` declares, wired here rather than left to the
        // convention that every kind also has a hand-written admissibility test.
        validate(program).unwrap_or_else(|v| panic!("{kind} is not admissible: {v:?}"));
        for &blowup in REGISTRY_BLOWUP_FACTORS {
            let options = GoldilocksCubicProofOptions::with_blowup(blowup).expect("proof options");
            let artifacts = build_artifacts_with_hasher(program, &options, REGISTRY_HASHER);
            println!("    LfmRegistryEntry {{");
            println!("        kind: LfmProgramKind::{kind},");
            println!("        blowup_factor: {blowup},");
            println!("        roots: [");
            for root in &artifacts.roots {
                println!("            {},", fmt_bytes(root));
            }
            println!("        ],");
            let heights = artifacts
                .log_heights
                .iter()
                .map(u8::to_string)
                .collect::<Vec<_>>()
                .join(", ");
            println!("        log_heights: [{heights}],");
            println!(
                "        keccak_rnd_chunks: {},",
                artifacts.keccak_rnd_chunks
            );
            println!("        hasher: HasherKind::{:?},", artifacts.hasher);
            println!(
                "        chip_set: ChipSet {{ keccak: {}, blake3: {} }},",
                artifacts.chip_set.keccak, artifacts.chip_set.blake3
            );
            println!("        program_id: {},", fmt_bytes(&artifacts.program_id));
            println!("        prep_root: {},", fmt_bytes(&artifacts.prep_root));
            let widths = artifacts
                .prep_widths
                .iter()
                .map(u16::to_string)
                .collect::<Vec<_>>()
                .join(", ");
            println!("        prep_widths: [{widths}],");
            println!("    }},");
        }
    }
    println!("];");
}
