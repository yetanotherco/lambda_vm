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
use lambda_vm_prover::lfm::registry::build_artifacts;

/// Blowups registered in v0 (extend alongside `STATIC_BLOWUP_FACTORS` when
/// other presets come online).
const REGISTRY_BLOWUP_FACTORS: &[u8] = &[2];

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
        for &blowup in REGISTRY_BLOWUP_FACTORS {
            let options = GoldilocksCubicProofOptions::with_blowup(blowup).expect("proof options");
            let artifacts = build_artifacts(program, &options);
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
            println!("        program_id: {},", fmt_bytes(&artifacts.program_id));
            println!("    }},");
        }
    }
    println!("];");
}
