//! D2 device parity at the proof level, on a VALID oracle (lane I-FIX-D2).
//!
//! The one-row RV64 VM bytes test this replaces compared a CPU-build proof with
//! a cuda-build proof of `test_mul_8`, but an RV64 VM proof is not a function
//! of the ELF and the format alone: six base-table builders dedup through a std
//! `HashMap` (`RandomState`) and lay rows out in iteration order, so the main
//! roots — and with them the whole transcript — change from process to process
//! (the lead's control: the same build differs from itself). A cross-build
//! `cmp` of such bytes means nothing.
//!
//! This test proves an LFM machine program instead: its trace is a function of
//! the program and the arenas, the proof is made at grinding 0 (no host nonce
//! search), and each format is proved TWICE in the same process with the two
//! byte strings asserted equal (the in-run determinism control), so a
//! cross-build `cmp` of the written files means "the device proof is the CPU
//! proof". The program (`TrivialV0`) has public outputs, so the statement the
//! transcript absorbs — and the LogUp balance through `z`, `α` — depends on the
//! proof actually being the one the verifier replays.
//!
//! Files: `$ZF_S2_PROOF_DIR/{cpu,cuda}_{format}.rkyv` for `legacy` (every lever
//! off), `one_row_1` (legacy + one row on every chip) and `production` (the
//! measured configuration of RULINGS 26: cap auto, fri dp, one_row auto).
//!
//! Under cuda the `one_row_1` proof must build one-row trees on the device and
//! take the one-row device FRI commit (a silent host fallback would still give
//! equal bytes, so the counters are what make the comparison mean "device");
//! run with `LAMBDA_VM_GPU_LDE_THRESHOLD` low enough that the LFM tables cross
//! it (the box line sets 1024).

use stark::proof::options::{FriMode, OneRowMode, ProofFormat, ProofOptions};

use crate::lfm::proof::{lfm_prove, verify_against_artifacts};
use crate::lfm::registry::{LfmProgramKind, build_artifacts};
use crate::lfm::word::LfmWord;
use crate::tables::types::FE;

/// The three formats compared across builds.
fn formats() -> [(&'static str, ProofFormat); 3] {
    let legacy = ProofFormat {
        merkle_cap: crypto::merkle_tree::cap::CapPolicy::Off,
        fri_mode: FriMode::Pair,
        one_row: OneRowMode::Off,
        fri_schedule_override: None,
    };
    [
        ("legacy", legacy),
        (
            "one_row_1",
            ProofFormat {
                one_row: OneRowMode::On,
                ..legacy
            },
        ),
        (
            "production",
            ProofFormat {
                merkle_cap: crypto::merkle_tree::cap::CapPolicy::Auto,
                fri_mode: FriMode::Dp,
                one_row: OneRowMode::Auto,
                fri_schedule_override: None,
            },
        ),
    ]
}

fn options(format: ProofFormat) -> ProofOptions {
    // Blowup 4 (the blowup the one-row static twins ship for), 128-bit
    // queries with NO grinding: the proof is then a function of the program,
    // the arenas and the format.
    let mut o = stark::proof::options::GoldilocksCubicProofOptions::with_params(4, 128, 0)
        .expect("valid options");
    assert_eq!(o.grinding_factor, 0);
    o.format = format;
    o
}

fn arenas() -> Vec<Vec<LfmWord>> {
    vec![
        (0..4u64)
            .map(|i| core::array::from_fn(|j| FE::from(1_000 * (i + 1) + j as u64)))
            .collect(),
    ]
}

#[test]
#[ignore = "box: set ZF_S2_PROOF_DIR, run twice in a CPU build and twice in a cuda build, then cmp the files"]
fn lfm_proof_bytes_for_the_device_comparison() {
    let dir = std::env::var("ZF_S2_PROOF_DIR").expect("set ZF_S2_PROOF_DIR");
    std::fs::create_dir_all(&dir).expect("create ZF_S2_PROOF_DIR");
    let build = if cfg!(feature = "cuda") {
        "cuda"
    } else {
        "cpu"
    };
    let kind = LfmProgramKind::TrivialV0;
    let program = kind.program();
    let arenas = arenas();
    for (name, format) in formats() {
        let o = options(format);
        let artifacts = build_artifacts(&program, &o);
        match format.one_row {
            OneRowMode::On => assert!(artifacts.one_row_roots.is_some(), "{name}: one-row roots"),
            OneRowMode::Off => assert!(artifacts.one_row_roots.is_none(), "{name}: no one-row roots"),
            _ => {}
        }
        #[cfg(feature = "cuda")]
        let (trees0, fri0) = (
            stark::gpu_lde::gpu_one_row_trees(),
            stark::gpu_lde::gpu_one_row_fri_calls(),
        );
        let mut runs: Vec<Vec<u8>> = Vec::with_capacity(2);
        let mut last = None;
        for _ in 0..2 {
            let proved = lfm_prove(&program, &artifacts, &arenas, &o)
                .unwrap_or_else(|e| panic!("{name}: the LFM program must prove: {e:?}"));
            assert!(
                !proved.public_words.is_empty(),
                "{name}: the program publishes words"
            );
            runs.push(
                rkyv::to_bytes::<rkyv::rancor::Error>(&proved.proof)
                    .expect("rkyv")
                    .to_vec(),
            );
            last = Some(proved);
        }
        #[cfg(feature = "cuda")]
        let (trees, fri) = (
            stark::gpu_lde::gpu_one_row_trees() - trees0,
            stark::gpu_lde::gpu_one_row_fri_calls() - fri0,
        );
        let proved = last.expect("proved twice");
        assert_eq!(
            runs[0], runs[1],
            "{name}: the same LFM proof, proved twice in one process, must be byte-identical \
             (otherwise a cross-build cmp is not an oracle)"
        );
        let bytes = &runs[0];
        let path = std::path::Path::new(&dir).join(format!("{build}_{name}.rkyv"));
        std::fs::write(&path, bytes).expect("write the proof bytes");
        // After the file is written, so a verify failure still leaves the
        // bytes for the cross-build cmp. The balance depends on z, α through
        // the published words, so this is the Phase-A replay too.
        assert!(
            verify_against_artifacts(&artifacts, &proved.proof, &proved.public_words, &o),
            "{name}: an honest LFM proof must verify"
        );

        let tables: Vec<String> = proved
            .proof
            .proofs
            .iter()
            .map(|p| {
                let one_row = p.deep_poly_openings[0]
                    .main_trace_polys
                    .evaluations_sym
                    .is_empty();
                format!("{}{}", p.trace_length, if one_row { "r" } else { "p" })
            })
            .collect();
        let one_row_tables = tables.iter().filter(|t| t.ends_with('r')).count();
        println!(
            "ZF LFMBYTES {build} {name}: {} bytes, twice equal, {one_row_tables} of {} tables one-row \
             [rows: {}] -> {}",
            bytes.len(),
            tables.len(),
            tables.join(" "),
            path.display()
        );
        if format.one_row == OneRowMode::On {
            assert_eq!(one_row_tables, tables.len(), "{name}: every chip one-row");
        }
        if format.one_row == OneRowMode::Off {
            assert_eq!(one_row_tables, 0, "{name}: no chip one-row");
        }
        #[cfg(feature = "cuda")]
        {
            println!(
                "ZF LFM DEVICE {name}: {trees} one-row device trees, {fri} one-row device FRI \
                 commits (two proofs)"
            );
            if format.one_row == OneRowMode::On {
                assert!(
                    trees > 0 && fri > 0,
                    "{name}: no one-row tree or FRI commit reached the device \
                     ({trees} trees, {fri} FRI commits): the proof would be a host proof \
                     (lower LAMBDA_VM_GPU_LDE_THRESHOLD)"
                );
            }
            if format.one_row == OneRowMode::Off {
                assert_eq!(trees + fri, 0, "{name}: no one-row device work without one row");
            }
            println!(
                "ZF LFM DEVMEM {name}: largest one-row tree {} B; device fallbacks {}; \
                 reserved high water {} B",
                stark::gpu_lde::gpu_one_row_tree_peak_bytes(),
                math_cuda::device::device_fallbacks(),
                math_cuda::device::reserved_high_water()
            );
        }
    }
}
