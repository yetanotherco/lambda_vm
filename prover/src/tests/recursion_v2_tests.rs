//! Host-side soundness tests for the v2 (`attest-commitment-id`) recursion
//! attestation — both the monolithic and continuation paths. Each mirrors the
//! guest exactly (encode the v2 blob, run `verify_and_attest_blob_v2` /
//! `verify_continuation_and_attest_v2`, then the consumer's recompute+compare)
//! without running the in-VM verifier. The tamper cases cover the identity
//! binding: a corrupted DECODE or page-genesis commitment, or a lying entry
//! point, must fail to both verify AND bind to the trusted ELF.

use crate::recursion::{self, MIN_PROOF_OPTIONS};

fn read_guest_elf(name: &str) -> Vec<u8> {
    let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("workspace root")
        .to_path_buf();
    let path = root.join(format!("executor/program_artifacts/recursion/{name}.elf"));
    std::fs::read(&path).unwrap_or_else(|e| {
        panic!(
            "read {} — run `make compile-recursion-elfs`: {e}",
            path.display()
        )
    })
}

// ============================================================================
// Monolithic v2
// ============================================================================

/// True iff the v2 monolithic blob both verifies AND binds to `trusted_elf`
/// (the guest accepts and the consumer's recompute matches) — the exact
/// accept-and-bind predicate the tamper cases must break.
fn mono_accepts_and_binds(
    blob: &[u8],
    trusted_elf: &[u8],
    opts: &stark::proof::options::ProofOptions,
) -> bool {
    match recursion::verify_and_attest_blob_v2(blob, opts) {
        Ok(Some(att)) => matches!(
            recursion::check_attestation_v2(&att, trusted_elf, opts),
            Ok(Some(_))
        ),
        _ => false,
    }
}

/// Build a v2 monolithic blob from parts (so a test can corrupt any field).
fn mono_blob_from_parts(
    proof: &crate::VmProof,
    inner_elf: &[u8],
    elf_digest: [u8; 32],
    entry_point: u64,
    decode_commitment: crate::Commitment,
    page_commitments: Vec<(u64, crate::Commitment)>,
) -> Vec<u8> {
    crate::encode_recursion_input_v2(&crate::GuestInputV2 {
        vm_proof: proof.clone(),
        inner_elf: inner_elf.to_vec(),
        elf_digest,
        entry_point,
        decode_commitment,
        page_commitments,
    })
    .expect("encode v2 blob")
}

#[test]
fn v2_monolithic_accepts_and_binds() {
    let elf = read_guest_elf("empty");
    let opts = MIN_PROOF_OPTIONS;
    let proof =
        crate::prove_with_options_and_inputs(&elf, &[], &opts, &crate::MaxRowsConfig::default())
            .expect("inner prove");

    // The production blob (no ELF bytes).
    let blob = recursion::encode_guest_input_v2(&proof, &elf, &opts, false).expect("encode");
    let att = recursion::verify_and_attest_blob_v2(&blob, &opts)
        .expect("verify errored")
        .expect("v2 guest must accept the honest proof");
    let out = recursion::check_attestation_v2(&att, &elf, &opts)
        .expect("check errored")
        .expect("v2 attestation must bind to the trusted ELF");
    assert_eq!(out, proof.public_output, "attested output must match");

    // Embedding the ELF bytes (measurement blob) must not change the attestation
    // — the ELF is never consumed on the v2 path — but it does grow the blob.
    let blob_embed = recursion::encode_guest_input_v2(&proof, &elf, &opts, true).expect("encode");
    let att_embed = recursion::verify_and_attest_blob_v2(&blob_embed, &opts)
        .expect("verify errored")
        .expect("embed blob must accept");
    assert_eq!(att, att_embed, "embed_elf must not change the attestation");
    assert!(
        blob_embed.len() > blob.len(),
        "embedding the ELF must grow the blob (blob-size saving is real)"
    );
}

#[test]
fn v2_monolithic_rejects_wrong_trusted_elf() {
    let elf = read_guest_elf("empty");
    let other = read_guest_elf("fibonacci");
    let opts = MIN_PROOF_OPTIONS;
    let proof =
        crate::prove_with_options_and_inputs(&elf, &[], &opts, &crate::MaxRowsConfig::default())
            .expect("inner prove");
    let blob = recursion::encode_guest_input_v2(&proof, &elf, &opts, false).expect("encode");
    let att = recursion::verify_and_attest_blob_v2(&blob, &opts)
        .expect("verify errored")
        .expect("accept");
    // A consumer trusting a DIFFERENT program must not accept this attestation.
    assert!(
        matches!(
            recursion::check_attestation_v2(&att, &other, &opts),
            Ok(None)
        ),
        "attestation must not bind to a different trusted ELF"
    );
}

#[test]
fn v2_monolithic_tamper_suite() {
    let elf = read_guest_elf("empty");
    let opts = MIN_PROOF_OPTIONS;
    let proof =
        crate::prove_with_options_and_inputs(&elf, &[], &opts, &crate::MaxRowsConfig::default())
            .expect("inner prove");
    let elf_parsed = executor::elf::Elf::load(&elf).expect("load");
    let digest = crate::statement::elf_digest(&elf);
    let (decode, pages) = recursion::precomputed_commitments(&elf, &opts).expect("roots");

    // Sanity: the honest reconstruction accepts and binds.
    let honest = mono_blob_from_parts(
        &proof,
        &[],
        digest,
        elf_parsed.entry_point,
        decode,
        pages.clone(),
    );
    assert!(
        mono_accepts_and_binds(&honest, &elf, &opts),
        "honest hand-built v2 blob must accept and bind"
    );

    // Tamper 1: corrupt the DECODE commitment.
    let mut bad_decode = decode;
    bad_decode[0] ^= 0xFF;
    let blob = mono_blob_from_parts(
        &proof,
        &[],
        digest,
        elf_parsed.entry_point,
        bad_decode,
        pages.clone(),
    );
    assert!(
        !mono_accepts_and_binds(&blob, &elf, &opts),
        "corrupted DECODE commitment must reject"
    );

    // Tamper 2: corrupt a page-genesis commitment.
    assert!(!pages.is_empty(), "empty program must have >=1 ELF page");
    let mut bad_pages = pages.clone();
    bad_pages[0].1[0] ^= 0xFF;
    let blob = mono_blob_from_parts(
        &proof,
        &[],
        digest,
        elf_parsed.entry_point,
        decode,
        bad_pages,
    );
    assert!(
        !mono_accepts_and_binds(&blob, &elf, &opts),
        "corrupted page commitment must reject"
    );

    // Tamper 3: lie about the entry point — it feeds the REGISTER preprocessed
    // commitment, so a wrong value must break verification (or, at minimum, not
    // bind to the true ELF).
    let blob = mono_blob_from_parts(
        &proof,
        &[],
        digest,
        elf_parsed.entry_point.wrapping_add(4),
        decode,
        pages.clone(),
    );
    assert!(
        !mono_accepts_and_binds(&blob, &elf, &opts),
        "lying entry_point must reject or fail to bind"
    );

    // Tamper 4: a 1-byte flip anywhere in the honest blob's proof region must
    // not still accept-and-bind. Flip a byte near the end (inside the proof
    // payload, past the prefix + metadata).
    let mut corrupt = honest.clone();
    let idx = corrupt.len() - 32;
    corrupt[idx] ^= 0x01;
    assert!(
        !mono_accepts_and_binds(&corrupt, &elf, &opts),
        "a 1-byte proof-region flip must not still accept-and-bind"
    );

    // Cross-scheme: a v1 blob must be rejected by the v2 verifier (version tag).
    let v1_blob = recursion::encode_guest_input(&proof, &elf, &opts).expect("v1 encode");
    assert!(
        recursion::verify_and_attest_blob_v2(&v1_blob, &opts).is_err(),
        "v2 verifier must reject a v1 blob on the version tag"
    );
}

/// Blob-size delta measurement (v1 vs v2, embed_elf=false): the v2 blob drops the
/// inner ELF bytes and adds only entry_point(8) + elf_digest(32). Prints exact
/// sizes for the `empty` inner and projects the delta for an arbitrary inner ELF
/// size (e.g. ethrex). Run with `--ignored --nocapture`.
#[test]
#[ignore = "measurement: prints v1 vs v2 blob sizes"]
fn v2_blob_size_delta() {
    let elf = read_guest_elf("empty");
    let opts = MIN_PROOF_OPTIONS;
    let proof =
        crate::prove_with_options_and_inputs(&elf, &[], &opts, &crate::MaxRowsConfig::default())
            .expect("inner prove");
    let v1 = recursion::encode_guest_input(&proof, &elf, &opts).expect("v1");
    let v2_no_elf =
        recursion::encode_guest_input_v2(&proof, &elf, &opts, false).expect("v2 no elf");
    let v2_elf = recursion::encode_guest_input_v2(&proof, &elf, &opts, true).expect("v2 elf");
    eprintln!("[blob-size] inner=empty  elf={} bytes", elf.len());
    eprintln!(
        "[blob-size]   v1 (elf embedded)        : {} bytes",
        v1.len()
    );
    eprintln!(
        "[blob-size]   v2 (elf embedded)        : {} bytes",
        v2_elf.len()
    );
    eprintln!(
        "[blob-size]   v2 (no elf, production)  : {} bytes",
        v2_no_elf.len()
    );
    eprintln!(
        "[blob-size]   delta v1 -> v2(no elf)   : {} bytes (removes elf {}, adds 40)",
        v1.len() as i64 - v2_no_elf.len() as i64,
        elf.len(),
    );
    // The v2(no-elf) blob is inner-ELF-size independent, so for a larger inner
    // ELF the delta ≈ len(elf) - 40. Projected for ethrex's 3,647,200 B ELF:
    let ethrex_elf = 3_647_200i64;
    eprintln!(
        "[blob-size]   projected delta for inner=ethrex ({ethrex_elf} B elf): ~{} bytes smaller",
        ethrex_elf - 40,
    );
}

/// Dump v1 and v2 continuation blobs for the SAME inner proof, for the
/// execute-only cycle/blob-size measurement (run baseline guest on the v1 blob,
/// attestid guest on the v2 blobs). Env:
/// * `MEAS_INNER_ELF`   (path, required) — inner program ELF (e.g. ethrex).
/// * `MEAS_INNER_INPUT` (path, default none) — inner private input.
/// * `MEAS_EPOCH_LOG2`  (int, default 16) — continuation epoch size.
/// * `MEAS_PRESET`      (min|blowup2|blowup4|blowup8, default blowup2).
/// * `MEAS_OUT_DIR`     (dir, default /tmp) — writes v1.bin, v2_embed.bin, v2.bin.
#[test]
#[ignore = "measurement: dumps v1/v2 continuation blobs for cycle+size measurement"]
fn dump_measurement_blobs() {
    let inner_elf_path = std::env::var("MEAS_INNER_ELF").expect("set MEAS_INNER_ELF");
    let inner_elf = std::fs::read(&inner_elf_path).expect("read MEAS_INNER_ELF");
    let inner_input = std::env::var("MEAS_INNER_INPUT")
        .ok()
        .map(|p| std::fs::read(&p).expect("read MEAS_INNER_INPUT"))
        .unwrap_or_default();
    let epoch_log2: u32 = std::env::var("MEAS_EPOCH_LOG2")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(16);
    let preset_name = std::env::var("MEAS_PRESET").unwrap_or_else(|_| "blowup2".to_string());
    let preset = recursion::Preset::ALL
        .into_iter()
        .find(|p| p.name() == preset_name)
        .expect("unknown MEAS_PRESET");
    let opts = preset.options();
    let out_dir = std::env::var("MEAS_OUT_DIR").unwrap_or_else(|_| "/tmp".to_string());

    eprintln!(
        "[dump] inner={inner_elf_path} ({} B) input={} B epoch=2^{epoch_log2} preset={}",
        inner_elf.len(),
        inner_input.len(),
        preset.name()
    );
    let bundle =
        crate::continuation::prove_continuation(&inner_elf, &inner_input, epoch_log2, &opts)
            .expect("inner continuation prove");
    eprintln!("[dump] epochs={}", bundle.num_epochs());

    let v1 = recursion::encode_continuation_guest_input(bundle.clone(), &inner_elf, &opts)
        .expect("v1 encode");
    let v2_embed =
        recursion::encode_continuation_guest_input_v2(bundle.clone(), &inner_elf, &opts, true)
            .expect("v2 embed encode");
    let v2 =
        recursion::encode_continuation_guest_input_v2(bundle.clone(), &inner_elf, &opts, false)
            .expect("v2 encode");

    // v2-specific tamper blob: an otherwise-honest v2 blob whose FIRST supplied
    // page-genesis commitment is corrupted. The supplied roots are the verifier's
    // preprocessed PAGE genesis, so a wrong root fails multi_verify -> the v2 guest
    // rejects (Ok(None) -> guest panic). Demonstrates execute-only that the supplied
    // commitments are load-bearing on the v2 path (the host
    // v2_continuation_tamper_suite proves the same binding). Only meaningful when
    // the program has >=1 touched ELF-data page; otherwise it degenerates to v2.bin.
    let (decode_commitment, mut page_commitments) =
        crate::continuation::continuation_precomputed_commitments(&inner_elf, &bundle, &opts)
            .expect("precomputed commitments");
    let elf_parsed = executor::elf::Elf::load(&inner_elf).expect("load inner elf");
    if let Some(first) = page_commitments.first_mut() {
        first.1[0] ^= 0xFF;
    }
    let v2_badpage = crate::encode_recursion_archive(
        &recursion::ContinuationGuestInputV2 {
            bundle,
            inner_elf: Vec::new(),
            elf_digest: crate::statement::elf_digest(&inner_elf),
            entry_point: elf_parsed.entry_point,
            decode_commitment,
            page_commitments,
        },
        crate::RECURSION_INPUT_VERSION_V2,
    )
    .expect("v2 badpage encode");

    std::fs::write(format!("{out_dir}/v1.bin"), &v1).unwrap();
    std::fs::write(format!("{out_dir}/v2_embed.bin"), &v2_embed).unwrap();
    std::fs::write(format!("{out_dir}/v2.bin"), &v2).unwrap();
    std::fs::write(format!("{out_dir}/v2_badpage.bin"), &v2_badpage).unwrap();
    eprintln!("[dump] BLOB SIZES (bytes):");
    eprintln!("[dump]   v1 (elf embedded)       : {}", v1.len());
    eprintln!("[dump]   v2 (elf embedded)       : {}", v2_embed.len());
    eprintln!("[dump]   v2 (no elf, production)  : {}", v2.len());
    eprintln!("[dump]   v2_badpage (tamper, no elf): {}", v2_badpage.len());
    eprintln!(
        "[dump]   blob-size delta v1->v2   : {} bytes",
        v1.len() as i64 - v2.len() as i64
    );
}

// ============================================================================
// Continuation v2
// ============================================================================

#[test]
fn v2_continuation_accepts_and_binds() {
    let elf = read_guest_elf("fibonacci");
    let opts = MIN_PROOF_OPTIONS;
    let inner_input = 10u64.to_le_bytes();

    let bundle = crate::continuation::prove_continuation(&elf, &inner_input, 4, &opts)
        .expect("continuation prove");
    assert!(
        bundle.num_epochs() > 1,
        "epoch=2^4 must split fibonacci(10) into multiple epochs"
    );

    // Consumer re-bind values, computed before the encode consumes the bundle.
    let expected_id =
        recursion::expected_continuation_program_id_v2(&elf, &bundle, &opts).expect("expected id");
    let expected_out = crate::continuation::verify_continuation(&elf, &bundle, &opts)
        .expect("verify_continuation errored")
        .expect("bundle must verify");

    let blob =
        recursion::encode_continuation_guest_input_v2(bundle, &elf, &opts, false).expect("encode");
    let att = recursion::verify_continuation_and_attest_v2(&blob, &opts)
        .expect("verify errored")
        .expect("v2 continuation guest must accept");
    let (id, out) = recursion::split_attestation(&att).expect("attestation too short");
    assert_eq!(
        id, expected_id,
        "v2 continuation id must match the recompute"
    );
    assert_eq!(out, &expected_out[..], "attested output must match");
}

#[test]
fn v2_continuation_tamper_suite() {
    let elf = read_guest_elf("fibonacci");
    let opts = MIN_PROOF_OPTIONS;
    let inner_input = 10u64.to_le_bytes();

    let bundle = crate::continuation::prove_continuation(&elf, &inner_input, 4, &opts)
        .expect("continuation prove");
    let elf_parsed = executor::elf::Elf::load(&elf).expect("load");
    let digest = crate::statement::elf_digest(&elf);
    let (decode, pages) =
        crate::continuation::continuation_precomputed_commitments(&elf, &bundle, &opts)
            .expect("roots");
    let expected_id =
        recursion::expected_continuation_program_id_v2(&elf, &bundle, &opts).expect("expected id");

    // Predicate: accepts AND the attested id matches the honest recompute.
    let accepts_and_binds = |blob: &[u8]| -> bool {
        match recursion::verify_continuation_and_attest_v2(blob, &opts) {
            Ok(Some(att)) => matches!(
                recursion::split_attestation(&att),
                Some((id, _)) if id == expected_id
            ),
            _ => false,
        }
    };
    let build = |elf_digest: [u8; 32],
                 entry_point: u64,
                 decode_commitment: crate::Commitment,
                 page_commitments: Vec<(u64, crate::Commitment)>|
     -> Vec<u8> {
        crate::encode_recursion_archive(
            &recursion::ContinuationGuestInputV2 {
                bundle: bundle.clone(),
                inner_elf: Vec::new(),
                elf_digest,
                entry_point,
                decode_commitment,
                page_commitments,
            },
            crate::RECURSION_INPUT_VERSION_V2,
        )
        .expect("encode v2 continuation blob")
    };

    // Honest reconstruction accepts and binds.
    assert!(
        accepts_and_binds(&build(
            digest,
            elf_parsed.entry_point,
            decode,
            pages.clone()
        )),
        "honest hand-built v2 continuation blob must accept and bind"
    );

    // Corrupt DECODE.
    let mut bad_decode = decode;
    bad_decode[0] ^= 0xFF;
    assert!(
        !accepts_and_binds(&build(
            digest,
            elf_parsed.entry_point,
            bad_decode,
            pages.clone()
        )),
        "corrupted DECODE commitment must reject"
    );

    // Corrupt a page-genesis commitment (if any touched data page exists).
    if !pages.is_empty() {
        let mut bad_pages = pages.clone();
        bad_pages[0].1[0] ^= 0xFF;
        assert!(
            !accepts_and_binds(&build(digest, elf_parsed.entry_point, decode, bad_pages)),
            "corrupted page commitment must reject"
        );
    }

    // Lie about the entry point.
    assert!(
        !accepts_and_binds(&build(
            digest,
            elf_parsed.entry_point.wrapping_add(4),
            decode,
            pages.clone()
        )),
        "lying entry_point must reject or fail to bind"
    );
}
