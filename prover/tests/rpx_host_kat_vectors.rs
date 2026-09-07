//! Generator for the Rust-oracle tables in
//! `crypto/math-cuda/tests/host_kat/rpx_kat_vectors.h`.
//!
//! The RPX device kernel (`crypto/math-cuda/kernels/rpx.cu`) is pinned to THIS
//! crate's `Rpx256` through the host known-answer harness `rpx_host_kat.cpp`.
//! miden publishes no RPX known-answer table (`lfm/rpx.rs`, "PROVENANCE"), so
//! the Rust host implementation IS the oracle, and this test prints the tables
//! the harness embeds, as C++ source:
//!
//!   Table 2 — the bare permutation on eleven states: all-zero, all-(p−1),
//!             `0..12`, alternating, two one-hot lanes, four random, and the
//!             canonicalisation witness (see `permutation_inputs`);
//!   Table 3 — the rate-8 OVERWRITE-duplex leaf (`algebraic_commit::sponge_leaf`)
//!             at lengths 0, 1, 7, 8, 9, 16, 17 felts;
//!   Table 4 — the parent `compress(l, r)` = one permutation of `[l ‖ r ‖ 0⁴]`.
//!
//! Every printed value is CANONICAL (`< p`), and the harness compares the
//! kernel's output against it RAW — the kernel's final canonicalisation loop
//! is part of what these tables pin, so nothing may canonicalise on its
//! behalf.
//!
//! `#[ignore]`d because it prints rather than asserts. Run with
//!
//!   cargo test -p lambda-vm-prover --test rpx_host_kat_vectors -- --ignored --nocapture
//!
//! and paste everything between the `>>> BEGIN` / `<<< END` lines over the
//! matching region of `rpx_kat_vectors.h`. The inputs are DERIVED HERE from
//! fixed seeds and printed alongside the outputs, so the header stays
//! self-contained data — the harness never regenerates anything.

use lambda_vm_prover::lfm::algebraic_commit::sponge_leaf;
use lambda_vm_prover::lfm::hash::{HASH_STATE_FELTS, HasherKind, LfmHasher};
use lambda_vm_prover::lfm::rpx::Rpx256;
use lambda_vm_prover::tables::types::FE;

/// The Goldilocks prime, for canonicalising raw values and for the `p − 1`
/// input.
const P: u64 = 0xFFFF_FFFF_0000_0001;

/// The leaf lengths the phase-1 gate names: empty, a partial block, one felt
/// under a block, an exact block (no trailing permutation), one over, two exact
/// blocks, two blocks plus one.
const LEAF_LENGTHS: [usize; 7] = [0, 1, 7, 8, 9, 16, 17];

/// Widest leaf in Table 3 — the header's fixed-width `felts[]` array.
const LEAF_MAX_FELTS: usize = 17;

/// splitmix64 — a fixed, documented PRNG so the inputs are reproducible from
/// the seed alone.
fn splitmix64(seed: &mut u64) -> u64 {
    *seed = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *seed;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// A canonical random felt.
fn random_felt(seed: &mut u64) -> u64 {
    splitmix64(seed) % P
}

/// The canonical `u64` of a field element. `value()` is the raw storage, which
/// the field allows to sit in `[p, 2^64)`; one subtraction canonicalises it.
fn canonical(f: &FE) -> u64 {
    let v = *f.value();
    if v >= P { v - P } else { v }
}

fn fe_array<const N: usize>(raw: &[u64; N]) -> [FE; N] {
    core::array::from_fn(|i| FE::from(raw[i]))
}

fn cpp_list(vals: &[u64]) -> String {
    vals.iter()
        .map(|v| format!("{v}ull"))
        .collect::<Vec<_>>()
        .join(", ")
}

/// The eleven permutation inputs, each with the name the harness prints on a
/// failure (and, for the witness, matches on).
fn permutation_inputs() -> Vec<(&'static str, [u64; HASH_STATE_FELTS])> {
    let mut v: Vec<(&'static str, [u64; HASH_STATE_FELTS])> = vec![
        ("all-zero", [0; HASH_STATE_FELTS]),
        ("all-(p-1)", [P - 1; HASH_STATE_FELTS]),
        ("lanes 0..12", core::array::from_fn(|i| i as u64)),
        (
            "alternating 0 / p-1",
            core::array::from_fn(|i| if i % 2 == 0 { 0 } else { P - 1 }),
        ),
        (
            "one-hot lane 0",
            core::array::from_fn(|i| u64::from(i == 0)),
        ),
        (
            "one-hot lane 11",
            core::array::from_fn(|i| u64::from(i == HASH_STATE_FELTS - 1)),
        ),
    ];
    let names = ["random #1", "random #2", "random #3", "random #4"];
    for (k, name) in names.iter().enumerate() {
        let mut seed = 0x5250_5800_0000_0000 + k as u64; // "RPX\0" + k
        v.push((name, core::array::from_fn(|_| random_felt(&mut seed))));
    }
    // ★ The canonicalisation witness (review finding on the phase-1 PR). The
    // kernel canonicalises its output in a final loop, and a check that
    // compares canonical values — or raw values that happen to be canonical,
    // which is all but a 2^-32 slice per lane — cannot see whether that loop
    // is there. This input is built so that it is not: its M-round MDS output
    // lane 0 is `p − ARK1[6][0] + 1`, so the device's final `add` yields the
    // raw twin `p + 1` where the field value is 1. Derived by inverting the
    // permutation from that target
    // (`crypto/math-cuda/tests/host_kat/rpx_canon_witness.py`); the harness
    // matches this row BY NAME, replays the rounds to assert the twin is still
    // produced, and compares `permute`'s output raw against the digits below.
    v.push((
        "canonicalisation witness",
        [
            15055324559807314153,
            10242425218814686878,
            9326602342065331773,
            15451135068213333861,
            17942679252967467289,
            9284164080268346300,
            5090350781253234438,
            9328738269791029498,
            18385380985273671691,
            3238854716908013220,
            5495049682105235955,
            15773368383738726538,
        ],
    ));
    v
}

#[test]
#[ignore = "prints the Rust-oracle tables for rpx_kat_vectors.h; run with --ignored --nocapture"]
fn print_rpx_host_kat_vectors() {
    let mut out = String::new();
    out.push_str("// >>> BEGIN RUST-ORACLE TABLES — generated by\n");
    out.push_str(
        "//   cargo test -p lambda-vm-prover --test rpx_host_kat_vectors -- --ignored --nocapture\n",
    );
    out.push_str(
        "// (prover/tests/rpx_host_kat_vectors.rs). Paste verbatim; do not edit by hand.\n\n",
    );

    // ---- Table 2: the bare permutation ------------------------------------
    let inputs = permutation_inputs();
    out.push_str(&format!(
        "inline constexpr int NUM_RPX_PERMUTATION_VECTORS = {};\n",
        inputs.len()
    ));
    out.push_str(
        "inline constexpr RpxPermutationVector RPX_PERMUTATION_VECTORS[NUM_RPX_PERMUTATION_VECTORS] = {\n",
    );
    for (name, input) in &inputs {
        let got = Rpx256.permute(fe_array(input));
        let got: Vec<u64> = got.iter().map(canonical).collect();
        out.push_str(&format!(
            "    {{\"{name}\",\n     {{{}}},\n     {{{}}}}},\n",
            cpp_list(input),
            cpp_list(&got)
        ));
    }
    out.push_str("};\n\n");

    // ---- Table 3: the leaf sponge -----------------------------------------
    out.push_str(&format!(
        "inline constexpr int NUM_RPX_LEAF_VECTORS = {};\n",
        LEAF_LENGTHS.len()
    ));
    out.push_str("inline constexpr RpxLeafVector RPX_LEAF_VECTORS[NUM_RPX_LEAF_VECTORS] = {\n");
    for &len in &LEAF_LENGTHS {
        let mut seed = 0x1EAF_0000_0000_0000 + len as u64;
        let raw: Vec<u64> = (0..len).map(|_| random_felt(&mut seed)).collect();
        let felts: Vec<FE> = raw.iter().map(|&r| FE::from(r)).collect();
        let digest = sponge_leaf(HasherKind::Rpx, &felts);
        let digest: Vec<u64> = digest.iter().map(canonical).collect();
        // Fixed-width row: the tail beyond `len` is zero and never read.
        let mut padded = raw.clone();
        padded.resize(LEAF_MAX_FELTS, 0);
        out.push_str(&format!(
            "    {{{len}u,\n     {{{}}},\n     {{{}}}}},\n",
            cpp_list(&padded),
            cpp_list(&digest)
        ));
    }
    out.push_str("};\n\n");

    // ---- Table 4: the parent ------------------------------------------------
    let mut seed = 0x5041_5245_4E54_0000; // "PARENT"
    let random_l: [u64; 4] = core::array::from_fn(|_| random_felt(&mut seed));
    let random_r: [u64; 4] = core::array::from_fn(|_| random_felt(&mut seed));
    let parents: [(&str, [u64; 4], [u64; 4]); 2] = [
        (
            "digits 0..8",
            core::array::from_fn(|i| i as u64),
            core::array::from_fn(|i| i as u64 + 4),
        ),
        ("random", random_l, random_r),
    ];
    out.push_str(&format!(
        "inline constexpr int NUM_RPX_PARENT_VECTORS = {};\n",
        parents.len()
    ));
    out.push_str(
        "inline constexpr RpxParentVector RPX_PARENT_VECTORS[NUM_RPX_PARENT_VECTORS] = {\n",
    );
    for (name, l, r) in &parents {
        let got = HasherKind::Rpx.compress(&fe_array(l), &fe_array(r));
        let got: Vec<u64> = got.iter().map(canonical).collect();
        out.push_str(&format!(
            "    {{\"{name}\",\n     {{{}}},\n     {{{}}},\n     {{{}}}}},\n",
            cpp_list(l),
            cpp_list(r),
            cpp_list(&got)
        ));
    }
    out.push_str("};\n");
    out.push_str("// <<< END RUST-ORACLE TABLES\n");

    println!("{out}");
}
