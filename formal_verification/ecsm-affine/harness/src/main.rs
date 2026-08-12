//! Dumps REAL `EcsmWitness` values from the repo's own witness generator, as JSON, so the
//! z3 gate's transcribed model can be evaluated on them.
//!
//! This is the cross-language half of the faithfulness anchor. The Python oracle
//! (`../oracle/ecsm_affine_ref.py`) is an independent reimplementation, which establishes
//! that the gate is reasoning about the right FUNCTION. This harness establishes that it is
//! reasoning about the right COLUMNS: every witness field the model reads is emitted here by
//! `ecsm::compute_witness_with_y` / `compute_witness`, so a column the model mis-transcribed
//! shows up as a mismatch rather than as a silently-wrong UNSAT.
//!
//! It is deliberately tiny and depends only on `crypto/ecsm` — building the prover is not
//! needed to check witness columns, and a heavier harness would not get run.
//!
//! Build and run (from this directory):
//!
//!     cargo run --release -- > ../gate/logs/real_witnesses.jsonl
//!
//! Then: `python ../gate/a6_real_witness.py`

use ecsm::{compute_witness, compute_witness_with_y};
use num_bigint::BigUint;

/// secp256k1 generator, and the curve order, as little-endian 32-byte values.
const GX_BE: &str = "79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798";
const GY_BE: &str = "483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8";
const N_BE: &str = "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141";
const P_BE: &str = "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F";

fn le32_from_be_hex(hex: &str) -> [u8; 32] {
    let v = BigUint::parse_bytes(hex.as_bytes(), 16).expect("hex");
    let mut out = [0u8; 32];
    for (i, b) in v.to_bytes_le().into_iter().enumerate() {
        out[i] = b;
    }
    out
}

fn le32(v: &BigUint) -> [u8; 32] {
    let mut out = [0u8; 32];
    for (i, b) in v.to_bytes_le().into_iter().enumerate() {
        out[i] = b;
    }
    out
}

fn hex_le(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// The scalars worth pinning: the two the x-only path called degenerate, the bit-length
/// extremes, and a couple of ordinary ones.
fn scalars() -> Vec<(String, BigUint)> {
    let n = BigUint::parse_bytes(N_BE.as_bytes(), 16).unwrap();
    let one = BigUint::from(1u8);
    vec![
        ("k=1".into(), one.clone()),
        ("k=2".into(), BigUint::from(2u8)),
        ("k=3".into(), BigUint::from(3u8)),
        ("k=N-1".into(), &n - &one),
        ("k=N-2".into(), &n - BigUint::from(2u8)),
        ("k=2^255".into(), BigUint::from(1u8) << 255),
        ("k=2^255-1".into(), (BigUint::from(1u8) << 255) - &one),
        ("k=(N-1)/2".into(), (&n - &one) / BigUint::from(2u8)),
        ("k=0xdeadbeef".into(), BigUint::from(0xdead_beefu64)),
    ]
}

/// Emits one JSON object per line. Every field the gate's model reads is present, plus the
/// mode, so `a6_real_witness.py` can check the x-only and affine paths separately.
fn emit(label: &str, mode: &str, k: &[u8; 32], xg: &[u8; 32], yg: Option<&[u8; 32]>) {
    let w = match yg {
        Some(y) => compute_witness_with_y(k, xg, y),
        None => compute_witness(k, xg),
    };
    let w = match w {
        Ok(w) => w,
        Err(e) => {
            println!("{{\"label\":\"{label}\",\"mode\":\"{mode}\",\"error\":\"{e}\"}}");
            return;
        }
    };
    // Only the fields this campaign's model consumes. The ECDAS step array is large and the
    // earlier board already anchors it (`gate/positive_real_witness.py` under
    // `thoughts/ec-recover-opt/` on branch `feat/ec-lincomb2`, which was never merged),
    // so it is summarised by its length rather than dumped.
    println!(
        "{{\"label\":\"{label}\",\"mode\":\"{mode}\",\
         \"k\":\"{}\",\"x_g\":\"{}\",\"y_g\":\"{}\",\
         \"x_r\":\"{}\",\"y_r\":\"{}\",\
         \"x_g_sub_p\":\"{}\",\"k_sub_n\":\"{}\",\
         \"x_r_sub_p\":\"{}\",\"y_r_sub_p\":\"{}\",\
         \"x2\":\"{}\",\"q0\":\"{}\",\"q1\":\"{}\",\
         \"len_k\":{},\"steps\":{}}}",
        hex_le(k),
        hex_le(&w.x_g),
        hex_le(&w.y_g),
        hex_le(&w.x_r),
        hex_le(&w.y_r),
        hex_le(&w.x_g_sub_p),
        hex_le(&w.k_sub_n),
        hex_le(&w.x_r_sub_p),
        hex_le(&w.y_r_sub_p),
        hex_le(&w.x2),
        hex_le(&w.q0),
        hex_le(&w.q1),
        w.len_k,
        w.steps.len(),
    );
}

fn main() {
    let gx = le32_from_be_hex(GX_BE);
    let gy = le32_from_be_hex(GY_BE);
    let p = BigUint::parse_bytes(P_BE.as_bytes(), 16).unwrap();
    let gy_big = BigUint::parse_bytes(GY_BE.as_bytes(), 16).unwrap();
    let gy_neg = le32(&(&p - &gy_big));

    for (label, k) in scalars() {
        let k = le32(&k);
        // x-only: yG is the canonical even lift, recovered internally from xG.
        emit(&label, "x-only", &k, &gx, None);
        // affine, both roots — the pair A3's forgery is built from. Both must produce a
        // valid witness (that is the gap), and their y_r must differ.
        emit(&label, "affine/+y", &k, &gx, Some(&gy));
        emit(&label, "affine/-y", &k, &gx, Some(&gy_neg));
    }

    // The y = 1 point from ../oracle/small_y_point.py, reached as 2·(2^-1·Q), so the honest
    // y_r sits at the very bottom of the non-canonical band and y_r_sub_p is at its extreme.
    let small_xg =
        le32_from_be_hex("F2E13FD883D5F5138E1658A6022391495DF397ACB9A83E861F6BF5181D6C4DBC");
    let small_yg =
        le32_from_be_hex("264A2700355E78B1E2D5B19FC29FFDDAEC27243E405D318525F49EFFA3007229");
    let two = le32(&BigUint::from(2u8));
    emit(
        "small-y (y_r = 1)",
        "affine/+y",
        &two,
        &small_xg,
        Some(&small_yg),
    );

    // Rejections the executor relies on: the validation set A4 of the oracle anchors.
    let zero = [0u8; 32];
    emit("k=0", "affine/+y", &zero, &gx, Some(&gy));
    let n_le = le32_from_be_hex(N_BE);
    emit("k=N", "affine/+y", &n_le, &gx, Some(&gy));
    let mut off_curve = gy;
    off_curve[0] ^= 1;
    emit("off-curve yG", "affine/+y", &two, &gx, Some(&off_curve));
    let p_le = le32_from_be_hex(P_BE);
    emit("yG=p", "affine/+y", &two, &gx, Some(&p_le));
}
