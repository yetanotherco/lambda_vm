//! Line-protocol harness exposing the repo's `ecsm` crate to the Python
//! oracle differential. All values are big-endian hex integers on the wire;
//! conversion to the crate's 32-byte little-endian ABI happens here.
//!
//! Commands (stdin, one per line) -> responses (stdout, one per line unless noted):
//!   mul <x_hex> <k_hex>        -> ok <xr_hex> | err <ErrorKind>
//!   recovery <x_hex>           -> y <y_hex> | none
//!   replay <x_hex> <k_hex>     -> steps <n> <final_x> <final_y>
//!                                 then n lines: step <round> <op> <next_op> <ax> <ay> <lambda> <rx> <ry>
//!                                 (rejects via prepare-equivalent checks first: err <ErrorKind>)
//!   lincomb2 <u1> <u2> <x1> <y1> <x2> <y2>
//!                              -> lincomb2_json {...}  | err <Lincomb2Error>
//!                                 Full `Lincomb2Witness` summary: Q, len, P12, the
//!                                 canonicalization witnesses, and every joint row
//!                                 (sel/round/op/d1/d2/a/addend/lambda/r). Consumed by
//!                                 the phase-D0 Python differential.

use ecsm::witness::{dinv_witness, lincomb2_witness, JointSel};
use ecsm::{
    compute_witness, recover_y_canonical, replay_double_and_add, scalar_mul_x, AffinePoint,
    EcsmError,
};
use num_bigint::BigUint;
use std::io::{BufRead, Write};

fn bytes_hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

fn i64s_json(c: &[i64]) -> String {
    let items: Vec<String> = c.iter().map(|x| x.to_string()).collect();
    format!("[{}]", items.join(","))
}

fn hex_to_le32(h: &str) -> [u8; 32] {
    let v = BigUint::parse_bytes(h.as_bytes(), 16).expect("bad hex");
    let mut bytes = v.to_bytes_le();
    assert!(bytes.len() <= 32, "value exceeds 256 bits");
    bytes.resize(32, 0);
    let mut out = [0u8; 32];
    out.copy_from_slice(&bytes);
    out
}

fn hex_to_big(h: &str) -> BigUint {
    BigUint::parse_bytes(h.as_bytes(), 16).expect("bad hex")
}

fn hx(v: &BigUint) -> String {
    v.to_str_radix(16)
}

/// 32 LE bytes -> big-endian hex integer, so the Python side can compare against
/// plain ints without knowing the ABI byte order.
fn le_hex(b: &[u8; 32]) -> String {
    hx(&BigUint::from_bytes_le(b))
}

fn sel_name(s: &JointSel) -> &'static str {
    match s {
        JointSel::Double => "Double",
        JointSel::AddP1 => "AddP1",
        JointSel::AddP2 => "AddP2",
        JointSel::AddP12 => "AddP12",
        JointSel::Precompute => "Precompute",
        JointSel::Correction => "Correction",
    }
}

fn err_kind(e: &EcsmError) -> &'static str {
    match e {
        EcsmError::ScalarIsZero => "ScalarIsZero",
        EcsmError::ScalarOutOfRange => "ScalarOutOfRange",
        EcsmError::NotOnCurve => "NotOnCurve",
        EcsmError::CoordinateOutOfRange => "CoordinateOutOfRange",
    }
}

fn main() {
    let stdin = std::io::stdin();
    let stdout = std::io::stdout();
    let mut out = std::io::BufWriter::new(stdout.lock());

    for line in stdin.lock().lines() {
        let line = line.unwrap();
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.is_empty() {
            continue;
        }
        match parts[0] {
            "mul" => {
                let x = hex_to_le32(parts[1]);
                let k = hex_to_le32(parts[2]);
                match scalar_mul_x(&k, &x) {
                    Ok(xr) => {
                        let v = BigUint::from_bytes_le(&xr);
                        writeln!(out, "ok {}", hx(&v)).unwrap();
                    }
                    Err(e) => writeln!(out, "err {}", err_kind(&e)).unwrap(),
                }
            }
            "recovery" => {
                let x = hex_to_big(parts[1]);
                match recover_y_canonical(&x) {
                    Some(y) => writeln!(out, "y {}", hx(&y)).unwrap(),
                    None => writeln!(out, "none").unwrap(),
                }
            }
            "replay" => {
                // Mirror `prepare`'s checks through the public API so invalid
                // inputs are reported instead of panicking inside k256 glue.
                let x = hex_to_big(parts[1]);
                let k = hex_to_big(parts[2]);
                let n = ecsm::n();
                let p = ecsm::p();
                if k == BigUint::from(0u8) {
                    writeln!(out, "err ScalarIsZero").unwrap();
                } else if k >= n {
                    writeln!(out, "err ScalarOutOfRange").unwrap();
                } else if x >= p {
                    writeln!(out, "err CoordinateOutOfRange").unwrap();
                } else if let Some(y) = recover_y_canonical(&x) {
                    let g = AffinePoint { x, y };
                    let (steps, result) = replay_double_and_add(&k, &g);
                    writeln!(out, "steps {} {} {}", steps.len(), hx(&result.x), hx(&result.y))
                        .unwrap();
                    for s in &steps {
                        writeln!(
                            out,
                            "step {} {} {} {} {} {} {} {}",
                            s.round,
                            s.op,
                            s.next_op,
                            hx(&s.a.x),
                            hx(&s.a.y),
                            hx(&s.lambda),
                            hx(&s.r.x),
                            hx(&s.r.y)
                        )
                        .unwrap();
                    }
                } else {
                    writeln!(out, "err NotOnCurve").unwrap();
                }
            }
            // witness <x_hex> <k_hex> -> one JSON line with the FULL EcsmWitness
            // (byte arrays as LE hex strings, carries as decimal arrays), or err <kind>.
            // Used by the z3 gate's byte-level positive controls: real prover
            // witnesses evaluated against the transcribed constraint model.
            "witness" => {
                let x = hex_to_le32(parts[1]);
                let k = hex_to_le32(parts[2]);
                match compute_witness(&k, &x) {
                    Ok(w) => {
                        let mut s = String::from("{");
                        s += &format!("\"x_g\":\"{}\",", bytes_hex(&w.x_g));
                        s += &format!("\"y_g\":\"{}\",", bytes_hex(&w.y_g));
                        s += &format!("\"k\":\"{}\",", bytes_hex(&w.k));
                        s += &format!("\"x2\":\"{}\",", bytes_hex(&w.x2));
                        s += &format!("\"q0\":\"{}\",", bytes_hex(&w.q0));
                        s += &format!("\"c0\":{},", i64s_json(&w.c0));
                        s += &format!("\"q1\":\"{}\",", bytes_hex(&w.q1));
                        s += &format!("\"c1\":{},", i64s_json(&w.c1));
                        s += &format!("\"x_g_sub_p\":\"{}\",", bytes_hex(&w.x_g_sub_p));
                        s += &format!("\"k_sub_n\":\"{}\",", bytes_hex(&w.k_sub_n));
                        s += &format!("\"x_r_sub_p\":\"{}\",", bytes_hex(&w.x_r_sub_p));
                        s += &format!("\"len_k\":{},", w.len_k);
                        s += &format!("\"x_r\":\"{}\",", bytes_hex(&w.x_r));
                        s += &format!("\"y_r\":\"{}\",", bytes_hex(&w.y_r));
                        s += "\"steps\":[";
                        let step_objs: Vec<String> = w
                            .steps
                            .iter()
                            .map(|st| {
                                format!(
                                    "{{\"x_a\":\"{}\",\"y_a\":\"{}\",\"x_g\":\"{}\",\"y_g\":\"{}\",\
                                     \"round\":{},\"op\":{},\"next_op\":{},\"lambda\":\"{}\",\
                                     \"x_r\":\"{}\",\"y_r\":\"{}\",\"q0\":\"{}\",\"q1\":\"{}\",\
                                     \"q2\":\"{}\",\"c0\":{},\"c1\":{},\"c2\":{}}}",
                                    bytes_hex(&st.x_a),
                                    bytes_hex(&st.y_a),
                                    bytes_hex(&st.x_g),
                                    bytes_hex(&st.y_g),
                                    st.round,
                                    st.op,
                                    st.next_op,
                                    bytes_hex(&st.lambda),
                                    bytes_hex(&st.x_r),
                                    bytes_hex(&st.y_r),
                                    bytes_hex(&st.q0),
                                    bytes_hex(&st.q1),
                                    bytes_hex(&st.q2),
                                    i64s_json(&st.c0),
                                    i64s_json(&st.c1),
                                    i64s_json(&st.c2)
                                )
                            })
                            .collect();
                        s += &step_objs.join(",");
                        s += "]}";
                        writeln!(out, "witness_json {s}").unwrap();
                    }
                    Err(e) => writeln!(out, "err {}", err_kind(&e)).unwrap(),
                }
            }
            // lincomb2 <u1> <u2> <x1> <y1> <x2> <y2>: the phase-A joint-chain
            // witness, dumped for the Python differential (Q, len, P12, the
            // canonicalization witnesses, and the full row schedule).
            "lincomb2" => {
                let u1 = hex_to_le32(parts[1]);
                let u2 = hex_to_le32(parts[2]);
                let p1 = AffinePoint {
                    x: hex_to_big(parts[3]),
                    y: hex_to_big(parts[4]),
                };
                let p2 = AffinePoint {
                    x: hex_to_big(parts[5]),
                    y: hex_to_big(parts[6]),
                };
                match lincomb2_witness(&u1, &u2, &p1, &p2) {
                    Ok(w) => {
                        let mut s = String::from("{");
                        s += &format!("\"len\":{},", w.len);
                        s += &format!("\"x_q\":\"{}\",", le_hex(&w.x_q));
                        s += &format!("\"y_q\":\"{}\",", le_hex(&w.y_q));
                        s += &format!("\"x_p12\":\"{}\",", le_hex(&w.x_p12));
                        s += &format!("\"y_p12\":\"{}\",", le_hex(&w.y_p12));
                        s += &format!("\"x_t0\":\"{}\",", le_hex(&w.x_t0));
                        s += &format!("\"y_t0\":\"{}\",", le_hex(&w.y_t0));
                        s += &format!("\"x_t0_pow\":\"{}\",", le_hex(&w.x_t0_pow));
                        s += &format!("\"y_t0_pow\":\"{}\",", le_hex(&w.y_t0_pow));
                        s += &format!("\"y_p2_sub_p\":\"{}\",", le_hex(&w.y_p2_sub_p));
                        s += &format!("\"x_q_sub_p\":\"{}\",", le_hex(&w.x_q_sub_p));
                        s += &format!("\"y_q_sub_p\":\"{}\",", le_hex(&w.y_q_sub_p));
                        s += &format!("\"u1_sub_n\":\"{}\",", le_hex(&w.u1_sub_n));
                        s += &format!("\"u2_sub_n\":\"{}\",", le_hex(&w.u2_sub_n));
                        s += "\"rows\":[";
                        let rows: Vec<String> = w
                            .steps
                            .iter()
                            .map(|js| {
                                let st = &js.step;
                                // The prover's OWN non-degeneracy columns, now
                                // that `dinv_witness` lives in `crypto/ecsm`.
                                let dw = dinv_witness(js);
                                format!(
                                    "{{\"sel\":\"{}\",\"round\":{},\"op\":{},\"d1\":{},\"d2\":{},\
                                     \"nb\":{},\"next_op\":{},\
                                     \"x_a\":\"{}\",\"y_a\":\"{}\",\"x_b\":\"{}\",\"y_b\":\"{}\",\
                                     \"q0\":\"{}\",\"q1\":\"{}\",\"q2\":\"{}\",\
                                     \"c0\":{},\"c1\":{},\"c2\":{},\
                                     \"d_inv\":\"{}\",\"q3\":\"{}\",\"c3\":{},\
                                     \"lambda\":\"{}\",\"x_r\":\"{}\",\"y_r\":\"{}\"}}",
                                    sel_name(&js.sel),
                                    st.round,
                                    st.op,
                                    js.d1,
                                    js.d2,
                                    js.nb,
                                    st.next_op,
                                    le_hex(&st.x_a),
                                    le_hex(&st.y_a),
                                    le_hex(&st.x_g),
                                    le_hex(&st.y_g),
                                    bytes_hex(&st.q0),
                                    bytes_hex(&st.q1),
                                    bytes_hex(&st.q2),
                                    i64s_json(&st.c0),
                                    i64s_json(&st.c1),
                                    i64s_json(&st.c2),
                                    bytes_hex(&dw.d_inv),
                                    bytes_hex(&dw.q3),
                                    i64s_json(&dw.c3),
                                    le_hex(&st.lambda),
                                    le_hex(&st.x_r),
                                    le_hex(&st.y_r)
                                )
                            })
                            .collect();
                        s += &rows.join(",");
                        s += "]}";
                        writeln!(out, "lincomb2_json {s}").unwrap();
                    }
                    Err(e) => writeln!(out, "err {e:?}").unwrap(),
                }
            }
            other => panic!("unknown command {other}"),
        }
        out.flush().unwrap();
    }
}
