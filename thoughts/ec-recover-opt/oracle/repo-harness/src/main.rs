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
            other => panic!("unknown command {other}"),
        }
        out.flush().unwrap();
    }
}
