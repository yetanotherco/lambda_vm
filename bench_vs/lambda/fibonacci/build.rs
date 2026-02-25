use std::env;
use std::fs;
use std::path::Path;

fn main() {
    let n = env::var("BENCH_N").unwrap_or_else(|_| "1000".to_string());
    let out_dir = env::var("OUT_DIR").unwrap();
    fs::write(Path::new(&out_dir).join("n.txt"), &n).unwrap();
    println!("cargo:rerun-if-env-changed=BENCH_N");
}
