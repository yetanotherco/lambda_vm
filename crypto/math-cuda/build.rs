use std::env;
use std::path::PathBuf;
use std::process::Command;

fn cuda_home() -> PathBuf {
    env::var_os("CUDA_HOME")
        .or_else(|| env::var_os("CUDA_PATH"))
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/usr/local/cuda"))
}

fn nvcc_path() -> PathBuf {
    cuda_home().join("bin").join("nvcc")
}

fn compile_ptx(src: &str, out_name: &str) {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let src_path = manifest_dir.join("kernels").join(src);
    let out_path = out_dir.join(out_name);

    println!("cargo:rerun-if-changed=kernels/{src}");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=CUDARC_NVCC_ARCH");

    // Emit PTX for a virtual architecture; the CUDA driver JIT-compiles it for the
    // actual GPU at load time, so one PTX works across Ada/Hopper/Blackwell. Override
    // with CUDARC_NVCC_ARCH to pin a specific compute capability.
    let arch = env::var("CUDARC_NVCC_ARCH").unwrap_or_else(|_| "compute_89".to_string());

    let status = Command::new(nvcc_path())
        .args([
            "--ptx",
            "-O3",
            "-std=c++17",
            "-arch",
            &arch,
            "-o",
        ])
        .arg(&out_path)
        .arg(&src_path)
        .status()
        .expect("failed to invoke nvcc — is CUDA installed and CUDA_HOME set?");

    if !status.success() {
        panic!("nvcc failed compiling {}", src_path.display());
    }
}

fn main() {
    // Header is not compiled; emit rerun-if-changed so edits trigger rebuilds.
    println!("cargo:rerun-if-changed=kernels/goldilocks.cuh");
    compile_ptx("arith.cu", "arith.ptx");
    compile_ptx("ntt.cu", "ntt.ptx");
}
