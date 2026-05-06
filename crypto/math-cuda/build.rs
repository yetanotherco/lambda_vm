use std::env;
use std::fs;
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

fn compile_ptx(src: &str, out_name: &str, have_nvcc: bool) {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let src_path = manifest_dir.join("kernels").join(src);
    let out_path = out_dir.join(out_name);

    println!("cargo:rerun-if-changed=kernels/{src}");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=CUDARC_NVCC_ARCH");

    // No nvcc on PATH → emit an empty PTX stub so the crate still compiles.
    // include_str! in src/device.rs needs the file to exist at build time.
    // Any runtime kernel call will then panic from cudarc when loading the
    // empty module — which is the right failure mode (we can't run GPU code
    // without nvcc on the build host anyway).
    if !have_nvcc {
        fs::write(&out_path, "").expect("failed to write empty PTX stub");
        return;
    }

    // Emit PTX for a virtual architecture; the CUDA driver JIT-compiles it for the
    // actual GPU at load time, so one PTX works across Ada/Hopper/Blackwell. Override
    // with CUDARC_NVCC_ARCH to pin a specific compute capability.
    let arch = env::var("CUDARC_NVCC_ARCH").unwrap_or_else(|_| "compute_89".to_string());

    let status = Command::new(nvcc_path())
        .args(["--ptx", "-O3", "-std=c++17", "-arch", &arch, "-o"])
        .arg(&out_path)
        .arg(&src_path)
        .status()
        .expect("failed to invoke nvcc — is CUDA installed and CUDA_HOME set?");

    if !status.success() {
        panic!("nvcc failed compiling {}", src_path.display());
    }
}

fn main() {
    // Headers are not compiled; emit rerun-if-changed so edits trigger rebuilds.
    println!("cargo:rerun-if-changed=kernels/goldilocks.cuh");
    println!("cargo:rerun-if-changed=kernels/ext3.cuh");

    // Probe for nvcc once. Workspace consumers (clippy, fmt, CPU-only test
    // runners) build math-cuda incidentally without using its kernels; allow
    // those to succeed by stubbing out PTX when nvcc is unavailable.
    let have_nvcc = Command::new(nvcc_path())
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false);
    if !have_nvcc {
        println!(
            "cargo:warning=math-cuda: nvcc not found at {} — emitting empty PTX stubs. \
             Runtime GPU calls will panic. Install CUDA and rebuild for a working backend.",
            nvcc_path().display()
        );
    }

    compile_ptx("arith.cu", "arith.ptx", have_nvcc);
    compile_ptx("ntt.cu", "ntt.ptx", have_nvcc);
    compile_ptx("keccak.cu", "keccak.ptx", have_nvcc);
}
