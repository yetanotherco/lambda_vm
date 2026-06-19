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

/// Query `nvidia-smi` for the local GPU's compute capability (e.g. "12.0"
/// for Blackwell). Returns a `compute_XX` target on success, falling back
/// to `compute_89` (Ada) when no GPU is visible or the query fails.
fn detect_arch() -> String {
    const FALLBACK: &str = "compute_89";
    let output = match Command::new("nvidia-smi")
        .args(["--query-gpu=compute_cap", "--format=csv,noheader"])
        .output()
    {
        Ok(o) if o.status.success() => o,
        _ => return FALLBACK.to_string(),
    };
    let line = match std::str::from_utf8(&output.stdout) {
        Ok(s) => s,
        Err(_) => return FALLBACK.to_string(),
    };
    // First line, first comma-separated value (covers multi-GPU hosts).
    let cap = match line.lines().next() {
        Some(l) => l.split(',').next().unwrap_or("").trim(),
        None => return FALLBACK.to_string(),
    };
    let (major, minor) = match cap.split_once('.') {
        Some((m, n)) => (m.trim(), n.trim()),
        None => return FALLBACK.to_string(),
    };
    if major.chars().all(|c| c.is_ascii_digit()) && minor.chars().all(|c| c.is_ascii_digit()) {
        format!("compute_{major}{minor}")
    } else {
        FALLBACK.to_string()
    }
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

    // When nvcc is missing from PATH, emit an empty PTX stub so the crate
    // still compiles. include_str! in src/device.rs needs the file to exist
    // at build time. Any runtime kernel call panics in cudarc when loading
    // the empty module. We can't run GPU code without nvcc on the build
    // host anyway.
    if !have_nvcc {
        fs::write(&out_path, "").expect("failed to write empty PTX stub");
        return;
    }

    // Emit PTX for a virtual architecture; the CUDA driver JIT-compiles it for the
    // actual GPU at load time. Override with CUDARC_NVCC_ARCH to pin a specific
    // compute capability. If unset, try `nvidia-smi` to match the host GPU
    // (avoids JIT failures like nvcc-13.0 PTX rejected on Blackwell drivers);
    // fall back to compute_89 (Ada) when detection fails.
    let arch = env::var("CUDARC_NVCC_ARCH").unwrap_or_else(|_| detect_arch());

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
    // Headers aren't compiled, so emit rerun-if-changed to rebuild on
    // header edits.
    println!("cargo:rerun-if-changed=kernels/goldilocks.cuh");
    println!("cargo:rerun-if-changed=kernels/ext3.cuh");

    // Probe for nvcc once. Workspace consumers (clippy, fmt, CPU-only test
    // runners) build math-cuda incidentally without using its kernels. Stub
    // out PTX when nvcc is unavailable so those builds succeed.
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
    compile_ptx("barycentric.cu", "barycentric.ptx", have_nvcc);
    compile_ptx("deep.cu", "deep.ptx", have_nvcc);
    compile_ptx("fri.cu", "fri.ptx", have_nvcc);
    compile_ptx("inverse.cu", "inverse.ptx", have_nvcc);
    compile_ptx("trace_primitives.cu", "trace_primitives.ptx", have_nvcc);
    compile_ptx("multiplicity_sort.cu", "multiplicity_sort.ptx", have_nvcc);
    compile_ptx("page_trace.cu", "page_trace.ptx", have_nvcc);
    compile_ptx("decode_trace.cu", "decode_trace.ptx", have_nvcc);
}
