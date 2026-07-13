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
/// for Blackwell) and return a *real* arch (`sm_XX`) suitable for cubin
/// (SASS) generation. Falls back to `sm_89` (Ada) when no GPU is visible or
/// the query fails, warning loudly because an arch-specific cubin built for
/// the wrong GPU cannot load on the run host.
fn detect_arch() -> String {
    const FALLBACK: &str = "sm_89";
    detect_arch_from_smi().unwrap_or_else(|| {
        println!(
            "cargo:warning=math-cuda: could not detect a GPU via nvidia-smi — cubins target \
             fallback {FALLBACK}; if the run host GPU differs, set CUDARC_NVCC_ARCH=sm_XX or \
             rebuild on the run host."
        );
        FALLBACK.to_string()
    })
}

/// Parse the compute capability out of `nvidia-smi` and format it as a real
/// `sm_XX` arch. Returns `None` on every path where no capability can be read
/// (nvidia-smi missing, command failed, or unparsable output) so the caller
/// warns before falling back.
fn detect_arch_from_smi() -> Option<String> {
    let output = match Command::new("nvidia-smi")
        .args(["--query-gpu=compute_cap", "--format=csv,noheader"])
        .output()
    {
        Ok(o) if o.status.success() => o,
        _ => return None,
    };
    let line = std::str::from_utf8(&output.stdout).ok()?;
    // First line, first comma-separated value (covers multi-GPU hosts).
    let cap = line.lines().next()?.split(',').next().unwrap_or("").trim();
    let (major, minor) = cap.split_once('.')?;
    let (major, minor) = (major.trim(), minor.trim());
    if major.chars().all(|c| c.is_ascii_digit()) && minor.chars().all(|c| c.is_ascii_digit()) {
        Some(format!("sm_{major}{minor}"))
    } else {
        None
    }
}

/// Normalize a user-supplied `CUDARC_NVCC_ARCH` override to a *real* arch
/// (`sm_XX`). cubin (SASS) generation rejects the *virtual* `compute_XX`
/// form, but we accept it (and a bare `XX`) for backwards compatibility.
fn to_real_arch(arch: &str) -> String {
    if let Some(n) = arch.strip_prefix("compute_") {
        format!("sm_{n}")
    } else if arch.starts_with("sm_") {
        arch.to_string()
    } else {
        format!("sm_{arch}")
    }
}

fn compile_kernel(src: &str, out_name: &str, have_nvcc: bool) {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let src_path = manifest_dir.join("kernels").join(src);
    let out_path = out_dir.join(out_name);

    println!("cargo:rerun-if-changed=kernels/{src}");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=CUDARC_NVCC_ARCH");

    // When nvcc is missing from PATH, emit an empty cubin stub so the crate
    // still compiles. include_bytes! in src/device.rs needs the file to exist
    // at build time. Any runtime kernel call fails to load the empty module and
    // the caller falls back to CPU. We can't run GPU code without nvcc on the
    // build host anyway.
    if !have_nvcc {
        fs::write(&out_path, "").expect("failed to write empty cubin stub");
        return;
    }

    // AOT-compile each kernel to a native cubin (SASS) for the host GPU's real
    // arch, NOT to PTX. This sidesteps the driver's PTX-ISA JIT version check:
    // a toolkit's PTX ISA is fixed by its CUDA version (e.g. CUDA 13.1 emits PTX
    // .version 9.1), and a driver older than that toolkit rejects the module at
    // load with CUDA_ERROR_UNSUPPORTED_PTX_VERSION -> every kernel silently
    // falls back to CPU. A cubin carries pre-compiled SASS for a real arch, so
    // the driver loads it directly as long as it supports that GPU (which the
    // driver installed for that GPU always does) — regardless of the toolkit's
    // CUDA version. See README "GPU Tests".
    //
    // Trade-off: a cubin is arch-specific (an `sm_120` cubin runs only on
    // `sm_120`). We build+run on the same GPU box in every flow and detect the
    // arch from that box's `nvidia-smi`, so this is exactly right. Override with
    // CUDARC_NVCC_ARCH (compute_XX / sm_XX / bare XX all accepted) to
    // cross-compile for a different arch; fall back to sm_89 (Ada) when
    // detection fails. If the toolkit is too old to know the GPU's arch, nvcc
    // fails loudly here at build time (better than a silent runtime fallback).
    let arch = env::var("CUDARC_NVCC_ARCH")
        .map(|a| to_real_arch(&a))
        .unwrap_or_else(|_| detect_arch());

    let status = Command::new(nvcc_path())
        .args(["--cubin", "-O3", "-std=c++17", "-arch", &arch, "-o"])
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
            "cargo:warning=math-cuda: nvcc not found at {} — emitting empty cubin stubs. \
             Runtime GPU calls fall back to CPU. Install CUDA and rebuild for a working backend.",
            nvcc_path().display()
        );
    }

    compile_kernel("arith.cu", "arith.cubin", have_nvcc);
    compile_kernel("ntt.cu", "ntt.cubin", have_nvcc);
    compile_kernel("keccak.cu", "keccak.cubin", have_nvcc);
    compile_kernel("barycentric.cu", "barycentric.cubin", have_nvcc);
    compile_kernel("deep.cu", "deep.cubin", have_nvcc);
    compile_kernel("fri.cu", "fri.cubin", have_nvcc);
    compile_kernel("inverse.cu", "inverse.cubin", have_nvcc);
    compile_kernel("logup.cu", "logup.cubin", have_nvcc);
    compile_kernel("constraint_interp.cu", "constraint_interp.cubin", have_nvcc);
}
