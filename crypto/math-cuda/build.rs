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
/// (SASS) generation. Hard-fails the build when no GPU is visible and the
/// query fails: a cubin is arch-locked, so there is no safe default — any
/// guess produces a binary that loads on exactly one GPU model and silently
/// CPU-falls-back everywhere else. Failing here is loud and fixable (set
/// `CUDARC_NVCC_ARCH` or build on the target host); a guess is neither.
///
/// This is only reached when `nvcc` is present but the arch can't be detected
/// (a toolkit-installed host with no visible GPU). A host without `nvcc` takes
/// the empty-stub path in `compile_kernel` and never calls this.
fn detect_arch() -> String {
    detect_arch_from_smi().unwrap_or_else(|| {
        panic!(
            "math-cuda: nvcc is present but no GPU arch could be detected via nvidia-smi, \
             and a cubin must target a concrete arch. Set CUDARC_NVCC_ARCH=sm_XX (e.g. sm_120 \
             for RTX 5090, sm_86 for RTX 3090) or build on the target GPU host."
        )
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

/// Single source for the barycentric multi-kernel eval-point cap. The CUDA
/// side sizes a per-thread accumulator array with it (`BARY_MAX_K`, passed via
/// `-D` below) and the Rust dispatch asserts against it (generated into
/// `bary_consts.rs`) — defining it twice invites stack corruption in the
/// kernel the day one side moves without the other.
const BARY_MAX_EVAL_POINTS: usize = 8;

fn compile_kernel(src: &str, out_name: &str, have_nvcc: bool) {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let src_path = manifest_dir.join("kernels").join(src);
    let out_path = out_dir.join(out_name);

    println!("cargo:rerun-if-changed=kernels/{src}");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=CUDARC_NVCC_ARCH");
    println!("cargo:rerun-if-env-changed=LAMBDA_VM_NVCC_LINEINFO");

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
    // cross-compile for a different arch. If nvcc is present but no GPU is
    // detectable and no override is given, `detect_arch` hard-fails rather than
    // guessing an arch that would load on one GPU model and CPU-fall-back on
    // every other.
    let arch = env::var("CUDARC_NVCC_ARCH")
        .map(|a| to_real_arch(&a))
        .unwrap_or_else(|_| detect_arch());

    let mut cmd = Command::new(nvcc_path());
    cmd.args(["--cubin", "-O3", "-std=c++17", "-arch", &arch]);
    cmd.arg(format!("-DBARY_MAX_K={BARY_MAX_EVAL_POINTS}"));
    // SASS→source line mapping for Nsight Compute. Unlike -G this does not
    // change codegen, but keep it opt-in so production cubins stay byte-stable.
    if env::var("LAMBDA_VM_NVCC_LINEINFO").is_ok_and(|v| v != "0" && !v.is_empty()) {
        cmd.arg("-lineinfo");
    }
    let status = cmd
        .arg("-o")
        .arg(&out_path)
        .arg(&src_path)
        .status()
        .expect("failed to invoke nvcc — is CUDA installed and CUDA_HOME set?");

    if !status.success() {
        panic!("nvcc failed compiling {}", src_path.display());
    }
}

fn main() {
    // Rust-side mirror of the kernel cap; see BARY_MAX_EVAL_POINTS above.
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    fs::write(
        out_dir.join("bary_consts.rs"),
        format!(
            "/// Compile-time cap of the multi kernels' per-thread accumulator array\n\
             /// (`BARY_MAX_K` in barycentric.cu — single-sourced from build.rs).\n\
             /// Callers with more evaluation points fall back to the per-point kernels.\n\
             pub const BARY_MAX_EVAL_POINTS: usize = {BARY_MAX_EVAL_POINTS};\n"
        ),
    )
    .expect("failed to write bary_consts.rs");

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
