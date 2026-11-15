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
/// the empty-stub path in `Kernels::build` and never calls this.
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
/// `-D` at the barycentric.cu call site below) and the Rust dispatch asserts
/// against it (generated into `bary_consts.rs`) — defining it twice invites
/// stack corruption in the kernel the day one side moves without the other.
const BARY_MAX_EVAL_POINTS: usize = 8;

/// The RPX permutation variant of the default RPX cubin, `rpx_v<V>.cubin`:
/// `RPX_PERMUTE_VARIANT` in `kernels/rpx.cu`, a bit set of limb multiply (1),
/// limb square (2), unrolled `square_n` (4) and unrolled lane loops (8). The
/// source is also built at 0, the 64-bit multiply, as `rpx_v0.cubin`;
/// `rpx_paths::limb_permute` picks one of the two when the backend loads.
/// Mirrored into Rust (`rpx_paths::LIMB_PERMUTE_VARIANT`) and into the cubin
/// table (`device.rs`) by the generated files below.
const RPX_LIMB_PERMUTE_VARIANT: u32 = 5;

/// One kernel's nvcc job: its source under `kernels/`, its cubin under
/// `OUT_DIR`, and its `-D` defines.
struct Kernel {
    src: &'static str,
    out: String,
    defines: Vec<String>,
}

/// The kernels to compile, in the order `main` lists them. `add` emits each
/// kernel's rerun-if lines; `build` compiles them all at the end, in parallel.
#[derive(Default)]
struct Kernels(Vec<Kernel>);

impl Kernels {
    fn add(&mut self, src: &'static str, out_name: &str, defines: &[&str]) {
        println!("cargo:rerun-if-changed=kernels/{src}");
        self.0.push(Kernel {
            src,
            out: out_name.to_string(),
            defines: defines.iter().map(|d| d.to_string()).collect(),
        });
    }

    /// Compile every kernel to its cubin; with no nvcc, write empty stubs.
    ///
    /// The nvcc runs are independent processes (each its own source, flags
    /// and `-o` path; nvcc names its temporary files by process), so they run
    /// on a pool of `LAMBDA_VM_NVCC_JOBS` workers (default: cargo's
    /// `NUM_JOBS`, else the host's parallelism), the largest sources first.
    /// Each cubin is the serial build's byte for byte; `LAMBDA_VM_NVCC_JOBS=1`
    /// compiles one at a time, as before. Every failure is reported after
    /// the pool ends, each with nvcc's own output, then the build panics
    /// naming them all.
    fn build(self, have_nvcc: bool) {
        let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
        let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
        println!("cargo:rerun-if-env-changed=CUDA_HOME");
        println!("cargo:rerun-if-env-changed=CUDA_PATH");
        println!("cargo:rerun-if-env-changed=CUDARC_NVCC_ARCH");
        println!("cargo:rerun-if-env-changed=LAMBDA_VM_NVCC_LINEINFO");
        println!("cargo:rerun-if-env-changed=LAMBDA_VM_RPX_MAXRREGCOUNT");
        println!("cargo:rerun-if-env-changed=LAMBDA_VM_NVCC_JOBS");

        // When nvcc is missing from PATH, emit an empty cubin stub so the crate
        // still compiles. include_bytes! in src/device.rs needs the file to exist
        // at build time. Any runtime kernel call fails to load the empty module and
        // the caller falls back to CPU. We can't run GPU code without nvcc on the
        // build host anyway.
        if !have_nvcc {
            for k in &self.0 {
                fs::write(out_dir.join(&k.out), "").expect("failed to write empty cubin stub");
            }
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
        // every other. Read once for every kernel.
        let arch = env::var("CUDARC_NVCC_ARCH")
            .map(|a| to_real_arch(&a))
            .unwrap_or_else(|_| detect_arch());
        let mut common: Vec<String> = ["--cubin", "-O3", "-std=c++17", "-arch", &arch]
            .iter()
            .map(|a| a.to_string())
            .collect();
        // SASS→source line mapping for Nsight Compute. Unlike -G this does not
        // change codegen, but keep it opt-in so production cubins stay byte-stable.
        if env::var("LAMBDA_VM_NVCC_LINEINFO").is_ok_and(|v| v != "0" && !v.is_empty()) {
            common.push("-lineinfo".to_string());
        }
        // Opt-in per-thread register cap for the round-3 occupancy discriminator.
        // Capping registers lets more blocks reside per SM, tracing the RPX
        // permutation's throughput-vs-occupancy curve across separate builds (the
        // production grind kernel cannot carry a per-kernel `__launch_bounds__`, and
        // it is what this sweep must move). Empty/unset/"0" ⇒ no cap ⇒ production
        // cubins stay byte-stable, exactly like `-lineinfo` above. Diagnostic only;
        // never set in a production or bench build.
        if let Ok(r) = env::var("LAMBDA_VM_RPX_MAXRREGCOUNT")
            && !r.is_empty()
            && r != "0"
        {
            common.push(format!("-maxrregcount={r}"));
        }

        // The largest sources first, so the longest compiles start at once and
        // the pool's tail is short (a source's size stands in for its compile
        // time; ties keep `main`'s order).
        let size = |k: &Kernel| {
            fs::metadata(manifest_dir.join("kernels").join(k.src)).map_or(0, |m| m.len())
        };
        let mut order: Vec<usize> = (0..self.0.len()).collect();
        order.sort_by_key(|&i| std::cmp::Reverse(size(&self.0[i])));
        let workers = env::var("LAMBDA_VM_NVCC_JOBS")
            .ok()
            .or_else(|| env::var("NUM_JOBS").ok())
            .and_then(|v| v.trim().parse::<usize>().ok())
            .filter(|&n| n > 0)
            .unwrap_or_else(|| std::thread::available_parallelism().map_or(1, |n| n.get()))
            .min(order.len())
            .max(1);

        let next = std::sync::atomic::AtomicUsize::new(0);
        let results: std::sync::Mutex<Vec<Option<std::process::Output>>> =
            std::sync::Mutex::new((0..self.0.len()).map(|_| None).collect());
        std::thread::scope(|scope| {
            for _ in 0..workers {
                scope.spawn(|| {
                    loop {
                        let at = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        let Some(&i) = order.get(at) else {
                            break;
                        };
                        let k = &self.0[i];
                        let output = Command::new(nvcc_path())
                            .args(&common)
                            .args(&k.defines)
                            .arg("-o")
                            .arg(out_dir.join(&k.out))
                            .arg(manifest_dir.join("kernels").join(k.src))
                            .output()
                            .expect("failed to invoke nvcc — is CUDA installed and CUDA_HOME set?");
                        results.lock().expect("an nvcc worker panicked")[i] = Some(output);
                    }
                });
            }
        });

        // nvcc's own output, in `main`'s order whatever order the pool ran in;
        // then every failure, named.
        let results = results.into_inner().expect("an nvcc worker panicked");
        let mut failed = Vec::new();
        for (k, output) in self.0.iter().zip(results) {
            let output = output.expect("every kernel compiled");
            eprint!("{}", String::from_utf8_lossy(&output.stdout));
            eprint!("{}", String::from_utf8_lossy(&output.stderr));
            if !output.status.success() {
                eprintln!(
                    "nvcc failed compiling kernels/{} ({})",
                    k.src, output.status
                );
                failed.push(format!("kernels/{} -> {}", k.src, k.out));
            }
        }
        if !failed.is_empty() {
            panic!(
                "nvcc failed compiling {} kernel(s): {}",
                failed.len(),
                failed.join(", ")
            );
        }
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
    // The limb cubin's variant, and the two RPX cubins the backend chooses from.
    let v = RPX_LIMB_PERMUTE_VARIANT;
    fs::write(
        out_dir.join("rpx_variants.rs"),
        format!(
            "/// The permutation variant `rpx_v<V>.cubin` is built at (build.rs).\n\
             pub const LIMB_PERMUTE_VARIANT: u32 = {v};\n"
        ),
    )
    .expect("failed to write rpx_variants.rs");
    let cubin = |name: &str, file: &str| {
        format!("const {name}: &[u8] = include_bytes!(concat!(env!(\"OUT_DIR\"), \"/{file}\"));\n")
    };
    fs::write(
        out_dir.join("rpx_cubins.rs"),
        cubin("RPX_V0_CUBIN", "rpx_v0.cubin")
            + &cubin("RPX_LIMB_CUBIN", &format!("rpx_v{v}.cubin")),
    )
    .expect("failed to write rpx_cubins.rs");

    // Headers aren't compiled, so emit rerun-if-changed to rebuild on
    // header edits.
    println!("cargo:rerun-if-changed=kernels/goldilocks.cuh");
    println!("cargo:rerun-if-changed=kernels/ext3.cuh");
    println!("cargo:rerun-if-changed=kernels/ext3_inv.cuh");

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

    let mut kernels = Kernels::default();
    kernels.add("arith.cu", "arith.cubin", &[]);
    // The table fills from records (`src/fill.rs`), loaded on first use, never
    // by the backend. Pinned on the host by `tests/host_kat/fill_host_kat.cpp`.
    kernels.add("fill.cu", "fill.cubin", &[]);
    // The card-built LOAD, STORE and MEMW_R rows from the cached CPU logs
    // (`src/card_rows.rs`, `LAMBDA_VM_CARD_ROWS`: the default, words, and a
    // test's derive and registers modes); includes fill.cu.
    kernels.add("card_rows.cu", "card_rows.cubin", &[]);
    kernels.add("ntt.cu", "ntt.cubin", &[]);
    // The column-major LDE engine (`src/lde_cm.rs`). Pinned on the host by
    // `tests/host_kat/ntt_cm_host_kat.cpp` (`make test-ntt-cm-host-kat`).
    kernels.add("ntt_cm.cu", "ntt_cm.cubin", &[]);
    kernels.add("keccak.cu", "keccak.cubin", &[]);
    let bary_define = format!("-DBARY_MAX_K={BARY_MAX_EVAL_POINTS}");
    kernels.add("barycentric.cu", "barycentric.cubin", &[&bary_define]);
    kernels.add("deep.cu", "deep.cubin", &[]);
    kernels.add("fri.cu", "fri.cubin", &[]);
    kernels.add("inverse.cu", "inverse.cubin", &[]);
    // RPX256 (XHash12) leaves, parents and grind — the algebraic hash's device
    // kernels, pinned on the host by `tests/host_kat/rpx_host_kat.cpp` at every
    // variant. Built twice: at 0 (no `-D`, the source's default: the 64-bit
    // multiply) and at the limb variant, the default the backend loads.
    kernels.add("rpx.cu", "rpx_v0.cubin", &[]);
    let limb_define = format!("-DRPX_PERMUTE_VARIANT={RPX_LIMB_PERMUTE_VARIANT}");
    kernels.add(
        "rpx.cu",
        &format!("rpx_v{RPX_LIMB_PERMUTE_VARIANT}.cubin"),
        &[&limb_define],
    );
    kernels.add("logup.cu", "logup.cubin", &[]);
    kernels.add("constraint_interp.cu", "constraint_interp.cubin", &[]);
    // The BLAKE3 kernels' round count is a compile-time knob with the same
    // polarity as the host tree's `blake3-6round` feature: 7 rounds (standard
    // BLAKE3) unless the feature selects the 6-round variant. The `.cu` defaults
    // to 7 on its own, so a stale `-D` can never silently pick 6.
    // `CARGO_FEATURE_*`, not `cfg!(feature = ..)`: cargo passes a build script
    // the active features as environment variables and does NOT cfg them into
    // its compilation, so the `cfg!` form here would silently always be false.
    let blake3_defines: &[&str] = if env::var_os("CARGO_FEATURE_BLAKE3_6ROUND").is_some() {
        &["-DBLAKE3_ROUNDS=6"]
    } else {
        &[]
    };
    kernels.add("blake3.cu", "blake3.cubin", blake3_defines);
    // The WHIR base proof's BLAKE3 kernels (`src/blake3_whir.rs`): `blake3.cu`
    // instantiated at 7 rounds by the file itself, so no define is passed —
    // least of all the 6-round one above. Loaded on first use, never by the
    // backend.
    println!("cargo:rerun-if-changed=kernels/blake3.cu");
    kernels.add("blake3_whir.cu", "blake3_whir.cubin", &[]);
    // The WHIR path's own kernels. They carry no compile-time knob, so they
    // take the empty define list rather than gaining a second signature.
    // D-HASH stage 1's Poseidon1 width-16 MEASUREMENT kernels (`src/p1w16.rs`),
    // pinned on the host by `tests/host_kat/p1w16_host_kat.cpp`. Loaded only by
    // that module, never by the backend.
    println!("cargo:rerun-if-changed=kernels/p1w16_constants.cuh");
    println!("cargo:rerun-if-changed=kernels/p1w8_constants.cuh");
    kernels.add("p1w16.cu", "p1w16.cubin", &[]);
    kernels.add("sumcheck.cu", "sumcheck.cubin", &[]);
    kernels.add("whir_fold.cu", "whir_fold.cubin", &[]);
    // The first-round cosets evaluated from the polynomial (`src/whir_coset.rs`),
    // a measurement module loaded on first use, never by the backend.
    kernels.add("whir_coset.cu", "whir_coset.cubin", &[]);
    kernels.build(have_nvcc);
}
