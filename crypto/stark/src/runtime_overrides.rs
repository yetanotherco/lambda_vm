//! Process-wide runtime overrides for the prover's tuning knobs, installed
//! once at startup (the CLI's `--config` file) and consulted by the knob
//! readers with the precedence `env var > override > code default` — a set
//! env var always wins, so existing scripts keep working unchanged.
//!
//! Install before the first prove: the readers cache their resolved value on
//! first use, so a later install would be silently ignored (`install` returns
//! whether it won the race).

use std::sync::OnceLock;

/// Values a config file may override. `None` = not set → code default.
#[derive(Debug, Clone, Default)]
pub struct RuntimeOverrides {
    /// Tables proven concurrently (`TABLE_PARALLELISM`).
    pub table_parallelism: Option<usize>,
    /// Minimum LDE size for the GPU commit paths
    /// (`LAMBDA_VM_GPU_LDE_THRESHOLD`).
    pub gpu_lde_threshold: Option<usize>,
    /// Minimum trace size for the GPU barycentric paths
    /// (`LAMBDA_VM_GPU_BARY_THRESHOLD`).
    pub gpu_bary_threshold: Option<usize>,
    /// Kill-switch: skip the GPU composition path
    /// (`LAMBDA_VM_DISABLE_GPU_COMPOSITION`).
    pub disable_gpu_composition: Option<bool>,
    /// Kill-switch: skip the GPU LogUp paths (`LAMBDA_VM_NO_GPU_LOGUP`).
    pub no_gpu_logup: Option<bool>,
    /// Kill-switch: keep host copies of device-resident buffers
    /// (`LAMBDA_VM_DISABLE_DEVICE_ONLY`).
    pub disable_device_only: Option<bool>,
    /// Device VRAM budget in MiB (`LAMBDA_VM_VRAM_BUDGET_MB`).
    pub vram_budget_mb: Option<u64>,
    /// CUDA mempool release threshold in MiB
    /// (`LAMBDA_VM_MEMPOOL_RELEASE_MB`).
    pub mempool_release_mb: Option<u64>,
}

static OVERRIDES: OnceLock<RuntimeOverrides> = OnceLock::new();

/// Install the overrides. Returns `false` if a set was already installed (the
/// first install wins; callers should treat `false` as a startup-order bug).
pub fn install(overrides: RuntimeOverrides) -> bool {
    #[cfg(feature = "cuda")]
    let device = (overrides.vram_budget_mb, overrides.mempool_release_mb);
    let won = OVERRIDES.set(overrides).is_ok();
    // The device-side knobs are read inside math-cuda; forward them.
    #[cfg(feature = "cuda")]
    if won {
        math_cuda::device::set_runtime_overrides(device.0, device.1);
    }
    won
}

pub(crate) fn get() -> &'static RuntimeOverrides {
    static EMPTY: RuntimeOverrides = RuntimeOverrides {
        table_parallelism: None,
        gpu_lde_threshold: None,
        gpu_bary_threshold: None,
        disable_gpu_composition: None,
        no_gpu_logup: None,
        disable_device_only: None,
        vram_budget_mb: None,
        mempool_release_mb: None,
    };
    OVERRIDES.get().unwrap_or(&EMPTY)
}
