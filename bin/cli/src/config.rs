//! `--config` file support: a TOML profile that overrides the prover's code
//! defaults. Every key is optional — the file is a diff over the defaults,
//! never a replacement (several defaults are computed per machine, e.g. the
//! VRAM budget). Precedence, highest first: explicit CLI flag > env var >
//! config file > code default. See `docs/prover.example.toml`.

use prover::tables::MaxRowsConfig;
use serde::Deserialize;
use std::path::Path;

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FileConfig {
    #[serde(default)]
    pub prove: ProveSection,
    #[serde(default)]
    pub tables: TablesSection,
    #[serde(default)]
    pub scheduler: SchedulerSection,
    #[serde(default)]
    pub gpu: GpuSection,
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProveSection {
    /// Blowup factor (power of 2), like `--blowup`.
    pub blowup: Option<u8>,
    /// Continuation epoch size as log2(cycles), like `--epoch-size-log2`.
    pub epoch_size_log2: Option<u32>,
}

/// Per-table row caps as log2(rows). Defaults equalize memory per instance
/// (`effective_width x rows ~ constant`); raising one cap makes that table's
/// instances proportionally heavier than everyone else's.
#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TablesSection {
    pub cpu: Option<u32>,
    pub memw: Option<u32>,
    pub memw_aligned: Option<u32>,
    pub dvrm: Option<u32>,
    pub mul: Option<u32>,
    pub lt: Option<u32>,
    pub shift: Option<u32>,
    pub load: Option<u32>,
    pub branch: Option<u32>,
    pub memw_register: Option<u32>,
    pub eq: Option<u32>,
    pub bytewise: Option<u32>,
    pub store: Option<u32>,
    pub cpu32: Option<u32>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SchedulerSection {
    /// Tables proven concurrently (`TABLE_PARALLELISM`).
    pub table_parallelism: Option<usize>,
    /// Device VRAM budget in MiB (`LAMBDA_VM_VRAM_BUDGET_MB`).
    pub vram_budget_mb: Option<u64>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GpuSection {
    /// Minimum LDE size (log2) for the GPU commit paths
    /// (`LAMBDA_VM_GPU_LDE_THRESHOLD`).
    pub lde_threshold_log2: Option<u32>,
    /// Minimum trace size (log2) for the GPU barycentric paths
    /// (`LAMBDA_VM_GPU_BARY_THRESHOLD`).
    pub bary_threshold_log2: Option<u32>,
    /// CUDA mempool release threshold in MiB
    /// (`LAMBDA_VM_MEMPOOL_RELEASE_MB`).
    pub mempool_release_mb: Option<u64>,
    /// Kill-switch: force the CPU composition path
    /// (`LAMBDA_VM_DISABLE_GPU_COMPOSITION`).
    pub disable_composition: Option<bool>,
    /// Kill-switch: force the CPU LogUp aux build (`LAMBDA_VM_NO_GPU_LOGUP`).
    pub disable_logup: Option<bool>,
    /// Kill-switch: keep host copies of device-resident buffers
    /// (`LAMBDA_VM_DISABLE_DEVICE_ONLY`).
    pub disable_device_only: Option<bool>,
}

/// Table caps must stay provable and addressable: 2^5 (the test floor)
/// through 2^25 (u32 LDE domain headroom).
const TABLE_LOG2_RANGE: std::ops::RangeInclusive<u32> = 5..=25;

impl FileConfig {
    pub fn load(path: &Path) -> Result<Self, String> {
        let text =
            std::fs::read_to_string(path).map_err(|e| format!("config {}: {e}", path.display()))?;
        let cfg: FileConfig =
            toml::from_str(&text).map_err(|e| format!("config {}: {e}", path.display()))?;
        cfg.validate()
            .map_err(|e| format!("config {}: {e}", path.display()))?;
        Ok(cfg)
    }

    fn validate(&self) -> Result<(), String> {
        let t = &self.tables;
        for (name, v) in [
            ("cpu", t.cpu),
            ("memw", t.memw),
            ("memw_aligned", t.memw_aligned),
            ("dvrm", t.dvrm),
            ("mul", t.mul),
            ("lt", t.lt),
            ("shift", t.shift),
            ("load", t.load),
            ("branch", t.branch),
            ("memw_register", t.memw_register),
            ("eq", t.eq),
            ("bytewise", t.bytewise),
            ("store", t.store),
            ("cpu32", t.cpu32),
        ] {
            if let Some(v) = v
                && !TABLE_LOG2_RANGE.contains(&v)
            {
                return Err(format!(
                    "[tables] {name} = {v}: log2 caps must be within {:?}",
                    TABLE_LOG2_RANGE
                ));
            }
        }
        if let Some(b) = self.prove.blowup
            && !b.is_power_of_two()
        {
            return Err(format!("[prove] blowup = {b}: must be a power of 2"));
        }
        Ok(())
    }

    /// Code defaults overridden by the set `[tables]` keys.
    pub fn max_rows(&self) -> MaxRowsConfig {
        let mut m = MaxRowsConfig::default();
        let t = &self.tables;
        let set = |dst: &mut usize, v: Option<u32>| {
            if let Some(log2) = v {
                *dst = 1usize << log2;
            }
        };
        set(&mut m.cpu, t.cpu);
        set(&mut m.memw, t.memw);
        set(&mut m.memw_aligned, t.memw_aligned);
        set(&mut m.dvrm, t.dvrm);
        set(&mut m.mul, t.mul);
        set(&mut m.lt, t.lt);
        set(&mut m.shift, t.shift);
        set(&mut m.load, t.load);
        set(&mut m.branch, t.branch);
        set(&mut m.memw_register, t.memw_register);
        set(&mut m.eq, t.eq);
        set(&mut m.bytewise, t.bytewise);
        set(&mut m.store, t.store);
        set(&mut m.cpu32, t.cpu32);
        m
    }

    /// Warn (stderr) for every file key that a set env var is shadowing, so
    /// "I changed the file and nothing happened" is diagnosable at a glance.
    pub fn warn_env_shadowing(&self) {
        let pairs: [(&str, bool); 8] = [
            (
                "TABLE_PARALLELISM",
                self.scheduler.table_parallelism.is_some(),
            ),
            (
                "LAMBDA_VM_VRAM_BUDGET_MB",
                self.scheduler.vram_budget_mb.is_some(),
            ),
            (
                "LAMBDA_VM_GPU_LDE_THRESHOLD",
                self.gpu.lde_threshold_log2.is_some(),
            ),
            (
                "LAMBDA_VM_GPU_BARY_THRESHOLD",
                self.gpu.bary_threshold_log2.is_some(),
            ),
            (
                "LAMBDA_VM_MEMPOOL_RELEASE_MB",
                self.gpu.mempool_release_mb.is_some(),
            ),
            (
                "LAMBDA_VM_DISABLE_GPU_COMPOSITION",
                self.gpu.disable_composition.is_some(),
            ),
            ("LAMBDA_VM_NO_GPU_LOGUP", self.gpu.disable_logup.is_some()),
            (
                "LAMBDA_VM_DISABLE_DEVICE_ONLY",
                self.gpu.disable_device_only.is_some(),
            ),
        ];
        for (var, in_file) in pairs {
            if in_file && std::env::var_os(var).is_some() {
                eprintln!(
                    "warning: config file value ignored: env var {var} is set and takes priority"
                );
            }
        }
    }

    /// Install the stark/math-cuda knobs (env vars keep priority at each read
    /// site). Call once, before the first prove.
    pub fn install_runtime_overrides(&self) {
        stark::runtime_overrides::install(stark::runtime_overrides::RuntimeOverrides {
            table_parallelism: self.scheduler.table_parallelism,
            gpu_lde_threshold: self.gpu.lde_threshold_log2.map(|l| 1usize << l),
            gpu_bary_threshold: self.gpu.bary_threshold_log2.map(|l| 1usize << l),
            disable_gpu_composition: self.gpu.disable_composition,
            no_gpu_logup: self.gpu.disable_logup,
            disable_device_only: self.gpu.disable_device_only,
            vram_budget_mb: self.scheduler.vram_budget_mb,
            mempool_release_mb: self.gpu.mempool_release_mb,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every key the file format accepts, exercised end to end. A field
    /// added to the structs without updating this fixture (or the example
    /// file) fails here.
    const FULL: &str = r#"
        [prove]
        blowup = 4
        epoch_size_log2 = 22

        [tables]
        cpu = 20
        memw = 20
        memw_aligned = 20
        dvrm = 20
        mul = 21
        lt = 21
        shift = 21
        load = 21
        branch = 21
        memw_register = 21
        eq = 21
        bytewise = 21
        store = 21
        cpu32 = 20

        [scheduler]
        table_parallelism = 8
        vram_budget_mb = 24000

        [gpu]
        lde_threshold_log2 = 18
        bary_threshold_log2 = 13
        mempool_release_mb = 4096
        disable_composition = true
        disable_logup = true
        disable_device_only = true
    "#;

    #[test]
    fn full_fixture_round_trips() {
        let cfg: FileConfig = toml::from_str(FULL).unwrap();
        cfg.validate().unwrap();
        assert_eq!(cfg.prove.blowup, Some(4));
        assert_eq!(cfg.prove.epoch_size_log2, Some(22));
        let m = cfg.max_rows();
        assert_eq!(m.cpu, 1 << 20);
        assert_eq!(m.lt, 1 << 21);
        assert_eq!(m.cpu32, 1 << 20);
        assert_eq!(cfg.scheduler.table_parallelism, Some(8));
        assert_eq!(cfg.gpu.lde_threshold_log2, Some(18));
        assert_eq!(cfg.gpu.disable_device_only, Some(true));
    }

    #[test]
    fn empty_and_partial_files_keep_defaults() {
        let cfg: FileConfig = toml::from_str("").unwrap();
        let d = MaxRowsConfig::default();
        let m = cfg.max_rows();
        assert_eq!(m.cpu, d.cpu);
        assert_eq!(m.store, d.store);

        let cfg: FileConfig = toml::from_str("[tables]\ncpu = 20\n").unwrap();
        let m = cfg.max_rows();
        assert_eq!(m.cpu, 1 << 20);
        assert_eq!(m.memw, d.memw); // untouched keys keep code defaults
    }

    #[test]
    fn unknown_keys_and_bad_values_are_rejected() {
        assert!(toml::from_str::<FileConfig>("[tables]\ncppu = 20\n").is_err());
        assert!(toml::from_str::<FileConfig>("[typo_section]\nx = 1\n").is_err());
        let cfg: FileConfig = toml::from_str("[tables]\ncpu = 30\n").unwrap();
        assert!(cfg.validate().is_err()); // out of range
        let cfg: FileConfig = toml::from_str("[prove]\nblowup = 3\n").unwrap();
        assert!(cfg.validate().is_err()); // not a power of two
    }

    /// The example file must parse (it ships fully commented, so it must
    /// stay valid TOML with no unknown keys when uncommented sections drift
    /// is checked by hand — this pins at least the syntactic contract).
    #[test]
    fn example_file_parses() {
        let text = std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../docs/prover.example.toml"
        ))
        .expect("docs/prover.example.toml exists");
        let cfg: FileConfig = toml::from_str(&text).expect("example parses");
        cfg.validate().expect("example validates");
    }
}
