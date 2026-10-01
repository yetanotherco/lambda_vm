#!/usr/bin/env bash
# One-time setup of the dedicated GPU profiling box (Ubuntu 22.04/24.04,
# NVIDIA driver already installed). Idempotent: safe to re-run. Needs sudo.
#
# What it does NOT do: install the NVIDIA driver or the repo build toolchain
# (clang-18, rust, sysroot) — follow scripts/SERVER_SETUP.md for those first.
#
# After the first run: REBOOT once so the nvidia module picks up the
# profiling permission (NVreg_RestrictProfilingToAdminUsers=0).
set -euo pipefail

echo "==> Sanity: driver + GPU visible"
nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader || {
  echo "ERROR: nvidia-smi failed — install the NVIDIA driver first." >&2
  exit 1
}

CC="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' ')"
if [[ "$CC" == 12.* ]]; then
  echo "    Blackwell (compute $CC) — CUDA >= 12.8 and nsys/ncu >= 2025.1 are REQUIRED."
fi

echo "==> apt packages"
sudo apt-get update -qq
sudo apt-get install -y -qq python3 sqlite3 tmux git curl wget gnupg time || true
# perf packaging differs: Debian ships `linux-perf`, Ubuntu `linux-tools-*`.
if grep -qi '^ID=ubuntu' /etc/os-release; then
  sudo apt-get install -y -qq linux-tools-common "linux-tools-$(uname -r)" || true
else
  sudo apt-get install -y -qq linux-perf || true
fi
# eBPF tools for off-CPU flamegraphs (offcputime-bpfcc)
sudo apt-get install -y -qq bpfcc-tools || echo "note: bpfcc-tools unavailable; off-CPU flamegraphs need it"
sudo apt-get install -y -qq hyperfine || echo "note: hyperfine not in this distro's apt; install via cargo if needed"

echo "==> NVIDIA CUDA apt repo (for CUDA toolkit + current nsight tools)"
if [[ ! -f /usr/share/keyrings/cuda-archive-keyring.gpg ]]; then
  . /etc/os-release
  DISTRO="ubuntu${VERSION_ID/./}"
  wget -qO /tmp/cuda-keyring.deb \
    "https://developer.download.nvidia.com/compute/cuda/repos/${DISTRO}/x86_64/cuda-keyring_1.1-1_all.deb"
  sudo dpkg -i /tmp/cuda-keyring.deb
  sudo apt-get update -qq
fi

echo "==> CUDA toolkit (nvcc) — needed to build real cubins"
if ! command -v nvcc >/dev/null && [[ ! -x /usr/local/cuda/bin/nvcc ]]; then
  # 12.8 matches the cudarc pin (crypto/math-cuda/Cargo.toml). A newer toolkit
  # is fine too (cubins are SASS; the pin covers driver symbols).
  sudo apt-get install -y cuda-toolkit-12-8
else
  echo "    nvcc present, skipping"
fi

echo "==> NVTX v2 dispatcher (libnvToolsExt) — the nvtx feature dlopens it at runtime"
# CUDA >= 12.9 removed NVTX v2 from the toolkit (migrate-to-v3 notice in the
# 12.9 release notes). Without the .so the nvtx feature's ranges silently
# no-op. No sudo needed: extract it from NVIDIA's cuda-nvtx-12-8 .deb into
# ~/nvtx and point LAMBDA_VM_NVTX_LIB at it (works on Debian and Ubuntu alike,
# with or without the NVIDIA apt repo).
if /sbin/ldconfig -p 2>/dev/null | grep -q libnvToolsExt \
  || [[ -e /usr/local/cuda/lib64/libnvToolsExt.so.1 || -e "$HOME/nvtx/libnvToolsExt.so.1" ]]; then
  echo "    present"
else
  mkdir -p "$HOME/nvtx" && pushd "$HOME/nvtx" >/dev/null
  NVREPO=https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64
  NVPKG="$(curl -s $NVREPO/ | grep -oE 'cuda-nvtx-12-8[a-zA-Z0-9._-]*\.deb' | sort -u | head -1)"
  if [[ -n "$NVPKG" ]] && curl -sO "$NVREPO/$NVPKG" && dpkg -x "$NVPKG" extract/; then
    NVTX_SO="$(find extract -name 'libnvToolsExt.so*' -type f | head -1)"
    cp "$NVTX_SO" libnvToolsExt.so.1
    echo "    extracted to ~/nvtx/libnvToolsExt.so.1"
    echo "    add to your shell profile: export LAMBDA_VM_NVTX_LIB=\$HOME/nvtx/libnvToolsExt.so.1"
  else
    echo "WARNING: could not fetch cuda-nvtx-12-8 — NVTX ranges will no-op."
    echo "         Get libnvToolsExt.so.1 from any CUDA <=12.8 install and set LAMBDA_VM_NVTX_LIB."
  fi
  popd >/dev/null
fi

echo "==> Nsight Systems + Nsight Compute (must be 2025.1+ for sm_120)"
# The cuda repo ships them standalone; toolkit meta-packages may carry older ones.
sudo apt-get install -y nsight-systems nsight-compute 2>/dev/null || \
  sudo apt-get install -y cuda-nsight-systems-12-8 cuda-nsight-compute-12-8 2>/dev/null || \
  echo "WARNING: could not apt-install nsight tools; download from developer.nvidia.com/nsight-systems"
command -v nsys >/dev/null && nsys --version | head -1
command -v ncu >/dev/null && ncu --version | tail -1

echo "==> perf/eBPF sysctls (persistent)"
printf 'kernel.perf_event_paranoid=-1\nkernel.kptr_restrict=0\n' | \
  sudo tee /etc/sysctl.d/99-gpu-profiling.conf >/dev/null
sudo sysctl --system >/dev/null
echo "    perf_event_paranoid=$(cat /proc/sys/kernel/perf_event_paranoid)"

echo "==> GPU perf counters for non-root ncu/nsys --gpu-metrics (persistent; needs reboot)"
echo 'options nvidia NVreg_RestrictProfilingToAdminUsers=0' | \
  sudo tee /etc/modprobe.d/nvidia-profiling.conf >/dev/null
sudo update-initramfs -u -k "$(uname -r)" >/dev/null 2>&1 || true

echo "==> Kill measurement noise"
sudo systemctl disable --now unattended-upgrades 2>/dev/null || true
sudo systemctl disable --now apt-daily.timer apt-daily-upgrade.timer 2>/dev/null || true

echo "==> cargo profiling tools (as current user)"
if command -v cargo >/dev/null; then
  for tool in flamegraph inferno samply; do
    command -v "$tool" >/dev/null || cargo install "$tool" --locked || true
  done
else
  echo "note: cargo not on PATH — install rust per scripts/SERVER_SETUP.md, then: cargo install flamegraph inferno samply"
fi

echo
echo "==> DONE. Next steps:"
echo "  1. REBOOT (activates the nvidia profiling permission)."
echo "  2. Verify counters: ncu --query-metrics >/dev/null && echo counters-ok"
echo "  3. Lock clocks:     sudo scripts/profiling/bench_mode.sh on"
echo "  4. Sanity-run GPU:  make test-cuda-integration"
echo "  5. Build a fixture: tooling/ethrex-fixtures/target/release/ethrex-fixtures 5 executor/tests/ethrex_5_transfers.bin distinct  (cargo build --release in tooling/ethrex-fixtures first)"
echo "  6. First profile:   scripts/profiling/run_profile.sh --nsys executor/program_artifacts/rust/ethrex.elf --private-input executor/tests/ethrex_5_transfers.bin"
