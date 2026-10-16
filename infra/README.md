# infra/

Scripts for renting and provisioning a Scaleway Elastic Metal server for
benchmark / feature-testing work.

| Script                     | Runs on | Purpose                                                                                                |
|----------------------------|---------|--------------------------------------------------------------------------------------------------------|
| `rent_baremetal.sh <name>` | local   | Creates the server (hourly billing, Debian, fr-par-2) and hands off to `provision_server.sh`.          |
| `provision_server.sh <ip>` | local   | Waits for sshd, runs `provision.sh` on the server over SSH. Re-runnable.                               |
| `provision.sh`             | remote  | Installs toolchain, creates `admin`/`app` users, clones `lambda_vm`, hardens sshd.                     |

## Prerequisites

1. Install `scw` and `jq`:
   ```bash
   brew install scw jq     # macOS
   ```

2. Create the `vm` scw profile (script refuses any other profile name):
   ```bash
   scw init --profile vm
   ```

## Rent + provision a new server

```bash
infra/rent_baremetal.sh test-1
```

End to end in one command — creates the server, waits for both
`status=ready` and `install.status=completed`, then provisions it (apt
packages, `admin`/`app` users, Rust toolchain, gh CLI, Claude Code,
lambda-vm sysroot, repo clone, ssh hardening).

Use a unique name (Scaleway rejects duplicates):

```bash
infra/rent_baremetal.sh <server_name>
```

After it finishes, log in as:

```bash
ssh admin@<ip>    # passwordless sudo
ssh app@<ip>      # workload user, no sudo, has ~/lambda_vm cloned
```

Root SSH is disabled at the end of provisioning.

## Re-provision an existing server

If `provision.sh` failed partway, or you want to re-apply changes, point
`provision_server.sh` at the IP directly. It's idempotent.

```bash
# Before hardening (root still works):
infra/provision_server.sh <ip>

# After hardening (root SSH is dead, use admin):
SSH_USER=admin infra/provision_server.sh <ip>
```

The wrapper switches to `sudo bash -s` automatically when `SSH_USER` isn't
root.

## Delete the server

To delete the server:

```bash
scw baremetal server delete <server_id> zone=<zone>
```

When `rent_baremetal.sh` finishes it prints the exact command (with
`<server_id>` and `<zone>` filled in) — copy that line when you're done
with the server.

## Configuration

Everything has a working default; override via env var only when needed.

| Var | Default | Used by | Notes |
|---|---|---|---|
| `SCW_TYPE` | `EM-I320E-NVME` | `rent_baremetal.sh` | Scaleway commercial type. Must have an `hourly` offer in `$SCW_ZONE` or the script refuses. |
| `SCW_ZONE` | `fr-par-2` | `rent_baremetal.sh` | One of `fr-par-1`, `fr-par-2`, `nl-ams-1`, `nl-ams-2`, `pl-waw-2`, `pl-waw-3`. |
| `SCW_OS_ID` | `83640d93-...` (Debian 12) | `rent_baremetal.sh` | Must have `cloud_init_supported: true`. |
| `SCW_PROJECT_ID` | `946cfb34-...` (lambda_vm) | `rent_baremetal.sh` | Determines which scw IAM SSH keys get installed. |
| `READY_TIMEOUT` | `1800` (s) | `rent_baremetal.sh` | How long to wait for `status=ready && install.status=completed`. |
| `PROVISION_FILE` | `<script_dir>/provision.sh` | both wrappers | Path to the remote provisioning script. |
| `SSH_USER` | `root` | `provision_server.sh` | Switch to `admin` for re-runs after sshd hardening. |
| `LLVM_VERSION` | `21` | provisioning scripts | LLVM major installed from apt.llvm.org for RISC-V assembly builds. |

### `SCW_TYPE` options

| Type | CPU | RAM | Disk | Price (€/h) |
|---|---|---|---|---|
| `EM-I220E-NVME` | AMD EPYC 8124P (16c/32t @ 2.5 GHz) | 128 GB | 2× 960 GB NVMe | 0.548 |
| `EM-I320E-NVME` | AMD EPYC 8224P (24c/48t @ 2.5 GHz) | 192 GB | 2× 1.92 TB NVMe | 0.822 (default) |
| `EM-I420E-NVME` | AMD EPYC 8324P (32c/64t @ 2.6 GHz) | 256 GB | 2× 1.92 TB NVMe | 1.096 |

### `SCW_ZONE` options

| Zone | Location |
|---|---|
| `fr-par-1` | Paris, France |
| `fr-par-2` | Paris, France (default) |
| `nl-ams-1` | Amsterdam, Netherlands |
| `nl-ams-2` | Amsterdam, Netherlands |
| `pl-waw-2` | Warsaw, Poland |
| `pl-waw-3` | Warsaw, Poland |
