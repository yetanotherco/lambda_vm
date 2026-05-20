# infra/

Scripts for renting and provisioning a Scaleway Elastic Metal server for
benchmark / feature-testing work.

| Script | Runs on | Purpose |
|---|---|---|
| `rent_baremetal.sh <name>` | local | Creates the server (hourly billing, Debian, fr-par-2) and hands off to `provision_server.sh`. |
| `provision_server.sh <ip>` | local | Waits for sshd, `scp`s the GitHub deploy key, runs `provision.sh` on the server over SSH. Re-runnable. |
| `provision.sh` | remote | Installs toolchain, creates `admin`/`app` users, clones `lambda_vm`, hardens sshd. |

## Prerequisites

- `scw` CLI configured with the **`vm`** profile active (the script refuses
  to run on any other profile). Check with `scw config get active-profile`
  or set with `export SCW_PROFILE=vm`.
- `jq`, `ssh`, `scp` on `$PATH`.
- Your SSH key registered in the Scaleway project so it gets installed on
  the server: `scw iam ssh-key list project-id=<project-id>`.
- A GitHub deploy key with read access to `yetanotherco/lambda_vm` at
  `~/.ssh/lambda_vm_read_only`. Create one with:

  ```bash
  ssh-keygen -t ed25519 -f ~/.ssh/lambda_vm_read_only \
      -C "lambda_vm read-only deploy key" -N ""
  ```

  Then paste `~/.ssh/lambda_vm_read_only.pub` into **GitHub → repo Settings
  → Deploy keys → Add deploy key** (leave "Allow write access" unchecked).

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
infra/rent_baremetal.sh bench-$(date +%s)
```

After it finishes, log in as:

```bash
ssh admin@<ip>    # passwordless sudo
ssh app@<ip>      # workload user, no sudo, has /workspace/lambda_vm cloned
```

Root SSH is disabled at the end of provisioning.

## Re-provision an existing server

If `provision.sh` failed partway or you want to re-apply changes, point
`provision_server.sh` at the IP directly. It's idempotent.

```bash
# Before hardening (root still works):
infra/provision_server.sh <ip>

# After hardening (root SSH is dead, use admin):
SSH_USER=admin infra/provision_server.sh <ip>
```

The wrapper switches to `sudo bash -s` automatically when `SSH_USER` isn't
root.

### Stuck on "Waiting for sshd"?

Scaleway recycles IPs across rentals. If you've SSH'd to that IP before,
your `~/.ssh/known_hosts` has the previous server's host key, the new
server returns a different one, and `ssh -o StrictHostKeyChecking=accept-new`
refuses the connection — so the retry loop runs forever. `rent_baremetal.sh`
handles this automatically; for a standalone `provision_server.sh` run,
clear the stale entry first:

```bash
ssh-keygen -R <ip>
infra/provision_server.sh <ip>
```

## Tear down

Hourly billing keeps ticking until you delete the server:

```bash
scw baremetal server delete <server-id> zone=fr-par-2
```

`rent_baremetal.sh` prints this line with the right ID at the end of a
successful run.

## Configuration

Everything has a working default; override via env var only when needed.

| Var | Default | Used by |
|---|---|---|
| `SCW_ZONE` | `fr-par-2` | `rent_baremetal.sh` |
| `SCW_TYPE` | `EM-I320E-NVME` | `rent_baremetal.sh` |
| `SCW_OS_ID` | Debian 12 UUID | `rent_baremetal.sh` |
| `SCW_PROJECT_ID` | team project UUID | `rent_baremetal.sh` |
| `READY_TIMEOUT` | `1800` (s) | `rent_baremetal.sh` |
| `PROVISION_FILE` | `infra/provision.sh` | both wrappers |
| `GITHUB_SSH_KEY_FILE` | `~/.ssh/lambda_vm_read_only` | `provision_server.sh` |
| `SSH_USER` | `root` | `provision_server.sh` |
