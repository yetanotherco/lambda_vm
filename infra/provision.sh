#!/bin/bash
# Provision a freshly rented Scaleway Elastic Metal Debian server.
# Invoked remotely from infra/provision_server.sh as:
#   ssh root@<ip> bash -s < infra/provision.sh
#
# Optional input (placed by provision_server.sh before SSH'ing in):
#   /tmp/lambda_vm_read_only_key   GitHub deploy key. If present it is installed
#                               to /home/app/.ssh/lambda_vm_read_only and used to
#                               clone yetanotherco/lambda_vm. If absent the
#                               clone is skipped.
#
# Idempotent — safe to re-run.

set -euo pipefail

log() { printf '\n=== %s ===\n' "$*"; }

# --- 1. apt update + upgrade -------------------------------------------------
log "apt update + upgrade"
export DEBIAN_FRONTEND=noninteractive
APT_OPTS=(-y -o Dpkg::Options::=--force-confdef -o Dpkg::Options::=--force-confold)

# Scaleway baremetal Debian ships grub-cloud-amd64; its postinst (fired as a
# trigger by initramfs-tools / shim-signed / kernel upgrades) runs grub-install
# against an ext2 root and fails ("will not proceed with blocklists"). The
# package isn't load-bearing on UEFI baremetal — purge it before any upgrade.
apt-get purge -y grub-cloud-amd64 2>/dev/null || true

apt-get update -y
apt-get upgrade "${APT_OPTS[@]}"

# --- 2. apt packages ---------------------------------------------------------
log "apt install base packages + clang/lld/llvm + xz-utils"
apt-get install "${APT_OPTS[@]}" \
    ca-certificates curl wget gnupg vim git zip unzip openssl libssl-dev \
    build-essential rsyslog htop rsync pkg-config locales ufw \
    clang lld llvm xz-utils

# --- 3. users: admin (sudo) + app (no sudo) ----------------------------------
log "users: admin (sudo) + app (no sudo)"
for u in admin app; do
    if ! id "$u" >/dev/null 2>&1; then
        useradd -m -s /bin/bash "$u"
    fi
done
echo 'admin ALL=(ALL) NOPASSWD:ALL' > /etc/sudoers.d/90-admin
chmod 0440 /etc/sudoers.d/90-admin

# --- 4. authorized_keys for admin and app ------------------------------------
log "authorized_keys: propagate root's keys + append hardcoded team keys"
if [ ! -s /root/.ssh/authorized_keys ]; then
    echo "ERROR: /root/.ssh/authorized_keys missing or empty — refusing to harden sshd." >&2
    exit 1
fi
TEAM_KEYS=(
    "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIFzvQKhE/xqRxHbit/dZNej7T5eVLmF8CAGL7to6o3QY joaquin@mail.com"
    "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIA2GAeixuqP4XwujuSK9KDgdmyglGzlQQsXztnve+bra gabriel@mail.com"
    "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIKQnPPUb4gzmsmjDP98mNKXbpHrp9bIIL7QiRjyWEG6f julian@mail.com"
    "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIJ6mrcWIyU+/LrNZLivNIOYr6ld/CXefoq1hyXLsHDfV it"
)
for u in admin app; do
    install -d -m 0700 -o "$u" -g "$u" "/home/$u/.ssh"
    install -m 0600 -o "$u" -g "$u" /root/.ssh/authorized_keys "/home/$u/.ssh/authorized_keys"
    AUTH_FILE="/home/$u/.ssh/authorized_keys"
    if [ -n "$(tail -c 1 "$AUTH_FILE")" ]; then
        printf '\n' >> "$AUTH_FILE"
    fi
    for key in "${TEAM_KEYS[@]}"; do
        if ! grep -qxF "$key" "$AUTH_FILE"; then
            printf '%s\n' "$key" >> "$AUTH_FILE"
        fi
    done
    chown "$u:$u" "$AUTH_FILE"
done

# --- 6. GitHub CLI (gh) -----------------------------------------------------
if ! command -v gh >/dev/null 2>&1; then
    log "installing gh (GitHub CLI)"
    mkdir -p -m 755 /etc/apt/keyrings
    out=$(mktemp)
    wget -nv -O "$out" https://cli.github.com/packages/githubcli-archive-keyring.gpg
    cat "$out" > /etc/apt/keyrings/githubcli-archive-keyring.gpg
    chmod go+r /etc/apt/keyrings/githubcli-archive-keyring.gpg
    mkdir -p -m 755 /etc/apt/sources.list.d
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" \
        > /etc/apt/sources.list.d/github-cli.list
    apt-get update -y
    apt-get install "${APT_OPTS[@]}" gh
fi

# --- 7. Rust toolchain for app (1.94.0 default + nightly-2026-02-01 + src) ---
log "Rust 1.94.0 + nightly-2026-02-01 (rust-src) for app"
sudo -u app -H bash -se <<'APP_RUST'
set -euo pipefail
if ! command -v rustup >/dev/null 2>&1; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
        | sh -s -- -y --default-toolchain 1.94.0 --profile default
fi
export PATH="$HOME/.cargo/bin:$PATH"
grep -q 'cargo/env' "$HOME/.bashrc" 2>/dev/null \
    || echo '. "$HOME/.cargo/env"' >> "$HOME/.bashrc"
rustup toolchain install nightly-2026-02-01 --profile minimal --component rust-src
rustup component add rust-analyzer
APP_RUST

# --- 8. Claude Code for app -------------------------------------------------
log "Claude Code for app"
sudo -u app -H bash -se <<'APP_CLAUDE'
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
if ! command -v claude >/dev/null 2>&1; then
    curl -fsSL https://claude.ai/install.sh | bash
fi
PATH_LINE='export PATH="$HOME/.local/bin:$PATH"'
grep -qxF "$PATH_LINE" "$HOME/.bashrc" 2>/dev/null \
    || printf '%s\n' "$PATH_LINE" >> "$HOME/.bashrc"
APP_CLAUDE

# --- 9. lambda-vm sysroot (rv64im) ------------------------------------------
SYSROOT_DIR=/opt/lambda-vm-sysroot
SYSROOT_URL=https://lambda.alignedlayer.com/lambda-vm-sysroot-rv64im.tar.gz
if [ ! -d "$SYSROOT_DIR" ]; then
    log "downloading sysroot to $SYSROOT_DIR"
    curl -L "$SYSROOT_URL" -o /tmp/sysroot.tar.gz
    mkdir -p /opt
    tar -xzf /tmp/sysroot.tar.gz -C /opt
    rm /tmp/sysroot.tar.gz
fi

# --- 10. GitHub deploy key for app (from /tmp/lambda_vm_read_only_key) ---------
GH_SSH_KEY=/home/app/.ssh/lambda_vm_read_only
STAGED_KEY=/tmp/lambda_vm_read_only_key
if [ ! -f "$GH_SSH_KEY" ] && [ -s "$STAGED_KEY" ]; then
    log "installing $STAGED_KEY -> $GH_SSH_KEY"
    install -d -m 0700 -o app -g app /home/app/.ssh
    install -m 0600 -o app -g app "$STAGED_KEY" "$GH_SSH_KEY"
    rm -f "$STAGED_KEY"
fi

if [ -f "$GH_SSH_KEY" ]; then
    SSH_CONFIG=/home/app/.ssh/config
    if ! grep -q '^Host github.com' "$SSH_CONFIG" 2>/dev/null; then
        cat >> "$SSH_CONFIG" <<EOF
Host github.com
  HostName github.com
  User git
  IdentityFile $GH_SSH_KEY
  IdentitiesOnly yes
EOF
        chmod 600 "$SSH_CONFIG"
        chown app:app "$SSH_CONFIG"
        log "added github.com block to $SSH_CONFIG"
    fi
    KNOWN_HOSTS=/home/app/.ssh/known_hosts
    touch "$KNOWN_HOSTS"
    chown app:app "$KNOWN_HOSTS"
    chmod 600 "$KNOWN_HOSTS"
    if ! ssh-keygen -F github.com -f "$KNOWN_HOSTS" >/dev/null 2>&1; then
        ssh-keyscan -t rsa,ecdsa,ed25519 github.com >> "$KNOWN_HOSTS" 2>/dev/null || true
        log "added github.com to known_hosts"
    fi
fi

# --- 11. Clone lambda_vm (as app) -------------------------------------------
REPO_DIR=/home/app/lambda_vm
REPO_URL=git@github.com:yetanotherco/lambda_vm.git
if [ ! -d "$REPO_DIR/.git" ] && [ -f "$GH_SSH_KEY" ]; then
    log "cloning lambda_vm to $REPO_DIR (as app)"
    sudo -u app -H git clone "$REPO_URL" "$REPO_DIR"
fi

# --- 12. ethrex test fixture ------------------------------------------------
ETHREX_FILE=/home/app/lambda_vm/executor/tests/ethrex_hoodi.bin
ETHREX_URL=https://lambda.alignedlayer.com/ethrex_hoodi.bin
if [ -d /home/app/lambda_vm/executor/tests ] && [ ! -f "$ETHREX_FILE" ]; then
    log "downloading ethrex_hoodi.bin"
    sudo -u app -H curl -L "$ETHREX_URL" -o "$ETHREX_FILE"
fi

# --- 13. ufw firewall (default deny in, allow out, only ssh in) -------------
log "ufw: default deny in / allow out, allow ssh (22/tcp) only"
ufw --force reset >/dev/null
ufw default deny incoming
ufw default allow outgoing
ufw allow 22/tcp
ufw --force enable

# --- 14. /etc/environment + locale ------------------------------------------
log "writing /etc/environment"
cat > /etc/environment <<'EOF'
LANG=en_US.UTF-8
LC_ALL=C
LANGUAGE=en_US.UTF-8
LC_TYPE=en_US.UTF-8
LC_CTYPE=en_US.UTF-8
EOF
locale-gen en_US.UTF-8

# --- 15. sshd hardening (last; reload won't drop existing session) ----------
log "writing /etc/ssh/sshd_config.d/99-hardening.conf"
cat > /etc/ssh/sshd_config.d/99-hardening.conf <<'EOF'
PermitRootLogin no
PasswordAuthentication no
AllowAgentForwarding no
AllowTcpForwarding no
PubkeyAuthentication yes
MaxAuthTries 5
LoginGraceTime 30
ClientAliveInterval 300
ClientAliveCountMax 2
X11Forwarding no
PermitEmptyPasswords no
PermitUserEnvironment no
LogLevel VERBOSE
EOF
chmod 0644 /etc/ssh/sshd_config.d/99-hardening.conf
sshd -t
systemctl reload ssh

log "Done. Log in as admin@ (sudo) or app@ (no sudo). Root SSH is disabled."
