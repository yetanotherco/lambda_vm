#!/bin/bash
# Provision a freshly rented Scaleway Elastic Metal Debian server.
# Invoked remotely from infra/provision_server.sh as:
#   ssh root@<ip> bash -s < infra/provision.sh
#
# Idempotent — safe to re-run.

set -euo pipefail

log() { printf '\n=== %s ===\n' "$*"; }

# --- 1. apt update + upgrade -------------------------------------------------
log "apt update + upgrade"
export DEBIAN_FRONTEND=noninteractive
APT_OPTS=(-y -o Dpkg::Options::=--force-confdef -o Dpkg::Options::=--force-confold)
LLVM_VERSION="${LLVM_VERSION:-21}"
LLVM_GPG_FINGERPRINT="6084F3CF814B57C1CF12EFD515CF4D18AF4F7421"

# Scaleway baremetal Debian ships grub-cloud-amd64; its postinst (fired as a
# trigger by initramfs-tools / shim-signed / kernel upgrades) runs grub-install
# against an ext2 root and fails ("will not proceed with blocklists"). The
# package isn't load-bearing on UEFI baremetal — purge it before any upgrade.
apt-get purge -y grub-cloud-amd64 2>/dev/null || true

apt-get update -y
apt-get upgrade "${APT_OPTS[@]}"

# --- 2. apt packages ---------------------------------------------------------
log "apt install base packages + xz-utils"
apt-get install "${APT_OPTS[@]}" \
    ca-certificates curl wget gnupg vim git zip unzip openssl libssl-dev jq \
    build-essential rsyslog htop rsync pkg-config locales ufw xz-utils

# Debian's default clang can lag behind RISC-V ISA attribute syntax emitted by
# the checked-in assembly fixtures. Use a pinned apt.llvm.org toolchain.
log "LLVM $LLVM_VERSION toolchain from apt.llvm.org"
. /etc/os-release
LLVM_CODENAME="${VERSION_CODENAME:-}"
if [ -z "$LLVM_CODENAME" ]; then
    echo "ERROR: could not determine Debian/Ubuntu codename from /etc/os-release" >&2
    exit 1
fi
install -d -m 0755 /etc/apt/keyrings
wget -qO /etc/apt/keyrings/apt.llvm.org.asc https://apt.llvm.org/llvm-snapshot.gpg.key
if ! LLVM_ACTUAL_FINGERPRINT="$(gpg --show-keys --with-colons /etc/apt/keyrings/apt.llvm.org.asc 2>/dev/null \
    | awk -F: '/^fpr:/ { print $10; exit }')"; then
    echo "ERROR: could not read apt.llvm.org GPG key fingerprint" >&2
    exit 1
fi
if [ "$LLVM_ACTUAL_FINGERPRINT" != "$LLVM_GPG_FINGERPRINT" ]; then
    echo "ERROR: apt.llvm.org GPG key fingerprint mismatch (got $LLVM_ACTUAL_FINGERPRINT, expected $LLVM_GPG_FINGERPRINT)" >&2
    exit 1
fi
chmod 0644 /etc/apt/keyrings/apt.llvm.org.asc
cat > /etc/apt/sources.list.d/apt.llvm.org.list <<EOF
deb [signed-by=/etc/apt/keyrings/apt.llvm.org.asc] http://apt.llvm.org/$LLVM_CODENAME/ llvm-toolchain-$LLVM_CODENAME-$LLVM_VERSION main
EOF
apt-get update -y
apt-get install "${APT_OPTS[@]}" "clang-$LLVM_VERSION" "lld-$LLVM_VERSION" "llvm-$LLVM_VERSION"
for tool in clang clang++ lld ld.lld llvm-ar llvm-ranlib; do
    versioned="/usr/bin/$tool-$LLVM_VERSION"
    if [ ! -x "$versioned" ]; then
        echo "ERROR: expected $versioned after installing LLVM $LLVM_VERSION" >&2
        exit 1
    fi
    ln -sf "$versioned" "/usr/local/bin/$tool"
done
clang --version | head -n 1

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
    "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIBzniAUYGJXguBjfz2+uGUUC7XLVmk58FhCsEBMx2r5k mauro@mail.com"
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

# --- 5. GitHub CLI (gh) -----------------------------------------------------
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

# --- 6. Rust toolchain for app (1.94.0 default + nightly-2026-02-01 + src) ---
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

# --- 7. Claude Code for app -------------------------------------------------
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

# --- 8. lambda-vm sysroot (rv64im) ------------------------------------------
# Guard on include/stdlib.h and re-extract from scratch so a partial/interrupted extract
# self-heals on re-run; a bare `[ ! -d ]` guard left a headerless sysroot that broke c-kzg.
SYSROOT_DIR=/opt/lambda-vm-sysroot
SYSROOT_URL=https://lambda.alignedlayer.com/lambda-vm-sysroot-rv64im.tar.gz
if [ -f "$SYSROOT_DIR/include/stdlib.h" ] && [ -d "$SYSROOT_DIR/lib" ]; then
    log "sysroot already present at $SYSROOT_DIR"
else
    log "provisioning sysroot at $SYSROOT_DIR"
    curl -fL --proto '=https' "$SYSROOT_URL" -o /tmp/sysroot.tar.gz \
        || { rm -f /tmp/sysroot.tar.gz; exit 1; }
    rm -rf "$SYSROOT_DIR"
    mkdir -p /opt
    tar -xzf /tmp/sysroot.tar.gz -C /opt --no-same-owner \
        || { rm -rf "$SYSROOT_DIR"; rm -f /tmp/sysroot.tar.gz; exit 1; }
    rm -f /tmp/sysroot.tar.gz
fi

# --- 9. Clone lambda_vm (as app, public repo over HTTPS) ---------------------
REPO_DIR=/home/app/lambda_vm
REPO_URL=https://github.com/yetanotherco/lambda_vm.git
if [ ! -d "$REPO_DIR/.git" ]; then
    log "cloning lambda_vm to $REPO_DIR (as app)"
    sudo -u app -H git clone "$REPO_URL" "$REPO_DIR"
fi

# --- 10. ufw firewall (default deny in, allow out, only ssh in) -------------
log "ufw: default deny in / allow out, allow ssh (22/tcp) only"
ufw --force reset >/dev/null
ufw default deny incoming
ufw default allow outgoing
ufw allow 22/tcp
ufw --force enable

# --- 11. /etc/environment + locale ------------------------------------------
log "writing /etc/environment"
cat > /etc/environment <<'EOF'
LANG=en_US.UTF-8
LC_ALL=C
LANGUAGE=en_US.UTF-8
LC_TYPE=en_US.UTF-8
LC_CTYPE=en_US.UTF-8
EOF
locale-gen en_US.UTF-8

# --- 12. sshd hardening (last; reload won't drop existing session) ----------
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
