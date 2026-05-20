#!/bin/bash
# Provision a freshly rented Scaleway Elastic Metal Debian server.
# Invoked remotely from infra/rent_baremetal.sh as:
#   ssh root@<ip> 'bash -s' < infra/provision.sh
#
# Idempotent where reasonable so a re-run on the same box does not break things.

set -euo pipefail

export DEBIAN_FRONTEND=noninteractive
APT_OPTS=(-y -o Dpkg::Options::=--force-confdef -o Dpkg::Options::=--force-confold)

echo "==> apt update + upgrade"
apt-get update -y
apt-get upgrade "${APT_OPTS[@]}"

echo "==> Installing packages"
apt-get install "${APT_OPTS[@]}" \
    ca-certificates \
    curl \
    wget \
    gnupg \
    vim \
    git \
    zip \
    unzip \
    openssl \
    libssl-dev \
    build-essential \
    rsyslog \
    htop \
    rsync \
    pkg-config \
    locales \
    ufw

echo "==> Creating users: admin (sudo), app (no sudo)"
for u in admin app; do
    if ! id "$u" >/dev/null 2>&1; then
        useradd -m -s /bin/bash "$u"
    fi
done

echo 'admin ALL=(ALL) NOPASSWD:ALL' > /etc/sudoers.d/90-admin
chmod 0440 /etc/sudoers.d/90-admin

echo "==> Propagating root's authorized_keys to admin and app"
if [ ! -s /root/.ssh/authorized_keys ]; then
    echo "ERROR: /root/.ssh/authorized_keys missing or empty — refusing to harden sshd, you would lose access." >&2
    exit 1
fi
for u in admin app; do
    install -d -m 0700 -o "$u" -g "$u" "/home/$u/.ssh"
    install -m 0600 -o "$u" -g "$u" /root/.ssh/authorized_keys "/home/$u/.ssh/authorized_keys"
done

echo "==> Writing /etc/environment"
cat > /etc/environment <<'EOF'
LANG=en_US.UTF-8
LC_ALL=C
LANGUAGE=en_US.UTF-8
LC_TYPE=en_US.UTF-8
LC_CTYPE=en_US.UTF-8
EOF
locale-gen en_US.UTF-8

echo "==> Writing /etc/ssh/sshd_config.d/99-hardening.conf"
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

echo "==> Validating sshd config and reloading"
sshd -t
systemctl reload ssh

echo "==> Done. From now on log in as admin@ (sudo) or app@ (no sudo). Root SSH is disabled."
