#!/bin/bash
# Run infra/provision.sh on a remote Scaleway baremetal server over SSH.
# Safe to run standalone after rent_baremetal.sh, or to re-provision an
# existing server.
#
# Usage: infra/provision_server.sh <ip>
#
# Env var overrides (all optional):
#   SSH_USER             default: root
#                        First-run servers accept root SSH; once provision.sh
#                        has hardened sshd, re-run as: SSH_USER=admin ...
#   PROVISION_FILE       default: <script-dir>/provision.sh
#   LLVM_VERSION         default: 21
#
# SSH wait is indefinite — Ctrl+C to abort.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BOLD='\033[1m'
NC='\033[0m'

SSH_USER="${SSH_USER:-root}"
PROVISION_FILE="${PROVISION_FILE:-$SCRIPT_DIR/provision.sh}"
LLVM_VERSION="${LLVM_VERSION:-21}"

err() { echo -e "${RED}error:${NC} $*" >&2; }
info() { echo -e "${BOLD}$*${NC}"; }
ok() { echo -e "${GREEN}$*${NC}"; }

if [ $# -lt 1 ] || [ -z "${1:-}" ]; then
    err "missing <ip>"
    echo "Usage: $0 <ip>" >&2
    exit 2
fi
IP="$1"

if [ ! -r "$PROVISION_FILE" ]; then
    err "provision script not found or unreadable: $PROVISION_FILE"
    exit 1
fi
if ! command -v ssh >/dev/null 2>&1; then
    err "ssh not found on PATH."
    exit 1
fi
case "$LLVM_VERSION" in
    ''|*[!0-9]*)
        err "LLVM_VERSION must be a numeric LLVM major version, got '$LLVM_VERSION'"
        exit 2
        ;;
esac

SSH_OPTS=(-o StrictHostKeyChecking=accept-new -o ConnectTimeout=10 -o BatchMode=yes)

info "Waiting for sshd on $SSH_USER@$IP (indefinite — Ctrl+C to abort)..."
attempt=1
while ! ssh "${SSH_OPTS[@]}" "$SSH_USER@$IP" true 2>/dev/null; do
    if [ $((attempt % 6)) -eq 0 ]; then
        echo -e "  ${YELLOW}still waiting (attempt $attempt, ~$((attempt * 10))s elapsed)${NC}"
    fi
    attempt=$((attempt + 1))
    sleep 10
done
ok "sshd reachable on $SSH_USER@$IP (attempt $attempt)"

if [ "$SSH_USER" = "root" ]; then
    REMOTE_CMD="env LLVM_VERSION=$LLVM_VERSION bash -s"
else
    REMOTE_CMD="sudo env LLVM_VERSION=$LLVM_VERSION bash -s"
fi

info "Running $PROVISION_FILE on $SSH_USER@$IP..."
ssh "${SSH_OPTS[@]}" "$SSH_USER@$IP" "$REMOTE_CMD" < "$PROVISION_FILE"

echo
ok "Provisioning complete."
echo "  ssh admin@$IP    # sudo"
echo "  ssh app@$IP      # no sudo"
