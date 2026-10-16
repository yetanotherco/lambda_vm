#!/bin/bash
# Rent a Scaleway Elastic Metal server with HOURLY billing.
#
# Usage: infra/rent_baremetal.sh <server-name>
#
# Env var overrides (all optional):
#   SCW_ZONE              default: fr-par-2
#   SCW_TYPE              default: EM-I320E-NVME
#   PROVISION_FILE        default: <repo>/infra/provision.sh
#   READY_TIMEOUT         default: 1800 (seconds)
#   LLVM_VERSION          default: 21
#
# Requires: scw, jq, ssh.
# To delete the server when done: scw baremetal server delete <id> zone=<zone>

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BOLD='\033[1m'
NC='\033[0m'

SCW_ZONE="${SCW_ZONE:-fr-par-2}"
SCW_TYPE="${SCW_TYPE:-EM-I320E-NVME}"
SCW_OS_ID="${SCW_OS_ID:-83640d93-a0b8-45ad-9c9f-30cae48380a4}"  # Debian
SCW_PROJECT_ID="${SCW_PROJECT_ID:-946cfb34-d351-48c4-8566-127e7727e15f}"
PROVISION_FILE="${PROVISION_FILE:-$SCRIPT_DIR/provision.sh}"
READY_TIMEOUT="${READY_TIMEOUT:-1800}"
LLVM_VERSION="${LLVM_VERSION:-21}"

err() { echo -e "${RED}error:${NC} $*" >&2; }
info() { echo -e "${BOLD}$*${NC}"; }
ok() { echo -e "${GREEN}$*${NC}"; }

if [ $# -lt 1 ] || [ -z "${1:-}" ]; then
    err "missing <server-name>"
    echo "Usage: $0 <server-name>" >&2
    exit 2
fi
SERVER_NAME="$1"

if ! command -v scw >/dev/null 2>&1; then
    err "scw CLI not found on PATH. Install: https://github.com/scaleway/scaleway-cli"
    exit 1
fi
if ! command -v jq >/dev/null 2>&1; then
    err "jq not found on PATH."
    exit 1
fi
if [ ! -r "$PROVISION_FILE" ]; then
    err "provision script not found or unreadable: $PROVISION_FILE"
    exit 1
fi
if ! command -v ssh >/dev/null 2>&1; then
    err "ssh not found on PATH."
    exit 1
fi

EXPECTED_PROFILE="vm"
if [ -n "${SCW_PROFILE:-}" ]; then
    ACTIVE_PROFILE="$SCW_PROFILE"
elif [ -r "$HOME/.config/scw/config.yaml" ]; then
    ACTIVE_PROFILE=$(grep -E '^active_profile:' "$HOME/.config/scw/config.yaml" \
        | head -n1 | awk '{print $2}' | tr -d '"' | tr -d "'")
else
    ACTIVE_PROFILE=""
fi
if [ "$ACTIVE_PROFILE" != "$EXPECTED_PROFILE" ]; then
    err "active scw profile is '${ACTIVE_PROFILE:-<none>}', expected '$EXPECTED_PROFILE'"
    echo "  Switch with: export SCW_PROFILE=$EXPECTED_PROFILE" >&2
    echo "  Or set 'active_profile: $EXPECTED_PROFILE' in ~/.config/scw/config.yaml" >&2
    exit 1
fi
ok "Using scw profile: $EXPECTED_PROFILE"

info "Resolving hourly offer ID for type=$SCW_TYPE zone=$SCW_ZONE..."
OFFER_JSON=$(scw baremetal offer list \
    zone="$SCW_ZONE" \
    name="$SCW_TYPE" \
    subscription-period=hourly \
    -o json)
OFFER_ID=$(echo "$OFFER_JSON" | jq -r --arg n "$SCW_TYPE" '[.[] | select(.name == $n)] | .[0].id // empty')
if [ -z "$OFFER_ID" ]; then
    err "no hourly offer named '$SCW_TYPE' in $SCW_ZONE — refusing to create a non-hourly server"
    echo "$OFFER_JSON" >&2
    exit 1
fi
ok "Hourly offer ID: $OFFER_ID"

info "Enumerating SSH keys in project $SCW_PROJECT_ID..."
SSH_KEYS_JSON=$(scw iam ssh-key list project-id="$SCW_PROJECT_ID" -o json)
SSH_KEY_IDS=()
while IFS= read -r line; do
    [ -n "$line" ] && SSH_KEY_IDS+=("$line")
done < <(echo "$SSH_KEYS_JSON" | jq -r '.[].id')
if [ ${#SSH_KEY_IDS[@]} -eq 0 ]; then
    err "no SSH keys found in project $SCW_PROJECT_ID — register one first ('scw iam ssh-key create')"
    exit 1
fi
ok "Found ${#SSH_KEY_IDS[@]} SSH key(s) to install"

SSH_KEY_ARGS=()
for i in "${!SSH_KEY_IDS[@]}"; do
    SSH_KEY_ARGS+=("common-configuration.install.ssh-key-ids.${i}=${SSH_KEY_IDS[$i]}")
done

info "Creating server name=$SERVER_NAME via batch-create..."
CREATE_JSON=$(scw baremetal server batch-create \
    zone="$SCW_ZONE" \
    common-configuration.offer-id="$OFFER_ID" \
    common-configuration.project-id="$SCW_PROJECT_ID" \
    common-configuration.name="$SERVER_NAME" \
    common-configuration.install.os-id="$SCW_OS_ID" \
    common-configuration.install.hostname="$SERVER_NAME" \
    "${SSH_KEY_ARGS[@]}" \
    servers.0.hostname="$SERVER_NAME" \
    -o json)

SERVER_ID=$(echo "$CREATE_JSON" | jq -r '.servers[0].id // empty')
if [ -z "$SERVER_ID" ]; then
    err "server create did not return an id. Raw response:"
    echo "$CREATE_JSON" >&2
    exit 1
fi
ok "Created server id=$SERVER_ID. Waiting up to ${READY_TIMEOUT}s for status=ready AND install.status=completed..."

DEADLINE=$(( $(date +%s) + READY_TIMEOUT ))
LAST_STATE=""
while [ "$(date +%s)" -lt "$DEADLINE" ]; do
    GET_JSON=$(scw baremetal server get "$SERVER_ID" zone="$SCW_ZONE" -o json)
    STATUS=$(echo "$GET_JSON" | jq -r '.status // empty')
    INSTALL_STATUS=$(echo "$GET_JSON" | jq -r '.install.status // "pending"')
    STATE="$STATUS / install=$INSTALL_STATUS"
    if [ "$STATE" != "$LAST_STATE" ]; then
        echo -e "  ${YELLOW}$STATE${NC}"
        LAST_STATE="$STATE"
    else
        echo -n "."
    fi
    if [ "$STATUS" = "ready" ] && [ "$INSTALL_STATUS" = "completed" ]; then
        echo
        break
    fi
    if [ "$STATUS" = "error" ] || [ "$STATUS" = "locked" ]; then
        echo
        err "server entered terminal status: $STATUS"
        echo "$GET_JSON" >&2
        exit 1
    fi
    if [ "$INSTALL_STATUS" = "error" ]; then
        echo
        err "install entered terminal status: $INSTALL_STATUS"
        echo "$GET_JSON" >&2
        exit 1
    fi
    sleep 15
done

GET_JSON=$(scw baremetal server get "$SERVER_ID" zone="$SCW_ZONE" -o json)
FINAL_STATUS=$(echo "$GET_JSON" | jq -r '.status // empty')
FINAL_INSTALL=$(echo "$GET_JSON" | jq -r '.install.status // "pending"')
if [ "$FINAL_STATUS" != "ready" ] || [ "$FINAL_INSTALL" != "completed" ]; then
    err "timed out after ${READY_TIMEOUT}s — last status=$FINAL_STATUS install=$FINAL_INSTALL"
    exit 1
fi

PUBLIC_IP=$(echo "$GET_JSON" | jq -r '.ips[] | select(.version == "IPv4") | .address' | head -n1)

echo
ok "Server ready."
echo "  id:     $SERVER_ID"
echo "  name:   $SERVER_NAME"
echo "  zone:   $SCW_ZONE"
echo "  type:   $SCW_TYPE (hourly offer $OFFER_ID)"
echo "  ip:     $PUBLIC_IP"
echo

info "Wiping any stale known_hosts entry for $PUBLIC_IP (Scaleway recycles IPs)..."
ssh-keygen -R "$PUBLIC_IP" >/dev/null 2>&1 || true

info "Handing off to provision_server.sh (Ctrl+C to skip and provision later)..."
PROVISION_FILE="$PROVISION_FILE" SSH_USER=root LLVM_VERSION="$LLVM_VERSION" "$SCRIPT_DIR/provision_server.sh" "$PUBLIC_IP"

echo
echo "To delete the server:"
echo "  scw baremetal server delete $SERVER_ID zone=$SCW_ZONE"
echo
echo "To re-provision this server later (sshd is hardened, so admin + sudo):"
echo "  SSH_USER=admin infra/provision_server.sh $PUBLIC_IP"
