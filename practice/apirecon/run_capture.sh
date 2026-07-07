#!/usr/bin/env bash
# =============================================================================
#  run_capture.sh — Peloton API traffic capture via mitmproxy
#  Run from the directory containing traffic_logger.py
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ADDON="$SCRIPT_DIR/traffic_logger.py"
LOG_FILE="$SCRIPT_DIR/api_log.jsonl"
PROXY_PORT=8080

# ── Colours ──────────────────────────────────────────────────────────────────
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

# ── 1. Locate mitmproxy (Homebrew install) ───────────────────────────────────
MITMDUMP="/opt/homebrew/bin/mitmdump"
if [[ ! -x "$MITMDUMP" ]]; then
    MITMDUMP="$(command -v mitmdump 2>/dev/null)" \
        || error "mitmdump not found. Expected /opt/homebrew/bin/mitmdump"
fi
info "mitmproxy: $($MITMDUMP --version 2>&1 | head -1)"

# ── 2. Generate mitmproxy CA cert (first run) ─────────────────────────────────
MITM_CERT="$HOME/.mitmproxy/mitmproxy-ca-cert.pem"
if [[ ! -f "$MITM_CERT" ]]; then
    info "Generating mitmproxy CA certificate (first-run)..."
    $MITMDUMP --version &>/dev/null
    sleep 1
fi

# ── 3. macOS: install CA cert into system keychain ───────────────────────────
CERT_INSTALLED=$(security find-certificate -c "mitmproxy" \
    /Library/Keychains/System.keychain 2>/dev/null | wc -l | tr -d ' ')

if [[ "$CERT_INSTALLED" -eq 0 ]]; then
    info "Installing mitmproxy CA cert into macOS System Keychain..."
    echo ""
    echo "  ┌────────────────────────────────────────────────────────────┐"
    echo "  │  macOS will prompt for your admin password.                │"
    echo "  │  This is needed to trust HTTPS interception.               │"
    echo "  └────────────────────────────────────────────────────────────┘"
    echo ""
    sudo security add-trusted-cert -d -r trustRoot \
        -k /Library/Keychains/System.keychain "$MITM_CERT"
    info "CA cert installed and trusted."
else
    info "mitmproxy CA cert already trusted — skipping."
fi

# ── 4. Enable macOS system proxy ──────────────────────────────────────────────
NETWORK_SERVICE=$(networksetup -listnetworkserviceorder \
    | grep -A1 'Hardware Port' \
    | grep -v 'Hardware Port\|--' \
    | head -1 \
    | sed 's/^(.*) //' \
    | awk -F') ' '{print $2}')

if [[ -z "$NETWORK_SERVICE" ]]; then
    NETWORK_SERVICE="Wi-Fi"
fi

info "Configuring system proxy on '$NETWORK_SERVICE' → 127.0.0.1:$PROXY_PORT"
networksetup -setwebproxy      "$NETWORK_SERVICE" 127.0.0.1 $PROXY_PORT
networksetup -setsecurewebproxy "$NETWORK_SERVICE" 127.0.0.1 $PROXY_PORT
networksetup -setwebproxystate      "$NETWORK_SERVICE" on
networksetup -setsecurewebproxystate "$NETWORK_SERVICE" on

# ── 5. Trap to restore proxy on exit ─────────────────────────────────────────
cleanup() {
    echo ""
    info "Restoring system proxy settings..."
    networksetup -setwebproxystate       "$NETWORK_SERVICE" off
    networksetup -setsecurewebproxystate  "$NETWORK_SERVICE" off
    info "Proxy disabled. Log saved to: $LOG_FILE"
}
trap cleanup EXIT INT TERM

# ── 6. Launch capture ─────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "  Listening on port ${YELLOW}$PROXY_PORT${NC}"
echo -e "  Logging to       ${YELLOW}$LOG_FILE${NC}"
echo -e "  Press ${RED}Ctrl-C${NC} to stop and restore proxy settings."
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# To filter to Peloton traffic only, add:
#   --set filter_host=peloton
# To disable body logging:
#   --set log_bodies=false

$MITMDUMP \
    -s "$ADDON" \
    --listen-port $PROXY_PORT \
    --set log_file="$LOG_FILE" \
    --set filter_host="" \
    --ssl-insecure
