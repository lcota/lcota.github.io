#!/usr/bin/env bash
# =============================================================================
#  init.sh — One-time setup: substitute placeholders for your target app.
#
#  Usage:
#    bash init.sh --app-name "Peloton" --app-slug "peloton" \
#                 --api-host "api.onepeloton.com" \
#                 [--user-agent "Peloton/3.x CFNetwork/..."]
#
#  What it does:
#    1. Validates inputs
#    2. Substitutes sentinel strings across all text files
#    3. Renames src/target_app/ → src/{slug}/
#    4. Renames the Bruno environment file
#    5. Prints a checklist of remaining manual TODOs
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; CYAN='\033[0;36m'; NC='\033[0m'
info()    { echo -e "${GREEN}[init]${NC}  $*"; }
warn()    { echo -e "${YELLOW}[warn]${NC}  $*"; }
error()   { echo -e "${RED}[error]${NC} $*" >&2; exit 1; }
step()    { echo -e "\n${CYAN}▶ $*${NC}"; }

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
APP_NAME=""
APP_SLUG=""
API_HOST=""
USER_AGENT=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --app-name)   APP_NAME="$2"; shift 2 ;;
        --app-slug)   APP_SLUG="$2"; shift 2 ;;
        --api-host)   API_HOST="$2"; shift 2 ;;
        --user-agent) USER_AGENT="$2"; shift 2 ;;
        *) error "Unknown argument: $1" ;;
    esac
done

# ---------------------------------------------------------------------------
# Validate inputs
# ---------------------------------------------------------------------------
[[ -z "$APP_NAME" ]] && error "--app-name is required (e.g. \"Peloton\")"
[[ -z "$APP_SLUG" ]] && error "--app-slug is required (e.g. \"peloton\")"
[[ -z "$API_HOST" ]] && error "--api-host is required (e.g. \"api.onepeloton.com\")"

# Slug: lowercase letters, digits, hyphens only
if ! [[ "$APP_SLUG" =~ ^[a-z0-9-]+$ ]]; then
    error "--app-slug must be lowercase alphanumeric/hyphens only (got: '$APP_SLUG')"
fi

# Host: basic hostname check (no protocol, no path)
if [[ "$API_HOST" =~ ^https?:// ]] || [[ "$API_HOST" =~ / ]]; then
    error "--api-host should be a bare hostname, not a URL (got: '$API_HOST')"
fi

# Default user agent if not provided
if [[ -z "$USER_AGENT" ]]; then
    USER_AGENT="${APP_NAME}/1.0"
    warn "--user-agent not provided, defaulting to '${USER_AGENT}'"
    warn "Update auth.py and client.py with the real User-Agent from captured traffic."
fi

# Guard: refuse to run if slug matches the sentinel (already initialized?)
if [[ "$APP_SLUG" == "target_app" ]]; then
    error "App slug cannot be 'target_app' — that is the placeholder value."
fi

echo ""
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "  App name   : ${YELLOW}${APP_NAME}${NC}"
echo -e "  App slug   : ${YELLOW}${APP_SLUG}${NC}"
echo -e "  API host   : ${YELLOW}${API_HOST}${NC}"
echo -e "  User agent : ${YELLOW}${USER_AGENT}${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# Confirm before proceeding
read -r -p "Proceed with substitution? [y/N] " confirm
confirm_lower=$(echo "$confirm" | tr '[:upper:]' '[:lower:]')
[[ "$confirm_lower" == "y" ]] || { echo "Aborted."; exit 0; }

# ---------------------------------------------------------------------------
# Helper: in-place sed, macOS-compatible (BSD sed requires '' after -i)
# ---------------------------------------------------------------------------
sub() {
    local from="$1" to="$2" file="$3"
    sed -i '' "s|${from}|${to}|g" "$file"
}

# Derive SCREAMING_SNAKE env prefix from slug (e.g. "my-app" → "MY_APP_")
ENV_PREFIX=$(echo "$APP_SLUG" | tr 'a-z-' 'A-Z_')_

# ---------------------------------------------------------------------------
# Step 1: Substitute sentinels across all text files
# ---------------------------------------------------------------------------
step "Substituting sentinel strings in text files..."

FILE_COUNT=0
while IFS= read -r -d '' f; do
    FILE_COUNT=$((FILE_COUNT + 1))
    # Order matters: substitute the most specific patterns first
    sub "TARGET_APP_NAME"       "$APP_NAME"    "$f"
    sub "TARGET_APP_SLUG"       "$APP_SLUG"    "$f"
    sub "TARGET_APP_USER_AGENT" "$USER_AGENT"  "$f"
    sub "TARGET_APP_"           "$ENV_PREFIX"  "$f"
    sub "TARGET_API_HOST"       "$API_HOST"    "$f"
    sub "target_app"            "$APP_SLUG"    "$f"
done < <(
    find "$SCRIPT_DIR" \
        -not -path '*/.git/*' \
        -not -path '*/\.venv/*' \
        -not -path '*/__pycache__/*' \
        -not -name '*.pyc' \
        -not -name 'init.sh' \
        -type f \
        \( -name "*.py" -o -name "*.sh" -o -name "*.toml" \
           -o -name "*.md" -o -name "*.bru" -o -name "*.json" \
           -o -name "*.env*" -o -name ".gitignore" \) \
        -print0
)

info "Sentinel substitution complete (${FILE_COUNT} files scanned)."

# ---------------------------------------------------------------------------
# Step 2: Rename src/target_app/ → src/{slug}/
# ---------------------------------------------------------------------------
step "Renaming Python package directory..."

SRC_OLD="$SCRIPT_DIR/src/target_app"
SRC_NEW="$SCRIPT_DIR/src/${APP_SLUG}"

if [[ -d "$SRC_OLD" ]]; then
    mv "$SRC_OLD" "$SRC_NEW"
    info "Renamed: src/target_app/ → src/${APP_SLUG}/"
elif [[ -d "$SRC_NEW" ]]; then
    info "src/${APP_SLUG}/ already exists — skipping rename."
else
    warn "src/target_app/ not found — skipping rename."
fi

# ---------------------------------------------------------------------------
# Step 3: Rename Bruno environment file
# ---------------------------------------------------------------------------
step "Renaming Bruno environment file..."

BRU_OLD="$SCRIPT_DIR/bruno_collection/environments/target-app.bru"
BRU_NEW="$SCRIPT_DIR/bruno_collection/environments/${APP_SLUG}.bru"

if [[ -f "$BRU_OLD" ]]; then
    mv "$BRU_OLD" "$BRU_NEW"
    info "Renamed: environments/target-app.bru → environments/${APP_SLUG}.bru"
elif [[ -f "$BRU_NEW" ]]; then
    info "${APP_SLUG}.bru already exists — skipping rename."
fi

# ---------------------------------------------------------------------------
# Step 4: Copy .env.example → .env (if not already present)
# ---------------------------------------------------------------------------
step "Setting up .env file..."

ENV_EXAMPLE="$SCRIPT_DIR/.env.example"
ENV_FILE="$SCRIPT_DIR/.env"

if [[ ! -f "$ENV_FILE" ]] && [[ -f "$ENV_EXAMPLE" ]]; then
    cp "$ENV_EXAMPLE" "$ENV_FILE"
    info "Created .env from .env.example — fill in your credentials."
elif [[ -f "$ENV_FILE" ]]; then
    info ".env already exists — skipping."
fi

# ---------------------------------------------------------------------------
# Done — print remaining manual TODOs
# ---------------------------------------------------------------------------
echo ""
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "  ${GREEN}✓ Initialisation complete!${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo -e "  ${YELLOW}Remaining manual TODOs (see CUSTOMIZATION.md for details):${NC}"
echo ""
echo -e "  1. Run ${CYAN}bash run_capture.sh${NC} to capture traffic from ${APP_NAME}."
echo -e "     Interact with the app, then Ctrl-C to stop."
echo ""
echo -e "  2. Inspect auth traffic:"
echo -e "     ${CYAN}python3 analyze_log.py --method POST --search login${NC}"
echo -e "     Then fill in ${CYAN}src/${APP_SLUG}/auth.py${NC} (_login_and_capture)."
echo ""
echo -e "  3. Define API models in ${CYAN}src/${APP_SLUG}/models.py${NC}"
echo -e "     based on observed response shapes."
echo ""
echo -e "  4. Add typed endpoint methods to ${CYAN}src/${APP_SLUG}/client.py${NC}."
echo ""
echo -e "  5. Register task types in ${CYAN}src/${APP_SLUG}/tasks/definitions.py${NC}."
echo ""
echo -e "  6. Fill in your credentials in ${CYAN}.env${NC}."
echo ""
