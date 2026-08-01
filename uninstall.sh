#!/usr/bin/env bash
# echowrite uninstaller
#
# Removes everything ./install.sh created (app files only):
#   - systemd user service (stopped + disabled + file removed)
#   - ~/.local/bin/echowrite launcher
#   - ~/.local/share/echowrite/  (venv + data dir)
#   - /tmp/voice2text_status runtime status file
#   - /etc/udev/rules.d/99-echowrite-input.rules   (sudo)
#   - /etc/modules-load.d/uinput.conf              (sudo)
#
# Intentionally NOT removed (shared / can affect other software):
#   - your `input` group membership
#   - system packages (portaudio, xclip, ...)
#   - your git checkout of this repo
# The script prints the exact commands to remove these so you can opt in.
#
# Usage:  ./uninstall.sh [-y|--yes]
# Re-running it is safe — it reports "not present" for anything already gone.

set -euo pipefail

# ---------- Configuration (mirrors install.sh verbatim — do not drift) ----------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
ECHOWRITE_REPO_DIR="${ECHOWRITE_REPO_DIR:-$SCRIPT_DIR}"
DATA_DIR="${XDG_DATA_HOME:-$HOME/.local/share}/echowrite"
VENV_DIR="$DATA_DIR/venv"
BIN_DIR="${XDG_BIN_HOME:-$HOME/.local/bin}"
SYSTEMD_USER_DIR="$HOME/.config/systemd/user"
SERVICE_NAME="echowrite"
SERVICE_FILE="$SYSTEMD_USER_DIR/$SERVICE_NAME.service"
LAUNCHER="$BIN_DIR/$SERVICE_NAME"
UDEV_RULE="/etc/udev/rules.d/99-echowrite-input.rules"
UINPUT_CONF="/etc/modules-load.d/uinput.conf"
# Runtime status file written by the app (configurable via I3_STATUS_FILE)
STATUS_FILE="${I3_STATUS_FILE:-/tmp/voice2text_status}"
ASSUME_YES=0

# ---------- Output helpers (mirrors install.sh) ----------
if [[ -t 1 ]]; then
    BOLD=$'\033[1m'; DIM=$'\033[2m'
    RED=$'\033[31m'; GREEN=$'\033[32m'; YELLOW=$'\033[33m'; BLUE=$'\033[34m'
    RESET=$'\033[0m'
else
    BOLD=""; DIM=""; RED=""; GREEN=""; YELLOW=""; BLUE=""; RESET=""
fi
info()    { printf "%b\n" "${BLUE}==>${RESET} ${BOLD}$*${RESET}"; }
success() { printf "%b\n" "${GREEN}✓${RESET} $*"; }
warn()    { printf "%b\n" "${YELLOW}!${RESET} $*"; }
error()   { printf "%b\n" "${RED}✗${RESET} $*" >&2; }
note()    { printf "%b\n" "    ${DIM}$*${RESET}"; }
heading() { printf "\n%b\n" "${BOLD}$*${RESET}"; }

# ---------- Pre-flight ----------
check_not_root() {
    if [[ $EUID -eq 0 ]]; then
        error "Don't run uninstall.sh as root. It calls sudo where needed."
        exit 1
    fi
}

detect_package_manager() {
    if command -v apt-get >/dev/null 2>&1; then echo apt
    elif command -v dnf >/dev/null 2>&1; then echo dnf
    elif command -v pacman >/dev/null 2>&1; then echo pacman
    else echo unsupported
    fi
}

usage() {
    cat <<EOF
echowrite uninstaller — removes what ./install.sh created.

Usage: ./uninstall.sh [-y|--yes] [-h|--help]

Options:
  -y, --yes     Skip the confirmation prompt (for automation)
  -h, --help    Show this help

Everything removed is re-creatable by re-running ./install.sh.
EOF
}

parse_args() {
    for arg in "$@"; do
        case "$arg" in
            -y|--yes) ASSUME_YES=1 ;;
            -h|--help) usage; exit 0 ;;
            *) error "Unknown argument: $arg"; usage; exit 1 ;;
        esac
    done
}

# ---------- Plan + confirm ----------
print_plan() {
    heading "This will remove:"
    note "  $SERVICE_FILE        (systemd user service; stopped + disabled)"
    note "  $LAUNCHER            (PATH launcher)"
    note "  $DATA_DIR            (venv + data dir)"
    note "  $STATUS_FILE         (runtime status file)"
    note "  $UDEV_RULE  (sudo)   (udev rules)"
    note "  $UINPUT_CONF (sudo)  (uinput module auto-load)"

    heading "NOT removed (you can do these manually — see end of run):"
    note "  - your \`input\` group membership"
    note "  - system packages (portaudio, xclip, ...)"
    note "  - your git checkout at $ECHOWRITE_REPO_DIR"
}

confirm() {
    if [[ $ASSUME_YES -eq 1 ]]; then
        return 0
    fi
    echo
    local reply
    read -r -p "$(printf '%b' "${BOLD}Proceed with uninstall? [y/N]${RESET} ")" reply
    case "${reply,,}" in
        y|yes) return 0 ;;
        *) echo "Aborted."; exit 0 ;;
    esac
}

# ---------- Removal steps ----------
remove_service() {
    heading "systemd user service"
    if [[ -f "$SERVICE_FILE" ]]; then
        # Stop + disable if loaded/enabled. Harmless if the unit isn't loaded.
        systemctl --user disable --now "$SERVICE_NAME.service" >/dev/null 2>&1 || true
        rm -f "$SERVICE_FILE"
        systemctl --user daemon-reload >/dev/null 2>&1 || true
        success "Removed $SERVICE_FILE"
    else
        note "Not present: $SERVICE_FILE"
    fi
}

remove_launcher() {
    heading "Launcher"
    if [[ -e "$LAUNCHER" ]] || [[ -L "$LAUNCHER" ]]; then
        rm -f "$LAUNCHER"
        success "Removed $LAUNCHER"
    else
        note "Not present: $LAUNCHER"
    fi
}

remove_data_dir() {
    heading "Data directory + venv"
    if [[ -d "$DATA_DIR" ]]; then
        rm -rf "$DATA_DIR"
        success "Removed $DATA_DIR"
    else
        note "Not present: $DATA_DIR"
    fi
}

remove_status_file() {
    heading "Runtime status file"
    if [[ -e "$STATUS_FILE" ]]; then
        rm -f "$STATUS_FILE"
        success "Removed $STATUS_FILE"
    else
        note "Not present: $STATUS_FILE"
    fi
}

remove_root_files() {
    heading "System files (requires sudo)"
    local needs_sudo=0
    [[ -f "$UDEV_RULE" ]] && needs_sudo=1
    [[ -f "$UINPUT_CONF" ]] && needs_sudo=1

    if [[ $needs_sudo -eq 0 ]]; then
        note "Nothing to do (neither file is present)"
        return
    fi

    if [[ -f "$UDEV_RULE" ]]; then
        sudo rm -f "$UDEV_RULE"
        sudo udevadm control --reload-rules
        sudo udevadm trigger
        success "Removed $UDEV_RULE"
    else
        note "Not present: $UDEV_RULE"
    fi

    if [[ -f "$UINPUT_CONF" ]]; then
        sudo rm -f "$UINPUT_CONF"
        success "Removed $UINPUT_CONF"
    else
        note "Not present: $UINPUT_CONF"
    fi
}

# ---------- Manual guidance for what's left ----------
print_manual_guidance() {
    heading "Optional: remove the shared / system-level bits"

    echo "Your git checkout is untouched:"
    note "$ECHOWRITE_REPO_DIR  (delete it if you no longer want the source)"

    echo
    echo "Drop yourself from the \`input\` group (only if nothing else uses it):"
    note "sudo gpasswd -d \"\$USER\" input"

    local pm
    pm=$(detect_package_manager)
    echo
    case "$pm" in
        apt)
            echo "Remove the system packages install.sh installed via apt:"
            note "sudo apt-get remove portaudio19-dev xclip"
            note "# python3 / python3-venv / python3-pip are left alone (commonly used)"
            ;;
        dnf)
            echo "Remove the system packages install.sh installed via dnf:"
            note "sudo dnf remove portaudio portaudio-devel xclip"
            note "# python3 / python3-devel are left alone (commonly used)"
            ;;
        pacman)
            echo "Remove the system packages install.sh installed via pacman:"
            note "sudo pacman -Rs portaudio xclip"
            note "# python is left alone (commonly used)"
            ;;
        *)
            warn "Could not detect a package manager; remove portaudio + xclip manually."
            ;;
    esac
}

# ---------- Main ----------
main() {
    parse_args "$@"
    check_not_root

    echo
    info "Uninstalling echowrite"
    print_plan
    confirm

    remove_service
    remove_launcher
    remove_data_dir
    remove_status_file
    remove_root_files

    print_manual_guidance

    echo
    success "Uninstall complete."
    echo
    note "Everything removed above can be restored by re-running ./install.sh."
    echo
}

main "$@"
