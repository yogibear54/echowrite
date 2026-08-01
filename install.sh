#!/usr/bin/env bash
# echowrite installer
#
# Sets up everything needed to run the voice dictation tool on Linux:
#   - System dependencies (PortAudio, xclip, Python venv tooling)
#   - udev rules so the user can access /dev/uinput without sudo
#   - Adds the current user to the `input` group
#   - Loads the uinput kernel module
#   - Python virtualenv with project requirements
#   - Launcher script on PATH (~/.local/bin/echowrite)
#   - systemd user service (installed but not enabled)
#
# Usage:  ./install.sh
# Re-run any time to repair or update the venv.

set -euo pipefail

# ---------- Configuration ----------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
ECHOWRITE_REPO_DIR="${ECHOWRITE_REPO_DIR:-$SCRIPT_DIR}"
DATA_DIR="${XDG_DATA_HOME:-$HOME/.local/share}/echowrite"
CONFIG_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/echowrite"
VENV_DIR="$DATA_DIR/venv"
BIN_DIR="${XDG_BIN_HOME:-$HOME/.local/bin}"
SYSTEMD_USER_DIR="$HOME/.config/systemd/user"
SERVICE_NAME="echowrite"
PYTHON_MIN_MAJOR=3
PYTHON_MIN_MINOR=10

# ---------- Output helpers ----------
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
        error "Don't run install.sh as root. It calls sudo where needed."
        exit 1
    fi
}

check_repo_layout() {
    if [[ ! -f "$ECHOWRITE_REPO_DIR/requirements.txt" ]] || [[ ! -f "$ECHOWRITE_REPO_DIR/start.py" ]]; then
        error "install.sh must be run from inside the echowrite repository."
        note "Expected: $ECHOWRITE_REPO_DIR/requirements.txt and start.py"
        exit 1
    fi
}

check_python() {
    if ! command -v python3 >/dev/null 2>&1; then
        error "python3 not found. Install it via your package manager and re-run."
        exit 1
    fi
    local ver
    ver=$(python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])')
    if ! python3 -c "import sys; sys.exit(0 if sys.version_info >= ($PYTHON_MIN_MAJOR, $PYTHON_MIN_MINOR) else 1)"; then
        error "Python $ver found, but ${PYTHON_MIN_MAJOR}.${PYTHON_MIN_MINOR}+ required."
        exit 1
    fi
    success "Python $ver"
}

detect_package_manager() {
    if command -v apt-get >/dev/null 2>&1; then echo apt
    elif command -v dnf >/dev/null 2>&1; then echo dnf
    elif command -v pacman >/dev/null 2>&1; then echo pacman
    else echo unsupported
    fi
}

# ---------- Steps ----------
install_system_deps() {
    local pm
    pm=$(detect_package_manager)
    heading "System dependencies ($pm)"

    case "$pm" in
        apt)
            sudo apt-get update
            sudo apt-get install -y \
                python3 python3-venv python3-pip python3-tk \
                portaudio19-dev \
                xclip
            ;;
        dnf)
            sudo dnf install -y \
                python3 python3-devel python3-tkinter \
                portaudio portaudio-devel \
                xclip
            ;;
        pacman)
            sudo pacman -S --noconfirm --needed \
                python tk \
                portaudio \
                xclip
            ;;
        *)
            error "Unsupported distro. Install these manually and re-run:"
            note "  - python3 (>= ${PYTHON_MIN_MAJOR}.${PYTHON_MIN_MINOR}) with venv and pip"
            note "  - tkinter (python3-tk / python3-tkinter / tk) — required for paste"
            note "  - PortAudio headers (portaudio19-dev / portaudio-devel / portaudio)"
            note "  - xclip (or xsel)"
            exit 1
            ;;
    esac
    success "System dependencies installed"
}

setup_udev() {
    heading "udev rules (/dev/uinput, /dev/input/event*)"
    local rule_file="/etc/udev/rules.d/99-echowrite-input.rules"
    local rule_content
    rule_content=$(cat <<'EOF'
# Allow members of the `input` group to access keyboard devices and uinput
# for global hotkey support (e.g. python `keyboard` library).
KERNEL=="event*",   GROUP="input", MODE="0640"
KERNEL=="uinput",   GROUP="input", MODE="0660"
SUBSYSTEM=="input", GROUP="input", MODE="0640"
EOF
)
    if [[ -f "$rule_file" ]] && diff -q <(echo "$rule_content") "$rule_file" >/dev/null 2>&1; then
        note "Already in place"
    else
        echo "$rule_content" | sudo tee "$rule_file" >/dev/null
        sudo udevadm control --reload-rules
        sudo udevadm trigger
        success "Installed $rule_file"
    fi
}

setup_input_group() {
    heading "input group"
    if ! getent group input >/dev/null; then
        sudo groupadd input
    fi
    if id -nG "$USER" | tr ' ' '\n' | grep -qx input; then
        note "$USER is already a member"
    else
        sudo usermod -aG input "$USER"
        warn "Added $USER to 'input' group — log out and back in for it to take effect."
    fi
}

setup_uinput_module() {
    heading "uinput kernel module"
    if [[ ! -f /etc/modules-load.d/uinput.conf ]] || ! grep -qx uinput /etc/modules-load.d/uinput.conf; then
        echo uinput | sudo tee /etc/modules-load.d/uinput.conf >/dev/null
    fi
    if ! lsmod | grep -q '^uinput'; then
        sudo modprobe uinput
    fi
    success "Loaded"
}

setup_venv() {
    heading "Python virtualenv"
    if [[ ! -d "$VENV_DIR" ]]; then
        info "Creating $VENV_DIR"
        python3 -m venv "$VENV_DIR"
    else
        note "Reusing $VENV_DIR"
    fi

    info "Upgrading pip"
    "$VENV_DIR/bin/pip" install --quiet --upgrade pip wheel

    info "Installing requirements from $ECHOWRITE_REPO_DIR/requirements.txt"
    "$VENV_DIR/bin/pip" install --quiet -r "$ECHOWRITE_REPO_DIR/requirements.txt"

    info "Installing vendored keyboard fork from $ECHOWRITE_REPO_DIR/vendor/keyboard"
    "$VENV_DIR/bin/pip" install --quiet "$ECHOWRITE_REPO_DIR/vendor/keyboard"

    success "Dependencies installed in $VENV_DIR"
}

create_launcher() {
    heading "Launcher ($BIN_DIR/echowrite)"
    mkdir -p "$BIN_DIR"
    local launcher="$BIN_DIR/echowrite"
    cat > "$launcher" <<EOF
#!/usr/bin/env bash
# echowrite launcher — auto-generated by install.sh; do not edit.
exec "$VENV_DIR/bin/python" "$ECHOWRITE_REPO_DIR/start.py" "\$@"
EOF
    chmod +x "$launcher"
    success "Installed $launcher"

    if [[ ":$PATH:" != *":$BIN_DIR:"* ]]; then
        warn "$BIN_DIR is not on your PATH. Add to your shell rc:"
        note "  export PATH=\"\$HOME/.local/bin:\$PATH\""
    fi
}

setup_systemd_service() {
    heading "systemd user service"

    if ! command -v systemctl >/dev/null 2>&1; then
        warn "systemctl not found — skipping service install."
        return
    fi
    if ! systemctl --user status >/dev/null 2>&1; then
        warn "systemd --user is unavailable (no user session?) — skipping service install."
        return
    fi

    mkdir -p "$SYSTEMD_USER_DIR"
    local service_file="$SYSTEMD_USER_DIR/$SERVICE_NAME.service"
    cat > "$service_file" <<EOF
[Unit]
Description=echowrite voice dictation
After=graphical-session.target
PartOf=graphical-session.target

[Service]
Type=simple
ExecStart=$VENV_DIR/bin/python $ECHOWRITE_REPO_DIR/start.py
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
EOF
    systemctl --user daemon-reload
    success "Installed $service_file (not enabled yet)"
}

# ---------- Main ----------
main() {
    echo
    info "Installing echowrite from $ECHOWRITE_REPO_DIR"
    echo

    check_not_root
    check_repo_layout
    check_python
    install_system_deps
    setup_udev
    setup_input_group
    setup_uinput_module
    setup_venv
    create_launcher
    setup_systemd_service

    echo
    success "Install complete."
    echo
    info "Next steps:"
    note "1. If you were added to the 'input' group, log out and back in (or run: exec \$SHELL)."
    note "2. Run \`echowrite\` to start the tool."
    note "3. Optional status-bar plugins (i3 / GNOME): ./setup_plugins.sh"
    note "4. Optional autostart:  systemctl --user enable --now $SERVICE_NAME"
    note "5. To update later:    git pull && ./install.sh"
    echo
}

main "$@"
