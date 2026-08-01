#!/usr/bin/env bash
# echowrite status-plugin setup
#
# Hybrid installer for the status-bar plugins:
#   - Installs the system dependencies each plugin needs
#   - Enables the chosen plugins in .env (STATUS_PLUGINS)
#   - Prints the remaining manual steps (i3 config wiring, GNOME extension)
#
# We automate only the safe, repeatable parts. The two environment-specific
# integrations are left as directions because they are fragile to script
# blindly: i3 configs vary wildly (status_command may be i3blocks/polybar/...),
# and the GNOME extension's package name/enabling depends on the session.
#
# Usage:
#   ./setup_plugins.sh                # set up both i3status and gnome
#   ./setup_plugins.sh i3status       # i3status only
#   ./setup_plugins.sh gnome          # gnome only
#
# Safe to re-run. Backs up .env to .env.bak before editing.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
ENV_FILE="$SCRIPT_DIR/.env"
WRAPPER="$SCRIPT_DIR/i3status_wrapper.py"

# ---------- args ----------
REQUESTED=("$@")
if [[ ${#REQUESTED[@]} -eq 0 ]]; then
    REQUESTED=(i3status gnome)
fi
for p in "${REQUESTED[@]}"; do
    case "$p" in
        i3status|gnome) ;;
        *) echo "Unknown plugin: $p (expected: i3status, gnome)" >&2; exit 1 ;;
    esac
done
has() { local x; for x in "${REQUESTED[@]}"; do [[ "$x" == "$1" ]] && return 0; done; return 1; }

# ---------- output helpers ----------
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

# ---------- helpers ----------
detect_package_manager() {
    if command -v apt-get >/dev/null 2>&1; then echo apt
    elif command -v dnf >/dev/null 2>&1; then echo dnf
    elif command -v pacman >/dev/null 2>&1; then echo pacman
    else echo unsupported
    fi
}

ensure_env_file() {
    if [[ ! -f "$ENV_FILE" ]]; then
        if [[ -f "$SCRIPT_DIR/.env.example" ]]; then
            cp "$SCRIPT_DIR/.env.example" "$ENV_FILE"
            success "Created $ENV_FILE from .env.example"
        else
            error ".env not found and no .env.example to copy from."
            exit 1
        fi
    fi
}

# set_env_var KEY VALUE — replace an existing KEY= line or append a new one.
# Backs up .env first. (Linux sed supports -i without a backup suffix.)
set_env_var() {
    local key="$1" val="$2"
    cp "$ENV_FILE" "$ENV_FILE.bak"
    if grep -q "^${key}=" "$ENV_FILE"; then
        sed -i "s|^${key}=.*|${key}=${val}|" "$ENV_FILE"
    else
        printf '%s=%s\n' "$key" "$val" >> "$ENV_FILE"
    fi
}

# ---------- plugin dependencies ----------
install_i3status_deps() {
    local pm="$1"
    heading "i3status — system dependency ($pm)"
    case "$pm" in
        apt)    sudo apt-get install -y i3status ;;
        dnf)    sudo dnf install -y i3status ;;
        pacman) sudo pacman -S --noconfirm --needed i3status ;;
        *)      warn "Install 'i3status' with your package manager." ;;
    esac
    success "i3status available"
}

install_gnome_deps() {
    local pm="$1"
    heading "gnome — system dependencies ($pm)"
    # pystray is a pip dep already installed via requirements.txt; on Linux it
    # needs GObject/GTK and an AppIndicator binding to actually render a tray.
    case "$pm" in
        apt)
            sudo apt-get update -qq
            sudo apt-get install -y python3-gi gir1.2-gtk-3.0
            # AppIndicator binding name varies by distro/version: try ayatana,
            # then the legacy one. Non-fatal — pystray can fall back to Gtk.
            if ! sudo apt-get install -y gir1.2-ayatanaappindicator3-0.1; then
                sudo apt-get install -y gir1.2-appindicator3-0.1 \
                    || warn "No AppIndicator gir binding found in apt; pystray will fall back to the Gtk backend."
            fi
            ;;
        dnf)
            sudo dnf install -y python3-gobject gtk3
            sudo dnf install -y libappindicator-gtk3 \
                || warn "libappindicator-gtk3 not found; pystray will fall back to the Gtk backend."
            ;;
        pacman)
            sudo pacman -S --noconfirm --needed python-gobject gtk3
            sudo pacman -S --noconfirm --needed libappindicator-gtk3 \
                || warn "libappindicator-gtk3 is not in the official repos (try the AUR); pystray will fall back to the Gtk backend."
            ;;
        *)
            warn "Install GObject/GTK + an AppIndicator binding manually for your distro."
            ;;
    esac
    success "GNOME backend dependencies installed"
}

# ---------- main ----------
main() {
    echo
    info "Setting up status plugins: ${REQUESTED[*]}"
    echo

    if [[ $EUID -eq 0 ]]; then
        error "Don't run this as root. It calls sudo where needed."
        exit 1
    fi

    local pm
    pm=$(detect_package_manager)

    # 1. Dependencies
    if has i3status; then install_i3status_deps "$pm"; fi
    if has gnome;    then install_gnome_deps "$pm"; fi

    # 2. Enable in .env
    heading ".env — STATUS_PLUGINS"
    ensure_env_file
    local joined
    joined=$(IFS=,; echo "${REQUESTED[*]}")
    set_env_var "STATUS_PLUGINS" "$joined"
    success "STATUS_PLUGINS=$joined  (written to $ENV_FILE; backup at .env.bak)"

    # 3. Manual integration steps (hybrid: we don't auto-edit dotfiles)
    if has i3status; then
        heading "i3status — manual step: wire it into your i3bar"
        note "Edit ~/.config/i3/config (or /etc/i3/config) and set your bar's status_command to:"
        printf "    %bstatus_command i3status | %s%b\n" "${BOLD}" "$WRAPPER" "${RESET}"
        note "Then reload i3:"
        printf "    %bi3-msg reload%b\n" "${BOLD}" "${RESET}"
        note "Using i3blocks / polybar / a custom command? See plugins/i3status/README.md"
    fi
    if has gnome; then
        heading "gnome — manual step: enable the AppIndicator shell extension"
        note "Install & enable 'AppIndicator and KStatusNotifierItem Support':"
        case "$pm" in
            apt)
                printf "    %bsudo apt-get install -y gnome-shell-extension-appindicator%b\n" "${BOLD}" "${RESET}"
                ;;
            dnf)
                printf "    %bsudo dnf install -y gnome-shell-extension-appindicator%b\n" "${BOLD}" "${RESET}"
                ;;
            pacman)
                note "On Arch, install via the AUR (e.g. gnome-shell-extension-appindicator) or extensions.gnome.org."
                ;;
        esac
        printf "    %bgnome-extensions enable appindicatorsupport@rgcjonas.gmail.com%b\n" "${BOLD}" "${RESET}"
        note "Log out and back in (or restart the GNOME shell) for the tray icon to appear."
        note "GNOME-only; irrelevant on other desktop environments."
    fi

    echo
    success "Plugin setup complete."
    note "Restart echowrite for the new STATUS_PLUGINS to take effect."
    echo
}

main "$@"
