# Vendored `keyboard` library (patched)

This is a vendored copy of [`boppreh/keyboard`](https://github.com/boppreh/keyboard)
upstream **0.13.5**, forked as **0.13.5+echowrite.3**. It carries several
echowrite-specific patches on top of upstream so the library runs as a
non-root daemon user and survives the real-world conditions a desktop hotkey
listener hits (suspend/resume, USB/BT hotplug, headless sessions).

Most changes from upstream are marked in-source with a `# PATCHED` comment,
so the fastest way to find the bulk of them is:

```sh
grep -RIn "PATCHED" vendor/keyboard/keyboard/
```

The hotplug-recovery additions in patch #3 (`EventDevice.close_input_file()`
and the `AggregatedEventDevice.start_reading` recovery loop) live in the same
two functions but aren't separately keyworded.

## Patches

### `_nixcommon.py`

1. **No-root operation.** `ensure_root()` is a no-op. Upstream hard-aborts when
   `os.geteuid() != 0`, *before* any device access. Our installer instead grants
   the running user access to `/dev/uinput` and `/dev/input/event*` via the
   `input` group plus a udev rule, so the UID check is unnecessary. With it
   gone, `make_uinput()`'s `open("/dev/uinput", "wb")` succeeds thanks to those
   rules.

2. **Skip unreadable devices instead of aborting.** The `EventDevice.input_file`
   property is accessed from a daemon reader thread. On a `Permission denied`
   open() it calls `exit()` — which ends only that thread, so the device is
   simply skipped — and does **not** print upstream's misleading "you must be
   sudo" message (sudo is not the fix here; some keyboard-class event devices
   such as power/sleep buttons simply aren't readable by the user even with a
   correct input-group setup).

3. **Survive device hotplug / suspend-resume.** In
   `AggregatedEventDevice.start_reading`, an `OSError` (e.g. ENODEV after
   suspend/resume, USB autosuspend, or a BT/USB unplug) is caught: the stale fd
   is closed via the new `close_input_file()` and the loop backs off ~1s. The
   lazy `input_file` property reopens the node on the next read, so hotkeys
   recover automatically. Upstream lets the reader thread die on ENODEV, which
   silently breaks **all** hotkeys (`AggregatedEventDevice` then blocks forever
   on an empty queue).

4. **Don't swallow non-permission open failures.** *(bug fix)* The `input_file`
   property's `except IOError` only handled `Permission denied`; every other
   open() failure (ENOENT/ENODEV — device unplugged, mid-run disappear after
   suspend/resume, or an unopenable keyboard-class device) was swallowed,
   leaving `_input_file == None`. The next `read_event()` then did `None.read()`
   → an `AttributeError` that `start_reading`'s `except OSError` can't catch —
   killing the reader thread and disabling all hotkeys (seen in the field as
   six reader threads dying at once after a suspend/resume). Non-permission
   open() errors now `raise`, so patch **#3** above handles them and recovers.

### `_nixkeyboard.py`

5. **Work without a controlling TTY / `dumpkeys`.** `build_tables()` wraps both
   `dumpkeys` invocations (`--keys-only` and `--long-info`) in
   `try/except (CalledProcessError, OSError, FileNotFoundError)`. When
   `dumpkeys` is unavailable — it needs a console file descriptor, which is
   missing in some SSH/container sessions or when stdin is `/dev/null` — it
   falls back to `_populate_minimal_keymap()`, a hardcoded keycode→name map for
   the keys echowrite actually cares about: modifiers (ctrl/alt/shift/super/menu),
   escape, and the F-keys used for paste-mode switching. Without this fallback,
   init() aborts and every key event reports `name='unknown'`, breaking anything
   that pattern-matches on `event.name`.

### Untouched

The `_darwin*.py` files (macOS-only) are unchanged.

## How

```bash
pip install ./vendor/keyboard
```

The install script does this after `pip install -r requirements.txt`, so the
vendored version wins in the venv. (Because the fork version string is
`+echowrite.N`, a plain reinstall won't pick up source changes on its own —
use `pip install --force-reinstall --no-deps ./vendor/keyboard` after editing.)

## Updating from upstream

1. Check upstream: <https://github.com/boppreh/keyboard/releases>
2. Download the new source: `pip download --no-deps --no-binary=:all: keyboard==<new-version>`
3. Unzip it over `vendor/keyboard/keyboard/`.
4. Re-apply every patch above (the surrounding code may have shifted). Most
   are marked `# PATCHED` in source — `grep -RIn "PATCHED" vendor/keyboard/keyboard/`
   finds those; the hotplug-recovery pieces (#3) live in
   `EventDevice.close_input_file` and `AggregatedEventDevice.start_reading`.
5. Bump the version in `vendor/keyboard/pyproject.toml`, keeping the
   `+echowrite.N` suffix (and incrementing `N`) so the fork is obvious and a
   plain reinstall will pick up the change.
6. Redeploy into the venv (see "How") and restart echowrite.

## License

MIT — same as upstream. See `LICENSE.txt`.
