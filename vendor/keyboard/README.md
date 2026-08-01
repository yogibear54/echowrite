# Vendored `keyboard` library (patched)

This is a vendored copy of [`boppreh/keyboard`](https://github.com/boppreh/keyboard) version **0.13.5** with a single change: the `ensure_root()` guard in `keyboard/_nixcommon.py` has been neutered so the library can be used as a non-root user on Linux.

## Why

`boppreh/keyboard` does this on Linux:

```python
def ensure_root():
    if os.geteuid() != 0:
        raise ImportError('You must be root to use this library on linux.')
```

This is a hard UID check that runs *before* any device access. The installer for echowrite sets up the correct udev rules and adds the current user to the `input` group so they can read `/dev/input/event*` and write to `/dev/uinput` without root — but the library refuses to even try without `euid == 0`.

The patch makes `ensure_root()` a no-op. The library then proceeds to call `make_uinput()`, which does `open("/dev/uinput", "wb")` — and that open now succeeds because of the udev rules the installer put in place.

The second patch (also in `_nixkeyboard.py`) makes `build_tables()` tolerant of `dumpkeys` failing. `dumpkeys` needs a console file descriptor, which isn't available when the process has no controlling TTY (some SSH/container sessions, or when stdin is `/dev/null`). When that happens, we fall back to a hardcoded minimal keycode→name map covering the keys echowrite actually needs: modifiers, escape, and the F-keys used for paste-mode switching. Without the fallback, every key event would report `name='unknown'` and any app pattern-matching on `event.name` would break.

The `_darwin*.py` files (macOS-only) are untouched.

## How

```bash
pip install ./vendor/keyboard
```

The install script does this after `pip install -r requirements.txt`, so the vendored version wins in the venv.

## Updating from upstream

1. Check upstream: <https://github.com/boppreh/keyboard/releases>
2. Download the new source: `pip download --no-deps --no-binary=:all: keyboard==<new-version>`
3. Unzip it over `vendor/keyboard/keyboard/`, preserving our patch
4. Re-apply the patch (the surrounding code may have shifted)
5. Bump the version in `vendor/keyboard/pyproject.toml` (keep the `+echowrite.N` suffix to make the fork obvious)

## License

MIT — same as upstream. See `LICENSE.txt`.
