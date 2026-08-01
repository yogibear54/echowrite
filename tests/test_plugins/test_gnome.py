"""Tests for GnomeStatusPlugin."""
import sys
from unittest.mock import MagicMock

import pytest
from PIL import Image

from status_manager import Status


class _FakeIcon:
    """Stand-in for pystray.Icon that records construction and updates."""

    instances = []

    def __init__(self, name, icon, title):
        self.name = name
        self.icon = icon
        self.title = title
        self._stopped = False
        self.run_called = False
        _FakeIcon.instances.append(self)

    def run(self):
        # In real life this blocks. Tests must never block here.
        self.run_called = True

    def stop(self):
        self._stopped = True


class _FakePystray:
    """Stand-in for the pystray module."""

    def __init__(self):
        self.Icon = _FakeIcon
        _FakeIcon.instances = []


@pytest.fixture
def fake_pystray(monkeypatch):
    """Install a fake pystray module before the plugin imports it."""
    fake = _FakePystray()
    monkeypatch.setitem(sys.modules, "pystray", fake)
    return fake


@pytest.fixture
def plugin(fake_pystray):
    """Create a GnomeStatusPlugin backed by the fake pystray."""
    # Import inside the fixture so the fake pystray is in place first.
    from plugins.gnome import GnomeStatusPlugin
    return GnomeStatusPlugin()


@pytest.mark.unit
class TestGnomeStatusPluginInitialization:
    """Test GnomeStatusPlugin initialization."""

    def test_init_creates_tray_icon(self, plugin, fake_pystray):
        """A pystray.Icon should be constructed with name=voice2text."""
        assert len(fake_pystray.Icon.instances) == 1
        icon = fake_pystray.Icon.instances[0]
        assert icon.name == "voice2text"

    def test_init_uses_initial_not_started_state(self, plugin, fake_pystray):
        """The initial icon/title should reflect the NOT_STARTED status."""
        icon = fake_pystray.Icon.instances[0]
        assert icon.title == "Voice2Text - Not Started"
        assert isinstance(icon.icon, Image.Image)

    def test_init_starts_tray_in_background_thread(self, plugin, fake_pystray):
        """The tray's run() should be invoked in a daemon thread."""
        icon = fake_pystray.Icon.instances[0]
        # run() is called from the worker thread; give it a moment to fire.
        import time
        for _ in range(50):
            if icon.run_called:
                break
            time.sleep(0.01)
        assert icon.run_called is True

    def test_init_thread_is_daemon(self, plugin, fake_pystray):
        """The tray thread must be a daemon so it can't block process exit."""
        assert plugin._tray_thread is not None
        assert plugin._tray_thread.daemon is True
        assert plugin._tray_thread.name == "gnome-status-tray"

    def test_init_with_custom_icon_size(self, fake_pystray):
        """icon_size constructor arg should control the generated icon size."""
        from plugins.gnome import GnomeStatusPlugin
        plugin = GnomeStatusPlugin(icon_size=32)
        icon = fake_pystray.Icon.instances[0]
        assert icon.icon.size == (32, 32)
        assert plugin.icon_size == 32

    def test_init_raises_helpful_error_without_pystray(self, monkeypatch):
        """If pystray is not importable, __init__ should raise ImportError."""
        # Make `import pystray` fail.
        monkeypatch.setitem(sys.modules, "pystray", None)
        # The plugin module may already have a cached pystray; clear it.
        if "plugins.gnome" in sys.modules:
            del sys.modules["plugins.gnome"]
        from plugins.gnome import GnomeStatusPlugin
        with pytest.raises(ImportError, match="pystray is required"):
            GnomeStatusPlugin()
        # Re-import the module so subsequent fixtures work.
        del sys.modules["plugins.gnome"]


@pytest.mark.unit
class TestGnomeStatusPluginUpdateStatus:
    """Test GnomeStatusPlugin.update_status() method."""

    def _expected_color(self, status):
        return {
            Status.RECORDING:   (255, 0, 0),
            Status.PROCESSING:  (255, 165, 0),
            Status.IDLE:        (136, 136, 136),
            Status.NOT_STARTED: (102, 102, 102),
        }[status]

    def _icon_center_pixel(self, img):
        # The ellipse is centered; check the very middle pixel.
        w, h = img.size
        return img.getpixel((w // 2, h // 2))

    def test_recording_status_updates_icon_and_title(self, plugin, fake_pystray):
        plugin.update_status(Status.RECORDING)
        icon = fake_pystray.Icon.instances[0]
        assert icon.title == "Voice2Text - Recording"
        # Center pixel of the ellipse should be the status color.
        pixel = self._icon_center_pixel(icon.icon)
        assert pixel[:3] == self._expected_color(Status.RECORDING)

    def test_processing_status_updates_icon_and_title(self, plugin, fake_pystray):
        plugin.update_status(Status.PROCESSING)
        icon = fake_pystray.Icon.instances[0]
        assert icon.title == "Voice2Text - Processing"
        assert self._icon_center_pixel(icon.icon)[:3] == self._expected_color(Status.PROCESSING)

    def test_idle_status_updates_icon_and_title(self, plugin, fake_pystray):
        plugin.update_status(Status.IDLE)
        icon = fake_pystray.Icon.instances[0]
        assert icon.title == "Voice2Text - Idle"
        assert self._icon_center_pixel(icon.icon)[:3] == self._expected_color(Status.IDLE)

    def test_not_started_status_updates_icon_and_title(self, plugin, fake_pystray):
        plugin.update_status(Status.NOT_STARTED)
        icon = fake_pystray.Icon.instances[0]
        assert icon.title == "Voice2Text - Not Started"
        assert self._icon_center_pixel(icon.icon)[:3] == self._expected_color(Status.NOT_STARTED)

    def test_update_overwrites_previous_status(self, plugin, fake_pystray):
        """Each update_status call should fully replace the previous icon."""
        icon = fake_pystray.Icon.instances[0]
        plugin.update_status(Status.RECORDING)
        first = icon.icon
        plugin.update_status(Status.PROCESSING)
        second = icon.icon
        assert first is not second
        assert self._icon_center_pixel(first)[:3] == self._expected_color(Status.RECORDING)
        assert self._icon_center_pixel(second)[:3] == self._expected_color(Status.PROCESSING)

    def test_update_errors_are_caught(self, plugin, fake_pystray, capsys):
        """An exception from the tray backend should not crash the caller."""
        icon = fake_pystray.Icon.instances[0]
        # `icon` is a property that raises on assignment.
        type(icon).icon = property(
            fget=lambda self: self.__dict__["icon"],
            fset=lambda self, v: (_ for _ in ()).throw(RuntimeError("boom")),
        )
        # Should not raise.
        plugin.update_status(Status.RECORDING)
        captured = capsys.readouterr()
        assert "Failed to update GNOME tray status" in captured.out
        # Restore so the fixture's teardown / other tests aren't poisoned.
        del type(icon).icon


@pytest.mark.unit
class TestGnomeStatusPluginCleanup:
    """Test GnomeStatusPlugin.cleanup() method."""

    def test_cleanup_stops_tray(self, plugin, fake_pystray):
        icon = fake_pystray.Icon.instances[0]
        plugin.cleanup()
        assert icon._stopped is True

    def test_cleanup_clears_tray_reference(self, plugin, fake_pystray):
        plugin.cleanup()
        assert plugin._tray is None

    def test_cleanup_does_not_raise_when_already_cleaned(self, plugin, fake_pystray):
        plugin.cleanup()
        # A second cleanup should be a safe no-op.
        plugin.cleanup()

    def test_cleanup_errors_are_caught(self, fake_pystray, capsys):
        """Exceptions from tray.stop() should be swallowed and logged."""
        from plugins.gnome import GnomeStatusPlugin
        plugin = GnomeStatusPlugin()
        icon = fake_pystray.Icon.instances[0]
        icon.stop = MagicMock(side_effect=RuntimeError("boom"))
        # Should not raise.
        plugin.cleanup()
        captured = capsys.readouterr()
        assert "Failed to stop GNOME tray" in captured.out


@pytest.mark.unit
class TestGnomeStatusPluginEncoding:
    """Regression: titles must be pure ASCII.

    pystray's X11 backend latin-1-encodes the WM_NAME property. Em-dashes,
    emoji, or any non-ASCII character in the title crash the icon creation
    silently (the host app's try/except swallows it and no icon appears).
    """

    def test_initial_title_is_ascii(self, plugin, fake_pystray):
        icon = fake_pystray.Icon.instances[0]
        icon.title.encode("ascii")

    @pytest.mark.parametrize("status", list(Status))
    def test_all_status_titles_are_ascii(self, plugin, fake_pystray, status):
        plugin.update_status(status)
        icon = fake_pystray.Icon.instances[0]
        icon.title.encode("ascii")
