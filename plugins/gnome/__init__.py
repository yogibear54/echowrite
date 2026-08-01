"""GNOME top-bar (system tray / AppIndicator) status plugin."""
import threading

from PIL import Image, ImageDraw

from status_manager import Status
from plugins.base import StatusPlugin


# Status -> (RGB color, tooltip title). Colors match the i3status plugin so
# the same status looks the same across plugins.
# NOTE: titles are sent through Xlib's WM_NAME which is latin-1, so they must
# be pure ASCII. Em-dash / emoji in the title will crash pystray's X backend.
_STATUS_DISPLAY = {
    Status.RECORDING:   ((255, 0, 0),     "Voice2Text - Recording"),
    Status.PROCESSING:  ((255, 165, 0),   "Voice2Text - Processing"),
    Status.IDLE:        ((136, 136, 136), "Voice2Text - Idle"),
    Status.NOT_STARTED: ((102, 102, 102), "Voice2Text - Not Started"),
}


class GnomeStatusPlugin(StatusPlugin):
    """Plugin that displays status in the GNOME top bar via a tray icon.

    Uses the AppIndicator protocol (via pystray), which GNOME shows in the
    top bar when the "AppIndicator and KStatusNotifierItem Support" extension
    is installed. The plugin is intentionally click-free: it just shows a
    colored icon and tooltip that update with the current status.
    """

    def __init__(self, icon_size: int = 64):
        """Initialize the GNOME tray status plugin.

        Args:
            icon_size: Tray icon size in pixels (default 64, recommended for
                crisp rendering on HiDPI top bars).

        Raises:
            ImportError: If pystray is not installed.
        """
        self.icon_size = icon_size
        self._tray = None
        self._tray_thread = None
        self._start_tray()

    def _start_tray(self):
        """Import pystray, build the icon, and start its loop in a daemon thread."""
        try:
            import pystray
        except ImportError as e:
            raise ImportError(
                "pystray is required for the GNOME status plugin. "
                "Install it with: pip install pystray"
            ) from e

        self._tray = pystray.Icon(
            name="voice2text",
            icon=self._make_icon(Status.NOT_STARTED),
            title="Voice2Text - Not Started",
        )
        # pystray.Icon.run() blocks until stop() is called, so it must run
        # in its own thread. Daemon=True so the thread won't block process exit.
        self._tray_thread = threading.Thread(
            target=self._tray.run,
            name="gnome-status-tray",
            daemon=True,
        )
        self._tray_thread.start()

    def _make_icon(self, status: Status) -> Image.Image:
        """Create a colored-circle tray icon for the given status."""
        color, _ = _STATUS_DISPLAY.get(status, ((255, 255, 255), ""))
        size = self.icon_size
        image = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(image)
        margin = max(2, size // 10)
        draw.ellipse(
            (margin, margin, size - margin, size - margin),
            fill=color + (255,),
        )
        return image

    def update_status(self, status: Status):
        """Update the tray icon and tooltip to reflect the new status.

        Args:
            status: The new application status.
        """
        if self._tray is None:
            return
        _, title = _STATUS_DISPLAY.get(status, (None, "Voice2Text"))
        try:
            self._tray.icon = self._make_icon(status)
            self._tray.title = title
        except Exception as e:
            print(f"⚠️  Failed to update GNOME tray status: {e}")

    def cleanup(self):
        """Stop the tray icon so the background thread can exit."""
        if self._tray is None:
            return
        try:
            self._tray.stop()
        except Exception as e:
            print(f"⚠️  Failed to stop GNOME tray: {e}")
        self._tray = None
