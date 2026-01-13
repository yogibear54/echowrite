# Implementing New Status Plugins

This guide explains how to add new status indicator plugins to the voice2text application. The application uses a plugin-based architecture that makes it easy to add support for different status display mechanisms (e.g., system tray, desktop notifications, status bars, etc.).

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Quick Start](#quick-start)
- [Step-by-Step Guide](#step-by-step-guide)
- [Plugin Interface](#plugin-interface)
- [Example: Desktop Notification Plugin](#example-desktop-notification-plugin)
- [Example: System Tray Plugin](#example-system-tray-plugin)
- [Example: File-Based Status Plugin](#example-file-based-status-plugin)
- [Configuration](#configuration)
- [Testing Your Plugin](#testing-your-plugin)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

## Architecture Overview

The status plugin system uses object-oriented programming principles:

- **Abstraction**: `StatusPlugin` base class defines the interface
- **Inheritance**: Each plugin inherits from the base class
- **Polymorphism**: `StatusManager` works with any plugin implementation
- **Encapsulation**: Plugin-specific logic is isolated in plugin classes

### Plugin Flow

```
StatusManager
    ↓
Status Change (NOT_STARTED, IDLE, RECORDING, PROCESSING)
    ↓
Notify All Registered Plugins
    ↓
Your Plugin (plugins/your_plugin/__init__.py)
    ↓
Update Status Display (file, notification, tray, etc.)
```

### Status States

The application has four status states that plugins receive:

| Status | Description |
|--------|-------------|
| **NOT_STARTED** | App is created but not yet initialized |
| **IDLE** | App is ready and waiting for input |
| **RECORDING** | Currently recording audio (Ctrl+Alt pressed) |
| **PROCESSING** | Transcribing recorded audio |

## Quick Start

1. Create a new directory: `plugins/your_plugin/`
2. Create `plugins/your_plugin/__init__.py`
3. Inherit from `StatusPlugin`
4. Implement the `update_status()` method
5. Register your plugin in `start.py`
6. Test your implementation

## Step-by-Step Guide

### Step 1: Create Plugin Directory

Create a new directory for your plugin:

```bash
mkdir -p plugins/your_plugin
```

### Step 2: Create Plugin Class

Create `plugins/your_plugin/__init__.py`:

```python
"""Your plugin name for status indicators."""
from status_manager import Status
from plugins.base import StatusPlugin


class YourStatusPlugin(StatusPlugin):
    """Plugin that displays status using Your Method."""
    
    def __init__(self, config_option: str = "default"):
        """Initialize your status plugin.
        
        Args:
            config_option: Configuration option for your plugin
        """
        self.config_option = config_option
        # Initialize your plugin resources here
        # self.setup_display()
    
    def update_status(self, status: Status):
        """Update the status indicator with the new status.
        
        Args:
            status: The new application status (Status enum)
        """
        # Map status to your display format
        if status == Status.RECORDING:
            display_text = "🔴 Recording..."
        elif status == Status.PROCESSING:
            display_text = "🔄 Processing..."
        elif status == Status.IDLE:
            display_text = "⏸️ Idle"
        elif status == Status.NOT_STARTED:
            display_text = "⚪ Not Started"
        else:
            display_text = ""
        
        # Update your status display
        # self.update_display(display_text)
    
    def cleanup(self):
        """Clean up resources when the application shuts down."""
        # Close connections, remove temporary files, etc.
        # self.close_display()
        pass
```

### Step 3: Register Plugin in Package

Update `plugins/__init__.py` to export your plugin:

```python
"""Status indicator plugins package."""
from .base import StatusPlugin
from .i3status import I3StatusPlugin
from .your_plugin import YourStatusPlugin  # Add this import

__all__ = ['StatusPlugin', 'I3StatusPlugin', 'YourStatusPlugin']  # Add to exports
```

### Step 4: Register Plugin in Application

Update `start.py` to register your plugin. Find where plugins are registered (usually in the `__init__` or setup method):

```python
# In start.py, find the plugin registration section
from plugins.your_plugin import YourStatusPlugin

# Register your plugin
your_plugin = YourStatusPlugin(config.YOUR_PLUGIN_OPTION)
self.status_manager.register_plugin(your_plugin)
```

### Step 5: Add Configuration (Optional)

If your plugin needs configuration, add it to `config.py`:

```python
# Status plugin configuration
YOUR_PLUGIN_OPTION = _get_str_env('YOUR_PLUGIN_OPTION', 'default_value')
```

And update `.env.example`:

```env
# Status Plugin Options
YOUR_PLUGIN_OPTION=default_value
```

## Plugin Interface

### Required Methods

#### `update_status(status: Status) -> None`

**Purpose**: Update the status indicator with the new application status.

**Parameters**:
- `status`: A `Status` enum value (NOT_STARTED, IDLE, RECORDING, or PROCESSING)

**Requirements**:
- Must handle all four status states
- Should be non-blocking (don't perform long operations)
- Should handle errors gracefully (don't raise exceptions that crash the app)
- Will be called frequently as status changes

**Example**:

```python
def update_status(self, status: Status):
    """Update status indicator."""
    try:
        if status == Status.RECORDING:
            self._show_recording()
        elif status == Status.PROCESSING:
            self._show_processing()
        elif status == Status.IDLE:
            self._show_idle()
        elif status == Status.NOT_STARTED:
            self._show_not_started()
    except Exception as e:
        print(f"⚠️  Error updating status plugin: {e}")
```

### Optional Methods

#### `cleanup() -> None`

**Purpose**: Clean up resources when the application shuts down.

**When to implement**:
- If your plugin opens connections that need closing
- If your plugin creates temporary files
- If your plugin has background threads/processes
- If your plugin needs to remove UI elements

**Example**:

```python
def cleanup(self):
    """Clean up resources."""
    try:
        if hasattr(self, 'connection'):
            self.connection.close()
        if hasattr(self, 'temp_file') and self.temp_file.exists():
            self.temp_file.unlink()
    except Exception as e:
        print(f"⚠️  Error cleaning up plugin: {e}")
```

## Example: Desktop Notification Plugin

Here's a complete example implementing desktop notifications using `notify-send` or a Python notification library:

```python
"""Desktop notification plugin for status indicators."""
import subprocess
from status_manager import Status
from plugins.base import StatusPlugin


class NotificationPlugin(StatusPlugin):
    """Plugin that shows desktop notifications for status changes."""
    
    def __init__(self, urgency: str = "normal"):
        """Initialize notification plugin.
        
        Args:
            urgency: Notification urgency level (low, normal, critical)
        """
        self.urgency = urgency
        self.last_status = None  # Track to avoid duplicate notifications
    
    def update_status(self, status: Status):
        """Show desktop notification for status changes."""
        # Only notify on status changes
        if status == self.last_status:
            return
        
        self.last_status = status
        
        # Map status to notification text
        if status == Status.RECORDING:
            title = "Voice2Text"
            message = "🔴 Recording audio..."
            urgency = "normal"
        elif status == Status.PROCESSING:
            title = "Voice2Text"
            message = "🔄 Processing transcription..."
            urgency = "normal"
        elif status == Status.IDLE:
            title = "Voice2Text"
            message = "⏸️ Ready - Press Ctrl+Alt to record"
            urgency = "low"
        elif status == Status.NOT_STARTED:
            # Don't notify on initial state
            return
        else:
            return
        
        try:
            # Use notify-send command (Linux)
            subprocess.run(
                ['notify-send', 
                 '--urgency', urgency,
                 '--app-name', 'Voice2Text',
                 title, message],
                check=False,
                timeout=2
            )
        except Exception as e:
            print(f"⚠️  Failed to show notification: {e}")
    
    def cleanup(self):
        """No cleanup needed for notifications."""
        pass
```

## Example: System Tray Plugin

Here's an example for a system tray icon (using `pystray`):

```python
"""System tray plugin for status indicators."""
import pystray
from PIL import Image, ImageDraw
from status_manager import Status
from plugins.base import StatusPlugin


class TrayPlugin(StatusPlugin):
    """Plugin that displays status in system tray."""
    
    def __init__(self):
        """Initialize tray plugin."""
        self.status = Status.NOT_STARTED
        self.icon = None
        self.tray = None
        self._create_tray()
    
    def _create_icon(self, color: str) -> Image.Image:
        """Create tray icon with specified color."""
        # Create a simple colored circle icon
        image = Image.new('RGB', (64, 64), color)
        draw = ImageDraw.Draw(image)
        draw.ellipse([16, 16, 48, 48], fill=color, outline='white', width=2)
        return image
    
    def _create_tray(self):
        """Create system tray icon."""
        menu = pystray.Menu(
            pystray.MenuItem('Quit', self._quit)
        )
        
        self.icon = self._create_icon('#666666')  # Default gray
        self.tray = pystray.Icon("voice2text", self.icon, "Voice2Text", menu)
    
    def update_status(self, status: Status):
        """Update tray icon based on status."""
        self.status = status
        
        # Map status to icon color
        if status == Status.RECORDING:
            color = '#ff0000'  # Red
            tooltip = "Voice2Text - Recording..."
        elif status == Status.PROCESSING:
            color = '#ffa500'  # Orange
            tooltip = "Voice2Text - Processing..."
        elif status == Status.IDLE:
            color = '#888888'  # Gray
            tooltip = "Voice2Text - Idle"
        elif status == Status.NOT_STARTED:
            color = '#666666'  # Dark gray
            tooltip = "Voice2Text - Not Started"
        else:
            color = '#ffffff'  # White
            tooltip = "Voice2Text"
        
        try:
            self.icon = self._create_icon(color)
            if self.tray:
                self.tray.icon = self.icon
                self.tray.title = tooltip
        except Exception as e:
            print(f"⚠️  Failed to update tray icon: {e}")
    
    def _quit(self, icon, item):
        """Quit handler for tray menu."""
        icon.stop()
    
    def run(self):
        """Run the tray icon (call this in a separate thread)."""
        if self.tray:
            self.tray.run()
    
    def cleanup(self):
        """Stop tray icon."""
        if self.tray:
            self.tray.stop()
```

**Note**: For tray plugins, you may need to run them in a separate thread:

```python
# In start.py
import threading
from plugins.tray import TrayPlugin

tray_plugin = TrayPlugin()
self.status_manager.register_plugin(tray_plugin)

# Run tray in separate thread
tray_thread = threading.Thread(target=tray_plugin.run, daemon=True)
tray_thread.start()
```

## Example: File-Based Status Plugin

Here's a simple example that writes status to a JSON file (similar to i3status but simpler):

```python
"""File-based status plugin for status indicators."""
import json
from pathlib import Path
from status_manager import Status
from plugins.base import StatusPlugin


class FileStatusPlugin(StatusPlugin):
    """Plugin that writes status to a JSON file."""
    
    def __init__(self, status_file: str = "/tmp/voice2text_status.json"):
        """Initialize file status plugin.
        
        Args:
            status_file: Path to the status file
        """
        self.status_file = Path(status_file)
        self.status_file.parent.mkdir(parents=True, exist_ok=True)
        self.update_status(Status.NOT_STARTED)
    
    def update_status(self, status: Status):
        """Write status to file."""
        status_data = {
            "status": status.value,
            "display": self._get_display_text(status),
            "color": self._get_color(status)
        }
        
        try:
            with open(self.status_file, 'w', encoding='utf-8') as f:
                json.dump(status_data, f, indent=2)
        except Exception as e:
            print(f"⚠️  Failed to write status file: {e}")
    
    def _get_display_text(self, status: Status) -> str:
        """Get display text for status."""
        mapping = {
            Status.RECORDING: "🔴 Recording...",
            Status.PROCESSING: "🔄 Processing...",
            Status.IDLE: "⏸️ Idle",
            Status.NOT_STARTED: "⚪ Not Started"
        }
        return mapping.get(status, "")
    
    def _get_color(self, status: Status) -> str:
        """Get color for status."""
        mapping = {
            Status.RECORDING: "#ff0000",
            Status.PROCESSING: "#ffa500",
            Status.IDLE: "#888888",
            Status.NOT_STARTED: "#666666"
        }
        return mapping.get(status, "#ffffff")
    
    def cleanup(self):
        """Remove status file on cleanup."""
        try:
            if self.status_file.exists():
                self.status_file.unlink()
        except Exception as e:
            print(f"⚠️  Failed to cleanup status file: {e}")
```

## Configuration

### Plugin Selection

Plugins are registered in `start.py`. You can conditionally register plugins based on configuration:

```python
# In start.py
import config
from plugins.your_plugin import YourStatusPlugin

# Conditionally register plugin
if config.ENABLE_YOUR_PLUGIN:
    your_plugin = YourStatusPlugin(config.YOUR_PLUGIN_OPTION)
    self.status_manager.register_plugin(your_plugin)
```

### Plugin-Specific Settings

Add plugin-specific settings to `config.py`:

```python
# Status plugin configuration
ENABLE_YOUR_PLUGIN = _get_bool_env('ENABLE_YOUR_PLUGIN', False)
YOUR_PLUGIN_OPTION = _get_str_env('YOUR_PLUGIN_OPTION', 'default_value')
```

And update `.env.example`:

```env
# Status Plugin Configuration
ENABLE_YOUR_PLUGIN=true
YOUR_PLUGIN_OPTION=default_value
```

## Testing Your Plugin

The project includes comprehensive tests for the status plugin system. You can reference the existing test files for examples:

- `tests/test_plugins/test_status_manager.py` - Tests for StatusManager (plugin registration, status updates, cleanup)
- `tests/test_plugins/test_base.py` - Tests for StatusPlugin base class
- `tests/test_plugins/test_i3status.py` - Tests for I3StatusPlugin (file operations, status updates, error handling)

### Unit Tests

Create `tests/test_plugins/test_your_plugin.py`:

```python
"""Unit tests for YourStatusPlugin."""
from unittest.mock import Mock, patch
import pytest

from status_manager import Status
from plugins.your_plugin import YourStatusPlugin


@pytest.mark.unit
class TestYourStatusPlugin:
    """Test YourStatusPlugin implementation."""
    
    @pytest.fixture
    def plugin(self):
        """Create YourStatusPlugin instance for testing."""
        return YourStatusPlugin()
    
    def test_initializes_correctly(self, plugin):
        """Test plugin initializes with default settings."""
        assert plugin is not None
    
    def test_update_status_recording(self, plugin):
        """Test status update for RECORDING state."""
        plugin.update_status(Status.RECORDING)
        # Assert your plugin's state changed correctly
        # assert plugin.display_text == "🔴 Recording..."
    
    def test_update_status_processing(self, plugin):
        """Test status update for PROCESSING state."""
        plugin.update_status(Status.PROCESSING)
        # Assert your plugin's state changed correctly
    
    def test_update_status_idle(self, plugin):
        """Test status update for IDLE state."""
        plugin.update_status(Status.IDLE)
        # Assert your plugin's state changed correctly
    
    def test_update_status_not_started(self, plugin):
        """Test status update for NOT_STARTED state."""
        plugin.update_status(Status.NOT_STARTED)
        # Assert your plugin's state changed correctly
    
    def test_cleanup(self, plugin):
        """Test plugin cleanup."""
        # Setup plugin state
        plugin.update_status(Status.RECORDING)
        
        # Cleanup should not raise
        plugin.cleanup()
        
        # Assert resources are cleaned up
        # assert not plugin.connection.is_open()
```

### Integration Tests

For plugins that interact with external systems, create integration tests:

```python
"""Integration tests for YourStatusPlugin."""
import pytest
from status_manager import Status
from plugins.your_plugin import YourStatusPlugin


@pytest.mark.integration
class TestYourStatusPluginIntegration:
    """Integration tests using real system interactions."""
    
    @pytest.fixture
    def plugin(self):
        """Create plugin instance."""
        return YourStatusPlugin()
    
    def test_status_updates_visible(self, plugin):
        """Test that status updates are actually visible."""
        plugin.update_status(Status.RECORDING)
        # Verify status is visible in your system
        # This might require checking files, notifications, etc.
        
        plugin.update_status(Status.IDLE)
        # Verify status changed
```

### Running Tests

```bash
# Run all plugin tests
pytest tests/test_plugins/

# Run unit tests (mocked, fast)
pytest tests/test_plugins/test_your_plugin.py

# Run integration tests (real system interactions)
pytest tests/test_plugins/test_your_plugin.py -m integration

# Run with coverage
pytest tests/test_plugins/ --cov=plugins --cov-report=term-missing
```

### Test Fixtures

The test suite provides helpful fixtures in `tests/conftest.py`:

- `temp_status_file` - Temporary file path for testing file-based plugins
- `mock_plugin` - Mock plugin with `update_status` and `cleanup` methods
- `incomplete_plugin` - Mock plugin without `update_status` method (for testing error cases)
- `temp_dir` - Temporary directory for test files

You can use these fixtures in your tests:

```python
def test_my_plugin(temp_status_file):
    """Test my plugin with temporary file."""
    plugin = MyPlugin(str(temp_status_file))
    # ... test implementation
```

## Best Practices

### 1. Error Handling

- **Never raise exceptions** that could crash the application
- **Handle errors gracefully** with try/except blocks
- **Provide user-friendly error messages** via `print()`
- **Continue functioning** even if one update fails

```python
def update_status(self, status: Status):
    """Update status indicator."""
    try:
        # Your update logic
        self._do_update(status)
    except ConnectionError:
        print("⚠️  Status plugin: Connection error, will retry on next update")
    except Exception as e:
        print(f"⚠️  Status plugin error: {e}")
        # Don't raise - allow app to continue
```

### 2. Performance

- **Keep updates fast** - don't block the main thread
- **Avoid duplicate updates** - check if status actually changed
- **Use efficient operations** - cache resources when possible
- **Consider async operations** for slow updates (files, network)

```python
def update_status(self, status: Status):
    """Update status indicator."""
    # Skip if status hasn't changed
    if status == self.last_status:
        return
    
    self.last_status = status
    # Perform update...
```

### 3. Resource Management

- **Implement `cleanup()`** if you create resources
- **Close connections** properly
- **Remove temporary files**
- **Stop background threads**

```python
def cleanup(self):
    """Clean up resources."""
    try:
        if hasattr(self, 'connection') and self.connection:
            self.connection.close()
        if hasattr(self, 'thread') and self.thread.is_alive():
            self.thread.stop()
    except Exception as e:
        print(f"⚠️  Cleanup error: {e}")
```

### 4. Status Mapping

Use consistent status-to-display mappings:

```python
STATUS_DISPLAY = {
    Status.RECORDING: {
        "text": "🔴 Recording...",
        "color": "#ff0000",
        "icon": "recording"
    },
    Status.PROCESSING: {
        "text": "🔄 Processing...",
        "color": "#ffa500",
        "icon": "processing"
    },
    Status.IDLE: {
        "text": "⏸️ Idle",
        "color": "#888888",
        "icon": "idle"
    },
    Status.NOT_STARTED: {
        "text": "⚪ Not Started",
        "color": "#666666",
        "icon": "not_started"
    }
}
```

### 5. Thread Safety

If your plugin uses threads or async operations:

- **Use thread-safe operations** for shared state
- **Don't block the main thread** in `update_status()`
- **Use queues** for thread communication
- **Handle thread cleanup** in `cleanup()`

### 6. Logging

Use consistent logging style:

```python
print("📊 Status plugin initialized")  # Info
print("⚠️  Status plugin warning")     # Warning
print("❌ Status plugin error")        # Error
```

## Troubleshooting

### Plugin Not Updating

**Symptom**: Status changes but plugin doesn't reflect them.

**Solutions**:
1. Check that plugin is registered: `status_manager.register_plugin(plugin)`
2. Verify `update_status()` is being called (add print statements)
3. Check for exceptions in `update_status()` (they should be caught)
4. Verify plugin initialization completed successfully

### Plugin Crashes Application

**Symptom**: Application crashes when status changes.

**Solutions**:
1. Wrap all code in `update_status()` with try/except
2. Don't raise exceptions from `update_status()`
3. Check for uninitialized resources
4. Verify thread safety if using threads

### Resource Leaks

**Symptom**: Resources not cleaned up on exit.

**Solutions**:
1. Implement `cleanup()` method
2. Ensure `cleanup()` is called (StatusManager does this automatically)
3. Close all connections, files, threads
4. Remove temporary files

### Import Errors

**Error**: `ModuleNotFoundError: No module named 'your_plugin'`

**Solutions**:
1. Ensure plugin is in `plugins/your_plugin/` directory
2. Add `__init__.py` file in plugin directory
3. Import in `plugins/__init__.py`
4. Check Python path

### Configuration Issues

**Symptom**: Plugin doesn't use correct configuration.

**Solutions**:
1. Verify configuration in `config.py`
2. Check environment variables are set
3. Verify `.env` file is loaded
4. Check configuration is passed to plugin constructor

## Plugin Checklist

Before submitting your plugin, ensure:

- [ ] Inherits from `StatusPlugin`
- [ ] Implements `update_status()` method correctly
- [ ] Handles all four status states (NOT_STARTED, IDLE, RECORDING, PROCESSING)
- [ ] Handles errors gracefully (doesn't raise exceptions)
- [ ] Implements `cleanup()` if needed
- [ ] Registered in `plugins/__init__.py`
- [ ] Registered in `start.py` (or conditionally based on config)
- [ ] Configuration added to `config.py` (if needed)
- [ ] Dependencies added to `requirements.txt` (if needed)
- [ ] Unit tests written and passing
- [ ] Integration tests written (if applicable)
- [ ] Documentation updated (if needed)
- [ ] Tested with actual status changes
- [ ] No resource leaks (connections, files, threads)

## Getting Help

If you encounter issues:

1. Check existing plugins (`plugins/i3status/`) for reference
2. Review test examples (`tests/test_plugins/`) - comprehensive test suite with 40+ tests
3. Check the base class interface (`plugins/base.py`)
4. Review `StatusManager` implementation (`status_manager.py`)
5. Run the test suite to verify your plugin works correctly: `pytest tests/test_plugins/`
5. Review this documentation

## Contributing

When contributing a new status plugin:

1. Follow the code style of existing plugins
2. Write comprehensive tests
3. Update this documentation if needed
4. Test with real status changes
5. Document any plugin-specific requirements
6. Consider cross-platform compatibility
7. Keep dependencies minimal

---

**Happy coding!** 🚀
