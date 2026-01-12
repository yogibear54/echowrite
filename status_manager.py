"""Status manager for voice dictation tool - tracks and broadcasts application state."""
from typing import List
from enum import Enum


class Status(Enum):
    """Application status states."""
    IDLE = "idle"
    RECORDING = "recording"
    PROCESSING = "processing"


class StatusManager:
    """Manages application status and notifies registered plugins."""
    
    def __init__(self):
        self.status = Status.IDLE
        self.plugins: List = []
    
    def register_plugin(self, plugin):
        """Register a status indicator plugin."""
        if hasattr(plugin, 'update_status'):
            self.plugins.append(plugin)
            # Initialize plugin with current status
            plugin.update_status(self.status)
        else:
            raise ValueError(f"Plugin {plugin} does not implement update_status method")
    
    def set_status(self, status: Status):
        """Update status and notify all plugins."""
        self.status = status
        for plugin in self.plugins:
            try:
                plugin.update_status(status)
            except Exception as e:
                print(f"⚠️  Error updating plugin {plugin}: {e}")
    
    def get_status(self) -> Status:
        """Get current status."""
        return self.status
    
    def cleanup(self):
        """Clean up all plugins."""
        for plugin in self.plugins:
            if hasattr(plugin, 'cleanup'):
                try:
                    plugin.cleanup()
                except Exception as e:
                    print(f"⚠️  Error cleaning up plugin {plugin}: {e}")
