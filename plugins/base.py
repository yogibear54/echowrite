"""Base class for status indicator plugins."""
from abc import ABC, abstractmethod
from status_manager import Status


class StatusPlugin(ABC):
    """Base class for status indicator plugins."""
    
    @abstractmethod
    def update_status(self, status: Status):
        """Update the status indicator with the new status.
        
        Args:
            status: The new application status
        """
        pass
    
    def cleanup(self):
        """Clean up resources when the application shuts down."""
        pass
