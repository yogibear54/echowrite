"""Status indicator plugins package."""
from .base import StatusPlugin
from .i3status import I3StatusPlugin

__all__ = ['StatusPlugin', 'I3StatusPlugin']
