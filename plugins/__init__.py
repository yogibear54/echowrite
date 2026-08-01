"""Status indicator plugins package."""
from .base import StatusPlugin
from .i3status import I3StatusPlugin
from .gnome import GnomeStatusPlugin

__all__ = ['StatusPlugin', 'I3StatusPlugin', 'GnomeStatusPlugin']
