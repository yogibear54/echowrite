"""Transcription provider package."""
from typing import Optional

from .base import TranscriptionProvider
from .replicate import ReplicateProvider

__all__ = ['TranscriptionProvider', 'ReplicateProvider', 'create_provider']


def create_provider(provider_name: Optional[str] = None, api_settings: Optional[dict] = None, api_token: Optional[str] = None) -> TranscriptionProvider:
    """Create a transcription provider instance based on provider name.
    
    Args:
        provider_name: Name of the provider to use (e.g., 'replicate'). 
                       If None, will use config.PROVIDER.
        api_settings: Provider-specific settings dictionary.
                      If None, will use config.API_SETTINGS.
        api_token: Provider API token. If None, provider will read from environment.
    
    Returns:
        TranscriptionProvider instance
        
    Raises:
        ValueError: If provider_name is not recognized or provider initialization fails
    """
    import config
    
    provider_name = (provider_name or config.PROVIDER).lower()
    api_settings = api_settings or config.API_SETTINGS
    
    if provider_name == 'replicate':
        return ReplicateProvider(api_token=api_token, api_settings=api_settings)
    else:
        raise ValueError(
            f"Unknown transcription provider: '{provider_name}'. "
            f"Available providers: replicate"
        )
