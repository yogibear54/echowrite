"""Base class for transcription providers."""
from abc import ABC, abstractmethod
from typing import Optional


class TranscriptionProvider(ABC):
    """Abstract base class for transcription providers.
    
    All transcription providers must inherit from this class and implement
    the transcribe method. This ensures a consistent interface across all providers.
    """
    
    @abstractmethod
    def transcribe(self, audio_file_path: str) -> Optional[str]:
        """Transcribe audio file to text.
        
        Args:
            audio_file_path: Path to the audio file to transcribe
            
        Returns:
            Transcribed text as a string, or None if transcription failed
            
        Raises:
            FileNotFoundError: If audio_file_path does not exist
            ProviderError: For provider-specific errors (to be defined by subclasses)
        """
    
    def cleanup(self):
        """Clean up resources when the provider is no longer needed.
        
        Subclasses can override this method to perform cleanup operations
        such as closing connections, cleaning up temporary files, etc.
        """
