"""Local Whisper provider for transcription services (uses faster-whisper, runs in-process)."""
import os
from typing import Optional

from faster_whisper import WhisperModel

from .base import TranscriptionProvider
import config


class LocalWhisperProvider(TranscriptionProvider):
    """Transcription provider using a local faster-whisper model.

    The model is loaded eagerly at construction time so the first recording
    does not pay the load cost. Configure via config.WHISPER_SETTINGS or
    the WHISPER_* environment variables.
    """

    def __init__(self, api_token: Optional[str] = None, api_settings: Optional[dict] = None):
        """Initialize local Whisper provider and load the model.

        Args:
            api_token: Unused for local transcription; accepted for interface
                compatibility with the provider factory.
            api_settings: Dict with keys model_size, device, compute_type,
                language, beam_size. If None, uses config.WHISPER_SETTINGS.
        """
        self.api_settings = api_settings if api_settings is not None else config.WHISPER_SETTINGS

        self.model_size = self.api_settings.get('model_size', 'small')
        self.device = self.api_settings.get('device', 'cpu')
        self.compute_type = self.api_settings.get('compute_type', 'int8')
        # language: None means auto-detect (faster-whisper accepts None)
        self.language = self.api_settings.get('language', None)
        self.beam_size = self.api_settings.get('beam_size', 5)

        print(f"🧠 Loading local Whisper model: {self.model_size} "
              f"(device={self.device}, compute_type={self.compute_type})...")
        self.model = WhisperModel(
            self.model_size,
            device=self.device,
            compute_type=self.compute_type,
        )
        print(f"✅ Local Whisper model loaded.")

    def transcribe(self, audio_file_path: str) -> Optional[str]:
        """Transcribe audio file using the local Whisper model.

        Args:
            audio_file_path: Path to the audio file to transcribe.

        Returns:
            Transcribed text as a string, or None if transcription failed
            or returned empty.
        """
        try:
            if not os.path.exists(audio_file_path):
                print(f"❌ Audio file not found: {audio_file_path}")
                return None

            print(f"🔄 Transcribing audio using local Whisper ({self.model_size})...")

            segments, info = self.model.transcribe(
                audio_file_path,
                beam_size=self.beam_size,
                language=self.language,
            )

            # segments is a generator; consume it and join
            transcribed_text = " ".join(seg.text.strip() for seg in segments).strip()

            if not transcribed_text:
                print("⚠️  Transcription returned empty result")
                return None

            return transcribed_text

        except Exception as e:
            print(f"❌ Transcription error: {e}")
            return None

    def cleanup(self):
        """Release model resources. faster-whisper holds no GPU state to release
        beyond what Python's GC handles, so this is a no-op."""
        pass