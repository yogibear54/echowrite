"""Shared fixtures and utilities for tests."""
import os
import sys
import tempfile
from pathlib import Path
from typing import Generator
from unittest.mock import MagicMock, Mock

import numpy as np
import pytest
from scipy.io.wavfile import write as wav_write


# Provide a fake faster_whisper module so tests can import providers.local_whisper
# without requiring the optional faster-whisper package to be installed.
_fake_faster_whisper = MagicMock()
_fake_faster_whisper.WhisperModel = MagicMock()
sys.modules.setdefault('faster_whisper', _fake_faster_whisper)


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_audio_data() -> np.ndarray:
    """Generate mock audio data as numpy array (float32)."""
    # Generate 1 second of audio at 44100 Hz (sine wave at 440 Hz)
    sample_rate = 44100
    duration = 1.0
    frequency = 440.0
    
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    audio = np.sin(2 * np.pi * frequency * t).astype(np.float32)
    
    # Reshape for mono channel
    return audio.reshape(-1, 1)


@pytest.fixture
def mock_audio_data_stereo() -> np.ndarray:
    """Generate mock stereo audio data."""
    sample_rate = 44100
    duration = 1.0
    frequency = 440.0
    
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    left = np.sin(2 * np.pi * frequency * t).astype(np.float32)
    right = np.sin(2 * np.pi * frequency * 1.5 * t).astype(np.float32)
    
    # Shape: (samples, 2)
    return np.column_stack([left, right])


@pytest.fixture
def test_audio_file(temp_dir: Path, mock_audio_data: np.ndarray) -> Path:
    """Create a test WAV file."""
    audio_file = temp_dir / "test_audio.wav"
    sample_rate = 44100
    channels = 1
    
    # Convert to int16 for WAV
    audio_clipped = np.clip(mock_audio_data, -1.0, 1.0)
    audio_int16 = (audio_clipped * 32767).astype(np.int16)
    
    wav_write(str(audio_file), sample_rate, audio_int16)
    return audio_file


@pytest.fixture
def mock_replicate_upload_response():
    """Mock successful Replicate file upload response."""
    return {
        'urls': {
            'get': 'https://replicate.delivery/test/audio.wav'
        }
    }


@pytest.fixture
def mock_replicate_transcribe_string():
    """Mock Replicate transcription response (string format)."""
    return "This is a test transcription."


@pytest.fixture
def mock_replicate_transcribe_dict():
    """Mock Replicate transcription response (dict format with 'text' key)."""
    return {'text': 'This is a test transcription from dict.'}


@pytest.fixture
def mock_replicate_transcribe_list():
    """Mock Replicate transcription response (list format)."""
    return ['This', 'is', 'a', 'test', 'transcription', 'from', 'list']


@pytest.fixture
def mock_sounddevice_stream(mock_audio_data: np.ndarray):
    """Mock sounddevice.InputStream that returns test audio data."""
    mock_stream = MagicMock()
    
    # Simulate reading chunks
    chunk_size = int(44100 * 0.1)  # 100ms chunks
    chunks = []
    for i in range(0, len(mock_audio_data), chunk_size):
        chunks.append(mock_audio_data[i:i+chunk_size])
    
    chunk_iter = iter(chunks)
    
    def read(size):
        try:
            chunk = next(chunk_iter)
            # Pad if needed
            if len(chunk) < size:
                padding = np.zeros((size - len(chunk), chunk.shape[1]), dtype=chunk.dtype)
                chunk = np.vstack([chunk, padding])
            return chunk, False  # (data, overflowed)
        except StopIteration:
            return np.zeros((size, mock_audio_data.shape[1]), dtype=mock_audio_data.dtype), False
    
    mock_stream.read = Mock(side_effect=read)
    mock_stream.__enter__ = Mock(return_value=mock_stream)
    mock_stream.__exit__ = Mock(return_value=None)
    
    return mock_stream


@pytest.fixture
def mock_pyperclip():
    """Mock pyperclip for clipboard operations."""
    clipboard_content = {}
    
    def copy(text):
        clipboard_content['text'] = text
    
    def paste():
        return clipboard_content.get('text', '')
    
    mock = MagicMock()
    mock.copy = Mock(side_effect=copy)
    mock.paste = Mock(side_effect=paste)
    return mock


@pytest.fixture
def mock_pyautogui():
    """Mock pyautogui for paste operations."""
    mock = MagicMock()
    mock.hotkey = Mock()
    return mock


@pytest.fixture
def mock_keyboard_events():
    """Mock keyboard events for hotkey testing."""
    events = []
    
    class MockKeyboardEvent:
        def __init__(self, name, event_type):
            self.name = name
            self.event_type = event_type
    
    return {
        'events': events,
        'MockEvent': MockKeyboardEvent
    }


@pytest.fixture
def mock_api_token():
    """Mock API token for testing."""
    return "test_replicate_api_token_12345"


@pytest.fixture
def mock_api_settings():
    """Mock API settings for testing."""
    return {
        'model': 'test/model:version',
        'task': 'transcribe',
        'language': 'None',
        'timestamp': 'chunk',
        'batch_size': 64,
        'diarise_audio': False,
    }


@pytest.fixture
def replicate_provider(mock_api_token, mock_api_settings):
    """Create a ReplicateProvider instance for testing."""
    from providers.replicate import ReplicateProvider
    
    # Set environment variable
    os.environ['REPLICATE_API_TOKEN'] = mock_api_token
    
    return ReplicateProvider(api_token=mock_api_token, api_settings=mock_api_settings)


@pytest.fixture
def mock_whisper_settings():
    """Default settings dict for LocalWhisperProvider."""
    return {
        'model_size': 'base',
        'device': 'cpu',
        'compute_type': 'int8',
        'language': None,
        'beam_size': 5,
    }


@pytest.fixture
def local_whisper_provider(mock_whisper_settings):
    """Create a LocalWhisperProvider instance with faster_whisper mocked at module level."""
    import sys
    from unittest.mock import MagicMock

    # Ensure faster_whisper is a fake module so the provider can import it
    fake = MagicMock()
    fake.WhisperModel = MagicMock(return_value=MagicMock())
    sys.modules['faster_whisper'] = fake

    from providers.local_whisper import LocalWhisperProvider
    return LocalWhisperProvider(api_settings=mock_whisper_settings)


@pytest.fixture(autouse=True)
def mock_pyautogui_module():
    """Mock pyautogui module to prevent X11 connection issues during tests."""
    import sys
    from unittest.mock import MagicMock
    
    # Create a mock pyautogui module if it doesn't exist
    if 'pyautogui' not in sys.modules:
        mock_pyautogui = MagicMock()
        mock_pyautogui.hotkey = MagicMock()
        sys.modules['pyautogui'] = mock_pyautogui
    else:
        # If it already exists, ensure hotkey is mocked
        if not hasattr(sys.modules['pyautogui'], 'hotkey'):
            sys.modules['pyautogui'].hotkey = MagicMock()
    
    yield
    
    # Cleanup: remove mock if we added it
    if 'pyautogui' in sys.modules and isinstance(sys.modules['pyautogui'], MagicMock):
        del sys.modules['pyautogui']


@pytest.fixture
def temp_status_file(temp_dir: Path) -> Path:
    """Create temporary status file path for testing."""
    return temp_dir / "test_status.json"


@pytest.fixture
def mock_plugin():
    """Create a mock plugin for testing StatusManager."""
    plugin = MagicMock()
    plugin.update_status = Mock()
    plugin.cleanup = Mock()
    return plugin


@pytest.fixture
def incomplete_plugin():
    """Create a plugin without update_status method."""
    return MagicMock()  # No update_status method


@pytest.fixture(autouse=True)
def cleanup_env():
    """Clean up environment variables after each test."""
    original_token = os.environ.get('REPLICATE_API_TOKEN')
    yield
    # Restore original or remove
    if original_token:
        os.environ['REPLICATE_API_TOKEN'] = original_token
    elif 'REPLICATE_API_TOKEN' in os.environ:
        del os.environ['REPLICATE_API_TOKEN']
