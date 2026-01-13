# Configuration settings for voice dictation tool
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def _get_int_env(key: str, default: int, min_val: int = None, max_val: int = None) -> int:
    """Get integer environment variable with validation."""
    try:
        value = int(os.getenv(key, default))
        if min_val is not None and value < min_val:
            print(f"⚠️  Warning: {key}={value} is below minimum {min_val}, using default {default}")
            return default
        if max_val is not None and value > max_val:
            print(f"⚠️  Warning: {key}={value} is above maximum {max_val}, using default {default}")
            return default
        return value
    except (ValueError, TypeError):
        print(f"⚠️  Warning: Invalid {key} value, using default {default}")
        return default


def _get_float_env(key: str, default: float, min_val: float = None, max_val: float = None) -> float:
    """Get float environment variable with validation."""
    try:
        value = float(os.getenv(key, default))
        if min_val is not None and value < min_val:
            print(f"⚠️  Warning: {key}={value} is below minimum {min_val}, using default {default}")
            return default
        if max_val is not None and value > max_val:
            print(f"⚠️  Warning: {key}={value} is above maximum {max_val}, using default {default}")
            return default
        return value
    except (ValueError, TypeError):
        print(f"⚠️  Warning: Invalid {key} value, using default {default}")
        return default


def _get_str_env(key: str, default: str) -> str:
    """Get string environment variable with validation."""
    value = os.getenv(key, default)
    if not value or not value.strip():
        print(f"⚠️  Warning: {key} is empty, using default '{default}'")
        return default
    return value


# Recording settings with environment variable overrides
RECORDING_SETTINGS = {
    'sample_rate': _get_int_env('SAMPLE_RATE', 44100, min_val=8000, max_val=48000),
    'channels': 1,
    'dtype': 'float32'  # float32 is more widely supported than float64
}

# Transcription provider selection
PROVIDER = _get_str_env('TRANSCRIPTION_PROVIDER', 'replicate').lower()

# Provider-specific API settings with environment variable overrides
# These settings are provider-agnostic but may be used differently by each provider
API_SETTINGS = {
    'model': _get_str_env('REPLICATE_MODEL', 'vaibhavs10/incredibly-fast-whisper:3ab86df6c8f54c11309d4d1f930ac292bad43ace52d10c80d87eb258b3c9f79c'),
    # Model parameters (optional, defaults are usually optimal)
    'task': 'transcribe',  # Options: 'transcribe' or 'translate'
    'language': 'None',  # Language code or 'None' for auto-detection
    'timestamp': 'chunk',  # Timestamp format: 'chunk' or 'word'
    'batch_size': 64,  # Batch size for processing
    'diarise_audio': False,  # Speaker diarization (requires hf_token if True)
}

# Hotkey configuration
HOTKEY = 'ctrl+alt'

# Custom vocabulary for better recognition
CUSTOM_VOCABULARY = {
    # Technical tools and platforms
    'n8n': ['n8n', 'n 8 n', 'n eight n', 'nateon', 'AN10', 'N810', 'N8N', 'A10'],
    'Retell': ['Retell', 'retell', 're-tell', 'retail', 'retale', 're tell'],
}

VOCABULARY_HINT = ', '.join(CUSTOM_VOCABULARY.keys())

# File settings
TEMP_FILE_PREFIX = 'voice_recording_'
RECORDINGS_FILE = 'recordings.json'
TEMP_DIR = 'temp'

# Audio settings with environment variable overrides
AUDIO_FORMAT = 'wav'
MIN_RECORDING_SECONDS = _get_float_env('MIN_RECORDING_SECONDS', 1.0, min_val=0.1)

# Max recording duration from environment (default: 5 minutes from PRD)
MAX_RECORDING_MINUTES = _get_float_env('MAX_RECORDING_MINUTES', 5.0, min_val=0.1, max_val=60.0)
MAX_RECORDING_SECONDS = MAX_RECORDING_MINUTES * 60

# Validate that MIN_RECORDING_SECONDS is less than MAX_RECORDING_SECONDS
if MIN_RECORDING_SECONDS >= MAX_RECORDING_SECONDS:
    print(f"⚠️  Warning: MIN_RECORDING_SECONDS ({MIN_RECORDING_SECONDS}) >= MAX_RECORDING_SECONDS ({MAX_RECORDING_SECONDS})")
    print(f"   Adjusting MIN_RECORDING_SECONDS to {MAX_RECORDING_SECONDS * 0.1}")
    MIN_RECORDING_SECONDS = MAX_RECORDING_SECONDS * 0.1

# Status indicator plugin configuration
STATUS_PLUGINS = _get_str_env('STATUS_PLUGINS', 'i3status').split(',')
STATUS_PLUGINS = [p.strip() for p in STATUS_PLUGINS if p.strip()]

# i3 status plugin configuration
I3_STATUS_FILE = _get_str_env('I3_STATUS_FILE', '/tmp/voice2text_status')
