# Implementing New Transcription Providers

This guide explains how to add new transcription providers to the voice2text application. The application uses a provider-based architecture that makes it easy to add support for new transcription services.

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Quick Start](#quick-start)
- [Step-by-Step Guide](#step-by-step-guide)
- [Provider Interface](#provider-interface)
- [Example: OpenAI Whisper API](#example-openai-whisper-api)
- [Example: Google Speech-to-Text](#example-google-speech-to-text)
- [Configuration](#configuration)
- [Testing Your Provider](#testing-your-provider)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

## Architecture Overview

The provider system uses object-oriented programming principles:

- **Abstraction**: `TranscriptionProvider` base class defines the interface
- **Inheritance**: Each provider inherits from the base class
- **Polymorphism**: `VoiceDictationTool` works with any provider implementation
- **Encapsulation**: Provider-specific logic is isolated in provider classes

### Provider Flow

```
VoiceDictationTool
    ↓
Provider Factory (providers/__init__.py)
    ↓
Your Provider (providers/your_provider.py)
    ↓
Transcription API
```

## Quick Start

1. Create a new file: `providers/your_provider.py`
2. Inherit from `TranscriptionProvider`
3. Implement the `transcribe()` method
4. Register your provider in `providers/__init__.py`
5. Add configuration to `config.py`
6. Test your implementation

## Step-by-Step Guide

### Step 1: Create Provider Class

Create a new file `providers/your_provider.py`:

```python
"""Your provider name for transcription services."""
import os
from typing import Optional

from .base import TranscriptionProvider
import config


class YourProvider(TranscriptionProvider):
    """Transcription provider using Your Service API."""
    
    def __init__(self, api_token: Optional[str] = None, api_settings: Optional[dict] = None):
        """Initialize Your provider.
        
        Args:
            api_token: API token. If None, will read from YOUR_API_TOKEN env var.
            api_settings: Provider-specific settings dict. If None, will use config.API_SETTINGS.
        """
        self.api_token = api_token or os.getenv('YOUR_API_TOKEN')
        self.api_settings = api_settings or config.API_SETTINGS
        
        if not self.api_token:
            raise ValueError("YOUR_API_TOKEN not found. Please set it in environment variables.")
        
        # Initialize your API client here
        # self.client = YourAPIClient(self.api_token)
    
    def transcribe(self, audio_file_path: str) -> Optional[str]:
        """Transcribe audio file using Your Service API.
        
        Args:
            audio_file_path: Path to the audio file to transcribe
            
        Returns:
            Transcribed text as a string, or None if transcription failed
        """
        try:
            # Verify file exists
            if not os.path.exists(audio_file_path):
                print(f"❌ Audio file not found: {audio_file_path}")
                return None
            
            # Your transcription logic here
            # 1. Upload/prepare audio file
            # 2. Call transcription API
            # 3. Parse response
            # 4. Return transcribed text
            
            # Example:
            # with open(audio_file_path, 'rb') as audio_file:
            #     result = self.client.transcribe(audio_file)
            # return result.text
            
            return "Transcribed text"
            
        except Exception as e:
            print(f"❌ Transcription error: {e}")
            return None
    
    def cleanup(self):
        """Clean up resources when provider is no longer needed."""
        # Close connections, cleanup temporary files, etc.
        pass
```

### Step 2: Register Provider in Factory

Update `providers/__init__.py` to include your provider:

```python
"""Transcription provider package."""
from typing import Optional

from .base import TranscriptionProvider
from .replicate import ReplicateProvider
from .your_provider import YourProvider  # Add this import

__all__ = ['TranscriptionProvider', 'ReplicateProvider', 'YourProvider', 'create_provider']


def create_provider(provider_name: Optional[str] = None, api_settings: Optional[dict] = None, api_token: Optional[str] = None) -> TranscriptionProvider:
    """Create a transcription provider instance based on provider name."""
    import config
    
    provider_name = (provider_name or config.PROVIDER).lower()
    api_settings = api_settings or config.API_SETTINGS
    
    if provider_name == 'replicate':
        return ReplicateProvider(api_token=api_token, api_settings=api_settings)
    elif provider_name == 'your_provider':  # Add this
        return YourProvider(api_token=api_token, api_settings=api_settings)
    else:
        raise ValueError(
            f"Unknown transcription provider: '{provider_name}'. "
            f"Available providers: replicate, your_provider"
        )
```

### Step 3: Add Configuration

Update `config.py` to support your provider:

```python
# Transcription provider selection
PROVIDER = _get_str_env('TRANSCRIPTION_PROVIDER', 'replicate').lower()

# Provider-specific API settings
API_SETTINGS = {
    # Replicate settings (existing)
    'model': _get_str_env('REPLICATE_MODEL', 'vaibhavs10/incredibly-fast-whisper:...'),
    'task': 'transcribe',
    # ... other Replicate settings
    
    # Your provider settings (add these)
    'your_provider_model': _get_str_env('YOUR_PROVIDER_MODEL', 'default-model'),
    'your_provider_language': _get_str_env('YOUR_PROVIDER_LANGUAGE', 'en'),
    # ... other settings
}
```

### Step 4: Update Requirements

Add your provider's dependencies to `requirements.txt`:

```txt
# Existing dependencies
...

# Your provider dependencies
your-api-client>=1.0.0
```

### Step 5: Update Environment Variables

Add your API token to `.env.example`:

```env
# Replicate API Token
REPLICATE_API_TOKEN=r8_your_token_here

# Your Provider API Token
YOUR_API_TOKEN=your_token_here

# Provider Selection
TRANSCRIPTION_PROVIDER=your_provider
```

## Provider Interface

### Required Methods

#### `transcribe(audio_file_path: str) -> Optional[str]`

**Purpose**: Transcribe an audio file to text.

**Parameters**:
- `audio_file_path`: Path to the WAV audio file (guaranteed to exist when called)

**Returns**:
- `str`: Transcribed text on success
- `None`: On failure (file not found, API error, network error, etc.)

**Requirements**:
- Must handle file existence validation
- Must handle API errors gracefully (return None, don't raise exceptions)
- Should provide user-friendly error messages via `print()`
- Must return `None` on any failure

**Example**:

```python
def transcribe(self, audio_file_path: str) -> Optional[str]:
    """Transcribe audio file."""
    try:
        if not os.path.exists(audio_file_path):
            print(f"❌ Audio file not found: {audio_file_path}")
            return None
        
        # Your transcription logic
        result = self._call_api(audio_file_path)
        
        if not result:
            return None
        
        return result.strip()
    except Exception as e:
        print(f"❌ Transcription error: {e}")
        return None
```

### Optional Methods

#### `cleanup() -> None`

**Purpose**: Clean up resources when provider is no longer needed.

**When to implement**:
- If your provider opens connections that need closing
- If your provider creates temporary files
- If your provider has background threads/processes

**Example**:

```python
def cleanup(self):
    """Clean up resources."""
    if hasattr(self, 'client'):
        self.client.close()
    # Clean up temporary files, etc.
```

## Example: OpenAI Whisper API

Here's a complete example implementing OpenAI's Whisper API:

```python
"""OpenAI Whisper API provider for transcription services."""
import os
from typing import Optional

import openai
from .base import TranscriptionProvider
import config


class OpenAIProvider(TranscriptionProvider):
    """Transcription provider using OpenAI Whisper API."""
    
    def __init__(self, api_token: Optional[str] = None, api_settings: Optional[dict] = None):
        """Initialize OpenAI provider."""
        self.api_token = api_token or os.getenv('OPENAI_API_TOKEN')
        self.api_settings = api_settings or config.API_SETTINGS
        
        if not self.api_token:
            raise ValueError("OPENAI_API_TOKEN not found. Please set it in environment variables.")
        
        # Initialize OpenAI client
        openai.api_key = self.api_token
        self.client = openai
    
    def transcribe(self, audio_file_path: str) -> Optional[str]:
        """Transcribe audio file using OpenAI Whisper API."""
        try:
            if not os.path.exists(audio_file_path):
                print(f"❌ Audio file not found: {audio_file_path}")
                return None
            
            print("🔄 Transcribing audio using OpenAI Whisper...")
            
            # OpenAI accepts file objects directly
            with open(audio_file_path, 'rb') as audio_file:
                transcript = self.client.Audio.transcribe(
                    model=self.api_settings.get('openai_model', 'whisper-1'),
                    file=audio_file,
                    language=self.api_settings.get('language', None),
                    response_format='text'
                )
            
            if not transcript:
                print("⚠️  Transcription returned empty result")
                return None
            
            return transcript.strip()
            
        except openai.error.AuthenticationError:
            print("⚠️  Authentication failed. Please check your OPENAI_API_TOKEN.")
            return None
        except openai.error.RateLimitError:
            print("⚠️  Rate limit exceeded. Please wait a moment and try again.")
            return None
        except Exception as e:
            print(f"❌ Transcription error: {e}")
            return None
    
    def cleanup(self):
        """Clean up OpenAI client."""
        # OpenAI client doesn't require explicit cleanup
        pass
```

## Example: Google Speech-to-Text

Here's an example for Google Cloud Speech-to-Text:

```python
"""Google Cloud Speech-to-Text provider for transcription services."""
import os
from typing import Optional

from google.cloud import speech
from .base import TranscriptionProvider
import config


class GoogleProvider(TranscriptionProvider):
    """Transcription provider using Google Cloud Speech-to-Text."""
    
    def __init__(self, api_token: Optional[str] = None, api_settings: Optional[dict] = None):
        """Initialize Google provider."""
        # Google uses service account JSON file or credentials
        credentials_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
        if not credentials_path:
            raise ValueError("GOOGLE_APPLICATION_CREDENTIALS not found. Please set path to service account JSON.")
        
        self.api_settings = api_settings or config.API_SETTINGS
        self.client = speech.SpeechClient()
    
    def transcribe(self, audio_file_path: str) -> Optional[str]:
        """Transcribe audio file using Google Speech-to-Text."""
        try:
            if not os.path.exists(audio_file_path):
                print(f"❌ Audio file not found: {audio_file_path}")
                return None
            
            print("🔄 Transcribing audio using Google Speech-to-Text...")
            
            # Read audio file
            with open(audio_file_path, 'rb') as audio_file:
                content = audio_file.read()
            
            # Configure recognition
            config_obj = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=44100,
                language_code=self.api_settings.get('language', 'en-US'),
            )
            
            audio = speech.RecognitionAudio(content=content)
            
            # Perform transcription
            response = self.client.recognize(config=config_obj, audio=audio)
            
            # Extract transcript
            if not response.results:
                print("⚠️  Transcription returned empty result")
                return None
            
            # Combine all results
            transcript = ' '.join([result.alternatives[0].transcript for result in response.results])
            
            return transcript.strip()
            
        except Exception as e:
            print(f"❌ Transcription error: {e}")
            return None
```

## Configuration

### Provider Selection

Users can select your provider via environment variable:

```bash
TRANSCRIPTION_PROVIDER=your_provider
```

Or in `.env`:

```env
TRANSCRIPTION_PROVIDER=your_provider
```

### Provider-Specific Settings

Add provider-specific settings to `config.API_SETTINGS`. The factory passes these to your provider's `__init__()`:

```python
# In config.py
API_SETTINGS = {
    # Shared settings
    'language': 'en',
    
    # Your provider settings
    'your_provider_model': 'default-model',
    'your_provider_quality': 'high',
}
```

Access in your provider:

```python
def transcribe(self, audio_file_path: str) -> Optional[str]:
    model = self.api_settings.get('your_provider_model', 'default')
    # Use model...
```

## Testing Your Provider

The project includes comprehensive tests for the provider system. You can reference the existing test files for examples:

- `tests/test_providers/test_base.py` - Tests for TranscriptionProvider base class
- `tests/test_providers/test_replicate.py` - Tests for ReplicateProvider (unit tests with mocking)
- `tests/test_providers/test_replicate_integration.py` - Tests for ReplicateProvider (real API tests)

### Unit Tests

Create `tests/test_providers/test_your_provider.py`:

```python
"""Unit tests for YourProvider."""
from unittest.mock import MagicMock, Mock, patch
import pytest

from providers.your_provider import YourProvider


@pytest.mark.unit
class TestYourProvider:
    """Test YourProvider implementation."""
    
    @pytest.fixture
    def provider(self, mock_api_token, mock_api_settings):
        """Create YourProvider instance for testing."""
        return YourProvider(api_token=mock_api_token, api_settings=mock_api_settings)
    
    def test_initializes_with_api_token(self, provider, mock_api_token):
        """Test provider initializes with API token."""
        assert provider.api_token == mock_api_token
    
    def test_raises_error_without_token(self, mock_api_settings):
        """Test provider raises error without API token."""
        with pytest.raises(ValueError, match="YOUR_API_TOKEN not found"):
            YourProvider(api_token=None, api_settings=mock_api_settings)
    
    @patch('providers.your_provider.YourAPIClient')
    def test_transcribe_success(self, mock_client_class, provider, test_audio_file):
        """Test successful transcription."""
        mock_client = MagicMock()
        mock_client.transcribe.return_value = MagicMock(text="Test transcription")
        mock_client_class.return_value = mock_client
        provider.client = mock_client
        
        result = provider.transcribe(str(test_audio_file))
        
        assert result == "Test transcription"
    
    def test_transcribe_handles_missing_file(self, provider):
        """Test transcription handles missing file."""
        result = provider.transcribe('nonexistent.wav')
        assert result is None
    
    @patch('providers.your_provider.YourAPIClient')
    def test_transcribe_handles_api_errors(self, mock_client_class, provider, test_audio_file):
        """Test transcription handles API errors."""
        mock_client = MagicMock()
        mock_client.transcribe.side_effect = Exception("API Error")
        mock_client_class.return_value = mock_client
        provider.client = mock_client
        
        result = provider.transcribe(str(test_audio_file))
        
        assert result is None
```

### Integration Tests

For real API tests, create `tests/test_providers/test_your_provider_integration.py`:

```python
"""Integration tests for YourProvider with real API calls."""
import os
import pytest

from providers.your_provider import YourProvider


@pytest.mark.integration
@pytest.mark.slow
class TestYourProviderIntegration:
    """Integration tests using real API."""
    
    @pytest.fixture
    def api_token(self):
        """Get API token from environment."""
        token = os.getenv('YOUR_API_TOKEN')
        if not token:
            pytest.skip("YOUR_API_TOKEN not set - skipping integration test")
        return token
    
    def test_transcribe_with_real_api(self, api_token, test_audio_file):
        """Test transcription with real API call."""
        provider = YourProvider(api_token=api_token)
        result = provider.transcribe(str(test_audio_file))
        
        assert result is None or isinstance(result, str)
        if result:
            assert len(result.strip()) > 0
```

### Running Tests

```bash
# Run all provider tests
pytest tests/test_providers/

# Run unit tests (mocked, fast)
pytest tests/test_providers/test_your_provider.py

# Run integration tests (real API, requires token)
pytest tests/test_providers/test_your_provider_integration.py -m integration

# Run with coverage
pytest tests/test_providers/ --cov=providers --cov-report=term-missing
```

### Test Fixtures

The test suite provides helpful fixtures in `tests/conftest.py`:

- `test_audio_file` - Temporary WAV file for testing transcription
- `mock_api_token` - Mock API token for testing
- `mock_api_settings` - Mock API settings
- `replicate_provider` - Example provider instance (can be adapted for your provider)
- `temp_dir` - Temporary directory for test files

You can use these fixtures in your tests:

```python
def test_my_provider(test_audio_file, mock_api_token):
    """Test my provider with test audio file."""
    provider = MyProvider(api_token=mock_api_token)
    result = provider.transcribe(str(test_audio_file))
    # ... test implementation
```

## Best Practices

### 1. Error Handling

- **Always return `None` on failure** (don't raise exceptions)
- **Provide user-friendly error messages** via `print()`
- **Handle specific error types** (authentication, rate limits, network errors)
- **Validate file existence** before processing

```python
def transcribe(self, audio_file_path: str) -> Optional[str]:
    try:
        if not os.path.exists(audio_file_path):
            print(f"❌ Audio file not found: {audio_file_path}")
            return None
        
        # Your logic...
        
    except AuthenticationError:
        print("⚠️  Authentication failed. Please check your API token.")
        return None
    except RateLimitError:
        print("⚠️  Rate limit exceeded. Please wait and try again.")
        return None
    except Exception as e:
        print(f"❌ Transcription error: {e}")
        return None
```

### 2. Response Parsing

Handle different response formats your API might return:

```python
# Handle string response
if isinstance(response, str):
    return response.strip()

# Handle dict response
elif isinstance(response, dict):
    if 'text' in response:
        return str(response['text']).strip()
    # Try other keys...

# Handle list response
elif isinstance(response, list):
    return ' '.join(str(item) for item in response).strip()
```

### 3. File Handling

- Audio files are WAV format, 44100 Hz sample rate, mono channel
- Files are temporary and will be deleted after processing
- Don't modify the original file
- If you need to convert format, use temporary files

### 4. Logging

Use consistent emoji/logging style:

```python
print("📤 Uploading audio file...")  # Upload/processing
print("🔄 Transcribing audio...")   # Transcription in progress
print("✓ Transcription successful") # Success
print("⚠️  Warning message")         # Warning
print("❌ Error message")            # Error
```

### 5. Configuration

- Use `config.API_SETTINGS` for provider-agnostic settings
- Add provider-specific settings with clear prefixes
- Support environment variable overrides
- Provide sensible defaults

### 6. Resource Management

- Implement `cleanup()` if you open connections/files
- Don't leave temporary files behind
- Close API clients properly

## Troubleshooting

### Provider Not Found Error

**Error**: `Unknown transcription provider: 'your_provider'`

**Solution**: Make sure you:
1. Imported your provider in `providers/__init__.py`
2. Added it to the `create_provider()` function
3. Used the correct provider name (case-insensitive)

### API Token Not Found

**Error**: `YOUR_API_TOKEN not found`

**Solution**: 
1. Set environment variable: `export YOUR_API_TOKEN=your_token`
2. Or add to `.env` file: `YOUR_API_TOKEN=your_token`
3. Make sure `.env` is loaded (application does this automatically)

### Import Errors

**Error**: `ModuleNotFoundError: No module named 'your_api_client'`

**Solution**: Add your provider's dependencies to `requirements.txt` and install:
```bash
pip install -r requirements.txt
```

### Tests Failing

**Common issues**:
- Mock not set up correctly → Check your test fixtures
- API client not initialized → Verify `__init__()` logic
- File paths incorrect → Use `test_audio_file` fixture from `conftest.py`

## Provider Checklist

Before submitting your provider, ensure:

- [ ] Inherits from `TranscriptionProvider`
- [ ] Implements `transcribe()` method correctly
- [ ] Handles errors gracefully (returns `None`, doesn't raise)
- [ ] Provides user-friendly error messages
- [ ] Registered in `providers/__init__.py`
- [ ] Configuration added to `config.py`
- [ ] Dependencies added to `requirements.txt`
- [ ] Unit tests written and passing
- [ ] Integration tests written (optional, for real API)
- [ ] Documentation updated (if needed)
- [ ] Tested with actual audio files
- [ ] Cleanup method implemented (if needed)

## Getting Help

If you encounter issues:

1. Check existing providers (`providers/replicate.py`) for reference
2. Review test examples (`tests/test_providers/`) - comprehensive test suite with 30+ tests
3. Check the base class interface (`providers/base.py`)
4. Review this documentation
5. Run the test suite to verify your provider works correctly: `pytest tests/test_providers/`

## Contributing

When contributing a new provider:

1. Follow the code style of existing providers
2. Write comprehensive tests
3. Update this documentation if needed
4. Test with real API calls (if possible)
5. Document any provider-specific requirements

---

**Happy coding!** 🚀
