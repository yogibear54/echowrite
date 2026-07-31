"""Unit tests for LocalWhisperProvider with mocking."""
from unittest.mock import MagicMock, patch

import pytest

from providers.local_whisper import LocalWhisperProvider


def _make_segment(text: str):
    """Build a mock faster-whisper Segment with the given text."""
    seg = MagicMock()
    seg.text = text
    return seg


# Patch the WhisperModel symbol where it's *used* (the provider module) so
# the mock is in effect for every test, regardless of import order.
@pytest.fixture
def mock_whisper_model_cls():
    """A mock WhisperModel class — patches providers.local_whisper.WhisperModel."""
    with patch('providers.local_whisper.WhisperModel') as cls:
        instance = MagicMock()
        cls.return_value = instance
        yield cls, instance


@pytest.fixture
def mock_whisper_settings():
    """Settings dict for LocalWhisperProvider."""
    return {
        'model_size': 'base',
        'device': 'cpu',
        'compute_type': 'int8',
        'language': None,
        'beam_size': 5,
    }


@pytest.fixture
def local_whisper_provider(mock_whisper_settings, mock_whisper_model_cls):
    """Build a LocalWhisperProvider with the underlying WhisperModel mocked."""
    _cls, _instance = mock_whisper_model_cls
    return LocalWhisperProvider(api_settings=mock_whisper_settings)


@pytest.mark.unit
class TestLocalWhisperProviderInitialization:
    """Test LocalWhisperProvider initialization and eager model loading."""

    def test_loads_model_in_init(self, mock_whisper_settings, mock_whisper_model_cls):
        """Model is loaded eagerly at construction time."""
        cls, instance = mock_whisper_model_cls
        LocalWhisperProvider(api_settings=mock_whisper_settings)
        cls.assert_called_once_with('base', device='cpu', compute_type='int8')
        # cls.return_value is what got assigned to provider.model
        assert instance is cls.return_value

    def test_reads_settings_from_api_settings(self, mock_whisper_settings, mock_whisper_model_cls):
        """All five settings are read from api_settings."""
        cls, instance = mock_whisper_model_cls
        provider = LocalWhisperProvider(api_settings=mock_whisper_settings)
        assert provider.model_size == 'base'
        assert provider.device == 'cpu'
        assert provider.compute_type == 'int8'
        assert provider.language is None
        assert provider.beam_size == 5
        assert provider.model is instance

    def test_falls_back_to_config_when_no_settings(self, mock_whisper_model_cls):
        """With api_settings=None, uses config.WHISPER_SETTINGS."""
        cls, _ = mock_whisper_model_cls
        provider = LocalWhisperProvider(api_settings=None)
        import config
        assert provider.api_settings == config.WHISPER_SETTINGS
        assert provider.model_size == config.WHISPER_SETTINGS['model_size']

    def test_ignores_api_token(self, mock_whisper_settings, mock_whisper_model_cls):
        """api_token is accepted for interface compatibility but unused."""
        cls, instance = mock_whisper_model_cls
        provider = LocalWhisperProvider(api_token='unused-token', api_settings=mock_whisper_settings)
        cls.assert_called_once()
        assert provider.model is instance

    def test_uses_settings_defaults_for_missing_keys(self, mock_whisper_model_cls):
        """Missing keys fall back to documented defaults."""
        cls, _ = mock_whisper_model_cls
        provider = LocalWhisperProvider(api_settings={})
        assert provider.model_size == 'small'
        assert provider.device == 'cpu'
        assert provider.compute_type == 'int8'
        assert provider.language is None
        assert provider.beam_size == 5


@pytest.mark.unit
class TestLocalWhisperProviderTranscription:
    """Test LocalWhisperProvider.transcribe()."""

    @patch('os.path.exists')
    def test_transcribe_joins_segments(self, mock_exists, local_whisper_provider):
        """Segments are joined with single spaces and stripped."""
        mock_exists.return_value = True
        local_whisper_provider.model.transcribe.return_value = (
            iter([_make_segment('Hello'), _make_segment('world'), _make_segment('!')]),
            MagicMock(),
        )

        result = local_whisper_provider.transcribe('/tmp/audio.wav')

        assert result == 'Hello world !'

    @patch('os.path.exists')
    def test_transcribe_passes_correct_args(self, mock_exists, local_whisper_provider):
        """beam_size and language are forwarded to model.transcribe."""
        mock_exists.return_value = True
        local_whisper_provider.model.transcribe.return_value = (iter([]), MagicMock())

        local_whisper_provider.transcribe('/tmp/audio.wav')

        call_args = local_whisper_provider.model.transcribe.call_args
        assert call_args[0][0] == '/tmp/audio.wav'
        assert call_args[1]['beam_size'] == 5
        assert call_args[1]['language'] is None

    def test_transcribe_forwards_language_when_set(self, mock_whisper_settings, mock_whisper_model_cls):
        """A forced language setting is forwarded to model.transcribe."""
        mock_whisper_settings['language'] = 'en'
        cls, instance = mock_whisper_model_cls
        provider = LocalWhisperProvider(api_settings=mock_whisper_settings)
        instance.transcribe.return_value = (iter([]), MagicMock())

        with patch('os.path.exists', return_value=True):
            provider.transcribe('/tmp/audio.wav')

        assert instance.transcribe.call_args[1]['language'] == 'en'

    @patch('os.path.exists')
    def test_transcribe_returns_none_when_file_missing(self, mock_exists, local_whisper_provider):
        """Missing file -> print message and return None; model not called."""
        mock_exists.return_value = False

        result = local_whisper_provider.transcribe('/tmp/missing.wav')

        assert result is None
        local_whisper_provider.model.transcribe.assert_not_called()

    @patch('os.path.exists')
    def test_transcribe_returns_none_on_empty_result(self, mock_exists, local_whisper_provider):
        """Empty segment stream -> None."""
        mock_exists.return_value = True
        local_whisper_provider.model.transcribe.return_value = (iter([]), MagicMock())

        result = local_whisper_provider.transcribe('/tmp/silence.wav')

        assert result is None

    @patch('os.path.exists')
    def test_transcribe_returns_none_on_whitespace_only(self, mock_exists, local_whisper_provider):
        """Whitespace-only output -> None (after strip)."""
        mock_exists.return_value = True
        local_whisper_provider.model.transcribe.return_value = (
            iter([_make_segment('   '), _make_segment('\t')]),
            MagicMock(),
        )

        result = local_whisper_provider.transcribe('/tmp/audio.wav')

        assert result is None

    @patch('os.path.exists')
    def test_transcribe_returns_none_on_exception(self, mock_exists, local_whisper_provider):
        """Model exception is caught and returns None (no re-raise)."""
        mock_exists.return_value = True
        local_whisper_provider.model.transcribe.side_effect = RuntimeError("boom")

        result = local_whisper_provider.transcribe('/tmp/audio.wav')

        assert result is None

    @patch('os.path.exists')
    def test_transcribe_strips_segment_text(self, mock_exists, local_whisper_provider):
        """Each segment's text is individually stripped; joined with single space."""
        mock_exists.return_value = True
        local_whisper_provider.model.transcribe.return_value = (
            iter([_make_segment('  hello  '), _make_segment(' world ')]),
            MagicMock(),
        )

        result = local_whisper_provider.transcribe('/tmp/audio.wav')

        assert result == 'hello world'


@pytest.mark.unit
class TestLocalWhisperProviderCleanup:
    """Test LocalWhisperProvider.cleanup()."""

    def test_cleanup_is_noop(self, local_whisper_provider):
        """cleanup() runs without error and does not touch the model."""
        model_before = local_whisper_provider.model
        local_whisper_provider.cleanup()
        assert local_whisper_provider.model is model_before


@pytest.mark.unit
class TestCreateProviderFactory:
    """Test that providers/__init__.py registers the local_whisper provider."""

    def test_create_local_whisper_provider(self, mock_whisper_settings, mock_whisper_model_cls):
        """Factory returns a LocalWhisperProvider when name='local_whisper'."""
        from providers import create_provider
        with patch('config.PROVIDER', 'local_whisper'), \
             patch('config.WHISPER_SETTINGS', mock_whisper_settings):
            provider = create_provider(provider_name='local_whisper')
        assert isinstance(provider, LocalWhisperProvider)

    def test_create_replicate_provider_still_works(self):
        """The existing replicate factory path is unchanged."""
        from providers import create_provider
        from providers.replicate import ReplicateProvider
        with patch.dict('os.environ', {'REPLICATE_API_TOKEN': 'test_token'}):
            provider = create_provider(provider_name='replicate', api_token='test_token')
        assert isinstance(provider, ReplicateProvider)

    def test_unknown_provider_raises_with_updated_message(self):
        """Factory error message lists the new provider."""
        from providers import create_provider
        with pytest.raises(ValueError, match='local_whisper'):
            create_provider(provider_name='not_a_real_provider')