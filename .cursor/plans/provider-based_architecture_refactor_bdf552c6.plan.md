---
name: Provider-Based Architecture Refactor
overview: Refactor the application to support multiple transcription providers using object-oriented principles. Create an abstract provider base class and move Replicate-specific code into a concrete provider implementation, making it easy to add new providers in the future.
todos:
  - id: create-base-provider
    content: Create abstract base class TranscriptionProvider in providers/base.py using ABC pattern (similar to StatusPlugin)
    status: completed
  - id: create-replicate-provider
    content: Move Replicate-specific code into ReplicateProvider class in providers/replicate.py, implementing the base interface
    status: completed
  - id: create-provider-factory
    content: Create provider factory function in providers/__init__.py to instantiate providers based on config
    status: completed
  - id: update-config
    content: Update config.py to add PROVIDER setting and refactor API_SETTINGS to support multiple providers
    status: completed
  - id: refactor-voice-dictation-tool
    content: Refactor VoiceDictationTool in start.py to use provider abstraction, removing Replicate-specific methods
    status: in_progress
  - id: update-imports
    content: Update imports in start.py to use provider classes instead of direct Replicate SDK
    status: pending
  - id: test-provider-base
    content: Test the base provider class interface and ensure it works correctly
    status: pending
  - id: test-replicate-provider
    content: Test ReplicateProvider with actual audio file and real API calls - verify transcription works, upload succeeds, errors handled correctly (DO NOT skip validation or mock critical functionality)
    status: pending
  - id: test-integration
    content: Test full integration - record audio, transcribe with real API, verify end-to-end workflow works completely (no shortcuts or skipped steps)
    status: pending
  - id: verify-70-30-principle
    content: Verify code follows 70-30 principle - 70% reusable/common code, 30% provider-specific
    status: pending
---

# Provider-Based Architecture Refactor

## Overview

Refactor the application to use a provider-based architecture for transcription services, following the existing plugin pattern used for status indicators. This will use abstraction, inheritance, and polymorphism to make the system extensible.

## Architecture Design

The design follows these OOP principles:

- **Abstraction**: Abstract base class defines the provider interface
- **Inheritance**: Concrete providers inherit from the base class
- **Polymorphism**: VoiceDictationTool works with any provider implementation
- **Encapsulation**: Provider-specific logic is encapsulated in provider classes

## File Structure Changes

```
providers/
  __init__.py          # Provider exports and factory function
  base.py              # Abstract base class (TranscriptionProvider)
  replicate.py         # ReplicateProvider implementation
  # Future providers can be added here (e.g., openai.py, google.py)
```

## Implementation Plan

### 1. Create Provider Base Class (`providers/base.py`)

Create an abstract base class that defines the interface all providers must implement:

- `transcribe(audio_file_path: str) -> Optional[str]`: Main method to transcribe audio
- Optional initialization parameters as needed
- Abstract methods ensure all providers implement required functionality

This follows the same pattern as `plugins/base.py` (StatusPlugin).

### 2. Create ReplicateProvider (`providers/replicate.py`)

Move Replicate-specific code from `VoiceDictationTool` into a concrete provider:

- Inherit from `TranscriptionProvider`
- Implement `transcribe()` method
- Move `_upload_audio_to_replicate()` logic into the provider (can be private method)
- Move `_transcribe_audio()` logic into the provider's `transcribe()` method
- Encapsulate Replicate-specific configuration and API calls
- Handle Replicate-specific error messages and response parsing

### 3. Create Provider Factory (`providers/__init__.py`)

Create a factory function to instantiate providers based on configuration:

- `create_provider(provider_name: str, config: dict) -> TranscriptionProvider`
- Returns appropriate provider instance based on provider name
- Handles provider registration and instantiation
- Provides error handling for unknown providers

### 4. Update Configuration (`config.py`)

Add provider configuration:

- Add `PROVIDER` setting (default: 'replicate')
- Refactor `API_SETTINGS` to be provider-agnostic or provider-specific
- Consider renaming to `PROVIDER_SETTINGS` or keeping `API_SETTINGS` as generic config
- Support environment variable override: `TRANSCRIPTION_PROVIDER`

### 5. Refactor VoiceDictationTool (`start.py`)

Update the main class to use the provider abstraction:

- Remove `_upload_audio_to_replicate()` method
- Remove `_transcribe_audio()` method
- Add provider initialization in `__init__()`
- Update `_process_recording()` to call `provider.transcribe()` instead
- Remove Replicate SDK import (move to provider)
- Update error checking to be provider-agnostic

### 6. Update Requirements (if needed)

- Keep `replicate` in requirements.txt (it's used by ReplicateProvider)
- Other providers will add their own dependencies as needed

## Key Design Decisions

1. **Follow Existing Pattern**: Use the same ABC pattern as StatusPlugin for consistency
2. **Backward Compatibility**: Default provider will be 'replicate' to maintain existing behavior
3. **Configuration**: Use a single `PROVIDER` config setting with provider-specific settings in `API_SETTINGS`
4. **Error Handling**: Each provider handles its own errors; base class can define common error patterns
5. **Vocabulary Corrections**: Keep in `VoiceDictationTool` as it's application-level, not provider-specific

## Migration Path

1. Create provider abstraction and ReplicateProvider
2. Update VoiceDictationTool to use provider
3. Update configuration
4. Test that existing Replicate functionality still works
5. Future providers can be added by creating new provider classes

## Testing Strategy

Test the application incrementally as we build it:

1. **Unit Tests for Base Provider**: Test the abstract interface and ensure it enforces the contract
2. **Unit Tests for ReplicateProvider**: Test ReplicateProvider in isolation with actual audio files (not mocks that skip validation)
3. **Integration Tests**: Test the full workflow (recording → transcription → paste) with actual audio and real API calls (no shortcuts)
4. **Regression Tests**: Verify existing Replicate functionality still works after refactoring
5. **Provider Switching Tests**: Test that provider can be switched via configuration

### Testing Approach

- Test each component as it's created (test-driven development)
- Use actual audio files for integration testing - do not mock or skip actual transcription
- Verify error handling works correctly - test real error scenarios, don't skip error validation
- Test configuration changes (switching providers) - verify it actually uses the new provider
- Ensure backward compatibility is maintained - test with existing configuration

### Testing Principles

**Critical Requirements - DO NOT SKIP:**

- Tests must use actual audio files and verify transcription results are meaningful
- Tests must verify API calls are actually made (not mocked to skip network calls when testing real functionality)
- Tests must verify error handling with real error conditions
- Tests must verify configuration changes actually affect behavior
- Integration tests must test the complete workflow, not shortcuts
- Tests must validate that provider switching works correctly
- Tests must verify vocabulary corrections are applied
- Tests must verify file upload happens (for providers that require it)
- Tests must verify API tokens are validated and errors are handled correctly

**What Tests Should Verify:**

- Provider interface enforces implementation of required methods
- ReplicateProvider actually calls Replicate API and gets real results
- Audio file upload succeeds or fails appropriately
- Transcription returns valid text from actual API responses
- Error messages are meaningful and helpful
- Configuration changes affect which provider is used
- Vocabulary corrections are applied correctly
- File cleanup happens properly

## 70-30 Principle Compliance

Ensure the architecture follows the 70-30 principle:

- **70% Reusable/Common Code**: 
  - Base provider class and interface
  - VoiceDictationTool core logic (recording, file handling, vocabulary corrections)
  - Configuration management
  - Error handling patterns
  - Factory pattern implementation

- **30% Provider-Specific Code**:
  - ReplicateProvider implementation (API calls, upload logic, response parsing)
  - Provider-specific configuration handling
  - Provider-specific error messages

### Code Distribution Targets

- Base classes and interfaces: ~30% (reusable)
- VoiceDictationTool core: ~40% (reusable)
- Provider implementations: ~30% (provider-specific)

This ensures that adding a new provider only requires implementing the 30% provider-specific code, while leveraging the 70% common infrastructure.

## Benefits

- **Extensibility**: Easy to add new providers (OpenAI Whisper API, Google Speech-to-Text, Azure, etc.)
- **Maintainability**: Provider-specific code is isolated
- **Testability**: Providers can be tested independently
- **Consistency**: Follows existing plugin architecture pattern
- **Reusability**: 70% of code is reusable across all providers
- **Efficiency**: New providers require minimal code (only 30% provider-specific)