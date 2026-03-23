# 🎙️ EchoWrite: The Voice Dictation Tool

A local Python application that captures voice input via global hotkeys, transcribes speech using Replicate's incredibly-fast-whisper model, and automatically pastes the result wherever your cursor is positioned.

**Platform note:** This project is currently developed for and tested on Linux only. macOS and Windows are not supported at this time.

## Features

- **Global Hotkey Support**: Press and hold Ctrl+Alt to start recording, release to stop (works even when the app isn't focused)
- **Recording Cancellation**: Press Escape during recording to cancel and discard without processing
- **Device Selection**: Interactive audio device selection at startup
- **Fast Transcription**: Uses Replicate's incredibly-fast-whisper model (transcribes 150 minutes in ~100 seconds)
- **Automatic Pasting**: Transcribed text is automatically copied to clipboard and pasted at cursor position
- **Configurable Paste Modes**: Runtime paste key switching with F1/F2/F3 shortcuts (Ctrl+V, Ctrl+Shift+V, Ctrl+Insert)
- **Recording History**: All transcriptions are saved to `recordings.json` with timestamps
- **Custom Vocabulary**: Supports custom vocabulary corrections for better recognition of technical terms
- **Environment Configuration**: Configure settings via `.env` file
- **Maximum Recording Duration**: Configurable limit to prevent excessive recordings
- **Status Indicators**: Visual indicators for "recording..." and "processing..." modes
- **Plugin System**: Extensible plugin architecture for custom status displays (e.g., i3 status bar integration)
- **Provider Architecture**: Extensible provider system supporting multiple transcription services (currently Replicate)
- **Comprehensive Testing**: Full test suite with 60+ tests covering all major functionality

## Requirements

### System Requirements

- Python 3.8 or higher
- Linux (tested)
- Microphone access
- Internet connection (for Replicate API)

### System Dependencies

#### Linux (Debian/Ubuntu)

```bash
sudo apt-get update
sudo apt-get install portaudio19-dev python3-dev
sudo apt-get install xclip  # Required for clipboard functionality
```

#### Other platforms

macOS and Windows have not been tested and are not supported in the current release.

### Python Dependencies

All Python dependencies are listed in `requirements.txt`:

```
keyboard==0.13.5          # Global hotkey detection
sounddevice==0.4.7        # Audio recording
scipy==1.11.4             # Audio file processing
replicate==0.34.0          # Replicate API client
pyperclip==1.8.2          # Clipboard operations
pyautogui==0.9.54         # Auto-paste functionality
python-dotenv==1.0.0      # Environment variable management
requests>=2.31.0          # HTTP requests for file upload

# Testing dependencies
pytest>=7.4.0             # Testing framework
pytest-mock>=3.11.0       # Enhanced mocking capabilities
pytest-cov>=4.1.0         # Code coverage reporting
pytest-timeout>=2.1.0     # Prevent hanging tests
responses>=0.23.0         # Mock HTTP requests
```

## Installation

1. **Clone or download this repository**

2. **Create a virtual environment** (recommended):

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install Python dependencies**:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

4. **Create `.env` file**:

```bash
cp .env.example .env
```

5. **Configure your `.env` file**:

Edit `.env` and add your Replicate API token:

```env
REPLICATE_API_TOKEN=r8_your_api_token_here
```

Get your API token from: https://replicate.com/account/api-tokens

### Optional Environment Variables

You can customize the application behavior by setting these variables in `.env`:

```env
# Maximum recording duration in minutes (default: 5.0, range: 0.1-60.0)
MAX_RECORDING_MINUTES=5

# Audio sample rate in Hz (default: 44100, range: 8000-48000)
SAMPLE_RATE=44100

# Replicate model name with version tag
# Default: vaibhavs10/incredibly-fast-whisper:3ab86df6c8f54c11309d4d1f930ac292bad43ace52d10c80d87eb258b3c9f79c
REPLICATE_MODEL=vaibhavs10/incredibly-fast-whisper:3ab86df6c8f54c11309d4d1f930ac292bad43ace52d10c80d87eb258b3c9f79c

# Minimum recording duration in seconds (default: 1.0, min: 0.1)
MIN_RECORDING_SECONDS=1.0

# Status indicator plugins (comma-separated, default: i3status)
# Available: i3status
STATUS_PLUGINS=i3status

# i3 status bar plugin configuration
# Path to status file that i3bar will read (default: /tmp/voice2text_status)
I3_STATUS_FILE=/tmp/voice2text_status

# Paste key configuration (default: ctrl,v)
# Format: comma-separated keys, e.g., 'ctrl,v' or 'ctrl,shift,v' or 'ctrl,insert'
# Available modes: ctrl+v (standard), ctrl+shift+v (terminal), ctrl+insert (alternative)
PASTE_KEYS=ctrl,v
```

## Usage

1. **Activate the virtual environment** (if not already activated):

```bash
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. **Run the application**:

**On Linux** (requires `sudo` for global keyboard hotkey access):

Option 1 - Use the helper script (recommended):
```bash
./run.sh
```
**Note:** The `run.sh` script requires `sudo` privileges. It will prompt you for your password to run the application with root permissions, which are necessary for global keyboard hotkey detection on Linux.

Option 2 - Use sudo with venv Python directly:
```bash
sudo venv/bin/python start.py
```

Option 3 - Activate venv first, then use sudo:
```bash
source venv/bin/activate
sudo $(which python) start.py
```

**Note**: On Linux, the `keyboard` library requires root permissions for global hotkey detection. Use `sudo` with the full path to your virtual environment's Python interpreter (`venv/bin/python`) to ensure it uses the correct Python and installed packages. Other operating systems are not currently supported or tested.

3. **Select your audio input device** when prompted (or press Enter for default)

4. **Start dictating**:
   - Position your cursor in the desired text field (any application)
   - Press and **hold** Ctrl+Alt to start recording
   - Speak your text
   - **Release** Ctrl+Alt to stop recording and process
   - Press **Escape** during recording to cancel and discard without processing
   - The transcribed text will automatically appear at your cursor position

5. **Change paste mode for different applications**:
   - Default paste mode: **Ctrl+V** (works in most applications)
   - Press **F1** to switch to next paste mode (e.g., **Ctrl+Shift+V** for terminals)
   - Press **F2** to cycle backward through modes
   - Press **F3** to display current paste mode

6. **Exit the application**: Press Ctrl+C in the terminal

## Architecture & Implementation

### Application Flow

```
Application Startup
    ↓
Query Available Audio Input Devices
    ↓
Interactive Device Selection Prompt
    ↓
Store Selected Device ID
    ↓
Initialize Global Hotkey Listener (keyboard.hook)
    ↓
[Main Loop - Waiting for Hotkey]
    ↓
User Presses Ctrl+Alt (both down)
    ↓
Start Audio Recording Thread (Using Selected Device)
    ↓
    ├─→ User Releases Ctrl+Alt (key-up event)
    │   ↓
    │   Stop Recording & Save to temp WAV file
    │
    └─→ User Presses Escape (cancel)
        ↓
        Stop Recording & Discard Audio Data
        ↓
        Return to Main Loop (no processing)
    ↓
Validate Recording Duration (min 1 second)
    ↓
Upload Audio File to Replicate API (/v1/files)
    ↓
Get Audio URL from Replicate
    ↓
Send Audio URL to Replicate Model API
    ↓
Receive Transcribed Text
    ↓
Apply Vocabulary Corrections
    ↓
Copy to Clipboard & Auto-Paste (using current paste mode - Ctrl+V by default, configurable via F1/F2/F3)
    ↓
Save Transcription to recordings.json
    ↓
Clean Up Temp WAV File
    ↓
[Return to Main Loop]
```

### Key Components

#### 1. Global Hotkey Detection (`start.py`)

- Uses `keyboard.hook()` for global key event monitoring
- Tracks Ctrl and Alt key states independently
- Starts recording when both keys are pressed down
- Stops recording when either key is released
- Supports cancellation via Escape key during recording
- Supports paste mode switching via F1 (next), F2 (previous), F3 (show current)
- Prevents duplicate recordings from rapid key presses

**Implementation Details:**
- Key events are normalized (handles 'ctrl', 'left ctrl', 'right ctrl', etc.)
- Uses threading to avoid blocking the main loop
- Recording thread runs as daemon thread
- Cancellation: Pressing Escape during recording sets `is_cancelled` flag, stops recording, clears audio data, and skips processing
- Cancellation state is reset when keys are released or a new recording starts

#### 2. Audio Recording (`start.py` - `_record_audio()`)

- Uses `sounddevice.InputStream()` for real-time audio capture
- Records in 100ms chunks to allow responsive stopping
- Monitors maximum recording duration and auto-stops if limit reached
- Saves audio data as numpy array (float32 format)

**Technical Details:**
- Sample rate: 44100 Hz (configurable)
- Channels: 1 (mono)
- Data type: float32 (more widely supported than float64)
- Recording stops when `is_recording` flag is set to False

#### 3. Audio File Processing (`start.py` - `_save_wav_file()`)

- Converts float32 audio data to int16 format for WAV
- Clips audio values to [-1.0, 1.0] range before conversion
- Uses `scipy.io.wavfile.write()` for reliable WAV file creation
- Saves to `temp/` directory with timestamped filename

#### 4. Transcription Provider System

The application uses a provider-based architecture for transcription services:

- **Provider Abstraction**: Abstract base class (`providers/base.py`) defines the interface
- **Replicate Provider**: Concrete implementation (`providers/replicate.py`) for Replicate API
- **Extensibility**: Easy to add new providers (OpenAI, Google, Azure, etc.)
- **Factory Pattern**: Provider factory (`providers/__init__.py`) instantiates providers based on config

#### 5. Replicate API Integration (via ReplicateProvider)

**File Upload (`_upload_audio_to_replicate()`):**
- Uploads WAV file to Replicate's file storage API
- Endpoint: `POST https://api.replicate.com/v1/files`
- Field name: `content` (multipart/form-data)
- Returns: URL to uploaded file (`urls.get`)

**Transcription (`transcribe()`):**
- Uses Replicate Python SDK (`replicate.run()`)
- Model: `vaibhavs10/incredibly-fast-whisper:3ab86df6c8f54c11309d4d1f930ac292bad43ace52d10c80d87eb258b3c9f79c`
- Input parameters:
  - `audio`: URL from file upload
  - `task`: 'transcribe' or 'translate'
  - `language`: Language code or 'None' for auto-detection
  - `timestamp`: 'chunk' or 'word'
  - `batch_size`: 64 (optimized for speed)
  - `diarise_audio`: False (speaker diarization disabled)
- Handles various response formats (string, dict with 'text' key, list, etc.)
- Returns transcribed text as a string

**Important Notes:**
- The model requires a version tag (specific commit hash)
- Audio must be uploaded first to get a URL (cannot pass file directly)
- The API returns transcribed text in various formats (handled by provider)

#### 6. Vocabulary Correction (`start.py` - `_apply_vocabulary_corrections()`)

- Uses regex for case-insensitive pattern matching
- Matches common mispronunciations against `CUSTOM_VOCABULARY`
- Replaces matches with canonical terms
- Applied after transcription, before pasting

#### 7. Text Insertion (`start.py` - `_paste_text()`)

- Copies text to clipboard using `pyperclip.copy()`
- Triggers paste using dynamically configured keys (default: Ctrl+V)
- Supports multiple paste modes: Ctrl+V, Ctrl+Shift+V, Ctrl+Insert
- Runtime mode switching via F1 (next), F2 (previous), F3 (show current)
- Includes small delay to ensure clipboard is ready
- Falls back gracefully if paste fails (text still saved to recordings.json)
- Default paste mode configurable in `config.py` via `PASTE_KEYS`

#### 8. Persistence (`start.py` - `_save_transcription()`)

- Saves all transcriptions to `recordings.json`
- Format: JSON array of objects with `timestamp` and `transcription`
- Timestamps in ISO format
- File is created automatically if it doesn't exist

### Configuration System (`config.py`)

The configuration system supports environment variable overrides with validation:

- **Integer values**: Validated with min/max ranges
- **Float values**: Validated with min/max ranges
- **String values**: Validated for non-empty
- **Defaults**: Fallback to hardcoded defaults if env vars missing/invalid
- **Warnings**: Prints warnings for invalid values but continues with defaults

**Configuration Loading Order:**
1. Check environment variable (from `.env` file)
2. Validate value (type, range, etc.)
3. Use default if validation fails
4. Print warning for invalid values

## Configuration

### Custom Vocabulary

Edit `config.py` to add custom vocabulary corrections:

```python
CUSTOM_VOCABULARY = {
    'n8n': ['n8n', 'n 8 n', 'n eight n', 'nateon', 'AN10', 'N810', 'N8N', 'A10'],
    'Retell': ['Retell', 'retell', 're-tell', 'retail', 'retale', 're tell'],
    # Add your own terms here
}
```

The application will automatically correct common mispronunciations to your specified canonical terms using case-insensitive regex matching.

### Paste Key Configuration

The application supports three different paste key combinations that can be changed at runtime or set as default in `config.py`:

**Available Paste Modes:**
1. **Ctrl+V** (default) - Standard paste, works in most applications
2. **Ctrl+Shift+V** - Terminal paste, required by some terminal emulators
3. **Ctrl+Insert** - Alternative paste, works in some applications

**Runtime Mode Switching:**
- Press **F1** to cycle to the next paste mode
- Press **F2** to cycle to the previous paste mode
- Press **F3** to display the current paste mode
- No restart needed - changes take effect immediately

**Default Configuration:**
You can set the default paste mode in `config.py`:

```python
# Format: comma-separated keys
PASTE_KEYS = 'ctrl,v'          # Default: Ctrl+V (standard paste)
# PASTE_KEYS = 'ctrl,shift,v'  # Default: Ctrl+Shift+V (terminal paste)
# PASTE_KEYS = 'ctrl,insert'    # Default: Ctrl+Insert (alternative paste)
```

Or override via environment variable in `.env`:

```env
PASTE_KEYS=ctrl,shift,v
```

**Use Case Example:**
- Use **Ctrl+V** when working in web browsers, text editors, office applications
- Press **F1** to switch to **Ctrl+Shift+V** when working in terminal applications
- Press **F2** to return to **Ctrl+V** when switching back to standard applications
- The application confirms mode changes with console output like "🔄 Paste mode changed to: CTRL+SHIFT+V"


### Recording Settings

You can adjust recording settings in `config.py`:

```python
RECORDING_SETTINGS = {
    'sample_rate': 44100,  # Audio sample rate (8000-48000 Hz)
    'channels': 1,          # Mono audio
    'dtype': 'float32'      # Data type (float32 is more widely supported)
}
```

Or override via environment variables (see Optional Environment Variables above).

### Replicate Model Settings

Model parameters can be adjusted in `config.py`:

```python
API_SETTINGS = {
    'model': 'vaibhavs10/incredibly-fast-whisper:3ab86df6c8f54c11309d4d1f930ac292bad43ace52d10c80d87eb258b3c9f79c',
    'task': 'transcribe',      # 'transcribe' or 'translate'
    'language': 'None',         # Language code or 'None' for auto-detection
    'timestamp': 'chunk',       # 'chunk' or 'word'
    'batch_size': 64,           # Batch size for processing
    'diarise_audio': False,     # Speaker diarization (requires hf_token if True)
}
```

**Note**: The model version tag is required. The default version is the latest stable version as of implementation.

### Status Indicators & Plugin System

The application includes a plugin-based status indicator system that shows the current state ("recording...", "processing...", or idle). This allows you to see the application status in external displays like i3 status bar.

#### Built-in Plugins

**i3 Status Bar Plugin** (`i3status`):
- Displays voice2text status in your i3bar
- Shows recording and processing states with visual indicators
- Uses a wrapper script to inject status into i3status output

**📖 For complete setup instructions, configuration, and troubleshooting, see [plugins/i3status/README.md](plugins/i3status/README.md)**

#### Creating Custom Plugins

**📖 For detailed documentation on creating status plugins, see [NEW_STATUS_PLUGINS.md](NEW_STATUS_PLUGINS.md)**

You can create custom status indicator plugins by:

1. **Create a new plugin file** in the `plugins/` directory (e.g., `plugins/myplugin.py`)
2. **Inherit from `StatusPlugin`** base class:
   ```python
   from plugins.base import StatusPlugin
   from status_manager import Status

   class MyCustomPlugin(StatusPlugin):
       def __init__(self):
           # Initialize your plugin
           pass
       
       def update_status(self, status: Status):
           # Update your indicator based on status
           if status == Status.RECORDING:
               # Show recording indicator
               pass
           elif status == Status.PROCESSING:
               # Show processing indicator
               pass
           else:  # IDLE
               # Hide or reset indicator
               pass
       
       def cleanup(self):
           # Clean up resources on shutdown
           pass
   ```

3. **Register your plugin** in `start.py`:
   ```python
   if 'myplugin' in config.STATUS_PLUGINS:
       from plugins.myplugin import MyCustomPlugin
       my_plugin = MyCustomPlugin()
       self.status_manager.register_plugin(my_plugin)
   ```

4. **Enable it** in your `.env` file:
   ```env
   STATUS_PLUGINS=i3status,myplugin
   ```

The plugin system is designed to be extensible - you can create plugins for:
- Desktop notifications (libnotify)
- System tray icons
- LED indicators
- Web dashboards
- Any other display mechanism you prefer

**See [NEW_STATUS_PLUGINS.md](NEW_STATUS_PLUGINS.md) for a complete guide with examples, best practices, and troubleshooting.**

## Project Structure

```
voice2text/
├── start.py              # Main application script (VoiceDictationTool class)
├── config.py             # Configuration settings with env variable support
├── status_manager.py      # Status manager for tracking and broadcasting application state
├── providers/            # Transcription provider implementations
│   ├── __init__.py       # Provider factory
│   ├── base.py           # Abstract base provider class
│   └── replicate.py      # Replicate provider implementation
├── plugins/              # Status indicator plugins directory
│   ├── __init__.py       # Plugin package initialization
│   ├── base.py           # Base class for status indicator plugins
│   └── i3status/         # i3 status bar plugin
│       ├── __init__.py
│       └── README.md
├── tests/                # Test suite
│   ├── conftest.py       # Shared test fixtures
│   ├── test_providers/    # Provider tests
│   ├── test_audio/        # Audio recording tests
│   ├── test_integration/ # Integration tests
│   └── fixtures/         # Test data and audio files
├── requirements.txt      # Python dependencies
├── pytest.ini           # Pytest configuration
├── .env.example          # Example environment variables file
├── .env                  # Your environment variables (create from .env.example)
├── .gitignore            # Git ignore rules
├── run.sh                # Helper script for Linux (requires sudo for global hotkeys)
├── run_tests.sh          # Helper script to run pytest with virtual environment
├── recordings.json       # Transcription history (created automatically)
├── temp/                 # Temporary audio files directory
├── venv/                 # Virtual environment (created during setup)
├── NEW_PROVIDERS.md      # Developer guide for creating new transcription providers
├── NEW_STATUS_PLUGINS.md # Developer guide for creating new status plugins
└── README.md             # This file
```

### File Descriptions

- **`start.py`**: Main application containing `VoiceDictationTool` class with all core functionality
- **`config.py`**: Configuration management with environment variable support and validation
- **`status_manager.py`**: Status manager that tracks application state and notifies registered plugins
- **`plugins/`**: Directory containing status indicator plugins
  - **`base.py`**: Base class (`StatusPlugin`) that all plugins must inherit from
  - **`i3status/`**: Plugin for i3 status bar integration
- **`NEW_PROVIDERS.md`**: Comprehensive developer guide for implementing new transcription providers
- **`NEW_STATUS_PLUGINS.md`**: Comprehensive developer guide for implementing new status indicator plugins
- **`run.sh`**: Helper script for Linux that automatically uses sudo with the correct Python path. **Requires sudo privileges** - it will prompt for your password to enable global keyboard hotkey detection.
- **`run_tests.sh`**: Helper script that automatically runs pytest using the virtual environment's Python interpreter. This ensures tests use the correct Python environment and dependencies without requiring manual activation of the venv.
- **`recordings.json`**: JSON file storing all transcription history with timestamps
- **`temp/`**: Directory for temporary WAV files (automatically cleaned up after processing)

## Troubleshooting

### "PortAudio library not found" Error

**Linux**: Install PortAudio:
```bash
sudo apt-get install portaudio19-dev
```

**macOS**: Usually works out of the box. If not, try:
```bash
brew install portaudio
```

### "Permission denied" for Keyboard Access (Linux)

The `keyboard` library requires root permissions on Linux for global hotkey detection. Use the helper script or run with sudo:

```bash
./run.sh
# or
sudo venv/bin/python start.py
```

**Important**: Always use the full path to the venv Python (`venv/bin/python`) when using sudo, as sudo runs with root's environment which doesn't have your virtual environment activated.

### "Invalid input sample format" Error

This error occurs when the audio device doesn't support the specified data type. The application uses `float32` by default, which is widely supported. If you encounter this:

1. Check that your audio device is properly connected
2. Try selecting a different audio device at startup
3. Verify PortAudio is installed correctly

### "No audio input devices found"

- Check that your microphone is connected and enabled
- Verify microphone permissions in your system settings
- On Linux, you may need to install audio drivers
- Try running `python -c "import sounddevice as sd; print(sd.query_devices())"` to list devices

### "REPLICATE_API_TOKEN not found"

- Make sure you've created a `.env` file from `.env.example`
- Verify your API token is set correctly in `.env`
- Check that the token starts with `r8_`
- Ensure `python-dotenv` is installed and `load_dotenv()` is called

### "400 Client Error: Bad Request" when uploading audio

This indicates an issue with the file upload format. The application uses:
- Field name: `content` (not `file`)
- Content-Type: `application/octet-stream`
- Multipart form-data format

If this error persists:
1. Check that the audio file was created successfully
2. Verify the file is a valid WAV file
3. Check your API token is valid
4. Review the error message for specific details

### "404 - Model not found" Error

- Verify the model name includes the version tag
- Check that the model exists on Replicate: https://replicate.com/vaibhavs10/incredibly-fast-whisper
- Ensure your API token has access to the model
- Try updating the model version in `config.py` if a newer version is available

### Clipboard/Paste Not Working (Linux)

Install `xclip` or `xsel`:

```bash
sudo apt-get install xclip
# or
sudo apt-get install xsel
```

### Recording Too Short Error

Recordings must be at least 1 second long (configurable via `MIN_RECORDING_SECONDS` in `.env`). Make sure you're holding the hotkeys long enough while speaking.

### Maximum Recording Duration Reached

The default maximum is 5 minutes. Adjust `MAX_RECORDING_MINUTES` in your `.env` file if you need longer recordings. The recording will automatically stop when the limit is reached.

### "Command not found" when using sudo

When using `sudo`, it runs with root's environment which doesn't have your virtual environment activated. Always use:

```bash
sudo venv/bin/python start.py
```

Or use the helper script (requires sudo):
```bash
./run.sh
```
**Note:** The `run.sh` script will prompt for your sudo password to enable global keyboard hotkey detection.

### Status Indicator Not Showing in i3 Bar

**📖 For detailed troubleshooting steps, see [plugins/i3status/README.md](plugins/i3status/README.md#troubleshooting)**

Common issues include:
- Plugin not enabled in `.env` file
- i3bar not configured to use the wrapper script
- Status file permissions or path issues
- JSON parsing errors in the wrapper script

## How It Works (Detailed)

### 1. Hotkey Detection

- Uses `keyboard.hook()` to register a global keyboard event handler
- Monitors all key press and release events
- Tracks state of Ctrl and Alt keys independently
- When both keys are pressed down simultaneously, starts recording
- When either key is released, stops recording and processes
- Pressing Escape during recording cancels and discards the recording without processing

### 2. Audio Recording

- Creates a separate thread for audio recording to avoid blocking
- Uses `sounddevice.InputStream()` with the selected device
- Records audio in 100ms chunks for responsive stopping
- Continuously checks if `is_recording` flag is False to stop
- Also checks maximum duration limit during recording
- Concatenates all chunks into a single numpy array

### 3. File Processing

- Audio data (float32 numpy array) is converted to int16 format
- Values are clipped to [-1.0, 1.0] range to prevent overflow
- Saved as WAV file using `scipy.io.wavfile.write()`
- File is saved with timestamp in filename for uniqueness

### 4. Transcription Provider Workflow

The application uses a provider-based architecture for transcription:

**Step 1: Provider Initialization**
- Provider factory creates appropriate provider based on `config.PROVIDER`
- Default provider is 'replicate' (ReplicateProvider)
- Provider is initialized with API token and settings

**Step 2: File Upload (ReplicateProvider)**
- Opens the WAV file in binary mode
- Creates multipart form-data with field name `content`
- POSTs to `https://api.replicate.com/v1/files`
- Includes Authorization header with API token
- Receives JSON response with file URL

**Step 3: Transcription (ReplicateProvider)**
- Uses the file URL from upload step
- Calls `replicate.run()` with model name and parameters
- Model processes audio and returns transcribed text
- Handles various response formats (string, dict, list)
- Returns transcribed text as a string

### 5. Post-Processing

- Vocabulary corrections are applied using regex
- Text is copied to system clipboard
- Auto-paste is triggered using dynamically configured keys (default: Ctrl+V, changeable via F1/F2/F3)
- Transcription is saved to JSON file with timestamp
- Temporary WAV file is deleted

## Technical Implementation Notes

### Audio Format

- **Input**: Float32 numpy array from sounddevice
- **Storage**: Int16 WAV file (standard audio format)
- **Conversion**: `(float32 * 32767).astype(np.int16)`
- **Clipping**: Applied before conversion to prevent overflow

### Threading Model

- **Main thread**: Runs hotkey listener and keeps application alive
- **Recording thread**: Daemon thread that captures audio
- **Synchronization**: Uses `is_recording` flag and `threading.Thread.join()`

### Error Handling

- All operations wrapped in try-except blocks
- User-friendly error messages with actionable guidance
- Graceful degradation (e.g., paste failure doesn't stop persistence)
- Detailed error information for debugging

### File Management

- Temporary files stored in `temp/` directory
- Files named with timestamp for uniqueness
- Automatic cleanup after successful processing
- Error handling for file operations

## Limitations

- Requires active internet connection for transcription
- On Linux, requires `sudo` for global hotkey detection
- Maximum recording duration is configurable but recommended to keep under 5 minutes for optimal performance
- Transcription accuracy depends on audio quality and clarity of speech
- Replicate API has rate limits (check your account limits)
- Audio files are temporarily uploaded to Replicate (privacy consideration)

## Future Development Ideas

- [ ] Support for multiple hotkey combinations
- [ ] Local transcription option (using local Whisper model)
- [ ] Real-time transcription during recording
- [ ] Custom hotkey configuration via .env
- [ ] Audio quality settings (bitrate, format options)
- [ ] Batch processing of multiple recordings
- [ ] Export transcriptions to different formats (txt, docx, etc.)
- [ ] GUI interface option
- [ ] Support for speaker diarization
- [ ] Language detection and auto-selection
- [ ] Recording history search and filtering
- [ ] Audio playback of recordings
- [ ] Integration with cloud storage for transcriptions

## Development Notes

### Key Design Decisions

1. **Float32 over Float64**: Changed from float64 to float32 for better device compatibility
2. **File Upload Required**: Replicate API requires uploading files first to get URLs (cannot pass files directly)
3. **Version Tag Required**: Model name must include version tag for API compatibility
4. **Threading for Recording**: Separate thread allows responsive stopping without blocking
5. **Environment Variables**: All configurable values support .env overrides with validation
6. **Plugin Architecture**: Status indicators use a plugin system for extensibility, allowing users to create custom displays
7. **Provider Architecture**: Transcription providers use abstract base classes for extensibility, following OOP principles (abstraction, inheritance, polymorphism)
8. **Comprehensive Testing**: Full test suite with unit tests, integration tests, and proper mocking for external dependencies

### Known Issues & Workarounds

1. **Linux Sudo Requirement**: Worked around by using full path to venv Python
2. **Audio Format Compatibility**: Resolved by using float32 instead of float64
3. **File Upload Format**: Required using 'content' field name, not 'file'
4. **Model Version**: Must include specific commit hash version tag

## Testing

The project includes a comprehensive test suite using pytest. All tests use proper mocking for external dependencies (APIs, system calls) while ensuring real functionality is validated.

### Running Tests

First, ensure test dependencies are installed:

```bash
source venv/bin/activate
pip install -r requirements.txt
```

**Recommended**: Use the helper script to automatically use the virtual environment's pytest:

```bash
./run_tests.sh
```

Alternatively, you can activate the virtual environment and run pytest directly:

```bash
source venv/bin/activate
pytest
```

#### Run All Tests

```bash
./run_tests.sh
```

Or with activated venv:
```bash
pytest
```

#### Run Only Unit Tests (Fast, No Network Required)

```bash
./run_tests.sh -m "not integration"
```

Or with activated venv:
```bash
pytest -m "not integration"
```

#### Run Integration Tests (Requires Network/API Token)

```bash
./run_tests.sh -m integration
```

Or with activated venv:
```bash
pytest -m integration
```

#### Run Tests with Coverage Report

```bash
./run_tests.sh --cov=providers --cov=start --cov=status_manager --cov=plugins --cov-report=html
```

Or with activated venv:
```bash
pytest --cov=providers --cov=start --cov=status_manager --cov=plugins --cov-report=html
```

This generates an HTML coverage report in `htmlcov/index.html`.

#### Run Specific Test Files

```bash
# Test providers only
./run_tests.sh tests/test_providers/

# Test audio functionality only
./run_tests.sh tests/test_audio/

# Test status plugins only
./run_tests.sh tests/test_plugins/

# Test integration/workflow only
./run_tests.sh tests/test_integration/

# Run a specific test file
./run_tests.sh tests/test_providers/test_replicate.py
```

Or with activated venv, use `pytest` instead of `./run_tests.sh`.

#### Verbose Output

```bash
./run_tests.sh -v
```

Or with activated venv:
```bash
pytest -v
```

### Test Structure

The test suite is organized into the following directories:

```
tests/
  ├── conftest.py              # Shared fixtures and test utilities
  ├── test_providers/
  │   ├── test_base.py         # Base provider interface tests
  │   ├── test_replicate.py    # ReplicateProvider unit tests (mocked)
  │   └── test_replicate_integration.py  # Real API tests (optional)
  ├── test_audio/
  │   ├── test_recording.py    # Audio recording logic tests
  │   └── test_file_handling.py # WAV file save/load tests
  ├── test_plugins/
  │   ├── test_status_manager.py  # StatusManager tests
  │   ├── test_base.py            # StatusPlugin base class tests
  │   └── test_i3status.py        # I3StatusPlugin implementation tests
  ├── test_integration/
  │   └── test_full_workflow.py # End-to-end workflow tests
  └── fixtures/
      └── test_audio.wav       # Test audio file for integration tests
```

### Test Coverage

The test suite covers:

- **Provider Abstraction**: Base class interface, ReplicateProvider implementation
- **Audio Recording**: Recording logic, chunk handling, duration limits
- **File Operations**: WAV saving, format conversion, cleanup
- **Paste Functionality**: Clipboard operations, paste triggering
- **Vocabulary Corrections**: Text correction logic
- **Status Plugin System**: StatusManager, StatusPlugin base class, I3StatusPlugin implementation
- **Error Handling**: Network errors, API errors, file errors, plugin errors
- **Integration**: End-to-end workflow validation

### Test Types

#### Unit Tests

Fast tests that use mocking for external dependencies:
- Provider initialization and configuration
- API request/response handling (mocked)
- Audio data processing
- File I/O operations
- Vocabulary corrections
- Status manager and plugin system

Run with: `pytest -m "not integration"`

#### Integration Tests

Tests that verify the complete workflow:
- Full recording → transcription → paste flow
- Provider integration with VoiceDictationTool
- Error handling in real scenarios
- File cleanup and resource management

Run with: `pytest -m integration`

#### Real API Tests (Optional)

Tests that make actual API calls to Replicate:
- Marked with `@pytest.mark.integration` and `@pytest.mark.slow`
- Require `REPLICATE_API_TOKEN` environment variable
- Will be skipped if token is not available
- Useful for verifying API compatibility

Run with: `pytest -m integration`

### Writing New Tests

When adding new features, follow these guidelines:

1. **Use fixtures from `conftest.py`**: Reuse existing mocks and test data
2. **Mock external dependencies**: APIs, system calls, file I/O
3. **Test real logic**: Don't mock core business logic
4. **Test error paths**: Verify error handling works correctly
5. **Use appropriate markers**: Mark integration tests with `@pytest.mark.integration`

Example test:

```python
@pytest.mark.unit
def test_new_feature(mock_audio_data, temp_dir):
    """Test description."""
    # Test implementation
    assert result == expected
```

### Test Configuration

Test configuration is in `pytest.ini`:
- Test discovery patterns
- Coverage settings
- Markers for different test types
- Timeout settings (300 seconds default)

### Continuous Integration

The test suite is designed to run in CI environments:
- Unit tests run quickly without network access
- Integration tests are optional and can be skipped
- Coverage reports help identify untested code
- All tests use proper mocking to avoid external dependencies

### Testing Recommendations

For manual testing, consider:

- Test with different audio devices
- Test with various recording durations
- Test hotkey detection across different applications
- Test error scenarios (no internet, invalid API token, etc.)
- Test vocabulary corrections with actual speech
- Test on different operating systems

## License

This project is provided as-is for personal and commercial use.

## Support

For issues, questions, or contributions, please refer to the project repository.

---

**Last Updated**: January 2025
**Version**: 1.0.0
**Status**: Production Ready
