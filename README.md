# 🎙️ Voice Dictation Tool

A local Python application that captures voice input via global hotkeys, transcribes speech using Replicate's incredibly-fast-whisper model, and automatically pastes the result wherever your cursor is positioned.

## Features

- **Global Hotkey Support**: Press and hold Ctrl+Alt to start recording, release to stop (works even when the app isn't focused)
- **Device Selection**: Interactive audio device selection at startup
- **Fast Transcription**: Uses Replicate's incredibly-fast-whisper model (transcribes 150 minutes in ~100 seconds)
- **Automatic Pasting**: Transcribed text is automatically copied to clipboard and pasted at cursor position
- **Recording History**: All transcriptions are saved to `recordings.json` with timestamps
- **Custom Vocabulary**: Supports custom vocabulary corrections for better recognition of technical terms
- **Environment Configuration**: Configure settings via `.env` file
- **Maximum Recording Duration**: Configurable limit to prevent excessive recordings

## Requirements

### System Requirements

- Python 3.8 or higher
- Linux, macOS, or Windows
- Microphone access
- Internet connection (for Replicate API)

### System Dependencies

#### Linux (Debian/Ubuntu)

```bash
sudo apt-get update
sudo apt-get install portaudio19-dev python3-dev
sudo apt-get install xclip  # Required for clipboard functionality
```

#### macOS

No additional system dependencies required (uses built-in audio system).

#### Windows

No additional system dependencies required.

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

Option 2 - Use sudo with venv Python directly:
```bash
sudo venv/bin/python start.py
```

Option 3 - Activate venv first, then use sudo:
```bash
source venv/bin/activate
sudo $(which python) start.py
```

**On macOS/Windows**:

```bash
source venv/bin/activate  # On Windows: venv\Scripts\activate
python start.py
```

**Note**: On Linux, the `keyboard` library requires root permissions for global hotkey detection. Use `sudo` with the full path to your virtual environment's Python interpreter (`venv/bin/python`) to ensure it uses the correct Python and installed packages.

3. **Select your audio input device** when prompted (or press Enter for default)

4. **Start dictating**:
   - Position your cursor in the desired text field (any application)
   - Press and **hold** Ctrl+Alt to start recording
   - Speak your text
   - **Release** Ctrl+Alt to stop recording and process
   - The transcribed text will automatically appear at your cursor position

5. **Exit the application**: Press Ctrl+C in the terminal

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
User Releases Ctrl+Alt (key-up event)
    ↓
Stop Recording & Save to temp WAV file
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
Copy to Clipboard & Auto-Paste (Ctrl+V)
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
- Prevents duplicate recordings from rapid key presses

**Implementation Details:**
- Key events are normalized (handles 'ctrl', 'left ctrl', 'right ctrl', etc.)
- Uses threading to avoid blocking the main loop
- Recording thread runs as daemon thread

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

#### 4. Replicate API Integration

**File Upload (`_upload_audio_to_replicate()`):**
- Uploads WAV file to Replicate's file storage API
- Endpoint: `POST https://api.replicate.com/v1/files`
- Field name: `content` (multipart/form-data)
- Returns: URL to uploaded file (`urls.get`)

**Transcription (`_transcribe_audio()`):**
- Uses Replicate Python SDK (`replicate.run()`)
- Model: `vaibhavs10/incredibly-fast-whisper:3ab86df6c8f54c11309d4d1f930ac292bad43ace52d10c80d87eb258b3c9f79c`
- Input parameters:
  - `audio`: URL from file upload
  - `task`: 'transcribe' or 'translate'
  - `language`: Language code or 'None' for auto-detection
  - `timestamp`: 'chunk' or 'word'
  - `batch_size`: 64 (optimized for speed)
  - `diarise_audio`: False (speaker diarization disabled)

**Important Notes:**
- The model requires a version tag (specific commit hash)
- Audio must be uploaded first to get a URL (cannot pass file directly)
- The API returns transcribed text as a string

#### 5. Vocabulary Correction (`start.py` - `_apply_vocabulary_corrections()`)

- Uses regex for case-insensitive pattern matching
- Matches common mispronunciations against `CUSTOM_VOCABULARY`
- Replaces matches with canonical terms
- Applied after transcription, before pasting

#### 6. Text Insertion (`start.py` - `_paste_text()`)

- Copies text to clipboard using `pyperclip.copy()`
- Triggers Ctrl+V using `pyautogui.hotkey('ctrl', 'v')`
- Includes small delay to ensure clipboard is ready
- Falls back gracefully if paste fails (text still saved to recordings.json)

#### 7. Persistence (`start.py` - `_save_transcription()`)

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

## Project Structure

```
voice2text/
├── start.py              # Main application script (VoiceDictationTool class)
├── config.py             # Configuration settings with env variable support
├── requirements.txt      # Python dependencies
├── .env.example          # Example environment variables file
├── .env                  # Your environment variables (create from .env.example)
├── .gitignore            # Git ignore rules
├── run.sh                # Helper script for Linux (runs with sudo)
├── recordings.json       # Transcription history (created automatically)
├── temp/                 # Temporary audio files directory
├── venv/                 # Virtual environment (created during setup)
└── README.md             # This file
```

### File Descriptions

- **`start.py`**: Main application containing `VoiceDictationTool` class with all core functionality
- **`config.py`**: Configuration management with environment variable support and validation
- **`run.sh`**: Helper script for Linux that automatically uses sudo with the correct Python path
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

Or use the helper script: `./run.sh`

## How It Works (Detailed)

### 1. Hotkey Detection

- Uses `keyboard.hook()` to register a global keyboard event handler
- Monitors all key press and release events
- Tracks state of Ctrl and Alt keys independently
- When both keys are pressed down simultaneously, starts recording
- When either key is released, stops recording

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

### 4. Replicate API Workflow

**Step 1: File Upload**
- Opens the WAV file in binary mode
- Creates multipart form-data with field name `content`
- POSTs to `https://api.replicate.com/v1/files`
- Includes Authorization header with API token
- Receives JSON response with file URL

**Step 2: Transcription**
- Uses the file URL from upload step
- Calls `replicate.run()` with model name and parameters
- Model processes audio and returns transcribed text
- Text is returned as a string

### 5. Post-Processing

- Vocabulary corrections are applied using regex
- Text is copied to system clipboard
- Auto-paste is triggered using Ctrl+V hotkey
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

### Known Issues & Workarounds

1. **Linux Sudo Requirement**: Worked around by using full path to venv Python
2. **Audio Format Compatibility**: Resolved by using float32 instead of float64
3. **File Upload Format**: Required using 'content' field name, not 'file'
4. **Model Version**: Must include specific commit hash version tag

### Testing Recommendations

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
