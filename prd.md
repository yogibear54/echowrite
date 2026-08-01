

# 🎙️ Product Requirements Document (PRD)

## 📌 Project Title: Local Voice Dictation Tool

## 👤 Target Audience

-   Developers and knowledge workers who want to speed up their typing workflow
-   Users who prefer speaking over typing for quick text input
-   People working in Cursor, browsers, documents, or any text-based applications
-   Users who want a simple, privacy-focused voice-to-text solution 

----------

## Objective

Create a local Python script that captures voice input via global hotkeys, transcribes speech using Replicate's incredibly-fast-whisper model, and automatically pastes the result wherever the user's cursor is currently positioned. 

----------

## 🧩 Core Features

### 1. Global Hotkey Detection

-   Listen for Ctrl+Alt keypress globally (works even when script isn't focused)
-   Recording starts when both keys are pressed down
-   The logic should detect when two keys are pressed down and wait patiently until they are released (key-up event)
-   Handle edge cases like rapid key presses or interruptions

### 2. Audio Recording

-   Capture microphone input in real-time using system default microphone
-   Save temporary audio file in common format (WAV recommended)
-   Automatic cleanup of temporary files after processing
-   Handle microphone permissions and availability gracefully

### 3. Speech Transcription

-   Send recorded audio to Replicate's incredibly-fast-whisper model
-   Use local WAV file as input (Replicate accepts local file objects via `open("file.wav", "rb")`)
-   The model returns transcribed text as a string (not FileOutput for transcription models)
-   Use optimal model settings for speed and accuracy (transcribes 150 minutes of audio in ~100 seconds)
-   Handle API errors, network issues, and rate limits
-   Example usage: `output = replicate.run("vaibhavs10/incredibly-fast-whisper", input={"audio": open("recording.wav", "rb")})`

### 4. Automatic Text Insertion

-   Copy transcribed text to system clipboard
-   Automatically trigger Ctrl+V to paste at current cursor position
-   Work universally across all applications (Cursor, browsers, documents, etc.)
-   If unable to paste, thats okay -> we will go to the "Transcription Persistance"

### 5. Recording Persistence

-   Save all transcriptions to local JSON file
-   Include timestamp & transcription
-   Provides backup in case clipboard/paste fails
-   Enables reviewing past transcriptions and debugging issues

### 6. Temp file
-   Have a temp file where the .wav file is stored only for the purpose of transcribing
-   Once its done, and it was successfull the file can be deleted

### 7. User Feedback

-   Clear console output showing recording status
-   Error handling with user-friendly messages
-   Processing indicators while waiting for transcription
-   Success confirmation when text is pasted

----------

## Tech Stack

keyboard==0.13.5 (vendored as a patched fork at `vendor/keyboard/` — see README for the why)
sounddevice==0.4.7
scipy==1.11.4
replicate==0.34.0
pyperclip==1.8.2
pyautogui==0.9.54
python-dotenv==1.0.0

**Note:** The current install flow is `./install.sh` (see README), not the manual pip command below. The tech stack list above is for reference only.

----------

## File Structure

```
voice_dictation/
├── start.py       # Main script
├── requirements.txt          # Python dependencies
├── config.py                 # Configuration settings
├── recordings.json           # Recording history and backup
├── temp/                     # Temporary audio files

```

----------

## Dependencies

Install via pip:

**Platform-specific requirements:**
- **Linux**: Install `xclip` (or `xsel`) for pyperclip: `sudo apt-get install xclip`. Global hotkeys no longer require `sudo` — the `install.sh` script sets up udev rules + the `input` group, and the `keyboard` library is vendored as a no-root-check fork (see README).
- **macOS**: No additional requirements
- **Windows**: No additional requirements
## Configuration Options

```python
# config.py
RECORDING_SETTINGS = {
    'sample_rate': 44100,
    'channels': 1,
    'dtype': 'float64'
}

# Replicate API settings
API_SETTINGS = {
    'model': 'vaibhavs10/incredibly-fast-whisper',
    # Model parameters (optional, defaults are usually optimal)
    # 'language': 'en',  # Optional: specify language for faster processing
    # 'task': 'transcribe',  # Options: 'transcribe' or 'translate'
    # 'batch_size': 24,  # Batch size for processing (default is optimized)
}

# Note: Replicate accepts local file objects directly:
# output = replicate.run(
#     "vaibhavs10/incredibly-fast-whisper",
#     input={"audio": open("path/to/audio.wav", "rb")}
# )
# The output will be a string containing the transcribed text

HOTKEY = 'ctrl+alt'
TEMP_FILE_PREFIX = 'voice_recording_'

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

# Audio settings
AUDIO_FORMAT = 'wav'
MIN_RECORDING_SECONDS = 1.0  # Minimum recording length to process 
```

----------

## Start file (main)

```python
import os
import json
import time
import threading
import wave
from datetime import datetime
from typing import Optional

import keyboard
import sounddevice as sd
import wave
import numpy as np
import pyperclip
import pyautogui
import replicate
from dotenv import load_dotenv

import config

# Load environment variables
load_dotenv()

# example vocab handling

def _apply_vocabulary_corrections(self, text: str) -> str:
        """Apply custom vocabulary corrections to improve transcription accuracy."""
        corrected_text = text

# example saving audio
def _save_wav_file(self, filename: str, audio_data, sample_rate: int, channels: int):
        """Save audio data to WAV file using scipy.io.wavfile.write (recommended)."""
        try:
            from scipy.io.wavfile import write
            # Convert float64 audio data to int16 for WAV format
            audio_int16 = (audio_data * 32767).astype(np.int16)
            # Save using scipy (more reliable than wave module)
            write(filename, sample_rate, audio_int16)

```

----------

## Usage Flow

1.  Run script: `python start.py`
2.  Position cursor in desired text field
3.  Press and hold Ctrl+Alt to start recording (hear beep/see console message)
4.  Speak your text clearly while holding keys
5.  Release Ctrl+Alt to stop recording and process
6.  Wait for transcription (1-3 seconds typically)
7.  Text automatically appears at cursor position
8.  Review recording history in recordings.json if needed
Note: if recording is less than 1 second, it should not be sent via API

----------

## Error Handling

-   **No microphone detected**: Graceful error with setup instructions
-   **API token missing**: Clear error message with setup guide
-   **Network issues**: Retry logic with user notification
-   **Audio recording fails**: Fallback options and troubleshooting
-   **Replicate API errors**: Handle prediction failures and rate limits gracefully
-   **Permission errors (Linux)**: Global hotkeys no longer need root. If hotkeys aren't detected, the most common cause is that the user wasn't logged out/back in after `install.sh` added them to the `input` group — `id -nG | grep -w input` to verify.
-   **Clipboard errors**: Fallback to transcription persistence if clipboard operations fail

----------

## Rules
-   Don't create a README.md file
-   There will be a .env file pre-made for you already, so just assume that's already created and that it contains something like:
    ```
    REPLICATE_API_TOKEN=r8_...
    MAX_RECORDING_MINUTES=5
    ```
-   Use vaibhavs10/incredibly-fast-whisper model which is incredibly fast (transcribes 150 minutes in ~100 seconds) and cost-effective (~$0.0036 per run)
-   Authenticate with Replicate by setting the REPLICATE_API_TOKEN environment variable (see https://replicate.com/account/api-tokens)
-   **Library Usage Notes:**
    - **keyboard**: Use `keyboard.on_press_key()` or `keyboard.add_hotkey()` for global hotkey detection. The library is vendored as a no-root-check fork at `vendor/keyboard/` so it works on Linux without `sudo` (provided the `input` group / udev rules from `install.sh` are in place).
    - **sounddevice**: Use `sd.rec()` for simple recording or `sd.InputStream()` with callbacks for real-time processing. Check `sd.query_devices()` for available microphones.
    - **scipy.io.wavfile**: Use `write(filename, sample_rate, data)` where data is int16 numpy array. More reliable than built-in `wave` module.
    - **replicate**: Pass local files using `open("file.wav", "rb")`. Output for transcription models is a string.
    - **pyperclip**: Use `pyperclip.copy(text)` and `pyperclip.paste()`. On Linux, requires xclip or xsel.
    - **pyautogui**: Use `pyautogui.hotkey('ctrl', 'v')` for pasting. May require root on Linux.
    - **python-dotenv**: Simply call `load_dotenv()` at startup to load .env file variables.

----------

# Tasks

-   [ ] Set up Python environment and dependencies
-   [ ] Implement global hotkey detection (Ctrl+Alt)
-   [ ] Add microphone recording functionality
-   [ ] Integrate Replicate incredibly-fast-whisper API calls
-   [ ] Implement clipboard and auto-paste functionality
-   [ ] Add error handling and user feedback
-   [ ] Test across different applications
-   [ ] Add configuration options

Critical Rules: Once a task is completed mark it with ✅

----------