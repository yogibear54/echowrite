import os
import json
import time
import threading
from datetime import datetime
from typing import Optional
from pathlib import Path

import keyboard
import sounddevice as sd
import numpy as np
import pyperclip
from dotenv import load_dotenv
from scipy.io.wavfile import write as wav_write

import config
from status_manager import StatusManager, Status
from providers import create_provider
from providers.base import TranscriptionProvider

# Load environment variables
load_dotenv()


class VoiceDictationTool:
    """Main class for voice dictation tool with global hotkey support."""
    
    def __init__(self):
        self.selected_device = None
        self.is_recording = False
        self.recording_thread = None
        self.audio_data = None
        self.ctrl_pressed = False
        self.alt_pressed = False
        self.recording_start_time = None
        self.is_cancelled = False
        
        # Paste key configuration - parse from config string (format: 'key1,key2,...')
        self.paste_keys = config.PASTE_KEYS.split(',')
        self.paste_mode_index = 0
        self.available_paste_modes = [
            ('ctrl', 'v'),           # Standard paste (Ctrl+V)
            ('ctrl', 'shift', 'v'),  # Terminal paste (Ctrl+Shift+V)
            ('ctrl', 'insert'),      # Alternative paste (Ctrl+Insert)
        ]
        # Set initial paste mode from config
        for i, mode in enumerate(self.available_paste_modes):
            if list(mode) == self.paste_keys:
                self.paste_mode_index = i
                break
        
        # Eagerly import pyautogui at startup. pyautogui's dependency
        # `mouseinfo` calls sys.exit() at import time when tkinter is missing,
        # which raises SystemExit (a BaseException, NOT caught by
        # `except Exception`). Importing lazily inside _paste_text meant that
        # SystemExit escaped up through the keyboard library's event-processing
        # loop and killed it permanently on the first recording, taking down
        # ALL hotkeys (Ctrl+Alt, F1/F2/F3, Escape). Import here so a missing
        # dependency is reported once and paste is simply disabled.
        self._pyautogui = None
        try:
            import pyautogui
            self._pyautogui = pyautogui
        except SystemExit as e:
            print(f"\u26a0\ufe0f  Paste disabled: pyautogui could not be imported ({e}).")
            print("    On Debian/Ubuntu, fix with: sudo apt-get install python3-tk")
        except Exception as e:
            print(f"\u26a0\ufe0f  Paste disabled: pyautogui could not be imported ({e}).")

        # Ensure temp directory exists
        self.temp_dir = Path(config.TEMP_DIR)
        self.temp_dir.mkdir(exist_ok=True)
        
        # Ensure recordings file exists
        self.recordings_file = Path(config.RECORDINGS_FILE)
        if not self.recordings_file.exists():
            with open(self.recordings_file, 'w') as f:
                json.dump([], f)
        
        # Initialize transcription provider
        try:
            self.provider: TranscriptionProvider = create_provider()
            print(f"✓ Transcription provider initialized: {config.PROVIDER}")
        except Exception as e:
            print(f"❌ Failed to initialize transcription provider: {e}")
            raise
        
        # Initialize status manager
        self.status_manager = StatusManager()
        # Register plugins based on config
        if 'i3status' in config.STATUS_PLUGINS:
            try:
                from plugins.i3status import I3StatusPlugin
                i3_plugin = I3StatusPlugin(config.I3_STATUS_FILE)
                self.status_manager.register_plugin(i3_plugin)
                print(f"✓ i3 status plugin enabled (status file: {config.I3_STATUS_FILE})")
            except Exception as e:
                print(f"⚠️  Failed to initialize i3 status plugin: {e}")
        if 'gnome' in config.STATUS_PLUGINS:
            try:
                from plugins.gnome import GnomeStatusPlugin
                gnome_plugin = GnomeStatusPlugin(config.GNOME_ICON_SIZE)
                self.status_manager.register_plugin(gnome_plugin)
                print(f"✓ GNOME status plugin enabled (icon size: {config.GNOME_ICON_SIZE})")
            except Exception as e:
                print(f"⚠️  Failed to initialize GNOME status plugin: {e}")
    
    def select_audio_device(self) -> Optional[int]:
        """Interactive device selection at startup."""
        try:
            devices = sd.query_devices()
            input_devices = []
            
            # Filter for input devices
            for i, device in enumerate(devices):
                if device['max_input_channels'] > 0:
                    input_devices.append((i, device))
            
            if not input_devices:
                print("❌ No audio input devices found!")
                print("Please check your microphone connection and try again.")
                return None
            
            # Display available devices
            print("\n🎤 Available Audio Input Devices:")
            print("-" * 60)
            for idx, (device_id, device) in enumerate(input_devices):
                default_marker = " (default)" if device_id == sd.default.device[0] else ""
                print(f"  {idx + 1}. [{device_id}] {device['name']}{default_marker}")
                print(f"     Channels: {device['max_input_channels']}, "
                      f"Sample Rate: {device['default_samplerate']} Hz")
            
            print("-" * 60)
            
            # Get user selection
            while True:
                try:
                    selection = input("\nSelect audio input device (or press Enter for default): ").strip()
                    
                    if not selection:
                        # Use default device
                        default_device_id = sd.default.device[0]
                        default_device = sd.query_devices(default_device_id)
                        print(f"✓ Using default device: [{default_device_id}] {default_device['name']}")
                        return default_device_id
                    
                    selection_num = int(selection)
                    if 1 <= selection_num <= len(input_devices):
                        selected_device_id, selected_device = input_devices[selection_num - 1]
                        print(f"✓ Selected device: [{selected_device_id}] {selected_device['name']}")
                        return selected_device_id
                    else:
                        print(f"❌ Invalid selection. Please enter a number between 1 and {len(input_devices)}.")
                except ValueError:
                    print("❌ Invalid input. Please enter a number or press Enter for default.")
                except KeyboardInterrupt:
                    print("\n\nExiting...")
                    return None
                    
        except Exception as e:
            print(f"❌ Error querying audio devices: {e}")
            print("Falling back to default device...")
            try:
                default_device_id = sd.default.device[0]
                return default_device_id
            except:
                return None
    
    def _apply_vocabulary_corrections(self, text: str) -> str:
        """Apply custom vocabulary corrections to improve transcription accuracy."""
        corrected_text = text
        
        for canonical_term, variations in config.CUSTOM_VOCABULARY.items():
            # Case-insensitive matching for all variations
            for variation in variations:
                # Use word boundaries to avoid partial matches
                import re
                pattern = re.compile(re.escape(variation), re.IGNORECASE)
                corrected_text = pattern.sub(canonical_term, corrected_text)
        
        return corrected_text
    
    def _save_wav_file(self, filename: str, audio_data: np.ndarray, sample_rate: int, channels: int):
        """Save audio data to WAV file using scipy.io.wavfile.write."""
        try:
            # Convert float32/float64 audio data to int16 for WAV format
            # Ensure data is in the range [-1.0, 1.0] and clip if necessary
            audio_clipped = np.clip(audio_data, -1.0, 1.0)
            audio_int16 = (audio_clipped * 32767).astype(np.int16)
            # Save using scipy (more reliable than wave module)
            wav_write(filename, sample_rate, audio_int16)
        except Exception as e:
            raise Exception(f"Failed to save WAV file: {e}")
    
    def _record_audio(self):
        """Record audio while hotkeys are held."""
        try:
            sample_rate = config.RECORDING_SETTINGS['sample_rate']
            channels = config.RECORDING_SETTINGS['channels']
            dtype = config.RECORDING_SETTINGS['dtype']
            
            # Record audio in a loop until stopped
            frames = []
            chunk_duration = 0.1  # Record in 100ms chunks
            
            with sd.InputStream(
                device=self.selected_device,
                samplerate=sample_rate,
                channels=channels,
                dtype=dtype
            ) as stream:
                while self.is_recording:
                    # Check if maximum recording duration has been reached
                    if self.recording_start_time is not None:
                        elapsed = time.time() - self.recording_start_time
                        if elapsed >= config.MAX_RECORDING_SECONDS:
                            self.is_recording = False
                            print(f"⏱️  Maximum recording duration reached ({config.MAX_RECORDING_MINUTES} minutes)")
                            self.status_manager.set_status(Status.PROCESSING)
                            break
                    
                    chunk, overflowed = stream.read(int(sample_rate * chunk_duration))
                    if overflowed:
                        print("⚠️  Audio buffer overflow detected")
                    frames.append(chunk)
            
            # Concatenate all chunks
            if frames:
                self.audio_data = np.concatenate(frames, axis=0)
            else:
                self.audio_data = None
                
        except Exception as e:
            print(f"❌ Recording error: {e}")
            self.audio_data = None
    
    
    def _save_transcription(self, text: str):
        """Save transcription to recordings.json."""
        try:
            # Load existing recordings
            if self.recordings_file.exists():
                with open(self.recordings_file, 'r') as f:
                    recordings = json.load(f)
            else:
                recordings = []
            
            # Add new recording
            recording = {
                'timestamp': datetime.now().isoformat(),
                'transcription': text
            }
            recordings.append(recording)
            
            # Save back to file
            with open(self.recordings_file, 'w') as f:
                json.dump(recordings, f, indent=2)
                
        except Exception as e:
            print(f"⚠️  Failed to save transcription to file: {e}")
    
    def _paste_text(self, text: str) -> bool:
        """Copy text to clipboard and paste it."""
        # Always copy to clipboard first so the text is never lost, even if
        # the paste keystroke itself fails or pyautogui is unavailable.
        try:
            pyperclip.copy(text)
        except Exception as e:
            print(f"⚠️  Failed to copy text to clipboard: {e}")
            return False

        if self._pyautogui is None:
            print("⚠️  Paste unavailable (pyautogui not loaded); text copied to clipboard only.")
            return False

        try:
            time.sleep(0.1)  # Small delay to ensure clipboard is ready
            current_mode = self.available_paste_modes[self.paste_mode_index]
            self._pyautogui.hotkey(*current_mode)
            return True
        except Exception as e:
            print(f"⚠️  Failed to paste text: {e}")
            return False
    
    def cycle_paste_mode(self, direction: int = 1):
        """Cycle through available paste modes.
        
        Args:
            direction: 1 to go forward, -1 to go backward
        """
        self.paste_mode_index = (self.paste_mode_index + direction) % len(self.available_paste_modes)
        current_mode = self.available_paste_modes[self.paste_mode_index]
        mode_name = '+'.join(current_mode).upper()
        # Clear entire line and print clean output (moves past trailing escape sequences)
        print("\r\x1b[2K\r", end="")  # Move to start and clear entire line
        print(f"🔄 Paste mode changed to: {mode_name}", flush=True)
        self.status_manager.set_status(Status.IDLE)  # Brief status update
    
    def get_current_paste_mode(self) -> str:
        """Get the current paste mode as a string for display."""
        current_mode = self.available_paste_modes[self.paste_mode_index]
        return '+'.join(current_mode).upper()
    
    
    def _process_recording(self):
        """Process the recorded audio: save, transcribe, and paste."""
        if self.audio_data is None or len(self.audio_data) == 0:
            print("⚠️  No audio data recorded")
            self.status_manager.set_status(Status.IDLE)
            return
        
        # Calculate duration
        sample_rate = config.RECORDING_SETTINGS['sample_rate']
        duration = len(self.audio_data) / sample_rate
        
        # Check minimum duration
        if duration < config.MIN_RECORDING_SECONDS:
            print(f"⚠️  Recording too short ({duration:.2f}s). Minimum is {config.MIN_RECORDING_SECONDS}s.")
            self.status_manager.set_status(Status.IDLE)
            return
        
        # Generate temp filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        temp_filename = self.temp_dir / f"{config.TEMP_FILE_PREFIX}{timestamp}.{config.AUDIO_FORMAT}"
        
        try:
            # Save audio to WAV file
            channels = config.RECORDING_SETTINGS['channels']
            self._save_wav_file(str(temp_filename), self.audio_data, sample_rate, channels)
            
            # Transcribe using provider
            transcribed_text = self.provider.transcribe(str(temp_filename))
            
            if transcribed_text:
                # Apply vocabulary corrections (application-level, not provider-specific)
                corrected_text = self._apply_vocabulary_corrections(transcribed_text)
                
                print(f"✓ Transcription: {corrected_text}")
                
                # Try to paste
                paste_success = self._paste_text(corrected_text)
                if paste_success:
                    print("✓ Text pasted successfully")
                else:
                    print("⚠️  Text copied to clipboard but paste failed")
                
                # Save transcription
                self._save_transcription(corrected_text)
            else:
                print("❌ Transcription failed")
            
        except Exception as e:
            print(f"❌ Error processing recording: {e}")
        finally:
            # Clean up temp file
            try:
                if temp_filename.exists():
                    temp_filename.unlink()
            except Exception as e:
                print(f"⚠️  Failed to delete temp file: {e}")
            # Reset status to idle
            self.status_manager.set_status(Status.IDLE)
    
    def _on_key_event(self, event):
        """Handle keyboard events for hotkey detection."""
        # Normalize key names
        key_name = event.name.lower() if event.name else ''
        is_ctrl = key_name in ['ctrl', 'left ctrl', 'right ctrl']
        is_alt = key_name in ['alt', 'left alt', 'right alt']
        # Check for escape key with multiple possible names (Linux may use 'esc' instead of 'escape')
        is_escape = key_name in ['escape', 'esc']
        # Check for function keys for paste mode control
        is_f1 = key_name in ['f1']
        is_f2 = key_name in ['f2']
        is_f3 = key_name in ['f3']
        
        
        
        if event.event_type == keyboard.KEY_DOWN:
            # Handle function keys for paste mode control
            if is_f1 and not self.is_recording:
                self.cycle_paste_mode(1)  # Next mode
                return
            elif is_f2 and not self.is_recording:
                self.cycle_paste_mode(-1)  # Previous mode
                return
            elif is_f3 and not self.is_recording:
                # Clear entire line and print clean output (moves past trailing escape sequences)
                print("\r\x1b[2K\r", end="")  # Move to start and clear entire line
                print(f"📋 Current paste mode: {self.get_current_paste_mode()}", flush=True)
                return
            # Handle Escape key cancellation during recording
            if is_escape and self.is_recording and not self.is_cancelled:
                self.is_cancelled = True
                self.is_recording = False
                print("❌ Recording cancelled")
                self.status_manager.set_status(Status.IDLE)
                self.audio_data = None  # Clear audio data to prevent processing
                
                # Wait for recording thread to finish
                if self.recording_thread:
                    self.recording_thread.join(timeout=2.0)
                return
            
            if is_ctrl:
                self.ctrl_pressed = True
            elif is_alt:
                self.alt_pressed = True
            
            # Start recording when both keys are pressed
            if self.ctrl_pressed and self.alt_pressed and not self.is_recording:
                self.is_recording = True
                self.is_cancelled = False  # Reset cancellation flag for new recording
                self.recording_start_time = time.time()
                print("🔴 Recording started... (Release Ctrl+Alt to stop, or press Escape to cancel)")
                self.status_manager.set_status(Status.RECORDING)
                
                # Start recording in a separate thread
                self.recording_thread = threading.Thread(target=self._record_audio, daemon=True)
                self.recording_thread.start()
        
        elif event.event_type == keyboard.KEY_UP:
            # Clear any trailing function key escape sequences
            if is_f1 or is_f2 or is_f3:
                print("\r\x1b[2K", end="", flush=True)  # Clear entire line
                return
            if is_ctrl:
                self.ctrl_pressed = False
            elif is_alt:
                self.alt_pressed = False
            
            # Reset cancellation flag if keys are released after cancellation
            if self.is_cancelled and not self.is_recording:
                self.is_cancelled = False
            
            # Stop recording when either key is released
            if self.is_recording and (not self.ctrl_pressed or not self.alt_pressed):
                self.is_recording = False
                
                # Wait for recording thread to finish
                if self.recording_thread:
                    self.recording_thread.join(timeout=2.0)
                
                # Only process if not cancelled
                if self.is_cancelled:
                    # Reset cancellation flag for next recording
                    self.is_cancelled = False
                    # Status already set to IDLE in cancellation handler
                else:
                    print("⏹️  Recording stopped. Processing...")
                    self.status_manager.set_status(Status.PROCESSING)
                    # Process the recording
                    self._process_recording()
    
    def start(self):
        """Start the voice dictation tool."""
        print("=" * 60)
        print("🎙️  Voice Dictation Tool")
        print("=" * 60)
        
        # Provider is already initialized in __init__, so we just need to verify it exists
        if not hasattr(self, 'provider') or self.provider is None:
            print("❌ Transcription provider not initialized!")
            print("Please check your configuration and API tokens.")
            return
        
        # Select audio device
        self.selected_device = self.select_audio_device()
        if self.selected_device is None:
            print("❌ Could not select audio device. Exiting.")
            return
        
        print("\n" + "=" * 60)
        print("✓ Ready! Press and hold Ctrl+Alt to start recording.")
        print("  Release Ctrl+Alt to stop recording and transcribe.")
        print("  Press Escape during recording to cancel.")
        print(f"  Current paste mode: {self.get_current_paste_mode()}")
        print("  Press F1/F2 to cycle through paste modes.")
        print("  Press F3 to show current paste mode.")
        print("  Press Ctrl+C to exit.")
        print("=" * 60 + "\n")
        
        # Set status to IDLE now that app is ready
        self.status_manager.set_status(Status.IDLE)
        
        # Set up global keyboard hook for all key events
        keyboard.hook(self._on_key_event)
        
        try:
            # Keep the main thread alive
            keyboard.wait()
        except KeyboardInterrupt:
            print("\n\n👋 Shutting down...")
        finally:
            # Clean up
            if hasattr(self, 'provider'):
                self.provider.cleanup()
            self.status_manager.cleanup()
            keyboard.unhook_all()


def main():
    """Main entry point."""
    tool = VoiceDictationTool()
    tool.start()


if __name__ == "__main__":
    main()
