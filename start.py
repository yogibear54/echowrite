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
import pyautogui
import replicate
import requests
from dotenv import load_dotenv
from scipy.io.wavfile import write as wav_write

import config
from status_manager import StatusManager, Status

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
        
        # Ensure temp directory exists
        self.temp_dir = Path(config.TEMP_DIR)
        self.temp_dir.mkdir(exist_ok=True)
        
        # Ensure recordings file exists
        self.recordings_file = Path(config.RECORDINGS_FILE)
        if not self.recordings_file.exists():
            with open(self.recordings_file, 'w') as f:
                json.dump([], f)
        
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
    
    def _upload_audio_to_replicate(self, audio_file_path: str) -> Optional[str]:
        """Upload audio file to Replicate and get the URL."""
        try:
            api_token = os.getenv('REPLICATE_API_TOKEN')
            if not api_token:
                return None
            
            print("📤 Uploading audio file to Replicate...")
            
            # Upload file to Replicate API
            # Based on curl example: -F "content=@$audio;type=application/octet-stream;filename=$audio"
            # The field name is 'content' (not 'file')
            with open(audio_file_path, "rb") as audio_file:
                filename = os.path.basename(audio_file_path)
                
                # Prepare multipart form data - field name is 'content'
                files = {
                    'content': (filename, audio_file, 'application/octet-stream')
                }
                headers = {
                    "Authorization": f"Bearer {api_token}"
                }
                
                # Don't set Content-Type header manually - requests will set it with boundary
                response = requests.post(
                    "https://api.replicate.com/v1/files",
                    headers=headers,
                    files=files
                )
                response.raise_for_status()
                
            # Get the URL from response
            file_data = response.json()
            audio_url = file_data.get('urls', {}).get('get')
            
            if not audio_url:
                print("❌ Failed to get audio URL from Replicate upload")
                return None
            
            return audio_url
            
        except requests.exceptions.HTTPError as e:
            error_detail = "Unknown error"
            try:
                error_response = e.response.json()
                error_detail = error_response.get('detail', str(e))
            except:
                error_detail = str(e)
            print(f"❌ Failed to upload audio file: {e.response.status_code} - {error_detail}")
            return None
        except Exception as e:
            print(f"❌ Failed to upload audio file: {e}")
            return None
    
    def _transcribe_audio(self, audio_file_path: str) -> Optional[str]:
        """Send audio to Replicate API for transcription."""
        try:
            api_token = os.getenv('REPLICATE_API_TOKEN')
            if not api_token:
                print("❌ REPLICATE_API_TOKEN not found in environment variables!")
                print("Please set REPLICATE_API_TOKEN in your .env file.")
                return None
            
            # Verify file exists
            if not os.path.exists(audio_file_path):
                print(f"❌ Audio file not found: {audio_file_path}")
                return None
            
            # Set the API token in environment (Replicate SDK reads from env)
            os.environ['REPLICATE_API_TOKEN'] = api_token
            
            # Upload audio file and get URL
            audio_url = self._upload_audio_to_replicate(audio_file_path)
            if not audio_url:
                return None
            
            model_name = config.API_SETTINGS['model']
            print(f"🔄 Transcribing audio using model: {model_name.split(':')[0]}...")
            
            # Prepare input parameters
            input_params = {
                "audio": audio_url,
                "task": config.API_SETTINGS.get('task', 'transcribe'),
                "language": config.API_SETTINGS.get('language', 'None'),
                "timestamp": config.API_SETTINGS.get('timestamp', 'chunk'),
                "batch_size": config.API_SETTINGS.get('batch_size', 64),
                "diarise_audio": config.API_SETTINGS.get('diarise_audio', False),
            }
            
            # Run the model
            output = replicate.run(model_name, input=input_params)
            
            # Output format can vary - handle different response types
            if isinstance(output, str):
                transcribed_text = output.strip()
            elif isinstance(output, dict):
                # Some models return dict with 'text' key
                if 'text' in output:
                    transcribed_text = str(output['text']).strip()
                else:
                    # Try to get the first value or convert entire dict
                    transcribed_text = str(list(output.values())[0] if output else "").strip()
            elif hasattr(output, '__iter__') and not isinstance(output, str):
                # Handle iterables (lists, etc.)
                transcribed_text = ' '.join(str(item) for item in output).strip()
            else:
                transcribed_text = str(output).strip()
            
            if not transcribed_text:
                print("⚠️  Transcription returned empty result")
                return None
            
            # Apply vocabulary corrections
            corrected_text = self._apply_vocabulary_corrections(transcribed_text)
            
            return corrected_text
            
        except Exception as e:
            error_msg = str(e)
            print(f"❌ Transcription error: {error_msg}")
            
            # Provide more specific error messages
            if "404" in error_msg or "not found" in error_msg.lower():
                print("⚠️  Model not found. Please check:")
                print(f"   1. Model name: {config.API_SETTINGS['model']}")
                print("   2. Your Replicate API token is valid")
                print("   3. The model exists on Replicate")
            elif "rate limit" in error_msg.lower():
                print("⚠️  Rate limit exceeded. Please wait a moment and try again.")
            elif "network" in error_msg.lower() or "connection" in error_msg.lower():
                print("⚠️  Network error. Please check your internet connection.")
            elif "401" in error_msg or "unauthorized" in error_msg.lower():
                print("⚠️  Authentication failed. Please check your REPLICATE_API_TOKEN.")
            else:
                print(f"⚠️  Full error details: {type(e).__name__}: {error_msg}")
            
            return None
    
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
        try:
            # Copy to clipboard
            pyperclip.copy(text)
            time.sleep(0.1)  # Small delay to ensure clipboard is ready
            
            # Trigger paste
            pyautogui.hotkey('ctrl', 'v')
            return True
        except Exception as e:
            print(f"⚠️  Failed to paste text: {e}")
            return False
    
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
            
            # Transcribe
            transcribed_text = self._transcribe_audio(str(temp_filename))
            
            if transcribed_text:
                print(f"✓ Transcription: {transcribed_text}")
                
                # Try to paste
                paste_success = self._paste_text(transcribed_text)
                if paste_success:
                    print("✓ Text pasted successfully")
                else:
                    print("⚠️  Text copied to clipboard but paste failed")
                
                # Save transcription
                self._save_transcription(transcribed_text)
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
        
        if event.event_type == keyboard.KEY_DOWN:
            if is_ctrl:
                self.ctrl_pressed = True
            elif is_alt:
                self.alt_pressed = True
            
            # Start recording when both keys are pressed
            if self.ctrl_pressed and self.alt_pressed and not self.is_recording:
                self.is_recording = True
                self.recording_start_time = time.time()
                print("🔴 Recording started... (Release Ctrl+Alt to stop)")
                self.status_manager.set_status(Status.RECORDING)
                
                # Start recording in a separate thread
                self.recording_thread = threading.Thread(target=self._record_audio, daemon=True)
                self.recording_thread.start()
        
        elif event.event_type == keyboard.KEY_UP:
            if is_ctrl:
                self.ctrl_pressed = False
            elif is_alt:
                self.alt_pressed = False
            
            # Stop recording when either key is released
            if self.is_recording and (not self.ctrl_pressed or not self.alt_pressed):
                self.is_recording = False
                print("⏹️  Recording stopped. Processing...")
                self.status_manager.set_status(Status.PROCESSING)
                
                # Wait for recording thread to finish
                if self.recording_thread:
                    self.recording_thread.join(timeout=2.0)
                
                # Process the recording
                self._process_recording()
    
    def start(self):
        """Start the voice dictation tool."""
        print("=" * 60)
        print("🎙️  Voice Dictation Tool")
        print("=" * 60)
        
        # Check for API token
        if not os.getenv('REPLICATE_API_TOKEN'):
            print("❌ REPLICATE_API_TOKEN not found in environment variables!")
            print("Please set REPLICATE_API_TOKEN in your .env file.")
            return
        
        # Select audio device
        self.selected_device = self.select_audio_device()
        if self.selected_device is None:
            print("❌ Could not select audio device. Exiting.")
            return
        
        print("\n" + "=" * 60)
        print("✓ Ready! Press and hold Ctrl+Alt to start recording.")
        print("  Release Ctrl+Alt to stop recording and transcribe.")
        print("  Press Ctrl+C to exit.")
        print("=" * 60 + "\n")
        
        # Set up global keyboard hook for all key events
        keyboard.hook(self._on_key_event)
        
        try:
            # Keep the main thread alive
            keyboard.wait()
        except KeyboardInterrupt:
            print("\n\n👋 Shutting down...")
        finally:
            # Clean up
            self.status_manager.cleanup()
            keyboard.unhook_all()


def main():
    """Main entry point."""
    tool = VoiceDictationTool()
    tool.start()


if __name__ == "__main__":
    main()
