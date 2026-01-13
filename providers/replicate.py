"""Replicate provider for transcription services."""
import os
from typing import Optional

import replicate
import requests

from .base import TranscriptionProvider
import config


class ReplicateProvider(TranscriptionProvider):
    """Transcription provider using Replicate's incredibly-fast-whisper model."""
    
    def __init__(self, api_token: Optional[str] = None, api_settings: Optional[dict] = None):
        """Initialize Replicate provider.
        
        Args:
            api_token: Replicate API token. If None, will read from REPLICATE_API_TOKEN env var.
            api_settings: Provider-specific settings dict. If None, will use config.API_SETTINGS.
        """
        self.api_token = api_token or os.getenv('REPLICATE_API_TOKEN')
        self.api_settings = api_settings or config.API_SETTINGS
        
        if not self.api_token:
            raise ValueError("REPLICATE_API_TOKEN not found. Please set it in environment variables or pass it to __init__.")
        
        # Set the API token in environment (Replicate SDK reads from env)
        os.environ['REPLICATE_API_TOKEN'] = self.api_token
    
    def _upload_audio_to_replicate(self, audio_file_path: str) -> Optional[str]:
        """Upload audio file to Replicate and get the URL.
        
        Args:
            audio_file_path: Path to the audio file to upload
            
        Returns:
            URL to the uploaded file, or None if upload failed
        """
        try:
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
                    "Authorization": f"Bearer {self.api_token}"
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
    
    def transcribe(self, audio_file_path: str) -> Optional[str]:
        """Transcribe audio file using Replicate API.
        
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
            
            # Upload audio file and get URL
            audio_url = self._upload_audio_to_replicate(audio_file_path)
            if not audio_url:
                return None
            
            model_name = self.api_settings['model']
            print(f"🔄 Transcribing audio using model: {model_name.split(':')[0]}...")
            
            # Prepare input parameters
            input_params = {
                "audio": audio_url,
                "task": self.api_settings.get('task', 'transcribe'),
                "language": self.api_settings.get('language', 'None'),
                "timestamp": self.api_settings.get('timestamp', 'chunk'),
                "batch_size": self.api_settings.get('batch_size', 64),
                "diarise_audio": self.api_settings.get('diarise_audio', False),
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
            
            return transcribed_text
            
        except Exception as e:
            error_msg = str(e)
            print(f"❌ Transcription error: {error_msg}")
            
            # Provide more specific error messages
            if "404" in error_msg or "not found" in error_msg.lower():
                print("⚠️  Model not found. Please check:")
                print(f"   1. Model name: {self.api_settings['model']}")
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
