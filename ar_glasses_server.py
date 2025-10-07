#!/usr/bin/env python3

import os
import sys
import json
import asyncio
import base64
import time
import warnings
import numpy as np
from datetime import datetime
from typing import Dict, Any, List
import soundfile as sf
from pathlib import Path
import websockets
import torch

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from speaker_diarization import DiarizationPipeline

class ARGlassesServer:
    def __init__(self):
        """Initialize the AR Glasses server with speaker recognition."""
        print("[SERVER] Initializing AR Glasses Server...")
        
        # Force CPU for memory efficiency on Render
        self.device = "cpu"
        print(f"[SERVER] Using device: {self.device} (forced CPU for memory efficiency)")
        
        self.load_environment()
        
        self.hf_token = os.getenv("HF_TOKEN")
        if not self.hf_token:
            print("[SERVER] Warning: HF_TOKEN not found in environment. Using fallback models.")
            self.hf_token = "hf_your_token_here"
        
        print("[SERVER] Loading diarization model...")
        print(f"[SERVER] Using HF_TOKEN: {self.hf_token[:10]}..." if self.hf_token else "[SERVER] No HF_TOKEN provided")
        
        try:
            # Initialize only diarization pipeline for faster processing
            print("[SERVER] Initializing Diarization Pipeline...")
            self.diarization_pipeline = DiarizationPipeline(self.hf_token)
            print("[SERVER] Diarization pipeline loaded successfully")
            
            print("[SERVER] Model loaded successfully!")
            print("[SERVER] Using diarization only for faster processing")
            
        except Exception as e:
            print(f"[SERVER] CRITICAL ERROR: Failed to load diarization model!")
            print(f"[SERVER] Error: {e}")
            print(f"[SERVER] Error type: {type(e).__name__}")
            import traceback
            print(f"[SERVER] Traceback: {traceback.format_exc()}")
            print("[SERVER] Server cannot start without diarization model!")
            print("[SERVER] Please check:")
            print("[SERVER] 1. HF_TOKEN is valid and set correctly")
            print("[SERVER] 2. All required packages are installed")
            print("[SERVER] 3. Internet connection for model downloads")
            sys.exit(1)
        
        self.host = os.getenv("SERVER_HOST", "0.0.0.0")
        self.port = int(os.getenv("SERVER_PORT", "8000"))
        self.active_connections = set()
        self.main_loop = None
        
        # Voice registration storage
        self.registered_voices = {}  # voice_id -> voice_data
        self.user_voice_exclusions = {}  # connection -> voice_id to exclude
        
        self.debug_json_dir = Path("debug_json_output")
        self.debug_json_dir.mkdir(exist_ok=True)
        self._clear_debug_output()
        
        # Memory management
        self._cleanup_memory()
        
        print("[SERVER] Server initialized successfully")
        
    def _cleanup_memory(self):
        """Clean up memory to prevent OOM."""
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _should_exclude_speaker(self, speaker_id, audio_array, sample_rate):
        """Check if a speaker should be excluded based on voice registration."""
        # For now, implement a simple check - in a real implementation,
        # you would compare the speaker's voice characteristics with registered voices
        # This is a placeholder for the voice matching logic
        
        # Check if any connection has voice exclusion enabled
        for connection in self.active_connections:
            if connection in self.user_voice_exclusions:
                excluded_voice_id = self.user_voice_exclusions[connection]
                if excluded_voice_id and excluded_voice_id in self.registered_voices:
                    # Here you would implement actual voice matching
                    # For now, we'll do a simple check based on speaker ID patterns
                    if speaker_id.startswith("SPEAKER_00") or speaker_id == "SPEAKER_0":
                        print(f"[SERVER] Voice exclusion: Speaker {speaker_id} matches exclusion pattern")
                        return True
        
        return False


    def load_environment(self):
        """Load environment variables from config.env file, but prioritize existing env vars."""
        env_file = "config.env"
        if os.path.exists(env_file):
            print(f"[SERVER] Loading environment from {env_file}")
            with open(env_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        # Only set if not already in environment (prioritize Docker env vars)
                        if key.strip() not in os.environ:
                            os.environ[key.strip()] = value.strip()
            print("[SERVER] Environment variables loaded successfully")
        else:
            print(f"[SERVER] Warning: {env_file} not found, using system environment variables")

    def _clear_debug_output(self):
        """Clear debug output directory."""
        try:
            for file in self.debug_json_dir.glob("*.json"):
                file.unlink()
            print("[SERVER] Cleared debug output directory")
        except Exception as e:
            print(f"[SERVER] Warning: Could not clear debug output: {e}")

    def _save_json_output(self, message_type: str, message: Dict[str, Any]):
        """Save JSON output to file for debugging."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            filename = f"{message_type}_{timestamp}.json"
            filepath = self.debug_json_dir / filename
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(message, f, indent=2, ensure_ascii=False)
            
            print(f"[DEBUG] Saved output to {filename}")
            
        except Exception as e:
            print(f"[DEBUG] Error saving debug output: {e}")

    async def _safe_send_message(self, websocket, message: Dict[str, Any]):
        """Safely send message to WebSocket."""
        try:
            # Check if connection is open using the new API
            if hasattr(websocket, 'state'):
                from websockets.protocol import State
                if websocket.state == State.OPEN:
                    await websocket.send(json.dumps(message))
                    return True
                else:
                    print(f"[SERVER] Cannot send message - WebSocket state: {websocket.state}")
                    return False
            else:
                # Fallback - just try to send
                await websocket.send(json.dumps(message))
                return True
        except Exception as e:
            print(f"[SERVER] Error sending message: {e}")
            return False

    def process_audio(self, audio_array: np.ndarray, sample_rate: int) -> dict:
        """Process audio using diarization only (faster processing)."""
        try:
            print(f"[SERVER] PROCESSING AUDIO: {len(audio_array)} samples at {sample_rate}Hz")
            print(f"[SERVER] Using diarization pipeline for faster processing")
            
            # Clean up memory before processing
            self._cleanup_memory()
            
            # Use only diarization pipeline (includes speaker detection and transcription)
            print("[SERVER] Running diarization pipeline...")
            diarization_result = self.diarization_pipeline.process_audio_array(audio_array, sample_rate)
            print(f"[SERVER] Diarization result: {diarization_result}")
            print(f"[SERVER] Processing method: {diarization_result.get('processing_method', 'unknown') if diarization_result else 'None'}")
            
            if not diarization_result or 'segments' not in diarization_result:
                print("[SERVER] Diarization failed!")
                return {
                    'segments': [],
                    'total_duration': len(audio_array) / sample_rate,
                    'processing_method': 'diarization_failed',
                    'error': 'Diarization pipeline failed'
                }
            
            segments = diarization_result.get('segments', [])
            print(f"[SERVER] Diarization found {len(segments)} segments")
            
            if not segments:
                print("[SERVER] No segments found by diarization!")
                return {
                    'segments': [],
                    'total_duration': len(audio_array) / sample_rate,
                    'processing_method': 'no_segments',
                    'error': 'No speech segments detected'
                }
            
            # Process segments directly from diarization (no need for additional speaker recognition)
            print("[SERVER] Processing segments from diarization...")
            processed_segments = []
            
            for i, segment in enumerate(segments):
                print(f"[SERVER] Processing segment {i+1}: {segment}")
                
                # Get data directly from diarization result
                speaker_id = segment.get('speaker_id', f'SPEAKER_{i:02d}')
                text = segment.get('transcription', segment.get('text', ''))
                
                # Check if this speaker should be excluded (voice exclusion feature)
                should_exclude = self._should_exclude_speaker(speaker_id, audio_array, sample_rate)
                if should_exclude:
                    print(f"[SERVER] Excluding segment from registered user voice: {speaker_id}")
                    continue
                
                processed_segments.append({
                    'speaker_id': speaker_id,
                    'text': text,
                    'start': segment.get('start', 0.0),
                    'end': segment.get('end', 0.0),
                    'duration': segment.get('duration', 0.0),
                    'confidence': segment.get('confidence', 0.8)
                })
                
                print(f"[SERVER] Segment {i+1}: {speaker_id} - '{text}'")
            
            result = {
                'segments': processed_segments,
                'total_duration': len(audio_array) / sample_rate,
                'processing_method': 'diarization',
                'speaker_count': len(set(seg['speaker_id'] for seg in processed_segments))
            }
            
            print(f"[SERVER] PROCESSING COMPLETE: {len(processed_segments)} segments, {result['speaker_count']} speakers")
            print(f"[SERVER] Final processing method: {result['processing_method']}")
            
            # Clean up memory after processing
            self._cleanup_memory()
            
            return result
            
        except Exception as e:
            print(f"[SERVER] Error in processing: {e}")
            import traceback
            print(f"[SERVER] Traceback: {traceback.format_exc()}")
            return {
                'segments': [],
                'total_duration': len(audio_array) / sample_rate,
                'processing_method': 'error',
                'error': str(e)
            }


    async def handle_websocket(self, websocket):
        """Handle WebSocket connections."""
        print(f"[SERVER] New connection from {websocket.remote_address}")
        self.active_connections.add(websocket)
        
        # No keep-alive needed - connection stays alive until user disconnects
        
        try:
            async for message in websocket:
                try:
                    data = json.loads(message)
                    message_type = data.get("type")
                    
                    print(f"[SERVER] Received: {message_type}")
                    
                    if message_type == "join_conversation":
                        # Send confirmation
                        response = {
                            "type": "conversation_joined",
                            "status_code": 200,
                            "message": "Successfully joined conversation",
                            "timestamp": time.time()
                        }
                        await self._safe_send_message(websocket, response)
                        print("[SERVER] Client joined conversation")
                        
                    elif message_type == "reset_session":
                        # Handle session reset from glasses
                        print("[SERVER] Session reset requested by glasses")
                        response = {
                            "type": "session_reset",
                            "status_code": 200,
                            "message": "Session reset successfully",
                            "timestamp": time.time()
                        }
                        await self._safe_send_message(websocket, response)
                        print("[SERVER] Session reset completed")
                        
                    elif message_type == "audio_from_glasses":
                        # Process audio data from glasses
                        chunk_id = data.get("chunk_id", "unknown")
                        audio_data = data.get("audio_data", "")
                        sample_rate = data.get("sample_rate", 16000)
                        
                        print(f"\n{'='*60}")
                        print(f"[SERVER] AUDIO PROCESSING STARTED")
                        print(f"[SERVER] Chunk ID: {chunk_id}")
                        print(f"[SERVER] Sample Rate: {sample_rate} Hz")
                        print(f"[SERVER] Audio Data Length: {len(audio_data)} characters")
                        print(f"{'='*60}")
                        
                        # Send audio received confirmation
                        received_response = {
                            "type": "audio_received",
                            "chunk_id": chunk_id,
                            "status_code": 200,
                            "timestamp": time.time()
                        }
                        await self._safe_send_message(websocket, received_response)
                        print("[SERVER] Sent audio received confirmation")
                        
                        # Decode audio
                        print("[SERVER] Decoding base64 audio data...")
                        try:
                            audio_bytes = base64.b64decode(audio_data)
                            print(f"[SERVER] Decoded audio bytes: {len(audio_bytes)} bytes")
                            
                            audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                            print(f"[SERVER] Converted to float32 array: {len(audio_array)} samples")
                            print(f"[SERVER] Audio array stats: min={np.min(audio_array):.4f}, max={np.max(audio_array):.4f}, mean={np.mean(audio_array):.4f}")
                            
                            # Save debug audio file
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                            debug_filename = f"received_audio_{chunk_id}_{timestamp}.wav"
                            debug_file_path = self.debug_json_dir / debug_filename
                            sf.write(debug_file_path, audio_array, sample_rate)
                            print(f"[SERVER] Saved debug audio: {debug_filename}")
                            
                        except Exception as e:
                            print(f"[SERVER] Error decoding audio: {e}")
                            import traceback
                            print(f"[SERVER] Traceback: {traceback.format_exc()}")
                            continue
                        
                        # Process audio
                        print("[SERVER] Starting audio processing...")
                        print(f"[SERVER] Audio array shape: {audio_array.shape}")
                        print(f"[SERVER] Audio array dtype: {audio_array.dtype}")
                        print(f"[SERVER] Audio array range: [{np.min(audio_array):.4f}, {np.max(audio_array):.4f}]")
                        
                        result = self.process_audio(audio_array, sample_rate)
                        
                        print(f"[SERVER] Processing result: {result}")
                        print(f"[SERVER] Processing method: {result.get('processing_method', 'unknown')}")
                        print(f"[SERVER] Number of segments: {len(result.get('segments', []))}")
                        
                        # Log each segment individually
                        for i, segment in enumerate(result.get('segments', [])):
                            print(f"[SERVER] Segment {i+1}: {segment}")
                        
                        # Check if we're getting the expected processing method
                        if result.get('processing_method') != 'diarization':
                            print(f"[SERVER] WARNING: Unexpected processing method: {result.get('processing_method')}")
                            print(f"[SERVER] Expected: diarization")
                            print(f"[SERVER] This indicates the processing is not working correctly!")
                        
                        # Save debug output
                        debug_result = {
                            "chunk_id": chunk_id,
                            "processing_result": result,
                            "timestamp": datetime.now().isoformat()
                        }
                        self._save_json_output("processing_result", debug_result)
                        
                        # Send results
                        if result.get('segments'):
                            for i, segment in enumerate(result['segments']):
                                segment_data = {
                                    "type": "segment_result",
                                    "chunk_id": chunk_id,
                                    "segment": segment,
                                    "timestamp": time.time()
                                }
                                
                                # Save debug output
                                self._save_json_output("segment_result", segment_data)
                                
                                # Send to client
                                await self._safe_send_message(websocket, segment_data)
                                print(f"[SERVER] Sent segment {i+1}: {segment.get('text', '')[:50]}")
                        
                        # Send completion message
                        completion_data = {
                            "type": "audio_processed",
                            "chunk_id": chunk_id,
                            "total_segments": len(result.get('segments', [])),
                            "timestamp": time.time()
                        }
                        self._save_json_output("completion", completion_data)
                        await self._safe_send_message(websocket, completion_data)
                        print(f"[SERVER] Audio processing completed for {chunk_id}")
                        
                    
                    elif message_type == "register_voice":
                        # Handle voice registration
                        voice_data = data.get("voice_data", "")
                        sample_rate = data.get("sample_rate", 16000)
                        
                        print(f"[SERVER] Voice registration request received")
                        print(f"[SERVER] Voice data length: {len(voice_data)} characters")
                        
                        # Generate unique voice ID
                        voice_id = f"USER_VOICE_{int(time.time() * 1000)}"
                        
                        # Store voice data
                        self.registered_voices[voice_id] = {
                            "voice_data": voice_data,
                            "sample_rate": sample_rate,
                            "timestamp": time.time()
                        }
                        
                        print(f"[SERVER] Voice registered with ID: {voice_id}")
                        
                        # Send confirmation
                        response = {
                            "type": "voice_registered",
                            "voice_id": voice_id,
                            "status_code": 200,
                            "message": "Voice registered successfully",
                            "timestamp": time.time()
                        }
                        await self._safe_send_message(websocket, response)
                        
                    elif message_type == "set_voice_exclusion":
                        # Handle voice exclusion setting
                        exclude_voice = data.get("exclude_voice", False)
                        voice_id = data.get("voice_id", None)
                        
                        print(f"[SERVER] Voice exclusion setting: {exclude_voice}")
                        if voice_id:
                            print(f"[SERVER] Voice ID to exclude: {voice_id}")
                            self.user_voice_exclusions[websocket] = voice_id if exclude_voice else None
                        else:
                            self.user_voice_exclusions[websocket] = None
                        
                        response = {
                            "type": "voice_exclusion_set",
                            "exclude_voice": exclude_voice,
                            "status_code": 200,
                            "timestamp": time.time()
                        }
                        await self._safe_send_message(websocket, response)
                        
                    elif message_type == "ping":
                        # Respond to ping
                        response = {
                            "type": "pong",
                            "status_code": 200,
                            "timestamp": time.time()
                        }
                        await self._safe_send_message(websocket, response)
                        
                    else:
                        print(f"[SERVER] Unknown message type: {message_type}")
                        
                except json.JSONDecodeError as e:
                    print(f"[SERVER] JSON decode error: {e}")
                except Exception as e:
                    print(f"[SERVER] Error handling message: {e}")
                    
        except websockets.exceptions.ConnectionClosed:
            print(f"[SERVER] Connection closed: {websocket.remote_address}")
        except Exception as e:
            print(f"[SERVER] WebSocket error: {e}")
        finally:
            self.active_connections.discard(websocket)

    async def start_server(self):
        """Start the WebSocket server."""
        print(f"[SERVER] Starting server on {self.host}:{self.port}")
        
        self.main_loop = asyncio.get_event_loop()
        
        async with websockets.serve(
            self.handle_websocket,
            self.host,
            self.port,
            ping_interval=None,    # Disable automatic pings - connection stays alive until user disconnects
            ping_timeout=None,     # No ping timeout
            close_timeout=30,      # 30 second timeout for close handshake
            max_size=10**7,        # 10MB max message size (for larger audio files)
            max_queue=128,         # More queued messages
            # Additional timeout settings:
            open_timeout=30,       # 30 seconds to complete opening handshake
            logger=None            # Disable internal logging for cleaner output
        ):
            print(f"[SERVER] Server running on ws://{self.host}:{self.port}")
            print("[SERVER] Press Ctrl+C to stop")
            await asyncio.Future()  # Run forever

def main():
    """Main function."""
    print("=" * 60)
    print("AR GLASSES WEBSOCKET SERVER")
    print("Speaker recognition with diarization")
    print("=" * 60)
    
    server = ARGlassesServer()
    
    try:
        asyncio.run(server.start_server())
    except KeyboardInterrupt:
        print("\n[SERVER] Shutting down...")
    except Exception as e:
        print(f"[SERVER] Error: {e}")

if __name__ == "__main__":
    main()
