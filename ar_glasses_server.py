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
from websockets.server import serve
import torch

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from speaker_recognition import OptimizedSpeakerRecognition
from speaker_diarization import OptimizedDiarizationPipeline

class ARGlassesServer:
    def __init__(self):
        """Initialize the AR Glasses server with speaker recognition."""
        print("[SERVER] Initializing AR Glasses Server...")
        
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f"[SERVER] Using device: {self.device}")
        
        self.load_environment()
        
        self.hf_token = os.getenv("HF_TOKEN")
        if not self.hf_token:
            print("[SERVER] Warning: HF_TOKEN not found in environment. Using fallback models.")
            self.hf_token = "hf_your_token_here"
        
        print("[SERVER] Loading diarization model...")
        print(f"[SERVER] Using HF_TOKEN: {self.hf_token[:10]}..." if self.hf_token else "[SERVER] No HF_TOKEN provided")
        
        try:
            # Initialize only diarization pipeline for faster processing
            print("[SERVER] Initializing Optimized Diarization Pipeline...")
            self.diarization_pipeline = OptimizedDiarizationPipeline(self.hf_token)
            print("[SERVER] Diarization pipeline loaded successfully")
            
            print("[SERVER] Model loaded successfully!")
            print("[SERVER] Using optimized diarization only for faster processing")
            
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
        
        self.debug_json_dir = Path("debug_json_output")
        self.debug_json_dir.mkdir(exist_ok=True)
        self._clear_debug_output()
        
        print("[SERVER] Server initialized successfully")


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
            await websocket.send(json.dumps(message))
        except Exception as e:
            print(f"[SERVER] Error sending message: {e}")

    def process_audio(self, audio_array: np.ndarray, sample_rate: int) -> dict:
        """Process audio using optimized diarization only (faster processing)."""
        try:
            print(f"[SERVER] PROCESSING AUDIO: {len(audio_array)} samples at {sample_rate}Hz")
            print(f"[SERVER] Using optimized diarization pipeline for faster processing")
            
            # Use only diarization pipeline (includes speaker detection and transcription)
            print("[SERVER] Running optimized diarization pipeline...")
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
                'processing_method': 'optimized_diarization',
                'speaker_count': len(set(seg['speaker_id'] for seg in processed_segments))
            }
            
            print(f"[SERVER] PROCESSING COMPLETE: {len(processed_segments)} segments, {result['speaker_count']} speakers")
            print(f"[SERVER] Final processing method: {result['processing_method']}")
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


    async def handle_websocket(self, websocket, path):
        """Handle WebSocket connections."""
        print(f"[SERVER] New connection from {websocket.remote_address}")
        self.active_connections.add(websocket)
        
        # Send periodic keep-alive messages more frequently
        async def keep_alive():
            while websocket.open:
                try:
                    await asyncio.sleep(10)  # Send every 10 seconds (very frequent)
                    if websocket.open:
                        await self._safe_send_message(websocket, {
                            "type": "keep_alive",
                            "timestamp": time.time()
                        })
                        print(f"[SERVER] Sent keep-alive to {websocket.remote_address}")
                except Exception as e:
                    print(f"[SERVER] Keep-alive error: {e}")
                    break
        
        # Start keep-alive task
        keep_alive_task = asyncio.create_task(keep_alive())
        
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
                        if result.get('processing_method') != 'optimized_diarization':
                            print(f"[SERVER] WARNING: Unexpected processing method: {result.get('processing_method')}")
                            print(f"[SERVER] Expected: optimized_diarization")
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
            # Cancel keep-alive task
            keep_alive_task.cancel()
            try:
                await keep_alive_task
            except asyncio.CancelledError:
                pass
            self.active_connections.discard(websocket)

    async def start_server(self):
        """Start the WebSocket server."""
        print(f"[SERVER] Starting server on {self.host}:{self.port}")
        
        self.main_loop = asyncio.get_event_loop()
        
        async with serve(
            self.handle_websocket,
            self.host,
            self.port,
            ping_interval=10,  # More aggressive ping every 10 seconds
            ping_timeout=30,   # 30 second timeout
            close_timeout=10,  # Faster close timeout
            max_size=2**20,    # 1MB max message size
            max_queue=32       # Max queued messages
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
    
    # Start health check server on port 8080
    health_server = HTTPServer(('0.0.0.0', 8080), HealthHandler)
    health_thread = threading.Thread(target=health_server.serve_forever, daemon=True)
    health_thread.start()
    print("[SERVER] Health check server running on port 8080")
    
    server = ARGlassesServer()
    
    try:
        asyncio.run(server.start_server())
    except KeyboardInterrupt:
        print("\n[SERVER] Shutting down...")
        health_server.shutdown()
    except Exception as e:
        print(f"[SERVER] Error: {e}")
        health_server.shutdown()

if __name__ == "__main__":
    main()
