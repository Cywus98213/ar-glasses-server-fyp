#!/usr/bin/env python3
"""
AR Glasses WebSocket Server - Lightweight Version
Basic audio processing without heavy AI models for Railway deployment
"""

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
import http.server
import socketserver
import threading

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

class HealthHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/health':
            self.send_response(200)
            self.send_header('Content-type', 'text/plain')
            self.end_headers()
            self.wfile.write(b'OK')
        else:
            self.send_response(404)
            self.end_headers()
    
    def log_message(self, format, *args):
        pass  # Suppress logs

class ARGlassesServerLite:
    def __init__(self):
        """Initialize the lightweight AR Glasses server."""
        print("[SERVER] Initializing AR Glasses Server (Lite Version)...")
        
        self.load_environment()
        
        self.host = os.getenv("SERVER_HOST", "0.0.0.0")
        self.port = int(os.getenv("SERVER_PORT", "8000"))
        self.active_connections = set()
        
        self.debug_json_dir = Path("debug_json_output")
        self.debug_json_dir.mkdir(exist_ok=True)
        
        print("[SERVER] Lite server initialized successfully")
        print("[SERVER] Note: This is a lightweight version without AI models")

    def load_environment(self):
        """Load environment variables from config.env file."""
        env_file = "config.env"
        if os.path.exists(env_file):
            print(f"[SERVER] Loading environment from {env_file}")
            with open(env_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        if key.strip() not in os.environ:
                            os.environ[key.strip()] = value.strip()
            print("[SERVER] Environment variables loaded successfully")
        else:
            print(f"[SERVER] Warning: {env_file} not found, using system environment variables")

    def process_audio_lite(self, audio_array: np.ndarray, sample_rate: int) -> dict:
        """Basic audio processing without AI models."""
        try:
            print(f"[SERVER] Processing audio: {len(audio_array)} samples at {sample_rate}Hz")
            
            # Basic audio analysis
            duration = len(audio_array) / sample_rate
            rms_energy = np.sqrt(np.mean(audio_array**2))
            
            # Simulate speaker detection (placeholder)
            segments = []
            if duration > 0.5:  # Only process if audio is long enough
                segments.append({
                    'speaker_id': 'SPEAKER_00',
                    'text': f'[Audio detected: {duration:.1f}s, Energy: {rms_energy:.3f}]',
                    'start': 0.0,
                    'end': duration,
                    'duration': duration,
                    'confidence': 0.8
                })
            
            result = {
                'segments': segments,
                'total_duration': duration,
                'processing_method': 'lite_version',
                'speaker_count': len(segments)
            }
            
            print(f"[SERVER] Lite processing complete: {len(segments)} segments")
            return result
            
        except Exception as e:
            print(f"[SERVER] Error in lite processing: {e}")
            return {
                'segments': [],
                'total_duration': len(audio_array) / sample_rate,
                'processing_method': 'lite_error',
                'error': str(e)
            }

    async def handle_websocket(self, websocket, path):
        """Handle WebSocket connections."""
        print(f"[SERVER] New connection from {websocket.remote_address}")
        self.active_connections.add(websocket)
        
        try:
            async for message in websocket:
                try:
                    data = json.loads(message)
                    message_type = data.get("type")
                    
                    print(f"[SERVER] Received: {message_type}")
                    
                    if message_type == "join_conversation":
                        response = {
                            "type": "conversation_joined",
                            "status_code": 200,
                            "message": "Successfully joined conversation (Lite Version)",
                            "timestamp": time.time()
                        }
                        await websocket.send(json.dumps(response))
                        
                    elif message_type == "audio_from_glasses":
                        chunk_id = data.get("chunk_id", "unknown")
                        audio_data = data.get("audio_data", "")
                        sample_rate = data.get("sample_rate", 16000)
                        
                        print(f"[SERVER] Processing audio chunk: {chunk_id}")
                        
                        # Send received confirmation
                        await websocket.send(json.dumps({
                            "type": "audio_received",
                            "chunk_id": chunk_id,
                            "status_code": 200,
                            "timestamp": time.time()
                        }))
                        
                        # Decode and process audio
                        try:
                            audio_bytes = base64.b64decode(audio_data)
                            audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                            
                            result = self.process_audio_lite(audio_array, sample_rate)
                            
                            # Send results
                            for i, segment in enumerate(result['segments']):
                                await websocket.send(json.dumps({
                                    "type": "segment_result",
                                    "chunk_id": chunk_id,
                                    "segment": segment,
                                    "timestamp": time.time()
                                }))
                            
                            # Send completion
                            await websocket.send(json.dumps({
                                "type": "audio_processed",
                                "chunk_id": chunk_id,
                                "total_segments": len(result['segments']),
                                "timestamp": time.time()
                            }))
                            
                        except Exception as e:
                            print(f"[SERVER] Audio processing error: {e}")
                            await websocket.send(json.dumps({
                                "type": "error",
                                "message": f"Audio processing failed: {e}",
                                "timestamp": time.time()
                            }))
                    
                    elif message_type == "ping":
                        await websocket.send(json.dumps({
                            "type": "pong",
                            "status_code": 200,
                            "timestamp": time.time()
                        }))
                        
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
        print(f"[SERVER] Starting lite server on {self.host}:{self.port}")
        
        async with serve(
            self.handle_websocket,
            self.host,
            self.port,
            ping_interval=30,
            ping_timeout=60,
            max_size=2**20
        ):
            print(f"[SERVER] Lite server running on ws://{self.host}:{self.port}")
            print("[SERVER] Press Ctrl+C to stop")
            await asyncio.Future()  # Run forever

def main():
    """Main function."""
    print("=" * 60)
    print("AR GLASSES WEBSOCKET SERVER - LITE VERSION")
    print("Basic audio processing for Railway deployment")
    print("=" * 60)
    
    # Start health server
    health_server = socketserver.TCPServer(("", 8080), HealthHandler)
    health_thread = threading.Thread(target=health_server.serve_forever, daemon=True)
    health_thread.start()
    print("[SERVER] Health server running on port 8080")
    
    server = ARGlassesServerLite()
    
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
