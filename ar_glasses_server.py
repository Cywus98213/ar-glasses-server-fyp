#!/usr/bin/env python3

import os
import sys
import json
import asyncio
import base64
import time
import warnings
import signal
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

# Force unbuffered output for real-time logging
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None
os.environ["PYTHONUNBUFFERED"] = "1"

from speaker_diarization import DiarizationPipeline
from gesture_recognition import GestureRecognizer

class ARGlassesServer:
    def __init__(self):
        """Initialize the AR Glasses server with speaker recognition."""
        print("[SERVER] Initializing AR Glasses Server...")
        
        # Load environment from config.env 
        self._load_env_files()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        print(f"[SERVER] Using device: {self.device}")
        
        # Load basic settings from env
        self.hf_token = os.getenv("HF_TOKEN")
        self.transcription_language = os.getenv("TRANSCRIPTION_LANGUAGE", "yue")
        self.translation_language = None  # Will be set by app request (handle_request)

        if not self.hf_token:
            print("[SERVER] Warning: HF_TOKEN not found in environment, please set it in the config.env file.")
            sys.exit(1)
        
        print(f"[SERVER] Default transcription language: {self.transcription_language}")
        print("[SERVER] Translation language will be set by app requests")
        
        print("[SERVER] Loading diarization model...")
        
        try:
            # Initialize only diarization pipeline for faster processing
            print("[SERVER] Initializing Diarization Pipeline...")

            # loading the model thru the hugging face token
            # Pass default transcription language to diarization pipeline
            # Translation language will be provided per request
            self.diarization_pipeline = DiarizationPipeline(
                self.hf_token,
                transcription_lang=self.transcription_language
            )
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
        
        # Initialize gesture recognizer
        print("[SERVER] Loading gesture recognition model...")
        try:
            self.gesture_recognizer = GestureRecognizer()
            if self.gesture_recognizer.is_available():
                print("[SERVER] Gesture recognition model loaded successfully")
            else:
                print("[SERVER] Warning: Gesture recognition not available (model not found or MediaPipe not installed)")
        except Exception as e:
            print(f"[SERVER] Warning: Could not initialize gesture recognizer: {e}")
            self.gesture_recognizer = None
        
        self.host = os.getenv("SERVER_HOST", "0.0.0.0")
        self.port = int(os.getenv("SERVER_PORT", "8080"))
        self.active_connections = set()
        
        # Track active processing tasks for parallel execution
        self.active_audio_tasks = {}  # websocket -> set of tasks
        self.active_gesture_tasks = {}  # websocket -> set of tasks
        self.pending_audio_chunks = {}  # websocket -> latest queued real-time chunk
        self.chunk_worker_tasks = {}  # websocket -> background worker task for real-time audio
        self.main_loop = None
        self.shutdown_event = asyncio.Event()  # Event to signal shutdown
        self.server = None  # WebSocket server instance
        
        # Track stop processing flag per connection to cancel queued chunks
        # Format: {websocket: bool} - True means stop processing and skip remaining chunks
        self.stop_processing_per_connection = {}
        
        # Voice registration storage - stores wearer's voice (192-dim embedding)
        # Only one voice at a time (the glasses wearer)
        self.registered_voices = {}  # voice_id -> {'embedding': np.array(192,), 'timestamp': float, 'sample_rate': int}
        
        # Speaker tracking per connection - tracks known speakers across chunks
        # Format: {websocket: {'speakers': [{'id': 'SPEAKER_00', 'embedding': np.array, 'is_wearer': bool}, ...], 'next_id': int}}
        self.speaker_tracking = {}  # Track speakers per connection for consistent IDs
        
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
    
    def _cleanup_connection(self, websocket):
        """Clean up all resources and processes for a disconnected connection."""
        print(f"[SERVER] ========== CLEANING UP CONNECTION ==========")
        print(f"[SERVER] Client: {websocket.remote_address}")
        
        # Set stop processing flag to cancel any ongoing processing
        if websocket in self.stop_processing_per_connection:
            was_processing = not self.stop_processing_per_connection[websocket]
            self.stop_processing_per_connection[websocket] = True
            if was_processing:
                print(f"[SERVER] ✓ Stopped processing for this connection")
        
        # Remove from active connections
        if websocket in self.active_connections:
            self.active_connections.discard(websocket)
            print(f"[SERVER] ✓ Removed from active connections")
        
        # Clean up stop processing flag
        if websocket in self.stop_processing_per_connection:
            del self.stop_processing_per_connection[websocket]
            print(f"[SERVER] ✓ Removed stop processing flag")
        
        # Try to cancel any pending async tasks related to this connection
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                pending_tasks = [t for t in asyncio.all_tasks(loop) if not t.done()]
                cancelled_count = 0
                for task in pending_tasks:
                    # Try to cancel tasks that might be related to this connection
                    # (This is a best-effort cleanup)
                    try:
                        if not task.done():
                            task.cancel()
                            cancelled_count += 1
                    except Exception:
                        pass
                if cancelled_count > 0:
                    print(f"[SERVER] ✓ Cancelled {cancelled_count} pending async task(s)")
        except Exception as e:
            print(f"[SERVER] Note: Could not cancel async tasks: {e}")
        
        # Clean up memory
        self._cleanup_memory()
        print(f"[SERVER] ✓ Memory cleaned up")
        
        print(f"[SERVER] ===========================================")

    def _log_remaining_processes(self, websocket):
        """Log all remaining processes and state when a connection disconnects."""
        print(f"[SERVER] ========== REMAINING PROCESSES STATUS ==========")
        print(f"[SERVER] Active connections: {len(self.active_connections)}")
        for conn in self.active_connections:
            if conn != websocket:
                print(f"[SERVER]   - Active: {conn.remote_address}")
        
        print(f"[SERVER] Stop processing flags: {len(self.stop_processing_per_connection)}")
        for conn, stop_flag in self.stop_processing_per_connection.items():
            if conn != websocket:
                print(f"[SERVER]   - {conn.remote_address}: stop_flag={stop_flag}")
        
        print(f"[SERVER] Registered voices: {len(self.registered_voices)}")
        if self.registered_voices:
            for voice_id, voice_data in self.registered_voices.items():
                num_samples = voice_data.get('num_samples', 1)
                timestamp = voice_data.get('timestamp', 0)
                age_seconds = time.time() - timestamp if timestamp > 0 else 0
                print(f"[SERVER]   - {voice_id}: {num_samples} sample(s), age: {age_seconds:.1f}s")
        else:
            print(f"[SERVER]   - No registered voices")
        
        # Check for any pending tasks in event loop
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                pending_tasks = [t for t in asyncio.all_tasks(loop) if not t.done()]
                print(f"[SERVER] Pending async tasks: {len(pending_tasks)}")
                for i, task in enumerate(pending_tasks[:10]):  # Show first 10
                    print(f"[SERVER]   - Task {i+1}: {task.get_name() if hasattr(task, 'get_name') else str(task)}")
                if len(pending_tasks) > 10:
                    print(f"[SERVER]   - ... and {len(pending_tasks) - 10} more tasks")
        except Exception as e:
            print(f"[SERVER] Could not check async tasks: {e}")
        
        # Memory status
        import gc
        import sys
        print(f"[SERVER] Memory status:")
        print(f"[SERVER]   - Python objects: {len(gc.get_objects())}")
        if torch.cuda.is_available():
            print(f"[SERVER]   - CUDA memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
            print(f"[SERVER]   - CUDA memory reserved: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
        
        print(f"[SERVER] =================================================")

    def _clear_debug_output(self):
        """Clear debug output directory (both JSON and WAV files)."""
        try:
            # Clear JSON files
            json_count = 0
            for file in self.debug_json_dir.glob("*.json"):
                file.unlink()
                json_count += 1
            
            # Clear WAV files
            wav_count = 0
            for file in self.debug_json_dir.glob("*.wav"):
                file.unlink()
                wav_count += 1
            
            if json_count > 0 or wav_count > 0:
                print(f"[SERVER] Cleared debug output directory ({json_count} JSON, {wav_count} WAV files)")
        except Exception as e:
            print(f"[SERVER] Warning: Could not clear debug output: {e}")

    def _load_env_files(self):
        """Load environment from config.env"""
        if os.path.exists('config.env'):
            with open('config.env', 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        os.environ[key.strip()] = value.strip()
            print("[SERVER] Environment variables loaded successfully")
        else:
            print("[SERVER] Warning: config.env not found, using OS environment only")

    def _save_json_output(self, message_type: str, message: Dict[str, Any]):
        """Save JSON output to file for debugging. Clears old files to prevent accumulation."""
        try:
            # Clear old debug files before saving new one (keep only recent)
            self._clear_old_debug_files()
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            filename = f"{message_type}_{timestamp}.json"
            filepath = self.debug_json_dir / filename
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(message, f, indent=2, ensure_ascii=False)
            
            # Only log for important messages to reduce spam
            if message_type in ["processing_result", "segment_result"]:
                print(f"[DEBUG] Saved output to {filename}")
            
        except Exception as e:
            print(f"[DEBUG] Error saving debug output: {e}")
    
    def _clear_old_debug_files(self, keep_recent: int = 10):
        """Clear old debug files (JSON and WAV), keeping only the most recent ones."""
        try:
            # Get all JSON files sorted by modification time (newest first)
            json_files = sorted(
                self.debug_json_dir.glob("*.json"),
                key=lambda f: f.stat().st_mtime,
                reverse=True
            )
            
            # Get all WAV files sorted by modification time (newest first)
            wav_files = sorted(
                self.debug_json_dir.glob("*.wav"),
                key=lambda f: f.stat().st_mtime,
                reverse=True
            )
            
            # Keep only the most recent JSON files, delete the rest
            if len(json_files) > keep_recent:
                files_to_delete = json_files[keep_recent:]
                for file in files_to_delete:
                    file.unlink()
                if len(files_to_delete) > 0:
                    print(f"[DEBUG] Cleared {len(files_to_delete)} old JSON files (kept {keep_recent} most recent)")
            
            # Keep only the most recent WAV files, delete the rest
            if len(wav_files) > keep_recent:
                files_to_delete = wav_files[keep_recent:]
                for file in files_to_delete:
                    file.unlink()
                if len(files_to_delete) > 0:
                    print(f"[DEBUG] Cleared {len(files_to_delete)} old WAV files (kept {keep_recent} most recent)")
        except Exception as e:
            print(f"[DEBUG] Warning: Could not clear old debug files: {e}")

    async def _safe_send_message(self, websocket, message: Dict[str, Any]):
        """Safely send message to WebSocket."""
        try:
            # Check if connection is open
            if hasattr(websocket, 'state'):
                from websockets.protocol import State
                if websocket.state != State.OPEN:
                    print(f"[SERVER] Cannot send message - WebSocket state: {websocket.state}")
                    return False
            
            # Try to send the message
            await websocket.send(json.dumps(message))
            return True
            
        except websockets.exceptions.ConnectionClosed:
            print(f"[SERVER] Connection closed while sending message")
            return False
        except Exception as e:
            print(f"[SERVER] Error sending message: {e}")
            return False

    def _track_audio_task(self, websocket, task: asyncio.Task):
        """Track audio-related background tasks for cleanup."""
        if websocket not in self.active_audio_tasks:
            self.active_audio_tasks[websocket] = set()
        self.active_audio_tasks[websocket].add(task)

        def _cleanup(done_task):
            tasks = self.active_audio_tasks.get(websocket)
            if tasks is None:
                return
            tasks.discard(done_task)
            if not tasks:
                del self.active_audio_tasks[websocket]

        task.add_done_callback(_cleanup)

    def _clear_pending_chunk(self, websocket):
        """Discard any queued real-time chunk for a connection."""
        self.pending_audio_chunks.pop(websocket, None)

    async def _queue_latest_chunk(self, websocket, audio_array: np.ndarray, sample_rate: int,
                                  translation_language: str, chunk_id: str):
        """Keep only the newest real-time chunk while one is already processing."""
        replaced_chunk = self.pending_audio_chunks.get(websocket)
        self.pending_audio_chunks[websocket] = {
            "audio_array": audio_array,
            "sample_rate": sample_rate,
            "translation_language": translation_language,
            "chunk_id": chunk_id,
        }

        if replaced_chunk:
            print(
                f"[SERVER] Replaced queued chunk {replaced_chunk['chunk_id']} with newer chunk {chunk_id}"
            )
            await self._safe_send_message(
                websocket,
                {
                    "type": "audio_processed",
                    "chunk_id": replaced_chunk["chunk_id"],
                    "total_segments": 0,
                    "dropped": True,
                    "timestamp": time.time(),
                },
            )

        existing_worker = self.chunk_worker_tasks.get(websocket)
        if existing_worker and not existing_worker.done():
            return

        worker = asyncio.create_task(self._process_latest_chunk_loop(websocket))
        self.chunk_worker_tasks[websocket] = worker
        self._track_audio_task(websocket, worker)

    async def _process_latest_chunk_loop(self, websocket):
        """Process real-time audio chunks serially and skip stale buffered chunks."""
        try:
            while True:
                queued_chunk = self.pending_audio_chunks.pop(websocket, None)
                if queued_chunk is None:
                    return

                await self._process_audio_async(
                    queued_chunk["audio_array"],
                    queued_chunk["sample_rate"],
                    queued_chunk["translation_language"],
                    websocket,
                    queued_chunk["chunk_id"],
                    True,
                )
        finally:
            self.chunk_worker_tasks.pop(websocket, None)

    async def _process_audio_async(self, audio_array: np.ndarray, sample_rate: int, translation_language: str, 
                                   websocket, chunk_id: str, is_chunk: bool = False):
        """Async wrapper for audio processing - runs in parallel with gesture processing."""
        try:
            # Get speaker tracking for this connection
            speaker_tracking = self.speaker_tracking.get(websocket, None)
            
            # Run CPU-intensive processing in executor to avoid blocking event loop
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, 
                self.process_audio, 
                audio_array, 
                sample_rate, 
                translation_language, 
                is_chunk,
                speaker_tracking
            )
            
            # Stop flag removed - always process and send results
            
            # Check connection after processing
            if hasattr(websocket, 'state'):
                from websockets.protocol import State
                if websocket.state != State.OPEN:
                    print(f"[SERVER] Connection closed during processing, skipping result send")
                    return
            
            # Process and send results
            segments = result.get('segments', [])
            num_segments = len(segments)
            
            # For chunks with 0 segments, skip completely silently
            if is_chunk and num_segments == 0:
                return
            
            # For full audio with 0 segments, still send completion
            if not is_chunk and num_segments == 0:
                print(f"[SERVER] No segments found in full audio (duration: {result.get('total_duration', 0):.2f}s)")
                response = {
                    "type": "no_speech",
                    "chunk_id": chunk_id,
                    "message": "No speech detected in audio",
                    "timestamp": time.time()
                }
                await self._safe_send_message(websocket, response)
                
                completion_data = {
                    "type": "audio_processed",
                    "chunk_id": chunk_id,
                    "total_segments": 0,
                    "timestamp": time.time()
                }
                await self._safe_send_message(websocket, completion_data)
                return
            
            # Log results if we have segments
            if not is_chunk:
                print(f"[SERVER] Processing method: {result.get('processing_method', 'unknown')}")
                print(f"[SERVER] Number of segments: {num_segments}")
                
                # Log each segment individually
                for i, segment in enumerate(segments):
                    speaker_marker = "[WEARER]" if segment.get('is_wearer') else "[OTHER]"
                    print(f"[SERVER] Segment {i+1}: {segment.get('speaker_id', 'UNKNOWN')} {speaker_marker} - '{segment.get('text', '')[:50]}'")
            
            # Save debug output
            debug_result = {
                "chunk_id": chunk_id,
                "processing_result": result,
                "timestamp": datetime.now().isoformat()
            }
            self._save_json_output("processing_result", debug_result)
            
            # Send all segments to client individually (like original)
            for i, segment in enumerate(segments):
                segment_data = {
                    "type": "segment_result",
                    "chunk_id": chunk_id,
                    "segment": segment,
                    "timestamp": time.time()
                }
                
                # Save debug output
                self._save_json_output("segment_result", segment_data)
                
                # Send to client (check connection for each segment)
                if hasattr(websocket, 'state'):
                    from websockets.protocol import State
                    if websocket.state != State.OPEN:
                        print(f"[SERVER] Connection closed during segment send, stopping")
                        break
                
                await self._safe_send_message(websocket, segment_data)
            
            # Send completion message (always send, no stop flag check)
            completion_data = {
                "type": "audio_processed",
                "chunk_id": chunk_id,
                "total_segments": num_segments,
                "timestamp": time.time()
            }
            self._save_json_output("completion", completion_data)
            await self._safe_send_message(websocket, completion_data)
            if not is_chunk:
                print(f"[SERVER] Audio processing completed for {chunk_id}: {num_segments} segments")
            
        except Exception as e:
            print(f"[SERVER] ERROR during async audio processing: {e}")
            import traceback
            print(f"[SERVER] Traceback: {traceback.format_exc()}")
            
            if hasattr(websocket, 'state'):
                from websockets.protocol import State
                if websocket.state == State.OPEN:
                    error_response = {
                        "type": "processing_error",
                        "chunk_id": chunk_id,
                        "error": str(e),
                        "timestamp": time.time()
                    }
                    await self._safe_send_message(websocket, error_response)
        finally:
            # Remove task from tracking
            if websocket in self.active_audio_tasks:
                self.active_audio_tasks[websocket].discard(asyncio.current_task())
    
    async def _process_gesture_async(self, image_data: str, websocket, timestamp):
        """Async wrapper for gesture processing - runs in parallel with audio processing."""
        try:
            # Run CPU-intensive processing in executor to avoid blocking event loop
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                self.gesture_recognizer.recognize_gesture_from_base64,
                image_data
            )
            
            # Check connection after processing
            if hasattr(websocket, 'state'):
                from websockets.protocol import State
                if websocket.state != State.OPEN:
                    print(f"[SERVER] Connection closed during gesture processing, skipping result send")
                    return
            
            processing_time = time.time() - timestamp
            
            if result.get('success'):
                gestures = result.get('gestures', [])
                hand_landmarks = result.get('hand_landmarks', [])
                num_hands = result.get('num_hands', 0)
                
                print(f"[SERVER] ✓ Gesture recognition successful (took {processing_time:.3f}s)")
                print(f"[SERVER]   Number of hands detected: {num_hands}")
                print(f"[SERVER]   Number of gestures detected: {len(gestures)}")
                
                if len(gestures) > 0:
                    for i, gesture in enumerate(gestures):
                        category = gesture.get('category_name', 'Unknown')
                        score = gesture.get('score', 0.0)
                        print(f"[SERVER]   Gesture {i+1}: {category} (confidence: {score:.3f} / {score*100:.1f}%)")
                else:
                    print(f"[SERVER]   No gestures detected in image")
                
                response = {
                    "type": "gesture_result",
                    "status_code": 200,
                    "gestures": gestures,
                    "hand_landmarks": hand_landmarks,
                    "num_hands": num_hands,
                    "timestamp": time.time()
                }
                
                await self._safe_send_message(websocket, response)
                print(f"[SERVER] ✓ Gesture result sent successfully")
            else:
                error_msg = result.get('error', 'Unknown error')
                print(f"[SERVER] ✗ Gesture recognition failed: {error_msg}")
                response = {
                    "type": "gesture_result",
                    "status_code": 500,
                    "error": error_msg,
                    "timestamp": time.time()
                }
                await self._safe_send_message(websocket, response)
            
        except Exception as e:
            print(f"[SERVER] ✗ EXCEPTION processing gesture: {e}")
            import traceback
            print(f"[SERVER] Traceback: {traceback.format_exc()}")
            
            if hasattr(websocket, 'state'):
                from websockets.protocol import State
                if websocket.state == State.OPEN:
                    response = {
                        "type": "gesture_result",
                        "status_code": 500,
                        "error": str(e),
                        "timestamp": time.time()
                    }
                    await self._safe_send_message(websocket, response)
        finally:
            # Remove task from tracking
            if websocket in self.active_gesture_tasks:
                self.active_gesture_tasks[websocket].discard(asyncio.current_task())
    
    def process_audio(self, audio_array: np.ndarray, sample_rate: int, translation_language: str, is_chunk: bool = False, speaker_tracking: Dict[str, Any] = None) -> dict:
        """Process audio using diarization (supports both full audio and real-time chunks).
        
        Args:
            audio_array: Audio data as numpy array
            sample_rate: Sample rate in Hz
            translation_language: Target language for translation
            is_chunk: If True, this is a real-time chunk (optimized for speed while keeping diarization)
        """
        try:
            # Only log for full audio, not chunks (will log after if segments found)
            if not is_chunk:
                mode = "FULL AUDIO"
                print(f"[SERVER] PROCESSING AUDIO ({mode}): {len(audio_array)} samples at {sample_rate}Hz")
                print(f"[SERVER] Using diarization pipeline with speaker identification")
            
            # Clean up memory before processing
            self._cleanup_memory()
            
            # Use only diarization pipeline (includes speaker detection and transcription)
            if not is_chunk:
                print("[SERVER] Running diarization pipeline...")
                print(f"[SERVER] DEBUG: self.registered_voices = {list(self.registered_voices.keys()) if self.registered_voices else 'None'}")

            # check if the wearer's voice is registered
            if not is_chunk:
                if self.registered_voices:
                    print(f"[SERVER] Wearer's voice registered - will identify wearer vs others")
                    for voice_id, voice_data in self.registered_voices.items():
                        print(f"[SERVER] DEBUG: Voice ID: {voice_id}")
                        if 'embeddings' in voice_data:
                            print(f"[SERVER] DEBUG: Multi-sample embeddings: {len(voice_data['embeddings'])} samples")
                        elif 'embedding' in voice_data:
                            print(f"[SERVER] DEBUG: Single embedding shape: {voice_data['embedding'].shape if voice_data.get('embedding') is not None else 'None'}")
                else:
                    print(f"[SERVER] No registered voice - processing all speakers normally")

            # pack the audio result with dynamic language settings
            diarization_result = self.diarization_pipeline.process_audio_array(
                audio_array, 
                sample_rate,
                registered_voices=self.registered_voices,
                transcription_lang=self.transcription_language,
                translation_lang=translation_language,
                is_chunk=is_chunk,
                speaker_tracking=speaker_tracking
            )
            # Only log diarization result details if we have segments or it's full audio
            if diarization_result and diarization_result.get('segments'):
                print(f"[SERVER] Processing method: {diarization_result.get('processing_method', 'unknown')}")
            elif not is_chunk:
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
            
            # Only log if we have segments or it's full audio (not chunks)
            if segments or not is_chunk:
                print(f"[SERVER] Diarization found {len(segments)} segments")
            
            if not segments:
                # For chunks, return silently. For full audio, log the issue.
                if not is_chunk:
                    print("[SERVER] No segments found by diarization!")
                return {
                    'segments': [],
                    'total_duration': len(audio_array) / sample_rate,
                    'processing_method': 'no_segments',
                    'error': 'No speech segments detected'
                }
            
            # Process segments directly from diarization
            print("[SERVER] Processing segments from diarization...")
            processed_segments = []
            
            for i, segment in enumerate(segments):
                print(f"[SERVER] Processing segment {i+1}: {segment}")
                
                # Get data directly from diarization result
                speaker_id = segment.get('speaker_id', f'SPEAKER_{i:02d}')
                text = segment.get('transcription', segment.get('text', ''))
                
                # Build segment data
                segment_data = {
                    'speaker_id': speaker_id,
                    'text': text,
                    'start': segment.get('start', 0.0),
                    'end': segment.get('end', 0.0),
                    'duration': segment.get('duration', 0.0),
                    'confidence': segment.get('confidence', 0.8)
                }
                
                # Add voice matching info if available (wearer identification)
                if 'is_wearer' in segment:
                    segment_data['is_wearer'] = segment.get('is_wearer')
                    segment_data['voice_similarity'] = segment.get('voice_similarity', 0.0)
                
                processed_segments.append(segment_data)
                
                # log if the segment is the wearer voice
                wearer_marker = "[WEARER]" if segment.get('is_wearer') else "[OTHER]"
                if 'is_wearer' in segment:
                    print(f"[SERVER] Segment {i+1}: {speaker_id} {wearer_marker} - '{text}'")
                else:
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
        # Initialize stop processing flag for this connection
        self.stop_processing_per_connection[websocket] = False
        # Initialize speaker tracking for this connection
        self.speaker_tracking[websocket] = {
            'speakers': [],  # List of known speakers (SPEAKER_00 reserved for wearer)
            'next_id': 1  # Next speaker ID to assign (start at 1, 0 is for wearer)
        }
        try:
            async for message in websocket:
                # Check connection state before processing any message
                if hasattr(websocket, 'state'):
                    from websockets.protocol import State
                    if websocket.state != State.OPEN:
                        print(f"[SERVER] Connection not open (state: {websocket.state}), stopping message processing")
                        break
                try:
                    data = json.loads(message)
                    message_type = data.get("type")
                    
                    print(f"[SERVER] Received: {message_type}")
                    
                    if message_type == "join_conversation":
                        # Reset stop processing flag when starting a new recording session
                        # (but NOT when stop is pressed - that's the user's requirement)
                        self.stop_processing_per_connection[websocket] = False
                        self._clear_pending_chunk(websocket)
                        print(f"[SERVER] Reset stop flag for new recording session")
                        
                        # allow client to set language at session start
                        client_translation_lang = data.get("translation_language")
                        
                        if client_translation_lang:
                            self.translation_language = client_translation_lang
                            print(f"[SERVER] Translation language set by client: {self.translation_language}")
                        
                        response = {
                            "type": "conversation_joined",
                            "status_code": 200,
                            "message": "Successfully joined conversation",
                            "translation_language": self.translation_language,
                            "timestamp": time.time()
                        }
                        await self._safe_send_message(websocket, response)
                        print("[SERVER] Client joined conversation")
                        
                    elif message_type == "reset_session":
                        # Handle session reset from glasses
                        print("[SERVER] Session reset requested by glasses")
                        self._clear_pending_chunk(websocket)
                        self.speaker_tracking[websocket] = {
                            'speakers': [],
                            'next_id': 1
                        }
                        response = {
                            "type": "session_reset",
                            "status_code": 200,
                            "message": "Session reset successfully",
                            "timestamp": time.time()
                        }
                        await self._safe_send_message(websocket, response)
                        print("[SERVER] Session reset completed")
                        
                    elif message_type == "stop_processing":
                        # Client requested to stop processing and dump remaining chunks
                        print(f"[SERVER] ========== STOP PROCESSING REQUESTED ==========")
                        print(f"[SERVER] Received stop_processing message from client")
                        print(f"[SERVER] Setting stop flag - will skip remaining queued chunks and discard results")
                        
                        # Set stop flag immediately
                        self.stop_processing_per_connection[websocket] = True
                        self._clear_pending_chunk(websocket)
                        
                        # Log current state
                        print(f"[SERVER] Stop flag set to: {self.stop_processing_per_connection.get(websocket, False)}")
                        print(f"[SERVER] Active connections: {len(self.active_connections)}")
                        
                        # Send confirmation
                        response = {
                            "type": "processing_stopped",
                            "status_code": 200,
                            "message": "Processing stopped - remaining chunks will be skipped",
                            "timestamp": time.time()
                        }
                        sent = await self._safe_send_message(websocket, response)
                        if sent:
                            print(f"[SERVER] Stop processing confirmation sent to client")
                        else:
                            print(f"[SERVER] WARNING: Failed to send stop processing confirmation")
                        
                        print(f"[SERVER] Stop processing flag set for connection - future chunks will be skipped")
                        
                    elif message_type == "audio_from_glasses":
                        # Process audio data from glasses (always process, no stop flag check)
                        chunk_id = data.get("chunk_id", "unknown")
                        audio_data = data.get("audio_data", "")
                        sample_rate = data.get("sample_rate", 16000)
                        translation_language = data.get("translation_language", None)
                        is_chunk = data.get("is_chunk", False)  # Detect if this is a real-time chunk
                        
                        # Only log initial info for full audio, not chunks (to reduce noise)
                        if not is_chunk:
                            print(f"\n{'='*60}")
                            print(f"[SERVER] AUDIO PROCESSING STARTED (FULL)")
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
                        
                        if not is_chunk:
                            print("[SERVER] Sent audio received confirmation")
                        
                        # Decode audio
                        if not is_chunk:
                            print("[SERVER] Decoding base64 audio data...")
                        try:
                            audio_bytes = base64.b64decode(audio_data)
                            
                            if not is_chunk:
                                print(f"[SERVER] Decoded audio bytes: {len(audio_bytes)} bytes")
                            
                            audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                            
                            # Note: No overlap trimming needed
                            # Client sends chunks without overlap for real-time processing
                            # Smart deduplication handles any edge cases (filters duplicates but allows legitimate similar speech from different speakers)
                            
                            if not is_chunk:
                                print(f"[SERVER] Converted to float32 array: {len(audio_array)} samples")
                                print(f"[SERVER] Audio array stats: min={np.min(audio_array):.4f}, max={np.max(audio_array):.4f}, mean={np.mean(audio_array):.4f}")
                            
                            # Save debug audio file (only for full audio, not chunks to reduce storage)
                            if not is_chunk:
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
                        
                        # Check connection state before starting processing
                        if hasattr(websocket, 'state'):
                            from websockets.protocol import State
                            if websocket.state != State.OPEN:
                                print(f"[SERVER] Connection closed before processing, skipping audio")
                                continue
                        
                        # Send processing status to keep connection alive
                        processing_status_msg = {
                            "type": "processing_status",
                            "chunk_id": chunk_id,
                            "status": "processing",
                            "timestamp": time.time()
                        }
                        sent = await self._safe_send_message(websocket, processing_status_msg)
                        if not sent:
                            print(f"[SERVER] Failed to send processing_status, connection may be closed")
                            continue
                        
                        # Process audio (silent for chunks until we know if there are segments)
                        # (no stop flag check - always process remaining chunks)
                        if not is_chunk:
                            print("[SERVER] Starting audio processing...")
                            print(f"[SERVER] Audio array shape: {audio_array.shape}")
                            print(f"[SERVER] Audio array dtype: {audio_array.dtype}")
                            print(f"[SERVER] Audio array range: [{np.min(audio_array):.4f}, {np.max(audio_array):.4f}]")
                            print(f"[SERVER] Using transcription language: {self.transcription_language}")
                            print(f"[SERVER] Using translation language: {translation_language or 'None'}")
                            print(f"[SERVER] Processing mode: FULL AUDIO")
                        
                        # Process audio in parallel using async task (non-blocking)
                        # (no stop flag check - always process remaining chunks)
                        # This allows gesture processing to run concurrently
                        # For real-time chunks, process immediately with high priority
                        if is_chunk:
                            print(f"[SERVER] Starting real-time processing for chunk {chunk_id}...")
                        
                        if is_chunk:
                            await self._queue_latest_chunk(
                                websocket,
                                audio_array,
                                sample_rate,
                                translation_language,
                                chunk_id
                            )
                        else:
                            task = asyncio.create_task(
                                self._process_audio_async(
                                    audio_array, 
                                    sample_rate, 
                                    translation_language, 
                                    websocket, 
                                    chunk_id, 
                                    is_chunk
                                )
                            )
                            self._track_audio_task(websocket, task)
                        
                        # Don't await - let it run in parallel with other messages
                        # The task will handle sending results when done
                        # For chunks, processing starts immediately
                        
                    
                    elif message_type == "register_voice":
                        # Register the wearer's voice 

                        # Clear any previously registered voice
                        self.registered_voices.clear()

                        voice_data = data.get("voice_data", "")
                        sample_rate = data.get("sample_rate", 16000)
                        try:
                            # Decode audio from base64
                            print("[SERVER] ==========================================")
                            print("[SERVER] Registering WEARER's voice with MULTI-SAMPLE...")
                            print("[SERVER] Decoding audio...")
                            audio_bytes = base64.b64decode(voice_data)
                            audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                            duration_sec = len(audio_array) / sample_rate
                            print(f"[SERVER] Decoded {len(audio_array)} samples ({duration_sec:.2f}s)")
                            
                            # Extract MULTIPLE 192-dim embeddings from segments
                            embeddings = self.diarization_pipeline.extract_multiple_embeddings(audio_array, sample_rate)
                            
                            if embeddings is None or len(embeddings) == 0:
                                print(f"[SERVER] ERROR: Failed to extract valid embeddings!")
                                response = {
                                    "type": "voice_registered",
                                    "status_code": 500,
                                    "message": "Failed to extract speaker embeddings",
                                    "timestamp": time.time()
                                }
                                await self._safe_send_message(websocket, response)
                                continue
                            
                            # Validate all embeddings
                            valid_embeddings = [emb for emb in embeddings if emb.shape[0] == 192]
                            
                            if len(valid_embeddings) == 0:
                                print(f"[SERVER] ERROR: No valid 192-dim embeddings!")
                                response = {
                                    "type": "voice_registered",
                                    "status_code": 500,
                                    "message": "No valid embeddings extracted",
                                    "timestamp": time.time()
                                }
                                await self._safe_send_message(websocket, response)
                                continue
                            
                            # wearer id
                            voice_id = f"WEARER_VOICE_{int(time.time() * 1000)}"
                            
                            # Build clean registered voice profile (filters outliers, builds centroid)
                            voice_profile = self.diarization_pipeline.build_registered_voice_profile(valid_embeddings)
                            profile_embeddings = voice_profile["embeddings"]
                            profile_embedding = voice_profile["profile_embedding"]
                            discarded = voice_profile["discarded_embeddings"]
                            
                            # Save the wearer's voice profile
                            self.registered_voices[voice_id] = {
                                "embeddings": profile_embeddings,
                                "profile_embedding": profile_embedding,
                                "num_samples": len(profile_embeddings),
                                "discarded_embeddings": discarded,
                                "timestamp": time.time(),
                                "sample_rate": sample_rate
                            }

                            
                            print(f"[SERVER] ✓✓ Wearer's voice registered with MULTI-SAMPLE!")
                            print(f"[SERVER] Voice ID: {voice_id}")
                            print(f"[SERVER] Number of samples: {len(profile_embeddings)} (discarded {discarded} outliers)")
                            print(f"[SERVER] Each embedding shape: (192,)")
                            print(f"[SERVER] Registration method: Multi-sample with profile centroid")
                            print(f"[SERVER] DEBUG: self.registered_voices after registration = {list(self.registered_voices.keys())}")
                            print(f"[SERVER] Will identify wearer using profile-based matching")
                            print(f"[SERVER] ==========================================")
                            
                            # Send confirmation
                            response = {
                                "type": "voice_registered",
                                "voice_id": voice_id,
                                "status_code": 200,
                                "message": f"Wearer's voice registered with {len(profile_embeddings)} samples (profile-based)",
                                "num_samples": len(profile_embeddings),
                                "registration_method": "profile-based",
                                "timestamp": time.time()
                            }
                            await self._safe_send_message(websocket, response)
                            
                        except Exception as e:
                            print(f"[SERVER] ERROR during voice registration: {e}")
                            import traceback
                            print(f"[SERVER] Traceback: {traceback.format_exc()}")
                            response = {
                                "type": "voice_registered",
                                "status_code": 500,
                                "message": f"Registration failed: {str(e)}",
                                "timestamp": time.time()
                            }
                            await self._safe_send_message(websocket, response)
                        
                    elif message_type == "gesture_from_glasses":
                        # Process gesture recognition request
                        print(f"[SERVER] ========== GESTURE RECOGNITION REQUEST ==========")
                        print(f"[SERVER] Client: {websocket.remote_address}")
                        print(f"[SERVER] Timestamp: {data.get('timestamp', 'N/A')}")
                        
                        if not self.gesture_recognizer or not self.gesture_recognizer.is_available():
                            print(f"[SERVER] ERROR: Gesture recognition not available")
                            response = {
                                "type": "gesture_result",
                                "status_code": 503,
                                "error": "Gesture recognition not available",
                                "timestamp": time.time()
                            }
                            await self._safe_send_message(websocket, response)
                            continue
                        
                        image_data = data.get("image_data", "")
                        if not image_data:
                            print(f"[SERVER] ERROR: No image data provided")
                            response = {
                                "type": "gesture_result",
                                "status_code": 400,
                                "error": "No image data provided",
                                "timestamp": time.time()
                            }
                            await self._safe_send_message(websocket, response)
                            continue
                        
                        print(f"[SERVER] Image data received: {len(image_data)} characters (base64)")
                        
                        # Process gesture in parallel using async task (non-blocking)
                        # This allows audio processing to run concurrently
                        request_timestamp = time.time()
                        task = asyncio.create_task(
                            self._process_gesture_async(image_data, websocket, request_timestamp)
                        )
                        # Track task for cleanup
                        if websocket not in self.active_gesture_tasks:
                            self.active_gesture_tasks[websocket] = set()
                        self.active_gesture_tasks[websocket].add(task)
                        
                        # Don't await - let it run in parallel with other messages
                        # The task will handle sending results when done
                    
                    elif message_type == "ping":
                        # Respond to ping
                        response = {
                            "type": "pong",
                            "status_code": 200,
                            "timestamp": time.time()
                        }
                        await self._safe_send_message(websocket, response)
                    
                    elif message_type == "audio_to_glasses":
                        # Forward TTS audio to glasses (echo back to client for playback)
                        print(f"[SERVER] Received audio_to_glasses message (TTS audio)")
                        audio_data = data.get("audio_data", "")
                        format_type = data.get("format", "wav")
                        sample_rate = data.get("sample_rate", 22050)
                        is_tts = data.get("is_tts", False)
                        text = data.get("text", "")
                        
                        if audio_data:
                            # Forward as tts_audio message for client to play
                            response = {
                                "type": "tts_audio",
                                "audio_data": audio_data,
                                "format": format_type,
                                "sample_rate": sample_rate,
                                "is_tts": is_tts,
                                "text": text,
                                "timestamp": time.time()
                            }
                            await self._safe_send_message(websocket, response)
                            print(f"[SERVER] Forwarded TTS audio to glasses ({len(audio_data)} chars base64)")
                        else:
                            print(f"[SERVER] ERROR: audio_to_glasses message missing audio_data")
                            response = {
                                "type": "error",
                                "error": "Missing audio_data in audio_to_glasses message",
                                "timestamp": time.time()
                            }
                            await self._safe_send_message(websocket, response)
                        
                    else:
                        print(f"[SERVER] Unknown message type: {message_type}")
                        
                except json.JSONDecodeError as e:
                    print(f"[SERVER] JSON decode error: {e}")
                except websockets.exceptions.ConnectionClosed:
                    # Connection closed during message processing - break the loop
                    print(f"[SERVER] Connection closed during message handling")
                    break
                except Exception as e:
                    print(f"[SERVER] Error handling message: {e}")
                    # Check if connection is still open, if not break
                    if hasattr(websocket, 'state'):
                        from websockets.protocol import State
                        if websocket.state != State.OPEN:
                            print(f"[SERVER] Connection closed after error, stopping")
                            break
                    
        except websockets.exceptions.ConnectionClosed as e:
            print(f"[SERVER] ========== CONNECTION CLOSED ==========")
            print(f"[SERVER] Client: {websocket.remote_address}")
            print(f"[SERVER] Close code: {e.code}, reason: {e.reason}")
        except Exception as e:
            print(f"[SERVER] ========== WEBSOCKET ERROR ==========")
            print(f"[SERVER] Error: {e}")
            import traceback
            print(f"[SERVER] Error traceback: {traceback.format_exc()}")
        finally:
            self._clear_pending_chunk(websocket)
            self.chunk_worker_tasks.pop(websocket, None)

            # Cancel any pending tasks for this connection
            if websocket in self.active_audio_tasks:
                for task in list(self.active_audio_tasks[websocket]):
                    if not task.done():
                        task.cancel()
                del self.active_audio_tasks[websocket]
            
            if websocket in self.active_gesture_tasks:
                for task in list(self.active_gesture_tasks[websocket]):
                    if not task.done():
                        task.cancel()
                del self.active_gesture_tasks[websocket]
            
            # Clean up speaker tracking for this connection
            if websocket in self.speaker_tracking:
                del self.speaker_tracking[websocket]
            
            self.active_connections.discard(websocket)
            # Clean up stop processing flag for this connection
            if websocket in self.stop_processing_per_connection:
                del self.stop_processing_per_connection[websocket]
            print(f"[SERVER] Cleaned up connection from {websocket.remote_address}")

    async def _cleanup(self):
        """Clean up all connections and tasks."""
        print("[SERVER] Closing all WebSocket connections...")
        
        # Close all active connections
        if self.server:
            self.server.close()
            await self.server.wait_closed()
        
        # Cancel all active tasks
        print("[SERVER] Cancelling active tasks...")
        for websocket in list(self.active_connections):
            self._clear_pending_chunk(websocket)
            self.chunk_worker_tasks.pop(websocket, None)
            if websocket in self.active_audio_tasks:
                for task in list(self.active_audio_tasks[websocket]):
                    if not task.done():
                        task.cancel()
                        try:
                            await task
                        except asyncio.CancelledError:
                            pass
                del self.active_audio_tasks[websocket]
            
            if websocket in self.active_gesture_tasks:
                for task in list(self.active_gesture_tasks[websocket]):
                    if not task.done():
                        task.cancel()
                        try:
                            await task
                        except asyncio.CancelledError:
                            pass
                del self.active_gesture_tasks[websocket]
        
        print("[SERVER] Cleanup complete")

    async def start_server(self):
        """Start the WebSocket server with HTTP handler for health checks."""
        print(f"[SERVER] Starting server on {self.host}:{self.port}")
        
        self.main_loop = asyncio.get_event_loop()
        
        # Create a simple HTTP handler for health checks (ngrok sends GET /)
        async def http_handler(reader, writer):
            """Handle HTTP requests (for ngrok health checks)."""
            try:
                request = await reader.read(1024)
                request_str = request.decode('utf-8', errors='ignore')
                
                # Simple HTTP response for health checks
                if request_str.startswith('GET'):
                    response = (
                        "HTTP/1.1 200 OK\r\n"
                        "Content-Type: text/plain\r\n"
                        "Content-Length: 13\r\n"
                        "Connection: close\r\n"
                        "\r\n"
                        "WebSocket OK"
                    )
                    writer.write(response.encode())
                    await writer.drain()
                writer.close()
            except Exception as e:
                pass  # Ignore errors in HTTP handler
        
        # Start HTTP server for health checks on a separate port (optional)
        # Or handle in WebSocket upgrade
        
        self.server = await websockets.serve(
            self.handle_websocket,
            self.host,
            self.port,
            ping_interval=10,      # Send ping every 10 seconds to keep connection alive
            ping_timeout=None,     # No timeout - wait forever for pong response
            close_timeout=None,    # No timeout - wait forever for close handshake
            max_size=10**7,        # 10MB max message size (for larger audio files)
            max_queue=128,         # More queued messages
            # Additional timeout settings:
            open_timeout=None,     # No timeout - wait forever for opening handshake
            logger=None            # Disable internal logging for cleaner output
        )
        
        print(f"[SERVER] Server running on ws://{self.host}:{self.port}")
        print("[SERVER] Note: 502 errors from ngrok are normal (HTTP health checks)")
        print("[SERVER] WebSocket connections work fine - ignore HTTP 502 errors")
        print("[SERVER] Press Ctrl+C to stop")
        
        # Set up signal handlers for graceful shutdown (works on Unix/Linux/Mac)
        if sys.platform != 'win32':
            try:
                for sig in (signal.SIGTERM, signal.SIGINT):
                    self.main_loop.add_signal_handler(sig, lambda s=sig: self.shutdown_event.set())
            except NotImplementedError:
                # Some platforms don't support add_signal_handler
                pass
        
        # Wait for shutdown signal
        # Use periodic checks to allow KeyboardInterrupt to be raised (important for Windows)
        try:
            while not self.shutdown_event.is_set():
                try:
                    # Wait with timeout to allow KeyboardInterrupt to be raised periodically
                    await asyncio.wait_for(self.shutdown_event.wait(), timeout=0.5)
                    break
                except asyncio.TimeoutError:
                    # Continue loop to check again (allows KeyboardInterrupt to be raised)
                    continue
        except KeyboardInterrupt:
            # Handle Ctrl+C (mainly for Windows)
            print("\n[SERVER] Keyboard interrupt received")
            self.shutdown_event.set()
            raise  # Re-raise to be caught by asyncio.run
        
        # Cleanup: close all connections and cancel tasks
        print("\n[SERVER] Shutting down gracefully...")
        await self._cleanup() 

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
        print("\n[SERVER] Keyboard interrupt received, shutting down...")
    except Exception as e:
        print(f"[SERVER] Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("[SERVER] Server stopped")

if __name__ == "__main__":
    main()
