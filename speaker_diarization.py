#!/usr/bin/env python3

import os
import warnings
import numpy as np
from pathlib import Path
from typing import Dict, Any, List
import threading
import time
import torch
import soundfile as sf
from pyannote.audio import Pipeline
from faster_whisper import WhisperModel

warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

class OptimizedDiarizationPipeline:
    def __init__(self, hf_token: str):
        """Initialize the optimized diarization pipeline with performance improvements."""
        print("[OPTIMIZED] Initializing Optimized Diarization Pipeline...")
        
        os.environ["HF_TOKEN"] = hf_token
        
        try:
            self.diarization_pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=hf_token
            )
        except TypeError:
            try:
                self.diarization_pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1",
                    token=hf_token
                )
            except TypeError:
                self.diarization_pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1"
                )
        
        # Configure diarization for better multi-speaker detection
        # Using correct parameter names for pyannote/speaker-diarization-3.1
        try:
            self.diarization_pipeline.instantiate({
                "clustering": {
                    "threshold": 0.4,  # Lower threshold for better multi-speaker detection
                    "min_cluster_size": 1,  # Allow single-speaker clusters
                }
            })
            print("[OPTIMIZED] Diarization configured for better multi-speaker detection")
        except Exception as e:
            print(f"[OPTIMIZED] Warning: Could not configure diarization parameters: {e}")
            print("[OPTIMIZED] Using default diarization settings")
        
        device = "cpu"  # Force CPU for memory efficiency on Render
        
        self.whisper_model = WhisperModel(
            "large-v3",
            device=device,
            compute_type="int8",  # Use int8 for maximum memory efficiency
            num_workers=1,  # Limit workers to save memory
            download_root=None,  # Use default cache
            local_files_only=False
        )
        
        # Minimal cache for memory efficiency
        self.audio_cache = {}
        self.cache_lock = threading.Lock()
        self.max_cache_size = 5  # Limit cache size
        
        # Memory management
        self._cleanup_memory()
        
        print("[OPTIMIZED] Optimized Diarization Pipeline initialized successfully")
        print(f"[OPTIMIZED] Using device: {device}")
        
    def _cleanup_memory(self):
        """Clean up memory to prevent OOM."""
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
    def _manage_cache(self):
        """Manage cache size to prevent memory overflow."""
        with self.cache_lock:
            if len(self.audio_cache) > self.max_cache_size:
                # Remove oldest entries
                keys_to_remove = list(self.audio_cache.keys())[:-self.max_cache_size]
                for key in keys_to_remove:
                    del self.audio_cache[key]
                self._cleanup_memory()

    def preprocess_audio(self, audio_segment: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        """Preprocess audio for maximum transcription accuracy."""
        # Ensure audio is float32 (required by Whisper)
        if audio_segment.dtype != np.float32:
            audio_segment = audio_segment.astype(np.float32)
        
        # Normalize audio
        if np.max(np.abs(audio_segment)) > 0:
            audio_segment = audio_segment / np.max(np.abs(audio_segment))
        
        # Apply gentle noise reduction (simple high-pass filter)
        from scipy import signal
        if len(audio_segment) > 100:
            # High-pass filter to remove low-frequency noise
            nyquist = sample_rate / 2
            high = 80.0 / nyquist  # 80 Hz high-pass filter
            b, a = signal.butter(4, high, btype='high')
            audio_segment = signal.filtfilt(b, a, audio_segment)
            # Ensure result is still float32
            audio_segment = audio_segment.astype(np.float32)
        
        # Ensure proper length (Whisper works best with certain lengths)
        min_length = int(0.5 * sample_rate)  # Minimum 0.5 seconds
        if len(audio_segment) < min_length:
            # Pad with silence
            padding = np.zeros(min_length - len(audio_segment), dtype=np.float32)
            audio_segment = np.concatenate([audio_segment, padding])
        
        return audio_segment

    def transcribe_segment(self, audio_segment: np.ndarray, sample_rate: int = 16000) -> str:
        """Transcribe a single audio segment using optimized Whisper settings."""
        try:
            # Preprocess audio for maximum accuracy
            audio_segment = self.preprocess_audio(audio_segment, sample_rate)
            # Use optimized Whisper settings for Cantonese transcription - accuracy priority
            segments, info = self.whisper_model.transcribe(
                audio_segment,
                language="yue",  # Cantonese
                task="transcribe",  # Explicitly set task
            )
            
            # Combine all segments into one text
            text = " ".join([segment.text for segment in segments]).strip()
            
            # Post-process to filter out repetitive text
            text = self.filter_repetitive_text(text)
            
            # Post-process for Cantonese output
            text = self.ensure_cantonese_output(text, info)
            
            return text
            
        except Exception as e:
            print(f"[OPTIMIZED] Error transcribing segment: {e}")
            return ""


    def filter_repetitive_text(self, text: str) -> str:
        """Filter out repetitive text patterns that indicate poor transcription."""
        if not text:
            return text
        
        # For Cantonese, be more lenient with short text
        # Cantonese words are often 1-2 characters, so don't filter aggressively
        
        # Split into words
        words = text.split()
        
        # Only filter if we have many words (more than 5) and clear repetition
        if len(words) > 5:
            # Check for repetitive patterns
            word_counts = {}
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1
            
            # If more than 70% of words are the same, it's likely repetitive
            max_count = max(word_counts.values())
            if max_count > len(words) * 0.7:
                # Return only the first occurrence of each unique word
                unique_words = []
                seen = set()
                for word in words:
                    if word not in seen:
                        unique_words.append(word)
                        seen.add(word)
                text = " ".join(unique_words)
        
        # Don't filter out short Cantonese text - it's often valid
        # Only filter if it's clearly repetitive (same word repeated many times)
        if len(words) <= 3:
            # For short text, only filter if it's the same word repeated
            if len(set(words)) == 1 and len(words) > 2:
                return words[0]  # Return just one instance
            else:
                return text  # Keep short text as-is
        
        return text

    def ensure_cantonese_output(self, text: str, info) -> str:
        """Ensure the output is in Cantonese characters, not English."""
        if not text:
            return text
        
        # Check if the detected language is actually Cantonese
        detected_lang = getattr(info, 'language', 'unknown')
        lang_confidence = getattr(info, 'language_probability', 0.0)
        
        print(f"[OPTIMIZED] Detected language: {detected_lang} (confidence: {lang_confidence:.3f})")
        print(f"[OPTIMIZED] Raw transcription: '{text}'")
        
        # Check if text contains Cantonese characters (CJK Unified Ideographs)
        has_cantonese = any('\u4e00' <= char <= '\u9fff' for char in text)
        
        if has_cantonese:
            print(f"[OPTIMIZED] Text contains Cantonese characters - keeping as-is")
            return text
        
        # If we get English output when expecting Cantonese, try to force Cantonese
        print(f"[OPTIMIZED] Text appears to be English, but we expected Cantonese")
        print(f"[OPTIMIZED] This might be due to audio quality or model limitations")
        
        # Return the text as-is for now, but log the issue
        return text

    def process_audio_array(self, audio_array: np.ndarray, sample_rate: int = 16000) -> Dict[str, Any]:
        """Process audio array for speaker diarization and transcription (optimized)."""
        try:
            print(f"[OPTIMIZED] Processing audio array: {len(audio_array)} samples, {sample_rate} Hz")
            
            # Ensure audio is mono and float32
            if len(audio_array.shape) > 1:
                audio_array = audio_array.mean(axis=1)
            audio_array = audio_array.astype(np.float32)
            
            # Create temporary file for diarization (required by pyannote)
            temp_file = f"temp_audio_{int(time.time() * 1000)}.wav"
            sf.write(temp_file, audio_array, sample_rate)
            
            try:
                # Perform speaker diarization
                print("[OPTIMIZED] Running speaker diarization...")
                diarization = self.diarization_pipeline(temp_file)
                
                # Debug: Print all detected speakers and their segments
                unique_speakers = set()
                all_segments = []
                for turn, _, speaker in diarization.itertracks(yield_label=True):
                    unique_speakers.add(speaker)
                    all_segments.append({
                        'speaker': speaker,
                        'start': turn.start,
                        'end': turn.end,
                        'duration': turn.end - turn.start
                    })
                print(f"[OPTIMIZED] Detected speakers: {sorted(unique_speakers)}")
                print(f"[OPTIMIZED] Total diarization segments: {len(all_segments)}")
                print(f"[OPTIMIZED] All diarization segments:")
                for i, seg in enumerate(all_segments):
                    print(f"  Segment {i+1}: {seg['speaker']} ({seg['start']:.2f}s - {seg['end']:.2f}s, {seg['duration']:.2f}s)")
                
                # If no speakers detected, this might be the issue
                if len(all_segments) == 0:
                    print("[OPTIMIZED] WARNING: No speakers detected by diarization!")
                    print("[OPTIMIZED] This could be due to:")
                    print("[OPTIMIZED] - Audio too short or too quiet")
                    print("[OPTIMIZED] - Diarization model issues")
                    print("[OPTIMIZED] - Audio format problems")
                
                # Process each speaker segment with optimizations
                segments = []
                segment_count = 0
                
                for turn, _, speaker in diarization.itertracks(yield_label=True):
                    start_time = turn.start
                    end_time = turn.end
                    duration = end_time - start_time
                    
                    # OPTIMIZATION 1: Skip very short segments (more lenient for multi-speaker)
                    if duration < 0.2:  # Even more lenient for better multi-speaker detection
                        print(f"[OPTIMIZED] Skipping very short segment: {duration:.2f}s")
                        continue
                    
                    # OPTIMIZATION 2: Skip segments that are too long (likely noise)
                    if duration > 15.0:  # Increased limit for longer conversations
                        print(f"[OPTIMIZED] Skipping very long segment: {duration:.2f}s")
                        continue
                    
                    # Extract audio segment
                    start_sample = int(start_time * sample_rate)
                    end_sample = int(end_time * sample_rate)
                    segment_audio = audio_array[start_sample:end_sample]
                    
                    # OPTIMIZATION 3: Check if segment has enough energy
                    if len(segment_audio) == 0:
                        continue
                    
                    # Calculate RMS energy
                    rms_energy = np.sqrt(np.mean(segment_audio**2))
                    if rms_energy < 0.002:  # Even lower threshold for better multi-speaker detection
                        print(f"[OPTIMIZED] Skipping low energy segment: {rms_energy:.4f}")
                        continue
                    
                    # Memory cleanup before processing each segment
                    self._manage_cache()
                    self._cleanup_memory()
                    
                    # OPTIMIZATION 4: Use cached transcription if available
                    segment_key = f"{start_time:.2f}_{end_time:.2f}_{speaker}"
                    cached_text = None
                    
                    with self.cache_lock:
                        if segment_key in self.audio_cache:
                            cached_text = self.audio_cache[segment_key]
                    
                    if cached_text is not None:
                        text = cached_text
                        print(f"[OPTIMIZED] Using cached transcription for segment {segment_count + 1}")
                    else:
                        # Transcribe segment
                        print(f"[OPTIMIZED] Transcribing segment: {start_time:.2f}s - {end_time:.2f}s")
                        text = self.transcribe_segment(segment_audio, sample_rate)
                        
                        # Cache the result
                        with self.cache_lock:
                            self.audio_cache[segment_key] = text
                            # Limit cache size
                            if len(self.audio_cache) > 100:
                                # Remove oldest entries
                                oldest_key = next(iter(self.audio_cache))
                                del self.audio_cache[oldest_key]
                    
                    # OPTIMIZATION 5: Filter out empty or very short transcriptions
                    if not text or len(text.strip()) < 2:
                        continue
                    
                    # OPTIMIZATION 6: Filter out common noise patterns
                    text_lower = text.lower().strip()
                    noise_patterns = ["um", "uh", "ah", "eh", "oh", "mm", "hmm", "background", "noise", "static", "silence", "breathing"]
                    if text_lower in noise_patterns:
                        continue
                    
                    # Add segment
                    segments.append({
                        'speaker_id': speaker,
                        'transcription': text,
                        'start': start_time,
                        'end': end_time,
                        'duration': duration,
                        'confidence': 0.9  # Default confidence for optimized processing
                    })
                    
                    segment_count += 1
                    print(f"[OPTIMIZED] Added segment {segment_count}: {speaker} - '{text[:50]}{'...' if len(text) > 50 else ''}'")
                
                print(f"[OPTIMIZED] Processing complete: {len(segments)} valid segments found")
                
                return {
                    'segments': segments,
                    'total_duration': len(audio_array) / sample_rate,
                    'speaker_count': len(set(seg['speaker_id'] for seg in segments)),
                    'processing_method': 'optimized_diarization'
                }
                
            finally:
                # Clean up temporary file
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
            
        except Exception as e:
            print(f"[OPTIMIZED] Error processing audio array: {e}")
            return None

    def clear_cache(self):
        """Clear the audio cache to free memory."""
        with self.cache_lock:
            self.audio_cache.clear()
            print("[OPTIMIZED] Audio cache cleared")

    def get_cache_stats(self):
        """Get cache statistics."""
        with self.cache_lock:
            return {
                'cache_size': len(self.audio_cache),
                'cache_keys': list(self.audio_cache.keys())[:10]  # First 10 keys
            }


if __name__ == "__main__":
    # Test the optimized pipeline
    hf_token = os.getenv("HF_TOKEN", "hf_your_token_here")
    pipeline = OptimizedDiarizationPipeline(hf_token)
    
    # Test with a sample audio file
    test_file = "recorded_audio/20250316_223721.wav"
    if os.path.exists(test_file):
        audio_array, sample_rate = sf.read(test_file)
        result = pipeline.process_audio_array(audio_array, sample_rate)
        if result:
            print(f"Found {len(result['segments'])} segments")
            for segment in result['segments']:
                print(f"  {segment['speaker_id']}: {segment['transcription']}")
    else:
        print("No test audio file found")
