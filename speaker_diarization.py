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
from speechbrain.pretrained import EncoderClassifier

warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

class DiarizationPipeline:
    def __init__(self, hf_token: str):
        """Initialize the diarization pipeline with performance improvements."""
        print("[DIARIZATION] Initializing Diarization Pipeline...")
        
        os.environ["HF_TOKEN"] = hf_token
        
        self.diarization_pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=hf_token
        )
        
        # Configure diarization for better multi-speaker detection
        # Using correct parameter names for pyannote/speaker-diarization-3.1
        try:
            self.diarization_pipeline.instantiate({
                "clustering": {
                    "threshold": 0.3,  # Lower threshold for better detection
                    "min_cluster_size": 1,  # Allow single segments
                }
            })
            print("[DIARIZATION] Diarization configured for better multi-speaker detection")
        except Exception as e:
            print(f"[DIARIZATION] Warning: Could not configure diarization parameters: {e}")
            print("[DIARIZATION] Using default diarization settings")
        
        device = "cpu"  # Force CPU for memory efficiency on Render
        
        self.whisper_model = WhisperModel(
            "large-v3",
            device=device,
            compute_type="int8",  # Use int8 for maximum memory efficiency
            num_workers=1,  # Limit workers to save memory
            download_root=None,  # Use default cache
            local_files_only=False
        )
        
        # Initialize speaker embedding model (ECAPA-TDNN for 192-dim embeddings)
        print("[DIARIZATION] Loading speaker embedding model...")
        self.speaker_model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="pretrained_models/spkrec-ecapa-voxceleb",
            run_opts={"device": device}
        )
        print("[DIARIZATION] Speaker embedding model loaded (192-dim ECAPA-TDNN)")
        
        # Minimal cache for memory efficiency
        self.audio_cache = {}
        self.cache_lock = threading.Lock()
        self.max_cache_size = 5  # Limit cache size
        
        # Memory management
        self._cleanup_memory()
        
        print("[DIARIZATION] Optimized Diarization Pipeline initialized successfully")
        print(f"[DIARIZATION] Using device: {device}")
        
    def _cleanup_memory(self):
        """Clean up memory to prevent OOM."""
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def extract_speaker_embedding(self, audio_array: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        """
        Extract 192-dimensional speaker embedding from audio.
        This function can be called from ar_glasses_server to get embeddings for voice registration.
        
        Args:
            audio_array: Audio as numpy array (float32, mono, normalized to [-1, 1])
            sample_rate: Sample rate in Hz (default: 16000)
            
        Returns:
            np.ndarray: 192-dimensional embedding, or None if failed
            
        Example:
            embedding = diarization_pipeline.extract_speaker_embedding(audio_array, 16000)
            # Returns: np.array with shape (192,)
        """
        try:
            print(f"[DIARIZATION] Extracting 192-dim speaker embedding...")
            print(f"[DIARIZATION] Audio: {len(audio_array)} samples at {sample_rate}Hz")
            
            # Ensure audio is mono
            if len(audio_array.shape) > 1:
                audio_array = audio_array.mean(axis=1)
            
            # Convert to float32
            audio_array = audio_array.astype(np.float32)
            
            # Normalize to [-1, 1] range
            if np.max(np.abs(audio_array)) > 0:
                audio_array = audio_array / np.max(np.abs(audio_array))
            
            # Convert to torch tensor [1, samples]
            audio_tensor = torch.from_numpy(audio_array).unsqueeze(0)
            
            # Extract embedding using SpeechBrain ECAPA-TDNN
            with torch.no_grad():
                embedding = self.speaker_model.encode_batch(audio_tensor)
                embedding = embedding.squeeze().cpu().numpy()  # Shape: (192,)
            
            print(f"[DIARIZATION] ✓ Embedding extracted successfully!")
            print(f"[DIARIZATION] Embedding shape: {embedding.shape}")
            print(f"[DIARIZATION] Embedding stats: mean={np.mean(embedding):.4f}, std={np.std(embedding):.4f}")
            
            return embedding
            
        except Exception as e:
            print(f"[DIARIZATION] ERROR extracting embedding: {e}")
            import traceback
            print(f"[DIARIZATION] Traceback: {traceback.format_exc()}")
            return None
        
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
            
            # Post-process for Cantonese output
            text = self.ensure_cantonese_output(text, info)
            
            return text
            
        except Exception as e:
            print(f"[DIARIZATION] Error transcribing segment: {e}")
            return ""

    def compare_embeddings(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compare two speaker embeddings using cosine similarity.
        
        Args:
            embedding1: First embedding (192-dim)
            embedding2: Second embedding (192-dim)
            
        Returns:
            float: Cosine similarity score (0-1), where higher means more similar
        """
        try:
            # Ensure embeddings are 1D
            emb1 = embedding1.flatten()
            emb2 = embedding2.flatten()
            
            # Compute cosine similarity
            dot_product = np.dot(emb1, emb2)
            norm1 = np.linalg.norm(emb1)
            norm2 = np.linalg.norm(emb2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity = dot_product / (norm1 * norm2)
            
            # Ensure similarity is in [0, 1] range
            similarity = np.clip(similarity, 0.0, 1.0)
            
            return float(similarity)
            
        except Exception as e:
            print(f"[DIARIZATION] Error comparing embeddings: {e}")
            return 0.0
    
    def ensure_cantonese_output(self, text: str, info) -> str:
        """Ensure the output is in Cantonese characters, not English."""
        if not text:
            return text
        
        # Check if the detected language is actually Cantonese
        detected_lang = getattr(info, 'language', 'unknown')
        lang_confidence = getattr(info, 'language_probability', 0.0)
        
        print(f"[DIARIZATION] Detected language: {detected_lang} (confidence: {lang_confidence:.3f})")
        print(f"[DIARIZATION] Raw transcription: '{text}'")
        
        # Check if text contains Cantonese characters (CJK Unified Ideographs)
        has_cantonese = any('\u4e00' <= char <= '\u9fff' for char in text)
        
        if has_cantonese:
            print(f"[DIARIZATION] Text contains Cantonese characters - keeping as-is")
            return text
        
        # If we get English output when expecting Cantonese, try to force Cantonese
        print(f"[DIARIZATION] Text appears to be English, but we expected Cantonese")
        print(f"[DIARIZATION] This might be due to audio quality or model limitations")
        
        # Return the text as-is for now, but log the issue
        return text

    def process_audio_array(self, audio_array: np.ndarray, sample_rate: int = 16000, registered_voices: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process audio array for speaker diarization and transcription.
        
        Args:
            audio_array: Audio data as numpy array
            sample_rate: Sample rate in Hz
            registered_voices: Dictionary of registered voice embeddings {voice_id: {'embedding': np.array, ...}}
        """
        try:
            print(f"[DIARIZATION] Processing audio array: {len(audio_array)} samples, {sample_rate} Hz")
            
            # Debug: Check what we received
            print(f"[DIARIZATION] DEBUG: registered_voices parameter = {type(registered_voices)}")
            print(f"[DIARIZATION] DEBUG: registered_voices is None? {registered_voices is None}")
            if registered_voices:
                print(f"[DIARIZATION] DEBUG: registered_voices keys = {list(registered_voices.keys())}")
                print(f"[DIARIZATION] DEBUG: Number of voices = {len(registered_voices)}")
            
            # Check if we have a registered voice (wearer's voice)
            has_registered_voice = registered_voices and len(registered_voices) > 0
            if has_registered_voice:
                print(f"[DIARIZATION] ✓ Registered voice detected - will identify wearer's segments")
            else:
                print(f"[DIARIZATION] ✗ No registered voice - processing normally")
            
            # Ensure audio is mono and float32
            if len(audio_array.shape) > 1:
                audio_array = audio_array.mean(axis=1)
            audio_array = audio_array.astype(np.float32)
            
            # Audio preprocessing for better diarization
            # Normalize audio to improve detection
            if np.max(np.abs(audio_array)) > 0:
                audio_array = audio_array / np.max(np.abs(audio_array))
            
            
            print(f"[DIARIZATION] Audio preprocessed: {len(audio_array)} samples, max={np.max(audio_array):.3f}")
            
            # Create temporary file for diarization (required by pyannote)
            temp_file = f"temp_audio_{int(time.time() * 1000)}.wav"
            sf.write(temp_file, audio_array, sample_rate)
            
            try:
                # Perform speaker diarization
                print("[DIARIZATION] Running speaker diarization...")
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
                print(f"[DIARIZATION] Detected speakers: {sorted(unique_speakers)}")
                print(f"[DIARIZATION] Total diarization segments: {len(all_segments)}")
                print(f"[DIARIZATION] All diarization segments:")
                for i, seg in enumerate(all_segments):
                    print(f"  Segment {i+1}: {seg['speaker']} ({seg['start']:.2f}s - {seg['end']:.2f}s, {seg['duration']:.2f}s)")
                
                # Calculate total coverage
                total_audio_duration = len(audio_array) / sample_rate
                covered_duration = sum(seg['duration'] for seg in all_segments)
                coverage_percent = (covered_duration / total_audio_duration) * 100
                print(f"[DIARIZATION] Audio coverage: {covered_duration:.2f}s / {total_audio_duration:.2f}s ({coverage_percent:.1f}%)")
                
                # If no speakers detected, this might be the issue
                if len(all_segments) == 0:
                    print("[DIARIZATION] WARNING: No speakers detected by diarization!")
                    print("[DIARIZATION] This could be due to:")
                    print("[DIARIZATION] - Audio too short or too quiet")
                    print("[DIARIZATION] - Diarization model issues")
                    print("[DIARIZATION] - Audio format problems")
                
                # Process each speaker segment with optimizations
                segments = []
                segment_count = 0
                skipped_short = 0
                skipped_energy = 0
                skipped_empty = 0
                
                print(f"[DIARIZATION] Processing {len(all_segments)} diarization segments...")
                
                for turn, _, speaker in diarization.itertracks(yield_label=True):
                    start_time = turn.start
                    end_time = turn.end
                    duration = end_time - start_time
                    
                    # OPTIMIZATION 1: Skip very short segments (more lenient for multi-speaker)
                    if duration < 0.05:  # Very lenient - only skip extremely short segments
                        print(f"[OPTIMIZED] Skipping very short segment: {duration:.2f}s")
                        skipped_short += 1
                        continue
                    
                    if duration > 15.0:  # Increased limit for longer conversations
                        print(f"[DIARIZATION] Skipping very long segment: {duration:.2f}s")
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
                    if rms_energy < 0.0005:  # Very low threshold - only skip extremely quiet segments
                        print(f"[OPTIMIZED] Skipping low energy segment: {rms_energy:.4f}")
                        skipped_energy += 1
                        continue
                    
                    # Memory cleanup before processing each segment
                    self._manage_cache()
                    self._cleanup_memory()
                    
                    #  Use cached transcription if available
                    segment_key = f"{start_time:.2f}_{end_time:.2f}_{speaker}"
                    cached_text = None
                    
                    with self.cache_lock:
                        if segment_key in self.audio_cache:
                            cached_text = self.audio_cache[segment_key]
                    
                    if cached_text is not None:
                        text = cached_text
                        print(f"[DIARIZATION] Using cached transcription for segment {segment_count + 1}")
                    else:
                        # Transcribe segment
                        print(f"[DIARIZATION] Transcribing segment: {start_time:.2f}s - {end_time:.2f}s")
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
                    if not text or len(text.strip()) < 1:  # Very lenient - only skip completely empty
                        skipped_empty += 1
                        continue
                    
                    # Filter out hallucination text from the segment (not the whole segment)
                    hallucination_patterns = [
                        "字幕由 Amara.org 社群提供",
                        "字幕由Amara.org社群提供",
                        "Amara.org",
                        "字幕由",
                        "社群提供",
                        "中文字幕",
                        "李宗盛",
                    ]
                    
                    original_text = text
                    for pattern in hallucination_patterns:
                        if pattern in text:
                            text = text.replace(pattern, "").strip()
                            print(f"[DIARIZATION] Removed hallucination '{pattern}' from segment")
                    
                    # Clean up spacing
                    text = " ".join(text.split())
                    
                    # If after removing hallucinations there's nothing left, skip the segment
                    if not text or len(text.strip()) < 1:
                        print(f"[DIARIZATION] Segment was entirely hallucination: '{original_text}'")
                        skipped_empty += 1
                        continue
                    
                    # Compare with registered voice if available
                    is_wearer = False
                    voice_similarity = 0.0
                    
                    if has_registered_voice:
                        # Extract embedding from this segment
                        segment_embedding = self.extract_speaker_embedding(segment_audio, sample_rate)
                        
                        if segment_embedding is not None:
                            # Get the wearer's registered voice (only one voice)
                            voice_id, voice_data = next(iter(registered_voices.items()))
                            registered_embedding = voice_data.get('embedding')
                            
                            if registered_embedding is not None:
                                similarity = self.compare_embeddings(segment_embedding, registered_embedding)
                                voice_similarity = similarity
                                print(f"[DIARIZATION] Similarity with wearer's voice: {similarity:.3f}")
                                
                                if similarity > 0.75:
                                    is_wearer = True
                                    print(f"[DIARIZATION] ✓ Segment identified as WEARER (similarity: {similarity:.3f})")
                                else:
                                    print(f"[DIARIZATION] → Segment is OTHER speaker (similarity: {similarity:.3f})")
                    
                    # Add segment with voice matching info
                    segment_data = {
                        'speaker_id': speaker,
                        'transcription': text,
                        'start': start_time,
                        'end': end_time,
                        'duration': duration,
                        'confidence': 0.9  # Default confidence for optimized processing
                    }
                    
                    # Add voice matching results if wearer's voice is registered
                    if has_registered_voice:
                        segment_data['is_wearer'] = is_wearer
                        segment_data['voice_similarity'] = voice_similarity
                    
                    segments.append(segment_data)
                    
                    segment_count += 1
                    if is_wearer:
                        print(f"[DIARIZATION] Added segment {segment_count}: {speaker} [WEARER] - '{text[:50]}{'...' if len(text) > 50 else ''}'")
                    else:
                        print(f"[DIARIZATION] Added segment {segment_count}: {speaker} [OTHER] - '{text[:50]}{'...' if len(text) > 50 else ''}'")
                
                print(f"[DIARIZATION] Processing complete: {len(segments)} valid segments found")
                print(f"[DIARIZATION] Filtering summary:")
                print(f"  - Skipped short segments: {skipped_short}")
                print(f"  - Skipped low energy: {skipped_energy}")
                print(f"  - Skipped empty transcriptions: {skipped_empty}")
                print(f"  - Final valid segments: {len(segments)}")
                
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
            print(f"[DIARIZATION] Error processing audio array: {e}")
            return None

    def clear_cache(self):
        """Clear the audio cache to free memory."""
        with self.cache_lock:
            self.audio_cache.clear()
            print("[DIARIZATION] Audio cache cleared")

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
    pipeline = DiarizationPipeline(hf_token)
    
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
