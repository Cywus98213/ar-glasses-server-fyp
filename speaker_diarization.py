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
from translation_module import TranslationModule
from threshold_settings import ThresholdSettings

warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

class DiarizationPipeline:
    def __init__(self, hf_token: str, transcription_lang: str = "zh-hk"):
        """Initialize the diarization pipeline with performance improvements.
        
        Args:
            hf_token: HuggingFace API token
            transcription_lang: Language code for transcription (default: 'zh' for Chinese)
        """
        print("[DIARIZATION] Initializing Diarization Pipeline...")
        
        os.environ["HF_TOKEN"] = hf_token

        # Centralized, env-driven thresholds (defaults match previous behavior)
        self.thresholds = ThresholdSettings.from_env()
        
        # Store default transcription language
        self.transcription_lang = transcription_lang
        
        print(f"[DIARIZATION] Default transcription language: {self.transcription_lang}")
        print("[DIARIZATION] Translation language will be set per request")
        
        self.diarization_pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=hf_token
        )
        
        # Using correct parameter names for pyannote/speaker-diarization-3.1
        try:
            self.diarization_pipeline.instantiate({
                "clustering": {
                    "threshold": self.thresholds.diar_clustering_threshold,
                    "min_cluster_size": self.thresholds.diar_min_cluster_size,
                },
                "segmentation": {
                    "min_duration_off": self.thresholds.diar_min_duration_off,
                }
            })
            print("[DIARIZATION] Diarization configured to reduce false speakers")
        except Exception as e:
            print(f"[DIARIZATION] Warning: Could not configure diarization parameters: {e}")
            print("[DIARIZATION] Using default diarization settings")
        
        device_str = "cuda" if torch.cuda.is_available() else "cpu"

        # Use int8 for memory efficiency (original setting)
        # For better accuracy, you can use "float16" on GPU or "float32" for maximum accuracy
        self.whisper_model = WhisperModel(
            "large-v3",
            device=device_str,
            compute_type="int8",  # Use int8 for maximum memory efficiency (original setting)
            num_workers=1,  # Limit workers to save memory
            download_root=None,  # Use default cache
            local_files_only=False
        )
        
        # Initialize speaker embedding model (ECAPA-TDNN for 192-dim embeddings)
        print("[DIARIZATION] Loading speaker embedding model...")
        self.speaker_model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="pretrained_models/spkrec-ecapa-voxceleb",
            run_opts={"device": device_str}
        )
        print("[DIARIZATION] Speaker embedding model loaded (192-dim ECAPA-TDNN)")
        
        # Initialize translation module (always available for dynamic use)
        self.translator = None
        try:
            print("[DIARIZATION] Initializing translation module...")
            self.translator = TranslationModule(device=device_str)
            print("[DIARIZATION] Translation module initialized and ready")
        except Exception as e:
            print(f"[DIARIZATION] Warning: Could not initialize translation module: {e}")
            print("[DIARIZATION] Translation will be disabled")
            self.translator = None
        
        # Minimal cache for memory efficiency
        self.audio_cache = {}
        self.cache_lock = threading.Lock()
        self.max_cache_size = 5  # Limit cache size
        
        # Memory management
        self._cleanup_memory()
        
        print("[DIARIZATION] Optimized Diarization Pipeline initialized successfully")
        print(f"[DIARIZATION] Using device: {device_str}")
        
    def _cleanup_memory(self):
        """Clean up memory to prevent OOM."""
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def extract_multiple_embeddings(self, audio_array: np.ndarray, sample_rate: int = 16000, 
                                   segment_duration: float = 1.0) -> List[np.ndarray]:
        """
        Extract multiple 192-dim embeddings from audio by segmenting it.
        This creates a more robust voice profile by capturing variations.
        
        Args:
            audio_array: Audio as numpy array (float32, mono, normalized to [-1, 1])
            sample_rate: Sample rate in Hz (default: 16000)
            segment_duration: Duration of each segment in seconds (default: 1.5s)
            
        Returns:
            List of embeddings, each with shape (192,)
            
        Example:
            embeddings = pipeline.extract_multiple_embeddings(audio_array, 16000)
            # Returns: [emb1, emb2, emb3, ...] - multiple 192-dim embeddings
        """
        try:
            duration_sec = len(audio_array) / sample_rate
            print(f"[DIARIZATION] ========== MULTI-SAMPLE REGISTRATION ==========")
            print(f"[DIARIZATION] Total audio: {duration_sec:.2f}s ({len(audio_array)} samples)")
            print(f"[DIARIZATION] Segment duration: {segment_duration}s")
            
            # Ensure audio is mono
            if len(audio_array.shape) > 1:
                audio_array = audio_array.mean(axis=1)
            audio_array = audio_array.astype(np.float32)
            
            # DO NOT normalize whole audio here - normalize each segment individually
            # to match how conversation segments are processed!
            
            embeddings = []
            segment_samples = int(segment_duration * sample_rate)
            overlap_samples = int(0.3 * sample_rate)  # 0.3s overlap (30% overlap for more samples)
            
            start_idx = 0
            segment_num = 0
            
            while start_idx < len(audio_array):
                end_idx = min(start_idx + segment_samples, len(audio_array))
                segment_audio = audio_array[start_idx:end_idx].copy()  # Copy to avoid modifying original
                
                # Skip if segment too short
                if len(segment_audio) < sample_rate * 0.6:  # Minimum 0.6 seconds (reduced for more samples)
                    print(f"[DIARIZATION] Segment {segment_num+1}: Too short, skipping")
                    break
                
                # Check segment energy BEFORE normalization
                rms_energy = np.sqrt(np.mean(segment_audio**2))
                if rms_energy < 0.001:  # Very quiet
                    print(f"[DIARIZATION] Segment {segment_num+1}: Too quiet (RMS: {rms_energy:.6f}), skipping")
                    start_idx += segment_samples - overlap_samples
                    segment_num += 1
                    continue
                
                # IMPORTANT: Normalize EACH segment individually (same as conversation processing!)
                if np.max(np.abs(segment_audio)) > 0:
                    segment_audio = segment_audio / np.max(np.abs(segment_audio))
                
                # Extract embedding from this segment
                segment_duration_actual = len(segment_audio) / sample_rate
                print(f"[DIARIZATION] Segment {segment_num+1}: {segment_duration_actual:.2f}s, RMS: {rms_energy:.6f}")
                
                audio_tensor = torch.from_numpy(segment_audio).unsqueeze(0)
                
                with torch.no_grad():
                    embedding = self.speaker_model.encode_batch(audio_tensor)
                    embedding = embedding.squeeze().cpu().numpy()
                
                # Validate embedding
                if embedding.shape[0] == 192:
                    embeddings.append(embedding)
                    emb_stats = f"mean={np.mean(embedding):.4f}, std={np.std(embedding):.4f}"
                    print(f"[DIARIZATION] ✓ Segment {segment_num+1} embedding extracted ({emb_stats})")
                else:
                    print(f"[DIARIZATION] ✗ Segment {segment_num+1} invalid shape: {embedding.shape}")
                
                # Move to next segment with overlap
                start_idx += segment_samples - overlap_samples
                segment_num += 1
            
            if len(embeddings) == 0:
                print(f"[DIARIZATION] ERROR: No valid embeddings extracted!")
                return None
            
            print(f"[DIARIZATION] ========================================")
            print(f"[DIARIZATION] ✓ Extracted {len(embeddings)} embeddings total")
            print(f"[DIARIZATION] Each embedding shape: (192,)")
            
            # Calculate stats across embeddings
            embeddings_array = np.array(embeddings)
            print(f"[DIARIZATION] Embeddings array shape: {embeddings_array.shape}")
            print(f"[DIARIZATION] Overall mean: {np.mean(embeddings_array):.4f}")
            print(f"[DIARIZATION] Overall std: {np.std(embeddings_array):.4f}")
            
            # Show individual embedding stats for debugging
            for i, emb in enumerate(embeddings):
                print(f"[DIARIZATION]   Sample {i+1}: mean={np.mean(emb):.4f}, std={np.std(emb):.4f}, norm={np.linalg.norm(emb):.4f}")
            
            print(f"[DIARIZATION] ========================================")
            
            return embeddings
            
        except Exception as e:
            print(f"[DIARIZATION] ERROR extracting multiple embeddings: {e}")
            import traceback
            print(f"[DIARIZATION] Traceback: {traceback.format_exc()}")
            return None
    
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
            
            # Check audio energy BEFORE normalization
            rms_energy = np.sqrt(np.mean(audio_array**2))
            if len(audio_array) / sample_rate > 3.0:  # Only log for longer audio
                print(f"[DIARIZATION] Audio RMS energy (before norm): {rms_energy:.6f}")
            
            # Normalize to [-1, 1] range
            if np.max(np.abs(audio_array)) > 0:
                audio_array = audio_array / np.max(np.abs(audio_array))
                if len(audio_array) / sample_rate > 3.0:  # Only log for longer audio
                    print(f"[DIARIZATION] Audio normalized to range: [{np.min(audio_array):.4f}, {np.max(audio_array):.4f}]")
            
            # Convert to torch tensor [1, samples]
            audio_tensor = torch.from_numpy(audio_array).unsqueeze(0)
            
            # Extract embedding using SpeechBrain ECAPA-TDNN
            with torch.no_grad():
                embedding = self.speaker_model.encode_batch(audio_tensor)
                embedding = embedding.squeeze().cpu().numpy()  # Shape: (192,)
            
            if len(audio_array) / sample_rate > 3.0:  # Only log for longer audio
                print(f"[DIARIZATION] ✓ Embedding extracted successfully!")
                print(f"[DIARIZATION] Embedding shape: {embedding.shape}")
                print(f"[DIARIZATION] Embedding stats: mean={np.mean(embedding):.4f}, std={np.std(embedding):.4f}, norm={np.linalg.norm(embedding):.4f}")
            
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
        
        # Normalize audio (simple normalization - original approach)
        if np.max(np.abs(audio_segment)) > 0:
            audio_segment = audio_segment / np.max(np.abs(audio_segment))
        
        return audio_segment

    def transcribe_segment(self, audio_segment: np.ndarray, sample_rate: int = 16000, 
                          transcription_lang: str = None, translation_lang: str = None,
                          is_chunk: bool = False) -> Dict[str, Any]:
        """Transcribe a single audio segment using optimized Whisper settings.
        
        Args:
            audio_segment: Audio data to transcribe
            sample_rate: Sample rate of audio
            transcription_lang: Language for transcription (uses default if None)
            translation_lang: Target language for translation (no translation if None)
            is_chunk: If True, use faster settings for real-time processing
        
        Returns:
            Dictionary containing:
                - 'text': Final text (translated if needed)
                - 'original_text': Original transcription
                - 'translated': Boolean indicating if translation was performed
                - 'source_lang': Source language
                - 'target_lang': Target language (if translated)
        """
        try:
            # Use provided language or default
            source_lang = transcription_lang if transcription_lang else self.transcription_lang
            
            # Preprocess audio for maximum accuracy
            audio_segment = self.preprocess_audio(audio_segment, sample_rate)
            
            # Use faster settings for chunks (real-time processing)
            if is_chunk:
                # Maximum speed settings for real-time chunks - prioritize speed over accuracy
                segments, info = self.whisper_model.transcribe(
                    audio_segment,
                    language=source_lang,
                    task="transcribe",
                    beam_size=1,  # Greedy decoding - fastest option
                    best_of=1,    # Single candidate - fastest
                    temperature=0,  # Deterministic - fastest
                    vad_filter=False,  # Disable VAD to avoid missing speech at boundaries
                    condition_on_previous_text=False,  # Disable context for speed
                    initial_prompt=None,  # No prompt needed for chunks
                    word_timestamps=False,  # Skip word timestamps for speed
                )
            else:
                # Use language-specific Whisper settings (original settings for best compatibility)
                segments, info = self.whisper_model.transcribe(
                    audio_segment,
                    language=source_lang,
                    task="transcribe",  # Explicitly set task
                )
            
            # Combine all segments into one text
            original_text = " ".join([segment.text for segment in segments]).strip()
            
            # Post-process for language output
            original_text = self.ensure_cantonese_output(original_text, info)
            
            # Prepare result
            result = {
                'text': original_text,
                'original_text': original_text,
                'translated': False,
                'source_lang': source_lang,
                'target_lang': None
            }
            
            # Apply translation if needed (and target language is provided)
            if self.translator and translation_lang:
                translation_result = self.translator.translate_text(
                    original_text,
                    source_lang,
                    translation_lang
                )
                
                if translation_result.get('translated'):
                    result['text'] = translation_result['text']
                    result['translated'] = True
                    result['target_lang'] = translation_lang
                    print(f"[DIARIZATION] Translated: '{original_text[:30]}...' -> '{result['text'][:30]}...'")
                elif translation_result.get('error'):
                    print(f"[DIARIZATION] Translation error: {translation_result['error']}")
                    # Keep original text if translation fails
            
            return result
            
        except Exception as e:
            print(f"[DIARIZATION] Error transcribing segment: {e}")
            return {
                'text': '',
                'original_text': '',
                'translated': False,
                'source_lang': source_lang if 'source_lang' in locals() else self.transcription_lang,
                'target_lang': None
            }

    def compare_with_multiple_embeddings(self, segment_embedding: np.ndarray, 
                                         registered_embeddings: List[np.ndarray]) -> Dict[str, Any]:
        """
        Compare a segment embedding with multiple registered embeddings.
        Uses voting and averaging for robust detection.
        
        Args:
            segment_embedding: Embedding from current segment (192,)
            registered_embeddings: List of registered embeddings [(192,), (192,), ...]
            
        Returns:
            Dictionary with similarity metrics:
                - 'max_similarity': Best match score
                - 'avg_similarity': Average across all samples
                - 'median_similarity': Median score
                - 'match_count': Number of samples above threshold
        """
        similarities = []
        
        for i, reg_emb in enumerate(registered_embeddings):
            sim = self.compare_embeddings(segment_embedding, reg_emb)
            similarities.append(sim)
        
        similarities_array = np.array(similarities)
        
        # Calculate different metrics
        max_sim = np.max(similarities_array)
        avg_sim = np.mean(similarities_array)
        median_sim = np.median(similarities_array)
        
        vote_thr = self.thresholds.wearer_multi_vote_sim_threshold
        match_count = np.sum(similarities_array >= vote_thr)
        
        return {
            'max_similarity': float(max_sim),
            'avg_similarity': float(avg_sim),
            'median_similarity': float(median_sim),
            'match_count': int(match_count),
            'total_samples': len(similarities),
            'all_similarities': similarities
        }
    
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

    def process_audio_array(self, audio_array: np.ndarray, sample_rate: int = 16000, 
                           registered_voices: Dict[str, Any] = None,
                           transcription_lang: str = None, 
                           translation_lang: str = None,
                           is_chunk: bool = False,
                           speaker_tracking: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process audio array for speaker diarization and transcription.
        
        Args:
            audio_array: Audio data as numpy array
            sample_rate: Sample rate in Hz
            registered_voices: Dictionary of registered voice embeddings {voice_id: {'embedding': np.array, ...}}
            transcription_lang: Language for transcription (uses default if None)
            translation_lang: Target language for translation (no translation if None)
            is_chunk: If True, this is a real-time chunk (optimize for speed while keeping diarization)
        """
        try:
            # Only log for full audio, not chunks (will log after if segments found)
            if not is_chunk:
                chunk_mode = "FULL"
                print(f"[DIARIZATION] Processing audio array ({chunk_mode}): {len(audio_array)} samples, {sample_rate} Hz")
                print(f"[DIARIZATION] Transcription language: {transcription_lang or self.transcription_lang}")
                print(f"[DIARIZATION] Translation language: {translation_lang or 'None (no translation)'}")
                
                # Debug: Check what we received
                print(f"[DIARIZATION] DEBUG: registered_voices parameter = {type(registered_voices)}")
                print(f"[DIARIZATION] DEBUG: registered_voices is None? {registered_voices is None}")
                if registered_voices:
                    print(f"[DIARIZATION] DEBUG: registered_voices keys = {list(registered_voices.keys())}")
                    print(f"[DIARIZATION] DEBUG: Number of voices = {len(registered_voices)}")
            
            # Check if we have a registered voice (wearer's voice)
            has_registered_voice = registered_voices and len(registered_voices) > 0
            if not is_chunk:
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
            
            if not is_chunk:
                print(f"[DIARIZATION] Audio preprocessed: {len(audio_array)} samples, max={np.max(audio_array):.3f}")
            
            # FAST PATH: For real-time chunks, skip diarization and use direct transcription
            if is_chunk:
                # Skip diarization for chunks - just transcribe the whole chunk directly
                # This is MUCH faster for real-time processing
                transcription_result = self.transcribe_segment(
                    audio_array,
                    sample_rate,
                    transcription_lang=transcription_lang,
                    translation_lang=translation_lang,
                    is_chunk=True  # Use fast settings
                )
                
                text = transcription_result.get('text', '') if transcription_result else ''
                original_text = transcription_result.get('original_text', text) if transcription_result else text
                
                # Skip only completely empty transcriptions (allow single characters/words)
                if not text or len(text.strip()) < 1:
                    return {
                        'segments': [],
                        'total_duration': len(audio_array) / sample_rate,
                        'speaker_count': 0,
                        'processing_method': 'fast_chunk_transcription',
                        'is_chunk': True
                    }
                
                # Filter out hallucination segments - skip entire segment if detected
                # Check both text and original_text, and normalize whitespace for matching
                skip_segment_patterns = [
                    "请不吝点赞 订阅 转发 打赏支持明镜与点点栏目",
                    "请不吝点赞订阅转发打赏支持明镜与点点栏目",  # Without spaces
                    "明镜与点点栏目",  # Key unique phrase from the hallucination
                    "谢谢大家"
                ]
                
                # Normalize text for comparison (remove extra whitespace)
                text_normalized = " ".join(text.split())
                original_text_normalized = " ".join(original_text.split())
                
                for pattern in skip_segment_patterns:
                    # Check both normalized text fields
                    if pattern in text_normalized or pattern in original_text_normalized:
                        print(f"[DIARIZATION] Skipping chunk segment containing hallucination: '{pattern}'")
                        print(f"[DIARIZATION] Detected in text: '{text[:100]}'")
                        print(f"[DIARIZATION] Detected in original_text: '{original_text[:100]}'")
                        return {
                            'segments': [],
                            'total_duration': len(audio_array) / sample_rate,
                            'speaker_count': 0,
                            'processing_method': 'fast_chunk_transcription',
                            'is_chunk': True
                        }
                
                # Speaker identification: Track speakers across chunks for consistent IDs
                is_wearer = False
                voice_similarity = 0.0
                speaker_id = 'SPEAKER_01'  # Default to OTHER if no match
                
                # Extract embedding from chunk for speaker identification
                chunk_duration = len(audio_array) / sample_rate
                chunk_embedding = None
                
                # Extract embedding from middle portion for speed
                if chunk_duration > 1.5:
                    middle_start = int(len(audio_array) * 0.25)
                    middle_end = int(len(audio_array) * 0.75)
                    middle_audio = audio_array[middle_start:middle_end]
                    chunk_embedding = self.extract_speaker_embedding(middle_audio, sample_rate)
                else:
                    chunk_embedding = self.extract_speaker_embedding(audio_array, sample_rate)
                
                if chunk_embedding is not None:
                    # First check if wearer's voice is registered
                    if has_registered_voice:
                        voice_id, voice_data = next(iter(registered_voices.items()))
                        registered_embeddings = voice_data.get('embeddings')
                        registered_embedding_single = voice_data.get('embedding')
                        
                        best_similarity = 0.0
                        
                        if registered_embeddings is not None and len(registered_embeddings) > 1:
                            result = self.compare_with_multiple_embeddings(chunk_embedding, registered_embeddings)
                            avg_sim = result['avg_similarity']
                            median_sim = result['median_similarity']
                            max_sim = result['max_similarity']
                            match_ratio = result['match_count'] / result['total_samples']
                            all_sims = np.array(result['all_similarities'])
                            
                            best_similarity = median_sim
                            voice_similarity = best_similarity
                            
                            t = self.thresholds
                            if avg_sim >= t.wearer_chunk_avg_threshold or \
                               (median_sim >= t.wearer_chunk_median_threshold and max_sim >= t.wearer_chunk_max_threshold) or \
                               (match_ratio >= t.wearer_chunk_match_ratio_threshold and max_sim >= t.wearer_chunk_match_ratio_max_threshold) or \
                               (np.sum(all_sims >= t.wearer_chunk_count_sim_threshold) >= (len(all_sims) * t.wearer_chunk_count_ratio_threshold)):
                                is_wearer = True
                                speaker_id = 'SPEAKER_00'  # Wearer is always SPEAKER_00
                                
                                # Add wearer to speaker tracking if not already there
                                if speaker_tracking and 'speakers' in speaker_tracking:
                                    wearer_exists = any(s.get('id') == 'SPEAKER_00' for s in speaker_tracking['speakers'])
                                    if not wearer_exists:
                                        speaker_tracking['speakers'].append({
                                            'id': 'SPEAKER_00',
                                            'embedding': chunk_embedding,
                                            'is_wearer': True
                                        })
                                        print(f"[DIARIZATION] Added WEARER (SPEAKER_00) to speaker tracking")
                                
                                print(f"[DIARIZATION] Chunk identified as WEARER (SPEAKER_00) - Similarity: {best_similarity:.4f}")
                            else:
                                # Not wearer - check against tracked speakers
                                if speaker_tracking and 'speakers' in speaker_tracking:
                                    # Compare with known speakers
                                    best_match_similarity = 0.0
                                    matched_speaker = None
                                    
                                    for known_speaker in speaker_tracking['speakers']:
                                        if known_speaker.get('is_wearer', False):
                                            continue  # Skip wearer, already checked
                                        
                                        known_emb = known_speaker.get('embedding')
                                        if known_emb is not None:
                                            sim = self.compare_embeddings(chunk_embedding, known_emb)
                                            if sim > best_match_similarity:
                                                best_match_similarity = sim
                                                matched_speaker = known_speaker
                                    
                                    t = self.thresholds
                                    if matched_speaker and best_match_similarity >= t.speaker_match_threshold:
                                        speaker_id = matched_speaker['id']
                                        # Update speaker embedding with running average for stability
                                        old_emb = matched_speaker['embedding']
                                        # Weighted average: 70% old, 30% new (keeps it stable but adapts)
                                        updated_emb = 0.7 * old_emb + 0.3 * chunk_embedding
                                        matched_speaker['embedding'] = updated_emb
                                        print(f"[DIARIZATION] Chunk matched to {speaker_id} - Similarity: {best_match_similarity:.4f}")
                                    else:
                                        t = self.thresholds
                                        if matched_speaker and (t.speaker_merge_band_low <= best_match_similarity < t.speaker_merge_band_high):
                                            # Close but below threshold - merge with closest speaker
                                            speaker_id = matched_speaker['id']
                                            old_emb = matched_speaker['embedding']
                                            updated_emb = 0.6 * old_emb + 0.4 * chunk_embedding
                                            matched_speaker['embedding'] = updated_emb
                                            print(f"[DIARIZATION] Merged chunk with {speaker_id} (similarity: {best_match_similarity:.4f} - below threshold but close)")
                                        else:
                                            # New speaker detected (only if similarity is very low < merge band low)
                                            next_id = speaker_tracking.get('next_id', 1)
                                            speaker_id = f'SPEAKER_{next_id:02d}'
                                            speaker_tracking['speakers'].append({
                                                'id': speaker_id,
                                                'embedding': chunk_embedding,
                                                'is_wearer': False
                                            })
                                            speaker_tracking['next_id'] = next_id + 1
                                            print(f"[DIARIZATION] New speaker detected: {speaker_id} (best match was {best_match_similarity:.4f})")
                                else:
                                    # No speaker tracking - default to SPEAKER_01
                                    speaker_id = 'SPEAKER_01'
                                    print(f"[DIARIZATION] Chunk identified as OTHER (SPEAKER_01) - Similarity: {best_similarity:.4f}")
                                    
                        elif registered_embedding_single is not None:
                            similarity = self.compare_embeddings(chunk_embedding, registered_embedding_single)
                            voice_similarity = similarity
                            
                            if similarity >= self.thresholds.wearer_single_threshold_chunk:
                                is_wearer = True
                                speaker_id = 'SPEAKER_00'
                                
                                # Add wearer to speaker tracking if not already there
                                if speaker_tracking and 'speakers' in speaker_tracking:
                                    wearer_exists = any(s.get('id') == 'SPEAKER_00' for s in speaker_tracking['speakers'])
                                    if not wearer_exists:
                                        speaker_tracking['speakers'].append({
                                            'id': 'SPEAKER_00',
                                            'embedding': chunk_embedding,
                                            'is_wearer': True
                                        })
                                        print(f"[DIARIZATION] Added WEARER (SPEAKER_00) to speaker tracking")
                                
                                print(f"[DIARIZATION] Chunk identified as WEARER (SPEAKER_00) - Similarity: {similarity:.4f}")
                            else:
                                # Not wearer - check tracked speakers
                                if speaker_tracking and 'speakers' in speaker_tracking:
                                    best_match_similarity = 0.0
                                    matched_speaker = None
                                    
                                    for known_speaker in speaker_tracking['speakers']:
                                        if known_speaker.get('is_wearer', False):
                                            continue
                                        
                                        known_emb = known_speaker.get('embedding')
                                        if known_emb is not None:
                                            sim = self.compare_embeddings(chunk_embedding, known_emb)
                                            if sim > best_match_similarity:
                                                best_match_similarity = sim
                                                matched_speaker = known_speaker
                                    
                                    t = self.thresholds
                                    if matched_speaker and best_match_similarity >= t.speaker_match_threshold:
                                        speaker_id = matched_speaker['id']
                                        # Update speaker embedding with running average for stability
                                        old_emb = matched_speaker['embedding']
                                        updated_emb = 0.7 * old_emb + 0.3 * chunk_embedding
                                        matched_speaker['embedding'] = updated_emb
                                        print(f"[DIARIZATION] Chunk matched to {speaker_id} - Similarity: {best_match_similarity:.4f}")
                                    else:
                                        t = self.thresholds
                                        if matched_speaker and (t.speaker_merge_band_low <= best_match_similarity < t.speaker_merge_band_high):
                                            # Close but below threshold - merge with closest speaker
                                            speaker_id = matched_speaker['id']
                                            old_emb = matched_speaker['embedding']
                                            updated_emb = 0.6 * old_emb + 0.4 * chunk_embedding
                                            matched_speaker['embedding'] = updated_emb
                                            print(f"[DIARIZATION] Merged chunk with {speaker_id} (similarity: {best_match_similarity:.4f} - below threshold but close)")
                                        else:
                                            # Truly new speaker - no max limit, but proactively merge similar speakers
                                            non_wearer_speakers = [s for s in speaker_tracking['speakers'] if not s.get('is_wearer', False)]
                                            
                                            # Proactively merge similar existing speakers before creating new one
                                            if len(non_wearer_speakers) >= 2:
                                                # Check if any two speakers are very similar and should be merged
                                                merged = False
                                                for i, s1 in enumerate(non_wearer_speakers):
                                                    for j, s2 in enumerate(non_wearer_speakers[i+1:], start=i+1):
                                                        sim = self.compare_embeddings(s1['embedding'], s2['embedding'])
                                                        if sim >= self.thresholds.speaker_proactive_merge_threshold:
                                                            # Merge s2 into s1
                                                            merged_emb = 0.5 * s1['embedding'] + 0.5 * s2['embedding']
                                                            s1['embedding'] = merged_emb
                                                            # Remove s2 from tracking
                                                            speaker_tracking['speakers'] = [s for s in speaker_tracking['speakers'] if s['id'] != s2['id']]
                                                            print(f"[DIARIZATION] Merged similar speakers {s2['id']} into {s1['id']} (similarity: {sim:.4f})")
                                                            merged = True
                                                            break
                                                    if merged:
                                                        break
                                                # Recalculate after potential merge
                                                if merged:
                                                    non_wearer_speakers = [s for s in speaker_tracking['speakers'] if not s.get('is_wearer', False)]
                                            
                                            # Create new speaker (no limit)
                                            next_id = speaker_tracking.get('next_id', 1)
                                            speaker_id = f'SPEAKER_{next_id:02d}'
                                            speaker_tracking['speakers'].append({
                                                'id': speaker_id,
                                                'embedding': chunk_embedding,
                                                'is_wearer': False
                                            })
                                            speaker_tracking['next_id'] = next_id + 1
                                            print(f"[DIARIZATION] New speaker detected: {speaker_id} (best match was {best_match_similarity:.4f})")
                                else:
                                    speaker_id = 'SPEAKER_01'
                                    print(f"[DIARIZATION] Chunk identified as OTHER (SPEAKER_01) - Similarity: {similarity:.4f}")
                    else:
                        # No registered voice - use speaker tracking only
                        if speaker_tracking and 'speakers' in speaker_tracking:
                            best_match_similarity = 0.0
                            matched_speaker = None
                            
                            for known_speaker in speaker_tracking['speakers']:
                                known_emb = known_speaker.get('embedding')
                                if known_emb is not None:
                                    sim = self.compare_embeddings(chunk_embedding, known_emb)
                                    if sim > best_match_similarity:
                                        best_match_similarity = sim
                                        matched_speaker = known_speaker
                            
                            t = self.thresholds
                            if matched_speaker and best_match_similarity >= t.speaker_match_threshold:
                                speaker_id = matched_speaker['id']
                                is_wearer = matched_speaker.get('is_wearer', False)
                                # Update speaker embedding with running average for stability
                                old_emb = matched_speaker['embedding']
                                updated_emb = 0.7 * old_emb + 0.3 * chunk_embedding
                                matched_speaker['embedding'] = updated_emb
                                print(f"[DIARIZATION] Chunk matched to {speaker_id} - Similarity: {best_match_similarity:.4f}")
                            else:
                                t = self.thresholds
                                if matched_speaker and (t.speaker_merge_band_low <= best_match_similarity < t.speaker_merge_band_high):
                                    # Close but below threshold - merge with closest speaker
                                    speaker_id = matched_speaker['id']
                                    is_wearer = matched_speaker.get('is_wearer', False)
                                    old_emb = matched_speaker['embedding']
                                    updated_emb = 0.6 * old_emb + 0.4 * chunk_embedding
                                    matched_speaker['embedding'] = updated_emb
                                    print(f"[DIARIZATION] Merged chunk with {speaker_id} (similarity: {best_match_similarity:.4f})")
                                else:
                                    # No max speaker limit - proactively merge similar speakers
                                    non_wearer_speakers = [s for s in speaker_tracking['speakers'] if not s.get('is_wearer', False)]
                                    
                                    # Proactively merge similar existing speakers before creating new one
                                    if len(non_wearer_speakers) >= 2:
                                        # Check if any two speakers are very similar and should be merged
                                        merged = False
                                        for i, s1 in enumerate(non_wearer_speakers):
                                            for j, s2 in enumerate(non_wearer_speakers[i+1:], start=i+1):
                                                sim = self.compare_embeddings(s1['embedding'], s2['embedding'])
                                                if sim >= self.thresholds.speaker_proactive_merge_threshold:
                                                    # Merge s2 into s1
                                                    merged_emb = 0.5 * s1['embedding'] + 0.5 * s2['embedding']
                                                    s1['embedding'] = merged_emb
                                                    # Remove s2 from tracking
                                                    speaker_tracking['speakers'] = [s for s in speaker_tracking['speakers'] if s['id'] != s2['id']]
                                                    print(f"[DIARIZATION] Merged similar speakers {s2['id']} into {s1['id']} (similarity: {sim:.4f})")
                                                    merged = True
                                                    break
                                            if merged:
                                                break
                                        # Recalculate after potential merge
                                        if merged:
                                            non_wearer_speakers = [s for s in speaker_tracking['speakers'] if not s.get('is_wearer', False)]
                                    
                                    # Create new speaker (no limit)
                                    next_id = speaker_tracking.get('next_id', 1)
                                    speaker_id = f'SPEAKER_{next_id:02d}'
                                    speaker_tracking['speakers'].append({
                                        'id': speaker_id,
                                        'embedding': chunk_embedding,
                                        'is_wearer': False
                                    })
                                    speaker_tracking['next_id'] = next_id + 1
                                    print(f"[DIARIZATION] New speaker detected: {speaker_id} (best match was {best_match_similarity:.4f})")
                        else:
                            # No tracking available - default
                            speaker_id = 'SPEAKER_00'
                
                # Create single segment for the chunk
                segment_data = {
                    'speaker_id': speaker_id,
                    'transcription': text,
                    'start': 0.0,
                    'end': len(audio_array) / sample_rate,
                    'duration': len(audio_array) / sample_rate,
                    'confidence': 0.85,
                    'original_text': transcription_result.get('original_text', text),
                    'translated': transcription_result.get('translated', False),
                    'source_lang': transcription_result.get('source_lang', transcription_lang or self.transcription_lang),
                    'is_wearer': is_wearer,
                    'voice_similarity': voice_similarity
                }
                
                if transcription_result.get('target_lang'):
                    segment_data['target_lang'] = transcription_result.get('target_lang')
                
                return {
                    'segments': [segment_data],
                    'total_duration': len(audio_array) / sample_rate,
                    'speaker_count': 1,
                    'processing_method': 'fast_chunk_transcription',
                    'is_chunk': True
                }
            
            # FULL PATH: For full audio, use full diarization
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
                
                # Only log diarization details for full audio or when segments found
                if not is_chunk or len(all_segments) > 0:
                    print(f"[DIARIZATION] Detected speakers: {sorted(unique_speakers)}")
                    print(f"[DIARIZATION] Total diarization segments: {len(all_segments)}")
                    if len(all_segments) > 0:
                        print(f"[DIARIZATION] All diarization segments:")
                        for i, seg in enumerate(all_segments):
                            print(f"  Segment {i+1}: {seg['speaker']} ({seg['start']:.2f}s - {seg['end']:.2f}s, {seg['duration']:.2f}s)")
                    
                    # Calculate total coverage
                    total_audio_duration = len(audio_array) / sample_rate
                    covered_duration = sum(seg['duration'] for seg in all_segments)
                    coverage_percent = (covered_duration / total_audio_duration) * 100
                    print(f"[DIARIZATION] Audio coverage: {covered_duration:.2f}s / {total_audio_duration:.2f}s ({coverage_percent:.1f}%)")
                
                # If no speakers detected, log warning only for full audio (not chunks)
                if len(all_segments) == 0:
                    if not is_chunk:
                        print("[DIARIZATION] WARNING: No speakers detected by diarization!")
                        print("[DIARIZATION] This could be due to:")
                        print("[DIARIZATION] - Audio too short or too quiet")
                        print("[DIARIZATION] - Diarization model issues")
                        print("[DIARIZATION] - Audio format problems")
                    # For chunks, completely silent - no logging at all
                
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
                    
                    # OPTIMIZATION 1: Skip very short segments (prevents noise/hallucination speakers)
                    if duration < 0.3:  # Increased from 0.05s to 0.3s - skip very short segments
                        print(f"[OPTIMIZED] Skipping very short segment: {duration:.2f}s (likely noise)")
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
                    
                    # Calculate RMS energy (skip quiet segments that are likely noise)
                    rms_energy = np.sqrt(np.mean(segment_audio**2))
                    if rms_energy < 0.002:  # Increased from 0.0005 to 0.002 - skip quiet segments
                        print(f"[OPTIMIZED] Skipping low energy segment: {rms_energy:.6f} (likely background noise)")
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
                    
                    transcription_result = None
                    if cached_text is not None:
                        # Cached text is just a string, convert to result dict
                        transcription_result = {
                            'text': cached_text,
                            'original_text': cached_text,
                            'translated': False,
                            'source_lang': self.transcription_lang,
                            'target_lang': None
                        }
                        print(f"[DIARIZATION] Using cached transcription for segment {segment_count + 1}")
                    else:
                        # Transcribe segment (returns dict with translation info)
                        print(f"[DIARIZATION] Transcribing segment: {start_time:.2f}s - {end_time:.2f}s")
                        transcription_result = self.transcribe_segment(
                            segment_audio, 
                            sample_rate,
                            transcription_lang=transcription_lang,
                            translation_lang=translation_lang,
                            is_chunk=False  # Full audio uses normal settings
                        )
                        
                        # Cache the text result
                        with self.cache_lock:
                            self.audio_cache[segment_key] = transcription_result.get('text', '')
                            # Limit cache size
                            if len(self.audio_cache) > 100:
                                # Remove oldest entries
                                oldest_key = next(iter(self.audio_cache))
                                del self.audio_cache[oldest_key]
                    
                    # Extract text from result
                    text = transcription_result.get('text', '') if transcription_result else ''
                    original_text = transcription_result.get('original_text', text) if transcription_result else text
                    
                    # Filter out empty or very short transcriptions
                    if not text or len(text.strip()) < 2:  # Increased from 1 to 2 - skip single character
                        print(f"[OPTIMIZED] Skipping empty/short text: '{text}'")
                        skipped_empty += 1
                        continue
                    
                    # Filter out hallucination segments - skip entire segment if detected
                    # Check both text and original_text, and normalize whitespace for matching
                    skip_segment_patterns = [
                        "请不吝点赞 订阅 转发 打赏支持明镜与点点栏目",
                        "请不吝点赞订阅转发打赏支持明镜与点点栏目",  # Without spaces
                        "明镜与点点栏目",  # Key unique phrase from the hallucination
                        "谢谢大家"
                    ]
                    
                    # Normalize text for comparison (remove extra whitespace)
                    text_normalized = " ".join(text.split())
                    original_text_normalized = " ".join(original_text.split())
                    
                    should_skip = False
                    for pattern in skip_segment_patterns:
                        # Check both normalized text fields
                        if pattern in text_normalized or pattern in original_text_normalized:
                            print(f"[DIARIZATION] Skipping segment containing hallucination: '{pattern}'")
                            print(f"[DIARIZATION] Detected in text: '{text[:100]}'")
                            print(f"[DIARIZATION] Detected in original_text: '{original_text[:100]}'")
                            skipped_empty += 1
                            should_skip = True
                            break
                    
                    if should_skip:
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
                        "嗯",  # Um/uh sounds
                        "啊",  # Ah sounds
                        "呃",  # Uh sounds
                        "哦",  # Oh sounds
                        "喔",  # Oh sounds (variant)
                    ]
                    
                    original_text = text
                    for pattern in hallucination_patterns:
                        if pattern in text:
                            text = text.replace(pattern, "").strip()
                            print(f"[DIARIZATION] Removed hallucination '{pattern}' from segment")
                    
                    # Clean up spacing
                    text = " ".join(text.split())
                    
                    # If after removing hallucinations there's nothing left, skip the segment
                    if not text or len(text.strip()) < 2:
                        print(f"[DIARIZATION] Segment was entirely hallucination/noise: '{original_text}'")
                        skipped_empty += 1
                        continue
                    
                    # Additional filter: Skip if text is only punctuation or symbols
                    if text.strip() and all(not c.isalnum() for c in text.strip()):
                        print(f"[DIARIZATION] Skipping non-alphanumeric text: '{text}'")
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
                            
                            # Check if using multi-sample registration
                            registered_embeddings = voice_data.get('embeddings')  # List of embeddings
                            registered_embedding_single = voice_data.get('embedding')  # Single embedding (legacy)
                            
                            if registered_embeddings is not None and len(registered_embeddings) > 1:
                                # Multi-sample comparison
                                print(f"[DIARIZATION] Comparing with {len(registered_embeddings)} registered samples")
                                
                                result = self.compare_with_multiple_embeddings(segment_embedding, registered_embeddings)
                                
                                max_sim = result['max_similarity']
                                avg_sim = result['avg_similarity']
                                median_sim = result['median_similarity']
                                match_count = result['match_count']
                                total_samples = result['total_samples']
                                
                                # Use median similarity for decision (more robust than average)
                                similarity = median_sim
                                voice_similarity = similarity
                                
                                # Get current segment embedding stats for comparison
                                seg_emb_mean = np.mean(segment_embedding)
                                seg_emb_std = np.std(segment_embedding)
                                seg_emb_norm = np.linalg.norm(segment_embedding)
                                
                                print(f"[DIARIZATION] ========== VOICE COMPARISON DEBUG ==========")
                                print(f"[DIARIZATION] Current segment embedding: mean={seg_emb_mean:.4f}, std={seg_emb_std:.4f}, norm={seg_emb_norm:.4f}")
                                print(f"[DIARIZATION] Comparing with {total_samples} registered embeddings:")
                                
                                # Show first registered embedding stats for comparison
                                reg_emb_0 = registered_embeddings[0]
                                reg_mean = np.mean(reg_emb_0)
                                reg_std = np.std(reg_emb_0)
                                reg_norm = np.linalg.norm(reg_emb_0)
                                print(f"[DIARIZATION] Registered sample 1: mean={reg_mean:.4f}, std={reg_std:.4f}, norm={reg_norm:.4f}")
                                
                                print(f"[DIARIZATION] Multi-sample results:")
                                print(f"[DIARIZATION]   Max: {max_sim:.4f}, Avg: {avg_sim:.4f}, Median: {median_sim:.4f}")
                                print(
                                    f"[DIARIZATION]   Matches: {match_count}/{total_samples} samples above "
                                    f"{self.thresholds.wearer_multi_vote_sim_threshold:.2f}"
                                )
                                print(f"[DIARIZATION] ===========================================")
                                
                                # Decision logic: Adjusted for real-world similarity ranges
                                # Based on logs: same speaker typically gets 0.50-0.70 similarity
                                # Different speaker typically gets 0.20-0.45 similarity
                                
                                match_ratio = match_count / total_samples
                                
                                # Get similarities array from result
                                all_sims = np.array(result['all_similarities'])
                                
                                # Multiple criteria for detection (prevents false positives)
                                
                                t = self.thresholds

                                # Criterion 1: High average (most reliable)
                                if avg_sim >= t.wearer_full_avg_threshold:
                                    is_wearer = True
                                    confidence_pct = avg_sim * 100
                                    print(f"[DIARIZATION] WEARER (AVG MATCH) - Avg: {confidence_pct:.1f}%, Median: {median_sim:.4f}")
                                
                                # Criterion 2: Good median + at least one strong match
                                elif median_sim >= t.wearer_full_median_threshold and max_sim >= t.wearer_full_median_max_threshold:
                                    is_wearer = True
                                    print(f"[DIARIZATION] WEARER (MEDIAN+MAX) - Median: {median_sim:.4f}, Max: {max_sim:.4f}")
                                
                                # Criterion 3: Enough samples vote strongly
                                elif match_ratio >= t.wearer_full_match_ratio_threshold and max_sim >= t.wearer_full_match_ratio_max_threshold:
                                    is_wearer = True
                                    print(f"[DIARIZATION] WEARER (VOTE MATCH) - {match_count}/{total_samples} samples, Max: {max_sim:.4f}")
                                
                                # Criterion 4: Majority are somewhat similar
                                elif np.sum(all_sims >= t.wearer_full_count_sim_threshold) >= (total_samples * t.wearer_full_count_ratio_threshold):
                                    is_wearer = True
                                    matches = int(np.sum(all_sims >= t.wearer_full_count_sim_threshold))
                                    print(
                                        f"[DIARIZATION] WEARER (MAJORITY) - {matches}/{total_samples} above "
                                        f"{t.wearer_full_count_sim_threshold:.2f}, Avg: {avg_sim:.4f}"
                                    )
                                
                                else:
                                    is_wearer = False
                                    print(f"[DIARIZATION] OTHER SPEAKER - Avg: {avg_sim:.4f}, Median: {median_sim:.4f}, Max: {max_sim:.4f}")
                                
                            elif registered_embedding_single is not None:
                                # Single-sample comparison (legacy/fallback)
                                print(f"[DIARIZATION] Using single-sample comparison")
                                similarity = self.compare_embeddings(segment_embedding, registered_embedding_single)
                                voice_similarity = similarity
                                
                                SIMILARITY_THRESHOLD = self.thresholds.wearer_single_threshold_full
                                
                                print(f"[DIARIZATION] Similarity: {similarity:.4f} (threshold: {SIMILARITY_THRESHOLD})")
                                
                                if similarity >= SIMILARITY_THRESHOLD:
                                    is_wearer = True
                                    confidence_pct = similarity * 100
                                    print(f"[DIARIZATION] ✓ WEARER - Confidence: {confidence_pct:.1f}%")
                                else:
                                    print(f"[DIARIZATION] → OTHER SPEAKER - {similarity:.4f} < {SIMILARITY_THRESHOLD}")
                    
                    # Add segment with voice matching info
                    segment_data = {
                        'speaker_id': speaker,
                        'transcription': text,
                        'start': start_time,
                        'end': end_time,
                        'duration': duration,
                        'confidence': 0.9  # Default confidence for optimized processing
                    }
                    
                    # Add translation info if available
                    if transcription_result:
                        segment_data['original_text'] = transcription_result.get('original_text', text)
                        segment_data['translated'] = transcription_result.get('translated', False)
                        segment_data['source_lang'] = transcription_result.get('source_lang', transcription_lang or self.transcription_lang)
                        if transcription_result.get('target_lang'):
                            segment_data['target_lang'] = transcription_result.get('target_lang')
                    
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
                
                # Only print detailed summary for full audio or when segments found
                if not is_chunk or len(segments) > 0:
                    print(f"[DIARIZATION] Processing complete: {len(segments)} valid segments found")
                    if len(segments) > 0:
                        print(f"[DIARIZATION] Filtering summary:")
                        print(f"  - Skipped short segments: {skipped_short}")
                        print(f"  - Skipped low energy: {skipped_energy}")
                        print(f"  - Skipped empty transcriptions: {skipped_empty}")
                        print(f"  - Final valid segments: {len(segments)}")
                # For empty chunks, completely silent - no logging at all
                
                processing_method = 'chunk_diarization' if is_chunk else 'optimized_diarization'
                return {
                    'segments': segments,
                    'total_duration': len(audio_array) / sample_rate,
                    'speaker_count': len(set(seg['speaker_id'] for seg in segments)),
                    'processing_method': processing_method,
                    'is_chunk': is_chunk
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


